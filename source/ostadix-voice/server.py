#!/usr/bin/env python3
"""Authenticated loopback UI for the local Codex agent."""
import argparse
import fcntl
import hmac
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import secrets
import signal
import subprocess
import threading
from urllib.parse import urlsplit

from agent import Agent, AppError, Store, Workspace, clean

DEFAULT_STATE = Path.home() / '.local/state/ostadix-voice'
STATIC = Path(__file__).parent / 'static'


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port, root, state, rpc_factory=None):
        self.root = Path(root).resolve(strict=True)
        if not self.root.is_dir():
            raise AppError('Select an existing workspace directory')
        state = Path(state).resolve()
        if state.is_relative_to(self.root):
            raise AppError('State must be outside the selected workspace so tools cannot read login credentials')
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_file = open(state / 'server.lock', 'a')
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_file.close()
            raise AppError('A session already uses this state directory. Run c --browser with the same --state-dir to restart the browser interface.')
        self.store = Store(state)
        self.store.setting('launch_workspace', str(self.root))
        self.cookie_token = self.store.setting('browser_cookie') or self.store.setting('browser_cookie', secrets.token_urlsafe(32))
        self.pair_token = secrets.token_urlsafe(32)
        from local_chat import create_agent
        self.agent = create_agent(self.store, self.root, rpc_factory=rpc_factory)
        sessions = self.sessions()
        selected = self.store.setting('selected')
        if not any(s['id'] == selected for s in sessions):
            selected = sessions[0]['id'] if sessions else self.store.create(self.root)
            self.store.setting('selected', selected)
        super().__init__(('127.0.0.1', port), Handler)
        self.origin = 'http://127.0.0.1:' + str(self.server_port)
        (state / 'browser.json').write_text(json.dumps({'url': self.origin, 'pair': self.pair_token, 'workspace': str(self.root)}))
        os.chmod(state / 'browser.json', 0o600)

    def sessions(self):
        with self.store.lock:
            return [dict(r) for r in self.store.db.execute('SELECT * FROM sessions ORDER BY created DESC')
                    if Path(r['workspace']).resolve().is_relative_to(self.root)]

    def server_close(self):
        self.agent.close()
        super().server_close()
        self.lock_file.close()


class Handler(BaseHTTPRequestHandler):
    server_version = 'OstadixVoice/0.1'

    def setup(self):
        super().setup()
        self.connection.settimeout(15)

    def log_message(self, *_):
        pass  # Request URLs, prompts, pairing capabilities and command logs never go to stderr.

    def send(self, status, body, content_type='application/json', cookie=None):
        if not isinstance(body, bytes):
            body = json.dumps(clean(body)).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('X-Frame-Options', 'DENY')
        self.send_header('Permissions-Policy', 'microphone=(self), camera=(), geolocation=()')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        if cookie:
            self.send_header('Set-Cookie', cookie)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def check(self, mutation=False, authenticated=True):
        if self.headers.get('Host') != urlsplit(self.server.origin).netloc:
            raise AppError('Invalid Host header')
        origin = self.headers.get('Origin')
        if origin and origin != self.server.origin:
            raise AppError('Cross-origin requests are not allowed')
        site = self.headers.get('Sec-Fetch-Site')
        # External links and Android browser intents may be cross-site. Allow
        # only the public entry document in a top-level GET navigation; this
        # does not pair the browser or expose any session/tool API.
        entry_navigation = (
            self.command == 'GET' and urlsplit(self.path).path == '/'
            and self.headers.get('Sec-Fetch-Mode') == 'navigate'
            and self.headers.get('Sec-Fetch-Dest') == 'document'
            and site in ('same-site', 'cross-site')
        )
        if site not in (None, 'none', 'same-origin') and not entry_navigation:
            raise AppError('Cross-site requests are not allowed')
        if mutation and (origin != self.server.origin or self.headers.get('Content-Type', '').split(';')[0] != 'application/json' or self.headers.get('X-Ostadix-Voice') != '1'):
            raise AppError('Same-origin JSON requests are required')
        if authenticated:
            jar = cookies.SimpleCookie()
            try:
                jar.load(self.headers.get('Cookie', ''))
                token = jar['ostadix_voice'].value
            except (KeyError, cookies.CookieError):
                token = ''
            if not hmac.compare_digest(token, self.server.cookie_token):
                self.send(401, {'error': 'Browser is not paired. Run the open command in Termux.'})
                return False
        return True

    def do_GET(self):
        try:
            path = urlsplit(self.path).path
            if not self.check(authenticated=path.startswith('/api/')):
                return
            if path == '/api/state':
                from urllib.parse import parse_qs
                query = parse_qs(urlsplit(self.path).query)
                sid = query.get('session', [self.server.store.setting('selected')])[0]
                self.server.agent.workspace(sid)
                after = int(query.get('after', ['0'])[0])
                if after < 0:
                    raise AppError('Invalid event cursor')
                with self.server.agent.lock:
                    approvals = [{k: v for k, v in a.items() if k not in ('event', 'decision')} for a in self.server.agent.approvals.values() if a['session'] == sid and not a['event'].is_set()]
                self.send(200, {'session': self.server.store.session(sid), 'sessions': self.server.sessions(),
                                'events': self.server.store.events(sid, after), 'approvals': approvals,
                                'root': str(self.server.root), 'state_directory': str(self.server.store.directory),
                                'action_policy': self.server.agent.action_policy(),
                                'agent_active': self.server.agent.active})
            elif path in ('/', '/app.js', '/style.css'):
                file = STATIC / ('index.html' if path == '/' else path[1:])
                self.send(200, file.read_bytes(), (mimetypes.guess_type(str(file))[0] or 'text/plain') + '; charset=utf-8')
            else:
                self.send(404, {'error': 'Not found'})
        except (AppError, ValueError, OSError) as error:
            self.send(403, {'error': str(error)})

    def do_POST(self):
        try:
            path = urlsplit(self.path).path
            if not self.check(mutation=True, authenticated=path != '/api/pair'):
                return
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 1_100_000:
                raise AppError('Invalid request size')
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise AppError('JSON object required')
            if path == '/api/pair':
                token = body.get('token', '')
                if not isinstance(token, str) or not hmac.compare_digest(token, self.server.pair_token):
                    raise AppError('Invalid pairing capability. Reopen from Termux.')
                self.send(200, {'paired': True}, cookie='ostadix_voice='+self.server.cookie_token+'; HttpOnly; SameSite=Strict; Path=/; Max-Age=2592000')
                return
            sid = body.get('session') or self.server.store.setting('selected')
            self.server.agent.workspace(sid)
            if path == '/api/task':
                self.send(202, self.server.agent.submit(sid, body.get('text'), body.get('request_id')))
            elif path == '/api/action-policy':
                self.send(200, self.server.agent.set_action_policy(body.get('mode')))
            elif path == '/api/cancel':
                self.server.agent.cancel(sid)
                self.send(200, {'requested': True})
            elif path == '/api/approve':
                self.server.agent.approve(sid, body.get('id'), body.get('allow'))
                self.send(200, {'recorded': True})
            elif path == '/api/select':
                if self.server.agent.active:
                    raise AppError('Wait for or cancel the active task before switching conversations')
                self.server.store.setting('selected', sid)
                self.send(200, {'session': sid})
            elif path == '/api/new':
                if self.server.agent.active:
                    raise AppError('Wait for or cancel the active task first')
                relative = body.get('path', '.')
                ws = Workspace(self.server.root)
                ws.list(relative)  # Reject traversal, protected paths, and symlink components.
                new_sid = self.server.store.create(self.server.root / relative)
                self.server.store.setting('selected', new_sid)
                self.send(200, {'session': new_sid})
            elif path == '/api/inspect':
                if self.server.agent.active:
                    raise AppError('Wait for or cancel the active task first')
                result = self.server.agent.workspace(sid).read(body.get('path', ''), body.get('offset', 0))
                self.server.store.event(sid, 'notice', {'text': 'Local file check (no model call). This result is UI evidence, not part of the Codex thread.'})
                self.server.store.event(sid, 'tool_result', {'success': True, 'result': result})
                self.send(200, result)
            else:
                self.send(404, {'error': 'Not found'})
        except (AppError, ValueError, OSError, TypeError) as error:
            self.send(400, {'error': str(error)})


def open_browser(state):
    import shutil
    info = json.loads((Path(state) / 'browser.json').read_text())
    url = info['url'] + '/#' + info['pair']
    opener = shutil.which('termux-open-url') or shutil.which('xdg-open')
    if not opener:
        raise AppError('No browser opener found. Install termux-tools, then rerun open.')
    subprocess.run([opener, url], check=True, timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    if os.geteuid() == 0:
        raise AppError('Do not launch this app as root. Open a normal Termux session (leave su), then run the launch command. Root shell access is not enabled.')
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', choices=['serve', 'open'], default='serve')
    parser.add_argument('--workspace', type=Path, help='Explicit root to expose (required for serve)')
    parser.add_argument('--state-dir', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--open', action='store_true', help='Pair and open the phone browser on launch')
    args = parser.parse_args()
    if args.action == 'open':
        open_browser(args.state_dir)
        return
    if not args.workspace:
        parser.error('serve requires --workspace; no directory is granted implicitly')
    server = Server(args.port, args.workspace, args.state_dir)
    print('Ostadix Voice: ' + server.origin, flush=True)
    print('Workspace: ' + str(server.root), flush=True)
    print('State: ' + str(server.store.directory), flush=True)
    print('Assistant: ' + getattr(server.agent, 'engine_label', 'Codex') + '. Ctrl-C stops the server.', flush=True)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown, daemon=True).start())
    from launch import release_launch_lock
    release_launch_lock(args.state_dir)
    if args.open:
        threading.Timer(.5, lambda: open_browser(args.state_dir)).start()
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    try:
        main()
    except (AppError, OSError, subprocess.SubprocessError) as error:
        raise SystemExit(str(clean(str(error))))
