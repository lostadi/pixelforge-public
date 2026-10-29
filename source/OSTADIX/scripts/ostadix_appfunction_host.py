#!/usr/bin/env python3
"""Private loopback adapter: Android AppFunction -> the existing MCP o_execute.

Run as the Termux user with a mode-0600 host configuration. The Android app
receives only a loopback port and bearer token in its private files directory.
There is no source interpretation, planner, JNI fallback, or execution retry.
"""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import select
import signal
import ssl
import subprocess
import threading
import time


def log(event, **fields):
    print(json.dumps(dict(event=event, **fields)), flush=True)


def send(process, frame):
    process.stdin.write(json.dumps(frame, separators=(",", ":")).encode() + b"\n")
    process.stdin.flush()


def mcp_arguments(body, max_timeout_ms):
    """Validate the Android envelope and map it to the current o_execute schema."""
    if not isinstance(body, dict):
        raise ValueError('request must be a JSON object')
    if set(body) - {'source', 'bindings', 'constraints', 'action'}:
        raise ValueError('only source, bindings, constraints and action are accepted')
    action = body.get('action', 'execute')
    if action not in ('execute', 'check'):
        raise ValueError('action must be execute or check; not dispatched')
    source = body.get('source')
    if not isinstance(source, str) or len(source.encode()) > 65536:
        raise ValueError('source must be text within the 65536-byte host limit')
    bindings = body.get('bindings', {})
    if not isinstance(bindings, dict):
        raise ValueError('bindings must be a JSON object')
    if bindings:
        raise ValueError('current o_execute does not support nonempty bindings; not dispatched')
    constraints = body.get('constraints', {})
    if not isinstance(constraints, dict):
        raise ValueError('constraints must be a JSON object')
    if set(constraints) - {'timeout_ms'}:
        raise ValueError('only constraints.timeout_ms is supported; not dispatched')
    timeout = constraints.get('timeout_ms', max_timeout_ms)
    if type(timeout) is not int or not 1 <= timeout <= max_timeout_ms:
        raise ValueError('timeout exceeds host policy')
    # MCP accepts whole seconds; the host still cancels at the exact ms budget.
    arguments = dict(source=source, timeout_secs=(timeout + 999) // 1000)
    if action == 'check':
        arguments['action'] = 'check'
    return arguments, timeout


def mcp_environment(root):
    env = dict(os.environ, O_LANG_ROOT=root, O_BACKENDS_DIR=root + '/backends',
               OSTADIX_O_CLI_BIN=root + '/target/release/o-cli',
               OSTADIX_RUNTIME_PATH_MODE='discover-local')
    try:
        context = Path('/proc/self/attr/current').read_text().strip('\0\n')
    except OSError:
        context = ''
    if context == 'u:r:ksu:s0':
        # Direct app-file execution was verified in this host context. Termux's
        # automatic linker wrapper otherwise makes O's current_exe() identify
        # linker64, so admitted --o-backend launches target the wrong image.
        # Set this before MCP initializes its preload library; changing the
        # already-running Python host's environment is too late for its cache.
        env['TERMUX_EXEC__SYSTEM_LINKER_EXEC__MODE'] = 'disable'
    return env


def stop_tree(process):
    """Emergency cleanup of the owned subprocess and observed descendants."""
    if process.poll() is not None:
        return
    pending = [process.pid]
    stopped = set()
    while pending:
        pid = pending.pop()
        if pid in stopped:
            continue
        try:
            os.kill(pid, signal.SIGSTOP)
        except ProcessLookupError:
            continue
        stopped.add(pid)
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                parent = int((entry / 'stat').read_text().rsplit(') ', 1)[1].split()[1])
                if parent == pid:
                    pending.append(int(entry.name))
            except (OSError, ValueError, IndexError):
                pass
    for pid in stopped:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=5)


class Host(ThreadingHTTPServer):
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.0'

    def log_message(self, *args):
        pass

    def respond(self, status, result):
        self.connection.settimeout(5)
        data = json.dumps(result, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        config = self.server.config
        if self.path != '/o_execute' or not hmac.compare_digest(
                self.headers.get('Authorization', ''), 'Bearer ' + config['token']):
            self.respond(403, dict(error='unauthorized'))
            return
        if not self.server.slots.acquire(blocking=False):
            self.respond(503, dict(error='host execution capacity reached; not dispatched'))
            return
        process = None
        dispatched = False
        caller_disconnected = False
        request_id = self.headers.get('X-Ostadix-Request', '')[:100]
        try:
            self.connection.settimeout(5)
            length = int(self.headers.get('Content-Length', '-1'))
            if not 0 <= length <= 2 * 1024 * 1024:
                raise ValueError('request exceeds host transport limit')
            body = json.loads(self.rfile.read(length))
            arguments, timeout = mcp_arguments(body, config['max_timeout_ms'])
            source = arguments['source']
            root = config['root']
            env = mcp_environment(root)
            process = subprocess.Popen([config['mcp_executable']], cwd=root, env=env,
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=subprocess.DEVNULL, start_new_session=True)
            frames = queue.Queue(maxsize=32)

            def read_frames():
                try:
                    while True:
                        line = process.stdout.readline(8 * 1024 * 1024 + 1)
                        if not line:
                            raise RuntimeError('MCP closed before result')
                        if len(line) > 8 * 1024 * 1024:
                            raise RuntimeError('MCP response exceeds transport limit')
                        frames.put(json.loads(line), timeout=1)
                except Exception as error:
                    try:
                        frames.put(error, timeout=1)
                    except queue.Full:
                        pass

            threading.Thread(target=read_frames, daemon=True).start()
            send(process, dict(jsonrpc='2.0', id=1, method='initialize', params=dict(
                protocolVersion='2025-03-26', capabilities={},
                clientInfo=dict(name='android-appfunction-host', version='1'))))
            initialized = frames.get(timeout=10)
            if isinstance(initialized, Exception) or 'error' in initialized:
                raise RuntimeError('MCP initialization failed; not dispatched')
            send(process, dict(jsonrpc='2.0', method='notifications/initialized'))
            deadline = time.monotonic() + timeout / 1000
            send(process, dict(jsonrpc='2.0', id=2, method='tools/call',
                               params=dict(name='o_execute', arguments=arguments)))
            dispatched = True
            self.connection.setblocking(False)
            log('mcp_dispatch', app_request=request_id, action=arguments.get('action', 'execute'),
                source_sha256=hashlib.sha256(source.encode()).hexdigest())
            cancellation_sent = False
            while True:
                try:
                    frame = frames.get(timeout=0.05)
                except queue.Empty:
                    frame = None
                if isinstance(frame, Exception):
                    raise frame
                if frame and frame.get('id') == 2:
                    if 'error' in frame:
                        raise RuntimeError('MCP returned a protocol error after dispatch; no retry')
                    result = frame['result']
                    structured = result.get('structuredContent', {})
                    if not isinstance(structured, dict):
                        structured = {}
                    log('mcp_result', app_request=request_id,
                        job_id=structured.get('job_id'), state=structured.get('state'),
                        is_error=result.get('isError', False), cancelled=cancellation_sent)
                    # A host deadline still has a caller awaiting the real result.
                    if not caller_disconnected:
                        self.respond(200, result)
                    break
                if not cancellation_sent:
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    if readable:
                        try:
                            caller_disconnected = not self.connection.recv(1)
                        except (ssl.SSLWantReadError, BlockingIOError):
                            pass
                        except (ConnectionResetError, BrokenPipeError, ssl.SSLEOFError):
                            # Android may abort TLS without a close_notify alert.
                            # Forward cancellation, then stop reading this socket.
                            caller_disconnected = True
                if not cancellation_sent and (caller_disconnected or time.monotonic() > deadline):
                    send(process, dict(jsonrpc='2.0', method='notifications/cancelled',
                                       params=dict(requestId=2, reason='Android caller cancellation or host deadline')))
                    log('mcp_cancel', app_request=request_id,
                        cause='caller_disconnect' if caller_disconnected else 'host_deadline')
                    cancellation_sent = True
                    deadline = time.monotonic() + 5
                elif cancellation_sent and time.monotonic() > deadline:
                    raise TimeoutError('MCP did not settle cancellation; no retry')
        except Exception as error:
            log('host_failure', app_request=request_id, dispatched=dispatched,
                error_class=type(error).__name__)
            if not caller_disconnected:
                try:
                    self.respond(502 if dispatched else 400, dict(
                        error=str(error), dispatched=dispatched, retry=False))
                except OSError:
                    pass
        finally:
            if process is not None:
                try:
                    process.stdin.close()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    stop_tree(process)
            self.server.slots.release()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if args.config.stat().st_mode & 0o077:
        raise SystemExit('host config must be private (mode 0600)')
    config = json.loads(args.config.read_text())
    if len(config['token']) < 64 or not 1 <= config['max_timeout_ms'] <= 900000:
        raise SystemExit('invalid host policy')
    server = Host(('127.0.0.1', config['port']), Handler)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(config['certificate'], config['private_key'])
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    server.config = config
    server.slots = threading.BoundedSemaphore(4)
    log('host_ready', uid=os.getuid(), port=config['port'], tool='o_execute')
    server.serve_forever()


if __name__ == '__main__':
    main()
