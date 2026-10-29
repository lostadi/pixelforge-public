import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Agent, AppError, Store, Workspace, clean
from server import Server


def eventually(check, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check(): return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for condition')


class FakeRPC:
    """Deterministic protocol test double; it never generates a model answer."""
    closed = False
    instances = []
    def __init__(self, command, cwd, callback):
        self.callback = callback; self.calls = []; self.sent = []; self.instances.append(self)
    def call(self, method, params, **kwargs):
        self.calls.append((method, params))
        if method == 'account/read': return {'account': {'type':'chatgpt'}}
        if method in ('thread/start', 'thread/resume'): return {'thread': {'id': params.get('threadId', 'thread-offline')}}
        if method == 'turn/start': return {'turn': {'id':'turn-offline'}}
        if method == 'turn/interrupt':
            self.callback({'method':'turn/completed', 'params':{'threadId':'thread-offline','turn':{'status':'interrupted'}}})
        return {}
    def send(self, message): self.sent.append(message)
    def close(self): self.closed = True


class Base(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch('agent.codex_command', return_value=['fixture-codex']))
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.root = self.directory/'workspace'; self.root.mkdir()
        self.state = self.directory/'state'
        self.store = Store(self.state)
        self.sid = self.store.create(self.root)
        self.agent = Agent(self.store, self.root, FakeRPC)
        (self.root/'hello.txt').write_text('The actual answer is 42.\n')
    def tearDown(self):
        self.agent.close(); self.store.db.close(); self.temp.cleanup()
    def start(self):
        self.agent.submit(self.sid, 'Inspect hello.txt', 'request-123456')
        eventually(lambda: self.store.session(self.sid)['turn_id'])
    def tool(self, args):
        thread = threading.Thread(target=self.agent._tool_request, args=({'id':99, 'method':'item/tool/call', 'params':{
            'threadId':'thread-offline', 'turnId':'turn-offline', 'tool':'workspace', 'arguments':args}},))
        thread.start(); return thread
    def approve(self, allow):
        eventually(lambda: bool(self.agent.approvals))
        ident = next(iter(self.agent.approvals))
        self.agent.approve(self.sid, ident, allow)


class WorkspaceTests(Base):
    def test_real_read_and_paging(self):
        ws = Workspace(self.root)
        first = ws.read('hello.txt', 0, 10)
        self.assertEqual(first['content'], 'The actual')
        self.assertTrue(first['more'])
        self.assertEqual(ws.read('hello.txt')['content'], 'The actual answer is 42.\n')
    def test_agent_read_limit_and_continuation_use_actual_byte_offsets(self):
        (self.root/'unicode.txt').write_text('café\nSECOND')
        self.agent.sid = self.sid
        first = self.agent.execute({'operation':'read','path':'unicode.txt','limit':6})
        self.assertEqual(first['content'], 'café\n')
        self.assertEqual(first['next_offset'], 6)
        self.assertEqual(first['size'], 12)
        self.assertTrue(first['more'])
        second = self.agent.execute({'operation':'read','path':'unicode.txt',
                                     'offset':first['next_offset'],'limit':6})
        self.assertEqual(second['content'], 'SECOND')
        self.assertEqual(second['next_offset'], 12)
        self.assertFalse(second['more'])
    def test_agent_read_retains_default_and_rejects_invalid_explicit_limits(self):
        (self.root/'large.txt').write_text('a'*24001)
        self.agent.sid = self.sid
        default = self.agent.execute({'operation':'read','path':'large.txt'})
        self.assertEqual(len(default['content']), 24000)
        self.assertEqual(default['next_offset'], 24000)
        self.assertTrue(default['more'])
        for limit in (0, True, 100001, 1.5):
            with self.subTest(limit=limit), self.assertRaises(AppError):
                self.agent.execute({'operation':'read','path':'large.txt','limit':limit})
    def test_traversal_symlink_and_secrets_denied(self):
        outside = self.directory/'outside'; outside.write_text('private')
        (self.root/'link').symlink_to(outside)
        (self.root/'escape').symlink_to(self.directory, target_is_directory=True)
        for path in ('../outside', str(outside), 'link', 'escape/outside', '.env', '.codex/auth.json'):
            with self.subTest(path=path), self.assertRaises((AppError, OSError)):
                Workspace(self.root).read(path)

    def test_hardlink_read_denied(self):
        import ctypes
        import errno
        outside = self.directory/'outside'; outside.write_text('private')
        libc = ctypes.CDLL(None, use_errno=True)
        result = libc.link(os.fsencode(outside), os.fsencode(self.root/'hardlink'))
        if result != 0 and ctypes.get_errno() in (errno.EPERM, errno.EACCES, errno.EOPNOTSUPP):
            self.skipTest('The host blocks hardlink creation in this security context')
        self.assertEqual(result, 0)
        with self.assertRaises(AppError):
            Workspace(self.root).read('hardlink')
    def test_atomic_edit_requires_unchanged_file(self):
        ws = Workspace(self.root); before = ws.fingerprint('hello.txt')
        ws.write('hello.txt', 'updated', before)
        self.assertEqual(ws.read('hello.txt')['content'], 'updated')
        with self.assertRaises(AppError): ws.write('hello.txt', 'stale', before)
    def test_fifo_is_not_read(self):
        os.mkfifo(self.root/'pipe')
        with self.assertRaises(AppError): Workspace(self.root).read('pipe')


class AgentTests(Base):
    def setUp(self):
        super().setUp()
        # This suite exercises explicit confirmation; automatic mode has its own suite.
        self.agent.set_action_policy('ask')

    def test_tool_result_is_actual_content(self):
        self.start(); thread = self.tool({'operation':'read','path':'hello.txt'}); thread.join(3)
        payload = self.agent.rpc.sent[-1]['result']
        self.assertIn('The actual answer is 42.', payload['contentItems'][0]['text'])
        self.assertFalse(any(e['kind']=='finished' for e in self.store.events(self.sid)))
    def test_destructive_action_does_not_execute_before_approval(self):
        self.start(); thread = self.tool({'operation':'delete','path':'hello.txt'})
        eventually(lambda: bool(self.agent.approvals))
        self.assertTrue((self.root/'hello.txt').exists())
        self.approve(False); thread.join(3)
        self.assertTrue((self.root/'hello.txt').exists())
        self.assertFalse(self.agent.rpc.sent[-1]['result']['success'])
    def test_exact_approved_edit_and_delete(self):
        self.start(); thread = self.tool({'operation':'write','path':'hello.txt','content':'new contents'})
        self.approve(True); thread.join(3)
        self.assertEqual((self.root/'hello.txt').read_text(), 'new contents')
        thread = self.tool({'operation':'delete','path':'hello.txt'})
        self.approve(True); thread.join(3)
        self.assertFalse((self.root/'hello.txt').exists())
    def test_changed_file_invalidates_approval(self):
        self.start(); thread = self.tool({'operation':'delete','path':'hello.txt'})
        eventually(lambda: bool(self.agent.approvals))
        (self.root/'hello.txt').write_text('changed after request')
        self.approve(True); thread.join(3)
        self.assertTrue((self.root/'hello.txt').exists())
        self.assertFalse(self.agent.rpc.sent[-1]['result']['success'])
    def test_cancel_while_awaiting_approval(self):
        self.start(); thread = self.tool({'operation':'delete','path':'hello.txt'})
        eventually(lambda: bool(self.agent.approvals))
        self.agent.cancel(self.sid); thread.join(3)
        self.assertTrue((self.root/'hello.txt').exists())
        self.assertEqual(self.store.session(self.sid)['status'], 'interrupted')
    def test_cancel_real_process(self):
        self.start(); thread = self.tool({'operation':'command','argv':[sys.executable,'-c','import time; print("started",flush=True); time.sleep(30)']})
        self.approve(True)
        eventually(lambda: self.agent.process is not None)
        proc = self.agent.process
        self.agent.cancel(self.sid); thread.join(4)
        self.assertIsNotNone(proc.poll())
        self.assertFalse(thread.is_alive())
        self.assertEqual(self.store.session(self.sid)['status'], 'interrupted')
    def test_command_failure_is_not_success(self):
        self.start(); thread = self.tool({'operation':'command','argv':[sys.executable,'-c','print("actual failure"); raise SystemExit(7)']})
        self.approve(True); thread.join(4)
        result = self.agent.rpc.sent[-1]['result']
        self.assertFalse(result['success'])
        self.assertIn('actual failure', result['contentItems'][0]['text'])
        self.assertIn('7', result['contentItems'][0]['text'])
    def test_network_and_auth_error_preserve_history(self):
        for message in ('HTTP 401: authentication expired', 'Network connection failed'):
            if not self.agent.active:
                self.agent.submit(self.sid, 'Keep this request', 'request-'+str(time.time_ns()))
                eventually(lambda: self.store.session(self.sid)['turn_id'])
            self.agent.on_event({'method':'turn/completed','params':{'threadId':'thread-offline','turn':{'status':'failed','error':{'message':message}}}})
            self.assertEqual(self.store.session(self.sid)['status'], 'failed')
            self.assertEqual(self.store.events(self.sid)[-1]['data']['text'], message)
        self.assertEqual(len([e for e in self.store.events(self.sid) if e['kind']=='user']), 2)
    def test_quit_during_task_cannot_be_recorded_as_a_disconnect(self):
        self.start()
        rpc = self.agent.rpc
        def disconnect_on_close():
            rpc.closed = True
            rpc.callback({'method':'local/disconnected','params':{}})
        rpc.close = disconnect_on_close
        self.agent.close()
        self.assertEqual(self.store.session(self.sid)['status'], 'interrupted')
        self.assertEqual(self.store.session(self.sid)['thread_id'], 'thread-offline')
        outcomes = [e['data'] for e in self.store.events(self.sid) if e['kind']=='finished']
        self.assertEqual(len(outcomes), 1)
        self.assertIn('Local server stopped', outcomes[0]['text'])
    def test_quit_after_completion_preserves_success(self):
        self.start()
        self.agent.finish('idle', 'Completed answer')
        self.agent.close()
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')
        outcomes = [e['data'] for e in self.store.events(self.sid) if e['kind']=='finished']
        self.assertEqual(outcomes, [{'status':'idle', 'text':'Completed answer'}])
    def test_same_request_id_never_repeats_action(self):
        self.start()
        self.assertTrue(self.agent.submit(self.sid,'Inspect hello.txt','request-123456')['duplicate'])
        self.assertEqual(len([e for e in self.store.events(self.sid) if e['kind']=='user']), 1)
    def test_native_escape_request_is_denied(self):
        self.start()
        self.agent.on_event({'method':'item/commandExecution/requestApproval','id':77,'params':{'command':'rm -rf something'}})
        self.assertEqual(self.agent.rpc.sent[-1]['result']['decision'], 'decline')
    def test_redaction(self):
        text = 'OPENAI_API_KEY=sk-abcdefghijklmnopqrstuv\nAuthorization: Bearer private-value'
        self.assertNotIn('abcdefghijkl', clean(text))
        self.assertNotIn('private-value', clean(text))

    def test_redaction_preserves_json_fields_format_and_is_idempotent(self):
        text = ' { "password": "fixture-only", "answer": "must remain", "Authorization": "Bearer fixture-token", "nested": {"API_KEY": "fixture-two", "value": 7} } '
        redacted = clean(text)
        self.assertEqual(json.loads(redacted), {'password': '[redacted]', 'answer': 'must remain',
                         'Authorization': 'Bearer [redacted]', 'nested': {'API_KEY': '[redacted]', 'value': 7}})
        self.assertTrue(redacted.startswith(' { "password": '))
        self.assertEqual(clean(redacted), redacted)
        encoded = json.dumps(clean({'password': 'fixture-only', 'answer': 'must remain'}), separators=(',', ':'))
        self.assertEqual(clean({'content': encoded})['content'], encoded)

    def test_redaction_keeps_split_clean_markers_and_hides_unclosed_secret_values(self):
        for marker in ('"[redacted]"', "'[redacted]'", '[redacted]'):
            for end in range(len(marker) + 1):
                text = 'API_KEY=' + marker[:end]
                self.assertEqual(clean(text), text)
        for end in range(len('[redacted]') + 1):
            text = 'Bearer ' + '[redacted]'[:end]
            self.assertEqual(clean(text), text)
        for text in ('API_KEY="fixture-private', "PASSWORD='fixture-private", '{"API_KEY":{"value":"fixture-private"'):
            self.assertNotIn('fixture-private', clean(text))


class PersistenceTests(Base):
    def test_restart_keeps_workspace_thread_and_history(self):
        self.start()
        self.store.event(self.sid,'assistant',{'text':'Saved answer'})
        replacement = Store(self.state)
        try:
            recovered = replacement.session(self.sid)
            self.assertEqual(recovered['workspace'], str(self.root))
            self.assertEqual(recovered['thread_id'], 'thread-offline')
            self.assertEqual(recovered['status'], 'interrupted')
            self.assertTrue(any(e['data'].get('text')=='Saved answer' for e in replacement.events(self.sid)))
            self.assertTrue((self.root/'hello.txt').exists())
        finally: replacement.db.close()


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.directory = Path(self.temp.name)
        self.root = self.directory/'workspace'; self.root.mkdir()
        (self.root/'hello.txt').write_text('actual HTTP file content')
        self.server = Server(0,self.root,self.directory/'state',FakeRPC)
        self.thread = threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.cookie = ''
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join();self.server.store.db.close();self.temp.cleanup()
    def request(self,path,body=None,headers=None):
        conn = http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=3)
        base={'Cookie':self.cookie}
        if body is not None: base.update({'Origin':self.server.origin,'X-Ostadix-Voice':'1','Content-Type':'application/json'})
        base.update(headers or {})
        conn.request('GET' if body is None else 'POST',path,json.dumps(body) if body is not None else None,base)
        response=conn.getresponse();data=response.read();result=(response.status,data,response.getheaders());conn.close();return result
    def pair(self):
        status,_,headers=self.request('/api/pair',{'token':self.server.pair_token},
                                     {'Sec-Fetch-Site':'same-origin', 'Sec-Fetch-Mode':'cors', 'Sec-Fetch-Dest':'empty'})
        self.assertEqual(status,200)
        self.cookie=dict(headers)['Set-Cookie'].split(';')[0]
    def test_requires_auth_and_rejects_cross_origin_and_dns_rebinding(self):
        self.assertEqual(self.request('/api/state')[0],401)
        self.pair()
        self.assertEqual(self.request('/api/state')[0],200)
        self.assertEqual(self.request('/api/state',headers={'Host':'evil.example'})[0],403)
        self.assertEqual(self.request('/api/task',{'text':'x'}, {'Origin':'https://evil.example'})[0],400)
        self.assertEqual(self.request('/api/task',{'text':'x'}, {'Sec-Fetch-Site':'cross-site'})[0],400)
    def test_external_browser_navigation_then_pairing(self):
        # Android intents/external links can mark a top-level page cross-site.
        for site in ('cross-site', 'same-site'):
            with self.subTest(site=site):
                navigation = {'Sec-Fetch-Site':site, 'Sec-Fetch-Mode':'navigate', 'Sec-Fetch-Dest':'document'}
                status, body, headers = self.request('/', headers=navigation)
                self.assertEqual(status, 200)
                self.assertIn(b'Ostadix Voice', body)
                self.assertNotIn('Set-Cookie', dict(headers))
                self.assertNotIn(self.server.pair_token.encode(), body)
                self.assertEqual(self.request('/api/state')[0], 401)
        self.pair()
        same_origin = {'Sec-Fetch-Site':'same-origin', 'Sec-Fetch-Mode':'cors', 'Sec-Fetch-Dest':'empty'}
        for path in ('/app.js', '/style.css', '/api/state'):
            self.assertEqual(self.request(path, headers=same_origin)[0], 200)
        status, body, _ = self.request('/api/inspect', {'path':'hello.txt'}, same_origin)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['content'], 'actual HTTP file content')
    def test_navigation_exception_does_not_grant_cross_site_api_access(self):
        navigation = {'Sec-Fetch-Site':'cross-site', 'Sec-Fetch-Mode':'navigate', 'Sec-Fetch-Dest':'document'}
        self.assertEqual(self.request('/api/state', headers=navigation)[0], 403)
        self.assertEqual(self.request('/api/pair', {'token':self.server.pair_token}, navigation)[0], 400)
        self.pair()
        self.assertEqual(self.request('/api/state', headers=navigation)[0], 403)
        for path, body in (('/api/task', {'text':'x'}), ('/api/inspect', {'path':'hello.txt'})):
            self.assertEqual(self.request(path, body, navigation)[0], 400)
        self.assertFalse(self.server.agent.active)
    def test_navigation_exception_rejects_embedding_fetches_and_wrong_host(self):
        navigation = {'Sec-Fetch-Site':'cross-site', 'Sec-Fetch-Mode':'navigate', 'Sec-Fetch-Dest':'document'}
        for change in ({'Sec-Fetch-Dest':'iframe'}, {'Sec-Fetch-Dest':'object'},
                       {'Sec-Fetch-Mode':'cors'}, {'Sec-Fetch-Mode':'no-cors'},
                       {'Host':'evil.example'}, {'Origin':'https://evil.example'},
                       {'Sec-Fetch-Site':'invalid'}):
            with self.subTest(change=change):
                self.assertEqual(self.request('/', headers=navigation | change)[0], 403)
        self.assertEqual(self.request('/', headers={'Sec-Fetch-Site':'cross-site'})[0], 403)
        self.assertEqual(self.request('/app.js', headers=navigation)[0], 403)
    def test_ui_and_actual_file_result_survive_browser_reopen(self):
        self.pair()
        self.assertIn(b'Ostadix Voice',self.request('/')[1])
        status,data,_=self.request('/api/inspect',{'path':'hello.txt'})
        self.assertEqual(status,200); self.assertEqual(json.loads(data)['content'],'actual HTTP file content')
        first=json.loads(self.request('/api/state')[1]); second=json.loads(self.request('/api/state')[1])
        self.assertEqual(first['session']['id'],second['session']['id'])
        self.assertEqual(first['events'],second['events'])
        self.assertEqual(self.request('/api/new',{'path':'..'})[0],400)
    def test_second_server_cannot_change_active_state(self):
        sid=self.server.store.setting('selected');self.server.store.update(sid,status='executing')
        with self.assertRaises(AppError): Server(0,self.root,self.directory/'state',FakeRPC)
        self.assertEqual(self.server.store.session(sid)['status'],'executing')
    def test_state_cannot_live_in_workspace(self):
        with self.assertRaises(AppError): Server(0,self.root,self.root/'state',FakeRPC)
    def test_full_server_restart_recovers_cookie_selection_and_transcript(self):
        self.pair();self.request('/api/inspect',{'path':'hello.txt'})
        before=json.loads(self.request('/api/state')[1])
        self.server.shutdown();self.server.server_close();self.thread.join();self.server.store.db.close()
        self.server=Server(0,self.root,self.directory/'state',FakeRPC)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        status,data,_=self.request('/api/state');after=json.loads(data)
        self.assertEqual(status,200)
        self.assertEqual(before['session']['id'],after['session']['id'])
        self.assertEqual(before['events'],after['events'])


if __name__=='__main__': unittest.main(verbosity=2)
