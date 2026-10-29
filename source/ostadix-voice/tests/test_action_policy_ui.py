"""Approval setting through authenticated HTTP and terminal controls; no cloud calls."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from agent import AppError
from server import Server
from terminal import TerminalSession, TerminalUI
import test_app
from test_terminal import Audio


class HTTPActionPolicyTests(unittest.TestCase):
    request = test_app.HTTPTests.request
    pair = test_app.HTTPTests.pair
    tearDown = test_app.HTTPTests.tearDown

    def setUp(self):
        command = patch('agent.codex_command', return_value=['offline-protocol-fixture'])
        command.start(); self.addCleanup(command.stop)
        test_app.HTTPTests.setUp(self)

    def state(self):
        status, body, _ = self.request('/api/state')
        self.assertEqual(status, 200)
        return json.loads(body)

    def test_default_auto_is_visible_and_toggle_survives_restart(self):
        self.pair()
        self.assertEqual(self.state()['action_policy'], 'auto')
        status, body, _ = self.request('/api/action-policy', {'mode': 'ask'})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body), {'mode': 'ask', 'automatic': False, 'released': 0})
        self.assertEqual(self.state()['action_policy'], 'ask')
        self.server.shutdown(); self.server.server_close(); self.thread.join(); self.server.store.db.close()
        self.server = Server(0, self.root, self.directory / 'state', test_app.FakeRPC)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.assertEqual(self.state()['action_policy'], 'ask')
        self.assertEqual(self.request('/api/action-policy', {'mode': 'auto'})[0], 200)
        self.assertEqual(self.state()['action_policy'], 'auto')

    def test_unpaired_and_cross_origin_requests_cannot_change_policy(self):
        self.assertEqual(self.request('/api/action-policy', {'mode': 'ask'})[0], 401)
        self.assertEqual(self.server.agent.action_policy(), 'auto')
        self.pair()
        for headers in ({'Origin': 'https://unrelated.invalid'},
                        {'Sec-Fetch-Site': 'cross-site'},
                        {'X-Ostadix-Voice': ''}, {'Content-Type': 'text/plain'},
                        {'Host': 'unrelated.invalid'}):
            with self.subTest(headers=headers):
                self.assertEqual(self.request('/api/action-policy', {'mode': 'ask'}, headers)[0], 400)
                self.assertEqual(self.server.agent.action_policy(), 'auto')

    def test_invalid_modes_and_wrong_session_leave_current_policy_unchanged(self):
        self.pair()
        for value in ('on', 'always', '', None, True, [], {}):
            with self.subTest(mode=value):
                self.assertEqual(self.request('/api/action-policy', {'mode': value})[0], 400)
                self.assertEqual(self.server.agent.action_policy(), 'auto')
        self.assertEqual(self.request('/api/action-policy', {'mode': 'ask', 'session': 'missing'})[0], 400)
        self.assertEqual(self.server.agent.action_policy(), 'auto')

    def test_enabling_auto_over_http_releases_displayed_action_once(self):
        self.pair()
        self.request('/api/action-policy', {'mode': 'ask'})
        sid = self.state()['session']['id']
        self.request('/api/task', {'text': 'Update hello.txt', 'request_id': 'http-task-12345'})
        test_app.eventually(lambda: self.server.store.session(sid)['turn_id'])
        args = {'operation': 'write', 'path': 'hello.txt', 'content': 'HTTP approved content'}
        worker = threading.Thread(target=self.server.agent._tool_request, args=({'id': 100,
            'params': {'threadId': 'thread-offline', 'turnId': 'turn-offline',
                       'tool': 'workspace', 'arguments': args}},))
        worker.start(); self.addCleanup(worker.join, 2)
        test_app.eventually(lambda: bool(self.server.agent.approvals))
        before = self.state()
        self.assertEqual(before['action_policy'], 'ask')
        self.assertEqual(before['approvals'][0]['arguments'], args)
        self.assertEqual((self.root / 'hello.txt').read_text(), 'actual HTTP file content')
        status, body, _ = self.request('/api/action-policy', {'mode': 'auto'})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)['released'], 1)
        worker.join(2)
        self.assertFalse(worker.is_alive())
        after = self.state()
        self.assertEqual(after['action_policy'], 'auto')
        self.assertEqual(after['approvals'], [])
        self.assertEqual((self.root / 'hello.txt').read_text(), 'HTTP approved content')
        decisions = [event['data'] for event in after['events'] if event['kind'] == 'approval_decision']
        self.assertEqual(len(decisions), 1)
        self.assertTrue(decisions[0]['automatic'])
        self.assertEqual(decisions[0]['arguments'], args)


class TerminalActionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'workspace'; self.root.mkdir()
        self.state = self.base / 'state'
        command = patch('agent.codex_command', return_value=['offline-protocol-fixture'])
        command.start(); self.addCleanup(command.stop)
        self.session = TerminalSession(self.root, self.state, test_app.FakeRPC)
        self.output = []
        self.ui = TerminalUI(self.session, Audio(), self.output.append)
        self.ui.companion = Mock()
        self.ui.companion.messages.return_value = {'messages': [], 'eligible': False}

    def tearDown(self):
        self.session.close(); self.temp.cleanup()

    def test_status_default_and_off_on_controls_use_unambiguous_persisted_modes(self):
        self.ui.handle('/approvals status')
        self.assertIn('Approve all: on.', self.output[-1])
        self.assertEqual(self.session.agent.action_policy(), 'auto')
        self.ui.handle('/approvals off')
        self.assertIn('Approve all: off.', self.output[-1])
        self.assertEqual(self.session.store.setting('action_policy'), 'ask')
        self.session.close()
        self.session = TerminalSession(self.root, self.state, test_app.FakeRPC)
        self.ui = TerminalUI(self.session, Audio(), self.output.append)
        self.ui.handle('/approvals')
        self.assertIn('Approve all: off.', self.output[-1])
        self.ui.handle('/approvals on')
        self.assertEqual(self.session.store.setting('action_policy'), 'auto')
        self.assertIn('Approve all: on.', self.output[-1])
        self.assertIsNone(self.session.agent.rpc)

    def test_invalid_control_does_not_toggle_or_contact_model(self):
        for value in ('yes', 'auto', 'off extra'):
            with self.subTest(value=value), self.assertRaises(AppError):
                self.ui.handle('/approvals ' + value)
        self.assertEqual(self.session.agent.action_policy(), 'auto')
        self.assertIsNone(self.session.agent.rpc)

    def test_approve_all_phrase_releases_pending_action_without_model_request(self):
        self.ui.handle('/approvals off')
        sid = self.session.sid
        self.session.agent.submit(sid, 'Write note.txt', 'terminal-task-12345')
        test_app.eventually(lambda: self.session.store.session(sid)['turn_id'])
        calls_before = list(self.session.agent.rpc.calls)
        args = {'operation': 'write', 'path': 'note.txt', 'content': 'actual saved note'}
        worker = threading.Thread(target=self.session.agent._tool_request, args=({'id': 101,
            'params': {'threadId': 'thread-offline', 'turnId': 'turn-offline',
                       'tool': 'workspace', 'arguments': args}},))
        worker.start(); self.addCleanup(worker.join, 2)
        test_app.eventually(lambda: bool(self.session.agent.approvals))
        self.ui.handle('Approve all!')
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual((self.root / 'note.txt').read_text(), 'actual saved note')
        self.assertEqual(self.session.agent.action_policy(), 'auto')
        self.assertEqual(self.session.agent.rpc.calls, calls_before)
