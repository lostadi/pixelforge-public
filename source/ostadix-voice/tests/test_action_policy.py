"""App-owned action policy: real temporary files/processes, no model or phone actions."""
import json
import sys
from unittest.mock import patch

from agent import Agent, AppError, Store
from test_app import Base, FakeRPC, eventually


class ActionPolicyTests(Base):
    def setUp(self):
        super().setUp()
        self.command_patch = patch('agent.codex_command', return_value=['offline-protocol-fixture'])
        self.command_patch.start()
        self.addCleanup(self.command_patch.stop)

    def decisions(self):
        return [e['data'] for e in self.store.events(self.sid) if e['kind'] == 'approval_decision']

    def test_default_auto_persists_and_executes_exact_edit_without_approval_ui(self):
        self.assertEqual(self.agent.action_policy(), 'auto')
        self.assertEqual(self.store.setting('action_policy'), 'auto')
        self.start()
        args = {'operation': 'write', 'path': 'hello.txt', 'content': 'changed by requested action'}
        thread = self.tool(args); thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual((self.root / 'hello.txt').read_text(), args['content'])
        self.assertFalse(self.agent.approvals)
        self.assertFalse(any(e['kind'] == 'approval' for e in self.store.events(self.sid)))
        self.assertEqual(self.decisions()[0]['arguments'], args)
        self.assertTrue(self.decisions()[0]['automatic'])
        self.assertTrue(self.decisions()[0]['approved'])

    def test_auto_command_has_real_result_and_exit_status(self):
        self.start()
        args = {'operation': 'command', 'argv': [sys.executable, '-c',
                'print("actual command result"); raise SystemExit(7)']}
        thread = self.tool(args); thread.join(3)
        result = self.agent.rpc.sent[-1]['result']
        self.assertFalse(result['success'])
        self.assertIn('actual command result', result['contentItems'][0]['text'])
        self.assertIn('7', result['contentItems'][0]['text'])
        self.assertEqual(self.decisions()[0]['arguments'], args)

    def test_auto_command_can_still_be_cancelled(self):
        self.start()
        thread = self.tool({'operation': 'command', 'argv': [sys.executable, '-c',
                            'import time; time.sleep(30)']})
        eventually(lambda: self.agent.process is not None)
        process = self.agent.process
        self.agent.cancel(self.sid); thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertIsNotNone(process.poll())
        self.assertEqual(self.store.session(self.sid)['status'], 'interrupted')

    def test_auto_mcp_records_effective_arguments_and_verified_revision(self):
        from ostadix_tools import OstadixTools
        from test_ostadix_tools import Client
        pair = {'root': '/fixture/repo', 'revision': 'a' * 40,
                'backends': '/fixture/backends', 'binary': '/fixture/mcp',
                'binaries': {name: '/fixture/' + name for name in
                    ('O', 'o-cli', 'octl', 'o-info', 'olangc', 'o-live-host', 'ogit',
                     'o-node', 'o-registry', 'ostadix-device')}}
        self.agent.ostadix = OstadixTools(lambda: pair, Client)
        self.start()
        thread = self.tool({'operation': 'command', 'argv': ['@ostadix', 'call',
            'o_execute', json.dumps({'source': 'fixture-only'})]})
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(self.agent.rpc.sent[-1]['result']['success'])
        self.assertEqual(self.decisions()[0]['arguments'], {
            'operation': 'ostadix', 'tool': 'o_execute',
            'arguments': {'source': 'fixture-only', 'cwd': str(self.root)},
            'repository': pair['root'], 'revision': pair['revision']})

    def test_auto_phone_still_consumes_inspected_screen_token(self):
        from phone_tools import PhoneTools
        from test_phone_tools import FakePhone
        self.agent.phone = PhoneTools(runner=FakePhone())
        self.start()
        with patch('phone_tools.os.path.isfile', return_value=True):
            screen = self.agent.execute({'operation': 'command', 'argv': ['@phone', 'screen']})
            argv = ['@phone', 'tap', screen['screen_token'], '20', '120']
            result = self.agent.execute({'operation': 'command', 'argv': argv})
            self.assertTrue(result['command_accepted'])
            self.assertEqual(self.decisions()[0]['arguments']['argv'], argv)
            self.assertTrue(self.decisions()[0]['automatic'])
            with self.assertRaises(AppError):
                self.agent.execute({'operation': 'command', 'argv': argv})

    def test_auto_phone_does_not_act_after_task_finishes_at_decision(self):
        from phone_tools import PhoneTools
        from test_phone_tools import FakePhone
        backend = FakePhone()
        self.agent.phone = PhoneTools(runner=backend)
        self.start()
        original = self.agent.wait_approval
        def finish_after_decision(args):
            original(args)
            self.agent.finish('idle', 'task already finished')
        with patch('phone_tools.os.path.isfile', return_value=True):
            screen = self.agent.execute({'operation': 'command', 'argv': ['@phone', 'screen']})
            before = list(backend.calls)
            with patch.object(self.agent, 'wait_approval', side_effect=finish_after_decision):
                with self.assertRaisesRegex(AppError, 'completed task'):
                    self.agent.execute({'operation': 'command', 'argv': [
                        '@phone', 'tap', screen['screen_token'], '20', '120']})
            self.assertEqual(backend.calls, before)

    def test_auto_decision_keeps_credential_redaction(self):
        self.start()
        self.agent.wait_approval({'operation': 'command', 'argv': ['fixture'],
                                  'password': 'secret-not-for-history'})
        self.assertEqual(self.decisions()[0]['arguments']['password'], '[redacted]')

    def test_ask_choice_survives_reopen(self):
        self.assertEqual(self.agent.set_action_policy('ask'),
                         {'mode': 'ask', 'automatic': False, 'released': 0})
        replacement = Store(self.state)
        new_agent = Agent(replacement, self.root, FakeRPC)
        try:
            self.assertEqual(new_agent.action_policy(), 'ask')
        finally:
            new_agent.close(); replacement.db.close()

    def test_invalid_choice_is_rejected_without_changing_policy(self):
        for value in ('', 'always', True, None, []):
            with self.subTest(value=value), self.assertRaises(AppError):
                self.agent.set_action_policy(value)
        self.assertEqual(self.agent.action_policy(), 'auto')

    def test_ask_blocks_until_a_real_decision_and_denial_prevents_edit(self):
        self.agent.set_action_policy('ask')
        self.start()
        thread = self.tool({'operation': 'delete', 'path': 'hello.txt'})
        eventually(lambda: bool(self.agent.approvals))
        self.assertTrue((self.root / 'hello.txt').exists())
        self.approve(False); thread.join(2)
        self.assertTrue((self.root / 'hello.txt').exists())
        self.assertFalse(self.decisions()[0]['automatic'])
        self.assertFalse(self.decisions()[0]['approved'])

    def test_enabling_auto_releases_current_pending_action_once(self):
        self.agent.set_action_policy('ask'); self.start()
        thread = self.tool({'operation': 'write', 'path': 'hello.txt', 'content': 'released'})
        eventually(lambda: bool(self.agent.approvals))
        self.assertEqual(self.agent.set_action_policy('auto')['released'], 1)
        thread.join(2)
        self.assertEqual((self.root / 'hello.txt').read_text(), 'released')
        self.assertEqual(len(self.decisions()), 1)
        self.assertTrue(self.decisions()[0]['automatic'])
        self.assertEqual(self.agent.set_action_policy('auto')['released'], 0)

    def test_enabling_auto_keeps_pending_file_change_check(self):
        self.agent.set_action_policy('ask'); self.start()
        thread = self.tool({'operation': 'delete', 'path': 'hello.txt'})
        eventually(lambda: bool(self.agent.approvals))
        (self.root / 'hello.txt').write_text('changed meanwhile')
        self.agent.set_action_policy('auto'); thread.join(2)
        self.assertEqual((self.root / 'hello.txt').read_text(), 'changed meanwhile')
        self.assertFalse(self.agent.rpc.sent[-1]['result']['success'])

    def test_cancelled_pending_action_is_never_released(self):
        self.agent.set_action_policy('ask'); self.start()
        thread = self.tool({'operation': 'delete', 'path': 'hello.txt'})
        eventually(lambda: bool(self.agent.approvals))
        self.agent.cancel(self.sid)
        self.assertEqual(self.agent.set_action_policy('auto')['released'], 0)
        thread.join(2)
        self.assertTrue((self.root / 'hello.txt').exists())
        self.assertFalse(self.decisions())

    def test_stale_generation_does_not_gain_auto_approval(self):
        self.agent.set_action_policy('ask'); self.start()
        thread = self.tool({'operation': 'delete', 'path': 'hello.txt'})
        eventually(lambda: bool(self.agent.approvals))
        with self.agent.lock:
            self.agent.task_generation += 1
            self.assertEqual(self.agent.set_action_policy('auto')['released'], 0)
        thread.join(2)
        self.assertTrue((self.root / 'hello.txt').exists())
        self.assertFalse(self.decisions())

    def test_stale_generation_cannot_be_approved_manually(self):
        self.agent.set_action_policy('ask'); self.start()
        thread = self.tool({'operation': 'delete', 'path': 'hello.txt'})
        eventually(lambda: bool(self.agent.approvals))
        with self.agent.lock:
            ident = next(iter(self.agent.approvals))
            self.agent.task_generation += 1
            with self.assertRaises(AppError):
                self.agent.approve(self.sid, ident, True)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue((self.root / 'hello.txt').exists())

    def test_auto_still_rejects_stale_tool_requests(self):
        self.start()
        rpc = self.agent.rpc
        self.agent._tool_request({'id': 90, 'params': {'threadId': 'wrong-thread',
            'tool': 'workspace', 'arguments': {'operation': 'delete', 'path': 'hello.txt'}}})
        self.assertTrue((self.root / 'hello.txt').exists())
        self.assertFalse(rpc.sent[-1]['result']['success'])
        self.assertFalse(self.decisions())

    def test_cancel_after_auto_decision_prevents_file_change(self):
        self.start()
        original = self.agent.wait_approval
        def cancel_after_decision(args):
            original(args)
            self.agent.cancelled.set()
        with patch.object(self.agent, 'wait_approval', side_effect=cancel_after_decision):
            thread = self.tool({'operation': 'delete', 'path': 'hello.txt'}); thread.join(2)
        self.assertTrue((self.root / 'hello.txt').exists())
        self.assertFalse(self.agent.rpc.sent[-1]['result']['success'])

    def test_automatic_file_action_still_checks_changed_target(self):
        self.start()
        original = self.agent.wait_approval
        def change_after_decision(args):
            original(args)
            (self.root / 'hello.txt').write_text('concurrent edit')
        with patch.object(self.agent, 'wait_approval', side_effect=change_after_decision):
            thread = self.tool({'operation': 'delete', 'path': 'hello.txt'}); thread.join(2)
        self.assertEqual((self.root / 'hello.txt').read_text(), 'concurrent edit')
        self.assertFalse(self.agent.rpc.sent[-1]['result']['success'])

    def test_auto_prompt_and_schema_do_not_demand_confirmation(self):
        self.start()
        params = next(params for method, params in self.agent.rpc.calls if method == 'thread/start')
        self.assertIn('Action policy: AUTO', params['developerInstructions'])
        self.assertIn('without another confirmation', params['dynamicTools'][0]['description'])
        self.assertNotIn('require explicit local approval', params['developerInstructions'])

    def test_switching_policy_refreshes_instructions_on_existing_connection(self):
        self.agent.IDLE_RELEASE_SECONDS = 60
        self.start()
        rpc = self.agent.rpc
        self.agent.finish('idle', 'done')
        self.agent.set_action_policy('ask')
        self.agent.submit(self.sid, 'Continue', 'request-next-12345')
        eventually(lambda: self.store.session(self.sid)['turn_id'])
        self.assertIs(self.agent.rpc, rpc)
        params = next(params for method, params in rpc.calls if method == 'thread/resume')
        self.assertIn('Action policy: ASK', params['developerInstructions'])

    def test_native_requests_cannot_bypass_dispatcher_in_auto_mode(self):
        self.start()
        self.agent.on_event({'method': 'item/commandExecution/requestApproval', 'id': 99,
                             'params': {'command': 'unrouted-command'}})
        self.assertEqual(self.agent.rpc.sent[-1]['result']['decision'], 'decline')
        self.assertFalse(self.decisions())
