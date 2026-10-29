"""Quota handoff preserves host actions and uses the NPU's legacy JSON format."""
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Store
from conversation_protocol import encode_conversation
from local_chat import LocalChatAgent
from local_runtime import LocalError, LocalRequestNotSent
from npu_chat import NPUConversationClient
from test_local_chat import FakeClient


ROLE = 'phone conversation assistant'
QUOTA = {'state': 'failed', 'error_code': 'http_429', 'safe_quota_fallback': True,
         'error': 'Gemini daily quota reached'}


class RepairNPUClient(FakeClient):
    def submit(self, owner, prompt, max_tokens, role, response_correction=None):
        job = super().submit(owner, prompt, max_tokens, role)
        self.submissions[-1]['response_correction'] = response_correction
        return job


def npu_status(**overrides):
    return dict({'ready': True, 'service_enabled': True, 'model': {
        'backend': 'litert-npu', 'model_name': 'Fixture Gemma', 'context_per_worker': 4096,
        'max_output_tokens': 512}}, **overrides)


class NPUConversationClientTests(unittest.TestCase):
    def test_enabled_supervisor_recovery_is_not_reported_as_ready_or_disabled(self):
        client = NPUConversationClient()
        raw = npu_status(ready=False, model=None, supervised=True, error='Local model server stopped')
        with patch('npu_chat.NPUClient.status', return_value=raw):
            status = client.status()
        self.assertFalse(status['ready'])
        self.assertTrue(status['recovering'])
        self.assertEqual(status['selection_state'], 'recovering')
        self.assertIn('supervisor', status['error'])
        self.assertEqual(status['recovery_cause'], 'Local model server stopped')

    def test_disabled_service_does_not_wait_or_start_itself(self):
        client = NPUConversationClient()
        with patch('npu_chat.NPUClient.status', return_value=npu_status(service_enabled=False)), \
                patch.object(client, 'start') as start, patch.object(client, 'submit') as submit:
            self.assertEqual(client.status()['selection_state'], 'disabled')
            with self.assertRaisesRegex(LocalError, 'disabled'):
                client.wait_until_ready()
        start.assert_not_called()
        submit.assert_not_called()

    def test_new_request_can_wait_for_enabled_recovery_without_submitting_any_job(self):
        client = NPUConversationClient()
        recovering = npu_status(ready=False, recovering=True)
        healthy = npu_status()
        notice = Mock()
        with patch.object(client, 'status', side_effect=[recovering, healthy]), \
                patch.object(client, 'start') as start, patch.object(client, 'submit') as submit:
            status = client.wait_until_ready(timeout=1, on_wait=notice)
        self.assertTrue(status['ready'])
        notice.assert_called_once()
        start.assert_not_called()
        submit.assert_not_called()

    def test_recovery_wait_is_bounded_and_cancellable_before_submission(self):
        client = NPUConversationClient()
        recovering = npu_status(ready=False, recovering=True)
        with patch.object(client, 'status', return_value=recovering), \
                self.assertRaisesRegex(LocalError, 'did not finish'):
            client.wait_until_ready(timeout=0)
        cancelled = threading.Event()
        with patch.object(client, 'status', return_value=recovering), \
                self.assertRaisesRegex(LocalError, 'cancelled'):
            client.wait_until_ready(cancelled=cancelled, on_wait=lambda state: cancelled.set())

    def test_adapter_uses_text_tool_format_without_native_capability_claim(self):
        client = NPUConversationClient()
        call = {'id': 'cloud-call', 'type': 'function', 'function': {'name': 'workspace',
                'arguments': '{"operation":"read","path":"fixture.txt"}'},
                'extra_content': {'google': {'thought_signature': 'c2lnbmF0dXJl'}}}
        messages = [{'role': 'user', 'content': 'Read fixture.txt'},
                    {'role': 'assistant', 'content': None, 'tool_calls': [call]},
                    {'role': 'tool', 'tool_call_id': 'cloud-call', 'name': 'workspace', 'content': 'actual evidence'}]
        task = encode_conversation(messages)
        with patch('npu_chat.NPUClient.status', return_value=npu_status()):
            status = client.status()
        self.assertNotIn('native_conversation_protocol', status['model'])
        self.assertIn('Tensor NPU', client.engine_label)
        with patch('npu_chat.NPUClient.submit', return_value={'id': 'fixture'}) as submit:
            client.submit('owner', task, max_tokens=2048, role=ROLE)
        prompt = submit.call_args.args[1]
        self.assertIn('"tool":"workspace"', prompt)
        self.assertIn('actual evidence', prompt)
        self.assertIn('cloud-call', prompt)
        self.assertNotIn('thought_signature', prompt)
        self.assertEqual(submit.call_args.kwargs['max_tokens'], 512)
        self.assertIn('thought_signature', task)
        self.assertNotIn('"final":', prompt)
        self.assertNotIn('"evidence":', prompt)
        self.assertNotIn('"tool-1"', prompt)
        self.assertIn('["@voice","status"]', prompt)
        self.assertIn('saved microphone phrases', prompt)

    def test_format_correction_is_host_guidance_not_a_new_conversation_or_action(self):
        client = NPUConversationClient()
        task = encode_conversation([{'role': 'user', 'content': 'What does review mean?'}])
        with patch('npu_chat.NPUClient.submit', return_value={'id': 'fixture'}) as submit:
            client.submit('owner', task, role=ROLE, response_correction='Invalid evidence format.')
        prompt = submit.call_args.args[1]
        self.assertIn('HOST FORMAT CORRECTION', prompt)
        self.assertIn('What does review mean?', prompt)
        self.assertIn('without evidence fields or citation IDs', prompt)
        self.assertNotIn('response_correction', submit.call_args.kwargs)

    def test_helpers_cannot_use_the_main_conversation_adapter(self):
        client = NPUConversationClient()
        with patch('npu_chat.NPUClient.submit') as submit, self.assertRaises(LocalError):
            client.submit('owner', 'tool-looking text', role='reviewer')
        submit.assert_not_called()

    def test_plain_token_limited_text_is_labelled_but_partial_actions_are_not(self):
        client = NPUConversationClient()
        for text, expected in [('Useful partial answer.', True),
                               ('{"tool":"workspace","arguments":{"operation":"write"', False),
                               ('I will inspect it.\n{"tool":"workspace","arguments":', False)]:
            response = {'state': 'completed', 'result': text, 'truncated': True,
                        'result_truncated': True, 'usage': {'token_limit_reached': True}}
            with patch('npu_chat.NPUClient.read', return_value=response):
                result = client.read('owner', 'fixture', limit=32768)
            self.assertEqual(result.get('answer_incomplete') is True, expected)
            if expected:
                self.assertIn('Answer incomplete', json.loads(result['result'])['final'])


class QuotaFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'workspace'
        self.root.mkdir()
        self.store = Store(Path(self.temp.name) / 'state')
        self.store.setting('action_policy', 'auto')
        self.sid = self.store.create(self.root)
        self.agent = None

    def tearDown(self):
        if self.agent:
            self.agent.close()
        self.store.db.close()
        self.temp.cleanup()

    def create(self, cloud_replies, npu_replies, status=None):
        cloud = FakeClient(cloud_replies)
        cloud.engine_name = 'gemini'
        cloud.engine_label = 'Gemini fixture'
        cloud.close = Mock()
        cloud.requires_native_tool_calls = True
        cloud.status = Mock(return_value={'ready': True, 'model': {
            'backend': 'google-gemini-api', 'context_per_worker': 32768,
            'max_output_tokens': 2048, 'native_conversation_protocol': 1}})
        npu = RepairNPUClient(npu_replies)
        npu.engine_name = 'local-npu'
        npu.engine_label = 'Fixture Gemma · Tensor NPU · Gemini quota fallback'
        npu.supports_host_conversation = True
        npu.status = Mock(return_value=status or npu_status())
        npu.start = Mock()
        factory = Mock(return_value=npu)
        self.agent = LocalChatAgent(self.store, self.root, client=cloud, fallback_factory=factory)
        self.agent.local.cancel = Mock()
        return cloud, npu, factory

    def run_turn(self, text='Hello', ident='request-quota-0001'):
        self.agent.submit(self.sid, text, ident)
        self.agent.start_thread.join(4)
        self.assertFalse(self.agent.start_thread.is_alive())

    def test_quota_switches_once_and_stays_local_for_following_turns(self):
        cloud, npu, factory = self.create([QUOTA], ['Local answer.', 'Another local answer.'])
        self.run_turn()
        self.assertEqual(self.agent.last_answer, 'Local answer.')
        self.assertIn('Tensor NPU', self.agent.engine_label)
        self.run_turn('Continue', 'request-quota-0002')
        self.assertEqual(self.agent.last_answer, 'Another local answer.')
        self.assertEqual(len(cloud.submissions), 1)
        self.assertEqual(len(npu.submissions), 2)
        self.assertEqual(npu.submissions[0]['max_tokens'], 512)
        factory.assert_called_once()
        cloud.close.assert_called_once()
        npu.start.assert_not_called()

    def test_known_quota_latches_npu_while_its_enabled_supervisor_recovers(self):
        cloud, npu, _ = self.create([QUOTA], ['Ready after recovery.', 'Still local.'],
                                   npu_status(ready=False, recovering=True))
        npu.wait_until_ready = Mock(return_value=npu_status())
        self.run_turn()
        self.assertEqual(self.agent.last_answer, 'Ready after recovery.')
        self.run_turn('Continue locally', 'request-recovered-local-0002')
        self.assertEqual(self.agent.last_answer, 'Still local.')
        self.assertEqual(len(cloud.submissions), 1)
        self.assertGreaterEqual(npu.wait_until_ready.call_count, 2)
        npu.start.assert_not_called()

    def test_only_a_definitely_unsent_generation_can_retry_after_recovery(self):
        cloud, npu, _ = self.create([QUOTA], ['Recovered without another quota request.'])
        submit = npu.submit
        attempts = []
        def request(*args, **kwargs):
            attempts.append(args)
            if len(attempts) == 1:
                raise LocalRequestNotSent('Connection refused before any request bytes')
            return submit(*args, **kwargs)
        npu.submit = request
        npu.wait_until_ready = Mock(return_value=npu_status())
        self.run_turn()
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(npu.submissions), 1)
        self.assertEqual(len(cloud.submissions), 1)
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')

    def test_ambiguous_submit_loss_is_not_replayed_and_a_fresh_user_request_can_recover(self):
        cloud, npu, _ = self.create([QUOTA], ['Fresh explicit retry succeeded.'])
        submit = npu.submit
        npu.submit = Mock(side_effect=LocalError('Reply was lost after submitting the generation'))
        self.run_turn()
        npu.submit.assert_called_once()
        self.assertEqual(self.store.session(self.sid)['status'], 'failed')
        npu.submit = submit
        self.run_turn('Try again now', 'request-explicit-recovery-0002')
        self.assertEqual(self.agent.last_answer, 'Fresh explicit retry succeeded.')
        self.assertEqual(len(cloud.submissions), 1)

    def test_restart_only_steering_drains_and_discards_stale_actions_and_partials(self):
        cloud, npu, _ = self.create([QUOTA], [
            '{"tool":"workspace","arguments":{"operation":"write","path":"stale.txt","content":"must not run"}}',
            'Only the newest answer.'])
        npu.interrupt_requires_restart = True
        entered, release = threading.Event(), threading.Event()
        read = npu.read
        drained = []
        def response(owner, ident, wait, limit):
            if ident == 'job-0':
                entered.set()
                if not release.wait(.03):
                    return {'state': 'running'}
                drained.append(ident)
                if len(drained) == 1:
                    return {'state': 'running', 'partial': 'OBSOLETE PARTIAL'}
            return read(owner, ident, wait, limit)
        npu.read = response
        with patch.object(self.agent, 'execute') as execute:
            self.agent.submit(self.sid, 'Original request', 'request-drain-0001')
            self.assertTrue(entered.wait(2))
            self.agent.steer(self.sid, 'Use the newest request', 'voice-drain-0002')
            release.set()
            self.agent.start_thread.join(4)
        self.assertFalse(self.agent.start_thread.is_alive())
        execute.assert_not_called()
        self.assertEqual(npu.cancelled, [])
        self.assertEqual(len(npu.submissions), 2)
        self.assertFalse((self.root / 'stale.txt').exists())
        self.assertEqual(self.agent.last_answer, 'Only the newest answer.')
        self.assertIn('Use the newest request', npu.submissions[1]['prompt'])
        deltas = [event['data']['text'] for event in self.store.events(self.sid) if event['kind'] == 'assistant_delta']
        self.assertNotIn('OBSOLETE PARTIAL', ''.join(deltas))

    def test_obsolete_completed_truncated_json_is_discarded_before_any_format_error(self):
        _, npu, _ = self.create([QUOTA], [{'state': 'completed', 'result': '{"tool":"workspace",',
                                         'truncated': True}, 'Fresh answer after draining.'])
        npu.interrupt_requires_restart = True
        read = npu.read
        def response(owner, ident, wait, limit):
            if ident == 'job-0':
                self.agent.steer(self.sid, 'A newer request', 'voice-stale-complete-0002')
            return read(owner, ident, wait, limit)
        npu.read = response
        self.run_turn('Original request')
        self.assertEqual(npu.cancelled, [])
        self.assertEqual(len(npu.submissions), 2)
        self.assertEqual(self.agent.last_answer, 'Fresh answer after draining.')
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')

    def test_explicit_cancel_still_interrupts_restart_only_inference(self):
        _, npu, _ = self.create([QUOTA], ['Must not be delivered.'])
        npu.interrupt_requires_restart = True
        npu.block = True
        self.agent.submit(self.sid, 'Request to cancel', 'request-drain-cancel-0001')
        self.assertTrue(npu.entered.wait(2))
        self.agent.cancel(self.sid)
        self.agent.start_thread.join(4)
        self.assertTrue(npu.cancelled)
        self.assertEqual(len(npu.submissions), 1)
        self.assertEqual(self.store.session(self.sid)['status'], 'interrupted')

    def test_completed_host_action_is_not_replayed_after_quota_handoff(self):
        arguments = {'operation': 'write', 'path': 'result.txt', 'content': 'written once'}
        call = {'id': 'cloud-write', 'type': 'function', 'function': {
            'name': 'workspace', 'arguments': json.dumps(arguments)}}
        cloud, npu, _ = self.create([
            {'state': 'completed', 'result': json.dumps({'tool': 'workspace', 'arguments': arguments}), 'native_call': call},
            QUOTA], ['The file was saved.'])
        with patch.object(self.agent, 'execute', wraps=self.agent.execute) as execute:
            self.run_turn('Write result.txt')
        execute.assert_called_once_with(arguments)
        self.assertEqual((self.root / 'result.txt').read_text(), 'written once')
        messages = json.loads(npu.submissions[0]['prompt'])['messages']
        self.assertEqual(messages[1]['tool_calls'][0]['id'], 'cloud-write')
        self.assertEqual(messages[2]['tool_call_id'], 'cloud-write')
        self.assertIn('result.txt', messages[2]['content'])
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')

    def test_npu_can_request_a_new_validated_host_action(self):
        (self.root / 'fixture.txt').write_text('Actual fixture content')
        _, npu, _ = self.create([QUOTA], [
            '{"tool":"workspace","arguments":{"operation":"read","path":"fixture.txt"}}',
            'Actual fixture content'])
        with patch.object(self.agent, 'execute', wraps=self.agent.execute) as execute:
            self.run_turn('Read fixture.txt')
        execute.assert_called_once_with({'operation': 'read', 'path': 'fixture.txt', 'limit': 4000})
        self.assertIn('Actual fixture content', npu.submissions[1]['prompt'])
        self.assertEqual(self.agent.last_answer, 'Actual fixture content')

    def test_ambiguous_output_authentication_and_network_failures_do_not_fallback(self):
        for index, response in enumerate((dict(QUOTA, safe_quota_fallback=False), dict(QUOTA, state='cancelled'),
                         {'state': 'failed', 'error_code': 'http_401', 'error': 'auth'},
                         {'state': 'failed', 'error_code': 'network', 'error': 'unknown completion'})):
            cloud, npu, factory = self.create([response], [])
            self.run_turn(ident='request-quota-failure-' + str(index))
            factory.assert_not_called()
            self.assertEqual(npu.submissions, [])
            self.assertEqual(self.store.session(self.sid)['status'], 'failed')
            self.agent.close()
            self.agent = None
            self.sid = self.store.create(self.root)

    def test_disabled_npu_is_never_started(self):
        _, npu, _ = self.create([QUOTA], [], npu_status(service_enabled=False))
        self.run_turn()
        self.assertEqual(self.store.session(self.sid)['status'], 'failed')
        npu.start.assert_not_called()
        self.assertEqual(npu.submissions, [])

    def test_a_cpu_backend_cannot_be_misreported_or_selected_as_npu(self):
        status = npu_status()
        status['model']['backend'] = 'llama-cpu'
        _, npu, _ = self.create([QUOTA], [], status)
        self.run_turn()
        self.assertEqual(self.store.session(self.sid)['status'], 'failed')
        self.assertEqual(npu.submissions, [])
        self.assertEqual(self.agent.engine_label, 'Gemini fixture')

    def test_cancel_during_fallback_preflight_never_submits_local_work(self):
        _, npu, _ = self.create([QUOTA], [])
        def status():
            self.agent.cancelled.set()
            return npu_status()
        npu.status.side_effect = status
        self.run_turn()
        self.assertEqual(npu.submissions, [])
        self.assertEqual(self.store.session(self.sid)['status'], 'interrupted')

    def test_incomplete_npu_action_cannot_write_a_file(self):
        self.create([QUOTA], [{'state': 'completed', 'result':
            '{"tool":"workspace","arguments":{"operation":"write","path":"unsafe.txt"',
            'truncated': True}])
        with patch.object(self.agent, 'execute') as execute:
            self.run_turn('Write unsafe.txt')
        execute.assert_not_called()
        self.assertFalse((self.root / 'unsafe.txt').exists())
        self.assertEqual(self.store.session(self.sid)['status'], 'failed')

    def test_invalid_npu_final_evidence_gets_one_fresh_plain_text_answer(self):
        _, npu, _ = self.create([QUOTA], [
            '{"final":"draft answer","evidence":"tool-1"}',
            'Review means saved microphone phrases awaiting your decision.'])
        with patch.object(self.agent, 'execute') as execute:
            self.run_turn('What does review mean?')
        execute.assert_not_called()
        self.assertEqual(len(npu.submissions), 2)
        self.assertIn('invalid evidence', npu.submissions[1]['response_correction'])
        self.assertIn('saved microphone phrases', self.agent.last_answer)
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')

    def test_prose_with_malformed_tool_gets_repaired_before_any_execution(self):
        (self.root / 'probe.txt').write_text('Observed metadata 82917')
        _, npu, _ = self.create([QUOTA], [
            'I need to inspect it.\n\n{"tool":"@npu","arguments":{"status":true}}',
            '{"tool":"workspace","arguments":{"operation":"read","path":"probe.txt"}}',
            'The probe reports 82917.'])
        original_execute = self.agent.execute
        with patch.object(self.agent, 'execute', wraps=original_execute) as execute:
            self.run_turn('Inspect probe.txt')
        self.assertEqual(execute.call_count, 1)
        self.assertEqual(execute.call_args.args[0]['path'], 'probe.txt')
        self.assertIn('mixed prose', npu.submissions[1]['response_correction'])
        self.assertIn('Observed metadata 82917', npu.submissions[2]['prompt'])
        self.assertEqual(self.agent.last_answer, 'The probe reports 82917.')

    def test_forged_npu_receipts_are_rejected_before_a_corrected_answer(self):
        _, npu, _ = self.create([QUOTA], ['{"final":"unsupported claim","evidence":["tool-99"]}',
                                          'I can check the current voice interface status.'])
        with patch.object(self.agent, 'execute') as execute:
            self.run_turn('On this interface')
        execute.assert_not_called()
        self.assertIn('does not exist', npu.submissions[1]['response_correction'])
        self.assertNotIn('unsupported claim', self.agent.last_answer)

    def test_malformed_request_is_regenerated_then_host_read_runs_once(self):
        (self.root / 'fixture.txt').write_text('Actual fixture content')
        _, npu, _ = self.create([QUOTA], [
            '{"tool":"workspace","arguments":{"operation":"read","path":"fixture.txt"}',
            '{"tool":"workspace","arguments":{"operation":"read","path":"fixture.txt"}}',
            '{"final":"Actual fixture content","evidence":["not-a-receipt"]}',
            'Actual fixture content'])
        with patch.object(self.agent, 'execute', wraps=self.agent.execute) as execute:
            self.run_turn('Read fixture.txt')
        execute.assert_called_once()
        self.assertEqual(len(npu.submissions), 4)
        self.assertEqual(self.agent.last_answer, 'Actual fixture content')
        self.assertIn('Actual fixture content', npu.submissions[-1]['prompt'])
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')

    def test_format_repairs_stop_after_two_without_executing_any_draft(self):
        _, npu, _ = self.create([QUOTA], ['{"tool":"workspace",'] * 3)
        with patch.object(self.agent, 'execute') as execute:
            self.run_turn('Find out')
        execute.assert_not_called()
        self.assertEqual(len(npu.submissions), 3)
        self.assertEqual(self.store.session(self.sid)['status'], 'failed')

    def test_live_interface_metadata_is_refreshed_without_persisting_it_as_user_text(self):
        self.create([QUOTA], ['Review means saved microphone phrases.', 'Your microphone is muted.'])
        status = {'available': True, 'interface': 'terminal', 'microphone_muted': False,
                  'speech_queue': {'review': 32, 'queued': 0, 'sending': 0}}
        with patch.object(self.agent.interface, 'read_context', create=True, return_value=status) as read:
            self.run_turn('What does review mean?')
            first = self.agent.local_client.submissions[0]['prompt']
            self.assertIn('LIVE INTERFACE STATUS', first)
            first_user = json.loads(first)['messages'][-1]['content']
            self.assertIn('"review":32', first_user)
            self.assertLess(first_user.index('LIVE INTERFACE STATUS'), first_user.index('CURRENT USER REQUEST:\n'))
            self.assertTrue(first_user.endswith('CURRENT USER REQUEST:\nWhat does review mean?'))
            status = dict(status, microphone_muted=True)
            read.return_value = status
            self.run_turn('Am I muted?', 'request-npu-mute-context-0002')
            second = self.agent.local_client.submissions[-1]['prompt']
            second_user = json.loads(second)['messages'][-1]['content']
            self.assertIn('"microphone_muted":true', second_user)
            self.assertLess(second_user.index('LIVE INTERFACE STATUS'), second_user.index('CURRENT USER REQUEST:\n'))
            self.assertTrue(second_user.endswith('CURRENT USER REQUEST:\nAm I muted?'))
            self.assertEqual(second_user.count('LIVE INTERFACE STATUS'), 1)
        contexts = [event['data']['conversation'] for event in self.store.events(self.sid)
                    if event['kind'] == 'local_context']
        self.assertTrue(contexts)
        self.assertTrue(all('LIVE INTERFACE STATUS' not in context for context in contexts))
        saved_users = [message['content'] for message in json.loads(contexts[-1])['messages']
                       if message['role'] == 'user']
        self.assertEqual(saved_users, ['What does review mean?', 'Am I muted?'])
        users = [event['data']['text'] for event in self.store.events(self.sid) if event['kind'] == 'user']
        self.assertEqual(users, ['What does review mean?', 'Am I muted?'])

    def test_legacy_cpu_prompt_retains_live_status_without_overrunning_its_budget(self):
        client = FakeClient(['Review contains saved speech.'])
        self.agent = LocalChatAgent(self.store, self.root, client=client)
        self.agent.local.cancel = Mock()
        status = {'available': True, 'interface': 'terminal', 'speech_queue': {'review': 32},
                  'commands': {'help': 'x' * 1400}}
        with patch.object(self.agent.interface, 'read_context', create=True, return_value=status):
            self.run_turn('What does review mean?')
        prompt = client.submissions[0]['prompt']
        self.assertLessEqual(len(prompt), self.agent.MAX_PROMPT)
        self.assertIn('LIVE INTERFACE STATUS', prompt)
        self.assertIn('"review":32', prompt)
        self.assertLess(prompt.index('LIVE INTERFACE STATUS'),
                        prompt.index('CURRENT USER REQUEST:\nWhat does review mean?'))
        self.assertTrue(prompt.endswith('CURRENT USER REQUEST:\nWhat does review mean?'))
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')
