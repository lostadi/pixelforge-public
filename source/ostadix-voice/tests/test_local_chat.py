import copy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent import Agent, AppError, Store
from local_chat import COMPACT_GUIDE, GUIDE, LocalChatAgent, create_agent, parse_response, tool_feedback, native_answer, explicit_ostadix_action
from voice_inbox import VoiceInbox


class FakeClient:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.submissions = []
        self.jobs = {}
        self.cancelled = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.context = 8192
        self.max_output_tokens = 512

    def status(self):
        return {'ready': True, 'model': {'backend': 'llama-cpu', 'context_per_worker': self.context,
                                      'max_output_tokens': self.max_output_tokens}}

    def submit(self, owner, prompt, max_tokens, role):
        ident = 'job-' + str(len(self.submissions))
        reply = next(self.replies)
        self.submissions.append({'id': ident, 'owner': owner, 'prompt': prompt,
                                 'max_tokens': max_tokens, 'role': role})
        self.jobs[ident] = reply
        return {'id': ident, 'state': 'queued'}

    def read(self, owner, ident, wait, limit):
        self.entered.set()
        if self.block:
            self.release.wait(.03)
            if not self.release.is_set():
                return {'state': 'running'}
        reply = self.jobs[ident]
        if isinstance(reply, dict):
            return reply
        return {'state': 'completed', 'result': reply, 'usage': {}, 'truncated': False}

    def cancel(self, owner, ident):
        self.cancelled.append((owner, ident))
        self.release.set()
        return {'state': 'cancelled'}


class LocalChatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'project'
        self.root.mkdir()
        self.store = Store(Path(self.temp.name) / 'state')
        self.sid = self.store.create(self.root)
        self.agent = None

    def tearDown(self):
        if self.agent:
            self.agent.close()
        self.store.db.close()
        self.temp.cleanup()

    def make_agent(self, replies):
        client = FakeClient(replies)
        self.agent = LocalChatAgent(self.store, self.root, client=client)
        # Cancel/close should not consult another installed helper service.
        self.agent.local.cancel = Mock()
        return client

    def run_turn(self, text='Hello', ident='request-0001'):
        self.agent.submit(self.sid, text, ident)
        self.agent.start_thread.join(8)
        self.assertFalse(self.agent.start_thread.is_alive())
        return self.store.events(self.sid)

    def wait_for(self, predicate):
        deadline = time.monotonic() + 3
        while not predicate() and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertTrue(predicate())

    def test_plain_reply_is_persisted_without_rpc_or_cloud(self):
        client = self.make_agent(['Gemma is ready.'])
        with patch('agent.codex_command', side_effect=AssertionError('Cloud route used')):
            events = self.run_turn()
        self.assertIsNone(self.agent.rpc)
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')
        self.assertEqual(self.agent.last_answer, 'Gemma is ready.')
        self.assertEqual([e['data']['text'] for e in events if e['kind'] == 'assistant'], ['Gemma is ready.'])
        self.assertEqual(client.submissions[0]['max_tokens'], 512)
        self.assertIn('CURRENT USER REQUEST:\nHello', client.submissions[0]['prompt'])

    def native_client(self, replies):
        client = self.make_agent(replies)
        client.status = Mock(return_value={'ready': True, 'model': {'backend': 'llama-cpu',
                             'context_per_worker': 8192, 'max_output_tokens': 512, 'native_conversation_protocol': 1}})
        return client

    def test_main_response_can_finish_after_old_deadline(self):
        client = self.make_agent(['Long response finished.'])
        original = client.read
        clock = [1000.0]
        reads = []

        def delayed_read(*args, **kwargs):
            reads.append(True)
            if len(reads) == 1:
                clock[0] += 600
                return {'state': 'running'}
            return original(*args, **kwargs)

        client.read = delayed_read
        with patch('local_chat.time.monotonic', side_effect=lambda: clock[0]):
            self.run_turn()
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')
        self.assertEqual(self.agent.last_answer, 'Long response finished.')
        self.assertEqual(client.cancelled, [])

    def test_native_history_keeps_actual_tool_pair_and_followup(self):
        (self.root / 'sample.txt').write_text('x' * 2400 + '\nActual release code: BERYL-73129')
        client = self.native_client([
            '{"tool":"workspace","arguments":{"operation":"read","path":"sample.txt"}}',
            'The release code is BERYL-73129.', 'It was BERYL-73129.',
        ])
        self.run_turn('Read sample.txt and find its release code.')
        second = json.loads(client.submissions[1]['prompt'])['messages']
        call, result = second[-2:]
        self.assertEqual(call['role'], 'assistant')
        self.assertEqual(result['role'], 'tool')
        self.assertEqual(call['tool_calls'][0]['id'], result['tool_call_id'])
        content = json.loads(result['content'])['result']
        self.assertIn('BERYL-73129', content['content'])
        self.assertFalse(content['more'])
        self.run_turn('What was that code?', 'request-0002')
        history = json.loads(client.submissions[2]['prompt'])['messages']
        self.assertTrue(any(m['role'] == 'tool' for m in history))
        self.assertEqual(history[-2]['content'], 'The release code is BERYL-73129.')
        self.assertEqual(self.agent.last_answer, 'It was BERYL-73129.')

    def test_native_interface_data_precedes_request_and_keeps_tool_pairs_and_saved_source(self):
        (self.root / 'sample.txt').write_text('Actual result: cedar')
        request = 'Read sample.txt.\nPreserve this text: café?'
        client = self.native_client([
            '{"tool":"workspace","arguments":{"operation":"read","path":"sample.txt"}}',
            'The file says cedar.',
        ])
        status = {'available': True, 'microphone_muted': True, 'interface': 'terminal'}
        with patch.object(self.agent.interface, 'read_context', create=True, return_value=status):
            events = self.run_turn(request)
        for submission in client.submissions:
            messages = json.loads(submission['prompt'])['messages']
            user = next(message for message in reversed(messages) if message['role'] == 'user')
            self.assertLess(user['content'].index('LIVE INTERFACE STATUS'),
                            user['content'].index('CURRENT USER REQUEST:\n'))
            self.assertTrue(user['content'].endswith('CURRENT USER REQUEST:\n' + request))
            self.assertEqual(user['content'].count('LIVE INTERFACE STATUS'), 1)
        tool_messages = json.loads(client.submissions[-1]['prompt'])['messages']
        self.assertEqual([message['role'] for message in tool_messages], ['user', 'assistant', 'tool'])
        self.assertEqual(tool_messages[1]['tool_calls'][0]['id'], tool_messages[2]['tool_call_id'])
        self.assertIn('Actual result: cedar', tool_messages[2]['content'])
        saved = [json.loads(event['data']['conversation'])['messages'] for event in events
                 if event['kind'] == 'local_context']
        self.assertTrue(saved)
        for messages in saved:
            self.assertEqual(messages[0], {'role': 'user', 'content': request})
            self.assertNotIn('LIVE INTERFACE STATUS', json.dumps(messages))
        self.assertEqual([event['data']['text'] for event in events if event['kind'] == 'user'], [request])

    def test_legacy_interface_data_precedes_request_without_displacing_tool_receipts(self):
        (self.root / 'sample.txt').write_text('Actual result: birch')
        request = 'Read sample.txt.\nPreserve this text: café?'
        client = self.make_agent([
            '{"tool":"workspace","arguments":{"operation":"read","path":"sample.txt"}}',
            'The file says birch.',
        ])
        status = {'available': True, 'microphone_muted': True, 'interface': 'terminal'}
        with patch.object(self.agent.interface, 'read_context', create=True, return_value=status):
            events = self.run_turn(request)
        for submission in client.submissions:
            prompt = submission['prompt']
            self.assertLess(prompt.index('LIVE INTERFACE STATUS'),
                            prompt.index('CURRENT USER REQUEST:\n' + request))
            self.assertEqual(prompt.count('LIVE INTERFACE STATUS'), 1)
        prompt = client.submissions[-1]['prompt']
        self.assertLess(prompt.index('CURRENT USER REQUEST:\n' + request),
                        prompt.index('THIS TURN TOOL RECEIPTS'))
        self.assertIn('Actual result: birch', prompt)
        self.assertEqual([event['data']['text'] for event in events if event['kind'] == 'user'], [request])
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')

    def test_large_tool_feedback_is_valid_json_with_a_durable_retrieval_handle(self):
        result = {'packages': ['package.name.' + str(index) for index in range(1000)],
                  'content': '"\\\n漢字' * 1000}
        preview = tool_feedback(result, True, limit=1400, event=271)
        data = json.loads(preview)
        self.assertLessEqual(len(preview), 1400)
        self.assertTrue(data['success'])
        self.assertTrue(data['preview_truncated'])
        self.assertEqual(data['saved_result'], {'event': 271, 'read_argv': ['@results', 'read', '271', '0', '4000']})
        self.assertNotIn('full result is saved in /logs', preview)

    @staticmethod
    def hardware_result():
        return {'phone_hardware': True, 'read_only': True, 'device_model': 'Fixture Phone',
                'soc': {'manufacturer': 'Fixture Maker', 'model': 'Fixture Chip'},
                'abi': ['arm64-v8a'], 'core_count': 8, 'status': 'partial',
                'board_platform': 'fixture-board', 'present_cpus': list(range(8)), 'online_cpus': list(range(8)),
                'cpu_identifiers': [{'implementer': '0x41', 'part': '0xd80', 'count': 8}],
                'clusters': [{'policy': 'policy0', 'cpus': list(range(8)), 'max_frequency_khz': 2000000}],
                'sources': {'device_model': {'property': 'ro.product.model', 'argv': ['/system/bin/getprop', 'ro.product.model']},
                            'soc.model': {'property': 'ro.soc.model', 'argv': ['/system/bin/getprop', 'ro.soc.model']},
                            'core_count': '/proc/cpuinfo',
                            'clusters.policy0.max_frequency_khz': '/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq'},
                'unavailable': {'online_cpus': 'permission denied'}, 'note': 'Full diagnostic caveat. ' * 80}

    def test_cpu_model_view_is_scoped_to_successful_saved_exact_hardware_reads(self):
        result = self.hardware_result()
        original = copy.deepcopy(result)
        args = {'operation': 'command', 'argv': ['@phone', 'cpu']}
        view = json.loads(tool_feedback(result, True, limit=20000, event=31, arguments=args))
        for field in ('device_model', 'soc', 'abi', 'core_count', 'unavailable'):
            self.assertEqual(view['result'][field], result[field])
        self.assertEqual(view['result']['sources']['soc.model'], {'property': 'ro.soc.model'})
        self.assertEqual(view['result']['sources']['core_count'], '/proc/cpuinfo')
        self.assertIn('clusters', view['result']['diagnostic_details_omitted'])
        self.assertEqual(view['saved_result']['read_argv'], ['@results', 'read', '31', '0', '4000'])
        self.assertLess(len(json.dumps(view)), 1100)
        self.assertEqual(result, original)
        for success, event, action in (
                (False, 31, args), (True, None, args), (True, True, args), (True, 0, args),
                (True, 31, None), (True, 31, {'operation': 'read', 'path': 'cpu.json'}),
                (True, 31, {'operation': 'command', 'argv': ['@phone', 'volume']}),
                (True, 31, {'operation': 'command', 'argv': ['@phone', 'cpu', 'extra']})):
            with self.subTest(success=success, event=event, action=action):
                data = json.loads(tool_feedback(result, success, limit=20000, event=event, arguments=action))
                self.assertEqual(data['result'], original)
        alias = json.loads(tool_feedback(result, True, limit=20000, event=31,
                           arguments={'operation': 'command', 'argv': ['@phone', 'hardware']}))
        self.assertEqual(alias['result'], view['result'])

    def test_cpu_projection_preserves_unknown_errors_and_requires_real_hardware_markers(self):
        args = {'operation': 'command', 'argv': ['@phone', 'cpu']}
        for changes in ({'error': 'unexpected hardware error'}, {'warning': 'new important field'},
                        {'phone_hardware': False}, {'read_only': False}, {'phone_hardware': 1}, {'read_only': 1}):
            result = dict(self.hardware_result(), **changes)
            with self.subTest(changes=changes):
                data = json.loads(tool_feedback(result, True, limit=20000, event=31, arguments=args))
                self.assertEqual(data['result'], result)
        partial = self.hardware_result()
        partial['soc']['model'] = None
        partial['unavailable']['soc.model'] = 'property unreadable'
        data = json.loads(tool_feedback(partial, True, limit=20000, event=31, arguments=args))
        self.assertIsNone(data['result']['soc']['model'])
        self.assertEqual(data['result']['unavailable'], partial['unavailable'])

    def test_projected_cpu_receipt_retrieves_the_full_unchanged_saved_evidence(self):
        self.make_agent([])
        self.agent.sid = self.sid
        result = self.hardware_result()
        event = self.store.event(self.sid, 'tool_result', {'success': True, 'result': result})
        content = json.loads(self.agent._tool_content(result, True, 'tool-1', event,
                             arguments={'operation': 'command', 'argv': ['@phone', 'cpu']}))
        argv, chunks = content['saved_result']['read_argv'], []
        while True:
            page = self.agent.execute({'operation': 'command', 'argv': argv})
            chunks.append(page['content'])
            if not page['more']:
                break
            argv = ['@results', 'read', str(event), str(page['next_offset']), '4000']
        self.assertEqual(json.loads(''.join(chunks)), result)
        self.assertIn('clusters', json.loads(''.join(chunks)))
        stored = next(item for item in self.store.events(self.sid) if item['seq'] == event)
        self.assertEqual(stored['data']['result'], result)

    def test_native_shortened_result_points_to_the_complete_saved_event(self):
        result = {'packages': ['package.name.' + str(index) for index in range(1000)]}
        client = self.native_client(['{"tool":"workspace","arguments":{"operation":"command","argv":["@phone","apps"]}}',
                                     'The installed package inventory was retrieved.'])
        with patch.object(self.agent, 'execute', return_value=result):
            events = self.run_turn('List installed packages.')
        receipt = next(event for event in events if event['kind'] == 'tool_result')
        self.assertEqual(receipt['data']['result'], result)
        messages = json.loads(client.submissions[1]['prompt'])['messages']
        content = json.loads(messages[-1]['content'])
        self.assertTrue(content['preview_truncated'])
        self.assertEqual(content['saved_result']['event'], receipt['seq'])
        self.assertEqual(content['saved_result']['read_argv'], ['@results', 'read', str(receipt['seq']), '0', '4000'])

    def test_later_turn_retrieves_saved_inventory_even_without_a_reusable_context(self):
        inventory = {'packages': ['package.name.' + str(index) for index in range(1000)], 'count': 1000}
        event = self.store.event(self.sid, 'tool_result', {'success': True, 'result': inventory})
        encoded = json.dumps(inventory, ensure_ascii=False, separators=(',', ':'))
        offset = len(encoded) - 3000
        client = self.native_client([
            json.dumps({'tool': 'workspace', 'arguments': {'operation': 'command', 'argv': ['@results', 'list']}}),
            json.dumps({'tool': 'workspace', 'arguments': {'operation': 'command',
                        'argv': ['@results', 'read', str(event), str(offset), '4000']}}),
            'The last saved package is package.name.999.',
        ])
        self.run_turn('Read the end of the package list you retrieved earlier.')
        listed = json.loads(json.loads(client.submissions[1]['prompt'])['messages'][-1]['content'])['result']
        self.assertEqual(listed['results'][0]['event'], event)
        retrieved = json.loads(json.loads(client.submissions[2]['prompt'])['messages'][-1]['content'])['result']
        self.assertEqual(retrieved['content'], encoded[offset:])
        self.assertFalse(retrieved['more'])
        self.assertIn('package.name.999', self.agent.last_answer)
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')

    def test_native_context_compacts_old_result_bodies_without_losing_calls_or_handles(self):
        messages = [{'role': 'user', 'content': 'Inspect all pages.'}]
        for index in range(7):
            identifier = 'call_' + str(index)
            messages.extend([
                {'role': 'assistant', 'content': None, 'tool_calls': [{'id': identifier, 'type': 'function',
                    'function': {'name': 'workspace', 'arguments': json.dumps({'operation': 'read', 'path': 'file.txt', 'offset': index * 4000})}}]},
                {'role': 'tool', 'tool_call_id': identifier, 'name': 'workspace', 'content':
                    LocalChatAgent._tool_content({'content': 'x' * 5000}, True, 'tool-' + str(index + 1), index + 100)},
            ])
        prompt = LocalChatAgent._fit_conversation(messages, 8192, 512)
        fitted = json.loads(prompt)['messages']
        self.assertEqual(len(fitted), 15)
        calls = [message['tool_calls'][0]['id'] for message in fitted if message['role'] == 'assistant']
        responses = [message for message in fitted if message['role'] == 'tool']
        self.assertEqual(calls, [message['tool_call_id'] for message in responses])
        self.assertTrue(any(json.loads(message['content']).get('context_compacted') for message in responses))
        self.assertEqual([json.loads(message['content'])['saved_result']['event'] for message in responses], list(range(100, 107)))

    def test_safe_incomplete_text_is_displayed_and_prior_successful_actions_stay_saved(self):
        self.native_client(['{"tool":"workspace","arguments":{"operation":"read","path":"sample.txt"}}',
                            {'state': 'completed', 'result': json.dumps({'final': 'Partial observed inventory. [This answer is incomplete.]'}),
                             'truncated': True, 'answer_incomplete': True}])
        with patch.object(self.agent, 'execute', return_value={'content': 'Actual retrieved inventory'}):
            events = self.run_turn('Read the inventory.')
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')
        self.assertIn('answer is incomplete', self.agent.last_answer)
        self.assertTrue(next(event['data']['success'] for event in events if event['kind'] == 'tool_result'))

    def test_incomplete_answer_flag_cannot_admit_a_partial_tool_object(self):
        self.native_client([{'state': 'completed', 'result': '{"tool":"workspace","arguments":{"operation":"write"}}',
                             'truncated': True, 'answer_incomplete': True}])
        with patch.object(self.agent, 'execute') as execute:
            self.run_turn('Do not execute incomplete actions.')
        execute.assert_not_called()
        self.assertEqual(self.store.session(self.sid)['status'], 'failed')

    def test_native_plain_text_preview_is_incremental_but_action_json_stays_hidden(self):
        client = self.native_client(['unused'])
        client.read = Mock(side_effect=[
            {'state': 'running', 'partial': '{"tool":"workspace"'},
            {'state': 'running', 'partial': 'Hello'},
            {'state': 'running', 'partial': 'Hello there'},
            {'state': 'completed', 'result': 'Hello there', 'usage': {}},
        ])
        events = self.run_turn()
        deltas = [e['data']['text'] for e in events if e['kind'] == 'assistant_delta']
        self.assertEqual(''.join(deltas), 'Hello there')
        self.assertEqual(self.agent.last_answer, 'Hello there')

    def test_native_voice_followup_replaces_inference_once_without_waiting_for_old_answer(self):
        client = self.native_client(['Stale answer', 'Updated answer'])
        client.block = True
        self.agent.submit(self.sid, 'Original request', 'request-0001')
        self.assertTrue(client.entered.wait(2))
        result = self.agent.steer(self.sid, 'Make it blue instead', 'voice-request-0001')
        self.assertTrue(result['steered'])
        duplicate = self.agent.steer(self.sid, 'Make it blue instead', 'voice-request-0001')
        self.assertTrue(duplicate['duplicate'])
        self.agent.start_thread.join(3)
        self.assertFalse(self.agent.start_thread.is_alive())
        self.assertEqual(self.agent.last_answer, 'Updated answer')
        self.assertTrue(client.cancelled)
        self.assertEqual(len(client.submissions), 2)
        prompt = json.loads(client.submissions[1]['prompt'])['messages']
        self.assertIn('Original request', prompt[-1]['content'])
        self.assertEqual(prompt[-1]['content'].count('Make it blue instead'), 1)
        self.assertEqual([e['data']['text'] for e in self.store.events(self.sid) if e['kind'] == 'assistant'], ['Updated answer'])

    def test_native_followup_during_action_preserves_completed_effect_without_replaying(self):
        client = self.native_client([
            '{"tool":"workspace","arguments":{"operation":"read","path":"sample.txt"}}',
            'Updated answer after the completed read.',
        ])
        entered, release = threading.Event(), threading.Event()
        def execute(_):
            entered.set()
            self.assertTrue(release.wait(2))
            return {'path': 'sample.txt', 'content': 'Actual evidence'}
        with patch.object(self.agent, 'execute', side_effect=execute) as executor:
            self.agent.submit(self.sid, 'Read sample.txt', 'request-0001')
            self.assertTrue(entered.wait(2))
            self.agent.steer(self.sid, 'Summarize it briefly', 'voice-request-0002')
            release.set()
            self.agent.start_thread.join(3)
            self.assertEqual(executor.call_count, 1)
        prompt = json.loads(client.submissions[1]['prompt'])['messages']
        self.assertEqual(prompt[-1]['content'], 'Summarize it briefly')
        self.assertEqual(prompt[-2]['role'], 'tool')
        self.assertIn('Actual evidence', prompt[-2]['content'])
        self.assertEqual(self.agent.last_answer, 'Updated answer after the completed read.')

    def test_completed_job_status_refreshes_the_pending_native_receipt(self):
        self.native_client([
            '{"tool":"workspace","arguments":{"operation":"command","argv":["@ostadix","call","o_execute","{}"]}}',
            '{"tool":"workspace","arguments":{"operation":"command","argv":["@ostadix","call","o_job_status","{\\"job_id\\":\\"job-one\\"}"]}}',
            'It returned 63.',
        ])
        results = [
            {'mcp_tool': 'o_execute', 'execution_pending': True, 'result': {'structuredContent':
             {'job_id': 'job-one', 'state': 'running'}}},
            {'mcp_tool': 'o_job_status', 'result': {'structuredContent': {'job_id': 'job-one',
             'state': 'completed', 'result': {'ok': True, 'value': {'t': 'number', 'v': {'kind': 'int', 'v': '63'}}}}}},
        ]
        with patch.object(self.agent, 'execute', side_effect=results):
            self.run_turn('Run the calculation and wait for its actual result')
        self.assertEqual(self.agent.last_answer, 'It returned 63.')

    def test_real_file_executor_result_reaches_model(self):
        (self.root / 'sample.txt').write_text('Actual file evidence: 42')
        client = self.make_agent([
            json.dumps({'tool': 'workspace', 'arguments': {'operation': 'read', 'path': 'sample.txt'}}),
            'The file says 42.',
        ])
        events = self.run_turn('Read sample.txt')
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')
        self.assertEqual(len(client.submissions), 2)
        self.assertIn('Actual file evidence: 42', client.submissions[1]['prompt'])
        self.assertEqual(len({job['owner'] for job in client.submissions}), 1)
        self.assertTrue(any(e['kind'] == 'tool_result' and e['data']['success'] for e in events))

    def test_ostadix_route_uses_existing_dispatcher(self):
        self.make_agent([
            json.dumps({'tool': 'workspace', 'arguments': {'operation': 'command',
                       'argv': ['@ostadix', 'call', 'o_execute', '{"source":"1 + 1","action":"execute"}']}}),
            'The runtime returned 2.',
        ])
        with patch.object(self.agent.ostadix, 'dispatch', return_value={'exit_code': 0, 'value': 2}) as dispatch:
            self.run_turn('Execute 1 + 1')
        self.assertEqual(dispatch.call_count, 1)
        self.assertEqual(dispatch.call_args.args[1]['argv'][0], '@ostadix')

    def test_explicit_ostadix_execution_requires_first_function_but_explanations_do_not(self):
        self.assertTrue(explicit_ostadix_action('Use OSTADIX to actually execute 7 * 9.'))
        self.assertFalse(explicit_ostadix_action('Explain how to execute code in OSTADIX.'))
        self.assertFalse(explicit_ostadix_action('Hello there'))
        client = self.make_agent([
            json.dumps({'tool': 'workspace', 'arguments': {'operation': 'command',
                        'argv': ['@ostadix', 'call', 'o_execute', '{"source":"test"}']}}),
            'The runtime returned its result.',
        ])
        with patch.object(self.agent.ostadix, 'dispatch', return_value={'exit_code': 0}):
            self.run_turn('Use OSTADIX to execute this example.')
        self.assertEqual([job['role'] for job in client.submissions],
                         ['phone OSTADIX action assistant', 'phone conversation assistant'])

    def test_mcp_native_value_survives_large_metadata_and_raw_result_is_saved(self):
        native = {'metadata': 'large diagnostic ' * 1000, 'ok': True, 'type': 'number',
                  'value': {'t': 'number', 'v': {'kind': 'int', 'v': '2'}}}
        result = {'mcp_tool': 'o_execute', 'exit_code': 0, 'execution_pending': False,
                  'result': {'content': [{'type': 'text', 'text': 'duplicate ' * 4000}],
                             'structuredContent': {'metadata': 'large job ' * 1000,
                                                   'action': 'execute', 'state': 'completed',
                                                   'exit_code': 0, 'result': native}}}
        client = self.make_agent([
            json.dumps({'tool': 'workspace', 'arguments': {'operation': 'command',
                        'argv': ['@ostadix', 'call', 'o_execute', '{}']}}), 'It returned 2.'])
        with patch.object(self.agent.ostadix, 'dispatch', return_value=result):
            events = self.run_turn('Execute the example')
        prompt = client.submissions[1]['prompt']
        self.assertIn('"v": "2"', prompt)
        self.assertIn('"state": "completed"', prompt)
        raw = next(event['data']['result'] for event in events if event['kind'] == 'tool_result')
        self.assertEqual(raw, result)
        pending = dict(result, execution_pending=True, result={'isError': False, 'structuredContent':
                       {'action': 'execute', 'state': 'running', 'job_id': 'pending-job'}})
        self.assertIn('"execution_pending": true', tool_feedback(pending, True))
        self.assertIn('"state": "running"', tool_feedback(pending, True))
        failed = dict(result, exit_code=1, result={'isError': True, 'structuredContent':
                      {'action': 'execute', 'state': 'failed', 'error': 'Actual runtime error'}})
        self.assertIn('Actual runtime error', tool_feedback(failed, False))
        self.assertIn('"isError": true', tool_feedback(failed, False))

    def test_ask_policy_denial_prevents_actual_write(self):
        client = self.make_agent([
            json.dumps({'tool': 'workspace', 'arguments': {'operation': 'write', 'path': 'new.txt', 'content': 'no'}}),
            'The change was denied.',
        ])
        self.agent.set_action_policy('ask')
        self.agent.submit(self.sid, 'Write new.txt', 'request-0001')
        self.wait_for(lambda: bool(self.agent.approvals))
        approval = next(iter(self.agent.approvals))
        self.agent.approve(self.sid, approval, False)
        self.agent.start_thread.join(3)
        self.assertFalse((self.root / 'new.txt').exists())
        self.assertIn('denied or cancelled', client.submissions[1]['prompt'])
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')

    def test_cancel_wakes_approval_and_does_not_run_next_inference(self):
        client = self.make_agent([json.dumps({'tool': 'workspace', 'arguments': {
            'operation': 'write', 'path': 'new.txt', 'content': 'no'}})])
        self.agent.set_action_policy('ask')
        self.agent.submit(self.sid, 'Write new.txt', 'request-0001')
        self.wait_for(lambda: bool(self.agent.approvals))
        self.agent.cancel(self.sid)
        self.agent.start_thread.join(3)
        self.assertFalse(self.agent.active)
        self.assertEqual(self.store.session(self.sid)['status'], 'interrupted')
        self.assertEqual(len(client.submissions), 1)
        self.assertFalse((self.root / 'new.txt').exists())

    def test_cancel_owns_only_current_chat_job_and_later_turn_is_separate(self):
        client = self.make_agent(['A late answer.', 'A fresh answer.'])
        client.block = True
        self.agent.submit(self.sid, 'First request', 'request-0001')
        self.assertTrue(client.entered.wait(2))
        self.agent.cancel(self.sid)
        self.agent.start_thread.join(3)
        self.assertEqual(self.store.session(self.sid)['status'], 'interrupted')
        self.assertTrue(client.cancelled)
        self.assertTrue(all(job == 'job-0' for _, job in client.cancelled))
        client.block = False
        self.run_turn('Second request', 'request-0002')
        self.assertEqual(self.agent.last_answer, 'A fresh answer.')
        self.assertNotEqual(client.submissions[0]['owner'], client.submissions[1]['owner'])
        self.assertFalse(any(e['kind'] == 'assistant' and e['data']['text'] == 'A late answer.'
                             for e in self.store.events(self.sid)))

    def test_unknown_or_malformed_action_never_executes(self):
        for response in ('{"tool":"shell","arguments":{}}', '{"tool":"workspace",',
                         '{"tool":"workspace","arguments":{"operation":"read"},"extra":true}',
                         '{"tool":"workspace","tool":"workspace","arguments":{"operation":"read"}}'):
            with self.subTest(response=response):
                if self.agent:
                    self.agent.close()
                self.make_agent([response])
                with patch.object(self.agent, 'execute') as execute:
                    self.run_turn('Try a tool', 'request-' + str(time.monotonic_ns()))
                execute.assert_not_called()
                self.assertEqual(self.store.session(self.sid)['status'], 'failed')

    def test_truncated_tool_json_never_executes(self):
        self.make_agent([{'state': 'completed', 'result': '{"tool":"workspace"}', 'truncated': True}])
        with patch.object(self.agent, 'execute') as execute:
            self.run_turn()
        execute.assert_not_called()
        self.assertEqual(self.store.session(self.sid)['status'], 'failed')

    def test_prose_with_standalone_tool_request_is_not_a_final_answer(self):
        self.make_agent(['I need to inspect it.\n\n{"tool":"@npu","arguments":{"status":true}}'])
        with patch.object(self.agent, 'execute') as execute:
            self.run_turn('Tell me concrete device details')
        execute.assert_not_called()
        self.assertEqual(self.store.session(self.sid)['status'], 'failed')
        self.assertFalse(any(e['kind'] == 'assistant' for e in self.store.events(self.sid)))

    def test_whole_fenced_json_uses_same_validated_file_executor(self):
        (self.root / 'evidence.txt').write_text('Actual evidence: 73129')
        client = self.make_agent([
            '```json\n{"tool":"workspace","arguments":{"operation":"read","path":"evidence.txt"}}\n```',
            '```json\n{"final":"73129","evidence":["tool-1"]}\n```',
        ])
        self.run_turn('Read evidence.txt')
        self.assertEqual(self.agent.last_answer, '73129')
        self.assertIn('Actual evidence: 73129', client.submissions[1]['prompt'])

    def test_fenced_json_does_not_extract_calls_from_extra_prose_or_accept_fake_receipts(self):
        for response in (
                '```json\n{"tool":"workspace","arguments":{"operation":"read","path":"evidence.txt"}}\n```\nMore text',
                '```json\n{"final":"Invented evidence","evidence":["tool-8"]}\n```'):
            with self.subTest(response=response):
                if self.agent:
                    self.agent.close()
                self.make_agent([response])
                with patch.object(self.agent, 'execute') as execute:
                    self.run_turn('Check it', 'request-' + str(time.monotonic_ns()))
                execute.assert_not_called()
                self.assertEqual(self.store.session(self.sid)['status'], 'failed')

    def test_history_uses_recent_tail_and_prompt_stays_bounded(self):
        for number in range(170):
            self.store.event(self.sid, 'user', {'text': ('old-%s ' % number) + 'x' * 300})
            self.store.event(self.sid, 'assistant', {'text': ('answer-%s ' % number) + 'y' * 300})
        client = self.make_agent(['Short answer'])
        self.run_turn('Current request')
        prompt = client.submissions[0]['prompt']
        self.assertLessEqual(len(prompt), 6000)
        self.assertIn('answer-169', prompt)
        self.assertNotIn('old-0 ', prompt)
        self.assertEqual(prompt.count('CURRENT USER REQUEST:\nCurrent request'), 1)

    def test_large_current_request_is_saved_but_not_silently_truncated(self):
        client = self.make_agent(['unused'])
        self.run_turn('x' * 6000)
        self.assertEqual(self.store.session(self.sid)['status'], 'failed')
        self.assertEqual(client.submissions, [])

    def test_32_tool_budget_prevents_33rd_action(self):
        action = json.dumps({'tool': 'workspace', 'arguments': {'operation': 'list', 'path': '.'}})
        self.make_agent([action] * 33)
        self.run_turn('List repeatedly')
        self.assertEqual(self.agent.tool_calls, 32)
        self.assertEqual(self.store.session(self.sid)['status'], 'failed')

    def test_small_context_uses_compact_guide_and_profile_output_limit(self):
        client = self.make_agent(['A concise answer.'])
        client.context, client.max_output_tokens = 1280, 128
        self.run_turn('Hello')
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')
        prompt = client.submissions[0]['prompt']
        self.assertIn(COMPACT_GUIDE, prompt)
        self.assertNotIn('The same status/spawn/list/read/cancel forms', prompt)
        self.assertIn('CURRENT USER REQUEST:\nHello', prompt)
        self.assertLessEqual(len(prompt), (1280 - 128 - 192) * 2)
        self.assertEqual(client.submissions[0]['max_tokens'], 128)

    def test_1280_context_retains_real_tool_evidence_for_the_answer(self):
        (self.root / 'short.txt').write_text('Actual short evidence: birch')
        client = self.make_agent([
            '{"tool":"workspace","arguments":{"operation":"read","path":"short.txt"}}',
            '{"final":"The file says birch.","evidence":["tool-1"]}',
        ])
        client.context = 1280
        self.run_turn('Read short.txt')
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')
        self.assertIn('Actual short evidence: birch', client.submissions[1]['prompt'])
        self.assertIn('tool-1 · completed', client.submissions[1]['prompt'])
        self.assertLessEqual(len(client.submissions[1]['prompt']), (1280 - 256 - 192) * 2)
        self.assertEqual(client.submissions[1]['max_tokens'], 256)

    def test_multiple_tool_results_and_receipts_remain_grounded_in_current_turn(self):
        (self.root / 'first.txt').write_text('First actual evidence: cedar')
        (self.root / 'second.txt').write_text('Second actual evidence: maple')
        client = self.make_agent([
            json.dumps({'tool': 'workspace', 'arguments': {'operation': 'read', 'path': 'first.txt'}}),
            json.dumps({'tool': 'workspace', 'arguments': {'operation': 'read', 'path': 'second.txt'}}),
            json.dumps({'final': 'The files say cedar and maple.', 'evidence': ['tool-1', 'tool-2']}),
        ])
        events = self.run_turn('Compare first.txt and second.txt')
        prompt = client.submissions[2]['prompt']
        self.assertIn('First actual evidence: cedar', prompt)
        self.assertIn('Second actual evidence: maple', prompt)
        self.assertIn('tool-1 · completed', prompt)
        self.assertIn('tool-2 · completed', prompt)
        self.assertLessEqual(len(prompt), 6000)
        receipts = [event['data'] for event in events if event['kind'] == 'tool_result']
        self.assertEqual([item['receipt_id'] for item in receipts], ['tool-1', 'tool-2'])
        self.assertTrue(all(item['outcome'] == 'completed' for item in receipts))
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')

    def test_fabricated_or_invalid_evidence_references_never_become_a_final_answer(self):
        for response in ('{"final":"It ran.","evidence":["tool-9"]}',
                         '{"final":"It ran.","evidence":["tool-1"]}',
                         '{"final":"It ran.","evidence":"tool-1"}',
                         '{"final":"It ran.","evidence":["from-previous-turn"]}'):
            with self.subTest(response=response):
                if self.agent:
                    self.agent.close()
                self.make_agent([response])
                events = self.run_turn('What happened?', 'request-' + str(time.monotonic_ns()))
                self.assertEqual(self.store.session(self.sid)['status'], 'failed')
                self.assertFalse(any(event['kind'] == 'assistant' and event['data']['text'] == 'It ran.' for event in events))

    def test_npu_delegation_uses_reserved_route_and_reads_actual_helper_result(self):
        action = lambda argv: json.dumps({'tool': 'workspace', 'arguments': {'operation': 'command', 'argv': argv}})
        client = self.make_agent([
            action(['@npu', 'status']),
            action(['@npu', 'spawn', '{"task":"Review this plan","role":"reviewer","max_tokens":128}']),
            action(['@npu', 'read', 'npu-job', '35']),
            '{"final":"The reviewer suggests a test.","evidence":["tool-3"]}',
        ])
        self.agent.npu = Mock()
        self.agent.npu.dispatch.side_effect = [
            {'local': True, 'ready': True, 'model': {'backend': 'litert-npu'}},
            {'local': True, 'worker': {'id': 'npu-job', 'state': 'queued'}, 'execution_pending': True},
            {'local': True, 'id': 'npu-job', 'state': 'completed', 'result': 'Add a regression test.', 'exit_code': 0},
        ]
        # The parent executor owns the reserved dispatcher; this test verifies
        # the conversation route, independent of NPU hardware and processes.
        def execute(arguments):
            self.assertEqual(arguments['argv'][0], '@npu')
            return self.agent.npu.dispatch(self.agent, arguments, self.agent.workspace(self.sid))
        with patch.object(self.agent, 'execute', side_effect=execute):
            events = self.run_turn('Ask the NPU reviewer to check this plan')
        self.assertIn('tool-2 · pending', client.submissions[2]['prompt'])
        self.assertIn('Add a regression test.', client.submissions[3]['prompt'])
        self.assertEqual(self.agent.npu.dispatch.call_count, 3)
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')
        self.assertIsNone(self.agent.rpc)
        self.assertTrue(any(event['kind'] == 'tool_result' and event['data']['outcome'] == 'pending' for event in events))

    def test_returned_error_is_failed_evidence_without_a_success_claim(self):
        client = self.make_agent([
            '{"tool":"workspace","arguments":{"operation":"list","path":"."}}',
            '{"final":"The tool failed.","evidence":["tool-1"]}',
        ])
        with patch.object(self.agent, 'execute', return_value={'error': 'Actual host failure'}):
            events = self.run_turn('Inspect this directory')
        self.assertIn('tool-1 · failed', client.submissions[1]['prompt'])
        self.assertIn('Actual host failure', client.submissions[1]['prompt'])
        result = next(event['data'] for event in events if event['kind'] == 'tool_result')
        self.assertFalse(result['success'])

    def test_voice_followup_is_durable_ordered_and_sent_after_current_turn(self):
        client = self.make_agent(['First answer', 'Follow-up answer'])
        client.block = True
        self.agent.submit(self.sid, 'First', 'request-0001')
        self.assertTrue(client.entered.wait(2))
        session = SimpleNamespace(sid=self.sid, store=self.store, agent=self.agent)
        inbox = VoiceInbox(session)
        inbox.RETRY_SECONDS = 0
        try:
            speech_id = inbox.add('Follow-up speech')
            inbox.pump()
            self.wait_for(lambda: inbox._worker is not None and not inbox._worker.is_alive())
            inbox.pump()
            self.assertTrue(inbox.pending)
            self.assertEqual(len(client.submissions), 1)
            client.release.set()
            self.agent.start_thread.join(3)
            deadline = time.monotonic() + 3
            while inbox.pending and time.monotonic() < deadline:
                inbox.pump()
                time.sleep(.005)
            self.assertFalse(inbox.pending)
            self.agent.start_thread.join(3)
            self.assertEqual(len(client.submissions), 2)
            with self.store.lock:
                row = self.store.db.execute('SELECT prompt FROM requests WHERE id=?', (speech_id,)).fetchone()
            self.assertEqual(row['prompt'], 'Follow-up speech')
        finally:
            inbox.close()

    def test_close_joins_worker_before_database_can_close_without_late_events(self):
        client = self.make_agent(['Must never be delivered'])
        client.block = True
        self.agent.submit(self.sid, 'First', 'request-0001')
        self.assertTrue(client.entered.wait(2))
        self.agent.close()
        self.assertFalse(self.agent.start_thread.is_alive())
        snapshot = self.store.events(self.sid)
        time.sleep(.03)
        self.assertEqual(self.store.events(self.sid), snapshot)
        self.assertFalse(any(e['kind'] == 'assistant' for e in snapshot))

    def test_factory_preserves_explicit_rpc_fixtures_and_saved_engine(self):
        self.agent = create_agent(self.store, self.root)
        self.assertIsInstance(self.agent, LocalChatAgent)
        self.agent.close()
        rpc_factory = Mock()
        self.agent = create_agent(self.store, self.root, rpc_factory=rpc_factory)
        self.assertIs(type(self.agent), Agent)
        self.assertIs(self.agent.rpc_factory, rpc_factory)
        self.assertEqual(self.agent.engine_label, 'Codex')
        self.agent.close()
        self.store.setting('chat_engine', 'codex')
        self.agent = create_agent(self.store, self.root)
        self.assertIs(type(self.agent), Agent)
        self.agent.close()
        self.store.setting('chat_engine', 'unknown')
        with self.assertRaisesRegex(AppError, 'local or codex'):
            create_agent(self.store, self.root)

    def test_engine_label_names_the_actual_manifest_model(self):
        self.make_agent([])
        manifest = Path(self.temp.name) / 'model.json'
        manifest.write_text(json.dumps({'model_name': 'Huihui-gemma-4-E2B-it-abliterated-v2'}))
        with patch('local_chat.MANIFEST', manifest):
            self.assertEqual(self.agent.engine_label, 'Local Huihui-gemma-4-E2B-it-abliterated-v2')

    def test_provider_tool_identity_and_signature_survive_real_tool_feedback(self):
        (self.root / 'sample.txt').write_text('actual content')
        call = {'id': 'gemini-original-call', 'type': 'function',
                'function': {'name': 'workspace', 'arguments': '{"operation":"read","path":"sample.txt"}'},
                'extra_content': {'google': {'thought_signature': 'b3BhcXVl'}}}
        client = self.native_client([
            {'state': 'completed', 'result': '{"tool":"workspace","arguments":{"operation":"read","path":"sample.txt"}}',
             'native_call': call},
            'actual content',
        ])
        self.run_turn('Read sample.txt')
        messages = json.loads(client.submissions[1]['prompt'])['messages']
        self.assertEqual(messages[1]['tool_calls'], [call])
        self.assertEqual(messages[2]['tool_call_id'], call['id'])
        self.assertIn('actual content', messages[2]['content'])
        self.assertEqual(self.agent.last_answer, 'actual content')

    def test_provider_tool_metadata_mismatch_cannot_execute(self):
        call = {'id': 'gemini-call', 'type': 'function',
                'function': {'name': 'workspace', 'arguments': '{"operation":"delete","path":"sample.txt"}'}}
        self.native_client([{'state': 'completed',
                            'result': '{"tool":"workspace","arguments":{"operation":"read","path":"sample.txt"}}',
                            'native_call': call}])
        with patch.object(self.agent, '_tool') as execute:
            self.run_turn('Read sample.txt')
        execute.assert_not_called()
        self.assertEqual(self.store.session(self.sid)['status'], 'failed')

    def test_quota_wait_is_visible_once_and_completed_reply_is_delivered(self):
        client = self.native_client(['Ready after the wait.'])
        client.engine_name = 'gemini'
        original_read = client.read
        polls = []
        def read(owner, ident, wait, limit):
            polls.append(ident)
            if len(polls) <= 2:
                return {'state': 'running', 'stage': 'quota_wait', 'retry_after_seconds': 30}
            return original_read(owner, ident, wait, limit)
        client.read = read
        events = self.run_turn('Hello')
        notices = [event for event in events if event['kind'] == 'notice' and 'rate-limited' in event['data'].get('text', '')]
        self.assertEqual(len(notices), 1)
        self.assertIn('/cancel', notices[0]['data']['text'])
        self.assertEqual(self.agent.last_answer, 'Ready after the wait.')
        self.assertFalse(any(event['kind'] == 'tool' for event in events))

    def test_factory_selects_gemini_and_closes_its_client(self):
        self.store.setting('chat_engine', 'gemini')
        client = FakeClient([])
        client.engine_label = 'Gemini 3.8 Flash'
        client.close = Mock()
        with patch('gemini_client.GeminiClient', return_value=client):
            self.agent = create_agent(self.store, self.root)
        self.assertEqual(self.agent.engine_label, 'Gemini 3.8 Flash')
        self.assertIs(self.agent.local_client, client)
        self.agent.close()
        client.close.assert_called_once()
        self.agent = None

    def test_guide_ostadix_example_contains_real_nested_json(self):
        example = next(line for line in GUIDE.splitlines() if line.startswith('["@ostadix","call","o_execute"'))
        argv = json.loads(example)
        source = json.loads(argv[3])
        self.assertEqual(source['action'], 'execute')
        self.assertIn('__oval_result__ = 1 + 1', source['source'])
        self.assertIn('\n', source['source'])
        self.assertNotIn('\\n', source['source'])
        self.assertEqual(parse_response('{"final":"Ready"}'), (None, 'Ready'))

    def test_native_result_cannot_be_replaced_with_invented_scalar_or_completion(self):
        result = {'mcp_tool': 'o_execute', 'result': {'structuredContent': {'state': 'completed',
                  'result': {'ok': True, 'value': {'t': 'text', 'v': {'utf8': '7 * 9'}}}}}}
        self.assertEqual(native_answer('18', (result, True)), 'OSTADIX returned: 7 * 9')
        result['result']['structuredContent']['result']['value'] = {'t': 'number', 'v': {'kind': 'int', 'v': '63'}}
        self.assertEqual(native_answer('It returned 63.', (result, True)), 'It returned 63.')
        self.assertEqual(native_answer('163', (result, True)), 'OSTADIX returned: 63')
        result.update(execution_pending=True)
        self.assertIn('still running', native_answer('Done', (result, True)))
        self.assertIn('did not complete', native_answer('63', ({'error': 'Command not found'}, False)))


if __name__ == '__main__':
    unittest.main()
