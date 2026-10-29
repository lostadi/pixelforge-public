"""Offline PixelForge conversation transport and ordinary host-tool contracts."""
import io
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Store
from conversation_protocol import encode_conversation
from local_chat import LocalChatAgent, create_agent
from local_runtime import LocalError
from pixelforge_chat import BACKEND, COMMAND, PixelForgeClient, conversation_prompt, public_response


ROLE = 'phone conversation assistant'
TASK = encode_conversation([{'role': 'user', 'content': 'Read probe.txt'}])
SELECTED = ['blk.0.ffn_gate.weight', 'blk.0.ffn_up.weight', 'blk.0.ffn_down.weight']


def done(ident, text='A completed answer', **overrides):
    return dict({'type': 'done', 'id': ident, 'text': text, 'generated_tokens': [5, 6],
                 'prompt_tokens': 500, 'npu_weight_matmul_operations': 42,
                 'cpu_dynamic_matmul_operations': 7, 'token_limit_reached': False,
                 'generation_seconds': 1000.0, 'prefill_cpu': True}, **overrides)


class HostOutput:
    def __init__(self):
        self.frames = queue.Queue()
        self.closed = False
        self.push({'type': 'ready', 'model': {'backend': BACKEND}})

    def push(self, value):
        self.frames.put(json.dumps(value) + '\n')

    def readline(self, limit):
        return self.frames.get(timeout=5)

    def close(self):
        self.closed = True
        self.frames.put('')


class HostInput(io.StringIO):
    def __init__(self, host):
        super().__init__()
        self.host = host

    def write(self, text):
        super().write(text)
        request = json.loads(text)
        self.host.requests.append(request)
        self.host.received.set()
        self.host.respond(self.host, request)
        return len(text)

    def close(self):
        if not self.closed:
            self.host.stopped.set()
            self.host.stdout.close()
        # Preserve harmless requests for assertions.


class Host:
    def __init__(self, respond=None):
        self.requests = []
        self.received = threading.Event()
        self.stopped = threading.Event()
        self.stdout = HostOutput()
        self.stdin = HostInput(self)
        self.respond = respond or (lambda host, request: host.stdout.push(done(request['id'])))

    def poll(self):
        return 0 if self.stopped.is_set() else None

    def wait(self, timeout=None):
        if not self.stopped.wait(timeout):
            raise subprocess.TimeoutExpired('fixture-host', timeout)
        return 0

    def terminate(self):
        self.stdin.close()


class PixelForgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.metadata = self.root / 'installation.json'
        self.metadata.write_text(json.dumps({'backend': BACKEND, 'model': {
            'model_name': 'Huihui fixture', 'provenance_repo': 'huihui-ai/Huihui-fixture',
            'model_sha256': 'a' * 64}}))
        self.client = PixelForgeClient(metadata=self.metadata)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.client.close)

    def reply(self, host, **options):
        with patch('pixelforge_chat.subprocess.Popen', return_value=host) as popen:
            job = self.client.submit('owner-a', TASK, role=ROLE, **options)
            result = self.client.read('owner-a', job['id'], wait=2, limit=32768)
        self.assertIn(result['state'], ('failed', 'completed'))
        return result, popen

    def test_status_is_passive_and_does_not_claim_verified_worker(self):
        with patch('pixelforge_chat.subprocess.Popen') as popen:
            status = self.client.status()
        self.assertTrue(status['ready'])
        self.assertFalse(status['running'])
        self.assertFalse(status['verified_worker_ready'])
        self.assertEqual(status['model']['backend'], BACKEND)
        self.assertEqual(status['model']['context_per_worker'], 8192)
        self.assertNotIn('native_conversation_protocol', status['model'])
        self.assertIn('CPU prompt processing', status['execution'])
        popen.assert_not_called()

    def test_initial_welcome_label_uses_passive_model_metadata(self):
        with patch('pixelforge_chat.subprocess.Popen') as popen:
            client = PixelForgeClient(metadata=self.metadata)
        self.addCleanup(client.close)
        self.assertEqual(client.engine_label, 'Huihui fixture · PixelForge NPU + CPU')
        popen.assert_not_called()
        missing = PixelForgeClient(metadata=self.root / 'missing.json')
        self.addCleanup(missing.close)
        self.assertEqual(missing.engine_label, 'Huihui · PixelForge NPU + CPU')
        self.assertFalse(missing.status()['ready'])

    def test_missing_or_non_huihui_installation_is_unavailable(self):
        self.metadata.write_text(json.dumps({'backend': BACKEND, 'model': {
            'model_name': 'Google fixture', 'provenance_repo': 'google/gemma', 'model_sha256': 'a' * 64}}))
        self.assertFalse(self.client.status()['ready'])
        with self.assertRaisesRegex(LocalError, 'installation'):
            self.client.submit('owner-a', TASK, role=ROLE)

    def test_helper_cannot_acquire_main_tools_and_transcript_is_required(self):
        for task, role in ((TASK, 'assistant'), ('plain text', ROLE)):
            with self.subTest(role=role), self.assertRaises((LocalError, ValueError)):
                self.client.submit('owner-a', task, role=role)
        with self.assertRaisesRegex(LocalError, 'between 2 and 512'):
            self.client.submit('owner-a', TASK, role=ROLE, max_tokens=1)

    def selected_metadata(self):
        data = json.loads(self.metadata.read_text())
        data['model']['selected_npu_weight_tensors'] = SELECTED
        self.metadata.write_text(json.dumps(data))

    def test_selected_placement_status_and_actual_counters_are_preserved(self):
        self.selected_metadata()
        before = {'tpu_inference_count': '0 189012 52926'}
        after = {'tpu_inference_count': '0 189012 52929'}
        host = Host(lambda h, r: h.stdout.push(done(r['id'],
            selected_npu_weight_tensors=SELECTED, cpu_placed_weight_matmul_operations=150,
            cpu_prefill_weight_matmul_operations=153, report='/data/adb/ostadix-pixelforge/jobs/fixture',
            hardware_before=before, hardware_after=after)))
        status = self.client.status()
        self.assertIn('selected dense-weight projections on NPU', status['execution'])
        self.assertIn('remaining dense projections', status['execution'])
        self.assertEqual(status['model']['selected_npu_weight_tensors'], SELECTED)
        result, _ = self.reply(host)
        self.assertEqual(result['state'], 'completed')
        usage = result['usage']
        self.assertEqual(usage['selected_npu_weight_tensors'], SELECTED)
        self.assertEqual(usage['cpu_placed_weight_matmul_operations'], 150)
        self.assertEqual(usage['cpu_prefill_weight_matmul_operations'], 153)
        self.assertEqual(usage['hardware_before'], before)
        self.assertEqual(usage['hardware_after'], after)
        self.assertEqual(usage['report'], '/data/adb/ostadix-pixelforge/jobs/fixture')
        self.assertEqual(usage['provenance_repo'], 'huihui-ai/Huihui-fixture')

    def test_selected_placement_requires_matching_selection_and_cpu_counters(self):
        self.selected_metadata()
        cases = [dict(selected_npu_weight_tensors=SELECTED),
                 dict(selected_npu_weight_tensors=SELECTED[:1], cpu_placed_weight_matmul_operations=1,
                      cpu_prefill_weight_matmul_operations=10),
                 dict(selected_npu_weight_tensors=SELECTED, cpu_placed_weight_matmul_operations=True,
                      cpu_prefill_weight_matmul_operations=10)]
        for override in cases:
            with self.subTest(override=override):
                host = Host(lambda h, r: h.stdout.push(done(r['id'], **override)))
                result, _ = self.reply(host)
                self.assertEqual(result['state'], 'failed')
                self.assertEqual(result['result'], '')

    def test_invalid_model_report_and_hardware_evidence_are_rejected(self):
        cases = [dict(model={'model_sha256': 'b' * 64}), dict(report='relative/report'),
                 dict(hardware_before='raw-string'), dict(hardware_after={'arbitrary_data': 'ignored'}),
                 dict(cpu_prefill_weight_matmul_operations=-1)]
        for override in cases:
            with self.subTest(override=override):
                host = Host(lambda h, r: h.stdout.push(done(r['id'], **override)))
                result, _ = self.reply(host)
                self.assertEqual(result['state'], 'failed')
                self.assertEqual(result['result'], '')

    def test_prompt_preserves_actual_receipts_without_provider_signatures(self):
        call = {'id': 'call-a', 'type': 'function', 'function': {'name': 'workspace',
                'arguments': '{"operation":"read","path":"probe.txt"}'},
                'extra_content': {'google': {'thought_signature': 'c2lnbmF0dXJl'}}}
        task = encode_conversation([{'role': 'user', 'content': 'Read probe.txt'},
            {'role': 'assistant', 'content': None, 'tool_calls': [call]},
            {'role': 'tool', 'tool_call_id': 'call-a', 'content': 'Actual receipt 42'}])
        prompt = conversation_prompt(task, ROLE, 'Use valid JSON.')
        self.assertIn('Actual receipt 42', prompt)
        self.assertIn('COMPLETED TOOL REQUEST call-a', prompt)
        self.assertIn('HOST FORMAT CORRECTION', prompt)
        self.assertNotIn('thought_signature', prompt)
        self.assertIn('thought_signature', task)

    def test_completed_request_has_real_npu_evidence_and_fixed_stdin_transport(self):
        host = Host()
        result, popen = self.reply(host)
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['result'], 'A completed answer')
        self.assertEqual(result['usage']['npu_weight_matmul_operations'], 42)
        self.assertEqual(result['usage']['generation_seconds'], 1000.0)
        self.assertEqual(result['partial'], '')
        self.assertTrue(self.client.status()['verified_worker_ready'])
        self.assertEqual(popen.call_args.args[0], COMMAND)
        self.assertEqual(popen.call_args.kwargs['stderr'], subprocess.DEVNULL)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        request = host.requests[0]
        self.assertIn('Read probe.txt', request['prompt'])
        self.assertEqual(request['context_size'], 8192)
        self.assertTrue(request['prefill_cpu'])
        self.assertNotIn('timeout', request)

    def test_consecutive_turns_reuse_one_resident_host_and_owner_isolation(self):
        host = Host()
        with patch('pixelforge_chat.subprocess.Popen', return_value=host) as popen:
            for owner in ('owner-a', 'owner-b'):
                job = self.client.submit(owner, TASK, role=ROLE)
                result = self.client.read(owner, job['id'], wait=2)
                self.assertEqual(result['state'], 'completed')
                with self.assertRaises(LocalError):
                    self.client.read('intruder', job['id'])
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(len(host.requests), 2)

    def test_zero_npu_ops_wrong_id_or_invalid_evidence_cannot_complete(self):
        cases = [dict(npu_weight_matmul_operations=0), dict(id='wrong-request'),
                 dict(prefill_cpu=False), dict(generation_seconds=float('nan')),
                 dict(generated_tokens=[True]), dict(prompt_tokens=9000)]
        for override in cases:
            with self.subTest(override=override):
                host = Host(lambda h, r: h.stdout.push(done(r['id'], **override)))
                result, _ = self.reply(host)
                self.assertEqual(result['state'], 'failed')
                self.assertEqual(result['result'], '')
                self.assertFalse(self.client.status()['verified_worker_ready'])

    def test_native_thought_or_tool_markers_never_become_public_answers(self):
        for text in ('<think>private</think>', '<|channel>analysis\nprivate',
                     '<|tool_call>workspace{path:"probe.txt"}'):
            with self.subTest(text=text), self.assertRaises(LocalError):
                public_response(text)
        self.assertEqual(public_response('<|channel>analysis\nprivate<|channel>final\nPublic answer<end_of_turn>'),
                         'Public answer')

    def test_token_limit_never_executes_partial_json_but_marks_plain_answer(self):
        host = Host(lambda h, r: h.stdout.push(done(r['id'], '{"tool":"workspace"', token_limit_reached=True)))
        result, _ = self.reply(host)
        self.assertEqual(result['state'], 'failed')
        host = Host(lambda h, r: h.stdout.push(done(r['id'], 'An incomplete explanation', token_limit_reached=True)))
        result, _ = self.reply(host)
        self.assertTrue(result['answer_incomplete'])
        self.assertIn('Answer incomplete', json.loads(result['result'])['final'])

    def test_streamed_raw_content_stays_private_and_cancel_closes_host_stdin(self):
        host = Host(lambda h, r: h.stdout.push({'type': 'chunk', 'id': r['id'], 'text': '<|channel>analysis private'}))
        with patch('pixelforge_chat.subprocess.Popen', return_value=host) as popen:
            job = self.client.submit('owner-a', TASK, role=ROLE)
            self.assertTrue(host.received.wait(1))
            snapshot = self.client.read('owner-a', job['id'])
            self.assertEqual(snapshot['partial'], '')
            self.client.cancel('owner-a', job['id'])
            self.assertTrue(host.stopped.is_set())
            result = self.client.read('owner-a', job['id'])
            self.assertEqual(result['state'], 'cancelled')
            self.assertEqual(result['result'], '')
            self.assertEqual(popen.call_count, 1)

    def test_cancelling_queued_owner_does_not_stop_other_active_owner(self):
        host = Host(lambda h, r: None)
        with patch('pixelforge_chat.subprocess.Popen', return_value=host):
            active = self.client.submit('owner-a', TASK, role=ROLE)
            self.assertTrue(host.received.wait(1))
            queued = self.client.submit('owner-b', TASK, role=ROLE)
            self.client.cancel('owner-b', queued['id'])
            self.assertFalse(host.stopped.is_set())
            self.assertEqual(self.client.read('owner-a', active['id'])['state'], 'running')
            self.client.cancel('owner-a', active['id'])

    def test_real_host_file_tool_result_returns_to_model_without_root_tool_execution(self):
        project = self.root / 'project'; project.mkdir()
        (project / 'probe.txt').write_text('Actual file result: BERYL-42')
        store = Store(self.root / 'state')
        sid = store.create(project)
        replies = iter(['{"tool":"workspace","arguments":{"operation":"read","path":"probe.txt"}}',
                        'Actual file result: BERYL-42'])
        host = Host(lambda h, r: h.stdout.push(done(r['id'], next(replies))))
        agent = LocalChatAgent(store, project, client=self.client)
        agent.local.cancel = Mock()
        try:
            with patch('pixelforge_chat.subprocess.Popen', return_value=host):
                agent.submit(sid, 'Read probe.txt and report its content.', 'fixture-request')
                agent.start_thread.join(4)
            self.assertFalse(agent.start_thread.is_alive())
            self.assertEqual(store.session(sid)['status'], 'idle')
            self.assertEqual(agent.last_answer, 'Actual file result: BERYL-42')
            self.assertEqual(len(host.requests), 2)
            self.assertIn('Actual file result: BERYL-42', host.requests[1]['prompt'])
            events = store.events(sid)
            self.assertTrue(any(e['kind'] == 'tool_result' and e['data']['success'] for e in events))
        finally:
            agent.close()
            store.db.close()

    def test_saved_pixelforge_engine_uses_the_new_client(self):
        project = self.root / 'project'; project.mkdir()
        store = Store(self.root / 'state')
        store.setting('chat_engine', 'pixelforge')
        with patch('pixelforge_chat.PixelForgeClient', return_value=self.client):
            agent = create_agent(store, project)
        try:
            self.assertIs(agent.local_client, self.client)
        finally:
            agent.close()
            store.db.close()


if __name__ == '__main__':
    unittest.main()
