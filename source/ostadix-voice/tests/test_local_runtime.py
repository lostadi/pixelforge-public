"""Offline resident worker tests: fake text engine and private loopback HTTP only."""
import http.server
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sqlite3
import struct
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_runtime import CONVERSATION_TOOLS, LIMITS, LlamaEngine, LocalClient, LocalError, LocalRequestNotSent, LocalRuntime, _dispatch, _interrupt, _model_options, selected_engine, serve, native_tool_response
from conversation_protocol import encode_conversation, NATIVE_GUIDE


def eventually(check, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for local worker fixture')


class FakeEngine:
    def __init__(self, state):
        self.model = {'name': 'offline-fixture'}
        self.calls = []
        self.gate = threading.Event()
        self.alive = True
        self.closed = False
    def start(self):
        return self.model
    def is_alive(self):
        return self.alive
    def run(self, task, role, max_tokens, cancelled, register):
        self.calls.append((task, role, max_tokens))
        while not self.gate.wait(.01):
            if cancelled.is_set():
                return '', {}, False
        return task, {'completion_tokens': 3}, False
    def close(self):
        self.closed = True
        self.alive = False
        self.gate.set()


class ModelOptionTests(unittest.TestCase):
    def test_native_ostadix_arguments_are_encoded_once_and_use_existing_dispatch(self):
        source = 'python^(\n__oval_result__ = "quoted value"\n)_python\n'
        call = json.loads(native_tool_response('ostadix_execute', json.dumps({'source': source})))
        self.assertEqual(call['arguments']['argv'][:3], ['@ostadix', 'call', 'o_execute'])
        self.assertEqual(json.loads(call['arguments']['argv'][3]), {'source': source, 'action': 'execute'})
        for args in ('{"source":"1","source":"2"}', '{"source":"1","path":"other.O"}',
                     '{"source":"1","action":"compile"}', '{"source":', '[]'):
            with self.subTest(args=args), self.assertRaises(LocalError):
                native_tool_response('ostadix_execute', args)

    def test_numeric_expressions_use_real_ostadix_backend_without_interpreting_other_source(self):
        for source in ('7 * 9', '-3 + 6 / 2', '10 % 3'):
            call = json.loads(native_tool_response('ostadix_execute', json.dumps({'source': source})))
            program = json.loads(call['arguments']['argv'][3])['source']
            self.assertEqual(program, 'python^(\n__oval_result__ = (' + source + ')\n)_python\n')
        for source in ('print("side effect")', '__import__("os").system("id")', '"7 * 9"', 'True', 'x + 1'):
            call = json.loads(native_tool_response('ostadix_execute', json.dumps({'source': source})))
            self.assertEqual(json.loads(call['arguments']['argv'][3])['source'], source)

    def test_typed_phone_function_uses_existing_phone_executor_and_explicit_fields(self):
        for request, argv in [({'action': 'volume'}, ['@phone', 'volume']),
                              ({'action': 'cpu'}, ['@phone', 'cpu']),
                              ({'action': 'npu'}, ['@phone', 'npu']),
                              ({'action': 'volume', 'level': 3}, ['@phone', 'volume', '3']),
                              ({'action': 'apps', 'query': 'calculator'}, ['@phone', 'apps', 'calculator']),
                              ({'action': 'open', 'package': 'com.example.app'}, ['@phone', 'open', 'com.example.app']),
                              ({'action': 'settings', 'section': 'audio'}, ['@phone', 'settings', 'audio'])]:
            call = json.loads(native_tool_response('phone_control', json.dumps(request)))
            self.assertEqual(call['arguments'], {'operation': 'command', 'argv': argv})
        for request in ({'action': 'volume', 'level': True}, {'action': 'screen', 'level': 4},
                        {'action': 'cpu', 'query': 'model'}, {'action': 'cpu', 'level': 4},
                        {'action': 'open'}, {'action': 'shell', 'query': 'id'}, {'action': []}):
            with self.subTest(request=request), self.assertRaises(LocalError):
                native_tool_response('phone_control', json.dumps(request))

    def test_native_phone_schema_exposes_read_only_cpu_hardware_query(self):
        function = next(tool['function'] for tool in CONVERSATION_TOOLS
                        if tool['function']['name'] == 'phone_control')
        self.assertIn('cpu', function['parameters']['properties']['action']['enum'])
        self.assertIn('npu', function['parameters']['properties']['action']['enum'])
        self.assertIn('CPU/chipset hardware facts', function['description'])

    def test_large_native_call_survives_statistics_budget_without_duplicate_job_payload(self):
        arguments = json.dumps({'operation': 'write', 'path': 'fixture.txt', 'content': '雪' * 6000},
                               ensure_ascii=False)
        call = {'id': 'provider-large-call', 'type': 'function',
                'function': {'name': 'workspace', 'arguments': arguments}}
        class NativeEngine(FakeEngine):
            def run(self, *_):
                return (native_tool_response('workspace', arguments),
                        {'oversized_stats': 'x' * 5000, 'native_call': call,
                         'native_call_id_source': 'provider'}, False)
        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalRuntime(Path(directory) / 'state', NativeEngine)
            try:
                eventually(lambda: runtime.status()['ready'])
                job = runtime.submit('fixture', 'Native metadata test', role='phone conversation assistant')
                result = runtime.read('fixture', job['id'], wait=3, limit=65536)
                self.assertEqual(result['state'], 'completed')
                self.assertEqual(result['native_call'], call)
                self.assertEqual(result['native_call_id_source'], 'provider')
                self.assertEqual(result['usage'], {})
                self.assertLess(len(json.dumps(result).encode()), 262144)
                short = runtime.read('fixture', job['id'], limit=20)
                self.assertTrue(short['truncated'])
                self.assertNotIn('native_call', short)
                self.assertNotIn('native_call', short['usage'])
            finally:
                runtime.close()

    def test_backend_selection_is_explicit_and_does_not_fall_back_from_npu(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'profile.json'
            with patch('local_runtime.MANIFEST', manifest):
                manifest.write_text('{}')
                self.assertIsInstance(selected_engine(Path(directory)), LlamaEngine)
                factory = Mock()
                with patch.dict(sys.modules, {'npu_runtime': types.SimpleNamespace(LiteRTEngine=factory)}):
                    manifest.write_text('{"backend":"litert-npu"}')
                    self.assertIs(selected_engine(Path(directory)), factory.return_value)
                    factory.assert_called_once_with(Path(directory), manifest_path=manifest)
                    factory.side_effect = LocalError('NPU unavailable')
                    with self.assertRaisesRegex(LocalError, 'NPU unavailable'):
                        selected_engine(Path(directory))
                manifest.write_text('{"backend":"unknown"}')
                with self.assertRaisesRegex(LocalError, 'Unsupported local model backend'):
                    selected_engine(Path(directory))

    def test_single_slot_engine_queues_second_request_and_reports_own_limit(self):
        class SerialEngine(FakeEngine):
            def __init__(self, state):
                super().__init__(state)
                self.model.update(parallel=1, max_output_tokens=64)
        with tempfile.TemporaryDirectory() as directory:
            runtime = LocalRuntime(Path(directory) / 'state', SerialEngine)
            try:
                eventually(lambda: runtime.ready)
                first = runtime.submit('a', 'first', 64)
                eventually(lambda: runtime.status()['active'] == 1)
                second = runtime.submit('a', 'second', 64)
                self.assertEqual(runtime.status()['queued'], 1)
                self.assertEqual(runtime.status()['limits']['active'], 1)
                self.assertEqual(runtime.status()['limits']['max_tokens'], 64)
                with self.assertRaisesRegex(LocalError, 'max_tokens'):
                    runtime.submit('a', 'too long', 65)
                runtime.engine.gate.set()
                eventually(lambda: all(job['state'] == 'completed' for job in runtime.jobs('a')))
                self.assertEqual([call[0] for call in runtime.engine.calls], ['first', 'second'])
            finally:
                runtime.close()

    def test_family_specific_sampling_preserves_top_k_and_presence_penalty(self):
        sampling = {'temperature': 1.0, 'top_p': 1.0, 'top_k': 20,
                    'min_p': 0.0, 'presence_penalty': 2.0, 'repeat_penalty': 1.0}
        self.assertEqual(_model_options({'sampling': sampling}), (sampling, {}))
        for field, value in [('top_k', True), ('top_k', 1.5), ('top_k', -1),
                             ('top_k', 1001), ('presence_penalty', 2.1),
                             ('presence_penalty', -2.1), ('presence_penalty', float('nan'))]:
            with self.subTest(field=field, value=value), self.assertRaises(LocalError):
                _model_options({'sampling': {field: value}})

    def test_absent_profile_preserves_lfm_sampling(self):
        self.assertEqual(_model_options({}), ({'temperature': .4}, {}))
    def test_minicpm_profile_requires_nonthinking_and_disabled_min_p(self):
        profile = {'provenance_repo': 'huihui-ai/Huihui-MiniCPM5-2B-abliterated',
                   'sampling': {'temperature': 1.0, 'top_p': .95, 'min_p': 0.0, 'repeat_penalty': 1.05},
                   'chat_template_kwargs': {'enable_thinking': False}}
        self.assertEqual(_model_options(profile), (profile['sampling'], profile['chat_template_kwargs']))
        for changes in [{'sampling': {'temperature': 1.0}}, {'sampling': {'min_p': .05}},
                        {'chat_template_kwargs': {}}, {'chat_template_kwargs': {'enable_thinking': True}}]:
            with self.subTest(changes=changes), self.assertRaisesRegex(LocalError, 'MiniCPM5 workers require'):
                _model_options({**profile, **changes})
    def test_profile_rejects_unknown_nonfinite_and_unbounded_options(self):
        for value in [{'sampling': None}, {'sampling': {'endpoint': 'http://other'}},
                      {'sampling': {'temperature': True}}, {'sampling': {'temperature': float('nan')}},
                      {'sampling': {'temperature': float('inf')}}, {'sampling': {'temperature': 2.01}},
                      {'sampling': {'temperature': -.1}}, {'sampling': {'top_p': 0}},
                      {'sampling': {'top_p': 1.1}}, {'sampling': {'min_p': -.1}},
                      {'sampling': {'min_p': 1.1}}, {'sampling': {'repeat_penalty': .9}},
                      {'sampling': {'repeat_penalty': 2.1}}, {'chat_template_kwargs': None},
                      {'chat_template_kwargs': {'enable_thinking': 'false'}},
                      {'chat_template_kwargs': {'enable_thinking': 0}},
                      {'chat_template_kwargs': {'tools': []}}]:
            with self.subTest(value=value), self.assertRaises(LocalError):
                _model_options(value)


class LlamaStartupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        runtime = self.root / 'runtime'
        runtime.mkdir()
        self.server = runtime / 'llama-server'
        self.server.write_text('#!/bin/sh\n')
        self.model = runtime / 'model.gguf'
        content = b'GGUF offline startup fixture'
        self.model.write_bytes(content)
        self.manifest_path = runtime / 'local-model.json'
        self.manifest = {
            'server_path': str(self.server), 'model_path': str(self.model),
            'model_name': 'offline-fixture', 'model_bytes': len(content),
            'model_sha256': hashlib.sha256(content).hexdigest(),
            'model_repo': 'fixture/pinned-GGUF', 'model_revision': 'a' * 40,
            'source_commit': 'b' * 40, 'quantization': 'Q4_K_M',
            'provenance_repo': 'huihui-ai/offline-abliterated',
            'provenance_revision': 'c' * 40, 'threads': 2, 'parallel': 2,
            'context_per_worker': 2048,
        }
        state = self.root / 'state'
        state.mkdir()
        self.engine = LlamaEngine(state)
        self.addCleanup(self.close_engine)
        for name, value in [('ROOT', self.root), ('MANIFEST', self.manifest_path)]:
            replacement = patch('local_runtime.' + name, value)
            replacement.start()
            self.addCleanup(replacement.stop)
        self.spawn_patch = patch('local_runtime.subprocess.Popen')
        self.spawn = self.spawn_patch.start()
        self.addCleanup(self.spawn_patch.stop)
        self.spawn.return_value.poll.return_value = None
        response = io.BytesIO(b'{}')
        response.status = 200
        self.health_patch = patch('local_runtime.urllib.request.build_opener')
        self.health = self.health_patch.start()
        self.addCleanup(self.health_patch.stop)
        self.health.return_value.open.return_value = response

    def close_engine(self):
        # The mocked process never exists; close private fixture files only.
        self.engine.process = None
        self.engine.close()

    def start(self, **changes):
        self.manifest_path.write_text(json.dumps({**self.manifest, **changes}))
        return self.engine.start()

    def engine_command(self):
        command = self.spawn.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[2:5], ['--engine-parent', str(os.getpid()), '--engine-exec'])
        return command[5:]

    def test_32k_context_is_accepted_and_passed_to_server(self):
        model = self.start(context_per_worker=32768, parallel=1)
        self.assertEqual(model['context_per_worker'], 32768)
        command = self.engine_command()
        self.assertEqual(command[command.index('--ctx-size') + 1], '32768')
        self.assertEqual(command[command.index('--parallel') + 1], '1')

    def test_context_above_32k_and_boolean_are_rejected_before_spawn(self):
        for context in (32769, True):
            with self.subTest(context=context), self.assertRaisesRegex(LocalError, 'context_per_worker'):
                self.start(context_per_worker=context)
        self.spawn.assert_not_called()
        self.health.assert_not_called()

    def test_q8_cache_flags_enable_flash_attention_for_quantized_values(self):
        model = self.start(cache_type_k='q8_0', cache_type_v='q8_0')
        self.assertEqual((model['cache_type_k'], model['cache_type_v']), ('q8_0', 'q8_0'))
        self.assertEqual(self.engine_command()[-6:], [
            '--cache-type-k', 'q8_0', '--cache-type-v', 'q8_0', '--flash-attn', 'on'])

    def test_unknown_cache_types_are_rejected_before_spawn(self):
        for key in ('cache_type_k', 'cache_type_v'):
            for value in ('q4_0', 'Q8_0', None, True):
                with self.subTest(key=key, value=value), self.assertRaisesRegex(LocalError, 'KV cache types'):
                    self.start(**{key: value})
        self.spawn.assert_not_called()
        self.health.assert_not_called()

    def test_manifest_without_cache_fields_preserves_existing_server_command(self):
        model = self.start()
        self.assertNotIn('cache_type_k', model)
        self.assertNotIn('cache_type_v', model)
        self.assertEqual(self.engine_command(), [
            str(self.server), '--model', str(self.model), '--host', '127.0.0.1',
            '--port', str(self.engine.port), '--api-key-file', str(self.engine.state / 'engine.key'),
            '--parallel', '2', '--ctx-size', '4096', '--threads', '2', '--threads-batch', '4',
            '--no-webui', '--no-context-shift', '--log-verbosity', '1'])


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'state'
        self.runtime = LocalRuntime(self.path, FakeEngine)
        eventually(lambda: self.runtime.ready)
    def tearDown(self):
        self.runtime.close()
        self.temp.cleanup()
    def test_idle_engine_never_generates_and_status_does_not_submit(self):
        self.assertTrue(self.runtime.status()['ready'])
        self.assertEqual(self.runtime.engine.calls, [])
        self.assertEqual(self.runtime.jobs('conversation'), [])
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.path / 'jobs.sqlite3').stat().st_mode & 0o777, 0o600)

    def test_only_helpers_receive_automatic_worker_deadlines(self):
        with patch.dict(LIMITS, timeout_seconds=.05):
            main = self.runtime.submit('main', 'Wait for cancellation', role='phone conversation assistant')
            helper = self.runtime.submit('helper', 'Wait for helper deadline', role='assistant')
            eventually(lambda: self.runtime.read('helper', helper['id'])['state'] == 'failed')
            self.assertEqual(self.runtime.read('main', main['id'])['state'], 'running')
            self.assertIn('time limit', self.runtime.read('helper', helper['id'])['error'])
            self.runtime.cancel('main', main['id'])
            eventually(lambda: self.runtime.status()['active'] == 0)
            self.assertEqual(self.runtime.read('main', main['id'])['state'], 'cancelled')

    def test_streamed_main_preview_is_owner_scoped_and_cancelled_without_late_output(self):
        def streamed(task, role, tokens, cancel, register, update):
            update('Early answer')
            result = self.runtime.engine.run(task, role, tokens, cancel, register)
            update(' Late cancelled text')
            return result
        self.runtime.engine.run_with_updates = streamed
        job = self.runtime.submit('a', 'Hello', role='phone conversation assistant')
        eventually(lambda: self.runtime.read('a', job['id']).get('partial') == 'Early answer')
        with self.assertRaises(LocalError):
            self.runtime.read('b', job['id'])
        self.runtime.cancel('a', job['id'])
        eventually(lambda: self.runtime.status()['active'] == 0)
        result = self.runtime.read('a', job['id'])
        self.assertEqual(result['state'], 'cancelled')
        self.assertNotIn('partial', result)
        self.assertEqual(result['usage'], {})

    def test_explicit_timeout_persists_known_stage_and_counts_without_replaying_or_saving_preview(self):
        for phase, stage, tokens, preview in (
                ('preflight', 'apply_template', None, ''),
                ('preflight', 'tokenize', None, ''),
                ('generation', 'waiting_for_output', 3391, ''),
                ('generation', 'streaming_output', 3391, 'éarly')):
            observed = threading.Event()
            connection = Mock()
            calls = []
            def streamed(task, role, limit, cancelled, register, update):
                calls.append(task)
                register(connection)
                register.observe(phase=phase, stage=stage, prompt_tokens=tokens,
                                 output_bytes=len(preview.encode()) if preview else None)
                if preview:
                    update(preview)
                observed.set()
                cancelled.wait(2)
                update(' Late output must not survive timeout')
                return 'Late final must not become a completed response', {'completion_tokens': 9}, False
            self.runtime.engine.run_with_updates = streamed
            with self.subTest(stage=stage), patch('local_runtime.threading.Timer') as timer:
                job = self.runtime.submit('a', 'private fixture prompt', role='phone conversation assistant')
                self.assertTrue(observed.wait(1))
                # No scheduled timeout owns main conversations. Explicitly
                # exercise the diagnostic handler below without a timer.
                timer.assert_not_called()
                with self.runtime.condition:
                    handle = self.runtime.active[job['id']]
                    handle['started_monotonic'] -= 7
                self.runtime._timeout(job['id'], handle)
                eventually(lambda: self.runtime.status()['active'] == 0)
                result = self.runtime.read('a', job['id'])
                telemetry = result['usage']['runtime_observation']
                self.assertEqual(result['runtime_observation'], telemetry)
                listed = next(item for item in self.runtime.jobs('a') if item['id'] == job['id'])
                self.assertEqual(listed['runtime_observation'], telemetry)
                self.assertEqual(result['state'], 'failed')
                self.assertEqual(result['error'], 'Local worker exceeded its 120-second time limit')
                self.assertEqual((telemetry['phase'], telemetry['stage']), (phase, stage))
                self.assertEqual(telemetry.get('prompt_tokens'), tokens)
                self.assertEqual(telemetry['preview_chars'], len(preview))
                self.assertEqual(telemetry['preview_bytes'], len(preview.encode()))
                self.assertGreaterEqual(telemetry['elapsed_seconds'], 7)
                self.assertEqual('first_output_seconds' in telemetry, bool(preview))
                self.assertTrue(handle['cancel'].is_set())
                self.assertEqual(calls, ['private fixture prompt'])
                self.assertEqual(result['result'], '')
                self.assertNotIn('partial', result)
                self.assertNotIn('private fixture prompt', json.dumps(telemetry))
                if preview:
                    self.assertNotIn(preview, json.dumps(telemetry, ensure_ascii=False))
                db = sqlite3.connect(self.path / 'jobs.sqlite3')
                try:
                    persisted = json.loads(db.execute('SELECT usage FROM jobs WHERE id=?', (job['id'],)).fetchone()[0])
                finally:
                    db.close()
                self.assertEqual(persisted, result['usage'])
                with self.assertRaises(LocalError):
                    self.runtime.read('another-owner', job['id'])

    def test_engine_failure_also_retains_its_last_known_stage(self):
        def broken(task, role, tokens, cancel, register):
            register.observe(phase='preflight', stage='tokenize', prompt_tokens=2000)
            raise LocalError('Fixture preprocessing failed')
        self.runtime.engine.run = broken
        job = self.runtime.submit('a', 'private fixture prompt')
        result = self.runtime.read('a', job['id'], wait=2)
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['usage']['runtime_observation']['stage'], 'tokenize')
        self.assertEqual(result['usage']['runtime_observation']['prompt_tokens'], 2000)

    def test_two_active_eight_queued_and_one_engine(self):
        self.runtime.submit('a', 'first')
        self.runtime.submit('b', 'second')
        eventually(lambda: self.runtime.status()['active'] == 2)
        for number in range(8):
            self.runtime.submit('a', f'queued-{number}')
        with self.assertRaisesRegex(LocalError, 'queue is full'):
            self.runtime.submit('a', 'overflow')
        self.assertEqual(self.runtime.status()['active'], 2)
        self.assertEqual(self.runtime.status()['queued'], 8)
        self.assertEqual(len(self.runtime.engine.calls), 2)
        self.runtime.engine.gate.set()
        eventually(lambda: all(job['state'] == 'completed' for owner in ('a', 'b') for job in self.runtime.jobs(owner)))
        self.assertEqual(len(self.runtime.engine.calls), 10)
    def test_results_owner_isolation_usage_and_explicit_truncation(self):
        self.runtime.engine.gate.set()
        job = self.runtime.submit('a', 'é' * 100, max_tokens=64, role='reviewer')
        result = self.runtime.read('a', job['id'], wait=2, limit=11)
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['result'], 'é' * 5)
        self.assertTrue(result['truncated'])
        self.assertTrue(result['result_truncated'])
        self.assertEqual(result['result_bytes'], 200)
        self.assertEqual(result['usage']['completion_tokens'], 3)
        self.assertEqual(self.runtime.jobs('b'), [])
        with self.assertRaisesRegex(LocalError, 'Unknown local job'):
            self.runtime.read('b', job['id'])
    def test_cancel_owner_only_stops_running_and_queued_without_replay(self):
        first = self.runtime.submit('a', 'running-a')
        self.runtime.submit('b', 'running-b')
        eventually(lambda: self.runtime.status()['active'] == 2)
        queued = self.runtime.submit('a', 'queued-a')
        result = self.runtime.cancel('a')
        self.assertEqual(result['cancelled'], 2)
        eventually(lambda: self.runtime.status()['active'] == 1)
        self.assertEqual(self.runtime.read('a', first['id'])['state'], 'cancelled')
        self.assertEqual(self.runtime.read('a', queued['id'])['state'], 'cancelled')
        self.assertEqual(self.runtime.jobs('b')[0]['state'], 'running')
        self.assertEqual(len(self.runtime.engine.calls), 2)
    def test_restart_marks_unfinished_interrupted_and_preserves_completed(self):
        self.runtime.engine.gate.set()
        completed = self.runtime.submit('a', 'saved result')
        self.assertEqual(self.runtime.read('a', completed['id'], wait=2)['state'], 'completed')
        stale = self.runtime.submit('a', 'must never replay')
        self.runtime.close()
        with sqlite3.connect(self.path / 'jobs.sqlite3') as db:
            db.execute("UPDATE jobs SET state='running' WHERE id=?", (stale['id'],))
        self.runtime = LocalRuntime(self.path, FakeEngine)
        eventually(lambda: self.runtime.ready)
        self.assertEqual(self.runtime.read('a', stale['id'])['state'], 'interrupted')
        self.assertIn('not replayed', self.runtime.read('a', stale['id'])['error'])
        self.assertEqual(self.runtime.read('a', completed['id'])['result'], 'saved result')
        self.assertEqual(self.runtime.engine.calls, [])
    def test_status_detects_dead_engine_before_loader_checks_it(self):
        with self.runtime.condition:
            self.assertTrue(self.runtime.status()['ready'])
            self.runtime.engine.alive = False
            # Hold the loader out to exercise the interval before it notices.
            self.assertTrue(self.runtime.ready)
            status = self.runtime.status()
            self.assertFalse(status['ready'])
            self.assertIn('server stopped', status['error'])
            self.assertIn('not replayed', status['error'])
            self.assertEqual(self.runtime.engine.calls, [])
        eventually(lambda: self.runtime.error is not None)

    def test_engine_crash_fails_waiting_and_running_without_recovery_loop(self):
        self.runtime.submit('a', 'first')
        self.runtime.submit('a', 'second')
        eventually(lambda: self.runtime.status()['active'] == 2)
        self.runtime.submit('a', 'queued')
        self.runtime.engine.alive = False
        eventually(lambda: self.runtime.error is not None)
        self.assertFalse(self.runtime.status()['ready'])
        self.assertTrue(all(job['state'] == 'failed' for job in self.runtime.jobs('a')))
        with self.assertRaises(LocalError):
            self.runtime.submit('a', 'do not restart')
        self.assertTrue(self.runtime.engine.closed)
    def test_strict_fields_limits_and_no_tool_execution_surface(self):
        for request in [
            {'command': 'submit', 'owner': 'a', 'task': 'x', 'argv': ['sh']},
            {'command': 'submit', 'owner': 'a', 'task': 'x', 'max_tokens': True},
            {'command': 'submit', 'owner': 'a', 'task': 'x', 'max_tokens': 1025},
            {'command': 'submit', 'owner': 'a', 'task': 'x' * 6001},
            {'command': 'submit', 'owner': '../other', 'task': 'x'},
            {'command': 'status', 'endpoint': 'https://example.invalid'},
            {'command': 'start', 'server_path': '/bin/sh'},
            {'command': 'submit', 'owner': 'a', 'task': 'x', 'role': 'bad\nrole'},
        ]:
            with self.subTest(request=request), self.assertRaises(LocalError):
                _dispatch(self.runtime, request)
        self.assertEqual(self.runtime.engine.calls, [])
    def test_result_storage_is_bounded_even_if_engine_overproduces(self):
        self.runtime.engine.run = lambda *_: ('x' * (LIMITS['result_bytes'] + 100), {}, False)
        job = self.runtime.submit('a', 'small task')
        result = self.runtime.read('a', job['id'], wait=2, limit=LIMITS['result_bytes'])
        self.assertEqual(len(result['result']), LIMITS['result_bytes'])
        self.assertTrue(result['truncated'])
    def test_status_when_absent_is_fast_and_never_starts(self):
        client = LocalClient(self.path / 'absent')
        before = time.monotonic()
        with patch('local_runtime.subprocess.Popen') as spawn:
            self.assertFalse(client.status()['running'])
            spawn.assert_not_called()
        self.assertLess(time.monotonic() - before, .5)
        self.assertFalse((self.path / 'absent').exists())

    def test_submit_connect_failure_is_proven_unsent_without_changing_status_calls(self):
        client = LocalClient(self.path / 'absent')
        with self.assertRaises(LocalRequestNotSent) as failure:
            client.submit('a', 'not accepted')
        self.assertIsInstance(failure.exception.__cause__, FileNotFoundError)
        with self.assertRaises(FileNotFoundError):
            client._call('status')

    def test_submit_loss_after_connect_is_not_proven_unsent(self):
        for phase in ('send', 'response'):
            with self.subTest(phase=phase), patch('local_runtime.socket.socket') as factory:
                connection = factory.return_value.__enter__.return_value
                connection.getsockopt.return_value = struct.pack('3i', 123, os.getuid(), os.getgid())
                if phase == 'send':
                    connection.sendall.side_effect = ConnectionResetError('lost while sending')
                else:
                    connection.makefile.return_value.__enter__.return_value.readline.side_effect = ConnectionResetError('lost reply')
                with self.assertRaises(LocalError) as failure:
                    LocalClient(self.path).submit('a', 'possibly accepted')
                self.assertNotIsInstance(failure.exception, LocalRequestNotSent)
                connection.connect.assert_called_once()
                connection.sendall.assert_called_once()
    def test_stop_treats_stale_socket_as_already_stopped(self):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stale:
            stale.bind(str(self.path / 'control.sock'))
        self.assertEqual(LocalClient(self.path)._stop_once(), {'stopping': False, 'running': False})


class StreamingHTTPTests(unittest.TestCase):
    def setUp(self):
        self.started = threading.Event()
        self.disconnected = threading.Event()
        self.mode = 'stall'
        self.response_delay = 0
        self.native_calls = None
        self.native_finish = 'tool_calls'
        self.token_count = 10
        self.requests = []
        fixture = self
        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                fixture.requests.append((self.path, body, self.headers.get('Authorization')))
                time.sleep(fixture.response_delay)
                if self.path == '/apply-template':
                    value = {'prompt': 'rendered chat'}
                elif self.path == '/tokenize':
                    value = {'tokens': list(range(2048 if fixture.mode == 'oversize' else fixture.token_count))}
                else:
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                    generation_count = sum(path == '/v1/chat/completions' for path, *_ in fixture.requests)
                    if fixture.mode.startswith('native') and not (fixture.mode == 'native_then_text' and generation_count > 1):
                        name = 'shell' if fixture.mode == 'native_unknown' else 'workspace'
                        calls = fixture.native_calls if fixture.native_calls is not None else [
                                {'index': 0, 'function': {'name': name, 'arguments': '{"operation":"read",'}},
                                {'index': 1 if fixture.mode == 'native_parallel' else 0,
                                 'function': {'arguments': '"path":"evidence.txt"}'}}]
                        for call in calls:
                            frame = {'choices': [{'delta': {'tool_calls': [call]}}]}
                            self.wfile.write(b'data: ' + json.dumps(frame).encode() + b'\n\n')
                        end = {'choices': [{'delta': {}, 'finish_reason': fixture.native_finish}]}
                        self.wfile.write(b'data: ' + json.dumps(end).encode() + b'\n\ndata: [DONE]\n\n')
                        self.wfile.flush()
                        fixture.started.set()
                        return
                    if fixture.mode in ('reasoning', 'reasoning_only'):
                        self.wfile.write(b'data: {"choices":[{"delta":{"content":null,"reasoning_content":"internal planning text"}}]}\n\n')
                    if fixture.mode != 'reasoning_only':
                        self.wfile.write(b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n')
                    self.wfile.flush()
                    fixture.started.set()
                    if fixture.mode in ('complete', 'length', 'reasoning', 'reasoning_only', 'native_then_text'):
                        if fixture.mode == 'reasoning':
                            self.wfile.write(b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n')
                        if fixture.mode == 'length':
                            self.wfile.write(b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n\n')
                        self.wfile.write(b'data: {"choices":[],"usage":{"completion_tokens":1}}\n\ndata: [DONE]\n\n')
                        self.wfile.flush()
                        return
                    self.connection.settimeout(3)
                    try:
                        if self.connection.recv(1) == b'':
                            fixture.disconnected.set()
                    except OSError:
                        pass
                    return
                encoded = json.dumps(value).encode()
                self.send_response(200)
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.temp = tempfile.TemporaryDirectory()
        self.engine = LlamaEngine(Path(self.temp.name))
        self.engine.port = self.server.server_port
        self.engine.key = 'fixture-private-key'
        self.engine.context = 2048
    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def test_main_http_preflight_and_generation_can_exceed_helper_timeout(self):
        self.mode = 'complete'
        self.response_delay = .06
        with patch.dict(LIMITS, timeout_seconds=.02):
            result, _, _ = self.engine.run('Keep working', 'phone conversation assistant', 32,
                                           threading.Event(), lambda _: None)
        self.assertEqual(result, 'hello')
        self.assertEqual([path for path, *_ in self.requests],
                         ['/apply-template', '/tokenize', '/v1/chat/completions'])

    def test_actual_http_phases_report_counts_without_prompt_or_credentials(self):
        self.mode = 'complete'
        observations = []
        def register(connection):
            pass
        register.observe = lambda **values: observations.append(values)
        result, _, _ = self.engine.run('private fixture prompt', 'assistant', 64,
                                       threading.Event(), register)
        self.assertEqual(result, 'hello')
        self.assertEqual(observations[0], {'phase': 'preflight', 'stage': 'apply_template'})
        self.assertEqual(observations[1], {'phase': 'preflight', 'stage': 'tokenize'})
        self.assertIn({'phase': 'preflight', 'stage': 'tokenize', 'prompt_tokens': 10}, observations)
        self.assertIn({'phase': 'generation', 'stage': 'waiting_for_output'}, observations)
        self.assertIn({'phase': 'generation', 'stage': 'streaming_output', 'output_bytes': 5}, observations)
        serialized = json.dumps(observations)
        self.assertNotIn('private fixture prompt', serialized)
        self.assertNotIn(self.engine.key, serialized)

    def test_cancel_closes_stalled_stream_before_any_further_tokens(self):
        cancel = threading.Event()
        connections, errors = [], []
        def run():
            try:
                self.engine.run('short task', 'phone conversation assistant', 32, cancel, connections.append)
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        self.assertTrue(self.started.wait(2))
        # urllib clears HTTPConnection.sock after returning its response.
        eventually(lambda: connections[-1].sock is None)
        self.assertIsNotNone(connections[-1]._local_socket)
        cancel.set()
        _interrupt(connections[-1])
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertTrue(self.disconnected.wait(1))
    def test_prompt_context_is_checked_before_generation(self):
        self.mode = 'oversize'
        with self.assertRaisesRegex(LocalError, 'exceeds the local model context'):
            self.engine.run('short task', 'assistant', 32, threading.Event(), lambda _: None)
        self.assertFalse(self.started.is_set())
        self.assertEqual([path for path, _, _ in self.requests], ['/apply-template', '/tokenize'])
    def test_32k_tokenization_response_above_old_byte_limit_is_accepted(self):
        self.mode = 'complete'
        self.engine.context = 32768
        self.token_count = 32768 - 32
        self.assertGreater(len(json.dumps({'tokens': list(range(self.token_count))}).encode()), 131072)
        result, usage, truncated = self.engine.run('short task', 'assistant', 32, threading.Event(), lambda _: None)
        self.assertEqual(result, 'hello')
        self.assertEqual(usage['completion_tokens'], 1)
        self.assertFalse(truncated)
        self.assertEqual([path for path, _, _ in self.requests],
                         ['/apply-template', '/tokenize', '/v1/chat/completions'])
    def test_32k_preprocessing_still_enforces_output_token_reserve(self):
        self.mode = 'complete'
        self.engine.context = 32768
        self.token_count = 32768 - 32 + 1
        with self.assertRaisesRegex(LocalError, 'exceeds the local model context'):
            self.engine.run('short task', 'assistant', 32, threading.Event(), lambda _: None)
        self.assertFalse(self.started.is_set())
        self.assertEqual([path for path, _, _ in self.requests], ['/apply-template', '/tokenize'])
    def test_stream_usage_and_loopback_ignore_ambient_proxy_credentials(self):
        self.mode = 'complete'
        with patch.dict(os.environ, {'http_proxy': 'http://127.0.0.1:1', 'HTTP_PROXY': 'http://127.0.0.1:1', 'OPENAI_API_KEY': 'must-not-be-sent'}):
            result, usage, truncated = self.engine.run('short task', 'reviewer', 32, threading.Event(), lambda _: None)
        self.assertEqual(result, 'hello')
        self.assertEqual(usage['completion_tokens'], 1)
        self.assertFalse(truncated)
        self.assertTrue(all(auth == 'Bearer fixture-private-key' for _, _, auth in self.requests))
        payload = self.requests[-1][1]
        self.assertNotIn('tools', payload)
        self.assertNotIn('endpoint', payload)
        self.assertEqual(payload['max_tokens'], 32)
        self.assertEqual(payload['temperature'], .4)
        self.assertNotIn('chat_template_kwargs', payload)
        self.assertNotIn('reasoning_format', payload)

    def test_main_native_function_fragments_reach_existing_host_protocol(self):
        self.mode = 'native'
        result, _, truncated = self.engine.run('Read evidence.txt', 'phone conversation assistant', 64,
                                               threading.Event(), lambda _: None)
        self.assertEqual(json.loads(result), {'tool': 'workspace',
                         'arguments': {'operation': 'read', 'path': 'evidence.txt'}})
        self.assertFalse(truncated)
        preflight, generation = self.requests[0][1], self.requests[-1][1]
        self.assertEqual(preflight['tools'], generation['tools'])
        self.assertEqual(generation['tools'][0]['function']['name'], 'workspace')
        self.assertFalse(generation['parallel_tool_calls'])
        self.assertEqual(generation['temperature'], 0.0)
        self.assertEqual(generation['tool_choice'], 'auto')

    def test_native_call_preserves_provider_id_function_and_exact_argument_bytes(self):
        self.mode = 'native'
        self.native_calls = [
            {'index': 0, 'id': 'provider.cpu:1', 'type': 'function',
             'function': {'name': 'phone_', 'arguments': '{ "action" :'}},
            {'index': 0, 'id': 'provider.cpu:1', 'function': {'name': 'control', 'arguments': ' "cpu" }'}},
        ]
        result, usage, truncated = self.engine.run('Inspect hardware', 'phone conversation assistant', 64,
                                                   threading.Event(), lambda _: None)
        self.assertFalse(truncated)
        self.assertEqual(json.loads(result)['arguments']['argv'], ['@phone', 'cpu'])
        self.assertEqual(usage['native_call'], {'id': 'provider.cpu:1', 'type': 'function',
                         'function': {'name': 'phone_control', 'arguments': '{ "action" : "cpu" }'}})
        self.assertEqual(usage['native_call_id_source'], 'provider')

    def test_missing_provider_id_is_explicitly_host_assigned_without_renaming_function(self):
        self.mode = 'native'
        _, usage, _ = self.engine.run('Read evidence.txt', 'phone conversation assistant', 64,
                                       threading.Event(), lambda _: None)
        self.assertEqual(usage['native_call_id_source'], 'host')
        self.assertRegex(usage['native_call']['id'], r'^call_[0-9a-f]{32}$')
        self.assertEqual(usage['native_call']['function'],
                         {'name': 'workspace', 'arguments': '{"operation":"read","path":"evidence.txt"}'})

    def test_conflicting_or_invalid_native_call_metadata_and_truncation_are_rejected(self):
        base = {'index': 0, 'id': 'call-1', 'type': 'function',
                'function': {'name': 'phone_control', 'arguments': '{"action":"cpu"}'}}
        variants = [
            [base, {'index': 0, 'id': 'call-2'}],
            [base, {'index': 0, 'function': {'name': 'workspace'}}],
            [dict(base, id=['invalid'])],
            [dict(base, type='shell')],
            [dict(base, function=[])],
            [base, {'index': 0, 'function': {'name': False}}],
            [base, {'index': 0, 'function': {'arguments': []}}],
            [dict(base, function={'name': 'phone_control', 'arguments': '{"action":"cpu","action":"status"}'})],
        ]
        self.mode = 'native'
        for calls in variants:
            with self.subTest(calls=calls), self.assertRaises(LocalError):
                self.native_calls = calls
                self.engine.run('Inspect hardware', 'phone conversation assistant', 64,
                                threading.Event(), lambda _: None)
        self.native_calls = [base]
        self.native_finish = 'length'
        with self.assertRaisesRegex(LocalError, 'incomplete'):
            self.engine.run('Inspect hardware', 'phone conversation assistant', 64,
                            threading.Event(), lambda _: None)

    def test_native_phone_call_survives_runtime_job_and_exact_tool_history(self):
        from agent import Store
        from local_chat import LocalChatAgent
        self.mode = 'native_then_text'
        raw_arguments = '{ "action" : "cpu" }'
        call = {'id': 'provider-hardware-1', 'type': 'function',
                'function': {'name': 'phone_control', 'arguments': raw_arguments}}
        self.native_calls = [dict(call, index=0)]
        self.engine.context = 8192
        self.engine.model = {'name': 'loopback-fixture', 'backend': 'llama-cpu',
                             'native_conversation_protocol': 1, 'context_per_worker': 8192, 'parallel': 1}
        self.engine.start = lambda: self.engine.model
        self.engine.is_alive = lambda: True
        self.engine.close = lambda: None
        runtime = LocalRuntime(Path(self.temp.name) / 'runtime-state', lambda _: self.engine)
        project = Path(self.temp.name) / 'project'
        project.mkdir()
        store = Store(Path(self.temp.name) / 'chat-state')
        sid = store.create(project)
        client = types.SimpleNamespace(status=runtime.status, submit=runtime.submit,
                                       read=runtime.read, cancel=runtime.cancel)
        agent = LocalChatAgent(store, project, client=client)
        agent.local.cancel = Mock()
        agent.execute = Mock(return_value={'soc': {'model': 'fixture-soc'}})
        try:
            eventually(lambda: runtime.status()['ready'])
            agent.submit(sid, 'Inspect the phone hardware.', 'native-history-regression')
            agent.start_thread.join(3)
            self.assertFalse(agent.start_thread.is_alive())
            self.assertEqual(agent.last_answer, 'hello')
            agent.execute.assert_called_once_with({'operation': 'command', 'argv': ['@phone', 'cpu']})
            generations = [body for path, body, _ in self.requests if path == '/v1/chat/completions']
            self.assertEqual(len(generations), 2)
            self.assertEqual(generations[1]['messages'][-2],
                             {'role': 'assistant', 'content': None, 'tool_calls': [call]})
            self.assertEqual(generations[1]['messages'][-1]['tool_call_id'], call['id'])
            self.assertEqual(generations[1]['messages'][-1]['name'], 'phone_control')
            with runtime.condition:
                row = runtime.db.execute("SELECT * FROM jobs WHERE usage LIKE '%native_call%' LIMIT 1").fetchone()
                restored = runtime._job(row, read=True)
            self.assertEqual(restored['native_call'], call)
            self.assertEqual(restored['native_call_id_source'], 'provider')
        finally:
            agent.close()
            runtime.close()
            store.db.close()

    def test_native_transcript_preserves_typed_tool_messages_and_uses_short_system_guide(self):
        self.mode = 'complete'
        messages = [{'role': 'user', 'content': 'Read sample.txt'},
                    {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'unique-call', 'type': 'function',
                     'function': {'name': 'workspace', 'arguments': '{"operation":"read","path":"sample.txt"}'}}]},
                    {'role': 'tool', 'tool_call_id': 'unique-call', 'name': 'workspace',
                     'content': '{"success":true,"result":{"content":"actual evidence"}}'}]
        result, _, _ = self.engine.run(encode_conversation(messages), 'phone conversation assistant', 64,
                                      threading.Event(), lambda _: None)
        payload = self.requests[-1][1]
        self.assertEqual(result, 'hello')
        self.assertEqual(payload['messages'][0], {'role': 'system', 'content': NATIVE_GUIDE})
        self.assertEqual(payload['messages'][1:], messages)
        self.assertLess(len(NATIVE_GUIDE), 1700)

    def test_native_text_chunks_are_available_before_completion(self):
        self.mode = 'reasoning'
        chunks = []
        result, usage, _ = self.engine.run_with_updates('Hello', 'phone conversation assistant', 64,
                                                       threading.Event(), lambda _: None, chunks.append)
        self.assertEqual(chunks, ['hello', ' world'])
        self.assertEqual(result, 'hello world')
        self.assertIn('first_token_seconds', usage)

    def test_explicit_execution_role_requires_native_ostadix_function(self):
        self.mode = 'complete'
        self.engine.run('Execute through OSTADIX', 'phone OSTADIX action assistant', 64,
                        threading.Event(), lambda _: None)
        payload = self.requests[-1][1]
        self.assertEqual(payload['tool_choice'], 'required')
        self.assertEqual([tool['function']['name'] for tool in payload['tools']],
                         ['workspace', 'ostadix_execute', 'phone_control'])
        self.assertEqual(self.requests[0][1]['tools'], payload['tools'])
        self.assertEqual(self.requests[0][1]['tool_choice'], payload['tool_choice'])

    def test_helper_cannot_request_native_actions_and_parallel_or_unknown_calls_fail(self):
        for role, mode, error in [
                ('reviewer', 'native', 'Unexpected'),
                ('phone conversation assistant', 'native_parallel', 'parallel'),
                ('phone conversation assistant', 'native_unknown', 'Unknown')]:
            with self.subTest(role=role, mode=mode), self.assertRaisesRegex(LocalError, error):
                self.mode = mode
                self.engine.run('task', role, 64, threading.Event(), lambda _: None)
    def test_model_profile_matches_template_preflight_and_generation(self):
        self.mode = 'complete'
        profile = {'provenance_repo': 'huihui-ai/Huihui-MiniCPM5-2B-abliterated',
                   'sampling': {'temperature': 1.0, 'top_p': .95, 'min_p': 0.0},
                   'chat_template_kwargs': {'enable_thinking': False}}
        self.engine.sampling, self.engine.chat_template_kwargs = _model_options(profile)
        self.engine.run('short task', 'reviewer', 32, threading.Event(), lambda _: None)
        preflight, generation = self.requests[0][1], self.requests[-1][1]
        self.assertEqual(preflight['messages'], generation['messages'])
        self.assertEqual(preflight['chat_template_kwargs'], generation['chat_template_kwargs'])
        self.assertIs(generation['chat_template_kwargs']['enable_thinking'], False)
        self.assertEqual(preflight['reasoning_format'], generation['reasoning_format'])
        self.assertEqual(generation['reasoning_format'], 'deepseek')
        for name, value in profile['sampling'].items():
            self.assertEqual(generation[name], value)
    def test_reasoning_is_excluded_but_all_answer_chunks_are_preserved(self):
        self.mode = 'reasoning'
        result, usage, truncated = self.engine.run('short task', 'assistant', 32, threading.Event(), lambda _: None)
        self.assertEqual(result, 'hello world')
        self.assertNotIn('planning', result)
        self.assertEqual(usage['completion_tokens'], 1)
        self.assertFalse(truncated)
    def test_reasoning_only_stream_is_not_reported_as_a_completed_answer(self):
        self.mode = 'reasoning_only'
        with self.assertRaisesRegex(LocalError, 'no final answer text'):
            self.engine.run('short task', 'assistant', 32, threading.Event(), lambda _: None)
    def test_token_budget_truncation_keeps_reading_final_usage(self):
        self.mode = 'length'
        result, usage, truncated = self.engine.run('short task', 'assistant', 1, threading.Event(), lambda _: None)
        self.assertEqual(result, 'hello')
        self.assertTrue(truncated)
        self.assertEqual(usage['completion_tokens'], 1)


@unittest.skipIf(os.getuid() == 0, 'Resident daemon must run as the normal application user')
class ServeCleanupTests(unittest.TestCase):
    def assert_unlocked(self, state):
        with (state / 'daemon.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def test_coordinator_init_failure_never_starts_model_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            with patch('local_assistants.AssistantCoordinator', side_effect=RuntimeError('fixture state invalid')), \
                 patch('local_runtime.LocalRuntime') as runtime:
                with self.assertRaisesRegex(RuntimeError, 'fixture state invalid'):
                    serve(state, FakeEngine)
                runtime.assert_not_called()
            self.assert_unlocked(state)
    def test_bind_failure_closes_coordinator_and_runtime_even_if_close_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            runtime = Mock()
            coordinator = Mock()
            coordinator.close.side_effect = RuntimeError('fixture close error')
            with patch('local_assistants.AssistantCoordinator', return_value=coordinator), \
                 patch('local_runtime.LocalRuntime', return_value=runtime), \
                 patch('local_runtime.socketserver.ThreadingUnixStreamServer.__init__', side_effect=OSError('fixture bind failure')):
                with self.assertRaisesRegex(RuntimeError, 'fixture close error'):
                    serve(state, FakeEngine)
            coordinator.close.assert_called_once_with()
            runtime.close.assert_called_once_with()
            self.assertFalse((state / 'control.sock').exists())
            self.assert_unlocked(state)
    def test_coordinator_start_failure_closes_bound_socket_and_model(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            runtime = Mock()
            coordinator = Mock()
            coordinator.start.side_effect = RuntimeError('fixture scheduler start failed')
            with patch('local_assistants.AssistantCoordinator', return_value=coordinator), \
                 patch('local_runtime.LocalRuntime', return_value=runtime):
                with self.assertRaisesRegex(RuntimeError, 'fixture scheduler start failed'):
                    serve(state, FakeEngine)
            coordinator.close.assert_called_once_with()
            runtime.close.assert_called_once_with()
            self.assertFalse((state / 'control.sock').exists())
            self.assert_unlocked(state)


@unittest.skipIf(os.getuid() == 0, 'Resident daemon must run as the normal application user')
class DaemonLifecycleTests(unittest.TestCase):
    def test_large_unicode_native_call_roundtrips_private_socket_under_frame_limit(self):
        fixture = r'''
import json, sys
from pathlib import Path
from local_runtime import native_tool_response, serve
class Engine:
    def __init__(self, state):
        self.model = {'name': 'offline-unicode-fixture'}
        self.alive = True
    def start(self):
        return self.model
    def is_alive(self):
        return self.alive
    def run(self, *args):
        arguments = json.dumps({'operation': 'write', 'path': 'fixture.txt',
                                'content': '\U0001f642' * 14000}, ensure_ascii=False)
        call = {'id': 'provider-unicode', 'type': 'function',
                'function': {'name': 'workspace', 'arguments': arguments}}
        return native_tool_response('workspace', arguments), {'native_call': call, 'native_call_id_source': 'provider'}, False
    def close(self):
        self.alive = False
serve(Path(sys.argv[1]), Engine)
'''
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / 'state'
            client = LocalClient(state)
            process = subprocess.Popen([sys.executable, '-c', fixture, str(state)],
                                       stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
            try:
                eventually(lambda: client.status()['ready'])
                job = client.submit('fixture', 'Offline Unicode metadata roundtrip',
                                    max_tokens=64, role='phone conversation assistant')
                result = client.read('fixture', job['id'], wait=3, limit=65536)
                self.assertEqual(result['state'], 'completed')
                self.assertFalse(result['truncated'])
                arguments = result['native_call']['function']['arguments']
                self.assertEqual(json.loads(arguments)['content'], '\U0001f642' * 14000)
                self.assertEqual(json.loads(result['result'])['arguments'], json.loads(arguments))
                self.assertEqual(result['native_call_id_source'], 'provider')
                self.assertGreater(len(json.dumps(result).encode()), 262144)
                self.assertLess(len(json.dumps(result, ensure_ascii=False).encode()), 262144)
                self.assertEqual((state / 'control.sock').stat().st_mode & 0o777, 0o600)
                self.assertEqual(state.stat().st_mode & 0o777, 0o700)
            finally:
                if process.poll() is None:
                    try:
                        client._stop_once()
                    finally:
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait(timeout=2)

    def test_immediate_stop_start_waits_for_previous_engine_cleanup(self):
        fixture = r'''
import os, sys, time
from pathlib import Path
from local_runtime import serve
class Engine:
    def __init__(self, state):
        self.state = state
        self.model = {'name': 'offline-subprocess-fixture'}
        self.alive = True
    def record(self, event):
        with (self.state / 'lifecycle.log').open('a') as log:
            log.write(str(os.getpid()) + ' ' + event + '\n')
    def start(self):
        self.record('started')
        return self.model
    def is_alive(self):
        return self.alive
    def run(self, task, role, max_tokens, cancelled, register):
        return task, {}, False
    def close(self):
        self.record('closing')
        time.sleep(.7)
        self.alive = False
        self.record('closed')
serve(Path(sys.argv[1]), Engine)
'''
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / 'state'
            client = LocalClient(state)
            processes = []
            real_spawn = subprocess.Popen
            def spawn_fixture(command, **kwargs):
                process = real_spawn([sys.executable, '-c', fixture, str(state)], **kwargs)
                processes.append(process)
                return process
            with patch('local_runtime.subprocess.Popen', side_effect=spawn_fixture):
                try:
                    first = client._start_once()
                    self.assertTrue(first['running'])
                    eventually(lambda: client.status()['ready'])
                    self.assertEqual(client._start_once()['daemon_pid'], first['daemon_pid'])
                    client._stop_once()
                    self.assertFalse(client.status()['running'])
                    second = client._start_once()
                    self.assertTrue(second['running'], second)
                    self.assertNotEqual(second['daemon_pid'], first['daemon_pid'])
                    eventually(lambda: client.status()['ready'])
                    events = (state / 'lifecycle.log').read_text().splitlines()
                    self.assertLess(events.index(str(first['daemon_pid']) + ' closed'),
                                    events.index(str(second['daemon_pid']) + ' started'))
                    job = client.submit('fixture', 'fresh daemon accepts work')
                    self.assertEqual(client.read('fixture', job['id'], wait=2)['result'], 'fresh daemon accepts work')
                finally:
                    client._stop_once()
                    for process in processes:
                        process.wait(timeout=4)
if __name__ == '__main__':
    unittest.main()
