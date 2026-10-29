"""Offline Gemini request, signature, owner isolation and cancellation checks."""
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conversation_protocol import encode_conversation
from gemini_client import GeminiClient, request_body
from local_runtime import LocalError


ROLE = 'phone conversation assistant'
TASK = encode_conversation([{'role': 'user', 'content': 'Read probe.txt'}])
SIGNATURE = 'c2lnbmF0dXJl'


def chunk(delta=None, finish=None, usage=None):
    value = {'choices': [{'index': 0, 'delta': delta or {}, 'finish_reason': finish}]}
    if usage is not None:
        value['usage'] = usage
    return {'type': 'chunk', 'chunk': value}


def events(*values):
    return ''.join(json.dumps(value) + '\n' for value in values)


class Input(io.StringIO):
    def close(self):
        # Keep the harmless request available to assertions after transport EOF.
        pass


class Process:
    def __init__(self, output):
        self.stdin = Input()
        self.stdout = io.StringIO(output)
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return 0


class BlockingOutput:
    def __init__(self, initial=''):
        self.initial = io.StringIO(initial)
        self.waiting = threading.Event()
        self.released = threading.Event()
        self.closed = False

    def readline(self, limit):
        line = self.initial.readline(limit)
        if line:
            return line
        self.waiting.set()
        self.released.wait(3)
        return ''

    def close(self):
        self.closed = True
        self.released.set()


class BlockingProcess(Process):
    def __init__(self, initial=''):
        super().__init__('')
        self.stdout = BlockingOutput(initial)

    def terminate(self):
        super().terminate()
        self.stdout.released.set()


class GeminiClientTests(unittest.TestCase):
    def setUp(self):
        self.client = GeminiClient(node='/fixture/node')
        self.addCleanup(self.client.close)

    def reply(self, process, task=TASK, role=ROLE, **kwargs):
        with patch('gemini_client.subprocess.Popen', return_value=process) as popen:
            job = self.client.submit('owner-a', task, role=role, **kwargs)
            result = self.client.read('owner-a', job['id'], wait=2, limit=32768)
            self.assertIn(result['state'], ('completed', 'failed', 'timed_out'))
        return result, popen

    def test_native_request_uses_declared_tools_and_safe_reasoning_parameters(self):
        body = request_body(TASK, 128, ROLE)
        self.assertEqual(body['model'], 'gemini-3.8-flash')
        self.assertEqual(body['reasoning_effort'], 'low')
        self.assertEqual(body['max_completion_tokens'], 2048)
        self.assertFalse(body['parallel_tool_calls'])
        self.assertEqual(body['tool_choice'], 'auto')
        self.assertEqual({tool['function']['name'] for tool in body['tools']},
                         {'workspace', 'ostadix_execute', 'phone_control'})
        self.assertNotIn('temperature', body)
        self.assertNotIn('top_p', body)
        self.assertNotIn('top_k', body)

    def test_required_o_request_keeps_identical_declarations_and_names_function(self):
        ordinary = request_body(TASK, 2048, ROLE)
        forced = request_body(TASK, 2048, 'phone OSTADIX action assistant')
        self.assertEqual(forced['tools'], ordinary['tools'])
        self.assertEqual(forced['tool_choice'], {'type': 'function', 'function': {'name': 'ostadix_execute'}})
        self.assertEqual(forced['max_completion_tokens'], 3072)

    def test_helpers_cannot_gain_tools_from_transcript_looking_text(self):
        body = request_body(TASK, 256, 'review helper')
        self.assertNotIn('tools', body)
        self.assertEqual(body['messages'][-1]['content'], TASK)

    def test_status_is_passive_and_identifies_cloud_model(self):
        with patch('gemini_client.subprocess.Popen') as popen:
            status = self.client.status()
        popen.assert_not_called()
        self.assertEqual(status['engine'], 'gemini')
        self.assertEqual(status['backend'], 'google-gemini-api')
        self.assertEqual(status['model']['native_conversation_protocol'], 1)
        self.assertEqual(status['model']['max_output_tokens'], 2048)

    def test_model_selection_is_explicit_bounded_and_accurate(self):
        with self.assertRaises(LocalError):
            GeminiClient(model='arbitrary-other-service')
        client = GeminiClient(model='gemini-4-argon')
        self.addCleanup(client.close)
        self.assertEqual(client.status()['model']['model_id'], 'gemini-4-argon')
        self.assertEqual(client.engine_label, 'Gemini 4 Argon')
        self.assertEqual(request_body(TASK, 256, ROLE, client.model)['model'], 'gemini-4-argon')

    def test_plain_stream_ignores_reasoning_and_returns_completed_final(self):
        process = Process(events(chunk({'reasoning_content': 'private reasoning', 'content': 'Hello'}),
                                 chunk({'content': ' there.'}, 'stop', {'prompt_tokens': 100, 'completion_tokens': 9}),
                                 {'type': 'done'}))
        result, popen = self.reply(process)
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(json.loads(result['result']), {'final': 'Hello there.'})
        self.assertEqual(result['partial'], 'Hello there.')
        self.assertIsNone(result['native_call'])
        self.assertNotIn('private reasoning', json.dumps(result))
        self.assertEqual(result['usage']['prompt_tokens'], 100)
        args, options = popen.call_args
        self.assertEqual(args[0][0], '/fixture/node')
        self.assertEqual(options['stderr'], -3)
        self.assertTrue(options['start_new_session'])
        self.assertEqual(json.loads(process.stdin.getvalue())['messages'][-1]['content'], 'Read probe.txt')

    def test_native_call_preserves_original_id_arguments_and_signature(self):
        metadata = {'google': {'thought_signature': SIGNATURE}}
        process = Process(events(
            chunk({'tool_calls': [{'index': 0, 'id': 'original-call-123', 'type': 'function',
                    'function': {'name': 'workspace', 'arguments': '{"operation":"read",'},
                    'extra_content': metadata}]}),
            chunk({'tool_calls': [{'index': 0, 'function': {'arguments': '"path":"probe.txt"}'}}]}, 'tool_calls'),
            {'type': 'done'}))
        result, _ = self.reply(process)
        self.assertEqual(result['state'], 'completed')
        call = result['native_call']
        self.assertEqual(call['id'], 'original-call-123')
        self.assertEqual(call['function']['arguments'], '{"operation":"read","path":"probe.txt"}')
        self.assertEqual(call['extra_content'], metadata)
        self.assertEqual(json.loads(result['result']), {'tool': 'workspace',
                         'arguments': {'operation': 'read', 'path': 'probe.txt'}})
        self.assertEqual(result['partial'], '')

    def test_o_call_normalizes_host_route_without_executing_anything(self):
        call = {'index': 0, 'id': 'o-call', 'function': {'name': 'ostadix_execute', 'arguments': '{"source":"7*9"}'},
                'extra_content': {'google': {'thought_signature': SIGNATURE}}}
        result, _ = self.reply(Process(events(chunk({'tool_calls': [call]}, 'tool_calls'), {'type': 'done'})))
        self.assertEqual(result['state'], 'completed')
        self.assertIn('@ostadix', result['result'])
        self.assertIn('__oval_result__ = (7*9)', result['result'])
        self.assertEqual(result['native_call']['function']['arguments'], '{"source":"7*9"}')

    def test_missing_signature_unknown_function_and_parallel_calls_fail_closed(self):
        cases = [
            [{'index': 0, 'id': 'a', 'function': {'name': 'workspace', 'arguments': '{"operation":"list"}'}}],
            [{'index': 0, 'id': 'a', 'function': {'name': 'unknown', 'arguments': '{}'},
              'extra_content': {'google': {'thought_signature': SIGNATURE}}}],
            [{'index': 0, 'id': 'a', 'function': {'name': 'workspace', 'arguments': '{}'}},
             {'index': 1, 'id': 'b', 'function': {'name': 'workspace', 'arguments': '{}'}}],
        ]
        for calls in cases:
            with self.subTest(calls=calls):
                result, _ = self.reply(Process(events(chunk({'tool_calls': calls}, 'tool_calls'), {'type': 'done'})))
                self.assertEqual(result['state'], 'failed')
                self.assertIsNone(result['native_call'])
                self.assertEqual(result['result'], '')

    def test_25_native_function_can_omit_optional_thought_signature(self):
        self.client.close()
        self.client = GeminiClient(node='/fixture/node', model='gemini-2.5-flash')
        call = {'index': 0, 'id': 'original-25-id',
                'function': {'name': 'workspace', 'arguments': '{"operation":"read","path":"probe.txt"}'}}
        result, _ = self.reply(Process(events(chunk({'tool_calls': [call]}, 'tool_calls'), {'type': 'done'})))
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['native_call']['id'], 'original-25-id')
        self.assertNotIn('extra_content', result['native_call'])
        self.client.close()

    def test_incomplete_stream_never_returns_a_partial_answer_or_action(self):
        for finish, done in [(None, True), ('stop', False)]:
            with self.subTest(finish=finish, done=done):
                output = [chunk({'content': 'partial'}, finish)]
                if done:
                    output.append({'type': 'done'})
                result, _ = self.reply(Process(events(*output)))
                self.assertEqual(result['state'], 'failed')
                self.assertEqual(result['partial'], '')
                self.assertEqual(result['result'], '')

    def test_output_limit_keeps_plain_answer_with_explicit_incomplete_label(self):
        result, popen = self.reply(Process(events(chunk({'content': 'Observed package list starts here.'}, 'length'),
                                                 {'type': 'done'})))
        self.assertEqual(result['state'], 'completed')
        self.assertTrue(result['answer_incomplete'])
        self.assertTrue(result['truncated'])
        self.assertIsNone(result['native_call'])
        answer = json.loads(result['result'])['final']
        self.assertTrue(answer.startswith('Observed package list starts here.'))
        self.assertIn('answer is incomplete', answer)
        self.assertIn('Earlier tool results remain saved', answer)
        self.assertEqual(popen.call_count, 1)

    def test_output_limit_never_completes_partial_function_arguments(self):
        call = {'index': 0, 'id': 'incomplete-write', 'function': {'name': 'workspace',
                'arguments': '{"operation":"write","path":"pending.txt","content":"unfinished'}}
        result, _ = self.reply(Process(events(chunk({'tool_calls': [call]}, 'length'), {'type': 'done'})))
        self.assertEqual(result['state'], 'failed')
        self.assertTrue(result['truncated'])
        self.assertIsNone(result['native_call'])
        self.assertEqual(result['result'], '')
        self.assertEqual(result['partial'], '')

    def test_transport_errors_discard_raw_message_and_never_replay(self):
        process = Process(events(chunk({'content': 'Already emitted text'}), {'type': 'error', 'code': 'http_429',
                                  'message': 'SECRET_API_KEY_AND_ACCOUNT'}))
        result, popen = self.reply(process)
        self.assertEqual(result['state'], 'failed')
        self.assertIn('429', result['error'])
        self.assertNotIn('SECRET', json.dumps(result))
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(result['error_code'], 'http_429')

    def test_quota_retries_once_before_output_with_identical_prompt(self):
        first = Process(events({'type': 'error', 'code': 'http_429', 'retry_seconds': 2.5}))
        second = Process(events(chunk({'content': 'Recovered reply'}, 'stop'), {'type': 'done'}))
        with patch('gemini_client.subprocess.Popen', side_effect=[first, second]) as popen, \
                patch.object(self.client, '_quota_wait', return_value=False) as wait:
            job = self.client.submit('owner-a', TASK, role=ROLE)
            result = self.client.read('owner-a', job['id'], wait=2, limit=32768)
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(json.loads(result['result'])['final'], 'Recovered reply')
        self.assertEqual(result['retry_count'], 1)
        self.assertEqual(result['stage'], 'completed')
        self.assertIsNone(result['retry_after_seconds'])
        self.assertNotIn('error_code', result)
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(wait.call_args.args[1], 2.5)
        bodies = [json.loads(p.stdin.getvalue()) for p in (first, second)]
        timeouts = [body.pop('_ostadix_timeout_ms') for body in bodies]
        self.assertEqual(bodies[0], bodies[1])
        self.assertEqual(timeouts, [0, 0])

    def test_fallback_mode_reports_safe_quota_immediately_without_wait(self):
        self.client.quota_fallback = True
        process = Process(events({'type': 'error', 'code': 'http_429', 'retry_seconds': 60}))
        with patch.object(self.client, '_quota_wait') as wait:
            result, popen = self.reply(process)
        self.assertEqual(result['state'], 'failed')
        self.assertTrue(result['safe_quota_fallback'])
        popen.assert_called_once()
        wait.assert_not_called()

    def test_fallback_mode_rejects_quota_after_received_output(self):
        self.client.quota_fallback = True
        result, _ = self.reply(Process(events(chunk({'content': 'Already started'}),
            {'type': 'error', 'code': 'http_429'})))
        self.assertFalse(result['safe_quota_fallback'])

    def test_second_quota_failure_is_final_and_unknown_delay_defaults_to_sixty(self):
        processes = [Process(events({'type': 'error', 'code': 'http_429'})) for _ in range(2)]
        with patch('gemini_client.subprocess.Popen', side_effect=processes) as popen, \
                patch.object(self.client, '_quota_wait', return_value=False) as wait:
            job = self.client.submit('owner-a', TASK, role=ROLE)
            result = self.client.read('owner-a', job['id'], wait=2)
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['error_code'], 'http_429')
        self.assertEqual(result['retry_count'], 1)
        self.assertEqual(popen.call_count, 2)
        self.assertEqual(wait.call_args.args[1], 60)

    def test_quota_wait_exposes_status_and_cancels_without_second_network_request(self):
        process = Process(events({'type': 'error', 'code': 'http_429', 'retry_seconds': 60}))
        with patch('gemini_client.subprocess.Popen', return_value=process) as popen:
            job = self.client.submit('owner-a', TASK, role=ROLE)
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                result = self.client.read('owner-a', job['id'])
                if result['stage'] == 'quota_wait':
                    break
                time.sleep(.001)
            self.assertEqual(result['state'], 'running')
            self.assertEqual(result['stage'], 'quota_wait')
            self.assertEqual(result['retry_after_seconds'], 60)
            start = time.monotonic()
            self.client.cancel('owner-a', job['id'])
            self.client.close()
            self.assertLess(time.monotonic() - start, .5)
            self.assertEqual(popen.call_count, 1)
        self.assertEqual(self.client.read('owner-a', job['id'])['state'], 'cancelled')

    def test_quota_after_native_fragment_never_retries_or_returns_action(self):
        process = Process(events(chunk({'tool_calls': [{'index': 0, 'id': 'partial-call',
                                  'function': {'name': 'workspace', 'arguments': '{'}}]}),
                                 {'type': 'error', 'code': 'http_429', 'retry_seconds': 1}))
        result, popen = self.reply(process)
        self.assertEqual(result['state'], 'failed')
        self.assertIsNone(result['native_call'])
        self.assertEqual(result['result'], '')
        self.assertEqual(popen.call_count, 1)

    def test_known_daily_quota_never_waits_or_retries(self):
        for limit in (0, 20):
            with self.subTest(limit=limit):
                process = Process(events({'type': 'error', 'code': 'http_429',
                    'retry_seconds': 30, 'quota_scope': 'day', 'quota_value': limit}))
                with patch.object(self.client, '_quota_wait') as wait:
                    result, popen = self.reply(process)
                self.assertEqual(result['state'], 'failed')
                self.assertEqual(result['quota_scope'], 'day')
                self.assertEqual(result['quota_value'], limit)
                self.assertIn('daily quota', result['error'])
                self.assertEqual(result['retry_count'], 0)
                wait.assert_not_called()
                self.assertEqual(popen.call_count, 1)

    def test_quota_does_not_wait_past_overall_request_deadline(self):
        process = Process(events({'type': 'error', 'code': 'http_429', 'retry_seconds': 60}))
        original = self.client._attempt
        def slow_attempt(job, timeout_ms):
            original(job, timeout_ms)
            clock[0] += 100
        clock = [1000.0]
        with patch('gemini_client.subprocess.Popen', return_value=process) as popen, \
                patch.object(self.client, '_attempt', side_effect=slow_attempt), \
                patch('gemini_client.time.monotonic', side_effect=lambda: clock[0]), \
                patch.object(self.client, '_quota_wait') as wait:
            job = self.client.submit('owner-a', 'Helper request', role='assistant')
            # Use the actual Condition to await the bounded worker, without a
            # mocked-monotonic timed read that cannot advance independently.
            with self.client.condition:
                self.client.condition.wait_for(lambda: self.client._jobs[job['id']]['state'] == 'failed', timeout=1)
            result = self.client.read('owner-a', job['id'])
        self.assertEqual(result['state'], 'failed')
        wait.assert_not_called()
        self.assertEqual(popen.call_count, 1)

    def test_main_reply_survives_long_request_and_quota_wait(self):
        first = Process(events({'type': 'error', 'code': 'http_429', 'retry_seconds': 60}))
        second = Process(events(chunk({'content': 'Finished after the old deadline'}, 'stop'), {'type': 'done'}))
        original = self.client._attempt
        clock = [1000.0]

        def slow_attempt(job, timeout_ms):
            original(job, timeout_ms)
            clock[0] += 600

        with patch('gemini_client.subprocess.Popen', side_effect=[first, second]), \
                patch.object(self.client, '_attempt', side_effect=slow_attempt), \
                patch('gemini_client.time.monotonic', side_effect=lambda: clock[0]), \
                patch.object(self.client, '_quota_wait', return_value=False) as wait:
            job = self.client.submit('owner-a', TASK, role=ROLE)
            with self.client.condition:
                self.assertTrue(self.client.condition.wait_for(
                    lambda: self.client._jobs[job['id']]['state'] == 'completed', timeout=1))
            result = self.client.read('owner-a', job['id'])
        self.assertEqual(result['state'], 'completed')
        self.assertIn('Finished after the old deadline', result['result'])
        self.assertEqual(wait.call_args.args[1], 60)
        self.assertEqual([json.loads(p.stdin.getvalue())['_ostadix_timeout_ms'] for p in (first, second)], [0, 0])

    def test_owner_isolation_cancel_clears_preview_and_terminates_only_owned_process(self):
        process = BlockingProcess(events(chunk({'content': 'Old streamed answer'})))
        with patch('gemini_client.subprocess.Popen', return_value=process):
            job = self.client.submit('owner-a', TASK, role=ROLE)
            self.assertTrue(process.stdout.waiting.wait(1))
            self.assertEqual(self.client.read('owner-a', job['id'])['partial'], 'Old streamed answer')
            with self.assertRaises(LocalError):
                self.client.read('owner-b', job['id'])
            with self.assertRaises(LocalError):
                self.client.cancel('owner-b', job['id'])
            self.assertEqual(self.client.cancel('owner-b')['cancelled'], [])
            self.assertFalse(process.terminated)
            self.assertEqual(self.client.cancel('owner-a', job['id'])['cancelled'], [job['id']])
            self.assertTrue(process.terminated)
            result = self.client.read('owner-a', job['id'])
            self.assertEqual(result['state'], 'cancelled')
            self.assertEqual(result['partial'], '')
            self.assertEqual(result['result'], '')
            self.client.close()

    def test_queue_is_bounded_and_close_is_idempotent(self):
        process = BlockingProcess()
        with patch('gemini_client.subprocess.Popen', return_value=process):
            first = self.client.submit('owner-a', TASK, role=ROLE)
            self.assertTrue(process.stdout.waiting.wait(1))
            queued = [self.client.submit('owner-b', TASK, role=ROLE) for _ in range(8)]
            with self.assertRaisesRegex(LocalError, 'queue is full'):
                self.client.submit('owner-b', TASK, role=ROLE)
            self.client.close()
            self.client.close()
            self.assertTrue(process.terminated)
            for job in [first, *queued]:
                owner = 'owner-a' if job is first else 'owner-b'
                self.assertEqual(self.client.read(owner, job['id'])['state'], 'cancelled')
            with self.assertRaisesRegex(LocalError, 'closed'):
                self.client.submit('owner-a', TASK, role=ROLE)

    def test_cancel_escalates_only_owned_stubborn_process_and_joins_worker(self):
        class Stubborn(BlockingProcess):
            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                if not self.killed:
                    raise subprocess.TimeoutExpired('fixture', timeout)
                return 0

            def kill(self):
                self.killed = True
                self.stdout.released.set()

        process = Stubborn()
        with patch('gemini_client.subprocess.Popen', return_value=process):
            job = self.client.submit('owner-a', TASK, role=ROLE)
            self.assertTrue(process.stdout.waiting.wait(1))
            self.client.cancel('owner-a', job['id'])
            self.client.close()
        self.assertTrue(process.terminated)
        self.assertTrue(process.killed)
        self.assertTrue(self.client._worker is None or not self.client._worker.is_alive())

    def test_result_read_limit_is_explicit_and_preserves_saved_complete_answer(self):
        process = Process(events(chunk({'content': 'A' * 100}, 'stop'), {'type': 'done'}))
        result, _ = self.reply(process)
        short = self.client.read('owner-a', result['id'], limit=12)
        self.assertTrue(short['result_truncated'])
        complete = self.client.read('owner-a', result['id'], limit=32768)
        self.assertFalse(complete['result_truncated'])
        self.assertEqual(json.loads(complete['result'])['final'], 'A' * 100)

    def test_invalid_input_is_rejected_before_starting_a_transport(self):
        with patch('gemini_client.subprocess.Popen') as popen:
            for parameters in [dict(owner='', task=TASK), dict(owner='a', task=''),
                               dict(owner='a', task=TASK, max_tokens=True),
                               dict(owner='a', task=TASK, max_tokens=2049),
                               dict(owner='a', task='{"protocol":"broken"}', role=ROLE)]:
                with self.assertRaises((ValueError, LocalError)):
                    self.client.submit(**parameters)
        popen.assert_not_called()

    @unittest.skipUnless(shutil.which('node'), 'Node is unavailable')
    def test_node_transport_decrypts_fixture_in_memory_and_parses_split_sse(self):
        # Patch only built-in IO in this isolated subprocess. No real credential
        # is read and no socket can be opened by this fixture.
        transport = Path(__file__).resolve().parents[1] / 'gemini_transport.mjs'
        program = r'''
import fs from 'node:fs';
import os from 'node:os';
import crypto from 'node:crypto';
import https from 'node:https';
import {EventEmitter} from 'node:events';
const expectedTimeout=EXPECTED_TIMEOUT;
const originalTimer=globalThis.setTimeout;
let timers=0;
globalThis.setTimeout=(callback,delay,...args)=>{
  if(expectedTimeout===0||delay!==expectedTimeout)throw Error('unexpected response deadline');
  timers++;
  return originalTimer(callback,delay,...args);
};
const secret='offline-fixture-never-a-real-key';
const derived=crypto.scryptSync('gemini-cli-oauth',`${os.hostname()}-${os.userInfo().username}-gemini-cli`,32);
const iv=Buffer.alloc(12,1), cipher=crypto.createCipheriv('aes-256-gcm',derived,iv);
const store={'gemini-cli-api-key':{'default-api-key':JSON.stringify({token:{accessToken:secret}})}};
const ciphertext=Buffer.concat([cipher.update(JSON.stringify(store)),cipher.final()]);
const encoded=[iv.toString('hex'),cipher.getAuthTag().toString('hex'),ciphertext.toString('hex')].join(':');
const originalRead=fs.readFileSync;
fs.readFileSync=(p,...args)=>String(p).endsWith('/.gemini/gemini-credentials.json')?encoded:originalRead(p,...args);
https.request=(options,callback)=>{
  if(options.timeout!==Math.min(30000,expectedTimeout)||timers!==(expectedTimeout===0?0:1))
    throw Error('incorrect timeout policy');
  if(options.hostname!=='generativelanguage.googleapis.com'||options.path!=='/v1beta/openai/chat/completions'||
     options.headers.Authorization!==`Bearer ${secret}`||options.method!=='POST')throw Error('unsafe target');
  const request=new EventEmitter();request.destroy=()=>{};
  request.end=body=>setImmediate(()=>{
    const response=new EventEmitter();response.statusCode=200;response.setEncoding=()=>{};callback(response);
    const chunks=[
      {choices:[{index:0,delta:{tool_calls:[{index:0,id:'original-id',type:'function',
        function:{name:'workspace',arguments:'{"operation":"list"}'},
        extra_content:{google:{thought_signature:'c2lnbmF0dXJl'}}}]},finish_reason:null}]},
      {choices:[{index:0,delta:{},finish_reason:'tool_calls'}]}
    ];
    const stream=chunks.map(c=>'data: '+JSON.stringify(c)+'\r\n\r\n').join('')+'data: [DONE]\r\n\r\n';
    // Split inside CRLF delimiters as well as JSON to exercise buffering.
    for(let i=0;i<stream.length;i+=7)response.emit('data',stream.slice(i,i+7));
    response.emit('end');
  });
  return request;
};
await import(TRANSPORT_URL);
'''.replace('TRANSPORT_URL', json.dumps(transport.as_uri()))
        for timeout in (None, 0, 75):
            with self.subTest(timeout=timeout):
                payload = request_body(TASK, 128, ROLE)
                if timeout is not None:
                    payload['_ostadix_timeout_ms'] = timeout
                source = program.replace('EXPECTED_TIMEOUT', str(120000 if timeout is None else timeout))
                response = subprocess.run([shutil.which('node'), '--input-type=module', '-e', source],
                    input=json.dumps(payload), capture_output=True, text=True, timeout=5)
                self.assertEqual(response.returncode, 0, response.stderr)
                self.assertNotIn('offline-fixture-never-a-real-key', response.stdout)
                frames = [json.loads(line) for line in response.stdout.splitlines()]
                self.assertEqual([frame['type'] for frame in frames], ['chunk', 'chunk', 'done'])
                call = frames[0]['chunk']['choices'][0]['delta']['tool_calls'][0]
                self.assertEqual(call['extra_content']['google']['thought_signature'], SIGNATURE)
                self.assertEqual(call['id'], 'original-id')

    @unittest.skipUnless(shutil.which('node'), 'Node is unavailable')
    def test_node_quota_emits_only_bounded_retry_delay_without_error_body(self):
        transport = Path(__file__).resolve().parents[1] / 'gemini_transport.mjs'
        program = r'''
import fs from 'node:fs';import os from 'node:os';import crypto from 'node:crypto';
import https from 'node:https';import {EventEmitter} from 'node:events';
const fixture=FIXTURE;
const derived=crypto.scryptSync('gemini-cli-oauth',`${os.hostname()}-${os.userInfo().username}-gemini-cli`,32);
const iv=Buffer.alloc(12,2), cipher=crypto.createCipheriv('aes-256-gcm',derived,iv);
const store={'gemini-cli-api-key':{'default-api-key':JSON.stringify({token:{accessToken:'fixture-key-never-real'}})}};
const encrypted=Buffer.concat([cipher.update(JSON.stringify(store)),cipher.final()]);
const encoded=[iv.toString('hex'),cipher.getAuthTag().toString('hex'),encrypted.toString('hex')].join(':');
const originalRead=fs.readFileSync;
fs.readFileSync=(p,...args)=>String(p).endsWith('/.gemini/gemini-credentials.json')?encoded:originalRead(p,...args);
https.request=(options,callback)=>{
  if(options.hostname!=='generativelanguage.googleapis.com')throw Error('unsafe target');
  const request=new EventEmitter();request.destroy=()=>{};
  request.end=()=>setImmediate(()=>{
    const response=new EventEmitter();response.statusCode=429;response.headers=fixture.headers;
    response.setEncoding=()=>{};callback(response);
    response.emit('data',fixture.body);response.emit('end');
  });return request;
};
await import(TRANSPORT_URL);
'''.replace('TRANSPORT_URL', json.dumps(transport.as_uri()))
        cases = [
            ({'retry-after': '2.5'}, 'SECRET_BODY_NEVER_PRINT', 2.5, {}),
            ({}, json.dumps({'error': {'message': 'SECRET_BODY_NEVER_PRINT', 'details': [
                {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '4.125s'}]}}), 4.125, {}),
            ({'retry-after': '900'}, 'SECRET_BODY_NEVER_PRINT', 60, {}),
            ({}, 'SECRET_BODY_NEVER_PRINT', None, {}),
            ({}, 'SECRET_BODY_NEVER_PRINT' * 1800, None, {}),
            ({'retry-after': '7'}, json.dumps({'error': {'details': [
                {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [
                    {'quotaId': 'GenerateRequestsPerMinutePerProject-SECRET', 'quotaValue': '15',
                     'quotaMetric': 'SECRET_METRIC', 'subject': 'SECRET_ACCOUNT'},
                    {'quotaId': 'GenerateRequestsPerDayPerProject-SECRET', 'quotaValue': '20'}]}]}}),
                7, {'quota_scope': 'day', 'quota_value': 20}),
            ({}, json.dumps({'error': {'details': [
                {'@type': 'type.googleapis.com/google.rpc.QuotaFailure', 'violations': [
                    {'quotaId': 'RequestsPerMinute-SECRET', 'quotaValue': '0'}]}]}}),
                None, {'quota_scope': 'minute', 'quota_value': 0}),
            ({}, json.dumps({'error': {'details': [
                {'@type': 'unknown.QuotaFailure', 'violations': [
                    {'quotaId': 'RequestsPerDay-SECRET', 'quotaValue': '99'}]}]}}), None, {}),
        ]
        for headers, body, expected, quota in cases:
            with self.subTest(expected=expected, oversized=len(body)>32768):
                source = program.replace('FIXTURE', json.dumps({'headers': headers, 'body': body}))
                response = subprocess.run([shutil.which('node'), '--input-type=module', '-e', source],
                    input=json.dumps(request_body(TASK, 128, ROLE)), capture_output=True, text=True, timeout=5)
                self.assertEqual(response.returncode, 1)
                frame = json.loads(response.stdout.strip())
                self.assertEqual(frame['type'], 'error')
                self.assertEqual(frame['code'], 'http_429')
                self.assertEqual(frame.get('retry_seconds'), expected)
                self.assertEqual({k: v for k, v in frame.items() if k.startswith('quota_')}, quota)
                self.assertLessEqual(set(frame), {'type', 'code', 'retry_seconds', 'quota_scope', 'quota_value'})
                self.assertNotIn('SECRET', response.stdout)
                self.assertEqual(response.stderr, '')


if __name__ == '__main__':
    unittest.main()
