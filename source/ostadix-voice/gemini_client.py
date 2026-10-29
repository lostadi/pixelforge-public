"""Bounded Gemini jobs with native host tools and an in-memory credential transport."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid

from conversation_protocol import NATIVE_GUIDE, decode_conversation, is_main_role
from local_runtime import LocalError, native_tool_response


MODEL = 'gemini-3.8-flash'
MODELS = {'gemini-3.8-flash': 'Gemini 3.8 Flash', 'gemini-3.7-flash': 'Gemini 3.7 Flash',
          'gemini-4-argon': 'Gemini 4 Argon', 'gemini-2.5-flash': 'Gemini 2.5 Flash'}
TRANSPORT = Path(__file__).with_name('gemini_transport.mjs')
TERMINAL = frozenset(('completed', 'failed', 'cancelled', 'timed_out'))
MAX_JOBS = 64
MAX_TEXT = 32768
MAX_ARGUMENTS = 16000
MAX_METADATA = 16000


def request_body(task, max_tokens, role, model=MODEL):
    """Use the same validated transcript/tool declarations as the local agent."""
    from local_runtime import CONVERSATION_TOOLS
    messages = decode_conversation(task, role)
    native = messages is not None
    if messages is None:
        messages = [{'role': 'user', 'content': task}]
    if model not in MODELS:
        raise LocalError('Unsupported Gemini model')
    body = {'model': model, 'messages': [{'role': 'system', 'content': NATIVE_GUIDE if native else
            'Answer the user briefly. You have no tools in this helper request.'}, *messages],
            'stream': True, 'reasoning_effort': 'low',
            # Thinking and visible output share Gemini's completion budget.
            'max_completion_tokens': min(4096, max(2048, max_tokens + 1024))}
    if native:
        tools = copy.deepcopy(CONVERSATION_TOOLS)
        required = role == 'phone OSTADIX action assistant'
        choice = {'type': 'function', 'function': {'name': 'ostadix_execute'}} if required else 'auto'
        body.update(tools=tools, tool_choice=choice, parallel_tool_calls=False)
    return body


class GeminiClient:
    engine_name = 'gemini'
    engine_label = 'Gemini 3.8 Flash'
    output_limit = 2048
    requires_native_tool_calls = True

    def __init__(self, *, node=None, model=MODEL, quota_fallback=False):
        if model not in MODELS:
            raise LocalError('Unsupported Gemini model')
        self.model = model
        self.engine_label = MODELS[model]
        self.quota_fallback = quota_fallback
        termux_node = Path('/data/data/com.termux/files/usr/bin/node')
        self.node = str(node or (termux_node if termux_node.exists() else shutil.which('node') or 'node'))
        self.condition = threading.Condition(threading.RLock())
        self._jobs = {}
        self._queue = []
        self._closed = False
        self._worker = None

    def status(self):
        # Passive status: never decrypt, authenticate, start a daemon or contact Google.
        with self.condition:
            available = Path.home().joinpath('.gemini/gemini-credentials.json').is_file()
            return {'running': not self._closed, 'ready': available and not self._closed,
                    'backend': 'google-gemini-api', 'engine': self.engine_name,
                    'active': sum(job['state'] == 'running' for job in self._jobs.values()),
                    'queued': len(self._queue), 'credential_present': available,
                    'error': None if available else 'Saved Gemini credential is unavailable',
                    'limits': {'active': 1, 'queued': 8, 'max_tokens': self.output_limit},
                    'model': {'model_name': self.engine_label, 'model_id': self.model,
                              'backend': 'google-gemini-api', 'context_length': 32768,
                              'context_per_worker': 32768,
                              'max_output_tokens': self.output_limit,
                              'native_conversation_protocol': 1, 'streaming_preview': True}}

    def submit(self, owner, task, max_tokens=512, role='assistant'):
        if not isinstance(owner, str) or not owner or len(owner) > 256:
            raise LocalError('Gemini owner must be bounded text')
        if not isinstance(task, str) or not task.strip() or len(task) > 24000:
            raise LocalError('Gemini task must contain 1 to 24000 characters')
        if type(max_tokens) is not int or not 1 <= max_tokens <= self.output_limit:
            raise LocalError('Gemini output limit must be between 1 and 2048 tokens')
        if not isinstance(role, str) or not role or len(role) > 128:
            raise LocalError('Gemini role must be bounded text')
        body = request_body(task, max_tokens, role, self.model)
        with self.condition:
            if self._closed:
                raise LocalError('Gemini client is closed')
            if len(self._queue) >= 8:
                raise LocalError('Gemini queue is full; wait for the current request')
            while len(self._jobs) >= MAX_JOBS:
                old = next((ident for ident, job in self._jobs.items() if job['state'] in TERMINAL), None)
                if old is None:
                    raise LocalError('Gemini job limit reached')
                del self._jobs[old]
            ident = uuid.uuid4().hex
            job = {'id': ident, 'owner': owner, 'state': 'queued', 'partial': '', 'result': '',
                   'error': None, 'native_call': None, 'finish_reason': None, 'truncated': False,
                   'result_truncated': False, 'created': time.time(), '_body': body,
                   'stage': 'queued', 'retry_count': 0, 'retry_after_seconds': None,
                   '_process': None, '_calls': {}, '_text': '', '_received': False,
                   '_timeout_seconds': None if is_main_role(role) else 120,
                   '_cancel': threading.Event()}
            self._jobs[ident] = job
            self._queue.append(ident)
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(target=self._work, name='gemini-network', daemon=True)
                try:
                    self._worker.start()
                except Exception:
                    self._queue.remove(ident)
                    del self._jobs[ident]
                    self._worker = None
                    raise LocalError('Gemini request worker could not start') from None
            self.condition.notify_all()
            return self._snapshot(job, 4000)

    @staticmethod
    def _snapshot(job, limit):
        result = {key: copy.deepcopy(value) for key, value in job.items()
                  if not key.startswith('_') and key != 'owner'}
        for field in ('result', 'partial'):
            if len(result[field]) > limit:
                result[field] = result[field][:limit]
                result['result_truncated'] = True
        return result

    def _owned(self, owner, ident):
        job = self._jobs.get(ident)
        if job is None or job['owner'] != owner:
            raise LocalError('Gemini job was not found for this owner')
        return job

    def read(self, owner, job_id, wait=0, limit=4000):
        if type(wait) not in (int, float) or not 0 <= wait <= 120:
            raise LocalError('Gemini wait must be between 0 and 120 seconds')
        if type(limit) is not int or not 1 <= limit <= 32768:
            raise LocalError('Gemini result limit must be between 1 and 32768 characters')
        deadline = time.monotonic() + wait
        with self.condition:
            job = self._owned(owner, job_id)
            while job['state'] not in TERMINAL and time.monotonic() < deadline:
                self.condition.wait(deadline - time.monotonic())
            return self._snapshot(job, limit)

    @staticmethod
    def _terminate(process):
        if process is not None:
            try:
                process.terminate()
                try:
                    process.wait(timeout=.3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1)
            except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
                pass

    def cancel(self, owner, job_id=None):
        processes = []
        with self.condition:
            jobs = [self._owned(owner, job_id)] if job_id is not None else [
                job for job in self._jobs.values() if job['owner'] == owner]
            changed = []
            for job in jobs:
                if job['state'] not in TERMINAL:
                    job['_cancel'].set()
                    job.update(state='cancelled', stage='cancelled', retry_after_seconds=None,
                               partial='', result='', native_call=None)
                    if job['id'] in self._queue:
                        self._queue.remove(job['id'])
                    processes.append(job['_process'])
                    changed.append(job['id'])
            self.condition.notify_all()
        for process in processes:
            self._terminate(process)
        return {'cancelled': changed}

    def _work(self):
        while True:
            with self.condition:
                if self._closed or not self._queue:
                    self._worker = None
                    return
                job = self._jobs[self._queue.pop(0)]
                if job['state'] != 'queued':
                    continue
                job.update(state='running', stage='requesting')
                self.condition.notify_all()
            self._run(job)

    def _chunk(self, job, chunk):
        """Assemble only public answer/tool fields; ignore reasoning content."""
        choices = chunk.get('choices', [])
        if not isinstance(choices, list):
            raise ValueError('choices')
        for choice in choices:
            if not isinstance(choice, dict) or choice.get('index', 0) != 0:
                continue
            delta = choice.get('delta') or {}
            if not isinstance(delta, dict):
                raise ValueError('delta')
            content = delta.get('content')
            if content is not None:
                if not isinstance(content, str) or len(job['_text']) + len(content) > MAX_TEXT:
                    raise ValueError('content limit')
                job['_text'] += content
                if content:
                    job['_received'] = True
            for fragment in delta.get('tool_calls') or []:
                job['_received'] = True
                if not isinstance(fragment, dict) or type(fragment.get('index', 0)) is not int:
                    raise ValueError('tool fragment')
                index = fragment.get('index', 0)
                if index != 0:
                    raise ValueError('parallel tool calls')
                call = job['_calls'].setdefault(index, {'id': '', 'type': 'function',
                                                       'function': {'name': '', 'arguments': ''}})
                if 'id' in fragment:
                    if not isinstance(fragment['id'], str):
                        raise ValueError('tool id')
                    call['id'] += fragment['id']
                if 'type' in fragment and fragment['type'] != 'function':
                    raise ValueError('tool type')
                function = fragment.get('function') or {}
                if not isinstance(function, dict):
                    raise ValueError('tool function')
                for field in ('name', 'arguments'):
                    if field in function:
                        if not isinstance(function[field], str):
                            raise ValueError('tool field')
                        call['function'][field] += function[field]
                if len(call['id']) > 128 or len(call['function']['name']) > 128 or len(call['function']['arguments']) > MAX_ARGUMENTS:
                    raise ValueError('tool limit')
                if 'extra_content' in fragment:
                    metadata = fragment['extra_content']
                    if len(json.dumps(metadata)) > MAX_METADATA:
                        raise ValueError('metadata limit')
                    if 'extra_content' in call and call['extra_content'] != metadata:
                        raise ValueError('conflicting metadata')
                    call['extra_content'] = copy.deepcopy(metadata)
            if choice.get('finish_reason') is not None:
                job['finish_reason'] = choice['finish_reason']
        # No pre-action prose is shown while a function call is being assembled.
        job['partial'] = '' if job['_calls'] else job['_text']
        usage = chunk.get('usage')
        if isinstance(usage, dict):
            job['usage'] = {key: value for key, value in usage.items()
                            if key in ('prompt_tokens', 'completion_tokens', 'total_tokens') and type(value) is int}

    def _complete(self, job):
        reason = job['finish_reason']
        if reason == 'length':
            if not job['_calls'] and job['_text'].strip():
                # A completed text stream is useful even when its token budget
                # was exhausted. Keep it explicit; function arguments still
                # require a complete, validated tool-call finish below.
                answer = job['_text'] + ('\n\n[This answer is incomplete because it reached the model output limit. '
                                        'Earlier tool results remain saved and can be retrieved.]')
                job.update(truncated=True, answer_incomplete=True, error=None,
                           result=json.dumps({'final': answer}, ensure_ascii=False, separators=(',', ':')),
                           partial=job['_text'], state='completed', stage='completed')
                return
            job.update(truncated=True, partial='', error='Gemini reached its output limit; no partial action was executed',
                       state='failed', stage='failed')
            return
        if reason not in ('stop', 'tool_calls'):
            raise ValueError('incomplete or blocked response')
        if job['_calls']:
            if reason != 'tool_calls' or len(job['_calls']) != 1:
                raise ValueError('tool completion')
            call = job['_calls'][0]
            if not re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', call['id']):
                raise ValueError('tool id')
            # Validate metadata through the same strict typed protocol used on replay.
            from conversation_protocol import encode_conversation
            encode_conversation([{'role': 'user', 'content': 'Validate native call'},
                                 {'role': 'assistant', 'content': None, 'tool_calls': [call]},
                                 {'role': 'tool', 'tool_call_id': call['id'], 'content': '{}'}])
            if self.model.startswith(('gemini-3', 'gemini-4')) and not call.get('extra_content', {}).get('google', {}).get('thought_signature'):
                raise ValueError('missing thought signature')
            job['result'] = native_tool_response(call['function']['name'], call['function']['arguments'])
            job['native_call'] = copy.deepcopy(call)
            job['partial'] = ''
        else:
            if reason == 'tool_calls' or not job['_text'].strip():
                raise ValueError('empty response')
            job['result'] = json.dumps({'final': job['_text']}, ensure_ascii=False, separators=(',', ':'))
        job.update(state='completed', stage='completed')

    def _quota_wait(self, job, seconds):
        return job['_cancel'].wait(seconds)

    def _run(self, job):
        seconds = job['_timeout_seconds']
        deadline = None if seconds is None else time.monotonic() + seconds
        try:
            for attempt in range(2):
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    with self.condition:
                        if not job['_cancel'].is_set():
                            job.update(state='timed_out', stage='timed_out', error='Gemini request timed out', partial='')
                    return
                # Zero disables both the transport deadline and socket idle timer.
                self._attempt(job, 0 if remaining is None else max(1, int(min(120, remaining) * 1000)))
                with self.condition:
                    if (attempt or self.quota_fallback or self._closed or job['_cancel'].is_set()
                            or job.get('error_code') != 'http_429' or job['_received']
                            or job.get('quota_scope') == 'day'):
                        return
                    delay = job.get('retry_after_seconds')
                    if type(delay) not in (int, float) or not 0 <= delay <= 60:
                        delay = 60
                    if deadline is not None and deadline - time.monotonic() <= delay + 1:
                        job.update(state='failed', stage='failed')
                        return
                    job.update(state='running', stage='quota_wait', partial='', error=None,
                               retry_after_seconds=delay, retry_count=1)
                    self.condition.notify_all()
                if self._quota_wait(job, delay):
                    return
                with self.condition:
                    if self._closed or job['_cancel'].is_set():
                        return
                    job.update(stage='requesting', retry_after_seconds=None, error=None,
                               finish_reason=None, native_call=None, result='', partial='')
                    job.pop('error_code', None)
                    job.pop('transport_code', None)
                    job.pop('quota_scope', None)
                    job.pop('quota_value', None)
        finally:
            with self.condition:
                job['_body'] = None
                if job['state'] in TERMINAL:
                    job['stage'] = job['state']
                self.condition.notify_all()

    def _attempt(self, job, timeout_ms):
        process = None
        try:
            with self.condition:
                if job['_cancel'].is_set() or self._closed:
                    return
                process = subprocess.Popen([self.node, str(TRANSPORT)], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, encoding='utf-8',
                    start_new_session=True)
                job['_process'] = process
            payload = dict(job['_body'], _ostadix_timeout_ms=timeout_ms)
            process.stdin.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + '\n')
            process.stdin.close()
            done = False
            while True:
                line = process.stdout.readline(131073)
                if not line:
                    break
                if len(line) > 131072 or not line.endswith('\n'):
                    raise ValueError('transport frame limit')
                event = json.loads(line)
                with self.condition:
                    if job['_cancel'].is_set() or self._closed:
                        return
                    if event.get('type') == 'chunk':
                        self._chunk(job, event['chunk'])
                    elif event.get('type') == 'done':
                        done = True
                    elif event.get('type') == 'error':
                        # Transport emits only fixed, credential-free error categories.
                        code = event.get('code')
                        labels = {'credentials': 'Saved Gemini credential could not be opened',
                                  'timeout': 'Gemini request timed out', 'network': 'Gemini network request failed',
                                  'response': 'Gemini returned an incomplete or invalid stream',
                                  'request': 'Gemini transport could not prepare the request',
                                  'http_400': 'Gemini rejected the request (HTTP 400)',
                                  'http_401': 'Gemini authentication failed (HTTP 401)',
                                  'http_403': 'Gemini access was denied (HTTP 403)',
                                  'http_404': 'Gemini model was not found (HTTP 404)',
                                  'http_429': 'Gemini rate or quota limit reached (HTTP 429)',
                                  'http_500': 'Gemini service failed (HTTP 500); the request was not replayed',
                                  'http_502': 'Gemini gateway failed (HTTP 502); the request was not replayed',
                                  'http_503': 'Gemini model is unavailable (HTTP 503); the request was not replayed',
                                  'http_504': 'Gemini gateway timed out (HTTP 504); the request was not replayed'}
                        may_retry = (code == 'http_429' and not self.quota_fallback and not job['_received']
                                     and job['retry_count'] == 0 and event.get('quota_scope') != 'day')
                        state = 'running' if may_retry else ('timed_out' if code == 'timeout' else 'failed')
                        job.update(state=state, stage=state if state in TERMINAL else job['stage'], partial='',
                                   error=labels.get(code, 'Gemini request failed'))
                        job['safe_quota_fallback'] = code == 'http_429' and not job['_received']
                        if isinstance(code, str) and (code in labels or re.fullmatch(r'http_[1-5][0-9]{2}', code)):
                            job['transport_code'] = code
                            job['error_code'] = code
                        delay = event.get('retry_seconds')
                        if code == 'http_429' and type(delay) in (int, float) and 0 <= delay <= 60:
                            job['retry_after_seconds'] = delay
                        if code == 'http_429' and event.get('quota_scope') in ('minute', 'day'):
                            job['quota_scope'] = event['quota_scope']
                            value = event.get('quota_value')
                            if type(value) is int and 0 <= value <= 1000000000000:
                                job['quota_value'] = value
                            if job['quota_scope'] == 'day':
                                job['error'] = 'Gemini daily quota reached; wait for the quota reset or use the local engine'
                        return
                    else:
                        raise ValueError('transport event')
                    self.condition.notify_all()
            if process.wait(timeout=3) != 0 or not done:
                raise ValueError('incomplete stream')
            with self.condition:
                if not job['_cancel'].is_set() and not self._closed:
                    self._complete(job)
        except Exception:
            with self.condition:
                if not job['_cancel'].is_set() and not self._closed:
                    job.update(state='failed', stage='failed', error='Gemini returned an invalid or incomplete response; no action was executed',
                               partial='', result='', native_call=None)
        finally:
            if process is not None:
                self._terminate(process)
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
                for stream in (process.stdin, process.stdout):
                    if stream is not None and not stream.closed:
                        stream.close()
            with self.condition:
                job['_process'] = None
                job['_calls'] = {}
                job['_text'] = ''
                self.condition.notify_all()

    def close(self):
        with self.condition:
            if self._closed:
                return
            owners = {job['owner'] for job in self._jobs.values() if job['state'] not in TERMINAL}
            self._closed = True
        for owner in owners:
            self.cancel(owner)
        worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(5)
