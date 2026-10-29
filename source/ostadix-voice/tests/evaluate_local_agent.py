"""Small, real-model usability diagnostic with an isolated, read-only executor.

Run explicitly with --run. This never starts/switches models, changes the phone,
opens a microphone, or touches the default conversation database. These five
short cases are a regression sample, not a comprehensive model benchmark.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import secrets
import struct
import sys
import tempfile
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import AppError, Store
from local_chat import LocalChatAgent, parse_response
from local_runtime import LocalClient


PROGRAM = 'python^(\n__oval_result__ = (23 * 17) - 8\n)_python\n'
CASES = (
    ('smalltalk', 'Hi, how are you? Reply in one short sentence.'),
    ('file_question', "What's the release code in release-notes.txt?"),
    ('followup', 'What was that release code again? Give only the code.'),
    ('phone_query', 'What is the current media volume and maximum? Just report the numbers; do not change it.'),
    ('ostadix_program', 'Execute this complete program through OSTADIX and report the real returned value:\n' + PROGRAM),
)


def numeric_expression(source):
    """Extract one expression without executing generated code."""
    if not isinstance(source, str) or len(source) > 512:
        return None
    block = re.fullmatch(r'\s*python\^\((.*?)\)_python\s*', source, re.S)
    try:
        if block:
            tree = ast.parse(block[1].strip())
            if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign)
                    or len(tree.body[0].targets) != 1
                    or not isinstance(tree.body[0].targets[0], ast.Name)
                    or tree.body[0].targets[0].id != '__oval_result__'):
                return None
            tree = tree.body[0].value
        else:
            tree = ast.parse(source.strip(), mode='eval').body
    except (SyntaxError, ValueError):
        return None
    return tree


def harmless_source(source):
    """Admit only one bounded numeric expression, optionally in O's Python block."""
    tree = numeric_expression(source)
    if tree is None:
        return False
    nodes = list(ast.walk(tree))
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
               ast.UAdd, ast.USub)
    return (len(nodes) <= 32 and all(isinstance(node, allowed) for node in nodes)
            and all(type(node.value) is int and abs(node.value) <= 10 ** 6
                    for node in nodes if isinstance(node, ast.Constant)))


def permit(arguments):
    """The live diagnostic cannot execute generated shell or mutate user state."""
    if arguments.get('operation') == 'read':
        return arguments.get('path') == 'release-notes.txt'
    if arguments.get('operation') == 'list':
        return arguments.get('path', '.') in ('.', '')
    if arguments.get('operation') != 'command':
        return False
    argv = arguments.get('argv')
    if not isinstance(argv, list):
        return False
    if argv in (['@phone', 'help'], ['@phone', 'status'], ['@phone', 'volume'],
                ['@ostadix', 'status'], ['@ostadix', 'tools'],
                ['@ostadix', 'schema', 'o_execute']):
        return True
    if len(argv) != 4 or argv[:3] != ['@ostadix', 'call', 'o_execute']:
        return False
    try:
        request = json.loads(argv[3])
    except (ValueError, TypeError):
        return False
    return (isinstance(request, dict) and set(request) <= {'source', 'action', 'timeout_secs'}
            and request.get('action') in ('execute', 'check')
            and harmless_source(request.get('source'))
            and type(request.get('timeout_secs', 20)) is int
            and 1 <= request.get('timeout_secs', 20) <= 30)


def native_value(value):
    if not isinstance(value, dict):
        return None
    kind, content = value.get('t'), value.get('v')
    if kind == 'text':
        return content.get('utf8')
    if kind == 'number' and isinstance(content, dict):
        if content.get('kind') == 'int':
            return int(content['v'])
        if content.get('kind') == 'binary_float':
            return struct.unpack('>d' if content['format'] == 'f64' else '>f', bytes(content['bits']))[0]
    return content


class MeasuredClient:
    """Keep the real service, bounded calls, and raw model replies for review."""
    def __init__(self, client=None):
        self.client = client or LocalClient()
        self.calls = 0
        self.replies = {}
        self.failures = []
        self.first_preview = None

    def status(self):
        return self.client.status()

    def submit(self, owner, task, max_tokens, role):
        self.calls += 1
        if self.calls > 4:
            raise AppError('Diagnostic stopped this case after four model requests')
        return self.client.submit(owner, task, max_tokens=min(256, max_tokens), role=role)

    def read(self, owner, ident, wait, limit):
        result = self.client.read(owner, ident, wait=wait, limit=limit)
        if result.get('partial') and self.first_preview is None:
            self.first_preview = time.monotonic()
        if result.get('state') == 'completed':
            self.replies[ident] = result.get('result')
        elif result.get('state') not in ('queued', 'running'):
            self.failures.append({key: result.get(key) for key in ('state', 'error', 'error_code', 'finish_reason')})
        return result

    def cancel(self, owner, ident):
        return self.client.cancel(owner, ident)

    def close(self):
        close = getattr(self.client, 'close', None)
        if callable(close):
            close()


def evaluate(output, selected=None, client=None):
    client = MeasuredClient(client)
    status = client.status()
    if not status.get('ready') or status.get('active') or status.get('queued'):
        raise AppError('Main model must already be ready and idle; diagnostic did not queue work')
    if (status.get('model') or {}).get('native_conversation_protocol') != 1:
        raise AppError('Diagnostic requires the actual native-conversation daemon; no inference was submitted')
    selected = set(selected or [name for name, _ in CASES])
    if 'followup' in selected:
        selected.add('file_question')
    report = {'schema': 'ostadix.agent-usability.v1', 'scope': __doc__.strip(),
              'model': {key: (status.get('model') or {}).get(key)
                        for key in ('model_name', 'backend', 'context_per_worker',
                                    'native_conversation_protocol', 'streaming_preview')},
              'sources_sha256': {name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
                                 for name in ('local_chat.py', 'local_runtime.py')},
              'output_token_cap': 256, 'cases': [], 'passed': False,
              'latency_note': 'First case uses the freshly restarted daemon; later cases reuse warmed state. First-token time measures the first available answer-text preview, excluding hidden function-call tokens.'}
    def save():
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with tempfile.TemporaryDirectory(prefix='ostadix-usability-') as directory:
        root = Path(directory) / 'workspace'; root.mkdir()
        code = 'EVAL-' + secrets.token_hex(4).upper()
        notes = 'Release notes:\n' + ''.join(
            f'- Review item {number:02}: documentation changes only; no release identifier on this line.\n'
            for number in range(30)) + 'Release code: ' + code + '\n'
        (root / 'release-notes.txt').write_text(notes)
        report['fixture'] = {'bytes': len(notes.encode()), 'release_code': code,
                             'code_byte_offset': notes.encode().index(code.encode()),
                             'purpose': 'Identifier follows ordinary notes, beyond a 1400-character result preview.'}
        store = Store(Path(directory) / 'state'); sid = store.create(root)
        agent = LocalChatAgent(store, root, client=client)
        agent.set_action_policy('auto')
        original = agent.execute
        def read_only(arguments):
            if not permit(arguments):
                raise AppError('Diagnostic rejected an unrelated or mutating action; nothing was executed')
            return original(arguments)
        agent.execute = read_only
        try:
            for name, prompt in CASES:
                if name not in selected:
                    continue
                before = max([event['seq'] for event in store.events(sid)] or [0])
                client.calls = 0; client.replies = {}; client.failures = []; client.first_preview = None
                started = time.monotonic()
                agent.submit(sid, prompt, str(uuid.uuid4()))
                agent.start_thread.join(90)
                timed_out = agent.start_thread.is_alive()
                if timed_out:
                    agent.cancel(sid); agent.start_thread.join(10)
                events = store.events(sid, after=before)
                tools = [event['data'] for event in events if event['kind'] == 'tool']
                results = [event['data'] for event in events if event['kind'] == 'tool_result']
                answer = agent.last_answer
                checks = {'completed': not timed_out and store.session(sid)['status'] == 'idle',
                          'nonempty_answer': bool(answer), 'no_failed_tools': all(item['success'] for item in results)}
                if name == 'smalltalk':
                    checks.update(no_unnecessary_tools=not tools, concise=len(answer) <= 250)
                elif name in ('file_question', 'followup'):
                    checks['actual_code_in_answer'] = code in answer
                    if name == 'file_question':
                        checks['read_fixture'] = any(item['arguments'].get('operation') == 'read'
                                                    and item['arguments'].get('path') == 'release-notes.txt' for item in tools)
                elif name == 'phone_query':
                    actual = [item['result'] for item in results if item['success']
                              and 'volume' in item['result'] and 'max_volume' in item['result']]
                    numbers = set(re.findall(r'(?<!\d)\d+(?!\d)', answer))
                    checks['read_real_media_volume'] = bool(actual)
                    checks['actual_numbers_in_answer'] = bool(actual) and all(str(actual[-1][key]) in numbers for key in ('volume', 'max_volume'))
                elif name == 'ostadix_program':
                    native = [(item['result'].get('result') or {}).get('structuredContent', {}) for item in results
                              if item['success'] and item['result'].get('mcp_tool') == 'o_execute']
                    checks['actual_native_execution'] = any(job.get('state') == 'completed' and job.get('exit_code') == 0
                        and (job.get('result') or {}).get('ok') is True
                        and native_value((job.get('result') or {}).get('value')) == 383 for job in native)
                    checks['correct_final_value'] = bool(re.search(r'(?<!\d)383(?!\d)', answer))
                    requested = ast.dump(numeric_expression(PROGRAM))
                    exact_sources = []
                    for tool in tools:
                        argv = tool['arguments'].get('argv') or []
                        if len(argv) == 4 and argv[:3] == ['@ostadix', 'call', 'o_execute']:
                            try:
                                call = json.loads(argv[3]); expression = numeric_expression(call.get('source'))
                                exact_sources.append(call.get('action') == 'execute' and expression is not None
                                                     and ast.dump(expression) == requested)
                            except (ValueError, TypeError):
                                pass
                    checks['requested_program_executed'] = any(exact_sources)
                raw = list(client.replies.values())
                last_raw_answer = None
                if raw:
                    try: _, last_raw_answer = parse_response(raw[-1])
                    except AppError: pass
                if name == 'ostadix_program':
                    checks['model_grounded_without_host_replacement'] = last_raw_answer == answer
                case = {'id': name, 'prompt': prompt, 'seconds': round(time.monotonic() - started, 3),
                        'first_token_seconds': round(client.first_preview - started, 3) if client.first_preview is not None else None,
                        'model_requests': client.calls, 'status': store.session(sid)['status'],
                        'answer': answer, 'raw_model_replies': raw,
                        'model_failures': client.failures,
                        'outcomes': [event['data'] for event in events if event['kind'] == 'finished'],
                        'host_replaced_final': last_raw_answer is not None and last_raw_answer != answer,
                        'tools': tools, 'results': results, 'checks': checks, 'passed': all(checks.values())}
                report['cases'].append(case); save()
                print(json.dumps({key: case[key] for key in ('id', 'seconds', 'first_token_seconds', 'model_requests', 'status', 'checks', 'passed')}) , flush=True)
                if agent.start_thread.is_alive():
                    raise AppError('Diagnostic worker has not released after cancellation; no later case was submitted')
                if any(failure.get('error_code') in ('http_401', 'http_403', 'http_429') for failure in client.failures):
                    report['stopped_early'] = 'Provider authentication, access or quota prevents further useful cases.'
                    save()
                    break
        finally:
            agent.close(); store.db.close()
    report['passed'] = all(case['passed'] for case in report['cases'])
    report['manual_review'] = 'Review conversational helpfulness and unsupported claims in the saved answers; mechanical checks cannot establish general quality.'
    save()
    return report['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Run the actual, already-idle local model')
    parser.add_argument('--output', type=Path, default=Path('runtime/repairs/agent-quality-audit-20261022/usability.json'))
    parser.add_argument('--case', action='append', choices=[name for name, _ in CASES])
    args = parser.parse_args()
    if not args.run:
        print(json.dumps({'inference_performed': False, 'cases': [name for name, _ in CASES],
                          'readonly': True, 'max_model_requests_per_case': 4, 'output_tokens': 256}))
        return
    raise SystemExit(0 if evaluate(args.output, args.case) else 1)


if __name__ == '__main__':
    main()
