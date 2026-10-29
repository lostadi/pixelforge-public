#!/usr/bin/env python3
"""Observe one real agent turn investigating this phone, in isolated state.

Run with native Termux Python and its normal app identity. Services must already
be ready and idle. This harness never starts them or changes the user's store.
Model output is admitted only through a narrow read-only command guard.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import tempfile
import threading
import time
import uuid
from unittest.mock import patch


APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))

from agent import AppError, Store
from local_chat import LocalChatAgent
from local_runtime import LocalClient
from npu_chat import NPUConversationClient


QUESTION = "Tell me concrete details about this phone's NPU."
INTERFACE = {
    'available': True, 'interface': 'terminal', 'microphone_muted': True,
    'speaker_muted': True, 'speech_queue': {'queued': 0, 'sending': 1, 'review': 0},
    'engine_health': {'state': 'ready', 'error': None},
}
COMMAND_DIRS = {'/system/bin', '/data/data/com.termux/files/usr/bin'}
SHELL_CHARS = set(';|&<>`$\n\r\x00')


def metadata_path(value):
    """Only kernel/device metadata; no app data, process secrets or raw devices."""
    if not isinstance(value, str) or '\x00' in value or '..' in PurePosixPath(value).parts:
        return False
    return (value == '/sys' or value.startswith('/sys/')
            or value in ('/proc/cpuinfo', '/proc/version', '/proc/meminfo', '/proc/uptime')
            or value == '/proc/device-tree' or value.startswith('/proc/device-tree/'))


def listing_path(value):
    return metadata_path(value) or value == '/dev' or value.startswith('/dev/')


def command_name(value):
    path = PurePosixPath(value)
    if '/' not in value:
        return value
    return path.name if str(path.parent) in COMMAND_DIRS else None


def readonly_argv(argv, *, wrapped=False):
    if (not isinstance(argv, list) or not argv or len(argv) > 100
            or any(not isinstance(value, str) or len(value) > 32000 or '\x00' in value for value in argv)):
        return False
    if argv[0].startswith('@'):
        if wrapped:
            return False
        if argv[0] == '@phone':
            return len(argv) == 2 and argv[1] in ('cpu', 'hardware', 'npu', 'help', 'status')
        if argv[0] in ('@npu', '@voice'):
            return argv[1:] == ['status']
        if argv == ['@results', 'list']:
            return True  # The harness store contains only this evaluation.
        return (len(argv) == 6 and argv[:2] == ['@results', 'read']
                and all(value.isdecimal() for value in argv[2:])
                and 0 < int(argv[-1]) <= 24000) or (
                    len(argv) == 5 and argv[:2] == ['@results', 'read']
                    and all(value.isdecimal() for value in argv[2:])
                    and 0 < int(argv[-1]) <= 24000)
    name, args = command_name(argv[0]), argv[1:]
    if (name in ('python', 'python3') and args == [str(APP / 'runtime/npu/npu_telemetry.py')]):
        return True  # Installed read-only probe advertised by tool discovery; no generated Python code.
    if name in ('sh', 'su'):
        if wrapped:
            return False
        if name == 'su' and args[:1] == ['0']:
            args = args[1:]
        if len(args) != 2 or args[0] != '-c' or any(char in SHELL_CHARS for char in args[1]):
            return False
        try:
            inner = shlex.split(args[1], comments=False, posix=True)
        except ValueError:
            return False
        return readonly_argv(inner, wrapped=True)
    if name == 'getprop':
        return len(args) == 1 and bool(re.fullmatch(r'(?:ro\.|vendor\.|sys\.)[A-Za-z0-9_.-]{1,120}', args[0]))
    if name == 'uname':
        return all(re.fullmatch(r'-[asnrvmpio]+', value) for value in args)
    if name in ('ls', 'readlink'):
        flags = (r'-[aAdfFhiklLnprRstuvxZ1]+|--' if name == 'ls' else r'-[efmnqsvz]+|--')
        paths = []
        for value in args:
            if value.startswith('-'):
                if not re.fullmatch(flags, value):
                    return False
            else:
                paths.append(value)
        return bool(paths) and all(listing_path(value) for value in paths)
    if name == 'cat':
        return bool(args) and all(metadata_path(value) for value in args)
    if name == 'find':
        index, saw_path = 0, False
        while index < len(args):
            value = args[index]
            if listing_path(value):
                saw_path = True
            elif value in ('-L', '-H', '-P', '-print', '-print0', '-a', '-o'):
                pass
            elif value in ('-maxdepth', '-mindepth', '-type', '-name', '-iname', '-path', '-ipath'):
                index += 1
                if index >= len(args):
                    return False
                operand = args[index]
                if value in ('-maxdepth', '-mindepth') and (not operand.isdecimal() or int(operand) > 12):
                    return False
                if value == '-type' and operand not in ('f', 'd', 'l', 'c'):
                    return False
            else:
                return False
            index += 1
        return saw_path
    return False


def check_guard():
    allowed = [
        ['@phone', 'cpu'], ['@npu', 'status'], ['ls', '-l', '/sys/class'],
        ['/system/bin/cat', '/proc/cpuinfo'], ['getprop', 'ro.soc.model'],
        ['find', '/sys/devices', '-maxdepth', '5', '-name', '*tpu*'],
        ['sh', '-c', 'ls -l /sys/class/*tpu*'], ['su', '-c', 'cat /proc/cpuinfo'],
    ]
    denied = [
        ['@phone', 'open', 'app'], ['@npu', 'spawn', '{}'], ['rm', '/sys/foo'],
        ['cat', '/proc/self/environ'], ['cat', '/dev/edgetpu'],
        ['find', '/sys', '-delete'], ['find', '/sys', '-exec', 'sh', '{}', ';'],
        ['sh', '-c', 'cat /proc/cpuinfo > /tmp/probe'],
        ['sh', '-c', 'ls /sys; reboot'], ['su', '-c', 'ls $(reboot)'],
        ['sh', '-c', 'ls /sys\nreboot'], ['/tmp/ls', '/sys'],
    ]
    if not all(readonly_argv(argv) for argv in allowed) or any(readonly_argv(argv) for argv in denied):
        raise RuntimeError('Read-only command guard self-check failed')


class Evidence:
    def __init__(self, output, engine, question):
        self.output = output
        self.lock = threading.RLock()
        self.value = {
            'schema': 'ostadix.autonomous-device-investigation/v1',
            'engine': engine, 'question': question, 'started_utc': utc(),
            'status': 'preflight', 'forced_tools': False, 'injected_hardware_facts': False,
            'canned_answer': False, 'user_store_access': False, 'mutating_actions_allowed': False,
            'interface_fixture': INTERFACE, 'jobs': [], 'tools': [],
        }

    def save(self):
        with self.lock:
            pending = self.output.with_name('.' + self.output.name + '.pending')
            pending.write_text(json.dumps(self.value, indent=2, ensure_ascii=False) + '\n')
            pending.replace(self.output)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class ObservedClient:
    def __init__(self, client, evidence):
        self.client, self.evidence = client, evidence

    def __getattr__(self, name):
        return getattr(self.client, name)

    def submit(self, owner, task, **options):
        result = self.client.submit(owner, task, **options)
        with self.evidence.lock:
            self.evidence.value['jobs'].append({
                'id': result['id'], 'submitted_utc': utc(), 'host_conversation': task,
                'options': options, 'submission': result,
            })
            self.evidence.save()
        return result

    def read(self, owner, ident, **options):
        result = self.client.read(owner, ident, **options)
        with self.evidence.lock:
            entry = next(value for value in self.evidence.value['jobs'] if value['id'] == ident)
            entry['last_state'] = result['state']
            if result['state'] not in ('queued', 'running'):
                entry['result'] = result
                entry['finished_utc'] = utc()
                self.evidence.save()
        return result


class ReadOnlyAgent(LocalChatAgent):
    MAX_TOOL_CALLS = 12

    def __init__(self, *args, evidence, **kwargs):
        self.evidence = evidence
        super().__init__(*args, **kwargs)

    def execute(self, arguments):
        argv = arguments.get('argv') if isinstance(arguments, dict) else None
        allowed = isinstance(arguments, dict) and arguments.get('operation') == 'command' and readonly_argv(argv)
        entry = {'utc': utc(), 'arguments': arguments, 'allowed': allowed,
                 'receipt_id': 'tool-' + str(self.tool_calls),
                 'native_call': self._chat_native_call}
        with self.evidence.lock:
            self.evidence.value['tools'].append(entry)
            self.evidence.save()
        if not allowed:
            error = ('Isolated read-only investigation denied this action. Only phone hardware/status/help, '
                     'NPU/voice status, isolated saved results, and read-only system metadata commands are permitted.')
            entry['error'] = error
            self.evidence.save()
            raise AppError(error)
        started = time.monotonic()
        try:
            result = super().execute(arguments)
            entry['result'] = result
            return result
        except Exception as error:
            entry['error'] = str(error)
            raise
        finally:
            entry['elapsed_seconds'] = round(time.monotonic() - started, 6)
            self.evidence.save()


def all_events(store, sid):
    events = []
    while True:
        page = store.events(sid, events[-1]['seq'] if events else 0)
        if not page:
            return events
        events.extend(page)


def evaluate(args):
    check_guard()
    if os.getuid() == 0:
        raise RuntimeError('Use native Termux Python with the ordinary Termux app identity')
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite existing evidence: ' + str(args.output))
    os.umask(0o077)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence = Evidence(args.output, args.engine, args.question)
    isolation = Path(tempfile.mkdtemp(prefix='device-investigation-', dir=args.output.parent))
    evidence.value.update(isolation_dir=str(isolation), uid=os.getuid(),
                          selinux_context=Path('/proc/self/attr/current').read_text().strip('\x00\n'),
                          source_sha256={name: hashlib.sha256((APP / name).read_bytes()).hexdigest()
                                         for name in ('agent.py', 'local_chat.py', 'npu_chat.py', 'local_runtime.py')})
    evidence.save()
    store = agent = None
    started = time.monotonic()
    try:
        selected = NPUConversationClient() if args.engine == 'npu' else LocalClient()
        client = ObservedClient(selected, evidence)
        before = selected.status()
        evidence.value['backend_before'] = before
        expected = 'litert-npu' if args.engine == 'npu' else 'llama-cpu'
        if (not before.get('ready') or before.get('active') or before.get('queued')
                or (before.get('model') or {}).get('backend') != expected):
            raise RuntimeError('Selected service must already be ready, idle, and have the expected backend')
        workspace = isolation / 'workspace'
        workspace.mkdir(mode=0o700)
        store = Store(isolation / 'state')
        store.setting('action_policy', 'auto')
        sid = store.create(workspace)
        agent = ReadOnlyAgent(store, workspace, client=client, evidence=evidence)
        evidence.value.update(session=sid, workspace=str(workspace), status='running')
        evidence.save()
        with patch.object(agent.interface, 'read_context', return_value=INTERFACE), \
                patch('agent.codex_command', side_effect=AssertionError('Cloud route blocked by integration harness')), \
                patch('gemini_client.GeminiClient.submit', side_effect=AssertionError('Cloud route blocked by integration harness')):
            evidence.value['submission'] = agent.submit(sid, args.question, 'device-probe-' + uuid.uuid4().hex)
            next_progress = 0
            while agent.start_thread.is_alive():
                elapsed = time.monotonic() - started
                if elapsed >= args.deadline:
                    agent.cancel(sid)
                    agent.start_thread.join(5)
                    raise TimeoutError('Isolated investigation exceeded its deadline')
                if elapsed >= next_progress:
                    progress = {'elapsed_seconds': round(elapsed, 2), 'jobs': len(evidence.value['jobs']),
                                'tools': len(evidence.value['tools'])}
                    print(json.dumps(progress), flush=True)
                    next_progress = elapsed + 10
                agent.start_thread.join(.2)
        events = all_events(store, sid)
        after = selected.status()
        answers = [event['data']['text'] for event in events if event['kind'] == 'assistant']
        finishes = [event['data'] for event in events if event['kind'] == 'finished']
        tool_results = [event['data'] for event in events if event['kind'] == 'tool_result']
        completed = bool(answers) and bool(finishes) and finishes[-1].get('status') == 'idle'
        evidence.value.update(
            events=events, backend_after=after, final_answer='\n'.join(answers),
            finished=finishes, status='completed' if completed else 'failed',
            autonomous_tool_use=bool(evidence.value['tools']),
            successful_tool_receipts=sum(item.get('success') is True for item in tool_results),
            diagnostic_tool_use_verified=completed and any(
                item.get('allowed') and 'result' in item and item.get('arguments', {}).get('argv') not in
                (['@voice', 'status'], ['@phone', 'help'], ['@phone', 'status'])
                for item in evidence.value['tools']),
            engine_pid_unchanged=before.get('engine_pid') == after.get('engine_pid'),
            semantic_verdict='Manual review required against actual retained tool results; no synthesized answer or automatic factual pass.',
        )
    except Exception as error:
        evidence.value.update(status='failed', error=type(error).__name__ + ': ' + str(error))
    finally:
        if agent is not None:
            agent.close()
        if store is not None:
            store.db.close()
        evidence.value.update(finished_utc=utc(), elapsed_seconds=round(time.monotonic() - started, 6))
        evidence.save()
    print(json.dumps({'output': str(args.output), 'status': evidence.value['status'],
                      'tools': len(evidence.value['tools']), 'final_answer': evidence.value.get('final_answer'),
                      'error': evidence.value.get('error')}, ensure_ascii=False), flush=True)
    return 0 if evidence.value['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', choices=('cpu', 'npu'), required=True)
    parser.add_argument('--question', default=QUESTION)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--deadline', type=float, default=240)
    args = parser.parse_args()
    if not 1 <= args.deadline <= 600:
        parser.error('deadline must be between 1 and 600 seconds')
    args.output = args.output.resolve()
    return evaluate(args)


if __name__ == '__main__':
    import sys
    if '--live' not in sys.argv:
        raise SystemExit('Live device/model integration is opt-in. Read this script, then pass --live explicitly.')
    sys.argv.remove('--live')
    raise SystemExit(main())
