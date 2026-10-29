"""Real Linux process-lifetime fixtures for the private worker guard; no NPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
import unittest


WORKER = Path(__file__).with_name('worker.py').resolve()


def identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return {'pid': pid, 'start_ticks': fields[19], 'state': fields[0]}
    except FileNotFoundError:
        return None


@unittest.skipUnless(sys.platform == 'linux' and hasattr(os, 'pidfd_open'),
                     'Worker lifetime guard requires Linux pidfds')
class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.processes = []
        self.owned = []
        self.owner = self.spawn([sys.executable, '-I', '-c', 'import time;time.sleep(120)'])
        self.owner_identity = identity(self.owner.pid)

    def tearDown(self):
        # On an assertion failure, terminate only fixture jobs with matching
        # start ticks. A pidfd prevents signaling an unrelated reused PID.
        # A detached child may have been adopted elsewhere if the worker
        # crashed before publishing complete target readiness. Its own
        # fixture identity record still lets us clean it up safely.
        for name in ('grandchild.json', 'ready.json'):
            try:
                record = json.loads((self.root / name).read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                continue
            self.owned.extend([record] if name == 'grandchild.json' else record.values())
        # Discover descendants first in case readiness failed before their
        # fixture identity record was published.
        known = {(record['pid'], record['start_ticks']) for record in self.owned}
        queue = list(self.owned)
        while queue:
            parent = queue.pop(0)
            observed = identity(parent['pid'])
            if observed is None or observed['start_ticks'] != parent['start_ticks']:
                continue
            try:
                children = Path(f"/proc/{parent['pid']}/task/{parent['pid']}/children").read_text().split()
            except FileNotFoundError:
                continue
            for child_pid in children:
                child = identity(int(child_pid))
                if child and (child['pid'], child['start_ticks']) not in known:
                    known.add((child['pid'], child['start_ticks']))
                    self.owned.append(child)
                    queue.append(child)
        for record in reversed(self.owned):
            self.signal_owned(record, signal.SIGKILL)
        for process in reversed(self.processes):
            if process.poll() is None:
                process.wait(timeout=5)
            else:
                process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()
        self.temporary.cleanup()

    def spawn(self, command, **kwargs):
        process = subprocess.Popen(command, start_new_session=True, **kwargs)
        self.processes.append(process)
        record = identity(process.pid)
        if record:
            self.owned.append(record)
        return process

    def signal_owned(self, record, signum):
        descriptor = None
        try:
            descriptor = os.pidfd_open(record['pid'])
            observed = identity(record['pid'])
            if observed and observed['start_ticks'] == record['start_ticks']:
                signal.pidfd_send_signal(descriptor, signum)
        except ProcessLookupError:
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def command(self, target, *, status, grace=.4, python=True, ticks=None):
        result = [sys.executable, str(WORKER), '--owner-pid', str(self.owner.pid),
                  '--owner-start-ticks', ticks or self.owner_identity['start_ticks'],
                  '--grace-seconds', str(grace), '--status-file', str(status)]
        if python:
            result.append('--python')
        return [*result, '--', *target]

    def status(self, path):
        value = json.loads(path.read_text())
        self.assertIs(value.get('cleanup_ok'), True, value)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        return value

    def assert_gone(self, record):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            observed = identity(record['pid'])
            if observed is None or observed['start_ticks'] != record['start_ticks']:
                return
            time.sleep(.01)
        self.fail(f'Owned fixture was not reaped after cleanup: {record}')

    def waiting_target(self, *, ignore_term=False, delayed_finally=False):
        ready = self.root / 'ready.json'
        grandchild_ready = self.root / 'grandchild.json'
        final = self.root / 'finally-complete'
        final_started = self.root / 'finally-started'
        child_code = f'''import json,os,pathlib,signal,time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
fields = pathlib.Path('/proc/%d/stat' % os.getpid()).read_text().split(') ', 1)[1].split()
pathlib.Path({str(grandchild_ready)!r}).write_text(json.dumps({{'pid':os.getpid(),'start_ticks':fields[19]}}))
time.sleep(120)
'''
        script = self.root / 'target.py'
        script.write_text(f'''import json,os,pathlib,signal,subprocess,sys,time
if {ignore_term!r}:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable,'-I','-c',{child_code!r}], start_new_session=True)
deadline = time.monotonic() + 3
while not pathlib.Path({str(grandchild_ready)!r}).exists():
    if time.monotonic() > deadline: raise RuntimeError('grandchild not ready')
    time.sleep(.01)
while True:
    try:
        grandchild = json.loads(pathlib.Path({str(grandchild_ready)!r}).read_text())
        break
    except json.JSONDecodeError:
        time.sleep(.01)
fields = pathlib.Path('/proc/%d/stat' % os.getpid()).read_text().split(') ', 1)[1].split()
try:
    pathlib.Path({str(ready)!r}).write_text(json.dumps({{'target':{{'pid':os.getpid(),'start_ticks':fields[19]}},'grandchild':grandchild}}))
    time.sleep(120)
finally:
    pathlib.Path({str(final_started)!r}).write_text('started')
    if {delayed_finally!r}: time.sleep(.8)
    pathlib.Path({str(final)!r}).write_text('complete')
''')
        return script, ready, final, final_started

    def read_ready(self, path, worker):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if path.exists():
                try:
                    value = json.loads(path.read_text())
                    self.owned.extend(value.values())
                    return value
                except json.JSONDecodeError:
                    pass
            if worker.poll() is not None:
                try:
                    out, error = worker.communicate(timeout=.2)
                except subprocess.TimeoutExpired:
                    self.fail('Exited worker left fixture descendants holding its pipes')
                self.fail(f'Worker exited before target readiness: {out}\n{error}')
            time.sleep(.01)
        self.fail('Target did not publish readiness')

    def launch_waiting(self, *, ignore_term=False, delayed_finally=False, grace=.4):
        script, ready, final, final_started = self.waiting_target(
            ignore_term=ignore_term, delayed_finally=delayed_finally)
        status = self.root / 'status.json'
        worker = self.spawn(self.command([sys.executable, str(script)], status=status, grace=grace),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        records = self.read_ready(ready, worker)
        return worker, records, status, final, final_started

    def test_owner_sigkill_runs_python_finally_and_reaps_detached_grandchild(self):
        worker, records, status, final, _ = self.launch_waiting()
        self.signal_owned(self.owner_identity, signal.SIGKILL)
        self.owner.wait(timeout=3)
        out, error = worker.communicate(timeout=6)
        self.assertNotEqual(worker.returncode, 0, f'{out}\n{error}')
        self.assertEqual(final.read_text(), 'complete')
        receipt = self.status(status)
        self.assertTrue(receipt['owner_exited'])
        self.assertEqual(receipt['reason'], 'owner_exited')
        for record in records.values():
            self.assert_gone(record)

    def test_ignored_term_escalates_only_owned_descendants(self):
        sentinel = self.spawn([sys.executable, '-I', '-c', 'import time;time.sleep(120)'])
        worker, records, status, final, _ = self.launch_waiting(ignore_term=True, grace=.3)
        started = time.monotonic()
        self.signal_owned(self.owner_identity, signal.SIGKILL)
        self.owner.wait(timeout=3)
        worker.communicate(timeout=6)
        self.assertNotEqual(worker.returncode, 0)
        self.assertGreaterEqual(time.monotonic() - started, .25)
        self.assertFalse(final.exists())
        receipt = self.status(status)
        self.assertTrue(receipt['forced_child_cleanup'])
        self.assertTrue(receipt['forced_descendant_cleanup'])
        for record in records.values():
            self.assert_gone(record)
        self.assertIsNone(sentinel.poll(), 'An unrelated fixture process was terminated')

    def test_repeated_term_preserves_delayed_python_finally_and_cleanup(self):
        worker, records, status, final, started = self.launch_waiting(delayed_finally=True, grace=2)
        worker.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + 3
        while not started.exists():
            if worker.poll() is not None or time.monotonic() > deadline:
                self.fail('Target finally did not begin before repeated TERM')
            time.sleep(.01)
        for _ in range(4):
            worker.send_signal(signal.SIGTERM)
            time.sleep(.05)
        worker.communicate(timeout=6)
        self.assertNotEqual(worker.returncode, 0)
        self.assertEqual(final.read_text(), 'complete')
        receipt = self.status(status)
        self.assertEqual(receipt['reason'], 'cancelled')
        self.assertEqual(receipt['cancel_signal'], signal.SIGTERM)
        for record in records.values():
            self.assert_gone(record)
        self.assertIsNone(self.owner.poll())

    def test_python_script_module_and_command_preserve_argv_stdio_and_exit7(self):
        arguments = ['space words', "'single' \"double\"", '$HOME;$(touch NEVER)',
                     '`uname`', 'line\nbreak', '雪 café', '--tokens', '0']
        code = '''import json,sys
value={'argv0':sys.argv[0],'arguments':sys.argv[1:],'input':sys.stdin.read()}
print(json.dumps(value,ensure_ascii=False),flush=True)
print('fixture-stderr',file=sys.stderr,flush=True)
sys.exit(7)
'''
        script = self.root / 'fixture_module.py'
        script.write_text(code)
        input_text = 'stdin survives: 雪\nsecond line\n'
        for mode, target, argv0 in [('script', [sys.executable, str(script)], str(script)),
                                    ('module', [sys.executable, '-m', 'fixture_module'], str(script)),
                                    ('command', [sys.executable, '-c', code], '-c')]:
            with self.subTest(mode=mode):
                status = self.root / f'status-{mode}.json'
                worker = self.spawn(self.command([*target, *arguments], status=status),
                                    cwd=self.root, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                out, error = worker.communicate(input=input_text, timeout=5)
                self.assertEqual(worker.returncode, 7, error)
                self.assertEqual(json.loads(out), {'argv0': argv0, 'arguments': arguments,
                                                  'input': input_text})
                self.assertEqual(error, 'fixture-stderr\n')
                receipt = self.status(status)
                self.assertTrue(receipt['started'])
                self.assertEqual(receipt['child_returncode'], 7)
        self.assertFalse((self.root / 'NEVER').exists())

    def test_non_python_command_preserves_exit7(self):
        status = self.root / 'raw-status.json'
        worker = self.spawn(self.command([sys.executable, '-I', '-c',
                                          "print('raw fixture');raise SystemExit(7)"],
                                         status=status, python=False),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, error = worker.communicate(timeout=5)
        self.assertEqual(worker.returncode, 7, error)
        self.assertEqual(out, 'raw fixture\n')
        self.assertEqual(error, '')
        self.status(status)

    def test_doctor_python_options_preserve_isolation_orig_argv_and_literal_arguments(self):
        status = self.root / 'doctor-status.json'
        arguments = ['literal $HOME;$(touch NEVER)', 'space \"quote\"', '雪 café', 'line\nbreak']
        code = '''import json,os,sys
print(json.dumps({'arguments':sys.argv[1:],'orig_argv':sys.orig_argv,
                  'isolated':sys.flags.isolated,'safe_path':sys.flags.safe_path,
                  'dont_write_bytecode':sys.dont_write_bytecode,
                  'path':sys.path,'cwd':os.getcwd()},ensure_ascii=False))
'''
        target = [sys.executable, '-B', '-I', '-c', code, *arguments]
        worker = self.spawn(self.command(target, status=status), cwd=self.root,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, error = worker.communicate(timeout=5)
        self.assertEqual(worker.returncode, 0, error)
        self.assertEqual(error, '')
        value = json.loads(out)
        self.assertEqual(value['arguments'], arguments)
        self.assertEqual(value['orig_argv'], target)
        self.assertEqual(value['isolated'], 1)
        self.assertIs(value['safe_path'], True)
        self.assertIs(value['dont_write_bytecode'], True)
        self.assertNotIn('', value['path'])
        self.assertNotIn(value['cwd'], value['path'])
        self.status(status)
        self.assertFalse((self.root / 'NEVER').exists())

    def test_non_python_self_signals_preserve_negative_returncodes_and_receipt(self):
        for signum in (signal.SIGTERM, signal.SIGKILL):
            with self.subTest(signum=signum):
                status = self.root / f'signal-{signum}-status.json'
                code = f'import os;os.kill(os.getpid(), {int(signum)})'
                worker = self.spawn(self.command([sys.executable, '-I', '-c', code],
                                                 status=status, python=False),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                out, error = worker.communicate(timeout=5)
                self.assertEqual(worker.returncode, -int(signum), error)
                self.assertEqual(out, '')
                self.assertEqual(error, '')
                receipt = self.status(status)
                self.assertEqual(receipt['child_returncode'], -int(signum))
                self.assertEqual(receipt['exit_code'], -int(signum))
                self.assertEqual(receipt['adopted_children_seen'], 0)
        self.assertIsNone(self.owner.poll())

    def test_stale_owner_start_ticks_refuses_to_start_target(self):
        marker = self.root / 'must-not-start'
        status = self.root / 'stale-status.json'
        code = f'import pathlib;pathlib.Path({str(marker)!r}).write_text("started")'
        worker = self.spawn(self.command([sys.executable, '-c', code], status=status,
                                         ticks=str(int(self.owner_identity['start_ticks']) + 1)),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out, error = worker.communicate(timeout=5)
        self.assertNotEqual(worker.returncode, 0, f'{out}\n{error}')
        self.assertFalse(marker.exists())
        self.assertIs(self.status(status)['started'], False)
        self.assertIsNone(self.owner.poll())


if __name__ == '__main__':
    unittest.main()
