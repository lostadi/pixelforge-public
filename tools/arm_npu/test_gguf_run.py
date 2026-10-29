"""Fake-engine deadline/cancellation and broker ownership fixtures; no NPU calls."""
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

from create_tiny_gguf import create
from gguf_broker import publish_owner


KIT = Path(__file__).resolve().parent


def process_identity(pid):
    """Return state and start ticks, or None after a fixture process disappears."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return None


class GGUFRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / 'fixture.gguf'
        create(self.model)

    def tearDown(self):
        self.temporary.cleanup()

    def fake_engine(self, ready, cleanup_marker=None):
        engine = self.root / 'fake-engine'
        # Both jobs inherit the engine session's process group. The engine
        # reaps its child after group termination but never signals it itself.
        # Therefore stopping only the engine leaves the sleeper alive and the
        # test below fails its child-cleanup assertion.
        source = f'''#!{sys.executable}
import json, os, pathlib, signal, subprocess, sys, time
child = subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(120)'])
def info(pid):
    fields = pathlib.Path('/proc/%d/stat' % pid).read_text().split(') ', 1)[1].split()
    return {{'pid': pid, 'start_ticks': fields[19]}}
def stopped(signum, frame):
    if {str(cleanup_marker) if cleanup_marker else None!r} is not None:
        pathlib.Path({str(cleanup_marker)!r}).write_text('cleanup-started')
        time.sleep(1)
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        sys.exit(81)
    sys.exit(0)
signal.signal(signal.SIGTERM, stopped)
signal.signal(signal.SIGHUP, stopped)
pathlib.Path({str(ready)!r}).write_text(json.dumps({{'engine': info(os.getpid()), 'child': info(child.pid)}}))
while True:
    time.sleep(60)
'''
        engine.write_text(source)
        engine.chmod(0o700)
        return engine

    def run_fake(self, *, cancel_signal=None, repeat_sigint=False, cleanup_wait_error=False):
        ready = self.root / 'ready.json'
        output = self.root / 'result'
        cleanup_marker = self.root / 'cleanup-started' if repeat_sigint else None
        engine = self.fake_engine(ready, cleanup_marker)
        # Supply a fake benchmark module before loading the real wrapper;
        # hardware_snapshot therefore cannot read phone hardware in this test.
        loader = (
            'import runpy,sys,types;sys.path.insert(0,' + repr(str(KIT)) + ');'
            'fixture=types.ModuleType("benchmark");fixture.hardware_snapshot=lambda:{};'
            'sys.modules["benchmark"]=fixture;')
        if cleanup_wait_error:
            # Fail the cleanup wait only after the real process exits. This
            # exercises final reporting without deliberately orphaning a job.
            injection = '''class FixturePopen(subprocess.Popen):
    def wait(self, *args, **kwargs):
        result = super().wait(*args, **kwargs)
        if kwargs.get('timeout') == 5:
            raise OSError('fixture cleanup wait failure')
        return result
subprocess.Popen = FixturePopen
'''
            loader += 'import subprocess;exec(' + repr(injection) + ');'
        loader += 'runpy.run_path(' + repr(str(KIT / 'run_gguf.py')) + ',run_name="__main__")'
        command = [sys.executable, '-I', '-c', loader, '--model', str(self.model),
                   '--engine', str(engine), '--cpu', '--tokens', '1', '--prompt', 'fixture',
                   '--timeout', '1' if cancel_signal is None else '30', '--output', str(output)]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, start_new_session=True)
        identities = {}
        try:
            deadline = time.monotonic() + 5
            while not ready.exists():
                if process.poll() is not None:
                    stdout, stderr = process.communicate()
                    self.fail(f'Wrapper exited before fake engine started: {stdout}\n{stderr}')
                if time.monotonic() >= deadline:
                    self.fail('Fake engine did not publish its readiness record')
                time.sleep(.01)
            # The tiny readiness file may exist just before write_text finishes.
            while True:
                try:
                    identities = json.loads(ready.read_text())
                    break
                except json.JSONDecodeError:
                    if time.monotonic() >= deadline:
                        self.fail('Fake engine readiness record stayed incomplete')
                    time.sleep(.01)
            if cancel_signal is not None:
                process.send_signal(cancel_signal)
            if repeat_sigint:
                deadline = time.monotonic() + 3
                while not cleanup_marker.exists():
                    if process.poll() is not None or time.monotonic() >= deadline:
                        self.fail('Fake engine cleanup did not begin before repeated SIGINT')
                    time.sleep(.01)
                # The marker follows the engine's SIGTERM receipt, proving
                # the wrapper has entered its finally block. SIGINT must not
                # abort its delayed wait or skip RESULT publication.
                for _ in range(4):
                    process.send_signal(signal.SIGINT)
                    time.sleep(.05)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 1, f'{stdout}\n{stderr}')
            report = json.loads((output / 'RESULT.json').read_text())
            self.assertFalse(report['success'])
            self.assertEqual(report['backend'], 'cpu-reference')
            self.assertEqual(report['hardware_before'], {})
            self.assertEqual(report['hardware_after'], {})
            if cancel_signal is None:
                self.assertIn('timed out', report['error'])
                self.assertGreaterEqual(report['seconds'], 1)
            elif cancel_signal == signal.SIGINT:
                self.assertIn('error', report)  # KeyboardInterrupt has no message.
            else:
                self.assertIn(f'interrupted by signal {cancel_signal}', report['error'])
            self.assertEqual(report['engine_cleanup_ok'], not cleanup_wait_error)
            self.assertTrue(report['broker_cleanup_ok'])
            if cleanup_wait_error:
                self.assertIn('fixture cleanup wait failure', '\n'.join(report['cleanup_errors']))
            self.assertEqual(stat.S_IMODE((output / 'RESULT.json').stat().st_mode), 0o600)
            for name, owned in identities.items():
                observed = process_identity(owned['pid'])
                self.assertTrue(observed is None or observed[1] != owned['start_ticks']
                                or observed[0] == 'Z',
                                f'Owned fake {name} remained alive after wrapper cleanup')
        finally:
            # Failure cleanup can signal only our recorded fixture PIDs after
            # checking start ticks through a pidfd; never target a reused PID.
            for owned in identities.values():
                descriptor = None
                try:
                    descriptor = os.pidfd_open(owned['pid'])
                    observed = process_identity(owned['pid'])
                    if observed and observed[1] == owned['start_ticks'] and observed[0] != 'Z':
                        signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                finally:
                    if descriptor is not None:
                        os.close(descriptor)
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
            process.communicate(timeout=5)

    def test_whole_engine_deadline_reports_failure_and_stops_its_group(self):
        self.run_fake()

    def test_sigterm_cancellation_reports_failure_and_stops_its_group(self):
        self.run_fake(cancel_signal=signal.SIGTERM)

    def test_sighup_cancellation_reports_failure_and_stops_its_group(self):
        self.run_fake(cancel_signal=signal.SIGHUP)

    def test_repeated_sigint_during_cleanup_does_not_skip_result_or_owned_exit(self):
        self.run_fake(cancel_signal=signal.SIGINT, repeat_sigint=True)

    def test_cleanup_wait_error_still_publishes_failed_result(self):
        self.run_fake(cancel_signal=signal.SIGTERM, cleanup_wait_error=True)

    def test_publish_owner_is_private_complete_and_does_not_replace_collision(self):
        path = self.root / 'BROKER_OWNER.json'
        owner = {'pid': 1234, 'start_ticks': '5678'}
        publish_owner(path, owner)
        self.assertEqual(json.loads(path.read_text()), owner)
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        original = path.read_bytes()
        inode = path.stat().st_ino
        with self.assertRaises(FileExistsError):
            publish_owner(path, {'pid': 9999, 'start_ticks': '0'})
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(path.stat().st_ino, inode)
        self.assertEqual(list(self.root.glob('.broker-owner-*')), [])


if __name__ == '__main__':
    unittest.main()
