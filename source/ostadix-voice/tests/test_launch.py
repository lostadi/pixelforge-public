"""Session stop and launch regressions; no models or audio."""
import contextlib
import ctypes
import errno
import fcntl
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import launch
from server import AppError


class MainThreadSignalTests(unittest.TestCase):
    def test_modern_pidfd_signal_targets_thread_without_numeric_pid_fallback(self):
        with patch.object(signal, 'pidfd_send_signal', create=True) as send, \
                patch('ctypes.CDLL') as library, patch.object(os, 'kill') as kill:
            self.assertTrue(launch._signal_ui_main_thread(12345, signal.SIGTERM, 44, lambda: True))
        send.assert_called_once_with(44, signal.SIGTERM, None, 1)
        library.assert_not_called()
        kill.assert_not_called()

    def test_old_kernel_uses_exact_main_tid_after_rechecking_identity(self):
        current = Mock(return_value=True)
        library = Mock()
        library.tgkill.return_value = 0
        with patch.object(signal, 'pidfd_send_signal', create=True,
                          side_effect=OSError(errno.EINVAL, 'thread flag unsupported')), \
                patch('ctypes.CDLL', return_value=library), patch.object(os, 'kill') as kill:
            self.assertTrue(launch._signal_ui_main_thread(12345, signal.SIGTERM, 44, current))
        library.tgkill.assert_called_once_with(12345, 12345, signal.SIGTERM)
        self.assertGreaterEqual(current.call_count, 2)
        kill.assert_not_called()

    def test_changed_owner_cannot_receive_numeric_tid_signal(self):
        library = Mock()
        current = Mock(side_effect=[True, False])
        with patch.object(signal, 'pidfd_send_signal', create=True,
                          side_effect=OSError(errno.EINVAL, 'thread flag unsupported')), \
                patch('ctypes.CDLL', return_value=library), patch.object(os, 'kill') as kill:
            self.assertFalse(launch._signal_ui_main_thread(12345, signal.SIGTERM, 44, current))
        library.tgkill.assert_not_called()
        kill.assert_not_called()

    def test_numeric_signal_permission_failure_is_not_retried_as_another_target(self):
        library = Mock()
        library.tgkill.return_value = -1
        with patch('ctypes.CDLL', return_value=library), \
                patch('ctypes.get_errno', return_value=errno.EPERM), \
                patch.object(os, 'kill') as kill, self.assertRaises(PermissionError):
            launch._signal_ui_main_thread(12345, signal.SIGTERM, is_current=lambda: True)
        kill.assert_not_called()

    def test_missing_thread_api_keeps_portable_process_signal_fallback(self):
        with patch('ctypes.CDLL', side_effect=OSError('no native thread API')), \
                patch.object(os, 'kill') as kill:
            self.assertTrue(launch._signal_ui_main_thread(12345, signal.SIGTERM, is_current=lambda: True))
        kill.assert_called_once_with(12345, signal.SIGTERM)


SESSION_FIXTURE = '''\
import fcntl
from pathlib import Path
import signal
import sys
import time

state, mode, ready, stopped = map(Path, sys.argv[1:])
state.mkdir(parents=True, exist_ok=True)
lock = (state / 'server.lock').open('a')

def shutdown(signum, frame):
    stopped.write_text('SIGTERM received')
    raise SystemExit(0)

signal.signal(signal.SIGTERM, signal.SIG_IGN if str(mode) == 'ignore' else shutdown)
if str(mode) != 'open-only':
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
ready.write_text('ready')
try:
    while True:
        time.sleep(.02)
finally:
    lock.close()
'''


@unittest.skipUnless(Path('/proc/self/fdinfo').is_dir(), 'Requires Linux process lock information')
class StopSessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.app = self.root / 'app'
        self.app.mkdir()
        self.state = self.root / 'state'
        self.processes = []
        self.app_patch = patch.object(launch, 'APP', self.app)
        self.app_patch.start()
        self.root_patch = patch.object(launch, 'root_session_owner', return_value=None)
        self.root_patch.start()
        for name in ('terminal.py', 'server.py', 'unknown_holder.py'):
            (self.app / name).write_text(SESSION_FIXTURE)

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            process.communicate(timeout=5)
        self.app_patch.stop()
        self.root_patch.stop()
        self.temp.cleanup()

    def start_session(self, script='terminal.py', state=None, mode='graceful'):
        number = len(self.processes)
        ready = self.root / ('ready-' + str(number))
        stopped = self.root / ('stopped-' + str(number))
        process = subprocess.Popen(
            [sys.executable, str(self.app / script), str(state or self.state),
             mode, str(ready), str(stopped)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        self.processes.append(process)
        deadline = time.monotonic() + 5
        while not ready.exists():
            if process.poll() is not None:
                output, error = process.communicate()
                self.fail('Session fixture failed: ' + output + error)
            if time.monotonic() >= deadline:
                self.fail('Session fixture did not acquire its lock')
            time.sleep(.01)
        return process, stopped

    def test_no_session_does_not_create_state(self):
        self.assertFalse(launch.stop(self.state, timeout=.2))
        self.assertFalse(self.state.exists())

    def test_stale_lock_is_preserved_and_does_not_signal_any_process(self):
        self.state.mkdir()
        lock = self.state / 'server.lock'
        lock.write_text('old metadata is not process ownership')
        inode = lock.stat().st_ino
        with patch.object(launch.os, 'kill') as kill:
            self.assertFalse(launch.stop(self.state, timeout=.2))
        kill.assert_not_called()
        self.assertEqual(lock.stat().st_ino, inode)
        self.assertEqual(lock.read_text(), 'old metadata is not process ownership')

    def test_terminal_without_metadata_shuts_down_gracefully_and_keeps_lock_inode(self):
        process, stopped = self.start_session()
        lock = self.state / 'server.lock'
        inode = lock.stat().st_ino
        saved = self.state / 'saved-conversation'
        saved.write_text('keep this session')
        self.assertFalse((self.state / 'browser.json').exists())

        self.assertTrue(launch.stop(self.state, timeout=3))

        self.assertEqual(process.wait(timeout=3), 0)
        self.assertEqual(stopped.read_text(), 'SIGTERM received')
        self.assertFalse(launch.running(self.state))
        self.assertEqual(lock.stat().st_ino, inode)
        self.assertEqual(saved.read_text(), 'keep this session')
        self.assertFalse(launch.stop(self.state, timeout=.2))

    def test_browser_server_without_metadata_can_be_stopped(self):
        process, stopped = self.start_session('server.py')

        self.assertTrue(launch.stop(self.state, timeout=3))

        self.assertEqual(process.wait(timeout=3), 0)
        self.assertTrue(stopped.exists())

    def test_suspended_session_shuts_down_gracefully_and_keeps_lock_inode(self):
        process, stopped = self.start_session()
        lock = self.state / 'server.lock'
        inode = lock.stat().st_ino
        process.send_signal(signal.SIGSTOP)
        status = Path('/proc') / str(process.pid) / 'status'
        deadline = time.monotonic() + 3
        while True:
            state_line = next(line for line in status.read_text().splitlines()
                              if line.startswith('State:'))
            if state_line.split()[1] == 'T':
                break
            if time.monotonic() >= deadline:
                self.fail('Session fixture did not suspend')
            time.sleep(.01)
        self.assertTrue(launch.running(self.state))

        self.assertTrue(launch.stop(self.state, timeout=3))

        self.assertEqual(process.wait(timeout=3), 0)
        self.assertEqual(stopped.read_text(), 'SIGTERM received')
        self.assertFalse(launch.running(self.state))
        self.assertEqual(lock.stat().st_ino, inode)

    def test_owner_exiting_before_starttime_read_needs_no_further_signal(self):
        process, stopped = self.start_session()
        target = Path('/proc') / str(process.pid) / 'stat'
        original = Path.read_text
        def read(path, *args, **kwargs):
            if path == target:
                process.terminate()
                process.wait(timeout=3)
                raise FileNotFoundError(str(target))
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', read), \
                patch.object(launch, '_signal_ui_main_thread') as signal_main:
            self.assertTrue(launch.stop(self.state, timeout=3))
        signal_main.assert_not_called()
        self.assertTrue(stopped.exists())

    def test_only_the_requested_state_is_stopped(self):
        other_state = self.root / 'other-state'
        other, other_stopped = self.start_session(state=other_state)
        target, target_stopped = self.start_session()

        self.assertTrue(launch.stop(self.state, timeout=3))

        self.assertEqual(target.wait(timeout=3), 0)
        self.assertTrue(target_stopped.exists())
        self.assertIsNone(other.poll())
        self.assertFalse(other_stopped.exists())
        self.assertTrue(launch.running(other_state))

    def test_open_file_descriptor_without_flock_is_not_a_session_owner(self):
        bystander, bystander_stopped = self.start_session(mode='open-only')
        target, target_stopped = self.start_session('server.py')

        self.assertTrue(launch.stop(self.state, timeout=3))

        self.assertEqual(target.wait(timeout=3), 0)
        self.assertTrue(target_stopped.exists())
        self.assertIsNone(bystander.poll())
        self.assertFalse(bystander_stopped.exists())

    def test_unrecognized_lock_holder_is_not_signalled(self):
        process, stopped = self.start_session('unknown_holder.py')
        lock = self.state / 'server.lock'
        inode = lock.stat().st_ino

        with self.assertRaises(AppError):
            launch.stop(self.state, timeout=.2)

        self.assertIsNone(process.poll())
        self.assertFalse(stopped.exists())
        self.assertTrue(launch.running(self.state))
        self.assertEqual(lock.stat().st_ino, inode)

    def test_timeout_keeps_held_lock_and_does_not_force_kill(self):
        process, stopped = self.start_session(mode='ignore')
        lock = self.state / 'server.lock'
        inode = lock.stat().st_ino

        with self.assertRaises(AppError):
            launch.stop(self.state, timeout=.15)

        self.assertIsNone(process.poll())
        self.assertFalse(stopped.exists())
        self.assertTrue(launch.running(self.state))
        self.assertEqual(lock.stat().st_ino, inode)

    def test_hidden_owner_fallback_is_given_the_exact_held_inode(self):
        process, stopped = self.start_session()
        info = (self.state / 'server.lock').stat()
        def root_signal(state, identity):
            self.assertEqual(state, self.state)
            self.assertEqual(identity, (info.st_dev, info.st_ino))
            process.terminate()
            return {'status': 'signalled'}
        with patch.object(launch, '_session_processes', return_value=iter(())), \
                patch.object(launch, 'root_session_owner', side_effect=root_signal) as fallback:
            self.assertTrue(launch.stop(self.state, timeout=3))
        fallback.assert_called_once()
        self.assertEqual(process.wait(timeout=3), 0)
        self.assertTrue(stopped.exists())

    def test_rejected_hidden_owner_is_not_signalled_by_normal_user(self):
        process, stopped = self.start_session()
        with patch.object(launch, '_session_processes', return_value=iter(())), \
                patch.object(launch, 'root_session_owner', return_value={'status': 'changed'}), \
                patch.object(launch.os, 'kill') as kill, \
                self.assertRaisesRegex(AppError, 'could not be identified'):
            launch.stop(self.state, timeout=.1)
        kill.assert_not_called()
        self.assertIsNone(process.poll())
        self.assertFalse(stopped.exists())


@unittest.skipUnless(Path('/proc/self/fdinfo').is_dir(), 'Requires Linux process lock information')
class RootSessionHelperTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.root.chmod(0o755)
        self.app = self.root / 'app'
        self.app.mkdir()
        self.state = self.root / 'state'
        self.state.mkdir()
        self.uid = 65534 if os.geteuid() == 0 else os.geteuid()
        if os.geteuid() == 0:
            os.chown(self.app, self.uid, self.uid)
            os.chown(self.state, self.uid, self.uid)
        script = self.app / 'terminal.py'
        script.write_text(SESSION_FIXTURE)
        self.ready = self.state / 'ready'
        self.stopped = self.state / 'stopped'
        def identity():
            os.setgid(self.uid)
            os.setuid(self.uid)
        self.process = subprocess.Popen(
            [sys.executable, str(script), str(self.state), 'graceful', str(self.ready), str(self.stopped)],
            preexec_fn=identity if os.geteuid() == 0 else None,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(self.close)
        deadline = time.monotonic() + 5
        while not self.ready.exists():
            if self.process.poll() is not None or time.monotonic() >= deadline:
                self.fail('Privileged fallback fixture did not start')
            time.sleep(.01)

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.communicate(timeout=5)

    def helper(self, **overrides):
        info = (self.state / 'server.lock').stat()
        request = {'uid': self.uid, 'app': str(self.app), 'state': str(self.state),
                   'python': str(Path(sys.executable).resolve()), 'dev': info.st_dev,
                   'ino': info.st_ino, 'action': 'probe'}
        request.update(overrides)
        # No su is used by unit tests. A normal-UID test can inspect its own
        # fixture; simulate only the entry check, without obtaining privileges.
        code = 'import os; os.geteuid = lambda: 0\n' + launch.ROOT_SESSION_HELPER
        result = subprocess.run([sys.executable, '-I', '-S', '-c', code], input=json.dumps(request),
                                capture_output=True, text=True, timeout=8)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_probe_proves_exact_owner_without_signalling(self):
        result = self.helper()
        self.assertEqual(result['status'], 'owner')
        self.assertEqual(result['pid'], self.process.pid)
        self.assertEqual(result['uid'], self.uid)
        self.assertIsNone(self.process.poll())
        self.assertFalse(self.stopped.exists())

    def test_wrong_owner_and_changed_inode_cannot_signal(self):
        self.assertEqual(self.helper(uid=self.uid + 1, action='stop')['status'], 'rejected')
        self.assertEqual(self.helper(ino=0, action='stop')['status'], 'changed')
        self.assertEqual(self.helper(uid=0, action='stop')['status'], 'rejected')
        self.assertIsNone(self.process.poll())
        self.assertFalse(self.stopped.exists())

    def test_stop_gracefully_signals_only_the_verified_fixture(self):
        self.assertEqual(self.helper(action='stop')['status'], 'signalled')
        self.assertEqual(self.process.wait(timeout=5), 0)
        self.assertTrue(self.stopped.exists())
        self.assertTrue((self.state / 'server.lock').exists())


class StopCommandTests(unittest.TestCase):
    def test_stop_exits_before_workspace_updates_or_local_model_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / 'separate-state'
            local_client = Mock(side_effect=AssertionError('Must not manage the local model'))
            local_service = Mock(side_effect=AssertionError('Must not start the local model'))
            modules = {
                'local_runtime': types.SimpleNamespace(LocalClient=local_client),
                'local_service': types.SimpleNamespace(LocalService=local_service),
            }
            for result, message in (
                (True, 'Stopped Ostadix Voice. Run c to start it again.'),
                (False, 'No Ostadix Voice session is running.'),
            ):
                with self.subTest(stopped=result), contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(launch.os, 'geteuid', return_value=1000))
                    stop = stack.enter_context(patch.object(launch, 'stop', return_value=result))
                    prepare = stack.enter_context(patch.object(launch, 'prepare_ostadix'))
                    workspace = stack.enter_context(patch.object(launch, 'saved_workspace'))
                    execute = stack.enter_context(patch.object(launch.os, 'execv'))
                    stack.enter_context(patch.dict(sys.modules, modules))
                    output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

                    launch.main(['--stop', '--state-dir', str(state)])

                    stop.assert_called_once_with(state.resolve())
                    self.assertEqual(output.getvalue().strip(), message)
                    prepare.assert_not_called()
                    workspace.assert_not_called()
                    execute.assert_not_called()
            local_client.assert_not_called()
            local_service.assert_not_called()
            self.assertFalse(state.exists())

    def test_stop_rejects_other_commands_and_launch_arguments_before_signalling(self):
        for argument in ('--local-status', '--start-local', '--stop-local',
                         '--optimizer-status', '--start-optimizer', '--stop-optimizer', '--optimizer-history',
                         '--companion-status', '--start-companion', '--stop-companion', '--companion-read',
                         '--update-ostadix', '--browser', '/nonexistent/workspace'):
            with self.subTest(argument=argument), \
                    patch.object(launch.os, 'geteuid', return_value=1000), \
                    patch.object(launch, 'stop') as stop, \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    launch.main(['--stop', argument])
                self.assertEqual(error.exception.code, 2)
                stop.assert_not_called()


class CompanionCommandTests(unittest.TestCase):
    def test_controls_exit_before_tty_updates_takeover_or_local_startup(self):
        cases = [('--companion-status', 'status'), ('--start-companion', 'enable'),
                 ('--stop-companion', 'disable'), ('--companion-read', 'read')]
        for argument, operation in cases:
            with self.subTest(argument=argument), contextlib.ExitStack() as stack:
                controller = Mock()
                getattr(controller, operation).return_value = {'operation': operation}
                factory = Mock(return_value=controller)
                local_client = Mock(side_effect=AssertionError('Control must not start a local model'))
                stack.enter_context(patch.dict(sys.modules, {
                    'companion': types.SimpleNamespace(CompanionControl=factory),
                    'local_runtime': types.SimpleNamespace(LocalClient=local_client),
                }))
                stack.enter_context(patch.object(launch.os, 'geteuid', return_value=10402))
                stack.enter_context(patch.object(launch.sys.stdin, 'isatty', return_value=False))
                prepare = stack.enter_context(patch.object(launch, 'prepare_ostadix'))
                workspace = stack.enter_context(patch.object(launch, 'saved_workspace'))
                stop = stack.enter_context(patch.object(launch, 'stop'))
                execute = stack.enter_context(patch.object(launch.os, 'execv'))
                output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                launch.main([argument])
                factory.assert_called_once_with()
                getattr(controller, operation).assert_called_once_with()
                self.assertEqual(len(controller.method_calls), 1)
                self.assertEqual(json.loads(output.getvalue())['operation'], operation)
                prepare.assert_not_called(); workspace.assert_not_called(); stop.assert_not_called()
                execute.assert_not_called(); local_client.assert_not_called()

    def test_control_errors_are_reported_without_launching(self):
        controller = Mock()
        controller.read.side_effect = RuntimeError('Message is unavailable')
        with patch.dict(sys.modules, {'companion': types.SimpleNamespace(CompanionControl=lambda: controller)}), \
                patch.object(launch.os, 'geteuid', return_value=10402), \
                patch.object(launch.os, 'execv') as execute:
            with self.assertRaisesRegex(AppError, 'Companion: Message is unavailable'):
                launch.main(['--companion-read'])
            execute.assert_not_called()


class OptimizerCommandTests(unittest.TestCase):
    def test_controls_exit_before_session_updates_or_local_startup(self):
        cases = [(['--optimizer-status'], 'status', {}),
                 (['--optimizer-history'], 'history', {'limit': 10}),
                 (['--stop-optimizer'], 'disable', {}),
                 (['--start-optimizer'], 'enable', {'objective': 'balanced'}),
                 (['--start-optimizer', '--optimizer-objective', 'performance'], 'enable', {'objective': 'performance'}),
                 (['--start-optimizer', '--optimizer-objective', 'battery'], 'enable', {'objective': 'battery'})]
        for argv, operation, expected in cases:
            with self.subTest(argv=argv), contextlib.ExitStack() as stack:
                controller = Mock()
                getattr(controller, operation).return_value = {'operation': operation, 'managed_effects': []}
                factory = Mock(return_value=controller)
                local_client = Mock(side_effect=AssertionError('CLI must not start the local model'))
                stack.enter_context(patch.dict(sys.modules, {
                    'optimizer': types.SimpleNamespace(OptimizerControl=factory),
                    'local_runtime': types.SimpleNamespace(LocalClient=local_client),
                }))
                stack.enter_context(patch.object(launch.os, 'geteuid', return_value=10402))
                prepare = stack.enter_context(patch.object(launch, 'prepare_ostadix'))
                workspace = stack.enter_context(patch.object(launch, 'saved_workspace'))
                stop = stack.enter_context(patch.object(launch, 'stop'))
                execute = stack.enter_context(patch.object(launch.os, 'execv'))
                output = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                launch.main(argv)
                factory.assert_called_once_with()
                getattr(controller, operation).assert_called_once_with(**expected)
                self.assertEqual(len(controller.method_calls), 1)
                self.assertEqual(json.loads(output.getvalue())['operation'], operation)
                prepare.assert_not_called(); workspace.assert_not_called()
                stop.assert_not_called(); execute.assert_not_called(); local_client.assert_not_called()

    def test_objective_requires_explicit_start_and_valid_choice(self):
        for argv in (['--optimizer-objective', 'battery'],
                     ['--optimizer-status', '--optimizer-objective', 'performance'],
                     ['--start-optimizer', '--optimizer-objective', 'root']):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    launch.main(argv)
                self.assertEqual(error.exception.code, 2)

    def test_controller_error_is_reported_without_launching(self):
        controller = Mock()
        controller.disable.side_effect = RuntimeError('Restoration is still pending')
        with patch.dict(sys.modules, {'optimizer': types.SimpleNamespace(OptimizerControl=lambda: controller)}), \
                patch.object(launch.os, 'geteuid', return_value=10402), \
                patch.object(launch.os, 'execv') as execute:
            with self.assertRaisesRegex(AppError, 'Optimizer: Restoration is still pending'):
                launch.main(['--stop-optimizer'])
            execute.assert_not_called()


class AutoRestartCommandTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / 'state'
        self.workspace = self.root / 'saved-project'
        self.workspace.mkdir()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(launch, 'DEFAULT_STATE', self.state))
        self.stack.enter_context(patch.object(launch.os, 'geteuid', return_value=1000))
        self.stdin = self.stack.enter_context(patch.object(launch.sys, 'stdin'))
        self.stdin.isatty.return_value = True
        self.output = self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.running = self.stack.enter_context(patch.object(launch, 'running', return_value=True))
        self.saved = self.stack.enter_context(
            patch.object(launch, 'saved_workspace', return_value=str(self.workspace)))
        self.stop = self.stack.enter_context(patch.object(launch, 'stop', return_value=True))
        self.prepare = self.stack.enter_context(patch.object(launch, 'prepare_ostadix'))
        self.execute = self.stack.enter_context(patch.object(launch.os, 'execv'))
        self.service = Mock(state=self.root / 'local-service')
        self.service.enable.return_value = {}
        self.service.resume.return_value = {}
        self.local_service = Mock(return_value=self.service)
        self.stack.enter_context(patch.dict(sys.modules, {
            'local_service': types.SimpleNamespace(LocalService=self.local_service),
        }))
        self.calls = Mock()
        for name, mock in (('stop', self.stop), ('prepare', self.prepare),
                           ('local_service', self.local_service), ('execute', self.execute)):
            self.calls.attach_mock(mock, name)

    def assert_not_launched(self):
        self.prepare.assert_not_called()
        self.local_service.assert_not_called()
        self.execute.assert_not_called()

    def test_plain_c_stops_existing_session_before_starting_saved_workspace(self):
        launch.main([])

        self.stop.assert_called_once_with(self.state.resolve())
        self.saved.assert_called_once_with(self.state.resolve())
        self.assertEqual([call[0] for call in self.calls.mock_calls],
                         ['prepare', 'stop', 'local_service', 'execute'])
        self.service.enable.assert_called_once_with()
        self.execute.assert_called_once_with(sys.executable, [
            sys.executable, str(launch.APP / 'terminal.py'),
            '--workspace', str(self.workspace), '--state-dir', str(self.state),
            '--voice',
        ])

    def test_fresh_launch_does_not_stop_any_session(self):
        self.running.return_value = False

        launch.main([])

        self.stop.assert_not_called()
        self.prepare.assert_called_once()
        self.execute.assert_called_once()

    def test_voice_launch_replaces_session_and_forwards_explicit_activation(self):
        launch.main(['--voice'])

        self.stop.assert_called_once_with(self.state.resolve())
        self.assertEqual(self.execute.call_args.args[1][-1], '--voice')

    def test_incompatible_voice_options_do_not_stop_existing_session(self):
        for option in ('--browser', '--no-audio', '--stop', '--text'):
            with self.subTest(option=option), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    launch.main(['--voice', option])
                self.assertEqual(error.exception.code, 2)
        self.stop.assert_not_called()
        self.assert_not_launched()

    def test_browser_switch_stops_old_session_and_opens_requested_workspace(self):
        requested = self.root / 'requested-project'
        requested.mkdir()
        self.stdin.isatty.return_value = False

        launch.main(['--browser', str(requested)])

        self.stop.assert_called_once_with(self.state.resolve())
        self.execute.assert_called_once_with(sys.executable, [
            sys.executable, str(launch.APP / 'server.py'),
            '--workspace', str(requested), '--state-dir', str(self.state), '--open',
        ])

    def test_invalid_workspace_leaves_existing_session_running(self):
        file_path = self.root / 'ordinary-file'
        file_path.write_text('not a project directory')
        for arguments in ([str(self.root / 'missing-project')], [str(file_path)], []):
            with self.subTest(arguments=arguments):
                self.saved.return_value = str(self.root / 'missing-saved-project')

                with self.assertRaises((AppError, OSError)):
                    launch.main(arguments)

                self.stop.assert_not_called()
                self.assert_not_launched()

    def test_noninteractive_terminal_leaves_existing_session_running(self):
        self.stdin.isatty.return_value = False

        with self.assertRaises(AppError):
            launch.main([])

        self.stop.assert_not_called()
        self.assert_not_launched()
        self.assertFalse(self.state.exists())

    def test_failed_stop_prevents_starting_replacement(self):
        self.stop.side_effect = AppError('The session is still shutting down.')

        with self.assertRaisesRegex(AppError, 'still shutting down'):
            launch.main([])

        self.stop.assert_called_once_with(self.state.resolve())
        self.prepare.assert_called_once_with(offline=False)
        self.local_service.assert_not_called()
        self.execute.assert_not_called()

    def test_failed_verification_leaves_existing_session_running(self):
        self.prepare.side_effect = AppError('Run c --update-ostadix to repair it.')

        with self.assertRaisesRegex(AppError, 'c --update-ostadix'):
            launch.main([])

        self.stop.assert_not_called()
        self.local_service.assert_not_called()
        self.execute.assert_not_called()

    def test_explicit_update_exits_without_replacing_session(self):
        self.stdin.isatty.return_value = False

        launch.main(['--update-ostadix'])

        self.prepare.assert_called_once_with(required_update=True)
        self.saved.assert_not_called()
        self.stop.assert_not_called()
        self.local_service.assert_not_called()
        self.execute.assert_not_called()
        self.assertFalse(self.state.exists())


class AutomaticUpdateTests(unittest.TestCase):
    def test_launch_reuses_fully_verified_pair_without_network_or_build(self):
        with patch('ostadix_pair.update_pair') as update, \
                patch('ostadix_pair.load_pair', return_value={'revision': 'a' * 40}) as load, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(launch.prepare_ostadix()['revision'], 'a' * 40)
            load.assert_called_once_with()
            update.assert_not_called()

    def test_missing_or_invalid_pair_bootstraps_before_launch(self):
        from ostadix_pair import PairError
        with patch('ostadix_pair.update_pair', return_value={'revision': 'b' * 40}) as update, \
                patch('ostadix_pair.load_pair', side_effect=PairError('invalid cached build')) as load, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(launch.prepare_ostadix()['revision'], 'b' * 40)
            load.assert_called_once()
            update.assert_called_once()
            self.assertIn('installing OSTADIX-BM', output.getvalue())

    def test_offline_skips_update(self):
        with patch('ostadix_pair.update_pair') as update, \
                patch('ostadix_pair.load_pair', return_value={'revision': 'a' * 40}), \
                contextlib.redirect_stdout(io.StringIO()):
            launch.prepare_ostadix(offline=True)
            update.assert_not_called()

    def test_explicit_update_failure_is_reported(self):
        from ostadix_pair import PairError
        with patch('ostadix_pair.update_pair', side_effect=PairError('build failed')), \
                patch('ostadix_pair.load_pair') as load, \
                self.assertRaisesRegex(AppError, 'build failed'):
            launch.prepare_ostadix(required_update=True)
        load.assert_not_called()

    def test_explicit_update_fetches_even_with_an_installed_pair(self):
        with patch('ostadix_pair.update_pair', return_value={'revision': 'b' * 40}) as update, \
                patch('ostadix_pair.load_pair') as load:
            self.assertEqual(launch.prepare_ostadix(required_update=True)['revision'], 'b' * 40)
        update.assert_called_once()
        load.assert_not_called()

    def test_offline_invalid_pair_never_bootstraps(self):
        from ostadix_pair import PairError
        with patch('ostadix_pair.update_pair') as update, \
                patch('ostadix_pair.load_pair', side_effect=PairError('invalid cached build')), \
                self.assertRaisesRegex(AppError, 'invalid cached build'):
            launch.prepare_ostadix(offline=True)
        update.assert_not_called()

    def test_offline_explicit_update_is_rejected_without_touching_pair(self):
        with patch('ostadix_pair.update_pair') as update, \
                patch('ostadix_pair.load_pair') as load, \
                self.assertRaisesRegex(AppError, 'cannot be combined'):
            launch.prepare_ostadix(offline=True, required_update=True)
        update.assert_not_called()
        load.assert_not_called()

    def test_failed_bootstrap_reports_failure_and_repair_instruction(self):
        from ostadix_pair import PairError
        with patch('ostadix_pair.update_pair', side_effect=PairError('network unavailable')), \
                patch('ostadix_pair.load_pair', side_effect=PairError('invalid cached build')), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(AppError, 'network unavailable') as error:
            launch.prepare_ostadix()
        self.assertIn('c --update-ostadix', str(error.exception))


@unittest.skipUnless(Path('/proc/self/fdinfo').is_dir(), 'Requires Linux process lock information')
class LaunchGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name) / 'state'

    def test_wait_is_bounded_and_lock_inode_survives_restart(self):
        with launch.launch_guard(self.state) as first:
            inode = os.fstat(first.fileno()).st_ino
            with self.assertRaisesRegex(AppError, 'still starting'):
                with launch.launch_guard(self.state, timeout=.05):
                    self.fail('Another launch entered the guarded startup')
        with launch.launch_guard(self.state) as second:
            self.assertEqual(os.fstat(second.fileno()).st_ino, inode)

    def test_exception_releases_guard_for_the_next_invocation(self):
        with self.assertRaisesRegex(RuntimeError, 'startup failed'):
            with launch.launch_guard(self.state):
                raise RuntimeError('startup failed')
        with launch.launch_guard(self.state, timeout=.1):
            pass

    def test_handoff_releases_only_the_matching_owned_flock(self):
        self.state.mkdir()
        descriptor = os.open(self.state / 'launch.lock', os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.dict(os.environ, {launch.LAUNCH_LOCK_FD: str(descriptor)}):
                self.assertTrue(launch.release_launch_lock(self.state))
                self.assertNotIn(launch.LAUNCH_LOCK_FD, os.environ)
            with self.assertRaises(OSError):
                os.fstat(descriptor)
            descriptor = None
            with launch.launch_guard(self.state, timeout=.1):
                pass
        finally:
            if descriptor is not None:
                os.close(descriptor)

    def test_untrusted_environment_cannot_close_unrelated_or_unlocked_descriptors(self):
        self.state.mkdir()
        for name in ('unrelated', 'launch.lock'):
            with self.subTest(name=name), (self.state / name).open('a') as stream:
                with patch.dict(os.environ, {launch.LAUNCH_LOCK_FD: str(stream.fileno())}):
                    self.assertFalse(launch.release_launch_lock(self.state))
                self.assertEqual(os.fstat(stream.fileno()).st_ino, (self.state / name).stat().st_ino)
        for value in ('0', '1', '2', '-1', 'not-a-descriptor', '999999999999999999999999999'):
            with self.subTest(value=value), patch.dict(os.environ, {launch.LAUNCH_LOCK_FD: value}):
                self.assertFalse(launch.release_launch_lock(self.state))


CONCURRENT_LAUNCH_FIXTURE = '''\
import os
from pathlib import Path
import sys
import types
sys.path.insert(0, sys.argv.pop(1))
import launch
launch.APP = Path(__file__).resolve().parent
real_geteuid = os.geteuid
checks = 0
def launch_identity():
    global checks
    checks += 1
    return 1000 if checks == 1 else real_geteuid()
launch.os.geteuid = launch_identity
launch.sys.stdin = types.SimpleNamespace(isatty=lambda: True)
launch.prepare_ostadix = lambda **kwargs: None
launch.main(sys.argv[1:])
'''


CONCURRENT_UI_FIXTURE = '''\
import argparse
import fcntl
import os
from pathlib import Path
import signal
import sys
import time
sys.path.insert(0, os.environ['TEST_REAL_APP'])
from launch import release_launch_lock
parser = argparse.ArgumentParser()
parser.add_argument('--workspace')
parser.add_argument('--state-dir', type=Path)
parser.add_argument('--no-audio', action='store_true')
args = parser.parse_args()
state = args.state_dir
marker = state / ('ui-' + str(os.getpid()))
marker.with_suffix('.starting').touch()
time.sleep(.3)  # Make the exec-to-session-lock race repeatable.
lock = (state / 'server.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
def stop(*_):
    marker.with_suffix('.stopped').touch()
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
assert release_launch_lock(state), 'The launcher did not hand off its startup guard'
marker.with_suffix('.ready').touch()
try:
    while True:
        time.sleep(.01)
finally:
    lock.close()
'''


@unittest.skipUnless(Path('/proc/self/fdinfo').is_dir(), 'Requires Linux process lock information')
class ConcurrentLaunchTests(unittest.TestCase):
    def test_concurrent_calls_serialize_through_exec_and_replace_the_first_session(self):
        real_app = Path(launch.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = root / 'app'
            app.mkdir()
            state = root / 'state'
            workspace = root / 'workspace'
            workspace.mkdir()
            (app / 'invoke.py').write_text(CONCURRENT_LAUNCH_FIXTURE)
            (app / 'terminal.py').write_text(CONCURRENT_UI_FIXTURE)
            command = [sys.executable, str(app / 'invoke.py'), str(real_app),
                       '--no-audio', '--no-local', '--state-dir', str(state), str(workspace)]
            children = []

            def wait_marker(child, suffix):
                marker = state / ('ui-' + str(child.pid) + suffix)
                deadline = time.monotonic() + 8
                while not marker.exists():
                    if child.poll() is not None:
                        output, errors = child.communicate()
                        self.fail('Concurrent launcher exited: ' + output + errors)
                    if time.monotonic() >= deadline:
                        self.fail('Concurrent launcher never created ' + marker.name)
                    time.sleep(.01)

            try:
                environment = dict(os.environ, TEST_REAL_APP=str(real_app))
                first = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, text=True)
                children.append(first)
                wait_marker(first, '.starting')
                second = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True)
                children.append(second)
                wait_marker(second, '.ready')
                self.assertEqual(first.wait(timeout=5), 0)
                self.assertTrue((state / ('ui-' + str(first.pid) + '.ready')).exists())
                self.assertTrue((state / ('ui-' + str(first.pid) + '.stopped')).exists())
                self.assertIsNone(second.poll())
                inode = (state / 'launch.lock').stat().st_ino
                # After handoff the launch gate is free even though the UI runs.
                with launch.launch_guard(state, timeout=.1):
                    self.assertEqual((state / 'launch.lock').stat().st_ino, inode)
            finally:
                for child in reversed(children):
                    if child.poll() is None:
                        child.terminate()
                    child.communicate(timeout=5)


if __name__ == '__main__':
    unittest.main()
