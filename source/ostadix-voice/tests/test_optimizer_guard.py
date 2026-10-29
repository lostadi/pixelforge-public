"""Guardian policy/ownership tests; no installed guardian or real service runs."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import optimizer_guard as guard


class OneTick:
    def __init__(self):
        self.stopped = False

    def is_set(self):
        return self.stopped

    def wait(self, _seconds):
        self.stopped = True


class GuardianTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.state = self.root / 'guardian'
        self.state.mkdir(mode=0o700)
        self.optimizer = self.root / 'optimizer.json'
        self.service = self.root / 'service.json'
        self.npu_service = self.root / 'npu-service.json'
        patches = {
            'ROOT_STATE': self.state,
            'OPTIMIZER_CONFIG': self.optimizer,
            'SERVICE_CONFIG': self.service,
            'NPU_SERVICE_CONFIG': self.npu_service,
            'ROOT_UID': os.getuid(),
            'TERMUX_UID': os.getuid(),
        }
        self.patcher = patch.multiple(guard, **patches)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def enable_configs(self):
        for path in (self.optimizer, self.service):
            path.write_text('{"enabled":true}')

    def test_enabled_requires_exact_boolean(self):
        for value, expected in [(True, True), (False, False), (1, False), ('true', False), (None, False)]:
            self.optimizer.write_text(json.dumps({'enabled': value}))
            self.assertEqual(guard.read_enabled(self.optimizer), expected)

    def test_missing_malformed_oversized_or_symlinked_config_is_disabled(self):
        self.assertFalse(guard.read_enabled(self.optimizer))
        for text in ('{', '[]', '{"enabled":true,"pad":"' + 'x' * 17000 + '"}'):
            self.optimizer.write_text(text)
            self.assertFalse(guard.read_enabled(self.optimizer))
        self.optimizer.unlink()
        self.service.write_text('{"enabled":true}')
        self.optimizer.symlink_to(self.service)
        self.assertFalse(guard.read_enabled(self.optimizer))

    def test_wrong_owner_config_is_disabled(self):
        self.enable_configs()
        with patch.object(guard, 'TERMUX_UID', os.getuid() + 1):
            self.assertFalse(guard.read_enabled(self.optimizer))

    def test_root_directory_rejects_shared_write_or_symlink(self):
        guard.prepare_root_state()
        self.state.chmod(0o777)
        with self.assertRaises(RuntimeError):
            guard.prepare_root_state()
        self.state.chmod(0o700)
        alias = self.root / 'alias'
        alias.symlink_to(self.state, target_is_directory=True)
        with patch.object(guard, 'ROOT_STATE', alias), self.assertRaises(RuntimeError):
            guard.prepare_root_state()

    def test_lock_prevents_duplicate_watchers(self):
        with guard.guardian_lock() as first:
            self.assertTrue(first)
            with guard.guardian_lock() as second:
                self.assertFalse(second)
        with guard.guardian_lock() as third:
            self.assertTrue(third)

    def test_log_symlink_is_rejected_without_touching_target(self):
        target = self.root / 'target'
        target.write_bytes(b'preserve')
        (self.state / 'stdout.log').symlink_to(target)
        with self.assertRaises(OSError):
            guard.append_log('stdout.log', b'replace')
        self.assertEqual(target.read_bytes(), b'preserve')

    def test_logs_rotate_and_remain_bounded(self):
        with patch.object(guard, 'LOG_LIMIT', 16):
            guard.append_log('stdout.log', b'1234567890')
            guard.append_log('stdout.log', b'abcdefghij')
            self.assertEqual((self.state / 'stdout.log.1').read_bytes(), b'1234567890')
            self.assertEqual((self.state / 'stdout.log').read_bytes(), b'abcdefghij')
            guard.append_log('stdout.log', b'x' * 100)
            self.assertLessEqual((self.state / 'stdout.log').stat().st_size, 16)
            self.assertLessEqual((self.state / 'stdout.log.1').stat().st_size, 16)

    def test_package_force_stop_or_uncertain_status_is_not_bypassed(self):
        for data, expected in [
            (b'    User 0: stopped=true enabled=0\n', True),
            (b'    User 0: stopped=false enabled=0\n', False),
            (b'    User 10: stopped=false\n', None),
            (b'    User 0: stopped=false\n    User 0: stopped=true\n', None),
        ]:
            with patch.object(guard.subprocess, 'run', return_value=Mock(returncode=0, stdout=data)):
                self.assertIs(guard.package_stopped(), expected)
        with patch.object(guard.subprocess, 'run', side_effect=subprocess.TimeoutExpired('probe', 3)):
            self.assertIsNone(guard.package_stopped())

    def test_disabled_optimizer_ensure_never_spawns(self):
        with patch.object(guard.subprocess, 'Popen') as spawn:
            self.assertEqual(guard.ensure()['reason'], 'optimizer_disabled')
            spawn.assert_not_called()

    def test_already_running_ensure_never_spawns(self):
        self.enable_configs()
        with guard.guardian_lock(), patch.object(guard.subprocess, 'Popen') as spawn:
            self.assertEqual(guard.ensure()['reason'], 'already_running')
            spawn.assert_not_called()

    def test_ensure_launches_only_fixed_installed_watcher(self):
        self.enable_configs()
        states = iter([True, False])
        @contextlib.contextmanager
        def lock():
            yield next(states)
        with patch.object(guard, 'guardian_lock', lock), \
                patch.object(guard.subprocess, 'Popen', return_value=Mock(pid=123)) as spawn:
            self.assertTrue(guard.ensure()['running'])
            args, kwargs = spawn.call_args
            self.assertEqual(args[0], [str(guard.PYTHON), '-I', str(guard.INSTALLED), '--watch'])
            self.assertTrue(kwargs['start_new_session'])
            self.assertEqual(kwargs['env'], guard.child_environment())

    def test_disabled_optimizer_watcher_exits_without_resume(self):
        with patch.object(guard, 'run_resume') as resume, patch.object(guard, 'package_stopped') as package:
            self.assertEqual(guard.watch(OneTick()), 0)
            resume.assert_not_called()
            package.assert_not_called()
        self.assertEqual(json.loads((self.state / 'guardian.json').read_text())['phase'], 'disabled')

    def test_disabled_local_service_pauses_without_enabling(self):
        self.optimizer.write_text('{"enabled":true}')
        with patch.object(guard, 'run_resume') as resume, patch.object(guard, 'package_stopped') as package:
            guard.watch(OneTick())
            resume.assert_not_called()
            package.assert_not_called()
        self.assertFalse(self.service.exists())

    def test_force_stop_and_unknown_package_state_pause_resume(self):
        self.enable_configs()
        for stopped in (True, None):
            with patch.object(guard, 'package_stopped', return_value=stopped), \
                    patch.object(guard, 'run_resume') as resume:
                guard.watch(OneTick())
                resume.assert_not_called()

    def test_enabled_service_resumes_once_per_tick(self):
        self.enable_configs()
        with patch.object(guard, 'package_stopped', return_value=False), \
                patch.object(guard, 'run_resume', return_value={'exit_code': 0}) as resume:
            guard.watch(OneTick())
            resume.assert_called_once()
        metadata = json.loads((self.state / 'guardian.json').read_text())
        self.assertEqual(metadata['uid'], os.getuid())
        self.assertEqual(metadata['last_resume'], {'exit_code': 0})

    def test_enabled_npu_recovers_independently_of_default_service_and_optimizer(self):
        self.npu_service.write_text('{"enabled":true}')
        self.assertTrue(guard.recovery_enabled())
        self.assertTrue(guard.resume_enabled())
        with patch.object(guard, 'package_stopped', return_value=False), \
                patch.object(guard, 'run_resume', return_value={'exit_code': 0}) as resume:
            guard.watch(OneTick())
        resume.assert_called_once()
        self.assertFalse(self.optimizer.exists())
        self.assertFalse(self.service.exists())

    def test_npu_disabled_or_force_stopped_never_recovers(self):
        self.npu_service.write_text('{"enabled":false}')
        with patch.object(guard, 'run_resume') as resume:
            guard.watch(OneTick())
        resume.assert_not_called()
        self.npu_service.write_text('{"enabled":true}')
        for stopped in (True, None):
            with self.subTest(stopped=stopped), patch.object(guard, 'package_stopped', return_value=stopped), \
                    patch.object(guard, 'run_resume') as resume:
                guard.watch(OneTick())
            resume.assert_not_called()

    def test_recovery_command_is_fixed_and_drops_identity_before_resume_all(self):
        with patch.object(guard, 'TERMUX_UID', 10402), \
                patch.object(guard, 'APP_CONTEXT', 'u:r:untrusted_app_27:s0:c1,c2'):
            command = guard.resume_command()
        self.assertEqual(command[-1], str(guard.APP_ROOT / 'local_service.py'))
        self.assertIn('os.setuid(10402)', command[-2])
        self.assertIn('"--resume-all"', command[-2])
        self.assertNotIn('--enable', ' '.join(command))

    def test_recovery_releases_inherited_root_oom_protection_before_dropping_uid(self):
        events = []
        library = Mock()
        library.setcon.return_value = 0
        with patch.object(guard, 'TERMUX_UID', 10402), \
                patch.object(guard, 'APP_CONTEXT', 'u:r:untrusted_app_27:s0:c1,c2'), \
                patch('ctypes.CDLL', return_value=library), \
                patch('os.open', side_effect=lambda path, flags: events.append(('open', path)) or 55), \
                patch('os.write', side_effect=lambda descriptor, data: events.append(('write', descriptor, data)) or len(data)), \
                patch('os.close', side_effect=lambda descriptor: events.append(('close', descriptor))), \
                patch('os.setgroups', side_effect=lambda groups: events.append(('groups', groups))), \
                patch('os.setgid', side_effect=lambda gid: events.append(('gid', gid))), \
                patch('os.setuid', side_effect=lambda uid: events.append(('uid', uid))), \
                patch('os.execv', side_effect=lambda *args: events.append(('exec', args))), \
                patch.object(sys, 'argv', ['guardian', str(guard.APP_ROOT / 'local_service.py')]):
            exec(guard.resume_command()[-2], {})
        self.assertEqual(events[:3], [('open', '/proc/self/oom_score_adj'), ('write', 55, b'0'), ('close', 55)])
        self.assertEqual([event[0] for event in events[3:]], ['groups', 'gid', 'uid', 'exec'])
        self.assertEqual(events[-1][1][1][-1], '--resume-all')

    def test_failed_oom_reset_never_drops_uid_or_resumes_protected_workers(self):
        for failure in (PermissionError('fixture failure'), 0):
            with self.subTest(failure=failure), patch.object(guard, 'TERMUX_UID', 10402), \
                    patch.object(guard, 'APP_CONTEXT', 'u:r:untrusted_app_27:s0:c1,c2'), patch('ctypes.CDLL'), \
                    patch('os.open', return_value=55), \
                    patch('os.write', side_effect=failure if isinstance(failure, Exception) else None,
                          return_value=failure if isinstance(failure, int) else None), \
                    patch('os.close'), patch('os.setuid') as set_uid, patch('os.execv') as execute:
                with self.assertRaises((PermissionError, AssertionError)):
                    exec(guard.resume_command()[-2], {})
                set_uid.assert_not_called()
                execute.assert_not_called()

    def test_root_guardian_requires_local_identity_configuration(self):
        with patch.object(guard, 'TERMUX_UID', None), self.assertRaises(RuntimeError):
            guard.resume_command()
        with patch.object(guard, 'TERMUX_UID', 10402), patch.object(guard, 'APP_CONTEXT', ''), self.assertRaises(RuntimeError):
            guard.resume_command()

    def test_bounded_child_output_and_timeout_only_signal_owned_child(self):
        self.enable_configs()
        harmless = [sys.executable, '-I', '-c', 'import time; print("guard fixture",flush=True); time.sleep(30)']
        with patch.object(guard, 'resume_command', return_value=harmless), \
                patch.object(guard, 'APP_ROOT', self.root), \
                patch.object(guard, 'CHILD_TIMEOUT', .15):
            result = guard.run_resume(threading.Event())
        self.assertEqual(result['reason'], 'resume_timeout')
        self.assertNotEqual(result['exit_code'], 0)
        self.assertLess(result['elapsed_seconds'], 3)
        self.assertIn(b'guard fixture', (self.state / 'stdout.log').read_bytes())

    def test_child_output_flood_is_bounded(self):
        self.enable_configs()
        harmless = [sys.executable, '-I', '-c', 'import os; os.write(1,b"x"*65536)']
        with patch.object(guard, 'resume_command', return_value=harmless), \
                patch.object(guard, 'APP_ROOT', self.root), patch.object(guard, 'LOG_LIMIT', 4096):
            result = guard.run_resume(threading.Event())
        self.assertEqual(result['reason'], 'resume_output_limit')
        for path in self.state.glob('*.log*'):
            self.assertLessEqual(path.stat().st_size, 4096)

    def test_boot_template_has_bounded_wait_and_no_enable(self):
        self.assertIn('"$waited" -lt 300', guard.BOOT_SCRIPT)
        self.assertIn('waited=$((waited + 5))', guard.BOOT_SCRIPT)
        self.assertIn('--ensure', guard.BOOT_SCRIPT)
        self.assertIn('ostadix-voice-npu/service.json', guard.BOOT_SCRIPT)
        self.assertNotIn('--enable', guard.BOOT_SCRIPT)
        self.assertNotIn('force-stop', guard.BOOT_SCRIPT)


if __name__ == '__main__':
    unittest.main()
