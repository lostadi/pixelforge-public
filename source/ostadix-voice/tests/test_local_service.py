"""Normal-UID supervision tests with a fake engine; no model or cloud calls."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_runtime import LocalClient, LocalError
from local_service import LocalService, Supervisor, service_status
import local_service
from optimizer_tuning import TuningError


def eventually(check, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = check()
        if result:
            return result
        time.sleep(.025)
    raise AssertionError('Timed out waiting for local supervisor fixture')


ENGINE = r'''
import sys, threading
from pathlib import Path
from local_runtime import serve
class Engine:
    def __init__(self, state):
        self.model = {'name': 'persistent-selected-fixture'}
    def start(self):
        return self.model
    def is_alive(self):
        return True
    def run(self, task, role, max_tokens, cancelled, register):
        if task == 'hold until interrupted':
            cancelled.wait(60)
        return task, {'completion_tokens': 1}, False
    def close(self):
        pass
serve(Path(sys.argv[1]), Engine)
'''

SUPERVISOR = r'''
import json, signal, subprocess, sys
from pathlib import Path
from local_service import Supervisor
state = Path(sys.argv[1])
engine_source = sys.argv[2]
def spawn(command, **kwargs):
    assert Path(command[1]).name == 'local_runtime.py' and command[2] == '--daemon'
    with (state / 'spawn-audit.jsonl').open('a') as log:
        log.write(json.dumps(command) + '\n')
    return subprocess.Popen([sys.executable, '-c', engine_source, str(state)], **kwargs)
supervisor = Supervisor(state, spawn=spawn, poll_interval=.05, backoff=(.1, .2, .4, .8))
signal.signal(signal.SIGTERM, lambda *_: supervisor.stop_event.set())
supervisor.run()
'''


class ServiceStateTests(unittest.TestCase):
    def test_isolated_guardian_entrypoint_status_stays_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory) / 'absent'
            result = subprocess.run(
                [sys.executable, '-I', str(Path(local_service.__file__).resolve()),
                 '--status', '--state-dir', str(state), '--manifest', str(Path(directory) / 'profile.json')],
                capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(json.loads(result.stdout)['service_enabled'])
            self.assertFalse(state.exists())

    def test_explicit_manifest_reaches_supervisor_without_changing_default_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            manifest = state / 'npu-profile.json'
            (state / 'service.json').write_text('{"version":1,"enabled":true}')
            for selected in (None, manifest):
                with self.subTest(manifest=selected), patch('local_service.LocalClient') as client_factory:
                    service = LocalService(state, manifest_path=selected)
                    service.status = Mock(return_value={'service_enabled': True, 'running': True,
                                                       'supervised': True, 'service_phase': 'loading'})
                    with patch('local_service.os.getuid', return_value=10402), \
                            patch('local_service._state', return_value=state), \
                            patch('local_service.service_status', return_value={'supervised': False}), \
                            patch('local_service.subprocess.Popen') as spawn:
                        service.resume()
                    command = spawn.call_args.args[0]
                    self.assertIn('--supervise', command)
                    if selected is None:
                        client_factory.assert_called_once_with(state)
                        self.assertNotIn('--manifest', command)
                    else:
                        client_factory.assert_called_once_with(state, manifest_path=manifest)
                        self.assertEqual(command[-2:], ['--manifest', str(manifest)])

    def test_explicit_manifest_reaches_each_daemon_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            manifest = state / 'npu-profile.json'
            (state / 'service.json').write_text('{"version":1,"enabled":true}')
            with patch('local_service.LocalClient') as client_factory:
                supervisor = Supervisor(state, manifest_path=manifest)
            client_factory.assert_called_once_with(state, manifest_path=manifest)
            def spawn(command, **kwargs):
                self.assertEqual(command[-2:], ['--manifest', str(manifest)])
                self.assertEqual(command[2], '--daemon')
                supervisor.stop_event.set()
                return Mock()
            supervisor.spawn = Mock(side_effect=spawn)
            supervisor._stop_child = Mock()
            with patch('local_service.os.getuid', return_value=10402), \
                    patch('local_service._raw_status', return_value={'running': False}):
                supervisor.run()
            supervisor.spawn.assert_called_once()

    def test_daemon_exit_keeps_the_specific_engine_failure_and_backoff(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            (state / 'service.json').write_text('{"version":1,"enabled":true}')
            supervisor = Supervisor(state, poll_interval=.001)
            supervisor.client = Mock()
            supervisor.child = Mock()
            supervisor.child.poll.side_effect = [None, 1]
            supervisor._stop_child = Mock()
            responses = iter([{'running': True, 'error': 'Native server hash does not match pinned deployment'},
                              {'running': False}])
            def status(_client):
                result = next(responses)
                if not result['running']:
                    supervisor.stop_event.set()
                return result
            with patch('local_service.os.getuid', return_value=10402), \
                    patch('local_service._raw_status', side_effect=status):
                supervisor.run()
            self.assertEqual(supervisor.failures, 1)
            metadata = json.loads((state / 'supervisor.json').read_text())
            self.assertEqual(metadata['error'], 'Native server hash does not match pinned deployment')

    def test_resume_all_uses_independent_fixed_services_and_never_enables(self):
        local, npu = Mock(), Mock()
        local.resume.side_effect = LocalError('damaged CPU state')
        npu.resume.return_value = {'service_enabled': False, 'running': False}
        with patch('local_service.LocalService', side_effect=[local, npu]) as factory:
            result = local_service.resume_all()
        self.assertEqual(factory.call_args_list[0].args, (local_service.DEFAULT_STATE,))
        self.assertEqual(factory.call_args_list[1].args, (local_service.NPU_STATE,))
        self.assertEqual(factory.call_args_list[1].kwargs, {'manifest_path': local_service.NPU_MANIFEST})
        local.resume.assert_called_once_with(); npu.resume.assert_called_once_with()
        local.enable.assert_not_called(); npu.enable.assert_not_called()
        self.assertIn('damaged CPU state', result['local']['error'])
        self.assertFalse(result['npu']['service_enabled'])

    def test_only_canonical_npu_enable_ensures_fixed_existing_guardian(self):
        with tempfile.TemporaryDirectory() as directory, patch('local_service.LocalClient'):
            cases = [(local_service.NPU_STATE, local_service.NPU_MANIFEST, True),
                     (Path(directory), local_service.NPU_MANIFEST, False),
                     (local_service.NPU_STATE, None, False)]
            for state, manifest, expected in cases:
                with self.subTest(state=state, manifest=manifest):
                    service = LocalService(state, manifest_path=manifest)
                    service._resume = Mock(return_value={'service_enabled': True})
                    with patch('optimizer.root_call', return_value={'running': True}) as root:
                        result = service.enable()
                    service._resume.assert_called_once_with(enable=True)
                    self.assertEqual(root.call_count, int(expected))
                    if expected:
                        self.assertEqual(root.call_args.args[0][-1], '--ensure')
                        self.assertTrue(result['guardian']['running'])

    def test_normal_launcher_resumes_saved_choice_and_only_enables_first_setup(self):
        import launch
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            workspace = base / 'workspace'; workspace.mkdir()
            service = Mock()
            service.state = base / 'local-state'; service.state.mkdir()
            service.enable.return_value = service.resume.return_value = {'error': None}
            for saved in (False, True):
                if saved:
                    (service.state / 'service.json').write_text('{"version":1,"enabled":false}')
                service.reset_mock()
                with patch('sys.argv', ['c', str(workspace), '--offline', '--state-dir', str(base / 'conversation')]), \
                     patch('launch.sys.stdin.isatty', return_value=True), \
                     patch('launch.os.geteuid', return_value=10402), patch('launch.running', return_value=False), \
                     patch('launch.saved_workspace', return_value=None), patch('launch.prepare_ostadix'), \
                     patch('local_service.LocalService', return_value=service), \
                     patch('launch.os.execv', side_effect=RuntimeError('fixture terminal launched')):
                    with self.assertRaisesRegex(RuntimeError, 'fixture terminal launched'):
                        launch.main()
                if saved:
                    service.resume.assert_called_once_with()
                    service.enable.assert_not_called()
                else:
                    service.enable.assert_called_once_with()
                    service.resume.assert_not_called()
    def test_status_is_read_only_and_disabled_resume_never_launches(self):
        with tempfile.TemporaryDirectory() as directory:
            absent = Path(directory) / 'absent'
            with patch('local_service.subprocess.Popen') as spawn:
                status = LocalService(absent).resume()
                self.assertFalse(status['service_enabled'])
                self.assertFalse(status['running'])
                self.assertFalse(status['supervised'])
                spawn.assert_not_called()
            self.assertFalse(absent.exists())
    def test_backoff_increases_but_caps_at_thirty_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            supervisor = Supervisor(directory)
            with patch('local_service.time.monotonic', return_value=10), patch('local_service.time.time', return_value=100):
                delays = []
                for _ in range(8):
                    supervisor.failed('fixture crash')
                    delays.append(supervisor.next_start - 10)
            self.assertEqual(delays, [1, 2, 4, 8, 16, 30, 30, 30])
    def test_invalid_enablement_never_enables_automatic_start(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'service.json').write_text('{"version":1,"enabled":"true"}')
            status = service_status(path)
            self.assertFalse(status['service_enabled'])
            self.assertIn('Invalid local service', status['service_error'])

    def test_disable_stops_model_and_reports_optimizer_cleanup_failure(self):
        for failure in (TuningError('Invalid priority journal'),
                        {'error': 'Priority restoration is incomplete'}):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                (state / 'optimizer').mkdir()
                (state / 'optimizer/config.json').write_text('{}')
                service = LocalService(state)
                service.client = Mock()
                service.client._stop_once.return_value = {'running': False, 'ready': False}
                options = {'side_effect': failure} if isinstance(failure, Exception) else {'return_value': failure}
                with patch('local_service.os.getuid', return_value=10402), \
                     patch('local_service._state', return_value=state), \
                     patch('optimizer.OptimizerControl.stop', **options):
                    result = service.disable()
                service.client._stop_once.assert_called_once_with()
                self.assertFalse(result['service_enabled'])
                self.assertFalse(json.loads((state / 'service.json').read_text())['enabled'])
                self.assertTrue(result['optimizer_error'])
                self.assertIn(result['optimizer_error'], result['error'])

    def test_supervisor_stops_child_and_records_optimizer_cleanup_failure(self):
        for failure in (TuningError('Invalid priority journal'),
                        {'error': 'Priority restoration is incomplete'}):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                state = Path(directory)
                (state / 'optimizer').mkdir()
                (state / 'optimizer/config.json').write_text('{}')
                (state / 'service.json').write_text('{"version":1,"enabled":false}')
                supervisor = Supervisor(state)
                supervisor._stop_child = Mock()
                options = {'side_effect': failure} if isinstance(failure, Exception) else {'return_value': failure}
                with patch('local_service.os.getuid', return_value=10402), \
                     patch('optimizer.OptimizerControl.stop', **options):
                    supervisor.run()
                supervisor._stop_child.assert_called_once_with()
                metadata = json.loads((state / 'supervisor.json').read_text())
                self.assertEqual(metadata['phase'], 'stopped')
                self.assertTrue(metadata['optimizer_error'])


@unittest.skipIf(os.getuid() == 0, 'Supervisor integration requires the normal application user')
class SupervisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='service-fixture-')
        self.path = Path(self.temp.name) / 'state'
        self.client = LocalClient(self.path)
        self.processes = []
        self.commands = []
        real_spawn = subprocess.Popen
        def fixture(command, **kwargs):
            self.commands.append(command)
            if Path(command[1]).name == 'local_service.py':
                self.assertIn('--supervise', command)
                argv = [sys.executable, '-c', SUPERVISOR, str(self.path), ENGINE]
            else:
                self.assertEqual(Path(command[1]).name, 'local_runtime.py')
                argv = [sys.executable, '-c', ENGINE, str(self.path)]
            process = real_spawn(argv, **kwargs)
            self.processes.append(process)
            return process
        self.patcher = patch('local_service.subprocess.Popen', side_effect=fixture)
        self.patcher.start()
    def tearDown(self):
        try:
            self.client.stop()
            for process in self.processes:
                process.wait(timeout=10)
        finally:
            self.patcher.stop()
            self.temp.cleanup()
    def ready(self):
        return eventually(lambda: self.client.status() if self.client.status()['ready'] else None)
    def test_real_daemon_kill_restarts_without_jobs_replay_or_codex(self):
        self.client.start()
        before = self.ready()
        job = self.client.submit('owner', 'hold until interrupted')
        eventually(lambda: self.client.status()['active'] == 1)
        os.kill(before['daemon_pid'], signal.SIGKILL)
        after = eventually(lambda: self.client.status() if self.client.status()['ready'] and self.client.status()['daemon_pid'] != before['daemon_pid'] else None)
        self.assertEqual(after['supervisor_pid'], before['supervisor_pid'])
        self.assertEqual(after['model']['name'], 'persistent-selected-fixture')
        self.assertGreaterEqual(after['restart_count'], 1)
        self.assertEqual(self.client.read('owner', job['id'])['state'], 'interrupted')
        self.assertEqual(after['active'], 0)
        self.assertEqual(after['queued'], 0)
        self.assertEqual(len(self.client.jobs('owner')), 1)
        fresh = self.client.submit('owner', 'new explicit task')
        self.assertEqual(self.client.read('owner', fresh['id'], wait=2)['result'], 'new explicit task')
        launches = [json.loads(line) for line in (self.path / 'spawn-audit.jsonl').read_text().splitlines()]
        self.assertEqual(len(launches), 2)
        self.assertTrue(all(Path(command[1]).name == 'local_runtime.py' for command in launches))
        self.assertNotIn('codex', json.dumps(self.commands + launches).lower())
    def test_intentional_stop_stays_disabled_and_boot_resume_does_not_reverse_it(self):
        self.client.start(); self.ready()
        self.client.stop()
        eventually(lambda: not self.client.status()['running'] and not self.client.status()['supervised'])
        count = len(self.commands)
        time.sleep(.3)
        self.assertFalse(LocalService(self.path).resume()['service_enabled'])
        self.assertFalse(self.client.status()['running'])
        self.assertEqual(len(self.commands), count)
        self.assertFalse(json.loads((self.path / 'service.json').read_text())['enabled'])
        self.client.start()
        self.assertTrue(self.ready()['service_enabled'])
    def test_supervisor_adopts_existing_daemon_without_second_engine(self):
        original = self.client._start_once()
        self.assertTrue(original['running'])
        self.client.start()
        adopted = self.ready()
        self.assertTrue(adopted['supervised'])
        self.assertEqual(adopted['daemon_pid'], original['daemon_pid'])
        self.assertFalse((self.path / 'spawn-audit.jsonl').exists())
        self.assertEqual(self.client.start()['supervisor_pid'], adopted['supervisor_pid'])
        self.assertEqual(len(self.commands), 2)
    def test_assistant_socket_routes_share_runtime_and_preserve_identity(self):
        self.client.start(); self.ready()
        assistant = self.client.assistant_create('owner', 'Reviewer', 'reviewer')
        self.assertTrue(assistant['waiting_for_assignment'])
        self.assertEqual(self.client.jobs('owner'), [])
        self.client.assistant_guide('owner', assistant['id'], 'Use only the supplied facts')
        self.client.assistant_assign('owner', assistant['id'], 'The bulb is blue. Restate its color.', max_steps=2, max_tokens=64)
        completed = eventually(lambda: self.client.assistant_read('owner', assistant['id'])
                               if self.client.assistant_read('owner', assistant['id'])['state'] == 'completed' else None)
        self.assertEqual(completed['assignment']['completed_steps'], 2)
        self.assertIn('blue', completed['assignment']['latest_result'])
        self.assertTrue(completed['memory'])
        self.assertEqual(len(self.client.jobs('owner')), 2)
        with self.assertRaises(LocalError):
            self.client.assistant_read('other-owner', assistant['id'])
        self.client.stop()
        eventually(lambda: not self.client.status()['supervised'])
        self.client.start(); self.ready()
        saved = self.client.assistant_read('owner', assistant['id'])
        self.assertEqual(saved['memory'], completed['memory'])
        self.assertEqual(saved['guidance'], 'Use only the supplied facts')
        self.assertEqual(saved['state'], 'completed')
        self.assertEqual(len(self.client.jobs('owner')), 2)
        self.assertNotIn('codex', json.dumps(self.commands).lower())
    def test_boot_script_only_resumes_enabled_local_service(self):
        self.client.stop()
        boot = Path(self.temp.name) / 'boot'
        result = LocalService(self.path).install_boot(boot)
        script = Path(result['boot_script'])
        text = script.read_text()
        self.assertIn('--resume-all', text)
        self.assertNotIn('--enable', text)
        self.assertNotIn('codex', text.lower())
        self.assertNotIn('wake-lock', text)
        self.assertEqual(script.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.path / 'service.json').stat().st_mode & 0o777, 0o600)


if __name__ == '__main__':
    unittest.main()
