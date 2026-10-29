import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('assistant_service',
    Path(__file__).resolve().parents[1] / 'scripts/ostadix_assistant_service.py')
service = importlib.util.module_from_spec(spec)
spec.loader.exec_module(service)


class AssistantServiceTests(unittest.TestCase):
    def test_repeated_start_migrates_old_mode_without_restarting_running_host(self):
        for previous_mode in ('local-all-v1', 'local-o-v1', 'null'):
            with self.subTest(previous_mode=previous_mode):
                before = dict(supervisor_running=True, listening=True, route=previous_mode)
                after = dict(before, route='local-o-v1')
                with patch.object(service, 'status', side_effect=[before, after]), \
                        patch.object(service, 'system') as settings, \
                        patch.object(service.subprocess, 'run') as launch:
                    self.assertEqual(service.start(), after)
                launch.assert_not_called()
                settings.assert_called_once_with('/system/bin/settings', 'put', 'global',
                                                 service.SETTING, 'local-o-v1')

    def test_cold_start_waits_for_host_before_enabling_explicit_selection(self):
        stopped = dict(supervisor_running=False, listening=False, route='null')
        listening = dict(supervisor_running=True, listening=True, route='null')
        enabled = dict(listening, route='local-o-v1')
        with patch.object(service, 'status', side_effect=[stopped, stopped, listening, enabled]), \
                patch.object(service, 'system') as settings, \
                patch.object(service.subprocess, 'run') as launch, \
                patch.object(service.time, 'sleep') as sleep:
            self.assertEqual(service.start(), enabled)
        launch.assert_called_once()
        self.assertEqual(launch.call_args.args[0], ['/system/bin/sh', str(service.BOOT)])
        sleep.assert_called_once_with(0.2)
        settings.assert_called_once_with('/system/bin/settings', 'put', 'global',
                                         service.SETTING, 'local-o-v1')

    def test_unmanaged_host_does_not_change_selection_or_launch_another_host(self):
        with patch.object(service, 'status', return_value=dict(supervisor_running=False, listening=True)), \
                patch.object(service, 'system') as settings, \
                patch.object(service.subprocess, 'run') as launch:
            with self.assertRaisesRegex(SystemExit, 'unmanaged host'):
                service.start()
        launch.assert_not_called()
        settings.assert_not_called()

    def test_failed_start_does_not_enable_selection(self):
        with patch.object(service, 'status', return_value=dict(supervisor_running=False, listening=False)), \
                patch.object(service, 'system') as settings, \
                patch.object(service.subprocess, 'run') as launch, \
                patch.object(service.time, 'sleep'):
            with self.assertRaisesRegex(SystemExit, 'Service did not start'):
                service.start()
        launch.assert_called_once()
        settings.assert_not_called()


if __name__ == '__main__':
    unittest.main()
