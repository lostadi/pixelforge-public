import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import AppError
from phone_tools import PhoneTools, _run
from hardware_info import CPU_ROOT, CPUINFO, format_cpu_info


FOCUS = 'Window{a1 u0 com.example.notes/.MainActivity}'
TERMUX_FOCUS = 'Window{b2 u0 com.termux/com.termux.app.TermuxActivity}'
XML = '''<hierarchy rotation="0">
<node package="com.example.notes" class="android.widget.FrameLayout" bounds="[0,0][200,400]">
 <node package="com.example.notes" class="android.widget.EditText" bounds="[10,10][190,90]"
 text="Public note" content-desc="Note text" focused="true" password="false"/>
 <node package="com.example.notes" class="android.widget.Button" bounds="[10,100][190,150]"
 text="Save" clickable="true" resource-id="com.example.notes:id/save"/>
 <node package="com.example.notes" class="android.widget.EditText" bounds="[10,160][190,200]"
 text="secret-password" content-desc="secret-description" password="true">
  <node package="com.example.notes" class="android.widget.TextView" bounds="[10,160][190,180]" text="secret-child"/>
 </node>
</node></hierarchy>'''

HARDWARE_PROPERTIES = {'ro.product.model': 'Pixel 10 Pro', 'ro.soc.manufacturer': 'Google',
                       'ro.soc.model': 'Tensor G5', 'ro.board.platform': 'laguna',
                       'ro.product.cpu.abilist': 'arm64-v8a'}
CPU_RECORDS = '\n\n'.join('processor : ' + str(cpu) + '\nCPU implementer : 0x41\nCPU architecture : 8\nCPU part : ' +
                         ('0xd80' if cpu < 2 else '0xd87' if cpu < 7 else '0xd82') for cpu in range(8))


def hardware_files():
    files = {str(CPU_ROOT / 'present'): '0-7\n', str(CPU_ROOT / 'online'): '0-7\n', str(CPUINFO): CPU_RECORDS}
    for policy, cpus, maximum in (('policy0', '0 1', '1958000'), ('policy2', '2-6', '2850000'), ('policy7', '7', '3780000')):
        root = CPU_ROOT / 'cpufreq' / policy
        files.update({str(root / 'related_cpus'): cpus, str(root / 'cpuinfo_max_freq'): maximum,
                      str(root / 'scaling_max_freq'): maximum})
    return files


class FakePhone:
    def __init__(self):
        self.calls = []
        self.focus = FOCUS
        self.focus_sequence = []
        self.display = 0
        self.xml = XML
        self.volume = 5
        self.max_volume = 15
        self.deny_packages = False
        self.am_error = None
        self.root_error = None
        self.api_read_error = None
        self.api_set_error = None
        self.hook = None
        self.hardware_properties = dict(HARDWARE_PROPERTIES)
        self.hardware_error = None
        self.timeouts = []

    def __call__(self, argv, timeout=8, cancel_event=None):
        self.calls.append(list(argv))
        self.timeouts.append(timeout)
        root = argv[0] == '/system/bin/su'
        command = shlex.split(argv[2]) if root else list(argv)
        if self.hook:
            self.hook(command)
        if root and self.root_error:
            return subprocess.CompletedProcess(argv, 1, self.root_error)
        if command[0] == '/system/bin/getprop':
            if self.hardware_error:
                raise AppError(self.hardware_error)
            output = self.hardware_properties.get(command[1], '') + '\n'
        elif command[:2] == ['/system/bin/dumpsys', 'window']:
            focus = self.focus_sequence.pop(0) if self.focus_sequence else self.focus
            output = f'mCurrentFocus={focus}\nmTopFocusedDisplayId={self.display}\n'
        elif command[:3] == ['/system/bin/uiautomator', 'dump', '--compressed']:
            output = 'UI hierarchy dumped\n'
        elif command[0] == '/system/bin/cat':
            output = self.xml
        elif command[:4] == ['/system/bin/cmd', 'package', 'list', 'packages']:
            if self.deny_packages and not root:
                return subprocess.CompletedProcess(argv, 1, 'Permission denied')
            output = 'package:com.example.notes\npackage:com.example.music\npackage:com.example.notes\npackage:invalid;command\n'
        elif command[:3] == ['/system/bin/cmd', 'package', 'resolve-activity']:
            output = 'priority=0 preferredOrder=0 match=0x108000\ncom.example.notes/.MainActivity\n'
        elif command[0] == '/fake/termux-volume':
            if len(command) == 3:
                if self.api_set_error:
                    raise AppError(self.api_set_error)
                self.volume = int(command[2])
                output = ''
            else:
                if self.api_read_error:
                    raise AppError(self.api_read_error)
                output = json.dumps([{'stream': 'music', 'volume': self.volume, 'max_volume': self.max_volume}])
        elif command[:5] == ['/system/bin/cmd', 'media_session', 'volume', '--stream', '3']:
            if command[5] == '--set':
                self.volume = int(command[6])
            output = f'[V] volume is {self.volume} in range [0..{self.max_volume}]\n'
        elif command[0] == '/fake/am':
            if self.am_error:
                return subprocess.CompletedProcess(argv, 0, self.am_error)
            output = 'Starting: Intent\n'
        elif command[:2] == ['/system/bin/cmd', 'input']:
            output = ''
        else:
            raise AssertionError('Unexpected command: ' + repr(command))
        return subprocess.CompletedProcess(argv, 0, output)

    def input_commands(self):
        return [shlex.split(argv[2]) for argv in self.calls
                if argv[0] == '/system/bin/su' and shlex.split(argv[2])[:2] == ['/system/bin/cmd', 'input']]


class NPUHardwareTests(unittest.TestCase):
    def setUp(self):
        self.snapshot = {'schema': 'ostadix.pixel-tpu-telemetry/v1', 'collector_uid': 10402,
                         'utc': 'fixture-time', 'raw': {'firmware_version': 'fixture-fw',
                         'power/runtime_status': 'suspended'}, 'errors': {},
                         'driver': {'name': 'fixture-driver', 'path': '/sys/bus/platform/drivers/fixture'}}
        self.runner = Mock(return_value=subprocess.CompletedProcess([], 0, json.dumps(self.snapshot)))
        self.tools = PhoneTools(runner=self.runner)
        self.agent = SimpleNamespace(sid='npu-probe', cancelled=threading.Event(), execute=Mock())

    def read(self):
        return self.tools.execute(self.agent, self.agent.sid, ['@phone', 'npu'])

    def test_snapshot_reports_actual_probe_data_without_root_or_service_start(self):
        result = self.read()
        self.assertEqual(result['driver']['name'], 'fixture-driver')
        self.assertEqual(result['raw'], self.snapshot['raw'])
        self.assertTrue(result['read_only'])
        self.assertFalse(result['root_access_used'])
        self.agent.execute.assert_not_called()
        self.assertEqual(self.runner.call_count, 1)

    def test_permission_failure_retries_exact_read_through_policy_executor(self):
        self.snapshot['errors'] = {'power/runtime_status': '[Errno 13] Permission denied'}
        self.runner.return_value.stdout = json.dumps(self.snapshot)
        privileged = dict(self.snapshot, collector_uid=0, errors={})
        self.agent.execute.return_value = {'exit_code': 0, 'output': json.dumps(privileged)}
        result = self.read()
        request = self.agent.execute.call_args.args[0]
        self.assertEqual(request['argv'][:2], ['/system/bin/su', '-c'])
        self.assertEqual(shlex.split(request['argv'][2]), result['probe_argv'])
        self.assertEqual(request['timeout'], 5)
        self.assertTrue(result['root_access_used'])
        self.assertEqual(result['collector_uid'], 0)
        self.assertEqual(result['ordinary_read_errors'], self.snapshot['errors'])

    def test_denied_root_read_remains_partial_and_is_not_replayed(self):
        self.snapshot['errors'] = {'firmware_version': 'Permission denied'}
        self.runner.return_value.stdout = json.dumps(self.snapshot)
        self.agent.execute.side_effect = AppError('Action denied or cancelled; it was not executed')
        result = self.read()
        self.assertFalse(result['root_access_used'])
        self.assertIn('existing_root_read', result['errors'])
        self.assertEqual(self.agent.execute.call_count, 1)

    def test_cancel_after_ordinary_probe_prevents_root_read(self):
        def cancel(*args, **kwargs):
            self.agent.cancelled.set()
            return subprocess.CompletedProcess([], 0, json.dumps(self.snapshot))
        self.runner.side_effect = cancel
        with self.assertRaisesRegex(AppError, 'stale or cancelled'):
            self.read()
        self.agent.execute.assert_not_called()

    def test_malformed_or_timed_out_probe_does_not_gain_root_retry(self):
        self.runner.return_value.stdout = 'not JSON'
        with self.assertRaisesRegex(AppError, 'invalid JSON'):
            self.read()
        self.runner.side_effect = AppError('Phone command timed out')
        with self.assertRaisesRegex(AppError, 'timed out'):
            self.read()
        self.agent.execute.assert_not_called()


class PhoneToolsTests(unittest.TestCase):
    def setUp(self):
        self.backend = FakePhone()
        self.now = 100.
        self.tools = PhoneTools(runner=self.backend, clock=lambda: self.now)
        self.approvals = []
        self.updates = []
        self.approval_hook = None
        self.agent = SimpleNamespace(sid='session-a', cancelled=threading.Event(),
            store=SimpleNamespace(update=lambda *a, **kw: self.updates.append((a, kw))),
            wait_approval=self.approve)
        self.which = patch('phone_tools.shutil.which', side_effect=lambda name: {
            'am': '/fake/am', 'termux-volume': '/fake/termux-volume'}.get(name))
        self.which.start()
        self.addCleanup(self.which.stop)
        real_isfile = os.path.isfile
        self.root_present = patch('phone_tools.os.path.isfile', side_effect=lambda path:
                                  path == '/system/bin/su' or real_isfile(path))
        self.root_present.start()
        self.addCleanup(self.root_present.stop)
        self.sleep = patch('phone_tools.time.sleep', side_effect=self.advance)
        self.sleep.start()
        self.addCleanup(self.sleep.stop)
        self.hardware_files = hardware_files()
        self.hardware_reader = patch('hardware_info._read_text', side_effect=self.read_hardware)
        self.hardware_read_mock = self.hardware_reader.start()
        self.addCleanup(self.hardware_reader.stop)
        policies = [CPU_ROOT / 'cpufreq' / name for name in ('policy0', 'policy2', 'policy7')]
        self.hardware_policy_patch = patch('hardware_info._policies', return_value=policies)
        self.hardware_policies = self.hardware_policy_patch.start()
        self.addCleanup(self.hardware_policy_patch.stop)

    def read_hardware(self, path):
        if str(path) not in self.hardware_files:
            raise PermissionError('hardware path unavailable')
        return self.hardware_files[str(path)]

    def advance(self, seconds):
        self.now += seconds

    def approve(self, request):
        self.approvals.append(request)
        if self.approval_hook:
            self.approval_hook()

    def command(self, *args):
        return self.tools.execute(self.agent, self.agent.sid, ['@phone', *args])

    def screen(self):
        return self.command('screen')['screen_token']

    def test_invalid_argv_never_runs_a_command(self):
        invalid = [None, 'status', ['@phone'], ['wrong', 'status'], ['@phone', ['status']],
                   ['@phone', 'apps', 'foo;bar'], ['@phone', 'status', 'extra'],
                   ['@phone', 'open', 'com.example.notes;touch'], ['@phone', 'settings', 'danger'],
                   ['@phone', 'open', 'com.example.\x00notes'], ['@phone', 'volume', '-1'],
                   ['@phone', 'volume', '1.5'], ['@phone', 'volume', '1001']]
        invalid.extend([['@phone', 'cpu', 'extra'], ['@phone', 'hardware', '../cpu']])
        for argv in invalid:
            with self.subTest(argv=argv), self.assertRaises(AppError):
                self.tools.execute(self.agent, 'session-a', argv)
        self.assertEqual(self.backend.calls, [])
        self.assertEqual(self.approvals, [])

    def test_help_and_status_do_not_probe_root_or_invent_permissions(self):
        self.assertIn('commands', self.command('help'))
        self.assertIn('not probed', self.command('status')['root_access'])
        self.assertEqual(self.backend.calls, [])

    def test_apps_filter_deduplicates_and_does_not_need_approval(self):
        result = self.command('apps', 'example')
        self.assertEqual(result['packages'], ['com.example.music', 'com.example.notes'])
        self.assertEqual(self.backend.calls[0][-1], 'example')
        self.assertEqual(self.approvals, [])

    def test_cpu_reports_actual_device_soc_ids_and_cpufreq_without_approval(self):
        self.agent.engine_label = 'Huihui Gemma model on another backend'
        result = self.command('cpu')
        self.assertEqual(result['device_model'], 'Pixel 10 Pro')
        self.assertEqual(result['soc'], {'manufacturer': 'Google', 'model': 'Tensor G5'})
        self.assertEqual(result['board_platform'], 'laguna')
        self.assertEqual(result['abi'], ['arm64-v8a'])
        self.assertEqual(result['core_count'], 8)
        self.assertEqual([(item['part'], item['count']) for item in result['cpu_identifiers']],
                         [('0xd80', 2), ('0xd87', 5), ('0xd82', 1)])
        self.assertEqual([item['max_frequency_khz'] for item in result['clusters']], [1958000, 2850000, 3780000])
        self.assertEqual(result['sources']['soc.model']['property'], 'ro.soc.model')
        self.assertEqual(result['sources']['core_count'], str(CPU_ROOT / 'present'))
        self.assertEqual(result['unavailable'], {})
        self.assertEqual(self.backend.calls, [['/system/bin/getprop', key] for key in HARDWARE_PROPERTIES])
        self.assertTrue(all(timeout <= .35 for timeout in self.backend.timeouts))
        self.assertEqual(self.approvals, [])
        text = format_cpu_info(result)
        self.assertIn('CPU / SoC: Google Tensor G5', text)
        self.assertIn('CPU cores: 8', text)
        self.assertIn('2× up to 1.958 GHz', text)
        self.assertNotIn('Huihui', text)

    def test_hardware_alias_uses_the_same_read_only_query(self):
        result = self.command('hardware')
        self.assertEqual(result['soc']['model'], 'Tensor G5')
        self.assertTrue(result['read_only'])
        self.assertEqual(self.approvals, [])
        self.assertFalse(any(argv[0] == '/system/bin/su' for argv in self.backend.calls))

    def test_missing_soc_property_is_unavailable_not_the_model_or_board_name(self):
        del self.backend.hardware_properties['ro.soc.model']
        self.agent.engine_label = 'Local Gemini-like model'
        result = self.command('cpu')
        self.assertIsNone(result['soc']['model'])
        self.assertIn('soc.model', result['unavailable'])
        self.assertEqual(result['core_count'], 8)
        text = format_cpu_info(result)
        self.assertIn('CPU / SoC: unavailable', text)
        self.assertNotIn('Gemini', text)
        self.assertNotIn('CPU / SoC: laguna', text)

    def test_missing_cpuinfo_does_not_hide_kernel_present_core_count(self):
        del self.hardware_files[str(CPUINFO)]
        result = self.command('cpu')
        self.assertEqual(result['core_count'], 8)
        self.assertEqual(result['cpu_identifiers'], [])
        self.assertIn('cpu_identifiers', result['unavailable'])

    def test_missing_present_list_falls_back_only_to_observed_processor_records(self):
        del self.hardware_files[str(CPU_ROOT / 'present')]
        result = self.command('cpu')
        self.assertIsNone(result['present_cpus'])
        self.assertEqual(result['core_count'], 8)
        self.assertEqual(result['sources']['core_count'], str(CPUINFO))
        del self.hardware_files[str(CPUINFO)]
        with patch('os.cpu_count', return_value=128) as count:
            result = self.command('cpu')
        self.assertIsNone(result['core_count'])
        count.assert_not_called()

    def test_policy_ceiling_is_not_substituted_for_unreadable_hardware_maximum(self):
        del self.hardware_files[str(CPU_ROOT / 'cpufreq/policy0/cpuinfo_max_freq')]
        result = self.command('cpu')
        self.assertIsNone(result['clusters'][0]['max_frequency_khz'])
        self.assertEqual(result['clusters'][0]['scaling_max_frequency_khz'], 1958000)
        self.assertIn('clusters.policy0.max_frequency_khz', result['unavailable'])
        self.assertNotIn('1.958 GHz', format_cpu_info(result))

    def test_malformed_cpu_metadata_is_not_used_to_invent_hardware(self):
        self.hardware_files[str(CPU_ROOT / 'present')] = '0-999999999999'
        self.hardware_files[str(CPUINFO)] = CPU_RECORDS.replace('0xd80', 'not-a-cpu')
        self.hardware_files[str(CPU_ROOT / 'cpufreq/policy0/cpuinfo_max_freq')] = '1958 MHz'
        result = self.command('cpu')
        self.assertIsNone(result['core_count'])
        self.assertEqual(result['cpu_identifiers'], [])
        self.assertIsNone(result['clusters'][0]['max_frequency_khz'])
        self.assertEqual(result['status'], 'partial')

    def test_cpu_property_timeouts_stay_bounded_partial_and_never_try_root(self):
        self.backend.hardware_error = 'Phone command timed out'
        result = self.command('cpu')
        self.assertIsNone(result['soc']['model'])
        self.assertEqual(result['core_count'], 8)
        self.assertEqual(len(self.backend.calls), 5)
        self.assertTrue(all(timeout <= .35 for timeout in self.backend.timeouts))
        self.assertTrue(all(argv[0] == '/system/bin/getprop' for argv in self.backend.calls))
        self.assertEqual(self.approvals, [])

    def test_cpu_property_overall_budget_prevents_following_commands(self):
        self.backend.hook = lambda argv: setattr(self, 'now', self.now + 3)
        result = self.command('cpu')
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(result['device_model'], 'Pixel 10 Pro')
        self.assertIsNone(result['soc']['model'])
        self.assertIn('time budget', result['unavailable']['soc.model'])
        self.assertEqual(result['core_count'], 8)

    def test_cancellation_during_cpu_query_stops_without_approval_or_other_reads(self):
        self.backend.hook = lambda argv: self.agent.cancelled.set()
        with self.assertRaisesRegex(AppError, 'cancelled'):
            self.command('cpu')
        self.assertEqual(len(self.backend.calls), 1)
        self.hardware_read_mock.assert_not_called()
        self.assertEqual(self.approvals, [])

    def test_unexpected_policy_path_cannot_read_arbitrary_files(self):
        self.hardware_policies.return_value = [Path('/etc/policy0')]
        result = self.command('cpu')
        self.assertEqual(result['clusters'], [])
        self.assertIn('clusters', result['unavailable'])
        self.assertFalse(any(str(call.args[0]).startswith('/etc/') for call in self.hardware_read_mock.call_args_list))

    def test_cpu_help_directs_read_only_hardware_questions_without_new_approval(self):
        from phone_tools import INSTRUCTIONS
        self.assertIn('@phone cpu', self.command('help')['commands'])
        self.assertIn('needs no additional\nconfirmation', INSTRUCTIONS)
        self.assertIn("AI model's name", INSTRUCTIONS)
        self.assertEqual(self.backend.calls, [])

    def test_complete_package_list_is_not_cut_at_200(self):
        packages = ['com.example.app' + str(i).zfill(3) for i in range(457)]
        output = '\n'.join('package:' + package for package in reversed(packages))
        with patch.object(self.tools, '_read_command', return_value=subprocess.CompletedProcess([], 0, output)):
            result = self.command('apps')
        self.assertEqual(result['packages'], packages)
        self.assertEqual(result['count'], 457)
        self.assertFalse(result['truncated'])
        self.assertFalse(result['more'])

    def test_package_pages_recover_every_package_in_order(self):
        packages = ['com.example.app' + str(i).zfill(3) for i in range(457)]
        output = '\n'.join('package:' + package for package in packages)
        argv, received = ['@phone', 'apps', '', '0', '150'], []
        with patch.object(self.tools, '_read_command', return_value=subprocess.CompletedProcess([], 0, output)):
            while argv:
                result = self.tools.execute(self.agent, self.agent.sid, argv)
                received.extend(result['packages'])
                self.assertLessEqual(len(result['packages']), 150)
                argv = result['next_argv']
        self.assertEqual(received, packages)

    def test_package_reads_can_fall_back_to_existing_root(self):
        self.backend.deny_packages = True
        self.assertEqual(self.command('apps')['count'], 2)
        self.assertEqual(len(self.backend.calls), 2)
        self.assertEqual(self.backend.calls[-1][0], '/system/bin/su')

    def test_open_resolves_exact_package_and_no_extra_approval(self):
        result = self.command('open', 'com.example.notes')
        self.assertTrue(result['command_accepted'])
        self.assertEqual(self.backend.calls[-1], ['/fake/am', 'start', '-a', 'android.intent.action.MAIN',
            '-c', 'android.intent.category.LAUNCHER', '-n', 'com.example.notes/.MainActivity'])
        self.assertEqual(self.approvals, [])
        with self.assertRaisesRegex(AppError, 'exact package'):
            self.command('open', 'com.example.other')

    def test_settings_home_back_are_bounded_direct_controls(self):
        self.command('settings', 'audio')
        self.assertEqual(self.backend.calls[-1], ['/fake/am', 'start', '-a', 'android.settings.SOUND_SETTINGS'])
        self.command('home'); self.command('back')
        self.assertEqual(self.backend.input_commands(), [
            ['/system/bin/cmd', 'input', '-d', '0', 'keyevent', '3'],
            ['/system/bin/cmd', 'input', '-d', '0', 'keyevent', '4']])
        self.assertEqual(self.approvals, [])

    def test_failed_mutation_is_never_retried_with_root(self):
        self.backend.am_error = 'Error: Activity could not start'
        with self.assertRaisesRegex(AppError, 'Android phone operation failed'):
            self.command('settings', 'main')
        self.assertEqual(len(self.backend.calls), 1)
        self.assertEqual(self.backend.calls[0][0], '/fake/am')

    def test_media_volume_is_bounded_by_actual_max_and_verified(self):
        self.assertEqual(self.command('volume')['volume'], 5)
        result = self.command('volume', '9')
        self.assertEqual(result['stream'], 'music')
        self.assertTrue(result['verified'])
        self.assertEqual(result['volume'], 9)
        with self.assertRaisesRegex(AppError, '0 to 15'):
            self.command('volume', '16')
        self.assertEqual(self.backend.volume, 9)
        self.assertEqual(self.approvals, [])

    def test_screen_redacts_passwords_and_nested_secret_text(self):
        result = self.command('screen')
        output = json.dumps(result)
        self.assertIn('Public note', output)
        for secret in ('secret-password', 'secret-description', 'secret-child'):
            self.assertNotIn(secret, output)
        self.assertEqual(sum(node['password'] for node in result['nodes']), 2)
        self.assertEqual(result['foreground'], 'com.example.notes/.MainActivity')
        self.assertEqual((result['width'], result['height']), (200, 400))
        self.assertEqual(self.approvals, [])

    def test_volume_read_uses_existing_root_when_termux_api_times_out(self):
        self.backend.api_read_error = 'Phone command timed out'
        result = self.command('volume')
        self.assertEqual(result['volume'], 5)
        self.assertEqual(result['backend'], 'android-root')
        self.assertEqual(len(self.backend.calls), 2)
        self.command('volume')
        self.assertEqual(len(self.backend.calls), 3)  # Cache the working read backend.
        self.assertEqual(self.approvals, [])

    def test_volume_set_and_verification_stay_with_selected_backend(self):
        self.backend.api_read_error = 'Phone command timed out'
        result = self.command('volume', '7')
        self.assertTrue(result['verified'])
        self.assertEqual(result['backend'], 'android-root')
        commands = [shlex.split(argv[2]) for argv in self.backend.calls if argv[0] == '/system/bin/su']
        self.assertEqual([command[5] for command in commands], ['--get', '--set', '--get'])
        self.assertEqual(commands[1][-1], '7')

    def test_volume_mutation_timeout_is_not_replayed_via_root(self):
        self.backend.api_set_error = 'Phone command timed out'
        with self.assertRaisesRegex(AppError, 'timed out'):
            self.command('volume', '7')
        self.assertEqual(len(self.backend.calls), 2)
        self.assertFalse(any(argv[0] == '/system/bin/su' for argv in self.backend.calls))

    def test_volume_verification_failure_does_not_switch_backends(self):
        def break_read_after_set(command):
            if command[:2] == ['/fake/termux-volume', 'music']:
                self.backend.api_read_error = 'Phone command timed out'
        self.backend.hook = break_read_after_set
        with self.assertRaisesRegex(AppError, 'timed out'):
            self.command('volume', '7')
        self.assertEqual(self.backend.volume, 7)
        self.assertFalse(any(argv[0] == '/system/bin/su' for argv in self.backend.calls))

    def test_cancelled_volume_read_does_not_probe_fallback(self):
        def cancel_read(command):
            self.agent.cancelled.set()
            raise AppError('Phone command cancelled')
        self.backend.hook = cancel_read
        with self.assertRaisesRegex(AppError, 'cancelled'):
            self.command('volume')
        self.assertEqual(len(self.backend.calls), 1)

    def test_screen_requires_focused_default_display_and_matching_hierarchy(self):
        self.backend.display = 1
        with self.assertRaisesRegex(AppError, 'display 0'):
            self.screen()
        self.backend.display = 0
        self.backend.focus = 'Window{abc u0 NotificationShade}'
        with self.assertRaisesRegex(AppError, 'no app component'):
            self.screen()
        self.backend.focus = 'Window{abc u0 com.example.other/.Main}'
        with self.assertRaisesRegex(AppError, 'does not match'):
            self.screen()
        self.assertEqual(self.tools.screens, {})

    def test_screen_changed_during_capture_issues_no_token(self):
        self.backend.focus_sequence = [FOCUS, TERMUX_FOCUS]
        with self.assertRaisesRegex(AppError, 'changed during'):
            self.screen()
        self.assertEqual(self.tools.screens, {})

    def test_tap_requires_exact_approval_and_consumes_token_once(self):
        token = self.screen()
        result = self.command('tap', token, '20', '120')
        self.assertTrue(result['command_accepted'])
        self.assertEqual(self.approvals[0]['argv'], ['@phone', 'tap', token, '20', '120'])
        self.assertIn('Save', self.approvals[0]['description'])
        self.assertIn('within 12 seconds', self.approvals[0]['description'])
        self.assertEqual(self.backend.input_commands(), [['/system/bin/cmd', 'input', '-d', '0', 'tap', '20', '120']])
        with self.assertRaisesRegex(AppError, 'Screen token'):
            self.command('tap', token, '20', '120')
        self.assertEqual(len(self.approvals), 1)

    def test_approval_waits_for_return_from_termux_to_original_window(self):
        token = self.screen()
        self.approval_hook = lambda: setattr(self.backend, 'focus_sequence', [TERMUX_FOCUS, 'Window{c u0 NotificationShade}', FOCUS])
        self.command('tap', token, '20', '120')
        self.assertGreater(self.now, 100)
        self.assertEqual(len(self.backend.input_commands()), 1)

    def test_changed_window_after_approval_times_out_without_input(self):
        token = self.screen()
        self.approval_hook = lambda: setattr(self.backend, 'focus', TERMUX_FOCUS)
        with self.assertRaisesRegex(AppError, 'within 12 seconds'):
            self.command('tap', token, '20', '120')
        self.assertEqual(self.backend.input_commands(), [])
        self.assertNotIn(token, self.tools.screens)

    def test_token_must_belong_to_same_session_and_be_unexpired(self):
        token = self.screen()
        self.agent.sid = 'session-b'
        with self.assertRaisesRegex(AppError, 'another conversation'):
            self.command('tap', token, '20', '120')
        self.agent.sid = 'session-a'
        self.now += 61
        with self.assertRaisesRegex(AppError, 'expired'):
            self.command('tap', token, '20', '120')
        self.assertEqual(self.approvals, [])
        self.assertEqual(self.backend.input_commands(), [])

    def test_expiry_during_approval_or_return_never_sends_input(self):
        token = self.screen()
        self.approval_hook = lambda: self.advance(61)
        with self.assertRaisesRegex(AppError, 'expired'):
            self.command('tap', token, '20', '120')
        token = self.screen()
        def leave_little_time():
            self.advance(59.9)
            self.backend.focus = TERMUX_FOCUS
        self.approval_hook = leave_little_time
        with self.assertRaisesRegex(AppError, 'expired'):
            self.command('tap', token, '20', '120')
        self.assertEqual(self.backend.input_commands(), [])

    def test_cancel_or_denial_while_approving_never_sends_input(self):
        token = self.screen()
        self.approval_hook = self.agent.cancelled.set
        with self.assertRaisesRegex(AppError, 'cancelled'):
            self.command('tap', token, '20', '120')
        self.assertNotIn(token, self.tools.screens)
        self.agent.cancelled.clear()
        token = self.screen()
        def deny():
            raise AppError('User denied this action')
        self.approval_hook = deny
        with self.assertRaisesRegex(AppError, 'denied'):
            self.command('tap', token, '20', '120')
        self.assertNotIn(token, self.tools.screens)
        self.assertEqual(self.backend.input_commands(), [])

    def test_cancel_while_returning_to_app_prevents_input(self):
        token = self.screen()
        def cancel_on_foreground(command):
            if command[:2] == ['/system/bin/dumpsys', 'window']:
                self.agent.cancelled.set()
        self.backend.hook = cancel_on_foreground
        with self.assertRaisesRegex(AppError, 'cancelled'):
            self.command('tap', token, '20', '120')
        self.assertEqual(self.backend.input_commands(), [])

    def test_ui_coordinates_and_swipe_duration_validate_before_approval(self):
        token = self.screen()
        for args in [('tap', token, '200', '1'), ('tap', token, '1', '400'),
                     ('tap', token, '-1', '1'), ('swipe', token, '1', '1', '2', '2', '99'),
                     ('swipe', token, '1', '1', '2', '2', '2001')]:
            with self.subTest(args=args), self.assertRaises(AppError):
                self.command(*args)
        self.assertEqual(self.approvals, [])
        self.assertEqual(self.backend.input_commands(), [])
        self.command('swipe', token, '10', '350', '10', '30', '500')
        self.assertEqual(self.backend.input_commands()[-1],
                         ['/system/bin/cmd', 'input', '-d', '0', 'swipe', '10', '350', '10', '30', '500'])

    def test_typing_preserves_literal_shell_characters_and_does_not_press_enter(self):
        token = self.screen()
        value = "hello ' \" $(touch /never) ; x"
        self.command('type', token, value)
        self.assertEqual(self.approvals[0]['argv'], ['@phone', 'type', token, value])
        self.assertEqual(self.backend.input_commands(),
                         [['/system/bin/cmd', 'input', '-d', '0', 'text', value.replace(' ', '%s')]])

    def test_typing_rejects_passwords_unicode_controls_and_percent(self):
        token = self.screen()
        for value in ('', 'a\nb', '100%', 'café'):
            with self.subTest(value=value), self.assertRaises(AppError):
                self.command('type', token, value)
        for node in self.tools.screens[token]['nodes']:
            if node['editable'] and node['focused']:
                node['password'] = True
        with self.assertRaisesRegex(AppError, 'non-password'):
            self.command('type', token, 'abc')
        self.assertEqual(self.approvals, [])

    def test_automated_input_into_termux_is_blocked(self):
        token = self.screen()
        self.tools.screens[token]['focus'] = TERMUX_FOCUS
        with self.assertRaisesRegex(AppError, 'Termux is blocked'):
            self.command('tap', token, '20', '120')
        self.assertEqual(self.approvals, [])
        self.assertEqual(self.backend.input_commands(), [])

    def test_new_screen_or_direct_navigation_invalidates_old_token(self):
        old = self.screen()
        new = self.screen()
        self.assertNotIn(old, self.tools.screens)
        self.assertIn(new, self.tools.screens)
        self.command('settings', 'main')
        self.assertNotIn(new, self.tools.screens)

    def test_token_consumption_is_atomic_across_competing_requests(self):
        token = self.screen()
        barrier = threading.Barrier(2)
        results = []
        def consume():
            barrier.wait()
            try:
                results.append(self.tools._snapshot('session-a', token, consume=True))
            except AppError:
                results.append(None)
        workers = [threading.Thread(target=consume) for _ in range(2)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(1)
            self.assertFalse(worker.is_alive())
        self.assertEqual(sum(result is not None for result in results), 1)

    def test_stale_or_cancelled_request_does_not_start_any_command(self):
        with self.assertRaisesRegex(AppError, 'stale'):
            self.tools.execute(self.agent, 'other', ['@phone', 'home'])
        self.agent.cancelled.set()
        with self.assertRaisesRegex(AppError, 'cancelled'):
            self.command('home')
        self.assertEqual(self.backend.calls, [])


class PhoneRunnerTests(unittest.TestCase):
    def test_already_cancelled_command_never_spawns(self):
        event = threading.Event(); event.set()
        with patch('phone_tools.subprocess.Popen') as popen, self.assertRaisesRegex(AppError, 'before it started'):
            _run([sys.executable, '-c', 'pass'], cancel_event=event)
        popen.assert_not_called()

    def test_closed_output_does_not_hide_timeout(self):
        source = 'import os,time; os.close(1); os.close(2); time.sleep(20)'
        start = time.monotonic()
        with self.assertRaisesRegex(AppError, 'timed out'):
            _run([sys.executable, '-c', source], timeout=.2)
        self.assertLess(time.monotonic() - start, 2)

    def test_closed_output_does_not_hide_cancellation(self):
        event = threading.Event()
        timer = threading.Timer(.15, event.set)
        timer.start()
        start = time.monotonic()
        try:
            with self.assertRaisesRegex(AppError, 'cancelled'):
                _run([sys.executable, '-c', 'import os,time; os.close(1); os.close(2); time.sleep(20)'],
                     timeout=10, cancel_event=event)
        finally:
            timer.cancel()
        self.assertLess(time.monotonic() - start, 2)


if __name__ == '__main__':
    unittest.main()
