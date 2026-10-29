"""Notification content isolation, quiet delivery and truthful readback."""
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import companion_notify as notify


class CompanionNotifyTests(unittest.TestCase):
    def setUp(self):
        self.run = self.enterContext(patch.object(notify.subprocess, 'run'))
        self.enterContext(patch.object(notify, '_api_environment', return_value={'PATH': notify.PREFIX + '/bin'}))
        self.enterContext(patch.object(notify.os, 'getuid', return_value=notify.TERMUX_UID))
        self.enterContext(patch.object(notify.time, 'sleep'))
        self.enterContext(patch.object(notify.time, 'time', return_value=1000))

    def reply(self, stdout='', code=0, stderr=''):
        return subprocess.CompletedProcess([], code, stdout, stderr)

    def channel(self):
        return self.reply('Created channel with id "' + notify.CHANNEL_ID + '" and name "Ostadix companion".')

    def record(self, text='What do you want to learn?', when=1000000):
        return self.reply('NotificationRecord(0x1: pkg=com.termux.api user=UserHandle{0} '
                          'key=' + notify.KEY + ': Notification(channel=' + notify.CHANNEL_ID + '))\n'
                          'when=' + str(when) + '/1000000\nandroid.text=String (' + text + ')\n'
                          'effectiveNotificationChannel=NotificationChannel{mImportance=2, mSound=null}')

    def absent(self):
        return self.reply('error: no active notification matching key: ' + notify.KEY, 1)

    def test_text_is_unicode_bounded_and_controls_removed(self):
        self.assertEqual(notify.sanitize_text('\x1b[31mCafé\x1b[0m\n\u202e world\0'), 'Café world')
        self.assertEqual(notify.sanitize_text('a' * 500), 'a' * 359 + '…')
        for value in ('\0\u202e', None, 'a' * 9000):
            with self.assertRaises(ValueError):
                notify.sanitize_text(value)

    def test_verified_delivery_and_fixed_action_separate_from_model_text(self):
        text = 'Learn $(touch /tmp/never); `whoami` --action?'
        self.run.side_effect = [self.channel(), self.reply(), self.record(text)]
        result = notify.deliver(text, checkin_id='checkin-1')
        self.assertTrue(result['ok']); self.assertEqual(result['status'], 'delivered')
        post = self.run.call_args_list[1]
        self.assertNotIn(text, post.args[0])
        self.assertEqual(post.kwargs['input'], text + '\n')
        self.assertIn(notify.TAP_ACTION, post.args[0])
        self.assertNotIn('--sound', post.args[0]); self.assertNotIn('--vibrate', post.args[0])
        for call in self.run.call_args_list:
            self.assertNotIn('shell', call.kwargs)
            self.assertEqual(call.kwargs['timeout'], notify.TIMEOUT)
        self.assertIn("'" + notify.KEY + "'", self.run.call_args_list[2].args[0][-1])

    def test_only_fixed_notification_is_read_and_no_all_notification_listing(self):
        self.run.side_effect = [self.channel(), self.reply(), self.record()]
        notify.deliver('What do you want to learn?')
        command = self.run.call_args_list[2].args[0]
        self.assertEqual(command[:2], ['/system/bin/su', '-c'])
        self.assertIn(' notification get ', command[2])
        self.assertNotIn('notification-list', command[2]); self.assertNotIn('notification list', command[2])

    def test_api_zero_exit_with_error_text_is_not_delivery(self):
        self.run.side_effect = [self.channel(), self.reply('{"error":"permission denied"}')]
        result = notify.deliver('Hello?')
        self.assertFalse(result['ok']); self.assertEqual(result['status'], 'unknown')
        self.assertEqual(self.run.call_count, 2)

    def test_channel_failure_stops_before_posting(self):
        self.run.return_value = self.reply('Could not create/delete channel.')
        self.assertFalse(notify.deliver('Hello?')['ok'])
        self.assertEqual(self.run.call_count, 1)

    def test_preflight_is_after_channel_and_requires_exact_true(self):
        for allowed in (False, None, 1, {}, 'yes'):
            self.run.reset_mock()
            self.run.return_value = self.channel()
            def preflight():
                self.assertEqual(self.run.call_count, 1)
                return allowed
            result = notify.deliver('Hello?', before_post=preflight)
            self.assertEqual(result['status'], 'suppressed')
            self.assertFalse(result['ok'])
            self.assertEqual(self.run.call_count, 1)

    def test_preflight_exception_suppresses_without_exposing_details(self):
        self.run.return_value = self.channel()
        result = notify.deliver('Hello?', before_post=Mock(side_effect=RuntimeError('private diagnostic')))
        self.assertEqual(result['status'], 'suppressed')
        self.assertNotIn('private', str(result))
        self.assertEqual(self.run.call_count, 1)

    def test_preflight_true_allows_verified_post(self):
        self.run.side_effect = [self.channel(), self.reply(), self.record()]
        preflight = Mock(return_value=True)
        self.assertTrue(notify.deliver('What do you want to learn?', before_post=preflight)['ok'])
        preflight.assert_called_once_with()

    def test_permission_failure_is_explicit_and_does_not_request_permission(self):
        self.run.side_effect = [self.channel(), self.reply(code=1, stderr='denied')]
        result = notify.deliver('Hello?')
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(self.run.call_count, 2)

    def test_missing_and_stale_notification_are_not_claimed_delivered(self):
        for evidence in (self.absent(), self.record('Old question?'), self.record(when=900000)):
            self.run.side_effect = [self.channel(), self.reply(), evidence, evidence, evidence]
            self.assertEqual(notify.deliver('What do you want to learn?')['status'], 'unknown')

    def test_asynchronous_enqueue_retries_then_confirms(self):
        self.run.side_effect = [self.channel(), self.reply(), self.absent(), self.record()]
        self.assertTrue(notify.deliver('What do you want to learn?')['ok'])

    def test_readback_denied_reports_accepted_but_unknown(self):
        self.run.side_effect = [self.channel(), self.reply(), self.reply(code=255)]
        result = notify.deliver('Hello?')
        self.assertTrue(result['accepted']); self.assertFalse(result['ok'])
        self.assertEqual(result['status'], 'unknown')

    def test_timeout_is_uncertain_and_bounded(self):
        self.run.side_effect = subprocess.TimeoutExpired('api', notify.TIMEOUT)
        result = notify.deliver('Hello?')
        self.assertEqual(result['status'], 'unknown'); self.assertIn('timeout', result['reason'])

    def test_missing_binary_is_failed(self):
        self.run.side_effect = FileNotFoundError('missing')
        self.assertEqual(notify.deliver('Hello?')['status'], 'failed')

    def test_root_cannot_post_or_remove(self):
        with patch.object(notify.os, 'getuid', return_value=0):
            self.assertEqual(notify.deliver('Hello?')['status'], 'failed')
            self.assertEqual(notify.clear()['status'], 'failed')
        self.run.assert_not_called()

    def test_invalid_input_never_executes(self):
        self.assertEqual(notify.deliver('\0')['status'], 'failed')
        with self.assertRaises(ValueError): notify.deliver('Hello?', checkin_id='$(bad)')
        self.run.assert_not_called()

    def test_clear_only_own_id_and_confirm_absence(self):
        self.run.side_effect = [self.reply(), self.absent()]
        result = notify.clear()
        self.assertTrue(result['ok']); self.assertEqual(result['status'], 'removed')
        self.assertEqual(self.run.call_args_list[0].args[0],
                         [notify.API, 'NotificationRemove', '--es', 'id', notify.NOTIFICATION_ID])

    def test_clear_waits_for_asynchronous_removal(self):
        self.run.side_effect = [self.reply(), self.record(), self.absent()]
        self.assertTrue(notify.clear()['verified'])

    def test_clear_does_not_claim_success_if_readback_unavailable(self):
        self.run.side_effect = [self.reply(), self.reply(code=255)]
        self.assertEqual(notify.clear()['status'], 'unknown')


class CompanionNotifyEnvironmentTests(unittest.TestCase):
    def test_parse_only_required_init_exports_without_shell_execution(self):
        values = notify._parse_classpaths('export BOOTCLASSPATH /apex/com.android.art/javalib/core-oj.jar:/system/framework/framework.jar\n'
                                         'export DEX2OATBOOTCLASSPATH /apex/com.android.art/javalib/core-oj.jar\n'
                                         'export SYSTEMSERVERCLASSPATH /system/framework/services.jar\n')
        self.assertEqual(set(values), {'BOOTCLASSPATH', 'DEX2OATBOOTCLASSPATH'})
        for line in ('export BOOTCLASSPATH $(bad)', 'export BOOTCLASSPATH /data/private.jar',
                     'export BOOTCLASSPATH /system/../data/private.jar'):
            with self.assertRaises(ValueError):
                notify._parse_classpaths(line + '\nexport DEX2OATBOOTCLASSPATH /system/framework/core.jar\n')

    def test_metadata_read_is_fixed_cached_and_only_affects_child_environment(self):
        reply = subprocess.CompletedProcess([], 0,
            'export BOOTCLASSPATH /system/framework/boot.jar\n'
            'export DEX2OATBOOTCLASSPATH /system/framework/dex.jar\n', '')
        with patch.object(notify, '_classpath_cache', None), patch.object(notify.os, 'environ', {'PATH': '/example'}), \
                patch.object(notify.subprocess, 'run', return_value=reply) as run:
            first = notify._api_environment()
            second = notify._api_environment()
            self.assertEqual(first, second)
            self.assertEqual(first['ANDROID_ART_ROOT'], '/apex/com.android.art')
            self.assertEqual(first['BOOTCLASSPATH'], '/system/framework/boot.jar')
            self.assertEqual(notify.os.environ, {'PATH': '/example'})
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0], ['/system/bin/su', '-c',
                                                    '/system/bin/cat /data/system/environ/classpath'])

    def test_unavailable_metadata_does_not_start_api(self):
        with patch.object(notify, '_classpath_cache', None), \
                patch.object(notify.subprocess, 'run', return_value=subprocess.CompletedProcess([], 1, '', 'denied')) as run:
            result = notify._run([notify.API, 'NotificationChannel'])
            self.assertFalse(result['ok'])
            self.assertEqual(result['error'], 'Android runtime metadata unavailable')
            self.assertEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main()
