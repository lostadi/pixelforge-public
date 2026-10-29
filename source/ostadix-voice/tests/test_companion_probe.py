import subprocess
import unittest
from unittest.mock import patch

import companion_probe as probe


def power(event=2, previous=0, last=999000, previous_duration=900000,
          current_duration=800000, wake='Awake', group_wake=1, interactive='true'):
    return f'''POWER MANAGER
  mWakefulness={wake}
Power Group User Activity:
groupId: 0
lastUserActivityTime={last} (1000 ms ago)
lastUserActivityTimeNoChangeLights=900000 (100000 ms ago)
mWakefulness={group_wake}

Wakefulness Session Observer:
  Wakefulness Session Power Group powerGroupId: 0
    current wakefulness: {group_wake}
    current user activity event: {event}
    current user activity duration: {current_duration}
    previous user activity event: {previous}
    previous user activity duration: {previous_duration}
    mIsInteractive: {interactive}
'''


def policy(showing='false', restricted='false', screen='ON', interactive='AWAKE'):
    return f'''Policy:
    KeyguardServiceDelegate
      showing={showing}
      inputRestricted={restricted}
      trusted=true
      screenState=SCREEN_STATE_{screen}
      interactiveState=INTERACTIVE_STATE_{interactive}
    Looper state:
      showing=true
'''


class CompanionProbeTests(unittest.TestCase):
    def test_continuous_touch_uses_latest_activity_not_event_transition(self):
        result = probe.parse_activity(power(current_duration=800000), policy(), 1000)
        self.assertTrue(result['active'])
        self.assertEqual(result['last_user_activity_age_seconds'], 1)
        self.assertEqual(result['activity_source'], 'current_touch')

    def test_locked_or_restricted_even_if_trusted_never_active(self):
        for opts in ({'showing': 'true'}, {'restricted': 'true'}, {'showing': 'unknown'}):
            with self.subTest(opts=opts):
                self.assertFalse(probe.parse_activity(power(), policy(**opts), 1000)['active'])

    def test_sleep_dream_doze_and_transition_never_active(self):
        for wake, group in [('Asleep', 0), ('Dreaming', 2), ('Dozing', 3), ('Awake', 3)]:
            with self.subTest(wake=wake, group=group):
                self.assertFalse(probe.parse_activity(power(wake=wake, group_wake=group), policy(), 1000)['active'])
        for opts in ({'screen': 'OFF'}, {'screen': 'TURNING_ON'}, {'interactive': 'SLEEP'}):
            self.assertFalse(probe.parse_activity(power(), policy(**opts), 1000)['active'])

    def test_attention_and_other_activity_are_insufficient(self):
        for event in (0, 3, 4, 5, 6):
            self.assertFalse(probe.parse_activity(power(event=event), policy(), 1000)['active'])

    def test_recent_previous_physical_event_survives_other_activity(self):
        result = probe.parse_activity(power(event=0, previous=1, previous_duration=20000), policy(), 1000)
        self.assertTrue(result['active'])
        self.assertEqual(result['last_user_activity_age_seconds'], 20)
        self.assertEqual(result['activity_source'], 'previous_button')
        self.assertFalse(probe.parse_activity(power(event=0, previous=1, previous_duration=120001), policy(), 1000)['active'])

    def test_timeout_boundary_and_no_change_lights(self):
        # Both timestamps are old; a recent no-change-lights timestamp may qualify.
        old = power(last=880000).replace('NoChangeLights=900000', 'NoChangeLights=800000')
        self.assertTrue(probe.parse_activity(old, policy(), 1000)['active'])
        self.assertFalse(probe.parse_activity(old.replace('Time=880000', 'Time=879999'), policy(), 1000)['active'])
        self.assertTrue(probe.parse_activity(old.replace('NoChangeLights=800000', 'NoChangeLights=999000'), policy(), 1000)['active'])

    def test_future_and_missing_activity_fail_closed(self):
        future = power(last=1000001).replace('NoChangeLights=900000', 'NoChangeLights=1000001')
        self.assertFalse(probe.parse_activity(future, policy(), 1000)['active'])
        self.assertFalse(probe.parse_activity(power().replace('lastUserActivityTime', 'removed'), policy(), 1000)['active'])
        self.assertFalse(probe.parse_activity(power(event=0, previous=2, previous_duration=1000001), policy(), 1000)['active'])

    def test_default_display_only_and_duplicate_groups_rejected(self):
        other = power().replace('groupId: 0', 'groupId: 1').replace('powerGroupId: 0', 'powerGroupId: 1')
        self.assertFalse(probe.parse_activity(other, policy(), 1000)['active'])
        duplicate = power().replace('groupId: 0', 'groupId: 0\nlastUserActivityTime=999000\nmWakefulness=1\ngroupId: 0')
        self.assertFalse(probe.parse_activity(duplicate, policy(), 1000)['active'])

    def test_unknown_missing_and_duplicate_fields_fail_closed(self):
        samples = ['', power().replace('    mIsInteractive: true', ''),
                   power().replace('    current user activity event: 2', ''),
                   power().replace('  mWakefulness=Awake', '  mWakefulness=Awake\n  mWakefulness=Awake')]
        for value in samples:
            with self.subTest(value=value):
                self.assertFalse(probe.parse_activity(value, policy(), 1000)['active'])
        self.assertFalse(probe.parse_activity(power(), '', 1000)['active'])

    def test_invalid_clock_and_dnd_are_not_truthy_defaults(self):
        for now in (float('nan'), float('inf'), -1, True, None):
            self.assertFalse(probe.parse_activity(power(), policy(), now)['active'])
        self.assertIs(probe.parse_dnd('0\n'), False)
        for value in ('1', '2', '3'):
            self.assertIs(probe.parse_dnd(value), True)
        for value in ('', 'null', '0\n1', '4', None):
            self.assertIsNone(probe.parse_dnd(value))

    def test_sampling_has_only_fixed_read_commands_and_no_content(self):
        with patch.object(probe, '_read_command', side_effect=[power(), policy(), '0']) as run, \
                patch.object(probe.time, 'monotonic', side_effect=[1000, 1000, 1000.1]):
            result = probe.sample()
        self.assertTrue(result['active'])
        self.assertEqual(run.call_args_list[0].args[0], ['/system/bin/dumpsys', 'power'])
        self.assertEqual(run.call_args_list[1].args[0], ['/system/bin/dumpsys', 'window', 'policy'])
        self.assertEqual(run.call_args_list[2].args[0], ['/system/bin/settings', 'get', 'global', 'zen_mode'])
        self.assertNotIn('power', result)
        self.assertNotIn('policy', result)

    def test_timeout_and_slow_snapshot_fail_closed(self):
        with patch.object(probe, '_read_command', side_effect=subprocess.TimeoutExpired('dumpsys', 2)):
            result = probe.sample()
        self.assertFalse(result['active'])
        self.assertEqual(result['errors']['power'], 'TimeoutExpired')
        with patch.object(probe, '_read_command', side_effect=[power(), policy(), '0']), \
                patch.object(probe.time, 'monotonic', side_effect=[1000, 1000, 1006]):
            result = probe.sample()
        self.assertFalse(result['active'])
        self.assertEqual(result['reason'], 'stale_snapshot')


if __name__ == '__main__':
    unittest.main()
