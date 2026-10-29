import json
import unittest

from optimizer_policy import PolicyError, decide, new_state, validate_recommendation


def sample(now, **changes):
    value = {'sampled_at': now, 'battery_percent': 70, 'battery_temp_c': 32,
             'charging': False, 'thermal_severity': 0, 'memory_available_mb': 4000,
             'cpu_busy_pct': 25, 'active': 0, 'queued': 0}
    value.update(changes)
    return value


NORMAL = {'action': 'normal', 'reason': 'Restore the baseline.'}
ECO = {'action': 'eco', 'reason': 'Reduce background CPU priority.'}


class OptimizerPolicyTests(unittest.TestCase):
    def test_strict_model_recommendation(self):
        self.assertEqual(validate_recommendation(json.dumps(ECO)), ECO)
        invalid = [None, [], {}, {'action': 'shell', 'reason': 'do it'},
                   dict(ECO, command='su -c reboot'), {'action': 'eco', 'reason': ''},
                   {'action': 'eco', 'reason': 'a' * 241}, {'action': 'eco', 'reason': 'line\nbreak'},
                   '{"action":"eco","action":"normal","reason":"x"}',
                   '\ud800',
                   '```json\n{"action":"eco","reason":"x"}\n```']
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(PolicyError):
                validate_recommendation(value)

    def test_missing_invalid_stale_or_future_telemetry_never_changes_priority(self):
        for key in sample(100):
            if key == 'thermal_severity':
                continue
            value = sample(100, battery_temp_c=50)
            del value[key]
            self.assertEqual(decide(value, ECO, now=100)['action'], 'observe')
        for changes in ({'sampled_at': 79}, {'sampled_at': 101}, {'cpu_busy_pct': None},
                        {'battery_percent': float('nan')}, {'charging': 1}, {'active': True}):
            self.assertEqual(decide(sample(100, **changes), ECO, now=100)['action'], 'observe')

    def test_critical_gates_do_not_need_model_and_override_cooldown(self):
        state = new_state()
        state['last_change_at'] = 99
        for changes in ({'thermal_severity': 3}, {'battery_temp_c': 45}, {'battery_percent': 10}):
            out = decide(sample(100, **changes), None, state, now=100)
            self.assertEqual((out['action'], out['nice_delta']), ('set_nice', 10))
        self.assertEqual(decide(sample(100, battery_percent=10, charging=True), now=100)['action'], 'observe')

    def test_noncritical_change_requires_model_approval(self):
        out = decide(sample(100, cpu_busy_pct=95), None, now=100)
        self.assertEqual(out['action'], 'observe')
        self.assertEqual(out['next_state']['candidate_delta'], None)

    def test_hysteresis_then_cooldown_and_stable_baseline_restore(self):
        state = decide(sample(100, cpu_busy_pct=95), ECO, now=100)['next_state']
        self.assertEqual(decide(sample(129, cpu_busy_pct=95), ECO, state, now=129)['action'], 'observe')
        out = decide(sample(130, cpu_busy_pct=95), ECO, state, now=130)
        self.assertEqual((out['action'], out['nice_delta']), ('set_nice', 5))
        state = out['next_state']
        state = decide(sample(131), NORMAL, state, now=131)['next_state']
        self.assertEqual(decide(sample(189), NORMAL, state, now=189)['action'], 'observe')
        out = decide(sample(190), NORMAL, state, now=190)
        self.assertEqual((out['action'], out['nice_delta']), ('set_nice', 0))

    def test_missing_sample_or_conflicting_recommendation_resets_candidate(self):
        state = decide(sample(100, cpu_busy_pct=95), ECO, now=100)['next_state']
        state = decide(sample(129, cpu_busy_pct=None), ECO, state, now=129)['next_state']
        self.assertIsNone(state['candidate_delta'])
        self.assertEqual(decide(sample(130, cpu_busy_pct=95), ECO, state, now=130)['action'], 'observe')

    def test_critical_recovery_has_distinct_temperature_battery_thresholds(self):
        state = decide(sample(100, battery_temp_c=45), now=100)['next_state']
        out = decide(sample(200, battery_temp_c=41), NORMAL, state, now=200)
        self.assertEqual((out['action'], out['nice_delta']), ('observe', 10))
        out = decide(sample(201, battery_temp_c=40, battery_percent=14), NORMAL, state, now=201)
        self.assertEqual(out['nice_delta'], 10)
        state = decide(sample(202, battery_temp_c=40, battery_percent=30), NORMAL, state, now=202)['next_state']
        out = decide(sample(232, battery_temp_c=40, battery_percent=30), NORMAL, state, now=232)
        self.assertEqual((out['action'], out['nice_delta']), ('set_nice', 0))

    def test_objectives_change_only_bounded_policy_conditions(self):
        telemetry = sample(100)
        self.assertIsNone(decide(telemetry, ECO, objective='balanced', now=100)['next_state']['candidate_delta'])
        self.assertEqual(decide(telemetry, ECO, objective='battery', now=100)['next_state']['candidate_delta'], 5)
        self.assertIsNone(decide(sample(100, cpu_busy_pct=99), ECO, objective='performance', now=100)['next_state']['candidate_delta'])

    def test_critical_eco_can_relax_to_regular_eco_after_recovery(self):
        state = decide(sample(100, battery_temp_c=45), now=100)['next_state']
        state = decide(sample(200), ECO, state, now=200)['next_state']
        out = decide(sample(230), ECO, state, now=230)
        self.assertEqual((out['action'], out['nice_delta']), ('set_nice', 5))

    def test_measured_rollback_is_bound_to_trial_and_obeys_cooldown(self):
        state = decide(sample(100, cpu_busy_pct=95), ECO, now=100)['next_state']
        state = decide(sample(130, cpu_busy_pct=95), ECO, state, now=130)['next_state']
        evaluation = {'change_at': 130, 'regression': True}
        self.assertEqual(decide(sample(189), state=state, now=189, evaluation=evaluation)['action'], 'observe')
        out = decide(sample(190), state=state, now=190, evaluation=evaluation)
        self.assertEqual((out['action'], out['nice_delta']), ('set_nice', 0))
        self.assertIsNone(out['next_state']['trial'])
        stale = {'change_at': 129, 'regression': True}
        self.assertEqual(decide(sample(190), state=state, now=190, evaluation=stale)['action'], 'observe')

    def test_rollback_cannot_override_critical_conditions(self):
        state = decide(sample(100, cpu_busy_pct=95), ECO, now=100)['next_state']
        state = decide(sample(130, cpu_busy_pct=95), ECO, state, now=130)['next_state']
        out = decide(sample(190, battery_temp_c=46), state=state, now=190,
                     evaluation={'change_at': 130, 'regression': True})
        self.assertEqual(out['nice_delta'], 10)

    def test_inputs_are_not_mutated_and_priority_never_exceeds_baseline(self):
        telemetry, state = sample(100, battery_temp_c=50), new_state()
        before = json.dumps([telemetry, state])
        out = decide(telemetry, ECO, state, now=100)
        self.assertEqual(json.dumps([telemetry, state]), before)
        self.assertIn(out['nice_delta'], (0, 5, 10))
        self.assertEqual(set(out), {'action', 'mode', 'nice_delta', 'reason', 'next_state'})

    def test_invalid_state_clock_and_objective_fail_closed(self):
        for changes in ({'nice_delta': -5}, {'last_change_at': 101}, {'candidate_since': 1}):
            with self.assertRaises(PolicyError):
                decide(sample(100), ECO, dict(new_state(), **changes), now=100)
        for value in (None, float('nan'), -1, True, 10 ** 1000):
            with self.assertRaises(PolicyError):
                decide(sample(100), now=value)
        with self.assertRaises(PolicyError):
            decide(sample(100), now=100, objective='overclock')
        with self.assertRaises(PolicyError):
            decide(sample(100), now=100, objective={})


if __name__ == '__main__':
    unittest.main()
