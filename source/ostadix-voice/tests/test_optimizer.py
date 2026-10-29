"""Monitor/controller regressions with fake telemetry, NPU and process tuning."""
import copy
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import optimizer
import optimizer_probe
from local_runtime import LocalError


class OptimizerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name) / 'local'
        # Any unintended command execution fails this suite immediately.
        self.root_call = self.enterContext(patch('optimizer.root_call', side_effect=AssertionError('No root calls in tests')))
        self.popen = self.enterContext(patch('optimizer.subprocess.Popen', side_effect=AssertionError('No process starts in tests')))

    def configure(self, enabled=True, local_enabled=True):
        state = self.base / 'optimizer'
        state.mkdir(parents=True, exist_ok=True)
        (state / 'config.json').write_text(json.dumps({'version': 1, 'enabled': enabled, 'objective': 'balanced'}))
        (self.base / 'service.json').write_text(json.dumps({'version': 1, 'enabled': local_enabled}))
        return state

    def monitor(self):
        self.configure()
        self.clock = 1000.0
        self.enterContext(patch('optimizer.time.monotonic', side_effect=lambda: self.clock))
        self.enterContext(patch('optimizer.process_stamp', return_value=456))
        self.telemetry = {'sampled_at': self.clock, 'battery_percent': 80, 'battery_temp_c': 32,
                          'charging': True, 'memory_available_mb': 4096, 'thermal_severity': 0,
                          'cpu_totals': {'total': 1100, 'idle': 550}}
        def fresh_sample():
            result = copy.deepcopy(dict(self.telemetry, sampled_at=self.clock))
            self.telemetry['cpu_totals']['total'] += 100
            self.telemetry['cpu_totals']['idle'] += 50
            return result
        sample = Mock(side_effect=fresh_sample)
        self.client = Mock()
        self.client.status.return_value = {'ready': True, 'active': 0, 'queued': 0,
                                           'engine_pid': 123, 'model': {'backend': 'litert-npu'}}
        self.client.submit.return_value = {'id': 'analysis-1', 'state': 'queued'}
        self.client.read.return_value = {'id': 'analysis-1', 'state': 'completed',
            'result': '{"action":"normal","reason":"Conditions are healthy."}',
            'result_truncated': False, 'usage': {'backend': 'litert-npu'}}
        self.tuner = Mock()
        self.tuner.restore.return_value = {'ok': True, 'restored': True}
        self.tuner.apply.return_value = {'ok': True, 'verified': True}
        result = optimizer.Monitor(self.base, sample=sample, client=self.client, tuner=self.tuner)
        self.addCleanup(result.db.close)
        result.last_cpu = {'total': 1000, 'idle': 500}
        return result

    def test_status_history_and_disabled_resume_do_not_create_or_start(self):
        control = optimizer.OptimizerControl(self.base)
        self.assertFalse(control.status()['enabled'])
        self.assertEqual(control.history()['events'], [])
        self.assertFalse(control.resume()['running'])
        self.assertFalse(self.base.exists())
        self.root_call.assert_not_called(); self.popen.assert_not_called()

    def test_resume_respects_explicitly_disabled_local_service(self):
        self.configure(local_enabled=False)
        with patch('optimizer.os.getuid', return_value=10402):
            status = optimizer.OptimizerControl(self.base).resume()
        self.assertIn('local service disabled', status['phase'])
        self.assertFalse(status['running'])
        self.popen.assert_not_called(); self.root_call.assert_not_called()

    def test_only_default_policy_base_selects_independent_npu_client(self):
        npu, local = Mock(), Mock()
        with patch('optimizer.DEFAULT_STATE', self.base), \
                patch('npu_tools.NPUClient', return_value=npu) as npu_factory:
            self.assertIs(optimizer.worker_client(self.base, local_factory=local), npu)
            other = self.base.parent / 'custom'
            self.assertIs(optimizer.worker_client(other, local_factory=local), local.return_value)
        npu_factory.assert_called_once_with()
        local.assert_called_once_with(other)
        npu.start.assert_not_called()

    def test_disabled_production_npu_waits_and_restores_without_starting_it(self):
        self.configure()
        npu = Mock(state=self.base.parent / 'npu')
        npu.status.return_value = {'running': False, 'ready': False, 'model': None}
        tuner = Mock(); tuner.restore.return_value = {'ok': True}
        sample = Mock(return_value={'sampled_at': 1000, 'cpu_totals': {'total': 100, 'idle': 50}})
        with patch('optimizer.DEFAULT_STATE', self.base), patch('npu_tools.NPUClient', return_value=npu):
            monitor = optimizer.Monitor(self.base, sample=sample, tuner=tuner)
        self.addCleanup(monitor.db.close)
        monitor.tick()
        self.assertEqual(monitor.snapshot['phase'], 'waiting for NPU model')
        tuner.restore.assert_called_once_with()
        npu.start.assert_not_called(); npu.submit.assert_not_called()

    def test_history_is_read_only_newest_first_and_bounded(self):
        state = self.configure()
        path = state / 'history.sqlite3'
        with closing(sqlite3.connect(path)) as db, db:
            db.execute('CREATE TABLE events(id INTEGER PRIMARY KEY, data TEXT NOT NULL)')
            db.executemany('INSERT INTO events(data) VALUES(?)', [(json.dumps({'number': n}),) for n in range(5)])
        before = path.read_bytes()
        control = optimizer.OptimizerControl(self.base)
        self.assertEqual([row['number'] for row in control.history(2)['events']], [4, 3])
        self.assertEqual(path.read_bytes(), before)
        for limit in (0, 101, True, '10'):
            with self.subTest(limit=limit), self.assertRaises(LocalError):
                control.history(limit)
        self.root_call.assert_not_called(); self.popen.assert_not_called()

    def test_tick_submits_only_idle_npu_work_with_fixed_owner_and_budget(self):
        monitor = self.monitor()
        cases = ({'active': 1}, {'queued': 1}, {'ready': False}, {'model': {'backend': 'llama.cpp'}})
        baseline = copy.deepcopy(self.client.status.return_value)
        for changes in cases:
            with self.subTest(changes=changes):
                self.client.status.return_value = dict(baseline, **changes)
                monitor.tick()
        self.client.submit.assert_not_called()
        self.tuner.apply.assert_not_called()
        self.client.status.return_value = baseline
        monitor.tick()
        args, kwargs = self.client.submit.call_args
        self.assertEqual(args[0], optimizer.OWNER)
        self.assertIn('Telemetry:', args[1])
        self.assertEqual(kwargs, {'max_tokens': 96, 'role': 'resource monitor'})
        self.assertEqual(monitor.snapshot['phase'], 'analyzing')
        self.assertEqual(self.client.submit.call_count, 1)

    def test_disabled_configuration_stops_before_sampling_or_actions(self):
        monitor = self.monitor()
        for enabled, local_enabled in ((False, True), (True, False)):
            with self.subTest(enabled=enabled, local_enabled=local_enabled):
                self.configure(enabled=enabled, local_enabled=local_enabled)
                monitor.stop_event.clear()
                monitor.tick()
                self.assertTrue(monitor.stop_event.is_set())
        monitor.sample.assert_not_called(); self.client.status.assert_not_called()
        self.tuner.apply.assert_not_called(); self.client.submit.assert_not_called()

    def test_stop_or_disable_during_sampling_prevents_new_analysis(self):
        monitor = self.monitor()
        for stopped_target in ('stop', 'optimizer', 'local'):
            with self.subTest(stopped_target=stopped_target):
                self.configure()
                monitor.stop_event.clear()
                def sample():
                    if stopped_target == 'stop':
                        monitor.stop_event.set()
                    else:
                        self.configure(enabled=stopped_target != 'optimizer',
                                       local_enabled=stopped_target != 'local')
                    return copy.deepcopy(self.telemetry)
                monitor.sample.side_effect = sample
                monitor.tick()
                self.assertTrue(monitor.stop_event.is_set())
                self.assertIsNone(monitor.pending)
                self.client.submit.assert_not_called()
                self.tuner.apply.assert_not_called()

    def test_analysis_admission_refreshes_idle_model_status(self):
        monitor = self.monitor()
        baseline = copy.deepcopy(self.client.status.return_value)
        for changes in ({'active': 1}, {'queued': 1}, {'ready': False},
                        {'engine_pid': 789}, {'model': {'backend': 'llama.cpp'}}):
            with self.subTest(changes=changes):
                self.client.status.side_effect = [baseline, dict(baseline, **changes)]
                monitor.tick()
                self.client.submit.assert_not_called()
                self.assertIsNone(monitor.pending)

    def test_invalid_model_answers_never_reach_policy_mutation(self):
        monitor = self.monitor()
        monitor.engine = (123, 456)
        valid = copy.deepcopy(self.client.read.return_value)
        cases = [dict(valid, state='failed'), dict(valid, result_truncated=True),
                 dict(valid, truncated=True), dict(valid, finish_reason='length'),
                 dict(valid, usage={'backend': 'litert-npu', 'token_limit_reached': True}),
                 dict(valid, usage={'backend': 'cpu'}), dict(valid, result='```json\n{}\n```'),
                 dict(valid, result='{"action":"eco","reason":"x","command":"anything"}'),
                 dict(valid, result='{"action":"normal","action":"eco","reason":"x"}')]
        for result in cases:
            with self.subTest(result=result):
                self.client.read.return_value = result
                monitor.pending = {'id': 'analysis-1'}
                with self.assertRaises((LocalError, ValueError)):
                    monitor.tick()
                self.assertIsNone(monitor.recommendation)
        self.tuner.apply.assert_not_called(); self.client.submit.assert_not_called()

    def test_whole_fenced_recommendation_uses_original_inference(self):
        monitor = self.monitor()
        self.client.read.return_value['result'] = '```json\n{"action":"normal","reason":"Healthy."}\n```'
        monitor.tick()
        self.clock += 10
        monitor.tick()
        self.assertEqual(monitor.recommendation['action'], 'normal')
        self.assertEqual(self.client.submit.call_count, 1)

    def test_fence_normalization_does_not_relax_schema(self):
        invalid = [
            'Explanation\n```json\n{"action":"normal","reason":"ok"}\n```',
            '```json\n{"action":"normal","reason":"ok"}\n```\nExtra prose',
            '```json\n{"action":"normal","action":"eco","reason":"ok"}\n```',
            '```json\n{"action":"normal","reason":"ok","command":"id"}\n```',
            '```json\n{"action":"shell","reason":"ok"}\n```',
            '```json\n{}\n```\n```json\n{}\n```',
            'x' * 4097]
        for value in invalid:
            with self.subTest(value=value[:80]), self.assertRaises(ValueError):
                optimizer.parse_model_recommendation(value)

    def test_valid_npu_analysis_is_recorded_and_does_not_recur_early(self):
        monitor = self.monitor()
        monitor.tick()
        self.clock += 10
        monitor.tick()
        self.assertEqual(monitor.snapshot['analysis_count'], 1)
        self.assertEqual(monitor.recommendation['action'], 'normal')
        self.assertEqual(self.client.submit.call_count, 1)
        self.client.read.assert_called_once_with(optimizer.OWNER, 'analysis-1', wait=0, limit=4096)
        kinds = [row['kind'] for row in optimizer.OptimizerControl(self.base).history()['events']]
        self.assertIn('model_analysis', kinds)
        self.clock += optimizer.ANALYSIS_SECONDS
        monitor.tick()
        self.assertEqual(self.client.submit.call_count, 1)
        self.assertEqual(monitor.snapshot['analysis_reuse_count'], 1)
        self.clock = 1120
        monitor.tick()
        self.assertEqual(self.client.submit.call_count, 2)

    def test_stable_hour_halves_inferences_without_extending_ttl(self):
        monitor = self.monitor()
        for elapsed in range(0, 3600, 10):
            self.clock = 1000 + elapsed
            monitor.tick()
        self.assertEqual(self.client.submit.call_count, 30)
        self.assertEqual(monitor.snapshot['analysis_reuse_count'], 30)
        self.assertEqual(monitor.sample.call_count, 360)

    def test_change_invalidates_previous_recommendation(self):
        monitor = self.monitor()
        monitor.tick()
        self.clock += 10
        monitor.tick()
        self.assertIsNotNone(monitor.recommendation)
        self.telemetry['charging'] = False
        self.clock += 10
        monitor.tick()
        self.assertIsNone(monitor.recommendation)

    def test_real_sampling_delay_does_not_look_like_future_telemetry(self):
        monitor = self.monitor()
        original = monitor.sample.side_effect
        def delayed_sample():
            self.clock += 0.5
            return original()
        monitor.sample.side_effect = delayed_sample
        monitor.tick()
        self.assertEqual(self.client.submit.call_count, 1)

    def test_change_after_reuse_does_not_wait_for_expiry(self):
        monitor = self.monitor()
        monitor.tick()
        self.clock = 1010
        monitor.tick()
        self.clock = 1060
        monitor.tick()
        self.assertEqual(self.client.submit.call_count, 1)
        self.clock = 1070
        self.telemetry['charging'] = False
        monitor.tick()
        self.assertEqual(self.client.submit.call_count, 2)

    def test_changed_evidence_rejects_inflight_answer(self):
        monitor = self.monitor()
        monitor.tick()
        self.clock += 10
        self.telemetry['charging'] = False
        monitor.tick()
        self.assertIsNone(monitor.recommendation)
        self.tuner.apply.assert_not_called()

    def test_pending_result_cannot_cross_engine_or_evidence_change(self):
        monitor = self.monitor()
        monitor.tick()
        self.client.status.return_value['engine_pid'] = 456
        self.clock += 10
        monitor.tick()
        self.assertIsNone(monitor.recommendation)
        self.assertIn('analysis_superseded', [r['kind'] for r in optimizer.OptimizerControl(self.base).history()['events']])

    def test_unknown_telemetry_skips_inference_and_critical_policy_still_runs(self):
        monitor = self.monitor()
        self.telemetry['battery_percent'] = None
        monitor.tick()
        self.client.submit.assert_not_called()
        self.telemetry.update(battery_percent=80, battery_temp_c=46)
        self.clock += 10
        monitor.tick()
        self.tuner.apply.assert_called_once_with(123, 456, 10)

    def test_failed_request_is_not_reused(self):
        monitor = self.monitor()
        monitor.tick()
        self.client.read.return_value['state'] = 'failed'
        self.clock += 10
        with self.assertRaises(LocalError):
            monitor.tick()
        self.assertIsNone(monitor.recommendation)
        self.clock = 1060
        monitor.tick()
        self.assertEqual(self.client.submit.call_count, 2)

    def test_disable_during_decision_prevents_priority_write(self):
        monitor = self.monitor()
        for disabled_target in ('optimizer', 'local'):
            with self.subTest(disabled_target=disabled_target):
                self.configure()
                self.tuner.reset_mock()
                def decide(*args, **kwargs):
                    self.configure(enabled=disabled_target != 'optimizer', local_enabled=disabled_target != 'local')
                    return {'action': 'set_nice', 'nice_delta': 5, 'next_state': optimizer.new_state()}
                with patch('optimizer.decide', side_effect=decide):
                    monitor.tick()
                self.tuner.apply.assert_not_called()
                self.assertEqual(monitor.snapshot['action_count'], 0)

    def test_failed_tuning_does_not_commit_policy_state_or_success_event(self):
        monitor = self.monitor()
        old_state = copy.deepcopy(monitor.policy_state)
        next_state = dict(old_state, mode='eco', nice_delta=5)
        decision = {'action': 'set_nice', 'nice_delta': 5, 'next_state': next_state}
        self.tuner.apply.side_effect = LocalError('Readback did not match')
        with patch('optimizer.decide', return_value=decision), self.assertRaisesRegex(LocalError, 'Readback'):
            monitor.tick()
        self.tuner.apply.assert_called_once_with(123, 456, 5)
        self.assertEqual(monitor.policy_state, old_state)
        self.assertEqual(monitor.snapshot['action_count'], 0)
        self.assertNotIn('priority_action', [row['kind'] for row in optimizer.OptimizerControl(self.base).history()['events']])

    def test_unsuccessful_tuning_result_does_not_commit_policy_state(self):
        monitor = self.monitor()
        old_state = copy.deepcopy(monitor.policy_state)
        next_state = dict(old_state, mode='eco', nice_delta=5)
        decision = {'action': 'set_nice', 'nice_delta': 5, 'next_state': next_state}
        self.tuner.apply.return_value = {'ok': False, 'error': 'Readback did not match'}
        with patch('optimizer.decide', return_value=decision), self.assertRaises(LocalError):
            monitor.tick()
        self.tuner.apply.assert_called_once_with(123, 456, 5)
        self.assertEqual(monitor.policy_state, old_state)
        self.assertEqual(monitor.snapshot['action_count'], 0)
        self.assertNotIn('priority_action', [row['kind'] for row in optimizer.OptimizerControl(self.base).history()['events']])

    def test_failed_restoration_prevents_analysis_and_tuning(self):
        monitor = self.monitor()
        self.tuner.restore.return_value = {'ok': False, 'error': 'Original priority was not restored'}
        with self.assertRaisesRegex(LocalError, 'restoration is incomplete'):
            monitor.tick()
        self.tuner.apply.assert_not_called(); self.client.submit.assert_not_called()
        self.assertIsNone(monitor.engine)

    def test_boolean_configuration_version_does_not_enable_monitor(self):
        state = self.configure()
        (state / 'config.json').write_text(json.dumps({'version': True, 'enabled': True, 'objective': 'balanced'}))
        status = optimizer.OptimizerControl(self.base).status()
        self.assertFalse(status['enabled'])
        self.assertEqual(status['phase'], 'error')
        with self.assertRaises(LocalError):
            optimizer.OptimizerControl(self.base).resume()
        self.root_call.assert_not_called(); self.popen.assert_not_called()

    def test_bounded_history_and_pruning_preserve_other_owners_and_active_jobs(self):
        monitor = self.monitor()
        for number in range(1030):
            monitor.event('observation', number=number)
        self.assertEqual(monitor.db.execute('SELECT count(*) FROM events').fetchone()[0], 1024)
        self.assertEqual(optimizer.OptimizerControl(self.base).history(1)['events'][0]['number'], 1029)
        npu_state = self.base.parent / 'npu'; npu_state.mkdir()
        self.client.state = npu_state
        cpu_history = self.base / 'jobs.sqlite3'; cpu_history.write_bytes(b'preserve CPU history')
        path = npu_state / 'jobs.sqlite3'
        with closing(sqlite3.connect(path)) as db, db:
            db.execute('CREATE TABLE jobs(id TEXT PRIMARY KEY, owner TEXT, state TEXT, created REAL)')
            db.executemany('INSERT INTO jobs VALUES(?,?,?,?)',
                           [(str(n), optimizer.OWNER, 'completed', n) for n in range(300)])
            db.executemany('INSERT INTO jobs VALUES(?,?,?,?)',
                           [('active', optimizer.OWNER, 'running', -2),
                            ('queued', optimizer.OWNER, 'queued', -1),
                            ('user-history', 'a-conversation-owner', 'completed', -3)])
        monitor.prune_own_jobs()
        with closing(sqlite3.connect(path)) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM jobs WHERE owner=? AND state=?',
                                        (optimizer.OWNER, 'completed')).fetchone()[0], 256)
            ids = {row[0] for row in db.execute('SELECT id FROM jobs')}
        self.assertTrue({'active', 'queued', 'user-history', '299'} <= ids)
        self.assertNotIn('0', ids)
        self.assertEqual(cpu_history.read_bytes(), b'preserve CPU history')


class OptimizerProbeTests(unittest.TestCase):
    def test_missing_telemetry_remains_unknown_and_never_changes_settings(self):
        with patch('optimizer_probe.read', side_effect=OSError('fixture unavailable')), \
                patch('optimizer_probe.subprocess.run', side_effect=OSError('fixture unavailable')), \
                patch('optimizer_probe.wake_lease') as lease:
            sample = optimizer_probe.sample()
        for field in ('battery_percent', 'battery_temp_c', 'charging', 'memory_available_mb', 'cpu_totals', 'thermal_severity'):
            self.assertIsNone(sample[field])
            self.assertIn(field, sample['errors'])
        lease.assert_not_called()


if __name__ == '__main__':
    unittest.main()
