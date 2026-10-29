"""Protect benchmark units, attribution, and refusal/observation behavior.

All service interactions use fakes; these tests perform no model inference.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

import benchmark


def sample(seconds, current=-500000, voltage=4000000, status='Discharging'):
    return {'monotonic_seconds': seconds, 'battery': {
        'current_ua': current, 'voltage_uv': voltage, 'status': status}}


def ready(active=0, queued=0):
    return {'ready': True, 'running': True, 'active': active, 'queued': queued,
            'engine_pid': 123, 'model': {'backend': 'litert-npu'}}


class PowerTests(unittest.TestCase):
    def test_microamp_microvolt_conversion_and_trapezoid(self):
        values = [sample(0), sample(1, -1000000), sample(2)]
        result = benchmark.integrate_battery(values)
        self.assertTrue(result['available'])
        self.assertAlmostEqual(result['energy_j'], 6)
        self.assertAlmostEqual(result['mean_net_discharge_w'], 3)
        self.assertEqual(result['observed_duration_seconds'], 2)
        self.assertNotIn('energy_per_token', result)

    def test_positive_discharge_current_is_preserved_and_supported(self):
        result = benchmark.integrate_battery([sample(0, 500000), sample(1, 500000)])
        self.assertTrue(result['available'])
        self.assertEqual(result['current_sign'], 1)
        self.assertAlmostEqual(result['energy_j'], 2)

    def test_rejects_charging_missing_and_sign_change(self):
        cases = [sample(1, status='Charging'), sample(1, status=None),
                 sample(1, current=None), sample(1, voltage=None),
                 sample(1, current=500000), sample(1, current=0)]
        for value in cases:
            with self.subTest(value=value):
                self.assertFalse(benchmark.integrate_battery([sample(0), value])['available'])

    def test_window_does_not_integrate_baseline_or_extrapolate(self):
        values = [sample(0, -9000000), sample(1), sample(2), sample(3, -9000000)]
        result = benchmark.integrate_battery(values, start=.5, end=2.5)
        self.assertAlmostEqual(result['energy_j'], 2)
        self.assertEqual(result['sample_count'], 2)
        self.assertEqual(result['first_monotonic_seconds'], 1)
        self.assertEqual(result['last_monotonic_seconds'], 2)

    def test_missing_status_does_not_claim_discharge_power(self):
        result = benchmark.battery_values({'current_now': '-500000', 'voltage_now': '4000000', 'temp': '387'})
        self.assertIsNone(result['net_battery_discharge_w'])
        self.assertAlmostEqual(result['temperature_c'], 38.7)

    def test_rejects_large_gap_or_nonincreasing_time(self):
        for second in (0, -1, 3):
            with self.subTest(second=second):
                self.assertFalse(benchmark.integrate_battery([sample(0), sample(second)])['available'])


class AttributionTests(unittest.TestCase):
    block = ('host_preflight_seconds=0.020000 generation_seconds=1.000000 prompt_tokens=50\n'
             'request_pid=123 time_to_first_token_seconds=0.200000 completion_tokens=10\n'
             'decode_turn=0 tokens=10 tokens_per_second=12.500\n')

    def test_one_unambiguous_new_native_block(self):
        result = benchmark.native_metrics(self.block, 123, True)
        self.assertTrue(result['available'])
        self.assertEqual(result['prompt_tokens'], 50)
        self.assertEqual(result['time_to_first_token_seconds'], .2)
        self.assertEqual(result['decode_turns'][0]['tokens_per_second'], 12.5)

    def test_rejects_overlap_multiple_requests_and_wrong_pid(self):
        for text, pid, allowed in ((self.block, 123, False),
                                   (self.block * 2, 123, True), (self.block, 124, True)):
            with self.subTest(pid=pid, allowed=allowed):
                self.assertFalse(benchmark.native_metrics(text, pid, allowed)['available'])

    def test_requires_native_tokens_and_timing_to_match_observed_job(self):
        good = {'completion_tokens': 10, 'elapsed_seconds': 1.021}
        self.assertTrue(benchmark.native_metrics(self.block, 123, True, good)['available'])
        for usage in ({'completion_tokens': 11, 'elapsed_seconds': 1.021},
                      {'completion_tokens': 10, 'elapsed_seconds': 2.0}):
            self.assertFalse(benchmark.native_metrics(self.block, 123, True, usage)['available'])

    def test_reads_only_new_log_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'engine.log'
            path.write_text('PRIVATE PREVIOUS REQUEST\n' + self.block)
            cursor = benchmark.NewLog(path)
            with path.open('a') as output:
                output.write('private runtime diagnostic is not returned\n' + self.block)
            result = cursor.finish(123, True)
            self.assertTrue(result['available'])
            self.assertNotIn('PRIVATE', json.dumps(result))
            self.assertNotIn('diagnostic', json.dumps(result))


class OdpmTests(unittest.TestCase):
    before = ('t=1000\nCH6(T=1000)[S7M_VDD_TPU], 1000000\n'
              'CH8(T=900)[S9M_VDD_TPU_M], 2000000\n'
              'CH13(T=1000)[VSYS_PWR_VBATT], 5000000\n')
    after = ('t=3000\nCH6(T=3000)[S7M_VDD_TPU], 4000000\n'
             'CH8(T=2900)[S9M_VDD_TPU_M], 3000000\n'
             'CH13(T=3000)[VSYS_PWR_VBATT], 11000000\n')

    def test_retains_channel_name_timestamp_duration_and_raw_counter(self):
        parsed = benchmark.parse_odpm_energy(self.before)
        self.assertEqual(parsed['timestamp_ms'], 1000)
        self.assertEqual(parsed['rails']['CH8'], {'channel': 8, 'name': 'S9M_VDD_TPU_M',
            'duration_ms': 900, 'energy_counter_raw': 2000000})
        self.assertFalse(parsed['exact_cpm_revision_verified'])

    def test_units_are_opt_in_and_tpu_sum_is_separate_from_battery(self):
        self.assertFalse(benchmark.energy_delta(self.before, self.after)['available'])
        result = benchmark.energy_delta(self.before, self.after, abi_assumed=True)
        self.assertTrue(result['available'])
        self.assertEqual(result['elapsed_seconds'], 2)
        self.assertAlmostEqual(result['tpu']['energy_j'], 4)
        self.assertAlmostEqual(result['tpu']['mean_power_w'], 2)
        self.assertAlmostEqual(result['battery_rail']['energy_j'], 6)
        self.assertAlmostEqual(result['battery_rail']['mean_power_w'], 3)
        self.assertFalse(result['exact_cpm_revision_verified'])

    def test_rejects_missing_wrong_names_reset_and_equal_timestamps(self):
        variants = (
            self.after.replace('CH8(T=2900)[S9M_VDD_TPU_M], 3000000\n', ''),
            self.after.replace('S9M_VDD_TPU_M', 'S2S_VDD_GPU'),
            self.after.replace('4000000', '999999'),
            self.after.replace('CH8(T=2900)', 'CH8(T=100)'),
            self.after.replace('CH8(T=2900)', 'CH8(T=900)'),
            self.after.replace('CH8(T=2900)', 'CH8(T=2800)'),
            self.before,
            self.before.replace('t=1000', 't=999').replace('T=1000', 'T=999'),
        )
        for text in variants:
            with self.subTest(text=text):
                self.assertFalse(benchmark.energy_delta(self.before, text, abi_assumed=True)['available'])

    def test_rejects_partial_format_or_duplicate_channel(self):
        variants = (self.before + 'unexpected log text\n',
                    self.before + 'CH6(T=1000)[S7M_VDD_TPU], 1000000\n',
                    self.before.replace('1000000', '-1'),
                    'CH6(T=1000)[S7M_VDD_TPU], 1000000\n',
                    self.before.replace('CH8(T=900)', 'CH8(T=1001)'))
        for text in variants:
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    benchmark.parse_odpm_energy(text)

    def test_zero_energy_is_valid_idle_reading_and_battery_is_optional(self):
        after = self.before.replace('t=1000', 't=2000').replace('T=1000', 'T=2000').replace('T=900', 'T=1900')
        before = self.before.replace('CH13(T=1000)[VSYS_PWR_VBATT], 5000000\n', '')
        after = after.replace('CH13(T=2000)[VSYS_PWR_VBATT], 5000000\n', '')
        result = benchmark.energy_delta(before, after, abi_assumed=True)
        self.assertTrue(result['available'])
        self.assertEqual(result['tpu']['energy_j'], 0)
        self.assertFalse(result['battery_rail']['available'])


class BehaviorTests(unittest.TestCase):
    def test_interpolated_percentile(self):
        self.assertEqual(benchmark.percentile([10, 0], 50), 5)
        self.assertAlmostEqual(benchmark.percentile([0, 10], 95), 9.5)
        self.assertIsNone(benchmark.percentile([], 95))

    def test_root_cannot_submit_live_requests(self):
        with mock.patch('benchmark.os.getuid', return_value=0), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                benchmark.parse_args(['--run', '--output', '/tmp/would-not-be-created.json'])
        self.assertEqual(caught.exception.code, 2)

    def test_inventory_as_root_does_not_load_service_client(self):
        with mock.patch('benchmark.os.getuid', return_value=0), mock.patch('benchmark.load_client') as load, \
                mock.patch('benchmark.hardware_snapshot', return_value={}), mock.patch('benchmark.public_profile', return_value={}), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(benchmark.main([]), 0)
        load.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())['requests'], [])

    def test_independent_collector_never_loads_or_contacts_service(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'telemetry.json'
            with mock.patch('benchmark.load_client') as load, \
                    mock.patch('benchmark.hardware_snapshot', return_value={'monotonic_seconds': 42}) as snapshot, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark.main(['--collect-seconds', '0', '--pid', '123', '--output', str(output)]), 0)
            load.assert_not_called()
            snapshot.assert_called_once_with(123)
            report = json.loads(output.read_text())
            self.assertEqual(report['requests'], [])
            self.assertEqual(report['hardware_samples'], [{'monotonic_seconds': 42}])
            self.assertFalse(report['collection_methodology']['service_contacted'])
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)

    def test_collection_and_inference_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                benchmark.parse_args(['--run', '--collect-seconds', '1', '--output', '/tmp/never.json'])
        self.assertEqual(caught.exception.code, 2)

    def test_existing_report_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'report.json'
            path.write_text('preserve me')
            with mock.patch('benchmark.os.getuid', return_value=0), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(benchmark.main(['--output', str(path)]), 1)
            self.assertEqual(path.read_text(), 'preserve me')

    def test_busy_service_does_not_submit(self):
        client = mock.Mock()
        client.status.return_value = ready(active=1)
        args = types.SimpleNamespace(busy_wait_seconds=0, max_tokens=96)
        result, stop = benchmark.request(client, args, 'benchmark-owner', 123,
            'literal', 'Reply READY', 'READY', benchmark.time.monotonic() + 2)
        self.assertTrue(stop)
        self.assertIn('service_busy', result['skipped'])
        client.submit.assert_not_called()

    def test_timeout_keeps_pending_job_without_cancellation(self):
        client = mock.Mock()
        client.status.return_value = ready()
        client.submit.return_value = {'id': 'own-job'}
        client.read.return_value = {'state': 'running'}
        with tempfile.TemporaryDirectory() as directory:
            args = types.SimpleNamespace(busy_wait_seconds=0, max_tokens=96,
                state_dir=Path(directory), request_timeout_seconds=.01)
            result, stop = benchmark.request(client, args, 'unique-owner', 123,
                'literal', 'Reply READY', 'READY', benchmark.time.monotonic() + 2)
        self.assertTrue(stop)
        self.assertEqual(result['pending_job']['id'], 'own-job')
        self.assertEqual(result['pending_job']['owner'], 'unique-owner')
        client.cancel.assert_not_called()
        self.assertLessEqual(client.read.call_args.kwargs['wait'], 1)


if __name__ == '__main__':
    unittest.main()
