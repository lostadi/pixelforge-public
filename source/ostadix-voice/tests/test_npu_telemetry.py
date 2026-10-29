"""Read actual temporary sysfs fixtures; never invoke a model or Android control."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


PROBE = Path(__file__).resolve().parents[1] / 'runtime/npu/npu_telemetry.py'
spec = importlib.util.spec_from_file_location('npu_telemetry', PROBE)
telemetry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(telemetry)


class NPUTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'device'
        self.root.mkdir()
        self.sysfs = patch.object(telemetry, 'SYSFS', self.root)
        self.sysfs.start()
        self.addCleanup(self.sysfs.stop)
        self.addCleanup(self.temp.cleanup)

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else value.encode())
        return path

    def test_snapshot_reports_actual_driver_compatible_raw_frequencies_and_sources(self):
        target = Path(self.temp.name) / 'drivers/fixture-tpu'
        target.mkdir(parents=True)
        (self.root / 'driver').symlink_to('../drivers/fixture-tpu')
        self.write('of_node/compatible', b'fixture,tpu-v3\0fixture,tpu\0')
        self.write('firmware_version', 'fixture-fw-7\n')
        self.write('power/runtime_status', 'suspended\n')
        values = {'available_frequencies': '150000000 300000000\n', 'cur_freq': '150000000\n',
                  'min_freq': '150000000\n', 'max_freq': '300000000\n', 'governor': 'fixture-policy\n'}
        for field, value in values.items():
            self.write('devfreq/fixture-domain/' + field, value)
        before = {str(path.relative_to(self.root)): path.read_bytes()
                  for path in self.root.rglob('*') if path.is_file()}
        result = telemetry.snapshot()
        self.assertEqual(result['driver'], {'link': '../drivers/fixture-tpu',
                         'path': str(target), 'name': 'fixture-tpu'})
        self.assertEqual(result['compatible'], ['fixture,tpu-v3', 'fixture,tpu'])
        self.assertEqual(result['raw']['of_node/compatible'], 'fixture,tpu-v3\0fixture,tpu\0')
        self.assertEqual(result['raw']['firmware_version'], 'fixture-fw-7\n')
        self.assertEqual(result['raw']['power/runtime_status'], 'suspended\n')
        self.assertEqual(result['devfreq'], [{'name': 'fixture-domain',
                         'path': str(self.root / 'devfreq/fixture-domain'), 'raw': values}])
        self.assertEqual(result['sources']['driver'], str(self.root / 'driver'))
        self.assertEqual(result['sources']['compatible'], str(self.root / 'of_node/compatible'))
        for field in values:
            key = 'devfreq/fixture-domain/' + field
            self.assertEqual(result['sources'][key], str(self.root / key))
            self.assertNotIn(key, result['errors'])
        self.assertEqual(result['collector_uid'], os.getuid())
        self.assertTrue(result['utc'])
        self.assertEqual(before, {str(path.relative_to(self.root)): path.read_bytes()
                         for path in self.root.rglob('*') if path.is_file()})
        self.assertNotIn('tops', result)
        self.assertNotIn('core_count', result)

    def test_missing_metadata_stays_explicit_without_hiding_telemetry(self):
        self.write('firmware_version', 'observed-firmware\n')
        result = telemetry.snapshot()
        self.assertEqual(result['raw']['firmware_version'], 'observed-firmware\n')
        self.assertEqual(result['driver'], {})
        self.assertEqual(result['compatible'], [])
        self.assertEqual(result['devfreq'], [])
        for field in ('driver', 'of_node/compatible', 'devfreq'):
            self.assertIn(field, result['errors'])

    def test_denied_metadata_reads_and_domain_listing_are_reported_independently(self):
        self.write('firmware_version', 'still-readable\n')
        denied = PermissionError(13, 'Permission denied')
        original = telemetry.read_bounded
        def read(path, limit=telemetry.MAX_BYTES):
            if Path(path) == self.root / 'of_node/compatible':
                raise denied
            return original(path, limit)
        with patch.object(telemetry, 'read_bounded', side_effect=read), \
                patch.object(telemetry.os, 'readlink', side_effect=denied), \
                patch.object(telemetry.os, 'scandir', side_effect=denied):
            result = telemetry.snapshot()
        self.assertEqual(result['raw']['firmware_version'], 'still-readable\n')
        for field in ('driver', 'of_node/compatible', 'devfreq'):
            self.assertIn('Permission denied', result['errors'][field])
        self.assertEqual(result['devfreq'], [])

    def test_domain_frequency_errors_preserve_successful_raw_fields(self):
        self.write('devfreq/domain/available_frequencies', 'unparsed-kernel-value\n')
        current = self.write('devfreq/domain/cur_freq', '17\n')
        original = telemetry.read_bounded
        def read(path, limit=telemetry.MAX_BYTES):
            if Path(path) == current:
                raise PermissionError(13, 'Permission denied', str(path))
            return original(path, limit)
        with patch.object(telemetry, 'read_bounded', side_effect=read):
            result = telemetry.snapshot()
        self.assertEqual(result['devfreq'][0]['raw'], {'available_frequencies': 'unparsed-kernel-value\n'})
        self.assertIn('Permission denied', result['errors']['devfreq/domain/cur_freq'])
        self.assertIn('devfreq/domain/max_freq', result['errors'])
        self.assertEqual(result['sources']['devfreq/domain/cur_freq'], str(current))

    def test_devfreq_listing_is_bounded_and_reports_uninspected_domains(self):
        for index in range(telemetry.MAX_DEVFREQ_DOMAINS + 3):
            self.write('devfreq/domain-' + str(index) + '/cur_freq', str(index))
        original = telemetry.read_bounded
        with patch.object(telemetry, 'read_bounded', wraps=original) as read:
            result = telemetry.snapshot()
        self.assertEqual(len(result['devfreq']), telemetry.MAX_DEVFREQ_DOMAINS)
        self.assertIn('remaining domains were not inspected', result['errors']['devfreq'])
        inspected = {Path(call.args[0]).parent.name for call in read.call_args_list
                     if Path(call.args[0]).parent.parent == self.root / 'devfreq'}
        self.assertEqual(len(inspected), telemetry.MAX_DEVFREQ_DOMAINS)

    def test_oversized_metadata_is_reported_without_partial_or_invented_value(self):
        self.write('of_node/compatible', b'x' * (telemetry.MAX_BYTES + 1))
        self.write('devfreq/domain/cur_freq', b'1' * (telemetry.MAX_BYTES + 1))
        result = telemetry.snapshot()
        self.assertEqual(result['compatible'], [])
        self.assertNotIn('of_node/compatible', result['raw'])
        self.assertNotIn('cur_freq', result['devfreq'][0]['raw'])
        self.assertIn('exceeded byte bound', result['errors']['of_node/compatible'])
        self.assertIn('exceeded byte bound', result['errors']['devfreq/domain/cur_freq'])


if __name__ == '__main__':
    unittest.main()
