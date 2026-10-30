"""Documented LiteRT graph performance policies with mock SDK calls only."""
import os
import unittest

import test_npu_pipe_cache as _cache


class NativePerformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _cache.NativeRuntimeCacheTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        _cache.NativeRuntimeCacheTests.tearDownClass.__func__(cls)

    def session(self, *, policy=None, version='0.20.0', support='both', mode='protocol'):
        env = dict(os.environ, PIXELFORGE_TEST_VENDOR_VERSION=version,
                   PIXELFORGE_TEST_PERFORMANCE_API=support)
        for key in ('PIXELFORGE_NPU_FD_SOCKET', 'PIXELFORGE_NPU_DEFER_WEIGHT_ALLOCATION',
                    'PIXELFORGE_NPU_PERFORMANCE_MODE'):
            env.pop(key, None)
        if policy is not None:
            env['PIXELFORGE_NPU_PERFORMANCE_MODE'] = policy
        wire = _cache.ProtocolSession(self, self.binary, env=env, mode=mode)
        self.addCleanup(wire.close)
        return wire

    def test_unset_policy_has_no_annotation_and_reports_default(self):
        with self.session() as wire:
            wire.send(b'I CAPABILITIES\nI PERFORMANCE_MODE\n' +
                      _cache.NativeRuntimeCacheTests.load() + b'Q\n')
            self.assertIn(b' PERFORMANCE_MODE_V1\n', wire.line())
            self.assertEqual(wire.line(), b'OK PERFORMANCE_MODE default 0\n')
            self.assertEqual(wire.line(), b'OK LOADED a 12\n')
            self.assertEqual(wire.line(), b'OK QUIT\n')
            self.assertIn(b'performance_annotations=0', wire.wait())

    def test_requested_enum_is_applied_before_edges_for_every_new_graph(self):
        for name, value in [('balanced', 2), ('high', 3), ('sustained', 4)]:
            with self.subTest(name=name), self.session(policy=name) as wire:
                wire.send(b'I PERFORMANCE_MODE\n' + _cache.NativeRuntimeCacheTests.load() +
                          b'R a\n' + b'x' * 16 + b'B 0\nR a\n' + b'y' * 16 + b'Q\n')
                self.assertEqual(wire.line(), f'OK PERFORMANCE_MODE {name} {value}\n'.encode())
                self.assertEqual(wire.line(), b'OK LOADED a 12\n')
                self.assertEqual(wire.output(), b'x' * 16)
                self.assertEqual(wire.line(), b'OK CACHE 0\n')
                self.assertEqual(wire.output(), b'y' * 16)
                self.assertEqual(wire.line(), b'OK QUIT\n')
                self.assertIn(b'performance_annotations=3', wire.wait())

    def test_supported_version_boundary_and_semver_suffix(self):
        for version in ('0.18.0', '0.20.0', '0.20.0-rc.1', '1.0.0'):
            with self.subTest(version=version), self.session(policy='sustained', version=version) as wire:
                wire.send(b'I PERFORMANCE_MODE\nQ\n')
                self.assertEqual(wire.line(), b'OK PERFORMANCE_MODE sustained 4\n')
                self.assertEqual(wire.line(), b'OK QUIT\n')
                wire.wait()

    def test_old_or_malformed_versions_reject_requested_policy(self):
        for version in ('0.17.99', '', '0.18', '0..18.0', '-1.20.0', '0.20.0.1', '0. 20.0'):
            with self.subTest(version=version), self.session(policy='sustained', version=version) as wire:
                self.assertEqual(wire.line(), b'ERR PERFORMANCE_MODE\n')
                self.assertIn(b'performance_annotations=0', wire.wait(2))

    def test_missing_optional_symbol_preserves_default_but_rejects_request(self):
        for support in ('missing-annotation', 'missing-version'):
            with self.subTest(support=support):
                with self.session(support=support) as wire:
                    wire.send(b'I CAPABILITIES\nI PERFORMANCE_MODE\nQ\n')
                    self.assertNotIn(b'PERFORMANCE_MODE_V1', wire.line())
                    self.assertEqual(wire.line(), b'OK PERFORMANCE_MODE default 0\n')
                    self.assertEqual(wire.line(), b'OK QUIT\n')
                    wire.wait()
                with self.session(policy='sustained', support=support) as wire:
                    self.assertEqual(wire.line(), b'ERR PERFORMANCE_MODE\n')
                    wire.wait(2)

    def test_invalid_policies_fail_before_any_graph(self):
        for name in ('', 'burst', '4', 'Sustained', 'high ', 'powersaver'):
            with self.subTest(name=name), self.session(policy=name) as wire:
                self.assertEqual(wire.line(), b'ERR PERFORMANCE_MODE\n')
                self.assertIn(b'load=0 graph=0', wire.wait(2))

    def test_annotation_failure_aborts_first_graph_without_replay(self):
        with self.session(policy='sustained', mode='protocol-annotation') as wire:
            wire.send(_cache.NativeRuntimeCacheTests.load())
            self.assertEqual(wire.line(), b'ERR PERFORMANCE_MODE\n')
            self.assertIn(b'FAIL annotation sq=1 graph=1 inv=0 ahb=0 reg=0', wire.wait(2))


if __name__ == '__main__':
    unittest.main()
