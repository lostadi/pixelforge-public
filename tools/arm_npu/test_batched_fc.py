"""Batched graph, protocol and sidecar regressions; no NPU or model downloads."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import schema_py_generated as schema

from author_batched_fc import author_batched_fc
from build_constant_fc import author_constant_fc
from compiled_model import ArtifactCache, batch_directory
from gguf_broker import serve
from npu_linear import NPULinear
from test_npu_protocol import FakeWeights
import test_npu_protocol


class BatchedFCTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.gguf'
        self.source.write_bytes(b'fixture source identity only')
        self.weights = FakeWeights()
        self.weights.path = self.source
        self.fingerprint = {'hardware': 'fixture'}
        self.plan = {'model_sha256': hashlib.sha256(self.source.read_bytes()).hexdigest(),
                     'unsupported_tensor_operations': [], 'dense_tensors': [
                         {'name': 'a.weight', 'rows': 2, 'columns': 2, 'bank_rows': 2}]}
        self.environment = patch.dict(os.environ, {
            'PIXELFORGE_NPU_STDERR': str(self.root / 'stderr'),
            'PIXELFORGE_NPU_FAILURE_DIR': str(self.root / 'failures')})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def backend(self, **kwargs):
        hook = NPULinear(self.weights,
            command=[sys.executable, test_npu_protocol.__file__, '--worker', 'normal'],
            weight_dtype='float32', batch_vectors=4, min_free_mib=0, cache_mib=0, **kwargs)
        self.addCleanup(hook.close)
        return hook

    def test_legacy_author_bytes_and_float_constants_are_preserved(self):
        matrix = self.weights.matrices['a.weight']
        self.assertEqual(author_batched_fc(matrix, 'float32'), author_constant_fc(matrix, 'float32'))
        graph = schema.Model.GetRootAsModel(author_batched_fc(matrix, batch_vectors=4), 0)
        subgraph = graph.Subgraphs(0)
        np.testing.assert_array_equal(subgraph.Tensors(0).ShapeAsNumpy(), [4, 2])
        np.testing.assert_array_equal(subgraph.Tensors(3).ShapeAsNumpy(), [4, 2])
        tensor = subgraph.Tensors(1)
        self.assertEqual(tensor.Type(), schema.TensorType.FLOAT32)
        actual = graph.Buffers(tensor.Buffer()).DataAsNumpy().tobytes()
        self.assertEqual(actual, matrix.tobytes())
        with self.assertRaisesRegex(ValueError, 'no additional int8'):
            author_batched_fc(matrix, 'int8', batch_vectors=4)

    def test_real_fc_batches_chunk_and_pad_tail_without_vector_loop(self):
        activation = np.asarray([[1, 2], [4209, -11], [.1, -.2], [0, 0], [-3, 7]], np.float32)
        hook = self.backend(native_runtime_cache_mib=32)
        np.testing.assert_allclose(hook('a.weight', activation),
                                   activation @ self.weights.matrices['a.weight'].T,
                                   rtol=1e-6, atol=.001)
        self.assertEqual(hook.invocations, 2)
        self.assertEqual([x['vectors'] for x in hook.records], [4, 1])
        self.assertEqual([x['batch_vectors'] for x in hook.records], [4, 4])
        self.assertEqual(hook.compile_records[0]['batch_vectors'], 4)
        self.assertEqual(hook.stats()['native_runtime_cache_mib'], 32)
        self.assertEqual(hook.runtime_cache_stats()['budget_bytes'], 32 * 1024**2)

    def test_row_banks_preserve_each_batch_vector(self):
        hook = self.backend(matrix_bank_rows={'a.weight': 1})
        x = np.arange(10, dtype=np.float32).reshape(5, 2)
        np.testing.assert_allclose(hook('a.weight', x), x @ self.weights.matrices['a.weight'].T,
                                   rtol=1e-6, atol=1e-6)
        self.assertEqual(hook.invocations, 4)

    def test_one_bank_budget_loads_each_bank_once_across_three_chunks(self):
        hook = self.backend(matrix_bank_rows={'a.weight': 1})
        hook.cache_budget = 1024**2  # Fake bytecode is 750000 bytes per bank.
        x = np.arange(18, dtype=np.float32).reshape(9, 2)
        np.testing.assert_allclose(hook('a.weight', x), x @ self.weights.matrices['a.weight'].T,
                                   rtol=1e-6, atol=1e-6)
        self.assertEqual(hook.invocations, 6)
        self.assertEqual(hook.stats()['compiled_models'], 2)
        self.assertEqual(len(hook.evictions), 1)

    def test_explicit_prefill_single_vector_uses_padded_batch_artifact(self):
        hook = self.backend()
        x = np.ones((1, 2), dtype=np.float32)
        np.testing.assert_allclose(hook('a.weight', x, prefill=True),
                                   x @ self.weights.matrices['a.weight'].T)
        self.assertEqual(hook.records[0]['batch_vectors'], 4)
        self.assertEqual(hook.records[0]['vectors'], 1)
        hook('a.weight', x)
        self.assertEqual(hook.records[1]['batch_vectors'], 1)

    def test_decode_can_explicitly_reuse_batch_graph_with_padding(self):
        hook = self.backend(decode_batch_vectors=4)
        x = np.ones((1, 2), dtype=np.float32)
        hook('a.weight', x, prefill=True)
        hook('a.weight', x)
        self.assertEqual(hook.stats()['compiled_models'], 1)
        self.assertEqual(hook.records[1]['batch_vectors'], 4)
        self.assertEqual(hook.records[1]['vectors'], 1)

    def test_missing_readonly_batch_does_not_compile(self):
        legacy = ArtifactCache(self.root / 'package', self.source,
                               fingerprint=self.fingerprint, plan=self.plan)
        readonly = ArtifactCache(legacy.directory, self.source, fingerprint=self.fingerprint,
                                 plan=self.plan, read_only=True)
        hook = self.backend(artifact_cache=readonly)
        with patch.object(self.weights, 'numpy_rows', side_effect=AssertionError('No compile')):
            with self.assertRaisesRegex(FileNotFoundError, 'No read-only artifact'):
                hook('a.weight', np.ones((4, 2), np.float32))
        self.assertTrue(hook.closed)

    def test_explicit_readonly_without_any_package_cannot_compile(self):
        hook = self.backend(read_only=True)
        with patch.object(self.weights, 'numpy_rows', side_effect=AssertionError('No compile')):
            with self.assertRaisesRegex(FileNotFoundError, 'No read-only artifact'):
                hook('a.weight', np.ones((1, 2), np.float32))
        self.assertTrue(hook.closed)

    def test_prefill_only_fusion_routes_all_decode_weights_to_ordinary_cache(self):
        hook = self.backend(fused_prefill_only=True)
        hook.weights.runtime_fused_names = {'a.weight'}
        ordinary, fused = object(), object()
        hook.batch_artifact_cache, hook.fused_batch_artifact_cache = ordinary, fused
        self.assertIs(hook._artifact_cache('a.weight', 4, prefill=True), fused)
        self.assertIs(hook._artifact_cache('a.weight', 4, prefill=False), ordinary)

    def test_fused_gate_up_is_one_real_fc_and_reuses_only_matching_sibling(self):
        from fused_ffn import FusedFFNWeights, FusedFFNDispatcher
        for name, source in (('blk.0.ffn_gate.weight', 'a.weight'), ('blk.0.ffn_up.weight', 'b.weight')):
            self.weights.matrices[name] = self.weights.matrices[source]
            self.weights.tensors[name] = self.weights.tensors[source]
        plan = dict(self.plan, dense_tensors=[dict(name=name, rows=2, columns=2, bank_rows=2)
            for name in ('blk.0.ffn_gate.weight', 'blk.0.ffn_up.weight')])
        weights = FusedFFNWeights(self.weights, plan, bank_rows=4)
        combined = np.concatenate([self.weights.matrices['a.weight'], self.weights.matrices['b.weight']])
        np.testing.assert_array_equal(weights.numpy_rows('blk.0.ffn_gate_up.weight', 1, 3), combined[1:3])
        hook = NPULinear(weights,
            command=[sys.executable, test_npu_protocol.__file__, '--worker', 'normal'],
            weight_dtype='float32', batch_vectors=4, min_free_mib=0, cache_mib=0)
        self.addCleanup(hook.close)
        fused = FusedFFNDispatcher(hook, weights)
        x = np.arange(8, dtype=np.float32).reshape(4, 2)
        for name, source in (('blk.0.ffn_gate.weight', 'a.weight'), ('blk.0.ffn_up.weight', 'b.weight')):
            np.testing.assert_allclose(fused(name, x, prefill=True),
                x @ self.weights.matrices[source].T, rtol=1e-6, atol=1e-6)
        self.assertEqual(hook.invocations, 1)
        self.assertEqual(fused.stats()['fused_sibling_reuses'], 1)
        fused('blk.0.ffn_gate.weight', x, prefill=True)
        fused('blk.0.ffn_up.weight', x + 1, prefill=True)
        self.assertEqual(hook.invocations, 3)

    def test_sidecar_reload_never_decodes_or_changes_legacy_manifest(self):
        directory = self.root / 'package'
        legacy = ArtifactCache(directory, self.source, fingerprint=self.fingerprint, plan=self.plan)
        before = legacy.manifest_path.read_bytes()
        side = batch_directory(directory, 4)
        cache = ArtifactCache(side, self.source, fingerprint=self.fingerprint,
                              plan=self.plan, batch_vectors=4)
        first = self.backend(batch_artifact_cache=cache)
        x = np.arange(8, dtype=np.float32).reshape(4, 2)
        result = first('a.weight', x)
        first.close()
        readonly = ArtifactCache(side, self.source, fingerprint=self.fingerprint,
                                 plan=self.plan, batch_vectors=4, read_only=True)
        second = self.backend(batch_artifact_cache=readonly)
        with patch.object(self.weights, 'numpy_rows', side_effect=AssertionError('Must load saved code')):
            np.testing.assert_array_equal(second('a.weight', x), result)
        self.assertEqual(second.stats()['loaded_models'], 1)
        self.assertEqual(legacy.manifest_path.read_bytes(), before)
        with self.assertRaisesRegex(ValueError, 'recipe'):
            ArtifactCache(side, self.source, fingerprint=self.fingerprint,
                          plan=self.plan, batch_vectors=8, read_only=True)

    def test_broker_passes_all_vectors_as_one_matrix_request(self):
        x = np.arange(10, dtype='<f4').reshape(5, 2)
        class Native:
            calls = []
            def _make_room(self, **kwargs): pass
            def __call__(self, name, activation):
                self.calls.append((name, activation.shape))
                return activation + 1
        native = Native()
        source = io.BytesIO(b'M 8 2 2 5\na.weight' + x.tobytes() + b'Q\n')
        sink = io.BytesIO()
        self.assertEqual(serve(source, sink, self.weights, self.plan, native), 1)
        self.assertEqual(native.calls, [('a.weight', (5, 2))])
        self.assertEqual(sink.getvalue(), b'OK 10\n' + (x + 1).tobytes())

    def test_phase_marker_routes_one_vector_prefill_and_rejects_unknown_phase(self):
        x = np.ones((1, 2), dtype='<f4')
        class Native:
            phases = []
            def _make_room(self, **kwargs): pass
            def __call__(self, name, activation, *, prefill=False):
                self.phases.append(prefill)
                return activation
        native = Native()
        source = io.BytesIO(b'M 8 2 2 1 P\na.weight' + x.tobytes() + b'Q\n')
        serve(source, io.BytesIO(), self.weights, self.plan, native)
        self.assertEqual(native.phases, [True])
        with self.assertRaisesRegex(ValueError, 'Expected M'):
            serve(io.BytesIO(b'M 8 2 2 1 X\n'), io.BytesIO(), self.weights, self.plan, native)

    def test_full_ffn_wire_is_explicit_and_counts_three_weight_operations(self):
        x = np.arange(10, dtype='<f4').reshape(5, 2)
        class Native:
            def _make_room(self, **kwargs): pass
        class Full:
            selected = {'a.weight'}
            cache = SimpleNamespace(layer=lambda name: {'hidden': 2})
            def __call__(self, name, activation, *, prefill=False):
                if not prefill:
                    raise AssertionError('P phase must be preserved')
                return activation + 2
        source = io.BytesIO(b'F 8 2 2 5 P\na.weight' + x.tobytes() + b'Q\n')
        sink = io.BytesIO()
        self.assertEqual(serve(source, sink, self.weights, self.plan, Native(), Full()), 3)
        self.assertEqual(sink.getvalue(), b'OK 10\n' + (x+2).tobytes())
        with self.assertRaisesRegex(ValueError, 'not enabled'):
            serve(io.BytesIO(b'F 8 2 2 5 P\na.weight'), io.BytesIO(), self.weights, self.plan, Native())
        full = Full()
        full.phases = 'D'
        with self.assertRaisesRegex(ValueError, 'configured phases'):
            serve(io.BytesIO(b'F 8 2 2 5 P\na.weight'), io.BytesIO(), self.weights, self.plan, Native(), full)

    def test_fused_artifact_recipe_cannot_be_opened_as_legacy_fc(self):
        cache = ArtifactCache(self.root/'fused', self.source, fingerprint=self.fingerprint,
                              plan=self.plan, fused_gate_up=True)
        code = b'\x08\0\0\0DGC0fixture'
        record = {'graph_sha256': 'b'*64, 'graph_bytes': 100, 'compiled_bytes': len(code)}
        cache.store('a.weight', 0, 2, 'float32', 2, record, code)
        with self.assertRaisesRegex(ValueError, 'recipe'):
            ArtifactCache(cache.directory, self.source, fingerprint=self.fingerprint,
                          plan=self.plan, read_only=True)
        with self.assertRaisesRegex(ValueError, 'float32'):
            cache.identity('a.weight', 0, 2, 'int8', 2)

    def test_fused_batched_cli_plan_is_readonly_and_selects_virtual_weight(self):
        import compile_gguf
        from create_tiny_gguf import create
        source = self.root/'tiny.gguf'
        create(source)
        before = source.read_bytes()
        output = self.root/'new-package'
        argv = ['compile_gguf.py', '--model', str(source), '--output', str(output),
                '--fuse-gate-up', '--tensor', 'blk.0.ffn_gate_up.weight',
                '--batch-vectors', '4', '--bank-rows', '64', '--weight-dtype', 'float32', '--plan-only']
        captured = io.StringIO()
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(captured), \
                patch('compile_gguf.NPULinear', side_effect=AssertionError('No native startup')):
            self.assertEqual(compile_gguf.main(), 0)
        self.assertIn('blk.0.ffn_gate_up.weight', captured.getvalue())
        self.assertIn('fused-ffn-v1/batch-fc-v1/b4', captured.getvalue())
        self.assertFalse(output.exists())
        self.assertEqual(source.read_bytes(), before)

    def test_bulk_pattern_selects_fused_and_direct_ffn_once(self):
        import compile_gguf
        from create_tiny_gguf import create
        source = self.root/'tiny-bulk.gguf'
        create(source)
        output = self.root/'bulk-package'
        argv = ['compile_gguf.py', '--model', str(source), '--output', str(output),
                '--fuse-gate-up', '--tensor-pattern', r'blk\.0\.ffn_(gate_up|down)\.weight',
                '--batch-vectors', '4', '--bank-rows', '2048', '--weight-dtype', 'float32', '--plan-only']
        captured = io.StringIO()
        original = compile_gguf.GGUFWeights.sha256
        calls = []
        def source_hash(weights):
            calls.append(weights.path)
            return original(weights)
        with patch.object(sys, 'argv', argv), contextlib.redirect_stdout(captured), \
                patch.object(compile_gguf.GGUFWeights, 'sha256', source_hash), \
                patch('compile_gguf.NPULinear', side_effect=AssertionError('No native startup')):
            self.assertEqual(compile_gguf.main(), 0)
        first = json.loads(captured.getvalue().splitlines()[0])
        self.assertEqual(first['selected_graphs'], 2)
        self.assertEqual(len(calls), 1)
        self.assertFalse(output.exists())

    def test_broker_opens_only_pinned_sidecar_without_a_single_vector_package(self):
        import gguf_broker
        directory = self.root/'package'
        config = self.root/'config.json'
        report = self.root/'report.json'
        config.write_text(json.dumps(dict(model=str(self.source), model_sha256=self.plan['model_sha256'],
            artifacts=str(directory), bank_rows=2, weight_dtype='float32', batch_vectors=4,
            decode_batch_vectors=4, read_only=True, report=str(report),
            required_artifact_manifests=['batch-fc-v1/b4/MODEL.json'])))
        weights = SimpleNamespace(sha256=lambda: self.plan['model_sha256'], source_identity=(0,)*5)
        native = SimpleNamespace(runtime_cache_stats=lambda: None, stats=lambda: {}, close=lambda: None,
                                 process=SimpleNamespace(returncode=0))
        cache = object()
        output = io.TextIOWrapper(io.BytesIO())
        plan = dict(self.plan, generation_adapter_supported=True, architecture='fixture')
        with patch.object(sys, 'argv', ['gguf_broker.py', '--config', str(config)]), \
                patch.object(sys, 'stdout', output), patch.object(gguf_broker, 'GGUFWeights', return_value=weights), \
                patch.object(gguf_broker, 'build_plan', return_value=plan), \
                patch.object(gguf_broker, 'ArtifactCache', return_value=cache) as caches, \
                patch.object(gguf_broker, 'NPULinear', return_value=native) as backend, \
                patch.object(gguf_broker, 'serve', return_value=1):
            self.assertEqual(gguf_broker.main(), 0)
        self.assertEqual(caches.call_count, 1)
        self.assertEqual(caches.call_args.args[0], directory/'batch-fc-v1/b4')
        self.assertIsNone(backend.call_args.kwargs['artifact_cache'])
        self.assertIs(backend.call_args.kwargs['batch_artifact_cache'], cache)
        self.assertTrue(backend.call_args.kwargs['read_only'])

    def test_broker_preserves_performance_default_and_only_configures_dma_resident_limit(self):
        import gguf_broker
        weights = SimpleNamespace(sha256=lambda: self.plan['model_sha256'], source_identity=(0,)*5)
        plan = dict(self.plan, generation_adapter_supported=True, architecture='fixture')
        variants = [({}, None, 'default'),
                    ({'weight_buffer_cache_mib': 0, 'native_max_resident_models': 9}, None, 'default'),
                    ({'weight_buffer_cache_mib': 512, 'native_max_resident_models': 9,
                      'npu_performance_mode': 'sustained'}, 9, 'sustained')]
        for index, (options, expected_limit, expected_mode) in enumerate(variants):
            with self.subTest(options=options):
                config = self.root/f'runtime-config-{index}.json'
                config.write_text(json.dumps(dict(model=str(self.source), model_sha256=self.plan['model_sha256'],
                    bank_rows=2, weight_dtype='float32', read_only=True,
                    report=str(self.root/f'runtime-report-{index}.json'), **options)))
                native = SimpleNamespace(runtime_cache_stats=lambda: None, stats=lambda: {}, close=lambda: None,
                                         process=SimpleNamespace(returncode=0))
                output = io.TextIOWrapper(io.BytesIO())
                with patch.object(sys, 'argv', ['gguf_broker.py', '--config', str(config)]), \
                        patch.object(sys, 'stdout', output), \
                        patch.object(gguf_broker, 'GGUFWeights', return_value=weights), \
                        patch.object(gguf_broker, 'build_plan', return_value=plan), \
                        patch.object(gguf_broker, 'NPULinear', return_value=native) as backend, \
                        patch.object(gguf_broker, 'serve', return_value=1):
                    self.assertEqual(gguf_broker.main(), 0)
                self.assertEqual(backend.call_args.kwargs['npu_performance_mode'], expected_mode)
                if expected_limit is None:
                    self.assertNotIn('native_max_resident_models', backend.call_args.kwargs)
                else:
                    self.assertEqual(backend.call_args.kwargs['native_max_resident_models'], expected_limit)

    def test_column_broker_uses_only_its_pinned_phase_family_and_original_tensor_plan(self):
        import gguf_broker
        import grouped_columns
        weights = SimpleNamespace(sha256=lambda: self.plan['model_sha256'], source_identity=(0,)*5)
        plan = dict(self.plan, generation_adapter_supported=True, architecture='fixture')
        directory = self.root/'column-package'
        for legacy in (False, True):
            with self.subTest(legacy_inventory=legacy):
                config = self.root/f'column-config-{legacy}.json'
                config.write_text(json.dumps(dict(model=str(self.source), model_sha256=self.plan['model_sha256'],
                    artifacts=str(directory), bank_rows=2, weight_dtype='float32', batch_vectors=64,
                    decode_batch_vectors=1, dynamic_weight_mode='original_columns', read_only=True,
                    native_runtime_cache_mib=128, report=str(self.root/f'column-report-{legacy}.json'),
                    required_artifact_manifests=(["grouped-fc-v1/b64/MODEL.json"] if legacy else
                        ["grouped-columns-v1/b64/MODEL.json", "grouped-columns-v1/b1/MODEL.json"]))))
                native = SimpleNamespace(runtime_cache_stats=lambda: None, stats=lambda: {}, close=lambda: None,
                                         process=SimpleNamespace(returncode=0))
                facade = SimpleNamespace(runtime_cache_stats=lambda: None, stats=lambda: {}, close=lambda: None,
                                         process=SimpleNamespace(returncode=0))
                prompt_cache, decode_cache = object(), object()
                output = io.TextIOWrapper(io.BytesIO())
                with patch.object(sys, 'argv', ['gguf_broker.py', '--config', str(config)]), \
                        patch.object(sys, 'stdout', output), \
                        patch.object(gguf_broker, 'GGUFWeights', return_value=weights), \
                        patch.object(gguf_broker, 'build_plan', return_value=plan) as planner, \
                        patch.object(gguf_broker, 'ArtifactCache', side_effect=AssertionError('No static family')), \
                        patch.object(grouped_columns, 'ColumnProgramCache', side_effect=[prompt_cache, decode_cache]) as caches, \
                        patch.object(grouped_columns, 'ColumnPhaseExecutor', return_value=facade) as phase, \
                        patch.object(gguf_broker, 'NPULinear', return_value=native) as backend, \
                        patch.object(gguf_broker, 'serve', return_value=1) as serving:
                    self.assertEqual(gguf_broker.main(), 1 if legacy else 0)
                self.assertEqual(planner.call_count, 1)
                self.assertIs(planner.call_args.args[0], weights)
                if legacy:
                    caches.assert_not_called()
                    backend.assert_not_called()
                    phase.assert_not_called()
                    serving.assert_not_called()
                else:
                    self.assertEqual([args.args[0] for args in caches.call_args_list],
                                     [directory/'grouped-columns-v1/b64', directory/'grouped-columns-v1/b1'])
                    for args in caches.call_args_list:
                        self.assertIs(args.kwargs['plan'], plan)
                        self.assertTrue(args.kwargs['read_only'])
                    phase.assert_called_once_with(native, weights, prompt_cache, decode_cache)
                    self.assertIs(serving.call_args.args[3], plan)
                    self.assertIs(serving.call_args.args[4], facade)


if __name__ == '__main__':
    unittest.main()
