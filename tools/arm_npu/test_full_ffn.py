"""Inspect and execute synthetic FFN graphs in NumPy; no hardware assertions."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import struct
import hashlib
import io
import contextlib
import os
import sys

import numpy as np
import schema_py_generated as schema
from author_full_ffn import author_full_ffn, numpy_columns
from gguf_hf_loader import GGUFWeights, QUANT_SIZES
from test_gguf_loader import write_gguf
from full_ffn import FullFFNProgramCache, FullFFNExecutor, build_full_ffn_plan, compile_full_ffn_chunk
from npu_linear import NPULinear
import test_npu_protocol


def evaluate(graph_bytes, x):
    model = schema.Model.GetRootAsModel(graph_bytes, 0)
    graph = model.Subgraphs(0)
    values = {0: x}
    for index in range(graph.TensorsLength()):
        tensor = graph.Tensors(index)
        if tensor.Buffer():
            values[index] = model.Buffers(tensor.Buffer()).DataAsNumpy().view('<f4').reshape(tuple(tensor.ShapeAsNumpy()))
    for index in range(graph.OperatorsLength()):
        op = graph.Operators(index)
        kind = model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        args = [values[int(i)] for i in op.InputsAsNumpy()]
        if kind == schema.BuiltinOperator.FULLY_CONNECTED:
            value = args[0] @ args[1].T + args[2]
        elif kind == schema.BuiltinOperator.LOGISTIC:
            value = 1 / (1 + np.exp(-args[0]))
        elif kind == schema.BuiltinOperator.MUL:
            value = args[0] * args[1]
        else:
            raise AssertionError('Unexpected graph operation')
        values[int(op.Outputs(0))] = value
    return values[int(graph.Outputs(0))]


class FullFFNTests(unittest.TestCase):
    def test_graph_executes_silu_product_and_three_original_float_weights(self):
        rng = np.random.default_rng(834)
        gate, up, down = (rng.normal(0, .1, size=shape).astype(np.float32)
                          for shape in ((7, 4), (7, 4), (4, 7)))
        x = rng.normal(0, 3, (3, 4)).astype(np.float32)
        data = author_full_ffn(gate, up, down, batch_vectors=3)
        a, b = x @ gate.T, x @ up.T
        expected = (a / (1 + np.exp(-a)) * b) @ down.T
        np.testing.assert_allclose(evaluate(data, x), expected, rtol=1e-6, atol=1e-6)
        model = schema.Model.GetRootAsModel(data, 0)
        graph = model.Subgraphs(0)
        kinds = [model.OperatorCodes(graph.Operators(i).OpcodeIndex()).BuiltinCode()
                 for i in range(graph.OperatorsLength())]
        self.assertEqual(kinds, [9, 9, 14, 18, 18, 9])
        for index, weights in ((1, gate), (4, up), (10, down)):
            tensor = graph.Tensors(index)
            self.assertEqual(tensor.Type(), schema.TensorType.FLOAT32)
            self.assertEqual(model.Buffers(tensor.Buffer()).DataAsNumpy().tobytes(), weights.tobytes())
        np.testing.assert_array_equal(graph.Tensors(0).ShapeAsNumpy(), [3, 4])
        np.testing.assert_array_equal(graph.Tensors(12).ShapeAsNumpy(), [3, 4])

    def test_hidden_chunk_outputs_sum_to_the_same_ffn_function(self):
        rng = np.random.default_rng(999)
        gate, up, down = (rng.normal(0, .1, size=shape).astype(np.float32)
                          for shape in ((9, 4), (9, 4), (4, 9)))
        x = rng.normal(0, 4, (2, 4)).astype(np.float32)
        chunks = [evaluate(author_full_ffn(gate[a:b], up[a:b], down[:, a:b], batch_vectors=2), x)
                  for a, b in ((0, 3), (3, 5), (5, 9))]
        np.testing.assert_allclose(sum(chunks),
            evaluate(author_full_ffn(gate, up, down, batch_vectors=2), x), rtol=1e-6, atol=1e-6)

    def test_shape_nonfinite_and_batch_contracts_are_strict(self):
        gate = np.ones((3, 4), np.float32)
        with self.assertRaisesRegex(ValueError, 'down'):
            author_full_ffn(gate, gate, np.ones((3, 4), np.float32))
        with self.assertRaisesRegex(ValueError, 'finite'):
            author_full_ffn(gate * np.nan, gate, gate.T)
        with self.assertRaisesRegex(ValueError, 'batch_vectors'):
            author_full_ffn(gate, gate, gate.T, batch_vectors=True)

    def test_column_slice_matches_exact_dequantization_across_block_edges(self):
        rng = np.random.default_rng(111)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.gguf'
            for kind in (0, 13, 14):
                with self.subTest(kind=kind):
                    block, size = QUANT_SIZES[kind]
                    rows, columns = 3, block * 3
                    if kind == 0:
                        raw = rng.normal(size=(rows, columns)).astype('<f4').tobytes()
                        start, stop = 1, 2
                    else:
                        blocks = rng.integers(0, 256, (rows*3, size), dtype=np.uint8)
                        for offset in ((0, 2) if kind == 13 else (208,)):
                            blocks[:, offset:offset+2] = np.frombuffer(struct.pack('<e', .03125), np.uint8)
                        raw, start, stop = blocks.tobytes(), 31, 289
                    write_gguf(path, [('weight', (rows, columns), kind, raw)])
                    before = path.read_bytes()
                    with GGUFWeights(path) as weights:
                        expected = weights.numpy('weight')[:, start:stop]
                        with patch.object(weights, 'numpy', side_effect=AssertionError('No full decode')), \
                                patch.object(weights, 'numpy_rows', side_effect=AssertionError('No full row decode')):
                            actual = numpy_columns(weights, 'weight', start, stop, max_block_batch=1)
                        np.testing.assert_array_equal(actual, expected)
                        self.assertEqual(numpy_columns(weights, 'weight', 0, 0).shape, (rows, 0))
                    self.assertEqual(path.read_bytes(), before)

    def test_compiled_chunk_cache_runs_real_batch_programs_without_decoding(self):
        rng = np.random.default_rng(105)
        gate, up, down = (rng.normal(0, .1, shape).astype('<f4')
                          for shape in ((9, 4), (9, 4), (4, 9)))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'fixture.gguf'
            write_gguf(path, [(name, values.shape, 0, values.tobytes()) for name, values in (
                ('blk.0.ffn_gate.weight', gate), ('blk.0.ffn_up.weight', up), ('blk.0.ffn_down.weight', down))])
            before = path.read_bytes()
            with GGUFWeights(path) as weights, patch.dict(os.environ, {'PIXELFORGE_NPU_STDERR': str(root/'stderr')}):
                digest = weights.sha256()
                plan = build_full_ffn_plan(weights, digest, hidden_chunk=4)
                fingerprint = {'device': 'fixture'}
                cache = FullFFNProgramCache(root/'full-ffn-v1/b4', path, fingerprint=fingerprint,
                    plan=plan, batch_vectors=4)
                native = NPULinear(weights, command=[sys.executable, test_npu_protocol.__file__, '--worker', 'normal'],
                    weight_dtype='float32', batch_vectors=4, cache_mib=1, min_free_mib=0,
                    native_runtime_cache_mib=0)
                try:
                    for start, stop in cache.ranges('blk.0.ffn_gate.weight'):
                        compile_full_ffn_chunk(native, weights, cache, 'blk.0.ffn_gate.weight', start, stop)
                    self.assertEqual(native.invocations, 0)
                    readonly = FullFFNProgramCache(cache.directory, path, fingerprint=fingerprint,
                        plan=plan, batch_vectors=4, read_only=True)
                    executor = FullFFNExecutor(native, weights, readonly, ['blk.0.ffn_gate.weight'])
                    x = rng.normal(0, 4, (9, 4)).astype(np.float32)
                    expected = evaluate(author_full_ffn(gate, up, down, batch_vectors=9), x)
                    with patch.object(weights, 'numpy_rows', side_effect=AssertionError('No source rows decoded')), \
                            patch('full_ffn.numpy_columns', side_effect=AssertionError('No source columns decoded')):
                        result = executor('blk.0.ffn_gate.weight', x, prefill=True)
                        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-6)
                        # The same B4 programs execute a one-vector D request.
                        np.testing.assert_allclose(executor('blk.0.ffn_gate.weight', x[:1]),
                                                   expected[:1], rtol=1e-5, atol=1e-6)
                    self.assertEqual(native.invocations, 12)  # 3 hidden chunks x (3 P chunks + 1 padded D).
                    self.assertEqual(native.stats()['loaded_models'], 6)
                    self.assertEqual(executor.operations, 2)
                    self.assertEqual(cache.manifest_path.name, 'MODEL.json')
                    self.assertEqual(cache._manifest()['format'], 'pixelforge.darwinn.full-ffn.v1')
                finally:
                    native.close()
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
