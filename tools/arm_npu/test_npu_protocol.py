"""Exercise binary framing, scaling, LRU release and timeouts without the NPU.

The fake helper interprets the generated FC graph with NumPy solely to verify
the client protocol. These tests make no claim of hardware inference.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
import struct

import numpy as np


def worker(mode):
    import schema_py_generated as schema

    source = sys.stdin.buffer
    sink = sys.stdout.buffer

    def exact(size):
        result = bytearray()
        while len(result) < size:
            part = source.read(size - len(result))
            if not part:
                raise EOFError
            result.extend(part)
        return bytes(result)

    sink.write(b"READY PIXELFORGE_NPU_PIPE 1\n")
    sink.flush()
    if mode == "stall":
        time.sleep(60)
        return
    models = {}
    codes = {}
    def decoded_graph(data):
        graph = schema.Model.GetRootAsModel(data, 0)
        tensor = graph.Subgraphs(0).Tensors(1)
        shape = tuple(tensor.ShapeAsNumpy())
        raw = graph.Buffers(tensor.Buffer()).DataAsNumpy()
        if tensor.Type() == schema.TensorType.INT8:
            weights = raw.view(np.int8).reshape(shape).astype(np.float32)
            weights *= tensor.Quantization().ScaleAsNumpy()[:, None]
        else:
            weights = raw.view("<f4").reshape(shape)
        return weights.copy()
    while line := source.readline():
        fields = line.decode().split()
        if fields[0] == "Q":
            break
        if fields[0] == "I":
            sink.write(b"OK CAPABILITIES C R F I D L Q\n" if mode != "old-helper" else b"ERR UNKNOWN_MODEL\n")
        elif fields[0] == "C":
            data = exact(int(fields[4]))
            models[fields[1]] = decoded_graph(data)
            codes[fields[1]] = (b"\x08\0\0\0DGC0" + struct.pack("<I", len(data)) + data).ljust(750000, b"\0")
            sink.write(f"OK COMPILED {fields[1]} 750000\n".encode())
        elif fields[0] == "D":
            code = codes[fields[1]]
            sink.write(f"OK BYTECODE {len(code)}\n".encode())
            sink.write(code)
        elif fields[0] == "L":
            code = exact(int(fields[4]))
            length = struct.unpack("<I", code[8:12])[0]
            models[fields[1]] = decoded_graph(code[12:12+length])
            codes[fields[1]] = code
            sink.write(f"OK LOADED {fields[1]} {len(code)}\n".encode())
        elif fields[0] == "R":
            weights = models[fields[1]]
            activation = np.frombuffer(exact(weights.shape[1] * 4), dtype="<f4")[None, :]
            output = activation @ weights.T
            if mode == "bad-output":
                output[:] = np.nan
            sink.write(f"OK OUTPUT 1 {output.nbytes}\n".encode())
            sink.write(output.astype("<f4").tobytes())
        elif fields[0] == "F":
            del models[fields[1]]
            del codes[fields[1]]
            sink.write(b"OK FREED\n")
        else:
            raise ValueError(fields)
        sink.flush()


class FakeWeights:
    path = Path("protocol-fixture.gguf")

    def __init__(self):
        self.matrices = {
            "a.weight": np.asarray([[0.12, -0.31], [0.48, 0.73]], dtype=np.float32),
            "b.weight": np.asarray([[-0.81, 0.17], [0.16, -0.32]], dtype=np.float32),
        }
        self.matrices["output.weight"] = np.tile(self.matrices["a.weight"], (2, 1))
        self.tensors = {name: type("Descriptor", (), {"shape": value.shape})()
                        for name, value in self.matrices.items()}

    def numpy_rows(self, name, start, stop):
        return self.matrices[name][start:stop].copy()


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        from npu_linear import NPULinear
        self.constructor = NPULinear
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)

    def backend(self, mode="normal", **kwargs):
        from unittest.mock import patch
        stderr = str(Path(self.temporary.name) / (mode + ".stderr"))
        failure = str(Path(self.temporary.name) / "failures")
        environment = patch.dict("os.environ", {"PIXELFORGE_NPU_STDERR": stderr,
                                                "PIXELFORGE_NPU_FAILURE_DIR": failure})
        environment.start()
        self.addCleanup(environment.stop)
        kwargs.setdefault("cache_mib", 1)
        kwargs.setdefault("min_free_mib", 0)
        hook = self.constructor(FakeWeights(), command=[sys.executable, __file__, "--worker", mode], **kwargs)
        self.addCleanup(hook.close)
        return hook

    def test_scaling_and_lru_recompilation(self):
        hook = self.backend()
        activation = np.asarray([[4209.0, -11.0]], dtype=np.float32)
        for name in ("a.weight", "b.weight", "a.weight"):
            matrix = hook.weights.matrices[name]
            scales = np.max(np.abs(matrix), axis=1) / 127
            quantized = np.rint(matrix / scales[:, None]) * scales[:, None]
            np.testing.assert_allclose(hook(name, activation), activation @ quantized.T, rtol=1e-6, atol=0.001)
        stats = hook.stats()
        self.assertEqual(stats["npu_invocations"], 3)
        self.assertEqual(stats["compiled_models"], 3)
        self.assertEqual(stats["resident_models"], 1)
        self.assertEqual(len(stats["evictions"]), 2)
        self.assertEqual(stats["invocations"][0]["activation_scale"], 4209.0)
        self.assertEqual(len(stats["matrices"][0]["graph_sha256"]), 64)
        self.assertFalse(stats["cpu_fallback"])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("ai_edge_litert", sys.modules)

    def test_nonfinite_native_output_stops_owned_helper(self):
        hook = self.backend("bad-output")
        with self.assertRaisesRegex(RuntimeError, "Nonfinite NPU"):
            hook("a.weight", np.ones((1, 2), dtype=np.float32))
        self.assertTrue(hook.closed)
        self.assertIsNotNone(hook.process.poll())

    def test_nonblocking_write_has_deadline(self):
        hook = self.backend("stall", timeout=10)
        hook.timeout = 0.15
        began = time.monotonic()
        with self.assertRaisesRegex(TimeoutError, "input timed out"):
            hook._write(bytes(1024 * 1024))
        self.assertLess(time.monotonic() - began, 1.5)
        hook.close()
        self.assertIsNotNone(hook.process.poll())

    def test_identical_row_banks_have_distinct_native_ids(self):
        hook = self.backend(cache_mib=0, lm_bank_rows=2)
        activation = np.ones((1, 2), dtype=np.float32)
        result = hook("output.weight", activation)
        np.testing.assert_array_equal(result[:, :2], result[:, 2:])
        records = list(hook.models.values())
        self.assertEqual(records[0]["graph_sha256"], records[1]["graph_sha256"])
        self.assertNotEqual(records[0]["native_id"], records[1]["native_id"])

    def test_matrix_bank_rows_applies_without_artifact_cache(self):
        hook = self.backend(cache_mib=0, matrix_bank_rows={"a.weight": 1})
        result = hook("a.weight", np.ones((1, 2), dtype=np.float32))
        self.assertEqual(result.shape, (1, 2))
        self.assertEqual(hook.invocations, 2)
        self.assertEqual([record["rows"] for record in hook.compile_records], [1, 1])

    def test_compile_guard_checks_full_working_set(self):
        from unittest.mock import patch
        from npu_linear import MIB
        hook = self.backend(min_free_mib=128)
        with patch("npu_linear._memory_available", return_value=128 * MIB + MIB // 2), \
                patch.object(hook, "_write", wraps=hook._write) as writes:
            with self.assertRaisesRegex(MemoryError, "compile working set"):
                hook("a.weight", np.ones((1, 2), dtype=np.float32))
        self.assertEqual(writes.call_count, 0)
        self.assertEqual(hook.invocations, 0)

    def test_cached_key_guard_stops_before_r(self):
        from unittest.mock import patch
        from npu_linear import MIB
        hook = self.backend()
        activation = np.ones((1, 2), dtype=np.float32)
        hook("a.weight", activation)
        hook.min_free = 128 * MIB
        with patch("npu_linear._memory_available", return_value=128 * MIB + MIB // 2), \
                patch.object(hook, "_write", wraps=hook._write) as writes:
            with self.assertRaisesRegex(MemoryError, "invocation working set"):
                hook("a.weight", activation)
        self.assertEqual(writes.call_count, 0)
        self.assertEqual(hook.invocations, 1)
        self.assertTrue(hook.closed)

    def test_invocation_pressure_evicts_other_key(self):
        from unittest.mock import patch
        from npu_linear import MIB
        hook = self.backend(cache_mib=0)
        activation = np.ones((1, 2), dtype=np.float32)
        hook("a.weight", activation)
        hook("b.weight", activation)
        protected = next(key for key in hook.models if key[0] == "a.weight")
        hook.min_free = 128 * MIB
        with patch("npu_linear._memory_available", side_effect=[128 * MIB + MIB // 2, 256 * MIB]):
            hook("a.weight", activation)
        self.assertEqual(list(hook.models), [protected])
        self.assertEqual(hook.evictions[-1]["name"], "b.weight")
        self.assertEqual(hook.invocations, 3)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker(sys.argv[2])
    else:
        unittest.main()
