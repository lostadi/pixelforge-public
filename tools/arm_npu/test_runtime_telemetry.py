"""Bounded diagnostics preserve exact counters and failed-call semantics."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from runtime_telemetry import TraceTail
import test_npu_protocol as protocol


class TraceTailTests(unittest.TestCase):
    def test_long_history_retains_only_latest_details_with_exact_total(self):
        trace = TraceTail(1024)
        for number in range(100000):
            trace.append({"operation": number})
        self.assertEqual(len(trace), 1024)
        self.assertEqual(trace[0], {"operation": 98976})
        self.assertEqual(trace[-1], {"operation": 99999})
        self.assertEqual(trace.evidence(),
                         {"policy": "bounded-most-recent", "limit": 1024,
                          "total": 100000, "retained": 1024, "dropped": 98976})

    def test_slice_and_snapshot_are_detached_recent_lists(self):
        trace = TraceTail(3)
        for number in range(5):
            trace.append({"operation": number})
        self.assertEqual(trace[1:], [{"operation": 3}, {"operation": 4}])
        self.assertEqual(trace[::-1],
                         [{"operation": 4}, {"operation": 3}, {"operation": 2}])
        snapshot = trace.snapshot()
        snapshot.clear()
        self.assertEqual(len(trace), 3)
        self.assertEqual(trace.total, 5)

    def test_empty_trace_explicitly_reports_no_dropped_events(self):
        trace = TraceTail(3)
        self.assertEqual(trace.snapshot(), [])
        self.assertEqual(trace.evidence()["total"], 0)
        self.assertEqual(trace.evidence()["dropped"], 0)
        with self.assertRaises(IndexError):
            _ = trace[0]

    def test_invalid_limits_are_rejected(self):
        for value in (0, -1, True, 1.5, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                TraceTail(value)


class RuntimeTraceTests(unittest.TestCase):
    def backend(self, temporary, mode="normal"):
        from npu_linear import NPULinear
        environment = {
            "PIXELFORGE_NPU_STDERR": str(Path(temporary) / "native.stderr"),
            "PIXELFORGE_NPU_FAILURE_DIR": str(Path(temporary) / "failures"),
        }
        context = patch.dict("os.environ", environment)
        context.start()
        self.addCleanup(context.stop)
        hook = NPULinear(protocol.FakeWeights(),
                         command=[sys.executable, protocol.__file__, "--worker", mode],
                         weight_dtype="float32", cache_mib=1, min_free_mib=0)
        self.addCleanup(hook.close)
        hook.records = TraceTail(3)
        hook.evictions = TraceTail(2)
        return hook

    def test_report_overflow_preserves_outputs_and_exact_native_totals(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            hook = self.backend(temporary)
            activation = np.asarray([[0.5, -0.25]], dtype=np.float32)
            for name in ("a.weight", "b.weight", "a.weight", "b.weight",
                         "a.weight", "b.weight", "a.weight"):
                np.testing.assert_allclose(
                    hook(name, activation), activation @ hook.weights.matrices[name].T,
                    rtol=1e-6, atol=1e-7)
            stats = hook.stats()
            self.assertEqual(stats["npu_invocations"], 7)
            self.assertEqual(stats["compiled_models"], 7)
            self.assertEqual(stats["compiled_bytes_total"], 7 * 750000)
            self.assertEqual(len(stats["matrices"]), 7)
            self.assertEqual(len(stats["invocations"]), 3)
            self.assertEqual(stats["invocation_trace"],
                             {"policy": "bounded-most-recent", "limit": 3,
                              "total": 7, "retained": 3, "dropped": 4})
            self.assertEqual(len(stats["evictions"]), 2)
            self.assertEqual(stats["npu_cache_evictions"], 6)
            self.assertEqual(stats["eviction_trace"]["total"], 6)
            self.assertEqual(stats["eviction_trace"]["dropped"], 4)
            self.assertGreater(stats["invoke_seconds"], 0)
            # A report is serializable after overflow and does not claim a full trace.
            decoded = json.loads(json.dumps(stats))
            self.assertEqual(decoded["invocation_trace"]["policy"], "bounded-most-recent")
            self.assertEqual(decoded["npu_invocations"], 7)
            hook.close()
            self.assertIsNotNone(hook.process.poll())

    def test_failed_native_result_remains_failed_and_is_not_counted(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()):
            hook = self.backend(temporary, "bad-output")
            with self.assertRaisesRegex(RuntimeError, "Nonfinite NPU"):
                hook("a.weight", np.ones((1, 2), dtype=np.float32))
            stats = hook.stats()
            self.assertEqual(stats["npu_invocations"], 0)
            self.assertEqual(stats["invocations"], [])
            self.assertEqual(stats["invocation_trace"]["total"], 0)
            self.assertEqual(stats["invocation_trace"]["dropped"], 0)
            self.assertTrue(hook.closed)
            self.assertIsNotNone(hook.process.poll())


if __name__ == "__main__":
    unittest.main()
