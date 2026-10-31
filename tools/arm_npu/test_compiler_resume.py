"""Compiler resume fixtures: saved bytecode is verified without native loads."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import compile_dynamic_fc
import compile_gguf
import compile_grouped_columns
from compiled_model import ArtifactCache
from npu_linear import NPULinear
from test_gguf_loader import write_gguf
import test_dynamic_fc
import test_npu_protocol


class CompilerResumeTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory(prefix="pixelforge-compiler-resume-")
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        self.fingerprint = {"device": "compiler-resume-fixture"}

    def source(self, family, *, fused=False):
        source = self.root / (family + ("-fused" if fused else "") + ".gguf")
        if source.exists():
            return source
        tensors = []
        specs = (("a.weight", 3, 4), ("b.weight", 3, 4), ("c.weight", 2, 8))
        if fused:
            specs = (("blk.0.ffn_gate.weight", 3, 4),
                     ("blk.0.ffn_up.weight", 3, 4), ("blk.0.ffn_down.weight", 4, 3))
        for name, rows, columns in specs:
            if family in ("grouped", "columns"):
                columns = (4096 if family == "columns" else 512) if name != "c.weight" else 256
                tensors.append((name, (rows, columns), 13, bytes(rows * (columns // 256) * 176)))
            else:
                value = (np.arange(rows * columns, dtype=np.float32).reshape(rows, columns) / 100).astype("<f4")
                tensors.append((name, value.shape, 0, value.tobytes()))
        write_gguf(source, tensors)
        return source

    def package(self, family, *, output=None, batch=1, fused=False):
        output = output or self.root / (family + "-package")
        suffix = {"dynamic": "dynamic-fc-v1", "grouped": "grouped-fc-v1", "columns": "grouped-columns-v1"}
        if family != "static":
            return output / suffix[family] / "b4"
        if fused:
            output = output / "fused-ffn-v1"
        return output if batch == 1 else output / "batch-fc-v1" / ("b" + str(batch))

    def compile(self, family, *, tensor=None, resume=False, allow_helper=True, output=None, batch=1, fused=False):
        module = {"static": compile_gguf, "dynamic": compile_dynamic_fc,
                  "grouped": compile_dynamic_fc, "columns": compile_grouped_columns}[family]
        source = self.source(family, fused=fused)
        output = output or self.root / (family + "-package")
        worker = ([sys.executable, test_npu_protocol.__file__, "--worker", "normal"]
                  if family == "static" else [sys.executable, test_dynamic_fc.__file__, "--worker"])
        args = ["--model", str(source), "--output", str(output), "--bank-rows", "2",
                "--native-command-json", json.dumps(worker)]
        args += ["--weight-dtype", "float32", "--batch-vectors", str(batch)] if family == "static" else ["--batch-vectors", "4"]
        if fused:
            args += ["--fuse-gate-up"]
        if tensor:
            args += ["--tensor", tensor]
        if resume:
            args += ["--resume"]
        counts = dict.fromkeys(("C", "L", "D", "F", "R", "V"), 0)
        counts["helper_starts"] = 0
        original_write = NPULinear._write

        def write(native, *items):
            operation = items[0].split(maxsplit=1)[0].decode()
            if operation in counts:
                counts[operation] += 1
            return original_write(native, *items)

        def start(*a, **kwargs):
            if not allow_helper:
                raise AssertionError("Verified cache hit must not start a native helper")
            counts["helper_starts"] += 1
            kwargs["min_free_mib"] = 0
            return NPULinear(*a, **kwargs)

        log = io.StringIO()
        with patch("compiled_model.device_fingerprint", return_value=self.fingerprint), \
                patch.object(NPULinear, "_write", write), patch.object(module, "NPULinear", start), \
                patch.dict(os.environ, {"PIXELFORGE_NPU_STDERR": str(self.root / "native.stderr")}), \
                contextlib.redirect_stdout(log):
            if family == "static":
                with patch.object(sys, "argv", ["compile_gguf.py"] + args):
                    result = module.main()
            else:
                result = module.main(args, original_groups=True) if family == "grouped" else module.main(args)
        self.assertEqual(result, 0)
        events = [json.loads(line) for line in log.getvalue().splitlines()]
        self.assertEqual(events[-1]["event"], "complete")
        self.assertEqual(events[-1]["npu_invocations"], 0)
        self.assertEqual(counts["R"] + counts["V"], 0)
        return events, counts

    def corrupt(self, family, *, output=None):
        directory = self.package(family, output=output)
        manifest = json.loads((directory / "MODEL.json").read_text())
        entry = next(iter(manifest["artifacts"].values()))
        record = json.loads((directory / entry["metadata"]).read_text())
        path = directory / record["bytecode_path"]
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            value = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([value[0] ^ 1]))

    def test_static_cold_partial_and_full_resume_report_actual_work(self):
        partial, first = self.compile("static", tensor="a.weight")
        self.assertEqual((first["C"], first["L"], first["F"]), (2, 0, 2))
        self.assertEqual(partial[-1]["coverage"], "partial")
        self.assertFalse(json.loads((self.package("static") / "MODEL.json").read_text())["complete"])

        events, counts = self.compile("static", resume=True)
        self.assertEqual((counts["C"], counts["L"], counts["F"]), (3, 0, 3))
        self.assertEqual((events[-1]["compiled_graphs"], events[-1]["loaded_graphs"], events[-1]["reused_graphs"]), (3, 0, 2))
        self.assertTrue(events[-1]["native_helper_started"])
        manifest = json.loads((self.package("static") / "MODEL.json").read_text())
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["completed_weight_dtypes"], ["float32"])
        self.assertEqual(len(manifest["artifacts"]), 5)

        events, counts = self.compile("static", resume=True, allow_helper=False)
        self.assertEqual(sum(counts.values()), 0)
        self.assertEqual((events[-1]["compiled_graphs"], events[-1]["loaded_graphs"], events[-1]["reused_graphs"]), (0, 0, 5))
        self.assertFalse(events[-1]["native_helper_started"])
        self.assertEqual(sum(event["event"] == "artifact_ready" for event in events), 5)

    def test_static_cached_partial_selection_does_not_mark_missing_banks_complete(self):
        self.compile("static", tensor="a.weight")
        events, counts = self.compile("static", tensor="a.weight", resume=True, allow_helper=False)
        self.assertEqual(events[-1]["coverage"], "partial")
        self.assertEqual(events[-1]["reused_graphs"], 2)
        self.assertEqual(sum(counts.values()), 0)
        self.assertFalse(json.loads((self.package("static") / "MODEL.json").read_text())["complete"])

    def test_static_batched_full_resume_needs_no_helper(self):
        self.compile("static", batch=4)
        events, counts = self.compile("static", batch=4, resume=True, allow_helper=False)
        self.assertEqual(events[-1]["batch_vectors"], 4)
        self.assertEqual(events[-1]["reused_graphs"], 5)
        self.assertEqual(sum(counts.values()), 0)
        self.assertTrue(json.loads((self.package("static", batch=4) / "MODEL.json").read_text())["complete"])

    def test_fused_and_batched_fused_full_resume_need_no_helper(self):
        source = self.source("static", fused=True)
        before = source.read_bytes()
        for batch in (1, 4):
            with self.subTest(batch=batch):
                output = self.root / ("fused-b" + str(batch))
                _, cold = self.compile("static", output=output, batch=batch, fused=True)
                self.assertEqual((cold["C"], cold["L"], cold["F"]), (5, 0, 5))
                events, counts = self.compile("static", output=output, batch=batch, fused=True,
                                             resume=True, allow_helper=False)
                self.assertEqual(sum(counts.values()), 0)
                self.assertEqual(events[-1]["reused_graphs"], 5)
                self.assertEqual(events[-1]["batch_vectors"], batch)
                self.assertFalse(events[-1]["native_helper_started"])
                manifest = json.loads((self.package("static", output=output, batch=batch, fused=True) / "MODEL.json").read_text())
                self.assertTrue(manifest["complete"])
                self.assertEqual(len(manifest["artifacts"]), 5)
                self.assertEqual(manifest["plan"]["fused_source_tensors"], {
                    "blk.0.ffn_gate_up.weight": ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"]})
        self.assertEqual(source.read_bytes(), before)

    def test_geometry_cold_partial_and_full_resume_need_only_missing_programs(self):
        for family in ("dynamic", "grouped", "columns"):
            with self.subTest(family=family):
                _, first = self.compile(family, tensor="a.weight")
                self.assertEqual((first["C"], first["L"], first["F"]), (2, 0, 2))
                events, counts = self.compile(family, resume=True)
                self.assertEqual((counts["C"], counts["L"], counts["F"]), (1, 0, 1))
                self.assertEqual(events[-1]["compiled_geometries"], 1)
                manifest = json.loads((self.package(family) / "MODEL.json").read_text())
                self.assertEqual(len(manifest["artifacts"]), 3)
                self.assertEqual(manifest["complete"], family == "columns")
                events, counts = self.compile(family, resume=True, allow_helper=False)
                self.assertEqual(events[-1]["compiled_geometries"], 0)
                self.assertEqual(sum(counts.values()), 0)

    def test_corrupt_saved_program_fails_before_helper_startup_in_every_family(self):
        for family in ("static", "dynamic", "grouped", "columns"):
            with self.subTest(family=family):
                self.compile(family)
                self.corrupt(family)
                with self.assertRaisesRegex(ValueError, "hash/identifier"):
                    self.compile(family, resume=True, allow_helper=False)

    def test_static_rechecks_saved_bytes_after_preflight(self):
        self.compile("static")
        original = compile_gguf.disk_preflight

        def preflight(*args, **kwargs):
            result = original(*args, **kwargs)
            self.corrupt("static")
            return result

        with patch.object(compile_gguf, "disk_preflight", preflight):
            with self.assertRaisesRegex(ValueError, "hash/identifier"):
                self.compile("static", resume=True, allow_helper=False)

    def test_static_source_replacement_after_preflight_is_rejected(self):
        self.compile("static")
        original = compile_gguf.disk_preflight

        def preflight(*args, **kwargs):
            result = original(*args, **kwargs)
            source = self.source("static")
            replacement = self.root / "replacement.gguf"
            replacement.write_bytes(source.read_bytes())
            replacement.replace(source)
            return result

        with patch.object(compile_gguf, "disk_preflight", preflight):
            with self.assertRaisesRegex(ValueError, "Source GGUF changed"):
                self.compile("static", resume=True, allow_helper=False)

    def test_static_full_resume_still_checks_every_bank_before_completion(self):
        # Publish all banks without the final marker, as after interruption.
        with patch.object(ArtifactCache, "mark_complete"):
            self.compile("static")
        self.assertFalse(json.loads((self.package("static") / "MODEL.json").read_text())["complete"])
        original = ArtifactCache.mark_complete

        def complete(cache, weight_dtype):
            self.corrupt("static")
            return original(cache, weight_dtype)

        with patch.object(ArtifactCache, "mark_complete", complete):
            with self.assertRaisesRegex(ValueError, "hash/identifier"):
                self.compile("static", resume=True, allow_helper=False)
        self.assertFalse(json.loads((self.package("static") / "MODEL.json").read_text())["complete"])


if __name__ == "__main__":
    unittest.main()
