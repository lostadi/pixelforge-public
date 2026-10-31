"""Explicit compiler registration handoff, using fake installers/helpers only."""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import compile_dynamic_fc
import compile_gguf
import compile_grouped_columns
import test_compiler_resume as resume_fixtures
import test_dynamic_fc
import test_npu_protocol
from gguf_hf_loader import GGUFWeights
from npu_linear import NPULinear
from test_gguf_loader import write_gguf
import voice_handoff


class VoiceHandoffTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory(prefix="pixelforge-voice-handoff-")
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        self.received = self.root / "received.json"
        self.installer = self.root / "fake-install.py"
        self.installer.write_text("import json,sys\nfrom pathlib import Path\n"
                                  "Path(" + repr(str(self.received)) + ").write_text(json.dumps(sys.argv[1:]))\n"
                                  "print(json.dumps({'large_install_manifest':'x'*10000}))\n")
        self.args = SimpleNamespace(voice_model_id="local-model", select_voice=True,
                                    replace_voice=True, plan_only=False)

    def register(self, **kwargs):
        log = io.StringIO()
        with patch.object(voice_handoff, "_installer", return_value=(sys.executable, self.installer)), contextlib.redirect_stdout(log):
            result = voice_handoff.register_compiled(self.args, package=self.root / "banks",
                                                     model=self.root / "source.gguf",
                                                     weight_dtype="float32", **kwargs)
        return result, json.loads(log.getvalue())

    def test_literal_installer_handoff_forwards_explicit_selection_and_runtime(self):
        result, event = self.register(selected_tensors=["a.weight", "b.weight"],
                                      dynamic_weight_mode="original_columns", batch_vectors=64)
        self.assertEqual(result, 0)
        argv = json.loads(self.received.read_text())
        self.assertEqual(argv[argv.index("--select-tensors") + 1], "a.weight,b.weight")
        self.assertEqual(argv[argv.index("--dynamic-weight-mode") + 1], "original_columns")
        self.assertEqual(argv[argv.index("--batch-vectors") + 1], "64")
        self.assertTrue({"--apply", "--select", "--replace"}.issubset(argv))
        self.assertTrue(event["compile_complete"])
        self.assertTrue(event["success"])
        self.assertTrue(event["selected"])
        self.assertLess(len(json.dumps(event)), 500)

    def test_handoff_failure_keeps_artifacts_and_reports_registration_separately(self):
        banks = self.root / "banks"
        banks.mkdir()
        saved = banks / "MODEL.json"
        saved.write_text("saved compiler package")
        self.installer.write_text("import sys\nprint('unsupported package',file=sys.stderr)\nsys.exit(2)\n")
        result, event = self.register()
        self.assertEqual(result, 3)
        self.assertTrue(event["compile_complete"])
        self.assertFalse(event["success"])
        self.assertIn("unsupported package", event["error"])
        self.assertEqual(saved.read_text(), "saved compiler package")

    def test_no_voice_id_never_discovers_an_installer(self):
        args = SimpleNamespace(voice_model_id=None)
        with patch.object(voice_handoff, "_installer", side_effect=AssertionError("No installation requested")):
            self.assertEqual(voice_handoff.register_compiled(args, package="unused", model="unused", weight_dtype="int8"), 0)

    def test_cli_selector_without_id_or_plan_only_handoff_is_rejected_early(self):
        for main, family in ((compile_gguf.main, "static"), (compile_dynamic_fc.main, "dynamic"),
                             (compile_grouped_columns.main, "columns")):
            for flags in (["--select-voice"], ["--replace-voice"],
                          ["--voice-model-id", "model", "--plan-only"], ["--voice-model-id", ""],
                          ["--voice-model-id", "Llama"], ["--voice-model-id", "llama.gguf"]):
                argv = ["--model", str(self.root / "unused.gguf"), "--output", str(self.root / "unused"),
                        "--bank-rows", "2", *flags]
                with self.subTest(family=family, flags=flags), contextlib.redirect_stderr(io.StringIO()), \
                        patch.object(voice_handoff, "_installer", side_effect=AssertionError("No installer allowed")):
                    with self.assertRaises(SystemExit) as error:
                        if family == "static":
                            with patch.object(sys, "argv", ["compile_gguf.py"] + argv):
                                main()
                        else:
                            main(argv)
                    self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / "unused").exists())

    def test_installed_kit_uses_only_pinned_trusted_bootstrap_paths(self):
        service = self.root / "trusted/service.py"
        service.parent.mkdir()
        installer = service.parent / "install.py"
        installer.write_text("# installed fixture")
        python = Path(sys.executable)
        bootstrap = dict(service=str(service), python=str(python), ubuntu_pins={
            str(installer): hashlib.sha256(installer.read_bytes()).hexdigest(),
            str(python): hashlib.sha256(python.read_bytes()).hexdigest()})
        with patch.object(voice_handoff, "__file__", str(self.root / "installed/kit/voice_handoff.py")), \
                patch.object(voice_handoff, "_trusted_bootstrap", return_value=bootstrap):
            self.assertEqual(voice_handoff._installer(), (str(python), installer))
            bootstrap["ubuntu_pins"][str(installer)] = "0" * 64
            with self.assertRaisesRegex(ValueError, "pinned"):
                voice_handoff._installer()
            bootstrap["ubuntu_pins"] = []
            with self.assertRaisesRegex(ValueError, "pins"):
                voice_handoff._installer()


class CompilerVoiceHandoffTests(unittest.TestCase):
    def setUp(self):
        self.fixture = resume_fixtures.CompilerResumeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.received = self.root / "installer-args.json"
        self.installer = self.root / "installer.py"
        self.installer.write_text("import json,sys\nfrom pathlib import Path\n"
                                  "Path(" + repr(str(self.received)) + ").write_text(json.dumps(sys.argv[1:]))\n"
                                  "print('{}')\n")

    def compile(self, family, *, fused=False, fail=False):
        source = self.fixture.source("static" if fused else family, fused=fused)
        if family in ("grouped", "columns"):
            with GGUFWeights(source) as weights:
                tensors = [(name, desc.shape, desc.type_id, weights.raw(name).tobytes())
                           for name, desc in weights.tensors.items()]
            value = np.ones((2, 4), dtype="<f4")
            tensors.append(("uncompiled-f32.weight", value.shape, 0, value.tobytes()))
            write_gguf(source, tensors)
        output = self.root / (family + "-compiled")
        module = {"static": compile_gguf, "dynamic": compile_dynamic_fc,
                  "grouped": compile_dynamic_fc, "columns": compile_grouped_columns}[family]
        worker = ([sys.executable, test_npu_protocol.__file__, "--worker", "normal"]
                  if family == "static" else [sys.executable, test_dynamic_fc.__file__, "--worker"])
        argv = ["--model", str(source), "--output", str(output), "--bank-rows", "2",
                "--native-command-json", json.dumps(worker), "--voice-model-id", family, "--select-voice"]
        argv += ["--weight-dtype", "float32"] if family == "static" else ["--batch-vectors", "4"]
        if family == "static" and not fused:
            argv += ["--tensor", "a.weight"]
        if fused:
            argv += ["--fuse-gate-up"]
        if fail:
            self.installer.write_text("import sys\nprint('missing selected banks',file=sys.stderr)\nsys.exit(2)\n")

        def native(*args, **kwargs):
            kwargs["min_free_mib"] = 0
            return NPULinear(*args, **kwargs)

        log = io.StringIO()
        with patch("compiled_model.device_fingerprint", return_value={"device": "compiler-resume-fixture"}), \
                patch.object(module, "NPULinear", native), \
                patch.object(voice_handoff, "_installer", return_value=(sys.executable, self.installer)), \
                patch.dict(os.environ, {"PIXELFORGE_NPU_STDERR": str(self.root / "native.stderr")}), \
                contextlib.redirect_stdout(log):
            if family == "static":
                with patch.object(sys, "argv", ["compile_gguf.py"] + argv):
                    result = module.main()
            else:
                result = module.main(argv, original_groups=True) if family == "grouped" else module.main(argv)
        return result, [json.loads(line) for line in log.getvalue().splitlines()], output

    def test_static_partial_handoff_forwards_actual_tensors_and_failure_keeps_compile(self):
        result, events, output = self.compile("static", fail=True)
        self.assertEqual(result, 3)
        self.assertEqual(events[-2]["event"], "complete")
        self.assertEqual(events[-2]["coverage"], "partial")
        self.assertEqual(events[-1]["event"], "voice_registration")
        self.assertTrue(events[-1]["compile_complete"])
        self.assertFalse(events[-1]["success"])
        self.assertTrue((output / "MODEL.json").is_file())
        self.assertTrue(list((output / "graphs").glob("*.dgc")))

    def test_static_success_has_short_voice_event_after_compiler_completion(self):
        result, events, output = self.compile("static")
        self.assertEqual(result, 0)
        self.assertEqual(events[-2]["event"], "complete")
        self.assertTrue(events[-1]["success"])
        argv = json.loads(self.received.read_text())
        self.assertEqual(argv[argv.index("--from-compiled") + 1], str(output))
        self.assertEqual(argv[argv.index("--select-tensors") + 1], "a.weight")

    def test_geometry_handoff_uses_leaf_and_only_original_types_actually_compiled(self):
        for family, mode, directory in (("dynamic", "float32", "dynamic-fc-v1"),
                                        ("grouped", "original_groups", "grouped-fc-v1"),
                                        ("columns", "original_columns", "grouped-columns-v1")):
            with self.subTest(family=family):
                result, events, output = self.compile(family)
                self.assertEqual(result, 0)
                self.assertTrue(events[-1]["success"])
                argv = json.loads(self.received.read_text())
                self.assertEqual(argv[argv.index("--from-compiled") + 1], str(output / directory / "b4"))
                self.assertEqual(argv[argv.index("--dynamic-weight-mode") + 1], mode)
                if family != "dynamic":
                    selected = argv[argv.index("--select-tensors") + 1].split(",")
                    self.assertEqual(selected, ["a.weight", "b.weight", "c.weight"])
                    self.assertNotIn("uncompiled-f32.weight", selected)

    def test_fused_handoff_expands_original_members_and_preserves_recipe_flags(self):
        result, events, output = self.compile("static", fused=True)
        self.assertEqual(result, 0)
        argv = json.loads(self.received.read_text())
        self.assertEqual(argv[argv.index("--from-compiled") + 1], str(output / "fused-ffn-v1"))
        self.assertIn("--fused-gate-up", argv)
        self.assertEqual(argv[argv.index("--fused-bank-rows") + 1], "2")
        self.assertEqual(argv[argv.index("--fused-gate-up-tensors") + 1], "blk.0.ffn_gate_up.weight")
        self.assertEqual(argv[argv.index("--select-tensors") + 1].split(","),
                         ["blk.0.ffn_down.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_up.weight"])


if __name__ == "__main__":
    unittest.main()
