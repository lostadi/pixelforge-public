"""Generic GGUF installer preparation and compiled-package registration inputs."""
from pathlib import Path
import hashlib
import json
import sys
import unittest
from unittest.mock import patch

CHAT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHAT))
sys.path.insert(0, str(CHAT.parent / "tools/arm_npu"))
import config
import install
import model_info
import test_install as installer_fixtures
from compiled_model import ArtifactCache, batch_directory


class GenericInstallerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = installer_fixtures.InstallerTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.args = self.fixture.args
        self.args.architecture = None
        self.args.model_name = None
        self.args.provenance_repo = None
        self.fixture.model_architecture = "llama"

    def snapshot(self):
        return {str(path.relative_to(self.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                if path.is_file() else "directory" for path in self.root.rglob("*")}

    def test_non_huihui_llama_infers_architecture_name_and_native_tokens(self):
        self.args.provenance_repo = "local-training/plain-llama"
        before = self.snapshot()
        manifest, sources, _ = self.fixture.prepare()
        model = manifest["profile"]["model"]
        self.assertEqual((model["architecture"], model["model_name"]), ("llama", "Local fixture decoder"))
        self.assertEqual(model["provenance_repo"], "local-training/plain-llama")
        self.assertEqual(model["prompt_mode"], "chat-template")
        self.assertEqual(manifest["profile"]["native_model_probe"]["bos_token"], "<fixture-bos>")
        self.assertIn("model_info.py", sources)
        self.assertEqual(before, self.snapshot())

    def test_model_leaf_symlink_prepares_canonical_pinned_profile(self):
        canonical = self.fixture.model.resolve(strict=True)
        alias = self.root / "model-alias.gguf"
        alias.symlink_to(canonical)
        self.args.model = alias
        self.args._compiled_sha256 = config.sha256(canonical)
        before = self.snapshot()
        manifest, _, profile_bytes = self.fixture.prepare()
        self.assertEqual(self.args.model, canonical)
        for profile in (manifest["profile"], json.loads(profile_bytes)):
            model = Path(profile["model_path"])
            self.assertEqual(model, canonical)
            self.assertTrue(model.is_file())
            self.assertFalse(model.is_symlink())
            self.assertEqual(profile["model"]["model_sha256"], config.sha256(canonical))
        self.assertTrue(alias.is_symlink())
        self.assertEqual(before, self.snapshot())

    def large_dense_plan(self, count):
        names = ["blk." + str(index) + ".ffn_up.weight" for index in range(count)]
        self.fixture.plan_overrides = {"dense_tensors": [
            {"name": name, "rows": 4, "columns": 2, "bank_rows": 2}
            for name in names]}
        manifest = self.fixture.banks / "MODEL.json"
        data = json.loads(manifest.read_text())
        data["complete"] = True
        manifest.write_text(json.dumps(data))
        return names

    def checked_preflight(self, observed):
        original = install.preflight_artifacts
        def check(profile, weights, plan):
            import compiled_model
            base = compiled_model.ArtifactCache
            class CheckedCache(base):
                def get(inner, name, *args, **kwargs):
                    observed.append(name)
                    return super().get(name, *args, **kwargs)
            return original(profile, weights, plan, cache_factory=CheckedCache)
        return check

    def test_exact_full_dense_csv_normalizes_both_phases_and_preflights_every_bank(self):
        for count in (281, 600):
            with self.subTest(count=count):
                names = self.large_dense_plan(count)
                self.args.select_tensors = ",".join(reversed(names))
                self.args.select_prefill_tensors = ",".join(names)
                self.args.npu_prefill = True
                observed = []
                before = self.snapshot()
                with patch.object(install, "preflight_artifacts", side_effect=self.checked_preflight(observed)):
                    manifest, _, profile_bytes = self.fixture.prepare()
                for profile in (manifest["profile"], json.loads(profile_bytes)):
                    self.assertIsNone(profile["selected_npu_weight_tensors"])
                    self.assertIsNone(profile["selected_prefill_npu_weight_tensors"])
                    self.assertEqual(profile["model"]["model_name"], "Local fixture decoder")
                    self.assertEqual(profile["model"]["model_sha256"], config.sha256(self.fixture.model))
                self.assertEqual(set(observed), set(names))
                self.assertEqual(len(observed), count * 2)
                self.assertEqual(before, self.snapshot())

    def test_partial_281_dense_csv_preserves_selection_and_preflights_only_members(self):
        names = self.large_dense_plan(600)
        selected = names[:281]
        self.args.select_tensors = ",".join(selected)
        self.args.select_prefill_tensors = ",".join(reversed(selected))
        self.args.npu_prefill = True
        observed = []
        before = self.snapshot()
        with patch.object(install, "preflight_artifacts", side_effect=self.checked_preflight(observed)):
            manifest, _, _ = self.fixture.prepare()
        self.assertEqual(manifest["profile"]["selected_npu_weight_tensors"], selected)
        self.assertEqual(manifest["profile"]["selected_prefill_npu_weight_tensors"], list(reversed(selected)))
        self.assertEqual(set(observed), set(selected))
        self.assertEqual(len(observed), len(selected) * 2)
        self.assertEqual(before, self.snapshot())

    def test_full_prefill_with_partial_decode_stays_explicit(self):
        names = self.large_dense_plan(281)
        self.args.select_tensors = names[0]
        self.args.select_prefill_tensors = ",".join(names)
        self.args.npu_prefill = True
        observed = []
        with patch.object(install, "preflight_artifacts", side_effect=self.checked_preflight(observed)):
            manifest, _, _ = self.fixture.prepare()
        self.assertEqual(manifest["profile"]["selected_npu_weight_tensors"], names[:1])
        self.assertEqual(manifest["profile"]["selected_prefill_npu_weight_tensors"], names)
        self.assertEqual(set(observed), set(names))

    def test_full_ffn_all_dense_members_stay_explicit(self):
        self.fixture.model_architecture = "qwen35"
        self.args.full_ffn = self.args.npu_prefill = True
        self.args.weight_dtype = "float32"
        self.args.full_ffn_tensors = "blk.0.ffn_gate.weight"
        self.args.full_ffn_phases = "PD"
        self.args.full_ffn_hidden_chunk = 2
        selected = ["blk.0.ffn_gate.weight", "blk.0.ffn_up.weight", "blk.0.ffn_down.weight"]
        self.args.select_tensors = ",".join(selected)
        self.args.select_prefill_tensors = ",".join(reversed(selected))
        sidecar = self.fixture.banks / "full-ffn-v1/b1/MODEL.json"
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text("{}")
        manifest, _, _ = self.fixture.prepare()
        self.assertEqual(manifest["profile"]["selected_npu_weight_tensors"], selected)
        self.assertEqual(manifest["profile"]["selected_prefill_npu_weight_tensors"], list(reversed(selected)))
        self.assertEqual(manifest["profile"]["full_ffn_tensors"], [selected[0]])

    def test_malformed_full_csv_cannot_normalize_away_duplicate_unknown_or_empty_members(self):
        names = self.large_dense_plan(600)
        for option in ("select_tensors", "select_prefill_tensors"):
            for selected in ("", ",".join(names + [names[0]]),
                             ",".join(names[:-1] + ["unknown.weight"]),
                             ",".join(names + [""])):
                with self.subTest(option=option, length=len(selected)):
                    self.args.select_tensors = names[0]
                    self.args.select_prefill_tensors = None
                    setattr(self.args, option, selected)
                    before = self.snapshot()
                    with patch.object(install, "preflight_artifacts", side_effect=AssertionError("Invalid CSV reached preflight")):
                        with self.assertRaises(ValueError):
                            self.fixture.prepare()
                    self.assertEqual(before, self.snapshot())

    def test_no_embedded_template_prepares_explicit_raw_completion(self):
        self.fixture.chat_template = None
        before = self.snapshot()
        manifest, _, _ = self.fixture.prepare()
        self.assertEqual(manifest["profile"]["model"]["prompt_mode"], "raw-completion")
        self.assertIsNone(manifest["profile"]["model"]["provenance_repo"])
        self.assertEqual(before, self.snapshot())

    def test_new_registration_rejects_older_native_probe_for_both_prompt_modes(self):
        self.fixture.native_probe_omissions = ("prompt_modes",)
        for template in (None, "{{ messages[0].content }}"):
            self.fixture.chat_template = template
            before = self.snapshot()
            with self.subTest(template=template), self.assertRaisesRegex(ValueError, "rebuild"):
                self.fixture.prepare()
            self.assertEqual(before, self.snapshot())

    def test_compiled_source_hash_mismatch_prevents_all_writes(self):
        self.args._compiled_sha256 = "0" * 64
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "different original GGUF bytes"):
            self.fixture.prepare()
        self.assertEqual(before, self.snapshot())

    def test_architecture_assertion_cannot_override_original_model(self):
        self.args.architecture = "gemma4"
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "supplied architecture"):
            self.fixture.prepare()
        self.assertEqual(before, self.snapshot())

    def test_native_decoder_rejection_prevents_all_writes(self):
        self.fixture.native_probe_error = ValueError("The native engine requires a decoder-only GGUF")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "decoder-only"):
            self.fixture.prepare()
        self.assertEqual(before, self.snapshot())

    def test_native_and_gguf_metadata_mismatch_prevents_all_writes(self):
        self.fixture.native_probe_overrides = {"architecture": "qwen35"}
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "architecture differs"):
            self.fixture.prepare()
        self.assertEqual(before, self.snapshot())

    def test_matrix_only_plan_is_rejected_without_writes(self):
        self.fixture.plan_overrides = {"generation_adapter_supported": False,
                                       "unsupported_tensor_operations": [{"reason": "indexed expert operations"}]}
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "matrix-only.*indexed expert"):
            self.fixture.prepare()
        self.assertEqual(before, self.snapshot())

    def test_invalid_context_and_output_limits_fail_before_writes(self):
        for key, value in (("context_size", 0), ("context_size", 128), ("context_size", 8193),
                           ("max_output_tokens", 0), ("max_output_tokens", 1),
                           ("max_output_tokens", 513), ("max_output_tokens", True)):
            old = getattr(self.args, key)
            setattr(self.args, key, value)
            before = self.snapshot()
            try:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.fixture.prepare()
                self.assertEqual(before, self.snapshot())
            finally:
                setattr(self.args, key, old)

    def test_context_uses_actual_model_limit_and_cannot_be_expanded_by_request(self):
        self.fixture.model_fields = {"llama.context_length": 256}
        manifest, _, _ = self.fixture.prepare()
        model = manifest["profile"]["model"]
        self.assertEqual(model["context_per_worker"], 256)
        self.assertEqual(model["max_output_tokens"], 248)
        self.args.context_size = 512
        with self.assertRaisesRegex(ValueError, "exceeds the GGUF/native model context"):
            self.fixture.prepare()

    def test_invalid_labels_and_malformed_template_fail_before_writes(self):
        for key, value in (("model_name", "x" * 257), ("provenance_repo", "x" * 513),
                           ("provenance_repo", "local\nmodel")):
            old = getattr(self.args, key)
            setattr(self.args, key, value)
            before = self.snapshot()
            try:
                with self.subTest(key=key), self.assertRaises(ValueError):
                    self.fixture.prepare()
                self.assertEqual(before, self.snapshot())
            finally:
                setattr(self.args, key, old)
        self.fixture.chat_template = "{% if %}"
        with self.assertRaisesRegex(ValueError, "template is malformed"):
            self.fixture.prepare()

    def package(self, *, batch=4):
        source = self.fixture.model
        digest = config.sha256(source)
        plan = {"architecture": "llama", "model_sha256": digest,
                "unsupported_tensor_operations": [], "dense_tensors": [
                    {"name": "blk.0.ffn_up.weight", "rows": 4, "columns": 2, "bank_rows": 2}]}
        base = self.root / "compiled"
        leaf = batch_directory(base, batch)
        cache = ArtifactCache(leaf, source, fingerprint={"device": "fixture"},
                              plan=plan, batch_vectors=batch)
        code = b"\x08\0\0\0DGC0" + b"saved fixture"
        record = {"graph_sha256": "b" * 64, "graph_bytes": 400, "compiled_bytes": len(code)}
        for start in (0, 2):
            cache.store("blk.0.ffn_up.weight", start, start + 2, "float32", 2, record, code)
        return leaf

    def test_real_compiled_leaf_infers_model_dtype_batch_and_reuses_only_platform(self):
        leaf = self.package()
        existing = self.root / "existing-service"
        bootstrap = {"ubuntu_root": str(self.root), "python": sys.executable,
                     "service": str(existing / "service.py")}
        profile = {"native_helper_path": str(self.fixture.helper),
                   "termux_prefix": str(self.fixture.home.parent / "usr"),
                   "model_path": "/old/unrelated.gguf", "architecture": "qwen35",
                   "artifacts": "/old/unrelated-banks", "weight_dtype": "int8"}
        old_manifest = {"writes": [str(self.fixture.runtime / "pixelforge-chat.json")]}
        seen = []

        def root_json(path):
            seen.append(Path(path))
            return bootstrap if Path(path).name == "bootstrap.json" else profile if Path(path).name == "profile.json" else old_manifest

        before = self.snapshot()
        with patch.object(config, "root_json", side_effect=root_json):
            args = install.parser().parse_args(["--from-compiled", str(leaf), "--model-id", "plain-llama",
                "--native-binary", str(self.fixture.helper),
                "--select-tensors", "blk.0.ffn_up.weight", "--state-dir", str(self.root / "state")])
        self.assertEqual(args.model, self.fixture.model)
        self.assertEqual(args.artifacts, self.root / "compiled")
        self.assertEqual((args.weight_dtype, args.batch_vectors, args.decode_batch_vectors), ("float32", 4, 4))
        self.assertEqual(args.dynamic_weight_mode, "off")
        self.assertTrue(args.npu_prefill)
        self.assertEqual(args.termux_home, self.fixture.home)
        self.assertEqual(args.app_runtime, self.fixture.runtime)
        self.assertTrue(any(path.name == "bootstrap.json" for path in seen))
        self.assertEqual(before, self.snapshot())
        self.fixture.args = args
        manifest, _, _ = self.fixture.prepare()
        self.assertEqual(manifest["registration"], {"model_id": "plain-llama", "select": False})
        self.assertTrue(str(args.service_dir).endswith("/models/plain-llama"))
        self.assertEqual(manifest["profile"]["model"]["architecture"], "llama")

    def test_platform_reuse_after_selecting_a_previously_unselected_registration(self):
        leaf = self.package()
        self.args.model_id = "stored-model"
        self.args.select = False
        manifest, _, _ = self.fixture.prepare()
        self.assertFalse(any(Path(path).name == "pixelforge-chat.json" for path in manifest["writes"]))
        self.assertEqual(manifest["app_runtime"], str(self.fixture.runtime))
        bootstrap = manifest["bootstrap"]
        profile = manifest["profile"]

        def root_json(path):
            return bootstrap if Path(path).name == "bootstrap.json" else profile if Path(path).name == "profile.json" else manifest

        before = self.snapshot()
        with patch.object(config, "root_json", side_effect=root_json):
            args = install.parser().parse_args(["--from-compiled", str(leaf), "--model-id", "another-model",
                "--native-binary", str(self.fixture.helper),
                "--select-tensors", "blk.0.ffn_up.weight", "--state-dir", str(self.root / "state")])
        self.assertEqual(args.app_runtime, self.fixture.runtime)
        self.assertEqual(args.termux_home, self.fixture.home)
        self.assertEqual(args.model, self.fixture.model)
        self.assertEqual(args.artifacts, self.root / "compiled")
        self.assertEqual(before, self.snapshot())

    def test_registry_rejection_precedes_bundle_writes_and_keeps_active_files(self):
        import os
        import registry
        self.args.ubuntu_root = Path("/")
        self.args.model_id = "active"
        self.args.state_dir.mkdir(mode=0o700)
        active_bootstrap = self.args.state_dir / "bootstrap.json"
        active_metadata = self.fixture.runtime / "pixelforge-chat.json"
        active_bootstrap.write_bytes(b"existing active bootstrap")
        active_metadata.write_bytes(b"existing active metadata")
        target = self.root / "new-model-bundle"
        manifest = {"service_dir": str(target), "termux_uid": 10001, "writes": [],
                    "client_metadata": {"backend": install.BACKEND}, "bootstrap": {"fixture": True}}
        original_stat = Path.stat

        def owners(path, *args, **kwargs):
            result = original_stat(path, *args, **kwargs)
            fields = list(result)
            fields[4] = fields[5] = 10001 if path in (self.fixture.home, self.fixture.runtime) else 0
            return os.stat_result(fields)

        before = self.snapshot()
        with patch.object(Path, "stat", owners), patch.object(install.os, "getuid", return_value=0), \
                patch.object(registry, "register_model", side_effect=ValueError("Model ID already registered")) as register, \
                patch.object(install, "atomic", side_effect=AssertionError("Registry rejection must precede writes")) as atomic:
            with self.assertRaisesRegex(ValueError, "already registered"):
                install.apply(self.args, manifest, {}, b"new profile")
        self.assertTrue(callable(register.call_args.kwargs["installer"]))
        atomic.assert_not_called()
        self.assertEqual(before, self.snapshot())
        self.assertFalse(target.exists())

    def test_explicit_cpu_prefill_survives_compiled_leaf_inference(self):
        leaf = self.package()
        args = install.parser().parse_args(["--from-compiled", str(leaf),
            "--ubuntu-root", str(self.root), "--python", sys.executable,
            "--helper", str(self.fixture.helper), "--termux-home", str(self.fixture.home),
            "--app-runtime", str(self.fixture.runtime), "--native-binary", str(self.fixture.helper),
            "--cpu-prefill", "--select-tensors", "blk.0.ffn_up.weight"])
        self.assertFalse(args.npu_prefill)


if __name__ == "__main__":
    unittest.main()
