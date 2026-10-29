"""Persistent artifact integrity and fake D/L protocol checks; no NPU calls."""
from __future__ import annotations

import concurrent.futures
import hashlib
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from compiled_model import ArtifactCache
from npu_linear import NPULinear
import test_npu_protocol


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source.gguf"
        self.source.write_bytes(b"fixture-only source identity; not a parsed GGUF")
        self.digest = hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.fingerprint = {"compiler_library_sha256": "a" * 64, "firmware_version": "fixture"}
        self.plan = {"architecture": "fixture", "model_sha256": self.digest,
                     "unsupported_tensor_operations": [],
                     "dense_tensors": [{"name": "a.weight", "rows": 2, "columns": 2, "bank_rows": 2}]}
        environment = patch.dict(os.environ, {"PIXELFORGE_NPU_STDERR": str(self.root / "native.stderr")})
        environment.start()
        self.addCleanup(environment.stop)

    def cache(self, directory=None, **kwargs):
        return ArtifactCache(directory or self.root / "package", self.source,
                             fingerprint=self.fingerprint, plan=self.plan, **kwargs)

    def backend(self, cache):
        weights = test_npu_protocol.FakeWeights()
        weights.path = self.source
        hook = NPULinear(weights, command=[sys.executable, test_npu_protocol.__file__, "--worker", "normal"],
                         artifact_cache=cache, min_free_mib=0, cache_mib=0)
        self.addCleanup(hook.close)
        return hook

    def saved_fixture(self):
        cache = self.cache()
        code = b"\x08\0\0\0DGC0" + b"fixture-bytecode"
        record = {"graph_sha256": "b"*64, "graph_bytes": 400, "compiled_bytes": len(code)}
        stored = cache.store("a.weight", 0, 2, "int8", 2, record, code)
        return cache, stored, code

    def test_fake_export_then_fresh_readonly_load_without_dequant(self):
        cache = self.cache()
        first = self.backend(cache)
        activation = np.asarray([[4209.0, -11.0]], np.float32)
        expected = first("a.weight", activation)
        self.assertEqual(first.stats()["compiled_models"], 1)
        first.close()
        readonly = self.cache(read_only=True)
        second = self.backend(readonly)
        with patch.object(second.weights, "numpy_rows", side_effect=AssertionError("Dense weights must not decode")):
            actual = second("a.weight", activation)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(second.stats()["compiled_models"], 0)
        self.assertEqual(second.stats()["loaded_models"], 1)
        self.assertEqual(second.stats()["npu_invocations"], 1)
        self.assertEqual(cache.directory.stat().st_mode & 0o777, 0o700)
        self.assertTrue(all((p.stat().st_mode & 0o777) == 0o600 for p in cache.graphs.iterdir()))

    def test_corrupt_bytecode_stops_before_load(self):
        cache, stored, _ = self.saved_fixture()
        path = cache.directory / stored["bytecode_path"]
        with path.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"X")
        with self.assertRaisesRegex(ValueError, "hash/identifier"):
            cache.get("a.weight", 0, 2, "int8", 2)

    def test_reservation_precedes_bytecode_read(self):
        cache, _, _ = self.saved_fixture()
        def reject(_):
            raise MemoryError("fixture memory pressure")
        with patch.object(Path, "read_bytes", side_effect=AssertionError("Must reserve first")):
            with self.assertRaisesRegex(MemoryError, "fixture memory pressure"):
                cache.get("a.weight", 0, 2, "int8", 2, reserve=reject)

    def test_persistent_load_pressure_stops_before_l_and_allocation(self):
        from npu_linear import MIB
        cache, _, _ = self.saved_fixture()
        hook = self.backend(cache)
        hook.min_free = 128 * MIB
        with patch("npu_linear._memory_available", return_value=128 * MIB), \
                patch.object(Path, "read_bytes", side_effect=AssertionError("Must reserve before code allocation")), \
                patch.object(hook, "_write", wraps=hook._write) as writes:
            with self.assertRaisesRegex(MemoryError, "compile working set"):
                hook("a.weight", np.ones((1, 2), np.float32))
        self.assertEqual(writes.call_count, 0)
        self.assertTrue(hook.closed)

    def test_readonly_missing_and_fingerprint_mismatch_fail(self):
        cache, _, _ = self.saved_fixture()
        readonly = self.cache(read_only=True)
        with self.assertRaises(FileNotFoundError):
            readonly.get("missing.weight", 0, 2, "int8", 2)
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            ArtifactCache(cache.directory, self.source, read_only=True,
                          fingerprint={"firmware_version": "changed"})

    def test_complete_requires_every_bank_and_hash(self):
        cache = self.cache()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            cache.mark_complete()
        _, _, _ = self.saved_fixture()
        cache.mark_complete()
        self.assertTrue(json.loads(cache.manifest_path.read_text())["complete"])
        self.assertEqual(json.loads(cache.manifest_path.read_text())["completed_weight_dtypes"], ["int8"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            cache.mark_complete("float32")

    def test_embedded_escape_and_symlink_rejected(self):
        cache = self.cache()
        (cache.directory / "model.gguf").symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "symlink"):
            cache.embed_source()
        value = json.loads(cache.manifest_path.read_text())
        value["source_gguf"].update(embedded=True, path="../source.gguf")
        cache.manifest_path.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "contained relative"):
            ArtifactCache(cache.directory, read_only=True, fingerprint=self.fingerprint)

    def test_concurrent_creators_keep_one_identity(self):
        barrier = threading.Barrier(2)
        def create(_):
            barrier.wait()
            return self.cache().source_sha256
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            self.assertEqual(list(executor.map(create, range(2))), [self.digest, self.digest])
        self.assertEqual(json.loads((self.root / "package/MODEL.json").read_text())["source_gguf"]["sha256"], self.digest)

    def test_failed_initial_manifest_publication_can_retry(self):
        import compiled_model
        original = compiled_model.os.replace
        def fail_manifest(source, destination):
            if Path(destination).name == "MODEL.json":
                raise OSError("injected first publication failure")
            return original(source, destination)
        with patch("compiled_model.os.replace", side_effect=fail_manifest):
            with self.assertRaisesRegex(OSError, "first publication failure"):
                self.cache()
        self.assertEqual(list((self.root / "package").iterdir()), [])
        cache = self.cache()
        self.assertEqual(cache.source_sha256, self.digest)
        self.assertTrue(cache.graphs.is_dir())

    def test_changed_bank_plan_and_source_inode_fail(self):
        cache = self.cache()
        changed = json.loads(json.dumps(self.plan))
        changed["dense_tensors"][0]["bank_rows"] = 1
        with self.assertRaisesRegex(ValueError, "architecture plan differs"):
            ArtifactCache(cache.directory, self.source, fingerprint=self.fingerprint, plan=changed)
        wrong_identity = (0, 0, self.source.stat().st_size, 0, 0)
        with self.assertRaisesRegex(ValueError, "inode differs"):
            ArtifactCache(self.root / "wrong-inode", self.source, fingerprint=self.fingerprint,
                          source_identity=wrong_identity)

    def test_unrecognized_directory_is_untouched(self):
        output = self.root / "unrelated"
        output.mkdir(mode=0o755)
        (output / "user.txt").write_text("keep this")
        before = output.stat().st_mode & 0o777
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.cache(output)
        self.assertEqual(output.stat().st_mode & 0o777, before)
        self.assertEqual(list(output.iterdir()), [output / "user.txt"])

    def test_source_replacement_detected_during_session(self):
        cache, _, _ = self.saved_fixture()
        replacement = self.root / "replacement.gguf"
        replacement.write_bytes(self.source.read_bytes())
        replacement.replace(self.source)
        with self.assertRaisesRegex(ValueError, "Source GGUF changed"):
            cache.get("a.weight", 0, 2, "int8", 2)

    def test_old_helper_capabilities_fail_before_compilation(self):
        cache = self.cache()
        weights = test_npu_protocol.FakeWeights()
        weights.path = self.source
        with patch.object(weights, "numpy_rows", side_effect=AssertionError("Must not compile")):
            with self.assertRaisesRegex(RuntimeError, "D/L capabilities"):
                NPULinear(weights, command=[sys.executable, test_npu_protocol.__file__, "--worker", "old-helper"],
                          artifact_cache=cache, min_free_mib=0)

    def test_cli_plan_only_low_disk_is_readonly(self):
        import compile_gguf
        from create_tiny_gguf import create
        source = self.root / "tiny.gguf"
        create(source)
        output = self.root / "new-plan"
        argv = ["compile_gguf.py", "--model", str(source), "--output", str(output), "--plan-only"]
        captured = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(captured), \
                patch("compile_gguf.shutil.disk_usage", return_value=SimpleNamespace(free=0)), \
                patch("compile_gguf.NPULinear", side_effect=AssertionError("Plan must not start native helper")):
            self.assertEqual(compile_gguf.main(), 0)
        self.assertIn('"would_fit": false', captured.getvalue())
        self.assertFalse(output.exists())

    def test_cli_plan_only_unsupported_tensors_are_reported(self):
        import compile_gguf
        from create_tiny_gguf import create
        source = self.root / "unsupported.gguf"
        create(source)
        output = self.root / "unsupported-plan"
        captured = io.StringIO()
        with patch.object(sys, "argv", ["compile_gguf.py", "--model", str(source), "--output", str(output), "--plan-only"]), \
                contextlib.redirect_stdout(captured), \
                patch.object(compile_gguf.GGUFWeights, "decode_supported_types", new=property(lambda _: set())), \
                patch("compile_gguf.NPULinear", side_effect=AssertionError("Plan must not start native helper")):
            self.assertEqual(compile_gguf.main(), 0)
        self.assertIn('"tensor_compile_supported": false', captured.getvalue())
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
