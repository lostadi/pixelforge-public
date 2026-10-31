"""Compiler completion validates full tile coverage with unique code reads."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import compiled_model
from compile_grouped_columns import mark_columns_complete
from gguf_hf_loader import GGUFWeights
from grouped_columns import ColumnProgramCache
from model_architecture import build_plan
from test_gguf_loader import write_gguf


class ColumnCompletionTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory(prefix="pixelforge-column-completion-")
        self.addCleanup(work.cleanup)
        self.root = Path(work.name)
        self.source = self.root / "original.gguf"
        tensors = []
        for name, rows, columns, kind in (
                ("a.weight", 3, 4096, 13), ("b.weight", 3, 4096, 13),
                ("c.weight", 2, 3072, 14), ("d.weight", 2, 1024, 13)):
            block_bytes = 176 if kind == 13 else 210
            tensors.append((name, (rows, columns), kind,
                            bytes(rows * (columns // 256) * block_bytes)))
        write_gguf(self.source, tensors)
        weights = GGUFWeights(self.source)
        self.addCleanup(weights.close)
        digest = weights.sha256()
        plan = build_plan(weights, model_sha256=digest, bank_rows=2)
        self.cache = ColumnProgramCache(
            self.root / "package", self.source, source_sha256=digest,
            source_identity=weights.source_identity, verify_source=False,
            plan=plan, fingerprint={"device": "completion-fixture"}, batch_vectors=4)
        self.tiles = []
        self.by_key = {}
        for name in self.cache.layout.aliases:
            for alias in self.cache.aliases_for(name):
                columns = self.cache.layer(alias)["columns"]
                for start, stop in self.cache.ranges(alias):
                    tile = (alias, start, stop, self.cache.request_dtype, columns)
                    self.tiles.append(tile)
                    key, _ = self.cache.identity(*tile)
                    self.by_key.setdefault(key, tile)
        for index, tile in enumerate(self.by_key.values()):
            code = bytes([8, 0, 0, 0]) + b"DGC0" + bytes([index]) * 256
            record = {"graph_sha256": hashlib.sha256(b"fixture" + bytes([index])).hexdigest(),
                      "graph_bytes": 64, "compiled_bytes": len(code)}
            self.cache.store(*tile, record, code)
        self.assertEqual(len(self.tiles), 10)
        self.assertEqual(len(self.by_key), 4)

    def manifest(self):
        return json.loads(self.cache.manifest_path.read_text())

    def records(self):
        for entry in self.manifest()["artifacts"].values():
            path = self.cache.directory / entry["metadata"]
            yield path, json.loads(path.read_text())

    def assert_not_complete(self):
        self.assertFalse(self.manifest()["complete"])

    def test_publication_matches_existing_full_coverage_method(self):
        original = self.manifest()
        expected = copy.deepcopy(original)
        actual = copy.deepcopy(original)
        expected["completed_weight_dtypes"] = ["float32"]
        actual["completed_weight_dtypes"] = ["float32"]
        with patch.object(self.cache, "_update", side_effect=lambda change: change(expected)):
            self.cache.mark_complete()
        with patch.object(self.cache, "_update", side_effect=lambda change: change(actual)):
            mark_columns_complete(self.cache)
        self.assertEqual(actual, expected)
        self.assertTrue(actual["complete"])
        self.assertEqual(actual["completed_weight_dtypes"], ["float32", "original_groups"])
        self.assertEqual(self.manifest(), original)

    def test_every_tile_identity_is_validated_and_each_code_is_hashed_once(self):
        identity = self.cache.identity
        identity_calls = []
        code_calls = []
        hashed = compiled_model.file_sha256

        def record_identity(*tile):
            identity_calls.append(tile)
            return identity(*tile)

        def record_hash(path):
            if Path(path).suffix == ".dgc":
                code_calls.append(Path(path))
            return hashed(path)

        with patch.object(self.cache, "identity", side_effect=record_identity), \
                patch.object(compiled_model, "file_sha256", side_effect=record_hash):
            mark_columns_complete(self.cache)
        self.assertTrue(set(self.tiles).issubset(identity_calls))
        self.assertEqual(len(code_calls), len(self.by_key))
        self.assertEqual(len(set(code_calls)), len(self.by_key))
        self.assertTrue(self.manifest()["complete"])

    def test_empty_plan_and_unknown_alias_never_publish_completion(self):
        aliases = self.cache.layout.aliases
        with patch.object(self.cache.layout, "aliases", {}):
            with self.assertRaisesRegex(ValueError, "empty"):
                mark_columns_complete(self.cache)
        aliases["a.weight"].append("outside.weight")
        with self.assertRaisesRegex(ValueError, "Unknown|noncanonical"):
            mark_columns_complete(self.cache)
        self.assert_not_complete()

    def test_missing_geometry_never_publishes_completion(self):
        manifest = self.manifest()
        manifest["artifacts"].pop(next(iter(manifest["artifacts"])))
        self.cache.manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            mark_columns_complete(self.cache)
        self.assert_not_complete()

    def test_missing_bytecode_never_publishes_completion(self):
        _, record = next(self.records())
        (self.cache.directory / record["bytecode_path"]).unlink()
        with self.assertRaises(FileNotFoundError):
            mark_columns_complete(self.cache)
        self.assert_not_complete()

    def test_corrupt_code_is_rechecked_even_when_manifest_was_complete(self):
        mark_columns_complete(self.cache)
        _, record = next(self.records())
        code_path = self.cache.directory / record["bytecode_path"]
        data = bytearray(code_path.read_bytes())
        data[-1] ^= 1
        code_path.write_bytes(data)
        with patch.object(self.cache, "_update") as publish:
            with self.assertRaisesRegex(ValueError, "hash/identifier"):
                mark_columns_complete(self.cache)
        publish.assert_not_called()

    def test_source_replacement_is_rejected_before_completion(self):
        temporary = self.root / "replacement.gguf"
        temporary.write_bytes(self.source.read_bytes())
        temporary.replace(self.source)
        with self.assertRaisesRegex(ValueError, "Source GGUF changed"):
            mark_columns_complete(self.cache)
        self.assert_not_complete()

    def test_changed_plan_and_fingerprint_are_rejected(self):
        original = self.manifest()
        for field in ("plan", "fingerprint"):
            with self.subTest(field=field):
                manifest = copy.deepcopy(original)
                if field == "plan":
                    manifest["plan"]["dense_tensors"][0]["columns"] += 256
                    message = "architecture plan"
                else:
                    manifest["device_fingerprint"]["device"] = "changed"
                    message = "fingerprint"
                self.cache.manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, message):
                    mark_columns_complete(self.cache)
                self.assert_not_complete()
        self.cache.manifest_path.write_text(json.dumps(original))

    def test_metadata_tamper_and_missing_metadata_are_rejected(self):
        metadata_path, _ = next(self.records())
        original = metadata_path.read_bytes()
        metadata_path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "metadata hash"):
            mark_columns_complete(self.cache)
        metadata_path.unlink()
        with self.assertRaises(FileNotFoundError):
            mark_columns_complete(self.cache)
        self.assert_not_complete()

    def test_wrong_dtype_and_conflicting_geometry_identity_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "geometry"):
            mark_columns_complete(self.cache, "float32")
        identity = self.cache.identity
        def collide(*tile):
            key, fields = identity(*tile)
            # This canonical split tail shares the first full tile's geometry.
            # Reject inconsistent fields even if a faulty identity function
            # returns the same key.
            if tile[0] == "a.weight::cols[3072:4096]" and tile[1] == 0:
                fields = dict(fields, original_group_size=999)
            return key, fields

        with patch.object(self.cache, "identity", side_effect=collide):
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                mark_columns_complete(self.cache)
        self.assert_not_complete()


if __name__ == "__main__":
    unittest.main()
