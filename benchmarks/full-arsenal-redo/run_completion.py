"""Synthetic source-only comparison of existing column-completion validators.

The dimensions and code sizes match an observed 281-tensor plan. The source is
a marker file, and saved code is deterministic DGC0-shaped test data, never an
executable NPU artifact. No helper, device probe, inference or compiler is used.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "snapshot"))
import compiled_model
from grouped_columns import ColumnProgramCache
from candidate_impl import mark_columns_complete


def fixture(root):
    source = root / "source.bin"
    source.write_bytes(b"completion-benchmark-source-v1\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    shape = json.loads((ROOT / "dimensions.json").read_text())
    tensors = [dict(name="tensor_" + str(index) + ".weight", rows=rows,
                    columns=columns, bank_rows=bank, type_id=kind)
               for index, (rows, columns, bank, kind)
               in enumerate(shape["dense_dimensions"])]
    plan = dict(model_sha256=digest, dense_tensors=tensors,
                unsupported_tensor_operations=[])
    cache = ColumnProgramCache(root / "package", source, plan=plan,
        fingerprint={"device": "completion-source-only-fixture"}, batch_vectors=64)
    sizes = {(p["kind"], p["rows"], p["columns"]): p["bytes"]
             for p in shape["program_sizes"]}
    logical = []
    representatives = {}
    for name in cache.layout.aliases:
        for alias in cache.aliases_for(name):
            columns = cache.layer(alias)["columns"]
            for start, stop in cache.ranges(alias):
                tile = (alias, start, stop, cache.request_dtype, columns)
                logical.append(tile)
                key, fields = cache.identity(*tile)
                representatives.setdefault(key, (tile, fields))
    for index, (tile, fields) in enumerate(representatives.values()):
        length = sizes[(fields["source_quantization_type"], fields["rows"], fields["columns"])]
        code = bytes([8, 0, 0, 0]) + b"DGC0" + bytes([index + 1]) * (length - 8)
        record = {"graph_sha256": hashlib.sha256(b"synthetic-graph-" + bytes([index])).hexdigest(),
                  "graph_bytes": 64, "compiled_bytes": len(code)}
        cache.store(*tile, record, code)
    return cache, logical, representatives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("implementation", choices=("baseline", "candidate"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args()
    if not 1 <= args.repetitions <= 20:
        parser.error("repetitions must be 1..20")
    with tempfile.TemporaryDirectory(prefix="pixelforge-completion-compare-") as temporary:
        cache, tiles, geometries = fixture(Path(temporary))
        manifest_raw = cache.manifest_path.read_bytes()
        original = json.loads(manifest_raw)
        publications = []
        counts = collections.Counter()
        hashed = compiled_model.file_sha256
        get = cache.get
        identity = cache.identity

        def count_hash(path):
            path = Path(path)
            counts["file_hash_calls"] += 1
            counts["file_hashed_bytes"] += path.stat().st_size
            if path.suffix == ".dgc":
                counts["code_hash_calls"] += 1
                counts["code_hashed_bytes"] += path.stat().st_size
            return hashed(path)

        def count_get(*tile, **options):
            counts["get_calls"] += 1
            return get(*tile, **options)

        def count_identity(*tile):
            counts["identity_calls"] += 1
            return identity(*tile)

        def publish(change):
            value = copy.deepcopy(original)
            change(value)
            publications.append(dict(complete=value["complete"],
                completed_weight_dtypes=value["completed_weight_dtypes"]))

        with patch.object(compiled_model, "file_sha256", side_effect=count_hash), \
                patch.object(cache, "get", side_effect=count_get), \
                patch.object(cache, "identity", side_effect=count_identity), \
                patch.object(cache, "_update", side_effect=publish):
            for _ in range(args.repetitions):
                if args.implementation == "baseline":
                    cache.mark_complete()
                else:
                    mark_columns_complete(cache)
        if cache.manifest_path.read_bytes() != manifest_raw:
            raise AssertionError("Completion validation mutated the saved package")
        expected = dict(complete=True, completed_weight_dtypes=["original_groups"])
        if publications != [expected] * args.repetitions:
            raise AssertionError("Completion metadata differs")
        result = dict(schema="pixelforge.column-completion-functional.v1",
            logical_tiles=len(tiles), geometry_programs=len(geometries),
            repetitions=args.repetitions, publications=publications,
            saved_manifest_unchanged=True,
            dummy_source_sha256=cache.source_sha256,
            synthetic_code=True, native_process_started=False)
        if args.evidence_dir:
            args.evidence_dir.mkdir(parents=True, exist_ok=True)
            evidence = dict(schema="pixelforge.column-completion-counters.v1",
                implementation=args.implementation, counts=dict(counts),
                functional_result_sha256=hashlib.sha256(compiled_model.canonical(result)).hexdigest(),
                scope="Whole Python fixture process: import/setup plus repeated completion; no NPU inference or native compilation")
            destination = args.evidence_dir / (args.implementation + "-" + str(os.getpid()) + ".json")
            destination.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
