"""Bounded CPU-only artifact stress: no NPU, Android, or vendor API calls.

Run with the generic offline Python environment. Native-client cases launch
only this repository's NumPy protocol stub; their counters are fake. Fixtures
are private, capped at 16 MiB, and removed after every case. Publication fault
tests inject exceptions, not a power loss or filesystem durability failure.
"""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import threading
import time
from unittest.mock import patch

import compiled_model
from compiled_model import ArtifactCache, canonical

LIMIT = 16 * 1024 ** 2
FINGERPRINT = {"compiler_library_sha256": "a" * 64,
               "firmware_version": "CPU stress fixture; no vendor calls",
               "native_protocol_abi": "pixelforge-pipe-DL-v1"}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def rejected(action, classes=(ValueError, OSError), contains=None):
    try:
        action()
    except classes as error:
        if contains is not None:
            require(contains in str(error), f"Wrong rejection: {error}")
        return str(error)
    raise AssertionError("Operation unexpectedly accepted an invalid artifact")


def code_record(seed, size=1024):
    random_bytes = random.Random(seed).randbytes(size - 8)
    code = b"\x08\0\0\0DGC0" + random_bytes
    record = {"graph_sha256": hashlib.sha256(b"authored fixture" + random_bytes).hexdigest(),
              "graph_bytes": 624, "compiled_bytes": len(code)}
    return record, code


class Fixture:
    def __init__(self, root, seed, tensors=None):
        self.root, self.seed = root, seed
        self.source = root / "source.gguf"
        self.source.write_bytes(b"Not a parsed GGUF; artifact identity fixture\0" + random.Random(seed).randbytes(96))
        self.digest = compiled_model.file_sha256(self.source)
        self.plan = {"architecture": "fixture", "model_sha256": self.digest,
                     "unsupported_tensor_operations": [],
                     "dense_tensors": tensors or [{"name": "a.weight", "rows": 2,
                                                   "columns": 2, "bank_rows": 2}]}
        self.output = root / "package"

    def cache(self, *, read_only=False, source=True, directory=None):
        return ArtifactCache(directory or self.output, self.source if source else None,
                             fingerprint=FINGERPRINT, plan=self.plan, read_only=read_only)

    def put(self, cache, seed=None, name="a.weight", start=0, stop=2, columns=2):
        record, code = code_record(self.seed if seed is None else seed)
        stored = cache.store(name, start, stop, "int8", columns, record, code)
        return stored, code


def initialization_case(fixture, rng):
    workers = rng.choice((4, 6, 8))
    barrier = threading.Barrier(workers)
    def initialize(_):
        barrier.wait(timeout=10)
        cache = fixture.cache()
        return cache.source_sha256
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        identities = list(pool.map(initialize, range(workers)))
    require(identities == [fixture.digest] * workers, "Concurrent creators mixed source identities")
    cache = fixture.cache(read_only=True)
    require(cache._manifest()["artifacts"] == {}, "Fresh package unexpectedly contains artifacts")
    require(fixture.output.stat().st_mode & 0o777 == 0o700, "Package permissions are not private")
    return {"creators": workers}


def publication_case(fixture, rng):
    cache = fixture.cache()
    workers = 8
    identical_key = bool(rng.getrandbits(1))
    barrier = threading.Barrier(workers)
    done = threading.Event()
    allowed = {code_record(fixture.seed + i)[1]: code_record(fixture.seed + i)[0]["graph_sha256"]
               for i in range(workers)}
    def publish(index):
        barrier.wait(timeout=10)
        local = fixture.cache()
        name = "a.weight" if identical_key else f"m{index}.weight"
        fixture.put(local, fixture.seed + index, name=name)
    def read_while_publishing():
        reads = 0
        while not done.is_set():
            manifest = cache._manifest()
            for entry in manifest["artifacts"].values():
                raw = json.loads((cache.directory / entry["metadata"]).read_text())
                record, code = cache.get(raw["tensor"], 0, 2, "int8", 2)
                require(code in allowed and record["graph_sha256"] == allowed[code],
                        "Reader observed mixed graph metadata/bytecode")
                reads += 1
            done.wait(.001)
        return reads
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers + 1) as pool:
        reader = pool.submit(read_while_publishing)
        jobs = [pool.submit(publish, i) for i in range(workers)]
        try:
            for job in jobs:
                job.result(timeout=15)
        finally:
            done.set()
        reads = reader.result(timeout=15)
    manifest = cache._manifest()
    require(len(manifest["artifacts"]) == (1 if identical_key else workers), "Manifest update was lost")
    for entry in manifest["artifacts"].values():
        raw = json.loads((cache.directory / entry["metadata"]).read_text())
        record, code = cache.get(raw["tensor"], 0, 2, "int8", 2)
        require(code in allowed and record["graph_sha256"] == allowed[code], "Published bank is incoherent")
    require(not list(cache.directory.rglob(".pending-*")), "Atomic temporary files leaked")
    return {"writers": workers, "same_key": identical_key, "concurrent_verified_reads": reads}


def rewrite_metadata(cache, change):
    manifest = cache._manifest()
    key = next(iter(manifest["artifacts"]))
    old = manifest["artifacts"][key]
    record = json.loads((cache.directory / old["metadata"]).read_text())
    change(record)
    data = canonical(record) + b"\n"
    digest = hashlib.sha256(data).hexdigest()
    relative = f"graphs/{key}-{digest}.json"
    (cache.directory / relative).write_bytes(data)
    manifest["artifacts"][key] = {"metadata": relative, "metadata_sha256": digest}
    cache.manifest_path.write_bytes(canonical(manifest))


def tamper_case(fixture, rng, index):
    cache = fixture.cache()
    stored, code = fixture.put(cache)
    kind = index % 12
    action = lambda: cache.get("a.weight", 0, 2, "int8", 2)
    code_path = cache.directory / stored["bytecode_path"]
    if kind == 0:
        altered = bytearray(code); altered[rng.randrange(8, len(code))] ^= 1
        code_path.write_bytes(altered)
    elif kind == 1:
        code_path.write_bytes(code[:-1])
    elif kind == 2:
        entry = next(iter(cache._manifest()["artifacts"].values()))
        (cache.directory / entry["metadata"]).write_text("{}")
    elif kind == 3:
        rewrite_metadata(cache, lambda value: value.update(tensor="different.weight"))
    elif kind == 4:
        rewrite_metadata(cache, lambda value: value.update(input_bytes=999))
    elif kind == 5:
        rewrite_metadata(cache, lambda value: value.update(graph_sha256="invalid"))
    elif kind == 6:
        rewrite_metadata(cache, lambda value: value.update(bytecode_path="../outside.dgc"))
    elif kind == 7:
        manifest = cache._manifest(); manifest["source_gguf"]["sha256"] = "b" * 64
        cache.manifest_path.write_bytes(canonical(manifest))
    elif kind == 8:
        manifest = cache._manifest(); manifest["device_fingerprint"] = {"changed": True}
        cache.manifest_path.write_bytes(canonical(manifest))
    elif kind == 9:
        manifest = cache._manifest(); manifest["plan"]["dense_tensors"][0]["bank_rows"] = 1
        cache.manifest_path.write_bytes(canonical(manifest))
    elif kind == 10:
        target = fixture.root / "outside.dgc"; target.write_bytes(code)
        code_path.unlink(); code_path.symlink_to(target)
    else:
        manifest = cache._manifest(); manifest["source_gguf"].update(embedded=True, path="../source.gguf")
        cache.manifest_path.write_bytes(canonical(manifest))
        action = lambda: fixture.cache(read_only=True, source=False)
    error = rejected(action)
    return {"tamper_kind": kind, "rejection": error}


def atomic_fault_case(fixture, rng, index):
    if index % 4 == 0:
        original = compiled_model.os.replace
        def fail_identity(source, destination):
            if Path(destination).name == "MODEL.json":
                raise OSError("injected initial identity publication failure")
            return original(source, destination)
        with patch("compiled_model.os.replace", side_effect=fail_identity):
            rejected(fixture.cache, classes=(OSError,), contains="initial identity")
        require(not list(fixture.output.iterdir()), "Initial failure left an unretryable partial package")
        cache = fixture.cache()
        require(cache.source_sha256 == fixture.digest, "Initial retry changed source identity")
        return {"fault_stage": "initial MODEL.json", "retry_ok": True}
    cache = fixture.cache()
    _, original_code = fixture.put(cache)
    original_manifest = cache.manifest_path.read_bytes()
    original = compiled_model.os.replace
    stage = ("bytecode", "metadata", "manifest")[(index % 4) - 1]
    def fail_selected(source, destination):
        path = Path(destination)
        chosen = (stage == "bytecode" and path.suffix == ".dgc" or
                  stage == "metadata" and path.suffix == ".json" and path.name != "MODEL.json" or
                  stage == "manifest" and path.name == "MODEL.json")
        if chosen:
            raise OSError("injected " + stage + " publication failure")
        return original(source, destination)
    with patch("compiled_model.os.replace", side_effect=fail_selected):
        rejected(lambda: fixture.put(cache, fixture.seed + 100), classes=(OSError,), contains=stage)
    require(cache.manifest_path.read_bytes() == original_manifest, "Failed publication changed committed manifest")
    require(cache.get("a.weight", 0, 2, "int8", 2)[1] == original_code, "Prior committed bank was damaged")
    require(not list(cache.directory.rglob(".pending-*")), "Failed publication leaked temporary files")
    fixture.put(cache, fixture.seed + 100)
    require(cache.get("a.weight", 0, 2, "int8", 2)[1] != original_code, "Publication retry did not replace bank")
    return {"fault_stage": stage, "retry_ok": True}


def resume_case(fixture, rng):
    rows = rng.randint(3, 9)
    fixture.plan["dense_tensors"][0].update(rows=rows, bank_rows=2)
    cache = fixture.cache()
    expected = {}
    expected[0] = fixture.put(cache, fixture.seed, stop=2)[1]
    rejected(lambda: cache.mark_complete(), contains="incomplete")
    for start in range(2, rows, 2):
        expected[start] = fixture.put(cache, fixture.seed + start, start=start, stop=min(start + 2, rows))[1]
    resumed = fixture.cache()
    resumed.mark_complete()
    require(resumed._manifest()["completed_weight_dtypes"] == ["int8"], "Completed formats were misreported")
    if rng.getrandbits(1):
        resumed.embed_source()
        fixture.source.unlink()
        relocated = fixture.root / "relocated"
        shutil.copytree(fixture.output, relocated)
        readonly = fixture.cache(read_only=True, source=False, directory=relocated)
    else:
        readonly = fixture.cache(read_only=True)
    before = readonly.manifest_path.read_bytes()
    for start, expected_code in expected.items():
        require(readonly.get("a.weight", start, min(start + 2, rows), "int8", 2)[1] == expected_code,
                "Resumed bank changed")
    rejected(lambda: readonly.get("a.weight", 0, 2, "float32", 2), classes=(FileNotFoundError,))
    rejected(lambda: readonly.store("a.weight", 0, 2, "int8", 2, *code_record(fixture.seed)),
             classes=(PermissionError,))
    require(readonly.manifest_path.read_bytes() == before, "Read-only package mutated")
    return {"banks": len(expected), "embedded_relocated": readonly.directory != fixture.output}


def source_identity_case(fixture, rng, index):
    cache = fixture.cache()
    fixture.put(cache)
    if index % 2:
        replacement = fixture.root / "replacement.gguf"
        replacement.write_bytes(fixture.source.read_bytes())
        replacement.replace(fixture.source)
    else:
        with fixture.source.open("r+b") as source:
            source.seek(rng.randrange(fixture.source.stat().st_size))
            source.write(b"X")
            source.flush(); os.fsync(source.fileno())
    error = rejected(lambda: cache.get("a.weight", 0, 2, "int8", 2), contains="Source GGUF changed")
    return {"mutation": "path replacement" if index % 2 else "in-place change", "rejection": error}


def native_stub_case(fixture, rng):
    import numpy as np
    from npu_linear import NPULinear
    import test_npu_protocol
    weights = test_npu_protocol.FakeWeights()
    weights.path = fixture.source
    random_state = np.random.default_rng(fixture.seed)
    for name in ("a.weight", "b.weight"):
        weights.matrices[name] = random_state.normal(0, .5, (2, 2)).astype(np.float32)
    activation = random_state.normal(0, 5000, (1, 2)).astype(np.float32)
    command = [sys.executable, str(Path(test_npu_protocol.__file__).resolve()), "--worker", "normal"]
    cache = fixture.cache()
    helpers = []
    with patch.dict(os.environ, {"PIXELFORGE_NPU_STDERR": str(fixture.root / "native-stub.stderr")}), \
            contextlib.redirect_stdout(io.StringIO()):
        try:
            first = NPULinear(weights, command=command, artifact_cache=cache, cache_mib=1, min_free_mib=0)
            helpers.append(first)
            expected = []
            for name in ("a.weight", "b.weight", "a.weight"):
                matrix = weights.matrices[name]
                scales = np.maximum(np.max(np.abs(matrix), axis=1) / 127, 1e-12)
                quantized = np.rint(matrix / scales[:, None]) * scales[:, None]
                reference = activation @ quantized.T
                result = first(name, activation)
                np.testing.assert_allclose(result, reference, rtol=1e-5, atol=.002)
                expected.append(result.copy())
            first.close()
            require(first.stats()["compiled_models"] == 2 and first.stats()["loaded_models"] == 1,
                    "Stub first pass did not compile/export and reload after eviction")
            second = NPULinear(weights, command=command, artifact_cache=fixture.cache(read_only=True),
                               cache_mib=1, min_free_mib=0)
            helpers.append(second)
            with patch.object(weights, "numpy_rows", side_effect=AssertionError("Read-only runtime decoded weights")):
                for name, reference in zip(("a.weight", "b.weight", "a.weight"), expected):
                    np.testing.assert_array_equal(second(name, activation), reference)
            require(second.stats()["compiled_models"] == 0 and second.stats()["loaded_models"] == 3,
                    "Fresh read-only stub pass secretly compiled a bank")
            require(second.invocations == 3, "Fresh stub pass lost an invocation")
        finally:
            for helper in helpers:
                helper.close()
    require(all(h.process.poll() == 0 for h in helpers), "Owned CPU stub failed cleanup")
    return {"cpu_stub_processes": len(helpers), "fake_invocations_checked": 6,
            "fresh_readonly_compiles": 0, "fresh_readonly_loads": 3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=741813)
    parser.add_argument("--report", type=Path, default=Path("/tmp/pixelforge-stress-artifacts.json"))
    args = parser.parse_args()
    os.umask(0o077)
    report_fd = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    before = time.monotonic()
    cases, peak_bytes, cleanup_ok = [], 0, True
    schedule = [("concurrent_initialization", initialization_case, 24),
                ("concurrent_publication", publication_case, 24),
                ("tamper_rejection", tamper_case, 24),
                ("atomic_fault_retry", atomic_fault_case, 16),
                ("resume_readonly_relocation", resume_case, 16),
                ("source_identity", source_identity_case, 8),
                ("native_cpu_stub", native_stub_case, 8)]
    with patch("compiled_model.device_fingerprint", side_effect=AssertionError("Vendor access forbidden in CPU stress")):
        for category, action, count in schedule:
            for index in range(count):
                seed = args.seed + len(cases)
                started = time.monotonic()
                result = {"category": category, "index": index, "seed": seed}
                temporary = tempfile.TemporaryDirectory(prefix="pixelforge-artifact-stress-")
                root = Path(temporary.name)
                try:
                    fixture = Fixture(root, seed)
                    rng = random.Random(seed)
                    result["observations"] = (action(fixture, rng, index) if action in
                        (tamper_case, atomic_fault_case, source_identity_case) else action(fixture, rng))
                    fixture_bytes = sum(p.stat().st_size for p in root.rglob("*") if p.is_file() and not p.is_symlink())
                    peak_bytes = max(peak_bytes, fixture_bytes)
                    require(fixture_bytes < LIMIT, "Fixture exceeded 16 MiB storage cap")
                    result["success"] = True
                except Exception as error:
                    result.update(success=False, error_type=type(error).__name__, error=str(error))
                finally:
                    temporary.cleanup()
                    cleaned = not root.exists()
                    cleanup_ok = cleanup_ok and cleaned
                    result.update(seconds=time.monotonic() - started, fixture_cleanup_ok=cleaned)
                cases.append(result)
    failures = [case for case in cases if not case["success"]]
    report = {"schema_version": 1, "seed": args.seed, "cases": len(cases),
              "passed": len(cases) - len(failures), "failed": len(failures),
              "category_counts": dict(Counter(case["category"] for case in cases)),
              "seconds": time.monotonic() - before, "hardware_api_calls": 0,
              "native_transport": "Explicit NumPy-only fake helper, never Android/vendor helper",
              "largest_case_fixture_bytes_after_operations": peak_bytes,
              "fixture_cap_bytes": LIMIT, "all_fixtures_removed": cleanup_ok,
              "limitations": ["Injected publication exceptions do not simulate power loss or storage durability.",
                              "Synthetic DGC0 fixtures validate storage semantics; they are not executable NPU code.",
                              "Native-client numerical checks use CPU stubs; no hardware inference is claimed."],
              "fixed_regression": {"trigger": "First MODEL.json rename raises OSError",
                    "before": "Left .lock/graphs and refused clean initialization retry",
                    "after": "Commit initial identity under directory flock before creating auxiliaries",
                    "source_sha256": compiled_model.file_sha256(Path(compiled_model.__file__))},
              "failures": failures, "case_results": cases}
    with os.fdopen(report_fd, "w") as destination:
        json.dump(report, destination, indent=2); destination.write("\n")
    print(json.dumps({key: report[key] for key in ("seed", "cases", "passed", "failed", "seconds",
          "hardware_api_calls", "largest_case_fixture_bytes_after_operations", "all_fixtures_removed")}))
    return 0 if not failures and cleanup_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
