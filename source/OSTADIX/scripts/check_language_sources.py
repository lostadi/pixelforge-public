#!/usr/bin/env python3
"""Account for and parse every tracked .O/.oc source without executing programs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "ci/language-sources.json"
CATEGORIES = {"example", "benchmark", "fixture", "package-mirror", "platform-demo", "kernel-module", "runtime-module", "world-module", "historical-snapshot"}


class SourceError(ValueError):
    pass


def nonblank(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def tracked_files(root: Path) -> set[str]:
    result = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True)
    return set(result.stdout.decode("utf-8").split("\0")) - {""}


def below_root(root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or PurePosixPath(value).is_absolute() or ".." in PurePosixPath(value).parts or str(PurePosixPath(value)) != value:
        raise SourceError(f"invalid repository-relative path: {value!r}")
    path = root / value
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise SourceError(f"missing regular repository file: {value}")
    return path


def load_manifest(root: Path, tracked: set[str] | None = None) -> dict[str, dict]:
    tracked = tracked_files(root) if tracked is None else tracked
    data = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"schema", "groups", "expected_parse_failures", "mirrors"} or data["schema"] != "ostadix.language-sources/v1":
        raise SourceError("invalid language-source manifest schema/fields")
    if not isinstance(data["groups"], list) or not data["groups"] or not isinstance(data["mirrors"], dict):
        raise SourceError("groups must be a nonempty list and mirrors a path map")
    entries: dict[str, dict] = {}
    group_ids: set[str] = set()
    for group in data["groups"]:
        if not isinstance(group, dict) or set(group) != {"id", "category", "reason", "coverage", "sources"}:
            raise SourceError("invalid classification group fields")
        if not nonblank(group["id"]) or group["id"] in group_ids or not isinstance(group["category"], str) or group["category"] not in CATEGORIES or not nonblank(group["reason"]):
            raise SourceError("invalid or duplicate classification group")
        group_ids.add(group["id"])
        if not isinstance(group["coverage"], list) or not group["coverage"] or not isinstance(group["sources"], list) or not group["sources"] or not all(isinstance(path, str) for path in group["sources"]):
            raise SourceError(f"empty coverage/sources for {group['id']}")
        for coverage in group["coverage"]:
            if not isinstance(coverage, dict) or set(coverage) != {"path", "kind", "conditions"} or not isinstance(coverage["kind"], str) or coverage["kind"] not in {"runtime", "build", "contract", "manual", "historical"} or not nonblank(coverage["conditions"]):
                raise SourceError(f"invalid coverage for {group['id']}")
            below_root(root, coverage["path"])
            if coverage["path"] not in tracked:
                raise SourceError(f"untracked coverage reference: {coverage['path']}")
        if group["sources"] != sorted(set(group["sources"])):
            raise SourceError(f"sources must be sorted and unique: {group['id']}")
        for path in group["sources"]:
            below_root(root, path)
            if not path.endswith((".O", ".oc")) or path in entries:
                raise SourceError(f"invalid or duplicate source: {path}")
            entries[path] = {"group": group["id"], "category": group["category"], "reason": group["reason"], "coverage": group["coverage"]}
    actual = {p for p in tracked if p.endswith((".O", ".oc"))}
    if set(entries) != actual:
        raise SourceError(f"tracked source coverage mismatch; missing={sorted(actual - entries.keys())}; stale={sorted(entries.keys() - actual)}")
    negatives = data["expected_parse_failures"]
    if not isinstance(negatives, dict):
        raise SourceError("expected_parse_failures must be a path map")
    for path, expectation in negatives.items():
        if path not in entries or not isinstance(expectation, dict) or set(expectation) != {"reason", "diagnostic"} or not all(nonblank(v) for v in expectation.values()):
            raise SourceError(f"expected parse failure needs a source, reason, and diagnostic: {path}")
        entries[path]["expected_failure"] = expectation
    for mirror, original in data["mirrors"].items():
        if mirror not in entries or not isinstance(original, str) or original not in entries or entries[mirror]["category"] != "package-mirror":
            raise SourceError(f"invalid package mirror: {mirror}")
        if (root / mirror).read_bytes() != (root / original).read_bytes():
            raise SourceError(f"package mirror drift: {mirror} != {original}")
    if set(data["mirrors"]) != {p for p, entry in entries.items() if entry["category"] == "package-mirror"}:
        raise SourceError("every package mirror requires an original source")
    # Reuse the existing runtime manifest; it owns editions, authority/opt-in
    # requirements and semantic expectations. Do not copy those facts here.
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        import example_manifest
        examples = example_manifest.load_manifest(root)
    finally:
        sys.path.pop(0)
    declared_examples = {"examples/" + entry["path"] for entry in examples}
    if declared_examples != {p for p, entry in entries.items() if entry["category"] == "example"}:
        raise SourceError("example classifications disagree with examples/manifest.json")
    return dict(sorted(entries.items()))


def parse_source(root: Path, path: str, entry: dict, o_bin: Path, ocorec: Path, timeout: float) -> dict:
    result = {"path": path, **entry, "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest()}
    with tempfile.TemporaryDirectory(prefix="ostadix-source-parse-") as temporary:
        command = ([str(o_bin), "--check", "--json", str(root / path), str(root / "backends")] if path.endswith(".O") else [str(ocorec), str(root / path), "--emit", "ast", "--output", str(Path(temporary) / "source.ast")])
        try:
            completed = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, OSError) as error:
            return {**result, "status": "failed", "detail": str(error)}
        output = completed.stdout + completed.stderr
        expected = entry.get("expected_failure")
        if expected:
            ok = completed.returncode == 1 and expected["diagnostic"] in output
            detail = "expected rejection observed" if ok else "expected rejection did not match"
        else:
            ok = completed.returncode == 0
            detail = "parsed"
            if ok and path.endswith(".O"):
                try:
                    payload = json.loads(completed.stdout)
                    ok = payload.get("ok") is True and payload.get("stage") == "parse"
                except (ValueError, AttributeError):
                    ok = False
                if not ok:
                    detail = "missing parse-stage success record"
        return {**result, "status": "passed" if ok else "failed", "detail": detail, "exit_code": completed.returncode, "diagnostic": output if not ok or expected else ""}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "check"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--o-bin", type=Path)
    parser.add_argument("--ocorec", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--json-report", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.jobs < 1 or args.timeout <= 0:
            raise SourceError("jobs and timeout must be positive")
        entries = load_manifest(root)
        results = []
        binaries = {}
        if args.action == "check":
            o_bin = (args.o_bin or root / "target/debug/O").resolve()
            ocorec = (args.ocorec or root / "target/debug/ocorec").resolve()
            for binary in (o_bin, ocorec):
                if not binary.is_file():
                    raise SourceError(f"required compiler/interpreter missing: {binary}")
            binaries = {name: {"path": str(binary), "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()} for name, binary in (("O", o_bin), ("ocorec", ocorec))}
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                results = list(pool.map(lambda pair: parse_source(root, *pair, o_bin, ocorec, args.timeout), entries.items()))
        report = {"schema": "ostadix.language-source-check/v1", "action": args.action, "sources": len(entries), "O": sum(p.endswith(".O") for p in entries), "oc": sum(p.endswith(".oc") for p in entries), "executed_programs": 0, "binaries": binaries, "results": results}
        if args.json_report:
            args.json_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        failures = [r for r in results if r["status"] == "failed"]
        for failure in failures:
            print(f"FAIL {failure['path']}: {failure['detail']}\n{failure.get('diagnostic', '')}", file=sys.stderr)
        print(f"language-sources: {len(entries)} classified ({report['O']} .O, {report['oc']} .oc); {len(results) - len(failures)} parsed as expected; {len(failures)} failed; 0 programs executed")
        return int(bool(failures))
    except (SourceError, ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"language-sources: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
