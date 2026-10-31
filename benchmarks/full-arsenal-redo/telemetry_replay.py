#!/usr/bin/env python3
"""Replay a complete broker invocation history to compare diagnostic retention.

This measures host allocations and serialized detail size, never model latency,
NPU performance, or numerical quality. Inputs stay private: emitted results
contain aggregate counts, sizes and invariant checks, without event bodies,
tensor names, prompts, report paths or workspace paths.

Example:
    python telemetry_replay.py --source-root . --report BROKER.json --output AGGREGATE.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tracemalloc


SCHEMA = "pixelforge.bounded-host-telemetry-replay.v1"
STRESS_EVENTS = 200000
TRACE_LIMIT = 1024


def positive_integer(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("Use a positive event count")
    return result


def load_runtime(source_root):
    """Import the selected checkout's actual retention implementation."""
    module_path = source_root.resolve(strict=True) / "tools/arm_npu/runtime_telemetry.py"
    spec = importlib.util.spec_from_file_location("pixelforge_replay_runtime_telemetry", module_path)
    if spec is None or spec.loader is None:
        raise ValueError("The selected source root has no runtime telemetry module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TraceTail, hashlib.sha256(module_path.read_bytes()).hexdigest()


def load_rows(report_path):
    """Require a complete history; a bounded tail cannot reconstruct dropped rows."""
    value = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("npu"), dict):
        raise ValueError("A broker report with NPU evidence is required")
    native = value["npu"]
    rows = native.get("invocations")
    operations = native.get("npu_invocations")
    if (not isinstance(rows, list) or not rows or type(operations) is not int
            or operations != len(rows)):
        raise ValueError("A complete invocation detail history is required")
    evidence = native.get("invocation_trace")
    if evidence is not None and (not isinstance(evidence, dict)
                                 or evidence.get("dropped") != 0):
        raise ValueError("A complete invocation detail history is required")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each invocation detail must be an object")
        seconds = row.get("seconds")
        if (type(seconds) not in (int, float) or not math.isfinite(seconds)
                or seconds < 0):
            raise ValueError("Invocation duration evidence must be finite and nonnegative")
        if "prefill" in row and type(row["prefill"]) is not bool:
            raise ValueError("Invocation phase evidence must be boolean")
    return rows, operations


def replay(rows, count, tail_class=None):
    """Measure fresh containers/dictionaries; preloaded field objects are shared."""
    gc.collect()
    tracemalloc.start()
    try:
        trace = tail_class(TRACE_LIMIT) if tail_class is not None else []
        operations = 0
        recorded_seconds = 0.0
        phases = Counter()
        for index in range(count):
            # Fresh dictionary per event, as in the owning Python runtime.
            # Both cases share the preloaded field objects, which are outside
            # this allocation measurement. This is not a whole-process RSS bound.
            row = dict(rows[index % len(rows)])
            trace.append(row)
            operations += 1
            recorded_seconds += row["seconds"]
            phase = "unknown" if "prefill" not in row else "P" if row["prefill"] else "D"
            phases[phase] += 1
        if not math.isfinite(recorded_seconds):
            raise ValueError("Recorded duration sum exceeds finite host representation")
        retained_current, retained_peak = tracemalloc.get_traced_memory()
        detail = trace.snapshot() if tail_class is not None else trace
        payload = json.dumps(detail, separators=(",", ":")).encode("utf-8")
        _, serialization_peak = tracemalloc.get_traced_memory()
        result = {
            "rows_processed": operations,
            "rows_retained": len(detail),
            "rows_dropped": operations - len(detail),
            "retained_traced_bytes": retained_current,
            "append_peak_traced_bytes": retained_peak,
            "serialization_peak_traced_bytes": serialization_peak,
            "serialized_detail_bytes": len(payload),
            "recorded_invoke_seconds_sum": recorded_seconds,
            "phase_calls": dict(sorted(phases.items())),
        }
        if tail_class is not None:
            evidence = trace.evidence()
            if (len(detail) != min(count, TRACE_LIMIT) or trace.total != count
                    or evidence["total"] != count
                    or evidence["retained"] != len(detail)
                    or evidence["limit"] != TRACE_LIMIT
                    or evidence["dropped"] != max(0, count - TRACE_LIMIT)
                    or evidence["policy"] != "bounded-most-recent"
                    or detail[-1] != rows[(count - 1) % len(rows)]):
                raise ValueError("Candidate retention evidence failed its invariants")
            result["retention_evidence"] = evidence
        return result
    finally:
        tracemalloc.stop()


def measure(source_root, report_path, stress_events=STRESS_EVENTS):
    tail_class, runtime_hash = load_runtime(source_root)
    rows, original_count = load_rows(report_path)
    cases = {}
    for count in dict.fromkeys((len(rows), stress_events)):
        baseline = replay(rows, count)
        candidate = replay(rows, count, tail_class)
        for field in ("rows_processed", "recorded_invoke_seconds_sum", "phase_calls"):
            if baseline[field] != candidate[field]:
                raise ValueError("Diagnostic retention changed an exact aggregate")
        cases[str(count)] = {"baseline_list": baseline, "candidate_tail": candidate}
    return {
        "schema": SCHEMA,
        "scope": "Real-history diagnostic replay and counter stress; no model/helper execution or latency benchmark.",
        "memory_boundary": "tracemalloc includes new containers and per-event dictionaries; preloaded shared field objects are excluded in both cases. This is not whole-process RSS.",
        "duration_boundary": "Recorded invocation durations are summed only to check aggregate equivalence; no execution speed is measured.",
        "source_native_invocations": original_count,
        "source_rows": len(rows),
        "runtime_telemetry_sha256": runtime_hash,
        "limit": TRACE_LIMIT,
        "stress_events": stress_events,
        "cases": cases,
        "all_exact_aggregate_invariants_passed": True,
        "residual_limit": "Compile/load graph records remain a complete history for existing validation consumers; only invocation and eviction detail tails are bounded.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="Explicit PixelForge source checkout containing runtime_telemetry.py")
    parser.add_argument("--report", type=Path, required=True,
                        help="Private broker JSON with a complete invocation detail history")
    parser.add_argument("--output", type=Path, required=True,
                        help="New aggregate-only JSON destination; an existing file is never replaced")
    parser.add_argument("--stress-events", type=positive_integer, default=STRESS_EVENTS,
                        help="Counter stress length; default 200000, use a small count for a correctness smoke")
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("Aggregate output already exists")
        result = measure(args.source_root, args.report, args.stress_events)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(result, output, indent=2, sort_keys=True)
            output.write("\n")
        print(json.dumps({"ok": True, "source_rows": result["source_rows"],
                          "stress_events": result["stress_events"],
                          "all_exact_aggregate_invariants_passed": True}))
    except (OSError, ValueError, TypeError, KeyError, ImportError, AttributeError, OverflowError) as error:
        # Exception details can contain private paths or raw input values.
        parser.exit(2, "Telemetry replay failed (" + type(error).__name__
                    + "); check source root, complete report and new output destination.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
