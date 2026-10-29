#!/usr/bin/env python3
"""Bounded, serial hardware stress using an existing tiny compiled GGUF.

This never compiles cache misses or alters services. It tests saved-code loads,
LRU evictions, repeated model execution, process exit and numerical consistency.
Use a small, synthetic model with a vocabulary <=4096 for full-logit checks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from benchmark import hardware_snapshot, process_sample


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": fields[19], "ppid": int(fields[1]),
                "state": fields[0]}
    except (OSError, ValueError, IndexError):
        return None


def child_of(pid):
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            current = identity(int(entry.name))
            if current and current["ppid"] == pid:
                try:
                    if b"npu_pipe_gguf" in (entry / "cmdline").read_bytes():
                        return current
                except OSError:
                    pass
    return None


def confirmed_exit(observed):
    current = identity(observed["pid"])
    return current is None or current["start_ticks"] != observed["start_ticks"]


def stop_owned_tree(process, grace=60):
    """Bound cleanup, pinning only descendants observed before termination."""
    entries = {}
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            entry = identity(int(path.name))
            if entry:
                entries[entry["pid"]] = entry
    owned = {process.pid}
    while True:
        descendants = {pid for pid, entry in entries.items() if entry["ppid"] in owned}
        expanded = owned | descendants
        if expanded == owned:
            break
        owned = expanded
    pinned = []
    try:
        for pid in sorted(owned):
            observed = entries.get(pid)
            if observed is None:
                continue
            try:
                descriptor = os.pidfd_open(pid)
            except ProcessLookupError:
                continue
            current = identity(pid)
            if not current or current["start_ticks"] != observed["start_ticks"]:
                os.close(descriptor)
                continue
            pinned.append(descriptor)
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            process.poll()  # Reap our immediate child so its pidfd becomes ready.
            if all(select.select([fd], [], [], 0)[0] for fd in pinned):
                return False
            time.sleep(.02)
        for descriptor in pinned:
            if not select.select([descriptor], [], [], 0)[0]:
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        process.wait(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if all(select.select([fd], [], [], 0)[0] for fd in pinned):
                return True
            time.sleep(.02)
        raise RuntimeError("An owned stress process did not exit after SIGKILL")
    finally:
        for descriptor in pinned:
            os.close(descriptor)


def telemetry(snapshot):
    raw = snapshot["tpu_raw"]
    return {"utc": snapshot["utc"], "memory_kib": snapshot["memory_kib"],
            "battery": {key: snapshot["battery"].get(key) for key in
                        ("status", "current_ua", "voltage_uv", "temperature_c",
                         "net_battery_discharge_w", "interpretation")},
            "tpu": {key: raw.get(key) for key in
                    ("inference_count", "firmware_crash_count", "watchdog_timeout_count")}}


def run_case(arguments, output):
    observed_native = None
    peaks = {"native_rss_kib": 0, "native_fd_count": 0, "samples": 0}
    descriptor = None
    started = time.monotonic()
    with (output.parent / (output.name + ".launcher.log")).open("x") as log:
        process = subprocess.Popen([sys.executable, str(HERE / "run_gguf.py"),
                                    *arguments, "--output", str(output)],
                                   stdout=log, stderr=log, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started > 90:
                    raise TimeoutError("Stress case exceeded its outer 90s deadline")
                if observed_native is None and (output / "BROKER_OWNER.json").is_file():
                    try:
                        owner = json.loads((output / "BROKER_OWNER.json").read_text())
                        broker = identity(owner["pid"])
                        if broker and broker["start_ticks"] == owner["start_ticks"]:
                            observed_native = child_of(owner["pid"])
                            if observed_native:
                                descriptor = os.pidfd_open(observed_native["pid"])
                    except (OSError, ValueError):
                        pass
                if observed_native:
                    sample = process_sample(observed_native["pid"])
                    if sample.get("starttime_ticks") == int(observed_native["start_ticks"]):
                        peaks["samples"] += 1
                        rss = sample.get("VmRSS", "0 kB").split()
                        if rss and rss[0].isdecimal():
                            peaks["native_rss_kib"] = max(peaks["native_rss_kib"], int(rss[0]))
                        try:
                            count = len(list(Path(f"/proc/{observed_native['pid']}/fd").iterdir()))
                            peaks["native_fd_count"] = max(peaks["native_fd_count"], count)
                        except OSError:
                            pass
                time.sleep(.1)
            if process.returncode:
                raise RuntimeError(f"Stress case exited {process.returncode}; see {output}")
            report = json.loads((output / "RESULT.json").read_text())
            if not report["success"] or not report["engine_cleanup_ok"] or not report["broker_cleanup_ok"]:
                raise AssertionError("Run did not confirm success and owned cleanup")
            native_exited = None
            if observed_native:
                native_exited = bool(select.select([descriptor], [], [], 0)[0]) and confirmed_exit(observed_native)
                if not native_exited:
                    raise AssertionError("An observed owned native helper remained after the run")
            if report["backend"] != "cpu-reference":
                if native_exited is not True:
                    raise AssertionError("Native process exit was not independently observed")
                stats = report["broker"]["npu"]
                if stats["compiled_models"] != 0 or stats["npu_invocations"] < 1:
                    raise AssertionError("Read-only run compiled a graph or did not invoke the NPU")
                if not report["broker"].get("native_cleanup_ok") or stats["cpu_fallback"]:
                    raise AssertionError("Native cleanup failed or eligible operations fell back")
                clients = report["hardware_after"]["tpu_raw"].get("clients", "")
                if any(line.split()[0] == str(stats["native_pid"]) for line in clients.splitlines() if line.split()):
                    raise AssertionError("Owned native client remained in final TPU telemetry")
            return report, {**peaks, "native_exit_independently_observed": native_exited,
                            "wall_seconds": time.monotonic() - started}
        finally:
            if process.poll() is None:
                stop_owned_tree(process)
            if descriptor is not None:
                os.close(descriptor)


def compare(left, right):
    a = np.asarray([step["logits"] for step in left["steps"]], dtype=np.float64)
    b = np.asarray([step["logits"] for step in right["steps"]], dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise AssertionError("Full finite logit arrays are required for the same number of steps")
    denominator = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    cosine = max(-1., min(1., float(np.sum(a * b) / (denominator * norm_b)))) if denominator * norm_b else None
    return {"values": int(a.size), "relative_l2": float(np.linalg.norm(a - b) / denominator) if denominator else None,
            "rmse": math.sqrt(float(np.mean((a - b) ** 2))), "cosine": cosine,
            "tokens_equal": left["generated_tokens"] == right["generated_tokens"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh private stress directory")
    parser.add_argument("--bank-rows", type=int, default=192)
    args = parser.parse_args()
    model = args.artifacts.resolve() / "model.gguf"
    if model.stat().st_size > 16 * 1024**2:
        parser.error("Use an existing synthetic model smaller than 16 MiB")
    if os.statvfs(args.output.parent).f_bavail * os.statvfs(args.output.parent).f_frsize < 32 * 1024**2:
        parser.error("At least 32 MiB free storage is required")
    args.output.mkdir(mode=0o700)
    started = time.monotonic()
    before = hardware_snapshot()
    report = {"format": "pixelforge.runtime-stress.v1", "success": False,
              "source_gguf_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
              "method": "Serial fresh engine/broker/helper processes; existing AOT only; no service changes",
              "limits": ["Tiny synthetic model; no general model-capacity or maximum-throughput claim",
                         "Int8 NPU dense projections; remaining architecture operations use CPU",
                         "Whole-device battery samples and global TPU counters include other workloads"],
              "hardware_before": telemetry(before), "runs": []}
    try:
        common = ["--model", str(model), "--bank-rows", str(args.bank_rows), "--timeout", "60"]
        references = {}
        repeated = {}
        for index, (prompt, tokens, threads, cache) in enumerate([
            ("Hi", 8, 1, 0), ("Hi", 8, 2, 0), ("Hi", 8, 4, 0), ("Hi", 8, 2, 1),
            ("Hi", 16, 1, 4), ("Hi", 16, 2, 0),
            ("A short synthetic prompt for NPU.", 8, 2, 0),
            ("A short synthetic prompt for NPU.", 8, 2, 1),
            ("Hi", 8, 2, 0), ("Hi", 16, 2, 0),
        ]):
            reference_key = (prompt, tokens, threads)
            varying = ["--prompt", prompt, "--tokens", str(tokens), "--threads", str(threads)]
            if reference_key not in references:
                reference, _ = run_case(common + varying + ["--cpu"], args.output / f"cpu-{len(references):02d}")
                references[reference_key] = reference["engine"]
            result, observed = run_case(common + varying + ["--read-only", "--artifacts", str(args.artifacts.resolve()),
                                                           "--cache-mib", str(cache)], args.output / f"npu-{index:02d}")
            stats = result["broker"]["npu"]
            numerical = compare(references[reference_key], result["engine"])
            if numerical["relative_l2"] is None or numerical["relative_l2"] > .1:
                raise AssertionError("Full-logit CPU-reference relative L2 exceeded the declared 0.1 stress threshold")
            if not numerical["tokens_equal"]:
                raise AssertionError("This synthetic fixture's generated tokens changed from the CPU reference")
            repeat_key = (prompt, tokens, threads)
            repeat = compare(repeated[repeat_key], result["engine"]) if repeat_key in repeated else None
            if repeat and (not repeat["tokens_equal"] or repeat["relative_l2"] > 1e-6):
                raise AssertionError("Same-prompt NPU results changed across cache budgets or fresh runs")
            repeated[repeat_key] = result["engine"]
            row = {"index": index, "prompt_utf8_bytes": len(prompt.encode()), "prompt_tokens": result["engine"]["prompt_tokens"],
                   "requested_tokens": tokens, "generated_tokens": len(result["engine"]["generated_tokens"]),
                   "threads": threads, "cache_mib": cache, "owned_npu_invocations": stats["npu_invocations"],
                   "graph_loads": stats["loaded_models"], "graph_compiles": stats["compiled_models"],
                   "evictions": len(stats["evictions"]), "resident_code_bytes_before_cleanup": stats["compiled_bytes"],
                   "native_cleanup_ok": result["broker"]["native_cleanup_ok"],
                   "cpu_reference": numerical, "repeat_comparison": repeat, **observed}
            report["runs"].append(row)
            print(json.dumps(row), flush=True)
        report.update(success=True, cpu_reference_runs=len(references),
                      owned_npu_invocations=sum(run["owned_npu_invocations"] for run in report["runs"]),
                      graph_loads=sum(run["graph_loads"] for run in report["runs"]),
                      graph_compiles=sum(run["graph_compiles"] for run in report["runs"]),
                      evictions=sum(run["evictions"] for run in report["runs"]))
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        after = hardware_snapshot()
        report["hardware_after"] = telemetry(after)
        report["seconds"] = time.monotonic() - started
        report["driver_fault_deltas"] = {
            name: int(after["tpu_raw"][name]) - int(before["tpu_raw"][name])
            for name in ("firmware_crash_count", "watchdog_timeout_count")
            if str(before["tpu_raw"].get(name, "")).isdecimal() and str(after["tpu_raw"].get(name, "")).isdecimal()}
        with (args.output / "STRESS.json").open("x") as destination:
            json.dump(report, destination, indent=2)
            destination.write("\n")
    return 0


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
