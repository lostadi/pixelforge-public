"""Run a supported GGUF using its own tokenizer and the NPU dense bridge."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import time

from gguf_hf_loader import GGUFWeights
from model_architecture import build_plan, write_bridge_plan

KIT = Path(__file__).resolve().parent
DEFAULT_ENGINE = KIT.parents[1] / "custom-models/gguf-runtime/bin/pixelforge-gguf"
DEFAULT_HELPER = "/data/local/tmp/pixelforge-npu/npu_pipe_gguf"


def main():
    def interrupted(signum, _frame):
        raise InterruptedError(f"Run interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt", default="Hi")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu", action="store_true", help="Explicit CPU reference run; no NPU request")
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--read-only", action="store_true", help="Require saved graphs; never compile cache misses")
    parser.add_argument("--bank-rows", type=int, default=4096)
    parser.add_argument("--weight-dtype", choices=("int8", "float32"), default="int8")
    parser.add_argument("--cache-mib", type=int, default=0, help="Optional owned-code RAM cache limit;0 automatic headroom")
    parser.add_argument("--native-command-json")
    parser.add_argument("--timeout", type=int, default=1800, help="Whole engine deadline in seconds")
    args = parser.parse_args()
    if not 1 <= args.tokens <= 128 or not 1 <= args.threads <= 32 or args.cache_mib < 0 or args.timeout < 1:
        parser.error("Invalid tokens, threads or cache limit")
    if args.read_only and args.artifacts is None:
        parser.error("--read-only requires --artifacts")
    engine = args.engine.resolve()
    if not engine.is_file() or not os.access(engine, os.X_OK):
        raise FileNotFoundError("Build the ARM engine with build_llama_npu.sh first: " + str(engine))
    weights = GGUFWeights(args.model)
    digest = weights.sha256()
    plan = build_plan(weights, model_sha256=digest, bank_rows=args.bank_rows)
    if not args.cpu and not plan["generation_adapter_supported"]:
        raise ValueError("Unsupported GGUF generation path: " + json.dumps(plan["unsupported_tensor_operations"]))
    output = args.output or Path.home() / ".local/state/pixelforge-npu" / f"gguf-{time.time_ns()}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)  # Refuse existing directories and redirected results.
    output = output.resolve()
    environment = os.environ.copy()
    for key in ("PIXELFORGE_GGML_BROKER", "PIXELFORGE_GGML_PLAN"):
        environment.pop(key, None)
    native_command = json.loads(args.native_command_json) if args.native_command_json else [
        "/usr/sbin/chroot", "/proc/1/root", "/system/bin/env",
        "LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:/data/data/com.termux/files/usr/lib",
        "LD_PRELOAD=/system/lib64/libbinder_ndk.so", DEFAULT_HELPER]
    (output / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n")
    if not args.cpu:
        write_bridge_plan(plan, output / "weights.plan")
        configuration = {"model": str(args.model.resolve()), "model_sha256": digest,
                         "bank_rows": args.bank_rows, "weight_dtype": args.weight_dtype,
                         "cache_mib": args.cache_mib, "native_command": native_command,
                         "artifacts": str(args.artifacts.resolve()) if args.artifacts else None,
                         "ownership": str(output / "BROKER_OWNER.json"),
                         "read_only": args.read_only, "report": str(output / "broker.json")}
        (output / "broker-config.json").write_text(json.dumps(configuration) + "\n")
        invocation = [str(Path(sys.executable).absolute()), "-I", "-c",
                      "import runpy,sys;sys.path.insert(0," + repr(str(KIT)) + ");"
                      "runpy.run_path(" + repr(str(KIT / "gguf_broker.py")) + ",run_name='__main__')",
                      "--config", str(output / "broker-config.json")]
        launcher = output / "broker-launch.sh"
        launcher.write_text("#!/bin/sh\nexec " + shlex.join(invocation) + "\n")
        launcher.chmod(0o700)
        environment["PIXELFORGE_GGML_BROKER"] = str(launcher)
        environment["PIXELFORGE_GGML_PLAN"] = str(output / "weights.plan")
        environment["PIXELFORGE_NPU_STDERR"] = str(output / "native.stderr")
    sys.path.insert(0, str(KIT.parents[1]))
    from benchmark import hardware_snapshot
    report = {"success": False, "source_gguf_sha256": digest, "model": str(args.model.resolve()),
              "architecture": plan["architecture"], "engine_sha256": hashlib.sha256(engine.read_bytes()).hexdigest(),
              "backend": "cpu-reference" if args.cpu else "google-tensor-npu-dense+ggml-cpu",
              "hardware_before": hardware_snapshot(), "prompt": args.prompt,
              "artifacts": str(args.artifacts) if args.artifacts else None}
    before = time.monotonic()
    process = None
    try:
        with (output / "engine.stdout").open("x") as stdout, (output / "engine.stderr").open("x") as stderr:
            process = subprocess.Popen([str(engine), str(args.model.resolve()), args.prompt,
                                        str(args.tokens), str(args.threads)], env=environment,
                                       stdout=stdout, stderr=stderr, start_new_session=True)
            process.wait(timeout=args.timeout)
        report["engine_exit_code"] = process.returncode
        if process.returncode:
            raise RuntimeError("Architecture engine failed; inspect " + str(output / "engine.stderr"))
        report["engine"] = json.loads((output / "engine.stdout").read_text())
        if args.cpu:
            report["engine"]["cpu_dynamic_matmul_operations"] = None
            report["engine"]["bridge_attribution_measured"] = False
        if not args.cpu:
            report["broker"] = json.loads((output / "broker.json").read_text())
            if not report["broker"].get("success") or not report["broker"].get("native_cleanup_ok"):
                raise RuntimeError("NPU broker or native cleanup failed")
            if report["broker"]["npu"]["npu_invocations"] < 1:
                raise RuntimeError("No NPU invocations were verified")
        weights.check_source_unchanged(check_path=True)
        report["success"] = True
    except BaseException as error:
        report["error"] = str(error)
    finally:
        # Repeated termination must not interrupt release of this run's jobs.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        def cleanup_failure(stage, error):
            message = f"{stage}: {type(error).__name__}: {error}"
            report["success"] = False
            report.setdefault("cleanup_errors", []).append(message)
            report.setdefault("error", message)

        report["engine_cleanup_ok"] = False
        try:
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass  # The owned engine exited between poll and killpg.
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
            report["engine_cleanup_ok"] = process is None or process.poll() is not None
            if not report["engine_cleanup_ok"]:
                raise RuntimeError("Owned engine exit was not confirmed")
        except Exception as error:
            cleanup_failure("engine cleanup", error)

        report["broker_cleanup_ok"] = args.cpu or process is None
        descriptor = None
        try:
            if not args.cpu and process is not None:
                owner_path = output / "BROKER_OWNER.json"
                startup_deadline = time.monotonic() + 2
                while not owner_path.exists() and time.monotonic() < startup_deadline:
                    time.sleep(.05)
                if not owner_path.exists():
                    raise RuntimeError("Owned broker exit cannot be confirmed without its ownership record")
                owner = json.loads(owner_path.read_text())
                if (type(owner.get("pid")) is not int or owner["pid"] < 1
                        or not isinstance(owner.get("start_ticks"), str)
                        or not owner["start_ticks"].isdecimal()):
                    raise ValueError("Invalid broker ownership record")
                try:
                    descriptor = os.pidfd_open(owner["pid"])
                    observed = Path(f"/proc/{owner['pid']}/stat").read_text().split(") ", 1)[1].split()[19]
                    if observed == owner["start_ticks"]:
                        if not select.select([descriptor], [], [], 20)[0]:
                            signal.pidfd_send_signal(descriptor, signal.SIGTERM)
                            if not select.select([descriptor], [], [], 20)[0]:
                                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                                if not select.select([descriptor], [], [], 5)[0]:
                                    raise RuntimeError("Owned broker did not exit after forced cleanup")
                                cleanup_failure("broker cleanup", RuntimeError("Owned broker required forced cleanup"))
                except (ProcessLookupError, FileNotFoundError):
                    pass  # Only an already-identified owned process has gone.
                # A different start tick means the recorded process has gone;
                # its reused PID must never receive a signal.
                report["broker_cleanup_ok"] = True
        except Exception as error:
            cleanup_failure("broker cleanup", error)
        finally:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError as error:
                    cleanup_failure("broker pidfd close", error)
        report["seconds"] = time.monotonic() - before
        try:
            report["hardware_after"] = hardware_snapshot()
        except Exception as error:
            cleanup_failure("final hardware snapshot", error)
        try:
            with (output / "RESULT.json").open("x") as destination:
                json.dump(report, destination, indent=2)
                destination.write("\n")
        finally:
            weights.close()
    print(json.dumps({"success": report["success"], "report": str(output / "RESULT.json"),
                      "text": report.get("engine", {}).get("text"), "error": report.get("error")}))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(main())
