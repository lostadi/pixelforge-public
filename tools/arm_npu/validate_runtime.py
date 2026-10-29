"""Benchmark two chat requests through the packaged custom-model NPU path.

Reports are private, exclusively created files. The hardware collector is
read-only. This runner starts and closes only its own native helper.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback

import numpy as np

from gguf_hf_loader import GGUFWeights
from model_host import MiniCPM5Host, tokenizer
from npu_linear import NPULinear
from runtime import NATIVE_DESTINATION, checked_file, verify_addon

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))
from benchmark import accelerator_process_evidence, energy_delta, hardware_snapshot, process_sample


def counter_delta(before, after, field):
    try:
        left = [int(value) for value in before["tpu_raw"][field].split()]
        right = [int(value) for value in after["tpu_raw"][field].split()]
        if len(left) == len(right):
            return [right_value - left_value for left_value, right_value in zip(left, right)]
    except (KeyError, ValueError):
        pass
    return None


def save_array(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as destination:
        np.save(destination, value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    addon = REPOSITORY / "custom-models/minicpm5"
    parser.add_argument("--model", type=Path, default=addon / "model.gguf")
    parser.add_argument("--tokenizer", type=Path, default=addon / "tokenizer")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--cpu-logits", type=Path)
    parser.add_argument("--cpu-golden", type=Path)
    parser.add_argument("--graph-reference", type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    args.report.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    report_fd = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    report = {"status": "started", "source": "huihui-ai/Huihui-MiniCPM5-2B-abliterated",
              "model": str(args.model), "model_bytes": args.model.stat().st_size,
              "model_backend": "native-google-tensor-npu-fc", "cpu_linear_fallback": False,
              "cpu_operations": "embeddings, RMS, RoPE, attention, KV cache, SiLU, tokenizer, sampling",
              "requests": [], "hardware_samples": [],
              "power_caveat": "ODPM rail power assumes the documented compatible kernel ABI; exact installed CPM revision is unverified. Charging battery readings are not discharge power."}
    stopped = threading.Event()
    native = None
    collector = None
    started = time.monotonic()

    def collect():
        while not stopped.is_set():
            report["hardware_samples"].append(hardware_snapshot(native.process.pid))
            stopped.wait(1.0)

    def event(event_name, **fields):
        print(json.dumps({"event": event_name, "elapsed_seconds": time.monotonic() - started, **fields}), flush=True)

    try:
        manifest = verify_addon(addon, all_files=False)
        pins = json.loads((Path(__file__).parent / "HUIHUI_ASSETS.json").read_text())
        if not os.path.samefile(args.model, addon / "model.gguf"):
            checked_file(args.model.parent, {"path": args.model.name,
                "bytes": pins["quantized_gguf_bytes"], "sha256": pins["quantized_gguf_sha256"]})
        for pin in pins["support_files"]:
            checked_file(args.tokenizer, pin)
        native_pin = next(item for item in manifest["files"] if item["path"] == "native/npu_pipe")
        report["provenance"] = {
            "original_repo": pins["original_checkpoint_repo"],
            "original_revision": pins["original_checkpoint_revision"],
            "gguf_repo": pins["quantized_checkpoint_repo"],
            "gguf_revision": pins["quantized_checkpoint_revision"],
            "gguf_sha256": pins["quantized_gguf_sha256"],
            "model_and_support_sha256_verified": True,
            "native_sha256": native_pin["sha256"],
            "recipe": "GGUF Q8 decode; per-row int8 requantization; float32 FC ports; maxabs activation scaling; native Android on-device compilation; lazy SQ NPU execution; no CPU linear fallback",
        }
        weights = GGUFWeights(args.model)
        tok = tokenizer(args.tokenizer)
        # This fixed comparison recipe does not inherit executable, dtype or
        # bank-size overrides. The ordinary host CLI exposes those separately.
        command = ["/usr/sbin/chroot", "/proc/1/root", "/system/bin/env",
                   "LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:/data/data/com.termux/files/usr/lib",
                   "LD_PRELOAD=/system/lib64/libbinder_ndk.so", str(NATIVE_DESTINATION)]
        native = NPULinear(weights, command=command, weight_dtype="int8", lm_bank_rows=4096)
        executed_binary = Path(f"/proc/{native.process.pid}/exe")
        executed_hash = hashlib.sha256(executed_binary.read_bytes()).hexdigest()
        if executed_hash != native_pin["sha256"]:
            raise ValueError("Executed native helper does not match the verified addon pin")
        report["provenance"]["executed_native_sha256_verified"] = True
        Path(f"/proc/{native.process.pid}/oom_score_adj").write_text("250")
        runner = MiniCPM5Host(weights, linear=native)
        report["before"] = hardware_snapshot(native.process.pid)
        report["native_process_evidence"] = accelerator_process_evidence(native.process.pid)
        collector = threading.Thread(target=collect, daemon=True)
        collector.start()
        prompts = [("hello", "Hi"), ("arithmetic", "Answer with one digit: what is 2+2?")]
        for name, prompt in prompts:
            runner.reset()
            prompt_ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                                tokenize=True, add_generation_prompt=True,
                                                enable_thinking=False)
            if isinstance(prompt_ids, Mapping):
                prompt_ids = prompt_ids["input_ids"]
            prompt_ids = [int(value) for value in prompt_ids]
            request = {"name": name, "prompt": prompt, "prompt_ids": prompt_ids,
                       "tokens": [], "token_timings": [], "before": hardware_snapshot(native.process.pid)}
            before_compiles = len(native.compile_records)
            before_invocations = native.invocations
            before_compile_seconds = native.compile_seconds
            before_invoke_seconds = native.invoke_seconds
            request_start = time.monotonic()
            event("request_start", name=name, prompt_tokens=len(prompt_ids))
            hidden = None
            for index, token in enumerate(prompt_ids):
                hidden = runner.hidden_step(token)
                if index == 0 or index + 1 == len(prompt_ids) or (index + 1) % 8 == 0:
                    event("prefill", name=name, index=index + 1, total=len(prompt_ids))
            logits = runner.logits(hidden)
            request["prefill_seconds"] = time.monotonic() - request_start
            array_path = args.report.with_name(args.report.stem + f".{name}.first-logits.npy")
            save_array(array_path, logits)
            request["first_logits_path"] = str(array_path)
            request["first_logits_sha256"] = hashlib.sha256(array_path.read_bytes()).hexdigest()
            request["first_top10"] = [{"id": int(index), "text": tok.decode([int(index)]),
                                        "logit": float(logits[index])}
                                       for index in np.argsort(logits)[-10:][::-1]]
            if name == "hello" and args.cpu_logits:
                cpu = np.load(args.cpu_logits)
                request["cpu_first_logits"] = {"argmax_match": int(np.argmax(cpu)) == int(np.argmax(logits)),
                    "max_absolute_error": float(np.max(np.abs(cpu - logits))),
                    "relative_l2_error": float(np.linalg.norm(cpu - logits) / np.linalg.norm(cpu)),
                    "cosine_similarity": float(np.dot(cpu, logits) / np.linalg.norm(cpu) / np.linalg.norm(logits))}
            for index in range(4):
                token = int(np.argmax(logits))
                request["tokens"].append(token)
                elapsed = time.monotonic() - request_start
                request["token_timings"].append({"id": token, "seconds": elapsed})
                event("token", name=name, id=token, text=tok.decode([token]),
                      output=tok.decode(request["tokens"], skip_special_tokens=True))
                if token in (1, 130073) or index == 3:
                    break
                logits = runner.step(token)
            request["seconds"] = time.monotonic() - request_start
            request["output"] = tok.decode(request["tokens"], skip_special_tokens=True)
            request["after"] = hardware_snapshot(native.process.pid)
            request["compile_count"] = len(native.compile_records) - before_compiles
            request["compile_seconds"] = native.compile_seconds - before_compile_seconds
            request["npu_invocations"] = native.invocations - before_invocations
            request["invoke_seconds"] = native.invoke_seconds - before_invoke_seconds
            request["tpu_inference_count_delta"] = counter_delta(request["before"], request["after"], "inference_count")
            request["energy_estimate"] = energy_delta(request["before"]["odpm_raw"].get("energy_value"),
                                                      request["after"]["odpm_raw"].get("energy_value"), abi_assumed=True)
            if len(request["token_timings"]) > 1:
                decode_seconds = request["token_timings"][-1]["seconds"] - request["token_timings"][0]["seconds"]
                request["decode_tokens_per_second"] = (len(request["tokens"]) - 1) / decode_seconds
            if name == "hello" and args.cpu_golden:
                golden = json.loads(args.cpu_golden.read_text())
                request["cpu_first_four_tokens_match"] = request["tokens"] == golden["output_ids"][:4]
            if name == "arithmetic":
                request["arithmetic_output_begins_with_4"] = request["output"].strip().startswith("4")
            report["requests"].append(request)
            event("request_complete", name=name, output=request["output"], seconds=request["seconds"],
                  compile_count=request["compile_count"], npu_invocations=request["npu_invocations"])
        report["backend_stats"] = native.stats()
        if args.graph_reference:
            original = json.loads(args.graph_reference.read_text())
            reference = {item["graph_sha256"] for item in original["backend_stats"]["matrices"]}
            current = {item["graph_sha256"] for item in native.compile_records}
            report["graph_reference_comparison"] = {"all_graphs_match": current == reference,
                                                    "current_graphs": len(current), "reference_graphs": len(reference)}
        report["native_process"] = process_sample(native.process.pid)
        report["python_process"] = process_sample(os.getpid())
        report["forbidden_imports"] = [name for name in ("torch", "transformers", "ai_edge_litert", "tensorflow", "jax") if name in sys.modules]
        report["after"] = hardware_snapshot(native.process.pid)
        report["tpu_inference_count_delta"] = counter_delta(report["before"], report["after"], "inference_count")
        report["status"] = "completed"
        event("complete", outputs=[request["output"] for request in report["requests"]],
              npu_invocations=native.invocations, forbidden_imports=report["forbidden_imports"])
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        if native:
            report["backend_stats"] = native.stats()
        event("failed", error=repr(error))
        traceback.print_exc()
    finally:
        stopped.set()
        if collector:
            collector.join(timeout=2)
        if native:
            native.close()
        report["after_cleanup"] = hardware_snapshot()
        report["total_seconds"] = time.monotonic() - started
        with os.fdopen(report_fd, "w") as destination:
            json.dump(report, destination, indent=2)
            destination.write("\n")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
