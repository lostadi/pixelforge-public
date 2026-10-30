"""Persist native NPU FC bytecode from a GGUF; resume without recompilation.

All rank-two matrices can be compiled. A matrix artifact is not a complete
architecture implementation; the plan records the required graph executor
and any unsupported operations explicitly. No NPU invocation occurs here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil

from compiled_model import ArtifactCache, batch_directory
from gguf_hf_loader import GGUFWeights
from npu_linear import NPULinear

MIB = 1024 ** 2
DEFAULT_NATIVE_COMMAND = [
    "/usr/sbin/chroot", "/proc/1/root", "/system/bin/env",
    "LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:/data/data/com.termux/files/usr/lib",
    "LD_PRELOAD=/system/lib64/libbinder_ndk.so", "/data/local/tmp/pixelforge-npu/npu_pipe_gguf",
]


def selected_banks(weights, plan, tensor=None, row_range=None, tensor_pattern=None):
    if row_range is not None and tensor is None:
        raise ValueError("--rows requires --tensor")
    if tensor is not None and tensor_pattern is not None:
        raise ValueError("Use --tensor or --tensor-pattern, not both")
    pattern = re.compile(tensor_pattern) if tensor_pattern is not None else None
    selected = [item for item in plan["dense_tensors"]
                if (tensor is None or item["name"] == tensor)
                and (pattern is None or pattern.fullmatch(item["name"]))]
    if tensor is not None and not selected:
        raise ValueError("Requested tensor is not a rank-two matrix in the architecture plan")
    if not selected:
        raise ValueError("No rank-two matrices are available for compilation")
    banks = []
    for item in selected:
        descriptor = weights.tensors[item["name"]]
        if len(descriptor.shape) != 2 or tuple(descriptor.shape) != (item["rows"], item["columns"]):
            raise ValueError("Architecture plan matrix dimensions differ from GGUF")
        bank_rows = int(item["bank_rows"])
        if bank_rows < 1:
            raise ValueError("Matrix bank rows must be positive")
        begin, end = row_range if row_range is not None else (0, item["rows"])
        if not 0 <= begin < end <= item["rows"]:
            raise ValueError("Selected row range lies outside the GGUF matrix")
        for start in range(begin, end, bank_rows):
            banks.append((item["name"], start, min(start + bank_rows, end), item["columns"]))
    return banks


def disk_preflight(output, banks, dtype, *, copied_source_bytes=0, existing=None):
    """Reject insufficient disk before native startup or source copying."""
    location = output.resolve()
    while not location.exists():
        location = location.parent
    available = shutil.disk_usage(location).free
    per_weight = 2 if dtype == "int8" else 4
    required = copied_source_bytes + 128 * MIB
    missing = 0
    for name, start, stop, columns in banks:
        key = existing.identity(name, start, stop, dtype, columns)[0] if existing else None
        if existing and key in existing._manifest()["artifacts"]:
            existing.get(name, start, stop, dtype, columns, load_code=False)
            continue
        # Device-validated FC sizes are 2/4 bytes per weight plus a fixed
        # compiler envelope. Reserve extra space for atomic files/metadata.
        required += (stop-start) * columns * per_weight + 2 * MIB
        missing += 1
    if available < required:
        raise OSError(f"Not enough disk for {missing} new compiled graphs: estimated {required/MIB:.1f} MiB "
                      f"including reserve, available {available/MIB:.1f} MiB. Use --tensor/--rows, "
                      "a filesystem with more space, or --resume; RAM cache settings do not reduce artifact disk size.")
    return {"available_bytes": available, "required_bytes_estimate": required, "missing_graphs": missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tensor")
    selection.add_argument("--tensor-pattern", help="Full-match regex selecting many tensors; source hashes once")
    parser.add_argument("--rows", help="START:STOP; only with --tensor")
    parser.add_argument("--bank-rows", type=int, default=4096)
    parser.add_argument("--batch-vectors", type=int, default=1,
                        help="Fixed FC batch; >1 writes a versioned sidecar and requires float32 constants")
    parser.add_argument("--fuse-gate-up", action="store_true",
                        help="Compile versioned stacked FFN gate/up weights; --tensor uses ffn_gate_up.weight")
    parser.add_argument("--weight-dtype", choices=("int8", "float32"), default="int8")
    parser.add_argument("--copy-source", action="store_true", help="Embed model.gguf instead of referencing source")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--native-command-json", help="Explicit JSON argv for the D/L-capable native helper")
    args = parser.parse_args()
    if args.bank_rows < 1:
        parser.error("--bank-rows must be positive")
    if args.tensor_pattern is not None:
        try:
            re.compile(args.tensor_pattern)
        except re.error:
            parser.error("--tensor-pattern must be a valid regular expression")
    if not 1 <= args.batch_vectors <= 512:
        parser.error("--batch-vectors must be from 1 to 512")
    if (args.batch_vectors > 1 or args.fuse_gate_up) and args.weight_dtype != "float32":
        parser.error("Batched compilation requires --weight-dtype float32; no additional int8 quantization")
    if (args.batch_vectors > 1 or args.fuse_gate_up) and args.copy_source:
        parser.error("Batched sidecars reference the original GGUF; do not use --copy-source")
    row_range = None
    if args.rows:
        try:
            values = args.rows.split(":")
            if len(values) != 2:
                raise ValueError
            row_range = tuple(int(value) for value in values)
        except ValueError:
            parser.error("--rows must contain integer START:STOP")
    with GGUFWeights(args.model) as weights:
        return compile_weights(args, row_range, weights)


def compile_weights(args, row_range, weights):
    batch_vectors = getattr(args, "batch_vectors", 1)
    fused = getattr(args, "fuse_gate_up", False)
    output = batch_directory(args.output / 'fused-ffn-v1' if fused else args.output, batch_vectors)
    source_sha256 = weights.sha256()
    from model_architecture import build_plan
    plan = build_plan(weights, model_sha256=source_sha256, bank_rows=args.bank_rows)
    if fused:
        from fused_ffn import FusedFFNWeights
        weights = FusedFFNWeights(weights, plan, bank_rows=args.bank_rows)
        plan = weights.plan
    if plan.get("model_sha256") != source_sha256:
        raise ValueError("Architecture plan refers to different GGUF bytes")
    pattern = getattr(args, 'tensor_pattern', None)
    banks = [] if args.plan_only and args.tensor is None and pattern is None and not plan["dense_tensors"] else selected_banks(
        weights, plan, args.tensor, row_range, pattern)
    print(json.dumps({"event": "plan", "architecture": plan["architecture"],
                      "selected_graphs": len(banks), "runtime_provider": plan.get("runtime_provider"),
                      "batch_vectors": batch_vectors, "fused_gate_up": fused, "output": str(output),
                      "unsupported_tensor_operations": plan.get("unsupported_tensor_operations", []),
                      "source_gguf_sha256": source_sha256}), flush=True)
    existing = None
    if (output / "MODEL.json").exists():
        if not args.resume:
            raise FileExistsError("Compiled package exists; use --resume to verify and extend it")
        existing = ArtifactCache(output, args.model, source_sha256=source_sha256,
                                 verify_source=False, source_identity=weights.source_identity,
                                 read_only=args.plan_only, plan=plan, batch_vectors=batch_vectors,
                                 fused_gate_up=fused)
    copied_bytes = args.model.stat().st_size if args.copy_source and not (args.output / "model.gguf").exists() else 0
    try:
        preflight = disk_preflight(output, banks, args.weight_dtype,
                                   copied_source_bytes=copied_bytes, existing=existing)
    except OSError as error:
        if not args.plan_only:
            raise
        print(json.dumps({"event": "disk_preflight", "would_fit": False, "reason": str(error)}), flush=True)
        print(json.dumps(plan, indent=2))
        return 0
    print(json.dumps({"event": "disk_preflight", "would_fit": True, **preflight}), flush=True)
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    cache = existing or ArtifactCache(output, args.model, source_sha256=source_sha256,
                                      verify_source=False, source_identity=weights.source_identity,
                                      plan=plan, batch_vectors=batch_vectors, fused_gate_up=fused)
    if args.copy_source:
        cache.embed_source()
    command = json.loads(args.native_command_json) if args.native_command_json else DEFAULT_NATIVE_COMMAND
    native = NPULinear(weights, command=command, weight_dtype=args.weight_dtype,
                       artifact_cache=cache if batch_vectors == 1 and not fused else None,
                       batch_artifact_cache=cache if batch_vectors > 1 and not fused else None,
                       fused_artifact_cache=cache if batch_vectors == 1 and fused else None,
                       fused_batch_artifact_cache=cache if batch_vectors > 1 and fused else None,
                       batch_vectors=batch_vectors, cache_mib=1)
    try:
        for name, start, stop, _ in banks:
            record = native._compile(name, start, stop, batch_vectors)
            # Compiled code already persisted atomically by NPULinear. Drop
            # every native graph before moving to the next to bound RAM.
            native._free_oldest("bulk compiler releases persisted graph")
            print(json.dumps({"event": "artifact_ready", "tensor": name,
                              "row_start": start, "row_stop": stop,
                              "bytecode_sha256": record["bytecode_sha256"]}), flush=True)
        full_selection = args.tensor is None and pattern is None and not plan.get("unsupported_tensor_operations")
        if full_selection:
            cache.mark_complete(args.weight_dtype)
        print(json.dumps({"event": "complete", "output": str(output.resolve()),
                          "batch_vectors": batch_vectors,
                          "coverage": "full-rank-two-plan" if full_selection else "partial",
                          "compiled_graphs": native.stats()["compiled_models"],
                          "loaded_graphs": native.stats()["loaded_models"],
                          "npu_invocations": native.invocations}), flush=True)
    finally:
        native.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
