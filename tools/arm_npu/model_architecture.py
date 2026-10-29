"""Plan bounded NPU matrix compilation without inventing a model's graph.

The pinned llama.cpp engine implements architecture-specific forward graphs,
metadata semantics and tokenization. This module describes tensor compilation
coverage separately from engine availability and actual hardware validation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

ENGINE = json.loads(Path(__file__).with_name("LLAMA_ENGINE.json").read_text())
NON_GENERATION_ARCHITECTURES = {
    "bert", "modern-bert", "nomic-bert", "nomic-bert-moe", "neo-bert", "jina-bert-v2",
    "jina-bert-v3", "eurobert", "t5encoder", "gemma-embedding", "llama-embed",
    "pangu-embedded", "wavtokenizer-dec", "qwen3tts", "pockettts",
}


def build_plan(weights, *, model_sha256=None, bank_rows=4096):
    if type(bank_rows) is not int or bank_rows < 1:
        raise ValueError("bank_rows must be a positive integer")
    architecture = weights.fields.get("general.architecture", "unknown")
    dense, unsupported, cpu_weights = [], [], []
    for name, tensor in weights.tensors.items():
        if len(tensor.shape) == 2 and name.endswith(".weight"):
            rows, columns = tensor.shape
            supported = tensor.type_id in weights.decode_supported_types
            if supported:
                block, _ = weights.TYPES[tensor.type_id]
                supported = columns % block == 0
            if not supported:
                unsupported.append({"tensor": name, "reason": "unsupported quantization or row alignment",
                                    "type_id": tensor.type_id, "shape": list(tensor.shape)})
                continue
            dense.append({"name": name, "rows": rows, "columns": columns,
                          "bank_rows": min(bank_rows, rows), "type_id": tensor.type_id,
                          "role": "lookup-and-tied-output" if name == "token_embd.weight" and
                          "output.weight" not in weights.tensors else "rank-two-weight",
                          "graphs": math.ceil(rows / bank_rows)})
        elif len(tensor.shape) >= 3 and ("_exps." in name or "_experts." in name):
            unsupported.append({"tensor": name, "reason": "indexed expert matmul adapter not implemented",
                                "type_id": tensor.type_id, "shape": list(tensor.shape)})
        else:
            cpu_weights.append({"name": name, "shape": list(tensor.shape), "type_id": tensor.type_id})
    core_available = architecture in ENGINE["architectures"]
    configuration = {key: value for key, value in weights.fields.items()
                     if key.startswith(architecture + ".") or key.startswith("general.")}
    split_count = weights.fields.get("split.count", 1)
    split_no = weights.fields.get("split.no", 0)
    split_tensors = weights.fields.get("split.tensors.count", len(weights.tensors))
    if split_count != 1 or split_no != 0 or split_tensors != len(weights.tensors):
        unsupported.append({"reason": "split GGUF shard assembly is not implemented"})
    return {"format": "pixelforge.gguf.plan.v1", "architecture": architecture,
            "model_sha256": model_sha256, "runtime_provider": "llama.cpp+pixelforge-dense-npu-bridge",
            "upstream_commit": ENGINE["commit"], "dense_tensors": dense,
            "configuration": configuration, "cpu_weight_tensors": cpu_weights,
            "unsupported_tensor_operations": unsupported,
            "tensor_compile_supported": bool(dense) and not unsupported,
            "generation_engine_recognized": core_available,
            "generation_adapter_supported": core_available and architecture not in
            NON_GENERATION_ARCHITECTURES and bool(dense) and not unsupported,
            "hardware_validated": False,
            "coverage_note": "2D constant weight MUL_MAT on NPU; other graph operations on CPU. "
                             "Engine name recognition does not establish correct generation on hardware.",
            "dense_graphs": sum(item["graphs"] for item in dense),
            "dense_parameters": sum(item["rows"] * item["columns"] for item in dense)}


def write_bridge_plan(plan, destination):
    """Write the strict allowlist: NCOLS NROWS TYPE_ID GGUF_TENSOR_NAME."""
    if not plan["generation_adapter_supported"]:
        raise ValueError("This GGUF has no complete supported generation adapter: " +
                         json.dumps(plan["unsupported_tensor_operations"]))
    lines = []
    for tensor in plan["dense_tensors"]:
        name = tensor["name"]
        if len(name.encode()) > 63 or re.search(r"\s", name) or not name:
            raise ValueError("GGML bridge requires nonempty tensor names <=63 bytes without whitespace")
        lines.append(f"{tensor['columns']} {tensor['rows']} {tensor['type_id']} {name}\n")
    Path(destination).write_text("".join(lines))
    Path(destination).chmod(0o600)
