"""Author fixed-weight LiteRT FC graphs without TensorFlow or a host compiler.

The generated FlatBuffers schema is vendored from ai-edge-litert 2.3.0.
The Android vendor compiler consumes the resulting TFLite graph on this device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import flatbuffers
import numpy as np

import schema_py_generated as schema


def author_constant_fc(weights, weight_dtype="int8", name="custom_fc"):
    """Return a TFLite FC with float32 ports and int8 or float32 constants.

    Int8 constants use a separate scale for each output row. They are
    requantized from decoded GGUF weights; this introduces a second quantization
    error. The host rescales input activations to avoid observed NPU overflow.
    """
    weights = np.asarray(weights, dtype=np.float32)
    if weights.ndim != 2 or min(weights.shape) < 1:
        raise ValueError("FC weights must be a nonempty [outputs, inputs] matrix")
    if not np.isfinite(weights).all():
        raise ValueError("FC weights must be finite")
    if weight_dtype not in ("int8", "float32"):
        raise ValueError("Only int8 and float32 constant weights are supported")

    model = schema.ModelT()
    model.version = 3
    model.description = "Custom Huihui GGUF fixed linear layer " + name

    def buffer(data=None):
        value = schema.BufferT()
        value.data = np.frombuffer(data, dtype=np.uint8) if data is not None else None
        return value

    quantization = None
    if weight_dtype == "int8":
        scales = np.maximum(np.max(np.abs(weights), axis=1) / 127,
                            np.finfo(np.float32).tiny).astype(np.float32)
        constants = np.clip(np.rint(weights / scales[:, None]), -127, 127).astype(np.int8)
        quantization = schema.QuantizationParametersT()
        quantization.scale = scales
        quantization.zeroPoint = np.zeros(weights.shape[0], dtype=np.int64)
        quantization.quantizedDimension = 0
        constant_type = schema.TensorType.INT8
    else:
        constants = weights
        constant_type = schema.TensorType.FLOAT32

    model.buffers = [buffer(), buffer(constants.tobytes()),
                     buffer(np.zeros(weights.shape[0], dtype=np.float32).tobytes())]

    def tensor(label, shape, dtype, buffer_index, quant=None):
        value = schema.TensorT()
        value.name = label
        value.shape = np.asarray(shape, dtype=np.int32)
        value.type = dtype
        value.buffer = buffer_index
        value.quantization = quant
        return value

    subgraph = schema.SubGraphT()
    subgraph.name = name
    subgraph.inputs = np.asarray([0], dtype=np.int32)
    subgraph.outputs = np.asarray([3], dtype=np.int32)
    subgraph.tensors = [
        tensor("activation", [1, weights.shape[1]], schema.TensorType.FLOAT32, 0),
        tensor(name, weights.shape, constant_type, 1, quantization),
        tensor("bias", [weights.shape[0]], schema.TensorType.FLOAT32, 2),
        tensor("result", [1, weights.shape[0]], schema.TensorType.FLOAT32, 0),
    ]
    operation = schema.OperatorT()
    operation.opcodeIndex = 0
    operation.inputs = np.asarray([0, 1, 2], dtype=np.int32)
    operation.outputs = np.asarray([3], dtype=np.int32)
    operation.builtinOptionsType = schema.BuiltinOptions.FullyConnectedOptions
    operation.builtinOptions = schema.FullyConnectedOptionsT()
    subgraph.operators = [operation]
    model.subgraphs = [subgraph]

    opcode = schema.OperatorCodeT()
    opcode.builtinCode = schema.BuiltinOperator.FULLY_CONNECTED
    opcode.deprecatedBuiltinCode = schema.BuiltinOperator.FULLY_CONNECTED
    opcode.version = 12 if weight_dtype == "int8" else 1
    model.operatorCodes = [opcode]
    builder = flatbuffers.Builder(1024)
    offset = model.Pack(builder)
    builder.Finish(offset, file_identifier=b"TFL3")
    return bytes(builder.Output())


def main():
    from gguf_hf_loader import GGUFWeights

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--weight-dtype", choices=("int8", "float32"), default="int8")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    weights = GGUFWeights(args.gguf)
    matrix = weights.numpy(args.tensor)
    graph = author_constant_fc(matrix, args.weight_dtype, args.tensor)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(graph)
    metadata = {
        "source_gguf": str(args.gguf), "gguf_tensor": args.tensor,
        "gguf_row_order_preserved": True, "rows": matrix.shape[0],
        "columns": matrix.shape[1], "weight_dtype": args.weight_dtype,
        "graph_bytes": len(graph), "graph_sha256": hashlib.sha256(graph).hexdigest(),
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
