"""Experimental fixed SwiGLU FFN graph with original decoded F32 constants.

This graph consumes raw activations. Maxabs rescaling across SiLU would change
the model function and is forbidden. Authoring preserves the decoded weight
values; vendor arithmetic precision and operator coverage need device tests.
"""
import flatbuffers
import numpy as np
import schema_py_generated as schema


def numpy_columns(weights, name, start, stop, *, max_block_batch=8192):
    """Decode only quantization blocks intersecting a matrix column interval.

    Kept outside the legacy reader so its pinned author/decoder fingerprint and
    existing bank identities remain unchanged. End blocks are decoded exactly
    and sliced; quantized values are never requantized.
    """
    from gguf_hf_loader import UnsupportedQuantizationError
    if type(max_block_batch) is not int or max_block_batch <= 0:
        raise ValueError("max_block_batch must be positive")
    weights.check_source_unchanged(check_path=True)
    tensor = weights.tensors[name]
    if (len(tensor.shape) != 2 or type(start) is not int or type(stop) is not int
            or not 0 <= start <= stop <= tensor.shape[1]):
        raise ValueError("Invalid GGUF matrix column range")
    rows, columns = tensor.shape
    if (not tensor.size_known or tensor.type_id not in weights.decode_supported_types
            or tensor.type_id not in weights.TYPES):
        raise UnsupportedQuantizationError("Unsupported column-slice quantization")
    block, block_bytes = weights.TYPES[tensor.type_id]
    if start == stop:
        return np.empty((rows, 0), dtype=np.float32)
    first, end = start // block, (stop + block - 1) // block
    raw = weights.raw(name).reshape(rows, columns // block, block_bytes)
    selected = np.ascontiguousarray(raw[:, first:end, :]).reshape(rows, -1)
    decoded = weights._decode_raw(selected, tensor, (rows, (end-first)*block),
                                  np.float32, max_block_batch)
    return np.ascontiguousarray(decoded[:, start-first*block:stop-first*block])


def author_full_ffn(gate, up, down, *, batch_vectors=1, name="full_ffn"):
    """Author down(SiLU(gate(X)) * up(X)), including hidden-column chunks."""
    if type(batch_vectors) is not int or not 1 <= batch_vectors <= 512:
        raise ValueError("batch_vectors must be an integer from 1 to 512")
    gate, up, down = (np.ascontiguousarray(w, dtype=np.float32) for w in (gate, up, down))
    if (gate.ndim != 2 or up.shape != gate.shape or min(gate.shape) < 1
            or down.shape != (gate.shape[1], gate.shape[0])):
        raise ValueError("FFN requires gate/up[R,H] and down[H,R]")
    if not all(np.isfinite(w).all() for w in (gate, up, down)):
        raise ValueError("FFN weights must be finite")
    intermediate, hidden = gate.shape
    model = schema.ModelT()
    model.version = 3
    model.description = "PixelForge full SwiGLU FFN F32 v1 " + name

    def buffer(data=None):
        value = schema.BufferT()
        value.data = None if data is None else np.frombuffer(data, dtype=np.uint8)
        return value

    model.buffers = [buffer(), buffer(gate.tobytes()),
        buffer(np.zeros(intermediate, dtype=np.float32).tobytes()), buffer(up.tobytes()),
        buffer(down.tobytes()), buffer(np.zeros(hidden, dtype=np.float32).tobytes())]

    def tensor(label, shape, index=0):
        value = schema.TensorT()
        value.name, value.shape = label, np.asarray(shape, dtype=np.int32)
        value.type, value.buffer = schema.TensorType.FLOAT32, index
        return value

    bh, br = [batch_vectors, hidden], [batch_vectors, intermediate]
    graph = schema.SubGraphT()
    graph.name = name
    graph.inputs, graph.outputs = np.asarray([0], np.int32), np.asarray([12], np.int32)
    graph.tensors = [tensor('input', bh), tensor('gate_weight', gate.shape, 1),
        tensor('gate_bias', [intermediate], 2), tensor('gate', br),
        tensor('up_weight', up.shape, 3), tensor('up_bias', [intermediate], 2),
        tensor('up', br), tensor('sigmoid', br), tensor('silu', br), tensor('product', br),
        tensor('down_weight', down.shape, 4), tensor('down_bias', [hidden], 5), tensor('output', bh)]

    def operation(opcode, inputs, output, options=None, options_type=0):
        value = schema.OperatorT()
        value.opcodeIndex = opcode
        value.inputs, value.outputs = np.asarray(inputs, np.int32), np.asarray([output], np.int32)
        value.builtinOptionsType, value.builtinOptions = options_type, options
        return value

    fc = lambda ins, out: operation(0, ins, out, schema.FullyConnectedOptionsT(),
                                    schema.BuiltinOptions.FullyConnectedOptions)
    mul = lambda ins, out: operation(2, ins, out, schema.MulOptionsT(), schema.BuiltinOptions.MulOptions)
    graph.operators = [fc([0, 1, 2], 3), fc([0, 4, 5], 6),
                       operation(1, [3], 7), mul([3, 7], 8), mul([8, 6], 9), fc([9, 10, 11], 12)]
    model.subgraphs = [graph]
    model.operatorCodes = []
    for builtin in (schema.BuiltinOperator.FULLY_CONNECTED, schema.BuiltinOperator.LOGISTIC,
                    schema.BuiltinOperator.MUL):
        code = schema.OperatorCodeT()
        code.builtinCode = code.deprecatedBuiltinCode = builtin
        code.version = 1
        model.operatorCodes.append(code)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())
