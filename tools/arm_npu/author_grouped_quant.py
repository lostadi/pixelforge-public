"""Experimental exact original Q5_K/Q6_K integer levels as runtime inputs.

This is unpacking, not quantization: all original quantized integers and
effective group coefficients are retained. The GGUF stays unchanged. Layout
follows the pinned MIT ggml decoder (see LICENSE.ggml). Device execution must
be validated separately; this module does not enable a runtime route.
"""
import flatbuffers
import numpy as np
import schema_py_generated as s
from gguf_quantizers import Q4_K


def unpack_groups(blocks, kind):
    size = {13: 176, 14: 210}.get(kind)
    a = np.asarray(blocks)
    if size is None or a.dtype != np.uint8 or a.ndim != 2 or a.shape[1] != size:
        raise ValueError('Expected original Q5_K/Q6_K blocks')
    a = np.ascontiguousarray(a)
    n = len(a)
    if kind == 13:
        d = a[:, :2].copy().view('<f2').astype(np.float32)
        minimum = a[:, 2:4].copy().view('<f2').astype(np.float32)
        scale, offset = Q4_K.get_scale_min(a[:, 4:16].copy())
        scale = (d * scale.astype(np.float32)).reshape(n, 8, 1)
        offset = (minimum * offset.astype(np.float32)).reshape(n, 8, 1)
        low = (a[:, 48:].reshape(n, 4, 1, 32) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)) & 15
        high = (a[:, 16:48].reshape(n, 1, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 1, 8, 1)) & 1
        levels = (low.reshape(n, 8, 32) | (high.reshape(n, 8, 32) << 4)).astype(np.int8)
    else:
        scale = (a[:, 208:210].copy().view('<f2').astype(np.float32)
                 * a[:, 192:208].copy().view(np.int8).astype(np.float32)).reshape(n, 16, 1)
        low = (a[:, :128].reshape(n, 2, 1, 64) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)) & 15
        high = (a[:, 128:192].reshape(n, 2, 1, 32) >> np.array([0, 2, 4, 6], np.uint8).reshape(1, 1, 4, 1)) & 3
        levels = ((low.reshape(n, 8, 32) | (high.reshape(n, 8, 32) << 4)).astype(np.int8) - 32).reshape(n, 16, 16)
        offset = None
    if not np.isfinite(scale).all() or (offset is not None and not np.isfinite(offset).all()):
        raise ValueError('Nonfinite original GGUF group coefficient')
    return levels, scale, offset


def author_grouped_fc(rows, columns, *, kind, batch_vectors=16):
    if kind not in (13, 14) or any(type(v) is not int or v < 1 for v in (rows, columns, batch_vectors)):
        raise ValueError('Positive dimensions and original Q5_K/Q6_K required')
    if columns % 256 or batch_vectors > 512:
        raise ValueError('Whole quantization blocks and batch 1..512 required')
    group = 32 if kind == 13 else 16
    model, graph = s.ModelT(), s.SubGraphT()
    model.version = 3
    model.description = 'PixelForge original integer levels; no requantization'
    model.buffers = [s.BufferT()]
    graph.name, graph.tensors = 'original_grouped_fc', []
    specs = [('activation', [batch_vectors, columns], s.TensorType.FLOAT32),
             ('original_levels', [rows, columns // group, group], s.TensorType.INT8),
             ('original_group_scale', [rows, columns // group, 1], s.TensorType.FLOAT32)]
    if kind == 13:
        specs.append(('original_group_offset', [rows, columns // group, 1], s.TensorType.FLOAT32))
    inputs = list(range(len(specs)))
    levels_float = len(specs)
    specs += [('levels_float', [rows, columns // group, group], s.TensorType.FLOAT32),
              ('scaled', [rows, columns // group, group], s.TensorType.FLOAT32)]
    if kind == 13:
        specs.append(('decoded_groups', [rows, columns // group, group], s.TensorType.FLOAT32))
    decoded = len(specs) - 1
    matrix = len(specs)
    specs += [('decoded_weights', [rows, columns], s.TensorType.FLOAT32),
              ('result', [batch_vectors, rows], s.TensorType.FLOAT32)]
    for label, shape, dtype in specs:
        t = s.TensorT()
        t.name, t.shape, t.type, t.buffer = label, np.asarray(shape, np.int32), dtype, 0
        if label == 'original_levels':
            t.quantization = s.QuantizationParametersT()
            t.quantization.scale = np.asarray([1.], np.float32)
            t.quantization.zeroPoint = np.asarray([0], np.int64)
        graph.tensors.append(t)
    codes = []
    graph.operators = []
    def op(builtin, source, target, options=None, options_type=0):
        if builtin not in codes:
            codes.append(builtin)
        value = s.OperatorT()
        value.opcodeIndex = codes.index(builtin)
        value.inputs, value.outputs = np.asarray(source, np.int32), np.asarray([target], np.int32)
        value.builtinOptions, value.builtinOptionsType = options, options_type
        graph.operators.append(value)
    op(s.BuiltinOperator.DEQUANTIZE, [1], levels_float)
    op(s.BuiltinOperator.MUL, [levels_float, 2], levels_float + 1, s.MulOptionsT(), s.BuiltinOptions.MulOptions)
    if kind == 13:
        op(s.BuiltinOperator.SUB, [levels_float + 1, 3], decoded, s.SubOptionsT(), s.BuiltinOptions.SubOptions)
    reshape = s.ReshapeOptionsT()
    reshape.newShape = np.asarray([rows, columns], np.int32)
    op(s.BuiltinOperator.RESHAPE, [decoded], matrix, reshape, s.BuiltinOptions.ReshapeOptions)
    bmm = s.BatchMatMulOptionsT()
    bmm.adjX, bmm.adjY = False, True
    op(s.BuiltinOperator.BATCH_MATMUL, [0, matrix], matrix + 1, bmm, s.BuiltinOptions.BatchMatMulOptions)
    graph.inputs, graph.outputs = np.asarray(inputs, np.int32), np.asarray([matrix + 1], np.int32)
    model.subgraphs, model.operatorCodes = [graph], []
    for builtin in codes:
        code = s.OperatorCodeT()
        code.builtinCode = code.deprecatedBuiltinCode = builtin
        code.version = 2 if builtin == s.BuiltinOperator.DEQUANTIZE else 1
        model.operatorCodes.append(code)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    return bytes(builder.Output())
