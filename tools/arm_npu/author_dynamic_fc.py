"""Static-shape matrix graph whose original decoded weights are runtime inputs.

No constants or quantizer are authored. Device support is established only by
an actual vendor compilation/invocation; this is not a CPU fallback executor.
"""
import flatbuffers
import numpy as np
import schema_py_generated as schema


def author_dynamic_fc(rows, columns, *, batch_vectors=1, name='dynamic_fc'):
    if any(type(value) is not int or value < 1 for value in (rows, columns, batch_vectors)):
        raise ValueError('Matrix dimensions and batch size must be positive integers')
    if batch_vectors > 512:
        raise ValueError('Batch size exceeds the bounded runtime')
    model = schema.ModelT()
    model.version = 3
    model.description = 'PixelForge runtime float32 weight matrix ' + name
    model.buffers = [schema.BufferT()]
    tensors = []
    for label, shape in (('activation', [batch_vectors, columns]),
                         ('original_decoded_weights', [rows, columns]),
                         ('result', [batch_vectors, rows])):
        tensor = schema.TensorT()
        tensor.name = label
        tensor.shape = np.asarray(shape, dtype=np.int32)
        tensor.type = schema.TensorType.FLOAT32
        tensor.buffer = 0
        tensors.append(tensor)
    graph = schema.SubGraphT()
    graph.name = name
    graph.tensors = tensors
    graph.inputs = np.asarray([0, 1], dtype=np.int32)
    graph.outputs = np.asarray([2], dtype=np.int32)
    operation = schema.OperatorT()
    operation.opcodeIndex = 0
    operation.inputs = np.asarray([0, 1], dtype=np.int32)
    operation.outputs = np.asarray([2], dtype=np.int32)
    operation.builtinOptionsType = schema.BuiltinOptions.BatchMatMulOptions
    operation.builtinOptions = schema.BatchMatMulOptionsT()
    operation.builtinOptions.adjX = False
    operation.builtinOptions.adjY = True
    graph.operators = [operation]
    model.subgraphs = [graph]
    opcode = schema.OperatorCodeT()
    opcode.builtinCode = schema.BuiltinOperator.BATCH_MATMUL
    opcode.deprecatedBuiltinCode = schema.BuiltinOperator.BATCH_MATMUL
    opcode.version = 1
    model.operatorCodes = [opcode]
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b'TFL3')
    return bytes(builder.Output())
