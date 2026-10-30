"""Geometry-only full SwiGLU graph: X, Wgate, Wup, Wdown are F32 inputs."""
import flatbuffers
import numpy as np
import schema_py_generated as s


def author_dynamic_full_ffn(hidden, intermediate, *, batch_vectors=16, name='dynamic_full_ffn'):
    if any(type(x) is not int or x < 1 for x in (hidden, intermediate, batch_vectors)) or batch_vectors > 512:
        raise ValueError('Positive dimensions and batch 1..512 required')
    model, graph = s.ModelT(), s.SubGraphT()
    model.version = 3
    model.description = 'PixelForge raw SwiGLU with original decoded runtime weight inputs'
    model.buffers = [s.BufferT()]
    graph.name, graph.tensors = name, []
    shapes = ([batch_vectors,hidden], [intermediate,hidden], [intermediate,hidden],
              [hidden,intermediate], [batch_vectors,intermediate], [batch_vectors,intermediate],
              [batch_vectors,intermediate], [batch_vectors,intermediate],
              [batch_vectors,intermediate], [batch_vectors,hidden])
    labels = ('input','gate_weight','up_weight','down_weight','gate','up','sigmoid','silu','product','output')
    for label, shape in zip(labels, shapes):
        t = s.TensorT()
        t.name, t.shape, t.type, t.buffer = label, np.asarray(shape,np.int32), s.TensorType.FLOAT32, 0
        graph.tensors.append(t)
    graph.inputs, graph.outputs = np.asarray([0,1,2,3],np.int32), np.asarray([9],np.int32)

    def op(code, inputs, output, options=None, kind=0):
        value = s.OperatorT()
        value.opcodeIndex, value.inputs, value.outputs = code, np.asarray(inputs,np.int32), np.asarray([output],np.int32)
        value.builtinOptionsType, value.builtinOptions = kind, options
        return value
    def bmm(inputs, output):
        options = s.BatchMatMulOptionsT()
        options.adjX, options.adjY = False, True
        return op(0,inputs,output,options,s.BuiltinOptions.BatchMatMulOptions)
    def mul(inputs, output):
        return op(2,inputs,output,s.MulOptionsT(),s.BuiltinOptions.MulOptions)
    graph.operators = [bmm([0,1],4), bmm([0,2],5), op(1,[4],6),
                       mul([4,6],7), mul([7,5],8), bmm([8,3],9)]
    model.subgraphs, model.operatorCodes = [graph], []
    for builtin in (s.BuiltinOperator.BATCH_MATMUL,s.BuiltinOperator.LOGISTIC,s.BuiltinOperator.MUL):
        code=s.OperatorCodeT()
        code.builtinCode=code.deprecatedBuiltinCode=builtin
        code.version=1
        model.operatorCodes.append(code)
    builder=flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder),file_identifier=b'TFL3')
    return bytes(builder.Output())
