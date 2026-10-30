"""Versioned batched FC author; the legacy single-vector author stays intact.

Float32 constants preserve decoded GGUF values at authoring time. Actual vendor
arithmetic precision still requires a device comparison before deployment.
"""
import flatbuffers

from build_constant_fc import author_constant_fc
import schema_py_generated as schema


def author_batched_fc(weights, weight_dtype="float32", name="custom_fc", *, batch_vectors=1):
    if type(batch_vectors) is not int or not 1 <= batch_vectors <= 512:
        raise ValueError("batch_vectors must be an integer from 1 to 512")
    graph = author_constant_fc(weights, weight_dtype, name)
    if batch_vectors == 1:
        return graph
    if weight_dtype != "float32":
        raise ValueError("Batched FC requires float32 constants; no additional int8 quantization")
    model = schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(graph, 0))
    subgraph = model.subgraphs[0]
    for index in (0, 3):
        subgraph.tensors[index].shape = subgraph.tensors[index].shape.copy()
        subgraph.tensors[index].shape[0] = batch_vectors
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=b"TFL3")
    return bytes(builder.Output())
