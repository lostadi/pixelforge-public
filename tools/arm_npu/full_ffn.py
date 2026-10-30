"""Pinned full SwiGLU programs and nonlinear, fixed-batch NPU execution."""
import hashlib
import json
import math
import time

import numpy as np

from author_full_ffn import author_full_ffn, numpy_columns
from compiled_model import ArtifactCache, canonical


def build_full_ffn_plan(weights, model_sha256, hidden_chunk=2048):
    if type(hidden_chunk) is not int or hidden_chunk < 1:
        raise ValueError("hidden_chunk must be positive")
    layers = []
    for gate, tensor in sorted(weights.tensors.items()):
        if not gate.endswith('.ffn_gate.weight') or len(tensor.shape) != 2:
            continue
        prefix = gate[:-len('ffn_gate.weight')]
        up, down = prefix+'ffn_up.weight', prefix+'ffn_down.weight'
        intermediate, hidden = tensor.shape
        if (up not in weights.tensors or down not in weights.tensors
                or weights.tensors[up].shape != tensor.shape
                or weights.tensors[down].shape != (hidden, intermediate)):
            continue
        layers.append(dict(gate=gate, up=up, down=down, hidden=hidden,
                           intermediate=intermediate, chunks=math.ceil(intermediate/hidden_chunk)))
    return dict(format='pixelforge.full-ffn.plan.v1', model_sha256=model_sha256,
                function='down(silu(gate(X))*up(X))', hidden_chunk=hidden_chunk, layers=layers)


class FullFFNProgramCache(ArtifactCache):
    """Separate format and identity; never opens a legacy FC as a full FFN."""
    def __init__(self, directory, source_gguf=None, **kwargs):
        super().__init__(directory, source_gguf, _program_kind='full-ffn-v1', **kwargs)
        self.full_plan = self._manifest()['plan']

    def layer(self, name):
        for layer in self.full_plan['layers']:
            if layer['gate'] == name:
                return layer
        raise ValueError("Full FFN name is absent from the pinned plan")

    def ranges(self, name):
        layer = self.layer(name)
        step = self.full_plan['hidden_chunk']
        return [(start, min(start+step, layer['intermediate']))
                for start in range(0, layer['intermediate'], step)]

    def identity(self, name, start, stop, dtype, columns):
        layer = self.layer(name)
        if (dtype != 'float32' or columns != layer['hidden'] or (start, stop) not in self.ranges(name)):
            raise ValueError("Full FFN identity must match its F32 hidden-chunk plan")
        fields = dict(tensor=name, source_tensors=[layer['gate'], layer['up'], layer['down']],
            hidden_start=start, hidden_stop=stop, rows=columns, columns=columns,
            batch_vectors=self.batch_vectors, weight_dtype='float32',
            source_gguf_sha256=self.source_sha256, recipe=self.recipe,
            device_fingerprint_sha256=hashlib.sha256(canonical(self.fingerprint)).hexdigest())
        return hashlib.sha256(canonical(fields)).hexdigest(), fields

    def _output_bytes(self, name, start, stop, columns):
        return self.batch_vectors * columns * 4


def compile_full_ffn_chunk(native, weights, cache, name, start, stop):
    """Compile/persist a program only; this function never invokes it."""
    if cache.read_only:
        raise PermissionError("Full FFN compilation requires a writable new package")
    layer = cache.layer(name)
    hidden, batch = layer['hidden'], cache.batch_vectors
    key, _ = cache.identity(name, start, stop, 'float32', hidden)
    existing = cache.get(name, start, stop, 'float32', hidden, load_code=False)
    if existing is not None:
        return existing[0]
    parameters = 3 * hidden * (stop-start)
    native._make_room(parameters*4 + 1024**2, transient=parameters*24)
    gate = weights.numpy_rows(layer['gate'], start, stop)
    up = weights.numpy_rows(layer['up'], start, stop)
    down = numpy_columns(weights, layer['down'], start, stop)
    graph = author_full_ffn(gate, up, down, batch_vectors=batch, name=name)
    del gate, up, down
    digest = hashlib.sha256(graph).hexdigest()
    identifier = 'ffn_' + hashlib.sha256((key+digest).encode()).hexdigest()[:32]
    began = time.monotonic()
    native._write(f'C {identifier} 1 1 {len(graph)} {batch*hidden*4} {batch*hidden*4}\n'.encode(), graph)
    fields = native._readline().split()
    if len(fields) != 4 or fields[:3] != ['OK', 'COMPILED', identifier]:
        raise RuntimeError('Full FFN compilation failed')
    size = int(fields[3])
    record = dict(name=name, rows=hidden, columns=hidden, batch_vectors=batch,
        compiled_bytes=size, graph_bytes=len(graph), graph_sha256=digest,
        native_id=identifier, operation='compile', compile_seconds=time.monotonic()-began)
    del graph
    try:
        native._ensure_invoke_room(None, record, additional_bytes=size)
        code = native.export_code(record)
        return cache.store(name, start, stop, 'float32', hidden, record, code)
    finally:
        native._write(f'F {identifier}\n'.encode())
        if native._readline() != 'OK FREED':
            raise RuntimeError('Failed to release compiled full FFN program')


class FullFFNExecutor:
    def __init__(self, native, weights, cache, selected=None, phases='PD'):
        if phases not in ('P', 'D', 'PD'):
            raise ValueError('Full FFN phases must be P, D or PD')
        self.phases = phases
        self.native, self.weights, self.cache = native, weights, cache
        planned = {x['gate'] for x in cache.full_plan['layers']}
        if selected is not None and (not isinstance(selected, list) or not selected
                or any(x not in planned for x in selected) or len(set(selected)) != len(selected)):
            raise ValueError('Invalid full FFN selection')
        self.selected = planned if selected is None else set(selected)
        self.operations = 0
        self.phase_operations = {'P': 0, 'D': 0}

    def _load(self, name, start, stop):
        hidden, batch = self.cache.layer(name)['hidden'], self.cache.batch_vectors
        key = ('full-ffn-v1', name, start, stop, batch)
        if key in self.native.models:
            self.native.models.move_to_end(key)
            return key, self.native.models[key]
        cached = self.cache.get(name, start, stop, 'float32', hidden,
            reserve=lambda size: self.native._make_room(size, transient=size*2))
        if cached is None:
            raise FileNotFoundError('Full FFN program is not compiled')
        stored, code = cached
        identifier = 'ffn_' + hashlib.sha256(canonical([key, stored['graph_sha256']])).hexdigest()[:32]
        began = time.monotonic()
        self.native._write(f'L {identifier} 1 1 {len(code)} {batch*hidden*4} {batch*hidden*4}\n'.encode(), code)
        if self.native._readline().split() != ['OK', 'LOADED', identifier, str(len(code))]:
            raise RuntimeError('Full FFN bytecode loading failed')
        record = dict(name=name, rows=hidden, columns=hidden, batch_vectors=batch,
            native_id=identifier, compiled_bytes=len(code), operation='load',
            graph_sha256=stored['graph_sha256'], bytecode_sha256=stored['bytecode_sha256'],
            compile_seconds=0.0, load_seconds=time.monotonic()-began,
            hidden_start=start, hidden_stop=stop)
        self.native.models[key] = record
        self.native.compile_records.append(record)
        self.native.load_seconds += record['load_seconds']
        while self.native.cache_budget and len(self.native.models)>1 and self.native._resident_bytes()>self.native.cache_budget:
            self.native._free_oldest('full FFN compiled-code budget', exclude=key)
        return key, record

    def __call__(self, name, activation, *, prefill=False):
        phase = 'P' if prefill else 'D'
        if phase not in self.phases:
            raise ValueError('Full FFN operation is outside its configured phases')
        if name not in self.selected:
            raise ValueError('Full FFN request is outside its selected program set')
        self.weights.check_source_unchanged(check_path=True)
        hidden, batch = self.cache.layer(name)['hidden'], self.cache.batch_vectors
        x = np.asarray(activation, dtype=np.float32)
        if (x.ndim != 2 or x.shape[0] < 1 or x.shape[1] != hidden
                or x.size*4 > 256*1024**2 or not np.isfinite(x).all()):
            raise ValueError('Invalid raw full FFN activation')
        try:
            self.native._make_room(0, transient=x.size*16 + batch*hidden*8)
            output = np.zeros(x.shape, dtype=np.float32)
            for start, stop in self.cache.ranges(name):
                key, record = self._load(name, start, stop)
                for begin in range(0, len(x), batch):
                    used = min(batch, len(x)-begin)
                    chunk = np.zeros((batch, hidden), dtype='<f4')
                    chunk[:used] = x[begin:begin+used]
                    self.native._ensure_invoke_room(key, record)
                    began = time.monotonic()
                    self.native._write(f'R {record["native_id"]}\n'.encode(), chunk.tobytes())
                    size = batch*hidden*4
                    if self.native._readline().split() != ['OK', 'OUTPUT', '1', str(size)]:
                        raise RuntimeError('Full FFN execution failed')
                    result = np.frombuffer(self.native._read(size), dtype='<f4').reshape(batch, hidden)
                    if not np.isfinite(result).all():
                        raise RuntimeError('Nonfinite full FFN output')
                    output[begin:begin+used] += result[:used]
                    if not np.isfinite(output[begin:begin+used]).all():
                        raise RuntimeError('Nonfinite full FFN partial sum')
                    elapsed = time.monotonic()-began
                    self.native.invoke_seconds += elapsed
                    self.native.invocations += 1
                    self.native.records.append(dict(name=name, kind='full-ffn', hidden_start=start,
                        hidden_stop=stop, vectors=used, batch_vectors=batch, seconds=elapsed, prefill=prefill))
            self.operations += 1
            self.phase_operations[phase] += 1
            return output
        except BaseException:
            self.native.close()
            raise
