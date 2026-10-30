"""Original Q5_K/Q6_K levels and coefficients; no additional quantizer."""
import hashlib
import numpy as np
from author_grouped_quant import author_grouped_fc,unpack_groups
from compiled_model import canonical
from dynamic_fc import DynamicProgramCache,DynamicWeightExecutor


class GroupedProgramCache(DynamicProgramCache):
    request_dtype='original_groups'

    def __init__(self,directory,source_gguf=None,**kwargs):
        super().__init__(directory,source_gguf,_program_kind='grouped-fc-v1',**kwargs)

    def group_kind(self,name):
        item=self.layer(name);kind=item['type_id']
        if kind not in (13,14) or item['columns']%256:
            raise ValueError('Original grouped execution requires whole Q5_K/Q6_K blocks')
        return kind

    def input_sizes(self,name,start,stop):
        kind=self.group_kind(name);columns=self.layer(name)['columns'];rows=stop-start
        group=32 if kind==13 else 16
        coefficients=rows*(columns//group)*4
        return [self.batch_vectors*columns*4,rows*columns,coefficients]+([coefficients] if kind==13 else [])

    def identity(self,name,start,stop,dtype,columns):
        item=self.layer(name);kind=self.group_kind(name)
        if dtype!='original_groups' or columns!=item['columns'] or (start,stop) not in self.ranges(name):
            raise ValueError('Original grouped geometry differs from the pinned GGUF layout')
        ports=self.input_sizes(name,start,stop)
        fields=dict(rows=stop-start,columns=columns,batch_vectors=self.batch_vectors,source_quantization_type=kind,
            original_group_size=32 if kind==13 else 16,weight_dtype='original_groups',
            weight_port_bytes=ports[1:],recipe=self.recipe,
            device_fingerprint_sha256=hashlib.sha256(canonical(self.fingerprint)).hexdigest())
        return hashlib.sha256(canonical(fields)).hexdigest(),fields


class GroupedWeightExecutor(DynamicWeightExecutor):
    mode_name='original_groups'

    def _weight_arrays(self,name,start,stop):
        kind=self.cache.group_kind(name);columns=self.cache.layer(name)['columns'];rows=stop-start
        tensor=self.weights.tensors[name]
        block_bytes=176 if kind==13 else 210
        row_bytes=(columns//256)*block_bytes
        # Read the original block bytes directly. No full F32 weight matrix is
        # created and no original integer or effective coefficient is changed.
        raw=self.weights.raw(name).reshape(tensor.shape[0],row_bytes)[start:stop]
        levels,scale,offset=unpack_groups(np.ascontiguousarray(raw).reshape(-1,block_bytes),kind)
        group=32 if kind==13 else 16
        arrays=[np.ascontiguousarray(levels.reshape(rows,columns//group,group),dtype=np.int8),
                np.ascontiguousarray(scale.reshape(rows,columns//group,1),dtype='<f4')]
        if offset is not None:arrays.append(np.ascontiguousarray(offset.reshape(rows,columns//group,1),dtype='<f4'))
        return arrays

    def stats(self):
        return dict(super().stats(),dynamic_weight_mode='original_groups',weight_dtype='original_groups',
            group_unpacks=self.weight_decodes,
            weight_decoding='CPU lossless original integer/group unpack; NPU applies original group coefficients',
            extra_weight_quantization=False)


class GroupedPhaseExecutor:
    """One native process, phase-specific geometry, shared immutable weight FDs."""
    def __init__(self,native,weights,prefill_cache,decode_cache):
        self.native=native
        self.prefill=GroupedWeightExecutor(native,weights,prefill_cache)
        self.decode=(self.prefill if decode_cache is prefill_cache else
                     GroupedWeightExecutor(native,weights,decode_cache))
        self.prefill_single_vector_operations=0

    def __getattr__(self,name):return getattr(self.native,name)

    def __call__(self,name,activation,*,prefill=False):
        single=False
        if prefill and self.prefill.cache.batch_vectors>1 and self.decode.cache.batch_vectors==1:
            shape=np.shape(activation)
            single=len(shape)==2 and shape[0]==1
        executor=self.decode if single or not prefill else self.prefill
        result=executor(name,activation,prefill=prefill)
        if single:self.prefill_single_vector_operations+=1
        return result

    def stats(self):
        children=[self.prefill] if self.prefill is self.decode else [self.prefill,self.decode]
        phase=lambda executor:dict(operations=executor.operations,group_unpacks=executor.weight_decodes,
            weight_uploads=executor.weight_uploads,weight_cache_hits=executor.weight_hits,
            batch_vectors=executor.cache.batch_vectors)
        return dict(self.native.stats(),dynamic_weight_mode='original_groups',weight_dtype='original_groups',
            group_unpacks=sum(x.weight_decodes for x in children),
            weight_uploads=sum(x.weight_uploads for x in children),
            weight_cache_hits=sum(x.weight_hits for x in children),
            dynamic_weight_operations=sum(x.operations for x in children),
            prefill_single_vector_operations=self.prefill_single_vector_operations,
            phase_weight_stats=dict(prefill=phase(self.prefill),decode=phase(self.decode)),
            extra_weight_quantization=False,activation_scaling='none; raw F32 matrix input',
            weight_decoding='CPU lossless original integer/group unpack; NPU applies original group coefficients')


def compile_grouped_geometry(native,cache,name,start,stop):
    if cache.read_only:raise PermissionError('Grouped geometry compilation requires a writable package')
    columns=cache.layer(name)['columns'];kind=cache.group_kind(name);batch=cache.batch_vectors
    key,_=cache.identity(name,start,stop,'original_groups',columns)
    existing=cache.get(name,start,stop,'original_groups',columns,load_code=False)
    if existing is not None:return existing[0]
    graph=author_grouped_fc(stop-start,columns,kind=kind,batch_vectors=batch)
    digest=hashlib.sha256(graph).hexdigest();identifier='group_'+hashlib.sha256((key+digest).encode()).hexdigest()[:32]
    ports=cache.input_sizes(name,start,stop)
    native._make_room(1024**2,transient=sum(ports)*3)
    header=f'C {identifier} {len(ports)} 1 {len(graph)} '+' '.join(map(str,ports))+f' {batch*(stop-start)*4}\n'
    native._write(header.encode(),graph)
    fields=native._readline().split()
    if len(fields)!=4 or fields[:3]!=['OK','COMPILED',identifier]:
        raise RuntimeError('Grouped geometry compilation failed for '
            +f'{stop-start}x{columns} B{batch}: '+' '.join(fields))
    record=dict(native_id=identifier,compiled_bytes=int(fields[3]),graph_sha256=digest,graph_bytes=len(graph))
    try:
        code=native.export_code(record)
        return cache.store(name,start,stop,'original_groups',columns,record,code)
    finally:
        native._write(f'F {identifier}\n'.encode())
        if native._readline()!='OK FREED':raise RuntimeError('Grouped geometry release failed')
