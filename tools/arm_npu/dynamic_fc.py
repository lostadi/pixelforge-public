"""Geometry-only programs; original GGUF banks become sticky F32 inputs."""
import hashlib
import time
import numpy as np

from author_dynamic_fc import author_dynamic_fc
from compiled_model import ArtifactCache, canonical


class DynamicProgramCache(ArtifactCache):
    request_dtype='float32'

    def __init__(self, directory, source_gguf=None, _program_kind='dynamic-fc-v1', **kwargs):
        super().__init__(directory, source_gguf, _program_kind=_program_kind, **kwargs)
        self.source_plan = self._manifest()['plan']

    def layer(self, name):
        for item in self.source_plan['dense_tensors']:
            if item['name'] == name:
                return item
        raise ValueError('Dynamic weight is outside the pinned source tensor plan')

    def ranges(self, name):
        item = self.layer(name)
        return [(start,min(start+item['bank_rows'],item['rows']))
                for start in range(0,item['rows'],item['bank_rows'])]

    def identity(self, name, start, stop, dtype, columns):
        item = self.layer(name)
        if dtype!='float32' or columns!=item['columns'] or (start,stop) not in self.ranges(name):
            raise ValueError('Dynamic geometry differs from the pinned GGUF bank layout')
        # The manifest separately pins source bytes and tensor quantization.
        # Equal matrix geometries can reuse code across unrelated source banks.
        fields=dict(rows=stop-start,columns=columns,batch_vectors=self.batch_vectors,
            weight_dtype='float32-runtime-input',weight_input_bytes=(stop-start)*columns*4,
            recipe=self.recipe,device_fingerprint_sha256=hashlib.sha256(canonical(self.fingerprint)).hexdigest())
        return hashlib.sha256(canonical(fields)).hexdigest(),fields

    def input_sizes(self,name,start,stop):
        columns=self.layer(name)['columns']
        return [self.batch_vectors*columns*4,(stop-start)*columns*4]


def compile_dynamic_geometry(native, cache, name, start, stop):
    if cache.read_only:
        raise PermissionError('Dynamic compilation requires a writable geometry package')
    columns,batch=cache.layer(name)['columns'],cache.batch_vectors
    key,_=cache.identity(name,start,stop,'float32',columns)
    existing=cache.get(name,start,stop,'float32',columns,load_code=False)
    if existing is not None:return existing[0]
    rows=stop-start
    graph=author_dynamic_fc(rows,columns,batch_vectors=batch)
    digest=hashlib.sha256(graph).hexdigest()
    identifier='dyn_'+hashlib.sha256((key+digest).encode()).hexdigest()[:32]
    native._make_room(1024**2,transient=rows*columns*12+batch*(rows+columns)*8)
    native._write(f'C {identifier} 2 1 {len(graph)} {batch*columns*4} {rows*columns*4} {batch*rows*4}\n'.encode(),graph)
    fields=native._readline().split()
    if len(fields)!=4 or fields[:3]!=['OK','COMPILED',identifier]:
        raise RuntimeError('Dynamic geometry compilation failed')
    record=dict(native_id=identifier,compiled_bytes=int(fields[3]),graph_sha256=digest,
                graph_bytes=len(graph))
    try:
        code=native.export_code(record)
        return cache.store(name,start,stop,'float32',columns,record,code)
    finally:
        native._write(f'F {identifier}\n'.encode())
        if native._readline()!='OK FREED':raise RuntimeError('Dynamic geometry release failed')


class DynamicWeightExecutor:
    mode_name='float32'
    def __init__(self,native,weights,cache):
        if 'STICKY_INPUTS_V1' not in (native.capabilities or ()):
            raise RuntimeError('Dynamic weights require STICKY_INPUTS_V1')
        if not native.native_runtime_cache_mib:
            raise ValueError('Dynamic weights require a positive device runtime cache budget')
        self.native,self.weights,self.cache=native,weights,cache
        self.weight_decodes=self.weight_uploads=self.weight_hits=self.operations=0

    def __getattr__(self,name):return getattr(self.native,name)

    def _load(self,name,start,stop):
        item=self.cache.layer(name);columns=item['columns'];rows=stop-start;batch=self.cache.batch_vectors
        geometry,_=self.cache.identity(name,start,stop,self.cache.request_dtype,columns)
        # Reuse the compiled runtime for equal geometry. Its bound source bank
        # is separate state, committed only after every weight port is rebound.
        key=(self.mode_name,geometry)
        if key in self.native.models:
            self.native.models.move_to_end(key)
            return key,self.native.models[key]
        cached=self.cache.get(name,start,stop,self.cache.request_dtype,columns,
            reserve=lambda size:self.native._make_room(size,transient=size*2+rows*columns*8))
        if cached is None:raise FileNotFoundError('Dynamic geometry was not precompiled')
        stored,code=cached
        identifier='dyn_'+hashlib.sha256(canonical([key,stored['graph_sha256']])).hexdigest()[:32]
        began=time.monotonic()
        ports=self.cache.input_sizes(name,start,stop)
        header=f'L {identifier} {len(ports)} 1 {len(code)} '+' '.join(map(str,ports))+f' {batch*rows*4}\n'
        self.native._write(header.encode(),code)
        if self.native._readline().split()!=['OK','LOADED',identifier,str(len(code))]:
            raise RuntimeError('Dynamic geometry load failed')
        record=dict(name=name,rows=rows,columns=columns,batch_vectors=batch,native_id=identifier,
            compiled_bytes=len(code),graph_sha256=stored['graph_sha256'],operation='load',compile_seconds=0.,
            load_seconds=time.monotonic()-began,weight_input_bytes=sum(ports[1:]),input_sizes=ports,
            weight_port_bytes={index:size for index,size in enumerate(ports[1:],1)})
        self.native.models[key]=record;self.native.compile_records.append(record)
        self.native.load_seconds+=record['load_seconds']
        while self.native.cache_budget and len(self.native.models)>1 and self.native._resident_bytes()>self.native.cache_budget:
            self.native._free_oldest('dynamic geometry code budget',exclude=key)
        return key,record

    def _weight_arrays(self,name,start,stop):
        matrix=self.weights.numpy_rows(name,start,stop)
        if (matrix.shape!=(stop-start,self.cache.layer(name)['columns']) or matrix.dtype!=np.float32
                or not np.isfinite(matrix).all()):
            raise ValueError('Invalid original decoded F32 weight bank')
        return [np.ascontiguousarray(matrix,dtype='<f4')]

    def _prime(self,name,start,stop,key,record):
        identifier=record['native_id']
        self.native._write(f'I MODEL {identifier}\n'.encode())
        state=self.native._readline().split()
        ports=record['input_sizes'];ninputs=len(ports)
        if (len(state)!=7 or state[:3]!=['OK','MODEL',identifier] or not state[3].isdecimal()
                or state[4] not in ('0','1') or state[5]!=str(ninputs) or len(state[6])!=ninputs
                or state[6][0]!='0' or any(c not in '01' for c in state[6])):
            raise RuntimeError('Invalid sticky model state')
        source_bank=(self.cache.source_sha256,name,start,stop,self.cache.layer(name)['type_id'],self.mode_name)
        if (state[4]=='1' and state[6]=='0'+'1'*(ninputs-1)
                and record.get('bound_source_bank')==source_bank
                and record.get('bound_generation')==int(state[3])):
            if int(state[3])==0:raise RuntimeError('Invalid ready weight generation')
            self.weight_hits+=1
            return int(state[3])
        weight_bytes=record['weight_input_bytes'];batch=self.cache.batch_vectors
        charged=record['compiled_bytes']+sum((size+4095)//4096*4096 for size in
            ports+[batch*record['rows']*4])
        if charged>self.native.native_runtime_cache_mib*1024**2:
            raise MemoryError('Dynamic weight bank exceeds the device cache budget; use smaller row banks')
        kind=self.cache.layer(name)['type_id']
        keys=[(self.cache.source_sha256,name,start,stop,index,kind,self.mode_name) for index in range(1,ninputs)]
        def cache_state():
            return [bool(hasattr(self.native,'weight_buffer_cached') and
                         self.native.weight_buffer_cached(k,size)) for k,size in zip(keys,ports[1:])]
        cached=cache_state()
        was_cached=all(cached)
        # Reimporting immutable shared buffers creates no decoded matrix or
        # upload copy. Reserving a cold unpack working set here evicts useful
        # weights even when every port is already available.
        self.native._ensure_invoke_room(key,record,additional_bytes=0 if was_cached else weight_bytes*4)
        cached=cache_state()
        if was_cached and not all(cached):
            self.native._ensure_invoke_room(key,record,additional_bytes=weight_bytes*4)
            cached=cache_state()
        arrays=None
        if not all(cached):
            arrays=self._weight_arrays(name,start,stop);self.weight_decodes+=1
            if len(arrays)!=ninputs-1 or any(x.nbytes!=size for x,size in zip(arrays,ports[1:])):
                raise ValueError('Original weight input layout differs from its pinned geometry')
        generation=int(state[3])
        for index,(cache_key,hit) in enumerate(zip(keys,cached),1):
            # Allocating an earlier port may evict a later cached DMA buffer.
            # Recheck instead of relying on the snapshot taken before uploads.
            if hasattr(self.native,'weight_buffer_cached'):
                hit=self.native.weight_buffer_cached(cache_key,ports[index])
            if not hit and arrays is None:
                arrays=self._weight_arrays(name,start,stop);self.weight_decodes+=1
            array=None if hit else arrays[index-1]
            if hasattr(self.native,'upload_weight'):
                next_generation=self.native.upload_weight(record,index,array,cache_key=cache_key)
            else:
                if array is None:raise RuntimeError('Cached weight transport unavailable')
                self.native._write(f'U {identifier} {index}\n'.encode(),array.tobytes())
                fields=self.native._readline().split()
                if len(fields)!=5 or fields[:4]!=['OK','UPLOADED',identifier,str(index)] or not fields[4].isdecimal():
                    raise RuntimeError('Sticky weight upload failed')
                next_generation=int(fields[4])
            if type(next_generation)is not int or next_generation<=generation:
                raise RuntimeError('Sticky weight generation did not advance after upload')
            generation=next_generation;self.weight_uploads+=1
        record['bound_source_bank']=source_bank
        record['bound_generation']=generation
        return generation

    def __call__(self,name,activation,*,prefill=False):
        self.weights.check_source_unchanged(check_path=True)
        item=self.cache.layer(name);rows,columns=item['rows'],item['columns'];batch=self.cache.batch_vectors
        x=np.asarray(activation,dtype=np.float32)
        if x.ndim!=2 or len(x)<1 or x.shape[1]!=columns or x.shape[0]*max(rows,columns)*4>256*1024**2 or not np.isfinite(x).all():
            raise ValueError('Invalid dynamic matrix activation')
        try:
            self.native._make_room(0,transient=len(x)*(columns*8+rows*8))
            output=np.empty((len(x),rows),dtype=np.float32)
            for start,stop in self.cache.ranges(name):
                key,record=self._load(name,start,stop)
                generation=self._prime(name,start,stop,key,record)
                for begin in range(0,len(x),batch):
                    used=min(batch,len(x)-begin);chunk=np.zeros((batch,columns),dtype='<f4')
                    chunk[:used]=x[begin:begin+used]
                    self.native._ensure_invoke_room(key,record)
                    began=time.monotonic()
                    self.native._write(f'V {record["native_id"]} {generation}\n'.encode(),chunk.tobytes())
                    size=batch*(stop-start)*4
                    if self.native._readline().split()!=['OK','OUTPUT','1',str(size)]:
                        raise RuntimeError('Dynamic sticky execution failed; no replay')
                    result=np.frombuffer(self.native._read(size),dtype='<f4').reshape(batch,stop-start)
                    if not np.isfinite(result).all():raise RuntimeError('Nonfinite dynamic NPU result')
                    output[begin:begin+used,start:stop]=result[:used]
                    elapsed=time.monotonic()-began
                    self.native.invocations+=1;self.native.invoke_seconds+=elapsed
                    self.native.records.append(dict(name=name,row_start=start,row_stop=stop,vectors=used,
                        batch_vectors=batch,kind='dynamic-f32',seconds=elapsed,prefill=prefill))
            self.operations+=1
            return output
        except BaseException:
            self.native.close()
            raise

    def stats(self):
        return dict(self.native.stats(),dynamic_weight_mode='float32',weight_decodes=self.weight_decodes,
            weight_uploads=self.weight_uploads,weight_cache_hits=self.weight_hits,
            dynamic_weight_operations=self.operations,activation_scaling='none; raw F32 matrix input',
            weight_decoding='CPU original GGUF to float32; no additional quantization')
