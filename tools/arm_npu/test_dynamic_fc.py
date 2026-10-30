"""Geometry identity and runtime-input author fixtures; no hardware inference."""
import tempfile
from pathlib import Path
import unittest
import sys,struct,os,contextlib,io
from unittest.mock import patch
import numpy as np
import schema_py_generated as s
from author_dynamic_fc import author_dynamic_fc
from author_dynamic_full_ffn import author_dynamic_full_ffn
from dynamic_fc import DynamicProgramCache,DynamicWeightExecutor,compile_dynamic_geometry
from npu_linear import NPULinear
from gguf_hf_loader import GGUFWeights
from model_architecture import build_plan
from test_gguf_loader import write_gguf


def evaluate(data,inputs):
    model=s.Model.GetRootAsModel(data,0);g=model.Subgraphs(0);values=dict(enumerate(inputs))
    for i in range(g.TensorsLength()):
        t=g.Tensors(i)
        if t.Buffer():values[i]=model.Buffers(t.Buffer()).DataAsNumpy().view('<f4').reshape(tuple(t.ShapeAsNumpy()))
    for i in range(g.OperatorsLength()):
        op=g.Operators(i);kind=model.OperatorCodes(op.OpcodeIndex()).BuiltinCode()
        args=[values[int(j)] for j in op.InputsAsNumpy()]
        if kind==s.BuiltinOperator.BATCH_MATMUL:value=args[0]@args[1].T
        elif kind==s.BuiltinOperator.FULLY_CONNECTED:value=args[0]@args[1].T+args[2]
        elif kind==s.BuiltinOperator.LOGISTIC:value=1/(1+np.exp(-args[0]))
        elif kind==s.BuiltinOperator.MUL:value=args[0]*args[1]
        elif kind==s.BuiltinOperator.SUB:value=args[0]-args[1]
        elif kind==s.BuiltinOperator.DEQUANTIZE:value=args[0].astype(np.float32)
        elif kind==s.BuiltinOperator.RESHAPE:
            shape=tuple(g.Tensors(int(op.Outputs(0))).ShapeAsNumpy());value=args[0].reshape(shape)
        else:raise AssertionError(kind)
        values[int(op.Outputs(0))]=value
    return values[int(g.Outputs(0))]


def worker():
    """Synthetic sticky helper; CPU math here tests transport, never hardware."""
    source,sink=sys.stdin.buffer,sys.stdout.buffer
    def exact(size):
        data=source.read(size)
        if len(data)!=size:raise EOFError
        return data
    models={};generation=0;budget=1024**2
    def state():
        return sum(x['charge'] for x in models.values() if x['loaded'])
    def reply(value):sink.write((value+'\n').encode());sink.flush()
    reply('READY PIXELFORGE_NPU_PIPE 1')
    while line:=source.readline():
        h=line.decode().split();op=h[0]
        if op=='Q':break
        if op=='I' and h[1]=='CAPABILITIES':reply('OK CAPABILITIES C L D F I B U V STICKY_INPUTS_V1');continue
        if op=='B':budget=int(h[1]);reply(f'OK CACHE {budget}');continue
        if op=='I' and h[1]=='CACHE':reply(f'OK CACHE {budget} {state()} {sum(x["loaded"] for x in models.values())} 0 0 0');continue
        if op in ('C','L'):
            nin=int(h[2]);data=exact(int(h[4]));ins=list(map(int,h[5:5+nin]));outs=list(map(int,h[5+nin:]))
            if op=='C':
                graph=data;code=(b'\x08\0\0\0DGC0'+struct.pack('<I',len(data))+data).ljust(750000,b'\0')
            else:
                code=data;n=struct.unpack('<I',code[8:12])[0];graph=code[12:12+n]
            generation+=1
            models[h[1]]=dict(graph=graph,code=code,inputs=ins,outputs=outs,gen=generation,
                loaded=False,weights={},charge=len(code)+sum((n+4095)//4096*4096 for n in ins+outs))
            reply(f'OK {"COMPILED" if op=="C" else "LOADED"} {h[1]} {len(code)}');continue
        name=h[2] if op=='I' else h[1];m=models[name]
        if op=='I':reply(f'OK MODEL {name} {m["gen"]} {int(m["loaded"])} {len(m["inputs"])} 0'+''.join('1' if i in m['weights'] else '0' for i in range(1,len(m['inputs']))));continue
        if op=='D':reply(f'OK BYTECODE {len(m["code"])}');sink.write(m['code']);sink.flush();continue
        if op=='F':del models[name];reply('OK FREED');continue
        if op=='U':
            index=int(h[2]);data=exact(m['inputs'][index])
            if not 1<=index<len(m['inputs']) or m['charge']>budget:reply('ERR STICKY_CACHE_BUDGET');break
            if not m['loaded']:
                for other in models.values():
                    if other is not m and other['loaded'] and state()+m['charge']>budget:
                        generation+=1;other.update(loaded=False,weights={},gen=generation)
            generation+=1;m.update(loaded=True,gen=generation);m['weights'][index]=data
            reply(f'OK UPLOADED {name} {index} {generation}');continue
        if op=='V':
            if int(h[2])!=m['gen'] or not m['loaded'] or len(m['weights'])!=len(m['inputs'])-1:
                reply('ERR STICKY_STATE');break
            graph=s.Model.GetRootAsModel(m['graph'],0).Subgraphs(0)
            raw=[exact(m['inputs'][0])]+[m['weights'][i] for i in range(1,len(m['inputs']))]
            types={s.TensorType.FLOAT32:'<f4',s.TensorType.INT8:'i1'}
            inputs=[np.frombuffer(data,types[graph.Tensors(int(graph.Inputs(i))).Type()]).reshape(
                tuple(graph.Tensors(int(graph.Inputs(i))).ShapeAsNumpy())) for i,data in enumerate(raw)]
            output=evaluate(m['graph'],inputs).astype('<f4')
            reply(f'OK OUTPUT 1 {output.nbytes}');sink.write(output.tobytes());sink.flush();continue
        raise ValueError(h)


class DynamicProgramTests(unittest.TestCase):
    def test_runtime_weight_graphs_have_no_model_constants(self):
        rng=np.random.default_rng(44)
        x=rng.normal(size=(3,4)).astype(np.float32)
        gate,up,down=[rng.normal(0,.1,shape).astype(np.float32) for shape in ((7,4),(7,4),(4,7))]
        data=author_dynamic_fc(7,4,batch_vectors=3)
        np.testing.assert_array_equal(evaluate(data,[x,gate]),x@gate.T)
        full=author_dynamic_full_ffn(4,7,batch_vectors=3)
        a=x@gate.T;b=x@up.T
        np.testing.assert_allclose(evaluate(full,[x,gate,up,down]),(a/(1+np.exp(-a))*b)@down.T,rtol=1e-6,atol=1e-6)
        for graph in (data,full):
            model=s.Model.GetRootAsModel(graph,0)
            self.assertEqual(model.BuffersLength(),1)
            self.assertEqual(model.Buffers(0).DataLength(),0)
            self.assertLess(len(graph),4096)

    def test_geometry_code_identity_is_shared_but_source_quantization_is_pinned(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'fixture.gguf'
            matrix=np.arange(8,dtype='<f4').reshape(2,4)
            write_gguf(source,[('a.weight',matrix.shape,0,matrix.tobytes()),
                               ('b.weight',matrix.shape,0,(-matrix).tobytes())])
            with GGUFWeights(source) as weights:
                plan=build_plan(weights,model_sha256=weights.sha256(),bank_rows=2)
                cache=DynamicProgramCache(root/'dynamic',source,plan=plan,batch_vectors=4,fingerprint={'device':'fixture'})
                self.assertEqual(cache.identity('a.weight',0,2,'float32',4),cache.identity('b.weight',0,2,'float32',4))
                key,fields=cache.identity('a.weight',0,2,'float32',4)
                self.assertNotIn('source_gguf_sha256',fields)
                self.assertNotIn('tensor',fields)
                self.assertEqual(cache._manifest()['source_gguf']['sha256'],weights.sha256())
                self.assertEqual(cache._manifest()['plan']['dense_tensors'][0]['type_id'],0)
                code=b'\x08\0\0\0DGC0geometry'
                record=dict(graph_sha256='b'*64,graph_bytes=460,compiled_bytes=len(code))
                cache.store('a.weight',0,2,'float32',4,record,code)
                got=cache.get('b.weight',0,2,'float32',4)
                self.assertEqual(got[1],code)
                path=cache.directory/got[0]['bytecode_path']
                path.write_bytes(code[:-1]+b'X')
                with self.assertRaisesRegex(ValueError,'hash/identifier'):
                    cache.get('b.weight',0,2,'float32',4)

    def test_sticky_weight_upload_once_per_live_bank_and_reload_after_eviction(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'weights.gguf';a=np.asarray([[.1,.2],[-.4,.7]],'<f4');b=-a
            write_gguf(source,[('a.weight',a.shape,0,a.tobytes()),('b.weight',b.shape,0,b.tobytes())])
            with GGUFWeights(source) as weights,patch.dict(os.environ,{'PIXELFORGE_NPU_STDERR':str(root/'stderr')}):
                plan=build_plan(weights,model_sha256=weights.sha256(),bank_rows=2)
                cache=DynamicProgramCache(root/'dynamic',source,plan=plan,batch_vectors=4,fingerprint={'device':'fixture'})
                native=NPULinear(weights,command=[sys.executable,__file__,'--worker'],weight_dtype='float32',
                    batch_vectors=4,decode_batch_vectors=4,min_free_mib=0,cache_mib=0,native_runtime_cache_mib=1)
                try:
                    compile_dynamic_geometry(native,cache,'a.weight',0,2)
                    self.assertEqual(native.invocations,0)
                    readonly=DynamicProgramCache(cache.directory,source,plan=plan,batch_vectors=4,
                                                  fingerprint={'device':'fixture'},read_only=True)
                    execute=DynamicWeightExecutor(native,weights,readonly)
                    x=np.arange(18,dtype=np.float32).reshape(9,2)
                    np.testing.assert_allclose(execute('a.weight',x,prefill=True),x@a.T,rtol=1e-6,atol=1e-6)
                    self.assertEqual(execute.weight_decodes,1);self.assertEqual(native.invocations,3)
                    with patch.object(weights,'numpy_rows',side_effect=AssertionError('Sticky hit must not decode')):
                        np.testing.assert_allclose(execute('a.weight',x[:1]),x[:1]@a.T,rtol=1e-6,atol=1e-6)
                    self.assertEqual(execute.weight_decodes,1);self.assertEqual(execute.weight_hits,1)
                    # Equal geometries reuse one runtime. A warm bitmap alone
                    # must never authorize reuse of another bank's weights.
                    np.testing.assert_allclose(execute('b.weight',x[:1]),x[:1]@b.T,rtol=1e-6,atol=1e-6)
                    np.testing.assert_allclose(execute('a.weight',x[:1]),x[:1]@a.T,rtol=1e-6,atol=1e-6)
                    self.assertEqual(execute.weight_decodes,3)
                    self.assertEqual(execute.weight_uploads,3)
                    self.assertEqual(native.stats()['loaded_models'],1)
                finally:native.close()

    def test_geometry_bulk_plan_hashes_once_without_decoding_or_creating_package(self):
        import compile_dynamic_fc
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=root/'fixture.gguf';matrix=np.ones((2,4),'<f4')
            write_gguf(source,[('a.weight',matrix.shape,0,matrix.tobytes()),
                               ('b.weight',matrix.shape,0,matrix.tobytes())])
            calls=[];original=GGUFWeights.sha256
            def sha(weights):calls.append(weights.path);return original(weights)
            output=root/'new-package';capture=io.StringIO()
            with contextlib.redirect_stdout(capture),patch.object(GGUFWeights,'sha256',sha), \
                    patch.object(GGUFWeights,'numpy_rows',side_effect=AssertionError('No weight decode')), \
                    patch.object(compile_dynamic_fc,'NPULinear',side_effect=AssertionError('No helper startup')):
                self.assertEqual(compile_dynamic_fc.main(['--model',str(source),'--output',str(output),
                    '--bank-rows','2','--batch-vectors','4','--plan-only']),0)
            import json
            plan=json.loads(capture.getvalue())
            self.assertEqual(plan['logical_banks'],2);self.assertEqual(plan['geometry_programs'],1)
            self.assertEqual(len(calls),1);self.assertFalse(output.exists())


if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='--worker':worker()
    else:unittest.main()
