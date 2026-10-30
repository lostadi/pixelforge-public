"""Original-level/group integrity, phase FD reuse and stale-binding fixtures."""
import os,struct,sys,tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import unittest
import numpy as np

from grouped_fc import GroupedProgramCache,GroupedWeightExecutor,GroupedPhaseExecutor,compile_grouped_geometry
from gguf_hf_loader import GGUFWeights
from model_architecture import build_plan
from npu_linear import NPULinear
from test_gguf_loader import write_gguf
import test_dynamic_fc


class GroupedTests(unittest.TestCase):
    def test_lossless_groups_shared_geometry_phase_buffers_and_partial_rebind_stop(self):
        rng=np.random.default_rng(50)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'original.gguf';tensors=[]
            for name,kind,size in (('a.weight',13,176),('b.weight',13,176),('c.weight',14,210)):
                raw=rng.integers(0,256,(2,size),dtype=np.uint8)
                for offset in ((0,2) if kind==13 else (208,)):
                    raw[:,offset:offset+2]=np.frombuffer(struct.pack('<e',.03125),np.uint8)
                tensors.append((name,(2,256),kind,raw.tobytes()))
            write_gguf(source,tensors);before=source.read_bytes()
            with GGUFWeights(source) as weights,patch.dict(os.environ,{'PIXELFORGE_NPU_STDERR':str(root/'stderr')}):
                plan=build_plan(weights,model_sha256=weights.sha256(),bank_rows=2)
                options=dict(plan=plan,fingerprint={'device':'fixture'})
                p=GroupedProgramCache(root/'grouped-fc-v1/b4',source,batch_vectors=4,**options)
                d=GroupedProgramCache(root/'grouped-fc-v1/b1',source,batch_vectors=1,**options)
                self.assertEqual(p.identity('a.weight',0,2,'original_groups',256),
                                 p.identity('b.weight',0,2,'original_groups',256))
                self.assertNotEqual(p.identity('a.weight',0,2,'original_groups',256)[0],
                                    p.identity('c.weight',0,2,'original_groups',256)[0])
                self.assertEqual(p.input_sizes('a.weight',0,2),[4096,512,64,64])
                self.assertEqual(p.input_sizes('c.weight',0,2),[4096,512,128])
                expected={name:weights.numpy(name) for name,_,_,_ in tensors}
                native=NPULinear(weights,command=[sys.executable,test_dynamic_fc.__file__,'--worker'],
                    weight_dtype='float32',batch_vectors=4,min_free_mib=0,cache_mib=0,native_runtime_cache_mib=1)
                try:
                    for cache,name in ((p,'a.weight'),(p,'c.weight'),(d,'a.weight')):
                        compile_grouped_geometry(native,cache,name,0,2)
                    execute=GroupedPhaseExecutor(native,weights,p,d)
                    # Identity1/zero0 INT8 transport is only an unpacked storage
                    # type: reconstructed values equal the source decoder.
                    for name in expected:
                        arrays=execute.prefill._weight_arrays(name,0,2)
                        restored=arrays[0].astype(np.float32)*arrays[1]
                        if len(arrays)==3:restored-=arrays[2]
                        np.testing.assert_array_equal(restored.reshape(2,256),expected[name])
                    buffers={};keys=[]
                    def cached(key,size):return key in buffers and buffers[key].nbytes==size
                    def upload(record,index,array=None,*,cache_key):
                        keys.append(cache_key)
                        if array is not None:
                            if cache_key in buffers:np.testing.assert_array_equal(buffers[cache_key],array)
                            buffers[cache_key]=array.copy()
                        self.assertIn(cache_key,buffers)
                        native._write(f'U {record["native_id"]} {index}\n'.encode(),buffers[cache_key].tobytes())
                        fields=native._readline().split()
                        self.assertEqual(fields[:4],['OK','UPLOADED',record['native_id'],str(index)])
                        return int(fields[4])
                    with patch.object(native,'weight_buffer_cached',cached,create=True), \
                            patch.object(native,'upload_weight',upload,create=True), \
                            patch.object(weights,'numpy_rows',side_effect=AssertionError('No F32 decode')):
                        x=rng.normal(size=(9,256)).astype('<f4')
                        for name in ('a.weight','b.weight','a.weight','c.weight','a.weight'):
                            np.testing.assert_allclose(execute(name,x,prefill=True),x@expected[name].T,rtol=1e-5,atol=2e-5)
                        self.assertEqual(execute.prefill.weight_decodes,3)
                        self.assertEqual(native.stats()['loaded_models'],2)
                        old_keys=set(keys)
                        np.testing.assert_allclose(execute('a.weight',x[:1]),x[:1]@expected['a.weight'].T,rtol=1e-5,atol=2e-5)
                        self.assertEqual(execute.decode.weight_decodes,0)
                        self.assertTrue(set(keys)<=old_keys)  # B1 and B4 share immutable source buffers.
                        self.assertEqual(execute.stats()['group_unpacks'],3)
                        # A single-vector prefill uses the precompiled B1
                        # geometry while retaining its P phase and FD identity.
                        records_before=len(native.records)
                        np.testing.assert_allclose(execute('a.weight',x[:1],prefill=True),
                            x[:1]@expected['a.weight'].T,rtol=1e-5,atol=2e-5)
                        self.assertEqual([(item['batch_vectors'],item['prefill']) for item in native.records[records_before:]],[(1,True)])
                        self.assertEqual(execute.decode.weight_decodes,0)
                        self.assertTrue(set(keys)<=old_keys)
                        self.assertEqual(execute.stats()['prefill_single_vector_operations'],1)
                        # Under memory pressure, a cold-matrix reservation
                        # would discard useful FD owners. Warm bank switches
                        # must not reserve an unpack/copy that never occurs.
                        def pressure(key,record,*,additional_bytes=0):
                            if additional_bytes:buffers.clear()
                        with patch.object(native,'_ensure_invoke_room',pressure):
                            for name in ('b.weight','a.weight'):
                                np.testing.assert_allclose(execute(name,x[:2],prefill=True),
                                    x[:2]@expected[name].T,rtol=1e-5,atol=2e-5)
                        self.assertEqual(execute.stats()['group_unpacks'],3)
                        # A new codes allocation can evict a coefficient port
                        # that was cached at the start of the binding pass.
                        b_keys={key[4]:key for key in buffers if key[1]=='b.weight'}
                        buffers.pop(b_keys[1])
                        def evict_second(record,index,array=None,*,cache_key):
                            generation=upload(record,index,array,cache_key=cache_key)
                            if cache_key[1]=='b.weight' and index==1:buffers.pop(b_keys[2],None)
                            return generation
                        with patch.object(native,'upload_weight',evict_second):
                            np.testing.assert_allclose(execute('b.weight',x[:2],prefill=True),
                                x[:2]@expected['b.weight'].T,rtol=1e-5,atol=2e-5)
                        np.testing.assert_allclose(execute('a.weight',x[:2],prefill=True),
                            x[:2]@expected['a.weight'].T,rtol=1e-5,atol=2e-5)
                        # A partial source-bank rebind must never mark B ready
                        # or invoke with A's old coefficient ports.
                        invocations=native.invocations
                        record=next(v for k,v in native.models.items() if k[0]=='original_groups' and v['batch_vectors']==4 and len(v['input_sizes'])==4)
                        bound=record['bound_source_bank']
                        def fail_second(record,index,array=None,*,cache_key):
                            if index==2:raise RuntimeError('injected partial binding failure')
                            return upload(record,index,array,cache_key=cache_key)
                        with patch.object(native,'upload_weight',fail_second):
                            with self.assertRaisesRegex(RuntimeError,'partial binding'):
                                execute('b.weight',x[:2],prefill=True)
                        self.assertTrue(native.closed)
                        self.assertEqual(native.invocations,invocations)
                        self.assertEqual(record['bound_source_bank'],bound)
                finally:native.close()
            self.assertEqual(source.read_bytes(),before)

    def test_single_vector_routing_requires_an_existing_b1_geometry(self):
        # Older configurations may share one B4 cache for both phases. Their
        # single-token work must keep that geometry rather than inventing B1.
        for prefill_batch,decode_batch,shared in ((4,4,True),(1,1,True),(4,4,False)):
            with self.subTest(prefill_batch=prefill_batch,decode_batch=decode_batch,shared=shared):
                p=Mock(batch_vectors=prefill_batch)
                d=p if shared else Mock(batch_vectors=decode_batch)
                instances=[]
                def make(native,weights,cache):
                    value=Mock(cache=cache)
                    instances.append(value)
                    return value
                with patch('grouped_fc.GroupedWeightExecutor',side_effect=make):
                    executor=GroupedPhaseExecutor(Mock(),Mock(),p,d)
                x=np.zeros((1,256),np.float32)
                executor('a.weight',x,prefill=True)
                instances[0].assert_called_once_with('a.weight',x,prefill=True)
                if not shared:instances[1].assert_not_called()
                self.assertEqual(executor.prefill_single_vector_operations,0)


if __name__=='__main__':unittest.main()
