"""Canonical original-block tiles, phase FD sharing and fail-closed partials."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

import compile_grouped_columns
from compiled_model import GROUPED_FC_RECIPE
from gguf_hf_loader import GGUFWeights
from grouped_columns import ColumnLayout, ColumnProgramCache, ColumnWeightExecutor, ColumnPhaseExecutor, column_ranges
from grouped_fc import compile_grouped_geometry
from model_architecture import build_plan
from npu_linear import NPULinear
from test_gguf_loader import write_gguf
import test_dynamic_fc


class ColumnTests(unittest.TestCase):
    def setUp(self):
        self.work=tempfile.TemporaryDirectory(prefix='pixelforge-column-fixture-')
        self.addCleanup(self.work.cleanup)
        self.root=Path(self.work.name)
        self.source=self.root/'original.gguf'
        rng=np.random.default_rng(119)
        tensors=[]
        for name,kind,columns in (('wide.weight',14,9216),('middle.weight',13,4096),('small.weight',13,2560)):
            size=176 if kind==13 else 210
            raw=rng.integers(0,256,(2*columns//256,size),dtype=np.uint8)
            for offset in ((0,2) if kind==13 else (208,)):
                raw[:,offset:offset+2]=np.frombuffer(struct.pack('<e',.03125),np.uint8)
            tensors.append((name,(2,columns),kind,raw.tobytes()))
        write_gguf(self.source,tensors)
        self.before=hashlib.sha256(self.source.read_bytes()).hexdigest()
        self.weights=GGUFWeights(self.source)
        self.addCleanup(self.weights.close)
        self.plan=build_plan(self.weights,model_sha256=self.weights.sha256(),bank_rows=2)

    def cache(self,batch):
        return ColumnProgramCache(self.root/f'grouped-columns-v1/b{batch}',self.source,
            source_sha256=self.weights.sha256(),source_identity=self.weights.source_identity,
            verify_source=False,plan=self.plan,batch_vectors=batch,fingerprint={'device':'fixture'})

    def native(self):
        self.env=patch.dict(os.environ,{'PIXELFORGE_NPU_STDERR':str(self.root/'stderr')})
        self.env.start()
        self.addCleanup(self.env.stop)
        native=NPULinear(self.weights,command=[sys.executable,test_dynamic_fc.__file__,'--worker'],
            weight_dtype='float32',batch_vectors=4,min_free_mib=0,cache_mib=0,native_runtime_cache_mib=1)
        self.addCleanup(native.close)
        return native

    def compile(self,native,cache,name='wide.weight'):
        for alias in cache.aliases_for(name):
            for start,stop in cache.ranges(alias):
                compile_grouped_geometry(native,cache,alias,start,stop)

    def test_canonical_greedy_layout_and_noncanonical_aliases_are_rejected(self):
        self.assertEqual(column_ranges(2560),[(0,2560)])
        self.assertEqual(column_ranges(4096),[(0,3072),(3072,4096)])
        self.assertEqual(column_ranges(9216),[(0,3072),(3072,6144),(6144,9216)])
        layout=ColumnLayout(self.plan)
        self.assertEqual(layout.aliases_for('small.weight'),['small.weight'])
        self.assertEqual(layout.layer('small.weight')['columns'],2560)
        self.assertEqual(layout.layer('middle.weight::cols[3072:4096]')['columns'],3072)
        self.assertEqual(layout.layer('middle.weight::cols[3072:4096]')['logical_columns'],1024)
        self.assertEqual(layout.binding('wide.weight::cols[3072:6144]'),('wide.weight',3072,6144))
        for alias in ('wide.weight','wide.weight::cols[0:3073]','wide.weight::cols[256:3328]',
                      'wide.weight::cols[03072:6144]','wide.weight::cols[6144:3072]',
                      'middle.weight::cols[0:2048]'):
            with self.subTest(alias=alias),self.assertRaises(ValueError):layout.binding(alias)
        for value in (0,255,3073,True):
            with self.assertRaises(ValueError):column_ranges(value)
        malformed=dict(self.plan,dense_tensors=[self.plan['dense_tensors'][0]]*2)
        with self.assertRaises(ValueError):ColumnLayout(malformed)
        collision=[dict(self.plan['dense_tensors'][0]),dict(self.plan['dense_tensors'][2],name='wide.weight::cols[0:3072]')]
        with self.assertRaisesRegex(ValueError,'collides'):ColumnLayout(dict(self.plan,dense_tensors=collision))

    def test_original_plan_and_both_author_pins_isolate_the_new_family(self):
        cache=self.cache(4)
        manifest=cache._manifest()
        self.assertEqual(manifest['plan'],self.plan)
        self.assertNotEqual(cache.recipe,GROUPED_FC_RECIPE)
        self.assertIn('grouped_fc_author_sha256',cache.fingerprint)
        self.assertIn('grouped_columns_author_sha256',cache.fingerprint)
        aliases=cache.aliases_for('wide.weight')
        self.assertEqual(cache.identity(aliases[0],0,2,cache.request_dtype,3072),
                         cache.identity(aliases[1],0,2,cache.request_dtype,3072))
        changed=cache._manifest()
        changed['plan']['dense_tensors'][0]['columns']+=256
        cache.manifest_path.write_text(json.dumps(changed))
        with self.assertRaisesRegex(ValueError,'architecture plan differs'):
            cache.get(aliases[0],0,2,cache.request_dtype,3072)

    def test_raw_row_and_column_slices_preserve_every_original_weight_bit(self):
        cache=self.cache(4)
        executor=ColumnWeightExecutor(Mock(capabilities=['STICKY_INPUTS_V1'],native_runtime_cache_mib=1),self.weights,cache)
        for name in ('wide.weight','middle.weight','small.weight'):
            original=self.weights.numpy(name)
            total_bytes=0
            for alias in cache.aliases_for(name):
                real,first,last=cache.binding(alias)
                arrays=executor._weight_arrays(alias,0,2)
                total_bytes+=sum(array.nbytes for array in arrays)
                restored=arrays[0].astype(np.float32)*arrays[1]
                if len(arrays)==3:restored-=arrays[2]
                physical=cache.layer(alias)['columns']
                restored=restored.reshape(2,physical)
                np.testing.assert_array_equal(restored[:,:last-first].copy().view(np.uint32),
                                              original[:,first:last].copy().view(np.uint32))
                np.testing.assert_array_equal(restored[:,last-first:].view(np.uint32),0)
                group=32 if self.weights.tensors[name].type_id==13 else 16
                for array in arrays:np.testing.assert_array_equal(array[:,(last-first)//group:,:],0)
                self.assertEqual(real,name)
            physical_columns=sum(cache.layer(alias)['columns'] for alias in cache.aliases_for(name))
            self.assertEqual(total_bytes,int(original.shape[0]*physical_columns*1.25))
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(),self.before)

    def test_prefill_decode_share_only_tile_fds_and_do_not_decode_or_multiply_on_host(self):
        p,d=self.cache(4),self.cache(1)
        native=self.native()
        self.compile(native,p);self.compile(native,d)
        execute=ColumnPhaseExecutor(native,self.weights,p,d)
        buffers={};keys=[]
        def cached(key,size):return key in buffers and buffers[key].nbytes==size
        def upload(record,index,array=None,*,cache_key):
            keys.append(cache_key)
            if array is not None:buffers[cache_key]=array.copy()
            native._write(f'U {record["native_id"]} {index}\n'.encode(),buffers[cache_key].tobytes())
            fields=native._readline().split()
            self.assertEqual(fields[:4],['OK','UPLOADED',record['native_id'],str(index)])
            return int(fields[4])
        x=np.random.default_rng(120).normal(0,.05,(5,9216)).astype(np.float32)
        expected=x@self.weights.numpy('wide.weight').T
        with patch.object(native,'weight_buffer_cached',side_effect=cached), \
             patch.object(native,'upload_weight',side_effect=upload), \
             patch.object(self.weights,'numpy_rows',side_effect=AssertionError('No CPU weight decoding')), \
             patch('grouped_columns.np.matmul',side_effect=AssertionError('No CPU weight matmul')), \
             patch('grouped_columns.np.dot',side_effect=AssertionError('No CPU weight dot')):
            np.testing.assert_allclose(execute('wide.weight',x,prefill=True),expected,rtol=3e-5,atol=3e-4)
            initial=set(keys)
            np.testing.assert_allclose(execute('wide.weight',x[:1]),expected[:1],rtol=3e-5,atol=3e-4)
            before=len(native.records)
            np.testing.assert_allclose(execute('wide.weight',x[:1],prefill=True),expected[:1],rtol=3e-5,atol=3e-4)
            self.assertEqual([(item['batch_vectors'],item['prefill']) for item in native.records[before:]],[(1,True)]*3)
        self.assertEqual(set(keys),initial)
        self.assertTrue(all('::cols[' in key[1] and key[-1]=='original_columns' for key in keys))
        self.assertEqual(execute.decode.weight_decodes,0)
        self.assertEqual(sum(array.nbytes for array in buffers.values()),int(2*9216*1.25))
        stats=execute.stats()
        self.assertEqual(stats['logical_projection_operations'],3)
        self.assertEqual(stats['column_partial_matmuls'],9)
        self.assertEqual(stats['cpu_partial_additions'],6)
        self.assertFalse(stats['cpu_fallback'])

    def test_second_tile_failure_closes_helper_and_returns_no_original_result(self):
        cache=self.cache(4);native=self.native();self.compile(native,cache)
        execute=ColumnPhaseExecutor(native,self.weights,cache,cache)
        original=native._write
        calls=0
        def fail_second(header,*payload):
            nonlocal calls
            if header.startswith(b'V '):
                calls+=1
                if calls==2:raise RuntimeError('injected second tile failure')
            return original(header,*payload)
        with patch.object(native,'_write',side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError,'second tile'):
                execute('wide.weight',np.ones((2,9216),np.float32),prefill=True)
        self.assertTrue(native.closed)
        self.assertEqual(native.invocations,1)
        self.assertEqual(execute.logical_projection_operations,0)
        self.assertEqual(execute.column_partial_matmuls,1)

    def test_nonfinite_host_partial_sum_stops_the_run(self):
        cache=self.cache(4);native=self.native()
        execute=ColumnPhaseExecutor(native,self.weights,cache,cache)
        huge=np.full((1,2),np.finfo(np.float32).max,np.float32)
        with patch('grouped_columns.GroupedPhaseExecutor.__call__',side_effect=lambda *a,**k:huge.copy()),np.errstate(over='ignore'):
            with self.assertRaisesRegex(ValueError,'Nonfinite F32'):
                execute('wide.weight',np.ones((1,9216),np.float32),prefill=True)
        self.assertTrue(native.closed)
        self.assertEqual(execute.logical_projection_operations,0)

    def test_plan_only_hashes_original_once_and_creates_no_package_or_native_worker(self):
        sha=GGUFWeights.sha256
        with patch.object(GGUFWeights,'sha256',autospec=True,side_effect=sha) as hashed, \
             patch('compile_grouped_columns.NPULinear',side_effect=AssertionError('No native planning')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(compile_grouped_columns.main(['--model',str(self.source),'--output',str(self.root/'plan-only'),
                '--bank-rows','2','--batch-vectors','4','--plan-only']),0)
        self.assertEqual(hashed.call_count,1)
        self.assertFalse((self.root/'plan-only').exists())

    def test_old_balanced_recipe_is_rejected_even_with_the_same_original_plan(self):
        cache=self.cache(4)
        manifest=cache._manifest()
        manifest['recipe']='original-groups-columns-v1-balanced-256-max3072-f32-partial-sum'
        cache.manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError,'recipe'):
            ColumnProgramCache(cache.directory,self.source,read_only=True,source_sha256=self.weights.sha256(),
                source_identity=self.weights.source_identity,verify_source=False,plan=self.plan,
                batch_vectors=4,fingerprint={'device':'fixture'})

    def test_split_tail_zero_padding_preserves_numerics_and_single_vector_phase(self):
        p,d=self.cache(4),self.cache(1)
        native=self.native()
        self.compile(native,p,'middle.weight');self.compile(native,d,'middle.weight')
        execute=ColumnPhaseExecutor(native,self.weights,p,d)
        x=np.random.default_rng(127).normal(0,.05,(5,4096)).astype(np.float32)
        expected=x@self.weights.numpy('middle.weight').T
        sent=[]
        write=native._write
        def capture(header,*data):
            if header.startswith(b'V '):
                values=np.frombuffer(data[0],dtype='<f4').reshape(-1,3072)
                sent.append(values.copy())
            return write(header,*data)
        with patch.object(native,'_write',side_effect=capture), \
             patch.object(self.weights,'numpy_rows',side_effect=AssertionError('No CPU weight decode')):
            np.testing.assert_allclose(execute('middle.weight',x,prefill=True),expected,rtol=3e-5,atol=3e-4)
            before=len(native.records)
            np.testing.assert_allclose(execute('middle.weight',x[:1],prefill=True),expected[:1],rtol=3e-5,atol=3e-4)
        # B4 sends two batches for each of two aliases; last alias's real
        # activation occupies only1024 columns and all extra columns are zero.
        for values in sent[2:4]:np.testing.assert_array_equal(values[:,1024:],0)
        np.testing.assert_array_equal(sent[-1][:,1024:],0)
        self.assertEqual([(r['batch_vectors'],r['prefill']) for r in native.records[before:]],[(1,True)]*2)
        stats=execute.stats()
        self.assertEqual(stats['logical_projection_operations'],2)
        self.assertEqual(stats['column_partial_matmuls'],4)
        self.assertEqual(stats['column_padded_partial_matmuls'],2)
        self.assertEqual(stats['column_padded_activation_values'],6*2048)

    def test_compiler_deduplicates_geometry_hashes_once_and_resumes_readonly(self):
        output=self.root/'compiled'
        argv=['--model',str(self.source),'--output',str(output),'--bank-rows','2','--batch-vectors','4',
              '--native-command-json',json.dumps([sys.executable,test_dynamic_fc.__file__,'--worker'])]
        sha=GGUFWeights.sha256
        with patch.object(GGUFWeights,'sha256',autospec=True,side_effect=sha) as hashed, \
             patch('compiled_model.device_fingerprint',return_value={'device':'fixture'}) as fingerprint, \
             patch.dict(os.environ,{'PIXELFORGE_NPU_STDERR':str(self.root/'compile.stderr')}), \
             contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(compile_grouped_columns.main(argv),0)
        self.assertEqual(hashed.call_count,1)
        fingerprint.assert_called_once_with()
        events=[json.loads(line) for line in log.getvalue().splitlines()]
        self.assertEqual(events[0]['column_partial_banks'],6)
        self.assertEqual(events[0]['geometry_programs'],3)
        self.assertEqual(events[-1]['compiled_geometries'],3)
        self.assertEqual(events[-1]['npu_invocations'],0)
        manifest=json.loads((output/'grouped-columns-v1/b4/MODEL.json').read_text())
        self.assertTrue(manifest['complete'])
        self.assertEqual(manifest['plan'],self.plan)
        with patch('compiled_model.device_fingerprint',return_value={'device':'fixture'}) as fingerprint, \
             patch('compile_grouped_columns.NPULinear',side_effect=AssertionError('Readonly resume must not start native')), \
             contextlib.redirect_stdout(io.StringIO()) as log:
            self.assertEqual(compile_grouped_columns.main(argv+['--resume','--plan-only']),0)
        self.assertEqual(json.loads(log.getvalue())['missing_geometries'],0)
        fingerprint.assert_called_once_with()


if __name__=='__main__':unittest.main()
