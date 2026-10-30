"""Offline native protocol/stage tests. Simulated bridge is not NPU evidence."""
from pathlib import Path
import json, math, os, resource, shutil, subprocess, sys, tempfile, unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/arm_npu'))
from create_tiny_gguf import create
from gguf_hf_loader import GGUFWeights
from model_architecture import build_plan, write_bridge_plan
from recurrent_fixture import create as create_recurrent
ROOT = Path(__file__).resolve().parents[1]
RUNNER = Path(os.environ.get('PIXELFORGE_CHAT_TEST_RUNNER', ROOT/'build/bin/pixelforge-chat'))
BROKER = '''#!/usr/bin/python3
import json, os, sys
import numpy as np
sys.path.insert(0,os.environ['CHAT_TEST_KIT'])
from gguf_hf_loader import GGUFWeights
weights=GGUFWeights(os.environ['CHAT_TEST_MODEL'])
source,sink=sys.stdin.buffer,sys.stdout.buffer
while True:
 line=source.readline()
 if not line:raise SystemExit('missing shutdown')
 if line==b'Q\\n':
  open(os.environ['CHAT_TEST_CLEAN'],'w').write('clean')
  break
 op,nl,c,r,v,*phase=line.split();nl,c,r,v=map(int,(nl,c,r,v))
 name=source.read(nl).decode()
 with open(os.environ['CHAT_TEST_SEEN'],'a') as out:out.write(name+'\\n')
 with open(os.environ['CHAT_TEST_VECTORS'],'a') as out:out.write(str(v)+'\\n')
 with open(os.environ['CHAT_TEST_PHASES'],'a') as out:out.write((phase[0].decode() if phase else 'D')+'\\n')
 if os.environ.get('CHAT_TEST_FAIL')=='1':
  sink.write(b'ERR chosen projection failed\\n');sink.flush();break
 raw=source.read(v*c*4)
 activation=np.frombuffer(raw,dtype='<f4').reshape(v,c)
 if op==b'F':
  prefix=name.removesuffix('.ffn_gate.weight')
  gate=activation@weights.numpy(prefix+'.ffn_gate.weight').T
  up=activation@weights.numpy(prefix+'.ffn_up.weight').T
  output=((gate/(np.float32(1)+np.exp(-gate)))*up)@weights.numpy(prefix+'.ffn_down.weight').T
 else:output=activation@weights.numpy(name).T
 output=np.asarray(output,dtype='<f4')
 sink.write(('OK '+str(output.size)+'\\n').encode()+output.tobytes());sink.flush()
'''
@unittest.skipUnless(RUNNER.is_file(), 'Build the native runner first')
class NativeTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory()
  self.root=Path(self.temp.name)
  self.model=self.root/'tiny.gguf';create(self.model)
 def tearDown(self):self.temp.cleanup()
 def run_requests(self,requests,bridge=False,selection=None,fail=False,expect_success=True,disable_fusion=False,prefill_batch=None,prefill_selection=None,full_ffn=None):
  env=os.environ.copy()
  for key in ('PIXELFORGE_GGML_PLAN','PIXELFORGE_GGML_BROKER','PIXELFORGE_GGML_NPU_TENSORS','PIXELFORGE_GGML_PREFILL_NPU_TENSORS','PIXELFORGE_GGML_FULL_FFN_TENSORS','PIXELFORGE_GGML_FULL_FFN_PHASES','GGML_CPU_DISABLE_FUSION','PIXELFORGE_CHAT_PREFILL_BATCH'):env.pop(key,None)
  if disable_fusion:env['GGML_CPU_DISABLE_FUSION']='1'
  if prefill_batch is not None:env['PIXELFORGE_CHAT_PREFILL_BATCH']=str(prefill_batch)
  if bridge:
   broker=self.root/'broker.py';broker.write_text(BROKER.replace('#!/usr/bin/python3', '#!'+sys.executable, 1));broker.chmod(0o700)
   plan=self.root/'weights.plan'
   with GGUFWeights(self.model) as weights:write_bridge_plan(build_plan(weights),plan)
   env.update(PIXELFORGE_GGML_PLAN=str(plan),PIXELFORGE_GGML_BROKER=str(broker),
              CHAT_TEST_MODEL=str(self.model),CHAT_TEST_CLEAN=str(self.root/'clean'),
              CHAT_TEST_KIT=str(ROOT.parent/'tools/arm_npu'),CHAT_TEST_SEEN=str(self.root/'seen'),
              CHAT_TEST_VECTORS=str(self.root/'vectors'),CHAT_TEST_PHASES=str(self.root/'phases'),CHAT_TEST_FAIL='1' if fail else '0')
   if selection is not None:env['PIXELFORGE_GGML_NPU_TENSORS']=selection
   if prefill_selection is not None:env['PIXELFORGE_GGML_PREFILL_NPU_TENSORS']=prefill_selection
   if full_ffn is not None:env['PIXELFORGE_GGML_FULL_FFN_TENSORS']=full_ffn
  result=subprocess.run([str(RUNNER),str(self.model),'1'],
   input=''.join(json.dumps(r)+'\n' for r in requests),text=True,capture_output=True,env=env,timeout=20,
   preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_CORE,(0,0)))
  if not expect_success:return result
  self.assertEqual(result.returncode,0,result.stderr)
  frames=[json.loads(line) for line in result.stdout.splitlines()]
  for frame in frames:
   if frame['type']=='done':
    for key in ('context_init_seconds','context_reset_seconds','prefill_seconds','decode_seconds','generation_seconds','npu_bridge_seconds'):
     self.assertIs(type(frame[key]),float,key)
     self.assertTrue(math.isfinite(frame[key]),key)
     self.assertGreaterEqual(frame[key],0,key)
    self.assertAlmostEqual(frame['prefill_seconds']+frame['decode_seconds'],frame['generation_seconds'],places=6)
    self.assertIs(type(frame['context_reused']),bool)
    self.assertGreater(frame['context_initializations'],0)
    self.assertGreaterEqual(frame['npu_weight_matmul_vectors'],frame['npu_weight_matmul_operations'])
    self.assertEqual(frame['npu_prefill_weight_matmul_operations']+frame['npu_decode_weight_matmul_operations'],
                     frame['npu_weight_matmul_operations'])
  return frames
 def test_cpu_multiple_turns_context_reset_errors_and_stream(self):
  request={'id':'first','prompt':'<s>Hello','max_tokens':4,'context_size':256,'prefill_cpu':True}
  frames=self.run_requests([{'id':'invalid','prompt':'hi','max_tokens':0},request,dict(request,id='second')])
  self.assertEqual(frames[0]['type'],'ready')
  self.assertEqual(frames[1]['type'],'error')
  done=[f for f in frames if f['type']=='done']
  self.assertEqual([f['id'] for f in done],['first','second'])
  self.assertEqual(done[0]['generated_tokens'],done[1]['generated_tokens'])
  self.assertEqual(done[0]['text'],done[1]['text'])
  for result in done:
   self.assertEqual(''.join(f['text'] for f in frames if f['type']=='chunk' and f['id']==result['id']),result['text'])
   self.assertEqual(result['npu_weight_matmul_operations'],0)
   self.assertEqual(len(result['generated_tokens']),4)
 def test_simulated_bridge_prefill_stage_and_persistent_shutdown(self):
  requests=[{'id':str(i),'prompt':'Hello','max_tokens':4,'context_size':256,'prefill_cpu':prefill}
            for i,prefill in enumerate((True,True,False))]
  frames=self.run_requests(requests,bridge=True)
  done=[f for f in frames if f['type']=='done']
  self.assertEqual(len(done),3)
  for result in done:
   self.assertGreater(result['npu_weight_matmul_operations'],0)
   self.assertEqual(result['cpu_placed_weight_matmul_operations'],0)
   self.assertEqual(result['cpu_prefill_weight_matmul_operations']>0,result['prefill_cpu'])
  self.assertEqual(done[0]['generated_tokens'],done[1]['generated_tokens'])
  self.assertGreater(done[2]['npu_weight_matmul_operations'],done[0]['npu_weight_matmul_operations'])
  self.assertEqual((self.root/'clean').read_text(),'clean')
 def test_explicit_selection_only_offloads_chosen_weights(self):
  names=['blk.0.ffn_up.weight','blk.0.ffn_gate.weight','blk.0.ffn_down.weight']
  frames=self.run_requests([{'id':'placed','prompt':'Hello','max_tokens':4,
                            'context_size':256,'prefill_cpu':True}],bridge=True,selection=','.join(names))
  self.assertEqual(frames[0]['selected_npu_weight_tensors'],names)
  result=next(f for f in frames if f['type']=='done')
  self.assertEqual(result['selected_npu_weight_tensors'],names)
  self.assertGreater(result['cpu_placed_weight_matmul_operations'],0)
  self.assertGreater(result['cpu_prefill_weight_matmul_operations'],0)
  self.assertEqual(result['npu_weight_matmul_operations'],9)
  self.assertEqual(set((self.root/'seen').read_text().splitlines()),set(names))
 def test_invalid_placement_rejected_before_ready(self):
  for choice in ('',',','blk.0.ffn_up.weight,','unknown.weight',
                 'blk.0.ffn_up.weight,blk.0.ffn_up.weight'):
   with self.subTest(choice=choice):
    result=self.run_requests([],bridge=True,selection=choice,expect_success=False)
    self.assertNotEqual(result.returncode,0)
    self.assertNotIn('"type":"ready"',result.stdout)
    self.assertIn('selection',result.stderr)
 def test_chosen_npu_failure_aborts_without_cpu_fallback(self):
  result=self.run_requests([{'id':'failed','prompt':'Hello','max_tokens':4,
                            'context_size':256,'prefill_cpu':True}],bridge=True,
                           selection='blk.0.ffn_up.weight',fail=True,expect_success=False)
  self.assertNotEqual(result.returncode,0)
  self.assertNotIn('"type":"done"',result.stdout)
  self.assertIn('broker rejected projection',result.stderr)
  self.assertEqual((self.root/'seen').read_text().splitlines(),['blk.0.ffn_up.weight'])
 def test_cpu_and_simulated_npu_fusion_preserve_tokens_and_placement(self):
  requests=[{'id':str(i),'prompt':'Hello','max_tokens':4,'context_size':256,'prefill_cpu':prefill}
            for i,prefill in enumerate((True,False))]
  modes=[(False,None),(True,None),(True,'blk.0.ffn_up.weight')]
  for bridge,selection in modes:
   with self.subTest(bridge=bridge,selection=selection):
    fused=self.run_requests(requests,bridge=bridge,selection=selection)
    separate=self.run_requests(requests,bridge=bridge,selection=selection,disable_fusion=True)
    fused=[f for f in fused if f['type']=='done']
    separate=[f for f in separate if f['type']=='done']
    self.assertEqual(len(fused),len(requests))
    self.assertEqual(len(separate),len(requests))
    for actual,reference in zip(fused,separate):
     for key in ('text','generated_tokens','prompt_tokens','prefill_cpu','selected_npu_weight_tensors',
                 'npu_weight_matmul_operations','cpu_dynamic_matmul_operations',
                 'cpu_prefill_weight_matmul_operations','cpu_placed_weight_matmul_operations'):
      self.assertEqual(actual[key],reference[key],key)
 def test_context_reuse_full_reset_and_size_changes_match_fresh(self):
  for recurrent in (False,True):
   if recurrent:
    self.model=self.root/'hybrid.gguf';create_recurrent(self.model)
   requests=[{'id':str(i),'prompt':prompt,'max_tokens':6,'context_size':size,'prefill_cpu':True}
             for i,(prompt,size) in enumerate([('Hello',256),('different prompt',256),('Hello',256),
                                                ('Hello',512),('different prompt',512),('Hello',256)])]
   with self.subTest(recurrent=recurrent):
    done=[f for f in self.run_requests(requests,prefill_batch=4) if f['type']=='done']
    self.assertEqual([f['context_reused'] for f in done],[False,True,True,False,True,False])
    self.assertEqual([f['context_initializations'] for f in done],[1,1,1,2,2,3])
    for request,actual in zip(requests,done):
     fresh=next(f for f in self.run_requests([request],prefill_batch=4) if f['type']=='done')
     self.assertEqual(actual['generated_tokens'],fresh['generated_tokens'])
     self.assertEqual(actual['text'],fresh['text'])
     self.assertEqual(actual['context_init_seconds']==0,actual['context_reused'])
     self.assertEqual(actual['context_reset_seconds']==0,not actual['context_reused'])
 def test_invalid_request_discards_prior_context(self):
  request={'id':'first','prompt':'Hello','max_tokens':4,'context_size':256,'prefill_cpu':True}
  frames=self.run_requests([request,dict(request,id='invalid',max_tokens=0),dict(request,id='last')])
  done=[f for f in frames if f['type']=='done']
  self.assertEqual([f['context_reused'] for f in done],[False,False])
  self.assertEqual([f['context_initializations'] for f in done],[1,2])
  self.assertEqual(done[0]['generated_tokens'],done[1]['generated_tokens'])
 def test_npu_prefill_transports_full_batches_and_partial_tail(self):
  request={'id':'batch','prompt':'HelloWorld!','max_tokens':4,'context_size':256,'prefill_cpu':False}
  # Final-layer FFN sees only the selected logits rows; Q sees every prompt vector.
  done=next(f for f in self.run_requests([request],bridge=True,selection='blk.0.attn_q.weight',prefill_batch=4)
            if f['type']=='done')
  vectors=[int(line) for line in (self.root/'vectors').read_text().splitlines()]
  self.assertEqual(vectors,[4,4,3,1,1,1])
  self.assertEqual(done['npu_weight_matmul_operations'],len(vectors))
  self.assertEqual(done['npu_weight_matmul_vectors'],sum(vectors))
  self.assertEqual(done['prefill_batch_size'],4)
  self.assertEqual((self.root/'phases').read_text().splitlines(),['P','P','P','D','D','D'])
  self.assertEqual(done['cpu_prefill_weight_matmul_operations'],0)
  self.assertGreater(done['cpu_placed_weight_matmul_operations'],0)
  reference=next(f for f in self.run_requests([request],bridge=True,selection='blk.0.attn_q.weight',prefill_batch=1)
                if f['type']=='done')
  self.assertEqual(done['generated_tokens'],reference['generated_tokens'])
 def test_distinct_prefill_decode_placement_and_phase_counters(self):
  request={'id':'phases','prompt':'Hello','max_tokens':4,'context_size':256,'prefill_cpu':False}
  prefill='blk.0.attn_q.weight,blk.0.attn_k.weight'
  done=next(f for f in self.run_requests([request],bridge=True,selection='blk.0.ffn_up.weight',prefill_selection=prefill)
            if f['type']=='done')
  self.assertEqual(done['selected_npu_weight_tensors'],['blk.0.ffn_up.weight'])
  self.assertEqual(done['selected_prefill_npu_weight_tensors'],prefill.split(','))
  self.assertEqual(done['npu_prefill_weight_matmul_operations'],2)
  self.assertEqual(done['npu_decode_weight_matmul_operations'],3)
  self.assertEqual(done['npu_weight_matmul_vectors'],13)
  self.assertEqual((self.root/'seen').read_text().splitlines(),prefill.split(',')+['blk.0.ffn_up.weight']*3)
  self.assertEqual((self.root/'phases').read_text().splitlines(),['P','P','D','D','D'])
 def test_invalid_prefill_placement_rejected_before_ready(self):
  for choice in ('',',','unknown.weight','blk.0.ffn_up.weight,blk.0.ffn_up.weight'):
   with self.subTest(choice=choice):
    result=self.run_requests([],bridge=True,prefill_selection=choice,expect_success=False)
    self.assertNotEqual(result.returncode,0)
    self.assertNotIn('"type":"ready"',result.stdout)
    self.assertIn('selection',result.stderr)
 def test_invalid_prefill_batch_is_rejected_before_model_load(self):
  for value in ('',0,513,'1x',' 4','+4'):
   with self.subTest(value=value):
    result=self.run_requests([],prefill_batch=value,expect_success=False)
    self.assertNotEqual(result.returncode,0)
    self.assertNotIn('"type":"ready"',result.stdout)
    self.assertIn('PIXELFORGE_CHAT_PREFILL_BATCH',result.stderr)
 def test_intermediate_prefill_chunks_skip_lm_head_and_match_single_chunk(self):
  request={'id':'logits','prompt':'HelloWorld!','max_tokens':4,'context_size':256,'prefill_cpu':False}
  for recurrent in (False,True):
   if recurrent:
    self.model=self.root/'hybrid.gguf';create_recurrent(self.model)
   with self.subTest(recurrent=recurrent):
    for name in ('seen','vectors','phases'):(self.root/name).unlink(missing_ok=True)
    done=next(f for f in self.run_requests([request],bridge=True,selection='token_embd.weight',prefill_batch=4)
              if f['type']=='done')
    self.assertEqual((self.root/'phases').read_text().splitlines(),['P','D','D','D'])
    self.assertEqual((self.root/'vectors').read_text().splitlines(),['1','1','1','1'])
    self.assertEqual(done['npu_prefill_weight_matmul_operations'],1)
    self.assertEqual(done['prefill_chunks'],3)
    self.assertEqual(done['prefill_logits_chunks_skipped'],2)
    reference=next(f for f in self.run_requests([request],bridge=True,selection='token_embd.weight',prefill_batch=512)
                   if f['type']=='done')
    self.assertEqual(done['generated_tokens'],reference['generated_tokens'])
    self.assertEqual(done['text'],reference['text'])
 def test_full_ffn_hook_preserves_complete_model_turn_and_logical_counts(self):
  request={'id':'full-ffn','prompt':'HelloWorld!','max_tokens':4,'context_size':256,'prefill_cpu':False}
  selection='blk.0.ffn_gate.weight,blk.0.ffn_up.weight,blk.0.ffn_down.weight'
  fused=next(f for f in self.run_requests([request],bridge=True,selection=selection,full_ffn='blk.0.ffn_gate.weight',prefill_batch=4)
             if f['type']=='done')
  reference=next(f for f in self.run_requests([request],bridge=True,selection=selection,prefill_batch=4) if f['type']=='done')
  self.assertEqual(fused['npu_fused_ffn_operations'],1)
  self.assertEqual(fused['npu_prefill_fused_ffn_operations'],1)
  self.assertEqual(fused['npu_decode_fused_ffn_operations'],0)
  self.assertEqual(reference['npu_fused_ffn_operations'],0)
  self.assertEqual(fused['full_ffn_tensors'],['blk.0.ffn_gate.weight'])
  self.assertEqual(fused['full_ffn_phases'],'P')
  self.assertEqual(fused['npu_weight_matmul_operations'],reference['npu_weight_matmul_operations'])
  self.assertEqual(fused['npu_weight_matmul_vectors'],reference['npu_weight_matmul_vectors'])
  self.assertEqual(fused['generated_tokens'],reference['generated_tokens'])
  self.assertEqual(fused['text'],reference['text'])
 def test_first_full_ffn_failure_aborts_native_worker_before_any_done(self):
  selection='blk.0.ffn_gate.weight,blk.0.ffn_up.weight,blk.0.ffn_down.weight'
  result=self.run_requests([{'id':'failed','prompt':'Hello','max_tokens':4,'context_size':256,'prefill_cpu':False}],
                           bridge=True,selection=selection,full_ffn='blk.0.ffn_gate.weight',fail=True,expect_success=False)
  self.assertNotEqual(result.returncode,0)
  self.assertNotIn('"type":"done"',result.stdout)
  self.assertIn('full FFN bridge failed',result.stderr)
 def test_invalid_full_ffn_profile_is_rejected_before_ready(self):
  for choice in ('','unknown.weight','blk.0.ffn_up.weight',
                 'blk.0.ffn_gate.weight,blk.0.ffn_gate.weight'):
   with self.subTest(choice=choice):
    result=self.run_requests([],bridge=True,full_ffn=choice,expect_success=False)
    self.assertNotEqual(result.returncode,0)
    self.assertNotIn('"type":"ready"',result.stdout)
    self.assertIn('selection',result.stderr)


class FusionGraphTests(unittest.TestCase):
 """Small compiled graph proves fusion runs even while the bridge is enabled."""
 @classmethod
 def setUpClass(cls):
  compiler=shutil.which(os.environ.get('CXX','c++'))
  includes=ROOT/'build/source/ggml/include'
  libraries=[ROOT/'build/ggml/src'/name for name in ('libggml.a','libggml-cpu.a','libggml-base.a')]
  if not compiler or not (includes/'ggml.h').is_file() or not all(p.is_file() for p in libraries):
   raise unittest.SkipTest('Build the pinned native source and static libraries first')
  cls.temp=tempfile.TemporaryDirectory(prefix='pixelforge-fusion-test-')
  cls.addClassCleanup(cls.temp.cleanup)
  cls.fixture=Path(cls.temp.name)/'fusion-fixture'
  subprocess.run([compiler,'-std=c++17','-O2','-I'+str(includes),str(ROOT/'tests/fusion_fixture.cc'),
                  *map(str,libraries),'-pthread','-ldl','-lm','-o',str(cls.fixture)],
                 check=True,capture_output=True,text=True,timeout=60)
 def run_graph(self,bridge=False,disable_fusion=False):
  env=os.environ.copy()
  for key in ('PIXELFORGE_GGML_PLAN','PIXELFORGE_GGML_BROKER','PIXELFORGE_GGML_NPU_TENSORS','GGML_CPU_DISABLE_FUSION'):
   env.pop(key,None)
  # This graph has no matmul: enabling the bridge must not launch any broker.
  if bridge:env['PIXELFORGE_GGML_BROKER']='/nonexistent/unused-test-broker'
  if disable_fusion:env['GGML_CPU_DISABLE_FUSION']='1'
  result=subprocess.run([str(self.fixture)],capture_output=True,text=True,env=env,timeout=10)
  self.assertEqual(result.returncode,0,result.stderr)
  return json.loads(result.stdout)
 def test_bridge_keeps_norm_scale_fusion_and_matches_unfused_output(self):
  for bridge in (False,True):
   with self.subTest(bridge=bridge):
    fused=self.run_graph(bridge=bridge)
    separate=self.run_graph(bridge=bridge,disable_fusion=True)
    self.assertTrue(fused['fused'],'RMS_NORM + MUL was split into separate kernels')
    self.assertFalse(separate['fused'],'Reference must execute the separate kernels')
    self.assertEqual(len(fused['output']),64)
    self.assertEqual(len(separate['output']),64)
    for actual,reference in zip(fused['output'],separate['output']):
     self.assertAlmostEqual(actual,reference,delta=1e-6)
if __name__=='__main__':unittest.main()
