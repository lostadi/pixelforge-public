"""Offline native protocol/stage tests. Simulated bridge is not NPU evidence."""
from pathlib import Path
import json, os, resource, subprocess, sys, tempfile, unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/arm_npu'))
from create_tiny_gguf import create
from gguf_hf_loader import GGUFWeights
from model_architecture import build_plan, write_bridge_plan
ROOT = Path(__file__).resolve().parents[1]
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
 _,nl,c,r,v=line.split();nl,c,r,v=map(int,(nl,c,r,v))
 name=source.read(nl).decode()
 with open(os.environ['CHAT_TEST_SEEN'],'a') as out:out.write(name+'\\n')
 if os.environ.get('CHAT_TEST_FAIL')=='1':
  sink.write(b'ERR chosen projection failed\\n');sink.flush();break
 raw=source.read(v*c*4)
 activation=np.frombuffer(raw,dtype='<f4').reshape(v,c)
 output=activation@weights.numpy(name).T
 output=np.asarray(output,dtype='<f4')
 sink.write(('OK '+str(output.size)+'\\n').encode()+output.tobytes());sink.flush()
'''
@unittest.skipUnless((ROOT/'build/bin/pixelforge-chat').is_file(), 'Build the native runner first')
class NativeTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory()
  self.root=Path(self.temp.name)
  self.model=self.root/'tiny.gguf';create(self.model)
 def tearDown(self):self.temp.cleanup()
 def run_requests(self,requests,bridge=False,selection=None,fail=False,expect_success=True):
  env=os.environ.copy()
  for key in ('PIXELFORGE_GGML_PLAN','PIXELFORGE_GGML_BROKER','PIXELFORGE_GGML_NPU_TENSORS'):env.pop(key,None)
  if bridge:
   broker=self.root/'broker.py';broker.write_text(BROKER.replace('#!/usr/bin/python3', '#!'+sys.executable, 1));broker.chmod(0o700)
   plan=self.root/'weights.plan'
   with GGUFWeights(self.model) as weights:write_bridge_plan(build_plan(weights),plan)
   env.update(PIXELFORGE_GGML_PLAN=str(plan),PIXELFORGE_GGML_BROKER=str(broker),
              CHAT_TEST_MODEL=str(self.model),CHAT_TEST_CLEAN=str(self.root/'clean'),
              CHAT_TEST_KIT=str(ROOT.parent/'tools/arm_npu'),CHAT_TEST_SEEN=str(self.root/'seen'),CHAT_TEST_FAIL='1' if fail else '0')
   if selection is not None:env['PIXELFORGE_GGML_NPU_TENSORS']=selection
  result=subprocess.run([str(ROOT/'build/bin/pixelforge-chat'),str(self.model),'1'],
   input=''.join(json.dumps(r)+'\n' for r in requests),text=True,capture_output=True,env=env,timeout=20,
   preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_CORE,(0,0)))
  if not expect_success:return result
  self.assertEqual(result.returncode,0,result.stderr)
  return [json.loads(line) for line in result.stdout.splitlines()]
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
if __name__=='__main__':unittest.main()
