"""Token-exact prompt state reuse; tiny Llama/Qwen35, no real NPU or disk cache."""
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'tools/arm_npu'))
from create_tiny_gguf import create as create_llama
from recurrent_fixture import create as create_qwen
from gguf_hf_loader import GGUFWeights
from model_architecture import build_plan,write_bridge_plan
from test_native import BROKER

RUNNER=Path(os.environ.get('PIXELFORGE_CHAT_PREFIX_TEST_RUNNER',os.environ.get('PIXELFORGE_CHAT_TEST_RUNNER',ROOT/'build/bin/pixelforge-chat')))


@unittest.skipUnless(RUNNER.is_file(),'Build the prefix-cache native runner first')
class PrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler=shutil.which(os.environ.get('CXX','c++'))
        source=ROOT/'build/source'
        libraries=[ROOT/'build/src/libllama.a']+[ROOT/'build/ggml/src'/name for name in
            ('libggml.a','libggml-cpu.a','libggml-base.a')]
        if not compiler or not all(path.is_file() for path in libraries):
            raise unittest.SkipTest('Build pinned static libraries for state fault fixtures')
        cls.work=tempfile.TemporaryDirectory(prefix='pixelforge-prefix-state-')
        cls.addClassCleanup(cls.work.cleanup)
        cls.fixture=Path(cls.work.name)/'prefix-state'
        subprocess.run([compiler,'-std=c++17','-O2','-DGGML_USE_CPU','-I'+str(ROOT/'native'),
            '-I'+str(source/'include'),'-I'+str(source/'src'),'-I'+str(source/'ggml/include'),
            '-I'+str(source/'vendor'),str(ROOT/'tests/prefix_state_fixture.cc'),*map(str,libraries),
            '-pthread','-ldl','-lm','-o',str(cls.fixture)],check=True,capture_output=True,text=True,timeout=60)

    def setUp(self):
        self.work=tempfile.TemporaryDirectory(prefix='pixelforge-prefix-run-')
        self.addCleanup(self.work.cleanup)
        self.root=Path(self.work.name)

    @staticmethod
    def requests():
        prompts=['a'*64+'x'*140,'a'*64+'y'*140,'b'*64+'z'*140,'a'*64+'x'*140]
        return [dict(id=str(i),prompt=text,max_tokens=4,context_size=256,prefill_cpu=True) for i,text in enumerate(prompts)]

    def run_model(self,model,requests,*,budget=1,bridge=False,fixture=None):
        env={key:value for key,value in os.environ.items() if not key.startswith('PIXELFORGE_')}
        env.update(PIXELFORGE_CHAT_PREFIX_CACHE_MIB=str(budget),PIXELFORGE_CHAT_PREFILL_BATCH='16')
        if bridge:
            plan=self.root/'plan'
            with GGUFWeights(model) as weights:write_bridge_plan(build_plan(weights),plan)
            broker=self.root/'broker.py'
            broker.write_text(BROKER.replace('#!/usr/bin/python3','#!'+sys.executable,1));broker.chmod(0o700)
            env.update(PIXELFORGE_GGML_PLAN=str(plan),PIXELFORGE_GGML_BROKER=str(broker),
                PIXELFORGE_GGML_NPU_TENSORS='blk.0.attn_q.weight',CHAT_TEST_MODEL=str(model),
                CHAT_TEST_CLEAN=str(self.root/'clean'),CHAT_TEST_KIT=str(ROOT.parent/'tools/arm_npu'),
                CHAT_TEST_SEEN=str(self.root/'seen'),CHAT_TEST_VECTORS=str(self.root/'vectors'),
                CHAT_TEST_PHASES=str(self.root/'phases'),CHAT_TEST_FAIL='0')
        command=[str(self.fixture),fixture,str(model)] if fixture else [str(RUNNER),str(model),'1']
        result=subprocess.run(command,input=''.join(json.dumps(item)+'\n' for item in requests),
            text=True,capture_output=True,env=env,timeout=30,
            preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_CORE,(0,0)))
        self.assertEqual(result.returncode,0,result.stderr)
        frames=[json.loads(line) for line in result.stdout.splitlines()]
        done=[frame for frame in frames if frame.get('type')=='done']
        for value in done:
            self.assertEqual(value['prefix_cache_hit'],value['prefix_tokens_reused']>0)
            self.assertLess(value['prefix_tokens_reused'],value['prompt_tokens'])
            self.assertEqual(value['prefill_tokens_processed'],value['prompt_tokens']-value['prefix_tokens_reused'])
            self.assertEqual(value['prefix_cache_mib'],budget)
            self.assertLessEqual(value['prefix_cache_bytes'],budget*1024**2)
        return frames,done,result

    def test_token_exact_prefixes_match_fresh_llama_and_hybrid_qwen(self):
        for name,create in (('llama',create_llama),('qwen35',create_qwen)):
            with self.subTest(model=name):
                model=self.root/(name+'.gguf');create(model)
                requests=self.requests()
                _,actual,_=self.run_model(model,requests)
                _,fresh,_=self.run_model(model,requests,budget=0)
                self.assertEqual([item['prefix_cache_hit'] for item in actual],[False,True,False,True])
                self.assertEqual([item['prefix_tokens_reused'] for item in actual],[0,64,0,64])
                for value,reference in zip(actual,fresh):
                    self.assertEqual(value['generated_tokens'],reference['generated_tokens'])
                    self.assertEqual(value['text'],reference['text'])
                for item in fresh:
                    self.assertFalse(item['prefix_cache_hit'])
                    self.assertEqual(item['prefix_cache_bytes'],0)
                    self.assertEqual(item['prefix_cache_entries'],0)

    def test_size_change_and_invalid_request_erase_snapshots(self):
        model=self.root/'llama.gguf';create_llama(model)
        first=self.requests()[0]
        requests=[first,dict(first,id='size512',context_size=512),dict(first,id='size256'),
                  dict(first,id='invalid',max_tokens=0),dict(first,id='after-invalid')]
        frames,done,_=self.run_model(model,requests)
        self.assertTrue(any(item['type']=='error' for item in frames))
        self.assertEqual([item['prefix_cache_hit'] for item in done],[False]*4)
        self.assertEqual([item['context_initializations'] for item in done],[1,2,3,4])

    def test_longest_checkpoint_and_only_two_selected_native_boundaries(self):
        model=self.root/'llama.gguf';create_llama(model)
        request=dict(id='a',prompt='a'*1240,max_tokens=3,context_size=2048,prefill_cpu=True)
        _,done,_=self.run_model(model,[request,dict(request,id='b')],budget=2)
        self.assertEqual(done[0]['prefix_cache_entries'],2)
        self.assertEqual(done[1]['prefix_tokens_reused'],1104)
        _,fresh,_=self.run_model(model,[request],budget=0)
        self.assertEqual(done[1]['generated_tokens'],fresh[0]['generated_tokens'])

    def test_cache_never_crosses_prefill_cpu_placement(self):
        model=self.root/'llama.gguf';create_llama(model)
        request=self.requests()[0]
        _,done,_=self.run_model(model,[request,dict(request,id='cpu-mode-change',prefill_cpu=False),
                                      dict(request,id='same-mode',prefill_cpu=False)])
        self.assertEqual([item['prefix_cache_hit'] for item in done],[False,False,True])

    def test_simulated_npu_reuses_fewer_actual_vectors_but_still_prefills_tail(self):
        model=self.root/'llama.gguf';create_llama(model)
        requests=[dict(request,prefill_cpu=False) for request in self.requests()[:2]]
        _,done,_=self.run_model(model,requests,bridge=True)
        self.assertGreater(done[1]['npu_prefill_weight_matmul_operations'],0)
        self.assertEqual(done[0]['npu_weight_matmul_vectors']-done[1]['npu_weight_matmul_vectors'],64)
        self.assertTrue(done[1]['prefix_cache_hit'])
        _,fresh,_=self.run_model(model,requests,budget=0,bridge=True)
        self.assertEqual([item['generated_tokens'] for item in done],[item['generated_tokens'] for item in fresh])

    def test_state_restore_and_snapshot_failures_never_continue_partial_state(self):
        for name,create in (('llama',create_llama),('qwen35',create_qwen)):
            model=self.root/(name+'.gguf');create(model)
            requests=self.requests()[:2]
            _,fresh,_=self.run_model(model,requests,budget=0)
            for mode in ('short-set','wrong-position','short-get'):
                with self.subTest(model=name,mode=mode):
                    _,done,result=self.run_model(model,requests,fixture=mode)
                    self.assertEqual([item['prefix_cache_hit'] for item in done],[False,False])
                    self.assertEqual([item['generated_tokens'] for item in done],[item['generated_tokens'] for item in fresh])
                    self.assertRegex(result.stderr,r'PREFIX_FIXTURE_FAULTS [1-9]')

    def test_bounded_lru_eviction_and_undersized_budget_with_real_state(self):
        for name,create in (('llama',create_llama),('qwen35',create_qwen)):
            with self.subTest(model=name):
                model=self.root/(name+'.gguf');create(model)
                frames,_,_=self.run_model(model,[],fixture='eviction')
                self.assertTrue(frames[0]['eviction'])
                self.assertEqual(frames[0]['undersized_entries'],0)

    def test_invalid_trusted_budget_is_rejected_before_model_loading(self):
        for value in ('','-1','+1','513','1x',' 1'):
            with self.subTest(value=value):
                env={key:item for key,item in os.environ.items() if not key.startswith('PIXELFORGE_')}
                env['PIXELFORGE_CHAT_PREFIX_CACHE_MIB']=value
                result=subprocess.run([str(RUNNER),str(self.root/'missing.gguf'),'1'],env=env,
                    capture_output=True,text=True,timeout=10)
                self.assertNotEqual(result.returncode,0)
                self.assertIn('PIXELFORGE_CHAT_PREFIX_CACHE_MIB',result.stderr)
                self.assertNotIn('"type":"ready"',result.stdout)


if __name__=='__main__':unittest.main()
