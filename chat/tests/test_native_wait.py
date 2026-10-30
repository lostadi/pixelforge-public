"""NPU-hook rendezvous sleeps during delayed RPCs; no NPU or model required."""
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
BROKER = '''import os,struct,sys,time
source,sink=sys.stdin.buffer,sys.stdout.buffer
while True:
 line=source.readline()
 if line==b'Q\\n':break
 if not line:raise SystemExit('missing shutdown')
 op,n,c,r,v,phase=line.split();n,c,r,v=map(int,(n,c,r,v))
 name=source.read(n)
 raw=source.read(c*v*4)
 x=struct.unpack('<'+str(c*v)+'f',raw)
 with open(os.environ['WAIT_TEST_CALLS'],'a') as out:out.write(op.decode()+'\\n')
 time.sleep(.10)
 if op==b'F':
  import math
  x=tuple(value*value/(1+math.exp(-value)) for value in x)
 sink.write(('OK '+str(r*v)+'\\n').encode()+struct.pack('<'+str(len(x))+'f',*x));sink.flush()
'''


class NativeWaitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which(os.environ.get('CXX', 'c++'))
        source = ROOT / 'build/source'
        libraries = [ROOT / 'build/ggml/src' / name for name in
                     ('libggml.a', 'libggml-cpu.a', 'libggml-base.a')]
        if not compiler or not (source / 'ggml/include/ggml.h').is_file() or not all(p.is_file() for p in libraries):
            raise unittest.SkipTest('Build the pinned native source and static libraries first')
        cls.temp = tempfile.TemporaryDirectory(prefix='pixelforge-npu-wait-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.fixture = Path(cls.temp.name) / 'wait-fixture'
        # Compile the owning bridge source directly so this regression cannot
        # silently exercise an older bridge in a cached static library.
        bridge = Path(os.environ.get('PIXELFORGE_TEST_BRIDGE_SOURCE', ROOT / 'native/ggml_npu_bridge.cc'))
        subprocess.run([compiler, '-std=c++17', '-O2', '-I'+str(ROOT/'native'), '-I'+str(source/'ggml/include'),
                        '-I'+str(source/'ggml/src'), '-I'+str(source/'ggml/src/ggml-cpu'),
                        '-I'+str(source/'vendor'),
                        str(ROOT/'tests/npu_wait_fixture.cc'), str(bridge), *map(str, libraries),
                        '-pthread', '-ldl', '-lm', '-o', str(cls.fixture)],
                       check=True, capture_output=True, text=True, timeout=60)

    def run_fixture(self, mode):
        with tempfile.TemporaryDirectory(prefix='pixelforge-npu-wait-run-') as folder:
            root = Path(folder)
            broker = root/'broker.py'
            broker.write_text('#!'+sys.executable+'\n'+BROKER)
            broker.chmod(0o700)
            plan = root/'plan'
            plan.write_text(''.join(f'4 4 0 blk.0.ffn_{part}.weight\n' for part in ('gate','up','down')))
            calls = root/'calls'
            env = {key:value for key,value in os.environ.items() if not key.startswith('PIXELFORGE_')}
            env.update(PIXELFORGE_GGML_PLAN=str(plan), PIXELFORGE_GGML_BROKER=str(broker),
                       PIXELFORGE_GGML_FULL_FFN_TENSORS='blk.0.ffn_gate.weight',
                       PIXELFORGE_GGML_FULL_FFN_PHASES='PD', WAIT_TEST_CALLS=str(calls))
            result = subprocess.run([str(self.fixture),mode], env=env, capture_output=True, text=True,
                                    timeout=15, preexec_fn=lambda:resource.setrlimit(resource.RLIMIT_CORE,(0,0)))
            self.assertEqual(result.returncode,0,result.stderr)
            value = json.loads(result.stdout)
            submitted = calls.read_text().splitlines()
            self.assertEqual(len(submitted),8)
            self.assertGreaterEqual(value['wall_seconds'],.75)
            # Sleeping peers use almost no CPU. This generous ratio fails the
            # former spinning ggml_barrier even on a single-core CPU quota.
            self.assertLess(value['cpu_seconds'],value['wall_seconds']*.5,value)
            return value,submitted

    def test_matrix_hooks_block_workers_and_publish_repeated_results(self):
        value,calls = self.run_fixture('matrix')
        self.assertEqual(calls,['M']*8)
        self.assertEqual(value['npu_weight_operations'],8)

    def test_full_ffn_hooks_block_workers_without_executing_cpu_intermediates(self):
        value,calls = self.run_fixture('full')
        self.assertEqual(calls,['F']*8)
        self.assertEqual(value['npu_fused_ffn_operations'],8)
        self.assertEqual(value['npu_weight_operations'],24)

    def test_two_threadpools_with_different_sizes_never_share_an_epoch(self):
        value,calls = self.run_fixture('pools')
        self.assertEqual(calls.count('M'),4)
        self.assertEqual(calls.count('F'),4)
        self.assertEqual(value['npu_weight_operations'],16)

    def test_concurrent_single_thread_calls_without_a_pool_do_not_share_a_latch(self):
        value,calls = self.run_fixture('single')
        self.assertEqual(calls,['M']*8)
        self.assertEqual(value['npu_weight_operations'],8)


if __name__ == '__main__':
    unittest.main()
