"""Experimental FFN graph hook: synthetic CPU graphs and a simulated broker."""
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
GATE = 'blk.0.ffn_gate.weight'
UP = 'blk.0.ffn_up.weight'
DOWN = 'blk.0.ffn_down.weight'
BROKER = '''import json,os,sys
import numpy as np
source,sink=sys.stdin.buffer,sys.stdout.buffer
gate=((np.arange(32,dtype=np.int32)%13-6).astype(np.float32)/np.float32(17)).reshape(8,4)
up=((np.arange(32,dtype=np.int32)%11-5).astype(np.float32)/np.float32(19)).reshape(8,4)
down=((np.arange(32,dtype=np.int32)%7-3).astype(np.float32)/np.float32(23)).reshape(4,8)
weights={'blk.0.ffn_gate.weight':gate,'blk.0.ffn_up.weight':up,'blk.0.ffn_down.weight':down}
while True:
 line=source.readline()
 if line==b'Q\\n':break
 if not line:raise SystemExit('missing Q shutdown')
 op,n,c,r,v,phase=line.split();n,c,r,v=map(int,(n,c,r,v))
 name=source.read(n).decode()
 with open(os.environ['FFN_TEST_CALLS'],'a') as out:out.write(json.dumps([op.decode(),name,phase.decode()])+'\\n')
 if os.environ.get('FFN_TEST_FAIL')=='1':
  sink.write(b'ERR selected full FFN failed\\n');sink.flush();break
 raw=source.read(v*c*4)
 x=np.frombuffer(raw,dtype='<f4').reshape(v,c)
 if op==b'F':
  g=x@gate.T;u=x@up.T
  result=((g/(np.float32(1)+np.exp(-g)))*u)@down.T
 else:result=x@weights[name].T
 result=np.asarray(result,dtype='<f4')
 sink.write(('OK '+str(result.size)+'\\n').encode()+result.tobytes());sink.flush()
'''


class FullFFNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which(os.environ.get('CXX', 'c++'))
        source = ROOT / 'build/source'
        libraries = [ROOT / 'build/ggml/src' / name for name in
                     ('libggml.a', 'libggml-cpu.a', 'libggml-base.a')]
        if not compiler or not (source / 'ggml/include/ggml.h').is_file() or not all(p.is_file() for p in libraries):
            raise unittest.SkipTest('Build the pinned native source and static libraries first')
        cls.temp = tempfile.TemporaryDirectory(prefix='pixelforge-full-ffn-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.fixture = Path(cls.temp.name) / 'full-ffn-fixture'
        subprocess.run([compiler, '-std=c++17', '-O2', '-I' + str(source / 'ggml/include'),
                        '-I' + str(source / 'ggml/src'), '-I' + str(source / 'vendor'), '-I' + str(ROOT / 'native'),
                        str(ROOT / 'tests/full_ffn_fixture.cc'), *map(str, libraries),
                        '-pthread', '-ldl', '-lm', '-o', str(cls.fixture)],
                       check=True, capture_output=True, text=True, timeout=60)

    def run_fixture(self, variant='exact', *, phase='P', selection=None, full=True,
                    phases='P', bridge=True, fail=False):
        with tempfile.TemporaryDirectory(prefix='pixelforge-full-ffn-run-') as folder:
            root = Path(folder)
            broker = root / 'broker.py'
            broker.write_text('#!' + sys.executable + '\n' + BROKER)
            broker.chmod(0o700)
            plan = root / 'plan'
            plan.write_text(f'4 8 0 {GATE}\n4 8 0 {UP}\n8 4 0 {DOWN}\n')
            calls = root / 'calls.jsonl'
            env = {k: v for k, v in os.environ.items() if not k.startswith('PIXELFORGE_')}
            if bridge:
                env.update(PIXELFORGE_GGML_PLAN=str(plan), PIXELFORGE_GGML_BROKER=str(broker),
                           FFN_TEST_CALLS=str(calls), FFN_TEST_FAIL='1' if fail else '0')
                env['PIXELFORGE_GGML_NPU_TENSORS'] = ','.join(selection or [GATE, UP, DOWN])
                if full:
                    env['PIXELFORGE_GGML_FULL_FFN_TENSORS'] = GATE
                    env['PIXELFORGE_GGML_FULL_FFN_PHASES'] = phases
            result = subprocess.run([str(self.fixture), variant, phase], capture_output=True, text=True,
                                    env=env, timeout=15,
                                    preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))
            submitted = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
            if fail:
                return result, submitted
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout), submitted

    def assert_same_math(self, actual, reference):
        self.assertEqual(len(actual['output']), len(reference['output']))
        for value, expected in zip(actual['output'], reference['output']):
            self.assertAlmostEqual(value, expected, delta=1e-5)

    def test_full_ffn_skips_cpu_intermediates_and_matches_original_graph(self):
        actual, calls = self.run_fixture()
        reference, _ = self.run_fixture(bridge=False)
        self.assertTrue(actual['pattern'])
        self.assertTrue(actual['intermediates_untouched'])
        self.assertFalse(reference['intermediates_untouched'])
        self.assertEqual(calls, [['F', GATE, 'P']])
        self.assertEqual(actual['npu_weight_operations'], 3)
        self.assertEqual(actual['npu_weight_vectors'], 6)
        self.assertEqual(actual['npu_fused_ffn_operations'], 1)
        self.assert_same_math(actual, reference)

    def test_unsupported_topology_uses_selected_individual_npu_projections(self):
        for variant in ('consumer', 'view', 'output', 'gelu', 'swapped'):
            with self.subTest(variant=variant):
                actual, calls = self.run_fixture(variant)
                reference, _ = self.run_fixture(variant, bridge=False)
                self.assertFalse(actual['pattern'])
                self.assertFalse(actual['intermediates_untouched'])
                self.assertEqual([call[0] for call in calls], ['M'] * 3)
                self.assertEqual(actual['npu_fused_ffn_operations'], 0)
                self.assertEqual(actual['npu_weight_operations'], 3)
                self.assert_same_math(actual, reference)

    def test_partial_selection_does_not_override_explicit_cpu_placement(self):
        actual, calls = self.run_fixture(selection=[GATE, UP])
        reference, _ = self.run_fixture(bridge=False)
        self.assertTrue(actual['pattern'])
        self.assertEqual([call[0] for call in calls], ['M', 'M'])
        self.assertEqual(actual['npu_weight_operations'], 2)
        self.assertEqual(actual['npu_fused_ffn_operations'], 0)
        self.assert_same_math(actual, reference)

    def test_fusion_is_opt_in_and_phase_specific(self):
        for phase, full, phases, expected in [('P', False, 'P', 0), ('D', True, 'P', 0),
                                               ('D', True, 'PD', 1)]:
            with self.subTest(phase=phase, full=full, phases=phases):
                actual, calls = self.run_fixture(phase=phase, full=full, phases=phases)
                self.assertEqual(actual['npu_fused_ffn_operations'], expected)
                self.assertEqual([call[0] for call in calls], ['F'] if expected else ['M'] * 3)

    def test_first_selected_full_ffn_failure_aborts_without_individual_replay(self):
        result, calls = self.run_fixture(fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')
        self.assertEqual(calls, [['F', GATE, 'P']])
        self.assertIn('full FFN bridge failed', result.stderr)

    def test_hadamard_matmul_hint_is_not_reinterpreted_as_dense_fc(self):
        actual, calls = self.run_fixture('hint', bridge=False)
        self.assertFalse(actual['pattern'])
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
