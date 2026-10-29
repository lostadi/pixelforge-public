"""Build-cache patch identity checks; no native compilation or inference."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
REVISION = '254b177307b14d055bbe4c050c9c441a1e06d8a4'
PRISTINE_CPU = '''#include "ggml-backend.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "quants.h"
#include "ggml-threading.h"

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD && !params->use_ref) {
        ggml_compute_forward_fwht(params, dst);
        return;
    }

'''
PRISTINE_CMAKE = '''
    list (APPEND GGML_CPU_SOURCES
        ggml-cpu/ggml-cpu.c
        ggml-cpu/ggml-cpu.cpp
        ggml-cpu/repack.cpp
        ggml-cpu/repack.h
        ggml-cpu/hbm.cpp
'''


@unittest.skipUnless(all(shutil.which(name) for name in ('bash', 'patch', 'rg')),
                     'Native build cache checks require bash, patch, and rg')
class BuildCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='pixelforge-build-cache-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.chat = self.root / 'chat'
        self.tree = self.chat / 'build/source'
        self.cpu = self.tree / 'ggml/src/ggml-cpu/ggml-cpu.c'
        self.cpu.parent.mkdir(parents=True)
        self.cpu.write_text(PRISTINE_CPU)
        (self.cpu.parent / 'CMakeLists.txt').write_text(PRISTINE_CMAKE)
        (self.tree / 'CMakeLists.txt').write_text('# pinned source fixture\n')
        self.owner = self.tree / '.pixelforge-chat-source-owner'
        self.owner.write_text(REVISION + '\n')
        self.marker = self.tree / '.pixelforge-chat-patch-sha256'
        native = self.chat / 'native'
        native.mkdir()
        for name in ('ggml_npu_bridge.cc', 'ggml_npu_bridge.h'):
            (native / name).write_text('// native fixture\n')
        shutil.copy2(ROOT / 'build_native.sh', self.chat / 'build_native.sh')
        self.patch = self.root / 'tools/arm_npu/llama_npu_bridge.patch'
        self.patch.parent.mkdir(parents=True)
        shutil.copy2(ROOT.parent / 'tools/arm_npu/llama_npu_bridge.patch', self.patch)
        self.expected = hashlib.sha256(self.patch.read_bytes()).hexdigest()
        self.cmake_log = self.root / 'cmake.jsonl'
        self.patch_log = self.root / 'patch.jsonl'
        commands = self.root / 'commands'
        commands.mkdir()
        for name, body in {
            'cmake': '''import json,os,sys
from pathlib import Path
with Path(os.environ['BUILD_TEST_CMAKE_LOG']).open('a') as out:
    out.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[1]=='--build':
    output=Path(sys.argv[2])/'bin/pixelforge-chat'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_bytes(b'fixture, not a native executable')
''',
            'patch': '''import json,os,sys
from pathlib import Path
with Path(os.environ['BUILD_TEST_PATCH_LOG']).open('a') as out:
    out.write(json.dumps(sys.argv[1:])+'\\n')
if os.environ.get('BUILD_TEST_PATCH_FAIL')=='1':sys.exit(41)
os.execv(os.environ['BUILD_TEST_REAL_PATCH'],[os.environ['BUILD_TEST_REAL_PATCH'],*sys.argv[1:]])
''',
        }.items():
            command = commands / name
            command.write_text('#!' + sys.executable + '\n' + body)
            command.chmod(0o700)
        self.env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ.get('PATH', ''),
                        BUILD_TEST_CMAKE_LOG=str(self.cmake_log), BUILD_TEST_PATCH_LOG=str(self.patch_log),
                        BUILD_TEST_REAL_PATCH=shutil.which('patch'))
    def run_build(self):
        return subprocess.run(['bash', str(self.chat / 'build_native.sh'), '--source', '/unused/source'],
                              capture_output=True, text=True, env=self.env, timeout=10)
    def mark_patched(self):
        self.cpu.write_text(PRISTINE_CPU + '#include "ggml_npu_bridge.h"\n'
                            'const int n_fused = pixelforge_ggml_npu_enabled() ? 0 : old_fusion();\n')
    def assert_rejected_before_tools(self, result):
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.cmake_log.exists(), 'Stale source must never reach CMake')
        self.assertFalse(self.patch_log.exists(), 'Stale source must not be patched in place')
        self.assertFalse((self.cpu.parent / 'ggml_npu_bridge.cc').exists())
    def test_old_global_guard_without_marker_requires_source_regeneration(self):
        self.mark_patched()
        before = self.cpu.read_bytes()
        result = self.run_build()
        self.assert_rejected_before_tools(result)
        self.assertIn('patch SHA-256', result.stderr)
        self.assertIn('regenerate source', result.stderr)
        self.assertIn(str(self.tree), result.stderr)
        self.assertEqual(self.cpu.read_bytes(), before)
        self.assertFalse(self.marker.exists())
    def test_mismatched_patch_marker_rejects_without_mutating_cached_source(self):
        self.mark_patched()
        self.marker.write_text('0' * 64 + '\n')
        before = self.cpu.read_bytes()
        result = self.run_build()
        self.assert_rejected_before_tools(result)
        self.assertIn('regenerate source', result.stderr)
        self.assertEqual(self.cpu.read_bytes(), before)
        self.assertEqual(self.marker.read_text(), '0' * 64 + '\n')
    def test_pristine_source_applies_exact_patch_and_reuses_matching_marker(self):
        first = self.run_build()
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(self.marker.read_text(), self.expected + '\n')
        self.assertIn('ggml_npu_bridge.h', self.cpu.read_text())
        self.assertNotIn('pixelforge_ggml_npu_enabled() ? 0', self.cpu.read_text())
        calls = [json.loads(line) for line in self.patch_log.read_text().splitlines()]
        self.assertEqual(len(calls), 1)
        self.assertIn('--fuzz=0', calls[0])
        second = self.run_build()
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(len(self.patch_log.read_text().splitlines()), 1)
        self.assertEqual(len(self.cmake_log.read_text().splitlines()), 4)
    def test_failed_patch_does_not_publish_marker_or_start_cmake(self):
        self.env['BUILD_TEST_PATCH_FAIL'] = '1'
        result = self.run_build()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(self.patch_log.exists())
        self.assertFalse(self.marker.exists())
        self.assertFalse(self.cmake_log.exists())
    def test_wrong_source_owner_pin_fails_before_patch_or_cmake(self):
        self.owner.write_text('different upstream revision\n')
        self.assert_rejected_before_tools(self.run_build())


if __name__ == '__main__':
    unittest.main()
