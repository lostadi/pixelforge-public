"""Synthetic CPU-only flash attention and recurrent context-reset checks.

Use PIXELFORGE_CHAT_FLASH_TEST_RUNNER for an isolated relink. This suite never
opens an installed model, launches a broker, or invokes Android/vendor APIs.
"""
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'tools/arm_npu'))
from create_tiny_gguf import create as create_llama
from recurrent_fixture import create as create_qwen

RUNNER = Path(os.environ.get('PIXELFORGE_CHAT_FLASH_TEST_RUNNER',
              os.environ.get('PIXELFORGE_CHAT_TEST_RUNNER', ROOT / 'build/bin/pixelforge-chat')))


@unittest.skipUnless(RUNNER.is_file(), 'Build a native runner with flash attention metadata first')
class FlashAttentionTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix='pixelforge-cpu-flash-')
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)

    def execute(self, model, requests, setting=None):
        env = {key: value for key, value in os.environ.items() if not key.startswith('PIXELFORGE_')}
        env['PIXELFORGE_CHAT_PREFILL_BATCH'] = '16'
        if setting is not None:
            env['PIXELFORGE_CHAT_FLASH_ATTN'] = setting
        return subprocess.run([str(RUNNER), str(model), '1'],
            input=''.join(json.dumps(request) + '\n' for request in requests),
            capture_output=True, text=True, env=env, timeout=20,
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_CORE, (0, 0)))

    def completed(self, model, requests, setting=None):
        result = self.execute(model, requests, setting)
        self.assertEqual(result.returncode, 0, result.stderr)
        frames = [json.loads(line) for line in result.stdout.splitlines()]
        ready = frames[0]
        self.assertEqual(ready['type'], 'ready')
        self.assertEqual(ready['flash_attention'], setting or 'off')
        self.assertEqual(ready['attention_backend'], 'CPU')
        self.assertIsNone(ready['cpu_flash_attention_enabled'])
        self.assertFalse(ready['dense_npu_requested'])
        self.assertFalse([frame for frame in frames if frame['type'] == 'error'])
        done = [frame for frame in frames if frame['type'] == 'done']
        self.assertEqual(len(done), len(requests))
        for frame in done:
            self.assertEqual(frame['flash_attention'], setting or 'off')
            self.assertEqual(frame['attention_backend'], 'CPU')
            self.assertIs(type(frame['cpu_flash_attention_enabled']), bool)
            self.assertEqual(frame['npu_weight_matmul_operations'], 0)
            if setting in (None, 'off'):
                self.assertFalse(frame['cpu_flash_attention_enabled'])
        return done

    @staticmethod
    def requests():
        return [dict(id=str(index), prompt=prompt, max_tokens=6, context_size=size, prefill_cpu=True)
                for index, (prompt, size) in enumerate((('Hello', 256), ('different prompt', 256),
                    ('Hello', 256), ('Hello', 512), ('different prompt', 512), ('Hello', 256)))]

    def test_off_and_auto_preserve_greedy_tokens_and_full_memory_reset(self):
        for name, create in (('llama', create_llama), ('qwen35-hybrid', create_qwen)):
            with self.subTest(model=name):
                model = self.root / (name + '.gguf')
                create(model)
                requests = self.requests()
                off = self.completed(model, requests, 'off')
                automatic = self.completed(model, requests, 'auto')
                for actual, reference in zip(automatic, off):
                    self.assertEqual(actual['generated_tokens'], reference['generated_tokens'])
                    self.assertEqual(actual['text'], reference['text'])
                self.assertEqual([frame['context_reused'] for frame in automatic],
                                 [False, True, True, False, True, False])
                self.assertEqual([frame['context_initializations'] for frame in automatic], [1,1,1,2,2,3])
                for request, actual in zip(requests, automatic):
                    fresh = self.completed(model, [request], 'auto')[0]
                    self.assertEqual(actual['generated_tokens'], fresh['generated_tokens'])
                    self.assertEqual(actual['text'], fresh['text'])

    def test_unset_defaults_to_off(self):
        model = self.root / 'llama.gguf'
        create_llama(model)
        self.completed(model, self.requests()[:1])

    def test_invalid_environment_is_rejected_before_loading_any_model(self):
        for setting in ('', 'AUTO', 'true', ' auto', 'on ', '1'):
            with self.subTest(setting=setting):
                result = self.execute(self.root / 'does-not-exist.gguf', [], setting)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('"type":"ready"', result.stdout)
                self.assertIn('PIXELFORGE_CHAT_FLASH_ATTN', result.stderr)

    def test_request_cannot_override_trusted_environment(self):
        model = self.root / 'llama.gguf'
        create_llama(model)
        request = self.requests()[0]
        result = self.execute(model, [dict(request, flash_attention='on'), dict(request, id='valid')], 'off')
        self.assertEqual(result.returncode, 0, result.stderr)
        frames = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertTrue(any(frame['type'] == 'error' and 'Unknown request field' in frame['error'] for frame in frames))
        done = [frame for frame in frames if frame['type'] == 'done']
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]['flash_attention'], 'off')
        self.assertFalse(done[0]['cpu_flash_attention_enabled'])

    def test_forced_on_is_reported_or_returns_a_clear_context_error(self):
        model = self.root / 'hybrid.gguf'
        create_qwen(model)
        result = self.execute(model, self.requests()[:1], 'on')
        frames = [json.loads(line) for line in result.stdout.splitlines()]
        done = [frame for frame in frames if frame['type'] == 'done']
        if done:
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(done[0]['flash_attention'], 'on')
            self.assertTrue(done[0]['cpu_flash_attention_enabled'])
            self.assertEqual(done[0]['attention_backend'], 'CPU')
        else:
            errors = [frame.get('error', '') for frame in frames if frame['type'] == 'error']
            self.assertTrue(any('CPU flash attention on' in error for error in errors), result.stderr)


if __name__ == '__main__':
    unittest.main()
