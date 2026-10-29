"""Installer provenance, compatibility, and atomic verification regressions."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup_local_model as setup


class ModelSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.runtime = Path(self.temp.name)
        self.runtime_patch = patch.object(setup, 'RUNTIME', self.runtime)
        self.runtime_patch.start()
        self.manifest = self.runtime / 'local-model.json'
        self.manifest.write_text('{"selected":"existing-lfm"}\n')
        self.original_manifest = self.manifest.read_bytes()

    def tearDown(self):
        self.runtime_patch.stop()
        self.temp.cleanup()

    def fixture_profile(self, payload=b'GGUF fixture'):
        profile = setup.PROFILES['minicpm5'].copy()
        profile['variants'] = {'Q8_0': {
            'filename': 'verified-mini.Q8_0.gguf',
            'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest(),
        }}
        return profile

    def test_legacy_lfm_manifest_has_unchanged_identity_and_defaults(self):
        value = setup.model_manifest('Q8_0', Path('/models/lfm.gguf'), Path('/bin/llama-server'))
        self.assertEqual(value['model_repo'], setup.MODEL_REPO)
        self.assertEqual(value['model_revision'], setup.MODEL_REVISION)
        self.assertEqual(value['model_sha256'], setup.VARIANTS['Q8_0']['sha256'])
        self.assertEqual(value['model_bytes'], 1246254720)
        self.assertEqual((value['threads'], value['threads_batch'], value['parallel'], value['context_per_worker']),
                         (2, 4, 2, 2048))
        self.assertNotIn('sampling', value)
        self.assertNotIn('chat_template_kwargs', value)
        self.assertEqual(setup.model_manifest('F16', Path('model'), Path('server'))['model_bytes'], 2343327360)

    def test_minicpm_manifest_pins_provenance_and_model_specific_sampling(self):
        value = setup.model_manifest('Q8_0', Path('model'), Path('server'), 'minicpm5')
        self.assertEqual(value['model_repo'], 'mradermacher/Huihui-MiniCPM5-2B-abliterated-GGUF')
        self.assertEqual(value['model_revision'], 'fc1e8318e85635d142e95a0f9977487717c40b58')
        self.assertEqual(value['provenance_repo'], 'huihui-ai/Huihui-MiniCPM5-2B-abliterated')
        self.assertEqual(value['provenance_revision'], 'aea962bb5558e659624eeefc8c1b9f418a3e6ffb')
        self.assertEqual(value['model_bytes'], 2679711680)
        self.assertEqual(value['model_sha256'], 'ffc8f95be9e7fab3c8d8a50c6634828433b31ca80a803349f47dc3b35aaaeeab')
        self.assertEqual(value['sampling'], {'temperature': 1.0, 'top_p': 0.95, 'min_p': 0.0})
        self.assertEqual(value['chat_template_kwargs'], {'enable_thinking': False})
        value['sampling']['min_p'] = 0.5
        self.assertEqual(setup.PROFILES['minicpm5']['sampling']['min_p'], 0.0)

    def test_spark_32k_profile_reuses_pinned_weights_and_serializes_runtime_settings(self):
        profile = setup.PROFILES['huihui-spark-32k']
        spark = setup.PROFILES['spark']
        for key in ('model_repo', 'model_revision', 'provenance_repo',
                    'provenance_revision', 'default_quantization', 'variants',
                    'sampling', 'chat_template_kwargs'):
            self.assertEqual(profile[key], spark[key], key)
        server = self.runtime / 'llama-server'
        server.write_text('#!/bin/sh\n')
        server.chmod(0o700)
        setup.activate('Q4_K_M', Path('/models/spark.gguf'), server, 'huihui-spark-32k')
        value = json.loads(self.manifest.read_text())
        self.assertEqual(value['model_name'], 'huihui-spark-32k')
        self.assertEqual(value['model_path'], '/models/spark.gguf')
        self.assertEqual(value['model_sha256'], spark['variants']['Q4_K_M']['sha256'])
        self.assertEqual(value['model_bytes'], spark['variants']['Q4_K_M']['bytes'])
        self.assertEqual((value['threads'], value['threads_batch'], value['parallel'],
                          value['context_per_worker']), (4, 4, 1, 32768))
        self.assertEqual((value['cache_type_k'], value['cache_type_v']), ('q8_0', 'q8_0'))
        self.assertEqual(value['sampling'], spark['sampling'])
        self.assertEqual(value['chat_template_kwargs'], {'enable_thinking': False})

    def test_existing_profiles_keep_runtime_defaults_without_cache_overrides(self):
        for model, profile in setup.PROFILES.items():
            if model in ('huihui-spark-32k', 'huihui-gemma4-8k'):
                continue
            with self.subTest(model=model):
                quantization = profile.get('default_quantization', 'Q8_0')
                value = setup.model_manifest(quantization, Path('model'), Path('server'), model)
                self.assertEqual((value['threads'], value['threads_batch'], value['parallel'],
                                  value['context_per_worker']), (2, 4, 2, 2048))
                self.assertNotIn('cache_type_k', value)
                self.assertNotIn('cache_type_v', value)

    def test_gemma_conversation_profile_keeps_verified_huihui_weights(self):
        base = setup.PROFILES['gemma4']
        value = setup.model_manifest('Q6_K', Path('model'), Path('server'), 'huihui-gemma4-8k')
        for key in ('model_name', 'model_repo', 'model_revision',
                    'provenance_repo', 'provenance_revision'):
            self.assertEqual(value[key], base[key])
        self.assertEqual(value['model_sha256'], base['variants']['Q6_K']['sha256'])
        self.assertEqual((value['parallel'], value['context_per_worker']), (1, 8192))
        self.assertEqual((value['cache_type_k'], value['cache_type_v']), ('q8_0', 'q8_0'))
        self.assertEqual(value['chat_template_kwargs'], {'enable_thinking': False})

    def test_verified_download_uses_selected_pinned_url_without_activation(self):
        payload = b'GGUF fixture'
        profile = self.fixture_profile(payload)
        with patch.dict(setup.PROFILES, {'minicpm5': profile}), \
                patch.object(setup.urllib.request, 'urlopen', return_value=io.BytesIO(payload)) as fetch:
            path = setup.download('Q8_0', 'minicpm5')
        self.assertEqual(path.read_bytes(), payload)
        expected_url = ('https://huggingface.co/' + profile['model_repo'] + '/resolve/'
                        + profile['model_revision'] + '/verified-mini.Q8_0.gguf')
        fetch.assert_called_once_with(expected_url, timeout=60)
        self.assertFalse(path.with_suffix('.gguf.part').exists())
        self.assertEqual(self.manifest.read_bytes(), self.original_manifest)

    def test_bad_checksum_cannot_install_or_change_active_model(self):
        profile = self.fixture_profile(b'expected')
        with patch.dict(setup.PROFILES, {'minicpm5': profile}), \
                patch.object(setup.urllib.request, 'urlopen', return_value=io.BytesIO(b'corrupt!')):
            with self.assertRaisesRegex(SystemExit, 'verification'):
                setup.download('Q8_0', 'minicpm5')
        self.assertFalse((self.runtime / 'models/verified-mini.Q8_0.gguf').exists())
        self.assertEqual(self.manifest.read_bytes(), self.original_manifest)

    def test_oversized_download_is_stopped_before_installation(self):
        profile = self.fixture_profile(b'small')
        with patch.dict(setup.PROFILES, {'minicpm5': profile}), \
                patch.object(setup.urllib.request, 'urlopen', return_value=io.BytesIO(b'too much content')):
            with self.assertRaisesRegex(SystemExit, 'exceeds'):
                setup.download('Q8_0', 'minicpm5')
        self.assertFalse((self.runtime / 'models/verified-mini.Q8_0.gguf').exists())
        self.assertEqual(self.manifest.read_bytes(), self.original_manifest)

    def test_existing_valid_file_is_verified_without_network(self):
        payload = b'verified'
        profile = self.fixture_profile(payload)
        destination = self.runtime / 'models/verified-mini.Q8_0.gguf'
        destination.parent.mkdir()
        destination.write_bytes(payload)
        with patch.dict(setup.PROFILES, {'minicpm5': profile}), patch.object(setup.urllib.request, 'urlopen') as fetch:
            self.assertEqual(setup.download('Q8_0', 'minicpm5'), destination)
        fetch.assert_not_called()

    def test_existing_corrupt_file_is_preserved_and_rejected(self):
        profile = self.fixture_profile(b'expected')
        destination = self.runtime / 'models/verified-mini.Q8_0.gguf'
        destination.parent.mkdir()
        destination.write_bytes(b'corrupt!')
        with patch.dict(setup.PROFILES, {'minicpm5': profile}), patch.object(setup.urllib.request, 'urlopen') as fetch:
            with self.assertRaisesRegex(SystemExit, 'refusing to overwrite'):
                setup.download('Q8_0', 'minicpm5')
        self.assertEqual(destination.read_bytes(), b'corrupt!')
        fetch.assert_not_called()

    def test_download_only_cli_never_builds_or_activates(self):
        argv = ['setup_local_model.py', '--model', 'minicpm5', '--download-only']
        with patch.object(sys, 'argv', argv), patch.object(setup.os, 'geteuid', return_value=10402), \
                patch.object(setup, 'build') as build, patch.object(setup, 'activate') as activate, \
                patch.object(setup, 'download') as download:
            setup.main()
        download.assert_called_once_with('Q8_0', 'minicpm5')
        build.assert_not_called()
        activate.assert_not_called()
        self.assertEqual(self.manifest.read_bytes(), self.original_manifest)

    def test_unsupported_mini_precision_is_rejected_before_build_or_download(self):
        argv = ['setup_local_model.py', '--model', 'minicpm5', '--quantization', 'F16']
        with patch.object(sys, 'argv', argv), patch.object(setup, 'build') as build, \
                patch.object(setup, 'download') as download:
            with self.assertRaisesRegex(SystemExit, 'minicpm5 supports only: Q8_0'):
                setup.main()
        build.assert_not_called()
        download.assert_not_called()

    def test_family_cli_uses_its_selected_precision_without_changing_active_model(self):
        expected = {'lfm2': 'Q8_0', 'minicpm5': 'Q8_0', 'gemma4': 'Q6_K',
                    'qwen35': 'Q8_0', 'granite41': 'Q6_K', 'llama32': 'Q6_K',
                    'neohorse': 'Q5_K', 'spark': 'Q4_K_M', 'huihui-spark-32k': 'Q4_K_M'}
        for model, quantization in expected.items():
            with self.subTest(model=model), patch.object(sys, 'argv', [
                    'setup_local_model.py', '--model', model, '--download-only']), \
                    patch.object(setup.os, 'geteuid', return_value=10402), \
                    patch.object(setup, 'build') as build, patch.object(setup, 'activate') as activate, \
                    patch.object(setup, 'download') as download:
                setup.main()
                download.assert_called_once_with(quantization, model)
                build.assert_not_called()
                activate.assert_not_called()
        self.assertEqual(self.manifest.read_bytes(), self.original_manifest)

    def test_selected_family_manifests_are_accepted_by_runtime_sampling_validation(self):
        from local_runtime import _model_options
        for model, profile in setup.PROFILES.items():
            with self.subTest(model=model):
                quantization = profile.get('default_quantization', 'Q8_0')
                manifest = setup.model_manifest(quantization, Path('model'), Path('server'), model)
                sampling, template = _model_options(manifest)
                self.assertTrue(manifest['provenance_repo'].startswith('huihui-ai/'))
                self.assertIn('abliterated', manifest['provenance_repo'])
                self.assertEqual(len(manifest['model_revision']), 40)
                self.assertEqual(len(manifest['provenance_revision']), 40)
                self.assertEqual(len(manifest['model_sha256']), 64)
                self.assertEqual(template, profile.get('chat_template_kwargs', {}))
                self.assertEqual(sampling, {'temperature': .4, **profile.get('sampling', {})})

    def test_six_new_downloads_fit_the_agreed_eighteen_gigabyte_budget(self):
        selected = ['gemma4', 'qwen35', 'granite41', 'llama32', 'neohorse', 'spark']
        total = 0
        for model in selected:
            profile = setup.PROFILES[model]
            variant = profile['variants'][profile.get('default_quantization', 'Q8_0')]
            total += variant['bytes']
        self.assertLessEqual(total, 18_000_000_000)


if __name__ == '__main__':
    unittest.main()
