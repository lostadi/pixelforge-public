"""Offline planner tests; never downloads, compiles, or loads a model."""
import argparse
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import plan_huihui

SHA = 'a' * 40


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / 'checkpoint'
        self.checkpoint.mkdir()
        (self.checkpoint / 'config.json').write_text(json.dumps({'model_type': 'qwen3', 'architectures': ['Qwen3ForCausalLM']}))
        (self.checkpoint / 'model.safetensors').write_bytes(b'not loaded by planner')
        (self.checkpoint / 'tokenizer.json').write_text('{}')

    def tearDown(self):
        self.temp.cleanup()

    def test_requires_huihui_namespace_and_exact_sha(self):
        for value in ('google/gemma-4-E2B-it', 'https://huggingface.co/huihui-ai/example', 'huihui-ai/../model'):
            with self.assertRaises(argparse.ArgumentTypeError):
                plan_huihui.validate_repo(value)
        for value in ('main', 'abcd', 'g' * 40):
            with self.assertRaises(argparse.ArgumentTypeError):
                plan_huihui.validate_revision(value)
        self.assertEqual(plan_huihui.validate_revision('A' * 40), SHA)

    def test_missing_checkpoint_and_revision_are_explicit_blockers(self):
        plan = plan_huihui.build_plan('huihui-ai/example', None, None, None)
        self.assertIn('checkpoint_not_supplied', plan['blockers'])
        self.assertIn('exact_source_repository_revision_required', plan['blockers'])
        self.assertFalse(plan['conversion_candidate']['command_runnable'])
        self.assertFalse(plan['would_execute'])

    def test_gguf_only_missing_config_and_unknown_architecture_are_blocked(self):
        (self.checkpoint / 'model.safetensors').unlink()
        (self.checkpoint / 'model.gguf').write_bytes(b'GGUF')
        (self.checkpoint / 'config.json').unlink()
        result = plan_huihui.inspect_checkpoint(self.checkpoint)
        self.assertIn('config_json_missing_invalid_or_symlink', result['blockers'])
        self.assertIn('local_hf_safetensors_required_GGUF_is_not_exporter_input', result['blockers'])
        (self.checkpoint / 'config.json').write_text('{"model_type":"unrecognized"}')
        self.assertIn('architecture_not_in_documented_exporter_model_types', plan_huihui.inspect_checkpoint(self.checkpoint)['blockers'])

    def test_symlink_weights_and_shard_index_cannot_escape_directory(self):
        (self.checkpoint / 'model.safetensors').unlink()
        (self.root / 'outside.safetensors').write_bytes(b'private')
        (self.checkpoint / 'model.safetensors').symlink_to(self.root / 'outside.safetensors')
        (self.checkpoint / 'model.safetensors.index.json').write_text('{"weight_map":{"weight":"../outside.safetensors"}}')
        result = plan_huihui.inspect_checkpoint(self.checkpoint)
        self.assertEqual(result['weights'], [])
        self.assertIn('safetensors_index_references_missing_or_nonregular_shards', result['blockers'])

    def test_source_pin_is_metadata_only_and_shell_text_is_never_evaluated(self):
        output = self.root / '$(touch injected); output'
        plan = plan_huihui.build_plan('huihui-ai/example', SHA, self.checkpoint, output)
        self.assertEqual(plan['status'], 'uncompiled/unvalidated')
        self.assertEqual(plan['source_provenance']['source_url'], 'https://huggingface.co/huihui-ai/example/tree/' + SHA)
        self.assertFalse(plan['source_provenance']['weight_checksums_verified'])
        self.assertIn('--output_dir=' + str(output), plan['conversion_candidate']['argv_display_only'])
        self.assertFalse(output.exists())
        self.assertFalse((self.root / 'injected').exists())
        self.assertIn('--aot_soc_model=Tensor_G5', plan['conversion_candidate']['argv_display_only'])
        self.assertFalse(plan['conversion_candidate']['trust_remote_code'])
        literal = next(value.split('=', 1)[1] for value in plan['conversion_candidate']['argv_display_only']
                       if value.startswith('--aot_compilation_config_dict='))
        self.assertEqual(ast.literal_eval(literal), {'google_tensor_enable_large_model_support': True})

    def test_remote_code_metadata_is_blocked_and_package_detection_does_not_import_models(self):
        (self.checkpoint / 'config.json').write_text('{"model_type":"qwen3","auto_map":{"AutoModel":"custom.Evil"}}')
        self.assertIn('auto_map_present_remote_model_code_will_not_be_loaded', plan_huihui.inspect_checkpoint(self.checkpoint)['blockers'])
        with mock.patch('plan_huihui.importlib.metadata.version', side_effect=plan_huihui.importlib.metadata.PackageNotFoundError):
            host = plan_huihui.inspect_host()
        self.assertTrue(all(version is None for version in host['python_packages'].values()))

    def test_saved_plan_is_exclusive_and_does_not_create_compiler_destination(self):
        save, destination = self.root / 'plan.json', self.root / 'compiled'
        args = ['--repo', 'huihui-ai/example', '--revision', SHA, '--checkpoint', str(self.checkpoint),
                '--output-dir', str(destination), '--save', str(save)]
        self.assertEqual(plan_huihui.main(args), 0)
        self.assertEqual(save.stat().st_mode & 0o777, 0o600)
        self.assertFalse(destination.exists())
        original = save.read_bytes()
        with self.assertRaises(FileExistsError):
            plan_huihui.main(args)
        self.assertEqual(save.read_bytes(), original)

    def test_sdk_environment_value_is_not_recorded_or_archive_read(self):
        archive = self.root / 'litert_plugin_compiler.tar.gz'
        archive.write_bytes(b'never opened or trusted')
        with mock.patch.dict('os.environ', {'GOOGLE_TENSOR_SDK_BETA': str(archive)}):
            host = plan_huihui.inspect_host()
        metadata = host['compiler_archive']
        self.assertTrue(metadata['environment_variable_present'])
        self.assertTrue(metadata['is_regular_file'])
        self.assertFalse(metadata['contents_checksums_verified'])
        self.assertNotIn(str(archive), json.dumps(host))

    def test_malformed_model_type_does_not_execute_or_crash(self):
        (self.checkpoint / 'config.json').write_text('{"model_type":{"unexpected":"object"}}')
        result = plan_huihui.inspect_checkpoint(self.checkpoint)
        self.assertIn('architecture_not_in_documented_exporter_model_types', result['blockers'])

    def test_nightly_litert_package_satisfies_metadata_presence_check(self):
        def version(name):
            if name in ('litert-torch', 'ai-edge-litert-nightly', 'ai-edge-quantizer-nightly'):
                return 'test-nightly'
            raise plan_huihui.importlib.metadata.PackageNotFoundError(name)
        with mock.patch('plan_huihui.importlib.metadata.version', side_effect=version):
            host = plan_huihui.inspect_host()
        self.assertNotIn('exporter_or_npu_sdk_python_package_missing', host['blockers'])
        self.assertEqual(host['python_packages']['ai-edge-quantizer-nightly'], 'test-nightly')

    def test_nightly_exporter_is_detected_without_stable_exporter(self):
        def version(name):
            if name in ('litert-torch-nightly', 'ai-edge-litert-nightly'):
                return 'test-nightly'
            raise plan_huihui.importlib.metadata.PackageNotFoundError(name)
        with mock.patch('plan_huihui.importlib.metadata.version', side_effect=version):
            host = plan_huihui.inspect_host()
        self.assertNotIn('exporter_or_npu_sdk_python_package_missing', host['blockers'])


if __name__ == '__main__':
    unittest.main()
