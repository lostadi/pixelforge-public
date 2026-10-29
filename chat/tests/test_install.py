"""Offline install manifests, configured bootstrap, and ownership boundaries."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'tools/arm_npu'))
import android_launcher
import config
import install
import lifecycle


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.home = self.root / 'home'
        self.runtime = self.home / 'voice/runtime'
        self.runtime.mkdir(parents=True)
        self.model = self.root / 'fixture.gguf'
        self.model.write_bytes(b'non-model fixture for manifest validation')
        self.helper = self.root / 'helper'
        self.helper.write_text('#!/bin/sh\nexit 1\n')
        self.helper.chmod(0o700)
        self.banks = self.root / 'banks'
        self.banks.mkdir()
        (self.banks / 'MODEL.json').write_text(json.dumps({'source_gguf': {'sha256': config.sha256(self.model)}, 'complete': False}))
        self.args = install.parser().parse_args([
            '--ubuntu-root', str(self.root), '--python', sys.executable, '--model', str(self.model),
            '--artifacts', str(self.banks), '--helper', str(self.helper), '--native-binary', str(self.helper),
            '--termux-home', str(self.home), '--app-runtime', str(self.runtime),
            '--model-name', 'Public test fixture', '--provenance-repo', 'huihui-ai/test-fixture',
            '--architecture', 'gemma4', '--select-tensors', 'blk.0.ffn_up.weight',
            '--service-dir', str(self.root / 'service'), '--state-dir', str(self.root / 'state')])
    def tearDown(self):
        self.temp.cleanup()
    def prepare(self):
        class Weights:
            fields = {'tokenizer.chat_template': '{{ messages[0].content }}'}
            def __init__(inner, _): pass
            def __enter__(inner): return inner
            def __exit__(inner, *_): pass
            def sha256(inner): return config.sha256(self.model)
        plan = {'generation_adapter_supported': True, 'architecture': 'gemma4',
                'dense_tensors': [{'name': 'blk.0.ffn_up.weight'}]}
        with patch('gguf_hf_loader.GGUFWeights', Weights), patch('model_architecture.build_plan', return_value=plan):
            return install.prepare(self.args)
    def test_dry_run_lists_exact_writes_and_keeps_assets_in_place(self):
        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob('*'))
        manifest, sources, profile_bytes = self.prepare()
        self.assertFalse(manifest['apply'])
        self.assertFalse(manifest['model_copied'])
        self.assertFalse(manifest['compiled_banks_copied'])
        self.assertEqual(manifest['termux_uid'], self.home.stat().st_uid)
        self.assertEqual(manifest['profile']['native_helper_path'], str(self.helper))
        self.assertEqual(json.loads(profile_bytes)['selected_npu_weight_tensors'], ['blk.0.ffn_up.weight'])
        self.assertEqual(manifest['client_metadata']['launcher'], str(self.root / 'state/chat.py'))
        self.assertNotIn(str(self.model), manifest['writes'])
        self.assertNotIn(str(self.banks), manifest['writes'])
        self.assertEqual(before, sorted(str(path.relative_to(self.root)) for path in self.root.rglob('*')))
        self.assertIn('lifecycle.py', sources)
        self.assertIn('kit/gguf_broker.py', sources)
        self.assertIn('kit/gguf_vendor/quants.py', sources)
        self.assertNotIn('run.py', sources)
    def test_placement_invalid_or_incomplete_fails_before_writing(self):
        for selection in (None, '', 'unknown.weight', 'blk.0.ffn_up.weight,blk.0.ffn_up.weight'):
            self.args.select_tensors = selection
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.prepare()
        self.assertFalse((self.root / 'service').exists())
    def test_other_app_metadata_destination_is_refused(self):
        self.args.app_runtime = self.root / 'other-app'
        with self.assertRaisesRegex(ValueError, 'inside'):
            self.prepare()
    @unittest.skipUnless(os.getuid() == 0, 'Root installer ownership fixture')
    def test_apply_writes_only_reviewed_outputs_without_asset_copy(self):
        # This isolated fixture never uses Android production paths or real weights.
        os.chown(self.home, 10001, 10001)
        os.chown(self.runtime, 10001, 10001)
        self.args.ubuntu_root = Path('/')
        self.args.apply = True
        manifest, sources, profile_bytes = self.prepare()
        install.apply(self.args, manifest, sources, profile_bytes)
        self.assertEqual((self.runtime / 'pixelforge-chat.json').stat().st_uid, 10001)
        generated = config.root_json(self.root / 'state/bootstrap.json')
        self.assertEqual(generated['service'], str(self.root / 'service/service.py'))
        for name, digest in manifest['source_pins'].items():
            self.assertEqual(config.sha256(self.root / 'service' / name), digest)
        self.assertEqual(self.model.read_bytes(), b'non-model fixture for manifest validation')
        self.assertFalse((self.root / 'service/model.gguf').exists())
        self.assertFalse((self.root / 'service/banks').exists())
        with self.assertRaisesRegex(ValueError, 'already exist'):
            install.apply(self.args, manifest, sources, profile_bytes)

    def test_root_json_rejects_symlink_or_writable_input(self):
        path = self.root / 'config.json'
        path.write_text('{}')
        path.chmod(0o600)
        self.assertEqual(config.root_json(path, owner=os.getuid()), {})
        link = self.root / 'alias'
        link.symlink_to(path)
        with self.assertRaises(OSError):
            config.root_json(link, owner=os.getuid())
        path.chmod(0o666)
        with self.assertRaises(ValueError):
            config.root_json(path, owner=os.getuid())


@unittest.skipUnless(os.getuid() == 0, 'Root-owned bootstrap fixtures')
class BootstrapTests(unittest.TestCase):
    def test_local_config_requires_service_and_interpreter_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bootstrap.json'
            data = {'ubuntu_root': '/example/rootfs', 'python': '/example/venv/bin/python',
                    'service': '/example/service.py',
                    'ubuntu_pins': {'/example/service.py': 'a' * 64, '/example/venv/bin/python': 'b' * 64}}
            path.write_text(json.dumps(data))
            path.chmod(0o600)
            self.assertEqual(android_launcher.read_bootstrap(path), data)
            del data['ubuntu_pins']['/example/venv/bin/python']
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                android_launcher.read_bootstrap(path)
    def test_cleanup_terminates_only_owned_identity(self):
        owned_process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        try:
            record = lifecycle.process_record(owned_process.pid)
            lifecycle.cleanup(owned_process, {owned_process.pid: record['stamp']})
            self.assertIsNotNone(owned_process.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            for child in (owned_process, unrelated):
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=2)


if __name__ == '__main__':
    unittest.main()
