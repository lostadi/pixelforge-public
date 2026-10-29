"""Offline asset verification and failure-report behavior without NPU calls."""
import argparse
import builtins
import contextlib
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

import runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.kit = self.root / 'kit'
        self.kit.mkdir()
        self.addon = self.root / 'addon'
        self.addon.mkdir()
        self.files = []
        for name in ('model.gguf', 'tokenizer/config.json', 'native/npu_pipe'):
            path = self.addon / name
            path.parent.mkdir(parents=True, exist_ok=True)
            content = name.encode()
            path.write_bytes(content)
            self.files.append({'path': name, 'bytes': len(content),
                               'sha256': hashlib.sha256(content).hexdigest()})
        self.manifest = {'files': self.files}
        self.save_manifest()
        pins = {'quantized_gguf_bytes': self.files[0]['bytes'],
                'quantized_gguf_sha256': self.files[0]['sha256'],
                'support_files': [{**self.files[1], 'path': 'config.json'}]}
        (self.kit / 'HUIHUI_ASSETS.json').write_text(json.dumps(pins))
        self.patch = mock.patch.object(runtime, 'KIT', self.kit)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def save_manifest(self):
        (self.addon / 'MANIFEST.json').write_text(json.dumps(self.manifest))

    def test_correct_assets_verify_then_same_size_corruption_is_rejected(self):
        runtime.verify_addon(self.addon)
        (self.addon / 'model.gguf').write_bytes(b'x' * self.files[0]['bytes'])
        with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
            runtime.verify_addon(self.addon)

    def test_modified_manifest_cannot_override_published_checkpoint_pin(self):
        content = b'different model'
        (self.addon / 'model.gguf').write_bytes(content)
        self.files[0].update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'published checkpoint pin'):
            runtime.verify_addon(self.addon)

    def test_manifest_escape_and_symlink_are_rejected(self):
        path = self.addon / 'native/npu_pipe'
        path.unlink()
        outside = self.root / 'outside'
        outside.write_bytes(b'native/npu_pipe')
        path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'redirected'):
            runtime.verify_addon(self.addon)
        with self.assertRaisesRegex(ValueError, 'stay inside'):
            runtime.checked_file(self.addon, {**self.files[2], 'path': '../outside'})

    def test_failure_is_private_and_does_not_start_native(self):
        args = argparse.Namespace(addon=self.addon, prompt='Hi', tokens=1,
                                  report=self.addon / 'failed.json')
        with mock.patch.object(runtime, 'verify_addon', side_effect=ValueError('fixture bad asset')), \
                mock.patch('npu_linear.NPULinear') as native, \
                contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runtime.worker(args), 1)
        native.assert_not_called()
        report = json.loads(args.report.read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertIn('fixture bad asset', report['error'])
        self.assertEqual(stat.S_IMODE(args.report.stat().st_mode), 0o600)
        with self.assertRaises(FileExistsError):
            runtime.worker(args)

    def test_missing_dependency_still_creates_failure_report(self):
        args = argparse.Namespace(addon=self.addon, prompt='Hi', tokens=1,
                                  report=self.addon / 'missing-dependency.json')
        original = builtins.__import__
        def importing(name, *positional, **keywords):
            if name == 'gguf_hf_loader':
                raise ModuleNotFoundError('fixture missing dependency')
            return original(name, *positional, **keywords)
        with mock.patch('builtins.__import__', side_effect=importing), \
                contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runtime.worker(args), 1)
        report = json.loads(args.report.read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertIn('fixture missing dependency', report['error'])
        self.assertEqual(stat.S_IMODE(args.report.stat().st_mode), 0o600)


if __name__ == '__main__':
    unittest.main()
