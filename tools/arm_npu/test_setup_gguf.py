"""Offline GGUF setup preflight tests with tiny synthetic files only."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import setup_gguf


class GGUFSetupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / 'addon'
        self.root.mkdir()
        self.entries = []
        for name in sorted(setup_gguf.REQUIRED_FILES | {'wheels/numpy-fixture.whl',
                                                       'wheels/flatbuffers-fixture.whl'}):
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            content = name.encode('ascii')
            path.write_bytes(content)
            self.entries.append({'path': name, 'bytes': len(content),
                                 'sha256': hashlib.sha256(content).hexdigest()})
        self.manifest = {'files': self.entries, 'engine_revision': setup_gguf.ENGINE_REVISION}
        self.public_assets = Path(self.temporary.name) / 'public-assets.json'
        self.public_assets.write_text(json.dumps({
            'format': 'pixelforge.gguf-runtime.assets.v1',
            'engine_revision': setup_gguf.ENGINE_REVISION,
            'files': self.entries,
        }))
        pins = mock.patch.object(setup_gguf, 'PUBLIC_ASSETS', self.public_assets)
        pins.start()
        self.addCleanup(pins.stop)
        self.save()

    def tearDown(self):
        self.temporary.cleanup()

    def save(self):
        (self.root / 'MANIFEST.json').write_text(json.dumps(self.manifest))

    def test_runtime_assets_verify_without_model_or_minicpm_pins(self):
        self.assertEqual(len(setup_gguf.verify_addon(self.root)['files']), 6)
        self.assertFalse((self.root / 'model.gguf').exists())

    def test_same_size_tamper_fails_before_any_install(self):
        path = self.root / 'bin/pixelforge-gguf'
        path.write_bytes(b'x' * path.stat().st_size)
        with mock.patch.object(setup_gguf.venv, 'EnvBuilder') as builder, \
                mock.patch.object(setup_gguf.subprocess, 'run') as pip, \
                mock.patch.object(setup_gguf, 'install_native') as native:
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                setup_gguf.setup(self.root)
        builder.assert_not_called()
        pip.assert_not_called()
        native.assert_not_called()

    def test_coherent_private_asset_and_manifest_tamper_cannot_replace_public_pins(self):
        for entry in self.entries:
            with self.subTest(path=entry['path']):
                path = self.root / entry['path']
                original = path.read_bytes()
                original_entry = entry.copy()
                changed = b'x' * len(original)
                path.write_bytes(changed)
                entry['sha256'] = hashlib.sha256(changed).hexdigest()
                self.save()
                with mock.patch.object(setup_gguf, 'checked_file') as checked, \
                        mock.patch.object(setup_gguf.venv, 'EnvBuilder') as builder, \
                        mock.patch.object(setup_gguf.subprocess, 'run') as pip, \
                        mock.patch.object(setup_gguf, 'install_native') as native:
                    with self.assertRaisesRegex(ValueError, 'Public GGUF runtime asset pin mismatch'):
                        setup_gguf.setup(self.root)
                checked.assert_not_called()
                builder.assert_not_called()
                pip.assert_not_called()
                native.assert_not_called()
                path.write_bytes(original)
                entry.update(original_entry)
                self.save()

    def test_coherent_private_file_size_change_cannot_replace_public_pin(self):
        entry = next(item for item in self.entries if item['path'] == 'native/npu_pipe_gguf')
        changed = (self.root / entry['path']).read_bytes() + b'changed'
        (self.root / entry['path']).write_bytes(changed)
        entry.update(bytes=len(changed), sha256=hashlib.sha256(changed).hexdigest())
        self.save()
        with mock.patch.object(setup_gguf, 'checked_file') as checked:
            with self.assertRaisesRegex(ValueError, 'Public GGUF runtime asset pin mismatch'):
                setup_gguf.verify_addon(self.root)
        checked.assert_not_called()

    def test_missing_publicly_pinned_wheel_is_rejected(self):
        self.entries[:] = [item for item in self.entries if item['path'] != 'wheels/numpy-fixture.whl']
        self.save()
        with self.assertRaisesRegex(ValueError, 'needs verified public file'):
            setup_gguf.verify_addon(self.root)

    def test_manifest_escape_and_alias_paths_are_rejected(self):
        for name in ('../outside', '/outside', './bin/pixelforge-gguf',
                     'bin//pixelforge-gguf', 'bin\\pixelforge-gguf'):
            with self.subTest(name=name):
                self.entries[0]['path'] = name
                self.save()
                with self.assertRaisesRegex(ValueError, '[Pp]ath'):
                    setup_gguf.verify_addon(self.root)

    def test_symlink_file_and_symlink_parent_are_rejected(self):
        path = self.root / 'bin/pixelforge-gguf'
        destination = self.root / 'original'
        path.rename(destination)
        path.symlink_to(destination)
        with self.assertRaisesRegex(ValueError, 'Redirected'):
            setup_gguf.verify_addon(self.root)
        path.unlink()
        destination.rename(path)
        directory = self.root / 'bin'
        replacement = self.root / 'actual-bin'
        directory.rename(replacement)
        directory.symlink_to(replacement, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'Redirected'):
            setup_gguf.verify_addon(self.root)

    def test_missing_required_file_and_duplicate_are_rejected(self):
        self.entries.pop(0)
        self.save()
        with self.assertRaisesRegex(ValueError, 'needs verified files'):
            setup_gguf.verify_addon(self.root)
        self.entries.append(self.entries[0].copy())
        self.save()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            setup_gguf.verify_addon(self.root)

    def test_unlisted_wheel_is_rejected(self):
        (self.root / 'wheels/unlisted.whl').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'Every wheel'):
            setup_gguf.verify_addon(self.root)

    def test_wrong_engine_revision_is_rejected(self):
        self.manifest['engine_revision'] = '0' * 40
        self.save()
        with self.assertRaisesRegex(ValueError, 'pinned architecture engine'):
            setup_gguf.verify_addon(self.root)

    def test_missing_addon_gives_build_guide_without_install(self):
        errors = io.StringIO()
        with contextlib.redirect_stderr(errors), mock.patch.object(setup_gguf, 'install_native') as native:
            self.assertEqual(setup_gguf.main(['--addon', str(self.root / 'missing')]), 1)
        native.assert_not_called()
        self.assertIn('GGUF_COMPILER.md', errors.getvalue())

    def test_valid_setup_uses_offline_flags_and_dedicated_helper(self):
        builder = mock.Mock()
        with mock.patch.object(setup_gguf.sys, 'version_info', (3, 12, 0)), \
                mock.patch.object(setup_gguf.os, 'uname', return_value=mock.Mock(machine='aarch64')), \
                mock.patch.object(setup_gguf.os, 'geteuid', return_value=0), \
                mock.patch.object(setup_gguf.venv, 'EnvBuilder', return_value=builder), \
                mock.patch.object(setup_gguf.subprocess, 'run') as pip, \
                mock.patch.object(setup_gguf, 'install_native') as native, \
                contextlib.redirect_stdout(io.StringIO()):
            result = setup_gguf.setup(self.root)
        builder.create.assert_called_once_with(self.root / 'venv')
        command = pip.call_args.args[0]
        for flag in ('--no-index', '--no-deps', '--require-hashes', '--force-reinstall'):
            self.assertIn(flag, command)
        native.assert_called_once_with(self.root / 'native/npu_pipe_gguf',
                                       destination=setup_gguf.NATIVE_DESTINATION)
        self.assertEqual(result['verified_files'], 6)
        self.assertTrue(result['native'].endswith('/npu_pipe_gguf'))

    def test_unsupported_machine_fails_before_venv_or_pip(self):
        with mock.patch.object(setup_gguf.sys, 'version_info', (3, 12, 0)), \
                mock.patch.object(setup_gguf.os, 'uname', return_value=mock.Mock(machine='x86_64')), \
                mock.patch.object(setup_gguf.venv, 'EnvBuilder') as builder, \
                mock.patch.object(setup_gguf.subprocess, 'run') as pip:
            with self.assertRaisesRegex(RuntimeError, 'ARM64'):
                setup_gguf.setup(self.root)
        builder.assert_not_called()
        pip.assert_not_called()


if __name__ == '__main__':
    unittest.main()
