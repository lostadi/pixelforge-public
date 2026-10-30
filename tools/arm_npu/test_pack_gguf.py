"""Self-extracting compiler export tests with tiny runtime assets and no hardware."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pack_gguf
import setup_gguf


class CompilerExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.repo = self.root / 'repository'
        self.kit = self.repo / 'tools/arm_npu'
        self.addon = self.repo / 'custom-models/gguf-runtime'
        self.kit.mkdir(parents=True)
        for name in pack_gguf.SOURCE_NAMES:
            target = self.kit / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pack_gguf.REPO / 'tools/arm_npu' / name, target)
        shutil.copyfile(pack_gguf.REPO / 'benchmark.py', self.repo / 'benchmark.py')
        public = json.loads((self.kit / 'GGUF_RUNTIME_ASSETS.json').read_text())
        self.files = []
        for pin in public['files']:
            path = self.addon / pin['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            data = ('fixture:' + pin['path']).encode()
            path.write_bytes(data)
            self.files.append({'path': pin['path'], 'bytes': len(data),
                               'sha256': hashlib.sha256(data).hexdigest()})
        public['files'] = self.files
        self.manifest = public
        self.save_manifest()
        (self.kit / 'GGUF_RUNTIME_ASSETS.json').write_text(json.dumps(public))
        self.archive = self.root / 'compiler.run'
        self.destination = self.root / 'unpacked compiler'
        self.pins = mock.patch.object(setup_gguf, 'PUBLIC_ASSETS', self.kit / 'GGUF_RUNTIME_ASSETS.json')
        self.pins.start()
        self.addCleanup(self.pins.stop)

    def tearDown(self):
        self.temporary.cleanup()

    def save_manifest(self):
        (self.addon / 'MANIFEST.json').write_text(json.dumps(self.manifest))

    def pack(self):
        return pack_gguf.pack(self.archive, repo=self.repo)

    def extract(self):
        return subprocess.run(['/bin/sh', str(self.archive), str(self.destination)],
                              capture_output=True, text=True, timeout=15)

    def test_extracts_verifiable_assets_source_and_licenses_without_private_inputs(self):
        private = [
            'payload/secret', 'settings.json', '.git/config', 'model.gguf',
            'tools/arm_npu/credentials.py', 'tools/arm_npu/diagnostics/private.py',
            'tools/arm_npu/__pycache__/secret.pyc',
            'custom-models/gguf-runtime/venv/bin/python',
            'custom-models/gguf-runtime/results/native.stderr',
            'custom-models/gguf-runtime/credentials.json',
            'custom-models/gguf-runtime/source/private-model.gguf',
        ]
        for name in private:
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'private fixture must never ship')
        result = self.pack()
        self.assertEqual(result['runtime_assets'], 6)
        self.assertTrue(self.archive.stat().st_mode & stat.S_IXUSR)
        process = self.extract()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('6 verified runtime assets', process.stdout)
        self.assertIn('Set up: python3', process.stdout)
        self.assertEqual(stat.S_IMODE(self.destination.stat().st_mode), 0o700)
        for name in private:
            self.assertFalse((self.destination / name).exists(), name)
        self.assertTrue((self.destination / 'README.md').is_file())
        self.assertIn('bash tools/arm_npu/gguf.sh inspect /path/to/your-model.gguf',
                      (self.destination / 'README.md').read_text())
        self.assertTrue((self.destination / 'benchmark.py').is_file())
        for name in ('compile_gguf.py', 'npu_pipe.cc', 'LICENSE.litert', 'LICENSE.ggml',
                     'gguf_vendor/quants.py', 'pack_gguf.py', 'GGUF_COMPILER.md'):
            self.assertTrue((self.destination / 'tools/arm_npu' / name).is_file(), name)
        for item in self.files:
            extracted = self.destination / 'custom-models/gguf-runtime' / item['path']
            self.assertEqual(extracted.stat().st_size, item['bytes'])
            self.assertEqual(hashlib.sha256(extracted.read_bytes()).hexdigest(), item['sha256'])
        checked = subprocess.run([sys.executable,
                                  str(self.destination / 'tools/arm_npu/setup_gguf.py'),
                                  '--verify-only'], capture_output=True, text=True, timeout=10)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertEqual(json.loads(checked.stdout)['verified_files'], 6)

    def test_output_collision_and_repository_output_are_rejected(self):
        self.archive.write_bytes(b'keep existing archive')
        with self.assertRaises(FileExistsError):
            self.pack()
        self.assertEqual(self.archive.read_bytes(), b'keep existing archive')
        with self.assertRaisesRegex(ValueError, 'outside the repository'):
            pack_gguf.pack(self.repo / 'compiler.run', repo=self.repo)
        self.assertFalse((self.repo / 'compiler.run').exists())

    def test_exported_grouped_runtime_and_fixtures_have_no_research_or_checkout_dependencies(self):
        self.pack()
        extracted = self.extract()
        self.assertEqual(extracted.returncode, 0, extracted.stderr)
        kit = self.destination / 'tools/arm_npu'
        self.assertTrue((kit / 'ORIGINAL_GROUPS.md').is_file())
        for name in ('author_packed_quant.py', 'author_packed_quant_float.py',
                     'author_dynamic_fc_runtime.py', 'author_grouped_full_ffn.py', 'test_packed_quant.py'):
            self.assertFalse((kit / name).exists())
        # Isolated Python cannot find this checkout through cwd/PYTHONPATH.
        # The synthetic helper workers must also come from the extracted kit.
        code = '''import sys,unittest
sys.path.insert(0,sys.argv[1])
import gguf_broker,compile_grouped_fc,grouped_fc,compile_grouped_columns,grouped_columns,dynamic_fc,full_ffn,dma_buffers
names=['test_batched_fc','test_dynamic_fc','test_grouped_fc','test_grouped_columns','test_dma_buffers','test_full_ffn']
suite=unittest.defaultTestLoader.loadTestsFromNames(names)
result=unittest.TextTestRunner(verbosity=1).run(suite)
raise SystemExit(not result.wasSuccessful())
'''
        process = subprocess.run([sys.executable, '-I', '-c', code, str(kit)],
                                 cwd=self.destination, capture_output=True, text=True, timeout=45)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
        self.assertIn('OK', process.stderr)

    def test_existing_extraction_destination_is_preserved(self):
        self.pack()
        self.destination.mkdir()
        marker = self.destination / 'important.txt'
        marker.write_text('keep existing directory')
        process = self.extract()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('Destination exists', process.stderr)
        self.assertEqual(marker.read_text(), 'keep existing directory')

    def test_corrupt_payload_fails_before_destination_creation(self):
        self.pack()
        with self.archive.open('r+b') as stream:
            stream.seek(-1, 2)
            original = stream.read(1)
            stream.seek(-1, 2)
            stream.write(bytes([original[0] ^ 1]))
        process = self.extract()
        self.assertNotEqual(process.returncode, 0)
        self.assertIn('payload SHA256 mismatch', process.stderr)
        self.assertFalse(self.destination.exists())

    def test_coherently_tampered_private_asset_is_rejected_before_output_creation(self):
        entry = next(item for item in self.files if item['path'] == 'native/npu_pipe_gguf')
        data = b'changed private runtime'
        (self.addon / entry['path']).write_bytes(data)
        entry.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'Public GGUF runtime asset pin mismatch'):
            self.pack()
        self.assertFalse(self.archive.exists())

    def test_coherent_runtime_and_manifest_mutation_after_preflight_is_rejected(self):
        original_export = pack_gguf.export_files

        def mutate_after_preflight(repo):
            selected = original_export(repo)
            entry = next(item for item in self.files if item['path'] == 'native/npu_pipe_gguf')
            data = b'changed after snapshot'
            (self.addon / entry['path']).write_bytes(data)
            entry.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            self.save_manifest()
            return selected

        with mock.patch.object(pack_gguf, 'export_files', side_effect=mutate_after_preflight):
            with self.assertRaisesRegex(ValueError, 'Changed compiler export input'):
                self.pack()
        self.assertFalse(self.archive.exists())

    def test_public_and_runtime_mutation_after_verification_cannot_mix_with_old_manifest(self):
        original_verify = pack_gguf.verify_addon

        def switch_after_verification(root):
            validated = original_verify(root)
            public_path = self.kit / 'GGUF_RUNTIME_ASSETS.json'
            public = json.loads(public_path.read_text())
            pin = next(item for item in public['files'] if item['path'] == 'native/npu_pipe_gguf')
            data = b'new public runtime after old private verification'
            (self.addon / pin['path']).write_bytes(data)
            pin.update(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            public_path.write_text(json.dumps(public))
            return validated

        with mock.patch.object(pack_gguf, 'verify_addon', side_effect=switch_after_verification):
            with self.assertRaisesRegex(ValueError, 'public and validated private runtime pins disagree'):
                self.pack()
        self.assertFalse(self.archive.exists())

    def test_captured_public_metadata_and_pin_schema_are_validated_again(self):
        original_verify = pack_gguf.verify_addon
        public_path = self.kit / 'GGUF_RUNTIME_ASSETS.json'
        original_public = public_path.read_text()
        for changed in ('format', 'engine_revision', 'files', 'duplicate', 'size', 'sha256'):
            with self.subTest(changed=changed):
                public_path.write_text(original_public)

                def switch_after_verification(root):
                    validated = original_verify(root)
                    public = json.loads(original_public)
                    if changed in ('format', 'engine_revision'):
                        public[changed] = 'changed'
                    elif changed == 'files':
                        public['files'] = None
                    elif changed == 'duplicate':
                        public['files'].append(public['files'][0].copy())
                    elif changed == 'size':
                        public['files'][0]['bytes'] = True
                    else:
                        public['files'][0]['sha256'] = 'invalid'
                    public_path.write_text(json.dumps(public))
                    return validated

                with mock.patch.object(pack_gguf, 'verify_addon', side_effect=switch_after_verification):
                    with self.assertRaisesRegex(ValueError, 'Invalid captured public'):
                        self.pack()
                self.assertFalse(self.archive.exists())

    def test_source_mutation_after_preflight_is_rejected(self):
        original_export = pack_gguf.export_files

        def mutate_after_preflight(repo):
            selected = original_export(repo)
            (self.kit / 'npu_pipe.cc').write_bytes(b'changed source after snapshot')
            return selected

        with mock.patch.object(pack_gguf, 'export_files', side_effect=mutate_after_preflight):
            with self.assertRaisesRegex(ValueError, 'Changed compiler export input'):
                self.pack()
        self.assertFalse(self.archive.exists())

    def test_runtime_mutation_after_open_is_detected_while_streaming(self):
        original_addfile = pack_gguf.tarfile.TarFile.addfile
        target = 'custom-models/gguf-runtime/native/npu_pipe_gguf'

        def mutate_after_open(archive, member, source=None):
            if member.name == target:
                path = self.repo / target
                data = path.read_bytes()
                path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
            return original_addfile(archive, member, source)

        with mock.patch.object(pack_gguf.tarfile.TarFile, 'addfile', new=mutate_after_open):
            with self.assertRaisesRegex(ValueError, 'Changed compiler export input while streaming'):
                self.pack()
        self.assertFalse(self.archive.exists())

    def test_failed_pack_removes_only_its_owned_output(self):
        sibling = self.root / 'keep.txt'
        sibling.write_bytes(b'unrelated')
        with mock.patch.object(pack_gguf.tarfile.TarFile, 'addfile', side_effect=RuntimeError('injected')):
            with self.assertRaisesRegex(RuntimeError, 'injected'):
                self.pack()
        self.assertFalse(self.archive.exists())
        self.assertEqual(sibling.read_bytes(), b'unrelated')

    def test_failed_pack_does_not_remove_a_replaced_output(self):
        def replace_output(*_args, **_kwargs):
            self.archive.unlink()
            self.archive.write_bytes(b'new owner')
            raise RuntimeError('injected replacement')
        with mock.patch.object(pack_gguf.tarfile.TarFile, 'addfile', side_effect=replace_output):
            with self.assertRaisesRegex(RuntimeError, 'injected replacement'):
                self.pack()
        self.assertEqual(self.archive.read_bytes(), b'new owner')

    def test_completed_pack_cannot_report_a_replaced_output_as_success(self):
        original_addfile = pack_gguf.tarfile.TarFile.addfile

        def replace_output(archive, member, source=None):
            if member.name == 'README.md':
                self.archive.unlink()
                self.archive.write_bytes(b'new owner')
            return original_addfile(archive, member, source)

        with mock.patch.object(pack_gguf.tarfile.TarFile, 'addfile', new=replace_output):
            with self.assertRaisesRegex(ValueError, 'output changed owner'):
                self.pack()
        self.assertEqual(self.archive.read_bytes(), b'new owner')


if __name__ == '__main__':
    unittest.main()
