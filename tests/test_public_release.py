"""Source release boundary, integrity and deterministic archive regressions."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import check_public_release as check
import release_source as release


class PublicSourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / 'repo'
        self.root.mkdir()
        self.put('README.md', '# PixelForge\n')
        self.put('LICENSE', 'Fixture source license\n')
        self.put('chat/service.py', 'print("model output is data")\n')
        self.put('tools/arm_npu/example.O', 'python^(\nanswer = 42\n)_python\n')
        self.put('source/ostadix-voice/runtime/npu/resident.c', 'int main(void) { return 0; }\n')

    def put(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        return path

    def inventory(self):
        return check.write_manifest(self.root)

    def test_private_and_runtime_neighbors_never_enter_the_source_archive(self):
        excluded = ('payload/private/credentials.json', 'custom-models/gguf-runtime/bin/engine',
            '.env', 'chat/.env.production', 'chat/venv/lib/private.py', 'chat/profile.json',
            'chat/profile-device.json', 'chat/private-profile.json',
            'source/ostadix-voice/runtime/local-model.json',
            'source/ostadix-voice/runtime/models/chosen.gguf',
            'source/ostadix-voice/runtime/pixelforge-bank/graphs/one.dgc',
            'source/ostadix-voice/runtime/npu/model-profile.json',
            'source/ostadix-voice/__pycache__/agent.pyc',
            'integrations/ostadix/pixelforge-npu.O', 'manifest.json', 'settings.json')
        for name in excluded:
            self.put(name, 'private neighbor must not be packaged\n')
        self.inventory()
        destination = self.base / 'release.tar.gz'
        release.build(self.root, destination, check_git=False)
        with tarfile.open(destination) as archive:
            names = {str(Path(member.name).relative_to('pixelforge-source')) for member in archive}
        self.assertFalse(set(excluded) & names)
        self.assertIn('chat/service.py', names)
        self.assertIn('source/ostadix-voice/runtime/npu/resident.c', names)
        self.assertIn('tools/arm_npu/example.O', names)

    def test_source_manifest_detects_changed_added_and_removed_files(self):
        self.inventory()
        check.verify_manifest(self.root)
        self.put('chat/new.py', '# New code must enter the manifest\n')
        with self.assertRaisesRegex(check.PublicReleaseError, 'added'):
            check.verify_manifest(self.root)
        self.inventory()
        self.put('chat/service.py', 'print("changed")\n')
        with self.assertRaisesRegex(check.PublicReleaseError, 'changed'):
            check.verify_manifest(self.root)
        self.inventory()
        (self.root / 'chat/service.py').unlink()
        with self.assertRaisesRegex(check.PublicReleaseError, 'missing'):
            check.verify_manifest(self.root)

    def test_voice_o_source_survives_archive_but_an_embedded_payload_is_rejected(self):
        name = 'source/ostadix-voice/runtime/pixelforge.O'
        program = 'python^(\n__oval_result__ = {"available": False}\n)_python\n'
        self.put(name, program)
        self.inventory()
        destination = self.base / 'with-o-source.tar.gz'
        release.build(self.root, destination, check_git=False)
        with tarfile.open(destination) as archive:
            entry = archive.extractfile('pixelforge-source/' + name)
            self.assertEqual(entry.read().decode(), program)
        self.put(name, program + '# OSTADIX_' + 'NPU_PAYLOAD_V1 arbitrary\n')
        with self.assertRaisesRegex(check.PublicReleaseError, 'embedded archive'):
            check.collect_sources(self.root)

    def test_upstream_license_notices_survive_source_archive(self):
        notices = {
            'chat/native/LICENSE.ggml': 'MIT notice for the architecture engine\n',
            'tools/arm_npu/LICENSE.ggml': 'MIT notice for imported GGUF code\n',
            'tools/arm_npu/LICENSE.litert': 'Apache notice for the graph schema\n',
        }
        for name, text in notices.items():
            self.put(name, text)
        self.inventory()
        destination = self.base / 'with-upstream-notices.tar.gz'
        release.build(self.root, destination, check_git=False)
        release.verify_archive(destination)
        with tarfile.open(destination) as archive:
            for name, text in notices.items():
                with self.subTest(name=name):
                    entry = archive.extractfile('pixelforge-source/' + name)
                    self.assertEqual(entry.read().decode(), text)

    def test_android_voice_and_ostadix_java_sources_survive_archive(self):
        sources = {
            'source/ostadix-voice/android_voice/src/org/ostadix/voicebridge/VoiceService.java':
                'package org.ostadix.voicebridge;\npublic class VoiceService {}\n',
            'source/OSTADIX/apps/android-terminal/app/src/main/java/org/ostadix/terminal/MainActivity.java':
                'package org.ostadix.terminal;\npublic class MainActivity {}\n',
        }
        for name, text in sources.items():
            self.put(name, text)
        self.inventory()
        destination = self.base / 'android-source.tar.gz'
        release.build(self.root, destination, check_git=False)
        release.verify_archive(destination)
        with tarfile.open(destination) as archive:
            for name, text in sources.items():
                with self.subTest(name=name):
                    entry = archive.extractfile('pixelforge-source/' + name)
                    self.assertEqual(entry.read().decode(), text)

    def test_documented_huihui_planner_and_example_survive_archive(self):
        sources = {
            'tools/plan_huihui.py': 'def build_plan():\n    return {"would_execute": False}\n',
            'tools/test_plan_huihui.py': 'import plan_huihui\nassert not plan_huihui.build_plan()["would_execute"]\n',
            'examples/huihui-gemma4-g5-plan-not-built.json': '{"status":"uncompiled/unvalidated"}\n',
        }
        for name, text in sources.items():
            self.put(name, text)
        self.inventory()
        destination = self.base / 'huihui-planner-source.tar.gz'
        release.build(self.root, destination, check_git=False)
        release.verify_archive(destination)
        with tarfile.open(destination) as archive:
            for name, text in sources.items():
                with self.subTest(name=name):
                    entry = archive.extractfile('pixelforge-source/' + name)
                    self.assertEqual(entry.read().decode(), text)

    def test_checksums_and_archive_bytes_are_deterministic(self):
        self.inventory()
        first, second = self.base / 'a.tar.gz', self.base / 'b.tar.gz'
        a = release.build(self.root, first, version='v0.1.0', epoch=1234, check_git=False)
        # Working-tree filesystem timestamps cannot change the release bytes.
        os.utime(self.root / 'README.md', (999999, 999999))
        b = release.build(self.root, second, version='v0.1.0', epoch=1234, check_git=False)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        self.assertEqual(a['sha256'], b['sha256'])
        self.assertEqual(a['sha256'], hashlib.sha256(first.read_bytes()).hexdigest())
        self.assertIn(a['sha256'], Path(str(first) + '.sha256').read_text())

    def test_safe_clean_extraction_preserves_source_manifest(self):
        self.inventory()
        archive = self.base / 'release.tar.gz'
        release.build(self.root, archive, version='test', check_git=False)
        release.verify_archive(archive)
        clean = self.base / 'clean'
        clean.mkdir()
        # Validate first, then materialize regular files without extractall().
        with tarfile.open(archive) as source:
            for member in source:
                target = clean / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.extractfile(member).read())
        check.verify_manifest(clean / 'pixelforge-test')

    def test_symbolic_files_and_parent_directories_are_rejected(self):
        path = self.put('outside.py', 'print("outside")\n')
        (self.root / 'chat/link.py').symlink_to(path)
        with self.assertRaisesRegex(check.PublicReleaseError, 'symlink'):
            check.collect_sources(self.root)
        (self.root / 'chat/link.py').unlink()
        (self.root / 'chat/linked').symlink_to(self.base, target_is_directory=True)
        with self.assertRaisesRegex(check.PublicReleaseError, 'linked source directory'):
            check.collect_sources(self.root)

    def test_model_or_native_binary_disguised_as_source_is_rejected(self):
        for content in (b'GGUF\x03\x00\x00\x00', b'\x7fELFhidden', b'PK\x03\x04archive', b'text\0binary'):
            with self.subTest(content=content[:4]):
                self.put('chat/hidden.py', content)
                with self.assertRaises(check.PublicReleaseError):
                    check.collect_sources(self.root)

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'Named pipes unavailable')
    def test_nonregular_source_is_rejected_without_blocking(self):
        os.mkfifo(self.root / 'chat/blocked.py')
        result = subprocess.run([sys.executable, str(Path(check.__file__)),
            '--root', str(self.root), '--write-manifest', '--skip-git-index'],
            capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 1)
        self.assertIn('only regular source files are allowed', result.stderr)

    def test_real_looking_secret_is_rejected_without_disclosing_value(self):
        secret = 'ghp_' + 'A' * 36
        self.put('chat/secret.py', 'token = ' + repr(secret) + '\n')
        with self.assertRaises(check.PublicReleaseError) as caught:
            check.collect_sources(self.root)
        self.assertIn('github-token', str(caught.exception))
        self.assertIn('values redacted', str(caught.exception))
        self.assertNotIn(secret, str(caught.exception))

    def test_fixture_credentials_and_mentions_are_not_treated_as_real_secrets(self):
        self.put('chat/fixture.py', "api_key = 'offline-fixture-never-a-real-key'\n"
                 "# Credential handling uses stdin. Device activation token is a static gate.\n")
        self.inventory()
        check.verify_manifest(self.root)

    def test_private_key_block_is_rejected(self):
        marker = '-----BEGIN ' + 'PRIVATE KEY-----'
        self.put('chat/key.py', marker + '\n')
        with self.assertRaisesRegex(check.PublicReleaseError, 'private-key'):
            check.collect_sources(self.root)

    def test_duplicate_or_traversing_manifest_paths_are_rejected(self):
        self.inventory()
        manifest = self.root / 'SOURCE_MANIFEST.json'
        value = json.loads(manifest.read_text())
        value['files'].append(dict(value['files'][0]))
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(check.PublicReleaseError, 'duplicate'):
            check.verify_manifest(self.root)
        value['files'][-1]['path'] = '../outside.py'
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(check.PublicReleaseError, 'Unsafe'):
            check.verify_manifest(self.root)

    def test_archive_traversal_links_extra_and_checksum_tamper_are_rejected(self):
        for name, kind, body in (('../escape.py', None, b'escape'),
                ('C:/escape.py', None, b'escape'),
                ('pixelforge-source/chat/link.py', tarfile.SYMTYPE, b''),
                ('pixelforge-source/payload/private.txt', None, b'private'),
                ('pixelforge-source/chat/service.py', None, b'changed')):
            with self.subTest(name=name, kind=kind):
                destination = self.base / 'bad.tar.gz'
                with tarfile.open(destination, 'w:gz') as archive:
                    entry = tarfile.TarInfo(name)
                    entry.size = len(body)
                    entry.mode = 0o644
                    if kind:
                        entry.type = kind
                        entry.linkname = '/outside'
                    archive.addfile(entry, io.BytesIO(body) if not kind else None)
                    value = {'schema': 'pixelforge.source-release.v1', 'files': []}
                    data = json.dumps(value).encode()
                    entry = tarfile.TarInfo('pixelforge-source/RELEASE_MANIFEST.json')
                    entry.mode = 0o644
                    entry.size = len(data)
                    archive.addfile(entry, io.BytesIO(data))
                with self.assertRaises(check.PublicReleaseError):
                    release.verify_archive(destination)

    def test_existing_output_or_receipt_is_never_overwritten(self):
        self.inventory()
        archive = self.base / 'release.tar.gz'
        receipt = Path(str(archive) + '.sha256')
        receipt.write_text('keep this receipt\n')
        with self.assertRaises(FileExistsError):
            release.build(self.root, archive, check_git=False)
        self.assertFalse(archive.exists())
        self.assertEqual(receipt.read_text(), 'keep this receipt\n')

    def test_tracked_private_file_blocks_public_repository_check(self):
        if not __import__('shutil').which('git'):
            self.skipTest('Git unavailable')
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        self.put('payload/private/credentials.json', '{}')
        subprocess.run(['git', '-C', str(self.root), 'add', 'payload/private/credentials.json'], check=True)
        with self.assertRaisesRegex(check.PublicReleaseError, 'remain tracked'):
            check.check_tracked_boundary(self.root)

    def test_unclassified_tracked_source_cannot_be_silently_omitted(self):
        if not __import__('shutil').which('git'):
            self.skipTest('Git unavailable')
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        name = 'chat/future-format.rs'
        self.put(name, 'fn main() {}\n')
        subprocess.run(['git', '-C', str(self.root), 'add', name], check=True)
        with self.assertRaisesRegex(check.PublicReleaseError, 'remain tracked') as caught:
            check.check_tracked_boundary(self.root)
        self.assertNotIn(name, str(caught.exception))

    @unittest.skipUnless(hasattr(os, 'mkfifo'), 'Named pipes unavailable')
    def test_tracked_nonregular_source_is_rejected_without_blocking(self):
        if not __import__('shutil').which('git'):
            self.skipTest('Git unavailable')
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True)
        name = 'chat/service.py'
        subprocess.run(['git', '-C', str(self.root), 'add', name], check=True)
        (self.root / name).unlink()
        os.mkfifo(self.root / name)
        result = subprocess.run([sys.executable, str(Path(check.__file__)),
            '--root', str(self.root), '--write-manifest'],
            capture_output=True, text=True, timeout=3)
        self.assertEqual(result.returncode, 1)
        self.assertIn('remain tracked', result.stderr)
        self.assertNotIn(name, result.stderr)


if __name__ == '__main__':
    unittest.main()
