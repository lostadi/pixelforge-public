#!/usr/bin/env python3
"""Check the public source boundary without reading private deployment state."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'pixelforge.public-source.v1'
MAX_SOURCE_FILE = 16 * 1024 * 1024
SOURCE_ROOTS = ('chat', 'source', 'tools/arm_npu', 'integrations/ostadix/tools',
                'docs', 'tests', 'scripts', 'benchmarks')
ROOT_FILES = frozenset(('README.md', 'LICENSE', 'COPYING', 'NOTICE', 'AUTHORS',
    'THIRD_PARTY_NOTICES.md', 'SECURITY.md', 'CONTRIBUTING.md', 'CHANGELOG.md',
    'RELEASE_NOTES.md', 'VERSION', 'SOURCE_MANIFEST.json', 'SOURCE_VALIDATION.json',
    'ARM_NPU.md', 'BENCHMARKS.md', 'CUSTOM_MODELS.md', 'INFERENCE_PATHS.md',
    'benchmark.py', 'test_benchmark.py', 'setup.sh', 'pyproject.toml',
    '.gitignore', '.gitattributes', 'requirements.txt', 'requirements-dev.txt',
    '.github/workflows/ci.yml', 'integrations/ostadix/README.md',
    'integrations/ostadix/COPYING', 'tools/README.md',
    'tools/plan_huihui.py', 'tools/test_plan_huihui.py',
    'examples/huihui-gemma4-g5-plan-not-built.json'))
SOURCE_SUFFIXES = frozenset(('.py', '.c', '.cc', '.cpp', '.h', '.hpp', '.sh',
    '.cmake', '.patch', '.md', '.rst', '.txt', '.json', '.toml', '.yaml', '.yml',
    '.xml', '.java', '.properties', '.gradle', '.kts', '.js', '.cjs', '.mjs', '.css',
    '.html', '.O', '.oc', '.svg', '.ini', '.cfg', '.list', '.prop'))
FORBIDDEN_DIRS = frozenset(('.git', '__pycache__', '.pytest_cache', '.mypy_cache',
    '.cache', '.venv', 'venv', 'node_modules', 'payload', 'custom-models', 'dist',
    'restore-backups', 'private', 'private-context', 'models', 'chat-jobs',
    'jobs', 'sessions', 'checkpoints', 'graphs', 'build', 'target'))
FORBIDDEN_SUFFIXES = frozenset(('.gguf', '.safetensors', '.litertlm', '.tflite',
    '.onnx', '.dgc', '.elf', '.so', '.a', '.o', '.apk', '.aar', '.jar', '.zip',
    '.gz', '.xz', '.bz2', '.run', '.pyc', '.sqlite', '.sqlite3', '.db', '.pem',
    '.key', '.keystore', '.jks', '.p12', '.pfx', '.bin'))
FORBIDDEN_NAMES = frozenset(('settings.json', 'packages.json', 'manifest.json',
    'profile.json', 'bootstrap.json', 'sessions.json', 'credentials.json',
    'auth.json', 'client_secret.json', 'pixelforge-chat.json', 'local-model.json',
    'model-profile.json', 'MODEL.json', 'weights.plan'))
RUNTIME_SOURCE_FILES = frozenset(('source/ostadix-voice/runtime/pixelforge.O',))
SOURCE_BASENAMES = frozenset(('LICENSE', 'NOTICE', 'COPYING', 'AUTHORS',
    'LICENSE.ggml', 'LICENSE.litert',
    'CMakeLists.txt', 'requirements-chat.txt', 'requirements.txt', '.gitignore',
    '.gitattributes', 'Makefile'))
MAGIC = ((b'\x7fELF', 'ELF binary'), (b'GGUF', 'GGUF weights'),
         (b'PK\x03\x04', 'ZIP payload'), (b'\x1f\x8b', 'gzip payload'),
         (b'SQLite format 3\x00', 'SQLite state'))
SECRET_RULES = (
    ('private-key', re.compile(r'-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----')),
    ('github-token', re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{40,255})\b')),
    ('provider-key', re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{24,255}\b')),
    ('google-api-key', re.compile(r'\bAIza[A-Za-z0-9_-]{35}\b')),
    ('aws-access-key', re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
    ('jwt', re.compile(r'\beyJ[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}\b')),
)
ASSIGNED_SECRET = re.compile(
    r'''(?i)["']?\b(?:api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|password|secret[_-]?key)\b["']?\s*[:=]\s*["']([^"'\r\n]{24,})["']''')
EXAMPLE_MARKERS = ('fixture', 'example', 'dummy', 'placeholder', 'redacted',
                   'changeme', 'replace_', 'replace-', 'not-a-real', 'never-real',
                   'never-a-real', 'your_api', 'your-api', '<', '${')


class PublicReleaseError(ValueError):
    pass


@dataclass(frozen=True)
class SourceFile:
    path: str
    bytes: int
    sha256: str
    mode: int

    def record(self):
        return {'path': self.path, 'bytes': self.bytes, 'sha256': self.sha256}


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or path.as_posix() != name
            or any(c in name for c in ('\0', '\n', '\r', '\\', ':'))):
        raise PublicReleaseError('Unsafe source path')
    return path


def excluded(name):
    path = safe_relative(name)
    if any(part in FORBIDDEN_DIRS for part in path.parts):
        return True
    if any(part == '.env' or part.startswith('.env.') for part in path.parts):
        return True
    if path.name in FORBIDDEN_NAMES:
        return True
    if (path.suffix == '.json' and
            (path.name.startswith('profile-') or path.name.startswith('private-profile'))
            and not path.name.endswith('.example.json')):
        return True
    if path.suffix.lower() in FORBIDDEN_SUFFIXES and path.suffix != '.O':
        return True
    if name == 'integrations/ostadix/pixelforge-npu.O':
        return True
    # Native runtime source and public headers are allowed; installed models,
    # state and manifests under the same runtime tree are not.
    if 'runtime' in path.parts:
        if name in RUNTIME_SOURCE_FILES:
            return False
        prefix = 'source/ostadix-voice/runtime/npu/'
        if name in ('source/ostadix-voice/runtime', prefix.rstrip('/')):
            return False
        if not name.startswith(prefix):
            return True
        if path.suffix == '' or path.name.startswith('c_api-'):
            return False
        if path.suffix not in ('.py', '.c', '.h', '.sh', '.md', '.txt') and path.name not in ('LICENSE', 'NOTICE', 'COPYING'):
            return True
    return False


def allowed(name):
    path = safe_relative(name)
    if excluded(name):
        return False
    if name in ROOT_FILES:
        return True
    if not any(name.startswith(root + '/') for root in SOURCE_ROOTS):
        return False
    return path.suffix in SOURCE_SUFFIXES or path.name in SOURCE_BASENAMES


def secret_findings(text):
    """Return only rule IDs and line numbers, never matched secret values."""
    findings = []
    for rule, pattern in SECRET_RULES:
        for match in pattern.finditer(text):
            if any(marker in match.group().lower() for marker in EXAMPLE_MARKERS):
                continue
            findings.append((rule, text.count('\n', 0, match.start()) + 1))
    for match in ASSIGNED_SECRET.finditer(text):
        value = match.group(1)
        if any(marker in value.lower() for marker in EXAMPLE_MARKERS):
            continue
        frequencies = {char: value.count(char) / len(value) for char in set(value)}
        entropy = -sum(p * math.log2(p) for p in frequencies.values())
        if entropy >= 3.6:
            findings.append(('assigned-secret', text.count('\n', 0, match.start()) + 1))
    return sorted(set(findings))


def inspect_bytes(name, data):
    for prefix, kind in MAGIC:
        if data.startswith(prefix):
            raise PublicReleaseError(f'{name}: {kind} is not public source')
    if b'\0' in data:
        raise PublicReleaseError(f'{name}: binary data is not public source')
    try:
        text = data.decode('utf-8')
    except UnicodeDecodeError as error:
        raise PublicReleaseError(f'{name}: source must be UTF-8 text') from error
    if PurePosixPath(name).suffix == '.O' and ('# OSTADIX_NPU_PAYLOAD_V1 ' in text or 'PIXELFORGE_GGUF_EXTRACTOR' in text):
        raise PublicReleaseError(f'{name}: embedded archive must not enter the source release')
    findings = secret_findings(text)
    if findings:
        safe = ', '.join(f'{rule} at line {line}' for rule, line in findings)
        raise PublicReleaseError(f'{name}: possible secret ({safe}); values redacted')


def source_bytes(root, name):
    path = root.joinpath(*safe_relative(name).parts)
    # Check every component without following a linked parent directory.
    current = root
    for part in safe_relative(name).parts:
        current = current / part
        metadata = current.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PublicReleaseError(f'{name}: symlinks are not allowed in a public source release')
    if not stat.S_ISREG(metadata.st_mode):
        raise PublicReleaseError(f'{name}: only regular source files are allowed')
    # Nonblocking open also prevents a concurrently substituted FIFO from
    # hanging the scanner before the descriptor's type can be checked.
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise PublicReleaseError(f'{name}: only regular source files are allowed')
        if before.st_size > MAX_SOURCE_FILE:
            raise PublicReleaseError(f'{name}: source file exceeds the public artifact inspection limit')
        data = stream.read(MAX_SOURCE_FILE + 1)
        after = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if identity(before) != identity(after) or identity(before) != identity(path.stat(follow_symlinks=False)):
        raise PublicReleaseError(f'{name}: source changed while reading')
    inspect_bytes(name, data)
    return data, 0o755 if before.st_mode & 0o111 else 0o644


def collect_sources(root=ROOT, include_manifest=True):
    root = Path(root).resolve(strict=True)
    names = set()
    for name in ROOT_FILES:
        path = root.joinpath(*safe_relative(name).parts)
        if path.exists() or path.is_symlink():
            names.add(name)
    for directory in SOURCE_ROOTS:
        base = root.joinpath(*safe_relative(directory).parts)
        if not base.exists() and not base.is_symlink():
            continue
        if base.is_symlink():
            raise PublicReleaseError(f'{directory}: linked source root is not allowed')
        for current, directories, files in os.walk(base, followlinks=False):
            relative = Path(current).relative_to(root)
            for directory_name in tuple(directories):
                name = (relative / directory_name).as_posix()
                target = root / name
                if target.is_symlink() and not excluded(name):
                    raise PublicReleaseError(f'{name}: linked source directory is not allowed')
            directories[:] = [d for d in directories if not excluded((relative / d).as_posix())]
            for file_name in files:
                name = (relative / file_name).as_posix()
                if allowed(name):
                    names.add(name)
                elif (root / name).is_symlink() and not excluded(name):
                    raise PublicReleaseError(f'{name}: linked file is not allowed')
    result = []
    for name in sorted(names):
        if not include_manifest and name == 'SOURCE_MANIFEST.json':
            continue
        if not allowed(name):
            continue
        data, mode = source_bytes(root, name)
        result.append(SourceFile(name, len(data), hashlib.sha256(data).hexdigest(), mode))
    return result


def manifest_data(files):
    return {'schema': SCHEMA, 'boundary': 'source-only; no model weights, compiled banks or private deployment state',
            'files': [entry.record() for entry in files if entry.path != 'SOURCE_MANIFEST.json']}


def write_manifest(root=ROOT, destination=None):
    root = Path(root).resolve(strict=True)
    destination = Path(destination) if destination else root / 'SOURCE_MANIFEST.json'
    data = manifest_data(collect_sources(root, include_manifest=False))
    descriptor, temporary = tempfile.mkstemp(prefix='.source-manifest-', dir=destination.parent)
    try:
        with os.fdopen(descriptor, 'w') as output:
            json.dump(data, output, indent=2, sort_keys=True)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return data


def verify_manifest(root=ROOT, destination=None, files=None):
    root = Path(root).resolve(strict=True)
    destination = Path(destination) if destination else root / 'SOURCE_MANIFEST.json'
    value = json.loads(destination.read_text())
    if value.get('schema') != SCHEMA or not isinstance(value.get('files'), list):
        raise PublicReleaseError('Public source manifest is missing or has an unsupported schema; regenerate it')
    expected = {}
    for row in value['files']:
        if (not isinstance(row, dict) or set(row) != {'path', 'bytes', 'sha256'}
                or type(row['bytes']) is not int or row['bytes'] < 0
                or not re.fullmatch('[0-9a-f]{64}', str(row['sha256']))):
            raise PublicReleaseError('Invalid public source manifest record')
        name = row['path']
        if not allowed(name) or name == 'SOURCE_MANIFEST.json' or name in expected:
            raise PublicReleaseError('Unsafe or duplicate public source manifest path')
        expected[name] = row
    files = collect_sources(root, include_manifest=False) if files is None else files
    actual = {entry.path: entry.record() for entry in files if entry.path != 'SOURCE_MANIFEST.json'}
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        added = sorted(set(actual) - set(expected))
        changed = sorted(name for name in set(expected) & set(actual) if expected[name] != actual[name])
        raise PublicReleaseError('Public source manifest differs: ' + json.dumps(
            {'missing': missing, 'added': added, 'changed': changed}, sort_keys=True))
    return value


def check_tracked_boundary(root=ROOT):
    """Require every present tracked file to belong to the source archive."""
    root = Path(root).resolve(strict=True)
    try:
        response = subprocess.run(['git', '-C', str(root), 'ls-files', '-z'],
                                  capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return  # An extracted source archive intentionally has no .git directory.
    bad = []
    for raw in response.stdout.split(b'\0'):
        if not raw:
            continue
        name = os.fsdecode(raw)
        path = root / name
        if path.exists() or path.is_symlink():
            if not allowed(name) or not stat.S_ISREG(path.lstat().st_mode):
                bad.append(name)
            else:
                # New source formats require an explicit boundary update;
                # publication must never silently exceed the reviewed archive.
                source_bytes(root, name)
    if bad:
        raise PublicReleaseError('Files outside the regular public source boundary remain tracked; paths redacted')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--write-manifest', action='store_true', help='Write a fresh source-only checksum inventory')
    parser.add_argument('--skip-git-index', action='store_true', help='Check an extracted release without a Git index')
    args = parser.parse_args(argv)
    try:
        if not args.skip_git_index:
            check_tracked_boundary(args.root)
        if args.write_manifest:
            write_manifest(args.root, args.manifest)
        files = collect_sources(args.root)
        verify_manifest(args.root, args.manifest, files)
    except (OSError, ValueError) as error:
        parser.exit(1, 'Public release check failed: ' + str(error) + '\n')
    print(json.dumps({'ok': True, 'files': len(files), 'source_bytes': sum(entry.bytes for entry in files),
                      'manifest': str(args.manifest or args.root / 'SOURCE_MANIFEST.json')}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
