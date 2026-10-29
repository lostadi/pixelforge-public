#!/usr/bin/env python3
"""Verify and install the separate GGUF runtime from its offline addon.

This entry point uses only the standard library. Chosen GGUF weights are
separate from the runtime addon and no MiniCPM checkpoint pins are required.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import venv

from runtime import checked_file, install_native


KIT = Path(__file__).resolve().parent
PUBLIC_ASSETS = KIT / 'GGUF_RUNTIME_ASSETS.json'
DEFAULT_ADDON = KIT.parents[1] / 'custom-models/gguf-runtime'
NATIVE_DESTINATION = Path('/data/local/tmp/pixelforge-npu/npu_pipe_gguf')
ENGINE_REVISION = '254b177307b14d055bbe4c050c9c441a1e06d8a4'
REQUIRED_FILES = frozenset({'bin/pixelforge-gguf', 'native/npu_pipe_gguf',
                            'source/llama-source.tar.gz', 'requirements-offline.txt'})


def manifest_path(root, name):
    """Reject escapes, aliases and symlinks before opening any addon file."""
    if not isinstance(name, str) or '\\' in name:
        raise ValueError('Manifest path must be a canonical relative POSIX path')
    relative = PurePosixPath(name)
    if (relative.is_absolute() or not relative.parts or '..' in relative.parts
            or relative.as_posix() != name):
        raise ValueError('Manifest path must stay inside the addon without aliases')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f'Redirected addon path: {name}')
    if not current.is_file() or not current.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Missing or non-regular addon file: {name}')
    return current


def verify_addon(root):
    """Validate every shipped file before any installation or native action."""
    root = Path(root)
    if not root.is_dir() or not (root / 'MANIFEST.json').is_file():
        raise FileNotFoundError(
            f'Offline GGUF runtime addon is missing: {root}. '
            'Provide its MANIFEST.json, ARM binaries, pinned source archive and '
            'offline wheels; see tools/arm_npu/GGUF_COMPILER.md for ARM build instructions.')
    manifest_file = manifest_path(root, 'MANIFEST.json')
    manifest = json.loads(manifest_file.read_text(encoding='utf8'))
    if not isinstance(manifest, dict) or not isinstance(manifest.get('files'), list):
        raise ValueError('GGUF addon manifest must contain a files list')
    if manifest.get('engine_revision') != ENGINE_REVISION:
        raise ValueError('GGUF addon does not match the pinned architecture engine revision')
    entries = {}
    for item in manifest['files']:
        if not isinstance(item, dict):
            raise ValueError('Malformed GGUF addon file entry')
        name = item.get('path')
        manifest_path(root, name)
        if name in entries:
            raise ValueError(f'Duplicate addon manifest path: {name}')
        if type(item.get('bytes')) is not int or item['bytes'] < 0:
            raise ValueError(f'Invalid addon file size: {name}')
        if not isinstance(item.get('sha256'), str) or not re.fullmatch('[0-9a-f]{64}', item['sha256']):
            raise ValueError(f'Invalid addon SHA256: {name}')
        entries[name] = item
    missing = REQUIRED_FILES - entries.keys()
    if missing:
        raise ValueError('GGUF runtime addon needs verified files: ' + ', '.join(sorted(missing)))
    # The addon cannot authorize its own replacement binaries or dependencies.
    # Bind all shipped public pins before hashing files or starting setup.
    public = json.loads(PUBLIC_ASSETS.read_text(encoding='utf8'))
    if (not isinstance(public, dict)
            or public.get('format') != 'pixelforge.gguf-runtime.assets.v1'
            or public.get('engine_revision') != ENGINE_REVISION
            or not isinstance(public.get('files'), list)):
        raise ValueError('Invalid public GGUF runtime asset pins')
    public_paths = set()
    for pin in public['files']:
        if (not isinstance(pin, dict) or not isinstance(pin.get('path'), str)
                or type(pin.get('bytes')) is not int or pin['bytes'] < 0
                or not isinstance(pin.get('sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', pin['sha256'])):
            raise ValueError('Invalid public GGUF runtime asset pin')
        name = pin['path']
        if name in public_paths:
            raise ValueError(f'Duplicate public GGUF runtime asset pin: {name}')
        public_paths.add(name)
        if name not in entries:
            raise ValueError(f'GGUF runtime addon needs verified public file: {name}')
        item = entries[name]
        if item['bytes'] != pin['bytes'] or item['sha256'] != pin['sha256']:
            raise ValueError(f'Public GGUF runtime asset pin mismatch: {name}')
    if not REQUIRED_FILES.issubset(public_paths):
        raise ValueError('Public GGUF runtime asset pins omit required files')
    wheels = [name for name in entries if name.startswith('wheels/') and name.endswith('.whl')]
    if not wheels:
        raise ValueError('GGUF runtime addon needs verified offline wheels')
    shipped_wheels = {path.relative_to(root).as_posix() for path in (root / 'wheels').glob('*.whl')}
    if shipped_wheels != set(wheels):
        raise ValueError('Every wheel must be listed exactly once in the addon manifest')
    for item in entries.values():
        checked_file(root, item)
    return manifest


def setup(root):
    root = Path(root)
    manifest = verify_addon(root)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Bundled wheels require CPython 3.12 on Linux ARM64')
    if sys.platform != 'linux' or os.uname().machine != 'aarch64':
        raise RuntimeError('Bundled wheels and native helper require Linux ARM64')
    if os.geteuid() != 0:
        raise PermissionError('Native helper setup requires this rooted Android/Ubuntu session')
    environment = root / 'venv'
    if environment.is_symlink() or (environment.exists() and not environment.is_dir()):
        raise ValueError('Runtime venv must be a local directory')
    if not (environment / 'bin/python').exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([str(environment / 'bin/python'), '-I', '-m', 'pip', 'install',
                    '--no-index', '--no-deps', '--require-hashes', '--force-reinstall',
                    '--find-links', str(root / 'wheels'),
                    '-r', str(root / 'requirements-offline.txt')], check=True)
    install_native(root / 'native/npu_pipe_gguf', destination=NATIVE_DESTINATION)
    result = {'status': 'offline-gguf-setup-complete',
              'verified_files': len(manifest['files']),
              'engine_revision': manifest['engine_revision'],
              'python': str(environment / 'bin/python'),
              'engine': str(root / 'bin/pixelforge-gguf'),
              'native': str(NATIVE_DESTINATION)}
    print(json.dumps(result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--addon', type=Path, default=DEFAULT_ADDON)
    parser.add_argument('--verify-only', action='store_true', help='Hash files without installing anything')
    args = parser.parse_args(argv)
    try:
        if args.verify_only:
            manifest = verify_addon(args.addon)
            print(json.dumps({'status': 'offline-gguf-addon-verified',
                              'verified_files': len(manifest['files']),
                              'engine_revision': manifest['engine_revision']}))
        else:
            setup(args.addon)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f'GGUF setup failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
