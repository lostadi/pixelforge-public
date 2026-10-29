#!/usr/bin/env python3
"""Export only the GGUF compiler and pinned offline runtime as an executable archive."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tarfile

from setup_gguf import ENGINE_REVISION, verify_addon

REPO = Path(__file__).resolve().parents[2]
# Explicit public inputs prevent private diagnostics or credentials from being
# swept into an export merely because they appear beside source files.
SOURCE_NAMES = frozenset('''
GGUF_RUNTIME_ASSETS.json test_gguf_run.py test_setup_gguf.py setup_gguf.py
GGUF_COMPILER.md GGUF_READER.md gguf.sh run_gguf.py test_gguf_loader.py
create_tiny_gguf.py llama_npu_bridge.patch build_llama_npu.sh ggml_npu_bridge.cc
ggml_npu_bridge.h test_compiled_model.py gguf_generate.cc gguf_inspect.py
gguf_broker.py model_architecture.py LLAMA_ENGINE.json gguf_vendor/__init__.py
gguf_vendor/quants.py gguf_vendor/lazy.py gguf_vendor/constants.py compile_gguf.py
GGUF_QUANTIZATION_PROVENANCE.json gguf_quantizers.py compiled_model.py
test_pipe_protocol.cc pipe_protocol.h test_runtime.py runtime.py PYTHON_RUNTIME.md
validate_runtime.py test_npu_protocol.py SCHEMA_PROVENANCE.json npu_linear.py
build_constant_fc.py schema_py_generated.py NATIVE_COMPILER.md build_native.sh
UPSTREAM_APIS.json NATIVE_VALIDATION.json LICENSE.litert sb_api.h
edgetpu_compiler_api.h npu_pipe.cc test_host_assets.py fetch_tokenizer.py
HUIHUI_ASSETS.json LICENSE.ggml model_host.py gguf_hf_loader.py
pack_gguf.py test_pack_gguf.py
'''.split())

README = b'''# PixelForge GGUF compiler

This package contains the ARM compiler source, documentation, pinned architecture
engine and offline runtime dependencies. Supply your own GGUF model separately.

On the supported rooted Android device with Ubuntu ARM64 and Python 3.12:

```sh
python3 tools/arm_npu/setup_gguf.py
bash tools/arm_npu/gguf.sh inspect /path/to/your-model.gguf
```

Replace /path/to/your-model.gguf with the GGUF model you choose.

Read tools/arm_npu/GGUF_COMPILER.md for compilation, model support, inference,
benchmarks and the explicit CPU/NPU execution boundary. Third-party licenses
are under tools/arm_npu/.
'''


def extractor_header(offset, digest):
    """Fixed-width fields allow streaming the payload before filling its hash."""
    return f'''#!/bin/sh
exec python3 - "$0" "$@" <<'PIXELFORGE_GGUF_EXTRACTOR'
import hashlib
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import sys
import tarfile

OFFSET = int("{offset:020d}")
PAYLOAD_SHA256 = "{digest}"
if len(sys.argv) > 3:
    raise SystemExit("Usage: " + sys.argv[1] + " [fresh-destination]")
destination = Path(sys.argv[2] if len(sys.argv) == 3 else "pixelforge-gguf").expanduser().absolute()
if destination.exists() or destination.is_symlink():
    raise SystemExit("Destination exists; choose a fresh directory: " + str(destination))
if not destination.parent.is_dir():
    raise SystemExit("Destination parent must already exist: " + str(destination.parent))
owned = None
try:
    with open(sys.argv[1], "rb") as packed:
        packed.seek(OFFSET)
        digest = hashlib.sha256()
        for chunk in iter(lambda: packed.read(1024 * 1024), b""):
            digest.update(chunk)
        if digest.hexdigest() != PAYLOAD_SHA256:
            raise ValueError("Compiler archive payload SHA256 mismatch")
        destination.mkdir(mode=0o700)
        owned = destination.stat()
        packed.seek(OFFSET)
        names = set()
        total = 0
        with tarfile.open(fileobj=packed, mode="r|gz") as archive:
            for member in archive:
                relative = PurePosixPath(member.name)
                if (not member.isfile() or relative.is_absolute() or not relative.parts
                        or ".." in relative.parts or relative.as_posix() != member.name
                        or member.name in names):
                    raise ValueError("Unsafe compiler archive member: " + member.name)
                names.add(member.name)
                total += member.size
                if len(names) > 512 or member.size < 0 or member.size > 512 * 1024**2 or total > 2 * 1024**3:
                    raise ValueError("Compiler archive exceeds bounded extraction limits")
                target = destination.joinpath(*relative.parts)
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as output:
                    remaining = member.size
                    while remaining:
                        chunk = source.read(min(remaining, 1024 * 1024))
                        if not chunk:
                            raise ValueError("Truncated compiler archive member: " + member.name)
                        output.write(chunk)
                        remaining -= len(chunk)
                target.chmod(0o700 if member.mode & 0o111 else 0o600)
    sys.path.insert(0, str(destination / "tools/arm_npu"))
    from setup_gguf import verify_addon
    addon = destination / "custom-models/gguf-runtime"
    manifest = verify_addon(addon)
    print("Extracted compiler and " + str(len(manifest["files"])) + " verified runtime assets into " + str(destination))
    print("Set up: " + shlex.join(["python3", str(destination / "tools/arm_npu/setup_gguf.py"), "--addon", str(addon)]))
except BaseException:
    if owned is not None:
        try:
            current = destination.lstat()
            if not destination.is_symlink() and (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                shutil.rmtree(destination)
        except FileNotFoundError:
            pass
    raise
PIXELFORGE_GGUF_EXTRACTOR
'''.encode('utf8')


class HashingWriter:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data):
        written = self.stream.write(data)
        self.digest.update(memoryview(data)[:written])
        return written


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()
        self.bytes = 0

    def read(self, size):
        data = self.stream.read(size)
        self.digest.update(data)
        self.bytes += len(data)
        return data


def file_identity(metadata):
    return (metadata.st_dev, metadata.st_ino, metadata.st_size,
            metadata.st_mtime_ns, metadata.st_ctime_ns)


@dataclass(frozen=True)
class ExportFile:
    name: str
    path: Path
    size: int
    sha256: str
    identity: tuple
    mode: int


def snapshot_file(name, path, expected=None):
    """Hash an immutable preflight snapshot without loading the file into RAM."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError('Non-regular compiler export input: ' + name)
    identity = file_identity(before)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as source:
        if file_identity(os.fstat(source.fileno())) != identity:
            raise ValueError('Changed compiler export input: ' + name)
        digest = hashlib.sha256()
        size = 0
        for data in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(data)
            size += len(data)
        if (file_identity(os.fstat(source.fileno())) != identity
                or file_identity(path.lstat()) != identity):
            raise ValueError('Changed compiler export input: ' + name)
    actual = {'bytes': size, 'sha256': digest.hexdigest()}
    if expected is not None and any(actual[key] != expected[key] for key in actual):
        raise ValueError('Compiler export pin mismatch: ' + name)
    return ExportFile(name, path, size, actual['sha256'], identity, before.st_mode)


def export_files(repo):
    kit = repo / 'tools/arm_npu'
    addon = repo / 'custom-models/gguf-runtime'
    manifest = verify_addon(addon)
    public_bytes = (kit / 'GGUF_RUNTIME_ASSETS.json').read_bytes()
    public = json.loads(public_bytes)
    if (not isinstance(public, dict)
            or public.get('format') != 'pixelforge.gguf-runtime.assets.v1'
            or public.get('engine_revision') != ENGINE_REVISION
            or not isinstance(public.get('files'), list)):
        raise ValueError('Invalid captured public compiler runtime asset pins')
    pins = {}
    for item in public['files']:
        if (not isinstance(item, dict) or not isinstance(item.get('path'), str)
                or type(item.get('bytes')) is not int or item['bytes'] < 0
                or not isinstance(item.get('sha256'), str)
                or not re.fullmatch('[0-9a-f]{64}', item['sha256'])
                or item['path'] in pins):
            raise ValueError('Invalid captured public compiler runtime asset pin')
        pins[item['path']] = item
    validated = {item['path']: item for item in manifest['files']}
    if set(pins) != set(validated):
        raise ValueError('Compiler export requires exactly the publicly pinned runtime assets')
    for name, pin in pins.items():
        if any(pin[field] != validated[name][field] for field in ('bytes', 'sha256')):
            raise ValueError('Captured public and validated private runtime pins disagree: ' + name)
    manifest_bytes = (addon / 'MANIFEST.json').read_bytes()
    if json.loads(manifest_bytes) != manifest:
        raise ValueError('Validated compiler runtime manifest changed during export preflight')
    expected = {f'custom-models/gguf-runtime/{name}': pin for name, pin in pins.items()}
    for name, data in (('tools/arm_npu/GGUF_RUNTIME_ASSETS.json', public_bytes),
                       ('custom-models/gguf-runtime/MANIFEST.json', manifest_bytes)):
        expected[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    names = {'benchmark.py'} | {f'tools/arm_npu/{name}' for name in SOURCE_NAMES}
    names |= {f'custom-models/gguf-runtime/{name}' for name in set(pins) | {'MANIFEST.json'}}
    for license_name in ('LICENSE', 'COPYING'):
        if (repo / license_name).is_file():
            names.add(license_name)
    result = []
    for name in sorted(names):
        path = repo / name
        current = repo
        for part in Path(name).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError('Redirected compiler export input: ' + name)
        if not path.is_file():
            raise ValueError('Missing compiler export input: ' + name)
        result.append(snapshot_file(name, path, expected.get(name)))
    return result


def pack(output, *, repo=REPO):
    repo = Path(repo).resolve()
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError('Output exists: ' + str(output))
    output = output.resolve()
    if output.is_relative_to(repo):
        raise ValueError('Write the compiler archive outside the repository')
    files = export_files(repo)
    header = extractor_header(0, '0' * 64)
    header = extractor_header(len(header), '0' * 64)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    descriptor = os.open(output, flags, 0o600)
    owned = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, 'w+b') as stream:
            stream.write(header)
            payload = HashingWriter(stream)
            with tarfile.open(fileobj=payload, mode='w|gz', format=tarfile.PAX_FORMAT) as archive:
                info = tarfile.TarInfo('README.md')
                info.size = len(README)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(README))
                for selected in files:
                    name, path = selected.name, selected.path
                    if file_identity(path.lstat()) != selected.identity:
                        raise ValueError('Changed compiler export input: ' + name)
                    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as source:
                        metadata = os.fstat(source.fileno())
                        if not stat.S_ISREG(metadata.st_mode) or file_identity(metadata) != selected.identity:
                            raise ValueError('Changed compiler export input: ' + name)
                        info = tarfile.TarInfo(name)
                        info.size = selected.size
                        info.mode = 0o755 if selected.mode & 0o111 else 0o644
                        reader = HashingReader(source)
                        archive.addfile(info, reader)
                        if (reader.bytes != selected.size or reader.digest.hexdigest() != selected.sha256
                                or file_identity(os.fstat(source.fileno())) != selected.identity
                                or file_identity(path.lstat()) != selected.identity):
                            raise ValueError('Changed compiler export input while streaming: ' + name)
            completed = extractor_header(len(header), payload.digest.hexdigest())
            if len(completed) != len(header):
                raise RuntimeError('Compiler archive header size changed')
            stream.seek(0)
            stream.write(completed)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), 0o700)
            packed_bytes = os.fstat(stream.fileno()).st_size
        current = output.lstat()
        if (current.st_dev, current.st_ino) != (owned.st_dev, owned.st_ino):
            raise ValueError('Compiler archive output changed owner')
    except BaseException:
        try:
            current = output.lstat()
            if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                output.unlink()
        except FileNotFoundError:
            pass
        raise
    runtime_assets = sum(selected.name.startswith('custom-models/gguf-runtime/')
                         and selected.name != 'custom-models/gguf-runtime/MANIFEST.json'
                         for selected in files)
    return {'self_extracting_archive': str(output), 'bytes': packed_bytes,
            'files': len(files) + 1, 'runtime_assets': runtime_assets}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, help='Fresh archive path outside the repository')
    args = parser.parse_args(argv)
    print(json.dumps(pack(args.output)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
