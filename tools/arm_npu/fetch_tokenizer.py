"""Fetch pinned public MiniCPM5 tokenizer files and verify SHA256 before use.

No model checkpoint is downloaded and no remote Python code is executed.
Existing files are reused only when they match the pinned manifest; mismatches
are reported and never overwritten. The largest download is about10MB.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
import urllib.request


MANIFEST_PATH = pathlib.Path(__file__).with_name('HUIHUI_ASSETS.json')


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        while data := stream.read(1024*1024):
            digest.update(data)
    return digest.hexdigest()


def fetch(destination, manifest=None):
    manifest = manifest or json.loads(MANIFEST_PATH.read_text())
    destination = pathlib.Path(destination)
    if destination.is_symlink():
        raise ValueError('Tokenizer destination must not be a symlink')
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    records = []
    for asset in manifest['support_files']:
        name = asset['path']
        if pathlib.PurePath(name).name != name:
            raise ValueError('Manifest assets must have a flat filename')
        path = destination/name
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or path.stat().st_size != asset['bytes'] or sha256(path) != asset['sha256']:
                raise ValueError(f'Existing tokenizer asset differs from pinned content: {path}')
            records.append({'path':str(path), 'status':'verified-existing'})
            continue
        descriptor, temporary = tempfile.mkstemp(prefix=f'.{name}.', dir=destination)
        temporary = pathlib.Path(temporary)
        try:
            digest = hashlib.sha256()
            total = 0
            request = urllib.request.Request(asset['source_url'], headers={'User-Agent':'PixelForge-pinned-tokenizer'})
            with os.fdopen(descriptor, 'wb') as output:
                with urllib.request.urlopen(request, timeout=30) as response:
                    while data := response.read(min(1024*1024, asset['bytes']-total+1)):
                        total += len(data)
                        if total > asset['bytes']:
                            raise ValueError(f'Tokenizer asset exceeds pinned byte count: {name}')
                        digest.update(data)
                        output.write(data)
                output.flush()
                os.fsync(output.fileno())
            if total != asset['bytes'] or digest.hexdigest() != asset['sha256']:
                raise ValueError(f'Tokenizer asset failed pinned SHA256 verification: {name}')
            # A hardlink publishes the completed download exclusively; a file
            # created by another process during fetch is never replaced.
            os.link(temporary, path, follow_symlinks=False)
            records.append({'path':str(path), 'status':'downloaded-verified', 'bytes':total})
        finally:
            temporary.unlink(missing_ok=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    print(json.dumps({'tokenizer_assets':fetch(args.destination)}, indent=2))


if __name__ == '__main__':
    main()
