#!/usr/bin/env python3
"""Create a deterministic, verified public source archive and checksum receipts."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile

from check_public_release import (ROOT, SCHEMA, MAX_SOURCE_FILE, PublicReleaseError, allowed,
    check_tracked_boundary, collect_sources, inspect_bytes, safe_relative,
    source_bytes, verify_manifest)


def tar_entry(name, data, mode, epoch):
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = mode
    info.mtime = epoch
    info.uid = info.gid = 0
    info.uname = info.gname = ''
    return info, io.BytesIO(data)


def verify_archive(path):
    """Verify all members before extraction; reject links, traversal and extras."""
    files = {}
    prefix = None
    total = 0
    with tarfile.open(path, 'r:gz') as archive:
        for member in archive:
            name = safe_relative(member.name)
            if not member.isfile() or len(name.parts) < 2:
                raise PublicReleaseError('Source archive contains a nonregular or unrooted member')
            if prefix is None:
                prefix = name.parts[0]
            if name.parts[0] != prefix:
                raise PublicReleaseError('Source archive has multiple root directories')
            relative = PurePosixPath(*name.parts[1:]).as_posix()
            if relative in files:
                raise PublicReleaseError('Source archive has duplicate members')
            if relative != 'RELEASE_MANIFEST.json' and not allowed(relative):
                raise PublicReleaseError('Source archive contains a member outside the public boundary')
            if member.mode not in (0o644, 0o755):
                raise PublicReleaseError('Source archive contains unsafe file permissions')
            total += member.size
            if not 0 <= member.size <= MAX_SOURCE_FILE or total > 256 * 1024 * 1024:
                raise PublicReleaseError('Source archive exceeds its inspection limits')
            content = archive.extractfile(member).read()
            if len(content) != member.size:
                raise PublicReleaseError('Source archive contains a truncated member')
            if relative != 'RELEASE_MANIFEST.json':
                inspect_bytes(relative, content)
            files[relative] = content
    if 'RELEASE_MANIFEST.json' not in files:
        raise PublicReleaseError('Source archive has no release manifest')
    manifest = json.loads(files.pop('RELEASE_MANIFEST.json'))
    if manifest.get('schema') != 'pixelforge.source-release.v1':
        raise PublicReleaseError('Unsupported source archive manifest schema')
    rows = manifest.get('files')
    if not isinstance(rows, list):
        raise PublicReleaseError('Invalid source archive manifest')
    expected = {}
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {'path', 'bytes', 'sha256'}
                or type(row['bytes']) is not int or row['bytes'] < 0
                or not re.fullmatch('[0-9a-f]{64}', str(row['sha256']))):
            raise PublicReleaseError('Invalid source archive manifest record')
        name = row['path']
        if not allowed(name) or name in expected:
            raise PublicReleaseError('Unsafe or duplicate source archive manifest path')
        expected[name] = row
    actual = {name: {'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
              for name, data in files.items()}
    if expected != actual:
        raise PublicReleaseError('Source archive contents do not match the release manifest')
    if 'SOURCE_MANIFEST.json' not in files:
        raise PublicReleaseError('Source archive has no public source inventory')
    index = json.loads(files['SOURCE_MANIFEST.json'])
    if index.get('schema') != SCHEMA or not isinstance(index.get('files'), list):
        raise PublicReleaseError('Source archive public inventory has an unsupported schema')
    indexed = {}
    for row in index['files']:
        if (not isinstance(row, dict) or set(row) != {'path', 'bytes', 'sha256'}
                or row['path'] in indexed or row['path'] == 'SOURCE_MANIFEST.json'
                or not allowed(row['path'])):
            raise PublicReleaseError('Source archive public inventory has an invalid record')
        indexed[row['path']] = row
    if indexed != {name: row for name, row in actual.items() if name != 'SOURCE_MANIFEST.json'}:
        raise PublicReleaseError('Source archive public inventory differs from its source files')
    return {'prefix': prefix, 'files': len(files), 'source_bytes': sum(map(len, files.values()))}


def build(root, output, *, version='source', epoch=0, manifest=None, check_git=True):
    root = Path(root).resolve(strict=True)
    output = Path(output).expanduser().absolute()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._+-]{0,79}', version) or not 0 <= epoch <= 0xffffffff:
        raise PublicReleaseError('Use a simple version label and nonnegative source epoch')
    if manifest is not None and Path(manifest).resolve() != root / 'SOURCE_MANIFEST.json':
        raise PublicReleaseError('Source archives require ROOT/SOURCE_MANIFEST.json as the embedded inventory')
    if not output.parent.is_dir():
        raise PublicReleaseError('Output parent directory does not exist')
    if output.is_relative_to(root) and output.relative_to(root).parts[0] != 'dist':
        raise PublicReleaseError('Write release artifacts outside source directories, or inside dist/')
    if check_git:
        check_tracked_boundary(root)
    files = collect_sources(root)
    verify_manifest(root, manifest, files)
    release = {'schema': 'pixelforge.source-release.v1', 'version': version,
               'source_date_epoch': epoch, 'files': [entry.record() for entry in files]}
    release_bytes = (json.dumps(release, sort_keys=True, indent=2) + '\n').encode()
    prefix = 'pixelforge-' + version
    targets = [output, Path(str(output) + '.sha256'), Path(str(output) + '.json')]
    owned = {}
    streams = []
    try:
        for target in targets:
            descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
            metadata = os.fstat(descriptor)
            owned[target] = (metadata.st_dev, metadata.st_ino)
            streams.append(os.fdopen(descriptor, 'wb'))
        with gzip.GzipFile(filename='', mode='wb', fileobj=streams[0], mtime=epoch) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as archive:
                for entry in files:
                    data, mode = source_bytes(root, entry.path)
                    if len(data) != entry.bytes or hashlib.sha256(data).hexdigest() != entry.sha256:
                        raise PublicReleaseError('Source changed during archive creation: ' + entry.path)
                    info, content = tar_entry(prefix + '/' + entry.path, data, mode, epoch)
                    archive.addfile(info, content)
                info, content = tar_entry(prefix + '/RELEASE_MANIFEST.json', release_bytes, 0o644, epoch)
                archive.addfile(info, content)
        streams[0].flush()
        os.fsync(streams[0].fileno())
        verified = verify_archive(output)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        streams[1].write((digest + '  ' + output.name + '\n').encode())
        receipt = dict(schema='pixelforge.source-artifact.v1', archive=output.name,
                       bytes=output.stat().st_size, sha256=digest, **verified)
        streams[2].write((json.dumps(receipt, sort_keys=True, indent=2) + '\n').encode())
        for stream in streams:
            stream.flush()
            os.fsync(stream.fileno())
        return receipt
    except BaseException:
        for target, identity in owned.items():
            try:
                metadata = target.lstat()
                if stat.S_ISREG(metadata.st_mode) and (metadata.st_dev, metadata.st_ino) == identity:
                    target.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for stream in streams:
            stream.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--version', default='source')
    parser.add_argument('--epoch', type=int, default=int(os.environ.get('SOURCE_DATE_EPOCH', '0')))
    parser.add_argument('--verify', type=Path, help='Verify an existing archive without extracting or modifying it')
    args = parser.parse_args(argv)
    if args.verify is None and args.output is None:
        parser.error('Provide --output or --verify')
    if args.verify is not None and args.output is not None:
        parser.error('--verify and --output are separate operations')
    try:
        value = verify_archive(args.verify) if args.verify else build(
            args.root, args.output, version=args.version, epoch=args.epoch, manifest=args.manifest)
    except (OSError, ValueError, tarfile.TarError) as error:
        parser.exit(1, 'Source release failed: ' + str(error) + '\n')
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
