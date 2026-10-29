"""Local installation integrity helpers; profiles contain metadata, never credentials."""
import hashlib
import json
import os
from pathlib import Path
import stat


def sha256(path):
    digest = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as source:
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError('Pinned input must be a regular file')
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def root_json(path, *, owner=0, limit=262144):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != owner or info.st_mode & 0o022:
            raise ValueError('Configuration must be an owned, non-writable regular file')
        content = source.read(limit + 1)
    if len(content) > limit:
        raise ValueError('Configuration exceeds its size limit')
    result = json.loads(content)
    if not isinstance(result, dict):
        raise ValueError('Configuration must be a JSON object')
    return result
