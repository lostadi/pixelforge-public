"""Fixed stdin-only Android root entrypoint; local bootstrap pins its Ubuntu code."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import stat
import sys


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read_bootstrap(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, 'rb') as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError('Bootstrap must be a root-owned regular file')
        data = source.read(262145)
    if len(data) > 262144:
        raise ValueError('Bootstrap exceeds its size limit')
    config = json.loads(data)
    required = {'ubuntu_root', 'python', 'service', 'ubuntu_pins'}
    if set(config) != required or not config['ubuntu_pins']:
        raise ValueError('Invalid bootstrap fields')
    for key in ('ubuntu_root', 'python', 'service'):
        if not isinstance(config[key], str) or not Path(config[key]).is_absolute() or '..' in Path(config[key]).parts:
            raise ValueError('Bootstrap paths must be absolute')
    for path, expected in config['ubuntu_pins'].items():
        if not Path(path).is_absolute() or '..' in Path(path).parts or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError('Invalid Ubuntu source pin')
    if config['python'] not in config['ubuntu_pins'] or config['service'] not in config['ubuntu_pins']:
        raise ValueError('Bootstrap must pin the interpreter and service')
    return config


def process_stamp(pid):
    proc = Path('/proc') / str(pid)
    try:
        fields = (proc / 'stat').read_text().rsplit(')', 1)[1].split()
        if proc.stat().st_uid != 0 or fields[0] == 'Z':
            return None
        return fields[19]
    except (OSError, ValueError, IndexError):
        return None


def ubuntu_handles(config):
    """Find an already-mounted root session, checking its root inode and pins."""
    expected = Path(config['ubuntu_root']).stat()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        stamp = process_stamp(int(proc.name))
        if stamp is None:
            continue
        root = proc / 'root'
        try:
            info = root.stat()
            if (info.st_dev, info.st_ino) != (expected.st_dev, expected.st_ino):
                continue
            for absolute, pin in config['ubuntu_pins'].items():
                candidate = root / absolute.lstrip('/')
                info = candidate.stat()
                if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022 or digest(candidate) != pin:
                    raise ValueError('Installed Ubuntu source or interpreter pin changed')
            root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            try:
                namespace_fd = os.open(proc / 'ns/mnt', os.O_RDONLY | os.O_CLOEXEC)
            except BaseException:
                os.close(root_fd)
                raise
            if process_stamp(int(proc.name)) == stamp:
                return root_fd, namespace_fd
            os.close(root_fd)
            os.close(namespace_fd)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    raise RuntimeError('Open the configured Ubuntu root environment first; no verified mounted session exists')


def main():
    installed = Path(__file__).absolute()
    info = installed.lstat()
    parent = installed.parent.stat()
    if (os.getuid() != 0 or len(sys.argv) != 1 or not stat.S_ISREG(info.st_mode)
            or info.st_uid != 0 or info.st_mode & 0o022 or parent.st_uid != 0 or parent.st_mode & 0o022):
        raise ValueError('Use the installed root-owned stdin-only chat launcher')
    config = read_bootstrap(installed.with_name('bootstrap.json'))
    root_fd, namespace_fd = ubuntu_handles(config)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.setns(namespace_fd, 0):
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))
        os.fchdir(root_fd)
        os.chroot('.')
        os.chdir(str(Path(config['service']).parent))
    finally:
        os.close(root_fd)
        os.close(namespace_fd)
    os.execve(config['python'], [config['python'], '-I', config['service']],
              {'HOME': '/root', 'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({'type': 'error', 'error': str(error)[:500]}), flush=True)
        raise SystemExit(1)
