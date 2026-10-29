#!/data/data/com.termux/files/usr/bin/python
"""Root bootstrap/watchdog for the explicitly enabled ordinary-user optimizer.

Install a root-owned copy as /data/adb/ostadix-optimizer/guardian.py before use.
This module never enables a service, reads process environments, changes Android
permissions, or performs telemetry/tuning. Its child drops to the existing
Termux identity before importing application code. Reboot behavior requires an
actual reboot test; a boot script alone is not proof of reboot persistence.
"""
import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import sys
import threading
import time

APP_ROOT = Path('/data/data/com.termux/files/home/ostadix-voice')
TERMUX_HOME = APP_ROOT.parent
PREFIX = Path('/data/data/com.termux/files/usr')
PYTHON = PREFIX / 'bin/python'
LOCAL_STATE = TERMUX_HOME / '.local/state/ostadix-voice-local'
OPTIMIZER_CONFIG = LOCAL_STATE / 'optimizer/config.json'
SERVICE_CONFIG = LOCAL_STATE / 'service.json'
ROOT_STATE = Path('/data/adb/ostadix-optimizer')
INSTALLED = ROOT_STATE / 'guardian.py'
TERMUX_UID = 10402
ROOT_UID = 0
APP_CONTEXT = 'u:r:untrusted_app_27:s0:c146,c257,c512,c768'
INTERVAL = 15.0
CHILD_TIMEOUT = 20.0
LOG_LIMIT = 256 * 1024

BOOT_SCRIPT = '''#!/system/bin/sh
# Ostadix optimizer bootstrap. Disabled services remain disabled.
# Wait at most five minutes for credential-encrypted Termux home/config.
waited=0
while [ "$waited" -lt 300 ]; do
    if [ -r /data/data/com.termux/files/home/.local/state/ostadix-voice-local/optimizer/config.json ] && [ -x /data/data/com.termux/files/usr/bin/python ]; then
        exec /data/data/com.termux/files/usr/bin/python -I /data/adb/ostadix-optimizer/guardian.py --ensure
    fi
    /system/bin/sleep 5
    waited=$((waited + 5))
done
exit 0
'''


def read_enabled(path):
    """Read only the fixed boolean; absent, malformed or unsafe means disabled."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        with os.fdopen(fd, 'rb') as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_uid != TERMUX_UID:
                return False
            data = source.read(16385)
        if len(data) > 16384:
            return False
        value = json.loads(data)
        return isinstance(value, dict) and value.get('enabled') is True
    except (OSError, ValueError, UnicodeError):
        return False


def prepare_root_state():
    ROOT_STATE.mkdir(mode=0o700, exist_ok=True)
    info = ROOT_STATE.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != ROOT_UID or info.st_mode & 0o022:
        raise RuntimeError('Guardian state must be a root-owned directory without shared write access')


def private_fd(name, flags):
    fd = os.open(ROOT_STATE / name, flags | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != ROOT_UID or info.st_mode & 0o022 or info.st_nlink != 1:
        os.close(fd)
        raise RuntimeError('Unsafe guardian state file: ' + name)
    return fd


@contextlib.contextmanager
def guardian_lock():
    fd = private_fd('guardian.lock', os.O_RDWR | os.O_CREAT)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
        else:
            yield True
    finally:
        os.close(fd)


def process_stamp(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return int(fields[19])


def write_metadata(value):
    data = (json.dumps(value, separators=(',', ':')) + '\n').encode()
    with os.fdopen(private_fd('guardian.json.new', os.O_WRONLY | os.O_CREAT | os.O_TRUNC), 'wb') as stream:
        stream.write(data)
        stream.flush()
    os.replace(ROOT_STATE / 'guardian.json.new', ROOT_STATE / 'guardian.json')


def append_log(name, data):
    """Each of stdout/stderr retains at most one bounded rotated file."""
    if not data:
        return
    data = data[-LOG_LIMIT:]
    with os.fdopen(private_fd(name, os.O_WRONLY | os.O_CREAT | os.O_APPEND), 'ab', buffering=0) as stream:
        if os.fstat(stream.fileno()).st_size + len(data) <= LOG_LIMIT:
            stream.write(data)
            return
    rotated = ROOT_STATE / (name + '.1')
    if rotated.is_symlink():
        raise RuntimeError('Refusing a symlinked rotated log')
    os.replace(ROOT_STATE / name, rotated)
    with os.fdopen(private_fd(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'wb', buffering=0) as stream:
        stream.write(data)


def package_stopped():
    """Respect explicit Android force-stop; never clear or work around it."""
    try:
        result = subprocess.run(['/system/bin/dumpsys', 'package', 'com.termux'],
                                stdin=subprocess.DEVNULL, capture_output=True, timeout=3)
        if result.returncode or len(result.stdout) > 128 * 1024:
            return None
        values = re.findall(rb'^\s*User 0:.*\bstopped=(true|false)\b', result.stdout, re.M)
        return values[0] == b'true' if len(values) == 1 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def resume_command():
    # No configurable commands, interpreter, arguments, UID, context, or import path.
    wrapper = ('import ctypes,os,sys; '
               's=ctypes.CDLL("libselinux.so",use_errno=True); '
               'os.setgroups([3003]); os.setgid(10402); os.setuid(10402); '
               'assert s.setcon(b"' + APP_CONTEXT + '")==0; '
               'os.execv(sys.executable,[sys.executable,"-I",sys.argv[1],"--resume"])')
    return [str(PYTHON), '-I', '-c', wrapper, str(APP_ROOT / 'optimizer.py')]


def child_environment():
    return {'HOME': str(TERMUX_HOME), 'PREFIX': str(PREFIX),
            'TMPDIR': str(PREFIX / 'tmp'), 'PATH': str(PREFIX / 'bin') + ':/system/bin',
            'ANDROID_ROOT': '/system', 'ANDROID_DATA': '/data', 'ANDROID_STORAGE': '/storage',
            'LANG': 'C.UTF-8'}


def run_resume(stop):
    started = time.monotonic()
    process = subprocess.Popen(resume_command(), env=child_environment(), cwd=APP_ROOT,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True)
    result = {'pid': process.pid, 'child_uid': TERMUX_UID, 'reason': 'completed'}
    counts = {'stdout.log': 0, 'stderr.log': 0}
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, 'stdout.log')
            selector.register(process.stderr, selectors.EVENT_READ, 'stderr.log')
            while process.poll() is None or selector.get_map():
                if stop.is_set() or not read_enabled(OPTIMIZER_CONFIG) or not read_enabled(SERVICE_CONFIG):
                    result['reason'] = 'disabled_or_stopping'
                    break
                if time.monotonic() - started >= CHILD_TIMEOUT:
                    result['reason'] = 'resume_timeout'
                    break
                for key, _ in selector.select(.25):
                    data = os.read(key.fileobj.fileno(), 8192)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    counts[key.data] += len(data)
                    append_log(key.data, data)
                if any(value > LOG_LIMIT for value in counts.values()):
                    result['reason'] = 'resume_output_limit'
                    break
    finally:
        # Only the exact child created above can be signalled, never its group.
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        process.stdout.close()
        process.stderr.close()
    result.update(exit_code=process.returncode, elapsed_seconds=round(time.monotonic() - started, 3))
    return result


def watch(stop=None):
    stop = stop or threading.Event()
    with guardian_lock() as acquired:
        if not acquired:
            return 0
        metadata = {'schema': 'ostadix.optimizer-guardian/v1', 'uid': os.getuid(),
                    'pid': os.getpid(), 'starttime_ticks': process_stamp(os.getpid()),
                    'phase': 'starting', 'last_resume': None}
        def record(phase, **changes):
            metadata.update(phase=phase, updated=time.time(), **changes)
            write_metadata(metadata)
        try:
            while not stop.is_set():
                started = time.monotonic()
                if not read_enabled(OPTIMIZER_CONFIG):
                    record('disabled')
                    return 0
                if not read_enabled(SERVICE_CONFIG):
                    record('local_service_disabled')
                else:
                    stopped = package_stopped()
                    if stopped is not False:
                        record('termux_force_stopped' if stopped else 'package_state_unavailable')
                    else:
                        record('resuming')
                        try:
                            result = run_resume(stop)
                            record('watching', last_resume=result)
                        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                            record('resume_error', error=type(error).__name__)
                stop.wait(max(0.0, INTERVAL - (time.monotonic() - started)))
            record('stopped')
        finally:
            if metadata['phase'] not in ('disabled', 'stopped'):
                record('stopped')
    return 0


def ensure():
    if not read_enabled(OPTIMIZER_CONFIG):
        return {'running': False, 'reason': 'optimizer_disabled'}
    with guardian_lock() as acquired:
        if not acquired:
            return {'running': True, 'reason': 'already_running'}
    child = subprocess.Popen([str(PYTHON), '-I', str(INSTALLED), '--watch'],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, env=child_environment(),
                             start_new_session=True, close_fds=True)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with guardian_lock() as acquired:
            if not acquired:
                return {'running': True, 'reason': 'started', 'pid': child.pid}
        if child.poll() is not None:
            return {'running': False, 'reason': 'watcher_exited', 'exit_code': child.returncode}
        time.sleep(.05)
    # Do not start another watcher or kill one that may have just acquired its lock.
    return {'running': False, 'reason': 'startup_not_confirmed', 'pid': child.pid}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--ensure', action='store_true')
    actions.add_argument('--watch', action='store_true')
    actions.add_argument('--print-boot-script', action='store_true')
    args = parser.parse_args()
    if args.print_boot_script:
        print(BOOT_SCRIPT, end='')
        return 0
    if os.getuid() != ROOT_UID:
        parser.error('Guardian management requires the existing root context')
    info = INSTALLED.lstat()
    if (Path(__file__).resolve() != INSTALLED or not stat.S_ISREG(info.st_mode)
            or info.st_uid != ROOT_UID or info.st_mode & 0o022):
        parser.error('Use only the root-owned installed guardian copy')
    prepare_root_state()
    if args.ensure:
        result = ensure()
        print(json.dumps(result))
        return 0 if result['running'] or result['reason'] == 'optimizer_disabled' else 1
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda _s, _f: stop.set())
    return watch(stop)


if __name__ == '__main__':
    sys.exit(main())
