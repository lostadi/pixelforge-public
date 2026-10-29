"""Owned root-process lifecycle; no device-specific imports or command paths."""
import ctypes
import os
from pathlib import Path
import signal
import time

PROC_ROOT = Path('/proc')

def process_record(pid):
    try:
        process = Path('/proc') / str(pid)
        fields = (process / 'stat').read_text().rsplit(')', 1)[1].split()
        if process.stat().st_uid != 0 or fields[0] == 'Z':
            return None
        return {'pid': pid, 'parent': int(fields[1]), 'stamp': fields[19]}
    except (OSError, ValueError, IndexError):
        return None


def open_pidfd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # Android Python omits these wrappers; the device kernel exposes the Linux
    # pidfd syscalls at the generic arm64 syscall numbers.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    descriptor = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if descriptor < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    return descriptor


def send_pidfd(descriptor, number):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(descriptor, number)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(424), ctypes.c_int(descriptor), ctypes.c_int(number),
                          ctypes.c_void_p(), ctypes.c_uint(0))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def collect_owned(owned):
    records = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            record = process_record(int(entry.name))
            if record:
                records[record['pid']] = record
    roots = {pid for pid, stamp in owned.items() if pid in records and records[pid]['stamp'] == stamp}
    while True:
        found = {pid for pid, record in records.items() if record['parent'] in roots}
        if found <= roots:
            break
        roots.update(found)
    for pid in roots:
        owned[pid] = records[pid]['stamp']


def collect_owned_children(owned):
    """Discover descendants without scanning unrelated processes periodically.

    Linux records children against the spawning thread, so every live task of
    each validated process is inspected. Full discovery remains the fallback
    when that interface cannot be read, and still owns final cleanup.
    """
    pending = list(owned.items())
    visited = set()

    def fallback():
        collect_owned(owned)
        return False

    def current(pid, stamp):
        record = process_record(pid)
        return record if record and record['stamp'] == stamp else None

    while pending:
        pid, stamp = pending.pop()
        if (pid, stamp) in visited or not current(pid, stamp):
            continue
        visited.add((pid, stamp))
        try:
            tasks = list((PROC_ROOT / str(pid) / 'task').iterdir())
        except OSError:
            if current(pid, stamp):
                return fallback()
            continue
        if not tasks and current(pid, stamp):
            return fallback()
        for task in tasks:
            if not task.name.isdecimal():
                continue
            try:
                with (task / 'children').open('rb') as source:
                    data = source.read(262145)
            except FileNotFoundError:
                if not current(pid, stamp):
                    break
                if task.exists():
                    return fallback()
                continue  # A worker thread exited during this snapshot.
            except OSError:
                if current(pid, stamp):
                    return fallback()
                break
            fields = data.split()
            if len(data) > 262144 or any(not item.isdigit() for item in fields):
                return fallback()
            for field in fields:
                child_pid = int(field)
                if not 0 < child_pid <= 2 ** 31 - 1:
                    return fallback()
                child = process_record(child_pid)
                if not child or child['parent'] != pid or not current(pid, stamp):
                    continue
                child_stamp = child['stamp']
                if child_pid in owned and owned[child_pid] != child_stamp:
                    continue  # Never replace a recorded identity after PID reuse.
                confirmed = current(child_pid, child_stamp)
                if not confirmed or confirmed['parent'] != pid or not current(pid, stamp):
                    continue
                owned[child_pid] = child_stamp
                pending.append((child_pid, child_stamp))
    return True


def signal_owned(owned, number):
    for pid, stamp in list(owned.items()):
        record = process_record(pid)
        if record is None or record['stamp'] != stamp:
            continue
        descriptor = None
        try:
            descriptor = open_pidfd(pid)
            current = process_record(pid)
            if current and current['stamp'] == stamp:
                send_pidfd(descriptor, number)
        except ProcessLookupError:
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)


def cleanup(process, owned):
    # Freeze only the owned tree before its final snapshot, preventing an exit
    # or concurrent fork from orphaning a child during cancellation.
    for _ in range(2):
        collect_owned(owned)
        signal_owned(owned, signal.SIGSTOP)
    signal_owned(owned, signal.SIGTERM)
    signal_owned(owned, signal.SIGCONT)
    deadline = time.monotonic() + 2
    def active():
        for pid, stamp in owned.items():
            record = process_record(pid)
            if record and record['stamp'] == stamp:
                return True
        return False
    while active():
        if time.monotonic() >= deadline:
            signal_owned(owned, signal.SIGKILL)
            break
        time.sleep(.05)
    if not owned and process.poll() is None:
        process.terminate()
    process.wait(timeout=2)
