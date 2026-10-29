"""Watch an O owner and release only this worker's Linux child processes."""
from __future__ import annotations

import argparse
import ctypes
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time


_LIBC = ctypes.CDLL(None, use_errno=True)
_LIBC.prctl.argtypes = [ctypes.c_int] + [ctypes.c_ulong] * 4
_LIBC.prctl.restype = ctypes.c_int

_PYTHON_ENTRY = r'''
import json, os, runpy, signal, sys
def stopped(signum, frame):
    raise KeyboardInterrupt("PixelForge worker termination")
signal.signal(signal.SIGTERM, stopped)
arguments = sys.argv[2:]
sys.orig_argv = json.loads(sys.argv[1]) + arguments
def entry_path(path):
    if not sys.flags.safe_path:
        if sys.path and sys.path[0] == '':
            sys.path[0] = path
        else:
            sys.path.insert(0, path)
try:
    if arguments[0] == '-m':
        module = arguments[1]
        sys.argv = [module] + arguments[2:]
        entry_path(os.getcwd())
        runpy.run_module(module, run_name='__main__', alter_sys=True)
    elif arguments[0] == '-c':
        source = arguments[1]
        sys.argv = ['-c'] + arguments[2:]
        exec(compile(source, '<string>', 'exec'), {'__name__': '__main__', '__package__': None})
    else:
        script = arguments[0]
        sys.argv = arguments
        entry_path(os.path.dirname(os.path.abspath(script)))
        runpy.run_path(script, run_name='__main__')
except KeyboardInterrupt:
    raise SystemExit(130)
'''


def _stat(pid):
    return Path(f'/proc/{pid}/stat').read_text().rpartition(') ')[2].split()


def _ready(descriptor):
    return bool(select.select([descriptor], [], [], 0)[0])


def _signal(descriptor, signum):
    try:
        signal.pidfd_send_signal(descriptor, signum)
    except ProcessLookupError:
        pass


def _prctl(option, value):
    if _LIBC.prctl(option, value, 0, 0, 0):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))


def _child_guard(parent):
    # This dedicated guard is single-threaded; the child runs no application
    # code before exec. A forced guard termination still interrupts its child.
    if _LIBC.prctl(1, signal.SIGTERM, 0, 0, 0) or os.getppid() != parent:
        os._exit(130)


def _reap():
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if not pid:
            return


def _children():
    pid = os.getpid()
    try:
        return [int(value) for value in
                Path(f'/proc/{pid}/task/{pid}/children').read_text().split()]
    except FileNotFoundError:
        # Android kernels can omit the optional proc task children file.
        # This scan is read-only; signaling later rechecks PPid via a pidfd.
        result = []
        for entry in Path('/proc').iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                fields = _stat(int(entry.name))
                if int(fields[1]) == pid:
                    result.append(int(entry.name))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        return result


def _python_command(command):
    options = []
    offset = 1
    while offset < len(command):
        value = command[offset]
        if value in ('-c', '-m', '-') or not value.startswith('-'):
            break
        if value == '--':
            offset += 1
            break
        options.append(value)
        offset += 1
        if value in ('-W', '-X', '--check-hash-based-pycs'):
            if offset == len(command):
                raise ValueError('Missing Python interpreter option value')
            options.append(command[offset])
            offset += 1
    entry = command[offset:]
    if not entry or (entry[0] in ('-m', '-c') and len(entry) < 2):
        raise ValueError('Python command requires a script, -m MODULE or -c CODE')
    if entry[0] == '-':
        raise ValueError('Interactive stdin Python is not a shipped tool entry point')
    original = [command[0]] + options
    return original + ['-c', _PYTHON_ENTRY, json.dumps(original)] + entry


def _stop_adopted(grace, status):
    """After the direct target was waited, drain its adopted descendants."""
    seen = set()
    term_until = time.monotonic() + min(5.0, grace)
    end = term_until + 5.0
    while True:
        _reap()
        children = _children()
        if not children:
            status['adopted_children_seen'] = len(seen)
            return True
        now = time.monotonic()
        if now >= end:
            status['adopted_children_seen'] = len(seen)
            return False
        for pid in children:
            descriptor = None
            try:
                descriptor = os.pidfd_open(pid)
                fields = _stat(pid)
                if int(fields[1]) != os.getpid():
                    continue  # The numeric PID no longer identifies our child.
                identity = (pid, fields[19])
                if _ready(descriptor):
                    continue
                if now >= term_until:
                    _signal(descriptor, signal.SIGKILL)
                    status['forced_descendant_cleanup'] = True
                elif identity not in seen:
                    _signal(descriptor, signal.SIGTERM)
                seen.add(identity)
            except (ProcessLookupError, FileNotFoundError):
                pass
            finally:
                if descriptor is not None:
                    os.close(descriptor)
        time.sleep(.02)


def _stop_target(process, descriptor, grace, status):
    if process.poll() is not None:
        return True
    _signal(descriptor, signal.SIGTERM)
    until = time.monotonic() + grace
    while process.poll() is None and time.monotonic() < until:
        select.select([descriptor], [], [], min(.05, max(0, until - time.monotonic())))
    if process.poll() is None:
        status['forced_child_cleanup'] = True
        _signal(descriptor, signal.SIGKILL)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--owner-pid', type=int, required=True)
    parser.add_argument('--owner-start-ticks', required=True)
    parser.add_argument('--grace-seconds', type=float, default=60)
    parser.add_argument('--status-file', type=Path)
    parser.add_argument('--python', action='store_true', dest='python_entry')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if (not command or args.owner_pid < 1 or not args.owner_start_ticks.isdecimal()
            or not 0 < args.grace_seconds < float('inf')):
        parser.error('Provide a command, positive owner PID and finite positive grace')
    if args.python_entry and len(command) < 2:
        parser.error('--python requires a script, -m MODULE or -c CODE')
    status = {'started': False, 'cleanup_ok': False, 'reason': None,
              'owner_exited': False, 'cancel_signal': None,
              'child_returncode': None, 'forced_child_cleanup': False,
              'forced_descendant_cleanup': False, 'adopted_children_seen': 0,
              'exit_code': 125}
    requested = [None]

    def stop(signum, _frame):
        if requested[0] is None:
            requested[0] = signum

    handlers = {sig: signal.signal(sig, stop)
                for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
    owner = child_fd = status_fd = None
    process = None
    try:
        if args.status_file:
            status_fd = os.open(args.status_file,
                                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        owner = os.pidfd_open(args.owner_pid)
        if _stat(args.owner_pid)[19] != args.owner_start_ticks:
            raise ValueError('Owner process identity changed; no target was started')
        _prctl(36, 1)  # PR_SET_CHILD_SUBREAPER; descendants stay owned here.
        if _ready(owner) or requested[0] is not None:
            status.update(owner_exited=_ready(owner), cancel_signal=requested[0],
                          reason='owner_exited' if _ready(owner) else 'cancelled',
                          exit_code=130)
        else:
            if args.python_entry:
                command = _python_command(command)
            parent = os.getpid()
            process = subprocess.Popen(command, start_new_session=True,
                                       preexec_fn=lambda: _child_guard(parent))
            status['started'] = True
            child_fd = os.pidfd_open(process.pid)
            while process.poll() is None:
                if requested[0] is not None or _ready(owner):
                    status.update(owner_exited=_ready(owner), cancel_signal=requested[0],
                                  reason='owner_exited' if _ready(owner) else 'cancelled')
                    break
                select.select([owner, child_fd], [], [], .1)
            if status['reason'] is not None:
                if not _stop_target(process, child_fd, args.grace_seconds, status):
                    raise RuntimeError('Owned target exit was not confirmed')
            status['child_returncode'] = process.wait(timeout=5)
            status['exit_code'] = 130 if status['reason'] else status['child_returncode']
    except Exception as error:
        status.update(reason='error', error=type(error).__name__ + ': ' + str(error), exit_code=125)
    finally:
        # Signal handlers only set a flag, so repeated cancellation cannot
        # interrupt cleanup, reaping or publication of the private receipt.
        try:
            if process is not None:
                if process.poll() is None:
                    if child_fd is None:
                        if owner is not None:
                            os.close(owner)
                            owner = None
                        child_fd = os.pidfd_open(process.pid)
                    if not _stop_target(process, child_fd, args.grace_seconds, status):
                        raise RuntimeError('Owned target survived forced termination')
                status['child_returncode'] = process.wait(timeout=5)
                if not _stop_adopted(args.grace_seconds, status):
                    raise RuntimeError('Owned descendants did not exit after cleanup')
                if not status['reason'] and status['adopted_children_seen']:
                    status.update(reason='unexpected_descendants', exit_code=125)
            status['cleanup_ok'] = True
        except Exception as error:
            status.update(cleanup_ok=False, cleanup_error=type(error).__name__ + ': ' + str(error),
                          exit_code=125)
        for descriptor in (owner, child_fd):
            if descriptor is not None:
                os.close(descriptor)
        if status_fd is not None:
            with os.fdopen(status_fd, 'w') as output:
                json.dump(status, output)
                output.write('\n')
                output.flush()
                os.fsync(output.fileno())
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    if status.get('error') or status.get('cleanup_error'):
        try:
            print('PixelForge worker: ' + str(status.get('cleanup_error', status.get('error'))),
                  file=sys.stderr)
        except OSError:
            pass
    code = status['exit_code']
    if code < 0:
        # Keep Popen's original signal return code after all children are gone.
        if -code not in (signal.SIGKILL, signal.SIGSTOP):
            signal.signal(-code, signal.SIG_DFL)
        os.kill(os.getpid(), -code)
        return 128 - code
    return code


if __name__ == '__main__':
    raise SystemExit(main())
