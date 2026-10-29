#!/usr/bin/env python3
"""Start Ostadix Voice here, automatically stopping the previous interface."""
import argparse
from contextlib import contextmanager
import errno
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import sqlite3
import subprocess
import sys
import time

from server import AppError, DEFAULT_STATE

APP = Path(__file__).resolve().parent
LAUNCH_LOCK_FD = 'OSTADIX_LAUNCH_LOCK_FD'

# Shared verbatim with the isolated root helper. Process-directed SIGTERM can
# land on a worker, leaving the Python main thread stuck in a resumed tty read.
# Direct it to the main thread so that read is interrupted before job-control
# can stop the background interface again. No application module is root-loaded.
_MAIN_THREAD_SIGNAL_CODE = r'''
def _signal_ui_main_thread(pid, number, pidfd=None, is_current=None):
    import errno, os, signal, sys
    if type(pid) is not int or pid <= 0:
        raise ValueError('A positive process ID is required')
    if is_current is not None and not is_current():
        return False
    if pidfd is not None and hasattr(signal, 'pidfd_send_signal'):
        try:
            # Linux 6.9+: retain the pidfd identity while targeting its main TID.
            signal.pidfd_send_signal(pidfd, number, None, 1)
            return True
        except OSError as error:
            if error.errno not in (errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP):
                raise
    if sys.platform.startswith(('linux', 'android')):
        try:
            import ctypes
            tgkill = ctypes.CDLL(None, use_errno=True).tgkill
        except (ImportError, AttributeError, OSError):
            tgkill = None
        if tgkill is not None:
            tgkill.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_int)
            tgkill.restype = ctypes.c_int
            # Loading libc may have yielded. Recheck the exact lock owner and
            # process start time immediately before the numeric-TID fallback.
            if is_current is not None and not is_current():
                return False
            if tgkill(pid, pid, number) == 0:
                return True
            error = ctypes.get_errno()
            if error != errno.ENOSYS:
                raise OSError(error, os.strerror(error))
    if is_current is not None and not is_current():
        return False
    if pidfd is not None and hasattr(signal, 'pidfd_send_signal'):
        signal.pidfd_send_signal(pidfd, number)
    else:
        os.kill(pid, number)
    return True
'''
exec(_MAIN_THREAD_SIGNAL_CODE)

# Android hidepid can hide a same-UID legacy interface from ordinary Termux.
# This isolated, stdlib-only helper uses the existing su grant solely to inspect
# the exact session lock and gracefully signal its verified owner. Never import
# application code or run the interface as root.
ROOT_SESSION_HELPER = _MAIN_THREAD_SIGNAL_CODE + r'''
import fcntl, json, os, signal, stat, sys, time
from pathlib import Path

def run():
    request = json.load(sys.stdin)
    uid = request['uid']
    if os.geteuid() != 0 or not isinstance(uid, int) or uid <= 0:
        return {'status': 'rejected'}
    app, state = Path(request['app']), Path(request['state'])
    executable = Path(request['python'])
    if (request['action'] not in ('probe', 'stop') or not app.is_absolute() or
            not state.is_absolute() or app.resolve() != app or state.resolve() != state or
            app.stat().st_uid != uid or executable != Path(sys.executable).resolve()):
        return {'status': 'rejected'}
    lock_path = state / 'server.lock'
    descriptor = os.open(lock_path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        identity = (info.st_dev, info.st_ino)
        if (identity != (request['dev'], request['ino']) or info.st_uid != uid or
                not stat.S_ISREG(info.st_mode)):
            return {'status': 'changed'}
        def current_lock():
            current = lock_path.stat(follow_symlinks=False)
            return (current.st_dev, current.st_ino) == identity and stat.S_ISREG(current.st_mode)
        def available():
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                return False
        if available():
            return {'status': 'available'}
        scripts = {app / 'terminal.py', app / 'server.py'}
        deadline = time.monotonic() + 4
        def stamp(process):
            return (process / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        def matches(process, started):
            try:
                if time.monotonic() >= deadline or not current_lock() or stamp(process) != started:
                    return False
                if process.stat().st_uid != uid or (process / 'exe').resolve() != executable:
                    return False
                status = (process / 'status').read_text().splitlines()
                credentials = next(line for line in status if line.startswith('Uid:')).split()[1:]
                if credentials != [str(uid)] * 4:
                    return False
                argv = (process / 'cmdline').read_bytes().split(b'\0')
                if len(argv) < 2:
                    return False
                script = Path(os.fsdecode(argv[1]))
                if not script.is_absolute():
                    script = (process / 'cwd').resolve() / script
                if script.resolve() not in scripts:
                    return False
                for entry in (process / 'fd').iterdir():
                    try:
                        opened = entry.stat()
                        if (opened.st_dev, opened.st_ino) != identity:
                            continue
                        for line in (process / 'fdinfo' / entry.name).read_text().splitlines():
                            fields = line.split()
                            if (fields[:1] == ['lock:'] and
                                    fields[2:6] == ['FLOCK', 'ADVISORY', 'WRITE', process.name]):
                                return stamp(process) == started
                    except OSError:
                        continue
            except (OSError, ValueError, StopIteration, IndexError):
                pass
            return False
        for process in Path('/proc').iterdir():
            if time.monotonic() >= deadline:
                break
            if not process.name.isdigit():
                continue
            try:
                started = stamp(process)
                if not matches(process, started):
                    continue
                if request['action'] == 'probe':
                    return {'status': 'owner', 'pid': int(process.name), 'uid': uid, 'started': started}
                pidfd = None
                try:
                    if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
                        pidfd = os.pidfd_open(int(process.name))
                    if not matches(process, started):
                        return {'status': 'changed'}
                    def send(number):
                        if number == signal.SIGTERM:
                            return _signal_ui_main_thread(int(process.name), number, pidfd,
                                                          lambda: matches(process, started))
                        elif pidfd is not None:
                            signal.pidfd_send_signal(pidfd, number)
                        else:
                            os.kill(int(process.name), number)
                        return True
                    if not send(signal.SIGTERM):
                        return {'status': 'changed'}
                    if matches(process, started):
                        send(signal.SIGCONT)
                    return {'status': 'signalled'}
                finally:
                    if pidfd is not None:
                        os.close(pidfd)
            except ProcessLookupError:
                return {'status': 'available' if available() else 'changed'}
            except OSError:
                continue
        return {'status': 'available' if available() else 'unidentified'}
    finally:
        os.close(descriptor)

try:
    result = run()
except (OSError, ValueError, KeyError, TypeError):
    result = {'status': 'rejected'}
print(json.dumps(result))
'''


def root_session_owner(state, identity, probe=False):
    """Bounded Android-only fallback for a hidden same-UID session owner."""
    if not Path('/system/bin/su').is_file():
        return None
    request = {'uid': os.geteuid(), 'app': str(APP), 'state': str(state),
               'python': str(Path(sys.executable).resolve()),
               'dev': identity[0], 'ino': identity[1], 'action': 'probe' if probe else 'stop'}
    # -I -S excludes Python environment/site imports in the privileged helper.
    # Paths and code are separately quoted; private request data goes via stdin.
    command = ['/system/bin/su', '-c', shlex.join([sys.executable, '-I', '-S', '-c', ROOT_SESSION_HELPER])]
    prefix = str(Path(sys.executable).resolve().parent.parent)
    environment = {'PATH': '/system/bin:/system/xbin', 'HOME': str(APP.parent),
                   'PREFIX': prefix, 'LD_LIBRARY_PATH': prefix + '/lib',
                   'ANDROID_ROOT': '/system', 'ANDROID_DATA': '/data', 'LANG': 'C.UTF-8'}
    try:
        result = subprocess.run(command, input=json.dumps(request), text=True, capture_output=True,
                                timeout=8, env=environment)
        if result.returncode or len(result.stdout) > 1000:
            raise ValueError('invalid helper result')
        response = json.loads(result.stdout)
        if not isinstance(response, dict) or response.get('status') not in (
                'owner', 'signalled', 'available', 'changed', 'unidentified', 'rejected'):
            raise ValueError('invalid helper status')
        return response
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        raise AppError('Android could not inspect the hidden session through its existing root grant. '
                       'Use /quit in its terminal, then run c again.') from error


@contextmanager
def launch_guard(state, timeout=60):
    """Serialize replacement until the new UI owns its persistent session lock."""
    state.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(state / 'launch.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, 'a') as lock:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise AppError('Another c invocation is still starting. Wait, then run c again.')
                time.sleep(.05)
        yield lock


def release_launch_lock(state):
    """Release only our inherited launch descriptor, after server.lock is held."""
    value = os.environ.pop(LAUNCH_LOCK_FD, '')
    try:
        descriptor = int(value)
        if descriptor < 3:
            return False
        actual = os.fstat(descriptor)
        expected = (Path(state) / 'launch.lock').stat(follow_symlinks=False)
        if ((actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino) or
                actual.st_uid != os.geteuid()):
            return False
        # An environment variable alone must never close an unrelated descriptor.
        # Confirm this exact descriptor carries our inherited kernel flock.
        records = Path('/proc/self/fdinfo', str(descriptor)).read_text().splitlines()
        if not any(line.split()[:1] == ['lock:'] and
                   line.split()[2:6] == ['FLOCK', 'ADVISORY', 'WRITE', str(os.getpid())]
                   for line in records):
            return False
        os.close(descriptor)
        return True
    except (OSError, ValueError, OverflowError):
        return False


def running(state):
    lock_path = state / 'server.lock'
    if not lock_path.exists():
        return False
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        return False


def _session_processes():
    """Find our UI processes, including sessions started before --stop existed."""
    scripts = {APP / 'terminal.py', APP / 'server.py'}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            if process.stat().st_uid != os.geteuid():
                continue
            argv = (process / 'cmdline').read_bytes().split(b'\0')
            if len(argv) < 2:
                continue
            script = Path(os.fsdecode(argv[1]))
            if not script.is_absolute():
                script = (process / 'cwd').resolve() / script
            if script.resolve() in scripts:
                yield process
        except (OSError, ValueError):
            continue  # A process may exit while /proc is being inspected.


def _owns_lock(process, identity):
    """Require kernel lock ownership, not merely an open file or a saved PID."""
    try:
        for fd in (process / 'fd').iterdir():
            try:
                info = fd.stat()
                if (info.st_dev, info.st_ino) != identity:
                    continue
                for line in (process / 'fdinfo' / fd.name).read_text().splitlines():
                    fields = line.split()
                    if (fields[:1] == ['lock:'] and
                            fields[2:6] == ['FLOCK', 'ADVISORY', 'WRITE', process.name]):
                        return True
            except OSError:
                continue
    except OSError:
        pass
    return False


def stop(state, timeout=50):
    """Gracefully close only the UI holding this store, preserving its history."""
    try:
        lock = (state / 'server.lock').open('r')
    except FileNotFoundError:
        return False
    with lock:
        def available():
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                return False

        if available():
            return False
        info = os.fstat(lock.fileno())
        identity = (info.st_dev, info.st_ino)
        owner = next((process for process in _session_processes()
                      if _owns_lock(process, identity)), None)
        if owner is None:
            if available():
                return False
            response = root_session_owner(state, identity)
            if not response or response.get('status') != 'signalled':
                if available():
                    return False
                raise AppError('The session lock is held, but its Ostadix Voice process could not be identified. '
                               'Use /quit in its terminal (Ctrl-C in browser mode).')
        # Recheck the kernel-owned lock immediately before sending the signal.
        # Never unlink server.lock: doing so would admit a second live interface.
        elif _owns_lock(owner, identity):
            pidfd = None
            try:
                started = (owner / 'stat').read_text().rsplit(')', 1)[1].split()[19]
                def current_owner():
                    try:
                        return ((owner / 'stat').read_text().rsplit(')', 1)[1].split()[19] == started and
                                _owns_lock(owner, identity))
                    except (OSError, ValueError, IndexError):
                        return False
                if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
                    try:
                        pidfd = os.pidfd_open(int(owner.name))
                    except OSError as error:
                        if error.errno not in (errno.ENOSYS, errno.EINVAL, errno.EPERM, errno.EACCES):
                            raise
                _signal_ui_main_thread(int(owner.name), signal.SIGTERM, pidfd, current_owner)
                # A suspended terminal job cannot run its SIGTERM handler.
                # Resume it only after requesting shutdown, so cleanup can run.
                if current_owner():
                    if pidfd is not None:
                        signal.pidfd_send_signal(pidfd, signal.SIGCONT)
                    else:
                        os.kill(int(owner.name), signal.SIGCONT)
            except (ProcessLookupError, FileNotFoundError):
                pass
            finally:
                if pidfd is not None:
                    os.close(pidfd)
        deadline = time.monotonic() + timeout
        while True:
            if available():
                return True
            if time.monotonic() >= deadline:
                raise AppError('The session is still shutting down. Wait, then retry c --stop. '
                               'Its session lock was kept in place.')
            time.sleep(.1)


def metadata(state):
    try:
        data = json.loads((state / 'browser.json').read_text())
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def saved_workspace(state):
    database = state / 'sessions.sqlite3'
    if database.exists():
        connection = sqlite3.connect(database.as_uri() + '?mode=ro', uri=True)
        try:
            row = connection.execute("SELECT value FROM settings WHERE key='launch_workspace'").fetchone()
            if row:
                return row[0]
        finally:
            connection.close()
    return metadata(state).get('workspace')


def prepare_ostadix(offline=False, required_update=False):
    """Verify the installed pair promptly; fetch/build only for setup or --update."""
    from ostadix_pair import PairError, load_pair, update_pair
    report = lambda message: print('[OSTADIX] ' + message, flush=True)
    if offline and required_update:
        raise AppError('--update-ostadix cannot be combined with --offline')
    if required_update:
        try:
            return update_pair(progress=report)
        except PairError as error:
            raise AppError(str(error)) from error
    try:
        pair = load_pair()
    except PairError as error:
        if not offline:
            report('No verified installed pair is available; installing OSTADIX-BM...')
            try:
                return update_pair(progress=report)
            except PairError as install_error:
                error = install_error
        raise AppError('OSTADIX is unavailable: ' + str(error) +
                       '\nRun c --update-ostadix to install or repair it, then run c again.') from error
    report('Using verified installed OSTADIX-BM revision ' + pair['revision'][:12] + '.')
    return pair


def main(argv=None):
    parser = argparse.ArgumentParser(prog='c', description=__doc__)
    parser.add_argument('workspace', nargs='?', type=Path, help='Project directory; defaults to the previous workspace, or the example workspace on first launch')
    parser.add_argument('--browser', '-b', action='store_true', help='Use the optional browser interface')
    parser.add_argument('--no-audio', action='store_true', help='Disable automatic speech in the terminal')
    parser.add_argument('--voice', action='store_true', help='Start continuous voice conversation (the default)')
    parser.add_argument('--text', action='store_true', help='Start without opening the microphone; /voice on enables it later')
    parser.add_argument('--no-local', action='store_true', help='Do not start the resident local model on launch')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--stop', action='store_true', help='Stop the running terminal/browser session, preserving saved history')
    action.add_argument('--local-status', action='store_true', help='Show local model/worker status without starting Codex')
    action.add_argument('--start-local', action='store_true', help='Start the resident local model, then exit')
    action.add_argument('--stop-local', action='store_true', help='Stop the local model and cancel its unfinished workers')
    action.add_argument('--optimizer-status', action='store_true', help='Inspect the local optimizer without starting it or Codex')
    action.add_argument('--start-optimizer', action='store_true', help='Enable the persistent local optimizer, then exit')
    action.add_argument('--stop-optimizer', action='store_true', help='Disable the optimizer and restore its managed settings')
    action.add_argument('--optimizer-history', action='store_true', help='Show the optimizer\'s recent observations and actions')
    action.add_argument('--companion-status', action='store_true', help='Inspect local companion check-ins without starting Codex')
    action.add_argument('--start-companion', action='store_true', help='Enable local companion check-ins while you use the phone')
    action.add_argument('--stop-companion', action='store_true', help='Disable local companion check-ins')
    action.add_argument('--companion-read', action='store_true', help='Read the latest local companion message')
    parser.add_argument('--optimizer-objective', choices=('balanced', 'performance', 'battery'),
                        help='Objective for --start-optimizer (default: balanced)')
    parser.add_argument('--state-dir', type=Path, default=DEFAULT_STATE, help='Optional separate local session store')
    action.add_argument('--update-ostadix', action='store_true', help='Update and verify the matched OSTADIX repository/MCP, then exit')
    parser.add_argument('--offline', action='store_true', help='Use the verified installed OSTADIX-BM pair without attempting first-time setup or repair')
    args = parser.parse_args(argv)
    if args.optimizer_objective is not None and not args.start_optimizer:
        parser.error('--optimizer-objective requires --start-optimizer')
    if args.voice and (args.browser or args.no_audio or args.stop or args.text):
        parser.error('--voice starts the terminal microphone; do not combine it with --browser, --no-audio, --text, or --stop')
    if args.update_ostadix and args.offline:
        parser.error('--update-ostadix cannot be combined with --offline')
    if args.stop and (args.workspace is not None or args.browser):
        parser.error('--stop selects the running session by --state-dir; omit the workspace and --browser')
    if os.geteuid() == 0:
        raise AppError('Run c in normal Termux, outside su/root. Phone commands can use your existing su grant when needed.')
    state = args.state_dir.expanduser().resolve()
    if args.stop:
        if state.exists():
            with launch_guard(state):
                stopped = stop(state)
        else:
            stopped = stop(state)
        if stopped:
            print('Stopped Ostadix Voice. Run c to start it again.')
        else:
            print('No Ostadix Voice session is running.')
        return
    if args.local_status or args.start_local or args.stop_local:
        from local_runtime import LocalClient
        client = LocalClient()
        result = client.start() if args.start_local else client.stop() if args.stop_local else client.status()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.optimizer_status or args.start_optimizer or args.stop_optimizer or args.optimizer_history:
        try:
            from optimizer import OptimizerControl
            control = OptimizerControl()
            if args.start_optimizer:
                result = control.enable(objective=args.optimizer_objective or 'balanced')
            elif args.stop_optimizer:
                result = control.disable()
            elif args.optimizer_history:
                result = control.history(limit=10)
            else:
                result = control.status()
        except Exception as error:
            raise AppError('Optimizer: ' + str(error)) from error
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.companion_status or args.start_companion or args.stop_companion or args.companion_read:
        try:
            from companion import CompanionControl
            control = CompanionControl()
            if args.start_companion:
                result = control.enable()
            elif args.stop_companion:
                result = control.disable()
            elif args.companion_read:
                result = control.read()
            else:
                result = control.status()
        except Exception as error:
            raise AppError('Companion: ' + str(error)) from error
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.update_ostadix:
        prepare_ostadix(required_update=True)
        return
    if not args.browser and not sys.stdin.isatty():
        raise AppError('Run c in an interactive terminal. Piped input cannot approve local actions.')
    requested = args.workspace.expanduser().resolve(strict=True) if args.workspace else None
    if requested is not None and not requested.is_dir():
        raise AppError('Select a project directory')
    with launch_guard(state) as lock:
        previous = saved_workspace(state)
        workspace = requested or (Path(previous) if previous else APP / 'example-workspace')
        if not workspace.is_absolute() or not workspace.is_dir():
            raise AppError('The previous workspace is unavailable. Run c /absolute/path/to/project to select one.')
        try:
            prepare_ostadix(offline=args.offline)
        except RuntimeError as error:
            raise AppError(str(error)) from error
        if running(state):
            print('Restarting Ostadix Voice here. Stopping the previous session...', flush=True)
            stop(state)
        if not args.no_local:
            try:
                from local_service import LocalService
                service = LocalService()
                # A normal UI restart respects an explicit stop. Only first setup
                # and the user's manual start control enable persistent supervision.
                status = service.resume() if (service.state / 'service.json').exists() else service.enable()
                if status.get('error'):
                    print('[Local] ' + str(status['error']), flush=True)
            except Exception as error:
                print('[Local] ' + str(error) + ' — Codex and saved sessions remain available.', flush=True)
        if args.browser:
            command = [sys.executable, str(APP / 'server.py'), '--workspace', str(workspace), '--state-dir', str(state), '--open']
        else:
            command = [sys.executable, str(APP / 'terminal.py'), '--workspace', str(workspace), '--state-dir', str(state)]
            if args.no_audio:
                command.append('--no-audio')
            if not args.no_audio and not args.text:
                command.append('--voice')
        # Keep the launch guard across exec; the UI releases it once it has
        # acquired server.lock. Closing it on exec would leave a startup race.
        previous_fd = os.environ.get(LAUNCH_LOCK_FD)
        os.set_inheritable(lock.fileno(), True)
        os.environ[LAUNCH_LOCK_FD] = str(lock.fileno())
        try:
            os.execv(sys.executable, command)
        finally:
            if previous_fd is None:
                os.environ.pop(LAUNCH_LOCK_FD, None)
            else:
                os.environ[LAUNCH_LOCK_FD] = previous_fd


if __name__ == '__main__':
    try:
        main()
    except (AppError, OSError, ValueError, sqlite3.Error) as error:
        raise SystemExit(str(error))
