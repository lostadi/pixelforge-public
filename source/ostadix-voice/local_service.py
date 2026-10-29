"""Termux supervision for the local model and explicitly enabled optimizer.

Never starts Codex. The optimizer may submit its separately authorized local
monitoring requests while its persistent enablement is on.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid

# The fixed root guardian drops to the app account before executing this file
# with -I. Add this known application directory only after that identity drop.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_runtime import DEFAULT_STATE, ROOT, LocalClient, LocalError, _environment, _private_file, _state

BACKOFF = (1, 2, 4, 8, 16, 30)
BOOT_NAME = '70-ostadix-local'
NPU_STATE = Path.home() / '.local/state/ostadix-voice-npu'
NPU_MANIFEST = ROOT / 'runtime/npu/model-profile.json'


def _command(script, action, state, manifest_path=None):
    command = [sys.executable, str(script), action, '--state-dir', str(state)]
    if manifest_path is not None:
        command.extend(['--manifest', str(manifest_path)])
    return command


def resume_all():
    """Recover independently enabled fixed services without changing saved choices."""
    result = {}
    for name, state, manifest in (('local', DEFAULT_STATE, None), ('npu', NPU_STATE, NPU_MANIFEST)):
        try:
            service = LocalService(state, manifest_path=manifest) if manifest is not None else LocalService(state)
            result[name] = service.resume()
        except (OSError, ValueError, LocalError) as error:
            # One damaged profile or service must not prevent the other recovery.
            result[name] = {'running': False, 'ready': False, 'error': str(error)[:300]}
    return result


def _read(path):
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'r') as stream:
            text = stream.read(8193)
        if len(text) > 8192:
            raise LocalError('Local service state is oversized')
        value = json.loads(text)
        if not isinstance(value, dict):
            raise LocalError('Local service state must be an object')
        return value
    except FileNotFoundError:
        return {}


def _write(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex)
    try:
        with os.fdopen(_private_file(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'w') as stream:
            json.dump(value, stream, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _control(state):
    with os.fdopen(_private_file(state / 'service-control.lock', os.O_WRONLY | os.O_CREAT), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _enabled(state):
    config = _read(state / 'service.json')
    if not config:
        return False
    if config.keys() - {'version', 'enabled', 'updated'} or type(config.get('version')) is not int or config.get('version') != 1 or type(config.get('enabled')) is not bool:
        raise LocalError('Invalid local service enablement; explicitly enable or disable it again')
    return config['enabled']


def _process_stamp(pid):
    if type(pid) is not int or pid < 1:
        return None
    try:
        process = Path('/proc') / str(pid)
        if process.stat().st_uid != os.getuid():
            return None
        fields = (process / 'stat').read_text().rsplit(') ', 1)[1].split()
        return fields[19] if fields[0] != 'Z' else None
    except (OSError, IndexError):
        return None


def service_status(state):
    """Fast metadata only; never creates directories or launches processes."""
    state = Path(state)
    error = None
    try:
        enabled = _enabled(state)
        metadata = _read(state / 'supervisor.json')
    except (OSError, ValueError, LocalError) as caught:
        enabled, metadata, error = False, {}, str(caught)[:300]
    pid = metadata.get('pid')
    stamp = _process_stamp(pid)
    supervised = stamp is not None and stamp == metadata.get('process_stamp')
    retry_at = metadata.get('retry_at', 0)
    retry_in = max(0, retry_at - time.time()) if type(retry_at) in (int, float) else 0
    return {'service_enabled': enabled, 'supervised': supervised, 'supervisor_pid': pid if supervised else None,
            'service_phase': metadata.get('phase', 'starting') if supervised else 'stopped' if enabled else 'disabled',
            'service_error': error or metadata.get('error'), 'restart_count': metadata.get('restart_count', 0),
            'restart_in_seconds': round(retry_in, 1) if enabled else 0}


def _raw_status(client):
    try:
        return client._call('status', timeout=.4)
    except (OSError, ValueError, LocalError):
        return {'running': False, 'ready': False, 'error': None}


class LocalService:
    def __init__(self, state_dir=None, manifest_path=None):
        self.state = Path(state_dir) if state_dir is not None else DEFAULT_STATE
        self.manifest_path = Path(manifest_path).expanduser().resolve() if manifest_path is not None else None
        self.client = (LocalClient(self.state, manifest_path=self.manifest_path)
                       if self.manifest_path is not None else LocalClient(self.state))

    def status(self):
        return self.client.status()

    def _resume_optimizer(self):
        if (self.state / 'optimizer/config.json').exists():
            from optimizer import OptimizerControl
            return OptimizerControl(self.state).resume()

    def _resume(self, enable=False):
        if os.getuid() == 0:
            raise LocalError('Manage local service from normal Termux, outside root')
        state = _state(self.state)
        with _control(state):
            if enable:
                _write(state / 'service.json', {'version': 1, 'enabled': True, 'updated': time.time()})
            if not _enabled(state):
                return self.status()
            metadata = service_status(state)
            if not metadata['supervised'] or metadata['service_phase'] in ('stopping', 'stopped'):
                with os.fdopen(_private_file(state / 'supervisor.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND), 'ab') as log:
                    subprocess.Popen(_command(Path(__file__).resolve(), '--supervise', state, self.manifest_path),
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=_environment(), start_new_session=True)
        # Only wait for control availability; model hashing/loading stays in background.
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            result = self.status()
            if not result['service_enabled'] or (result['running'] and result['supervised'] and result['service_phase'] != 'stopping'):
                if result['service_enabled']:
                    self._resume_optimizer()
                return result
            time.sleep(.025)
        result['starting'] = result['service_enabled']
        return result

    def enable(self):
        result = self._resume(enable=True)
        if (self.state.expanduser().resolve() == NPU_STATE.resolve() and
                self.manifest_path == NPU_MANIFEST.resolve()):
            # Re-enabling NPU after all services were disabled must also resume
            # the existing root watchdog. It only recovers saved enabled choices.
            try:
                from optimizer import GUARDIAN, root_call
                result['guardian'] = root_call([sys.executable, '-I', str(GUARDIAN), '--ensure'])
            except (OSError, ValueError, LocalError, subprocess.SubprocessError) as error:
                result['guardian_error'] = str(error)[:300]
        return result

    def resume(self):
        # Boot and shell recovery never reverse an intentional disable.
        if not _enabled(self.state):
            return self.status()
        return self._resume()

    def disable(self):
        if os.getuid() == 0:
            raise LocalError('Manage local service from normal Termux, outside root')
        state = _state(self.state)
        with _control(state):
            _write(state / 'service.json', {'version': 1, 'enabled': False, 'updated': time.time()})
        optimizer_error = None
        try:
            if (state / 'optimizer/config.json').exists():
                from optimizer import OptimizerControl
                optimizer_error = OptimizerControl(state).stop().get('error')
        except Exception as error:
            # A damaged priority journal must not keep the model running after
            # an explicit disable. Preserve the cleanup failure for the caller.
            optimizer_error = str(error)[:300]
        finally:
            result = self.client._stop_once()
        result.update(service_enabled=False, supervised=service_status(state)['supervised'])
        if optimizer_error:
            result['optimizer_error'] = optimizer_error
            result['error'] = 'Local model disabled; optimizer cleanup failed: ' + optimizer_error
        return result

    def install_boot(self, boot_directory=None):
        if os.getuid() == 0:
            raise LocalError('Install the boot hook from normal Termux, outside root')
        directory = Path(boot_directory) if boot_directory is not None else Path.home() / '.termux/boot'
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if directory.is_symlink() or directory.stat().st_uid != os.getuid():
            raise LocalError('Boot directory must belong to this account and not be a symlink')
        path = directory / BOOT_NAME
        if path.exists() and '# Ostadix local model only;' not in path.read_text():
            raise LocalError('An unrelated script already occupies the Ostadix boot path')
        command = [sys.executable, str(Path(__file__).resolve()), '--resume-all']
        body = '#!/data/data/com.termux/files/usr/bin/sh\n# Ostadix local model only; disabled services remain disabled.\nexec ' + shlex.join(command) + '\n'
        with os.fdopen(_private_file(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC), 'w') as stream:
            stream.write(body)
        path.chmod(0o700)
        return {'boot_script': str(path), 'installed': True,
                'note': 'Termux:Boot must be opened once. Android can still suspend or terminate Termux.'}


class Supervisor:
    def __init__(self, state, spawn=subprocess.Popen, poll_interval=1.0, backoff=BACKOFF, manifest_path=None):
        self.state = _state(state)
        self.manifest_path = Path(manifest_path).expanduser().resolve() if manifest_path is not None else None
        self.client = (LocalClient(self.state, manifest_path=self.manifest_path)
                       if self.manifest_path is not None else LocalClient(self.state))
        self.spawn = spawn
        self.poll_interval = poll_interval
        self.backoff = backoff
        self.stop_event = threading.Event()
        self.child = None
        self.spawned_at = 0
        self.had_runtime = False
        self.healthy_since = None
        self.failures = 0
        self.next_start = 0
        self.next_optimizer_check = 0
        self.metadata = {'pid': os.getpid(), 'process_stamp': _process_stamp(os.getpid()),
                         'phase': 'starting', 'restart_count': 0, 'error': None, 'retry_at': 0}

    def record(self, **changes):
        if all(self.metadata.get(key) == value for key, value in changes.items()):
            return
        self.metadata.update(changes)
        self.metadata['updated'] = time.time()
        _write(self.state / 'supervisor.json', self.metadata)

    def failed(self, reason):
        self.failures += 1
        delay = self.backoff[min(self.failures - 1, len(self.backoff) - 1)]
        self.next_start = time.monotonic() + delay
        self.healthy_since = None
        self.record(phase='backoff', error=str(reason)[:300], retry_at=time.time() + delay)

    def _stop_child(self):
        try:
            self.client._stop_once()
        except (OSError, LocalError):
            pass
        if self.child is not None:
            try:
                self.child.wait(timeout=4)
            except subprocess.TimeoutExpired:
                self.child.terminate()
                try:
                    self.child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait(timeout=2)
            self.child = None

    def run(self):
        if os.getuid() == 0:
            raise LocalError('The local service supervisor must run as the normal Termux user')
        os.umask(0o077)
        lock = os.fdopen(_private_file(self.state / 'supervisor.lock', os.O_WRONLY | os.O_CREAT), 'w')
        deadline = time.monotonic() + 15
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                existing = service_status(self.state)
                if existing['supervised'] and existing['service_phase'] not in ('stopping', 'stopped'):
                    lock.close()
                    return
                if time.monotonic() >= deadline:
                    lock.close()
                    raise LocalError('Previous local supervisor did not finish stopping')
                time.sleep(.05)
        self.record(phase='monitoring')
        try:
            while not self.stop_event.is_set():
                with _control(self.state):
                    if not _enabled(self.state):
                        self.record(phase='stopping')
                        break
                status = _raw_status(self.client)
                if time.monotonic() >= self.next_optimizer_check:
                    self.next_optimizer_check = time.monotonic() + 10
                    if (self.state / 'optimizer/config.json').exists():
                        try:
                            from optimizer import OptimizerControl
                            OptimizerControl(self.state).resume()
                        except (OSError, ValueError, LocalError) as error:
                            self.record(optimizer_error=str(error)[:300])
                exited = self.child is not None and self.child.poll() is not None
                if exited:
                    self.child.wait()
                    self.child = None
                if status.get('running'):
                    self.had_runtime = True
                    if status.get('error'):
                        try:
                            self.client._stop_once()
                        except (OSError, LocalError):
                            pass
                        self.had_runtime = False
                        self.failed(status['error'])
                    else:
                        phase = 'ready' if status.get('ready') else 'loading'
                        self.record(phase=phase, error=None, retry_at=0)
                        if status.get('ready'):
                            self.healthy_since = self.healthy_since or time.monotonic()
                            if time.monotonic() - self.healthy_since >= 60:
                                self.failures = 0
                        else:
                            self.healthy_since = None
                else:
                    if self.had_runtime or exited:
                        self.had_runtime = False
                        # An engine error already entered backoff before its
                        # daemon exited. Keep that cause and its retry deadline.
                        if self.metadata.get('phase') != 'backoff' or not self.metadata.get('error'):
                            self.failed('Local model daemon stopped; unfinished jobs will not be replayed')
                    if self.child is not None and time.monotonic() - self.spawned_at > 20:
                        self._stop_child()
                        self.failed('Local daemon control did not become available within 20 seconds')
                    if self.child is None and time.monotonic() >= self.next_start:
                        with _control(self.state):
                            if not _enabled(self.state):
                                self.record(phase='stopping')
                                break
                            try:
                                with os.fdopen(_private_file(self.state / 'daemon.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND), 'ab') as log:
                                    self.child = self.spawn(_command(ROOT / 'local_runtime.py', '--daemon', self.state, self.manifest_path),
                                                            stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                                            env=_environment(), start_new_session=True)
                                self.spawned_at = time.monotonic()
                                self.record(phase='starting', restart_count=self.metadata['restart_count'] + int(self.failures > 0), retry_at=0)
                            except OSError as error:
                                self.failed('Could not launch local daemon: ' + str(error))
                self.stop_event.wait(self.poll_interval)
        finally:
            with _control(self.state):
                self.record(phase='stopping')
            try:
                if (self.state / 'optimizer/config.json').exists():
                    from optimizer import OptimizerControl
                    outcome = OptimizerControl(self.state).stop()
                    if outcome.get('error'):
                        self.record(optimizer_error=outcome['error'][:300])
            except Exception as error:
                self.record(optimizer_error=str(error)[:300])
            finally:
                try:
                    self._stop_child()
                    self.record(phase='stopped', retry_at=0)
                finally:
                    lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    for option in ('enable', 'disable', 'status', 'resume', 'resume-all', 'supervise', 'install-boot'):
        action.add_argument('--' + option, action='store_true')
    parser.add_argument('--state-dir', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--manifest', type=Path, help='Explicit independent local model profile')
    args = parser.parse_args(argv)
    if args.resume_all:
        if args.manifest is not None or args.state_dir != DEFAULT_STATE:
            parser.error('--resume-all uses the fixed local and NPU services; omit --state-dir and --manifest')
        print(json.dumps(resume_all(), indent=2))
        return
    service = (LocalService(args.state_dir, manifest_path=args.manifest)
               if args.manifest is not None else LocalService(args.state_dir))
    if args.supervise:
        supervisor = (Supervisor(args.state_dir, manifest_path=args.manifest)
                      if args.manifest is not None else Supervisor(args.state_dir))
        signal.signal(signal.SIGTERM, lambda *_: supervisor.stop_event.set())
        signal.signal(signal.SIGINT, lambda *_: supervisor.stop_event.set())
        supervisor.run()
        return
    operation = 'install_boot' if args.install_boot else next(name for name in ('enable', 'disable', 'status', 'resume') if getattr(args, name))
    print(json.dumps(getattr(service, operation)(), indent=2))


if __name__ == '__main__':
    main()
