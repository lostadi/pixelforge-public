"""An owned, read-only hardware Volume Down → Volume Up microphone hotkey.

Only the running terminal owns this listener. A root helper reads gpio_keys
without grabbing the device, forwards completed gestures only, and releases its
input descriptor when the terminal closes its control pipe.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import queue
import selectors
import shlex
import shutil
import subprocess
import sys
import threading
import time


class VolumeTapSequence:
    """Recognize released Down then released Up; reject holds and chords."""
    DOWN, UP = 114, 115

    def __init__(self, gap=.7, tap=.45):
        self.gap, self.tap = gap, tap
        self.reset()

    def reset(self):
        self.held = set()
        self.stage = 'idle'
        self.pressed_at = self.released_at = self.last_at = None

    def feed(self, code, value, at):
        if self.last_at is not None and at < self.last_at:
            self.reset()
        self.last_at = at
        if value == 2:
            self.held.add(code)
            self.stage = 'idle'
            return False
        if value == 1:
            duplicate = code in self.held
            self.held.add(code)
            if duplicate or len(self.held) != 1 or code not in (self.DOWN, self.UP):
                self.stage = 'idle'
            elif code == self.DOWN:
                self.stage, self.pressed_at = 'down', at
            elif (self.stage == 'waiting' and self.released_at is not None
                  and 0 <= at - self.released_at <= self.gap):
                self.stage, self.pressed_at = 'up', at
            else:
                self.stage = 'idle'
            return False
        if value != 0:
            self.stage = 'idle'
            return False
        was_held = code in self.held
        self.held.discard(code)
        if (not was_held or self.held or self.pressed_at is None
                or not 0 <= at - self.pressed_at <= self.tap):
            self.stage = 'idle'
            return False
        if self.stage == 'down' and code == self.DOWN:
            self.stage, self.released_at = 'waiting', at
            return False
        matched = self.stage == 'up' and code == self.UP
        self.stage = 'idle'
        return matched


_HELPER_BODY = r'''
def emit(kind, message=None):
    payload = {'event': kind}
    if message is not None:
        payload['message'] = message
    print(json.dumps(payload, separators=(',', ':')), flush=True)


def run(input_root='/dev/input', sysfs_root='/sys/class/input', fixture_fd=None):
    device = None
    selector = selectors.DefaultSelector()
    try:
        if fixture_fd is not None:
            # Only offline tests call run with an inherited fixture pipe.
            device = fixture_fd
        else:
            for entry in sorted(Path(sysfs_root).glob('event[0-9]*')):
                try:
                    name = (entry / 'device/name').read_text().strip().replace('-', '_')
                except OSError:
                    continue
                if name != 'gpio_keys':
                    continue
                path = Path(input_root) / entry.name
                if not stat.S_ISCHR(path.lstat().st_mode):
                    continue
                device = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW)
                break
            if device is None:
                raise RuntimeError('the gpio_keys volume-button device was not found')
        selector.register(0, selectors.EVENT_READ, 'control')
        selector.register(device, selectors.EVENT_READ, 'keys')
        packet = struct.Struct('@llHHi')
        pending = bytearray()
        sequence = VolumeTapSequence()
        dropped = False
        emit('ready')
        while True:
            ready = selector.select()
            # Shutdown wins over a simultaneous final key release.
            for key, _ in ready:
                if key.data == 'control':
                    if not os.read(0, 1024):
                        return
            for key, _ in ready:
                if key.data != 'keys':
                    continue
                chunk = os.read(device, packet.size * 32)
                if not chunk:
                    raise RuntimeError('the volume-button device disconnected')
                pending.extend(chunk)
                while len(pending) >= packet.size:
                    seconds, micros, kind, code, value = packet.unpack(pending[:packet.size])
                    del pending[:packet.size]
                    if kind == 0 and code == 3:  # SYN_DROPPED invalidates partial gestures.
                        sequence.reset()
                        dropped = True
                    elif kind == 0 and code == 0:
                        dropped = False
                    elif kind == 1 and not dropped:
                        if sequence.feed(code, value, seconds + micros / 1000000):
                            emit('toggle')
    except (OSError, RuntimeError) as error:
        emit('error', 'Volume-key listener: ' + str(error))
    finally:
        selector.close()
        if device is not None:
            os.close(device)


if __name__ == '__main__':
    run()
'''


def helper_source():
    """Pass fixed code directly to isolated Python; root imports no app module."""
    return ('import json, os, selectors, stat, struct\nfrom pathlib import Path\n' +
            inspect.getsource(VolumeTapSequence) + '\n' + _HELPER_BODY)


def root_command():
    executable = Path(sys.executable).resolve()
    prefix = Path(os.environ.get('PREFIX', str(executable.parent.parent)))
    if (not str(prefix).startswith('/data/data/com.termux/files/usr')
            or not Path('/system/bin/getevent').exists()):
        raise RuntimeError('Volume-button shortcuts require this phone’s native Termux session.')
    su = shutil.which('su')
    if not su and os.access('/system/bin/su', os.X_OK):
        su = '/system/bin/su'
    if not su:
        raise RuntimeError('Volume-button shortcuts need the phone’s existing root command.')
    # Root receives no model/project environment or writable import paths. No
    # shell source, device path or permission change comes from a model request.
    argv = ['/system/bin/env', '-i', 'PATH=/system/bin',
            'LD_LIBRARY_PATH=' + str(prefix / 'lib'),
            str(executable), '-I', '-S', '-B', '-c', helper_source()]
    return [su, '-c', shlex.join(argv)]


class VolumeHotkey:
    """Background key reader; consumers handle its queued events on the UI thread.

    Events are ('ready', None), ('toggle', None), and ('error', message). Start
    never enables capture itself. close() closes the helper's lifetime pipe and
    waits finitely for release; it does not kill unrelated root processes.
    """
    def __init__(self, command_factory=root_command, process_factory=subprocess.Popen,
                 startup_timeout=8, close_timeout=2):
        self.events = queue.Queue()
        self.command_factory, self.process_factory = command_factory, process_factory
        self.startup_timeout, self.close_timeout = startup_timeout, close_timeout
        self.status = 'off'
        self.error = None
        self.process = None
        self.worker = None
        self._lock = threading.RLock()
        self._closed = threading.Event()

    def _event(self, kind, message=None):
        with self._lock:
            if kind == 'error':
                self.status, self.error = 'error', message
            elif kind == 'ready':
                self.status = 'ready'
            if not self._closed.is_set():
                self.events.put((kind, message))

    @staticmethod
    def _release(process):
        if process is not None and process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass

    def start(self):
        with self._lock:
            if self._closed.is_set() or self.worker is not None:
                return
            self.status = 'starting'
            self.worker = threading.Thread(target=self._run, name='volume-microphone-shortcut', daemon=True)
            self.worker.start()

    def _run(self):
        process = None
        selector = selectors.DefaultSelector()
        ready = False
        diagnostic = ''
        try:
            if self._closed.is_set():
                return
            command = self.command_factory()
            process = self.process_factory(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            with self._lock:
                self.process = process
                if self._closed.is_set():
                    self._release(process)
            selector.register(process.stdout, selectors.EVENT_READ)
            pending = bytearray()
            deadline = time.monotonic() + self.startup_timeout
            while selector.get_map():
                if self._closed.is_set():
                    self._release(process)
                if not ready and time.monotonic() >= deadline:
                    raise RuntimeError('Volume-key listener did not become ready; check the existing root grant.')
                for key, _ in selector.select(.1):
                    chunk = os.read(key.fileobj.fileno(), 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        break
                    pending.extend(chunk)
                    if len(pending) > 8192:
                        raise RuntimeError('Volume-key listener returned an oversized status message.')
                    while b'\n' in pending:
                        line, _, pending = pending.partition(b'\n')
                        try:
                            event = json.loads(line)
                        except (ValueError, UnicodeError):
                            diagnostic = line.decode('utf-8', 'replace')[:240]
                            continue
                        if not isinstance(event, dict):
                            continue
                        kind = event.get('event')
                        if kind == 'ready':
                            ready = True
                            self._event('ready')
                        elif kind == 'toggle' and ready:
                            self._event('toggle')
                        elif kind == 'error':
                            raise RuntimeError(str(event.get('message', 'Volume-key listener failed.'))[:500])
            if not self._closed.is_set():
                raise RuntimeError(diagnostic or 'Volume-key listener exited unexpectedly.')
        except (OSError, RuntimeError, subprocess.SubprocessError) as error:
            self._event('error', str(error))
        finally:
            selector.close()
            self._release(process)
            if process is not None:
                try:
                    process.wait(timeout=self.close_timeout)
                except subprocess.TimeoutExpired:
                    self._event('error', 'Volume-key listener did not confirm shutdown.')
                if process.stdout is not None:
                    process.stdout.close()
            with self._lock:
                self.process = None
                if self._closed.is_set() and self.status != 'error':
                    self.status = 'closed'

    def close(self):
        self._closed.set()
        with self._lock:
            self._release(self.process)
            worker = self.worker
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=self.close_timeout + .3)
            if worker.is_alive():
                with self._lock:
                    self.status, self.error = 'error', 'Volume-key listener did not confirm shutdown.'
        elif self.status != 'error':
            self.status = 'closed'
