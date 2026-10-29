"""Authenticated, loopback-only transport to the installed Android voice helper.

Creating or probing a bridge never activates the microphone. start() must be
called when the user enters a voice conversation; close() releases microphone
and playback. The native service also stops if this connection disappears.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import threading
import time
from typing import Callable

PACKAGE = 'org.ostadix.voicebridge'
COMPONENT = PACKAGE + '/.BridgeActivity'
PROTOCOL = 1

class NativeVoiceError(RuntimeError):
    pass

class NativeVoiceBridge:
    def __init__(self, on_event: Callable[[dict], None], *, language: str = 'en-US'):
        self.on_event = on_event
        self.language = language
        self._listener = None
        self._socket = None
        self._send_lock = threading.Lock()
        self._closed = threading.Event()
        self._connected = threading.Event()
        self._ready = threading.Event()
        self._capabilities = {}
        self._error = None
        self._probe = False
        self._token = secrets.token_hex(32)
        self._thread = None
        self._last_sequence = 0
        self._native_stopped = threading.Event()
        self._close_lock = threading.RLock()
        self._microphone_lock = threading.RLock()
        self._microphone_muted = False

    @staticmethod
    def installed() -> bool:
        # Package query via pm is unreliable across Android package-visibility
        # boundaries. This local receipt is installed only after successful pm.
        return (Path(__file__).parent/'build'/'installed.json').is_file()

    @property
    def capabilities(self):
        return dict(self._capabilities)

    def _launch(self, probe=False):
        if self._closed.is_set():
            raise NativeVoiceError('Android voice conversation was closed')
        if self._thread is not None:
            raise NativeVoiceError('This voice connection has already been started')
        self._probe = probe
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.bind(('127.0.0.1', 0))
        self._listener.listen(4)
        self._listener.settimeout(1)
        port = self._listener.getsockname()[1]
        self._thread = threading.Thread(target=self._receive, name='native-voice-events', daemon=True)
        self._thread.start()
        am = '/system/bin/am' if os.getuid() in (0, 1000, 2000) else '/data/data/com.termux/files/usr/bin/am'
        args = [am, 'start', '-n', COMPONENT,
                '--ei', 'port', str(port), '--es', 'token', self._token,
                '--es', 'language', self.language]
        if probe:
            args.extend(['--ez', 'probe', 'true'])
        env = os.environ.copy()
        try:
            from companion_notify import _api_environment
            env = _api_environment()
        except (ImportError, RuntimeError, OSError, ValueError, subprocess.SubprocessError):
            pass
        try:
            launched = subprocess.run(args, env=env, capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.close()
            raise NativeVoiceError('Could not open the Android voice service') from exc
        if self._closed.is_set():
            raise NativeVoiceError('Android voice conversation was closed')
        # Never include launch arguments/output: the one-session token is private.
        if launched.returncode or 'Error:' in launched.stdout or 'Error:' in launched.stderr:
            self.close()
            raise NativeVoiceError('Android voice helper could not start; rebuild/install android_voice/build.py')

    def start(self, timeout=20):
        """Start microphone and return once Android reports recognition ready.

        Call on a worker thread. Events arrive before this returns; timeout or
        failure shuts down the connection and microphone. No half-duplex fallback.
        """
        self._launch()
        deadline = time.monotonic()+timeout
        if not self._connected.wait(max(0,deadline-time.monotonic())):
            self.close()
            raise NativeVoiceError('Android voice helper did not connect')
        with self._microphone_lock:
            command = {'cmd': 'start'}
            if self._microphone_muted:
                command['microphone_muted'] = True
            self.send(command)
        threading.Thread(target=self._heartbeat, name='native-voice-heartbeat', daemon=True).start()
        if not self._ready.wait(max(0,deadline-time.monotonic())):
            error=self._error or 'Android continuous recognition did not become ready'
            self.close()
            raise NativeVoiceError(error)
        if self._error:
            error=self._error
            self.close()
            raise NativeVoiceError(error)
        return self.capabilities

    def probe(self, timeout=10):
        """Read Android capability flags without recording or playing audio."""
        self._launch(probe=True)
        if not self._ready.wait(timeout):
            self.close()
            raise NativeVoiceError('Android voice capability probe timed out')
        data=self.capabilities
        self.close()
        return data

    def _receive(self):
        connection=None
        stream=None
        try:
            deadline=time.monotonic()+25
            listener=self._listener
            while not self._closed.is_set() and time.monotonic()<deadline:
                try:
                    candidate,_=listener.accept()
                except socket.timeout:
                    continue
                candidate.settimeout(3)
                stream=candidate.makefile('rb')
                try:
                    hello=stream.readline(4097)
                    data=json.loads(hello)
                    if (len(hello)>4096 or data.get('type')!='hello'
                            or data.get('protocol')!=PROTOCOL
                            or not secrets.compare_digest(str(data.get('token','')),self._token)):
                        stream.close(); candidate.close(); continue
                except (ValueError,OSError,TypeError,AttributeError):
                    stream.close(); candidate.close(); continue
                connection=candidate
                self._socket=connection
                connection.settimeout(None)
                self._connected.set()
                break
            if connection is None:
                return
            while not self._closed.is_set():
                line=stream.readline(65537)
                if not line: break
                if len(line)>65536:
                    raise NativeVoiceError('Android voice event exceeded its size limit')
                event=json.loads(line)
                if not isinstance(event,dict): continue
                kind=event.get('type')
                if kind=='stopped' and event.get('mic_active') is False:
                    self._capabilities.update(mic_active=False, capture_continuous=False, speaking=False, status_fresh=True, capture_state='stopped')
                    self._native_stopped.set()
                if kind=='capabilities':
                    self._capabilities.update(event)
                    if self._probe: self._ready.set()
                if kind=='ready':
                    self._capabilities.update({k:v for k,v in event.items() if k not in ('type','seq','time_ms')})
                    self._ready.set()
                if kind=='error' and event.get('fatal'):
                    self._error=event.get('message','Android voice capture failed')
                    self._ready.set()
                # Sequence is scoped to this connection, not a saved conversation.
                seq=event.get('seq')
                if isinstance(seq,int):
                    if seq<=self._last_sequence: continue
                    self._last_sequence=seq
                if kind=='pong':
                    for key in ('frames','dropped_frames'):
                        if type(event.get(key)) is int and event[key]>=0:
                            self._capabilities[key]=event[key]
                    if type(event.get('mic_active')) is bool:
                        self._capabilities.update(mic_active=event['mic_active'],status_fresh=True)
                if kind=='microphone_muted' and type(event.get('muted')) is bool:
                    self._capabilities.update(microphone_muted=event['muted'],
                                              mic_active=event.get('mic_active') is True,
                                              capture_continuous=event.get('mic_active') is True,
                                              status_fresh=True)
                    if type(event.get('audio_mode_owned')) is bool:
                        self._capabilities['audio_mode_owned']=event['audio_mode_owned']
                try: self.on_event(event)
                except Exception: pass  # UI errors must not kill capture ownership.
        except (OSError,ValueError,NativeVoiceError) as error:
            if not self._closed.is_set():
                self._error='Android voice connection ended: '+str(error)
        finally:
            if stream is not None:
                try: stream.close()
                except OSError: pass
            if connection is not None:
                try: connection.close()
                except OSError: pass
            if not self._closed.is_set() and not self._probe:
                if not self._native_stopped.is_set():
                    self._invalidate_capture_status()
                self._error=self._error or 'Android voice conversation closed'
                self._ready.set()
                try: self.on_event({'type':'stopped','message':self._error})
                except Exception: pass
                self._closed.set()

    def _heartbeat(self):
        while not self._closed.wait(5):
            try: self.send({'cmd':'ping'})
            except NativeVoiceError: return

    def send(self, payload):
        if self._closed.is_set() or self._socket is None:
            raise NativeVoiceError('Android voice conversation is not connected')
        encoded=(json.dumps(payload,separators=(',',':'))+'\n').encode('utf-8')
        if len(encoded)>32768:
            raise NativeVoiceError('Voice command is too large')
        try:
            with self._send_lock: self._socket.sendall(encoded)
        except OSError as exc:
            raise NativeVoiceError('Android voice connection ended') from exc

    def speak(self, text, utterance_id='answer'):
        self.send({'cmd':'speak','text':str(text),'id':str(utterance_id)})

    def stop_speaking(self):
        """Cancel playback without touching microphone capture or recognition."""
        self.send({'cmd':'stop_speaking'})

    def set_microphone_muted(self, muted):
        """Release/resume microphone capture without stopping speech playback.

        Before start(), remember the requested state so muted startup never
        briefly records. Confirmation arrives as a microphone_muted event.
        """
        if type(muted) is not bool:
            raise NativeVoiceError('Microphone mute state must be boolean')
        with self._microphone_lock:
            if self._closed.is_set():
                raise NativeVoiceError('Android voice conversation is not connected')
            previous, self._microphone_muted = self._microphone_muted, muted
            if self._socket is not None:
                try:
                    self.send({'cmd': 'microphone_muted', 'muted': muted})
                except NativeVoiceError:
                    self._microphone_muted = previous
                    raise

    def _invalidate_capture_status(self):
        for key in ('mic_active', 'capture_continuous', 'speaking'):
            self._capabilities.pop(key, None)
        self._capabilities.update(status_fresh=False, capture_state='unknown')

    def close(self):
        """Wait briefly for native microphone/playback release before returning."""
        with self._close_lock:
            if not self._closed.is_set() and self._socket is not None:
                try:
                    self.send({'cmd':'stop'})
                    if not self._probe and self._thread is not threading.current_thread():
                        self._native_stopped.wait(timeout=3)
                except NativeVoiceError:
                    pass
            if not self._probe and not self._native_stopped.is_set():
                self._invalidate_capture_status()
            self._closed.set()
            self._connected.set()
            self._ready.set()
            for connection in (self._socket,self._listener):
                if connection is not None:
                    try: connection.shutdown(socket.SHUT_RDWR)
                    except OSError: pass
                    try: connection.close()
                    except OSError: pass
            self._socket=None
            self._listener=None
            if self._thread is not None and self._thread is not threading.current_thread():
                self._thread.join(timeout=1)
