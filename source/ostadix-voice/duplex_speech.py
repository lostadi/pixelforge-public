"""Terminal audio adapter for a continuously capturing Android voice service."""
from collections import deque
import queue
import threading
import time
import uuid

from android_voice.bridge import NativeVoiceBridge, NativeVoiceError
from agent import AppError
from speech_text import speech_chunks


class DuplexSpeech:
    duplex = True

    def __init__(self, fallback, bridge_factory=NativeVoiceBridge):
        self.events = queue.Queue()
        self.fallback = fallback
        self.fallback.events = self.events
        self.bridge_factory = bridge_factory
        self.bridge = None
        self.lock = threading.RLock()
        self.generation = 0
        self.closed = False
        self.starting = False
        self.listening = False
        self.speaking = False
        self.user_speaking = False
        self.microphone_muted = False
        self.microphone_muted_confirmed = False
        self._stopping = False
        self.playback_id = None
        self.capabilities = {}
        self._pending_speech = None
        self._pending_chunks = deque()
        self._fallback_pending = None
        self._fallback_worker = None
        self._connect_thread = None
        self._stop_thread = None
        self._mute_thread = None

    @property
    def mode(self):
        with self.lock:
            if self.starting:
                return 'starting microphone'
            if self.listening and self.speaking:
                return 'listening and speaking'
            if self.listening:
                return 'listening'
            if self.speaking:
                return 'speaking'
            if self.bridge is not None and self.microphone_muted:
                return 'microphone muted'
            if self._fallback_pending is not None:
                return 'speaking'
            return self.fallback.mode

    @property
    def stopping(self):
        with self.lock:
            return self._stopping or self.fallback.stopping

    def _clear_native_state(self):
        self.starting = self.listening = self.speaking = self.user_speaking = False
        self.playback_id = None
        self._pending_speech = None
        self._pending_chunks.clear()
        self.microphone_muted_confirmed = False

    def _event(self, generation, event):
        with self.lock:
            if generation != self.generation or self.closed:
                return
            kind = event.get('type')
            if kind == 'capabilities':
                self.capabilities.update(event)
                self.capabilities['status_fresh'] = True
                if type(event.get('microphone_muted')) is bool:
                    self.microphone_muted_confirmed = event['microphone_muted'] and event.get('mic_active') is False
            elif kind == 'ready':
                self.starting = False
                self.listening = event.get('mic_active') is True
                self.capabilities.update(event)
                self.capabilities['status_fresh'] = True
                self.capabilities['recognition_state'] = 'ready'
                self.capabilities.pop('recognition_error_code', None)
                if self.microphone_muted:
                    self.listening = False
                self.microphone_muted_confirmed = event.get('microphone_muted') is True
                if not self.listening and not self.microphone_muted:
                    self.events.put(('error', 'The native microphone did not start.'))
                    self.stop()
                elif self._pending_speech is not None:
                    text, identifier = self._pending_speech
                    self._pending_speech = None
                    self._speak_native(text, identifier)
            elif kind == 'microphone_muted':
                if type(event.get('muted')) is not bool:
                    return
                self.microphone_muted_confirmed = event['muted'] and event.get('mic_active') is False
                self.listening = event.get('mic_active') is True and not self.microphone_muted
                self.user_speaking = False
                self.capabilities.update(microphone_muted=event['muted'],
                                         mic_active=event.get('mic_active') is True,
                                         capture_continuous=event.get('mic_active') is True,
                                         status_fresh=True)
                if type(event.get('audio_mode_owned')) is bool:
                    self.capabilities['audio_mode_owned'] = event['audio_mode_owned']
                self.events.put(('microphone_muted', {'muted': event['muted'],
                                 'mic_active': event.get('mic_active') is True}))
            elif kind == 'partial':
                if self.microphone_muted:
                    return
                text = str(event.get('text', ''))[:8000]
                if text:
                    self.user_speaking = True
                    if self.speaking:
                        self.interrupt_speech()
                    self.events.put(('partial', text))
            elif kind == 'final':
                if self.microphone_muted:
                    return
                self.user_speaking = False
                text = str(event.get('text', '')).strip()
                if text:
                    self.events.put(('heard', text[:8000]))
            elif kind == 'speech_start':
                if self.microphone_muted:
                    return
                self.user_speaking = True
                if self.speaking:
                    self.interrupt_speech()
                self.events.put(('speech_start', ''))
            elif kind == 'pong':
                for key in ('frames', 'dropped_frames'):
                    if type(event.get(key)) is int and event[key] >= 0:
                        self.capabilities[key] = event[key]
                if type(event.get('mic_active')) is bool:
                    self.capabilities.update(mic_active=event['mic_active'], status_fresh=True)
            elif kind == 'speech_end':
                self.user_speaking = False
            elif kind == 'tts_start' and event.get('id') == self.playback_id:
                self.speaking = True
            elif kind == 'tts_done' and event.get('id') == self.playback_id:
                if self._pending_chunks and not event.get('interrupted'):
                    self.playback_id = uuid.uuid4().hex
                    self._speak_native(self._pending_chunks.popleft(), self.playback_id)
                else:
                    self.speaking = False
                    self.playback_id = None
                    self._pending_chunks.clear()
            elif kind == 'error':
                if self.microphone_muted and event.get('scope') == 'recognition':
                    return
                if event.get('scope') == 'recognition':
                    self.user_speaking = False
                    self.capabilities.update(recognition_state='recovering' if event.get('recovering') else 'failed',
                                             recognition_silence=event.get('silence') is True,
                                             recognition_restarts=self.capabilities.get('recognition_restarts', 0) + 1)
                    if type(event.get('code')) is int:
                        self.capabilities['recognition_error_code'] = event['code']
                if not event.get('recovering'):
                    if event.get('scope') == 'tts':
                        # A replaced chunk can report an error after a new
                        # response starts. It no longer owns playback state.
                        if event.get('id') is not None and event['id'] != self.playback_id:
                            return
                        self.speaking = False
                        self.playback_id = None
                        self._pending_speech = None
                        self._pending_chunks.clear()
                    self.events.put(('playback_error' if event.get('scope') == 'tts' else 'error',
                                     str(event.get('message', 'Native voice failed.'))))
                    if event.get('fatal'):
                        self.stop()
            elif kind == 'stopped':
                self.events.put(('error', str(event.get('message', 'Native voice disconnected.'))))
                self.stop()

    def _speak_native(self, text, identifier):
        try:
            self.bridge.speak(text, identifier)
        except (NativeVoiceError, OSError) as error:
            self.speaking = False
            self.playback_id = None
            self._pending_speech = None
            self._pending_chunks.clear()
            self.events.put(('playback_error', str(error)))

    def _start_fallback_playback(self, text):
        """Replace playback after its old Android service finishes stopping."""
        if self.fallback.mode == 'listening':
            # Reviewed dictation owns its microphone until capture finishes.
            return self.fallback.start('speaking', text)
        if (not self._stopping and not self.fallback.stopping and
                self.fallback.mode is None and self._fallback_worker is None):
            return self.fallback.start('speaking', text)
        self._fallback_pending = text or ''
        if self.fallback.mode == 'speaking':
            self.fallback.stop()
        if self._fallback_worker is not None:
            return

        def replace():
            deadline = time.monotonic() + 45
            waiter = threading.Event()
            try:
                while True:
                    with self.lock:
                        if self.closed or self._fallback_pending is None:
                            self._fallback_worker = None
                            return
                        if not self._stopping and not self.fallback.stopping:
                            latest, self._fallback_pending = self._fallback_pending, None
                            # Release worker ownership before starting playback;
                            # another answer may immediately replace this one.
                            self._fallback_worker = None
                            if getattr(self.fallback, 'cleanup_failed', False):
                                # The cleanup owner already reported its error.
                                # An unconfirmed stop cannot release queued TTS.
                                return
                            try:
                                self.fallback.start('speaking', latest)
                            except (AppError, OSError) as error:
                                self.events.put(('playback_error', str(error)))
                            return
                        if time.monotonic() >= deadline:
                            self._fallback_pending = None
                            self._fallback_worker = None
                            self.events.put(('playback_error', 'Audio cleanup did not finish; playback was not restarted.'))
                            return
                        stopper = (self._stop_thread if self._stopping else
                                   getattr(self.fallback, 'stop_thread', None))
                    if stopper is not None and stopper is not threading.current_thread():
                        stopper.join(timeout=.05)
                    else:
                        waiter.wait(.05)
            finally:
                with self.lock:
                    if self._fallback_worker is threading.current_thread():
                        self._fallback_worker = None

        self._fallback_worker = threading.Thread(target=replace, name='voice-replace-playback', daemon=True)
        self._fallback_worker.start()

    def start_once(self):
        """Reviewed one-shot dictation, without enabling continuous capture."""
        with self.lock:
            if self.closed or self.stopping:
                raise AppError('Audio is stopping; try again shortly.')
            if self.microphone_muted:
                raise AppError('Your microphone is muted. Unmute it before recording.')
            if self.listening or self.starting or self.bridge is not None:
                raise AppError('Continuous voice is already listening. Turn /voice off before reviewed dictation.')
            if self._fallback_pending is not None:
                raise AppError('Wait for the current audio to finish before opening the microphone.')
            return self.fallback.start('listening')

    def start(self, mode, text=None):
        with self.lock:
            if self.closed or (self.stopping and mode != 'speaking'):
                raise AppError('Audio is stopping; try again shortly.')
            if mode == 'speaking':
                if self.bridge is None:
                    # /hear or /speaker in text mode must not open a microphone.
                    return self._start_fallback_playback(text)
                # Android ignores whitespace-only utterances without a done
                # callback, so those must not block the rest of a response.
                chunks = deque(chunk for chunk in speech_chunks(text or '') if chunk.strip())
                if not chunks:
                    return
                first = chunks.popleft()
                self._pending_chunks = chunks
                self.playback_id = uuid.uuid4().hex
                self.speaking = True
                if self.starting:
                    self._pending_speech = (first, self.playback_id)
                else:
                    self._speak_native(first, self.playback_id)
                return
            if mode != 'listening':
                raise AppError('Unknown audio mode')
            if self.bridge is not None or self.listening or self.starting:
                return
            if self.fallback.mode or self._fallback_pending is not None:
                raise AppError('Wait for the current audio to finish before opening voice mode.')
            self.generation += 1
            generation = self.generation
            self.starting = True
            self.capabilities = {'status_fresh': False, 'capture_state': 'starting'}
            try:
                self.bridge = bridge = self.bridge_factory(lambda event: self._event(generation, event))
                if self.microphone_muted:
                    bridge.set_microphone_muted(True)
            except Exception:
                self.starting = False
                raise

            def connect():
                try:
                    result = bridge.start()
                    with self.lock:
                        if generation == self.generation and not self.closed:
                            self.capabilities.update(result)
                except (NativeVoiceError, OSError, RuntimeError) as error:
                    with self.lock:
                        if generation == self.generation and not self.closed:
                            self.events.put(('error', str(error)))
                            self.stop()
                finally:
                    with self.lock:
                        stale = generation != self.generation or self.closed
                    if stale:
                        bridge.close()

            self._connect_thread = threading.Thread(target=connect, name='voice-connect', daemon=True)
            self._connect_thread.start()

    def interrupt_speech(self):
        """Barge-in cancels the answer audio without stopping capture."""
        with self.lock:
            self.speaking = False
            self.playback_id = None
            self._pending_speech = None
            self._pending_chunks.clear()
            self._fallback_pending = None
            if self.bridge is not None:
                if not self.starting:
                    try:
                        self.bridge.stop_speaking()
                    except (NativeVoiceError, OSError) as error:
                        self.events.put(('playback_error', str(error)))
            elif self.fallback.mode == 'speaking':
                self.fallback.stop()

    def set_microphone_muted(self, muted):
        """Mute only the user's microphone; native TTS remains connected."""
        if type(muted) is not bool:
            raise AppError('Microphone mute state must be boolean')
        with self.lock:
            if self.closed or self.stopping:
                raise AppError('Audio is stopping; try again shortly.')
            previous = self.microphone_muted
            self.microphone_muted = muted
            self.microphone_muted_confirmed = False
            self.user_speaking = False
            try:
                if self.bridge is not None:
                    self.bridge.set_microphone_muted(muted)
                    if muted:
                        self.listening = False
                elif muted:
                    if self.fallback.mode == 'listening':
                        self.fallback.stop()
                    if self.fallback.stopping:
                        generation = self.generation
                        def confirm_fallback():
                            stopper = getattr(self.fallback, 'stop_thread', None)
                            if stopper is not None:
                                stopper.join(timeout=6.5)
                            with self.lock:
                                if (self.closed or generation != self.generation or not self.microphone_muted
                                        or self.fallback.stopping or getattr(self.fallback, 'cleanup_failed', False)):
                                    return
                                self.microphone_muted_confirmed = True
                                self.capabilities.update(microphone_muted=True, mic_active=False,
                                                         capture_continuous=False, status_fresh=True)
                                self.events.put(('microphone_muted', {'muted': True, 'mic_active': False}))
                        self._mute_thread = threading.Thread(target=confirm_fallback, name='voice-mute-confirm', daemon=True)
                        self._mute_thread.start()
                    else:
                        self.microphone_muted_confirmed = True
                        self.capabilities.update(microphone_muted=True, mic_active=False,
                                                 capture_continuous=False, status_fresh=True)
                        self.events.put(('microphone_muted', {'muted': True, 'mic_active': False}))
                else:
                    self.capabilities.update(microphone_muted=False)
                    self.events.put(('microphone_muted', {'muted': False, 'mic_active': False}))
            except (NativeVoiceError, OSError) as error:
                self.microphone_muted = previous
                raise AppError('Microphone mute change was not confirmed: ' + str(error)) from error
            return {'microphone_muted': self.microphone_muted,
                    'confirmed': self.microphone_muted_confirmed, 'mic_active': self.listening}

    def stop(self):
        with self.lock:
            self.generation += 1
            bridge, self.bridge = self.bridge, None
            connector, self._connect_thread = self._connect_thread, None
            self._clear_native_state()
            self._fallback_pending = None
            self.fallback.stop()
            if bridge is None and connector is None:
                return  # A previous teardown may still own _stopping.
            self._stopping = True
            for key in ('mic_active', 'capture_continuous', 'speaking'):
                self.capabilities.pop(key, None)
            self.capabilities.update(status_fresh=False, capture_state='stopping')

            def disconnect():
                try:
                    if bridge is not None:
                        bridge.close()
                    # A pending Activity launch must finish before a new c can
                    # acquire the session lock and start its own native service.
                    if connector is not None and connector is not threading.current_thread():
                        connector.join(timeout=36)
                finally:
                    with self.lock:
                        if self._stop_thread is threading.current_thread():
                            self.capabilities.update(getattr(bridge, 'capabilities', {'status_fresh': False, 'capture_state': 'unknown'}))
                            self._stopping = False

            self._stop_thread = threading.Thread(target=disconnect, name='voice-close', daemon=True)
            self._stop_thread.start()

    def close(self):
        with self.lock:
            self.closed = True
            self.stop()
            worker = self._stop_thread
            replacement = self._fallback_worker
            mute_worker = self._mute_thread
        # Keep the terminal session lock until native teardown and any Activity
        # launcher have completed. Ordinary closure is milliseconds; bounds cover
        # the launch command's 15-second timeout plus transport cleanup.
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=41)
        self.fallback.close()
        if replacement is not None and replacement is not threading.current_thread():
            replacement.join(timeout=.1)
        if mute_worker is not None and mute_worker is not threading.current_thread():
            mute_worker.join(timeout=6.5)
