import json
import queue
import threading
import unittest
from unittest.mock import patch

from agent import AppError
from android_voice.bridge import NativeVoiceError
from duplex_speech import DuplexSpeech
from speech_text import speech_chunks


class SpeechChunkTests(unittest.TestCase):
    def test_chunks_preserve_all_content_and_fit_both_android_limits(self):
        for text in ('Sentence one. Sentence two.\n' * 700,
                     'x' * 20000, '😀漢字\\"\n' * 4000):
            with self.subTest(prefix=text[:30]):
                chunks = list(speech_chunks(text))
                self.assertEqual(''.join(chunks), text)
                self.assertGreater(len(chunks), 1)
                for chunk in chunks:
                    self.assertLessEqual(len(chunk.encode('utf-16-le')) // 2, 3000)
                    message = {'cmd': 'speak', 'text': chunk, 'id': 'f' * 32}
                    self.assertLess(len(json.dumps(message).encode('utf-8')) + 1, 32768)

    def test_prefers_word_boundary_without_losing_spaces_or_final_words(self):
        text = 'First answer.  Second answer.\nFinal words.'
        chunks = list(speech_chunks(text, limit=20))
        self.assertEqual(''.join(chunks), text)
        self.assertEqual(chunks[0], 'First answer.  ')
        self.assertEqual(list(speech_chunks('')), [])

    def test_small_chunks_keep_astral_characters_whole(self):
        self.assertEqual(list(speech_chunks('😀😀a', limit=3)), ['😀', '😀a'])
        with self.assertRaises(ValueError):
            list(speech_chunks('😀', limit=1))


class FakeFallback:
    def __init__(self):
        self.mode = None
        self.stopping = False
        self.started = []
        self.stop_count = 0
        self.closed = False
        self.events = queue.Queue()

    def start(self, mode, text=None):
        if self.mode:
            raise AppError('busy')
        self.mode = mode
        self.started.append((mode, text))

    def stop(self):
        self.mode = None
        self.stop_count += 1

    def close(self):
        self.closed = True
        self.stop()


class FakeBridge:
    def __init__(self, callback, start_gate=None, close_gate=None, failure=None):
        self.callback = callback
        self.start_gate = start_gate
        self.close_gate = close_gate
        self.failure = failure
        self.started = threading.Event()
        self.ready = threading.Event()
        self.close_entered = threading.Event()
        self.closed = threading.Event()
        self.spoken = []
        self.interruptions = 0
        self.capabilities = {}
        self.microphone_muted = False
        self.mute_changes = []

    def start(self):
        self.started.set()
        if self.start_gate is not None:
            self.start_gate.wait(2)
        if self.failure:
            raise self.failure
        self.capabilities.update(mic_active=not self.microphone_muted, capture_continuous=not self.microphone_muted)
        self.callback({'type': 'ready', 'mic_active': not self.microphone_muted,
                       'microphone_muted': self.microphone_muted})
        self.ready.set()
        return {'on_device_available': True}

    def speak(self, text, identifier):
        self.spoken.append((text, identifier))

    def stop_speaking(self):
        self.interruptions += 1

    def set_microphone_muted(self, muted):
        self.microphone_muted = muted
        self.mute_changes.append(muted)
        if self.ready.is_set():
            self.callback({'type': 'microphone_muted', 'muted': muted, 'mic_active': not muted})

    def close(self):
        self.close_entered.set()
        if self.close_gate is not None:
            self.close_gate.wait(2)
        self.capabilities.update(mic_active=False, capture_continuous=False, status_fresh=True, capture_state='stopped')
        self.closed.set()


class GatedFallback(FakeFallback):
    """Model NativeSpeech's asynchronous Android service teardown."""
    def __init__(self, stop_gate):
        super().__init__()
        self.stop_gate = stop_gate
        self.stop_thread = None
        self.cleanup_failed = False
        self.playback_started = threading.Event()

    def start(self, mode, text=None):
        if self.stopping or self.closed:
            raise AppError('busy')
        super().start(mode, text)
        self.playback_started.set()

    def stop(self):
        if self.stopping:
            return
        had_audio = self.mode is not None
        super().stop()
        if not had_audio:
            return
        self.stopping = True

        def finish():
            self.stop_gate.wait(2)
            self.stopping = False

        self.stop_thread = threading.Thread(target=finish)
        self.stop_thread.start()


class DuplexSpeechTests(unittest.TestCase):
    def create(self, **settings):
        fallback = FakeFallback()
        created = []
        def factory(callback):
            bridge = FakeBridge(callback, **settings)
            created.append(bridge)
            return bridge
        audio = DuplexSpeech(fallback, factory)
        self.addCleanup(audio.close)
        return audio, fallback, created

    def ready(self, audio, created):
        audio.start('listening')
        self.assertTrue(created[-1].ready.wait(1))
        return created[-1]

    def replacement_audio(self):
        gate = threading.Event()
        fallback = GatedFallback(gate)
        created = []
        def factory(callback):
            bridge = FakeBridge(callback)
            created.append(bridge)
            return bridge
        audio = DuplexSpeech(fallback, factory)
        self.addCleanup(audio.close)
        self.addCleanup(gate.set)
        audio.start('speaking', 'Old answer')
        fallback.playback_started.clear()
        return audio, fallback, gate, created

    def test_reviewed_dictation_stays_one_shot(self):
        audio, fallback, created = self.create()
        audio.start_once()
        self.assertEqual(fallback.started, [('listening', None)])
        self.assertEqual(audio.mode, 'listening')
        self.assertFalse(audio.listening)
        self.assertEqual(created, [])
        fallback.events.put(('heard', 'review before sending'))
        self.assertEqual(audio.events.get_nowait(), ('heard', 'review before sending'))

    def test_text_mode_playback_does_not_start_microphone(self):
        audio, fallback, created = self.create()
        audio.start('speaking', 'Hello')
        self.assertEqual(audio.mode, 'speaking')
        self.assertEqual(fallback.started, [('speaking', 'Hello')])
        self.assertEqual(created, [])

    def test_microphone_mute_preserves_playback_and_suppresses_all_transcripts(self):
        audio, fallback, created = self.create()
        bridge = self.ready(audio, created)
        answer = 'Continue the entire spoken answer. ' * 400
        audio.start('speaking', answer)
        current = audio.playback_id
        result = audio.set_microphone_muted(True)
        self.assertTrue(result['confirmed'])
        self.assertTrue(audio.microphone_muted_confirmed)
        self.assertFalse(audio.listening)
        self.assertEqual(audio.mode, 'speaking')
        self.assertEqual(audio.playback_id, current)
        for event in ({'type': 'speech_start'}, {'type': 'partial', 'text': 'private speech'},
                      {'type': 'final', 'text': 'private speech'}):
            bridge.callback(event)
        self.assertFalse(audio.user_speaking)
        self.assertEqual(bridge.interruptions, 0)
        self.assertEqual(fallback.stop_count, 0)
        self.assertEqual([kind for kind, _ in list(audio.events.queue)], ['microphone_muted'])
        while audio.speaking:
            bridge.callback({'type': 'tts_done', 'id': audio.playback_id})
        self.assertEqual(''.join(text for text, _ in bridge.spoken), answer)
        self.assertEqual(audio.mode, 'microphone muted')
        self.assertFalse(bridge.closed.is_set())

    def test_unmute_resumes_same_native_connection_without_restarting_tts(self):
        audio, _, created = self.create()
        bridge = self.ready(audio, created)
        audio.start('speaking', 'The AI keeps speaking.')
        identifier = audio.playback_id
        audio.set_microphone_muted(True)
        audio.set_microphone_muted(False)
        self.assertFalse(audio.microphone_muted)
        self.assertTrue(audio.listening)
        self.assertTrue(audio.speaking)
        self.assertEqual(audio.playback_id, identifier)
        self.assertEqual(bridge.mute_changes, [True, False])
        self.assertEqual(len(created), 1)
        self.assertEqual(bridge.interruptions, 0)
        bridge.callback({'type': 'final', 'text': 'New request'})
        self.assertIn(('heard', 'New request'), list(audio.events.queue))

    def test_muted_startup_never_opens_capture_and_still_allows_playback(self):
        audio, _, created = self.create()
        audio.set_microphone_muted(True)
        bridge = self.ready(audio, created)
        self.assertTrue(audio.microphone_muted_confirmed)
        self.assertFalse(audio.listening)
        self.assertFalse(bridge.capabilities['mic_active'])
        self.assertEqual(bridge.mute_changes, [True])
        audio.start('speaking', 'Playback works while muted.')
        self.assertEqual(bridge.spoken[0][0], 'Playback works while muted.')
        self.assertNotIn('error', [kind for kind, _ in list(audio.events.queue)])

    def test_mute_during_native_startup_is_applied_before_ready(self):
        gate = threading.Event()
        audio, _, created = self.create(start_gate=gate)
        self.addCleanup(gate.set)
        audio.start('listening')
        self.assertTrue(created[0].started.wait(1))
        audio.set_microphone_muted(True)
        audio.start('speaking', 'Queued speech stays available.')
        gate.set()
        self.assertTrue(created[0].ready.wait(1))
        self.assertFalse(audio.listening)
        self.assertTrue(audio.microphone_muted_confirmed)
        self.assertTrue(audio.speaking)
        self.assertEqual(created[0].spoken[0][0], 'Queued speech stays available.')
        self.assertFalse(created[0].closed.is_set())

    def test_a_ready_callback_already_in_flight_cannot_override_requested_mute(self):
        audio, _, created = self.create()
        bridge = self.ready(audio, created)
        with patch.object(bridge, 'set_microphone_muted'):
            audio.set_microphone_muted(True)
        bridge.callback({'type': 'ready', 'mic_active': True, 'microphone_muted': False})
        self.assertTrue(audio.microphone_muted)
        self.assertFalse(audio.listening)
        self.assertFalse(bridge.closed.is_set())
        self.assertTrue(audio.events.empty())
        bridge.callback({'type': 'microphone_muted', 'muted': True, 'mic_active': False})
        self.assertTrue(audio.microphone_muted_confirmed)

    def test_microphone_mute_in_text_mode_does_not_stop_fallback_speech(self):
        audio, fallback, created = self.create()
        audio.start('speaking', 'Keep this answer audible.')
        audio.set_microphone_muted(True)
        self.assertEqual(fallback.mode, 'speaking')
        self.assertEqual(fallback.stop_count, 0)
        self.assertEqual(created, [])
        with self.assertRaisesRegex(AppError, 'microphone is muted'):
            audio.start_once()

    def test_microphone_mute_stops_only_active_reviewed_dictation(self):
        audio, fallback, _ = self.create()
        audio.start_once()
        audio.set_microphone_muted(True)
        self.assertEqual(fallback.stop_count, 1)
        self.assertIsNone(fallback.mode)
        self.assertTrue(audio.microphone_muted_confirmed)

    def test_text_playback_replacement_waits_for_old_service_to_stop(self):
        audio, fallback, gate, created = self.replacement_audio()
        audio.start('speaking', 'Latest complete answer')
        self.assertEqual(audio.mode, 'speaking')
        self.assertTrue(audio.stopping)
        self.assertFalse(fallback.playback_started.is_set())
        self.assertEqual(fallback.started, [('speaking', 'Old answer')])
        gate.set()
        self.assertTrue(fallback.playback_started.wait(1))
        self.assertEqual(fallback.started[-1], ('speaking', 'Latest complete answer'))
        self.assertFalse(audio.stopping)
        self.assertTrue(audio.events.empty())
        self.assertEqual(created, [])

    def test_rapid_playback_replacements_keep_only_the_latest_answer(self):
        audio, fallback, gate, _ = self.replacement_audio()
        audio.start('speaking', 'Superseded answer')
        worker = audio._fallback_worker
        audio.start('speaking', 'Another superseded answer')
        audio.start('speaking', 'Latest answer')
        self.assertIs(audio._fallback_worker, worker)
        self.assertEqual(fallback.stop_count, 1)
        gate.set()
        self.assertTrue(fallback.playback_started.wait(1))
        self.assertEqual(fallback.started, [('speaking', 'Old answer'),
                                            ('speaking', 'Latest answer')])
        self.assertTrue(audio.events.empty())

    def test_stop_or_interrupt_cancels_a_queued_playback_replacement(self):
        for action in ('stop', 'interrupt_speech'):
            with self.subTest(action=action):
                audio, fallback, gate, _ = self.replacement_audio()
                audio.start('speaking', 'Must not resume')
                worker = audio._fallback_worker
                getattr(audio, action)()
                worker.join(1)
                gate.set()
                fallback.stop_thread.join(1)
                self.assertIsNone(audio._fallback_pending)
                self.assertFalse(fallback.playback_started.is_set())
                self.assertEqual(fallback.started, [('speaking', 'Old answer')])

    def test_close_cancels_a_queued_playback_replacement(self):
        audio, fallback, gate, _ = self.replacement_audio()
        audio.start('speaking', 'Must not resume')
        audio.close()
        gate.set()
        fallback.stop_thread.join(1)
        self.assertEqual(fallback.started, [('speaking', 'Old answer')])
        self.assertIsNone(audio._fallback_worker)
        self.assertFalse(fallback.playback_started.is_set())

    def test_replacement_does_not_open_or_stop_a_reviewed_microphone(self):
        audio, fallback, created = self.create()
        audio.start_once()
        with self.assertRaises(AppError):
            audio.start('speaking', 'Wait for dictation')
        self.assertEqual(fallback.mode, 'listening')
        self.assertEqual(fallback.stop_count, 0)
        self.assertEqual(created, [])

    def test_pending_replacement_is_busy_for_new_microphone_capture(self):
        audio, _, gate, created = self.replacement_audio()
        audio.start('speaking', 'Latest answer')
        with self.assertRaises(AppError):
            audio.start_once()
        with self.assertRaises(AppError):
            audio.start('listening')
        self.assertEqual(created, [])
        gate.set()

    def test_queued_playback_failure_is_reported_without_starting_microphone(self):
        audio, fallback, gate, created = self.replacement_audio()
        audio.start('speaking', 'Latest answer')
        worker = audio._fallback_worker
        with patch.object(fallback, 'start', side_effect=AppError('Engine unavailable')):
            gate.set()
            worker.join(1)
        self.assertEqual(audio.events.get_nowait(), ('playback_error', 'Engine unavailable'))
        self.assertIsNone(audio._fallback_pending)
        self.assertIsNone(fallback.mode)
        self.assertEqual(created, [])

    def test_failed_cleanup_does_not_restart_a_queued_answer(self):
        audio, fallback, gate, created = self.replacement_audio()
        audio.start('speaking', 'Must not overlap the old answer')
        worker = audio._fallback_worker
        fallback.cleanup_failed = True
        gate.set()
        worker.join(1)
        self.assertEqual(fallback.started, [('speaking', 'Old answer')])
        self.assertIsNone(audio._fallback_pending)
        self.assertIsNone(audio._fallback_worker)
        self.assertFalse(fallback.playback_started.is_set())
        self.assertEqual(created, [])

    def test_answer_waits_for_native_teardown_before_text_playback(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        audio, fallback, created = self.create(close_gate=gate)
        bridge = self.ready(audio, created)
        audio.stop()
        self.assertTrue(bridge.close_entered.wait(1))
        audio.start('speaking', 'Answer after voice off')
        self.assertEqual(audio.mode, 'speaking')
        self.assertEqual(fallback.started, [])
        worker = audio._fallback_worker
        gate.set()
        worker.join(1)
        self.assertEqual(fallback.started, [('speaking', 'Answer after voice off')])
        self.assertEqual(len(created), 1)
        self.assertTrue(audio.events.empty())

    def test_recovering_recognition_is_visible_without_stopping_capture(self):
        audio, fallback, created = self.create()
        bridge = self.ready(audio, created)
        bridge.callback({'type': 'error', 'scope': 'recognition', 'code': 8,
                         'recovering': True, 'silence': False})
        self.assertEqual(audio.capabilities['recognition_state'], 'recovering')
        self.assertEqual(audio.capabilities['recognition_restarts'], 1)
        self.assertEqual(audio.capabilities['recognition_error_code'], 8)
        self.assertTrue(audio.listening)
        self.assertFalse(bridge.closed.is_set())
        self.assertEqual(fallback.stop_count, 0)
        self.assertTrue(audio.events.empty())
        bridge.callback({'type': 'ready', 'mic_active': True})
        self.assertEqual(audio.capabilities['recognition_state'], 'ready')
        self.assertNotIn('recognition_error_code', audio.capabilities)

    def test_passive_heartbeat_refreshes_capture_metrics_without_restarting(self):
        audio, fallback, created = self.create()
        bridge = self.ready(audio, created)
        bridge.callback({'type': 'pong', 'mic_active': True, 'frames': 32000, 'dropped_frames': 640})
        self.assertEqual(audio.capabilities['frames'], 32000)
        self.assertEqual(audio.capabilities['dropped_frames'], 640)
        self.assertTrue(audio.capabilities['status_fresh'])
        self.assertTrue(audio.listening)
        self.assertEqual(len(created), 1)
        self.assertEqual(fallback.stop_count, 0)
        bridge.callback({'type': 'pong', 'frames': -1, 'dropped_frames': 'invalid'})
        self.assertEqual(audio.capabilities['frames'], 32000)
        self.assertEqual(audio.capabilities['dropped_frames'], 640)

    def test_continuous_and_reviewed_capture_cannot_overlap(self):
        audio, _, created = self.create()
        self.ready(audio, created)
        with self.assertRaises(AppError):
            audio.start_once()

    def test_barge_in_stops_playback_and_keeps_microphone(self):
        audio, _, created = self.create()
        bridge = self.ready(audio, created)
        audio.start('speaking', 'An answer')
        identifier = audio.playback_id
        self.assertEqual(audio.mode, 'listening and speaking')
        bridge.callback({'type': 'speech_start', 'during_playback': True})
        self.assertEqual(bridge.interruptions, 1)
        self.assertTrue(audio.listening)
        self.assertTrue(audio.user_speaking)
        self.assertEqual(audio.mode, 'listening')
        self.assertFalse(bridge.closed.is_set())
        bridge.callback({'type': 'tts_start', 'id': identifier})
        bridge.callback({'type': 'tts_done', 'id': identifier})
        self.assertFalse(audio.speaking)
        bridge.callback({'type': 'final', 'text': 'My next request'})
        self.assertFalse(audio.user_speaking)
        self.assertIn(('heard', 'My next request'), list(audio.events.queue))

    def test_partial_barge_in_without_begin_callback(self):
        audio, _, created = self.create()
        bridge = self.ready(audio, created)
        audio.start('speaking', 'An answer')
        bridge.callback({'type': 'partial', 'text': 'Wait'})
        self.assertTrue(audio.user_speaking)
        self.assertFalse(audio.speaking)
        self.assertEqual(bridge.interruptions, 1)
        self.assertTrue(audio.listening)

    def test_late_old_playback_events_do_not_end_new_answer(self):
        audio, _, created = self.create()
        bridge = self.ready(audio, created)
        audio.start('speaking', 'First')
        previous = audio.playback_id
        audio.start('speaking', 'Second')
        latest = audio.playback_id
        bridge.callback({'type': 'tts_done', 'id': previous})
        self.assertTrue(audio.speaking)
        self.assertEqual(audio.playback_id, latest)
        bridge.callback({'type': 'tts_done', 'id': latest})
        self.assertFalse(audio.speaking)
        self.assertTrue(audio.listening)

    def test_full_long_answer_is_spoken_sequentially_until_its_last_chunk(self):
        audio, _, created = self.create()
        bridge = self.ready(audio, created)
        answer = 'A complete sentence with details. ' * 600 + 'Last sentence.'
        audio.start('speaking', answer)
        self.assertEqual(len(bridge.spoken), 1)
        previous_ids = set()
        while audio.speaking:
            identifier = audio.playback_id
            self.assertNotIn(identifier, previous_ids)
            previous_ids.add(identifier)
            count = len(bridge.spoken)
            bridge.callback({'type': 'tts_start', 'id': identifier})
            self.assertTrue(audio.speaking)
            bridge.callback({'type': 'tts_done', 'id': identifier})
            self.assertEqual(len(bridge.spoken), count + int(audio.speaking))
        self.assertEqual(''.join(text for text, _ in bridge.spoken), answer)
        self.assertGreater(len(bridge.spoken), 1)
        self.assertIsNone(audio.playback_id)
        self.assertTrue(audio.listening)
        self.assertTrue(audio.events.empty())

    def test_barge_in_discards_all_remaining_answer_chunks(self):
        audio, _, created = self.create()
        bridge = self.ready(audio, created)
        audio.start('speaking', 'Long answer. ' * 1000)
        identifier = audio.playback_id
        bridge.callback({'type': 'partial', 'text': 'Wait'})
        bridge.callback({'type': 'tts_done', 'id': identifier})
        self.assertEqual(len(bridge.spoken), 1)
        self.assertEqual(bridge.interruptions, 1)
        self.assertFalse(audio.speaking)
        self.assertFalse(audio._pending_chunks)
        self.assertTrue(audio.listening)

    def test_replacement_and_late_events_cannot_resume_previous_answer(self):
        audio, _, created = self.create()
        bridge = self.ready(audio, created)
        audio.start('speaking', 'Obsolete answer. ' * 500)
        obsolete = audio.playback_id
        answer = 'Replacement answer. ' * 500
        audio.start('speaking', answer)
        replacement = audio.playback_id
        bridge.callback({'type': 'tts_done', 'id': obsolete})
        bridge.callback({'type': 'error', 'scope': 'tts', 'id': obsolete,
                         'message': 'Late old error'})
        self.assertEqual(audio.playback_id, replacement)
        self.assertEqual(len(bridge.spoken), 2)
        self.assertTrue(audio.events.empty())
        while audio.speaking:
            bridge.callback({'type': 'tts_done', 'id': audio.playback_id})
        self.assertEqual(''.join(text for text, _ in bridge.spoken[1:]), answer)

    def test_duplicate_done_does_not_skip_a_chunk(self):
        audio, _, created = self.create()
        bridge = self.ready(audio, created)
        audio.start('speaking', 'Long answer. ' * 1000)
        identifier = audio.playback_id
        bridge.callback({'type': 'tts_done', 'id': identifier})
        current = audio.playback_id
        bridge.callback({'type': 'tts_done', 'id': identifier})
        self.assertEqual(audio.playback_id, current)
        self.assertEqual(len(bridge.spoken), 2)

    def test_interrupted_or_failed_utterance_discards_remaining_chunks(self):
        for event in ({'type': 'tts_done', 'interrupted': True},
                      {'type': 'error', 'scope': 'tts', 'message': 'Playback failed'}):
            with self.subTest(event=event):
                audio, _, created = self.create()
                bridge = self.ready(audio, created)
                audio.start('speaking', 'Long answer. ' * 1000)
                identifier = audio.playback_id
                bridge.callback(dict(event, id=identifier))
                bridge.callback({'type': 'tts_done', 'id': identifier})
                self.assertFalse(audio.speaking)
                self.assertFalse(audio._pending_chunks)
                self.assertEqual(len(bridge.spoken), 1)
                self.assertTrue(audio.listening)
                if event['type'] == 'error':
                    self.assertEqual(audio.events.get_nowait(), ('playback_error', 'Playback failed'))

    def test_transport_failure_discards_remaining_chunks(self):
        audio, _, created = self.create()
        bridge = self.ready(audio, created)
        with patch.object(bridge, 'speak', side_effect=NativeVoiceError('Disconnected')):
            audio.start('speaking', 'Long answer. ' * 1000)
        self.assertFalse(audio.speaking)
        self.assertFalse(audio._pending_chunks)
        self.assertEqual(audio.events.get_nowait(), ('playback_error', 'Disconnected'))
        self.assertTrue(audio.listening)

    def test_whitespace_only_chunks_do_not_block_later_content(self):
        audio, _, created = self.create()
        bridge = self.ready(audio, created)
        audio.start('speaking', 'Beginning.' + ' ' * 10000 + 'Final sentence.')
        while audio.speaking:
            bridge.callback({'type': 'tts_done', 'id': audio.playback_id})
        self.assertTrue(all(text.strip() for text, _ in bridge.spoken))
        self.assertTrue(bridge.spoken[-1][0].endswith('Final sentence.'))

    def test_stop_during_start_discards_pending_answer_and_all_chunks(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        audio, _, created = self.create(start_gate=gate)
        audio.start('listening')
        bridge = created[0]
        self.assertTrue(bridge.started.wait(1))
        audio.start('speaking', 'Long answer. ' * 1000)
        audio.stop()
        gate.set()
        audio.close()
        self.assertEqual(bridge.spoken, [])
        self.assertIsNone(audio._pending_speech)
        self.assertFalse(audio._pending_chunks)

    def test_playback_during_start_waits_for_native_ready(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        audio, fallback, created = self.create(start_gate=gate)
        audio.start('listening')
        bridge = created[0]
        self.assertTrue(bridge.started.wait(1))
        audio.start('speaking', 'Queued answer')
        self.assertEqual(bridge.spoken, [])
        self.assertEqual(fallback.started, [])
        gate.set()
        self.assertTrue(bridge.ready.wait(1))
        self.assertEqual(bridge.spoken[0][0], 'Queued answer')

    def test_close_waits_for_native_teardown(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        audio, fallback, created = self.create(close_gate=gate)
        bridge = self.ready(audio, created)
        done = threading.Event()
        closer = threading.Thread(target=lambda: (audio.close(), done.set()))
        closer.start()
        self.assertTrue(bridge.close_entered.wait(1))
        self.assertFalse(done.is_set())
        self.assertFalse(fallback.closed)
        self.assertFalse(audio.capabilities['status_fresh'])
        self.assertNotIn('mic_active', audio.capabilities)
        self.assertEqual(audio.capabilities['capture_state'], 'stopping')
        gate.set()
        self.assertTrue(done.wait(1))
        closer.join(1)
        self.assertTrue(fallback.closed)
        self.assertTrue(bridge.closed.is_set())
        self.assertFalse(audio.capabilities['mic_active'])
        self.assertTrue(audio.capabilities['status_fresh'])
        self.assertEqual(audio.capabilities['capture_state'], 'stopped')

    def test_close_waits_for_inflight_launcher_and_ignores_its_events(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        audio, _, created = self.create(start_gate=gate)
        audio.start('listening')
        bridge = created[0]
        self.assertTrue(bridge.started.wait(1))
        done = threading.Event()
        closer = threading.Thread(target=lambda: (audio.close(), done.set()))
        closer.start()
        self.assertTrue(bridge.close_entered.wait(1))
        self.assertFalse(done.is_set())
        bridge.callback({'type': 'final', 'text': 'stale private words'})
        gate.set()
        self.assertTrue(done.wait(1))
        closer.join(1)
        self.assertFalse(audio.listening)
        self.assertTrue(audio.events.empty())

    def test_repeated_stop_does_not_clear_stopping_early(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        audio, _, created = self.create(close_gate=gate)
        bridge = self.ready(audio, created)
        audio.stop()
        self.assertTrue(bridge.close_entered.wait(1))
        audio.stop()
        self.assertTrue(audio.stopping)
        with self.assertRaises(AppError):
            audio.start('listening')
        gate.set()
        audio.close()
        self.assertFalse(audio.stopping)

    def test_start_failure_closes_bridge_and_reports_once(self):
        audio, _, created = self.create(failure=NativeVoiceError('recognizer unavailable'))
        audio.start('listening')
        self.assertEqual(audio.events.get(timeout=1), ('error', 'recognizer unavailable'))
        audio.close()
        self.assertTrue(created[0].closed.is_set())
        self.assertFalse(audio.listening)
        self.assertTrue(audio.events.empty())

    def test_native_failure_does_not_fall_back_to_half_duplex(self):
        audio, fallback, created = self.create()
        bridge = self.ready(audio, created)
        bridge.callback({'type': 'error', 'scope': 'capture', 'fatal': True, 'message': 'Microphone blocked'})
        audio.close()
        self.assertEqual(fallback.started, [])
        self.assertTrue(bridge.closed.is_set())
        self.assertFalse(audio.listening)


if __name__ == '__main__':
    unittest.main()
