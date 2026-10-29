"""Continuous UI conversation using a fake recorder, real inbox, and offline RPC."""
import contextlib
import io
import os
import queue
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import launch
from terminal import TerminalPrompt, TerminalSession, TerminalUI
from agent import SteerUnavailable
from test_app import FakeRPC, eventually


class DuplexRPC(FakeRPC):
    def call(self, method, params, **kwargs):
        if method == 'turn/steer':
            self.calls.append((method, params))
            return {'turnId': params['expectedTurnId']}
        return super().call(method, params, **kwargs)


class Recorder:
    duplex = True

    def __init__(self):
        self.events = queue.Queue()
        self.listening = False
        self.speaking = False
        self.user_speaking = False
        self.stopping = False
        self.capabilities = {'mic_active': True, 'aec_enabled': True, 'mic_silenced': False}
        self.calls = []
        self.once = False
        self.microphone_muted = False
        self.microphone_muted_confirmed = False

    @property
    def mode(self):
        if self.listening and self.speaking:
            return 'listening and speaking'
        if self.listening:
            return 'listening'
        return 'speaking' if self.speaking else None

    def start(self, mode, text=None):
        self.calls.append((mode, text))
        if mode == 'listening':
            self.listening = not self.microphone_muted
        elif mode == 'speaking':
            self.speaking = True

    def start_once(self):
        self.calls.append(('once', None))
        self.once = self.listening = True

    def interrupt_speech(self):
        self.calls.append(('interrupt_speech', None))
        self.speaking = False

    def set_microphone_muted(self, muted):
        self.calls.append(('microphone_muted', muted))
        self.microphone_muted = self.microphone_muted_confirmed = muted
        self.listening = not muted
        return {'microphone_muted': muted, 'confirmed': True, 'mic_active': self.listening}

    def stop(self):
        self.calls.append(('stop', None))
        self.listening = self.speaking = self.user_speaking = self.once = False

    def close(self):
        self.stop()

    def heard(self, text):
        if self.once:
            self.once = self.listening = False
        self.user_speaking = False
        self.events.put(('heard', text))


class ContinuousTerminalTests(unittest.TestCase):
    def test_cpu_control_reads_hardware_without_submitting_a_model_request(self):
        rpc = self.active()
        model_calls_before = list(rpc.calls)
        measured = {'phone_hardware': True, 'read_only': True, 'device_model': 'Pixel 10 Pro',
                    'soc': {'manufacturer': 'Google', 'model': 'Tensor G5'}, 'abi': ['arm64-v8a'],
                    'core_count': 8, 'present_cpus': list(range(8)), 'online_cpus': list(range(8)),
                    'clusters': [], 'cpu_identifiers': [], 'sources': {}, 'unavailable': {}, 'status': 'ok'}
        with patch.object(self.session.agent.phone, 'execute', return_value=measured) as read, \
                patch.object(self.session.agent, 'submit', side_effect=AssertionError('No model request')):
            self.ui.handle('/cpu')
            self.pump_until(lambda: any('Tensor G5' in line for line in self.output))
        self.assertEqual(read.call_args.args[2], ['@phone', 'cpu'])
        self.assertIn('Tensor G5', self.ui.last_answer)
        self.assertIn('8', self.ui.last_answer)
        self.assertTrue(self.session.agent.active)
        self.assertEqual(rpc.calls, model_calls_before)
        self.assertEqual(self.session.agent.approvals, {})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'workspace'; self.root.mkdir()
        command = patch('agent.codex_command', return_value=['offline-protocol-fixture'])
        command.start(); self.addCleanup(command.stop)
        self.session = TerminalSession(self.root, self.base / 'state', DuplexRPC)
        self.speech = Recorder()
        self.output = []
        self.ui = TerminalUI(self.session, self.speech, self.output.append)
        self.ui.local = None
        self.ui.companion_next_poll = float('inf')
        self.display = io.StringIO()
        self.ui.prompt = self.prompt = TerminalPrompt(self.display)

    def tearDown(self):
        self.speech.close()
        self.session.close()
        self.temp.cleanup()

    def active(self):
        self.session.agent.submit(self.session.sid, 'Original question', 'duplex-start-12345')
        eventually(lambda: self.session.store.session(self.session.sid)['turn_id'])
        return self.session.agent.rpc

    def pump_until(self, check):
        def tick():
            self.ui.pump()
            return check()
        eventually(tick, timeout=8)

    @staticmethod
    def steers(rpc):
        return [params for method, params in rpc.calls if method == 'turn/steer']

    def test_capture_stays_open_while_agent_runs_tts_plays_and_user_types(self):
        self.ui.set_voice(True)
        self.active()
        self.ui.speak('Answer playing while the microphone stays open.')
        self.prompt.feed(b'a typed follow-up')
        self.ui.pump()
        self.assertTrue(self.speech.listening)
        self.assertTrue(self.speech.speaking)
        self.assertEqual(self.speech.mode, 'listening and speaking')
        self.assertNotIn(('stop', None), self.speech.calls)
        self.assertEqual(self.prompt.text, 'a typed follow-up')

    def test_playback_failure_keeps_continuous_microphone_enabled(self):
        self.ui.set_voice(True)
        self.speech.events.put(('playback_error', 'Speech engine unavailable'))
        self.ui.pump()
        self.assertTrue(self.ui.handsfree)
        self.assertTrue(self.speech.listening)
        self.assertNotIn(('stop', None), self.speech.calls)
        self.assertIn('Speech engine unavailable', '\n'.join(self.output))

    def test_microphone_mute_keeps_ai_speech_and_filters_late_transcripts(self):
        self.ui.set_voice(True)
        self.ui.speak('Keep reading this answer.')
        self.ui.handle('/mic mute')
        self.speech.events.put(('partial', 'late private words'))
        self.speech.events.put(('heard', 'late private words'))
        self.ui.pump()
        self.assertTrue(self.ui.audio)
        self.assertTrue(self.ui.handsfree)
        self.assertTrue(self.speech.speaking)
        self.assertFalse(self.speech.listening)
        self.assertEqual(self.session.store.setting('terminal_microphone_muted'), 'on')
        self.assertEqual(self.ui.inbox.counts()['queued'], 0)
        self.assertFalse(self.session.agent.active)
        self.assertNotIn(('stop', None), self.speech.calls)
        self.assertNotIn(('interrupt_speech', None), self.speech.calls)
        self.assertIn('mic muted', self.ui.activity_badge(self.ui.speech_counts()))

    def test_down_up_gesture_toggles_microphone_but_not_spoken_replies(self):
        from types import SimpleNamespace
        from volume_hotkey import VolumeTapSequence
        self.ui.set_voice(True)
        self.ui.speak('An answer is playing.')
        self.ui.volume_hotkey = SimpleNamespace(events=queue.Queue(), status='ready')
        detector = VolumeTapSequence()
        for base in (1.0, 2.0):
            for delta, code, value in ((0,114,1),(.1,114,0),(.2,115,1),(.3,115,0)):
                if detector.feed(code, value, base + delta):
                    self.ui.volume_hotkey.events.put(('toggle', None))
            self.ui.pump()
            self.assertTrue(self.ui.audio)
            self.assertTrue(self.speech.speaking)
            self.assertEqual(self.ui.microphone_muted, base == 1.0)
        self.assertTrue(self.speech.listening)
        self.assertEqual(self.session.store.setting('terminal_microphone_muted'), 'off')

    def test_muted_voice_does_not_automatically_reopen_capture(self):
        self.ui.set_voice(True)
        self.ui.handle('/mic mute')
        before = list(self.speech.calls)
        for _ in range(3):
            self.ui.pump_voice()
        self.assertEqual(self.speech.calls, before)
        self.ui.handle('/mic unmute')
        self.assertTrue(self.speech.listening)
        self.assertTrue(self.ui.handsfree)

    def test_gesture_mutes_reviewed_one_shot_capture_in_text_mode(self):
        from duplex_speech import DuplexSpeech
        from test_duplex_speech import FakeFallback
        fallback = FakeFallback()
        audio = DuplexSpeech(fallback)
        self.addCleanup(audio.close)
        self.ui.speech = audio
        audio.start_once()
        self.assertEqual(fallback.mode, 'listening')
        self.assertFalse(self.ui.handsfree)
        self.ui.toggle_microphone()
        self.assertTrue(self.ui.microphone_muted)
        self.assertIsNone(fallback.mode)
        self.assertFalse(self.ui.handsfree)
        self.assertTrue(self.ui.audio)

    def test_model_voice_mute_control_keeps_ai_output_and_task_state(self):
        self.ui.set_voice(True)
        self.ui.speak('AI output continues.')
        status = self.ui.voice_control('mute')
        self.assertTrue(status['microphone_muted'])
        self.assertTrue(status['automatic_speech'])
        self.assertTrue(self.speech.speaking)
        self.assertFalse(self.speech.listening)
        self.assertFalse(self.session.agent.active)
        status = self.ui.voice_control('unmute')
        self.assertFalse(status['microphone_muted'])
        self.assertTrue(self.speech.speaking)
        self.assertTrue(self.speech.listening)

    def test_saved_microphone_mute_is_applied_before_voice_start(self):
        self.session.store.setting('terminal_microphone_muted', 'on')
        recorder = Recorder()
        ui = TerminalUI(self.session, recorder, self.output.append, voice=True)
        self.assertTrue(ui.microphone_muted)
        self.assertEqual(recorder.calls, [('microphone_muted', True)])
        ui.set_voice(True, preserve_mute=True)
        self.assertFalse(recorder.listening)
        self.assertTrue(ui.handsfree)
        ui.handle('/voice on')
        self.assertFalse(ui.microphone_muted)
        self.assertTrue(recorder.listening)

    def test_completed_long_response_reaches_android_in_full_through_real_adapter(self):
        from duplex_speech import DuplexSpeech
        from terminal import spoken_text
        from test_duplex_speech import FakeBridge, FakeFallback
        created = []
        def factory(callback):
            bridge = FakeBridge(callback)
            created.append(bridge)
            return bridge
        audio = DuplexSpeech(FakeFallback(), factory)
        self.addCleanup(audio.close)
        self.ui.speech = audio
        self.ui.set_voice(True)
        self.assertTrue(created[0].ready.wait(1))
        self.active()
        answer = 'Opening sentence.\n\n' + ('Full explanation 中文 😀. ' * 350) + '\n\nUse `get_value`.\n\nFinal sentence.'
        self.session.agent.on_event({'method': 'item/completed', 'params': {
            'item': {'type': 'agentMessage', 'text': answer}}})
        self.session.agent.on_event({'method': 'turn/completed', 'params': {'turn': {'status': 'completed'}}})
        self.ui.pump()
        bridge = created[0]
        self.assertTrue(audio.speaking)
        while audio.speaking:
            text, identifier = bridge.spoken[-1]
            self.assertLessEqual(len(text.encode('utf-16-le')) // 2, 3000)
            bridge.callback({'type': 'tts_done', 'id': identifier})
        self.assertEqual(''.join(text for text, _ in bridge.spoken), spoken_text(answer))
        self.assertTrue(audio.listening)

    def test_saved_speech_startup_is_compact_and_draft_review_never_sends(self):
        text = 'Long saved speech fragment. ' * 150
        self.ui.inbox.add(text)
        self.ui.inbox.pause_pending()
        self.ui.restore()
        self.assertNotIn(text, '\n'.join(self.output))
        self.assertIn('/draft', '\n'.join(self.output))
        self.ui.handle('/draft')
        self.assertIn(text, '\n'.join(self.output))
        self.assertEqual(self.ui.inbox.review_text(), text)
        self.assertFalse(self.session.agent.active)

    def test_32_review_question_is_answered_locally_without_sending_saved_words(self):
        for i in range(32):
            self.ui.inbox.add('private saved fragment ' + str(i))
        self.ui.inbox.pause_pending()
        self.ui.handle('what does 32 review mean')
        shown = '\n'.join(self.output)
        self.assertIn('saved microphone phrases', shown)
        self.assertIn('32', shown)
        self.assertIn('/draft', shown)
        self.assertNotIn('private saved fragment', shown)
        self.assertFalse(self.session.agent.active)
        self.assertEqual(self.ui.inbox.counts()['review'], 32)
        self.ui.handle('/review')
        self.assertEqual(self.ui.inbox.counts()['review'], 32)

    def test_send_of_32_phrases_to_unavailable_model_preserves_every_word(self):
        self.session.agent.engine_label = 'Selected Tensor NPU'
        self.session.agent.local_client = SimpleNamespace(engine_name='local-npu', status=lambda: {
            'ready': False, 'selection_state': 'unavailable', 'error': 'Selected NPU is not ready'})
        self.ui.local_status = {'ready': True, 'model': {'name': 'CPU model', 'backend': 'llama.cpp'}}
        for i in range(32):
            self.ui.inbox.add('private saved fragment ' + str(i))
        self.ui.inbox.pause_pending()
        self.ui.handle('/send')
        self.pump_until(lambda: self.ui.inbox.counts()['review'] == 32)
        self.ui.pump()
        self.assertIsNone(self.session.agent.rpc)
        self.assertFalse(self.session.agent.active)
        self.assertEqual(self.ui.inbox.review_text(), '\n'.join('private saved fragment ' + str(i) for i in range(32)))
        self.assertIn('engine unavailable', self.prompt.local)
        self.assertIn('CPU helper ready', self.prompt.local)
        self.ui.handle('/controls')
        shown = '\n'.join(self.output)
        self.assertIn('Engine: Selected Tensor NPU · unavailable', shown)
        self.assertNotIn('You: private saved fragment', shown)

    def test_failed_task_pauses_remaining_32_phrases_before_dispatch(self):
        self.ui.set_voice(True)
        rpc = self.active()
        for i in range(32):
            self.ui.inbox.add('queued fragment ' + str(i))
        self.session.agent.finish('failed', 'The selected model is not ready')
        self.ui.pump()
        self.assertEqual(self.ui.inbox.counts(), {'queued': 0, 'sending': 0, 'review': 32})
        self.assertEqual(len([method for method, _ in rpc.calls if method == 'turn/start']), 1)
        self.assertEqual(self.steers(rpc), [])
        self.assertTrue(self.speech.listening)
        self.assertTrue(self.ui.handsfree)
        self.assertIn('engine failed', self.prompt.local)

    def test_helper_readiness_does_not_clear_selected_engine_failure(self):
        self.active()
        self.session.agent.finish('failed', 'Selected NPU stopped during the request')
        self.ui.pump()
        self.ui.local_status = {'ready': True, 'model': {'name': 'CPU model'}}
        self.ui.npu_status = {'ready': True, 'model': {'backend': 'litert-npu'}}
        client = getattr(self.session.agent, 'local_client', None)
        self.ui.observe_engine(self.session.agent, client, {'ready': True})
        self.ui.handle('/controls')
        shown = '\n'.join(self.output)
        self.assertIn('last request failed: Selected NPU stopped during the request', shown)
        self.assertIn('CPU helper:', shown)
        self.assertIn('NPU helper:', shown)
        self.assertEqual(self.session.agent.interface.read_context(self.session.sid)['engine_health']['state'],
                         'last request failed')

    def test_live_context_reports_review_and_independent_microphone_speaker_state(self):
        self.ui.inbox.add('saved words')
        self.ui.inbox.pause_pending()
        self.ui.set_voice(True)
        self.ui.handle('/mic mute')
        context = self.session.agent.interface.read_context(self.session.sid)
        self.assertTrue(context['available'])
        self.assertEqual(context['speech_queue']['review'], 1)
        self.assertTrue(context['microphone_muted'])
        self.assertTrue(context['automatic_speech'])
        self.assertEqual(context['commands']['read_saved'], '/draft')
        self.assertNotIn('saved words', str(context))

    def test_active_task_elapsed_and_muted_microphone_state_are_visible_together(self):
        self.ui.set_voice(True)
        self.active()
        self.ui.audio = False
        self.ui.pump()
        self.assertEqual(self.prompt.status, 'thinking')
        self.ui.task_clock = time.monotonic() - 12.2
        self.ui.pump()
        self.assertIn('Thinking 12s', self.prompt.frame())
        self.assertIn('mic on', self.prompt.frame())
        self.assertIn('muted', self.prompt.frame())
        self.prompt.feed(b'draft input')
        with patch('terminal.shutil.get_terminal_size', return_value=os.terminal_size((40, 20))):
            frame = self.prompt.frame()
            shown = frame.split('\x1b[2K', 1)[1].split('\r', 1)[0]
            self.assertLess(len(shown), 40)
        self.assertEqual(self.prompt.text, 'draft input')
        self.assertTrue(self.speech.listening)

    def test_captured_followup_is_visible_while_queued_and_clears_after_delivery(self):
        self.ui.set_voice(True)
        self.active()
        self.prompt.feed(b'unsent typing')
        with patch.object(self.session.agent, 'steer', side_effect=SteerUnavailable('Busy', 'starting')):
            self.speech.heard('captured \x1b[31mwords')
            self.ui.pump()
            self.assertIn('Saved speech', self.prompt.transcript)
            self.assertIn('captured', self.prompt.transcript)
            self.assertNotIn('\x1b', self.prompt.transcript)
            counts = self.ui.inbox.counts()
            self.assertEqual(counts['queued'] + counts['sending'], 1)
            self.assertIn('1 queued', self.prompt.local)
        self.pump_until(lambda: not self.ui.inbox.pending)
        self.assertEqual(self.prompt.transcript, '')
        self.assertEqual(self.prompt.text, 'unsent typing')
        self.assertTrue(self.speech.listening)

    def test_repeated_queue_notices_do_not_fill_the_conversation(self):
        self.ui.set_voice(True)
        self.active()
        self.output.clear()
        for _ in range(3):
            self.session.store.event(self.session.sid, 'notice', {'text': 'Follow-up speech is saved in order.'})
        self.ui.pump()
        self.assertEqual(self.output.count('Speech saved for the next reply.'), 1)

    def test_recovering_and_review_status_are_visible_without_changing_audio(self):
        self.ui.set_voice(True)
        self.ui.inbox.add('saved words')
        self.ui.inbox.pause_pending()
        self.speech.capabilities.update(recognition_state='recovering', recognition_restarts=3,
                                        recognition_silence=False, dropped_frames=640)
        self.ui.pump()
        self.assertIn('1 review', self.prompt.local)
        self.assertIn('mic retrying', self.prompt.local)
        self.ui.handle('/audio')
        self.assertIn('Recognition: recovering · retries 3 · dropped samples 640', '\n'.join(self.output))
        self.assertTrue(self.speech.listening)

    def test_muted_voice_banner_shows_how_to_restore_spoken_replies(self):
        self.ui.start_voice_requested = True
        self.ui.audio = False
        self.ui.banner()
        self.assertIn('speaker off: /unmute', '\n'.join(self.output))
        self.assertFalse(self.ui.audio)
        self.assertEqual(self.speech.calls, [])

    def test_live_partial_and_answer_preview_use_independent_rows(self):
        self.ui.set_voice(True)
        self.active()
        self.session.store.event(self.session.sid, 'assistant_delta',
                                 {'item_id': 'answer-1', 'text': 'The answer is arriving'})
        self.speech.events.put(('partial', 'Wait, another detail'))
        self.ui.pump()
        self.assertEqual(self.prompt.preview, 'Ostadix · The answer is arriving')
        self.assertEqual(self.prompt.transcript, 'You · Wait, another detail')
        self.prompt.visible = True
        self.prompt.render()
        rendered = self.display.getvalue()
        self.assertIn('The answer is arriving', rendered)
        self.assertIn('Wait, another detail', rendered)
        self.assertEqual(self.prompt.rendered_rows, 2)

    def test_final_speech_during_task_steers_without_draft_or_mic_restart(self):
        self.ui.set_voice(True)
        rpc = self.active()
        self.speech.heard('Use the latest notes instead')
        self.pump_until(lambda: bool(self.steers(rpc)))
        self.assertEqual(self.steers(rpc)[0]['input'], [{'type': 'text', 'text': 'Use the latest notes instead'}])
        self.assertEqual(self.session.draft(), '')
        self.assertTrue(self.speech.listening)
        self.assertEqual(self.speech.calls.count(('listening', None)), 1)
        self.assertFalse(any('Wait for' in text or 'press Enter' in text for text in self.output))

    def test_typing_during_reply_interrupts_only_speech_and_steers_task(self):
        self.ui.set_voice(True)
        rpc = self.active()
        self.ui.speak('An answer to interrupt')
        self.ui.handle('Also include the battery reading')
        self.assertFalse(self.speech.speaking)
        self.assertTrue(self.speech.listening)
        self.pump_until(lambda: bool(self.steers(rpc)))
        self.assertEqual(self.steers(rpc)[0]['input'][0]['text'], 'Also include the battery reading')
        self.assertEqual(self.session.draft(), '')

    def test_mute_stops_playback_and_keeps_continuous_capture_enabled(self):
        self.ui.set_voice(True)
        self.ui.speak('Spoken answer')
        self.ui.handle('/mute')
        self.ui.pump()
        self.assertFalse(self.ui.audio)
        self.assertFalse(self.speech.speaking)
        self.assertTrue(self.ui.handsfree)
        self.assertTrue(self.speech.listening)
        self.assertEqual(self.session.store.setting('terminal_auto_audio'), 'off')
        self.assertNotIn(('stop', None), self.speech.calls)

    def test_stop_discards_already_queued_partial_and_final_capture(self):
        self.ui.set_voice(True)
        self.prompt.transcript = 'You · unfinished words'
        self.speech.events.put(('partial', 'late partial already queued'))
        self.speech.heard('late final already queued')
        self.ui.handle('/stop')
        self.ui.pump()
        self.assertFalse(self.ui.handsfree)
        self.assertFalse(self.speech.listening)
        self.assertEqual(self.prompt.transcript, '')
        self.assertEqual(self.session.draft(), '')
        self.assertFalse(self.ui.inbox.pending)
        self.assertFalse(self.session.agent.active)

    def test_manual_talk_uses_reviewed_one_shot_and_preserves_draft(self):
        self.ui.handle('/talk')
        self.assertFalse(self.ui.handsfree)
        self.assertIn(('once', None), self.speech.calls)
        self.assertNotIn(('listening', None), self.speech.calls)
        self.speech.heard('Review these words before sending')
        self.ui.pump()
        self.assertEqual(self.session.draft(), 'Review these words before sending')
        self.assertFalse(self.session.agent.active)
        self.assertFalse(self.speech.listening)

    def test_manual_talk_queued_during_task_uses_one_shot_after_reply(self):
        self.active()
        self.ui.handle('/talk')
        self.assertEqual(self.ui.pending_talk_sid, self.session.sid)
        self.assertNotIn(('once', None), self.speech.calls)
        self.ui.audio = False
        self.session.agent.finish('idle', 'Finished silently')
        self.ui.pump()
        self.assertIn(('once', None), self.speech.calls)
        self.assertFalse(self.ui.handsfree)

    def test_recognized_slash_commands_are_user_data_and_cannot_change_controls(self):
        self.ui.set_voice(True)
        rpc = self.active()
        self.speech.heard('/quit /approvals off /cancel')
        self.pump_until(lambda: bool(self.steers(rpc)))
        self.assertEqual(self.steers(rpc)[0]['input'][0]['text'], '/quit /approvals off /cancel')
        self.assertFalse(self.ui.quit)
        self.assertTrue(self.session.agent.active)
        self.assertEqual(self.session.agent.action_policy(), 'auto')
        self.assertTrue(self.speech.listening)

    def test_cancel_pauses_queued_speech_before_it_can_restart_work(self):
        self.ui.set_voice(True)
        rpc = self.active()
        self.ui.inbox.add('Queued words before cancellation')
        self.ui.handle('/cancel')
        self.pump_until(lambda: not self.session.agent.active)
        self.ui.pump()
        self.assertEqual(self.steers(rpc), [])
        self.assertIn('Queued words before cancellation', self.ui.inbox.review_text())
        self.assertFalse(self.session.agent.active)

    def test_new_answer_is_spoken_while_older_cancelled_input_stays_saved_for_review(self):
        self.ui.set_voice(True)
        self.active()
        self.ui.inbox.add('Older cancelled input')
        self.ui.handle('/cancel')
        self.pump_until(lambda: not self.session.agent.active)
        self.speech.heard('A new question after cancellation')
        self.pump_until(lambda: self.session.agent.active and
                        self.session.store.session(self.session.sid)['turn_id'])
        self.assertIn('Older cancelled input', self.ui.inbox.review_text())
        self.session.agent.finish('idle', 'Here is the answer to your new question.')
        self.pump_until(lambda: ('speaking', 'Here is the answer to your new question.') in self.speech.calls)
        self.assertIn(('speaking', 'Here is the answer to your new question.'), self.speech.calls)
        self.assertTrue(self.speech.listening)

    def test_fast_backend_reply_waits_for_speech_delivery_acknowledgement(self):
        expected = 'A fast reply from the normal backend pipeline.'

        class FastRPC(DuplexRPC):
            def call(self, method, params, **kwargs):
                result = super().call(method, params, **kwargs)
                if method == 'turn/start':
                    self.callback({'method': 'item/completed', 'params': {'threadId': 'thread-offline',
                                   'item': {'type': 'agentMessage', 'text': expected}}})
                    self.callback({'method': 'turn/completed', 'params': {'threadId': 'thread-offline',
                                   'turn': {'id': 'turn-offline', 'status': 'completed'}}})
                return result

        self.session.agent.rpc_factory = FastRPC
        self.ui.set_voice(True)
        receipt = threading.Event()
        self.addCleanup(receipt.set)
        deliver = self.ui.inbox._deliver

        def delayed_acknowledgement(*args):
            outcome = deliver(*args)
            receipt.wait(15)
            return outcome

        self.ui.inbox._deliver = delayed_acknowledgement
        self.speech.heard('A short question with a fast response')
        try:
            self.pump_until(lambda: any(event['kind'] == 'finished' and event['data'].get('text') == expected
                                       for event in self.session.store.events(self.session.sid)))
            self.assertTrue(self.ui.inbox.delivering)
            self.assertNotIn(('speaking', expected), self.speech.calls)
        finally:
            receipt.set()
        self.pump_until(lambda: ('speaking', expected) in self.speech.calls)
        self.assertEqual(self.speech.calls.count(('speaking', expected)), 1)

    def defer_reply(self):
        self.ui.set_voice(True)
        self.ui.inbox.add('Speech awaiting its delivery receipt')
        self.ui.last_answer = 'A deferred reply'
        self.ui.speak_completed_reply(self.ui.last_answer)
        self.assertIsNotNone(self.ui.pending_reply_audio)
        self.assertNotIn(('speaking', self.ui.last_answer), self.speech.calls)

    def test_stop_or_mute_does_not_resume_deferred_reply(self):
        for command in ('/stop', '/mute'):
            with self.subTest(command=command):
                self.ui.audio = True
                self.defer_reply()
                self.ui.handle(command)
                self.assertIsNone(self.ui.pending_reply_audio)
                self.ui.audio = True
                self.ui.pump_reply_audio()
                self.assertNotIn(('speaking', 'A deferred reply'), self.speech.calls)

    def test_new_task_generation_discards_deferred_reply(self):
        self.defer_reply()
        self.session.agent.task_generation += 1
        self.ui.pump_reply_audio()
        self.assertIsNone(self.ui.pending_reply_audio)
        self.assertNotIn(('speaking', 'A deferred reply'), self.speech.calls)

    def test_new_speech_discards_deferred_reply(self):
        self.defer_reply()
        self.speech.user_speaking = True
        self.ui.pump_reply_audio()
        self.assertIsNone(self.ui.pending_reply_audio)
        self.assertNotIn(('speaking', 'A deferred reply'), self.speech.calls)

    def test_task_failure_keeps_mic_open_without_replaying_failed_task(self):
        self.ui.set_voice(True)
        rpc = self.active()
        self.session.agent.finish('failed', 'Connection lost; the prior task may be incomplete.')
        self.ui.pump()
        self.assertTrue(self.ui.handsfree)
        self.assertTrue(self.speech.listening)
        self.assertFalse(self.session.agent.active)
        self.assertEqual(sum(method == 'turn/start' for method, _ in rpc.calls), 1)
        self.assertEqual(self.steers(rpc), [])


class DefaultVoiceLaunchTests(unittest.TestCase):
    def test_c_defaults_to_voice_and_explicit_text_or_no_audio_never_opens_mic(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for arguments, voice, no_audio in (([], True, False), (['--voice'], True, False),
                    (['--text'], False, False), (['--no-audio'], False, True)):
                with self.subTest(arguments=arguments), contextlib.ExitStack() as stack:
                    stack.enter_context(patch.object(launch.os, 'geteuid', return_value=10402))
                    stack.enter_context(patch.object(launch.sys.stdin, 'isatty', return_value=True))
                    stack.enter_context(patch.object(launch, 'saved_workspace', return_value=str(root)))
                    stack.enter_context(patch.object(launch, 'prepare_ostadix'))
                    stack.enter_context(patch.object(launch, 'running', return_value=False))
                    execute = stack.enter_context(patch.object(launch.os, 'execv'))
                    launch.main(['--no-local', '--state-dir', str(root / 'state'), *arguments])
                    argv = execute.call_args.args[1]
                    self.assertEqual('--voice' in argv, voice)
                    self.assertEqual('--no-audio' in argv, no_audio)
                    self.assertEqual(argv[:2], [sys.executable, str(launch.APP / 'terminal.py')])
