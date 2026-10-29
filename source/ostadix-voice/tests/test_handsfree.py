"""Hands-free lifecycle checks with simulated audio and offline Codex RPC."""
import io
from pathlib import Path
import queue
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import AppError
from terminal import NativeSpeech, TerminalPrompt, TerminalSession, TerminalUI
from test_app import FakeRPC, eventually


class Speech:
    def __init__(self):
        self.mode = None
        self.stopping = False
        self.events = queue.Queue()
        self.calls = []

    def start(self, mode, text=None):
        if self.mode or self.stopping:
            raise AppError('Audio busy')
        self.mode = mode
        self.calls.append((mode, text))

    def stop(self):
        self.mode = None
        self.calls.append(('stop', None))

    def complete(self, kind=None, text=''):
        self.mode = None
        if kind:
            self.events.put((kind, text))


class HandsfreeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.root = self.base / 'project'
        self.root.mkdir()
        self.codex = patch('agent.codex_command', return_value=['offline-fixture'])
        self.codex.start()
        self.session = TerminalSession(self.root, self.base / 'state', FakeRPC)
        self.speech = Speech()
        self.output = []
        self.ui = TerminalUI(self.session, self.speech, self.output.append)
        self.ui.local = None
        self.ui.restore()
        self.now = 100.0
        self.clock = patch('terminal.time.monotonic', side_effect=lambda: self.now)
        self.clock.start()

    def tearDown(self):
        self.clock.stop()
        self.session.close()
        self.codex.stop()
        self.tmp.cleanup()

    def heard(self, text):
        self.speech.complete('heard', text)
        self.ui.pump()

    def finish(self, text='Here is the answer.', status='idle'):
        eventually(lambda: self.session.store.session(self.session.sid)['turn_id'])
        self.session.agent.finish(status, text)
        self.ui.pump()

    def voice_tool(self, action):
        """A real Agent tool dispatch waits for the terminal's UI thread."""
        self.session.submit('Please change the voice controls.')
        eventually(lambda: self.session.store.session(self.session.sid)['turn_id'])
        agent = self.session.agent
        self.tool_thread = threading.Thread(target=agent._tool_request, args=({
            'id': 781, 'method': 'item/tool/call', 'params': {
                'threadId': 'thread-offline', 'turnId': 'turn-offline',
                'tool': 'workspace',
                'arguments': {'operation': 'command', 'argv': ['@voice', action]},
            }},))
        self.tool_thread.start()
        for _ in range(100):
            if agent.interface._pending:
                return
            threading.Event().wait(.01)
        self.fail('Voice tool did not reach the UI bridge')

    def tool_result(self):
        self.tool_thread.join(1)
        self.assertFalse(self.tool_thread.is_alive())
        return [event['data'] for event in self.session.store.events(self.session.sid)
                if event['kind'] == 'tool_result'][-1]

    def test_model_voice_on_acknowledges_then_waits_for_task_and_spoken_reply(self):
        self.voice_tool('on')
        self.assertFalse(self.ui.voice_enabled)
        self.ui.pump()
        result = self.tool_result()
        self.assertTrue(result['success'])
        state = result['result']
        self.assertTrue(state['available'])
        self.assertTrue(state['enabled'])
        self.assertTrue(state['task_active'])
        self.assertEqual(state['audio'], 'idle')
        self.assertEqual(self.speech.calls, [])
        self.finish('Hands-free is on.')
        self.assertEqual(self.speech.mode, 'speaking')
        self.speech.complete()
        self.ui.pump()
        self.assertIsNone(self.speech.mode)
        self.now += .66
        self.ui.pump()
        self.assertEqual(self.speech.mode, 'listening')

    def test_model_voice_on_does_not_discard_a_saved_draft(self):
        self.voice_tool('on')
        self.session.draft('Keep this unsubmitted request.')
        self.ui.pump()
        result = self.tool_result()
        self.assertFalse(result['success'])
        self.assertIn('/send or /discard', result['result']['error'])
        self.assertFalse(self.ui.voice_enabled)
        self.assertEqual(self.session.draft(), 'Keep this unsubmitted request.')
        self.assertEqual(self.speech.calls, [])

    def test_closing_terminal_rejects_a_pending_model_voice_on(self):
        self.voice_tool('on')
        self.ui.quit = True
        self.ui.pump()
        result = self.tool_result()
        self.assertFalse(result['success'])
        self.assertIn('closed', result['result']['error'])
        self.assertFalse(self.ui.voice_enabled)
        self.assertEqual(self.speech.calls, [])

    def test_two_turns_listen_submit_speak_listen_without_keyboard(self):
        self.ui.handle('/voice on')
        for request in ('First request', 'Second request'):
            self.assertEqual(self.speech.mode, 'listening')
            self.heard(request)
            self.assertTrue(self.session.agent.active)
            self.assertEqual(self.session.draft(), '')
            self.assertIsNone(self.speech.mode)
            self.finish()
            self.assertEqual(self.speech.mode, 'speaking')
            self.speech.complete()
            self.ui.pump()
            self.assertIsNone(self.speech.mode)
            self.now += .64
            self.ui.pump()
            self.assertIsNone(self.speech.mode)
            self.now += .02
            self.ui.pump()
        requests = [event['data']['text'] for event in self.session.store.events(self.session.sid)
                    if event['kind'] == 'user']
        self.assertEqual(requests, ['First request', 'Second request'])
        self.assertEqual([call[0] for call in self.speech.calls],
                         ['listening', 'speaking', 'listening', 'speaking', 'listening'])

    def test_no_microphone_during_task_tool_lock_approval_or_speaker(self):
        self.session.agent.active = True
        self.ui.handle('/voice on')
        self.assertEqual(self.speech.calls, [])
        self.session.agent.active = False
        self.session.agent.tool_lock.acquire()
        self.ui.pump()
        self.session.agent.tool_lock.release()
        self.assertEqual(self.speech.calls, [])
        with patch.object(self.session, 'approvals', return_value=[{'id': 'pending'}]):
            self.ui.pump()
        self.assertEqual(self.speech.calls, [])
        self.speech.mode = 'speaking'
        self.ui.pump()
        self.assertEqual(self.speech.calls, [])
        self.speech.mode = None
        self.ui.pump()
        self.assertEqual(self.speech.mode, 'listening')

    def test_stop_disables_loop_and_discards_already_queued_capture(self):
        self.ui.handle('/voice on')
        self.speech.complete('heard', 'This was queued before stop')
        self.ui.handle('/stop')
        self.now += 100
        self.ui.pump()
        self.assertFalse(self.ui.voice_enabled)
        self.assertFalse(self.session.agent.active)
        self.assertEqual(self.session.draft(), '')
        self.assertIsNone(self.speech.mode)

    def test_voice_off_mute_and_stop_phrase_prevent_restart(self):
        for stop in ('/voice off', '/mute', 'stop listening'):
            self.ui.handle('/voice on')
            if stop.startswith('/'):
                self.ui.handle(stop)
            else:
                self.heard(stop)
            self.now += 10
            self.ui.pump()
            self.assertFalse(self.ui.voice_enabled)
            self.assertFalse(self.session.agent.active)
            self.assertIsNone(self.speech.mode)

    def test_ctrl_c_disables_loop_before_next_input_cycle(self):
        self.ui.handle('/voice on')
        calls = []

        def pump():
            if not calls:
                calls.append(True)
                raise KeyboardInterrupt
            self.ui.quit = True

        with patch.object(self.ui, 'pump', side_effect=pump), \
                patch('terminal.select.select', return_value=([], [], [])):
            self.ui.run_loop()
        self.assertFalse(self.ui.voice_enabled)
        self.assertIsNone(self.speech.mode)

    def test_typed_input_survives_and_pauses_capture(self):
        self.ui.prompt = TerminalPrompt(io.StringIO())
        self.ui.handle('/voice on')
        self.ui.prompt.feed('Partially typed 🦙 request'.encode())
        self.ui.pump()
        self.assertIsNone(self.speech.mode)
        self.assertEqual(self.ui.prompt.text, 'Partially typed 🦙 request')
        self.now += 10
        self.ui.pump()
        self.assertIsNone(self.speech.mode)
        self.assertTrue(self.ui.voice_enabled)

    def test_heard_result_is_reviewed_if_user_started_typing(self):
        self.ui.handle('/voice on')
        self.ui.prompt = TerminalPrompt(io.StringIO())
        self.ui.prompt.feed(b'Typed draft')
        self.heard('Recognized draft')
        self.assertEqual(self.ui.prompt.text, 'Typed draft')
        self.assertEqual(self.session.draft(), 'Recognized draft')
        self.assertFalse(self.session.agent.active)

    def test_submission_error_preserves_recognized_draft_and_pauses(self):
        self.ui.handle('/voice on')
        with patch.object(self.session, 'submit', side_effect=AppError('Busy now')):
            self.heard('Keep this request')
        self.assertFalse(self.ui.voice_enabled)
        self.assertEqual(self.session.draft(), 'Keep this request')
        self.assertIn('Your speech is saved as a draft', '\n'.join(self.output))

    def test_audio_error_and_model_failure_pause_without_retry(self):
        self.ui.handle('/voice on')
        self.speech.complete('error', 'Microphone permission denied')
        self.ui.pump()
        self.assertFalse(self.ui.voice_enabled)
        self.now += 60
        self.ui.pump()
        self.assertIsNone(self.speech.mode)
        self.ui.handle('/voice on')
        self.heard('Hello')
        self.finish('Connection failed', status='failed')
        self.assertFalse(self.ui.voice_enabled)
        self.assertIsNone(self.speech.mode)

    def test_silence_keeps_voice_enabled_with_bounded_backoff_until_stopped(self):
        self.ui.handle('/voice on')
        for attempt in range(1, 6):
            self.speech.complete('empty', 'No speech was heard.')
            self.ui.pump()
            self.assertIsNone(self.speech.mode)
            self.ui.pump()
            self.assertIsNone(self.speech.mode)
            self.now += min(3, attempt * 1.5)
            self.ui.pump()
            self.assertEqual(self.speech.mode, 'listening')
        self.assertTrue(self.ui.voice_enabled)
        self.assertFalse(self.session.agent.active)
        self.assertEqual(len([call for call in self.speech.calls if call[0] == 'listening']), 6)
        self.assertIsNone(self.session.agent.rpc)
        self.ui.handle('/stop')
        self.now += 10
        self.ui.pump()
        self.assertFalse(self.ui.voice_enabled)
        self.assertIsNone(self.speech.mode)

    def test_recognition_remains_in_source_conversation_after_switch(self):
        source = self.session.sid
        self.ui.handle('/voice on')
        self.ui.handle('/new')
        target = self.session.sid
        self.session.draft('Destination draft')
        self.heard('Original conversation request')
        self.assertNotEqual(source, target)
        self.assertEqual(self.session.draft(), 'Destination draft')
        self.assertEqual(self.session.store.setting('terminal_draft_' + source), 'Original conversation request')
        self.assertFalse(self.session.agent.active)
        self.assertFalse(self.ui.voice_enabled)

    def test_recognized_approval_command_is_only_model_input(self):
        self.ui.handle('/voice on')
        with patch.object(self.session, 'approve') as approve:
            self.heard('/approve deadbeef')
        approve.assert_not_called()
        events = self.session.store.events(self.session.sid)
        self.assertTrue(any(event['kind'] == 'user' and event['data']['text'] == '/approve deadbeef' for event in events))
        self.assertFalse(any(event['kind'] == 'approval_decision' for event in events))

    def test_voice_mode_temporarily_speaks_without_overwriting_saved_mute(self):
        self.ui.handle('/mute')
        self.ui.handle('/voice on')
        self.heard('A spoken response please')
        self.finish()
        self.assertEqual(self.speech.mode, 'speaking')
        self.assertFalse(self.ui.audio)
        self.assertEqual(self.session.store.setting('terminal_auto_audio'), 'off')
        self.ui.handle('/voice off')
        replacement = TerminalUI(self.session, Speech(), self.output.append)
        self.assertFalse(replacement.voice_enabled)
        self.assertFalse(replacement.audio)

    def test_manual_talk_keeps_review_before_send(self):
        self.ui.handle('/talk')
        self.heard('Manual speech')
        self.assertFalse(self.ui.voice_enabled)
        self.assertEqual(self.session.draft(), 'Manual speech')
        self.assertFalse(self.session.agent.active)

    def test_explicit_alias_starts_mode_and_saved_draft_blocks_activation(self):
        self.ui.handle('start voice mode')
        self.assertTrue(self.ui.voice_enabled)
        self.ui.handle('/stop')
        self.session.draft('Review me')
        with self.assertRaisesRegex(AppError, '/send or /discard'):
            self.ui.handle('/voice on')
        self.assertFalse(self.ui.voice_enabled)

    def test_startup_without_voice_flag_never_records(self):
        with patch.object(self.ui, 'run_loop'):
            self.ui.run()
        self.assertEqual(self.speech.calls, [])
        self.ui.start_voice_requested = True
        with patch.object(self.ui, 'run_loop'):
            self.ui.run()
        self.assertEqual(self.speech.mode, 'listening')

    def test_saved_interrupted_turn_shows_ready_and_companion_prompt(self):
        self.session.store.update(self.session.sid, status='interrupted')
        self.session.store.event(self.session.sid, 'finished',
                                 {'status': 'interrupted', 'text': 'Task interrupted.'})
        self.ui.prompt = TerminalPrompt(io.StringIO())
        self.ui.restore()
        self.ui.pump()
        self.assertEqual(self.ui.prompt.status, 'ready')
        self.assertIn('You ›', self.ui.prompt.frame())
        self.assertNotIn('[ready]', self.output)
        self.assertFalse(self.session.agent.active)


class SilenceClassificationTests(unittest.TestCase):
    def test_only_empty_or_explicit_silence_errors_are_retryable(self):
        for output, kind in (('', 'empty'), ('ERROR_NO_MATCH', 'empty'),
                             ('ERROR_SPEECH_TIMEOUT', 'empty'),
                             ('ERROR_INSUFFICIENT_PERMISSIONS', 'error'),
                             ('{"error":"permission denied"}', 'error')):
            with self.subTest(output=output):
                speech = NativeSpeech()
                speech.mode = 'listening'
                speech.generation = 1
                with patch('terminal.subprocess.Popen') as popen, patch.object(speech, 'stop'):
                    popen.return_value.communicate.return_value = (output, '')
                    popen.return_value.returncode = 0
                    speech._run(1, 'listening', ['offline-speech'], None)
                self.assertEqual(speech.events.get_nowait()[0], kind)


if __name__ == '__main__':
    unittest.main(verbosity=2)
