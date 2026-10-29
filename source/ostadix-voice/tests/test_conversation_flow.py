"""Conversation flow regressions; simulated audio, no cloud/model requests."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terminal import TerminalPrompt, TerminalSession, TerminalUI
from test_app import FakeRPC, eventually
from test_handsfree import Speech


class ConversationFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        (root / 'project').mkdir()
        self.codex = patch('agent.codex_command', return_value=['fixture'])
        self.codex.start()
        self.addCleanup(self.codex.stop)
        self.session = TerminalSession(root / 'project', root / 'state', FakeRPC)
        self.addCleanup(self.session.close)
        self.speech = Speech()
        self.output = []
        self.ui = TerminalUI(self.session, self.speech, self.output.append)
        self.ui.local = None
        self.ui.companion = Mock()
        self.ui.companion.messages.return_value = {'messages': [], 'eligible': False}
        self.ui.prompt = TerminalPrompt(io.StringIO())
        self.ui.restore()
        self.output.clear()

    def start(self):
        self.ui.handle('Help with this task')
        eventually(lambda: self.session.store.session(self.session.sid)['turn_id'])

    def event(self, kind, data):
        self.session.store.event(self.session.sid, kind, data)

    def test_rich_turn_keeps_reply_and_draft_without_delegation_or_status_flood(self):
        self.ui.prompt.feed(b'Next question')
        self.ui.local = Mock()
        self.ui.local_next_poll = float('inf')
        self.ui.local_status = {'ready': True, 'running': True, 'active': 0, 'queued': 0,
                                'model': {'name': 'Gemma', 'backend': 'litert-npu'}}
        self.event('local_assistant', {'action': 'assign', 'id': 'listener', 'name': 'Listener',
                   'state': 'queued', 'assignment': {'id': 'assignment-1'}})
        for active in (1, 0, 1, 0):
            self.ui.local_status['active'] = active
            self.session.store.update(self.session.sid, status='thinking' if active else 'executing')
            self.event('tool_result', {'success': True, 'result': {'local': True, 'execution_pending': bool(active)}})
            self.ui.pump()
        for payload in (
                {'action': 'read', 'assignment': {'id': 'assignment-1', 'result': 'Internal draft'}},
                {'assignment': {'id': 'assignment-1', 'latest_result': 'Internal draft'}}):
            self.event('local_assistant', dict(id='listener', name='Listener', state='completed', **payload))
        self.event('assistant', {'text': 'Here is your useful answer.'})
        self.event('finished', {'status': 'idle', 'text': 'Here is your useful answer.'})
        self.session.store.update(self.session.sid, status='idle')
        self.ui.pump()
        self.assertEqual(self.output, ['Ostadix: Here is your useful answer.'])
        self.assertEqual(self.ui.prompt.text, 'Next question')
        self.assertEqual(self.ui.prompt.local, 'NPU helper ready')
        self.assertIn('Speaking', self.ui.prompt.frame())
        self.assertEqual(self.speech.calls, [('speaking', 'Here is your useful answer.')])
        self.ui.handle('/logs')
        self.assertIn('Internal draft', '\n'.join(self.output))

    def test_manual_assistant_result_is_shown_once_and_failures_remain_visible(self):
        result = {'id': 'helper', 'name': 'Helper', 'state': 'completed',
                  'assignment': {'id': 'manual-1', 'latest_result': 'Finished review'}}
        self.event('local_assistant', result)
        self.event('local_assistant', result)
        self.event('tool_result', {'success': False, 'result': {'local': True, 'error': 'Worker failed'}})
        self.event('approval', {'id': 'abcdef123456', 'warning': 'Review this action', 'arguments': {'path': 'important.txt'}})
        self.ui.pump()
        self.assertEqual(self.output.count('Finished review'), 1)
        self.assertIn('Worker failed', '\n'.join(self.output))
        self.assertIn('important.txt', '\n'.join(self.output))
        self.assertIn('/approve abcdef12', '\n'.join(self.output))

    def test_delegated_background_result_is_not_hidden_if_main_never_read_it(self):
        self.event('local_assistant', {'action': 'assign', 'id': 'helper', 'name': 'Helper',
                   'state': 'queued', 'assignment': {'id': 'background-1'}})
        self.ui.pump()
        self.assertEqual(self.output, [])
        self.event('local_assistant', {'id': 'helper', 'name': 'Helper', 'state': 'completed',
                   'assignment': {'id': 'background-1', 'latest_result': 'The saved background answer'}})
        self.ui.pump()
        self.assertIn('The saved background answer', self.output)

    def test_unrelated_completion_during_main_task_remains_discoverable(self):
        self.start()
        self.event('local_assistant', {'id': 'helper', 'name': 'Helper', 'state': 'completed',
                   'assignment': {'id': 'manual-1', 'latest_result': 'A separate result'}})
        self.ui.pump()
        self.assertIn('Helper finished · /assistant read Helper', self.output)

    def test_enter_during_task_waits_for_reply_and_speaker_tail_then_listens_once(self):
        self.start()
        self.ui.handle(''); self.ui.handle('')
        self.assertEqual(self.speech.calls, [])
        self.assertEqual(sum('listen when this reply finishes' in line for line in self.output), 1)
        self.session.agent.finish('idle', 'Answer')
        self.ui.pump()
        self.assertEqual(self.speech.mode, 'speaking')
        self.speech.complete()
        with patch('terminal.time.monotonic', return_value=100.0):
            self.ui.pump()
            self.assertIsNone(self.speech.mode)
        with patch('terminal.time.monotonic', return_value=100.7):
            self.ui.pump()
        self.assertEqual(self.speech.mode, 'listening')
        self.assertIsNone(self.ui.pending_talk_sid)
        self.assertFalse(self.ui.handsfree)

    def test_queued_listening_is_cancelled_by_stop_and_audio_failure(self):
        self.speech.mode = 'speaking'
        self.ui.handle('')
        self.ui.handle('/stop')
        self.ui.pump()
        self.assertIsNone(self.ui.pending_talk_sid)
        self.assertNotIn(('listening', None), self.speech.calls)
        self.speech.mode = 'speaking'
        self.ui.handle('')
        self.speech.complete('error', 'Android did not confirm the speech service stopped.')
        self.ui.pump()
        self.assertIsNone(self.ui.pending_talk_sid)
        self.assertNotIn(('listening', None), self.speech.calls)

    def test_direct_talk_consumes_queued_request_and_discard_does_not_reopen_mic(self):
        self.speech.mode = 'speaking'
        self.ui.handle('')
        self.speech.complete()
        self.ui.handle('/talk')
        self.assertIsNone(self.ui.pending_talk_sid)
        self.speech.complete('heard', 'A speech draft')
        self.ui.pump()
        self.ui.handle('/discard')
        self.ui.pump()
        self.assertEqual(self.speech.calls.count(('listening', None)), 1)

    def test_busy_typed_messages_are_saved_together_and_not_sent_automatically(self):
        self.start()
        self.ui.handle('First thought')
        self.ui.handle('Second thought')
        self.assertEqual(self.session.draft(), 'First thought\nSecond thought')
        self.session.agent.finish('idle', 'Done')
        self.ui.pump()
        self.assertFalse(self.session.agent.active)
        self.assertEqual(len([e for e in self.session.store.events(self.session.sid) if e['kind'] == 'user']), 1)
        self.ui.handle('/send')
        eventually(lambda: self.session.agent.active)
        self.assertEqual(self.session.draft(), '')

    def test_successful_history_is_kept_without_reprinting_on_startup(self):
        self.event('finished', {'status': 'idle', 'text': 'Saved answer'})
        self.ui.restore()
        self.assertFalse(any('Saved answer' in line for line in self.output))
        self.assertEqual(self.ui.last_answer, 'Saved answer')
        self.ui.handle('/hear')
        self.assertEqual(self.speech.calls, [('speaking', 'Saved answer')])


if __name__ == '__main__':
    unittest.main()
