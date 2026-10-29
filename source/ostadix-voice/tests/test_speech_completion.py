"""Completed recognition must not turn ordinary silence into failed cleanup.

All speech and Android service processes are mocked; these checks never record
or play audio. Termux:API 0.53.0's SpeechToTextService finishes its IntentService
request when the silent result closes, just as it does for normal dictation.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terminal import NativeSpeech, TerminalSession, TerminalUI
from test_app import FakeRPC


class SpeechCompletionTests(unittest.TestCase):
    def speech(self):
        speech = NativeSpeech()
        speech.mode = 'listening'
        speech.generation = 1
        return speech

    def test_completed_silence_does_not_request_an_android_stop(self):
        for output in ('', 'ERROR_NO_MATCH', 'ERROR_SPEECH_TIMEOUT',
                       '{"error":"ERROR_NO_MATCH"}'):
            with self.subTest(output=output):
                speech = self.speech()
                process = Mock(returncode=0)
                process.communicate.return_value = (output, '')
                with patch('terminal.subprocess.Popen', return_value=process), \
                        patch.object(speech, 'kill') as kill, \
                        patch('terminal.subprocess.run') as android_stop:
                    speech._run(1, 'listening', ['/fixture/speech', '-p'], None)
                    speech.close()
                self.assertEqual(speech.events.get_nowait()[0], 'empty')
                self.assertTrue(speech.events.empty())
                self.assertIsNone(speech.mode)
                self.assertIsNone(speech.process)
                self.assertFalse(speech.stopping)
                kill.assert_not_called()
                android_stop.assert_not_called()
                for stream in (process.stdin, process.stdout, process.stderr):
                    stream.close.assert_called_once()

    def test_silent_capture_is_released_before_a_retry_consumer_receives_it(self):
        speech = self.speech()
        process = Mock(returncode=0)
        process.communicate.return_value = ('', '')
        received = []

        def receive(event):
            self.assertEqual(event[0], 'empty')
            self.assertIsNone(speech.mode)
            self.assertIsNone(speech.process)
            self.assertFalse(speech.stopping)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close.assert_called_once()
            speech.start('listening')
            received.append(event)

        with patch('terminal.subprocess.Popen', return_value=process), \
                patch('terminal.subprocess.run') as android_stop, \
                patch('terminal.shutil.which', return_value='/fixture/speech'), \
                patch('terminal.threading.Thread') as retry, \
                patch.object(speech.events, 'put', side_effect=receive):
            speech._run(1, 'listening', ['/fixture/speech', '-p'], None)
        self.assertEqual(len(received), 1)
        self.assertEqual(speech.mode, 'listening')
        self.assertEqual(retry.call_args.kwargs['args'][1], 'listening')
        retry.return_value.start.assert_called_once()
        android_stop.assert_not_called()

    def test_native_silence_keeps_handsfree_enabled_and_retries_after_backoff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'workspace'
            root.mkdir()
            session = TerminalSession(root, Path(directory) / 'state', FakeRPC)
            speech = self.speech()
            ui = TerminalUI(session, speech, lambda text: None, audio=False)
            ui.local = None
            ui.companion_next_poll = float('inf')
            ui.handsfree = True
            ui.voice_sid = session.sid
            ui.capture_sid = session.sid
            process = Mock(returncode=0)
            process.communicate.return_value = ('', '')
            try:
                with patch('terminal.subprocess.Popen', return_value=process), \
                        patch('terminal.subprocess.run') as android_stop, \
                        patch('terminal.time.monotonic', return_value=100.0):
                    speech._run(1, 'listening', ['/fixture/speech', '-p'], None)
                    ui.pump()
                self.assertTrue(ui.voice_enabled)
                self.assertEqual(ui.voice_empty_attempts, 1)
                self.assertFalse(session.agent.active)
                self.assertEqual(session.draft(), '')
                android_stop.assert_not_called()
                with patch.object(speech, 'start') as retry:
                    with patch('terminal.time.monotonic', return_value=101.4):
                        ui.pump()
                    retry.assert_not_called()
                    with patch('terminal.time.monotonic', return_value=101.6):
                        ui.pump()
                    retry.assert_called_once_with('listening')
                self.assertTrue(ui.voice_enabled)
            finally:
                session.agent.interface.detach()
                speech.close()
                session.close()

    def test_failed_transport_still_requests_cleanup_even_with_silence_text(self):
        speech = self.speech()
        process = Mock(returncode=1)
        process.communicate.return_value = ('', 'ERROR_SPEECH_TIMEOUT')
        stopped = subprocess.CompletedProcess([], 0, 'Service stopped', '')
        with patch('terminal.subprocess.Popen', return_value=process), \
                patch.object(speech, 'kill') as kill, \
                patch('terminal.shutil.which', return_value='/fixture/am'), \
                patch('terminal.subprocess.run', return_value=stopped) as android_stop:
            speech._run(1, 'listening', ['/fixture/speech', '-p'], None)
            speech.close()
        kill.assert_called_once_with(process)
        android_stop.assert_called_once()
        self.assertEqual(android_stop.call_args.args[0][-1], NativeSpeech.COMPONENTS['listening'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
