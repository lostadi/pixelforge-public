"""Speech transport regressions; no Android recording or playback is started."""
from pathlib import Path
import queue
import subprocess
import sys
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terminal import NativeSpeech


class NativeSpeechTests(unittest.TestCase):
    def test_long_playback_sends_every_chunk_and_remains_busy_until_finished(self):
        speech = NativeSpeech()
        speech.mode = 'speaking'
        speech.generation = 1
        answer = ('Read every paragraph in order. ' * 160) + '\n\nLast sentence.'
        processes = []

        def create(*args, **kwargs):
            self.assertEqual(speech.mode, 'speaking')
            self.assertIsNone(speech.process)
            process = Mock(returncode=0)
            process.communicate.return_value = ('', '')
            processes.append(process)
            return process

        with patch('terminal.subprocess.Popen', side_effect=create):
            speech._run(1, 'speaking', ['/fixture/speech'], answer)
        self.assertGreater(len(processes), 1)
        self.assertEqual(''.join(process.communicate.call_args.kwargs['input'][:-1]
                                 for process in processes), answer)
        for process in processes:
            self.assertLessEqual(len(process.communicate.call_args.kwargs['input']) - 1, 400)
            self.assertEqual(process.communicate.call_args.kwargs['timeout'], 60)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close.assert_called_once()
        self.assertIsNone(speech.mode)
        self.assertIsNone(speech.process)
        self.assertTrue(speech.events.empty())

    def test_stop_during_long_playback_does_not_start_remaining_chunks(self):
        speech = NativeSpeech()
        speech.mode = 'speaking'
        speech.generation = 1
        process = Mock(returncode=0)

        def finish(**kwargs):
            speech.stop()
            return '', ''

        process.communicate.side_effect = finish
        stopped = subprocess.CompletedProcess([], 0, 'Service stopped', '')
        with patch('terminal.subprocess.Popen', return_value=process) as launch, \
                patch.object(speech, 'kill') as kill, \
                patch('terminal.shutil.which', return_value='/fixture/am'), \
                patch('terminal.subprocess.run', return_value=stopped):
            speech._run(1, 'speaking', ['/fixture/speech'], 'Long answer. ' * 500)
            speech.close()
        launch.assert_called_once()
        kill.assert_called_once_with(process)
        self.assertIsNone(speech.mode)
        self.assertTrue(speech.events.empty())

    def test_error_in_later_chunk_reports_failure_without_reading_the_rest(self):
        speech = NativeSpeech()
        speech.mode = 'speaking'
        speech.generation = 1
        processes = [Mock(returncode=0), Mock(returncode=0)]
        processes[0].communicate.return_value = ('', '')
        processes[1].communicate.return_value = ('{"error":"Speech engine unavailable"}', '')
        with patch('terminal.subprocess.Popen', side_effect=processes) as launch, \
                patch.object(speech, 'stop') as stop:
            speech._run(1, 'speaking', ['/fixture/speech'], 'Long answer. ' * 500)
        self.assertEqual(launch.call_count, 2)
        stop.assert_called_once()
        self.assertIn('Speech engine unavailable', speech.events.get_nowait()[1])
        self.assertTrue(speech.events.empty())

    def response(self, mode, stdout, stderr=''):
        speech = NativeSpeech()
        speech.mode = mode
        speech.generation = 1
        process = Mock(returncode=0)
        process.communicate.return_value = (stdout, stderr)
        stopped = subprocess.CompletedProcess([], 0, 'Service stopped', '')
        with patch('terminal.subprocess.Popen', return_value=process), \
                patch.object(speech, 'kill'), \
                patch('terminal.shutil.which', return_value='/fixture/am'), \
                patch('terminal.subprocess.run', return_value=stopped) as run:
            speech._run(1, mode, ['/fixture/speech'], None)
            speech.close()
            events = []
            while True:
                try:
                    events.append(speech.events.get_nowait())
                except queue.Empty:
                    break
            return events, run.call_args_list

    def test_recognition_keeps_full_api_output(self):
        speech = NativeSpeech()
        started = threading.Event()
        with patch('terminal.shutil.which', return_value='/fixture/speech'), \
                patch.object(speech, '_run', side_effect=lambda *args: started.set()) as run:
            speech.start('listening')
            self.assertTrue(started.wait(1))
            self.assertEqual(run.call_args.args[2], ['/fixture/speech', '-p'])

    def test_completed_capture_is_released_before_a_consumer_receives_transcript(self):
        speech = NativeSpeech()
        speech.mode = 'listening'
        speech.generation = 1
        process = Mock(returncode=0)
        process.communicate.return_value = ('Please answer now.', '')
        received = []

        def receive(event):
            # A fast UI/model response can request playback as soon as the
            # transcript is visible. It must not see an obsolete live mic.
            self.assertIsNone(speech.mode)
            self.assertIsNone(speech.process)
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close.assert_called_once()
            speech.start('speaking', 'Here is the answer.')
            received.append(event)

        with patch('terminal.subprocess.Popen', return_value=process), \
                patch('terminal.shutil.which', return_value='/fixture/speech'), \
                patch('terminal.threading.Thread') as next_worker, \
                patch.object(speech.events, 'put', side_effect=receive):
            speech._run(1, 'listening', ['/fixture/speech', '-p'], None)
        self.assertEqual(received, [('heard', 'Please answer now.')])
        self.assertEqual(speech.mode, 'speaking')
        self.assertEqual(next_worker.call_args.kwargs['args'][1], 'speaking')
        next_worker.return_value.start.assert_called_once()

    def test_pretty_json_error_never_becomes_closing_brace_transcript(self):
        events, stops = self.response('listening', '{\n  "error": "Microphone permission denied"\n}\n')
        self.assertEqual([kind for kind, _ in events], ['error'])
        self.assertIn('permission denied', events[0][1])
        self.assertEqual(stops[0].args[0][-1], NativeSpeech.COMPONENTS['listening'])

    def test_error_after_partial_transcript_rejects_whole_capture(self):
        events, _ = self.response('listening', 'Read the\n{\n "error": "Recognition unavailable"\n}\n')
        self.assertEqual([kind for kind, _ in events], ['error'])
        self.assertIn('Recognition unavailable', events[0][1])

    def test_zero_exit_speaker_error_is_reported(self):
        events, stops = self.response('speaking', '{\n "error": "Speech engine unavailable"\n}\n')
        self.assertEqual([kind for kind, _ in events], ['playback_error'])
        self.assertIn('Speech engine unavailable', events[0][1])
        self.assertEqual(stops[0].args[0][-1], NativeSpeech.COMPONENTS['speaking'])

    def test_structured_stderr_error_is_reported(self):
        events, _ = self.response('speaking', '', '{"error":"Voice is not installed"}')
        self.assertEqual([kind for kind, _ in events], ['playback_error'])
        self.assertIn('Voice is not installed', events[0][1])

    def test_truncated_service_response_cannot_be_a_request(self):
        for text in ('}', ']', '{\n "error":', 'ERROR_INSUFFICIENT_PERMISSIONS'):
            with self.subTest(text=text):
                events, _ = self.response('listening', text)
                self.assertEqual([kind for kind, _ in events], ['error'])

    def test_normal_dictation_can_begin_with_error_or_permission(self):
        for text in ('Error handling needs a fix', 'Permission to edit is already granted'):
            with self.subTest(text=text):
                events, stops = self.response('listening', 'Partial words\n' + text + '\n')
                self.assertEqual(events, [('heard', text)])
                self.assertEqual(stops, [])

    def test_timeout_never_performs_an_unbounded_pipe_drain(self):
        speech = NativeSpeech()
        speech.mode = 'listening'
        speech.generation = 1
        process = Mock()
        process.communicate.side_effect = subprocess.TimeoutExpired('speech', 40)
        stopped = subprocess.CompletedProcess([], 0, 'Service stopped', '')
        with patch('terminal.subprocess.Popen', return_value=process), \
                patch.object(speech, 'kill') as kill, \
                patch('terminal.shutil.which', return_value='/fixture/am'), \
                patch('terminal.subprocess.run', return_value=stopped):
            speech._run(1, 'listening', ['/fixture/speech', '-p'], None)
            speech.close()
        self.assertEqual(process.communicate.call_count, 1)
        self.assertEqual(process.communicate.call_args.kwargs['timeout'], 40)
        kill.assert_called_once_with(process)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close.assert_called_once()
        self.assertIn('timed out', speech.events.get_nowait()[1])

    def test_stop_is_nonblocking_and_close_waits_for_owned_service(self):
        speech = NativeSpeech()
        speech.mode = 'speaking'
        speech.process = Mock()
        killing = threading.Event()
        allow_kill = threading.Event()
        stopping = threading.Event()
        allow_stop = threading.Event()
        closed = threading.Event()

        def kill(process):
            killing.set()
            self.assertTrue(allow_kill.wait(2))

        def stop_service(*args, **kwargs):
            stopping.set()
            self.assertTrue(allow_stop.wait(2))
            return subprocess.CompletedProcess([], 0, 'Service stopped', '')

        with patch.object(speech, 'kill', side_effect=kill) as kill_call, \
                patch('terminal.shutil.which', return_value='/fixture/am'), \
                patch('terminal.subprocess.run', side_effect=stop_service) as run:
            speech.stop()  # Must return while process cleanup is still blocked.
            self.assertTrue(killing.wait(1))
            self.assertTrue(speech.stopping)
            closer = threading.Thread(target=lambda: (speech.close(), closed.set()))
            closer.start()
            try:
                self.assertFalse(closed.wait(.05))
                allow_kill.set()
                self.assertTrue(stopping.wait(1))
                self.assertFalse(closed.wait(.05))
                speech.stop()  # Repeated stop must not create another worker.
                allow_stop.set()
                self.assertTrue(closed.wait(1))
            finally:
                allow_kill.set()
                allow_stop.set()
                closer.join(2)
            kill_call.assert_called_once()
            run.assert_called_once_with(
                ['/fixture/am', 'stopservice', '-n', NativeSpeech.COMPONENTS['speaking']],
                capture_output=True, text=True, timeout=5)
            self.assertFalse(speech.stopping)

    def test_close_has_a_finite_cleanup_wait(self):
        speech = NativeSpeech()
        worker = Mock()
        speech.stop_thread = worker
        speech.close()
        worker.join.assert_called_once_with(timeout=6.5)

    def test_service_stop_still_runs_when_cli_cleanup_fails(self):
        speech = NativeSpeech()
        speech.mode = 'listening'
        stopped = subprocess.CompletedProcess([], 0, 'Service stopped', '')
        with patch.object(speech, 'kill', side_effect=subprocess.TimeoutExpired('speech', 1)), \
                patch('terminal.shutil.which', return_value='/fixture/am'), \
                patch('terminal.subprocess.run', return_value=stopped) as run:
            speech.close()
        self.assertEqual(run.call_args.args[0][-1], NativeSpeech.COMPONENTS['listening'])
        self.assertIn('Audio command cleanup', speech.events.get_nowait()[1])


if __name__ == '__main__':
    unittest.main(verbosity=2)
