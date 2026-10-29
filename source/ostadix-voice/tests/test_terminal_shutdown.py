"""Deterministic terminal cancellation/shutdown races; no model or audio calls."""
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import AppError
from terminal import TerminalSession, TerminalUI
from test_app import FakeRPC
from test_terminal import Audio


class CancellationShutdownTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'workspace'
        self.root.mkdir()
        self.state = Path(self.temp.name) / 'state'
        self.session = TerminalSession(self.root, self.state, FakeRPC)
        self.audio = Audio()
        self.ui = TerminalUI(self.session, self.audio, lambda _: None, audio=False)
        self.ui.local = None

    def tearDown(self):
        self.session.close()
        self.temp.cleanup()

    def close_in_thread(self, errors):
        def close():
            try:
                self.session.close()
            except Exception as error:
                errors.append(error)
        worker = threading.Thread(target=close)
        worker.start()
        return worker

    def test_close_preserves_late_cancellation_notice_and_session_lock(self):
        sid = self.session.sid
        entered = threading.Event()
        release = threading.Event()
        agent_closed = threading.Event()
        errors = []
        original_close = self.session.agent.close

        def local_cancel(agent, captured_sid):
            # The task can become inactive before local cancellation's final
            # write. Neither active=False nor tool_lock alone signals completion.
            agent.finish('interrupted', 'Cancellation acknowledged.')
            entered.set()
            if not release.wait(3):
                raise AssertionError('test did not release local cancellation')
            with agent.tool_lock:
                agent.store.event(captured_sid, 'notice', {'text': 'Late cancellation saved.'})

        def close_agent():
            original_close()
            agent_closed.set()

        self.session.agent.sid = sid
        self.session.agent.active = True
        closer = None
        with patch.object(self.session.agent.local, 'cancel', side_effect=local_cancel), \
                patch.object(self.session.agent, 'close', side_effect=close_agent), \
                patch('threading.excepthook', side_effect=lambda args: errors.append(args.exc_value)):
            try:
                self.ui.handle('/cancel')
                self.assertTrue(entered.wait(2))
                self.assertFalse(self.session.agent.active)
                closer = self.close_in_thread(errors)
                self.assertTrue(agent_closed.wait(2))
                self.assertTrue(closer.is_alive())
                self.assertEqual(self.session.store.session(sid)['status'], 'interrupted')
                with self.assertRaises(AppError):
                    TerminalSession(self.root, self.state, FakeRPC)
                with self.assertRaises(AppError):
                    self.ui.handle('/cancel')
            finally:
                release.set()
                if closer is not None:
                    closer.join(3)
        self.assertFalse(closer.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(self.audio.events.empty())
        self.session = TerminalSession(self.root, self.state, FakeRPC)
        notices = [event['data'].get('text') for event in self.session.store.events(sid)
                   if event['kind'] == 'notice']
        self.assertIn('Late cancellation saved.', notices)

    def test_cancel_captures_conversation_before_worker_is_scheduled(self):
        source = self.session.sid
        queued = threading.Event()
        release = threading.Event()
        completed = threading.Event()
        errors = []
        captured = []
        threads = []
        real_thread = threading.Thread

        def delayed_thread(*args, **kwargs):
            target = kwargs.pop('target')
            def delayed():
                queued.set()
                if not release.wait(3):
                    raise AssertionError('test did not schedule cancellation')
                target()
            worker = real_thread(*args, target=delayed, **kwargs)
            threads.append(worker)
            return worker

        def cancel(sid):
            captured.append(sid)
            self.session.store.event(sid, 'notice', {'text': 'Correct conversation cancelled.'})
            completed.set()

        with patch.object(self.session.agent, 'cancel', side_effect=cancel), \
                patch('terminal.threading.Thread', side_effect=delayed_thread), \
                patch('threading.excepthook', side_effect=lambda args: errors.append(args.exc_value)):
            try:
                self.ui.handle('/cancel')
                self.assertTrue(queued.wait(2))
                self.ui.handle('/new')
                destination = self.session.sid
                self.assertNotEqual(source, destination)
                release.set()
                self.assertTrue(completed.wait(2))
            finally:
                release.set()
                for worker in threads:
                    worker.join(3)
        self.assertEqual(errors, [])
        self.assertEqual(captured, [source])
        self.assertTrue(any(event['data'].get('text') == 'Correct conversation cancelled.'
                            for event in self.session.store.events(source)))
        self.assertFalse(any(event['data'].get('text') == 'Correct conversation cancelled.'
                             for event in self.session.store.events(destination)))

    def test_repeated_cancel_is_coalesced_until_worker_finishes(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []
        errors = []

        def cancel(sid):
            calls.append(sid)
            entered.set()
            if not release.wait(3):
                raise AssertionError('test did not release cancellation')

        with patch.object(self.session.agent, 'cancel', side_effect=cancel), \
                patch('threading.excepthook', side_effect=lambda args: errors.append(args.exc_value)):
            try:
                self.ui.handle('/cancel')
                self.assertTrue(entered.wait(2))
                self.ui.handle('/cancel')
                self.ui.handle('/cancel')
                self.assertEqual(calls, [self.session.sid])
            finally:
                release.set()
                self.session.close()
        self.assertEqual(errors, [])

    def test_agent_close_unblocks_cancellation_before_waiting_for_it(self):
        entered = threading.Event()
        rpc_closed = threading.Event()
        errors = []
        original_close = self.session.agent.close

        def cancel(sid):
            entered.set()
            if not rpc_closed.wait(3):
                raise AssertionError('close waited for cancellation before closing RPC')
            self.session.store.event(sid, 'notice', {'text': 'Interrupt released by close.'})

        def close_agent():
            original_close()
            rpc_closed.set()

        with patch.object(self.session.agent, 'cancel', side_effect=cancel), \
                patch.object(self.session.agent, 'close', side_effect=close_agent), \
                patch('threading.excepthook', side_effect=lambda args: errors.append(args.exc_value)):
            self.ui.handle('/cancel')
            self.assertTrue(entered.wait(2))
            closer = self.close_in_thread(errors)
            try:
                closer.join(2)
                self.assertFalse(closer.is_alive())
            finally:
                rpc_closed.set()
                closer.join(3)
        self.assertEqual(errors, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
