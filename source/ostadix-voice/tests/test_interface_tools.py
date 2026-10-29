import sys
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import AppError
from interface_tools import InterfaceBridge


class InterfaceBridgeTests(unittest.TestCase):
    def setUp(self):
        self.bridge = InterfaceBridge(wait_timeout=.25)
        self.agent = SimpleNamespace(sid='session-a', task_generation=1,
                                     cancelled=threading.Event(), closed=False)
        self.calls = []
        self.results = []
        self.threads = []

    def tearDown(self):
        self.bridge.close()
        for thread in self.threads:
            thread.join(1)
            self.assertFalse(thread.is_alive())

    def handler(self, action):
        self.calls.append((action, threading.get_ident()))
        return {'available': True, 'handsfree': action == 'on', 'mode': 'idle'}

    def attach(self):
        self.bridge.attach(self.handler)

    def request(self, action='on'):
        def worker():
            try:
                result = self.bridge.dispatch(self.agent, {'operation': 'command',
                                              'argv': ['@voice', action]})
            except Exception as error:
                result = error
            self.results.append(result)
        thread = threading.Thread(target=worker)
        thread.start()
        self.threads.append(thread)
        deadline = time.monotonic() + 1
        while not self.bridge._pending and not self.results and time.monotonic() < deadline:
            time.sleep(.001)
        self.assertTrue(self.bridge._pending or self.results)
        return thread

    def finish(self, thread):
        thread.join(1)
        self.assertFalse(thread.is_alive())
        return self.results[-1]

    def test_roundtrip_runs_only_on_ui_thread(self):
        self.attach()
        thread = self.request()
        self.assertEqual(self.calls, [])
        self.assertEqual(self.bridge.pump('session-a', 1), 1)
        self.assertEqual(self.finish(thread), {'available': True, 'handsfree': True, 'mode': 'idle'})
        self.assertEqual(self.calls, [('on', threading.get_ident())])

    def test_microphone_mute_and_unmute_are_acknowledged_on_ui_thread(self):
        self.attach()
        for action in ('mute', 'unmute'):
            thread = self.request(action)
            self.bridge.pump('session-a', 1)
            self.assertTrue(self.finish(thread)['available'])
        self.assertEqual(self.calls, [('mute', threading.get_ident()), ('unmute', threading.get_ident())])

    def test_status_without_terminal_is_truthful(self):
        result = self.bridge.dispatch(self.agent, {'argv': ['@voice', 'status']})
        self.assertFalse(result['available'])
        with self.assertRaisesRegex(AppError, 'No active terminal'):
            self.bridge.dispatch(self.agent, {'argv': ['@voice', 'on']})

    def test_status_uses_live_ui_acknowledgement(self):
        self.attach()
        thread = self.request('status')
        self.bridge.pump('session-a', 1)
        self.assertTrue(self.finish(thread)['available'])
        self.assertEqual(self.calls[0][0], 'status')

    def test_cached_interface_context_is_fresh_scoped_passive_and_copied(self):
        self.attach()
        with patch('interface_tools.time.monotonic', return_value=100.0):
            self.bridge.update_context('session-a', {'available': True, 'interface': 'terminal',
                                                    'speech_queue': {'review': 32}})
        with patch('interface_tools.time.monotonic', return_value=104.0):
            context = self.bridge.read_context('session-a')
            self.assertEqual(context['speech_queue']['review'], 32)
            context['speech_queue']['review'] = 99
            self.assertEqual(self.bridge.read_context('session-a')['speech_queue']['review'], 32)
            self.assertFalse(self.bridge.read_context('session-b')['available'])
        with patch('interface_tools.time.monotonic', return_value=105.1):
            self.assertFalse(self.bridge.read_context('session-a')['available'])
        self.assertEqual(self.calls, [])
        self.bridge.detach()
        self.assertFalse(self.bridge.read_context('session-a')['available'])

    def test_context_cannot_be_published_from_a_worker(self):
        self.attach()
        errors = []
        def publish():
            try:
                self.bridge.update_context('session-a', {'available': True})
            except AppError as error:
                errors.append(str(error))
        thread = threading.Thread(target=publish)
        thread.start(); thread.join(1)
        self.assertEqual(len(errors), 1)
        self.assertFalse(self.bridge.read_context('session-a')['available'])

    def test_expired_request_cannot_execute_later(self):
        self.bridge.wait_timeout = .04
        self.attach()
        thread = self.request()
        self.assertIsInstance(self.finish(thread), AppError)
        self.bridge.pump('session-a', 1)
        self.assertEqual(self.calls, [])

    def test_pump_rejects_expiry_before_waiter_observes_it(self):
        self.attach()
        thread = self.request()
        self.bridge._pending[0].deadline = time.monotonic() - 1
        self.bridge.pump('session-a', 1)
        self.assertIsInstance(self.finish(thread), AppError)
        self.assertEqual(self.calls, [])

    def test_cancelled_request_cannot_execute(self):
        self.attach()
        thread = self.request()
        self.agent.cancelled.set()
        self.bridge.pump('session-a', 1)
        self.assertIsInstance(self.finish(thread), AppError)
        self.assertEqual(self.calls, [])

    def test_cancellation_wakes_without_ui_pump(self):
        self.attach()
        thread = self.request()
        self.agent.cancelled.set()
        self.assertIsInstance(self.finish(thread), AppError)
        self.assertEqual(self.calls, [])

    def test_stale_session_cannot_execute(self):
        self.attach()
        thread = self.request()
        self.bridge.pump('session-b', 1)
        self.assertIsInstance(self.finish(thread), AppError)
        self.assertEqual(self.calls, [])

    def test_stale_generation_cannot_execute(self):
        self.attach()
        thread = self.request()
        self.bridge.pump('session-a', 2)
        self.assertIsInstance(self.finish(thread), AppError)
        self.assertEqual(self.calls, [])

    def test_changed_agent_is_rechecked(self):
        self.attach()
        thread = self.request()
        self.agent.task_generation = 2
        self.bridge.pump('session-a', 1)
        self.assertIsInstance(self.finish(thread), AppError)
        self.assertEqual(self.calls, [])

    def test_close_wakes_waiters_and_prevents_reattachment(self):
        self.attach()
        thread = self.request()
        self.bridge.close()
        self.assertIsInstance(self.finish(thread), AppError)
        with self.assertRaises(AppError):
            self.attach()
        self.assertEqual(self.calls, [])

    def test_detach_and_attach_do_not_replay_controls(self):
        self.attach()
        thread = self.request()
        self.bridge.detach()
        self.attach()
        self.bridge.pump('session-a', 1)
        self.assertIsInstance(self.finish(thread), AppError)
        self.assertEqual(self.calls, [])

    def test_reattachment_invalidates_waiting_controls(self):
        self.attach()
        thread = self.request()
        self.attach()
        self.bridge.pump('session-a', 1)
        self.assertIsInstance(self.finish(thread), AppError)
        self.assertEqual(self.calls, [])

    def test_invalid_actions_are_never_queued(self):
        self.attach()
        for argv in (['@voice'], ['@voice', 'on', 'extra'], ['@voice', '/approve'],
                     ['other', 'on'], ['@voice', ['on']], 'on'):
            with self.subTest(argv=argv), self.assertRaises(AppError):
                self.bridge.dispatch(self.agent, {'argv': argv})
        self.assertEqual(len(self.bridge._pending), 0)

    def test_non_ui_thread_cannot_invoke_handler(self):
        self.attach()
        errors = []
        def wrong_thread():
            try:
                self.bridge.pump('session-a', 1)
            except Exception as error:
                errors.append(error)
        thread = threading.Thread(target=wrong_thread)
        thread.start(); thread.join(1)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertEqual(self.calls, [])

    def test_handler_error_is_returned_to_waiter(self):
        def fail(action):
            raise AppError('Microphone service unavailable')
        self.bridge.attach(fail)
        thread = self.request()
        self.bridge.pump('session-a', 1)
        self.assertRegex(str(self.finish(thread)), 'Microphone service unavailable')

    def test_aliases_are_normalized(self):
        self.attach()
        thread = self.request('stop')
        self.bridge.pump('session-a', 1)
        self.finish(thread)
        self.assertEqual(self.calls[0][0], 'off')


if __name__ == '__main__':
    unittest.main()
