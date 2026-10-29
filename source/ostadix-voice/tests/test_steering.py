"""Deterministic turn/steer delivery and race tests; no cloud requests."""
from unittest.mock import patch

from agent import Agent, AppError, RPCRejected, SteerUnavailable, Store
from test_app import Base, FakeRPC


class SteeringTests(Base):
    def setUp(self):
        super().setUp()
        command = patch('agent.codex_command', return_value=['offline-protocol-fixture'])
        command.start(); self.addCleanup(command.stop)
        self.start()
        self.rpc = self.agent.rpc
        self.original = self.rpc.call
        self.sent = []
        self.hook = None
        self.result = {'turnId': 'turn-offline'}
        def call(method, params, **kwargs):
            if method != 'turn/steer':
                return self.original(method, params, **kwargs)
            self.sent.append(params)
            if self.hook:
                self.hook()
            return self.result
        self.rpc.call = call

    def steer(self, text='Actually, use the other file', ident='follow-up-12345'):
        return self.agent.steer(self.sid, text, ident)

    def record(self, ident='follow-up-12345'):
        with self.store.lock:
            row = self.store.db.execute('SELECT * FROM steering_requests WHERE id=?', (ident,)).fetchone()
            return dict(row) if row else None

    def user_followups(self):
        return [e['data'] for e in self.store.events(self.sid)
                if e['kind'] == 'user' and e['data'].get('steered')]

    def test_sends_exact_protocol_and_records_only_after_acceptance(self):
        self.hook = lambda: self.assertFalse(self.user_followups())
        self.assertEqual(self.steer(), {'steered': True, 'turn_id': 'turn-offline', 'duplicate': False})
        self.assertEqual(self.sent, [{'threadId': 'thread-offline', 'expectedTurnId': 'turn-offline',
            'input': [{'type': 'text', 'text': 'Actually, use the other file'}],
            'clientUserMessageId': 'follow-up-12345'}])
        self.assertEqual(self.record()['status'], 'accepted')
        self.assertEqual(len(self.user_followups()), 1)
        self.assertEqual(self.store.session(self.sid)['turn_id'], 'turn-offline')

    def test_duplicate_accepted_followup_does_not_send_twice(self):
        self.steer()
        self.assertTrue(self.steer()['duplicate'])
        self.agent.finish('idle', 'done')
        self.assertTrue(self.steer()['duplicate'])
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(len(self.user_followups()), 1)

    def test_duplicate_id_with_other_input_is_rejected(self):
        self.steer()
        with self.assertRaises(AppError):
            self.steer('Different words')
        self.assertEqual(len(self.sent), 1)

    def test_regular_task_id_cannot_be_reused_for_followup(self):
        with self.assertRaisesRegex(AppError, 'separate task'):
            self.steer(ident='request-123456')
        self.assertFalse(self.sent)

    def test_idle_preflight_allows_a_new_request_without_claiming_delivery(self):
        self.agent.finish('idle', 'done')
        with self.assertRaises(SteerUnavailable) as caught:
            self.steer()
        self.assertTrue(caught.exception.safe_to_submit)
        self.assertEqual(caught.exception.reason, 'idle')
        self.assertFalse(self.sent)
        self.assertIsNone(self.record())

    def test_starting_preflight_waits_without_sending(self):
        self.store.update(self.sid, turn_id=None)
        with self.assertRaises(SteerUnavailable) as caught:
            self.steer()
        self.assertFalse(caught.exception.safe_to_submit)
        self.assertEqual(caught.exception.reason, 'starting')
        self.assertFalse(self.sent)
        self.assertIsNone(self.record())

    def test_cancelled_preflight_never_implicitly_starts_a_new_task(self):
        self.agent.cancel(self.sid)
        with self.assertRaises(AppError) as caught:
            self.steer()
        self.assertNotIsInstance(caught.exception, SteerUnavailable)
        self.assertFalse(self.sent)
        self.assertIsNone(self.record())

    def test_wrong_conversation_does_not_send(self):
        sid = self.store.create(self.root)
        with self.assertRaises(AppError):
            self.agent.steer(sid, 'Other conversation', 'follow-other-1234')
        self.assertFalse(self.sent)

    def test_definite_rejection_keeps_text_without_user_message_or_replay(self):
        def reject():
            raise RPCRejected('Expected turn no longer active')
        self.hook = reject
        with self.assertRaisesRegex(AppError, 'was rejected'):
            self.steer()
        self.assertEqual(self.record()['status'], 'rejected')
        self.assertEqual(self.record()['prompt'], 'Actually, use the other file')
        with self.assertRaisesRegex(AppError, 'not replayed'):
            self.steer()
        self.assertEqual(len(self.sent), 1)
        self.assertFalse(self.user_followups())

    def test_no_active_turn_rejection_exposes_narrow_completion_retry_signal(self):
        def reject():
            raise RPCRejected('no active turn to steer')
        self.hook = reject
        with self.assertRaises(SteerUnavailable) as caught:
            self.steer()
        self.assertEqual(caught.exception.reason, 'rejected_idle')
        self.assertFalse(caught.exception.safe_to_submit)
        self.assertEqual(self.record()['status'], 'rejected')
        self.assertEqual(self.record()['error'], 'no active turn to steer')
        self.assertFalse(self.user_followups())
        with self.assertRaisesRegex(AppError, 'not replayed'):
            self.steer()
        self.assertEqual(len(self.sent), 1)

    def test_unconfirmed_error_with_no_active_turn_text_cannot_retry(self):
        def disconnect():
            raise AppError('no active turn to steer')
        self.hook = disconnect
        with self.assertRaisesRegex(AppError, 'unconfirmed') as caught:
            self.steer()
        self.assertNotIsInstance(caught.exception, SteerUnavailable)
        self.assertEqual(self.record()['status'], 'unknown')
        self.assertFalse(self.user_followups())

    def test_transport_failure_is_durable_and_cannot_fall_back_to_submit(self):
        def disconnect():
            raise AppError('Connection lost after writing request')
        self.hook = disconnect
        with self.assertRaisesRegex(AppError, 'unconfirmed'):
            self.steer()
        self.assertEqual(self.record()['status'], 'unknown')
        self.agent.finish('idle', 'done')
        with self.assertRaisesRegex(AppError, 'cannot be replayed'):
            self.agent.submit(self.sid, 'Actually, use the other file', 'follow-up-12345')
        self.assertEqual(len(self.sent), 1)
        self.assertFalse(self.user_followups())
        replacement = Store(self.state)
        new_agent = Agent(replacement, self.root, FakeRPC)
        try:
            with self.assertRaisesRegex(AppError, 'not replayed'):
                new_agent.steer(self.sid, 'Actually, use the other file', 'follow-up-12345')
        finally:
            new_agent.close(); replacement.db.close()

    def test_pending_delivery_is_not_replayed_after_interruption(self):
        with self.store.lock, self.store.db:
            self.store.db.execute('INSERT INTO steering_requests VALUES(?,?,?,?,?,NULL)',
                ('follow-up-12345', self.sid, 'Actually, use the other file', 'pending', 'turn-offline'))
        with self.assertRaisesRegex(AppError, 'not replayed'):
            self.steer()
        self.assertFalse(self.sent)

    def test_wrong_returned_turn_id_is_not_accepted(self):
        self.result = {'turnId': 'another-turn'}
        with self.assertRaisesRegex(AppError, 'unconfirmed'):
            self.steer()
        self.assertEqual(self.record()['status'], 'unknown')
        self.assertFalse(self.user_followups())

    def test_completion_during_rpc_does_not_resurrect_task_or_attach_new_input(self):
        self.hook = lambda: self.agent.finish('idle', 'already finished')
        with self.assertRaisesRegex(AppError, 'previous turn accepted'):
            self.steer()
        self.assertEqual(self.record()['status'], 'accepted_stale')
        self.assertEqual(self.store.session(self.sid)['status'], 'idle')
        self.assertIsNone(self.store.session(self.sid)['turn_id'])
        self.assertFalse(self.user_followups())

    def test_changed_generation_during_rpc_does_not_confirm_current_task(self):
        self.hook = lambda: setattr(self.agent, 'task_generation', self.agent.task_generation + 1)
        with self.assertRaisesRegex(AppError, 'task has since changed'):
            self.steer()
        self.assertEqual(self.record()['status'], 'accepted_stale')
        self.assertFalse(self.user_followups())

    def test_cancellation_during_rpc_does_not_confirm_current_task(self):
        self.hook = self.agent.cancelled.set
        with self.assertRaisesRegex(AppError, 'task has since changed'):
            self.steer()
        self.assertEqual(self.record()['status'], 'accepted_stale')

    def test_invalid_input_never_sends(self):
        for text, ident in [('', 'follow-invalid-1234'), ('x' * 32001, 'follow-invalid-1234'),
                            ('words', 'short'), ('words', 'illegal/id')]:
            with self.subTest(text=text[:20], ident=ident), self.assertRaises(AppError):
                self.agent.steer(self.sid, text, ident)
        self.assertFalse(self.sent)
