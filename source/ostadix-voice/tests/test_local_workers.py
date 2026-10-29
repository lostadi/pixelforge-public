"""Delegation limits and cloud lifecycle regressions without model inference."""
import json
from pathlib import Path
import sys
import threading
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Agent, AppError, Store
from local_workers import LocalWorkers, owner_key
from test_app import FakeRPC, eventually


class Client:
    def __init__(self):
        self.started = 0
        self.submitted = []
        self.cancelled = []
    def status(self): return {'running': True, 'ready': True}
    def start(self): self.started += 1
    def submit(self, owner, task, **kwargs):
        self.submitted.append((owner, task, kwargs))
        return {'id': str(len(self.submitted)), 'state': 'queued'}
    def jobs(self, owner): return [{'id': '1', 'state': 'running'}] if self.submitted else []
    def read(self, owner, ident, **kwargs):
        return {'id': ident, 'state': 'completed', 'result': 'Local answer', 'truncated': False}
    def cancel(self, owner, job_id=None):
        self.cancelled.append((owner, job_id)); return {'cancelled': [job_id]}


class AssistantClient(Client):
    def __init__(self):
        super().__init__()
        self.assistants = {}
        self.assistant_calls = []

    def assistant_create(self, owner, name, role='assistant'):
        item = {'id': name, 'name': name, 'role': role, 'state': 'waiting',
                'waiting_for_assignment': True, 'assignment': None}
        self.assistants[owner, name] = item
        self.assistant_calls.append(('create', owner, name))
        return item.copy()

    def assistant_assign(self, owner, ident, task, max_steps=2, max_tokens=512):
        item = self.assistants[owner, ident]
        item.update(state='queued', waiting_for_assignment=False, assignment={
            'id': 'assignment', 'state': 'queued', 'max_steps': max_steps,
            'max_tokens': max_tokens, 'steps_submitted': 0, 'completed_steps': 0,
            'steps_remaining': max_steps, 'tokens_requested': 0, 'latest_result': ''})
        self.assistant_calls.append(('assign', owner, ident, task))
        return item.copy()

    def assistant_read(self, owner, ident):
        return self.assistants[owner, ident].copy()

    def assistant_resume(self, owner, ident):
        self.assistants[owner, ident]['state'] = 'queued'
        self.assistant_calls.append(('resume', owner, ident))
        return self.assistant_read(owner, ident)

    def assistant_list(self, owner):
        return [item for (scope, _), item in self.assistants.items() if scope == owner]

    def assistant_cancel_owner(self, owner):
        self.assistant_calls.append(('cancel_owner', owner))
        for item in self.assistant_list(owner):
            item['state'] = 'cancelled'
        return {'cancelled': 1}


class LocalDelegationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'workspace'; self.root.mkdir()
        self.store = Store(Path(self.temp.name) / 'state')
        self.sid = self.store.create(self.root)
        self.patch = patch('agent.codex_command', return_value=['fixture']); self.patch.start()
        self.agent = Agent(self.store, self.root, FakeRPC)
        self.client = Client()
        self.agent.local = LocalWorkers(lambda: self.client)
        self.agent.submit(self.sid, 'Delegate a task', 'local-test-request')
        eventually(lambda: self.store.session(self.sid)['turn_id'])
    def tearDown(self):
        self.agent.close(); self.store.db.close(); self.patch.stop(); self.temp.cleanup()
    def call(self, *argv):
        return self.agent.execute({'operation':'command', 'argv':['@local', *argv]})
    def spawn(self, **data):
        return self.call('spawn', json.dumps(dict(task='Review the supplied text', **data)))
    def test_spawn_returns_real_job_without_shell_or_approval(self):
        result = self.spawn()
        self.assertTrue(result['execution_pending'])
        self.assertEqual(result['worker']['id'], '1')
        self.assertFalse(self.agent.approvals)
        self.assertEqual(self.client.submitted[0][0], owner_key(self.store.directory,self.sid,self.root))
    def test_six_worker_budget_prevents_unbounded_delegation(self):
        for _ in range(6): self.spawn(max_tokens=1)
        with self.assertRaisesRegex(AppError, 'budget reached'): self.spawn(max_tokens=1)
        self.assertEqual(len(self.client.submitted), 6)
    def test_total_output_budget_and_per_job_limit(self):
        with self.assertRaises(AppError): self.spawn(max_tokens=1025)
        for _ in range(4): self.spawn(max_tokens=1024)
        with self.assertRaisesRegex(AppError, 'budget reached'): self.spawn(max_tokens=1)
    def test_profile_limit_rejects_request_before_start_or_budget_charge(self):
        self.client.status = lambda: {'running': False, 'limits': {'max_tokens': 512},
                                      'model': {'context_per_worker': 1280}}
        with self.assertRaisesRegex(AppError, '1–512'):
            self.spawn(max_tokens=513)
        self.assertEqual(self.client.started, 0)
        self.assertEqual(self.client.submitted, [])
        self.assertEqual(self.agent.local.turns[self.sid]['tokens'], 0)
    def test_default_output_allowance_fits_smaller_profile(self):
        self.client.status = lambda: {'running': True, 'limits': {'max_tokens': 256}}
        result = self.spawn()
        self.assertEqual(self.client.submitted[0][2]['max_tokens'], 256)
        self.assertEqual(result['budget']['tokens'], 256)
    def test_unavailable_or_invalid_profile_limits_keep_legacy_cap(self):
        for limits in (None, {}, {'max_tokens': True}, {'max_tokens': 0}, {'max_tokens': '512'}, {'max_tokens': 4096}):
            with self.subTest(limits=limits):
                self.agent.local.begin_turn(self.sid, self.agent.task_generation)
                self.client.status = lambda: {'running': True, 'limits': limits}
                self.spawn(max_tokens=1024)
                with self.assertRaisesRegex(AppError, '1–1024'):
                    self.spawn(max_tokens=1025)
    def test_scope_checks_before_any_model_request(self):
        (self.root / '.env').write_text('private')
        outside = Path(self.temp.name) / 'outside'; outside.write_text('private')
        (self.root / 'link').symlink_to(outside)
        for path in ['.env', '../outside', str(outside), 'link']:
            with self.subTest(path=path), self.assertRaises((AppError,OSError)):
                self.spawn(paths=[path])
        self.assertEqual(self.client.submitted, [])
    def test_selected_small_file_passed_as_data_only(self):
        (self.root / 'small.txt').write_text('Data for review')
        self.spawn(paths=['small.txt'])
        self.assertIn('Data for review', self.client.submitted[0][1])
    def test_large_file_is_not_silently_truncated(self):
        (self.root / 'large.txt').write_text('a' * 7000)
        with self.assertRaisesRegex(AppError, 'exceeds'): self.spawn(paths=['large.txt'])
        self.assertEqual(self.client.submitted, [])
    def test_local_completion_does_not_start_another_cloud_turn(self):
        self.spawn()
        rpc = self.agent.rpc
        calls = len(rpc.calls)
        result = self.call('read', '1', '1')
        self.assertEqual(result['result'], 'Local answer')
        self.assertEqual(len(rpc.calls), calls)

    def assistants(self):
        self.client = AssistantClient()
        self.agent.local = LocalWorkers(lambda: self.client)
        self.agent.local.begin_turn(self.sid, self.agent.task_generation)
        self.call('assistant', 'create', json.dumps({'name': 'reviewer'}))
        return self.agent.local.owner(self.agent, self.sid)

    def test_assistant_steps_share_one_shot_request_budget(self):
        self.assistants()
        result = self.call('assistant', 'assign', json.dumps({'id': 'reviewer', 'task': 'Review', 'max_steps': 4, 'max_tokens': 512}))
        self.assertEqual(result['budget']['workers'], 4)
        self.assertFalse(result['cloud_wakeup'])
        for _ in range(2):
            self.spawn(max_tokens=512)
        with self.assertRaisesRegex(AppError, 'budget reached'):
            self.spawn(max_tokens=1)

    def test_assistant_profile_limit_rejects_before_start_or_assignment(self):
        self.assistants()
        self.client.started = 0
        self.client.status = lambda: {'running': False, 'limits': {'max_tokens': 512}}
        with self.assertRaisesRegex(AppError, '1–512 tokens per step'):
            self.call('assistant', 'assign', json.dumps({'id': 'reviewer', 'task': 'Review', 'max_tokens': 513}))
        self.assertEqual(self.client.started, 0)
        self.assertEqual([call for call in self.client.assistant_calls if call[0] == 'assign'], [])
        self.assertEqual(self.agent.local.turns[self.sid]['tokens'], 0)

    def test_assistant_resume_charges_remaining_steps_to_new_request(self):
        owner = self.assistants()
        self.call('assistant', 'assign', json.dumps({'id': 'reviewer', 'task': 'Review', 'max_steps': 4, 'max_tokens': 512}))
        item = self.client.assistants[owner, 'reviewer']
        item['state'] = 'paused'
        item['assignment']['steps_submitted'] = 1
        self.agent.local.begin_turn(self.sid, self.agent.task_generation)
        for _ in range(4):
            self.spawn(max_tokens=1)
        with self.assertRaisesRegex(AppError, 'budget reached'):
            self.call('assistant', 'resume', 'reviewer')
        self.assertEqual(item['state'], 'paused')
        self.agent.local.begin_turn(self.sid, self.agent.task_generation)
        result = self.call('assistant', 'resume', 'reviewer')
        self.assertEqual(result['budget']['workers'], 3)
        self.assertEqual(result['budget']['tokens'], 1536)

    def test_assistant_read_is_scoped_bounded_and_does_not_wake_codex(self):
        owner = self.assistants()
        self.call('assistant', 'assign', json.dumps({'id': 'reviewer', 'task': 'Review'}))
        item = self.client.assistants[owner, 'reviewer']
        item['state'] = 'completed'
        item['assignment']['latest_result'] = 'x' * 5000
        calls = len(self.agent.rpc.calls)
        result = self.call('assistant', 'read', 'reviewer')
        self.assertEqual(len(result['assignment']['result']), 2000)
        self.assertTrue(result['assignment']['truncated'])
        self.assertEqual(len(self.agent.rpc.calls), calls)
        self.assertEqual(self.client.assistant_calls[0][1], owner_key(self.store.directory, self.sid, self.root))

    def test_cancel_finished_cloud_turn_cancels_persistent_assignment(self):
        owner = self.assistants()
        self.call('assistant', 'assign', json.dumps({'id': 'reviewer', 'task': 'Review'}))
        self.agent.finish('idle', 'Local work continues')
        self.agent.cancel(self.sid)
        self.assertIn(('cancel_owner', owner), self.client.assistant_calls)

    def test_disabled_service_is_not_reenabled_by_delegation(self):
        self.client.status = lambda: {'running': False, 'service_enabled': False}
        with self.assertRaisesRegex(AppError, 'disabled'):
            self.spawn()
        self.assertEqual(self.client.started, 0)
        self.assertEqual(self.client.submitted, [])
    def test_cancel_after_cloud_turn_ends_cancels_local_jobs(self):
        self.spawn(); self.agent.finish('idle', 'Worker is running')
        self.agent.cancel(self.sid)
        self.assertTrue(self.client.cancelled)
    def test_quit_leaves_local_worker_service_running(self):
        self.spawn(); self.agent.close()
        self.assertEqual(self.client.cancelled, [])
    def test_finished_cloud_process_released_and_thread_resumable(self):
        rpc = self.agent.rpc
        self.agent.finish('idle', 'Done')
        self.agent._release_idle(self.agent.task_generation)
        self.assertIsNone(self.agent.rpc); self.assertTrue(rpc.closed)
        self.agent.submit(self.sid, 'Next explicit request', 'next-user-request')
        eventually(lambda: self.store.session(self.sid)['turn_id'])
        self.assertIn('thread/resume', [method for method,_ in self.agent.rpc.calls])
    def test_old_idle_timer_cannot_close_a_new_task(self):
        generation = self.agent.task_generation
        self.agent.finish('idle','Done')
        self.agent.submit(self.sid,'Next request','second-user-request')
        self.agent._release_idle(generation)
        self.assertIsNotNone(self.agent.rpc)

    def _delayed_previous_start(self, fail):
        self.agent.finish('idle', 'Initial task done')
        self.agent._release_idle(self.agent.task_generation)
        entered, release = threading.Event(), threading.Event()

        class DelayedRPC(FakeRPC):
            def __init__(self, *args):
                super().__init__(*args)
                self.turn_count = 0

            def call(self, method, params, **kwargs):
                if method != 'turn/start':
                    return super().call(method, params, **kwargs)
                self.calls.append((method, params))
                self.turn_count += 1
                number = self.turn_count
                if number == 1:
                    self.callback({'method': 'turn/completed', 'params': {
                        'threadId': 'thread-offline',
                        'turn': {'id': 'turn-first', 'status': 'completed'}}})
                    entered.set()
                    release.wait(3)
                    if fail:
                        raise AppError('Delayed previous turn/start failed')
                return {'turn': {'id': 'turn-first' if number == 1 else 'turn-second'}}

        self.agent.rpc_factory = DelayedRPC
        self.agent.submit(self.sid, 'First request', 'delayed-first-request')
        previous_starter = self.agent.start_thread
        try:
            self.assertTrue(entered.wait(1))
            self.assertFalse(self.agent.active)
            self.agent.submit(self.sid, 'Second request', 'delayed-second-request')
            eventually(lambda: self.store.session(self.sid)['turn_id'] == 'turn-second')
            release.set()
            previous_starter.join(2)
            self.assertFalse(previous_starter.is_alive())
            self.assertEqual(self.store.session(self.sid)['turn_id'], 'turn-second')
            self.assertEqual(self.store.session(self.sid)['status'], 'thinking')
            self.assertTrue(self.agent.active)
        finally:
            release.set()
            previous_starter.join(2)

    def test_delayed_previous_turn_start_reply_cannot_replace_current_turn(self):
        self._delayed_previous_start(False)

    def test_delayed_previous_turn_start_error_cannot_fail_current_turn(self):
        self._delayed_previous_start(True)

    def test_spawn_waits_for_control_socket_without_waiting_for_model(self):
        available, probed = threading.Event(), threading.Event()
        errors, results = [], []
        case = self

        class ColdClient(Client):
            def start(self):
                self.started += 1
                return {'running': False, 'starting': True, 'ready': False}

            def status(self):
                probed.set()
                return {'running': available.is_set(), 'ready': False}

            def submit(self, owner, task, **kwargs):
                case.assertTrue(available.is_set())
                return super().submit(owner, task, **kwargs)

        client = ColdClient()
        self.agent.local = LocalWorkers(lambda: client)
        self.agent.local.begin_turn(self.sid, self.agent.task_generation)

        def spawn():
            try:
                results.append(self.spawn())
            except Exception as error:
                errors.append(error)

        worker = threading.Thread(target=spawn)
        worker.start()
        try:
            self.assertTrue(probed.wait(1))
            self.assertEqual(client.submitted, [])
            available.set()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(errors, [])
            self.assertTrue(results[0]['execution_pending'])
        finally:
            available.set()
            worker.join(2)

    def test_new_turn_has_no_wall_clock_interrupt_timer(self):
        self.agent.finish('idle', 'Previous task complete')
        with patch('agent.threading.Timer') as timer:
            self.agent.submit(self.sid, 'Long-running work', 'no-wall-clock-deadline')
            self.agent.start_thread.join(2)
            self.assertFalse(self.agent.start_thread.is_alive())
            timer.assert_not_called()
            self.assertTrue(self.agent.active)
            self.assertFalse(hasattr(self.agent, 'task_timer'))
        self.agent.cancel(self.sid)
        self.assertFalse(self.agent.active)

    def test_tool_budget_interrupts_without_replay(self):
        self.agent._budget_expired(self.sid,self.agent.task_generation, '32-tool-call limit')
        self.assertEqual(self.store.session(self.sid)['status'],'interrupted')
        self.assertFalse(self.agent.active)
        self.assertEqual(sum(method=='turn/start' for method,_ in self.agent.rpc.calls),1)

    def test_quit_while_connection_starts_does_not_leave_cloud_process(self):
        self.agent.finish('idle', 'Initial task done')
        self.agent._release_idle(self.agent.task_generation)
        entered, release = threading.Event(), threading.Event()
        created = []
        def factory(command, cwd, callback):
            rpc = FakeRPC(command, cwd, callback)
            created.append(rpc); entered.set(); release.wait(3)
            return rpc
        self.agent.rpc_factory = factory
        self.agent.submit(self.sid, 'Close during startup', 'startup-close-test')
        self.assertTrue(entered.wait(1))
        closing = threading.Thread(target=self.agent.close); closing.start()
        eventually(lambda: self.agent.closed)
        release.set(); closing.join(3)
        self.assertFalse(closing.is_alive())
        self.assertTrue(created[0].closed)
        self.assertFalse(any(method=='turn/start' for method,_ in created[0].calls))
        self.assertIsNone(self.agent.rpc)


if __name__ == '__main__': unittest.main()
