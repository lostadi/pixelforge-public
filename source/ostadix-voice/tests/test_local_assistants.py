"""Durable assistant tests with a fake local pool; no model/cloud/OS actions."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_assistants import AssistantCoordinator, AssistantError, LIMITS


class Client:
    def __init__(self):
        self.calls = []
        self.jobs = {}
        self.lose_submit_response = False

    def submit(self, owner, task, max_tokens=512, role='assistant'):
        ident = uuid.uuid4().hex
        self.calls.append(('submit', owner, task, max_tokens, role))
        self.jobs[ident] = {'id': ident, 'owner': owner, 'state': 'queued', 'result': '', 'max_tokens': max_tokens, 'usage': {}}
        if self.lose_submit_response:
            raise TimeoutError('Submit acknowledgement was lost')
        return dict(self.jobs[ident])

    def read(self, owner, job_id, wait=0, limit=4000):
        self.calls.append(('read', owner, job_id, wait, limit))
        row = self.jobs[job_id]
        if row['owner'] != owner:
            raise RuntimeError('Wrong owner')
        return dict(row)

    def cancel(self, owner, job_id=None):
        self.calls.append(('cancel', owner, job_id))
        row = self.jobs[job_id]
        if row['owner'] != owner:
            raise RuntimeError('Wrong owner')
        row['state'] = 'cancelled'
        return {'cancelled': 1}

    def finish(self, job_id, text, state='completed', **extra):
        self.jobs[job_id].update(state=state, result=text, usage={'completion_tokens': 12}, **extra)


class AssistantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = Path(self.temp.name) / 'assistants'
        self.client = Client()
        self.coordinator = AssistantCoordinator(self.state, lambda: self.client)
        self.assistant = self.coordinator.create('owner-a', 'helper', 'reviewer')
        self.ident = self.assistant['id']

    def tearDown(self):
        self.coordinator.close()
        self.temp.cleanup()

    def assign(self, **kwargs):
        return self.coordinator.assign('owner-a', assistant_id=self.ident, task='Return the sum of 19 and 23.', **kwargs)

    def current(self):
        return self.coordinator.read('owner-a', assistant_id=self.ident)

    def job(self):
        return self.current()['assignment']['steps'][-1]['job_id']

    def test_identity_creation_and_idle_polling_make_no_model_calls(self):
        for _ in range(4):
            self.coordinator.tick()
        self.assertEqual(self.current()['state'], 'waiting')
        self.assertTrue(self.current()['waiting_for_assignment'])
        self.assertEqual(self.client.calls, [])
        self.assertEqual(self.coordinator.list('owner-b'), [])

    def test_names_and_ids_are_owner_scoped(self):
        other = self.coordinator.create('owner-b', 'helper')
        self.assertNotEqual(other['id'], self.ident)
        with self.assertRaises(AssistantError):
            self.coordinator.read('owner-b', assistant_id=self.ident)
        with self.assertRaises(AssistantError):
            self.coordinator.assign('owner-b', self.ident, 'Cross-owner task')
        with self.assertRaises(AssistantError):
            self.coordinator.create('owner-a', 'helper')
        self.assertEqual(self.coordinator.read('owner-a', 'helper')['id'], self.ident)
        self.assertEqual(self.coordinator.read('owner-a', self.ident[:8])['id'], self.ident)

    def test_two_step_draft_review_stops_at_fixed_budget(self):
        self.assign()
        self.coordinator.tick()
        first = self.job()
        self.client.finish(first, 'The sum is 42.')
        self.coordinator.tick()
        self.assertEqual(self.current()['state'], 'queued')
        self.coordinator.tick()
        second = self.job()
        self.assertNotEqual(first, second)
        submit = [call for call in self.client.calls if call[0] == 'submit'][-1]
        self.assertIn('Review the prior draft', submit[2])
        self.assertIn('The sum is 42.', submit[2])
        self.assertLess(len(submit[2]), 6000)
        self.client.finish(second, '42')
        self.coordinator.tick()
        for _ in range(5):
            self.coordinator.tick()
        state = self.current()
        self.assertEqual(state['state'], 'completed')
        self.assertTrue(state['waiting_for_assignment'])
        self.assertEqual(state['memory'], '42')
        self.assertEqual(state['assignment']['steps_submitted'], 2)
        self.assertEqual(state['assignment']['tokens_requested'], 1024)
        self.assertEqual(len([call for call in self.client.calls if call[0] == 'submit']), 2)

    def test_pause_finishes_current_step_but_does_not_start_next(self):
        self.assign()
        self.coordinator.tick()
        job_id = self.job()
        paused = self.coordinator.pause('owner-a', assistant_id=self.ident)
        self.assertEqual(paused['state'], 'pause_pending')
        self.client.finish(job_id, 'Draft response')
        self.coordinator.tick()
        self.assertEqual(self.current()['state'], 'paused')
        for _ in range(3):
            self.coordinator.tick()
        self.assertEqual(len([call for call in self.client.calls if call[0] == 'submit']), 1)
        self.coordinator.guide('owner-a', assistant_id=self.ident, text='Reply with one number only.')
        self.coordinator.resume('owner-a', assistant_id=self.ident)
        self.coordinator.tick()
        submit = [call for call in self.client.calls if call[0] == 'submit'][-1]
        self.assertIn('Reply with one number only.', submit[2])
        self.assertEqual(self.current()['assignment']['tokens_requested'], 1024)

    def test_cancel_owner_stops_generation_and_future_steps_only_for_owner(self):
        other = self.coordinator.create('owner-b', 'other')
        self.assign()
        self.coordinator.assign('owner-b', other['id'], 'Other task')
        self.coordinator.tick()
        owned_job = self.job()
        other_job = self.coordinator.read('owner-b', other['id'])['assignment']['steps'][-1]['job_id']
        self.assertTrue(self.coordinator.has_work('owner-a'))
        result = self.coordinator.cancel_owner('owner-a')
        self.assertEqual(result['cancelled'], 1)
        self.assertFalse(self.coordinator.has_work('owner-a'))
        self.assertEqual(self.client.jobs[owned_job]['state'], 'cancelled')
        self.assertEqual(self.client.jobs[other_job]['state'], 'queued')
        self.coordinator.tick()
        self.assertEqual(self.current()['state'], 'cancelled')

    def test_restart_preserves_identity_but_never_replays_assignment(self):
        self.coordinator.guide('owner-a', self.ident, 'Use plain language.')
        self.assign()
        self.coordinator.tick()
        original_job = self.job()
        self.coordinator.close()
        self.coordinator = AssistantCoordinator(self.state, lambda: self.client)
        self.coordinator.tick()
        state = self.current()
        self.assertEqual(state['guidance'], 'Use plain language.')
        self.assertEqual(state['state'], 'interrupted')
        self.assertEqual(state['assignment']['steps'][0]['job_id'], original_job)
        self.assertEqual(len([call for call in self.client.calls if call[0] == 'submit']), 1)
        with self.assertRaises(AssistantError):
            self.coordinator.resume('owner-a', self.ident)
        self.assign(max_steps=1)
        self.coordinator.tick()
        self.assertEqual(len([call for call in self.client.calls if call[0] == 'submit']), 2)

    def test_lost_submit_acknowledgement_is_not_retried(self):
        self.client.lose_submit_response = True
        self.assign()
        self.coordinator.tick()
        self.coordinator.tick()
        state = self.current()
        self.assertEqual(state['state'], 'interrupted')
        self.assertEqual(state['assignment']['steps_submitted'], 1)
        self.assertEqual(state['assignment']['tokens_requested'], 512)
        self.assertEqual(state['assignment']['steps'][0]['state'], 'submitting')
        self.assertEqual(len([call for call in self.client.calls if call[0] == 'submit']), 1)

    def test_failed_generation_does_not_trigger_another_step(self):
        self.assign(max_steps=4)
        self.coordinator.tick()
        self.client.finish(self.job(), '', state='failed', error='Engine stopped')
        self.coordinator.tick()
        self.coordinator.tick()
        self.assertEqual(self.current()['state'], 'failed')
        self.assertEqual(len([call for call in self.client.calls if call[0] == 'submit']), 1)

    def test_limits_reject_unbounded_work_and_boolean_budgets(self):
        for options in ({'max_steps': 5}, {'max_steps': True}, {'max_steps': 4, 'max_tokens': 1024}, {'max_tokens': 1025}):
            with self.subTest(options=options), self.assertRaises(AssistantError):
                self.assign(**options)
        with self.assertRaises(AssistantError):
            self.coordinator.assign('owner-a', self.ident, 'x' * 2001)
        self.assertEqual(self.client.calls, [])

    def test_untrusted_generated_actions_remain_text_and_memory_is_bounded(self):
        self.assign(max_steps=1)
        self.coordinator.tick()
        payload = '{"tool":"shell","command":"echo this must never execute"}\n' + 'x' * 3000
        self.client.finish(self.job(), payload, result_truncated=True)
        self.coordinator.tick()
        result = self.current()
        self.assertEqual(result['state'], 'completed')
        self.assertLessEqual(len(result['memory']), LIMITS['memory_chars'])
        self.assertLessEqual(len(result['assignment']['latest_result']), LIMITS['step_result_chars'])
        self.assertTrue(result['assignment']['latest_result_truncated'])
        self.assertTrue(all(call[0] in ('submit', 'read') for call in self.client.calls))
        self.assertEqual(result['assignment']['steps'][0]['job_id'], self.job())

    def test_second_coordinator_cannot_interrupt_first(self):
        self.assign()
        with self.assertRaises(AssistantError):
            AssistantCoordinator(self.state, lambda: self.client)
        self.assertEqual(self.current()['state'], 'queued')
        self.assertEqual(os.stat(self.state / 'assistants.sqlite3').st_mode & 0o777, 0o600)

    def test_time_budget_expiry_cancels_known_job_and_stops_workflow(self):
        self.assign()
        self.coordinator.tick()
        current_job = self.job()
        with self.coordinator.db:
            self.coordinator.db.execute('UPDATE assignments SET deadline=0')
        self.coordinator.tick()
        self.assertEqual(self.current()['state'], 'interrupted')
        self.assertEqual(self.client.jobs[current_job]['state'], 'cancelled')
        self.assertEqual(len([call for call in self.client.calls if call[0] == 'submit']), 1)

    def test_resident_scheduler_completes_explicit_work_and_then_waits(self):
        self.assign(max_steps=1, max_tokens=64)
        self.coordinator.start()
        deadline = time.monotonic() + 3
        while not self.client.jobs and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(len(self.client.jobs), 1)
        self.client.finish(next(iter(self.client.jobs)), '42')
        while self.current()['state'] != 'completed' and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertEqual(self.current()['state'], 'completed')
        calls = len(self.client.calls)
        time.sleep(.6)
        self.assertEqual(len(self.client.calls), calls)
        self.coordinator.close()
        self.assertFalse(self.coordinator.thread.is_alive())

    def test_unicode_lists_and_batch_cancellation_fit_socket_frame(self):
        for index in range(16):
            assistant = self.assistant if index == 0 else self.coordinator.create('owner-a', 'worker-' + str(index))
            self.coordinator.guide('owner-a', assistant['id'], '🦙' * 600)
            current = self.coordinator.assign('owner-a', assistant['id'], '🦙' * 2000, max_steps=4)
            assignment_id = current['assignment']['id']
            with self.coordinator.db:
                self.coordinator.db.execute('UPDATE assistants SET memory=? WHERE id=?', ('🦙' * 1000, assistant['id']))
                self.coordinator.db.execute("UPDATE assignments SET state='completed',latest_result=?,error=? WHERE id=?", ('🦙' * 1400, '🦙' * 500, assignment_id))
                for ordinal in range(1, 5):
                    self.coordinator.db.execute("INSERT INTO steps(assignment_id,ordinal,job_id,state,result,usage) VALUES(?,?,?,'completed',?,?)",
                                                (assignment_id, ordinal, uuid.uuid4().hex, '🦙' * 1400, json.dumps({'text': '🦙' * 300})))
        listing = self.coordinator.list('owner-a')
        frame = json.dumps({'ok': True, 'result': listing}).encode() + b'\n'
        self.assertLess(len(frame), 262144)
        self.assertEqual(len(listing), 16)
        self.assertNotIn('memory', listing[0])
        self.assertNotIn('guidance', listing[0])
        self.assertNotIn('task', listing[0]['assignment'])
        self.assertEqual(len(listing[0]['assignment']['latest_result']), 400)
        self.assertTrue(listing[0]['assignment']['latest_result_truncated'])
        self.assertNotIn('usage', listing[0]['assignment']['steps'][0])
        detailed = self.current()
        self.assertEqual(len(detailed['memory']), 1000)
        self.assertEqual(len(detailed['assignment']['task']), 2000)
        with self.coordinator.db:
            self.coordinator.db.execute("UPDATE assignments SET state='paused'")
        cancelled = self.coordinator.cancel_owner('owner-a')
        frame = json.dumps({'ok': True, 'result': cancelled}).encode() + b'\n'
        self.assertLess(len(frame), 262144)
        self.assertEqual(cancelled['cancelled'], 16)
        self.assertTrue(all(item['state'] == 'cancelled' for item in cancelled['assistants']))
        self.assertFalse(self.coordinator.has_work('owner-a'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
