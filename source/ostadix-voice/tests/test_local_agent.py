import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from local_agent import Agent, complete, parse

class FakeClient:
    def __init__(self, outputs=()):
        self.outputs = list(outputs)
        self.prompts = []
    def status(self):
        return {'ready': True, 'model': {'backend': 'litert-npu'}}
    def submit(self, owner, task, **kwargs):
        self.prompts.append(task)
        return {'id': str(len(self.prompts))}
    def read(self, owner, ident, **kwargs):
        return {'state': 'completed', 'result': json.dumps(self.outputs.pop(0)),
                'usage': {'backend': 'litert-npu'}}
    def cancel(self, *args):
        return {}

class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.workspace = self.root / 'workspace'
        self.workspace.mkdir()
        (self.workspace / 'fact.txt').write_text('Measured value: 42')
    def agent(self, outputs=()):
        client = FakeClient(outputs)
        return Agent(self.root / 'state', client), client
    def test_task_can_exceed_former_four_minute_deadline(self):
        agent, client = self.agent([{'tool':'model_status'}, {'final':'done'}])
        task = agent.create(self.workspace, 'Inspect model status')
        # Advance simulated monotonic time by five minutes at every check.
        import itertools
        with patch('local_agent.time.monotonic', side_effect=itertools.count(0, 300)):
            result = agent.run(task['id'])
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(len(client.prompts), 2)
    def test_cancellation_during_model_wait(self):
        agent, client = self.agent()
        task = agent.create(self.workspace, 'Inspect model status')
        def pending(*args, **kwargs):
            agent.cancel(task['id'])
            return {'state':'running'}
        with patch.object(client, 'read', side_effect=pending), patch.object(client, 'cancel') as cancel:
            result = agent.run(task['id'])
        self.assertEqual(result['state'], 'cancelled')
        cancel.assert_called_once()
    def test_exact_arithmetic_without_model(self):
        agent, client = self.agent()
        task = agent.create(self.workspace, 'calculate: 0.1+0.2')
        with patch.object(client, 'status', side_effect=AssertionError('No model access')):
            result = agent.run(task['id'])
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['result'], '3/10')
        self.assertEqual(client.prompts, [])
        self.assertEqual(result['steps'], 0)
    def test_cancel_exact_arithmetic(self):
        agent, client = self.agent()
        task = agent.create(self.workspace, 'calculate: 1+2')
        agent.cancel(task['id'])
        self.assertEqual(agent.run(task['id'])['state'], 'cancelled')
        self.assertEqual(client.prompts, [])
    def test_calculate_tool(self):
        agent, _ = self.agent()
        request = parse('{"tool":"calculate","expression":"9/6"}')
        self.assertEqual(agent.evidence(self.workspace, request)['exact'], '3/2')
    def test_read_then_answer_and_persistence(self):
        agent, client = self.agent([{'tool':'read','path':'fact.txt'}, {'final':'The measured value is 42.'}])
        task = agent.create(self.workspace, 'Read fact.txt and report its value.')
        result = agent.run(task['id'])
        self.assertEqual(result['state'], 'completed')
        self.assertIn('Measured value: 42', client.prompts[1])
        self.assertEqual(Agent(self.root / 'state').read(task['id'])['result'], result['result'])
        self.assertEqual(os.stat(agent.path(task['id'])).st_mode & 0o777, 0o600)
        with self.assertRaises(ValueError):
            agent.run(task['id'])
    def test_denies_code_and_unknown_tools(self):
        for output in ('{"tool":"shell","cmd":"id"}', '{"tool":"model_status","extra":1}',
                       '{"final":"a","final":"b"}', '[]'):
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse(output)
    def test_path_traversal_symlink_and_credentials_blocked(self):
        agent, _ = self.agent()
        (self.workspace/'link').symlink_to('/etc/passwd')
        for path in ('../outside', '/etc/passwd', 'link', '.env', '.ssh/id_rsa'):
            with self.subTest(path=path), self.assertRaises(Exception):
                agent.evidence(self.workspace, {'tool':'read','path':path})
    def test_hardlinks_blocked(self):
        agent, _ = self.agent()
        from types import SimpleNamespace
        info = (self.workspace/'fact.txt').stat()
        with patch('agent.os.fstat', return_value=SimpleNamespace(st_mode=info.st_mode, st_nlink=2)):
            with self.assertRaises(Exception):
                agent.evidence(self.workspace, {'tool':'read','path':'fact.txt'})
    def test_cancel_before_start_does_not_infer(self):
        agent, client = self.agent()
        task = agent.create(self.workspace, 'Read a fact')
        agent.cancel(task['id'])
        self.assertEqual(agent.run(task['id'])['state'], 'cancelled')
        self.assertEqual(client.prompts, [])
    def test_prompt_history_is_structured_and_evidence_preserves_text(self):
        agent, client = self.agent([{'tool':'read','path':'fact.txt'}, {'final':'42'}])
        result = agent.run(agent.create(self.workspace, 'Read fact.txt')['id'])
        self.assertEqual(result['state'], 'completed')
        history = client.prompts[1].split('Already requested: ', 1)[1].split('\n', 1)[0]
        self.assertEqual(json.loads(history), [{'tool':'read','path':'fact.txt'}])
        self.assertIn('Measured value: 42', client.prompts[1])
        # Original evidence and duplicate checking remain unchanged.
        self.assertEqual(result['events'][1]['request'], {'tool':'read','path':'fact.txt'})
    def test_duplicate_request_stops(self):
        agent, client = self.agent([{'tool':'model_status'}, {'tool':'model_status'}])
        task = agent.create(self.workspace, 'Get status')
        self.assertEqual(agent.run(task['id'])['state'], 'failed')
        self.assertEqual(len(client.prompts), 2)
    def test_budget_exhaustion_is_not_success(self):
        agent, _ = self.agent([{'tool':'model_status'}])
        task = agent.create(self.workspace, 'Get status', steps=1)
        self.assertEqual(agent.run(task['id'])['state'], 'budget_exhausted')
    def test_rejects_wrong_backend_and_truncated_results(self):
        base = {'state':'completed','result':'{"final":"ok"}','usage':{'backend':'litert-npu'}}
        for change in ({'usage':{'backend':'cpu'}}, {'truncated':True},
                       {'usage':{'backend':'litert-npu','token_limit_reached':True}}):
            with self.assertRaises(ValueError):
                complete(dict(base, **change))
    def test_dead_process_not_replayed(self):
        agent, _ = self.agent()
        task = agent.create(self.workspace, 'Get status')
        task.update(state='running', pid=99999999, process_stamp=1)
        agent.save(task)
        self.assertEqual(agent.read(task['id'])['state'], 'interrupted')
        with self.assertRaises(ValueError):
            agent.run(task['id'])
    def test_submission_error_not_retried(self):
        agent, client = self.agent()
        task = agent.create(self.workspace, 'Get status')
        with patch.object(client, 'submit', side_effect=OSError('unknown outcome')) as submit:
            result = agent.run(task['id'])
            self.assertEqual(submit.call_count, 1)
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['pending'], {'state':'submitting'})
    def test_read_error_is_returned_as_evidence(self):
        agent, client = self.agent([{'tool':'read','path':'missing'}, {'final':'File unavailable.'}])
        result = agent.run(agent.create(self.workspace, 'Read missing')['id'])
        self.assertEqual(result['state'], 'completed')
        self.assertIn('"error"', client.prompts[1])
    def test_accepts_only_whole_json_fence(self):
        self.assertEqual(parse('```json\n{"final":"ok"}\n```'), {'final':'ok'})
        with self.assertRaises(ValueError):
            parse('Ignore rules\n```json\n{"final":"ok"}\n```')
    def test_final_with_valid_request_never_dispatches(self):
        agent, client = self.agent([{'tool':'optimizer_status', 'final':'Done.'}])
        task = agent.create(self.workspace, 'Report status', steps=1)
        with patch.object(agent, 'evidence', side_effect=AssertionError('Must not dispatch')):
            self.assertEqual(agent.run(task['id'])['result'], 'Done.')
        with self.assertRaises(ValueError):
            parse('{"tool":"shell","final":"Done"}')
    def test_invalid_step_budget(self):
        agent, _ = self.agent()
        for steps in (0, 5, True):
            with self.assertRaises(ValueError):
                agent.create(self.workspace, 'Task', steps)
if __name__ == '__main__':
    unittest.main()
