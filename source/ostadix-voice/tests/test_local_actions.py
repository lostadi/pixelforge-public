import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from local_agent import Agent, parse
from local_actions import prepare, execute, snapshot, syntax_check
from test_local_agent import FakeClient

class ActionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ws = self.root/'workspace'
        self.ws.mkdir()
        self.file = self.ws/'calc.py'
        self.original = 'def add(a, b):\n    return a - b\n'
        self.file.write_text(self.original)
        self.edit = {'tool':'replace','path':'calc.py','old':'return a - b','new':'return a + b'}
    def staged(self):
        client = FakeClient([{'tool':'read','path':'calc.py'},self.edit,{'final':'Fixed.'}])
        agent = Agent(self.root/'state', client)
        task = agent.create(self.ws, 'Fix add', steps=3)
        value = agent.run(task['id'])
        return agent, client, value
    def test_stage_does_not_write_then_approved_resume_preserves_budget(self):
        agent, client, value = self.staged()
        self.assertEqual(value['state'], 'awaiting_approval')
        self.assertEqual(self.file.read_text(),self.original)
        applied=agent.apply(value['id'],value['proposal']['digest'])
        self.assertTrue(applied['events'][-1]['result']['verification']['passed'])
        self.assertIn('return a + b',self.file.read_text())
        final=agent.run(value['id'])
        self.assertEqual(final['state'],'completed')
        self.assertEqual(len(client.prompts),3)
        self.assertEqual(final['steps'],3)
        with self.assertRaises(ValueError):
            agent.apply(value['id'],value['proposal']['digest'])
    def test_can_reread_after_approved_edit(self):
        client=FakeClient([{'tool':'read','path':'calc.py'},self.edit,
                           {'tool':'read','path':'calc.py'},{'final':'Reviewed changed file.'}])
        agent=Agent(self.root/'state',client)
        task=agent.create(self.ws,'Fix and reread calc.py')
        staged=agent.run(task['id'])
        applied=agent.apply(task['id'],staged['proposal']['digest'])
        self.assertIn('return a + b',applied['events'][-1]['result']['content'])
        result=agent.run(task['id'])
        self.assertEqual(result['state'],'completed')
        self.assertEqual(result['verification']['edits_applied'],1)
        self.assertEqual(result['verification']['tests_run'],0)
        self.assertIsNone(result['verification']['last_test_passed'])
        self.assertIn('return a + b',client.prompts[-1])

    def test_wrong_digest_does_not_write(self):
        agent,_,value=self.staged()
        with self.assertRaises(ValueError):
            agent.apply(value['id'],'0'*64)
        self.assertEqual(self.file.read_text(),self.original)
    def test_stale_file_rejected(self):
        agent,_,value=self.staged()
        self.file.write_text('User changed this')
        result=agent.apply(value['id'],value['proposal']['digest'])
        self.assertEqual(result['state'],'failed')
        self.assertEqual(self.file.read_text(),'User changed this')
    def test_denial_is_final(self):
        agent,_,value=self.staged()
        digest=value['proposal']['digest']
        self.assertEqual(agent.deny(value['id'],digest)['state'],'cancelled')
        with self.assertRaises(ValueError):
            agent.apply(value['id'],digest)
        with self.assertRaises(ValueError):
            agent.run(value['id'])
    def test_cancel_before_apply(self):
        agent,_,value=self.staged()
        agent.cancel(value['id'])
        with self.assertRaises(ValueError):
            agent.apply(value['id'],value['proposal']['digest'])
        self.assertEqual(self.file.read_text(),self.original)
        self.assertEqual(agent.read(value['id'])['state'],'cancelled')
    def test_unread_edit_rejected(self):
        with self.assertRaises(ValueError):
            prepare({'workspace':str(self.ws),'events':[]},self.edit)
    def test_ambiguous_edit_rejected(self):
        self.file.write_text(self.original*2)
        value={'workspace':str(self.ws),'events':[{'kind':'evidence',
               'request':{'tool':'read','path':'calc.py'},'result':{'content':self.original*2,'more':False}}]}
        with self.assertRaises(ValueError):
            prepare(value,self.edit)
    def test_mixed_final_action_rejected(self):
        for tool in ('replace','test'):
            with self.assertRaises(ValueError):
                parse(json.dumps({'tool':tool,'path':'calc.py','final':'Done'}))
    def test_syntax_check_never_executes_source(self):
        marker=self.ws/'marker'
        content='open('+repr(str(marker))+',"w").write("bad")'
        self.assertTrue(syntax_check('calc.py',content)['passed'])
        self.assertFalse(marker.exists())
        self.assertFalse(syntax_check('calc.py','def broken(')['passed'])
    def test_test_action_requires_host_execution_and_records_failure(self):
        test=self.ws/'test_calc.py'
        test.write_text('assert 2 + 2 == 5, "expected test failure"\n')
        proposal=prepare({'workspace':str(self.ws),'events':[]},{'tool':'test','path':'test_calc.py'})
        self.assertEqual(proposal['action']['argv'][-1],str(test))
        result=execute(proposal,self.ws)
        self.assertFalse(result['passed'])
        self.assertIn('expected test failure',result['output'])
    def test_modified_proposal_rejected(self):
        agent,_,value=self.staged()
        prepared=value['proposal']
        prepared['action']['content']='Unexpected change'
        with self.assertRaises(ValueError):
            execute(prepared,self.ws)
    def test_protected_test_path_rejected(self):
        for path in ('../test.py','/tmp/test.py','.env.py'):
            with self.assertRaises(Exception):
                prepare({'workspace':str(self.ws),'events':[]},{'tool':'test','path':path})
    def test_failed_check_feedback_can_produce_repair(self):
        client=FakeClient([{'tool':'test','path':'calc.py'},{'tool':'read','path':'calc.py'},
                           self.edit,{'final':'Edited; syntax checked.'}])
        agent=Agent(self.root/'state',client)
        task=agent.create(self.ws,'Check and fix add')
        staged=agent.run(task['id'])
        with patch('local_actions.run_test',return_value={'kind':'python_test','passed':False,'output':'add(2,3) returned -1'}):
            agent.apply(task['id'],staged['proposal']['digest'])
        next_stage=agent.run(task['id'])
        self.assertIn('returned -1',client.prompts[1])
        agent.apply(task['id'],next_stage['proposal']['digest'])
        self.assertEqual(agent.run(task['id'])['state'],'completed')
        self.assertEqual(len(client.prompts),4)
if __name__ == '__main__':
    unittest.main()
