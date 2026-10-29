"""Offline inference fixtures, not evidence of a real model's tool selection.
Natural questions reach inference. Only an emitted tool request reads hardware.
"""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from agent import Agent, AppError, Store
from hardware_info import CPU_ROOT, CPUINFO
from local_chat import LocalChatAgent
from local_runtime import native_tool_response

QUESTION = 'What CPU is it?'
PROPERTIES = {'ro.product.model': 'Fixture Phone', 'ro.soc.manufacturer': 'Fixture Maker',
              'ro.soc.model': 'Fixture Chip 482', 'ro.board.platform': 'fixture-board',
              'ro.product.cpu.abilist': 'arm64-v8a'}
FILES = {str(CPU_ROOT / 'present'): '0-3', str(CPU_ROOT / 'online'): '0-3',
         str(CPUINFO): '\n\n'.join('processor : ' + str(cpu) +
             '\nCPU implementer : 0x41\nCPU part : 0xd80\nCPU architecture : 8' for cpu in range(4))}

class InferenceFixture:
    engine_name = 'fixture-inference'
    engine_label = 'Offline inference fixture'
    def __init__(self, replies, ready=True):
        self.replies=iter(replies); self.ready=ready; self.status_calls=0
        self.submissions=[]; self.jobs={}
    def status(self):
        self.status_calls += 1
        return {'ready': self.ready, 'error': None if self.ready else 'Selected fixture model is unavailable',
                'model': {'backend': 'fixture', 'context_per_worker': 8192,
                          'max_output_tokens': 512, 'native_conversation_protocol': 1}}
    def submit(self, owner, prompt, **options):
        self.submissions.append({'owner': owner, 'prompt': prompt, 'options': options})
        reply=next(self.replies)
        if callable(reply): reply=reply(json.loads(prompt)['messages'])
        identifier='fixture-job-'+str(len(self.submissions)); self.jobs[identifier]=copy.deepcopy(reply)
        return {'id': identifier, 'state': 'queued'}
    def read(self, owner, identifier, wait=0, limit=4000):
        reply=self.jobs[identifier]
        return reply if isinstance(reply,dict) else {'state': 'completed', 'result': reply}
    def cancel(self, owner, identifier): return {'state': 'cancelled'}

def cpu_tool_reply(arguments=None):
    call={'id': 'model-cpu-call-482', 'type': 'function',
          'function': {'name': 'phone_control', 'arguments': json.dumps(arguments or {'action':'cpu'})}}
    normalized=native_tool_response('phone_control',json.dumps({'action':'cpu'}))
    return {'state':'completed','result':normalized,'native_call':call}

class InferenceHardwareTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name)/'project'; self.root.mkdir()
        self.store=Store(Path(self.temporary.name)/'state'); self.sid=self.store.create(self.root)
        self.agents=[]; self.phone_calls=[]
    def tearDown(self):
        for agent in self.agents: agent.close()
        self.store.db.close(); self.temporary.cleanup()
    def make(self,replies,ready=True):
        client=InferenceFixture(replies,ready)
        agent=LocalChatAgent(self.store,self.root,client=client)
        agent.phone.runner=self.property_read; self.agents.append(agent)
        return agent,client
    def property_read(self,argv,timeout=8,cancel_event=None):
        self.phone_calls.append(list(argv))
        if len(argv)!=2 or argv[0]!='/system/bin/getprop' or argv[1] not in PROPERTIES:
            raise AssertionError('Unexpected hardware command: '+repr(argv))
        return subprocess.CompletedProcess(argv,0,PROPERTIES[argv[1]]+'\n')
    def run_turn(self,agent,text=QUESTION,identifier='inference-hardware-001'):
        def read(path):
            if str(path) not in FILES: raise FileNotFoundError(str(path))
            return FILES[str(path)]
        with patch('hardware_info._read_text',side_effect=read), patch('hardware_info._policies',return_value=[]):
            agent.submit(self.sid,text,identifier); agent.start_thread.join(3)
        self.assertFalse(agent.start_thread.is_alive())
        return self.store.events(self.sid)
    def test_every_natural_cpu_question_uses_existing_start_path(self):
        agent=Agent(self.store,self.root,rpc_factory=Mock()); self.agents.append(agent)
        agent.phone.execute=Mock(side_effect=AssertionError('host must not intercept'))
        for index,text in enumerate((QUESTION,'what cpu is it buddy','what cpu is it faggot',
                "What's my processor?",'what is a CPU','what CPU should I buy',
                'what CPU is in my Mac','what CPU is it and reboot')):
            with self.subTest(text=text), patch.object(agent,'_start') as start:
                agent.submit(self.sid,text,'natural-cpu-'+str(index)); agent.start_thread.join(1)
                start.assert_called_once_with(self.sid,text,agent.task_generation)
                agent.finish('idle','fixture turn completed')
        agent.phone.execute.assert_not_called()
    def test_cpu_question_reaches_inference_without_hardware_shortcut(self):
        answer='The fixture inference chose to answer without a tool.'
        agent,client=self.make([answer])
        with patch('hardware_info.format_cpu_info',side_effect=AssertionError('no host-written natural reply')):
            events=self.run_turn(agent)
        self.assertEqual(len(client.submissions),1)
        messages=json.loads(client.submissions[0]['prompt'])['messages']
        self.assertEqual(messages[-1],{'role':'user','content':QUESTION})
        self.assertGreater(client.status_calls,0); self.assertEqual(self.phone_calls,[])
        self.assertFalse(any(event['kind']=='tool' for event in events))
        self.assertEqual(agent.last_answer,answer)

    @unittest.skipUnless(Path('/proc/sys/kernel/ostype').is_file(), 'requires Linux kernel metadata')
    def test_model_selected_command_reads_system_metadata_outside_workspace(self):
        """The generic executor supports investigation beyond reserved CPU queries."""
        argv = [sys.executable, '-c',
                'from pathlib import Path; print(Path("/proc/sys/kernel/ostype").read_text().strip())']
        arguments = json.dumps({'operation': 'command', 'argv': argv})
        call = {'id': 'model-system-probe', 'type': 'function',
                'function': {'name': 'workspace', 'arguments': arguments}}
        reply = {'state': 'completed', 'result': native_tool_response('workspace', arguments),
                 'native_call': call}
        observed = []

        def final_reply(messages):
            result = json.loads(messages[-1]['content'])
            observed.append(result)
            return 'The kernel reports ' + result['result']['output'].strip() + '.'

        agent, client = self.make([reply, final_reply])
        agent.set_action_policy('auto')
        events = self.run_turn(agent, 'Inspect the operating system on this device.')
        self.assertEqual(len(client.submissions), 2)
        self.assertEqual(self.phone_calls, [])
        self.assertEqual(observed[0]['result']['exit_code'], 0)
        self.assertEqual(observed[0]['result']['output'].strip(),
                         Path('/proc/sys/kernel/ostype').read_text().strip())
        decisions = [e['data'] for e in events if e['kind'] == 'approval_decision']
        self.assertTrue(decisions)
        self.assertTrue(decisions[0]['automatic'])
    def test_emitted_native_cpu_call_executes_then_returns_observed_receipt_to_inference(self):
        observed=[]
        def final_reply(messages):
            observed.append(messages[-1])
            chip=json.loads(messages[-1]['content'])['result']['soc']['model']
            return 'The observed processor is '+chip+'.'
        agent,client=self.make([cpu_tool_reply(),final_reply])
        agent.set_action_policy('ask')
        agent.wait_approval=Mock(side_effect=AssertionError('read-only hardware needs no additional approval'))
        with patch('hardware_info.format_cpu_info',side_effect=AssertionError('inference must produce answer')):
            events=self.run_turn(agent)
        self.assertEqual(self.phone_calls,[['/system/bin/getprop',key] for key in PROPERTIES])
        self.assertEqual(len(client.submissions),2)
        self.assertEqual(client.submissions[0]['options']['role'],'phone conversation assistant')
        self.assertNotIn('tool_choice',client.submissions[0]['options'])
        receipt=next(event for event in events if event['kind']=='tool_result')
        self.assertTrue(receipt['data']['success'])
        self.assertEqual(receipt['data']['result']['soc']['model'],'Fixture Chip 482')
        self.assertEqual(receipt['data']['result']['core_count'],4)
        self.assertEqual(receipt['data']['result']['sources']['soc.model']['property'],'ro.soc.model')
        self.assertEqual(observed[0]['role'],'tool'); self.assertEqual(observed[0]['name'],'phone_control')
        self.assertEqual(observed[0]['tool_call_id'],'model-cpu-call-482')
        model_view=json.loads(observed[0]['content'])['result']
        self.assertIn('cpu_identifiers',model_view['diagnostic_details_omitted'])
        self.assertNotIn('cpu_identifiers',model_view)
        self.assertIn('cpu_identifiers',receipt['data']['result'])
        self.assertEqual(model_view['sources']['soc.model'],{'property':'ro.soc.model'})
        self.assertIn('argv',receipt['data']['result']['sources']['soc.model'])
        self.assertEqual(json.loads(observed[0]['content'])['receipt_id'],receipt['data']['receipt_id'])
        self.assertEqual(agent.last_answer,'The observed processor is Fixture Chip 482.')
        self.assertEqual([e['kind'] for e in events if e['kind'] in ('tool_result','assistant')],['tool_result','assistant'])
        agent.wait_approval.assert_not_called()
    def test_unavailable_inference_does_not_fall_back_to_automatic_hardware_read(self):
        agent,client=self.make([],ready=False); events=self.run_turn(agent)
        self.assertGreater(client.status_calls,0); self.assertEqual(client.submissions,[])
        self.assertEqual(self.phone_calls,[]); self.assertEqual(self.store.session(self.sid)['status'],'failed')
        self.assertFalse(any(e['kind'] in ('tool','assistant') for e in events))
    def test_duplicate_request_does_not_repeat_inference_or_hardware_read(self):
        agent,client=self.make([cpu_tool_reply(),'Fixture Chip 482.']); self.run_turn(agent)
        calls=copy.deepcopy(self.phone_calls)
        self.assertTrue(agent.submit(self.sid,QUESTION,'inference-hardware-001')['duplicate'])
        self.assertEqual(len(client.submissions),2); self.assertEqual(self.phone_calls,calls)
        self.assertEqual(len([e for e in self.store.events(self.sid) if e['kind']=='user']),1)
    def test_malformed_native_cpu_arguments_never_reach_executor(self):
        agent,client=self.make([cpu_tool_reply({'action':'cpu','level':7})]); events=self.run_turn(agent)
        self.assertEqual(len(client.submissions),1); self.assertEqual(self.phone_calls,[])
        self.assertFalse(any(e['kind']=='tool' for e in events))
        self.assertEqual(self.store.session(self.sid)['status'],'failed')
    def test_failed_hardware_receipt_returns_to_inference_before_answer(self):
        observed=[]
        def answer(messages):
            observed.append(json.loads(messages[-1]['content']))
            return 'The hardware read failed; the processor is not confirmed.'
        agent,client=self.make([cpu_tool_reply(),answer])
        agent.phone.execute=Mock(side_effect=AppError('hardware fixture unavailable'))
        events=self.run_turn(agent)
        self.assertEqual(len(client.submissions),2); self.assertFalse(observed[0]['success'])
        self.assertIn('hardware fixture unavailable',observed[0]['result']['error'])
        self.assertFalse(next(e for e in events if e['kind']=='tool_result')['data']['success'])
        self.assertEqual(agent.last_answer,'The hardware read failed; the processor is not confirmed.')

if __name__=='__main__': unittest.main()
