import json
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch

from test_app import Base, eventually
from agent import AppError
from ostadix_tools import OstadixTools, paired_environment


class Client:
    def __init__(self, command, cwd, env):
        self.command, self.cwd, self.env = command, cwd, env
        self.alive = True
        self.info = {'serverInfo': {'name': 'offline MCP fixture'}}
        self.calls = []
        self.result = {'structuredContent': {'state': 'completed', 'exit_code': 0, 'result': 'actual fixture output'}}
        self.hook = None
        self.cancel_state = 'cancelled'

    def call(self, method, params, **kwargs):
        self.calls.append((method, params))
        if method == 'tools/list':
            return {'tools': [{'name': name, 'inputSchema': {'properties': {'cwd': {'type': 'string'}}}}
                              for name in ('o_execute', 'o_capabilities', 'o_guide', 'o_job_list', 'o_job_cancel')]}
        if params['name'] == 'o_job_list':
            return {'structuredContent': {'jobs': [{'job_id': 'job-1', 'state': 'running'}]}}
        if params['name'] == 'o_job_cancel':
            return {'structuredContent': {'job_id': params['arguments']['job_id'], 'state': self.cancel_state}}
        if self.hook:
            self.hook()
        return self.result

    def close(self):
        self.alive = False


class BridgeTests(Base):
    def setUp(self):
        super().setUp()
        self.agent.set_action_policy('ask')
        self.pair = {'root': '/private/managed/repo', 'revision': 'a'*40,
                     'backends': '/private/managed/repo/backends', 'binary': '/private/managed/mcp',
                     'binaries': {name: '/private/managed/bin/' + name for name in
                         ('O', 'o-cli', 'octl', 'o-info', 'olangc', 'o-live-host', 'ogit',
                          'o-node', 'o-registry', 'ostadix-device')}}
        self.clients = []
        def factory(*args):
            client = Client(*args)
            self.clients.append(client)
            return client
        self.agent.ostadix = OstadixTools(lambda: self.pair, factory)

    def command(self, name='o_execute', **params):
        return {'operation': 'command', 'argv': ['@ostadix', 'call', name, json.dumps(params)]}

    def result(self):
        return self.agent.rpc.sent[-1]['result']

    def test_metadata_needs_no_approval_and_connection_is_reused(self):
        self.start()
        for argv in (['@ostadix', 'status'], ['@ostadix', 'tools'], ['@ostadix', 'schema', 'o_execute'],
                     ['@ostadix', 'call', 'o_capabilities', '{"query":"runtime"}']):
            self.tool({'operation': 'command', 'argv': argv}).join(2)
            self.assertTrue(self.result()['success'])
        self.assertFalse(self.agent.approvals)
        self.assertEqual(len(self.clients), 1)
        self.assertEqual(self.clients[0].cwd, self.root)
        self.assertEqual(self.clients[0].env['OSTADIX_O_CLI_BIN'], self.pair['binaries']['o-cli'])

    def test_approval_precedes_execution_and_denial_never_calls_mcp(self):
        self.start()
        thread = self.tool(self.command(source='delete something'))
        eventually(lambda: bool(self.agent.approvals))
        self.assertEqual([c[0] for c in self.clients[0].calls], ['tools/list'])
        approval = next(iter(self.agent.approvals.values()))
        self.assertEqual(approval['arguments'], {'operation': 'ostadix', 'tool': 'o_execute',
            'arguments': {'source': 'delete something', 'cwd': str(self.root)},
            'repository': self.pair['root'], 'revision': self.pair['revision']})
        self.approve(False); thread.join(2)
        self.assertFalse(self.result()['success'])
        self.assertEqual(len(self.clients[0].calls), 1)

    def test_approved_result_and_failure_are_grounded_in_mcp_response(self):
        self.start()
        thread = self.tool(self.command(source='harmless'))
        self.approve(True); thread.join(2)
        self.assertTrue(self.result()['success'])
        self.assertIn('actual fixture output', self.result()['contentItems'][0]['text'])
        self.clients[0].result = {'structuredContent': {'exit_code': 17, 'state': 'failed', 'stderr': 'real failure'}}
        thread = self.tool(self.command(source='harmless'))
        self.approve(True); thread.join(2)
        self.assertFalse(self.result()['success'])
        self.assertIn('real failure', self.result()['contentItems'][0]['text'])

    def test_duplicate_mcp_text_does_not_truncate_model_evidence_and_full_result_is_saved(self):
        self.start()
        self.agent.ostadix.status(self.sid, self.root)
        evidence = 'native output ' + '42. ' * 1650 + ' end of verified evidence'
        structured = {'action': 'execute', 'state': 'completed', 'exit_code': 0,
                      'result': {'ok': True, 'stdout': evidence}}
        response = {'content': [{'type': 'text', 'text': json.dumps(structured, indent=2)}],
                    'structuredContent': structured}
        self.assertGreater(len(json.dumps(response)), 12000)
        self.clients[0].result = response

        thread = self.tool(self.command(source='harmless'))
        self.approve(True)
        thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertTrue(self.result()['success'])
        model_text = self.result()['contentItems'][0]['text']
        self.assertLess(len(model_text), 12000)
        model_result = json.loads(model_text)
        self.assertEqual(model_result['result']['structuredContent'], structured)
        self.assertEqual(model_result['result']['structuredContent']['result']['stdout'], evidence)
        self.assertNotIn('content', model_result['result'])
        saved = [event for event in self.store.events(self.sid) if event['kind'] == 'tool_result'][-1]
        self.assertEqual(saved['data']['result']['result'], response)
        self.assertEqual(self.clients[0].result, response)

    def test_meaningful_mcp_text_is_preserved_in_model_and_saved_result(self):
        self.start()
        self.agent.ostadix.status(self.sid, self.root)
        structured = {'state': 'completed', 'exit_code': 0, 'result': 'actual fixture output'}
        cases = [
            [{'type': 'text', 'text': 'Compiler note: execution used the Python backend.'}],
            [{'type': 'text', 'text': json.dumps({'additional_evidence': 'compiler note'})}],
            [{'type': 'text', 'text': json.dumps(structured)},
             {'type': 'text', 'text': 'Additional execution evidence.'}],
        ]
        for content in cases:
            with self.subTest(content=content):
                response = {'content': content, 'structuredContent': structured}
                self.clients[0].result = response
                thread = self.tool(self.command(source='harmless'))
                self.approve(True)
                thread.join(2)

                self.assertFalse(thread.is_alive())
                self.assertTrue(self.result()['success'])
                self.assertEqual(json.loads(self.result()['contentItems'][0]['text'])['result'], response)
                saved = [event for event in self.store.events(self.sid) if event['kind'] == 'tool_result'][-1]
                self.assertEqual(saved['data']['result']['result'], response)

    def test_scope_and_runtime_override_rejection(self):
        self.start()
        (self.root/'escape').symlink_to(self.directory, target_is_directory=True)
        for params in ({'cwd': str(self.directory)}, {'cwd': '../'}, {'cwd': 'escape'},
                       {'env': {'O_LANG_ROOT': '/old'}}, {'env': {'PATH': '/old'}}):
            self.tool(self.command(**params)).join(2)
            self.assertFalse(self.result()['success'], params)
        self.tool(self.command('o_capabilities', cwd='/outside')).join(2)
        self.assertFalse(self.result()['success'])
        self.assertFalse(self.agent.approvals)
        self.assertEqual(len(self.clients[0].calls), 1)

    def test_lost_connection_is_reported_before_explicit_reconnect(self):
        self.start()
        self.agent.ostadix.status(self.sid, self.root)
        self.clients[0].alive = False
        self.tool(self.command(source='must not replay')).join(2)
        self.assertFalse(self.result()['success'])
        self.assertIn('No execution was replayed', self.result()['contentItems'][0]['text'])
        self.assertEqual(len(self.clients), 1)
        self.agent.ostadix.status(self.sid, self.root)
        self.assertEqual(len(self.clients), 2)

    def test_background_job_cancels_after_model_turn_finishes(self):
        self.start()
        self.agent.ostadix.status(self.sid, self.root)
        self.clients[0].result = {'structuredContent': {'state': 'running', 'job_id': 'job-1'}}
        thread = self.tool(self.command(source='harmless', background=True))
        self.approve(True); thread.join(2)
        self.assertTrue(json.loads(self.result()['contentItems'][0]['text'])['execution_pending'])
        self.agent.finish('idle', 'Job remains pending')
        self.agent.cancel(self.sid)
        self.assertFalse(self.agent.ostadix.has_jobs(self.sid))
        self.assertEqual(self.clients[0].calls[-1][1]['name'], 'o_job_cancel')

    def test_cancel_during_job_submission_discovers_and_cancels_job(self):
        self.start()
        self.agent.ostadix.status(self.sid, self.root)
        self.clients[0].result = {'structuredContent': {'state': 'running', 'job_id': 'job-1'}}
        self.clients[0].hook = self.agent.cancelled.set
        thread = self.tool(self.command(source='harmless', background=True))
        self.approve(True); thread.join(2)
        self.assertFalse(self.result()['success'])
        self.assertFalse(self.agent.ostadix.has_jobs(self.sid))
        self.assertEqual([c[1]['name'] for c in self.clients[0].calls[1:]], ['o_execute', 'o_job_list', 'o_job_cancel'])

    def test_unconfirmed_job_cancellation_remains_visible_and_tracked(self):
        self.start()
        entry = self.agent.ostadix.connection(self.sid, self.root)
        entry['jobs'].add('job-1'); self.clients[0].cancel_state = 'running'
        self.agent.finish('idle', 'Still running')
        self.agent.cancel(self.sid)
        self.assertTrue(self.agent.ostadix.has_jobs(self.sid))
        self.assertTrue(any('unconfirmed' in e['data'].get('text', '') for e in self.store.events(self.sid)))

    def test_cancel_other_conversation_job_does_not_stop_active_task(self):
        self.start()
        other = self.store.create(self.root)
        entry = self.agent.ostadix.connection(other, self.root); entry['jobs'].add('job-1')
        with patch.object(self.agent, 'kill_process') as kill:
            self.agent.process = object()
            self.agent.cancel(other)
            self.agent.process = None
            kill.assert_not_called()
        self.assertFalse(self.agent.cancelled.is_set())
        self.assertTrue(self.agent.active)

    def test_stale_turn_and_generation_cannot_execute(self):
        self.start()
        message = {'id': 100, 'params': {'threadId': 'thread-offline', 'turnId': 'old-turn',
                    'tool': 'workspace', 'arguments': self.command(source='never')}}
        self.agent._tool_request(message)
        self.assertFalse(self.result()['success'])
        message['params']['turnId'] = 'turn-offline'
        self.agent._tool_request(message, generation=self.agent.task_generation-1)
        self.assertFalse(self.result()['success'])
        self.assertFalse(self.clients)

    def test_closed_pool_cannot_spawn_and_resume_keeps_preference(self):
        self.start()
        started = [params for method, params in self.agent.rpc.calls if method == 'thread/start'][0]
        self.assertIn('OSTADIX MCP is your preferred tool', started['developerInstructions'])
        self.assertEqual([tool['name'] for tool in started['dynamicTools']], ['workspace'])
        self.agent.finish('idle', 'Done'); self.agent.loaded_thread = None
        self.agent.submit(self.sid, 'resume', 'resume-123456')
        eventually(lambda: any(m == 'thread/resume' for m, _ in self.agent.rpc.calls))
        resumed = [params for method, params in self.agent.rpc.calls if method == 'thread/resume'][0]
        self.assertIn('OSTADIX MCP is your preferred tool', resumed['developerInstructions'])
        self.agent.ostadix.close()
        with self.assertRaises(AppError): self.agent.ostadix.status(self.sid, self.root)
        self.assertFalse(self.clients)

    def test_secret_environment_not_forwarded_to_mcp(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'fixture-secret', 'GITHUB_TOKEN': 'fixture-token',
                                    'LD_PRELOAD': '/termux/lib/libtermux-exec-ld-preload.so'}):
            env = paired_environment(self.pair)
        self.assertNotIn('OPENAI_API_KEY', env)
        self.assertNotIn('GITHUB_TOKEN', env)
        self.assertNotIn('LD_PRELOAD', env)
        self.assertEqual(env['OSTADIX_RUNTIME_PATH_MODE'], 'inherited-only')


class LaunchTests(unittest.TestCase):
    def test_explicit_startup_update_verifies_ostadix_bm_pair(self):
        from launch import prepare_ostadix
        updated = {'revision': 'b'*40, 'upstream': 'https://github.com/lostadi/ostadix-BM.git'}
        with patch('ostadix_pair.update_pair', return_value=updated) as update, \
             patch('ostadix_pair.load_pair') as load:
            self.assertEqual(prepare_ostadix(required_update=True), updated)
            update.assert_called_once()
            self.assertTrue(callable(update.call_args.kwargs['progress']))
            load.assert_not_called()

    def test_offline_startup_uses_verified_ostadix_bm_pair_without_upstream_or_build(self):
        from launch import prepare_ostadix
        with patch('ostadix_pair.update_pair') as update, \
             patch('ostadix_pair.load_pair', return_value={'revision': 'a'*40}) as load, \
             patch('builtins.print') as output:
            self.assertEqual(prepare_ostadix(offline=True)['revision'], 'a'*40)
            load.assert_called_once_with()
            update.assert_not_called()
            self.assertIn('verified installed OSTADIX-BM revision', str(output.call_args_list))

    def test_missing_or_invalid_pair_requires_repair_when_update_unavailable(self):
        from launch import prepare_ostadix
        from ostadix_pair import PairError
        for reason in ('missing', 'binary does not match the recorded build'):
            with self.subTest(reason=reason), \
                 patch('ostadix_pair.update_pair', side_effect=PairError('network unavailable')) as update, \
                 patch('ostadix_pair.load_pair', side_effect=PairError(reason)) as load, \
                 patch('builtins.print'):
                with self.assertRaisesRegex(AppError, 'c --update-ostadix'):
                    prepare_ostadix()
                update.assert_called_once()
                load.assert_called_once_with()

    def test_explicit_update_calls_updater_and_reports_failure(self):
        from launch import prepare_ostadix
        from ostadix_pair import PairError
        with patch('ostadix_pair.update_pair', return_value={'revision': 'b'*40}) as update, \
             patch('ostadix_pair.load_pair') as load:
            self.assertEqual(prepare_ostadix(required_update=True)['revision'], 'b'*40)
            update.assert_called_once()
            self.assertTrue(callable(update.call_args.kwargs['progress']))
            update.side_effect = PairError('fetch failed')
            with self.assertRaisesRegex(AppError, 'fetch failed'):
                prepare_ostadix(required_update=True)
            load.assert_not_called()


if __name__ == '__main__': unittest.main()
