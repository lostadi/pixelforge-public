"""Offline terminal tests; Android speech is simulated, never a paid model call."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import AppError
from launch import saved_workspace
from server import Server
from terminal import NativeSpeech, TerminalSession, TerminalUI, display_text, main, spoken_text
from test_app import FakeRPC, eventually


class Audio:
    def __init__(self):
        self.mode = None
        self.stopping = False
        self.events = queue.Queue()
        self.calls = []
    def start(self, mode, text=None):
        self.calls.append((mode, text))
    def stop(self):
        self.calls.append(('stop', None))


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'workspace'; self.root.mkdir()
        (self.root / 'answer.txt').write_text('The actual answer is 42.')
        self.state = self.base / 'state'
        self.patcher = patch('agent.codex_command', return_value=['offline-fixture'])
        self.patcher.start()
        self.session = TerminalSession(self.root, self.state, FakeRPC)
        self.audio = Audio(); self.output = []
        self.ui = TerminalUI(self.session, self.audio, self.output.append)
        self.ui.companion = Mock()
        self.ui.companion.messages.return_value = {'messages': [], 'eligible': False}
        self.ui.restore()
    def tearDown(self):
        self.session.close()
        self.patcher.stop()
        self.temp.cleanup()
    def start(self):
        self.ui.handle('Read answer.txt')
        eventually(lambda: self.session.store.session(self.session.sid)['turn_id'])
    def tool(self, arguments):
        self.session.agent.on_event({'id':123,'method':'item/tool/call',
                                     'params':{'threadId':'thread-offline','tool':'workspace','arguments':arguments}})
    def test_actual_read_and_spoken_final_share_agent_results(self):
        self.start()
        self.tool({'operation':'read','path':'answer.txt'})
        eventually(lambda: any(e['kind']=='tool_result' for e in self.session.store.events(self.session.sid)))
        self.ui.pump()
        self.assertTrue(any('The actual answer is 42.' in line for line in self.output))
        self.assertEqual(self.audio.calls, [])
        self.session.agent.on_event({'method':'item/completed','params':{'item':{'type':'agentMessage','text':'The actual answer is 42.'}}})
        self.ui.pump(); self.assertEqual(self.audio.calls, [])
        self.session.agent.on_event({'method':'turn/completed','params':{'turn':{'status':'completed'}}})
        self.ui.pump()
        self.assertEqual(self.audio.calls, [('speaking','The actual answer is 42.')])

    def test_full_completed_answer_and_hear_include_every_paragraph(self):
        answer = 'Opening sentence.\n\n' + ('The detailed explanation continues here. ' * 150) + '\n\nFinal sentence.'
        self.start()
        self.session.agent.on_event({'method': 'item/completed', 'params': {
            'item': {'type': 'agentMessage', 'text': answer}}})
        self.ui.pump()
        self.assertEqual(self.audio.calls, [])
        self.session.agent.on_event({'method': 'turn/completed', 'params': {'turn': {'status': 'completed'}}})
        self.ui.pump()
        self.assertEqual(self.audio.calls, [('speaking', answer)])
        self.ui.handle('/hear')
        self.assertEqual(self.audio.calls, [('speaking', answer), ('speaking', answer)])

    def test_phone_package_results_are_displayed_in_full(self):
        packages = ['com.example.app' + str(i).zfill(3) for i in range(457)]
        self.session.store.event(self.session.sid, 'tool_result', {
            'success': True, 'result': {'packages': packages, 'count': len(packages), 'truncated': False}})
        self.ui.pump()
        shown = '\n'.join(self.output)
        self.assertIn('457 total', shown)
        for package in packages:
            self.assertIn(package, shown)
        self.assertFalse(self.session.agent.active)

    def test_apps_control_returns_complete_list_without_a_model_call(self):
        packages = ['com.example.first', 'com.example.last']
        result = {'packages': packages, 'count': 2, 'truncated': False}
        with patch.object(self.session.agent.phone, 'execute', return_value=result) as phone:
            self.ui.handle('/apps example')
            eventually(lambda: not self.ui.phone_events.empty())
            self.ui.pump()
        self.assertEqual(phone.call_args.args[2], ['@phone', 'apps', 'example'])
        self.assertIn('\n'.join(packages), '\n'.join(self.output))
        self.assertFalse(self.session.agent.active)

    def test_evidence_agent_command_routes_locally(self):
        with patch.object(self.ui, 'local_action') as local:
            self.ui.handle('/agent ask Read answer.txt')
            local.assert_called_once_with('evidence_ask', 'Read answer.txt')
        self.assertFalse(self.session.agent.active)

    def test_evidence_agent_read_and_cancel_routes(self):
        ident = 'a' * 32
        with patch.object(self.ui, 'local_action') as local:
            self.ui.handle('/agent read ' + ident)
            local.assert_called_with('evidence_read', ident)
            self.ui.handle('/agent cancel ' + ident)
            local.assert_called_with('evidence_cancel', ident)

    def local_assistant_client(self):
        from local_workers import LocalWorkers
        from test_local_workers import AssistantClient
        client = AssistantClient()
        self.session.agent.local = LocalWorkers(lambda: client)
        self.start()
        return client

    def assistant_tool(self, *arguments):
        return self.session.agent.execute({
            'operation': 'command', 'argv': ['@local', 'assistant', *arguments]})

    def test_agent_created_assistant_event_keeps_reply_and_prompt_working(self):
        self.local_assistant_client()
        self.assistant_tool('create', json.dumps({'name': 'Listener'}))
        event = next(e for e in self.session.store.events(self.session.sid)
                     if e['kind'] == 'local_assistant')
        self.assertNotIn('label', event['data'])
        self.assertNotIn('assignment', event['data'])
        answer = 'Listener is created and waiting.'
        self.session.agent.on_event({'method': 'item/completed', 'params': {
            'item': {'type': 'agentMessage', 'text': answer}}})
        self.session.agent.on_event({'method': 'turn/completed', 'params': {
            'turn': {'status': 'completed'}}})

        self.ui.pump()

        self.assertIn('Assistant Listener · waiting', self.output)
        self.assertEqual(self.output.count('Ostadix: ' + answer), 1)
        self.assertEqual(self.audio.calls, [('speaking', answer)])
        self.assertFalse(self.ui.quit)
        self.ui.handle('/inspect answer.txt')
        self.ui.pump()
        self.assertTrue(any('The actual answer is 42.' in line for line in self.output))

    def test_agent_assistant_compact_result_and_truncation_are_displayed(self):
        client = self.local_assistant_client()
        self.assistant_tool('create', json.dumps({'name': 'Listener'}))
        self.assistant_tool('assign', json.dumps({'id': 'Listener', 'task': 'Review this text'}))
        owner = self.session.agent.local.owner(self.session.agent, self.session.sid)
        item = client.assistants[owner, 'Listener']
        item['state'] = 'completed'
        item['assignment'].update(state='completed', completed_steps=2,
                                  latest_result='A' * 2001)
        result = self.assistant_tool('read', 'Listener')
        self.assertEqual(result['assignment']['result'], 'A' * 2000)
        self.assertTrue(result['assignment']['truncated'])

        self.ui.pump()

        self.assertIn('Assistant Listener · queued · 0/2 steps', self.output)
        self.assertIn('Assistant Listener · completed · 2/2 steps', self.output)
        self.assertIn('A' * 2000, self.output)
        self.assertTrue(any('worker reached its output limit' in line for line in self.output))

    def test_poll_assistant_event_keeps_label_result_and_worker_reference(self):
        # This assertion concerns event deduplication; exclude unrelated,
        # asynchronously polled resident-model status from its output stream.
        self.ui.local = None
        self.session.store.event(self.session.sid, 'local_assistant', {
            'id': 'listener-id', 'name': 'Listener', 'state': 'completed',
            'label': 'Listener · completed · 2/2 steps',
            'assignment': {'latest_result': 'Saved local review',
                           'latest_result_truncated': True,
                           'steps': [{'job_id': 'worker-id'}]}})
        self.ui.pump()
        self.assertIn('Assistant Listener · completed · 2/2 steps', self.output)
        self.assertIn('Saved local review', self.output)
        self.assertIn('Full saved worker output: /workers read worker-id', self.output)
        self.assertTrue(any('worker reached its output limit' in line for line in self.output))
        before = list(self.output)
        self.ui.pump()
        self.assertEqual(self.output, before)
    def test_destructive_request_waits_for_exact_live_approval(self):
        self.session.agent.set_action_policy('ask')
        self.start(); self.tool({'operation':'delete','path':'answer.txt'})
        eventually(lambda: self.session.approvals())
        self.ui.pump()
        self.assertTrue((self.root/'answer.txt').exists())
        self.assertTrue(any('AWAITING APPROVAL' in line for line in self.output))
        with self.assertRaises(AppError): self.ui.handle('/approve')
        ident = self.session.approvals()[0]['id'][:8]
        self.ui.handle('/deny ' + ident)
        eventually(lambda: not self.session.agent.tool_lock.locked())
        self.assertTrue((self.root/'answer.txt').exists())
        with self.assertRaises(AppError): self.ui.handle('/approve ' + ident)
        self.tool({'operation':'delete','path':'answer.txt'})
        eventually(lambda: self.session.approvals())
        self.ui.handle('/approve ' + self.session.approvals()[0]['id'][:8])
        eventually(lambda: not (self.root/'answer.txt').exists())
    def test_stop_audio_does_not_cancel_task(self):
        self.start(); self.ui.handle('/stop')
        self.assertTrue(self.session.agent.active)
        self.assertFalse(self.session.agent.cancelled.is_set())
        self.assertEqual(self.audio.calls, [('stop',None)])
        self.ui.handle('/cancel')
        eventually(lambda: not self.session.agent.active)
        self.assertEqual(self.audio.calls, [('stop',None)])
    def test_cancellation_terminates_real_approved_process(self):
        self.session.agent.set_action_policy('ask')
        self.start()
        self.tool({'operation':'command','argv':[sys.executable,'-c','import time; time.sleep(30)']})
        eventually(lambda: self.session.approvals())
        self.ui.handle('/approve ' + self.session.approvals()[0]['id'][:8])
        eventually(lambda: self.session.agent.process)
        process = self.session.agent.process
        self.ui.handle('/cancel')
        eventually(lambda: process.poll() is not None)
        eventually(lambda: not self.session.agent.tool_lock.locked())
    def test_voice_is_saved_draft_and_cannot_approve(self):
        self.audio.events.put(('heard','/approve deadbeef'))
        self.ui.pump()
        self.assertEqual(self.session.draft(), '/approve deadbeef')
        self.assertFalse(self.session.agent.active)
        self.ui.handle('')
        eventually(lambda: self.session.store.session(self.session.sid)['turn_id'])
        events = self.session.store.events(self.session.sid)
        self.assertTrue(any(e['kind']=='user' and e['data']['text']=='/approve deadbeef' for e in events))
        self.assertFalse(any(e['kind']=='approval_decision' for e in events))
    def test_close_reopen_keeps_selection_thread_draft_and_history(self):
        self.start(); self.session.agent.finish('idle','Saved answer')
        self.session.draft('Unsent recognized speech')
        sid = self.session.sid
        self.session.close()
        self.session = TerminalSession(self.root,self.state,FakeRPC)
        self.assertEqual(self.session.sid,sid)
        self.assertEqual(self.session.store.session(sid)['thread_id'],'thread-offline')
        self.assertEqual(self.session.draft(),'Unsent recognized speech')
        ui = TerminalUI(self.session,self.audio,self.output.append)
        ui.restore();ui.pump()
        self.assertEqual(self.audio.calls, [])
        self.assertEqual(ui.last_answer,'Saved answer')
        self.assertEqual(saved_workspace(self.state),str(self.root))
    def test_browser_history_opens_in_terminal(self):
        sid = self.session.sid
        self.session.close()
        server = Server(0,self.root,self.state,FakeRPC)
        server.store.event(sid,'finished',{'status':'idle','text':'From browser'})
        server.server_close();server.store.db.close()
        self.session = TerminalSession(self.root,self.state,FakeRPC)
        self.assertEqual(self.session.sid,sid)
        self.assertTrue(any(e['data'].get('text')=='From browser' for e in self.session.store.events(sid)))
    def test_second_interface_cannot_reset_live_state(self):
        self.session.store.update(self.session.sid,status='executing')
        with self.assertRaises(AppError): TerminalSession(self.root,self.state,FakeRPC)
        with self.assertRaises(AppError): Server(0,self.root,self.state,FakeRPC)
        self.assertEqual(self.session.store.session(self.session.sid)['status'],'executing')
    def test_network_failure_is_visible_and_session_recoverable(self):
        self.start()
        self.session.agent.on_event({'method':'local/disconnected','params':{}})
        self.ui.pump()
        self.assertTrue(any('disconnected' in line for line in self.output))
        self.assertTrue(any(e['kind']=='user' for e in self.session.store.events(self.session.sid)))
        self.assertEqual(self.session.store.session(self.session.sid)['thread_id'],'thread-offline')
    def restore_failed_session(self):
        self.start()
        self.session.agent.finish('failed', 'Codex disconnected. Saved history remains.')
        sid = self.session.sid
        self.session.draft('Unsent retry')
        self.session.close()
        self.session = TerminalSession(self.root, self.state, FakeRPC)
        self.output.clear()
        self.ui = TerminalUI(self.session, self.audio, self.output.append, audio=False)
        self.ui.restore(); self.ui.pump()
        return sid
    def test_restored_failure_is_history_until_explicit_retry(self):
        sid = self.restore_failed_session()
        self.assertEqual(self.session.sid, sid)
        self.assertEqual(self.session.store.session(sid)['status'], 'failed')
        self.assertEqual(self.session.store.session(sid)['thread_id'], 'thread-offline')
        self.assertEqual(self.session.draft(), 'Unsent retry')
        self.assertIsNone(self.session.agent.rpc)
        self.assertFalse(self.session.agent.active)
        self.assertIn('[ready for a new request; previous task failed]', self.output)
        self.assertNotIn('[disconnected / failed; saved history retained]', self.output)
        self.assertIn('Previous task failed (saved history): Codex disconnected. Saved history remains.', self.output)
        self.assertIn('Codex connection is checked when you send a new request.', self.output)
        self.assertEqual(self.audio.calls, [])
        self.assertEqual(len([e for e in self.session.store.events(sid) if e['kind'] == 'user']), 1)
    def test_retry_launcher_failure_is_current_even_without_rpc(self):
        self.restore_failed_session()
        self.output.clear()
        with patch('agent.codex_command', side_effect=AppError('Codex launch failed now')):
            self.ui.handle('Retry explicitly')
            eventually(lambda: not self.session.agent.active)
        self.ui.pump()
        self.assertIsNone(self.session.agent.rpc)
        self.assertIn('Task failed: Codex launch failed now', self.output)
        self.assertIn('[disconnected / failed; saved history retained]', self.output)
        self.assertNotIn('[ready for a new request; previous task failed]', self.output)
    def test_retry_after_saved_failure_resumes_and_reports_live_disconnect(self):
        self.restore_failed_session()
        self.output.clear()
        self.start()
        self.ui.pump()
        self.assertIn('[thinking]', self.output)
        self.assertIn('thread/resume', [method for method, _ in self.session.agent.rpc.calls])
        self.session.agent.on_event({'method':'local/disconnected','params':{}})
        self.ui.pump()
        self.assertIn('[disconnected / failed; saved history retained]', self.output)
        self.assertNotIn('[ready for a new request; previous task failed]', self.output)
    def test_inspect_is_local_and_scope_cannot_expand(self):
        self.ui.handle('/inspect answer.txt');self.ui.pump()
        self.assertTrue(any('42' in line for line in self.output))
        self.assertIsNone(self.session.agent.rpc)
        with self.assertRaises(AppError): self.ui.handle('/inspect ../private')
        with self.assertRaises(AppError): self.ui.handle('/new ..')
    def test_history_paginates_full_logs(self):
        for i in range(305): self.session.store.event(self.session.sid,'log',{'text':f'line-{i}'})
        self.ui.handle('/logs')
        self.assertTrue(any('line-304' in line for line in self.output))
    def test_audio_errors_do_not_submit_requests(self):
        self.audio.events.put(('error','Microphone permission denied'))
        self.ui.pump()
        self.assertFalse(self.session.agent.active)
        self.assertEqual(self.session.draft(),'')
        self.assertTrue(any('permission denied' in line for line in self.output))

    def test_mute_preference_survives_session_close_and_reopen(self):
        self.assertTrue(self.ui.audio)
        self.ui.handle('/mute')
        self.session.close()
        self.session = TerminalSession(self.root, self.state, FakeRPC)
        self.ui = TerminalUI(self.session, self.audio, self.output.append)
        self.assertFalse(self.ui.audio)
        self.session.store.event(self.session.sid, 'finished',
                                 {'status': 'idle', 'text': 'A quiet answer.'})
        self.ui.pump()
        self.assertFalse(any(mode == 'speaking' for mode, _ in self.audio.calls))

    def test_explicit_audio_override_does_not_replace_saved_preference(self):
        for saved, explicit in ((True, False), (False, True)):
            with self.subTest(saved=saved, explicit=explicit):
                self.ui.handle('/unmute' if saved else '/mute')
                temporary = TerminalUI(self.session, self.audio, self.output.append,
                                       audio=explicit)
                self.assertEqual(temporary.audio, explicit)
                restored = TerminalUI(self.session, self.audio, self.output.append)
                self.assertEqual(restored.audio, saved)

    def test_normal_terminal_launch_uses_saved_mute_preference(self):
        self.ui.handle('/mute')
        self.session.close()
        observed = []
        try:
            with patch('terminal.os.geteuid', return_value=10000), \
                    patch('terminal.sys.stdin.isatty', return_value=True), \
                    patch('terminal.TerminalUI.run', autospec=True,
                          side_effect=lambda ui: observed.append(ui.audio)):
                main(['--workspace', str(self.root), '--state-dir', str(self.state)])
        finally:
            self.session = TerminalSession(self.root, self.state, FakeRPC)
        self.assertEqual(observed, [False])

    def test_unmute_is_saved_and_enables_spoken_answers(self):
        self.ui.handle('/mute')
        self.ui.handle('/unmute')
        self.ui = TerminalUI(self.session, self.audio, self.output.append)
        self.assertTrue(self.ui.audio)
        self.session.store.event(self.session.sid, 'finished',
                                 {'status': 'idle', 'text': 'An audible answer.'})
        self.ui.pump()
        self.assertIn(('speaking', 'An audible answer.'), self.audio.calls)

    def test_mute_leaves_microphone_capture_running(self):
        self.ui.handle('/talk')
        self.audio.mode = 'listening'
        self.ui.handle('/mute')
        self.assertFalse(self.ui.audio)
        self.assertNotIn(('stop', None), self.audio.calls)
        self.audio.mode = None
        self.audio.events.put(('heard', 'Keep my dictated request'))
        self.ui.pump()
        self.assertEqual(self.session.draft(), 'Keep my dictated request')
        self.assertFalse(self.session.agent.active)

    def test_mute_stops_current_speaker_playback(self):
        self.audio.mode = 'speaking'
        self.ui.handle('/mute')
        self.assertEqual(self.audio.calls, [('stop', None)])

    def test_recognition_stays_with_original_conversation_after_new(self):
        source = self.session.sid
        self.ui.handle('/talk')
        self.audio.mode = 'listening'
        self.ui.handle('/new')
        target = self.session.sid
        self.assertNotEqual(source, target)
        self.session.draft('Draft belonging to the new conversation')
        self.output.clear()
        self.audio.mode = None
        self.audio.events.put(('heard', 'Capture belonging to the original conversation'))
        self.ui.pump()
        self.assertEqual(self.session.draft(), 'Draft belonging to the new conversation')
        self.assertEqual(self.session.store.setting('terminal_draft_' + source),
                         'Capture belonging to the original conversation')
        self.assertIn('/sessions', '\n'.join(self.output))
        self.assertFalse(self.session.agent.active)
        self.session.select(source)
        self.assertEqual(self.session.draft(), 'Capture belonging to the original conversation')

    def test_recognition_stays_with_original_conversation_after_use(self):
        source = self.session.sid
        self.ui.handle('/new')
        target = self.session.sid
        self.session.draft('Existing destination draft')
        self.session.select(source)
        self.ui.handle('/talk')
        self.audio.mode = 'listening'
        number = next(index for index, session in enumerate(self.session.sessions(), 1)
                      if session['id'] == target)
        self.ui.handle('/use ' + str(number))
        self.output.clear()
        self.audio.mode = None
        self.audio.events.put(('heard', 'The original conversation was recording'))
        self.ui.pump()
        self.assertEqual(self.session.sid, target)
        self.assertEqual(self.session.draft(), 'Existing destination draft')
        self.assertEqual(self.session.store.setting('terminal_draft_' + source),
                         'The original conversation was recording')
        self.assertIn('/sessions', '\n'.join(self.output))
        self.assertFalse(self.session.agent.active)

    def test_repeated_enter_keeps_existing_recognition(self):
        self.audio.mode = 'listening'
        before = list(self.output)
        for _ in range(3):
            self.ui.handle('')
        self.assertEqual(self.audio.calls, [])
        self.assertFalse(self.session.agent.active)
        self.assertEqual(self.output, before)
        self.audio.mode = None
        self.audio.events.put(('heard', 'Read answer.txt'))
        self.ui.pump()
        self.assertEqual(self.session.draft(), 'Read answer.txt')
        self.ui.handle('')
        eventually(lambda: self.session.store.session(self.session.sid)['turn_id'])

    def test_enter_during_playback_does_not_start_microphone(self):
        self.audio.mode = 'speaking'
        self.ui.handle('')
        self.assertEqual(self.audio.calls, [])
        self.assertTrue(any('listen when this reply finishes' in line for line in self.output))
        self.assertEqual(self.ui.pending_talk_sid, self.session.sid)

    def test_stopping_audio_stays_visible_until_ready(self):
        self.audio.stopping = True
        self.ui.pump()
        self.assertEqual(self.output[-1], '[stopping audio]')
        self.ui.handle('')
        self.assertEqual(self.audio.calls, [])
        self.audio.stopping = False
        self.ui.handle('')
        self.assertEqual(self.audio.calls, [('listening', None)])


class SpeechTests(unittest.TestCase):
    def test_already_stopped_android_service_is_not_an_audio_error(self):
        speech = NativeSpeech(); speech.mode = 'speaking'
        with patch('terminal.shutil.which', return_value='/fixture/am'), patch('terminal.subprocess.run') as run:
            run.return_value.returncode = 0
            run.return_value.stdout = 'Stopping service: Intent {}\nService not stopped: was not running.\n'
            run.return_value.stderr = ''
            speech.stop()
            eventually(lambda: not speech.stopping)
            self.assertTrue(speech.events.empty())

    def test_failed_android_stop_still_reports_uncertain_audio(self):
        speech = NativeSpeech(); speech.mode = 'speaking'
        with patch('terminal.shutil.which', return_value='/fixture/am'), patch('terminal.subprocess.run') as run:
            run.return_value.returncode = 1
            run.return_value.stdout = 'Service not stopped: was not running.'
            run.return_value.stderr = 'Permission denied'
            speech.stop()
            eventually(lambda: not speech.stopping)
            self.assertIn('did not confirm', speech.events.get_nowait()[1])

    def test_native_timeout_reports_error_and_requests_service_stop(self):
        speech=NativeSpeech()
        with patch('terminal.shutil.which',return_value='/fixture/speech'), patch('terminal.subprocess.Popen') as popen, \
             patch.object(speech,'kill'), patch('terminal.subprocess.run') as run:
            popen.return_value.communicate.side_effect=[subprocess.TimeoutExpired('speech',40),('','')]
            run.return_value.returncode=0;run.return_value.stdout='Service stopped';run.return_value.stderr=''
            speech.start('listening')
            eventually(lambda: not speech.mode and not speech.stopping)
            kind,text=speech.events.get_nowait()
            self.assertEqual(kind,'error');self.assertIn('timed out',text)
            self.assertEqual(run.call_args.args[0][-1],NativeSpeech.COMPONENTS['listening'])
    def test_android_error_json_is_not_a_recognized_request(self):
        speech=NativeSpeech()
        with patch('terminal.shutil.which',return_value='/fixture/speech'), patch('terminal.subprocess.Popen') as popen, \
             patch.object(speech,'kill'), patch('terminal.subprocess.run') as run:
            popen.return_value.communicate.return_value=('{"error":"permission denied"}','')
            popen.return_value.returncode=0
            run.return_value.returncode=0;run.return_value.stdout='Service stopped';run.return_value.stderr=''
            speech.start('listening')
            eventually(lambda: not speech.mode and not speech.stopping)
            self.assertEqual(speech.events.get_nowait()[0],'error')
            self.assertTrue(speech.events.empty())
    def test_sanitizes_terminal_output_and_preserves_code_in_speech(self):
        self.assertNotIn('\x1b', display_text('\x1b]52;c;payload\x07\u202e'))
        self.assertNotIn('\u202e', display_text('\u202e'))
        self.assertEqual(spoken_text('Use `get_value` here.\n\n```python\nprint(42)\n```\n\nFinished.'),
                         'Use get_value here.\n\nprint(42)\n\nFinished.')

    def test_full_speech_removes_markdown_decoration_without_losing_content(self):
        self.assertEqual(spoken_text('# Steps\n\n- **First**: read [the guide](https://example.test).\n- Second: continue.'),
                         'Steps\n\nFirst: read the guide.\nSecond: continue.')

    def test_long_single_paragraph_is_not_cut_at_600_characters(self):
        answer = 'Keep reading the entire paragraph. ' * 200 + 'The end.'
        self.assertEqual(spoken_text(answer), answer)
    def test_real_child_recognition_result_reaches_draft_queue(self):
        with tempfile.TemporaryDirectory() as directory:
            script=Path(directory)/'speech'
            script.write_text('#!'+sys.executable+'\nprint("Read answer.txt")\n');script.chmod(0o700)
            speech=NativeSpeech()
            with patch('terminal.shutil.which',return_value=str(script)):
                speech.start('listening')
                eventually(lambda: not speech.mode)
            self.assertEqual(speech.events.get_nowait(),('heard','Read answer.txt'))
            speech.close()
    def test_stop_reaches_android_service_not_only_cli(self):
        speech=NativeSpeech();speech.mode='speaking'
        with patch('terminal.shutil.which',return_value='/fixture/am'), patch('terminal.subprocess.run') as run:
            run.return_value.returncode=0;run.return_value.stdout='Service stopped';run.return_value.stderr=''
            speech.stop()
            eventually(lambda: run.called)
            self.assertEqual(run.call_args.args[0],['/fixture/am','stopservice','-n',NativeSpeech.COMPONENTS['speaking']])
    def test_missing_native_microphone_is_actionable(self):
        speech=NativeSpeech()
        with patch('terminal.shutil.which',return_value=None), self.assertRaisesRegex(AppError,'keyboard dictation'):
            speech.start('listening')


if __name__ == '__main__': unittest.main(verbosity=2)
