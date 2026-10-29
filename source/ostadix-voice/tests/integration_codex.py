"""Real Codex app-server + real executor + fake loopback model. No paid calls.

The fixed, harmless command or managed OSTADIX MCP call is explicitly approved
by this test harness. No arbitrary generated action is auto-approved. Run with
--ostadix-mcp, --python-smoke, or the legacy --backends PATH.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Agent, RPC, Store


SMOKE_SOURCE = 'python^(\n__oval_result__ = 1 + 1\n)_python\n'


def nested_objects(value):
    """Find tool results inside actual Responses inputs, including JSON text."""
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from nested_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from nested_objects(child)
    elif isinstance(value, str) and value.lstrip().startswith(('{', '[')):
        try:
            decoded = json.loads(value)
        except ValueError:
            return
        yield from nested_objects(decoded)


def actual_mcp_two(value, revision):
    """Require native completed-job evidence, not a digit in logs or metadata."""
    if (not isinstance(value, dict) or value.get('mcp_tool') != 'o_execute'
            or value.get('exit_code') != 0 or value.get('repository_revision') != revision):
        return False
    result = value.get('result')
    if not isinstance(result, dict) or result.get('isError'):
        return False
    job = result.get('structuredContent')
    if (not isinstance(job, dict) or job.get('action') != 'execute'
            or job.get('state') != 'completed' or job.get('exit_code') != 0):
        return False
    native = job.get('result')
    return (isinstance(native, dict) and native.get('ok') is True
            and native.get('type') == 'number'
            and native.get('value') == {'t': 'number', 'v': {'kind': 'int', 'v': '2'}})


class Model(BaseHTTPRequestHandler):
    def log_message(self, *_): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if getattr(self.server, 'http_error', None):
            payload=json.dumps({'error':{'message':'Offline fixture: authentication failed','type':'authentication_error'}}).encode()
            self.send_response(self.server.http_error);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)
            return
        self.server.requests.append(body)
        index = len(self.server.requests)-1
        actions = self.server.actions
        if index < len(actions):
            output = [{'id': 'fc_'+str(index), 'type':'custom_tool_call', 'call_id':'call_'+str(index),
                       'name':'exec', 'namespace':'functions', 'input':actions[index]}]
            # Newer Codex versions honor code_mode=false and advertise direct
            # tools. Exercise the catalog actually sent by the installed CLI.
            if any(tool.get('name') == 'workspace' for tool in body.get('tools', [])):
                action = actions[index]
                name, arguments = action.removeprefix('text(await tools.').split('(', 1)
                value = json.JSONDecoder().raw_decode(arguments)[0]
                output = [{'id': 'fc_'+str(index), 'call_id': 'call_'+str(index),
                           'type': 'function_call' if name == 'workspace' else 'custom_tool_call',
                           'name': name,
                           **({'arguments': json.dumps(value)} if name == 'workspace' else {'input': value})}]
        else:
            observed = json.dumps(self.server.requests)
            expected_output = getattr(self.server,'expected_output','[number] 2')
            if getattr(self.server, 'ostadix_revision', None):
                received_output = any(actual_mcp_two(value, self.server.ostadix_revision)
                                      for value in nested_objects(self.server.requests))
            else:
                received_output = expected_output in observed
            if not received_output or '1 + 1' not in observed:
                self.server.failure = 'Actual tool results were not returned to the model'
            summary = ('Offline fixture failed. Inspect the recorded tool error.' if self.server.failure else
                       'Offline integration fixture completed. Actual command output: ' + expected_output + '.')
            output = [{'type':'message', 'id':'msg_final', 'status':'completed', 'role':'assistant',
                       'content':[{'type':'output_text','text':summary,'annotations':[]}]}]
        response = {'id':'resp_'+str(index), 'object':'response', 'status':'completed', 'output':output,
                    'usage':{'input_tokens':20,'output_tokens':20,'total_tokens':40}}
        events = [{'type':'response.created','response':dict(response,status='in_progress',output=[])},
                  {'type':'response.output_item.done','output_index':0,'item':output[0]},
                  {'type':'response.completed','response':response}]
        payload=''.join('event: '+e['type']+'\ndata: '+json.dumps(e)+'\n\n' for e in events).encode()
        self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Content-Length',str(len(payload)));self.end_headers();self.wfile.write(payload)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--backends',type=Path)
    parser.add_argument('--terminal',action='store_true',help='Drive requests, approvals and speech events through the terminal interface')
    parser.add_argument('--record',type=Path,help='Save local fixture model requests for protocol diagnostics')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--python-smoke',action='store_true',help='Use a harmless Python command to test the agent independently of OSTADIX runtime health')
    mode.add_argument('--ostadix-mcp',action='store_true',help='Use the verified managed MCP through the agent and terminal, with mocked audio')
    args=parser.parse_args()
    executable=shutil.which('O')
    if not args.python_smoke and not args.ostadix_mcp and (not executable or not args.backends):
        parser.error('OSTADIX executable O and --backends are required unless --python-smoke or --ostadix-mcp is selected')
    pair=None
    if args.ostadix_mcp:
        from ostadix_pair import load_pair, PairError
        try:
            pair=load_pair()
        except PairError as error:
            parser.error(str(error))
    backends=args.backends.resolve(strict=True) if args.backends else None
    with tempfile.TemporaryDirectory(prefix='voice-native-') as directory:
        root=Path(directory)/'workspace';root.mkdir()
        (root/'hello.O').write_text(SMOKE_SOURCE)
        (root/'keep.txt').write_text('Deletion must be denied.')
        if args.ostadix_mcp:
            parameters={'source':SMOKE_SOURCE,'action':'execute'}
            command=['@ostadix','call','o_execute',json.dumps(parameters)]
            expected_approval={'operation':'ostadix','tool':'o_execute',
                               'arguments':dict(parameters,cwd=str(root)),
                               'repository':pair['root'],'revision':pair['revision']}
        else:
            command=([sys.executable,'-c','print("voice smoke:", 1 + 1)'] if args.python_smoke else
                     [str(Path(executable).resolve()),'hello.O',str(backends)])
            expected_approval={'operation':'command','argv':command}
        model=ThreadingHTTPServer(('127.0.0.1',0),Model);model.requests=[];model.failure=None
        model.ostadix_revision=pair['revision'] if pair else None
        model.expected_output=('OSTADIX MCP result: 2' if args.ostadix_mcp else
                               'voice smoke: 2' if args.python_smoke else '[number] 2')
        model.actions=[
            'text(await tools.workspace('+json.dumps({'operation':'read','path':'hello.O'})+'));',
            'text(await tools.workspace('+json.dumps({'operation':'command','argv':command})+'));',
            'text(await tools.workspace('+json.dumps({'operation':'delete','path':'keep.txt'})+'));',
            'text(await tools.apply_patch('+json.dumps('*** Begin Patch\n*** Add File: native-unapproved.txt\n+must not exist\n*** End Patch')+'));',
        ]
        threading.Thread(target=model.serve_forever,daemon=True).start()
        class OfflineRPC(RPC):
            def call(self, method, params, **kwargs):
                # A custom local provider reports no OpenAI account. Supply a test
                # account solely for this offline harness; production auth is unchanged.
                if method == 'account/read': return {'account': {'type': 'chatgpt'}}
                return super().call(method, params, **kwargs)
            def __init__(self, command, cwd, callback):
                for key,value in {
                    'model_provider':'voice_offline_test',
                    'model_providers.voice_offline_test.name':'Offline fixture',
                    'model_providers.voice_offline_test.base_url':f'http://127.0.0.1:{model.server_port}/v1',
                    'model_providers.voice_offline_test.wire_api':'responses',
                    'model_providers.voice_offline_test.requires_openai_auth':False,
                    'model_providers.voice_offline_test.request_max_retries':0,
                    'model_providers.voice_offline_test.stream_max_retries':0,
                    'features.enable_request_compression':False,
                    'features.responses_websockets':False,'features.responses_websockets_v2':False,
                }.items(): command+=['-c',key+'='+json.dumps(value)]
                super().__init__(command,cwd,callback)
        runtime=None;ui=None;spoken=[]
        if args.terminal or args.ostadix_mcp:
            from terminal import TerminalSession, TerminalUI
            import queue
            class ProbeAudio:
                mode=None
                stopping=False
                events=queue.Queue()
                def start(self,mode,text=None): spoken.append(text)
            runtime=TerminalSession(root,Path(directory)/'state',OfflineRPC)
            store=runtime.store;sid=runtime.sid;agent=runtime.agent
            ui=TerminalUI(runtime,ProbeAudio(),lambda _:None)
        else:
            store=Store(Path(directory)/'state');sid=store.create(root);agent=Agent(store,root,OfflineRPC)
        agent.set_action_policy('ask')  # This fixture explicitly exercises approval and denial.
        try:
            if ui:ui.handle('Run the fixed offline acceptance fixture.')
            else:agent.submit(sid,'Run the fixed offline acceptance fixture.','native-offline-test')
            end=time.monotonic()+50
            while agent.active and time.monotonic()<end:
                with agent.lock: approvals=list(agent.approvals.values())
                for approval in approvals:
                    if approval['event'].is_set():continue
                    allow=approval['arguments']==expected_approval
                    if ui:ui.handle(('/approve ' if allow else '/deny ')+approval['id'])
                    else:agent.approve(sid,approval['id'],allow)
                    print('APPROVAL:', ('fixed harmless MCP call accepted' if args.ostadix_mcp else 'fixed harmless command accepted') if allow else 'unapproved action denied',flush=True)
                if ui:ui.pump()
                time.sleep(.05)
            if ui:
                ui.pump()
            events=store.events(sid)
            if args.record:
                args.record.write_text(json.dumps(model.requests, indent=2))
            for event in events:
                if event['kind'] in ('tool_result','finished','notice'):
                    print(event['kind'].upper()+':',json.dumps(event['data']),flush=True)
            if ui:
                assert len(spoken)==1 and model.expected_output in spoken[0],spoken
                print('PASS: terminal request, exact approval/denial and final-only spoken summary use real Codex results (audio mocked).',flush=True)
            assert not agent.active,'Integration timed out'
            assert store.session(sid)['status']=='idle','Turn did not complete'
            assert len(model.requests)>=5,'Missing real tool round trips'
            assert not model.failure,model.failure
            assert (root/'keep.txt').exists(),'Denied delete was executed'
            assert not (root/'native-unapproved.txt').exists(),'Native edit bypassed approval'
            blocked_request = any(e['kind']=='notice' and 'native Codex' in e['data'].get('text','') for e in events)
            # 0.162.1 removes the native edit tool from this restricted catalog
            # entirely. Its attempted call must receive an explicit rejection.
            rejected_tool = any(item.get('call_id') == 'call_3' and
                                'unsupported custom tool call: apply_patch' in str(item.get('output', ''))
                                for request in model.requests for item in request.get('input', []))
            assert blocked_request or rejected_tool,'Native edit was neither denied nor unavailable'
            results=[e['data']['result'] for e in events if e['kind']=='tool_result']
            if args.ostadix_mcp:
                executed=[r for r in results if r.get('mcp_tool')=='o_execute']
                assert len(executed)==1 and actual_mcp_two(executed[0],pair['revision']),executed
                print('PASS: verified MCP '+pair['revision']+' returned the native number 2 through workspace; actual structured result reached the local model.',flush=True)
            else:
                assert any(r.get('exit_code')==0 and r.get('output','').strip()==model.expected_output for r in results)
            thread_id=store.session(sid)['thread_id']
            agent.close()
            # Verify thread resume in a new app-server process, not merely an in-memory join.
            from agent import codex_command
            resumed=OfflineRPC(codex_command(root),store.directory,lambda _:None)
            try:
                result=resumed.call('thread/resume',{'threadId':thread_id,'cwd':str(root),'sandbox':'read-only'})
                assert result['thread']['id']==thread_id
                print('PASS: new Codex process resumed the same persisted thread.',flush=True)
            finally:resumed.close()
            model.http_error=401
            failed_agent=Agent(store,root,OfflineRPC)
            try:
                failed_agent.submit(sid,'Offline authentication failure test.','offline-http-401')
                end=time.monotonic()+20
                while failed_agent.active and time.monotonic()<end: time.sleep(.05)
                assert store.session(sid)['status']=='failed','HTTP 401 was not reported as failure'
                failure=[e for e in store.events(sid) if e['kind']=='finished'][-1]['data']
                assert '401' in failure['text'] or 'authentication' in failure['text'].lower(),failure
                assert store.session(sid)['thread_id']==thread_id
                print('PASS: real HTTP 401 reached the UI state as failure; thread and prior results remained.',flush=True)
            finally:failed_agent.close()
            model.http_error=None;model.requests=[]
            sleep_command=[sys.executable,'-c','import time; print("started",flush=True); time.sleep(30)']
            model.actions=['text(await tools.workspace('+json.dumps({'operation':'command','argv':sleep_command})+'));']
            cancel_sid=store.create(root);cancel_agent=Agent(store,root,OfflineRPC)
            try:
                cancel_agent.submit(cancel_sid,'Offline cancellation fixture.','offline-cancel-test')
                end=time.monotonic()+20
                while cancel_agent.process is None and time.monotonic()<end:
                    with cancel_agent.lock:approvals=list(cancel_agent.approvals.values())
                    for approval in approvals:
                        if not approval['event'].is_set():
                            assert approval['arguments']=={'operation':'command','argv':sleep_command}
                            cancel_agent.approve(cancel_sid,approval['id'],True)
                    time.sleep(.02)
                process=cancel_agent.process
                assert process is not None,'Command did not start'
                cancel_agent.cancel(cancel_sid)
                end=time.monotonic()+10
                while cancel_agent.active and time.monotonic()<end:time.sleep(.05)
                assert process.poll() is not None,'Local child survived cancellation'
                assert store.session(cancel_sid)['status']=='interrupted','Codex did not confirm interruption'
                print('PASS: real Codex turn/interrupt and local process termination.',flush=True)
            finally:cancel_agent.close()
            print('PASS: real Codex protocol, scoped read, approved '+('OSTADIX MCP' if args.ostadix_mcp else 'Python' if args.python_smoke else 'OSTADIX')+' execution, destructive denial, native-tool denial. Model responses were local fixtures; zero paid inference calls.',flush=True)
        finally:
            if runtime:runtime.close()
            else:agent.close();store.db.close()
            model.shutdown();model.server_close()


if __name__ == '__main__':
    if '--live' not in sys.argv:
        raise SystemExit('Live device/model integration is opt-in. Read this script, then pass --live explicitly.')
    sys.argv.remove('--live')
    main()
