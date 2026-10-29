"""Exercise the installed Codex protocol against a loopback fake model, never OpenAI."""
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import RPC, TOOL_SCHEMA, codex_command


class Model(BaseHTTPRequestHandler):
    requests = []
    def log_message(self, *_): pass
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        self.requests.append(body)
        output = [{'type': 'message', 'id': 'msg_test', 'status': 'completed', 'role': 'assistant',
                   'content': [{'type': 'output_text', 'text': 'Offline protocol probe completed.', 'annotations': []}]}]
        response = {'id': 'resp_test', 'object': 'response', 'status': 'completed', 'output': output,
                    'usage': {'input_tokens': 10, 'output_tokens': 10, 'total_tokens': 20}}
        events = [
            {'type': 'response.created', 'response': dict(response, status='in_progress', output=[])},
            {'type': 'response.output_item.added', 'output_index': 0, 'item': dict(output[0], status='in_progress')},
            {'type': 'response.output_text.delta', 'item_id': 'msg_test', 'output_index': 0, 'content_index': 0, 'delta': 'Offline protocol probe completed.'},
            {'type': 'response.output_item.done', 'output_index': 0, 'item': output[0]},
            {'type': 'response.completed', 'response': response}]
        payload = ''.join('event: '+e['type']+'\ndata: '+json.dumps(e)+'\n\n' for e in events).encode()
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream'); self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)


def run():
    with tempfile.TemporaryDirectory(prefix='voice-protocol-') as directory:
        root = Path(directory)
        model = ThreadingHTTPServer(('127.0.0.1', 0), Model)
        threading.Thread(target=model.serve_forever, daemon=True).start()
        command = codex_command(root)
        for key, value in {
            'model_provider': 'voice_offline_test',
            'model_providers.voice_offline_test.name': 'Offline local test',
            'model_providers.voice_offline_test.base_url': f'http://127.0.0.1:{model.server_port}/v1',
            'model_providers.voice_offline_test.wire_api': 'responses',
            'model_providers.voice_offline_test.requires_openai_auth': False,
            'features.enable_request_compression': False,
            'features.responses_websockets': False, 'features.responses_websockets_v2': False,
        }.items():
            command += ['-c', key+'='+json.dumps(value)]
        events = []; complete = threading.Event()
        def receive(event):
            events.append(event)
            if event.get('method') in ('turn/completed', 'local/disconnected'): complete.set()
        rpc = RPC(command, root, receive)
        try:
            response = rpc.call('thread/start', {'cwd': directory, 'dynamicTools': [TOOL_SCHEMA], 'sandbox': 'read-only', 'approvalPolicy': 'on-request'})
            ident = response['thread']['id']
            print('thread/start: accepted dynamic workspace tool; model =', response['model'])
            rpc.call('turn/start', {'threadId': ident, 'input': [{'type': 'text', 'text': 'Return the offline test message.'}]})
            if not complete.wait(20): raise RuntimeError('No completion from local fake model')
            print('turn/completed:', json.dumps([e for e in events if e.get('method') == 'turn/completed']))
            tools = next((r['tools'] for r in Model.requests if 'tools' in r), [])
            for request in Model.requests:
                for item in request.get('input', []):
                    if item.get('type') == 'additional_tools': tools += item['tools']
            encoded = json.dumps(tools)
            print('Native catalog includes workspace:', 'workspace' in encoded)
            assert tools and 'workspace' in encoded
            # GPT-6 supplies namespace tools in additional_tools. Built-in apply_patch
            # remains present, but the app declines all native approval requests.
            for forbidden in ('### `exec_command`', '### `view_image`', '### `read_file`', '### `write_stdin`', '### `mcp__'):
                assert forbidden not in encoded, 'Unexpected local tool: ' + forbidden
            print('Requests to local fake model:', len(Model.requests))
            rpc.call('thread/resume', {'threadId': ident})
            print('thread/resume: accepted')
            return tools
        finally:
            rpc.close(); model.shutdown(); model.server_close()


if __name__ == '__main__':
    import sys
    if '--live' not in sys.argv:
        raise SystemExit('Live device/model integration is opt-in. Read this script, then pass --live explicitly.')
    sys.argv.remove('--live')
    run()
