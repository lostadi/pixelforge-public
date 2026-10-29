"""Real managed OSTADIX background jobs through Agent; no cloud/model calls.

The local RPC fixture proposes only a fixed Python sleep program and exact job
status requests. Each non-catalog action still goes through real once-only
Agent approval. The real native MCP and runtime execute and cancel the job.
"""

from collections import deque
import json
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Agent, Store
from ostadix_pair import load_pair


SOURCE = '''python^(
import time
print("background fixture started", flush=True)
time.sleep(30)
__oval_result__ = 1 + 1
)_python
'''


def main():
    pair = load_pair()
    actions = deque()
    replies = []
    interrupted = []

    class FixtureRPC:
        """Deterministic local model transport; never starts a Codex process."""
        def __init__(self, _command, _cwd, callback):
            self.callback = callback
            self.closed = False
            self.turn = 0
            self.pending = {}

        def call(self, method, parameters, **_kwargs):
            if method == 'account/read':
                return {'account': {'type': 'chatgpt'}}
            if method in ('thread/start', 'thread/resume'):
                return {'thread': {'id': 'offline-native-job-thread'}}
            if method == 'turn/start':
                self.turn += 1
                turn = f'offline-job-turn-{self.turn}'
                request = f'offline-job-call-{self.turn}'
                self.pending[request] = turn
                self.callback({'method': 'turn/started', 'params': {
                    'threadId': 'offline-native-job-thread', 'turn': {'id': turn},
                }})
                self.callback({'id': request, 'method': 'item/tool/call', 'params': {
                    'threadId': 'offline-native-job-thread', 'turnId': turn,
                    'tool': 'workspace', 'arguments': actions.popleft(),
                }})
                return {'turn': {'id': turn}}
            if method == 'turn/interrupt':
                interrupted.append(parameters)
                raise AssertionError('An idle model turn must not be interrupted')
            raise AssertionError(f'Unexpected fixture RPC: {method}')

        def send(self, message):
            result = message['result']
            value = json.loads(result['contentItems'][0]['text'])
            replies.append(value)
            job = value.get('result', {}).get('structuredContent', {})
            if job.get('state') == 'cancelled':
                summary = 'OSTADIX confirmed that the background job was cancelled.'
            elif job.get('state') == 'running':
                summary = 'The background job is running. Execution has not completed.'
            else:
                summary = 'The MCP request returned. Inspect its actual result.'
            thread = 'offline-native-job-thread'
            self.callback({'method': 'item/completed', 'params': {
                'threadId': thread, 'item': {'type': 'agentMessage', 'text': summary},
            }})
            self.callback({'method': 'turn/completed', 'params': {
                'threadId': thread,
                'turn': {'id': self.pending.pop(message['id']), 'status': 'completed'},
            }})

        def close(self):
            self.closed = True

    with tempfile.TemporaryDirectory(prefix='voice-mcp-jobs-') as directory:
        root = Path(directory) / 'workspace'
        root.mkdir()
        store = Store(Path(directory) / 'state')
        sid = store.create(root)
        agent = Agent(store, root, FixtureRPC)
        agent.set_action_policy('ask')  # Only the fixture's exact expected calls may execute.

        def proposal(name, parameters):
            return {'operation': 'ostadix', 'tool': name, 'arguments': parameters,
                    'repository': pair['root'], 'revision': pair['revision']}

        def run(name, parameters, effective, request_id):
            start = len(replies)
            expected = proposal(name, effective)
            actions.append({'operation': 'command', 'argv': [
                '@ostadix', 'call', name, json.dumps(parameters),
            ]})
            agent.submit(sid, 'Run the fixed offline native job fixture.', request_id)
            approvals_seen = 0
            deadline = time.monotonic() + 25
            while (agent.active or agent.tool_lock.locked()) and time.monotonic() < deadline:
                with agent.lock:
                    approvals = list(agent.approvals.values())
                for approval in approvals:
                    if approval['event'].is_set():
                        continue
                    allow = approval['arguments'] == expected
                    agent.approve(sid, approval['id'], allow)
                    assert allow, ('Unexpected action denied', approval['arguments'])
                    approvals_seen += 1
                time.sleep(.02)
            assert not agent.active and not agent.tool_lock.locked(), 'Native job fixture timed out'
            assert approvals_seen == 1, 'Exactly one matching local approval is required'
            assert len(replies) == start + 1, 'Native result did not reach the local model fixture'
            assert store.session(sid)['status'] == 'idle'
            return replies[-1]

        try:
            # Real tools/list discovery, then verify the schema before execution.
            entry = agent.ostadix.connection(sid, root)
            assert entry['pair']['revision'] == pair['revision']
            execute_schema = entry['tools']['o_execute']['inputSchema']['properties']
            assert {'source', 'action', 'background', 'cwd', 'timeout_secs'} <= execute_schema.keys()
            assert 'job_id' in entry['tools']['o_job_status']['inputSchema']['properties']
            client = entry['client']
            parameters = {'source': SOURCE, 'action': 'execute', 'background': True,
                          'timeout_secs': 40}
            started = run('o_execute', parameters, dict(parameters, cwd=str(root)),
                          'native-background-submit')
            job = started.get('result', {}).get('structuredContent', {})
            assert started.get('exit_code') == 0 and job.get('state') == 'running', started
            assert job.get('exit_code') is None, 'Submission must not claim execution success'
            job_id = job['job_id']
            child_pid = job['pid']
            assert agent.ostadix.has_jobs(sid), 'The real background job was not retained'
            print('SUBMITTED:', json.dumps(job), flush=True)

            status_args = {'job_id': job_id}
            status = run('o_job_status', status_args, status_args, 'native-background-status')
            assert status['result']['structuredContent']['state'] == 'running', status
            assert not agent.active and store.session(sid)['status'] == 'idle'

            # Both model turns have ended; /cancel must still stop this job.
            agent.cancel(sid)
            assert not interrupted, 'Idle job cancellation called model turn/interrupt'
            assert not agent.ostadix.has_jobs(sid), 'Native job cleanup was not confirmed'
            cancelled = run('o_job_status', status_args, status_args,
                            'native-background-cancel-status')
            final = cancelled['result']['structuredContent']
            assert final['state'] == 'cancelled', final
            assert final['cleanup']['child_reaped'] is True, final
            assert cancelled['exit_code'] != 0, 'Cancelled work was reported as execution success'
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError('The cancelled native child is still present')
            print('CANCELLED:', json.dumps(final), flush=True)

            events = store.events(sid)
            results = [event['data'] for event in events if event['kind'] == 'tool_result']
            assert len(results) == 3
            assert results[0]['success'] is True
            assert results[-1]['success'] is False
            assert results[-1]['result'] == cancelled
            assert any(event['kind'] == 'notice' and 'Cancellation requested' in
                       event['data'].get('text', '') for event in events)
            assert any(event['kind'] == 'assistant' and 'Execution has not completed' in
                       event['data'].get('text', '') for event in events)
            assert agent.ostadix.connections[sid]['client'] is client and client.alive
            agent.close()
            assert client.shutdown_graceful, 'Native MCP did not finish graceful EOF shutdown'
            print('PASS: real managed MCP background execution, exact approvals, idle-turn cancellation, '
                  'native child reaping, truthful saved results, and graceful MCP shutdown. '
                  'Model transport was a local fixture; zero inference calls.', flush=True)
        finally:
            agent.close()
            store.db.close()


if __name__ == '__main__':
    import sys
    if '--live' not in sys.argv:
        raise SystemExit('Live device/model integration is opt-in. Read this script, then pass --live explicitly.')
    sys.argv.remove('--live')
    main()
