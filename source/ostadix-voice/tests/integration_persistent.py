#!/usr/bin/env python3
"""Real Codex delegation to resident assistants; uses existing ChatGPT allowance.

Run explicitly from normal Termux after enabling the local service. Uses an
isolated conversation store, performs no shell/file actions through Codex, and
never approves generated actions. The local assignment continues after Codex
finishes and its process is released. Results are retained for inspection.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import Agent, Store
from local_runtime import LocalClient
from local_workers import owner_key


def wait_for(predicate, seconds, message):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(.2)
    raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.getuid() == 0:
        raise SystemExit('Run outside root in the normal Termux security context.')
    client = LocalClient()
    before = client.status()
    if not before.get('ready') or not before.get('service_enabled') or before['active'] or before['queued']:
        raise SystemExit('Enable the selected local service and wait until it is ready and idle.')
    directory = ROOT / 'runtime' / ('persistent-integration-' + uuid.uuid4().hex[:8])
    workspace = directory / 'workspace'; workspace.mkdir(parents=True)
    store = Store(directory / 'state')
    sid = store.create(workspace)
    owner = owner_key(store.directory, sid, workspace)
    agent = Agent(store, workspace)
    agent.set_action_policy('ask')  # This model probe must never execute an unreviewed mutation.
    report = {'created': time.time(), 'uid': os.getuid(), 'state_directory': str(store.directory),
              'owner': owner, 'status_before': before}
    process = None
    try:
        prompt = ('Test the persistent local assistant integration. Use the @local assistant route to create '
                  'an assistant named persistence-check with role reviewer. Assign it this exact task: '
                  'Write a numbered checklist of 24 concrete checks for a small Python command-line todo app. '
                  'Each entry should mention a different behavior and its expected result. Do not execute anything. '
                  'Set max_steps=2 and max_tokens=512. After the assignment is accepted, immediately finish your '
                  'response with its assistant ID and say that local work is continuing. Do not wait, read, poll, '
                  'run commands, edit files, or invoke any other tools. This request authorizes only those two local actions.')
        agent.submit(sid, prompt, 'persistent-smoke-' + uuid.uuid4().hex)
        deadline = time.monotonic() + 180
        while agent.active and time.monotonic() < deadline:
            if agent.rpc:
                process = agent.rpc.process
            if agent.approvals:
                raise RuntimeError('Unexpected approval request; no generated action was approved')
            time.sleep(.2)
        if agent.active:
            raise RuntimeError('Codex did not finish within the integration deadline')
        report['cloud_session'] = store.session(sid)
        if report['cloud_session']['status'] != 'idle':
            raise RuntimeError('Codex delegation failed: ' + str(report['cloud_session']))
        wait_for(lambda: agent.rpc is None, 10, 'Idle Codex process was not released')
        report['codex_process_released'] = process is not None and process.poll() is not None
        assistants = client.assistant_list(owner)
        if len(assistants) != 1 or not assistants[0].get('assignment'):
            raise RuntimeError('Expected exactly one assigned persistent assistant')
        ident = assistants[0]['id']
        report['assistant_after_cloud_release'] = client.assistant_read(owner, ident)
        report['local_work_outlived_codex'] = report['assistant_after_cloud_release']['state'] in ('running', 'queued')
        agent.close()
        report['interface_closed_at'] = time.time()
        item = wait_for(lambda: (value if (value := client.assistant_read(owner, ident))['state']
                                not in ('queued', 'running', 'pause_pending') else None),
                        260, 'Local assignment did not finish after interface close')
        report['assistant_final'] = item
        report['tool_calls'] = agent.tool_calls
        report['events'] = store.events(sid)
        report['status_after'] = client.status()
        if item['state'] != 'completed' or item['assignment']['completed_steps'] != 2:
            raise RuntimeError('Local two-step assignment failed: ' + str(item['assignment'].get('error')))
        if not report['local_work_outlived_codex'] or not report['codex_process_released']:
            raise RuntimeError('Trial did not establish local work after Codex release')
        if agent.tool_calls != 2:
            raise RuntimeError('Codex made an unexpected number of tool calls')
        report['passed'] = True
    except Exception as error:
        report.update(passed=False, error=str(error))
        raise
    finally:
        agent.close()
        client.assistant_cancel_owner(owner)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        store.db.close()
        print(json.dumps({key: report.get(key) for key in
                          ('passed', 'error', 'codex_process_released', 'local_work_outlived_codex', 'tool_calls')}, indent=2))


if __name__ == '__main__':
    import sys
    if '--live' not in sys.argv:
        raise SystemExit('Live device/model integration is opt-in. Read this script, then pass --live explicitly.')
    sys.argv.remove('--live')
    main()
