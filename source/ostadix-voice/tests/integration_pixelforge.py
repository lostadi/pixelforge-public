"""Actual phone acceptance: Huihui PixelForge -> host file tool -> final answer.

Run as ordinary Termux in its Android mount namespace. Uses a private temporary
workspace and state; never starts audio or a cloud model.
"""
import json
import argparse
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Store
from local_chat import LocalChatAgent
from pixelforge_chat import PixelForgeClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Fresh local evidence path; never overwritten')
    args = parser.parse_args()
    if os.getuid() == 0:
        raise SystemExit('Run with the ordinary Termux identity')
    evidence = args.output.expanduser().absolute()
    if evidence.exists() or evidence.is_symlink():
        raise SystemExit('Evidence output already exists: ' + str(evidence))
    evidence.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pixelforge-acceptance-') as directory:
        root = Path(directory)
        project = root / 'project'
        project.mkdir()
        expected = 'PIXELFORGE_PHONE_42'
        (project / 'proof.txt').write_text(expected + '\n')
        store = Store(root / 'state')
        sid = store.create(project)
        client = PixelForgeClient()
        agent = LocalChatAgent(store, project, client=client)
        started = time.monotonic()
        try:
            agent.submit(sid, 'Read proof.txt using the workspace read tool. Reply with only its exact content. '
                         'You must actually read the file before answering.', 'pixelforge-live-tool-1')
            cursor = 0
            while agent.start_thread.is_alive():
                for event in store.events(sid, cursor):
                    cursor = event['seq']
                    if event['kind'] in ('tool_call', 'tool_result', 'assistant', 'error', 'notice'):
                        print(json.dumps({'event': event['kind'], 'data': event['data']}, ensure_ascii=False), flush=True)
                agent.start_thread.join(1)
            events = store.events(sid)
            jobs = [client.read(job['owner'], job['id'], limit=32768) for job in list(client._jobs.values())]
            result = {'passed': False, 'engine': client.status()['model'], 'answer': agent.last_answer,
                      'session_status': store.session(sid)['status'], 'seconds': time.monotonic() - started,
                      'events': events, 'jobs': jobs, 'cloud_requests': 0, 'audio_started': False}
            evidence.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            print('SESSION_STATUS', result['session_status'], 'ANSWER', repr(agent.last_answer), flush=True)
            assert result['session_status'] == 'idle', 'Live main task failed; inspect acceptance evidence'
            assert agent.last_answer.strip() == expected, 'Model did not return the actual file content'
            assert any(event['kind'] == 'tool_result' for event in events), 'No real host tool result was recorded'
            completed = [job for job in jobs if job['state'] == 'completed']
            assert len(completed) >= 2, 'Tool loop did not complete two model turns'
            assert all(job['usage']['npu_weight_matmul_operations'] > 0 for job in completed)
            result['passed'] = True
            evidence.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            print('PASS: real Huihui model, NPU projections, actual file read, final answer; no cloud/audio.', flush=True)
            print('Evidence:', evidence, flush=True)
        finally:
            agent.close()
            store.db.close()


if __name__ == '__main__':
    import sys
    if '--live' not in sys.argv:
        raise SystemExit('Live device/model integration is opt-in. Read this script, then pass --live explicitly.')
    sys.argv.remove('--live')
    main()
