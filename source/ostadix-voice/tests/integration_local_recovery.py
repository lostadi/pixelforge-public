#!/usr/bin/env python3
"""Explicit native daemon fault/recovery trial. Requires an idle enabled service.

Kills only the current owned daemon during this harness's text-only assignment,
verifies supervised recovery without replay, and verifies persistent disable.
Finally re-enables the service. No Codex calls or phone reboot are performed.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_runtime import LocalClient
from local_service import LocalService


def wait_for(check, seconds, label):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(.2)
    raise RuntimeError('Timed out: ' + label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if os.getuid() == 0:
        raise SystemExit('Run from normal Termux, outside root.')
    client, service = LocalClient(), LocalService()
    before = client.status()
    if not before.get('ready') or not before.get('supervised') or before['active'] or before['queued']:
        raise SystemExit('Recovery test needs a supervised, ready, idle model.')
    with sqlite3.connect(f'file:{client.state}/assistants/assistants.sqlite3?mode=ro', uri=True) as db:
        if db.execute("SELECT count(*) FROM assignments WHERE state IN ('queued','running','paused','pause_pending')").fetchone()[0]:
            raise SystemExit('Existing assignments must finish before this recovery test.')
    owner = 'recovery-test:' + uuid.uuid4().hex
    report = {'created': time.time(), 'uid': os.getuid(), 'status_before': before, 'owner': owner}
    try:
        assistant = client.assistant_create(owner, 'recovery-check', 'reviewer')
        ident = assistant['id']
        client.assistant_guide(owner, ident, 'Give concise evidence-based answers. Do not invent actions.')
        assigned = client.assistant_assign(owner, ident, 'Write 40 distinct numbered checks for a todo application with expected outcomes.', 2, 512)
        running = wait_for(lambda: [job for job in client.jobs(owner) if job['state'] == 'running'], 30, 'assigned local step')
        report['assignment_before_kill'] = client.assistant_read(owner, ident)
        report['jobs_before_kill'] = running
        if client.status()['active'] != len(running) or client.status()['queued']:
            raise RuntimeError('Another local job appeared; declining to interrupt its daemon')
        os.kill(before['daemon_pid'], signal.SIGKILL)
        def recovered():
            value = client.status()
            return value if value.get('ready') and value.get('daemon_pid') != before['daemon_pid'] else None
        report['status_recovered'] = wait_for(recovered, 180, 'supervised model recovery')
        item = client.assistant_read(owner, ident)
        report['assistant_recovered'] = item
        report['jobs_recovered'] = client.jobs(owner)
        if item['id'] != ident or item['guidance'] != assigned['guidance'] or item['state'] != 'interrupted':
            raise RuntimeError('Assistant identity/guidance/interruption did not survive daemon recovery')
        if any(job['state'] in ('queued', 'running') for job in report['jobs_recovered']):
            raise RuntimeError('Interrupted local work was replayed')
        if report['status_recovered']['supervisor_pid'] != before['supervisor_pid']:
            raise RuntimeError('Supervisor unexpectedly changed during daemon recovery')
        service.disable()
        wait_for(lambda: not client.status()['running'] and not client.status()['supervised'], 20, 'explicit stop')
        report['disabled_resume'] = service.resume()
        time.sleep(3)
        report['disabled_after_wait'] = client.status()
        if report['disabled_after_wait']['running'] or report['disabled_after_wait']['service_enabled']:
            raise RuntimeError('Disabled service restarted without an explicit enable')
        report['passed'] = True
    except Exception as error:
        report.update(passed=False, error=str(error))
        raise
    finally:
        service.enable()
        report['status_final'] = wait_for(lambda: (value if (value := client.status()).get('ready') else None), 180, 'final model ready')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'passed': report.get('passed', False), 'error': report.get('error'),
                          'recovered': report.get('status_recovered', {}).get('ready'),
                          'enabled_final': report['status_final']['service_enabled']}, indent=2))


if __name__ == '__main__':
    import sys
    if '--live' not in sys.argv:
        raise SystemExit('Live device/model integration is opt-in. Read this script, then pass --live explicitly.')
    sys.argv.remove('--live')
    main()
