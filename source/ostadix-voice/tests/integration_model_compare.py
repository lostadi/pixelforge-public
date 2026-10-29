#!/usr/bin/env python3
"""Run a small explicit local-model comparison against the resident daemon.

This submits real, bounded local jobs. It never starts Codex or changes models.
Run once per selected model from the normal Termux shell.
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
from local_runtime import LocalClient


def memory(pid):
    try:
        values = {}
        for line in Path('/proc', str(pid), 'status').read_text().splitlines():
            if line.startswith(('VmRSS:', 'VmHWM:')):
                key, value = line.split(':', 1)
                values[key + '_KiB'] = int(value.split()[0])
        return values
    except (OSError, ValueError):
        return {}


def check_output(text, checks):
    outcomes = []
    for check in checks:
        kind = check['type']
        if kind == 'exact':
            passed = text.strip() == check['value']
        elif kind == 'json_subset':
            try:
                value = json.loads(text)
                passed = isinstance(value, dict) and all(value.get(key) == expected for key, expected in check['value'].items())
            except ValueError:
                passed = False
        elif kind == 'contains_all':
            passed = all(value.casefold() in text.casefold() for value in check['values'])
        else:
            raise ValueError('Unsupported comparison check: ' + kind)
        outcomes.append({'check': check, 'passed': passed})
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tasks', type=Path, default=ROOT / 'runtime/model-evaluation-tasks.json')
    args = parser.parse_args()
    if os.getuid() == 0:
        raise SystemExit('Run model comparisons from normal Termux, outside root.')
    fixture = json.loads(args.tasks.read_text())
    client = LocalClient()
    status = client.status()
    if not status.get('ready') or status['active'] or status['queued']:
        raise SystemExit('The selected local model must be ready and idle before a comparison.')
    owner = 'model-eval:' + uuid.uuid4().hex
    report = {'label': args.label, 'created': time.time(), 'uid': os.getuid(),
              'scope_note': fixture['scope_note'], 'status_before': status,
              'memory_before': memory(status['engine_pid']), 'results': []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        for task in fixture['tasks']:
            began = time.monotonic()
            job = client.submit(owner, task['task'], max_tokens=task['max_tokens'], role='assistant')
            while True:
                result = client.read(owner, job['id'], wait=1, limit=65536)
                if result['state'] not in ('running', 'queued'):
                    break
                if time.monotonic() - began > 130:
                    client.cancel(owner, job['id'])
                    raise RuntimeError('Comparison job did not stop within its deadline: ' + task['id'])
            checks = check_output(result.get('result', ''), task.get('checks', []))
            complete = result['state'] == 'completed' and not result.get('truncated')
            entry = {'task': task, 'wall_seconds': round(time.monotonic() - began, 3),
                     'result': result, 'checks': checks,
                     'automatic_pass': complete and all(item['passed'] for item in checks),
                     'memory_after': memory(status['engine_pid'])}
            report['results'].append(entry)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print(json.dumps({'task': task['id'], 'seconds': entry['wall_seconds'],
                              'state': result['state'], 'truncated': result.get('truncated'),
                              'automatic_pass': entry['automatic_pass'], 'text': result.get('result', '')}, ensure_ascii=False), flush=True)
    finally:
        client.cancel(owner)
        report['wall_seconds'] = round(time.monotonic() - started, 3)
        report['status_after'] = client.status()
        report['memory_after'] = memory(status['engine_pid'])
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('Saved comparison:', args.output, flush=True)


if __name__ == '__main__':
    import sys
    if '--live' not in sys.argv:
        raise SystemExit('Live device/model integration is opt-in. Read this script, then pass --live explicitly.')
    sys.argv.remove('--live')
    main()
