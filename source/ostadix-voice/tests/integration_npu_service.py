"""Activate the verified NPU profile and exercise the resident local service.

Explicit hardware integration only. Requires ordinary Termux permissions, idle
workers and --activate. Restores the previous profile on failure. On success the
NPU service remains enabled. No model output is executed as code.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from local_runtime import ROOT, MANIFEST, DEFAULT_STATE, LocalClient
from local_service import LocalService


def write_atomic(path, content):
    temporary = path.with_suffix('.new')
    temporary.write_bytes(content)
    temporary.chmod(0o600)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--activate', action='store_true', required=True)
    parser.add_argument('--recovery', action='store_true')
    args = parser.parse_args()
    if os.getuid() == 0:
        parser.error('Run as the ordinary Termux user.')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    destination = ROOT / 'runtime/npu/evidence' / ('persistent-' + stamp)
    destination.mkdir(mode=0o700)
    print('EVIDENCE ' + str(destination), flush=True)
    client, service = LocalClient(), LocalService()
    previous = service.status()
    assert previous.get('active', 0) == 0 and previous.get('queued', 0) == 0, 'Local jobs are active'
    original = MANIFEST.read_bytes()
    (destination / 'previous-profile.json').write_bytes(original)
    backup = ROOT / 'runtime/npu/cpu-model-before-npu.json'
    if not backup.exists() and json.loads(original).get('backend') != 'litert-npu':
        backup.write_bytes(original)
    report = {'started_utc': stamp, 'requests': [], 'idle_seconds': 30,
              'backend': 'litert-npu', 'recovery_test_requested': args.recovery}

    def save():
        write_atomic(destination / 'report.json', (json.dumps(report, indent=2) + '\n').encode())

    def preserve_log(name):
        source = DEFAULT_STATE / 'engine.log'
        if source.is_file():
            shutil.copyfile(source, destination / name)

    def ready(exclude=None):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status = client.status()
            if (status.get('ready') and status.get('model', {}).get('backend') == 'litert-npu'
                    and status.get('engine_pid') != exclude):
                return status
            time.sleep(.2)
        raise AssertionError('NPU service did not become ready: ' + json.dumps(client.status()))

    def request(task, tokens=64):
        status = ready()
        job = client.submit('npu-verification-' + stamp, task, max_tokens=tokens)
        deadline = time.monotonic() + 125
        while time.monotonic() < deadline:
            result = client.read('npu-verification-' + stamp, job['id'], wait=1, limit=65536)
            if result['state'] not in ('queued', 'running'):
                break
        report['requests'].append({'prompt': task, 'engine_pid': status['engine_pid'], 'job': result})
        save()
        assert result['state'] == 'completed', result
        assert result['result'].strip(), result
        assert result['usage'].get('backend') == 'litert-npu', result
        assert client.status()['engine_pid'] == status['engine_pid'], 'Engine changed during request'
        print('ANSWER ' + json.dumps({'pid': status['engine_pid'], 'text': result['result'],
                                     'usage': result['usage']}), flush=True)
        return result

    try:
        service.disable()
        write_atomic(MANIFEST, (ROOT / 'runtime/npu/model-profile.json').read_bytes())
        service.enable()
        status = ready()
        report['initial_status'] = status
        save()
        print('READY ' + str(status['engine_pid']), flush=True)
        answer = request('What is 2 plus 3? Reply with only the number.')
        assert answer['result'].strip().strip('. ') == '5', answer['result']
        request('Explain what a telephone does in two short sentences.', 128)
        before = client.status()
        print('Checking 30 seconds of idle residency.', flush=True)
        time.sleep(report['idle_seconds'])
        after = client.status()
        assert after['ready'] and after['engine_pid'] == before['engine_pid'], after
        report['same_pid_after_idle'] = after['engine_pid']
        request('Reply with exactly the word READY.')
        if args.recovery:
            status = client.status()
            assert not status['active'] and not status['queued'], 'New work arrived; not interrupting it'
            old_pid = status['engine_pid']
            executable = (Path('/proc') / str(old_pid) / 'exe').resolve()
            assert executable == (ROOT / 'runtime/npu/litert_npu_resident').resolve()
            preserve_log('engine-before-recovery.log')
            os.kill(old_pid, signal.SIGTERM)
            recovered = ready(exclude=old_pid)
            report['recovery'] = {'old_engine_pid': old_pid, 'new_engine_pid': recovered['engine_pid']}
            request('What is 3 plus 4? Reply with only the number.')
        report['final_status'] = client.status()
        report['success'] = True
        preserve_log('engine-final.log')
        save()
        print('PASS: NPU helper remains resident and supervised. Report: ' + str(destination / 'report.json'), flush=True)
    except BaseException as error:
        preserve_log('engine-failed.log')
        report['error'] = str(error)
        service.disable()
        write_atomic(MANIFEST, original)
        if previous.get('service_enabled'):
            service.enable()
        report['success'] = False
        report['previous_profile_restored'] = True
        save()
        raise


if __name__ == '__main__':
    import sys
    if '--live' not in sys.argv:
        raise SystemExit('Live device/model integration is opt-in. Read this script, then pass --live explicitly.')
    sys.argv.remove('--live')
    main()
