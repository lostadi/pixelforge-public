#!/usr/bin/env python3
"""Serially trial installed profiles, then restore the original local manifest.

This is an explicit device integration runner: it changes the selected model
temporarily and submits the seven comparison fixtures. Run as normal Termux
with the persistent supervisor disabled and the current daemon idle.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup_local_model as setup
from local_runtime import LocalClient


def alive(pid):
    if not pid:
        return False
    try:
        return Path('/proc', str(pid), 'stat').read_text().split(') ', 1)[1][0] != 'Z'
    except (OSError, IndexError):
        return False


def memory(pid):
    result = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith(('MemAvailable:', 'SwapFree:')):
            key, value = line.split(':', 1)
            result[key + '_KiB'] = int(value.split()[0])
    if pid:
        try:
            for line in Path('/proc', str(pid), 'status').read_text().splitlines():
                if line.startswith(('VmRSS:', 'VmHWM:')):
                    key, value = line.split(':', 1)
                    result[key + '_KiB'] = int(value.split()[0])
        except OSError:
            pass
    return result


def stop(client):
    status = client.status()
    pids = [status.get('daemon_pid'), status.get('engine_pid')]
    client._stop_once()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not client.status().get('running') and not any(alive(pid) for pid in pids):
            return
        time.sleep(.1)
    raise RuntimeError('Prior daemon or engine did not exit; refusing another model loader')


class Monitor:
    def __init__(self):
        self.peak_rss_kib = 0
        self.minimum_available_kib = None
        self.minimum_swap_free_kib = None
        self.low_since = None
        self.last_print = time.monotonic()

    def sample(self, pid, label):
        data = memory(pid)
        self.peak_rss_kib = max(self.peak_rss_kib, data.get('VmRSS_KiB', 0), data.get('VmHWM_KiB', 0))
        available = data['MemAvailable_KiB']
        self.minimum_available_kib = min(self.minimum_available_kib or available, available)
        self.minimum_swap_free_kib = min(self.minimum_swap_free_kib if self.minimum_swap_free_kib is not None else data['SwapFree_KiB'], data['SwapFree_KiB'])
        if available < 512 * 1024:
            self.low_since = self.low_since or time.monotonic()
            if time.monotonic() - self.low_since >= 3:
                raise RuntimeError('MemAvailable below 512 MiB for at least three seconds; stopping this trial')
        else:
            self.low_since = None
        if time.monotonic() - self.last_print >= 30:
            print(json.dumps({'progress': label, 'memory': data}), flush=True)
            self.last_print = time.monotonic()

    def summary(self):
        return {'peak_engine_rss_KiB': self.peak_rss_kib,
                'minimum_MemAvailable_KiB': self.minimum_available_kib,
                'minimum_SwapFree_KiB': self.minimum_swap_free_kib,
                'sampling_interval_seconds': .25,
                'caveat': 'Single phone session, ambient load and thermal state uncontrolled; file cache and Android activity affect available memory.'}


def ready(client, monitor, label):
    started = time.monotonic()
    client._start_once()
    while time.monotonic() - started < 180:
        status = client.status()
        monitor.sample(status.get('engine_pid'), label + ':startup')
        if status.get('error'):
            raise RuntimeError(status['error'])
        if status.get('ready'):
            return status, round(time.monotonic() - started, 3)
        time.sleep(.25)
    raise RuntimeError('Model did not become ready within 180 seconds')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='+', choices=list(setup.PROFILES))
    args = parser.parse_args()
    if os.getuid() == 0:
        raise SystemExit('Run as normal Termux, outside root')
    client = LocalClient()
    initial = client.status()
    if initial.get('service_enabled') or initial.get('supervised'):
        raise SystemExit('Disable the supervisor before explicit model trials')
    if initial.get('active') or initial.get('queued'):
        raise SystemExit('Current local model must be idle before explicit trials')
    manifest_path = setup.RUNTIME / 'local-model.json'
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    output_dir = setup.RUNTIME / 'model-trials'
    output_dir.mkdir(exist_ok=True)
    saved = output_dir / ('manifest-before-profiles-' + str(int(time.time())) + '.json')
    saved.write_bytes(original)
    os.chmod(saved, 0o600)
    summary = {'created': time.time(), 'saved_manifest': str(saved), 'models': [], 'restored': False}
    summary_path = output_dir / 'profile-trials-summary.json'
    child = None
    try:
        stop(client)
        for name in args.models:
            profile = setup.PROFILES[name]
            quant = profile.get('default_quantization', 'Q8_0')
            variant = profile['variants'][quant]
            label = name + '-' + quant
            model_path = setup.RUNTIME / 'models' / variant['filename']
            if model_path.stat().st_size != variant['bytes']:
                raise RuntimeError('Installed model size differs from pin: ' + name)
            monitor = Monitor()
            item = {'model': name, 'label': label, 'quantization': quant,
                    'model_sha256': variant['sha256'], 'outcome': 'starting'}
            print(json.dumps({'starting': label}), flush=True)
            try:
                setup.activate(quant, model_path, Path(manifest['server_path']), name)
                status, item['startup_seconds'] = ready(client, monitor, label)
                item['status_ready'] = status
                report_path = output_dir / (label + '.json')
                item['comparison_path'] = str(report_path)
                with (output_dir / (label + '.log')).open('w') as log:
                    child = subprocess.Popen([sys.executable, str(ROOT / 'tests/integration_model_compare.py'),
                                              '--label', label, '--output', str(report_path)],
                                             cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    deadline = time.monotonic() + 7 * 140
                    while child.poll() is None:
                        monitor.sample(status.get('engine_pid'), label + ':comparison')
                        if time.monotonic() > deadline:
                            raise RuntimeError('Comparison process exceeded its bounded deadline')
                        time.sleep(.25)
                item['comparison_exit_code'] = child.returncode
                child = None
                if report_path.exists():
                    report = json.loads(report_path.read_text())
                    item['comparison_seconds'] = report.get('wall_seconds')
                    item['automatic_passes'] = sum(bool(row['automatic_pass']) for row in report.get('results', []))
                    item['task_count'] = len(report.get('results', []))
                    item['failed_tasks'] = [row['task']['id'] for row in report.get('results', []) if not row['automatic_pass']]
                item['outcome'] = 'completed' if item['comparison_exit_code'] == 0 else 'comparison_failed'
            except Exception as exc:
                item['outcome'] = 'failed'
                item['error'] = str(exc)
                print(json.dumps({'trial_error': label, 'error': str(exc)}), flush=True)
            finally:
                if child is not None and child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill(); child.wait()
                child = None
                stop(client)
                item['memory'] = monitor.summary()
                summary['models'].append(item)
                summary_path.write_text(json.dumps(summary, indent=2) + '\n')
                print(json.dumps({'finished': item}), flush=True)
    finally:
        stop(client)
        temporary = manifest_path.with_name(manifest_path.name + '.restore')
        temporary.write_bytes(original)
        os.chmod(temporary, 0o600)
        temporary.replace(manifest_path)
        restore_monitor = Monitor()
        if initial.get('running'):
            restored_status, restore_seconds = ready(client, restore_monitor, 'restore-original')
            summary['restore_startup_seconds'] = restore_seconds
            summary['restore_status'] = restored_status
        summary['restored'] = manifest_path.read_bytes() == original
        summary['finished'] = time.time()
        summary_path.write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'restored': summary['restored'], 'summary_path': str(summary_path)}), flush=True)


if __name__ == '__main__':
    import sys
    if '--live' not in sys.argv:
        raise SystemExit('Live device/model integration is opt-in. Read this script, then pass --live explicitly.')
    sys.argv.remove('--live')
    main()
