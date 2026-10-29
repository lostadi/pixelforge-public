#!/data/data/com.termux/files/usr/bin/python
"""Read-only Pixel TPU telemetry; emit timestamped JSON lines.

These device-wide counters are supporting evidence, not proof that any selected
model used the NPU. Correlate a successful accelerator-only inference, its PID,
device/library evidence, and a before/during/after sample. Other Android services
can contribute to counters. Missing access is recorded, never repaired here.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import sys
import time

SYSFS = Path('/sys/devices/platform/sswrp_tpu@36000000/36500000.buenos')
FIELDS = (
    'clients', 'groups', 'device_utilization', 'tpu_utilization',
    'inference_count', 'tpu_active_cycle_count', 'tpu_op_count',
    'tpu_usage', 'firmware_version', 'firmware_crash_count',
    'watchdog_timeout_count', 'power/runtime_status',
    'power/runtime_active_time', 'power/runtime_suspended_time',
)
MAX_BYTES = 65536
MAX_DEVFREQ_DOMAINS = 8
DEVFREQ_FIELDS = ('available_frequencies', 'cur_freq', 'min_freq', 'max_freq', 'governor')
TPU_NAMES = re.compile(r'edgetpu|darwinn|buenos|tachyon|liblitert|libgoogle3|'
                       r'llm_wrapper|runtime_model_loader', re.I)


def read_bounded(path, limit=MAX_BYTES):
    with open(path, 'rb') as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError('read exceeded byte bound')
    return data.decode('utf-8', errors='replace')


def process_identity(pid):
    """No argument strings, environment, app files, or private data are read."""
    root = Path('/proc') / str(pid)
    stat = read_bounded(root / 'stat').rsplit(')', 1)[1].split()
    status = read_bounded(root / 'status')
    status_fields = dict(line.split(':', 1) for line in status.splitlines() if ':' in line)
    return {
        'pid': pid,
        'tgid': int(status_fields['Tgid']),
        'ppid': int(status_fields['PPid']),
        'name': read_bounded(root / 'comm').strip(),
        'uid': [int(value) for value in status_fields['Uid'].split()],
        'starttime_ticks': int(stat[19]),
        'selinux_context': read_bounded(root / 'attr/current').rstrip('\x00\r\n '),
    }


def process_evidence(pid):
    result = {'requested_pid': pid}
    root = Path('/proc') / str(pid)
    try:
        before = process_identity(pid)
        result['identity'] = before
    except (OSError, ValueError, IndexError, StopIteration) as error:
        result['error'] = str(error)
        return result
    try:
        # Maps are bounded and only matching pathname suffixes are retained.
        matches = set()
        for line in read_bounded(root / 'maps', 2 * 1024 * 1024).splitlines():
            parts = line.split(None, 5)
            if len(parts) == 6 and TPU_NAMES.search(parts[5]):
                matches.add(parts[5])
        result['tpu_related_mapped_paths'] = sorted(matches)
    except (OSError, ValueError) as error:
        result['maps_error'] = str(error)
    try:
        matches = []
        with os.scandir(root / 'fd') as entries:
            for count, entry in enumerate(entries):
                if count >= 4096:
                    raise ValueError('descriptor listing exceeded bound')
                try:
                    target = os.readlink(entry.path)
                except FileNotFoundError:
                    continue
                if target.startswith('/dev/') and TPU_NAMES.search(target):
                    matches.append({'fd': entry.name, 'device': target})
        result['tpu_device_descriptors'] = matches
    except (OSError, ValueError) as error:
        result['fd_error'] = str(error)
    try:
        result['identity_stable'] = process_identity(pid) == before
    except (OSError, ValueError, IndexError, StopIteration):
        result['identity_stable'] = False
    return result


def device_metadata(result):
    """Read kernel-reported identities and raw frequency fields, without inference."""
    driver = SYSFS / 'driver'
    result['sources']['driver'] = str(driver)
    try:
        result['driver']['link'] = os.readlink(driver)
        target = driver.resolve(strict=True)
        result['driver'].update(path=str(target), name=target.name)
    except (OSError, RuntimeError, ValueError) as error:
        result['errors']['driver'] = str(error)

    compatible = SYSFS / 'of_node/compatible'
    result['sources']['compatible'] = str(compatible)
    try:
        raw = read_bounded(compatible)
        result['raw']['of_node/compatible'] = raw
        result['compatible'] = [value for value in raw.split('\0') if value]
    except (OSError, ValueError) as error:
        result['errors']['of_node/compatible'] = str(error)

    frequency_root = SYSFS / 'devfreq'
    result['sources']['devfreq'] = str(frequency_root)
    try:
        with os.scandir(frequency_root) as entries:
            for index, entry in enumerate(entries):
                if index >= MAX_DEVFREQ_DOMAINS:
                    result['errors']['devfreq'] = f'Directory listing exceeded {MAX_DEVFREQ_DOMAINS} entries; remaining domains were not inspected'
                    break
                path = frequency_root / entry.name
                try:
                    if not entry.is_dir(follow_symlinks=True):
                        result['errors']['devfreq/' + entry.name] = 'Not a directory; not inspected'
                        continue
                except OSError as error:
                    result['errors']['devfreq/' + entry.name] = str(error)
                    continue
                domain = {'name': entry.name, 'path': str(path), 'raw': {}}
                result['devfreq'].append(domain)
                for field in DEVFREQ_FIELDS:
                    key = 'devfreq/' + entry.name + '/' + field
                    result['sources'][key] = str(path / field)
                    try:
                        domain['raw'][field] = read_bounded(path / field)
                    except (OSError, ValueError) as error:
                        result['errors'][key] = str(error)
    except (OSError, ValueError) as error:
        result['errors']['devfreq'] = str(error)


def snapshot(pids=()):
    started = time.monotonic()
    result = {
        'schema': 'ostadix.pixel-tpu-telemetry/v1',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'monotonic_ns': time.monotonic_ns(),
        'collector_uid': os.getuid(),
        'device_sysfs': str(SYSFS),
        'raw': {}, 'errors': {}, 'client_process_candidates': [],
        'driver': {}, 'compatible': [], 'devfreq': [], 'sources': {},
        'interpretation': 'Device-wide samples; alone they do not prove model offload. '
            'Raw client column meanings depend on the installed driver; numeric '
            'first-column PID candidates below are correlated but not asserted.',
    }
    for field in FIELDS:
        try:
            result['raw'][field] = read_bounded(SYSFS / field)
        except (OSError, ValueError) as error:
            result['errors'][field] = str(error)
    device_metadata(result)
    for line in result['raw'].get('clients', '').splitlines()[:64]:
        fields = line.split()
        # Labeled public driver format and this device's compact numeric format.
        candidate = fields[1] if fields and fields[0] == 'pid' and len(fields) > 1 else fields[0] if fields else ''
        if not candidate.isdecimal():
            continue
        try:
            result['client_process_candidates'].append({
                'raw_row': line, 'identity': process_identity(int(candidate)),
            })
        except (OSError, ValueError, IndexError, StopIteration) as error:
            result['client_process_candidates'].append({'raw_row': line, 'error': str(error)})
    result['inspected_processes'] = [process_evidence(pid) for pid in pids]
    result['collection_seconds'] = round(time.monotonic() - started, 6)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pid', type=int, action='append', default=[],
                        help='Inspect this process identity and only TPU-related maps/device FDs')
    parser.add_argument('--duration', type=float, default=0,
                        help='Seconds to sample, from 0 (one snapshot) to 3600')
    parser.add_argument('--interval', type=float, default=1,
                        help='Seconds between samples, minimum 0.2')
    parser.add_argument('--output', type=Path,
                        help='New NDJSON evidence file; existing files are never overwritten')
    args = parser.parse_args()
    if not 0 <= args.duration <= 3600 or not 0.2 <= args.interval <= 60:
        parser.error('duration must be 0..3600 and interval 0.2..60')
    if len(args.pid) > 16 or any(pid <= 0 for pid in args.pid):
        parser.error('at most 16 positive PIDs are allowed')
    output = args.output.open('x', encoding='utf-8') if args.output else sys.stdout
    started = time.monotonic()
    try:
        while True:
            print(json.dumps(snapshot(args.pid), separators=(',', ':')), file=output, flush=True)
            remaining = started + args.duration - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(args.interval, remaining))
    except KeyboardInterrupt:
        pass
    finally:
        if output is not sys.stdout:
            output.close()


if __name__ == '__main__':
    main()
