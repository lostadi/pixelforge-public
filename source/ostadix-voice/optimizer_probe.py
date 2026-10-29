"""Fixed phone telemetry and a dedicated, expiring optimizer wake lease.

The only write is the fixed optimizer wake-source name, with a 30-second expiry.
No app content, device setting, clock, governor or scheduling write is supported.
"""
import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import time

WAKE_NAME = 'ostadix_npu_optimizer'


def wake_lease(release=False):
    path = '/sys/power/wake_unlock' if release else '/sys/power/wake_lock'
    if not release or WAKE_NAME in read('/sys/power/wake_lock').split():
        with open(path, 'w') as stream:
            stream.write(WAKE_NAME + ('\n' if release else ' 30000000000\n'))
    held = WAKE_NAME in read('/sys/power/wake_lock').split()
    return {'name': WAKE_NAME, 'held': held, 'expires_without_renewal_seconds': 30,
            'released': release and not held}


def read(path):
    with Path(path).open('r') as stream:
        value = stream.read(65537)
    if len(value) > 65536:
        raise ValueError('Telemetry file exceeds bound')
    return value


def sample():
    result = {'sampled_at': time.monotonic(), 'wall_time': time.time(),
              'battery_percent': None, 'battery_temp_c': None, 'charging': None,
              'memory_available_mb': None, 'cpu_totals': None,
              'thermal_severity': None, 'cpu_pressure_avg10': None,
              'memory_pressure_avg10': None, 'errors': {}}

    def attempt(key, operation):
        try:
            result[key] = operation()
        except (OSError, ValueError, IndexError, KeyError, subprocess.SubprocessError) as error:
            result['errors'][key] = type(error).__name__

    battery = '/sys/class/power_supply/battery/'
    attempt('battery_percent', lambda: float(read(battery + 'capacity')))
    attempt('battery_temp_c', lambda: float(read(battery + 'temp')) / 10)
    attempt('charging', lambda: read(battery + 'status').strip() in ('Charging', 'Full'))

    def memory():
        values = dict(line.split(':', 1) for line in read('/proc/meminfo').splitlines())
        return int(values['MemAvailable'].split()[0]) / 1024

    def cpu():
        values = [int(value) for value in read('/proc/stat').splitlines()[0].split()[1:9]]
        if len(values) != 8 or any(value < 0 for value in values):
            raise ValueError('Unexpected CPU counters')
        return {'total': sum(values), 'idle': values[3] + values[4]}

    def pressure(kind):
        text = read('/proc/pressure/' + kind)
        match = re.search(r'^some avg10=([0-9.]+)', text)
        if not match:
            raise ValueError('Pressure reading unavailable')
        return float(match[1])

    def thermal():
        process = subprocess.run(['/system/bin/dumpsys', 'thermalservice'],
                                 stdin=subprocess.DEVNULL, capture_output=True,
                                 text=True, timeout=2)
        if process.returncode:
            raise ValueError('Thermal service unavailable')
        match = re.search(r'Thermal Status:\s*(\d+)', process.stdout[:65536])
        if not match or not 0 <= int(match[1]) <= 6:
            raise ValueError('Thermal severity unavailable')
        return int(match[1])

    attempt('memory_available_mb', memory)
    attempt('cpu_totals', cpu)
    attempt('cpu_pressure_avg10', lambda: pressure('cpu'))
    attempt('memory_pressure_avg10', lambda: pressure('memory'))
    attempt('thermal_severity', thermal)
    for key, value in list(result.items()):
        if isinstance(value, float) and not math.isfinite(value):
            result[key] = None
            result['errors'][key] = 'NonfiniteReading'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--lease', action='store_true')
    group.add_argument('--release', action='store_true')
    args = parser.parse_args()
    result = {} if args.release else sample()
    if args.lease or args.release:
        try:
            result['wake_lease'] = wake_lease(release=args.release)
        except OSError as error:
            result['wake_lease'] = {'held': False, 'error': str(error),
                                   'expires_without_renewal_seconds': 30}
    print(json.dumps(result, allow_nan=False))
