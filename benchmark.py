#!/usr/bin/env python3
"""Inspect PixelForge, or benchmark its already-running Gemma NPU service.

Default: read-only profile/status/telemetry inventory. --run submits bounded
text-only requests from ordinary Termux. Never starts/stops a service, changes
a profile, executes generated text, or cancels an inference. Reports contain
only this run's prompts/results and newly appended timing metadata.
"""
import argparse
import datetime
import importlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import threading
import time
import uuid

DEFAULT_VOICE = Path('/data/data/com.termux/files/home/ostadix-voice')
TPU_SYSFS = Path('/sys/devices/platform/sswrp_tpu@36000000/36500000.buenos')
ODPM = Path('/sys/bus/iio/devices/iio:device0')
ODPM_UNIT_SOURCE = ('https://android.googlesource.com/kernel/google-modules/soc/gs/'
    '+/204078758737776959ecc85c527a00fd2cabb62f/drivers/iio/power/odpm-whi.c#1315')
PROFILE_FIELDS = ('schema_version', 'backend', 'model_name', 'quantization',
                  'model_path', 'model_bytes', 'model_sha256', 'model_repo',
                  'model_revision', 'runtime_release', 'dispatch_release',
                  'parallel', 'context_per_worker', 'max_output_tokens')
TPU_FIELDS = ('clients', 'groups', 'device_utilization', 'tpu_utilization',
              'inference_count', 'tpu_active_cycle_count', 'tpu_op_count',
              'firmware_crash_count', 'watchdog_timeout_count',
              'power/runtime_status', 'power/runtime_active_time')
CORRECTNESS = (
    ('arithmetic', 'What is 2 plus 3? Reply with only the number.', '5'),
    ('literal', 'Reply with exactly the word READY.', 'READY'),
    ('json_extraction', 'From "blue boxes: 3", extract the color and count. '
     'Reply with only JSON containing exactly color and count.', {'color': 'blue', 'count': 3}),
)
GENERATION_PROMPT = ('Explain how a telephone carries a conversation in four short '
                     'sentences. Use plain language and no headings.')


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def bounded_read(path, limit=8192):
    with Path(path).open('rb') as source:
        value = source.read(limit + 1)
    if len(value) > limit:
        raise ValueError('read exceeds byte limit')
    return value.decode('utf-8', errors='replace').strip()


def read_fields(base, names, errors, prefix=''):
    result = {}
    for name in names:
        try:
            result[name] = bounded_read(base / name)
        except (OSError, ValueError) as error:
            errors[prefix + name] = type(error).__name__
    return result


def integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def battery_values(raw):
    """Linux power_supply units: current µA, voltage µV, battery temp 0.1 C."""
    current, voltage = integer(raw.get('current_now')), integer(raw.get('voltage_now'))
    temperature = integer(raw.get('temp'))
    return {
        'status': raw.get('status'), 'capacity_percent': integer(raw.get('capacity')),
        'current_ua': current, 'voltage_uv': voltage,
        'temperature_c': temperature / 10 if temperature is not None else None,
        'net_battery_discharge_w': (abs(current) * voltage / 1e12
            if raw.get('status', '').lower() == 'discharging' and current is not None
            and voltage is not None and voltage > 0 else None),
        'interpretation': 'Whole-device net battery discharge; not NPU power. '
            'Raw current sign is retained; charging samples are not discharge power.',
    }


def process_sample(pid):
    """Read only process identity, CPU ticks, memory, and selected accelerator paths."""
    result = {'pid': pid}
    if type(pid) is not int or pid <= 0:
        return result
    root = Path('/proc') / str(pid)
    try:
        fields = bounded_read(root / 'stat').rsplit(')', 1)[1].split()
        result.update(starttime_ticks=int(fields[19]), cpu_user_ticks=int(fields[11]),
                      cpu_system_ticks=int(fields[12]), clock_ticks_per_second=os.sysconf('SC_CLK_TCK'))
        status_fields = dict(line.split(':', 1) for line in bounded_read(root / 'status').splitlines() if ':' in line)
        for name in ('VmRSS', 'VmHWM', 'VmSize', 'Threads', 'Uid'):
            result[name] = status_fields.get(name, '').strip()
    except (OSError, ValueError, IndexError) as error:
        result['error'] = type(error).__name__
    return result


def accelerator_process_evidence(pid):
    root = Path('/proc') / str(pid)
    result = {'pid': pid, 'mapped_paths': [], 'device_descriptors': [], 'errors': {}}
    pattern = re.compile(r'edgetpu|darwinn|buenos|tachyon|liblitert|llm_wrapper', re.I)
    try:
        for line in bounded_read(root / 'maps', 2 * 1024 * 1024).splitlines():
            parts = line.split(None, 5)
            if len(parts) == 6 and pattern.search(parts[5]):
                result['mapped_paths'].append(parts[5])
        result['mapped_paths'] = sorted(set(result['mapped_paths']))
    except (OSError, ValueError) as error:
        result['errors']['maps'] = type(error).__name__
    try:
        with os.scandir(root / 'fd') as entries:
            for index, entry in enumerate(entries):
                if index >= 4096:
                    raise ValueError('descriptor limit')
                try:
                    target = os.readlink(entry.path)
                except FileNotFoundError:
                    continue
                if target.startswith('/dev/') and pattern.search(target):
                    result['device_descriptors'].append(target)
    except (OSError, ValueError) as error:
        result['errors']['fd'] = type(error).__name__
    return result


def hardware_snapshot(pid=None):
    result = {'utc': utc_now(), 'monotonic_seconds': time.monotonic(), 'errors': {}}
    errors = result['errors']
    try:
        fields = dict(line.split(':', 1) for line in bounded_read('/proc/meminfo').splitlines() if ':' in line)
        result['memory_kib'] = {name: integer(fields.get(name, '').split()[0])
            for name in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree') if fields.get(name)}
    except (OSError, ValueError) as error:
        errors['meminfo'] = type(error).__name__
    battery = read_fields(Path('/sys/class/power_supply/battery'),
        ('status', 'capacity', 'current_now', 'voltage_now', 'temp'), errors, 'battery/')
    result['battery'] = battery_values(battery)
    result['battery']['raw'] = battery
    result['power_supply_online'] = {}
    for name in ('usb', 'wireless'):
        result['power_supply_online'][name] = read_fields(
            Path('/sys/class/power_supply') / name, ('online',), errors, name + '/').get('online')
    result['thermal'] = []
    try:
        zones = sorted(Path('/sys/class/thermal').glob('thermal_zone*'))[:64]
        for zone in zones:
            raw = read_fields(zone, ('type', 'temp'), errors, zone.name + '/')
            value = integer(raw.get('temp'))
            verified_millidegrees = raw.get('type') in ('battery', 'soc_therm')
            result['thermal'].append({'zone': zone.name, 'type': raw.get('type'),
                'raw_temp': value,
                'unit': 'millidegrees_C' if verified_millidegrees else 'unverified_driver_units',
                'temperature_c': value / 1000 if verified_millidegrees and value is not None
                    and -40000 <= value <= 150000 else None})
    except OSError as error:
        errors['thermal'] = type(error).__name__
    result['tpu_raw'] = read_fields(TPU_SYSFS, TPU_FIELDS, errors, 'tpu/')
    result['odpm_raw'] = read_fields(ODPM,
        ('name', 'enabled_rails', 'energy_value', 'inst_power'), errors, 'odpm/')
    result['odpm_interpretation'] = 'Raw read-only rail telemetry; units unverified by this harness. No computed rail power/energy.'
    if pid:
        result['engine_process'] = process_sample(pid)
    return result


def public_profile(voice_dir):
    try:
        profile = json.loads(bounded_read(voice_dir / 'runtime/local-model.json', 65536))
        if not isinstance(profile, dict):
            raise ValueError('profile is not an object')
        return {key: profile[key] for key in PROFILE_FIELDS if key in profile}
    except (OSError, ValueError) as error:
        return {'error': type(error).__name__}


def public_status(status):
    result = {key: status.get(key) for key in ('running', 'ready', 'active', 'queued',
        'engine_pid', 'daemon_pid', 'service_enabled', 'supervised', 'service_phase')}
    model = status.get('model') or {}
    result['model'] = {key: model[key] for key in PROFILE_FIELDS if key in model}
    limits = status.get('limits') or {}
    result['limits'] = {key: limits[key] for key in ('active', 'queued', 'max_tokens',
        'task_chars', 'result_bytes', 'timeout_seconds') if key in limits}
    return result


def load_client(voice_dir, state_dir):
    if not (voice_dir / 'local_runtime.py').is_file():
        raise ValueError('voice directory has no local_runtime.py')
    sys.path.insert(0, str(voice_dir.resolve()))
    module = importlib.import_module('local_runtime')
    if Path(module.__file__).resolve() != (voice_dir / 'local_runtime.py').resolve():
        raise ValueError('local_runtime was imported from a different directory')
    return module.LocalClient(state_dir=state_dir)


def percentile(values, percentile_value):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value / 100
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def integrate_battery(samples, start=None, end=None):
    """Trapezoid integration of observed net discharge, with strict input guards."""
    samples = [item for item in samples if (start is None or item['monotonic_seconds'] >= start)
               and (end is None or item['monotonic_seconds'] <= end)]
    rejection = {'available': False, 'interpretation': 'Whole-device sampled net battery discharge, not NPU energy.'}
    if len(samples) < 2:
        return dict(rejection, reason='fewer_than_two_samples')
    signs, points = set(), []
    for sample in samples:
        battery = sample.get('battery', {})
        if str(battery.get('status', '')).lower() != 'discharging':
            return dict(rejection, reason='charging_or_status_missing')
        current, voltage = battery.get('current_ua'), battery.get('voltage_uv')
        if type(current) is not int or current == 0 or type(voltage) is not int or voltage <= 0:
            return dict(rejection, reason='current_or_voltage_missing_or_invalid')
        signs.add(1 if current > 0 else -1)
        points.append((sample['monotonic_seconds'], abs(current) * voltage / 1e12))
    if len(signs) != 1:
        return dict(rejection, reason='current_sign_changed')
    joules = 0.0
    for (left_t, left_w), (right_t, right_w) in zip(points, points[1:]):
        gap = right_t - left_t
        if not 0 < gap <= 2:
            return dict(rejection, reason='sample_times_invalid_or_gap_over_two_seconds')
        joules += gap * (left_w + right_w) / 2
    duration = points[-1][0] - points[0][0]
    return {'available': True, 'energy_j': joules, 'mean_net_discharge_w': joules / duration,
        'observed_duration_seconds': duration, 'sample_count': len(points),
        'first_monotonic_seconds': points[0][0], 'last_monotonic_seconds': points[-1][0],
        'current_sign': next(iter(signs)),
        'interpretation': 'Whole-device sampled net battery discharge, not NPU energy. '
            'Integration covers first through last sampled point only; no energy/token estimate.'}


def parse_odpm_energy(text):
    """Parse the entire Google ODPM energy_value format, retaining raw counters.

    Matching syntax alone does not verify the installed CPM driver's units.
    Google's cited driver specifies milliseconds and micro-watt-seconds (µJ).
    Conversion is separately opt-in through energy_delta(abi_assumed=True).
    """
    if not isinstance(text, str) or len(text.encode('utf-8')) > 65536:
        raise ValueError('ODPM input must be text of at most 65536 bytes')
    lines = text.strip().splitlines()
    if not 2 <= len(lines) <= 257 or not re.fullmatch(r't=\d+', lines[0]):
        raise ValueError('ODPM measurement timestamp or channel rows missing')
    timestamp = int(lines[0][2:])
    rails = {}
    pattern = re.compile(r'CH(\d+)\(T=(\d+)\)\[([A-Za-z0-9_.+-]{1,128})\],[ \t]+(\d+)')
    for line in lines[1:]:
        match = pattern.fullmatch(line)
        if match is None:
            raise ValueError('ODPM channel row does not fully match energy_value format')
        channel, duration, name, counter = match.groups()
        key = 'CH' + str(int(channel))
        if key in rails:
            raise ValueError('ODPM duplicate channel')
        if int(duration) > timestamp:
            raise ValueError('ODPM channel duration exceeds measurement timestamp')
        rails[key] = {'channel': int(channel), 'name': name,
                      'duration_ms': int(duration), 'energy_counter_raw': int(counter)}
    return {'timestamp_ms': timestamp, 'rails': rails,
            'format_compatible': 'Google ODPM energy_value syntax',
            'exact_cpm_revision_verified': False, 'unit_source': ODPM_UNIT_SOURCE}


def energy_delta(before_text, after_text, *, abi_assumed=False):
    """Estimate TPU rail joules/watts only with explicit compatible-ABI assumption.

    CH6 S7M_VDD_TPU and CH8 S9M_VDD_TPU_M are summed, not CPU or whole-device
    power. CH13 VSYS_PWR_VBATT is reported separately when valid. Inst_power
    is deliberately ignored: its installed-driver units remain unverified.
    """
    result = {'available': False, 'abi_assumed': abi_assumed is True,
              'exact_cpm_revision_verified': False, 'unit_source': ODPM_UNIT_SOURCE,
              'unit_interpretation': 'ABI-compatible inference: energy_value counters assumed µWs=µJ, timestamps/durations assumed ms. Exact installed CPM revision is unverified.'}
    try:
        before, after = parse_odpm_energy(before_text), parse_odpm_energy(after_text)
    except ValueError:
        return dict(result, reason='malformed_or_partial_odpm_snapshot')
    result.update(before_timestamp_ms=before['timestamp_ms'], after_timestamp_ms=after['timestamp_ms'])
    if abi_assumed is not True:
        return dict(result, reason='explicit_abi_assumption_required')
    elapsed_ms = after['timestamp_ms'] - before['timestamp_ms']
    if elapsed_ms <= 0:
        return dict(result, reason='measurement_timestamp_reset_or_no_elapsed_time')

    def rail_delta(channel, expected_name):
        left, right = before['rails'].get(channel), after['rails'].get(channel)
        if left is None or right is None:
            return {'available': False, 'reason': 'missing_rail', 'channel': channel}
        if left['name'] != expected_name or right['name'] != expected_name:
            return {'available': False, 'reason': 'wrong_or_changed_rail_name', 'channel': channel}
        duration_delta = right['duration_ms'] - left['duration_ms']
        if duration_delta <= 0:
            return {'available': False, 'reason': 'rail_duration_reset_or_no_elapsed_time', 'channel': channel}
        if duration_delta != elapsed_ms:
            return {'available': False, 'reason': 'rail_measurement_origin_changed', 'channel': channel}
        counter_delta = right['energy_counter_raw'] - left['energy_counter_raw']
        if counter_delta < 0:
            return {'available': False, 'reason': 'energy_counter_reset', 'channel': channel}
        return {'available': True, 'channel': channel, 'name': expected_name,
                'before_duration_ms': left['duration_ms'], 'after_duration_ms': right['duration_ms'],
                'before_energy_counter_raw': left['energy_counter_raw'],
                'after_energy_counter_raw': right['energy_counter_raw'],
                'energy_counter_delta_raw': counter_delta, 'energy_j': counter_delta / 1e6,
                'elapsed_seconds': duration_delta / 1000,
                'mean_power_w': counter_delta / 1000 / duration_delta}

    rails = [rail_delta('CH6', 'S7M_VDD_TPU'), rail_delta('CH8', 'S9M_VDD_TPU_M')]
    for rail in rails:
        if not rail['available']:
            return dict(result, reason=rail['reason'], rejected_channel=rail['channel'])
    total_joules = sum(rail['energy_j'] for rail in rails)
    result.update(available=True, elapsed_seconds=elapsed_ms / 1000,
                  tpu={'energy_j': total_joules, 'mean_power_w': total_joules / (elapsed_ms / 1000),
                       'rails': rails,
                       'interpretation': 'Combined CH6 TPU and CH8 TPU_M rail energy; all TPU clients may contribute. Excludes host CPU and other rails.'},
                  battery_rail=rail_delta('CH13', 'VSYS_PWR_VBATT'))
    return result


def native_metrics(text, pid, attributable, usage=None):
    """Keep timing numbers only, never runtime log text; require one unique block."""
    if not attributable:
        return {'available': False, 'reason': 'queue_overlap_or_engine_identity_changed'}
    host = re.findall(r'^host_preflight_seconds=([0-9.]+) generation_seconds=([0-9.]+) prompt_tokens=(\d+)$', text, re.M)
    first = re.findall(r'^request_pid=(\d+) time_to_first_token_seconds=([0-9.]+) completion_tokens=(-?\d+)$', text, re.M)
    decode = re.findall(r'^decode_turn=(\d+) tokens=(\d+) tokens_per_second=([0-9.]+)$', text, re.M)
    if len(host) != 1 or len(first) != 1 or int(first[0][0]) != pid or not decode:
        return {'available': False, 'reason': 'new_log_does_not_have_one_unambiguous_request'}
    values = [float(value) for value in host[0][:2]] + [float(first[0][1])]
    if not all(math.isfinite(value) and value >= 0 for value in values):
        return {'available': False, 'reason': 'invalid_native_timing'}
    if usage is not None:
        expected_tokens, expected_elapsed = usage.get('completion_tokens'), usage.get('elapsed_seconds')
        if type(expected_tokens) is int and int(first[0][2]) != expected_tokens:
            return {'available': False, 'reason': 'native_token_count_does_not_match_job'}
        if type(expected_elapsed) in (int, float) and math.isfinite(expected_elapsed):
            difference = expected_elapsed - values[0] - values[1]
            if not -.00001 <= difference <= max(.05, expected_elapsed * .1):
                return {'available': False, 'reason': 'native_elapsed_does_not_match_job'}
    turns = [{'turn': int(turn), 'tokens': int(tokens), 'tokens_per_second': float(rate)}
             for turn, tokens, rate in decode]
    if [item['turn'] for item in turns] != list(range(len(turns))) or any(
            not math.isfinite(item['tokens_per_second']) or item['tokens_per_second'] < 0 for item in turns):
        return {'available': False, 'reason': 'invalid_or_duplicate_decode_turns'}
    return {'available': True, 'host_preflight_seconds': values[0], 'generation_seconds': values[1],
        'prompt_tokens': int(host[0][2]), 'time_to_first_token_seconds': values[2],
        'completion_tokens': int(first[0][2]), 'decode_turns': turns,
        'attribution': 'One newly appended timing block, stable PID, no observed queue overlap; concurrent arrivals cannot be ruled out absolutely.'}


class NewLog:
    """Open at EOF before submission; never read previous requests or log text out."""
    def __init__(self, path):
        self.source = None
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
                os.close(fd)
                return
            self.source = os.fdopen(fd, 'rb')
            self.offset = self.source.seek(0, os.SEEK_END)
        except OSError:
            pass

    def finish(self, pid, attributable, usage=None):
        if self.source is None:
            return {'available': False, 'reason': 'new_log_unavailable'}
        try:
            if os.fstat(self.source.fileno()).st_size < self.offset:
                return {'available': False, 'reason': 'log_truncated'}
            data = self.source.read(262145)
            if len(data) > 262144:
                return {'available': False, 'reason': 'new_log_exceeds_bound'}
            return native_metrics(data.decode('utf-8', errors='replace'), pid, attributable, usage)
        except OSError:
            return {'available': False, 'reason': 'new_log_read_failed'}
        finally:
            self.source.close()
            self.source = None


class Monitor:
    def __init__(self, client, pid):
        self.client, self.pid = client, pid
        self.phase = 'baseline'
        self.samples = []
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.collect, name='benchmark-telemetry', daemon=True)

    def collect(self):
        while not self.stop_event.is_set():
            sample = hardware_snapshot(self.pid)
            sample['phase'] = self.phase
            try:
                sample['service'] = public_status(self.client.status())
            except Exception as error:
                sample['service_error'] = type(error).__name__
            self.samples.append(sample)
            self.stop_event.wait(.5)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=3)


def pause(seconds, deadline):
    end = min(time.monotonic() + seconds, deadline)
    while time.monotonic() < end:
        time.sleep(min(.1, end - time.monotonic()))


def idle_ready(client, wait_seconds, deadline, pid):
    until = min(time.monotonic() + wait_seconds, deadline)
    while True:
        status = public_status(client.status())
        if not status['ready'] or status['model'].get('backend') != 'litert-npu' or status['engine_pid'] != pid:
            return status, 'existing_npu_engine_not_ready_or_changed'
        if not status.get('active') and not status.get('queued'):
            return status, None
        if time.monotonic() >= until:
            return status, 'service_busy_no_request_submitted'
        pause(.25, until)


def check_answer(name, text, expected):
    if name == 'arithmetic':
        return text.strip().strip('. ') == expected
    if name == 'json_extraction':
        try:
            value = json.loads(text.strip())
            return value == expected and type(value.get('count')) is int
        except (ValueError, AttributeError):
            return False
    return text.strip() == expected


def request(client, args, owner, pid, name, prompt, expected, deadline):
    status, reason = idle_ready(client, args.busy_wait_seconds, deadline, pid)
    record = {'name': name, 'prompt': prompt, 'max_tokens': args.max_tokens,
              'status_before': status, 'success': False}
    if reason or time.monotonic() >= deadline:
        record['skipped'] = reason or 'total_budget_exhausted'
        return record, True
    log = NewLog(args.state_dir / 'engine.log')
    started = time.monotonic()
    record['client_started_monotonic_seconds'] = started
    try:
        job = client.submit(owner, prompt, max_tokens=args.max_tokens)
        record['job_id'] = job['id']
        until = min(started + args.request_timeout_seconds, deadline)
        while True:
            result = client.read(owner, job['id'], wait=min(1, max(0, until - time.monotonic())), limit=65536)
            if result['state'] not in ('queued', 'running'):
                break
            if time.monotonic() >= until:
                record['pending_job'] = {'id': job['id'], 'owner': owner, 'state': result['state']}
                record['error'] = 'Observation deadline exceeded; job was not cancelled and may finish in the service.'
                return record, True
        finished = time.monotonic()
        record.update(client_finished_monotonic_seconds=finished,
                      client_elapsed_seconds=finished - started, state=result['state'],
                      result=result.get('result', ''), result_truncated=result.get('result_truncated', False),
                      usage=result.get('usage', {}))
        usage = record['usage']
        after = public_status(client.status())
        record['status_after'] = after
        record['success'] = (result['state'] == 'completed' and bool(record['result'].strip())
                             and usage.get('backend') == 'litert-npu' and after['engine_pid'] == pid)
        if not record['success']:
            record['error'] = 'Request failed, returned no NPU answer, or engine identity changed.'
        if expected is not None:
            record['correct'] = record['success'] and not record['result_truncated'] and check_answer(name, record['result'], expected)
        created, job_started, job_finished = (result.get(key) for key in ('created', 'started', 'finished'))
        if all(type(value) in (int, float) and math.isfinite(value) for value in (created, job_started, job_finished)):
            record['service_queue_seconds'] = max(0, job_started - created)
            record['service_execution_seconds'] = max(0, job_finished - job_started)
            record['service_total_seconds'] = max(0, job_finished - created)
        tokens, elapsed = usage.get('completion_tokens'), usage.get('elapsed_seconds')
        if type(tokens) is int and tokens >= 0 and type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed > 0:
            record['effective_completion_tokens_per_second'] = tokens / elapsed
            record['throughput_interpretation'] = 'Completion tokens divided by native request time including preflight/prefill; not pure decode speed.'
        attributable = (record['success'] and not after.get('active') and not after.get('queued')
                        and record.get('service_queue_seconds', 1) <= .05)
        record['native_metrics'] = log.finish(pid, attributable, usage)
        return record, not record['success']
    except BaseException as error:
        record['error'] = type(error).__name__ + ': request observation failed; service was not stopped or changed.'
        if record.get('job_id'):
            record['pending_job'] = {'id': record['job_id'], 'owner': owner, 'state': 'unknown'}
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            record['interrupted'] = True
        return record, True
    finally:
        if log.source is not None:
            log.source.close()


def summary(report):
    records = report['requests']
    measured = [item for item in records if item['name'].startswith('generation_') and item.get('success')]
    checks = [item for item in records if item['name'] in {case[0] for case in CORRECTNESS}]
    result = {'submitted_requests': sum('job_id' in item for item in records),
        'completed_requests': sum(item.get('success', False) for item in records),
        'correctness_passed': sum(item.get('correct', False) for item in checks),
        'correctness_expected': len(CORRECTNESS), 'measured_generation_requests': len(measured),
        'warmup_excluded': True}
    for key in ('client_elapsed_seconds', 'service_queue_seconds', 'service_execution_seconds',
                'effective_completion_tokens_per_second'):
        values = [item[key] for item in measured if key in item]
        result[key] = {'sample_count': len(values), 'median': percentile(values, 50), 'p95': percentile(values, 95)}
    native = [item['native_metrics'] for item in measured if item.get('native_metrics', {}).get('available')]
    values = [item['time_to_first_token_seconds'] for item in native]
    result['native_time_to_first_token_seconds'] = {'sample_count': len(values), 'median': percentile(values, 50), 'p95': percentile(values, 95)}
    result['functional_pass'] = (len(checks) == len(CORRECTNESS) and all(item.get('correct') for item in checks)
        and len(measured) == report['settings']['repeats'] and all(item.get('success') for item in records))
    return result


def finish_measurements(report):
    samples = report.get('hardware_samples', [])
    background = any((item.get('service', {}).get('queued', 0) or
        (item.get('phase') != 'inference' and item.get('service', {}).get('active', 0))) for item in samples)
    report['background_service_work_observed'] = bool(background)
    report['battery_baseline'] = integrate_battery([item for item in samples if item.get('phase') == 'baseline'])
    report['battery_after'] = integrate_battery([item for item in samples if item.get('phase') == 'after'])
    successful = [item for item in report['requests'] if item.get('success')]
    if not report['summary']['functional_pass'] or background or not successful:
        report['battery_inference'] = {'available': False,
            'reason': 'incomplete_or_failed_suite_or_background_service_work',
            'interpretation': 'Raw samples retained; no NPU energy or energy/token claim.'}
    else:
        report['battery_inference'] = integrate_battery(samples,
            min(item['client_started_monotonic_seconds'] for item in successful),
            max(item['client_finished_monotonic_seconds'] for item in successful))
    process = [item.get('engine_process', {}) for item in samples]
    process = [item for item in process if 'starttime_ticks' in item and 'cpu_user_ticks' in item]
    if len(process) >= 2 and len({(item['pid'], item['starttime_ticks']) for item in process}) == 1:
        first, last = process[0], process[-1]
        ticks = last['cpu_user_ticks'] + last['cpu_system_ticks'] - first['cpu_user_ticks'] - first['cpu_system_ticks']
        if ticks >= 0:
            report['engine_cpu_seconds_observed'] = ticks / first['clock_ticks_per_second']
            report['engine_cpu_interpretation'] = 'Engine process CPU time across observed phases; auxiliary CPU work is expected in NPU inference.'


def run_benchmark(args, client, report):
    initial = public_status(client.status())
    report['initial_status'] = initial
    if not initial['ready'] or initial['model'].get('backend') != 'litert-npu' or not initial['engine_pid']:
        report['error'] = 'Existing service must already be ready with the litert-npu backend. No service change made.'
        return 1
    pid = initial['engine_pid']
    report['accelerator_process_evidence_before'] = accelerator_process_evidence(pid)
    report['owner'] = 'pixelforge-benchmark-' + uuid.uuid4().hex
    deadline = time.monotonic() + args.total_budget_seconds
    monitor = Monitor(client, pid)
    monitor.start()
    try:
        pause(args.baseline_seconds, deadline)
        monitor.phase = 'inference'
        cases = [('warmup', 'Reply with exactly the word READY.', None)] + list(CORRECTNESS)
        cases += [('generation_' + str(index + 1), GENERATION_PROMPT, None) for index in range(args.repeats)]
        for name, prompt, expected in cases:
            item, stop = request(client, args, report['owner'], pid, name, prompt, expected, deadline)
            report['requests'].append(item)
            if stop:
                break
        monitor.phase = 'after'
        pause(args.after_seconds, deadline)
    except (KeyboardInterrupt, SystemExit):
        report['interrupted'] = True
    finally:
        monitor.stop()
        report['hardware_samples'] = list(monitor.samples)
    report['final_status'] = public_status(client.status())
    report['accelerator_process_evidence_after'] = accelerator_process_evidence(pid)
    report['summary'] = summary(report)
    finish_measurements(report)
    return 0 if report['summary']['functional_pass'] else 1


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--run', action='store_true', help='Submit a bounded live suite; requires ordinary Termux and an already-ready NPU service')
    modes.add_argument('--collect-seconds', type=float, help='Read-only hardware collection for 0..600 seconds at 0.5 s intervals; root or app, no service access')
    parser.add_argument('--pid', type=int, help='Optional verified engine PID for read-only collection of CPU/memory counters')
    parser.add_argument('--voice-dir', type=Path, default=DEFAULT_VOICE)
    parser.add_argument('--state-dir', type=Path, default=Path.home() / '.local/state/ostadix-voice-local')
    parser.add_argument('--output', type=Path, help='New private JSON report; existing files are never overwritten')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--max-tokens', type=int, default=96)
    parser.add_argument('--baseline-seconds', type=float, default=10)
    parser.add_argument('--after-seconds', type=float, default=10)
    parser.add_argument('--busy-wait-seconds', type=float, default=10)
    parser.add_argument('--request-timeout-seconds', type=float, default=45)
    parser.add_argument('--total-budget-seconds', type=float, default=240)
    args = parser.parse_args(argv)
    for name, low, high in (('repeats', 1, 20), ('max_tokens', 8, 128),
        ('baseline_seconds', 0, 30), ('after_seconds', 0, 30), ('busy_wait_seconds', 0, 30),
        ('request_timeout_seconds', 1, 120), ('total_budget_seconds', 5, 600)):
        value = getattr(args, name)
        if not math.isfinite(value) or not low <= value <= high:
            parser.error(name.replace('_', '-') + f' must be between {low} and {high}')
    if args.run and os.getuid() == 0:
        parser.error('--run requires ordinary Termux, outside root; no requests submitted')
    if args.run and args.output is None:
        parser.error('--run requires --output naming a new JSON report')
    if args.collect_seconds is not None:
        if not math.isfinite(args.collect_seconds) or not 0 <= args.collect_seconds <= 600:
            parser.error('collect-seconds must be between 0 and 600')
        if args.output is None:
            parser.error('--collect-seconds requires --output naming a new private JSON report')
    if args.pid is not None and (args.pid <= 0 or args.collect_seconds is None):
        parser.error('--pid must be positive and used with --collect-seconds')
    args.voice_dir, args.state_dir = args.voice_dir.expanduser().resolve(), args.state_dir.expanduser().resolve()
    return args


def collect_hardware(seconds, pid=None):
    """Bounded public hardware reads only; never imports or contacts a service."""
    samples = []
    started = time.monotonic()
    try:
        while True:
            samples.append(hardware_snapshot(pid))
            remaining = started + seconds - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(.5, remaining))
    except KeyboardInterrupt:
        pass
    return samples


def main(argv=None):
    args = parse_args(argv)
    report = {'schema': 'pixelforge.device-benchmark/v1', 'started_utc': utc_now(),
        'mode': ('live_existing_npu' if args.run else 'read_only_telemetry_collection'
                 if args.collect_seconds is not None else 'read_only_inventory'), 'collector_uid': os.getuid(),
        'voice_dir': str(args.voice_dir), 'state_dir': str(args.state_dir),
        'profile': public_profile(args.voice_dir), 'requests': [],
        'settings': {key: getattr(args, key) for key in ('repeats', 'max_tokens', 'baseline_seconds',
            'after_seconds', 'busy_wait_seconds', 'request_timeout_seconds', 'total_budget_seconds')},
        'interpretation': 'This is a small functionality/performance smoke benchmark, not a general model ranking. '
            'TPU counters are device-wide supporting evidence; CPU tokenization and orchestration are expected. '
            'Whole-device battery readings do not isolate NPU energy. No cloud calls or model output execution.'}
    output = None
    exit_code = 0
    try:
        if args.output:
            fd = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            os.fchmod(fd, 0o600)
            output = os.fdopen(fd, 'w', encoding='utf-8')
        if args.collect_seconds is not None:
            report['hardware_samples'] = collect_hardware(args.collect_seconds, args.pid)
            report['collection_methodology'] = {'requested_duration_seconds': args.collect_seconds,
                'sampling_interval_seconds': .5, 'requested_pid': args.pid,
                'read_only': True, 'service_contacted': False,
                'interpretation': 'Independent collector for parallel ordinary-Termux benchmark runs. '
                    'Correlate monotonic sample timestamps with request windows. '
                    'Missing permissions are recorded; no Android permissions or services are changed.'}
        elif os.getuid() == 0:
            report['status_access'] = 'Not attempted as root: service requires its ordinary Termux owner. Inventory does not start services.'
            report['hardware_sample'] = hardware_snapshot()
        else:
            client = load_client(args.voice_dir, args.state_dir)
            if args.run:
                exit_code = run_benchmark(args, client, report)
            else:
                report['status'] = public_status(client.status())
                report['hardware_sample'] = hardware_snapshot(report['status'].get('engine_pid'))
    except (OSError, ValueError, RuntimeError) as error:
        report['error'] = type(error).__name__ + ': ' + str(error)[:250]
        exit_code = 1
    finally:
        report['finished_utc'] = utc_now()
        if output:
            json.dump(report, output, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
            output.close()
    if args.output:
        print(json.dumps({'report': str(args.output), 'mode': report['mode'],
            'summary': report.get('summary'), 'error': report.get('error')}, allow_nan=False))
    else:
        print(json.dumps(report, indent=2, allow_nan=False))
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
