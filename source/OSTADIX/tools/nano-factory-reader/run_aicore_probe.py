#!/usr/bin/env python3
"""Send one explicit local-probe broadcast; retain actual AICore process evidence.

Requires the separately installed, explicitly armed receiver and staged files.
Generation requires --generate and a matching private generation manifest.
This runner does not start AICore, change UIDs, install hooks, or stage models.
It never resends a broadcast or follows a replacement PID.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import select
import signal
import stat
import subprocess
import sys
import threading
import time
from android_pidfd import open_pidfd, send_pidfd_signal

PACKAGE = 'com.google.android.aicore'
EXPECTED_UID = 10173
ACTION = 'org.ostadix.aicore.NANO_LOCAL_PROBE'
EVENT_PATH = Path('/data/user/0/com.google.android.aicore/files/ostadix-nano-probe/events.jsonl')
MAX_EVENT_BYTES = 16 * 1024 * 1024
LOG_PATTERN = re.compile(
    r'ostadix|nano|edgetpu|tachyon|inferenceexception|(?:^|[^a-z])jni|'
    r'model_loader|llm_|tokeniz|native(?:Load|Create|Free|Unload)|'
    r'Stack trace|Source Location|third_party/|java/com/|\s+at\s+|\s+@\s+', re.I)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='milliseconds')


class RequestDeadline:
    """One immutable lease; both clocks must remain within the original budget."""
    def __init__(self, seconds):
        self.started = time.monotonic()
        self.started_boottime = time.clock_gettime(time.CLOCK_BOOTTIME)
        self.monotonic_deadline = self.started + seconds
        self.deadline_elapsed_ms = int((self.started_boottime + seconds) * 1000)

    def remaining(self):
        return min(self.monotonic_deadline - time.monotonic(),
                   self.deadline_elapsed_ms / 1000 - time.clock_gettime(time.CLOCK_BOOTTIME))


def broadcast_command(request_id, deadline_elapsed_ms, generate):
    command = ['/system/bin/am', 'broadcast', '-a', ACTION, '-p', PACKAGE,
               '--es', 'request_id', request_id,
               '--el', 'deadline_elapsed_ms', str(deadline_elapsed_ms)]
    if generate:
        command.extend(['--ez', 'allow_generation', 'true'])
    return command


def record_generation_evidence(report, event):
    """Null means the process stopped between native entry and durable evidence."""
    dispatched = event.get('inference_dispatched')
    returned = event.get('inference_returned')
    if dispatched is None and returned is None and not report['generation_requested']:
        return  # Older load-only receivers did not emit these fields.
    if type(dispatched) is not bool or type(returned) is not bool:
        raise RuntimeError('Correlated event lacks boolean inference dispatch/return evidence')
    if returned and not dispatched:
        raise RuntimeError('Inference return evidence lacks dispatch evidence')
    if (dispatched or event.get('phase') == 'generation_enter') and not report['generation_requested']:
        raise RuntimeError('Unexpected generation evidence without --generate')
    for key, value in (('generation_dispatched', dispatched), ('generation_returned', returned)):
        if report[key] is True and not value:
            raise RuntimeError('Inference evidence regressed: ' + key)
        report[key] = value
    if event.get('phase') == 'generation_enter':
        # The Java entry event is persisted before Method.invoke. A kill after
        # this event cannot establish whether invocation or return occurred.
        if not dispatched:
            report['generation_dispatched'] = None
        if not returned:
            report['generation_returned'] = None


def process_identity(pid):
    root = Path('/proc') / str(pid)
    command = root.joinpath('cmdline').read_bytes().split(b'\0', 1)[0].decode('utf-8')
    status = root.joinpath('status').read_text()
    uid_line = next(line for line in status.splitlines() if line.startswith('Uid:'))
    uids = [int(value) for value in uid_line.split()[1:]]
    # stat comm may contain spaces/parentheses; fields following its last ')' start at #3.
    stat_fields = root.joinpath('stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, cmdline=command, uids=uids, starttime_ticks=int(stat_fields[19]),
                selinux_context=root.joinpath('attr/current').read_text().strip())


def observe_aicore():
    matches = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            if entry.joinpath('cmdline').read_bytes().split(b'\0', 1)[0] != PACKAGE.encode():
                continue
            identity = process_identity(int(entry.name))
        except (OSError, ValueError, StopIteration, UnicodeError):
            continue
        matches.append(identity)
    if len(matches) != 1:
        raise RuntimeError(f'Expected exactly one already-running {PACKAGE} main process; found {len(matches)}')
    identity = matches[0]
    if identity['uids'] != [EXPECTED_UID] * 4:
        raise RuntimeError(f'AICore main process has unexpected real/effective/saved/fs UIDs: {identity["uids"]}')
    return identity


def process_exited(pidfd):
    return bool(select.select([pidfd], [], [], 0)[0])


def resource_snapshot(pid, elapsed):
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, _, value = line.partition(':')
        if name in ('MemAvailable', 'SwapFree'):
            memory[name + '_kB'] = int(value.split()[0])
    if 'MemAvailable_kB' not in memory:
        raise RuntimeError('MemAvailable unavailable; cannot enforce memory guard')
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            name, _, value = line.partition(':')
            if name in ('VmRSS', 'VmHWM'):
                memory[name + '_kB'] = int(value.split()[0])
    except FileNotFoundError:
        pass
    result = subprocess.run(['/system/bin/dumpsys', 'thermalservice'],
                            capture_output=True, text=True, timeout=1.5, check=True)
    match = re.search(r'^\s*Thermal Status:\s*(\d+)\s*$', result.stdout, re.M)
    if match is None:
        raise RuntimeError('Thermal Status unavailable; cannot enforce thermal guard')
    return dict(utc=utc_now(), elapsed_seconds=round(elapsed, 3),
                thermal_status=int(match.group(1)), **memory)


def resource_stop(snapshot):
    if snapshot['MemAvailable_kB'] < 512 * 1024:
        return 'host_low_memory_guard'
    if snapshot['thermal_status'] >= 3:
        return 'host_thermal_guard'
    return None


class EventTail:
    """Follow only the original append-only event file, with bounded input."""
    def __init__(self, request_id):
        self.request_id = request_id
        self.identity = None
        self.offset = 0
        self.buffer = b''
        self.bytes_read = 0
        self._baseline()

    @staticmethod
    def _open():
        fd = os.open(EVENT_PATH, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != EXPECTED_UID:
            os.close(fd)
            raise RuntimeError('Probe event path must be a regular file owned by the actual AICore UID')
        return fd, info

    def _baseline(self):
        try:
            fd, info = self._open()
        except FileNotFoundError:
            return
        try:
            if info.st_size > MAX_EVENT_BYTES:
                raise RuntimeError('Existing event history exceeds bounded scan size')
            data = b''
            while len(data) < info.st_size:
                chunk = os.read(fd, min(65536, info.st_size - len(data)))
                if not chunk:
                    break
                data += chunk
            for line in data.splitlines():
                try:
                    event = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if isinstance(event, dict) and event.get('request_id') == self.request_id:
                    raise RuntimeError('Request ID already exists in AICore event history; choose a fresh ID')
            self.identity = (info.st_dev, info.st_ino)
            self.offset = len(data)
            if data and not data.endswith(b'\n'):
                raise RuntimeError('Event history ends with an incomplete record; no broadcast sent')
        finally:
            os.close(fd)

    def poll(self):
        try:
            fd, info = self._open()
        except FileNotFoundError:
            if self.identity is not None:
                raise RuntimeError('Event file disappeared during observation')
            return []
        try:
            identity = (info.st_dev, info.st_ino)
            if self.identity is None:
                self.identity = identity
            if identity != self.identity or info.st_size < self.offset:
                raise RuntimeError('Event file replaced/truncated; refusing ambiguous request evidence')
            pending = info.st_size - self.offset
            if self.bytes_read + pending > MAX_EVENT_BYTES:
                raise RuntimeError('New event stream exceeded host byte limit')
            os.lseek(fd, self.offset, os.SEEK_SET)
            data = bytearray()
            while len(data) < pending:
                chunk = os.read(fd, min(65536, pending - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
            self.offset += len(data)
            self.bytes_read += len(data)
            self.buffer += data
            lines = self.buffer.split(b'\n')
            self.buffer = lines.pop()
            events = []
            for line in lines:
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise RuntimeError('Event record is not an object')
                if event.get('request_id') == self.request_id:
                    events.append(event)
            return events
        finally:
            os.close(fd)


def stop_child(process):
    if process is None:
        return
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait(timeout=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request-id', required=True)
    parser.add_argument('--evidence-prefix', type=Path, required=True)
    parser.add_argument('--deadline-seconds', type=int, default=120)
    parser.add_argument('--generate', action='store_true',
                        help='explicitly allow the staged manifest generation request')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', args.request_id):
        parser.error('request-id must be 1..80 ASCII letters/digits/underscore/dot/hyphen, starting alphanumeric')
    if not 1 <= args.deadline_seconds <= 120:
        parser.error('deadline-seconds must be between 1 and 120')

    suffixes = ('.json', '.stdout', '.logcat', '.logcat.stderr', '.broadcast.stdout', '.broadcast.stderr')
    paths = {suffix: Path(str(args.evidence_prefix) + suffix) for suffix in suffixes}
    if any(path.exists() or path.is_symlink() for path in paths.values()):
        parser.error('evidence prefix already used; existing artifacts will not be overwritten')
    args.evidence_prefix.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    try:
        for suffix, path in paths.items():
            files[suffix] = path.open('x', encoding='utf-8')
    except BaseException:
        for stream in files.values():
            stream.close()
        raise

    report = dict(schema='ostadix.aicore-process-probe-run/v1', request_id=args.request_id,
                  started_utc=utc_now(), deadline_seconds=args.deadline_seconds,
                  automatic_retry=False, generation_requested=args.generate,
                  generation_dispatched=False, generation_returned=False,
                  generation_evidence_note='null means native entry was observed without later dispatch/return evidence',
                  broadcast_sent=False,
                  event_source=str(EVENT_PATH), evidence_files={k: str(v) for k, v in paths.items()},
                  resource_snapshots=[], signal_actions=[], complete_event=None)
    pidfd = None
    identity = None
    broadcast = None
    logcat = None
    log_thread = None
    tail = None
    native_active = False
    cancellation_signal = None
    stop_reason = None
    deadline = RequestDeadline(args.deadline_seconds)
    started = deadline.started
    report['deadline_elapsed_ms'] = deadline.deadline_elapsed_ms
    previous_handlers = {}
    event_count = 0
    correlation_seen = False

    def cancel(signum, _frame):
        nonlocal cancellation_signal
        cancellation_signal = signum

    def collect_events():
        nonlocal native_active, event_count, correlation_seen
        for event in tail.poll():
            if type(event.get('pid')) is not int or event['pid'] != identity['pid']:
                raise RuntimeError('Correlated event PID differs from the pinned AICore process')
            if type(event.get('uid')) is not int or event['uid'] != EXPECTED_UID:
                raise RuntimeError('Correlated event UID differs from actual AICore UID')
            if report['complete_event'] is not None:
                raise RuntimeError('Correlated event followed terminal completion')
            files['.stdout'].write(json.dumps(event, separators=(',', ':')) + '\n')
            files['.stdout'].flush()
            event_count += 1
            correlation_seen = True
            if type(event.get('native_call_active')) is not bool:
                raise RuntimeError('Correlated event lacks required boolean native_call_active')
            native_active = event['native_call_active']
            record_generation_evidence(report, event)
            if event.get('phase') == 'complete':
                if type(event.get('success')) is not bool:
                    raise RuntimeError('Terminal complete event lacks a boolean success value')
                report['complete_event'] = event
                native_active = False
                if (event['success'] and args.generate
                        and (report['generation_dispatched'] is not True
                             or report['generation_returned'] is not True)):
                    raise RuntimeError('Successful generation completion lacks actual dispatch/return evidence')

    def terminate_observed_native_call(reason):
        nonlocal native_active
        if (pidfd is None or identity is None or not correlation_seen
                or report['complete_event'] is not None):
            report['signal_actions'].append(dict(utc=utc_now(), action='not_signalled', reason=reason,
                correlated_request_seen=correlation_seen, native_call_active=native_active))
            return
        if process_exited(pidfd):
            report['signal_actions'].append(dict(utc=utc_now(), action='already_exited', reason=reason,
                                                pid=identity['pid']))
            return
        current = process_identity(identity['pid'])
        if current != identity:
            raise RuntimeError('Pinned process identity changed; refusing signal')
        send_pidfd_signal(pidfd, signal.SIGKILL)
        report['signal_actions'].append(dict(utc=utc_now(), action='pidfd_SIGKILL', reason=reason,
            pid=identity['pid'], uid=EXPECTED_UID, native_call_active=native_active,
            correlated_request_unfinished=True))
        report['terminated_process_exit_observed'] = bool(select.select([pidfd], [], [], 2)[0])
        native_active = False

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, cancel)
        identity = observe_aicore()
        pidfd = open_pidfd(identity['pid'])
        if process_exited(pidfd) or process_identity(identity['pid']) != identity:
            raise RuntimeError('AICore process changed while obtaining pidfd; no broadcast sent')
        report['process_identity'] = identity
        report['process_observed_utc'] = utc_now()
        tail = EventTail(args.request_id)
        first_snapshot = resource_snapshot(identity['pid'], time.monotonic() - started)
        report['resource_snapshots'].append(first_snapshot)
        stop_reason = resource_stop(first_snapshot)
        if stop_reason:
            raise RuntimeError('Resource guard rejected dispatch: ' + stop_reason)
        if cancellation_signal is not None:
            stop_reason = 'host_cancelled_before_dispatch'
            raise RuntimeError(stop_reason)

        log_command = ['/system/bin/logcat', '--pid=' + str(identity['pid']), '-T', '1', '-v', 'threadtime']
        report['logcat_command'] = log_command
        logcat = subprocess.Popen(log_command, stdout=subprocess.PIPE,
                                  stderr=files['.logcat.stderr'], text=True, errors='replace')

        def collect_logcat():
            assert logcat.stdout is not None
            for line in logcat.stdout:
                if LOG_PATTERN.search(line):
                    files['.logcat'].write(line)
                    files['.logcat'].flush()

        log_thread = threading.Thread(target=collect_logcat, name='aicore-probe-logcat', daemon=True)
        log_thread.start()
        if deadline.remaining() <= 0:
            stop_reason = 'host_deadline_before_dispatch'
            raise RuntimeError(stop_reason)
        if process_exited(pidfd):
            stop_reason = 'observed_aicore_process_exited_before_dispatch'
            raise RuntimeError(stop_reason)
        # Android elapsedRealtime and Linux CLOCK_BOOTTIME include suspend time.
        # The receiver rejects delayed deliveries after this lease expires.
        command = broadcast_command(args.request_id, deadline.deadline_elapsed_ms, args.generate)
        report['broadcast_command'] = command
        report['broadcast_started_utc'] = utc_now()
        broadcast = subprocess.Popen(command, stdout=files['.broadcast.stdout'],
                                     stderr=files['.broadcast.stderr'])
        report['broadcast_sent'] = True
        next_resource_check = time.monotonic() + 1
        while True:
            collect_events()
            if report['complete_event'] is not None:
                stop_reason = 'complete'
                break
            if process_exited(pidfd):
                # Final appended completion may race the process-exit notification.
                collect_events()
                stop_reason = 'complete' if report['complete_event'] is not None else 'observed_aicore_process_exited'
                break
            elapsed = time.monotonic() - started
            if cancellation_signal is not None:
                stop_reason = 'host_cancelled_signal_' + str(cancellation_signal)
            elif deadline.remaining() <= 0:
                stop_reason = 'host_deadline'
            elif time.monotonic() >= next_resource_check:
                snapshot = resource_snapshot(identity['pid'], elapsed)
                report['resource_snapshots'].append(snapshot)
                stop_reason = resource_stop(snapshot)
                next_resource_check = time.monotonic() + 1
                if cancellation_signal is not None:
                    stop_reason = 'host_cancelled_signal_' + str(cancellation_signal)
                elif deadline.remaining() <= 0:
                    stop_reason = 'host_deadline'
            if stop_reason:
                collect_events()
                if report['complete_event'] is None:
                    terminate_observed_native_call(stop_reason)
                else:
                    stop_reason = 'complete'
                break
            if broadcast.poll() is not None and broadcast.returncode != 0 and not correlation_seen:
                stop_reason = 'broadcast_command_failed'
                break
            time.sleep(min(0.1, max(0.001, deadline.remaining())))
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        stop_reason = stop_reason or 'runner_error'
        try:
            terminate_observed_native_call(stop_reason)
        except BaseException as signal_error:
            report['signal_error'] = f'{type(signal_error).__name__}: {signal_error}'
    finally:
        for process in (broadcast, logcat):
            try:
                stop_child(process)
            except BaseException as cleanup_error:
                report.setdefault('cleanup_errors', []).append(str(cleanup_error))
        if log_thread is not None:
            log_thread.join(timeout=2)
            if log_thread.is_alive():
                report.setdefault('cleanup_errors', []).append('logcat reader did not exit')
        if pidfd is not None:
            report['observed_process_exited_at_finish'] = process_exited(pidfd)
            os.close(pidfd)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        report.update(finished_utc=utc_now(), elapsed_ms=round((time.monotonic() - started) * 1000),
                      elapsed_including_suspend_ms=round((time.clock_gettime(time.CLOCK_BOOTTIME)
                                                          - deadline.started_boottime) * 1000),
                      stop_reason=stop_reason, correlated_event_count=event_count,
                      native_call_active_at_finish=native_active,
                      broadcast_exit_code=None if broadcast is None else broadcast.returncode)
        report['success'] = (stop_reason == 'complete' and report['complete_event']['success']
                             and 'error' not in report)
        files['.json'].write(json.dumps(report, indent=2) + '\n')
        for stream in files.values():
            stream.close()
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ('resource_snapshots', 'complete_event')}, indent=2))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    sys.exit(main())
