"""User-enabled continuous local NPU monitoring and reversible host scheduling.

No cloud calls or free-form model commands. Global telemetry is supplied by a
fixed probe; only the verified Ostadix NPU host's nice values can be adjusted.
"""
import argparse
from contextlib import closing, contextmanager
import fcntl
import json
import math
import re
import os
from pathlib import Path
import shlex
import signal
import socket
import sqlite3
import struct
import subprocess
import sys
import threading
import time
import uuid

# The root guardian drops to the app identity and invokes this script with -I.
# Add this fixed application directory only after that privilege drop.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from local_runtime import DEFAULT_STATE, ROOT, LocalClient, LocalError, _environment, _private_file, _state
from optimizer_policy import OBJECTIVES, decide, new_state, validate_recommendation

from optimizer_reuse import evidence_key

OWNER = 'ostadix-system-optimizer-v1'
SAMPLE_SECONDS = 10
ANALYSIS_SECONDS = 60
RECOMMENDATION_TTL = 120
GUARDIAN = Path('/data/adb/ostadix-optimizer/guardian.py')


def worker_client(base, local_factory=None):
    """Keep production NPU analysis separate from the default Huihui CPU pool."""
    if Path(base).expanduser().resolve() == DEFAULT_STATE.expanduser().resolve():
        from npu_tools import NPUClient
        return NPUClient()
    return (local_factory or LocalClient)(base)


def worker_state(client, fallback):
    state = getattr(client, 'state', None)
    return Path(state) if isinstance(state, (str, os.PathLike)) else Path(fallback)


def read_json(path):
    try:
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'r') as stream:
            text = stream.read(262145)
        if len(text) > 262144:
            raise LocalError('Optimizer state exceeds its size limit')
        value = json.loads(text)
        if not isinstance(value, dict):
            raise LocalError('Invalid optimizer state')
        return value
    except FileNotFoundError:
        return {}


def write_json(path, value):
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex)
    try:
        with os.fdopen(_private_file(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL), 'w') as stream:
            json.dump(value, stream, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def process_stamp(pid):
    if type(pid) is not int or pid < 2:
        return None
    try:
        directory = Path('/proc') / str(pid)
        if directory.stat().st_uid != os.getuid():
            return None
        fields = (directory / 'stat').read_text().rsplit(') ', 1)[1].split()
        return int(fields[19]) if fields[0] != 'Z' else None
    except (OSError, ValueError, IndexError):
        return None


def config_at(state):
    value = read_json(state / 'config.json')
    if not value:
        return {'version': 1, 'enabled': False, 'objective': 'balanced'}
    if (set(value) != {'version', 'enabled', 'objective'} or type(value['version']) is not int or value['version'] != 1
            or type(value['enabled']) is not bool or value['objective'] not in OBJECTIVES):
        raise LocalError('Invalid optimizer configuration; no automatic activation')
    return value


def control_request(state, operation='status', expected=None):
    """Authenticate the live daemon over its private socket, not proc visibility."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(.5)
        connection.connect(str(state / 'control.sock'))
        peer_pid, peer_uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
        if peer_uid != os.getuid():
            raise LocalError('Optimizer control peer belongs to another account')
        request = {'operation': operation}
        if operation == 'stop':
            if not expected or expected.get('pid') != peer_pid:
                raise LocalError('Optimizer instance changed before stop')
            request.update(pid=peer_pid, process_stamp=expected.get('process_stamp'))
        connection.sendall((json.dumps(request) + '\n').encode())
        value = bytearray()
        while b'\n' not in value:
            chunk = connection.recv(4096)
            if not chunk:
                raise LocalError('Optimizer control closed before responding')
            value.extend(chunk)
            if len(value) > 4096:
                raise LocalError('Optimizer control reply is oversized')
        result = json.loads(value.partition(b'\n')[0])
        if not isinstance(result, dict) or result.get('pid') != peer_pid or result.get('uid') != peer_uid:
            raise LocalError('Optimizer control identity mismatch')
        if expected and result.get('process_stamp') != expected.get('process_stamp'):
            raise LocalError('Optimizer instance changed during control request')
        return result


@contextmanager
def control_lock(state):
    with os.fdopen(_private_file(state / 'control.lock', os.O_WRONLY | os.O_CREAT), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def root_call(arguments, timeout=8):
    """Only callers with fixed argv use this; model output is never an argument."""
    env = _environment()
    env.pop('LD_PRELOAD', None)
    result = subprocess.run(['/system/bin/su', '-c', shlex.join(arguments)],
                            stdin=subprocess.DEVNULL, capture_output=True, text=True,
                            timeout=timeout, env=env)
    if result.returncode:
        raise LocalError('Optimizer fixed root helper failed: ' + result.stderr.strip()[:300])
    if len(result.stdout) > 262144:
        raise LocalError('Optimizer helper response exceeds bound')
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise LocalError('Invalid optimizer helper response')
    return value


def probe(release=False):
    return root_call([sys.executable, '-I', str(ROOT / 'optimizer_probe.py'),
                      '--release' if release else '--lease'])


class OptimizerControl:
    def __init__(self, state_dir=None):
        self.base = Path(state_dir) if state_dir is not None else DEFAULT_STATE
        self.state = self.base / 'optimizer'

    def status(self):
        """Read local metadata only. Never starts any service or root command."""
        try:
            config = config_at(self.state)
            metadata = read_json(self.state / 'status.json')
            pid = metadata.get('pid')
            alive = process_stamp(pid)
            running = alive is not None and alive == metadata.get('process_stamp')
            try:
                peer = control_request(self.state)
                running = (peer['pid'] == pid and peer.get('process_stamp') == metadata.get('process_stamp'))
            except (OSError, ValueError, LocalError):
                if metadata.get('phase') == 'stopped':
                    running = False
            result = dict(metadata, enabled=config['enabled'], running=running,
                          objective=config['objective'], pid=pid if running else None,
                          scope='Phone telemetry; priority of owned Ostadix NPU host threads only',
                          sample_seconds=SAMPLE_SECONDS, analysis_seconds=ANALYSIS_SECONDS,
                          model_backend='litert-npu', cloud_calls=False)
            result.setdefault('phase', 'stopped')
            if not running:
                result['phase'] = 'stopped' if not config['enabled'] else 'awaiting recovery'
            result['heartbeat_age_seconds'] = round(max(0, time.time() - metadata.get('updated', time.time())), 1)
            result['managed_effects'] = read_json(self.state / 'effects.json')
            return result
        except (OSError, ValueError, LocalError) as error:
            return {'enabled': False, 'running': False, 'phase': 'error', 'error': str(error)}

    def history(self, limit=10):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise LocalError('History limit must be 1–100')
        path = self.state / 'history.sqlite3'
        if not path.exists():
            return {'events': [], 'scope': 'optimizer only'}
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
            events = [json.loads(row[0]) for row in db.execute(
                'SELECT data FROM events ORDER BY id DESC LIMIT ?', (limit,))]
        return {'events': events, 'scope': 'optimizer only'}

    def enable(self, objective='balanced'):
        if objective not in OBJECTIVES:
            raise LocalError('Objective must be balanced, performance or battery')
        if os.getuid() == 0:
            raise LocalError('Enable the optimizer from ordinary Termux')
        state = _state(self.state)
        with control_lock(state):
            write_json(state / 'config.json', {'version': 1, 'enabled': True, 'objective': objective})
        from local_service import LocalService
        LocalService(self.base).enable()
        result = self.resume()
        # The installed private guardian is optional off-device, but failures
        # stay visible so process recovery is never claimed without evidence.
        if self.base == DEFAULT_STATE:
            try:
                result['guardian'] = root_call([sys.executable, '-I', str(GUARDIAN), '--ensure'])
            except (OSError, ValueError, LocalError, subprocess.SubprocessError) as error:
                result['guardian_error'] = str(error)
        return result

    def resume(self):
        if not config_at(self.state)['enabled']:
            return self.status()
        if os.getuid() == 0:
            raise LocalError('Run the optimizer as the ordinary Termux account')
        from local_service import _enabled
        if not _enabled(self.base):
            return dict(self.status(), phase='paused: local service disabled')
        state = _state(self.state)
        with control_lock(state):
            if not config_at(state)['enabled'] or self.status()['running']:
                return self.status()
            with os.fdopen(_private_file(state / 'daemon.log', os.O_WRONLY | os.O_CREAT | os.O_APPEND), 'ab') as log:
                subprocess.Popen([sys.executable, str(ROOT / 'optimizer.py'), '--daemon',
                                  '--state-dir', str(self.base)], stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, env=_environment(), start_new_session=True)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            result = self.status()
            if result['running']:
                return result
            time.sleep(.025)
        return dict(self.status(), starting=True)

    def stop(self):
        """Pause the process without changing the saved enabled choice."""
        result = self.status()
        if result['running']:
            # A live authenticated daemon stops itself. Some phone kernels hide
            # a su caller from other same-UID processes, including /proc checks.
            control_request(self.state, 'stop', expected=result)
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline and self.status()['running']:
                time.sleep(.05)
            if self.status()['running']:
                return dict(self.status(), error='Optimizer is still restoring its owned settings')
        # A previous abrupt death can leave a journal. Restore it even when no
        # process is alive; the tuner verifies every identity before touching it.
        if self.state.exists():
            from optimizer_tuning import ProcessTuner
            restored = ProcessTuner(self.state / 'effects.json').restore()
            output = self.status()
            output['restoration'] = restored
            if restored.get('ok') is not True:
                output['error'] = 'Optimizer priority restoration is incomplete; inspect restoration details'
            try:
                output['wake_release'] = probe(release=True)
            except (OSError, ValueError, LocalError, subprocess.SubprocessError) as error:
                output['wake_release_error'] = str(error) + '; the dedicated lease expires within 30 seconds'
            return output
        return self.status()

    def disable(self):
        if os.getuid() == 0:
            raise LocalError('Disable the optimizer from ordinary Termux')
        state = _state(self.state)
        with control_lock(state):
            config = config_at(state)
            config['enabled'] = False
            write_json(state / 'config.json', config)
        return self.stop()


def cpu_busy(previous, current):
    try:
        total = current['total'] - previous['total']
        idle = current['idle'] - previous['idle']
        if total > 0 and 0 <= idle <= total:
            return round(100 * (total - idle) / total, 2)
    except (KeyError, TypeError):
        pass
    return None


def model_prompt(telemetry, objective):
    metrics = {key: telemetry.get(key) for key in ('battery_percent', 'battery_temp_c',
               'charging', 'thermal_severity', 'memory_available_mb', 'cpu_busy_pct', 'active', 'queued')}
    guidance = {
        'balanced': 'Prefer normal when healthy. Consider eco when CPU busy is at least 85%, memory available below1536MB, battery temperature at least41C, or unplugged battery at most25%. ',
        'performance': 'Prefer normal unless battery temperature is at least42C, memory available below768MB, or unplugged battery at most15%. ',
        'battery': 'While unplugged and no user jobs are running or queued, prefer eco to give other workloads CPU priority. Otherwise consider eco for heat, low battery or memory pressure. '
    }[objective]
    return ('Assess phone resource telemetry for the ' + objective + ' objective. '
            'Choose normal (original CPU priority) or eco (lower CPU priority for the local AI host). '
            'Eco may reduce competition for CPU time; it does not directly set NPU frequency or guarantee battery savings. '
            + guidance +
            'Return ONLY one JSON object with exactly action and reason. '
            'Reason must be one short sentence. Example: {"action":"normal","reason":"Conditions are healthy."}. '
            'No shell commands. Telemetry: ' + json.dumps(metrics, separators=(',', ':')))


def parse_model_recommendation(value):
    """Normalize one whole Markdown fence, then apply the strict policy schema."""
    if isinstance(value, str) and len(value) <= 4096:
        fenced = re.fullmatch(r"```(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n```", value.strip())
        if fenced:
            value = fenced.group(1)
    return validate_recommendation(value)


class Monitor:
    def __init__(self, base, sample=probe, client=None, tuner=None):
        self.base = Path(base)
        self.state = _state(self.base / 'optimizer')
        self.client = client or worker_client(self.base)
        if tuner is None:
            from optimizer_tuning import ProcessTuner
            tuner = ProcessTuner(self.state / 'effects.json')
        self.tuner = tuner
        self.sample = sample
        self.stop_event = threading.Event()
        self.control_closed = threading.Event()
        self.control_socket = None
        self.control_thread = None
        self.policy_state = new_state()
        self.last_cpu = None
        self.engine = None
        self.pending = None
        self.next_analysis = 0
        self.recommendation = None
        self.recommendation_time = 0
        self.recommendation_key = None
        self.recommendation_engine = None
        self.objective = config_at(self.state)['objective']
        self.companion = None
        self.snapshot = {'pid': os.getpid(), 'process_stamp': process_stamp(os.getpid()),
                         'phase': 'starting', 'analysis_count': 0, 'action_count': 0,
                         'analysis_reuse_count': 0, 'reuse_version': 1,
                         'last_observation': None, 'last_decision': None, 'last_action': None,
                         'error': None, 'model_backend': 'litert-npu'}
        self.db = sqlite3.connect(self.state / 'history.sqlite3')
        self.db.execute('CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, data TEXT NOT NULL)')
        (self.state / 'history.sqlite3').chmod(0o600)

    def event(self, kind, **details):
        row = dict(kind=kind, utc_seconds=time.time(), **details)
        with self.db:
            self.db.execute('INSERT INTO events(data) VALUES(?)', (json.dumps(row, allow_nan=False),))
            self.db.execute('DELETE FROM events WHERE id NOT IN (SELECT id FROM events ORDER BY id DESC LIMIT 1024)')

    def save(self, **changes):
        self.snapshot.update(changes, updated=time.time())
        write_json(self.state / 'status.json', self.snapshot)

    def restore(self):
        result = self.tuner.restore()
        if result.get('ok') is not True:
            raise LocalError('Optimizer priority restoration is incomplete: ' + json.dumps(result)[:400])
        return result

    def start_control(self):
        path = self.state / 'control.sock'
        if path.exists() or path.is_symlink():
            # The singleton lock is held before this method. A stale socket
            # cannot be a concurrent monitor, but never replace another file.
            import stat
            if not stat.S_ISSOCK(path.lstat().st_mode):
                raise LocalError('Optimizer control path is not a socket')
            path.unlink()
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path)); path.chmod(0o600)
        listener.listen(4); listener.settimeout(.2)
        self.control_socket = listener

        def serve():
            while not self.control_closed.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    return
                with connection:
                    try:
                        connection.settimeout(.5)
                        _, uid, _ = struct.unpack('3i', connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))
                        if uid != os.getuid():
                            continue
                        encoded = bytearray()
                        while b'\n' not in encoded and len(encoded) <= 256:
                            block = connection.recv(257 - len(encoded))
                            if not block:
                                break
                            encoded.extend(block)
                        if len(encoded) > 256 or not encoded.endswith(b'\n'):
                            continue
                        request = json.loads(encoded)
                        if request == {'operation': 'stop', 'pid': os.getpid(),
                                       'process_stamp': self.snapshot['process_stamp']}:
                            self.stop_event.set()
                        elif request != {'operation': 'status'}:
                            continue
                        response = {'pid': os.getpid(), 'uid': os.getuid(),
                                    'process_stamp': self.snapshot['process_stamp'],
                                    'stopping': self.stop_event.is_set()}
                        connection.sendall((json.dumps(response) + '\n').encode())
                    except (OSError, ValueError):
                        pass
        self.control_thread = threading.Thread(target=serve, name='optimizer-control', daemon=True)
        self.control_thread.start()

    def close_control(self):
        self.control_closed.set()
        if self.control_socket is not None:
            self.control_socket.close()
        if self.control_thread is not None:
            self.control_thread.join(timeout=1)
        (self.state / 'control.sock').unlink(missing_ok=True)

    def poll_analysis(self, now, current_key=None):
        if not self.pending:
            return
        result = self.client.read(OWNER, self.pending['id'], wait=0, limit=4096)
        if result['state'] in ('queued', 'running'):
            return
        submitted = self.pending
        submitted_objective = submitted.get('objective')
        self.pending = None
        self.snapshot['analysis_count'] += 1
        if submitted_objective is not None and submitted_objective != self.objective:
            self.event('analysis_superseded', job_id=result['id'])
            return
        if (result['state'] != 'completed' or result.get('result_truncated') or result.get('truncated')
                or result.get('usage', {}).get('token_limit_reached') is True
                or result.get('finish_reason') == 'length'):
            raise LocalError('Background NPU analysis did not complete: ' + str(result.get('error') or result['state']))
        if result.get('usage', {}).get('backend') != 'litert-npu':
            raise LocalError('Background analysis did not report the required NPU backend')
        validated = parse_model_recommendation(result['result'])
        if (submitted.get('evidence_key') != current_key
                or submitted.get('engine') != self.engine or current_key is None
                or now - submitted.get('submitted_at', now) >= RECOMMENDATION_TTL):
            self.recommendation = None
            self.event('analysis_superseded', job_id=result['id'], reason='Evidence or engine changed, or result expired')
            return
        self.recommendation = validated
        self.recommendation_key = current_key
        self.recommendation_engine = self.engine
        self.recommendation_time = submitted['submitted_at']
        self.event('model_analysis', recommendation=self.recommendation, job_id=result['id'], usage=result['usage'])
        self.snapshot['last_model_analysis'] = {'recommendation': self.recommendation, 'job_id': result['id'],
                                                'usage': result['usage'], 'at': time.time()}

    def prune_own_jobs(self):
        """Bound only this explicitly recurring monitor's completed job records."""
        path = worker_state(self.client, self.base) / 'jobs.sqlite3'
        if not path.exists():
            return
        with closing(sqlite3.connect(path, timeout=.2)) as db:
            with db:
                db.execute("DELETE FROM jobs WHERE owner=? AND state IN ('completed','failed','interrupted','cancelled') "
                           'AND id NOT IN (SELECT id FROM jobs WHERE owner=? ORDER BY created DESC LIMIT 256)', (OWNER, OWNER))

    def tick(self):
        now = time.monotonic()
        config = config_at(self.state)
        from local_service import _enabled
        if not config['enabled'] or not _enabled(self.base):
            self.stop_event.set()
            return
        if self.objective != config['objective']:
            self.objective = config['objective']
            self.recommendation = None
            self.next_analysis = 0
            self.policy_state['candidate_delta'] = self.policy_state['candidate_since'] = None
        status = self.client.status()
        telemetry = self.sample()
        current = telemetry.pop('cpu_totals', None)
        telemetry['cpu_busy_pct'] = cpu_busy(self.last_cpu, current)
        self.last_cpu = current
        telemetry.update(active=status.get('active', 0), queued=status.get('queued', 0))
        if (self.stop_event.is_set() or not config_at(self.state)['enabled']
                or not _enabled(self.base)):
            self.stop_event.set()
            return
        self.save(last_observation=telemetry, error=None)
        self.event('observation', telemetry=telemetry)
        if not status.get('ready') or (status.get('model') or {}).get('backend') != 'litert-npu':
            self.restore()
            self.engine = None
            self.policy_state = new_state()
            self.save(phase='waiting for NPU model')
            return
        pid = status.get('engine_pid')
        stamp = process_stamp(pid)
        if stamp is None:
            raise LocalError('NPU engine identity is unavailable')
        if self.engine != (pid, stamp):
            self.restore()
            self.engine = (pid, stamp)
            self.next_analysis = 0
            self.policy_state = new_state()
            self.recommendation = None
        # Sampling timestamps are captured after tick entry. Validate against
        # the current clock, not the pre-sampling timestamp.
        now = time.monotonic()
        current_key = evidence_key(telemetry, config['objective'], now)
        self.poll_analysis(now, current_key)
        reusable = (self.recommendation is not None and current_key is not None
                    and current_key == self.recommendation_key
                    and self.engine == self.recommendation_engine
                    and 0 <= now - self.recommendation_time < RECOMMENDATION_TTL)
        if not reusable:
            if self.recommendation is not None:
                self.next_analysis = min(self.next_analysis,
                                         self.recommendation_time + ANALYSIS_SECONDS)
            self.recommendation = None
        recommendation = self.recommendation if reusable else None
        decision = decide(telemetry, recommendation, self.policy_state, config['objective'], now=time.monotonic())
        if decision['action'] == 'set_nice':
            # Recheck the durable stop choice immediately before a mutation.
            if (not config_at(self.state)['enabled'] or not _enabled(self.base)
                    or self.stop_event.is_set()):
                return
            outcome = self.tuner.apply(pid, stamp, decision['nice_delta'])
            if outcome.get('ok') is not True:
                self.event('priority_action_failed', result=outcome)
                raise LocalError('Optimizer priority change failed and rollback was attempted: ' + json.dumps(outcome)[:400])
            if outcome.get('skipped'):
                restored = self.restore()
                self.event('priority_action_conflict', result=outcome, restoration=restored)
                self.policy_state = new_state()
                raise LocalError('Thread identity or priority changed externally; restored owned changes without overriding it')
            # Never advance policy state without verified successful readback.
            self.policy_state = decision['next_state']
            self.snapshot['action_count'] += 1
            self.snapshot['last_action'] = {'decision': decision, 'result': outcome, 'at': time.time(),
                                              'benefit': 'Unproven; priority readback is not a speed or battery measurement'}
            self.event('priority_action', **self.snapshot['last_action'])
        else:
            self.policy_state = decision['next_state']
        if not self.pending and now >= self.next_analysis and reusable:
            self.snapshot['analysis_reuse_count'] += 1
            self.next_analysis = min(now + ANALYSIS_SECONDS,
                                     self.recommendation_time + RECOMMENDATION_TTL)
            self.event('analysis_reused', age_seconds=now - self.recommendation_time,
                       expires_in_seconds=self.recommendation_time + RECOMMENDATION_TTL - now)
        if (not self.pending and now >= self.next_analysis and current_key is not None
                and not status.get('active') and not status.get('queued')):
            task = model_prompt(telemetry, config['objective'])
            # Sampling and policy work can outlive a stop request or a user's
            # arrival. Recheck admission immediately before starting new work.
            latest = self.client.status()
            if (self.stop_event.is_set() or not config_at(self.state)['enabled']
                    or not _enabled(self.base)):
                self.stop_event.set()
                return
            if (not latest.get('ready') or latest.get('active') or latest.get('queued')
                    or latest.get('engine_pid') != pid
                    or (latest.get('model') or {}).get('backend') != 'litert-npu'):
                return
            self.pending = self.client.submit(OWNER, task, max_tokens=96, role='resource monitor')
            self.pending.update(objective=config['objective'], evidence_key=current_key,
                                engine=self.engine, submitted_at=now)
            self.next_analysis = now + ANALYSIS_SECONDS
            self.event('analysis_submitted', job_id=self.pending['id'], engine_pid=pid)
            try:
                self.prune_own_jobs()
            except sqlite3.Error as error:
                self.event('history_cleanup_deferred', reason=type(error).__name__)
        self.save(phase='analyzing' if self.pending else 'monitoring', last_decision=decision,
                  engine_pid=pid, policy_state=self.policy_state)

    def run(self):
        if os.getuid() == 0:
            raise LocalError('The monitoring and policy process must run as ordinary Termux')
        os.umask(0o077)
        with os.fdopen(_private_file(self.state / 'daemon.lock', os.O_WRONLY | os.O_CREAT), 'w') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                self.db.close()
                return
            self.save()
            try:
                self.start_control()
                restored = self.restore()
                self.event('started', recovery_restoration=restored)
                while not self.stop_event.is_set():
                    started = time.monotonic()
                    try:
                        self.tick()
                    except Exception as error:
                        self.recommendation = None
                        self.save(phase='degraded: observing only', error=str(error)[:500])
                        self.event('error', message=str(error)[:500])
                    # Optional, separately enabled conversation check-ins share
                    # this already supervised loop and resident NPU. Their
                    # failures never prevent host-priority monitoring/cleanup.
                    if not self.stop_event.is_set() and (self.base / 'companion/state.json').exists():
                        try:
                            from companion import CompanionLoop
                            from local_service import _enabled
                            if self.companion is None:
                                self.companion = CompanionLoop(self.base, client=self.client)
                            self.companion.tick(self.snapshot.get('last_observation'),
                                stopped=lambda: self.stop_event.is_set() or not config_at(self.state)['enabled']
                                                or not _enabled(self.base))
                            if self.snapshot.get('companion_error'):
                                self.save(companion_error=None)
                        except Exception as error:
                            self.save(companion_error=str(error)[:400])
                            self.event('companion_error', message=str(error)[:400])
                    self.stop_event.wait(max(.1, SAMPLE_SECONDS - (time.monotonic() - started)))
            finally:
                errors = []
                try:
                    restored = self.restore()
                    self.event('stopped', restoration=restored,
                               pending_analysis_may_finish=self.pending['id'] if self.pending else None)
                except Exception as error:
                    errors.append('Priority restoration: ' + str(error))
                try:
                    wake = self.sample(release=True)
                    self.snapshot['wake_release'] = wake
                except Exception as error:
                    errors.append('Wake lease release: ' + str(error) + '; lease expires within 30 seconds')
                self.save(phase='stopped', error='; '.join(errors) or None)
                self.close_control()
                self.db.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    options = parser.add_mutually_exclusive_group(required=True)
    for flag in ('status', 'enable', 'disable', 'resume', 'history', 'daemon'):
        options.add_argument('--' + flag, action='store_true')
    parser.add_argument('--state-dir', type=Path, default=DEFAULT_STATE)
    parser.add_argument('--objective', choices=sorted(OBJECTIVES), default='balanced')
    args = parser.parse_args(argv)
    if args.daemon:
        monitor = Monitor(args.state_dir)
        signal.signal(signal.SIGTERM, lambda *_: monitor.stop_event.set())
        signal.signal(signal.SIGINT, lambda *_: monitor.stop_event.set())
        monitor.run()
        return
    control = OptimizerControl(args.state_dir)
    if args.resume:
        from local_service import LocalService
        # Guardian/boot recovery never changes a saved disabled choice.
        if config_at(control.state)['enabled']:
            LocalService(args.state_dir).resume()
        result = control.resume()
    elif args.enable:
        result = control.enable(args.objective)
    else:
        name = next(flag for flag in ('status', 'disable', 'history') if getattr(args, flag))
        result = getattr(control, name)()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
