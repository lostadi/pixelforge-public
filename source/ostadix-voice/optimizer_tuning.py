"""Reversible nice adjustments for one verified Ostadix NPU host, never root.

Each write is journaled before setpriority and checked afterwards. Recovery
requires an explicit restore() call. PID/TID start times, UID, executable path
and inode are rechecked; vanished or externally changed threads are skipped.
New threads whose priority equals the current managed priority are treated as
having inherited it from an owned thread and restored to the common baseline.
Other new-thread priorities are left alone. Numeric setpriority has no atomic
PID-identity parameter, so identity is checked immediately before and after it.
No process is killed and no affinity, frequency, voltage or governor is changed.
"""
import json
import os
from pathlib import Path
import stat
import threading
import uuid


ROOT = Path(__file__).resolve().parent
MAX_THREADS = 512
MAX_JOURNAL = 262144


class TuningError(RuntimeError):
    pass


class IdentityChanged(TuningError):
    pass


def _identity(base):
    base = Path(base)
    try:
        fields = (base / 'stat').read_text().rsplit(') ', 1)[1].split()
        start = int(fields[19])
        rows = (base / 'status').read_text().splitlines()
        uid = next(row.split()[1:] for row in rows if row.startswith('Uid:'))
    except (IndexError, ValueError, StopIteration) as error:
        raise TuningError('Process identity data is incomplete') from error
    if len(uid) != 4 or len(set(uid)) != 1:
        raise TuningError('Process credentials are inconsistent')
    return {'start_ticks': start, 'uid': int(uid[0])}


def _process_identity(pid):
    base = Path('/proc') / str(pid)
    result = _identity(base)
    result['exe'] = os.readlink(base / 'exe')
    info = (base / 'exe').stat()
    result.update(exe_dev=info.st_dev, exe_inode=info.st_ino)
    return result


def _thread_identity(pid, tid):
    return _identity(Path('/proc') / str(pid) / 'task' / str(tid))


def _thread_ids(pid):
    return sorted(int(path.name) for path in (Path('/proc') / str(pid) / 'task').iterdir()
                  if path.name.isdecimal())


def _priority(tid):
    return os.getpriority(os.PRIO_PROCESS, tid)


def _set_priority(tid, value):
    os.setpriority(os.PRIO_PROCESS, tid, value)


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


class ProcessTuner:
    def __init__(self, journal_path, engine_path=ROOT / 'runtime/npu/litert_npu_resident'):
        self.path = Path(journal_path)
        self.engine_path = Path(engine_path).resolve()
        self.lock = threading.RLock()
        self.record = self._load()
        self.last = None

    def _account(self):
        uid = os.getuid()
        if uid == 0 or os.geteuid() != uid:
            raise TuningError('Priority tuning must run as the ordinary Termux account')
        return uid

    def _load(self):
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise TuningError('Unable to open private tuning journal') from error
        with os.fdopen(fd, 'rb') as source:
            info = os.fstat(source.fileno())
            if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
                raise TuningError('Tuning journal must be a private file owned by this account')
            data = source.read(MAX_JOURNAL + 1)
        if len(data) > MAX_JOURNAL:
            raise TuningError('Tuning journal exceeds its size limit')
        try:
            record = json.loads(data)
            self._validate_record(record)
        except (ValueError, TypeError, KeyError) as error:
            raise TuningError('Tuning journal is invalid; no priority changes were made') from error
        return record

    def _validate_record(self, record):
        fields = {'version', 'uid', 'engine_path', 'exe_dev', 'exe_inode', 'pid',
                  'start_ticks', 'baseline', 'delta', 'threads'}
        if not isinstance(record, dict) or set(record) != fields or type(record['version']) is not int or record['version'] != 1:
            raise ValueError('Unexpected journal fields')
        if record['uid'] != os.getuid() or record['engine_path'] != str(self.engine_path):
            raise ValueError('Journal account or executable changed')
        for key in ('uid', 'exe_dev', 'exe_inode', 'pid', 'start_ticks'):
            if not _integer(record[key], 0, 2 ** 64 - 1):
                raise ValueError('Invalid identity value')
        if not _integer(record['baseline'], -20, 19) or record['delta'] not in (0, 5, 10) or type(record['delta']) is not int:
            raise ValueError('Invalid priority')
        threads = record['threads']
        if not isinstance(threads, list) or not 1 <= len(threads) <= MAX_THREADS:
            raise ValueError('Invalid thread list')
        seen = set()
        allowed_priorities = {min(19, record['baseline'] + delta) for delta in (0, 5, 10)}
        for entry in threads:
            if not isinstance(entry, dict) or set(entry) != {'tid', 'start_ticks', 'baseline', 'managed', 'pending', 'external', 'inherited'}:
                raise ValueError('Invalid thread entry')
            if not _integer(entry['tid'], 1, 2 ** 31 - 1) or not _integer(entry['start_ticks'], 1, 2 ** 64 - 1):
                raise ValueError('Invalid thread identity')
            if entry['tid'] in seen:
                raise ValueError('Duplicate thread')
            seen.add(entry['tid'])
            if (entry['baseline'] != record['baseline'] or not _integer(entry['managed'], -20, 19)
                    or entry['managed'] not in allowed_priorities):
                raise ValueError('Invalid thread baseline')
            if entry['pending'] is not None and (not _integer(entry['pending'], -20, 19)
                                                  or entry['pending'] not in allowed_priorities):
                raise ValueError('Invalid pending priority')
            if type(entry['external']) is not bool or type(entry['inherited']) is not bool:
                raise ValueError('Invalid thread flags')

    def _save(self):
        uid = self._account()
        parent = self.path.parent
        if parent.is_symlink():
            raise TuningError('Tuning journal directory must not be a symlink')
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = parent.stat()
        if info.st_uid != uid or info.st_mode & 0o077:
            raise TuningError('Tuning journal directory must be private and owned by this account')
        if self.path.is_symlink():
            raise TuningError('Tuning journal must not be a symlink')
        if self.record is None:
            self.path.unlink(missing_ok=True)
            return
        self._validate_record(self.record)
        data = (json.dumps(self.record, separators=(',', ':')) + '\n').encode()
        if len(data) > MAX_JOURNAL:
            raise TuningError('Tuning journal exceeds its size limit')
        temporary = parent / ('.' + self.path.name + '.' + uuid.uuid4().hex)
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'wb') as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.path)
            fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _process(self, pid, start_ticks, record=None):
        uid = self._account()
        actual = _process_identity(pid)
        expected = self.engine_path.stat()
        if (actual['uid'] != uid or actual['start_ticks'] != start_ticks
                or actual['exe'] != str(self.engine_path)
                or (actual['exe_dev'], actual['exe_inode']) != (expected.st_dev, expected.st_ino)):
            raise IdentityChanged('NPU engine identity does not match the requested process')
        if record is not None and (actual['exe_dev'], actual['exe_inode']) != (record['exe_dev'], record['exe_inode']):
            raise IdentityChanged('NPU executable changed since tuning was recorded')
        return actual

    def _thread(self, entry):
        record = self.record
        self._process(record['pid'], record['start_ticks'], record)
        actual = _thread_identity(record['pid'], entry['tid'])
        if actual != {'uid': record['uid'], 'start_ticks': entry['start_ticks']}:
            raise IdentityChanged('Thread identity changed')
        return _priority(entry['tid'])

    def _include_new_threads(self, skipped):
        record = self.record
        known = {entry['tid'] for entry in record['threads']}
        ids = _thread_ids(record['pid'])
        if len(ids) > MAX_THREADS:
            raise TuningError('Too many NPU engine threads')
        inherited_values = {min(19, record['baseline'] + record['delta'])}
        # A process may crash between writes, before the aggregate delta is
        # committed. Per-thread confirmed/pending values cover that recovery.
        for entry in record['threads']:
            if not entry['external']:
                inherited_values.add(entry['managed'])
                if entry['pending'] is not None:
                    inherited_values.add(entry['pending'])
        added = False
        for tid in ids:
            if tid in known:
                continue
            info = _thread_identity(record['pid'], tid)
            current = _priority(tid)
            if info['uid'] != record['uid'] or current not in inherited_values:
                skipped.append({'tid': tid, 'reason': 'New thread priority does not match managed inheritance'})
                continue
            record['threads'].append({'tid': tid, 'start_ticks': info['start_ticks'],
                'baseline': record['baseline'], 'managed': current, 'pending': None,
                'external': False, 'inherited': True})
            added = True
        if added:
            self._save()

    def _write(self, entry, value):
        current = self._thread(entry)
        allowed = (entry['managed'], entry['pending'])
        if entry['external'] or current not in allowed:
            entry['external'] = True
            entry['pending'] = None
            self._save()
            return False, 'Priority was changed externally'
        if current == value:
            entry['managed'], entry['pending'] = value, None
            self._save()
            return False, None
        entry['pending'] = value
        self._save()  # Recoverable intent is durable before the numeric write.
        if self._thread(entry) != current:
            entry['external'], entry['pending'] = True, None
            self._save()
            return False, 'Priority changed before the write'
        _set_priority(entry['tid'], value)
        if self._thread(entry) != value:
            raise TuningError('Thread priority readback did not match the requested value')
        entry['managed'], entry['pending'] = value, None
        self._save()
        return True, None

    def apply(self, pid, start_ticks, delta):
        """Apply a bounded offset, or restore with delta zero; never tune root."""
        self._account()
        if not _integer(pid, 1, 2 ** 31 - 1) or not _integer(start_ticks, 1, 2 ** 64 - 1) or type(delta) is not int or delta not in (0, 5, 10):
            raise TuningError('Invalid engine identity or priority delta')
        with self.lock:
            if self.record and (self.record['pid'], self.record['start_ticks']) != (pid, start_ticks):
                raise TuningError('Restore the previous tuning before selecting another engine')
            if delta == 0:
                return self.restore()
            skipped, changed = [], 0
            try:
                process = self._process(pid, start_ticks, self.record)
                if self.record is None:
                    ids = _thread_ids(pid)
                    if not 1 <= len(ids) <= MAX_THREADS:
                        raise TuningError('Invalid NPU engine thread count')
                    entries = []
                    for tid in ids:
                        identity = _thread_identity(pid, tid)
                        if identity['uid'] != os.getuid():
                            raise TuningError('NPU thread belongs to another account')
                        current = _priority(tid)
                        entries.append({'tid': tid, 'start_ticks': identity['start_ticks'],
                            'baseline': current, 'managed': current, 'pending': None,
                            'external': False, 'inherited': False})
                    baselines = {entry['baseline'] for entry in entries}
                    if len(baselines) != 1:
                        raise TuningError('Initial NPU threads must have one common baseline priority')
                    self.record = {'version': 1, 'uid': os.getuid(), 'engine_path': str(self.engine_path),
                        'exe_dev': process['exe_dev'], 'exe_inode': process['exe_inode'],
                        'pid': pid, 'start_ticks': start_ticks, 'baseline': baselines.pop(),
                        'delta': 0, 'threads': entries}
                    self._save()
                else:
                    self._include_new_threads(skipped)
                target = min(19, self.record['baseline'] + delta)
                for entry in self.record['threads']:
                    wrote, reason = self._write(entry, target)
                    changed += int(wrote)
                    if reason:
                        skipped.append({'tid': entry['tid'], 'reason': reason})
                self.record['delta'] = delta
                self._save()
                self.last = {'ok': True, 'applied': changed, 'restored': 0,
                             'skipped': skipped, 'delta': delta, 'active': True}
            except (OSError, ValueError, TuningError) as error:
                rollback = self._restore_locked()
                self.last = {'ok': False, 'applied': changed, 'skipped': skipped,
                             'error': str(error), 'rollback': rollback,
                             'active': self.record is not None}
            return dict(self.last)

    def _restore_locked(self):
        skipped, restored, errors = [], 0, []
        if self.record is None:
            return {'ok': True, 'restored': 0, 'skipped': [], 'active': False, 'delta': 0}
        try:
            self._process(self.record['pid'], self.record['start_ticks'], self.record)
        except (FileNotFoundError, ProcessLookupError, IdentityChanged) as error:
            # A stale identity is never a license to modify a replacement PID.
            self.record = None
            try:
                self._save()
            except (OSError, TuningError) as cleanup:
                return {'ok': False, 'restored': 0, 'skipped': [], 'error': str(cleanup), 'active': False}
            return {'ok': True, 'restored': 0, 'skipped': [{'reason': str(error)}], 'active': False, 'delta': 0}
        except (OSError, ValueError, TuningError) as error:
            return {'ok': False, 'restored': 0, 'skipped': [], 'error': str(error), 'active': True}
        try:
            self._include_new_threads(skipped)
        except (OSError, ValueError, TuningError) as error:
            errors.append(str(error))
        pending = []
        for entry in list(self.record['threads']):
            try:
                wrote, reason = self._write(entry, entry['baseline'])
                restored += int(wrote)
                if reason:
                    skipped.append({'tid': entry['tid'], 'reason': reason})
            except (FileNotFoundError, ProcessLookupError, IdentityChanged) as error:
                skipped.append({'tid': entry['tid'], 'reason': str(error)})
            except (OSError, ValueError, TuningError) as error:
                errors.append(str(error))
                pending.append(entry)
        self.record['threads'] = pending
        if not pending:
            self.record = None
        try:
            self._save()
        except (OSError, ValueError, TuningError) as error:
            errors.append(str(error))
        return {'ok': not errors, 'restored': restored, 'skipped': skipped,
                'errors': errors, 'active': self.record is not None, 'delta': 0 if self.record is None else self.record['delta']}

    def restore(self):
        self._account()
        with self.lock:
            self.last = self._restore_locked()
            return dict(self.last)

    def status(self):
        """Report journaled intent without altering processes or replaying writes."""
        with self.lock:
            record = self.record
            return {'active': record is not None, 'pid': record['pid'] if record else None,
                    'start_ticks': record['start_ticks'] if record else None,
                    'delta': record['delta'] if record else 0,
                    'baseline': record['baseline'] if record else None,
                    'tracked_threads': len(record['threads']) if record else 0,
                    'last_result': self.last}
