"""Fake proc/priority tests: this suite never changes a live process priority."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import optimizer_tuning as tuning


ACCOUNT_GUARD = tuning.ProcessTuner._account


class ProcessTunerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ostadix-tuner-test-')
        self.root = Path(self.temp.name)
        self.root.chmod(0o700)
        self.exe = self.root / 'litert_npu_resident'
        self.exe.write_bytes(b'not an executable; fake proc fixture')
        self.journal = self.root / 'journal.json'
        info = self.exe.stat()
        self.proc = {'uid': os.getuid(), 'start_ticks': 1234, 'exe': str(self.exe),
                     'exe_dev': info.st_dev, 'exe_inode': info.st_ino}
        self.threads = {101: {'start_ticks': 1234, 'uid': os.getuid(), 'nice': 0},
                        102: {'start_ticks': 1235, 'uid': os.getuid(), 'nice': 0}}
        self.writes = []
        self.fail_tid = None
        self.persisted_before_write = []
        self.patches = [
            patch.object(tuning.ProcessTuner, '_account', return_value=os.getuid()),
            patch.object(tuning, '_process_identity', side_effect=lambda pid: dict(self.proc)),
            patch.object(tuning, '_thread_ids', side_effect=lambda pid: sorted(self.threads)),
            patch.object(tuning, '_thread_identity', side_effect=self.thread_identity),
            patch.object(tuning, '_priority', side_effect=lambda tid: self.threads[tid]['nice']),
            patch.object(tuning, '_set_priority', side_effect=self.set_priority),
        ]
        for mocked in self.patches:
            mocked.start()
        self.tuner = tuning.ProcessTuner(self.journal, self.exe)

    def tearDown(self):
        for mocked in reversed(self.patches):
            mocked.stop()
        self.temp.cleanup()

    def thread_identity(self, pid, tid):
        if tid not in self.threads:
            raise FileNotFoundError('thread vanished')
        return {key: self.threads[tid][key] for key in ('uid', 'start_ticks')}

    def set_priority(self, tid, value):
        record = json.loads(self.journal.read_text())
        entry = next(item for item in record['threads'] if item['tid'] == tid)
        self.persisted_before_write.append((tid, value, entry['pending']))
        if tid == self.fail_tid:
            raise PermissionError('fixture write refused')
        self.threads[tid]['nice'] = value
        self.writes.append((tid, value))

    def test_apply_and_restore_only_verified_threads_with_durable_intents(self):
        out = self.tuner.apply(101, 1234, 5)
        self.assertTrue(out['ok'])
        self.assertEqual(out['applied'], 2)
        self.assertEqual([t['nice'] for t in self.threads.values()], [5, 5])
        self.assertTrue(all(value == pending for _, value, pending in self.persisted_before_write))
        self.assertEqual(self.journal.stat().st_mode & 0o777, 0o600)
        out = self.tuner.apply(101, 1234, 0)
        self.assertTrue(out['ok'])
        self.assertEqual(out['restored'], 2)
        self.assertEqual([t['nice'] for t in self.threads.values()], [0, 0])
        self.assertFalse(self.journal.exists())

    def test_new_instance_does_not_replay_but_explicit_restore_recovers(self):
        self.tuner.apply(101, 1234, 10)
        self.writes.clear()
        resumed = tuning.ProcessTuner(self.journal, self.exe)
        self.assertEqual(self.writes, [])
        self.assertTrue(resumed.status()['active'])
        self.assertTrue(resumed.restore()['ok'])
        self.assertEqual([t['nice'] for t in self.threads.values()], [0, 0])

    def test_crash_between_write_and_commit_is_recoverable(self):
        self.tuner.apply(101, 1234, 5)
        record = json.loads(self.journal.read_text())
        record['threads'][0]['pending'] = 10
        self.threads[101]['nice'] = 10
        self.journal.write_text(json.dumps(record))
        resumed = tuning.ProcessTuner(self.journal, self.exe)
        self.assertTrue(resumed.restore()['ok'])
        self.assertEqual([t['nice'] for t in self.threads.values()], [0, 0])

    def test_partial_failure_restores_successful_writes(self):
        self.fail_tid = 102
        out = self.tuner.apply(101, 1234, 5)
        self.assertFalse(out['ok'])
        self.assertTrue(out['rollback']['ok'])
        self.assertEqual(self.writes, [(101, 5), (101, 0)])
        self.assertEqual([t['nice'] for t in self.threads.values()], [0, 0])
        self.assertFalse(self.journal.exists())

    def test_external_changes_are_never_overwritten(self):
        self.tuner.apply(101, 1234, 5)
        self.threads[102]['nice'] = 7
        self.writes.clear()
        out = self.tuner.apply(101, 1234, 10)
        self.assertTrue(out['ok'])
        self.assertEqual(self.threads[102]['nice'], 7)
        self.assertTrue(any(item['tid'] == 102 for item in out['skipped']))
        self.tuner.restore()
        self.assertEqual(self.threads[102]['nice'], 7)
        self.assertNotIn(102, [tid for tid, _ in self.writes])

    def test_reused_pid_or_changed_executable_receives_no_writes(self):
        self.tuner.apply(101, 1234, 5)
        self.writes.clear()
        self.proc['start_ticks'] = 9999
        out = self.tuner.restore()
        self.assertEqual(self.writes, [])
        self.assertFalse(out['active'])
        self.proc['start_ticks'] = 1234
        self.proc['exe'] = '/another/executable'
        self.assertFalse(self.tuner.apply(101, 1234, 5)['ok'])
        self.assertEqual(self.writes, [])

    def test_reused_tid_is_skipped_on_restore(self):
        self.tuner.apply(101, 1234, 5)
        self.threads[102]['start_ticks'] = 9999
        self.writes.clear()
        out = self.tuner.restore()
        self.assertTrue(any(item.get('tid') == 102 for item in out['skipped']))
        self.assertEqual(self.writes, [(101, 0)])
        self.assertEqual(self.threads[102]['nice'], 5)

    def test_new_inherited_thread_restored_but_other_priority_left_alone(self):
        self.tuner.apply(101, 1234, 5)
        self.threads[103] = {'start_ticks': 1250, 'uid': os.getuid(), 'nice': 5}
        self.threads[104] = {'start_ticks': 1251, 'uid': os.getuid(), 'nice': 8}
        out = self.tuner.restore()
        self.assertEqual(self.threads[103]['nice'], 0)
        self.assertEqual(self.threads[104]['nice'], 8)
        self.assertTrue(any(item.get('tid') == 104 for item in out['skipped']))

    def test_different_initial_priorities_refuse_all_changes(self):
        self.threads[102]['nice'] = 3
        out = self.tuner.apply(101, 1234, 5)
        self.assertFalse(out['ok'])
        self.assertEqual(self.writes, [])
        self.assertFalse(self.journal.exists())

    def test_offset_is_capped_at_19_and_baseline_not_zero_is_preserved(self):
        for thread in self.threads.values():
            thread['nice'] = 17
        self.tuner.apply(101, 1234, 10)
        self.assertEqual([t['nice'] for t in self.threads.values()], [19, 19])
        self.tuner.restore()
        self.assertEqual([t['nice'] for t in self.threads.values()], [17, 17])

    def test_restore_permission_failure_keeps_journal_for_later_recovery(self):
        self.tuner.apply(101, 1234, 5)
        self.fail_tid = 102
        out = self.tuner.restore()
        self.assertFalse(out['ok'])
        self.assertTrue(out['active'])
        self.assertTrue(self.journal.exists())
        self.fail_tid = None
        resumed = tuning.ProcessTuner(self.journal, self.exe)
        self.assertTrue(resumed.restore()['ok'])
        self.assertEqual(self.threads[102]['nice'], 0)

    def test_failed_readback_keeps_restore_intent_for_recovery(self):
        self.tuner.apply(101, 1234, 5)
        with patch.object(tuning, '_set_priority', return_value=None):
            result = self.tuner.restore()
        self.assertFalse(result['ok'])
        self.assertTrue(result['active'])
        recorded = json.loads(self.journal.read_text())
        self.assertTrue(all(item['pending'] == 0 for item in recorded['threads']))
        recovered = tuning.ProcessTuner(self.journal, self.exe)
        self.assertTrue(recovered.restore()['ok'])

    def test_temporarily_unreadable_identity_retains_recovery_journal(self):
        self.tuner.apply(101, 1234, 5)
        with patch.object(tuning, '_process_identity', side_effect=PermissionError('temporarily unavailable')):
            result = self.tuner.restore()
        self.assertFalse(result['ok'])
        self.assertTrue(result['active'])
        self.assertTrue(self.journal.exists())
        self.assertTrue(self.tuner.restore()['ok'])

    def test_root_and_invalid_delta_are_refused(self):
        with patch.object(tuning.os, 'getuid', return_value=0), patch.object(tuning.os, 'geteuid', return_value=0):
            with self.assertRaises(tuning.TuningError):
                ACCOUNT_GUARD(self.tuner)
        for delta in (-1, 1, 20, True):
            with self.assertRaises(tuning.TuningError):
                self.tuner.apply(101, 1234, delta)
        self.assertEqual(self.writes, [])

    def test_proc_identity_parser_handles_command_spaces_and_truncation(self):
        proc = self.root / 'fake-proc'
        proc.mkdir()
        fields = ['S'] + ['0'] * 18 + ['9876']
        (proc / 'stat').write_text('101 (host with ) spaces) ' + ' '.join(fields))
        uid = str(os.getuid())
        (proc / 'status').write_text('Name:\thost\nUid:\t' + '\t'.join([uid] * 4) + '\n')
        self.assertEqual(tuning._identity(proc), {'uid': os.getuid(), 'start_ticks': 9876})
        (proc / 'stat').write_text('truncated')
        with self.assertRaises(tuning.TuningError):
            tuning._identity(proc)

    def test_insecure_or_oversized_journal_is_rejected(self):
        self.journal.write_text('{}')
        self.journal.chmod(0o644)
        with self.assertRaises(tuning.TuningError):
            tuning.ProcessTuner(self.journal, self.exe)
        self.journal.chmod(0o600)
        self.journal.write_bytes(b'x' * (tuning.MAX_JOURNAL + 1))
        with self.assertRaises(tuning.TuningError):
            tuning.ProcessTuner(self.journal, self.exe)
        self.journal.unlink()
        self.journal.symlink_to(self.exe)
        with self.assertRaises(tuning.TuningError):
            tuning.ProcessTuner(self.journal, self.exe)
        self.journal.unlink()
        os.mkfifo(self.journal, 0o600)
        with self.assertRaises(tuning.TuningError):
            tuning.ProcessTuner(self.journal, self.exe)


if __name__ == '__main__':
    unittest.main()
