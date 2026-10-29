"""Regression checks for the opt-in compiler harness; no real benchmarks."""
import errno
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'benchmarks' / 'compile_affinity.py'
spec = importlib.util.spec_from_file_location('compile_affinity', SCRIPT)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class CompileAffinityTests(unittest.TestCase):
    def test_summary_excludes_warmups_failures_and_observed_context_changes(self):
        rows = [dict(condition='cpu7', seconds=seconds, warmup=warmup,
                     success=success, context_changed=changed)
                for seconds, warmup, success, changed in (
                    (0.001, True, True, False), (0.002, False, False, False),
                    (0.003, False, True, True), (2.0, False, True, False),
                    (4.0, False, True, False), (3.0, False, True, False))]
        result = bench.summarize(rows)['cpu7']
        self.assertEqual(result['median_seconds'], 3.0)
        self.assertEqual(result['min_seconds'], 2.0)
        self.assertEqual(result['max_seconds'], 4.0)
        self.assertEqual(result['usable_runs'], 3)
        self.assertEqual(result['measured_runs'], 5)
        self.assertEqual(result['failures'], 1)
        self.assertEqual(result['changed_context_runs'], 1)
        self.assertIsNone(bench.summarize(rows)['default']['median_seconds'])

    def test_schedule_has_one_warmup_each_and_all_six_orders(self):
        rows = list(bench.schedule(6))
        self.assertEqual(len(rows), 21)
        self.assertEqual({row[2] for row in rows if row[1]}, set(bench.CONDITIONS))
        orders = [tuple(row[2] for row in rows if row[0] == index) for index in range(1, 7)]
        self.assertEqual(len(set(orders)), 6)
        for position in range(3):
            self.assertEqual([order[position] for order in orders].count('cpu7'), 2)

    def test_reference_requires_same_readable_cpuset(self):
        self.assertTrue(bench.reference_matches({'cpuset': '/top-app'}))
        self.assertTrue(bench.reference_matches({'cpuset': '/top-app', 'cgroup': 'group', 'reference': {'cpuset': '/top-app', 'cgroup': 'group'}}))
        self.assertFalse(bench.reference_matches({'cpuset': '/top-app', 'reference': {'cpuset': '/foreground'}}))
        self.assertFalse(bench.reference_matches({'cpuset': '/top-app', 'reference': {'cpuset': {'unavailable_errno': errno.ESRCH}}}))
        self.assertFalse(bench.reference_matches({'cpuset': '/top-app', 'cgroup': 'root', 'reference': {'cpuset': '/top-app', 'cgroup': 'app'}}))

    def run_child(self, command, cpu=None, timeout=2):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'fixture.elf'
            return bench.compile_once([*command, str(output)], cpu, output, timeout)

    def test_default_child_does_not_change_inherited_or_parent_affinity(self):
        original = os.sched_getaffinity(0)
        result = self.run_child([sys.executable, '-c',
                                'import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b"\\x7fELF" + b"x"*60)'])
        self.assertTrue(result['success'])
        self.assertEqual(result['launch']['affinity'], sorted(original))
        self.assertEqual(os.sched_getaffinity(0), original)
        self.assertGreater(result['seconds'], 0)

    def test_pinned_child_leaves_parent_affinity_unchanged(self):
        original = os.sched_getaffinity(0)
        cpu = min(original)
        result = self.run_child([sys.executable, '-c',
                                'import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b"\\x7fELF" + b"x"*60)'], cpu)
        self.assertTrue(result['success'])
        self.assertEqual(result['launch']['affinity'], [cpu])
        self.assertEqual(os.sched_getaffinity(0), original)

    def test_affinity_error_is_distinct_from_exec_error(self):
        original = os.sched_getaffinity(0)
        with patch.object(bench.os, 'sched_setaffinity', side_effect=OSError(errno.EINVAL, 'fixture denial')):
            result = self.run_child(['/does/not/exist'], 7)
        self.assertEqual(result['error']['phase'], 'affinity')
        self.assertEqual(result['error']['errno'], errno.EINVAL)
        self.assertIsNone(result['seconds'])
        self.assertFalse(result['success'])
        self.assertEqual(os.sched_getaffinity(0), original)
        result = self.run_child(['/does/not/exist'])
        self.assertEqual(result['error']['phase'], 'exec')
        self.assertEqual(result['error']['errno'], errno.ENOENT)
        self.assertFalse(result['success'])

    def test_zero_exit_without_elf_is_not_a_success(self):
        result = self.run_child([sys.executable, '-c', 'pass'])
        self.assertEqual(result['returncode'], 0)
        self.assertFalse(result['elf_output'])
        self.assertFalse(result['success'])

    def test_timeout_kills_only_child_group_and_records_failure(self):
        original = os.sched_getaffinity(0)
        result = self.run_child([sys.executable, '-c', 'import time; time.sleep(20)'], timeout=0.1)
        self.assertTrue(result['timed_out'])
        self.assertFalse(result['success'])
        self.assertEqual(result['returncode'], -9)
        self.assertEqual(os.sched_getaffinity(0), original)

    def test_root_is_rejected_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(bench.os, 'getuid', return_value=0):
            output = Path(directory) / 'report'
            with self.assertRaises(SystemExit) as failure:
                bench.main(['--source', __file__, '--compiler', sys.executable,
                            '--flag=-O3', '--output', str(output)])
            self.assertEqual(failure.exception.code, 2)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
