import json
import os
from pathlib import Path
import sys
import tempfile
import selectors
import subprocess
import unittest
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import service
import lifecycle


class RequestTests(unittest.TestCase):
    def test_only_inference_data_crosses_the_root_boundary(self):
        value = service.validate_request({'id': 'job-1', 'prompt': 'Read a file'})
        self.assertEqual(value['max_tokens'], 512)
        self.assertTrue(value['prefill_cpu'])
        for extra in ('command', 'model_path', 'artifacts', 'native_command', 'output'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                service.validate_request({'id': 'job-1', 'prompt': 'Hi', extra: '/anything'})

    def test_bounded_frames_and_explicit_prefill(self):
        for change in ({'max_tokens': True}, {'max_tokens': 1}, {'max_tokens': 513},
                       {'context_size': 8193}, {'prefill_cpu': False}, {'id': '../../file'},
                       {'prompt': 'a\0b'}, {'prompt': 'x' * 48001}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                service.validate_request(dict(id='job-1', prompt='Hi', **change) if not {'id', 'prompt'} & change.keys()
                                         else {**{'id': 'job-1', 'prompt': 'Hi'}, **change})

    def test_model_template_renders_user_text_as_data_and_disables_thinking(self):
        render = service.renderer('{{ bos_token }}{{ messages[0].content }}|{{ enable_thinking }}|{{ add_generation_prompt }}')
        self.assertEqual(render('{{ 7*9 }}'), '<bos>{{ 7*9 }}|False|True')

    def test_template_cannot_access_python_object_internals(self):
        from jinja2.exceptions import SecurityError
        with self.assertRaises(SecurityError):
            service.renderer('{{ messages.__class__.__mro__ }}')('Hi')


@unittest.skipUnless(os.getuid() == 0, 'Root-owned broker fixtures')
class OwnerTests(unittest.TestCase):
    def test_exact_live_broker_is_adopted_for_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'owner.json'
            path.write_text(json.dumps({'pid': 123, 'start_ticks': '456'}))
            path.chmod(0o600)
            cleanup = Mock()
            cleanup.process_record.return_value = {'pid': 123, 'stamp': '456'}
            owned = {}
            service.adopt_broker(cleanup, owned, path)
            self.assertEqual(owned, {123: '456'})
            cleanup.collect_owned.assert_called_once_with(owned)

    def test_recycled_broker_pid_is_never_adopted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'owner.json'
            path.write_text(json.dumps({'pid': 123, 'start_ticks': '456'}))
            path.chmod(0o600)
            cleanup = Mock()
            cleanup.process_record.return_value = {'pid': 123, 'stamp': '999'}
            owned = {}
            service.adopt_broker(cleanup, owned, path)
            self.assertEqual(owned, {})
            cleanup.collect_owned.assert_not_called()

    def test_writable_or_symlink_broker_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'owner.json'
            path.write_text('{}')
            path.chmod(0o666)
            with self.assertRaises(ValueError):
                service.adopt_broker(Mock(), {}, path)
            link = Path(directory) / 'alias'
            link.symlink_to(path)
            with self.assertRaises(OSError):
                service.adopt_broker(Mock(), {}, link)


class ChildDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.proc = Path(self.temp.name)
        self.records = {}
        self.root_patch = patch.object(lifecycle, 'PROC_ROOT', self.proc)
        self.root_patch.start()
        self.record_patch = patch.object(lifecycle, 'process_record', side_effect=lambda pid: self.records.get(pid))
        self.record_patch.start()
    def tearDown(self):
        self.record_patch.stop()
        self.root_patch.stop()
        self.temp.cleanup()
    def process(self, pid, parent, stamp, children=(), tid=None):
        self.records[pid] = {'pid': pid, 'parent': parent, 'stamp': stamp}
        task = self.proc / str(pid) / 'task' / str(tid or pid)
        task.mkdir(parents=True, exist_ok=True)
        (task / 'children').write_text(' '.join(map(str, children)))
        return task
    def test_fast_path_reads_only_owned_tasks_and_recurses(self):
        self.process(10, 1, 'root', [11], tid=101)
        self.process(11, 10, 'child', [12])
        self.process(12, 11, 'grandchild')
        owned = {10: 'root'}
        original = Path.iterdir
        def forbid_global(path):
            if path == self.proc:
                raise AssertionError('Periodic discovery must not scan unrelated processes')
            return original(path)
        with patch.object(Path, 'iterdir', forbid_global), patch.object(lifecycle, 'collect_owned') as full:
            self.assertTrue(lifecycle.collect_owned_children(owned))
        self.assertEqual(owned, {10: 'root', 11: 'child', 12: 'grandchild'})
        full.assert_not_called()
    def test_recycled_root_or_recorded_child_is_not_adopted(self):
        self.process(10, 1, 'new-root', [11])
        self.process(11, 10, 'new-child', [12])
        self.process(12, 11, 'grandchild')
        owned = {10: 'old-root'}
        self.assertTrue(lifecycle.collect_owned_children(owned))
        self.assertEqual(owned, {10: 'old-root'})
        owned = {10: 'new-root', 11: 'old-child'}
        self.assertTrue(lifecycle.collect_owned_children(owned))
        self.assertEqual(owned, {10: 'new-root', 11: 'old-child'})
    def test_tracker_backs_off_only_when_children_interface_is_unavailable(self):
        cleanup = Mock()
        cleanup.collect_owned_children.side_effect = [False, True]
        stopped = Mock()
        stopped.wait.side_effect = [False, False, True]
        owned = {10: 'root'}
        service.track_owned(cleanup, owned, stopped)
        self.assertEqual(stopped.wait.call_args_list, [call(.2), call(2.0), call(.2)])
        self.assertEqual(cleanup.collect_owned_children.call_args_list, [call(owned), call(owned)])
        cleanup.collect_owned.assert_not_called()

    def test_child_identity_change_during_validation_is_rejected(self):
        self.process(10, 1, 'root', [11])
        self.process(11, 10, 'old-child')
        reads = [0]
        def changing_record(pid):
            if pid == 11:
                reads[0] += 1
                return {'pid': 11, 'parent': 10, 'stamp': 'old-child' if reads[0] == 1 else 'replacement'}
            return self.records.get(pid)
        owned = {10: 'root'}
        with patch.object(lifecycle, 'process_record', side_effect=changing_record):
            self.assertTrue(lifecycle.collect_owned_children(owned))
        self.assertEqual(owned, {10: 'root'})
    def test_parent_identity_change_before_adoption_is_rejected(self):
        self.process(10, 1, 'root', [11])
        self.process(11, 10, 'child')
        reads = [0]
        def changing_record(pid):
            if pid == 10:
                reads[0] += 1
                return {'pid': 10, 'parent': 1, 'stamp': 'root' if reads[0] == 1 else 'replacement'}
            return self.records.get(pid)
        owned = {10: 'root'}
        with patch.object(lifecycle, 'process_record', side_effect=changing_record):
            self.assertTrue(lifecycle.collect_owned_children(owned))
        self.assertEqual(owned, {10: 'root'})
    def test_missing_children_interface_for_live_process_falls_back(self):
        task = self.process(10, 1, 'root')
        (task / 'children').unlink()
        owned = {10: 'root'}
        with patch.object(lifecycle, 'collect_owned') as full:
            self.assertFalse(lifecycle.collect_owned_children(owned))
        full.assert_called_once_with(owned)
    def test_unreadable_children_interface_falls_back(self):
        task = self.process(10, 1, 'root')
        original = Path.open
        def denied(path, *args, **kwargs):
            if path == task / 'children':
                raise PermissionError('Synthetic denied children interface')
            return original(path, *args, **kwargs)
        owned = {10: 'root'}
        with patch.object(Path, 'open', denied), patch.object(lifecycle, 'collect_owned') as full:
            self.assertFalse(lifecycle.collect_owned_children(owned))
        full.assert_called_once_with(owned)
    def test_child_of_unrelated_parent_is_not_adopted(self):
        self.process(10, 1, 'root', [11])
        self.process(11, 99, 'unrelated')
        owned = {10: 'root'}
        self.assertTrue(lifecycle.collect_owned_children(owned))
        self.assertEqual(owned, {10: 'root'})


@unittest.skipUnless(os.getuid() == 0 and Path('/proc/self/task').is_dir(), 'Linux root-owned process fixture')
class LiveChildDiscoveryTests(unittest.TestCase):
    def test_worker_thread_child_and_separate_session_grandchild_are_discovered(self):
        # This fixture spawns only sleeping synthetic processes, never inference.
        child_code = ("import json,os,subprocess,sys,time;os.setsid();"
                      "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                      "print(json.dumps({'grandchild':child.pid}),flush=True);time.sleep(60)")
        parent_code = ("import json,os,subprocess,sys,threading\n"
                       "def spawn():\n"
                       " child=subprocess.Popen([sys.executable,'-c'," + repr(child_code) + "],stdout=subprocess.PIPE,text=True)\n"
                       " record=json.loads(child.stdout.readline())\n"
                       " print(json.dumps({'child':child.pid,'grandchild':record['grandchild'],'worker_tid':threading.get_native_id()}),flush=True)\n"
                       " child.wait()\n"
                       "worker=threading.Thread(target=spawn);worker.start();worker.join()\n")
        parent = subprocess.Popen([sys.executable, '-c', parent_code], stdout=subprocess.PIPE, text=True)
        record = lifecycle.process_record(parent.pid)
        owned = {parent.pid: record['stamp']}
        try:
            with selectors.DefaultSelector() as ready:
                ready.register(parent.stdout, selectors.EVENT_READ)
                self.assertTrue(ready.select(5), 'Synthetic process fixture did not start')
            descendants = json.loads(parent.stdout.readline())
            self.assertNotEqual(descendants['worker_tid'], parent.pid)
            main_children_path = Path('/proc', str(parent.pid), 'task', str(parent.pid), 'children')
            supported = main_children_path.exists()
            if supported:
                self.assertNotIn(str(descendants['child']), main_children_path.read_text().split())
            self.assertEqual(os.getpgid(descendants['child']), descendants['child'])
            with patch.object(lifecycle, 'collect_owned', wraps=lifecycle.collect_owned) as full:
                self.assertEqual(lifecycle.collect_owned_children(owned), supported)
            if supported:
                full.assert_not_called()
            else:
                full.assert_called_once_with(owned)
            for pid in (descendants['child'], descendants['grandchild']):
                self.assertEqual(owned[pid], lifecycle.process_record(pid)['stamp'])
        finally:
            # Final cleanup intentionally retains the original full discovery.
            # Cancel from the original root identity alone: the immediate full
            # collector must find both descendants without a periodic wait.
            with patch.object(lifecycle, 'collect_owned', wraps=lifecycle.collect_owned) as full:
                lifecycle.cleanup(parent, {parent.pid: record['stamp']})
            self.assertGreaterEqual(full.call_count, 2)
            parent.stdout.close()


if __name__ == '__main__':
    unittest.main()
