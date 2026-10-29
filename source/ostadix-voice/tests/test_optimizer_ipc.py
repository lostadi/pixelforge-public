"""Optimizer Unix control authentication/lifecycle; no model or root commands."""
import json
import os
from pathlib import Path
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import optimizer
from local_runtime import LocalError


def receive_json(connection):
    value = bytearray()
    while b'\n' not in value:
        chunk = connection.recv(4096)
        if not chunk:
            raise LocalError('Fixture peer closed')
        value.extend(chunk)
        if len(value) > 4096:
            raise LocalError('Fixture reply exceeded bound')
    return json.loads(value.partition(b'\n')[0])


class OptimizerIPCTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name) / 'local'
        self.state = self.base / 'optimizer'
        self.state.mkdir(parents=True)
        optimizer.write_json(self.state / 'config.json', {'version': 1, 'enabled': True, 'objective': 'balanced'})
        self.root_calls = self.enterContext(patch('optimizer.root_call', side_effect=AssertionError('No root calls')))
        self.spawn = self.enterContext(patch('optimizer.subprocess.Popen', side_effect=AssertionError('No model or process starts')))
        self.signals = self.enterContext(patch('optimizer.os.kill', side_effect=AssertionError('No PID signals')))
        self.tuner = Mock()
        self.tuner.restore.return_value = {'ok': True, 'restored': True}
        self.monitor = optimizer.Monitor(self.base, sample=Mock(), client=Mock(), tuner=self.tuner)
        self.monitor.save(phase='monitoring')
        self.addCleanup(self.cleanup_monitor)

    def cleanup_monitor(self):
        if self.monitor.control_socket is not None and not self.monitor.control_closed.is_set():
            self.monitor.close_control()
        self.monitor.db.close()

    def start(self):
        self.monitor.start_control()
        return optimizer.OptimizerControl(self.base)

    def connect(self):
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(1)
        connection.connect(str(self.state / 'control.sock'))
        self.addCleanup(connection.close)
        return connection

    def test_live_status_works_when_proc_is_hidden_without_root_or_spawning(self):
        control = self.start()
        with patch('optimizer.process_stamp', return_value=None):
            result = control.status()
        self.assertTrue(result['running'])
        self.assertEqual(result['pid'], os.getpid())
        self.assertEqual(result['process_stamp'], self.monitor.snapshot['process_stamp'])
        self.root_calls.assert_not_called()
        self.spawn.assert_not_called()
        self.signals.assert_not_called()

    def test_stale_metadata_stamp_does_not_identify_the_live_peer(self):
        control = self.start()
        metadata = optimizer.read_json(self.state / 'status.json')
        metadata['process_stamp'] += 1
        optimizer.write_json(self.state / 'status.json', metadata)
        with patch('optimizer.process_stamp', return_value=None):
            result = control.status()
        self.assertFalse(result['running'])
        self.assertIsNone(result['pid'])

    def test_stream_fragmentation_is_supported(self):
        self.start()
        connection = self.connect()
        connection.sendall(b'{"operation":')
        time.sleep(.025)
        connection.sendall(b'"status"}\n')
        result = receive_json(connection)
        self.assertEqual(result['pid'], os.getpid())
        self.assertFalse(self.monitor.stop_event.is_set())

    def test_invalid_or_oversized_operations_do_not_stop_monitor(self):
        self.start()
        for request in (
            {'operation': 'execute', 'command': 'anything'},
            {'operation': 'status', 'extra': True},
            {'operation': 'stop'},
            {'operation': 'status', 'padding': 'x' * 1000},
        ):
            with self.subTest(request=request):
                connection = self.connect()
                connection.sendall((json.dumps(request) + '\n').encode())
                try:
                    answer = connection.recv(4096)
                except ConnectionResetError:
                    answer = b''
                self.assertFalse(answer)
                connection.close()
                self.assertFalse(self.monitor.stop_event.is_set())
        self.assertEqual(optimizer.control_request(self.state)['pid'], os.getpid())

    def test_stale_stop_identity_does_not_stop_monitor(self):
        self.start()
        expected = dict(self.monitor.snapshot)
        expected['process_stamp'] += 1
        with self.assertRaises((LocalError, OSError, ValueError)):
            optimizer.control_request(self.state, 'stop', expected=expected)
        self.assertFalse(self.monitor.stop_event.is_set())

    def test_wrong_stop_peer_pid_is_rejected_before_mutation(self):
        self.start()
        expected = dict(self.monitor.snapshot, pid=os.getpid() + 1)
        with self.assertRaises(LocalError):
            optimizer.control_request(self.state, 'stop', expected=expected)
        self.assertFalse(self.monitor.stop_event.is_set())

    def test_stop_sets_event_but_socket_survives_until_explicit_cleanup(self):
        self.start()
        result = optimizer.control_request(self.state, 'stop', expected=self.monitor.snapshot)
        self.assertTrue(result['stopping'])
        self.assertTrue(self.monitor.stop_event.is_set())
        self.assertTrue((self.state / 'control.sock').exists())
        self.assertTrue(optimizer.control_request(self.state)['stopping'])
        self.monitor.close_control()
        self.assertFalse((self.state / 'control.sock').exists())

    def test_stop_waits_for_restore_and_control_close_without_pid_signals(self):
        control = self.start()
        output = []
        failures = []
        journal = Mock()
        journal.restore.return_value = {'ok': True, 'restored': True}
        def stop():
            try:
                output.append(control.stop())
            except BaseException as error:
                failures.append(error)
        with patch('optimizer.process_stamp', return_value=None), \
                patch('optimizer_tuning.ProcessTuner', return_value=journal), \
                patch('optimizer.probe', return_value={'released': True}) as probe:
            worker = threading.Thread(target=stop)
            worker.start()
            try:
                self.assertTrue(self.monitor.stop_event.wait(2))
                time.sleep(.05)
                self.assertTrue(worker.is_alive(), 'Stop returned before daemon cleanup')
                self.assertFalse(output)
                self.monitor.restore()
                self.monitor.save(phase='stopped')
                self.monitor.close_control()
                worker.join(timeout=2)
                self.assertFalse(worker.is_alive())
            finally:
                if not self.monitor.control_closed.is_set():
                    self.monitor.close_control()
                worker.join(timeout=2)
        self.assertFalse(failures, failures)
        self.assertEqual(len(output), 1)
        self.assertFalse(output[0]['running'])
        self.assertTrue(output[0]['restoration']['ok'])
        self.tuner.restore.assert_called_once()
        journal.restore.assert_called_once()
        probe.assert_called_once_with(release=True)
        self.signals.assert_not_called()

    def test_foreign_uid_peer_rejected_before_any_request(self):
        connection = Mock()
        connection.__enter__ = Mock(return_value=connection)
        connection.__exit__ = Mock(return_value=False)
        connection.getsockopt.return_value = struct.pack('3i', os.getpid(), os.getuid() + 1, os.getgid())
        with patch('optimizer.socket.socket', return_value=connection):
            with self.assertRaises(LocalError):
                optimizer.control_request(self.state)
        connection.sendall.assert_not_called()

    def test_response_cannot_claim_another_process_pid(self):
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.state / 'control.sock'))
        listener.listen(1)
        self.addCleanup(listener.close)
        def counterfeit():
            connection, _ = listener.accept()
            with connection:
                connection.recv(4096)
                connection.sendall((json.dumps({'pid': os.getpid() + 1, 'uid': os.getuid(), 'process_stamp': 1}) + '\n').encode())
        worker = threading.Thread(target=counterfeit)
        worker.start()
        try:
            with self.assertRaises(LocalError):
                optimizer.control_request(self.state)
        finally:
            worker.join(timeout=1)
            self.assertFalse(worker.is_alive())

    def test_start_control_preserves_non_socket_file(self):
        path = self.state / 'control.sock'
        path.write_text('preserve')
        with self.assertRaises(LocalError):
            self.monitor.start_control()
        self.assertEqual(path.read_text(), 'preserve')

    def test_start_control_rejects_symlink_without_unlinking_target(self):
        target = self.state / 'target'
        target.write_text('preserve')
        path = self.state / 'control.sock'
        path.symlink_to(target)
        with self.assertRaises(LocalError):
            self.monitor.start_control()
        self.assertTrue(path.is_symlink())
        self.assertEqual(target.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
