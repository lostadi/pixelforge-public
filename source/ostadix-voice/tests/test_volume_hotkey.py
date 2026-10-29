"""Volume hotkey tests use synthetic input packets; no live buttons or root."""
from pathlib import Path
import os
import queue
import shlex
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from volume_hotkey import VolumeHotkey, VolumeTapSequence, helper_source, root_command


class TapSequenceTests(unittest.TestCase):
    def feed(self, events):
        sequence = VolumeTapSequence()
        return [sequence.feed(code, value, at) for code, value, at in events]

    def test_complete_down_up_taps_toggle_only_after_the_final_release(self):
        self.assertEqual(self.feed([(114, 1, 0), (114, 0, .1), (115, 1, .3), (115, 0, .4)]),
                         [False, False, False, True])

    def test_long_holds_repeats_chords_wrong_order_and_expired_gaps_never_toggle(self):
        cases = [
            [(114, 1, 0), (114, 0, .6), (115, 1, .7), (115, 0, .8)],
            [(114, 1, 0), (114, 0, .1), (115, 1, .2), (115, 0, .8)],
            [(114, 1, 0), (114, 2, .1), (114, 0, .2), (115, 1, .3), (115, 0, .4)],
            [(114, 1, 0), (114, 0, .1), (115, 1, .2), (115, 2, .3), (115, 0, .4)],
            [(114, 1, 0), (115, 1, .1), (114, 0, .2), (115, 0, .3)],
            [(115, 1, 0), (115, 0, .1), (114, 1, .2), (114, 0, .3)],
            [(114, 1, 0), (114, 0, .1), (115, 1, .9), (115, 0, 1)],
            [(114, 1, 0), (114, 0, .1), (116, 1, .2), (116, 0, .3), (115, 1, .4), (115, 0, .5)],
            [(114, 1, 1), (114, 0, 1.1), (115, 1, .5), (115, 0, .6)],
            [(114, 0, 0), (115, 1, .1), (115, 0, .2)],
        ]
        for events in cases:
            with self.subTest(events=events):
                self.assertFalse(any(self.feed(events)))

    def test_each_complete_sequence_toggles_once_and_recovers_after_a_chord(self):
        events = [(114, 1, 0), (115, 1, .1), (114, 0, .2), (115, 0, .3)]
        for base in (1, 2):
            events.extend([(114, 1, base), (114, 0, base + .1),
                           (115, 1, base + .2), (115, 0, base + .3)])
        self.assertEqual(sum(self.feed(events)), 2)


class ReaderTests(unittest.TestCase):
    @staticmethod
    def close_fd(descriptor):
        try:
            os.close(descriptor)
        except OSError:
            pass

    def fixture(self, **settings):
        source_fd, write_fd = os.pipe()
        created = []
        source = ('namespace={"__name__":"fixture"};exec(' + repr(helper_source()) +
                  ',namespace);namespace["run"](fixture_fd=' + str(source_fd) + ')')
        def factory(command, **kwargs):
            process = subprocess.Popen([sys.executable, '-I', '-S', '-B', '-c', source],
                                       pass_fds=(source_fd,), **kwargs)
            created.append(process)
            return process
        listener = VolumeHotkey(lambda: ['offline-fixture'], factory, **settings)
        self.addCleanup(listener.close)
        self.addCleanup(self.close_fd, write_fd)
        self.addCleanup(self.close_fd, source_fd)
        listener.start()
        self.assertEqual(listener.events.get(timeout=2), ('ready', None))
        return listener, write_fd, created

    @staticmethod
    def packet(kind, code, value, at):
        seconds = int(at)
        return struct.pack('@llHHi', seconds, round((at - seconds) * 1000000), kind, code, value)

    def taps(self, fd, base=1):
        for code, value, at in ((114, 1, base), (114, 0, base + .1),
                                (115, 1, base + .2), (115, 0, base + .3)):
            os.write(fd, self.packet(1, code, value, at))

    def test_owned_helper_emits_gestures_only_and_exits_when_control_pipe_closes(self):
        listener, fd, created = self.fixture()
        # Power events and arbitrary bytes never become a raw event stream to UI.
        os.write(fd, self.packet(1, 116, 1, 0) + self.packet(1, 116, 0, .1))
        self.taps(fd)
        self.assertEqual(listener.events.get(timeout=2), ('toggle', None))
        listener.close()
        self.assertFalse(listener.worker.is_alive())
        self.assertIsNotNone(created[0].poll())
        self.assertTrue(created[0].stdin.closed)
        self.assertTrue(created[0].stdout.closed)
        self.assertEqual(listener.status, 'closed')
        self.assertTrue(listener.events.empty())

    def test_split_input_packets_are_buffered_without_losing_a_gesture(self):
        listener, fd, _ = self.fixture()
        packets = b''.join(self.packet(1, code, value, at) for code, value, at in
                           ((114, 1, 1), (114, 0, 1.1), (115, 1, 1.2), (115, 0, 1.3)))
        for start in range(0, len(packets), 7):
            os.write(fd, packets[start:start + 7])
        self.assertEqual(listener.events.get(timeout=2), ('toggle', None))

    def test_dropped_input_resets_partial_gesture_until_a_syn_report(self):
        listener, fd, _ = self.fixture()
        for kind, code, value, at in ((1, 114, 1, 0), (1, 114, 0, .1), (0, 3, 0, .15),
                                     (1, 115, 1, .2), (1, 115, 0, .3), (0, 0, 0, .4)):
            os.write(fd, self.packet(kind, code, value, at))
        self.taps(fd, base=1)
        self.assertEqual(listener.events.get(timeout=2), ('toggle', None))
        with self.assertRaises(queue.Empty):
            listener.events.get(timeout=.05)

    def test_device_disconnection_reports_error_and_stops_the_helper(self):
        listener, fd, created = self.fixture()
        os.close(fd)
        kind, message = listener.events.get(timeout=2)
        self.assertEqual(kind, 'error')
        self.assertIn('disconnected', message)
        listener.close()
        self.assertIsNotNone(created[0].poll())

    def test_unavailable_root_reports_failure_without_a_ready_event(self):
        def unavailable():
            raise RuntimeError('existing root command is unavailable')
        listener = VolumeHotkey(unavailable)
        self.addCleanup(listener.close)
        listener.start()
        self.assertEqual(listener.events.get(timeout=2), ('error', 'existing root command is unavailable'))
        self.assertEqual(listener.status, 'error')
        self.assertTrue(listener.events.empty())

    def test_root_denial_is_reported_without_claiming_ready(self):
        listener = VolumeHotkey(lambda: [sys.executable, '-c', 'print("Permission denied");raise SystemExit(1)'])
        self.addCleanup(listener.close)
        listener.start()
        self.assertEqual(listener.events.get(timeout=2), ('error', 'Permission denied'))
        self.assertEqual(listener.status, 'error')

    def test_start_and_close_are_idempotent_and_close_before_start_does_nothing(self):
        listener, _, created = self.fixture()
        listener.start()
        self.assertEqual(len(created), 1)
        listener.close()
        listener.close()
        unused = VolumeHotkey(lambda: self.fail('A closed listener must not launch'))
        unused.close()
        unused.start()
        self.assertIsNone(unused.worker)

    def test_startup_timeout_closes_the_owned_lifetime_pipe(self):
        # Fixture exits on stdin EOF without ever claiming device readiness.
        listener = VolumeHotkey(lambda: [sys.executable, '-c', 'import sys;sys.stdin.buffer.read()'],
                                startup_timeout=.1, close_timeout=.3)
        self.addCleanup(listener.close)
        listener.start()
        kind, message = listener.events.get(timeout=2)
        self.assertEqual(kind, 'error')
        self.assertIn('did not become ready', message)
        listener.close()
        self.assertFalse(listener.worker.is_alive())

    def test_root_command_uses_isolated_python_and_no_permission_changes(self):
        with patch('volume_hotkey.sys.executable', '/data/data/com.termux/files/usr/bin/python3'), \
                patch('volume_hotkey.Path.resolve', return_value=Path('/data/data/com.termux/files/usr/bin/python3')), \
                patch('volume_hotkey.Path.exists', return_value=True), \
                patch('volume_hotkey.shutil.which', return_value='/data/data/com.termux/files/usr/bin/su'), \
                patch.dict(os.environ, {'PREFIX': '/data/data/com.termux/files/usr'}):
            command = root_command()
        self.assertEqual(command[:2], ['/data/data/com.termux/files/usr/bin/su', '-c'])
        argv = shlex.split(command[2])
        self.assertIn('-I', argv)
        self.assertIn('-S', argv)
        self.assertIn('-B', argv)
        self.assertIn('run()', argv[-1])
        self.assertNotIn('EVIOCGRAB', argv[-1])
        self.assertNotIn('chmod', argv[-1])


if __name__ == '__main__':
    unittest.main()
