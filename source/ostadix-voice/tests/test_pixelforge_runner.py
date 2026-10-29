"""Bounded request and owned-process cleanup tests; no NPU or model calls."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pixelforge_runner as runner


class RequestTests(unittest.TestCase):
    def request(self, value):
        return runner.read_request(io.BytesIO(json.dumps(value).encode()))

    def test_status_and_bounded_generate_only(self):
        self.assertEqual(self.request({'action': 'status'}), {'action': 'status'})
        self.assertEqual(self.request({'action': 'generate', 'prompt': 'Hi'})['tokens'], 1)
        self.assertEqual(self.request({'action': 'generate', 'prompt': 'Hi', 'tokens': 2})['tokens'], 2)

    def test_invalid_fields_prompts_tokens_and_oversized_frames_are_rejected(self):
        cases = [{}, [], {'action': 'stop'}, {'action': 'status', 'prompt': 'Hi'},
                 {'action': 'generate', 'prompt': ''}, {'action': 'generate', 'prompt': 'x' * 2001},
                 {'action': 'generate', 'prompt': 'a\0b'}, {'action': 'generate', 'prompt': 'Hi', 'tokens': True},
                 {'action': 'generate', 'prompt': 'Hi', 'tokens': 3},
                 {'action': 'generate', 'prompt': 'Hi', 'command': 'anything'}]
        for value in cases:
            with self.subTest(value=str(value)[:100]), self.assertRaises(ValueError):
                self.request(value)
        with self.assertRaises(ValueError):
            runner.read_request(io.BytesIO(b'x' * 8193))

    def test_prompt_stays_in_stdin_and_command_is_fixed(self):
        code = runner.child_code()
        self.assertIn('json.load(sys.stdin)', code)
        self.assertIn('--read-only', code)
        self.assertIn('--timeout', code)
        self.assertNotIn('shell=True', code)

    def test_asset_hashing_refuses_symlinks_and_expired_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'asset'; path.write_bytes(b'owned bytes')
            alias = Path(directory) / 'alias'; alias.symlink_to(path)
            with self.assertRaises(OSError):
                runner.digest(alias)
            with self.assertRaises(TimeoutError):
                runner.digest(path, deadline=0)

    def test_recycled_pid_stamp_is_never_signalled(self):
        with patch.object(runner, 'process_record', return_value={'stamp': 'new'}), \
                patch.object(runner, 'open_pidfd') as opened:
            runner.signal_owned({1234: 'old'}, 15)
        opened.assert_not_called()


@unittest.skipUnless(os.getuid() == 0, 'Root helper fixture uses only its own root children')
class RootHelperTests(unittest.TestCase):
    def test_tampered_pins_or_model_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'model.gguf'; model.write_bytes(b'fixture model')
            metadata = root / 'MODEL.json'
            metadata.write_text(json.dumps({'complete': True, 'completed_weight_dtypes': ['int8'],
                'artifacts': {str(index): {} for index in range(556)},
                'source_gguf': {'sha256': runner.MODEL_SHA, 'path': str(model), 'bytes': model.stat().st_size}}))
            pins = root / 'pins.json'
            pins.write_text(json.dumps({'version': 1, 'sha256': {'metadata': runner.digest(metadata)}}))
            with patch.object(runner, 'ROOT', root), patch.object(runner, 'MODEL', model), \
                    patch.object(runner, 'ARTIFACTS', root), \
                    patch.object(runner, 'pinned_files', return_value={'metadata': metadata}):
                self.assertTrue(runner.verify()['complete'])
                with self.assertRaisesRegex(ValueError, 'GGUF changed'):
                    runner.verify(full_model=True)
                metadata.write_text('{}')
                with self.assertRaisesRegex(ValueError, 'asset changed'):
                    runner.verify()

    def test_cleanup_stops_only_the_owned_tree_and_keeps_bystander_alive(self):
        code = ('import subprocess,sys,time; '
                'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                'print(child.pid,flush=True); time.sleep(60)')
        owned = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, text=True, start_new_session=True)
        other = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            child = int(owned.stdout.readline())
            record = runner.process_record(owned.pid)
            identities = {owned.pid: record['stamp']}
            runner.collect_owned(identities)
            self.assertIn(child, identities)
            runner.cleanup(owned, identities)
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(runner.process_record(child))
            self.assertIsNone(other.poll())
        finally:
            if owned.poll() is None:
                owned.kill()
            owned.communicate(timeout=3)
            other.terminate(); other.wait(timeout=3)


if __name__ == '__main__':
    unittest.main()
