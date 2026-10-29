import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import service


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


if __name__ == '__main__':
    unittest.main()
