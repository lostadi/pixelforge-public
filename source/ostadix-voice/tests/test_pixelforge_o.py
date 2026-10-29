import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch


PROGRAM = Path(__file__).resolve().parents[1] / 'runtime/pixelforge.O'


class PixelForgeOTests(unittest.TestCase):
    def run_program(self, request=None, result=None, returncode=0):
        source = PROGRAM.read_text()
        self.assertTrue(source.startswith('python^(\n') and source.endswith(')_python\n'))
        code = compile(source[len('python^(\n'):-len(')_python\n')], str(PROGRAM), 'exec')
        environment = {} if request is None else {'PIXELFORGE_REQUEST_JSON':
                      request if isinstance(request, str) else json.dumps(request)}
        stdout = json.dumps({'ok': True, 'available': True} if result is None else result)
        process = SimpleNamespace(stdout=stdout, stderr='', returncode=returncode)
        namespace = {}
        with patch.dict(os.environ, environment, clear=True), patch('subprocess.run', return_value=process) as run:
            self.last_run = run
            exec(code, namespace)
        return namespace['__oval_result__'], run

    def test_default_status_returns_actual_host_result(self):
        value, run = self.run_program(result={'ok': True, 'available': True, 'compiled_banks': 556})
        self.assertEqual(value['compiled_banks'], 556)
        self.assertEqual(json.loads(run.call_args.kwargs['input']), {'action': 'status'})
        self.assertEqual(run.call_args.args[0], ['/system/bin/su', '-c',
            '/data/data/com.termux/files/usr/bin/python3 -I -S /data/adb/ostadix-pixelforge/run.py'])

    def test_prompt_stays_literal_json_stdin_without_command_interpolation(self):
        prompt = "literal '; $(touch /tmp/not-executed) `id`\nsecond line"
        value, run = self.run_program({'action': 'generate', 'prompt': prompt, 'tokens': 2})
        self.assertNotIn(prompt, repr(run.call_args.args[0]))
        self.assertEqual(json.loads(run.call_args.kwargs['input'])['prompt'], prompt)
        self.assertNotIn('shell', run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs['timeout'], 120)

    def test_unsupported_actions_fields_and_bad_limits_never_spawn(self):
        for request in ({'action': 'shell'}, {'action': 'status', 'command': 'id'},
                        {'action': 'generate', 'prompt': 'test', 'tokens': True},
                        {'action': 'generate', 'prompt': 'test', 'tokens': 3},
                        {'action': 'generate', 'prompt': 'a' * 2001},
                        {'action': 'generate', 'prompt': 'a\0b'},
                        '{"action":"status","action":"generate"}'):
            with self.subTest(request=str(request)[:100]):
                with self.assertRaises(ValueError):
                    self.run_program(request)
                self.last_run.assert_not_called()

    def test_real_host_error_is_visible(self):
        with self.assertRaisesRegex(RuntimeError, 'Pinned Huihui GGUF changed'):
            self.run_program(result={'ok': False, 'error': 'Pinned Huihui GGUF changed'}, returncode=1)

    def test_status_unavailable_is_preserved_without_inventing_generation(self):
        value, run = self.run_program(result={'ok': True, 'available': False})
        self.assertFalse(value['available'])
        self.assertEqual(run.call_count, 1)
        self.assertEqual(json.loads(run.call_args.kwargs['input'])['action'], 'status')


if __name__ == '__main__':
    unittest.main()
