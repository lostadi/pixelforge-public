"""Verify that the real-model usability diagnostic cannot mutate device state."""
import ast
import json
import unittest

from tests.evaluate_local_agent import PROGRAM, harmless_source, native_value, numeric_expression, permit


class AgentEvaluationSafetyTests(unittest.TestCase):
    def test_fixture_reads_and_phone_queries_are_permitted(self):
        self.assertTrue(permit({'operation': 'read', 'path': 'release-notes.txt'}))
        for argv in (['@phone', 'volume'], ['@phone', 'help'], ['@phone', 'status']):
            self.assertTrue(permit({'operation': 'command', 'argv': argv}))

    def test_mutating_or_unrelated_requests_are_rejected(self):
        for arguments in ({'operation': 'write', 'path': 'release-notes.txt', 'content': 'changed'},
                          {'operation': 'delete', 'path': 'release-notes.txt'},
                          {'operation': 'read', 'path': '../private.txt'},
                          {'operation': 'command', 'argv': ['sh', '-c', 'id']},
                          {'operation': 'command', 'argv': ['@voice', 'on']},
                          {'operation': 'command', 'argv': ['@phone', 'volume', '0']},
                          {'operation': 'command', 'argv': ['@phone', 'open', 'com.example.app']},
                          {'operation': 'command', 'argv': ['@phone', 'tap', 'token', '1', '2']}):
            with self.subTest(arguments=arguments):
                self.assertFalse(permit(arguments))

    def test_only_bounded_arithmetic_can_reach_the_real_native_executor(self):
        self.assertTrue(harmless_source(PROGRAM))
        self.assertTrue(harmless_source('-(23 * 17) + 8'))
        for source in ('__import__("os").system("id")', '2 ** 999999999',
                       'python^(\nopen("file","w").write("x")\n)_python',
                       'python^(\n__oval_result__ = 2\nprint("extra")\n)_python',
                       'True', '10 ** 3', 'x + 1'):
            self.assertFalse(harmless_source(source))
        request = {'source': PROGRAM, 'action': 'execute'}
        self.assertTrue(permit({'operation': 'command', 'argv': ['@ostadix', 'call', 'o_execute', json.dumps(request)]}))
        for extra in ({'env': {'UNRELATED': '1'}}, {'path': '/private/program.O'},
                      {'timeout_secs': 120}, {'background': True}):
            self.assertFalse(permit({'operation': 'command', 'argv':
                ['@ostadix', 'call', 'o_execute', json.dumps(dict(request, **extra))]}))

    def test_native_numeric_evidence_is_decoded_from_real_ovalue_shape(self):
        self.assertEqual(native_value({'t': 'number', 'v': {'kind': 'int', 'v': '383'}}), 383)
        self.assertEqual(native_value({'t': 'number', 'v': {'kind': 'binary_float', 'format': 'f64',
                         'bits': [64, 84, 32, 0, 0, 0, 0, 0]}}), 80.5)

    def test_inferred_literal_answer_is_not_the_requested_program(self):
        requested = ast.dump(numeric_expression(PROGRAM))
        self.assertEqual(requested, ast.dump(numeric_expression('(23*17)-8')))
        self.assertNotEqual(requested, ast.dump(numeric_expression('383')))


if __name__ == '__main__':
    unittest.main()
