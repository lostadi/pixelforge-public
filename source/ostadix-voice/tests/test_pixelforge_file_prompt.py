"""The rendered text-model guide uses the real workspace file contract."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Agent, Store
from conversation_protocol import encode_conversation
from pixelforge_chat import conversation_prompt


ROLE = 'phone conversation assistant'


class PixelForgeFilePromptTests(unittest.TestCase):
    def test_rendered_file_example_runs_through_workspace_executor(self):
        prompt = conversation_prompt(encode_conversation([
            {'role': 'user', 'content': 'Read a project file.'}]), ROLE)
        examples = [json.loads(line) for line in prompt.splitlines()
                    if line.startswith('{"tool":"workspace","arguments":')]
        reads = [example for example in examples
                 if example['arguments']['operation'] == 'read']
        self.assertEqual(len(reads), 1)
        self.assertEqual(reads[0]['arguments'],
                         {'operation': 'read', 'path': 'example.txt'})
        self.assertIn('Use argv only for operation command.', prompt)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / 'project'
            project.mkdir()
            (project / 'example.txt').write_text('Actual workspace file content')
            store = Store(root / 'state')
            agent = Agent(store, project)
            try:
                agent.sid = store.create(project)
                result = agent.execute(reads[0]['arguments'])
                self.assertEqual(result['content'], 'Actual workspace file content')
            finally:
                agent.close()
                store.db.close()

    def test_missing_path_receipt_remains_failed_data_in_rendered_prompt(self):
        arguments = {'operation': 'read', 'argv': ['example.txt']}
        receipt = {'success': False,
                   'result': {'error': 'A file path is required'}}
        call = {'id': 'malformed-read', 'type': 'function',
                'function': {'name': 'workspace',
                             'arguments': json.dumps(arguments)}}
        task = encode_conversation([
            {'role': 'user', 'content': 'Read example.txt.'},
            {'role': 'assistant', 'content': None, 'tool_calls': [call]},
            {'role': 'tool', 'tool_call_id': 'malformed-read',
             'name': 'workspace', 'content': json.dumps(receipt)}])
        prompt = conversation_prompt(task, ROLE)
        self.assertIn(json.dumps(receipt), prompt)
        self.assertIn('A missing-path error means the request is malformed, '
                      'not that the file is absent.', prompt)
        self.assertIn('Correct that read request with a path field before '
                      'answering from its actual result.', prompt)
        # The adapter preserves the failed request; it never translates argv
        # into a path or fabricates a successful receipt.
        rendered_call = next(json.loads(line) for line in prompt.splitlines()
                             if line.startswith('{"name":"workspace","arguments":'))
        self.assertEqual(json.loads(rendered_call["arguments"]), arguments)


if __name__ == '__main__':
    unittest.main()
