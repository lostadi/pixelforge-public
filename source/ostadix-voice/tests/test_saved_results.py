"""Saved receipt retrieval stays lossless, paginated and conversation-scoped."""
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import AppError, Store, clean
from local_chat import LocalChatAgent
from saved_results import dispatch


class SavedResultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.agent = SimpleNamespace(store=self.store, sid='first', cancelled=threading.Event())

    def tearDown(self):
        self.store.db.close()
        self.temp.cleanup()

    def call(self, *argv):
        return dispatch(self.agent, {'argv': ['@results', *argv]})

    def test_long_unicode_result_can_be_reassembled_after_restart(self):
        result = {'packages': ['com.example.app' + str(i) for i in range(800)], 'text': '中文😀' * 3000}
        event = self.store.event('first', 'tool_result', {'success': True, 'result': result})
        self.store.db.close()
        self.store = Store(self.temp.name)
        self.agent.store = self.store
        parts, offset = [], 0
        while True:
            page = self.call('read', str(event), str(offset), '4000')
            self.assertLessEqual(len(page['content']), 4000)
            self.assertTrue(page['success'])
            parts.append(page['content'])
            if not page['more']:
                break
            offset = page['next_offset']
        self.assertEqual(json.loads(''.join(parts)), result)

    def test_redacted_result_pages_survive_model_feedback_and_persistence_cleaning(self):
        original = {'password': 'fixture-only', 'answer': 'must remain',
                    'note': 'Authorization: Bearer fixture-token', 'nested': {'API_KEY': 'fixture-two'}}
        result = clean(original)
        event = self.store.event('first', 'tool_result', {'success': True, 'result': original})
        parts, offset = [], 0
        while True:
            page = self.call('read', str(event), str(offset), '7')
            model_message = json.loads(LocalChatAgent._tool_content(page, True, 'tool-1', event))
            self.assertEqual(model_message['result']['content'], page['content'])
            saved = self.store.event('first', 'tool_result', {'success': True, 'result': page})
            with self.store.lock:
                row = self.store.db.execute('SELECT data FROM events WHERE seq=?', (saved,)).fetchone()
            self.assertEqual(json.loads(row['data'])['result']['content'], page['content'])
            parts.append(page['content'])
            if not page['more']:
                break
            offset = page['next_offset']
        self.assertEqual(json.loads(''.join(parts)), result)
        self.assertNotIn('fixture-', ''.join(parts))

    def test_other_conversation_and_non_tool_events_are_not_exposed(self):
        other = self.store.event('second', 'tool_result', {'success': True, 'result': {'text': 'other'}})
        notice = self.store.event('first', 'notice', {'text': 'notice'})
        for event in (other, notice):
            with self.assertRaisesRegex(AppError, 'this conversation'):
                self.call('read', str(event))
        self.assertEqual(self.call('list')['results'], [])

    def test_index_pages_include_failed_receipts_and_working_read_handles(self):
        events = [self.store.event('first', 'tool_result', {'success': i != 1, 'result': {'value': i}})
                  for i in range(3)]
        first = self.call('list', '0', '2')
        self.assertEqual([entry['event'] for entry in first['results']], events[:0:-1])
        self.assertTrue(first['more'])
        failed = dispatch(self.agent, {'argv': first['results'][1]['read_argv']})
        self.assertFalse(failed['success'])
        self.assertEqual(json.loads(failed['content']), {'value': 1})
        last = self.call('list', str(first['next_offset']), '2')
        self.assertEqual(last['results'][0]['event'], events[0])
        self.assertFalse(last['more'])

    def test_invalid_limits_ids_and_offsets_are_rejected(self):
        event = self.store.event('first', 'tool_result', {'success': True, 'result': {}})
        for argv in (('read', '-1'), ('read', str(event), '-1'), ('read', str(event), '0', '0'),
                     ('read', str(event), '0', '12001'), ('read', str(event), '99999'),
                     ('list', '0', '51'), ('read', '/absolute/log')):
            with self.subTest(argv=argv), self.assertRaises(AppError):
                self.call(*argv)

    def test_cancelled_read_does_not_return_data(self):
        self.agent.cancelled.set()
        with self.assertRaisesRegex(AppError, 'cancelled'):
            self.call('list')

    def test_escaped_pages_fit_the_actual_model_envelope_and_recover_every_character(self):
        from local_chat import tool_feedback
        result = {'text': '\\"' * 10000}
        event = self.store.event('first', 'tool_result', {'success': True, 'result': result})
        offset, parts = 0, []
        while True:
            page = self.call('read', str(event), str(offset), '12000')
            envelope = json.loads(tool_feedback(page, True, limit=6000, event=event))
            self.assertNotIn('preview_truncated', envelope)
            self.assertEqual(envelope['result']['content'], page['content'])
            parts.append(envelope['result']['content'])
            if not page['more']:
                break
            self.assertGreater(page['next_offset'], offset)
            offset = page['next_offset']
        self.assertEqual(json.loads(''.join(parts)), result)

    def test_large_result_index_pages_do_not_clip_retrieval_handles(self):
        from local_chat import tool_feedback
        events = [self.store.event('first', 'tool_result', {'success': True, 'result': {'text': '\\"' * 500}})
                  for _ in range(60)]
        offset, recovered = 0, []
        while True:
            page = self.call('list', str(offset), '50')
            envelope = json.loads(tool_feedback(page, True, limit=6000))
            self.assertNotIn('preview_truncated', envelope)
            recovered.extend(entry['event'] for entry in envelope['result']['results'])
            if not page['more']:
                break
            offset = page['next_offset']
        self.assertEqual(recovered, list(reversed(events)))


if __name__ == '__main__':
    unittest.main()
