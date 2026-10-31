"""Registered GGUF selection, limits and stale-host isolation without root or hardware."""
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from local_runtime import LocalError
from pixelforge_chat import (BACKEND, COMMAND, PixelForgeClient, registered_models, select_model)
from test_pixelforge_chat import Host, ROLE, TASK, done


class ModelSelectionTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.catalog = self.root / 'models.json'
        self.metadata = self.root / 'active.json'
        self.model = {'model_name': 'Granite fixture', 'model_sha256': 'a' * 64,
                      'context_per_worker': 4096, 'max_output_tokens': 64}
        self.descriptor = {'backend': BACKEND, 'experimental': True, 'model': self.model}
        self.write_catalog()
        self.publish('granite')

    def write_catalog(self):
        self.catalog.write_text(json.dumps({'schema': 'pixelforge.models.v1',
            'selected_model_id': 'granite', 'models': {'granite': self.descriptor}}))

    def publish(self, ident):
        self.metadata.write_text(json.dumps(dict(self.descriptor, model_id=ident)))

    def client(self):
        client = PixelForgeClient(metadata=self.metadata)
        self.addCleanup(client.close)
        return client

    def test_catalog_listing_is_passive_and_preserves_selected_name_and_limits(self):
        with patch('pixelforge_chat.subprocess.run') as run, patch('pixelforge_chat.subprocess.Popen') as spawn:
            catalog = registered_models(self.catalog)
        self.assertEqual(catalog['selected_model_id'], 'granite')
        self.assertEqual(catalog['models']['granite']['model_name'], 'Granite fixture')
        self.assertEqual(catalog['models']['granite']['max_output_tokens'], 64)
        run.assert_not_called(); spawn.assert_not_called()

    def test_invalid_catalog_identity_and_selected_entry_never_reach_root(self):
        cases = [
            {'schema': 'wrong', 'models': {'granite': self.descriptor}},
            {'schema': 'pixelforge.models.v1', 'models': []},
            {'schema': 'pixelforge.models.v1', 'models': {'../granite': self.descriptor}},
            {'schema': 'pixelforge.models.v1', 'selected_model_id': 'absent', 'models': {'granite': self.descriptor}},
            {'schema': 'pixelforge.models.v1', 'models': {'granite': dict(self.descriptor,
                model=dict(self.model, model_sha256='invalid'))}},
        ]
        with patch('pixelforge_chat.subprocess.run') as run:
            for value in cases:
                self.catalog.write_text(json.dumps(value))
                with self.subTest(value=value), self.assertRaises(LocalError):
                    select_model('granite', catalog=self.catalog, metadata=self.metadata)
        run.assert_not_called()

    def test_explicit_selection_passes_only_registered_identifier_to_fixed_launcher(self):
        result = subprocess.CompletedProcess(COMMAND, 0, '', '')
        with patch('pixelforge_chat.subprocess.run', return_value=result) as run:
            model = select_model('granite', catalog=self.catalog, metadata=self.metadata)
        self.assertEqual(model['model_id'], 'granite')
        self.assertEqual(run.call_args.args[0], COMMAND[:2] + [COMMAND[2] + ' --select-model granite'])
        self.assertEqual(run.call_args.kwargs['input'], '')
        self.assertNotIn('shell', run.call_args.kwargs)

    def test_unknown_or_shell_identifier_never_starts_privileged_process(self):
        with patch('pixelforge_chat.subprocess.run') as run:
            for ident in ('missing', '../granite', 'granite;id', 'granite extra', 'x'*65, '-granite'):
                with self.subTest(ident=ident), self.assertRaises(LocalError):
                    select_model(ident, catalog=self.catalog, metadata=self.metadata)
        run.assert_not_called()

    def test_selection_failure_and_unknown_outcome_never_claim_success_or_retry(self):
        with patch('pixelforge_chat.subprocess.run', return_value=subprocess.CompletedProcess(COMMAND, 1, '', 'busy')) as run:
            with self.assertRaisesRegex(LocalError, 'busy'):
                select_model('granite', catalog=self.catalog, metadata=self.metadata)
            self.assertEqual(run.call_count, 1)
        with patch('pixelforge_chat.subprocess.run', side_effect=subprocess.TimeoutExpired(COMMAND, 120)) as run:
            with self.assertRaisesRegex(LocalError, 'outcome is unknown'):
                select_model('granite', catalog=self.catalog, metadata=self.metadata)
            self.assertEqual(run.call_count, 1)
        self.publish('other')
        with patch('pixelforge_chat.subprocess.run', return_value=subprocess.CompletedProcess(COMMAND, 0, '', '')):
            with self.assertRaisesRegex(LocalError, 'not confirmed'):
                select_model('granite', catalog=self.catalog, metadata=self.metadata)

    def test_selected_context_and_output_cap_drive_actual_request_and_receipt(self):
        client = self.client(); host = Host()
        with patch('pixelforge_chat.subprocess.Popen', return_value=host):
            job = client.submit('owner', TASK, role=ROLE)
            result = client.read('owner', job['id'], wait=2)
        self.assertEqual(result['state'], 'completed', result)
        self.assertEqual(host.requests[0]['context_size'], 4096)
        self.assertEqual(host.requests[0]['max_tokens'], 64)
        self.assertEqual(client.status()['limits']['max_tokens'], 64)
        self.assertIsNone(result['usage']['provenance_repo'])
        with self.assertRaisesRegex(LocalError, 'between 2 and 64'):
            client.submit('owner', TASK, role=ROLE, max_tokens=65)

    def test_invalid_model_limits_make_installation_unavailable_without_host_start(self):
        for values in ({'context_per_worker': True}, {'context_per_worker': 255},
                       {'context_per_worker': 8193}, {'max_output_tokens': True},
                       {'max_output_tokens': 1}, {'max_output_tokens': 513},
                       {'context_per_worker': 256, 'max_output_tokens': 256}):
            self.metadata.write_text(json.dumps(dict(self.descriptor,
                model=dict(self.model, **values))))
            with self.subTest(values=values), patch('pixelforge_chat.subprocess.Popen') as spawn:
                self.assertFalse(self.client().status()['ready'])
                spawn.assert_not_called()

    def test_changed_descriptor_drops_old_host_before_next_request(self):
        client = self.client(); first, second = Host(), Host()
        with patch('pixelforge_chat.subprocess.Popen', side_effect=[first, second]) as spawn:
            job = client.submit('owner', TASK, role=ROLE)
            self.assertEqual(client.read('owner', job['id'], wait=2)['state'], 'completed')
            self.model['model_sha256'] = 'b' * 64
            self.publish('granite')
            self.assertFalse(client.status()['verified_worker_ready'])
            job = client.submit('owner', TASK, role=ROLE)
            self.assertEqual(client.read('owner', job['id'], wait=2)['state'], 'completed')
        self.assertTrue(first.stopped.is_set())
        self.assertEqual(spawn.call_count, 2)
        self.assertEqual((len(first.requests), len(second.requests)), (1, 1))

    def test_queued_stale_descriptor_is_rejected_without_starting_host(self):
        client = self.client()
        job = {'_cancel': threading.Event(), '_model': client._model()}
        self.model['model_sha256'] = 'b' * 64
        self.publish('granite')
        with patch('pixelforge_chat.subprocess.Popen') as spawn:
            with self.assertRaisesRegex(LocalError, 'selection changed'):
                client._ensure_host(job)
        spawn.assert_not_called()

    def test_prompt_tokens_must_fit_the_selected_context(self):
        client = self.client()
        host = Host(lambda h, r: h.stdout.push(done(r['id'], prompt_tokens=4097)))
        with patch('pixelforge_chat.subprocess.Popen', return_value=host):
            job = client.submit('owner', TASK, role=ROLE)
            result = client.read('owner', job['id'], wait=2)
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['result'], '')


if __name__ == '__main__':
    unittest.main()
