"""Offline PixelForge conversation transport and ordinary host-tool contracts."""
import io
import json
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Store
from conversation_protocol import encode_conversation
from local_chat import LocalChatAgent, create_agent
from local_runtime import LocalError
from pixelforge_chat import BACKEND, COMMAND, PixelForgeClient, conversation_prompt, public_response


ROLE = 'phone conversation assistant'
TASK = encode_conversation([{'role': 'user', 'content': 'Read probe.txt'}])
SELECTED = ['blk.0.ffn_gate.weight', 'blk.0.ffn_up.weight', 'blk.0.ffn_down.weight']


def done(ident, text='A completed answer', **overrides):
    return dict({'type': 'done', 'id': ident, 'text': text, 'generated_tokens': [5, 6],
                 'prompt_tokens': 500, 'npu_weight_matmul_operations': 42,
                 'cpu_dynamic_matmul_operations': 7, 'token_limit_reached': False,
                 'generation_seconds': 1000.0, 'prefill_cpu': True,
                 'prefix_cache_mib':0,'prefix_cache_hit':False,'prefix_tokens_reused':0,
                 'prefix_cache_bytes':0,'prefix_cache_entries':0,'prefill_tokens_processed':500}, **overrides)


class HostOutput:
    def __init__(self):
        self.frames = queue.Queue()
        self.closed = False
        self.push({'type': 'ready', 'model': {'backend': BACKEND},'prefix_cache_mib':0})

    def push(self, value):
        self.frames.put(json.dumps(value) + '\n')

    def readline(self, limit):
        return self.frames.get(timeout=5)

    def close(self):
        self.closed = True
        self.frames.put('')


class HostInput(io.StringIO):
    def __init__(self, host):
        super().__init__()
        self.host = host

    def write(self, text):
        super().write(text)
        request = json.loads(text)
        self.host.requests.append(request)
        self.host.received.set()
        self.host.respond(self.host, request)
        return len(text)

    def close(self):
        if not self.closed:
            self.host.stopped.set()
            self.host.stdout.close()
        # Preserve harmless requests for assertions.


class Host:
    def __init__(self, respond=None):
        self.requests = []
        self.received = threading.Event()
        self.stopped = threading.Event()
        self.stdout = HostOutput()
        self.stdin = HostInput(self)
        self.respond = respond or (lambda host, request: host.stdout.push(done(request['id'])))

    def poll(self):
        return 0 if self.stopped.is_set() else None

    def wait(self, timeout=None):
        if not self.stopped.wait(timeout):
            raise subprocess.TimeoutExpired('fixture-host', timeout)
        return 0

    def terminate(self):
        self.stdin.close()


class PixelForgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.metadata = self.root / 'installation.json'
        self.metadata.write_text(json.dumps({'backend': BACKEND, 'model': {
            'model_name': 'Huihui fixture', 'provenance_repo': 'huihui-ai/Huihui-fixture',
            'model_sha256': 'a' * 64}}))
        self.client = PixelForgeClient(metadata=self.metadata)
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.client.close)

    def reply(self, host, **options):
        with patch('pixelforge_chat.subprocess.Popen', return_value=host) as popen:
            job = self.client.submit('owner-a', TASK, role=ROLE, **options)
            result = self.client.read('owner-a', job['id'], wait=2, limit=32768)
        self.assertIn(result['state'], ('failed', 'completed'))
        return result, popen

    def test_status_is_passive_and_does_not_claim_verified_worker(self):
        with patch('pixelforge_chat.subprocess.Popen') as popen:
            status = self.client.status()
        self.assertTrue(status['ready'])
        self.assertFalse(status['running'])
        self.assertFalse(status['verified_worker_ready'])
        self.assertEqual(status['model']['backend'], BACKEND)
        self.assertEqual(status['model']['context_per_worker'], 8192)
        self.assertNotIn('native_conversation_protocol', status['model'])
        self.assertIn('CPU prompt processing', status['execution'])
        popen.assert_not_called()

    def test_initial_welcome_label_uses_passive_model_metadata(self):
        with patch('pixelforge_chat.subprocess.Popen') as popen:
            client = PixelForgeClient(metadata=self.metadata)
        self.addCleanup(client.close)
        self.assertEqual(client.engine_label, 'Huihui fixture · PixelForge NPU + CPU')
        popen.assert_not_called()
        missing = PixelForgeClient(metadata=self.root / 'missing.json')
        self.addCleanup(missing.close)
        self.assertEqual(missing.engine_label, 'Huihui · PixelForge NPU + CPU')
        self.assertFalse(missing.status()['ready'])

    def test_missing_or_non_huihui_installation_is_unavailable(self):
        self.metadata.write_text(json.dumps({'backend': BACKEND, 'model': {
            'model_name': 'Google fixture', 'provenance_repo': 'google/gemma', 'model_sha256': 'a' * 64}}))
        self.assertFalse(self.client.status()['ready'])
        with self.assertRaisesRegex(LocalError, 'installation'):
            self.client.submit('owner-a', TASK, role=ROLE)

    def test_helper_cannot_acquire_main_tools_and_transcript_is_required(self):
        for task, role in ((TASK, 'assistant'), ('plain text', ROLE)):
            with self.subTest(role=role), self.assertRaises((LocalError, ValueError)):
                self.client.submit('owner-a', task, role=role)
        with self.assertRaisesRegex(LocalError, 'between 2 and 512'):
            self.client.submit('owner-a', TASK, role=ROLE, max_tokens=1)

    def selected_metadata(self):
        data = json.loads(self.metadata.read_text())
        data['model']['selected_npu_weight_tensors'] = SELECTED
        self.metadata.write_text(json.dumps(data))

    def phase_metadata(self, cpu_prefill=False):
        data = json.loads(self.metadata.read_text())
        data['model'].update(cpu_prefill=cpu_prefill, weight_dtype='float32', batch_vectors=16,
            decode_batch_vectors=16, prefill_batch_size=512, phase_evidence_version=1,
            native_runtime_cache_mib=96, selected_npu_weight_tensors=SELECTED,
            selected_prefill_npu_weight_tensors=SELECTED[:2])
        self.metadata.write_text(json.dumps(data))
        return data['model']

    def phase_host(self, configured_model, **overrides):
        model = configured_model
        evidence = dict(prefill_cpu=model['cpu_prefill'], selected_npu_weight_tensors=SELECTED,
            selected_prefill_npu_weight_tensors=SELECTED[:2], prefill_batch_size=512,
            npu_prefill_weight_matmul_operations=0 if model['cpu_prefill'] else 40,
            npu_decode_weight_matmul_operations=42 if model['cpu_prefill'] else 2,
            cpu_placed_weight_matmul_operations=150,
            cpu_prefill_weight_matmul_operations=153 if model['cpu_prefill'] else 0, model=model)
        evidence.update(overrides)
        host = Host(lambda h, r: h.stdout.push(done(r['id'], **evidence)))
        host.stdout.frames.get_nowait()
        host.stdout.push({'type': 'ready', 'model': model,'prefix_cache_mib':model.get('prefix_cache_mib',0)})
        return host

    def test_npu_prefill_metadata_request_and_receipt_remain_consistent(self):
        model = self.phase_metadata()
        status = self.client.status()
        self.assertFalse(status['model']['cpu_prefill'])
        self.assertNotIn('CPU prompt processing', status['execution'])
        host = self.phase_host(model)
        result, _ = self.reply(host)
        self.assertEqual(result['state'], 'completed', result)
        self.assertFalse(host.requests[0]['prefill_cpu'])
        self.assertNotIn('timeout', host.requests[0])
        self.assertFalse(result['usage']['cpu_prefill'])
        self.assertEqual(result['usage']['npu_prefill_weight_matmul_operations'], 40)
        self.assertEqual(result['usage']['npu_decode_weight_matmul_operations'], 2)
        self.assertEqual(result['usage']['selected_prefill_npu_weight_tensors'], SELECTED[:2])
        self.assertEqual(result['usage']['weight_dtype'], 'float32')
        self.assertEqual(result['usage']['batch_vectors'], 16)

    def test_npu_prefill_refuses_wrong_phase_counters_selection_or_mode(self):
        model = self.phase_metadata()
        for change in ({'prefill_cpu': True},
                       {'selected_prefill_npu_weight_tensors': SELECTED},
                       {'selected_npu_weight_tensors': SELECTED[:1]},
                       {'npu_prefill_weight_matmul_operations': 0, 'npu_decode_weight_matmul_operations': 42},
                       {'npu_prefill_weight_matmul_operations': None},
                       {'npu_prefill_weight_matmul_operations': 42, 'npu_decode_weight_matmul_operations': 0},
                       {'cpu_prefill_weight_matmul_operations': 1},
                       {'npu_decode_weight_matmul_operations': 1}, {'prefill_batch_size': 16}):
            with self.subTest(change=change):
                result, _ = self.reply(self.phase_host(model, **change))
                self.assertEqual(result['state'], 'failed')
                self.assertEqual(result['result'], '')

    def test_strict_cpu_prefill_requires_zero_prefill_npu_and_keeps_mode(self):
        model = self.phase_metadata(cpu_prefill=True)
        host = self.phase_host(model)
        result, _ = self.reply(host)
        self.assertEqual(result['state'], 'completed', result)
        self.assertTrue(host.requests[0]['prefill_cpu'])
        self.client._disconnect(host)
        result, _ = self.reply(self.phase_host(model, npu_prefill_weight_matmul_operations=1,
                                               npu_decode_weight_matmul_operations=41))
        self.assertEqual(result['state'], 'failed')

    def test_host_readiness_cannot_change_new_profile_execution(self):
        model = self.phase_metadata()
        host = self.phase_host(model)
        host.stdout.frames.get_nowait()
        host.stdout.push({'type': 'ready', 'model': dict(model, cpu_prefill=True)})
        result, _ = self.reply(host)
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(host.requests, [])

    def test_strict_profile_retains_phase_timing_and_rejects_missing_model_identity(self):
        model = self.phase_metadata()
        host = self.phase_host(model, prefill_seconds=2.5, decode_seconds=1.5,
                               npu_bridge_seconds=2.0, npu_weight_matmul_vectors=64)
        result, _ = self.reply(host)
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['usage']['prefill_seconds'], 2.5)
        self.assertEqual(result['usage']['decode_seconds'], 1.5)
        self.assertEqual(result['usage']['npu_weight_matmul_vectors'], 64)
        self.client._disconnect(host)
        result, _ = self.reply(self.phase_host(model, model=None))
        self.assertEqual(result['state'], 'failed')

    def test_invalid_phase_metadata_is_unavailable_before_startup(self):
        self.phase_metadata()
        for key, value in (('weight_dtype', 'int8'), ('batch_vectors', True), ('decode_batch_vectors', 2),
                           ('native_runtime_cache_mib', 2049), ('prefill_batch_size', 0)):
            data = json.loads(self.metadata.read_text())
            old = data['model'][key]
            data['model'][key] = value
            self.metadata.write_text(json.dumps(data))
            with self.subTest(key=key), patch('pixelforge_chat.subprocess.Popen') as popen:
                self.assertFalse(self.client.status()['ready'])
                popen.assert_not_called()
            data['model'][key] = old
            self.metadata.write_text(json.dumps(data))

    def full_metadata(self):
        model=self.phase_metadata()
        model.update(full_ffn=True,full_ffn_tensors=['blk.0.ffn_gate.weight'],full_ffn_phases='PD',
                     full_ffn_hidden_chunk=2048,selected_prefill_npu_weight_tensors=SELECTED)
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        return model

    def full_host(self,model,**overrides):
        evidence=dict(selected_prefill_npu_weight_tensors=SELECTED,npu_weight_matmul_operations=6,
            npu_prefill_weight_matmul_operations=3,npu_decode_weight_matmul_operations=3,
            full_ffn_tensors=['blk.0.ffn_gate.weight'],full_ffn_phases='PD',npu_fused_ffn_operations=2,
            npu_prefill_fused_ffn_operations=1,npu_decode_fused_ffn_operations=1)
        evidence.update(overrides)
        return self.phase_host(model,**evidence)

    def test_configured_full_feed_forward_hook_requires_positive_phase_evidence(self):
        model=self.full_metadata()
        self.assertIn('feed-forward compute on NPU',self.client.status()['execution'])
        result,_=self.reply(self.full_host(model))
        self.assertEqual(result['state'],'completed',result)
        self.assertTrue(result['usage']['full_ffn'])
        self.assertEqual(result['usage']['npu_fused_ffn_operations'],2)
        self.assertEqual(result['usage']['npu_prefill_fused_ffn_operations'],1)
        self.assertEqual(result['usage']['npu_decode_fused_ffn_operations'],1)
        self.assertEqual(result['usage']['full_ffn_tensors'],['blk.0.ffn_gate.weight'])

    def test_full_feed_forward_rejects_missing_hook_wrong_program_and_phase_substitution(self):
        model=self.full_metadata()
        for change in ({'npu_fused_ffn_operations':0}, {'npu_fused_ffn_operations':None},
                       {'npu_prefill_fused_ffn_operations':2,'npu_decode_fused_ffn_operations':0},
                       {'full_ffn_tensors':[]}, {'full_ffn_phases':'P'}):
            with self.subTest(change=change):
                result,_=self.reply(self.full_host(model,**change))
                self.assertEqual(result['state'],'failed')
                self.assertEqual(result['result'],'')

    def test_full_feed_forward_metadata_requires_explicit_three_weight_placement(self):
        model=self.full_metadata()
        model['selected_prefill_npu_weight_tensors']=SELECTED[:2]
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        self.assertFalse(self.client.status()['ready'])

    def dynamic_metadata(self):
        model=self.phase_metadata()
        model.update(dynamic_weight_mode='float32',selected_npu_weight_tensors=None,
                     selected_prefill_npu_weight_tensors=None)
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        return model

    def dynamic_host(self,model,**overrides):
        evidence=dict(dynamic_weight_mode='float32',selected_npu_weight_tensors=None,
                      selected_prefill_npu_weight_tensors=None,cpu_placed_weight_matmul_operations=0)
        evidence.update(overrides)
        return self.phase_host(model,**evidence)

    def test_dynamic_float32_metadata_and_receipt_keep_original_source_identity(self):
        model=self.dynamic_metadata()
        self.assertIn('original GGUF weight decoding on CPU',self.client.status()['execution'])
        result,_=self.reply(self.dynamic_host(model))
        self.assertEqual(result['state'],'completed',result)
        self.assertEqual(result['usage']['dynamic_weight_mode'],'float32')
        self.assertEqual(result['usage']['model_sha256'],'a'*64)
        self.assertIsNone(result['usage']['selected_npu_weight_tensors'])
        self.assertEqual(result['usage']['cpu_placed_weight_matmul_operations'],0)

    def test_dynamic_mode_cannot_substitute_static_packed_or_cpu_weight_execution(self):
        model=self.dynamic_metadata()
        for change in ({'dynamic_weight_mode':'off'},{'dynamic_weight_mode':'packed'},
                       {'dynamic_weight_mode':None},{'cpu_placed_weight_matmul_operations':1}):
            with self.subTest(change=change):
                result,_=self.reply(self.dynamic_host(model,**change))
                self.assertEqual(result['state'],'failed')
                self.assertEqual(result['result'],'')

    def test_unverified_packed_metadata_is_refused_before_host_startup(self):
        model=self.dynamic_metadata();model['dynamic_weight_mode']='packed'
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        with patch('pixelforge_chat.subprocess.Popen') as popen:
            self.assertFalse(self.client.status()['ready'])
            popen.assert_not_called()

    def test_dynamic_weights_require_a_positive_sticky_device_cache_budget(self):
        model=self.dynamic_metadata();model['native_runtime_cache_mib']=0
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        self.assertFalse(self.client.status()['ready'])

    def flash_metadata(self):
        model=self.phase_metadata();model['flash_attention']='auto'
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        return model

    def flash_host(self,model,enabled=True,**overrides):
        evidence=dict(flash_attention='auto',attention_backend='CPU',cpu_flash_attention_enabled=enabled,
                      cpu_dynamic_matmul_operations=0)
        evidence.update(overrides)
        host=self.phase_host(model,**evidence)
        host.stdout.frames.get_nowait()
        host.stdout.push(dict(type='ready',model=model,flash_attention='auto',attention_backend='CPU',
                              cpu_flash_attention_enabled=None,prefix_cache_mib=model.get('prefix_cache_mib',0)))
        return host

    def test_cpu_flash_choice_and_resolved_fallback_are_honest_in_usage(self):
        model=self.flash_metadata()
        for enabled in (True,False):
            with self.subTest(enabled=enabled):
                host=self.flash_host(model,enabled)
                result,_=self.reply(host)
                self.assertEqual(result['state'],'completed',result)
                self.assertEqual(result['usage']['flash_attention'],'auto')
                self.assertEqual(result['usage']['cpu_flash_attention_enabled'],enabled)
                self.assertEqual(result['usage']['attention_backend'],'CPU')
                self.assertEqual(result['usage']['cpu_dynamic_matmul_operations'],0)
                self.assertIn('excludes other CPU operations',result['usage']['cpu_dynamic_matmul_operations_scope'])
                self.assertNotIn('flash_attention',host.requests[0])
                self.client._disconnect(host)

    def test_flash_setting_cannot_change_or_be_reported_as_npu_attention(self):
        model=self.flash_metadata()
        for change in ({'flash_attention':'off'},{'attention_backend':'NPU'},
                       {'cpu_flash_attention_enabled':None},{'cpu_flash_attention_enabled':1}):
            with self.subTest(change=change):
                result,_=self.reply(self.flash_host(model,**change))
                self.assertEqual(result['state'],'failed')
                self.assertEqual(result['result'],'')

    def grouped_metadata(self):
        model=self.phase_metadata()
        model.update(dynamic_weight_mode='original_groups',batch_vectors=16,decode_batch_vectors=1,
                     native_runtime_cache_mib=128,weight_buffer_cache_mib=6144)
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        return model

    def test_original_group_metadata_preserves_source_and_separate_weight_buffer_budget(self):
        model=self.grouped_metadata()
        host=self.phase_host(model,dynamic_weight_mode='original_groups')
        result,_=self.reply(host)
        self.assertEqual(result['state'],'completed',result)
        self.assertEqual(result['usage']['dynamic_weight_mode'],'original_groups')
        self.assertEqual(result['usage']['weight_buffer_cache_mib'],6144)
        self.assertEqual(result['usage']['native_runtime_cache_mib'],128)
        self.assertEqual(result['usage']['decode_batch_vectors'],1)
        self.assertEqual(result['usage']['model_sha256'],'a'*64)
        self.assertIn('unpacked losslessly on CPU',self.client.status()['execution'])
        self.assertIn('weight reconstruction on NPU',self.client.status()['execution'])

    def test_original_group_mode_and_buffer_bounds_cannot_be_substituted(self):
        model=self.grouped_metadata()
        result,_=self.reply(self.phase_host(model,dynamic_weight_mode='float32'))
        self.assertEqual(result['state'],'failed')
        model['weight_buffer_cache_mib']=6145
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        self.assertFalse(self.client.status()['ready'])

    def test_original_column_execution_is_opt_in_source_named_and_discloses_cpu_sums(self):
        model=self.grouped_metadata();model['dynamic_weight_mode']='original_columns';model['batch_vectors']=64
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        host=self.phase_host(model,dynamic_weight_mode='original_columns')
        result,_=self.reply(host)
        self.assertEqual(result['state'],'completed',result)
        self.assertEqual(result['usage']['dynamic_weight_mode'],'original_columns')
        self.assertEqual(result['usage']['batch_vectors'],64)
        self.assertEqual(result['usage']['selected_npu_weight_tensors'],SELECTED)
        execution=self.client.status()['execution']
        self.assertIn('column partial sums on CPU',execution)
        self.assertIn('original GGUF unchanged',execution)
        self.assertIn('no additional weight quantization',execution)
        self.assertNotIn('dynamic_weight_mode',host.requests[0])
        self.client._disconnect(host)
        result,_=self.reply(self.phase_host(model,dynamic_weight_mode='original_groups'))
        self.assertEqual(result['state'],'failed')
        self.assertEqual(result['result'],'')
        model['selected_npu_weight_tensors']=['blk.0.ffn_down.weight::cols[0:3072]']
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        self.assertFalse(self.client.status()['ready'])

    def test_trusted_model_cap_and_sdk_preference_survive_without_becoming_request_flags(self):
        model=self.grouped_metadata()
        model.update(native_max_resident_models=9,npu_performance_mode='sustained')
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        host=self.phase_host(model,dynamic_weight_mode='original_groups')
        result,_=self.reply(host)
        self.assertEqual(result['state'],'completed',result)
        self.assertEqual(result['usage']['native_max_resident_models'],9)
        self.assertEqual(result['usage']['npu_performance_mode'],'sustained')
        for key in ('native_max_resident_models','npu_performance_mode'):
            self.assertNotIn(key,host.requests[0])
        self.assertEqual(self.client.status()['model']['native_max_resident_models'],9)
        self.assertEqual(self.client.status()['model']['npu_performance_mode'],'sustained')

    def test_inactive_cap_is_disclosed_and_invalid_or_changed_runtime_policy_is_refused(self):
        model=self.phase_metadata()
        model['native_max_resident_models']=9
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        host=self.phase_host(model)
        result,_=self.reply(host)
        self.assertEqual(result['state'],'completed',result)
        self.assertIsNone(result['usage']['native_max_resident_models'])
        self.assertEqual(result['usage']['npu_performance_mode'],'default')
        self.client._disconnect(host)
        result,_=self.reply(self.phase_host(model,model=dict(model,npu_performance_mode='high')))
        self.assertEqual(result['state'],'failed')
        self.assertEqual(result['result'],'')
        for key,value in (('native_max_resident_models',0),('native_max_resident_models',65),
                          ('native_max_resident_models',True),('npu_performance_mode','burst')):
            with self.subTest(key=key,value=value):
                invalid=dict(model,**{key:value})
                self.metadata.write_text(json.dumps({'backend':BACKEND,'model':invalid}))
                self.assertFalse(self.client.status()['ready'])

    def test_selected_placement_status_and_actual_counters_are_preserved(self):
        self.selected_metadata()
        before = {'tpu_inference_count': '0 189012 52926'}
        after = {'tpu_inference_count': '0 189012 52929'}
        host = Host(lambda h, r: h.stdout.push(done(r['id'],
            selected_npu_weight_tensors=SELECTED, cpu_placed_weight_matmul_operations=150,
            cpu_prefill_weight_matmul_operations=153, report='/data/adb/ostadix-pixelforge/jobs/fixture',
            hardware_before=before, hardware_after=after)))
        status = self.client.status()
        self.assertIn('selected dense-weight projections on NPU', status['execution'])
        self.assertIn('remaining dense projections', status['execution'])
        self.assertEqual(status['model']['selected_npu_weight_tensors'], SELECTED)
        result, _ = self.reply(host)
        self.assertEqual(result['state'], 'completed')
        usage = result['usage']
        self.assertEqual(usage['selected_npu_weight_tensors'], SELECTED)
        self.assertEqual(usage['cpu_placed_weight_matmul_operations'], 150)
        self.assertEqual(usage['cpu_prefill_weight_matmul_operations'], 153)
        self.assertEqual(usage['hardware_before'], before)
        self.assertEqual(usage['hardware_after'], after)
        self.assertEqual(usage['report'], '/data/adb/ostadix-pixelforge/jobs/fixture')
        self.assertEqual(usage['provenance_repo'], 'huihui-ai/Huihui-fixture')

    def test_explicit_cpu_mask_requires_ready_and_completion_attestation(self):
        model=self.phase_metadata();mask=[2,3,4,5,6,7];model['cpu_affinity']=mask
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        def host_for(ready_mask,done_mask):
            host=self.phase_host(model,cpu_affinity=done_mask)
            host.stdout.frames.get_nowait()
            host.stdout.push({'type':'ready','model':model,'cpu_affinity':ready_mask,'prefix_cache_mib':model.get('prefix_cache_mib',0)})
            return host
        host=host_for(mask,mask);result,_=self.reply(host)
        self.assertEqual(result['state'],'completed',result)
        self.assertEqual(result['usage']['cpu_affinity'],mask)
        self.assertEqual(self.client.status()['model']['cpu_affinity'],mask)
        self.assertNotIn('cpu_affinity',host.requests[0])
        self.client._disconnect(host)
        for ready_mask,done_mask in ((None,mask),([0,1],mask),(mask,None),(mask,[0,1])):
            with self.subTest(ready_mask=ready_mask,done_mask=done_mask):
                host=host_for(ready_mask,done_mask);result,_=self.reply(host)
                self.assertEqual(result['state'],'failed')
                self.assertEqual(result['result'],'')
                if ready_mask!=mask:self.assertEqual(host.requests,[])
                self.client._disconnect(host)

    def test_invalid_cpu_mask_metadata_is_unavailable_before_host_start(self):
        model=self.phase_metadata()
        for mask in ([],[2,2],[True],[256],[-1],'2,3'):
            self.metadata.write_text(json.dumps({'backend':BACKEND,'model':dict(model,cpu_affinity=mask)}))
            with self.subTest(mask=mask),patch('pixelforge_chat.subprocess.Popen') as spawn:
                self.assertFalse(self.client.status()['ready'])
                spawn.assert_not_called()

    def test_prefix_cache_hit_usage_preserves_actual_npu_counts_and_trusted_budget(self):
        model=self.phase_metadata();model['prefix_cache_mib']=1
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        metrics=dict(prefix_cache_mib=1,prefix_cache_hit=True,prefix_tokens_reused=384,
                     prefix_cache_bytes=512000,prefix_cache_entries=1,prefill_tokens_processed=116)
        host=self.phase_host(model,**metrics);result,_=self.reply(host)
        self.assertEqual(result['state'],'completed',result)
        for key,value in metrics.items():self.assertEqual(result['usage'][key],value)
        self.assertEqual(result['usage']['npu_weight_matmul_operations'],42)
        self.assertEqual(result['usage']['npu_prefill_weight_matmul_operations'],40)
        self.assertNotIn('prefix_cache_mib',host.requests[0])
        self.client._disconnect(host)
        for change in ({'prefix_cache_mib':0},{'prefix_cache_hit':False},{'prefix_cache_hit':1},
                       {'prefix_tokens_reused':500},{'prefix_tokens_reused':True},
                       {'prefix_cache_bytes':1024**2+1},{'prefix_cache_entries':-1},
                       {'prefill_tokens_processed':117},{'prefill_tokens_processed':None}):
            with self.subTest(change=change):
                host=self.phase_host(model,**dict(metrics,**change));result,_=self.reply(host)
                self.assertEqual(result['state'],'failed')
                self.assertEqual(result['result'],'')
                self.client._disconnect(host)

    def test_prefix_cache_ready_budget_must_match_before_an_inference_request(self):
        model=self.phase_metadata();model['prefix_cache_mib']=1
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        host=self.phase_host(model);host.stdout.frames.get_nowait()
        host.stdout.push({'type':'ready','model':model,'prefix_cache_mib':0})
        result,_=self.reply(host)
        self.assertEqual(result['state'],'failed');self.assertEqual(host.requests,[])
        model['prefix_cache_mib']=513
        self.metadata.write_text(json.dumps({'backend':BACKEND,'model':model}))
        self.assertFalse(self.client.status()['ready'])

    def test_legacy_no_cache_profile_can_omit_prefix_fields_but_new_profile_cannot(self):
        keys={'prefix_cache_mib','prefix_cache_hit','prefix_tokens_reused','prefix_cache_bytes',
              'prefix_cache_entries','prefill_tokens_processed'}
        def reply_without_metrics(host,request):
            frame=done(request['id'])
            host.stdout.push({key:value for key,value in frame.items() if key not in keys})
        host=Host(reply_without_metrics);host.stdout.frames.get_nowait()
        host.stdout.push({'type':'ready','model':{'backend':BACKEND}})
        result,_=self.reply(host)
        self.assertEqual(result['state'],'completed',result)
        self.assertEqual(result['usage']['prefix_cache_mib'],0)
        self.assertFalse(result['usage']['prefix_cache_hit'])
        self.assertEqual(result['usage']['prefix_tokens_reused'],0)
        self.assertEqual(result['usage']['prefill_tokens_processed'],500)
        self.client._disconnect(host)
        model=self.phase_metadata();host=self.phase_host(model);normal_reply=host.respond
        def omit_new_profile_metrics(h,request):
            normal_reply(h,request)
            frame=json.loads(h.stdout.frames.get_nowait())
            h.stdout.push({key:value for key,value in frame.items() if key not in keys})
        host.respond=omit_new_profile_metrics
        result,_=self.reply(host)
        self.assertEqual(result['state'],'failed');self.assertEqual(result['result'],'')
        self.client._disconnect(host)
        model=self.phase_metadata();host=self.phase_host(model,prefix_cache_entries=1)
        result,_=self.reply(host)
        self.assertEqual(result['state'],'failed')
        self.assertEqual(result['result'],'')
    def test_selected_placement_requires_matching_selection_and_cpu_counters(self):
        self.selected_metadata()
        cases = [dict(selected_npu_weight_tensors=SELECTED),
                 dict(selected_npu_weight_tensors=SELECTED[:1], cpu_placed_weight_matmul_operations=1,
                      cpu_prefill_weight_matmul_operations=10),
                 dict(selected_npu_weight_tensors=SELECTED, cpu_placed_weight_matmul_operations=True,
                      cpu_prefill_weight_matmul_operations=10)]
        for override in cases:
            with self.subTest(override=override):
                host = Host(lambda h, r: h.stdout.push(done(r['id'], **override)))
                result, _ = self.reply(host)
                self.assertEqual(result['state'], 'failed')
                self.assertEqual(result['result'], '')

    def test_invalid_model_report_and_hardware_evidence_are_rejected(self):
        cases = [dict(model={'model_sha256': 'b' * 64}), dict(report='relative/report'),
                 dict(hardware_before='raw-string'), dict(hardware_after={'arbitrary_data': 'ignored'}),
                 dict(cpu_prefill_weight_matmul_operations=-1)]
        for override in cases:
            with self.subTest(override=override):
                host = Host(lambda h, r: h.stdout.push(done(r['id'], **override)))
                result, _ = self.reply(host)
                self.assertEqual(result['state'], 'failed')
                self.assertEqual(result['result'], '')

    def test_prompt_preserves_actual_receipts_without_provider_signatures(self):
        call = {'id': 'call-a', 'type': 'function', 'function': {'name': 'workspace',
                'arguments': '{"operation":"read","path":"probe.txt"}'},
                'extra_content': {'google': {'thought_signature': 'c2lnbmF0dXJl'}}}
        task = encode_conversation([{'role': 'user', 'content': 'Read probe.txt'},
            {'role': 'assistant', 'content': None, 'tool_calls': [call]},
            {'role': 'tool', 'tool_call_id': 'call-a', 'content': 'Actual receipt 42'}])
        prompt = conversation_prompt(task, ROLE, 'Use valid JSON.')
        self.assertIn('Actual receipt 42', prompt)
        self.assertIn('COMPLETED TOOL REQUEST call-a', prompt)
        self.assertIn('HOST FORMAT CORRECTION', prompt)
        self.assertNotIn('thought_signature', prompt)
        self.assertIn('thought_signature', task)

    def test_completed_request_has_real_npu_evidence_and_fixed_stdin_transport(self):
        host = Host()
        result, popen = self.reply(host)
        self.assertEqual(result['state'], 'completed')
        self.assertEqual(result['result'], 'A completed answer')
        self.assertEqual(result['usage']['npu_weight_matmul_operations'], 42)
        self.assertEqual(result['usage']['generation_seconds'], 1000.0)
        self.assertEqual(result['partial'], '')
        self.assertTrue(self.client.status()['verified_worker_ready'])
        self.assertEqual(popen.call_args.args[0], COMMAND)
        self.assertEqual(popen.call_args.kwargs['stderr'], subprocess.DEVNULL)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        request = host.requests[0]
        self.assertIn('Read probe.txt', request['prompt'])
        self.assertEqual(request['context_size'], 8192)
        self.assertTrue(request['prefill_cpu'])
        self.assertNotIn('timeout', request)

    def test_consecutive_turns_reuse_one_resident_host_and_owner_isolation(self):
        host = Host()
        with patch('pixelforge_chat.subprocess.Popen', return_value=host) as popen:
            for owner in ('owner-a', 'owner-b'):
                job = self.client.submit(owner, TASK, role=ROLE)
                result = self.client.read(owner, job['id'], wait=2)
                self.assertEqual(result['state'], 'completed')
                with self.assertRaises(LocalError):
                    self.client.read('intruder', job['id'])
        self.assertEqual(popen.call_count, 1)
        self.assertEqual(len(host.requests), 2)

    def test_zero_npu_ops_wrong_id_or_invalid_evidence_cannot_complete(self):
        cases = [dict(npu_weight_matmul_operations=0), dict(id='wrong-request'),
                 dict(prefill_cpu=False), dict(generation_seconds=float('nan')),
                 dict(generated_tokens=[True]), dict(prompt_tokens=9000)]
        for override in cases:
            with self.subTest(override=override):
                host = Host(lambda h, r: h.stdout.push(done(r['id'], **override)))
                result, _ = self.reply(host)
                self.assertEqual(result['state'], 'failed')
                self.assertEqual(result['result'], '')
                self.assertFalse(self.client.status()['verified_worker_ready'])

    def test_native_thought_or_tool_markers_never_become_public_answers(self):
        for text in ('<think>private</think>', '<|channel>analysis\nprivate',
                     '<|tool_call>workspace{path:"probe.txt"}'):
            with self.subTest(text=text), self.assertRaises(LocalError):
                public_response(text)
        self.assertEqual(public_response('<|channel>analysis\nprivate<|channel>final\nPublic answer<end_of_turn>'),
                         'Public answer')

    def test_token_limit_never_executes_partial_json_but_marks_plain_answer(self):
        host = Host(lambda h, r: h.stdout.push(done(r['id'], '{"tool":"workspace"', token_limit_reached=True)))
        result, _ = self.reply(host)
        self.assertEqual(result['state'], 'failed')
        host = Host(lambda h, r: h.stdout.push(done(r['id'], 'An incomplete explanation', token_limit_reached=True)))
        result, _ = self.reply(host)
        self.assertTrue(result['answer_incomplete'])
        self.assertIn('Answer incomplete', json.loads(result['result'])['final'])

    def test_streamed_raw_content_stays_private_and_cancel_closes_host_stdin(self):
        host = Host(lambda h, r: h.stdout.push({'type': 'chunk', 'id': r['id'], 'text': '<|channel>analysis private'}))
        with patch('pixelforge_chat.subprocess.Popen', return_value=host) as popen:
            job = self.client.submit('owner-a', TASK, role=ROLE)
            self.assertTrue(host.received.wait(1))
            snapshot = self.client.read('owner-a', job['id'])
            self.assertEqual(snapshot['partial'], '')
            self.client.cancel('owner-a', job['id'])
            self.assertTrue(host.stopped.is_set())
            result = self.client.read('owner-a', job['id'])
            self.assertEqual(result['state'], 'cancelled')
            self.assertEqual(result['result'], '')
            self.assertEqual(popen.call_count, 1)

    def test_cancelling_queued_owner_does_not_stop_other_active_owner(self):
        host = Host(lambda h, r: None)
        with patch('pixelforge_chat.subprocess.Popen', return_value=host):
            active = self.client.submit('owner-a', TASK, role=ROLE)
            self.assertTrue(host.received.wait(1))
            queued = self.client.submit('owner-b', TASK, role=ROLE)
            self.client.cancel('owner-b', queued['id'])
            self.assertFalse(host.stopped.is_set())
            self.assertEqual(self.client.read('owner-a', active['id'])['state'], 'running')
            self.client.cancel('owner-a', active['id'])

    def test_real_host_file_tool_result_returns_to_model_without_root_tool_execution(self):
        project = self.root / 'project'; project.mkdir()
        (project / 'probe.txt').write_text('Actual file result: BERYL-42')
        store = Store(self.root / 'state')
        sid = store.create(project)
        replies = iter(['{"tool":"workspace","arguments":{"operation":"read","path":"probe.txt"}}',
                        'Actual file result: BERYL-42'])
        host = Host(lambda h, r: h.stdout.push(done(r['id'], next(replies))))
        agent = LocalChatAgent(store, project, client=self.client)
        agent.local.cancel = Mock()
        try:
            with patch('pixelforge_chat.subprocess.Popen', return_value=host):
                agent.submit(sid, 'Read probe.txt and report its content.', 'fixture-request')
                agent.start_thread.join(4)
            self.assertFalse(agent.start_thread.is_alive())
            self.assertEqual(store.session(sid)['status'], 'idle')
            self.assertEqual(agent.last_answer, 'Actual file result: BERYL-42')
            self.assertEqual(len(host.requests), 2)
            self.assertIn('Actual file result: BERYL-42', host.requests[1]['prompt'])
            events = store.events(sid)
            self.assertTrue(any(e['kind'] == 'tool_result' and e['data']['success'] for e in events))
        finally:
            agent.close()
            store.db.close()

    def test_saved_pixelforge_engine_uses_the_new_client(self):
        project = self.root / 'project'; project.mkdir()
        store = Store(self.root / 'state')
        store.setting('chat_engine', 'pixelforge')
        with patch('pixelforge_chat.PixelForgeClient', return_value=self.client):
            agent = create_agent(store, project)
        try:
            self.assertIs(agent.local_client, self.client)
        finally:
            agent.close()
            store.db.close()


if __name__ == '__main__':
    unittest.main()
