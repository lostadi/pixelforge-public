import json
import os
from pathlib import Path
import sys
import tempfile
import selectors
import subprocess
import unittest
from unittest.mock import Mock, call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/arm_npu'))
import service
import lifecycle
import config
from types import SimpleNamespace


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

    def test_trusted_profile_fixes_prefill_and_request_cannot_override_it(self):
        profile = {'cpu_prefill': False, 'weight_dtype': 'float32', 'batch_vectors': 16}
        self.assertFalse(service.validate_request({'id': 'job-1', 'prompt': 'Hi'}, profile)['prefill_cpu'])
        self.assertFalse(service.validate_request({'id': 'job-1', 'prompt': 'Hi', 'prefill_cpu': False}, profile)['prefill_cpu'])
        for value in (True, 0, None, 'false'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                service.validate_request({'id': 'job-1', 'prompt': 'Hi', 'prefill_cpu': value}, profile)
        with self.assertRaises(ValueError):
            service.validate_request({'id':'job-1','prompt':'Hi','flash_attention':'on'},profile)

    def test_model_template_renders_user_text_as_data_and_disables_thinking(self):
        render = service.renderer('{{ bos_token }}{{ messages[0].content }}|{{ enable_thinking }}|{{ add_generation_prompt }}')
        self.assertEqual(render('{{ 7*9 }}'), '<bos>{{ 7*9 }}|False|True')

    def test_template_cannot_access_python_object_internals(self):
        from jinja2.exceptions import SecurityError
        with self.assertRaises(SecurityError):
            service.renderer('{{ messages.__class__.__mro__ }}')('Hi')


class BrokerRuntimeOptionTests(unittest.TestCase):
    def test_defaults_leave_resident_model_override_inactive_without_weight_buffers(self):
        options = config.runtime_options({})
        self.assertEqual(options['native_max_resident_models'], 2)
        self.assertEqual(options['npu_performance_mode'], 'default')
        broker = service.broker_runtime_options(options)
        self.assertNotIn('native_max_resident_models', broker)
        self.assertEqual(broker['npu_performance_mode'], 'default')
        options = config.runtime_options({'native_max_resident_models': 9})
        self.assertNotIn('native_max_resident_models', service.broker_runtime_options(options))

    def test_dma_profile_passes_exact_bounded_model_cap_and_sdk_preference(self):
        profile = dict(phase_evidence_version=1, dynamic_weight_mode='original_groups',
                       weight_dtype='float32', weight_buffer_cache_mib=6144,
                       native_max_resident_models=9, npu_performance_mode='sustained')
        broker = service.broker_runtime_options(config.runtime_options(profile))
        self.assertEqual(broker['native_max_resident_models'], 9)
        self.assertEqual(broker['npu_performance_mode'], 'sustained')
        self.assertEqual(broker['native_runtime_cache_mib'], 128)
        for count in (1, 64):
            self.assertEqual(config.runtime_options(dict(profile,native_max_resident_models=count))['native_max_resident_models'],count)
        for key, values in (('native_max_resident_models', (0,65,True,9.0,'9')),
                            ('npu_performance_mode', ('HIGH','burst',None,3))):
            for value in values:
                with self.subTest(key=key,value=value),self.assertRaisesRegex(ValueError,key):
                    config.runtime_options(dict(profile,**{key:value}))

    def test_inference_request_cannot_override_device_cache_or_sdk_policy(self):
        for key,value in (('native_max_resident_models',9),('npu_performance_mode','sustained')):
            with self.subTest(key=key),self.assertRaisesRegex(ValueError,'Unexpected'):
                service.validate_request(dict(id='fixture',prompt='Hi',**{key:value}))


class CPUAffinityTests(unittest.TestCase):
    def test_default_preserves_the_current_process_mask_without_scheduler_calls(self):
        self.assertIsNone(config.runtime_options({})['cpu_affinity'])
        with patch.object(service.os,'sched_getaffinity') as get,patch.object(service.os,'sched_setaffinity') as setmask:
            self.assertIsNone(service.apply_cpu_affinity(None))
        get.assert_not_called();setmask.assert_not_called()

    def test_allowed_subset_is_applied_only_to_self_and_verified(self):
        requested=[7,2,3,4,5,6]
        with patch.object(service.os,'sched_getaffinity',side_effect=[set(range(8)),set(requested)]) as get, \
                patch.object(service.os,'sched_setaffinity') as setmask:
            self.assertEqual(service.apply_cpu_affinity(requested),list(range(2,8)))
        self.assertEqual(get.call_args_list,[call(0),call(0)])
        setmask.assert_called_once_with(0,list(range(2,8)))

    def test_disallowed_mask_stops_startup_before_any_native_process_spawn(self):
        with patch.object(service.sys,'argv',['service.py']),patch.object(service.os,'getuid',return_value=0), \
                patch.object(service.os,'umask'), \
                patch.object(service,'read_profile',return_value={'cpu_affinity':[2,7]}), \
                patch.object(service.os,'sched_getaffinity',return_value={2,3}), \
                patch.object(service.os,'sched_setaffinity') as setmask,patch.object(service.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(ValueError,'outside the allowed'):
                service.main()
        setmask.assert_not_called();spawn.assert_not_called()

    def test_invalid_or_unverified_mask_is_rejected_and_ipc_cannot_override_it(self):
        for value in ([],[2,2],[-1],[256],[True],'2,3',[2.0]):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'cpu_affinity'):
                config.runtime_options({'cpu_affinity':value})
        with patch.object(service.os,'sched_getaffinity',side_effect=[{2,3},{2}]),patch.object(service.os,'sched_setaffinity'):
            with self.assertRaisesRegex(ValueError,'did not match'):
                service.apply_cpu_affinity([2,3])
        with self.assertRaisesRegex(ValueError,'Unexpected'):
            service.validate_request({'id':'fixture','prompt':'Hi','cpu_affinity':[2,3]})


class PhaseEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.profile = dict(cpu_prefill=False, weight_dtype='float32', batch_vectors=16,
            decode_batch_vectors=16, selected_npu_weight_tensors=['blk.0.ffn_down.weight'],
            selected_prefill_npu_weight_tensors=['blk.0.ffn_up.weight'], phase_evidence_version=1)
        self.frame = dict(prefill_cpu=False, npu_weight_matmul_operations=5,
            npu_prefill_weight_matmul_operations=3, npu_decode_weight_matmul_operations=2,
            selected_npu_weight_tensors=['blk.0.ffn_down.weight'],
            selected_prefill_npu_weight_tensors=['blk.0.ffn_up.weight'], prefill_batch_size=512,
            prompt_tokens=17,prefix_cache_mib=0,prefix_cache_hit=False,prefix_tokens_reused=0,
            prefix_cache_bytes=0,prefix_cache_entries=0,prefill_tokens_processed=17)
    def test_matching_npu_prefill_and_decode_are_accepted(self):
        config.validate_phase_evidence(self.frame, self.profile)
    def test_wrong_phase_mode_selection_or_counters_are_rejected(self):
        for change in ({'prefill_cpu': True}, {'npu_weight_matmul_operations': 0},
                       {'selected_prefill_npu_weight_tensors': ['blk.0.ffn_down.weight']},
                       {'npu_prefill_weight_matmul_operations': 0, 'npu_decode_weight_matmul_operations': 5},
                       {'npu_decode_weight_matmul_operations': True}, {'npu_decode_weight_matmul_operations': 3},
                       {'cpu_prefill_weight_matmul_operations': 1},
                       {'prefill_batch_size': 16}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                config.validate_phase_evidence(dict(self.frame, **change), self.profile)
    def test_cpu_prefill_requires_zero_prefill_npu_operations(self):
        self.profile['cpu_prefill'] = True
        frame = dict(self.frame, prefill_cpu=True, npu_prefill_weight_matmul_operations=0,
                     npu_decode_weight_matmul_operations=5)
        config.validate_phase_evidence(frame, self.profile)
        with self.assertRaises(ValueError):
            config.validate_phase_evidence(dict(self.frame, prefill_cpu=True), self.profile)
    def test_new_profile_rejects_missing_phase_fields_but_legacy_cpu_profile_works(self):
        legacy = dict(prefill_cpu=True, selected_npu_weight_tensors=None, npu_weight_matmul_operations=1)
        config.validate_phase_evidence(legacy, {})
        with self.assertRaises(ValueError):
            config.validate_phase_evidence(legacy, {'phase_evidence_version': 1})
    def test_native_ready_must_confirm_exact_phase_placements(self):
        ready = dict(self.frame, dense_npu_requested=True)
        service.validate_ready(ready, self.profile)
        with self.assertRaises(ValueError):
            service.validate_ready(dict(ready, selected_prefill_npu_weight_tensors=None), self.profile)

    def test_full_feed_forward_requires_the_hook_in_each_configured_phase(self):
        names=['blk.0.ffn_gate.weight','blk.0.ffn_up.weight','blk.0.ffn_down.weight']
        profile=dict(self.profile,full_ffn=True,full_ffn_tensors=[names[0]],full_ffn_phases='PD',
                     selected_npu_weight_tensors=names,selected_prefill_npu_weight_tensors=names)
        frame=dict(self.frame,selected_npu_weight_tensors=names,selected_prefill_npu_weight_tensors=names,
            npu_weight_matmul_operations=12,npu_prefill_weight_matmul_operations=9,npu_decode_weight_matmul_operations=3,
            generated_tokens=[1,2],npu_fused_ffn_operations=3,npu_prefill_fused_ffn_operations=2,
            npu_decode_fused_ffn_operations=1,full_ffn_tensors=[names[0]],full_ffn_phases='PD')
        config.validate_phase_evidence(frame,profile)
        for change in ({'npu_fused_ffn_operations':0}, {'npu_fused_ffn_operations':None},
                       {'npu_prefill_fused_ffn_operations':0,'npu_decode_fused_ffn_operations':3},
                       {'full_ffn_tensors':[]}, {'full_ffn_phases':'P'}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                config.validate_phase_evidence(dict(frame,**change),profile)
        with self.assertRaises(ValueError):
            service.validate_ready(dict(frame,dense_npu_requested=True,full_ffn_tensors=[]),profile)

    def test_dynamic_mode_must_match_and_all_dense_placement_cannot_report_cpu_weights(self):
        profile=dict(self.profile,dynamic_weight_mode='float32',selected_npu_weight_tensors=None,
                     selected_prefill_npu_weight_tensors=None)
        frame=dict(self.frame,dynamic_weight_mode='float32',selected_npu_weight_tensors=None,
                   selected_prefill_npu_weight_tensors=None,cpu_placed_weight_matmul_operations=0)
        config.validate_phase_evidence(frame,profile)
        for change in ({'dynamic_weight_mode':'off'},{'dynamic_weight_mode':'packed'},
                       {'cpu_placed_weight_matmul_operations':1}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                config.validate_phase_evidence(dict(frame,**change),profile)

    def test_cpu_flash_auto_accepts_resolved_fallback_without_claiming_npu_attention(self):
        profile=dict(self.profile,flash_attention='auto')
        for enabled in (True,False):
            frame=dict(self.frame,flash_attention='auto',cpu_flash_attention_enabled=enabled,attention_backend='CPU')
            config.validate_phase_evidence(frame,profile)
        for change in ({'flash_attention':'off'},{'cpu_flash_attention_enabled':None},
                       {'attention_backend':'NPU'},{'cpu_flash_attention_enabled':1}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                config.validate_phase_evidence(dict(frame,**change),profile)
        with self.assertRaises(ValueError):
            config.validate_phase_evidence(dict(frame,flash_attention='on'),dict(profile,flash_attention='on'))

    def test_ready_reports_flash_choice_without_pretending_context_is_resolved(self):
        profile=dict(self.profile,flash_attention='auto')
        frame=dict(self.frame,dense_npu_requested=True,flash_attention='auto',
                   cpu_flash_attention_enabled=None,attention_backend='CPU')
        service.validate_ready(frame,profile)
        with self.assertRaises(ValueError):
            service.validate_ready(dict(frame,cpu_flash_attention_enabled=True),profile)

    def test_completion_requires_the_mask_verified_by_the_root_service(self):
        profile=dict(self.profile,cpu_affinity=[2,3,4,5,6,7])
        config.validate_phase_evidence(dict(self.frame,cpu_affinity=[2,3,4,5,6,7]),profile)
        for mask in (None,[0,1],[],[True]):
            with self.subTest(mask=mask),self.assertRaises(ValueError):
                config.validate_phase_evidence(dict(self.frame,cpu_affinity=mask),profile)

    def test_prefix_cache_hit_keeps_actual_npu_evidence_and_exact_remaining_prefill_count(self):
        profile=dict(self.profile,prefix_cache_mib=1)
        frame=dict(self.frame,prefix_cache_mib=1,prefix_cache_hit=True,prefix_tokens_reused=12,
                   prefix_cache_bytes=123456,prefix_cache_entries=1,prefill_tokens_processed=5)
        config.validate_phase_evidence(frame,profile)
        values=config.validate_prefix_evidence(frame,profile)
        self.assertEqual(values['prefix_tokens_reused'],12)
        self.assertEqual(frame['npu_weight_matmul_operations'],5)
        for change in ({'prefix_cache_mib':0},{'prefix_cache_mib':True},{'prefix_cache_hit':False},
                       {'prefix_cache_hit':1},{'prefix_tokens_reused':17},{'prefix_tokens_reused':True},
                       {'prefix_cache_bytes':1024**2+1},{'prefix_cache_bytes':-1},
                       {'prefix_cache_entries':-1},{'prefix_cache_entries':True},
                       {'prefill_tokens_processed':4},{'prefill_tokens_processed':None}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                config.validate_phase_evidence(dict(frame,**change),profile)

    def test_disabled_prefix_cache_is_zero_and_legacy_missing_fields_are_compatible(self):
        fields=('prefix_cache_mib','prefix_cache_hit','prefix_tokens_reused','prefix_cache_bytes',
                'prefix_cache_entries','prefill_tokens_processed')
        legacy={key:value for key,value in self.frame.items() if key not in fields}
        self.assertEqual(config.validate_prefix_evidence(legacy,{}),{})
        with self.assertRaisesRegex(ValueError,'budget'):
            config.validate_prefix_evidence(legacy,self.profile)
        config.validate_prefix_evidence(self.frame,self.profile)
        with self.assertRaises(ValueError):
            config.validate_prefix_evidence(dict(self.frame,prefix_cache_entries=1),self.profile)
        with self.assertRaises(ValueError):
            config.validate_prefix_evidence(dict(self.frame,prefix_cache_hit=True,prefix_tokens_reused=1,
                                                prefill_tokens_processed=16),self.profile)

    def test_prefix_cache_budget_is_root_only_bounded_and_ready_attested(self):
        self.assertEqual(config.runtime_options({})['prefix_cache_mib'],0)
        self.assertEqual(config.runtime_options({'prefix_cache_mib':512})['prefix_cache_mib'],512)
        for value in (-1,513,True,1.0,'1'):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'prefix_cache_mib'):
                config.runtime_options({'prefix_cache_mib':value})
        ready=dict(self.frame,dense_npu_requested=True,prefix_cache_mib=256)
        service.validate_ready(ready,dict(self.profile,prefix_cache_mib=256))
        with self.assertRaisesRegex(ValueError,'budget'):
            service.validate_ready(dict(ready,prefix_cache_mib=0),dict(self.profile,prefix_cache_mib=256))
        self.assertNotIn('prefix_cache_mib',service.broker_runtime_options(config.runtime_options({'prefix_cache_mib':256})))
        with self.assertRaisesRegex(ValueError,'Unexpected'):
            service.validate_request({'id':'fixture','prompt':'Hi','prefix_cache_mib':256})


class ProfileManifestPinTests(unittest.TestCase):
    def test_sidecar_change_is_rejected_by_the_root_profile_before_runtime_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model, helper = root/'source.gguf', root/'native-helper'
            model.write_bytes(b'fixture model bytes'); helper.write_bytes(b'fixture helper bytes')
            manifest = root/'banks/batch-fc-v1/b16/MODEL.json'
            manifest.parent.mkdir(parents=True)
            manifest.write_text('{}')
            profile = dict(backend=service.BACKEND, architecture='qwen35', model_path=str(model),
                artifacts=str(root/'banks'), native_helper_path=str(helper), native_helper_sha256=config.sha256(helper),
                runtime_sha256={}, model={'model_sha256': config.sha256(model), 'provenance_repo':'huihui-ai/fixture'},
                artifact_manifests_sha256={'batch-fc-v1/b16/MODEL.json': config.sha256(manifest)},
                cpu_prefill=False, weight_dtype='float32', batch_vectors=16, decode_batch_vectors=16)
            with patch.object(service, 'root_json', return_value=profile):
                self.assertEqual(service.read_profile(), profile)
                manifest.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError, 'manifest SHA-256'):
                    service.read_profile()


class BankPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.planned = [dict(name='blk.0.ffn_up.weight', rows=4, columns=2, bank_rows=2),
                        dict(name='blk.0.ffn_gate.weight', rows=4, columns=2, bank_rows=2),
                        dict(name='blk.0.ffn_down.weight', rows=2, columns=4, bank_rows=2)]
        self.plan = {'dense_tensors': self.planned}
        self.weights = SimpleNamespace(source_identity=(1, 2, 3, 4, 5),
            tensors={item['name']: SimpleNamespace(shape=(item['rows'], item['columns']), type_id=0) for item in self.planned})
        self.profile = dict(artifacts=str(self.root), model_path=str(self.root/'weights.gguf'),
            model={'model_sha256': 'a'*64}, weight_dtype='float32', cpu_prefill=False,
            batch_vectors=16, selected_npu_weight_tensors=['blk.0.ffn_down.weight'],
            selected_prefill_npu_weight_tensors=['blk.0.ffn_up.weight'])
        self.calls = []
        outer = self
        class Cache:
            def __init__(inner, directory, *_args, **kwargs):
                inner.directory = Path(directory)
                outer.calls.append(('open', inner.directory, kwargs))
            def get(inner, *args, **kwargs):
                outer.calls.append(('get', inner.directory, args, kwargs))
                return ({'compiled_bytes':12}, None)
            def _manifest(inner): return {'complete': False}
        self.factory = Cache
    def sidecar(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
        return path
    def test_all_selected_prompt_and_decode_banks_are_verified_without_loading_code(self):
        ordinary = self.sidecar('MODEL.json')
        batch = self.sidecar('batch-fc-v1/b16/MODEL.json')
        pins = config.preflight_artifacts(self.profile, self.weights, self.plan, cache_factory=self.factory)
        self.assertEqual(pins, {'MODEL.json': config.sha256(ordinary), 'batch-fc-v1/b16/MODEL.json': config.sha256(batch)})
        gets = [item for item in self.calls if item[0]=='get']
        self.assertEqual(len(gets), 3)
        self.assertEqual({item[2][0] for item in gets}, {'blk.0.ffn_down.weight','blk.0.ffn_up.weight'})
        for item in gets:
            self.assertEqual(item[2][3], 'float32')
            self.assertEqual(item[3], {'load_code': False})
        self.assertTrue(all(item[2]['read_only'] for item in self.calls if item[0]=='open'))
    def test_batched_decode_reuses_sidecar_without_base_manifest(self):
        self.profile['decode_batch_vectors'] = 16
        self.sidecar('batch-fc-v1/b16/MODEL.json')
        pins = config.preflight_artifacts(self.profile, self.weights, self.plan, cache_factory=self.factory)
        self.assertEqual(list(pins), ['batch-fc-v1/b16/MODEL.json'])
    def test_missing_prompt_bank_is_fatal_before_any_runtime_can_start(self):
        self.sidecar('MODEL.json'); self.sidecar('batch-fc-v1/b16/MODEL.json')
        factory = self.factory
        class Missing(factory):
            def get(inner, name, *args, **kwargs):
                return None if name == 'blk.0.ffn_up.weight' else super().get(name, *args, **kwargs)
        with self.assertRaisesRegex(FileNotFoundError, 'ffn_up'):
            config.preflight_artifacts(self.profile, self.weights, self.plan, cache_factory=Missing)
    def test_fused_phase_routes_gate_up_and_down_to_pinned_fused_sidecar(self):
        self.profile.update(fused_gate_up=True, fused_bank_rows=8, decode_batch_vectors=16,
                            selected_prefill_npu_weight_tensors=[item['name'] for item in self.planned])
        self.sidecar('fused-ffn-v1/batch-fc-v1/b16/MODEL.json')
        pins = config.preflight_artifacts(self.profile, self.weights, self.plan, cache_factory=self.factory)
        self.assertEqual(list(pins), ['fused-ffn-v1/batch-fc-v1/b16/MODEL.json'])
        gets = [item[2] for item in self.calls if item[0]=='get']
        self.assertEqual({item[0] for item in gets}, {'blk.0.ffn_gate_up.weight','blk.0.ffn_down.weight'})
        self.assertIn(('blk.0.ffn_gate_up.weight', 0, 8, 'float32', 2), gets)

    def full_program_factory(self, missing=False):
        outer=self
        class Program:
            def __init__(inner,directory,*_args,**kwargs):
                inner.plan=kwargs['plan'];inner.directory=Path(directory)
                outer.calls.append(('program-open',inner.directory,kwargs))
            def layer(inner,name): return next(item for item in inner.plan['layers'] if item['gate']==name)
            def ranges(inner,name): return [(0,2),(2,4)]
            def get(inner,*args,**kwargs):
                outer.calls.append(('program-get',inner.directory,args,kwargs))
                return None if missing and args[1]==2 else ({},None)
        return Program
    def full_profile(self, phases='PD'):
        self.plan['architecture']='qwen35'
        self.profile.update(full_ffn=True,full_ffn_tensors=['blk.0.ffn_gate.weight'],full_ffn_phases=phases,
            full_ffn_hidden_chunk=2,phase_evidence_version=1,
            selected_prefill_npu_weight_tensors=[item['name'] for item in self.planned],
            selected_npu_weight_tensors=[item['name'] for item in self.planned])
        self.sidecar('full-ffn-v1/b16/MODEL.json')
    def test_full_program_checks_every_chunk_and_avoids_all_three_ordinary_bank_routes(self):
        self.full_profile()
        pins=config.preflight_artifacts(self.profile,self.weights,self.plan,cache_factory=self.factory,
                                        program_factory=self.full_program_factory())
        self.assertEqual(list(pins),['full-ffn-v1/b16/MODEL.json'])
        self.assertFalse(any(item[0]=='open' for item in self.calls))
        gets=[item for item in self.calls if item[0]=='program-get']
        self.assertEqual([item[2] for item in gets],[('blk.0.ffn_gate.weight',0,2,'float32',2),
                                                  ('blk.0.ffn_gate.weight',2,4,'float32',2)])
        self.assertTrue(all(item[3]=={'load_code':False} for item in gets))
    def test_full_prefill_program_keeps_ordinary_decode_banks_separate(self):
        self.full_profile('P');self.sidecar('MODEL.json')
        pins=config.preflight_artifacts(self.profile,self.weights,self.plan,cache_factory=self.factory,
                                        program_factory=self.full_program_factory())
        self.assertEqual(set(pins),{'MODEL.json','full-ffn-v1/b16/MODEL.json'})
        self.assertTrue(all(item[1]==self.root for item in self.calls if item[0]=='get'))
    def test_missing_full_program_chunk_and_wrong_activation_architecture_fail_closed(self):
        self.full_profile()
        with self.assertRaisesRegex(FileNotFoundError,'full FFN'):
            config.preflight_artifacts(self.profile,self.weights,self.plan,cache_factory=self.factory,
                                        program_factory=self.full_program_factory(missing=True))
        self.plan['architecture']='gemma4'
        with self.assertRaisesRegex(ValueError,'qwen35'):
            config.preflight_artifacts(self.profile,self.weights,self.plan,cache_factory=self.factory,
                                        program_factory=self.full_program_factory())
    def test_full_program_rejects_implicit_or_incomplete_three_weight_selection(self):
        self.full_profile()
        for selection in (None,['blk.0.ffn_gate.weight','blk.0.ffn_up.weight']):
            self.profile['selected_npu_weight_tensors']=selection
            with self.subTest(selection=selection),self.assertRaises(ValueError):
                config.preflight_artifacts(self.profile,self.weights,self.plan,cache_factory=self.factory,
                                            program_factory=self.full_program_factory())
    def test_real_full_program_cache_preflight_verifies_every_content_hash_without_hardware(self):
        import struct
        from test_gguf_loader import write_gguf
        from gguf_hf_loader import GGUFWeights
        from full_ffn import FullFFNProgramCache,build_full_ffn_plan
        self.full_profile()
        model=self.root/'tiny.gguf'
        write_gguf(model,[(item['name'],(item['rows'],item['columns']),0,
                          struct.pack('<8f',*range(8))) for item in self.planned])
        fingerprint={'fixture':'no-vendor-no-inference'}
        with GGUFWeights(model) as weights:
            digest=weights.sha256()
            self.profile.update(model_path=str(model),model={'model_sha256':digest})
            directory=self.root/'real-programs'
            self.profile['artifacts']=str(directory)
            cache=FullFFNProgramCache(directory/'full-ffn-v1/b16',model,source_sha256=digest,
                source_identity=weights.source_identity,verify_source=False,
                plan=build_full_ffn_plan(weights,digest,2),batch_vectors=16,fingerprint=fingerprint)
            code=struct.pack('<I4sI',8,b'DGC0',0)
            for start,stop in cache.ranges('blk.0.ffn_gate.weight'):
                record=cache.store('blk.0.ffn_gate.weight',start,stop,'float32',2,
                    {'graph_sha256':'a'*64,'graph_bytes':12,'compiled_bytes':len(code)},code)
            def factory(*args,**kwargs): return FullFFNProgramCache(*args,fingerprint=fingerprint,**kwargs)
            pins=config.preflight_artifacts(self.profile,weights,self.plan,program_factory=factory)
            self.assertEqual(set(pins),{'full-ffn-v1/b16/MODEL.json'})
            (cache.directory/record['bytecode_path']).write_bytes(code[:-1]+b'x')
            with self.assertRaisesRegex(ValueError,'bytecode hash'):
                config.preflight_artifacts(self.profile,weights,self.plan,program_factory=factory)

    def test_dynamic_all_projection_geometry_is_verified_without_constant_bank_fallback(self):
        self.profile.update(dynamic_weight_mode='float32',decode_batch_vectors=16,phase_evidence_version=1,
                            selected_npu_weight_tensors=None,selected_prefill_npu_weight_tensors=None)
        sidecar=self.sidecar('dynamic-fc-v1/b16/MODEL.json')
        forbidden=Mock(side_effect=AssertionError('No static constant package may be opened'))
        with patch.dict(sys.modules,{'dynamic_fc':SimpleNamespace(DynamicProgramCache=self.factory)}):
            pins=config.preflight_artifacts(self.profile,self.weights,self.plan,cache_factory=forbidden,
                                            dynamic_factory=self.factory)
        self.assertEqual(pins,{'dynamic-fc-v1/b16/MODEL.json':config.sha256(sidecar)})
        forbidden.assert_not_called()
        opens=[item for item in self.calls if item[0]=='open']
        self.assertEqual(len(opens),1)
        self.assertEqual(opens[0][2]['source_sha256'],'a'*64)
        self.assertEqual(opens[0][2]['source_identity'],self.weights.source_identity)
        self.assertTrue(opens[0][2]['read_only'])
        gets=[item for item in self.calls if item[0]=='get']
        self.assertEqual(len(gets),5)
        self.assertEqual({item[2][0] for item in gets},{item['name'] for item in self.planned})
        self.assertTrue(all(item[3]=={'load_code':False} for item in gets))

    def test_missing_dynamic_geometry_fails_before_runtime_and_never_compiles(self):
        self.profile.update(dynamic_weight_mode='float32',decode_batch_vectors=16,phase_evidence_version=1)
        self.sidecar('dynamic-fc-v1/b16/MODEL.json')
        factory=self.factory
        class Missing(factory):
            def get(inner,*_args,**_kwargs): return None
        with patch.dict(sys.modules,{'dynamic_fc':SimpleNamespace(DynamicProgramCache=Missing)}),self.assertRaisesRegex(FileNotFoundError,'geometry'):
            config.preflight_artifacts(self.profile,self.weights,self.plan,dynamic_factory=Missing)

    def test_dynamic_weight_bank_must_fit_the_bounded_sticky_device_cache(self):
        self.profile.update(dynamic_weight_mode='float32',decode_batch_vectors=16,phase_evidence_version=1,
                            native_runtime_cache_mib=1)
        self.sidecar('dynamic-fc-v1/b16/MODEL.json')
        factory=self.factory
        class TooLarge(factory):
            def get(inner,*_args,**_kwargs): return ({'compiled_bytes':2*1024**2},None)
        with patch.dict(sys.modules,{'dynamic_fc':SimpleNamespace(DynamicProgramCache=TooLarge)}),self.assertRaisesRegex(ValueError,'device cache budget'):
            config.preflight_artifacts(self.profile,self.weights,self.plan,dynamic_factory=TooLarge)

    def test_real_dynamic_geometry_cache_preflight_pins_source_and_rejects_code_tampering(self):
        import struct
        from test_gguf_loader import write_gguf
        from gguf_hf_loader import GGUFWeights
        from dynamic_fc import DynamicProgramCache
        self.profile.update(dynamic_weight_mode='float32',decode_batch_vectors=16,phase_evidence_version=1,
                            selected_npu_weight_tensors=None,selected_prefill_npu_weight_tensors=None)
        model=self.root/'geometry-source.gguf'
        write_gguf(model,[(item['name'],(item['rows'],item['columns']),0,
                          struct.pack('<8f',*range(8))) for item in self.planned])
        fingerprint={'fixture':'no-vendor-no-inference'}
        with GGUFWeights(model) as weights:
            digest=weights.sha256();plan=dict(self.plan,model_sha256=digest)
            self.profile.update(model_path=str(model),model={'model_sha256':digest})
            directory=self.root/'real-dynamic'
            self.profile['artifacts']=str(directory)
            cache=DynamicProgramCache(directory/'dynamic-fc-v1/b16',model,source_sha256=digest,
                source_identity=weights.source_identity,verify_source=False,plan=plan,
                batch_vectors=16,fingerprint=fingerprint)
            code=struct.pack('<I4sI',8,b'DGC0',0)
            for item in self.planned:
                for start,stop in cache.ranges(item['name']):
                    record=cache.store(item['name'],start,stop,'float32',item['columns'],
                        {'graph_sha256':'a'*64,'graph_bytes':12,'compiled_bytes':len(code)},code)
            self.assertEqual(len(cache._manifest()['artifacts']),2)
            def factory(*args,**kwargs): return DynamicProgramCache(*args,fingerprint=fingerprint,**kwargs)
            pins=config.preflight_artifacts(self.profile,weights,plan,dynamic_factory=factory)
            self.assertEqual(set(pins),{'dynamic-fc-v1/b16/MODEL.json'})
            (cache.directory/record['bytecode_path']).write_bytes(code[:-1]+b'x')
            with self.assertRaisesRegex(ValueError,'bytecode hash'):
                config.preflight_artifacts(self.profile,weights,plan,dynamic_factory=factory)

    def grouped_profile(self):
        for item in self.planned:
            item.update(rows=2,columns=256,bank_rows=1,type_id=14 if 'down' in item['name'] else 13)
        self.profile.update(dynamic_weight_mode='original_groups',decode_batch_vectors=1,
                            phase_evidence_version=1,weight_buffer_cache_mib=6144)
        for batch in (1,16):self.sidecar(f'grouped-fc-v1/b{batch}/MODEL.json')
        outer=self;factory=self.factory
        class Grouped(factory):
            def __init__(inner,*args,**kwargs):
                super().__init__(*args,**kwargs);inner.batch_vectors=kwargs['batch_vectors']
            def input_sizes(inner,name,start,stop):
                item=next(x for x in outer.planned if x['name']==name);kind=item['type_id']
                coeff=(stop-start)*(item['columns']//(32 if kind==13 else 16))*4
                return [inner.batch_vectors*item['columns']*4,(stop-start)*item['columns'],coeff]+([coeff] if kind==13 else [])
        return Grouped

    def test_original_groups_verify_both_phase_caches_with_original_kind_and_all_ports(self):
        grouped=self.grouped_profile()
        with patch.dict(sys.modules,{'grouped_fc':SimpleNamespace(GroupedProgramCache=grouped)}):
            pins=config.preflight_artifacts(self.profile,self.weights,self.plan,grouped_factory=grouped)
        self.assertEqual(set(pins),{'grouped-fc-v1/b1/MODEL.json','grouped-fc-v1/b16/MODEL.json'})
        gets=[item for item in self.calls if item[0]=='get']
        self.assertEqual(len(gets),6)
        self.assertTrue(all(item[2][3]=='original_groups' for item in gets))
        self.assertEqual({item[2][0] for item in gets if item[1].name=='b1'},
                         {'blk.0.ffn_up.weight','blk.0.ffn_down.weight'})
        self.assertEqual({item[2][0] for item in gets if item[1].name=='b16'},
                         {'blk.0.ffn_up.weight'})
        opens=[item for item in self.calls if item[0]=='open']
        self.assertEqual([item[2]['batch_vectors'] for item in opens],[1,16])
        self.assertEqual(config.runtime_options(self.profile)['native_runtime_cache_mib'],128)

    def test_original_group_prompt_only_weight_requires_b1_geometry_for_single_vector_prompt(self):
        grouped=self.grouped_profile()
        class MissingPromptB1(grouped):
            def get(inner,name,*args,**kwargs):
                if inner.directory.name=='b1' and name=='blk.0.ffn_up.weight':return None
                return super().get(name,*args,**kwargs)
        with patch.dict(sys.modules,{'grouped_fc':SimpleNamespace(GroupedProgramCache=MissingPromptB1)}), \
                self.assertRaisesRegex(FileNotFoundError,r'geometry is missing \(b1\): blk.0.ffn_up.weight'):
            config.preflight_artifacts(self.profile,self.weights,self.plan,grouped_factory=MissingPromptB1)

    def test_cpu_prefill_does_not_require_unused_prompt_b1_geometries(self):
        grouped=self.grouped_profile();self.profile['cpu_prefill']=True
        with patch.dict(sys.modules,{'grouped_fc':SimpleNamespace(GroupedProgramCache=grouped)}):
            pins=config.preflight_artifacts(self.profile,self.weights,self.plan,grouped_factory=grouped)
        self.assertEqual(set(pins),{'grouped-fc-v1/b1/MODEL.json'})
        self.assertEqual({item[2][0] for item in self.calls if item[0]=='get'}, {'blk.0.ffn_down.weight'})

    def column_profile(self):
        self.grouped_profile();self.profile.update(dynamic_weight_mode='original_columns',batch_vectors=64)
        self.planned[0]['columns']=4096;self.planned[-1]['columns']=9216
        for batch in (1,64):self.sidecar(f'grouped-columns-v1/b{batch}/MODEL.json')
        outer=self;factory=self.factory
        class Columns(factory):
            request_dtype='original_groups'
            def __init__(inner,*args,**kwargs):
                super().__init__(*args,**kwargs);inner.batch_vectors=kwargs['batch_vectors']
            def aliases_for(inner,name):
                item=next(item for item in outer.planned if item['name']==name)
                bounds={4096:[0,3072,4096],9216:[0,3072,6144,9216]}[item['columns']]
                return [f'{name}::cols[{start}:{stop}]' for start,stop in zip(bounds,bounds[1:])]
            def layer(inner,alias):
                name,bounds=alias.split('::cols[');start,stop=map(int,bounds[:-1].split(':'))
                return dict(next(item for item in outer.planned if item['name']==name),name=alias,
                            columns=3072,logical_columns=stop-start)
            def ranges(inner,alias):
                item=inner.layer(alias)
                return [(start,min(start+item['bank_rows'],item['rows'])) for start in range(0,item['rows'],item['bank_rows'])]
            def input_sizes(inner,alias,start,stop):
                item=inner.layer(alias);width=item['columns'];kind=item['type_id'];group=32 if kind==13 else 16
                coefficient=(stop-start)*(width//group)*4
                return [inner.batch_vectors*width*4,(stop-start)*width,coefficient]+([coefficient] if kind==13 else [])
        return Columns

    def test_original_column_preflight_expands_source_names_and_pins_b1_union_and_prompt_maximum(self):
        columns=self.column_profile()
        forbidden=Mock(side_effect=AssertionError('No ordinary or unsplit grouped fallback'))
        with patch.dict(sys.modules,{'grouped_columns':SimpleNamespace(ColumnProgramCache=columns)}):
            pins=config.preflight_artifacts(self.profile,self.weights,self.plan,column_factory=columns,
                                            grouped_factory=forbidden,cache_factory=forbidden)
        self.assertEqual(set(pins),{'grouped-columns-v1/b1/MODEL.json','grouped-columns-v1/b64/MODEL.json'})
        gets=[item for item in self.calls if item[0]=='get']
        self.assertEqual(len(gets),14)
        self.assertEqual({item[2][4] for item in gets},{3072})
        self.assertTrue(all(item[2][3]=='original_groups' and item[3]=={'load_code':False} for item in gets))
        self.assertEqual(len({item[2][0] for item in gets if item[1].name=='b1'}),5)
        self.assertEqual(len({item[2][0] for item in gets if item[1].name=='b64'}),2)
        for item in self.calls:
            if item[0]=='open':
                self.assertIs(item[2]['plan'],self.plan)
                self.assertTrue(item[2]['read_only'])
                self.assertEqual(item[2]['source_sha256'],'a'*64)
        forbidden.assert_not_called()

    def test_original_column_missing_prompt_only_b1_alias_is_fatal_and_internal_alias_is_not_a_placement(self):
        columns=self.column_profile()
        class Missing(columns):
            def get(inner,name,*args,**kwargs):
                if inner.directory.name=='b1' and name=='blk.0.ffn_up.weight::cols[3072:4096]':return None
                return super().get(name,*args,**kwargs)
        with patch.dict(sys.modules,{'grouped_columns':SimpleNamespace(ColumnProgramCache=Missing)}), \
                self.assertRaisesRegex(FileNotFoundError,r'geometry is missing \(b1\).*cols\[3072:4096\]'):
            config.preflight_artifacts(self.profile,self.weights,self.plan,column_factory=Missing)
        names=['blk.0.ffn_gate.weight','blk.0.ffn_up.weight','blk.0.ffn_down.weight']
        with self.assertRaisesRegex(ValueError,'no constant fusion'):
            config.runtime_options(dict(self.profile,full_ffn=True,full_ffn_tensors=[names[0]],full_ffn_phases='PD',
                selected_npu_weight_tensors=names,selected_prefill_npu_weight_tensors=names))
        self.profile['selected_npu_weight_tensors']=['blk.0.ffn_down.weight::cols[0:3072]']
        with self.assertRaisesRegex(ValueError,'NPU selection'):
            config.preflight_artifacts(self.profile,self.weights,self.plan,column_factory=columns)

    def test_original_column_preflight_charges_the_physical_padded_tail_ports(self):
        columns=self.column_profile();self.profile['native_runtime_cache_mib']=1
        class LargeTail(columns):
            def get(inner,name,*args,**kwargs):
                if name.endswith('::cols[3072:4096]'):return ({'compiled_bytes':400*1024},None)
                return super().get(name,*args,**kwargs)
        tail='blk.0.ffn_up.weight::cols[3072:4096]'
        cache=columns(self.root/'grouped-columns-v1/b64',batch_vectors=64)
        self.assertEqual(cache.layer(tail)['logical_columns'],1024)
        self.assertEqual(cache.layer(tail)['columns'],3072)
        self.assertEqual(cache.input_sizes(tail,0,1)[0],64*3072*4)
        with patch.dict(sys.modules,{'grouped_columns':SimpleNamespace(ColumnProgramCache=LargeTail)}), \
                self.assertRaisesRegex(ValueError,'device cache budget'):
            config.preflight_artifacts(self.profile,self.weights,self.plan,column_factory=LargeTail)

    def test_real_original_column_cache_keeps_source_plan_and_verifies_all_selected_tiles(self):
        import struct
        from test_gguf_loader import write_gguf
        from gguf_hf_loader import GGUFWeights
        from grouped_columns import ColumnProgramCache
        model=self.root/'column-source.gguf'
        write_gguf(model,[('blk.0.ffn_up.weight',(2,4096),13,bytes(2*4096//256*176)),
                          ('blk.0.ffn_down.weight',(2,9216),14,bytes(2*9216//256*210))])
        self.profile.update(dynamic_weight_mode='original_columns',batch_vectors=64,decode_batch_vectors=1,
            phase_evidence_version=1,selected_npu_weight_tensors=['blk.0.ffn_down.weight'],
            selected_prefill_npu_weight_tensors=['blk.0.ffn_up.weight'])
        fingerprint={'fixture':'no-vendor-no-inference'};code=struct.pack('<I4sI',8,b'DGC0',0)
        with GGUFWeights(model) as weights:
            digest=weights.sha256();self.profile.update(model_path=str(model),model={'model_sha256':digest})
            plan={'model_sha256':digest,'dense_tensors':[dict(name=name,rows=2,columns=t.shape[1],bank_rows=1,type_id=t.type_id)
                                                       for name,t in weights.tensors.items()]}
            for batch in (1,64):
                cache=ColumnProgramCache(self.root/f'grouped-columns-v1/b{batch}',model,source_sha256=digest,
                    source_identity=weights.source_identity,verify_source=False,plan=plan,
                    batch_vectors=batch,fingerprint=fingerprint)
                self.assertEqual(cache._manifest()['plan'],plan)
                for name in weights.tensors:
                    if batch==1 and name=='blk.0.ffn_up.weight':continue
                    for alias in cache.aliases_for(name):
                        layer=cache.layer(alias)
                        for start,stop in cache.ranges(alias):
                            cache.store(alias,start,stop,cache.request_dtype,layer['columns'],
                                {'graph_sha256':'a'*64,'graph_bytes':12,'compiled_bytes':len(code)},code)
            def factory(*args,**kwargs):return ColumnProgramCache(*args,fingerprint=fingerprint,**kwargs)
            with self.assertRaisesRegex(FileNotFoundError,r'geometry is missing \(b1\).*ffn_up.weight::cols\[0:3072\]'):
                config.preflight_artifacts(self.profile,weights,plan,column_factory=factory)
            single=ColumnProgramCache(self.root/'grouped-columns-v1/b1',model,source_sha256=digest,
                source_identity=weights.source_identity,verify_source=False,plan=plan,
                batch_vectors=1,fingerprint=fingerprint)
            for alias in single.aliases_for('blk.0.ffn_up.weight'):
                for start,stop in single.ranges(alias):
                    record=single.store(alias,start,stop,single.request_dtype,single.layer(alias)['columns'],
                        {'graph_sha256':'a'*64,'graph_bytes':12,'compiled_bytes':len(code)},code)
            pins=config.preflight_artifacts(self.profile,weights,plan,column_factory=factory)
            self.assertEqual(set(pins),{'grouped-columns-v1/b1/MODEL.json','grouped-columns-v1/b64/MODEL.json'})
            self.assertEqual(pins['grouped-columns-v1/b1/MODEL.json'],config.sha256(single.directory/'MODEL.json'))
            bytecode=single.directory/record['bytecode_path'];bytecode.write_bytes(code[:-1]+b'x')
            with self.assertRaisesRegex(ValueError,'bytecode hash'):
                config.preflight_artifacts(self.profile,weights,plan,column_factory=factory)
            bytecode.write_bytes(code)
            manifest=single.directory/'MODEL.json';changed=json.loads(manifest.read_text())
            changed['plan']['dense_tensors'][0]['name']='blk.0.ffn_up.weight::cols[0:3072]'
            manifest.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError,'architecture plan differs'):
                config.preflight_artifacts(self.profile,weights,plan,column_factory=factory)
            weights.check_source_unchanged(check_path=True)

    def test_original_groups_refuse_selected_f32_without_fallback(self):
        grouped=self.grouped_profile();self.planned[-1]['type_id']=0
        with patch.dict(sys.modules,{'grouped_fc':SimpleNamespace(GroupedProgramCache=grouped)}),self.assertRaisesRegex(ValueError,'Q5_K/Q6_K'):
            config.preflight_artifacts(self.profile,self.weights,self.plan,grouped_factory=grouped)

    def test_original_groups_require_each_selected_decode_bank_manifest(self):
        grouped=self.grouped_profile();(self.root/'grouped-fc-v1/b1/MODEL.json').unlink()
        with patch.dict(sys.modules,{'grouped_fc':SimpleNamespace(GroupedProgramCache=grouped)}),self.assertRaises(FileNotFoundError):
            config.preflight_artifacts(self.profile,self.weights,self.plan,grouped_factory=grouped)

    def test_real_original_group_cache_checks_q5_q6_dual_phase_code_integrity(self):
        import struct
        from test_gguf_loader import write_gguf
        from gguf_hf_loader import GGUFWeights
        from grouped_fc import GroupedProgramCache
        model=self.root/'original-quantized-source.gguf'
        write_gguf(model,[('blk.0.ffn_up.weight',(2,256),13,bytes(2*176)),
                          ('blk.0.ffn_down.weight',(2,256),14,bytes(2*210))])
        self.profile.update(dynamic_weight_mode='original_groups',decode_batch_vectors=1,phase_evidence_version=1,
            selected_npu_weight_tensors=['blk.0.ffn_down.weight'],selected_prefill_npu_weight_tensors=['blk.0.ffn_up.weight'])
        fingerprint={'fixture':'no-vendor-no-inference'};code=struct.pack('<I4sI',8,b'DGC0',0)
        with GGUFWeights(model) as weights:
            digest=weights.sha256();self.profile.update(model_path=str(model),model={'model_sha256':digest})
            plan={'model_sha256':digest,'dense_tensors':[dict(name=name,rows=2,columns=256,bank_rows=1,type_id=t.type_id)
                                                      for name,t in weights.tensors.items()]}
            for batch in (1,16):
                cache=GroupedProgramCache(self.root/f'grouped-fc-v1/b{batch}',model,source_sha256=digest,
                    source_identity=weights.source_identity,verify_source=False,plan=plan,
                    batch_vectors=batch,fingerprint=fingerprint)
                for name in weights.tensors:
                    if batch==1 and name=='blk.0.ffn_up.weight':continue
                    for start,stop in cache.ranges(name):
                        record=cache.store(name,start,stop,'original_groups',256,
                            {'graph_sha256':'a'*64,'graph_bytes':12,'compiled_bytes':len(code)},code)
            def factory(*args,**kwargs):return GroupedProgramCache(*args,fingerprint=fingerprint,**kwargs)
            with self.assertRaisesRegex(FileNotFoundError,r'geometry is missing \(b1\): blk.0.ffn_up.weight'):
                config.preflight_artifacts(self.profile,weights,plan,grouped_factory=factory)
            single=GroupedProgramCache(self.root/'grouped-fc-v1/b1',model,source_sha256=digest,
                source_identity=weights.source_identity,verify_source=False,plan=plan,
                batch_vectors=1,fingerprint=fingerprint)
            for start,stop in single.ranges('blk.0.ffn_up.weight'):
                single.store('blk.0.ffn_up.weight',start,stop,'original_groups',256,
                    {'graph_sha256':'a'*64,'graph_bytes':12,'compiled_bytes':len(code)},code)
            pins=config.preflight_artifacts(self.profile,weights,plan,grouped_factory=factory)
            self.assertEqual(set(pins),{'grouped-fc-v1/b1/MODEL.json','grouped-fc-v1/b16/MODEL.json'})
            self.assertEqual(pins['grouped-fc-v1/b1/MODEL.json'],config.sha256(single.directory/'MODEL.json'))
            (cache.directory/record['bytecode_path']).write_bytes(code[:-1]+b'x')
            with self.assertRaisesRegex(ValueError,'bytecode hash'):
                config.preflight_artifacts(self.profile,weights,plan,grouped_factory=factory)


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


class ChildDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.proc = Path(self.temp.name)
        self.records = {}
        self.root_patch = patch.object(lifecycle, 'PROC_ROOT', self.proc)
        self.root_patch.start()
        self.record_patch = patch.object(lifecycle, 'process_record', side_effect=lambda pid: self.records.get(pid))
        self.record_patch.start()
    def tearDown(self):
        self.record_patch.stop()
        self.root_patch.stop()
        self.temp.cleanup()
    def process(self, pid, parent, stamp, children=(), tid=None):
        self.records[pid] = {'pid': pid, 'parent': parent, 'stamp': stamp}
        task = self.proc / str(pid) / 'task' / str(tid or pid)
        task.mkdir(parents=True, exist_ok=True)
        (task / 'children').write_text(' '.join(map(str, children)))
        return task
    def test_fast_path_reads_only_owned_tasks_and_recurses(self):
        self.process(10, 1, 'root', [11], tid=101)
        self.process(11, 10, 'child', [12])
        self.process(12, 11, 'grandchild')
        owned = {10: 'root'}
        original = Path.iterdir
        def forbid_global(path):
            if path == self.proc:
                raise AssertionError('Periodic discovery must not scan unrelated processes')
            return original(path)
        with patch.object(Path, 'iterdir', forbid_global), patch.object(lifecycle, 'collect_owned') as full:
            self.assertTrue(lifecycle.collect_owned_children(owned))
        self.assertEqual(owned, {10: 'root', 11: 'child', 12: 'grandchild'})
        full.assert_not_called()
    def test_recycled_root_or_recorded_child_is_not_adopted(self):
        self.process(10, 1, 'new-root', [11])
        self.process(11, 10, 'new-child', [12])
        self.process(12, 11, 'grandchild')
        owned = {10: 'old-root'}
        self.assertTrue(lifecycle.collect_owned_children(owned))
        self.assertEqual(owned, {10: 'old-root'})
        owned = {10: 'new-root', 11: 'old-child'}
        self.assertTrue(lifecycle.collect_owned_children(owned))
        self.assertEqual(owned, {10: 'new-root', 11: 'old-child'})
    def test_tracker_backs_off_only_when_children_interface_is_unavailable(self):
        cleanup = Mock()
        cleanup.collect_owned_children.side_effect = [False, True]
        stopped = Mock()
        stopped.wait.side_effect = [False, False, True]
        owned = {10: 'root'}
        service.track_owned(cleanup, owned, stopped)
        self.assertEqual(stopped.wait.call_args_list, [call(.2), call(2.0), call(.2)])
        self.assertEqual(cleanup.collect_owned_children.call_args_list, [call(owned), call(owned)])
        cleanup.collect_owned.assert_not_called()

    def test_child_identity_change_during_validation_is_rejected(self):
        self.process(10, 1, 'root', [11])
        self.process(11, 10, 'old-child')
        reads = [0]
        def changing_record(pid):
            if pid == 11:
                reads[0] += 1
                return {'pid': 11, 'parent': 10, 'stamp': 'old-child' if reads[0] == 1 else 'replacement'}
            return self.records.get(pid)
        owned = {10: 'root'}
        with patch.object(lifecycle, 'process_record', side_effect=changing_record):
            self.assertTrue(lifecycle.collect_owned_children(owned))
        self.assertEqual(owned, {10: 'root'})
    def test_parent_identity_change_before_adoption_is_rejected(self):
        self.process(10, 1, 'root', [11])
        self.process(11, 10, 'child')
        reads = [0]
        def changing_record(pid):
            if pid == 10:
                reads[0] += 1
                return {'pid': 10, 'parent': 1, 'stamp': 'root' if reads[0] == 1 else 'replacement'}
            return self.records.get(pid)
        owned = {10: 'root'}
        with patch.object(lifecycle, 'process_record', side_effect=changing_record):
            self.assertTrue(lifecycle.collect_owned_children(owned))
        self.assertEqual(owned, {10: 'root'})
    def test_missing_children_interface_for_live_process_falls_back(self):
        task = self.process(10, 1, 'root')
        (task / 'children').unlink()
        owned = {10: 'root'}
        with patch.object(lifecycle, 'collect_owned') as full:
            self.assertFalse(lifecycle.collect_owned_children(owned))
        full.assert_called_once_with(owned)
    def test_unreadable_children_interface_falls_back(self):
        task = self.process(10, 1, 'root')
        original = Path.open
        def denied(path, *args, **kwargs):
            if path == task / 'children':
                raise PermissionError('Synthetic denied children interface')
            return original(path, *args, **kwargs)
        owned = {10: 'root'}
        with patch.object(Path, 'open', denied), patch.object(lifecycle, 'collect_owned') as full:
            self.assertFalse(lifecycle.collect_owned_children(owned))
        full.assert_called_once_with(owned)
    def test_child_of_unrelated_parent_is_not_adopted(self):
        self.process(10, 1, 'root', [11])
        self.process(11, 99, 'unrelated')
        owned = {10: 'root'}
        self.assertTrue(lifecycle.collect_owned_children(owned))
        self.assertEqual(owned, {10: 'root'})


@unittest.skipUnless(os.getuid() == 0 and Path('/proc/self/task').is_dir(), 'Linux root-owned process fixture')
class LiveChildDiscoveryTests(unittest.TestCase):
    def test_worker_thread_child_and_separate_session_grandchild_are_discovered(self):
        # This fixture spawns only sleeping synthetic processes, never inference.
        child_code = ("import json,os,subprocess,sys,time;os.setsid();"
                      "child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                      "print(json.dumps({'grandchild':child.pid}),flush=True);time.sleep(60)")
        parent_code = ("import json,os,subprocess,sys,threading\n"
                       "def spawn():\n"
                       " child=subprocess.Popen([sys.executable,'-c'," + repr(child_code) + "],stdout=subprocess.PIPE,text=True)\n"
                       " record=json.loads(child.stdout.readline())\n"
                       " print(json.dumps({'child':child.pid,'grandchild':record['grandchild'],'worker_tid':threading.get_native_id()}),flush=True)\n"
                       " child.wait()\n"
                       "worker=threading.Thread(target=spawn);worker.start();worker.join()\n")
        parent = subprocess.Popen([sys.executable, '-c', parent_code], stdout=subprocess.PIPE, text=True)
        record = lifecycle.process_record(parent.pid)
        owned = {parent.pid: record['stamp']}
        try:
            with selectors.DefaultSelector() as ready:
                ready.register(parent.stdout, selectors.EVENT_READ)
                self.assertTrue(ready.select(5), 'Synthetic process fixture did not start')
            descendants = json.loads(parent.stdout.readline())
            self.assertNotEqual(descendants['worker_tid'], parent.pid)
            main_children_path = Path('/proc', str(parent.pid), 'task', str(parent.pid), 'children')
            supported = main_children_path.exists()
            if supported:
                self.assertNotIn(str(descendants['child']), main_children_path.read_text().split())
            self.assertEqual(os.getpgid(descendants['child']), descendants['child'])
            with patch.object(lifecycle, 'collect_owned', wraps=lifecycle.collect_owned) as full:
                self.assertEqual(lifecycle.collect_owned_children(owned), supported)
            if supported:
                full.assert_not_called()
            else:
                full.assert_called_once_with(owned)
            for pid in (descendants['child'], descendants['grandchild']):
                self.assertEqual(owned[pid], lifecycle.process_record(pid)['stamp'])
        finally:
            # Final cleanup intentionally retains the original full discovery.
            # Cancel from the original root identity alone: the immediate full
            # collector must find both descendants without a periodic wait.
            with patch.object(lifecycle, 'collect_owned', wraps=lifecycle.collect_owned) as full:
                lifecycle.cleanup(parent, {parent.pid: record['stamp']})
            self.assertGreaterEqual(full.call_count, 2)
            parent.stdout.close()


if __name__ == '__main__':
    unittest.main()
