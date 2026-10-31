"""Generic GGUF profile, prompt-format and selected-limit regression fixtures."""
from contextlib import ExitStack
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'tools/arm_npu'))
import config
import service
import model_info
import gguf_hf_loader
import model_architecture


class GenericRenderingTests(unittest.TestCase):
    def test_actual_model_tokens_replace_invented_bos_and_eos_defaults(self):
        template = '{{ bos_token }}{{ messages[0].content }}{{ eos_token }}'
        render = service.renderer(template, {'bos_token': '<|begin_of_text|>', 'eos_token': '<|eot_id|>'},
                                  prompt_mode='chat-template')
        self.assertEqual(render('{{ 7*9 }}'), '<|begin_of_text|>{{ 7*9 }}<|eot_id|>')
        self.assertEqual(service.renderer(template)('hello'), 'hello')
        self.assertEqual(service.renderer(template, {'bos_token': '<s>', 'eos_token': '</s>'})('hello'),
                         '<s>hello</s>')

    def test_raw_completion_keeps_literal_prompt_and_does_not_render_it(self):
        prompt = '<user> {{ 7*9 }} </user>'
        self.assertEqual(service.renderer(None, prompt_mode='raw-completion')(prompt), prompt)
        self.assertEqual(service.renderer(None)(prompt), prompt)
        with self.assertRaisesRegex(ValueError, 'override'):
            service.renderer('{{ raise_exception("unused") }}', prompt_mode='raw-completion')

    def test_template_controls_and_raise_exception_remain_available(self):
        render = service.renderer('{{ enable_thinking }}|{{ add_generation_prompt }}|{{ tools|length }}')
        self.assertEqual(render('hello'), 'False|True|0')
        with self.assertRaisesRegex(ValueError, 'unsupported role'):
            service.renderer('{{ raise_exception("unsupported role") }}')('hello')
        with self.assertRaises(ValueError) as raised:
            service.renderer('{{ raise_exception("x" * 400) }}')('hello')
        self.assertLessEqual(len(str(raised.exception)), 300)

    def test_date_helper_is_available_without_exposing_python_internals(self):
        year = service.renderer('{{ strftime_now("%Y") }}')('hello')
        self.assertEqual(year, datetime.now().strftime('%Y'))
        from jinja2.exceptions import SecurityError
        for template in ('{{ strftime_now.__globals__["os"] }}', '{{ raise_exception.__globals__["os"] }}',
                         '{{ messages.__class__.__mro__ }}'):
            with self.subTest(template=template), self.assertRaises(SecurityError):
                service.renderer(template)('hello')

    def test_malformed_template_variables_and_prompt_modes_fail_before_rendering(self):
        from jinja2.exceptions import TemplateSyntaxError
        for value in ([], [('bos_token', '<s>')], 'bad', 1,
                      {'bos_token': 3}, {'eos_token': 'bad\x00token'},
                      {'messages': 'override'}, {3: 'not-a-variable-name'}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                service.renderer('{{ bos_token }}', value)
        for template, mode in (('', 'chat-template'), (None, 'chat-template'),
                               ('{{ bad', 'chat-template'), ('bad\x00template', 'chat-template'),
                               ('valid', 'not-a-prompt-mode')):
            with self.subTest(template=template, mode=mode), self.assertRaises((ValueError, TemplateSyntaxError)):
                service.renderer(template, prompt_mode=mode)


class SelectedModelLimitTests(unittest.TestCase):
    def setUp(self):
        self.profile = {'model_id': 'granite', 'architecture': 'granite',
                        'model': {'context_per_worker': 4096, 'max_output_tokens': 64,
                                  'provenance_repo': 'ibm-granite/fixture', 'prompt_mode': 'chat-template'}}
        self.request = {'id': 'job-1', 'prompt': 'A literal request'}

    def test_legacy_defaults_and_selected_defaults_are_distinct(self):
        self.assertEqual(config.model_limits(), (8192, 512))
        self.assertEqual(config.model_limits({}), (8192, 512))
        value = service.validate_request(self.request, self.profile)
        self.assertEqual((value['context_size'], value['max_tokens']), (4096, 64))
        self.assertEqual(set(value), {'id', 'prompt', 'max_tokens', 'context_size', 'prefill_cpu'})

    def test_selected_caps_bound_requests_and_preserve_configured_prefill(self):
        for change in ({'context_size': 4097}, {'context_size': True}, {'max_tokens': 65},
                       {'max_tokens': True}, {'context_size': 255}, {'prefill_cpu': False}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                service.validate_request(dict(self.request, **change), self.profile)
        value = service.validate_request(dict(self.request, context_size=256, max_tokens=32), self.profile)
        self.assertEqual((value['context_size'], value['max_tokens']), (256, 32))
        profile = dict(self.profile, cpu_prefill=False, weight_dtype='float32')
        self.assertFalse(service.validate_request(self.request, profile)['prefill_cpu'])
        with self.assertRaises(ValueError):
            service.validate_request(dict(self.request, prefill_cpu=True), profile)

    def test_output_plus_native_reserve_must_fit_the_requested_context(self):
        profile = {'model': {'context_per_worker': 1024, 'max_output_tokens': 512}}
        with self.assertRaisesRegex(ValueError, 'context'):
            service.validate_request(dict(self.request, context_size=256, max_tokens=249), profile)
        self.assertEqual(service.validate_request(dict(self.request, context_size=256, max_tokens=248), profile)['max_tokens'], 248)

    def test_invalid_profile_limit_types_ranges_and_shapes_fail_cleanly(self):
        for value in (None, [], 'model'):
            with self.subTest(model=value), self.assertRaises(ValueError):
                config.model_limits({'model': value})
        for change in ({'context_per_worker': True}, {'context_per_worker': 255},
                       {'context_per_worker': 8193}, {'max_output_tokens': True},
                       {'max_output_tokens': 1}, {'max_output_tokens': 513},
                       {'context_per_worker': 256, 'max_output_tokens': 249}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                config.model_limits({'model': dict(self.profile['model'], **change)})

    def test_inference_frames_cannot_choose_model_profile_architecture_or_limits(self):
        for key, value in (('model_id', 'other'), ('model_path', '/other.gguf'), ('model', {}),
                           ('profile', {}), ('architecture', 'other'), ('context_per_worker', 8192),
                           ('max_output_tokens', 512), ('prompt_mode', 'raw-completion')):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'Unexpected'):
                service.validate_request(dict(self.request, **{key: value}), self.profile)


class NativePromptModeTests(unittest.TestCase):
    def test_registered_mode_must_match_both_native_ready_and_done_evidence(self):
        for mode in ('chat-template', 'raw-completion'):
            profile = {'model': {'prompt_mode': mode}}
            service.validate_ready({'type': 'ready', 'prompt_mode': mode}, profile)
            service.validate_prompt_mode({'type': 'done', 'prompt_mode': mode}, profile)
            for frame in ({}, {'prompt_mode': None}, {'prompt_mode': 'wrong'},
                          {'prompt_mode': 'raw-completion' if mode == 'chat-template' else 'chat-template'}):
                with self.subTest(mode=mode, frame=frame):
                    with self.assertRaisesRegex(ValueError, 'prompt token policy'):
                        service.validate_ready(dict(frame, type='ready'), profile)
                    with self.assertRaisesRegex(ValueError, 'prompt token policy'):
                        service.validate_prompt_mode(dict(frame, type='done'), profile)

    def test_legacy_mode_absence_remains_compatible_but_user_cannot_override_it(self):
        service.validate_prompt_mode({}, {})
        service.validate_ready({}, {'model': {}})
        for mode in ('chat-template', 'raw-completion'):
            profile = {'model': {'prompt_mode': mode}}
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'Unexpected'):
                service.validate_request({'id': 'fixture', 'prompt': 'literal',
                                          'prompt_mode': mode}, profile)


class GenericProfileAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix='generic-service-')
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.model = self.root/'source.gguf'; self.model.write_bytes(b'GGUF generic source')
        self.helper = self.root/'helper'; self.helper.write_bytes(b'helper')
        self.manifest = self.root/'banks/MODEL.json'
        self.manifest.parent.mkdir(); self.manifest.write_text('{}')
        self.runtime = self.root/'runtime.py'; self.runtime.write_text('runtime')
        self.profile = dict(backend=service.BACKEND, model_id='granite', architecture='granite',
            model_path=str(self.model), artifacts=str(self.manifest.parent),
            native_helper_path=str(self.helper), native_helper_sha256=config.sha256(self.helper),
            runtime_sha256={'runtime.py': config.sha256(self.runtime)},
            artifact_manifests_sha256={'MODEL.json': config.sha256(self.manifest)},
            model={'model_sha256': config.sha256(self.model), 'provenance_repo': 'ibm-granite/fixture',
                   'context_per_worker': 4096, 'max_output_tokens': 64, 'prompt_mode': 'chat-template'})

    def read(self, profile=None):
        with patch.object(service, 'ROOT', self.root), patch.object(service, 'root_json', return_value=profile or self.profile):
            return service.read_profile()

    def test_any_publisher_and_non_hardcoded_architecture_can_pass_integrity_checks(self):
        self.assertEqual(self.read(), self.profile)
        for provenance in ('google/gemma', 'independent/model', None):
            profile = copy.deepcopy(self.profile)
            if provenance is None:
                profile['model'].pop('provenance_repo')
            else:
                profile['model']['provenance_repo'] = provenance
            self.assertEqual(self.read(profile), profile)
        # Acceptance here proves integrity only. main still checks the source
        # plan and the pinned native metadata probe before runtime startup.

    def test_generic_profile_still_requires_valid_backend_architecture_model_and_caps(self):
        for change in ({'backend': 'other'}, {'architecture': ''}, {'architecture': None},
                       {'model': []}, {'model': {'context_per_worker': 9000, 'model_sha256': 'a'*64}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read(dict(self.profile, **change))

    def test_generic_profile_does_not_relax_any_source_artifact_helper_or_runtime_pin(self):
        for path in (self.model, self.manifest, self.helper, self.runtime):
            original = path.read_bytes()
            path.write_bytes(original+b'changed')
            try:
                with self.subTest(path=path.name), self.assertRaisesRegex(ValueError, 'mismatch'):
                    self.read()
            finally:
                path.write_bytes(original)

    def test_generic_metadata_does_not_relax_native_ready_or_phase_evidence(self):
        names = ['blk.0.ffn_down.weight']
        profile = dict(self.profile, phase_evidence_version=1, cpu_prefill=False,
            weight_dtype='float32', batch_vectors=16, decode_batch_vectors=16,
            selected_npu_weight_tensors=names, selected_prefill_npu_weight_tensors=names)
        frame = dict(prompt_mode='chat-template', dense_npu_requested=True, prefill_cpu=False, prefill_batch_size=512,
            selected_npu_weight_tensors=names, selected_prefill_npu_weight_tensors=names,
            prefix_cache_mib=0, prefix_cache_hit=False, prefix_tokens_reused=0,
            prefix_cache_bytes=0, prefix_cache_entries=0, prefill_tokens_processed=17, prompt_tokens=17,
            npu_weight_matmul_operations=5, npu_prefill_weight_matmul_operations=3,
            npu_decode_weight_matmul_operations=2)
        service.validate_ready(frame, profile)
        config.validate_phase_evidence(frame, profile)
        for change in ({'dense_npu_requested': False}, {'selected_npu_weight_tensors': None},
                       {'selected_prefill_npu_weight_tensors': []}, {'prefill_batch_size': 16}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                service.validate_ready(dict(frame, **change), profile)
        with self.assertRaises(ValueError):
            config.validate_phase_evidence(dict(frame, npu_decode_weight_matmul_operations=0), profile)


class SelectedBundleGuardTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix='selected-bundle-')
        self.addCleanup(self.work.cleanup)
        self.state = Path(self.work.name)
        self.profile = {'state_dir': str(self.state), 'model_id': 'current',
                        'model': {'model_sha256': 'a'*64}}
        self.pin = 'b'*64
        self.bootstrap = {'service': str(service.ROOT/'service.py'),
                          'ubuntu_pins': {str(service.PROFILE): self.pin}}
        self.registry = {'schema': 'pixelforge.models.registry.v1', 'selected_model_id': 'current',
                         'models': {'current': {'bootstrap': self.bootstrap}}}

    def verify(self, *, registry=None, bootstrap=None, pin=None):
        values = {self.state/'models.json': self.registry if registry is None else registry,
                  self.state/'bootstrap.json': self.bootstrap if bootstrap is None else bootstrap}
        with patch.object(service, 'root_json', side_effect=lambda path: copy.deepcopy(values[Path(path)])), \
                patch.object(service, 'sha256', return_value=self.pin if pin is None else pin):
            return service.verify_selected_bundle(self.state, self.profile)

    def test_legacy_skips_registry_and_exact_selected_bundle_is_accepted(self):
        with patch.object(service, 'root_json') as read, patch.object(service, 'sha256') as digest:
            service.verify_selected_bundle(self.state, self.profile)
        read.assert_not_called(); digest.assert_not_called()
        (self.state/'models.json').write_text('{}')
        self.verify()

    def test_stale_selection_bootstrap_service_and_profile_pin_are_rejected(self):
        (self.state/'models.json').write_text('{}')
        stale_registry = copy.deepcopy(self.registry)
        stale_registry['selected_model_id'] = 'missing'
        with self.assertRaises(ValueError):
            self.verify(registry=stale_registry)
        with self.assertRaises(ValueError):
            self.verify(bootstrap=dict(self.bootstrap, service='/stale/service.py'))
        foreign = dict(self.bootstrap, service='/foreign/service.py')
        foreign_registry = copy.deepcopy(self.registry)
        foreign_registry['models']['current']['bootstrap'] = foreign
        with self.assertRaises(ValueError):
            self.verify(registry=foreign_registry, bootstrap=foreign)
        with self.assertRaises(ValueError):
            self.verify(pin='c'*64)
        missing_pin = dict(self.bootstrap, ubuntu_pins={})
        missing_registry = copy.deepcopy(self.registry)
        missing_registry['models']['current']['bootstrap'] = missing_pin
        with self.assertRaises(ValueError):
            self.verify(registry=missing_registry, bootstrap=missing_pin)

    def test_main_checks_selected_bundle_after_global_lock_before_loading_model_or_engine(self):
        events = []
        def locked(*args):
            events.append('lock')
        def reject(state, profile):
            self.assertEqual(events, ['lock'])
            self.assertEqual(state, self.state)
            self.assertIs(profile, self.profile)
            events.append('guard')
            raise ValueError('stale selected bundle')
        with ExitStack() as stack:
            stack.enter_context(patch.object(service.sys, 'argv', ['service.py']))
            stack.enter_context(patch.object(service.os, 'getuid', return_value=0))
            stack.enter_context(patch.object(service.os, 'umask'))
            stack.enter_context(patch.object(service, 'read_profile', return_value=self.profile))
            stack.enter_context(patch.object(service.Path, 'open', return_value=Mock()))
            stack.enter_context(patch.object(service.fcntl, 'flock', side_effect=locked))
            stack.enter_context(patch.object(service, 'verify_selected_bundle', side_effect=reject))
            load = stack.enter_context(patch.object(gguf_hf_loader, 'GGUFWeights'))
            spawn = stack.enter_context(patch.object(service.subprocess, 'Popen'))
            with self.assertRaisesRegex(ValueError, 'stale selected bundle'):
                service.main()
            load.assert_not_called(); spawn.assert_not_called()
        self.assertEqual(events, ['lock', 'guard'])


class StopBeforeRuntime(Exception):
    pass


class GenericMainPromptTests(unittest.TestCase):
    def exercise_main(self, *, raw=False, cached_probe=True, prompt_mode=None, supported=True):
        fields = {'general.architecture': 'granite', 'general.name': 'Generic fixture',
                  'granite.context_length': 4096}
        template = '{{ bos_token }}{{ messages[0].content }}{{ eos_token }}'
        if not raw:
            fields['tokenizer.chat_template'] = template
        probe = {'schema': model_info.PROBE_SCHEMA, 'architecture': 'granite', 'decoder_only': True,
                 'metadata_only': True, 'context_length': 4096, 'vocabulary_size': 100,
                 'supported_prompt_modes': ['chat-template', 'raw-completion'],
                 'bos_token': '<real-bos>', 'eos_token': '<real-eos>',
                 'chat_template': None if raw else template}
        profile = {'model_path': '/fixture/source.gguf', 'architecture': 'granite', 'state_dir': '/fixture',
                   'bank_rows': 4096, 'artifact_manifests_sha256': {'MODEL.json': 'pin'},
                   'model': {'model_sha256': 'a'*64, 'context_per_worker': 4096, 'max_output_tokens': 64,
                             'prompt_mode': prompt_mode or ('raw-completion' if raw else 'chat-template')}}
        if cached_probe:
            profile['native_model_probe'] = probe
        plan = {'architecture': 'granite', 'generation_adapter_supported': supported}
        weights = SimpleNamespace(fields=fields, path='/fixture/source.gguf')
        actual_renderer = service.renderer
        rendered = []
        def render(template, variables=None, *, prompt_mode=None):
            rendered.append(actual_renderer(template, variables, prompt_mode=prompt_mode)('{{ literal user }}'))
            raise StopBeforeRuntime()
        with ExitStack() as stack:
            stack.enter_context(patch.object(service.sys, 'argv', ['service.py']))
            stack.enter_context(patch.object(service.os, 'getuid', return_value=0))
            stack.enter_context(patch.object(service.os, 'umask'))
            stack.enter_context(patch.object(service, 'read_profile', return_value=profile))
            stack.enter_context(patch.object(service.Path, 'open', return_value=Mock()))
            stack.enter_context(patch.object(service.fcntl, 'flock'))
            stack.enter_context(patch.object(gguf_hf_loader, 'GGUFWeights', return_value=weights))
            stack.enter_context(patch.object(model_architecture, 'build_plan', return_value=plan))
            stack.enter_context(patch.object(service, 'preflight_artifacts', return_value={'MODEL.json': 'pin'}))
            native_probe = stack.enter_context(patch.object(model_info, 'probe_model', return_value=probe))
            stack.enter_context(patch.object(service, 'renderer', side_effect=render))
            spawn = stack.enter_context(patch.object(service.subprocess, 'Popen'))
            if not supported:
                with self.assertRaisesRegex(ValueError, 'unsupported'):
                    service.main()
            elif prompt_mode == 'wrong-mode':
                with self.assertRaisesRegex(ValueError, 'prompt format'):
                    service.main()
            else:
                with self.assertRaises(StopBeforeRuntime):
                    service.main()
            spawn.assert_not_called()
            if cached_probe or not supported:
                native_probe.assert_not_called()
            else:
                native_probe.assert_called_once_with(service.ROOT/'bin/pixelforge-chat', profile['model_path'])
        return rendered

    def test_main_uses_model_native_tokens_from_trusted_probe(self):
        self.assertEqual(self.exercise_main(), ['<real-bos>{{ literal user }}<real-eos>'])

    def test_legacy_main_probes_missing_metadata_before_rendering(self):
        self.assertEqual(self.exercise_main(cached_probe=False), ['<real-bos>{{ literal user }}<real-eos>'])

    def test_main_raw_mode_and_unsupported_or_mismatched_format_fail_before_runtime(self):
        self.assertEqual(self.exercise_main(raw=True), ['{{ literal user }}'])
        self.assertEqual(self.exercise_main(prompt_mode='wrong-mode'), [])
        self.assertEqual(self.exercise_main(supported=False), [])


if __name__ == '__main__':
    unittest.main()
