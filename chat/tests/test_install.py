"""Offline install manifests, configured bootstrap, and ownership boundaries."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'tools/arm_npu'))
import android_launcher
import config
import install
import lifecycle


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.home = self.root / 'home'
        self.runtime = self.home / 'voice/runtime'
        self.runtime.mkdir(parents=True)
        self.model = self.root / 'fixture.gguf'
        self.model.write_bytes(b'non-model fixture for manifest validation')
        self.helper = self.root / 'helper'
        self.helper.write_text('#!/bin/sh\nexit 1\n')
        self.helper.chmod(0o700)
        self.banks = self.root / 'banks'
        self.banks.mkdir()
        (self.banks / 'MODEL.json').write_text(json.dumps({'source_gguf': {'sha256': config.sha256(self.model)}, 'complete': False}))
        self.args = install.parser().parse_args([
            '--ubuntu-root', str(self.root), '--python', sys.executable, '--model', str(self.model),
            '--artifacts', str(self.banks), '--helper', str(self.helper), '--native-binary', str(self.helper),
            '--termux-home', str(self.home), '--app-runtime', str(self.runtime),
            '--model-name', 'Public test fixture', '--provenance-repo', 'huihui-ai/test-fixture',
            '--architecture', 'gemma4', '--select-tensors', 'blk.0.ffn_up.weight',
            '--service-dir', str(self.root / 'service'), '--state-dir', str(self.root / 'state')])
    def tearDown(self):
        self.temp.cleanup()
    def prepare(self):
        class Weights:
            fields = {'tokenizer.chat_template': '{{ messages[0].content }}'}
            source_identity = (1, 2, 3, 4, 5)
            from types import SimpleNamespace
            tensors = {'blk.0.ffn_gate.weight': SimpleNamespace(shape=(4,2)),
                       'blk.0.ffn_up.weight': SimpleNamespace(shape=(4,2)),
                       'blk.0.ffn_down.weight': SimpleNamespace(shape=(2,4))}
            def __init__(inner, _): pass
            def __enter__(inner): return inner
            def __exit__(inner, *_): pass
            def sha256(inner): return config.sha256(self.model)
        if self.args.dynamic_weight_mode in ('original_groups','original_columns'):
            Weights.tensors={name:SimpleNamespace(shape=(2,256),type_id=(14 if 'down' in name else 13)) for name in Weights.tensors}
        plan = {'generation_adapter_supported': True, 'architecture': self.args.architecture,
                'dense_tensors': [{'name': name, 'rows': tensor.shape[0], 'columns': tensor.shape[1], 'bank_rows': 2,
                                   'type_id':getattr(tensor,'type_id',0)}
                                  for name,tensor in Weights.tensors.items()]}
        class Cache:
            def __init__(inner, directory, *_args, **_kwargs): inner.directory = Path(directory)
            def get(inner, *_args, **_kwargs): return ({'compiled_bytes':12}, None)
            def _manifest(inner): return json.loads((inner.directory / 'MODEL.json').read_text())
        class Program(Cache):
            def __init__(inner, *args, **kwargs):
                super().__init__(*args, **kwargs); inner.plan = kwargs['plan']
            def layer(inner, name): return next(item for item in inner.plan['layers'] if item['gate']==name)
            def ranges(inner, name):
                step=inner.plan['hidden_chunk'];width=inner.layer(name)['intermediate']
                return [(start,min(start+step,width)) for start in range(0,width,step)]
        class Grouped(Cache):
            def __init__(inner,*args,**kwargs):
                super().__init__(*args,**kwargs);inner.plan=kwargs['plan'];inner.batch_vectors=kwargs['batch_vectors']
            def input_sizes(inner,name,start,stop):
                item=next(item for item in inner.plan['dense_tensors'] if item['name']==name)
                rows,columns=stop-start,item['columns'];kind=item['type_id'];group=32 if kind==13 else 16
                coefficient=rows*(columns//group)*4
                return [inner.batch_vectors*columns*4,rows*columns,coefficient]+([coefficient] if kind==13 else [])
        class Columns(Grouped):
            request_dtype='original_groups'
            def aliases_for(inner,name):return [name]
            def layer(inner,name):return next(item for item in inner.plan['dense_tensors'] if item['name']==name)
            def ranges(inner,name):
                item=inner.layer(name)
                return [(start,min(start+item['bank_rows'],item['rows'])) for start in range(0,item['rows'],item['bank_rows'])]
        with patch('gguf_hf_loader.GGUFWeights', Weights), patch('model_architecture.build_plan', return_value=plan), \
                patch('compiled_model.ArtifactCache', Cache), patch('full_ffn.FullFFNProgramCache', Program), \
                patch.dict(sys.modules,{'dynamic_fc':SimpleNamespace(DynamicProgramCache=Cache),
                                        'grouped_fc':SimpleNamespace(GroupedProgramCache=Grouped),
                                        'grouped_columns':SimpleNamespace(ColumnProgramCache=Columns)}):
            return install.prepare(self.args)
    def test_dry_run_lists_exact_writes_and_keeps_assets_in_place(self):
        before = sorted(str(path.relative_to(self.root)) for path in self.root.rglob('*'))
        manifest, sources, profile_bytes = self.prepare()
        self.assertFalse(manifest['apply'])
        self.assertFalse(manifest['model_copied'])
        self.assertFalse(manifest['compiled_banks_copied'])
        self.assertEqual(manifest['termux_uid'], self.home.stat().st_uid)
        self.assertEqual(manifest['profile']['native_helper_path'], str(self.helper))
        self.assertEqual(manifest['profile']['flash_attention'],'off')
        self.assertEqual(manifest['profile']['native_max_resident_models'],2)
        self.assertEqual(manifest['profile']['npu_performance_mode'],'default')
        self.assertIsNone(manifest['profile']['cpu_affinity'])
        self.assertEqual(manifest['profile']['prefix_cache_mib'],0)
        self.assertEqual(json.loads(profile_bytes)['selected_npu_weight_tensors'], ['blk.0.ffn_up.weight'])
        self.assertEqual(manifest['client_metadata']['launcher'], str(self.root / 'state/chat.py'))
        self.assertNotIn(str(self.model), manifest['writes'])
        self.assertNotIn(str(self.banks), manifest['writes'])
        self.assertEqual(before, sorted(str(path.relative_to(self.root)) for path in self.root.rglob('*')))
        self.assertIn('lifecycle.py', sources)
        self.assertIn('kit/gguf_broker.py', sources)
        self.assertIn('kit/gguf_vendor/quants.py', sources)
        self.assertNotIn('run.py', sources)
    def test_placement_invalid_or_incomplete_fails_before_writing(self):
        for selection in (None, '', 'unknown.weight', 'blk.0.ffn_up.weight,blk.0.ffn_up.weight'):
            self.args.select_tensors = selection
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                self.prepare()
        self.assertFalse((self.root / 'service').exists())
    def test_other_app_metadata_destination_is_refused(self):
        self.args.app_runtime = self.root / 'other-app'
        with self.assertRaisesRegex(ValueError, 'inside'):
            self.prepare()
    def test_npu_prefill_and_batched_decode_pin_only_required_sidecar(self):
        self.args.npu_prefill = True
        self.args.weight_dtype = 'float32'
        self.args.batch_vectors = self.args.decode_batch_vectors = 16
        self.args.select_prefill_tensors = 'blk.0.ffn_up.weight'
        self.args.native_runtime_cache_mib = 96
        sidecar = self.banks / 'batch-fc-v1/b16/MODEL.json'
        sidecar.parent.mkdir(parents=True)
        sidecar.write_text((self.banks / 'MODEL.json').read_text())
        (self.banks / 'MODEL.json').unlink()
        manifest, _, _ = self.prepare()
        profile = manifest['profile']
        self.assertFalse(profile['cpu_prefill'])
        self.assertEqual(profile['weight_dtype'], 'float32')
        self.assertEqual(profile['batch_vectors'], 16)
        self.assertEqual(profile['decode_batch_vectors'], 16)
        self.assertEqual(profile['prefill_batch_size'], 512)
        self.assertEqual(profile['native_runtime_cache_mib'], 96)
        self.assertEqual(profile['artifact_manifests_sha256'], {'batch-fc-v1/b16/MODEL.json': config.sha256(sidecar)})
        self.assertNotIn('artifacts_manifest_sha256', profile)
        metadata = manifest['client_metadata']['model']
        self.assertEqual(metadata['phase_evidence_version'], 1)
        self.assertEqual(metadata['selected_prefill_npu_weight_tensors'], ['blk.0.ffn_up.weight'])
        self.assertNotIn('CPU prompt processing', metadata['execution'])

    def test_invalid_phase_options_cannot_create_an_install(self):
        for key, value in (('batch_vectors', 16), ('native_runtime_cache_mib', 2049),
                           ('decode_batch_vectors', 16), ('prefill_batch_size', 0), ('fused_bank_rows', 0)):
            original = getattr(self.args, key)
            setattr(self.args, key, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare()
            setattr(self.args, key, original)
        self.assertFalse((self.root / 'service').exists())

    def test_cpu_flash_attention_is_a_trusted_opt_in_profile_setting(self):
        self.args.flash_attention='auto'
        manifest,_,_=self.prepare()
        self.assertEqual(manifest['profile']['flash_attention'],'auto')
        self.assertEqual(manifest['client_metadata']['model']['flash_attention'],'auto')
        self.assertIn('attention, norms and sampling on CPU',manifest['client_metadata']['model']['execution'])
        self.args.flash_attention='AUTO'
        with self.assertRaisesRegex(ValueError,'flash_attention'):
            self.prepare()

    def test_full_feed_forward_install_pins_program_chunks_without_ordinary_banks(self):
        self.args.architecture = 'qwen35'
        self.args.npu_prefill = self.args.full_ffn = True
        self.args.weight_dtype = 'float32'
        self.args.full_ffn_tensors = 'blk.0.ffn_gate.weight'
        self.args.full_ffn_phases = 'PD'
        self.args.full_ffn_hidden_chunk = 2
        self.args.batch_vectors = 16
        self.args.select_tensors = 'blk.0.ffn_gate.weight,blk.0.ffn_up.weight,blk.0.ffn_down.weight'
        sidecar=self.banks/'full-ffn-v1/b16/MODEL.json'
        sidecar.parent.mkdir(parents=True);sidecar.write_text('{}')
        (self.banks/'MODEL.json').unlink()
        manifest, _, _ = self.prepare()
        profile=manifest['profile'];model=manifest['client_metadata']['model']
        self.assertTrue(profile['full_ffn'])
        self.assertEqual(profile['full_ffn_phases'], 'PD')
        self.assertEqual(profile['full_ffn_tensors'], ['blk.0.ffn_gate.weight'])
        self.assertEqual(profile['artifact_manifests_sha256'], {'full-ffn-v1/b16/MODEL.json':config.sha256(sidecar)})
        self.assertTrue(model['full_ffn'])
        self.assertIn('feed-forward compute on NPU',model['execution'])

    def test_dynamic_weight_install_uses_source_pins_and_only_geometry_manifest(self):
        self.args.npu_prefill=True
        self.args.weight_dtype='float32'
        self.args.dynamic_weight_mode='float32'
        self.args.batch_vectors=self.args.decode_batch_vectors=16
        self.args.select_tensors=None
        sidecar=self.banks/'dynamic-fc-v1/b16/MODEL.json'
        sidecar.parent.mkdir(parents=True);sidecar.write_text('{}')
        (self.banks/'MODEL.json').unlink()
        manifest,_,_=self.prepare()
        profile=manifest['profile'];model=manifest['client_metadata']['model']
        self.assertEqual(profile['dynamic_weight_mode'],'float32')
        self.assertEqual(profile['model']['model_sha256'],config.sha256(self.model))
        self.assertEqual(profile['artifact_manifests_sha256'],{'dynamic-fc-v1/b16/MODEL.json':config.sha256(sidecar)})
        self.assertIsNone(model['selected_npu_weight_tensors'])
        self.assertEqual(model['dynamic_weight_mode'],'float32')
        self.assertIn('original GGUF weight decoding on CPU',model['execution'])

    def test_unverified_packed_and_mixed_dynamic_recipes_are_refused(self):
        self.args.dynamic_weight_mode='packed'
        with self.assertRaisesRegex(ValueError,'unvalidated'):
            self.prepare()
        self.args.dynamic_weight_mode='float32'
        self.args.weight_dtype='float32'
        self.args.batch_vectors=16
        with self.assertRaisesRegex(ValueError,'shared batch'):
            self.prepare()
        self.args.decode_batch_vectors=16
        self.args.native_runtime_cache_mib=0
        with self.assertRaisesRegex(ValueError,'positive device cache'):
            self.prepare()

    def test_original_group_install_pins_both_phase_geometries_and_separate_buffer_budget(self):
        self.args.dynamic_weight_mode='original_groups'
        self.args.weight_dtype='float32';self.args.npu_prefill=True
        self.args.batch_vectors=16;self.args.decode_batch_vectors=1
        self.args.weight_buffer_cache_mib=6144
        self.args.native_max_resident_models=9
        self.args.npu_performance_mode='sustained'
        self.args.cpu_affinity=[7,2,3,4,5,6]
        self.args.select_tensors='blk.0.ffn_gate.weight,blk.0.ffn_up.weight,blk.0.ffn_down.weight'
        for batch in (1,16):
            path=self.banks/f'grouped-fc-v1/b{batch}/MODEL.json'
            path.parent.mkdir(parents=True);path.write_text('{}')
        (self.banks/'MODEL.json').unlink()
        manifest,_,_=self.prepare();profile=manifest['profile'];metadata=manifest['client_metadata']['model']
        self.assertEqual(set(profile['artifact_manifests_sha256']),{'grouped-fc-v1/b1/MODEL.json','grouped-fc-v1/b16/MODEL.json'})
        self.assertEqual(profile['native_runtime_cache_mib'],128)
        self.assertEqual(profile['weight_buffer_cache_mib'],6144)
        self.assertEqual(profile['native_max_resident_models'],9)
        self.assertEqual(profile['npu_performance_mode'],'sustained')
        self.assertEqual(metadata['native_max_resident_models'],9)
        self.assertEqual(metadata['npu_performance_mode'],'sustained')
        self.assertEqual(profile['cpu_affinity'],[2,3,4,5,6,7])
        self.assertEqual(metadata['cpu_affinity'],[2,3,4,5,6,7])
        self.assertEqual(metadata['dynamic_weight_mode'],'original_groups')
        self.assertEqual(metadata['decode_batch_vectors'],1)
        self.assertIn('unpacked losslessly on CPU',metadata['execution'])
        self.assertIn('weight reconstruction on NPU',metadata['execution'])
        self.args.weight_buffer_cache_mib=6145
        with self.assertRaises(ValueError):self.prepare()

    def test_resident_model_limit_and_performance_preference_are_strict_install_options(self):
        args=install.parser().parse_args(['--native-max-resident-models','9','--npu-performance-mode','sustained',
            *sum((['--'+key,str(getattr(self.args,key.replace('-','_')))] for key in
                  ('ubuntu-root','python','model','artifacts','helper','termux-home','app-runtime',
                   'model-name','provenance-repo','architecture')),[])])
        self.assertEqual(args.native_max_resident_models,9)
        self.assertEqual(args.npu_performance_mode,'sustained')
        for key,value in (('native_max_resident_models',65),('native_max_resident_models',True),
                          ('npu_performance_mode','burst')):
            with self.subTest(key=key,value=value):
                old=getattr(self.args,key);setattr(self.args,key,value)
                try:
                    with self.assertRaisesRegex(ValueError,key):self.prepare()
                finally:setattr(self.args,key,old)

    def test_original_column_install_requires_separate_phase_manifests_and_discloses_cpu_sums(self):
        self.args.dynamic_weight_mode='original_columns';self.args.weight_dtype='float32';self.args.npu_prefill=True
        self.args.batch_vectors=64;self.args.decode_batch_vectors=1
        self.args.select_tensors='blk.0.ffn_gate.weight,blk.0.ffn_up.weight,blk.0.ffn_down.weight'
        for batch in (1,64):
            path=self.banks/f'grouped-columns-v1/b{batch}/MODEL.json'
            path.parent.mkdir(parents=True);path.write_text('{}')
        (self.banks/'MODEL.json').unlink()
        manifest,_,_=self.prepare();profile=manifest['profile'];model=manifest['client_metadata']['model']
        self.assertEqual(profile['dynamic_weight_mode'],'original_columns')
        self.assertEqual(set(profile['artifact_manifests_sha256']),{'grouped-columns-v1/b1/MODEL.json','grouped-columns-v1/b64/MODEL.json'})
        self.assertEqual(model['selected_npu_weight_tensors'],self.args.select_tensors.split(','))
        self.assertIn('column partial sums on CPU',model['execution'])
        self.assertIn('original GGUF unchanged',model['execution'])
        self.assertIn('no additional weight quantization',model['execution'])
        self.args.fused_gate_up=True
        with self.assertRaisesRegex(ValueError,'no constant fusion'):
            self.prepare()

    def test_cpu_affinity_csv_is_explicit_bounded_and_canonical(self):
        self.assertEqual(install.affinity_csv('7,2,3,4,5,6'),[2,3,4,5,6,7])
        for value in ('','1,1','256','-1','true','1,,2'):
            with self.subTest(value=value),self.assertRaises(install.argparse.ArgumentTypeError):
                install.affinity_csv(value)
        self.args.cpu_affinity=[]
        with self.assertRaisesRegex(ValueError,'cpu_affinity'):
            self.prepare()

    def test_prefix_cache_budget_is_installed_as_trusted_metadata_only(self):
        self.args.prefix_cache_mib=256
        manifest,_,_=self.prepare()
        self.assertEqual(manifest['profile']['prefix_cache_mib'],256)
        self.assertEqual(manifest['client_metadata']['model']['prefix_cache_mib'],256)
        for value in (-1,513,True):
            self.args.prefix_cache_mib=value
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'prefix_cache_mib'):
                self.prepare()

    def test_app_metadata_uses_trusted_home_selinux_categories_when_runtime_label_is_generic(self):
        # Exercise apply + atomic replacement without root or real SELinux writes.
        home_label=b'u:object_r:app_data_file:s0:c1,c2,c3,c4\0'
        generic_label=b'u:object_r:app_data_file:s0\0'
        labels={self.home:home_label,self.runtime:generic_label}
        original_stat=Path.stat
        def application_home_owner(path,*args,**kwargs):
            result=original_stat(path,*args,**kwargs)
            if path==self.home:
                fields=list(result);fields[4]=10001;fields[5]=10001
                return os.stat_result(fields)
            return result
        def read_label(path,name):
            self.assertEqual(name,'security.selinux')
            return labels[Path(path)]
        self.args.ubuntu_root=Path('/')
        manifest={'termux_uid':10001,'writes':[],
                  'client_metadata':{'backend':install.BACKEND},'bootstrap':{'fixture':True}}
        with patch.object(Path,'stat',application_home_owner),patch.object(install.os,'getuid',return_value=0), \
                patch.object(install.os,'chown') as chown, \
                patch.object(install.os,'getxattr',side_effect=read_label) as getlabel, \
                patch.object(install.os,'setxattr') as setlabel:
            install.apply(self.args,manifest,{},b'{}\n')
        getlabel.assert_called_once_with(self.home,'security.selinux')
        self.assertEqual(setlabel.call_count,1)
        self.assertEqual(setlabel.call_args.args[1:],('security.selinux',home_label))
        self.assertNotEqual(setlabel.call_args.args[2],generic_label)
        self.assertTrue(any(call.args[1:]==(10001,10001) for call in chown.call_args_list))
        metadata=self.runtime/'pixelforge-chat.json'
        self.assertEqual(json.loads(metadata.read_text())['backend'],install.BACKEND)
        self.assertEqual(metadata.stat().st_mode&0o777,0o600)

    @unittest.skipUnless(os.getuid() == 0, 'Root installer ownership fixture')
    def test_apply_writes_only_reviewed_outputs_without_asset_copy(self):
        # This isolated fixture never uses Android production paths or real weights.
        os.chown(self.home, 10001, 10001)
        os.chown(self.runtime, 10001, 10001)
        self.args.ubuntu_root = Path('/')
        self.args.apply = True
        manifest, sources, profile_bytes = self.prepare()
        install.apply(self.args, manifest, sources, profile_bytes)
        self.assertEqual((self.runtime / 'pixelforge-chat.json').stat().st_uid, 10001)
        generated = config.root_json(self.root / 'state/bootstrap.json')
        self.assertEqual(generated['service'], str(self.root / 'service/service.py'))
        for name, digest in manifest['source_pins'].items():
            self.assertEqual(config.sha256(self.root / 'service' / name), digest)
        self.assertEqual(self.model.read_bytes(), b'non-model fixture for manifest validation')
        self.assertFalse((self.root / 'service/model.gguf').exists())
        self.assertFalse((self.root / 'service/banks').exists())
        with self.assertRaisesRegex(ValueError, 'already exist'):
            install.apply(self.args, manifest, sources, profile_bytes)

    def test_root_json_rejects_symlink_or_writable_input(self):
        path = self.root / 'config.json'
        path.write_text('{}')
        path.chmod(0o600)
        self.assertEqual(config.root_json(path, owner=os.getuid()), {})
        link = self.root / 'alias'
        link.symlink_to(path)
        with self.assertRaises(OSError):
            config.root_json(link, owner=os.getuid())
        path.chmod(0o666)
        with self.assertRaises(ValueError):
            config.root_json(path, owner=os.getuid())


@unittest.skipUnless(os.getuid() == 0, 'Root-owned bootstrap fixtures')
class BootstrapTests(unittest.TestCase):
    def test_local_config_requires_service_and_interpreter_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bootstrap.json'
            data = {'ubuntu_root': '/example/rootfs', 'python': '/example/venv/bin/python',
                    'service': '/example/service.py',
                    'ubuntu_pins': {'/example/service.py': 'a' * 64, '/example/venv/bin/python': 'b' * 64}}
            path.write_text(json.dumps(data))
            path.chmod(0o600)
            self.assertEqual(android_launcher.read_bootstrap(path), data)
            del data['ubuntu_pins']['/example/venv/bin/python']
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                android_launcher.read_bootstrap(path)
    def test_cleanup_terminates_only_owned_identity(self):
        owned_process = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        try:
            record = lifecycle.process_record(owned_process.pid)
            lifecycle.cleanup(owned_process, {owned_process.pid: record['stamp']})
            self.assertIsNotNone(owned_process.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            for child in (owned_process, unrelated):
                if child.poll() is None:
                    child.kill()
                child.wait(timeout=2)


if __name__ == '__main__':
    unittest.main()
