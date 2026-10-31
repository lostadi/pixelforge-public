"""Compiled-package discovery fixtures; no model execution or vendor runtime."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compiled_package as package


class CompiledPackageTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory(prefix='compiled-package-')
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.model = self.root / 'ibm-granite.gguf'
        self.model.write_bytes(b'GGUF synthetic non-Huihui source binding')
        self.source_hash = hashlib.sha256(self.model.read_bytes()).hexdigest()
        self.base = self.root / 'artifacts'

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def fixture(self, family='static', batch=1, *, dtypes=None, base=None, model=None):
        base = base or self.base
        model = model or self.model
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        if family == 'static':
            leaf = base if batch == 1 else base / 'batch-fc-v1' / ('b'+str(batch))
            fmt, recipe = package.FORMAT, package.RECIPE if batch == 1 else package.BATCH_RECIPE
        elif family == 'fused-ffn-v1':
            leaf = base / family
            if batch > 1:
                leaf = leaf / 'batch-fc-v1' / ('b'+str(batch))
            fmt, recipe = package.FORMAT, package.FUSED_RECIPE
        else:
            leaf = base / family / ('b'+str(batch))
            fmt, recipe = {
                'dynamic-fc-v1': (package.DYNAMIC_FC_FORMAT, package.DYNAMIC_FC_RECIPE),
                'grouped-fc-v1': (package.GROUPED_FC_FORMAT, package.GROUPED_FC_RECIPE),
                'grouped-columns-v1': (package.GROUPED_COLUMNS_FORMAT, package.GROUPED_COLUMNS_RECIPE),
                'full-ffn-v1': (package.FULL_FFN_FORMAT, package.FULL_FFN_RECIPE),
            }[family]
        plan = {'model_sha256': digest, 'dense_tensors': [
            {'name': 'a.weight', 'rows': 10, 'columns': 4, 'bank_rows': 4},
            {'name': 'small.weight', 'rows': 2, 'columns': 4, 'bank_rows': 2}]}
        if family == 'full-ffn-v1':
            plan = {'model_sha256': digest, 'hidden_chunk': 2048,
                    'layers': [{'gate': 'blk.0.ffn_gate.weight', 'hidden': 4, 'intermediate': 8}]}
        if dtypes is None:
            dtypes = ('float32-runtime-input',) if family == 'dynamic-fc-v1' else (
                ('original_groups',) if family in ('grouped-fc-v1', 'grouped-columns-v1') else ('float32',))
        manifest = {'format': fmt, 'recipe': recipe, 'batch_vectors': batch,
                    'source_gguf': {'path': str(model), 'embedded': False, 'sha256': digest,
                                    'bytes': model.stat().st_size},
                    'plan': plan, 'complete': True, 'artifacts': {}}
        for index, dtype in enumerate(dtypes):
            code = b'0000DGC0' + (family+dtype).encode()
            code_hash = hashlib.sha256(code).hexdigest()
            key = hashlib.sha256(('artifact-'+str(index)).encode()).hexdigest()
            record = {'recipe': recipe, 'weight_dtype': dtype, 'batch_vectors': batch,
                      'bytecode_path': 'graphs/'+code_hash+'.dgc', 'bytecode_sha256': code_hash,
                      'compiled_bytes': len(code)}
            if family not in ('dynamic-fc-v1', 'grouped-fc-v1', 'grouped-columns-v1'):
                record['source_gguf_sha256'] = digest
            raw = json.dumps(record).encode()
            record_hash = hashlib.sha256(raw).hexdigest()
            relative = 'graphs/'+key+'-'+record_hash+'.json'
            (leaf/'graphs').mkdir(parents=True, exist_ok=True)
            (leaf/record['bytecode_path']).write_bytes(code)
            (leaf/relative).write_bytes(raw)
            manifest['artifacts'][key] = {'metadata': relative, 'metadata_sha256': record_hash}
        self.write_json(leaf/'MODEL.json', manifest)
        return leaf

    def manifest(self, leaf):
        return json.loads((leaf/'MODEL.json').read_text())

    def update_manifest(self, leaf, **fields):
        value = self.manifest(leaf)
        value.update(fields)
        self.write_json(leaf/'MODEL.json', value)

    def change_record(self, leaf, **fields):
        value = self.manifest(leaf)
        key, entry = next(iter(value['artifacts'].items()))
        record = json.loads((leaf/entry['metadata']).read_text())
        record.update(fields)
        raw = json.dumps(record).encode()
        digest = hashlib.sha256(raw).hexdigest()
        relative = 'graphs/'+key+'-'+digest+'.json'
        (leaf/relative).write_bytes(raw)
        value['artifacts'][key] = {'metadata': relative, 'metadata_sha256': digest}
        self.write_json(leaf/'MODEL.json', value)

    def test_generic_source_and_static_float32_binding_have_no_publisher_gate_or_cpu_selection(self):
        leaf = self.fixture()
        value = package.resolve_package(leaf/'MODEL.json')
        self.assertEqual(value['model'], self.model)
        self.assertEqual(value['source_sha256'], self.source_hash)
        self.assertEqual(value['artifacts'], self.base)
        self.assertEqual((value['bank_rows'], value['weight_dtype']), (4, 'float32'))
        self.assertEqual((value['batch_vectors'], value['decode_batch_vectors']), (1, 1))
        self.assertEqual(value['dynamic_weight_mode'], 'off')
        self.assertTrue(value['npu_prefill'])
        self.assertFalse(any('select' in key for key in value))
        self.assertEqual(package.resolve_package(leaf), value)

    def test_static_int8_is_distinguished_from_decoded_float32(self):
        leaf = self.fixture(dtypes=('int8',))
        self.assertEqual(package.resolve_package(leaf)['weight_dtype'], 'int8')
        with self.assertRaisesRegex(ValueError, 'dtype'):
            package.resolve_package(leaf, weight_dtype='float32')

    def test_mixed_static_dtypes_require_explicit_choice_from_actual_records(self):
        leaf = self.fixture(dtypes=('int8', 'float32'))
        with self.assertRaisesRegex(ValueError, 'multiple weight dtypes'):
            package.resolve_package(leaf)
        self.assertEqual(package.resolve_package(leaf, weight_dtype='float32')['weight_dtype'], 'float32')
        with self.assertRaisesRegex(ValueError, 'dtype'):
            package.resolve_package(leaf, weight_dtype='q6_k')

    def test_dynamic_families_infer_runtime_float32_and_b64_only_decodes_b64(self):
        for family, mode in (('dynamic-fc-v1', 'float32'), ('grouped-fc-v1', 'original_groups'),
                             ('grouped-columns-v1', 'original_columns')):
            with self.subTest(family=family):
                leaf = self.fixture(family, 64)
                value = package.resolve_package(leaf)
                self.assertEqual(value['artifacts'], self.base)
                self.assertEqual((value['dynamic_weight_mode'], value['weight_dtype']), (mode, 'float32'))
                self.assertEqual((value['batch_vectors'], value['decode_batch_vectors']), (64, 64))
                if family == 'dynamic-fc-v1':
                    with self.assertRaisesRegex(ValueError, 'same prefill and decode batch'):
                        package.resolve_package(leaf, decode_batch_vectors=1)
                else:
                    with self.assertRaisesRegex(FileNotFoundError, 'B1'):
                        package.resolve_package(leaf, decode_batch_vectors=1)

    def test_matching_b1_sibling_selects_b64_prefill_and_b1_decode(self):
        for family in ('static', 'grouped-fc-v1', 'grouped-columns-v1'):
            with self.subTest(family=family):
                base = self.root / family
                self.fixture(family, 1, base=base)
                leaf = self.fixture(family, 64, base=base)
                value = package.resolve_package(leaf)
                self.assertEqual((value['batch_vectors'], value['decode_batch_vectors']), (64, 1))
                self.assertEqual(package.resolve_package(leaf, decode_batch_vectors=64)['decode_batch_vectors'], 64)

    def test_float32_dynamic_keeps_one_batch_even_when_a_matching_b1_sibling_exists(self):
        self.fixture('dynamic-fc-v1', 1)
        leaf = self.fixture('dynamic-fc-v1', 64)
        value = package.resolve_package(leaf)
        self.assertEqual((value['batch_vectors'], value['decode_batch_vectors']), (64, 64))
        self.assertEqual(package.resolve_package(leaf, decode_batch_vectors=64)['decode_batch_vectors'], 64)
        with self.assertRaisesRegex(ValueError, 'same prefill and decode batch'):
            package.resolve_package(leaf, decode_batch_vectors=1)

    def test_multiple_leafs_require_explicit_selection_and_root_manifest_is_preferred(self):
        self.fixture('grouped-columns-v1', 1)
        selected = self.fixture('grouped-columns-v1', 64)
        with self.assertRaisesRegex(ValueError, 'explicit leaf'):
            package.resolve_package(self.base)
        self.assertEqual(package.resolve_package(selected)['batch_vectors'], 64)
        self.fixture('static', 1)
        self.assertEqual(package.resolve_package(self.base)['batch_vectors'], 1)

    def test_single_supported_nested_leaf_can_be_discovered(self):
        self.fixture('dynamic-fc-v1', 16)
        self.assertEqual(package.resolve_package(self.base)['batch_vectors'], 16)
        self.assertEqual(package.resolve_package(self.base/'dynamic-fc-v1')['batch_vectors'], 16)

    def test_b1_with_different_source_is_not_a_compatible_default_or_explicit_decode(self):
        other = self.root/'other.gguf'; other.write_bytes(b'GGUF other source')
        self.fixture('grouped-fc-v1', 1, model=other)
        leaf = self.fixture('grouped-fc-v1', 64)
        self.assertEqual(package.resolve_package(leaf)['decode_batch_vectors'], 64)
        with self.assertRaisesRegex(ValueError, 'family/source'):
            package.resolve_package(leaf, decode_batch_vectors=1)

    def test_b1_with_wrong_static_dtype_cannot_satisfy_explicit_decode(self):
        self.fixture('static', 1, dtypes=('int8',))
        leaf = self.fixture('static', 64)
        self.assertEqual(package.resolve_package(leaf)['decode_batch_vectors'], 64)
        with self.assertRaisesRegex(ValueError, 'dtype'):
            package.resolve_package(leaf, decode_batch_vectors=1)

    def test_batch_hints_directory_and_manifest_must_agree(self):
        leaf = self.fixture('grouped-columns-v1', 64)
        for options in ({'batch_vectors': 32}, {'batch_vectors': True}, {'decode_batch_vectors': 16}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'batch'):
                package.resolve_package(leaf, **options)
        self.update_manifest(leaf, batch_vectors=32)
        with self.assertRaisesRegex(ValueError, 'bN directory'):
            package.resolve_package(leaf)
        self.update_manifest(leaf, batch_vectors=True)
        with self.assertRaisesRegex(ValueError, 'batch'):
            package.resolve_package(leaf)

    def test_malformed_format_and_boolean_record_batch_are_rejected(self):
        leaf = self.fixture()
        self.update_manifest(leaf, recipe=['not-a-recipe'])
        with self.assertRaisesRegex(ValueError, 'format/recipe'):
            package.resolve_package(leaf)
        leaf = self.fixture()
        self.change_record(leaf, batch_vectors=True)
        with self.assertRaisesRegex(ValueError, 'batch'):
            package.resolve_package(leaf)

    def test_weight_input_mode_hint_cannot_reinterpret_a_recipe(self):
        leaf = self.fixture('grouped-columns-v1', 64)
        with self.assertRaisesRegex(ValueError, 'mode conflicts'):
            package.resolve_package(leaf, mode='float32')
        with self.assertRaisesRegex(ValueError, 'dtype'):
            package.resolve_package(leaf, weight_dtype='int8')

    def test_unknown_format_and_recipe_are_rejected(self):
        leaf = self.fixture()
        self.update_manifest(leaf, format='foreign-package')
        with self.assertRaisesRegex(ValueError, 'format/recipe'):
            package.resolve_package(leaf)
        self.update_manifest(leaf, format=package.FORMAT, recipe='old-unknown-recipe')
        with self.assertRaisesRegex(ValueError, 'format/recipe'):
            package.resolve_package(leaf)

    def test_missing_empty_or_non_manifest_path_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            package.resolve_package(self.root/'absent')
        empty = self.root/'empty'; empty.mkdir()
        with self.assertRaisesRegex(FileNotFoundError, 'MODEL.json'):
            package.resolve_package(empty)
        with self.assertRaisesRegex(ValueError, 'MODEL.json'):
            package.resolve_package(self.model)
        leaf = self.fixture()
        link = self.root/'MODEL.json'; link.symlink_to(leaf/'MODEL.json')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            package.resolve_package(link)

    def test_embedded_model_is_resolved_only_inside_its_package(self):
        leaf = self.fixture()
        embedded = leaf/'source/model.gguf'; embedded.parent.mkdir()
        embedded.write_bytes(self.model.read_bytes())
        source = dict(self.manifest(leaf)['source_gguf'], path='source/model.gguf', embedded=True)
        self.update_manifest(leaf, source_gguf=source)
        self.assertEqual(package.resolve_package(leaf)['model'], embedded)
        for path in ('../ibm-granite.gguf', str(self.model)):
            self.update_manifest(leaf, source_gguf=dict(source, path=path))
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'contained'):
                package.resolve_package(leaf)
        escape = leaf/'linked.gguf'; escape.symlink_to(self.model)
        self.update_manifest(leaf, source_gguf=dict(source, path='linked.gguf'))
        with self.assertRaisesRegex(ValueError, 'symlink'):
            package.resolve_package(leaf)

    def test_external_source_requires_absolute_path_and_explicit_relocation_keeps_hash_pin(self):
        leaf = self.fixture()
        source = self.manifest(leaf)['source_gguf']
        self.update_manifest(leaf, source_gguf=dict(source, path='relative.gguf'))
        with self.assertRaisesRegex(ValueError, 'absolute'):
            package.resolve_package(leaf)
        self.update_manifest(leaf, source_gguf=dict(source, path=str(self.root/'moved-away.gguf')))
        with self.assertRaisesRegex(ValueError, 'supply --model'):
            package.resolve_package(leaf)
        selected = package.resolve_package(leaf, model=self.model)
        self.assertEqual(selected['model'], self.model)
        self.assertEqual(selected['source_sha256'], self.source_hash)
        # Root prepare must verify this pin against an explicitly supplied file;
        # package discovery alone does not assert byte equality.
        other = self.root/'different.gguf'; other.write_bytes(b'GGUF different')
        self.assertEqual(package.resolve_package(leaf, model=other)['source_sha256'], self.source_hash)

    def test_bank_rows_allow_small_tensors_but_reject_inconsistent_large_banks(self):
        leaf = self.fixture()
        self.assertEqual(package.resolve_package(leaf)['bank_rows'], 4)
        plan = self.manifest(leaf)['plan']
        plan['dense_tensors'].append({'name': 'bad.weight', 'rows': 10, 'columns': 4, 'bank_rows': 3})
        self.update_manifest(leaf, plan=plan)
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            package.resolve_package(leaf)

    def test_stale_complete_flag_does_not_hide_missing_code(self):
        leaf = self.fixture()
        entry = next(iter(self.manifest(leaf)['artifacts'].values()))
        record = json.loads((leaf/entry['metadata']).read_text())
        (leaf/record['bytecode_path']).unlink()
        with self.assertRaisesRegex(ValueError, 'missing'):
            package.resolve_package(leaf)

    def test_metadata_hash_paths_and_artifact_dtype_are_checked(self):
        leaf = self.fixture()
        entry = next(iter(self.manifest(leaf)['artifacts'].values()))
        (leaf/entry['metadata']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'metadata hash'):
            package.resolve_package(leaf)
        leaf = self.fixture()
        self.change_record(leaf, weight_dtype='unsupported-qtype')
        with self.assertRaisesRegex(ValueError, 'dtype'):
            package.resolve_package(leaf)
        leaf = self.fixture()
        self.change_record(leaf, bytecode_path='../outside.dgc')
        with self.assertRaisesRegex(ValueError, 'identity/path'):
            package.resolve_package(leaf)
        leaf = self.fixture('grouped-fc-v1', 16)
        self.change_record(leaf, weight_dtype='float32')
        with self.assertRaisesRegex(ValueError, 'wrong weight input dtype'):
            package.resolve_package(leaf)

    def test_fused_packages_expose_flags_without_inventing_selections(self):
        for batch in (1, 64):
            with self.subTest(batch=batch):
                leaf = self.fixture('fused-ffn-v1', batch)
                value = package.resolve_package(leaf)
                self.assertEqual(value['artifacts'], self.base)
                self.assertTrue(value['fused_gate_up'])
                self.assertEqual(value['fused_bank_rows'], 4)
                self.assertNotIn('fused_gate_up_tensors', value)

    def test_full_ffn_uses_base_bank_rows_and_exposes_only_program_flags(self):
        self.fixture('static', 1)
        leaf = self.fixture('full-ffn-v1', 16)
        value = package.resolve_package(leaf)
        self.assertEqual(value['bank_rows'], 4)
        self.assertTrue(value['full_ffn'])
        self.assertEqual(value['full_ffn_hidden_chunk'], 2048)
        self.assertNotIn('full_ffn_tensors', value)
        self.assertNotIn('full_ffn_phases', value)

    def test_empty_artifacts_fail_even_if_marked_complete(self):
        leaf = self.fixture()
        self.update_manifest(leaf, artifacts={})
        with self.assertRaisesRegex(ValueError, 'no saved artifacts'):
            package.resolve_package(leaf)


if __name__ == '__main__':
    unittest.main()
