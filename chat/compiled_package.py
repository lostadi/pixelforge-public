"""Describe one compiled GGUF package without compiling or choosing CPU fallbacks.

This is package discovery, not runtime admission. The installer must still hash
its source model and run the authoritative source/device/coverage preflight.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys

_ROOT = Path(__file__).resolve().parent
_KIT = _ROOT / 'kit'
if not _KIT.is_dir():
    _KIT = _ROOT.parent / 'tools/arm_npu'
if str(_KIT) not in sys.path:
    sys.path.insert(0, str(_KIT))
from compiled_model import (FORMAT, RECIPE, BATCH_RECIPE, FUSED_RECIPE,
    FULL_FFN_FORMAT, FULL_FFN_RECIPE, DYNAMIC_FC_FORMAT, DYNAMIC_FC_RECIPE,
    GROUPED_FC_FORMAT, GROUPED_FC_RECIPE, GROUPED_COLUMNS_FORMAT,
    GROUPED_COLUMNS_RECIPE, MAX_CODE_BYTES)


_FAMILIES = {
    (FORMAT, RECIPE): ('static', 'off'),
    (FORMAT, BATCH_RECIPE): ('static', 'off'),
    (FORMAT, FUSED_RECIPE): ('fused-ffn-v1', 'off'),
    (FULL_FFN_FORMAT, FULL_FFN_RECIPE): ('full-ffn-v1', 'off'),
    (DYNAMIC_FC_FORMAT, DYNAMIC_FC_RECIPE): ('dynamic-fc-v1', 'float32'),
    (GROUPED_FC_FORMAT, GROUPED_FC_RECIPE): ('grouped-fc-v1', 'original_groups'),
    (GROUPED_COLUMNS_FORMAT, GROUPED_COLUMNS_RECIPE): ('grouped-columns-v1', 'original_columns'),
}
_BATCH_FAMILIES = ('dynamic-fc-v1', 'grouped-fc-v1', 'grouped-columns-v1', 'full-ffn-v1')
_DIGEST = re.compile(r'[0-9a-f]{64}')
_BATCH = re.compile(r'b([1-9][0-9]*)')
_MAX_JSON_BYTES = 16 * 1024**2


def _digest(value):
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _json(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Compiled package requires a regular JSON file: ' + str(path))
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError('Compiled package JSON is oversized')
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError('Compiled package JSON must be an object')
    return value


def _contained_file(directory, relative):
    if not isinstance(relative, str) or not relative or '\x00' in relative:
        raise ValueError('Compiled package file must have a contained relative path')
    name = PurePosixPath(relative)
    if name.is_absolute() or '..' in name.parts or not name.parts:
        raise ValueError('Compiled package file must have a contained relative path')
    path = directory.joinpath(*name.parts)
    current = path
    while current != directory:
        if current.is_symlink():
            raise ValueError('Compiled package file must not traverse a symlink')
        current = current.parent
    if not path.is_file() or not path.resolve().is_relative_to(directory):
        raise ValueError('Compiled package file is missing or outside its package: ' + relative)
    return path


def _leaf(package):
    path = Path(package).expanduser()
    if path.is_symlink():
        raise ValueError('Compiled package must not be a symlink')
    if path.is_file():
        if path.name != 'MODEL.json':
            raise ValueError('Select a compiled package MODEL.json')
        return path.parent.resolve(), _json(path)
    if not path.is_dir():
        raise FileNotFoundError('Compiled package path does not exist: ' + str(path))
    if (path / 'MODEL.json').exists():
        return path.resolve(), _json(path / 'MODEL.json')
    candidates = []
    patterns = ['batch-fc-v1/b*/MODEL.json', 'fused-ffn-v1/MODEL.json',
                'fused-ffn-v1/batch-fc-v1/b*/MODEL.json']
    patterns += [family + '/b*/MODEL.json' for family in _BATCH_FAMILIES]
    if path.name in (*_BATCH_FAMILIES, 'batch-fc-v1'):
        patterns.append('b*/MODEL.json')
    for pattern in patterns:
        candidates.extend(path.glob(pattern))
    candidates = sorted(set(candidates))
    if not candidates:
        raise FileNotFoundError('No compiled package MODEL.json was found')
    if len(candidates) != 1:
        raise ValueError('Multiple compiled packages are present; select an explicit leaf MODEL.json')
    return _leaf(candidates[0])


def _batch(value):
    if type(value) is not int or not 1 <= value <= 512:
        raise ValueError('Compiled package batch must be an integer from 1 to 512')
    return value


def _layout(directory, manifest, family):
    batch = _batch(manifest.get('batch_vectors', 1))
    recipe = manifest['recipe']
    if family == 'static' and recipe == RECIPE:
        if batch != 1 or directory.parent.name in (*_BATCH_FAMILIES, 'batch-fc-v1'):
            raise ValueError('Static B1 recipe has a conflicting batch/layout')
        base = directory
    elif family == 'static':
        if recipe != BATCH_RECIPE or batch == 1 or directory.parent.name != 'batch-fc-v1':
            raise ValueError('Batched static package has a conflicting recipe/layout')
        base = directory.parent.parent
        if base.name == 'fused-ffn-v1':
            raise ValueError('Static package occupies a fused recipe directory')
    elif family == 'fused-ffn-v1':
        if batch == 1:
            if directory.name != family:
                raise ValueError('Fused B1 package must use its fused-ffn-v1 directory')
            base = directory.parent
        else:
            if directory.parent.name != 'batch-fc-v1' or directory.parent.parent.name != family:
                raise ValueError('Fused batched package has an invalid directory layout')
            base = directory.parent.parent.parent
    else:
        if directory.parent.name != family:
            raise ValueError('Compiled program family differs from its directory layout')
        base = directory.parent.parent
    if batch > 1 or family in _BATCH_FAMILIES:
        match = _BATCH.fullmatch(directory.name)
        if match is None or int(match[1]) != batch:
            raise ValueError('Compiled package batch differs from its bN directory')
    return base, batch


def _source(directory, manifest, model):
    source = manifest.get('source_gguf')
    if (not isinstance(source, dict) or not _digest(source.get('sha256'))
            or type(source.get('embedded')) is not bool
            or not isinstance(source.get('path'), str) or '\x00' in source['path']):
        raise ValueError('Compiled package has an invalid source GGUF identity')
    if source['embedded']:
        inferred = _contained_file(directory, source['path'])
    else:
        inferred = Path(source['path'])
        if not inferred.is_absolute():
            raise ValueError('External source GGUF path must be absolute')
    selected = Path(model).expanduser() if model is not None else inferred
    if selected.is_symlink() or not selected.is_file():
        raise ValueError('Source GGUF must be an existing regular file; supply --model if it moved')
    # Binding the supplied file's bytes remains the authoritative installer's job.
    return selected.resolve(), source['sha256']


def _bank_rows(plan):
    dense = plan.get('dense_tensors') if isinstance(plan, dict) else None
    if not isinstance(dense, list) or not dense:
        raise ValueError('Compiled package has no dense tensor bank plan')
    for tensor in dense:
        if (not isinstance(tensor, dict) or type(tensor.get('rows')) is not int
                or type(tensor.get('bank_rows')) is not int
                or not 1 <= tensor['bank_rows'] <= tensor['rows']):
            raise ValueError('Compiled package has an invalid bank row plan')
    bank = max(tensor['bank_rows'] for tensor in dense)
    if any(tensor['bank_rows'] != min(bank, tensor['rows']) for tensor in dense):
        raise ValueError('Compiled package has inconsistent tensor bank rows')
    return bank


def _artifact_dtypes(directory, manifest, family, batch):
    artifacts = manifest.get('artifacts')
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError('Compiled package contains no saved artifacts')
    dtypes = set()
    for key, entry in artifacts.items():
        if not _digest(key) or not isinstance(entry, dict) or not _digest(entry.get('metadata_sha256')):
            raise ValueError('Invalid compiled artifact metadata identity')
        expected = 'graphs/' + key + '-' + entry['metadata_sha256'] + '.json'
        if entry.get('metadata') != expected:
            raise ValueError('Invalid compiled artifact metadata path')
        path = _contained_file(directory, expected)
        if path.stat().st_size > _MAX_JSON_BYTES:
            raise ValueError('Compiled artifact metadata is oversized')
        raw = path.read_bytes()
        if len(raw) > _MAX_JSON_BYTES or hashlib.sha256(raw).hexdigest() != entry['metadata_sha256']:
            raise ValueError('Compiled artifact metadata hash mismatch')
        record = json.loads(raw)
        if (not isinstance(record, dict) or record.get('recipe') != manifest['recipe']
                or _batch(record.get('batch_vectors', 1)) != batch):
            raise ValueError('Compiled artifact recipe/batch differs from its package')
        if (record.get('source_gguf_sha256') is not None
                and record['source_gguf_sha256'] != manifest['source_gguf']['sha256']):
            raise ValueError('Compiled artifact belongs to different source GGUF bytes')
        digest = record.get('bytecode_sha256')
        if not _digest(digest) or record.get('bytecode_path') != 'graphs/' + digest + '.dgc':
            raise ValueError('Invalid compiled bytecode identity/path')
        code = _contained_file(directory, record['bytecode_path'])
        size = record.get('compiled_bytes')
        if type(size) is not int or not 8 <= size <= MAX_CODE_BYTES or code.stat().st_size != size:
            raise ValueError('Compiled bytecode size does not match its metadata')
        with code.open('rb') as stream:
            if stream.read(8)[4:8] != b'DGC0':
                raise ValueError('Compiled bytecode identifier is invalid')
        dtype = record.get('weight_dtype')
        expected_dtype = ('float32-runtime-input' if family == 'dynamic-fc-v1' else
                          'original_groups' if family in ('grouped-fc-v1', 'grouped-columns-v1') else None)
        if expected_dtype is not None:
            if dtype != expected_dtype:
                raise ValueError('Compiled program has the wrong weight input dtype')
            dtypes.add('float32')
        elif dtype in ('int8', 'float32'):
            if (family != 'static' or batch > 1) and dtype != 'float32':
                raise ValueError('Batched/fused/full-FFN programs require float32 constants')
            dtypes.add(dtype)
        else:
            raise ValueError('Unsupported compiled weight dtype')
    return dtypes


def _describe(directory, manifest, model=None):
    if not isinstance(manifest.get('format'), str) or not isinstance(manifest.get('recipe'), str):
        raise ValueError('Unsupported compiled package format/recipe')
    family_mode = _FAMILIES.get((manifest['format'], manifest['recipe']))
    if family_mode is None:
        raise ValueError('Unsupported compiled package format/recipe')
    family, mode = family_mode
    base, batch = _layout(directory, manifest, family)
    source, digest = _source(directory, manifest, model)
    plan = manifest.get('plan')
    if not isinstance(plan, dict) or plan.get('model_sha256') != digest:
        raise ValueError('Compiled package plan belongs to different source GGUF bytes')
    if family == 'full-ffn-v1':
        chunk = plan.get('hidden_chunk')
        if type(chunk) is not int or chunk < 1 or not isinstance(plan.get('layers'), list) or not plan['layers']:
            raise ValueError('Full FFN package has an invalid hidden chunk plan')
        dense_manifest = _json(base / 'MODEL.json') if (base / 'MODEL.json').exists() else None
        bank = _bank_rows(dense_manifest.get('plan')) if dense_manifest is not None else 4096
        if dense_manifest is not None and dense_manifest.get('source_gguf', {}).get('sha256') != digest:
            raise ValueError('Full FFN base banks belong to different source GGUF bytes')
    else:
        bank = _bank_rows(plan)
    dtypes = _artifact_dtypes(directory, manifest, family, batch)
    return {'family': family, 'mode': mode, 'base': base, 'batch': batch,
            'model': source, 'sha256': digest, 'bank_rows': bank, 'dtypes': dtypes, 'plan': plan}


def _b1(directory, description):
    family, base = description['family'], description['base']
    if family == 'static':
        return base
    if family == 'fused-ffn-v1':
        return base / family
    return base / family / 'b1'


def resolve_package(package, *, model=None, mode=None, weight_dtype=None,
                    batch_vectors=None, decode_batch_vectors=None):
    """Infer explicit install settings for one leaf; never trust a complete flag.

    Code existence, size and identifier are checked here. Full code hashes,
    compiler/device identity, original source bytes, exact tensor coverage and
    advanced FFN architecture admission are checked again by root preflight.
    """
    directory, manifest = _leaf(package)
    description = _describe(directory, manifest, model)
    inferred_mode = description['mode']
    if mode is not None and mode != inferred_mode:
        raise ValueError('Requested weight input mode conflicts with the compiled package')
    batch = description['batch']
    if batch_vectors is not None and _batch(batch_vectors) != batch:
        raise ValueError('Requested batch differs from the selected compiled package')
    dtypes = description['dtypes']
    if weight_dtype is None:
        if len(dtypes) != 1:
            raise ValueError('Compiled package has multiple weight dtypes; select an explicit weight dtype')
        dtype = next(iter(dtypes))
    else:
        if weight_dtype not in ('int8', 'float32') or weight_dtype not in dtypes:
            raise ValueError('Requested weight dtype is absent from the compiled package')
        dtype = weight_dtype
    decode = batch
    if decode_batch_vectors is not None:
        decode = _batch(decode_batch_vectors)
        if decode not in (1, batch):
            raise ValueError('Decode batch must be 1 or the selected prefill batch')
    if inferred_mode == 'float32' and decode != batch:
        raise ValueError('Float32 dynamic programs require the same prefill and decode batch')
    if batch > 1 and inferred_mode != 'float32' and (decode_batch_vectors is None or decode == 1):
        sibling = _b1(directory, description)
        if (sibling / 'MODEL.json').exists():
            sibling_manifest = _json(sibling / 'MODEL.json')
            format_recipe = sibling_manifest.get('format'), sibling_manifest.get('recipe')
            if any(not isinstance(item, str) for item in format_recipe):
                raise ValueError('B1 package has an invalid format/recipe')
            sibling_family = _FAMILIES.get(format_recipe)
            same_identity = (sibling_family is not None and sibling_family[0] == description['family']
                and sibling_manifest.get('source_gguf', {}).get('sha256') == description['sha256']
                and sibling_manifest.get('plan') == description['plan'])
            if same_identity:
                other = _describe(sibling, sibling_manifest, description['model'])
                if other['batch'] == 1 and dtype in other['dtypes']:
                    decode = 1
                elif decode_batch_vectors == 1:
                    raise ValueError('Matching B1 package lacks the requested weight dtype')
            elif decode_batch_vectors == 1:
                raise ValueError('B1 package does not match the selected family/source plan')
        elif decode_batch_vectors == 1:
            raise FileNotFoundError('Requested B1 decode package is missing')
    result = dict(model=description['model'], artifacts=description['base'],
                  source_sha256=description['sha256'], bank_rows=description['bank_rows'],
                  weight_dtype=dtype, dynamic_weight_mode=inferred_mode,
                  batch_vectors=batch, decode_batch_vectors=decode, npu_prefill=True)
    if description['family'] == 'fused-ffn-v1':
        result.update(fused_gate_up=True, fused_bank_rows=description['bank_rows'])
    elif description['family'] == 'full-ffn-v1':
        result.update(full_ffn=True, full_ffn_hidden_chunk=description['plan']['hidden_chunk'])
    return result
