#!/usr/bin/env python3
"""Deterministic, bounded CPU stress of synthetic GGUF parsing and decoding.

No models, NPU helpers or Android vendor libraries are opened. The isolated
worker uses a small disposable fixture and reports both rejection and safe
acceptance. Success establishes these cases, not inference or arbitrary-GGUF
correctness. Run with the offline runtime's Python interpreter for NumPy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import subprocess
import sys
import tempfile
import time


KIT = Path(__file__).resolve().parent
MAX_FIXTURE_BYTES = 16 * 1024 * 1024
MAX_DECODE_ELEMENTS = 65536
MEMORY_LIMIT_BYTES = 512 * 1024 * 1024
SOURCE_FILES = ('gguf_hf_loader.py', 'gguf_quantizers.py', 'gguf_vendor/constants.py',
                'gguf_vendor/lazy.py', 'gguf_vendor/quants.py')
SCALAR_FORMATS = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i', 6:'f',
                  7:'?', 10:'Q', 11:'q', 12:'d'}


def string(value):
    encoded = value.encode('utf8')
    return struct.pack('<Q', len(encoded)) + encoded


def metadata_value(kind, value):
    if kind == 8:
        return string(value)
    if kind == 9:
        subtype, values = value
        return struct.pack('<IQ', subtype, len(values)) + b''.join(
            metadata_value(subtype, item) for item in values)
    return struct.pack('<' + SCALAR_FORMATS[kind], value)


def fixture(tensors=None, extra=(), alignment=32, version=3):
    """Build a tiny independent container and byte locations for mutations."""
    tensors = tensors if tensors is not None else [('weight', (2, 16), 0, bytes(128), None)]
    fields = [('general.architecture', 8, 'stress'), ('general.alignment', 4, alignment), *extra]
    data = bytearray(b'GGUF' + struct.pack('<IQQ', version, len(tensors), len(fields)))
    locations = {'metadata': [], 'tensors': []}
    for name, kind, value in fields:
        positions = {'key_length': len(data), 'key': len(data) + 8}
        data += string(name)
        positions['kind'] = len(data)
        data += struct.pack('<I', kind)
        positions['value'] = len(data)
        data += metadata_value(kind, value)
        locations['metadata'].append(positions)
    payload = bytearray()
    for name, shape, kind, raw, offset in tensors:
        payload += bytes((-len(payload)) % alignment)
        positions = {'name_length': len(data), 'name': len(data) + 8}
        data += string(name)
        positions['ndim'] = len(data)
        data += struct.pack('<I', len(shape))
        positions['dimensions'] = len(data)
        data += struct.pack('<' + 'Q' * len(shape), *reversed(shape))
        positions['kind'], positions['offset'] = len(data), len(data) + 4
        data += struct.pack('<IQ', kind, len(payload) if offset is None else offset)
        payload += raw
        locations['tensors'].append(positions)
    data += bytes((-len(data)) % alignment) + payload
    if len(data) > MAX_FIXTURE_BYTES:
        raise ValueError('Generated fixture exceeded its file cap')
    return data, locations


PARSER_CATEGORIES = (
    'valid_geometry', 'valid_metadata', 'valid_skipped_tokens', 'valid_empty',
    'unknown_type', 'unknown_type_huge_geometry', 'random_bitflip',
    'bad_magic', 'bad_version', 'tensor_count_huge', 'metadata_count_huge',
    'bad_alignment', 'alignment_wrong_type', 'unaligned_offset', 'offset_outside',
    'zero_dimension', 'huge_dimension', 'bad_ndim', 'bad_quantized_row',
    'duplicate_metadata', 'duplicate_tensor_name', 'duplicate_tensor_offset',
    'overlapping_tensor_extent', 'unknown_metadata_kind', 'nested_array',
    'unknown_array_kind', 'huge_array_count', 'huge_skipped_array_count',
    'huge_string_length', 'invalid_utf8', 'truncated',
)


def parser_case(index, rng):
    category = PARSER_CATEGORIES[index % len(PARSER_CATEGORIES)]
    data, positions = fixture()
    meta, tensor = positions['metadata'], positions['tensors'][0]
    expected = 'reject'
    if category == 'valid_geometry':
        rows, columns = int(rng.integers(1, 9)), int(rng.integers(1, 33))
        alignment = int(rng.choice([8, 16, 24, 32, 40, 48, 64, 96]))
        data, _ = fixture([('weight', (rows, columns), 0, bytes(rows * columns * 4), None)],
                          alignment=alignment, version=int(rng.choice([2, 3])))
        expected = 'accept'
    elif category == 'valid_metadata':
        kind = int(rng.choice(list(SCALAR_FORMATS)))
        value = (True if kind == 7 else .125 if kind in (6,12) else 17)
        data, _ = fixture(extra=[('stress.scalar', kind, value), ('stress.text', 8, '雪 café'),
                                  ('stress.array', 9, (4, [1,2,3]))])
        expected = 'accept'
    elif category == 'valid_skipped_tokens':
        data, _ = fixture(extra=[('tokenizer.ggml.tokens', 9, (8, ['雪', 'word'] * int(rng.integers(1,65))))])
        expected = 'accept'
    elif category == 'valid_empty':
        data, _ = fixture(tensors=[])
        expected = 'accept'
    elif category in ('unknown_type', 'unknown_type_huge_geometry'):
        shape = (2,16) if category == 'unknown_type' else ((1 << 64)-1, (1 << 64)-1)
        data, _ = fixture([('weight', shape, int(rng.integers(1000, 1000000)), bytes(17), None)])
        expected = 'accept'
    elif category == 'random_bitflip':
        for _ in range(int(rng.integers(1,5))):
            offset = int(rng.integers(0,len(data)))
            data[offset] ^= 1 << int(rng.integers(0,8))
        expected = 'either'
    elif category == 'bad_magic':
        data[:4] = b'BAD!'
    elif category == 'bad_version':
        struct.pack_into('<I', data, 4, int(rng.choice([0,1,4,0xffffffff])))
    elif category in ('tensor_count_huge', 'metadata_count_huge'):
        struct.pack_into('<Q', data, 8 if category == 'tensor_count_huge' else 16,
                         [1000001, 1 << 32, (1 << 64)-1][int(rng.integers(0,3))])
    elif category == 'bad_alignment':
        struct.pack_into('<I', data, meta[1]['value'], int(rng.choice([0,1,2,4,7,12,33])))
    elif category == 'alignment_wrong_type':
        struct.pack_into('<I', data, meta[1]['kind'], 6)
        struct.pack_into('<f', data, meta[1]['value'], 32.)
    elif category in ('unaligned_offset', 'offset_outside'):
        offset = int(rng.integers(1,32)) if category == 'unaligned_offset' else 0xffffffffffffffc0
        struct.pack_into('<Q', data, tensor['offset'], offset)
    elif category in ('zero_dimension', 'huge_dimension'):
        struct.pack_into('<Q', data, tensor['dimensions'], 0 if category == 'zero_dimension' else (1 << 64)-1)
    elif category == 'bad_ndim':
        struct.pack_into('<I', data, tensor['ndim'], int(rng.choice([0,5,1000,0xffffffff])))
    elif category == 'bad_quantized_row':
        data, _ = fixture([('weight', (2,31), 2, bytes(36), None)])
    elif category == 'duplicate_metadata':
        data, _ = fixture(extra=[('duplicate',4,1),('duplicate',4,2)])
    elif category in ('duplicate_tensor_name', 'duplicate_tensor_offset', 'overlapping_tensor_extent'):
        second = 'weight' if category == 'duplicate_tensor_name' else 'second'
        offset = None if category == 'duplicate_tensor_name' else 0 if category == 'duplicate_tensor_offset' else 32
        data, _ = fixture([('weight',(2,16),0,bytes(128),None), (second,(1,),0,bytes(4),offset)])
    elif category == 'unknown_metadata_kind':
        struct.pack_into('<I', data, meta[0]['kind'], 0xffffffff)
    elif category in ('nested_array', 'unknown_array_kind', 'huge_array_count', 'huge_skipped_array_count'):
        key = 'tokenizer.ggml.tokens' if category == 'huge_skipped_array_count' else 'stress.array'
        data, positions = fixture(extra=[(key,9,(4,[1,2,3]))])
        offset = positions['metadata'][-1]['value']
        if category in ('nested_array', 'unknown_array_kind'):
            struct.pack_into('<I', data, offset, 9 if category == 'nested_array' else 999)
        else:
            struct.pack_into('<Q', data, offset+4, (1 << 64)-1)
    elif category == 'huge_string_length':
        struct.pack_into('<Q', data, meta[0]['value'], (1 << 64)-1)
    elif category == 'invalid_utf8':
        data[meta[0]['key']] = 255
    elif category == 'truncated':
        data = data[:int(rng.integers(0,len(data)))]
    return category, expected, data


def safe_accept(weights, data, np, unsupported):
    if weights.sha256(chunk_bytes=17) != hashlib.sha256(data).hexdigest():
        raise AssertionError('Retained-descriptor digest differed from fixture bytes')
    weights.architecture_metadata()
    for tensor in weights.tensors.values():
        raw = weights.raw(tensor.name)
        if len(raw) != tensor.nbytes or weights.data_offset + tensor.offset + tensor.nbytes > len(data):
            raise AssertionError('Tensor view escaped its mapped file')
        if tensor.size_known and tensor.type_id in weights.decode_supported_types:
            if math.prod(tensor.shape) > MAX_DECODE_ELEMENTS:
                raise AssertionError('Accepted known tensor exceeded fixture decode cap')
            with np.errstate(all='ignore'):
                decoded = weights.numpy(tensor.name, max_block_batch=3)
            if decoded.shape != tensor.shape:
                raise AssertionError('Decoded tensor shape differed from descriptor')
        else:
            try:
                weights.numpy(tensor.name)
            except unsupported:
                pass
            else:
                raise AssertionError('Unimplemented tensor decoded without an explicit error')


def c_layout_reference(blocks, kind, np):
    """Independent scalar interpretation of four pinned ggml C layouts."""
    output = []
    for row in blocks:
        raw = row.tobytes()
        if kind == 15:  # Q8_K: d, 256 qs, 16 unused block sums.
            scale = struct.unpack('<f',raw[:4])[0]
            output.extend(scale * value for value in struct.unpack('<256b',raw[4:260]))
        else:
            scale = struct.unpack('<e',raw[:2])[0]
            if kind == 9:  # Q8_1: half d, half sum, 32 qs.
                output.extend(scale * value for value in struct.unpack('<32b',raw[4:]))
            elif kind == 41:
                output.extend(scale if (raw[2+j//8] >> (j%8)) & 1 else -scale for j in range(128))
            elif kind == 42:
                output.extend(scale * (((raw[2+j//4] >> (2*(j%4))) & 3)-1) for j in range(64))
            else:
                raise AssertionError(kind)
    return np.array(output, dtype=np.float32)


def decode_case(kind, rng, np, sizes, scalar_types, primary, enum):
    block, size = sizes[kind]
    rows, count = int(rng.integers(2,8)), int(rng.integers(2,8))
    shape = rows, count * block
    reference = 'pinned-gguf-py'
    if kind in scalar_types:
        dtype = scalar_types[kind]
        values = rng.integers(-2000,2000,shape).astype(dtype) if kind in (24,25,26,27) else rng.normal(0,2,shape).astype(dtype)
        raw = values.tobytes()
        if kind in (0,1):
            expected = primary(np.frombuffer(raw,np.uint8).reshape(rows,count*size), enum(kind))
        else:
            expected = np.frombuffer(raw,dtype).reshape(shape).astype(np.float32)
            reference = 'scalar-ggml-storage-ABI'
    elif kind == 30:
        values = rng.normal(0,2,shape).astype(np.float32)
        raw = (values.view(np.uint32) >> 16).astype('<u2').tobytes()
        expected = primary(np.frombuffer(raw,np.uint8).reshape(rows,count*size),enum(kind))
    else:
        blocks = rng.integers(0,256,(rows*count,size),dtype=np.uint8)
        offsets = {2:[0],3:[0,2],6:[0],7:[0,2],8:[0],9:[0,2],10:[80,82],11:[108],
                   12:[0,2],13:[0,2],14:[208],16:[0],17:[0],18:[0],19:[0],20:[0],
                   21:[0],22:[0],23:[0],34:[size-2],35:[size-2],41:[0],42:[0]}
        for offset in offsets.get(kind,[]):
            scales = rng.uniform(.05,2,len(blocks)).astype('<f2').view(np.uint8).reshape(-1,2)
            blocks[:,offset:offset+2] = scales
        if kind == 15:
            blocks[:,:4] = rng.uniform(.05,2,len(blocks)).astype('<f4').view(np.uint8).reshape(-1,4)
        raw = blocks.tobytes()
        with np.errstate(all='ignore'):
            if kind in (9,15,41,42):
                expected = c_layout_reference(blocks,kind,np).reshape(shape)
                reference = 'scalar-pinned-ggml-C-layout'
            else:
                expected = primary(blocks.reshape(rows,count*size),enum(kind))
    data, _ = fixture([('weight',shape,kind,raw,None)], alignment=int(rng.choice([8,24,32,40,64])))
    return data, expected, reference


def child_run(args):
    resource.setrlimit(resource.RLIMIT_AS,(MEMORY_LIMIT_BYTES,MEMORY_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_CPU,(30,35))
    resource.setrlimit(resource.RLIMIT_FSIZE,(MAX_FIXTURE_BYTES,MAX_FIXTURE_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
    import numpy as np
    from gguf_hf_loader import (GGUFWeights, QUANT_SIZES, SCALAR_DTYPES,
                               UnsupportedQuantizationError, dequantization_available, quantization_name)
    from gguf_vendor.constants import GGMLQuantizationType
    from gguf_vendor.quants import dequantize
    rng = np.random.default_rng(args.seed)
    started = time.monotonic()
    categories, outcomes, references = {}, Counter(), Counter()
    errors, error_count, max_bytes, nonfinite = [], 0, 0, 0
    formats = {}
    first_failure = None

    def failure(group, index, category, expected, error, data):
        nonlocal error_count, first_failure
        error_count += 1
        if len(errors) < 10:
            errors.append({'group':group,'index':index,'category':category,'expected':expected,
                           'error':type(error).__name__ + ': ' + str(error)[:500],
                           'fixture_bytes':len(data),'fixture_sha256':hashlib.sha256(data).hexdigest()})
        if first_failure is None:
            first_failure = bytes(data)

    with tempfile.TemporaryDirectory(prefix='pixelforge-reader-stress-') as directory:
        path = Path(directory) / 'fixture.gguf'
        for index in range(args.parser_cases):
            category, expected, data = parser_case(index,rng)
            max_bytes = max(max_bytes,len(data))
            path.write_bytes(data)
            statistics = categories.setdefault(category,Counter())
            statistics['cases'] += 1
            try:
                weights = GGUFWeights(path)
            except (ValueError,OSError) as error:
                outcomes['rejected'] += 1
                statistics['rejected'] += 1
                if expected == 'accept':
                    failure('parser',index,category,expected,error,data)
                else:
                    outcomes['expected_reject' if expected == 'reject' else 'mutation_reject'] += 1
            except Exception as error:
                failure('parser',index,category,expected,error,data)
            else:
                try:
                    with weights:
                        safe_accept(weights,data,np,UnsupportedQuantizationError)
                    outcomes['accepted_safe'] += 1
                    statistics['accepted_safe'] += 1
                    if expected == 'reject':
                        failure('parser',index,category,expected,AssertionError('Malformed case was accepted'),data)
                    else:
                        outcomes['valid_control_accept' if expected == 'accept' else 'mutation_accept_safe'] += 1
                except Exception as error:
                    failure('parser',index,category,expected,error,data)
        for kind in sorted(QUANT_SIZES):
            name = quantization_name(kind)
            if not dequantization_available(kind):
                formats[name] = {'cases':0,'skipped':'Pinned decoder does not implement this kind'}
                continue
            statistics = formats[name] = {'cases':args.decode_per_format,'passed':0,'reference':None}
            for index in range(args.decode_per_format):
                data = b''
                try:
                    data, expected, reference = decode_case(kind,rng,np,QUANT_SIZES,SCALAR_DTYPES,dequantize,GGMLQuantizationType)
                    max_bytes = max(max_bytes,len(data))
                    path.write_bytes(data)
                    with GGUFWeights(path) as weights, np.errstate(all='ignore'):
                        np.testing.assert_array_equal(weights.numpy('weight',max_block_batch=int(rng.integers(1,12))),expected)
                        start = int(rng.integers(0,len(expected)-1))
                        stop = int(rng.integers(start+1,len(expected)+1))
                        np.testing.assert_array_equal(weights.numpy_rows('weight',start,stop,max_block_batch=int(rng.integers(1,12))),expected[start:stop])
                        np.testing.assert_array_equal(weights.numpy('weight',dtype=np.float16,max_block_batch=5),expected.astype(np.float16))
                    statistics['passed'] += 1
                    statistics['reference'] = reference
                    references[reference] += 1
                    nonfinite += int(not np.isfinite(expected).all())
                except Exception as error:
                    failure('decode',index,name,'reference-match',error,data)
    result = {'schema_version':1,'success':error_count == 0,'seed':args.seed,
              'scope':'Synthetic CPU parsing/dequantization only; no inference or NPU/vendor calls',
              'created_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-started,
              'parser_cases':args.parser_cases,'parser_outcomes':dict(outcomes),
              'parser_categories':{k:dict(v) for k,v in categories.items()},
              'decode_cases':sum(value['cases'] for value in formats.values()),'formats':formats,
              'decode_reference_cases':dict(references),'decode_comparisons_per_case':3,
              'random_block_cases_with_nonfinite_reference':nonfinite,
              'nonfinite_note':'Arbitrary bitfields can encode nonfinite values; equality includes matching NaNs, not model suitability.',
              'error_count':error_count,'errors':errors,'max_fixture_bytes':max_bytes,
              'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'bounds':{'address_space_bytes':MEMORY_LIMIT_BYTES,'cpu_soft_seconds':30,'cpu_hard_seconds':35,
                        'single_file_cap_bytes':MAX_FIXTURE_BYTES,'decode_elements_cap':MAX_DECODE_ELEMENTS,
                        'wall_timeout_seconds':args.timeout,'temporary_fixture_files':1},
              'source_hashes':{name:hashlib.sha256((KIT/name).read_bytes()).hexdigest() for name in SOURCE_FILES}}
    if first_failure is not None:
        result['first_failure_hex'] = first_failure.hex()
    write_private(args.report,result)
    return 0 if result['success'] else 1


def write_private(path, value):
    path = Path(path)
    descriptor = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,'w') as output:
        json.dump(value,output,indent=2)
        output.write('\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=2026101401)
    parser.add_argument('--parser-cases',type=int,default=1500)
    parser.add_argument('--decode-per-format',type=int,default=20)
    parser.add_argument('--timeout',type=int,default=45)
    parser.add_argument('--report',type=Path,default=Path('/tmp/pixelforge-stress-reader.json'))
    parser.add_argument('--_child',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.parser_cases < 1 or args.decode_per_format < 1 or args.timeout < 1:
        parser.error('Case counts and timeout must be positive')
    if args._child:
        return child_run(args)
    if args.report.exists() or args.report.is_symlink():
        parser.error('Report must be a fresh path; existing results are preserved')
    started = time.monotonic()
    environment = os.environ.copy()
    environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with tempfile.TemporaryDirectory(prefix='pixelforge-stress-driver-') as directory:
        result_file = Path(directory)/'result.json'
        loader = ('import runpy,sys;sys.path.insert(0,'+repr(str(KIT))+');'
                  'runpy.run_path('+repr(str(Path(__file__).resolve()))+',run_name="__main__")')
        command = [sys.executable,'-I','-c',loader,'--_child','--seed',str(args.seed),
                   '--parser-cases',str(args.parser_cases),'--decode-per-format',str(args.decode_per_format),
                   '--timeout',str(args.timeout),'--report',str(result_file)]
        try:
            process = subprocess.run(command,env=environment,capture_output=True,text=True,timeout=args.timeout)
            if result_file.exists():
                result = json.loads(result_file.read_text())
                result['isolated_worker_exit_code'] = process.returncode
                if process.returncode:
                    result['success'] = False
            else:
                result = {'success':False,'seed':args.seed,'error':'Isolated worker did not publish its result',
                          'isolated_worker_exit_code':process.returncode,'stderr':process.stderr[-2000:]}
        except subprocess.TimeoutExpired:
            result = {'success':False,'seed':args.seed,'error':'Isolated worker exceeded its wall deadline',
                      'wall_timeout_seconds':args.timeout}
    result['driver_seconds'] = time.monotonic()-started
    write_private(args.report,result)
    print(json.dumps({'success':result['success'],'seed':args.seed,'parser_cases':result.get('parser_cases'),
                      'decode_cases':result.get('decode_cases'),'error_count':result.get('error_count'),
                      'seconds':result['driver_seconds'],'report':str(args.report)}))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
