"""Synthetic GGUF parser, same-inode hashing, and primary-reference decode tests."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import numpy as np

from gguf_hf_loader import (GGUFWeights, QUANT_SIZES, SCALAR_DTYPES,
                            UnsupportedQuantizationError, dequantization_available)
from gguf_inspect import inspect_gguf
from gguf_vendor.constants import GGMLQuantizationType
from gguf_vendor.quants import dequantize as primary_dequantize


def string(value):
    data = value.encode('utf8')
    return struct.pack('<Q', len(data)) + data


def field(key, kind, value):
    if kind == 8:
        data = string(value)
    elif kind == 9:
        subtype, values = value
        data = struct.pack('<IQ', subtype, len(values))
        data += b''.join(string(v) if subtype == 8 else struct.pack('<I', v) for v in values)
    else:
        data = struct.pack({4:'<I', 5:'<i', 10:'<Q'}[kind], value)
    return string(key) + struct.pack('<I', kind) + data


def write_gguf(path, tensors, metadata=(), alignment=32):
    """tensors: name, NumPy shape, type ID, raw payload, optional offset."""
    metadata = [('general.architecture', 8, 'synthetic'),
                ('general.alignment', 4, alignment), *metadata]
    descriptors = bytearray()
    payload = bytearray()
    for item in tensors:
        name, shape, kind, data = item[:4]
        while len(payload) % alignment:
            payload.append(0)
        offset = item[4] if len(item) == 5 else len(payload)
        descriptors += string(name) + struct.pack('<I', len(shape))
        descriptors += struct.pack('<' + 'Q' * len(shape), *reversed(shape))
        descriptors += struct.pack('<IQ', kind, offset)
        payload += data
    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(metadata))
    header += b''.join(field(*f) for f in metadata) + descriptors
    header += b'\0' * ((-len(header)) % alignment)
    result = header + payload
    path.write_bytes(result)
    return result


def scalar_reference(blocks, kind):
    """Independent scalar interpretations of the pinned C block layouts."""
    result = []
    for block in blocks:
        data = block.tobytes()
        if kind == 15:
            scale = struct.unpack('<f', data[:4])[0]
            result.extend(scale * value for value in struct.unpack('<256b', data[4:260]))
        else:
            scale = struct.unpack('<e', data[:2])[0]
            if kind == 9:
                result.extend(scale * value for value in struct.unpack('<32b', data[4:]))
            elif kind == 41:
                result.extend(scale if (data[2 + j // 8] >> (j % 8)) & 1 else -scale
                              for j in range(128))
            elif kind == 42:
                result.extend(scale * (((data[2 + j // 4] >> (2 * (j % 4))) & 3) - 1)
                              for j in range(64))
            else:
                raise AssertionError(kind)
    return np.array(result, dtype=np.float32)


class GGUFTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / 'synthetic.gguf'

    def tearDown(self):
        self.temporary.cleanup()

    def small_float(self, **kwargs):
        values = np.arange(16, dtype=np.float32).reshape(2, 8)
        return write_gguf(self.path, [('weight', values.shape, 0, values.tobytes())], **kwargs)

    def test_every_bundled_format_multiple_rows_and_bounded_batches(self):
        rng = np.random.default_rng(20261013)
        tested = []
        for kind, (block, size) in sorted(QUANT_SIZES.items()):
            if not dequantization_available(kind):
                continue
            with self.subTest(kind=kind):
                rows, columns = 3, block * 2
                if kind in SCALAR_DTYPES:
                    data = (np.arange(rows * columns) % 31 - 15).astype(SCALAR_DTYPES[kind]).reshape(rows, columns)
                    raw, expected = data.tobytes(), data.astype(np.float32)
                elif kind == 30:
                    data = np.linspace(-3, 4, rows * columns, dtype=np.float32)
                    raw = (data.view(np.uint32) >> 16).astype('<u2').tobytes()
                    expected = (np.frombuffer(raw, '<u2').astype(np.uint32) << 16).view(np.float32).reshape(rows, columns)
                else:
                    blocks = rng.integers(0, 256, (rows * 2, size), dtype=np.uint8)
                    # Keep common FP16/FP32 scales finite while varying all bitfields.
                    offsets = {2:[0], 3:[0,2], 6:[0], 7:[0,2], 8:[0], 9:[0,2],
                               10:[80,82], 11:[108], 12:[0,2], 13:[0,2], 14:[208],
                               16:[0], 17:[0], 18:[0], 19:[0], 20:[0], 21:[0],
                               22:[0], 23:[0], 34:[size-2], 35:[size-2], 41:[0], 42:[0]}
                    for offset in offsets.get(kind, []):
                        blocks[:, offset:offset+2] = np.frombuffer(struct.pack('<e', 0.3125), np.uint8)
                    if kind == 15:
                        blocks[:, :4] = np.frombuffer(struct.pack('<f', 0.3125), np.uint8)
                    raw = blocks.tobytes()
                    with np.errstate(all='ignore'):
                        if kind in (9,15,41,42):
                            expected = scalar_reference(blocks, kind).reshape(rows, columns)
                        else:
                            expected = primary_dequantize(blocks.reshape(rows, 2*size), GGMLQuantizationType(kind))
                write_gguf(self.path, [('weight', (rows,columns), kind, raw)])
                with GGUFWeights(self.path) as weights, np.errstate(all='ignore'):
                    np.testing.assert_array_equal(weights.numpy('weight', max_block_batch=1), expected)
                    np.testing.assert_array_equal(weights.numpy_rows('weight', 1, 3, max_block_batch=5), expected[1:])
                    self.assertEqual(weights.numpy_rows('weight', 2, 2).shape, (0, columns))
                tested.append(kind)
        self.assertEqual(set(tested), set(QUANT_SIZES))

    def test_non_power_of_two_alignment_is_supported(self):
        self.small_float(alignment=24)
        with GGUFWeights(self.path) as weights:
            self.assertEqual(weights.alignment, 24)
            self.assertEqual(weights.data_offset % 24, 0)
            np.testing.assert_array_equal(weights.numpy('weight'), np.arange(16).reshape(2,8))

    def test_unknown_type_remains_inspectable_but_not_decodable(self):
        write_gguf(self.path, [('mystery', (2,8), 123456, bytes(range(16)))])
        report = inspect_gguf(self.path, include_tensors=True)
        self.assertEqual(report['architecture'], 'synthetic')
        self.assertFalse(report['all_tensor_types_decodable'])
        self.assertFalse(report['tensors'][0]['size_known'])
        self.assertEqual(report['tensors'][0]['nbytes'], 16)
        with GGUFWeights(self.path) as weights:
            with self.assertRaises(UnsupportedQuantizationError):
                weights.numpy('mystery')
            with self.assertRaises(UnsupportedQuantizationError):
                weights.numpy_rows('mystery', 0, 1)

    def test_large_token_array_is_skipped(self):
        self.small_float(metadata=[('tokenizer.ggml.tokens',9,(8,['token']*2000))])
        with GGUFWeights(self.path) as weights:
            self.assertNotIn('tokenizer.ggml.tokens', weights.fields)
            self.assertIn('tokenizer.ggml.tokens', weights.skipped_fields)

    def test_hash_and_mmap_use_retained_open_file(self):
        raw = self.small_float()
        with GGUFWeights(self.path) as weights:
            self.assertEqual(weights.source_identity[:2], (self.path.stat().st_dev,self.path.stat().st_ino))
            with mock.patch.object(Path, 'open', side_effect=AssertionError('Source reopened')):
                self.assertEqual(weights.sha256(chunk_bytes=17), hashlib.sha256(raw).hexdigest())
                np.testing.assert_array_equal(weights.numpy_rows('weight',0,1), np.arange(8).reshape(1,8))

    def test_cached_digest_rejects_source_changes(self):
        self.small_float()
        with GGUFWeights(self.path) as weights:
            weights.sha256()
            with self.path.open('r+b') as f:
                f.seek(-1,2);f.write(b'\xff')
            # The Android-backed filesystem can coalesce writes within one
            # timestamp tick. Give this synthetic mutation a distinct mtime.
            info = self.path.stat()
            os.utime(self.path, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
            with self.assertRaisesRegex(ValueError, 'changed'):
                weights.sha256()
            with self.assertRaisesRegex(ValueError, 'changed'):
                weights.numpy('weight')

    def test_path_replacement_is_rejected(self):
        self.small_float()
        with GGUFWeights(self.path) as weights:
            replacement = self.path.with_name('replacement.gguf')
            replacement.write_bytes(self.path.read_bytes())
            os.replace(replacement, self.path)
            with self.assertRaisesRegex(ValueError, 'changed'):
                weights.check_source_unchanged(check_path=True)

    def test_non_regular_source_is_rejected_without_fifo_block(self):
        os.mkfifo(self.path)
        with self.assertRaisesRegex(ValueError, 'regular'):
            GGUFWeights(self.path)

    def test_truncated_tensor_is_rejected(self):
        raw = self.small_float();self.path.write_bytes(raw[:-1])
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            GGUFWeights(self.path)

    def test_invalid_quantized_row_is_rejected(self):
        write_gguf(self.path, [('weight',(2,16),2,b'\0'*18)])
        with self.assertRaisesRegex(ValueError, 'quantization row'):
            GGUFWeights(self.path)

    def test_duplicate_tensor_is_rejected(self):
        write_gguf(self.path, [('same',(1,),0,b'\0'*4)]*2)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            GGUFWeights(self.path)

    def test_overlapping_tensor_extents_are_rejected(self):
        for offset in (0, 32):
            with self.subTest(offset=offset):
                write_gguf(self.path, [('first',(16,),0,b'\0'*64),
                                       ('second',(1,),0,b'\0'*4,offset)])
                with self.assertRaisesRegex(ValueError, '[Oo]verlapping'):
                    GGUFWeights(self.path)

    def test_duplicate_metadata_is_rejected(self):
        self.small_float(metadata=[('same',4,1),('same',4,2)])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            GGUFWeights(self.path)

    def test_invalid_alignment_is_rejected(self):
        for alignment in (1,2,4,12):
            with self.subTest(alignment=alignment):
                self.small_float(alignment=alignment)
                with self.assertRaisesRegex(ValueError, 'multiple of eight'):
                    GGUFWeights(self.path)

    def test_skipped_array_cannot_seek_past_file(self):
        key = string('tokenizer.ggml.tokens')
        self.path.write_bytes(b'GGUF'+struct.pack('<IQQ',3,0,1)+key+struct.pack('<IIQ',9,8,100))
        with self.assertRaisesRegex(ValueError, 'array length'):
            GGUFWeights(self.path)

    def test_unknown_metadata_type_is_rejected(self):
        self.path.write_bytes(b'GGUF'+struct.pack('<IQQ',3,0,1)+string('key')+struct.pack('<I',999))
        with self.assertRaisesRegex(ValueError, 'metadata field type'):
            GGUFWeights(self.path)

    def test_zero_decode_batch_is_rejected(self):
        self.small_float()
        with GGUFWeights(self.path) as weights:
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                weights.numpy('weight',max_block_batch=0)


if __name__ == '__main__':
    unittest.main()
