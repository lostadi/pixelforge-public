"""Memory-light GGUF v2/v3 metadata and weight decoding.

The NumPy-only common quantizers follow pinned llama.cpp (MIT; LICENSE.ggml).
Pinned bundled gguf-py modules decode IQ/TQ and other implemented formats.
Parsing a container does not imply support for its model architecture on NPU.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import functools
import importlib
import math
import os
import pathlib
import re
import struct
import stat

import numpy as np
from gguf_quantizers import COMMON_QUANTIZERS


TYPE_NAMES = {0:'F32', 1:'F16', 2:'Q4_0', 3:'Q4_1', 6:'Q5_0', 7:'Q5_1',
              8:'Q8_0', 9:'Q8_1', 10:'Q2_K', 11:'Q3_K', 12:'Q4_K',
              13:'Q5_K', 14:'Q6_K', 15:'Q8_K', 16:'IQ2_XXS', 17:'IQ2_XS',
              18:'IQ3_XXS', 19:'IQ1_S', 20:'IQ4_NL', 21:'IQ3_S', 22:'IQ2_S',
              23:'IQ4_XS', 24:'I8', 25:'I16', 26:'I32', 27:'I64', 28:'F64',
              29:'IQ1_M', 30:'BF16', 34:'TQ1_0', 35:'TQ2_0', 39:'MXFP4',
              40:'NVFP4', 41:'Q1_0', 42:'Q2_0'}
# Elements per block, bytes per block. Pinned llama.cpp gguf-py constants.
QUANT_SIZES = {0:(1,4), 1:(1,2), 2:(32,18), 3:(32,20), 6:(32,22),
               7:(32,24), 8:(32,34), 9:(32,36), 10:(256,84), 11:(256,110),
               12:(256,144), 13:(256,176), 14:(256,210), 15:(256,292),
               16:(256,66), 17:(256,74), 18:(256,98), 19:(256,50),
               20:(32,18), 21:(256,110), 22:(256,82), 23:(256,136),
               24:(1,1), 25:(1,2), 26:(1,4), 27:(1,8), 28:(1,8),
               29:(256,56), 30:(1,2), 34:(256,54), 35:(256,66),
               39:(32,17), 40:(64,36), 41:(128,18), 42:(64,18)}
SCALAR_DTYPES = {0:'<f4', 1:'<f2', 24:'i1', 25:'<i2', 26:'<i4', 27:'<i8', 28:'<f8'}


@functools.lru_cache(maxsize=1)
def _pinned_gguf():
    """Load pinned local support only; never fetch or run code from a GGUF."""
    try:
        constants = importlib.import_module('gguf_vendor.constants')
        quants = importlib.import_module('gguf_vendor.quants')
        return constants, quants
    except ImportError:
        return None


def quantization_name(kind):
    if kind in TYPE_NAMES:
        return TYPE_NAMES[kind]
    package = _pinned_gguf()
    if package:
        try:
            return package[0].GGMLQuantizationType(kind).name
        except ValueError:
            pass
    return f'UNKNOWN_{kind}'


def dequantization_available(kind):
    if kind in SCALAR_DTYPES or kind == 30 or kind in COMMON_QUANTIZERS:
        return True
    package = _pinned_gguf()
    if not package:
        return False
    try:
        qtype = package[0].GGMLQuantizationType(kind)
        return qtype in getattr(package[1], '_type_traits', {})
    except ValueError:
        return False


class UnsupportedQuantizationError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple[int, ...]  # NumPy row-major order, reversed from GGUF dimensions.
    type_id: int
    offset: int
    nbytes: int
    size_known: bool = True  # Unknown kinds retain the available storage extent.


class GGUFWeights:
    """Inspect any architecture while skipping large tokenizer arrays.

    Unknown tensor kinds remain visible in the inventory. Their nbytes is the
    available extent up to the next tensor offset (including possible padding),
    and size_known is false. Decoding such a kind raises an explicit error.
    """
    TYPES = QUANT_SIZES

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.file_size = None
        self._file = None
        self._sha256 = None
        self.fields = {}
        self.tensors = {}
        self.skipped_fields = []
        with self._open_regular_source() as f:
            def take(n, skip=False):
                if n < 0 or n > self.file_size - f.tell():
                    raise ValueError('Truncated GGUF header or metadata')
                if skip:
                    f.seek(n, 1)
                    return None
                data = f.read(n)
                if len(data) != n:
                    raise ValueError('Truncated GGUF header or metadata')
                return data

            def scalar(fmt):
                return struct.unpack('<' + fmt, take(struct.calcsize('<' + fmt)))[0]

            def string(skip=False):
                n = scalar('Q')
                if not skip and n > 16 * 1024 * 1024:
                    raise ValueError('GGUF string exceeds metadata allocation limit')
                data = take(n, skip)
                if skip:
                    return None
                try:
                    return data.decode('utf8')
                except UnicodeDecodeError as exc:
                    raise ValueError('Invalid UTF-8 in GGUF string') from exc

            fmts = {0:'B', 1:'b', 2:'H', 3:'h', 4:'I', 5:'i',
                    6:'f', 7:'?', 10:'Q', 11:'q', 12:'d'}

            def value(kind, skip=False):
                if kind == 8:
                    return string(skip)
                if kind == 9:
                    subtype, n = scalar('I'), scalar('Q')
                    if subtype == 9 or subtype not in (*fmts, 8):
                        raise ValueError(f'Invalid GGUF array subtype {subtype}')
                    minimum = struct.calcsize('<' + fmts[subtype]) if subtype in fmts else 8
                    if n > (self.file_size - f.tell()) // minimum:
                        raise ValueError('Invalid or truncated GGUF array length')
                    if skip and subtype in fmts:
                        take(n * minimum, True)
                        return None
                    if not skip and n > 1_000_000:
                        raise ValueError('GGUF array exceeds metadata allocation limit')
                    if skip:
                        for _ in range(n):
                            value(subtype, True)
                        return None
                    return [value(subtype) for _ in range(n)]
                if kind not in fmts:
                    raise ValueError(f'Unknown GGUF metadata field type {kind}')
                if skip:
                    take(struct.calcsize('<' + fmts[kind]), True)
                    return None
                return scalar(fmts[kind])

            if take(4) != b'GGUF':
                raise ValueError('Expected little-endian GGUF v2/v3')
            self.version = scalar('I')
            if self.version not in (2, 3):
                raise ValueError('Expected little-endian GGUF v2/v3')
            count, fields = scalar('Q'), scalar('Q')
            if count > self.file_size // 32 or fields > self.file_size // 13:
                raise ValueError('Invalid GGUF header counts')
            if count > 1_000_000 or fields > 1_000_000:
                raise ValueError('GGUF header count exceeds allocation limit')
            seen_fields = set()
            for _ in range(fields):
                key, kind = string(), scalar('I')
                if not key or key in seen_fields:
                    raise ValueError(f'Empty or duplicate GGUF metadata key: {key!r}')
                seen_fields.add(key)
                skip = key.startswith('tokenizer.') and key not in {
                    'tokenizer.chat_template', 'tokenizer.ggml.bos_token_id',
                    'tokenizer.ggml.eos_token_id', 'tokenizer.ggml.padding_token_id',
                    'tokenizer.ggml.add_bos_token', 'tokenizer.ggml.add_eos_token',
                }
                data = value(kind, skip)
                if skip:
                    self.skipped_fields.append(key)
                else:
                    self.fields[key] = data
            alignment = self.fields.get('general.alignment', 32)
            if type(alignment) is not int or alignment <= 0 or alignment % 8:
                raise ValueError('GGUF alignment must be a positive multiple of eight')
            self.alignment = alignment
            descriptors = []
            seen_tensors = set()
            seen_offsets = set()
            for _ in range(count):
                name, ndim = string(), scalar('I')
                if not name or name in seen_tensors:
                    raise ValueError(f'Empty or duplicate GGUF tensor name: {name!r}')
                seen_tensors.add(name)
                if not 1 <= ndim <= 4:
                    raise ValueError('Invalid GGUF tensor dimensions')
                shape = tuple(reversed([scalar('Q') for _ in range(ndim)]))
                if any(dim == 0 for dim in shape):
                    raise ValueError(f'Zero GGUF tensor dimension: {name}')
                kind, offset = scalar('I'), scalar('Q')
                if offset % alignment:
                    raise ValueError(f'Nonaligned GGUF tensor offset: {name}')
                if offset in seen_offsets:
                    raise ValueError(f'Overlapping GGUF tensor offset: {name}')
                seen_offsets.add(offset)
                sizes = self.TYPES.get(kind)
                if sizes is None:
                    package = _pinned_gguf()
                    if package:
                        try:
                            sizes = package[0].GGML_QUANT_SIZES.get(package[0].GGMLQuantizationType(kind))
                        except ValueError:
                            pass
                if sizes is not None:
                    block, size = sizes
                    if shape[-1] % block:
                        raise ValueError(f'Nonaligned quantization row: {name}')
                    nbytes = math.prod(shape) // block * size
                else:
                    nbytes = None
                descriptors.append((name, shape, kind, offset, nbytes))
            self.data_offset = (f.tell() + alignment - 1) // alignment * alignment
            available = self.file_size - self.data_offset
            if descriptors and available < 0:
                raise ValueError('Truncated GGUF tensor data alignment')
            offsets = sorted({d[3] for d in descriptors})
            next_offset = {offset: offsets[i + 1] if i + 1 < len(offsets) else available
                           for i, offset in enumerate(offsets)}
            for name, shape, kind, offset, nbytes in descriptors:
                if not 0 <= offset <= available:
                    raise ValueError(f'GGUF tensor offset outside file: {name}')
                extent = next_offset[offset] - offset
                if nbytes is not None and (nbytes > available - offset or nbytes > extent):
                    raise ValueError(f'Truncated or overlapping GGUF tensor: {name}')
                self.tensors[name] = Tensor(name, shape, kind, offset,
                                            extent if nbytes is None else nbytes,
                                            nbytes is not None)
            self.mapping = np.memmap(f, dtype=np.uint8, mode='r')

    @staticmethod
    def _identity(info):
        return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)

    @contextlib.contextmanager
    def _open_regular_source(self):
        flags = os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NONBLOCK', 0)
        fd = os.open(self.path, flags)
        try:
            handle = os.fdopen(fd, 'rb')
        except BaseException:
            os.close(fd)
            raise
        try:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('GGUF source must be a regular file')
            self.source_identity = self._identity(info)
            self.file_size = info.st_size
            self._file = handle
            yield handle
            self.check_source_unchanged(check_path=True)
        except BaseException:
            handle.close()
            raise

    def check_source_unchanged(self, check_path=False):
        """Verify the opened inode, optionally also that its path still names it."""
        if self._file is None or self._file.closed:
            raise ValueError('GGUF source reader is closed')
        if self._identity(os.fstat(self._file.fileno())) != self.source_identity:
            raise ValueError('GGUF source changed while open')
        if check_path:
            try:
                identity = self._identity(self.path.stat())
            except OSError as exc:
                raise ValueError('GGUF source path no longer names the opened file') from exc
            if identity != self.source_identity:
                raise ValueError('GGUF source path changed while open')
        return True

    def sha256(self, chunk_bytes=8 * 1024 * 1024):
        """Hash the same regular-file descriptor used for parsing and mapping."""
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ValueError('chunk_bytes must be a positive integer')
        self.check_source_unchanged(check_path=True)
        if self._sha256 is None:
            digest = hashlib.sha256()
            for offset in range(0, self.file_size, chunk_bytes):
                wanted = min(chunk_bytes, self.file_size - offset)
                data = os.pread(self._file.fileno(), wanted, offset)
                if len(data) != wanted:
                    raise ValueError('GGUF source truncated while hashing')
                digest.update(data)
            self.check_source_unchanged(check_path=True)
            self._sha256 = digest.hexdigest()
        return self._sha256

    def close(self):
        """Release this reader's resources; returned raw views can retain mapping."""
        if self._file is not None:
            self._file.close()
        self.mapping = None

    def __enter__(self):
        self.check_source_unchanged(check_path=True)
        return self

    def __exit__(self, *args):
        self.close()

    def raw(self, name):
        self.check_source_unchanged()
        tensor = self.tensors[name]
        start = self.data_offset + tensor.offset
        return self.mapping[start:start + tensor.nbytes]

    @staticmethod
    def dequantize_blocks(blocks, kind):
        blocks = np.asarray(blocks)
        sizes = QUANT_SIZES.get(kind)
        package = None
        if sizes is None:
            package = _pinned_gguf()
            if package:
                try:
                    sizes = package[0].GGML_QUANT_SIZES.get(package[0].GGMLQuantizationType(kind))
                except ValueError:
                    pass
        if sizes is None or not dequantization_available(kind):
            raise UnsupportedQuantizationError(
                f'Cannot decode GGUF type {quantization_name(kind)} ({kind}); '
                'this pinned decoder does not implement that quantizer; choose a supported quantization.')
        if blocks.dtype != np.uint8 or blocks.ndim != 2 or blocks.shape[1] != sizes[1]:
            raise ValueError('Invalid GGUF quantization block array')
        if len(blocks) == 0:
            return np.empty((0, sizes[0]), dtype=np.float32)
        blocks = np.ascontiguousarray(blocks)
        if kind in COMMON_QUANTIZERS:
            return COMMON_QUANTIZERS[kind].dequantize_blocks(blocks)
        package = package or _pinned_gguf()
        qtype = package[0].GGMLQuantizationType(kind)
        return package[1].dequantize(blocks, qtype).reshape(len(blocks), sizes[0])

    def _decode_raw(self, raw, tensor, shape, dtype, max_block_batch):
        if type(max_block_batch) is not int or max_block_batch <= 0:
            raise ValueError('max_block_batch must be a positive integer')
        if not tensor.size_known or not dequantization_available(tensor.type_id):
            raise UnsupportedQuantizationError(
                f'Cannot decode tensor {tensor.name!r}: '
                f'{quantization_name(tensor.type_id)} ({tensor.type_id}); inspection remains available.')
        if tensor.type_id in SCALAR_DTYPES:
            return raw.view(SCALAR_DTYPES[tensor.type_id]).reshape(shape).astype(dtype, copy=True)
        if tensor.type_id == 30:
            integers = raw.view('<u2').astype(np.uint32) << np.uint32(16)
            return integers.view(np.float32).reshape(shape).astype(dtype, copy=False)
        sizes = self.TYPES.get(tensor.type_id)
        if sizes is None:
            constants = _pinned_gguf()[0]
            sizes = constants.GGML_QUANT_SIZES[constants.GGMLQuantizationType(tensor.type_id)]
        block_size, byte_size = sizes
        blocks = raw.reshape(-1, byte_size)
        result = np.empty(math.prod(shape), dtype=dtype)
        for start in range(0, len(blocks), max_block_batch):
            decoded = self.dequantize_blocks(blocks[start:start + max_block_batch], tensor.type_id)
            offset = start * block_size
            result[offset:offset + decoded.size] = decoded.reshape(-1)
        return result.reshape(shape)

    def numpy(self, name, dtype=np.float32, max_block_batch=8192):
        """Allocate one output tensor; bound temporary common decode batches."""
        tensor = self.tensors[name]
        return self._decode_raw(self.raw(name), tensor, tensor.shape, dtype, max_block_batch)

    def numpy_rows(self, name, start, stop, dtype=np.float32, max_block_batch=8192):
        """Decode contiguous matrix rows with bounded intermediate batches."""
        tensor = self.tensors[name]
        if len(tensor.shape) != 2 or not 0 <= start <= stop <= tensor.shape[0]:
            raise ValueError('Invalid GGUF matrix row range')
        if not tensor.size_known or not dequantization_available(tensor.type_id):
            raise UnsupportedQuantizationError(
                f'Cannot decode tensor {name!r}: {quantization_name(tensor.type_id)} ({tensor.type_id})')
        columns = tensor.shape[1]
        sizes = self.TYPES.get(tensor.type_id)
        if sizes is None:
            constants = _pinned_gguf()[0]
            sizes = constants.GGML_QUANT_SIZES[constants.GGMLQuantizationType(tensor.type_id)]
        block, size = sizes
        if columns % block:
            raise ValueError('Nonaligned GGUF matrix row')
        row_bytes = columns // block * size
        raw = self.raw(name)[start * row_bytes:stop * row_bytes]
        return self._decode_raw(raw, tensor, (stop-start, columns), dtype, max_block_batch)

    @property
    def decode_supported_types(self):
        """Actual implemented decoding types, distinct from known size layouts."""
        return frozenset(kind for kind in self.TYPES if dequantization_available(kind))

    def architecture_metadata(self):
        """Return source facts without assuming a transformer implementation."""
        architecture = self.fields.get('general.architecture')
        prefix = f'{architecture}.' if isinstance(architecture, str) else None
        metadata = {k: v for k, v in self.fields.items() if prefix and k.startswith(prefix)}
        embedding = self.tensors.get('token_embd.weight')
        return {'architecture': architecture,
                'architecture_fields': metadata,
                'vocabulary_size_from_embedding': embedding.shape[0] if embedding and len(embedding.shape) == 2 else None,
                'output_weight_present': 'output.weight' in self.tensors,
                'tensor_count': len(self.tensors),
                'tensor_elements': sum(math.prod(t.shape) for t in self.tensors.values())}

    def llama_config(self):
        if self.fields.get('general.architecture') != 'llama':
            raise ValueError('This HF mapping only supports standard Llama architectures')
        f = self.fields
        hidden, heads = f['llama.embedding_length'], f['llama.attention.head_count']
        return dict(model_type='llama', architectures=['LlamaForCausalLM'],
                    hidden_size=hidden, intermediate_size=f['llama.feed_forward_length'],
                    num_hidden_layers=f['llama.block_count'], num_attention_heads=heads,
                    num_key_value_heads=f['llama.attention.head_count_kv'],
                    head_dim=f.get('llama.attention.key_length', hidden // heads),
                    vocab_size=self.tensors['token_embd.weight'].shape[0],
                    rms_norm_eps=f['llama.attention.layer_norm_rms_epsilon'],
                    rope_theta=f.get('llama.rope.freq_base', 10000),
                    max_position_embeddings=f['llama.context_length'],
                    tie_word_embeddings='output.weight' not in self.tensors,
                    hidden_act='silu', attention_bias=False, mlp_bias=False)

    def hf_names(self):
        config = self.llama_config()
        fixed = {'token_embd.weight':'model.embed_tokens.weight',
                 'output_norm.weight':'model.norm.weight', 'output.weight':'lm_head.weight'}
        suffixes = {'attn_q':'self_attn.q_proj', 'attn_k':'self_attn.k_proj',
                    'attn_v':'self_attn.v_proj', 'attn_output':'self_attn.o_proj',
                    'attn_norm':'input_layernorm', 'ffn_norm':'post_attention_layernorm',
                    'ffn_down':'mlp.down_proj', 'ffn_gate':'mlp.gate_proj', 'ffn_up':'mlp.up_proj'}
        for name in self.tensors:
            if name in fixed:
                yield name, fixed[name], None
                continue
            if name == 'rope_freqs.weight':
                continue  # Must supply the matching original HF rope_scaling config.
            match = re.fullmatch(r'blk\.(\d+)\.([a-z_]+)\.weight', name)
            if not match or match[2] not in suffixes:
                raise ValueError(f'Unmapped GGUF tensor: {name}')
            heads = None
            if match[2] == 'attn_q':
                heads = config['num_attention_heads']
            elif match[2] == 'attn_k':
                heads = config['num_key_value_heads']
            yield name, f'model.layers.{match[1]}.{suffixes[match[2]]}.weight', heads

    @staticmethod
    def undo_rope_permutation(weights, heads):
        """Inverse of convert_hf_to_gguf LlamaModel.permute."""
        shape = weights.shape
        return weights.reshape(heads, shape[0] // heads // 2, 2, *shape[1:]).swapaxes(1, 2).reshape(shape)

    def load_hf_model(self, config, dtype='float32', progress=None):
        """Build on torch.meta and assign each real tensor once, with no checkpoint file.

        config must include original HF RoPE scaling and token IDs where applicable.
        Converted tensors preserve the installed GGUF approximation of Huihui weights.
        """
        import torch
        from transformers import LlamaConfig, LlamaForCausalLM
        torch_dtype = getattr(torch, dtype)
        numpy_dtype = np.float16 if dtype == 'float16' else np.float32
        with torch.device('meta'):
            model = LlamaForCausalLM(LlamaConfig(**config))
        expected = dict(model.named_parameters())
        for gguf_name, hf_name, heads in self.hf_names():
            weights = self.numpy(gguf_name, dtype=numpy_dtype)
            if heads is not None:
                weights = self.undo_rope_permutation(weights, heads)
            target = expected.get(hf_name)
            if target is None or tuple(target.shape) != tuple(weights.shape):
                raise ValueError(f'HF tensor mismatch: {hf_name}')
            parent_name, leaf = hf_name.rsplit('.', 1)
            tensor = torch.from_numpy(np.ascontiguousarray(weights)).to(torch_dtype)
            model.get_submodule(parent_name)._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)
            if progress:
                progress(hf_name, weights.shape)
        if config.get('tie_word_embeddings'):
            model.tie_weights()
        # Meta construction also puts rotary inv_freq buffers on meta. Recreate
        # this small module on CPU using the same original HF RoPE config.
        rotary = model.model.rotary_emb
        model.model.rotary_emb = type(rotary)(config=model.config)
        unresolved = [n for n, p in model.named_parameters() if p.is_meta]
        if unresolved:
            raise ValueError(f'Unassigned model parameters: {unresolved}')
        unresolved_buffers = [n for n, p in model.named_buffers() if p.is_meta]
        if unresolved_buffers:
            raise ValueError(f'Unassigned model buffers: {unresolved_buffers}')
        return model.eval()
