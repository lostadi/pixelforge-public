"""Synthetic two-layer Qwen35 with recurrent and full-attention memory.

Only a reset-correctness fixture; no downloaded weights or useful model.
"""
from pathlib import Path
import struct

import numpy as np

def create(path):
    path = Path(path)
    # Same deterministic byte vocabulary as the one-layer Llama fixture.
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    symbols, extra = byte_values.copy(), 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            symbols.append(256 + extra)
            extra += 1
    alphabet = dict(zip(byte_values, map(chr, symbols)))
    fields = {
        'general.architecture': 'qwen35', 'general.name': 'PixelForge synthetic hybrid reset fixture',
        'general.alignment': 32,
        'tokenizer.ggml.model': 'gpt2', 'tokenizer.ggml.pre': 'gpt-2',
        'tokenizer.ggml.tokens': [alphabet[byte] for byte in range(256)] + ['<s>', '</s>', '<unk>'],
        'tokenizer.ggml.token_type': [1] * 256 + [3, 3, 2], 'tokenizer.ggml.merges': [],
        'tokenizer.ggml.bos_token_id': 256, 'tokenizer.ggml.eos_token_id': 257,
        'tokenizer.ggml.unknown_token_id': 258,
        'tokenizer.ggml.add_bos_token': True, 'tokenizer.ggml.add_eos_token': False,
        'qwen35.context_length': 256, 'qwen35.embedding_length': 32,
        'qwen35.feed_forward_length': 64, 'qwen35.block_count': 2,
        'qwen35.attention.head_count': 2, 'qwen35.attention.head_count_kv': 1,
        'qwen35.attention.key_length': 16, 'qwen35.attention.value_length': 16,
        'qwen35.attention.layer_norm_rms_epsilon': 1e-6,
        'qwen35.rope.dimension_count': 16, 'qwen35.rope.dimension_sections': [2, 2, 4, 0],
        'qwen35.rope.freq_base': 10000.0, 'qwen35.full_attention_interval': 2,
        'qwen35.ssm.conv_kernel': 4, 'qwen35.ssm.inner_size': 32,
        'qwen35.ssm.state_size': 16, 'qwen35.ssm.time_step_rank': 2,
        'qwen35.ssm.group_count': 1,
    }
    random = np.random.default_rng(517)
    tensors = {'token_embd.weight': random.normal(0, .15, (259, 32)).astype('<f4'),
               'output_norm.weight': np.ones(32, dtype='<f4')}
    for layer in range(2):
        prefix = f'blk.{layer}.'
        for name in ('attn_norm.weight', 'post_attention_norm.weight'):
            tensors[prefix + name] = np.ones(32, dtype='<f4')
        matrices = {'ffn_gate.weight': (64, 32), 'ffn_up.weight': (64, 32),
                    'ffn_down.weight': (32, 64)}
        if layer == 0:
            matrices.update({'attn_qkv.weight': (64, 32), 'attn_gate.weight': (32, 32),
                             'ssm_conv1d.weight': (64, 4), 'ssm_beta.weight': (2, 32),
                             'ssm_alpha.weight': (2, 32), 'ssm_out.weight': (32, 32)})
            tensors[prefix + 'ssm_a'] = np.full(2, -.5, dtype='<f4')
            tensors[prefix + 'ssm_dt.bias'] = np.zeros(2, dtype='<f4')
            tensors[prefix + 'ssm_norm.weight'] = np.ones(16, dtype='<f4')
        else:
            matrices.update({'attn_q.weight': (64, 32), 'attn_k.weight': (16, 32),
                             'attn_v.weight': (16, 32), 'attn_output.weight': (32, 32)})
            tensors[prefix + 'attn_q_norm.weight'] = np.ones(16, dtype='<f4')
            tensors[prefix + 'attn_k_norm.weight'] = np.ones(16, dtype='<f4')
        for name, shape in matrices.items():
            tensors[prefix + name] = random.normal(0, .05, shape).astype('<f4')

    def string(value):
        raw = value.encode()
        return struct.pack('<Q', len(raw)) + raw

    def field(value):
        if isinstance(value, str):
            return struct.pack('<I', 8) + string(value)
        if isinstance(value, bool):
            return struct.pack('<I?', 7, value)
        if isinstance(value, float):
            return struct.pack('<If', 6, value)
        if isinstance(value, (list, tuple)):
            subtype = 8 if not value or isinstance(value[0], str) else 5
            data = b''.join(string(v) if subtype == 8 else struct.pack('<i', v) for v in value)
            return struct.pack('<IIQ', 9, subtype, len(value)) + data
        return struct.pack('<II', 4, value)

    header = b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(fields))
    header += b''.join(string(key) + field(value) for key, value in fields.items())
    info, data = bytearray(), bytearray()
    for name, tensor in tensors.items():
        shape = tensor.shape[::-1]
        info.extend(string(name) + struct.pack('<I', len(shape)))
        info.extend(struct.pack('<' + 'Q' * len(shape), *shape) + struct.pack('<IQ', 0, len(data)))
        data.extend(tensor.tobytes())
        data.extend(b'\0' * (-len(data) % 32))
    header += info
    header += b'\0' * (-len(header) % 32)
    with path.open('xb') as out:
        out.write(header)
        out.write(data)
