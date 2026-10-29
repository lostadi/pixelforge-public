"""Create a small deterministic, synthetic GGUF for actual engine/NPU tests.

This fixture is not a useful language model. It tests GGUF loading, embedded
tokenization, a complete one-layer Llama forward graph, and tied output weights.
"""
import argparse
from pathlib import Path
import struct

import numpy as np


def create(path, *, zero_projections=False):
    hidden, intermediate, vocab = 256, 512, 259
    random = np.random.default_rng(7418)
    tensors = {"token_embd.weight": random.normal(0, .1, (vocab, hidden)).astype("<f4"),
               "output_norm.weight": np.ones(hidden, dtype="<f4"),
               "blk.0.attn_norm.weight": np.ones(hidden, dtype="<f4"),
               "blk.0.ffn_norm.weight": np.ones(hidden, dtype="<f4")}
    for name, rows, columns in (("attn_q", hidden, hidden), ("attn_k", 128, hidden),
                                ("attn_v", 128, hidden), ("attn_output", hidden, hidden),
                                ("ffn_gate", intermediate, hidden), ("ffn_up", intermediate, hidden),
                                ("ffn_down", hidden, intermediate)):
        tensors[f"blk.0.{name}.weight"] = random.normal(0, .05, (rows, columns)).astype("<f4")
        if zero_projections:
            tensors[f"blk.0.{name}.weight"].fill(0)
    byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    symbols = byte_values.copy()
    extra = 0
    for byte in range(256):
        if byte not in byte_values:
            byte_values.append(byte)
            symbols.append(256 + extra)
            extra += 1
    alphabet = dict(zip(byte_values, map(chr, symbols)))
    tokens = [alphabet[byte] for byte in range(256)] + ["<s>", "</s>", "<unk>"]
    fields = {
        "general.architecture": (8, "llama"), "general.name": (8, "PixelForge synthetic bridge fixture"),
        "general.alignment": (4, 32), "llama.context_length": (4, 256),
        "llama.embedding_length": (4, hidden), "llama.feed_forward_length": (4, intermediate),
        "llama.block_count": (4, 1), "llama.attention.head_count": (4, 4),
        "llama.attention.head_count_kv": (4, 2), "llama.attention.layer_norm_rms_epsilon": (6, 1e-5),
        "llama.rope.freq_base": (6, 10000.), "tokenizer.ggml.model": (8, "gpt2"),
        "tokenizer.ggml.pre": (8, "gpt-2"), "tokenizer.ggml.tokens": (9, (8, tokens)),
        "tokenizer.ggml.token_type": (9, (5, [1]*256 + [3, 3, 2])),
        "tokenizer.ggml.merges": (9, (8, [])), "tokenizer.ggml.bos_token_id": (4, 256),
        "tokenizer.ggml.eos_token_id": (4, 257), "tokenizer.ggml.unknown_token_id": (4, 258),
        "tokenizer.ggml.add_bos_token": (7, True), "tokenizer.ggml.add_eos_token": (7, False),
    }
    def string(value):
        data = value.encode()
        return struct.pack("<Q", len(data)) + data
    def encode(kind, value):
        if kind == 8:
            return string(value)
        if kind == 9:
            subtype, values = value
            return struct.pack("<IQ", subtype, len(values)) + b"".join(encode(subtype, v) for v in values)
        return struct.pack({4:"<I", 5:"<i", 6:"<f", 7:"<?"}[kind], value)
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(fields))
    header += b"".join(string(name) + struct.pack("<I", kind) + encode(kind, value)
                       for name, (kind, value) in fields.items())
    offset, info, payload = 0, bytearray(), bytearray()
    for name, value in tensors.items():
        dimensions = tuple(reversed(value.shape))
        info.extend(string(name) + struct.pack("<I", len(dimensions)))
        info.extend(struct.pack("<" + "Q"*len(dimensions), *dimensions))
        info.extend(struct.pack("<IQ", 0, offset))
        data = value.tobytes()
        payload.extend(data)
        padding = (-len(data)) % 32
        payload.extend(b"\0" * padding)
        offset += len(data) + padding
    header += info
    header += b"\0" * ((-len(header)) % 32)
    with Path(path).open("xb") as out:
        out.write(header)
        out.write(payload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--zero-projections", action="store_true", help="CPU ablation control for sensitivity")
    args = parser.parse_args()
    create(args.output, zero_projections=args.zero_projections)
