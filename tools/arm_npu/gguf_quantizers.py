"""NumPy-only common GGUF dequantizers adapted from pinned llama.cpp.

Copyright (c) 2023-2026 The ggml authors. MIT: see LICENSE.ggml.
Source: llama.cpp 254b177307b14d055bbe4c050c9c441a1e06d8a4,
gguf-py/gguf/quants.py. Only dequantization methods are retained.
Q8_1 / Q8_K follow the same revision's ggml-common.h block layouts.
"""
import numpy as np
QK_K = 256


class Q4_0:
    block_size = 32

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        d, qs = np.hsplit(blocks, [2])
        d = d.view(np.float16).astype(np.float32)
        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(15)).reshape((n_blocks, -1)).astype(np.int8) - np.int8(8)
        return d * qs.astype(np.float32)

class Q4_1:
    block_size = 32

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        m, qs = np.hsplit(rest, [2])
        d = d.view(np.float16).astype(np.float32)
        m = m.view(np.float16).astype(np.float32)
        qs = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(15)).reshape((n_blocks, -1)).astype(np.float32)
        return d * qs + m

class Q5_0:
    block_size = 32

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        qh, qs = np.hsplit(rest, [4])
        d = d.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint32)
        qh = qh.reshape((n_blocks, 1)) >> np.array([i for i in range(32)], dtype=np.uint32).reshape((1, 32))
        ql = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qh = (qh & np.uint32(1)).astype(np.uint8)
        ql = (ql & np.uint8(15)).reshape((n_blocks, -1))
        qs = (ql | qh << np.uint8(4)).astype(np.int8) - np.int8(16)
        return d * qs.astype(np.float32)

class Q5_1:
    block_size = 32

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        m, rest = np.hsplit(rest, [2])
        qh, qs = np.hsplit(rest, [4])
        d = d.view(np.float16).astype(np.float32)
        m = m.view(np.float16).astype(np.float32)
        qh = qh.view(np.uint32)
        qh = qh.reshape((n_blocks, 1)) >> np.array([i for i in range(32)], dtype=np.uint32).reshape((1, 32))
        ql = qs.reshape((n_blocks, -1, 1, cls.block_size // 2)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qh = (qh & np.uint32(1)).astype(np.uint8)
        ql = (ql & np.uint8(15)).reshape((n_blocks, -1))
        qs = (ql | qh << np.uint8(4)).astype(np.float32)
        return d * qs + m

class Q8_0:
    block_size = 32

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        d, x = np.split(blocks, [2], axis=1)
        d = d.view(np.float16).astype(np.float32)
        x = x.view(np.int8).astype(np.float32)
        return x * d

class Q2_K:
    block_size = 256

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        scales, rest = np.hsplit(blocks, [QK_K // 16])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        d, dmin = np.hsplit(rest, [2])
        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)
        dl = (d * (scales & 15).astype(np.float32)).reshape((n_blocks, QK_K // 16, 1))
        ml = (dmin * (scales >> 4).astype(np.float32)).reshape((n_blocks, QK_K // 16, 1))
        shift = np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qs = qs.reshape((n_blocks, -1, 1, 32)) >> shift & np.uint8(3)
        qs = qs.reshape((n_blocks, QK_K // 16, 16)).astype(np.float32)
        qs = dl * qs - ml
        return qs.reshape((n_blocks, -1))

class Q3_K:
    block_size = 256

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        hmask, rest = np.hsplit(blocks, [QK_K // 8])
        qs, rest = np.hsplit(rest, [QK_K // 4])
        scales, d = np.hsplit(rest, [12])
        d = d.view(np.float16).astype(np.float32)
        lscales, hscales = np.hsplit(scales, [8])
        lscales = lscales.reshape((n_blocks, 1, 8)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 2, 1))
        lscales = lscales.reshape((n_blocks, 16))
        hscales = hscales.reshape((n_blocks, 1, 4)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 4, 1))
        hscales = hscales.reshape((n_blocks, 16))
        scales = lscales & np.uint8(15) | (hscales & np.uint8(3)) << np.uint8(4)
        scales = (scales.astype(np.int8) - np.int8(32)).astype(np.float32)
        dl = (d * scales).reshape((n_blocks, 16, 1))
        ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = hmask.reshape(n_blocks, -1, 1, 32) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
        ql = ql.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(3)
        qh = qh.reshape((n_blocks, 16, QK_K // 16)) & np.uint8(1)
        qh = qh ^ np.uint8(1)
        q = (ql.astype(np.int8) - (qh << np.uint8(2)).astype(np.int8)).astype(np.float32)
        return (dl * q).reshape((n_blocks, QK_K))

class Q4_K:
    block_size = 256
    K_SCALE_SIZE = 12

    @staticmethod
    def get_scale_min(scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n_blocks = scales.shape[0]
        scales = scales.view(np.uint8)
        scales = scales.reshape((n_blocks, 3, 4))
        d, m, m_d = np.split(scales, 3, axis=-2)
        sc = np.concatenate([d & 63, m_d & 15 | d >> 2 & 48], axis=-1)
        min = np.concatenate([m & 63, m_d >> 4 | m >> 2 & 48], axis=-1)
        return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, qs = np.hsplit(rest, [cls.K_SCALE_SIZE])
        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)
        sc, m = Q4_K.get_scale_min(scales)
        d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
        dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))
        qs = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qs = (qs & np.uint8(15)).reshape((n_blocks, -1, 32)).astype(np.float32)
        return (d * qs - dm).reshape((n_blocks, QK_K))

class Q5_K:
    block_size = 256

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        d, rest = np.hsplit(blocks, [2])
        dmin, rest = np.hsplit(rest, [2])
        scales, rest = np.hsplit(rest, [Q4_K.K_SCALE_SIZE])
        qh, qs = np.hsplit(rest, [QK_K // 8])
        d = d.view(np.float16).astype(np.float32)
        dmin = dmin.view(np.float16).astype(np.float32)
        sc, m = Q4_K.get_scale_min(scales)
        d = (d * sc.astype(np.float32)).reshape((n_blocks, -1, 1))
        dm = (dmin * m.astype(np.float32)).reshape((n_blocks, -1, 1))
        ql = qs.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 8, 1))
        ql = (ql & np.uint8(15)).reshape((n_blocks, -1, 32))
        qh = (qh & np.uint8(1)).reshape((n_blocks, -1, 32))
        q = (ql | qh << np.uint8(4)).astype(np.float32)
        return (d * q - dm).reshape((n_blocks, QK_K))

class Q6_K:
    block_size = 256

    @classmethod
    def dequantize_blocks(cls, blocks: np.ndarray) -> np.ndarray:
        n_blocks = blocks.shape[0]
        ql, rest = np.hsplit(blocks, [QK_K // 2])
        qh, rest = np.hsplit(rest, [QK_K // 4])
        scales, d = np.hsplit(rest, [QK_K // 16])
        scales = scales.view(np.int8).astype(np.float32)
        d = d.view(np.float16).astype(np.float32)
        d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
        ql = ql.reshape((n_blocks, -1, 1, 64)) >> np.array([0, 4], dtype=np.uint8).reshape((1, 1, 2, 1))
        ql = (ql & np.uint8(15)).reshape((n_blocks, -1, 32))
        qh = qh.reshape((n_blocks, -1, 1, 32)) >> np.array([0, 2, 4, 6], dtype=np.uint8).reshape((1, 1, 4, 1))
        qh = (qh & np.uint8(3)).reshape((n_blocks, -1, 32))
        q = (ql | qh << np.uint8(4)).astype(np.int8) - np.int8(32)
        q = q.reshape((n_blocks, QK_K // 16, -1)).astype(np.float32)
        return (d * q).reshape((n_blocks, QK_K))


class Q8_1:
    @staticmethod
    def dequantize_blocks(blocks):
        # FP16 d, FP16 precomputed sum s, then 32 signed quantized values.
        d = blocks[:, :2].copy().view('<f2').astype(np.float32)
        return d * blocks[:, 4:].view(np.int8).astype(np.float32)

class Q8_K:
    @staticmethod
    def dequantize_blocks(blocks):
        # FP32 d, 256 signed quantized values, then unused 16 INT16 sums.
        d = blocks[:, :4].copy().view('<f4')
        return d * blocks[:, 4:260].view(np.int8).astype(np.float32)

class Q1_0:
    @staticmethod
    def dequantize_blocks(blocks):
        # Pinned ggml-quants.c: low-bit-first +/-d in each byte.
        d = blocks[:, :2].copy().view('<f2').astype(np.float32)
        q = np.unpackbits(blocks[:, 2:], axis=1, bitorder='little').astype(np.int8)
        return d * (q * 2 - 1).astype(np.float32)

class Q2_0:
    @staticmethod
    def dequantize_blocks(blocks):
        # Pinned ggml-quants.c: 00=-1, 01=0, 10=1, 11=2, low pair first.
        d = blocks[:, :2].copy().view('<f2').astype(np.float32)
        q = blocks[:, 2:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)
        q = (q & 3).astype(np.int8).reshape(len(blocks), 64) - 1
        return d * q.astype(np.float32)

COMMON_QUANTIZERS = {
    2: Q4_0, 3: Q4_1, 6: Q5_0, 7: Q5_1, 8: Q8_0, 9: Q8_1,
    10: Q2_K, 11: Q3_K, 12: Q4_K, 13: Q5_K, 14: Q6_K, 15: Q8_K,
    41: Q1_0, 42: Q2_0,
}
