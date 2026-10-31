"""Owned DMA-heap buffers shared with the NPU helper by file descriptor.

The bounded cache survives helper invocation-context eviction. CPU writes use
the kernel DMA coherency protocol; files and memfd are never substitutes for
device DMA buffers. The model file is opened only by the GGUF reader.
"""
from collections import OrderedDict
from dataclasses import dataclass
import fcntl
import mmap
import os
import struct
import time

import numpy as np

MIB = 1024 * 1024
DMA_HEAP_IOCTL_ALLOC = 0xc0184800
DMA_BUF_IOCTL_SYNC = 0x40086200


@dataclass
class WeightBuffer:
    fd: int
    bytes: int
    allocation_bytes: int
    transient: bool = False

    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class DmaBufferCache:
    def __init__(self, budget_mib, *, min_free_bytes, memory_available):
        if type(budget_mib) is not int or not 1 <= budget_mib <= 6144:
            raise ValueError('DMA weight buffer cache must be 1 to 6144 MiB')
        self.budget = budget_mib * MIB
        self.min_free = min_free_bytes
        self.memory_available = memory_available
        self.buffers = OrderedDict()
        self._resident_bytes = 0
        self.allocations = self.hits = self.evictions = 0
        self.admission_paused = False
        self.transient_allocations = 0
        self.allocate_seconds = 0.

    @property
    def resident_bytes(self):
        return self._resident_bytes

    def get(self, key, expected_bytes):
        item = self.buffers.get(key)
        if item is None:
            return None
        if item.bytes != expected_bytes or item.fd < 0:
            raise ValueError('DMA cache identity reused with different buffer dimensions')
        self.buffers.move_to_end(key)
        self.hits += 1
        return item

    def evict_oldest(self):
        # Memory-pressure reclamation must not immediately refill the freed
        # space while a cyclic model scan continues through uncached weights.
        self.admission_paused = True
        if not self.buffers:
            return False
        _, item = self.buffers.popitem(last=False)
        self._resident_bytes -= item.allocation_bytes
        item.close()
        self.evictions += 1
        return True

    def put(self, key, array):
        data = np.ascontiguousarray(array)
        view = memoryview(data).cast('B')
        size = len(view)
        if size < 1:
            raise ValueError('Cannot allocate an empty device weight buffer')
        existing = self.get(key, size)
        if existing is not None:
            return existing
        rounded = (size + 4095) // 4096 * 4096
        if rounded > self.budget:
            raise MemoryError('Weight buffer exceeds the explicit DMA cache budget')
        # Keep already admitted immutable buffers. Overflow is streamed with
        # a temporary parent FD instead of evicting the entire retained set.
        available = self.memory_available()
        while available is not None and available < self.min_free + rounded and self.buffers:
            self.evict_oldest()
            available = self.memory_available()
        if available is not None and available < self.min_free + rounded:
            raise MemoryError('Insufficient kernel memory headroom for a shared weight buffer')
        transient = self.admission_paused or self.resident_bytes + rounded > self.budget
        began = time.monotonic()
        descriptor = -1
        heap = os.open('/dev/dma_heap/system', os.O_RDONLY | os.O_CLOEXEC)
        try:
            request = bytearray(struct.pack('=QIIQ', rounded, 0, os.O_RDWR | os.O_CLOEXEC, 0))
            fcntl.ioctl(heap, DMA_HEAP_IOCTL_ALLOC, request, True)
            _, descriptor, _, _ = struct.unpack('=QIIQ', request)
            with mmap.mmap(descriptor, rounded, flags=mmap.MAP_SHARED,
                           prot=mmap.PROT_READ | mmap.PROT_WRITE) as mapped:
                fcntl.ioctl(descriptor, DMA_BUF_IOCTL_SYNC, struct.pack('=Q', 2))
                try:
                    mapped[:size] = view
                finally:
                    fcntl.ioctl(descriptor, DMA_BUF_IOCTL_SYNC, struct.pack('=Q', 6))
            item = WeightBuffer(descriptor, size, rounded, transient=transient)
            if transient:
                self.transient_allocations += 1
            else:
                self.buffers[key] = item
                self._resident_bytes += rounded
            descriptor = -1
            self.allocations += 1
            self.allocate_seconds += time.monotonic() - began
            return item
        finally:
            os.close(heap)
            if descriptor >= 0:
                os.close(descriptor)

    def stats(self):
        return dict(budget_bytes=self.budget, resident_bytes=self.resident_bytes,
                    resident_buffers=len(self.buffers), allocations=self.allocations,
                    hits=self.hits, evictions=self.evictions,
                    allocate_seconds=self.allocate_seconds,
                    admission_paused=self.admission_paused,
                    transient_allocations=self.transient_allocations)

    def close(self):
        while self.buffers:
            _, item = self.buffers.popitem(last=False)
            self._resident_bytes -= item.allocation_bytes
            item.close()
