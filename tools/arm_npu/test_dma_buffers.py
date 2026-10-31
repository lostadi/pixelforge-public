"""Cache ownership, coherency framing, and memory-pressure regression tests."""
import os
import array
from pathlib import Path
import socket
import struct
import sys
import unittest
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dma_buffers import DmaBufferCache, DMA_HEAP_IOCTL_ALLOC, DMA_BUF_IOCTL_SYNC


class DmaOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.kernel_open = os.open
        self.descriptors = []
        self.sync = []

    def open_heap(self, path, flags):
        self.assertEqual(path, '/dev/dma_heap/system')
        return self.kernel_open('/dev/null', flags)

    def ioctl(self, fd, operation, argument, mutate=False):
        if operation == DMA_HEAP_IOCTL_ALLOC:
            size, _, flags, heap_flags = struct.unpack('=QIIQ', argument)
            self.assertEqual(flags, os.O_RDWR | os.O_CLOEXEC)
            self.assertEqual(heap_flags, 0)
            descriptor = os.memfd_create('mock-dma', os.MFD_CLOEXEC)
            os.ftruncate(descriptor, size)
            self.descriptors.append(descriptor)
            argument[:] = struct.pack('=QIIQ', size, descriptor, flags, 0)
        elif operation == DMA_BUF_IOCTL_SYNC:
            self.sync.append(struct.unpack('=Q', argument)[0])
        else:
            self.fail('Unexpected device ioctl')
        return 0

    def test_exact_bytes_shared_and_survive_external_fd_owner(self):
        cache = DmaBufferCache(1, min_free_bytes=0, memory_available=lambda: 8*1024**2)
        values = np.arange(257, dtype='<f4').reshape(257, 1)
        with patch('dma_buffers.os.open', side_effect=self.open_heap), patch('dma_buffers.fcntl.ioctl', side_effect=self.ioctl):
            item = cache.put(('source-sha', 'weight', 0, 1), values)
        self.assertEqual(os.pread(item.fd, values.nbytes, 0), values.tobytes())
        self.assertEqual(self.sync, [2, 6])
        other_owner = os.dup(item.fd)
        cache.close()
        self.assertEqual(os.pread(other_owner, values.nbytes, 0), values.tobytes())
        os.close(other_owner)
        self.assertEqual(cache.resident_bytes, 0)

    def test_budget_overflow_streams_scan_tail_and_retains_admitted_weight(self):
        cache = DmaBufferCache(1, min_free_bytes=0, memory_available=lambda: 8*1024**2)
        with patch('dma_buffers.os.open', side_effect=self.open_heap), patch('dma_buffers.fcntl.ioctl', side_effect=self.ioctl):
            first = cache.put(('sourceA', 0), np.zeros(700*1024, np.uint8))
            second = cache.put(('sourceB', 0), np.ones(700*1024, np.uint8))
            third = cache.put(('sourceC', 0), np.ones(700*1024, np.uint8))
        self.assertGreaterEqual(first.fd, 0)
        self.assertFalse(first.transient)
        self.assertTrue(second.transient and third.transient)
        self.assertIs(cache.get(('sourceA', 0), 700*1024), first)
        self.assertIsNone(cache.get(('sourceB', 0), 700*1024))
        self.assertIsNone(cache.get(('sourceC', 0), 700*1024))
        self.assertEqual(cache.evictions, 0)
        self.assertEqual(cache.transient_allocations, 2)
        self.assertFalse(cache.admission_paused)
        second.close()
        third.close()
        cache.close()

    def test_physical_headroom_eviction_pauses_admission_for_lifetime(self):
        available = Mock(return_value=8*1024**2)
        cache = DmaBufferCache(1, min_free_bytes=1024**2, memory_available=available)
        with patch('dma_buffers.os.open', side_effect=self.open_heap), patch('dma_buffers.fcntl.ioctl', side_effect=self.ioctl):
            first = cache.put('A', np.zeros(256*1024, np.uint8))
            retained = cache.put('B', np.zeros(256*1024, np.uint8))
            available.side_effect = [0, 8*1024**2]
            temporary = cache.put('C', np.zeros(256*1024, np.uint8))
            available.side_effect = None
            later = cache.put('D', np.zeros(256*1024, np.uint8))
        self.assertEqual(first.fd, -1)
        self.assertIs(cache.get('B', 256*1024), retained)
        self.assertTrue(temporary.transient and later.transient)
        self.assertTrue(cache.admission_paused)
        self.assertEqual(cache.evictions, 1)
        self.assertEqual(cache.resident_bytes, 256*1024)
        self.assertIsNone(cache.get('C', 256*1024))
        temporary.close()
        later.close()
        cache.close()

    def test_absolute_bank_limit_still_fails_before_allocation(self):
        cache = DmaBufferCache(1, min_free_bytes=0, memory_available=lambda: 8*1024**2)
        cache.admission_paused = True
        with patch('dma_buffers.os.open') as device:
            with self.assertRaisesRegex(MemoryError, 'explicit DMA cache budget'):
                cache.put('oversize', np.zeros(1024**2+1, np.uint8))
            device.assert_not_called()

    def test_transient_z_parent_closes_after_ack_but_helper_duplicate_survives(self):
        from npu_linear import NPULinear
        cache = DmaBufferCache(1, min_free_bytes=0, memory_available=lambda: 8*1024**2)
        cache.admission_paused = True
        sender, receiver = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.addCleanup(sender.close)
        self.addCleanup(receiver.close)
        native = NPULinear.__new__(NPULinear)
        native.weight_buffers, native.fd_socket, native.dma_imports = cache, sender, 0
        native._write = Mock()
        original_fd = []
        def allocate(key, data):
            item = DmaBufferCache.put(cache, key, data)
            original_fd.append(item.fd)
            return item
        def acknowledge():
            frame, ancillary, flags, _ = receiver.recvmsg(256, socket.CMSG_SPACE(array.array('i').itemsize))
            self.assertEqual(frame, b'Z model 1 16\n')
            self.assertEqual(flags, 0)
            descriptors = array.array('i')
            descriptors.frombytes(ancillary[0][2])
            self.assertEqual(len(descriptors), 1)
            peer_fd = descriptors[0]
            self.addCleanup(os.close, peer_fd)
            original_fd.append(peer_fd)
            os.fstat(original_fd[0])  # Parent must still own it before ACK.
            return 'OK IMPORTED model 1 9'
        native._readline = acknowledge
        data = np.arange(16, dtype=np.uint8)
        with patch('dma_buffers.os.open', side_effect=self.open_heap), patch('dma_buffers.fcntl.ioctl', side_effect=self.ioctl), patch.object(cache, 'put', side_effect=allocate):
            self.assertEqual(native.upload_weight({'native_id':'model','weight_port_bytes':{1:16}},1,data,cache_key='sourceA'),9)
        with self.assertRaises(OSError):os.fstat(original_fd[0])
        self.assertEqual(os.pread(original_fd[1],16,0),data.tobytes())
        self.assertEqual(cache.resident_bytes,0)
        self.assertEqual(cache.transient_allocations,1)
        cache.close()

    def test_transient_upload_failures_always_close_parent_cached_fd_is_retained(self):
        from npu_linear import NPULinear
        for stage in ('sendmsg','short-sendmsg','write','ack','malformed-ack'):
            for transient in (True,False):
                with self.subTest(stage=stage,transient=transient):
                    cache = DmaBufferCache(1,min_free_bytes=0,memory_available=lambda:8*1024**2)
                    cache.admission_paused = transient
                    native = NPULinear.__new__(NPULinear)
                    native.weight_buffers,native.fd_socket,native.dma_imports = cache,Mock(),0
                    native.fd_socket.sendmsg.return_value = len(b'Z model 1 16\n')
                    native._write = Mock()
                    native._readline = Mock(return_value='not an import acknowledgement')
                    if stage=='sendmsg':native.fd_socket.sendmsg.side_effect=OSError('injected send failure')
                    if stage=='short-sendmsg':native.fd_socket.sendmsg.return_value=0
                    if stage=='write':native._write.side_effect=OSError('injected write failure')
                    if stage=='ack':native._readline.side_effect=RuntimeError('injected ACK failure')
                    with patch('dma_buffers.os.open',side_effect=self.open_heap),patch('dma_buffers.fcntl.ioctl',side_effect=self.ioctl):
                        with self.assertRaises((OSError,RuntimeError)):
                            native.upload_weight({'native_id':'model','weight_port_bytes':{1:16}},1,
                                np.zeros(16,np.uint8),cache_key='sourceA')
                    descriptor=self.descriptors[-1]
                    if transient:
                        with self.assertRaises(OSError):os.fstat(descriptor)
                        self.assertIsNone(cache.get('sourceA',16))
                    else:
                        os.fstat(descriptor)
                        self.assertEqual(cache.get('sourceA',16).fd,descriptor)
                    cache.close()

    def test_ample_budget_reuses_the_original_six_allocations(self):
        cache=DmaBufferCache(1,min_free_bytes=0,memory_available=lambda:8*1024**2)
        with patch('dma_buffers.os.open',side_effect=self.open_heap),patch('dma_buffers.fcntl.ioctl',side_effect=self.ioctl):
            items=[cache.put(('source',i),np.full(16,i,np.uint8)) for i in range(6)]
            for i,item in enumerate(items):self.assertIs(cache.get(('source',i),16),item)
        self.assertEqual(cache.allocations,6)
        self.assertEqual(cache.transient_allocations,0)
        self.assertEqual(cache.evictions,0)
        self.assertFalse(cache.admission_paused)
        cache.close()

    def test_headroom_failure_happens_before_allocation(self):
        cache = DmaBufferCache(1, min_free_bytes=1024**2, memory_available=lambda: 1024**2)
        with patch('dma_buffers.os.open') as device:
            with self.assertRaises(MemoryError):
                cache.put(('sourceA', 0), np.zeros(1, np.uint8))
            device.assert_not_called()

    def test_changed_dimensions_never_reuse_old_buffer(self):
        cache = DmaBufferCache(1, min_free_bytes=0, memory_available=lambda: 8*1024**2)
        with patch('dma_buffers.os.open', side_effect=self.open_heap), patch('dma_buffers.fcntl.ioctl', side_effect=self.ioctl):
            cache.put(('sourceA', 0), np.zeros(4, np.uint8))
        try:
            with self.assertRaises(ValueError):
                cache.get(('sourceA', 0), 8)
        finally:
            cache.close()


    def test_resident_bytes_track_page_rounding_reuse_eviction_and_close(self):
        cache = DmaBufferCache(1, min_free_bytes=0, memory_available=lambda: 8*1024**2)
        self.addCleanup(cache.close)
        small = np.ones(1, np.uint8)
        larger = np.ones(4097, np.uint8)
        with patch('dma_buffers.os.open', side_effect=self.open_heap), patch('dma_buffers.fcntl.ioctl', side_effect=self.ioctl):
            first = cache.put('small', small)
            self.assertEqual(cache.resident_bytes, 4096)
            cache.put('larger', larger)
            self.assertEqual(cache.resident_bytes, 4096 + 8192)
            self.assertIs(cache.put('small', small), first)
            self.assertEqual(cache.resident_bytes, 4096 + 8192)
        # Reusing small moved it to the end; the larger allocation leaves first.
        self.assertTrue(cache.evict_oldest())
        self.assertEqual(cache.resident_bytes, 4096)
        self.assertEqual(cache.stats()['resident_bytes'], 4096)
        cache.close()
        self.assertEqual(cache.resident_bytes, 0)
        self.assertFalse(cache.evict_oldest())
        cache.close()
        self.assertEqual(cache.resident_bytes, 0)

    def test_failed_device_allocation_or_sync_never_charges_resident_bytes(self):
        for failure in ('allocate', 'sync-start', 'sync-end'):
            with self.subTest(failure=failure):
                cache = DmaBufferCache(1, min_free_bytes=0, memory_available=lambda: 8*1024**2)
                try:
                    with patch('dma_buffers.os.open', side_effect=self.open_heap), patch('dma_buffers.fcntl.ioctl', side_effect=self.ioctl):
                        retained = cache.put('retained', np.zeros(1, np.uint8))
                    previous_descriptors = len(self.descriptors)
                    def failing_ioctl(fd, operation, argument, mutate=False):
                        if operation == DMA_HEAP_IOCTL_ALLOC and failure == 'allocate':
                            raise OSError('injected allocation failure')
                        if operation == DMA_BUF_IOCTL_SYNC:
                            sync = struct.unpack('=Q', argument)[0]
                            if (failure == 'sync-start' and sync == 2) or (failure == 'sync-end' and sync == 6):
                                raise OSError('injected sync failure')
                        return self.ioctl(fd, operation, argument, mutate)
                    with patch('dma_buffers.os.open', side_effect=self.open_heap), patch('dma_buffers.fcntl.ioctl', side_effect=failing_ioctl):
                        with self.assertRaises(OSError):
                            cache.put('failed', np.zeros(8193, np.uint8))
                    self.assertEqual(cache.resident_bytes, 4096)
                    self.assertEqual(cache.allocations, 1)
                    self.assertIs(cache.get('retained', 1), retained)
                    self.assertIsNone(cache.get('failed', 8193))
                    for fd in self.descriptors[previous_descriptors:]:
                        with self.assertRaises(OSError):
                            os.fstat(fd)
                finally:
                    cache.close()
                self.assertEqual(cache.resident_bytes, 0)


if __name__ == '__main__':
    unittest.main()
