"""Real SCM_RIGHTS/FD lifecycle with mock DMA sync and mock vendor APIs only."""
import array
import mmap
import os
import socket
import unittest

import test_npu_pipe_cache as _cache


class Memfd:
    def __init__(self, value=5, *, size=4096, name='pixelforge-dma-test'):
        self.fd = os.memfd_create(name, os.MFD_CLOEXEC)
        os.ftruncate(self.fd, size)
        with mmap.mmap(self.fd, size) as memory:
            memory[:16] = bytes([value]) * 16
    def close(self):
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1
    def __enter__(self):
        return self
    def __exit__(self, *_):
        self.close()


@unittest.skipUnless(hasattr(os, 'memfd_create') and hasattr(socket, 'SOCK_SEQPACKET'),
                     'Linux FD transport fixture')
class NativeDmaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _cache.NativeRuntimeCacheTests.setUpClass.__func__(cls)
    @classmethod
    def tearDownClass(cls):
        _cache.NativeRuntimeCacheTests.tearDownClass.__func__(cls)

    def session(self, *, mode='protocol', kind=socket.SOCK_SEQPACKET, budget=None, limit=None, deferred=None):
        sender, child = socket.socketpair(socket.AF_UNIX, kind)
        self.addCleanup(sender.close)
        env = dict(os.environ, PIXELFORGE_NPU_FD_SOCKET=str(child.fileno()))
        if limit is not None:
            env['PIXELFORGE_NPU_MAX_RESIDENT_MODELS'] = str(limit)
        if deferred is not None:
            env['PIXELFORGE_NPU_DEFER_WEIGHT_ALLOCATION'] = str(deferred)
        try:
            wire = _cache.ProtocolSession(self, self.binary, env=env,
                                          pass_fds=(child.fileno(),), mode=mode)
        finally:
            child.close()
        self.addCleanup(wire.close)
        if budget is not None:
            wire.send(f'B {budget}\n'.encode())
            self.assertEqual(wire.line(), f'OK CACHE {budget}\n'.encode())
        return wire, sender

    def load(self, wire, name='a', ports=2):
        wire.send(_cache.NativeRuntimeCacheTests.load(name, inputs=[16] * ports))
        self.assertEqual(wire.line(), f'OK LOADED {name} 12\n'.encode())

    def queue(self, sender, header, descriptors=()):
        ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array('i', descriptors))] if descriptors else []
        sender.sendmsg([header], ancillary)

    def imported(self, wire, sender, donor, *, name='a', index=1):
        header = f'Z {name} {index} 16\n'.encode()
        self.queue(sender, header, [donor.fd])
        wire.send(header)
        fields = wire.line().split()
        self.assertEqual(fields[:4], [b'OK', b'IMPORTED', name.encode(), str(index).encode()])
        self.assertEqual(len(fields), 5)
        return int(fields[4])

    def test_import_aliases_descriptor_and_survives_sender_close(self):
        wire, sender = self.session()
        with wire, Memfd(5) as donor:
            wire.send(b'I CAPABILITIES\n')
            self.assertIn(b' Z DMA_BUF_IMPORT_V1\n', wire.line())
            self.load(wire)
            generation = self.imported(wire, sender, donor)
            donor.close()
            self.assertEqual(wire.info(), dict(generation=generation, loaded=1, ports=2, ready='01'))
            for value in (1, 7):
                self.assertEqual(wire.invoke(generation, bytes([value]) * 16), bytes([value + 5]) * 16)
            wire.send(b'Q\n')
            self.assertEqual(wire.line(), b'OK QUIT\n')
            stats = wire.wait()
            self.assertIn(b'dma_register=1 dma_unregister=1 fd_close=1', stats)

    def test_multiport_import_replace_and_lru_reimport(self):
        wire, sender = self.session(budget=16396)
        with wire, Memfd(3) as first, Memfd(5) as second:
            self.load(wire, ports=3)
            one = self.imported(wire, sender, first)
            two = self.imported(wire, sender, second, index=2)
            self.assertGreater(two, one)
            self.assertEqual(wire.info()['ready'], '011')
            self.assertEqual(wire.invoke(two, bytes([9]) * 16), bytes([17]) * 16)
            replaced = self.imported(wire, sender, second)
            self.assertGreater(replaced, two)
            self.assertEqual(wire.invoke(replaced, bytes([9]) * 16), bytes([19]) * 16)
            self.load(wire, 'b', ports=3)
            state = wire.info()
            self.assertEqual((state['loaded'], state['ready']), (0, '000'))
            self.assertGreater(state['generation'], replaced)
            again = self.imported(wire, sender, first)
            last = self.imported(wire, sender, second, index=2)
            self.assertGreater(last, again)
            self.assertEqual(wire.invoke(last, bytes([1]) * 16), bytes([9]) * 16)
            wire.send(b'Q\n')
            self.assertEqual(wire.line(), b'OK QUIT\n')
            stats = wire.wait()
            self.assertIn(b'dma_register=5 dma_unregister=5', stats)

    def test_extra_missing_truncated_and_mismatched_fd_packets_close_every_fd(self):
        for variant, count in [('missing', 0), ('extra', 2), ('truncated', 24), ('mismatch', 1), ('payload', 1)]:
            with self.subTest(variant=variant):
                wire, sender = self.session()
                with wire, Memfd() as donor:
                    self.load(wire)
                    header = b'Z a 1 16\n'
                    packet = b'Z b 1 16\n' if variant == 'mismatch' else header
                    if variant == 'payload':
                        packet = b'x' * 1024
                    self.queue(sender, packet, [donor.fd] * count)
                    wire.send(header)
                    self.assertEqual(wire.line(), b'ERR DMA_FRAME\n')
                    stats = wire.wait(2)
                    self.assertIn(b'dma_register=0 dma_unregister=0', stats)
                    received = min(count, 16)
                    self.assertIn(f'fd_close={received}'.encode(), stats)

    def test_fd_size_type_and_dma_sync_fail_before_vendor_registration(self):
        for size, name, mode in [(8192, 'pixelforge-dma-test', 'protocol'),
                                 (4096, 'ordinary-memfd', 'protocol'),
                                 (4096, 'pixelforge-dma-test', 'protocol-dma-sync'),
                                 (4096, 'pixelforge-dma-test', 'protocol-dma-sync-end')]:
            with self.subTest(size=size, name=name, mode=mode):
                wire, sender = self.session(mode=mode)
                with wire, Memfd(size=size, name=name) as donor:
                    self.load(wire)
                    header = b'Z a 1 16\n'
                    self.queue(sender, header, [donor.fd]);wire.send(header)
                    self.assertEqual(wire.line(), b'ERR DMA_BUFFER\n')
                    self.assertIn(b'dma_register=0 dma_unregister=0 fd_close=1', wire.wait(2))

    def test_imported_ports_reject_legacy_raw_or_upload_writes(self):
        for command in (b'R a\n' + b'x' * 32, b'U a 1\n' + b'x' * 16):
            with self.subTest(command=command[:1]):
                wire, sender = self.session()
                with wire, Memfd() as donor:
                    self.load(wire);self.imported(wire, sender, donor)
                    wire.send(command)
                    self.assertEqual(wire.line(), b'ERR DMA_LEGACY_INPUT\n')
                    stats = wire.wait(2)
                    self.assertIn(b'invoke=0', stats)
                    self.assertIn(b'dma_register=1 dma_unregister=1 fd_close=1', stats)

    def test_bad_socket_type_and_header_size_are_rejected(self):
        wire, _ = self.session(kind=socket.SOCK_STREAM)
        with wire:
            self.assertEqual(wire.line(), b'ERR DMA_SOCKET\n')
            wire.wait(2)
        wire, sender = self.session()
        with wire, Memfd() as donor:
            self.load(wire)
            header = b'Z a 1 15\n'
            self.queue(sender, header, [donor.fd]);wire.send(header)
            self.assertEqual(wire.line(), b'ERR DMA_PORT_SIZE\n')
            self.assertIn(b'dma_register=0', wire.wait(2))

    def test_resident_context_cap_evicts_by_count_and_invalidates_generation(self):
        wire, sender = self.session(limit=1)
        with wire, Memfd(5) as donor:
            wire.send(b'I CACHE LIMIT\n')
            self.assertEqual(wire.line(), b'OK CACHE_LIMIT MODELS 1\n')
            self.load(wire)
            old = self.imported(wire, sender, donor)
            self.load(wire, 'b')  # Only 24KiB would fit byte budget; cap still evicts.
            state = wire.info()
            self.assertEqual((state['loaded'], state['ready']), (0, '00'))
            self.assertGreater(state['generation'], old)
            current = self.imported(wire, sender, donor)
            self.assertGreater(current, old)
            self.assertEqual(wire.invoke(current, bytes([1]) * 16), bytes([6]) * 16)
            wire.send(b'Q\n')
            self.assertEqual(wire.line(), b'OK QUIT\n')
            self.assertIn(b'dma_register=2 dma_unregister=2', wire.wait())

    def test_invalid_resident_context_limit_is_rejected(self):
        for limit in ('', '0', '-1', '+1', '65', '2x', ' 2'):
            with self.subTest(limit=limit):
                wire, _ = self.session(limit=limit)
                with wire:
                    self.assertEqual(wire.line(), b'ERR CACHE_MODEL_LIMIT\n')
                    wire.wait(2)

    def test_deferred_weights_allocate_only_activation_and_output_until_every_import(self):
        wire, sender = self.session(deferred=1)
        with wire, Memfd(3) as first, Memfd(5) as second:
            wire.send(b'I CAPABILITIES\n')
            self.assertIn(b' DEFERRED_DMA_WEIGHTS_V1\n', wire.line())
            self.load(wire, ports=3)
            generation = self.imported(wire, sender, first)
            self.assertEqual(wire.info()['ready'], '010')
            generation = self.imported(wire, sender, second, index=2)
            wire.send(b'I a\nI CACHE\n')
            self.assertEqual(wire.line(), b'OK INFO 12 0\n')  # No invocation before V.
            self.assertEqual(wire.line().split()[3:5], [b'16396', b'1'])
            self.assertEqual(wire.invoke(generation, bytes([9]) * 16), bytes([17]) * 16)
            self.assertEqual(wire.invoke(generation, bytes([1]) * 16), bytes([9]) * 16)
            wire.send(b'I a\nQ\n')
            self.assertEqual(wire.line(), b'OK INFO 12 1\n')
            self.assertEqual(wire.line(), b'OK QUIT\n')
            stats = wire.wait()
            self.assertIn(b'load=2 graph=2 inv=1 alloc=4 invoke=2', stats)
            self.assertIn(b'dma_register=2 dma_unregister=2 fd_close=2', stats)

    def test_deferred_validation_and_partial_import_never_allocate_weight_ahwb_or_invoke(self):
        wire, _ = self.session(deferred=1)
        with wire:
            self.load(wire, ports=4)
            wire.send(b'Q\n')
            self.assertEqual(wire.line(), b'OK QUIT\n')
            self.assertIn(b'inv=0 alloc=2 invoke=0', wire.wait())
        wire, sender = self.session(deferred=1)
        with wire, Memfd() as donor:
            self.load(wire, ports=3)
            generation = self.imported(wire, sender, donor)
            wire.send(f'V a {generation}\n'.encode())  # Header alone must fail before a payload read.
            self.assertEqual(wire.line(), b'ERR STICKY_STATE\n')
            stats = wire.wait(2)
            self.assertIn(b'inv=0 alloc=4 invoke=0', stats)
            self.assertIn(b'dma_register=1 dma_unregister=1 fd_close=1', stats)

    def test_deferred_legacy_headers_fail_before_payload_but_constant_raw_still_works(self):
        for command in (b'R a\n', b'U a 1\n'):
            with self.subTest(command=command):
                wire, _ = self.session(deferred=1)
                with wire:
                    self.load(wire)
                    wire.send(command)
                    self.assertEqual(wire.line(), b'ERR DMA_LEGACY_INPUT\n')
                    self.assertIn(b'inv=0 alloc=2 invoke=0', wire.wait(2))
        wire, _ = self.session(deferred=1)
        with wire:
            self.load(wire, ports=1)
            wire.send(b'R a\n' + b'x' * 16)
            self.assertEqual(wire.output(), b'x' * 16)
            wire.send(b'Q\n')
            self.assertEqual(wire.line(), b'OK QUIT\n')
            self.assertIn(b'inv=1 alloc=4 invoke=1', wire.wait())

    def test_deferred_context_eviction_reimports_and_rejects_old_generation(self):
        wire, sender = self.session(deferred=1, limit=1)
        with wire, Memfd() as donor:
            self.load(wire)
            previous = self.imported(wire, sender, donor)
            self.assertEqual(wire.invoke(previous, b'a' * 16), b'f' * 16)
            self.load(wire, 'b')
            self.assertEqual(wire.info()['ready'], '00')
            current = self.imported(wire, sender, donor)
            self.assertGreater(current, previous)
            self.assertEqual(wire.invoke(current, b'b' * 16), b'g' * 16)
            wire.send(f'V a {previous}\n'.encode())
            self.assertEqual(wire.line(), b'ERR STICKY_STATE\n')
            stats = wire.wait(2)
            self.assertIn(b'inv=2 alloc=8 invoke=2', stats)
            self.assertIn(b'dma_register=2 dma_unregister=2 fd_close=2', stats)

    def test_deferred_mode_requires_an_authenticated_socket_and_strict_boolean(self):
        wire = _cache.ProtocolSession(self, self.binary,
            env=dict(os.environ, PIXELFORGE_NPU_DEFER_WEIGHT_ALLOCATION='1'))
        with wire:
            self.assertEqual(wire.line(), b'ERR DMA_SOCKET\n')
            wire.wait(2)
        for value in ('', '-1', '+1', '2', 'true', ' 1'):
            with self.subTest(value=value):
                wire, _ = self.session(deferred=value)
                with wire:
                    self.assertEqual(wire.line(), b'ERR DMA_SOCKET\n')
                    wire.wait(2)


if __name__ == '__main__':
    unittest.main()
