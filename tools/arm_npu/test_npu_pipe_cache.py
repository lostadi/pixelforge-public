"""Compile the actual helper against mock vendor APIs; no Android/NPU access."""
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import struct
import tempfile
import unittest

KIT = Path(__file__).resolve().parent


class ProtocolSession:
    """Bounded interactive access to the compiled helper with mock vendor APIs."""
    def __init__(self, test, binary, *, env=None, pass_fds=(), mode='protocol'):
        self.test = test
        if env is None:
            env = dict(os.environ)
            env.pop('PIXELFORGE_NPU_FD_SOCKET', None)
        self.process = subprocess.Popen([str(binary), mode], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
                                        env=env, pass_fds=pass_fds)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = bytearray()
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if not self.process.stdin.closed:
                self.process.stdin.close()
            self.process.wait(timeout=2)
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            self.process.kill()
            self.process.wait(timeout=2)
        finally:
            self.selector.close()
            for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
                stream.close()

    def send(self, value):
        self.process.stdin.write(value)
        self.process.stdin.flush()

    def receive(self):
        self.test.assertTrue(self.selector.select(5), 'Mock helper response timed out')
        value = os.read(self.process.stdout.fileno(), 65536)
        self.test.assertTrue(value, 'Mock helper ended before its complete response')
        self.buffer.extend(value)

    def line(self):
        while b'\n' not in self.buffer:
            self.receive()
        end = self.buffer.index(b'\n') + 1
        result = bytes(self.buffer[:end])
        del self.buffer[:end]
        return result

    def output(self):
        fields = self.line().split()
        self.test.assertEqual(fields[:3], [b'OK', b'OUTPUT', b'1'])
        count = int(fields[3])
        while len(self.buffer) < count:
            self.receive()
        result = bytes(self.buffer[:count])
        del self.buffer[:count]
        return result

    def info(self, name='a'):
        self.send(f'I MODEL {name}\n'.encode())
        fields = self.line().split()
        self.test.assertEqual(fields[:3], [b'OK', b'MODEL', name.encode()])
        self.test.assertEqual(len(fields), 7)
        return dict(generation=int(fields[3]), loaded=int(fields[4]),
                    ports=int(fields[5]), ready=fields[6].decode())

    def upload(self, index, data, name='a'):
        self.send(f'U {name} {index}\n'.encode() + data)
        fields = self.line().split()
        self.test.assertEqual(fields[:4], [b'OK', b'UPLOADED', name.encode(), str(index).encode()])
        self.test.assertEqual(len(fields), 5)
        return int(fields[4])

    def invoke(self, generation, data, name='a'):
        self.send(f'V {name} {generation}\n'.encode() + data)
        return self.output()

    def wait(self, expected=0):
        if not self.process.stdin.closed:
            self.process.stdin.close()
        self.test.assertEqual(self.process.wait(timeout=5), expected)
        return self.process.stderr.read()

    def tail(self):
        self.test.assertIsNotNone(self.process.poll())
        return bytes(self.buffer) + self.process.stdout.read()


class NativeRuntimeCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which(os.environ.get('CXX', 'c++'))
        if not compiler:
            raise unittest.SkipTest('C++17 compiler unavailable')
        cls.work = tempfile.TemporaryDirectory(prefix='pixelforge-mock-device-cache-')
        cls.binary = Path(cls.work.name) / 'test-cache'
        try:
            subprocess.run([compiler, '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                            '-D__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__',
                            '-I', str(KIT / 'test_stubs'), str(KIT / 'test_npu_pipe_cache.cc'),
                            '-ldl', '-o', str(cls.binary)], check=True, capture_output=True,
                           text=True, timeout=60)
        except BaseException:
            cls.work.cleanup()
            raise

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def run_case(self, scenario, expected=0, error=None):
        result = subprocess.run([str(self.binary), scenario], capture_output=True,
                                text=True, timeout=5)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        if error:
            self.assertEqual(result.stdout, f'ERR {error}\n')
            self.assertIn('FAIL ', result.stderr)
        else:
            self.assertIn('PASS NO_VENDOR_INITIALIZATION NO_NPU_INVOCATIONS', result.stdout)
        return result

    def test_device_resources_and_invocation_reused(self):
        self.run_case('reuse')

    def test_lru_eviction_holds_budget(self):
        self.run_case('lru')

    def test_batch_io_page_rounding_in_budget(self):
        self.run_case('batch-accounting')

    def test_zero_and_oversize_are_transient(self):
        for case in ('zero', 'oversize'):
            with self.subTest(case=case):
                self.run_case(case)

    def test_shrink_free_replacement_shutdown(self):
        self.run_case('shrink-free-replace')

    def test_load_validation_remains_transient(self):
        self.run_case('validate')

    def test_partial_failure_releases_resources_and_can_retry(self):
        for api in ('load', 'graph', 'node', 'assign', 'edge', 'connect', 'set-edge',
                    'allocate', 'register', 'invocation', 'attach'):
            with self.subTest(api=api):
                self.run_case('partial-' + api)

    def test_prepare_failure_does_not_invoke(self):
        self.run_case('prepare-error')

    def test_uncertain_vendor_ownership_exits_before_dependents_are_freed(self):
        cases = {'detach': ('BUFFER_DETACH', 'inv=1 ahb=2 reg=2'),
                 'delete-invocation': ('INVOCATION_DELETE', 'inv=1 ahb=2 reg=2'),
                 'unregister': ('BUFFER_UNREGISTER', 'inv=0 ahb=2 reg=2'),
                 'delete-graph': ('GRAPH_DELETE', 'graph=1 inv=0 ahb=0 reg=0'),
                 'unload': ('SQ_UNLOAD', 'sq=1 graph=0 inv=0 ahb=0 reg=0'),
                 'unlock': ('BUFFER_UNLOCK', 'inv=1 ahb=2 reg=2'),
                 'invoke': ('INVOKE', 'inv=1 ahb=2 reg=2'),
                 'wait': ('INVOKE', 'inv=1 ahb=2 reg=2')}
        for api, (error, retained) in cases.items():
            with self.subTest(api=api):
                result = self.run_case('fatal-' + api, 2, error)
                self.assertIn(retained, result.stderr)

    def protocol(self, payload, *, mode='protocol', expected=0):
        result = subprocess.run([str(self.binary), mode], input=payload,
                                capture_output=True, timeout=5)
        self.assertEqual(result.returncode, expected, repr(result.stdout) + repr(result.stderr))
        return result

    @staticmethod
    def load(name='a', inputs=16, outputs=16):
        code = struct.pack('<I4sI', 8, b'DGC0', 0)
        inputs = [inputs] if isinstance(inputs, int) else list(inputs)
        outputs = [outputs] if isinstance(outputs, int) else list(outputs)
        sizes = ' '.join(map(str, inputs + outputs))
        return f'L {name} {len(inputs)} {len(outputs)} {len(code)} {sizes}\n'.encode() + code

    def sticky_session(self, ports=2, budget=None):
        wire = ProtocolSession(self, self.binary)
        self.addCleanup(wire.close)
        if budget is not None:
            wire.send(f'B {budget}\n'.encode())
            self.assertEqual(wire.line(), f'OK CACHE {budget}\n'.encode())
        wire.send(self.load(inputs=[16] * ports))
        self.assertEqual(wire.line(), b'OK LOADED a 12\n')
        return wire

    def test_protocol_reuse_cache_inspect_free_and_quit(self):
        first, second = bytes(range(16)), bytes(range(16, 32))
        result = self.protocol(b'I CAPABILITIES\nB 8204\n' + self.load() +
                               b'I a\nR a\n' + first + b'R a\n' + second +
                               b'I a\nI CACHE\nF a\nI CACHE\nQ\n')
        self.assertEqual(result.stdout,
                         b'OK CAPABILITIES C R F I D L Q B PERSISTENT_RUNTIME U V STICKY_INPUTS_V1 RESIDENT_MODEL_LIMIT_V1\n'
                         b'OK CACHE 8204\nOK LOADED a 12\nOK INFO 12 0\n'
                         b'OK OUTPUT 1 16\n' + first + b'OK OUTPUT 1 16\n' + second +
                         b'OK INFO 12 1\nOK CACHE 8204 8204 1 1 1 0\n'
                         b'OK FREED\nOK CACHE 8204 0 0 1 1 0\nOK QUIT\n')
        self.assertIn(b'load=2 graph=2 inv=1 alloc=4 invoke=2', result.stderr)

    def test_protocol_zero_budget_load_validation_evicts_and_eof_cleans(self):
        tensor = b'a' * 16
        result = self.protocol(b'B 8204\n' + self.load() + b'R a\n' + tensor +
                               self.load('b') + b'I CACHE\nB 0\nR a\n' + tensor +
                               b'I CACHE\nR a\n' + tensor + b'I CACHE\n')
        self.assertIn(b'OK CACHE 8204 0 0 1 0 1\n', result.stdout)
        self.assertIn(b'OK CACHE 0 0 0 2 0 1\n', result.stdout)
        self.assertTrue(result.stdout.endswith(b'OK CACHE 0 0 0 3 0 1\n'))
        self.assertIn(b'load=5 graph=5 inv=3 alloc=10 invoke=3', result.stderr)

    def test_protocol_rejects_invalid_budget_and_failed_invoke(self):
        for budget in (b'-1', b'+1', b'2147483649', b'1 extra', b''):
            with self.subTest(budget=budget):
                result = self.protocol(b'B ' + budget + b'\n', expected=2)
                self.assertEqual(result.stdout, b'ERR CACHE_HEADER\n')
        result = self.protocol(self.load() + b'R a\n' + b'a' * 16,
                               mode='protocol-wait', expected=2)
        self.assertEqual(result.stdout, b'OK LOADED a 12\nERR INVOKE\n')
        self.assertIn(b'inv=1 ahb=2 reg=2', result.stderr)

    def test_sticky_weights_uploaded_once_reused_with_new_activations(self):
        with self.sticky_session() as wire:
            cold = wire.info()
            self.assertEqual((cold['loaded'], cold['ports'], cold['ready']), (0, 2, '00'))
            generation = wire.upload(1, bytes([5]) * 16)
            hot = wire.info()
            self.assertGreater(generation, cold['generation'])
            self.assertEqual(hot, dict(generation=generation, loaded=1, ports=2, ready='01'))
            for value in (1, 7, 22):
                self.assertEqual(wire.invoke(generation, bytes([value]) * 16), bytes([value + 5]) * 16)
                self.assertEqual(wire.info(), hot)
            wire.send(b'I a\nI CACHE\nQ\n')
            self.assertEqual(wire.line(), b'OK INFO 12 1\n')
            cache = wire.line().split()
            self.assertEqual(cache[3:5], [b'12300', b'1'])  # Includes weight AHWB.
            self.assertEqual(wire.line(), b'OK QUIT\n')
            self.assertIn(b'load=2 graph=2 inv=1 alloc=6 invoke=3', wire.wait())

    def test_multiple_sticky_weight_ports_and_replacement_generation(self):
        with self.sticky_session(ports=3) as wire:
            first = wire.upload(1, bytes([3]) * 16)
            self.assertEqual(wire.info()['ready'], '010')
            second = wire.upload(2, bytes([5]) * 16)
            self.assertGreater(second, first)
            self.assertEqual(wire.info()['ready'], '011')
            self.assertEqual(wire.invoke(second, bytes([9]) * 16), bytes([17]) * 16)
            third = wire.upload(1, bytes([11]) * 16)
            self.assertGreater(third, second)
            self.assertEqual(wire.info()['ready'], '011')
            self.assertEqual(wire.invoke(third, bytes([9]) * 16), bytes([25]) * 16)
            wire.send(f'V a {second}\n'.encode() + b'X' * 16 + b'I CACHE\nQ\n')
            self.assertEqual(wire.line(), b'ERR STICKY_STATE\n')
            self.assertIn(b'invoke=2', wire.wait(2))
            self.assertEqual(wire.tail(), b'')

    def test_sticky_invoke_rejects_an_unuploaded_weight_port(self):
        with self.sticky_session(ports=3) as wire:
            generation = wire.upload(1, b'w' * 16)
            wire.send(f'V a {generation}\n'.encode() + b'X' * 16)
            self.assertEqual(wire.line(), b'ERR STICKY_STATE\n')
            self.assertIn(b'invoke=0', wire.wait(2))

    def test_lru_eviction_changes_generation_and_stale_v_never_invokes(self):
        with self.sticky_session(budget=12300) as wire:
            generation = wire.upload(1, bytes([5]) * 16)
            wire.send(self.load('b', inputs=[16, 16]))
            self.assertEqual(wire.line(), b'OK LOADED b 12\n')
            evicted = wire.info()
            self.assertGreater(evicted['generation'], generation)
            self.assertEqual((evicted['loaded'], evicted['ready']), (0, '00'))
            wire.send(f'V a {generation}\n'.encode() + b'X' * 16 + b'I CACHE\nQ\n')
            self.assertEqual(wire.line(), b'ERR STICKY_STATE\n')
            self.assertIn(b'invoke=0', wire.wait(2))
            self.assertEqual(wire.tail(), b'')

    def test_lru_eviction_can_be_reprimed_before_v(self):
        with self.sticky_session(budget=12300) as wire:
            old = wire.upload(1, bytes([5]) * 16)
            wire.send(self.load('b', inputs=[16, 16]))
            self.assertEqual(wire.line(), b'OK LOADED b 12\n')
            self.assertEqual(wire.info()['ready'], '00')
            new = wire.upload(1, bytes([9]) * 16)
            self.assertGreater(new, old)
            self.assertEqual(wire.invoke(new, bytes([1]) * 16), bytes([10]) * 16)
            wire.send(b'Q\n')
            self.assertEqual(wire.line(), b'OK QUIT\n')
            wire.wait()

    def test_legacy_raw_r_invalidates_sticky_identity_and_still_reads_all_ports(self):
        with self.sticky_session() as wire:
            old = wire.upload(1, bytes([5]) * 16)
            wire.send(b'R a\n' + bytes([7]) * 16 + bytes([11]) * 16)
            self.assertEqual(wire.output(), bytes([18]) * 16)
            state = wire.info()
            self.assertGreater(state['generation'], old)
            self.assertEqual(state['ready'], '00')
            wire.send(f'V a {state["generation"]}\n'.encode() + b'X' * 16)
            self.assertEqual(wire.line(), b'ERR STICKY_STATE\n')
            self.assertIn(b'invoke=1', wire.wait(2))

    def test_model_replacement_never_reuses_old_generation_even_same_geometry(self):
        with self.sticky_session() as wire:
            old = wire.upload(1, bytes([5]) * 16)
            wire.send(self.load(inputs=[16, 16]))
            self.assertEqual(wire.line(), b'OK LOADED a 12\n')
            self.assertGreater(wire.info()['generation'], old)
            new = wire.upload(1, bytes([9]) * 16)
            self.assertGreater(new, old)
            self.assertEqual(wire.invoke(new, bytes([1]) * 16), bytes([10]) * 16)
            wire.send(f'V a {old}\n'.encode() + b'X' * 16)
            self.assertEqual(wire.line(), b'ERR STICKY_STATE\n')
            self.assertIn(b'invoke=1', wire.wait(2))

    def test_budget_shrink_evicts_and_upload_rejects_zero_or_oversize_budget(self):
        with self.sticky_session() as wire:
            old = wire.upload(1, b'w' * 16)
            wire.send(b'B 0\n')
            self.assertEqual(wire.line(), b'OK CACHE 0\n')
            state = wire.info()
            self.assertGreater(state['generation'], old)
            self.assertEqual((state['loaded'], state['ready']), (0, '00'))
            wire.send(b'U a 1\n' + b'w' * 16)
            self.assertEqual(wire.line(), b'ERR STICKY_CACHE_BUDGET\n')
            self.assertIn(b'invoke=0', wire.wait(2))
        for budget in (0, 12299):
            with self.subTest(budget=budget):
                result = self.protocol(f'B {budget}\n'.encode() + self.load(inputs=[16, 16]) +
                                       b'U a 1\n' + b'w' * 16, expected=2)
                self.assertTrue(result.stdout.endswith(b'ERR STICKY_CACHE_BUDGET\n'))
                self.assertIn(b'load=1 graph=1 inv=0 alloc=3 invoke=0', result.stderr)

    def test_sticky_headers_reject_bad_indices_generations_and_short_uploads(self):
        for index in (b'0', b'-1', b'+1', b'512', b'', b'1 extra'):
            with self.subTest(index=index):
                result = self.protocol(b'U a ' + index + b'\n', expected=2)
                self.assertEqual(result.stdout, b'ERR STICKY_UPLOAD_HEADER\n')
        for generation in (b'0', b'-1', b'+1', b'18446744073709551616', b'', b'1 extra'):
            with self.subTest(generation=generation):
                result = self.protocol(b'V a ' + generation + b'\n', expected=2)
                self.assertEqual(result.stdout, b'ERR STICKY_INVOKE_HEADER\n')
        result = self.protocol(self.load(inputs=[16, 16]) + b'U a 2\n', expected=2)
        self.assertTrue(result.stdout.endswith(b'ERR STICKY_INPUT_INDEX\n'))
        result = self.protocol(self.load(inputs=[16, 16]) + b'U a 1\nshort', expected=2)
        self.assertTrue(result.stdout.endswith(b'ERR TENSOR_INPUT\n'))
        self.assertIn(b'invoke=0', result.stderr)

    def test_old_i_model_name_and_new_model_inspection_are_unambiguous(self):
        result = self.protocol(self.load('MODEL', inputs=[16, 16]) +
                               b'I MODEL\nI MODEL MODEL\nI MODEL missing\nQ\n')
        lines = result.stdout.splitlines()
        self.assertEqual(lines[:2], [b'OK LOADED MODEL 12', b'OK INFO 12 0'])
        state = lines[2].split()
        self.assertEqual(state[:3], [b'OK', b'MODEL', b'MODEL'])
        self.assertGreater(int(state[3]), 0)
        self.assertEqual(state[4:], [b'0', b'2', b'00'])
        self.assertEqual(lines[3:], [b'ERR UNKNOWN_MODEL', b'OK QUIT'])

    def test_sticky_upload_lock_failure_cleans_and_uncertain_unlock_stops(self):
        payload = self.load(inputs=[16, 16]) + b'U a 1\n' + b'w' * 16
        locked = self.protocol(payload, mode='protocol-lock', expected=2)
        self.assertTrue(locked.stdout.endswith(b'ERR TENSOR_INPUT\n'))
        self.assertIn(b'invoke=0', locked.stderr)
        unlocked = self.protocol(payload, mode='protocol-unlock', expected=2)
        self.assertTrue(unlocked.stdout.endswith(b'ERR BUFFER_UNLOCK\n'))
        self.assertIn(b'inv=1 ahb=3 reg=3', unlocked.stderr)


if __name__ == '__main__':
    unittest.main()
