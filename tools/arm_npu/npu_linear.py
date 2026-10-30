"""Execute custom GGUF dense layers using the Android Tensor NPU.

The native helper compiles constants on the phone and runs the resulting
DarwiNN code. Every dense layer goes through this helper; failed calls stop the
run. Attention, normalization, RoPE, embeddings and sampling run on the host.
Compiled code is cached in RAM. Its optional LRU budget trades RAM for repeated
compilation, without imposing a fixed RAM requirement for an entire model.
"""
from __future__ import annotations

import atexit
import array
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import socket
import subprocess
import time

import numpy as np

from build_constant_fc import author_constant_fc
from author_batched_fc import author_batched_fc

MIB = 1024 * 1024


def _memory_available():
    """Read the actual Android kernel's available memory, when exposed."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


class NPULinear:
    def __init__(self, weights, *, command=None, weight_dtype=None,
                 lm_bank_rows=None, timeout=180, cache_mib=None, min_free_mib=None,
                 artifact_cache=None, matrix_bank_rows=None, batch_vectors=1,
                 batch_artifact_cache=None, native_runtime_cache_mib=None,
                 fused_artifact_cache=None, fused_batch_artifact_cache=None,
                 decode_batch_vectors=1, read_only=False, fused_prefill_only=False,
                 weight_buffer_cache_mib=0, native_max_resident_models=2,
                 npu_performance_mode='default'):
        if npu_performance_mode not in ('default', 'balanced', 'high', 'sustained'):
            raise ValueError('Unsupported NPU performance mode')
        self.npu_performance_mode = npu_performance_mode
        if type(native_max_resident_models) is not int or not 1 <= native_max_resident_models <= 64:
            raise ValueError('native_max_resident_models must be 1 to 64')
        self.native_max_resident_models = native_max_resident_models
        if type(weight_buffer_cache_mib) is not int or not 0 <= weight_buffer_cache_mib <= 6144:
            raise ValueError('weight_buffer_cache_mib must be 0 to 6144')
        self.weight_buffer_cache_mib = weight_buffer_cache_mib
        self.weight_buffers = None
        self.fd_socket = None
        self.dma_imports = self.pipe_weight_uploads = 0
        if type(read_only) is not bool:
            raise ValueError("read_only must be a boolean")
        self.read_only = read_only
        if type(fused_prefill_only) is not bool:
            raise ValueError("fused_prefill_only must be a boolean")
        self.fused_prefill_only = fused_prefill_only
        if type(batch_vectors) is not int or not 1 <= batch_vectors <= 512:
            raise ValueError("batch_vectors must be an integer from 1 to 512")
        self.batch_vectors = batch_vectors
        if type(decode_batch_vectors) is not int or decode_batch_vectors not in (1, batch_vectors):
            raise ValueError("decode_batch_vectors must be 1 or the configured batch_vectors")
        self.decode_batch_vectors = decode_batch_vectors
        self.batch_artifact_cache = batch_artifact_cache
        self.fused_artifact_cache = fused_artifact_cache
        self.fused_batch_artifact_cache = fused_batch_artifact_cache
        self.native_runtime_cache_mib = native_runtime_cache_mib
        if native_runtime_cache_mib is not None and (type(native_runtime_cache_mib) is not int
                or not 0 <= native_runtime_cache_mib <= 2048):
            raise ValueError("native_runtime_cache_mib must be an integer from 0 to 2048")
        self.weights = weights
        self.artifact_cache = artifact_cache
        if artifact_cache and Path(weights.path).resolve() != artifact_cache.source_gguf:
            from compiled_model import file_sha256
            if file_sha256(weights.path) != artifact_cache.source_sha256:
                raise ValueError("Runtime weights do not match the persistent artifact source")
        self.weight_dtype = weight_dtype or os.environ.get("PIXELFORGE_FC_WEIGHT_DTYPE", "int8")
        if self.weight_dtype not in ("int8", "float32"):
            raise ValueError("weight dtype must be int8 or float32; FP16 modes are unvalidated")
        if (batch_vectors > 1 or getattr(weights, 'fused_sources', {})) and self.weight_dtype != "float32":
            raise ValueError("Batched FC requires float32 constants; no additional int8 quantization")
        for cache, vectors in ((artifact_cache, 1), (batch_artifact_cache, batch_vectors),
                               (fused_artifact_cache, 1), (fused_batch_artifact_cache, batch_vectors)):
            if cache is not None:
                if getattr(cache, "batch_vectors", 1) != vectors:
                    raise ValueError("Artifact batch dimensions do not match this runtime")
                if Path(weights.path).resolve() != cache.source_gguf:
                    raise ValueError("Batch runtime weights do not match the artifact source")
        self.lm_bank_rows = int(lm_bank_rows if lm_bank_rows is not None else
                                os.environ.get("PIXELFORGE_LM_BANK_ROWS", "4096"))
        self.matrix_bank_rows = dict(matrix_bank_rows or {})
        if any(not isinstance(name, str) or name not in weights.tensors or
               type(rows) is not int or rows < 1 for name, rows in self.matrix_bank_rows.items()):
            raise ValueError("matrix_bank_rows must map known tensor names to positive integers")
        self.cache_budget = int(cache_mib if cache_mib is not None else
                                os.environ.get("PIXELFORGE_NPU_CACHE_MIB", "0")) * MIB
        self.min_free = int(min_free_mib if min_free_mib is not None else
                            os.environ.get("PIXELFORGE_NPU_MIN_FREE_MIB", "128")) * MIB
        if weight_buffer_cache_mib:
            # A full original-code cache is several GiB. Leave room for the
            # Android foreground app and opaque vendor working allocations.
            self.min_free = max(self.min_free, 1024 * MIB)
        self.timeout = float(timeout)
        if self.lm_bank_rows < 1 or self.cache_budget < 0 or self.min_free < 0:
            raise ValueError("bank rows must be positive; memory settings must be nonnegative")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be positive and finite")
        self.models = OrderedDict()
        self.compile_records = []
        self.records = []
        self.evictions = []
        self.invocations = 0
        self.compile_seconds = 0.0
        self.load_seconds = 0.0
        self.invoke_seconds = 0.0
        self.closed = False
        self.buffer = bytearray()
        state_home = Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
        self.state_dir = state_home / "pixelforge-npu"
        self.stderr_path = Path(os.environ.get("PIXELFORGE_NPU_STDERR", str(
            self.state_dir / f"native-{os.getpid()}-{time.monotonic_ns()}.stderr")))
        self.stderr_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        fd = os.open(self.stderr_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND |
                     os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        os.fchmod(fd, 0o600)
        self.stderr = os.fdopen(fd, "wb")
        if command is None:
            override = os.environ.get("PIXELFORGE_NPU_COMMAND_JSON")
            command = json.loads(override) if override else [
                "/usr/sbin/chroot", "/proc/1/root", "/system/bin/env",
                "LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:/data/data/com.termux/files/usr/lib",
                "LD_PRELOAD=/system/lib64/libbinder_ndk.so",
                "/data/local/tmp/pixelforge-npu/npu_pipe",
            ]
        if not isinstance(command, (list, tuple)) or not command or not all(
                isinstance(part, str) and part for part in command):
            self.stderr.close()
            raise ValueError("native command must be a nonempty list of strings")
        self.command = list(command)
        spawn_options = {}
        child_socket = None
        environment = os.environ.copy()
        environment.pop('PIXELFORGE_NPU_PERFORMANCE_MODE', None)
        environment.pop('PIXELFORGE_NPU_DEFER_WEIGHT_ALLOCATION', None)
        environment.pop('PIXELFORGE_NPU_FD_SOCKET', None)
        environment.pop('PIXELFORGE_NPU_MAX_RESIDENT_MODELS', None)
        spawn_options['env'] = environment
        if npu_performance_mode != 'default':
            environment['PIXELFORGE_NPU_PERFORMANCE_MODE'] = npu_performance_mode
            spawn_options['env'] = environment
        if weight_buffer_cache_mib:
            from dma_buffers import DmaBufferCache
            self.weight_buffers = DmaBufferCache(weight_buffer_cache_mib,
                min_free_bytes=self.min_free, memory_available=_memory_available)
            self.fd_socket, child_socket = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
            self.fd_socket.settimeout(self.timeout)
            environment['PIXELFORGE_NPU_FD_SOCKET'] = str(child_socket.fileno())
            environment['PIXELFORGE_NPU_MAX_RESIDENT_MODELS'] = str(native_max_resident_models)
            environment['PIXELFORGE_NPU_DEFER_WEIGHT_ALLOCATION'] = '1'
            spawn_options = dict(env=environment, pass_fds=(child_socket.fileno(),))
        try:
            self.process = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr, bufsize=0, start_new_session=True, **spawn_options)
        except BaseException:
            if self.fd_socket is not None:
                self.fd_socket.close()
            if self.weight_buffers is not None:
                self.weight_buffers.close()
            self.stderr.close()
            raise
        finally:
            if child_socket is not None:
                child_socket.close()
        os.set_blocking(self.process.stdin.fileno(), False)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.write_selector = selectors.DefaultSelector()
        self.write_selector.register(self.process.stdin, selectors.EVENT_WRITE)
        atexit.register(self.close)
        try:
            for _ in range(64):
                line = self._readline()
                if line == "READY PIXELFORGE_NPU_PIPE 1":
                    break
            else:
                raise RuntimeError("Native NPU helper did not send protocol version 1 READY")
            self.capabilities = None
            if (self.artifact_cache or self.batch_artifact_cache or fused_artifact_cache
                    or fused_batch_artifact_cache or native_runtime_cache_mib is not None or weight_buffer_cache_mib
                    or npu_performance_mode != 'default'):
                self._write(b"I CAPABILITIES\n")
                fields = self._readline().split()
                if fields[:2] != ["OK", "CAPABILITIES"] or not {"D", "L"}.issubset(fields[2:]):
                    raise RuntimeError("Persistent artifacts require a native helper with D/L capabilities: " + " ".join(fields))
                self.capabilities = fields[2:]
                if weight_buffer_cache_mib and not {'Z', 'DMA_BUF_IMPORT_V1', 'STICKY_INPUTS_V1', 'RESIDENT_MODEL_LIMIT_V1', 'DEFERRED_DMA_WEIGHTS_V1'}.issubset(self.capabilities):
                    raise RuntimeError('Shared weight buffers require native DMA and sticky-input capabilities')
                if weight_buffer_cache_mib:
                    self._write(b'I CACHE LIMIT\n')
                    if self._readline() != f'OK CACHE_LIMIT MODELS {native_max_resident_models}':
                        raise RuntimeError('Shared weights require the configured native residency limit')
                if npu_performance_mode != 'default':
                    if 'PERFORMANCE_MODE_V1' not in self.capabilities:
                        raise RuntimeError('Native helper does not support requested NPU performance mode')
                    self._write(b'I PERFORMANCE_MODE\n')
                    value = {'balanced': 2, 'high': 3, 'sustained': 4}[npu_performance_mode]
                    if self._readline() != f'OK PERFORMANCE_MODE {npu_performance_mode} {value}':
                        raise RuntimeError('Native helper performance mode differs from its trusted configuration')
                if native_runtime_cache_mib is not None:
                    if "B" not in self.capabilities:
                        raise RuntimeError("Native helper lacks runtime cache budget capability B")
                    budget = native_runtime_cache_mib * MIB
                    self._write(f"B {budget}\n".encode())
                    if self._readline().split() != ["OK", "CACHE", str(budget)]:
                        raise RuntimeError("Native helper rejected runtime cache budget")
        except BaseException:
            self.close()
            raise

    def runtime_cache_stats(self):
        """Read helper-owned residency counters only between completed requests."""
        if self.closed or "B" not in (self.capabilities or ()):
            return None
        self._write(b"I CACHE\n")
        fields = self._readline().split()
        if len(fields) != 8 or fields[:2] != ["OK", "CACHE"] or not all(x.isdecimal() for x in fields[2:]):
            raise RuntimeError("Native helper returned invalid runtime cache statistics")
        return dict(zip(("budget_bytes", "charged_bytes", "resident_models", "loads", "hits", "evictions"),
                        map(int, fields[2:])))

    def weight_buffer_cached(self, cache_key, expected_bytes):
        return self.weight_buffers is not None and self.weight_buffers.get(cache_key, expected_bytes) is not None

    def upload_weight(self, record, index, data=None, *, cache_key=None):
        """Upload one exact weight port, or reimport an already cached DMA FD."""
        if type(index) is not int or index < 1:
            raise ValueError('Sticky uploads are restricted to weight input ports')
        identifier = record['native_id']
        if self.weight_buffers is not None:
            if cache_key is None:
                raise ValueError('Shared weights require a source-specific cache identity')
            expected = record.get('weight_port_bytes', {}).get(index)
            if expected is None:
                if data is None:
                    raise ValueError('Weight byte length is required for a cached DMA import')
                expected = np.asarray(data).nbytes
            item = self.weight_buffers.get(cache_key, expected)
            if item is None:
                if data is None or np.asarray(data).nbytes != expected:
                    raise ValueError('Missing original weight data for a DMA cache miss')
                item = self.weight_buffers.put(cache_key, data)
            try:
                header = f'Z {identifier} {index} {item.bytes}\n'.encode()
                descriptors = array.array('i', [item.fd])
                accepted = self.fd_socket.sendmsg([header], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors)])
                if accepted != len(header):
                    raise RuntimeError('Incomplete device-buffer descriptor transfer')
                self._write(header)
                fields = self._readline().split()
                prefix = ['OK', 'IMPORTED', identifier, str(index)]
                self.dma_imports += 1
                if len(fields) != 5 or fields[:4] != prefix or not fields[4].isdecimal() or int(fields[4]) < 1:
                    raise RuntimeError('Native weight transfer failed; no replay')
                return int(fields[4])
            finally:
                # Z's acknowledged helper duplicate owns the imported port.
                # Temporary parent FDs also close on every transport failure.
                if item.transient:
                    item.close()
        else:
            if data is None:
                raise ValueError('Original weight data required when shared buffers are disabled')
            payload = np.ascontiguousarray(data)
            self._write(f'U {identifier} {index}\n'.encode(), memoryview(payload).cast('B'))
            fields = self._readline().split()
            prefix = ['OK', 'UPLOADED', identifier, str(index)]
            self.pipe_weight_uploads += 1
        if len(fields) != 5 or fields[:4] != prefix or not fields[4].isdecimal() or int(fields[4]) < 1:
            raise RuntimeError('Native weight transfer failed; no replay')
        return int(fields[4])

    def _fill(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not self.selector.select(remaining):
            raise TimeoutError("Native NPU response timed out")
        chunk = os.read(self.process.stdout.fileno(), 65536)
        if not chunk:
            raise RuntimeError(f"Native NPU helper exited ({self.process.poll()}); see {self.stderr_path}")
        self.buffer.extend(chunk)

    def _readline(self):
        deadline = time.monotonic() + self.timeout
        while True:
            end = self.buffer.find(b"\n")
            if end >= 0:
                if end > 65536:
                    raise RuntimeError("Native protocol status line exceeds limit")
                line = bytes(self.buffer[:end])
                del self.buffer[:end + 1]
                return line.decode("utf8", errors="strict")
            if len(self.buffer) > 65536:
                raise RuntimeError("Native protocol status line exceeds limit")
            self._fill(deadline)

    def _read(self, size):
        if size < 0:
            raise ValueError("Negative native output length")
        deadline = time.monotonic() + self.timeout
        while len(self.buffer) < size:
            self._fill(deadline)
        data = bytes(self.buffer[:size])
        del self.buffer[:size]
        return data

    def _write(self, *items):
        if self.closed:
            raise RuntimeError("NPU hook is closed")
        deadline = time.monotonic() + self.timeout
        for item in items:
            view = memoryview(item)
            while view:
                try:
                    accepted = os.write(self.process.stdin.fileno(), view)
                except BlockingIOError:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not self.write_selector.select(remaining):
                        raise TimeoutError("Native NPU input timed out")
                    continue
                if not accepted:
                    raise RuntimeError("Native NPU helper did not accept input")
                view = view[accepted:]

    def _resident_bytes(self):
        return sum(record["compiled_bytes"] for record in self.models.values())

    def _free_oldest(self, reason, exclude=None):
        key = next((candidate for candidate in self.models if candidate != exclude), None)
        if key is None:
            return False
        record = self.models[key]
        self._write(f"F {record['native_id']}\n".encode())
        status = self._readline()
        if status != "OK FREED":
            raise RuntimeError("Failed to release owned NPU code: " + status)
        del self.models[key]
        event = {"native_id": record["native_id"], "name": record["name"],
                 "compiled_bytes": record["compiled_bytes"], "reason": reason}
        self.evictions.append(event)
        print(json.dumps({"event": "npu_cache_evicted", **event}), flush=True)
        return True

    def _make_room(self, estimate, transient):
        # A single matrix larger than the budget is allowed, then replaced on
        # demand. Splitting the vocabulary projection bounds its largest graph.
        while self.cache_budget and self.models and self._resident_bytes() + estimate > self.cache_budget:
            self._free_oldest("configured compiled-code budget")
        if not self.min_free:
            return
        available = _memory_available()
        required = self.min_free + transient + estimate
        while available is not None and available < required and self.models:
            self._free_oldest("kernel memory headroom")
            available = _memory_available()
        while available is not None and available < required and self.weight_buffers is not None and self.weight_buffers.evict_oldest():
            available = _memory_available()
        if available is not None and available < required:
            raise MemoryError(f"Available memory {available // MIB} MiB is below the estimated compile "
                              f"working set {required // MIB} MiB after releasing this run's NPU cache")

    def _ensure_invoke_room(self, key, record, *, additional_bytes=0):
        if not self.min_free:
            return
        # Lazy SQ loading creates a driver copy and activation buffers even
        # when the compiled graph is already cached. Protect this graph only.
        buffers = 8 * record.get("batch_vectors", 1) * (record["columns"] + record["rows"])
        required = self.min_free + record["compiled_bytes"] + buffers + MIB + additional_bytes
        available = _memory_available()
        while available is not None and available < required:
            if not self._free_oldest("invocation memory headroom", exclude=key):
                if self.weight_buffers is None or not self.weight_buffers.evict_oldest():
                    break
            available = _memory_available()
        if available is not None and available < required:
            raise MemoryError(f"Available memory {available // MIB} MiB is below the estimated invocation "
                              f"working set {required // MIB} MiB; current graph was protected")

    def _model_key(self, name, start, stop, batch_vectors=1):
        key = (name, start, stop, self.weight_dtype)
        return key if batch_vectors == 1 else key + (batch_vectors,)

    def _artifact_cache(self, name, batch_vectors, prefill=False):
        if (not self.fused_prefill_only or prefill) and name in getattr(self.weights, 'runtime_fused_names', {}):
            cache = self.fused_artifact_cache if batch_vectors == 1 else self.fused_batch_artifact_cache
            if cache is not None or name in getattr(self.weights, 'fused_sources', {}):
                return cache
        return self.artifact_cache if batch_vectors == 1 else self.batch_artifact_cache

    def _compile(self, name, start, stop, batch_vectors=1, prefill=False):
        checker = getattr(self.weights, "check_source_unchanged", None)
        if checker is not None:
            checker(check_path=True)
        key = self._model_key(name, start, stop, batch_vectors)
        cache = self._artifact_cache(name, batch_vectors, prefill)
        if cache is None and (self.read_only or any(c is not None and c.read_only for c in (
                self.artifact_cache, self.batch_artifact_cache, self.fused_artifact_cache,
                self.fused_batch_artifact_cache))):
            raise FileNotFoundError(f"No read-only artifact package for {name} batch {batch_vectors}")
        if key in self.models:
            self.models.move_to_end(key)
            return self.models[key]
        began = time.monotonic()
        columns = self.weights.tensors[name].shape[1]
        cached = cache.get(name, start, stop, self.weight_dtype, columns,
            reserve=lambda size: self._make_room(size, transient=size * 2)) if cache else None
        if cached is not None:
            stored, code = cached
            identity = json.dumps(key, separators=(",", ":")).encode() + bytes.fromhex(stored["graph_sha256"])
            identifier = "fc_" + hashlib.sha256(identity).hexdigest()[:32]
            self._write(f"L {identifier} 1 1 {len(code)} {batch_vectors*columns*4} {batch_vectors*(stop-start)*4}\n".encode(), code)
            status = self._readline().split()
            if status != ["OK", "LOADED", identifier, str(len(code))]:
                raise RuntimeError("Failed to load verified compiled artifact: " + " ".join(status))
            record = {"name": name, "row_start": start, "row_stop": stop,
                      "rows": stop-start, "columns": columns, "weight_dtype": self.weight_dtype,
                      "graph_sha256": stored["graph_sha256"], "graph_bytes": stored["graph_bytes"],
                      "compiled_bytes": len(code), "bytecode_sha256": stored["bytecode_sha256"],
                      "native_id": identifier, "operation": "load", "compile_seconds": 0.0,
                      "load_seconds": time.monotonic()-began}
            if batch_vectors > 1:
                record["batch_vectors"] = batch_vectors
            self.load_seconds += record["load_seconds"]
            self.models[key] = record
            self.compile_records.append(record)
            while self.cache_budget and len(self.models) > 1 and self._resident_bytes() > self.cache_budget:
                self._free_oldest("actual loaded-code budget", exclude=key)
            print(json.dumps({"event": "npu_loaded", **record}), flush=True)
            return record
        parameters = (stop - start) * columns
        estimate = parameters * (2 if self.weight_dtype == "int8" else 4) + MIB
        self._make_room(estimate, transient=parameters * (24 if batch_vectors > 1 else 16)
                        + batch_vectors * (columns + stop - start) * 8)
        matrix = self.weights.numpy_rows(name, start, stop)
        graph = (author_constant_fc(matrix, self.weight_dtype, name) if batch_vectors == 1 else
                 author_batched_fc(matrix, self.weight_dtype, name, batch_vectors=batch_vectors))
        del matrix
        graph_sha256 = hashlib.sha256(graph).hexdigest()
        identity = json.dumps(key, separators=(",", ":")).encode() + bytes.fromhex(graph_sha256)
        identifier = "fc_" + hashlib.sha256(identity).hexdigest()[:32]
        self._write(f"C {identifier} 1 1 {len(graph)} {batch_vectors*columns*4} {batch_vectors*(stop-start)*4}\n".encode(), graph)
        status = self._readline().split()
        if len(status) != 4 or status[:2] != ["OK", "COMPILED"] or status[2] != identifier:
            raise RuntimeError(f"NPU compilation failed for {key}: " + " ".join(status))
        compiled_bytes = int(status[3])
        if compiled_bytes < 1:
            raise RuntimeError("Native compiler returned an empty artifact")
        record = {"name": name, "row_start": start, "row_stop": stop,
                  "rows": stop - start, "columns": columns, "weight_dtype": self.weight_dtype,
                  "graph_sha256": graph_sha256, "graph_bytes": len(graph),
                  "compiled_bytes": compiled_bytes, "compile_seconds": time.monotonic() - began,
                  "native_id": identifier, "operation": "compile"}
        if batch_vectors > 1:
            record["batch_vectors"] = batch_vectors
        del graph
        self.compile_seconds += record["compile_seconds"]
        self.models[key] = record
        self.compile_records.append(record)
        if cache:
            self._ensure_invoke_room(key, record, additional_bytes=compiled_bytes)
            code = self.export_code(record)
            stored = cache.store(name, start, stop, self.weight_dtype, columns, record, code)
            record["bytecode_sha256"] = stored["bytecode_sha256"]
        while self.cache_budget and len(self.models) > 1 and self._resident_bytes() > self.cache_budget:
            self._free_oldest("actual compiled-code budget", exclude=key)
        print(json.dumps({"event": "npu_compiled", **record}), flush=True)
        return record

    def export_code(self, record):
        """Export an owned graph's real native bytes; never synthesize weights."""
        self._write(f"D {record['native_id']}\n".encode())
        status = self._readline().split()
        if len(status) != 3 or status[:2] != ["OK", "BYTECODE"]:
            raise RuntimeError("Native bytecode export failed: " + " ".join(status))
        size = int(status[2])
        if size != record["compiled_bytes"] or not 8 <= size <= 2 * 1024 ** 3:
            raise RuntimeError("Native export byte count does not match compilation")
        return self._read(size)

    def __call__(self, name, activation, *, prefill=False):
        if self.closed:
            raise RuntimeError("NPU hook is closed")
        descriptor = self.weights.tensors[name]
        if len(descriptor.shape) != 2:
            raise ValueError("Linear weights must be a matrix")
        rows, columns = descriptor.shape
        activation = np.asarray(activation, dtype=np.float32)
        if activation.ndim != 2 or not 1 <= activation.shape[0] or activation.shape[1] != columns:
            raise ValueError(f"{name} activation must be [vectors,{columns}], got {activation.shape}")
        vectors = activation.shape[0]
        if vectors * max(rows, columns) * 4 > 256 * MIB:
            raise ValueError("NPU matrix batch exceeds the bounded transfer limit")
        if not np.isfinite(activation).all():
            raise ValueError("Nonfinite activation")
        bank = self.lm_bank_rows if name in ("output.weight", "token_embd.weight") else rows
        bank = self.matrix_bank_rows.get(name, bank)
        batch_vectors = self.batch_vectors if vectors > 1 or prefill else self.decode_batch_vectors
        cache = self._artifact_cache(name, batch_vectors, prefill)
        if cache:
            bank = cache.bank_rows(name) or bank
        try:
            self._make_room(estimate=0, transient=vectors * (columns*8 + rows*12))
            output = np.empty((vectors, rows), dtype=np.float32)
            maxima = np.max(np.abs(activation), axis=1)
            scales = np.maximum(maxima, 1.0)
            normalized_all = np.ascontiguousarray(activation / scales[:, None], dtype=np.float32)
            # Consume a bank across every input chunk before moving to the next
            # bank. Small caches must not reread the whole matrix per chunk.
            for start in range(0, rows, bank):
                stop = min(start + bank, rows)
                record = self._compile(name, start, stop, batch_vectors, prefill)
                for begin in range(0, vectors, batch_vectors):
                    used = min(batch_vectors, vectors - begin)
                    chunk = activation[begin:begin + used]
                    chunk_scales = np.ones(batch_vectors, dtype=np.float32)
                    chunk_scales[:used] = scales[begin:begin + used]
                    normalized = np.zeros((batch_vectors, columns), dtype=np.float32)
                    normalized[:used] = normalized_all[begin:begin + used]
                    payload = normalized.tobytes()
                    self._ensure_invoke_room(self._model_key(name, start, stop, batch_vectors), record)
                    began = time.monotonic()
                    self._write(f"R {record['native_id']}\n".encode(), payload)
                    status = self._readline().split()
                    expected_bytes = batch_vectors * (stop - start) * 4
                    if len(status) != 4 or status[:3] != ["OK", "OUTPUT", "1"] or int(status[3]) != expected_bytes:
                        raise RuntimeError(f"NPU invocation failed for {name}: " + " ".join(status))
                    result = np.frombuffer(self._read(expected_bytes), dtype="<f4").reshape(batch_vectors, stop-start)
                    scale = float(chunk_scales[0]) if batch_vectors == 1 else chunk_scales.tolist()
                    if not np.isfinite(result).all():
                        self._dump_failure(name, start, stop, record, chunk, normalized, result, scale, "raw output")
                        raise RuntimeError(f"Nonfinite NPU result for {name}")
                    raw_max = float(np.max(np.abs(result)))
                    result = np.asarray(result * chunk_scales[:, None], dtype=np.float32)
                    if not np.isfinite(result).all():
                        self._dump_failure(name, start, stop, record, chunk, normalized, result, scale, "rescaled output")
                        raise RuntimeError(f"Nonfinite rescaled NPU result for {name}")
                    output[begin:begin+used, start:stop] = result[:used]
                    elapsed = time.monotonic() - began
                    self.invoke_seconds += elapsed
                    self.invocations += 1
                    self.records.append({"name": name, "row_start": start, "row_stop": stop,
                                         "native_id": record["native_id"], "seconds": elapsed,
                                         "activation_scale": scale,
                                         "activation_max": float(np.max(maxima[begin:begin + used])),
                                         "vectors": used, "batch_vectors": batch_vectors,
                                         "raw_result_max": raw_max})
            return output
        except BaseException:
            # A timeout or malformed response can leave the binary stream out
            # of sync. Terminate only this helper, then stop the model run.
            self.close()
            raise

    def _dump_failure(self, name, start, stop, record, activation, normalized, result, scale, stage):
        folder = Path(os.environ.get("PIXELFORGE_NPU_FAILURE_DIR", str(self.state_dir / "failures")))
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
        stem = name.replace(".", "_") + f"_{start}_{stop}_{time.monotonic_ns()}"
        np.save(folder / (stem + ".input.npy"), activation)
        np.save(folder / (stem + ".normalized_input.npy"), normalized)
        np.save(folder / (stem + ".output.npy"), result)
        (folder / (stem + ".json")).write_text(json.dumps({
            "name": name, "row_start": start, "row_stop": stop, "activation_scale": scale,
            "stage": stage, "compiled_model": record, "source_gguf": str(self.weights.path),
        }, indent=2) + "\n")

    def stats(self):
        return {
            "backend": "native-google-tensor-npu-fc", "weight_dtype": self.weight_dtype,
            "batch_vectors": self.batch_vectors,
            "decode_batch_vectors": self.decode_batch_vectors,
            "native_runtime_cache_mib": self.native_runtime_cache_mib,
            "weight_buffer_cache_mib": self.weight_buffer_cache_mib,
            "native_max_resident_models": self.native_max_resident_models if self.weight_buffer_cache_mib else None,
            "npu_performance_mode": self.npu_performance_mode,
            "weight_buffer_cache": self.weight_buffers.stats() if self.weight_buffers is not None else None,
            "dma_weight_imports": self.dma_imports, "pipe_weight_uploads": self.pipe_weight_uploads,
            "deferred_dma_weight_allocation": bool(self.weight_buffer_cache_mib),
            "source_gguf": str(self.weights.path), "native_pid": self.process.pid,
            "compiled_models": sum(record.get("operation", "compile") == "compile" for record in self.compile_records),
            "loaded_models": sum(record.get("operation") == "load" for record in self.compile_records),
            "resident_models": len(self.models),
            "compiled_bytes": self._resident_bytes(),
            "compiled_bytes_total": sum(record["compiled_bytes"] for record in self.compile_records),
            "compile_seconds": self.compile_seconds, "npu_invocations": self.invocations,
            "load_seconds": self.load_seconds,
            "invoke_seconds": self.invoke_seconds, "matrices": self.compile_records,
            "invocations": self.records, "evictions": self.evictions,
            "cache_budget_bytes": self.cache_budget, "min_free_bytes": self.min_free,
            "native_identity": "SHA256(canonical matrix key + full serialized graph SHA256)",
            "native_capabilities": self.capabilities,
            "stderr_path": str(self.stderr_path), "native_command": self.command,
            "cpu_operations": "architecture engine: tokenization, embedding lookup, normalization, RoPE, "
                              "attention, KV/recurrent state, convolution, elementwise activations, sampling",
            "cpu_fallback": False,
            "activation_scaling": "max(maxabs(x),1); NPU(x/scale) * scale",
        }

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.process.poll() is None:
                try:
                    os.write(self.process.stdin.fileno(), b"Q\n")
                except (BlockingIOError, BrokenPipeError):
                    pass
                self.process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=5)
        finally:
            self.selector.close()
            self.write_selector.close()
            self.process.stdin.close()
            self.process.stdout.close()
            self.stderr.close()
            if self.fd_socket is not None:
                self.fd_socket.close()
            if self.weight_buffers is not None:
                self.weight_buffers.close()
            atexit.unregister(self.close)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def factory(weights):
    return NPULinear(weights)
