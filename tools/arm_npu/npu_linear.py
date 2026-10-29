"""Execute custom GGUF dense layers using the Android Tensor NPU.

The native helper compiles constants on the phone and runs the resulting
DarwiNN code. Every dense layer goes through this helper; failed calls stop the
run. Attention, normalization, RoPE, embeddings and sampling run on the host.
Compiled code is cached in RAM. Its optional LRU budget trades RAM for repeated
compilation, without imposing a fixed RAM requirement for an entire model.
"""
from __future__ import annotations

import atexit
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

import numpy as np

from build_constant_fc import author_constant_fc

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
                 artifact_cache=None, matrix_bank_rows=None):
        self.weights = weights
        self.artifact_cache = artifact_cache
        if artifact_cache and Path(weights.path).resolve() != artifact_cache.source_gguf:
            from compiled_model import file_sha256
            if file_sha256(weights.path) != artifact_cache.source_sha256:
                raise ValueError("Runtime weights do not match the persistent artifact source")
        self.weight_dtype = weight_dtype or os.environ.get("PIXELFORGE_FC_WEIGHT_DTYPE", "int8")
        if self.weight_dtype not in ("int8", "float32"):
            raise ValueError("weight dtype must be int8 or float32; FP16 modes are unvalidated")
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
        try:
            self.process = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr, bufsize=0, start_new_session=True)
        except BaseException:
            self.stderr.close()
            raise
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
            if self.artifact_cache:
                self._write(b"I CAPABILITIES\n")
                fields = self._readline().split()
                if fields[:2] != ["OK", "CAPABILITIES"] or not {"D", "L"}.issubset(fields[2:]):
                    raise RuntimeError("Persistent artifacts require a native helper with D/L capabilities: " + " ".join(fields))
                self.capabilities = fields[2:]
        except BaseException:
            self.close()
            raise

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
        if available is not None and available < required:
            raise MemoryError(f"Available memory {available // MIB} MiB is below the estimated compile "
                              f"working set {required // MIB} MiB after releasing this run's NPU cache")

    def _ensure_invoke_room(self, key, record, *, additional_bytes=0):
        if not self.min_free:
            return
        # Lazy SQ loading creates a driver copy and activation buffers even
        # when the compiled graph is already cached. Protect this graph only.
        buffers = 8 * (record["columns"] + record["rows"])
        required = self.min_free + record["compiled_bytes"] + buffers + MIB + additional_bytes
        available = _memory_available()
        while available is not None and available < required:
            if not self._free_oldest("invocation memory headroom", exclude=key):
                break
            available = _memory_available()
        if available is not None and available < required:
            raise MemoryError(f"Available memory {available // MIB} MiB is below the estimated invocation "
                              f"working set {required // MIB} MiB; current graph was protected")

    def _compile(self, name, start, stop):
        checker = getattr(self.weights, "check_source_unchanged", None)
        if checker is not None:
            checker(check_path=True)
        key = (name, start, stop, self.weight_dtype)
        if key in self.models:
            self.models.move_to_end(key)
            return self.models[key]
        began = time.monotonic()
        columns = self.weights.tensors[name].shape[1]
        cached = self.artifact_cache.get(name, start, stop, self.weight_dtype, columns,
            reserve=lambda size: self._make_room(size, transient=size * 2)) if self.artifact_cache else None
        if cached is not None:
            stored, code = cached
            identity = json.dumps(key, separators=(",", ":")).encode() + bytes.fromhex(stored["graph_sha256"])
            identifier = "fc_" + hashlib.sha256(identity).hexdigest()[:32]
            self._write(f"L {identifier} 1 1 {len(code)} {columns*4} {(stop-start)*4}\n".encode(), code)
            status = self._readline().split()
            if status != ["OK", "LOADED", identifier, str(len(code))]:
                raise RuntimeError("Failed to load verified compiled artifact: " + " ".join(status))
            record = {"name": name, "row_start": start, "row_stop": stop,
                      "rows": stop-start, "columns": columns, "weight_dtype": self.weight_dtype,
                      "graph_sha256": stored["graph_sha256"], "graph_bytes": stored["graph_bytes"],
                      "compiled_bytes": len(code), "bytecode_sha256": stored["bytecode_sha256"],
                      "native_id": identifier, "operation": "load", "compile_seconds": 0.0,
                      "load_seconds": time.monotonic()-began}
            self.load_seconds += record["load_seconds"]
            self.models[key] = record
            self.compile_records.append(record)
            while self.cache_budget and len(self.models) > 1 and self._resident_bytes() > self.cache_budget:
                self._free_oldest("actual loaded-code budget", exclude=key)
            print(json.dumps({"event": "npu_loaded", **record}), flush=True)
            return record
        parameters = (stop - start) * columns
        estimate = parameters * (2 if self.weight_dtype == "int8" else 4) + MIB
        self._make_room(estimate, transient=parameters * 16)
        matrix = self.weights.numpy_rows(name, start, stop)
        graph = author_constant_fc(matrix, self.weight_dtype, name)
        del matrix
        graph_sha256 = hashlib.sha256(graph).hexdigest()
        identity = json.dumps(key, separators=(",", ":")).encode() + bytes.fromhex(graph_sha256)
        identifier = "fc_" + hashlib.sha256(identity).hexdigest()[:32]
        self._write(f"C {identifier} 1 1 {len(graph)} {columns * 4} {(stop-start) * 4}\n".encode(), graph)
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
        del graph
        self.compile_seconds += record["compile_seconds"]
        self.models[key] = record
        self.compile_records.append(record)
        if self.artifact_cache:
            self._ensure_invoke_room(key, record, additional_bytes=compiled_bytes)
            code = self.export_code(record)
            stored = self.artifact_cache.store(name, start, stop, self.weight_dtype, columns, record, code)
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

    def __call__(self, name, activation):
        if self.closed:
            raise RuntimeError("NPU hook is closed")
        descriptor = self.weights.tensors[name]
        if len(descriptor.shape) != 2:
            raise ValueError("Linear weights must be a matrix")
        rows, columns = descriptor.shape
        activation = np.asarray(activation, dtype=np.float32)
        if activation.shape != (1, columns):
            raise ValueError(f"{name} activation must be [1,{columns}], got {activation.shape}")
        if not np.isfinite(activation).all():
            raise ValueError("Nonfinite activation")
        bank = self.lm_bank_rows if name in ("output.weight", "token_embd.weight") else rows
        if self.artifact_cache:
            bank = self.artifact_cache.bank_rows(name) or bank
        bank = self.matrix_bank_rows.get(name, bank)
        output = np.empty((1, rows), dtype=np.float32)
        activation_max = float(np.max(np.abs(activation)))
        scale = max(activation_max, 1.0)
        normalized = np.ascontiguousarray(activation / scale, dtype=np.float32)
        payload = normalized.tobytes()
        try:
            for start in range(0, rows, bank):
                stop = min(start + bank, rows)
                record = self._compile(name, start, stop)
                self._ensure_invoke_room((name, start, stop, self.weight_dtype), record)
                began = time.monotonic()
                self._write(f"R {record['native_id']}\n".encode(), payload)
                status = self._readline().split()
                expected_bytes = (stop - start) * 4
                if len(status) != 4 or status[:3] != ["OK", "OUTPUT", "1"] or int(status[3]) != expected_bytes:
                    raise RuntimeError(f"NPU invocation failed for {name}: " + " ".join(status))
                result = np.frombuffer(self._read(expected_bytes), dtype="<f4")
                if not np.isfinite(result).all():
                    self._dump_failure(name, start, stop, record, activation, normalized, result, scale, "raw output")
                    raise RuntimeError(f"Nonfinite NPU result for {name}")
                raw_max = float(np.max(np.abs(result)))
                result = np.asarray(result * scale, dtype=np.float32)
                if not np.isfinite(result).all():
                    self._dump_failure(name, start, stop, record, activation, normalized, result, scale, "rescaled output")
                    raise RuntimeError(f"Nonfinite rescaled NPU result for {name}")
                output[0, start:stop] = result
                elapsed = time.monotonic() - began
                self.invoke_seconds += elapsed
                self.invocations += 1
                self.records.append({"name": name, "row_start": start, "row_stop": stop,
                                     "native_id": record["native_id"], "seconds": elapsed,
                                     "activation_scale": scale, "activation_max": activation_max,
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
            atexit.unregister(self.close)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def factory(weights):
    return NPULinear(weights)
