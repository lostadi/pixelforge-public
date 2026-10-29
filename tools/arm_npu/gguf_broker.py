"""Binary GGML bridge: execute named original GGUF weight matrices on the NPU.

stdout is exclusively the binary protocol. Compilation diagnostics use stderr.
The architecture engine owns attention, recurrent state, norms and sampling.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile

import numpy as np

from compiled_model import ArtifactCache
from gguf_hf_loader import GGUFWeights
from model_architecture import build_plan
from npu_linear import NPULinear

MAX_BUFFER_BYTES = 256 * 1024 ** 2


def publish_owner(path, value):
    """Expose a complete private ownership record, without replacing a file."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".broker-owner-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as out:
            os.fchmod(out.fileno(), 0o600)
            json.dump(value, out)
            out.flush()
            os.fsync(out.fileno())
        os.link(temporary, path, follow_symlinks=False)
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_exact(source, size):
    if not 0 <= size <= MAX_BUFFER_BYTES:
        raise ValueError("GGML request exceeds the bounded transfer limit")
    result = bytearray()
    while len(result) < size:
        part = source.read(size - len(result))
        if not part:
            raise EOFError("Truncated GGML request")
        result.extend(part)
    return result


def serve(source, sink, weights, plan, native):
    allowed = {item["name"]: item for item in plan["dense_tensors"]}
    operations = 0
    while True:
        line = source.readline(4097)
        if not line:
            raise EOFError("Engine disconnected without a clean Q shutdown")
        if len(line) > 4096 or not line.endswith(b"\n"):
            raise ValueError("Invalid GGML protocol header")
        if line == b"Q\n":
            break
        fields = line.decode("ascii").split()
        if len(fields) != 5 or fields[0] != "M" or any(not f.isdecimal() for f in fields[1:]):
            raise ValueError("Expected M NAME_BYTES NCOLS NROWS NVEC")
        name_bytes, columns, rows, vectors = map(int, fields[1:])
        if not 1 <= name_bytes <= 63 or min(columns, rows, vectors) < 1:
            raise ValueError("Invalid GGML matrix dimensions/name")
        if vectors * max(rows, columns) * 4 > MAX_BUFFER_BYTES:
            raise ValueError("GGML matrix request exceeds the bounded transfer limit")
        name = bytes(read_exact(source, name_bytes)).decode("utf8", errors="strict")
        item = allowed.get(name)
        if item is None or (item["rows"], item["columns"]) != (rows, columns):
            raise ValueError("GGML operation does not match the verified GGUF tensor plan")
        working_bytes = vectors * (columns * 8 + rows * 12)
        native._make_room(estimate=0, transient=working_bytes)
        activation = np.frombuffer(read_exact(source, vectors * columns * 4), dtype="<f4").reshape(vectors, columns)
        if not np.isfinite(activation).all():
            raise ValueError("Nonfinite GGML activation")
        output = np.empty((vectors, rows), dtype="<f4")
        for index in range(vectors):
            output[index] = native(name, activation[index:index + 1])[0]
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite NPU matrix result")
        sink.write(f"OK {output.size}\n".encode())
        sink.write(output.tobytes())
        sink.flush()
        operations += 1
    if operations < 1:
        raise ValueError("No model weight operations were executed on the NPU")
    return operations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    def interrupted(signum, _frame):
        raise InterruptedError(f"Broker interrupted by signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    # Reserve output before native startup. An existing file or symlink cannot
    # be overwritten, even when a standalone caller supplies a bad config.
    report_fd = os.open(config["report"], os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                        os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    if config.get("ownership"):
        publish_owner(config["ownership"], {"pid": os.getpid(), "start_ticks":
                      Path(f"/proc/{os.getpid()}/stat").read_text().split(") ", 1)[1].split()[19]})
    source, sink = sys.stdin.buffer, sys.stdout.buffer
    native = None
    result = {"success": False, "weight_matmul_failure_policy": "stop engine; no eligible CPU fallback"}
    try:
        with contextlib.redirect_stdout(sys.stderr):
            weights = GGUFWeights(config["model"])
            source_sha256 = weights.sha256()
            if source_sha256 != config["model_sha256"]:
                raise ValueError("GGUF source identity changed before broker startup")
            plan = build_plan(weights, model_sha256=source_sha256, bank_rows=config["bank_rows"])
            if not plan["generation_adapter_supported"]:
                raise ValueError("This model's generation adapter is unsupported")
            cache = None
            if config.get("artifacts"):
                cache = ArtifactCache(config["artifacts"], config["model"],
                                      source_sha256=source_sha256,
                                      source_identity=weights.source_identity, verify_source=False,
                                      read_only=config.get("read_only", False), plan=plan)
            native = NPULinear(weights, command=config.get("native_command"),
                               weight_dtype=config.get("weight_dtype", "int8"),
                               artifact_cache=cache, cache_mib=config.get("cache_mib", 0),
                               matrix_bank_rows={item["name"]: item["bank_rows"] for item in plan["dense_tensors"]})
            result["matrix_operations"] = serve(source, sink, weights, plan, native)
            result.update(success=True, architecture=plan["architecture"], source_gguf_sha256=source_sha256)
    except BaseException as error:
        message = str(error).replace("\n", " ")[:2048]
        result["error"] = message
        try:
            sink.write(("ERR " + message + "\n").encode("utf8"))
            sink.flush()
        except (BrokenPipeError, OSError):
            pass
    finally:
        if native is not None:
            result["npu"] = native.stats()
            try:
                native.close()
                if native.process.returncode:
                    raise RuntimeError(f"Native helper cleanup exited {native.process.returncode}")
                result["native_cleanup_ok"] = True
            except BaseException as error:
                result.update(success=False, native_cleanup_ok=False, cleanup_error=str(error))
        with os.fdopen(report_fd, "w") as out:
            json.dump(result, out, indent=2)
            out.write("\n")
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
