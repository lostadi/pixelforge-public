"""Binary GGML bridge: execute named original GGUF weight matrices on the NPU.

stdout is exclusively the binary protocol. Compilation diagnostics use stderr.
The architecture engine owns attention, recurrent state, norms and sampling.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import sys
import tempfile

import numpy as np

from compiled_model import ArtifactCache, batch_directory
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


def serve(source, sink, weights, plan, native, full_ffn=None):
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
        if (len(fields) not in (5, 6) or fields[0] not in ("M", "F")
                or any(not f.isdecimal() for f in fields[1:5])
                or (len(fields) == 6 and fields[5] not in ("P", "D"))):
            raise ValueError("Expected M/F NAME_BYTES NCOLS NROWS NVEC [P|D]")
        name_bytes, columns, rows, vectors = map(int, fields[1:5])
        prefill = len(fields) == 6 and fields[5] == "P"
        if not 1 <= name_bytes <= 63 or min(columns, rows, vectors) < 1:
            raise ValueError("Invalid GGML matrix dimensions/name")
        if vectors * max(rows, columns) * 4 > MAX_BUFFER_BYTES:
            raise ValueError("GGML matrix request exceeds the bounded transfer limit")
        name = bytes(read_exact(source, name_bytes)).decode("utf8", errors="strict")
        if fields[0] == 'F':
            if full_ffn is None or name not in full_ffn.selected:
                raise ValueError('Full FFN operation is not enabled for this tensor')
            if ('P' if prefill else 'D') not in getattr(full_ffn, 'phases', 'PD'):
                raise ValueError('Full FFN operation is outside its configured phases')
            hidden = full_ffn.cache.layer(name)['hidden']
            if rows != hidden or columns != hidden:
                raise ValueError('Full FFN request dimensions differ from its pinned plan')
        else:
            item = allowed.get(name)
            if item is None or (item["rows"], item["columns"]) != (rows, columns):
                raise ValueError("GGML operation does not match the verified GGUF tensor plan")
        working_bytes = vectors * (columns * 8 + rows * 12)
        native._make_room(estimate=0, transient=working_bytes)
        activation = np.frombuffer(read_exact(source, vectors * columns * 4), dtype="<f4").reshape(vectors, columns)
        if not np.isfinite(activation).all():
            raise ValueError("Nonfinite GGML activation")
        executor = full_ffn if fields[0] == 'F' else native
        output = np.asarray(executor(name, activation, prefill=True) if prefill else
                            executor(name, activation), dtype="<f4")
        if output.shape != (vectors, rows):
            raise ValueError("NPU returned invalid batch dimensions")
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite NPU matrix result")
        sink.write(f"OK {output.size}\n".encode())
        sink.write(output.tobytes())
        sink.flush()
        operations += 3 if fields[0] == 'F' else 1
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
    full_executor = None
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
            batch_cache = None
            fused_cache = fused_batch_cache = None
            required = config.get("required_artifact_manifests")
            if required is not None:
                if (not isinstance(required, list)
                        or any(not isinstance(name, str) or not name
                            or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts
                            or PurePosixPath(name).as_posix() != name or '\\' in name
                            or PurePosixPath(name).name != 'MODEL.json' for name in required)
                        or len(set(required)) != len(required)):
                    raise ValueError("Invalid required artifact manifest inventory")
                required = set(required)
            loaded = set()
            def wanted(directory, default):
                if required is None:
                    return default
                try:
                    name = (Path(directory).resolve()/'MODEL.json').relative_to(
                        Path(config['artifacts']).resolve()).as_posix()
                except (ValueError, KeyError) as error:
                    raise ValueError("Required artifacts must remain within the configured package") from error
                if name in required:
                    loaded.add(name)
                    return True
                return False
            batch_vectors = config.get("batch_vectors", 1)
            decode_batch_vectors = config.get("decode_batch_vectors", 1)
            fused = config.get("fused_gate_up", False)
            if type(fused) is not bool:
                raise ValueError("fused_gate_up must be a boolean")
            full = config.get('full_ffn', False)
            dynamic_mode = config.get('dynamic_weight_mode','off')
            if dynamic_mode not in ('off','float32','original_groups','original_columns'):
                raise ValueError('Packed dynamic weights are not hardware-validated')
            dynamic = dynamic_mode != 'off'
            if dynamic and (full or fused or config.get('weight_dtype')!='float32'
                    or (dynamic_mode=='float32' and decode_batch_vectors!=batch_vectors)
                    or decode_batch_vectors not in (1,batch_vectors) or not config.get('read_only',False)):
                raise ValueError('Dynamic weights require readonly F32 geometry, one batch, and separate recipes')
            if type(full) is not bool or (full and fused):
                raise ValueError('Full FFN and paired gate/up recipes must be selected separately')
            if full and config.get('weight_dtype') != 'float32':
                raise ValueError('Full FFN requires original decoded float32 constants')
            if type(batch_vectors) is not int or not 1 <= batch_vectors <= 512:
                raise ValueError("batch_vectors must be an integer from 1 to 512")
            if config.get("artifacts") and not dynamic:
                if wanted(config["artifacts"], (decode_batch_vectors == 1 and not full) or (Path(config["artifacts"])/'MODEL.json').exists()):
                    cache = ArtifactCache(config["artifacts"], config["model"],
                                          source_sha256=source_sha256,
                                          source_identity=weights.source_identity, verify_source=False,
                                          read_only=config.get("read_only", False), plan=plan)
                if batch_vectors > 1:
                    directory = Path(config.get("batch_artifacts") or batch_directory(config["artifacts"], batch_vectors))
                    if wanted(directory, not (fused or full) or config.get("batch_artifacts") or (directory/'MODEL.json').exists()):
                        batch_cache = ArtifactCache(directory, config["model"], source_sha256=source_sha256,
                            source_identity=weights.source_identity, verify_source=False,
                            read_only=config.get("read_only", False), plan=plan, batch_vectors=batch_vectors)
            runtime_weights = weights
            if fused:
                from fused_ffn import FusedFFNWeights, FusedFFNDispatcher
                runtime_weights = FusedFFNWeights(weights, plan,
                    bank_rows=config.get("fused_bank_rows", config["bank_rows"]),
                    selected=config.get("fused_gate_up_tensors"))
                if config.get("artifacts"):
                    directory = Path(config.get("fused_artifacts") or Path(config["artifacts"]) / 'fused-ffn-v1')
                    options = dict(source_sha256=source_sha256, source_identity=weights.source_identity,
                                   verify_source=False, read_only=config.get("read_only", False),
                                   plan=runtime_weights.plan, fused_gate_up=True)
                    if wanted(directory, decode_batch_vectors == 1 and (not config.get("fused_prefill_only", False) or batch_vectors == 1)):
                        fused_cache = ArtifactCache(directory, config["model"], **options)
                    if batch_vectors > 1 and wanted(batch_directory(directory, batch_vectors), True):
                        fused_batch_cache = ArtifactCache(batch_directory(directory, batch_vectors),
                            config["model"], batch_vectors=batch_vectors, **options)
            full_cache = None
            if full:
                from full_ffn import FullFFNProgramCache, FullFFNExecutor, build_full_ffn_plan
                directory = Path(config.get('full_ffn_artifacts') or
                                 Path(config['artifacts'])/'full-ffn-v1'/f'b{batch_vectors}')
                if not wanted(directory, True):
                    raise ValueError('Full FFN recipe is absent from required artifact inventory')
                full_cache = FullFFNProgramCache(directory, config['model'], source_sha256=source_sha256,
                    source_identity=weights.source_identity, verify_source=False, read_only=True,
                    plan=build_full_ffn_plan(weights, source_sha256, config.get('full_ffn_hidden_chunk', 2048)),
                    batch_vectors=batch_vectors)
            dynamic_cache = None
            dynamic_decode_cache = None
            if dynamic:
                from dynamic_fc import DynamicProgramCache, DynamicWeightExecutor
                DynamicCache,DynamicExecutor=DynamicProgramCache,DynamicWeightExecutor
                dynamic_directory='dynamic-fc-v1'
                if dynamic_mode=='original_groups':
                    from grouped_fc import GroupedProgramCache,GroupedWeightExecutor
                    DynamicCache,DynamicExecutor=GroupedProgramCache,GroupedWeightExecutor
                    dynamic_directory='grouped-fc-v1'
                elif dynamic_mode=='original_columns':
                    from grouped_columns import ColumnProgramCache,ColumnPhaseExecutor
                    DynamicCache=ColumnProgramCache
                    dynamic_directory='grouped-columns-v1'
                directory=Path(config.get('dynamic_artifacts') or
                               Path(config['artifacts'])/dynamic_directory/f'b{batch_vectors}')
                if not wanted(directory,True):
                    raise ValueError('Dynamic geometry is absent from required artifact inventory')
                dynamic_cache=DynamicCache(directory,config['model'],source_sha256=source_sha256,
                    source_identity=weights.source_identity,verify_source=False,read_only=True,
                    plan=plan,batch_vectors=batch_vectors)
                dynamic_decode_cache=dynamic_cache
                if dynamic_mode in ('original_groups','original_columns') and decode_batch_vectors!=batch_vectors:
                    directory=Path(config.get('dynamic_decode_artifacts') or
                                   Path(config['artifacts'])/dynamic_directory/f'b{decode_batch_vectors}')
                    if not wanted(directory,True):raise ValueError('Grouped decode geometry inventory is missing')
                    dynamic_decode_cache=DynamicCache(directory,config['model'],source_sha256=source_sha256,
                        source_identity=weights.source_identity,verify_source=False,read_only=True,
                        plan=plan,batch_vectors=decode_batch_vectors)
            if required is not None and loaded != required:
                raise ValueError("Required artifact inventory differs from the configured graph recipes")
            runtime_options = {'npu_performance_mode': config.get('npu_performance_mode', 'default')}
            if config.get('weight_buffer_cache_mib', 0) > 0:
                runtime_options['native_max_resident_models'] = config.get('native_max_resident_models', 2)
            native = NPULinear(runtime_weights, command=config.get("native_command"),
                               weight_dtype=config.get("weight_dtype", "int8"),
                               artifact_cache=cache, cache_mib=config.get("cache_mib", 0),
                               batch_vectors=batch_vectors, batch_artifact_cache=batch_cache,
                               decode_batch_vectors=decode_batch_vectors,
                               weight_buffer_cache_mib=config.get('weight_buffer_cache_mib',0),
                               read_only=config.get("read_only", False),
                               fused_prefill_only=config.get("fused_prefill_only", False),
                               fused_artifact_cache=fused_cache, fused_batch_artifact_cache=fused_batch_cache,
                               native_runtime_cache_mib=config.get("native_runtime_cache_mib", 128 if (full or dynamic) else None),
                               matrix_bank_rows={item["name"]: item["bank_rows"] for item in plan["dense_tensors"]},
                               **runtime_options)
            if fused:
                native = FusedFFNDispatcher(native, runtime_weights,
                    cache_mib=config.get("fused_result_cache_mib", 64),
                    selected=config.get("fused_gate_up_tensors"),
                    prefill_only=config.get("fused_prefill_only", False))
            if dynamic:
                if dynamic_mode=='original_columns':
                    native=ColumnPhaseExecutor(native,weights,dynamic_cache,dynamic_decode_cache)
                elif dynamic_mode=='original_groups':
                    from grouped_fc import GroupedPhaseExecutor
                    native=GroupedPhaseExecutor(native,weights,dynamic_cache,dynamic_decode_cache)
                else:native=DynamicExecutor(native,weights,dynamic_cache)
            if full:
                full_executor = FullFFNExecutor(native, weights, full_cache, config.get('full_ffn_tensors'),
                                               phases=config.get('full_ffn_phases', 'PD'))
            result["matrix_operations"] = (serve(source, sink, weights, plan, native, full_executor)
                                           if full else serve(source, sink, weights, plan, native))
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
            try:
                result["native_runtime_cache"] = native.runtime_cache_stats()
            except BaseException as error:
                result.update(success=False, runtime_cache_error=str(error))
            result["npu"] = native.stats()
            if full_executor is not None:
                result['full_ffn_operations'] = full_executor.operations
                result['full_ffn'] = dict(operations=full_executor.operations,
                    hidden_chunk=full_executor.cache.full_plan['hidden_chunk'],
                    batch_vectors=full_executor.cache.batch_vectors,
                    phase_operations=full_executor.phase_operations,
                    activation_scaling='none; raw activation crosses the nonlinear FFN')
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
