"""Standard-library control plane embedded verbatim in pixelforge-npu.O."""
import argparse
import ast
import ctypes
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time


_PF_TAIL_BYTES = 65536
_PF_CLEANUP_SECONDS = 80
_PF_REPORT_BYTES = 64 * 1024 * 1024
_PF_LIBC = ctypes.CDLL(None, use_errno=True) if sys.platform == "linux" else None
_PF_ACTIONS = {
    "help": "Show this command interface without extracting the kit.",
    "extract": "Extract and verify the embedded compiler kit.",
    "verify": "Verify all pinned offline runtime assets without installing them.",
    "setup": "Install the offline Python environment and dedicated Android helper.",
    "doctor": "Read prerequisites, asset locations and hardware observations.",
    "inspect": "Read a GGUF header and optionally its tensor descriptors.",
    "plan": "Plan compilation and disk requirements without starting the NPU.",
    "compile": "Compile and save eligible GGUF weight matrices.",
    "run": "Generate with the pinned architecture engine and NPU dense bridge.",
    "cpu": "Generate an explicit CPU reference using the same engine.",
    "compare": "Run fresh CPU/NPU references and compare their generated tokens.",
    "compare-reports": "Compare two existing RESULT.json files without extraction.",
    "build-native": "Build a separate Android helper; refuse to overwrite a helper.",
    "build-engine": "Build an isolated engine without replacing the pinned executable.",
    "package": "Export the existing generic compiler kit as a fresh .run archive.",
    "test": "Run shipped Python tests and the native protocol fixture, without hardware.",
    "test-native": "Compile and run the native protocol fixture without hardware.",
    "fixture": "Create a fresh synthetic GGUF test model.",
    "graph": "Author a constant FC graph from one GGUF tensor.",
    "collect": "Collect read-only hardware observations using benchmark.py.",
    "tool": "Invoke a whitelisted shipped CLI; legacy tools need their separate addon.",
}
_PF_TOOLS = {
    "gguf_inspect.py": "numeric", "compile_gguf.py": "numeric",
    "run_gguf.py": "numeric", "create_tiny_gguf.py": "numeric",
    "build_constant_fc.py": "numeric", "gguf_broker.py": "numeric",
    "setup_gguf.py": "stdlib", "pack_gguf.py": "stdlib",
    "benchmark.py": "stdlib", "build_native.sh": "shell",
    "build_llama_npu.sh": "shell", "gguf.sh": "generic-shell",
    "runtime.py": "legacy", "model_host.py": "legacy",
    "validate_runtime.py": "legacy", "fetch_tokenizer.py": "legacy",
}
_PF_SCRIPTS = {
    "verify": "setup_gguf.py", "setup": "setup_gguf.py",
    "inspect": "gguf_inspect.py", "plan": "compile_gguf.py",
    "compile": "compile_gguf.py", "run": "run_gguf.py", "cpu": "run_gguf.py",
    "package": "pack_gguf.py", "fixture": "create_tiny_gguf.py",
    "graph": "build_constant_fc.py", "collect": "benchmark.py",
}


class _PFParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


class _PFCancelled(BaseException):
    pass


def _pf_child_guard(parent_pid):
    # The child owns only its new session. Parent death asks the actual runner
    # to clean its detached engine/broker, even if O kills the Python shim.
    if _PF_LIBC is not None:
        if _PF_LIBC.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
            os._exit(126)
        if os.getppid() != parent_pid:
            os.kill(os.getpid(), signal.SIGTERM)


def _pf_returncode(returncode):
    return returncode if returncode >= 0 else 128 - returncode


def _pf_caller_cwd():
    return Path(os.environ.get("OSTADIX_NPU_CALLER_CWD", os.getcwd())).expanduser().absolute()


def _pf_user_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else _pf_caller_cwd() / path


def _pf_emit(data):
    # O's Python shim reserves stdout for its own JSON framing.
    sys.stderr.write(data.decode("utf8", errors="replace"))
    sys.stderr.flush()


def _pf_parse_tail(tail):
    for line in reversed(tail.splitlines()):
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _pf_worker_source():
    embedded = globals().get("WORKER_SOURCE")
    if isinstance(embedded, str) and embedded.strip():
        return embedded
    location = globals().get("__file__")
    if location:
        sibling = Path(location).with_name("worker.py")
        if sibling.is_file():
            return sibling.read_text(encoding="utf8")
    return None


def _pf_guarded_argv(argv, parent_pid):
    original = [os.fspath(value) for value in argv]
    worker = _pf_worker_source()
    if not worker or sys.platform != "linux":
        return original
    owner_pid = os.environ.get("OSTADIX_NPU_LAUNCHER_PID", "")
    owner_ticks = os.environ.get("OSTADIX_NPU_LAUNCHER_START_TICKS", "")
    if owner_pid.isdecimal() and int(owner_pid) > 0 and owner_ticks.isdecimal():
        owner_pid, ticks = str(int(owner_pid)), owner_ticks
    else:
        owner_pid = str(parent_pid)
        ticks = Path(f"/proc/{parent_pid}/stat").read_text().split(") ", 1)[1].split()[19]
    wrapped = [_pf_stdlib_python(), "-I", "-c", worker,
               "--owner-pid", owner_pid, "--owner-start-ticks", ticks,
               "--grace-seconds", "60"]
    python = (Path(original[0]).name.startswith("python") or
              (len(original) > 1 and (original[1].endswith(".py") or original[1] in ("-m", "-c", "-I", "-B"))))
    if python:
        wrapped.append("--python")
    return wrapped + ["--"] + original


class _PFExecution:
    def __init__(self, deadline):
        self.ends = time.monotonic() + deadline
        self.cancel_file = os.environ.get("OSTADIX_NPU_CANCEL_FILE")

    def cancelled(self):
        return bool(self.cancel_file and Path(self.cancel_file).exists())

    def run(self, argv, cwd, *, env=None):
        if self.cancelled():
            return {"exit_code": 130, "cancelled": True, "started": False}
        if time.monotonic() >= self.ends:
            return {"exit_code": 124, "timed_out": True, "started": False}
        started = time.monotonic()
        parent_pid = os.getpid()
        process = subprocess.Popen(
            _pf_guarded_argv(argv, parent_pid), cwd=os.fspath(cwd), env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=False, start_new_session=True,
            preexec_fn=(lambda: _pf_child_guard(parent_pid)) if os.name == "posix" else None,
        )
        tails = {"stdout": bytearray(), "stderr": bytearray()}
        selector = selectors.DefaultSelector()
        for name, pipe in (("stdout", process.stdout), ("stderr", process.stderr)):
            os.set_blocking(pipe.fileno(), False)
            selector.register(pipe, selectors.EVENT_READ, name)
        reason = None
        cleanup_until = None
        forced = False

        def request_stop(kind):
            nonlocal reason, cleanup_until
            if reason is None:
                reason = kind
                cleanup_until = time.monotonic() + _PF_CLEANUP_SECONDS
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass

        try:
            while process.poll() is None or selector.get_map():
                try:
                    if self.cancelled():
                        request_stop("cancelled")
                    if time.monotonic() >= self.ends:
                        request_stop("timed_out")
                    if cleanup_until is not None and time.monotonic() >= cleanup_until:
                        if process.poll() is None:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            forced = True
                        # A detached descendant may retain a pipe. Bound the
                        # wait without sending a signal to any unowned process.
                        if process.poll() is not None:
                            break
                    for key, _ in selector.select(0.1):
                        data = os.read(key.fileobj.fileno(), 8192)
                        if not data:
                            selector.unregister(key.fileobj)
                            key.fileobj.close()
                            continue
                        tail = tails[key.data]
                        tail.extend(data)
                        if len(tail) > _PF_TAIL_BYTES:
                            del tail[:-_PF_TAIL_BYTES]
                        _pf_emit(data)
                except (KeyboardInterrupt, _PFCancelled):
                    request_stop("cancelled")
            process.wait(timeout=5)
        finally:
            if process.poll() is None:
                # An unexpected Python I/O error still releases this child.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=_PF_CLEANUP_SECONDS)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=5)
            selector.close()
            for pipe in (process.stdout, process.stderr):
                if not pipe.closed:
                    pipe.close()
        stdout = bytes(tails["stdout"]).decode("utf8", errors="replace")
        result = {
            "exit_code": 130 if reason == "cancelled" else 124 if reason == "timed_out"
                         else _pf_returncode(process.returncode),
            "returncode": process.returncode, "seconds": time.monotonic() - started,
            "stdout_tail": stdout,
            "stderr_tail": bytes(tails["stderr"]).decode("utf8", errors="replace"),
            "cleanup_forced": forced,
        }
        if reason:
            result[reason] = True
        parsed = _pf_parse_tail(stdout)
        if parsed is not None:
            result["result"] = parsed
        return result


def _pf_stdlib_python():
    python = Path("/usr/bin/python3")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError("This kit requires /usr/bin/python3 (CPython 3.12 on Linux ARM64)")
    return os.fspath(python)


def _pf_numeric_python(kit, override=None, *, legacy=False):
    addon = "minicpm5" if legacy else "gguf-runtime"
    default = kit / "custom-models" / addon / "venv/bin/python"
    # absolute() preserves the venv's executable symlink and pyvenv.cfg lookup.
    python = _pf_user_path(override) if override else default
    if not python.is_file() or not os.access(python, os.X_OK):
        instruction = "Run the separate MiniCPM runtime setup" if legacy else "Run setup first"
        raise RuntimeError(f"{instruction}: Python environment is missing at {python}")
    return os.fspath(python)


def _pf_script(kit, name):
    script = kit / "benchmark.py" if name == "benchmark.py" else kit / "tools/arm_npu" / name
    if not script.is_file() or not script.resolve().is_relative_to(kit.resolve()):
        raise FileNotFoundError(f"Shipped CLI is missing or redirected: {script}")
    return script


def _pf_tool_command(kit, name, arguments, python_override):
    if name not in _PF_TOOLS:
        raise ValueError("Unknown tool; choose one of: " + ", ".join(sorted(_PF_TOOLS)))
    kind = _PF_TOOLS[name]
    script = _pf_script(kit, name)
    if kind == "generic-shell":
        scripts = {"inspect": "gguf_inspect.py", "compile": "compile_gguf.py", "run": "run_gguf.py"}
        if not arguments or arguments[0] not in scripts:
            raise ValueError("gguf.sh requires inspect|compile|run [options]")
        # Match the shell dispatch while keeping the Python CLI under the
        # worker's TERM-to-KeyboardInterrupt wrapper for its finally cleanup.
        return _pf_tool_command(kit, scripts[arguments[0]], arguments[1:], python_override)
    if name == "build_native.sh":
        if len(arguments) > 1:
            raise ValueError("build_native.sh takes one fresh output path")
        destination = _pf_user_path(arguments[0]) if arguments else kit / "custom-models/gguf-runtime/build/native/npu_pipe_gguf"
        if destination.exists() or destination.is_symlink():
            raise FileExistsError("Refusing to overwrite an existing native helper: " + os.fspath(destination))
        arguments = [os.fspath(destination)]
    environment = None
    if kind in ("shell", "generic-shell"):
        shell = shutil.which("bash")
        if not shell:
            raise RuntimeError("bash is required for this build tool")
        command = [shell, os.fspath(script)]
        if kind == "generic-shell":
            environment = os.environ.copy()
            environment["PIXELFORGE_PYTHON"] = _pf_numeric_python(kit, python_override)
    else:
        python = _pf_stdlib_python() if kind == "stdlib" else _pf_numeric_python(
            kit, python_override, legacy=kind == "legacy")
        command = [python, os.fspath(script)]
    return command + list(arguments), environment


def _pf_source_help(kit, name):
    path = _pf_script(kit, name)
    result = {"tool": name, "description": None, "arguments": []}
    if path.suffix == ".py":
        tree = ast.parse(path.read_text(encoding="utf8"))
        result["description"] = ast.get_docstring(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
                flags = []
                for value in node.args:
                    try:
                        literal = ast.literal_eval(value)
                    except (ValueError, TypeError):
                        continue
                    if isinstance(literal, str):
                        flags.append(literal)
                entry = {"flags": flags}
                for keyword in node.keywords:
                    if keyword.arg in ("help", "required", "choices", "default", "action"):
                        try:
                            entry[keyword.arg] = ast.literal_eval(keyword.value)
                        except (ValueError, TypeError):
                            pass
                result["arguments"].append(entry)
    else:
        result["description"] = "Build script takes one optional output/build-directory path."
    return result


def _pf_read_report(path):
    path = _pf_user_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Run report is missing: {path}")
    with path.open("rb") as source:
        content = source.read(_PF_REPORT_BYTES + 1)
    if len(content) > _PF_REPORT_BYTES:
        raise ValueError(f"Run report exceeds the bounded JSON reader: {path}")
    report = json.loads(content, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError("Nonfinite JSON constant: " + value)))
    if not isinstance(report, dict):
        raise ValueError(f"Run report must be a JSON object: {path}")
    return report


def _pf_report_identity(report, backend):
    if report.get("success") is not True or report.get("backend") != backend:
        raise ValueError(f"Expected a successful {backend} report")
    engine = report.get("engine")
    if not isinstance(engine, dict) or engine.get("success") is not True:
        raise ValueError("Architecture engine report is missing or unsuccessful")
    for field in ("source_gguf_sha256", "engine_sha256"):
        digest = report.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Report needs a valid " + field)
    if not isinstance(report.get("prompt"), str):
        raise ValueError("Report needs its original prompt")
    tokens = engine.get("generated_tokens")
    if not isinstance(tokens, list) or not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Report needs nonempty generated token IDs")
    if "prompt_tokens" in engine and (type(engine["prompt_tokens"]) is not int or engine["prompt_tokens"] < 1):
        raise ValueError("Engine prompt_tokens must be a positive count")
    if "steps" in engine:
        steps = engine["steps"]
        if not isinstance(steps, list) or len(steps) != len(tokens):
            raise ValueError("Engine step count contradicts its generated tokens")
        for index, step in enumerate(steps):
            if (not isinstance(step, dict) or type(step.get("step")) is not int
                    or step["step"] != index or type(step.get("token")) is not int
                    or step["token"] != tokens[index]):
                raise ValueError("Engine step token/index contradicts its generated tokens")
            if step.get("eos") is True and index != len(steps) - 1:
                raise ValueError("Engine generated tokens after an end-of-generation step")
    if "dense_npu_requested" in engine:
        expected = backend != "cpu-reference"
        if engine["dense_npu_requested"] is not expected:
            raise ValueError("Engine dense_npu_requested contradicts its reported backend")
    for field in ("engine_cleanup_ok", "broker_cleanup_ok"):
        if field in report and report[field] is not True:
            raise ValueError("Report records unsuccessful " + field)
    if backend != "cpu-reference":
        broker = report.get("broker")
        if not isinstance(broker, dict) or broker.get("success") is not True or broker.get("native_cleanup_ok") is not True:
            raise ValueError("NPU report needs successful broker and native cleanup")
        native = broker.get("npu")
        if not isinstance(native, dict):
            raise ValueError("NPU report needs its native statistics")
        calls = native.get("npu_invocations")
        if type(calls) is not int or calls < 1:
            raise ValueError("NPU report has no verified native invocations")
        for part in (report, engine, broker, native):
            if "cpu_fallback" in part and part["cpu_fallback"] is not False:
                raise ValueError("NPU report contradicts its no-CPU-fallback policy")
            if "dense_npu_requested" in part and part["dense_npu_requested"] is not True:
                raise ValueError("NPU report contradicts its requested dense NPU backend")
            for field in ("source_gguf_sha256", "model_sha256"):
                if field in part and part[field] != report["source_gguf_sha256"]:
                    raise ValueError("NPU source/model hash contradicts the parent report")
        for part in (broker, native):
            for field in ("source_gguf", "model"):
                if field in part and "model" in report:
                    source = part[field]
                    if not isinstance(source, str) or not isinstance(report["model"], str):
                        raise ValueError("NPU source/model path must be a string")
                    if _pf_user_path(source).resolve() != _pf_user_path(report["model"]).resolve():
                        raise ValueError("NPU source/model path contradicts the parent report")
        for field in ("invocations", "calls"):
            if field in native:
                receipts = native[field]
                if not isinstance(receipts, list) or len(receipts) != calls or any(not isinstance(receipt, dict) for receipt in receipts):
                    raise ValueError("NPU per-call receipts contradict its invocation count")
        for field in ("npu_weight_matmul_operations",):
            if field in engine and (type(engine[field]) is not int or engine[field] < 1):
                raise ValueError("Engine reports no NPU matrix operations")
        if "matrix_operations" in broker:
            operations = broker["matrix_operations"]
            if type(operations) is not int or operations < 1:
                raise ValueError("Broker reports no NPU matrix operations")
            if "npu_weight_matmul_operations" in engine and operations != engine["npu_weight_matmul_operations"]:
                raise ValueError("Engine and broker matrix operation counts disagree")
    elif "npu_weight_matmul_operations" in engine and engine["npu_weight_matmul_operations"] != 0:
        raise ValueError("CPU reference report contains NPU matrix operations")
    return tokens


def _pf_compare_reports(cpu_path, npu_path):
    cpu, npu = _pf_read_report(cpu_path), _pf_read_report(npu_path)
    cpu_tokens = _pf_report_identity(cpu, "cpu-reference")
    npu_tokens = _pf_report_identity(npu, "google-tensor-npu-dense+ggml-cpu")
    for field in ("source_gguf_sha256", "engine_sha256", "prompt"):
        if cpu[field] != npu[field]:
            raise ValueError("CPU/NPU reports disagree on " + field)
    if "prompt_tokens" in cpu["engine"] and "prompt_tokens" in npu["engine"]:
        if cpu["engine"]["prompt_tokens"] != npu["engine"]["prompt_tokens"]:
            raise ValueError("CPU/NPU reports disagree on prompt_tokens")
    matches = cpu_tokens == npu_tokens
    return {
        "exit_code": 0 if matches else 1, "tokens_match": matches,
        "cpu_report": os.fspath(_pf_user_path(cpu_path)),
        "npu_report": os.fspath(_pf_user_path(npu_path)),
        "source_gguf_sha256": cpu["source_gguf_sha256"],
        "engine_sha256": cpu["engine_sha256"],
        "cpu_tokens": cpu_tokens, "npu_tokens": npu_tokens,
        "npu_invocations": npu["broker"]["npu"]["npu_invocations"],
        "full_logit_equivalence_claimed": False,
        "interpretation": "Token agreement is a bounded functionality check; int8 constants can change logits.",
    }


def _pf_child_action(action, child):
    result = {"action": action, "exit_code": child["exit_code"], "child": child}
    if "result" in child:
        result["result"] = child["result"]
    return result


def _pf_native_test(kit, execution, arguments):
    parser = _PFParser(add_help=False)
    parser.add_argument("--cxx")
    args = parser.parse_args(arguments)
    compiler = args.cxx or shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        raise RuntimeError("A C++17 compiler is required for the native protocol fixture")
    source = _pf_script(kit, "test_pipe_protocol.cc")
    with tempfile.TemporaryDirectory(prefix="pixelforge-protocol-") as directory:
        binary = Path(directory) / "test-pipe-protocol"
        build = execution.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-O2", source, "-o", binary], kit)
        if build["exit_code"]:
            return {"exit_code": build["exit_code"], "build": build, "hardware_used": False}
        run = execution.run([binary], kit)
        return {"exit_code": run["exit_code"], "build": build, "run": run, "hardware_used": False}


def _pf_doctor(kit, execution, python_override):
    checks = {
        "root": os.geteuid() == 0, "linux": sys.platform == "linux",
        "arm64": os.uname().machine == "aarch64", "controller_python": sys.version.split()[0],
        "stdlib_python": "/usr/bin/python3", "chroot": shutil.which("chroot"),
        "vendor_library": "/proc/1/root/vendor/lib64/libedgetpu_litert.so",
        "vendor_library_exists": Path("/proc/1/root/vendor/lib64/libedgetpu_litert.so").is_file(),
        "binder_preload_exists": Path("/proc/1/root/system/lib64/libbinder_ndk.so").is_file(),
        "native_helper": "/data/local/tmp/pixelforge-npu/npu_pipe_gguf",
        "native_helper_exists": Path("/data/local/tmp/pixelforge-npu/npu_pipe_gguf").is_file(),
        "engine": os.fspath(kit / "custom-models/gguf-runtime/bin/pixelforge-gguf"),
        "engine_exists": (kit / "custom-models/gguf-runtime/bin/pixelforge-gguf").is_file(),
        "rebuild_tools": {name: shutil.which(name) for name in ("bash", "cmake", "make", "g++", "clang++", "patch", "tar", "rg")},
        "termux_clang": Path("/data/data/com.termux/files/usr/bin/clang").is_file(),
        "disk_free_bytes": shutil.disk_usage(kit).free,
        "device_supplied_dependencies": ["compatible Android vendor compiler/runtime", "Binder", "rooted Android filesystem", "Ubuntu ARM64 CPython 3.12", "user-supplied GGUF weights"],
    }
    try:
        checks["numeric_python"] = _pf_numeric_python(kit, python_override)
        checks["numeric_environment_exists"] = True
    except RuntimeError as error:
        checks["numeric_environment_exists"] = False
        checks["numeric_environment_error"] = str(error)
    script = (
        "import importlib.util,json,sys;"
        "print(json.dumps({'version':sys.version.split()[0],"
        "'python312':sys.version_info[:2]==(3,12)}))"
    )
    probe = execution.run([_pf_stdlib_python(), "-B", "-I", "-c", script], kit)
    checks["stdlib_probe"] = probe.get("result")
    if probe["exit_code"]:
        return {"action": "doctor", "exit_code": probe["exit_code"], "checks": checks, "child": probe}
    if (kit / "benchmark.py").is_file():
        snippet = "import sys,json;sys.path.insert(0," + repr(os.fspath(kit)) + ");from benchmark import hardware_snapshot;print(json.dumps(hardware_snapshot()))"
        hardware = execution.run([_pf_stdlib_python(), "-B", "-I", "-c", snippet], kit)
        checks["hardware_snapshot"] = hardware.get("result")
        if hardware["exit_code"]:
            checks["hardware_snapshot_error"] = hardware["stderr_tail"]
        if hardware["exit_code"] in (124, 130):
            return {"action": "doctor", "exit_code": hardware["exit_code"], "checks": checks, "child": hardware}
    if checks["numeric_environment_exists"]:
        snippet = "import json,sys,numpy,flatbuffers;print(json.dumps({'python312':sys.version_info[:2]==(3,12),'numpy':numpy.__version__,'flatbuffers':flatbuffers.__version__}))"
        numeric = execution.run([checks["numeric_python"], "-B", "-I", "-c", snippet], kit)
        checks["numeric_dependencies"] = numeric.get("result")
        checks["numeric_dependencies_available"] = numeric["exit_code"] == 0
        if numeric["exit_code"]:
            checks["numeric_dependencies_error"] = numeric["stderr_tail"]
        if numeric["exit_code"] in (124, 130):
            return {"action": "doctor", "exit_code": numeric["exit_code"], "checks": checks, "child": numeric}
    ready = all(checks[key] for key in ("root", "linux", "arm64", "vendor_library_exists", "binder_preload_exists", "native_helper_exists", "engine_exists", "numeric_environment_exists"))
    ready = ready and bool(checks["chroot"]) and checks.get("stdlib_probe", {}).get("python312") is True
    ready = ready and checks.get("numeric_dependencies_available", False) and checks.get("numeric_dependencies", {}).get("python312") is True
    return {"action": "doctor", "exit_code": 0, "ready_for_default_npu": ready, "read_only": True, "checks": checks}


def _pf_compare_parser():
    parser = _PFParser(prog="compare", add_help=False, description="Generate fresh CPU and NPU references, then compare token choices.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Hi")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output")
    parser.add_argument("--engine")
    parser.add_argument("--artifacts")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--bank-rows", type=int, default=4096)
    parser.add_argument("--weight-dtype", choices=("int8", "float32"), default="int8")
    parser.add_argument("--cache-mib", type=int, default=0)
    parser.add_argument("--native-command-json")
    parser.add_argument("--timeout", type=int, default=1800)
    return parser


def _pf_compare(kit, execution, arguments, python_override):
    parser = _pf_compare_parser()
    args = parser.parse_args(arguments)
    if not 1 <= args.tokens <= 128 or not 1 <= args.threads <= 32 or args.bank_rows < 1 or args.cache_mib < 0 or args.timeout < 1:
        raise ValueError("Invalid comparison token/thread/bank/cache/timeout values")
    if args.read_only and not args.artifacts:
        raise ValueError("--read-only requires --artifacts")
    # Check the environment before reserving a result directory.
    _pf_numeric_python(kit, python_override)
    output = _pf_user_path(args.output) if args.output else Path.home() / ".local/state/ostadix-pixelforge" / ("compare-" + str(time.time_ns()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(mode=0o700)
    common = ["--model", args.model, "--prompt", args.prompt, "--tokens", str(args.tokens), "--threads", str(args.threads), "--bank-rows", str(args.bank_rows), "--weight-dtype", args.weight_dtype, "--cache-mib", str(args.cache_mib), "--timeout", str(args.timeout)]
    if args.engine:
        common.extend(["--engine", args.engine])
    npu_extra = []
    if args.artifacts:
        npu_extra.extend(["--artifacts", args.artifacts])
    if args.read_only:
        npu_extra.append("--read-only")
    if args.native_command_json:
        npu_extra.extend(["--native-command-json", args.native_command_json])
    children = {}
    for backend, extras in (("cpu", ["--cpu"]), ("npu", npu_extra)):
        command, environment = _pf_tool_command(kit, "run_gguf.py", common + extras + ["--output", os.fspath(output / backend)], python_override)
        child = execution.run(command, _pf_caller_cwd(), env=environment)
        children[backend] = child
        if child["exit_code"]:
            return {"action": "compare", "exit_code": child["exit_code"], "output": os.fspath(output), "children": children}
    comparison = _pf_compare_reports(output / "cpu/RESULT.json", output / "npu/RESULT.json")
    comparison.update(action="compare", output=os.fspath(output), children=children)
    return comparison


def integration_main(argv, materialize):
    """Execute one integrated action and return an OValue-compatible result."""
    action = "help"
    old_handlers = {}
    try:
        parser = _PFParser(add_help=False, allow_abbrev=False)
        parser.add_argument("--kit-dir")
        parser.add_argument("--python")
        parser.add_argument("--deadline", type=float, default=3500)
        parser.add_argument("--help", "-h", action="store_true")
        parser.add_argument("action", nargs="?", default="help")
        parser.add_argument("arguments", nargs=argparse.REMAINDER)
        options = parser.parse_args(list(argv))
        action, arguments = options.action, options.arguments
        if action not in _PF_ACTIONS:
            raise ValueError("Unknown action: " + action)
        if not math.isfinite(options.deadline) or options.deadline <= 0 or options.deadline > 3500:
            raise ValueError("--deadline must be greater than zero and at most 3500 seconds")
        if options.help or action == "help":
            return {"action": "help", "exit_code": 0, "usage": "pixelforge-npu.O [--kit-dir DIR] [--python EXECUTABLE] [--deadline SECONDS] ACTION [ARGS...]", "actions": _PF_ACTIONS, "tools": sorted(_PF_TOOLS), "global_options_before_action": True}
        if action == "compare-reports":
            if len(arguments) != 2:
                raise ValueError("compare-reports requires CPU_RESULT.json NPU_RESULT.json")
            result = _pf_compare_reports(*arguments)
            result["action"] = action
            return result
        for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda _signum, _frame: (_ for _ in ()).throw(_PFCancelled()))
        execution = _PFExecution(options.deadline)
        if execution.cancelled():
            return {"action": action, "exit_code": 130, "cancelled": True}
        kit_destination = os.fspath(_pf_user_path(options.kit_dir)) if options.kit_dir else None
        kit = Path(materialize(kit_destination)).absolute()
        if not kit.is_dir():
            raise FileNotFoundError("Verified compiler kit is not a directory: " + os.fspath(kit))
        if action == "extract":
            if arguments:
                raise ValueError("extract takes only the global --kit-dir option")
            return {"action": action, "exit_code": 0, "kit_dir": os.fspath(kit), "verified": True}
        if action == "doctor":
            if arguments:
                raise ValueError("doctor takes no action arguments")
            return _pf_doctor(kit, execution, options.python)
        if action == "compare":
            if arguments == ["--help"] or arguments == ["-h"]:
                return {"action": action, "exit_code": 0, "help": _pf_compare_parser().format_help(), "output": "--output chooses a fresh parent containing cpu and npu result directories"}
            return _pf_compare(kit, execution, arguments, options.python)
        if action in ("test", "test-native"):
            if action == "test":
                if arguments:
                    raise ValueError("test takes no action arguments; use test-native --cxx for a compiler override")
                python = _pf_numeric_python(kit, options.python)
                child = execution.run([python, "-m", "unittest", "discover", "-s", os.fspath(kit / "tools/arm_npu"), "-p", "test_*.py"], kit)
                if child["exit_code"]:
                    return _pf_child_action(action, child)
                native = _pf_native_test(kit, execution, [])
                return {"action": action, "exit_code": native["exit_code"], "python_tests": child, "native_fixture": native, "hardware_used": False}
            result = _pf_native_test(kit, execution, arguments)
            result["action"] = action
            return result
        if action in ("build-native", "build-engine"):
            if arguments in (["--help"], ["-h"]):
                return {"action": action, "exit_code": 0, "usage": action + " [fresh-output-path]"}
            if len(arguments) > 1:
                raise ValueError(action + " accepts at most one output/build-directory path")
            location = _pf_user_path(arguments[0]) if arguments else kit / "custom-models/gguf-runtime/build" / ("native/npu_pipe_gguf" if action == "build-native" else "engine")
            if action == "build-native" and (location.exists() or location.is_symlink()):
                raise FileExistsError("Refusing to overwrite an existing native helper: " + os.fspath(location))
            name = "build_native.sh" if action == "build-native" else "build_llama_npu.sh"
            command, environment = _pf_tool_command(kit, name, [os.fspath(location)], options.python)
            result = _pf_child_action(action, execution.run(command, _pf_caller_cwd(), env=environment))
            result["build_output"] = os.fspath(location)
            result["pinned_engine_replaced"] = False
            return result
        if action == "tool":
            if not arguments:
                raise ValueError("tool requires a shipped CLI name followed by its arguments")
            name, arguments = arguments[0], arguments[1:]
            if name not in _PF_TOOLS:
                raise ValueError("Tool is not in the shipped CLI whitelist: " + name)
        else:
            name = _PF_SCRIPTS[action]
            if action == "verify":
                arguments = ["--verify-only"] + arguments
            elif action == "plan":
                arguments = ["--plan-only"] + arguments
            elif action == "cpu":
                arguments = ["--cpu"] + arguments
            elif action == "collect" and not any(value == "--collect-seconds" or value.startswith("--collect-seconds=") for value in arguments):
                if arguments not in (["--help"], ["-h"]):
                    raise ValueError("collect requires --collect-seconds SECONDS [--output NEW_REPORT.json]")
            if action == "collect" and "--run" in arguments:
                raise ValueError("collect is read-only; service benchmark invocation is available explicitly through tool benchmark.py")
        if "--help" in arguments or "-h" in arguments:
            return {"action": action, "exit_code": 0, "help": _pf_source_help(kit, name)}
        command, environment = _pf_tool_command(kit, name, arguments, options.python)
        return _pf_child_action(action, execution.run(command, _pf_caller_cwd(), env=environment))
    except _PFCancelled:
        return {"action": action, "exit_code": 130, "cancelled": True}
    except KeyboardInterrupt:
        return {"action": action, "exit_code": 130, "cancelled": True}
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, TypeError, KeyError) as error:
        return {"action": action, "exit_code": 2, "error": str(error), "error_type": type(error).__name__}
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)
