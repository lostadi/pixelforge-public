"""Single-file .O bootstrap and verified offline-kit materialization.

This source is embedded into the generated .O program, rather than imported
from this developer checkout at runtime.
"""
import base64
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time


def _file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_kit(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Compiler kit must be a real directory: " + str(root))
    for item in ARCHIVE_FILES:
        current = root
        for part in Path(item["path"]).parts:
            current = current / part
            if current.is_symlink():
                raise ValueError("Redirected compiler kit file: " + item["path"])
        if (not current.is_file() or current.stat().st_size != item["bytes"]
                or _file_sha256(current) != item["sha256"]):
            raise ValueError("Compiler kit integrity check failed: " + item["path"])
    return root


def _decode_archive(source_path, destination):
    start = b"# OSTADIX_" + b"NPU_PAYLOAD_V1 " + ARCHIVE_SHA256.encode() + b"\n"
    end = b"# END_OSTADIX_" + b"NPU_PAYLOAD_V1\n"
    digest = hashlib.sha256()
    total = 0
    found = False
    finished = False
    with Path(source_path).open("rb") as source, destination.open("xb") as output:
        for line in source:
            if not found:
                if line == start:
                    found = True
                continue
            if line == end:
                finished = True
                if source.read().strip():
                    raise ValueError("Unexpected data after embedded compiler payload")
                break
            if not line.startswith(b"# "):
                raise ValueError("Malformed embedded compiler payload")
            chunk = base64.b64decode(line[2:].rstrip(b"\r\n"), validate=True)
            total += len(chunk)
            if total > ARCHIVE_BYTES:
                raise ValueError("Embedded compiler payload exceeds recorded size")
            digest.update(chunk)
            output.write(chunk)
    if not found or not finished or total != ARCHIVE_BYTES or digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Embedded compiler archive hash or size mismatch")
    return destination


def _request_byte_limit():
    fallback = 16 * 1024**2
    try:
        arg_max = os.sysconf("SC_ARG_MAX")
    except (AttributeError, OSError, ValueError):
        return fallback
    if not isinstance(arg_max, int) or arg_max <= 0:
        return fallback
    # A single original byte can expand to six ASCII JSON characters. Include
    # source-path/array overhead while retaining a finite host-derived bound.
    return max(fallback, 6 * arg_max + 64 * 1024)


def _request():
    request_file = os.environ.get("OSTADIX_NPU_ARGV_FILE")
    if request_file:
        limit = _request_byte_limit()
        descriptor = os.open(request_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(descriptor, "rb") as source:
            metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > limit:
                raise ValueError("Argument request must be a regular file no larger than " + str(limit) + " bytes")
            data = source.read(limit + 1)
            if len(data) > limit:
                raise ValueError("Argument request exceeds " + str(limit) + " bytes")
        value = json.loads(data)
    else:
        value = json.loads(os.environ.get("OSTADIX_NPU_ARGV", "[]"))
    if not isinstance(value, list) or not all(isinstance(arg, str) for arg in value):
        raise ValueError("OSTADIX_NPU_ARGV must be a JSON array of strings")
    if value:
        return Path(value[0]).expanduser().absolute(), value[1:]
    source = os.environ.get("OSTADIX_NPU_SOURCE", "/root/pixelforge-npu.O")
    args = json.loads(os.environ.get("OSTADIX_NPU_ARGS_JSON", '["help"]'))
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        raise ValueError("OSTADIX_NPU_ARGS_JSON must be a JSON array of strings")
    return Path(source).expanduser().absolute(), args


def materialize_kit(kit_dir=None):
    source, _args = _request()
    root = (Path(kit_dir).expanduser().absolute() if kit_dir else
            Path.home() / ".local/share/ostadix/pixelforge-npu" / ARCHIVE_SHA256[:16])
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_path = root.parent / ("." + root.name + ".lock")
    descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if root.exists() or root.is_symlink():
            return _verify_kit(root)
        required = ARCHIVE_BYTES + sum(item["bytes"] for item in ARCHIVE_FILES) + 16 * 1024**2
        if shutil.disk_usage(root.parent).free < required:
            raise OSError("Not enough disk to extract the compiler kit; requires about "
                          + str(required // 1024**2) + " MiB temporarily")
        with tempfile.TemporaryDirectory(prefix=".pixelforge-extract-", dir=root.parent) as temporary:
            temporary = Path(temporary)
            packed = _decode_archive(source, temporary / "compiler.run")
            destination = temporary / "kit"
            environment = os.environ.copy()
            # The kit's shell extractor invokes python3. Use the Ubuntu Python
            # installation, independent of an inherited Termux/Bionic PATH.
            environment["PATH"] = "/usr/bin:/bin:" + environment.get("PATH", "")
            result = subprocess.run(["/bin/sh", str(packed), str(destination)],
                                    stdin=subprocess.DEVNULL, capture_output=True,
                                    text=True, env=environment, timeout=180)
            if result.returncode:
                raise RuntimeError("Compiler extraction failed: " + result.stderr[-6000:])
            _verify_kit(destination)
            destination.rename(root)
    return root


def _write_result(result):
    destination = os.environ.get("OSTADIX_NPU_RESULT_FILE")
    if destination:
        with open(destination, "x", encoding="utf8") as output:
            json.dump(result, output)
            output.write("\n")


def _find_ostadix():
    roots = [os.environ.get("O_LANG_ROOT"),
             "/data/data/com.termux/files/home/ostadix-BM",
             str(Path.home() / "ostadix-BM")]
    root = next((Path(item).expanduser() for item in roots
                 if item and (Path(item).expanduser() / "backends/python_shim.py").is_file()), None)
    if root is None:
        raise FileNotFoundError("Set O_LANG_ROOT to your ostadix-BM checkout")
    candidates = [os.environ.get("OSTADIX_NPU_O"), str(Path.home() / ".local/bin/O"),
                  str(root / "target/release/O"), shutil.which("O")]
    executable = next((item for item in candidates if item and Path(item).is_file()
                       and os.access(item, os.X_OK)), None)
    if executable is None:
        raise FileNotFoundError("Install ostadix-BM's O evaluator or set OSTADIX_NPU_O")
    return executable, root


def _o_parent_death(parent_pid):
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        os._exit(126)
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def _bootstrap():
    source = Path(sys.argv[1]).expanduser().absolute()
    args = sys.argv[2:]
    executable, root = _find_ostadix()
    environment = os.environ.copy()
    environment["O_LANG_ROOT"] = str(root)
    environment["O_BACKENDS_DIR"] = str(root / "backends")
    environment["OSTADIX_NPU_CALLER_CWD"] = os.getcwd()
    launcher_pid = os.getpid()
    environment["OSTADIX_NPU_LAUNCHER_PID"] = str(launcher_pid)
    environment["OSTADIX_NPU_LAUNCHER_START_TICKS"] = Path(
        "/proc/" + str(launcher_pid) + "/stat").read_text().rsplit(")", 1)[1].split()[19]
    # One environment string is limited to about 128 KiB on Linux. Escaped
    # Unicode JSON can exceed it even when the original argv fits comfortably.
    # Keep the complete literal request in this launch's private directory.
    environment.pop("OSTADIX_NPU_ARGV", None)
    environment.pop("OSTADIX_NPU_ARGS_JSON", None)
    environment["O_BACKEND_OPERATION_TIMEOUT_MS"] = "3600000"
    with tempfile.TemporaryDirectory(prefix="pixelforge-o-control-") as temporary:
        request_path = Path(temporary) / "request.json"
        request_fd = os.open(request_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        with os.fdopen(request_fd, "w", encoding="utf8") as request:
            # JSON escapes also preserve Python's surrogateescaped Unix argv
            # bytes; strict UTF-8 cannot encode those strings directly.
            json.dump([str(source)] + args, request, ensure_ascii=True)
        environment["OSTADIX_NPU_ARGV_FILE"] = str(request_path)
        result_path = Path(temporary) / "result.json"
        cancel_path = Path(temporary) / "cancel"
        environment["OSTADIX_NPU_RESULT_FILE"] = str(result_path)
        environment["OSTADIX_NPU_CANCEL_FILE"] = str(cancel_path)
        process = subprocess.Popen([executable, str(source), str(root / "backends")],
                                   env=environment, start_new_session=True,
                                   preexec_fn=lambda: _o_parent_death(launcher_pid))
        interrupted_at = None

        def interrupt(signum, _frame):
            nonlocal interrupted_at
            if interrupted_at is None:
                interrupted_at = time.monotonic()
                cancel_path.write_text(str(signum) + "\n")
                print("Cancelling the owned command and waiting for cleanup…", file=sys.stderr)

        previous = {sig: signal.signal(sig, interrupt)
                    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            while process.poll() is None:
                if interrupted_at is not None and time.monotonic() - interrupted_at > 100:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(0.1)
            native_status = process.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            status = result.get("exit_code", 1)
            if type(status) is not int or not 0 <= status <= 255:
                status = 1
            if status == 0 and native_status != 0:
                status = min(255, native_status if native_status > 0 else 128 - native_status)
        else:
            status = 130 if interrupted_at is not None else (native_status if native_status > 0 else 1)
        raise SystemExit(status)


def _execute_integrated():
    try:
        _source, args = _request()
        result = integration_main(args, materialize_kit)
    except Exception as error:
        result = {"exit_code": 1, "success": False,
                  "error": type(error).__name__ + ": " + str(error)}
    _write_result(result)
    if result.get("exit_code", 1):
        print(json.dumps(result), file=sys.stderr)
        raise RuntimeError("PixelForge command failed with exit code " + str(result["exit_code"]))
    # The backend carries this JSON as a UTF-8 CBOR string. Escape Unix path
    # surrogate bytes so a successful command also crosses that boundary.
    return json.dumps(result, ensure_ascii=True)
