#!/usr/bin/env python3
"""Exercise source-first MCP execution against a disposable TLS loopback node.

This gate uses two real native processes, ostadix-mcp and o-node, and octl
children. It does not start, stop, enroll, or modify a user's normal node.
Hosted V1 submits one complete document; this is not graph partitioning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
import uuid

# Keep importing the transport helpers free of source-tree bytecode writes.
sys.dont_write_bytecode = True
from smoke_ostadix_mcp import (  # noqa: E402
    PROTOCOL_VERSION,
    ResponseReader,
    SmokeError,
    _content_object,
    _content_text,
    _send,
)


def _stop_owned(process: subprocess.Popen[bytes] | None) -> None:
    """Terminate and reap only the exact child owned by this fixture."""
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        process.wait(timeout=10)
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _checked_command(
    arguments: list[str], environment: dict[str, str], cwd: Path, timeout: float
) -> subprocess.CompletedProcess[bytes]:
    completed = subprocess.run(
        arguments, cwd=cwd, env=environment, capture_output=True, timeout=timeout
    )
    if completed.returncode != 0:
        raise SmokeError(
            f"native fixture command failed ({completed.returncode}): {arguments}\n"
            f"{completed.stdout.decode('utf-8', 'replace')}\n"
            f"{completed.stderr.decode('utf-8', 'replace')}"
        )
    return completed


def _integer(value: Any) -> int:
    if (
        not isinstance(value, dict)
        or value.get("t") != "number"
        or not isinstance(value.get("v"), dict)
        or value["v"].get("kind") != "int"
    ):
        raise SmokeError(f"expected native integer OValue, got {value!r}")
    return int(value["v"]["v"])


def _remote_contract(
    result: dict[str, Any], completion: str, timeouts: tuple[int, int] | None = None
) -> None:
    remote = result.get("remote_execution", {})
    expected = {
        "completion": completion,
        "automatic_retry": False,
        "cancellation_scope": "local-client-only",
        "remote_effects_may_continue": True,
        "deadline_cancels_effects": False,
    }
    if timeouts is not None:
        expected["native_io_timeout_secs"] = timeouts[0]
        expected["native_publication_deadline_secs"] = timeouts[1]
    if any(remote.get(key) != value for key, value in expected.items()):
        raise SmokeError(f"node execution misstated completion, cancellation, or native timeouts: {remote}")
    placement = result.get("placement", {})
    if placement.get("local_fallback") is not False or placement.get("ordinary_graph_partitioning") is not False:
        raise SmokeError(f"node execution overstated its native placement scope: {placement}")
    if result.get("input", {}).get("context_scope") != "local-client-only":
        raise SmokeError("node execution did not explain its local cwd/environment scope")


def _receipt(
    result: dict[str, Any], node_id: str, source: str, *, succeeded: bool
) -> dict[str, Any]:
    native = result.get("result")
    if not isinstance(native, dict):
        raise SmokeError(f"node execution omitted native receipt: {result}")
    if native.get("schema") != "ostadix.hosted-operation-receipt/v1":
        raise SmokeError(f"node returned unexpected receipt schema: {native}")
    if native.get("node_id") != node_id:
        raise SmokeError(f"receipt selected the wrong node: {native}")
    if native.get("source_sha256") != hashlib.sha256(source.encode()).hexdigest():
        raise SmokeError(f"receipt source digest did not bind the submitted document: {native}")
    for field in ("operation_sha256", "backend_catalog_sha256", "outcome_sha256", "receipt_sha256"):
        digest = native.get(field)
        if not isinstance(digest, str) or len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise SmokeError(f"receipt omitted native {field}: {native}")
    for field in ("task_id", "attempt_id"):
        if not isinstance(native.get(field), str) or not native[field]:
            raise SmokeError(f"receipt omitted generated {field}: {native}")
    expected = "succeeded" if succeeded else "failed"
    _remote_contract(result, expected)
    if native.get("outcome", {}).get("status") != expected:
        raise SmokeError(f"receipt expected {expected}: {native}")
    if (
        result.get("state") != ("completed" if succeeded else "failed")
        or not isinstance(result.get("exit_code"), int)
        or (result["exit_code"] == 0) != succeeded
    ):
        raise SmokeError(f"native receipt and client process status disagree: {result}")
    if result.get("placement", {}).get("route") != "selected-node-document":
        raise SmokeError(f"node execution omitted selected-document placement: {result}")
    stdout = result.get("stdout", {}).get("text")
    if not isinstance(stdout, str) or json.loads(stdout) != native:
        raise SmokeError(f"projected receipt and retained native stdout disagree: {result}")
    if succeeded and result.get("value") != native["outcome"].get("value"):
        raise SmokeError(f"node execution did not return its typed value directly: {result}")
    return native


def run_smoke(root: Path, binary: Path, runtime_bin_dir: Path, timeout: float) -> dict[str, Any]:
    commands = {name: runtime_bin_dir / name for name in ("O", "o-node", "octl")}
    for path in (binary, *commands.values()):
        if not path.is_file() or not os.access(path, os.X_OK):
            raise SmokeError(f"required native executable is unavailable: {path}")
    if not (root / "backends/python_shim.py").is_file():
        raise SmokeError(f"backend directory is unavailable beneath {root}")
    if timeout <= 0:
        raise SmokeError("--timeout must be positive")

    evidence: dict[str, Any] = {
        "schema": "ostadix.mcp-node-smoke/v1",
        "root": os.fspath(root),
        "mcp_binary": os.fspath(binary),
        "runtime_bin_dir": os.fspath(runtime_bin_dir),
        "transport": "loopback-tls1.3-mutual-x509",
        "checks": [],
        "ordinary_graph_partitioning_tested": False,
        "remote_cancellation_supported": False,
    }
    with tempfile.TemporaryDirectory(prefix="ostadix-mcp-node-smoke-") as temporary:
        fixture = Path(temporary).resolve()
        node_id = f"mcp-smoke-{uuid.uuid4().hex}"
        pki = fixture / "pki"
        client_cwd = fixture / "client-work"
        node_cwd = fixture / "node-work"
        client_cwd.mkdir()
        node_cwd.mkdir()
        barrier = node_cwd / "release-background"
        ready_marker = node_cwd / "background-ready"
        environment = os.environ.copy()
        for key in ("OSTADIX_RUNTIME_PATH", "OLANG", "A18_WORK"):
            environment.pop(key, None)
        environment.update(
            {
                "O_LANG_ROOT": os.fspath(root),
                "O_BACKENDS_DIR": os.fspath(root / "backends"),
                "PATH": os.fspath(runtime_bin_dir) + os.pathsep + environment.get("PATH", ""),
                "XDG_CONFIG_HOME": os.fspath(fixture / "config"),
                "XDG_STATE_HOME": os.fspath(fixture / "state"),
                "XDG_CACHE_HOME": os.fspath(fixture / "cache"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "RUST_LOG": "warn",
            }
        )
        probe_key = "OSTADIX_MCP_NODE_SMOKE_SERVER_VALUE"
        environment.pop(probe_key, None)
        node_environment = environment.copy()
        node_environment[probe_key] = "271828"

        _checked_command(
            [os.fspath(commands["o-node"]), "pki", "init", "--directory", os.fspath(pki),
             "--server-name", "localhost"],
            environment, fixture, timeout,
        )
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        address = f"127.0.0.1:{port}"
        peer_directory = fixture / "config/ostadix/peers" / node_id
        peer_directory.mkdir(parents=True, mode=0o700)
        for filename in ("ca.pem", "client-cert.pem", "client-key.pem"):
            destination = peer_directory / filename
            shutil.copyfile(pki / filename, destination)
            destination.chmod(0o600)
        peer_metadata = peer_directory / "peer.json"
        peer_metadata.write_text(json.dumps({
            "schema": "ostadix.lan-peer/v1", "node_id": node_id,
            "server_name": "localhost", "address": address, "service_port": port,
            "security_mode": "paired-public-key", "supports_v2": False,
        }), encoding="utf-8")
        peer_metadata.chmod(0o600)

        node: subprocess.Popen[bytes] | None = None
        process: subprocess.Popen[bytes] | None = None
        responses: ResponseReader | None = None
        # File-backed diagnostics cannot deadlock when a backend writes stderr.
        with (fixture / "node.stderr").open("w+b") as node_stderr, (
            fixture / "mcp.stderr"
        ).open("w+b") as mcp_stderr:
            try:
                node = subprocess.Popen(
                    [os.fspath(commands["o-node"]), "serve", "--manual", "--node-id", node_id,
                     "--runtime-binary", os.fspath(commands["O"]), "--shim-dir", os.fspath(root / "backends"),
                     "--bind", address, "--cert", os.fspath(pki / "node-cert.pem"),
                     "--key", os.fspath(pki / "node-key.pem"), "--client-ca", os.fspath(pki / "ca.pem")],
                    cwd=node_cwd, env=node_environment, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=node_stderr,
                )
                readiness_deadline = time.monotonic() + timeout
                while True:
                    profile = subprocess.run(
                        [os.fspath(commands["octl"]), "node", "profile", "--manual", "--address", address,
                         "--server-name", "localhost", "--ca", os.fspath(pki / "ca.pem"),
                         "--cert", os.fspath(pki / "client-cert.pem"), "--key", os.fspath(pki / "client-key.pem"),
                         "--connect-timeout-seconds", "1", "--io-timeout-seconds", "1"],
                        cwd=client_cwd, env=environment, capture_output=True, timeout=5,
                    )
                    if profile.returncode == 0:
                        if json.loads(profile.stdout).get("node_id") != node_id:
                            raise SmokeError("loopback readiness returned a different node identity")
                        break
                    if node.poll() is not None or time.monotonic() >= readiness_deadline:
                        raise SmokeError(f"disposable node did not become ready: {profile.stderr.decode('utf-8', 'replace')}")
                    time.sleep(0.05)

                process = subprocess.Popen(
                    [os.fspath(binary)], cwd=client_cwd, env=environment,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=mcp_stderr, bufsize=0,
                )
                if process.stdout is None:
                    raise SmokeError("MCP stdout is unavailable")
                responses = ResponseReader(process.stdout)
                request_id = 0

                def request(method: str, params: dict[str, Any]) -> dict[str, Any]:
                    nonlocal request_id
                    request_id += 1
                    _send(process, {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
                    return responses.response(request_id, timeout)

                def call(name: str, arguments: dict[str, Any], *, error: bool = False) -> dict[str, Any]:
                    result = request("tools/call", {"name": name, "arguments": arguments})
                    if (result.get("isError") is True) != error:
                        raise SmokeError(f"{name} returned unexpected error status: {_content_text(result)}")
                    return _content_object(result)

                initialized = request("initialize", {
                    "protocolVersion": PROTOCOL_VERSION, "capabilities": {},
                    "clientInfo": {"name": "ostadix-node-smoke", "version": "1"},
                })
                if initialized.get("protocolVersion") != PROTOCOL_VERSION:
                    raise SmokeError("MCP initialized with an unexpected protocol version")
                _send(process, {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
                capabilities = call("o_capabilities", {"query": "octl"})
                octl = next((entry for entry in capabilities.get("commands", []) if entry.get("id") == "octl"), {})
                if Path(octl.get("resolved_path", "")).resolve() != commands["octl"].resolve():
                    raise SmokeError(f"MCP selected a different octl than the fixture: {octl}")
                evidence["checks"].append("exact-native-client-provenance")

                source = f"python^(\nimport os\n__oval_result__ = int(os.environ[{probe_key!r}])\n)_python\n"
                inline = call("o_execute", {
                    "source": source, "node": node_id, "cwd": os.fspath(client_cwd),
                    "env": {probe_key: "99"},
                })
                receipt = _receipt(inline, node_id, source, succeeded=True)
                _remote_contract(inline, "succeeded", (60, 300))
                if _integer(inline["value"]) != 271828:
                    raise SmokeError("submitted document did not execute in the remote server's environment")
                evidence["checks"].append("complete-source-remote-environment-and-receipt")
                evidence["source_receipt_sha256"] = receipt["receipt_sha256"]

                path_source = "python^( __oval_result__ = 6 * 7 )_python\n"
                program = client_cwd / "complete document.O"
                program.write_text(path_source, encoding="utf-8")
                by_path = call("o_execute", {"path": os.fspath(program), "node": node_id, "timeout_secs": 30})
                _receipt(by_path, node_id, path_source, succeeded=True)
                _remote_contract(by_path, "succeeded", (30, 30))
                if _integer(by_path["value"]) != 42:
                    raise SmokeError("path execution returned the wrong native value")
                evidence["checks"].append("complete-path-with-spaces")

                unbounded_client = call("o_execute", {"path": os.fspath(program), "node": node_id, "timeout_secs": 0})
                _receipt(unbounded_client, node_id, path_source, succeeded=True)
                _remote_contract(unbounded_client, "succeeded", (60, 300))
                evidence["checks"].append("client-timeout-zero-preserves-native-bounds")

                failed_source = "python^( raise RuntimeError('node-smoke-evaluation-failure') )_python\n"
                failed = call("o_execute", {"source": failed_source, "node": node_id}, error=True)
                failed_receipt = _receipt(failed, node_id, failed_source, succeeded=False)
                if failed_receipt["outcome"].get("stage") != "evaluate":
                    raise SmokeError(f"remote evaluation failure lost its native stage: {failed_receipt}")
                evidence["checks"].append("failed-evaluation-preserves-receipt")

                # The second document releases the first; a globally serialized
                # client/server path fails, without an elapsed-time overlap guess.
                waiting_source = (
                    "python^(\nfrom pathlib import Path\nimport time\n"
                    f"barrier = Path({os.fspath(barrier)!r})\n"
                    f"Path({os.fspath(ready_marker)!r}).write_text('ready')\n"
                    "deadline = time.monotonic() + 20\n"
                    "while not barrier.exists():\n"
                    "    if time.monotonic() > deadline: raise RuntimeError('concurrent node execution stalled')\n"
                    "    time.sleep(0.02)\n"
                    "__oval_result__ = 11\n)_python\n"
                )
                waiting = call("o_execute", {"source": waiting_source, "node": node_id, "background": True, "timeout_secs": 30})
                _remote_contract(waiting, "unknown", (30, 30))
                if waiting.get("state") != "running" or not waiting.get("job_id"):
                    raise SmokeError(f"background node submission did not return a live client job: {waiting}")
                ready_deadline = time.monotonic() + min(timeout, 15)
                while not ready_marker.exists():
                    if time.monotonic() >= ready_deadline:
                        raise SmokeError("background node document did not begin evaluation")
                    time.sleep(0.02)
                releasing_source = (
                    "python^(\nfrom pathlib import Path\n"
                    f"Path({os.fspath(barrier)!r}).write_text('release')\n"
                    "__oval_result__ = 12\n)_python\n"
                )
                releasing = call("o_execute", {"source": releasing_source, "node": node_id, "timeout_secs": 30})
                _receipt(releasing, node_id, releasing_source, succeeded=True)
                wait_deadline = time.monotonic() + timeout
                while True:
                    status = call("o_job_status", {"job_id": waiting["job_id"]})
                    if status.get("state") != "running":
                        if status.get("state") != "completed" or status.get("exit_code") != 0:
                            raise SmokeError(f"background node execution failed: {status}")
                        break
                    if time.monotonic() >= wait_deadline:
                        raise SmokeError("background node client did not complete")
                    time.sleep(0.05)
                page = call("o_job_read", {"job_id": waiting["job_id"], "stream": "stdout", "offset": 0, "limit": 65536})
                if page.get("eof") is not True:
                    raise SmokeError("small background receipt unexpectedly exceeded one log page")
                background_receipt = json.loads(page["text"])
                if background_receipt.get("node_id") != node_id or _integer(background_receipt["outcome"]["value"]) != 11:
                    raise SmokeError(f"background log omitted its native remote value: {background_receipt}")
                evidence["checks"].append("concurrent-background-document-progress")

                job_ids_before = {job["job_id"] for job in call("o_job_list", {}).get("jobs", [])}
                invalid_arguments = [
                    {"action": "check"}, {"action": "plan"},
                    {"action": "compile", "target": "ir"}, {"mode": "admitted"},
                    {"placement": "local"}, {"placement": "mesh-required"},
                    {"workers": 2}, {"stdin": "application input"}, {"route": "build"},
                ]
                for invalid in invalid_arguments:
                    call("o_execute", {"source": path_source, "node": node_id, **invalid}, error=True)
                project = fixture / "project"
                project.mkdir()
                call("o_execute", {"path": os.fspath(project), "node": node_id}, error=True)
                bundle = "# O-PROJECT-BUNDLE-V1 BEGIN\n#olang-bundle-payload-begin\n"
                call("o_execute", {"source": bundle, "node": node_id}, error=True)
                job_ids_after = {job["job_id"] for job in call("o_job_list", {}).get("jobs", [])}
                if job_ids_after != job_ids_before:
                    raise SmokeError("unsupported node options launched native jobs before rejection")
                evidence["checks"].append("unsupported-combinations-rejected-before-launch")

                fallback_marker = client_cwd / "must-not-execute-locally"
                fallback_source = (
                    "python^(\nfrom pathlib import Path\n"
                    f"Path({os.fspath(fallback_marker)!r}).write_text('unexpected fallback')\n"
                    "__oval_result__ = 13\n)_python\n"
                )
                missing = call("o_execute", {"source": fallback_source, "node": node_id + "-missing"}, error=True)
                _remote_contract(missing, "unknown", (60, 300))
                if missing.get("state") == "completed" or fallback_marker.exists():
                    raise SmokeError(f"missing selected node silently executed locally: {missing}")
                evidence["checks"].append("unknown-selected-node-has-no-local-fallback")

                oversize = "#" + "x" * (1024 * 1024) + "\n" + fallback_source
                rejected = call("o_execute", {"source": oversize, "node": node_id}, error=True)
                if rejected.get("state") == "completed" or fallback_marker.exists():
                    raise SmokeError("oversized node source silently executed or fell back locally")
                diagnostic = json.dumps(rejected)
                if not any(word in diagnostic.lower() for word in ("maximum", "limit", "1048576", "1 mib", "1mib")):
                    raise SmokeError("oversized node source did not explain its native bound")
                evidence["checks"].append("native-source-bound-is-explicit")

                # Native selection must have persisted solely in the disposable
                # XDG registry supplied to this server and all of its children.
                preferred = fixture / "config/ostadix/peers/_preferred"
                if preferred.read_text(encoding="utf-8").strip() != node_id:
                    raise SmokeError("native peer preference did not use the isolated config root")
                evidence["checks"].append("isolated-peer-registry")
                evidence["node_id"] = node_id
                evidence["status"] = "passed"
            except (OSError, SmokeError, ValueError, subprocess.TimeoutExpired) as error:
                for label, stream in (("node", node_stderr), ("MCP", mcp_stderr)):
                    stream.flush()
                    stream.seek(0)
                    diagnostics = stream.read().decode("utf-8", "replace")
                    if diagnostics:
                        print(f"{label} stderr:\n{diagnostics}", file=sys.stderr)
                raise SmokeError(str(error)) from error
            finally:
                # Release the only potentially waiting remote evaluation before
                # shutting down either owned process, including failure paths.
                barrier.touch(exist_ok=True)
                if process is not None and process.stdin is not None:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                _stop_owned(process)
                _stop_owned(node)
                if responses is not None:
                    responses.join(5)
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--runtime-bin-dir", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--evidence", type=Path, help="Optional destination for the successful JSON evidence record")
    arguments = parser.parse_args()
    root = arguments.root.expanduser().resolve()
    binary = (arguments.binary or root / "mcp/ostadix_lang_mcp_server/target/release/ostadix-mcp").expanduser().resolve()
    runtime_bin_dir = (arguments.runtime_bin_dir or root / "target/release").expanduser().resolve()
    try:
        evidence = run_smoke(root, binary, runtime_bin_dir, arguments.timeout)
        if arguments.evidence:
            arguments.evidence.expanduser().resolve().write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    except (OSError, SmokeError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"ostadix-mcp selected-node loopback smoke: PASS ({len(evidence['checks'])} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
