#!/usr/bin/env python3
"""Exercise the released ostadix-mcp server over its real stdio transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, BinaryIO


PROTOCOL_VERSION = "2025-03-26"
EXPECTED_TOOLS = {
    "o_analyze_intent",
    "o_capabilities",
    "o_cli",
    "o_doctor",
    "o_env",
    "o_eval",
    "o_execute",
    "o_execute_intent",
    "o_information_inspect",
    "o_guide",
    "o_job_cancel",
    "o_job_list",
    "o_job_read",
    "o_job_status",
    "o_job_write",
    "o_olangc",
    "o_run",
    "o_runtimes",
    "o_search_run",
    "o_smoke",
}


def _snapshot_tree(root: Path) -> tuple[tuple[Any, ...], ...]:
    entries: list[tuple[Any, ...]] = []
    for path in sorted(root.rglob("*")):
        metadata = path.lstat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        entries.append(
            (
                path.relative_to(root).as_posix(),
                path.is_dir(),
                metadata.st_ino,
                metadata.st_mode,
                metadata.st_size,
                metadata.st_mtime_ns,
                digest,
            )
        )
    return tuple(entries)


class SmokeError(RuntimeError):
    """The MCP server did not satisfy its released transport contract."""


_EOF = object()


def _current_catalog_schema(root: Path) -> str:
    """Read the one authoritative catalog-generation identifier."""
    catalog = root / "crates" / "ostadix-api" / "src" / "backend_catalog.inc.rs"
    match = re.search(
        r'backend_catalog_metadata!\s*\{\s*current_schema:\s*"([^"]+)"',
        catalog.read_text(encoding="utf-8"),
        re.DOTALL,
    )
    if match is None:
        raise SmokeError(f"backend catalog does not declare current_schema: {catalog}")
    return match.group(1)


class ResponseReader:
    """Drain newline-framed MCP stdout continuously and retain replies by id."""

    def __init__(self, stream: BinaryIO) -> None:
        self._stream = stream
        self._frames: queue.Queue[bytes | BaseException | object] = queue.Queue()
        self._pending: dict[int, dict[str, Any]] = {}
        self._thread = threading.Thread(
            target=self._read_frames,
            name="ostadix-mcp-stdout",
            daemon=True,
        )
        self._thread.start()

    def _read_frames(self) -> None:
        try:
            while line := self._stream.readline():
                self._frames.put(line)
        except BaseException as error:  # surfaced synchronously by response()
            self._frames.put(error)
        finally:
            self._frames.put(_EOF)

    def response(self, request_id: int, timeout: float) -> dict[str, Any]:
        waiting = self._pending.pop(request_id, None)
        if waiting is not None:
            return _checked_result(waiting, request_id)

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SmokeError(f"timeout waiting for MCP response {request_id}")
            try:
                frame = self._frames.get(timeout=remaining)
            except queue.Empty as error:
                raise SmokeError(
                    f"timeout waiting for MCP response {request_id}"
                ) from error
            if frame is _EOF:
                raise SmokeError(
                    f"MCP stdout closed while waiting for response {request_id}"
                )
            if isinstance(frame, BaseException):
                raise SmokeError(f"failed reading MCP stdout: {frame}") from frame
            try:
                message = json.loads(frame.decode("utf-8", "strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SmokeError(f"invalid MCP JSON response: {frame!r}") from error
            if not isinstance(message, dict):
                raise SmokeError(f"MCP response is not a JSON object: {frame!r}")
            response_id = message.get("id")
            if response_id == request_id:
                return _checked_result(message, request_id)
            if isinstance(response_id, int):
                if response_id in self._pending:
                    raise SmokeError(f"duplicate MCP response id {response_id}")
                self._pending[response_id] = message

    def join(self, timeout: float) -> None:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise SmokeError("MCP stdout reader did not stop")


def _send(process: subprocess.Popen[bytes], message: dict[str, Any]) -> None:
    if process.stdin is None:
        raise SmokeError("MCP stdin is unavailable")
    process.stdin.write(
        json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )
    process.stdin.flush()


def _checked_result(message: dict[str, Any], request_id: int) -> dict[str, Any]:
    if "error" in message:
        raise SmokeError(f"MCP request {request_id} failed: {message['error']}")
    result = message.get("result")
    if not isinstance(result, dict):
        raise SmokeError(f"MCP response {request_id} has no object result")
    return result


def _content_text(result: dict[str, Any]) -> str:
    content = result.get("content")
    if not isinstance(content, list):
        raise SmokeError("MCP tool result has no content list")
    pieces = [
        item.get("text", "")
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    if not pieces:
        raise SmokeError("MCP tool result has no text content")
    return "\n".join(pieces)


def _record_field(text: str, key: str) -> str:
    prefix = f"{key}="
    for line in text.splitlines():
        if line.startswith(prefix):
            value = line[len(prefix) :]
            if value:
                return value
    raise SmokeError(f"MCP result omitted nonempty {key}= record:\n{text}")


def _content_object(result: dict[str, Any]) -> dict[str, Any]:
    """Read the structured contract, checking its text-only client equivalent."""
    try:
        text_value = json.loads(_content_text(result))
    except json.JSONDecodeError as error:
        raise SmokeError("MCP structured tool returned invalid JSON text") from error
    if not isinstance(text_value, dict):
        raise SmokeError("MCP structured tool returned a non-object JSON value")
    structured = result.get("structuredContent", text_value)
    if not isinstance(structured, dict) or structured != text_value:
        raise SmokeError("MCP structuredContent and JSON text disagree")
    return structured


def _native_execute_result(result: dict[str, Any]) -> dict[str, Any]:
    """Require native JSON and the retained process evidence to agree."""
    if not isinstance(result.get("job_id"), str) or not result["job_id"]:
        raise SmokeError(f"o_execute omitted retained job identity: {result}")
    native = result.get("result")
    if not isinstance(native, dict):
        raise SmokeError(f"o_execute omitted its native result object: {result}")
    stdout = result.get("stdout", {})
    if not isinstance(stdout, dict) or not isinstance(stdout.get("text"), str):
        raise SmokeError(f"o_execute omitted raw stdout evidence: {result}")
    try:
        raw = json.loads(stdout["text"])
    except json.JSONDecodeError as error:
        raise SmokeError(f"o_execute raw stdout is not native JSON: {stdout}") from error
    if raw != native:
        raise SmokeError(f"o_execute native result disagrees with raw stdout: {result}")
    succeeded = native.get("ok") is True
    expected_state = "completed" if succeeded else "failed"
    if result.get("state") != expected_state or (
        (result.get("exit_code") == 0) != succeeded
        or not isinstance(result.get("exit_code"), int)
    ):
        raise SmokeError(f"o_execute native result disagrees with process outcome: {result}")
    return native


def _run_agent_surface_smoke(
    process: subprocess.Popen[bytes],
    responses: ResponseReader,
    root: Path,
    timeout: float,
    fixture: Path,
) -> None:
    """Exercise agent discovery and job transport using local disposable effects."""
    next_request = 100

    def request(method: str, params: dict[str, Any]) -> dict[str, Any]:
        nonlocal next_request
        request_id = next_request
        next_request += 1
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params,
            },
        )
        return responses.response(request_id, timeout)

    def call(
        name: str, arguments: dict[str, Any], *, error: bool = False
    ) -> dict[str, Any]:
        result = request("tools/call", {"name": name, "arguments": arguments})
        if (result.get("isError") is True) != error:
            raise SmokeError(f"{name} returned unexpected error status: {_content_text(result)}")
        return _content_object(result)

    def wait_job(job_id: str, expected: str = "completed") -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = call("o_job_status", {"job_id": job_id})
            if status.get("state") != "running":
                if status.get("state") != expected:
                    raise SmokeError(f"job {job_id} expected {expected}: {status}")
                return status
            time.sleep(0.05)
        raise SmokeError(f"job {job_id} did not finish within {timeout}s")

    def read_all(job_id: str, stream: str, limit: int = 512) -> str:
        offset = 0
        chunks: list[str] = []
        while True:
            page = call(
                "o_job_read",
                {"job_id": job_id, "stream": stream, "offset": offset, "limit": limit},
            )
            if page.get("offset") != offset or not isinstance(page.get("text"), str):
                raise SmokeError(f"job read returned an invalid cursor/text: {page}")
            advance = page.get("next_offset")
            count = page.get("bytes_read")
            if not isinstance(advance, int) or not isinstance(count, int):
                raise SmokeError(f"job read omitted byte cursor metadata: {page}")
            if not (0 <= count <= limit) or advance != offset + count:
                raise SmokeError(f"job read violated byte pagination: {page}")
            chunks.append(page["text"])
            if page.get("eof") is True:
                return "".join(chunks)
            if advance <= offset:
                raise SmokeError(f"job read stalled before EOF: {page}")
            offset = advance

    capabilities = call("o_capabilities", {})
    commands = capabilities.get("commands", [])
    command_names = {command.get("id") for command in commands if isinstance(command, dict)}
    required = {"O", "olangc", "o-link", "o-unlink", "ocorec", "octl", "o-node", "ogit"}
    if not required.issubset(command_names):
        raise SmokeError(f"o_capabilities omitted commands: {sorted(required - command_names)}")
    if capabilities.get("schema") != "ostadix.mcp-capabilities/v1" or capabilities.get("runtime_readiness_verified") is not False:
        raise SmokeError("capability discovery omitted its schema or presence-only boundary")
    located_o = next(command for command in commands if command.get("id") == "O")
    if located_o.get("available") is not True or not Path(located_o.get("resolved_path", "")).is_file():
        raise SmokeError(f"capability discovery did not locate the installed interpreter: {located_o}")
    front_door = next(command for command in commands if command.get("id") == "o")
    native_cli = next(command for command in commands if command.get("id") == "o-cli")
    if (front_door.get("available") is not True or front_door.get("prefix_args") != []
            or front_door.get("resolved_path") != native_cli.get("resolved_path")):
        raise SmokeError(f"o did not resolve directly to the native CLI: {front_door}")
    evaluated = call("o_cli", {"command": "o", "args": ["e", "python^(40 + 2)_python", "--json"]})
    if evaluated.get("exit_code") != 0:
        raise SmokeError(f"native o eval failed: {evaluated}")
    native_value = json.loads(evaluated.get("stdout", {}).get("text", "{}"))
    if native_value.get("value") != {"t": "number", "v": {"kind": "int", "v": "42"}}:
        raise SmokeError(f"native o eval changed its JSON value: {native_value}")
    guide = call("o_guide", {})
    if not isinstance(guide.get("guide"), str) or not guide["guide"].strip():
        raise SmokeError(f"o_guide omitted its agent instructions: {guide}")
    resource_list = request("resources/list", {})
    resources = resource_list.get("resources", [])
    uris = {resource.get("uri") for resource in resources if isinstance(resource, dict)}
    if not {"ostadix://capabilities", "ostadix://guide/all", "ostadix://guide/mesh"}.issubset(uris):
        raise SmokeError("MCP resources omitted the capability catalog or task guides")
    for uri, expected in (
        ("ostadix://capabilities", capabilities),
        ("ostadix://guide/all", guide["guide"]),
    ):
        resource = request("resources/read", {"uri": uri})
        contents = resource.get("contents", [])
        if len(contents) != 1 or contents[0].get("uri") != uri:
            raise SmokeError(f"MCP resource returned unexpected contents: {resource}")
        text = contents[0].get("text", "")
        actual = json.loads(text) if isinstance(expected, dict) else text
        if actual != expected:
            raise SmokeError(f"MCP resource {uri} disagrees with its tool equivalent")

    help_result = call("o_cli", {"command": "O", "args": ["--help"]})
    if help_result.get("exit_code") != 0 or "Usage:" not in help_result.get("stdout", {}).get("text", ""):
        raise SmokeError(f"o_cli O --help failed: {help_result}")
    if not isinstance(help_result.get("job_id"), str):
        raise SmokeError("foreground CLI call omitted retained job identity")

    # An environment value containing shell syntax must stay literal. Each job
    # gets its own environment, and cwd must apply to the evaluated backend.
    shell_marker = fixture / "must-not-be-created"
    literal = f"$(touch {shell_marker}) `touch {shell_marker}`"
    source = (
        "python^(\nimport json, os\n"
        "__oval_result__ = json.dumps({'value': os.environ.get('OSTADIX_MCP_SMOKE_LITERAL'), "
        "'cwd': os.getcwd()}, sort_keys=True)\n)_python\n"
    )
    evaluated = call(
        "o_eval",
        {"source": source, "cwd": os.fspath(fixture), "env": {"OSTADIX_MCP_SMOKE_LITERAL": literal}},
    )
    evaluation_text = evaluated.get("stdout", {}).get("text", "")
    if literal not in evaluation_text or os.fspath(fixture) not in evaluation_text or shell_marker.exists():
        raise SmokeError(f"inline eval changed literal env/cwd: {evaluated}")
    isolated = call("o_eval", {"source": source, "cwd": os.fspath(fixture)})
    if '"value": null' not in isolated.get("stdout", {}).get("text", ""):
        raise SmokeError(f"per-job environment escaped into later evaluation: {isolated}")

    failed = call("o_cli", {"command": "O", "args": ["--mcp-smoke-invalid-option"]}, error=True)
    if failed.get("exit_code") in (None, 0) or failed.get("state") != "failed":
        raise SmokeError(f"failed CLI command lost exit evidence: {failed}")

    # File barriers verify independent jobs make progress without timing-based
    # assumptions about worker startup or backend imports.
    barrier = fixture / "release-first-job"
    waiting_source = (
        "python^(\nfrom pathlib import Path\nimport time\n"
        f"barrier = Path({os.fspath(barrier)!r})\n"
        "while not barrier.exists():\n    time.sleep(0.02)\n"
        "__oval_result__ = 'first-job-released'\n)_python\n"
    )
    waiting = call("o_eval", {"source": waiting_source, "background": True, "timeout_secs": 30})
    listed_jobs = call("o_job_list", {})
    if waiting["job_id"] not in {job.get("job_id") for job in listed_jobs.get("jobs", [])}:
        raise SmokeError(f"job list omitted a running job: {listed_jobs}")
    releasing = call(
        "o_eval",
        {"source": "python^(\nfrom pathlib import Path\n" + f"Path({os.fspath(barrier)!r}).write_text('released')\n" + "__oval_result__ = 'second-job-completed'\n)_python\n"},
    )
    if "second-job-completed" not in releasing.get("stdout", {}).get("text", ""):
        raise SmokeError(f"second job failed while first was waiting: {releasing}")
    wait_job(waiting["job_id"])
    if "first-job-released" not in read_all(waiting["job_id"], "stdout", limit=7):
        raise SmokeError("paged background log omitted the completed result")

    repl = call(
        "o_cli",
        {"command": "O", "args": ["--repl", os.fspath(root / "backends")], "background": True, "timeout_secs": 30},
    )
    call("o_job_write", {"job_id": repl["job_id"], "input": "python^( __oval_result__ = 1 + 1 )_python\n", "close": True})
    wait_job(repl["job_id"])
    if "[number] 2" not in read_all(repl["job_id"], "stdout"):
        raise SmokeError("background REPL did not receive input and EOF")

    # The descendant writes only after cancellation should have killed its
    # process group. The ready marker proves it was actually launched first.
    ready = fixture / "descendant-ready"
    escaped = fixture / "descendant-survived"
    child_code = f"import time; from pathlib import Path; time.sleep(2); Path({os.fspath(escaped)!r}).write_text('survived')"
    spawn_descendant = (
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        if os.name == "posix"
        else ""
    )
    cancellable_source = (
        "python^(\nfrom pathlib import Path\nimport subprocess, sys, time\n"
        + spawn_descendant
        +
        f"Path({os.fspath(ready)!r}).write_text('ready')\n"
        "time.sleep(30)\n)_python\n"
    )
    cancellable = call("o_eval", {"source": cancellable_source, "background": True, "timeout_secs": 30})
    deadline = time.monotonic() + timeout
    while not ready.exists():
        if time.monotonic() >= deadline:
            raise SmokeError("cancellation fixture did not launch its descendant")
        time.sleep(0.02)
    call("o_job_cancel", {"job_id": cancellable["job_id"]})
    cancelled = wait_job(cancellable["job_id"], "cancelled")
    if cancelled.get("cleanup", {}).get("child_reaped") is not True:
        raise SmokeError(f"cancelled job omitted child reap evidence: {cancelled}")
    if os.name == "posix":
        time.sleep(2.2)
        if escaped.exists():
            raise SmokeError("cancelled job left a descendant able to commit an effect")


def _run_unified_surface_smoke(
    process: subprocess.Popen[bytes],
    responses: ResponseReader,
    root: Path,
    timeout: float,
    fixture: Path,
) -> None:
    """Exercise source-first execution through the actual MCP transport."""
    next_request = 500

    def call(
        name: str, arguments: dict[str, Any], *, error: bool = False
    ) -> dict[str, Any]:
        nonlocal next_request
        request_id = next_request
        next_request += 1
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )
        result = responses.response(request_id, timeout)
        if (result.get("isError") is True) != error:
            raise SmokeError(f"{name} returned unexpected error status: {_content_text(result)}")
        return _content_object(result)

    def execute(arguments: dict[str, Any], *, error: bool = False) -> dict[str, Any]:
        return call("o_execute", arguments, error=error)

    source = "python^( __oval_result__ = 6 * 7 )_python\n"
    program = fixture / "source with spaces.O"
    program.write_text(source, encoding="utf-8")
    for supplied in ({"source": source}, {"path": program.name, "cwd": os.fspath(fixture)}):
        result = execute(supplied)
        native = _native_execute_result(result)
        if native.get("value") != {"t": "number", "v": {"kind": "int", "v": "42"}}:
            raise SmokeError(f"o_execute changed the native numeric value: {native}")
        evidence = native.get("execution_evidence", {})
        admission = evidence.get("admission")
        identity_fields = ("oir_sha256", "plan_sha256", "analyzed_graph_sha256",
                           "evidence_sha256", "admitted_graph_sha256", "admission_sha256")
        if (
            evidence.get("schema") != "ostadix.native-execution-evidence/v1"
            or evidence.get("execution_mode") != "graph"
            or evidence.get("source_sha256") != hashlib.sha256(source.encode()).hexdigest()
            or evidence.get("parsed_source_sha256") != evidence.get("source_sha256")
            or evidence.get("source_intent_gate") is not None
            or not isinstance(admission, dict)
            or admission.get("schema") != "oexec.admission/v6"
            or any(re.fullmatch(r"[0-9a-f]{64}", admission.get(name, "")) is None
                   for name in identity_fields)
            or re.fullmatch(r"[0-9a-f]{64}", evidence.get("result_content_identity", "")) is None
        ):
            raise SmokeError(f"o_execute omitted actual native admission/input/result evidence: {native}")

    # One call must compose actual backend values. The SQL integer reaches
    # Python as an integer, and the nested text reaches it as Unicode text.
    composed = execute({
        "source": (
            "python^(\n"
            "number = sql^( SELECT 21 )_sql\n"
            "label = text^(nested OValue)_text\n"
            "__oval_result__ = {'answer': number * 2, 'label': label, "
            "'number_is_int': type(number) is int}\n)_python\n"
        ),
    })
    composed_value = _native_execute_result(composed).get("value", {})
    if composed_value != {
        "t": "map",
        "v": {
            "answer": {"t": "number", "v": {"kind": "int", "v": "42"}},
            "label": {"t": "text", "v": {"utf8": "nested OValue", "encoding": "utf-8"}},
            "number_is_int": {"t": "bool", "v": True},
        },
    }:
        raise SmokeError(f"nested heterogeneous execution lost its typed values: {composed}")

    execution_failed = execute({
        "source": "python^( raise RuntimeError('unified-backend-failure') )_python",
    }, error=True)
    failure = _native_execute_result(execution_failed)
    if failure.get("stage") != "eval" or "unified-backend-failure" not in failure.get("error", ""):
        raise SmokeError(f"backend failure lost its native structured error: {execution_failed}")

    # Admitted inline source lives in a private snapshot; relative artifacts
    # must still be written into the explicitly requested workspace.
    artifact_run = execute({
        "source": (
            "python^(\nfrom pathlib import Path\n"
            "output = Path('artifacts/relative-result.txt')\n"
            "output.parent.mkdir()\n"
            "output.write_text('workspace-artifact')\n"
            "__oval_result__ = str(output.resolve())\n)_python\n"
        ),
        "cwd": os.fspath(fixture),
        "mode": "admitted",
    })
    expected_artifact = fixture / "artifacts" / "relative-result.txt"
    reported_artifact = _native_execute_result(artifact_run).get("value", {}).get("v", {}).get("utf8")
    artifact_input = artifact_run.get("input", {})
    if (
        reported_artifact != os.fspath(expected_artifact)
        or not expected_artifact.is_file()
        or expected_artifact.read_text(encoding="utf-8") != "workspace-artifact"
        or artifact_input.get("temporary_source") is not True
        or Path(artifact_input.get("path", "")).parent == fixture
    ):
        raise SmokeError(f"inline snapshot changed the artifact working directory: {artifact_run}")

    # Neither member can finish until the other has entered its backend.
    # Identical source with one worker is a negative control: serialization
    # must reach the fixture's own error instead of falsely passing the barrier.
    # Cross-process intervals use wall time: macOS Python before 3.10 has
    # process-specific monotonic origins. Deadlines remain monotonic locally.
    for workers in (2, 1):
        parallel_root = fixture / f"graph-workers-{workers}"
        parallel_root.mkdir()

        def member(mine: str, other: str) -> str:
            return (
                "python^(\nfrom pathlib import Path\nimport time\n"
                f"Path('{mine}.ready').write_text(str(time.time_ns()))\n"
                "deadline = time.monotonic() + 8\n"
                f"while not Path('{other}.ready').exists():\n"
                "    if time.monotonic() > deadline:\n"
                "        raise RuntimeError('unified-workers-did-not-overlap')\n"
                "    time.sleep(0.01)\n"
                f"Path('{mine}.finished').write_text(str(time.time_ns()))\n"
                f"__oval_result__ = '{mine}'\n)_python"
            )

        overlapping = execute({
            "source": f"autonomous(batch({member('left', 'right')}, {member('right', 'left')}))",
            "cwd": os.fspath(parallel_root),
            "mode": "admitted",
            "workers": workers,
            "timeout_secs": 30,
        }, error=workers == 1)
        overlap_native = _native_execute_result(overlapping)
        if workers == 1:
            if (
                overlap_native.get("stage") != "eval"
                or "unified-workers-did-not-overlap" not in overlap_native.get("error", "")
            ):
                raise SmokeError(f"one-worker control did not reject the overlap barrier: {overlapping}")
        else:
            paths = [parallel_root / f"{side}.{event}" for side in ("left", "right") for event in ("ready", "finished")]
            if not all(path.is_file() for path in paths):
                raise SmokeError(f"two-worker graph did not complete both barrier members: {overlapping}")
            left_start, left_end, right_start, right_end = [int(path.read_text()) for path in paths]
            if not max(left_start, right_start) < min(left_end, right_end):
                raise SmokeError(
                    "two-worker graph returned success without actual backend overlap: "
                    f"left=({left_start}, {left_end}) right=({right_start}, {right_end})"
                )

    # Source-first execution must adapt past the host's single-argument limit.
    # The source is still ordinary O; only its transport needs a private file.
    large_source = " " * (300 * 1024) + source
    large_input = execute({"source": large_source})
    if (
        _native_execute_result(large_input).get("value")
        != {"t": "number", "v": {"kind": "int", "v": "42"}}
        or large_input.get("input", {}).get("temporary_source") is not True
    ):
        raise SmokeError(f"large inline source did not adapt to a file snapshot: {large_input}")

    large_output = execute({"source": "python^( __oval_result__ = 'L' * (1024 * 1024) )_python"})
    retrieval = large_output.get("result_retrieval", {})
    if (
        large_output.get("state") != "completed"
        or large_output.get("exit_code") != 0
        or large_output.get("result") is not None
        or not large_output.get("result_projection_error")
        or retrieval.get("tool") != "o_job_read"
        or retrieval.get("job_id") != large_output.get("job_id")
        or retrieval.get("stream") != "stdout"
        or retrieval.get("full_output_retained") is not True
    ):
        raise SmokeError(f"large native result omitted deferred retrieval evidence: {large_output}")
    offset = 0
    chunks: list[str] = []
    while True:
        page = call("o_job_read", {
            "job_id": retrieval["job_id"], "stream": "stdout", "offset": offset, "limit": 65536,
        })
        text = page.get("text")
        advance = page.get("next_offset")
        if (
            not isinstance(text, str)
            or not isinstance(advance, int)
            or page.get("offset") != offset
            or advance != offset + len(text.encode("utf-8"))
        ):
            raise SmokeError(f"large output pagination lost byte evidence: {page}")
        chunks.append(text)
        if page.get("eof") is True:
            break
        if advance <= offset:
            raise SmokeError("large output retrieval stalled before EOF")
        offset = advance
    full_result = json.loads("".join(chunks))
    if (
        full_result.get("ok") is not True
        or full_result.get("value", {}).get("v", {}).get("utf8") != "L" * (1024 * 1024)
    ):
        raise SmokeError("deferred large result did not retain the complete native value")

    # Validation, planning, and denied placement must never dispatch effects.
    marker = fixture / "must-not-execute"
    effect_source = (
        "python^(\nfrom pathlib import Path\n"
        f"Path({os.fspath(marker)!r}).write_text('executed')\n"
        "__oval_result__ = 42\n)_python\n"
    )
    checked = execute({"source": effect_source, "action": "check"})
    if _native_execute_result(checked).get("stage") != "parse" or marker.exists():
        raise SmokeError(f"source check did not remain parse-only: {checked}")
    planned = execute({"source": effect_source, "action": "plan", "workers": 2})
    plan = planned.get("result", {})
    if (
        planned.get("state") != "completed"
        or planned.get("exit_code") != 0
        or plan.get("schema") != "oexec.schedule-explanation/v2"
        or plan.get("realizability", {}).get("dispatch") != "not-run"
        or json.loads(planned.get("stdout", {}).get("text", "{}")) != plan
        or marker.exists()
    ):
        raise SmokeError(f"source plan lost its nonexecuting native schedule: {planned}")
    for target, expected in (("ir", "; OIrProgram"), ("dot", "digraph")):
        compiled = execute({"source": effect_source, "action": "compile", "target": target})
        if (
            compiled.get("state") != "completed"
            or compiled.get("exit_code") != 0
            or not isinstance(compiled.get("result"), str)
            or expected not in compiled["result"]
            or compiled["result"] != compiled.get("stdout", {}).get("text")
            or marker.exists()
        ):
            raise SmokeError(f"source {target} compilation failed or executed effects: {compiled}")
    malformed = execute({"source": "python^( unterminated", "action": "check"}, error=True)
    parse_error = _native_execute_result(malformed)
    if parse_error.get("stage") != "parse" or not parse_error.get("error"):
        raise SmokeError(f"malformed source lost its structured parse error: {malformed}")
    for invalid in ({}, {"source": effect_source, "path": os.fspath(program)}):
        rejected = execute(invalid, error=True)
        if rejected.get("job_id") is not None or marker.exists():
            raise SmokeError(f"source/path XOR rejection started a job: {rejected}")
    denied_mesh = execute({"source": effect_source, "placement": "mesh-required"}, error=True)
    if denied_mesh.get("job_id") is not None or marker.exists():
        raise SmokeError(f"ordinary source required mesh placement dispatched locally: {denied_mesh}")
    for options in (
        {"action": "compile", "target": "script"},
        {"action": "compile", "target": "binary"},
        {"action": "check", "mode": "admitted"},
        {"route": "-alternate"},
    ):
        rejected = execute({"source": effect_source, **options}, error=True)
        if rejected.get("job_id") is not None or marker.exists():
            raise SmokeError(f"unsupported operation options dispatched an effect: {rejected}")

    # Each call receives its own literal environment and working directory.
    literal = f"$(touch {marker}) `touch {marker}`"
    context_source = (
        "python^(\nimport json, os, sys\n"
        "print('unified-stderr-preserved', file=sys.stderr)\n"
        "__oval_result__ = json.dumps({'cwd': os.getcwd(), "
        "'value': os.environ.get('OSTADIX_UNIFIED_SMOKE')}, sort_keys=True)\n)_python\n"
    )
    contextual = execute({
        "source": context_source,
        "cwd": os.fspath(fixture),
        "env": {"OSTADIX_UNIFIED_SMOKE": literal},
        "placement": "local",
        "workers": 2,
    })
    context_json = _native_execute_result(contextual).get("value", {}).get("v", {}).get("utf8", "{}")
    if json.loads(context_json) != {"cwd": os.fspath(fixture), "value": literal} or marker.exists():
        raise SmokeError(f"o_execute changed literal environment/cwd: {contextual}")
    if "unified-stderr-preserved" not in contextual.get("stderr", {}).get("text", ""):
        raise SmokeError(f"o_execute lost raw stderr: {contextual}")
    isolated = execute({"source": context_source, "cwd": os.fspath(fixture)})
    isolated_json = _native_execute_result(isolated).get("value", {}).get("v", {}).get("utf8", "{}")
    if json.loads(isolated_json) != {"cwd": os.fspath(fixture), "value": None} or marker.exists():
        raise SmokeError(f"o_execute per-call environment leaked: {isolated}")

    # Admitted execution must produce the same native result while retaining
    # the analysis evidence that binds dispatch to this exact source.
    for supplied in ({"source": source}, {"path": os.fspath(program)}):
        admitted = execute({**supplied, "mode": "admitted"})
        if _native_execute_result(admitted).get("ok") is not True:
            raise SmokeError(f"admitted execution failed: {admitted}")
        analysis = admitted.get("analysis", {})
        analyzed_intent = analysis.get("intent", {})
        analysis_job = analysis.get("job", {})
        if (
            analyzed_intent.get("schema") != "oexec.execution-intent/v1"
            or analyzed_intent.get("source_sha256") != hashlib.sha256(source.encode()).hexdigest()
            or analysis_job.get("state") != "completed"
            or analysis_job.get("exit_code") != 0
            or not isinstance(analysis_job.get("job_id"), str)
            or analysis_job["job_id"] == admitted.get("job_id")
        ):
            raise SmokeError(f"admitted execution omitted its analysis evidence: {admitted}")

    # A source snapshot used for admitted execution must remain available to
    # the managed job. A second call releases it without serializing work.
    barrier = fixture / "release-unified-background"
    waiting_source = (
        "python^(\nfrom pathlib import Path\nimport time\n"
        f"barrier = Path({os.fspath(barrier)!r})\n"
        "while not barrier.exists():\n    time.sleep(0.02)\n"
        "__oval_result__ = 42\n)_python\n"
    )
    background = execute({
        "source": waiting_source,
        "mode": "admitted",
        "background": True,
        "timeout_secs": 30,
    })
    job_id = background.get("job_id")
    if not isinstance(job_id, str) or background.get("state") != "running":
        raise SmokeError(f"background o_execute omitted its running job: {background}")
    snapshot = background.get("input", {}).get("path")
    if not isinstance(snapshot, str) or not Path(snapshot).is_file():
        raise SmokeError(f"background job lost its retained source snapshot: {background}")
    if Path(snapshot).read_text(encoding="utf-8") != waiting_source:
        raise SmokeError("background job source snapshot changed before dispatch")
    release = execute({
        "source": "python^(\nfrom pathlib import Path\n"
        f"Path({os.fspath(barrier)!r}).write_text('released')\n"
        "__oval_result__ = 42\n)_python\n",
    })
    _native_execute_result(release)
    deadline = time.monotonic() + timeout
    while True:
        status = call("o_job_status", {"job_id": job_id})
        if status.get("state") != "running":
            if status.get("state") != "completed" or status.get("exit_code") != 0:
                raise SmokeError(f"background source snapshot did not execute: {status}")
            break
        if time.monotonic() >= deadline:
            raise SmokeError(f"background source execution did not finish: {status}")
        time.sleep(0.05)
    page = call("o_job_read", {"job_id": job_id, "stream": "stdout"})
    if json.loads(page.get("text", "{}")).get("ok") is not True:
        raise SmokeError(f"background source execution lost native result: {page}")
    deadline = time.monotonic() + timeout
    while Path(snapshot).exists():
        if time.monotonic() >= deadline:
            raise SmokeError(f"completed job retained its temporary source: {snapshot}")
        time.sleep(0.02)

    project = fixture / "route project"
    project.mkdir()
    project_marker = fixture / "project-route-executed"
    alternate_marker = fixture / "alternate-route-executed"
    (project / "main.py").write_text(
        "from pathlib import Path\n"
        f"marker = Path({os.fspath(project_marker)!r})\n"
        "count = int(marker.read_text()) if marker.exists() else 0\n"
        "marker.write_text(str(count + 1))\n"
        "print(42)\n",
        encoding="utf-8",
    )
    (project / "alternate.py").write_text(
        "from pathlib import Path\n"
        f"marker = Path({os.fspath(alternate_marker)!r})\n"
        "count = int(marker.read_text()) if marker.exists() else 0\n"
        "marker.write_text(str(count + 1))\n"
        "print(84)\n",
        encoding="utf-8",
    )
    (project / "olang.project.toml").write_text(
        '[project]\nname = "unified-smoke"\ndefault_route = "main"\n'
        '[[routes]]\nid = "main"\ncommand = ["python3", "main.py"]\n'
        'result_codec = "json"\n'
        '[[routes]]\nid = "-alternate"\ncommand = ["python3", "alternate.py"]\n'
        'result_codec = "json"\n',
        encoding="utf-8",
    )
    bundle = fixture / "lifted project.O"
    linked = call("o_cli", {
        "command": "o-link",
        "args": [os.fspath(project), "--project", "-o", os.fspath(bundle)],
    })
    if linked.get("exit_code") != 0 or not bundle.is_file() or project_marker.exists():
        raise SmokeError(f"native project lifting failed or dispatched a route: {linked}")
    for target, expected in (("ir", "ProjectExecutionPlan"), ("dot", "digraph")):
        compiled = execute({"source": bundle.read_text(encoding="utf-8"), "action": "compile", "target": target})
        if (
            compiled.get("state") != "completed"
            or compiled.get("exit_code") != 0
            or expected not in compiled.get("result", "")
            or compiled.get("placement", {}).get("route") != "project-compiler"
            or project_marker.exists()
        ):
            raise SmokeError(f"project {target} compilation lost its nonexecuting route: {compiled}")
    # Explicit selection must reach the native planner/compiler rather than
    # silently keeping the project's declared default route. A leading hyphen
    # is valid in a route ID and must remain a value in the native argument list.
    selected_plan = execute({"path": os.fspath(project), "action": "plan", "route": "-alternate"})
    plan_result = selected_plan.get("result", {})
    if (
        selected_plan.get("state") != "completed"
        or selected_plan.get("exit_code") != 0
        or plan_result.get("schema") != "ostadix.intent-plan-summary/v1"
        or "run-route:-alternate" not in plan_result.get("static_plan", "")
        or "run-route:main" in plan_result.get("static_plan", "")
        or project_marker.exists()
        or alternate_marker.exists()
    ):
        raise SmokeError(f"explicit project planning ignored the selected route: {selected_plan}")
    # Both front doors must preserve the same literal route identity even
    # though their check/plan and result envelopes deliberately differ.
    structured_plan = call("o_run", {
        "path": os.fspath(project), "mode": "check", "route": "-alternate",
    })
    if (
        structured_plan.get("execution_mode") != "nonexecuting_unified_static_plan"
        or "run-route:-alternate" not in structured_plan.get("static_plan", "")
        or "run-route:main" in structured_plan.get("static_plan", "")
        or project_marker.exists()
        or alternate_marker.exists()
    ):
        raise SmokeError(f"structured project planning ignored the literal route: {structured_plan}")
    for target in ("ir", "dot"):
        selected_compile = execute({
            "source": bundle.read_text(encoding="utf-8"),
            "action": "compile",
            "target": target,
            "route": "-alternate",
        })
        if (
            selected_compile.get("state") != "completed"
            or selected_compile.get("exit_code") != 0
            or "run-route:-alternate" not in selected_compile.get("result", "")
            or "run-route:main" in selected_compile.get("result", "")
            or project_marker.exists()
            or alternate_marker.exists()
        ):
            raise SmokeError(f"project {target} compilation ignored explicit route: {selected_compile}")
    invalid_artifact = fixture / "must-not-compile"
    rejected_route = execute({
        "path": os.fspath(project), "action": "compile", "target": "binary",
        "output": os.fspath(invalid_artifact), "route": "-alternate",
    }, error=True)
    if (
        rejected_route.get("job_id") is not None
        or invalid_artifact.exists()
        or project_marker.exists()
        or alternate_marker.exists()
    ):
        raise SmokeError(f"binary compilation accepted a route selected at compile time: {rejected_route}")
    # Project directories, bundle paths, and inline bundles all retain the
    # route executor; interpreting their payload as inert text is not success.
    for expected_count, supplied in enumerate((
        {"path": os.fspath(project)},
        {"path": os.fspath(bundle)},
        {"source": bundle.read_text(encoding="utf-8")},
        {"path": os.fspath(project), "mode": "admitted"},
    ), start=1):
        result = execute({**supplied, "placement": "local"})
        native = result.get("result", {})
        if (
            result.get("state") != "completed"
            or result.get("exit_code") != 0
            or native.get("schema") != "ostadix.run-summary/v1"
            or native.get("disposition") != "succeeded"
            or json.loads(result.get("stdout", {}).get("text", "{}")) != native
            or not project_marker.is_file()
            or project_marker.read_text(encoding="utf-8") != str(expected_count)
        ):
            raise SmokeError(f"unified project input did not run its native route: {result}")
    default_dispatch_count = project_marker.read_text(encoding="utf-8")
    structured_route = call("o_run", {
        "source": bundle.read_text(encoding="utf-8"), "route": "-alternate",
    })
    structured_routes = structured_route.get("result", {}).get("route_results", [])
    if (
        structured_route.get("disposition") != "succeeded"
        or len(structured_routes) != 1
        or structured_routes[0].get("route_id") != "-alternate"
        or alternate_marker.read_text(encoding="utf-8") != "1"
        or project_marker.read_text(encoding="utf-8") != default_dispatch_count
    ):
        raise SmokeError(f"structured project execution ignored the literal route: {structured_route}")
    for selected_count, supplied in enumerate((
        {"path": os.fspath(project)},
        {"source": bundle.read_text(encoding="utf-8")},
    ), start=2):
        selected_run = execute({**supplied, "placement": "local", "route": "-alternate"})
        selected_records = selected_run.get("record", {}).get("record", {}).get("route_results", [])
        if (
            selected_run.get("state") != "completed"
            or selected_run.get("exit_code") != 0
            or selected_run.get("result", {}).get("disposition") != "succeeded"
            or len(selected_records) != 1
            or selected_records[0].get("route_id") != "-alternate"
            or selected_records[0].get("value") != 84
            or not alternate_marker.is_file()
            or alternate_marker.read_text(encoding="utf-8") != str(selected_count)
            or project_marker.read_text(encoding="utf-8") != default_dispatch_count
        ):
            raise SmokeError(f"explicit nondefault route did not control native dispatch: {selected_run}")

    # The marked-operation planner chooses its own route. Its exact descriptor
    # digests bind these files, so copy the native fixture without rewriting it.
    operation_project = fixture / "marked operation"
    shutil.copytree(root / "examples" / "normalize", operation_project)
    operation_before = _snapshot_tree(operation_project)
    overridden_operation = execute({
        "path": os.fspath(operation_project), "route": "normalize_scalar", "placement": "local",
    }, error=True)
    overridden_summary = overridden_operation.get("result", {})
    if (
        overridden_operation.get("state") != "failed"
        or overridden_operation.get("exit_code") in (None, 0)
        or not isinstance(overridden_operation.get("job_id"), str)
        or overridden_summary.get("disposition") != "preflight_failed"
        or "realization planning owns the exact route" not in overridden_summary.get("failure", {}).get("message", "")
        or _snapshot_tree(operation_project) != operation_before
    ):
        raise SmokeError(f"marked operation did not retain native route-override rejection: {overridden_operation}")
    operation = execute({"path": os.fspath(operation_project)})
    operation_summary = operation.get("result", {})
    operation_record = operation.get("record", {}).get("record", {})
    route_results = operation_record.get("route_results", [])
    if (
        operation.get("state") != "completed"
        or operation.get("exit_code") != 0
        or operation_summary.get("disposition") != "succeeded"
        or operation.get("placement", {}).get("route") != "project-operation"
        or len(route_results) != 1
        or route_results[0].get("route_id") != "normalize_chunked"
        or route_results[0].get("value") != {"values": [0.2, 0.4, 0.6, 0.8, 1.0]}
        or operation_record.get("run_id") != operation_summary.get("run_id")
        or _snapshot_tree(operation_project) != operation_before
    ):
        raise SmokeError(f"default auto placement bypassed the marked-operation planner: {operation}")


def run_smoke(
    root: Path,
    binary: Path,
    timeout: float,
    *,
    o_info: Path | None = None,
    runtime_bin_dir: Path | None = None,
    server_cwd: Path | None = None,
    require_wasm: bool = False,
    require_wasm_materialization: bool = False,
    wasm_release_manifest: Path | None = None,
    wasm_release_artifact: Path | None = None,
    wasm_source_tree: str | None = None,
    wasm_base_commit: str | None = None,
    wasm_source_archive_sha256: str | None = None,
    wasm_timeout: float = 900.0,
) -> dict[str, Any] | None:
    catalog_schema = _current_catalog_schema(root)
    config = json.loads((root / ".mcp.json").read_text(encoding="utf-8"))
    registered = config.get("mcpServers", {}).get("ostadix", {})
    if registered.get("command") != "ostadix-mcp":
        raise SmokeError(".mcp.json does not register the released ostadix-mcp command")
    if registered.get("args") != []:
        raise SmokeError(".mcp.json must register ostadix-mcp with an empty argv")
    if "env" in registered:
        raise SmokeError(
            ".mcp.json must not rely on client-specific shell expansion in environment values"
        )

    environment = os.environ.copy()
    environment.pop("O_LANG_ROOT", None)
    environment.pop("O_BACKENDS_DIR", None)
    environment.pop("OLANG", None)
    environment.pop("OSTADIX_RUNTIME_PATH", None)
    environment.pop("OSTADIX_O_INFO_BIN", None)
    environment.pop("A18_WORK", None)
    environment["OSTADIX_RUNTIME_PATH_MODE"] = "discover-local"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    # Model the restricted environment used by GUI-launched MCP clients. The
    # server must restore local runtime locations without shell startup files.
    restricted_path = ["/usr/bin", "/bin", "/usr/sbin", "/sbin"]
    if runtime_bin_dir is not None:
        if (
            not runtime_bin_dir.is_absolute()
            or runtime_bin_dir.is_symlink()
            or not runtime_bin_dir.is_dir()
        ):
            raise SmokeError(
                "installed runtime bin directory is not an absolute "
                f"non-symlink directory: {runtime_bin_dir}"
            )
        restricted_path.insert(0, os.fspath(runtime_bin_dir))
    environment["PATH"] = os.pathsep.join(restricted_path)
    environment["RUST_LOG"] = "warn"
    launch_cwd = server_cwd if server_cwd is not None else root
    if not launch_cwd.is_absolute() or launch_cwd.is_symlink() or not launch_cwd.is_dir():
        raise SmokeError(
            "MCP launch directory is not an absolute non-symlink directory: "
            f"{launch_cwd}"
        )
    if not (1.0 <= wasm_timeout <= 1800.0):
        raise SmokeError("WASM timeout must be from 1 through 1800 seconds")
    if require_wasm and require_wasm_materialization:
        raise SmokeError("fresh WASM compilation and materialization are mutually exclusive")
    if require_wasm_materialization:
        if server_cwd is None:
            raise SmokeError("WASM materialization requires an explicit MCP server cwd")
        required_release_values = {
            "manifest": wasm_release_manifest,
            "artifact": wasm_release_artifact,
            "source tree": wasm_source_tree,
            "base commit": wasm_base_commit,
            "source archive SHA-256": wasm_source_archive_sha256,
        }
        missing = [label for label, value in required_release_values.items() if value is None]
        if missing:
            raise SmokeError(
                "WASM materialization omitted release bindings: " + ", ".join(missing)
            )
        for label, path in (
            ("WASM release manifest", wasm_release_manifest),
            ("WASM release artifact", wasm_release_artifact),
        ):
            assert path is not None
            if not path.is_absolute() or path.is_symlink() or not path.is_file():
                raise SmokeError(f"{label} is not an absolute regular non-symlink file: {path}")

    stderr_capture = tempfile.TemporaryFile()
    home_fixture = tempfile.TemporaryDirectory(prefix=".mcp-home-smoke-")
    environment["HOME"] = home_fixture.name
    intent_fixture = tempfile.TemporaryDirectory(prefix=".mcp-intent-smoke-")
    information_fixture = tempfile.TemporaryDirectory(prefix=".mcp-information-smoke-")
    wasm_fixture = (
        tempfile.TemporaryDirectory(prefix=".mcp-wasm-smoke-")
        if require_wasm
        else None
    )
    wasm_output = (
        Path(wasm_fixture.name) / "ostadix-mcp-hello.wasm"
        if wasm_fixture is not None
        else None
    )
    materialize_fixture = (
        tempfile.TemporaryDirectory(
            prefix=".mcp-wasm-materialize-", dir=os.fspath(launch_cwd)
        )
        if require_wasm_materialization
        else None
    )
    materialize_project = (
        Path(materialize_fixture.name) / "generated"
        if materialize_fixture is not None
        else None
    )
    materialize_output = (
        Path(materialize_fixture.name) / "hello.wasm"
        if materialize_fixture is not None
        else None
    )
    wasm_materialization_evidence: dict[str, Any] | None = None
    information_state = Path(information_fixture.name) / "state"
    information_binary = o_info if o_info is not None else root / "target/release/o-info"
    if (
        not information_binary.is_absolute()
        or information_binary.is_symlink()
        or not information_binary.is_file()
    ):
        raise SmokeError(
            "fixed local o-info binary is not an absolute non-symlink file: "
            f"{information_binary}"
        )
    if o_info is not None:
        environment["OSTADIX_O_INFO_BIN"] = os.fspath(information_binary)
    initialized_information = subprocess.run(
        [os.fspath(information_binary), "init", "--state", os.fspath(information_state)],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if initialized_information.returncode != 0:
        raise SmokeError(
            "could not initialize MCP information smoke state: "
            + initialized_information.stderr.decode("utf-8", "replace")
        )
    information_before = _snapshot_tree(information_state)
    intent_program = Path(intent_fixture.name) / "intent.O"
    intent_marker = Path(intent_fixture.name) / "executed.marker"

    def write_intent_fixture(label: str) -> None:
        intent_program.write_text(
            "python^(\n"
            "from pathlib import Path\n"
            f"Path({json.dumps(os.fspath(intent_marker))}).write_text({label!r})\n"
            f"__oval_result__ = {label!r}\n"
            ")_python\n",
            encoding="utf-8",
        )

    write_intent_fixture("intent-original")
    lifted_fixture = tempfile.TemporaryDirectory(prefix=".mcp-lifted-project-")
    lifted_project = Path(lifted_fixture.name) / "project"
    lifted_project.mkdir()
    (lifted_project / "payload.txt").write_text("source-closed\n", encoding="utf-8")
    (lifted_project / "olang.project.toml").write_text(
        "[project]\n"
        'name = "mcp-lifted-acceptance"\n'
        "\n"
        "[[routes]]\n"
        'id = "main"\n'
        'label = "MCP lifted route"\n'
        'kind = "shell"\n'
        'command = ["sh", "-c", "printf lifted-mcp-ok"]\n'
        "pure = true\n"
        'guards = { requires_command = "sh" }\n\n'
        "[[routes]]\n"
        'id = "other"\n'
        'label = "other route"\n'
        'kind = "shell"\n'
        'command = ["sh", "-c", "printf wrong-route"]\n'
        "pure = true\n"
        'guards = { requires_command = "sh" }\n',
        encoding="utf-8",
    )
    lifted_program = Path(lifted_fixture.name) / "project.O"
    link_binary = (runtime_bin_dir / "o-link" if runtime_bin_dir is not None
                   else root / "target" / "release" / "o-link")
    linked = subprocess.run(
        [
            os.fspath(link_binary),
            os.fspath(lifted_project),
            "--project",
            "-o",
            os.fspath(lifted_program),
        ],
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if linked.returncode != 0 or not lifted_program.is_file():
        raise SmokeError(
            "could not build lifted-project MCP smoke fixture: "
            + linked.stderr.decode("utf-8", "replace")
        )
    fake_octl = Path(lifted_fixture.name) / "octl"
    fake_octl.write_text(
        "#!/bin/sh\n"
        "set -eu\n"
        "test \"$1\" = node\n"
        "test \"$2\" = run\n"
        "test -f \"$3\"\n"
        "grep -q '__oval_result__ = 9' \"$3\"\n"
        "test \"$4\" = --node\n"
        "test \"$5\" = smoke-node\n"
        "test \"$6\" = --deadline-seconds\n"
        "test \"$7\" = 45\n"
        "printf '%s\\n' '{\"schema\":\"mock.hosted-receipt/v1\",\"outcome\":\"succeeded\"}'\n",
        encoding="utf-8",
    )
    fake_octl.chmod(0o700)
    environment["OSTADIX_OCTL_BIN"] = os.fspath(fake_octl)
    process = subprocess.Popen(
        [os.fspath(binary)],
        cwd=launch_cwd,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=stderr_capture,
        bufsize=0,
    )
    if process.stdout is None:
        process.kill()
        process.wait()
        stderr_capture.close()
        raise SmokeError("MCP stdout is unavailable")
    responses = ResponseReader(process.stdout)
    try:
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "ostadix-release-smoke", "version": "1"},
                },
            },
        )
        initialized = responses.response(1, timeout)
        if initialized.get("protocolVersion") != PROTOCOL_VERSION:
            raise SmokeError("MCP initialize negotiated an unexpected protocol version")
        if not {"tools", "resources"}.issubset(initialized.get("capabilities", {})):
            raise SmokeError("MCP initialize did not advertise tools and resources")

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
                "params": {},
            },
        )
        _send(
            process,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        )
        listed = responses.response(2, timeout)
        tools = listed.get("tools")
        if not isinstance(tools, list):
            raise SmokeError("tools/list did not return a tool list")
        names = {
            tool["name"]
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("name"), str)
        }
        if names != EXPECTED_TOOLS:
            raise SmokeError(
                f"unexpected MCP tool set: expected {sorted(EXPECTED_TOOLS)}, got {sorted(names)}"
            )
        for tool in tools:
            if not isinstance(tool, dict):
                raise SmokeError(f"tools/list returned a non-object tool: {tool!r}")
            schema = tool.get("inputSchema")
            if (
                not isinstance(schema, dict)
                or schema.get("type") != "object"
                or not isinstance(schema.get("properties"), dict)
            ):
                raise SmokeError(
                    f"{tool.get('name', '<unnamed>')} has a non-object input schema: "
                    f"{schema!r}"
                )
            if "$schema" in schema or "title" in schema:
                raise SmokeError(f"{tool['name']} still requires the compatibility schema adapter")
            for property_schema in schema["properties"].values():
                if isinstance(property_schema, dict) and (
                    "nullable" in property_schema or property_schema.get("format") == "uint64"
                ):
                    raise SmokeError(f"{tool['name']} retained an unnormalized property schema")
        olangc_tools = [tool for tool in tools if tool.get("name") == "o_olangc"]
        if len(olangc_tools) != 1 or "materialize_only" not in olangc_tools[0][
            "inputSchema"
        ]["properties"]:
            raise SmokeError("o_olangc schema omitted materialize_only")
        execute_tools = [tool for tool in tools if tool.get("name") == "o_execute"]
        execute_fields = {
            "source", "path", "action", "placement", "mode", "cwd", "env", "stdin",
            "timeout_secs", "background", "workers", "target", "output", "route",
        }
        if len(execute_tools) != 1 or not execute_fields.issubset(
            execute_tools[0]["inputSchema"]["properties"]
        ):
            raise SmokeError("o_execute schema omitted its unified input controls")
        run_tools = [tool for tool in tools if tool.get("name") == "o_run"]
        if len(run_tools) != 1:
            raise SmokeError("tools/list did not expose exactly one o_run tool")
        run_properties = run_tools[0]["inputSchema"]["properties"]
        for required_property in (
            "source",
            "path",
            "cwd",
            "mode",
            "placement",
            "node_id",
            "route",
        ):
            if required_property not in run_properties:
                raise SmokeError(f"o_run schema omitted {required_property}")

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "o_env", "arguments": {}},
            },
        )
        environment_result = responses.response(3, timeout)
        if environment_result.get("isError") is True:
            raise SmokeError("o_env returned an MCP tool error")
        environment_text = _content_text(environment_result)
        required_environment = {
            f"O_LANG_ROOT={root}",
            f"O_BACKENDS_DIR={root / 'backends'}",
        }
        if not all(value in environment_text for value in required_environment):
            raise SmokeError(f"o_env returned unexpected paths:\n{environment_text}")
        if "runtime-summary backend-count=30" not in environment_text:
            raise SmokeError(f"o_env omitted the all-runtime summary:\n{environment_text}")

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {"name": "o_runtimes", "arguments": {}},
            },
        )
        runtimes_result = responses.response(4, timeout)
        runtimes_text = _content_text(runtimes_result)
        if runtimes_result.get("isError") is True:
            raise SmokeError(f"o_runtimes returned an MCP tool error:\n{runtimes_text}")
        required_runtime_markers = {
            f"runtime-catalog-schema={catalog_schema}",
            "runtime-catalog-legacy-schema-v5=ostadix.backend-catalog/v5",
            "runtime-catalog-legacy-schema-v4=ostadix.backend-catalog/v4",
            "runtime-catalog-projection=compiled-mcp-snapshot",
            "runtime-search-mode=discover-local",
            "runtime-summary backend-count=30",
            "runtime backends=python status=located",
            "runtime backends=java status=",
            "runtime backends=webassembly status=",
            "precision=conservative-all-sources",
            "invocable=not-probed",
            "admitted=operation-scoped-not-evaluated",
            "path-sources=[python3=",
            "backend=python integer-exactness=arbitrary rich-numbers=preserved "
            "state-support=semantic-snapshot codec=ostadix.python-graph/v1 "
            "compatibility=exact-implementation morphism-profile=python-plain-data",
            "backend=javascript integer-exactness=exact-magnitude-bits:53 "
            "rich-numbers=collapsed state-support=stateless "
            "morphism-profile=javascript-binding-stdout",
            "backend=html integer-exactness=arbitrary rich-numbers=collapsed "
            "state-support=stateless morphism-profile=none",
            "morphism profiles are bounded shadow descriptions; they do not authorize "
            "execution or claim generic backend crossings",
        }
        required_runtime_markers.update(
            f"runtime-search-entry index={index} source=inherited:{index} path={path}"
            for index, path in enumerate(restricted_path)
        )
        if not all(marker in runtimes_text for marker in required_runtime_markers):
            raise SmokeError(
                "o_runtimes omitted required backend discovery markers:\n"
                f"{runtimes_text}"
            )

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {"name": "o_smoke", "arguments": {}},
            },
        )
        smoke_result = responses.response(5, timeout)
        smoke_text = _content_text(smoke_result)
        if smoke_result.get("isError") is True or "SMOKE_OK" not in smoke_text:
            raise SmokeError(f"o_smoke failed:\n{smoke_text}")
        if "[number] 2" not in smoke_text:
            raise SmokeError(f"o_smoke omitted the expected result 2:\n{smoke_text}")

        integer_two = {"t": "number", "v": {"kind": "int", "v": "2"}}
        run_calls = [
            (6, {"path": "examples/hello.O", "timeout_secs": 45}, integer_two),
            (7, {"path": "hello.O", "cwd": "examples", "timeout_secs": 45}, integer_two),
            (
                88,
                {
                    "source": "python^(\n__oval_result__ = 1 + 1\n)_python\n",
                    "timeout_secs": 45,
                },
                integer_two,
            ),
            (
                85,
                {
                    "source": (
                        "python^(\n"
                        "__oval_result__ = open('Cargo.toml', encoding='utf-8').read()"
                        ".startswith('[package]')\n"
                        ")_python\n"
                    ),
                    "cwd": os.fspath(root),
                    "timeout_secs": 45,
                },
                {"t": "bool", "v": True},
            ),
            (
                84,
                {
                    "source": (
                        "python^(\n"
                        "page = html^(<p>nested python^(\n"
                        "__oval_result__ = 6 * 7\n"
                        ")_python</p>)_html\n"
                        "__oval_result__ = page\n"
                        ")_python\n"
                    ),
                    "timeout_secs": 45,
                },
                {"t": "html", "v": "<p>nested 42</p>"},
            ),
        ]
        for request_id, arguments, expected_value in run_calls:
            _send(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {"name": "o_run", "arguments": arguments},
                },
            )
            result = responses.response(request_id, timeout)
            structured = result.get("structuredContent")
            decoded = (
                structured.get("result", {}).get("decoded_value")
                if isinstance(structured, dict)
                else None
            )
            if (
                result.get("isError") is True
                or not isinstance(structured, dict)
                or structured.get("execution_mode") != "local_unified_front_door"
                or decoded != expected_value
            ):
                raise SmokeError(
                    "o_run source/path structured-result smoke failed:\n"
                    f"{json.dumps(result, sort_keys=True)}"
                )

        check_marker = Path(lifted_fixture.name) / "check-must-not-execute.marker"
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 90,
                "method": "tools/call",
                "params": {
                    "name": "o_run",
                    "arguments": {
                        "source": (
                            "python^(\n"
                            "from pathlib import Path\n"
                            f"Path({json.dumps(os.fspath(check_marker))}).write_text('effect')\n"
                            "__oval_result__ = 42\n"
                            ")_python\n"
                        ),
                        "mode": "check",
                        "timeout_secs": 45,
                    },
                },
            },
        )
        checked = responses.response(90, timeout)
        checked_structured = checked.get("structuredContent")
        if (
            checked.get("isError") is True
            or not isinstance(checked_structured, dict)
            or checked_structured.get("schema")
            != "ostadix.intent-plan-summary/v1"
            or checked_structured.get("execution_mode")
            != "nonexecuting_unified_static_plan"
            or check_marker.exists()
        ):
            raise SmokeError(
                "o_run mode=check did not return a static plan without effects:\n"
                f"{json.dumps(checked, sort_keys=True)}"
            )

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 86,
                "method": "tools/call",
                "params": {
                    "name": "o_run",
                    "arguments": {"source": "python^(\nunclosed", "timeout_secs": 45},
                },
            },
        )
        failed_run = responses.response(86, timeout)
        failed_structured = failed_run.get("structuredContent")
        if (
            failed_run.get("isError") is not True
            or not isinstance(failed_structured, dict)
            or failed_structured.get("disposition") != "preflight_failed"
            or failed_structured.get("failure", {}).get("stage") != "preflight"
        ):
            raise SmokeError(
                "o_run did not return a structured preflight failure:\n"
                f"{json.dumps(failed_run, sort_keys=True)}"
            )

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 92,
                "method": "tools/call",
                "params": {
                    "name": "o_run",
                    "arguments": {
                        "source": "python^(\nraise RuntimeError('mcp-runtime-failure')\n)_python\n",
                        "timeout_secs": 45,
                    },
                },
            },
        )
        runtime_failure = responses.response(92, timeout)
        runtime_failure_structured = runtime_failure.get("structuredContent")
        if (
            runtime_failure.get("isError") is not True
            or not isinstance(runtime_failure_structured, dict)
            or runtime_failure_structured.get("disposition") != "execution_failed"
            or runtime_failure_structured.get("failure", {}).get("stage") != "execution"
            or "mcp-runtime-failure"
            not in runtime_failure_structured.get("failure", {}).get("message", "")
        ):
            raise SmokeError(
                "o_run did not distinguish a runtime execution failure:\n"
                f"{json.dumps(runtime_failure, sort_keys=True)}"
            )

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 83,
                "method": "tools/call",
                "params": {
                    "name": "o_run",
                    "arguments": {
                        "source": "python^(\n__oval_result__ = 9\n)_python\n",
                        "placement": "node",
                        "node_id": "smoke-node",
                        "timeout_secs": 45,
                    },
                },
            },
        )
        node_run = responses.response(83, timeout)
        node_structured = node_run.get("structuredContent")
        if (
            node_run.get("isError") is True
            or not isinstance(node_structured, dict)
            or node_structured.get("schema") != "mock.hosted-receipt/v1"
            or node_structured.get("outcome") != "succeeded"
            or node_structured.get("execution_mode")
            != "selected_node_complete_document"
        ):
            raise SmokeError(
                "o_run hosted-node adapter did not preserve whole-document placement:\n"
                f"{json.dumps(node_run, sort_keys=True)}"
            )

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 87,
                "method": "tools/call",
                "params": {
                    "name": "o_run",
                    "arguments": {
                        "path": os.fspath(lifted_program),
                        "route": "main",
                        "timeout_secs": 45,
                    },
                },
            },
        )
        lifted_result = responses.response(87, timeout)
        lifted_structured = lifted_result.get("structuredContent")
        lifted_routes = (
            lifted_structured.get("result", {}).get("route_results")
            if isinstance(lifted_structured, dict)
            else None
        )
        if (
            lifted_result.get("isError") is True
            or not isinstance(lifted_routes, list)
            or len(lifted_routes) != 1
            or lifted_routes[0].get("route_id") != "main"
            or lifted_routes[0].get("disposition") != "executed"
            or lifted_routes[0].get("exit_code") != 0
        ):
            raise SmokeError(
                "o_run loaded a lifted bundle without executing its default route:\n"
                f"{json.dumps(lifted_result, sort_keys=True)}"
            )

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 91,
                "method": "tools/call",
                "params": {
                    "name": "o_run",
                    "arguments": {
                        "path": os.fspath(lifted_program),
                        "timeout_secs": 45,
                    },
                },
            },
        )
        ambiguous = responses.response(91, timeout)
        ambiguous_structured = ambiguous.get("structuredContent")
        if (
            ambiguous.get("isError") is not True
            or not isinstance(ambiguous_structured, dict)
            or ambiguous_structured.get("disposition") != "preflight_failed"
            or "no unambiguous default route"
            not in ambiguous_structured.get("failure", {}).get("message", "")
        ):
            raise SmokeError(
                "o_run did not report lifted-project route ambiguity precisely:\n"
                f"{json.dumps(ambiguous, sort_keys=True)}"
            )

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 89,
                "method": "tools/call",
                "params": {
                    "name": "o_run",
                    "arguments": {
                        "path": os.fspath(lifted_program),
                        "route": "main",
                        "placement": "project_mesh",
                        "timeout_secs": 45,
                    },
                },
            },
        )
        mesh_result = responses.response(89, timeout)
        mesh_structured = mesh_result.get("structuredContent")
        if (
            mesh_result.get("isError") is not True
            or not isinstance(mesh_structured, dict)
            or mesh_structured.get("execution_mode")
            != "authenticated_project_mesh_required"
            or mesh_structured.get("mesh_trace", {}).get("candidates") != []
            or "no authenticated peer eligible" not in _content_text(mesh_result)
        ):
            raise SmokeError(
                "o_run project mesh did not require authenticated remote placement:\n"
                f"{json.dumps(mesh_result, sort_keys=True)}"
            )

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 93,
                "method": "tools/call",
                "params": {
                    "name": "o_run",
                    "arguments": {
                        "source": "python^(\n__oval_result__ = 2\n)_python\n",
                        "placement": "project_mesh",
                        "timeout_secs": 45,
                    },
                },
            },
        )
        ordinary_mesh = responses.response(93, timeout)
        ordinary_mesh_structured = ordinary_mesh.get("structuredContent")
        if (
            ordinary_mesh.get("isError") is not True
            or not isinstance(ordinary_mesh_structured, dict)
            or ordinary_mesh_structured.get("disposition") != "preflight_failed"
            or ordinary_mesh_structured.get("mesh_trace_status", {}).get("complete")
            is not False
            or "ordinary OIR execution uses only the local HGraph worker pool"
            not in ordinary_mesh_structured.get("failure", {}).get("message", "")
        ):
            raise SmokeError(
                "o_run silently accepted or obscured unsupported ordinary OIR mesh:\n"
                f"{json.dumps(ordinary_mesh, sort_keys=True)}"
            )

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 8,
                "method": "tools/call",
                "params": {
                    "name": "o_olangc",
                    "arguments": {
                        "path": "examples/hello.O",
                        "target": "ir",
                        "timeout_secs": 45,
                    },
                },
            },
        )
        compiler_result = responses.response(8, timeout)
        compiler_text = _content_text(compiler_result)
        if compiler_result.get("isError") is True or "; OIrProgram" not in compiler_text:
            raise SmokeError(f"o_olangc relative-path smoke failed:\n{compiler_text}")

        # Analyze is nonexecuting; mutation after analysis must be rejected by
        # O's recomputation, and the failed attempt must still consume the
        # handle so it cannot be replayed.
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 9,
                "method": "tools/call",
                "params": {
                    "name": "o_analyze_intent",
                    "arguments": {
                        "path": os.fspath(intent_program),
                        "ttl_secs": 60,
                        "timeout_secs": 45,
                    },
                },
            },
        )
        analyzed = responses.response(9, timeout)
        analyzed_text = _content_text(analyzed)
        if analyzed.get("isError") is True:
            raise SmokeError(f"o_analyze_intent failed:\n{analyzed_text}")
        handle = _record_field(analyzed_text, "intent-handle")
        if "intent-schema=oexec.execution-intent/v1" not in analyzed_text:
            raise SmokeError(
                f"o_analyze_intent omitted the stable schema:\n{analyzed_text}"
            )
        if len(_record_field(analyzed_text, "source-sha256")) != 64:
            raise SmokeError(f"o_analyze_intent emitted a bad source digest:\n{analyzed_text}")
        if intent_marker.exists():
            raise SmokeError("o_analyze_intent executed the inspected Python backend")

        write_intent_fixture("intent-mutated")
        execute_arguments = {
            "handle": handle,
            "path": os.fspath(intent_program),
            "timeout_secs": 45,
        }
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 10,
                "method": "tools/call",
                "params": {
                    "name": "o_execute_intent",
                    "arguments": execute_arguments,
                },
            },
        )
        mutated = responses.response(10, timeout)
        mutated_text = _content_text(mutated)
        if mutated.get("isError") is not True or not (
            "source" in mutated_text.lower() and "mismatch" in mutated_text.lower()
        ):
            raise SmokeError(
                "o_execute_intent did not reject source mutation with a source mismatch:\n"
                f"{mutated_text}"
            )
        if intent_marker.exists():
            raise SmokeError("rejected source mutation dispatched the Python backend")

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 11,
                "method": "tools/call",
                "params": {
                    "name": "o_execute_intent",
                    "arguments": execute_arguments,
                },
            },
        )
        replay = responses.response(11, timeout)
        replay_text = _content_text(replay)
        if replay.get("isError") is not True or "already-consumed" not in replay_text:
            raise SmokeError(f"consumed intent handle was replayable:\n{replay_text}")

        # A fresh handle over the mutated source succeeds exactly once.
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 12,
                "method": "tools/call",
                "params": {
                    "name": "o_analyze_intent",
                    "arguments": {
                        "path": os.fspath(intent_program),
                        "timeout_secs": 45,
                    },
                },
            },
        )
        fresh = responses.response(12, timeout)
        fresh_text = _content_text(fresh)
        if fresh.get("isError") is True:
            raise SmokeError(f"fresh o_analyze_intent failed:\n{fresh_text}")
        fresh_handle = _record_field(fresh_text, "intent-handle")
        fresh_arguments = {
            "handle": fresh_handle,
            "path": os.fspath(intent_program),
            "timeout_secs": 45,
        }
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 13,
                "method": "tools/call",
                "params": {
                    "name": "o_execute_intent",
                    "arguments": fresh_arguments,
                },
            },
        )
        executed = responses.response(13, timeout)
        executed_text = _content_text(executed)
        if (
            executed.get("isError") is True
            or "intent-consumed=true" not in executed_text
            or "intent-mutated" not in executed_text
        ):
            raise SmokeError(f"fresh intent execution failed:\n{executed_text}")
        if intent_marker.read_text(encoding="utf-8") != "intent-mutated":
            raise SmokeError("matching intent did not commit the expected backend effect")
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 14,
                "method": "tools/call",
                "params": {
                    "name": "o_execute_intent",
                    "arguments": fresh_arguments,
                },
            },
        )
        successful_replay = responses.response(14, timeout)
        successful_replay_text = _content_text(successful_replay)
        if (
            successful_replay.get("isError") is not True
            or "already-consumed" not in successful_replay_text
        ):
            raise SmokeError(
                f"successfully consumed intent handle was replayable:\n{successful_replay_text}"
            )

        # Echoed target arguments are part of the handle binding. A mismatch
        # is rejected before O starts and consumes the attempted handle.
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 15,
                "method": "tools/call",
                "params": {
                    "name": "o_analyze_intent",
                    "arguments": {"path": os.fspath(intent_program)},
                },
            },
        )
        mismatch_analysis = responses.response(15, timeout)
        mismatch_text = _content_text(mismatch_analysis)
        mismatch_handle = _record_field(mismatch_text, "intent-handle")
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 16,
                "method": "tools/call",
                "params": {
                    "name": "o_execute_intent",
                    "arguments": {
                        "handle": mismatch_handle,
                        "path": "examples/hello.O",
                    },
                },
            },
        )
        mismatched = responses.response(16, timeout)
        mismatched_text = _content_text(mismatched)
        if mismatched.get("isError") is not True or "program mismatch" not in mismatched_text:
            raise SmokeError(f"intent target mismatch was accepted:\n{mismatched_text}")
        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 17,
                "method": "tools/call",
                "params": {
                    "name": "o_execute_intent",
                    "arguments": {
                        "handle": mismatch_handle,
                        "path": os.fspath(intent_program),
                    },
                },
            },
        )
        mismatch_replay = responses.response(17, timeout)
        mismatch_replay_text = _content_text(mismatch_replay)
        if (
            mismatch_replay.get("isError") is not True
            or "already-consumed" not in mismatch_replay_text
        ):
            raise SmokeError(
                f"mismatched intent attempt did not consume its handle:\n{mismatch_replay_text}"
            )

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 18,
                "method": "tools/call",
                "params": {
                    "name": "o_information_inspect",
                    "arguments": {
                        "state": os.fspath(information_state),
                        "head": "main",
                        "timeout_secs": 10,
                    },
                },
            },
        )
        information_result = responses.response(18, timeout)
        information_text = _content_text(information_result)
        if information_result.get("isError") is True:
            raise SmokeError(
                f"o_information_inspect failed on initialized local state:\n{information_text}"
            )
        required_information = {
            "head=main",
            "facts=0",
            "authority=information presence and signatures grant no execution authority",
            "source=local-o-info-read-only",
        }
        if not all(marker in information_text for marker in required_information):
            raise SmokeError(
                f"o_information_inspect omitted bounded records:\n{information_text}"
            )
        if os.fspath(information_state) in information_text or "state=" in information_text:
            raise SmokeError(
                f"o_information_inspect leaked its request-local state path:\n{information_text}"
            )
        if _snapshot_tree(information_state) != information_before:
            raise SmokeError("o_information_inspect mutated the local information store")

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 19,
                "method": "tools/call",
                "params": {
                    "name": "o_information_inspect",
                    "arguments": {
                        "state": os.fspath(information_state),
                        "head": "../main",
                    },
                },
            },
        )
        invalid_information = responses.response(19, timeout)
        if invalid_information.get("isError") is not True:
            raise SmokeError("o_information_inspect accepted a non-token head name")
        if _snapshot_tree(information_state) != information_before:
            raise SmokeError("rejected information inspection mutated the local store")

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 20,
                "method": "tools/call",
                "params": {"name": "o_doctor", "arguments": {}},
            },
        )
        doctor_result = responses.response(20, timeout)
        doctor_text = _content_text(doctor_result)
        required_doctor = {
            f"O_LANG_ROOT={root} exists=true",
            f"search-work={root}",
            f"search-corpus={root / 'examples'} bundled=true",
        }
        if doctor_result.get("isError") is True or not all(
            marker in doctor_text for marker in required_doctor
        ):
            raise SmokeError(f"o_doctor omitted installed-layout records:\n{doctor_text}")

        _send(
            process,
            {
                "jsonrpc": "2.0",
                "id": 21,
                "method": "tools/call",
                "params": {
                    "name": "o_search_run",
                    "arguments": {"name": "hello", "timeout_secs": 45},
                },
            },
        )
        search_result = responses.response(21, timeout)
        search_text = _content_text(search_result)
        required_search = {
            f"program={root / 'examples/hello.O'}",
            f"corpus={root / 'examples'}",
            "[number] 2",
        }
        if search_result.get("isError") is True or not all(
            marker in search_text for marker in required_search
        ):
            raise SmokeError(f"o_search_run bundled corpus failed:\n{search_text}")

        for request_id, rejected_name in (
            (22, "../hello"),
            (23, os.fspath(root / "examples/hello.O")),
        ):
            _send(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "tools/call",
                    "params": {
                        "name": "o_search_run",
                        "arguments": {"name": rejected_name},
                    },
                },
            )
            rejected_search = responses.response(request_id, timeout)
            rejected_text = _content_text(rejected_search)
            if (
                rejected_search.get("isError") is not True
                or "leaf token" not in rejected_text
            ):
                raise SmokeError(
                    "o_search_run accepted a path outside its leaf-token contract:\n"
                    f"{rejected_text}"
                )

        with tempfile.TemporaryDirectory(prefix=".mcp-agent-smoke-") as agent_fixture:
            _run_agent_surface_smoke(
                process, responses, root, timeout, Path(agent_fixture).resolve()
            )

        with tempfile.TemporaryDirectory(prefix=".mcp unified smoke ") as unified_fixture:
            _run_unified_surface_smoke(
                process, responses, root, timeout, Path(unified_fixture).resolve()
            )

        if require_wasm:
            assert wasm_output is not None
            _send(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": 24,
                    "method": "tools/call",
                    "params": {
                        "name": "o_olangc",
                        "arguments": {
                            "path": "examples/wasm_hello.O",
                            "target": "wasm",
                            "output": os.fspath(wasm_output),
                            "timeout_secs": int(wasm_timeout),
                        },
                    },
                },
            )
            wasm_result = responses.response(24, wasm_timeout + 30.0)
            wasm_text = _content_text(wasm_result)
            if wasm_result.get("isError") is True or "exit=0" not in wasm_text:
                raise SmokeError(f"o_olangc WASM compile failed:\n{wasm_text}")
            if (
                wasm_output.is_symlink()
                or not wasm_output.is_file()
                or wasm_output.read_bytes()[:4] != b"\x00asm"
            ):
                raise SmokeError(
                    "o_olangc WASM compile omitted a regular WebAssembly artifact"
                )
        elif require_wasm_materialization:
            assert materialize_project is not None
            assert materialize_output is not None
            _send(
                process,
                {
                    "jsonrpc": "2.0",
                    "id": 24,
                    "method": "tools/call",
                    "params": {
                        "name": "o_olangc",
                        "arguments": {
                            "path": "examples/wasm_hello.O",
                            "target": "wasm",
                            "output": os.fspath(materialize_output),
                            "materialize_only": os.fspath(materialize_project),
                            "timeout_secs": int(timeout),
                        },
                    },
                },
            )
            materialize_result = responses.response(24, timeout + 30.0)
            materialize_text = _content_text(materialize_result)
            expected_record = (
                "olangc: materialize-only target=wasm rust-target=wasm32-wasip1 "
                f"cargo-invoked=false dir={materialize_project}"
            )
            if (
                materialize_result.get("isError") is True
                or "exit=0" not in materialize_text
                or expected_record not in materialize_text
            ):
                raise SmokeError(
                    f"o_olangc WASM materialization failed:\n{materialize_text}"
                )
            required_project_files = (
                "Cargo.toml",
                "Cargo.lock",
                "rust-toolchain.toml",
                "src/lib.rs",
                "src/main.rs",
                "src/program.O",
            )
            if (
                materialize_project.is_symlink()
                or not materialize_project.is_dir()
                or any(
                    not (materialize_project / relative).is_file()
                    for relative in required_project_files
                )
                or (materialize_project / "target").exists()
                or materialize_output.exists()
                or (materialize_project / "src/program.O").read_bytes()
                != (root / "examples/wasm_hello.O").read_bytes()
            ):
                raise SmokeError(
                    "o_olangc materialization omitted or mutated its exact no-build project"
                )
            assert wasm_release_manifest is not None
            assert wasm_release_artifact is not None
            assert wasm_source_tree is not None
            assert wasm_base_commit is not None
            assert wasm_source_archive_sha256 is not None
            generator = (
                runtime_bin_dir / "olangc"
                if runtime_bin_dir is not None
                else root / "target/release/olangc"
            )
            verifier = root / "scripts/ostadix_wasm_release.py"
            verified = subprocess.run(
                [
                    sys.executable,
                    os.fspath(verifier),
                    "verify",
                    "--manifest",
                    os.fspath(wasm_release_manifest),
                    "--project",
                    os.fspath(materialize_project),
                    "--artifact",
                    os.fspath(wasm_release_artifact),
                    "--input",
                    os.fspath(root / "examples/wasm_hello.O"),
                    "--generator",
                    os.fspath(generator),
                    "--source-tree",
                    wasm_source_tree,
                    "--base-commit",
                    wasm_base_commit,
                    "--source-archive-sha256",
                    wasm_source_archive_sha256,
                ],
                cwd=launch_cwd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
            )
            if verified.returncode != 0:
                raise SmokeError(
                    "MCP-materialized WASM project failed its release binding:\n"
                    + verified.stderr.decode("utf-8", "replace")
                )
            try:
                wasm_materialization_evidence = json.loads(
                    verified.stdout.decode("utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise SmokeError(
                    "WASM release verifier returned malformed JSON"
                ) from error
    finally:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise SmokeError("ostadix-mcp did not exit after stdin closed")
        responses.join(timeout)
        stderr_capture.seek(0)
        stderr = stderr_capture.read().decode("utf-8", "replace")
        stderr_capture.close()
        intent_fixture.cleanup()
        information_fixture.cleanup()
        if wasm_fixture is not None:
            wasm_fixture.cleanup()
        if materialize_fixture is not None:
            materialize_fixture.cleanup()
        home_fixture.cleanup()
        if process.returncode != 0:
            raise SmokeError(
                f"ostadix-mcp exited {process.returncode}; stderr:\n{stderr}"
            )
    return wasm_materialization_evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--o-info", type=Path)
    parser.add_argument("--runtime-bin-dir", type=Path)
    parser.add_argument("--server-cwd", type=Path)
    wasm_mode = parser.add_mutually_exclusive_group()
    wasm_mode.add_argument("--require-wasm", action="store_true")
    wasm_mode.add_argument("--require-wasm-materialization", action="store_true")
    parser.add_argument("--wasm-release-manifest", type=Path)
    parser.add_argument("--wasm-release-artifact", type=Path)
    parser.add_argument("--wasm-source-tree")
    parser.add_argument("--wasm-base-commit")
    parser.add_argument("--wasm-source-archive-sha256")
    parser.add_argument("--wasm-timeout", type=float, default=900.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    arguments = parser.parse_args()

    root = arguments.root.expanduser().resolve()
    binary = (
        arguments.binary.expanduser().resolve()
        if arguments.binary
        else root / "mcp/ostadix_lang_mcp_server/target/release/ostadix-mcp"
    )
    if not binary.is_file():
        print(f"error: MCP binary not found: {binary}", file=sys.stderr)
        return 2
    o_info = arguments.o_info.expanduser() if arguments.o_info else None
    runtime_bin_dir = (
        arguments.runtime_bin_dir.expanduser()
        if arguments.runtime_bin_dir
        else None
    )
    server_cwd = (
        arguments.server_cwd.expanduser().resolve()
        if arguments.server_cwd
        else None
    )
    try:
        wasm_evidence = run_smoke(
            root,
            binary,
            arguments.timeout,
            o_info=o_info,
            runtime_bin_dir=runtime_bin_dir,
            server_cwd=server_cwd,
            require_wasm=arguments.require_wasm,
            require_wasm_materialization=arguments.require_wasm_materialization,
            wasm_release_manifest=arguments.wasm_release_manifest,
            wasm_release_artifact=arguments.wasm_release_artifact,
            wasm_source_tree=arguments.wasm_source_tree,
            wasm_base_commit=arguments.wasm_base_commit,
            wasm_source_archive_sha256=arguments.wasm_source_archive_sha256,
            wasm_timeout=arguments.wasm_timeout,
        )
    except (OSError, SmokeError, json.JSONDecodeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if arguments.require_wasm:
        print("ostadix-mcp o_olangc wasm: PASS")
    if arguments.require_wasm_materialization:
        assert wasm_evidence is not None
        project = wasm_evidence["project"]
        source = wasm_evidence["source"]
        artifact = wasm_evidence["artifact"]
        print(
            "ostadix-mcp o_olangc wasm materialization: PASS "
            f"root_sha256={project['root_sha256']}"
        )
        print(
            "ostadix-mcp o_olangc wasm artifact: PASS "
            f"tree={source['staged_tree']} bytes={artifact['bytes']} "
            f"sha256={artifact['sha256']}"
        )
    print("ostadix-mcp stdio release smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
