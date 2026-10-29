#!/usr/bin/env python3
"""Retain real MCP lifted-project cancellation, deadline, failure and recovery evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from smoke_ostadix_mcp import ResponseReader, _send


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--runtime-bin-dir", type=Path)
    parser.add_argument("--evidence-output", type=Path)
    arguments = parser.parse_args()
    binary = (arguments.binary or root / "mcp/ostadix_lang_mcp_server/target/release/ostadix-mcp").expanduser().resolve()
    runtime_bin = (arguments.runtime_bin_dir or root / "target/release").expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    evidence = arguments.evidence_output or (
        root / "audits/source-first-mcp" / f"lifted-lifecycle-{stamp}.json"
    )
    evidence.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the evidence destination before any effectful dispatch.
    with evidence.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"status": "started", "utc": stamp}) + "\n")
    records = []
    checks = []
    env = dict(os.environ, O_LANG_ROOT=str(root), OSTADIX_RUNTIME_PATH_MODE="discover-local",
               OSTADIX_O_CLI_BIN=str(runtime_bin / "o-cli"))
    with tempfile.TemporaryDirectory(prefix="ostadix-lifted-lifecycle-") as temp:
        work = Path(temp)
        project = work / "project"
        project.mkdir()
        (project / "olang.project.toml").write_text('''[project]
name = "lifted-lifecycle"
default_route = "main"
[[routes]]
id = "main"
command = ["python3", "main.py"]
result_codec = "json"
default = true
''')
        with (evidence.with_suffix(".stderr")).open("xb") as stderr:
            proc = subprocess.Popen([str(binary)], cwd=root, env=env, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=stderr)
            executable_image = Path(f"/proc/{proc.pid}/exe")
            if not executable_image.is_file():
                executable_image = binary
            tested_binary_sha256 = hashlib.sha256(executable_image.read_bytes()).hexdigest()
            reader = ResponseReader(proc.stdout)

            def call(req, timeout=15):
                _send(proc, req)
                result = reader.response(req["id"], timeout)
                records.append(dict(request=req, result=result))
                return result

            status = "incomplete"
            failure = None
            try:
                call(dict(jsonrpc="2.0", id=1, method="initialize", params=dict(
                    protocolVersion="2025-03-26", capabilities={},
                    clientInfo=dict(name="lifted-lifecycle", version="1"))))
                _send(proc, dict(jsonrpc="2.0", method="notifications/initialized"))
                for request_id, mode in enumerate(["cancel", "deadline", "failure"], 2):
                    marker = work / (mode + "-started")
                    late = work / (mode + "-late")
                    workspace_marker = work / (mode + "-workspace")
                    program = f'''from pathlib import Path
import os, time, subprocess, sys, json
Path({str(workspace_marker)!r}).write_text(os.getcwd())
marker = Path({str(marker)!r})
with marker.open("a") as stream:
    stream.write("started\\n")
    stream.flush()
'''
                    if mode == "failure":
                        program += "raise SystemExit(7)\n"
                    else:
                        child_code = f"import time; from pathlib import Path; time.sleep(2); Path({str(late)!r}).write_text('late')"
                        program += f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\ntime.sleep(10)\nprint(json.dumps(dict(done=True)))\n"
                    (project / "main.py").write_text(program)
                    bundle = work / "lifecycle.O"
                    subprocess.run([str(runtime_bin / "o-link"), "--project", str(project),
                                    "-o", str(bundle)], check=True, capture_output=True)
                    arguments = dict(source=bundle.read_text(), placement="local",
                                     timeout_secs=1 if mode == "deadline" else 10)
                    request = dict(jsonrpc="2.0", id=request_id, method="tools/call",
                                   params=dict(name="o_execute", arguments=arguments))
                    started = time.monotonic()
                    _send(proc, request)
                    if mode == "cancel":
                        deadline = time.monotonic() + 5
                        while not marker.exists():
                            assert time.monotonic() < deadline, "route did not dispatch"
                            time.sleep(0.01)
                        # Allow the route to spawn its subprocess before cancellation.
                        time.sleep(0.1)
                        notification = dict(jsonrpc="2.0", method="notifications/cancelled",
                                            params=dict(requestId=request_id, reason="lifecycle probe"))
                        _send(proc, notification)
                        records.append(dict(notification=notification))
                    result = reader.response(request_id, 15)
                    records.append(dict(request=request, result=result,
                                        wall_ms=round((time.monotonic() - started) * 1000)))
                    assert result.get("isError"), (mode, result)
                    structured = result.get("structuredContent", {})
                    expected_state = {"cancel": "cancelled", "deadline": "timed_out", "failure": "failed"}[mode]
                    assert structured.get("state") == expected_state, (mode, result)
                    assert structured.get("cleanup", {}).get("child_reaped") is True, (mode, result)
                    assert structured.get("cleanup", {}).get("logs_drained") is True, (mode, result)
                    time.sleep(2.2 if mode != "failure" else 0)
                    count = len(marker.read_text().splitlines()) if marker.exists() else 0
                    assert count == 1, (mode, "dispatch count", count)
                    assert not late.exists(), (mode, "descendant survived and committed")
                    assert not Path(workspace_marker.read_text()).exists(), (mode, "workspace leaked")
                    checks.append(dict(mode=mode, dispatch_count=count,
                                       late_descendant_write=False,
                                       workspace_removed=True,
                                       state=structured.get("state"),
                                       cleanup=structured.get("cleanup")))
                result = call(dict(jsonrpc="2.0", id=5, method="tools/call", params=dict(
                    name="o_execute", arguments=dict(source="python^( __oval_result__ = 7 )_python"))))
                assert result["structuredContent"]["result"]["value"]["v"]["v"] == "7"
                checks.append(dict(recovery=7))
                status = "passed"
            except BaseException as error:
                status = "failed"
                failure = {"type": type(error).__name__, "message": str(error)}
                raise
            finally:
                proc.stdin.close()
                proc.wait(timeout=15)
                evidence.write_text(json.dumps(dict(
                    schema="ostadix.mcp-lifted-lifecycle/v2", status=status,
                    failure=failure, utc=stamp, binary_sha256=tested_binary_sha256,
                    exchange=records, checks=checks), indent=2) + "\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
