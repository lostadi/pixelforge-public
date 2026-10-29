"""Android envelope compatibility checks; no sockets or MCP processes are opened."""

import copy
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ostadix_appfunction_host.py"
SPEC = importlib.util.spec_from_file_location("appfunction_host", SCRIPT)
HOST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HOST)


class AppFunctionHostTests(unittest.TestCase):
    def handler(self, body):
        handler = HOST.Handler.__new__(HOST.Handler)
        payload = json.dumps(body).encode()
        handler.path = "/o_execute"
        handler.headers = {"Authorization": "Bearer synthetic-test-token",
                           "Content-Length": str(len(payload))}
        handler.rfile = io.BytesIO(payload)
        handler.connection = Mock()
        handler.respond = Mock()
        handler.server = SimpleNamespace(
            config={"token": "synthetic-test-token", "max_timeout_ms": 900000,
                    "root": "/unused", "mcp_executable": "/unused/mcp"},
            slots=HOST.threading.BoundedSemaphore(1))
        return handler

    def test_preserves_source_and_input_envelope(self):
        source = '# λ\r\npython^(\n__oval_result__ = "$x\\n"\n)_python\n'
        body = {"source": source, "bindings": {}, "constraints": {"timeout_ms": 1001}}
        original = copy.deepcopy(body)
        arguments, timeout = HOST.mcp_arguments(body, 900000)
        self.assertEqual(arguments, {"source": source, "timeout_secs": 2})
        self.assertEqual(arguments["source"].encode(), source.encode())
        self.assertEqual(timeout, 1001)
        self.assertEqual(body, original)

    def test_timeout_boundaries_and_omitted_options(self):
        for milliseconds, seconds in [(1, 1), (999, 1), (1000, 1), (900000, 900)]:
            with self.subTest(milliseconds=milliseconds):
                arguments, timeout = HOST.mcp_arguments(
                    {"source": "1", "constraints": {"timeout_ms": milliseconds}}, 900000)
                self.assertEqual(arguments["timeout_secs"], seconds)
                self.assertEqual(timeout, milliseconds)
        self.assertEqual(HOST.mcp_arguments({"source": "1"}, 1501),
                         ({"source": "1", "timeout_secs": 2}, 1501))

    def test_parse_check_uses_same_tool_without_execution_options(self):
        arguments, timeout = HOST.mcp_arguments(
            {"source": "python^( invalid )", "action": "check",
             "constraints": {"timeout_ms": 15000}}, 120000)
        self.assertEqual(arguments, {"source": "python^( invalid )", "action": "check", "timeout_secs": 15})
        self.assertEqual(timeout, 15000)
        self.assertEqual(HOST.mcp_arguments({"source": "1", "action": "execute"}, 1000)[0],
                         {"source": "1", "timeout_secs": 1})

    def test_unsupported_inputs_never_spawn_or_dispatch(self):
        invalid = [None, [], "source", {"source": "1", "path": "program.O"},
                   {"source": "1", "bindings": {"answer": 7}},
                   {"source": "1", "bindings": []},
                   {"source": "1", "bindings": None},
                   {"source": "1", "constraints": []},
                   {"source": "1", "constraints": None},
                   {"source": "1", "constraints": {"max_source_bytes": 64}}]
        invalid += [{"source": "1", "constraints": {"timeout_ms": value}}
                    for value in [True, 0, -1, 900001, 1.5, "1000", None]]
        invalid += [{"source": "1", "action": value}
                    for value in [None, False, [], {}, "plan", "compile", "EXECUTE"]]
        with patch.object(HOST.subprocess, "Popen") as popen, \
                patch.object(HOST, "send") as send, patch.object(HOST, "log"):
            for body in invalid:
                with self.subTest(body=body):
                    handler = self.handler(body)
                    handler.do_POST()
                    status, result = handler.respond.call_args.args
                    self.assertEqual(status, 400)
                    self.assertIs(result["dispatched"], False)
                    self.assertIs(result["retry"], False)
            popen.assert_not_called()
            send.assert_not_called()

    def run_mock_exchange(self, timeout_ms, result, cancel=False):
        handler = self.handler({"source": "1", "bindings": {},
                                "constraints": {"timeout_ms": timeout_ms}})
        frames = Mock()
        frames.get.side_effect = ([{"id": 1, "result": {}}]
                                  + ([None] if cancel else [])
                                  + [{"id": 2, "result": result}])
        with patch.object(HOST.subprocess, "Popen") as popen, \
                patch.object(HOST.queue, "Queue", return_value=frames), \
                patch.object(HOST.threading, "Thread"), \
                patch.object(HOST, "send") as send, patch.object(HOST, "log"), \
                patch.object(HOST.select, "select", return_value=([], [], [])), \
                patch.object(HOST.time, "monotonic", side_effect=[100, 100.002, 100.002]):
            handler.do_POST()
            popen.assert_called_once()
            return handler, [call.args[1] for call in send.call_args_list]

    def test_current_structured_result_and_error_pass_through(self):
        for is_error in [False, True]:
            with self.subTest(is_error=is_error):
                result = {"isError": is_error, "content": [], "structuredContent": {
                    "job_id": "synthetic-job", "state": "failed" if is_error else "completed",
                    "exit_code": 1 if is_error else 0,
                    "result": {"value": 1, "evidence": {"synthetic": True}}}}
                handler, frames = self.run_mock_exchange(1001, result)
                self.assertEqual(frames[-1]["params"], {
                    "name": "o_execute", "arguments": {"source": "1", "timeout_secs": 2}})
                handler.respond.assert_called_once_with(200, result)
                self.assertIs(handler.respond.call_args.args[1], result)

    def test_ms_deadline_cancels_before_rounded_mcp_second_without_retry(self):
        result = {"isError": True, "structuredContent": {"state": "timed_out"}}
        handler, frames = self.run_mock_exchange(1, result, cancel=True)
        calls = [frame for frame in frames if frame["method"] == "tools/call"]
        cancellations = [frame for frame in frames if frame["method"] == "notifications/cancelled"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["params"]["arguments"]["timeout_secs"], 1)
        self.assertEqual(len(cancellations), 1)
        self.assertEqual(cancellations[0]["params"]["requestId"], 2)
        handler.respond.assert_called_once_with(200, result)


if __name__ == "__main__":
    unittest.main()
