"""Controller regression checks using fresh kits and owned subprocesses only."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


CONTROLLER_PATH = Path(__file__).with_name("controller.py")
SPEC = importlib.util.spec_from_file_location("pixelforge_controller", CONTROLLER_PATH)
controller = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controller)


def report(backend, tokens=None):
    value = {
        "success": True, "backend": backend, "source_gguf_sha256": "a" * 64,
        "engine_sha256": "b" * 64, "prompt": "Hi", "model": "/same/model.gguf",
        "engine": {"success": True, "generated_tokens": tokens if tokens is not None else [2, 5]},
        "engine_cleanup_ok": True, "broker_cleanup_ok": True,
    }
    if backend != "cpu-reference":
        value["broker"] = {"success": True, "native_cleanup_ok": True,
                           "npu": {"npu_invocations": 3, "compiled_models": 0, "loaded_models": 2}}
    return value


def process_alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0]
        return state != "Z"
    except FileNotFoundError:
        return False


def process_start_ticks(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[19]
    except FileNotFoundError:
        return None


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pixelforge-controller-test-")
        self.kit = Path(self.temporary.name) / "kit with spaces"
        self.tools = self.kit / "tools/arm_npu"
        self.tools.mkdir(parents=True)
        self.environment = self.kit / "custom-models/gguf-runtime/venv/bin"
        self.environment.mkdir(parents=True)
        self.python = self.environment / "python"
        self.python.symlink_to(sys.executable)
        self.saved_stderr = io.StringIO()

    def tearDown(self):
        self.temporary.cleanup()

    def script(self, name, source):
        path = self.kit / "benchmark.py" if name == "benchmark.py" else self.tools / name
        path.write_text(source)
        return path

    def call(self, arguments, materialize=None):
        callback = materialize or (lambda _directory=None: self.kit)
        with contextlib.redirect_stderr(self.saved_stderr):
            return controller.integration_main(arguments, callback)

    def write_reports(self, cpu=None, npu=None):
        cpu_path, npu_path = Path(self.temporary.name) / "cpu.json", Path(self.temporary.name) / "npu.json"
        cpu_path.write_text(json.dumps(cpu if cpu is not None else report("cpu-reference")))
        npu_path.write_text(json.dumps(npu if npu is not None else report("google-tensor-npu-dense+ggml-cpu")))
        return cpu_path, npu_path

    def test_help_never_extracts_or_requires_venv(self):
        self.python.unlink()
        callback = mock.Mock(side_effect=AssertionError("help must not extract"))
        for arguments in ([], ["--help"], ["help"]):
            result = self.call(arguments, callback)
            self.assertEqual(result["exit_code"], 0)
            self.assertIn("compile", result["actions"])
        callback.assert_not_called()

    def test_kit_destination_forwarded_exactly(self):
        destination = str(Path(self.temporary.name) / "requested destination")
        callback = mock.Mock(return_value=self.kit)
        result = self.call(["--kit-dir", destination, "extract"], callback)
        self.assertEqual(result["exit_code"], 0)
        callback.assert_called_once_with(destination)

    def test_argv_is_literal_and_failure_exit_is_preserved(self):
        marker = Path(self.temporary.name) / "shell-executed"
        payload = "model with spaces;$(touch " + str(marker) + ")`echo injected`"
        self.script("gguf_inspect.py", "import json,sys\nprint(json.dumps({'argv':sys.argv[1:]}))\nprint('diagnostic',file=sys.stderr)\nraise SystemExit(7)\n")
        result = self.call(["inspect", payload, "--tensors"])
        self.assertEqual(result["exit_code"], 7)
        self.assertEqual(result["child"]["returncode"], 7)
        self.assertEqual(result["result"]["argv"], [payload, "--tensors"])
        self.assertFalse(marker.exists())
        self.assertIn("diagnostic", self.saved_stderr.getvalue())

    def test_relative_user_model_and_output_use_caller_directory(self):
        caller = Path(self.temporary.name) / "caller directory"
        caller.mkdir()
        (caller / "relative model.gguf").write_text("chosen model bytes")
        self.script("compile_gguf.py", "import json,pathlib,sys\na=sys.argv[1:]\nmodel=pathlib.Path(a[a.index('--model')+1])\nout=pathlib.Path(a[a.index('--output')+1]);out.mkdir()\n(out/'source.txt').write_text(model.read_text())\nprint(json.dumps({'cwd':str(pathlib.Path.cwd()),'model':model.read_text()}))\n")
        with mock.patch.dict(os.environ, {"OSTADIX_NPU_CALLER_CWD": str(caller)}):
            result = self.call(["compile", "--model", "relative model.gguf", "--output", "relative output"])
        self.assertEqual(result["exit_code"], 0, result)
        self.assertEqual(result["result"]["cwd"], str(caller))
        self.assertEqual((caller / "relative output/source.txt").read_text(), "chosen model bytes")
        self.assertFalse((self.kit / "relative output").exists())

    def test_venv_python_symlink_is_not_resolved(self):
        self.assertEqual(controller._pf_numeric_python(self.kit), str(self.python))
        self.assertEqual(controller._pf_numeric_python(self.kit, str(self.python)), str(self.python))
        self.assertNotEqual(str(self.python.resolve()), str(self.python))

    def test_worker_guard_argv_preserves_command_and_selects_python_cleanup(self):
        source = "# embedded worker test source"
        original = [str(self.python), "script with spaces.py", "--prompt", "literal $(shell)"]
        with mock.patch.object(controller, "_pf_worker_source", return_value=source):
            guarded = controller._pf_guarded_argv(original, os.getpid())
            self.assertEqual(guarded[:4], ["/usr/bin/python3", "-I", "-c", source])
            separator = guarded.index("--")
            self.assertEqual(guarded[separator + 1:], original)
            self.assertIn("--python", guarded[:separator])
            self.assertEqual(guarded[guarded.index("--owner-pid") + 1], str(os.getpid()))
            self.assertEqual(guarded[guarded.index("--owner-start-ticks") + 1], process_start_ticks(os.getpid()))
            native = controller._pf_guarded_argv(["/bin/native", "argument with spaces"], os.getpid())
            self.assertNotIn("--python", native[:native.index("--")])

    def test_worker_guard_prefers_valid_outer_launcher_owner(self):
        ticks = process_start_ticks(os.getpid())
        with mock.patch.object(controller, "_pf_worker_source", return_value="# embedded guard"):
            with mock.patch.dict(os.environ, {"OSTADIX_NPU_LAUNCHER_PID": str(os.getpid()), "OSTADIX_NPU_LAUNCHER_START_TICKS": ticks}):
                command = controller._pf_guarded_argv([sys.executable, "-c", "pass"], 1)
                self.assertEqual(command[command.index("--owner-pid") + 1], str(os.getpid()))
                self.assertEqual(command[command.index("--owner-start-ticks") + 1], ticks)
            with mock.patch.dict(os.environ, {"OSTADIX_NPU_LAUNCHER_PID": "malformed", "OSTADIX_NPU_LAUNCHER_START_TICKS": ticks}):
                command = controller._pf_guarded_argv([sys.executable, "-c", "pass"], os.getpid())
                self.assertEqual(command[command.index("--owner-pid") + 1], str(os.getpid()))
                self.assertEqual(command[command.index("--owner-start-ticks") + 1], ticks)

    def test_gguf_shell_alias_selects_guarded_python_cli(self):
        self.script("gguf.sh", "raise AssertionError('shell alias must dispatch directly')\n")
        self.script("compile_gguf.py", "import json,sys\nprint(json.dumps({'argv':sys.argv[1:]}))\n")
        result = self.call(["tool", "gguf.sh", "compile", "--model", "model with spaces"])
        self.assertEqual(result["exit_code"], 0, result)
        self.assertEqual(result["result"]["argv"], ["--model", "model with spaces"])

    def test_missing_numeric_environment_is_helpful(self):
        self.python.unlink()
        self.script("gguf_inspect.py", "raise AssertionError('must not start')\n")
        result = self.call(["inspect", "model.gguf"])
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("Run setup first", result["error"])

    def test_script_help_reads_ast_without_importing_numeric_modules(self):
        self.python.unlink()
        self.script("gguf_inspect.py", "'''Inspector description.'''\nimport module_that_does_not_exist\nimport argparse\np=argparse.ArgumentParser()\np.add_argument('model')\np.add_argument('--tensors',action='store_true',help='Tensor descriptors')\n")
        result = self.call(["inspect", "--help"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["help"]["description"], "Inspector description.")
        self.assertIn({"flags": ["--tensors"], "action": "store_true", "help": "Tensor descriptors"}, result["help"]["arguments"])

    def test_plan_and_cpu_append_correct_backend_flags(self):
        for action, name, flag in (("plan", "compile_gguf.py", "--plan-only"), ("cpu", "run_gguf.py", "--cpu")):
            self.script(name, "import json,sys\nprint(json.dumps({'argv':sys.argv[1:]}))\n")
            result = self.call([action, "--model", "chosen model"])
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["result"]["argv"], [flag, "--model", "chosen model"])

    def test_verify_runs_with_system_python_and_no_numeric_venv(self):
        self.python.unlink()
        self.script("setup_gguf.py", "import json,sys\nprint(json.dumps({'argv':sys.argv[1:],'python':sys.executable}))\n")
        result = self.call(["verify", "--addon", "directory with spaces"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["result"]["python"], "/usr/bin/python3")
        self.assertEqual(result["result"]["argv"], ["--verify-only", "--addon", "directory with spaces"])

    def test_tool_whitelist_and_legacy_environment(self):
        result = self.call(["tool", "../arbitrary.py"])
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("whitelist", result["error"])
        self.script("model_host.py", "raise AssertionError('legacy environment missing')\n")
        result = self.call(["tool", "model_host.py", "--prompt", "Hi"])
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("separate MiniCPM", result["error"])

    def test_redirected_script_is_refused(self):
        outside = Path(self.temporary.name) / "outside.py"
        outside.write_text("raise AssertionError('redirected script')\n")
        (self.tools / "gguf_inspect.py").symlink_to(outside)
        result = self.call(["inspect", "model"])
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("redirected", result["error"])

    def test_output_tails_are_bounded(self):
        self.script("gguf_inspect.py", "import json\nprint('x'*200000)\nprint(json.dumps({'complete':True}))\n")
        result = self.call(["inspect", "model"])
        self.assertEqual(result["exit_code"], 0)
        self.assertLessEqual(len(result["child"]["stdout_tail"].encode()), controller._PF_TAIL_BYTES)
        self.assertTrue(result["result"]["complete"])

    def test_compare_reports_is_read_only_and_needs_no_extraction(self):
        cpu, npu = self.write_reports()
        original_cpu = cpu.read_bytes()
        original_npu = npu.read_bytes()
        callback = mock.Mock(side_effect=AssertionError("must not extract"))
        result = self.call(["compare-reports", str(cpu), str(npu)], callback)
        self.assertEqual(result["exit_code"], 0)
        self.assertTrue(result["tokens_match"])
        self.assertFalse(result["full_logit_equivalence_claimed"])
        self.assertEqual(cpu.read_bytes(), original_cpu)
        self.assertEqual(npu.read_bytes(), original_npu)
        callback.assert_not_called()

    def test_compare_reports_relative_paths_use_caller_directory(self):
        cpu, npu = self.write_reports()
        with mock.patch.dict(os.environ, {"OSTADIX_NPU_CALLER_CWD": self.temporary.name}):
            result = self.call(["compare-reports", cpu.name, npu.name])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["cpu_report"], str(cpu))

    def test_contradictory_optional_npu_receipts_are_rejected(self):
        mutations = [
            lambda r: r["engine"].update(dense_npu_requested=False),
            lambda r: r["broker"]["npu"].update(cpu_fallback=True),
            lambda r: r["broker"].update(source_gguf_sha256="c" * 64),
            lambda r: r["broker"]["npu"].update(model_sha256="c" * 64),
            lambda r: r["broker"]["npu"].update(source_gguf="/different/model.gguf"),
            lambda r: r["broker"]["npu"].update(invocations=[{}, {}]),
            lambda r: r["engine"].update(steps=[{"step": 0, "token": 2}]),
            lambda r: r["engine"].update(steps=[{"step": 0, "token": 2}, {"step": 1, "token": 99}]),
            lambda r: r["engine"].update(steps=[{"step": 1, "token": 2}, {"step": 2, "token": 5}]),
            lambda r: r["engine"].update(steps=[{"step": 0, "token": 2, "eos": True}, {"step": 1, "token": 5}]),
            lambda r: (r["engine"].update(npu_weight_matmul_operations=4), r["broker"].update(matrix_operations=5)),
        ]
        for index, mutate in enumerate(mutations):
            npu_report = report("google-tensor-npu-dense+ggml-cpu")
            mutate(npu_report)
            cpu, npu = self.write_reports(npu=npu_report)
            self.assertEqual(self.call(["compare-reports", str(cpu), str(npu)])["exit_code"], 2, index)

    def test_matching_optional_provenance_and_prompt_tokens_are_validated(self):
        cpu_report = report("cpu-reference")
        npu_report = report("google-tensor-npu-dense+ggml-cpu")
        for r in (cpu_report, npu_report):
            r["engine"].update(prompt_tokens=2, steps=[{"step": 0, "token": 2}, {"step": 1, "token": 5}])
        cpu_report["engine"].update(dense_npu_requested=False, npu_weight_matmul_operations=0)
        npu_report["engine"].update(dense_npu_requested=True, npu_weight_matmul_operations=4)
        npu_report["broker"].update(source_gguf_sha256="a" * 64, matrix_operations=4)
        npu_report["broker"]["npu"].update(cpu_fallback=False, source_gguf="/same/model.gguf", invocations=[{}, {}, {}])
        cpu, npu = self.write_reports(cpu_report, npu_report)
        self.assertEqual(self.call(["compare-reports", str(cpu), str(npu)])["exit_code"], 0)
        npu_report["engine"]["prompt_tokens"] = 3
        cpu, npu = self.write_reports(cpu_report, npu_report)
        result = self.call(["compare-reports", str(cpu), str(npu)])
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("prompt_tokens", result["error"])

    def test_compare_help_has_its_own_options(self):
        result = self.call(["compare", "--help"])
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("--artifacts", result["help"])
        self.assertNotIn("--cpu", result["help"])

    def test_compare_missing_failed_and_mismatched_reports_are_rejected(self):
        cpu, npu = self.write_reports()
        npu.unlink()
        result = self.call(["compare-reports", str(cpu), str(npu)])
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("missing", result["error"])
        for field, value in (("success", False), ("backend", "cpu-reference"), ("engine_sha256", "c" * 64), ("source_gguf_sha256", "d" * 64), ("prompt", "different")):
            changed = report("google-tensor-npu-dense+ggml-cpu")
            changed[field] = value
            cpu, npu = self.write_reports(npu=changed)
            result = self.call(["compare-reports", str(cpu), str(npu)])
            self.assertEqual(result["exit_code"], 2, field)
        changed = report("google-tensor-npu-dense+ggml-cpu")
        changed["broker"]["npu"]["npu_invocations"] = 0
        cpu, npu = self.write_reports(npu=changed)
        self.assertEqual(self.call(["compare-reports", str(cpu), str(npu)])["exit_code"], 2)

    def test_token_disagreement_is_an_unsuccessful_comparison(self):
        cpu, npu = self.write_reports(npu=report("google-tensor-npu-dense+ggml-cpu", [2, 6]))
        result = self.call(["compare-reports", str(cpu), str(npu)])
        self.assertEqual(result["exit_code"], 1)
        self.assertFalse(result["tokens_match"])

    def test_nonfinite_report_json_is_refused(self):
        cpu, npu = self.write_reports()
        npu.write_text('{"value": NaN}')
        result = self.call(["compare-reports", str(cpu), str(npu)])
        self.assertEqual(result["exit_code"], 2)
        self.assertIn("Nonfinite", result["error"])

    def test_compare_runs_same_configuration_in_fresh_directories(self):
        self.script("run_gguf.py", "import json,pathlib,sys\na=sys.argv[1:]\nout=pathlib.Path(a[a.index('--output')+1]);out.mkdir(mode=0o700)\nprompt=a[a.index('--prompt')+1]\nv=" + repr(report("cpu-reference")) + "\nv['prompt']=prompt\nif '--cpu' not in a:\n v=" + repr(report("google-tensor-npu-dense+ggml-cpu")) + "\n v['prompt']=prompt\n(out/'RESULT.json').write_text(json.dumps(v))\nprint(json.dumps({'report':str(out/'RESULT.json'),'argv':a,'success':True}))\n")
        output = Path(self.temporary.name) / "comparison output"
        result = self.call(["compare", "--model", "chosen GGUF", "--prompt", "literal $(not shell)", "--tokens", "2", "--output", str(output), "--artifacts", "saved banks", "--read-only"])
        self.assertEqual(result["exit_code"], 0, result)
        cpu_argv = result["children"]["cpu"]["result"]["argv"]
        npu_argv = result["children"]["npu"]["result"]["argv"]
        self.assertIn("--cpu", cpu_argv)
        self.assertNotIn("--artifacts", cpu_argv)
        self.assertNotIn("--cpu", npu_argv)
        self.assertIn("--read-only", npu_argv)
        self.assertEqual(cpu_argv[cpu_argv.index("--prompt") + 1], npu_argv[npu_argv.index("--prompt") + 1])
        repeated = self.call(["compare", "--model", "chosen GGUF", "--output", str(output)])
        self.assertEqual(repeated["exit_code"], 2)
        self.assertIn("FileExistsError", repeated["error_type"])

    def test_build_native_refuses_existing_output(self):
        output = Path(self.temporary.name) / "validated helper"
        output.write_text("validated")
        result = self.call(["build-native", str(output)])
        self.assertEqual(result["exit_code"], 2)
        self.assertEqual(output.read_text(), "validated")

    def test_native_tool_alias_uses_safe_output_and_refuses_overwrite(self):
        self.script("build_native.sh", "#!/bin/bash\n/usr/bin/python3 -c 'import json,sys;print(json.dumps({\"destination\":sys.argv[1]}))' \"$1\"\n")
        result = self.call(["tool", "build_native.sh"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["result"]["destination"], str(self.kit / "custom-models/gguf-runtime/build/native/npu_pipe_gguf"))
        output = Path(self.temporary.name) / "validated helper"
        output.write_text("validated")
        self.assertEqual(self.call(["tool", "build_native.sh", str(output)])["exit_code"], 2)
        self.assertEqual(output.read_text(), "validated")

    def test_native_protocol_fixture_never_needs_vendor_runtime(self):
        if not (shutil.which("g++") or shutil.which("clang++")):
            self.skipTest("No C++ compiler")
        self.script("test_pipe_protocol.cc", "#include <cassert>\nint main(){assert(2+2==4);return 0;}\n")
        result = self.call(["test-native"])
        self.assertEqual(result["exit_code"], 0, result)
        self.assertFalse(result["hardware_used"])

    def test_deadline_requests_graceful_cleanup_and_returns_124(self):
        cleaned = Path(self.temporary.name) / "cleanup done"
        self.script("gguf_inspect.py", "import pathlib,signal,time\ndef stop(s,f):\n pathlib.Path(" + repr(str(cleaned)) + ").write_text('cleaned')\n raise SystemExit(0)\nsignal.signal(signal.SIGTERM,stop)\nprint('ready',flush=True)\nwhile True: time.sleep(.01)\n")
        before = time.monotonic()
        result = self.call(["--deadline", "0.35", "inspect", "model"])
        self.assertEqual(result["exit_code"], 124)
        expected = 130 if sys.platform == "linux" and controller._pf_worker_source() else 0
        self.assertEqual(result["child"]["returncode"], expected)
        self.assertTrue(cleaned.exists())
        self.assertFalse(result["child"]["cleanup_forced"])
        self.assertLess(time.monotonic() - before, 5)

    def test_real_worker_unwinds_python_finally_on_deadline(self):
        if not controller._pf_worker_source() or sys.platform != "linux":
            self.skipTest("Real Linux worker source is not available")
        cleaned = Path(self.temporary.name) / "compile finally done"
        self.script("compile_gguf.py", "import pathlib,time\ntry:\n print('ready',flush=True)\n while True: time.sleep(.01)\nfinally:\n pathlib.Path(" + repr(str(cleaned)) + ").write_text('finally released')\n")
        result = self.call(["--deadline", "0.35", "compile", "--model", "stub model"])
        self.assertEqual(result["exit_code"], 124, result)
        self.assertEqual(cleaned.read_text(), "finally released")
        self.assertFalse(result["child"]["cleanup_forced"])

    def test_real_worker_preserves_python_interpreter_flags(self):
        if not controller._pf_worker_source() or sys.platform != "linux":
            self.skipTest("Real Linux worker source is not available")
        source = "import json,sys;print(json.dumps({'isolated':sys.flags.isolated,'no_bytecode':sys.flags.dont_write_bytecode}))"
        with contextlib.redirect_stderr(self.saved_stderr):
            result = controller._PFExecution(5).run([sys.executable, "-B", "-I", "-c", source], self.kit)
        self.assertEqual(result["exit_code"], 0, result)
        self.assertEqual(result["result"], {"isolated": 1, "no_bytecode": 1})

    def test_cancel_file_releases_owned_child_and_returns_130(self):
        cancelled = Path(self.temporary.name) / "cancel"
        ready = Path(self.temporary.name) / "ready"
        cleaned = Path(self.temporary.name) / "cleaned"
        self.script("gguf_inspect.py", "import pathlib,signal,time\ndef stop(s,f):\n pathlib.Path(" + repr(str(cleaned)) + ").write_text('cleaned')\n raise SystemExit(0)\nsignal.signal(signal.SIGTERM,stop)\npathlib.Path(" + repr(str(ready)) + ").touch()\nwhile True: time.sleep(.01)\n")
        def cancel():
            until = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < until:
                time.sleep(.01)
            cancelled.touch()
        thread = threading.Thread(target=cancel)
        thread.start()
        try:
            with mock.patch.dict(os.environ, {"OSTADIX_NPU_CANCEL_FILE": str(cancelled)}):
                result = self.call(["--deadline", "5", "inspect", "model"])
            self.assertEqual(result["exit_code"], 130)
            expected = 130 if sys.platform == "linux" and controller._pf_worker_source() else 0
            self.assertEqual(result["child"]["returncode"], expected)
            self.assertTrue(cleaned.exists())
        finally:
            thread.join(timeout=5)

    @unittest.skipUnless(sys.platform == "linux" and hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"), "Parent-death fixture cleanup requires Linux pidfds")
    def test_parent_death_signal_runs_child_cleanup_of_detached_descendant(self):
        ready = Path(self.temporary.name) / "owned-processes.json"
        cleaned = Path(self.temporary.name) / "parent-death-cleaned"
        child = Path(self.temporary.name) / "child.py"
        child.write_text("import json,os,pathlib,signal,subprocess,sys,time\nworker=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True)\ndef stop(s,f):\n worker.terminate();worker.wait(timeout=3)\n pathlib.Path(" + repr(str(cleaned)) + ").write_text('detached child cleaned')\n raise SystemExit(0)\nsignal.signal(signal.SIGTERM,stop)\npathlib.Path(" + repr(str(ready)) + ").write_text(json.dumps({'child':os.getpid(),'worker':worker.pid}))\nwhile True: time.sleep(.02)\n")
        wrapper = "import importlib.util;from pathlib import Path;s=importlib.util.spec_from_file_location('c'," + repr(str(CONTROLLER_PATH)) + ");c=importlib.util.module_from_spec(s);s.loader.exec_module(c);c._PFExecution(30).run(" + repr([sys.executable, str(child)]) + ",Path(" + repr(str(self.kit)) + "))"
        parent = subprocess.Popen([sys.executable, "-c", wrapper], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        owned = None
        ticks = {}
        descriptors = {}
        try:
            until = time.monotonic() + 5
            while not ready.exists() and time.monotonic() < until:
                time.sleep(.02)
            self.assertTrue(ready.exists(), "Child did not publish owned PIDs")
            owned = json.loads(ready.read_text())
            ticks = {pid: process_start_ticks(pid) for pid in owned.values()}
            for pid, expected in ticks.items():
                descriptor = os.pidfd_open(pid)
                if expected is None or process_start_ticks(pid) != expected:
                    os.close(descriptor)
                    raise AssertionError("Owned process identity changed before pidfd acquisition")
                descriptors[pid] = descriptor
            parent.kill()
            parent.wait(timeout=5)
            until = time.monotonic() + 5
            while not cleaned.exists() and time.monotonic() < until:
                time.sleep(.02)
            self.assertTrue(cleaned.exists(), "PDEATHSIG did not reach the runner cleanup handler")
            self.assertFalse(process_alive(owned["worker"]), "Detached owned worker survived cleanup")
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            if owned:
                for pid in owned.values():
                    descriptor = descriptors.get(pid)
                    if descriptor is not None:
                        try:
                            signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        finally:
                            os.close(descriptor)

    def test_invalid_deadline_unknown_action_and_collect_mode(self):
        for arguments in (["--deadline", "nan", "extract"], ["--deadline", "0", "extract"], ["--deadline", "3501", "extract"], ["missing-action"]):
            self.assertEqual(self.call(arguments)["exit_code"], 2)
        self.script("benchmark.py", "raise AssertionError('must not run')\n")
        self.assertEqual(self.call(["collect", "--run"])["exit_code"], 2)


if __name__ == "__main__":
    unittest.main()
