"""Offline LiteRT adapter tests; fake hosts never load a model or use hardware."""
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import npu_runtime
from local_runtime import LocalError, LocalRuntime


HOST = r'''
import base64, json, os, sys, time
from pathlib import Path
root = Path(__file__).resolve().parent
mode = (root / "mode").read_text()
with (root / "launches").open("a") as file:
    file.write(json.dumps({"pid": os.getpid(), "sid": os.getsid(0),
                           "argv": sys.argv[1:], "preload": os.environ.get("LD_PRELOAD"),
                           "libraries": os.environ.get("LD_LIBRARY_PATH")}) + "\n")
if mode == "startup_exit":
    sys.exit(17)
if mode == "startup_error":
    print("ERROR\t" + base64.b64encode(b"fixture cannot load model").decode(), flush=True)
    sys.exit(19)
if mode == "startup_block":
    time.sleep(60)
if mode == "startup_wrong_pid":
    print("READY\t0\t0.125", flush=True)
elif mode == "startup_nan":
    print("READY\t%d\tnan" % os.getpid(), flush=True)
else:
    print("READY\t%d\t0.125" % os.getpid(), flush=True)
def encoded_reply(text):
    return base64.b64encode(json.dumps({"role":"assistant","content":[{"type":"text","text":text}]}, ensure_ascii=False).encode()).decode()
for line in sys.stdin:
    limit, encoded = line.rstrip("\n").split("\t")
    prompt = base64.b64decode(encoded, validate=True).decode()
    with (root / "requests").open("a") as file:
        file.write(json.dumps({"limit": int(limit), "prompt": prompt, "pid": os.getpid()}) + "\n")
    if mode == "block":
        time.sleep(60)
    elif mode == "delayed":
        time.sleep(.2)
        print("JSONRESULT\t0.2\t3\t" + encoded_reply("delayed reply"), flush=True)
    elif mode == "malformed":
        print("NOT_A_RESULT", flush=True)
    elif mode == "invalid_base64":
        print("JSONRESULT\t0.01\t3\t!!!", flush=True)
    elif mode == "invalid_utf8":
        print("JSONRESULT\t0.01\t3\t/w==", flush=True)
    elif mode == "oversized":
        print("JSONRESULT\t0.01\t3\t" + "A" * (2 * 1024 * 1024), flush=True)
    elif mode == "truncated":
        sys.stdout.write("JSONRESULT\t0.01\t3\taGVsbG8=")
        sys.stdout.flush()
        sys.exit(0)
    elif mode.startswith("time:"):
        print("JSONRESULT\t" + mode.split(":", 1)[1] + "\t3\t" + encoded_reply("ok"), flush=True)
    elif mode.startswith("count:"):
        print("JSONRESULT\t0.01\t" + mode.split(":", 1)[1] + "\t" + encoded_reply("ok"), flush=True)
    elif mode == "long_result":
        print("JSONRESULT\t0.01\t3\t" + encoded_reply("λ" * 40000), flush=True)
    elif mode == "error":
        print("ERROR\t" + base64.b64encode(b"fixture generation failed").decode(), flush=True)
    elif mode == "analysis_final":
        message = {"role":"assistant", "content":[
            {"type":"text", "channel":"analysis", "text":"private internal draft"},
            {"type":"text", "channel":"final", "text":"Visible answer"}]}
        print("JSONRESULT\t0.01\t3\t" + base64.b64encode(json.dumps(message).encode()).decode(), flush=True)
    elif mode == "invalid_json":
        print("JSONRESULT\t0.01\t3\t" + base64.b64encode(b"{broken").decode(), flush=True)
    elif mode == "invalid_content":
        print("JSONRESULT\t0.01\t3\t" + base64.b64encode(b'{"role":"assistant","content":["wrong"]}').decode(), flush=True)
    else:
        reply = "fixture reply: " + prompt
        print("JSONRESULT\t0.25\t3\t" + encoded_reply(reply), flush=True)
'''


def eventually(check, timeout=4):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for fake LiteRT host')


class LiteRTEngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ostadix-npu-adapter-')
        self.root = Path(self.temp.name)
        self.runtime = self.root / 'runtime' / 'npu'
        self.runtime.mkdir(parents=True)
        self.state = self.root / 'state'
        self.state.mkdir(mode=0o700)
        self.host = self.runtime / 'fake-host'
        self.host.write_text('#!' + sys.executable + '\n' + HOST)
        self.host.chmod(0o700)
        self.model = self.runtime / 'fixture.litertlm'
        self.model.write_bytes(b'not a model; offline protocol fixture\n')
        self.capi = self.runtime / 'lib' / 'liblitert-lm.so'
        self.capi.parent.mkdir()
        self.capi.write_bytes(b'not a shared library\n')
        self.dispatch = self.runtime / 'dispatch' / 'libLiteRtDispatch_GoogleTensor.so'
        self.dispatch.parent.mkdir()
        self.dispatch.write_bytes(b'not a dispatch library\n')
        (self.runtime / 'mode').write_text('normal')
        self.manifest_path = self.runtime / 'npu-model.json'
        self.manifest = self.profile()
        self.write_profile()
        self.patch_root = patch.object(npu_runtime, 'ROOT', self.root)
        self.patch_root.start()
        self.engines = []

    def tearDown(self):
        for engine in self.engines:
            engine.close()
        self.patch_root.stop()
        self.temp.cleanup()

    def profile(self):
        # This fixture deliberately supplies content pins for all executable inputs.
        manifest = {'schema_version': 1, 'backend': 'litert-npu',
                    'model_name': 'Offline fixture', 'model_repo': 'fixture/offline',
                    'model_revision': 'a' * 40, 'quantization': 'fixture',
                    'provenance_repo': 'fixture/offline', 'provenance_revision': 'b' * 40,
                    'runtime_release': 'fixture', 'dispatch_release': 'fixture',
                    'parallel': 1, 'context_per_worker': 1280, 'max_output_tokens': 512}
        for prefix, path in [('model', self.model), ('server', self.host),
                             ('runtime_library', self.capi), ('dispatch_library', self.dispatch)]:
            content = path.read_bytes()
            manifest[prefix + '_path'] = str(path.relative_to(self.root))
            if prefix == 'model':
                manifest['model_bytes'] = len(content)
            manifest[prefix + '_sha256'] = hashlib.sha256(content).hexdigest()
        return manifest

    def write_profile(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def engine(self, mode='normal'):
        (self.runtime / 'mode').write_text(mode)
        engine = npu_runtime.LiteRTEngine(self.state, manifest_path=self.manifest_path)
        self.engines.append(engine)
        return engine

    def records(self, name):
        path = self.runtime / name
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def run_task(self, engine, task='hello', max_tokens=32):
        registered = []
        answer = engine.run(task, 'helper', max_tokens, threading.Event(), registered.append)
        return answer, registered

    def test_main_phone_role_can_request_host_tools_while_helpers_remain_text_only(self):
        engine = self.engine()
        engine.start()
        engine.run('Return a tool request', 'phone conversation assistant', 32,
                   threading.Event(), lambda handle: None)
        self.run_task(engine)
        main, helper = self.records('requests')
        self.assertIn('host validates and executes', main['prompt'])
        self.assertNotIn('You have no tools, shell', main['prompt'])
        self.assertIn('You have no tools, shell', helper['prompt'])

    def test_conversation_json_exposes_only_final_text_and_rejects_bad_schema(self):
        engine = self.engine('analysis_final')
        engine.start()
        (answer, _, _), _ = self.run_task(engine)
        self.assertEqual(answer, 'Visible answer')
        engine.close()
        for mode in ('invalid_json', 'invalid_content'):
            with self.subTest(mode=mode):
                engine = self.engine(mode)
                engine.start()
                with self.assertRaises(LocalError):
                    self.run_task(engine)
                self.assertFalse(engine.is_alive())

    def test_two_requests_reuse_process_and_completed_handle_cleanup_is_harmless(self):
        engine = self.engine()
        model = engine.start()
        self.assertEqual(model['backend'], 'litert-npu')
        pid = engine.process.pid
        first, registered = self.run_task(engine, 'first task: λ\nnext line', 16)
        self.assertIn('first task: λ\nnext line', first[0])
        self.assertEqual(first[1]['completion_tokens'], 3)
        self.assertTrue(first[1]['token_counts_available'])
        self.assertTrue(registered)
        for handle in registered:
            handle.close()
        self.assertTrue(engine.is_alive())
        second, registered = self.run_task(engine, 'second independent task', 32)
        self.assertIn('second independent task', second[0])
        self.assertNotIn('first task:', second[0])
        for handle in registered:
            handle.close()
        self.assertEqual(engine.process.pid, pid)
        self.assertEqual(len(self.records('launches')), 1)
        self.assertEqual([row['limit'] for row in self.records('requests')], [16, 32])
        self.assertEqual({row['pid'] for row in self.records('requests')}, {pid})

    def test_requested_json_format_is_not_contradicted(self):
        engine = self.engine()
        engine.start()
        self.run_task(engine, 'Return JSON only: {"final":"ok"}')
        prompt = self.records('requests')[0]['prompt']
        self.assertIn('Follow the requested response format', prompt)
        self.assertNotIn('Return concise plain text', prompt)
        self.assertIn('Return JSON only:', prompt)

    def test_child_has_dedicated_session_and_process_local_library_environment(self):
        engine = self.engine()
        with patch.dict(os.environ, {'LD_PRELOAD': '/invalid/inherited/preload.so'}):
            engine.start()
        launched = self.records('launches')[0]
        self.assertEqual(launched['sid'], launched['pid'])
        self.assertIsNone(launched['preload'])
        self.assertEqual(launched['libraries'], str(self.capi.parent) + ':/vendor/lib64')
        self.assertEqual(launched['argv'][0], str(self.model))
        self.assertIn(str(self.dispatch.parent), launched['argv'])
        self.assertIn('1280', launched['argv'])

    def test_all_content_pins_checked_before_any_process_starts(self):
        original = dict(self.manifest)
        for prefix in ('model', 'server', 'runtime_library', 'dispatch_library'):
            mutations = [('sha256', '0' * 64)]
            if prefix == 'model':
                mutations.append(('bytes', 999999))
            for suffix, value in mutations:
                with self.subTest(prefix=prefix, suffix=suffix):
                    self.manifest = {**original, prefix + '_' + suffix: value}
                    self.write_profile()
                    with patch.object(npu_runtime.subprocess, 'Popen') as spawn:
                        with self.assertRaises(LocalError):
                            self.engine().start()
                        spawn.assert_not_called()

    def test_outside_runtime_and_symlink_escape_rejected_before_spawn(self):
        external = self.root / 'outside.litertlm'
        external.write_bytes(self.model.read_bytes())
        escape = self.runtime / 'escape.litertlm'
        escape.symlink_to(external)
        original = dict(self.manifest)
        for path in (str(external), 'runtime/npu/../../outside.litertlm', str(escape)):
            with self.subTest(path=path):
                self.manifest = {**original, 'model_path': path}
                self.write_profile()
                with patch.object(npu_runtime.subprocess, 'Popen') as spawn:
                    with self.assertRaises(LocalError):
                        self.engine().start()
                    spawn.assert_not_called()

    def test_cancellation_terminates_owned_host_without_replaying_request(self):
        engine = self.engine('block')
        engine.start()
        process = engine.process
        cancelled = threading.Event()
        registered, failures = [], []
        def run():
            try:
                engine.run('cancel this one', 'helper', 32, cancelled, registered.append)
            except LocalError as error:
                failures.append(str(error))
        thread = threading.Thread(target=run)
        thread.start()
        try:
            eventually(lambda: self.records('requests') and registered)
            cancelled.set()
            registered[0].close()
            thread.join(4)
            self.assertFalse(thread.is_alive())
            eventually(lambda: process.poll() is not None)
            self.assertFalse(engine.is_alive())
            self.assertEqual(len(self.records('requests')), 1)
            self.assertEqual(len(self.records('launches')), 1)
            with self.assertRaises(LocalError):
                self.run_task(engine, 'must not silently restart')
            self.assertEqual(len(self.records('launches')), 1)
        finally:
            cancelled.set()
            engine.close()
            thread.join(4)

    def test_close_does_not_stop_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        try:
            engine = self.engine()
            engine.start()
            owned = engine.process
            engine.close()
            engine.close()
            eventually(lambda: owned.poll() is not None)
            self.assertIsNone(unrelated.poll())
        finally:
            unrelated.terminate()
            unrelated.wait(timeout=4)

    def test_malformed_oversized_and_truncated_frames_fail_closed(self):
        for mode in ('malformed', 'invalid_base64', 'invalid_utf8', 'oversized', 'truncated'):
            with self.subTest(mode=mode):
                engine = self.engine(mode)
                engine.start()
                with self.assertRaises(LocalError):
                    self.run_task(engine)
                engine.close()

    def test_nonfinite_or_negative_generation_time_is_rejected(self):
        for value in ('nan', 'inf', '-1'):
            with self.subTest(value=value):
                engine = self.engine('time:' + value)
                engine.start()
                with self.assertRaises(LocalError):
                    self.run_task(engine)
                engine.close()

    def test_explicit_generation_error_is_not_reported_as_an_answer(self):
        engine = self.engine('error')
        engine.start()
        pid = engine.process.pid
        for task in ('rejected request', 'another request'):
            with self.assertRaisesRegex(LocalError, 'fixture generation failed'):
                self.run_task(engine, task)
            self.assertTrue(engine.is_alive())
        self.assertEqual(engine.process.pid, pid)
        self.assertEqual(len(self.records('launches')), 1)
        self.assertEqual(len(self.records('requests')), 2)

    def test_scheduler_retains_model_between_jobs_and_advertises_real_limits(self):
        service = LocalRuntime(
            self.state,
            lambda state: npu_runtime.LiteRTEngine(state, manifest_path=self.manifest_path))
        try:
            eventually(lambda: service.ready or service.error)
            status = service.status()
            self.assertTrue(status['ready'], status['error'])
            self.assertEqual(status['limits']['active'], 1)
            self.assertEqual(status['limits']['max_tokens'], 512)
            pid = status['engine_pid']
            for task in ('first scheduled job', 'second scheduled job'):
                job = service.submit('fixture-owner', task, max_tokens=32)
                result = service.read('fixture-owner', job['id'], wait=3)
                self.assertEqual(result['state'], 'completed', result['error'])
                self.assertIn(task, result['result'])
                self.assertEqual(result['usage']['completion_tokens'], 3)
                self.assertTrue(service.engine.is_alive())
                self.assertEqual(service.status()['engine_pid'], pid)
            with self.assertRaises(LocalError):
                service.submit('fixture-owner', 'exceeds output limit', max_tokens=513)
            self.assertEqual(len(self.records('requests')), 2)
            self.assertEqual(len(self.records('launches')), 1)
        finally:
            service.close()

    def test_failed_startup_does_not_leave_a_live_process(self):
        for mode in ('startup_exit', 'startup_error', 'startup_wrong_pid', 'startup_nan'):
            with self.subTest(mode=mode):
                engine = self.engine(mode)
                with self.assertRaises(LocalError):
                    engine.start()
                self.assertFalse(engine.is_alive())

    def test_startup_and_generation_have_bounded_timeouts(self):
        with patch.object(npu_runtime, 'START_TIMEOUT', .1):
            engine = self.engine('startup_block')
            with self.assertRaisesRegex(LocalError, 'timed out'):
                engine.start()
            self.assertFalse(engine.is_alive())
        engine = self.engine('block')
        engine.start()
        with patch.object(npu_runtime, 'REQUEST_TIMEOUT', .1):
            with self.assertRaisesRegex(LocalError, 'timed out'):
                self.run_task(engine)
        self.assertFalse(engine.is_alive())
        self.assertEqual(len(self.records('requests')), 1)

    def test_main_generation_outlives_helper_deadline_and_still_cancels(self):
        engine = self.engine('delayed')
        engine.start()
        with patch.object(npu_runtime, 'REQUEST_TIMEOUT', .02):
            result, _, _ = engine.run('Keep working', 'phone conversation assistant', 32,
                                      threading.Event(), lambda _: None)
        self.assertEqual(result, 'delayed reply')
        self.assertTrue(engine.is_alive())
        engine.close()
        engine = self.engine('block')
        engine.start()
        cancelled = threading.Event()
        registered, errors = [], []
        def run():
            try:
                engine.run('Cancel this request', 'phone conversation assistant', 32,
                           cancelled, registered.append)
            except LocalError as error:
                errors.append(str(error))
        thread = threading.Thread(target=run)
        thread.start()
        try:
            eventually(lambda: len(self.records('requests')) == 2 and registered)
            cancelled.set()
            registered[0].close()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertTrue(errors)
            self.assertFalse(engine.is_alive())
            self.assertEqual(len(self.records('requests')), 2)
        finally:
            cancelled.set()
            engine.close()
            thread.join(2)

    def test_unknown_count_stays_unknown_and_token_limit_is_reported(self):
        for count, available, limited in [(-1, False, None), (16, True, True), (3, True, False)]:
            with self.subTest(count=count):
                engine = self.engine('count:' + str(count))
                engine.start()
                answer, _ = self.run_task(engine, max_tokens=16)
                usage = answer[1]
                self.assertEqual(usage['token_counts_available'], available)
                self.assertEqual(usage['token_limit_reached'], limited)
                if available:
                    self.assertEqual(usage['completion_tokens'], count)
                else:
                    self.assertIsNone(usage.get('completion_tokens'))
                engine.close()

    def test_large_utf8_result_is_bounded_without_corrupting_text(self):
        engine = self.engine('long_result')
        engine.start()
        answer, _ = self.run_task(engine)
        self.assertEqual(answer[0], 'λ' * 32768)
        self.assertTrue(answer[2])
        self.assertTrue(engine.is_alive())

    def test_output_limit_is_checked_before_request_submission(self):
        engine = self.engine()
        engine.start()
        for value in (0, 513, True, 1.5):
            with self.subTest(value=value), self.assertRaises(LocalError):
                self.run_task(engine, max_tokens=value)
        self.assertEqual(self.records('requests'), [])


if __name__ == '__main__':
    unittest.main()
