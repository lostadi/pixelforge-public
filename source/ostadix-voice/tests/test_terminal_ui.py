"""Offline UI, live PTY, and local routing checks. No cloud or model requests."""
import copy
import errno
import io
import json
import os
from pathlib import Path
import pty
import select
import subprocess
import sys
import tempfile
import termios
import threading
import time
import types
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from terminal import TerminalPrompt, TerminalSession, TerminalUI, cell_width
from agent import Agent, AppError
from ostadix_tools import OstadixTools
from test_app import FakeRPC, eventually
from test_ostadix_tools import Client as OstadixClient
from test_terminal import Audio


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.prompt = TerminalPrompt(self.output)

    def test_split_unicode_and_editing_preserve_exact_request(self):
        encoded = 'Ask 🦙 مرحبا'.encode()
        lines = []
        for byte in encoded:
            lines += self.prompt.feed(bytes([byte]))
        self.assertEqual(lines, [])
        self.assertEqual(self.prompt.text, 'Ask 🦙 مرحبا')
        self.prompt.feed(b'\x1b[D\x7f')
        self.assertEqual(self.prompt.feed(b'\r'), ['Ask 🦙 مرحا'])

    def test_bracketed_paste_cannot_dispatch_commands(self):
        self.assertEqual(self.prompt.feed(b'\x1b[200~hello\n/approve deadbeef\n/quit\x1b[201~'), [])
        self.assertEqual(self.prompt.feed(b'\r'), ['hello /approve deadbeef /quit'])

    def test_edit_while_output_arrives_preserves_input(self):
        self.prompt.visible = True
        self.prompt.feed('Draft 🦙 request'.encode())
        self.prompt.feed(b'\x1b[D\x1b[D')
        position = self.prompt.position
        self.prompt.write('Local worker completed')
        self.assertEqual(self.prompt.text, 'Draft 🦙 request')
        self.assertEqual(self.prompt.position, position)
        self.assertIn('Local worker completed\n', self.output.getvalue())
        self.assertIn('Draft 🦙 request', self.output.getvalue().split('\n')[-1])

    def test_crlf_is_one_submit_and_history_restores_draft(self):
        self.assertEqual(self.prompt.feed(b'one\r\n'), ['one'])
        self.prompt.feed(b'draft\x1b[A')
        self.assertEqual(self.prompt.text, 'one')
        self.prompt.feed(b'\x1b[B')
        self.assertEqual(self.prompt.text, 'draft')

    def test_resize_bounds_wide_input_without_discarding_text(self):
        self.prompt.feed(('全' * 90).encode())
        with patch('terminal.shutil.get_terminal_size', return_value=os.terminal_size((32, 20))):
            frame = self.prompt.frame()
            shown = frame.split('\x1b[2K', 1)[1].split('\r', 1)[0]
            self.assertLess(cell_width(shown), 32)
            self.assertEqual(self.prompt.text, '全' * 90)

    def test_stream_preview_does_not_erase_pending_input(self):
        self.prompt.visible = True
        self.prompt.feed(b'next question')
        self.prompt.preview = 'Codex · Answer arriving'
        self.prompt.render()
        self.prompt.preview = ''
        self.prompt.write('Codex: Answer arriving now.')
        self.assertIn('\x1b[1A', self.output.getvalue())
        self.assertEqual(self.prompt.feed(b'\n'), ['next question'])

    def test_real_pty_keeps_bytes_and_restores_terminal_modes(self):
        master, slave = pty.openpty()
        stream = os.fdopen(os.dup(slave), 'w', buffering=1)
        input_stream = os.fdopen(os.dup(slave), 'r')
        original = termios.tcgetattr(slave)
        try:
            prompt = TerminalPrompt(stream, input_stream)
            with prompt:
                self.assertFalse(termios.tcgetattr(slave)[3] & termios.ICANON)
                os.write(master, 'Phone dictation 🦙'.encode())
                self.assertTrue(select.select([slave], [], [], 2)[0])
                self.assertEqual(prompt.feed(os.read(slave, 4096)), [])
                prompt.write('Worker finished while typing')
                os.write(master, b'\r')
                self.assertTrue(select.select([slave], [], [], 2)[0])
                self.assertEqual(prompt.feed(os.read(slave, 4096)), ['Phone dictation 🦙'])
            self.assertEqual(termios.tcgetattr(slave), original)
            self.assertTrue(select.select([master], [], [], 2)[0])
            output = b''
            deadline = time.monotonic() + 2
            while b'\x1b[?2004l' not in output and time.monotonic() < deadline:
                if select.select([master], [], [], .1)[0]:
                    output += os.read(master, 8192)
            self.assertIn(b'\x1b[?2004h', output)
            self.assertIn(b'\x1b[?2004l', output)
        finally:
            stream.close()
            input_stream.close()
            os.close(master)
            os.close(slave)

    def test_real_ui_typing_survives_background_event_and_quits(self):
        source = '''
import pathlib, tempfile, threading, time
from terminal import TerminalSession, TerminalUI
with tempfile.TemporaryDirectory() as directory:
    root = pathlib.Path(directory) / 'workspace'; root.mkdir()
    session = TerminalSession(root, pathlib.Path(directory) / 'state')
    ui = TerminalUI(session, audio=False); ui.local = None
    ui.companion_next_poll = float('inf')
    def notice():
        time.sleep(.3)
        session.store.event(session.sid, 'local_assistant', {
            'action': 'create', 'id': 'listener-fixture', 'name': 'Listener', 'state': 'waiting'})
        session.store.event(session.sid, 'notice', {'text': 'BACKGROUND EVENT'})
    threading.Thread(target=notice, daemon=True).start()
    try: ui.run()
    finally: session.close()
'''
        master, slave = pty.openpty()
        process = subprocess.Popen([sys.executable, '-c', source], stdin=slave, stdout=slave, stderr=slave,
                                   cwd=Path(__file__).resolve().parents[1], env=dict(os.environ, TERM='xterm-256color'))
        captured = b''
        def read_until(marker):
            nonlocal captured
            deadline = time.monotonic() + 4
            while marker not in captured and time.monotonic() < deadline:
                if select.select([master], [], [], .1)[0]:
                    captured += os.read(master, 16384)
            self.assertIn(marker, captured)
        try:
            read_until(b'\x1b[?2004h')
            os.write(master, b'/sta')
            read_until(b'BACKGROUND EVENT')
            self.assertNotIn(b'Traceback', captured)
            os.write(master, b'tus\r')
            read_until(b'saved task ')
            os.write(master, b'/quit\r')
            process.wait(timeout=4)
            self.assertEqual(process.returncode, 0, captured.decode(errors='replace'))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=3)
            os.close(master)
            os.close(slave)

    def test_stopped_background_session_exits_and_preserves_shell_terminal(self):
        # A controlling terminal and separate process groups are necessary:
        # openpty/Popen alone cannot expose SIGTTIN/SIGTTOU job-control stops.
        source = '''
import copy, fcntl, json, os, signal, sys, termios, time, threading
from pathlib import Path

fcntl.ioctl(0, termios.TIOCSCTTY, 0)
signal.signal(signal.SIGTTOU, signal.SIG_IGN)
original = termios.tcgetattr(0)
ready_read, ready_write = os.pipe()
go_read, go_write = os.pipe()
child = os.fork()
if child == 0:
    os.setpgid(0, 0)
    signal.signal(signal.SIGTTOU, signal.SIG_DFL)
    os.close(ready_read); os.close(go_write)
    os.read(go_read, 1)
    import terminal
    # A background Python worker must not receive the shutdown signal instead
    # of the main thread that is blocked in the terminal read.
    worker_ready = threading.Event()
    def hold_worker():
        worker_ready.set()
        threading.Event().wait()
    threading.Thread(target=hold_worker, daemon=True).start()
    assert worker_ready.wait(2), 'worker did not start'
    def wait_for_stop(ui):
        os.write(ready_write, b'R')
        if sys.argv[3] == 'read':
            os.read(0, 1)
        else:
            while not ui.quit:
                time.sleep(.01)
    terminal.TerminalUI.run_loop = wait_for_stop
    # The fixture has no model/audio requests and must also run in root CI.
    terminal.os.geteuid = lambda: 10000
    terminal.main(['--workspace', sys.argv[1], '--state-dir', sys.argv[2], '--no-audio'])
    raise SystemExit(0)

os.close(ready_write); os.close(go_read)
reaped = False
try:
    os.setpgid(child, child)
    os.tcsetpgrp(0, child)
    os.write(go_write, b'G')
    assert os.read(ready_read, 1) == b'R', 'app did not reach input loop'
    os.kill(child, signal.SIGSTOP)
    _, status = os.waitpid(child, os.WUNTRACED)
    assert os.WIFSTOPPED(status), 'app did not suspend'
    os.tcsetpgrp(0, os.getpgrp())
    shell_modes = copy.deepcopy(original)
    shell_modes[3] ^= termios.ECHONL
    shell_modes[3] |= termios.TOSTOP
    termios.tcsetattr(0, termios.TCSANOW, shell_modes)
    from launch import _signal_ui_main_thread
    _signal_ui_main_thread(child, signal.SIGTERM)
    os.kill(child, signal.SIGCONT)
    deadline = time.monotonic() + 3
    outcome = {'timeout': True}
    while time.monotonic() < deadline:
        pid, status = os.waitpid(child, os.WNOHANG | os.WUNTRACED)
        if pid:
            if os.WIFSTOPPED(status):
                outcome = {'stopped_again': os.WSTOPSIG(status)}
            else:
                reaped = True
                outcome = {'exit': os.waitstatus_to_exitcode(status)}
            break
        time.sleep(.01)
    outcome['shell_modes_preserved'] = termios.tcgetattr(0) == shell_modes
    if reaped:
        with (Path(sys.argv[2]) / 'server.lock').open('r') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        outcome['session_released'] = True
    print('JOB_CONTROL_RESULT=' + json.dumps(outcome), flush=True)
finally:
    if not reaped:
        os.kill(child, signal.SIGKILL)
        os.waitpid(child, 0)
'''
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'workspace'
            root.mkdir()
            for mode in ('read', 'idle'):
                with self.subTest(mode=mode):
                    master, slave = pty.openpty()
                    process = subprocess.Popen(
                        [sys.executable, '-c', source, str(root), str(Path(directory) / mode), mode],
                        stdin=slave, stdout=slave, stderr=slave, start_new_session=True,
                        cwd=Path(__file__).resolve().parents[1],
                        env=dict(os.environ, TERM='xterm-256color'))
                    os.close(slave)
                    captured = b''
                    try:
                        deadline = time.monotonic() + 8
                        while time.monotonic() < deadline:
                            if select.select([master], [], [], .1)[0]:
                                try:
                                    chunk = os.read(master, 16384)
                                except OSError as error:
                                    if error.errno != errno.EIO:
                                        raise
                                    break
                                if not chunk:
                                    break
                                captured += chunk
                            elif process.poll() is not None:
                                break
                        process.wait(timeout=2)
                        output = captured.decode(errors='replace')
                        self.assertEqual(process.returncode, 0, output)
                        self.assertIn('JOB_CONTROL_RESULT=', output)
                        result = json.loads(output.split('JOB_CONTROL_RESULT=', 1)[1].splitlines()[0])
                        self.assertEqual(result, {'exit': 0, 'shell_modes_preserved': True,
                                                  'session_released': True}, output)
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait(timeout=3)
                        os.close(master)


class LocalFixture:
    def __init__(self):
        self.calls = []
        self.job = {'id': '12345678abcdef', 'state': 'queued', 'role': 'assistant'}
    def status(self):
        self.calls.append(('status',))
        return {'running': True, 'ready': True, 'model': {'name': 'Huihui LFM2', 'quantization': 'Q8_0'}, 'active': 0, 'queued': 0}
    def submit(self, owner, task, **options):
        self.calls.append(('submit', owner, task, options))
        return self.job.copy()
    def jobs(self, owner):
        self.calls.append(('jobs', owner))
        return [self.job.copy()]
    def read(self, owner, job_id, **options):
        self.calls.append(('read', owner, job_id, options))
        return dict(self.job, state='completed', result='Local answer', usage={'completion_tokens': 4})
    def cancel(self, owner, job_id=None):
        self.calls.append(('cancel', owner, job_id))
        return {'cancelled': 1}


class AssistantFixture(LocalFixture):
    def __init__(self):
        super().__init__()
        self.assistants = {}

    def assistant_create(self, owner, name, role='assistant'):
        self.calls.append(('assistant_create', owner, name, role))
        item = {'id': 'abcdef1234567890abcdef1234567890', 'owner': owner, 'name': name, 'role': role,
                'state': 'waiting', 'guidance': '', 'memory': '', 'assignment': None}
        self.assistants[(owner, name)] = item
        return copy.deepcopy(item)

    def assistant_list(self, owner):
        self.calls.append(('assistant_list', owner))
        return [copy.deepcopy(item) for (scope, _), item in self.assistants.items() if scope == owner]

    def assistant_read(self, owner, name):
        self.calls.append(('assistant_read', owner, name))
        return copy.deepcopy(self.assistants[(owner, name)])

    def assistant_assign(self, owner, name, task, max_steps=2, max_tokens=512):
        self.calls.append(('assistant_assign', owner, name, task, max_steps, max_tokens))
        item = self.assistants[(owner, name)]
        item.update(state='queued', assignment={'id': 'assignment-123', 'state': 'queued', 'task': task,
                    'max_steps': max_steps, 'max_tokens': max_tokens, 'completed_steps': 0, 'latest_result': '', 'steps': []})
        return copy.deepcopy(item)

    def assistant_guide(self, owner, name, text):
        self.calls.append(('assistant_guide', owner, name, text))
        self.assistants[(owner, name)]['guidance'] = text
        return copy.deepcopy(self.assistants[(owner, name)])

    def assistant_pause(self, owner, name):
        self.calls.append(('assistant_pause', owner, name))
        self.assistants[(owner, name)]['state'] = 'pause_pending'
        return copy.deepcopy(self.assistants[(owner, name)])

    def assistant_resume(self, owner, name):
        self.calls.append(('assistant_resume', owner, name))
        self.assistants[(owner, name)]['state'] = 'running'
        return copy.deepcopy(self.assistants[(owner, name)])

    def assistant_cancel(self, owner, name):
        self.calls.append(('assistant_cancel', owner, name))
        self.assistants[(owner, name)]['state'] = 'cancelled'
        return copy.deepcopy(self.assistants[(owner, name)])


class FluidUITests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        base = Path(self.directory.name)
        self.root = base / 'workspace'
        self.root.mkdir()
        self.session = TerminalSession(self.root, base / 'state', FakeRPC)
        self.output = []
        self.audio = Audio()
        self.ui = TerminalUI(self.session, self.audio, self.output.append)
        self.ui.local = None
        self.ui.companion = Mock()
        self.ui.companion.messages.return_value = {'messages': [], 'eligible': False}
        self.ui.restore()
    def tearDown(self):
        self.session.close()
        self.directory.cleanup()

    def settle_local(self):
        eventually(lambda: not self.ui.local_events.empty())
        self.ui.pump_local()

    def test_compact_banner_and_full_help_on_request(self):
        self.output.clear()
        self.ui.banner()
        banner = '\n'.join(self.output)
        self.assertLessEqual(len(banner.splitlines()), 2)
        self.assertIn('Ostadix', banner)
        self.assertIn('spoken replies', banner)
        self.assertIn('Type to chat', banner)
        self.assertIn('Enter to talk', banner)
        for command in ('/voice on', '/help'):
            self.assertIn(command, banner)
        self.assertNotIn('/approve ID', banner)
        self.ui.handle('/help')
        self.assertIn('/approvals on|off', self.output[-1])
        self.assertIn('/help all', self.output[-1])
        self.assertLessEqual(len(self.output[-1].splitlines()), 10)
        self.assertNotIn('/assistant create', self.output[-1])
        self.ui.handle('/help all')
        self.assertIn('/audio', self.output[-1])
        self.assertIn('/assistant create', self.output[-1])

    def test_controls_show_current_settings_without_contacting_devices_or_models(self):
        self.output.clear()
        self.session.agent.engine_label = 'Local Huihui-gemma-4-E2B-it-abliterated-v2'
        self.ui.audio = False
        self.session.agent.set_action_policy('ask')
        with patch('terminal.subprocess.Popen') as popen, patch('terminal.subprocess.run') as run:
            self.ui.handle('/controls')
        output = '\n'.join(self.output)
        self.assertLessEqual(len(self.output), 8)
        for value in ('Local Huihui-gemma-4-E2B-it-abliterated-v2', 'Voice: off',
                      'Speaker: off', 'Approve all: off', 'OSTADIX-BM embedded', '/o SOURCE', '/engine gemini|local|codex',
                      'NPU helper: checking', '/npu on|off', '@npu'):
            self.assertIn(value, output)
        self.assertNotIn('Gemini', output)
        self.assertEqual(self.audio.calls, [])
        self.assertIsNone(self.session.agent.rpc)
        self.assertEqual(self.session.agent.ostadix.connections, {})
        popen.assert_not_called()
        run.assert_not_called()

    def test_engine_switch_preserves_voice_settings_and_saved_conversation(self):
        previous, sid = self.session.agent, self.session.sid
        self.session.draft('unsent text')
        self.ui.audio = False
        replacement = Agent(self.session.store, self.root, rpc_factory=FakeRPC)
        replacement.engine_label = 'Local Huihui Gemma'
        with patch('local_chat.create_agent', return_value=replacement) as factory:
            self.ui.handle('/engine local')
        factory.assert_called_once_with(self.session.store, self.session.root)
        self.assertTrue(previous.closed)
        self.assertIs(self.session.agent, replacement)
        self.assertEqual(self.session.store.setting('chat_engine'), 'local')
        self.assertEqual(self.session.sid, sid)
        self.assertEqual(self.session.draft(), 'unsent text')
        self.assertFalse(self.ui.audio)
        self.assertFalse(self.ui.handsfree)
        self.assertEqual(self.audio.calls, [])
        self.assertIn('Engine: Local Huihui Gemma', self.output[-1])
        self.ui.handle('/engine')
        self.assertIn('Local Huihui Gemma', self.output[-1])

    def test_selected_engine_unavailable_is_distinct_from_ready_helpers(self):
        client = types.SimpleNamespace(engine_name='local-npu', status=Mock())
        self.session.agent.local_client = client
        self.session.agent.engine_label = 'Selected NPU model'
        self.ui.observe_engine(self.session.agent, client, {
            'ready': False, 'error': 'Selected model is disabled'})
        self.ui.local = LocalFixture()
        self.ui.local_status = {'ready': True, 'model': {'name': 'CPU model'}}
        self.ui.npu_status = {'ready': True, 'model': {'backend': 'litert-npu'}}
        self.output.clear()
        self.ui.handle('/controls')
        self.assertLessEqual(len(self.output), 8)
        shown = '\n'.join(self.output)
        self.assertIn('Engine: Selected NPU model · unavailable: Selected model is disabled', shown)
        self.assertIn('CPU helper: Local ready', shown)
        self.assertIn('NPU helper:', shown)
        self.assertIn('engine unavailable', self.ui.activity_badge(self.ui.speech_counts()))
        self.ui.handle('/status')
        self.assertNotIn('ready on demand', '\n'.join(self.output))
        client.status.assert_not_called()

    def test_selected_engine_recovery_is_visible_during_active_wait(self):
        client = types.SimpleNamespace(engine_name='local-npu')
        self.session.agent.local_client = client
        self.ui.observe_engine(self.session.agent, client, {
            'ready': False, 'recovering': True, 'error': 'Waiting for the NPU supervisor'})
        self.session.agent.active = True
        try:
            self.ui.handle('/controls')
            self.assertIn('recovering: Waiting for the NPU supervisor', '\n'.join(self.output))
            self.assertIn('engine recovering', self.ui.activity_badge(self.ui.speech_counts()))
        finally:
            self.session.agent.active = False

    def test_stale_engine_status_cannot_change_current_selection_health(self):
        old = types.SimpleNamespace(engine_name='local-npu')
        current = types.SimpleNamespace(engine_name='gemini')
        self.session.agent.local_client = current
        self.ui.observe_engine(self.session.agent, current, {'ready': True})
        self.ui.observe_engine(self.session.agent, old, {'ready': False, 'error': 'Old NPU failed'})
        self.assertEqual(self.ui.engine_health(), ('configured; connection checked on request', ''))
        self.assertNotIn('Old NPU failed', self.ui.engine_summary())

    def test_saved_failure_from_before_engine_switch_does_not_fault_new_engine(self):
        self.session.store.event(self.session.sid, 'finished', {'status': 'failed', 'text': 'Old engine failed'})
        replacement = Agent(self.session.store, self.root, rpc_factory=FakeRPC)
        replacement.engine_label = 'New selected engine'
        with patch('local_chat.create_agent', return_value=replacement):
            self.ui.handle('/engine local')
        self.ui.pump()
        self.assertEqual(self.ui.engine_health(), ('unchecked; checked on request', ''))
        self.assertNotIn('Old engine failed', self.ui.engine_summary())

    def test_engine_switch_rejects_busy_or_invalid_requests(self):
        previous = self.session.agent
        with patch('local_chat.create_agent') as factory:
            with self.assertRaisesRegex(AppError, 'Use /engine'):
                self.ui.handle('/engine unknown')
            previous.active = True
            with self.assertRaisesRegex(AppError, 'current task'):
                self.ui.handle('/engine local')
            previous.active = False
        self.assertIs(self.session.agent, previous)
        self.assertFalse(previous.closed)
        factory.assert_not_called()

    def test_audio_controls_report_installed_commands_without_using_hardware(self):
        self.output.clear()
        with patch('terminal.shutil.which', side_effect=lambda name: '/fixture/' + name), \
                patch('terminal.subprocess.Popen') as popen, \
                patch('terminal.subprocess.run') as run:
            self.ui.handle('/audio')
        output = '\n'.join(self.output)
        self.assertIn('idle', output)
        self.assertIn('automatic speech on', output)
        self.assertIn('installed', output.lower())
        self.assertIn('microphone', output.lower())
        self.assertIn('speaker', output.lower())
        self.assertIn('allow microphone', output.lower())
        self.assertIn('foreground', output.lower())
        for claim in ('microphone verified', 'speaker verified', 'audio verified'):
            self.assertNotIn(claim, output.lower())
        self.assertEqual(self.audio.calls, [])
        self.assertIsNone(self.session.agent.rpc)
        popen.assert_not_called()
        run.assert_not_called()

    def test_audio_controls_explain_missing_commands_and_keyboard_fallback(self):
        self.output.clear()
        with patch('terminal.shutil.which', return_value=None):
            self.ui.handle('/audio')
        output = '\n'.join(self.output)
        self.assertIn('missing', output.lower())
        self.assertIn('termux-api', output)
        self.assertIn('Termux:API', output)
        self.assertIn('keyboard', output.lower())
        self.assertEqual(self.audio.calls, [])
        self.assertIsNone(self.session.agent.rpc)

    def test_status_uses_the_audio_summary_and_shows_stopping(self):
        self.audio.stopping = True
        self.ui.audio = False
        with patch('terminal.shutil.which', return_value=None):
            self.output.clear()
            self.ui.handle('/audio')
            audio_lines = [line for line in self.output
                           if line.startswith(('Audio:', 'Microphone:', 'Speaker:'))]
            self.output.clear()
            self.ui.handle('/status')
        self.assertEqual(len(audio_lines), 3)
        self.assertTrue(all(line in self.output for line in audio_lines))
        output = '\n'.join(self.output)
        self.assertIn('stopping', output.lower())
        self.assertIn('automatic speech off', output)
        self.assertEqual(self.audio.calls, [])
        self.assertIsNone(self.session.agent.rpc)

    def test_failed_or_cancelled_task_never_speaks_as_an_answer(self):
        for status in ('failed', 'interrupted'):
            self.session.store.event(self.session.sid, 'finished', {'status': status, 'text': 'Task error'})
            self.ui.pump()
        self.assertEqual(self.audio.calls, [])
        self.assertEqual(self.ui.last_answer, '')

    def test_companion_controls_are_local_and_reply_text_is_data(self):
        controller = self.ui.companion
        controller.status.return_value = {'enabled': True, 'running': True, 'phase': 'waiting'}
        controller.read.return_value = {'message': {'id': 'greeting', 'text': 'What would you like to learn?'}}
        controller.reply.return_value = {'phase': 'queued'}
        controller.snooze.return_value = {'enabled': True, 'snoozed_until': time.time() + 3600}
        controller.dismiss.return_value = {}
        for command, operation, expected in (
                ('/companion', 'status', ()), ('/companion read', 'read', ()),
                ('/companion reply /quit is a command I am learning', 'reply', ('/quit is a command I am learning',)),
                ('/companion snooze', 'snooze', ()), ('/companion dismiss', 'dismiss', ())):
            self.ui.handle(command)
            eventually(lambda: not self.ui.companion_busy)
            self.ui.companion_next_poll = time.monotonic() + 100
            self.ui.pump_companion()
            if operation == 'snooze':
                controller.snooze.assert_called_once_with(minutes=60)
            else:
                getattr(controller, operation).assert_called_once_with(*expected)
        self.assertTrue(any('What would you like to learn?' in text for text in self.output))
        self.assertTrue(any('reply queued on your phone' in text for text in self.output))
        self.assertFalse(self.ui.quit)
        controller.enable.assert_not_called(); controller.disable.assert_not_called()
        self.assertIsNone(self.session.agent.rpc)
        self.assertEqual(self.audio.calls, [])

    def test_companion_control_requests_remain_responsive_and_ordered(self):
        controller = self.ui.companion
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def enable():
            entered.set()
            self.assertTrue(release.wait(3))
            return {'enabled': True, 'running': True}
        controller.enable.side_effect = enable
        controller.disable.return_value = {'enabled': False}
        self.ui.handle('/companion on')
        self.assertTrue(entered.wait(1))
        self.ui.handle('/companion off')
        self.ui.handle('/help all')
        self.assertIn('/companion off', self.output[-1].replace('/companion on|off', '/companion off'))
        controller.disable.assert_not_called()
        release.set()
        eventually(lambda: not self.ui.companion_busy)
        self.ui.companion_next_poll = time.monotonic() + 100
        self.ui.pump_companion()
        self.assertEqual([item[0] for item in controller.method_calls[:2]], ['enable', 'disable'])
        self.assertIn('Local companion · off', '\n'.join(self.output))
        self.assertEqual(self.audio.calls, [])
        self.assertIsNone(self.session.agent.rpc)

    def test_companion_invalid_controls_never_reach_service(self):
        for command in ('/companion on extra', '/companion read extra', '/companion reply',
                        '/companion snooze 0', '/companion snooze -1', '/companion snooze 1441',
                        '/companion snooze 5 hours', '/companion snooze ' + '9' * 5000,
                        '/companion dismiss extra', '/companion unknown'):
            with self.subTest(command=command[:60]), self.assertRaisesRegex(Exception, 'Use /companion'):
                self.ui.handle(command)
        self.assertEqual(self.ui.companion.method_calls, [])

    def companion_poll_event(self, *, eligible=True, ident='checkin-one', text='What do you predict?', **fields):
        message = dict(id=ident, text=text, kind='checkin', display_pending=True)
        message.update(fields)
        self.ui.companion_next_poll = time.monotonic() + 100
        self.ui.companion_events.put(('poll', {'eligible': eligible, 'messages': [message]}, None))

    def test_companion_never_interrupts_typing_drafts_audio_or_tasks(self):
        self.output.clear()
        guards = [
            (lambda: setattr(self.ui, 'prompt', types.SimpleNamespace(text='draft', pasting=False)),
             lambda: setattr(self.ui, 'prompt', None)),
            (lambda: self.session.draft('saved speech'), lambda: self.session.draft('')),
            (lambda: setattr(self.audio, 'mode', 'speaking'), lambda: setattr(self.audio, 'mode', None)),
            (lambda: setattr(self.audio, 'mode', 'listening'), lambda: setattr(self.audio, 'mode', None)),
            (lambda: setattr(self.ui, 'handsfree', True), lambda: setattr(self.ui, 'handsfree', False)),
            (lambda: setattr(self.session.agent, 'active', True), lambda: setattr(self.session.agent, 'active', False)),
            (lambda: self.ui.local_status.update(active=1), lambda: self.ui.local_status.update(active=0)),
            (lambda: self.ui.local_status.update(queued=1), lambda: self.ui.local_status.update(queued=0)),
        ]
        for enable, disable in guards:
            enable()
            try:
                self.companion_poll_event()
                self.ui.pump_companion()
                self.assertEqual(self.output, [])
            finally:
                disable()
        self.ui.pump_companion()
        self.assertEqual(len(self.output), 1)
        self.assertIn('Local companion: What do you predict?', self.output[0])
        self.assertIn('/companion reply YOUR MESSAGE', self.output[0])
        self.assertEqual(self.audio.calls, [])
        self.assertIsNone(self.session.agent.rpc)

    def test_companion_pending_messages_require_current_eligibility_and_are_not_replayed(self):
        self.output.clear()
        self.companion_poll_event(eligible=False)
        self.ui.pump_companion()
        self.assertEqual(self.output, [])
        self.companion_poll_event()
        self.ui.pump_companion()
        self.companion_poll_event()
        self.ui.pump_companion()
        self.assertEqual(len(self.output), 1)
        self.assertIn('checkin-one', json.loads(self.session.store.setting('terminal_companion_seen')))
        self.ui = TerminalUI(self.session, self.audio, self.output.append)
        self.ui.local = None
        self.ui.companion = Mock()
        self.companion_poll_event()
        self.ui.pump_companion()
        self.assertEqual(len(self.output), 1)
        self.companion_poll_event(ident='new-reply', kind='reply')
        self.ui.pump_companion()
        self.assertEqual(len(self.output), 2)

    def test_companion_errors_and_shutdown_do_not_start_audio_or_show_pending_message(self):
        self.output.clear()
        self.ui.companion_next_poll = time.monotonic() + 100
        self.ui.companion_events.put(('poll', None, 'Temporary unavailable'))
        self.ui.pump_companion()
        self.assertEqual(self.output, [])
        self.ui.companion_events.put(('read', None, 'Message unavailable'))
        self.ui.pump_companion()
        self.assertEqual(self.output, ['Local companion: Message unavailable'])
        self.ui.quit = True
        self.companion_poll_event()
        self.ui.pump_companion()
        self.assertEqual(len(self.output), 1)
        self.assertEqual(self.audio.calls, [])
        self.ui.companion.enable.assert_not_called()

    def test_optimizer_status_and_history_are_local_read_only_controls(self):
        controller = Mock()
        controller.status.return_value = {'enabled': False, 'running': False, 'objective': 'balanced',
                                          'model_backend': 'litert-npu', 'managed_effects': []}
        controller.history.return_value = {'events': [{'kind': 'stopped', 'message': 'Restoration completed'}]}
        self.ui.optimizer = controller
        for command, operation in (('/optimizer', 'status'), ('/optimizer history', 'history')):
            self.ui.handle(command)
            eventually(lambda: not self.ui.optimizer_events.empty())
            self.ui.pump_optimizer()
            self.assertIn('Optimizer ' + operation, self.output[-1])
        controller.status.assert_called_once_with()
        controller.history.assert_called_once_with(limit=10)
        controller.enable.assert_not_called(); controller.disable.assert_not_called()
        self.assertIn('completed', self.output[-1])
        self.assertIsNone(self.session.agent.rpc)
        self.assertEqual(self.audio.calls, [])

    def test_optimizer_controls_remain_responsive_and_keep_request_order(self):
        controller = Mock()
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def enable(**kwargs):
            entered.set()
            self.assertTrue(release.wait(3))
            return {'enabled': True, 'objective': kwargs['objective']}
        controller.enable.side_effect = enable
        controller.disable.return_value = {'enabled': False, 'managed_effects': [],
                                           'restoration': {'ok': True, 'restored': 5, 'skipped': []}}
        self.ui.optimizer = controller
        started = time.monotonic()
        self.ui.handle('/optimizer on battery')
        self.assertLess(time.monotonic() - started, .5)
        self.assertTrue(entered.wait(1))
        self.ui.handle('/optimizer off')
        self.ui.handle('/help all')
        self.assertIn('/optimizer off', self.output[-1])
        controller.disable.assert_not_called()
        release.set()
        eventually(lambda: not self.ui.optimizer_busy)
        self.ui.pump_optimizer()
        self.assertEqual([call[0] for call in controller.method_calls], ['enable', 'disable'])
        controller.enable.assert_called_once_with(objective='battery')
        self.assertIn('Restoration: 5 threads restored', self.output[-1])
        self.assertIsNone(self.session.agent.rpc)
        self.assertEqual(self.audio.calls, [])

    def test_optimizer_invalid_controls_never_reach_controller(self):
        controller = Mock()
        self.ui.optimizer = controller
        for command in ('/optimizer on root', '/optimizer off performance', '/optimizer run',
                        '/optimizer history 99', '/optimizer on balanced extra'):
            with self.subTest(command=command), self.assertRaisesRegex(Exception, 'Use /optimizer'):
                self.ui.handle(command)
        self.assertEqual(controller.method_calls, [])
        self.assertFalse(self.ui.optimizer_busy)

    def test_optimizer_failure_does_not_claim_settings_were_restored(self):
        controller = Mock()
        controller.disable.side_effect = RuntimeError('Baseline restoration could not be confirmed')
        self.ui.optimizer = controller
        self.ui.handle('/optimizer off')
        eventually(lambda: not self.ui.optimizer_events.empty())
        self.ui.pump_optimizer()
        self.assertEqual(self.output[-1], 'Optimizer: Baseline restoration could not be confirmed')
        self.assertIsNone(self.session.agent.rpc)
        self.assertEqual(self.audio.calls, [])

    def test_optimizer_status_is_compact_and_shows_actual_change_not_pending_advice(self):
        result = {'enabled': True, 'running': True, 'objective': 'balanced', 'phase': 'monitoring',
                  'model_backend': 'litert-npu', 'sample_seconds': 10, 'analysis_seconds': 60,
                  'last_observation': {'battery_percent': 52, 'battery_temp_c': 32.4, 'memory_available_mb': 3686.4},
                  'last_model_analysis': {'recommendation': {'action': 'normal', 'reason': 'Conditions are healthy.'}},
                  'last_decision': {'action': 'observe', 'nice_delta': 0},
                  'last_action': {'decision': {'nice_delta': 5}, 'result': {'ok': True, 'applied': 5}},
                  'managed_effects': {'threads': [{'internal_journal_marker': 'hidden'}] * 512}}
        self.ui.optimizer_events.put(('status', result, None))
        self.ui.pump_optimizer()
        output = self.output[-1]
        self.assertIn('running · balanced · monitoring', output)
        self.assertIn('Backend NPU · samples 10s · analysis 60s when idle', output)
        self.assertIn('battery 52% · 32.4°C · 3.6 GiB available', output)
        self.assertIn('Model: normal · Conditions are healthy.', output)
        self.assertIn('Last change: nice +5 from baseline · 5 threads', output)
        self.assertNotIn('internal_journal_marker', output)
        self.assertNotIn('last_decision', output)
        self.assertLess(len(output), 600)
        self.assertEqual(len(output.splitlines()), 5)
        self.assertEqual(self.audio.calls, [])
        self.assertIsNone(self.session.agent.rpc)

    def test_optimizer_compact_output_preserves_unknowns_and_restoration_errors(self):
        result = {'enabled': False, 'running': False, 'objective': 'battery', 'phase': 'stopped',
                  'last_observation': {'battery_percent': None, 'battery_temp_c': None},
                  'restoration': {'ok': False, 'error': 'Readback did not match'},
                  'error': 'Priority restoration remains incomplete', 'guardian_error': 'Recovery unavailable',
                  'wake_release_error': 'Lease release failed; expires within 30 seconds'}
        self.ui.optimizer_events.put(('off', result, None))
        self.ui.pump_optimizer()
        output = self.output[-1]
        self.assertIn('Optimizer off · off · battery', output)
        self.assertIn('Phone: battery unknown · unknown · unknown', output)
        self.assertIn('Restoration: incomplete · Readback did not match', output)
        self.assertIn('Error: Priority restoration remains incomplete', output)
        self.assertIn('Recovery error: Recovery unavailable', output)
        self.assertIn('Wake release error: Lease release failed', output)
        self.assertNotIn('threads restored', output)
        result['running'] = True
        self.ui.optimizer_events.put(('off', result, None))
        self.ui.pump_optimizer()
        self.assertIn('disabled, stopping', self.output[-1])

    def test_optimizer_history_is_bounded_and_shows_assessments_actions_and_errors(self):
        events = [{'kind': 'model_analysis', 'recommendation': {'action': 'eco', 'reason': 'Reduce CPU pressure.'}},
                  {'kind': 'priority_action', 'decision': {'nice_delta': 0}, 'result': {'ok': True, 'restored': 5}},
                  {'kind': 'error', 'message': 'NPU unavailable\n' + 'x' * 1000},
                  {'kind': 'observation', 'telemetry': {'battery_percent': 50, 'battery_temp_c': 33,
                                                      'memory_available_mb': 2048}}]
        events += [{'kind': 'analysis_submitted', 'job_id': 'private_job_marker'}] * 8
        self.ui.optimizer_events.put(('history', {'events': events}, None))
        self.ui.pump_optimizer()
        output = self.output[-1]
        self.assertIn('Optimizer history · newest first', output)
        self.assertIn('model eco · Reduce CPU pressure.', output)
        self.assertIn('priority restored baseline · 5 threads', output)
        self.assertIn('error · NPU unavailable', output)
        self.assertIn('battery 50% · 33.0°C · 2.0 GiB available', output)
        self.assertIn('model analysis queued', output)
        self.assertNotIn('private_job_marker', output)
        self.assertNotIn('x' * 200, output)
        self.assertEqual(len(output.splitlines()), 11)

    def test_completed_answer_printed_once_and_spoken_once(self):
        self.session.store.event(self.session.sid, 'assistant', {'text': 'Here is the result.'})
        self.session.store.event(self.session.sid, 'finished', {'status': 'idle', 'text': 'Here is the result.'})
        self.ui.pump()
        self.assertEqual(sum('Here is the result.' in line for line in self.output), 1)
        self.assertEqual(self.audio.calls, [('speaking', 'Here is the result.')])

    def test_new_and_steered_user_events_clear_previous_answer_preview(self):
        self.ui.prompt = TerminalPrompt(io.StringIO())
        self.ui.prompt.feed(b'unsent draft')
        for steered in (False, True):
            with self.subTest(steered=steered):
                self.session.store.event(self.session.sid, 'assistant_delta', {
                    'item_id': 'old-answer', 'text': 'Previous answer fragment'})
                self.ui.pump()
                self.assertIn('Previous answer fragment', self.ui.prompt.preview)
                self.session.store.event(self.session.sid, 'user', {
                    'text': 'Updated request', 'steered': steered})
                self.ui.pump()
                self.assertEqual(self.ui.prompt.preview, '')
                self.assertEqual(self.ui.stream_text, '')
                self.assertIsNone(self.ui.stream_item)
                self.assertEqual(self.ui.prompt.text, 'unsent draft')

    def test_tool_preview_is_bounded_and_logs_are_lossless(self):
        content = 'long output\n' * 1000 + 'FINAL SENTINEL'
        self.session.store.event(self.session.sid, 'tool_result', {'success': True, 'result': {'content': content}})
        self.ui.pump()
        self.assertLess(len('\n'.join(self.output)), 1200)
        self.assertNotIn('FINAL SENTINEL', '\n'.join(self.output))
        self.ui.handle('/logs')
        self.assertIn('FINAL SENTINEL', '\n'.join(self.output))

    def test_approval_full_arguments_remain_visible(self):
        argument = 'x' * 1800 + 'EXACT END'
        self.session.store.event(self.session.sid, 'approval', {'id': 'abcdef123456', 'warning': 'Review exact action', 'arguments': {'command': argument}})
        self.ui.pump()
        output = '\n'.join(self.output)
        self.assertIn(argument, output)
        self.assertIn('/approve abcdef12', output)

    def test_local_job_is_bounded_saved_and_never_wakes_codex(self):
        fixture = LocalFixture()
        self.ui.local = fixture
        self.ui.handle('/local ask Summarize this locally')
        self.settle_local()
        self.assertIsNone(self.session.agent.rpc)
        submit = next(call for call in fixture.calls if call[0] == 'submit')
        self.assertEqual(submit[2], 'Summarize this locally')
        self.assertEqual(submit[3]['max_tokens'], 512)
        self.assertEqual(submit[1], self.session.local_owner())
        self.ui.local_next_poll = 0
        self.ui.pump_local()
        self.settle_local()
        self.ui.pump()
        events = self.session.store.events(self.session.sid)
        self.assertEqual(len([event for event in events if event['kind'] == 'local_result']), 1)
        self.assertFalse(any(event['kind'] in ('user', 'finished') for event in events))
        self.assertFalse(self.session.agent.active)
        self.assertIsNone(self.session.agent.rpc)
        self.assertEqual(self.audio.calls, [])

    def test_local_async_completion_keeps_original_conversation(self):
        self.ui.local = LocalFixture()
        original = self.session.sid
        self.ui.handle('/local ask First conversation')
        self.session.new('.')
        self.settle_local()
        self.assertTrue(any(event['kind'] == 'local_job' for event in self.session.store.events(original)))
        self.assertEqual(self.session.store.events(self.session.sid), [])

    def test_worker_read_shows_full_saved_result_beyond_preview_limit(self):
        self.ui.local = LocalFixture()
        content = 'saved result line\n' * 500 + 'FINAL SAVED SENTINEL'
        full_result = dict(self.ui.local.job, state='completed', result=content,
                           result_bytes=len(content.encode()), result_truncated=False)
        with patch.object(self.ui.local, 'read', return_value=full_result) as read:
            self.ui.handle('/workers read 12345678')
            self.settle_local()
            self.assertEqual(read.call_args.kwargs, {'wait': 0, 'limit': 65536})
        self.assertIn(content, self.output)
        self.assertFalse(any('display limit' in line for line in self.output))
        self.assertIsNone(self.session.agent.rpc)

    def test_worker_read_distinguishes_source_limit_from_display_truncation(self):
        self.ui.local = LocalFixture()
        limited = dict(self.ui.local.job, state='completed', result='Saved incomplete answer',
                       result_truncated=True, finish_reason='length')
        with patch.object(self.ui.local, 'read', return_value=limited):
            self.ui.handle('/workers read 12345678')
            self.settle_local()
        self.assertIn('Saved incomplete answer', self.output)
        self.assertIn('Worker output reached a token or storage limit; the complete saved portion is shown.', self.output)
        self.assertFalse(any('display limit' in line for line in self.output))

    def test_workers_cancel_all_cancels_only_current_conversation(self):
        self.ui.local = LocalFixture()
        owner = self.session.local_owner()
        self.ui.handle('/workers cancel all')
        self.settle_local()
        self.assertIn(('cancel', owner, None), self.ui.local.calls)
        self.assertIsNone(self.session.agent.rpc)
        self.assertEqual(self.audio.calls, [])

    def test_local_starting_status_is_loading(self):
        self.ui.local = LocalFixture()
        self.ui.local_status = {'running': False, 'ready': False, 'starting': True}
        self.assertIn('Local loading', self.ui.local_summary())
        self.assertNotIn('stopped', self.ui.local_summary())

    def test_local_preview_points_to_worker_full_result(self):
        self.session.store.event(self.session.sid, 'local_result', {
            'id': '12345678abcdef', 'state': 'completed', 'result': 'part ' * 1000,
            'result_truncated': True, 'result_bytes': 10000})
        self.ui.pump()
        output = '\n'.join(self.output)
        self.assertIn('/workers read 12345678', output)
        self.assertNotIn('Full result: /logs', output)

    def assistant_command(self, line):
        self.ui.handle(line)
        self.settle_local()
        self.ui.pump()

    def test_named_assistant_controls_are_local_scoped_and_preserve_task_text(self):
        self.ui.local = AssistantFixture()
        owner = self.session.local_owner()
        self.assistant_command('/assistant create "code helper" reviewer')
        task = 'Return JSON {"answer": 42} and keep `code` unchanged.'
        self.assistant_command('/assistant ask "code helper" ' + task)
        self.assistant_command('/assistant guide "code helper" Use only the supplied facts.')
        self.assistant_command('/assistant pause "code helper"')
        self.assistant_command('/assistant resume "code helper"')
        self.assistant_command('/assistant read "code helper"')
        self.assistant_command('/assistant cancel "code helper"')
        self.assistant_command('/assistants')
        self.assertIn(('assistant_create', owner, 'code helper', 'reviewer'), self.ui.local.calls)
        self.assertIn(('assistant_assign', owner, 'code helper', task, 2, 512), self.ui.local.calls)
        self.assertIn(('assistant_guide', owner, 'code helper', 'Use only the supplied facts.'), self.ui.local.calls)
        for operation in ('pause', 'resume', 'read', 'cancel'):
            self.assertIn(('assistant_' + operation, owner, 'code helper'), self.ui.local.calls)
        self.assertIsNone(self.session.agent.rpc)
        self.assertFalse(self.session.agent.active)
        self.assertEqual(self.audio.calls, [])
        self.assertTrue(any('Pausing after the current step' in line for line in self.output))

    def test_assistant_completion_is_saved_and_rendered_once_across_ui_reopen(self):
        fixture = AssistantFixture()
        self.ui.local = fixture
        owner = self.session.local_owner()
        fixture.assistant_create(owner, 'helper')
        fixture.assistant_assign(owner, 'helper', 'A local task')
        item = fixture.assistants[(owner, 'helper')]
        item['state'] = 'completed'
        item['assignment'].update(state='completed', completed_steps=2, latest_result='Unique local result.',
                                  steps=[{'ordinal': 2, 'state': 'completed', 'job_id': '12345678abcdef12345678abcdef123456'}])
        for _ in range(2):
            self.ui.local_action('poll', {})
            self.settle_local()
            self.ui.pump()
        self.assertEqual(self.output.count('Unique local result.'), 1)
        old_sid = self.session.sid
        reopened = TerminalUI(self.session, self.audio, self.output.append, audio=False)
        reopened.local = fixture
        reopened.restore()
        self.ui = reopened
        self.ui.local_action('poll', {})
        self.settle_local()
        self.ui.pump()
        self.assertEqual(self.output.count('Unique local result.'), 1)
        self.assertEqual(len([event for event in self.session.store.events(old_sid) if event['kind'] == 'local_assistant']), 1)
        self.assertIsNone(self.session.agent.rpc)

    def test_async_assistant_reply_never_appears_in_switched_conversation(self):
        fixture = AssistantFixture()
        self.ui.local = fixture
        sid, owner = self.session.sid, self.session.local_owner()
        fixture.assistant_create(owner, 'private-helper')
        fixture.assistant_assign(owner, 'private-helper', 'Original conversation only')
        self.ui.handle('/assistant read private-helper')
        self.session.new('.')
        self.output.clear()
        self.settle_local()
        self.ui.pump()
        self.assertNotIn('Original conversation only', '\n'.join(self.output))
        self.assertTrue(any(event['kind'] == 'local_assistant_detail' for event in self.session.store.events(sid)))
        self.assertFalse(any(event['kind'].startswith('local_assistant') for event in self.session.store.events(self.session.sid)))
        self.assertIn(('assistant_read', owner, 'private-helper'), fixture.calls)

    def test_queued_assistant_control_retains_original_owner_after_switch(self):
        fixture = AssistantFixture()
        self.ui.local = fixture
        sid, owner = self.session.sid, self.session.local_owner()
        self.ui.local_busy = True
        self.ui.handle('/assistant create queued-helper')
        self.session.new('.')
        self.ui.local_busy = False
        self.ui.pump_local()
        self.settle_local()
        self.assertIn(('assistant_create', owner, 'queued-helper', 'assistant'), fixture.calls)
        self.assertTrue(any(event['kind'] == 'notice' for event in self.session.store.events(sid)))
        self.assertNotIn((self.session.local_owner(), 'queued-helper'), fixture.assistants)

    def test_status_exposes_resident_service_phase_without_starting_it(self):
        self.ui.local_status = {'running': False, 'ready': False, 'service_enabled': True,
                               'supervised': True, 'service_phase': 'backoff', 'restart_count': 2}
        self.ui.handle('/status')
        self.assertTrue(any('Resident service: enabled · backoff · restarts 2' in line for line in self.output))
        self.assertIsNone(self.session.agent.rpc)


class NPUControlTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        base = Path(self.directory.name)
        self.root = base / 'workspace'
        self.root.mkdir()
        self.session = TerminalSession(self.root, base / 'state', FakeRPC)
        self.output, self.audio = [], Audio()
        self.ui = TerminalUI(self.session, self.audio, self.output.append)
        self.ui.local = None
        self.ui.npu = Mock()
        self.ui.npu_next_poll = float('inf')
        self.ui.companion_next_poll = float('inf')
        self.ui.npu.status.return_value = self.status()
        self.ui.npu.start.return_value = self.status(ready=False, service_phase='starting')
        self.ui.npu.stop.return_value = self.status(ready=False, running=False, service_enabled=False,
                                                  service_phase='disabled')

    def tearDown(self):
        self.session.close()
        self.directory.cleanup()

    @staticmethod
    def status(**fields):
        value = {'service_enabled': True, 'service_phase': 'ready', 'restart_count': 2,
                 'running': True, 'ready': True, 'active': 1, 'queued': 0,
                 'model': {'name': 'Google Gemma 4 E2B Tensor G5', 'backend': 'litert-npu'}}
        value.update(fields)
        return value

    def settle(self):
        eventually(lambda: not self.ui.npu_busy)
        self.ui.pump_npu()

    def test_explicit_status_reports_actual_model_without_starting_or_cloud_calls(self):
        self.ui.handle('/npu')
        self.settle()
        self.ui.npu.status.assert_called_once_with()
        self.ui.npu.start.assert_not_called()
        self.ui.npu.stop.assert_not_called()
        output = '\n'.join(self.output)
        for text in ('automatic on · NPU ready', 'Google Gemma 4 E2B Tensor G5',
                     'backend litert-npu', 'restarts 2', '@npu', '1 active / 0 queued'):
            self.assertIn(text, output)
        self.assertNotIn('Huihui', output)
        self.assertEqual(self.audio.calls, [])
        self.assertIsNone(self.session.agent.rpc)

    def test_on_off_are_serialized_without_blocking_input_or_changing_main_engine(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        agent = self.session.agent
        def start():
            entered.set()
            self.assertTrue(release.wait(3))
            return self.status(ready=False, service_phase='starting')
        self.ui.npu.start.side_effect = start
        self.ui.handle('/npu on')
        self.assertTrue(entered.wait(1))
        self.ui.handle('/npu off')
        self.ui.handle('/help all')
        self.assertIn('/npu [status|on|off]', self.output[-1])
        self.ui.npu.stop.assert_not_called()
        release.set()
        self.settle()
        self.assertEqual([call[0] for call in self.ui.npu.method_calls], ['start', 'stop'])
        self.assertIn('automatic off', '\n'.join(self.output))
        self.assertIs(self.session.agent, agent)
        self.assertEqual(self.audio.calls, [])
        self.assertIsNone(agent.rpc)

    def test_controls_and_status_use_cached_state_and_show_native_ostadix_action(self):
        self.ui.npu_status = self.status()
        self.ui.handle('/controls')
        output = '\n'.join(self.output)
        self.assertIn('NPU helper: automatic on · NPU ready', output)
        self.assertIn('/npu on|off · @npu', output)
        self.assertIn('OSTADIX-BM embedded', output)
        self.assertIn('/o SOURCE', output)
        self.ui.npu.assert_not_called()
        self.assertEqual(self.ui.npu.method_calls, [])
        self.output.clear()
        self.ui.handle('/status')
        self.assertIn('NPU helper: automatic on · NPU ready', '\n'.join(self.output))
        self.assertEqual(self.ui.npu.method_calls, [])

    def test_polling_is_read_only_quiet_and_preserves_explicit_disable(self):
        self.ui.npu.status.return_value = self.status(ready=False, running=False, service_enabled=False)
        self.ui.npu_next_poll = 0
        self.ui.pump_npu()
        eventually(lambda: not self.ui.npu_busy)
        self.ui.pump_npu()
        self.assertEqual(self.output, [])
        self.assertIn('automatic off', self.ui.npu_summary())
        self.ui.npu.status.assert_called_once_with()
        self.ui.npu.start.assert_not_called()
        self.ui.npu.stop.assert_not_called()

    def test_unverified_backend_and_service_errors_do_not_claim_npu_readiness(self):
        self.ui.npu_status = self.status(model={'name': 'CPU model', 'backend': 'llama-cpu'})
        self.assertIn('backend unverified', self.ui.npu_summary())
        self.assertNotIn('NPU ready', self.ui.npu_summary())
        self.ui.npu.status.side_effect = RuntimeError('Actual service unavailable')
        self.ui.handle('/npu status')
        self.settle()
        self.assertIn('Actual service unavailable', '\n'.join(self.output))
        self.assertIn('/npu on', '\n'.join(self.output))
        self.assertIn('unavailable', self.ui.npu_summary())

    def test_invalid_busy_and_shutdown_controls_never_start_a_service(self):
        for command in ('/npu start', '/npu on extra', '/npu ask hello', '/npu status extra'):
            with self.subTest(command=command), self.assertRaisesRegex(AppError, 'Use /npu'):
                self.ui.handle(command)
        self.ui.npu_busy = True
        for _ in range(16):
            self.ui.npu_controls.put_nowait('status')
        with self.assertRaisesRegex(AppError, 'busy'):
            self.ui.handle('/npu on')
        self.ui.quit = True
        with self.assertRaisesRegex(AppError, 'closing'):
            self.ui.handle('/npu on')
        self.ui.npu_events.put(('status', self.status(), None))
        self.ui.pump_npu()
        self.assertEqual(self.output, [])
        self.assertEqual(self.ui.npu.method_calls, [])


class EmbeddedOstadixTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.base = Path(self.directory.name)
        self.root = self.base / 'workspace'
        self.root.mkdir()
        self.session = TerminalSession(self.root, self.base / 'state', FakeRPC)
        self.output, self.clients = [], []
        self.audio = Audio()
        self.ui = TerminalUI(self.session, self.audio, self.output.append)
        self.ui.local = None
        self.ui.companion_next_poll = float('inf')
        self.ui.restore()
        self.pair = {'root': '/private/fixture/repo', 'revision': 'a' * 40,
                     'backends': '/private/fixture/repo/backends', 'binary': '/private/fixture/mcp',
                     'binaries': {name: '/private/fixture/bin/' + name for name in
                                  ('O', 'o-cli', 'octl', 'o-info', 'olangc', 'o-live-host', 'ogit',
                                   'o-node', 'o-registry', 'ostadix-device')}}
        def factory(*args):
            client = OstadixClient(*args)
            self.clients.append(client)
            return client
        self.factory = factory
        self.session.agent.ostadix = OstadixTools(lambda: self.pair, self.factory)

    def tearDown(self):
        self.session.close()
        self.directory.cleanup()

    def settle(self):
        eventually(lambda: not self.session.agent.active)
        eventually(lambda: not self.session.agent.tool_lock.locked())
        self.ui.pump()

    def test_inline_source_runs_locally_and_preserves_exact_code(self):
        source = 'python^("quoted $source /quit 🦙", 1 + 1)_python'
        self.ui.handle('/o ' + source)
        self.settle()
        call = self.clients[0].calls[-1]
        self.assertEqual(call, ('tools/call', {'name': 'o_execute', 'arguments': {
            'source': source, 'cwd': str(self.root)}}))
        self.assertIsNone(self.session.agent.rpc)
        self.assertFalse(self.ui.quit)
        self.assertEqual(self.audio.calls, [])
        self.assertIn('actual fixture output', '\n'.join(self.output))
        events = self.session.store.events(self.session.sid)
        self.assertTrue(any(event['kind'] == 'approval_decision' and event['data']['automatic'] for event in events))
        result = next(event for event in events if event['kind'] == 'tool_result')
        self.assertEqual(result['data']['result']['result'], self.clients[0].result)

    def test_inline_execution_with_local_chat_does_not_submit_a_model_task(self):
        from local_chat import LocalChatAgent
        previous, client = self.session.agent, Mock()
        bridge = previous.ostadix
        # Keep the fixture bridge while closing the unused cloud agent.
        previous.ostadix = OstadixTools()
        previous.close()
        self.session.agent = LocalChatAgent(self.session.store, self.root, client=client)
        self.session.agent.ostadix = bridge
        self.session.agent.interface.attach(self.ui.voice_control)
        self.ui.handle('/o python^(2)_python')
        self.settle()
        self.assertEqual(self.clients[0].calls[-1][1]['name'], 'o_execute')
        self.assertTrue(all(call[0] == 'status' for call in client.method_calls))
        client.submit.assert_not_called()
        self.assertIsNone(self.session.agent.rpc)

    def test_inline_execution_respects_individual_approval_and_denial(self):
        self.session.agent.set_action_policy('ask')
        self.ui.handle('/o python^(1 + 1)_python')
        eventually(lambda: bool(self.session.approvals()))
        approval = self.session.approvals()[0]
        self.assertEqual([method for method, _ in self.clients[0].calls], ['tools/list'])
        self.ui.handle('/help')
        self.assertIn('/o SOURCE', self.output[-1])
        self.ui.handle('/deny ' + approval['id'][:8])
        self.settle()
        self.assertEqual([method for method, _ in self.clients[0].calls], ['tools/list'])
        self.assertIn('Action denied', '\n'.join(self.output))
        self.assertIsNone(self.session.agent.rpc)

    def test_cancel_releases_a_pending_approval_without_executing(self):
        self.session.agent.set_action_policy('ask')
        self.ui.handle('/o python^(1 + 1)_python')
        eventually(lambda: bool(self.session.approvals()))
        self.ui.handle('/cancel')
        self.settle()
        self.assertEqual(self.session.store.session(self.session.sid)['status'], 'interrupted')
        self.assertEqual([method for method, _ in self.clients[0].calls], ['tools/list'])
        self.assertIn('OSTADIX cancelled', '\n'.join(self.output))

    def test_slow_connection_does_not_block_controls_or_allow_a_second_task(self):
        entered, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def slow_factory(*args):
            entered.set()
            if not release.wait(3):
                raise AssertionError('fixture was not released')
            return self.factory(*args)
        self.session.agent.ostadix = OstadixTools(lambda: self.pair, slow_factory)
        self.ui.handle('/ostadix')
        self.assertTrue(entered.wait(1))
        self.ui.handle('/controls')
        with self.assertRaisesRegex(AppError, 'current task'):
            self.ui.handle('/o python^(2)_python')
        with self.assertRaisesRegex(AppError, 'current task'):
            self.ui.handle('/engine local')
        with self.assertRaises(AppError):
            self.session.new('.')
        self.assertFalse(self.ui.quit)
        release.set()
        self.settle()
        self.assertIn('OSTADIX connected · aaaaaaaaaaaa · 5 tools', '\n'.join(self.output))
        self.assertEqual([method for method, _ in self.clients[0].calls], ['tools/list'])

    def test_real_mcp_failure_remains_a_failure_with_its_actual_output(self):
        self.session.agent.ostadix.status(self.session.sid, self.root)
        self.clients[0].result = {'isError': True, 'structuredContent': {
            'state': 'failed', 'exit_code': 17, 'stderr': 'actual syntax error'}}
        self.ui.handle('/o broken source')
        self.settle()
        self.assertEqual(self.session.store.session(self.session.sid)['status'], 'failed')
        self.assertIn('actual syntax error', '\n'.join(self.output))
        self.assertIn('OSTADIX failed', '\n'.join(self.output))
        self.assertEqual(self.audio.calls, [])

    def test_native_result_preview_shows_values_and_keeps_full_diagnostics_in_logs(self):
        self.session.agent.ostadix.status(self.session.sid, self.root)
        native = {'ok': True, 'type': 'map', 'value': {'t': 'map', 'v': {
            'answer': {'t': 'number', 'v': {'kind': 'int', 'v': '63'}},
            'seconds': {'t': 'number', 'v': {'kind': 'binary_float', 'format': 'f64',
                                           'bits': [64, 91, 25, 22, 135, 43, 2, 12]}},
            'text': {'t': 'text', 'v': {'utf8': 'actual output'}},
            'ok': {'t': 'bool', 'v': True}}}, 'admission': 'diagnostic hash ' * 2000}
        response = {'structuredContent': {'state': 'completed', 'exit_code': 0,
                                         'result': native, 'metadata': 'transport metadata ' * 2000}}
        self.clients[0].result = response
        self.output.clear()
        self.ui.handle('/o python^(7 * 9)_python')
        self.settle()
        preview = next(text for text in self.output if text.startswith('OSTADIX completed'))
        self.assertIn('"answer":63', preview)
        self.assertIn('"seconds":108.392', preview)
        self.assertIn('actual output', preview)
        self.assertIn('"ok":true', preview)
        self.assertNotIn('admission', preview)
        self.assertNotIn('transport metadata', preview)
        self.assertLessEqual(len(preview), 600)
        saved = [event for event in self.session.store.events(self.session.sid) if event['kind'] == 'tool_result'][-1]
        self.assertEqual(saved['data']['result']['result'], response)

    def test_native_text_and_scalar_preview_are_bounded_without_changing_the_result(self):
        self.session.agent.ostadix.status(self.session.sid, self.root)
        for tag, value, expected in (('number', {'kind': 'int', 'v': '63'}, '\n63'),
                                     ('text', {'utf8': ''}, '\n""'),
                                     ('text', {'utf8': 'line\n' * 200}, 'line')):
            with self.subTest(tag=tag):
                response = {'structuredContent': {'state': 'completed', 'exit_code': 0,
                                                  'result': {'ok': True, 'value': {'t': tag, 'v': value}}}}
                self.clients[0].result = response
                self.output.clear()
                self.ui.handle('/o source')
                self.settle()
                preview = next(text for text in self.output if text.startswith('OSTADIX completed'))
                self.assertIn(expected, preview)
                self.assertLessEqual(len(preview), 600)
                self.assertLessEqual(len(preview.splitlines()), 6)
                if tag == 'text' and len(value['utf8']) > 540:
                    self.assertIn('/logs', preview)
                saved = [event for event in self.session.store.events(self.session.sid) if event['kind'] == 'tool_result'][-1]
                self.assertEqual(saved['data']['result']['result'], response)

    def test_pending_native_call_shows_running_job_id_instead_of_completion(self):
        self.session.agent.ostadix.status(self.session.sid, self.root)
        self.clients[0].result = {'structuredContent': {'state': 'running', 'job_id': 'job-actual-123'}}
        self.ui.handle('/o source')
        self.settle()
        preview = next(text for text in self.output if text.startswith('OSTADIX running'))
        self.assertIn('job-actual-123', preview)
        self.assertNotIn('completed', preview)
        self.assertIn('/logs', preview)

    def test_native_error_is_visible_even_when_the_transport_completed(self):
        self.session.store.event(self.session.sid, 'tool_result', {'success': True, 'result': {
            'mcp_tool': 'o_execute', 'exit_code': 0, 'result': {'structuredContent': {
                'state': 'completed', 'result': {'ok': False, 'error': {'message': 'Actual backend error'}}}}}})
        self.ui.pump()
        preview = next(text for text in self.output if text.startswith('OSTADIX failed'))
        self.assertIn('Actual backend error', preview)
        self.assertNotIn('OSTADIX completed', preview)

    def test_shutdown_waits_for_local_execution_before_closing_store(self):
        self.session.agent.ostadix.status(self.session.sid, self.root)
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)
        def blocked():
            entered.set()
            self.assertTrue(release.wait(5))
        self.clients[0].hook = blocked
        self.ui.handle('/o python^(2)_python')
        self.assertTrue(entered.wait(1))
        errors = []
        def close():
            try:
                self.session.close()
            except Exception as error:
                errors.append(error)
            finally:
                closed.set()
        closer = threading.Thread(target=close)
        closer.start()
        self.assertFalse(closed.wait(.1))
        release.set()
        closer.join(timeout=4)
        self.assertTrue(closed.is_set())
        self.assertEqual(errors, [])

    def test_invalid_controls_and_start_failure_never_submit_execution(self):
        for command in ('/o', '/ostadix unknown', '/controls extra', '/o ' + 'x' * 24001):
            with self.subTest(command=command[:40]), self.assertRaises(AppError):
                self.ui.handle(command)
        with patch('terminal.threading.Thread.start', side_effect=RuntimeError('thread unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'thread unavailable'):
                self.ui.handle('/o python^(2)_python')
        self.assertFalse(self.session.agent.active)
        self.assertEqual(self.clients, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
