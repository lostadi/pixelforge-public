#!/usr/bin/env python3
"""Terminal interface to the same persistent agent; no HTTP server or browser."""
import argparse
import codecs
import copy
import errno
import fcntl
import io
import json
import math
import os
from pathlib import Path
import queue
import re
import select
import shutil
import shlex
import signal
import struct
import subprocess
import sys
import threading
import termios
import time
import unicodedata
import uuid

from agent import AppError, Store, Workspace, clean
from speech_text import speech_chunks

DEFAULT_STATE = Path.home() / '.local/state/ostadix-voice'


def display_text(value):
    # Workspace/model output must not inject terminal escapes, OSC links or bidi.
    return ''.join(c for c in str(clean(value))
                   if c in '\n\t' or not unicodedata.category(c).startswith('C'))


def spoken_text(text):
    """Prepare the complete answer for speech, removing Markdown decoration."""
    text = re.sub(r'(?m)^[ \t]*(?:`{3,}|~{3,})[^\n]*(?:\n|$)', '', text)
    text = re.sub(r'`+([^`]+)`+', r'\1', text)
    text = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'(?m)^[ \t]*(?:#{1,6}\s+|>\s*|[-*+]\s+)', '', text)
    text = re.sub(r'(\*\*|__)(.*?)\1', r'\2', text, flags=re.DOTALL)
    return display_text(text).strip() or 'See the saved result in the terminal.'


def cell_width(text):
    return sum(0 if unicodedata.combining(c) else
               2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1 for c in text)


def excerpt(value, limit=700, lines=8, more_hint='… Full result: /logs'):
    text = display_text(value)
    short = '\n'.join(text.splitlines()[:lines])[:limit]
    return short + ('\n' + more_hint if short != text else '')


def ostadix_value(value, depth=0):
    """Readable previews of native OValues; preserve unknown wire values."""
    if depth > 6 or not isinstance(value, dict) or set(value) != {'t', 'v'}:
        return value
    tag, data = value['t'], value['v']
    if tag == 'text' and isinstance(data, dict) and isinstance(data.get('utf8'), str):
        return data['utf8']
    if tag == 'number' and isinstance(data, dict):
        if data.get('kind') == 'int' and isinstance(data.get('v'), (str, int)):
            number = str(data['v'])
            return int(number) if len(number) <= 256 and re.fullmatch(r'-?[0-9]+', number) else value
        if data.get('kind') == 'binary_float' and data.get('format') in ('f32', 'f64'):
            bits = data.get('bits')
            size, code = (4, '>f') if data['format'] == 'f32' else (8, '>d')
            if isinstance(bits, list) and len(bits) == size and all(type(x) is int and 0 <= x <= 255 for x in bits):
                number = struct.unpack(code, bytes(bits))[0]
                return number if math.isfinite(number) else str(number)
    if tag in ('bool', 'null'):
        return data
    if tag == 'map' and isinstance(data, dict):
        return {key: ostadix_value(item, depth + 1) for key, item in data.items()}
    if tag == 'list' and isinstance(data, list):
        return [ostadix_value(item, depth + 1) for item in data]
    return value


def ostadix_preview(result, success):
    """Keep actual outcomes visible while full MCP receipts remain in /logs."""
    response = result.get('result') or {}
    response = response if isinstance(response, dict) else {}
    structured = response.get('structuredContent') or {}
    structured = structured if isinstance(structured, dict) else {}
    tool = str(result.get('mcp_tool') or 'MCP')
    native = structured.get('result', structured)
    success = (success and not response.get('isError') and
               structured.get('state') not in ('failed', 'cancelled', 'timed_out') and
               not (isinstance(native, dict) and native.get('ok') is False))
    if success and result.get('execution_pending'):
        job = structured.get('job_id')
        return 'OSTADIX running · ' + tool + (' · job ' + str(job) if job else '') + '\nFull job receipt: /logs'
    header = 'OSTADIX ' + ('completed' if success else 'failed') + ' · ' + tool
    if success:
        if isinstance(native, dict) and 'value' in native:
            useful = ostadix_value(native['value'])
        elif isinstance(native, dict):
            useful = next((native[key] for key in ('stdout', 'output', 'text', 'message') if native.get(key)), native)
        else:
            useful = native
    else:
        useful = next((record[key] for record in (result, structured, native if isinstance(native, dict) else {})
                       for key in ('error', 'stderr', 'message') if record.get(key)), None)
        if isinstance(useful, dict) and useful.get('message'):
            useful = useful['message']
    if useful in (None, {}):
        useful = next((block['text'] for block in response.get('content', [])
                       if isinstance(block, dict) and block.get('type') == 'text' and block.get('text')),
                      'Full result: /logs')
    text = useful if isinstance(useful, str) and useful.strip() else json.dumps(useful, ensure_ascii=False, separators=(',', ':'))
    return header + '\n' + excerpt(text, 540, 4)


def optimizer_summary(action, result):
    """Phone-sized status and history; detailed journals stay in CLI JSON."""
    def record(value):
        return value if isinstance(value, dict) else {}

    def brief(value, limit=160):
        text = ' '.join(display_text(value).split())
        return text if len(text) <= limit else text[:limit - 1] + '…'

    def number(value, suffix='', divisor=1, precision=0):
        if type(value) not in (int, float) or not math.isfinite(value):
            return 'unknown'
        return f'{value / divisor:.{precision}f}' + suffix

    def phone(value):
        value = record(value)
        return ('battery ' + number(value.get('battery_percent'), '%') + ' · ' +
                number(value.get('battery_temp_c'), '°C', precision=1) + ' · ' +
                number(value.get('memory_available_mb'), ' GiB available', 1024, 1))

    def recommendation(value):
        value = record(value)
        return brief(value.get('action', 'pending'), 20) + (
            ' · ' + brief(value['reason']) if value.get('reason') else '')

    def priority(value):
        value = record(value)
        decision, outcome = record(value.get('decision')), record(value.get('result'))
        delta = decision.get('nice_delta')
        if type(delta) is not int or delta not in (0, 5, 10) or outcome.get('ok') is not True:
            return 'not confirmed'
        label = 'restored baseline' if delta == 0 else f'nice +{delta} from baseline'
        count = outcome.get('restored' if delta == 0 else 'applied')
        return label + (f' · {count} threads' if type(count) is int else '')

    result = record(result)
    if action == 'history':
        events = result.get('events')
        if not isinstance(events, list) or not events:
            return 'Optimizer history · no recorded events yet'
        lines = ['Optimizer history · newest first']
        for event in events[:10]:
            event = record(event)
            try:
                stamp = time.strftime('%H:%M:%S', time.localtime(event['utc_seconds']))
            except (KeyError, TypeError, ValueError, OverflowError, OSError):
                stamp = '--:--:--'
            kind = event.get('kind')
            if kind == 'observation':
                detail = phone(event.get('telemetry'))
            elif kind == 'model_analysis':
                detail = 'model ' + recommendation(event.get('recommendation'))
            elif kind == 'priority_action':
                detail = 'priority ' + priority(event)
            elif kind == 'analysis_submitted':
                detail = 'model analysis queued'
            else:
                detail = brief(kind or 'event', 40).replace('_', ' ')
                message = event.get('message') or event.get('reason') or record(event.get('result')).get('error')
                if message:
                    detail += ' · ' + brief(message)
            lines.append(stamp + '  ' + detail)
        return '\n'.join(lines)

    enabled, running = result.get('enabled'), result.get('running')
    state = ('running' if running and enabled else 'disabled, stopping' if running else 'starting' if result.get('starting') else
             'enabled, stopped' if enabled else 'off')
    phase = brief(result.get('phase') or '', 70)
    lines = ['Optimizer ' + action + ' · ' + state + ' · ' + brief(result.get('objective', 'unknown'), 20) +
             (' · ' + phase if phase and phase != 'stopped' else '')]
    backend = 'NPU' if result.get('model_backend') == 'litert-npu' else brief(result.get('model_backend', 'unknown'), 30)
    lines.append('Backend ' + backend + ' · samples ' + number(result.get('sample_seconds'), 's') +
                 ' · analysis ' + number(result.get('analysis_seconds'), 's') + ' when idle')
    if result.get('last_observation'):
        lines.append('Phone: ' + phone(result['last_observation']))
    analysis = record(result.get('last_model_analysis'))
    lines.append('Model: ' + (recommendation(analysis.get('recommendation')) if analysis else 'no assessment yet'))
    lines.append('Last change: ' + (priority(result['last_action']) if result.get('last_action') else 'none'))
    restoration = record(result.get('restoration'))
    if restoration:
        if restoration.get('ok') is True:
            count = restoration.get('restored')
            label = f'{count} threads restored' if type(count) is int else 'completed'
            skipped = restoration.get('skipped')
            if isinstance(skipped, list) and skipped:
                label += f' · {len(skipped)} skipped'
        else:
            label = 'incomplete · ' + brief(restoration.get('error') or 'check c --optimizer-status')
        lines.append('Restoration: ' + label)
    for key, label in (('error', 'Error'), ('guardian_error', 'Recovery error'), ('wake_release_error', 'Wake release error')):
        if result.get(key):
            lines.append(label + ': ' + brief(result[key], 240))
    return '\n'.join(lines)


def companion_summary(action, result):
    """Keep local check-in controls readable without exposing their job journal."""
    result = result if isinstance(result, dict) else {}
    message = result.get('message')
    if action == 'read':
        if isinstance(message, dict) and message.get('text'):
            return ('Local companion: ' + excerpt(message['text'], 1600, 16,
                    '… Full message: c --companion-read') +
                    '\nReply locally: /companion reply YOUR MESSAGE')
        return 'Local companion: no message yet.'
    if action == 'reply':
        return 'Local companion: reply queued on your phone. Its answer will appear here when you are idle.'
    if action == 'dismiss':
        return 'Local companion: check-in dismissed.'
    enabled, running = result.get('enabled'), result.get('running')
    label = 'off' if enabled is False else 'on' if enabled else 'unknown'
    lines = ['Local companion · ' + label + (' · running' if running else '')]
    snoozed = result.get('snoozed_until')
    if type(snoozed) in (int, float) and snoozed > time.time():
        lines.append('Snoozed until ' + time.strftime('%H:%M', time.localtime(snoozed)))
    elif enabled:
        lines.append('Quiet check-ins while you use the phone; /companion read shows the latest.')
    if result.get('phase'):
        lines.append('Status: ' + excerpt(result['phase'], 120, 1, ''))
    if result.get('error'):
        lines.append('Error: ' + excerpt(result['error'], 240, 2, ''))
    return '\n'.join(lines)


class TerminalPrompt:
    """Single-line editor that redraws typed input around asynchronous output.

    Own the input buffer instead of relying on terminal echo: model/audio output
    can then never erase a partially entered request. Android keyboard text and
    UTF-8 arriving across separate reads are kept intact. Bracketed paste is data
    until Enter; a pasted newline cannot accidentally approve or send a request.
    """
    def __init__(self, stream=None, input_stream=None):
        self.stream = stream or sys.stdout
        self.input = input_stream or sys.stdin
        self.text = ''
        self.position = 0
        self.decoder = codecs.getincrementaldecoder('utf-8')('replace')
        self.escape = ''
        self.pasting = False
        self.history = []
        self.history_position = 0
        self.history_draft = ''
        self.status = 'ready'
        self.activity = ''
        self.local = ''
        self.saved = None
        self.visible = False
        self.last_frame = None
        self.last_cr = False
        self.preview = ''
        self.transcript = ''
        self.rendered_preview = False
        self.rendered_rows = 0

    def __enter__(self):
        self.saved = termios.tcgetattr(self.input.fileno())
        settings = copy.deepcopy(self.saved)
        settings[3] &= ~(termios.ICANON | termios.ECHO)
        settings[6][termios.VMIN] = 1
        settings[6][termios.VTIME] = 0
        termios.tcsetattr(self.input.fileno(), termios.TCSANOW, settings)
        self.stream.write('\x1b[?2004h')
        self.visible = True
        self.render(force=True)
        return self

    def __exit__(self, *unused):
        self.visible = False
        # A shell has already restored its own terminal modes after suspending
        # this job. Background cleanup must not redraw its prompt, overwrite
        # those modes, or suspend again on SIGTTOU during tcsetattr.
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTTOU})
        try:
            fd = self.input.fileno()
            try:
                foreground = os.tcgetpgrp(fd) == os.getpgrp()
            except OSError as error:
                if error.errno == errno.ENOTTY:
                    # A standalone test PTY need not be our controlling TTY.
                    foreground = os.isatty(fd)
                elif error.errno in (errno.EIO, errno.EBADF):
                    foreground = False
                else:
                    raise
            if not foreground:
                return
            try:
                try:
                    self.erase()
                    self.stream.write('\x1b[?2004l')
                    self.stream.flush()
                finally:
                    if self.saved is not None:
                        # TCSANOW preserves any input still queued at shutdown.
                        termios.tcsetattr(fd, termios.TCSANOW, self.saved)
            except (OSError, termios.error) as error:
                if error.args[0] not in (errno.EIO, errno.ENOTTY, errno.EBADF):
                    raise
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def frame(self):
        columns = shutil.get_terminal_size((80, 24)).columns
        label = {'idle': 'You', 'ready': 'You', 'draft': 'Draft saved',
                 'thinking': 'Thinking', 'executing': 'Working',
                 'listening': 'Listening', 'speaking': 'Speaking',
                 'listening and speaking': 'Listening · speaking',
                 'starting microphone': 'Opening microphone',
                 'listening soon': 'Listening soon',
                 'stopping audio': 'Stopping audio', 'awaiting_approval': 'Approval needed'}.get(
                     self.status, 'Ready' if self.status.startswith('ready') else 'Check messages')
        label = self.activity or label
        badge = ' · ' + self.local if self.local and columns >= 32 else ''
        prefix = label + badge + ' › '
        if columns < 25:
            prefix = '› '
        maximum = max(1, columns - min(12, max(4, columns // 3)))
        if cell_width(prefix) > maximum:
            short = label + badge
            while short and cell_width(short + '… › ') > maximum:
                short = short[:-1]
            prefix = short + '… › '
        room = max(1, columns - cell_width(prefix) - 2)
        start = self.position
        left = room * 2 // 3 if self.position < len(self.text) else room
        while start and cell_width(self.text[start - 1:self.position]) <= left:
            start -= 1
        end = start
        while end < len(self.text) and cell_width(self.text[start:end + 1]) <= room:
            end += 1
        shown = self.text[start:end]
        cursor = cell_width(prefix + self.text[start:self.position])
        # Generated ANSI only. User/model data is sanitized before it arrives.
        return '\r\x1b[2K' + prefix + shown + '\r' + (f'\x1b[{cursor}C' if cursor else '')

    def render(self, force=False):
        if not self.visible:
            return
        frame = self.frame()
        rows = []
        for value in (self.preview, self.transcript):
            if not value:
                continue
            columns = max(1, shutil.get_terminal_size((80, 24)).columns - 1)
            preview = display_text(value).replace('\n', ' ').replace('\t', ' ')
            if cell_width(preview) > columns:
                while preview and cell_width('… ' + preview) > columns:
                    preview = preview[1:]
                preview = '… ' + preview
            rows.append(preview)
        frame = ''.join(row + '\r\n' for row in rows) + frame
        if force or frame != self.last_frame:
            self.erase()
            self.stream.write(frame)
            self.stream.flush()
            self.last_frame = frame
            self.rendered_preview = bool(self.preview)
            self.rendered_rows = len(rows)

    def erase(self):
        self.stream.write('\r\x1b[2K')
        for _ in range(self.rendered_rows):
            self.stream.write('\x1b[1A\r\x1b[2K')
        self.rendered_preview = False
        self.rendered_rows = 0

    def write(self, text):
        if self.visible:
            self.erase()
        self.stream.write(display_text(text) + '\n')
        self.stream.flush()
        self.render(force=True)

    def state(self, status, local=None, activity=''):
        self.status = status
        self.activity = display_text(activity)
        if local is not None:
            self.local = local
        self.render()

    def insert(self, value):
        self.text = self.text[:self.position] + value + self.text[self.position:]
        self.position += len(value)

    def recall(self, direction):
        if self.history_position == len(self.history):
            self.history_draft = self.text
        self.history_position = max(0, min(len(self.history), self.history_position + direction))
        self.text = (self.history[self.history_position] if self.history_position < len(self.history)
                     else self.history_draft)
        self.position = len(self.text)

    def sequence(self, value):
        if value == '\x1b[200~':
            self.pasting = True
        elif value == '\x1b[201~':
            self.pasting = False
        elif self.pasting:
            return
        elif value in ('\x1b[D', '\x1bOD'):
            self.position = max(0, self.position - 1)
        elif value in ('\x1b[C', '\x1bOC'):
            self.position = min(len(self.text), self.position + 1)
        elif value in ('\x1b[H', '\x1bOH', '\x1b[1~', '\x1b[7~'):
            self.position = 0
        elif value in ('\x1b[F', '\x1bOF', '\x1b[4~', '\x1b[8~'):
            self.position = len(self.text)
        elif value == '\x1b[3~':
            self.text = self.text[:self.position] + self.text[self.position + 1:]
        elif value in ('\x1b[A', '\x1bOA'):
            self.recall(-1)
        elif value in ('\x1b[B', '\x1bOB'):
            self.recall(1)

    def feed(self, chunk):
        lines = []
        for char in self.decoder.decode(chunk):
            if self.escape:
                self.escape += char
                if len(self.escape) == 2 and char not in '[O':
                    self.escape = ''
                    # A standalone Escape must not swallow the next typed letter.
                elif len(self.escape) >= 3 and ('@' <= char <= '~' or len(self.escape) > 32):
                    self.sequence(self.escape)
                    self.escape = ''
                    continue
                else:
                    continue
            if char == '\x1b':
                self.escape = char
            elif self.pasting:
                if char in '\r\n\t':
                    self.insert(' ')
                elif not unicodedata.category(char).startswith('C'):
                    self.insert(char)
            elif char in '\r\n':
                if char == '\n' and self.last_cr:
                    self.last_cr = False
                    continue
                self.last_cr = char == '\r'
                lines.append(self.text)
                if self.text and (not self.history or self.history[-1] != self.text):
                    self.history.append(self.text)
                    self.history = self.history[-100:]
                self.text = ''
                self.position = 0
                self.history_position = len(self.history)
                self.history_draft = ''
            else:
                self.last_cr = False
                if char in ('\x7f', '\b'):
                    if self.position:
                        self.text = self.text[:self.position - 1] + self.text[self.position:]
                        self.position -= 1
                elif char == '\x01':
                    self.position = 0
                elif char == '\x05':
                    self.position = len(self.text)
                elif char == '\x15':
                    self.text = self.text[self.position:]
                    self.position = 0
                elif char == '\x0b':
                    self.text = self.text[:self.position]
                elif char == '\x17':
                    before = self.text[:self.position]
                    trimmed = re.sub(r'\S+\s*$', '', before)
                    self.text = trimmed + self.text[self.position:]
                    self.position = len(trimmed)
                elif char == '\x04':
                    if not self.text:
                        lines.append(None)
                    else:
                        self.text = self.text[:self.position] + self.text[self.position + 1:]
                elif char == '\t':
                    self.insert('    ')
                elif not unicodedata.category(char).startswith('C'):
                    self.insert(char)
        self.render()
        return lines


class NativeSpeech:
    """Bounded native Android API calls; audio errors never become model input."""
    COMPONENTS = {
        'listening': 'com.termux.api/.apis.SpeechToTextAPI$SpeechToTextService',
        'speaking': 'com.termux.api/.apis.TextToSpeechAPI$TextToSpeechService',
    }

    def __init__(self):
        self.events = queue.Queue()
        self.lock = threading.RLock()
        self.process = None
        self.mode = None
        self.generation = 0
        self.closed = False
        self.stopping = False
        self.stop_thread = None
        self.cleanup_failed = False

    @staticmethod
    def kill(process):
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=.3)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=1)
            except ProcessLookupError:
                pass

    @staticmethod
    def response_error(output):
        # Termux:API can report a JSON error with exit status 0. Recognition
        # also emits partial transcripts, so inspect the entire response before
        # choosing its final line. Ordinary dictated words like "error" remain
        # valid text; only structured API output is rejected here.
        lines = output.splitlines()
        for index, line in enumerate(lines):
            if not line.lstrip().startswith(('{', '[')):
                continue
            try:
                payload, _ = json.JSONDecoder().raw_decode('\n'.join(lines[index:]).lstrip())
            except ValueError:
                return 'Android speech returned an incomplete service response.'
            if isinstance(payload, dict):
                detail = payload.get('error') or payload.get('errors') or payload.get('message')
                if detail:
                    return 'Android speech: ' + str(detail)[:600]
            return 'Android speech returned a service response instead of speech.'
        return None

    def start(self, mode, text=None):
        with self.lock:
            if self.closed or self.mode or self.stopping:
                raise AppError('Audio is busy. Use /stop, then try again.')
            name = 'termux-speech-to-text' if mode == 'listening' else 'termux-tts-speak'
            executable = shutil.which(name)
            if not executable:
                raise AppError(name + ' is missing. Install termux-api and the matching Termux:API Android app, or use keyboard dictation.')
            self.mode = mode
            self.generation += 1
            generation = self.generation
            # -p preserves API errors as well as partial/final recognition;
            # the default shell wrapper uses tail -1, losing JSON context.
            command = [executable] + (['-s', 'MUSIC'] if mode == 'speaking' else ['-p'])
            threading.Thread(target=self._run, args=(generation, mode, command, text), daemon=True).start()

    def _run(self, generation, mode, command, text):
        process = None
        completed_event = None
        response_complete = False
        try:
            # Keep playback busy until all chunks finish. Small utterances fit
            # Android's input limit and the existing per-command timeout.
            parts = (list(speech_chunks(text or '', limit=400)) or ['']) if mode == 'speaking' else [text]
            for part in parts:
                response_complete = False
                with self.lock:
                    if generation != self.generation:
                        return
                    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE, text=True, start_new_session=True)
                    self.process = process
                try:
                    out, err = process.communicate(input=(part or '') + '\n', timeout=40 if mode == 'listening' else 60)
                    response_complete = True
                except subprocess.TimeoutExpired:
                    # stop() owns bounded process/service cleanup. A second,
                    # unbounded communicate could hang on inherited pipe handles.
                    raise AppError('Android speech service timed out. Keep Termux foreground; check Termux:API permissions. Keyboard microphone dictation still works as text input.')
                with self.lock:
                    if generation != self.generation:
                        return
                if process.returncode:
                    raise AppError(f'Android speech command failed ({process.returncode}): ' + (err.strip() or out.strip())[:600])
                response_error = self.response_error(out) or self.response_error(err)
                if response_error:
                    raise AppError(response_error)
                if mode == 'listening':
                    lines = [line.strip() for line in out.splitlines() if line.strip()]
                    heard = lines[-1] if lines else ''
                    if re.fullmatch(r'ERROR_[A-Z_]+', heard):
                        raise AppError('Android speech: ' + heard)
                    if heard in ('}', ']'):
                        raise AppError('Android speech returned an incomplete service response.')
                    if not heard:
                        raise AppError('No speech was heard. Keep Termux foreground, allow microphone access for Termux:API, and pause other microphone apps or Adaptive Sound. Keyboard dictation also works.')
                    completed_event = ('heard', heard)
                if mode == 'speaking':
                    completed_process = process
                    with self.lock:
                        if generation == self.generation:
                            self.process = None
                    process = None
                    for stream in (completed_process.stdin, completed_process.stdout, completed_process.stderr):
                        if stream is not None:
                            stream.close()
        except (AppError, OSError) as error:
            with self.lock:
                if generation == self.generation:
                    message = str(error)
                    # Silence can be retried briefly in hands-free mode;
                    # permission, service and network failures cannot.
                    silent = mode == 'listening' and (
                        message.startswith('No speech was heard.') or
                        bool(re.search(r'\bERROR_(?:NO_MATCH|SPEECH_TIMEOUT)\b', message)))
                    if silent and response_complete and process.returncode == 0:
                        # Termux:API's SpeechToTextService is an IntentService:
                        # its silent result ends onHandleIntent and destroys the
                        # recognizer. As for successful dictation, a completed
                        # response needs no stopservice request. An unnecessary
                        # stop can fail and turn a retryable silence into a
                        # hands-free failure. Publish after releasing capture.
                        completed_event = ('empty', message)
                    else:
                        self.stop()
                        self.events.put(('playback_error' if mode == 'speaking' else
                                         'empty' if silent else 'error', message))
        finally:
            if process is not None:
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
            with self.lock:
                if generation == self.generation:
                    self.process = None
                    self.mode = None
                    # A fast model turn may follow immediately. Publish only
                    # after capture has released its mode and process, so the
                    # UI cannot mistake a completed capture for a live mic.
                    if completed_event:
                        self.events.put(completed_event)

    def stop(self):
        with self.lock:
            if self.stopping:
                return
            mode, process = self.mode, self.process
            self.generation += 1
            self.mode = None
            self.process = None
            self.stopping = bool(mode)
            if not mode:
                return
            self.cleanup_failed = False
            # Killing the CLI alone does not stop Android's service. Request
            # teardown of ONLY the speech service this interface was using.
            # Both steps run off the UI thread, including the bounded kill wait.
            def stop_service():
                try:
                    try:
                        self.kill(process)
                    except (OSError, subprocess.TimeoutExpired) as error:
                        self.events.put(('playback_error' if mode == 'speaking' else 'error',
                                         'Audio command cleanup: ' + str(error)))
                    am = shutil.which('am')
                    if not am:
                        raise AppError('Android service stop is unavailable')
                    result = subprocess.run([am, 'stopservice', '-n', self.COMPONENTS[mode]],
                                            capture_output=True, text=True, timeout=5)
                    message = result.stdout + result.stderr
                    if result.returncode or not re.search(r'Service (?:stopped\b|(?:not stopped: )?was not running\b)', message, re.I):
                        raise AppError('Android did not confirm the speech service stopped')
                except (AppError, OSError, subprocess.TimeoutExpired) as error:
                    self.cleanup_failed = True
                    self.events.put(('playback_error' if mode == 'speaking' else 'error',
                                     str(error) + '. Audio may finish its current utterance; agent task cancellation is separate.'))
                finally:
                    with self.lock:
                        self.stopping = False
            self.stop_thread = threading.Thread(target=stop_service, daemon=True)
            self.stop_thread.start()

    def close(self):
        with self.lock:
            self.closed = True
            self.stop()
            worker = self.stop_thread
        # Keep the session lock until its audio service is stopped, so a new c
        # session cannot race an old daemon worker tearing its audio down.
        if worker is not None and worker is not threading.current_thread():
            worker.join(timeout=6.5)


class TerminalSession:
    def __init__(self, root, state, rpc_factory=None):
        self.root = Path(root).resolve(strict=True)
        Workspace(self.root)
        state = Path(state).resolve()
        if state.is_relative_to(self.root):
            raise AppError('State must be outside the selected workspace')
        state.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock_file = (state / 'server.lock').open('a')
        try:
            fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock_file.close()
            raise AppError('Another interface is using this state directory. Run c to restart it here. Use the same --state-dir option if set. Your history is saved.')
        self.store = Store(state)
        from local_chat import create_agent
        self.agent = create_agent(self.store, self.root, rpc_factory=rpc_factory)
        sessions = self.sessions()
        selected = self.store.setting('selected')
        self.sid = next((s['id'] for s in sessions if s['id'] == selected),
                        sessions[0]['id'] if sessions else None) or self.store.create(self.root)
        self.store.setting('selected', self.sid)
        self.store.setting('launch_workspace', str(self.root))
        self._cancel_lock = threading.Lock()
        self._cancel_workers = {}
        self._close_lock = threading.Lock()
        self._closing = False
        self._closed = False
        self.voice_inbox = None

    def sessions(self):
        with self.store.lock:
            return [dict(r) for r in self.store.db.execute('SELECT * FROM sessions ORDER BY created DESC')
                    if Path(r['workspace']).resolve().is_relative_to(self.root)]

    def local_owner(self):
        from local_workers import owner_key
        return owner_key(self.store.directory, self.sid, self.store.session(self.sid)['workspace'])

    def approvals(self):
        with self.agent.lock:
            return [a.copy() for a in self.agent.approvals.values()
                    if a['session'] == self.sid and not a['event'].is_set()]

    def approve(self, prefix, allow):
        matches = [a for a in self.approvals() if len(prefix) >= 8 and a['id'].startswith(prefix)]
        if len(matches) != 1:
            raise AppError('Use the current approval ID shown above (at least 8 characters).')
        self.agent.approve(self.sid, matches[0]['id'], allow)

    def select(self, sid):
        with self.agent.lock:
            if self.agent.active or self.agent.tool_lock.locked():
                raise AppError('Wait for or cancel the task before switching conversations.')
            self.agent.workspace(sid)
            self.sid = sid
            self.store.setting('selected', sid)

    def new(self, relative):
        if self.agent.active or self.agent.tool_lock.locked():
            raise AppError('Wait for or cancel the task first.')
        Workspace(self.root).list(relative)
        self.select(self.store.create((self.root / relative).resolve()))

    def draft(self, text=None):
        return self.store.setting('terminal_draft_' + self.sid, clean(text) if text is not None else None) or ''

    def submit(self, text):
        self.draft(text)
        result = self.agent.submit(self.sid, text, uuid.uuid4().hex)
        self.draft('')
        return result

    def ostadix(self, source=None):
        """Run an explicit embedded control through the existing local executor.

        Reserving a normal turn keeps approval, cancellation and shutdown tied
        to this exact call. Neither discovery nor execution asks a model.
        """
        if source is not None and (not isinstance(source, str) or not source.strip() or
                                   len(source) > 24000 or '\x00' in source):
            raise AppError('Use /o SOURCE with up to 24,000 characters of complete OSTADIX code.')
        agent, sid = self.agent, self.sid
        with agent.lock:
            if agent.closed or self._closing:
                raise AppError('This interface is closing. Start c again to continue.')
            if agent.active or agent.tool_lock.locked():
                raise AppError('Wait for or /cancel the current task before using OSTADIX.')
            agent.workspace(sid)
            if agent.idle_timer:
                agent.idle_timer.cancel()
            # Detach idle cloud callbacks before this unrelated local turn.
            rpc, agent.rpc = agent.rpc, None
            agent.connection_generation += 1
            agent.loaded_thread = agent.loaded_action_policy = None
            agent.active, agent.sid = True, sid
            agent.cancelled.clear()
            agent.task_generation += 1
            generation = agent.task_generation
            agent.last_answer = ''
            agent.delta_buffers = {}
            agent.local.begin_turn(sid, generation)
            self.store.update(sid, status='connecting OSTADIX', turn_id=None)
            self.store.event(sid, 'notice', {'text': 'OSTADIX ' + ('connection check' if source is None else 'inline execution') + ' · local MCP, no model call.'})

            def work():
                # Own the tool lock for all I/O and persistence: close waits
                # here before closing SQLite, including after cancellation.
                with agent.tool_lock:
                    try:
                        if rpc:
                            rpc.close()
                        with agent.lock:
                            if agent.closed or not agent.active or generation != agent.task_generation:
                                return
                        argv = (['@ostadix', 'status'] if source is None else
                                ['@ostadix', 'call', 'o_execute', json.dumps({'source': source}, ensure_ascii=False)])
                        result = agent.execute({'operation': 'command', 'argv': argv, 'timeout': 120})
                        with agent.lock:
                            if agent.closed or not agent.active or generation != agent.task_generation:
                                return
                            if source is None:
                                self.store.event(sid, 'ostadix_status', result)
                            else:
                                self.store.event(sid, 'tool_result', {'success': result.get('exit_code', 0) == 0, 'result': result})
                            failed = result.get('exit_code', 0) != 0
                            cancelled = agent.cancelled.is_set()
                            agent.finish('interrupted' if cancelled else 'failed' if failed else 'idle',
                                         'OSTADIX cancelled. Review /logs for completed effects.' if cancelled else
                                         'OSTADIX execution failed. Check /logs.' if failed else '')
                    except Exception as error:
                        with agent.lock:
                            if not agent.closed and agent.active and generation == agent.task_generation:
                                cancelled = agent.cancelled.is_set()
                                agent.finish('interrupted' if cancelled else 'failed',
                                             'OSTADIX cancelled. Review /logs for completed effects.' if cancelled else str(error))

            agent.start_thread = threading.Thread(target=work, daemon=True)
            try:
                agent.start_thread.start()
            except Exception:
                agent.start_thread = None
                agent.finish('failed', 'OSTADIX could not start. No execution was submitted.')
                # No worker owns cleanup when thread startup itself fails.
                if rpc:
                    rpc.close()
                raise

    def cancel_background(self, events):
        # Capture the conversation before dispatch: the UI may switch while the
        # cancellation waits for a local worker or the Codex interrupt reply.
        with self._cancel_lock:
            if self._closing:
                raise AppError('This interface is closing. Start c again to continue.')
            sid = self.sid
            agent = self.agent
            if sid in self._cancel_workers:
                return

            def cancel():
                try:
                    agent.cancel(sid)
                except AppError as error:
                    events.put(('error', 'Task cancellation: ' + str(error)))
                finally:
                    with self._cancel_lock:
                        self._cancel_workers.pop(sid, None)

            worker = threading.Thread(target=cancel, daemon=True)
            self._cancel_workers[sid] = worker
            try:
                worker.start()
            except Exception:
                self._cancel_workers.pop(sid, None)
                raise

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            with self._cancel_lock:
                self._closing = True
                workers = list(self._cancel_workers.values())
            # Closing RPC first releases a pending interrupt without waiting
            # for its timeout. Never hold an agent/store/tool lock while joining:
            # cancellation still needs those locks for its final saved notices.
            if self.voice_inbox is not None:
                self.voice_inbox.stop()
            self.agent.close()
            if self.voice_inbox is not None:
                self.voice_inbox.close()
            for worker in workers:
                worker.join()
            # All cancellation and tool persistence must finish before SQLite
            # closes and before a replacement c can acquire the session lock.
            with self.agent.tool_lock:
                self.store.db.close()
            self.lock_file.close()
            self._closed = True


HELP = '''Type + Enter          Send a request (keyboard dictation also works)
Empty Enter           Start microphone, or send the recognized draft
/voice on|off|status   Continuous conversation; talk during spoken answers
/mic mute|unmute      Mute your microphone without muting AI speech
/hotkey on|off|status Volume Down then Up toggles your microphone
/talk                 Start native microphone input
/send   /discard      Send or discard the saved speech draft
/draft                Review saved speech without sending it
/review               Explain the saved-speech review counter
/apps [FILTER]        Show the complete installed package list
/cpu                  Read the phone's CPU and hardware directly
/speaker              Test Android text-to-speech
/hear                 Read the entire latest completed answer aloud
/mute   /unmute        Remember automatic spoken answers off / on
/audio                Audio status and Termux setup help
/stop                 Stop speech/listening; keep the agent task running
/cancel               Cancel main and local tasks in this conversation
/status  /help  /quit  Status / controls / save and exit
Ctrl-C                Stop audio; use /cancel for tasks, /quit to exit

More controls:
/approve ID /deny ID  Decide one displayed action when approve-all is off
/approvals on|off     Approve all actions, or require individual approval
/inspect FILE         Read a workspace file locally, with no model call
/controls             Current voice, speaker, engine and action controls
/engine gemini|local|codex|pixelforge   Choose the main chat engine while idle (saved)
/engine gemini MODEL        Choose a Gemini model explicitly
/engine pixelforge ID      Choose a registered compiled GGUF while idle
/models                   List registered PixelForge models (passive)
/npu [status|on|off]  Inspect or control the automatic NPU text helper
/o SOURCE             Run complete OSTADIX code locally; follows action policy
/ostadix              Check the embedded OSTADIX MCP connection and revision
/local [status]       Resident local model status (no model request)
/local start|stop     Start or stop the resident local model
/local ask REQUEST    Give the local assistant a bounded task
/agent ask REQUEST    NPU evidence workflow (requires an NPU model profile)
/agent list           Saved local evidence tasks in this workspace
/agent read|cancel ID Read a task or request cancellation
/optimizer [status]  Inspect the local optimizer and its managed settings
/optimizer on [balanced|performance|battery] Enable local monitoring
/optimizer off        Disable monitoring and restore its managed settings
/optimizer history    Show recent observations and actions
/companion [status]  Local check-ins while you use your phone
/companion on|off     Enable or disable proactive check-ins
/companion read       Read its latest message
/companion reply TEXT Reply to the local companion
/companion snooze [MINUTES] Pause check-ins (default: 60 minutes)
/companion dismiss    Close the current check-in
/assistants           List saved named assistants in this conversation
/assistant create NAME [ROLE] Create a persistent local assistant
/assistant ask NAME TASK Give it a bounded draft/review assignment
/assistant guide NAME TEXT Save guidance for its next step
/assistant read NAME  Read memory, result and saved worker references
/assistant pause|resume|cancel NAME Control its current assignment
/workers              List this conversation's local workers
/workers read ID      Read a local worker's saved result
/workers cancel [ID|all] Cancel one worker, or all in this conversation
/history   /logs       Show saved conversation / full tool results and logs
/sessions  /use NUMBER Switch a saved conversation
/new [subdirectory]   New conversation inside the selected workspace'''


QUICK_HELP = '''Type to chat. Voice mode hears you during replies.
/voice on|off   Continuous conversation on or off
/stop /cancel   Stop audio / cancel the current task
/hear /mute /unmute Speak the last answer / speaker off / on
/review /draft /send /discard Saved speech controls
/approvals on|off Approve all actions, or ask before each action
/o SOURCE /apps /cpu Embedded code / packages / hardware
/controls /status Current settings / full details
/help all       All controls, assistants and history
/quit           Save and close the conversation'''


class TerminalUI:
    def __init__(self, session, speech=None, output=print, audio=None, voice=False):
        self.session = session
        self.speech = speech or NativeSpeech()
        self.output = output
        self.audio = (session.store.setting('terminal_auto_audio') != 'off'
                      if audio is None else audio)
        self.capture_sid = None
        # c selects voice by default; c --text leaves capture off. No service
        # resumes microphone capture by itself after this UI exits.
        self.handsfree = False
        self.voice_sid = None
        self.voice_next_listen = 0
        self.voice_empty_attempts = 0
        self.voice_was_speaking = False
        self.start_voice_requested = voice
        self.microphone_muted = session.store.setting('terminal_microphone_muted') == 'on'
        if self.microphone_muted and hasattr(self.speech, 'set_microphone_muted'):
            self.speech.set_microphone_muted(True)
        self.volume_hotkey = None
        self.hotkey_enabled = session.store.setting('terminal_volume_hotkey') != 'off'
        self.inbox = None
        if getattr(self.speech, 'duplex', False):
            from voice_inbox import VoiceInbox
            self.inbox = session.voice_inbox = VoiceInbox(session)
        self.cursor = 0
        self.last_answer = ''
        self.last_status = None
        self.task_clock = None
        self.task_clock_generation = None
        self.captured_preview_id = None
        self.voice_queue_notice = False
        self.restored_failure = False
        self.quit = False
        self.pending_talk_sid = None
        self.pending_reply_audio = None
        self.talk_was_speaking = False
        self.talk_after = 0
        self.assistant_results_seen = set()
        self.agent_assistant_assignments = set()
        self.prompt = None
        self.shown_answer = ''
        self.local = None
        self.phone_events = queue.Queue()
        self.phone_cancel = threading.Event()
        self.phone_busy = False
        self.phone_worker = None
        self.local_events = queue.Queue()
        self.local_busy = False
        self.local_controls = []
        self.local_jobs = {}
        self.local_status = {'running': None, 'ready': False}
        self.local_next_poll = 0
        self.local_label = None
        self.engine_status = {}
        self.engine_status_key = None
        self.engine_events = queue.Queue()
        self.engine_poll = None
        self.engine_next_poll = 0
        self.engine_failure = None
        self.engine_outcome_after = 0
        self.npu = None
        self.npu_status = {'running': None, 'ready': False}
        self.npu_controls = queue.Queue(maxsize=16)
        self.npu_events = queue.Queue()
        self.npu_lock = threading.Lock()
        self.npu_busy = False
        self.npu_next_poll = 0
        self.optimizer = None
        self.optimizer_controls = queue.Queue(maxsize=16)
        self.optimizer_events = queue.Queue()
        self.optimizer_lock = threading.Lock()
        self.optimizer_busy = False
        self.companion = None
        self.companion_controls = queue.Queue(maxsize=16)
        self.companion_events = queue.Queue()
        self.companion_lock = threading.Lock()
        self.companion_busy = False
        self.companion_next_poll = 0
        self.companion_pending = []
        self.companion_pending_until = 0
        try:
            seen = json.loads(session.store.setting('terminal_companion_seen') or '[]')
            self.companion_seen = [item for item in seen if isinstance(item, str)][-64:] if isinstance(seen, list) else []
        except (TypeError, ValueError):
            self.companion_seen = []
        self.stream_text = ''
        self.stream_item = None
        self.session.agent.interface.attach(self.voice_control)
        try:
            from local_runtime import LocalClient
            self.local = LocalClient()
        except ImportError:
            pass
        self.publish_interface_context()

    def say(self, text):
        self.output(display_text(text))

    def restore(self):
        self.pending_talk_sid = None
        self.pending_reply_audio = None
        self.last_answer = ''
        self.shown_answer = ''
        session = self.session.store.session(self.session.sid)
        self.restored_failure = (session['status'] == 'failed'
                                 and not self.session.agent.active
                                 and self.session.agent.rpc is None)
        with self.session.store.lock:
            rows = self.session.store.db.execute('SELECT * FROM events WHERE session=? ORDER BY seq DESC LIMIT 20', (self.session.sid,)).fetchall()
        self.cursor = rows[0]['seq'] if rows else 0
        for row in rows:
            data = json.loads(row['data'])
            if row['kind'] == 'finished':
                successful = data.get('status') in ('idle', 'completed')
                self.last_answer = data.get('text', '') if successful else ''
                label = ('Previous task failed (saved history): ' if data.get('status') == 'failed'
                         else 'Last saved outcome (history): ')
                if not successful:
                    self.say(label + excerpt(data.get('text', ''), 260, 3))
                break
        self.say('Engine: ' + self.engine_label() + ' · /controls')
        self.say('Files: ' + Path(session['workspace']).name + ' · /status for details')
        if self.restored_failure:
            self.say(self.engine_label() + ' connection is checked when you send a new request.')
        if self.session.draft():
            self.say('Saved draft: ' + self.session.draft() + '\nEnter to send; /discard to discard.')
        if self.inbox is not None and self.inbox.review_text():
            count = self.inbox.counts()['review']
            self.say(str(count) + ' saved speech segment(s) need review · /draft to read them. '
                     '/send explicitly retries; /discard clears them. Nothing was replayed.')

    def local_summary(self):
        data = self.local_status
        if not self.local:
            return 'Local not installed yet'
        if data.get('error'):
            return 'Local unavailable · /local status'
        model = data.get('model') or {}
        model_name = (model.get('name', 'local model') + ' ' + model.get('quantization', '')).strip() if isinstance(model, dict) else str(model)
        if isinstance(model, dict) and model.get('backend') == 'litert-npu':
            model_name += ' · Tensor G5 NPU'
        if data.get('ready'):
            return 'Local ready · ' + model_name + f" · {data.get('active', 0)} active / {data.get('queued', 0)} queued"
        if data.get('running') or data.get('starting'):
            return 'Local loading · ' + (model_name or 'llama.cpp')
        if data.get('service_enabled') and data.get('supervised'):
            return 'Local service · ' + str(data.get('service_phase') or 'starting')
        if data.get('running') is None:
            return 'Local checking…'
        return 'Local stopped · /local start'

    def local_badge(self):
        data = self.local_status
        if data.get('error'):
            return 'CPU helper unavailable'
        if data.get('ready'):
            model = data.get('model') or {}
            return 'NPU helper ready' if isinstance(model, dict) and model.get('backend') == 'litert-npu' else 'CPU helper ready'
        if data.get('running') or data.get('starting'):
            return 'CPU helper loading'
        return ''

    def service_summary(self):
        data = self.local_status
        if 'service_enabled' not in data:
            return
        enabled = bool(data['service_enabled'])
        self.say('Resident service: ' + ('enabled' if enabled else 'disabled') +
                 ' · ' + str(data.get('service_phase') or ('running' if data.get('supervised') else 'stopped')) +
                 ' · restarts ' + str(data.get('restart_count', 0)))
        if data.get('service_error'):
            self.say('Service: ' + str(data['service_error']))

    def engine_label(self):
        return str(getattr(self.session.agent, 'engine_label', 'Codex'))

    def engine_key(self):
        agent = self.session.agent
        return agent, getattr(agent, 'local_client', None)

    def observe_engine(self, agent, client, status):
        if (agent, client) != self.engine_key() or not isinstance(status, dict):
            return
        self.engine_status_key = (agent, client)
        self.engine_status = dict(status)

    def engine_health(self):
        key = self.engine_key()
        if self.engine_status_key != key:
            self.engine_status_key, self.engine_status = key, {}
        data = self.engine_status
        if data.get('recovering'):
            return 'recovering', str(data.get('error') or 'Waiting for the selected model')
        if data and not data.get('ready'):
            return 'unavailable', str(data.get('error') or data.get('service_error') or 'The selected model is not ready')
        if self.session.agent.active:
            return 'working', ''
        if self.engine_failure and self.engine_failure[:3] == (self.session.sid, *key):
            return 'last request failed', self.engine_failure[3]
        if data.get('ready'):
            # A credential file confirms configuration, never cloud reachability
            # or remaining quota. Local status checks the actual model process.
            if (getattr(key[1], 'engine_name', None) == 'gemini' or
                    getattr(key[1], 'engine_name', None) == 'pixelforge' and not data.get('verified_worker_ready')):
                return 'configured; connection checked on request', ''
            return 'ready', ''
        return 'unchecked; checked on request', ''

    def engine_summary(self):
        state, error = self.engine_health()
        return ('Engine: ' + self.engine_label() + ' · ' + state +
                (': ' + excerpt(error, 180, 1, '') if error else ''))

    def pump_engine(self):
        while not self.engine_events.empty():
            agent, client, status = self.engine_events.get_nowait()
            self.observe_engine(agent, client, status)
        key = self.engine_key()
        client = key[1]
        if (self.quit or client is None or self.engine_poll == key
                or time.monotonic() < self.engine_next_poll and self.engine_status_key == key):
            return
        self.engine_poll = key
        self.engine_next_poll = time.monotonic() + 2

        def work():
            try:
                status = client.status()  # Passive, bounded; never starts a service.
            except Exception as error:
                status = {'ready': False, 'error': str(error)}
            self.engine_events.put((*key, status))
            if self.engine_poll == key:
                self.engine_poll = None

        threading.Thread(target=work, daemon=True, name='selected-engine-status').start()

    def npu_summary(self):
        data = self.npu_status
        if data.get('service_enabled') is False:
            phase = 'automatic off'
        elif data.get('error') or data.get('service_error'):
            phase = 'unavailable'
        elif data.get('ready'):
            model = data.get('model') or {}
            model = model if isinstance(model, dict) else {}
            automatic = 'automatic on' if data.get('service_enabled') is True else 'automatic unknown'
            phase = (automatic + ' · NPU ready' if model.get('backend') == 'litert-npu' else
                     'backend unverified')
        elif data.get('service_enabled') is True:
            phase = 'automatic on · ' + str(data.get('service_phase') or 'starting')
        else:
            phase = 'checking'
        return 'NPU helper: ' + phase + ' · /npu on|off · @npu'

    def ostadix_summary(self):
        bridge = self.session.agent.ostadix
        with bridge.lock:
            entry = bridge.connections.get(self.session.sid)
            if entry and entry['client'].alive:
                return 'OSTADIX-BM connected · ' + entry['pair']['revision'][:12] + ' · /o SOURCE'
        return 'OSTADIX-BM embedded · /ostadix to connect · /o SOURCE to run'

    def controls(self):
        """A passive, phone-sized panel: rendering it never starts devices."""
        self.say(self.engine_summary() + ' · /engine gemini|local|codex|pixelforge')
        self.say('Voice: ' + ('on' if self.handsfree else 'off') +
                 (' · mic muted' if self.microphone_muted else '') + ' · /voice on|off · /mic mute|unmute')
        self.say('Speaker: ' + ('on' if self.audio else 'off') + ' · /mute /unmute · /hear')
        self.say('Approve all: ' + ('on' if self.session.agent.action_policy() == 'auto' else 'off') + ' · /approvals on|off')
        self.say('CPU helper: ' + self.local_summary())
        self.say(self.npu_summary())
        self.say(self.ostadix_summary())
        self.say('/cancel task · /help controls · /quit save and exit')

    def speech_counts(self):
        return self.inbox.counts() if self.inbox is not None else dict(queued=0, sending=0, review=0)

    def publish_interface_context(self):
        """Give the main agent the same passive state shown by this interface."""
        context = self.voice_status()
        context.pop('capabilities', None)
        state, error = self.engine_health()
        context.update(available=not self.quit, interface='terminal', engine_label=self.engine_label(),
                       engine_health={'state': state, 'error': error},
                       workspace_name=Path(self.session.store.session(self.session.sid)['workspace']).name,
                       task_active=bool(self.session.agent.active),
                       approval_pending=bool(self.session.approvals()),
                       commands={'review': '/review', 'read_saved': '/draft', 'retry_saved': '/send',
                                 'discard_saved': '/discard', 'mute_mic': '/mic mute',
                                 'unmute_mic': '/mic unmute', 'mute_speaker': '/mute',
                                 'unmute_speaker': '/unmute'})
        self.session.agent.interface.update_context(self.session.sid, context)

    def explain_review(self):
        count = self.speech_counts()['review']
        self.say('The review number counts saved microphone phrases waiting for your review. '
                 'You currently have ' + str(count) + '. Their earlier delivery was not confirmed, '
                 'so they were kept without automatically replaying them. '
                 '/draft reads them; /send explicitly retries them; /discard clears them.')

    def activity_badge(self, counts):
        parts = []
        pending = counts['queued'] + counts['sending']
        if pending:
            parts.append(str(pending) + ' queued')
        if counts['review']:
            parts.append(str(counts['review']) + ' review')
        if self.handsfree:
            caps = getattr(self.speech, 'capabilities', {})
            if self.microphone_muted:
                confirmed = getattr(self.speech, 'microphone_muted_confirmed', False)
                parts.append('mic muted' if confirmed else 'mic muting')
            elif caps.get('recognition_state') == 'recovering' and not caps.get('recognition_silence'):
                parts.append('mic retrying')
            elif getattr(self.speech, 'starting', False):
                parts.append('mic starting')
            else:
                parts.append('mic on' if getattr(self.speech, 'listening', self.speech.mode == 'listening') else 'mic off')
        if not self.audio:
            parts.append('muted')
        engine_state, _ = self.engine_health()
        if self.engine_key()[1] is not None or engine_state == 'last request failed':
            parts.append('engine ' + {'last request failed': 'failed'}.get(engine_state, engine_state.split(';')[0]))
        if self.local_badge():
            parts.append(self.local_badge())
        return ' · '.join(parts)

    def show_models(self):
        from pixelforge_chat import registered_models
        from local_runtime import LocalError
        try:
            catalog = registered_models()
        except LocalError as error:
            raise AppError(str(error)) from None
        self.say('PixelForge models · /engine pixelforge ID')
        for ident, model in catalog['models'].items():
            selected = '* ' if ident == catalog['selected_model_id'] else '  '
            self.say(selected + ident + ' · ' + model['model_name'] + ' · context ' +
                     str(model['context_per_worker']) + ' · output ' + str(model['max_output_tokens']))

    def select_engine(self, name):
        parts = name.split()
        model = None
        pixelforge_model = None
        if len(parts) == 2 and parts[0] == 'gemini':
            from gemini_client import MODELS
            if parts[1] not in MODELS:
                raise AppError('Gemini models: ' + ', '.join(MODELS))
            name, model = 'gemini', parts[1]
        elif len(parts) == 2 and parts[0] == 'pixelforge':
            from pixelforge_chat import registered_models
            from local_runtime import LocalError
            try:
                catalog = registered_models()
            except LocalError as error:
                raise AppError(str(error)) from None
            if parts[1] not in catalog['models']:
                raise AppError('Unknown PixelForge model ID; use /models')
            name, pixelforge_model = 'pixelforge', parts[1]
        if name not in ('local', 'codex', 'gemini', 'pixelforge'):
            raise AppError('Use /engine gemini, /engine local, /engine codex, /engine pixelforge, or /engine for the current engine.')
        previous = self.session.agent
        with previous.lock:
            if previous.active or previous.tool_lock.locked():
                raise AppError('Wait for or /cancel the current task before switching engines.')
            if self.session._closing:
                raise AppError('This interface is closing. Start c again to continue.')
            previous.closed = True
        previous.close()
        if pixelforge_model is not None:
            from pixelforge_chat import select_model
            try:
                select_model(pixelforge_model)
            except Exception as error:
                # Selection may fail or have an uncertain outcome. Keep the saved
                # engine choice and restore an idle client without replaying it.
                from local_chat import create_agent
                self.session.agent = create_agent(self.session.store, self.session.root)
                self.session.agent.interface.attach(self.voice_control)
                from local_runtime import LocalError
                if isinstance(error, LocalError):
                    raise AppError(str(error)) from None
                raise
        if model is not None:
            self.session.store.setting('gemini_model', model)
        self.session.store.setting('chat_engine', name)
        from local_chat import create_agent
        self.session.agent = create_agent(self.session.store, self.session.root)
        self.session.agent.interface.attach(self.voice_control)
        with self.session.store.lock:
            self.engine_outcome_after = self.session.store.db.execute(
                'SELECT coalesce(max(seq),0) FROM events').fetchone()[0]
        self.engine_failure = None
        self.engine_next_poll = 0
        self.say('Engine: ' + self.engine_label() + ' · preference saved.')

    @staticmethod
    def assistant_label(item):
        assignment = item.get('assignment') or {}
        state = item.get('state', 'waiting')
        progress = (f" · {assignment.get('completed_steps', 0)}/{assignment.get('max_steps', 0)} steps"
                    if assignment else '')
        return str(item.get('name') or item['id'][:8]) + ' · ' + state + progress

    def assistant_update(self, sid, owner, item, notify=True):
        if item.get('owner') != owner:
            return
        assignment = item.get('assignment') or {}
        signature = json.dumps([assignment.get('id'), item.get('state'), assignment.get('completed_steps')])
        key = 'terminal_assistant_seen_' + sid + '_' + item['id']
        if self.session.store.setting(key) == signature:
            return
        self.session.store.setting(key, signature)
        if not assignment or not notify:
            return
        self.session.store.event(sid, 'local_assistant', {
            'id': item['id'], 'name': item.get('name'), 'state': item.get('state'),
            'assignment': assignment, 'label': self.assistant_label(item)})

    def show_assistant(self, item):
        self.say('Assistant ' + self.assistant_label(item) + ' [' + item['id'][:8] + ']')
        self.say('Role: ' + str(item.get('role', 'assistant')))
        self.say('Guidance: ' + (item.get('guidance') or 'None'))
        self.say('Saved memory: ' + (item.get('memory') or 'None'))
        assignment = item.get('assignment') or {}
        if assignment:
            self.say('Assignment: ' + str(assignment.get('task', '')))
            self.say('Budget: ' + str(assignment.get('max_steps', 0)) + ' steps × ' + str(assignment.get('max_tokens', 0)) + ' output tokens')
            if assignment.get('latest_result'):
                self.say('Latest saved result: ' + assignment['latest_result'])
            if assignment.get('latest_result_truncated'):
                self.say('Saved context is compact or source output was limited; worker references contain the full saved output.')
            if assignment.get('error'):
                self.say('Assignment: ' + str(assignment['error']))
            for step in assignment.get('steps', []):
                if step.get('job_id'):
                    self.say(f"Step {step['ordinal']} · {step.get('state', 'unknown')} · /workers read {step['job_id']}")
        if item.get('state') == 'pause_pending':
            self.say('Pause takes effect after the current step finishes. /assistant cancel stops the current assignment.')

    @staticmethod
    def assistant_argument(text):
        # Parse only the identity, retaining JSON quotes and code in the task.
        stream = io.StringIO(text)
        lexer = shlex.shlex(stream, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = ''
        name = lexer.get_token()
        return name or '', text[stream.tell():].strip()

    def local_action(self, action, argument='', scope=None):
        if not self.local:
            raise AppError('The local runtime is not installed yet. Check /status for the selected chat engine.')
        sid, owner = scope or (self.session.sid, self.session.local_owner())
        if self.local_busy:
            if action == 'poll':
                return
            if len(self.local_controls) >= 16:
                raise AppError('Local controls are busy. Wait for the pending requests to finish.')
            self.local_controls.append((action, argument, (sid, owner)))
            return
        self.local_busy = True
        def work():
            try:
                if action == 'status':
                    result = self.local.status()
                elif action == 'start':
                    result = self.local.start()
                elif action == 'stop':
                    result = self.local.stop()
                elif action == 'ask':
                    result = self.local.submit(owner, argument, max_tokens=512, role='assistant')
                elif action.startswith('evidence_'):
                    from local_agent import Agent
                    evidence = Agent()
                    workspace = str(self.session.agent.workspace(sid).root)
                    operation = action[len('evidence_'):]
                    if operation == 'ask':
                        result = evidence.start(workspace, argument)
                    elif operation == 'list':
                        result = [evidence.read(path.stem) for path in evidence.directory.glob('*.json')]
                        result = [item for item in result if item['workspace'] == workspace]
                    else:
                        result = evidence.read(argument)
                        if result['workspace'] != workspace:
                            raise AppError('Task belongs to a different workspace.')
                        if operation == 'cancel':
                            result = evidence.cancel(argument)
                elif action.startswith('assistant_'):
                    operation = action[len('assistant_'):]
                    if operation == 'list':
                        result = self.local.assistant_list(owner)
                    elif operation == 'create':
                        result = self.local.assistant_create(owner, argument['name'], argument['role'])
                    elif operation == 'ask':
                        result = self.local.assistant_assign(owner, argument['name'], argument['text'], max_steps=2, max_tokens=512)
                    elif operation == 'guide':
                        result = self.local.assistant_guide(owner, argument['name'], argument['text'])
                    else:
                        result = getattr(self.local, 'assistant_' + operation)(owner, argument['name'])
                elif action in ('list', 'read', 'cancel'):
                    jobs = self.local.jobs(owner)
                    if action == 'list':
                        result = jobs
                    elif action == 'cancel' and argument in ('', 'all'):
                        result = self.local.cancel(owner)
                    else:
                        matches = [job for job in jobs if len(argument) >= 8 and job['id'].startswith(argument)]
                        if len(matches) != 1:
                            raise AppError('Use a current worker ID from /workers (at least 8 characters).')
                        result = (self.local.read(owner, matches[0]['id'], wait=0, limit=65536)
                                  if action == 'read' else self.local.cancel(owner, matches[0]['id']))
                elif action == 'poll':
                    result = {'status': self.local.status(), 'jobs': [], 'evidence': []}
                    if getattr(self, 'evidence_tasks', None):
                        from local_agent import Agent
                        evidence = Agent()
                        for task_id, task_sid in list(self.evidence_tasks.items()):
                            try:
                                item = evidence.read(task_id)
                                result['evidence'].append((task_sid, item))
                            except Exception as error:
                                result['evidence'].append((task_sid, {'id': task_id, 'state': 'failed', 'error': str(error)}))
                    for job_id, job_owner in list(argument.items()):
                        result['jobs'].append(self.local.read(job_owner, job_id, wait=0, limit=4000))
                    if result['status'].get('running') and hasattr(self.local, 'assistant_list'):
                        try:
                            result['assistants'] = self.local.assistant_list(owner)
                        except Exception as error:
                            result['assistant_error'] = str(error)
                self.local_events.put((sid, owner, action, result, None))
            except Exception as error:
                self.local_events.put((sid, owner, action, None, str(error)))
        threading.Thread(target=work, daemon=True).start()

    def pump_local(self):
        while not self.local_events.empty():
            sid, owner, action, result, error = self.local_events.get_nowait()
            self.local_busy = False
            self.local_next_poll = time.monotonic() + 2
            if error:
                if action == 'poll':
                    self.local_status = {'running': False, 'ready': False, 'error': error}
                elif action.startswith('assistant_'):
                    self.session.store.event(sid, 'notice', {'text': 'Local assistant: ' + error})
                else:
                    self.say('Local: ' + error)
                continue
            if action == 'status':
                self.local_status = result
                self.local_label = self.local_summary()
                self.say(self.local_label)
                self.service_summary()
                if result.get('error'):
                    self.say('Local: ' + str(result['error']))
            elif action in ('start', 'stop'):
                self.say('Local ' + ('start requested. Loading continues in the background.' if action == 'start' else 'stop requested.'))
                if action == 'stop':
                    self.local_status = {'running': False, 'ready': False}
                elif isinstance(result, dict):
                    self.local_status = result
                self.local_next_poll = 0
            elif action == 'ask':
                self.local_jobs[result['id']] = (sid, owner)
                self.session.store.event(sid, 'local_job', {'id': result['id'], 'state': result.get('state', 'queued')})
            elif action.startswith('evidence_'):
                if action == 'evidence_ask':
                    if not hasattr(self, 'evidence_tasks'):
                        self.evidence_tasks = {}
                    self.evidence_tasks[result['id']] = sid
                    self.say('Local evidence task started: ' + result['id'])
                elif action == 'evidence_list':
                    for item in result:
                        self.say(item['id'] + '  ' + item['state'] + '  ' + item['task'][:80])
                    if not result:
                        self.say('No local evidence tasks in this workspace.')
                else:
                    self.say(result['id'] + '  ' + result['state'])
                    self.say(result.get('result') or result.get('error') or 'No answer yet.')
                    if result.get('proposal'):
                        self.say(json.dumps(result['proposal'], indent=2))
                        self.say('Review the exact proposal. Apply or deny it through local_agent.py with its digest.')
                    if action == 'evidence_cancel':
                        self.say('Cancellation requested; the task records completion when stopped.')
                    elif result.get('result') and sid == self.session.sid:
                        self.speak(result['result'])
            elif action == 'list':
                if not result:
                    self.say('No local workers in this conversation. /local ask REQUEST starts one.')
                for job in result:
                    self.say(f"{job['id'][:8]}  {job.get('state', job.get('status', 'unknown'))}  {job.get('role', 'assistant')}")
            elif action == 'read':
                self.say('Local ' + str(result.get('id', ''))[:8] + ' · ' + str(result.get('state', 'unknown')))
                self.say(result.get('result') or result.get('error') or 'No result yet.')
                if result.get('result_truncated') or result.get('truncated') or result.get('finish_reason') == 'length':
                    self.say('Worker output reached a token or storage limit; the complete saved portion is shown.')
            elif action == 'cancel':
                self.say('Local worker cancellation requested. The main task and audio are unchanged.')
            elif action.startswith('assistant_'):
                operation = action[len('assistant_'):]
                if operation == 'list':
                    scoped = [item for item in result if item.get('owner') == owner]
                    self.session.store.event(sid, 'local_assistant_list', {'assistants': scoped})
                elif result.get('owner') == owner:
                    if operation == 'read':
                        self.session.store.event(sid, 'local_assistant_detail', {'assistant': result})
                    elif operation in ('create', 'guide'):
                        label = ('Created assistant ' + str(result.get('name')) + '. Use /assistant ask NAME TASK.'
                                 if operation == 'create' else
                                 'Guidance saved for ' + str(result.get('name')) + '; it applies to the next step or assignment.')
                        self.session.store.event(sid, 'notice', {'text': label})
                    self.assistant_update(sid, owner, result, notify=operation != 'read')
            elif action == 'poll':
                self.local_status = result['status']
                for task_sid, item in result.get('evidence', []):
                    if item['state'] == 'awaiting_approval':
                        self.evidence_tasks.pop(item['id'], None)
                        self.session.store.event(task_sid, 'notice', {'text': 'Local task has a proposed edit or test. /agent read ' + item['id'] + ' shows the exact action; nothing has been applied.'})
                    elif item['state'] in ('completed', 'failed', 'cancelled', 'interrupted', 'budget_exhausted'):
                        self.evidence_tasks.pop(item['id'], None)
                        self.session.store.event(task_sid, 'local_result', item)
                for job in result['jobs']:
                    if job.get('state') in ('completed', 'failed', 'cancelled', 'interrupted'):
                        tracked = self.local_jobs.pop(job['id'], None)
                        if tracked:
                            self.session.store.event(tracked[0], 'local_result', job)
                for assistant in result.get('assistants', []):
                    self.assistant_update(sid, owner, assistant)
        if self.local:
            label = self.local_summary()
            if label != self.local_label:
                self.local_label = label
                if not self.prompt or self.local_status.get('error'):
                    self.say(label)
            if not self.local_busy and self.local_controls:
                self.local_action(*self.local_controls.pop(0))
            elif not self.local_busy and time.monotonic() >= self.local_next_poll:
                self.local_next_poll = time.monotonic() + 2
                self.local_action('poll', {key: value[1] for key, value in self.local_jobs.items()})

    def npu_action(self, action):
        """Control the independent helper without waiting on the UI thread."""
        with self.npu_lock:
            if self.quit:
                raise AppError('The terminal is closing; NPU control was not applied.')
            try:
                self.npu_controls.put_nowait(action)
            except queue.Full as error:
                raise AppError('NPU controls are busy. Wait for the pending requests to finish.') from error
            if self.npu_busy:
                return
            self.npu_busy = True

        def work():
            while True:
                with self.npu_lock:
                    try:
                        operation = self.npu_controls.get_nowait()
                    except queue.Empty:
                        self.npu_busy = False
                        return
                try:
                    if self.npu is None:
                        from npu_tools import NPUClient
                        self.npu = NPUClient()
                    if operation == 'on':
                        result = self.npu.start()
                    elif operation == 'off':
                        result = self.npu.stop()
                    else:
                        result = self.npu.status()
                    if not isinstance(result, dict):
                        raise AppError('The NPU service returned an invalid status.')
                    self.npu_events.put((operation, result, None))
                except Exception as error:
                    self.npu_events.put((operation, None, str(error)))

        try:
            threading.Thread(target=work, daemon=True).start()
        except Exception:
            with self.npu_lock:
                self.npu_busy = False
                while not self.npu_controls.empty():
                    self.npu_controls.get_nowait()
            raise

    def pump_npu(self):
        while not self.npu_events.empty():
            action, result, error = self.npu_events.get_nowait()
            if self.quit:
                continue
            if error:
                self.npu_status = dict(self.npu_status, error=error)
                if action != 'poll':
                    self.say('NPU helper: ' + error + ' Check /npu status, or /npu on to enable it.')
                continue
            self.npu_status = result
            if action == 'poll':
                continue
            self.say(self.npu_summary())
            model = result.get('model') or {}
            if isinstance(model, dict) and model:
                self.say('NPU model: ' + str(model.get('name') or model.get('model_name') or 'unknown') +
                         ' · backend ' + str(model.get('backend') or 'unverified'))
            if result.get('service_phase'):
                self.say('NPU service: ' + str(result['service_phase']) +
                         ' · restarts ' + str(result.get('restart_count', 0)))
            self.say('Text helpers via @npu · ' + str(result.get('active', 0)) + ' active / ' +
                     str(result.get('queued', 0)) + ' queued. The main chat engine is unchanged.')
            for key in ('error', 'service_error', 'guardian_error'):
                if result.get(key):
                    self.say('NPU: ' + str(result[key]) + ' · /npu status')
        if not self.quit and not self.npu_busy and time.monotonic() >= self.npu_next_poll:
            self.npu_next_poll = time.monotonic() + 5
            self.npu_action('poll')

    def optimizer_action(self, action, objective='balanced'):
        """Serialize explicit controls off the UI thread, independently of jobs."""
        with self.optimizer_lock:
            try:
                self.optimizer_controls.put_nowait((action, objective))
            except queue.Full as error:
                raise AppError('Optimizer controls are busy. Wait for the pending requests to finish.') from error
            if self.optimizer_busy:
                return
            self.optimizer_busy = True

        def work():
            while True:
                with self.optimizer_lock:
                    try:
                        operation, goal = self.optimizer_controls.get_nowait()
                    except queue.Empty:
                        self.optimizer_busy = False
                        return
                try:
                    if self.optimizer is None:
                        from optimizer import OptimizerControl
                        self.optimizer = OptimizerControl()
                    if operation == 'on':
                        result = self.optimizer.enable(objective=goal)
                    elif operation == 'off':
                        result = self.optimizer.disable()
                    elif operation == 'history':
                        result = self.optimizer.history(limit=10)
                    else:
                        result = self.optimizer.status()
                    self.optimizer_events.put((operation, result, None))
                except Exception as error:
                    self.optimizer_events.put((operation, None, str(error)))

        threading.Thread(target=work, daemon=True).start()

    def pump_optimizer(self):
        while not self.optimizer_events.empty():
            action, result, error = self.optimizer_events.get_nowait()
            if self.quit:
                continue
            if error:
                self.say('Optimizer: ' + error)
            else:
                self.say(optimizer_summary(action, result))

    def companion_action(self, action, argument=None):
        """Keep local control and read-only message polling off the UI thread."""
        with self.companion_lock:
            try:
                self.companion_controls.put_nowait((action, argument))
            except queue.Full as error:
                raise AppError('Companion controls are busy. Wait for the pending requests to finish.') from error
            if self.companion_busy:
                return
            self.companion_busy = True

        def work():
            while True:
                with self.companion_lock:
                    try:
                        operation, value = self.companion_controls.get_nowait()
                    except queue.Empty:
                        self.companion_busy = False
                        return
                try:
                    if self.companion is None:
                        from companion import CompanionControl
                        self.companion = CompanionControl()
                    if operation == 'on':
                        result = self.companion.enable()
                    elif operation == 'off':
                        result = self.companion.disable()
                    elif operation == 'snooze':
                        result = self.companion.snooze(minutes=value)
                    elif operation == 'reply':
                        result = self.companion.reply(value)
                    elif operation == 'poll':
                        result = self.companion.messages()
                    elif operation == 'read':
                        result = self.companion.read()
                    elif operation == 'dismiss':
                        result = self.companion.dismiss()
                    else:
                        result = self.companion.status()
                    self.companion_events.put((operation, result, None))
                except Exception as error:
                    self.companion_events.put((operation, None, str(error)))

        threading.Thread(target=work, daemon=True).start()

    def remember_companion_message(self, message):
        ident = message.get('id') if isinstance(message, dict) else None
        if isinstance(ident, str) and ident and ident not in self.companion_seen:
            self.companion_seen = (self.companion_seen + [ident])[-64:]
            self.session.store.setting('terminal_companion_seen', json.dumps(self.companion_seen))

    def companion_can_display(self):
        return not (self.quit or self.handsfree or self.companion_busy or self.voice_busy() or self.session.draft() or
                    self.speech.mode or self.speech.stopping or self.local_jobs or self.local_controls or
                    self.local_status.get('active') or self.local_status.get('queued') or
                    (self.prompt and (self.prompt.text or self.prompt.pasting)))

    def pump_companion(self):
        while not self.companion_events.empty():
            action, result, error = self.companion_events.get_nowait()
            if self.quit:
                continue
            if error:
                if action != 'poll':
                    self.say('Local companion: ' + error)
                else:
                    self.companion_pending = []
                continue
            if action == 'poll':
                self.companion_pending_until = time.monotonic() + 2
                self.companion_pending = ([item for item in result.get('messages', [])
                                           if isinstance(item, dict) and item.get('display_pending') is True
                                           and isinstance(item.get('id'), str) and item['id']
                                           and item.get('kind') in ('checkin', 'reply')
                                           and isinstance(item.get('text'), str) and item['text']]
                                          if isinstance(result, dict) and result.get('eligible') is True else [])
            else:
                if action == 'read':
                    self.remember_companion_message(result.get('message'))
                # Explicit actions can close or supersede a queued check-in.
                self.companion_pending = []
                self.companion_next_poll = 0
                self.say(companion_summary(action, result))
        if self.companion_can_display() and time.monotonic() < self.companion_pending_until:
            pending = [item for item in self.companion_pending if item['id'] not in self.companion_seen]
            if pending:
                message = pending[-1]
                self.remember_companion_message(message)
                self.say(companion_summary('read', {'message': message}))
        if not self.quit and not self.companion_busy and time.monotonic() >= self.companion_next_poll:
            self.companion_next_poll = time.monotonic() + 2
            self.companion_action('poll')

    def speak(self, text):
        if (self.handsfree and getattr(self.speech, 'duplex', False) and
                (self.speech.user_speaking or (self.inbox and self.inbox.delivering))):
            return  # A newer spoken request takes precedence over an old answer.
        self.speech.start('speaking', spoken_text(text))
        if self.handsfree:
            self.voice_was_speaking = True

    def speak_completed_reply(self, text):
        if (self.handsfree and getattr(self.speech, 'duplex', False) and
                self.inbox is not None and self.inbox.delivering and not self.speech.user_speaking):
            # A fast backend can complete before the inbox's delivery receipt
            # is drained. Keep this reply until that acknowledgement, but never
            # let it outlive its conversation, request generation or controls.
            self.pending_reply_audio = (self.session.sid, self.session.agent,
                                        self.session.agent.task_generation, text)
            return
        self.pending_reply_audio = None
        self.speak(text)

    def pump_reply_audio(self):
        pending = self.pending_reply_audio
        if pending is None:
            return
        sid, agent, generation, text = pending
        if (self.quit or not self.audio or not self.handsfree or self.speech.user_speaking or
                self.session.sid != sid or self.session.agent is not agent or
                agent.task_generation != generation or agent.cancelled.is_set() or text != self.last_answer):
            self.pending_reply_audio = None
            return
        if agent.active or self.inbox is not None and self.inbox.delivering:
            return
        self.pending_reply_audio = None
        self.speak(text)

    def show_apps(self, result):
        packages = result.get('packages', [])
        total = result.get('count', len(packages))
        offset = result.get('offset', 0)
        self.say('Installed packages · ' + str(total) + ' total' +
                 (' · showing ' + str(offset + 1) + '–' + str(offset + len(packages))
                  if result.get('truncated') and packages else ''))
        self.say('\n'.join(packages) if packages else 'No matching packages.')
        if result.get('more'):
            self.say('More packages remain. Next page: ' + json.dumps(result.get('next_argv')))

    def close_phone_queries(self):
        self.phone_cancel.set()
        if self.phone_worker is not None and self.phone_worker is not threading.current_thread():
            self.phone_worker.join(timeout=9)
        while not self.phone_events.empty():
            sid, success, result = self.phone_events.get_nowait()
            self.session.store.event(sid, 'tool_result', {'success': success, 'result': result})

    def set_voice(self, enabled, announce=True, preserve_mute=False):
        """UI-thread hook for explicit user voice-mode controls."""
        if not enabled:
            self.pending_reply_audio = None
        if enabled:
            if self.microphone_muted and not preserve_mute:
                self.set_microphone_muted(False, announce=False, start_voice=False)
            if self.session.draft() and not getattr(self.speech, 'duplex', False):
                raise AppError('A speech draft is saved. /send or /discard it before /voice on.')
            if not self.handsfree:
                self.pending_talk_sid = None
                self.handsfree = True
                self.voice_sid = self.session.sid
                self.voice_empty_attempts = 0
                self.voice_next_listen = time.monotonic()
                self.voice_was_speaking = self.speech.mode == 'speaking'
            if (self.microphone_muted and getattr(self.speech, 'duplex', False)
                    and hasattr(self.speech, 'set_microphone_muted')
                    and getattr(self.speech, 'bridge', None) is None
                    and not self.speech.mode and not self.speech.stopping):
                # A muted native connection still owns TTS. Initialize it with
                # capture disabled so later unmute never races fallback audio.
                self.speech.start('listening')
            if announce:
                if getattr(self.speech, 'duplex', False):
                    self.say('Voice on · live transcription · talk during replies. /stop ends voice.')
                else:
                    self.say('Hands-free on. Speak when the prompt says Listening; requests send automatically. '
                             'Say "stop listening", or use /stop or Ctrl-C to stop.')
            self.pump_voice()
        else:
            self.handsfree = False
            self.pending_talk_sid = None
            self.voice_sid = None
            self.voice_was_speaking = False
            self.speech.stop()
            self.discard_capture()
            if self.prompt:
                self.prompt.transcript = ''
            if announce:
                self.say('Hands-free off. Empty Enter still records one reviewed draft.')
        self.publish_interface_context()

    @property
    def voice_enabled(self):
        return self.handsfree

    def set_microphone_muted(self, muted, announce=True, start_voice=True):
        """Mute only capture; playback and the current agent turn keep running."""
        muted = bool(muted)
        if hasattr(self.speech, 'set_microphone_muted'):
            self.speech.set_microphone_muted(muted)
        elif muted and self.speech.mode == 'listening':
            self.speech.stop()
        self.microphone_muted = muted
        self.session.store.setting('terminal_microphone_muted', 'on' if muted else 'off')
        if muted:
            self.pending_talk_sid = None
            self.discard_capture()
            if self.prompt:
                self.prompt.transcript = ''
        elif start_voice and not self.handsfree:
            self.set_voice(True, announce=False)
        if announce:
            self.say('Microphone ' + ('mute' if muted else 'unmute') + ' requested. AI speech setting unchanged.')
        self.publish_interface_context()

    def toggle_microphone(self):
        # An explicit hardware gesture can resume a deliberately stopped mic;
        # it never changes the independent spoken-answer preference.
        capture_active = (self.handsfree or getattr(self.speech, 'listening', False)
                          or getattr(self.speech, 'starting', False) or self.speech.mode == 'listening'
                          or getattr(getattr(self.speech, 'fallback', None), 'mode', None) == 'listening')
        self.set_microphone_muted(not self.microphone_muted if capture_active else False)

    def start_volume_hotkey(self):
        if self.volume_hotkey is not None:
            return
        from volume_hotkey import VolumeHotkey
        self.volume_hotkey = VolumeHotkey()
        self.volume_hotkey.start()

    def close_volume_hotkey(self):
        if self.volume_hotkey is not None:
            self.volume_hotkey.close()
            self.volume_hotkey = None

    def pump_volume_hotkey(self):
        if self.volume_hotkey is None:
            return
        while not self.volume_hotkey.events.empty():
            kind, message = self.volume_hotkey.events.get_nowait()
            if kind == 'toggle' and not self.quit:
                try:
                    self.toggle_microphone()
                except (AppError, OSError) as error:
                    self.say('Microphone shortcut: ' + str(error))
            elif kind == 'error':
                self.say('Volume shortcut unavailable: ' + str(message) + ' Use /mic mute or /mic unmute.')

    def voice_status(self):
        return {'enabled': self.handsfree,
                'microphone_muted': self.microphone_muted,
                'microphone_mute_confirmed': getattr(self.speech, 'microphone_muted_confirmed', False),
                'audio': 'stopping' if self.speech.stopping else self.speech.mode or 'idle',
                'automatic_speech': self.audio,
                'draft_saved': bool(self.session.draft()),
                'speech_queue': self.speech_counts(),
                'saved_speech_review_meaning': 'Saved microphone phrases awaiting explicit review; not a model review.',
                'continuous_capture': bool(self.handsfree and getattr(self.speech, 'listening', False)),
                'duplex': getattr(self.speech, 'duplex', False),
                'capabilities': getattr(self.speech, 'capabilities', {}),
                'microphone_mute': ['@voice', 'mute'], 'microphone_unmute': ['@voice', 'unmute'],
                'activation': '/voice on', 'stop': '/stop'}

    def voice_control(self, action):
        if self.quit:
            raise AppError('The terminal is closing; voice control was not applied.')
        if action in ('on', 'off'):
            self.set_voice(action == 'on')
        elif action in ('mute', 'unmute'):
            self.set_microphone_muted(action == 'mute')
        return dict(self.voice_status(), available=True, interface='terminal',
                    task_active=bool(self.session.agent.active),
                    approval_pending=bool(self.session.approvals()))

    def discard_capture(self):
        # stop() invalidates NativeSpeech's generation before returning. Remove
        # any result already queued by that generation without hiding errors.
        kept = []
        while not self.speech.events.empty():
            event = self.speech.events.get_nowait()
            if event[0] not in ('heard', 'partial', 'speech_start'):
                kept.append(event)
        for event in kept:
            self.speech.events.put(event)
        self.capture_sid = None

    def voice_busy(self):
        return (bool(self.session.agent.active) or
                self.session.agent.tool_lock.locked() or
                bool(self.session.approvals()))

    def pump_voice(self):
        if not self.handsfree or self.quit:
            return
        if self.voice_sid != self.session.sid:
            self.set_voice(False, announce=False)
            self.say('Hands-free paused after changing conversations. /voice on to resume.')
            return
        if self.microphone_muted:
            return
        if getattr(self.speech, 'duplex', False):
            # The native recorder stays open while the agent works, TTS plays,
            # or the user types. Partial speech is independently redrawn.
            if not self.speech.mode and not self.speech.stopping:
                try:
                    self.capture_sid = self.session.sid
                    self.speech.start('listening')
                except (AppError, OSError) as error:
                    self.set_voice(False, announce=False)
                    self.say('Voice paused: ' + str(error) + ' /voice on to retry.')
            return
        if self.voice_was_speaking and self.speech.mode != 'speaking':
            # The TTS command completes after Android reports utterance done.
            # Let the speaker tail settle before opening the microphone.
            self.voice_was_speaking = False
            self.voice_next_listen = max(self.voice_next_listen, time.monotonic() + .65)
        typed = bool(self.prompt and (self.prompt.text or self.prompt.pasting))
        busy = self.voice_busy()
        if (typed or busy) and self.speech.mode == 'listening':
            self.speech.stop()
            self.discard_capture()
        if (typed or busy or self.session.draft() or self.speech.mode or
                self.speech.stopping or time.monotonic() < self.voice_next_listen):
            return
        try:
            self.speech.start('listening')
            self.capture_sid = self.session.sid
            self.pending_talk_sid = None
        except (AppError, OSError) as error:
            self.set_voice(False, announce=False)
            self.say('Hands-free paused: ' + str(error) + ' /voice on to retry.')

    def pump_pending_talk(self):
        if self.pending_talk_sid != self.session.sid or self.quit:
            self.pending_talk_sid = None
            return
        if self.speech.mode == 'speaking':
            self.talk_was_speaking = True
        elif self.talk_was_speaking:
            self.talk_was_speaking = False
            self.talk_after = time.monotonic() + .65
        if (self.voice_busy() or self.session.draft() or self.speech.mode or self.speech.stopping or
                time.monotonic() < self.talk_after or
                (self.prompt and (self.prompt.text or self.prompt.pasting))):
            return
        self.pending_talk_sid = None
        try:
            if hasattr(self.speech, 'start_once'):
                self.speech.start_once()
            else:
                self.speech.start('listening')
            self.capture_sid = self.session.sid
        except (AppError, OSError) as error:
            self.say('Microphone: ' + str(error) + ' Press Enter to retry.')

    def audio_status(self, help=False):
        mode = 'stopping' if self.speech.stopping else self.speech.mode or 'idle'
        self.say('Audio: ' + mode + ' · automatic speech ' + ('on' if self.audio else 'off') +
                 (' · microphone muted' if self.microphone_muted else ''))
        self.say('Hands-free: ' + ('on · requests send automatically' if self.handsfree else 'off · /voice on to start'))
        if getattr(self.speech, 'duplex', False):
            caps = self.speech.capabilities
            flag = lambda key: 'yes' if caps.get(key) is True else 'no' if caps.get(key) is False else 'not measured'
            self.say('Native voice · on-device recognition · continuous microphone with simultaneous playback')
            self.say('Mic active: ' + flag('mic_active') + ' · echo cancellation: ' + flag('aec_enabled') +
                     ' · capture silenced: ' + flag('mic_silenced'))
            self.say('Recognition: ' + str(caps.get('recognition_state') or 'not measured') +
                     ' · retries ' + str(caps.get('recognition_restarts', 0)) +
                     ' · dropped samples ' + str(caps.get('dropped_frames', 'not measured')))
            if help:
                self.say('c starts voice; c --text starts without recording. /voice on opens the mic; /stop closes it.')
                self.say('Your words appear live and send at speech pauses, including during a reply. /mute keeps listening without spoken replies.')
                self.say('Android’s microphone indicator and Ostadix Voice notification show capture. The notification’s Stop also closes it.')
            return
        missing = []
        for label, name in (('Microphone', 'termux-speech-to-text'),
                            ('Speaker', 'termux-tts-speak')):
            installed = bool(shutil.which(name))
            self.say(label + ': ' + ('command installed' if installed else 'command missing'))
            if not installed:
                missing.append(name)
        if missing:
            self.say('Install the termux-api package and the matching Termux:API Android app. Keyboard dictation also works.')
        if help:
            self.say('/voice on starts hands-free conversation; it speaks each reply and listens again. /voice off or /stop ends it. It never resumes automatically after restarting c.')
            self.say('Test microphone: empty Enter → speak → review draft → Enter to send. /discard clears it.')
            self.say('Test speaker: /speaker. /hear replays the entire latest answer. Adjust Android media volume.')
            self.say('/mute or /unmute saves automatic speech off or on. /stop or Ctrl-C stops audio.')
            self.say('Keep Termux foreground and allow Microphone for Termux:API in Android settings. Installed commands alone do not verify audio access.')
            self.say('If capture is silent, pause other microphone apps or Adaptive Sound, or use the keyboard microphone to dictate into this prompt.')

    def pump(self):
        self.pump_volume_hotkey()
        self.publish_interface_context()
        while not self.phone_events.empty():
            sid, success, result = self.phone_events.get_nowait()
            self.phone_busy = False
            self.session.store.event(sid, 'tool_result', {'success': success, 'result': result})
        self.session.agent.interface.pump(self.session.sid, self.session.agent.task_generation,
                                          closed=self.quit)
        self.pump_local()
        self.pump_npu()
        self.pump_engine()
        self.pump_optimizer()
        while not self.speech.events.empty():
            kind, text = self.speech.events.get_nowait()
            if kind in ('heard', 'partial', 'speech_start') and self.microphone_muted:
                continue
            if kind == 'partial':
                if self.prompt:
                    self.prompt.transcript = 'You · ' + text
            elif kind == 'speech_start':
                # Native adapter already stopped playback, retaining capture.
                self.pending_reply_audio = None
            elif kind == 'heard':
                # Recognition is data, never /approve, /quit or a shell command.
                sid = self.capture_sid or self.session.sid
                continuous = self.handsfree and getattr(self.speech, 'duplex', False)
                if not continuous:
                    self.capture_sid = None
                if self.prompt:
                    self.prompt.transcript = ''
                if self.handsfree and sid == self.voice_sid == self.session.sid and text.strip().lower().rstrip('.!?') in ('stop listening', 'stop voice mode'):
                    self.set_voice(False)
                    continue
                if continuous and sid == self.voice_sid == self.session.sid:
                    try:
                        self.captured_preview_id = self.inbox.add(text)
                        captured = excerpt(text, 200, 1, '')
                        if self.prompt:
                            self.prompt.transcript = 'Saved speech · ' + captured
                        elif self.session.agent.active:
                            self.say('Saved speech: ' + captured)
                    except (AppError, OSError, ValueError) as error:
                        # Keep the captured text even if delivery is unavailable.
                        saved = self.session.draft()
                        self.session.draft(saved + '\n' + text if saved else text)
                        self.set_voice(False, announce=False)
                        self.say('Voice paused: ' + str(error) + ' Speech saved; /send or /discard.')
                    continue
                self.session.store.setting('terminal_draft_' + sid, clean(text))
                if (self.handsfree and sid == self.voice_sid == self.session.sid and
                        not self.voice_busy() and not (self.prompt and (self.prompt.text or self.prompt.pasting))):
                    try:
                        self.session.submit(text)
                        self.voice_empty_attempts = 0
                    except (AppError, OSError, ValueError) as error:
                        self.set_voice(False, announce=False)
                        self.say('Hands-free paused: ' + str(error) + ' Your speech is saved as a draft; /send or /discard.')
                elif sid == self.session.sid:
                    self.say('Heard: ' + text + '\nEnter to send; /discard to discard.')
                else:
                    self.say('Speech draft saved in its original conversation [' + sid[:8] + ']. /sessions to return and review it.')
            elif kind == 'empty' and self.handsfree:
                self.capture_sid = None
                self.voice_empty_attempts += 1
                self.voice_next_listen = time.monotonic() + min(3, self.voice_empty_attempts * 1.5)
                if not self.prompt and self.voice_empty_attempts == 1:
                    self.say('No speech heard; listening again shortly. /stop ends hands-free mode.')
            elif kind == 'microphone_muted':
                if isinstance(text, dict) and text.get('muted') == self.microphone_muted:
                    self.say('Microphone muted.' if text['muted'] else 'Microphone on.')
            elif kind == 'playback_error':
                self.say('Audio: ' + text + ' Use /hear to replay.')
            else:
                self.pending_talk_sid = None
                if self.handsfree:
                    self.set_voice(False, announce=False)
                    self.say('Hands-free paused. /voice on to retry after checking /audio.')
                self.say('Audio: ' + text)
        for event in self.session.store.events(self.session.sid, self.cursor):
            self.cursor = event['seq']
            kind, data = event['kind'], event['data']
            if kind == 'user':
                self.pending_reply_audio = None
                self.shown_answer = ''
                self.stream_text = ''
                self.stream_item = None
                if self.prompt:
                    self.prompt.preview = ''
                self.say('You: ' + data.get('text', ''))
            elif kind == 'assistant_delta':
                if data.get('item_id') != self.stream_item:
                    self.stream_item = data.get('item_id')
                    self.stream_text = ''
                self.stream_text = (self.stream_text + data.get('text', ''))[-2000:]
                if self.prompt:
                    self.prompt.preview = 'Ostadix · ' + self.stream_text
            elif kind == 'assistant':
                if self.prompt:
                    self.prompt.preview = ''
                self.shown_answer = data.get('text', '')
                self.say('Ostadix: ' + self.shown_answer)
            elif kind == 'notice':
                notice = data.get('text', '')
                if notice.startswith('Follow-up speech'):
                    if not self.voice_queue_notice:
                        self.say('Speech saved for the next reply.')
                        self.voice_queue_notice = True
                else:
                    self.say(excerpt(notice))
            elif kind == 'log':
                # Full technical logs remain persisted and available via /logs.
                pass
            elif kind == 'approval':
                self.say('AWAITING APPROVAL\n' + data['warning'] + '\n' + json.dumps(data['arguments'], ensure_ascii=False, indent=2)
                         + '\n/approve ' + data['id'][:8] + '  or  /deny ' + data['id'][:8])
            elif kind == 'tool_result':
                result = data['result']
                if data['success'] and result.get('terminal_hardware_display'):
                    from hardware_info import format_cpu_info
                    self.last_answer = format_cpu_info(result)
                    self.say(self.last_answer)
                    continue
                if data['success'] and isinstance(result.get('packages'), list):
                    self.show_apps(result)
                    continue
                if result.get('mcp_tool'):
                    self.say(ostadix_preview(result, data['success']))
                    continue
                if self.prompt and data['success'] and not result.get('error') and (result.get('local') or not result):
                    continue  # Internal delegation receipts remain available in /logs.
                state = 'pending' if result.get('execution_pending') else 'completed' if data['success'] else 'failed'
                label = result.get('path') or result.get('tool') or result.get('operation') or ''
                self.say('Tool ' + state + (': ' + str(label) if label else ''))
                useful = next((result[key] for key in ('error', 'content', 'stdout', 'output', 'text', 'message') if result.get(key)), None)
                if useful is not None:
                    self.say(excerpt(useful))
                elif not data['success'] or result.get('execution_pending'):
                    self.say(excerpt(json.dumps(result, ensure_ascii=False)))
            elif kind == 'ostadix_status':
                self.say('OSTADIX connected · ' + str(data.get('revision', 'unknown'))[:12] +
                         ' · ' + str(data.get('tool_count', '?')) + ' tools · /o SOURCE')
            elif kind == 'local_job':
                self.say('Local worker ' + data['id'][:8] + ' queued · /workers')
            elif kind == 'local_result':
                state = data.get('state', data.get('status', 'completed'))
                self.say('Local ' + data.get('id', '')[:8] + ' · ' + state)
                hint = 'Full saved result: /workers read ' + data.get('id', '')[:8]
                self.say(excerpt(data.get('result') or data.get('error') or 'No text result.', 2400, 24, '… ' + hint))
                if data.get('result_truncated') or data.get('truncated') or data.get('finish_reason') == 'length':
                    self.say('Worker preview or source output was limited. ' + hint)
            elif kind == 'local_assistant_list':
                if not data['assistants']:
                    self.say('No named assistants yet. /assistant create helper starts an idle identity.')
                for assistant in data['assistants']:
                    self.say(self.assistant_label(assistant) + ' [' + assistant['id'][:8] + ']')
            elif kind == 'local_assistant_detail':
                self.show_assistant(data['assistant'])
            elif kind == 'local_assistant':
                assignment = data.get('assignment') or {}
                if self.prompt:
                    origin = (self.session.sid, data.get('id'), assignment.get('id'))
                    state = data.get('state')
                    finished = state in ('completed', 'cancelled', 'failed', 'interrupted')
                    if data.get('action') and state == 'completed' and (
                            assignment.get('result') or assignment.get('latest_result')):
                        self.agent_assistant_assignments.add(origin)
                        if len(self.agent_assistant_assignments) > 512:
                            self.agent_assistant_assignments = {origin}
                    if finished and assignment.get('id'):
                        key = (self.session.sid, data.get('id'), assignment['id'], state)
                        if key in self.assistant_results_seen:
                            continue
                        self.assistant_results_seen.add(key)
                        if len(self.assistant_results_seen) > 512:
                            self.assistant_results_seen = {key}
                    if not finished and (data.get('action') or self.session.agent.active):
                        continue
                    if state == 'completed' and origin in self.agent_assistant_assignments:
                        continue  # A tool has already returned this result to the main assistant.
                    if state == 'completed' and self.session.agent.active:
                        name = str(data.get('name') or data.get('id', 'helper'))
                        self.say(name + ' finished · /assistant read ' + shlex.quote(name))
                        continue
                # Chat tools emit compact records; polling adds a display label
                # and uses the full saved assignment fields. Accept both.
                self.say('Assistant ' + (data.get('label') or self.assistant_label(data)))
                if data.get('state') == 'pause_pending':
                    self.say('Pausing after the current step; /assistant cancel NAME stops it immediately.')
                if data.get('state') in ('completed', 'cancelled', 'failed', 'interrupted'):
                    result = assignment.get('latest_result') or assignment.get('result')
                    if result:
                        self.say(result)
                    if assignment.get('error'):
                        self.say('Assignment: ' + str(assignment['error']))
                    steps = assignment.get('steps') or []
                    if steps and steps[-1].get('job_id'):
                        self.say('Full saved worker output: /workers read ' + steps[-1]['job_id'])
                    if assignment.get('latest_result_truncated') or assignment.get('truncated'):
                        self.say('Saved context is compact or the worker reached its output limit.')
            elif kind == 'finished':
                if self.prompt:
                    self.prompt.preview = ''
                self.restored_failure = False
                successful = data['status'] in ('idle', 'completed')
                text = data.get('text', '')
                if event['seq'] > self.engine_outcome_after:
                    self.engine_failure = (None if successful else
                                           (self.session.sid, *self.engine_key(), text))
                if not successful and self.inbox is not None:
                    self.inbox.pause_pending(self.session.sid, event['seq'])
                if text or not successful:
                    self.last_answer = text if successful else ''
                if not successful:
                    self.pending_reply_audio = None
                    self.pending_talk_sid = None
                    self.say('Task ' + data['status'] + ': ' + text)
                    if self.handsfree and not getattr(self.speech, 'duplex', False):
                        self.set_voice(False, announce=False)
                        self.say('Hands-free paused because the task did not complete. Review the result before /voice on.')
                elif text and text != self.shown_answer:
                    self.say('Ostadix: ' + text)
                if (self.audio or (self.handsfree and not getattr(self.speech, 'duplex', False))) and successful and text:
                    try:
                        self.speak_completed_reply(self.last_answer)
                    except AppError as error:
                        self.pending_talk_sid = None
                        self.say('Audio: ' + str(error) + ' Use /hear to replay.')
        # Handle task failure before allowing another saved phrase to submit.
        # The inbox repeats that check under the agent lock to close worker races.
        if self.inbox is not None:
            self.inbox.pump()
            while not self.inbox.events.empty():
                kind, message = self.inbox.events.get_nowait()
                if kind in ('notice', 'error'):
                    self.say(message)
                elif kind == 'health':
                    self.observe_engine(message['agent'], message['client'], message['status'])
                elif kind == 'delivered' and self.captured_preview_id in (message.get('id'), message.get('previous_id')):
                    self.captured_preview_id = None
                    if self.prompt:
                        self.prompt.transcript = ''
        self.pump_reply_audio()
        self.pump_voice()
        self.pump_pending_talk()
        self.publish_interface_context()
        self.pump_companion()
        task_state = self.session.store.session(self.session.sid)['status']
        active = self.session.agent.active
        state = (task_state if active else 'stopping audio' if self.speech.stopping else
                 task_state if task_state == 'failed' else self.speech.mode or task_state)
        if active:
            generation = self.session.agent.task_generation
            if self.task_clock_generation != generation or self.task_clock is None:
                self.task_clock_generation, self.task_clock = generation, time.monotonic()
            label = {'thinking': 'Thinking', 'executing': 'Working', 'awaiting approval': 'Approval needed',
                     'connecting OSTADIX': 'Connecting OSTADIX'}.get(state, 'Working')
            if state == 'thinking' and self.stream_text:
                label = 'Replying'
            activity = label + ' ' + str(max(0, int(time.monotonic() - self.task_clock))) + 's'
        else:
            self.task_clock = self.task_clock_generation = None
            activity = ''
        counts = self.speech_counts()
        if not active and not (counts['queued'] + counts['sending']):
            self.voice_queue_notice = False
        if (self.handsfree and not self.voice_busy() and not self.speech.mode and not self.speech.stopping and
                not self.session.draft() and time.monotonic() < self.voice_next_listen):
            state = 'listening soon'
        if state in ('interrupted', 'cancelled', 'completed') and not self.session.agent.active:
            state = 'ready'
        if state == 'failed':
            state = ('ready for a new request; previous task failed'
                     if self.restored_failure and not self.session.agent.active and self.session.agent.rpc is None
                     else 'disconnected / failed; saved history retained')
        if state != self.last_status:
            if not self.prompt:
                self.say('[' + state + ']')
            self.last_status = state
        if self.prompt:
            self.prompt.state('draft' if self.session.draft() and not self.session.agent.active and
                              not self.speech.mode and not self.speech.stopping else state,
                              self.activity_badge(counts), activity)

    def history(self, logs=False):
        cursor = 0
        while True:
            events = self.session.store.events(self.session.sid, cursor)
            for e in events:
                if logs or e['kind'] in ('user', 'assistant', 'finished', 'notice', 'local_result', 'local_assistant'):
                    self.say(e['kind'] + ': ' + json.dumps(e['data'], ensure_ascii=False))
                cursor = e['seq']
            if len(events) < 300:
                break

    def handle(self, line):
        line = line.strip()
        aliases = {'start voice mode': '/voice on', 'hands free on': '/voice on',
                   'keep listening': '/voice on', 'stop voice mode': '/voice off',
                   'stop listening': '/voice off', 'approve all': '/approvals on'}
        line = aliases.get(line.lower().rstrip('.!?'), line)
        command, _, argument = line.partition(' ')
        argument = argument.strip()
        if (not line.startswith('/') and len(line) < 160
                and re.match(r'(?i)^(?:what|why|explain)\b', line)
                and re.search(r'(?i)\b(?:\d+\s+review|review\s+(?:count|counter|number))\b', line)):
            self.explain_review()
            return
        if not line:
            line = '/send' if self.session.draft() or (self.inbox and self.inbox.review_text()) else '/talk'
            command = line
        if not line.startswith('/'):
            self.pending_talk_sid = None
            if self.handsfree and self.speech.mode == 'listening' and not getattr(self.speech, 'duplex', False):
                self.speech.stop()
                self.discard_capture()
            if self.handsfree and self.inbox is not None:
                self.speech.interrupt_speech()
                self.inbox.add(line)
            elif self.session.agent.active:
                saved = self.session.draft()
                self.session.draft(saved + '\n' + line if saved else line)
                self.say('Saved for your next turn. Press Enter to send it after this reply, or /discard.')
            else:
                self.session.submit(line)
        elif command == '/help':
            if argument not in ('', 'all'):
                raise AppError('Use /help or /help all.')
            self.say(HELP if argument == 'all' else QUICK_HELP)
        elif command == '/controls':
            if argument:
                raise AppError('Use /controls for current settings, or /help for controls.')
            self.controls()
        elif command == '/models':
            if argument:
                raise AppError('Use /models, then /engine pixelforge ID.')
            self.show_models()
        elif command == '/engine':
            if argument in ('', 'status'):
                self.say(self.engine_summary() + ' · /engine gemini|local|codex|pixelforge')
            else:
                self.select_engine(argument)
        elif command == '/voice':
            if argument in ('on', 'off'):
                self.set_voice(argument == 'on')
            elif argument in ('', 'status'):
                self.say('Hands-free ' + ('on' if self.handsfree else 'off') + '. /voice on starts; /voice off stops.')
            else:
                raise AppError('Use /voice on, /voice off, or /voice status.')
        elif command == '/mic':
            if argument in ('mute', 'off'):
                self.set_microphone_muted(True)
            elif argument in ('unmute', 'on'):
                self.set_microphone_muted(False)
            elif argument in ('', 'status'):
                self.say('Microphone: ' + ('muted' if self.microphone_muted else 'on' if self.handsfree else 'off') +
                         ' · Volume Down → Up toggles · /mic mute|unmute')
            else:
                raise AppError('Use /mic mute, /mic unmute, or /mic status.')
        elif command == '/hotkey':
            if argument in ('on', 'off'):
                self.hotkey_enabled = argument == 'on'
                self.session.store.setting('terminal_volume_hotkey', 'on' if self.hotkey_enabled else 'off')
                if self.hotkey_enabled:
                    self.start_volume_hotkey()
                else:
                    self.close_volume_hotkey()
            elif argument not in ('', 'status'):
                raise AppError('Use /hotkey on, /hotkey off, or /hotkey status.')
            state = self.volume_hotkey.status if self.volume_hotkey is not None else 'off'
            self.say('Volume shortcut: ' + state + ' · Down → Up toggles your microphone.')
        elif command == '/talk':
            if self.handsfree and getattr(self.speech, 'duplex', False):
                self.speech.interrupt_speech()
                self.pump_voice()
                return
            if self.session.draft():
                raise AppError('A draft is saved. /send or /discard it first.')
            if self.voice_busy() or self.speech.stopping or self.speech.mode == 'speaking':
                if self.pending_talk_sid is None:
                    self.say('I’ll listen when this reply finishes. /stop cancels listening.')
                self.pending_talk_sid = self.session.sid
                self.talk_was_speaking = self.speech.mode == 'speaking'
                return
            if self.speech.mode == 'listening':
                return
            self.pending_talk_sid = None
            if hasattr(self.speech, 'start_once'):
                self.speech.start_once()
            else:
                self.speech.start('listening')
            self.capture_sid = self.session.sid
            self.say('Listening through Android. Speak now; /stop aborts capture.')
        elif command == '/send':
            self.pending_talk_sid = None
            if self.inbox is not None and self.inbox.review_text():
                count = self.inbox.retry_review()
                self.say('Retrying ' + str(count) + ' saved speech message(s).')
            elif self.handsfree and self.inbox is not None and self.session.draft():
                self.inbox.add(self.session.draft())
                self.session.draft('')
            elif self.session.agent.active:
                self.say('Your draft is saved. Press Enter to send it when this reply finishes.')
            else:
                self.session.submit(self.session.draft())
        elif command == '/draft':
            if argument:
                raise AppError('Use /draft to review saved speech without sending it.')
            draft = self.session.draft()
            review = self.inbox.review_text() if self.inbox is not None else ''
            if draft:
                self.say('Saved draft:\n' + draft)
            if review:
                self.say('Saved speech for review:\n' + review)
            self.say('/send explicitly sends saved text; /discard clears it.' if draft or review
                     else 'No saved speech or draft.')
        elif command == '/review':
            if argument:
                raise AppError('Use /review for the saved-speech counter, or /draft to read the saved words.')
            self.explain_review()
        elif command in ('/apps', '/cpu'):
            hardware = command == '/cpu'
            if hardware and argument:
                raise AppError('Use /cpu to read the phone hardware directly.')
            if not hardware and argument and not re.fullmatch(r'[A-Za-z0-9_.]{1,150}', argument):
                raise AppError('Use /apps or /apps PACKAGE_SUBSTRING.')
            if self.phone_busy:
                raise AppError('A phone query is already running.')
            from types import SimpleNamespace
            sid = self.session.sid
            phone_request = SimpleNamespace(sid=sid, cancelled=self.phone_cancel,
                                            action_policy=self.session.agent.action_policy)
            def read_apps():
                try:
                    result = self.session.agent.phone.execute(phone_request, sid,
                                ['@phone', 'cpu'] if hardware else ['@phone', 'apps'] + ([argument] if argument else []))
                    if hardware:
                        result = dict(result, terminal_hardware_display=True)
                    self.phone_events.put((sid, True, result))
                except Exception as error:
                    self.phone_events.put((sid, False, {'error': str(error)}))
            # Phone reads may wait for Android; keep input and audio responsive.
            worker = threading.Thread(target=read_apps, daemon=True)
            self.phone_worker = worker
            self.phone_busy = True
            try:
                worker.start()
            except Exception:
                self.phone_busy = False
                raise
        elif command == '/discard':
            self.session.draft('')
            if self.inbox is not None:
                self.inbox.discard_review()
            self.say('Speech draft discarded. Enter to speak again.')
        elif command == '/speaker':
            self.speak('Ostadix Voice is running in your terminal.')
            self.say('Speaker test requested through Android media audio. /stop to stop; /audio for setup help.')
        elif command == '/hear':
            if not self.last_answer:
                raise AppError('No completed answer to read yet.')
            self.speak(self.last_answer)
        elif command == '/stop':
            self.set_voice(False, announce=False)
            self.say('Audio stop requested. Hands-free off. The agent task is unchanged.')
        elif command == '/cancel':
            self.pending_reply_audio = None
            # RPC cancellation can wait for the server; input remains responsive.
            if self.inbox is not None:
                self.inbox.pause_pending()
            self.session.cancel_background(self.speech.events)
        elif command == '/approvals':
            if argument not in ('', 'status', 'on', 'off'):
                raise AppError('Use /approvals on for automatic approval, or /approvals off to require approval.')
            if argument in ('on', 'off'):
                self.session.agent.set_action_policy('auto' if argument == 'on' else 'ask')
            automatic = self.session.agent.action_policy() == 'auto'
            self.say('Approve all: ' + ('on. Actions run directly; /approvals off to require approval.' if automatic else
                                      'off. Actions wait for approval; /approvals on to run them directly.'))
        elif command in ('/approve', '/deny'):
            self.session.approve(argument, command == '/approve')
        elif command in ('/history', '/logs'):
            self.history(logs=command == '/logs')
        elif command == '/sessions':
            for i, session in enumerate(self.session.sessions(), 1):
                self.say(f"{i}. {session['workspace']} [{session['id'][:8]}]" + (' current' if session['id'] == self.session.sid else ''))
        elif command == '/use':
            sessions = self.session.sessions()
            if not argument.isdecimal() or not 1 <= int(argument) <= len(sessions):
                raise AppError('Use a conversation number from /sessions.')
            self.session.select(sessions[int(argument)-1]['id'])
            self.restore()
        elif command == '/new':
            self.session.new(argument or '.')
            self.restore()
        elif command == '/inspect':
            if self.session.agent.active:
                raise AppError('Wait for or cancel the task first.')
            result = self.session.agent.workspace(self.session.sid).read(argument)
            self.session.store.event(self.session.sid, 'notice', {'text':'Local inspection, no model call; saved in /logs.'})
            self.session.store.event(self.session.sid, 'tool_result', {'success':True, 'result':result})
        elif command in ('/mute', '/unmute'):
            self.audio = command == '/unmute'
            if not self.audio:
                self.pending_reply_audio = None
            self.session.store.setting('terminal_auto_audio', 'on' if self.audio else 'off')
            if not self.audio and getattr(self.speech, 'duplex', False):
                self.speech.interrupt_speech()
            elif not self.audio and self.speech.mode == 'speaking':
                self.speech.stop()
            if not self.audio and self.handsfree and not getattr(self.speech, 'duplex', False):
                self.set_voice(False, announce=False)
                self.say('Hands-free off.')
            self.say('Automatic speech ' + ('enabled.' if self.audio else 'disabled.') + ' Preference saved.')
        elif command == '/audio':
            self.audio_status(help=True)
        elif command == '/status':
            session = self.session.store.session(self.session.sid)
            self.say('Approve all: ' + ('on' if self.session.agent.action_policy() == 'auto' else 'off'))
            self.say(self.engine_summary() + ' · saved task ' + session['status'])
            self.say('CPU helper: ' + self.local_summary())
            self.service_summary()
            self.say(self.npu_summary())
            self.say(self.ostadix_summary())
            self.audio_status()
            self.say('Workspace: ' + session['workspace'])
            self.say('Session: ' + self.session.sid + '\nState: ' + str(self.session.store.directory))
        elif command == '/companion':
            action, _, value = argument.partition(' ')
            action, value = action or 'status', value.strip()
            minutes = 60
            valid = action in ('status', 'on', 'off', 'read', 'reply', 'snooze', 'dismiss')
            if action == 'snooze' and value:
                valid = len(value) <= 4 and value.isascii() and value.isdecimal() and 1 <= int(value) <= 1440
                if valid:
                    minutes = int(value)
            elif action == 'reply':
                valid = bool(value)
            elif value:
                valid = False
            if not valid:
                raise AppError('Use /companion status, on, off, read, reply TEXT, snooze [1–1440 MINUTES], or dismiss.')
            self.companion_action(action, minutes if action == 'snooze' else value if action == 'reply' else None)
        elif command == '/optimizer':
            words = argument.split()
            action = words[0] if words else 'status'
            objective = words[1] if len(words) == 2 else 'balanced'
            if (action not in ('status', 'on', 'off', 'history') or len(words) > 2 or
                    (action != 'on' and len(words) > 1) or
                    objective not in ('balanced', 'performance', 'battery')):
                raise AppError('Use /optimizer status, /optimizer on [balanced|performance|battery], /optimizer off, or /optimizer history.')
            self.optimizer_action(action, objective)
        elif command == '/agent':
            action, _, request = argument.partition(' ')
            action = action or 'list'
            if action not in ('ask', 'list', 'read', 'cancel') or (action != 'list' and not request.strip()) or (action == 'list' and request.strip()):
                raise AppError('Use /agent ask TASK, /agent list, /agent read ID, or /agent cancel ID.')
            self.local_action('evidence_' + action, request.strip())
        elif command == '/npu':
            if argument not in ('', 'status', 'on', 'off'):
                raise AppError('Use /npu status, /npu on, or /npu off.')
            self.npu_action(argument or 'status')
        elif command == '/local':
            action, _, request = argument.partition(' ')
            action = action or 'status'
            if action not in ('status', 'start', 'stop', 'ask') or (action == 'ask' and not request.strip()):
                raise AppError('Use /local status, /local start, /local stop, or /local ask REQUEST.')
            self.local_action(action, request.strip())
        elif command == '/workers':
            action, _, ident = argument.partition(' ')
            action = action or 'list'
            if action not in ('list', 'read', 'cancel') or (action == 'read' and not ident.strip()):
                raise AppError('Use /workers, /workers read ID, or /workers cancel [ID|all].')
            self.local_action(action, ident.strip())
        elif command == '/assistants':
            if argument:
                raise AppError('Use /assistants to list names, or /assistant for a control.')
            self.local_action('assistant_list')
        elif command == '/assistant':
            action, _, text = argument.partition(' ')
            if action not in ('create', 'ask', 'guide', 'read', 'pause', 'resume', 'cancel'):
                raise AppError('Use /assistant create|ask|guide|read|pause|resume|cancel NAME. /help lists examples.')
            name, rest = self.assistant_argument(text.strip())
            if not name or (action in ('ask', 'guide') and not rest) or (action in ('read', 'pause', 'resume', 'cancel') and rest):
                raise AppError('Specify a name (quote names with spaces), then a task for ask or text for guide.')
            role = ' '.join(shlex.split(rest)) if action == 'create' and rest else 'assistant'
            self.local_action('assistant_' + action, {'name': name, 'text': rest, 'role': role})
        elif command == '/ostadix':
            if argument not in ('', 'status'):
                raise AppError('Use /ostadix to check the connection, or /o SOURCE to run OSTADIX code.')
            self.session.ostadix()
        elif command == '/o':
            if not argument:
                raise AppError('Use /o SOURCE, for example: /o python^(1 + 1)_python')
            self.session.ostadix(argument)
        elif command == '/quit':
            self.quit = True
        else:
            raise AppError('Unknown control. Use /help, or type your request without a leading slash.')

    def banner(self):
        self.say('Ostadix · ' + ('voice conversation' if self.handsfree or self.start_voice_requested else
                                'spoken replies' if self.audio else 'text conversation') + ' · OSTADIX embedded')
        if (self.handsfree or self.start_voice_requested) and not self.audio:
            self.say('Talk or type · speaker off: /unmute · /stop · /controls')
        elif getattr(self.speech, 'duplex', False) and (self.handsfree or self.start_voice_requested):
            self.say('Talk or type · speak during replies · /stop · /help')
        else:
            self.say('Type to chat · Enter to talk · /voice on for hands-free · /help')

    def run_loop(self):
        pending = b''  # Canonical/nonTTY fallback; tests may inject output.
        while not self.quit:
            try:
                self.pump()
                if select.select([sys.stdin], [], [], .1)[0]:
                    chunk = os.read(sys.stdin.fileno(), 4096)
                    if not chunk:
                        break
                    if self.prompt:
                        lines = self.prompt.feed(chunk)
                    else:
                        pending += chunk
                        parts = pending.split(b'\n')
                        pending = parts.pop()
                        lines = [part.decode('utf-8', errors='replace') for part in parts]
                    for line in lines:
                        if line is None:
                            self.quit = True
                        if self.quit:
                            break
                        try:
                            self.pump()
                            self.handle(line)
                        except (AppError, OSError, ValueError) as error:
                            self.say('Error: ' + str(error))
            except KeyboardInterrupt:
                self.set_voice(False, announce=False)
                self.say('Audio stop requested. Hands-free off. /cancel cancels the task; /quit saves and exits.')

    def run(self):
        rich = self.output is print and sys.stdin.isatty() and sys.stdout.isatty() and os.environ.get('TERM') != 'dumb'
        self.banner()
        self.restore()
        if self.hotkey_enabled:
            self.start_volume_hotkey()
        try:
            if self.start_voice_requested:
                try:
                    self.set_voice(True, preserve_mute=True)
                except AppError as error:
                    self.say('Hands-free paused: ' + str(error))
            if rich:
                self.prompt = TerminalPrompt()
                self.output = self.prompt.write
                try:
                    with self.prompt:
                        self.run_loop()
                finally:
                    self.output = print
                    self.prompt = None
            else:
                self.run_loop()
        finally:
            self.close_volume_hotkey()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--state-dir', type=Path, default=DEFAULT_STATE)
    audio_options = parser.add_mutually_exclusive_group()
    audio_options.add_argument('--no-audio', action='store_true', help='Start with automatic spoken answers disabled')
    audio_options.add_argument('--voice', action='store_true', help='Start hands-free conversation for this terminal session')
    args = parser.parse_args(argv)
    if os.geteuid() == 0:
        raise AppError('Run c from normal Termux, outside su/root.')
    os.umask(0o077)
    if not sys.stdin.isatty():
        raise AppError('Run c in an interactive terminal.')
    session = TerminalSession(args.workspace, args.state_dir)
    from duplex_speech import DuplexSpeech
    ui = TerminalUI(session, speech=DuplexSpeech(NativeSpeech()),
                    audio=False if args.no_audio else None, voice=args.voice)
    def stop(*_):
        ui.quit = True
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        # Raising also interrupts a read/write which Python would otherwise
        # restart after a flag-only handler, including a suspended terminal read.
        raise SystemExit(0)

    previous = signal.signal(signal.SIGTERM, stop)
    try:
        from launch import release_launch_lock
        release_launch_lock(args.state_dir)
        ui.run()
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        try:
            try:
                ui.session.agent.interface.detach()
                ui.close_volume_hotkey()
                ui.phone_cancel.set()
                ui.speech.close()
            finally:
                try:
                    ui.close_phone_queries()
                finally:
                    session.close()
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    try:
        main()
    except (AppError, OSError) as error:
        raise SystemExit(display_text(error))
