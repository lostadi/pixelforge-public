"""A bounded, persistent JSON-RPC MCP stdio connection (standard library only)."""

import json
import os
from pathlib import Path
import queue
import re
import select
import signal
import subprocess
import threading
import time


MAX_FRAME = 16 * 1024 * 1024


class MCPError(RuntimeError):
    pass


def _safe_message(value):
    text = str(value)[:2000]
    text = re.sub(r"(?i)(bearer\s+|(?:api[_-]?key|token|password)\s*[=:]\s*)\S+", r"\1[redacted]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", text)
    return ''.join(char for char in text if char in '\n\t' or ord(char) >= 32).replace('\x7f', '')


class MCPClient:
    """Own one child process and never replay a request after any transport failure.

    Cancellation is best effort: a cancelled tool may already have changed files.
    Callers must reconcile the tool's state before choosing to retry it.
    """

    def __init__(self, command: list[str], cwd: Path, env: dict):
        self._state_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._pending = {}
        self._next_id = 0
        self._failure = None
        self._closed_pipes = False
        self.shutdown_graceful = False
        self._stderr_bytes = 0
        self.info = {}
        try:
            self.process = subprocess.Popen(
                command, cwd=cwd, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True, bufsize=0,
            )
        except (OSError, ValueError) as error:
            raise MCPError('Could not start MCP server: ' + _safe_message(error)) from error
        os.set_blocking(self.process.stdin.fileno(), False)
        self._reader = threading.Thread(target=self._read, name='mcp-stdio', daemon=True)
        self._stderr = threading.Thread(target=self._drain_stderr, name='mcp-stderr', daemon=True)
        self._reader.start()
        self._stderr.start()
        try:
            info = self.call('initialize', {
                'protocolVersion': '2025-03-26', 'capabilities': {},
                'clientInfo': {'name': 'ostadix_voice', 'version': '0.1'},
            }, timeout=15)
            if not isinstance(info, dict) or info.get('protocolVersion') != '2025-03-26':
                raise MCPError('MCP server did not negotiate supported protocol 2025-03-26')
            self.info = info
            self._send({'jsonrpc': '2.0', 'method': 'notifications/initialized'}, time.monotonic() + 2)
        except BaseException:
            self.close()
            raise

    @property
    def alive(self):
        return self._failure is None and self.process.poll() is None

    @property
    def stderr_tail(self):
        # Stderr can contain arbitrary credential formats. Never retain its text.
        return f'{self._stderr_bytes} bytes of MCP stderr suppressed' if self._stderr_bytes else ''

    @staticmethod
    def _raise_for_call(method, message):
        if method == 'tools/call':
            message += ' Tool outcome may be unknown; do not automatically retry.'
        raise MCPError(message)

    def _fail(self, reason):
        with self._state_lock:
            if self._failure is None:
                self._failure = reason
            pending = list(self._pending.values())
        for mailbox in pending:
            try:
                mailbox.put_nowait(('failure', self._failure))
            except queue.Full:
                pass

    def _send(self, message, deadline, cancel_event=None):
        try:
            encoded = (json.dumps(message, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')
        except (TypeError, ValueError, RecursionError) as error:
            raise MCPError('MCP request is not valid JSON') from error
        if len(encoded) > MAX_FRAME:
            raise MCPError('MCP request exceeds the 16 MiB frame limit')
        acquired = False
        written = 0
        try:
            while not acquired:
                if cancel_event is not None and cancel_event.is_set():
                    raise MCPError('MCP request cancelled')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPError('MCP request timed out while sending')
                acquired = self._send_lock.acquire(timeout=min(.05, remaining))
            while written < len(encoded):
                if self._failure is not None:
                    raise MCPError(self._failure)
                if cancel_event is not None and cancel_event.is_set():
                    raise MCPError('MCP request cancelled')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise MCPError('MCP request timed out while sending')
                _, writable, _ = select.select([], [self.process.stdin], [], min(.05, remaining))
                if writable:
                    try:
                        written += os.write(self.process.stdin.fileno(), encoded[written:written + 65536])
                    except BlockingIOError:
                        continue
        except (OSError, ValueError) as error:
            self._fail('MCP server connection closed while sending')
            raise MCPError(self._failure) from error
        except MCPError:
            if written:
                # A truncated JSON frame cannot safely share a later request.
                self._fail('MCP connection interrupted during a request; connection cannot be reused')
            raise
        finally:
            if acquired:
                self._send_lock.release()

    def _notify_cancel(self, ident, reason):
        try:
            self._send({'jsonrpc': '2.0', 'method': 'notifications/cancelled',
                        'params': {'requestId': ident, 'reason': reason}}, time.monotonic() + .25)
        except (MCPError, OSError):
            pass

    def call(self, method, params, timeout=120, cancel_event=None):
        deadline = time.monotonic() + max(0, timeout)
        mailbox = queue.Queue(maxsize=1)
        with self._state_lock:
            if self._failure is not None:
                self._raise_for_call(method, self._failure)
            self._next_id += 1
            ident = self._next_id
            self._pending[ident] = mailbox
        try:
            try:
                self._send({'jsonrpc': '2.0', 'id': ident, 'method': method, 'params': params}, deadline, cancel_event)
            except MCPError as error:
                self._notify_cancel(ident, 'Request could not finish sending')
                self._raise_for_call(method, str(error))
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    self._notify_cancel(ident, 'Cancelled by user')
                    self._raise_for_call(method, 'MCP request cancelled')
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._notify_cancel(ident, 'Request timeout')
                    self._raise_for_call(method, 'MCP request timed out')
                try:
                    kind, payload = mailbox.get(timeout=min(.05, remaining))
                except queue.Empty:
                    continue
                if kind == 'failure':
                    self._raise_for_call(method, payload)
                if 'error' in payload:
                    error = payload['error']
                    if not isinstance(error, dict):
                        self._raise_for_call(method, 'Invalid MCP JSON-RPC error')
                    self._raise_for_call(method, f"MCP error {error.get('code', 'unknown')}: {_safe_message(error.get('message', 'Request failed'))}")
                if 'result' not in payload:
                    self._raise_for_call(method, 'MCP response has no result')
                return payload['result']
        finally:
            with self._state_lock:
                self._pending.pop(ident, None)

    def _read(self):
        # Unbuffered read + explicit byte buffer keeps memory bounded even for a
        # server which emits an unterminated frame indefinitely.
        buffer = bytearray()
        try:
            while True:
                chunk = self.process.stdout.read(65536)
                if not chunk:
                    self._fail('MCP server disconnected; no requests were replayed')
                    return
                buffer.extend(chunk)
                while True:
                    newline = buffer.find(b'\n')
                    if newline < 0:
                        if len(buffer) >= MAX_FRAME:
                            raise MCPError('MCP response exceeds the 16 MiB frame limit')
                        break
                    if newline + 1 > MAX_FRAME:
                        raise MCPError('MCP response exceeds the 16 MiB frame limit')
                    frame = bytes(buffer[:newline])
                    del buffer[:newline + 1]
                    if not frame.strip():
                        continue
                    message = json.loads(frame)
                    if not isinstance(message, dict) or message.get('jsonrpc') != '2.0':
                        raise MCPError('Invalid MCP JSON-RPC response')
                    if 'method' in message:
                        if 'id' in message:
                            self._send({'jsonrpc': '2.0', 'id': message['id'], 'error': {
                                'code': -32601, 'message': 'Client does not support server-initiated requests',
                            }}, time.monotonic() + .5)
                        continue
                    ident = message.get('id')
                    if not isinstance(ident, int) or isinstance(ident, bool):
                        continue
                    with self._state_lock:
                        mailbox = self._pending.get(ident)
                    if mailbox is not None:
                        try:
                            mailbox.put_nowait(('response', message))
                        except queue.Full:
                            pass
        except (OSError, ValueError, RecursionError, MCPError):
            self._fail('MCP server sent an invalid or oversized frame, or its connection closed')

    def _drain_stderr(self):
        try:
            while chunk := self.process.stderr.read(8192):
                self._stderr_bytes += len(chunk)
        except (OSError, ValueError):
            pass

    def close(self):
        with self._close_lock:
            if self._closed_pipes:
                return
            self._fail('MCP client closed; no requests were replayed')

            # EOF lets the native server finish jobs.shutdown(), including jobs
            # in separate sessions that killpg(server_pid) cannot reach. Writers
            # use nonblocking pipes and observe _failure; never wait indefinitely
            # for a sender or for a server that ignores EOF.
            acquired = self._send_lock.acquire(timeout=.25)
            try:
                if acquired:
                    self.process.stdin.close()
            finally:
                if acquired:
                    self._send_lock.release()
            if acquired:
                try:
                    self.shutdown_graceful = self.process.wait(timeout=2.5) == 0
                except subprocess.TimeoutExpired:
                    pass
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=.25)
                except subprocess.TimeoutExpired:
                    pass
            # Clean up any remaining members of the server's original group.
            # Detached jobs require graceful shutdown above; forced termination
            # cannot claim their cleanup succeeded.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=.25)
            except subprocess.TimeoutExpired:
                pass
            for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
                pipe.close()
            self._closed_pipes = True
            for thread in (self._reader, self._stderr):
                if thread is not threading.current_thread():
                    thread.join(timeout=.5)
