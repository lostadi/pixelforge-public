"""Ephemeral, session-scoped controls acknowledged by the terminal UI thread."""
from collections import deque
import copy
from dataclasses import dataclass, field
import threading
import time

from agent import AppError


INSTRUCTIONS = '''You are speaking through Ostadix's phone voice interface. The interface
captures microphone audio and can speak your final answer. Answer questions about these
built-in controls directly; do not search the project for interface documentation.

Use workspace operation command with these reserved argv arrays:
  ["@voice", "status"]  Read the current terminal's voice settings and availability.
  ["@voice", "on"]      Enable hands-free conversation in this terminal.
  ["@voice", "off"]     Pause hands-free conversation.
  ["@voice", "mute"]    Mute only the user's microphone; AI speech can continue.
  ["@voice", "unmute"]  Unmute the user's microphone and resume conversation.
These are UI controls, not shell commands or project paths.
Use on/off only when the current user explicitly asks to enable or pause microphone
capture. Typing a question, asking for an explanation, requesting audio playback,
or changing speaker mute does not request turning the microphone off or on.
Check the returned state:
enabled voice mode does not prove the microphone started: check continuous_capture and
capabilities. With the native duplex interface, capture continues while the task runs
and while answers play. Speech interrupts playback and is sent to the active turn as
a follow-up. If no terminal is attached, the
status reports unavailable and controls fail; do not claim that recording started.

Manual mode: type a request and Enter to send it; empty Enter or /talk starts microphone
capture. Recognized speech is saved as a draft; Enter or /send sends it, and /discard
clears it. Plain c starts native continuous voice; c --text leaves the microphone off.
Voice mode shows partial transcription and sends finalized speech without another Enter.
/stop or Ctrl-C closes the microphone and stops playback; /cancel cancels the
agent task. /speaker tests speech playback; /hear replays the last answer; /mute and
/unmute control automatic spoken answers without stopping native microphone capture.
/mic mute and /mic unmute control the user's microphone independently. While c is
open, a quick Volume Down then Volume Up toggles that microphone setting. Do not
use @voice off or speaker mute for a request to mute only the user's microphone.
/audio shows audio controls and /quit exits. Capture ends when the terminal closes.
Speech transcripts are conversation input. Only the interface's listed spoken controls
are recognized; arbitrary slash commands in a transcript are not executed. Approve all
is on by default: requested actions run under the user's saved automatic action policy.
/approvals off enables individual approvals, and /approvals on restores automatic mode.
Do not ask for another confirmation when automatic mode is active.
Keep answers natural aloud: lead with the result, normally one or two short sentences.
'''


@dataclass
class _Request:
    action: str
    sid: str
    generation: int
    attachment: int
    deadline: float
    agent: object
    cancelled: object
    ready: threading.Event = field(default_factory=threading.Event)
    result: object = None
    error: str = ''


class InterfaceBridge:
    """Run small, nonblocking UI callbacks only from the attached UI thread.

    Requests are never stored in conversation history or replayed on attachment.
    Timeout/cancellation wins until a callback begins; callbacks must not block.
    """

    def __init__(self, wait_timeout=2.0):
        self.wait_timeout = wait_timeout
        self._lock = threading.RLock()
        self._pending = deque()
        self._handler = None
        self._thread = None
        self._attachment = 0
        self._closed = False
        self._context = None

    def update_context(self, sid, context):
        """Publish passive status on the UI thread, never microphone content."""
        with self._lock:
            if self._closed or self._handler is None:
                return
            if threading.get_ident() != self._thread:
                raise AppError('Interface context must be published by its UI thread')
            self._context = (sid, time.monotonic(), copy.deepcopy(context))

    def read_context(self, sid):
        """Return fresh, scoped UI metadata without running a UI callback."""
        with self._lock:
            if (self._closed or self._handler is None or self._context is None
                    or self._context[0] != sid or time.monotonic() - self._context[1] > 5):
                return {'available': False, 'interface': 'unavailable'}
            return copy.deepcopy(self._context[2])

    @staticmethod
    def _finish(request, *, result=None, error=''):
        request.result = result
        request.error = error
        request.ready.set()

    def _invalidate(self, message):
        self._attachment += 1
        while self._pending:
            request = self._pending.popleft()
            if not request.ready.is_set():
                self._finish(request, error=message)

    def attach(self, handler):
        if not callable(handler):
            raise TypeError('A voice control handler is required')
        with self._lock:
            if self._closed:
                raise AppError('The voice interface is closed')
            self._invalidate('The voice interface changed; no control was applied')
            self._handler = handler
            self._thread = threading.get_ident()
            self._context = None

    def detach(self):
        with self._lock:
            self._handler = None
            self._thread = None
            self._context = None
            self._invalidate('The voice interface closed; no control was applied')

    def close(self):
        with self._lock:
            self._closed = True
            self.detach()

    def dispatch(self, agent, args, workspace=None):
        argv = args.get('argv') if isinstance(args, dict) else None
        if (not isinstance(argv, list) or len(argv) != 2 or argv[0] != '@voice'
                or not isinstance(argv[1], str)):
            raise AppError('Voice controls require ["@voice", "status" | "on" | "off" | "mute" | "unmute"]')
        action = {'start': 'on', 'stop': 'off'}.get(argv[1], argv[1])
        if action not in ('status', 'on', 'off', 'mute', 'unmute'):
            raise AppError('Unknown voice control; use status, on, off, mute or unmute')
        with self._lock:
            if self._closed or self._handler is None:
                if action == 'status':
                    return {'available': False, 'interface': 'unavailable',
                            'message': 'No active terminal voice interface is attached. Open c in Termux.'}
                raise AppError('No active terminal voice interface is attached. Open c in Termux.')
            if agent.cancelled.is_set() or getattr(agent, 'closed', False):
                raise AppError('Task cancelled; voice control was not applied')
            request = _Request(action, agent.sid, agent.task_generation,
                               self._attachment, time.monotonic() + self.wait_timeout,
                               agent, agent.cancelled)
            self._pending.append(request)
        while not request.ready.wait(.05):
            with self._lock:
                if request.ready.is_set():
                    break
                if request.cancelled.is_set() or getattr(agent, 'closed', False):
                    self._finish(request, error='Task cancelled; voice control was not applied')
                elif (agent.sid != request.sid
                      or agent.task_generation != request.generation):
                    self._finish(request, error='The task changed; voice control was not applied')
                elif time.monotonic() >= request.deadline:
                    self._finish(request, error='The terminal did not acknowledge the voice control in time; no control was applied')
        if request.error:
            raise AppError(request.error)
        return request.result

    def pump(self, current_sid, current_generation, closed=False):
        if closed:
            self.detach()
            return 0
        processed = 0
        with self._lock:
            if self._handler is None or self._closed:
                return processed
            if threading.get_ident() != self._thread:
                raise RuntimeError('Voice controls must be pumped on the attached UI thread')
            while self._pending:
                request = self._pending.popleft()
                if request.ready.is_set():
                    continue
                if (request.attachment != self._attachment
                        or request.sid != current_sid
                        or request.generation != current_generation
                        or request.agent.sid != request.sid
                        or request.agent.task_generation != request.generation):
                    self._finish(request, error='The conversation or task changed; voice control was not applied')
                elif request.cancelled.is_set() or getattr(request.agent, 'closed', False):
                    self._finish(request, error='Task cancelled; voice control was not applied')
                elif time.monotonic() >= request.deadline:
                    self._finish(request, error='The voice control expired; no control was applied')
                else:
                    try:
                        result = self._handler(request.action)
                        if not isinstance(result, dict):
                            raise AppError('The voice interface returned an invalid acknowledgement')
                        self._finish(request, result=result)
                    except Exception as error:
                        self._finish(request, error=str(error) or 'Voice control failed')
                    processed += 1
        return processed
