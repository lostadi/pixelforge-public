"""Durable ordered delivery of finalized live speech to the current agent turn.

UI-thread contract: add() persists immediately; pump() processes worker results
and launches at most one delivery. Read events as (kind, payload) tuples, where
kind is notice/error (text), delivered ({sid,id,text,steered}), or health
({sid,agent,client,status}; ephemeral identity-scoped passive status). A restart or
uncertain outcome preserves text for explicit /send review, never automatic
replay. review_text() returns that text; retry_review() explicitly gives those
review entries fresh request IDs. pause_pending() is /cancel's queue boundary;
new speech afterwards is allowed. stop() permanently prevents new dispatch.

Shutdown ordering is stop(), agent.close(), close(), then store.db.close().
close() joins the delivery worker before applying its final persistence result.
This module never records audio or sends speech to a different conversation.
"""
from __future__ import annotations
import json
import queue
import threading
import time
import uuid
from agent import AppError, SteerUnavailable


class _TaskFailure(AppError):
    def __init__(self, seq):
        super().__init__('The previous task did not complete. Review /history before retrying saved speech.')
        self.seq = seq


class VoiceInbox:
    MAX_ITEMS = 64
    MAX_CHARACTERS = 128000
    RETRY_SECONDS = .2

    def __init__(self, session):
        self.session = session
        self.events = queue.Queue()
        self._outcomes = queue.Queue()
        self._entries = {}
        self._worker = None
        self._inflight = None
        self._retry_at = 0.0
        self._stopped = False
        self._closed = False
        self._lock = threading.RLock()
        self.restore()

    @staticmethod
    def _key(sid):
        return 'terminal_voice_pending_' + sid

    def _save(self, sid):
        self.session.store.setting(self._key(sid), json.dumps(self._entries[sid], ensure_ascii=False))

    def _boundary(self, sid):
        with self.session.store.lock:
            row = self.session.store.db.execute(
                'SELECT max(seq) FROM events WHERE session=?', (sid,)).fetchone()
        return row[0] or 0

    def _check_failure(self, sid, item):
        # A submit acknowledgement precedes asynchronous model failure. Never
        # let the next saved phrase cross that failure as a fresh request.
        with self.session.store.lock:
            rows = self.session.store.db.execute(
                "SELECT seq,data FROM events WHERE session=? AND seq>? AND kind='finished' ORDER BY seq",
                (sid, item.get('after_event', 0))).fetchall()
        for row in rows:
            result = json.loads(row['data'])
            if result.get('status') not in ('idle', 'completed'):
                raise _TaskFailure(row['seq'])

    def _check_model(self, agent, sid):
        # This runs in the delivery worker under the agent lock, never from a
        # controls renderer. Client status is passive and cannot start a model.
        client = getattr(agent, 'local_client', None)
        if client is None:
            return
        try:
            status = client.status()
        except Exception as error:
            status = {'ready': False, 'error': str(error)}
        self.events.put(('health', {'sid': sid, 'agent': agent, 'client': client, 'status': status}))
        if not status.get('ready'):
            message = status.get('error') or 'The selected model is not ready. Check /controls'
            if status.get('recovering'):
                raise SteerUnavailable(message, 'starting')
            raise AppError(message)

    def _load(self, sid):
        if sid in self._entries:
            return self._entries[sid]
        raw = self.session.store.setting(self._key(sid))
        try:
            entries = json.loads(raw) if raw else []
            if not isinstance(entries, list) or any(
                not isinstance(item, dict) or not isinstance(item.get('id'), str)
                or not isinstance(item.get('text'), str) or item.get('state') not in ('queued', 'sending', 'review')
                for item in entries
            ):
                raise ValueError('Invalid saved voice input')
        except (TypeError, ValueError):
            raise AppError('Saved voice input could not be read; its original data was kept.') from None
        entries = [{'id': item['id'], 'text': item['text'], 'state': 'review'} for item in entries]
        self._entries[sid] = entries
        if entries:
            self._save(sid)
            self.events.put(('notice', 'Saved speech needs review; delivery may already have occurred. '
                             '/send explicitly retries it, or /discard clears it. It was not automatically replayed.'))
        return entries

    def restore(self):
        with self._lock:
            self._load(self.session.sid)

    @property
    def pending(self):
        with self._lock:
            return bool(self._load(self.session.sid))

    @property
    def delivering(self):
        """Current speech awaiting delivery; review-only text does not mute replies."""
        with self._lock:
            return any(item['state'] in ('queued', 'sending') for item in self._load(self.session.sid))

    def counts(self):
        """Current conversation queue state, without exposing or dispatching text."""
        with self._lock:
            result = dict(queued=0, sending=0, review=0)
            for item in self._load(self.session.sid):
                result[item['state']] += 1
            return result

    def add(self, text):
        if not isinstance(text, str) or not text.strip() or len(text) > 32000:
            raise AppError('Voice input must contain 1–32,000 characters; this segment was not queued.')
        with self._lock:
            if self._stopped:
                raise AppError('Voice delivery is closing; this segment was not queued.')
            sid = self.session.sid
            entries = self._load(sid)
            if len(entries) >= self.MAX_ITEMS or sum(len(item['text']) for item in entries)+len(text) > self.MAX_CHARACTERS:
                raise AppError('The saved voice queue is full. Review /send or /discard before continuing; this segment was not queued.')
            item = {'id': uuid.uuid4().hex, 'text': text, 'state': 'queued',
                    'after_event': self._boundary(sid)}
            entries.append(item)
            try:
                self._save(sid)
            except Exception:
                entries.remove(item)
                raise
            return item['id']

    def review_text(self):
        with self._lock:
            return '\n'.join(item['text'] for item in self._load(self.session.sid) if item['state'] == 'review')

    def retry_review(self):
        """Explicit user retry only: delivery may already have happened before review."""
        with self._lock:
            if self._stopped:
                raise AppError('Voice delivery is closing.')
            sid = self.session.sid
            entries = self._load(sid)
            if self._inflight and self._inflight[0] == sid and any(item['id'] == self._inflight[1] and item['state'] == 'review' for item in entries):
                raise AppError('The previous voice delivery is still finishing; its saved text was kept.')
            reviewed = [item for item in entries if item['state'] == 'review']
            boundary = self._boundary(sid)
            for item in reviewed:
                item['id'] = uuid.uuid4().hex
                item['state'] = 'queued'
                item['after_event'] = boundary
            if reviewed:
                self._save(sid)
                self._retry_at = 0
                self.events.put(('notice', 'Retrying saved speech as explicitly requested. Earlier delivery may already have occurred.'))
            return len(reviewed)

    def discard_review(self):
        with self._lock:
            sid = self.session.sid
            entries = self._load(sid)
            reviewed = [item for item in entries if item['state'] == 'review']
            self._entries[sid] = [item for item in entries if item['state'] != 'review']
            if reviewed:
                self._save(sid)
            return len(reviewed)

    def pause_pending(self, sid=None, before_event=None):
        """Save pending speech; optional session/event scope excludes newer retries."""
        with self._lock:
            self._load(self.session.sid)
            count = 0
            for entry_sid, entries in self._entries.items():
                if sid is not None and sid != entry_sid:
                    continue
                for item in entries:
                    if (item['state'] != 'review' and
                            (before_event is None or item.get('after_event', 0) < before_event)):
                        item['state'] = 'review'
                        count += 1
                if entries:
                    self._save(entry_sid)
            if count:
                self.events.put(('notice', 'Pending speech was saved for review. /send retries it; /discard clears it.'))
            return count

    def _dispatch_current(self, sid, item, agent):
        # Caller owns agent.lock and _lock. Acquiring the agent lock first lets
        # the UI pause pending speech while a worker waits for agent state.
        saved = next((entry for entry in self._entries.get(sid, []) if entry['id'] == item['id']), None)
        return (not self._stopped and self.session.sid == sid and self.session.agent is agent
                and self._inflight == (sid, item['id'])
                and saved is not None and saved['state'] == 'sending')

    def _deliver(self, sid, item):
        agent = self.session.agent
        try:
            # submit resets a stale cancellation flag. Calling steer first while
            # idle would reject a perfectly valid new post-cancellation request.
            with agent.lock:
                with self._lock:
                    if not self._dispatch_current(sid, item, agent):
                        return ('held', False, None)
                    self._check_failure(sid, item)
                self._check_model(agent, sid)
                with self._lock:
                    if not self._dispatch_current(sid, item, agent):
                        return ('held', False, None)
                    self._check_failure(sid, item)
                    if item.get('wait_for_idle') and agent.active:
                        raise SteerUnavailable('Waiting for the previous turn to finish locally', 'starting')
                    if not agent.active:
                        if agent.tool_lock.locked():
                            raise SteerUnavailable('Previous tools are finishing', 'starting')
                        result = agent.submit(sid, item['text'], item['id'])
                        return ('delivered', False, result)
            try:
                result = agent.steer(sid, item['text'], item['id'])
                return ('delivered', True, result)
            except SteerUnavailable as error:
                if error.reason != 'idle':
                    raise
                with agent.lock, self._lock:
                    if not self._dispatch_current(sid, item, agent):
                        return ('held', False, None)
                    self._check_failure(sid, item)
                    if agent.active or agent.tool_lock.locked():
                        raise SteerUnavailable('Task changed during voice delivery', 'starting')
                    result = agent.submit(sid, item['text'], item['id'])
                    return ('delivered', False, result)
        except _TaskFailure as error:
            return ('failed_task', False, {'error': str(error), 'before_event': error.seq})
        except SteerUnavailable as error:
            if error.reason in ('starting', 'idle'):
                return ('retry', False, None)
            if error.reason == 'rejected_idle':
                return ('rejected_idle', False, None)
            return ('review', False, str(error))
        except Exception as error:
            return ('review', False, str(error))

    def _run(self, sid, item):
        self._outcomes.put((sid, item['id'], self._deliver(sid, item)))

    def _drain(self):
        while True:
            try:
                sid, request_id, (outcome, steered, result) = self._outcomes.get_nowait()
            except queue.Empty:
                return
            entries = self._entries[sid]
            item = next((item for item in entries if item['id'] == request_id), None)
            self._inflight = None
            self._worker = None
            if item is None:  # Explicit discard while delivery was finishing.
                continue
            if outcome == 'delivered':
                entries.remove(item)
                self._save(sid)
                payload = {'sid': sid, 'id': request_id, 'text': item['text'], 'steered': steered}
                if item.get('previous_id'):
                    payload['previous_id'] = item['previous_id']
                self.events.put(('delivered', payload))
            elif outcome == 'held':
                item['state'] = 'review'
                if self._stopped or sid != self.session.sid:
                    for pending in entries:
                        pending['state'] = 'review'
                self._save(sid)
            elif outcome == 'retry':
                if item['state'] != 'review':
                    item['state'] = 'review' if self._stopped else 'queued'
                self._retry_at = time.monotonic()+self.RETRY_SECONDS
                self._save(sid)
            elif outcome == 'rejected_idle':
                # The peer explicitly reported no active turn, so no speech was
                # accepted. This narrow completion race may retry safely, but
                # stop, cancellation, review and conversation changes still win.
                if item['state'] == 'review' or self._stopped or sid != self.session.sid:
                    item['state'] = 'review'
                    if self._stopped or sid != self.session.sid:
                        for pending in entries:
                            pending['state'] = 'review'
                else:
                    item['previous_id'] = item.get('previous_id', request_id)
                    item['id'] = uuid.uuid4().hex
                    item['state'] = 'queued'
                    item['wait_for_idle'] = True
                self._retry_at = time.monotonic()+self.RETRY_SECONDS
                self._save(sid)
            else:
                # Subsequent chunks were spoken before this uncertain outcome;
                # keep their order for one explicit review instead of skipping.
                for pending in entries:
                    if outcome != 'failed_task' or pending.get('after_event', 0) < result['before_event']:
                        pending['state'] = 'review'
                self._save(sid)
                message = result['error'] if outcome == 'failed_task' else str(result)
                self.events.put(('error', 'Voice delivery needs review: ' + message +
                                 ' Saved speech was kept; no automatic replay. /send retries or /discard clears it.'))

    def pump(self):
        with self._lock:
            if self._closed:
                return
            self._drain()
            if self._stopped or self._inflight or time.monotonic() < self._retry_at:
                return
            sid = self.session.sid
            entries = self._load(sid)
            item = next((item for item in entries if item['state'] == 'queued'), None)
            if item is None:
                return
            # Old conversation queues are preserved for that conversation only.
            item['state'] = 'sending'
            self._save(sid)
            self._inflight = (sid, item['id'])
            self._worker = threading.Thread(target=self._run, args=(sid, item.copy()), daemon=True,
                                            name='voice-input-delivery')
            try:
                self._worker.start()
            except Exception:
                self._worker = None; self._inflight = None
                item['state'] = 'review'; self._save(sid)
                raise

    def stop(self):
        with self._lock:
            self._stopped = True

    def close(self):
        with self._lock:
            if self._closed:
                return
        self.stop()
        with self._lock:
            worker = self._worker
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        with self._lock:
            self._drain()
            for sid, entries in self._entries.items():
                for item in entries:
                    item['state'] = 'review'
                if entries:
                    self._save(sid)
            self._closed = True
