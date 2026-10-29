import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agent import Store, AppError, SteerUnavailable
from voice_inbox import VoiceInbox


class FakeAgent:
    def __init__(self):
        self.lock=threading.RLock(); self.tool_lock=threading.Lock()
        self.active=False; self.calls=[]; self.starting=False
        self.error=None; self.block=None; self.cancelled=threading.Event()
    def submit(self,sid,text,rid):
        self.calls.append(('submit',sid,text,rid))
        self.active=True; self.cancelled.clear()
        return {'accepted':True}
    def steer(self,sid,text,rid):
        self.calls.append(('steer',sid,text,rid))
        if self.block:self.block.wait(2)
        if self.error:raise self.error
        if self.starting:raise SteerUnavailable('Starting','starting')
        return {'steered':True}


class VoiceInboxTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.store=Store(Path(self.temp.name)/'state')
        self.agent=FakeAgent(); self.session=SimpleNamespace(sid='first',store=self.store,agent=self.agent)
        self.inbox=VoiceInbox(self.session); self.inbox.RETRY_SECONDS=0
    def tearDown(self):
        if self.agent.block:self.agent.block.set()
        self.inbox.close();self.store.db.close();self.temp.cleanup()
    def pump_until(self,predicate):
        until=time.monotonic()+2
        while time.monotonic()<until:
            self.inbox.pump()
            if predicate():return
            time.sleep(.005)
        self.fail('Inbox did not reach expected state')
    def saved(self,sid='first'):
        return json.loads(self.store.setting('terminal_voice_pending_'+sid) or '[]')
    def test_add_is_immediate_lossless_and_preserves_order(self):
        phrases=[' first sentence  ','second\nline','third']
        for text in phrases:self.inbox.add(text)
        self.assertEqual([x['text'] for x in self.saved()],phrases)
        self.assertEqual(self.agent.calls,[])
        self.pump_until(lambda:not self.inbox.pending)
        self.assertEqual([x[2] for x in self.agent.calls],phrases)
        self.assertEqual([x[0] for x in self.agent.calls],['submit','steer','steer'])
    def test_review_is_pending_but_not_delivering(self):
        self.assertFalse(self.inbox.delivering)
        self.inbox.add('speech');self.assertTrue(self.inbox.delivering)
        self.inbox.pause_pending()
        self.assertTrue(self.inbox.pending);self.assertFalse(self.inbox.delivering)
        self.inbox.add('new speech');self.assertTrue(self.inbox.delivering)

    def test_counts_are_passive_and_scoped_to_current_conversation(self):
        self.inbox.add('first');self.inbox.add('second')
        self.inbox.pause_pending();self.inbox.add('new')
        counts=self.inbox.counts()
        self.assertEqual(counts,{'queued':1,'sending':0,'review':2})
        counts['review']=99
        self.assertEqual(self.inbox.counts()['review'],2)
        self.assertEqual(self.agent.calls,[])
        self.session.sid='other'
        self.assertEqual(self.inbox.counts(),{'queued':0,'sending':0,'review':0})

    def test_hardware_question_requires_the_selected_inference_engine(self):
        checked = threading.Event()
        def status():
            checked.set()
            return {'ready': False, 'error': 'Selected model unavailable'}
        self.agent.local_client = SimpleNamespace(status=status)
        self.inbox.add('what cpu is it')
        self.pump_until(lambda: self.inbox.counts()['review'] == 1)
        self.assertTrue(checked.is_set())
        self.assertEqual(self.agent.calls, [])
        self.assertEqual(self.saved()[0]['text'], 'what cpu is it')

    def test_hardware_question_is_sent_to_the_active_model_like_other_speech(self):
        self.agent.active = True
        request_id = self.inbox.add('which cpu does my phone use')
        self.pump_until(lambda: not self.inbox.pending)
        self.assertEqual(self.agent.calls, [('steer', 'first', 'which cpu does my phone use', request_id)])

    def test_active_task_uses_steer(self):
        self.agent.active=True;rid=self.inbox.add('follow up')
        self.pump_until(lambda:not self.inbox.pending)
        self.assertEqual(self.agent.calls,[('steer','first','follow up',rid)])

    def test_unavailable_model_keeps_all_32_reviewed_phrases_without_acceptance(self):
        self.agent.local_client = SimpleNamespace(status=lambda: {
            'ready': False, 'selection_state': 'disabled', 'error': 'Selected NPU is disabled'})
        for i in range(32):
            self.inbox.add('saved phrase ' + str(i))
        self.inbox.pause_pending()
        self.assertEqual(self.inbox.retry_review(), 32)
        ids = [item['id'] for item in self.saved()]
        self.pump_until(lambda: self.inbox.counts()['review'] == 32)
        self.assertEqual(self.agent.calls, [])
        self.assertEqual([item['id'] for item in self.saved()], ids)
        self.assertEqual([item['text'] for item in self.saved()], ['saved phrase ' + str(i) for i in range(32)])
        events = list(self.inbox.events.queue)
        self.assertTrue(any(kind == 'health' and data['status']['selection_state'] == 'disabled'
                            for kind, data in events))

    def test_recovering_model_waits_with_same_ids_until_ready(self):
        status = {'ready': False, 'recovering': True, 'error': 'Selected NPU is recovering'}
        checked = threading.Event()
        def inspect():
            checked.set()
            return dict(status)
        self.agent.local_client = SimpleNamespace(status=inspect)
        first = self.inbox.add('first'); second = self.inbox.add('second')
        self.pump_until(checked.is_set)
        self.inbox.pump()
        self.assertEqual(self.agent.calls, [])
        self.assertEqual(self.inbox.counts()['review'], 0)
        self.assertEqual([item['id'] for item in self.saved()], [first, second])
        status.update(ready=True, recovering=False)
        self.pump_until(lambda: not self.inbox.pending)
        self.assertEqual([call[3] for call in self.agent.calls], [first, second])

    def test_async_failure_after_acceptance_stops_remaining_31_phrases(self):
        def fail_after_submit(sid, text, rid):
            self.agent.calls.append(('submit', sid, text, rid))
            self.store.event(sid, 'user', {'text': text, 'request_id': rid})
            self.store.event(sid, 'finished', {'status': 'failed', 'text': 'Selected model is not ready'})
            self.agent.active = False
            return {'accepted': True}
        self.agent.submit = fail_after_submit
        for i in range(32):
            self.inbox.add('saved phrase ' + str(i))
        self.inbox.pause_pending(); self.inbox.retry_review()
        self.pump_until(lambda: self.inbox.counts()['review'] == 31)
        self.assertEqual([call[2] for call in self.agent.calls], ['saved phrase 0'])
        self.assertEqual([item['text'] for item in self.saved()], ['saved phrase ' + str(i) for i in range(1, 32)])
        for _ in range(5): self.inbox.pump()
        self.assertEqual(len(self.agent.calls), 1)
        self.assertEqual(self.store.events('first')[0]['data']['text'], 'saved phrase 0')

    def test_failure_while_worker_waits_for_agent_lock_prevents_submit(self):
        with self.agent.lock:
            self.inbox.add('held phrase'); self.inbox.add('following phrase'); self.inbox.pump()
            self.store.event('first', 'finished', {'status': 'failed', 'text': 'Backend failed'})
        self.pump_until(lambda: self.inbox.counts()['review'] == 2)
        self.assertEqual(self.agent.calls, [])

    def test_explicit_retry_after_failure_can_submit_when_model_is_ready(self):
        self.inbox.add('held phrase')
        self.store.event('first', 'finished', {'status': 'failed', 'text': 'Backend failed'})
        self.pump_until(lambda: self.inbox.counts()['review'] == 1)
        self.agent.local_client = SimpleNamespace(status=lambda: {'ready': True})
        self.inbox.retry_review()
        self.pump_until(lambda: not self.inbox.pending)
        self.assertEqual([call[2] for call in self.agent.calls], ['held phrase'])

    def test_other_session_failure_does_not_pause_current_delivery(self):
        self.inbox.add('current phrase')
        self.store.event('second', 'finished', {'status': 'failed', 'text': 'Other backend failed'})
        self.pump_until(lambda: not self.inbox.pending)
        self.assertEqual([call[1:3] for call in self.agent.calls], [('first', 'current phrase')])

    def test_scoped_pause_preserves_other_session_and_newer_request(self):
        self.inbox.add('before failure')
        failure = self.store.event('first', 'finished', {'status': 'failed', 'text': 'Backend failed'})
        self.inbox.add('after failure')
        self.session.sid = 'second'; self.inbox.add('other session')
        self.inbox.pause_pending('first', failure)
        self.assertEqual([item['state'] for item in self.saved()], ['review', 'queued'])
        self.assertEqual(self.saved('second')[0]['state'], 'queued')

    def test_worker_failure_boundary_does_not_hold_newer_post_failure_speech(self):
        self.inbox.add('before failure')
        self.store.event('first', 'finished', {'status': 'failed', 'text': 'Backend failed'})
        fresh_id = self.inbox.add('after failure')
        self.pump_until(lambda: self.inbox.counts() == {'queued': 0, 'sending': 0, 'review': 1})
        self.assertEqual(self.agent.calls, [('submit', 'first', 'after failure', fresh_id)])
        self.assertEqual(self.inbox.review_text(), 'before failure')

    def test_pause_during_preflight_prevents_later_submit(self):
        checking = threading.Event(); release = threading.Event()
        def status():
            checking.set(); release.wait(2)
            return {'ready': True}
        self.agent.local_client = SimpleNamespace(status=status)
        self.inbox.add('held phrase'); self.inbox.pump()
        try:
            self.assertTrue(checking.wait(1))
            self.inbox.pause_pending()
        finally:
            release.set()
        self.pump_until(lambda: self.inbox._worker is None)
        self.assertEqual(self.agent.calls, [])
        self.assertEqual(self.inbox.review_text(), 'held phrase')
    def test_starting_retries_same_id_without_reordering(self):
        self.agent.active=True;self.agent.starting=True
        rid=self.inbox.add('first');self.inbox.add('second')
        self.pump_until(lambda:len(self.agent.calls)>=2)
        self.assertTrue(all(x[3]==rid for x in self.agent.calls))
        self.agent.starting=False;self.pump_until(lambda:not self.inbox.pending)
        self.assertEqual(self.agent.calls[-1][2],'second')
        self.assertEqual(len(set(x[3] for x in self.agent.calls if x[2]=='first')),1)
    def test_idle_after_cancellation_submits_instead_of_steering(self):
        self.agent.cancelled.set();self.inbox.add('new task')
        self.pump_until(lambda:not self.inbox.pending)
        self.assertEqual(self.agent.calls[0][0],'submit');self.assertFalse(self.agent.cancelled.is_set())
    def test_idle_race_falls_back_to_submit_same_id(self):
        self.agent.active=True
        def became_idle(*args):
            self.agent.active=False;raise SteerUnavailable('Idle','idle')
        self.agent.steer=became_idle;rid=self.inbox.add('new request')
        self.pump_until(lambda:not self.inbox.pending)
        self.assertEqual(self.agent.calls,[('submit','first','new request',rid)])

    def test_confirmed_completed_turn_waits_then_submits_fresh_id_in_order(self):
        self.agent.active=True
        self.agent.error=SteerUnavailable('The previous turn finished', 'rejected_idle')
        old=self.inbox.add('first phrase');second=self.inbox.add('next phrase')
        self.pump_until(lambda:self.saved()[0]['id']!=old)
        fresh=self.saved()[0]['id']
        self.assertIn(self.saved()[0]['state'],('queued','sending'))
        self.assertEqual([x['text'] for x in self.saved()],['first phrase','next phrase'])
        for _ in range(8):
            self.inbox.pump();time.sleep(.005)
        self.assertEqual(self.agent.calls,[('steer','first','first phrase',old)])
        self.agent.error=None;self.agent.active=False
        self.pump_until(lambda:not self.inbox.pending)
        self.assertEqual(self.agent.calls,[('steer','first','first phrase',old),
            ('submit','first','first phrase',fresh),('steer','first','next phrase',second)])
        delivered=[value for kind,value in list(self.inbox.events.queue) if kind=='delivered']
        self.assertEqual(delivered[0]['previous_id'],old)
        self.assertEqual(delivered[0]['id'],fresh)
        self.assertFalse(self.inbox.review_text())

    def test_pause_wins_over_confirmed_completed_turn_rejection(self):
        self.agent.active=True;self.agent.block=threading.Event()
        self.agent.error=SteerUnavailable('The previous turn finished', 'rejected_idle')
        old=self.inbox.add('held');self.inbox.pump()
        self.pump_until(lambda:len(self.agent.calls)==1)
        self.inbox.pause_pending();self.agent.block.set()
        self.pump_until(lambda:self.inbox._worker is None)
        self.assertEqual(self.saved()[0]['id'],old)
        self.assertEqual(self.saved()[0]['state'],'review')
        self.agent.active=False;self.agent.error=None
        for _ in range(8):self.inbox.pump()
        self.assertEqual(len(self.agent.calls),1)

    def test_conversation_switch_wins_over_confirmed_rejection(self):
        self.agent.active=True;self.agent.block=threading.Event()
        self.agent.error=SteerUnavailable('The previous turn finished', 'rejected_idle')
        self.inbox.add('held');self.inbox.add('following');self.inbox.pump()
        self.pump_until(lambda:len(self.agent.calls)==1)
        self.session.sid='second';self.agent.block.set()
        self.pump_until(lambda:self.inbox._worker is None)
        self.assertEqual([item['state'] for item in self.saved()],['review','review'])
        self.assertEqual(len(self.agent.calls),1)

    def test_stop_wins_over_confirmed_rejection(self):
        self.agent.active=True;self.agent.block=threading.Event()
        self.agent.error=SteerUnavailable('The previous turn finished', 'rejected_idle')
        self.inbox.add('held');self.inbox.add('following');self.inbox.pump()
        self.pump_until(lambda:len(self.agent.calls)==1)
        self.inbox.stop();self.agent.block.set()
        self.pump_until(lambda:self.inbox._worker is None)
        self.assertEqual([item['state'] for item in self.saved()],['review','review'])
        self.assertEqual(len(self.agent.calls),1)

    def test_restart_of_known_rejected_retry_still_requires_explicit_review(self):
        self.agent.active=True
        self.agent.error=SteerUnavailable('The previous turn finished', 'rejected_idle')
        old=self.inbox.add('preserved phrase')
        self.pump_until(lambda:self.saved()[0]['id']!=old)
        self.inbox.stop()
        other=VoiceInbox(self.session)
        try:
            other.pump()
            self.assertEqual(other.review_text(),'preserved phrase')
            self.assertEqual(len(self.agent.calls),1)
        finally:other.close()

    def test_tools_finishing_waits_without_attempting_submit(self):
        self.agent.tool_lock.acquire();self.inbox.add('next task')
        self.inbox.pump();time.sleep(.01);self.inbox.pump()
        self.assertEqual(self.agent.calls,[])
        self.agent.tool_lock.release();self.pump_until(lambda:not self.inbox.pending)
    def test_uncertain_failure_preserves_all_existing_text_and_never_replays(self):
        self.agent.active=True;self.agent.error=AppError('Unknown network outcome')
        self.inbox.add('first');self.inbox.add('second')
        self.pump_until(lambda:bool(self.inbox.review_text()))
        self.assertEqual(self.inbox.review_text(),'first\nsecond')
        for _ in range(8):self.inbox.pump()
        self.assertEqual(len(self.agent.calls),1)
        self.assertEqual([x['state'] for x in self.saved()],['review','review'])
    def test_explicit_retry_uses_fresh_ids(self):
        self.agent.active=True;self.agent.error=AppError('Unknown outcome')
        old=self.inbox.add('retry');self.pump_until(lambda:bool(self.inbox.review_text()))
        self.agent.error=None;self.assertEqual(self.inbox.retry_review(),1)
        self.assertNotEqual(self.saved()[0]['id'],old)
        self.pump_until(lambda:not self.inbox.pending)
        self.assertEqual(len(self.agent.calls),2)
    def test_restart_restores_as_review_without_dispatch(self):
        self.inbox.add('saved queued');self.inbox.stop()
        other=VoiceInbox(self.session)
        try:
            self.assertEqual(other.review_text(),'saved queued')
            other.pump();self.assertEqual(self.agent.calls,[])
            self.assertEqual(self.saved()[0]['state'],'review')
        finally:other.close()
    def test_restart_sending_is_ambiguous_review(self):
        self.store.setting('terminal_voice_pending_second',json.dumps([{'id':'a'*32,'text':'maybe delivered','state':'sending'}]))
        self.session.sid='second';self.inbox.restore()
        self.assertEqual(self.inbox.review_text(),'maybe delivered')
        self.inbox.pump();self.assertEqual(self.agent.calls,[])
    def test_discard_removes_review_only(self):
        self.inbox.add('old');self.inbox.pause_pending();self.inbox.add('new')
        self.assertEqual(self.inbox.discard_review(),1)
        self.assertEqual([x['text'] for x in self.saved()],['new'])
    def test_pause_keeps_old_for_review_and_accepts_new_speech(self):
        self.inbox.add('old');self.assertEqual(self.inbox.pause_pending(),1)
        self.inbox.add('new');self.pump_until(lambda:bool(self.agent.calls) and len(self.saved())==1)
        self.assertEqual(self.inbox.review_text(),'old');self.assertEqual(self.agent.calls[0][2],'new')
    def test_session_switch_never_retargets_input(self):
        self.inbox.add('first session');self.session.sid='second';self.inbox.add('second session')
        self.pump_until(lambda:not self.inbox.pending)
        self.assertEqual(self.agent.calls[0][1:3],('second','second session'))
        self.assertEqual(self.saved('first')[0]['text'],'first session')
        self.session.sid='first';self.pump_until(lambda:not self.inbox.pending)
        self.assertEqual(self.agent.calls[1][1:3],('first','first session'))
    def test_pause_inflight_retry_cannot_restart_automatically(self):
        self.agent.active=True;self.agent.starting=True;self.agent.block=threading.Event()
        self.inbox.add('held');self.inbox.pump()
        self.pump_until(lambda:len(self.agent.calls)==1)
        self.inbox.pause_pending()
        self.agent.block.set();self.pump_until(lambda:self.inbox._worker is None)
        self.assertEqual(self.inbox.review_text(),'held');self.assertEqual(len(self.agent.calls),1)
    def test_close_waits_for_worker_and_preserves_failure_before_store_close(self):
        self.agent.active=True;self.agent.block=threading.Event();self.agent.error=AppError('closed connection')
        self.inbox.add('held');self.inbox.pump()
        self.pump_until(lambda:len(self.agent.calls)==1)
        closer=threading.Thread(target=self.inbox.close);closer.start();time.sleep(.02)
        self.assertTrue(closer.is_alive())
        self.agent.block.set();closer.join(1);self.assertFalse(closer.is_alive())
        self.assertEqual(self.saved()[0]['state'],'review')
    def test_delayed_idle_worker_pause_does_not_start_task(self):
        with self.agent.lock:
            self.inbox.add('old');self.inbox.pump();self.inbox.pause_pending()
        self.pump_until(lambda:self.inbox._worker is None)
        self.assertEqual(self.agent.calls,[])
        self.assertEqual(self.saved()[0]['state'],'review')
        self.inbox.add('new');self.pump_until(lambda:len(self.agent.calls)==1)
        self.assertEqual(self.agent.calls[0][2],'new')
    def test_delayed_idle_worker_stop_does_not_start_task(self):
        with self.agent.lock:
            self.inbox.add('old');self.inbox.add('following');self.inbox.pump();self.inbox.stop()
        self.pump_until(lambda:self.inbox._worker is None)
        self.assertEqual(self.agent.calls,[])
        self.assertEqual([item['state'] for item in self.saved()],['review','review'])
    def test_delayed_idle_worker_switch_does_not_start_old_or_new_task(self):
        with self.agent.lock:
            self.inbox.add('old');self.inbox.add('following');self.inbox.pump();self.session.sid='second'
        self.pump_until(lambda:self.inbox._worker is None)
        self.assertEqual(self.agent.calls,[])
        self.assertEqual([item['state'] for item in self.saved()],['review','review'])
    def test_idle_fallback_rechecks_pause_before_submit(self):
        self.agent.active=True
        def became_idle(*args):
            self.inbox.pause_pending();self.agent.active=False
            raise SteerUnavailable('Idle','idle')
        self.agent.steer=became_idle
        self.inbox.add('held');self.inbox.pump()
        self.pump_until(lambda:self.inbox._worker is None)
        self.assertEqual(self.agent.calls,[])
        self.assertEqual(self.saved()[0]['state'],'review')
    def test_idle_fallback_rechecks_switch_before_submit(self):
        self.agent.active=True
        def became_idle(*args):
            with self.agent.lock:
                self.agent.active=False;self.session.sid='second'
            raise SteerUnavailable('Idle','idle')
        self.agent.steer=became_idle
        self.inbox.add('held');self.inbox.pump()
        self.pump_until(lambda:self.inbox._worker is None)
        self.assertEqual(self.agent.calls,[])
        self.assertEqual(self.saved()[0]['state'],'review')

    def test_queue_limit_rejects_without_losing_saved_inputs(self):
        self.inbox.MAX_ITEMS=2;self.inbox.add('a');self.inbox.add('b')
        with self.assertRaises(AppError):self.inbox.add('c')
        self.assertEqual([x['text'] for x in self.saved()],['a','b'])
    def test_corrupt_saved_state_is_not_overwritten(self):
        self.store.setting('terminal_voice_pending_second','{broken')
        self.session.sid='second'
        with self.assertRaises(AppError):self.inbox.restore()
        self.assertEqual(self.store.setting('terminal_voice_pending_second'),'{broken')
    def test_close_is_idempotent_without_reopening_persistence(self):
        self.inbox.add('saved');self.inbox.close()
        original=self.store.setting
        self.store.setting=lambda *args: self.fail('Persistence touched after inbox closed')
        try:self.inbox.close()
        finally:self.store.setting=original

    def test_stop_blocks_dispatch_and_new_input(self):
        self.inbox.add('saved');self.inbox.stop();self.inbox.pump()
        self.assertEqual(self.agent.calls,[])
        with self.assertRaises(AppError):self.inbox.add('later')

if __name__=='__main__':unittest.main()
