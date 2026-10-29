"""Real stdio transport checks; no model, network, or paid calls."""

import concurrent.futures
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mcp_transport import MCPClient, MCPError


SERVER = r'''
import json, os, signal, subprocess, sys, threading, time
lock = threading.Lock()
initialized = False
count = 0
children = []

def emit(message):
    with lock:
        print(json.dumps({'jsonrpc':'2.0', **message}), flush=True)

def record(message):
    with open(sys.argv[1], 'a') as log:
        log.write(json.dumps(message) + '\n')

def delayed(ident, value, delay):
    time.sleep(delay)
    emit({'id':ident, 'result':{'value':value}})

for line in sys.stdin:
    message = json.loads(line)
    record(message)
    method = message.get('method')
    ident = message.get('id')
    if method == 'initialize':
        emit({'id':ident, 'result':{'protocolVersion':'2025-03-26', 'capabilities':{'tools':{}}, 'serverInfo':{'name':'fixture','version':'1'}}})
    elif method == 'notifications/initialized':
        initialized = True
    elif method == 'tools/list':
        emit({'id':ident, 'result':{'tools':[{'name':'counter','inputSchema':{'type':'object'}}], 'initialized':initialized}})
    elif method == 'tools/call':
        count += 1
        name = message['params']['name']
        args = message['params'].get('arguments', {})
        if name == 'counter':
            emit({'id':ident, 'result':{'count':count, 'pid':os.getpid()}})
        elif name == 'error':
            emit({'id':ident, 'error':{'code':-32000,'message':'Fixture rejected request; token=private-value'}})
        elif name == 'hang':
            pass
        elif name == 'drop':
            os._exit(0)
        elif name == 'delay':
            threading.Thread(target=delayed, args=(ident, args['value'], args.get('delay', .02)), daemon=True).start()
        elif name == 'server_request':
            emit({'id':'server-request-1', 'method':'sampling/createMessage','params':{}})
            emit({'id':ident, 'result':{'requested':True}})
        elif name == 'stderr':
            sys.stderr.write('Authorization: Bearer super-private-value\n' * 10000)
            sys.stderr.flush()
            emit({'id':ident, 'result':{'written':True}})
        elif name == 'invalid':
            print('{not-json}', flush=True)
        elif name == 'oversized':
            sys.stdout.write('x' * (16 * 1024 * 1024 + 1))
            sys.stdout.flush()
        elif name == 'stop_reading':
            emit({'id':ident, 'result':{'stopped':True}})
            time.sleep(30)
        elif name == 'detached':
            child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True)
            children.append(child)
            emit({'id':ident, 'result':{'pid':child.pid}})

# Match native MCP shutdown: detached jobs are owned by this server and cleaned
# up on stdin EOF, not by signals to the server's original process group.
for child in children:
    time.sleep(.05)
    os.killpg(child.pid, signal.SIGTERM)
    child.wait(timeout=2)
    record({'cleanup':child.pid, 'returncode':child.returncode})
'''


def eventually(check, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(.01)
    raise AssertionError('Timed out waiting for stdio fixture')


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.log = self.root / 'requests.jsonl'
        self.client = MCPClient([sys.executable, '-u', '-c', SERVER, str(self.log)], self.root, os.environ.copy())

    def tearDown(self):
        self.client.close()
        self.temp.cleanup()

    def messages(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def tool(self, name, **kwargs):
        return self.client.call('tools/call', {'name':name, 'arguments':{}}, **kwargs)

    def test_initialize_list_and_persistent_process_state(self):
        self.assertTrue(self.client.alive)
        self.assertEqual(self.client.info['serverInfo']['name'], 'fixture')
        listed = self.client.call('tools/list', {})
        self.assertTrue(listed['initialized'])
        self.assertEqual(listed['tools'][0]['name'], 'counter')
        first = self.tool('counter')
        second = self.tool('counter')
        self.assertEqual((first['count'], second['count']), (1, 2))
        self.assertEqual(first['pid'], second['pid'])
        init = self.messages()[0]['params']
        self.assertEqual(init, {'protocolVersion':'2025-03-26', 'capabilities':{}, 'clientInfo':{'name':'ostadix_voice','version':'0.1'}})

    def test_concurrent_requests_dispatch_out_of_order_responses(self):
        def call(value):
            return self.client.call('tools/call', {'name':'delay', 'arguments':{'value':value, 'delay':(10-value)/100}})['value']
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            self.assertEqual(list(pool.map(call, range(10))), list(range(10)))

    def test_json_rpc_error_does_not_end_connection(self):
        with self.assertRaisesRegex(MCPError, r'MCP error -32000.*Fixture rejected') as error:
            self.tool('error')
        self.assertIn('outcome may be unknown', str(error.exception))
        self.assertNotIn('private-value', str(error.exception))
        self.assertTrue(self.client.alive)
        self.assertEqual(self.tool('counter')['count'], 2)

    def test_disconnect_reports_uncertainty_and_never_replays(self):
        with self.assertRaisesRegex(MCPError, 'outcome may be unknown'):
            self.tool('drop', timeout=2)
        self.assertFalse(self.client.alive)
        with self.assertRaises(MCPError):
            self.tool('counter')
        self.assertEqual(len([m for m in self.messages() if m.get('method') == 'tools/call']), 1)

    def test_timeout_sends_cancel_and_never_replays(self):
        before = time.monotonic()
        with self.assertRaisesRegex(MCPError, 'timed out.*outcome may be unknown'):
            self.tool('hang', timeout=.12)
        self.assertLess(time.monotonic() - before, 1)
        eventually(lambda: any(m.get('method') == 'notifications/cancelled' for m in self.messages()))
        self.assertEqual(self.tool('counter')['count'], 2)
        messages = self.messages()
        hanging = next(m for m in messages if m.get('params', {}).get('name') == 'hang')
        cancellation = next(m for m in messages if m.get('method') == 'notifications/cancelled')
        self.assertEqual(hanging['id'], cancellation['params']['requestId'])
        self.assertEqual(len([m for m in messages if m.get('params', {}).get('name') == 'hang']), 1)

    def test_user_cancel_sends_notification_and_preserves_connection(self):
        cancel = threading.Event()
        timer = threading.Timer(.1, cancel.set)
        timer.start()
        try:
            with self.assertRaisesRegex(MCPError, 'cancelled.*outcome may be unknown'):
                self.tool('hang', cancel_event=cancel)
        finally:
            timer.join()
        eventually(lambda: any(m.get('method') == 'notifications/cancelled' for m in self.messages()))
        self.assertTrue(self.client.alive)
        self.assertEqual(self.tool('counter')['count'], 2)

    def test_cancel_before_send_does_not_execute(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaisesRegex(MCPError, 'cancelled'):
            self.tool('counter', cancel_event=cancel)
        self.assertEqual(self.tool('counter')['count'], 1)

    def test_server_initiated_requests_are_rejected(self):
        self.assertTrue(self.tool('server_request')['requested'])
        eventually(lambda: any(m.get('id') == 'server-request-1' for m in self.messages()))
        reply = next(m for m in self.messages() if m.get('id') == 'server-request-1')
        self.assertEqual(reply['error']['code'], -32601)

    def test_stderr_is_drained_without_disclosing_secrets(self):
        self.assertTrue(self.tool('stderr')['written'])
        eventually(lambda: bool(self.client.stderr_tail))
        self.assertIn('bytes of MCP stderr suppressed', self.client.stderr_tail)
        self.assertNotIn('private-value', self.client.stderr_tail)
        self.assertLess(len(self.client.stderr_tail), 100)

    def test_invalid_frame_fails_pending_request(self):
        with self.assertRaisesRegex(MCPError, 'invalid or oversized frame.*outcome may be unknown'):
            self.tool('invalid', timeout=2)
        self.assertFalse(self.client.alive)

    def test_oversized_response_is_bounded_and_disconnects(self):
        with self.assertRaisesRegex(MCPError, 'oversized frame.*outcome may be unknown'):
            self.tool('oversized', timeout=5)
        self.assertFalse(self.client.alive)

    def test_oversized_request_is_not_transmitted(self):
        with patch('mcp_transport.MAX_FRAME', 4096):
            with self.assertRaisesRegex(MCPError, '16 MiB frame limit'):
                self.client.call('tools/call', {'name':'counter', 'arguments':{'large':'x' * 5000}})
        self.assertEqual(self.tool('counter')['count'], 1)

    def test_blocked_stdin_obeys_deadline_without_replaying_partial_frame(self):
        self.assertTrue(self.tool('stop_reading')['stopped'])
        before = time.monotonic()
        with self.assertRaisesRegex(MCPError, 'timed out while sending.*outcome may be unknown'):
            self.client.call('tools/call', {'name':'counter', 'arguments':{'large':'x' * (2 * 1024 * 1024)}}, timeout=.12)
        self.assertLess(time.monotonic() - before, 1)
        self.assertFalse(self.client.alive)
        self.assertEqual(len([m for m in self.messages() if m.get('method') == 'tools/call']), 1)

    def test_close_wakes_pending_and_terminates_owned_process(self):
        errors = []
        def waiting():
            try:
                self.tool('hang', timeout=30)
            except MCPError as error:
                errors.append(str(error))
        worker = threading.Thread(target=waiting)
        worker.start()
        eventually(lambda: any(m.get('params', {}).get('name') == 'hang' for m in self.messages()))
        before = time.monotonic()
        self.client.close()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic() - before, 3)
        self.assertIsNotNone(self.client.process.poll())
        self.assertFalse(self.client.alive)
        self.assertIn('outcome may be unknown', errors[0])
        self.client.close()  # Idempotent.

    def test_close_allows_eof_cleanup_of_detached_jobs(self):
        child = self.tool('detached')['pid']
        try:
            self.assertEqual(os.getpgid(child), child)
            self.assertNotEqual(os.getpgid(child), self.client.process.pid)
            self.client.close()
            self.assertTrue(self.client.shutdown_graceful)
            self.assertEqual(self.client.process.returncode, 0)
            self.assertTrue(any(message.get('cleanup') == child for message in self.messages()))
            with self.assertRaises(ProcessLookupError):
                os.kill(child, 0)
        finally:
            try:
                os.killpg(child, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_close_interrupts_blocked_writer_and_bounds_forced_shutdown(self):
        self.assertTrue(self.tool('stop_reading')['stopped'])
        errors = []
        def sending():
            try:
                self.client.call('tools/call', {'name':'counter', 'arguments':{'large':'x' * (2 * 1024 * 1024)}}, timeout=30)
            except MCPError as error:
                errors.append(str(error))
        worker = threading.Thread(target=sending)
        worker.start()
        eventually(lambda: self.client._send_lock.locked())
        before = time.monotonic()
        self.client.close()
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic() - before, 4)
        self.assertFalse(self.client.shutdown_graceful)
        self.assertIsNotNone(self.client.process.poll())
        self.assertIn('outcome may be unknown', errors[0])


if __name__ == '__main__':
    unittest.main()
