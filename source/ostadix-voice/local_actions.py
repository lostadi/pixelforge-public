"""Host-side, explicitly approved actions for local model proposals."""
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

from agent import Workspace

def digest(action):
    return hashlib.sha256(json.dumps(action, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()

def snapshot(workspace, path):
    ws = Workspace(workspace)
    data = ws.read(path, limit=16000)
    if data['more']:
        raise ValueError('Action files must be at most 16000 bytes')
    actual = ws.fingerprint(path)
    if hashlib.sha256(data['content'].encode()).hexdigest() != actual:
        raise ValueError('File changed during read or is not valid UTF-8')
    return data['content'], actual

def prepare(value, proposal):
    ws = Workspace(value['workspace'])
    path = proposal['path']
    content, expected = snapshot(ws.root, path)
    if proposal['tool'] == 'replace':
        prior = next((event for event in reversed(value['events'])
                      if event.get('kind') == 'evidence'
                      and event.get('request') == {'tool':'read', 'path':path}), None)
        if (prior is None or prior['result'].get('more')
                or prior['result'].get('content') != content):
            raise ValueError('Read the complete current file before proposing an edit')
        old, new = proposal['old'], proposal['new']
        if content.count(old) != 1:
            raise ValueError('Replacement must match exactly once')
        updated = content.replace(old, new, 1)
        if len(updated.encode()) > 16000 or updated == content:
            raise ValueError('Edit is empty or exceeds the file size limit')
        action = dict(operation='write', path=path, expected=expected, content=updated,
                      diff=''.join(difflib.unified_diff(content.splitlines(True),
                                    updated.splitlines(True), fromfile=path, tofile=path)))
    elif proposal['tool'] == 'test':
        if not path.endswith('.py'):
            raise ValueError('Only Python test scripts are supported')
        action = dict(operation='test', path=path, expected=expected,
                      argv=[sys.executable, '-I', str(ws.root / path)],
                      timeout=15, preview=content,
                      warning='Runs this Python script with Termux account access; it is not OS-sandboxed.')
    else:
        raise ValueError('Unsupported action proposal')
    action['workspace'] = str(ws.root)
    return dict(action=action, digest=digest(action))

def syntax_check(path, content):
    if not path.endswith('.py'):
        return {'kind':'not_applicable'}
    try:
        ast.parse(content, filename=path)
        return {'kind':'python_syntax', 'passed':True}
    except SyntaxError as error:
        return {'kind':'python_syntax', 'passed':False,
                'line':error.lineno, 'error':error.msg}

def run_test(action, cancelled):
    if cancelled():
        raise ValueError('Cancelled before test execution')
    env = {k:v for k,v in os.environ.items() if k in
           ('PATH','HOME','PREFIX','TMPDIR','LANG','ANDROID_ROOT','ANDROID_DATA')}
    process = subprocess.Popen(action['argv'], cwd=action['workspace'], env=env,
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True)
    output = bytearray()
    reason = None
    started = time.monotonic()
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while selector.get_map():
                if cancelled() or time.monotonic()-started > action['timeout']:
                    reason = 'cancelled' if cancelled() else 'timeout'
                    break
                for key, _ in selector.select(.1):
                    data = os.read(key.fileobj.fileno(), 4096)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = 12000-len(output)
                    output.extend(data[:remaining])
                    if len(data) > remaining:
                        reason = 'output_limit'
                        break
                if reason:
                    break
            if reason is None:
                process.wait(timeout=1)
    finally:
        # Clean up the owned group even when the leader exited.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)
        process.stdout.close()
    return dict(kind='python_test', argv=action['argv'], exit_code=process.returncode,
                output=output.decode(errors='replace'), interruption=reason,
                passed=process.returncode == 0 and reason is None)

def execute(prepared, workspace, cancelled=lambda: False):
    action = prepared['action']
    if digest(action) != prepared['digest']:
        raise ValueError('Proposal changed after review')
    ws = Workspace(workspace)
    if str(ws.root) != action['workspace']:
        raise ValueError('Proposal workspace mismatch')
    content, expected = snapshot(ws.root, action['path'])
    if expected != action['expected']:
        raise ValueError('File changed since proposal; action refused')
    if cancelled():
        raise ValueError('Task cancelled')
    if action['operation'] == 'write':
        result = ws.write(action['path'], action['content'], action['expected'])
        result['verification'] = syntax_check(action['path'], action['content'])
        result['sha256'] = ws.fingerprint(action['path'])
        result.update(operation='write', applied=True, content=action['content'][:900],
                      content_truncated=len(action['content']) > 900,
                      next_step='Request the test tool for the appropriate Python test script; syntax is not a behavioral test.')
        return result
    if action['operation'] == 'test':
        # Exact argv is generated by the host, not taken from model output.
        expected_argv = [sys.executable, '-I', str(ws.root / action['path'])]
        if action['argv'] != expected_argv or action['timeout'] != 15:
            raise ValueError('Invalid test command')
        return run_test(action, cancelled)
    raise ValueError('Unsupported action')
