#!/usr/bin/env python3
"""Exercise automatic installed-c takeover in isolated real PTYs, without inference."""
import errno
import os
from pathlib import Path
import pty
import select
import signal
import sqlite3
import subprocess
import tempfile
import time


class Terminal:
    def __init__(self, shortcut, state, project=None):
        self.master, slave = pty.openpty()
        self.received = ''
        args = [str(shortcut), '--no-audio', '--no-local',
                '--state-dir', str(state)]
        if project is not None:
            args.append(str(project))
        try:
            self.process = subprocess.Popen(
                args, stdin=slave, stdout=slave, stderr=slave,
                start_new_session=True, env=dict(os.environ, TERM='xterm-256color'))
        except BaseException:
            os.close(self.master)
            raise
        finally:
            os.close(slave)

    def until(self, marker, after=0, timeout=25):
        deadline = time.monotonic() + timeout
        while marker not in self.received[after:] and time.monotonic() < deadline:
            if not select.select([self.master], [], [], .1)[0]:
                continue
            try:
                chunk = os.read(self.master, 65536)
            except OSError as error:
                if error.errno != errno.EIO:
                    raise
                break
            if not chunk:
                break
            self.received += chunk.decode(errors='replace')
        assert marker in self.received[after:], (
            f'Missing {marker!r}; exit={self.process.poll()}\n' + self.received[-6000:])

    def ready(self, project):
        self.until('Files: ' + project.name)
        self.until('\x1b[?2004h')  # The actual interactive input loop is ready.
        self.command('/status', 'Workspace: ' + str(project))
        assert self.process.poll() is None, self.received[-6000:]

    def command(self, command, marker):
        after = len(self.received)
        os.write(self.master, (command + '\r').encode())
        self.until(marker, after=after)

    def close(self):
        try:
            if self.process.poll() is None:
                # Only this probe's own child can be resumed or stopped here.
                self.process.send_signal(signal.SIGCONT)
                os.write(self.master, b'/quit\r')
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=5)
        finally:
            os.close(self.master)


def snapshot(state, expected_text):
    connection = sqlite3.connect((state / 'sessions.sqlite3').as_uri() + '?mode=ro', uri=True)
    try:
        selected = connection.execute("SELECT value FROM settings WHERE key='selected'").fetchone()[0]
        workspace = connection.execute('SELECT workspace FROM sessions WHERE id=?', (selected,)).fetchone()[0]
        events = connection.execute(
            "SELECT data FROM events WHERE session=? AND kind='tool_result'", (selected,)).fetchall()
        assert any(expected_text in row[0] for row in events), events
        return selected, workspace
    finally:
        connection.close()


def main():
    if os.geteuid() == 0:
        raise SystemExit('Run this probe as the normal Termux user, outside root.')
    shortcut = Path.home() / '.local/bin/c'
    with tempfile.TemporaryDirectory(prefix='voice-takeover-') as directory:
        project = Path(directory) / 'project'
        project.mkdir()
        state = Path(directory) / 'state'
        other_state = Path(directory) / 'other-state'
        actual_text = 'Automatic takeover preserves actual file result: 42.'
        (project / 'probe.txt').write_text(actual_text + '\n')
        terminals = []

        def start(store, workspace=None):
            terminal = Terminal(shortcut, store, workspace)
            terminals.append(terminal)
            terminal.ready(project)
            return terminal

        try:
            first = start(state, project)
            first.command('/inspect probe.txt', actual_text)
            original = snapshot(state, actual_text)
            other = start(other_state, project)

            second = start(state)  # No --stop and no project: recover the saved workspace.
            assert first.process.wait(timeout=10) == 0, first.received[-6000:]
            assert snapshot(state, actual_text) == original
            second.command('/logs', actual_text)
            print('PASS: plain c stopped the existing terminal, started here, and preserved '
                  'the workspace, selected conversation and inspected file history.', flush=True)

            second.process.send_signal(signal.SIGSTOP)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status = Path(f'/proc/{second.process.pid}/status').read_text()
                if any(line.startswith('State:\tT') for line in status.splitlines()):
                    break
                time.sleep(.05)
            else:
                raise AssertionError('The test terminal did not suspend before takeover.')
            third = start(state)
            assert second.process.wait(timeout=10) == 0, second.received[-6000:]
            assert snapshot(state, actual_text) == original
            third.command('/logs', actual_text)
            print('PASS: plain c resumed and stopped the suspended terminal, then restored '
                  'the same saved conversation and file history.', flush=True)

            assert other.process.poll() is None, other.received[-6000:]
            other.command('/status', 'State: ' + str(other_state))
            print('PASS: the separately scoped session stayed running and responsive.', flush=True)
            third.command('/status', 'Session: ' + original[0])
            os.write(third.master, b'/quit\r')
            assert third.process.wait(timeout=10) == 0
            os.write(other.master, b'/quit\r')
            assert other.process.wait(timeout=10) == 0
        finally:
            for terminal in reversed(terminals):
                terminal.close()
        print('No real user session, cloud inference, audio or model request was involved.', flush=True)


if __name__ == '__main__':
    import sys
    if '--live' not in sys.argv:
        raise SystemExit('Live device/model integration is opt-in. Read this script, then pass --live explicitly.')
    sys.argv.remove('--live')
    main()
