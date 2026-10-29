"""Exercise the installed c command in a real PTY; no model or audio calls."""
import json
import os
from pathlib import Path
import pty
import select
import sqlite3
import subprocess
import tempfile
import time


def main():
    if os.geteuid() == 0:
        raise SystemExit('Run this probe as the normal Termux user, outside root.')
    shortcut = Path.home()/'.local/bin/c'
    with tempfile.TemporaryDirectory(prefix='voice-launch-') as directory:
        project = Path(directory)/'project'; project.mkdir()
        state = Path(directory)/'state'
        (project/'probe.txt').write_text('Persistent launcher probe: 42.\n')
        before = None
        for attempt in range(2):
            master, slave = pty.openpty()
            args = [str(shortcut), '--no-audio', '--no-local', '--state-dir', str(state)]
            if attempt == 0:
                args.append(str(project))
            child = subprocess.Popen(args, stdin=slave, stdout=slave, stderr=slave, start_new_session=True)
            os.close(slave)
            received = ''

            def until(marker, timeout=25):
                nonlocal received
                deadline = time.monotonic()+timeout
                while marker not in received and time.monotonic() < deadline:
                    if select.select([master], [], [], .2)[0]:
                        try:
                            data = os.read(master, 65536)
                        except OSError:
                            break
                        if not data:
                            break
                        received += data.decode(errors='replace')
                assert marker in received, received[-6000:]

            try:
                until('Files: '+project.name)
                until('\x1b[?2004h')
                os.write(master, b'/status\n')
                until('Workspace: '+str(project))
                os.write(master, b'/ostadix\n')
                until('OSTADIX connected · ')
                until(' tools · /o SOURCE')
                if attempt == 0:
                    os.write(master, b'/inspect probe.txt\n')
                    until('Persistent launcher probe: 42.')
                stopped = subprocess.run([str(shortcut), '--stop', '--state-dir', str(state)],
                                         capture_output=True, text=True, timeout=25)
                assert stopped.returncode == 0, stopped.stdout + stopped.stderr
                assert 'Stopped Ostadix Voice.' in stopped.stdout, stopped.stdout
                print(stopped.stdout.strip(), flush=True)
                assert child.wait(timeout=10) == 0
                stopped_again = subprocess.run([str(shortcut), '--stop', '--state-dir', str(state)],
                                               capture_output=True, text=True, timeout=10)
                assert stopped_again.returncode == 0, stopped_again.stdout + stopped_again.stderr
                assert 'No Ostadix Voice session is running.' in stopped_again.stdout
            finally:
                if child.poll() is None:
                    child.terminate()
                    child.wait(timeout=10)
                os.close(master)
            db = sqlite3.connect(state/'sessions.sqlite3')
            try:
                selected = db.execute("SELECT value FROM settings WHERE key='selected'").fetchone()[0]
                workspace = db.execute('SELECT workspace FROM sessions WHERE id=?', (selected,)).fetchone()[0]
                events = db.execute('SELECT data FROM events WHERE session=?', (selected,)).fetchall()
                assert any('Persistent launcher probe: 42.' in row[0] for row in events)
                status = json.loads(db.execute(
                    "SELECT data FROM events WHERE session=? AND kind='ostadix_status' ORDER BY seq DESC LIMIT 1",
                    (selected,)).fetchone()[0])
                assert status['connected'] is True and status['tool_count'] > 0, status
                assert status['primary_tool'] == 'o_execute', status
                snapshot = (selected, workspace)
                if before is not None:
                    assert snapshot == before
                before = snapshot
            finally:
                db.close()
            print('PASS: installed c '+('launch' if attempt == 0 else 'restart')+
                  ', external --stop, live MCP identity, selected project and saved actual file result.', flush=True)
        print('No cloud inference or audio calls were made.', flush=True)


if __name__ == '__main__':
    import sys
    if '--live' not in sys.argv:
        raise SystemExit('Live device/model integration is opt-in. Read this script, then pass --live explicitly.')
    sys.argv.remove('--live')
    main()
