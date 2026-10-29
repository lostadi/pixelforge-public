#!/usr/bin/env python3
"""Keep the local MCP host available without replaying any request."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def start_ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()[19]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--host', type=Path, required=True)
    args = parser.parse_args()
    directory = args.config.parent
    if os.getuid() == 0 or args.config.stat().st_uid != os.getuid():
        raise SystemExit('Supervisor must run as the owner of the private host configuration')
    lock_fd = os.open(directory / 'host-supervisor.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(75)
    config = json.loads(args.config.read_text())
    stopping = False
    child = None

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    state_path = directory / 'host-supervisor.json'
    state = dict(supervisor_pid=os.getpid(), supervisor_start_ticks=start_ticks(os.getpid()),
                 uid=os.getuid(), launches=0, host=str(args.host), config=str(args.config))

    def save():
        temp = state_path.with_suffix('.tmp')
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'w') as stream:
            json.dump(state, stream); stream.write('\n')
        temp.replace(state_path)

    while not stopping:
        log = directory / 'host-service.log'
        if log.exists() and log.stat().st_size > 4 * 1024 * 1024:
            log.replace(directory / 'host-service.previous.log')
        fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, 'ab') as output:
            child = subprocess.Popen([sys.executable, str(args.host), '--config', str(args.config)],
                                     cwd=config['root'], stdin=subprocess.DEVNULL,
                                     stdout=output, stderr=output, start_new_session=True)
            state.update(host_pid=child.pid, host_start_ticks=start_ticks(child.pid),
                         launches=state['launches'] + 1, running=True)
            save()
            exit_code = child.wait()
        state.update(running=False, last_exit_code=exit_code)
        save()
        if not stopping:
            time.sleep(2)
    os.close(lock_fd)


if __name__ == '__main__':
    main()
