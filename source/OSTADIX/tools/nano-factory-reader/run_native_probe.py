#!/usr/bin/env python3
"""Run one owned JNI probe with a deadline and observed resource bounds."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
from materialize import android_environment
from android_pidfd import open_pidfd, send_pidfd_signal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jar', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--evidence-prefix', type=Path, required=True)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--deadline-seconds', type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.deadline_seconds <= 120:
        raise ValueError('Probe deadline must be between 1 and 120 seconds')
    if args.jar.stat().st_mode & 0o222:
        raise ValueError('Android requires the staged probe DEX/JAR to be read-only')
    env = android_environment(args.jar)
    library_dir = '/data/local/tmp/ostadix-ai-audit-20260915/aicore-native'
    entrypoint = ('org.ostadix.nanoloader.SyntheticGenerationProbe' if args.generate
                  else 'org.ostadix.nanoloader.LocalModelProbe')
    arguments = ['/system/bin/app_process', '-Djava.library.path=' + library_dir,
                 '/system/bin', entrypoint, library_dir, str(args.manifest.resolve())]
    command = ['/system/bin/su', '-p', '10402', '-c', shlex.join(arguments)]
    stdout_path = Path(str(args.evidence_prefix) + '.stdout')
    stderr_path = Path(str(args.evidence_prefix) + '.stderr')
    if stdout_path.exists() or stderr_path.exists():
        raise ValueError('Evidence prefix already used')
    pidfd = None
    java_pid = None
    snapshots = []
    stop_reason = None
    started = time.monotonic()
    with stdout_path.open('w') as stdout, stderr_path.open('w') as stderr:
        process = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                for line in stdout_path.read_text().splitlines():
                    try:
                        event = json.loads(line)
                        candidate = event.get('pid')
                        if (pidfd is None and type(candidate) is int and candidate > 0
                                and event.get('uid') == 10402):
                            pidfd = open_pidfd(candidate)
                            java_pid = candidate
                    except (ValueError, OSError):
                        pass
                memory = {}
                for line in Path('/proc/meminfo').read_text().splitlines():
                    name, _, value = line.partition(':')
                    if name in ('MemAvailable', 'SwapFree'):
                        memory[name + '_kB'] = int(value.split()[0])
                if java_pid:
                    try:
                        for line in Path(f'/proc/{java_pid}/status').read_text().splitlines():
                            name, _, value = line.partition(':')
                            if name in ('VmRSS', 'VmHWM'):
                                memory[name + '_kB'] = int(value.split()[0])
                    except OSError:
                        pass
                thermal_status = None
                try:
                    thermal = subprocess.run(['/system/bin/dumpsys', 'thermalservice'],
                                             capture_output=True, text=True, timeout=2)
                    for line in thermal.stdout.splitlines():
                        if line.startswith('Thermal Status:'):
                            thermal_status = int(line.split(':')[1].strip())
                except (subprocess.TimeoutExpired, ValueError):
                    pass
                snapshots.append(dict(elapsed_seconds=round(elapsed, 3),
                                      thermal_status=thermal_status, **memory))
                if elapsed >= args.deadline_seconds:
                    stop_reason = 'host_deadline'
                elif memory.get('MemAvailable_kB', 1024 * 1024) < 512 * 1024:
                    stop_reason = 'host_low_memory_guard'
                elif thermal_status is not None and thermal_status >= 3:
                    stop_reason = 'host_thermal_guard'
                if stop_reason:
                    if pidfd is not None:
                        try:
                            send_pidfd_signal(pidfd, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    if process.poll() is None:
                        process.kill()
                    break
                time.sleep(0.5)
            exit_code = process.wait(timeout=10)
        finally:
            if process.poll() is None:
                if pidfd is not None:
                    try:
                        send_pidfd_signal(pidfd, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.kill()
                process.wait(timeout=10)
            if pidfd is not None:
                os.close(pidfd)
    report = {'command': command, 'java_pid': java_pid, 'exit_code': exit_code,
              'stop_reason': stop_reason, 'elapsed_ms': int((time.monotonic()-started)*1000),
              'resource_snapshots': snapshots, 'stdout_file': str(stdout_path),
              'stderr_file': str(stderr_path), 'automatic_retry': False}
    Path(str(args.evidence_prefix) + '.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'resource_snapshots'}, indent=2))
    print(stdout_path.read_text())


if __name__ == '__main__':
    main()
