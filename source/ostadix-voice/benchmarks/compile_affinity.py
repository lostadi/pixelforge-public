#!/usr/bin/env python3
"""Opt-in C compilation comparison; never changes the parent's CPU affinity.

Run from an ordinary Termux shell with Termux visible, for example:
  python benchmarks/compile_affinity.py --source /absolute/test.c \
    --compiler /data/data/com.termux/files/usr/bin/clang --flag=-O3 \
    --rounds 6 --output /absolute/new-report

Use the original source and flags to compare a previous workload. The source is
compiled from its parent directory to preserve relative includes; a copy and its
hash are saved.
Timing begins immediately before the child sends launch metadata and execs the
compiler, ending when waitid observes exit. It includes exec/metadata/exit-notice
overhead, excludes Python startup, and is wall time, not CPU time. Every compiler
gets the same environment/cwd/arguments except its unique output path. Included
headers/libraries are not snapshotted: keep them unchanged during the experiment.
"""
import argparse
import datetime
import errno
import hashlib
import itertools
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import threading
import time


CONDITIONS = {'default': None, 'cpu2': 2, 'cpu7': 7}


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return {'unavailable_errno': exc.errno}


def snapshot(reference_pid=None):
    cpuset = read('/proc/self/cpuset')
    status = read('/proc/self/status')
    allowed = next((line.split(':', 1)[1].strip() for line in status.splitlines()
                    if line.startswith('Cpus_allowed_list:')), None) if isinstance(status, str) else status
    group = Path('/dev/cpuset') / cpuset.lstrip('/') if isinstance(cpuset, str) else None
    result = {'uid': os.getuid(), 'euid': os.geteuid(), 'gid': os.getgid(),
            'selinux': read('/proc/self/attr/current'), 'cpuset': cpuset,
            'cgroup': read('/proc/self/cgroup'),
            'cpuset_cpus': read(group / 'cpus') if group else None,
            'cpuset_effective_cpus': read(group / 'effective_cpus') if group else None,
            'online_cpus': read('/sys/devices/system/cpu/online'),
            'affinity': sorted(os.sched_getaffinity(0)), 'allowed_list': allowed}
    if reference_pid is not None:
        stat = read(f'/proc/{reference_pid}/stat')
        result['reference'] = {'pid': reference_pid,
                               'cpuset': read(f'/proc/{reference_pid}/cpuset'),
                               'cgroup': read(f'/proc/{reference_pid}/cgroup'),
                               'start_ticks': stat.rsplit(')', 1)[-1].split()[19] if isinstance(stat, str) else stat}
    return result


def reference_matches(value):
    reference = value.get('reference')
    return reference is None or (isinstance(reference['cpuset'], str)
                                  and reference['cpuset'] == value['cpuset']
                                  and isinstance(reference.get('cgroup'), str)
                                  and reference['cgroup'] == value.get('cgroup'))


def policy(snapshot_value):
    return {key: value for key, value in snapshot_value.items()
            if key not in ('affinity', 'allowed_list')}


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def file_identity(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def elf_output(path):
    try:
        with Path(path).open('rb') as stream:
            return stream.read(4) == b'\x7fELF' and os.fstat(stream.fileno()).st_size >= 52
    except OSError:
        return False


def compile_once(command, cpu, output, timeout=30, reference_pid=None):
    """Fork, restrict only that child, exec directly; retain distinct failures."""
    before = snapshot(reference_pid)
    reader, writer = os.pipe()
    stdout = os.open(str(output) + '.stdout', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    stderr = os.open(str(output) + '.stderr', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    child = os.fork()
    if child == 0:
        phase = 'launch'
        try:
            os.close(reader)
            os.setpgid(0, 0)
            null = os.open(os.devnull, os.O_RDONLY)
            for source, target in ((null, 0), (stdout, 1), (stderr, 2)):
                os.dup2(source, target)
            for fd in (null, stdout, stderr):
                if fd > 2:
                    os.close(fd)
            inherited = snapshot(reference_pid)
            if cpu is not None:
                phase = 'affinity'
                os.sched_setaffinity(0, {cpu})
            launch = snapshot(reference_pid)
            if cpu is not None and launch['affinity'] != [cpu]:
                raise OSError(errno.EINVAL, 'Requested CPU was not granted')
            metadata = {'inherited': inherited, 'launch': launch}
            phase = 'exec'
            metadata['start_ns'] = time.monotonic_ns()
            os.write(writer, (json.dumps(metadata) + '\n').encode())
            os.execv(command[0], command)
        except BaseException as exc:
            failure = {'error': {'phase': phase, 'errno': getattr(exc, 'errno', None),
                                  'message': str(exc)}}
            if 'inherited' in locals():
                failure['inherited'] = inherited
            os.write(writer, (json.dumps(failure) + '\n').encode())
        finally:
            os._exit(127)
    os.close(writer)
    os.close(stdout)
    os.close(stderr)
    # Establish the group from both sides, including when a timeout is immediate.
    try:
        os.setpgid(child, child)
    except OSError as exc:
        if exc.errno not in (errno.EACCES, errno.ESRCH):
            raise
    done, ended = threading.Event(), []

    def observe_exit():
        # Keep the zombie unreaped until after timeout handling: its PID cannot
        # be reused while the parent might still signal this process group.
        os.waitid(os.P_PID, child, os.WEXITED | os.WNOWAIT)
        ended.append(time.monotonic_ns())
        done.set()

    waiter = threading.Thread(target=observe_exit, daemon=True)
    waiter.start()
    interrupted = False
    try:
        timed_out = not done.wait(timeout)
    except KeyboardInterrupt:
        interrupted, timed_out = True, False
    if timed_out or interrupted:
        try:
            os.killpg(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
    done.wait()
    _, status = os.waitpid(child, 0)
    waiter.join()
    with os.fdopen(reader) as stream:
        messages = [json.loads(line) for line in stream]
    after = snapshot(reference_pid)
    result = {'before': before, 'after': after, 'returncode': os.waitstatus_to_exitcode(status),
              'timed_out': timed_out, 'interrupted': interrupted}
    for message in messages:
        result.update(message)
    result['context_changed'] = (before != after or
        ('inherited' in result and before != result['inherited']) or
        ('launch' in result and policy(before) != policy(result['launch'])) or
        any(not reference_matches(value) for value in
            (before, after, result.get('inherited', before), result.get('launch', before))))
    result['seconds'] = (ended[0] - result['start_ns']) / 1e9 if 'start_ns' in result else None
    result['elf_output'] = elf_output(output)
    result['success'] = (not timed_out and not interrupted and not result.get('error')
                         and result['returncode'] == 0 and result['elf_output'])
    return result


def summarize(records):
    result = {}
    for condition in CONDITIONS:
        rows = [row for row in records if row['condition'] == condition and not row['warmup']]
        values = [row['seconds'] for row in rows if row['success'] and not row['context_changed']]
        result[condition] = {'measured_runs': len(rows), 'usable_runs': len(values),
                             'failures': sum(not row['success'] for row in rows),
                             'changed_context_runs': sum(row['context_changed'] for row in rows),
                             'median_seconds': statistics.median(values) if values else None,
                             'min_seconds': min(values) if values else None,
                             'max_seconds': max(values) if values else None}
    return result


def schedule(rounds):
    yield from ((0, True, condition) for condition in CONDITIONS)
    orders = list(itertools.permutations(CONDITIONS))
    for index in range(rounds):
        yield from ((index + 1, False, condition) for condition in orders[index % len(orders)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--compiler', required=True, type=Path, help='Existing compiler executable; no ccache wrapper')
    parser.add_argument('--flag', required=True, action='append', help='Repeat explicit compiler flags, e.g. --flag=-O3')
    parser.add_argument('--rounds', type=int, default=6, help='Positive multiple of six; default six')
    parser.add_argument('--output', required=True, type=Path, help='New directory for all raw runs and artifacts')
    parser.add_argument('--reference-pid', type=int, help='Require the same cpuset and cgroups as this existing Termux shell throughout observed snapshots')
    args = parser.parse_args(argv)
    if os.getuid() == 0 or os.geteuid() == 0:
        parser.error('Run from your ordinary Termux shell, not root or su.')
    if args.rounds <= 0 or args.rounds % 6:
        parser.error('--rounds must be a positive multiple of six to balance the order.')
    if not args.compiler.is_absolute() or not args.compiler.is_file() or not os.access(args.compiler, os.X_OK):
        parser.error('--compiler must name an existing, executable absolute path.')
    source, compiler = args.source.resolve(), args.compiler.resolve()
    if not source.is_file() or source.suffix.lower() != '.c':
        parser.error('--source must name an existing C source file.')
    if 'ccache' in compiler.name or any(flag == '-o' or flag.startswith('-o') or flag.startswith('@') for flag in args.flag):
        parser.error('Use a compiler directly; output flags and response files are reserved.')
    report = args.output.resolve()
    os.chdir(source.parent)
    try:
        report.mkdir(mode=0o700)
    except OSError as exc:
        parser.error(f'Cannot create a new report directory: {exc}')
    source_hash, compiler_hash = sha256(source), sha256(compiler)
    compiler_identity = file_identity(compiler)
    (report / 'source.c').write_bytes(source.read_bytes())
    if sha256(report / 'source.c') != source_hash:
        parser.error('Source changed while making its snapshot; rerun with a new report directory.')
    version = subprocess.run([str(compiler), '--version'], capture_output=True, text=True, timeout=10)
    manifest = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'pid': os.getpid(), 'parent_pid': os.getppid(),
                'source': str(source), 'source_sha256': source_hash,
                'compiler': str(compiler), 'compiler_sha256': compiler_hash,
                'compiler_version': version.stdout + version.stderr, 'version_exit': version.returncode,
                'flags': args.flag, 'cwd': str(Path.cwd()), 'context': snapshot(args.reference_pid),
                'rounds': args.rounds, 'timeout_seconds': 30,
                'timing': 'Child monotonic time immediately before metadata write/exec through waitid exit observation; includes this overhead, excludes Python startup.',
                'comparison': 'Medians exclude warmups, failures, and runs with observed context changes. Transient changes between snapshots may escape detection.'}
    (report / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    records, aborted = [], None
    with (report / 'runs.jsonl').open('x') as log:
        for index, (round_number, warmup, condition) in enumerate(schedule(args.rounds), 1):
            if sha256(source) != source_hash or file_identity(compiler) != compiler_identity:
                aborted = 'Source or compiler changed; no further compilations were launched.'
                break
            output = report / f'{index:03d}-{condition}.elf'
            command = [str(compiler), *args.flag, str(source), '-o', str(output)]
            row = compile_once(command, CONDITIONS[condition], output, reference_pid=args.reference_pid)
            row.update({'index': index, 'round': round_number, 'warmup': warmup,
                        'condition': condition, 'command': command, 'output': str(output)})
            if sha256(source) != source_hash or file_identity(compiler) != compiler_identity:
                row['success'] = False
                row['input_changed'] = True
                aborted = 'Source or compiler changed during compilation; this run is excluded.'
            records.append(row)
            log.write(json.dumps(row) + '\n')
            log.flush()
            elapsed = f'{row["seconds"]:.6f}s' if row['seconds'] is not None else 'untimed'
            label = 'warmup' if warmup else f'round {round_number}'
            status = 'OK' if row['success'] else 'FAILED'
            if row.get('error'):
                status += f' {row["error"]["phase"]} errno={row["error"]["errno"]}'
            if row['timed_out']:
                status += ' timeout'
            if row['context_changed']:
                status += ' context changed (excluded)'
            print(f'{label:8} {condition:7} {elapsed:12} {status}', flush=True)
            if row['interrupted']:
                aborted = 'Interrupted; the active compiler process group was stopped.'
            if aborted:
                break
    final_compiler_hash = sha256(compiler)
    if final_compiler_hash != compiler_hash:
        aborted = 'Compiler content changed during the experiment; the comparison is invalid.'
        for row in records:
            row['success'], row['input_changed'] = False, True
        (report / 'runs.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in records))
    summary = {'conditions': summarize(records), 'aborted': aborted,
               'compiler_final_sha256': final_compiler_hash}
    (report / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for condition, values in summary['conditions'].items():
        median = values['median_seconds']
        duration = f'{median:.6f}s' if median is not None else 'unavailable'
        print(f'{condition}: median {duration}, {values["usable_runs"]}/{values["measured_runs"]} usable measured runs')
    if aborted:
        print(aborted)
    print(f'Raw evidence: {report}')
    return 1 if aborted or any(row['failures'] or row['changed_context_runs'] for row in summary['conditions'].values()) else 0


if __name__ == '__main__':
    raise SystemExit(main())
