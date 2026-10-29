#!/usr/bin/env python3
"""Bounded deterministic parser/sanitizer and stubbed cleanup stress; no NPU calls.

Requires a Linux C++17 compiler. The optional real-Model cleanup fixture requires
Termux clang and rooted access to Android; its replacement entry point never
loads vendor libraries or calls the original native main.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import tempfile
import time

KIT = Path(__file__).resolve().parent
MIB = 1024 * 1024
DEFAULT_SEED = 0x504958454C4E5055
ANDROID_CXX = Path('/data/data/com.termux/files/usr/bin/clang++')


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(MIB), b''):
            result.update(block)
    return result.hexdigest()


def stress(args):
    started = time.monotonic()
    deadline = started + args.timeout
    report = {'schema': 'pixelforge.native-protocol-stress.v1',
              'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'seed': args.seed, 'seed_hex': hex(args.seed),
              'iterations_per_variant': args.iterations,
              'vendor_initializations': 0, 'npu_invocations': 0,
              'hardware_touched': False, 'runs': [], 'failures': [],
              'sanitizers': {'requested': ['address', 'undefined'], 'status': 'not-run',
                             'address': {'status': 'not-run', 'probes': []},
                             'undefined': {'status': 'not-run'}},
              'limits': {'timeout_seconds': args.timeout, 'build_bytes': 16 * MIB}}
    compiler = shutil.which(args.cxx)
    report['compiler'] = {'path': compiler}
    build_peak = 0

    def run(command, *, environment=None, maximum=20):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Native stress whole-run deadline expired')
        with subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, start_new_session=True) as child:
            try:
                stdout, stderr = child.communicate(timeout=min(maximum, remaining))
            except BaseException:
                # A compiler can have cc1/as/ld children. Retire the entire
                # owned group on timeout instead of leaving builders behind.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.communicate(timeout=5)
                raise
        return subprocess.CompletedProcess(command, child.returncode, stdout, stderr)

    def check_build_bytes(work):
        nonlocal build_peak
        size = sum(path.stat().st_size for path in work.rglob('*') if path.is_file())
        build_peak = max(build_peak, size)
        if size > 16 * MIB:
            raise RuntimeError('Native stress build outputs exceed the 16 MiB limit')

    try:
        if not compiler:
            raise FileNotFoundError('No C++ compiler found: ' + args.cxx)
        report['compiler']['version'] = run([compiler, '--version'], maximum=5).stdout.splitlines()[0]
        report['source_sha256'] = {name: digest(KIT / name) for name in
                                  ('pipe_protocol.h', 'stress_pipe_protocol.cc',
                                   'npu_pipe.cc', 'stress_native_cleanup.cc',
                                   'stress_native_protocol.py')}
        parent = args.work_parent or Path('/data/local/tmp')
        if not parent.is_dir() or not os.access(parent, os.W_OK):
            parent = Path(tempfile.gettempdir())
        with tempfile.TemporaryDirectory(prefix='pixelforge-native-stress-', dir=parent) as owned:
            work = Path(owned)
            # A zero-work main distinguishes unavailable runtimes/address-space
            # mappings from a sanitizer finding in the actual stress harness.
            probe_source = work / 'sanitizer-probe.cc'
            probe_source.write_text('int main(){return 0;}\n')
            address_compiler = None
            candidates = list(dict.fromkeys(path for path in (compiler, shutil.which('clang++')) if path))
            for candidate in candidates:
                probe_binary = work / 'sanitizer-probe'
                built = run([candidate, '-fsanitize=address,undefined', str(probe_source), '-o', str(probe_binary)], maximum=10)
                evidence = {'compiler': candidate, 'compile_exit': built.returncode,
                            'compile_diagnostic': built.stderr[-4000:]}
                report['sanitizers']['address']['probes'].append(evidence)
                if built.returncode:
                    continue
                check_build_bytes(work)
                environment = os.environ.copy()
                environment['ASAN_OPTIONS'] = 'detect_leaks=1:abort_on_error=1:quarantine_size_mb=8:thread_local_quarantine_size_kb=64'
                executed = run([str(probe_binary)], environment=environment, maximum=5)
                evidence.update(run_exit=executed.returncode, runtime_diagnostic=executed.stderr[-4000:])
                if executed.returncode == 0:
                    address_compiler = candidate
                    break
            report['sanitizers']['address']['status'] = 'available' if address_compiler else 'unavailable'
            variants = [('plain', [], compiler), ('ubsan',
                        ['-fsanitize=undefined', '-fno-omit-frame-pointer'], compiler)]
            if address_compiler:
                variants.append(('asan-ubsan', ['-fsanitize=address,undefined',
                                 '-fno-omit-frame-pointer'], address_compiler))
            for name, extra, selected_compiler in variants:
                binary = work / name
                command = [selected_compiler, '-std=c++17', '-O1', '-g0', '-Wall', '-Wextra', '-Werror',
                           *extra, str(KIT / 'stress_pipe_protocol.cc'), '-o', str(binary)]
                before = time.monotonic()
                built = run(command)
                record = {'variant': name, 'compile_seconds': time.monotonic() - before,
                          'compiler': selected_compiler, 'compiler_flags': command[1:-3],
                          'compile_exit': built.returncode}
                report['runs'].append(record)
                if built.returncode:
                    record['diagnostic'] = built.stderr[-12000:]
                    if name != 'plain':
                        key = 'address' if name == 'asan-ubsan' else 'undefined'
                        report['sanitizers'][key].update(status='unavailable', reason=record['diagnostic'])
                        continue
                    raise RuntimeError('Plain native stress harness did not compile')
                check_build_bytes(work)
                environment = os.environ.copy()
                if name != 'plain':
                    environment['ASAN_OPTIONS'] = 'detect_leaks=1:abort_on_error=1:quarantine_size_mb=8:thread_local_quarantine_size_kb=64'
                    environment['UBSAN_OPTIONS'] = 'halt_on_error=1:print_stacktrace=1'
                    report['sanitizers'].update(asan_options=environment['ASAN_OPTIONS'],
                                               ubsan_options=environment['UBSAN_OPTIONS'])
                executed = run([str(binary), str(args.seed), str(args.iterations)],
                               environment=environment, maximum=30)
                record['run_exit'] = executed.returncode
                record['stderr'] = executed.stderr[-12000:]
                try:
                    record['result'] = json.loads(executed.stdout)
                except ValueError:
                    record['stdout'] = executed.stdout[-12000:]
                    report['failures'].append(name + ': no valid stress result')
                if executed.returncode:
                    report['failures'].append(name + ': stress/sanitizer process exited ' + str(executed.returncode))
                if record.get('result', {}).get('failures'):
                    report['failures'].append(name + ': reference/implementation mismatch')
                if name != 'plain':
                    key = 'address' if name == 'asan-ubsan' else 'undefined'
                    report['sanitizers'][key]['status'] = 'passed' if executed.returncode == 0 else 'failed'
            cleanup = {'status': 'not-run', 'cycles': args.cleanup_cycles, 'cases': 0,
                       'vendor_api_calls': 0, 'original_native_main_called': False, 'scenarios': {}}
            report['mock_sq_cleanup'] = cleanup
            android_root = Path('/proc/1/root')
            if not ANDROID_CXX.is_file() or not (android_root / 'system/bin/env').is_file():
                cleanup.update(status='unavailable', reason='Termux clang or Android root is unavailable')
            else:
                report['mock_compiler'] = {'path': str(ANDROID_CXX),
                                          'version': run([str(ANDROID_CXX), '--version'], maximum=5).stdout.splitlines()[0]}
                mock_binary = work / 'mock-sq-cleanup'
                built = run([str(ANDROID_CXX), '-std=c++17', '-O1', '-g0', '-Wall', '-Wextra', '-Werror',
                             '-D__ANDROID_UNAVAILABLE_SYMBOLS_ARE_WEAK__',
                             str(KIT / 'stress_native_cleanup.cc'), '-ldl', '-o', str(mock_binary)])
                if built.returncode:
                    raise RuntimeError('Mock cleanup fixture compile failed: ' + built.stderr[-12000:])
                check_build_bytes(work)
                # Shared /data is directly Android-visible. A fallback Ubuntu
                # /tmp remains accessible through this live parent's proc root.
                visible = str(mock_binary) if str(work).startswith('/data/') else f'/proc/{os.getpid()}/root{mock_binary}'
                chroot = shutil.which('chroot')
                if not chroot:
                    raise FileNotFoundError('Rooted Android cleanup fixture requires chroot')
                prefix = [chroot, '/proc/1/root', '/system/bin/env',
                          'LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:/data/data/com.termux/files/usr/lib',
                          'LD_PRELOAD=',
                          visible]
                scenarios = {'drop-failure': 2, 'make-failure': 2, 'replacement-failure': 2,
                             'zero-handle-failure': 2, 'invocation-cleanup-failure': 2,
                             'success': 0, 'double-success': 0}
                for _cycle in range(args.cleanup_cycles):
                    for scenario, expected in scenarios.items():
                        executed = run(prefix + [scenario], maximum=5)
                        entry = cleanup['scenarios'].setdefault(scenario, {'expected_exit': expected,
                                                                         'cases': 0, 'failures': 0})
                        entry['cases'] += 1
                        cleanup['cases'] += 1
                        valid = executed.returncode == expected and 'MOCK OWNER_VALID\n' in executed.stderr
                        if expected == 2:
                            valid = valid and executed.stdout == 'ERR SQ_UNLOAD\n'
                            valid = valid and 'MOCK ATEXIT' not in executed.stderr and 'MOCK DESTRUCTOR' not in executed.stderr
                            valid = valid and 'RETURNED' not in executed.stderr
                        else:
                            valid = valid and executed.stdout == 'MOCK RETURNED_SUCCESS\n'
                            valid = valid and 'MOCK ATEXIT' in executed.stderr and 'MOCK DESTRUCTOR' in executed.stderr
                        if not valid:
                            entry['failures'] += 1
                            entry['last_exit'] = executed.returncode
                            entry['last_stdout'] = executed.stdout[-1000:]
                            entry['last_stderr'] = executed.stderr[-1000:]
                            report['failures'].append('mock cleanup: ' + scenario)
                cleanup['status'] = 'passed' if all(not entry['failures'] for entry in cleanup['scenarios'].values()) else 'failed'
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.TimeoutExpired) as error:
        report['failures'].append(type(error).__name__ + ': ' + str(error))
    report['build_bytes_peak'] = build_peak
    statuses = {report['sanitizers'][name]['status'] for name in ('address', 'undefined')}
    report['sanitizers']['status'] = ('failed' if 'failed' in statuses else
                                     'passed' if statuses == {'passed'} else
                                     'partial' if 'passed' in statuses else 'unavailable')
    report['elapsed_seconds'] = time.monotonic() - started
    report['success'] = not report['failures']
    report['case_count'] = sum(record.get('result', {}).get('cases', 0) for record in report['runs'])
    report['mock_cleanup_case_count'] = report.get('mock_sq_cleanup', {}).get('cases', 0)
    return report


def main(argv=None):
    # Sanitizer startup failures must not leave large core files on the phone.
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=lambda value: int(value, 0), default=DEFAULT_SEED)
    parser.add_argument('--iterations', type=int, default=120000)
    parser.add_argument('--cleanup-cycles', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--cxx', default='g++')
    parser.add_argument('--work-parent', type=Path)
    parser.add_argument('--report', type=Path, default=Path('/tmp/pixelforge-stress-native.json'))
    args = parser.parse_args(argv)
    if not 0 <= args.seed < 2**64 or not 100000 <= args.iterations <= 10000000:
        parser.error('Seed must be uint64; iterations must be 100000..10000000')
    if not 1 <= args.cleanup_cycles <= 64 or not 10 <= args.timeout <= 60:
        parser.error('cleanup-cycles must be 1..64; timeout must be 10..60 seconds')
    if args.report.is_symlink():
        parser.error('Report destination must not be a symlink')
    report = stress(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.pixelforge-native-stress-', dir=args.report.parent)
    try:
        with os.fdopen(fd, 'w') as output:
            json.dump(report, output, indent=2)
            output.write('\n')
        os.replace(temporary, args.report)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(json.dumps({'success': report['success'], 'report': str(args.report),
                      'cases': report['case_count'], 'mock_cleanup_cases': report['mock_cleanup_case_count'],
                      'sanitizers': report['sanitizers']['status'], 'elapsed_seconds': report['elapsed_seconds'],
                      'failures': report['failures']}))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
