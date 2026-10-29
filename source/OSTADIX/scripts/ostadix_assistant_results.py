#!/usr/bin/env python3
"""Install the owned result viewer or recover private records on the pinned Pixel.

Run from the authorized root terminal. Recovery never invokes a model or program.
It prints counts only; prompts, answers and credentials remain on the phone.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'org.ostadix.aicore.extension'
GSA = Path('/data/user/0/com.google.android.googlequicksearchbox/files')
EVENTS = Path('/data/user/0/com.google.android.aicore/files/ostadix-nano-probe/events.jsonl')
ID = re.compile(r'ostadix-turn-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def package_uid(package):
    data = command('/system/bin/pm', 'list', 'packages', '-U', '--user', '0', package)
    return int(re.search(r'^package:' + re.escape(package) + r' uid:(\d+)$', data, re.M)[1])


def assert_idle():
    if EVENTS.is_file():
        last = json.loads(EVENTS.read_text().splitlines()[-1])
        if last.get('phase') != 'complete':
            raise SystemExit('Nano has not recorded cleanup completion; wait before installing')
    turns = list(GSA.glob('ostadix-turn-*.json'))
    if turns:
        last = json.loads(max(turns, key=lambda p: p.stat().st_mtime).read_text())
        if last.get('phase') not in ('answer_ready', 'answer_ready_without_interpretation', 'failed'):
            raise SystemExit('Latest assistant turn is not terminal; wait before installing')


def recover():
    uid = package_uid(PACKAGE)
    files = Path('/data/user/0') / PACKAGE / 'files'
    directory = files / 'history'
    for path in (files, directory):
        path.mkdir(exist_ok=True); os.chown(path, uid, uid); os.chmod(path, 0o700)
    imported = skipped = 0
    for source in GSA.glob('ostadix-turn-*.json'):
        if not ID.fullmatch(source.stem) or source.stat().st_size > 16 * 1024 * 1024:
            skipped += 1; continue
        target = directory / source.name
        if target.exists():
            skipped += 1; continue
        try:
            record = json.loads(source.read_text())
            if record.get('request_id') != source.stem:
                skipped += 1; continue
            record['history_recovered'] = True
            record['history_updated_at'] = round(source.stat().st_mtime * 1000)
            record['history_terminal'] = record.get('phase') in ('answer_ready', 'answer_ready_without_interpretation', 'failed')
            # ResultHistory renders old records from answer, native output or error.
            pending = directory / (source.stem + '.pending')
            with pending.open('w') as out:
                out.write(json.dumps(record) + '\n'); out.flush(); os.fsync(out.fileno())
            os.chown(pending, uid, uid); os.chmod(pending, 0o600)
            os.replace(pending, target)
            os.utime(target, (source.stat().st_mtime, source.stat().st_mtime))
            imported += 1
        except (OSError, ValueError):
            skipped += 1
    subprocess.run(['/system/bin/restorecon', '-RF', str(files)], check=True)
    return dict(imported=imported, skipped=skipped, execution_attempts=0)


def install():
    assert_idle()
    apk = ROOT / 'apps/aicore-ostadix-extension/build/outputs/apk/debug/OstadixAicoreExtension-debug.apk'
    expected = hashlib.sha256(apk.read_bytes()).hexdigest()
    command('/system/bin/pm', 'install', '-r', str(apk))
    installed = command('/system/bin/pm', 'path', PACKAGE).removeprefix('package:')
    actual = hashlib.sha256(Path(installed).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('Installed APK identity mismatch')
    command('/system/bin/pm', 'grant', PACKAGE, 'android.permission.POST_NOTIFICATIONS')
    # Restart the exact scoped hosts so no previously installed candidate hook survives
    # an upgrade that disables it. Use pidfds to avoid PID reuse.
    sys.path.insert(0, str(ROOT / 'tools/nano-factory-reader'))
    from android_pidfd import open_pidfd, send_pidfd_signal
    restarted = []
    for package, suffix in [('com.google.android.aicore', ''),
                            ('com.google.android.googlequicksearchbox', ':search'),
                            ('com.google.android.as.oss', ''), ('com.google.android.as', '')]:
        uid = package_uid(package); name = package + suffix
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit():
                continue
            try:
                if (proc / 'cmdline').read_bytes().split(b'\0')[0].decode() != name:
                    continue
                fd = open_pidfd(int(proc.name))
                try:
                    uids = [int(v) for line in (proc / 'status').read_text().splitlines()
                            if line.startswith('Uid:') for v in line.split()[1:]]
                    if uids != [uid] * 4 or (proc / 'cmdline').read_bytes().split(b'\0')[0].decode() != name:
                        raise RuntimeError('Host process identity changed')
                    send_pidfd_signal(fd, signal.SIGTERM)
                    restarted.append(dict(pid=int(proc.name), uid=uid, name=name))
                finally:
                    os.close(fd)
            except FileNotFoundError:
                continue
    return dict(apk_sha256=actual, installed_apk=installed, restarted=restarted, recovery=recover())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['install', 'recover', 'open'])
    parser.add_argument('--request-id')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if os.getuid() != 0:
        raise SystemExit('Use the authorized root terminal for deployment and recovery')
    if args.action == 'open':
        cmd = ['/system/bin/am', 'start', '-n', PACKAGE + '/.ResultHistoryActivity']
        if args.request_id:
            if not ID.fullmatch(args.request_id):
                raise SystemExit('Invalid request identity')
            cmd += ['--es', 'request_id', args.request_id]
        result = dict(opened=True, activity_output=command(*cmd))
    else:
        result = globals()[args.action]()
    output = json.dumps(result, indent=2) + '\n'
    if args.report:
        args.report.write_text(output)
    print(output, end='')


if __name__ == '__main__':
    main()
