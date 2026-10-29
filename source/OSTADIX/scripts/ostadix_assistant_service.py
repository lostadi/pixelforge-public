#!/usr/bin/env python3
"""Install/start/status/disable the owned Pixel assistant's local MCP service.

Requires the existing private host configuration and, for installation,
KernelSU's service.d directory. Never copies or prints credentials.
"""
import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import shlex
import socket
import ssl
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
TERMUX_HOME = Path('/data/data/com.termux/files/home')
PREFIX = Path('/data/data/com.termux/files/usr')
CONFIG = TERMUX_HOME / '.config/ostadix/gemini/host.json'
DEPLOY = Path('/data/local/ostadix-assistant')
BOOT = Path('/data/adb/service.d/40-ostadix-assistant.sh')
SETTING = 'ostadix_gemini_nano_tool_mode'


def system(*args):
    return subprocess.check_output(list(args), text=True).strip()


def current_process(pid, ticks):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        return fields[0] != 'Z' and fields[19] == ticks
    except OSError:
        return False


def status():
    config = json.loads(CONFIG.read_text())
    record = dict(route=system('/system/bin/settings', 'get', 'global', SETTING),
                  boot_script_installed=BOOT.is_file(), supervisor_running=False, host_running=False)
    state = CONFIG.parent / 'host-supervisor.json'
    if state.exists():
        data = json.loads(state.read_text())
        for kind in ('supervisor', 'host'):
            record[kind + '_running'] = current_process(data.get(kind + '_pid'), data.get(kind + '_start_ticks'))
        record['host_pid'] = data.get('host_pid')
        record['host_launches'] = data.get('launches')
    try:
        with socket.create_connection(('127.0.0.1', config['port']), timeout=1):
            record['listening'] = True
    except OSError:
        record['listening'] = False
    return record


def install():
    if os.getuid() != 0:
        raise SystemExit('Install from the authorized root terminal')
    if not BOOT.parent.is_dir():
        raise SystemExit('KernelSU service.d is missing; no alternate boot mechanism was installed')
    config = json.loads(CONFIG.read_text())
    uid = CONFIG.stat().st_uid
    if uid == 0 or CONFIG.stat().st_mode & 0o077 or config['root'] != str(ROOT):
        raise SystemExit('Expected an app-owned private configuration for this canonical checkout')
    if status()['supervisor_running']:
        names = ('ostadix_appfunction_host.py', 'ostadix_appfunction_supervisor.py')
        if all((DEPLOY / name).is_file() and (DEPLOY / name).read_bytes()
               == (ROOT / 'scripts' / name).read_bytes() for name in names) and BOOT.is_file():
            return dict(installed=True, already_current=True, uid=uid)
        raise SystemExit('Supervisor running with different scripts; disable the route and stop the service before upgrading')
    DEPLOY.mkdir(mode=0o755, exist_ok=True)
    os.chown(DEPLOY, 0, 0); os.chmod(DEPLOY, 0o755)
    hashes = {}
    for name in ('ostadix_appfunction_host.py', 'ostadix_appfunction_supervisor.py'):
        data = (ROOT / 'scripts' / name).read_bytes()
        target = DEPLOY / name
        target.write_bytes(data); os.chown(target, 0, 0); os.chmod(target, 0o644)
        hashes[name] = hashlib.sha256(data).hexdigest()
    env = [str(PREFIX / 'bin/env'), '-i', 'HOME=' + str(TERMUX_HOME), 'PREFIX=' + str(PREFIX),
           'TMPDIR=' + str(PREFIX / 'tmp'),
           'PATH=' + ':'.join(map(str, [PREFIX / 'bin', ROOT / 'target/release',
                                      TERMUX_HOME / '.cargo/bin', Path('/system/bin')])),
           'LD_PRELOAD=' + str(PREFIX / 'lib/libtermux-exec-ld-preload.so'),
           str(PREFIX / 'bin/python'), str(DEPLOY / 'ostadix_appfunction_supervisor.py'),
           '--config', str(CONFIG), '--host', str(DEPLOY / 'ostadix_appfunction_host.py')]
    launch = shlex.join(['/system/bin/su', str(uid), '-c', shlex.join(env)])
    # Wait for credential-encrypted Termux files after boot. A supervisor exits
    # 75 if another owns its lock, so repeated start commands create no loop.
    text = f'''#!/system/bin/sh
worker() {{
    while [ "$(/system/bin/getprop sys.boot_completed)" != "1" ] || [ ! -r {shlex.quote(str(CONFIG))} ]; do
        /system/bin/sleep 2
    done
    while :; do
        {launch}
        code=$?
        [ "$code" -eq 75 ] && exit 0
        /system/bin/sleep 2
    done
}}
worker >>{shlex.quote(str(DEPLOY / 'boot.log'))} 2>&1 &
'''
    BOOT.write_text(text); os.chown(BOOT, 0, 0); os.chmod(BOOT, 0o755)
    (DEPLOY / 'installed.json').write_text(json.dumps(dict(root=str(ROOT), uid=uid, hashes=hashes), indent=2) + '\n')
    return dict(installed=True, uid=uid, hashes=hashes)


def start():
    record = status()
    if not record['supervisor_running']:
        if record['listening']:
            raise SystemExit('An unmanaged host is listening. Stop that idle host before starting supervision.')
        subprocess.run(['/system/bin/sh', str(BOOT)], check=True, stdin=subprocess.DEVNULL,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(60):
            if status()['listening']:
                break
            time.sleep(0.2)
        else:
            raise SystemExit('Service did not start; inspect the private service logs')
    # Keep ordinary Gemini requests on Google's flow, including after repeated
    # starts or migration from an older catch-all configuration.
    system('/system/bin/settings', 'put', 'global', SETTING, 'local-o-v1')
    return status()


def check():
    config = json.loads(CONFIG.read_text())
    context = ssl.create_default_context(cafile=config['certificate'])
    context.check_hostname = False
    expected = hashlib.sha256(ssl.PEM_cert_to_DER_cert(Path(config['certificate']).read_text())).hexdigest()
    connection = http.client.HTTPSConnection('127.0.0.1', config['port'], context=context, timeout=20)
    connection.connect()
    if hashlib.sha256(connection.sock.getpeercert(binary_form=True)).hexdigest() != expected:
        raise SystemExit('Local host certificate mismatch')
    body = dict(source='python^( __oval_result__ = 1 + 1 )_python', bindings={},
                constraints=dict(timeout_ms=15000))
    connection.request('POST', '/o_execute', json.dumps(body), {
        'Authorization': 'Bearer ' + config['token'], 'Content-Type': 'application/json',
        'X-Ostadix-Request': 'service-readiness'})
    response = connection.getresponse(); result = json.loads(response.read()); connection.close()
    content = result.get('structuredContent', {}); native = content.get('result') or {}
    report = dict(http_status=response.status, state=content.get('state'),
                  actual_result=native.get('value'), execution_evidence=native.get('execution_evidence'))
    report['passed'] = (response.status == 200 and content.get('state') == 'completed'
                        and native.get('value', {}).get('v', {}).get('v') == '2')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['install', 'start', 'status', 'check', 'disable'])
    action = parser.parse_args().action
    if action == 'disable':
        system('/system/bin/settings', 'delete', 'global', SETTING)
        result = dict(route_disabled=True, service_retained=True)
    else:
        result = globals()[action]()
    print(json.dumps(result, indent=2))
    if action == 'check' and not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
