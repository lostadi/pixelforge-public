#!/usr/bin/env python3
"""Read-only permission and credential checks for the configured Pixel assistant.

Run from the authorized root terminal to inspect private app configuration.
--probe-transport verifies pinned TLS and rejects unauthenticated HTTP calls;
it does not dispatch an Ostadix program. Secrets never enter the report.
"""
import argparse
import hashlib
import hmac
import http.client
import json
import os
from pathlib import Path
import re
import ssl
import stat
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path('/data/data/com.termux/files/home/.config/ostadix/gemini/host.json')
DEPLOY = Path('/data/local/ostadix-assistant')
BOOT = Path('/data/adb/service.d/40-ostadix-assistant.sh')
FUNCTION = 'org.ostadix.terminal.OstadixAppFunctionService#executeO'
REQUIRED = {
    'org.ostadix.terminal': ['INTERNET', 'FOREGROUND_SERVICE',
                             'FOREGROUND_SERVICE_SPECIAL_USE', 'POST_NOTIFICATIONS'],
    'org.ostadix.aicore.extension': ['POST_NOTIFICATIONS'],
    'com.google.android.googlequicksearchbox': ['INTERNET', 'EXECUTE_APP_FUNCTIONS'],
    'com.google.android.aicore': ['ACCESS_NPU_MODEL_MANAGER_API',
                                'USE_ON_DEVICE_INTELLIGENCE'],
}
EXTENSION = ROOT / 'apps/aicore-ostadix-extension'


def command(*args):
    env = dict(os.environ, PATH='/system/bin:' + os.environ.get('PATH', ''))
    result = subprocess.run(args, env=env, capture_output=True, text=True, timeout=20)
    if result.returncode:
        # Command stderr can contain private data. Report only the operation.
        raise RuntimeError(Path(args[0]).name + ' exited ' + str(result.returncode))
    return result.stdout.strip()


def package_snapshot(dump):
    """Use the active package, not the second copy under Hidden system packages."""
    active = dump.split('Hidden system packages:', 1)[0]
    uid = re.search(r'^\s+appId=(\d+)', active, re.M)
    version = re.search(r'^\s+versionName=(.*)$', active, re.M)
    code = re.search(r'^\s+versionCode=(\d+)', active, re.M)
    requested = re.search(r'^\s+requested permissions:\n((?:[ \t]+[^\n]*\n)*)', active, re.M)
    names = []
    if requested:
        for line in requested.group(1).splitlines():
            value = line.strip()
            if not re.fullmatch(r'[a-zA-Z0-9_.]+', value):
                break
            names.append(value)
    granted = dict((name, value == 'true') for name, value in re.findall(
        r'^\s+([a-zA-Z0-9_.]+): granted=(true|false)', active, re.M))
    return {'uid': int(uid.group(1)) if uid else None,
            'version_code': int(code.group(1)) if code else None,
            'version_name': version.group(1) if version else None,
            'declared_permissions': {name: granted.get(name) for name in sorted(set(names))}}


def file_status(path, expected_uid, private=False, executable=False):
    info = path.lstat()
    mode = stat.S_IMODE(info.st_mode)
    acceptable_mode = (not mode & 0o077) if private else (not mode & 0o022)
    return {'path': str(path), 'uid': info.st_uid, 'mode': oct(mode),
            'regular_file': stat.S_ISREG(info.st_mode),
            'passed': stat.S_ISREG(info.st_mode) and info.st_uid == expected_uid
            and acceptable_mode and (not executable or bool(mode & 0o100))}


def client_matches(client, host, fingerprint):
    token = client.get('token')
    expected = host.get('token')
    return (type(token) is str and type(expected) is str and len(token) >= 64
            and hmac.compare_digest(token, expected)
            and type(client.get('port')) is int and client['port'] == host.get('port')
            and client.get('certificate_sha256') == fingerprint)


def allowed_package_identities(source):
    """Read the exact identity tuples actually passed to the Java package gate."""
    constants = {}
    for name, text, number in re.findall(r'\b(?:String|long)\s+([A-Z_0-9]+)\s*=\s*'
                                        r'(?:"([^"\n]*)"|([0-9]+)L)\s*;', source):
        constants[name] = int(number) if number else text
    identities = {}
    for package, code, name, apk, signer in re.findall(
            r'packageMatches\(\s*context\s*,\s*([A-Z_0-9]+)\s*,\s*([A-Z_0-9]+)\s*,'
            r'\s*([A-Z_0-9]+)\s*,\s*([A-Z_0-9]+)\s*,\s*([A-Z_0-9]+)\s*\)', source):
        identities.setdefault(constants[package], []).append({
            'version_code': constants[code], 'version_name': constants[name],
            'apk_sha256': constants[apk], 'signer_sha256': constants[signer]})
    return identities


def identity_matches(snapshot, apk_digest, signers, allowed):
    return len(signers) == 1 and any(
        snapshot['version_code'] == identity['version_code']
        and snapshot['version_name'] == identity['version_name']
        and apk_digest == identity['apk_sha256']
        and signers[0] == identity['signer_sha256'] for identity in allowed)


def signers_for_sdk(text, sdk):
    signers = []
    for minimum, maximum, digest in re.findall(
            r'^(?:V[0-9.]+ Signer:|Signer #[0-9]+:?) '
            r'(?:\(minSdkVersion=(\d+), maxSdkVersion=(\d+)\) )?'
            r'certificate SHA-256 digest: ([0-9a-f]{64})$', text, re.M):
        if minimum and not int(minimum) <= sdk <= int(maximum):
            continue
        if digest not in signers:
            signers.append(digest)
    return signers


def source_identity():
    records = []
    for source in sorted((EXTENSION / 'app/src/main/java').rglob('*.java')):
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        records.append(digest + '  ' + str(source) + '\n')
    return hashlib.sha256(''.join(records).encode()).hexdigest()


def assistant_route_status(route):
    """Require explicit selection; older modules can intercept all text otherwise."""
    return dict(passed=route == 'local-o-v1', state=route, expected='local-o-v1',
                unsafe_legacy_catch_all=route == 'local-all-v1')


def transport_denials(host, fingerprint):
    """Wrong/missing credentials must be denied before any MCP dispatch."""
    checks = []
    context = ssl.create_default_context(cafile=host['certificate'])
    context.check_hostname = False
    for label, authorization in [('missing_token', None), ('wrong_token', 'Bearer invalid-probe')]:
        connection = http.client.HTTPSConnection('127.0.0.1', host['port'],
                                                context=context, timeout=5)
        try:
            connection.connect()
            actual = hashlib.sha256(connection.sock.getpeercert(binary_form=True)).hexdigest()
            if not hmac.compare_digest(actual, fingerprint):
                raise RuntimeError('certificate pin mismatch')
            headers = {'Content-Type': 'application/json'}
            if authorization is not None:
                headers['Authorization'] = authorization
            connection.request('POST', '/o_execute', '{}', headers)
            response = connection.getresponse()
            response.read(1024)
            checks.append({'boundary': label, 'status': response.status,
                           'passed': response.status == 403})
        finally:
            connection.close()
    return checks


def inspect(probe_transport=False):
    checks = []
    packages = {}

    def check(name, passed, **details):
        checks.append(dict(check=name, passed=bool(passed), **details))

    for package, required in REQUIRED.items():
        try:
            dump = command('/system/bin/dumpsys', 'package', package)
            snapshot = package_snapshot(dump)
            packages[package] = snapshot
            permissions = snapshot['declared_permissions']
            for short in required:
                permission = 'android.permission.' + short
                check(package + ':' + short, permissions.get(permission) is True,
                      granted=permissions.get(permission))
            if package == 'org.ostadix.terminal':
                check('app_function_system_binding_permission', bool(re.search(
                    r'OstadixAppFunctionService[^\n]*permission android\.permission\.BIND_APP_FUNCTION_SERVICE',
                    dump)))
        except Exception as error:
            check(package + ':inspection', False, error_class=type(error).__name__)

    try:
        gate = EXTENSION / 'app/src/main/java/org/ostadix/aicore/extension/ExtensionGate.java'
        identities = allowed_package_identities(gate.read_text())
        sdk = int(command('/system/bin/getprop', 'ro.build.version.sdk'))
        for package in ('com.google.android.googlequicksearchbox', 'com.google.android.aicore'):
            paths = command('/system/bin/pm', 'path', package).splitlines()
            base = next(Path(line.removeprefix('package:')) for line in paths
                        if line.endswith('/base.apk'))
            with base.open('rb') as apk:
                digest = hashlib.file_digest(apk, 'sha256').hexdigest()
            signatures = command('apksigner', 'verify', '--print-certs', str(base))
            signers = signers_for_sdk(signatures, sdk)
            check(package + ':source_reviewed_identity', identity_matches(packages[package],
                  digest, signers, identities.get(package, [])), apk_sha256=digest)
        module_path = command('/system/bin/pm', 'path', 'org.ostadix.aicore.extension')
        with zipfile.ZipFile(module_path.removeprefix('package:')) as archive:
            expected_source = source_identity().encode()
            installed_source_matches = any(expected_source in archive.read(name)
                                           for name in archive.namelist() if name.endswith('.dex'))
        check('installed_extension_matches_source_identity', installed_source_matches)
    except Exception as error:
        check('package_compatibility_inspection', False, error_class=type(error).__name__)

    try:
        holders = command('/system/bin/cmd', 'role', 'get-role-holders',
                          'android.app.role.ASSISTANT').splitlines()
        check('assistant_role', 'com.google.android.googlequicksearchbox' in holders)
        enabled = command('/system/bin/cmd', 'app_function', 'is-enabled', '--package',
                          'org.ostadix.terminal', '--function', FUNCTION)
        check('ostadix_app_function_enabled', enabled == 'true')
        route = command('/system/bin/settings', 'get', 'global', 'ostadix_gemini_nano_tool_mode')
        route_status = assistant_route_status(route)
        check('local_assistant_route_enabled', route_status.pop('passed'), **route_status)
    except Exception as error:
        check('android_broker_inspection', False, error_class=type(error).__name__)

    try:
        host = json.loads(CONFIG.read_text())
        owner = CONFIG.stat().st_uid
        check('host_non_root_owner', owner != 0, uid=owner)
        status = file_status(CONFIG, owner, private=True)
        check('host_config_permissions', status.pop('passed'), **status)
        check('host_runtime_root', host.get('root') == str(ROOT))
        for key in ('certificate', 'private_key'):
            status = file_status(Path(host[key]), owner, private=key == 'private_key')
            check('host_' + key + '_permissions', status.pop('passed'), **status)
        fingerprint = hashlib.sha256(ssl.PEM_cert_to_DER_cert(
            Path(host['certificate']).read_text())).hexdigest()
        for package in ('org.ostadix.terminal', 'com.google.android.googlequicksearchbox'):
            path = Path('/data/user/0') / package / 'files/ostadix-mcp-host.json'
            status = file_status(path, packages.get(package, {}).get('uid'), private=True)
            check(package + ':client_config_permissions', status.pop('passed'), **status)
            client = json.loads(path.read_text())
            check(package + ':client_credentials_match', client_matches(client, host, fingerprint))
        state = json.loads((CONFIG.parent / 'host-supervisor.json').read_text())
        for kind in ('supervisor', 'host'):
            pid = state.get(kind + '_pid')
            fields = Path('/proc/' + str(pid) + '/stat').read_text().rsplit(') ', 1)[1].split()
            process_uid = Path('/proc/' + str(pid)).stat().st_uid
            check(kind + '_owner_and_liveness', fields[0] != 'Z'
                  and fields[19] == state.get(kind + '_start_ticks') and process_uid == owner,
                  uid=process_uid)
        if probe_transport:
            for result in transport_denials(host, fingerprint):
                check('transport:' + result['boundary'], result['passed'], http_status=result['status'])
    except Exception as error:
        check('private_host_inspection', False, error_class=type(error).__name__)

    try:
        status = file_status(BOOT, 0, executable=True)
        check('root_boot_entry', status.pop('passed'), **status)
        for name in ('ostadix_appfunction_host.py', 'ostadix_appfunction_supervisor.py'):
            path = DEPLOY / name
            status = file_status(path, 0)
            check('deployed:' + name, status.pop('passed'), **status)
            check('deployment_matches_source:' + name,
                  path.read_bytes() == (ROOT / 'scripts' / name).read_bytes())
    except Exception as error:
        check('boot_inspection', False, error_class=type(error).__name__)
    return {'passed': all(item['passed'] for item in checks), 'checks': checks,
            'packages': packages, 'transport_probed': probe_transport,
            'scope': 'Declared permission state, reviewed package/source identity, private configuration, '
                     'supervised host and optional TLS authentication. '
                     'Does not prove Nano generation, workload access to every device resource, or reboot recovery.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-transport', action='store_true')
    args = parser.parse_args()
    report = inspect(args.probe_transport)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()
