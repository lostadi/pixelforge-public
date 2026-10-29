#!/usr/bin/env python3
"""Keep a verified OSTADIX repository and its compiled MCP/runtime together.

Only the app-owned runtime/ostadix tree is modified. A failed fetch, build, or
health check leaves current.json and the previous working pair untouched.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import tempfile
import time

APP = Path(__file__).resolve().parent
PAIR_HOME = APP / 'runtime' / 'ostadix'
UPSTREAM = 'https://github.com/lostadi/ostadix-BM.git'
BINS = ('O', 'o-cli', 'olangc', 'ocorec', 'o-link', 'o-unlink', 'ogit',
        'o-live-host', 'o-node', 'octl', 'o-registry', 'o-info',
        'ostadix-device', 'ocore-kernel-world-record')
SMOKE_SOURCE = 'python^(\n__oval_result__ = 1 + 1\n)_python\n'


class PairError(RuntimeError):
    pass


def _now():
    return datetime.now(timezone.utc).isoformat()


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _network_environment():
    from agent import AppError, codex_environment, _ensure_codex_network
    try:
        env = codex_environment()
        _ensure_codex_network(env)
        return env
    except AppError as error:
        raise PairError('OSTADIX network setup failed: ' + str(error)) from error


def _git(args, cwd=None, timeout=20):
    env = _network_environment() if args[0] in ('ls-remote', 'fetch') else dict(os.environ)
    env['GIT_TERMINAL_PROMPT'] = '0'
    try:
        result = subprocess.run(['git', *args], cwd=cwd, env=env, text=True,
                                capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PairError(f'OSTADIX repository check failed: {error}') from error
    if result.returncode:
        raise PairError(f'OSTADIX repository check failed: {result.stderr.strip()[-1500:]}')
    return result.stdout.strip()


def load_pair():
    """Return the current manifest only after checking revision and binary hashes."""
    try:
        pair = json.loads((PAIR_HOME / 'current.json').read_text())
        revision = pair['revision']
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('invalid revision')
        root = PAIR_HOME / 'checkouts' / revision
        if Path(pair['root']) != root or root.is_symlink():
            raise ValueError('unexpected repository location')
        if pair['upstream'] != UPSTREAM:
            raise ValueError('unexpected upstream')
        if _git(['rev-parse', 'HEAD'], root) != revision:
            raise ValueError('repository revision changed')
        _git(['diff', '--exit-code', 'HEAD', '--'], root)
        if Path(pair['backends']) != root / 'backends' or not Path(pair['backends']).is_dir():
            raise ValueError('matching backends are missing')
        expected = {name: str(root / 'target' / 'release' / name) for name in BINS}
        expected['ostadix-mcp'] = str(root / 'mcp' / 'ostadix_lang_mcp_server' / 'target' / 'release' / 'ostadix-mcp')
        if pair['binary'] != expected['ostadix-mcp'] or pair['binaries'] != expected:
            raise ValueError('unexpected native binary locations')
        if not isinstance(pair['sha256'], dict):
            raise ValueError('invalid executable hash records')
        for name, path in expected.items():
            if Path(path).is_symlink() or not os.access(path, os.X_OK):
                raise ValueError(f'{name} is missing or not executable')
            if _hash(path) != pair['sha256'].get(name):
                raise ValueError(f'{name} does not match the recorded build')
        return pair
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PairError(f'No verified OSTADIX pair is available: {error}') from error


def pair_environment(pair, environment=None):
    """Explicit matching runtime paths; never resolve a CLI from an older install."""
    bins = pair['binaries']
    env = dict(os.environ if environment is None else environment)
    # Termux's exec shim relaunches native executables through Android's linker.
    # OSTADIX workers self-relaunch their current executable, which then resolves
    # to that linker and rejects --o-backend. Native Termux binaries work when
    # this shim is omitted from this MCP process tree only. Preserve other
    # preloads and the user's global environment.
    preloads = [entry for entry in re.split(r'[:\s]+', env.get('LD_PRELOAD', ''))
                if entry and not Path(entry).name.startswith('libtermux-exec')]
    if preloads:
        env['LD_PRELOAD'] = ':'.join(preloads)
    else:
        env.pop('LD_PRELOAD', None)
    env.update({
        'O_LANG_ROOT': pair['root'], 'O_BACKENDS_DIR': pair['backends'],
        'OLANG': bins['O'], 'OSTADIX_O_CLI_BIN': bins['o-cli'],
        'OSTADIX_OCTL_BIN': bins['octl'], 'OSTADIX_O_INFO_BIN': bins['o-info'],
        'OSTADIX_RUNTIME_PATH_MODE': 'inherited-only',
        'PATH': str(Path(bins['O']).parent) + os.pathsep + env.get('PATH', ''),
    })
    for suffix, binary in [('OCLI', 'o-cli'), ('OLANGC', 'olangc'),
                           ('EVALUATOR', 'O'), ('LIVE', 'o-live-host'),
                           ('OGIT', 'ogit'), ('NODE', 'o-node'),
                           ('OCTL', 'octl'), ('REGISTRY', 'o-registry'),
                           ('INFO', 'o-info'), ('DEVICE', 'ostadix-device')]:
        env['O_LANG_' + suffix + '_BIN'] = bins[binary]
    return env


def _smoke_result(result, action):
    """Validate the actual job result, never an incidental digit in its metadata."""
    if result.get('isError'):
        raise PairError(f'OSTADIX {action} health check failed: {json.dumps(result)[:3000]}')
    value = result.get('structuredContent')
    if not isinstance(value, dict):
        raise PairError(f'OSTADIX {action} returned no structured execution result')
    if value.get('action') != action or value.get('state') != 'completed' or value.get('exit_code') != 0:
        raise PairError(f'OSTADIX {action} did not complete successfully: {json.dumps(value)[:3000]}')
    output = value.get('result')
    if not isinstance(output, dict) or output.get('ok') is not True:
        raise PairError(f'OSTADIX {action} returned no successful native result: {output!r}')
    if action == 'check' and output.get('stage') != 'parse':
        raise PairError('OSTADIX check did not confirm parsing')
    if action == 'execute':
        expected = {'t': 'number', 'v': {'kind': 'int', 'v': '2'}}
        if output.get('type') != 'number' or output.get('value') != expected:
            raise PairError(f'OSTADIX smoke program returned {output.get("value")!r}; expected the actual number 2')
    return value


def _health(pair):
    """Actual native MCP initialize/list plus parse and execute of harmless 1 + 1."""
    # A tiny synchronous probe keeps the updater independent from the application
    # client. All output is bounded in time; it never calls a model or a paid API.
    with tempfile.TemporaryDirectory(prefix='ostadix-pair-check-') as work:
        with tempfile.TemporaryFile() as errors:
            child = subprocess.Popen([pair['binary']], cwd=work, env=pair_environment(pair),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=errors, start_new_session=True, bufsize=0)
            selector = selectors.DefaultSelector()
            selector.register(child.stdout, selectors.EVENT_READ)
            pending = bytearray()
            counter = 0

            def send(message):
                child.stdin.write((json.dumps(message) + '\n').encode())

            def call(method, params, timeout=45):
                nonlocal counter
                counter += 1
                send({'jsonrpc': '2.0', 'id': counter, 'method': method, 'params': params})
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    if b'\n' not in pending:
                        if not selector.select(max(0, deadline - time.monotonic())):
                            break
                        part = os.read(child.stdout.fileno(), 65536)
                        if not part:
                            errors.seek(0)
                            raise PairError('OSTADIX MCP exited during its health check: ' + errors.read(2000).decode(errors='replace'))
                        pending.extend(part)
                    while b'\n' in pending:
                        line, _, rest = pending.partition(b'\n')
                        pending[:] = rest
                        response = json.loads(line)
                        if response.get('id') != counter:
                            continue
                        if 'error' in response:
                            raise PairError('OSTADIX MCP health check failed: ' + json.dumps(response['error']))
                        return response['result']
                raise PairError(f'OSTADIX MCP health check timed out: {method}')

            try:
                initialized = call('initialize', {
                    'protocolVersion': '2025-03-26', 'capabilities': {},
                    'clientInfo': {'name': 'ostadix-voice-pair-check', 'version': '1'}})
                if initialized.get('protocolVersion') != '2025-03-26':
                    raise PairError('OSTADIX MCP did not negotiate supported protocol 2025-03-26')
                send({'jsonrpc': '2.0', 'method': 'notifications/initialized'})
                catalog = call('tools/list', {})
                names = {tool['name'] for tool in catalog['tools']}
                if not {'o_execute', 'o_capabilities', 'o_guide'} <= names:
                    raise PairError('The built MCP is missing its primary OSTADIX tools')
                results = {}
                for action in ('check', 'execute'):
                    result = call('tools/call', {'name': 'o_execute', 'arguments': {
                        'source': SMOKE_SOURCE, 'action': action, 'cwd': work}})
                    _smoke_result(result, action)
                    results[action] = result
                return {'server': initialized.get('serverInfo'), 'tool_count': len(names),
                        'check': results['check'], 'execute': results['execute']}
            finally:
                selector.close()
                if child.poll() is None:
                    import signal
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                child.stdin.close()
                child.stdout.close()


def _write_manifest(pair):
    descriptor, path = tempfile.mkstemp(prefix='pair-', suffix='.json', dir=PAIR_HOME)
    try:
        with os.fdopen(descriptor, 'w') as stream:
            json.dump(pair, stream, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(path, PAIR_HOME / 'current.json')
        directory = os.open(PAIR_HOME, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def _run_build(command, root, output, environment, timeout=3600):
    """Own Cargo's process group so interrupted updates also stop its compilers."""
    child = subprocess.Popen(command, cwd=root, stdout=output, stderr=subprocess.STDOUT,
                             env=environment, start_new_session=True)
    try:
        return child.wait(timeout=timeout)
    except BaseException:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=3)
        except subprocess.TimeoutExpired:
            pass
        # Cargo may exit before its compiler descendants. Signal the original
        # group even when its leader has already been reaped.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()
        raise


def update_pair(progress=None):
    """Check official HEAD, build/check a new pair, and atomically activate it.

    Raises PairError on failure; the caller can load_pair() for an explicitly
    labelled cached fallback. There is no silent claim of freshness offline.
    """
    report = progress or (lambda _message: None)
    PAIR_HOME.mkdir(parents=True, exist_ok=True)
    with (PAIR_HOME / 'update.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise PairError('Another OSTADIX pair update is already running') from error
        report('Checking OSTADIX-BM for updates…')
        try:
            remote = _git(['ls-remote', UPSTREAM, 'HEAD'], timeout=15)
            revision = remote.split()[0] if remote else ''
            fetch_source = 'origin'
        except PairError as error:
            try:
                cached = load_pair()
            except PairError:
                cached = None
            if cached:
                report(str(error) + '; remote freshness is unknown. Using verified installed OSTADIX-BM '
                       + cached['revision'][:12] + '.')
                return cached
            config = PAIR_HOME / 'source.json'
            if not config.exists():
                raise
            try:
                local = Path(json.loads(config.read_text())['repository']).resolve(strict=True)
                if _git(['remote', 'get-url', 'origin'], local).removesuffix('.git') != UPSTREAM.removesuffix('.git'):
                    raise ValueError('local repository origin is not OSTADIX-BM')
            except (OSError, ValueError, KeyError, TypeError) as source_error:
                raise PairError('Invalid OSTADIX-BM source configuration: ' + str(source_error)) from source_error
            report(str(error) + '; checking the installed OSTADIX-BM source instead.')
            revision = _git(['rev-parse', 'HEAD'], local)
            fetch_source = str(local)
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise PairError('The upstream repository did not return a valid HEAD revision')
        try:
            previous = load_pair()
        except PairError:
            previous = None
        if previous and previous['revision'] == revision:
            previous['checked_at'] = _now()
            _write_manifest(previous)
            report(f'OSTADIX-BM {revision[:12]} is current and verified.')
            return previous
        root = PAIR_HOME / 'checkouts' / revision
        root.parent.mkdir(parents=True, exist_ok=True)
        if not (root / '.git').is_dir():
            root.mkdir(exist_ok=True)
            _git(['init', '--quiet'], root)
            _git(['remote', 'add', 'origin', UPSTREAM], root)
        if _git(['remote', 'get-url', 'origin'], root) != UPSTREAM:
            raise PairError('Managed OSTADIX checkout has an unexpected origin')
        _git(['fetch', '--depth', '1', fetch_source, revision], root, timeout=180)
        _git(['checkout', '--detach', revision], root)
        _git(['diff', '--exit-code', 'HEAD', '--'], root)
        # Reuse dependency artifacts by copying app-owned caches only. Cargo still
        # verifies its fingerprints and builds every runtime from this checkout.
        if previous:
            for relative in ('target', 'mcp/ostadix_lang_mcp_server/target'):
                old = Path(previous['root']) / relative
                target = root / relative
                if old.is_dir() and not target.exists():
                    report('Reusing OSTADIX dependency build artifacts…')
                    shutil.copytree(old, target)
        log = PAIR_HOME / ('build-' + revision[:12] + '.log')
        report(f'Building matched OSTADIX runtime and MCP {revision[:12]}; log: {log}')
        commands = [
            ['cargo', 'build', '--release', '--locked', '--target-dir', str(root / 'target'),
             '--package', 'o-lang', '--bins'],
            ['cargo', 'build', '--release', '--locked', '--manifest-path',
             'mcp/ostadix_lang_mcp_server/Cargo.toml', '--target-dir',
             str(root / 'mcp/ostadix_lang_mcp_server/target')],
        ]
        build_environment = _network_environment()
        build_environment.pop('CARGO_BUILD_TARGET', None)
        try:
            with log.open('a') as output:
                for command in commands:
                    output.write('\n' + ' '.join(command) + '\n')
                    output.flush()
                    code = _run_build(command, root, output, build_environment)
                    if code:
                        raise PairError(f'OSTADIX build failed (exit {code}); see {log}')
        except (OSError, subprocess.TimeoutExpired) as error:
            raise PairError(f'OSTADIX build did not finish: {error}; see {log}') from error
        binaries = {name: str(root / 'target' / 'release' / name) for name in BINS}
        binaries['ostadix-mcp'] = str(root / 'mcp' / 'ostadix_lang_mcp_server' / 'target' / 'release' / 'ostadix-mcp')
        pair = {'root': str(root), 'backends': str(root / 'backends'),
                'binary': binaries['ostadix-mcp'], 'revision': revision,
                'upstream': UPSTREAM, 'binaries': binaries,
                'sha256': {name: _hash(path) for name, path in binaries.items()}}
        report('Checking the native MCP and executing the harmless 1 + 1 example…')
        pair['health'] = _health(pair)
        pair['verified_at'] = pair['checked_at'] = _now()
        _write_manifest(pair)
        report(f'Activated verified OSTADIX {revision[:12]}.')
        return pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--update', action='store_true', help='Fetch, build, and verify the latest official pair')
    mode.add_argument('--status', action='store_true', help='Verify and show the installed pair')
    args = parser.parse_args()
    try:
        pair = update_pair(print) if args.update else load_pair()
        print(json.dumps(pair, indent=2))
    except PairError as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
