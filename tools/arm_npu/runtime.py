#!/usr/bin/env python3
"""Verify, install offline, and run the pinned Huihui model on this phone's NPU."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import venv

KIT = Path(__file__).resolve().parent
DEFAULT_ADDON = KIT.parents[1] / 'custom-models/minicpm5'
NATIVE_DESTINATION = Path('/data/local/tmp/pixelforge-npu/npu_pipe')


def checked_file(root, item):
    relative = PurePosixPath(item['path'])
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Manifest path must stay inside the addon')
    path = root.joinpath(*relative.parts)
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Missing or redirected addon file: {relative}')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    if path.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
        raise ValueError(f'Size or SHA256 mismatch: {relative}')
    return path


def verify_addon(root, *, all_files=True):
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    pins = json.loads((KIT / 'HUIHUI_ASSETS.json').read_text())
    expected = {
        'model.gguf': {'bytes': pins['quantized_gguf_bytes'], 'sha256': pins['quantized_gguf_sha256']},
        **{'tokenizer/' + item['path']: item for item in pins['support_files']},
    }
    entries = {}
    for item in manifest['files']:
        if item['path'] in entries:
            raise ValueError('Duplicate addon manifest path')
        entries[item['path']] = item
    for name, pin in expected.items():
        item = entries.get(name)
        if item is None or any(item[key] != pin[key] for key in ('bytes', 'sha256')):
            raise ValueError(f'Addon does not match published checkpoint pin: {name}')
    if 'native/npu_pipe' not in entries:
        raise ValueError('Offline addon needs its verified ARM64 native/npu_pipe helper')
    selected = entries if all_files else {name: entries[name] for name in [*expected, 'native/npu_pipe']}
    for item in selected.values():
        checked_file(root, item)
    return manifest


def install_native(source, destination=NATIVE_DESTINATION):
    """Atomically publish the helper in a path visible inside Android's root."""
    if os.geteuid() != 0:
        raise PermissionError('Native helper setup requires this rooted Android/Ubuntu session')
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if destination.is_symlink():
        raise ValueError('Native destination is a symlink')
    fd, temporary = tempfile.mkstemp(prefix='.npu_pipe-', dir=destination.parent)
    try:
        with os.fdopen(fd, 'wb') as output, source.open('rb') as input_file:
            shutil.copyfileobj(input_file, output)
            os.fchmod(output.fileno(), 0o700)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def setup(root):
    manifest = verify_addon(root)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Bundled wheels require CPython 3.12 on Linux ARM64')
    if os.uname().machine != 'aarch64':
        raise RuntimeError('Bundled wheels and native helper require ARM64')
    environment = root / 'venv'
    if environment.is_symlink():
        raise ValueError('Runtime venv must be a local directory')
    if not (environment / 'bin/python').exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([str(environment / 'bin/python'), '-I', '-m', 'pip', 'install', '--force-reinstall',
                    '--no-index', '--no-deps', '--require-hashes', '--find-links', str(root / 'wheels'),
                    '-r', str(root / 'requirements-offline.txt')], check=True)
    install_native(root / 'native/npu_pipe')
    print(json.dumps({'status': 'offline-setup-complete', 'verified_files': len(manifest['files']),
                      'python': str(environment / 'bin/python'), 'native': str(NATIVE_DESTINATION)}))


def worker(args):
    os.umask(0o077)
    # -I excludes this script directory from sys.path. Add only our bundled kit.
    sys.path.insert(0, str(KIT))
    report_path = args.report or args.addon / 'results' / f'run-{time.time_ns()}.json'
    report_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd = os.open(report_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    report = {'status': 'started', 'source': 'huihui-ai/Huihui-MiniCPM5-2B-abliterated',
              'backend': 'native-google-tensor-npu-fc', 'cpu_linear_fallback': False,
              'cpu_operations': 'tokenization, embeddings, RMSNorm, RoPE, attention, KV cache, SiLU, sampling',
              'prompt': args.prompt, 'token_timings': []}
    native = None
    started = time.monotonic()
    try:
        from gguf_hf_loader import GGUFWeights
        from model_host import MiniCPM5Host, tokenizer
        from npu_linear import NPULinear
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from benchmark import hardware_snapshot
        manifest = verify_addon(args.addon, all_files=False)
        assets = json.loads((KIT / 'HUIHUI_ASSETS.json').read_text())
        report['provenance'] = {
            'original_repository': assets['original_checkpoint_repo'],
            'original_revision': assets['original_checkpoint_revision'],
            'quantized_repository': assets['quantized_checkpoint_repo'],
            'quantized_revision': assets['quantized_checkpoint_revision'],
            'gguf_sha256': assets['quantized_gguf_sha256'],
            'model_and_support_sha256_verified': True,
            'quantization': 'Q8_0 GGUF dequantized and requantized per output row to int8 constant FC weights',
        }
        pin = next(item for item in manifest['files'] if item['path'] == 'native/npu_pipe')
        checked_file(NATIVE_DESTINATION.parent, {**pin, 'path': NATIVE_DESTINATION.name})
        report['provenance']['native_sha256'] = pin['sha256']
        report['provenance']['installed_native_sha256_verified'] = True
        weights = GGUFWeights(args.addon / 'model.gguf')
        tok = tokenizer(args.addon / 'tokenizer')
        prompt_ids = tok.apply_chat_template([{'role': 'user', 'content': args.prompt}],
                                             enable_thinking=False)
        report['prompt_ids'] = prompt_ids
        command = ['/usr/sbin/chroot', '/proc/1/root', '/system/bin/env',
                   'LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:/data/data/com.termux/files/usr/lib',
                   'LD_PRELOAD=/system/lib64/libbinder_ndk.so', str(NATIVE_DESTINATION)]
        native = NPULinear(weights, command=command, weight_dtype='int8', lm_bank_rows=4096)
        try:
            Path(f'/proc/{native.process.pid}/oom_score_adj').write_text('250')
        except OSError:
            pass
        report['hardware_before'] = hardware_snapshot(native.process.pid)
        runner = MiniCPM5Host(weights, linear=native)
        print(json.dumps({'event': 'start', 'prompt_tokens': len(prompt_ids),
                          'report': str(report_path), 'backend': report['backend']}), flush=True)
        def emitted(token, index, seconds):
            report['token_timings'].append({'id': token, 'seconds': seconds})
            print(json.dumps({'event': 'token', 'id': token, 'text': tok.decode([token]),
                              'seconds': seconds}), flush=True)
        output = runner.generate(prompt_ids, max_new_tokens=args.tokens, callback=emitted)
        report.update(status='completed', output_ids=output,
                      output=tok.decode(output, skip_special_tokens=True))
        report['hardware_after'] = hardware_snapshot(native.process.pid)
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        if native is not None:
            try:
                report['npu_stats'] = native.stats()
                native.close()
            except BaseException as error:
                report.update(status='failed', cleanup_error=repr(error))
        report['seconds'] = time.monotonic() - started
        with os.fdopen(fd, 'w') as destination:
            json.dump(report, destination, indent=2)
            destination.write('\n')
    print(json.dumps({'event': report['status'], 'output': report.get('output'),
                      'seconds': report['seconds'], 'report': str(report_path)}), flush=True)
    return 0 if report['status'] == 'completed' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('verify', 'setup', 'run', '_worker'))
    parser.add_argument('--addon', type=Path, default=DEFAULT_ADDON)
    parser.add_argument('--prompt', default='Hi')
    parser.add_argument('--tokens', type=int, default=8)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    args.addon = args.addon.resolve()
    if not 1 <= args.tokens <= 128:
        parser.error('--tokens must be 1..128')
    if args.action == 'verify':
        manifest = verify_addon(args.addon)
        print(json.dumps({'status': 'verified', 'files': len(manifest['files'])}))
    elif args.action == 'setup':
        setup(args.addon)
    elif args.action == '_worker':
        return worker(args)
    else:
        python = args.addon / 'venv/bin/python'
        if not python.exists():
            raise RuntimeError('Run runtime.py setup once to construct the offline Python environment')
        command = [str(python), '-I', str(Path(__file__).resolve()), '_worker', '--addon',
                   str(args.addon), '--prompt', args.prompt, '--tokens', str(args.tokens)]
        if args.report:
            command += ['--report', str(args.report.resolve())]
        os.execv(python, command)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
