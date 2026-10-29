"""Prepare a reviewable local chat installation; --apply writes only named outputs.

Run inside the already configured Ubuntu root environment. Weights and compiled
banks remain at their supplied paths and are never copied or downloaded.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import sha256

BACKEND = 'pixelforge-npu-dense+ggml-cpu'


def absolute(value):
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts:
        raise argparse.ArgumentTypeError('Use an absolute path without parent traversal')
    return path


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('ubuntu-root', 'python', 'model', 'artifacts', 'helper', 'termux-home', 'app-runtime'):
        result.add_argument('--' + name, type=absolute, required=True)
    result.add_argument('--model-name', required=True)
    result.add_argument('--provenance-repo', required=True)
    result.add_argument('--architecture', choices=('gemma4', 'qwen35'), required=True)
    result.add_argument('--select-tensors', help='Comma-separated explicit NPU placements; omit only for complete all-dense banks')
    result.add_argument('--bank-rows', type=int, default=4096)
    result.add_argument('--cache-mib', type=int, default=256)
    result.add_argument('--service-dir', type=absolute, default=Path('/opt/pixelforge-chat'))
    result.add_argument('--state-dir', type=absolute, default=Path('/data/adb/ostadix-pixelforge'))
    result.add_argument('--native-binary', type=absolute, default=ROOT / 'build/bin/pixelforge-chat')
    result.add_argument('--hardware-counter-path', type=absolute)
    result.add_argument('--apply', action='store_true')
    result.add_argument('--replace', action='store_true', help='Replace an idle existing chat installation; no model files change')
    return result


def prepare(args):
    if not args.provenance_repo.startswith('huihui-ai/') or len(args.provenance_repo.split('/')) != 2:
        raise ValueError('Supply the actual Huihui model provenance repository')
    if args.bank_rows < 1 or args.cache_mib < 0 or not args.model_name.strip():
        raise ValueError('Invalid bank rows, cache budget, or model name')
    if not args.app_runtime.is_relative_to(args.termux_home):
        raise ValueError('App runtime must be inside the supplied Termux home')
    termux_uid = args.termux_home.stat().st_uid
    if args.app_runtime.stat().st_uid != termux_uid:
        raise ValueError('App runtime must belong to the discovered Termux account')
    prefix = args.termux_home.parent / 'usr'
    if not args.python.is_file() or not os.access(args.python, os.X_OK) or not os.access(args.native_binary, os.X_OK) or not os.access(args.helper, os.X_OK):
        raise ValueError('Build the native runner and supply an executable Ubuntu Python first')
    kit = ROOT.parent / 'tools/arm_npu'
    sys.path.insert(0, str(kit))
    from gguf_hf_loader import GGUFWeights
    from model_architecture import build_plan
    with GGUFWeights(args.model) as weights:
        model_hash = weights.sha256()
        plan = build_plan(weights, model_sha256=model_hash, bank_rows=args.bank_rows)
        if not plan['generation_adapter_supported'] or plan['architecture'] != args.architecture:
            raise ValueError('Model does not match a supported architecture/complete dense plan')
        if not isinstance(weights.fields.get('tokenizer.chat_template'), str):
            raise ValueError('Model must contain its own chat template')
    manifest_path = args.artifacts / 'MODEL.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('source_gguf', {}).get('sha256') != model_hash:
        raise ValueError('Compiled bank manifest belongs to different model bytes')
    selection = args.select_tensors.split(',') if args.select_tensors is not None else None
    planned = {item['name'] for item in plan['dense_tensors']}
    if selection is not None and (not selection or len(set(selection)) != len(selection) or any(name not in planned for name in selection)):
        raise ValueError('Explicit NPU tensor selection contains empty, unknown, or duplicate names')
    if selection is None and not manifest.get('complete'):
        raise ValueError('All-dense placement requires a complete bank package')
    execution = ('CPU prompt processing; selected projections on NPU; remaining model operations on CPU'
                 if selection is not None else 'CPU prompt processing; all planned dense decode weights on NPU; other operations on CPU')
    metadata = {'model_name': args.model_name, 'provenance_repo': args.provenance_repo,
                'model_sha256': model_hash, 'backend': BACKEND, 'context_per_worker': 8192,
                'max_output_tokens': 512, 'cpu_prefill': True, 'npu_dense_decode': True,
                'streaming_preview': False, 'execution': execution,
                'selected_npu_weight_tensors': selection}
    sources = {path.name: path for path in ROOT.glob('*.py')}
    sources.update({'requirements-chat.txt': ROOT / 'requirements-chat.txt', 'bin/pixelforge-chat': args.native_binary})
    for path in kit.rglob('*'):
        if '__pycache__' not in path.parts and path.is_file() and (path.suffix in ('.py', '.json') or path.name.startswith('LICENSE')):
            sources['kit/' + str(path.relative_to(kit))] = path
    pins = {relative: sha256(path) for relative, path in sources.items()}
    profile = {'backend': BACKEND, 'architecture': args.architecture, 'model_path': str(args.model),
               'artifacts': str(args.artifacts), 'bank_rows': args.bank_rows, 'cache_mib': args.cache_mib,
               'state_dir': str(args.state_dir), 'termux_prefix': str(prefix),
               'native_helper_path': str(args.helper), 'native_helper_sha256': sha256(args.helper),
               'artifacts_manifest_sha256': sha256(manifest_path), 'runtime_sha256': pins,
               'selected_npu_weight_tensors': selection, 'model': metadata}
    if args.hardware_counter_path:
        profile['hardware_counter_path'] = str(args.hardware_counter_path)
    profile_bytes = (json.dumps(profile, ensure_ascii=False, indent=2) + '\n').encode()
    import hashlib
    ubuntu_pins = {str(args.service_dir / name): digest for name, digest in pins.items()}
    ubuntu_pins[str(args.service_dir / 'profile.json')] = hashlib.sha256(profile_bytes).hexdigest()
    ubuntu_pins[str(args.python)] = sha256(args.python.resolve())
    bootstrap = {'ubuntu_root': str(args.ubuntu_root), 'python': str(args.python),
                 'service': str(args.service_dir / 'service.py'), 'ubuntu_pins': ubuntu_pins}
    client = {'backend': BACKEND, 'experimental': True, 'model': metadata, 'launcher': str(args.state_dir / 'chat.py')}
    writes = [str(args.service_dir / name) for name in sources]
    writes += [str(args.service_dir / 'profile.json'), str(args.state_dir / 'bootstrap.json'),
               str(args.state_dir / 'chat.py'), str(args.state_dir / 'chat.lock'),
               str(args.service_dir / 'INSTALL_MANIFEST.json'), str(args.app_runtime / 'pixelforge-chat.json')]
    return {'schema': 'pixelforge.chat.install.v1', 'apply': args.apply, 'termux_uid': termux_uid,
            'writes': writes, 'model_copied': False, 'compiled_banks_copied': False,
            'source_pins': pins, 'profile': profile, 'bootstrap': bootstrap,
            'client_metadata': client}, sources, profile_bytes


def atomic(path, data, *, uid=0, mode=0o600, label_source=None):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary = tempfile.mkstemp(prefix='.pixelforge-install-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as destination:
            destination.write(data)
            destination.flush()
            os.fsync(destination.fileno())
        os.chown(temporary, uid, uid)
        os.chmod(temporary, mode)
        if label_source:
            import errno
            try:
                label = os.getxattr(label_source, 'security.selinux')
                os.setxattr(temporary, 'security.selinux', label)
            except OSError as error:
                if error.errno not in (errno.ENODATA, errno.ENOTSUP):
                    raise
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def apply(args, manifest, sources, profile_bytes):
    if os.getuid() != 0:
        raise ValueError('--apply requires the existing Ubuntu root environment')
    if args.termux_home.stat().st_uid == 0:
        raise ValueError('Termux home must belong to its application account, not root')
    here, configured = Path('/').stat(), args.ubuntu_root.stat()
    if (here.st_dev, here.st_ino) != (configured.st_dev, configured.st_ino):
        raise ValueError('Run --apply inside the configured Ubuntu root filesystem')
    for directory in (args.service_dir, args.state_dir):
        if directory.exists() and (directory.is_symlink() or directory.stat().st_uid != 0 or directory.stat().st_mode & 0o022):
            raise ValueError('Installation directories must be owned by root and not group/world writable')
    if not args.replace and any(Path(path).exists() or Path(path).is_symlink() for path in manifest['writes']):
        raise ValueError('Installation outputs already exist; review --replace for an idle installation')
    args.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    with (args.state_dir / 'chat.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for relative, source in sources.items():
            atomic(args.service_dir / relative, source.read_bytes(), mode=0o700 if relative == 'bin/pixelforge-chat' else 0o600)
        atomic(args.service_dir / 'profile.json', profile_bytes)
        atomic(args.state_dir / 'chat.py', (ROOT / 'android_launcher.py').read_bytes(), mode=0o700)
        atomic(args.state_dir / 'bootstrap.json', (json.dumps(manifest['bootstrap'], indent=2) + '\n').encode())
        atomic(args.app_runtime / 'pixelforge-chat.json', (json.dumps(manifest['client_metadata'], ensure_ascii=False, indent=2) + '\n').encode(),
               uid=manifest['termux_uid'], label_source=args.app_runtime)
        atomic(args.service_dir / 'INSTALL_MANIFEST.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())


def main():
    args = parser().parse_args()
    manifest, sources, profile_bytes = prepare(args)
    if args.apply:
        apply(args, manifest, sources, profile_bytes)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
