"""Pinned PixelForge inference service. Model output never executes host actions."""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parent
KIT = ROOT / 'kit'
PROFILE = ROOT / 'profile.json'
FRAME_LIMIT = 131072
BACKEND = 'pixelforge-npu-dense+ggml-cpu'
read_profile_counter = None


# -I keeps the launch isolated; only this installed source directory is added.
sys.path.insert(0, str(ROOT))
from config import sha256, root_json
import lifecycle


def read_profile():
    profile = root_json(PROFILE)
    if (profile.get('backend') != BACKEND or not profile['model']['provenance_repo'].startswith('huihui-ai/')
            or profile.get('architecture') not in ('gemma4', 'qwen35')):
        raise ValueError('Profile is not a supported Huihui decoder model')
    model = Path(profile['model_path'])
    if model.is_symlink() or not model.is_file() or sha256(model) != profile['model']['model_sha256']:
        raise ValueError('PixelForge source model SHA-256 mismatch')
    artifacts = Path(profile['artifacts'])
    manifest = artifacts / 'MODEL.json'
    if manifest.is_symlink() or sha256(manifest) != profile['artifacts_manifest_sha256']:
        raise ValueError('PixelForge compiled artifact manifest SHA-256 mismatch')
    if sha256(profile['native_helper_path']) != profile['native_helper_sha256']:
        raise ValueError('PixelForge native helper SHA-256 mismatch')
    for relative, expected in profile['runtime_sha256'].items():
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT) or path.is_symlink() or sha256(path) != expected:
            raise ValueError('PixelForge runtime pin mismatch: ' + relative)
    return profile


def renderer(template):
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def fail(message):
        raise ValueError(str(message)[:300])
    environment.globals['raise_exception'] = fail
    compiled = environment.from_string(template)
    def render(prompt):
        return compiled.render(messages=[{'role': 'user', 'content': prompt}],
                               add_generation_prompt=True, enable_thinking=False,
                               tools=[], bos_token='<bos>', eos_token='<eos>')
    return render


def validate_request(request):
    if not isinstance(request, dict) or request.keys() - {'id', 'prompt', 'max_tokens', 'context_size', 'prefill_cpu'}:
        raise ValueError('Unexpected PixelForge request fields')
    ident = request.get('id')
    if not isinstance(ident, str) or not ident or len(ident) > 128 or not all(c.isalnum() or c in '-_.' for c in ident):
        raise ValueError('Invalid PixelForge request identity')
    prompt = request.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 48000 or '\0' in prompt:
        raise ValueError('PixelForge prompt must contain 1 to 48000 characters without NUL')
    count = request.get('max_tokens', 512)
    context = request.get('context_size', 8192)
    if type(count) is not int or not 2 <= count <= 512:
        raise ValueError('PixelForge requires 2 to 512 output tokens')
    if type(context) is not int or not 256 <= context <= 8192 or request.get('prefill_cpu', True) is not True:
        raise ValueError('Invalid PixelForge context or explicit CPU-prefill setting')
    return dict(id=ident, prompt=prompt, max_tokens=count, context_size=context, prefill_cpu=True)


def hardware_counter():
    counter_path = read_profile_counter
    if not counter_path:
        return {'tpu_inference_count': None}
    path = Path(counter_path)
    try:
        return {'tpu_inference_count': path.read_text().strip()[:200]}
    except OSError:
        return {'tpu_inference_count': None}



def adopt_broker(cleanup, owned, path):
    """Recover the broker identity even if its original engine parent crashed."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except FileNotFoundError:
        return
    with os.fdopen(fd, 'rb') as source:
        info = os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise ValueError('Invalid PixelForge broker ownership file')
        data = source.read(4097)
    if len(data) > 4096:
        raise ValueError('Oversized PixelForge broker ownership file')
    owner = json.loads(data)
    if type(owner.get('pid')) is not int or owner['pid'] < 1 or not str(owner.get('start_ticks', '')).isdecimal():
        raise ValueError('Invalid PixelForge broker identity')
    record = cleanup.process_record(owner['pid'])
    if record and record['stamp'] == owner['start_ticks']:
        owned[record['pid']] = record['stamp']
        cleanup.collect_owned(owned)


def track_owned(cleanup, owned, stopped):
    """Keep periodic discovery cheap; final cleanup never waits on this timer."""
    interval = .2
    while not stopped.wait(interval):
        interval = .2 if cleanup.collect_owned_children(owned) else 2.0


def main():
    global read_profile_counter
    if len(sys.argv) != 1:
        raise ValueError('Service reads only its installed local profile')
    if os.getuid() != 0:
        raise ValueError('Use the fixed installed PixelForge chat launcher')
    os.umask(0o077)
    profile = read_profile()
    read_profile_counter = profile.get('hardware_counter_path')
    state = Path(profile['state_dir'])
    lock = (state / 'chat.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('Another PixelForge chat owns the NPU runtime; close it first')
    sys.path.insert(0, str(KIT))
    from gguf_hf_loader import GGUFWeights
    from model_architecture import build_plan, write_bridge_plan
    from compiled_model import ArtifactCache
    weights = GGUFWeights(profile['model_path'])
    plan = build_plan(weights, model_sha256=profile['model']['model_sha256'], bank_rows=profile['bank_rows'])
    if not plan['generation_adapter_supported'] or plan['architecture'] != profile['architecture']:
        raise ValueError('This model has unsupported PixelForge generation operations')
    artifacts = Path(profile['artifacts'])
    cache = ArtifactCache(artifacts, profile['model_path'], source_sha256=profile['model']['model_sha256'],
                          source_identity=weights.source_identity, verify_source=False, read_only=True, plan=plan)
    selection = profile.get('selected_npu_weight_tensors')
    if selection is None:
        if not cache._manifest().get('complete'):
            raise ValueError('PixelForge full offload requires a complete compiled artifact set')
    else:
        planned = {item['name']: item for item in plan['dense_tensors']}
        if (not isinstance(selection, list) or not selection or len(set(selection)) != len(selection)
                or any(name not in planned for name in selection)):
            raise ValueError('PixelForge selected NPU placements are invalid')
        # A partial package is sufficient only when every selected NPU bank is
        # present and verified. All other planned weights are explicitly CPU.
        for name in selection:
            item = planned[name]
            for begin in range(0, item['rows'], item['bank_rows']):
                cache.get(name, begin, min(begin + item['bank_rows'], item['rows']),
                          'int8', item['columns'], load_code=False)
    render = renderer(weights.fields['tokenizer.chat_template'])
    jobs = state / 'chat-jobs'
    jobs.mkdir(mode=0o700, exist_ok=True)
    job = jobs / uuid.uuid4().hex
    job.mkdir(mode=0o700)
    write_bridge_plan(plan, job / 'weights.plan')
    config = {'model': profile['model_path'], 'model_sha256': profile['model']['model_sha256'],
              'bank_rows': profile['bank_rows'], 'weight_dtype': 'int8',
              'cache_mib': profile.get('cache_mib', 512), 'artifacts': profile['artifacts'], 'read_only': True,
              'ownership': str(job / 'BROKER_OWNER.json'), 'report': str(job / 'broker.json'),
              'native_command': ['/usr/sbin/chroot', '/proc/1/root', '/system/bin/env',
                  'LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:' + profile['termux_prefix'] + '/lib',
                  'LD_PRELOAD=/system/lib64/libbinder_ndk.so', profile['native_helper_path']]}
    (job / 'broker-config.json').write_text(json.dumps(config))
    import shlex
    command = [sys.executable, '-I', '-c',
               'import runpy,sys;sys.path.insert(0,' + repr(str(KIT)) + ');runpy.run_path(' +
               repr(str(KIT / 'gguf_broker.py')) + ',run_name="__main__")', '--config', str(job / 'broker-config.json')]
    launcher = job / 'broker-launch.sh'
    launcher.write_text('#!/bin/sh\nexec ' + shlex.join(command) + '\n')
    launcher.chmod(0o700)
    env = {'HOME': '/root', 'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8',
           'PIXELFORGE_GGML_PLAN': str(job / 'weights.plan'), 'PIXELFORGE_GGML_BROKER': str(launcher),
           'PIXELFORGE_NPU_STDERR': str(job / 'native.stderr'),
           'PIXELFORGE_NPU_FAILURE_DIR': str(job / 'failures')}
    if selection is not None:
        env['PIXELFORGE_GGML_NPU_TENSORS'] = ','.join(selection)
    # This is an individual matrix transport fault timer, never a turn deadline.
    cleanup = lifecycle
    stopped = threading.Event()
    output_lock, request_lock = threading.Lock(), threading.Lock()
    current = {}
    owned = {}
    process = None
    tracker = reader = None
    def emit(value):
        with output_lock:
            print(json.dumps(value, ensure_ascii=False), flush=True)
    def interrupted(*_):
        raise InterruptedError('PixelForge chat closed; unfinished work was not replayed')
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, interrupted)
    try:
        with (job / 'engine.stderr').open('xb') as log:
            process = subprocess.Popen([str(ROOT / 'bin/pixelforge-chat'), profile['model_path'], '4'],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                    text=True, encoding='utf-8', env=env, start_new_session=True)
            record = cleanup.process_record(process.pid)
            if record is None:
                raise RuntimeError('Could not identify owned PixelForge engine')
            owned[process.pid] = record['stamp']
            def output():
                try:
                    for line in iter(lambda: process.stdout.readline(FRAME_LIMIT + 1), ''):
                        if len(line) > FRAME_LIMIT or not line.endswith('\n'):
                            raise ValueError('Invalid PixelForge native response frame')
                        frame = json.loads(line)
                        kind = frame.get('type')
                        if kind == 'ready':
                            emit(dict(type='ready', backend=BACKEND, model=profile['model'], report=str(job)))
                            continue
                        with request_lock:
                            if frame.get('id') != current.get('id') or not current:
                                raise ValueError('Native response does not match the active request')
                            if kind == 'done':
                                frame.update(backend=BACKEND, model=profile['model'], report=str(job),
                                             hardware_before=current['hardware_before'], hardware_after=hardware_counter())
                                if type(frame.get('npu_weight_matmul_operations')) is not int or frame['npu_weight_matmul_operations'] <= 0:
                                    frame = dict(type='error', id=frame['id'], error='No NPU weight operations were verified; response rejected')
                                current.clear()
                            elif kind == 'error':
                                current.clear()
                            elif kind != 'chunk':
                                raise ValueError('Unknown PixelForge native event')
                        if kind == 'done':
                            with (job / 'turns.jsonl').open('a') as receipts:
                                receipts.write(json.dumps(frame, ensure_ascii=False) + '\n')
                        emit(frame)
                    if not stopped.is_set():
                        raise RuntimeError('PixelForge engine disconnected; request was not replayed')
                except Exception as error:
                    emit(dict(type='error', id=current.get('id'), error=str(error)[:500]))
                    if not stopped.is_set():
                        os.kill(os.getpid(), signal.SIGTERM)
            reader = threading.Thread(target=output, name='pixelforge-output', daemon=True)
            reader.start()
            tracker = threading.Thread(target=track_owned, args=(cleanup, owned, stopped),
                                       name='pixelforge-owned', daemon=True)
            tracker.start()
            for line in iter(lambda: sys.stdin.buffer.readline(FRAME_LIMIT + 1), b''):
                if len(line) > FRAME_LIMIT or not line.endswith(b'\n'):
                    raise ValueError('PixelForge input frame exceeds its limit')
                request = validate_request(json.loads(line))
                with request_lock:
                    if current:
                        raise ValueError('PixelForge engine already has an active request')
                    current.update(id=request['id'], hardware_before=hardware_counter())
                request['prompt'] = render(request['prompt'])
                process.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
                process.stdin.flush()
    finally:
        stopped.set()
        for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(number, signal.SIG_IGN)
        if tracker is not None:
            tracker.join(timeout=2)
        if process is not None:
            with contextlib.suppress(OSError):
                process.stdin.close()
            # Idle EOF can flush Q and the broker receipt. Active cancellation
            # takes the immediate owned-tree shutdown path without a turn timer.
            if not current:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            adopt_broker(cleanup, owned, job / 'BROKER_OWNER.json')
            cleanup.cleanup(process, owned)
            if reader is not None:
                reader.join(timeout=2)
            process.stdout.close()
        weights.close()
        lock.close()


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, InterruptedError) as error:
        print(json.dumps({'type': 'error', 'error': str(error)[:500]}), flush=True)
        raise SystemExit(1)
