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
from config import sha256, root_json, runtime_options, preflight_artifacts, validate_phase_evidence, validate_flash_evidence, validate_prefix_evidence, tensor_selection, cpu_affinity_selection, model_limits
import lifecycle


def read_profile():
    profile = root_json(PROFILE)
    if (profile.get('backend') != BACKEND or not isinstance(profile.get('model'), dict)
            or not isinstance(profile.get('architecture'), str) or not profile['architecture']):
        raise ValueError('Profile is not a valid registered GGUF decoder model')
    model_limits(profile)
    model = Path(profile['model_path'])
    if model.is_symlink() or not model.is_file() or sha256(model) != profile['model']['model_sha256']:
        raise ValueError('PixelForge source model SHA-256 mismatch')
    artifacts = Path(profile['artifacts'])
    pins = profile.get('artifact_manifests_sha256')
    if pins is None:
        pins = {'MODEL.json': profile['artifacts_manifest_sha256']}
    if not isinstance(pins, dict) or not pins:
        raise ValueError('PixelForge artifact manifest pins are missing')
    for relative, expected in pins.items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or '..' in Path(relative).parts:
            raise ValueError('Invalid PixelForge artifact manifest pin path')
        manifest = artifacts / relative
        if (manifest.is_symlink() or not manifest.resolve().is_relative_to(artifacts.resolve())
                or sha256(manifest) != expected):
            raise ValueError('PixelForge compiled artifact manifest SHA-256 mismatch')
    if sha256(profile['native_helper_path']) != profile['native_helper_sha256']:
        raise ValueError('PixelForge native helper SHA-256 mismatch')
    for relative, expected in profile['runtime_sha256'].items():
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT) or path.is_symlink() or sha256(path) != expected:
            raise ValueError('PixelForge runtime pin mismatch: ' + relative)
    runtime_options(profile)
    return profile


def renderer(template, template_variables=None, *, prompt_mode=None):
    """Render this GGUF's template and tokenizer tokens, or explicit completion text."""
    if prompt_mode not in (None, 'chat-template', 'raw-completion'):
        raise ValueError('Unknown model prompt mode')
    if template_variables is not None and (not isinstance(template_variables, dict)
            or template_variables.keys() - {'bos_token', 'eos_token'}):
        raise ValueError('Invalid model tokenizer template variables')
    if prompt_mode == 'raw-completion' or (template is None and prompt_mode is None):
        if template is not None:
            raise ValueError('Raw completion must not override a model chat template')
        return lambda prompt: prompt
    if not isinstance(template, str) or not template or chr(0) in template:
        raise ValueError('Selected chat template must be a nonempty string without NUL')
    from datetime import datetime
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    def fail(message):
        raise ValueError(str(message)[:300])
    environment.globals['raise_exception'] = fail
    environment.globals['strftime_now'] = lambda fmt: datetime.now().strftime(fmt)
    compiled = environment.from_string(template)
    variables = dict(bos_token='', eos_token='')
    variables.update(template_variables or {})
    if any(not isinstance(value, str) or '\0' in value for value in variables.values()):
        raise ValueError('Invalid model tokenizer template variables')
    def render(prompt):
        return compiled.render(messages=[{'role': 'user', 'content': prompt}],
                               add_generation_prompt=True, enable_thinking=False,
                               tools=[], **variables)
    return render


def validate_request(request, profile=None):
    if not isinstance(request, dict) or request.keys() - {'id', 'prompt', 'max_tokens', 'context_size', 'prefill_cpu'}:
        raise ValueError('Unexpected PixelForge request fields')
    ident = request.get('id')
    if not isinstance(ident, str) or not ident or len(ident) > 128 or not all(c.isalnum() or c in '-_.' for c in ident):
        raise ValueError('Invalid PixelForge request identity')
    prompt = request.get('prompt')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 48000 or '\0' in prompt:
        raise ValueError('PixelForge prompt must contain 1 to 48000 characters without NUL')
    context_limit, output_limit = model_limits(profile)
    count = request.get('max_tokens', output_limit)
    context = request.get('context_size', context_limit)
    if type(count) is not int or not 2 <= count <= output_limit:
        raise ValueError('PixelForge output tokens exceed the selected model limit')
    cpu_prefill = runtime_options(profile or {})['cpu_prefill']
    if (type(context) is not int or not 256 <= context <= context_limit or count + 8 > context
            or request.get('prefill_cpu', cpu_prefill) is not cpu_prefill):
        raise ValueError('Invalid PixelForge context or configured prefill setting')
    return dict(id=ident, prompt=prompt, max_tokens=count, context_size=context, prefill_cpu=cpu_prefill)


def validate_prompt_mode(frame, profile):
    expected = profile.get('model', {}).get('prompt_mode')
    if expected is not None and frame.get('prompt_mode') != expected:
        raise ValueError('Native prompt token policy does not match the registered model')


def validate_ready(frame, profile):
    validate_prompt_mode(frame, profile)
    options = runtime_options(profile)
    validate_prefix_evidence(frame, profile, ready=True)
    validate_flash_evidence(frame, options, ready=True)
    if profile.get('phase_evidence_version') == 1:
        if frame.get('dense_npu_requested') is not True or frame.get('prefill_batch_size') != options['prefill_batch_size']:
            raise ValueError('Native readiness does not match the configured NPU runtime')
        for key in ('selected_npu_weight_tensors', 'selected_prefill_npu_weight_tensors'):
            if key not in frame or tensor_selection(frame[key]) != options[key]:
                raise ValueError('Native readiness changed the configured phase placement')
        if options['full_ffn'] or 'full_ffn_tensors' in frame:
            if frame.get('full_ffn_tensors') != options['full_ffn_tensors'] or frame.get('full_ffn_phases') != options['full_ffn_phases']:
                raise ValueError('Native readiness changed the configured feed-forward placement')


def broker_runtime_options(options):
    """Forward trusted runtime settings; retained DMA weights enable the model cap."""
    keys = ('batch_vectors', 'decode_batch_vectors', 'native_runtime_cache_mib', 'fused_gate_up',
            'fused_bank_rows', 'fused_gate_up_tensors', 'fused_prefill_only', 'fused_result_cache_mib',
            'full_ffn', 'full_ffn_tensors', 'full_ffn_phases', 'full_ffn_hidden_chunk', 'dynamic_weight_mode',
            'weight_buffer_cache_mib', 'npu_performance_mode')
    result = {key: options[key] for key in keys}
    if options['weight_buffer_cache_mib']:
        result['native_max_resident_models'] = options['native_max_resident_models']
    return result


def apply_cpu_affinity(requested):
    """Constrain this process before spawning children; default keeps its mask."""
    requested = cpu_affinity_selection(requested)
    if requested is None:
        return None
    allowed = os.sched_getaffinity(0)
    if not set(requested).issubset(allowed):
        raise ValueError('Configured cpu_affinity includes a CPU outside the allowed process mask')
    os.sched_setaffinity(0, requested)
    actual = sorted(os.sched_getaffinity(0))
    if actual != requested:
        raise ValueError('Configured cpu_affinity did not match the applied process mask')
    return actual


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


def verify_selected_bundle(state, profile):
    """Reject a launch whose selected bootstrap changed before it acquired the lock."""
    registry_path = state / 'models.json'
    if not registry_path.exists():
        return
    registry = root_json(registry_path)
    selected = registry.get('selected_model_id')
    models = registry.get('models')
    if not isinstance(models, dict) or selected not in models:
        raise ValueError('No registered model is currently selected')
    bootstrap = root_json(state / 'bootstrap.json')
    entry = models[selected]
    if (not isinstance(entry, dict) or bootstrap != entry.get('bootstrap')
            or bootstrap.get('service') != str(ROOT / 'service.py')
            or bootstrap.get('ubuntu_pins', {}).get(str(ROOT / 'profile.json')) != sha256(PROFILE)):
        raise ValueError('Selected model changed before inference startup; reopen the selected model')


def main():
    global read_profile_counter
    if len(sys.argv) != 1:
        raise ValueError('Service reads only its installed local profile')
    if os.getuid() != 0:
        raise ValueError('Use the fixed installed PixelForge chat launcher')
    os.umask(0o077)
    profile = read_profile()
    options = runtime_options(profile)
    actual_cpu_affinity = apply_cpu_affinity(options['cpu_affinity'])
    read_profile_counter = profile.get('hardware_counter_path')
    state = Path(profile['state_dir'])
    lock = (state / 'chat.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise ValueError('Another PixelForge chat owns the NPU runtime; close it first')
    verify_selected_bundle(state, profile)
    sys.path.insert(0, str(KIT))
    from gguf_hf_loader import GGUFWeights
    from model_architecture import build_plan, write_bridge_plan
    weights = GGUFWeights(profile['model_path'])
    plan = build_plan(weights, model_sha256=profile['model']['model_sha256'], bank_rows=profile['bank_rows'])
    if not plan['generation_adapter_supported'] or plan['architecture'] != profile['architecture']:
        raise ValueError('This model has unsupported PixelForge generation operations')
    manifest_pins = preflight_artifacts(profile, weights, plan)
    expected_pins = profile.get('artifact_manifests_sha256', {'MODEL.json': profile.get('artifacts_manifest_sha256')})
    if manifest_pins != expected_pins:
        raise ValueError('PixelForge phase bank manifests do not match the pinned profile')
    selection = options['selected_npu_weight_tensors']
    from model_info import describe_model, probe_model
    probe = profile.get('native_model_probe')
    if probe is None:
        probe = probe_model(ROOT / 'bin/pixelforge-chat', profile['model_path'])
    info = describe_model(weights, plan, probe=probe)
    if profile['model'].get('prompt_mode', info['prompt_mode']) != info['prompt_mode']:
        raise ValueError('Selected model prompt format does not match its GGUF')
    render = renderer(info['chat_template'], info['template_variables'], prompt_mode=info['prompt_mode'])
    jobs = state / 'chat-jobs'
    jobs.mkdir(mode=0o700, exist_ok=True)
    job = jobs / uuid.uuid4().hex
    job.mkdir(mode=0o700)
    write_bridge_plan(plan, job / 'weights.plan')
    config = {'model': profile['model_path'], 'model_sha256': profile['model']['model_sha256'],
              'bank_rows': profile['bank_rows'], 'weight_dtype': options['weight_dtype'],
              'cache_mib': profile.get('cache_mib', 512), 'artifacts': profile['artifacts'], 'read_only': True,
              'ownership': str(job / 'BROKER_OWNER.json'), 'report': str(job / 'broker.json'),
              'native_command': ['/usr/sbin/chroot', '/proc/1/root', '/system/bin/env',
                  'LD_LIBRARY_PATH=/system/lib64:/vendor/lib64:' + profile['termux_prefix'] + '/lib',
                  'LD_PRELOAD=/system/lib64/libbinder_ndk.so', profile['native_helper_path']]}
    config.update(broker_runtime_options(options))
    config['required_artifact_manifests'] = sorted(manifest_pins)
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
    env['PIXELFORGE_CHAT_PROMPT_MODE'] = info['prompt_mode']
    env['PIXELFORGE_CHAT_PREFILL_BATCH'] = str(options['prefill_batch_size'])
    env['PIXELFORGE_CHAT_FLASH_ATTN'] = options['flash_attention']
    env['PIXELFORGE_CHAT_PREFIX_CACHE_MIB'] = str(options['prefix_cache_mib'])
    if selection is not None:
        env['PIXELFORGE_GGML_NPU_TENSORS'] = ','.join(selection)
    if options['selected_prefill_npu_weight_tensors'] is not None:
        env['PIXELFORGE_GGML_PREFILL_NPU_TENSORS'] = ','.join(options['selected_prefill_npu_weight_tensors'])
    if options['full_ffn']:
        env['PIXELFORGE_GGML_FULL_FFN_TENSORS'] = ','.join(options['full_ffn_tensors'])
        env['PIXELFORGE_GGML_FULL_FFN_PHASES'] = options['full_ffn_phases']
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
                            validate_ready(frame, profile)
                            emit(dict(type='ready', backend=BACKEND, model=profile['model'], report=str(job),
                                      flash_attention=options['flash_attention'], attention_backend='CPU',
                                      cpu_flash_attention_enabled=None, cpu_affinity=actual_cpu_affinity,
                                      prefix_cache_mib=options['prefix_cache_mib']))
                            continue
                        with request_lock:
                            if frame.get('id') != current.get('id') or not current:
                                raise ValueError('Native response does not match the active request')
                            if kind == 'done':
                                frame.update(backend=BACKEND, model=profile['model'], report=str(job),
                                             hardware_before=current['hardware_before'], hardware_after=hardware_counter(),
                                             dynamic_weight_mode=options['dynamic_weight_mode'],
                                             cpu_affinity=actual_cpu_affinity)
                                try:
                                    validate_prompt_mode(frame, profile)
                                    validate_phase_evidence(frame, profile)
                                except ValueError as error:
                                    frame = dict(type='error', id=frame['id'], error=str(error))
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
                request = validate_request(json.loads(line), profile)
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
