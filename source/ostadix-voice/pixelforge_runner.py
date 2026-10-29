"""Fixed root-only entrypoint for explicit, bounded converted Huihui NPU probes."""
import hashlib
import ctypes
import json
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time
import uuid

ROOT = Path('/data/adb/ostadix-pixelforge')
INSTALLED = ROOT / 'run.py'
ROOTFS = Path('/data/data/com.termux/files/usr/var/lib/proot-distro/containers/ubuntu/rootfs')
APP = Path(os.environ.get('PIXELFORGE_VOICE_APP', str(Path.home() / 'ostadix-voice')))
MODEL = APP / 'runtime/models/Huihui-gemma-4-E2B-it-abliterated-v2.Q6_K.gguf'
MODEL_SHA = 'd58c9604501caf56d272f5abc848b18d05970b3ccad3b019eb32df072f1871c0'
ARTIFACTS = APP / 'runtime/pixelforge-gemma4-int8'
KIT = Path(os.environ.get('PIXELFORGE_KIT', '/opt/pixelforge'))
PYTHON = Path(os.environ.get('PIXELFORGE_PYTHON', str(KIT / 'custom-models/gguf-runtime/venv/bin/python')))
ENGINE = KIT / 'custom-models/gguf-runtime/bin/pixelforge-gguf'
NATIVE = Path('/data/local/tmp/pixelforge-npu/npu_pipe_gguf')
DEADLINE = 110
JOB_LIMIT = 16 * 1024 ** 2


def host(path):
    return ROOTFS / path.relative_to('/') if str(path).startswith('/root/') else path


def digest(path, deadline=None):
    value = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    with os.fdopen(descriptor, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('Pinned asset is not a regular file')
        for block in iter(lambda: stream.read(4 * 1024 ** 2), b''):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError('NPU probe asset verification exceeded its deadline')
            value.update(block)
    return value.hexdigest()


def json_file(path, limit, root_owned=False):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(descriptor, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or
                root_owned and (info.st_uid != 0 or info.st_mode & 0o022)):
            raise ValueError('Unsafe evidence file')
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError('Evidence exceeds its size limit')
    return json.loads(content)


def read_request(source):
    content = source.readline(8193)
    if len(content) > 8192:
        raise ValueError('Request exceeds 8192 bytes')
    request = json.loads(content)
    if not isinstance(request, dict) or request.get('action') not in ('status', 'generate'):
        raise ValueError('Use action status or generate')
    allowed = {'action'} if request['action'] == 'status' else {'action', 'prompt', 'tokens'}
    if request.keys() - allowed:
        raise ValueError('Unexpected request fields')
    if request['action'] == 'generate':
        prompt, tokens = request.get('prompt'), request.get('tokens', 1)
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000 or '\x00' in prompt:
            raise ValueError('Prompt must contain 1 to 2000 characters without NUL')
        if type(tokens) is not int or not 1 <= tokens <= 2:
            raise ValueError('Converted NPU probes support only 1 or 2 tokens')
        request['tokens'] = tokens
    return request


def pinned_files():
    files = {'metadata': ARTIFACTS / 'MODEL.json', 'plan': ARTIFACTS / 'weights.plan',
             'engine': host(ENGINE), 'python': host(PYTHON), 'native': NATIVE}
    for path in sorted((host(KIT) / 'tools/arm_npu').rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.json'):
            files['kit/' + str(path.relative_to(host(KIT)))] = path
    files['benchmark'] = host(KIT) / 'benchmark.py'
    return files


def verify(full_model=False, deadline=None):
    pins = json_file(ROOT / 'pins.json', 131072, root_owned=True)
    files = pinned_files()
    if pins.get('version') != 1 or set(pins.get('sha256', {})) != set(files):
        raise ValueError('Installed PixelForge pin set changed')
    for name, path in files.items():
        if path.is_symlink() or digest(path, deadline=deadline) != pins['sha256'][name]:
            raise ValueError('Pinned PixelForge asset changed: ' + name)
    metadata = json_file(ARTIFACTS / 'MODEL.json', 2 * 1024 ** 2)
    if (metadata.get('complete') is not True or metadata.get('completed_weight_dtypes') != ['int8'] or
            len(metadata.get('artifacts', {})) != 556 or
            metadata.get('source_gguf', {}).get('sha256') != MODEL_SHA or
            metadata.get('source_gguf', {}).get('path') != str(MODEL) or MODEL.is_symlink() or
            MODEL.stat().st_size != metadata['source_gguf'].get('bytes')):
        raise ValueError('Converted Huihui profile is incomplete or changed')
    if full_model and digest(MODEL, deadline=deadline) != MODEL_SHA:
        raise ValueError('Pinned Huihui GGUF changed')
    return metadata


def ubuntu_handles():
    """Use a verified live Ubuntu view; the bare rootfs omits Android mounts."""
    expected = ROOTFS.stat()
    files = pinned_files()
    files['model'] = MODEL
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        record = process_record(int(entry.name))
        if record is None:
            continue
        root = entry / 'root'
        try:
            info = root.stat()
            if (info.st_dev, info.st_ino) != (expected.st_dev, expected.st_ino):
                continue
            for name, path in files.items():
                relative = path.relative_to(ROOTFS) if path.is_relative_to(ROOTFS) else path.relative_to('/')
                actual, reference = (root / relative).stat(), path.stat()
                if (actual.st_dev, actual.st_ino) != (reference.st_dev, reference.st_ino):
                    break
            else:
                root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
                try:
                    namespace_fd = os.open(entry / 'ns/mnt', os.O_RDONLY | os.O_CLOEXEC)
                except BaseException:
                    os.close(root_fd)
                    raise
                current = process_record(record['pid'])
                if current and current['stamp'] == record['stamp']:
                    return root_fd, namespace_fd
                os.close(root_fd)
                os.close(namespace_fd)
        except OSError:
            continue
    raise RuntimeError('No verified mounted Ubuntu session is available; open the existing Ubuntu environment first')


def process_record(pid):
    try:
        process = Path('/proc') / str(pid)
        fields = (process / 'stat').read_text().rsplit(')', 1)[1].split()
        if process.stat().st_uid != 0 or fields[0] == 'Z':
            return None
        return {'pid': pid, 'parent': int(fields[1]), 'stamp': fields[19]}
    except (OSError, ValueError, IndexError):
        return None


def open_pidfd(pid):
    if hasattr(os, 'pidfd_open'):
        return os.pidfd_open(pid)
    # Android Python omits these wrappers; the device kernel exposes the Linux
    # pidfd syscalls at the generic arm64 syscall numbers.
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    descriptor = libc.syscall(ctypes.c_long(434), ctypes.c_int(pid), ctypes.c_uint(0))
    if descriptor < 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    return descriptor


def send_pidfd(descriptor, number):
    if hasattr(signal, 'pidfd_send_signal'):
        return signal.pidfd_send_signal(descriptor, number)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(424), ctypes.c_int(descriptor), ctypes.c_int(number),
                          ctypes.c_void_p(), ctypes.c_uint(0))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def collect_owned(owned):
    records = {}
    for entry in Path('/proc').iterdir():
        if entry.name.isdigit():
            record = process_record(int(entry.name))
            if record:
                records[record['pid']] = record
    roots = {pid for pid, stamp in owned.items() if pid in records and records[pid]['stamp'] == stamp}
    while True:
        found = {pid for pid, record in records.items() if record['parent'] in roots}
        if found <= roots:
            break
        roots.update(found)
    for pid in roots:
        owned[pid] = records[pid]['stamp']


def signal_owned(owned, number):
    for pid, stamp in list(owned.items()):
        record = process_record(pid)
        if record is None or record['stamp'] != stamp:
            continue
        descriptor = None
        try:
            descriptor = open_pidfd(pid)
            current = process_record(pid)
            if current and current['stamp'] == stamp:
                send_pidfd(descriptor, number)
        except ProcessLookupError:
            pass
        finally:
            if descriptor is not None:
                os.close(descriptor)


def cleanup(process, owned):
    # Freeze only the owned tree before its final snapshot, preventing an exit
    # or concurrent fork from orphaning a child during cancellation.
    for _ in range(2):
        collect_owned(owned)
        signal_owned(owned, signal.SIGSTOP)
    signal_owned(owned, signal.SIGTERM)
    signal_owned(owned, signal.SIGCONT)
    deadline = time.monotonic() + 2
    def active():
        for pid, stamp in owned.items():
            record = process_record(pid)
            if record and record['stamp'] == stamp:
                return True
        return False
    while active():
        if time.monotonic() >= deadline:
            signal_owned(owned, signal.SIGKILL)
            break
        time.sleep(.05)
    if not owned and process.poll() is None:
        process.terminate()
    process.wait(timeout=2)


def child_code():
    return ('import json,runpy,sys; r=json.load(sys.stdin); '
            'sys.path.insert(0,' + repr(str(KIT / 'tools/arm_npu')) + '); '
            'sys.argv=[' + repr(str(KIT / 'tools/arm_npu/run_gguf.py')) + ',"--model",' + repr(str(MODEL)) +
            ',"--artifacts",' + repr(str(ARTIFACTS)) + ',"--read-only","--engine",' + repr(str(ENGINE)) +
            ',"--prompt",r["prompt"],"--tokens",str(r["tokens"]),"--threads","1",'
            '"--cache-mib","512","--output",r["output"],"--timeout","100"]; '
            'runpy.run_path(' + repr(str(KIT / 'tools/arm_npu/run_gguf.py')) + ',run_name="__main__")')


def generate(request):
    started = time.monotonic()
    verify(full_model=True, deadline=started + DEADLINE)
    jobs = ROOT / 'jobs'
    jobs.mkdir(mode=0o700, exist_ok=True)
    job = jobs / uuid.uuid4().hex
    job.mkdir(mode=0o700)
    output = job / 'result'
    environment = {'HOME': '/root', 'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8',
                   'XDG_STATE_HOME': str(job / 'state'), 'PIXELFORGE_NPU_FAILURE_DIR': str(job / 'failures')}
    root_fd, namespace_fd = ubuntu_handles()
    def enter():
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.setns(namespace_fd, 0):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        os.fchdir(root_fd)
        os.chroot('.')
        os.chdir(KIT)
        os.close(root_fd)
        os.close(namespace_fd)
    process = None
    owned = {}
    try:
        with (job / 'stdout.log').open('xb') as stdout, (job / 'stderr.log').open('xb') as stderr:
            try:
                process = subprocess.Popen([str(PYTHON), '-I', '-c', child_code()], stdin=subprocess.PIPE,
                                           stdout=stdout, stderr=stderr, env=environment,
                                           preexec_fn=enter, start_new_session=True,
                                           pass_fds=(root_fd, namespace_fd))
            finally:
                os.close(root_fd)
                os.close(namespace_fd)
            record = process_record(process.pid)
            if record is None:
                raise RuntimeError('NPU probe process did not become identifiable')
            owned[process.pid] = record['stamp']
            process.stdin.write(json.dumps(dict(request, output=str(output))).encode())
            process.stdin.close()
            while process.poll() is None:
                collect_owned(owned)
                if time.monotonic() - started >= DEADLINE:
                    raise TimeoutError('Converted NPU probe exceeded its bounded deadline; no request was replayed')
                if sum(path.stat().st_size for path in job.rglob('*') if path.is_file()) > JOB_LIMIT:
                    raise RuntimeError('NPU probe evidence exceeded its size limit')
                time.sleep(.25)
        result = json_file(output / 'RESULT.json', 4 * 1024 ** 2, root_owned=True)
        npu = result.get('broker', {}).get('npu', {})
        if (process.returncode or result.get('success') is not True or not result.get('engine_cleanup_ok') or
                not result.get('broker_cleanup_ok') or npu.get('compiled_models') != 0 or
                not isinstance(npu.get('npu_invocations'), int) or npu['npu_invocations'] < 1):
            raise RuntimeError('Converted NPU probe failed; inspect its private root evidence')
        return {'ok': True, 'backend': 'converted-huihui-npu-dense+ggml-cpu',
                'text': result['engine'].get('text', '')[:2000],
                'generated_tokens': result['engine'].get('generated_tokens'),
                'npu_invocations': npu['npu_invocations'], 'compiled_models': 0,
                'seconds': round(time.monotonic() - started, 3), 'report': str(output / 'RESULT.json')}
    finally:
        if process is not None:
            for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(number, signal.SIG_IGN)
            cleanup(process, owned)


def main():
    info = INSTALLED.lstat()
    if (os.getuid() != 0 or Path(__file__).resolve() != INSTALLED or
            not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022):
        raise ValueError('Use only the fixed root-owned installed PixelForge launcher')
    os.umask(0o077)
    def interrupted(*_):
        raise InterruptedError('NPU probe cancelled; no request was replayed')
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(number, interrupted)
    request = read_request(sys.stdin.buffer)
    if request['action'] == 'status':
        metadata = verify()
        handles = ubuntu_handles()
        for descriptor in handles:
            os.close(descriptor)
        result = {'ok': True, 'available': True, 'backend': 'converted-huihui-npu-dense+ggml-cpu',
                  'model_sha256': MODEL_SHA, 'compiled_banks': len(metadata['artifacts']),
                  'tokens_limit': 2, 'deadline_seconds': DEADLINE + 4,
                  'reference_generation_seconds_for_2_tokens': 80.5,
                  'graph_verification': 'Saved bank hashes and device fingerprint are checked when loaded; compilation disabled.'}
    else:
        result = generate(request)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, TimeoutError, InterruptedError) as error:
        print(json.dumps({'ok': False, 'error': str(error)[:500]}))
        raise SystemExit(1)
