#!/usr/bin/env python3
"""Bounded CPU stress of a frozen .O artifact and its embedded lifetime guard.

Requires an already verified/extracted compiler cache and a small existing
synthetic GGUF. Scheduled actions are read-only: help, inspect, plan-only,
verify-only and deliberately invalid requests. No setup, extraction, native
compiler, inference, hardware observation or service action is requested.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time

from bootstrap import _find_ostadix


HERE = Path(__file__).resolve().parent
TAIL_BYTES = 65536
MAX_MODEL_BYTES = 4 * 1024 * 1024
MAX_FIXTURE_BYTES = 8 * 1024 * 1024
PYTHON = '/usr/bin/python3'


def start_ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19]


def file_pin(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('Stress inputs must be regular files')
        for chunk in iter(lambda:source.read(1024*1024),b''):
            digest.update(chunk)
        after = os.fstat(source.fileno())
    keys = ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')
    identity = tuple(getattr(before,key) for key in keys)
    if identity != tuple(getattr(after,key) for key in keys) or identity != tuple(getattr(Path(path).stat(),key) for key in keys):
        raise ValueError('Stress input changed while hashing')
    return {'bytes':before.st_size,'sha256':digest.hexdigest(),'identity':list(identity)}


def embedded_worker(path):
    with Path(path).open('r',encoding='utf8') as source:
        prefix = source.read(262144)
    # The shebang itself contains both delimiter strings in its reader code.
    body = prefix.split('\n',1)[1].split('python^(',1)[1].split(')_python',1)[0]
    for node in ast.parse(body).body:
        if isinstance(node,ast.Assign) and any(isinstance(name,ast.Name) and name.id == 'WORKER_SOURCE' for name in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value,str) and value.strip():
                compile(value,'<artifact-worker>','exec')
                return value
    raise ValueError('Artifact does not contain its literal lifetime guard')


def private_json(path,value):
    descriptor = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(descriptor,'w') as target:
        json.dump(value,target,indent=2,ensure_ascii=False)
        target.write('\n')


def parsed_tail(data):
    for line in reversed(data.splitlines()):
        try:
            value = json.loads(line)
        except (TypeError,ValueError):
            continue
        if isinstance(value,dict):
            return value
    return None


def guarded_command(worker,status,command):
    return [PYTHON,str(worker),'--owner-pid',str(os.getpid()),
            '--owner-start-ticks',start_ticks(os.getpid()),'--grace-seconds','3',
            '--status-file',str(status),'--',*command]


def bounded_run(command,*,cwd,environment,timeout,cancel):
    started = time.monotonic()
    process = subprocess.Popen(command,cwd=cwd,env=environment,stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    tails = {'stdout':bytearray(),'stderr':bytearray()}
    totals = {'stdout':0,'stderr':0}
    selector = selectors.DefaultSelector()
    for name,stream in (('stdout',process.stdout),('stderr',process.stderr)):
        os.set_blocking(stream.fileno(),False)
        selector.register(stream,selectors.EVENT_READ,name)
    stopped = None
    forced = False
    try:
        while process.poll() is None or selector.get_map():
            now = time.monotonic()
            if stopped is None and (cancel.is_set() or now-started >= timeout):
                stopped = now
                if process.poll() is None:
                    process.terminate()  # The embedded outer guard cleans its children.
            if stopped is not None and now-stopped >= 15:
                if process.poll() is None:
                    process.kill()
                    forced = True
                break
            for key,_ in selector.select(.05):
                data = os.read(key.fileobj.fileno(),8192)
                if not data:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                totals[key.data] += len(data)
                tails[key.data].extend(data)
                if len(tails[key.data]) > TAIL_BYTES:
                    del tails[key.data][:-TAIL_BYTES]
        process.wait(timeout=5)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        selector.close()
        for stream in (process.stdout,process.stderr):
            if not stream.closed:
                stream.close()
    return {'returncode':process.returncode,'seconds':time.monotonic()-started,
            'deadline_or_cancellation':stopped is not None,'forced_guard_kill':forced,
            'stdout_bytes':totals['stdout'],'stderr_bytes':totals['stderr'],
            **{name+'_tail':bytes(value).decode('utf8',errors='replace') for name,value in tails.items()}}


def guard_fixtures(worker,directory,environment,timeout,cancel):
    destination = directory/'guard-fixtures.json'
    guard_status = directory/'guard-fixtures-owner.json'
    loader = ('import sys,json,pathlib,unittest;sys.path.insert(0,'+repr(str(HERE))+');'
              'import test_worker;test_worker.WORKER=pathlib.Path('+repr(str(worker))+');'
              'suite=unittest.defaultTestLoader.loadTestsFromModule(test_worker);'
              'result=unittest.TextTestRunner(stream=sys.stderr,verbosity=2).run(suite);'
              'value={"success":result.wasSuccessful(),"tests_run":result.testsRun,'
              '"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped)};'
              'pathlib.Path('+repr(str(destination))+').write_text(json.dumps(value));'
              'raise SystemExit(0 if value["success"] else 1)')
    command = guarded_command(worker,guard_status,[PYTHON,'-B','-I','-c',loader])
    captured = bounded_run(command,cwd=directory,environment=environment,timeout=timeout,cancel=cancel)
    receipt = json.loads(destination.read_text()) if destination.is_file() else {'success':False,'error':'Guard fixtures did not publish a result'}
    outer = json.loads(guard_status.read_text()) if guard_status.is_file() else {}
    receipt['outer_guard_cleanup_ok'] = outer.get('cleanup_ok')
    receipt['success'] = receipt['success'] and captured['returncode'] == 0 and outer.get('cleanup_ok') is True
    return {**receipt,**captured}


def cases(model,directory):
    long = [('雪'*16384)+f' literal $HOME;$(touch SENTINEL) {index}' for index in range(4)]
    missing = directory/'missing $(touch SENTINEL); 雪.gguf'
    return [
        ('help',['help'],'help'),
        ('global-help',['--help'],'help'),
        ('literal-help',['help','$HOME;$(touch SENTINEL)',"'quoted' 雪 café",'line\nbreak'],'help'),
        ('long-unicode-help',['help',*long],'help'),
        ('long-unicode-help-repeat',['help',*reversed(long)],'help'),
        ('inspect',['inspect',str(model)],'inspect'),
        ('inspect-tensors',['inspect',str(model),'--tensors'],'inspect'),
        ('inspect-repeat',['inspect',str(model)],'inspect'),
        ('plan',['plan','--model',str(model),'--output',str(directory/'plan-never-created')],'plan'),
        ('plan-banks',['plan','--model',str(model),'--output',str(directory/'banks-never-created'),'--bank-rows','64'],'plan'),
        ('plan-rows',['plan','--model',str(model),'--output',str(directory/'rows-never-created'),'--tensor','blk.0.attn_q.weight','--rows','0:128','--bank-rows','32'],'plan'),
        ('verify',['verify'],'verify'),
        ('verify-repeat',['verify'],'verify'),
        ('unknown-action',['unknown-action-雪'],'failure'),
        ('missing-inspect',['inspect',str(missing)],'failure'),
        ('missing-plan',['plan','--model',str(missing),'--output',str(directory/'missing-never-created')],'failure'),
        ('invalid-deadline',['--deadline','0','help'],'failure'),
        ('invalid-bank-rows',['plan','--model',str(model),'--output',str(directory/'invalid-never-created'),'--bank-rows','0'],'failure'),
    ]


def evaluate(name,arguments,expected,args,directory,worker,environment,cancel,model_sha,cache_identity):
    if not args.kit_dir.is_dir() or (args.kit_dir.stat().st_dev,args.kit_dir.stat().st_ino) != cache_identity:
        raise ValueError('Existing cache disappeared or changed; refusing materialization')
    status = directory/(name+'-owner.json')
    invocation = [str(args.artifact),'--kit-dir',str(args.kit_dir),'--python',str(args.numeric_python),*arguments]
    captured = bounded_run(guarded_command(worker,status,invocation),cwd=directory,
                           environment=environment,timeout=args.timeout,cancel=cancel)
    parsed = parsed_tail(captured['stdout_tail']) or parsed_tail(captured['stderr_tail'])
    outer = json.loads(status.read_text()) if status.is_file() else {}
    errors = []
    if captured['deadline_or_cancellation'] or captured['forced_guard_kill']:
        errors.append('Invocation exceeded its bound or was cancelled')
    if outer.get('cleanup_ok') is not True:
        errors.append('Outer owned-process cleanup was not confirmed')
    if expected == 'failure':
        if captured['returncode'] == 0 or (parsed and parsed.get('exit_code') == 0):
            errors.append('Expected failure returned success')
    else:
        if captured['returncode'] != 0 or not parsed or parsed.get('exit_code') != 0 or parsed.get('action') != expected:
            errors.append('Expected successful action/receipt was not observed')
        elif expected == 'inspect':
            result = parsed.get('result')
            if not isinstance(result,dict):
                try:
                    result = json.loads(parsed.get('child',{}).get('stdout_tail',''))
                except ValueError:
                    result = {}
            if result.get('architecture') != 'llama' or result.get('filename') != Path(arguments[1]).name:
                errors.append('Literal synthetic model path/architecture did not survive transport')
        elif expected == 'plan':
            output = Path(arguments[arguments.index('--output')+1])
            if output.exists() or model_sha not in parsed.get('child',{}).get('stdout_tail',''):
                errors.append('Plan-only created output or did not report its verified source hash')
        elif expected == 'verify':
            if parsed.get('result',{}).get('status') != 'offline-gguf-addon-verified':
                errors.append('Runtime asset verification receipt is missing')
    if (directory/'SENTINEL').exists():
        errors.append('Literal metacharacters caused a shell side effect')
    requested = json.dumps(arguments,ensure_ascii=True).encode()
    return {'name':name,'expected':expected,'success':not errors,'errors':errors,
            'argument_count':len(arguments),'argv_utf8_bytes':sum(len(value.encode())+1 for value in arguments),
            'escaped_json_bytes':len(requested),'arguments_sha256':hashlib.sha256(requested).hexdigest(),
            'outer_guard_cleanup_ok':outer.get('cleanup_ok'),**captured}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--kit-dir',type=Path,required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--numeric-python',type=Path,required=True)
    parser.add_argument('--report',type=Path,default=Path('/tmp/pixelforge-stress-integrated.json'))
    parser.add_argument('--timeout',type=float,default=45)
    parser.add_argument('--concurrency',type=int,choices=(1,2),default=2)
    args = parser.parse_args(argv)
    args.artifact,args.model,args.kit_dir = args.artifact.absolute(),args.model.absolute(),args.kit_dir.absolute()
    args.numeric_python = args.numeric_python.absolute()
    if args.report.exists() or args.report.is_symlink() or args.timeout <= 0:
        parser.error('Use a fresh report path and positive deadline')
    if args.kit_dir.is_symlink() or not args.kit_dir.is_dir():
        parser.error('An already verified/extracted real cache directory is required')
    if not args.numeric_python.is_file() or not os.access(args.numeric_python,os.X_OK):
        parser.error('Provide the existing numeric Python; this harness never installs it')
    artifact_pin,model_pin = file_pin(args.artifact),file_pin(args.model)
    if model_pin['bytes'] > MAX_MODEL_BYTES:
        parser.error('This CPU fixture harness requires a synthetic GGUF smaller than 4 MiB')
    source = embedded_worker(args.artifact)
    cache_identity = args.kit_dir.stat().st_dev,args.kit_dir.stat().st_ino
    executable, ostadix_root = _find_ostadix()
    executable = Path(executable).expanduser().absolute()
    ostadix_root = Path(ostadix_root).expanduser().absolute()
    environment = os.environ.copy()
    environment.update(PATH='/usr/bin:/bin:'+environment.get('PATH',''),PYTHONDONTWRITEBYTECODE='1',
                       O_LANG_ROOT=str(ostadix_root),OSTADIX_NPU_O=str(executable),
                       OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    for key in ('OSTADIX_NPU_ARGV','OSTADIX_NPU_ARGS_JSON','OSTADIX_NPU_ARGV_FILE',
                'OSTADIX_NPU_CANCEL_FILE','OSTADIX_NPU_RESULT_FILE','OSTADIX_NPU_LAUNCHER_PID',
                'OSTADIX_NPU_LAUNCHER_START_TICKS'):
        environment.pop(key,None)
    cancel = threading.Event()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='pixelforge-integrated-stress-') as temporary:
        directory = Path(temporary)
        worker = directory/'embedded-worker.py'
        worker.write_text(source)
        worker.chmod(0o600)
        model = directory/"fixture $(touch SENTINEL); 'quoted' 雪.gguf"
        shutil.copyfile(args.model,model)
        if file_pin(model)['sha256'] != model_pin['sha256']:
            raise ValueError('Synthetic fixture copy changed')
        guards = guard_fixtures(worker,directory,environment,args.timeout,cancel)
        jobs = cases(model,directory)
        results = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {pool.submit(evaluate,name,arguments,expected,args,directory,worker,
                                   environment,cancel,model_pin['sha256'],cache_identity):name
                       for name,arguments,expected in jobs}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as error:
                    results.append({'name':futures[future],'success':False,'errors':[type(error).__name__+': '+str(error)]})
        sentinel_absent = not (directory/'SENTINEL').exists()
        fixture_bytes = sum(path.stat().st_size for path in directory.rglob('*') if path.is_file())
        if fixture_bytes > MAX_FIXTURE_BYTES:
            raise ValueError('Stress fixtures exceeded their 8 MiB disk budget')
    stable = file_pin(args.artifact) == artifact_pin and file_pin(args.model) == model_pin
    stable_cache = args.kit_dir.is_dir() and (args.kit_dir.stat().st_dev,args.kit_dir.stat().st_ino) == cache_identity
    result = {'schema_version':1,'success':guards['success'] and all(item['success'] for item in results)
              and stable and stable_cache and sentinel_absent,'created_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'Actual frozen .O read-only actions and exact embedded worker CPU process fixtures',
              'artifact':{'path':str(args.artifact),**artifact_pin},'source_model':model_pin,
              'embedded_worker_sha256':hashlib.sha256(source.encode()).hexdigest(),
              'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'fixture_tests_sha256':hashlib.sha256((HERE/'test_worker.py').read_bytes()).hexdigest(),
              'artifact_invocations':len(results),'artifact_invocations_passed':sum(item['success'] for item in results),
              'embedded_guard_fixtures':guards,'invocations':sorted(results,key=lambda item:item['name']),
              'input_files_unchanged':stable,'existing_cache_same_directory':stable_cache,
              'sentinel_absent':sentinel_absent,'temporary_fixture_bytes':fixture_bytes,
              'bounds':{'artifact_concurrency':args.concurrency,'per_invocation_deadline_seconds':args.timeout,
                        'owned_cleanup_grace_seconds':15,'captured_tail_bytes_per_stream':TAIL_BYTES,
                        'fixture_disk_cap_bytes':MAX_FIXTURE_BYTES},
              'requested_npu_or_vendor_invocations':0,'requested_installer_or_extraction_actions':0,
              'requested_hardware_observations':0,'seconds':time.monotonic()-started}
    private_json(args.report,result)
    print(json.dumps({'success':result['success'],'artifact_invocations':len(results),
                      'artifact_invocations_passed':result['artifact_invocations_passed'],
                      'embedded_guard_tests':guards.get('tests_run'),'seconds':result['seconds'],
                      'report':str(args.report)}))
    return 0 if result['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
