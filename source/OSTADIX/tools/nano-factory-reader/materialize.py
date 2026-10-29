#!/usr/bin/env python3
"""Materialize only manifest-listed factory files, validating decoded size/SHA1.

Raw large files remain links to the read-only factory store. Small files use
the installed TrustyDecrypt API through the owned Java reader. This does not
modify AICore, its databases, the factory mount, or any service policy.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import time

ANDROID_KEYS = {
    "ANDROID_ART_ROOT", "ANDROID_I18N_ROOT", "ANDROID_TZDATA_ROOT",
    "ANDROID_ROOT", "ANDROID_DATA", "BOOTCLASSPATH", "DEX2OATBOOTCLASSPATH",
    "SYSTEMSERVERCLASSPATH",
}


def android_environment(jar):
    env = os.environ.copy()
    found = False
    for process in Path('/proc').iterdir():
        if not process.name.isdecimal():
            continue
        try:
            name = (process / 'cmdline').read_bytes().split(b'\0')[0]
            if name not in (b'zygote', b'zygote64', b'zygote_next'):
                continue
            for item in (process / 'environ').read_bytes().split(b'\0'):
                key, separator, value = item.partition(b'=')
                if separator and key.decode(errors='ignore') in ANDROID_KEYS:
                    env[key.decode()] = value.decode()
            found = True
            break
        except OSError:
            continue
    if not found or not env.get('BOOTCLASSPATH'):
        raise RuntimeError('No usable live Android runtime environment')
    env.pop('LD_PRELOAD', None)
    env['CLASSPATH'] = str(jar.resolve())
    return env


def sha1(path):
    digest = hashlib.sha1()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('manifest_report', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--journal', type=Path, required=True)
    args = parser.parse_args()
    reader_root = Path(__file__).resolve().parent
    env = android_environment(reader_root / 'build/factory-reader.jar')
    groups = json.loads(args.manifest_report.read_text())['preload_file_groups']
    if len(groups) != 1 or groups[0]['group_name'] != ['feature_234']:
        raise ValueError('Expected the recovered factory feature_234 group')
    args.destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    destination = args.destination.resolve()
    factory = Path('/data/vendor/intelligence').resolve()
    missing = []
    verified = 0
    with args.journal.open('a') as journal:
        def record(value):
            line = json.dumps(value, sort_keys=True)
            journal.write(line + '\n')
            journal.flush()
            print(line, flush=True)

        for item in groups[0]['files']:
            relative = PurePosixPath(item['relative_filename'])
            if relative.is_absolute() or '..' in relative.parts or not relative.parts:
                raise ValueError('Unsafe relative manifest filename')
            basename = item['mapped_factory_filename']
            if Path(basename).name != basename:
                raise ValueError('Expected a factory basename')
            source = factory / basename
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if target.parent.resolve().is_relative_to(destination) is False:
                raise ValueError('Destination escapes owned staging directory')
            expected_size = item['byte_size'][0]
            expected_sha1 = item['checksum'][0]
            if not source.is_file():
                missing.append(str(relative))
                record({'event': 'missing_factory_file', 'file': str(relative),
                        'expected_bytes': expected_size, 'expected_sha1': expected_sha1})
                continue
            if source.resolve().parent != factory:
                raise ValueError('Factory entry resolves outside expected store')
            started = time.monotonic()
            mode = 'existing_verified'
            if not target.exists() and not target.is_symlink():
                if source.stat().st_size >= 104857600:
                    target.symlink_to(source)
                    mode = 'read_only_factory_link'
                else:
                    command = ['/system/bin/app_process', '/system/bin',
                               'org.ostadix.nano.FactoryReader', basename, str(target)]
                    result = subprocess.run(command, env=env, capture_output=True,
                                            text=True, timeout=60)
                    record({'event': 'decrypt_process', 'file': str(relative),
                            'exit_code': result.returncode, 'stdout': result.stdout,
                            'stderr': result.stderr})
                    if result.returncode != 0 or not target.is_file():
                        raise RuntimeError('Factory decryption failed; no automatic retry')
                    mode = 'stock_trusty_decrypt'
            actual_size = target.stat().st_size
            actual_sha1 = sha1(target)
            if actual_size != expected_size or actual_sha1 != expected_sha1:
                record({'event': 'integrity_failure', 'file': str(relative),
                        'actual_bytes': actual_size, 'actual_sha1': actual_sha1,
                        'expected_bytes': expected_size, 'expected_sha1': expected_sha1})
                raise RuntimeError('Materialized file does not match recovered manifest')
            verified += 1
            record({'event': 'file_verified', 'file': str(relative), 'mode': mode,
                    'bytes': actual_size, 'sha1': actual_sha1,
                    'elapsed_ms': int((time.monotonic() - started) * 1000)})
        record({'event': 'materialization_complete', 'verified_files': verified,
                'missing_files': missing, 'complete_manifest': not missing})


if __name__ == '__main__':
    main()
