#!/usr/bin/env python3
"""Build the small Android helper using the existing local SDK; no downloads."""
from pathlib import Path
import os
import json
import time
import shutil
import subprocess
import sys
import zipfile
import re

ROOT = Path(__file__).resolve().parent
PREFIX = Path(os.environ.get('PREFIX', '/data/data/com.termux/files/usr'))
SDK_HOME = Path(os.environ.get('ANDROID_HOME', os.environ.get('ANDROID_SDK_ROOT', str(Path.home() / 'android-sdk'))))
_platforms = sorted(SDK_HOME.glob('platforms/android-*/android.jar'),
                    key=lambda path: tuple(map(int, re.findall(r'\d+', path.parent.name))))
SDK = Path(os.environ['ANDROID_JAR']) if 'ANDROID_JAR' in os.environ else (_platforms[-1] if _platforms else None)
BUILD = ROOT / 'build'

def run(*args):
    subprocess.run([str(x) for x in args], check=True)

def tool(name):
    installed = PREFIX / 'bin' / name
    result = str(installed) if installed.is_file() else shutil.which(name)
    if not result:
        raise SystemExit('Missing Android build tool: ' + name)
    return result

def main():
    if SDK is None or not SDK.is_file():
        raise SystemExit('Set ANDROID_JAR to a local Android SDK android.jar, or configure ANDROID_HOME. No SDK is downloaded automatically.')
    BUILD.mkdir(exist_ok=True)
    classes = BUILD / 'classes'
    classes.mkdir(exist_ok=True)
    run(tool('javac'), '-source', '8', '-target', '8', '-Xlint:-options', '-classpath', SDK, '-d', classes, *sorted(ROOT.glob('src/**/*.java')))
    run(tool('d8'), '--lib', SDK, '--min-api', '33', '--output', BUILD, *sorted(classes.glob('**/*.class')))
    unsigned = BUILD/'unsigned.apk'
    run(tool('aapt2'), 'link', '-I', SDK, '--manifest', ROOT/'AndroidManifest.xml', '-o', unsigned)
    with zipfile.ZipFile(unsigned, 'a') as archive:
        archive.write(BUILD/'classes.dex', 'classes.dex', compress_type=zipfile.ZIP_DEFLATED)
    key = ROOT/'local-signing.p12'
    if not key.exists():
        run(tool('keytool'), '-genkeypair', '-keystore', key, '-storepass', 'ostadix-local', '-keypass', 'ostadix-local', '-alias', 'ostadix', '-dname', 'CN=Ostadix local voice', '-keyalg', 'RSA', '-keysize', '2048', '-validity', '3650')
        key.chmod(0o600)
    apk=BUILD/'ostadix-voice.apk'
    run(tool('apksigner'), 'sign', '--ks', key, '--ks-pass', 'pass:ostadix-local', '--out', apk, unsigned)
    run(tool('apksigner'), 'verify', apk)
    print(apk)
    if '--install' in sys.argv:
        if os.getuid()!=0:
            raise SystemExit('Build succeeded. Install with the Android package installer or root pm install.')
        temporary=Path('/data/local/tmp/ostadix-voice-bridge.apk')
        shutil.copyfile(apk, temporary)
        temporary.chmod(0o644)
        try:
            run('/system/bin/pm', 'install', '-r', temporary)
            run('/system/bin/pm', 'grant', 'org.ostadix.voicebridge', 'android.permission.RECORD_AUDIO')
            run('/system/bin/pm', 'grant', 'org.ostadix.voicebridge', 'android.permission.POST_NOTIFICATIONS')
            (BUILD/'installed.json').write_text(json.dumps({'package':'org.ostadix.voicebridge','version':1,'installed_at':time.time()}))
        finally:
            temporary.unlink(missing_ok=True)
if __name__=='__main__': main()
