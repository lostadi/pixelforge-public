"""Quiet, same-phone companion notifications through the installed Termux API.

Posting runs as Termux's ordinary UID. Fixed privileged reads obtain Android's
boot classpaths and one notification key to verify delivery; they never list
other notifications or read arbitrary app/process environments.
The fixed tap action opens Termux's main activity. It does not run ``c`` or send
model text to a terminal. Android may retain a user's chosen channel settings.
No notification, microphone, or notification-listener permission is requested.
"""
import os
import re
import shlex
import subprocess
import time
import unicodedata

PREFIX = '/data/data/com.termux/files/usr'
API = PREFIX + '/libexec/termux-api'
NOTIFICATION_ID = 'Ostadix-companion'
CHANNEL_ID = 'ostadix-companion-quiet-v1'
TITLE = 'Ostadix companion'
# The notification belongs to this installation's ordinary Termux account.
try:
    TERMUX_UID = os.stat(PREFIX).st_uid
except OSError:
    TERMUX_UID = os.getuid() if os.getuid() != 0 else None
KEY = '0|com.termux.api|0|' + NOTIFICATION_ID + '|' + str(TERMUX_UID)
TAP_ACTION = (PREFIX + '/bin/am start -a android.intent.action.MAIN '
              '-c android.intent.category.LAUNCHER -n com.termux/.app.TermuxActivity '
              '-f 0x10200000')
TIMEOUT = 5
MAX_TEXT = 360
MAX_OUTPUT = 65536
_ANSI = re.compile(r'\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]')
_CLASSPATH_KEYS = {'BOOTCLASSPATH', 'DEX2OATBOOTCLASSPATH'}
_classpath_cache = None


def _parse_classpaths(output):
    """Parse Android init exports as data, never source them as shell code."""
    result = {}
    if not isinstance(output, str) or len(output) > MAX_OUTPUT:
        raise ValueError('Android runtime metadata exceeds its bound')
    for line in output.splitlines():
        fields = line.split()
        if len(fields) < 2 or fields[1] not in _CLASSPATH_KEYS:
            continue
        if len(fields) != 3 or fields[0] != 'export' or fields[1] in result:
            raise ValueError('Invalid Android runtime metadata')
        paths = fields[2].split(':')
        if not paths or any(not re.fullmatch(r'/(?:apex|system|system_ext|product|vendor)/[A-Za-z0-9/_.+\-]+\.jar', path)
                            or '..' in path.split('/') for path in paths):
            raise ValueError('Invalid Android runtime classpath')
        result[fields[1]] = fields[2]
    if result.keys() != _CLASSPATH_KEYS:
        raise ValueError('Android runtime classpaths are unavailable')
    return result


def _api_environment():
    """Restore Java launch metadata deliberately omitted by the model daemon.

    Termux API's native bridge invokes Termux ``am`` and Android app_process.
    Java requires the boot classpaths even though the NPU process does not.
    The fixed system file contains init runtime metadata, not user/app content.
    A boot restarts this process, so its validated per-process cache is scoped
    to the same boot as Android's classpath file.
    """
    global _classpath_cache
    if _classpath_cache is None:
        response = subprocess.run(['/system/bin/su', '-c',
                                   '/system/bin/cat /data/system/environ/classpath'],
                                  stdin=subprocess.DEVNULL, capture_output=True,
                                  text=True, timeout=TIMEOUT, check=False)
        if response.returncode or response.stderr:
            raise ValueError('Android runtime metadata could not be read')
        _classpath_cache = _parse_classpaths(response.stdout)
    environment = os.environ.copy()
    environment.update(_classpath_cache)
    environment.update(ANDROID_ROOT='/system', ANDROID_DATA='/data',
                       ANDROID_ART_ROOT='/apex/com.android.art',
                       ANDROID_I18N_ROOT='/apex/com.android.i18n',
                       ANDROID_TZDATA_ROOT='/apex/com.android.tzdata')
    return environment


def sanitize_text(text):
    """Keep a short readable Unicode message, without terminal/bidi controls."""
    if not isinstance(text, str) or len(text) > 8192:
        raise ValueError('Notification text must be a short string')
    text = unicodedata.normalize('NFC', _ANSI.sub('', text))
    text = ''.join(' ' if character.isspace() else character
                   for character in text
                   if character.isspace() or not unicodedata.category(character).startswith('C'))
    text = ' '.join(text.split())
    if not text:
        raise ValueError('Notification text is empty')
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT - 1].rstrip() + '…'
    return text


def _run(argv, content=''):
    """Use fixed argument arrays; never let generated content become shell code."""
    try:
        environment = _api_environment() if argv[0] == API else None
        result = subprocess.run(argv, input=content, capture_output=True, text=True,
                                timeout=TIMEOUT, check=False, env=environment)
    except subprocess.TimeoutExpired:
        return {'ok': False, 'error': 'timeout', 'uncertain': True}
    except OSError as error:
        return {'ok': False, 'error': type(error).__name__, 'uncertain': False}
    except ValueError:
        return {'ok': False, 'error': 'Android runtime metadata unavailable', 'uncertain': False}
    if len(result.stdout) > MAX_OUTPUT or len(result.stderr) > MAX_OUTPUT:
        return {'ok': False, 'error': 'oversized tool response', 'uncertain': True}
    # Do not return command output to callers: Android diagnostics may include
    # the content. All public diagnostics are deliberately fixed strings.
    return {'ok': result.returncode == 0, 'code': result.returncode,
            'stdout': result.stdout.strip(), 'stderr': result.stderr.strip(),
            'uncertain': False}


def _base(checkin_id=None):
    result = {'ok': False, 'status': 'unknown', 'notification_id': NOTIFICATION_ID,
              'channel_id': CHANNEL_ID, 'verified': False}
    if checkin_id is not None:
        if not isinstance(checkin_id, str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,96}', checkin_id):
            raise ValueError('Invalid check-in id')
        result['checkin_id'] = checkin_id
    return result


def _verify(text=None, since_ms=None):
    # Fixed read-only command and fixed key. No caller/model argument enters su.
    command = shlex.join(['/system/bin/cmd', 'notification', 'get', KEY])
    answer = _run(['/system/bin/su', '-c', command])
    output = answer.get('stdout', '')
    if 'error: no active notification matching key: ' + KEY in output:
        return {'state': 'absent'}
    if not answer['ok']:
        return {'state': 'unknown', 'reason': answer.get('error', 'notification readback unavailable')}
    if (not output.startswith('NotificationRecord(') or 'pkg=com.termux.api ' not in output
            or 'key=' + KEY not in output or 'channel=' + CHANNEL_ID not in output):
        return {'state': 'unknown', 'reason': 'unrecognized notification readback'}
    when = re.search(r'\bwhen=(\d+)(?:/|\s)', output)
    fresh = since_ms is None or (when is not None and int(when.group(1)) >= since_ms - 1000)
    matches = text is None or ('android.text=String (' + text + ')') in output
    return {'state': 'present', 'matches': matches, 'fresh': fresh,
            'quiet_channel': 'mImportance=2,' in output or 'mImportance=1,' in output}


def _problem(result, response, stage):
    result.update(status='unknown' if response.get('uncertain') else 'failed',
                  reason=stage + ': ' + response.get('error', 'Android API rejected the request'))
    return result


def deliver(text, checkin_id=None, before_post=None):
    """Post one quiet question; return ok only after current-content readback.

    ``status`` is delivered, failed, suppressed, or unknown. A supplied
    ``before_post`` callback must return exactly True after channel creation;
    false or an exception suppresses posting. An unknown response can still
    mean Android posted it; callers must avoid aggressive duplicate retries.
    Channel blocking/permission denial surfaces as failed or unknown rather
    than claiming visibility. A successful post does not prove the user read it.
    """
    result = _base(checkin_id)
    try:
        text = sanitize_text(text)
    except ValueError as error:
        return dict(result, status='failed', reason=str(error))
    if os.getuid() != TERMUX_UID:
        return dict(result, status='failed', reason='Notification posting requires the Termux app UID')
    channel = _run([API, 'NotificationChannel', '--es', 'id', CHANNEL_ID,
                    '--es', 'name', TITLE, '--es', 'priority', 'low'])
    if not channel['ok']:
        return _problem(result, channel, 'quiet channel')
    if not channel.get('stdout', '').startswith('Created channel with id "' + CHANNEL_ID + '"'):
        return dict(result, status='unknown', reason='Quiet channel creation was not acknowledged')
    if before_post is not None:
        try:
            allowed = before_post() is True
        except Exception:
            return dict(result, status='suppressed', reason='Final activity check was unavailable')
        if not allowed:
            return dict(result, status='suppressed', reason='Final activity check did not permit delivery')
    started = int(time.time() * 1000)
    post = _run([API, 'Notification', '--es', 'id', NOTIFICATION_ID,
                 '--es', 'channel', CHANNEL_ID, '--es', 'title', TITLE,
                 '--es', 'priority', 'low', '--ez', 'alert-once', 'true',
                 '--es', 'action', TAP_ACTION], text + '\n')
    if not post['ok']:
        return _problem(result, post, 'notification post')
    if post.get('stdout') or post.get('stderr'):
        return dict(result, status='unknown', reason='Unexpected notification API response')
    result['accepted'] = True
    # Android's asynchronous notification enqueue can take a moment to appear.
    for attempt in range(3):
        evidence = _verify(text, started)
        if evidence['state'] == 'present' and evidence.get('matches') and evidence.get('fresh'):
            return dict(result, ok=True, status='delivered', verified=True,
                        quiet_channel=evidence['quiet_channel'],
                        tap='opens Termux; enter c if its interface is not running')
        if evidence['state'] == 'unknown':
            break
        if attempt < 2:
            time.sleep(0.15)
    return dict(result, status='unknown', reason='Android accepted the request, but current delivery was not verified')


def clear():
    """Remove only the companion's stable notification and verify absence."""
    result = _base()
    if os.getuid() != TERMUX_UID:
        return dict(result, status='failed', reason='Notification removal requires the Termux app UID')
    removed = _run([API, 'NotificationRemove', '--es', 'id', NOTIFICATION_ID])
    if not removed['ok']:
        return _problem(result, removed, 'notification remove')
    if removed.get('stdout') or removed.get('stderr'):
        return dict(result, status='unknown', reason='Unexpected notification API response')
    for attempt in range(3):
        evidence = _verify()
        if evidence['state'] == 'absent':
            return dict(result, ok=True, status='removed', verified=True)
        if evidence['state'] == 'unknown':
            break
        if attempt < 2:
            time.sleep(0.15)
    return dict(result, status='unknown', reason='Notification removal was not verified')
