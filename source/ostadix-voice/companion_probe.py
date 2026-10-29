"""Fixed, read-only Android activity probe for the local companion.

Only power/keyguard state, coarse physical-activity timing, and the DND setting
leave this helper. No screen content or input events are read; app names and
other power-dump details are neither retained nor returned. Notification
contents are never read. Unknown and inconsistent state fails closed.

Android uptimeMillis uses CLOCK_MONOTONIC, not /proc/uptime (which includes
suspend). WakefulnessSessionObserver durations describe changes of event type,
not every touch; current BUTTON/TOUCH state therefore uses the power group's
last activity time. A preceding physical state is accepted only when its older
transition timestamp is itself recent. Attention/other events cannot by
themselves make the device active.
"""
import argparse
import json
import math
import re
import subprocess
import time

MAX_ACTIVITY_AGE_SECONDS = 120
MAX_SAMPLE_SECONDS = 5
MAX_DUMP_BYTES = 1024 * 1024
PHYSICAL_EVENTS = {1: 'button', 2: 'touch'}


def _one(pattern, text):
    matches = re.findall(pattern, text, re.MULTILINE)
    return matches[0] if len(matches) == 1 else None


def _indented_section(text, header):
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if re.fullmatch(header, line.strip())]
    if len(starts) != 1:
        return ''
    start = starts[0]
    indent = len(lines[start]) - len(lines[start].lstrip())
    result = []
    for line in lines[start + 1:]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        result.append(line)
    return '\n'.join(result)


def _default_power_group(text):
    sections = text.split('Power Group User Activity:')
    if len(sections) != 2:
        return ''
    section = re.split(r'\n\s*\n', sections[1].lstrip('\n'), maxsplit=1)[0]
    groups = re.split(r'^\s*groupId:\s*(\d+)\s*$', section, flags=re.MULTILINE)
    selected = [groups[i + 1] for i in range(1, len(groups), 2) if groups[i] == '0']
    return selected[0] if len(selected) == 1 else ''


def _boolean(value):
    return {'true': True, 'false': False}.get(value)


def _age(milliseconds, uptime):
    if milliseconds is None:
        return None
    age = uptime - int(milliseconds) / 1000
    return age if math.isfinite(age) and 0 <= age <= uptime else None


def parse_activity(power, policy, uptime):
    """Parse only the default display; `uptime` must be current monotonic time."""
    result = {'awake': None, 'unlocked': None,
              'last_user_activity_age_seconds': None,
              'system_activity_age_seconds': None,
              'activity_source': None, 'activity_event': None,
              'active': False, 'reason': 'activity_unknown'}
    if isinstance(uptime, bool) or not isinstance(uptime, (int, float)) or not math.isfinite(uptime) or uptime < 0:
        result['reason'] = 'clock_unknown'
        return result
    if not isinstance(power, str) or not isinstance(policy, str):
        return result
    group = _default_power_group(power)
    observer = _indented_section(power, r'Wakefulness Session Power Group powerGroupId: 0')
    keyguard = _indented_section(policy, r'KeyguardServiceDelegate')
    wakefulness = _one(r'^\s*mWakefulness=([A-Za-z]+)\s*$', power)
    group_awake = _one(r'^\s*mWakefulness=(\d+)\s*$', group)
    observed_awake = _one(r'^\s*current wakefulness:\s*(\d+)\s*$', observer)
    observed_interactive = _boolean(_one(r'^\s*mIsInteractive:\s*(true|false)\s*$', observer))
    screen = _one(r'^\s*screenState=(\w+)\s*$', keyguard)
    interactive = _one(r'^\s*interactiveState=(\w+)\s*$', keyguard)
    flags = [None if wakefulness is None else wakefulness == 'Awake',
             None if group_awake is None else group_awake == '1',
             None if observed_awake is None else observed_awake == '1',
             observed_interactive,
             None if screen is None else screen == 'SCREEN_STATE_ON',
             None if interactive is None else interactive == 'INTERACTIVE_STATE_AWAKE']
    result['awake'] = False if False in flags else True if all(x is True for x in flags) else None
    showing = _boolean(_one(r'^\s*showing=(true|false)\s*$', keyguard))
    restricted = _boolean(_one(r'^\s*inputRestricted=(true|false)\s*$', keyguard))
    result['unlocked'] = False if True in (showing, restricted) else True if showing is False and restricted is False else None

    times = [_age(_one(r'^\s*lastUserActivityTime=(\d+)(?:\s+\([^\n]*\))?\s*$', group), uptime),
             _age(_one(r'^\s*lastUserActivityTimeNoChangeLights=(\d+)(?:\s+\([^\n]*\))?\s*$', group), uptime)]
    ages = [age for age in times if age is not None]
    if ages:
        result['system_activity_age_seconds'] = min(ages)
    event = _one(r'^\s*current user activity event:\s*(\d+)\s*$', observer)
    previous = _one(r'^\s*previous user activity event:\s*(\d+)\s*$', observer)
    if event is not None and int(event) in PHYSICAL_EVENTS and ages:
        result.update(last_user_activity_age_seconds=min(ages), activity_event=int(event),
                      activity_source='current_' + PHYSICAL_EVENTS[int(event)])
    elif event is not None and previous is not None and int(previous) in PHYSICAL_EVENTS:
        duration = _one(r'^\s*previous user activity duration:\s*(\d+)\s*$', observer)
        if duration is not None and 0 <= int(duration) / 1000 <= uptime:
            result.update(last_user_activity_age_seconds=int(duration) / 1000,
                          activity_event=int(previous),
                          activity_source='previous_' + PHYSICAL_EVENTS[int(previous)])

    age = result['last_user_activity_age_seconds']
    if result['awake'] is not True:
        result['reason'] = 'screen_inactive' if result['awake'] is False else 'screen_unknown'
    elif result['unlocked'] is not True:
        result['reason'] = 'locked' if result['unlocked'] is False else 'lock_unknown'
    elif age is None:
        result['reason'] = 'physical_activity_unknown'
    elif age > MAX_ACTIVITY_AGE_SECONDS:
        result['reason'] = 'inactive_timeout'
    else:
        result.update(active=True, reason='recent_physical_activity')
    return result


def parse_dnd(text):
    value = text.strip() if isinstance(text, str) else ''
    return False if value == '0' else True if value in ('1', '2', '3') else None


def _read_command(argv):
    process = subprocess.run(argv, stdin=subprocess.DEVNULL, capture_output=True,
                             timeout=2, text=True)
    if process.returncode or len(process.stdout.encode('utf-8')) > MAX_DUMP_BYTES:
        raise ValueError('Activity telemetry unavailable')
    return process.stdout


def sample():
    started = time.monotonic()
    errors = {}

    def attempt(name, argv):
        try:
            return _read_command(argv)
        except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as error:
            errors[name] = type(error).__name__
            return ''

    power = attempt('power', ['/system/bin/dumpsys', 'power'])
    captured = time.monotonic()
    policy = attempt('keyguard', ['/system/bin/dumpsys', 'window', 'policy'])
    result = parse_activity(power, policy, captured)
    result.update(sampled_at=started, wall_time=time.time(),
                  dnd=parse_dnd(attempt('dnd', ['/system/bin/settings', 'get', 'global', 'zen_mode'])),
                  errors=errors)
    if time.monotonic() - started > MAX_SAMPLE_SECONDS:
        result.update(active=False, reason='stale_snapshot')
    return result


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(sample(), allow_nan=False))
