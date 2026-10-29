"""Pure policy for adjusting only Ostadix's owned NPU host thread priorities.

``decide`` never reads the phone, runs a command, or changes a process. The
controller supplies a fresh telemetry sample and captures each owned thread's
original nice value. A ``set_nice`` decision means ``min(19, baseline + delta)``
for those verified threads only. Delta zero restores their baseline; it never
raises priority above that baseline or changes clocks, voltages, or affinity.

Persist ``next_state`` from observe decisions. For set_nice decisions, commit
it only after every intended priority change succeeds; failed/partial writes
must retain the previous state and be handled by the controller.

Normal model-approved adjustments have a 30-second stable-candidate window and
a minimum 60-second cooldown. Critical heat or unplugged battery <=10% forces
eco(+10) immediately, overriding the cooldown. Missing/stale telemetry prevents
all changes. Measured rollback is part of the original approved trial: provide
``evaluation={"change_at": previous_change_time, "regression": bool}``. Only a
matching, true regression requests restoration, after the cooldown, and never
overrides a currently critical condition. A controller can supply a comparable-workload evaluation; the current monitor
does not yet run that measurement. This policy claims no performance or
battery improvement.
"""
import json
import math


COOLDOWN_SECONDS = 60.0
HYSTERESIS_SECONDS = 30.0
MAX_SAMPLE_AGE_SECONDS = 20.0
OBJECTIVES = frozenset(('balanced', 'performance', 'battery'))


class PolicyError(ValueError):
    pass


def _number(value, low, high):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and low <= value <= high
    except OverflowError:
        return False


def _unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise PolicyError('Duplicate recommendation key')
        out[key] = value
    return out


def validate_recommendation(value):
    """Accept exactly normal/eco plus a short printable explanation; no commands."""
    if isinstance(value, str):
        if len(value) > 4096:
            raise PolicyError('Recommendation is too large')
        try:
            encoded_size = len(value.encode('utf-8'))
        except UnicodeError as error:
            raise PolicyError('Recommendation must be valid Unicode') from error
        if encoded_size > 4096:
            raise PolicyError('Recommendation is too large')
        try:
            value = json.loads(value, object_pairs_hook=_unique_pairs)
        except (ValueError, RecursionError) as error:
            raise PolicyError('Recommendation must be one strict JSON object') from error
    if not isinstance(value, dict) or set(value) != {'action', 'reason'}:
        raise PolicyError('Recommendation must contain only action and reason')
    if value['action'] not in ('normal', 'eco'):
        raise PolicyError('Recommendation action must be normal or eco')
    reason = value['reason']
    if (not isinstance(reason, str) or not 1 <= len(reason.strip()) <= 240
            or any(not char.isprintable() for char in reason)):
        raise PolicyError('Recommendation reason must be 1–240 printable characters')
    return {'action': value['action'], 'reason': reason.strip()}


def new_state():
    """Initial state after the controller captures the engine's baseline priorities."""
    return {'mode': 'normal', 'nice_delta': 0, 'last_change_at': None,
            'candidate_delta': None, 'candidate_since': None, 'trial': None}


def _state(value, now):
    if value is None:
        return new_state()
    if not isinstance(value, dict) or set(value) != set(new_state()):
        raise PolicyError('Invalid optimizer state fields')
    out = dict(value)
    if (type(out['nice_delta']) is not int or out['nice_delta'] not in (0, 5, 10)
            or out['mode'] != ('normal' if out['nice_delta'] == 0 else 'eco')):
        raise PolicyError('Invalid optimizer priority state')
    if out['last_change_at'] is not None and not _number(out['last_change_at'], 0, now):
        raise PolicyError('Invalid optimizer change timestamp')
    if out['candidate_delta'] is not None and (type(out['candidate_delta']) is not int or out['candidate_delta'] not in (0, 5)):
        raise PolicyError('Invalid optimizer candidate')
    if out['candidate_delta'] is None:
        if out['candidate_since'] is not None:
            raise PolicyError('Candidate timestamp requires a candidate')
    elif not _number(out['candidate_since'], 0, now):
        raise PolicyError('Invalid optimizer candidate timestamp')
    trial = out['trial']
    if trial is not None:
        if (not isinstance(trial, dict) or set(trial) != {'change_at', 'from_delta'}
                or not _number(trial['change_at'], 0, now)
                or trial['change_at'] != out['last_change_at']
                or type(trial['from_delta']) is not int or trial['from_delta'] not in (0, 5, 10)):
            raise PolicyError('Invalid optimizer trial')
        out['trial'] = dict(trial)
    return out


def _telemetry(value, now):
    required = {'sampled_at', 'battery_percent', 'battery_temp_c', 'charging',
                'memory_available_mb', 'cpu_busy_pct', 'active', 'queued'}
    if not isinstance(value, dict) or not required <= value.keys():
        return None
    checks = (('sampled_at', max(0, now - MAX_SAMPLE_AGE_SECONDS), now),
              ('battery_percent', 0, 100), ('battery_temp_c', -30, 90),
              ('memory_available_mb', 0, 1048576), ('cpu_busy_pct', 0, 100))
    if any(not _number(value[key], low, high) for key, low, high in checks):
        return None
    if type(value['charging']) is not bool:
        return None
    if any(type(value[key]) is not int or not 0 <= value[key] <= 10000 for key in ('active', 'queued')):
        return None
    severity = value.get('thermal_severity')
    if severity is not None and (type(severity) is not int or not 0 <= severity <= 6):
        return None
    return dict(value)


def _critical(telemetry):
    if telemetry.get('thermal_severity') is not None and telemetry['thermal_severity'] >= 3:
        return 'Android thermal severity is severe or higher'
    if telemetry['battery_temp_c'] >= 45:
        return 'Battery temperature is at least 45°C'
    if telemetry['battery_percent'] <= 10 and not telemetry['charging']:
        return 'Battery is at or below 10% while unplugged'
    return None


def _recovered(telemetry):
    """Separate recovery thresholds avoid toggling around critical boundaries."""
    return (telemetry['battery_temp_c'] <= 40
            and (telemetry.get('thermal_severity') is None or telemetry['thermal_severity'] <= 1)
            and (telemetry['charging'] or telemetry['battery_percent'] >= 15))


def _soft_pressure(telemetry, objective):
    if objective == 'performance':
        return (telemetry['battery_temp_c'] >= 42 or telemetry['memory_available_mb'] < 768
                or telemetry['battery_percent'] <= 15 and not telemetry['charging'])
    pressure = (telemetry['battery_temp_c'] >= 41 or telemetry['memory_available_mb'] < 1536
                or telemetry['cpu_busy_pct'] >= 85
                or telemetry['battery_percent'] <= 25 and not telemetry['charging'])
    if objective == 'battery':
        pressure |= (not telemetry['charging'] and
                     (telemetry['battery_percent'] <= 60 or telemetry['active'] + telemetry['queued'] == 0))
    return pressure


def _observe(state, reason, clear_candidate=False):
    if clear_candidate:
        state['candidate_delta'] = state['candidate_since'] = None
    return {'action': 'observe', 'mode': state['mode'], 'nice_delta': state['nice_delta'],
            'reason': reason, 'next_state': state}


def _change(state, delta, now, reason, trial):
    previous = state['nice_delta']
    state.update(mode='normal' if delta == 0 else 'eco', nice_delta=delta,
                 last_change_at=now, candidate_delta=None, candidate_since=None,
                 trial={'change_at': now, 'from_delta': previous} if trial else None)
    return {'action': 'set_nice', 'mode': state['mode'], 'nice_delta': delta,
            'reason': reason, 'next_state': state}


def decide(telemetry, recommendation=None, state=None, objective='balanced', now=None,
           evaluation=None):
    """Return an inspectable priority plan and next state, with no side effects.

    ``now`` and telemetry ``sampled_at`` use the same monotonic clock. Invalid
    telemetry/model output leads to observation only. Corrupt policy state or
    unsupported objectives raise PolicyError, so the controller cannot mistake
    them for an authorized change.
    """
    if not isinstance(objective, str) or objective not in OBJECTIVES:
        raise PolicyError('Objective must be balanced, performance, or battery')
    if not _number(now, 0, 1e15):
        raise PolicyError('A finite monotonic timestamp is required')
    state = _state(state, now)
    telemetry = _telemetry(telemetry, now)
    if telemetry is None:
        return _observe(state, 'Complete fresh telemetry is required; no priority change', True)
    critical = _critical(telemetry)
    if critical:
        if state['nice_delta'] == 10:
            return _observe(state, critical + '; retain eco priority', True)
        return _change(state, 10, now, critical + '; apply bounded eco priority', False)

    cooldown = state['last_change_at'] is not None and now - state['last_change_at'] < COOLDOWN_SECONDS
    trial = state['trial']
    if evaluation is not None:
        valid = (isinstance(evaluation, dict) and set(evaluation) == {'change_at', 'regression'}
                 and _number(evaluation['change_at'], 0, now)
                 and type(evaluation['regression']) is bool)
        if not valid:
            return _observe(state, 'Invalid measurement evaluation; observe without changes', True)
        if trial is None or evaluation['change_at'] != trial['change_at']:
            return _observe(state, 'Evaluation does not match the current trial; no rollback', True)
        if evaluation['regression']:
            if cooldown:
                return _observe(state, 'Measured regression; wait for the 60-second rollback cooldown', True)
            return _change(state, trial['from_delta'], now,
                           'Measured regression; restore the priority before this trial', False)
        state['trial'] = None

    try:
        approved = validate_recommendation(recommendation)
    except PolicyError:
        return _observe(state, 'No valid local model recommendation; observe without changes', True)
    target = 0 if approved['action'] == 'normal' else 5
    if target == state['nice_delta']:
        return _observe(state, 'Recommendation agrees with current priority', True)
    if state['nice_delta'] == 10 and not _recovered(telemetry):
        return _observe(state, 'Keep critical eco priority until recovery thresholds are met', True)
    pressure = _soft_pressure(telemetry, objective)
    if target == 5 and state['nice_delta'] == 0 and not pressure:
        return _observe(state, 'Telemetry does not justify eco priority for this objective', True)
    if target == 0 and (not _recovered(telemetry) or pressure and objective != 'performance'):
        return _observe(state, 'Conditions do not yet justify restoring baseline priority', True)
    if state['candidate_delta'] != target:
        state['candidate_delta'], state['candidate_since'] = target, now
    if cooldown:
        return _observe(state, 'Wait for the 60-second priority cooldown')
    if now - state['candidate_since'] < HYSTERESIS_SECONDS:
        return _observe(state, 'Wait for 30 seconds of consistent telemetry and recommendation')
    return _change(state, target, now,
                   'Local model approved bounded ' + approved['action'] + ' priority: ' + approved['reason'], True)
