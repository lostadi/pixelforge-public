"""Bounded equivalence for monitoring recommendations, never user answers."""
from optimizer_policy import _telemetry, _critical, _soft_pressure, _recovered, OBJECTIVES

def evidence_key(telemetry, objective, now):
    """Require fresh complete measurements and preserve every policy boundary.

    Each numeric predicate used by the policy is represented separately, so
    crossing one boundary is detected even when another pressure is present.
    Raw telemetry still goes to both the policy and the model.
    """
    if objective not in OBJECTIVES:
        return None
    t = _telemetry(telemetry, now)
    if t is None:
        return None
    return (1, objective, _critical(t), _soft_pressure(t, objective), _recovered(t),
            t['charging'], t.get('thermal_severity'), t['active'], t['queued'],
            tuple(t['battery_percent'] <= x for x in (10, 15, 25, 60)),
            t['battery_percent'] >= 15, t['battery_temp_c'] <= 40,
            tuple(t['battery_temp_c'] >= x for x in (41, 42, 45)),
            tuple(t['memory_available_mb'] < x for x in (768, 1536)),
            t['cpu_busy_pct'] >= 85)
