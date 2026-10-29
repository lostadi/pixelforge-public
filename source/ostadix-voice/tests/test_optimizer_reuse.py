import unittest
from optimizer_reuse import evidence_key

class ReuseTests(unittest.TestCase):
    def setUp(self):
        self.t = dict(sampled_at=100, battery_percent=80, battery_temp_c=32.2,
                      charging=True, thermal_severity=0, memory_available_mb=4200,
                      cpu_busy_pct=21, active=0, queued=0)
    def key(self, **changes):
        return evidence_key(dict(self.t, **changes), 'balanced', 100)
    def test_small_variations_share_key_without_mutating_input(self):
        self.assertEqual(self.key(), self.key(battery_temp_c=32.3, cpu_busy_pct=22))
        self.assertEqual(self.t['battery_temp_c'], 32.2)
    def test_material_changes_do_not_share_key(self):
        for changes in [dict(charging=False), dict(thermal_severity=1), dict(active=1),
                        dict(queued=1), dict(cpu_busy_pct=85), dict(memory_available_mb=1400),
                        dict(battery_percent=60), dict(battery_temp_c=41)]:
            self.assertNotEqual(self.key(), self.key(**changes))
    def test_each_exact_policy_boundary_invalidates(self):
        for field,a,b,extra in [('cpu_busy_pct',84,85,{}),
                                ('battery_temp_c',40,40.1,{}),
                                ('battery_percent',10,11,{'charging':False}),
                                ('battery_percent',25,26,{'charging':False}),
                                ('memory_available_mb',767,768,{}),
                                ('memory_available_mb',1535,1536,{}),
                                ('battery_temp_c',41.9,42,{}),
                                ('battery_temp_c',44.9,45,{})]:
            self.assertNotEqual(self.key(**{field:a},**extra), self.key(**{field:b},**extra))
    def test_missing_stale_and_nonfinite_evidence_is_not_cacheable(self):
        for changes in [dict(sampled_at=0),dict(cpu_busy_pct=None),dict(battery_temp_c=float('nan'))]:
            self.assertIsNone(self.key(**changes))
    def test_objective_is_part_of_key(self):
        self.assertNotEqual(self.key(), evidence_key(self.t,'performance',100))

class MaskedBoundaryTests(unittest.TestCase):
    def test_memory_transition_invalidates_even_while_cpu_already_high(self):
        t=dict(sampled_at=100,battery_percent=80,battery_temp_c=32,charging=True,
               thermal_severity=0,memory_available_mb=2000,cpu_busy_pct=90,active=0,queued=0)
        self.assertNotEqual(evidence_key(t,'balanced',100),
                            evidence_key(dict(t,memory_available_mb=1400),'balanced',100))
