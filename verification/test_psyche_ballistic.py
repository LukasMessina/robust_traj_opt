"""Physical and transcription regressions for the compulsory Mars coast."""

import unittest
import casadi as ca
import numpy as np
from deterministic_trajectory_optimization.psyche import (
    Mission,
    Settings,
    Transcription,
)
from integrator.rk import get_arc_flow


class BallisticMars(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mission = Mission(
            Settings(initial_intervals_per_phase=(4, 80, 4), print_level=0)
        )

    def test_mars_rollout_has_zero_propulsion_and_constant_mass(self):
        seed = self.mission.seed
        self.assertTrue(np.all(seed["phase2_thrust_n"] == 0.0))
        self.assertTrue(np.all(seed["phase2_sigma_n"] == 0.0))
        self.assertEqual(np.ptp(seed["phase2_states_si"][:, 6]), 0.0)

    def test_mars_flow_ignores_controls_and_conserves_mass(self):
        m = self.mission
        y = np.r_[m.xm, 2400.0] / m.scales[1].state
        flow = get_arc_flow(m, 1, 20)
        a = np.asarray(flow(y, np.zeros(3), 0.0, 0.01)[0]).ravel()
        b = np.asarray(flow(y, np.ones(3), 1.0, 0.01)[0]).ravel()
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a[6], y[6])

    def test_mars_has_no_control_variables_or_direction_constraints(self):
        m = self.mission
        p = Transcription(m, m.initial_meshes(), method="hs")
        _, u, throttle = p.variables[1]
        self.assertEqual(len(ca.symvar(ca.MX(u))), 0)
        self.assertEqual(len(ca.symvar(ca.MX(throttle))), 0)
        self.assertEqual(sum(b - a for a, b in p.group_ranges["unit_direction"]), 8)


if __name__ == "__main__":
    unittest.main()
