"""Physical scaling, polynomial order and moving-event derivative checks."""

import unittest
from unittest.mock import patch
from pathlib import Path
import json
import tempfile

import casadi as ca
import numpy as np

from deterministic_optimization.rk import rk7
from deterministic_optimization.psyche import Mission, Scale, Settings, Transcription
from deterministic_optimization.psyche_validation import relative_interval_error, interval_defects
from deterministic_optimization import psyche_validation as ocp_validation


class DiscretisationChecks(unittest.TestCase):
    def test_relative_defect_uses_endpoint_magnitudes_and_is_unit_invariant(self):
        initial = np.array([1e8, 0, 0, 2e3, 0, 0, 2000.0])
        final = np.array([2e8, 0, 0, 1e3, 0, 0, 1900.0])
        errors = np.array([1.0, 1e-6, 1e-5])
        scale = Scale(1e8, 4e14, 2747.0)
        # Relative errors: 5e-9 in r, 5e-10 in v, 5e-9 in mass.
        actual = relative_interval_error(initial, final, errors, scale)
        self.assertAlmostEqual(actual, 5e-9, places=20)
        nd_errors = errors / np.array([scale.length, scale.velocity, scale.mass])
        converted = relative_interval_error(
            initial / scale.state, final / scale.state, nd_errors, Scale(1.0, 1.0, 1.0)
        )
        self.assertAlmostEqual(actual, converted, places=20)

    def test_rk7_has_high_order_on_kepler_orbit(self):
        def rhs(x):
            return np.r_[x[2:], -x[:2] / np.linalg.norm(x[:2]) ** 3]

        exact = np.array([np.cos(1.0), np.sin(1.0), -np.sin(1.0), np.cos(1.0)])
        errors = []
        for n in (4, 8):
            x = np.array([1.0, 0, 0, 1.0])
            for _ in range(n):
                x = rk7(rhs, x, 1.0 / n)
            errors.append(np.linalg.norm(x - exact))
        self.assertGreater(errors[0] / errors[1], 80.0)
        self.assertLess(errors[1], 1e-8)


class MissionTranscriptionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mission = Mission(Settings())

    def test_small_thrust_seed_has_two_rk7_steps_and_preserves_anchor_and_mass_links(self):
        m = self.mission
        self.assertEqual([len(t) - 1 for t in m.initial_meshes()], [60, 50, 90, 50])
        thrust = np.full(3, 1e-4)
        sigma = np.linalg.norm(thrust)
        h = m.seed["phase1_elapsed_s"][1] / 2.0
        rhs = lambda y: m.rhs_si(0, y, thrust, sigma)
        first_endpoint = rk7(rhs, rk7(rhs, m.x0.copy(), h), h)
        np.testing.assert_allclose(m.seed["phase1_states_si"][1], first_endpoint, rtol=1e-14)
        np.testing.assert_array_equal(m.seed["phase3_states_si"][0, :6], m.xm)
        for phase in range(1, 5):
            intervals = (60, 50, 90, 50)[phase - 1]
            self.assertEqual(m.seed[f"phase{phase}_states_si"].shape, (intervals + 1, 7))
            self.assertEqual(m.seed[f"phase{phase}_thrust_n"].shape, (intervals, 3))
            initial, final = m.seed[f"phase{phase}_states_si"][[0, -1]]
            duration = m.seed[f"phase{phase}_elapsed_s"][-1]
            self.assertAlmostEqual(
                initial[6] - final[6], sigma * duration / (9.80665 * m.settings.isp_s), places=9
            )
            if phase < 4:
                self.assertEqual(final[6], m.seed[f"phase{phase+1}_states_si"][0, 6])
        for phase, body in ((1, 399), (3, 499)):
            end = m.seed[f"phase{phase}_states_si"][-1, :6]
            epoch = m.seed[f"phase{phase}_epochs_et_tdb_s"][-1]
            np.testing.assert_allclose(
                m.seed[f"phase{phase+1}_states_si"][0, :6], end + m.cache.state(body, epoch)
            )

    def test_interval_defects_use_64_substeps_without_doubling(self):
        m = self.mission
        seed = {
            "phases": [
                {
                    "mesh": m.seed["phase1_tau"],
                    "duration_s": m.seed["phase1_elapsed_s"][-1],
                    "states_si": m.seed["phase1_states_si"],
                    "thrust_n": m.seed["phase1_thrust_n"],
                    "sigma_n": m.seed["phase1_sigma_n"],
                }
            ]
        }
        with patch.object(ocp_validation, "rk7", wraps=rk7) as integrate:
            defects = interval_defects(m, seed)
        self.assertEqual(integrate.call_count, 60 * 64)
        self.assertEqual(defects[0]["rk7_substeps_per_interval"], 64)
        self.assertNotIn("rk7_estimator_resolved", defects[0])

    def test_scaled_rhs_matches_si_and_rocket_mass_flow(self):
        m = self.mission
        u = np.array([0.1, -0.2, 0.3])
        sigma = np.linalg.norm(u)
        for i, scale in enumerate(m.scales):
            state = m.seed_splines[i](0.41)
            computed = (
                np.array(m.rhs(i, state / scale.state, u, sigma)).ravel() * scale.state / scale.time
            )
            expected = m.rhs_si(
                i, state, u * m.settings.thrust_max_n, sigma * m.settings.thrust_max_n
            )
            np.testing.assert_allclose(computed, expected, atol=1e-10, rtol=1e-12)
            self.assertAlmostEqual(
                computed[6],
                -sigma * m.settings.thrust_max_n / (9.80665 * m.settings.isp_s),
                places=15,
            )

    def test_free_epoch_derivatives_and_unmodified_seed(self):
        m = self.mission
        problem = Transcription(m, m.initial_meshes(), method="shooting")
        self.assertEqual(problem.ipopt_options["max_iter"], 5000)
        self.assertNotIn("max_wall_time", problem.ipopt_options)
        self.assertNotIn("max_cpu_time", problem.ipopt_options)
        op = problem.opti
        initial = np.asarray(op.debug.value(op.x, op.initial())).ravel()
        f = ca.Function("constraints", [op.x], [op.g])
        j = ca.Function("exit_jacobian", [op.x], [ca.jacobian(op.g, op.x)[:, :2]])
        analytic = np.array(j(initial))
        for i in range(2):
            step = np.zeros_like(initial)
            step[i] = 1e-5
            numerical = np.array((f(initial + step) - f(initial - step)) / (2e-5)).ravel()
            error = abs(numerical - analytic[:, i]) / (1.0 + abs(analytic[:, i]))
            self.assertLess(error.max(), 2e-6)
        endpoint = op.debug.value(problem.variables[3][0][:, -1], op.initial()) * m.scales[3].state
        np.testing.assert_allclose(endpoint, m.seed["phase4_states_si"][-1], rtol=1e-14, atol=1e-5)
        self.assertGreater(np.linalg.norm(endpoint[:3] - m.xf[:3]), 1e11)
        self.assertEqual(m.tf, 933552000.0)

    def test_surface_only_geometry_and_zero_margin_exit_direction(self):
        m = self.mission
        for method in ("hs", "shooting"):
            with self.subTest(method=method):
                problem = Transcription(m, m.initial_meshes(), method=method)
                self.assertNotIn("planet_inside_soi", problem.group_ranges)
                self.assertNotIn("solar_outside_soi", problem.group_ranges)
                self.assertIn("surface_exclusion", problem.group_ranges)
                self.assertEqual(len(problem.group_ranges["exit_surface"]), 2)
                op = problem.opti
                scale = m.scales[0]
                terminal = problem.variables[0][0][:, -1]
                radius = m.soi[399] / scale.length
                row = problem.group_ranges["exit_direction"][0][0]
                expression = op.g[row] - op.lbg[row]
                # A positive speed BELOW the previous 1 m/s margin must pass;
                # zero is the exact closed boundary, while inward fails.
                for speed, sign in ((0.2, 1), (0.0, 0), (-0.2, -1)):
                    op.set_initial(terminal, [radius, 0, 0, speed / scale.velocity, 0, 0, 1.0])
                    value = float(op.debug.value(expression, op.initial()))
                    self.assertEqual(np.sign(value), sign)


if __name__ == "__main__":
    unittest.main()
