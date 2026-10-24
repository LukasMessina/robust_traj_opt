"""Physical mass flow, coast derivatives and warm-start preservation."""

import unittest
from copy import deepcopy

import casadi as ca
import numpy as np

from deterministic_optimization.psyche import (
    Mission,
    Settings,
    Transcription,
    throttle_direction_seed,
)
from deterministic_optimization.flows import get_arc_flow
from deterministic_optimization.derivatives import seed_solution
from deterministic_optimization.derivatives import control_derivative_checks
from deterministic_optimization.psyche_driver import verify_hs_handoff


class ThrottleDirectionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mission = Mission(Settings(mass_flow_model="throttle_direction"))
        cls.source = seed_solution(cls.mission)
        cls.source.update(method="shooting", shooting_substeps=20)

    def test_zero_and_tiny_thrust_factorisation_without_floor(self):
        thrust = np.array([[0.0, 0.0, 0.0], [1e-25, -2e-25, 3e-25], [0.01, -0.02, 0.03]])
        alpha, direction = throttle_direction_seed(thrust, 0.2793)
        np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1.0, atol=3e-16)
        np.testing.assert_allclose(
            0.2793 * alpha[:, None] * direction, thrust, atol=0.0, rtol=4e-16
        )
        self.assertEqual(alpha[0], 0.0)
        self.assertGreater(alpha[1], 0.0)

    def test_exact_mass_loss_and_zero_throttle_direction_independence(self):
        m = self.mission
        for i, scale in enumerate(m.scales):
            state = m.seed[f"phase{i+1}_states_si"][0] / scale.state
            seconds = 100.0
            flow = get_arc_flow(m, i, 20)
            for alpha in (0.0, 0.4, 1.0):
                d = np.array([0.6, 0.0, 0.8])
                end, samples = flow(state, alpha * d, alpha, seconds / scale.time)
                mass = float(end[6]) * scale.mass
                expected = state[6] * scale.mass - m.settings.thrust_max_n * alpha * seconds / (
                    m.settings.isp_s * 9.80665
                )
                self.assertAlmostEqual(mass, expected, delta=2e-11)
                self.assertEqual(samples.shape[1], 19)
            z = ca.MX.sym("z", 4)
            end, _ = flow(state, z[:3] * z[3], z[3], seconds / scale.time)
            jac = ca.Function("coast_derivative", [z], [ca.jacobian(end, z)])
            value = np.asarray(jac([0.6, 0.0, 0.8, 0.0]))
            np.testing.assert_array_equal(value[:, :3], np.zeros((7, 3)))
            self.assertTrue(np.all(np.isfinite(value)))

    def test_control_and_duration_first_second_derivatives(self):
        checks = control_derivative_checks(self.mission, self.source, 20)
        self.assertLess(checks["maximum_jacobian_error"], 2e-6)
        self.assertLess(checks["maximum_hessian_vector_error"], 2e-5)

    def test_fresh_hs_rollout_and_physical_energy_objective(self):
        m = self.mission
        problems = [
            Transcription(m, m.initial_meshes(), "hs", objective=objective, barrier="adaptive")
            for objective in ("fuel", "energy")
        ]
        for name in ("x", "g", "lbg", "ubg"):
            values = [
                np.asarray(p.opti.debug.value(getattr(p.opti, name), p.opti.initial()))
                for p in problems
            ]
            np.testing.assert_array_equal(*values)
        for p in problems:
            for i, (state, thrust, alpha, _) in enumerate(p.variables):
                value = lambda expr: np.asarray(p.opti.debug.value(expr, p.opti.initial()))
                np.testing.assert_allclose(
                    value(state).T,
                    m.seed[f"phase{i+1}_states_si"] / m.scales[i].state,
                    atol=32 * np.finfo(float).eps,
                    rtol=32 * np.finfo(float).eps,
                )
                np.testing.assert_allclose(
                    value(thrust).T * m.settings.thrust_max_n,
                    m.seed[f"phase{i+1}_thrust_n"],
                    rtol=3e-15,
                )
                np.testing.assert_allclose(
                    value(alpha).ravel() * m.settings.thrust_max_n,
                    m.seed[f"phase{i+1}_sigma_n"],
                    rtol=3e-15,
                )
        p = problems[1]
        # Check the actual vector norm, even away from the unit-vector
        # constraint, so a sigma-squared surrogate would fail this check.
        p.opti.set_initial(p.direction_variables[0][:, 0], [2.0, 0.0, 0.0])
        p.opti.set_initial(p.variables[0][2][0, 0], 0.3)
        epochs = np.asarray(p.opti.debug.value(ca.vertcat(*p.epochs), p.opti.initial())).ravel()
        integral = 0.0
        for i, (mesh, (_, thrust, _, _)) in enumerate(zip(p.meshes, p.variables)):
            values = np.asarray(p.opti.debug.value(thrust, p.opti.initial())).T
            integral += (
                0.5 * (epochs[i + 1] - epochs[i]) * np.dot(np.diff(mesh), np.sum(values**2, axis=1))
            )
        self.assertAlmostEqual(
            float(p.opti.debug.value(p.opti.f, p.opti.initial())),
            integral / (epochs[-1] - epochs[0]),
            delta=1e-16,
        )

    def test_hs_handoff_preserves_raw_throttle_and_coast_direction(self):
        m = self.mission
        source = deepcopy(self.source)
        source.update(
            method="hs",
            objective="fuel",
            solver_success=True,
            solver_status="fixture",
            mass_flow_model="throttle_direction",
        )
        for phase in source["phases"]:
            alpha, direction = throttle_direction_seed(phase["thrust_n"], m.settings.thrust_max_n)
            alpha[0], direction[0] = 0.0, [0.0, 1.0, 0.0]
            direction[1] *= 1.0 + 1e-12
            phase.update(
                throttle=alpha,
                direction=direction,
                thrust_n=m.settings.thrust_max_n * alpha[:, None] * direction,
                sigma_n=m.settings.thrust_max_n * alpha,
            )
        problem = Transcription(m, m.initial_meshes(), "shooting", source, "fuel", "adaptive", 20)
        check = verify_hs_handoff(problem, source)
        self.assertTrue(check["verified"])
        self.assertTrue(all(p["throttle_and_direction_identical"] for p in check["phases"]))


if __name__ == "__main__":
    unittest.main()
