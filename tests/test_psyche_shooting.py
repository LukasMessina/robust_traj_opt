"""Shooting propagation, derivatives, preserved boundaries and rollout seeds."""

import unittest
import numpy as np

from deterministic_optimization.psyche import Mission, Settings, Transcription
from deterministic_optimization.rk import rk7
from deterministic_optimization.flows import get_arc_flow
from deterministic_optimization.derivatives import derivative_checks, exit_time_derivative_checks
from deterministic_optimization.derivatives import seed_solution
from deterministic_optimization.psyche_driver import verify_hs_handoff


class ShootingChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mission = Mission(Settings())
        cls.problem = Transcription(
            cls.mission, cls.mission.initial_meshes(), "shooting", objective="energy"
        )

    def test_direct_rollout_initialisation_and_dimensions(self):
        m, p = self.mission, self.problem
        self.assertEqual(p.opti.nx, 2780)
        self.assertEqual(p.opti.ng, 12308)
        for i, (x, u, sigma, stages) in enumerate(p.variables):
            # elapsed_s / duration can move a spline knot by one floating-point
            # increment. Compare in phase units to accommodate that roundoff.
            np.testing.assert_allclose(
                p.opti.debug.value(x, p.opti.initial()).T,
                m.seed[f"phase{i+1}_states_si"] / m.scales[i].state,
                rtol=32 * np.finfo(float).eps,
                atol=32 * np.finfo(float).eps,
            )
            np.testing.assert_allclose(
                p.opti.debug.value(u, p.opti.initial()).T * m.settings.thrust_max_n,
                m.seed[f"phase{i+1}_thrust_n"],
                rtol=1e-14,
            )
            self.assertEqual(stages, [])
        self.assertNotIn("planet_inside_soi", p.group_ranges)
        self.assertNotIn("solar_outside_soi", p.group_ranges)
        self.assertEqual(len(p.group_ranges["surface_exclusion"]), 20 * 250 + 4)

    def test_flow_first_and_second_derivatives(self):
        checks = derivative_checks(self.mission, 20)
        self.assertLess(checks["maximum_jacobian_error"], 2e-6)
        self.assertLess(checks["maximum_hessian_vector_error"], 2e-5)

    def test_fuel_objective_preserves_constraints_and_rollout(self):
        m, reference = self.mission, self.problem
        fuel = Transcription(
            m, m.initial_meshes(), "shooting", objective="fuel", shooting_substeps=20
        )
        self.assertEqual(fuel.ipopt_options, reference.ipopt_options)
        self.assertEqual(fuel.group_ranges, reference.group_ranges)
        for name in ("x", "g", "lbg", "ubg"):
            values = [
                np.asarray(p.opti.debug.value(getattr(p.opti, name), p.opti.initial()))
                for p in (fuel, reference)
            ]
            np.testing.assert_array_equal(*values)
        # Changing final mass by 100 kg must reduce the reported fuel objective
        # by 100 kg; a constant feasibility objective would fail this check.
        mass = fuel.variables[-1][0][6, -1]
        for final_mass in (2037.0, 2137.0):
            fuel.opti.set_initial(mass, final_mass / m.settings.wet_mass_kg)
            value = float(fuel.opti.debug.value(fuel.opti.f, fuel.opti.initial()))
            self.assertAlmostEqual(
                value * m.settings.wet_mass_kg, m.settings.wet_mass_kg - final_mass, places=10
            )

    def test_exactly_twenty_uniform_steps_without_extra_splits(self):
        m, scale = self.mission, self.mission.scales[0]
        dt = m.seed["phase1_elapsed_s"][1]
        rhs = lambda x: m.rhs_si(0, x, m.seed_thrust_n, m.seed_sigma_n)
        endpoint, reference_samples = m.x0.copy(), []
        for k in range(20):
            endpoint = rk7(rhs, endpoint, dt / 20)
            if k < 19:
                reference_samples.append(endpoint.copy())
        end, samples = get_arc_flow(m, 0, 20)(
            m.x0 / scale.state,
            m.seed_thrust_n / m.settings.thrust_max_n,
            m.seed_sigma_n / m.settings.thrust_max_n,
            dt / scale.time,
        )
        self.assertEqual(samples.shape, (7, 19))
        np.testing.assert_allclose(
            np.asarray(samples).T * scale.state, reference_samples, rtol=1e-12, atol=1e-7
        )
        np.testing.assert_allclose(
            np.asarray(end).ravel() * scale.state, endpoint, rtol=1e-12, atol=1e-7
        )

    def test_exit_time_derivatives_include_propagation_and_frame_links(self):
        checks = exit_time_derivative_checks(self.problem)
        self.assertLess(checks["earth_exit_duration_error"], 2e-6)
        self.assertLess(checks["mars_exit_duration_error"], 2e-6)

    def test_event_constraints_match_collocation(self):
        m, shooting = self.mission, self.problem
        hs = Transcription(m, m.initial_meshes(), "hs", objective="energy")
        for group in (
            "positive_duration",
            "fixed_tip",
            "fixed_mars_entry",
            "fixed_arrival",
            "frame_links",
            "mass_links",
            "exit_surface",
            "exit_direction",
        ):
            values = []
            for problem in (shooting, hs):
                op = problem.opti
                rows = np.concatenate([np.arange(a, b) for a, b in problem.group_ranges[group]])
                values.append(np.asarray(op.debug.value(op.g, op.initial())).ravel()[rows])
            np.testing.assert_allclose(*values, rtol=0.0, atol=0.0)

    def test_hs_handoff_transfers_candidate_and_rejects_failed_seed(self):
        m = self.mission
        # Synthetic handoff fixture, deliberately distinct from the rollout.
        source = seed_solution(m)
        source.update(
            method="hs", objective="fuel", solver_success=True, solver_status="test fixture"
        )
        for phase in source["phases"]:
            phase["states_si"] = phase["states_si"].copy()
            phase["states_si"][:, 6] -= 10.0
            phase["thrust_n"] = np.full_like(phase["thrust_n"], 0.01)
            phase["sigma_n"] = np.full_like(phase["sigma_n"], np.sqrt(3.0) * 0.01)
            phase["direction"] = phase["thrust_n"] / phase["sigma_n"][:, None]
            phase["throttle"] = phase["sigma_n"] / m.settings.thrust_max_n
        shooting = Transcription(m, m.initial_meshes(), "shooting", source, objective="fuel")
        self.assertTrue(verify_hs_handoff(shooting, source)["verified"])
        with self.assertRaisesRegex(ValueError, "converged Hermite-Simpson"):
            verify_hs_handoff(shooting, dict(source, solver_success=False))
        with self.assertRaisesRegex(ValueError, "same objective"):
            verify_hs_handoff(shooting, dict(source, objective="energy"))


if __name__ == "__main__":
    unittest.main()
