"""Verify the conditional HS/shooting experiment without running any NLP."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import casadi as ca
import numpy as np

from deterministic_optimization import single_phase as shooting


OPTIONS = dict(
    max_iter=5000,
    tol=1e-9,
    constr_viol_tol=1e-10,
    acceptable_iter=0,
    bound_relax_factor=0.0,
    mu_strategy="adaptive",
    print_level=5,
    sb="yes",
    hessian_approximation="exact",
    nlp_scaling_method="gradient-based",
    bound_push=1e-8,
    bound_frac=1e-8,
)


class DirectShootingChecks(unittest.TestCase):

    def test_original_rollout_and_native_boundary_states(self):
        for name, factory in shooting.reference.DETERMINISTIC_CASE_REGISTRY.items():
            with self.subTest(case=name):
                case = factory()
                p = shooting.ShootingProblem(case, "fuel", OPTIONS, intervals=5)
                states, controls, magnitudes = shooting.reference.rollout_initial_guess(
                    case, p.mesh
                )
                np.testing.assert_array_equal(p.seed["states_nd"], states)
                np.testing.assert_array_equal(p.seed["control_nd"], controls)
                np.testing.assert_array_equal(p.seed["magnitude_nd"], magnitudes)
                np.testing.assert_allclose(
                    p.seed["throttle"][:, None] * p.seed["direction"] * case.max_thrust_n,
                    controls.T * case.thrust_unit,
                    rtol=5e-16,
                )
                value = lambda expr: np.asarray(p.opti.debug.value(expr, p.opti.initial()))
                np.testing.assert_array_equal(value(p.X)[:, 0], case.x0_augmented_state)
                self.assertGreater(np.linalg.norm(value(p.X)[:6, -1] - case.xf_state), 1e-5)
                self.assertEqual(p.opti.nx, 7 * 6 + 4 * 5)
                self.assertEqual(p.ipopt_options, OPTIONS)
                groups = {
                    "mass_floor",
                    "throttle",
                    "unit_direction",
                    "dynamics",
                    "initial_state",
                    "terminal_state",
                }
                if shooting.is_cr3bp(case):
                    groups.add("moon_surface_nodes")
                self.assertEqual(set(p.group_ranges), groups)

    def test_zero_throttle_derivatives_and_independent_integrators(self):
        for name, factory in shooting.reference.DETERMINISTIC_CASE_REGISTRY.items():
            with self.subTest(case=name):
                p = shooting.ShootingProblem(factory(), "energy", OPTIONS, intervals=3)
                for check in shooting.flow_checks(p):
                    self.assertLess(check["jacobian_error"], 2e-6)
                    self.assertLess(check["hessian_vector_error"], 2e-5)
                    self.assertLess(check["manual_rk7_difference_nd"], 1e-12)
                    self.assertLess(check["dop853_difference_nd"], 1e-10)
                case = p.case
                for alpha in (0.0, 0.3):
                    dt = 0.01
                    end, samples = p.flow(
                        case.x0_augmented_state, alpha * np.array([0.6, 0.0, 0.8]), alpha, dt
                    )
                    self.assertEqual(samples.shape[1], p.substeps - 1)
                    expected = 1.0 - case.max_thrust_nd * alpha * dt / case.exhaust_velocity_nd
                    self.assertAlmostEqual(float(end[6]), expected, places=13)

    def test_fifty_initial_arcs_for_both_objectives(self):
        for name, factory in shooting.reference.DETERMINISTIC_CASE_REGISTRY.items():
            for objective in ("fuel", "energy"):
                self.assertEqual(shooting.default_intervals(factory(), objective), 50)

    def test_bisection_selects_only_failures_and_respects_capacity(self):
        mesh = np.linspace(0.0, 1.0, 5)
        refined, selected = shooting.refine_mesh(mesh, np.array([1.0, 4.0, 3.0, 2.0]), 2.0, 5)
        self.assertEqual(selected, [1])
        np.testing.assert_array_equal(refined, [0.0, 0.25, 0.375, 0.5, 0.75, 1.0])
        unchanged, selected = shooting.refine_mesh(mesh, np.zeros(4), 2.0, 20)
        np.testing.assert_array_equal(unchanged, mesh)
        self.assertEqual(selected, [])
        with self.assertRaises(ValueError):
            shooting.refine_mesh(mesh, np.array([1.0, 2.0, np.nan, 4.0]), 2.0, 20)

    def test_refined_handoff_preserves_old_nodes_controls_and_integrals(self):
        case = shooting.reference.EarthMars()
        p = shooting.ShootingProblem(case, "fuel", OPTIONS, intervals=5)
        arrays = dict(
            mesh=p.mesh,
            states_nd=p.seed["states_nd"],
            direction=p.seed["direction"].T,
            throttle=p.seed["throttle"],
        )
        mesh, _ = shooting.refine_mesh(p.mesh, np.ones(5), 0.5, 10)
        record = dict(converged=True, method="shooting", objective="fuel", case=case.test_case_id)
        refined = shooting.ShootingProblem(
            case, "fuel", OPTIONS, incoming=(record, arrays), mesh=mesh
        )
        value = lambda expr: np.asarray(refined.opti.debug.value(expr, refined.opti.initial()))
        np.testing.assert_array_equal(value(refined.X)[:, ::2], arrays["states_nd"])
        np.testing.assert_array_equal(value(refined.D), np.repeat(arrays["direction"], 2, axis=1))
        np.testing.assert_array_equal(value(refined.A).ravel(), np.repeat(arrays["throttle"], 2))
        self.assertAlmostEqual(
            np.dot(np.diff(mesh), value(refined.A).ravel()),
            np.dot(np.diff(p.mesh), arrays["throttle"]),
            places=15,
        )
        self.assertAlmostEqual(
            np.dot(np.diff(mesh), np.sum(value(refined.U) ** 2, axis=0)),
            np.dot(np.diff(p.mesh), arrays["throttle"] ** 2),
            places=15,
        )
        self.assertTrue(np.all(np.isfinite(value(refined.X))))
        # Missing old knots must be rejected; this is refinement, not re-meshing.
        with self.assertRaises(ValueError):
            shooting.resample_candidate(case, arrays, np.linspace(0.0, 1.0, 4))

    def test_adaptive_local_criterion_matches_original_driver(self):
        case = shooting.reference.HaloL2ToHaloL1()
        p = shooting.ShootingProblem(case, "fuel", OPTIONS, intervals=8)
        arrays = dict(
            mesh=p.mesh,
            states_nd=p.seed["states_nd"],
            thrust_n=p.seed["control_nd"] * case.thrust_unit,
            throttle=p.seed["throttle"],
        )
        _, relative = shooting.local_defects(p, arrays, 64)
        from deterministic_optimization.rk import rk7

        expected = []
        for k, fraction in enumerate(np.diff(p.mesh)):
            start, target = arrays["states_nd"][:, k : k + 2].T
            endpoint = start.copy()
            rhs = lambda x: shooting.reference.eom(
                case, x, p.seed["control_nd"][:, k], float(p.seed["magnitude_nd"].ravel()[k])
            )
            for _ in range(64):
                endpoint = rk7(rhs, endpoint, case.tof_nd * fraction / 64)
            error = endpoint - target
            expected.append(
                max(
                    np.linalg.norm(error[sl])
                    / max(np.linalg.norm(start[sl]), np.linalg.norm(target[sl]), 1e-13)
                    for sl in (slice(0, 3), slice(3, 6), slice(6, 7))
                )
            )
        np.testing.assert_allclose(relative, expected, atol=1e-13, rtol=1e-8)

    def test_two_steps_only_for_cr3bp(self):
        for factory in shooting.reference.DETERMINISTIC_CASE_REGISTRY.values():
            case = factory()
            p = shooting.ShootingProblem(case, "fuel", OPTIONS, intervals=3)
            self.assertEqual(p.substeps, 2 if shooting.is_cr3bp(case) else 20)

    def test_moon_exclusion_at_mesh_boundaries_in_both_methods(self):
        for method in ("hs", "shooting"):
            for factory in shooting.reference.CASE_REGISTRY.values():
                case = factory()
                p = shooting.ShootingProblem(case, "fuel", OPTIONS, intervals=4, method=method)
                rows = np.array([start for start, end in p.group_ranges["moon_surface_nodes"]])
                self.assertEqual(len(rows), 5)  # Endpoints only; no interior sampling.
                constraint = ca.Function(
                    "moon_constraint", [p.opti.x], [p.opti.g[rows] - p.opti.lbg[rows]]
                )
                jac = ca.Function(
                    "moon_jacobian", [p.opti.x], [ca.jacobian(p.opti.g[rows], p.opti.x)]
                )
                for distance_factor in (0.5, 1.0, 2.0):
                    states = p.seed["states_nd"].copy()
                    states[:3] = np.array(
                        [1.0 - case.mu + distance_factor * case.moon_radius_lu, 0.0, 0.0]
                    )[:, None]
                    p.opti.set_initial(p.X, states)
                    primal = np.asarray(p.opti.debug.value(p.opti.x, p.opti.initial())).ravel()
                    np.testing.assert_allclose(
                        np.asarray(constraint(primal)).ravel(), distance_factor**2 - 1.0, atol=1e-12
                    )
                    derivative = np.asarray(jac(primal))
                    np.testing.assert_allclose(
                        derivative[np.arange(5), np.arange(5) * 7],
                        2 * distance_factor / case.moon_radius_lu,
                        rtol=1e-12,
                    )
                    self.assertTrue(np.all(derivative[:, 7 * 5 :] == 0.0))
                self.assertGreater(
                    shooting.moon_clearance_km(
                        case, np.column_stack([case.x0_state, case.xf_state])
                    ).min(),
                    0.0,
                )

    def test_hs_equations_and_exact_handoff(self):
        case = shooting.reference.EarthMars()
        hs = shooting.ShootingProblem(case, "fuel", OPTIONS, intervals=5, method="hs")
        initial = lambda expr: np.asarray(hs.opti.debug.value(expr, hs.opti.initial()))
        x, u, a = initial(hs.X), initial(hs.U), initial(hs.A).ravel()
        residuals = initial(hs.opti.g).ravel()
        for k, (start, end) in enumerate(hs.group_ranges["dynamics"]):
            h = case.tof_nd * np.diff(hs.mesh)[k]
            fa = hs.mission.rhs(0, x[:, k], u[:, k], a[k])
            fb = hs.mission.rhs(0, x[:, k + 1], u[:, k], a[k])
            middle = 0.5 * (x[:, k] + x[:, k + 1]) + h / 8 * (fa - fb)
            fm = hs.mission.rhs(0, middle, u[:, k], a[k])
            expected = x[:, k + 1] - x[:, k] - h / 6 * (fa + 4 * fm + fb)
            np.testing.assert_allclose(residuals[start:end], expected, atol=5e-16)
        record = dict(converged=True, method="hs", objective="fuel", case=case.test_case_id)
        arrays = dict(mesh=hs.mesh, states_nd=x, direction=initial(hs.D), throttle=a)
        ms = shooting.ShootingProblem(case, "fuel", OPTIONS, intervals=5, incoming=(record, arrays))
        value = lambda expr: np.asarray(ms.opti.debug.value(expr, ms.opti.initial()))
        np.testing.assert_array_equal(value(ms.X), x)
        np.testing.assert_array_equal(value(ms.D), initial(hs.D))
        np.testing.assert_array_equal(value(ms.A).ravel(), a)
        record["converged"] = False
        with self.assertRaises(ValueError):
            shooting.ShootingProblem(case, "fuel", OPTIONS, intervals=5, incoming=(record, arrays))

    def test_diagnostics_report_physical_units_without_an_acceptance_gate(self):
        p = shooting.ShootingProblem(shooting.reference.EarthMars(), "fuel", OPTIONS)
        x = np.zeros_like(p.seed["states_nd"])
        x[:, 0] = p.case.x0_augmented_state
        u = p.seed["control_nd"] * p.case.thrust_unit
        for k, fraction in enumerate(np.diff(p.mesh)):
            end, _ = p.flow(
                x[:, k],
                u[:, k] / p.case.max_thrust_n,
                p.seed["throttle"][k],
                p.case.tof_nd * fraction,
            )
            x[:, k + 1] = np.asarray(end).ravel()
        arrays = dict(mesh=p.mesh, states_nd=x, thrust_n=u, throttle=p.seed["throttle"])
        with tempfile.TemporaryDirectory() as temporary:
            report = shooting.integration_diagnostics(p, arrays, Path(temporary))
        self.assertTrue(report["report_only"])
        self.assertFalse(report["used_for_acceptance"])
        self.assertEqual(report["local"]["substeps"], 64)
        self.assertLess(report["local"]["max_absolute_scaled_component"], 1e-11)
        self.assertLess(
            report["continuous"]["comparisons"]["tightest"]["error_against_optimizer"][
                "max_mass_kg"
            ],
            1e-8,
        )
        self.assertFalse(report["continuous"]["state_resets"])


if __name__ == "__main__":
    unittest.main()
