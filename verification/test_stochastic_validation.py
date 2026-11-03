"""Independent checks of saved-run cost, covariance scaling and terminal rules."""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import jax.numpy as jnp
import numpy as np

import stochastic_trajectory_optimization.dynamics as d
from verification.stochastic.single_phase import TrajectoryResiduals, recover_solution


class SavedRunResiduals(unittest.TestCase):
    def make_problem(self, terminal_cost):
        options = replace(
            d.Options(),
            state_covariance_weight=2.0,
            control_covariance_weight=3.0,
            position_covariance_reduction=4.0,
            velocity_covariance_reduction=9.0,
            control_norm_eps=0.0,
        )
        means = np.zeros((7, 3))
        feed = np.array([[0.1, 0.2], [0.0, 0.0], [0.0, 0.0]])
        ref = d.ReferenceTraj(
            np.array([0.0, 1.0, 4.0]), np.array([1.0, 3.0]), means, feed, 0.0
        )
        std = np.arange(1.0, 22.0).reshape(3, 7)
        data = dict(
            terminal_covariance_cost=int(terminal_cost),
            terminal_covariance_weight=1.5,
            gain_scale=np.ones(2),
            node_std=std.T,
            covariance_scale_variances=std**2,
        )
        case = SimpleNamespace(x0_augmented_state=means[:, 0], xf_state=means[:6, -1])

        def arc(mean, covariance, control, gain, duration):
            return mean, covariance, jnp.diag(0.02 + control**2)

        residuals = TrajectoryResiduals(
            case, options, ref, d.Normalization(np.ones(7), np.ones(7)), arc, data
        )
        factors = np.stack([d.LTRI.pack(np.diag(1.0 / s)) for s in std[1:]])
        x = dict(
            means=means.ravel(),
            feedforward=feed.ravel(),
            gains=np.zeros(36),
            cholesky_factor=factors.ravel(),
            terminal_margin=np.asarray(d.MTRI.pack(np.eye(6))),
        )
        return residuals, x

    def test_duration_weighted_cost_and_terminal_penalty(self):
        residuals, x = self.make_problem(terminal_cost=True)
        values, failed = residuals.objconfun(x)
        self.assertFalse(failed)
        # Duration weights are 0.5 and 1.5; each covariance is the identity.
        mean = 0.5 * 0.1 + 1.5 * 0.2
        state = 2.0 * 2.0 * 6.0 / 9.0
        control = 3.0 * (0.5 * 0.07 + 1.5 * 0.10)
        terminal = 1.5 * (3 * 4.0 + 3 * 9.0) / 9.0
        self.assertAlmostEqual(
            float(values["objective"]), mean + state + control + terminal, places=13
        )
        np.testing.assert_allclose(values["state_covariance_defects"], 0.0, atol=1e-15)
        np.testing.assert_array_equal(values["terminal_residual"], 0.0)
        solution = recover_solution(residuals, x, solver_success=True)
        self.assertAlmostEqual(
            solution.diagnostics["objective_mean_control"], mean, places=14
        )
        self.assertAlmostEqual(
            solution.diagnostics["objective_terminal_covariance"], terminal, places=14
        )
        np.testing.assert_allclose(
            solution.covariances, np.tile(np.eye(7), (3, 1, 1)), atol=1e-15
        )

    def test_step4_enforces_terminal_bound_without_terminal_cost(self):
        residuals, x = self.make_problem(terminal_cost=False)
        residuals.options = replace(residuals.options, objective="quantile")
        values, failed = residuals.objconfun(x)
        self.assertFalse(failed)
        weights = np.array([0.5, 1.5])
        # Distinct largest eigenvalues avoid degeneracy in the independent check.
        radii = [
            float(
                d.spectral_radius(
                    jnp.diag(jnp.array(t)),
                    residuals.options.spectral_radius_floor,
                    residuals.options.eigenvalue_smoothing,
                )
            )
            for t in ([0.03, 0.02, 0.02], [0.06, 0.02, 0.02])
        ]
        expected = weights @ (
            np.array([0.1, 0.2]) + d.psi_inverse(3, 0.01) * np.array(radii)
        )
        self.assertAlmostEqual(float(values["objective"]), expected, places=13)
        np.testing.assert_allclose(
            values["terminal_residual"],
            d.MTRI.pack(-np.diag([4.0] * 3 + [9.0] * 3)),
            atol=1e-14,
        )


if __name__ == "__main__":
    unittest.main()
