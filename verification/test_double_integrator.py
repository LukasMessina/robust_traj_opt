"""Physical equivalence and derivative checks for the SNOPT coordinate changes."""

import unittest
import numpy as np
import jax
from scipy.stats import norm
import stochastic_trajectory_optimization.double_integrator as d


class Coordinates(unittest.TestCase):
    def test_zero_control_has_a_unit_direction(self):
        o = d.Options(n_arcs=3)
        controls = np.array([[0.0, 1.0, -0.2], [0.0, -0.3, 0.4]])
        polar = d.polar_seed(controls, o)
        np.testing.assert_allclose(
            np.sum(polar["direction"].reshape(2, 3) ** 2, axis=0), 1.0, atol=2e-16
        )
        np.testing.assert_allclose(
            d.cartesian_variables(polar, o)["feedforward"].reshape(2, 3),
            controls,
            atol=1e-16,
        )

    def test_scaled_factor_defects_and_polar_derivatives(self):
        o = d.Options(n_arcs=3)
        a, b, g = d.discrete_time_matrices(o)
        ac, _ = d.continuous_time_matrices()
        _, lc = d.calibrate_continuous_diffusion(ac, g @ g.T, o.dt)
        arc = d.build_propagation_arc_map(o, lc)
        gains = np.tile(-0.1 * b.T, (o.n_arcs, 1, 1))
        controls = np.array([[0.3, -0.2, 0.1], [0.1, 0.25, -0.3]])
        means = np.empty((d.NX, 4))
        means[:, 0] = o.x0_mean
        covariances = [o.x0_covariance]
        for k in range(o.n_arcs):
            mean, covariance, _ = arc(
                means[:, k], covariances[-1], controls[:, k], gains[k]
            )
            means[:, k + 1] = mean
            covariances.append(np.asarray(covariance))
        covariances = np.asarray(covariances)
        std = np.sqrt(np.maximum(np.diagonal(covariances, axis1=1, axis2=2), 1e-12))
        packed = np.stack(
            [
                d.LTRI.pack_numeric(np.linalg.cholesky(p / np.outer(s, s)))
                for p, s in zip(covariances, std)
            ]
        )
        x = dict(
            means=means.ravel(),
            feedforward=controls.ravel(),
            gains=gains.ravel(),
            cholesky_factor=packed.ravel(),
            terminal_margin=d.LTRI.pack_numeric(np.eye(d.NX)),
        )
        psi = d.psi_inverse(d.NU, 1 - o.control_confidence)
        phi = norm.ppf(o.path_confidence)
        values = jax.jit(lambda v: d._stochastic_funcs(v, o, arc, psi, phi, std))
        result = values(x)
        np.testing.assert_allclose(result["state_covariance_defects"], 0.0, atol=2e-13)
        np.testing.assert_allclose(result["initial_covariance"], 0.0, atol=2e-13)
        np.testing.assert_allclose(result["mean_defects"], 0.0, atol=2e-13)
        blocks = jax.jit(
            lambda v: d._nonlinear_jacobian_stochastic(v, o, arc, psi, phi, std)
        )(x)
        jac = d._assemble_nonlinear_global_jacobian_stochastic(
            *map(np.asarray, blocks), o
        )
        polar = {k: v for k, v in x.items() if k != "feedforward"} | d.polar_seed(
            controls, o
        )
        rng = np.random.default_rng(7)
        direction = {k: rng.normal(size=v.shape) * 0.01 for k, v in polar.items()}

        def evaluate(v):
            result = dict(values(d.cartesian_variables(v, o)))
            result["unit_direction"] = np.sum(v["direction"].reshape(2, 3) ** 2, axis=0)
            return result

        # Keep all structurally present entries, including zeros at this point.
        sparse = {}
        for name, blocks in jac.items():
            sparse[name] = {}
            for variable, block in blocks.items():
                if name == "objective":
                    sparse[name][variable] = block
                    continue
                rows, cols = np.indices(block.shape)
                sparse[name][variable] = {
                    "coo": [rows.ravel(), cols.ravel(), block.ravel()],
                    "shape": block.shape,
                }
        pj = d.polar_jacobian(sparse, polar, o)
        h = 1e-5
        plus = evaluate({k: v + h * direction[k] for k, v in polar.items()})
        minus = evaluate({k: v - h * direction[k] for k, v in polar.items()})
        from scipy.sparse import coo_matrix

        for name, blocks in pj.items():
            predicted = 0.0
            for variable, block in blocks.items():
                if name == "objective":
                    predicted += block @ direction[variable]
                else:
                    rows, cols, vals = block["coo"]
                    predicted += (
                        coo_matrix((vals, (rows, cols)), shape=block["shape"])
                        @ direction[variable]
                    )
            finite = (np.asarray(plus[name]) - np.asarray(minus[name])) / (2 * h)
            np.testing.assert_allclose(
                predicted, finite, rtol=2e-5, atol=2e-8, err_msg=name
            )


if __name__ == "__main__":
    unittest.main()
