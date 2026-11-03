"""Chance-constrained covariance steering of a planar double integrator.

SNOPT uses throttle/unit-direction controls and seed-scaled covariance factors.
The numerical model and Sobol validation share the original ShARK propagation.
"""

from __future__ import annotations


import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
from scipy.stats import chi2, norm


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyoptsparse import Optimization


from integrator.double_integrator import (
    NX,
    NU,
    unscented_weights as unscented_weights,
    discrete_time_matrices as discrete_time_matrices,
    continuous_time_matrices as continuous_time_matrices,
    accumulated_process_covariance as accumulated_process_covariance,
    calibrate_continuous_diffusion as calibrate_continuous_diffusion,
    _drift_vector_field as _drift_vector_field,
    _TERM as _TERM,
    _SOLVER as _SOLVER,
    diffrax_step as diffrax_step,
    _diffrax_step_batch as _diffrax_step_batch,
    _diffusion_vector_field as _diffusion_vector_field,
    _SDE_SOLVER as _SDE_SOLVER,
    PrescribedSpaceTimeLevyPath as PrescribedSpaceTimeLevyPath,
    prescribed_integration_sde_step as prescribed_integration_sde_step,
    _prescribed_sde_integration_step_batch as _prescribed_sde_integration_step_batch,
)

OUTPUT_DIR = Path("output/double_integrator")


# --------------------------------------------------------------------------- #
#                              Problem data
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Options:
    dt: float = 0.25
    tf: float = 5.0
    n_arcs: int = 20

    x0_mean: np.ndarray = field(default_factory=lambda: np.array([2.0, 4.0, 3.0, 2.0]))
    x0_covariance: np.ndarray = field(
        default_factory=lambda: np.diag([0.1, 0.1, 0.02, 0.02])
    )
    xf_mean: np.ndarray = field(default_factory=lambda: np.array([8.0, 2.0, 0.0, 0.0]))
    xf_covariance: np.ndarray = field(
        default_factory=lambda: np.diag([0.06, 0.06, 0.006, 0.006])
    )

    a1: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 0.0, 0.0]))
    b1: float = 12.75
    a2: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.1, 0.0, 0.0]))
    b2: float = 8.75
    path_confidence: float = 0.9973  # probability of path constraint satisfaction (p_x)

    u_max: float = 2.0
    control_confidence: float = (
        0.9973  # probability of control constraint satisfaction (p_u)
    )

    # Cost weights (Bryson's rule is used to warm-start the TVLQR gains instead of these)
    Q: np.ndarray = field(default_factory=lambda: 0.01 * np.eye(NX))
    R: np.ndarray = field(default_factory=lambda: np.eye(NU))

    scaling_parameter: float = 0.0  # UT scaling parameter
    spectral_eigenvalue_smoothing: float = 1e-6
    terminal_margin_floor: float = 1e-7
    warm_start_gains: bool = True

    control_norm_epsilon: float = 1e-6

    major_max_iter: int = 2000000
    minor_max_iter: int = 100 * major_max_iter

    major_optimality_tol: float = 1e-5
    major_feasibility_tol: float = 1e-9
    minor_feasibility_tol: float = 1e-9
    print_level: int = 1
    print_sparsity: bool = False

    monte_carlo_samples: int = 8192  # Sobol sampling requires a power of two.
    monte_carlo_seed: int = 42

    @property
    def path_constraints(self) -> tuple[tuple[np.ndarray, float], ...]:
        return ((self.a1, self.b1), (self.a2, self.b2))

    @property
    def path_matrix(self) -> jnp.ndarray:
        """Stacked path-constraint normal vectors, shape (n_path, NX)."""

        return jnp.stack(
            [jnp.asarray(a_vector) for a_vector, _ in self.path_constraints]
        )

    @property
    def inverse_terminal_std(self) -> np.ndarray:
        """Elementwise 1/sigma for the terminal state_covariance matrix targets."""

        return 1.0 / np.sqrt(np.diag(self.xf_covariance))


def psi_inverse(dimension: int, beta: float) -> float:
    """Psi_d^-1(beta) = sqrt(Phi_d^-1(1 - beta)) with Phi_d the chi-squared CDF."""

    return float(np.sqrt(chi2.ppf(1.0 - beta, dimension)))


# --------------------------------------------------------------------------- #
#                   Lower-triangular packing helper
# --------------------------------------------------------------------------- #


class LowerTriangular:
    """Maps between an NxN matrix and its packed lower-triangular entries."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.indices: tuple[tuple[int, int], ...] = tuple(
            (row, column) for row in range(dimension) for column in range(row + 1)
        )
        self.rows = np.array([row for row, _ in self.indices])
        self.columns = np.array([column for _, column in self.indices])
        self.diagonal_positions: np.ndarray = np.array(
            [
                position
                for position, (row, column) in enumerate(self.indices)
                if row == column
            ]
        )

    def __len__(self) -> int:
        return len(self.indices)

    def to_matrix(self, entries: jnp.ndarray) -> jnp.ndarray:
        """Assemble the full lower-triangular matrix from packed entries."""

        matrix = jnp.zeros((self.dimension, self.dimension), dtype=entries.dtype)
        matrix = matrix.at[self.rows, self.columns].set(entries)
        return matrix

    def to_matrix_numeric(self, entries: np.ndarray) -> np.ndarray:
        matrix = np.zeros((self.dimension, self.dimension))
        matrix[self.rows, self.columns] = np.asarray(entries).ravel()
        return matrix

    def pack(self, matrix: jnp.ndarray) -> jnp.ndarray:
        return matrix[self.rows, self.columns]

    def pack_numeric(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix)[self.rows, self.columns]


LTRI = LowerTriangular(NX)


def nearest_pd_cholesky_factor(matrix: np.ndarray, jitter: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    psd = eigenvectors @ np.diag(np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
    return np.linalg.cholesky(psd + jitter * np.eye(matrix.shape[0]))


def apply_jacobian_sparsity(jacobian, sparsity: dict) -> dict:
    """Return sensitivities with fixed pyOptSparse COO row/column indices."""

    sparse_jacobian = {}
    for output, variable_blocks in jacobian.items():
        if output not in sparsity:
            sparse_jacobian[output] = {
                variable: np.asarray(block)
                for variable, block in variable_blocks.items()
            }
            continue

        sparse_blocks = {}
        for variable, pattern in sparsity[output].items():
            rows, columns, _ = pattern["coo"]
            shape = tuple(pattern["shape"])
            derivative = np.asarray(variable_blocks[variable]).reshape(shape)
            sparse_blocks[variable] = {
                "coo": [
                    rows.astype(np.intc),
                    columns.astype(np.intc),
                    derivative[rows, columns],
                ],
                "shape": shape,
            }
        sparse_jacobian[output] = sparse_blocks

    return sparse_jacobian


def get_jacobian_sparsity(jacobian_template: dict) -> dict:
    """Build fixed COO (coordinate format) patterns from a structural
    jacobian template."""

    sparsity = {}
    for output, variable_blocks in jacobian_template.items():
        if output == "objective":
            continue
        sparsity[output] = {}
        for variable, block in variable_blocks.items():
            array = np.atleast_2d(np.asarray(block))
            rows, columns = np.nonzero(array)
            if rows.size:
                sparsity[output][variable] = {
                    "coo": [
                        rows.astype(np.intc),
                        columns.astype(np.intc),
                        np.ones(rows.size),
                    ],
                    "shape": array.shape,
                }
    return sparsity


# --------------------------------------------------------------------------- #
# Discrete-time (ZOH) and continuous-time system matrices
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#             Propagation of the linear, zero-order-held dynamics
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#           Smoothed spectral radius of the control covariance
# --------------------------------------------------------------------------- #


def numerical_psqrt_spectral_radius(
    matrix: np.ndarray,
    eigenvalue_smoothing: float,
) -> float:
    array = np.asarray(matrix, dtype=float)
    symmetric = 0.5 * (array + array.T)
    a11, a22, a12 = symmetric[0, 0], symmetric[1, 1], symmetric[0, 1]
    mean = 0.5 * (a11 + a22)
    radicand = 0.25 * (a11 - a22) ** 2 + a12**2
    half_spread = np.sqrt(radicand + eigenvalue_smoothing**2)
    lambda_max = mean + half_spread
    return float(np.sqrt(max(lambda_max, 0.0)))


def symbolic_psqrt_spectral_radius(
    matrix: jnp.ndarray,
    eigenvalue_smoothing: float,
) -> jnp.ndarray:
    """Smooth upper approximation of sqrt(lambda_max(A)) for symmetric 2x2 A (jax)."""

    a11, a22, a12 = matrix[0, 0], matrix[1, 1], matrix[0, 1]
    mean = 0.5 * (a11 + a22)
    radicand = 0.25 * (a11 - a22) ** 2 + a12**2
    half_spread = jnp.sqrt(radicand + eigenvalue_smoothing**2)
    lambda_max = mean + half_spread
    return jnp.sqrt(jnp.maximum(lambda_max, 0.0))


def get_smoothing_diagnostics(
    control_covariances: np.ndarray, options: Options
) -> tuple[np.ndarray, dict[str, float]]:
    exact_eigenvalues = np.linalg.eigvalsh(control_covariances)[..., -1]
    exact_psqrt_spectral_radii = np.sqrt(np.clip(exact_eigenvalues, 0.0, None))
    estimated_psqrt_spectral_radii = np.array(
        [
            numerical_psqrt_spectral_radius(
                cov,
                options.spectral_eigenvalue_smoothing,
            )
            for cov in control_covariances
        ]
    )
    bias = estimated_psqrt_spectral_radii - exact_psqrt_spectral_radii
    peak_exact_psqrt_spectral_radius = float(np.max(exact_psqrt_spectral_radii))
    psi_inv_u = psi_inverse(NU, 1.0 - options.control_confidence)
    return estimated_psqrt_spectral_radii, {
        "max_psqrt_spectral_radius_smoothing_bias": float(np.max(bias)),
        "relative_psqrt_spectral_radius_smoothing_bias": float(
            np.max(bias) / peak_exact_psqrt_spectral_radius
        ),
        "max_control_margin_smoothing_bias": float(psi_inv_u * np.max(bias)),
    }


# --------------------------------------------------------------------------- #
#        Augmented Unscented Transform for state covariance propagation
# --------------------------------------------------------------------------- #


def build_propagation_arc_map(options: Options, Lc: np.ndarray):
    """Return an arc propagation map.

    Uses an augmented-UT construction implementation:
    sigma points over ``z_k = [x_k; xi_k; eta_k]`` with covariance
    ``blkdiag(P_k, I, I)`` and a batched propagation. The two independent
    standard-normal coordinates define the Brownian increment and normalized
    space-time Levy area as ``W = sqrt(dt) xi`` and
    ``H = sqrt(dt / 12) eta``.
    """

    noise_matrix = np.asarray(Lc, dtype=float)
    noise_dimension = noise_matrix.shape[1]
    stochastic_dimension = 2 * noise_dimension
    total_dimension = NX + stochastic_dimension
    ut_scale = total_dimension + options.scaling_parameter
    if ut_scale <= 0.0:
        raise ValueError("unscented transform scale must be positive")

    weights = jnp.asarray(unscented_weights(options.scaling_parameter, total_dimension))
    diffusion_j = jnp.asarray(noise_matrix)
    dt = options.dt

    def propagation_arc(
        mean: jnp.ndarray,
        state_covariance: jnp.ndarray,
        feedforward: jnp.ndarray,
        gain: jnp.ndarray,
    ):
        state_spread = jnp.linalg.cholesky(ut_scale * state_covariance)
        state_offsets = jnp.concatenate(
            [state_spread, jnp.zeros((NX, stochastic_dimension))], axis=1
        )
        stochastic_offsets = jnp.concatenate(
            [
                jnp.zeros((stochastic_dimension, NX)),
                jnp.sqrt(ut_scale) * jnp.eye(stochastic_dimension),
            ],
            axis=1,
        )
        state_matrix_pts = jnp.concatenate(
            [
                mean[:, None],
                mean[:, None] + state_offsets,
                mean[:, None] - state_offsets,
            ],
            axis=1,
        )
        stochastic_matrix_pts = jnp.concatenate(
            [
                jnp.zeros((stochastic_dimension, 1)),
                stochastic_offsets,
                -stochastic_offsets,
            ],
            axis=1,
        )

        controls = feedforward[:, None] + gain @ (
            state_matrix_pts - mean[:, None]
        )  # (NU, n_sigma)

        brownian_increments = jnp.sqrt(dt) * stochastic_matrix_pts[:noise_dimension]
        space_time_levy_areas = (
            jnp.sqrt(dt / 12.0) * stochastic_matrix_pts[noise_dimension:]
        )
        propagated_states = _prescribed_sde_integration_step_batch(
            state_matrix_pts,
            controls,
            brownian_increments,
            space_time_levy_areas,
            diffusion_j,
            dt,
        )

        mean_next = propagated_states @ weights
        residual = propagated_states - mean_next[:, None]
        covariance_next = (residual * weights[None, :]) @ residual.T
        covariance_next = 0.5 * (covariance_next + covariance_next.T)

        control_mean = controls @ weights
        control_residual = controls - control_mean[:, None]
        control_covariance = (control_residual * weights[None, :]) @ control_residual.T
        control_covariance = 0.5 * (control_covariance + control_covariance.T)

        return mean_next, covariance_next, control_covariance

    return propagation_arc


def propagate_statistical_moments(
    arc_fn,
    means: np.ndarray,
    feedforward: np.ndarray,
    gains: np.ndarray,
    initial_state_covariance: np.ndarray,
):
    """Forward-integrate node state covariances arc by arc, seeding each from the last."""

    n_arcs = feedforward.shape[1]
    state_covariances = np.empty((n_arcs + 1, NX, NX))
    control_covariances = np.empty((n_arcs, NU, NU))
    propagated_means = np.empty((NX, n_arcs + 1))
    state_covariances[0] = initial_state_covariance
    propagated_means[:, 0] = means[:, 0]
    for k in range(n_arcs):
        mean_next, covariance_next, control_covariance = arc_fn(
            jnp.asarray(means[:, k]),
            jnp.asarray(state_covariances[k]),
            jnp.asarray(feedforward[:, k]),
            jnp.asarray(gains[k]),
        )
        propagated_means[:, k + 1] = np.asarray(mean_next)
        state_covariances[k + 1] = np.asarray(covariance_next)
        control_covariances[k] = np.asarray(control_covariance)
    return propagated_means, state_covariances, control_covariances


def evaluate_node_matching_defects(
    arc_fn, means, feedforward, gains, state_covariances
):
    """Compare each node's given mean/state_covariance against one arc's prediction from the prior node."""

    n_arcs = feedforward.shape[1]
    control_covariances = np.empty((n_arcs, NU, NU))
    max_mean_defect = 0.0
    max_state_covariance_defect = 0.0
    for k in range(n_arcs):
        mean_next, state_covariance_next, control_covariance = arc_fn(
            jnp.asarray(means[:, k]),
            jnp.asarray(state_covariances[k]),
            jnp.asarray(feedforward[:, k]),
            jnp.asarray(gains[k]),
        )
        control_covariances[k] = np.asarray(control_covariance)
        mean_error = np.asarray(mean_next) - means[:, k + 1]
        state_covariance_error = (
            np.asarray(state_covariance_next) - state_covariances[k + 1]
        )
        max_mean_defect = max(max_mean_defect, float(np.max(np.abs(mean_error))))
        max_state_covariance_defect = max(
            max_state_covariance_defect, float(np.max(np.abs(state_covariance_error)))
        )
    return control_covariances, max_mean_defect, max_state_covariance_defect


# --------------------------------------------------------------------------- #
#                   TVLQR gain warm start (Bryson's rule)
# --------------------------------------------------------------------------- #


def tvlqr_gains(options: Options, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Return a TVLQR gain seed using componentwise weights derived from Bryson's rule.

    The acceptable running state deviations are the initial one-sigma values,
    the acceptable control deviation is the maximum control in each direction,
    and the acceptable terminal deviations are the requested terminal
    one-sigma values.  Bryson's rule therefore gives

        Q_ii = 1 / x0_sigma_i**2,
        R_jj = 1 / u_max**2,
        Q_N,ii = 1 / xf_sigma_i**2.
    """

    initial_variances = np.diag(options.x0_covariance)
    terminal_variances = np.diag(options.xf_covariance)
    if np.any(initial_variances <= 0.0):
        raise ValueError(
            "initial state covariance diagonal must be positive for Bryson weights"
        )
    if np.any(terminal_variances <= 0.0):
        raise ValueError(
            "final covariance diagonal must be positive for Bryson weights"
        )
    if options.u_max <= 0.0:
        raise ValueError("maximum control must be positive for Bryson weights")

    Q = np.diag(1.0 / initial_variances)
    R = np.eye(NU) / options.u_max**2
    P = np.diag(1.0 / terminal_variances)
    gains = np.zeros((options.n_arcs, NU, NX))
    for k in reversed(range(options.n_arcs)):
        gain = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        gains[k] = -gain
        closed_loop = A - B @ gain
        P = Q + gain.T @ R @ gain + closed_loop.T @ P @ closed_loop
        P = 0.5 * (P + P.T)
    return gains


# --------------------------------------------------------------------------- #
#                        SNOPT settings and diagnostics
# --------------------------------------------------------------------------- #


def snopt_options(options: Options, tag: str) -> dict:
    work_dir = Path(tempfile.mkdtemp(prefix=f"pyopt-snopt-{tag}-"))
    return {
        "Major iterations limit": options.major_max_iter,
        "Minor iterations limit": max(500, options.minor_max_iter),
        "Iterations limit": options.minor_max_iter,
        "Major optimality tolerance": options.major_optimality_tol,
        "Major feasibility tolerance": options.major_feasibility_tol,
        "Minor feasibility tolerance": options.minor_feasibility_tol,
        "Print file": str(work_dir / "SNOPT_print.out"),
        # pyOptSparse uses Fortran unit 6 for stdout.  Send SNOPT's live major-
        # iteration summary there when terminal output is enabled; keep the
        # more detailed print stream in SNOPT_print.out.
        "iSumm": 6 if options.print_level else 0,
        "Summary file": str(work_dir / "SNOPT_summary.out"),
        "Major print level": 1,
        "Minor print level": 0,
    }


def optinform_text(sol) -> str:
    """SNOPT's exit message.

    pyoptsparse >= ~2.11 returns a ``SolutionInform`` dataclass (``.message``);
    older releases return a plain ``{"text": ...}`` dict. Both are handled.
    """

    optInform = sol.optInform
    if hasattr(optInform, "message"):
        return str(optInform.message)
    return str(optInform.get("text", "unknown"))


def solver_diagnostics(
    sol, optProb: Optimization, xdict: dict, funcs: dict, options: Options
) -> dict[str, float | str]:
    """Feasibility/optimality diagnostics."""

    funcs = dict(funcs)
    # NOTE: the linear constraints are not automatically evaluated in
    # the pyoptsparse solution object, so we must do it here to get the
    # constraint values.
    optProb.evaluateLinearConstraints(optProb.processXtoVec(xdict), funcs)
    max_violation = 0.0
    for con_name, con in optProb.constraints.items():
        value = np.asarray(funcs[con_name]).ravel()
        lower = np.asarray(con.lower, dtype=float).ravel()
        upper = np.asarray(con.upper, dtype=float).ravel()
        lower_violation = np.maximum(lower - value, 0.0)
        upper_violation = np.maximum(value - upper, 0.0)
        max_violation = max(
            max_violation,
            float(np.max(lower_violation)),
            float(np.max(upper_violation)),
        )

    acceptance_tolerance = 10.0 * options.major_feasibility_tol
    status_text = optinform_text(sol)
    optimality_satisfied = (
        "optimality conditions satisfied" in status_text.lower()
        and "infeasible" not in status_text.lower()
    )
    feasible = max_violation <= acceptance_tolerance
    return {
        "converged": float(optimality_satisfied and feasible),
        "optimality_satisfied": float(optimality_satisfied),
        "feasible": float(feasible),
        "max_constraint_violation": max_violation,
        "solver_status": status_text,
    }


# --------------------------------------------------------------------------- #
#                           Deterministic trajectory
# --------------------------------------------------------------------------- #


@dataclass
class NominalTrajectory:
    means: np.ndarray
    feedforward: np.ndarray
    converged: bool


def _linear_jacobians(options: Options, *, stochastic: bool) -> dict:
    """Constant COO (coordinate sparse matrix) format blocks for endpoints, nominal paths, and factor diagonals."""

    n_arcs = options.n_arcs
    state_indices = np.arange(NX, dtype=np.intc)
    mean_size = NX * (n_arcs + 1)

    def build_coo_block(rows, columns, values, shape):
        return {
            "coo": [
                np.asarray(rows, dtype=np.intc),
                np.asarray(columns, dtype=np.intc),
                np.asarray(values, dtype=float),
            ],
            "shape": shape,
        }

    jacobian = {
        name: {
            "means": build_coo_block(
                state_indices,
                state_indices * (n_arcs + 1) + node,
                np.ones(NX),
                (NX, mean_size),
            )
        }
        for name, node in (("boundary_start", 0), ("boundary_end", n_arcs))
    }
    if stochastic:
        n_tril = len(LTRI)
        positions = np.asarray(LTRI.diagonal_positions)
        jacobian["margin_diagonal"] = {
            "terminal_margin": build_coo_block(
                state_indices, positions, np.ones(NX), (NX, n_tril)
            )
        }
        arc_count = NX * (n_arcs + 1)
        columns = (np.arange(n_arcs + 1)[:, None] * n_tril + positions).reshape(-1)
        jacobian["cholesky_diagonal"] = {
            "cholesky_factor": build_coo_block(
                np.arange(arc_count),
                columns,
                np.ones(arc_count),
                (arc_count, (n_arcs + 1) * n_tril),
            )
        }
    else:
        path = np.zeros((len(options.path_constraints) * n_arcs, mean_size))
        for path_index, (normal, _) in enumerate(options.path_constraints):
            for node in range(n_arcs):
                path[
                    path_index * n_arcs + node, state_indices * (n_arcs + 1) + node
                ] = normal
        rows, columns = np.nonzero(path)
        jacobian["path"] = {
            "means": build_coo_block(rows, columns, path[rows, columns], path.shape)
        }
    return jacobian


def _nominal_propagation_step(
    state: jnp.ndarray, control: jnp.ndarray, dt: float
) -> jnp.ndarray:
    return diffrax_step(state, control, dt)


def _nominal_funcs(xdict: dict, options: Options):
    means = xdict["means"].reshape(NX, options.n_arcs + 1)
    feedforward = xdict["feedforward"].reshape(NU, options.n_arcs)

    propagation_step_batch = jax.vmap(
        _nominal_propagation_step, in_axes=(1, 1, None), out_axes=1
    )
    predicted_next = propagation_step_batch(means[:, :-1], feedforward, options.dt)
    defects = (means[:, 1:] - predicted_next).reshape(-1)

    control_energy = options.dt * jnp.sum(feedforward * feedforward)

    funcs = {
        "defects": defects,
        "objective": control_energy,
    }
    return funcs


def _nonlinear_arc_outputs_nominal(
    local_input: jnp.ndarray, options: Options
) -> jnp.ndarray:
    """Nonlinear (Diffrax-propagated) objective and defect quantities owned by one arc."""

    mean = local_input[:NX]
    feedforward = local_input[NX:]
    predicted_mean = _nominal_propagation_step(mean, feedforward, options.dt)
    local_objective = options.dt * jnp.dot(feedforward, feedforward)
    return jnp.concatenate([predicted_mean, jnp.atleast_1d(local_objective)])


def _nonlinear_jacobian_nominal(xdict: dict, options: Options) -> jnp.ndarray:
    """AD-differentiate one nominal arc's nonlinear outputs and batch over all arcs."""

    means = xdict["means"].reshape(NX, options.n_arcs + 1)
    feedforward = xdict["feedforward"].reshape(NU, options.n_arcs)
    local_inputs = jnp.concatenate([means[:, :-1].T, feedforward.T], axis=1)
    local_jacobian = jax.jacrev(
        lambda values: _nonlinear_arc_outputs_nominal(values, options)
    )
    return jax.vmap(local_jacobian)(local_inputs)


def _assemble_nonlinear_jacobian_nominal(
    nonlinear_jacobian_blocks: np.ndarray, options: Options
) -> dict:
    """Assemble the nonlinear per-arc AD Jacobian blocks in global ordering."""

    n_arcs = options.n_arcs
    mean_size = NX * (n_arcs + 1)
    feedforward_size = NU * n_arcs
    jacobian = {
        "defects": {
            "means": np.zeros((NX * n_arcs, mean_size)),
            "feedforward": np.zeros((NX * n_arcs, feedforward_size)),
        },
        "objective": {
            "means": np.zeros(mean_size),
            "feedforward": np.zeros(feedforward_size),
        },
    }

    state_indices = np.arange(NX)
    control_indices = np.arange(NU)
    for arc_index in range(n_arcs):
        local = nonlinear_jacobian_blocks[arc_index]
        defect_rows = state_indices * n_arcs + arc_index
        mean_columns = state_indices * (n_arcs + 1) + arc_index
        next_mean_columns = mean_columns + 1
        feedforward_columns = control_indices * n_arcs + arc_index

        jacobian["defects"]["means"][np.ix_(defect_rows, mean_columns)] = -local[
            :NX, :NX
        ]
        jacobian["defects"]["means"][np.ix_(defect_rows, next_mean_columns)] += np.eye(
            NX
        )
        jacobian["defects"]["feedforward"][
            np.ix_(defect_rows, feedforward_columns)
        ] = -local[:NX, slice(NX, NX + NU)]

        jacobian["objective"]["means"][mean_columns] = local[NX, :NX]
        jacobian["objective"]["feedforward"][feedforward_columns] = local[
            NX, slice(NX, NX + NU)
        ]

    return jacobian


def _nonlinear_jacobian_sparsity_template_nominal(
    A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Exact local nonzero structure for the affine nominal problem."""

    structure = np.zeros((NX + 1, NX + NU))
    structure[:NX, :NX] = A != 0.0
    structure[:NX, NX:] = B != 0.0
    structure[-1, NX:] = 1.0
    return structure


def polar_seed(feedforward, options):
    """Encode physical controls as nonnegative throttle and unit directions."""
    magnitude = np.linalg.norm(feedforward, axis=0)
    direction = np.zeros_like(feedforward)
    direction[0] = 1.0
    np.divide(feedforward, magnitude[None], out=direction, where=magnitude[None] > 0)
    return {"direction": direction.ravel(), "throttle": magnitude / options.u_max}


def cartesian_variables(variables, options):
    """Decode controls while retaining the original numerical model."""
    result = {
        key: value
        for key, value in variables.items()
        if key not in ("direction", "throttle")
    }
    direction = variables["direction"].reshape(NU, options.n_arcs)
    result["feedforward"] = (
        options.u_max * direction * variables["throttle"][None]
    ).ravel()
    return result


def polar_jacobian(jacobian, variables, options):
    """Apply the control-coordinate chain rule to fixed sparse Jacobian blocks."""
    from scipy.sparse import coo_matrix

    n = options.n_arcs
    direction = np.asarray(variables["direction"])
    throttle = np.asarray(variables["throttle"])
    result = {}
    for name, blocks in jacobian.items():
        transformed = {
            key: value for key, value in blocks.items() if key != "feedforward"
        }
        if "feedforward" in blocks:
            block = blocks["feedforward"]
            if name == "objective":
                gradient = np.asarray(block).reshape(NU, n)
                transformed["direction"] = (
                    options.u_max * gradient * throttle[None]
                ).ravel()
                transformed["throttle"] = options.u_max * np.sum(
                    gradient * direction.reshape(NU, n), axis=0
                )
            else:
                rows, columns, values = block["coo"]
                rows, columns, values = (
                    np.asarray(rows),
                    np.asarray(columns),
                    np.asarray(values),
                )
                transformed["direction"] = {
                    "coo": [
                        rows,
                        columns,
                        options.u_max * values * throttle[columns % n],
                    ],
                    "shape": block["shape"],
                }
                # Merge contributions of all direction components to the same throttle.
                merged = coo_matrix(
                    (options.u_max * values * direction[columns], (rows, columns % n)),
                    shape=(block["shape"][0], n),
                )
                merged.sum_duplicates()
                transformed["throttle"] = {
                    "coo": [
                        merged.row.astype(np.intc),
                        merged.col.astype(np.intc),
                        merged.data,
                    ],
                    "shape": merged.shape,
                }
        result[name] = transformed
    columns = np.arange(NU * n, dtype=np.intc)
    result["unit_direction"] = {
        "direction": {
            "coo": [columns % n, columns, 2 * direction],
            "shape": (n, NU * n),
        }
    }
    return result


def polar_callbacks(value_callback, derivative_callback, options):
    """Adapt the existing Cartesian callbacks to throttle/direction variables."""

    def values(variables):
        result, fail = value_callback(cartesian_variables(variables, options))
        result["unit_direction"] = np.sum(
            np.asarray(variables["direction"]).reshape(NU, options.n_arcs) ** 2, axis=0
        )
        return result, fail

    def derivatives(variables, funcs):
        jacobian, fail = derivative_callback(
            cartesian_variables(variables, options), funcs
        )
        return polar_jacobian(jacobian, variables, options), fail

    return values, derivatives


def solve_deterministic_optimization(
    options: Options, A: np.ndarray, B: np.ndarray
) -> NominalTrajectory:
    """Deterministic minimum-energy transfer via multiple shooting on the state vectors.

    Each arc is propagated with a batched fixed-step Diffrax solve.
    """

    n_arcs = options.n_arcs

    # A reachable minimum-energy seed avoids the polar coordinate singularity at zero thrust.
    reachability = np.hstack(
        [np.linalg.matrix_power(A, n_arcs - 1 - k) @ B for k in range(n_arcs)]
    )
    target = options.xf_mean - np.linalg.matrix_power(A, n_arcs) @ options.x0_mean
    feedforward_seed = (
        np.linalg.lstsq(reachability, target, rcond=None)[0].reshape(n_arcs, NU).T
    )
    means_seed = np.empty((NX, n_arcs + 1))
    means_seed[:, 0] = options.x0_mean
    for k in range(n_arcs):
        means_seed[:, k + 1] = A @ means_seed[:, k] + B @ feedforward_seed[:, k]

    nominal_seed = {
        "means": means_seed.reshape(-1),
        "feedforward": feedforward_seed.reshape(-1),
    }

    nonlinear_residual_fn = jax.jit(lambda xdict: _nominal_funcs(xdict, options))
    nonlinear_jac_fn = jax.jit(
        lambda xdict: _nonlinear_jacobian_nominal(xdict, options)
    )
    nonlinear_local_structure = _nonlinear_jacobian_sparsity_template_nominal(A, B)
    nonlinear_global_jacobian_structure = _assemble_nonlinear_jacobian_nominal(
        np.broadcast_to(
            nonlinear_local_structure, (n_arcs,) + nonlinear_local_structure.shape
        ),
        options,
    )
    nonlinear_jac_sparsity = get_jacobian_sparsity(nonlinear_global_jacobian_structure)
    linear_jac = _linear_jacobians(options, stochastic=False)

    # Compile both callbacks before entering the optimization
    jax.block_until_ready(nonlinear_residual_fn(nominal_seed))
    jax.block_until_ready(nonlinear_jac_fn(nominal_seed))

    def objconfun(xdict):
        funcs = nonlinear_residual_fn(xdict)
        return {key: np.asarray(value) for key, value in funcs.items()}, False

    cartesian_sparsity = nonlinear_jac_sparsity

    def sens(xdict, funcs):
        nonlinear_jacobian_blocks = nonlinear_jac_fn(xdict)
        jacobian = _assemble_nonlinear_jacobian_nominal(
            np.asarray(nonlinear_jacobian_blocks), options
        )
        return apply_jacobian_sparsity(jacobian, cartesian_sparsity), False

    controls = polar_seed(feedforward_seed, options)
    # Use nonzero structural values so zero-throttle arcs retain every derivative slot.
    structural_controls = {
        "direction": np.ones(NU * n_arcs),
        "throttle": np.ones(n_arcs),
    }
    nonlinear_jac_sparsity = polar_jacobian(
        nonlinear_jac_sparsity, structural_controls, options
    )
    objconfun, sens = polar_callbacks(objconfun, sens, options)
    from pyoptsparse import Optimization
    from pyoptsparse.pySNOPT.pySNOPT import SNOPT

    optProb = Optimization("nominal", objconfun)
    optProb.addVarGroup("means", NX * (n_arcs + 1), value=means_seed.reshape(-1))
    optProb.addVarGroup("direction", NU * n_arcs, value=controls["direction"])
    optProb.addVarGroup(
        "throttle", n_arcs, value=controls["throttle"], lower=0.0, upper=None
    )
    optProb.addConGroup(
        "unit_direction",
        n_arcs,
        lower=1.0,
        upper=1.0,
        wrt=["direction"],
        jac=nonlinear_jac_sparsity["unit_direction"],
    )

    optProb.addConGroup(
        "defects",
        NX * n_arcs,
        lower=0.0,
        upper=0.0,
        wrt=list(nonlinear_jac_sparsity["defects"]),
        jac=nonlinear_jac_sparsity["defects"],
    )
    optProb.addConGroup(
        "boundary_start",
        NX,
        lower=options.x0_mean,
        upper=options.x0_mean,
        linear=True,
        wrt=list(linear_jac["boundary_start"]),
        jac=linear_jac["boundary_start"],
    )
    optProb.addConGroup(
        "boundary_end",
        NX,
        lower=options.xf_mean,
        upper=options.xf_mean,
        linear=True,
        wrt=list(linear_jac["boundary_end"]),
        jac=linear_jac["boundary_end"],
    )
    upper_bounds = np.concatenate(
        [np.full(n_arcs, b) for _, b in options.path_constraints]
    )
    optProb.addConGroup(
        "path",
        len(options.path_constraints) * n_arcs,
        upper=upper_bounds,
        linear=True,
        wrt=list(linear_jac["path"]),
        jac=linear_jac["path"],
    )

    optProb.addObj("objective")

    if options.print_sparsity:
        optProb.printSparsity()

    opt = SNOPT(options=snopt_options(options, "nominal"))
    sol = opt(optProb, sens=sens)

    solved_means = np.asarray(sol.xStar["means"]).reshape(NX, n_arcs + 1)
    solved_feedforward = np.asarray(
        cartesian_variables(sol.xStar, options)["feedforward"]
    ).reshape(NU, n_arcs)
    converged = "optimality conditions satisfied" in optinform_text(sol).lower()

    return NominalTrajectory(
        means=solved_means, feedforward=solved_feedforward, converged=converged
    )


# --------------------------------------------------------------------------- #
#            Stochastic optimization warm start generation
# --------------------------------------------------------------------------- #


@dataclass
class StochasticOptimizationSeed:
    means: np.ndarray
    feedforward: np.ndarray
    gains: np.ndarray
    state_covariances: np.ndarray
    margin: np.ndarray


def build_stochastic_optimization_seed(
    options: Options, A: np.ndarray, B: np.ndarray, arc_fn, nominal: NominalTrajectory
) -> StochasticOptimizationSeed:
    gains_seed = (
        tvlqr_gains(options, A, B)
        if options.warm_start_gains
        else np.zeros((options.n_arcs, NU, NX))
    )

    _, seed_covariances, _ = propagate_statistical_moments(
        arc_fn, nominal.means, nominal.feedforward, gains_seed, options.x0_covariance
    )

    terminal_scaling = options.inverse_terminal_std
    scaled_terminal_state_covariance = (
        np.outer(terminal_scaling, terminal_scaling) * seed_covariances[-1]
    )
    terminal_slack = np.eye(NX) - scaled_terminal_state_covariance
    margin_factor = nearest_pd_cholesky_factor(
        terminal_slack, options.terminal_margin_floor**2
    )

    return StochasticOptimizationSeed(
        means=nominal.means,
        feedforward=nominal.feedforward,
        gains=gains_seed,
        state_covariances=seed_covariances,
        margin=LTRI.pack_numeric(margin_factor),
    )


# --------------------------------------------------------------------------- #
# Covariance multiple-shooting solve
# --------------------------------------------------------------------------- #


@dataclass
class RobustTrajectory:
    means: np.ndarray
    feedforward: np.ndarray
    gains: np.ndarray
    state_covariances: np.ndarray
    control_covariances: np.ndarray
    radius: np.ndarray
    objective: float
    diagnostics: dict[str, float | str] = field(default_factory=dict)


def regularized_control_norm(
    feedforward_k: jnp.ndarray, options: Options
) -> jnp.ndarray:
    return jnp.sqrt(
        jnp.dot(feedforward_k, feedforward_k) + options.control_norm_epsilon**2
    )


def _terminal_constraint_slack(
    terminal_state_covariance: jnp.ndarray, options: Options
) -> jnp.ndarray:
    """``I - diag(1/sigma_f) @ Sigma @ diag(1/sigma_f)``: terminal state covariance headroom."""

    terminal_scaling = jnp.diag(jnp.asarray(options.inverse_terminal_std))
    return jnp.eye(NX) - terminal_scaling @ terminal_state_covariance @ terminal_scaling


def _unpack_stochastic_vars(xdict: dict, options: Options):
    n_arcs = options.n_arcs
    n_tril = len(LTRI)
    means = xdict["means"].reshape(NX, n_arcs + 1)
    feedforward = xdict["feedforward"].reshape(NU, n_arcs)
    gains = xdict["gains"].reshape(n_arcs, NU, NX)
    cholesky_factor_entries = xdict["cholesky_factor"].reshape(n_arcs + 1, n_tril)
    margin_entries = xdict["terminal_margin"]
    return means, feedforward, gains, cholesky_factor_entries, margin_entries


def _stochastic_funcs(
    xdict: dict,
    options: Options,
    arc_fn,
    psi_inv_u: float,
    phi_inv_path: float,
    node_std,
):
    """Evaluate every independent shooting arc in one batched JAX program.

    Each multiple-shooting arc starts from its own
    decision-variable mean and state_covariance. Through `vmap` all arcs, and all sigma points inside each arc, are
    exposed to XLA as nested batch dimensions in one compiled callback.
    """

    n_arcs = options.n_arcs
    n_path = len(options.path_constraints)
    means, feedforward, gains, cholesky_factor_entries, margin_entries = (
        _unpack_stochastic_vars(xdict, options)
    )

    factors = jax.vmap(LTRI.to_matrix)(cholesky_factor_entries) * node_std[:, :, None]
    node_state_covariances = factors @ jnp.swapaxes(factors, 1, 2)

    Q = jnp.asarray(options.Q)
    R = jnp.asarray(options.R)
    path_vectors = options.path_matrix  # (n_path, NX)

    batched_propagation_arc = jax.vmap(arc_fn, in_axes=(1, 0, 1, 0), out_axes=(1, 0, 0))
    predicted_means, predicted_covariances, control_covariances = (
        batched_propagation_arc(
            means[:, :-1], node_state_covariances[:-1], feedforward, gains
        )
    )

    mean_defects = (means[:, 1:] - predicted_means).reshape(-1)
    scaled_predictions = predicted_covariances / (
        node_std[1:, :, None] * node_std[1:, None, :]
    )
    predicted_factors = jax.vmap(
        lambda covariance: LTRI.pack(jnp.linalg.cholesky(covariance))
    )(scaled_predictions)
    covariance_defects = (cholesky_factor_entries[1:] - predicted_factors).reshape(-1)

    control_norms = jax.vmap(
        lambda control: regularized_control_norm(control, options)
    )(feedforward.T)
    control_radii = control_norms + psi_inv_u * jax.vmap(
        lambda covariance: symbolic_psqrt_spectral_radius(
            covariance,
            options.spectral_eigenvalue_smoothing,
        )
    )(control_covariances)

    path_variances = jnp.einsum(
        "pi,kij,pj->pk", path_vectors, node_state_covariances[:-1], path_vectors
    )
    path_values = path_vectors @ means[:, :-1] + phi_inv_path * jnp.sqrt(
        jnp.maximum(path_variances, 0.0)
    )

    covariance_trace_cost = jnp.einsum("ij,kji->", Q, node_state_covariances[:-1])
    control_trace_cost = jnp.einsum("ij,kji->", R, control_covariances)
    total_objective = options.dt * (
        jnp.sum(control_norms) + covariance_trace_cost + control_trace_cost
    )

    terminal_slack = _terminal_constraint_slack(node_state_covariances[n_arcs], options)
    margin_factor = LTRI.to_matrix(margin_entries)
    terminal_residual = LTRI.pack(terminal_slack - margin_factor @ margin_factor.T)

    funcs = {
        "initial_covariance": cholesky_factor_entries[0]
        - LTRI.pack(
            jnp.linalg.cholesky(
                jnp.asarray(options.x0_covariance) / jnp.outer(node_std[0], node_std[0])
            )
        ),
        "mean_defects": mean_defects,
        "state_covariance_defects": covariance_defects,
        "control_chance": control_radii,
        "terminal_residual": terminal_residual,
        "objective": total_objective,
    }
    for idx in range(n_path):
        funcs[f"path_{idx}"] = path_values[idx]
    return funcs


def _nonlinear_arc_outputs_stochastic(
    local_input: jnp.ndarray,
    options: Options,
    arc_fn,
    psi_inv_u: float,
    phi_inv_path: float,
    start_std,
    end_std,
) -> jnp.ndarray:
    """Nonlinear objective and defect quantities owned by one arc."""

    n_tril = len(LTRI)
    mean = local_input[:NX]
    feedforward = local_input[NX : NX + NU]
    gain = local_input[NX + NU : NX + NU + NU * NX].reshape(NU, NX)
    cholesky_factor_entries = local_input[-n_tril:]
    cholesky_factor = start_std[:, None] * LTRI.to_matrix(cholesky_factor_entries)
    state_covariance = cholesky_factor @ cholesky_factor.T

    mean_next, covariance_next, control_covariance = arc_fn(
        mean, state_covariance, feedforward, gain
    )
    control_norm = regularized_control_norm(feedforward, options)
    control_chance = control_norm + psi_inv_u * symbolic_psqrt_spectral_radius(
        control_covariance,
        options.spectral_eigenvalue_smoothing,
    )

    path_vectors = options.path_matrix
    path_variances = jnp.einsum(
        "pi,ij,pj->p", path_vectors, state_covariance, path_vectors
    )
    path_values = path_vectors @ mean + phi_inv_path * jnp.sqrt(
        jnp.maximum(path_variances, 0.0)
    )

    local_objective = options.dt * (
        control_norm
        + jnp.trace(jnp.asarray(options.Q) @ state_covariance)
        + jnp.trace(jnp.asarray(options.R) @ control_covariance)
    )
    return jnp.concatenate(
        [
            mean_next,
            LTRI.pack(
                jnp.linalg.cholesky(covariance_next / jnp.outer(end_std, end_std))
            ),
            jnp.atleast_1d(control_chance),
            path_values,
            jnp.atleast_1d(local_objective),
        ]
    )


def get_terminal_residual(
    terminal_input: jnp.ndarray, options: Options, terminal_std
) -> jnp.ndarray:
    n_tril = len(LTRI)
    cholesky_factor = terminal_std[:, None] * LTRI.to_matrix(terminal_input[:n_tril])
    terminal_state_covariance = cholesky_factor @ cholesky_factor.T
    margin_factor = LTRI.to_matrix(terminal_input[n_tril:])
    terminal_slack = _terminal_constraint_slack(terminal_state_covariance, options)
    return LTRI.pack(terminal_slack - margin_factor @ margin_factor.T)


def _nonlinear_jacobian_stochastic(
    xdict: dict,
    options: Options,
    arc_fn,
    psi_inv_u: float,
    phi_inv_path: float,
    node_std,
):
    """AD-differentiate small nonlinear arc maps, then batch them over the trajectory."""

    means, feedforward, gains, cholesky_factor_entries, margin_entries = (
        _unpack_stochastic_vars(xdict, options)
    )
    local_inputs = jnp.concatenate(
        [
            means[:, :-1].T,
            feedforward.T,
            gains.reshape(options.n_arcs, -1),
            cholesky_factor_entries[:-1],
        ],
        axis=1,
    )
    local_jacobian = jax.jacrev(
        lambda values, start_std, end_std: _nonlinear_arc_outputs_stochastic(
            values, options, arc_fn, psi_inv_u, phi_inv_path, start_std, end_std
        )
    )
    nonlinear_jacobian_blocks = jax.vmap(local_jacobian)(
        local_inputs, node_std[:-1], node_std[1:]
    )
    state_covariance_jacobians = jnp.broadcast_to(
        jnp.eye(len(LTRI)), (options.n_arcs + 1, len(LTRI), len(LTRI))
    )
    terminal_input = jnp.concatenate([cholesky_factor_entries[-1], margin_entries])
    terminal_jacobian = jax.jacrev(
        lambda values: get_terminal_residual(values, options, node_std[-1])
    )(terminal_input)
    return nonlinear_jacobian_blocks, state_covariance_jacobians, terminal_jacobian


def _assemble_nonlinear_global_jacobian_stochastic(
    nonlinear_jacobian_blocks: np.ndarray,
    state_covariance_jacobians: np.ndarray,
    terminal_jacobian: np.ndarray,
    options: Options,
) -> dict:
    """Assemble the nonlinear per-arc AD Jacobian blocks in the global pyOptSparse ordering."""

    n_arcs = options.n_arcs
    n_path = len(options.path_constraints)
    n_tril = len(LTRI)
    variable_sizes = {
        "means": NX * (n_arcs + 1),
        "feedforward": NU * n_arcs,
        "gains": n_arcs * NU * NX,
        "cholesky_factor": (n_arcs + 1) * n_tril,
        "terminal_margin": n_tril,
    }

    def allocate_blocks(rows: int, variables: tuple[str, ...]) -> dict[str, np.ndarray]:
        return {
            variable: np.zeros((rows, variable_sizes[variable]))
            for variable in variables
        }

    jacobian = {
        "initial_covariance": allocate_blocks(n_tril, ("cholesky_factor",)),
        "mean_defects": allocate_blocks(
            NX * n_arcs, ("means", "feedforward", "gains", "cholesky_factor")
        ),
        "state_covariance_defects": allocate_blocks(
            n_tril * n_arcs,
            ("means", "feedforward", "gains", "cholesky_factor"),
        ),
        "control_chance": allocate_blocks(
            n_arcs, ("means", "feedforward", "gains", "cholesky_factor")
        ),
        "terminal_residual": allocate_blocks(
            n_tril, ("cholesky_factor", "terminal_margin")
        ),
        "objective": {
            variable: np.zeros(size) for variable, size in variable_sizes.items()
        },
    }
    for path_index in range(n_path):
        jacobian[f"path_{path_index}"] = allocate_blocks(
            n_arcs, ("means", "cholesky_factor")
        )

    state_indices = np.arange(NX)
    control_indices = np.arange(NU)
    gain_indices = np.arange(NU * NX)
    tril_indices = np.arange(n_tril)
    jacobian["initial_covariance"]["cholesky_factor"][:, :n_tril] = (
        state_covariance_jacobians[0]
    )

    input_mean = slice(0, NX)
    input_feedforward = slice(NX, NX + NU)
    input_gain = slice(NX + NU, NX + NU + NU * NX)
    input_cholesky = slice(NX + NU + NU * NX, NX + NU + NU * NX + n_tril)
    output_mean = slice(0, NX)
    output_covariance = slice(NX, NX + n_tril)
    output_control_chance = NX + n_tril
    output_path_start = output_control_chance + 1
    output_objective = output_path_start + n_path

    for arc_index in range(n_arcs):
        local_jac = nonlinear_jacobian_blocks[arc_index]
        mean_rows = state_indices * n_arcs + arc_index
        covariance_rows = arc_index * n_tril + tril_indices
        mean_columns = state_indices * (n_arcs + 1) + arc_index
        next_mean_columns = mean_columns + 1
        feedforward_columns = control_indices * n_arcs + arc_index
        gain_columns = arc_index * NU * NX + gain_indices
        cholesky_factor_columns = arc_index * n_tril + tril_indices
        next_cholesky_factor_columns = cholesky_factor_columns + n_tril

        jacobian["mean_defects"]["means"][np.ix_(mean_rows, mean_columns)] = -local_jac[
            output_mean, input_mean
        ]
        jacobian["mean_defects"]["means"][np.ix_(mean_rows, next_mean_columns)] += (
            np.eye(NX)
        )
        jacobian["mean_defects"]["feedforward"][
            np.ix_(mean_rows, feedforward_columns)
        ] = -local_jac[output_mean, input_feedforward]
        jacobian["mean_defects"]["gains"][np.ix_(mean_rows, gain_columns)] = -local_jac[
            output_mean, input_gain
        ]
        jacobian["mean_defects"]["cholesky_factor"][
            np.ix_(mean_rows, cholesky_factor_columns)
        ] = -local_jac[output_mean, input_cholesky]

        jacobian["state_covariance_defects"]["means"][
            np.ix_(covariance_rows, mean_columns)
        ] = -local_jac[output_covariance, input_mean]
        jacobian["state_covariance_defects"]["feedforward"][
            np.ix_(covariance_rows, feedforward_columns)
        ] = -local_jac[output_covariance, input_feedforward]
        jacobian["state_covariance_defects"]["gains"][
            np.ix_(covariance_rows, gain_columns)
        ] = -local_jac[output_covariance, input_gain]
        jacobian["state_covariance_defects"]["cholesky_factor"][
            np.ix_(covariance_rows, cholesky_factor_columns)
        ] = -local_jac[output_covariance, input_cholesky]
        jacobian["state_covariance_defects"]["cholesky_factor"][
            np.ix_(covariance_rows, next_cholesky_factor_columns)
        ] += state_covariance_jacobians[arc_index + 1]

        control_row = np.array([arc_index])
        for variable, columns, input_slice in (
            ("means", mean_columns, input_mean),
            ("feedforward", feedforward_columns, input_feedforward),
            ("gains", gain_columns, input_gain),
            ("cholesky_factor", cholesky_factor_columns, input_cholesky),
        ):
            jacobian["control_chance"][variable][np.ix_(control_row, columns)] = (
                local_jac[output_control_chance, input_slice][None, :]
            )

        for path_index in range(n_path):
            path_name = f"path_{path_index}"
            output_index = output_path_start + path_index
            jacobian[path_name]["means"][arc_index, mean_columns] = local_jac[
                output_index, input_mean
            ]
            jacobian[path_name]["cholesky_factor"][
                arc_index, cholesky_factor_columns
            ] = local_jac[output_index, input_cholesky]

        for variable, columns, input_slice in (
            ("means", mean_columns, input_mean),
            ("feedforward", feedforward_columns, input_feedforward),
            ("gains", gain_columns, input_gain),
            ("cholesky_factor", cholesky_factor_columns, input_cholesky),
        ):
            jacobian["objective"][variable][columns] = local_jac[
                output_objective, input_slice
            ]

    jacobian["terminal_residual"]["cholesky_factor"][:, -n_tril:] = terminal_jacobian[
        :, :n_tril
    ]
    jacobian["terminal_residual"]["terminal_margin"][:, :] = terminal_jacobian[
        :, n_tril:
    ]
    return jacobian


def solve_stochastic_optimization(
    options: Options,
    A: np.ndarray,
    B: np.ndarray,
    Lc: np.ndarray,
    nominal_traj: NominalTrajectory,
) -> RobustTrajectory:
    arc_fn = build_propagation_arc_map(options, Lc)
    psi_inv_u = psi_inverse(NU, 1.0 - options.control_confidence)
    phi_inv_path = float(norm.ppf(options.path_confidence))
    n_arcs = options.n_arcs
    n_tril = len(LTRI)

    seed = build_stochastic_optimization_seed(options, A, B, arc_fn, nominal_traj)

    # Same diagonal seed scaling as the native Step 3/4 models.
    node_std = np.sqrt(
        np.maximum(np.diagonal(seed.state_covariances, axis1=1, axis2=2), 1e-12)
    )
    cholesky_factor_seed = np.stack(
        [
            LTRI.pack_numeric(
                np.linalg.cholesky(
                    seed.state_covariances[k] / np.outer(node_std[k], node_std[k])
                )
            )
            for k in range(n_arcs + 1)
        ]
    )

    nonlinear_residual_fn = jax.jit(
        lambda xdict: _stochastic_funcs(
            xdict, options, arc_fn, psi_inv_u, phi_inv_path, jnp.asarray(node_std)
        )
    )
    nonlinear_jac_fn = jax.jit(
        lambda xdict: _nonlinear_jacobian_stochastic(
            xdict, options, arc_fn, psi_inv_u, phi_inv_path, jnp.asarray(node_std)
        )
    )
    n_local_inputs = NX + NU + NU * NX + n_tril
    n_local_outputs = NX + n_tril + 1 + len(options.path_constraints) + 1
    nonlinear_global_jacobian_structure = (
        _assemble_nonlinear_global_jacobian_stochastic(
            np.ones((n_arcs, n_local_outputs, n_local_inputs)),
            np.broadcast_to(np.eye(n_tril), (n_arcs + 1, n_tril, n_tril)),
            np.ones((n_tril, 2 * n_tril)),
            options,
        )
    )
    nonlinear_jac_sparsity = get_jacobian_sparsity(nonlinear_global_jacobian_structure)
    linear_jac = _linear_jacobians(options, stochastic=True)

    def objconfun(xdict):
        funcs = nonlinear_residual_fn(xdict)
        return {key: np.asarray(value) for key, value in funcs.items()}, False

    cartesian_sparsity = nonlinear_jac_sparsity

    def sens(xdict, funcs):
        nonlinear_jacobian_blocks, state_covariance_jacobians, terminal_jacobian = (
            nonlinear_jac_fn(xdict)
        )
        jacobian = _assemble_nonlinear_global_jacobian_stochastic(
            np.asarray(nonlinear_jacobian_blocks),
            np.asarray(state_covariance_jacobians),
            np.asarray(terminal_jacobian),
            options,
        )
        return apply_jacobian_sparsity(jacobian, cartesian_sparsity), False

    controls = polar_seed(seed.feedforward, options)
    # Use nonzero structural values so zero-throttle arcs retain every derivative slot.
    structural_controls = {
        "direction": np.ones(NU * n_arcs),
        "throttle": np.ones(n_arcs),
    }
    nonlinear_jac_sparsity = polar_jacobian(
        nonlinear_jac_sparsity, structural_controls, options
    )
    objconfun, sens = polar_callbacks(objconfun, sens, options)
    from pyoptsparse import Optimization
    from pyoptsparse.pySNOPT.pySNOPT import SNOPT

    optProb = Optimization("stochastic", objconfun)
    optProb.addVarGroup("means", NX * (n_arcs + 1), value=seed.means.reshape(-1))
    optProb.addVarGroup("direction", NU * n_arcs, value=controls["direction"])
    optProb.addVarGroup(
        "throttle", n_arcs, value=controls["throttle"], lower=0.0, upper=1.0
    )
    optProb.addConGroup(
        "unit_direction",
        n_arcs,
        lower=1.0,
        upper=1.0,
        wrt=["direction"],
        jac=nonlinear_jac_sparsity["unit_direction"],
    )
    optProb.addVarGroup("gains", n_arcs * NU * NX, value=seed.gains.reshape(-1))
    optProb.addVarGroup(
        "cholesky_factor",
        (n_arcs + 1) * n_tril,
        value=cholesky_factor_seed.reshape(-1),
    )
    optProb.addVarGroup("terminal_margin", n_tril, value=seed.margin)
    optProb.addConGroup(
        "boundary_start",
        NX,
        lower=options.x0_mean,
        upper=options.x0_mean,
        linear=True,
        wrt=list(linear_jac["boundary_start"]),
        jac=linear_jac["boundary_start"],
    )
    optProb.addConGroup(
        "boundary_end",
        NX,
        lower=options.xf_mean,
        upper=options.xf_mean,
        linear=True,
        wrt=list(linear_jac["boundary_end"]),
        jac=linear_jac["boundary_end"],
    )
    optProb.addConGroup(
        "initial_covariance",
        n_tril,
        lower=0.0,
        upper=0.0,
        wrt=list(nonlinear_jac_sparsity["initial_covariance"]),
        jac=nonlinear_jac_sparsity["initial_covariance"],
    )
    optProb.addConGroup(
        "mean_defects",
        NX * n_arcs,
        lower=0.0,
        upper=0.0,
        wrt=list(nonlinear_jac_sparsity["mean_defects"]),
        jac=nonlinear_jac_sparsity["mean_defects"],
    )
    optProb.addConGroup(
        "state_covariance_defects",
        n_tril * n_arcs,
        lower=0.0,
        upper=0.0,
        wrt=list(nonlinear_jac_sparsity["state_covariance_defects"]),
        jac=nonlinear_jac_sparsity["state_covariance_defects"],
    )
    optProb.addConGroup(
        "control_chance",
        n_arcs,
        upper=options.u_max,
        wrt=list(nonlinear_jac_sparsity["control_chance"]),
        jac=nonlinear_jac_sparsity["control_chance"],
    )
    for idx, (_, b_value) in enumerate(options.path_constraints):
        name = f"path_{idx}"
        optProb.addConGroup(
            name,
            n_arcs,
            upper=b_value,
            wrt=list(nonlinear_jac_sparsity[name]),
            jac=nonlinear_jac_sparsity[name],
        )
    optProb.addConGroup(
        "terminal_residual",
        n_tril,
        lower=0.0,
        upper=0.0,
        wrt=list(nonlinear_jac_sparsity["terminal_residual"]),
        jac=nonlinear_jac_sparsity["terminal_residual"],
    )
    optProb.addConGroup(
        "margin_diagonal",
        NX,
        lower=options.terminal_margin_floor,
        linear=True,
        wrt=list(linear_jac["margin_diagonal"]),
        jac=linear_jac["margin_diagonal"],
    )
    optProb.addConGroup(
        "cholesky_diagonal",
        (n_arcs + 1) * NX,
        lower=0.0,
        linear=True,
        wrt=list(linear_jac["cholesky_diagonal"]),
        jac=linear_jac["cholesky_diagonal"],
    )

    optProb.addObj("objective")

    if options.print_sparsity:
        optProb.printSparsity()

    opt = SNOPT(options=snopt_options(options, "stochastic"))
    sol = opt(optProb, sens=sens)
    # Polar coast arcs can exhaust an SQP Hessian approximation near the solution.
    # Restart once from a feasible accuracy-limit candidate, with identical settings.
    if int(sol.optInform["value"]) in (3, 41):
        candidate_values = objconfun(sol.xStar)[0]
        if solver_diagnostics(sol, optProb, sol.xStar, candidate_values, options)[
            "feasible"
        ]:
            print(
                "Restarting feasible SNOPT accuracy-limit candidate with unchanged tolerances.",
                flush=True,
            )
            optProb.setDVs(sol.xStar)
            sol = opt(optProb, sens=sens)

    solved_means = np.asarray(sol.xStar["means"]).reshape(NX, n_arcs + 1)
    solved_feedforward = np.asarray(
        cartesian_variables(sol.xStar, options)["feedforward"]
    ).reshape(NU, n_arcs)
    solved_gains = np.asarray(sol.xStar["gains"]).reshape(n_arcs, NU, NX)
    solved_cholesky = np.asarray(sol.xStar["cholesky_factor"]).reshape(
        n_arcs + 1, n_tril
    )
    solved_state_covariances = np.stack(
        [
            (node_std[k, :, None] * LTRI.to_matrix_numeric(solved_cholesky[k]))
            @ (node_std[k, :, None] * LTRI.to_matrix_numeric(solved_cholesky[k])).T
            for k in range(n_arcs + 1)
        ]
    )

    control_covariances, mean_defect, state_covariance_defect = (
        evaluate_node_matching_defects(
            arc_fn,
            solved_means,
            solved_feedforward,
            solved_gains,
            solved_state_covariances,
        )
    )
    solved_radius, smoothing_metrics = get_smoothing_diagnostics(
        control_covariances, options
    )

    final_funcs = objconfun(sol.xStar)[0]
    solver_metrics = solver_diagnostics(sol, optProb, sol.xStar, final_funcs, options)

    return RobustTrajectory(
        means=solved_means,
        feedforward=solved_feedforward,
        gains=solved_gains,
        state_covariances=solved_state_covariances,
        control_covariances=control_covariances,
        radius=solved_radius,
        objective=float(final_funcs["objective"]),
        diagnostics={
            **solver_metrics,
            "covariance_defect_form": "seed-scaled square-root",
            "covariance_scale_min_variance": float(np.min(node_std**2)),
            "covariance_scale_max_variance": float(np.max(node_std**2)),
            "matching_defect": mean_defect,
            "covariance_matching_defect": state_covariance_defect,
            **smoothing_metrics,
        },
    )


# --------------------------------------------------------------------------- #
#                                Verification
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
#                               Plotting
# --------------------------------------------------------------------------- #


def print_summary(options: Options, solution: RobustTrajectory) -> None:
    diagnostics = solution.diagnostics

    feasibility_text = "true" if diagnostics["feasible"] else "no"

    rows = [
        ("solver status", diagnostics["solver_status"]),
        ("feasibility", feasibility_text),
        ("max constraint viol.", f"{diagnostics['max_constraint_violation']:.3e}"),
        ("max mean matching defect", f"{diagnostics['matching_defect']:.3e}"),
        (
            "max state cov. matching defect",
            f"{diagnostics['covariance_matching_defect']:.3e}",
        ),
        (
            "max psqrt spectr. radius smoothing bias",
            f"{diagnostics['max_psqrt_spectral_radius_smoothing_bias']:.3e}",
        ),
        (
            "max chance con. margin bias",
            f"{diagnostics['max_control_margin_smoothing_bias']:.3e}",
        ),
        (
            "bias / peak exact psqrt spectr. radius",
            f"{diagnostics['relative_psqrt_spectral_radius_smoothing_bias']:.3e}",
        ),
        ("objective function", f"{solution.objective:.6f}"),
        ("terminal mean state", solution.means[:, -1]),
        (
            "terminal state covariance diag",
            np.diag(solution.state_covariances[-1]),
        ),
        (
            "target state covariance diag",
            np.diag(options.xf_covariance),
        ),
    ]

    label_width = max(len(name) for name, _ in rows)
    separator = "=" * 88

    lines = [
        "",
        separator,
        f"{'Optimization Summary':^88}",
        separator,
    ]

    for name, value in rows:
        lines.append(f"{name:<{label_width}}  │  {value}")

    lines.append(separator)

    print("\n".join(lines), flush=True)


# --------------------------------------------------------------------------- #
#                                    Driver
# --------------------------------------------------------------------------- #


def main() -> None:
    from verification.double_integrator import (
        validate_shark_process_covariance,
        monte_carlo_rollout,
    )
    from plotter.double_integrator import plot_trajectory, plot_control_profiles

    options = Options()
    A, B, G = discrete_time_matrices(options)
    Ac, _ = continuous_time_matrices()
    Qc, Lc = calibrate_continuous_diffusion(Ac, G @ G.T, options.dt)

    if (
        np.max(np.abs(accumulated_process_covariance(Ac, Qc, options.dt) - G @ G.T))
        > 1e-15
    ):
        raise ValueError(
            "Continuous diffusion state_covariance calibration error exceeds 1e-15."
        )

    shark_covariance_error = validate_shark_process_covariance(Ac, Qc, Lc, options.dt)
    if shark_covariance_error > 1e-13:
        raise ValueError(
            "ShARK process covariance error exceeds 1e-13: "
            f"{shark_covariance_error:.3e}."
        )
    print(
        "One-step ShARK process covariance validation: "
        f"max absolute error = {shark_covariance_error:.3e}",
        flush=True,
    )

    print(
        "\nSolving the deterministic trajectory optimization ... ",
        flush=True,
    )

    nominal_sol = solve_deterministic_optimization(options, A, B)
    if not nominal_sol.converged:
        print("Deterministic nominal solve did not converge; aborting. \n", flush=True)
        return

    print(
        "\nSolving the robust trajectory optimization ... \n",
        flush=True,
    )

    stochastic_sol = solve_stochastic_optimization(options, A, B, Lc, nominal_sol)
    print_summary(options, stochastic_sol)

    closed_loop_monte_carlo = monte_carlo_rollout(
        options, Lc, stochastic_sol, closed_loop=True
    )
    open_loop_monte_carlo = monte_carlo_rollout(
        options, Lc, stochastic_sol, closed_loop=False
    )

    plot_trajectory(
        options,
        stochastic_sol.means,
        stochastic_sol.state_covariances,
        closed_loop_monte_carlo,
        OUTPUT_DIR / "closed_loop_traj.png",
        feedforward=stochastic_sol.feedforward,
    )
    plot_trajectory(
        options,
        stochastic_sol.means,
        open_loop_monte_carlo["state_covariances"],
        open_loop_monte_carlo,
        OUTPUT_DIR / "open_loop_traj.png",
    )
    plot_control_profiles(
        options,
        stochastic_sol,
        OUTPUT_DIR / "control_magnitude.png",
        OUTPUT_DIR / "control_covariance.png",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_DIR / "solutions.npz",
        means=stochastic_sol.means,
        state_covariances=stochastic_sol.state_covariances,
        gains=stochastic_sol.gains,
        feedforward=stochastic_sol.feedforward,
    )


if __name__ == "__main__":
    main()
