"""
Chance-constrained covariance steering for Earth-Moon CR3BP low-thrust transfers test cases.

The mean trajectory is transcribed by multiple shooting (the node means are decision variables,
tied by matching conditions) while the covariance is single-shot forward from Sigma_0, so that the
propagated covariances are positive semi-definite by construction.

Uncertainty is propagated by the Unscented Transform with
kappa = 0 and a lower-triangular Cholesky factor. The sigma points are integrated
with a fixed-step RK4 built directly as a CasADi expression, so the optimiser sees
through the integration and supplies exact first and second derivatives.

Configuration: one Gaussian component (no GMM split, hence a single control
policy), navigation error (R_bar, H = I), deterministic dynamics (Q_k = 0).
The initial guess is the fuel-optimal solution restricted to its knot
points, thus the interval durations are not uniform and the objective sum is weighted by them.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import casadi
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

import deterministic_cr3bp
from deterministic_cr3bp import CASE_REGISTRY, TestCase
from plotter import Plotter 

NX = 7                       # augmented state dimension
NU = 3                       # control dimension
NP = 6                       # primed (position-velocity) state dimension
N_SIGMA = 2 * NX + 1         # unscented sigma points

REFERENCE_DIR = Path("output/cr3bp_fuel_optimal")
OUTPUT_DIR = Path("output/cr3bp_covariance_steering")

# Initial position and velocity standard deviations per test case [-].
# Order: (sigma_x, sigma_y, sigma_z, sigma_xdot, sigma_ydot, sigma_zdot).
INITIAL_STATE_STD_ND: dict[
    str, tuple[float, float, float, float, float, float]
] = {
    "halo_l2_to_halo_l1": (
        1e-6,  # sigma_x
        1e-6,  # sigma_y
        1e-6,  # sigma_z
        1e-6,  # sigma_xdot
        1e-6,  # sigma_ydot
        1e-6,  # sigma_zdot
    ),
    "lyapunov_l1_to_l2": (
        1e-6,  # sigma_x
        1e-6,  # sigma_y
        1e-6,  # sigma_z
        1e-6,  # sigma_xdot
        1e-6,  # sigma_ydot
        1e-6,  # sigma_zdot
    ),
}
# Uniform gain used when the Riccati warm start is disabled.
COLD_START_GAIN = 1e-3

INITIAL_MASS_STD = 0.0      # [kg]
MASS_SCALE = 1e-3        # [kg]

@dataclass(frozen=True)
class Options:
    """Solver and transcription settings."""

    # Keep every `mesh_stride`-th knot point of the DIRTRAN mesh. 
    mesh_stride: int = 1
    integrator_substeps: int = 4

    violation_parameter: float = 0.05   
    # Terminal covariance reduction factors relative to the corresponding
    # initial covariance blocks. They may be set independently.
    position_covariance_reduction: float = 1e4
    velocity_covariance_reduction: float = 1e3
    scaling_parameter: float = 0.0
    # Navigation (measurement) error covariance as a fraction of the initial state
    # covariance: R_bar = navigation_error_ratio * Sigma_0. Zero disables it and
    # recovers the 2*n_x+1 sigma-point scheme.
    #
    # With R_bar != 0 the sigma points are generated on the augmented vector
    # z = [x ; eta] with covariance blkdiag(Sigma, R_bar), giving 2*(2 n_x)+1 = 29
    # points instead of 15. 
    navigation_error_ratio: float = 1e-4

    # Floors the covariance at jitter / n_x, which must stay far below the
    # terminal target while covering the negative eigenvalues of order 1e-15 that
    # round-off leaves behind once the steering contracts a direction.
    cholesky_jitter: float = 1e-11
    spectral_radius_floor: float = 1e-9

    # Floor on diag(G) in the terminal covariance constraint (see solve_rocp):
    # I_6 - Dt^-1/2 P'_N Dt^-1/2 = G G^T, G lower-triangular. Any PSD left-hand
    # side admits a factor with strictly positive pivots; it
    # exists purely to keep G -> G G^T non-degenerate near diag(G) = 0, where the
    # constraint Jacobian would otherwise lose rank.
    terminal_margin_floor: float = 1e-3


    # Target contribution of the cl control effort margin up to a
    # confidence level p after seeding
    cl_control_effort_seed_margin: float = 0.2        

    # Warm-start the feedback gains from the Bryson-rule Riccati recursion. With
    # this disabled the gains start at zero, the covariance grows open-loop, and
    # the terminal constraint begins many orders of magnitude violated. Starting
    # with a zero gain is still recoverable, but consumes a lot of additional iterations.
    warm_start_gains: bool = True

    solver: str = "snopt"

    # SNOPT settings
    major_max_iter: int = 5000
    minor_max_iter: int = 100 * major_max_iter
    major_optimality_tol: float = 5e-7
    major_feasibility_tol: float = 1e-9
    minor_feasibility_tol: float = 1e-9
    expand_graph: bool = False
    print_level: int = 1

    monte_carlo_samples: int = 10000
    monte_carlo_seed: int = 42


def psi_inverse(dimension: int, beta: float) -> float:
    """Psi_d^-1(beta) = sqrt(Phi_d^-1(1 - beta)) with Phi_d the chi-squared CDF."""

    return float(np.sqrt(chi2.ppf(1.0 - beta, dimension)))


def control_covariance_weight(options: Options) -> float:
    """Unscented trasform weight."""

    return 1.0 / (2.0 * (augmented_dimension(options) + options.scaling_parameter))


def augmented_dimension(options: Options) -> int:
    """Dimension the sigma points are generated on.

    `NX` without navigation error; `2*NX` with it, the extra block being the
    measurement noise, so that state and noise are sampled jointly.
    """

    return 2 * NX if options.navigation_error_ratio > 0.0 else NX


def navigation_covariance(options: Options, normalization: "Normalization") -> np.ndarray:
    """Navigation covariance in normalised coordinates: D0^-1 R_bar D0^-1 = ratio * P_0."""

    return options.navigation_error_ratio * normalization.initial_covariance


def unscented_weights(kappa: float, dimension: int) -> np.ndarray:
    """Sigma-point weights c_j of Eq. 5.82-5.83; c_0 vanishes for kappa = 0."""

    weights = np.full(2 * dimension + 1, 1.0 / (2.0 * (dimension + kappa)))
    weights[0] = kappa / (dimension + kappa)
    return weights


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Normalization:
    """Congruence scaling of the covariance by fixed deviations from the reference trajectory.

    D0 is the diagonal matrix of the initial standard deviations of the augmented state, 
    whilst D0' is the diagonal matrix of the initial standard deviations of the primed state.
    The implememation carries P = D0^-1 Sigma D0^-1 rather than Sigma (where
    Sigma represents the original covariance matrix), so the terminal
    requirement has independently configurable position and velocity targets
    with O(1) entries. Gains are carried as
    K_tilde = K D0' / T_max and the open-loop control as S_tilde = S / T_max, so
    every decision variable is approximately O(1).

    `scale` contains the strictly positive normalization scales used to form D0.
    Every entry must be nonzero so that D0 is invertible.
    `initial_std` holds the actual initial dispersions, which may be zero on a
    channel. The two coincide except on the mass.
    """

    scale: np.ndarray
    initial_std: np.ndarray

    @property
    def matrix(self) -> np.ndarray:
        return np.diag(self.scale)

    @property
    def inverse_matrix(self) -> np.ndarray:
        return np.diag(1.0 / self.scale)

    @property
    def initial_covariance(self) -> np.ndarray:
        """P_0, the normalised initial covariance."""

        return np.diag((self.initial_std / self.scale) ** 2)


def initial_std(case: TestCase) -> np.ndarray:
    """Initial 1-sigma values of the augmented state [-]."""

    position_velocity_std = np.asarray(
        INITIAL_STATE_STD_ND[case.test_case_id], dtype=float
    )
    if position_velocity_std.shape != (NP,):
        raise ValueError(
            f"Initial standard deviations for '{case.test_case_id}' must contain "
            f"exactly {NP} values ordered as (x, y, z, xdot, ydot, zdot)."
        )
    if not np.all(np.isfinite(position_velocity_std)) or np.any(
        position_velocity_std <= 0.0
    ):
        raise ValueError(
            "Initial position and velocity standard deviations must be finite "
            "and strictly positive so the normalization matrix is invertible."
        )
    mass_std = INITIAL_MASS_STD / case.m0_wet
    return np.concatenate((position_velocity_std, [mass_std]))


def build_normalization(case: TestCase) -> Normalization:
    initial_sigma = initial_std(case)
    scale = initial_sigma.copy()
    scale[6] = MASS_SCALE / case.m0_wet
    return Normalization(scale=scale, initial_std=initial_sigma)


# --------------------------------------------------------------------------- #
# Dynamics and the integration arc map
# --------------------------------------------------------------------------- #


class Dynamics:
    """Batched CR3BP dynamics and the integration arc map with its exact Jacobian.

    The state and control Jacobians of the EOM are generated once with
    CasADi and evaluated numerically inside the blackbox; they are never exposed to
    the optimiser.
    """

    def __init__(self, case: TestCase) -> None:
        state = casadi.SX.sym("state", NX)
        control = casadi.SX.sym("control", NU)
        eom = deterministic_cr3bp.eom(case, state, control)
        self._augm_state_derivatives = casadi.Function("augm_state_derivatives", [state, control], [eom])
        self._full_derivatives = casadi.Function(
            "full_derivatives",
            [state, control],
            [
                eom,
                casadi.reshape(casadi.jacobian(eom, state), NX * NX, 1),
                casadi.reshape(casadi.jacobian(eom, control), NX * NU, 1),
            ],
        )
        self._augm_state_derivatives_maps: dict[int, casadi.Function] = {}
        self._full_derivatives_maps: dict[int, casadi.Function] = {}

    def _mapped(self, cache: dict[int, casadi.Function], base: casadi.Function, count: int) -> casadi.Function:
        if count not in cache:
            cache[count] = base.map(count)
        return cache[count]

    def augm_state_derivatives(self, states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        mapped = self._mapped(self._augm_state_derivatives_maps, self._augm_state_derivatives, states.shape[1])
        return np.asarray(mapped(states, controls).full(), dtype=float)

    def augm_state_derivatives_and_jacobians(
        self, states: np.ndarray, controls: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        count = states.shape[1]
        mapped = self._mapped(self._full_derivatives_maps, self._full_derivatives, count)
        dynamics, jacobian_state, jacobian_control = mapped(states, controls)
        state_blocks = np.asarray(jacobian_state.full(), dtype=float).reshape(NX, NX, count, order="F")
        control_blocks = np.asarray(jacobian_control.full(), dtype=float).reshape(NX, NU, count, order="F")
        return (
            np.asarray(dynamics.full(), dtype=float),
            np.moveaxis(state_blocks, 2, 0),
            np.moveaxis(control_blocks, 2, 0),
        )

    def propagate(
        self,
        states,
        controls,
        step: float,
        substeps: int,
        with_jacobian: bool = False,
    ):
        """Fixed-step RK4 over one arc for symbolic or numerical inputs.

        With MX/SX inputs, the RK4 map is built natively from a compact mapped SX
        dynamics function. CasADi can therefore differentiate the returned map to
        any required order, including the exact Hessian used by IPOPT.

        With NumPy inputs, the same RK4 stages are evaluated numerically. If
        ``with_jacobian`` is true, the exact partial derivatives of the discrete
        RK4 map are accumulated through the stages and substeps. The control is
        held constant over the complete arc in both modes.

        Returns ``(propagated, state_sensitivity, control_sensitivity)``. The two
        sensitivities are ``None`` unless requested in numerical mode.
        """

        symbolic = isinstance(states, (casadi.MX, casadi.SX)) or isinstance(
            controls, (casadi.MX, casadi.SX)
        )
        if symbolic and with_jacobian:
            raise ValueError(
                "with_jacobian is only used for numerical propagation; "
                "differentiate the returned CasADi expression in symbolic mode."
            )

        count = states.shape[1]
        substep = step / substeps
        if symbolic:
            mapped = self._mapped(
                self._augm_state_derivatives_maps,
                self._augm_state_derivatives,
                count,
            )
            propagated = states
            state_sensitivity = None
            control_sensitivity = None
        else:
            propagated = np.array(states, dtype=float, copy=True)
            controls = np.asarray(controls, dtype=float)
            identity = np.eye(NX)
            state_sensitivity = (
                np.broadcast_to(identity, (count, NX, NX)).copy()
                if with_jacobian
                else None
            )
            control_sensitivity = (
                np.zeros((count, NX, NU)) if with_jacobian else None
            )

        for _ in range(substeps):
            if with_jacobian:
                k1, a1, b1 = self.augm_state_derivatives_and_jacobians(propagated, controls)
                k2, a2, b2 = self.augm_state_derivatives_and_jacobians(propagated + 0.5 * substep * k1, controls)
                k3, a3, b3 = self.augm_state_derivatives_and_jacobians(propagated + 0.5 * substep * k2, controls)
                k4, a4, b4 = self.augm_state_derivatives_and_jacobians(propagated + substep * k3, controls)

                d1_state, d1_control = a1, b1
                d2_state = a2 @ (identity + 0.5 * substep * d1_state)
                d2_control = b2 + a2 @ (0.5 * substep * d1_control)
                d3_state = a3 @ (identity + 0.5 * substep * d2_state)
                d3_control = b3 + a3 @ (0.5 * substep * d2_control)
                d4_state = a4 @ (identity + substep * d3_state)
                d4_control = b4 + a4 @ (substep * d3_control)

                step_state = identity + (substep / 6.0) * (d1_state + 2.0 * d2_state + 2.0 * d3_state + d4_state)
                step_control = (substep / 6.0) * (
                    d1_control + 2.0 * d2_control + 2.0 * d3_control + d4_control
                )
                control_sensitivity = step_state @ control_sensitivity + step_control
                state_sensitivity = step_state @ state_sensitivity
            else:
                derivative = mapped if symbolic else self.augm_state_derivatives
                k1 = derivative(propagated, controls)
                k2 = derivative(propagated + 0.5 * substep * k1, controls)
                k3 = derivative(propagated + 0.5 * substep * k2, controls)
                k4 = derivative(propagated + substep * k3, controls)

            propagated = propagated + (substep / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return propagated, state_sensitivity, control_sensitivity


def cholesky_lower_triangular_matrix(matrix, dimension: int, pivot_floor: float = 0.0):                     
    """Lower-triangular Cholesky factor, built from scalar operations.

    Lower-triangularity is required, not merely convenient: it confines all
    mass-state coupling to the last row, so column n_x of the factor perturbs the
    mass alone at every node and the two mass sigma points keep a zero primed
    deviation even after mass becomes correlated with position and velocity.
    """

    factor: list[list] = [[casadi.MX(0.0)] * dimension for _ in range(dimension)]
    for row in range(dimension):
        for column in range(row + 1):
            total = matrix[row, column]
            for inner in range(column):
                total = total - factor[row][inner] * factor[column][inner]
            if row == column:
                # NOTE: Floor the pivot, not merely the square root. A trial point where
                # the propagated covariance is numerically indefinite gives a
                # negative pivot; clamping it to something infinitesimal avoids the
                # NaN but the next line divides by it, so an infinitesimal clamp
                # produces factor entries of order 1e150. `pivot_floor` is the jitter
                # already added to the matrix, which is exactly the smallest pivot an exactly
                # positive semi-definite input could have, so the clamp only ever
                # activates on round-off.
                factor[row][column] = casadi.sqrt(casadi.fmax(total, pivot_floor))
            else:
                factor[row][column] = total / factor[column][column]
    return casadi.vertcat(*[casadi.horzcat(*row) for row in factor])


def determinant(matrix, dimension: int):
    """Determinant by cofactor expansion, memoised over the remaining columns.

    `casadi.det` builds a `Determinant` node that MX graphs cannot evaluate
    numerically, so the expansion is written out. Sharing the sub-determinants
    turns the factorial expansion into `O(2^n n)`.
    """

    cache: dict[tuple[int, tuple[int, ...]], object] = {}

    def expand(row: int, columns: tuple[int, ...]):
        if not columns:
            return casadi.MX(1.0)
        key = (row, columns)
        if key not in cache:
            total = casadi.MX(0.0)
            for position, column in enumerate(columns):
                remaining = columns[:position] + columns[position + 1:]
                term = matrix[row, column] * expand(row + 1, remaining)
                total = total + term if position % 2 == 0 else total - term
            cache[key] = total
        return cache[key]

    return expand(0, tuple(range(dimension)))


def max_eigval_sym_3by3(matrix, relative_floor: float = 1e-14):
    """Closed-form largest eigenvalue of a symmetric 3x3 matrix.
    Ref: https://dl.acm.org/doi/pdf/10.1145/355578.366316
    
    The closed form is scale-free instead: its
    relative accuracy does not depend on the magnitude of the matrix.

    `p` is floored relative to the trace rather than by an absolute constant, so
    the removable singularity at an isotropic matrix is handled at any scale. The
    formula is smooth wherever the largest eigenvalue is simple; a repeated
    largest eigenvalue is non-generic for `Sigma^T = (n_x/6) K P' K^T` with a
    generic gain.
    """

    q = (matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) / 3.0
    p1 = matrix[0, 1] ** 2 + matrix[0, 2] ** 2 + matrix[1, 2] ** 2
    p2 = (
        (matrix[0, 0] - q) ** 2 + (matrix[1, 1] - q) ** 2 + (matrix[2, 2] - q) ** 2
    ) + 2.0 * p1
    p = casadi.sqrt(p2 / 6.0 + (relative_floor * q) ** 2 + 1e-300)
    B = (matrix - q * casadi.DM.eye(NU)) / p
    r = determinant(B, NU) / 2.0
    r = casadi.fmax(casadi.fmin(r, 1.0 - 1e-12), -1.0 + 1e-12)
    return q + 2.0 * p * casadi.cos(casadi.acos(r) / 3.0)


def symbolic_psqrt_spectral_radius(matrix, floor: float):
    """rho(A) = sqrt(lambda_max(A)), floored so the gradient stays bounded at A = 0.
    """

    return casadi.sqrt(max_eigval_sym_3by3(matrix) + floor ** 2)


def get_arc_function(
    case: TestCase,
    options: Options,
    dynamics: Dynamics,
    normalization: Normalization,
    step: float,
    index: int,
) -> casadi.Function:
    """
    Build a function for propagating the state and covariance through one arc.

    Everything around the integration is symbolic, so the chain rule assembles
    itself; only the propagation is opaque, since is evaluated numerically.
    """

    dimension = augmented_dimension(options)
    n_sigma = 2 * augmented_dimension(options) + 1

    mean = casadi.MX.sym("mu", NX)
    covariance = casadi.MX.sym("P", NX, NX)
    feedforward = casadi.MX.sym("S", NU)
    gain = casadi.MX.sym("K", NU, NP)

    weights = unscented_weights(options.scaling_parameter, dimension)
    scale = casadi.DM(normalization.matrix)
    inverse_scale = casadi.DM(normalization.inverse_matrix)

    # Sigma points on the augmented vector z = [x ; eta] with covariance
    # blkdiag(Sigma, R_bar). A Cholesky factor of a block-diagonal matrix is
    # block diagonal, so only the state block needs a symbolic factorisation --
    # the noise block is constant and is factorised once. The columns of
    # the state block therefore carry no measurement noise, and the columns of the
    # noise block sit at the mean state with a perturbed thrust.
    state_spread = cholesky_lower_triangular_matrix(
        dimension * covariance + options.cholesky_jitter * casadi.DM.eye(NX),
        NX,
        pivot_floor=options.cholesky_jitter,
    )
    state_columns = [state_spread[:, column] for column in range(NX)]
    measurement_columns = [state_spread[0:NP, column] for column in range(NX)]

    if dimension != NX:
        # R_bar is diagonal, so its Cholesky factor is the elementwise square
        # root. 
        noise_root = casadi.DM(
            np.diag(
                np.sqrt(
                    np.clip(
                        dimension * np.diag(navigation_covariance(options, normalization)),
                        0.0,
                        None,
                    )
                )
            )
        )
        state_columns += [casadi.MX.zeros(NX)] * NX
        measurement_columns += [noise_root[0:NP, column] for column in range(NX)]

    # NOTE: The mean state is in original units, not normalised. Whilst the nominal thrust is normalised.
    states = [mean]
    controls = [feedforward]
    for sign in (1.0, -1.0):
        for column in range(dimension):
            states.append(mean + sign * (scale @ state_columns[column]))
            controls.append(feedforward + sign * (gain @ measurement_columns[column]))

    sigma_states = casadi.horzcat(*states)
    sigma_controls = case.max_thrust_nd * casadi.horzcat(*controls)
    propagated, _, _ = dynamics.propagate(
        sigma_states, sigma_controls, step, options.integrator_substeps
    )

    mean_next = sum(weights[j] * propagated[:, j] for j in range(n_sigma))
    covariance_next = casadi.MX.zeros(NX, NX)
    for j in range(n_sigma):
        residual = inverse_scale @ (propagated[:, j] - mean_next)
        covariance_next = covariance_next + weights[j] * residual @ residual.T
    # NOTE: Enforces exact numerical symmetry.
    covariance_next = 0.5 * (covariance_next + covariance_next.T)

    control_covariance = casadi.MX.zeros(NU, NU)
    for column in range(dimension):
        deviation = gain @ measurement_columns[column]
        control_covariance = control_covariance + deviation @ deviation.T
    control_covariance = 2.0 * control_covariance_weight(options) * control_covariance
    control_covariance = 0.5 * (control_covariance + control_covariance.T)

    return casadi.Function(
        f"arc_{index}",
        [mean, covariance, feedforward, gain],
        [mean_next, covariance_next, control_covariance],
        ["mu", "P", "S", "K"],
        ["mu_next", "P_next", "SigmaT"],
    )


# --------------------------------------------------------------------------- #
# Reference Trajectory
# --------------------------------------------------------------------------- #


@dataclass
class ReferenceTraj:
    """Fuel-optimal DIRTRAN solution restricted to a subset of its knot points."""

    node_times: np.ndarray   # [-] non-dimensional, first entry zero
    steps: np.ndarray        # [-] arc durations, generally non-uniform
    states: np.ndarray       # [-] (NX, N + 1)
    controls: np.ndarray     # [-] (NU, N) thrust, zero-order hold
    fuel_consumed: float     # [kg]

    @property
    def n_arcs(self) -> int:
        return self.steps.size


def load_ref_traj(case: TestCase, options: Options) -> ReferenceTraj:
    data = np.load(REFERENCE_DIR / f"{case.test_case_id}.npz", allow_pickle=True)
    mesh = np.asarray(data["mesh_fraction"], dtype=float)
    states = np.asarray(data["x"], dtype=float)
    controls = np.asarray(data["u"], dtype=float)

    stride = max(int(options.mesh_stride), 1)
    kept = sorted(set(list(range(0, mesh.size - 1, stride)) + [mesh.size - 1]))
    kept = np.asarray(kept, dtype=int)

    node_times = mesh[kept] * case.tof_nd
    fine_steps = np.diff(mesh) * case.tof_nd
    coarse_controls = np.empty((NU, kept.size - 1), dtype=float)
    for k in range(kept.size - 1):
        window = slice(kept[k], kept[k + 1])
        weights = fine_steps[window]
        coarse_controls[:, k] = controls[:, window] @ weights / weights.sum()

    history = data["history"]
    fuel_consumed = float(history[-1]["fuel_consumed_kg"]) if history.size else float("nan")
    return ReferenceTraj(
        node_times=node_times,
        steps=np.diff(node_times),
        states=states[:, kept],
        controls=coarse_controls,
        fuel_consumed=fuel_consumed,
    )


# --------------------------------------------------------------------------- #
# Moment propagation and gain seeding
# --------------------------------------------------------------------------- #


def propagate_stochastic_moments(
    arc_functions: list[casadi.Function],
    means: np.ndarray,
    feedforward: np.ndarray,
    gains: np.ndarray,
    initial_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerically evaluate the covariance recursion along a given mean trajectory.

    Calls the very same arc functions used by the NLP, so no logic is duplicated.
    """

    n_arcs = feedforward.shape[1]
    covariances = np.empty((n_arcs + 1, NX, NX), dtype=float)
    control_covariances = np.empty((n_arcs, NU, NU), dtype=float)
    propagated_means = np.empty((NX, n_arcs + 1), dtype=float)
    covariances[0] = initial_covariance
    propagated_means[:, 0] = means[:, 0]

    for k in range(n_arcs):
        mean_next, covariance_next, control_covariance = arc_functions[k](
            means[:, k], covariances[k], feedforward[:, k], gains[k]
        )
        propagated_means[:, k + 1] = np.asarray(mean_next.full(), dtype=float).ravel()
        covariances[k + 1] = np.asarray(covariance_next.full(), dtype=float)
        control_covariances[k] = np.asarray(control_covariance.full(), dtype=float)

    return propagated_means, covariances, control_covariances


def psqrt_spectral_radii(matrices: np.ndarray) -> np.ndarray:
    """rho(A) = sqrt(lambda_max(A)) for a stack of symmetric matrices."""

    eigenvalues = np.linalg.eigvalsh(matrices)
    return np.sqrt(np.clip(eigenvalues[..., -1], 0.0, None))


def normalized_arc_jacobians(
    case: TestCase,
    dynamics: Dynamics,
    normalization: Normalization,
    reference_states: np.ndarray,
    reference_controls: np.ndarray,
    steps: np.ndarray,
    substeps: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Normalised open-loop arc Jacobians A_tilde, B_tilde of the discrete map."""

    scale = normalization.matrix
    inverse_scale = normalization.inverse_matrix
    state_matrices: list[np.ndarray] = []
    control_matrices: list[np.ndarray] = []
    for k in range(steps.size):
        _, state_sensitivity, control_sensitivity = dynamics.propagate(
            reference_states[:, k : k + 1],
            reference_controls[:, k : k + 1],
            float(steps[k]),
            substeps,
            with_jacobian=True,
        )
        state_matrices.append(inverse_scale @ state_sensitivity[0] @ scale)
        control_matrices.append(inverse_scale @ control_sensitivity[0] * case.max_thrust_nd)
    return state_matrices, control_matrices


def tvlqr_gains(
    control_weight: float,
    terminal_weight: float | np.ndarray,
    state_matrices: list[np.ndarray],
    control_matrices: list[np.ndarray],
) -> np.ndarray:
    """Backward Riccati recursion on the normalised linearisation."""

    Q_normalized = np.eye(NX)
    R_normalized = control_weight * np.eye(NU)
    terminal_weights = np.asarray(terminal_weight, dtype=float)
    if terminal_weights.ndim == 0:
        terminal_weights = np.full(NX, float(terminal_weights))
    elif terminal_weights.shape != (NX,):
        raise ValueError(f"terminal_weight must be a scalar or have shape ({NX},).")
    P_normalized = np.diag(terminal_weights)
    gains = np.zeros((len(state_matrices), NU, NP), dtype=float)

    for k in reversed(range(len(state_matrices))):
        state_matrix = state_matrices[k]
        control_matrix = control_matrices[k]
        gain = np.linalg.solve(
            R_normalized + control_matrix.T @ P_normalized @ control_matrix,
            control_matrix.T @ P_normalized @ state_matrix,
        )
        gains[k] = -gain[:, 0:NP]
        closed_loop_matrix = state_matrix - control_matrix @ gain
        P_normalized = Q_normalized + gain.T @ R_normalized @ gain + closed_loop_matrix.T @ P_normalized @ closed_loop_matrix
        P_normalized = 0.5 * (P_normalized + P_normalized.T)

    return gains


def seed_gains(
    options: Options,
    state_matrices: list[np.ndarray],
    control_matrices: list[np.ndarray],
) -> tuple[np.ndarray, tuple[float, np.ndarray]]:
    """Warm-start gains from a Riccati recursion with Bryson-rule weights.

    Bryson's rule weights each channel by the inverse square of its largest
    acceptable deviation. In physical units, with `c` the sigma factor,

        Q_ii = 1 / (c sigma_0,i)^2,  R_jj = 1 / T_max^2,  Qf_ii = 1 / (c sigma_f,i)^2

    taking the initial dispersion as the acceptable state deviation, the thrust
    bound as the acceptable control, and the terminal requirement for `Qf`.
    Transforming into the normalised coordinates the recursion runs in
    (`dx = D0 dx~`, `dT = T_max du~`) turns the cost `dx' Q dx + du' R du` into
    `dx~' (D0 Q D0) dx~ + T_max^2 du~' R du~`, so

        Q~  = D0 Q D0  = (1/c^2) I,   R~ = T_max^2 R = I,
        Qf~ = (1/c^2) diag(r_p I_3, r_v I_3)

    because `sigma_f = sigma_0 / sqrt(reduction)` makes each terminal
    `Qf/Q` ratio equal its requested reduction. Only ratios matter, so this is
    equivalent to `Q = I`, `R = c^2`, and terminal weights `r_p` and `r_v`.
    The unconstrained mass state retains the stricter of those weights so that
    equal position and velocity reductions reproduce the previous seed exactly.

    """

    control_weight = 3.0 ** 2
    position_reduction = float(options.position_covariance_reduction)
    velocity_reduction = float(options.velocity_covariance_reduction)
    terminal_weights = np.array(
        [position_reduction] * 3
        + [velocity_reduction] * 3
        + [max(position_reduction, velocity_reduction)],
        dtype=float,
    )
    gains = tvlqr_gains(
        control_weight, terminal_weights, state_matrices, control_matrices
    )
    return gains, (control_weight, terminal_weights)


@dataclass
class InitialGuess:
    means: np.ndarray
    feedforward: np.ndarray
    slack: np.ndarray
    gains: np.ndarray
    radius: np.ndarray
    gain_scale: np.ndarray
    terminal_margin: np.ndarray   # lower-triangular entries of G, row-major


def build_initial_guess(
    case: TestCase,
    options: Options,
    dynamics: Dynamics,
    normalization: Normalization,
    ref_traj: ReferenceTraj,
    arc_functions: list[casadi.Function],
    psi_inv: float,
) -> InitialGuess:
    """Warm start that already satisfies the thrust chance constraint.

    The fuel optimal solution is bang-bang at exactly T_max, so the probabilistic thrust constraint is violated at
    every thrust arc as soon as any feedback is present. The seed gains are scaled
    down until the closed-loop share of the budget leaves room, then the open-loop
    magnitudes are trimmed to fit under what remains.
    """

    means = np.array(ref_traj.states, dtype=float)
    means[:, 0] = case.x0_augmented_state
    feedforward = np.array(ref_traj.controls, dtype=float) / case.max_thrust_nd

    state_matrices, control_matrices = normalized_arc_jacobians(
        case, dynamics, normalization, ref_traj.states, ref_traj.controls,
        ref_traj.steps, options.integrator_substeps,
    )
    # Variable scaling for the gains. In normalised units the control matrix norm ||B_tilde|| 
    # can be very large, because correcting a 1-sigma deviation costs a negligible fraction of
    # T_max; a unit step in K_tilde would therefore change the closed-loop
    # transition drastically and every optimizer step overshoots by orders of
    # magnitude. Scaling by 1 / ||B_tilde|| makes a unit step in the decision
    # variable an O(1) change in the closed-loop dynamics, which is the condition
    # the solver implicitly assumes.
    gain_scale = np.array(
        [1.0 / max(np.linalg.norm(matrix, 2), 1e-300) for matrix in control_matrices]
    )
    if options.warm_start_gains:
        gains, seed_weights = seed_gains(options, state_matrices, control_matrices)
    else:
        # Uniform negative gains rather than zeros.
        gains = -COLD_START_GAIN * np.tile(
            gain_scale[:, None, None], (1, NU, NP)
        )
        seed_weights = None

    radius = np.zeros(ref_traj.n_arcs, dtype=float)
    covariances = np.tile(normalization.initial_covariance, (ref_traj.n_arcs + 1, 1, 1))
    for iteration in range(30):
        _, covariances, control_covariances = propagate_stochastic_moments(
            arc_functions, means, feedforward, gains, normalization.initial_covariance
        )
        radius = psqrt_spectral_radii(control_covariances)
        worst = float(np.max(psi_inv * radius))
        if not np.isfinite(worst):
            raise FloatingPointError(
                "Non-finite closed-loop control-effort seed margin at gain-scaling "
                f"iteration {iteration + 1}."
            )
        if worst <= options.cl_control_effort_seed_margin:
            break
        gains *= np.sqrt(options.cl_control_effort_seed_margin / worst)

    headroom = np.clip(1.0 - psi_inv * radius, 0.0, None)
    magnitudes = np.linalg.norm(feedforward, axis=0)
    factor = np.ones_like(magnitudes)
    active = magnitudes > 1e-12
    factor[active] = np.minimum(1.0, headroom[active] / magnitudes[active])
    feedforward = feedforward * factor


    position_target = 1.0 / options.position_covariance_reduction
    velocity_target = 1.0 / options.velocity_covariance_reduction
    seed_terminal_covariance = covariances[-1]
    position_ratio = float(
        np.linalg.eigvalsh(seed_terminal_covariance[0:3, 0:3])[-1]
        / position_target
    )
    velocity_ratio = float(
        np.linalg.eigvalsh(seed_terminal_covariance[3:NP, 3:NP])[-1]
        / velocity_target
    )
    origin = (
        f"uniform initial gains (K_hat = {-COLD_START_GAIN:g})"
        if seed_weights is None
        else (
            "Bryson's rule warm start "
            f"(Q=I, R={seed_weights[0]:.4g}, "
            f"Qf_pos={seed_weights[1][0]:.4g}, "
            f"Qf_vel={seed_weights[1][3]:.4g})"
        )
    )
    print(
        f"[{case.test_case_id}] {origin}: terminal blocks pos {position_ratio:.4e} / "
        f"vel {velocity_ratio:.4e} x their respective targets "
        f"({'feasible' if max(position_ratio, velocity_ratio) <= 1.0 else 'INFEASIBLE start'})",
        flush=True,
    )

    # Cholesky factor of the terminal Loewner slack at the seed, used as the
    # initial guess for the margin decision variables in solve_rocp. A seed that
    # already violates the target leaves terminal_slack indefinite, so any
    # negative eigenvalues are floored before factorising -- this is only a
    # starting point for the solver, not a feasibility claim.
    inverse_target_std = 1.0 / np.sqrt(
        [position_target] * 3 + [velocity_target] * 3
    )
    scaled_terminal = np.outer(inverse_target_std, inverse_target_std) * seed_terminal_covariance[0:NP, 0:NP]
    terminal_slack = np.eye(NP) - scaled_terminal
    eigvals, eigvecs = np.linalg.eigh(0.5 * (terminal_slack + terminal_slack.T))
    terminal_slack_psd = eigvecs @ np.diag(np.clip(eigvals, 0.0, None)) @ eigvecs.T
    G_seed = np.linalg.cholesky(
        terminal_slack_psd + options.terminal_margin_floor ** 2 * np.eye(NP)
    )
    terminal_margin = np.array(
        [G_seed[row, col] for row in range(NP) for col in range(row + 1)]
    )

    return InitialGuess(
        means=means,
        feedforward=feedforward,
        slack=np.linalg.norm(feedforward, axis=0),
        gains=gains,
        radius=radius,
        gain_scale=gain_scale,
        terminal_margin=terminal_margin,
    )


# --------------------------------------------------------------------------- #
# Nonlinear program
# --------------------------------------------------------------------------- #


@dataclass
class RobustSolution:
    means: np.ndarray
    feedforward: np.ndarray            # S_tilde, normalised 
    gains: np.ndarray                  # K_tilde, normalised
    radius: np.ndarray                 # r_tilde
    covariances: np.ndarray            # P, normalised
    control_covariances: np.ndarray    # SigmaT, normalised
    objective: float
    # Every instance will have a different set of diagnostics.
    diagnostics: dict[str, float] = field(default_factory=dict)


def run_snopt_solver(opti: casadi.Opti, test_case_id: str) -> casadi.OptiSol:
    """Solve from a unique directory so SNOPT gets a fresh output file.

    CasADi initializes every Opti SNOPT instance with the relative output path
    ``solver.out``.  With SNOPT 7.7.7 on Windows, a second initialization at
    the same absolute path in one Python process can fail because the Fortran
    output unit from the first solve is not reusable.  A unique working
    directory makes the absolute output path unique for every solve.

    The directory is intentionally retained for the lifetime of the process:
    attempting to remove ``solver.out`` while SNOPT still owns its Fortran unit
    is not reliable on Windows.
    """

    snopt_work_dir = Path(
        tempfile.mkdtemp(prefix=f"casadi-snopt-{test_case_id}-")
    )
    original_work_dir = Path.cwd()
    try:
        os.chdir(snopt_work_dir)
        return opti.solve()
    finally:
        os.chdir(original_work_dir)


def solve_rocp(
    case: TestCase,
    options: Options,
    ref_traj: ReferenceTraj,
    arc_functions: list[casadi.Function],
    initial_guess: InitialGuess,
    normalization: Normalization,
    psi_inv: float,
) -> RobustSolution:
    """Transcribe and solve the robust optimal control problem."""

    n_arcs = ref_traj.n_arcs
    position_target = 1.0 / options.position_covariance_reduction
    velocity_target = 1.0 / options.velocity_covariance_reduction

    opti = casadi.Opti()
    means = opti.variable(NX, n_arcs + 1)
    feedforward = opti.variable(NU, n_arcs)
    slack = opti.variable(1, n_arcs)
    gains = [opti.variable(NU, NP) for _ in range(n_arcs)]

    opti.subject_to(means[:, 0] == case.x0_augmented_state)
    opti.subject_to(casadi.vec(slack) >= 0.0)

    covariance = casadi.DM(normalization.initial_covariance)
    objective = 0.0
    for k in range(n_arcs):
        mean_next, covariance, control_covariance = arc_functions[k](
            means[:, k], covariance, feedforward[:, k], float(initial_guess.gain_scale[k]) * gains[k]
        )
        # Multiple shooting on the mean, single shooting on the covariance.
        opti.subject_to(means[:, k + 1] == mean_next)

        # ||S_k|| via a slack, as in deterministic case, so the cost stays smooth at
        # the coast arcs where S_k vanishes.
        opti.subject_to(
            casadi.dot(feedforward[:, k], feedforward[:, k])
            <= slack[0, k] ** 2
        )

        # rho(SigmaT_k) in closed form 
        radius = symbolic_psqrt_spectral_radius(control_covariance, options.spectral_radius_floor)

        # Transcription of the control chance constraint
        opti.subject_to(slack[0, k] + psi_inv * radius <= 1.0)

        # weighted by the arc duration because the mesh is not uniform.
        objective = objective + float(ref_traj.steps[k]) * (
            slack[0, k] + psi_inv * radius
        )

    opti.subject_to(means[0:NP, n_arcs] == case.xf_state)

    # Terminal covariance constraint: the full 6x6 Loewner order
    # P'_N <= Dt, with Dt = diag(position_target * I_3, velocity_target * I_3),
    # encoded by a Cholesky residual. Writing
    # the normalised slack
    #     terminal_slack = I_6 - Dt^-1/2 P'_N Dt^-1/2,
    # any PSD matrix admits a factorisation terminal_slack = G G^T with G
    # lower-triangular, so constraining the 21 lower-triangular entries of
    # (terminal_slack - G G^T) to zero is equivalent to terminal_slack >= 0, i.e.
    # P'_N <= Dt exactly -- jointly over position, velocity, and their
    # cross-covariance, unlike the block-diagonal eigenvalue-bound relaxation.
    # diag(G) >= terminal_margin_floor keeps G -> G G^T non-degenerate near the
    # boundary.
    inverse_target_std = casadi.DM(
        1.0 / np.sqrt([position_target] * 3 + [velocity_target] * 3)
    )
    scaled_terminal = (
        casadi.diag(inverse_target_std)
        @ covariance[0:NP, 0:NP]
        @ casadi.diag(inverse_target_std)
    )
    terminal_slack = casadi.MX.eye(NP) - scaled_terminal

    n_margin = NP * (NP + 1) // 2
    margin = opti.variable(n_margin)
    lower_indices = [(row, col) for row in range(NP) for col in range(row + 1)]
    margin_matrix = [[casadi.MX(0.0)] * NP for _ in range(NP)]
    for (row, col), entry in zip(lower_indices, casadi.vertsplit(margin)):
        margin_matrix[row][col] = entry
    G = casadi.vertcat(*[casadi.horzcat(*row) for row in margin_matrix])

    opti.subject_to(casadi.diag(G) >= options.terminal_margin_floor)
    residual = terminal_slack - G @ G.T
    opti.subject_to(
        casadi.vertcat(*[residual[row, col] for row, col in lower_indices]) == 0.0
    )

    opti.minimize(objective)

    opti.set_initial(means, initial_guess.means)
    opti.set_initial(feedforward, initial_guess.feedforward)
    opti.set_initial(slack, np.maximum(initial_guess.slack, 1e-9).reshape(1, -1))
    for k in range(n_arcs):
        opti.set_initial(gains[k], initial_guess.gains[k] / initial_guess.gain_scale[k])
    opti.set_initial(margin, initial_guess.terminal_margin)

    # SNOPT major/minor tolerances play the role IPOPT's tol/constr_viol_tol
    # play: "Major optimality tolerance" is the KKT/reduced-gradient tolerance,
    # and "Major feasibility tolerance" is the nonlinear constraint-violation
    # tolerance. CasADi 3.7.2's SNOPT interface tries numeric options as integers
    # before trying them as reals. Fractional Python floats can consequently be
    # converted to zero and leave SNOPT's defaults in effect. Pass real-valued
    # SNOPT options as strings so they reach SNOPT through its generic parameter
    # parser with their intended values.
    #
    # The default elastic weight (1e4) was too small to hold feasibility without
    # repeatedly escalating the penalty parameter (observed climbing to ~1e11,
    # which loosens the effective optimality/feasibility test since both are
    # scaled by the multiplier/penalty magnitude). NOTE: an earlier attempt that
    # also set "Scale option": 0 (disabling SNOPT's automatic scaling, reasoning
    # the problem is already hand-normalised) made things worse - more major
    # iterations, penalty parameter still climbing to ~1.5e9, looser constraint
    # violation - so scaling is left at its default here. Only the elastic
    # weight is changed, and this needs to be re-verified against a plain
    # (no override) run before trusting it fixed anything.
    snopt_options = {
        "Major iterations limit": options.major_max_iter,
        "Minor iterations limit": max(500, options.minor_max_iter),
        "Major optimality tolerance": f"{options.major_optimality_tol:.13g}",
        "Major feasibility tolerance": f"{options.major_feasibility_tol:.13g}",
        "Minor feasibility tolerance": f"{options.minor_feasibility_tol:.13g}",
        # CasADi opens solver.out while initializing SNOPT, before it applies
        # these native options. "Print file": 0 suppresses subsequent output
        # to that file but cannot prevent it from being opened. The unique
        # working directory in run_snopt_solver is what prevents consecutive
        # solves from reusing the same absolute solver.out path. "Summary
        # file": 6 keeps the major-iteration log on standard output.
        "Print file": 0,
        "Summary file": 6 if options.print_level else 0,
        "Major print level": options.print_level,
        "Minor print level": 0,
    }
    opti.solver(
        "snopt",
        {"expand": options.expand_graph, "print_time": True},
        snopt_options,
    )
    solver_name = "SNOPT"


    try:
        solution = run_snopt_solver(opti, case.test_case_id)
        converged = True
    except RuntimeError as error:
        print(f"  {solver_name} did not converge ({error}); returning the last iterate.", flush=True)
        solution = opti.debug
        converged = False

    solved_means = np.asarray(solution.value(means), dtype=float)
    solved_feedforward = np.asarray(solution.value(feedforward), dtype=float).reshape(NU, n_arcs)
    solved_gains = np.stack(
        [
            initial_guess.gain_scale[k] * np.asarray(solution.value(gain), dtype=float).reshape(NU, NP)
            for k, gain in enumerate(gains)
        ]
    )

    propagated_means, covariances, control_covariances = propagate_stochastic_moments(
        arc_functions, solved_means, solved_feedforward, solved_gains, normalization.initial_covariance
    )
    matching_defect = float(np.max(np.abs(propagated_means[:, 1:] - solved_means[:, 1:])))
    # rho is an expression, so recover it exactly from the propagated moments.
    solved_radius = np.sqrt(
        psqrt_spectral_radii(control_covariances) ** 2 + options.spectral_radius_floor ** 2
    )

    return RobustSolution(
        means=solved_means,
        feedforward=solved_feedforward,
        gains=solved_gains,
        radius=solved_radius,
        covariances=covariances,
        control_covariances=control_covariances,
        objective=float(solution.value(objective)),
        diagnostics={"converged": float(converged), "matching_defect_nd": matching_defect},
    )


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #


def monte_carlo_rollouts(
    case: TestCase,
    options: Options,
    dynamics: Dynamics,
    normalization: Normalization,
    solution: RobustSolution,
    steps: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Apply the converged policy with thrust saturation.

    Violation statistics use the unsaturated commanded thrust so they still test
    the chance constraint.  Propagation, accumulated effort and peak applied
    thrust use the command clipped to the maximum thrust, matching the physical actuator.
    """

    generator = np.random.default_rng(options.monte_carlo_seed)
    samples = options.monte_carlo_samples
    states = solution.means[:, 0:1] + normalization.initial_std[:, None] * generator.standard_normal((NX, samples))
    # Navigation error: an independent draw at every arc, added to the measured
    # deviation the policy acts on. 
    navigation_std = np.sqrt(np.clip(np.diag(navigation_covariance(options, normalization)), 0.0, None))

    # Cumulative control effort is stored for each sample
    cumulative_control_effort = np.zeros(samples, dtype=float)
    # Stores one violation fraction per arc, averaged over all samples. 
    violation_fraction = np.zeros(steps.size, dtype=float)
    # The worst exceedance over all samples and arcs.
    worst_exceedance = -np.inf
    peak_applied_thrust = np.zeros(samples, dtype=float)

    for k in range(steps.size):
        deviation = (states[0:NP] - solution.means[0:NP, k : k + 1]) / normalization.scale[0:NP, None]
        deviation = deviation + navigation_std[0:NP, None] * generator.standard_normal((NP, samples))
        commanded_control = case.max_thrust_nd * (
            solution.feedforward[:, k : k + 1] + solution.gains[k] @ deviation
        )
        commanded_magnitudes = np.linalg.norm(commanded_control, axis=0)

        violation_fraction[k] = float(np.mean(commanded_magnitudes > case.max_thrust_nd))
        worst_exceedance = max(
            worst_exceedance,
            float(np.max(commanded_magnitudes) - case.max_thrust_nd),
        )

        # Radially project each over-limit command onto the thrust ball, preserving
        # its direction. Diagnostics above retain the original command so actuator
        # saturation cannot hide a chance-constraint violation.
        saturation_scale = np.ones_like(commanded_magnitudes)
        saturated = commanded_magnitudes > case.max_thrust_nd
        saturation_scale[saturated] = (
            case.max_thrust_nd / commanded_magnitudes[saturated]
        )
        applied_control = commanded_control * saturation_scale[None, :]
        applied_magnitudes = np.minimum(commanded_magnitudes, case.max_thrust_nd)

        peak_applied_thrust = np.maximum(peak_applied_thrust, applied_magnitudes)
        cumulative_control_effort += float(steps[k]) * applied_magnitudes
        states, _, _ = dynamics.propagate(
            states, applied_control, float(steps[k]), options.integrator_substeps
        )

    terminal_deviation = (states - solution.means[:, -1:]) / normalization.scale[:, None]
    sampled_covariance = np.cov(terminal_deviation)

    return {
        "cumulative_control_effort": cumulative_control_effort,
        "percentile": float(np.percentile(cumulative_control_effort, 100.0 * (1.0 - options.violation_parameter))),
        "mean_cost": float(np.mean(cumulative_control_effort)),
        "violation_fraction": violation_fraction,
        "max_violation_fraction": float(np.max(violation_fraction)),
        "worst_exceedance_n": float(worst_exceedance) * case.thrust_unit,
        "peak_applied_thrust_n": peak_applied_thrust * case.thrust_unit,
        "terminal_covariance": sampled_covariance,
        "terminal_deviation": terminal_deviation,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def compute_diagnostics(
    case: TestCase,
    options: Options,
    ref_traj: ReferenceTraj,
    solution: RobustSolution,
    normalization: Normalization,
    monte_carlo_result: dict,
    psi_inv: float,
) -> dict[str, float]:
    position_target = 1.0 / options.position_covariance_reduction
    velocity_target = 1.0 / options.velocity_covariance_reduction
    target_diagonal = np.array(
        [position_target] * 3 + [velocity_target] * 3, dtype=float
    )
    inverse_target_std = 1.0 / np.sqrt(target_diagonal)
    steps = ref_traj.steps

    feedforward_magnitude = np.linalg.norm(solution.feedforward, axis=0)
    deterministic_effort = float(np.sum(steps * feedforward_magnitude))
    feedback_effort = float(np.sum(steps * psi_inv * solution.radius))
    total_effort = deterministic_effort + feedback_effort

    terminal_covariance_target_ratio = (
        solution.covariances[-1][0:NP, 0:NP]
        * np.outer(inverse_target_std, inverse_target_std)
    )
    terminal_covariance_target_ratio_eigvals = np.linalg.eigvalsh(terminal_covariance_target_ratio)

    componentwise_mean_error = float(np.max(np.abs(solution.means[0:NP, -1] - case.xf_state)))
    thrust_budget = feedforward_magnitude + psi_inv * solution.radius

    empirical = (
        monte_carlo_result["terminal_covariance"][0:NP, 0:NP]
        * np.outer(inverse_target_std, inverse_target_std)
    )
    empirical_eigenvalues = np.linalg.eigvalsh(empirical)
    position_block_ratio = np.linalg.eigvalsh(terminal_covariance_target_ratio[0:3, 0:3])[-1]
    velocity_block_ratio = np.linalg.eigvalsh(terminal_covariance_target_ratio[3:NP, 3:NP])[-1]

    return {
        "n_arcs": float(ref_traj.n_arcs),
        "mesh_stride": float(options.mesh_stride),
        "converged": solution.diagnostics["converged"],
        "matching_defect_nd": solution.diagnostics["matching_defect_nd"],
        "objective_nd": float(solution.objective),
        "deterministic_effort_nd": deterministic_effort,
        "feedback_effort_nd": feedback_effort,
        "total_effort_nd": total_effort,
        "deterministic_effort_kg": deterministic_effort * case.m0_wet * case.max_thrust_nd / case.exhaust_velocity_nd,
        "feedback_effort_kg": feedback_effort * case.m0_wet * case.max_thrust_nd / case.exhaust_velocity_nd,
        "total_effort_kg": total_effort * case.m0_wet * case.max_thrust_nd / case.exhaust_velocity_nd,
        "nominal_fuel_consumed": ref_traj.fuel_consumed,
        "max_thrust_budget": float(np.max(thrust_budget)),
        "max_feedforward_n": float(np.max(feedforward_magnitude) * case.max_thrust_nd * case.thrust_unit),
        "max_feedback_n": float(
            np.max(psi_inv * solution.radius) * case.max_thrust_nd * case.thrust_unit
        ),
        "terminal_componentwise_mean_error_nd": componentwise_mean_error,
        # Position/velocity block ratios are reported for interpretability, but
        # solve_rocp constrains the full 6x6 Loewner order P'_N <= Dt directly
        # (Cholesky-residual encoding), not these two blocks separately.
        "terminal_position_block_ratio": float(position_block_ratio),
        "terminal_velocity_block_ratio": float(velocity_block_ratio),
        "position_covariance_reduction_requested": float(
            options.position_covariance_reduction
        ),
        "velocity_covariance_reduction_requested": float(
            options.velocity_covariance_reduction
        ),
        # The quantity actually constrained: lambda_max(Dt^-1/2 P'_N Dt^-1/2) <= 1.
        "terminal_covariance_max_eigenvalue": float(terminal_covariance_target_ratio_eigvals[-1]),
        "terminal_covariance_satisfied": float(
            terminal_covariance_target_ratio_eigvals[-1] <= 1.0 + 1e-12
        ),
        "terminal_position_std_m": float(
            np.sqrt(solution.covariances[-1][0, 0]) * normalization.scale[0] * case.length_unit * 1e3
        ),
        "terminal_velocity_std_ms": float(
            np.sqrt(solution.covariances[-1][3, 3]) * normalization.scale[3] * case.velocity_unit * 1e3
        ),
        "monte_carlo_percentile_nd": float(monte_carlo_result["percentile"]),
        "predicted_percentile_nd": total_effort * case.max_thrust_nd,
        "monte_carlo_mean_nd": float(monte_carlo_result["mean_cost"]),
        "monte_carlo_percentile_kg": float(monte_carlo_result["percentile"]) * case.m0_wet / case.exhaust_velocity_nd,
        "monte_carlo_max_violation_fraction": float(monte_carlo_result["max_violation_fraction"]),
        "monte_carlo_worst_exceedance_n": float(monte_carlo_result["worst_exceedance_n"]),
        "monte_carlo_terminal_max_eigenvalue": float(empirical_eigenvalues[-1]),
    }


def print_summary(
    case: TestCase,
    diagnostics: dict[str, float],
    psi_inv: float,
) -> None:
    to_n = case.max_thrust_nd * case.thrust_unit
    lines = [
        "",
        f"{'=' * 68}",
        f"{case.display_name}  --  robust optimization summary",
        f"{'=' * 68}",
        f"arcs                      : {diagnostics['n_arcs']:.0f} "
        f"converged           : {'yes' if diagnostics['converged'] else 'no'}",
        f"max matching defect       : {diagnostics['matching_defect_nd']:.3e} [-]",
        f"non dimensional objective J   : {diagnostics['total_effort_nd']:.6e} [-]"
        f"   (NLP slack value {diagnostics['objective_nd']:.6e})",
        f"  open-loop  T_d          : {diagnostics['deterministic_effort_kg']:.6f} [kg]",
        f"  closed-loop T_s         : {diagnostics['feedback_effort_kg']:.6f} [kg]",
        f"  total                   : {diagnostics['total_effort_kg']:.6f} [kg]",
        f"Reference trajectory fuel consumption     : {diagnostics['nominal_fuel_consumed']:.6f} [kg]",
        f"",
        f"max ||S|| + Psi rho       : {diagnostics['max_thrust_budget']:.6f} "
        f"(must be <= 1, i.e. {to_n:.4f} [N])",
        f"",
        f"terminal mean error       : {diagnostics['terminal_componentwise_mean_error_nd']:.3e} [-]",
        f"terminal pos. block ratio : {diagnostics['terminal_position_block_ratio']:.6f} "
        f"(diagnostic only; reduction {diagnostics['position_covariance_reduction_requested']:.4g})",
        f"terminal vel. block ratio : {diagnostics['terminal_velocity_block_ratio']:.6f} "
        f"(diagnostic only; reduction {diagnostics['velocity_covariance_reduction_requested']:.4g})",
        f"full 6x6 max eigval       : {diagnostics['terminal_covariance_max_eigenvalue']:.6f} "
        f"(must be <= 1, the constrained quantity)",
        f"  terminal 1-sigma pos.   : {diagnostics['terminal_position_std_m']:.6e} [m]",
        f"  terminal 1-sigma vel.   : {diagnostics['terminal_velocity_std_ms']:.6e} [m/s]",
        f"MC 95th pct. of the non dimensional objective function     : {diagnostics['monte_carlo_percentile_nd']:.6e} [-] "
        f"vs predicted {diagnostics['predicted_percentile_nd'] :.6e} [-]",
        f"MC worst-arc where P(||T|| > Tmax) : {diagnostics['monte_carlo_max_violation_fraction']:.4f} "
        f"(must be <= 0.05); worst exceedance "
        f"{diagnostics['monte_carlo_worst_exceedance_n']:+.3e} [N]",
        f"MC terminal max eigval    : {diagnostics['monte_carlo_terminal_max_eigenvalue']:.6f}",
        f"{'=' * 68}",
    ]
    print("\n".join(lines), flush=True)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def std_physical_units(
    case: TestCase, normalization: Normalization, covariances: np.ndarray
) -> np.ndarray:
    """1-sigma of each augmented state in m, m/s and kg."""

    diagonal = np.sqrt(np.clip(np.einsum("kii->ki", covariances), 0.0, None))
    physical = diagonal * normalization.scale[None, :]
    scale = np.array(
        [case.length_unit * 1e3] * 3 + [case.velocity_unit * 1e3] * 3 + [case.m0_wet], dtype=float
    )
    return physical * scale[None, :]


def plot_outputs(
    case: TestCase,
    options: Options,
    ref_traj: ReferenceTraj,
    solution: RobustSolution,
    normalization: Normalization,
    monte_carlo_result: dict,
    psi_inv: float,
    output_prefix: Path,
) -> None:
    node_days = ref_traj.node_times * case.time_unit / 86400.0
    position_target = 1.0 / options.position_covariance_reduction
    velocity_target = 1.0 / options.velocity_covariance_reduction
    target_covariance = np.zeros((NX, NX), dtype=float)
    target_covariance[0:3, 0:3] = position_target * np.eye(3)
    target_covariance[3:NP, 3:NP] = velocity_target * np.eye(3)
    sigma_physical = std_physical_units(case, normalization, solution.covariances)
    target_sigma = std_physical_units(
        case, normalization, target_covariance[None, :, :]
    )[0]
    plotter = Plotter(output_prefix)
    rollout_history, rollout_thrust_n = plotter.plot_monte_carlo_rollouts(
        case,
        options,
        Dynamics(case),
        normalization,
        solution,
        ref_traj.steps,
    )

    # 1 - dispersion evolution.
    figure, axes = plt.subplots(
        1,
        3,
        figsize=Plotter.THREE_PANEL_FIGSIZE,
        dpi=Plotter.FIGURE_DPI,
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.0)},
    )
    panels = (
        (slice(0, 3), r"$\sigma$ [m]", (r"$x$", r"$y$", r"$z$"), (Plotter.BLUE, Plotter.RED, Plotter.GREY)),
        (
            slice(3, 6),
            r"$\sigma$ [m/s]",
            (r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"),
            (Plotter.BLUE, Plotter.RED, Plotter.GREY),
        ),
        (slice(6, 7), r"$\sigma$ [kg]", (r"$m$",), (Plotter.GREY,)),
    )
    for axis, (columns, label, names, colors) in zip(axes, panels):
        axis.set_box_aspect(1.0)
        for offset, (name, color) in enumerate(zip(names, colors)):
            values = np.maximum(sigma_physical[:, columns][:, offset], 1e-16)
            axis.semilogy(
                node_days,
                values,
                lw=Plotter.REFERENCE_LINE_WIDTH,
                color=color,
                label=name,
            )
        if columns.start < NP:
            axis.axhline(
                target_sigma[columns.start],
                color=Plotter.BLACK,
                ls="--",
                lw=Plotter.GUIDE_LINE_WIDTH,
                label="Target",
            )
        axis.set_xlabel("time [days]")
        axis.set_ylabel(label)
        plotter._style_2d_axis(
            axis,
            grid=True,
            grid_which="both",
            tick_size=Plotter.VERIFICATION_TICK_SIZE,
            label_size=Plotter.VERIFICATION_LABEL_SIZE,
        )
        plotter._legend(
            axis,
            loc="best",
            fontsize=7.0,
            frameon=True,
            fancybox=False,
            edgecolor=Plotter.BLACK,
            facecolor="white",
            framealpha=1.0,
            borderpad=0.28,
            handlelength=1.4,
        )
    figure.align_ylabels(axes)
    figure.tight_layout()
    figure.savefig(
        output_prefix.parent / f"{output_prefix.name}_dispersion_evolution.png",
        dpi=Plotter.FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)

    # 2 - thrust magnitude, Monte Carlo values and covariance-predicted envelope.
    nominal_thrust_n = np.linalg.norm(solution.feedforward, axis=0) * case.max_thrust_nd * case.thrust_unit
    feedback_n = psi_inv * solution.radius * case.max_thrust_nd * case.thrust_unit
    lower_bound_n = np.maximum(nominal_thrust_n - feedback_n, 0.0)
    upper_bound_n = nominal_thrust_n + feedback_n
    thrust_magnification = plotter.get_thrust_deviation_magnification(
        feedback_n, case.max_thrust_n
    )
    thrust_scale_label = plotter.plot_magnification_label(
        thrust_magnification
    )
    displayed_lower_n = nominal_thrust_n + thrust_magnification * (
        lower_bound_n - nominal_thrust_n
    )
    displayed_upper_n = nominal_thrust_n + thrust_magnification * (
        upper_bound_n - nominal_thrust_n
    )
    displayed_rollout_n = nominal_thrust_n[:, None] + thrust_magnification * (
        rollout_thrust_n - nominal_thrust_n[:, None]
    )
    step_days, nominal_step = plotter.plot_zero_order_hold_data(
        node_days, nominal_thrust_n
    )
    _, lower_step = plotter.plot_zero_order_hold_data(
        node_days, displayed_lower_n
    )
    _, upper_step = plotter.plot_zero_order_hold_data(
        node_days, displayed_upper_n
    )
    _, rollout_step = plotter.plot_zero_order_hold_data(
        node_days, displayed_rollout_n
    )

    figure, axes = plt.subplots(1, 2, figsize=Plotter.WIDE_FIGSIZE, dpi=Plotter.FIGURE_DPI)
    axis = axes[0]
    rollout_lines = axis.plot(
        step_days,
        rollout_step,
        color=Plotter.DARK_GREY,
        alpha=0.15,
        lw=0.48,
        zorder=1,
    )
    if rollout_lines:
        rollout_lines[0].set_label(f"Monte Carlo ({thrust_scale_label})")
    axis.plot(
        step_days,
        lower_step,
        color=Plotter.BLUE,
        ls="--",
        lw=0.82,
        label=f"Predicted bound ({thrust_scale_label})",
        zorder=2,
    )
    axis.plot(step_days, upper_step, color=Plotter.BLUE, ls="--", lw=0.82, zorder=2)
    axis.plot(
        step_days,
        nominal_step,
        color=Plotter.BLACK,
        lw=1.18,
        label="Thrust magnitude",
        zorder=4,
    )
    axis.axhline(
        case.max_thrust_n,
        color=Plotter.RED,
        ls=":",
        lw=0.88,
        label="Maximum thrust",
    )
    axis.set_xlabel("time [days]")
    axis.set_ylabel("thrust [N]")
    plotted_peak = max(case.max_thrust_n, float(np.max(displayed_upper_n)))
    axis.set_ylim(0.0, 1.06 * plotted_peak)
    plotter._style_2d_axis(
        axis,
        tick_size=Plotter.DIAGNOSTIC_TICK_SIZE,
        label_size=Plotter.DIAGNOSTIC_LABEL_SIZE,
    )
    plotter._legend(
        axis,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor=Plotter.BLACK,
        facecolor="white",
        framealpha=1.0,
        fontsize=6.2,
        borderaxespad=0.0,
    )

    axis = axes[1]
    _, feedback_step = plotter.plot_zero_order_hold_data(
        node_days, np.maximum(feedback_n, 1e-16)
    )
    axis.plot(
        step_days,
        feedback_step,
        lw=0.95,
        color=Plotter.BLACK,
    )
    axis.set_yscale("log")
    axis.set_xlabel("time [days]")
    axis.set_ylabel(r"$\Psi^{-1}_{n_T}(\beta)\,\rho(\Sigma^T_k)$ [N]")
    plotter._style_2d_axis(
        axis,
        grid=True,
        grid_which="both",
        tick_size=Plotter.DIAGNOSTIC_TICK_SIZE,
        label_size=Plotter.DIAGNOSTIC_LABEL_SIZE,
    )

    figure.tight_layout()
    figure.savefig(
        output_prefix.parent / f"{output_prefix.name}_thrust_budget.png",
        dpi=Plotter.FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)

    # 3 - transfer projections with scaled uncertainty layers.
    axes_to_plot = plotter._projection_axes_for_case(case)
    if len(axes_to_plot) == 1:
        figure, axis = plt.subplots(figsize=Plotter.SINGLE_FIGSIZE, dpi=Plotter.FIGURE_DPI)
        projection_axes = [axis]
    else:
        figure, axes = plt.subplots(
            1, len(axes_to_plot), figsize=Plotter.TRIPLE_FIGSIZE, dpi=Plotter.FIGURE_DPI
        )
        projection_axes = list(np.atleast_1d(axes))

    lagrange = deterministic_cr3bp.get_collinear_lagrange_points(case)
    departure_orbit = None
    target_orbit = None
    if case.departure_period_nd is not None:
        departure_orbit = deterministic_cr3bp.propagate_periodic_orbit(
            case, case.x0_augmented_state, case.departure_period_nd
        )
    if case.target_period_nd is not None:
        target_orbit = deterministic_cr3bp.propagate_periodic_orbit(
            case, case.xf_augmented_state, case.target_period_nd
        )

    magnification = plotter.get_projection_magnification(
        solution, normalization
    )
    projection_scale_label = plotter.plot_magnification_label(magnification)
    covariance_stride = max(solution.covariances.shape[0] // 24, 1)
    covariance_nodes = list(range(0, solution.covariances.shape[0], covariance_stride))
    if covariance_nodes[-1] != solution.covariances.shape[0] - 1:
        covariance_nodes.append(solution.covariances.shape[0] - 1)
    control_nd = case.max_thrust_nd * solution.feedforward

    for projection_index, (axis, (axis_0, axis_1)) in enumerate(zip(projection_axes, axes_to_plot)):
        plotter._plot_projection(
            axis,
            case,
            solution.means,
            control_nd,
            departure_orbit,
            target_orbit,
            lagrange,
            axis_0,
            axis_1,
        )
        displayed_0 = solution.means[axis_0, None, :] + magnification * (
            rollout_history[axis_0] - solution.means[axis_0, None, :]
        )
        displayed_1 = solution.means[axis_1, None, :] + magnification * (
            rollout_history[axis_1] - solution.means[axis_1, None, :]
        )
        rollout_lines = axis.plot(
            displayed_0.T,
            displayed_1.T,
            color=Plotter.DARK_GREY,
            alpha=0.15,
            lw=0.38,
            zorder=1,
        )
        if projection_index == 0 and rollout_lines:
            rollout_lines[0].set_label(f"Monte Carlo ({projection_scale_label})")

        indices = [axis_0, axis_1]
        physical_scale = normalization.scale[indices]
        for covariance_index, node in enumerate(covariance_nodes):
            block = solution.covariances[node][np.ix_(indices, indices)]
            block = block * np.outer(physical_scale, physical_scale)
            points = magnification * plotter.plot_covariance_ellipse_points(
                block, 3.0
            )
            axis.plot(
                solution.means[axis_0, node] + points[0],
                solution.means[axis_1, node] + points[1],
                color=Plotter.RED,
                lw=0.80,
                alpha=0.90,
                zorder=2,
                label=f"Predicted covariance ({projection_scale_label})"
                if projection_index == 0 and covariance_index == 0
                else "_nolegend_",
            )

    is_single_projection = len(axes_to_plot) == 1
    plotter._figure_legend(
        figure,
        projection_axes,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90 if is_single_projection else 0.995),
        frameon=True,
        fancybox=False,
        edgecolor=Plotter.BLACK,
        facecolor="white",
        framealpha=1.0,
        fontsize=6.7,
        columnspacing=0.85,
        handlelength=1.5,
        borderpad=0.30,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.70 if is_single_projection else 0.80))
    figure.savefig(
        output_prefix.parent / f"{output_prefix.name}_projections.png",
        dpi=Plotter.FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)

    # 4 - six-dimensional terminal position-velocity distribution matrix.
    plotter.plot_terminal_distribution(
        case,
        options,
        solution,
        normalization,
        monte_carlo_result,
    )


def save_outputs(
    case: TestCase,
    options: Options,
    ref_traj: ReferenceTraj,
    solution: RobustSolution,
    normalization: Normalization,
    monte_carlo_result: dict,
    diagnostics: dict[str, float],
    psi_inv: float,
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    node_days = ref_traj.node_times * case.time_unit / 86400.0

    # Physical gains, undoing the normalisation.
    physical_gains = (
        case.max_thrust_nd * solution.gains / normalization.scale[None, None, 0:NP]
    )
    sigma_physical = std_physical_units(case, normalization, solution.covariances)

    np.savez(
        output_prefix.with_suffix(".npz"),
        node_times_nd=ref_traj.node_times,
        node_days=node_days,
        steps_nd=ref_traj.steps,
        means=solution.means,
        feedfoward_normalized=solution.feedforward,
        feedforward_n=solution.feedforward * case.max_thrust_nd * case.thrust_unit,
        gains_normalized=solution.gains,
        gains=physical_gains,
        radius_normalized=solution.radius,
        covariance_normalized=solution.covariances,
        control_covariance_normalized=solution.control_covariances,
        sigma_physical=sigma_physical,
        normalization_scale_nd=normalization.scale,
        initial_state_std_nd=normalization.initial_std,
        position_covariance_reduction=options.position_covariance_reduction,
        velocity_covariance_reduction=options.velocity_covariance_reduction,
        position_target_ratio=1.0 / options.position_covariance_reduction,
        velocity_target_ratio=1.0 / options.velocity_covariance_reduction,
        target_ratio=np.array(
            [1.0 / options.position_covariance_reduction] * 3
            + [1.0 / options.velocity_covariance_reduction] * 3,
            dtype=float,
        ),
        psi_inv=psi_inv,
        monte_carlo_cumulative=monte_carlo_result["cumulative_control_effort"],
        monte_carlo_violation_fraction=monte_carlo_result["violation_fraction"],
        monte_carlo_terminal_covariance=monte_carlo_result["terminal_covariance"],
        diagnostics=np.array(diagnostics, dtype=object),
    )

    feedforward_magnitude_n = np.linalg.norm(solution.feedforward, axis=0) * case.max_thrust_nd * case.thrust_unit
    feedback_magnitude_n = psi_inv * solution.radius * case.max_thrust_nd * case.thrust_unit
    table = np.column_stack(
        [
            node_days[:-1],
            ref_traj.steps,
            solution.means[:, :-1].T,
            solution.feedforward.T * case.max_thrust_nd * case.thrust_unit,
            feedforward_magnitude_n,
            feedback_magnitude_n,
            feedforward_magnitude_n + feedback_magnitude_n,
            sigma_physical[:-1, 0:3],
            sigma_physical[:-1, 3:6],
            sigma_physical[:-1, 6:7],
        ]
    )
    header = (
        "t_days,dt_nd,x,y,z,vx,vy,vz,m_nd,Sx_N,Sy_N,Sz_N,S_norm_N,feedback_N,total_N,"
        "sigma_x_m,sigma_y_m,sigma_z_m,sigma_vx_ms,sigma_vy_ms,sigma_vz_ms,sigma_m_kg"
    )
    np.savetxt(output_prefix.with_suffix(".csv"), table, delimiter=",", header=header, comments="")

    plot_outputs(
        case,
        options,
        ref_traj,
        solution,
        normalization,
        monte_carlo_result,
        psi_inv,
        output_prefix,
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_test_case(test_case_id: str, options: Options | None = None) -> dict[str, float]:
    case = CASE_REGISTRY[test_case_id]()
    log_prefix = f"[{test_case_id}] "
    psi_inv = psi_inverse(NU, options.violation_parameter)

    ref_traj = load_ref_traj(case, options)
    normalization = build_normalization(case)
    dynamics = Dynamics(case)

    arc_functions = _build_arc_functions(case, options, dynamics, normalization, ref_traj)

    initial_guess = build_initial_guess(
        case,
        options,
        dynamics,
        normalization,
        ref_traj,
        arc_functions,
        psi_inv,
    )

    solution = solve_rocp(
        case, options, ref_traj, arc_functions, initial_guess, normalization, psi_inv
    )


    monte_carlo_result = monte_carlo_rollouts(case, options, dynamics, normalization, solution, ref_traj.steps)

    diagnostics = compute_diagnostics(
        case,
        options,
        ref_traj,
        solution,
        normalization,
        monte_carlo_result,
        psi_inv,
    )
    print_summary(case, diagnostics, psi_inv)

    output_prefix = OUTPUT_DIR / case.test_case_id
    save_outputs(
        case, options, ref_traj, solution, normalization,
        monte_carlo_result, diagnostics, psi_inv, output_prefix,
    )
    return diagnostics


def _build_arc_functions(
    case: TestCase,
    options: Options,
    dynamics: Dynamics,
    normalization: Normalization,
    ref_traj: ReferenceTraj,
) -> list[casadi.Function]:
    """One arc function is built per distinct duration, reused across arcs.

    The deterministic mesh is refined by h-adaptive method, so only a handful of distinct step
    sizes occur; sharing the functions keeps the expression graph small.
    """

    cache: dict[float, casadi.Function] = {}
    arc_functions: list[casadi.Function] = []
    for step in ref_traj.steps:
        key = float(np.round(step, 14))
        if key not in cache:
            cache[key] = get_arc_function(case, options, dynamics, normalization, float(step), len(cache))
        arc_functions.append(cache[key])
    return arc_functions


def main() -> None:
    options = Options()
    for test_case_id in ("lyapunov_l1_to_l2", "halo_l2_to_halo_l1"):
        run_test_case(test_case_id, options)


if __name__ == "__main__":
    main()
