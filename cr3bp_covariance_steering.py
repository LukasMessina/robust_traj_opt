"""
Chance-constrained covariance steering for Earth-Moon CR3BP low-thrust transfers test cases.

The mean trajectory and covariance are both transcribed by multiple shooting: the node means
and node covariance factors are treated as decision variables and are tied across consecutive
arcs by mean and covariance matching conditions. The covariance is parameterized through Cholesky
factors, ensuring that the node covariances are positive semi-definite by construction.

Uncertainty is propagated by the Unscented Transform with
kappa = 0 and a lower-triangular Cholesky factor. The sigma points are integrated
with a fixed-step integrator built directly as a CasADi expression, so the optimiser sees
through the integration and supplies exact first and second derivatives.

Configuration: one Gaussian component (no GMM split, hence a single control
policy), navigation error (R_bar, H = I), deterministic dynamics (Q_k = 0).
The initial guess is the energy-optimal solution restricted to its knot
points, thus the interval durations are uniform.
The stochastic running objective is the regularized mean-control plus
Bryson-weighted state- and control-covariance traces, with no terminal Qf cost.
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
import integrator
from deterministic_cr3bp import CASE_REGISTRY, TestCase
from plotter import Plotter 

NX = 7                       # augmented state dimension
NU = 3                       # control dimension
NP = 6                       # primed (position-velocity) state dimension

REFERENCE_DIR = Path("output/cr3bp_energy_optimal")
OUTPUT_DIR = Path("output/cr3bp_covariance_steering")

# Initial position and velocity standard deviations per test case [-].
# Order: (sigma_x, sigma_y, sigma_z, sigma_xdot, sigma_ydot, sigma_zdot).
INITIAL_STATE_STD_ND: dict[
    str, tuple[float, float, float, float, float, float]
] = {
    "halo_l2_to_halo_l1": (
        1e-5,  # sigma_x
        1e-5,  # sigma_y
        1e-5,  # sigma_z
        1e-4,  # sigma_xdot
        1e-4,  # sigma_ydot
        1e-4,  # sigma_zdot
    ),
    "nrho_l2_to_dro": (
        1e-5,  # sigma_x
        1e-5,  # sigma_y
        1e-5,  # sigma_z
        1e-4,  # sigma_xdot
        1e-4,  # sigma_ydot
        1e-4,  # sigma_zdot
    ),
    "lyapunov_l1_to_l2": (
        1e-5,  # sigma_x
        1e-5,  # sigma_y
        1e-5,  # sigma_z
        1e-4,  # sigma_xdot
        1e-4,  # sigma_ydot
        1e-4,  # sigma_zdot
    ),
}
# Equal-duration arcs per test case, at five arcs per day. The two transfers
# have different times of flight -- 21.2 and 20 days -- so a single count would
# give them different arc durations.
UNIFORM_ARCS_BY_CASE: dict[str, int] = {
    "nrho_l2_to_dro": 128,      
    "halo_l2_to_halo_l1": 120,  
    "lyapunov_l1_to_l2": 72,   
}

# Uniform gain used when the Riccati warm start is disabled.
COLD_START_GAIN = 1e-3

INITIAL_MASS_STD = 1e-3      # [kg]
MASS_SCALE = 1e-3            # [kg]

@dataclass(frozen=True)
class Options:
    """Solver and transcription settings."""

    # Number of equal-duration arcs the reference is resampled onto.
    # NOTE: Node states come from the dense trajectory the deterministic
    # solve saves, thus a node between two collocation knots is still an
    # integrated point of the reference rather than an interpolation of it.
    uniform_mesh_arcs: int | dict[str, int] = field(
        default_factory=lambda: dict(UNIFORM_ARCS_BY_CASE)
    )
    # When set, restrict the uniform mesh arcs to the first specified number of arcs
    # and use the state at that relative endpoint as the terminal
    # mean target. None preserves the complete uniform mesh.
    truncated_uniform_mesh_arcs: int | None = None
    # Control-norm regularizer: the thrust magnitude is carried as
    # sqrt(u'u + eps_1^2) on the up-to-the-unit control, everywhere it appears.
    control_norm_eps: float = 1e-6
    bryson_sigma_factor: float = 3.0
    integrator_substeps: int = 1

    violation_parameter: float = 0.01   
    # Terminal covariance reduction factors relative to the corresponding
    # initial covariance blocks. They may be set independently.
    position_covariance_reduction: float = 1e4
    velocity_covariance_reduction: float = 1e4
    scaling_parameter: float = 0.0
    # Navigation (measurement) error covariance as a fraction of the initial state
    # covariance: R_bar = navigation_error_ratio * Sigma_0. Zero disables it and
    # recovers the 2*n_x+1 sigma-point scheme
    navigation_error_ratio: float = 1e-4

    # Floors the covariance at jitter / n_x, which must stay far below the
    # terminal target while covering the negative eigenvalues of order 1e-15 that
    # round-off leaves behind once the steering contracts a direction.
    cholesky_jitter: float = 1e-12
    spectral_radius_floor: float = 1e-8

    # Floor on diag(G) in the terminal covariance constraint (see solve_rocp):
    # I_6 - Dt^-1/2 P'_N Dt^-1/2 = G G^T, G lower-triangular. Any PSD left-hand
    # side admits a factor with strictly positive pivots; it
    # exists purely to keep G -> G G^T non-degenerate near diag(G) = 0, where the
    # constraint Jacobian would otherwise lose rank.
    terminal_margin_floor: float = 1e-4


    # Warm-start the feedback gains from the Bryson-rule Riccati recursion. With
    # this disabled the gains start at zero, the covariance grows open-loop, and
    # the terminal constraint begins many orders of magnitude violated. Starting
    # with a zero gain is still recoverable, but consumes a lot of additional iterations.
    warm_start_gains: bool = True

    solver: str = "snopt"

    # SNOPT settings
    major_max_iter: int = 50000
    minor_max_iter: int = 100 * major_max_iter
    major_optimality_tol: float = 1e-5
    major_feasibility_tol: float = 1e-8
    minor_feasibility_tol: float = 1e-8
    # SNOPT partial pricing. The user guide recommends raising it when the
    # problem has many more variables than constraints, and suggests the number
    # of time stages for time-staged models; this file optimises a full feedback
    # gain per arc, so `n_arcs` is the natural value. None leaves SNOPT's own
    # default of 1.
    partial_price: int | None = None
    # Initial weight for SNOPT's nonlinear elastic-mode constraint
    # relaxations. Setting this to None leaves SNOPT's native default.
    elastic_weight: float | None = None
    expand_graph: bool = False
    print_level: int = 1
    summary_file: int = 6

    monte_carlo_samples: int = 50000
    monte_carlo_seed: int = 42


def psi_inverse(dimension: int, beta: float) -> float:
    """Psi_d^-1(beta) = sqrt(Phi_d^-1(1 - beta)) with Phi_d the chi-squared CDF."""

    return float(np.sqrt(chi2.ppf(1.0 - beta, dimension)))


def bryson_running_cost_weights(options: Options) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized Bryson weights ``Q = I_6 / c^2`` and ``R = I_3``.

    Bryson's rule weights each channel by the inverse square of its largest
    acceptable deviation: ``c`` initial standard deviations for the state and
    the maximum thrust for the control. In the normalised coordinates the NLP
    runs in (``dx = D0 dx~``, ``dT = T_max du~``) that transformation leaves

        Q~ = D0 Q D0 = (1 / c^2) I_6,   R~ = T_max^2 R = I_3,

    which is the unrescaled normalised form returned here. The mass channel is
    deliberately excluded.
    """

    sigma_factor = float(options.bryson_sigma_factor)
    if sigma_factor <= 0.0:
        raise ValueError("The sigma factor associated with the Bryson's"
        "rule derived weights must be positive")
    return np.eye(NP) / sigma_factor**2, np.eye(NU)


def leaked_mean_control_norm(control, epsilon: float):
    """``sqrt(u'u + eps_1^2)``, the smooth stand-in for ``||u||_2``.

    Applied to the up-to-the-unit control, so ``eps_1`` is a fraction of the
    thrust bound. See ``Options.control_norm_eps``.
    """

    return casadi.sqrt(casadi.dot(control, control) + epsilon**2)


def composite_running_cost(
    mean_control_norm,
    state_covariance,
    control_covariance,
    state_weight,
    control_weight,
):
    """Regularized mean-control cost plus Bryson's rule derived control and state covariance traces."""

    return (
        mean_control_norm
        + casadi.trace(state_weight @ state_covariance[0:NP, 0:NP])
        + casadi.trace(control_weight @ control_covariance)
    )


def build_symbolic_lower_triangular_matrix(entries, dimension: int):
    """Assemble a lower-triangular CasADi matrix from row-major entries."""

    indices = [
        (row, column)
        for row in range(dimension)
        for column in range(row + 1)
    ]
    rows = [[casadi.MX(0.0)] * dimension for _ in range(dimension)]
    for (row, column), entry in zip(indices, casadi.vertsplit(entries)):
        rows[row][column] = entry
    return casadi.vertcat(*[casadi.horzcat(*row) for row in rows])


def state_covariance_node_scaling(
    seed_covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed node-dependent diagonal scales from a state covariance seed."""

    state_covariances = np.asarray(seed_covariances, dtype=float)
    if state_covariances.ndim != 3 or state_covariances.shape[1] != state_covariances.shape[2]:
        raise ValueError("state covariances must have shape (nodes, nx, nx)")

    variances = np.maximum(
        np.diagonal(state_covariances, axis1=1, axis2=2),
        1e-12,
    )
    standard_deviations = np.sqrt(variances)
    return variances, standard_deviations, 1.0 / standard_deviations


def state_covariance_cholesky_factor_seed(matrix: np.ndarray) -> np.ndarray:
    """Return an exact Cholesky factor of the state covariance matrix ."""

    symmetric_matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    cholesky_factor = np.linalg.cholesky(symmetric_matrix)
    return cholesky_factor



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
    """Unscented transform weights."""

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
    """Batched CR3BP dynamics and the integration arc map with its exact Jacobians."""

    def __init__(self, case: TestCase) -> None:
        state = casadi.SX.sym("state", NX)
        control = casadi.SX.sym("control", NU)
        eom = deterministic_cr3bp.eom(
            case, state, control, casadi.sqrt(casadi.dot(control, control))
        )
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
        # A second dynamics/RHS function taking the thrust magnitude explicitly,
        # used by the deterministic reference reoptimization, which carries an
        # regularized control norm. The unscented transform cannot use it, for the reason
        # given above.
        magnitude = casadi.SX.sym("sigma")
        self._augm_state_derivatives_with_regularization = casadi.Function(
            "slack_derivatives",
            [state, control, magnitude],
            [deterministic_cr3bp.eom(case, state, control, magnitude)],
        )
        self._augm_state_derivatives_maps: dict[int, casadi.Function] = {}
        self._full_derivatives_maps: dict[int, casadi.Function] = {}

    def propagate_with_regularization(self, states, controls, magnitude, step: float, substeps: int):
        """Fixed-step RK7 with the mass flow driven by an explicit magnitude.

        `magnitude` is held constant across the arc, as the control is.
        """

        substep = step / substeps
        propagated = states
        for _ in range(substeps):
            stages = []
            for i in range(integrator.RK7_STAGES):
                stage_state = propagated
                for j in range(i):
                    coefficient = integrator.RK7_A[i, j]
                    if coefficient != 0.0:
                        stage_state = stage_state + substep * coefficient * stages[j]
                stages.append(
                    self._augm_state_derivatives_with_regularization(stage_state, controls, magnitude)
                )
            for i in range(integrator.RK7_STAGES):
                if integrator.RK7_B[i] != 0.0:
                    propagated = (
                        propagated + substep * integrator.RK7_B[i] * stages[i]
                    )
        return propagated

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
        """Fixed-step RK7 over one arc for symbolic or numerical inputs.

        With MX/SX inputs, the RK47map is built natively from a compact mapped SX
        dynamics function. CasADi can therefore differentiate the returned map to
        any required order.

        With numerical inputs, the same RK7 stages are evaluated numerically. If
        ``with_jacobian`` is true, the exact partial derivatives of the discrete
        RK7 map are accumulated through the stages and substeps. The control is
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

        a_matrix, b_weights = integrator.RK7_A, integrator.RK7_B
        n_stages = integrator.RK7_STAGES

        for _ in range(substeps):
            if with_jacobian:
                # Differentiate the Runge-Kutta step stage by stage. For stage i
                # the state is x + h * sum_j a[i,j] k_j, so its derivatives are
                #   dk_i/dx = A_i (I + h sum_j a[i,j] dk_j/dx)
                #   dk_i/du = B_i + A_i (h sum_j a[i,j] dk_j/du)
                # with A_i, B_i the right-hand-side Jacobians at that stage.
                stages, d_state, d_control = [], [], []
                for i in range(n_stages):
                    stage_state = propagated
                    for j in range(i):
                        if a_matrix[i, j] != 0.0:
                            stage_state = (
                                stage_state + substep * a_matrix[i, j] * stages[j]
                            )
                    k_i, a_i, b_i = self.augm_state_derivatives_and_jacobians(
                        stage_state, controls
                    )
                    inner_state = np.zeros((k_i.shape[1], NX, NX))
                    inner_control = np.zeros((k_i.shape[1], NX, NU))
                    for j in range(i):
                        if a_matrix[i, j] != 0.0:
                            inner_state = (
                                inner_state + a_matrix[i, j] * d_state[j]
                            )
                            inner_control = (
                                inner_control + a_matrix[i, j] * d_control[j]
                            )
                    stages.append(k_i)
                    d_state.append(a_i @ (identity + substep * inner_state))
                    d_control.append(b_i + a_i @ (substep * inner_control))

                step_state = np.broadcast_to(
                    identity, (stages[0].shape[1], NX, NX)
                ).copy()
                step_control = np.zeros((stages[0].shape[1], NX, NU))
                for i in range(n_stages):
                    if b_weights[i] != 0.0:
                        step_state = step_state + substep * b_weights[i] * d_state[i]
                        step_control = (
                            step_control + substep * b_weights[i] * d_control[i]
                        )
                control_sensitivity = step_state @ control_sensitivity + step_control
                state_sensitivity = step_state @ state_sensitivity
            else:
                derivative = mapped if symbolic else self.augm_state_derivatives
                stages = []
                for i in range(n_stages):
                    stage_state = propagated
                    for j in range(i):
                        if a_matrix[i, j] != 0.0:
                            stage_state = (
                                stage_state + substep * a_matrix[i, j] * stages[j]
                            )
                    stages.append(derivative(stage_state, controls))

            for i in range(n_stages):
                if b_weights[i] != 0.0:
                    propagated = propagated + substep * b_weights[i] * stages[i]

        return propagated, state_sensitivity, control_sensitivity


def cholesky_lower_triangular_matrix(matrix, dimension: int, pivot_floor: float = 1e-11):                     
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


def max_eigval_sym_3by3(matrix, relative_floor: float = 1e-20):
    """Closed-form largest eigenvalue of a symmetric 3x3 matrix.
    Ref: https://dl.acm.org/doi/pdf/10.1145/355578.366316

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
    p = casadi.sqrt(p2 / 6.0 + (relative_floor) ** 2 )
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
    """Energy-optimal deterministic solution restricted to a subset of its knot points."""

    node_times: np.ndarray   # [-] non-dimensional, first entry zero
    steps: np.ndarray        # [-] arc durations
    states: np.ndarray       # [-] (NX, N + 1)
    controls: np.ndarray     # [-] (NU, N) thrust, zero-order hold
    fuel_consumed: float     # [kg]

    @property
    def n_arcs(self) -> int:
        return self.steps.size


def uniform_arc_count(case: TestCase, options: Options) -> int:
    """Equal-duration arc count for `case`."""

    requested = options.uniform_mesh_arcs
    if isinstance(requested, dict):
        value = requested.get(case.test_case_id)
        if value is None:
            raise ValueError(
                "Options.uniform_mesh_arcs has no entry for "
                f"'{case.test_case_id}'; every case needs an arc count "
                "because the mesh must have equal-duration arcs"
            )
        count = int(value)
    else:
        count = int(requested)
    if count < 1:
        raise ValueError("uniform_mesh_arcs must be positive")
    return count


def _build_uniform_ref_traj(
    case: TestCase,
    options: Options,
    data,
    mesh: np.ndarray,
    states: np.ndarray,
) -> ReferenceTraj:
    """Resample the saved reference onto arcs of equal duration.

    The control is zero-order held on the dense grid, so each arc takes the
    duration-weighted mean of the dense controls it spans.

    `Options.truncated_uniform_mesh_arcs` truncates this uniform mesh, keeping its first N
    arcs. 
    """

    n_arcs = uniform_arc_count(case, options)
    if n_arcs is None or n_arcs < 1:
        raise ValueError("uniform_mesh_arcs must be positive or None")

    requested_arcs = options.truncated_uniform_mesh_arcs
    if requested_arcs is not None:
        requested_arcs = int(requested_arcs)
        if requested_arcs < 1:
            raise ValueError("the requested arc cound for the truncation must be positive or None")
        if requested_arcs > n_arcs:
            raise ValueError(
                f"the requested arc count fr the truncation is ={requested_arcs} and exceeds the {n_arcs} uniform "
                f"arcs available for {case.test_case_id}"
            )

    dense_days = np.asarray(data["t_dense_days"], dtype=float)
    dense_states = np.asarray(data["x_dense"], dtype=float)
    dense_controls = np.asarray(data["u_dense"], dtype=float)

    unique_days, unique_index = np.unique(dense_days, return_index=True)
    dense_times = unique_days * 86400.0 / case.time_unit
    unique_states = dense_states[:, unique_index]

    horizon = float(mesh[-1]) * case.tof_nd
    if dense_times[-1] < horizon - 1e-12:
        raise ValueError("the dense reference stops before the requested horizon")

    # Build the full uniform mesh, then keep the first truncation arcs of it, so
    # truncation shortens the horizon without changing the arc duration.
    node_times = np.linspace(0.0, horizon, n_arcs + 1)
    if requested_arcs is not None:
        node_times = node_times[: requested_arcs + 1]
        n_arcs = requested_arcs
    node_states = np.empty((NX, n_arcs + 1), dtype=float)
    for row in range(NX):
        node_states[row] = np.interp(node_times, dense_times, unique_states[row])
    # Pin the endpoints to the stored states so the boundary conditions are the
    # deterministic solver's own, not a resampling of them.
    node_states[:, 0] = states[:, 0]
    if requested_arcs is None:
        node_states[:, -1] = states[:, -1]

    edge_times = dense_days * 86400.0 / case.time_unit
    segment_starts = edge_times[:-1]
    segment_ends = edge_times[1:]
    segment_controls = dense_controls[:, :-1]
    node_controls = np.empty((NU, n_arcs), dtype=float)
    for k in range(n_arcs):
        lo, hi = node_times[k], node_times[k + 1]
        overlap = np.clip(
            np.minimum(segment_ends, hi) - np.maximum(segment_starts, lo), 0.0, None
        )
        total = overlap.sum()
        node_controls[:, k] = segment_controls @ overlap / total

    if requested_arcs is None:
        history = data["history"]
        fuel_consumed = (
            float(history[-1]["fuel_consumed_kg"]) 
        )
    else:
        fuel_consumed = float(case.m0_wet * (node_states[6, 0] - node_states[6, -1]))

    return ReferenceTraj(
        node_times=node_times,
        steps=np.diff(node_times),
        states=node_states,
        controls=node_controls,
        fuel_consumed=fuel_consumed,
    )


def load_ref_traj(case: TestCase, options: Options) -> ReferenceTraj:
    source_path = REFERENCE_DIR / f"{case.test_case_id}.npz"
    data = np.load(source_path, allow_pickle=True)
    mesh = np.asarray(data["mesh_fraction"], dtype=float)
    states = np.asarray(data["x"], dtype=float)
    return _build_uniform_ref_traj(case, options, data, mesh, states)


def terminal_mean_state_target(
    case: TestCase,
    options: Options,
    ref_traj: ReferenceTraj,
) -> np.ndarray:
    """Return the complete-transfer or relative short-horizon mean target."""

    if options.truncated_uniform_mesh_arcs is None:
        return np.asarray(case.xf_state, dtype=float)
    return np.asarray(ref_traj.states[0:NP, -1], dtype=float)


def configure_snopt(opti: casadi.Opti, options: Options) -> None:
    """Apply the common SNOPT settings used by deterministic and stochastic solves."""

    # NOTE: CasADi's SNOPT interface tries numeric options as integers before
    # trying them as reals. Pass fractional tolerances as strings so they reach
    # SNOPT through its generic real-valued parameter parser.
    snopt_options = {
        "Major iterations limit": options.major_max_iter,
        "Minor iterations limit": options.minor_max_iter,
        "Iterations limit": options.minor_max_iter,
        "Major optimality tolerance": f"{options.major_optimality_tol:.13g}",
        "Major feasibility tolerance": f"{options.major_feasibility_tol:.13g}",
        "Minor feasibility tolerance": f"{options.minor_feasibility_tol:.13g}",
        "Print file": 0,
        "Summary file": options.summary_file,
        "Major print level": options.print_level,
        "Minor print level": 0,
        "Function_precision": 1e-12,
    }
    if options.partial_price is not None:
        partial_price = int(options.partial_price)
        if partial_price < 1:
            raise ValueError("The partial price parameter must be a positive integer")
        # Integer-valued in SNOPT, so it needs no string round-trip.
        snopt_options["Partial price"] = partial_price
    if options.elastic_weight is not None:
        elastic_weight = float(options.elastic_weight)
        if not np.isfinite(elastic_weight) or elastic_weight <= 0.0:
            raise ValueError("The elastic weight must be finite and positive")
        snopt_options["Elastic weight"] = f"{elastic_weight:.13g}"
    opti.solver(
        "snopt",
        {"expand": options.expand_graph, "print_time": True},
        snopt_options,
    )


def reoptimize_reference_trajectory(
    case: TestCase,
    options: Options,
    source: ReferenceTraj,
    dynamics: Dynamics,
) -> ReferenceTraj:
    """Reoptimize the full or truncated energy reference on the NLP integration map.

    A full transfer uses the exact case terminal position/velocity; a truncated
    transfer uses the saved relative endpoint. Terminal mass remains free, and
    the objective is the normalized control-energy integral with the regularized control norm.
    """

    control_norm_eps = float(options.control_norm_eps)
    if control_norm_eps <= 0.0:
        raise ValueError("control_norm_eps must be positive")

    n_arcs = source.n_arcs
    opti = casadi.Opti()
    means = opti.variable(NX, n_arcs + 1)
    normalized_controls = opti.variable(NU, n_arcs)
    opti.subject_to(means[:, 0] == case.x0_augmented_state)

    objective = 0.0
    for k in range(n_arcs):
        # Leaked thrust magnitude: carries the mass flow, the thrust bound and
        # the running cost, in place of the epigraph slack.
        mean_control_norm = leaked_mean_control_norm(
            normalized_controls[:, k], control_norm_eps
        )
        propagated = dynamics.propagate_with_regularization(
            means[:, k],
            case.max_thrust_nd * normalized_controls[:, k],
            case.max_thrust_nd * mean_control_norm,
            float(source.steps[k]),
            options.integrator_substeps,
        )
        opti.subject_to(means[:, k + 1] == propagated)
        opti.subject_to(mean_control_norm <= 1.0)
        # Control energy on the leaked magnitude, matching the mass flow.
        objective = objective + float(source.steps[k]) * mean_control_norm**2

    opti.subject_to(
        means[0:NP, -1]
        == casadi.DM(terminal_mean_state_target(case, options, source))
    )
    opti.minimize(objective)
    opti.set_initial(means, source.states)
    opti.set_initial(
        normalized_controls,
        source.controls / case.max_thrust_nd,
    )
    configure_snopt(opti, options)

    try:
        solution = run_snopt_solver(
            opti, f"{case.test_case_id}-deterministic_trajectory-{n_arcs}-arcs"
        )
        status = "converged"
    except RuntimeError as error:
        print(
            f"[{case.test_case_id}:deterministic_trajectory] SNOPT did not converge "
            f"({error}); returning the last iterate.",
            flush=True,
        )
        solution = opti.debug
        status = "failed"

    solved_states = np.asarray(solution.value(means), dtype=float)
    solved_normalized_controls = np.asarray(
        solution.value(normalized_controls), dtype=float
    ).reshape(NU, n_arcs)
    solved_controls = case.max_thrust_nd * solved_normalized_controls
    mass_consumed = float(
        case.m0_wet * (solved_states[6, 0] - solved_states[6, -1])
    )
    print(
        f"[{case.test_case_id}] minimum-energy deterministic trajectory ({n_arcs} arcs): "
        f"{status}, max |u|/u_max="
        f"{np.max(np.linalg.norm(solved_normalized_controls, axis=0)):.6f}",
        flush=True,
    )
    return ReferenceTraj(
        node_times=source.node_times.copy(),
        steps=source.steps.copy(),
        states=solved_states,
        controls=solved_controls,
        fuel_consumed=mass_consumed,
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
    state_weight: float,
    control_weight: float,
    terminal_weight: float | np.ndarray,
    state_matrices: list[np.ndarray],
    control_matrices: list[np.ndarray],
) -> np.ndarray:
    """Backward Riccati recursion on the normalised state/control subsystem.

    The propagated dynamics include mass as a seventh augmented state, but the
    feedback law observes only the six position/velocity deviations.  Restrict
    the linearisation before starting the recursion so mass dynamics and
    state--mass coupling cannot influence the initial feedback gains.
    """

    Q_normalized = state_weight * np.eye(NP)
    R_normalized = control_weight * np.eye(NU)
    terminal_weights = np.asarray(terminal_weight, dtype=float)
    if terminal_weights.ndim == 0:
        terminal_weights = np.full(NP, float(terminal_weights))
    elif terminal_weights.shape != (NP,):
        raise ValueError(f"terminal_weight must be a scalar or have shape ({NP},).")
    P_normalized = np.diag(terminal_weights)
    gains = np.zeros((len(state_matrices), NU, NP), dtype=float)

    for k in reversed(range(len(state_matrices))):
        state_matrix = state_matrices[k][0:NP, 0:NP]
        control_matrix = control_matrices[k][0:NP, :]
        gain = np.linalg.solve(
            R_normalized + control_matrix.T @ P_normalized @ control_matrix,
            control_matrix.T @ P_normalized @ state_matrix,
        )
        gains[k] = -gain
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
    `Qf/Q` ratio equal its requested reduction. These normalised weights are
    used directly, in the same unrescaled form as
    `bryson_running_cost_weights`.

    This cost and the Riccati recursion are defined only on the six-dimensional
    position/velocity subsystem. Mass remains part of nonlinear moment
    propagation, but it is not observed by the feedback law and therefore must
    not affect the gain seed.

    """

    sigma_factor = float(options.bryson_sigma_factor)
    if sigma_factor <= 0.0:
        raise ValueError("The sigma factor associated with the Bryson's"
        "rule derived weights must be positive")
    state_weight = 1.0 / sigma_factor**2
    control_weight = 1.0
    position_reduction = float(options.position_covariance_reduction)
    velocity_reduction = float(options.velocity_covariance_reduction)
    terminal_weights = state_weight * np.array(
        [position_reduction] * 3
        + [velocity_reduction] * 3,
        dtype=float,
    )
    gains = tvlqr_gains(
        state_weight,
        control_weight,
        terminal_weights,
        state_matrices,
        control_matrices,
    )
    return gains, (control_weight, terminal_weights)


@dataclass
class InitialGuess:
    means: np.ndarray
    feedforward: np.ndarray
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
) -> InitialGuess:
    """Warm start from the energy optimal reference trajectory and feedback-gain seed.

    The feedforward controls are exactly the normalized controls coming from the energy-optimal deterministictrajectory and
    the gains come from the TVLQR solution. Covariance is then
    propagated once from that combination. 
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

    _, covariances, control_covariances = propagate_stochastic_moments(
        arc_functions,
        means,
        feedforward,
        gains,
        normalization.initial_covariance,
    )
    radius = psqrt_spectral_radii(control_covariances)

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
    # Fixed per-node diagonal covariance scales the Cholesky factors are expressed in.
    covariance_scale_variances: np.ndarray = field(
        default_factory=lambda: np.empty((0, NX), dtype=float)
    )
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
    control_norm_eps = float(options.control_norm_eps)
    if control_norm_eps <= 0.0:
        raise ValueError("control_norm_eps must be positive")
    # The running cost sums the per-arc terms unweighted, which is a pure
    # rescaling of the time integral only while every arc lasts the same. On a
    # non-uniform mesh it would instead reweight the problem, counting a short
    # arc as heavily as a long one. Fail loudly rather than silently optimise a
    # different objective.
    step_spread = float(np.max(ref_traj.steps) - np.min(ref_traj.steps))
    if step_spread > 1e-9 * float(np.max(ref_traj.steps)):
        raise ValueError(
            "solve_rocp assumes equal-duration arcs because the running cost "
            f"carries no arc duration, but the mesh spans {np.min(ref_traj.steps):.6e} "
            f"to {np.max(ref_traj.steps):.6e}. Set Options.uniform_mesh_arcs, or "
            "restore the per-arc duration weighting in the objective."
        )
    position_target = 1.0 / options.position_covariance_reduction
    velocity_target = 1.0 / options.velocity_covariance_reduction
    # Seed the node covariances by propagating the initial guess open-loop
    # through the same arc functions the NLP uses, then build the fixed
    # per-node diagonal scales the Cholesky factors are expressed in.
    _, seed_covariances, _ = propagate_stochastic_moments(
        arc_functions,
        initial_guess.means,
        initial_guess.feedforward,
        initial_guess.gains,
        normalization.initial_covariance,
    )
    (
        covariance_scale_variances,
        covariance_scale_std,
        covariance_scale_inv_std,
    ) = state_covariance_node_scaling(
        seed_covariances,
    )

    opti = casadi.Opti()
    means = opti.variable(NX, n_arcs + 1)
    feedforward = opti.variable(NU, n_arcs)
    gains = [opti.variable(NU, NP) for _ in range(n_arcs)]

    opti.subject_to(means[:, 0] == case.x0_augmented_state)

    cholesky_factor_entries = []
    node_covariances = [casadi.DM(normalization.initial_covariance)]
    n_cholesky_factor_entries = NX * (NX + 1) // 2
    for node in range(1, n_arcs + 1):
        entries = opti.variable(n_cholesky_factor_entries)
        factor = build_symbolic_lower_triangular_matrix(entries, NX)
        # The diagonal is bounded below by zero to fix the sign of each factor
        # column: P = L L^T is invariant under flipping the sign of any column,
        # so without this the covariance variables are not identifiable.
        opti.subject_to(casadi.diag(factor) > 0.0)
        cholesky_factor_entries.append(entries)
        normalized_state_covariance = factor @ factor.T
        scale = casadi.DM(np.diag(covariance_scale_std[node]))
        node_covariances.append(
            scale @ normalized_state_covariance @ scale
        )

    bryson_state_weight, bryson_control_weight = bryson_running_cost_weights(
        options
    )
    Q_running = casadi.DM(bryson_state_weight)
    R_running = casadi.DM(bryson_control_weight)
    objective = 0.0
    for k in range(n_arcs):
        covariance_current = node_covariances[k]
        mean_next, covariance_next, control_covariance = arc_functions[k](
            means[:, k],
            covariance_current,
            feedforward[:, k],
            float(initial_guess.gain_scale[k]) * gains[k],
        )
        opti.subject_to(means[:, k + 1] == mean_next)
        # Covariance matching, on the lower triangle only: a dense symmetric
        # equality would duplicate every off-diagonal row.
        defect = node_covariances[k + 1] - covariance_next
        inverse_scale = casadi.DM(
            np.diag(covariance_scale_inv_std[k + 1])
        )
        scaled_defect = inverse_scale @ defect @ inverse_scale
        opti.subject_to(
            casadi.vertcat(
                *[
                    scaled_defect[row, column]
                    for row in range(NX)
                    for column in range(row + 1)
                ]
            )
            == 0.0
        )

        # regularized control norm, which is the leaked magnitude of the feedforward
        mean_control_norm = leaked_mean_control_norm(
            feedforward[:, k], control_norm_eps
        )

        # rho(SigmaT_k) in closed form
        radius = symbolic_psqrt_spectral_radius(control_covariance, options.spectral_radius_floor)

        # Transcription of the control chance constraint
        opti.subject_to(mean_control_norm + psi_inv * radius <= 1.0)

        # Composite running cost used by the short-horizon comparison: the
        # leaked mean-control norm plus the weighted state and control
        # covariance traces. Weights are derived through Bryson's rule.
        # The chance radius is constrained above but is not
        # charged separately, and there is no terminal Qf objective term.
        objective = objective + composite_running_cost(
            mean_control_norm,
            covariance_current,
            control_covariance,
            Q_running,
            R_running,
        )

    opti.subject_to(
        means[0:NP, n_arcs]
        == casadi.DM(terminal_mean_state_target(case, options, ref_traj))
    )

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
    terminal_covariance = node_covariances[-1]
    scaled_terminal = (
        casadi.diag(inverse_target_std)
        @ terminal_covariance[0:NP, 0:NP]
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
    for k in range(n_arcs):
        opti.set_initial(gains[k], initial_guess.gains[k] / initial_guess.gain_scale[k])
    opti.set_initial(margin, initial_guess.terminal_margin)
    cholesky_factor_lower_indices = [
        (row, column)
        for row in range(NX)
        for column in range(row + 1)
    ]
    for node, entries in enumerate(cholesky_factor_entries, start=1):
        inverse_scale = np.diag(covariance_scale_inv_std[node])
        normalized_seed = (
            inverse_scale
            @ seed_covariances[node]
            @ inverse_scale
        )
        factor = state_covariance_cholesky_factor_seed(
            normalized_seed,
        )
        opti.set_initial(
            entries,
            np.asarray(
                [
                    factor[row, column]
                    for row, column in cholesky_factor_lower_indices
                ]
            ),
        )

    configure_snopt(opti, options)
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
    # The leaked norm is an expression, not a variable: rebuild it from the
    # solved feedforward exactly as the objective and chance constraint saw it.
    solved_control_norm = np.sqrt(
        np.sum(solved_feedforward**2, axis=0) + control_norm_eps**2
    )
    solved_gains = np.stack(
        [
            initial_guess.gain_scale[k] * np.asarray(solution.value(gain), dtype=float).reshape(NU, NP)
            for k, gain in enumerate(gains)
        ]
    )

    # Recover the solved node covariances from their scaled Cholesky factors,
    # then re-propagate each arc to report both the mean and the covariance
    # shooting defects.
    covariances = np.empty((n_arcs + 1, NX, NX), dtype=float)
    covariances[0] = normalization.initial_covariance
    for node, entries in enumerate(cholesky_factor_entries, start=1):
        values = np.asarray(solution.value(entries), dtype=float).ravel()
        factor = np.zeros((NX, NX), dtype=float)
        for (row, column), value in zip(
            cholesky_factor_lower_indices,
            values,
        ):
            factor[row, column] = value
        scale = np.diag(covariance_scale_std[node])
        covariances[node] = scale @ factor @ factor.T @ scale

    propagated_means = np.empty_like(solved_means)
    propagated_means[:, 0] = solved_means[:, 0]
    control_covariances = np.empty((n_arcs, NU, NU), dtype=float)
    covariance_matching_defect = 0.0
    scaled_covariance_matching_defect = 0.0
    for k in range(n_arcs):
        mean_next, covariance_next, control_covariance = arc_functions[k](
            solved_means[:, k],
            covariances[k],
            solved_feedforward[:, k],
            solved_gains[k],
        )
        propagated_means[:, k + 1] = np.asarray(
            mean_next.full(), dtype=float
        ).ravel()
        predicted_covariance = np.asarray(
            covariance_next.full(), dtype=float
        )
        control_covariances[k] = np.asarray(
            control_covariance.full(), dtype=float
        )
        raw_defect = predicted_covariance - covariances[k + 1]
        covariance_matching_defect = max(
            covariance_matching_defect,
            float(np.max(np.abs(raw_defect))),
        )
        inverse_scale = np.diag(covariance_scale_inv_std[k + 1])
        scaled_defect = inverse_scale @ raw_defect @ inverse_scale
        scaled_covariance_matching_defect = max(
            scaled_covariance_matching_defect,
            float(np.max(np.abs(scaled_defect))),
        )

    state_matching_defect = float(np.max(np.abs(propagated_means[:, 1:] - solved_means[:, 1:])))
    # rho is an expression, so recover it exactly from the propagated moments.
    solved_radius = np.sqrt(
        psqrt_spectral_radii(control_covariances) ** 2 + options.spectral_radius_floor ** 2
    )
    objective_mean_control = float(np.sum(ref_traj.steps * solved_control_norm))
    objective_state_covariance = float(
        np.sum(
            [
                ref_traj.steps[k]
                * np.trace(
                    bryson_state_weight @ covariances[k, 0:NP, 0:NP]
                )
                for k in range(n_arcs)
            ]
        )
    )
    objective_control_covariance = float(
        np.sum(
            [
                ref_traj.steps[k]
                * np.trace(
                    bryson_control_weight @ control_covariances[k]
                )
                for k in range(n_arcs)
            ]
        )
    )

    return RobustSolution(
        means=solved_means,
        feedforward=solved_feedforward,
        gains=solved_gains,
        radius=solved_radius,
        covariances=covariances,
        control_covariances=control_covariances,
        objective=float(solution.value(objective)),
        covariance_scale_variances=covariance_scale_variances,
        diagnostics={
            "converged": float(converged),
            "state_matching_defect_nd": state_matching_defect,
            "state_covariance_matching_defect": covariance_matching_defect,
            "scaled_state_covariance_matching_defect": (
                scaled_covariance_matching_defect
            ),
            "state_covariance_scale_min_variance": float(
                np.min(covariance_scale_variances)
            ),
            "state_covariance_scale_max_variance": float(
                np.max(covariance_scale_variances)
            ),
            "objective_mean_control": objective_mean_control,
            "objective_state_covariance": objective_state_covariance,
            "objective_control_covariance": objective_control_covariance,
        },
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

    mean_target = terminal_mean_state_target(case, options, ref_traj)
    componentwise_mean_error = float(
        np.max(np.abs(solution.means[0:NP, -1] - mean_target))
    )
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
        "duration_days": float(
            ref_traj.node_times[-1] * case.time_unit / 86400.0
        ),
        "short_horizon_solve": float(options.truncated_uniform_mesh_arcs is not None),
        "short_horizon_reference_arcs": float(
            options.truncated_uniform_mesh_arcs
            if options.truncated_uniform_mesh_arcs is not None
            else ref_traj.n_arcs
        ),
        "converged": solution.diagnostics["converged"],
        "state_matching_defect_nd": solution.diagnostics["state_matching_defect_nd"],
        "state_covariance_matching_defect": solution.diagnostics[
            "state_covariance_matching_defect"
        ],
        "scaled_state_covariance_matching_defect": solution.diagnostics[
            "scaled_state_covariance_matching_defect"
        ],
        "state_covariance_scale_min_variance": solution.diagnostics[
            "state_covariance_scale_min_variance"
        ],
        "state_covariance_scale_max_variance": solution.diagnostics[
            "state_covariance_scale_max_variance"
        ],
        "objective_nd": float(solution.objective),
        "objective_mean_control": solution.diagnostics[
            "objective_mean_control"
        ],
        "objective_state_covariance": solution.diagnostics[
            "objective_state_covariance"
        ],
        "objective_control_covariance": solution.diagnostics[
            "objective_control_covariance"
        ],
        "bryson_sigma_factor": float(options.bryson_sigma_factor),
        "deterministic_effort_nd": deterministic_effort,
        "stochastic_effort_nd": feedback_effort,
        "total_effort_nd": total_effort,
        "deterministic_effort_kg": deterministic_effort * case.m0_wet * case.max_thrust_nd / case.exhaust_velocity_nd,
        "stochastic_effort_kg": feedback_effort * case.m0_wet * case.max_thrust_nd / case.exhaust_velocity_nd,
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
    sigma_factor = diagnostics["bryson_sigma_factor"]

    rows = [
        ("case", case.display_name),
        ("arcs", f"{diagnostics['n_arcs']:.0f}"),
        ("converged", "true" if diagnostics["converged"] else "false"),
        (
            "horizon",
            f"{'relative' if diagnostics['short_horizon_solve'] else 'full transfer'}, "
            f"{diagnostics['duration_days']:.6f} [days]",
        ),
        ("max mean matching defect", f"{diagnostics['state_matching_defect_nd']:.3e} [-]"),
        (
            "max state cov. matching defect (raw)",
            f"{diagnostics['state_covariance_matching_defect']:.3e}",
        ),
        (
            "max state cov. matching defect (scaled)",
            f"{diagnostics['scaled_state_covariance_matching_defect']:.3e}",
        ),
        ("composite objective", f"{diagnostics['objective_nd']:.6e} [-]"),
        ("  mean-control term", f"{diagnostics['objective_mean_control']:.6e}"),
        ("  state covariance term", f"{diagnostics['objective_state_covariance']:.6e}"),
        (
            "  control covariance term",
            f"{diagnostics['objective_control_covariance']:.6e}",
        ),
        (
            "  Bryson weights",
            f"Q={1.0 / sigma_factor ** 2:g} I6, R=I3, no Qf",
        ),
        ("robust effort diagnostic", f"{diagnostics['total_effort_nd']:.6e} [-]"),
        ("  open-loop  T_d", f"{diagnostics['deterministic_effort_kg']:.6f} [kg]"),
        ("  closed-loop T_s", f"{diagnostics['stochastic_effort_kg']:.6f} [kg]"),
        ("  total", f"{diagnostics['total_effort_kg']:.6f} [kg]"),
        (
            "reference trajectory fuel",
            f"{diagnostics['nominal_fuel_consumed']:.6f} [kg]",
        ),
        (
            "max thrust budget",
            f"{diagnostics['max_thrust_budget']:.6f} "
            f"(must be <= 1, i.e. {to_n:.4f} [N])",
        ),
        (
            "terminal componentwise mean error",
            f"{diagnostics['terminal_componentwise_mean_error_nd']:.3e} [-]",
        ),
        (
            "terminal cov. pos. block ratio",
            f"{diagnostics['terminal_position_block_ratio']:.6f} "
            f"(diagnostic only; reduction "
            f"{diagnostics['position_covariance_reduction_requested']:.4g})",
        ),
        (
            "terminal cov. vel. block ratio",
            f"{diagnostics['terminal_velocity_block_ratio']:.6f} "
            f"(diagnostic only; reduction "
            f"{diagnostics['velocity_covariance_reduction_requested']:.4g})",
        ),
        (
            "full 6x6 max eigval",
            f"{diagnostics['terminal_covariance_max_eigenvalue']:.6f} "
            f"(must be <= 1, the constrained quantity)",
        ),
        (
            "  terminal 1-sigma pos.",
            f"{diagnostics['terminal_position_std_m']:.6e} [m]",
        ),
        (
            "  terminal 1-sigma vel.",
            f"{diagnostics['terminal_velocity_std_ms']:.6e} [m/s]",
        ),
        (
            "MC 95th pct. of control effort",
            f"{diagnostics['monte_carlo_percentile_nd']:.6e} [-] vs predicted "
            f"{diagnostics['predicted_percentile_nd']:.6e} [-]",
        ),
        (
            "MC worst-arc where P(||T|| > Tmax)",
            f"{diagnostics['monte_carlo_max_violation_fraction']:.4f} "
            f"(must be <= 0.05); worst exceedance "
            f"{diagnostics['monte_carlo_worst_exceedance_n']:+.3e} [N]",
        ),
        (
            "MC terminal max eigval",
            f"{diagnostics['monte_carlo_terminal_max_eigenvalue']:.6f}",
        ),
    ]

    label_width = max(len(name) for name, _ in rows)
    separator = "=" * 88

    lines = [
        "",
        separator,
        f"{'Robust Optimization Summary':^88}",
        separator,
    ]

    for name, value in rows:
        lines.append(f"{name:<{label_width}}  |  {value}")

    lines.append(separator)

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


def densified_mean_trajectory(
    case: TestCase,
    options: Options,
    dynamics: Dynamics,
    solution: RobustSolution,
    steps: np.ndarray,
    samples_per_arc: int = 20,
) -> np.ndarray:
    """Re-propagate the solved mean trajectory for plotting only.

    `solution.means` holds one state per node, so drawing it directly joins the
    nodes with straight lines and a curved transfer reads as a chain of
    segments. Here each arc is re-integrated from its own node state under that
    arc's zero-order-hold control, sampling `samples_per_arc` intermediate
    points, exactly as `monte_carlo_rollouts` propagates a sample.

    The result is the trajectory the solution actually flies between nodes, not
    an interpolation of the node states, and it reproduces every node state by
    construction -- so the covariance ellipses drawn at the nodes stay attached
    to it. It is display-only: nothing here feeds the NLP, the diagnostics or
    the saved arrays.
    """

    n_arcs = solution.feedforward.shape[1]
    fractions = np.arange(1, samples_per_arc) / float(samples_per_arc)
    columns = [solution.means[:, 0:1]]
    for k in range(n_arcs):
        node_state = solution.means[:, k]
        control = case.max_thrust_nd * solution.feedforward[:, k]
        # Each interior sample is propagated from the node rather than chained
        # from the previous sample, so no sample inherits another's error.
        if fractions.size:
            batch_states = np.repeat(node_state[:, None], fractions.size, axis=1)
            batch_controls = np.repeat(control[:, None], fractions.size, axis=1)
            interior = np.empty((NX, fractions.size), dtype=float)
            for index, fraction in enumerate(fractions):
                propagated, _, _ = dynamics.propagate(
                    batch_states[:, index : index + 1],
                    batch_controls[:, index : index + 1],
                    float(steps[k]) * float(fraction),
                    options.integrator_substeps,
                )
                interior[:, index] = np.asarray(propagated, dtype=float).ravel()
            columns.append(interior)
        # Close the arc on the solved node state itself, so the drawn curve
        # passes through every node exactly.
        columns.append(solution.means[:, k + 1 : k + 2])
    return np.concatenate(columns, axis=1)


def plot_outputs(
    case: TestCase,
    options: Options,
    ref_traj: ReferenceTraj,
    solution: RobustSolution,
    normalization: Normalization,
    monte_carlo_result: dict,
    psi_inv: float,
    output_prefix: Path,
    dynamics: Dynamics,
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
    # The centre panel plots departures from the nominal, so the envelope is the
    # symmetric +/- feedback_n. The zero clip that keeps an absolute thrust
    # non-negative is deliberately not applied here: a negative deviation is a
    # thrust below nominal, not a negative thrust, and clipping it would pinch
    # the envelope inward on exactly the low-thrust arcs.
    rollout_deviation_n = rollout_thrust_n - nominal_thrust_n[:, None]
    step_days, nominal_step = plotter.plot_zero_order_hold_data(
        node_days, nominal_thrust_n
    )
    _, lower_deviation_step = plotter.plot_zero_order_hold_data(
        node_days, -feedback_n
    )
    _, upper_deviation_step = plotter.plot_zero_order_hold_data(
        node_days, feedback_n
    )
    _, rollout_deviation_step = plotter.plot_zero_order_hold_data(
        node_days, rollout_deviation_n
    )

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(1.5 * Plotter.WIDE_FIGSIZE[0], Plotter.WIDE_FIGSIZE[1]),
        dpi=Plotter.FIGURE_DPI,
    )
    axis = axes[0]
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
    plotted_peak = max(case.max_thrust_n, float(np.max(nominal_thrust_n)))
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
    rollout_lines = axis.plot(
        step_days,
        rollout_deviation_step,
        color=Plotter.DARK_GREY,
        alpha=0.28,
        lw=0.48,
        zorder=1,
    )
    if rollout_lines:
        rollout_lines[0].set_label("Monte Carlo")
    axis.plot(
        step_days,
        lower_deviation_step,
        color=Plotter.BLUE,
        ls="--",
        lw=0.82,
        label="Predicted bound",
        zorder=2,
    )
    axis.plot(
        step_days,
        upper_deviation_step,
        color=Plotter.BLUE,
        ls="--",
        lw=0.82,
        zorder=2,
    )
    axis.set_xlabel("time [days]")
    axis.set_ylabel("thrust deviation from nominal [N]")
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

    axis = axes[2]
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
            1,
            len(axes_to_plot),
            figsize=Plotter.TRIPLE_SQUARE_FIGSIZE,
            dpi=Plotter.FIGURE_DPI,
        )
        projection_axes = list(np.atleast_1d(axes))

    lagrange = deterministic_cr3bp.get_collinear_lagrange_points(case)
    departure_orbit = None
    target_orbit = None
    if case.departure_period_nd is not None:
        departure_orbit = deterministic_cr3bp.propagate_periodic_orbit(
            case, case.x0_augmented_state, case.departure_period_nd
        )
    # A truncated transfer ends at a relative reference node, not at the full
    # target periodic orbit.
    if options.truncated_uniform_mesh_arcs is None and case.target_period_nd is not None:
        target_orbit = deterministic_cr3bp.propagate_periodic_orbit(
            case, case.xf_augmented_state, case.target_period_nd
        )

    # Display-only dense mean trajectory: the arcs re-propagated through the
    # true dynamics, so the drawn nominal is a smooth curve instead of the
    # straight chords that joining the node states produces.
    dense_samples_per_arc = 20
    dense_means = densified_mean_trajectory(
        case,
        options,
        dynamics,
        solution,
        ref_traj.steps,
        samples_per_arc=dense_samples_per_arc,
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
    # The dense curve carries `samples_per_arc` columns per arc plus a final
    # node, so the zero-order-hold control is repeated to match it column for
    # column; `_plot_projection` indexes states and controls together.
    dense_control_nd = np.concatenate(
        [
            np.repeat(
                control_nd[:, k : k + 1],
                dense_samples_per_arc,
                axis=1,
            )
            for k in range(control_nd.shape[1])
        ]
        + [control_nd[:, -1:]],
        axis=1,
    )

    for projection_index, (axis, (axis_0, axis_1)) in enumerate(zip(projection_axes, axes_to_plot)):
        # The smooth nominal is drawn from the re-propagated arcs, while the
        # thrust arrows stay on the node arrays: `_plot_thrust_arrows_2d`
        # selects a fixed number of arrows out of the samples whose thrust is
        # active, so handing it the 20x denser grid would pack them solid.
        plotter._plot_projection(
            axis,
            case,
            dense_means,
            dense_control_nd,
            departure_orbit,
            target_orbit,
            lagrange,
            axis_0,
            axis_1,
            arrow_states=solution.means,
            arrow_controls=control_nd,
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
                color=Plotter.DARK_BLUE,
                lw=0.80,
                alpha=0.90,
                zorder=2,
                label=f"Predicted covariance ({projection_scale_label})"
                if projection_index == 0 and covariance_index == 0
                else "_nolegend_",
            )

    if len(axes_to_plot) > 1:
        # Congruent square panels. This has to run after `_plot_projection`,
        # which calls `_style_2d_axis(equal_axis=True)` and so sets
        # `set_aspect("equal", adjustable="box")`: that keeps the data scale
        # undistorted but shrinks each panel's box to its own data range, so
        # x-y, x-z and y-z otherwise render at three different sizes.
        # Re-asserting the equal data scale with `adjustable="datalim"` moves
        # the freedom into the axis limits, leaving the box free to be squared.
        for projection_axis in projection_axes:
            projection_axis.set_aspect("equal", adjustable="datalim")
            projection_axis.set_box_aspect(1.0)

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
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.70 if is_single_projection else 0.90))
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
    dynamics: Dynamics,
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
        short_horizon_reference_arcs=(
            -1 if options.truncated_uniform_mesh_arcs is None else options.truncated_uniform_mesh_arcs
        ),
        short_horizon_solve=options.truncated_uniform_mesh_arcs is not None,
        covariance_transcription=np.asarray("multiple_shooting"),
        covariance_scale_variances=solution.covariance_scale_variances,
        terminal_mean_state_target=terminal_mean_state_target(case, options, ref_traj),
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
        bryson_sigma_factor=options.bryson_sigma_factor,
        bryson_state_weight=np.eye(NP),
        bryson_control_weight=(
            options.bryson_sigma_factor**2 * np.eye(NU)
        ),
        terminal_Qf_objective_weight=0.0,
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
        dynamics,
    )


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def run_test_case(test_case_id: str, options: Options | None = None) -> dict[str, float]:
    case = CASE_REGISTRY[test_case_id]()
    psi_inv = psi_inverse(NU, options.violation_parameter)

    dynamics = Dynamics(case)
    ref_traj = reoptimize_reference_trajectory(
        case,
        options,
        load_ref_traj(case, options),
        dynamics,
    )
    normalization = build_normalization(case)

    arc_functions = _build_arc_functions(case, options, dynamics, normalization, ref_traj)

    initial_guess = build_initial_guess(
        case,
        options,
        dynamics,
        normalization,
        ref_traj,
        arc_functions,
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

    output_name = case.test_case_id
    if options.truncated_uniform_mesh_arcs is not None:
        output_name = f"{output_name}_{options.truncated_uniform_mesh_arcs}_arcs_truncated"
    output_prefix = OUTPUT_DIR / output_name
    save_outputs(
        case, options, ref_traj, solution, normalization,
        monte_carlo_result, diagnostics, psi_inv, output_prefix, dynamics,
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
    """Run the selected cases.

    Every setting is edited here or in `Options`, not on the command line:
    change `case_ids` to pick which cases run, and pass any non-default
    `Options` key below (for example `truncated_uniform_mesh_arcs`,
    `elastic_weight`, `partial_price`).
    """

    # Cases to run, keyed into CASE_REGISTRY. Comment out the ones to skip.
    case_ids = (
        "nrho_l2_to_dro",
        "lyapunov_l1_to_l2",
        "halo_l2_to_halo_l1",
    )

    options = Options(
        # Use only the first N source-reference arcs and their relative terminal
        # state (for example, 30). None runs the full transfer.
        truncated_uniform_mesh_arcs=None,
        # Equal-duration arcs the reference is resampled onto: an int applies to
        # every case, a dict keys per case. Defaults to UNIFORM_ARCS_BY_CASE.
        uniform_mesh_arcs=dict(UNIFORM_ARCS_BY_CASE),
        # SNOPT partial pricing; None keeps SNOPT's native default of 1.
        partial_price=None,
        # Initial SNOPT nonlinear elastic-mode weight; None keeps SNOPT's
        # native default of 1e4.
        elastic_weight=None,
    )

    for test_case_id in case_ids:
        run_test_case(test_case_id, options)


if __name__ == "__main__":
    main()
