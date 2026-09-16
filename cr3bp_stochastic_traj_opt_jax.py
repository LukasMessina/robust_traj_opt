"""CR3BP stochastic trajectory optimization with JAX, Diffrax and pyOptSparse.

The mathematical model and defaults are those of cr3bp_stochastic_traj_opt.
This is a standalone implementation relative to that CasADi/SNOPT stochastic
script: its own NLPs, integration maps, derivatives and reporting. CasADi
itself is not required to run it, since the case/dynamics-constant definitions
below are imported from the deterministic module rather than duplicated, and
every dynamics, integration and differentiation routine specific to this file
is implemented directly in JAX/Diffrax.
Saved deterministic reference data and the generic plotter module are required.
Run with ``conda run --no-capture-output -n project python <this file>``.

The stochastic objective is an UNWEIGHTED sum over equal-duration arcs.
Navigation error and continuous acceleration diffusion are sampled by the
augmented UT. The proportional-only Gates maneuver-execution error is sampled
by a nested cubature: each outer UT sigma point's own commanded control spawns
six conditional inner sigma points, so enabling Gates multiplies the outer
sigma-point count by 6 in both the NLP and the Monte Carlo rollout.
Sigma-point mass flow uses the unsmoothed thrust norm, as in the original.
The deterministic warm-start NLP alone uses the regularized mass-flow norm.
"""

from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cuda")
# Set JAX_PLATFORMS=cpu before launching to override this default. GPU settings:
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
# os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

from dataclasses import dataclass, field, replace
from pathlib import Path
import tempfile

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import diffrax
import numpy as np
from pyoptsparse import Optimization
from pyoptsparse.pySNOPT.pySNOPT import SNOPT

from scipy.stats import chi2
from scipy.optimize import brentq
from plotter import Plotter
import matplotlib.pyplot as plt

from xmds2_rk9 import XMDS2RK9

from cr3bp_deterministic_traj_opt import (
    CR3BPEarthMoon,
    TestCase,
    NrhoL2ToDro,
    LyapunovL1ToL2,
    HaloL2ToHaloL1,
    CASE_TYPES,
    CASE_REGISTRY,
)

# Match the double-integrator script: plotting does not require external LaTeX.
plt.rcParams.update({"text.usetex": False, "mathtext.fontset": "cm"})


NX = 7                       # augmented state dimension


NU = 3                       # control dimension


NP = 6                       # primed (position-velocity) state dimension


NW = 3                       # independent acceleration Wiener processes


ACCELERATION_DIFFUSION_KM_S32 = 1e-10  # [km s^(-3/2)]


# Proportional-only Gates maneuver-execution error model (fixed terms sigma_1 =
# sigma_3 = 0). Both are dimensionless 1-sigma proportional coefficients applied
# directly to the commanded control vector, so they carry over unchanged
# whether that vector is expressed in physical thrust or in the max-thrust-
# normalised units used by the NLP: Q_G(T_max c) = T_max^2 Q_G(c).
GATES_PROPORTIONAL_MAGNITUDE_STD = 8.0e-3      # sigma_2 [-]
GATES_PROPORTIONAL_POINTING_STD = 9.5993e-3    # sigma_4 [rad]


REFERENCE_DIR = Path("output/cr3bp_deterministic_traj_opt/energy_optimal")


INITIAL_POSITION_STD_KM = 50.0
INITIAL_VELOCITY_STD_KM_S = 1.0e-3
FINAL_POSITION_STD_KM = 10.0
FINAL_VELOCITY_STD_KM_S = 1.0e-4


UNIFORM_ARCS_BY_CASE: dict[str, int] = {
    "nrho_l2_to_dro": 85,      
    "halo_l2_to_halo_l1": 80,  
    "lyapunov_l1_to_l2": 48,   
}


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
    truncated_uniform_mesh_arcs: int | None = 30
    # Control-norm regularizer: the thrust magnitude is carried as
    # sqrt(u'u + eps_1^2) on the up-to-the-unit control, everywhere it appears.
    control_norm_eps: float = 1e-6
    bryson_sigma_factor: float = 3.0
    integrator_substeps: int = 1
    # Fixed XMDS2 RK9 substeps per shooting arc. Each substep receives one
    # externally prescribed Brownian increment. Used by the NLP's arc map
    # (and its seed propagation); Monte Carlo uses this value unless
    # `monte_carlo_sde_integrator_substeps` overrides it below.
    sde_integrator_substeps: int = 1
    # When set, monte_carlo_rollouts uses this substep count instead of
    # `sde_integrator_substeps`, decoupling Monte Carlo validation fidelity
    # from the NLP's own transcription. None (default) reuses
    # `sde_integrator_substeps`, matching prior behaviour exactly.
    monte_carlo_sde_integrator_substeps: int | None = 3
    acceleration_diffusion_km_s32: float = ACCELERATION_DIFFUSION_KM_S32

    # Proportional-only Gates maneuver-execution error model. Setting both to
    # zero disables the execution-error nested cubature (inner transform
    # degenerates to the single point delta_c_G = 0) and the Monte Carlo draw.
    gates_proportional_magnitude_std: float = GATES_PROPORTIONAL_MAGNITUDE_STD
    gates_proportional_pointing_std: float = GATES_PROPORTIONAL_POINTING_STD

    violation_parameter: float = 0.01
    # Terminal covariance reduction factors relative to the corresponding
    # initial covariance blocks: (50 km / 10 km)^2 and (1 m/s / 0.1 m/s)^2.
    position_covariance_reduction: float = (
        INITIAL_POSITION_STD_KM / FINAL_POSITION_STD_KM
    ) ** 2
    velocity_covariance_reduction: float = (
        INITIAL_VELOCITY_STD_KM_S / FINAL_VELOCITY_STD_KM_S
    ) ** 2
    scaling_parameter: float = 0.0
    # Navigation (measurement) standard deviations. Setting both to zero
    # disables navigation error in the UT and Monte Carlo propagation.
    navigation_position_std_km: float = 10.0
    navigation_velocity_std_km_s: float = 1.0e-4

    # Floors the covariance at jitter / n_x, which must stay far below the
    # terminal target while covering the negative eigenvalues of order 1e-15 that
    # round-off leaves behind once the steering contracts a direction.
    cholesky_jitter: float = 1e-12
    spectral_radius_floor: float = 1e-7

    # Floor on diag(G) in the terminal covariance constraint (see terminal_output):
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
    print_sparsity: bool = False


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


def augmented_dimension(options: Options) -> int:
    """Dimension the sigma points are generated on.

    The blocks are the state, optional navigation error, and the independent
    ``delta W`` coordinates required by every XMDS2 RK9 substep.
    """

    navigation_dimension = NP if navigation_error_enabled(options) else 0
    process_dimension = (
        NW * int(options.sde_integrator_substeps)
        if options.acceleration_diffusion_km_s32 > 0.0
        else 0
    )
    return NX + navigation_dimension + process_dimension


def acceleration_diffusion_nd(case: TestCase, options: Options) -> float:
    """Convert acceleration diffusion from km s^(-3/2) to CR3BP units.

    With ``r_dim = L r_nd`` and ``t_dim = T t_nd``, Brownian scaling gives
    ``sigma_nd = sigma_dim T^(3/2) / L``.
    """

    sigma = float(options.acceleration_diffusion_km_s32)
    if not np.isfinite(sigma) or sigma < 0.0:
        raise ValueError("acceleration diffusion must be finite and nonnegative")
    return sigma * case.time_unit**1.5 / case.length_unit


def acceleration_diffusion_matrix(case: TestCase, options: Options) -> np.ndarray:
    """Return the additive 7-by-3 nondimensional acceleration diffusion."""

    matrix = np.zeros((NX, NW), dtype=float)
    matrix[3:6] = acceleration_diffusion_nd(case, options) * np.eye(NW)
    return matrix


def navigation_error_enabled(options: Options) -> bool:
    """Whether either position or velocity navigation error is enabled."""

    values = np.array(
        [
            options.navigation_position_std_km,
            options.navigation_velocity_std_km_s,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(
            "navigation standard deviations must be finite and nonnegative"
        )
    return bool(np.any(values > 0.0))


def navigation_covariance(
    case: TestCase,
    options: Options,
    normalization: "Normalization",
) -> np.ndarray:
    """Six-state navigation covariance in normalized coordinates."""

    navigation_error_enabled(options)
    navigation_std_nd = np.array(
        [options.navigation_position_std_km / case.length_unit] * 3
        + [options.navigation_velocity_std_km_s / case.velocity_unit] * 3,
        dtype=float,
    )
    normalized_std = navigation_std_nd / normalization.scale[:NP]
    return np.diag(normalized_std**2)


def gates_execution_error_enabled(options: Options) -> bool:
    """Whether the proportional-only Gates execution-error model is active."""

    values = np.array(
        [
            options.gates_proportional_magnitude_std,
            options.gates_proportional_pointing_std,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(
            "Gates proportional standard deviations must be finite and nonnegative"
        )
    return bool(np.any(values > 0.0))


def gates_execution_covariance(control: jnp.ndarray, options: Options) -> jnp.ndarray:
    """Proportional-only Gates covariance Q_G(u) = sigma_4^2||u||^2 I + (sigma_2^2 - sigma_4^2) u u^T.

    `control` may be in any linear units (physical thrust, normalised thrust);
    the formula is homogeneous of degree 2, so the returned covariance is in
    the corresponding squared units. Vanishes identically at ``control = 0``,
    matching the physical statement that zero commanded thrust produces zero
    execution error.
    """

    magnitude_variance = options.gates_proportional_magnitude_std ** 2
    pointing_variance = options.gates_proportional_pointing_std ** 2
    return pointing_variance * jnp.dot(control, control) * jnp.eye(
        control.shape[0]
    ) + (magnitude_variance - pointing_variance) * jnp.outer(control, control)


def unscented_weights(kappa: float, dimension: int) -> np.ndarray:
    """Unscented transform weights."""

    weights = np.full(2 * dimension + 1, 1.0 / (2.0 * (dimension + kappa)))
    weights[0] = kappa / (dimension + kappa)
    return weights


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

    position_velocity_std = np.array(
        [INITIAL_POSITION_STD_KM / case.length_unit] * 3
        + [INITIAL_VELOCITY_STD_KM_S / case.velocity_unit] * 3,
        dtype=float,
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


def monte_carlo_rollouts(
    case: TestCase,
    options: Options,
    normalization: Normalization,
    solution: RobustSolution,
    steps: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Apply the converged policy with maneuver-execution error and thrust saturation.

    The proportional-only Gates execution error is sampled conditionally on the
    commanded (unsaturated) thrust and added to it (Eq. (6)-(14) of the
    execution-error model). Violation statistics test this Gates-inclusive,
    pre-saturation command against max_thrust, matching the NLP's own chance
    constraint, which bounds the commanded mean plus the nested (Gates-
    inclusive) executed-control dispersion. The sum is then what is clipped to
    the maximum thrust; propagation, accumulated effort and peak applied thrust
    all use this saturated, executed command, matching the physical actuator.

    Initial-state, navigation-error, process-noise and maneuver-execution-error
    coordinates come from four reproducible, mutually independent pseudorandom
    Gaussian streams. Navigation, process and execution-error draws are
    independent across both arcs and samples.
    """

    samples = options.monte_carlo_samples
    if samples < 2:
        raise ValueError("monte_carlo_samples must be at least 2")
    substeps_override = options.monte_carlo_sde_integrator_substeps
    substeps = int(
        substeps_override if substeps_override is not None
        else options.sde_integrator_substeps
    )
    if substeps != (substeps_override if substeps_override is not None
                     else options.sde_integrator_substeps) or substeps < 1:
        raise ValueError(
            "monte_carlo_sde_integrator_substeps (or sde_integrator_substeps) "
            "must be a positive integer"
        )
    # stochastic_integration_map reads sde_integrator_substeps from `options`
    # directly, so substitute the effective value rather than threading a
    # separate substeps argument through it.
    sde_options = (
        options if substeps_override is None
        else replace(options, sde_integrator_substeps=substeps)
    )
    initial_seed, navigation_seed, process_seed, gates_seed = np.random.SeedSequence(
        options.monte_carlo_seed
    ).spawn(4)
    initial_rng, navigation_rng, process_rng, gates_rng = (
        np.random.default_rng(seed)
        for seed in (initial_seed, navigation_seed, process_seed, gates_seed)
    )
    initial_normals = initial_rng.standard_normal((NX, samples))
    states = (
        solution.means[:, 0:1]
        + normalization.initial_std[:, None] * initial_normals
    )
    # Navigation error: an independent draw at every arc, added to the measured
    # deviation the policy acts on. 
    navigation_std = np.sqrt(
        np.clip(
            np.diag(navigation_covariance(case, options, normalization)),
            0.0,
            None,
        )
    )
    sde_step = stochastic_integration_map(case, sde_options)
    sde_step_batch = jax.jit(
        jax.vmap(sde_step, in_axes=(0, 0, 0, 0, None))
    )
    # Same computation as sde_step_batch, but also returns every intermediate
    # RK9 substep state; applied only to the plotted subset below so the
    # trajectory bundle traces the actual finer SDE path between arc nodes
    # instead of chording straight across each (possibly coarse) arc.
    sde_step_history_batch = jax.jit(
        jax.vmap(sde_step.with_history, in_axes=(0, 0, 0, 0, None))
    )
    process_noise_enabled = options.acceleration_diffusion_km_s32 > 0.0

    # Cumulative control effort is stored for each sample
    cumulative_control_effort = np.zeros(samples, dtype=float)
    # Stores one violation fraction per arc, averaged over all samples.
    violation_fraction = np.zeros(steps.size, dtype=float)
    # The worst exceedance over all samples and arcs.
    worst_exceedance = -np.inf
    peak_applied_thrust = np.zeros(samples, dtype=float)
    # Retain a small subset of these exact same SDE realizations for figures.
    # This avoids plotting a separate ensemble with different uncertainty or
    # integration assumptions. One entry per RK9 substep (not just per arc),
    # so the plotted bundle resolves the true path within each arc.
    plot_count = min(160, samples)
    plot_trajectory_history = np.empty(
        (NX, plot_count, steps.size * substeps + 1), dtype=float
    )
    plot_trajectory_history[:, :, 0] = states[:, :plot_count]
    plot_applied_control_deviation_norm_n = np.empty(
        (steps.size, plot_count), dtype=float
    )

    for k in range(steps.size):
        deviation = (states[0:NP] - solution.means[0:NP, k : k + 1]) / normalization.scale[0:NP, None]
        if navigation_error_enabled(options):
            navigation_normals = navigation_rng.standard_normal((NP, samples))
            deviation = deviation + navigation_std[0:NP, None] * navigation_normals
        commanded_control = case.max_thrust_nd * (
            solution.feedforward[:, k : k + 1] + solution.gains[k] @ deviation
        )
        commanded_magnitudes = np.linalg.norm(commanded_control, axis=0)

        # Maneuver-execution error: proportional-only Gates model, sampled
        # conditionally on the commanded (unsaturated) thrust of each sample,
        # then added to it before the actuator saturates (Eq. (6)-(14)).
        if gates_execution_error_enabled(options):
            magnitude_variance = options.gates_proportional_magnitude_std ** 2
            pointing_variance = options.gates_proportional_pointing_std ** 2
            gates_normals = gates_rng.standard_normal((NU, samples))
            unit_command = commanded_control / np.clip(commanded_magnitudes, 1e-300, None)
            parallel = np.sum(unit_command * gates_normals, axis=0, keepdims=True) * unit_command
            transverse = gates_normals - parallel
            execution_error = (
                commanded_magnitudes[None, :]
                * (np.sqrt(magnitude_variance) * parallel + np.sqrt(pointing_variance) * transverse)
            )
            executed_command = commanded_control + execution_error
        else:
            executed_command = commanded_control
        executed_magnitudes = np.linalg.norm(executed_command, axis=0)

        # Chance-constraint violation statistics test the Gates-inclusive,
        # pre-saturation command, matching what the NLP's own chance
        # constraint bounds: commanded mean plus the nested (Gates-inclusive)
        # executed-control dispersion against max_thrust.
        violation_fraction[k] = float(np.mean(executed_magnitudes > case.max_thrust_nd))
        worst_exceedance = max(
            worst_exceedance,
            float(np.max(executed_magnitudes) - case.max_thrust_nd),
        )

        # Radially project each over-limit command onto the thrust ball, preserving
        # its direction. Diagnostics above retain the pre-saturation magnitude so
        # actuator saturation cannot hide a chance-constraint violation.
        saturation_scale = np.ones_like(executed_magnitudes)
        saturated = executed_magnitudes > case.max_thrust_nd
        saturation_scale[saturated] = (
            case.max_thrust_nd / executed_magnitudes[saturated]
        )
        applied_control = executed_command * saturation_scale[None, :]
        applied_magnitudes = np.minimum(executed_magnitudes, case.max_thrust_nd)

        nominal_control = (
            case.max_thrust_nd * solution.feedforward[:, k : k + 1]
        )
        plot_applied_control_deviation_norm_n[k] = (
            np.linalg.norm(
                applied_control[:, :plot_count] - nominal_control,
                axis=0,
            )
            * case.thrust_unit
        )

        peak_applied_thrust = np.maximum(peak_applied_thrust, applied_magnitudes)
        cumulative_control_effort += float(steps[k]) * applied_magnitudes
        if process_noise_enabled:
            xi = process_rng.standard_normal((samples, substeps, NW))
            substep_duration = float(steps[k]) / substeps
            brownian_increments = np.sqrt(substep_duration) * xi
        else:
            brownian_increments = np.zeros((samples, substeps, NW))
        # The plotted subset also gets its intermediate RK9 substep states,
        # using the exact same states/control/Brownian draws as the full
        # batch below -- same realizations, just with the interior of the
        # arc kept instead of discarded.
        _, substep_history = sde_step_history_batch(
            jnp.asarray(states[:, :plot_count].T),
            jnp.asarray(applied_control[:, :plot_count].T),
            jnp.asarray(applied_magnitudes[:plot_count]),
            jnp.asarray(brownian_increments[:plot_count]),
            float(steps[k]),
        )
        # substep_history has shape (plot_count, substeps, NX); transpose to
        # (NX, plot_count, substeps) to match plot_trajectory_history's axes.
        plot_trajectory_history[:, :, k * substeps + 1 : (k + 1) * substeps + 1] = (
            np.asarray(substep_history).transpose(2, 0, 1)
        )
        states = np.asarray(
            sde_step_batch(
                jnp.asarray(states.T),
                jnp.asarray(applied_control.T),
                jnp.asarray(applied_magnitudes),
                jnp.asarray(brownian_increments),
                float(steps[k]),
            )
        ).T

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
        "plot_trajectory_history": plot_trajectory_history,
        "plot_applied_control_deviation_norm_n": (
            plot_applied_control_deviation_norm_n
        ),
    }


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
        "sde_integrator_substeps": float(options.sde_integrator_substeps),
        "acceleration_diffusion_km_s32": float(
            options.acceleration_diffusion_km_s32
        ),
        "acceleration_diffusion_nd": acceleration_diffusion_nd(case, options),
        "navigation_position_std_km": float(
            options.navigation_position_std_km
        ),
        "navigation_velocity_std_ms": float(
            options.navigation_velocity_std_km_s * 1e3
        ),
        "gates_proportional_magnitude_std": float(
            options.gates_proportional_magnitude_std
        ),
        "gates_proportional_pointing_std": float(
            options.gates_proportional_pointing_std
        ),
        "ut_augmented_dimension": float(augmented_dimension(options)),
        "ut_sigma_points": float(2 * augmented_dimension(options) + 1),
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
    points with the deterministic Dopri8 map. Monte Carlo uses the XMDS2 RK9 SDE
    map and is intentionally handled separately.

    This is a nominal mean-command visualization rather than one stochastic
    realization. It reproduces every optimized node state by construction, so
    the covariance ellipses drawn at the nodes stay attached to it. It is
    display-only: nothing here feeds the NLP, diagnostics, or saved arrays.
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


def get_collinear_lagrange_points(case: CR3BPEarthMoon) -> dict[str, float]:
    def equilibrium_condition(x):
        r1 = abs(x + case.mu)
        r2 = abs(x - (1.0 - case.mu))
        return (
            x
            - (1.0 - case.mu) * (x + case.mu) / r1**3
            - case.mu * (x - (1.0 - case.mu)) / r2**3
        )

    eps = 1e-9
    return {
        "L1": brentq(equilibrium_condition, -case.mu + eps, 1.0 - case.mu - eps),
        "L2": brentq(equilibrium_condition, 1.0 - case.mu + eps, 1.5),
        "L3": brentq(equilibrium_condition, -1.5, -case.mu - eps),
    }


def propagate_periodic_orbit(case, initial_state, period_nd, steps=1600):
    """Ballistic plotting trajectory using the same Diffrax Dopri8 map."""
    step = integration_map(case, 1)
    @jax.jit
    def rollout(initial):
        def advance(state, _):
            next_state = step(state, jnp.zeros(NU), 0.0, period_nd / steps)
            return next_state, next_state
        return jax.lax.scan(advance, initial, None, length=steps)[1]
    samples = np.asarray(rollout(jnp.asarray(initial_state)))
    return np.column_stack((initial_state, samples.T))


OUTPUT_DIR = Path("output/cr3bp_stochastic_traj_opt_jax")


class LowerTriangular:
    """Row-major packing, identical to the CasADi and double-integrator NLPs."""

    def __init__(self, dimension):
        self.dimension = dimension
        self.rows, self.columns = np.tril_indices(dimension)
        self.diagonal = np.flatnonzero(self.rows == self.columns)

    def __len__(self):
        return len(self.rows)

    def matrix(self, entries):
        return jnp.zeros((self.dimension, self.dimension)).at[
            self.rows, self.columns
        ].set(entries)

    def pack(self, matrix):
        return matrix[self.rows, self.columns]

    def covariance_pattern(self):
        """Exact polynomial support of d tril(L L.T) / d tril(L)."""
        i, j = self.rows[:, None], self.columns[:, None]
        r, c = self.rows[None, :], self.columns[None, :]
        return ((r == i) & (c <= j)) | ((r == j) & (c <= i))


LTRI, MTRI = LowerTriangular(NX), LowerTriangular(NP)
NL, NM = len(LTRI), len(MTRI)


def vector_field(case, state, control, magnitude):
    """Nondimensional rotating-frame CR3BP with thrust and variable mass."""
    x, y, z, vx, vy, vz, mass = state
    r1 = jnp.sqrt((x + case.mu) ** 2 + y*y + z*z)
    r2 = jnp.sqrt((x - 1 + case.mu) ** 2 + y*y + z*z)
    gravity = (1 - case.mu) / r1**3 + case.mu / r2**3
    return jnp.stack((
        vx, vy, vz,
        2*vy + x - (1-case.mu)*(x+case.mu)/r1**3
        - case.mu*(x-1+case.mu)/r2**3 + control[0]/mass,
        -2*vx + y - gravity*y + control[1]/mass,
        -gravity*z + control[2]/mass,
        -magnitude/case.exhaust_velocity_nd,
    ))


def integration_map(case, substeps):
    """Fixed-substep Diffrax Dopri8 (order 8), differentiated through its steps.

    Integrating on [0,1] avoids a final tiny step from roundoff in arc times.
    Reverse-mode AD uses Diffrax's discretize-then-optimize checkpoint adjoint.
    No adaptive tolerance or stochastic differential equation changes the model.
    """
    if int(substeps) != substeps or substeps < 1:
        raise ValueError("integrator_substeps must be a positive integer")
    substeps = int(substeps)

    def rhs(t, state, args):
        control, magnitude, duration = args
        return duration * vector_field(case, state, control, magnitude)

    term, solver = diffrax.ODETerm(rhs), diffrax.Dopri8()

    def step(state, control, magnitude, duration):
        result = diffrax.diffeqsolve(
            term, solver, t0=0.0, t1=1.0, dt0=1.0/substeps,
            y0=state, args=(control, magnitude, duration),
            stepsize_controller=diffrax.ConstantStepSize(),
            saveat=diffrax.SaveAt(t1=True), max_steps=substeps + 1,
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
        )
        return result.ys[0]

    return step


class PrescribedWienerPath(diffrax.AbstractPath):
    """Linear path carrying one externally prescribed Wiener increment.

    XMDS2 holds the sampled Wiener rate constant over a fixed RK step. Hence
    an interval occupying a fraction of the step receives the same fraction
    of ``brownian_increment``. No random sampling occurs inside this path.
    """

    brownian_increment: jnp.ndarray
    duration: float

    @property
    def t0(self):
        return 0.0

    @property
    def t1(self):
        return self.duration

    def evaluate(self, t0, t1=None, left=True):
        del left
        if t1 is None:
            t1 = t0
            t0 = 0.0
        interval = t1 - t0
        fraction = interval / self.duration
        return fraction * self.brownian_increment


def stochastic_integration_map(case, options):
    """Return a fixed-grid XMDS2 RK9 map for the additive CR3BP SDE.

    ``brownian_increments`` has shape ``(M, 3)``, where ``M`` is
    ``sde_integrator_substeps``. Each substep is one fixed RK9 step with a
    constant Wiener rate, represented by the prescribed path's total
    ``delta W``. The thrust command and its mass-flow magnitude remain held
    over the complete shooting arc.
    """

    substeps = int(options.sde_integrator_substeps)
    if substeps != options.sde_integrator_substeps or substeps < 1:
        raise ValueError("sde_integrator_substeps must be a positive integer")
    diffusion = jnp.asarray(acceleration_diffusion_matrix(case, options))
    drift_term = diffrax.ODETerm(
        lambda t, state, args: vector_field(case, state, args[0], args[1])
    )
    solver = XMDS2RK9()

    def one_substep(state, brownian_increment, control, magnitude, substep_duration):
        path = PrescribedWienerPath(
            brownian_increment=brownian_increment,
            duration=substep_duration,
        )
        diffusion_term = diffrax.ControlTerm(
            lambda t, state, args: diffusion,
            path,
        )
        result = diffrax.diffeqsolve(
            diffrax.MultiTerm(drift_term, diffusion_term),
            solver,
            t0=0.0,
            t1=substep_duration,
            dt0=substep_duration,
            y0=state,
            args=(control, magnitude),
            stepsize_controller=diffrax.ConstantStepSize(),
            saveat=diffrax.SaveAt(t1=True),
            max_steps=1,
            adjoint=diffrax.RecursiveCheckpointAdjoint(),
        )
        return result.ys[0]

    def scan_step(current_state, brownian_increment, control, magnitude, substep_duration):
        next_state = one_substep(
            current_state,
            brownian_increment,
            control,
            magnitude,
            substep_duration,
        )
        return next_state, next_state

    def step(
        state,
        control,
        magnitude,
        brownian_increments,
        duration,
    ):
        substep_duration = duration / substeps
        final_state, _ = jax.lax.scan(
            lambda current_state, brownian_increment: scan_step(
                current_state, brownian_increment, control, magnitude, substep_duration),
            state,
            brownian_increments,
        )
        return final_state

    def step_with_history(
        state,
        control,
        magnitude,
        brownian_increments,
        duration,
    ):
        """Like `step`, but also returns every intermediate substep state.

        Returns ``(final_state, substep_states)`` with ``substep_states`` of
        shape ``(sde_integrator_substeps, NX)``, the state after each of the
        ``substeps`` RK9 stages (the last row equals ``final_state``). Display
        use only: propagation, effort accounting and moment matching all use
        `step`, which discards these intermediate states.
        """

        substep_duration = duration / substeps
        final_state, substep_states = jax.lax.scan(
            lambda current_state, brownian_increment: scan_step(
                current_state, brownian_increment, control, magnitude, substep_duration),
            state,
            brownian_increments,
        )
        return final_state, substep_states

    step.with_history = step_with_history
    return step


class Dynamics:
    """Diffrax maps and batched exact discrete derivatives for seeds/reporting."""

    def __init__(self, case):
        self.case = case
        self._cache = {}

    def _maps(self, substeps):
        if substeps not in self._cache:
            step = integration_map(self.case, substeps)

            def raw(state, control, duration):
                return step(state, control, jnp.linalg.norm(control), duration)

            self._cache[substeps] = (
                jax.jit(jax.vmap(raw, in_axes=(1, 1, None), out_axes=1)),
                jax.jit(jax.vmap(jax.jacrev(raw, argnums=(0, 1)),
                                in_axes=(1, 1, None))),
            )
        return self._cache[substeps]

    def propagate(self, states, controls, step, substeps, with_jacobian=False):
        propagation, derivatives = self._maps(substeps)
        result = np.asarray(propagation(states, controls, step))
        if with_jacobian:
            a, b = derivatives(states, controls, step)
            return result, np.asarray(a), np.asarray(b)
        return result, None, None


def cholesky_lower(matrix, pivot_floor):
    """Match the original scalar Cholesky recursion, including pivot clamps."""
    factor = [[jnp.asarray(0.0) for _ in range(NX)] for _ in range(NX)]
    for row in range(NX):
        for column in range(row + 1):
            total = matrix[row, column]
            for inner in range(column):
                total = total - factor[row][inner] * factor[column][inner]
            factor[row][column] = (
                jnp.sqrt(jnp.maximum(total, pivot_floor)) if row == column
                else total / factor[column][column]
            )
    return jnp.stack([jnp.stack(row) for row in factor])


def spectral_radius(matrix, floor):
    """Smoothed analytic 3x3 largest eigenvalue."""
    q = jnp.trace(matrix)/3
    p1 = matrix[0, 1]**2 + matrix[0, 2]**2 + matrix[1, 2]**2
    p2 = jnp.sum((jnp.diag(matrix)-q)**2) + 2*p1
    p = jnp.sqrt(p2/6 + 1e-30)
    b = (matrix-q*jnp.eye(NU))/p
    det = (b[0, 0]*(b[1, 1]*b[2, 2]-b[1, 2]*b[2, 1])
           - b[0, 1]*(b[1, 0]*b[2, 2]-b[1, 2]*b[2, 0])
           + b[0, 2]*(b[1, 0]*b[2, 1]-b[1, 1]*b[2, 0]))
    r = jnp.clip(det/2, -1+1e-12, 1-1e-12)
    return jnp.sqrt(q + 2*p*jnp.cos(jnp.arccos(r)/3) + floor**2)


def gates_inner_sigma_points(control, options):
    """Six-point kappa=0 cubature set for delta_c_G | control, and its weight.

    Returns ``(executed_controls, weight)`` with ``executed_controls`` of shape
    ``(6, NU)`` holding ``control + delta_c_G^(ell,+/-)`` and ``weight = 1/6``.
    The formal central point ``delta_c_G = 0`` carries zero weight at kappa=0
    (Eq. (15)-(16) of the execution-error model) and is omitted, matching
    ``unscented_weights`` dropping no mass by doing so.
    """

    covariance = gates_execution_covariance(control, options)
    factor = jnp.linalg.cholesky(covariance + options.cholesky_jitter * jnp.eye(NU))
    offsets = jnp.sqrt(3.0) * factor
    executed = control[None, :] + jnp.concatenate((offsets.T, -offsets.T), axis=0)
    return executed, 1.0 / 6.0


def build_propagation_arc_map(case, options, normalization):
    dimension = augmented_dimension(options)
    outer_weights = jnp.asarray(unscented_weights(options.scaling_parameter, dimension))
    n_outer = outer_weights.shape[0]
    scale = jnp.asarray(normalization.scale)
    sde_substeps = int(options.sde_integrator_substeps)
    if sde_substeps != options.sde_integrator_substeps or sde_substeps < 1:
        raise ValueError("sde_integrator_substeps must be a positive integer")
    navigation_dimension = NP if navigation_error_enabled(options) else 0
    process_dimension = dimension - NX - navigation_dimension
    expected_process_dimension = (
        NW * sde_substeps
        if options.acceleration_diffusion_km_s32 > 0.0
        else 0
    )
    if process_dimension != expected_process_dimension:
        raise ValueError("inconsistent process-noise augmentation dimension")
    gates_enabled = gates_execution_error_enabled(options)
    n_inner = 6 if gates_enabled else 1
    # Product weights W_i * V_j, flattened outer-major to match the (n_outer,
    # n_inner) -> (n_outer*n_inner,) reshape used for every nested array below.
    inner_weight = (1.0 / 6.0) if gates_enabled else 1.0
    weights = jnp.repeat(outer_weights, n_inner) * inner_weight

    step = stochastic_integration_map(case, options)
    batch_step = jax.vmap(step, in_axes=(0, 0, 0, 0, None))
    noise_root = jnp.diag(jnp.sqrt(jnp.clip(
        dimension * np.diag(navigation_covariance(case, options, normalization)),
        0.0, None)))

    state_padding = navigation_dimension + process_dimension
    stochastic_columns = jnp.concatenate(
        (
            jnp.zeros((process_dimension, NX + navigation_dimension)),
            jnp.sqrt(dimension) * jnp.eye(process_dimension),
        ),
        axis=1,
    )

    def nest_gates(commanded):
        """Expand each of the ``n_outer`` commanded controls into its own
        conditional Gates sigma points, returning ``(n_outer*n_inner, NU)``
        executed controls in outer-major order."""

        if not gates_enabled:
            return commanded
        executed, _ = jax.vmap(lambda c: gates_inner_sigma_points(c, options))(commanded)
        return executed.reshape(n_outer * n_inner, NU)

    def arc(mean, covariance, feedforward, gain, duration):
        # Deliberately dimension*P, not (dimension+kappa)*P: preserve the source
        # even for non-default kappa, rather than silently correcting its UT.
        spread = cholesky_lower(
            dimension*covariance + options.cholesky_jitter*jnp.eye(NX),
            options.cholesky_jitter)
        state_columns = jnp.concatenate(
            (spread, jnp.zeros((NX, state_padding))), axis=1
        )
        measurement_parts = [spread[:NP]]
        if navigation_dimension:
            measurement_parts.append(noise_root)
        if process_dimension:
            measurement_parts.append(jnp.zeros((NP, process_dimension)))
        measurement_columns = jnp.concatenate(measurement_parts, axis=1)
        state_offsets = (scale[:, None]*state_columns).T
        control_offsets = (gain@measurement_columns).T
        states = jnp.concatenate((mean[None], mean+state_offsets, mean-state_offsets))
        # Commanded control per outer sigma point (normalised, i.e. c^(i)).
        commanded = jnp.concatenate((
            feedforward[None], feedforward+control_offsets, feedforward-control_offsets))
        # Conditional Gates execution error: each outer point's own commanded
        # control generates its own six-point inner cubature set (Eq. (6)-(17)
        # of the execution-error model). Disabled, this is the identity map.
        executed = nest_gates(commanded)
        controls = case.max_thrust_nd * executed
        # Outer-major repeat: the same state sigma point and Brownian draw is
        # shared across all n_inner Gates realizations conditioned on it.
        states_nested = jnp.repeat(states, n_inner, axis=0)

        if process_dimension:
            stochastic_points = jnp.concatenate(
                (
                    jnp.zeros((1, process_dimension)),
                    stochastic_columns.T,
                    -stochastic_columns.T,
                ),
                axis=0,
            )
            xi = stochastic_points.reshape(-1, sde_substeps, NW)
            substep_duration = duration / sde_substeps
            brownian_increments = jnp.sqrt(substep_duration) * xi
        else:
            brownian_increments = jnp.zeros((n_outer, sde_substeps, NW))
        brownian_increments_nested = jnp.repeat(brownian_increments, n_inner, axis=0)

        propagated = batch_step(
            states_nested,
            controls,
            jnp.linalg.norm(controls, axis=1),
            brownian_increments_nested,
            duration,
        )
        mean_next = weights@propagated
        deviations = (propagated-mean_next)/scale
        covariance_next = (weights[:, None]*deviations).T@deviations
        # Executed-control covariance from the same nested points (Eq. (26)-
        # (27)), representing the actually executed rather than the commanded
        # control; feeds the thrust chance constraint and the objective. Built
        # from `executed` (unit-ball, max-thrust-normalised), matching the
        # units of `feedforward`/`norm`, not from the absolute `controls` used
        # for dynamics propagation -- otherwise it would carry a spurious
        # extra factor of max_thrust_nd^2 relative to every other term the
        # chance constraint and objective combine it with.
        mean_executed = weights@executed
        executed_deviations = executed - mean_executed
        control_covariance = (weights[:, None]*executed_deviations).T@executed_deviations
        return (mean_next, (covariance_next+covariance_next.T)/2,
                (control_covariance+control_covariance.T)/2)

    return arc


def propagate_stochastic_moments(arc, means, feedforward, gains, covariance, steps):
    covariances, controls, propagated = [np.asarray(covariance)], [], [means[:, 0]]
    for k, duration in enumerate(steps):
        mu, p, t = arc(means[:, k], covariances[-1], feedforward[:, k], gains[k], duration)
        propagated.append(np.asarray(mu))
        covariances.append(np.asarray(p))
        controls.append(np.asarray(t))
    return np.stack(propagated, axis=1), np.stack(covariances), np.stack(controls)


def build_initial_guess(case, options, dynamics, normalization, ref_traj, arc):
    means = ref_traj.states.copy()
    means[:, 0] = case.x0_augmented_state
    feedforward = ref_traj.controls/case.max_thrust_nd
    a, b = normalized_arc_jacobians(
        case, dynamics, normalization, ref_traj.states, ref_traj.controls,
        ref_traj.steps, options.integrator_substeps)
    gain_scale = np.array([1/max(np.linalg.norm(matrix, 2), 1e-300) for matrix in b])
    gains = (seed_gains(options, a, b)[0] if options.warm_start_gains else
             -COLD_START_GAIN*np.tile(gain_scale[:, None, None], (1, NU, NP)))
    _, covariances, controls = propagate_stochastic_moments(
        arc, means, feedforward, gains, normalization.initial_covariance, ref_traj.steps)
    target_inv_std = np.sqrt([options.position_covariance_reduction]*3
                             + [options.velocity_covariance_reduction]*3)
    slack = np.eye(NP)-np.outer(target_inv_std, target_inv_std)*covariances[-1, :NP, :NP]
    vals, vecs = np.linalg.eigh((slack+slack.T)/2)
    margin = np.linalg.cholesky((vecs*np.clip(vals, 0, None))@vecs.T
                               + options.terminal_margin_floor**2*np.eye(NP))
    return InitialGuess(means, feedforward, gains, psqrt_spectral_radii(controls),
                        gain_scale, np.asarray(MTRI.pack(margin)))


def snopt_options(options, tag):
    if options.solver.lower() != "snopt":
        raise ValueError("This implementation uses pyOptSparse's SNOPT solver")
    work = Path(tempfile.mkdtemp(prefix=f"pyopt-cr3bp-{tag}-"))
    result = {
        "Major iterations limit": options.major_max_iter,
        "Minor iterations limit": options.minor_max_iter,
        "Iterations limit": options.minor_max_iter,
        "Major optimality tolerance": options.major_optimality_tol,
        "Major feasibility tolerance": options.major_feasibility_tol,
        "Minor feasibility tolerance": options.minor_feasibility_tol,
        "Function precision": 1e-12,
        "Major print level": options.print_level,
        "Minor print level": 0,
        "Print file": str(work/"SNOPT_print.out"),
        "Summary file": str(work/"SNOPT_summary.out"),
        "iSumm": options.summary_file if options.print_level else 0,
    }
    if options.partial_price is not None:
        if int(options.partial_price) < 1:
            raise ValueError("partial_price must be positive")
        result["Partial price"] = int(options.partial_price)
    if options.elastic_weight is not None:
        if not np.isfinite(options.elastic_weight) or options.elastic_weight <= 0:
            raise ValueError("elastic_weight must be finite and positive")
        result["Elastic weight"] = float(options.elastic_weight)
    return result


def solution_status(solution):
    info = solution.optInform
    if isinstance(info, dict):
        return int(info["value"]), str(info.get("text", ""))
    return int(info.value), str(info.message)


class SparseAssembler:
    """Scatter local AD blocks into COO without allocating a dense global Jacobian.

    Patterns are constructed from structural masks, never from numerical zeros
    in a seed. Sensitivity callbacks keep every declared entry, even when zero.
    """

    def __init__(self, sizes, row_sizes):
        self.sizes, self.row_sizes = sizes, row_sizes
        self.parts = {}
        self.objective = {key: np.zeros(size) for key, size in sizes.items()}

    def add(self, output, variable, rows, columns, values, mask=None):
        values = np.atleast_2d(values)
        if mask is None:
            mask = np.ones(values.shape, dtype=bool)
        i, j = np.nonzero(mask)
        if not len(i):
            return
        part = (np.asarray(rows)[i], np.asarray(columns)[j], values[i, j])
        self.parts.setdefault(output, {}).setdefault(variable, []).append(part)

    def finish(self):
        result = {"objective": self.objective}
        for output, blocks in self.parts.items():
            result[output] = {}
            for variable, parts in blocks.items():
                rows, columns, values = [np.concatenate(items) for items in zip(*parts)]
                result[output][variable] = {
                    "coo": [rows.astype(np.intc), columns.astype(np.intc), values],
                    "shape": (self.row_sizes[output], self.sizes[variable]),
                }
        return result


class TrajectoryNLP:
    """Arc-local reverse AD, vmap batching, and explicit global sparse assembly.

    Both NLPs share mean/control ordering with the double-integrator script.
    Stochastic covariances at node zero are fixed data, as in the CR3BP source.
    Only nodes 1..N own Cholesky variables; gains are K_hat = K_tilde/gain_scale.
    """

    def __init__(self, case, options, ref_traj, normalization=None, seed=None):
        self.case, self.options, self.ref = case, options, ref_traj
        self.n = n = ref_traj.n_arcs
        self.stochastic = seed is not None
        self.normalization, self.seed = normalization, seed
        if options.control_norm_eps <= 0:
            raise ValueError("control_norm_eps must be positive")
        if np.any(ref_traj.steps <= 0):
            raise ValueError("Arc durations must be positive")
        if self.stochastic and np.ptp(ref_traj.steps) > 1e-9*np.max(ref_traj.steps):
            raise ValueError("The unweighted stochastic cost requires equal-duration arcs")
        self.sizes = {"means": NX*(n+1), "feedforward": NU*n}
        self.rows = {"mean_defects": NX*n, "control_chance": n}
        if self.stochastic:
            self.arc = jax.jit(build_propagation_arc_map(case, options, normalization))
            _, covariance_seed, _ = propagate_stochastic_moments(
                self.arc, seed.means, seed.feedforward, seed.gains,
                normalization.initial_covariance, ref_traj.steps)
            self.variances, self.std, self.inv_std = state_covariance_node_scaling(covariance_seed)
            chol = np.stack([np.linalg.cholesky(
                covariance_seed[k]*np.outer(self.inv_std[k], self.inv_std[k]))
                for k in range(1, n+1)])
            self.x0 = {
                "means": seed.means.ravel(), "feedforward": seed.feedforward.ravel(),
                "gains": (seed.gains/seed.gain_scale[:, None, None]).ravel(),
                "cholesky_factor": chol[:, LTRI.rows, LTRI.columns].ravel(),
                "terminal_margin": seed.terminal_margin.copy(),
            }
            self.initial_factor = np.asarray(LTRI.pack(np.linalg.cholesky(
                normalization.initial_covariance)*self.inv_std[0, :, None]))
            self.sizes.update(gains=n*NU*NP, cholesky_factor=n*NL, terminal_margin=NM)
            self.rows.update(state_covariance_defects=n*NL, terminal_residual=NM)
        else:
            self.step = integration_map(case, options.integrator_substeps)
            self.x0 = {"means": ref_traj.states.ravel(),
                       "feedforward": (ref_traj.controls/case.max_thrust_nd).ravel()}

        # Only these small maps are differentiated. No dense trajectory Jacobian.
        self.local_values = jax.jit(jax.vmap(self.local_output, in_axes=(0, 0)))
        self.local_derivatives = jax.jit(jax.vmap(jax.jacrev(self.local_output, argnums=0),
                                                in_axes=(0, 0)))
        if self.stochastic:
            self.cov_derivatives = jax.jit(jax.vmap(jax.jacrev(self.packed_covariance)))
            self.terminal_derivatives = jax.jit(jax.jacrev(self.terminal_output))
        self.values = jax.jit(self.functions)
        # The same scatter builds the topology and the runtime sensitivity data.
        width = NX+NU+(NU*NP+NL if self.stochastic else 0)
        height = NX+(NL if self.stochastic else 0)+2
        self.pattern = self.assemble(np.ones((n, height, width)),
            np.ones((n, NL, NL)) if self.stochastic else None,
            np.ones((NM, NL+NM)) if self.stochastic else None)
        self.problem = self._optimization()

    def packed_covariance(self, entries):
        factor = LTRI.matrix(entries)
        return LTRI.pack(factor@factor.T)

    def local_inputs(self, x):
        n = self.n
        parts = [x["means"].reshape(NX, n+1)[:, :-1].T,
                 x["feedforward"].reshape(NU, n).T]
        if self.stochastic:
            entries = x["cholesky_factor"].reshape(n, NL)
            parts += [x["gains"].reshape(n, NU*NP),
                      jnp.concatenate((self.initial_factor[None], entries[:-1]))]
        return jnp.concatenate(parts, axis=1)

    def local_output(self, values, k):
        mean, feed = values[:NX], values[NX:NX+NU]
        norm = jnp.sqrt(jnp.dot(feed, feed)+self.options.control_norm_eps**2)
        if not self.stochastic:
            propagated = self.step(mean, self.case.max_thrust_nd*feed,
                self.case.max_thrust_nd*norm, jnp.asarray(self.ref.steps)[k])
            return jnp.concatenate((propagated, jnp.array([norm, jnp.asarray(self.ref.steps)[k]*norm**2])))
        gain = values[NX+NU:NX+NU+NU*NP].reshape(NU, NP)*jnp.asarray(self.seed.gain_scale)[k]
        factor = jnp.asarray(self.std)[k, :, None]*LTRI.matrix(values[-NL:])
        covariance = factor@factor.T
        propagated, covariance_next, control_covariance = self.arc(
            mean, covariance, feed, gain, jnp.asarray(self.ref.steps)[k])
        inv = jnp.asarray(self.inv_std)[k+1]
        covariance_next = covariance_next*inv[:, None]*inv[None, :]
        chance = norm + psi_inverse(NU, self.options.violation_parameter)*spectral_radius(
            control_covariance, self.options.spectral_radius_floor)
        objective = norm + jnp.trace(covariance[:NP, :NP])/self.options.bryson_sigma_factor**2 + jnp.trace(control_covariance)
        return jnp.concatenate((propagated, LTRI.pack(covariance_next), jnp.array([chance, objective])))

    def terminal_output(self, values):
        """Terminal covariance Loewner order P'_N <= Dt via a Cholesky-residual encoding.

        Writing the normalised slack terminal_slack = I_6 - Dt^-1/2 P'_N Dt^-1/2,
        any PSD matrix admits a factorisation terminal_slack = G G^T with G
        lower-triangular, so constraining (terminal_slack - G G^T) to zero is
        equivalent to terminal_slack >= 0, i.e. P'_N <= Dt exactly -- jointly
        over position, velocity, and their cross-covariance.
        """

        factor = LTRI.matrix(values[:NL])*self.std[-1, :, None]
        covariance = factor@factor.T
        inv = jnp.sqrt(jnp.array([self.options.position_covariance_reduction]*3
                                 + [self.options.velocity_covariance_reduction]*3))
        target = covariance[:NP, :NP]*inv[:, None]*inv[None, :]
        margin = MTRI.matrix(values[NL:NL+NM])
        residual = jnp.eye(NP)-target-margin@margin.T
        return MTRI.pack(residual)

    def terminal_inputs(self, x):
        return jnp.concatenate((x["cholesky_factor"].reshape(self.n, NL)[-1], x["terminal_margin"]))

    def functions(self, x):
        local = self.local_values(self.local_inputs(x), jnp.arange(self.n))
        means = x["means"].reshape(NX, self.n+1)
        result = {
            "mean_defects": (means[:, 1:]-local[:, :NX].T).ravel(),
            "control_chance": local[:, -2], "objective": jnp.sum(local[:, -1]),
        }
        if self.stochastic:
            entries = x["cholesky_factor"].reshape(self.n, NL)
            result["state_covariance_defects"] = (
                jax.vmap(self.packed_covariance)(entries)-local[:, NX:NX+NL]).ravel()
            result["terminal_residual"] = self.terminal_output(self.terminal_inputs(x))
        return result

    def assemble(self, local, covariance, terminal):
        n = self.n
        result = SparseAssembler(self.sizes, self.rows)
        for k in range(n):
            variables = [
                ("means", np.arange(NX)*(n+1)+k, slice(0, NX)),
                ("feedforward", np.arange(NU)*n+k, slice(NX, NX+NU)),
            ]
            if self.stochastic:
                variables.append(("gains", k*NU*NP+np.arange(NU*NP), slice(NX+NU, NX+NU+NU*NP)))
                if k:
                    variables.append(("cholesky_factor", (k-1)*NL+np.arange(NL), slice(-NL, None)))
            for name, columns, inputs in variables:
                mean_mask = np.ones((NX, len(columns)), dtype=bool)
                if name == "means":
                    mean_mask[-1, :-1] = False  # mass flow does not observe position/velocity
                if name == "cholesky_factor":
                    mean_mask[-1, LTRI.rows == NP] = False
                result.add("mean_defects", name, np.arange(NX)*n+k, columns,
                           -local[k, :NX, inputs], mean_mask)
                if self.stochastic:
                    result.add("state_covariance_defects", name, k*NL+np.arange(NL), columns,
                               -local[k, NX:NX+NL, inputs])
                # Thrust chance and running cost do not depend on the mean or
                # the mass row of the covariance factor. No false global blocks.
                if name != "means":
                    mask = np.ones((1, len(columns)), dtype=bool)
                    if name == "cholesky_factor":
                        mask[:, LTRI.rows == NP] = False
                    result.add("control_chance", name, [k], columns, local[k, -2, inputs][None], mask)
                    result.objective[name][columns] += local[k, -1, inputs]*mask[0]
            result.add("mean_defects", "means", np.arange(NX)*n+k,
                       np.arange(NX)*(n+1)+k+1, np.eye(NX), np.eye(NX, dtype=bool))
            if self.stochastic:
                result.add("state_covariance_defects", "cholesky_factor", k*NL+np.arange(NL),
                    k*NL+np.arange(NL), covariance[k], LTRI.covariance_pattern())
        if self.stochastic:
            count = self.rows["terminal_residual"]
            mask = LTRI.covariance_pattern()[:NM] if count == NM else np.arange(NL)[None, :] < NM
            result.add("terminal_residual", "cholesky_factor", np.arange(count),
                (n-1)*NL+np.arange(NL), terminal[:count, :NL], mask)
            result.add("terminal_residual", "terminal_margin", np.arange(count), np.arange(NM),
                       terminal[:count, NL:NL+NM], MTRI.covariance_pattern())
        return result.finish()

    def objconfun(self, x):
        values = {key: np.asarray(value) for key, value in self.values(x).items()}
        return values, not all(np.all(np.isfinite(v)) for v in values.values())

    def sens(self, x, funcs=None):
        local = np.asarray(self.local_derivatives(self.local_inputs(x), jnp.arange(self.n)))
        cov = terminal = None
        if self.stochastic:
            entries = jnp.asarray(x["cholesky_factor"]).reshape(self.n, NL)
            cov = np.asarray(self.cov_derivatives(entries))
            terminal = np.asarray(self.terminal_derivatives(self.terminal_inputs(x)))
        result = self.assemble(local, cov, terminal)
        finite = np.all(np.isfinite(local))
        if self.stochastic:
            finite = finite and np.all(np.isfinite(cov)) and np.all(np.isfinite(terminal))
        return result, not finite

    def _optimization(self):
        problem = Optimization("CR3BP stochastic" if self.stochastic else "CR3BP energy reference", self.objconfun)
        for variable, size in self.sizes.items():
            lower = np.full(size, -np.inf)
            if variable == "cholesky_factor":
                lower.reshape(self.n, NL)[:, LTRI.diagonal] = 0
            elif variable == "terminal_margin":
                lower[MTRI.diagonal] = self.options.terminal_margin_floor
            problem.addVarGroup(variable, size, value=self.x0[variable], lower=lower)
        for name, columns, target in (
            ("boundary_start", np.arange(NX)*(self.n+1), self.case.x0_augmented_state),
            ("boundary_end", np.arange(NP)*(self.n+1)+self.n,
             terminal_mean_state_target(self.case, self.options, self.ref)),
        ):
            problem.addConGroup(name, len(columns), lower=target, upper=target,
                linear=True, wrt=["means"], jac={"means": {
                    "coo": [np.arange(len(columns), dtype=np.intc), columns.astype(np.intc), np.ones(len(columns))],
                    "shape": (len(columns), self.sizes["means"]),
                }})
        for name, size in self.rows.items():
            pattern = self.pattern[name]
            inequality = name == "control_chance"
            problem.addConGroup(name, size, lower=None if inequality else 0,
                upper=1 if name == "control_chance" else 0,
                wrt=list(pattern), jac=pattern)
        problem.addObj("objective")
        if self.options.print_sparsity:
            problem.printSparsity()
        return problem

    def solve(self, *, feasibility_tolerance=None):
        # Warm compilation before entering Fortran; expose failures directly.
        print(f"[{self.case.test_case_id}] compiling {'stochastic' if self.stochastic else 'reference'} "
              f"values and local derivatives on {jax.default_backend()} ({self.n} arcs)", flush=True)
        if self.objconfun(self.x0)[1] or self.sens(self.x0)[1]:
            raise FloatingPointError("Nonfinite values or derivatives at initial guess")
        settings = snopt_options(self.options, self.case.test_case_id)
        if feasibility_tolerance is not None:
            settings["Major feasibility tolerance"] = feasibility_tolerance
            settings["Minor feasibility tolerance"] = feasibility_tolerance
        solution = SNOPT(options=settings)(self.problem, sens=self.sens)
        code, message = solution_status(solution)
        print(f"[{self.case.test_case_id}] SNOPT {code}: {message}", flush=True)
        return solution


def reoptimize_reference_trajectory(case, options, source, dynamics=None):
    nlp = TrajectoryNLP(case, options, source)
    solution = nlp.solve()
    means = np.asarray(solution.xStar["means"]).reshape(NX, source.n_arcs+1)
    controls = np.asarray(solution.xStar["feedforward"]).reshape(NU, source.n_arcs)*case.max_thrust_nd
    return ReferenceTraj(source.node_times.copy(), source.steps.copy(), means, controls,
                         float(case.m0_wet*(means[6, 0]-means[6, -1])))


def recover_solution(nlp, optimizer_solution):
    x = optimizer_solution.xStar
    n, options, seed = nlp.n, nlp.options, nlp.seed
    means = np.asarray(x["means"]).reshape(NX, n+1)
    feed = np.asarray(x["feedforward"]).reshape(NU, n)
    gains = np.asarray(x["gains"]).reshape(n, NU, NP)*seed.gain_scale[:, None, None]
    entries = np.asarray(x["cholesky_factor"]).reshape(n, NL)
    covariances = [nlp.normalization.initial_covariance]
    for k in range(n):
        factor = np.asarray(LTRI.matrix(entries[k]))*nlp.std[k+1, :, None]
        covariances.append(factor@factor.T)
    covariances = np.stack(covariances)
    controls, predicted_cov = [], []
    for k in range(n):
        _, p, t = nlp.arc(means[:, k], covariances[k], feed[:, k], gains[k], nlp.ref.steps[k])
        controls.append(np.asarray(t))
        predicted_cov.append(np.asarray(p))
    controls = np.stack(controls)
    values, fail = nlp.objconfun(x)
    code, _ = solution_status(optimizer_solution)
    norm = np.sqrt(np.sum(feed**2, axis=0)+options.control_norm_eps**2)
    radius = np.asarray(jax.vmap(lambda t: spectral_radius(t, options.spectral_radius_floor))(controls))
    boundary_error = max(np.max(np.abs(means[:, 0]-nlp.case.x0_augmented_state)),
        np.max(np.abs(means[:NP, -1]-terminal_mean_state_target(nlp.case, options, nlp.ref))))
    equality_error = max(boundary_error, *(float(np.max(np.abs(values[key])))
        for key in ("mean_defects", "state_covariance_defects", "terminal_residual")))
    chance_violation = max(0.0, float(np.max(values["control_chance"])-1))
    bound_violation = max(
        0.0,
        float(-np.min(entries[:, LTRI.diagonal])),
        float(options.terminal_margin_floor - np.min(np.asarray(x["terminal_margin"])[MTRI.diagonal])),
    )
    terminal_violation = float(np.max(np.abs(values["terminal_residual"])))
    diagnostics = {
        "solver_converged": float(code == 1 and not fail),
        "absolute_feasibility_satisfied": float(max(equality_error, chance_violation, bound_violation)
                                               <= options.major_feasibility_tol),
        "converged": float(code == 1 and not fail and max(equality_error, chance_violation, bound_violation)
                           <= options.major_feasibility_tol),
        "snopt_inform": float(code), "max_equality_residual": equality_error,
        "max_control_chance_violation": chance_violation, "max_bound_violation": bound_violation,
        "max_terminal_constraint_violation": terminal_violation,
        "state_matching_defect_nd": float(np.max(np.abs(values["mean_defects"]))),
        "state_covariance_matching_defect": float(np.max(np.abs(np.stack(predicted_cov)-covariances[1:]))),
        "scaled_state_covariance_matching_defect": float(np.max(np.abs(values["state_covariance_defects"]))),
        "state_covariance_scale_min_variance": float(np.min(nlp.variances)),
        "state_covariance_scale_max_variance": float(np.max(nlp.variances)),
        "objective_mean_control": float(nlp.ref.steps@norm),
        "objective_state_covariance": float(sum(nlp.ref.steps[k]*np.trace(covariances[k, :NP, :NP])
            / options.bryson_sigma_factor**2 for k in range(n))),
        "objective_control_covariance": float(nlp.ref.steps@np.trace(controls, axis1=1, axis2=2)),
    }
    return RobustSolution(means, feed, gains, radius, covariances, controls,
                          float(values["objective"]), nlp.variances, diagnostics)


def solve_stochastic_nlp(nlp):
    """Solve once with the configured tolerances and report the returned residuals."""
    raw = nlp.solve()
    return raw, recover_solution(nlp, raw)


def solve_rocp(case, options, ref_traj, initial_guess, normalization):
    nlp = TrajectoryNLP(case, options, ref_traj, normalization, initial_guess)
    return solve_stochastic_nlp(nlp)[1]


# Plot/export layouts mirror the original script. Keep these local because its
# plot_outputs creates a CasADi Dynamics object internally, ignoring the supplied
# dynamics for plotted rollouts. Every rollout here uses the Diffrax object.

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
    rollout_history = np.asarray(
        monte_carlo_result["plot_trajectory_history"], dtype=float
    )
    rollout_applied_control_deviation_norm_n = np.asarray(
        monte_carlo_result["plot_applied_control_deviation_norm_n"],
        dtype=float,
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

    # 2 - nominal thrust and the applied-control deviation norm. The Monte
    # Carlo values use the saturated applied control, while the predicted
    # bound is formed for the unsaturated feedback vector. Radial projection
    # onto the thrust ball cannot increase its distance from the feasible
    # nominal control, so the same bound remains conservative after saturation.
    nominal_thrust_n = np.linalg.norm(solution.feedforward, axis=0) * case.max_thrust_nd * case.thrust_unit
    feedback_n = psi_inv * solution.radius * case.max_thrust_nd * case.thrust_unit
    step_days, nominal_step = plotter.plot_zero_order_hold_data(
        node_days, nominal_thrust_n
    )
    _, feedback_bound_step = plotter.plot_zero_order_hold_data(
        node_days, feedback_n
    )
    _, rollout_deviation_norm_step = plotter.plot_zero_order_hold_data(
        node_days, rollout_applied_control_deviation_norm_n
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(1.5 * Plotter.WIDE_FIGSIZE[0], 1.5 * Plotter.WIDE_FIGSIZE[1]),
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
        fontsize=8,
        borderaxespad=0.0,
    )

    axis = axes[1]
    rollout_lines = axis.plot(
        step_days,
        rollout_deviation_norm_step,
        color=Plotter.DARK_GREY,
        alpha=0.28,
        lw=0.48,
        zorder=1,
    )
    if rollout_lines:
        rollout_lines[0].set_label("Monte Carlo")
    axis.plot(
        step_days,
        feedback_bound_step,
        color=Plotter.BLUE,
        ls="--",
        lw=0.82,
        label="Predicted bound",
        zorder=2,
    )
    axis.set_xlabel("time [days]")
    axis.set_ylabel(r"$\|T_{\mathrm{applied}}-\bar{T}\|$ [N]")
    axis.set_ylim(
        0.0,
        1.05
        * max(
            float(np.max(feedback_n)),
            float(np.max(rollout_applied_control_deviation_norm_n)),
        ),
    )
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
        fontsize=Plotter.DIAGNOSTIC_LABEL_SIZE,
        borderaxespad=0.0,
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

    lagrange = get_collinear_lagrange_points(case)
    departure_orbit = None
    target_orbit = None
    if case.departure_period_nd is not None:
        departure_orbit = propagate_periodic_orbit(
            case, case.x0_augmented_state, case.departure_period_nd
        )
    # A truncated transfer ends at a relative reference node, not at the full
    # target periodic orbit.
    if options.truncated_uniform_mesh_arcs is None and case.target_period_nd is not None:
        target_orbit = propagate_periodic_orbit(
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

    # The Monte Carlo trajectory bundle carries one entry per RK9 substep
    # (see monte_carlo_rollouts), landing at uniform arc fractions
    # j/rollout_substeps for j = 1..rollout_substeps -- exactly the fraction
    # grid densified_mean_trajectory samples at, so the two align column for
    # column when built with matching samples_per_arc.
    rollout_substeps_override = options.monte_carlo_sde_integrator_substeps
    rollout_substeps = int(
        rollout_substeps_override if rollout_substeps_override is not None
        else options.sde_integrator_substeps
    )
    rollout_dense_means = (
        dense_means
        if rollout_substeps == dense_samples_per_arc
        else densified_mean_trajectory(
            case, options, dynamics, solution, ref_traj.steps,
            samples_per_arc=rollout_substeps,
        )
    )

    magnification = 50.0
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
        # Compare each Monte Carlo substep against the mean state at that same
        # substep (not just the coarse arc-node mean), since rollout_history
        # now resolves the interior of each arc rather than only its endpoints.
        displayed_0 = rollout_dense_means[axis_0, None, :] + magnification * (
            rollout_history[axis_0] - rollout_dense_means[axis_0, None, :]
        )
        displayed_1 = rollout_dense_means[axis_1, None, :] + magnification * (
            rollout_history[axis_1] - rollout_dense_means[axis_1, None, :]
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
        initial_position_std_km=INITIAL_POSITION_STD_KM,
        initial_velocity_std_ms=INITIAL_VELOCITY_STD_KM_S * 1e3,
        final_position_std_km=FINAL_POSITION_STD_KM,
        final_velocity_std_ms=FINAL_VELOCITY_STD_KM_S * 1e3,
        navigation_position_std_km=options.navigation_position_std_km,
        navigation_velocity_std_ms=options.navigation_velocity_std_km_s * 1e3,
        gates_proportional_magnitude_std=options.gates_proportional_magnitude_std,
        gates_proportional_pointing_std=options.gates_proportional_pointing_std,
        acceleration_diffusion_km_s32=options.acceleration_diffusion_km_s32,
        acceleration_diffusion_nd=acceleration_diffusion_nd(case, options),
        sde_integrator=np.asarray("XMDS2_RK9"),
        sde_integrator_substeps=options.sde_integrator_substeps,
        ut_augmented_dimension=augmented_dimension(options),
        ut_sigma_points=2 * augmented_dimension(options) + 1,
        bryson_sigma_factor=options.bryson_sigma_factor,
        # Store the actual NLP weights, rather than the original exporter's
        # uniformly rescaled pair from an earlier objective convention.
        bryson_state_weight=np.eye(NP) / options.bryson_sigma_factor**2,
        bryson_control_weight=np.eye(NU),
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
        monte_carlo_sampling=np.asarray("independent_pseudorandom_gaussian"),
        monte_carlo_samples=options.monte_carlo_samples,
        monte_carlo_cumulative=monte_carlo_result["cumulative_control_effort"],
        monte_carlo_violation_fraction=monte_carlo_result["violation_fraction"],
        monte_carlo_terminal_covariance=monte_carlo_result["terminal_covariance"],
        monte_carlo_plot_trajectory_history=monte_carlo_result[
            "plot_trajectory_history"
        ],
        monte_carlo_plot_applied_control_deviation_norm_n=monte_carlo_result[
            "plot_applied_control_deviation_norm_n"
        ],
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


def print_summary(case, diagnostics, psi_inv, options):
    """Report the configured probability rather than legacy hardcoded labels."""
    print(f"\n[{case.test_case_id}] JAX/Diffrax/pyOptSparse, {int(diagnostics['n_arcs'])} arcs")
    print(
        "  Process model: XMDS2 RK9, "
        f"{int(diagnostics['sde_integrator_substeps'])} substeps, "
        f"sigma_a={diagnostics['acceleration_diffusion_km_s32']:.6g} "
        "km s^(-3/2), "
        f"{int(diagnostics['ut_sigma_points'])} UT sigma points"
    )
    for label, key in (
        ("Objective", "objective_nd"),
        ("SNOPT status", "snopt_inform"),
        ("Converged and independently feasible", "converged"),
        ("Maximum equality residual", "max_equality_residual"),
        ("Maximum chance-constraint violation", "max_control_chance_violation"),
        ("Terminal mean error", "terminal_componentwise_mean_error_nd"),
        ("Terminal covariance / target, largest eigenvalue", "terminal_covariance_max_eigenvalue"),
        ("Monte Carlo terminal covariance / target, largest eigenvalue", "monte_carlo_terminal_max_eigenvalue"),
    ):
        print(f"  {label}: {diagnostics[key]:.10g}")
    confidence = 100*(1-options.violation_parameter)
    print(f"  {confidence:g}th percentile of Monte Carlo effort: "
          f"{diagnostics['monte_carlo_percentile_nd']:.8g} (nondimensional)")
    print(f"  Worst-arc Gates-inclusive commanded-thrust violation frequency: "
          f"{diagnostics['monte_carlo_max_violation_fraction']:.4%}; "
          f"configured limit: {options.violation_parameter:.2%}; chi-square multiplier: {psi_inv:.6g}")


def run_test_case(test_case_id, options=None):
    options = Options() if options is None else options
    case = CASE_REGISTRY[test_case_id]()
    print(
        f"[{test_case_id}] XMDS2 RK9 process model: "
        f"sigma_a={options.acceleration_diffusion_km_s32:.6g} km s^(-3/2), "
        f"sigma_a_nd={acceleration_diffusion_nd(case, options):.6g}, "
        f"substeps={options.sde_integrator_substeps}, "
        f"UT dimension={augmented_dimension(options)} "
        f"({2 * augmented_dimension(options) + 1} sigma points)",
        flush=True,
    )
    dynamics = Dynamics(case)
    normalization = build_normalization(case)
    source = load_ref_traj(case, options)
    ref_traj = reoptimize_reference_trajectory(case, options, source)
    arc = jax.jit(build_propagation_arc_map(case, options, normalization))
    seed = build_initial_guess(case, options, dynamics, normalization, ref_traj, arc)
    solution = solve_rocp(case, options, ref_traj, seed, normalization)
    monte_carlo = monte_carlo_rollouts(case, options, normalization,
                                              solution, ref_traj.steps)
    psi = psi_inverse(NU, options.violation_parameter)
    diagnostics = {**compute_diagnostics(case, options, ref_traj, solution,
                    normalization, monte_carlo, psi), **solution.diagnostics}
    print_summary(case, diagnostics, psi, options)
    name = case.test_case_id
    if options.truncated_uniform_mesh_arcs is not None:
        name += f"_{options.truncated_uniform_mesh_arcs}_arcs_truncated"
    save_outputs(case, options, ref_traj, solution, normalization, monte_carlo,
                           diagnostics, psi, OUTPUT_DIR/name, dynamics)
    return diagnostics


def main():
    case_ids = ("lyapunov_l1_to_l2", "nrho_l2_to_dro", "halo_l2_to_halo_l1")
    options = Options(truncated_uniform_mesh_arcs=None)
    print(f"JAX {jax.__version__}; Diffrax {diffrax.__version__}; devices: {jax.devices()}", flush=True)
    for case_id in case_ids:
        run_test_case(case_id, options)


if __name__ == "__main__":
    main()
