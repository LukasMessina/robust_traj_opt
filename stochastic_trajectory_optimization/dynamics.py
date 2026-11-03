"""JAX propagation, reference data and uncertainty models."""

from __future__ import annotations


import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
from dataclasses import dataclass, field
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import chi2
from deterministic_trajectory_optimization.models import (
    TestCase,
    CASE_REGISTRY,
    EarthMars,
)


jax.config.update("jax_enable_x64", True)


from integrator.stochastic import (
    step_adjoint as step_adjoint,
    vector_field as vector_field,
    integration_map as integration_map,
    PrescribedWienerPath as PrescribedWienerPath,
    stochastic_integration_map as stochastic_integration_map,
    Dynamics as Dynamics,
)

STOCHASTIC_CASE_REGISTRY = {"earth_mars": EarthMars, **CASE_REGISTRY}


ROOT = Path(__file__).resolve().parents[1]


EARTH_MARS_REFERENCE_DIR = (
    ROOT / "output/earth_mars_deterministic_traj_opt/energy_optimal"
)


NX = 7  # augmented state dimension


NU = 3  # control dimension


NP = 6  # primed (position-velocity) state dimension


NW = 3  # independent acceleration Wiener processes


ACCELERATION_DIFFUSION_KM_S32 = 1e-10  # [km s^(-3/2)]


GATES_PROPORTIONAL_MAGNITUDE_STD = 8.0e-3  # sigma_2 [-]


GATES_PROPORTIONAL_POINTING_STD = 9.5993e-3  # sigma_4 [rad]


REFERENCE_DIR = Path("output/cr3bp_deterministic_traj_opt/energy_optimal")


INITIAL_POSITION_STD_KM = 50.0


INITIAL_VELOCITY_STD_KM_S = 1.0e-3


FINAL_POSITION_STD_KM = 10.0


FINAL_VELOCITY_STD_KM_S = 1.0e-4


UNIFORM_ARCS_BY_CASE: dict[str, int] = {
    "earth_mars": 100,
    "nrho_l2_to_dro": 85,
    "halo_l2_to_halo_l1": 80,
    "lyapunov_l1_to_l2": 48,
}


COLD_START_GAIN = 1e-3


INITIAL_MASS_STD = 1e-3  # [kg]


MASS_SCALE = 1e-3  # [kg]


@dataclass(frozen=True)
class Options:
    """Numerical propagation, uncertainty and validation settings."""

    initial_position_std_km: float = INITIAL_POSITION_STD_KM
    initial_velocity_std_km_s: float = INITIAL_VELOCITY_STD_KM_S
    initial_mass_std_kg: float = INITIAL_MASS_STD
    mass_scale_kg: float = MASS_SCALE

    # Number of equal-duration arcs the reference is resampled onto.
    # Node states are interpolated from the saved dense integrated trajectory;
    # a separate shooting solve restores dynamic consistency on the new mesh.
    uniform_mesh_arcs: int | dict[str, int] = field(
        default_factory=lambda: dict(UNIFORM_ARCS_BY_CASE)
    )
    # When set, restrict the uniform mesh arcs to the first specified number of arcs
    # and use the state at that relative endpoint as the terminal
    # mean target. None preserves the complete uniform mesh.
    truncated_uniform_mesh_arcs: int | None = None
    # Control-norm regularizer: the thrust magnitude is carried as
    # sqrt(u'u + eps_1^2) on the up-to-the-unit control, everywhere it appears.
    control_norm_eps: float = 1e-5
    # None follows control_norm_eps, matching MadNLP; zero restores plain norm.
    mass_flow_smoothing: float | None = None
    # True: mass_flow_smoothing on the outer sigma points whose command carries feedback only (state columns of
    # the position/velocity block and navigation columns); the others command the feedforward and keep ||u||.
    mass_flow_smoothing_feedback_only: bool = False
    bryson_sigma_factor: float = 3.0
    # Factor of trace(control covariance) in the stochastic cost; the TVLQR seed
    # uses the same control weight R~ = control_covariance_weight * I. 1 is the
    # original Bryson-rule cost (CR3BP cases); Earth--Mars sets 1000 in
    # default_options, where the unweighted feedback-gain directions are nearly
    # flat and stall the interior-point (MadNLP) solves.
    control_covariance_weight: float = 1.0
    # Factor w_P of trace(P)/c^2 in the stochastic cost and of the TVLQR seed's
    # running state weight Q~ = w_P I/c^2 (the terminal seed weight is unchanged).
    # Per-case values in default_options (same as MadNLP's TRACKING_WEIGHTS).
    state_covariance_weight: float = 1.0
    # "standard": sum of ||S||_eps + tr(P)/c^2 + w tr(T) per arc. "quantile": sum of the
    # thrust chance left-hand side ||S||_eps + psi spectral_radius(T), the (1 - beta)
    # quantile bound of the executed thrust magnitude (same as the MadNLP option).
    objective: str = "standard"
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
    eigenvalue_smoothing: float = 1e-6
    spectral_radius_floor: float = 1e-6

    # Floor on diag(G) in the terminal covariance constraint (see terminal_output):
    # I_6 - Dt^-1/2 P'_N Dt^-1/2 = G G^T, G lower-triangular. Any PSD left-hand
    # side admits a factor with strictly positive pivots; it
    # exists purely to keep G -> G G^T non-degenerate near diag(G) = 0, where the
    # constraint Jacobian would otherwise lose rank.
    terminal_margin_floor: float = 1e-5

    # Box bound |K_hat| <= gain_bound on the scaled feedback gains K_hat = K_tilde/gain_scale
    # (gain_scale = 1/||B_tilde_k[:6]||_2). Inactive at the optimum of every case (largest
    # |K_hat| below 4); it keeps the first interior-point steps of the MadNLP GPU solve bounded.
    # None removes the bound.
    gain_bound: float | None = 50.0

    # Warm-start the feedback gains from the Bryson-rule Riccati recursion. With
    # this disabled the gains start at zero, the covariance grows open-loop, and
    # the terminal constraint begins many orders of magnitude violated. Starting
    # with a zero gain is still recoverable, but consumes a lot of additional iterations.
    warm_start_gains: bool = True

    monte_carlo_samples: int = 50000
    monte_carlo_seed: int = 42


def default_options(case_id: str) -> Options:
    """Earth--Mars dispersions, navigation and Gates errors (chosen for the earlier
    reference transfer and retained for this benchmark) with the existing
    objective and diffusion."""
    if case_id == "earth_mars":
        return Options(
            initial_position_std_km=30000.0,
            initial_velocity_std_km_s=0.020,
            initial_mass_std_kg=INITIAL_MASS_STD,
            mass_scale_kg=MASS_SCALE,
            position_covariance_reduction=225.0,
            # 20 m/s / sqrt(100) = 2 m/s at arrival (unchanged target).
            velocity_covariance_reduction=100.0,
            navigation_position_std_km=200.0,
            navigation_velocity_std_km_s=0.0001,
            gates_proportional_magnitude_std=GATES_PROPORTIONAL_MAGNITUDE_STD,
            gates_proportional_pointing_std=GATES_PROPORTIONAL_POINTING_STD,
            violation_parameter=0.01,
            **TRACKING_WEIGHTS[case_id],
        )
    if case_id == "nrho_l2_to_dro":
        # Terminal target 6 km / 0.05 m/s (sigma_0 = 50 km / 1 m/s): the 10 km / 0.1 m/s target was inactive
        # (arrival dispersion about half of it), leaving a flat direction between the solvers' optima.
        return Options(
            position_covariance_reduction=(50.0 / 6.0) ** 2,
            velocity_covariance_reduction=400.0,
            **TRACKING_WEIGHTS[case_id],
        )
    return Options(**TRACKING_WEIGHTS.get(case_id, {}))


TRACKING_WEIGHTS = {
    "lyapunov_l1_to_l2": dict(
        state_covariance_weight=0.019944701695746685,
        control_covariance_weight=33.55903293181279,
    ),
    "halo_l2_to_halo_l1": dict(
        state_covariance_weight=0.2128404007964649,
        control_covariance_weight=118.38122955698798,
    ),
    "nrho_l2_to_dro": dict(
        state_covariance_weight=0.15314443530196445,
        control_covariance_weight=129.24846631819943,
    ),
    "earth_mars": dict(
        state_covariance_weight=0.1530371475171174,
        control_covariance_weight=151.8285791060962,
    ),
}


def load_case(case_id: str) -> TestCase:
    """Instantiate a case of the deterministic module.

    Earth--Mars builds its fixed tabulated boundary states and
    two-body plot tracks itself, so no ephemeris kernels or Tudat are needed;
    Saved native runs supply their own reference states, controls and time grid.
    """
    return STOCHASTIC_CASE_REGISTRY[case_id]()


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
        raise ValueError(
            "The sigma factor associated with the Bryson's"
            "rule derived weights must be positive"
        )
    return np.eye(NP) / sigma_factor**2, np.eye(NU)


def state_covariance_node_scaling(
    seed_covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build fixed node-dependent diagonal scales from a state covariance seed."""

    state_covariances = np.asarray(seed_covariances, dtype=float)
    if (
        state_covariances.ndim != 3
        or state_covariances.shape[1] != state_covariances.shape[2]
    ):
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

    magnitude_variance = options.gates_proportional_magnitude_std**2
    pointing_variance = options.gates_proportional_pointing_std**2
    return pointing_variance * jnp.dot(control, control) * jnp.eye(control.shape[0]) + (
        magnitude_variance - pointing_variance
    ) * jnp.outer(control, control)


def unscented_weights(kappa: float, dimension: int) -> np.ndarray:
    """Unscented transform weights."""

    weights = np.full(2 * dimension + 1, 1.0 / (2.0 * (dimension + kappa)))
    weights[0] = kappa / (dimension + kappa)
    return weights


def split_over_devices(fn, batched):
    """Split the leading axis of a batch-mapped ``fn`` across all local devices.

    ``batched[i]`` says whether argument ``i`` carries the batch axis; the
    others are replicated. With one device (the GPU, or the CPU without
    JAX_NUM_CPU_DEVICES) ``fn`` is returned unchanged. Otherwise the batch is
    padded to a multiple of the device count with copies of its last entry,
    each device evaluates its block, and the padding is dropped.
    """
    devices = jax.devices()
    if len(devices) == 1:
        return fn
    count = len(devices)
    mesh = jax.sharding.Mesh(np.array(devices), ("batch",))
    split = jax.sharding.PartitionSpec("batch")
    # Blocks are independent, so the varying-axis check is off: Diffrax's
    # internal while loops start from replicated constants and fail it.
    mapped = jax.shard_map(
        fn,
        mesh=mesh,
        out_specs=split,
        check_vma=False,
        in_specs=tuple(split if b else jax.sharding.PartitionSpec() for b in batched),
    )

    def run(*args):
        args = tuple(jnp.asarray(a) for a in args)
        size = next(a.shape[0] for a, b in zip(args, batched) if b)
        pad = (-size) % count
        if pad:
            args = tuple(
                jnp.concatenate((a, jnp.repeat(a[-1:], pad, axis=0))) if b else a
                for a, b in zip(args, batched)
            )
        return jax.tree.map(lambda out: out[:size], mapped(*args))

    return run


@dataclass(frozen=True)
class Normalization:
    """Congruence scaling of the covariance by fixed deviations from the reference trajectory.

    D0 is the diagonal matrix of the initial standard deviations of the augmented state,
    whilst D0' is the diagonal matrix of the initial standard deviations of the primed state.
    The implementation carries P = D0^-1 Sigma D0^-1 rather than Sigma (where
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


def initial_std(case: TestCase, options: Options | None = None) -> np.ndarray:
    """Initial 1-sigma values of the augmented state [-]."""

    options = default_options(case.test_case_id) if options is None else options
    position_velocity_std = np.array(
        [options.initial_position_std_km / case.length_unit] * 3
        + [options.initial_velocity_std_km_s / case.velocity_unit] * 3,
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
    mass_std = options.initial_mass_std_kg / case.m0_wet
    return np.concatenate((position_velocity_std, [mass_std]))


def build_normalization(
    case: TestCase, options: Options | None = None
) -> Normalization:
    options = default_options(case.test_case_id) if options is None else options
    initial_sigma = initial_std(case, options)
    scale = initial_sigma.copy()
    scale[6] = options.mass_scale_kg / case.m0_wet
    return Normalization(scale=scale, initial_std=initial_sigma)


@dataclass
class ReferenceTraj:
    """Saved deterministic reference on the native solver mesh."""

    node_times: np.ndarray  # [-] non-dimensional, first entry zero
    steps: np.ndarray  # [-] arc durations
    states: np.ndarray  # [-] (NX, N + 1)
    controls: np.ndarray  # [-] (NU, N) thrust, zero-order hold
    fuel_consumed: float  # [kg]

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


def terminal_mean_state_target(
    case: TestCase,
    options: Options,
    ref_traj: ReferenceTraj,
) -> np.ndarray:
    """Return the complete-transfer or relative short-horizon mean target."""

    if options.truncated_uniform_mesh_arcs is None:
        return np.asarray(case.xf_state, dtype=float)
    return np.asarray(ref_traj.states[0:NP, -1], dtype=float)


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

    _, derivatives = dynamics._maps(substeps, batched_steps=True)
    state_sensitivity, control_sensitivity = derivatives(
        reference_states[:, :-1], reference_controls, steps
    )
    state_matrices = (
        normalization.inverse_matrix
        @ np.asarray(state_sensitivity)
        @ normalization.matrix
    )
    control_matrices = (
        normalization.inverse_matrix
        @ np.asarray(control_sensitivity)
        * case.max_thrust_nd
    )
    return list(state_matrices), list(control_matrices)


def tvlqr_gains(
    state_weight: float,
    control_weight: float,
    terminal_weight: float | np.ndarray,
    state_matrices: list[np.ndarray],
    control_matrices: list[np.ndarray],
    arc_weights=None,
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

    weights = (
        np.ones(len(state_matrices)) if arc_weights is None else np.asarray(arc_weights)
    )
    for k in reversed(range(len(state_matrices))):
        q, r = weights[k] * Q_normalized, weights[k] * R_normalized
        state_matrix = state_matrices[k][0:NP, 0:NP]
        control_matrix = control_matrices[k][0:NP, :]
        gain = np.linalg.solve(
            r + control_matrix.T @ P_normalized @ control_matrix,
            control_matrix.T @ P_normalized @ state_matrix,
        )
        gains[k] = -gain
        closed_loop_matrix = state_matrix - control_matrix @ gain
        P_normalized = (
            q
            + gain.T @ r @ gain
            + closed_loop_matrix.T @ P_normalized @ closed_loop_matrix
        )
        P_normalized = 0.5 * (P_normalized + P_normalized.T)

    return gains


@dataclass
class RobustSolution:
    means: np.ndarray
    feedforward: np.ndarray  # S_tilde, normalised
    gains: np.ndarray  # K_tilde, normalised
    radius: np.ndarray  # r_tilde
    covariances: np.ndarray  # P, normalised
    control_covariances: np.ndarray  # SigmaT, normalised
    objective: float
    # Fixed per-node diagonal covariance scales the Cholesky factors are expressed in.
    covariance_scale_variances: np.ndarray = field(
        default_factory=lambda: np.empty((0, NX), dtype=float)
    )
    # Every instance will have a different set of diagnostics.
    diagnostics: dict[str, float] = field(default_factory=dict)


class LowerTriangular:
    """Row-major packing, identical to the CasADi and double-integrator NLPs."""

    def __init__(self, dimension):
        self.dimension = dimension
        self.rows, self.columns = np.tril_indices(dimension)
        self.diagonal = np.flatnonzero(self.rows == self.columns)

    def __len__(self):
        return len(self.rows)

    def matrix(self, entries):
        return (
            jnp.zeros((self.dimension, self.dimension))
            .at[self.rows, self.columns]
            .set(entries)
        )

    def pack(self, matrix):
        return matrix[self.rows, self.columns]

    def covariance_pattern(self):
        """Exact polynomial support of d tril(L L.T) / d tril(L)."""
        i, j = self.rows[:, None], self.columns[:, None]
        r, c = self.rows[None, :], self.columns[None, :]
        return ((r == i) & (c <= j)) | ((r == j) & (c <= i))


LTRI, MTRI = LowerTriangular(NX), LowerTriangular(NP)


NL, NM = len(LTRI), len(MTRI)


def cholesky_lower(matrix, pivot_floor):
    """Match the original scalar Cholesky recursion, including pivot clamps."""
    factor = [[jnp.asarray(0.0) for _ in range(NX)] for _ in range(NX)]
    for row in range(NX):
        for column in range(row + 1):
            total = matrix[row, column]
            for inner in range(column):
                total = total - factor[row][inner] * factor[column][inner]
            factor[row][column] = (
                jnp.sqrt(jnp.maximum(total, pivot_floor))
                if row == column
                else total / factor[column][column]
            )
    return jnp.stack([jnp.stack(row) for row in factor])


def spectral_radius(matrix, floor, eigenvalue_smoothing=1e-6):
    """Smoothed analytic 3x3 largest eigenvalue."""
    q = jnp.trace(matrix) / 3
    p1 = matrix[0, 1] ** 2 + matrix[0, 2] ** 2 + matrix[1, 2] ** 2
    p2 = jnp.sum((jnp.diag(matrix) - q) ** 2) + 2 * p1
    smoothing_sq = 1e-30 if eigenvalue_smoothing == 1e-15 else eigenvalue_smoothing**2
    p = jnp.sqrt(p2 / 6 + smoothing_sq)
    b = (matrix - q * jnp.eye(NU)) / p
    det = (
        b[0, 0] * (b[1, 1] * b[2, 2] - b[1, 2] * b[2, 1])
        - b[0, 1] * (b[1, 0] * b[2, 2] - b[1, 2] * b[2, 0])
        + b[0, 2] * (b[1, 0] * b[2, 1] - b[1, 1] * b[2, 0])
    )
    r = jnp.clip(det / 2, -1 + 1e-12, 1 - 1e-12)
    return jnp.sqrt(q + 2 * p * jnp.cos(jnp.arccos(r) / 3) + floor**2)


def gates_inner_sigma_points(control, options):
    """Six-point kappa=0 cubature set for delta_c_G | control, and its weight.

    Returns ``(executed_controls, weight)`` with ``executed_controls`` of shape
    ``(6, NU)`` holding ``control + delta_c_G^(ell,+/-)`` and ``weight = 1/6``.
    The formal central point ``delta_c_G = 0`` carries zero weight at kappa=0
    under the execution-error model and is omitted, matching
    ``unscented_weights`` dropping no mass by doing so.
    """

    covariance = gates_execution_covariance(control, options)
    factor = jnp.linalg.cholesky(covariance + options.cholesky_jitter * jnp.eye(NU))
    offsets = jnp.sqrt(3.0) * factor
    executed = control[None, :] + jnp.concatenate((offsets.T, -offsets.T), axis=0)
    return executed, 1.0 / 6.0


def build_propagation_arc_map(case, options, normalization, integration_refinement=1):
    flow_smoothing = (
        options.control_norm_eps
        if options.mass_flow_smoothing is None
        else options.mass_flow_smoothing
    )
    if not np.isfinite(flow_smoothing) or flow_smoothing < 0:
        raise ValueError("mass_flow_smoothing must be finite and nonnegative")
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
        NW * sde_substeps if options.acceleration_diffusion_km_s32 > 0.0 else 0
    )
    if process_dimension != expected_process_dimension:
        raise ValueError("inconsistent process-noise augmentation dimension")
    gates_enabled = gates_execution_error_enabled(options)
    n_inner = 6 if gates_enabled else 1
    # Product weights W_i * V_j, flattened outer-major to match the (n_outer,
    # n_inner) -> (n_outer*n_inner,) reshape used for every nested array below.
    inner_weight = (1.0 / 6.0) if gates_enabled else 1.0
    weights = jnp.repeat(outer_weights, n_inner) * inner_weight

    step = stochastic_integration_map(
        case, options, integration_refinement=integration_refinement
    )
    batch_step = jax.vmap(step, in_axes=(0, 0, 0, 0, None))
    noise_root = jnp.diag(
        jnp.sqrt(
            jnp.clip(
                dimension
                * np.diag(navigation_covariance(case, options, normalization)),
                0.0,
                None,
            )
        )
    )

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
        executed, _ = jax.vmap(lambda c: gates_inner_sigma_points(c, options))(
            commanded
        )
        return executed.reshape(n_outer * n_inner, NU)

    # Points smoothed in the mass flow: all, or (mass_flow_smoothing_feedback_only) the outer points whose command
    # carries feedback: position/velocity state columns and navigation columns (the mass column of the lower
    # spread has no position/velocity rows; the process columns none), in the order center, +columns, -columns.
    if options.mass_flow_smoothing_feedback_only:
        feedback = np.array(
            [j < NP for j in range(NX)]
            + [True] * navigation_dimension
            + [False] * process_dimension
        )
        flow_mask = jnp.asarray(
            np.repeat(np.concatenate(([False], feedback, feedback)), n_inner).astype(
                float
            )
        )
    else:
        flow_mask = 1.0

    def arc(mean, covariance, feedforward, gain, duration):
        # Deliberately dimension*P, not (dimension+kappa)*P: preserve the source
        # even for non-default kappa, rather than silently correcting its UT.
        spread = cholesky_lower(
            dimension * covariance + options.cholesky_jitter * jnp.eye(NX),
            options.cholesky_jitter,
        )
        state_columns = jnp.concatenate(
            (spread, jnp.zeros((NX, state_padding))), axis=1
        )
        measurement_parts = [spread[:NP]]
        if navigation_dimension:
            measurement_parts.append(noise_root)
        if process_dimension:
            measurement_parts.append(jnp.zeros((NP, process_dimension)))
        measurement_columns = jnp.concatenate(measurement_parts, axis=1)
        state_offsets = (scale[:, None] * state_columns).T
        control_offsets = (gain @ measurement_columns).T
        states = jnp.concatenate(
            (mean[None], mean + state_offsets, mean - state_offsets)
        )
        # Commanded control per outer sigma point (normalised, i.e. c^(i)).
        commanded = jnp.concatenate(
            (
                feedforward[None],
                feedforward + control_offsets,
                feedforward - control_offsets,
            )
        )
        # Conditional Gates execution error: each outer point's own commanded
        # control generates its own six-point inner cubature set (conditional cubature
        # of the execution-error model). Disabled, this is the identity map.
        # An engine-off phase has no execution draw and no mass-flow norm to
        # differentiate. In particular sqrt(0**2 + (0*eps)**2) has an undefined
        # derivative even though its value is zero.
        executed = (
            jnp.zeros((n_outer * n_inner, NU))
            if case.max_thrust_nd == 0
            else nest_gates(commanded)
        )
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
            (
                jnp.zeros(n_outer * n_inner)
                if case.max_thrust_nd == 0
                else jnp.linalg.norm(controls, axis=1)
                if flow_smoothing == 0
                else jnp.sqrt(
                    jnp.sum(controls**2, axis=1)
                    + flow_mask * (case.max_thrust_nd * flow_smoothing) ** 2
                )
            ),
            brownian_increments_nested,
            duration,
        )
        mean_next = weights @ propagated
        deviations = (propagated - mean_next) / scale
        covariance_next = (weights[:, None] * deviations).T @ deviations
        # Executed-control covariance from the same nested points (conditional covariance
        # (27)), representing the actually executed rather than the commanded
        # control; feeds the thrust chance constraint and objective. Built
        # from `executed` (unit-ball, max-thrust-normalised), matching the
        # units of `feedforward`/`norm`, not from the absolute `controls` used
        # for dynamics propagation -- otherwise it would carry a spurious
        # extra factor of max_thrust_nd^2 relative to every other term the
        # chance constraint and objective combine it with.
        mean_executed = weights @ executed
        executed_deviations = executed - mean_executed
        control_covariance = (
            weights[:, None] * executed_deviations
        ).T @ executed_deviations
        return (
            mean_next,
            (covariance_next + covariance_next.T) / 2,
            (control_covariance + control_covariance.T) / 2,
        )

    return arc


def propagate_stochastic_moments(arc, means, feedforward, gains, covariance, steps):
    covariances, controls, propagated = [np.asarray(covariance)], [], [means[:, 0]]
    for k, duration in enumerate(steps):
        mu, p, t = arc(
            means[:, k], covariances[-1], feedforward[:, k], gains[k], duration
        )
        propagated.append(np.asarray(mu))
        covariances.append(np.asarray(p))
        controls.append(np.asarray(t))
    return np.stack(propagated, axis=1), np.stack(covariances), np.stack(controls)
