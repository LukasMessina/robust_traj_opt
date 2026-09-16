"""
Chance-constrained state_covariance steering for the linear double-integrator system.

* The transcription is multiple shooting on both the mean and the cholesky factor
of the node state_covariance. Only the computational stack changes:

"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# Replace 'cpu' with 'cuda' if GPU capability is requested,
# while avoiding unnecessary GPU memory allocation for this small problem.
os.environ.setdefault("JAX_PLATFORMS", "cuda")

# If switching JAX_PLATFORMS to "cuda", these control how much GPU memory jax
# grabs up front. Uncomment and adjust as needed -- all three must be set before
# `import jax` to take effect.
# os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")   # no upfront grab; allocate on demand
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")    # or: preallocate exactly this fraction (0-1) instead of ~0.75
# os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")  # or: most conservative allocator, grows as needed, no big block

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, qmc
from scipy.integrate import quad_vec
from scipy.linalg import expm

from diffrax import (
    AbstractPath,
    ControlTerm,
    ODETerm,
    MultiTerm,
    SaveAt,
    ConstantStepSize,
    SpaceTimeLevyArea,
    diffeqsolve,
    ShARK,
    Tsit5,
)

from pyoptsparse import Optimization
from pyoptsparse.pySNOPT.pySNOPT import SNOPT
from plotter import Plotter


plt.rcParams.update(
    {
        "text.usetex": False,
        "font.serif": ["cmr10"],
        "axes.formatter.use_mathtext": True,
    }
)


NX = 4  # state: [x1, x2, x3, x4] = [px, py, vx, vy]
NU = 2  # control: [ux, uy]

OUTPUT_DIR = Path("output/double_integrator_stochastic_traj_opt")


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
    control_confidence: float = 0.9973  # probability of control constraint satisfaction (p_u)

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


def unscented_weights(kappa: float, dimension: int) -> np.ndarray:
    weights = np.full(2 * dimension + 1, 1.0 / (2.0 * (dimension + kappa)))
    weights[0] = kappa / (dimension + kappa)
    return weights


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


def discrete_time_matrices(options: Options) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = options.dt
    A = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    B = np.array(
        [
            [0.5 * dt**2, 0.0],
            [0.0, 0.5 * dt**2],
            [dt, 0.0],
            [0.0, dt],
        ]
    )
    G = 0.01 * np.eye(NX)
    return A, B, G


def continuous_time_matrices() -> tuple[np.ndarray, np.ndarray]:
    Ac = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    Bc = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    return Ac, Bc


def accumulated_process_covariance(
    Ac: np.ndarray, Qc: np.ndarray, duration: float
) -> np.ndarray:
    """Compute the accumulated continuous-time process noise
    state_covariance by numerical quadrature to verify equivalence with the discrete noise model."""

    state_covariance, _ = quad_vec(
        lambda s: expm(Ac * s) @ Qc @ expm(Ac.T * s),
        0.0,
        duration,
        epsabs=1e-14,
        epsrel=1e-14,
    )
    return 0.5 * (state_covariance + state_covariance.T)


def calibrate_continuous_diffusion(
    Ac: np.ndarray, discrete_covariance: np.ndarray, duration: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover continuous time process noise state_covariance matrix Qc and its
    cholesky factor from a desired arc state_covariance.

    The finite-horizon continuous Lyapunov operator is formed column-by-column,
    then inverted. This deliberately calibrates the physical continuous SDE.
    """

    dimension = Ac.shape[0]
    lyapunov_operator = np.empty((dimension**2, dimension**2))
    for column in range(dimension**2):
        basis = np.zeros((dimension, dimension))
        basis.flat[column] = 1.0
        image, _ = quad_vec(
            lambda s: expm(Ac * s) @ basis @ expm(Ac.T * s),
            0.0,
            duration,
            epsabs=1e-14,
            epsrel=1e-14,
        )
        lyapunov_operator[:, column] = image.reshape(-1)
    Qc = np.linalg.solve(lyapunov_operator, discrete_covariance.reshape(-1)).reshape(
        dimension, dimension
    )
    Qc = 0.5 * (Qc + Qc.T)
    return Qc, np.linalg.cholesky(Qc)


# --------------------------------------------------------------------------- #
#             Propagation of the linear, zero-order-held dynamics
# --------------------------------------------------------------------------- #


def _drift_vector_field(t, state, control):
    Ac, Bc = continuous_time_matrices()
    return jnp.asarray(Ac) @ state + jnp.asarray(Bc) @ control


_TERM = ODETerm(lambda t, y, args: _drift_vector_field(t, y, args))
_SOLVER = Tsit5()


def diffrax_step(state: jnp.ndarray, control: jnp.ndarray, step: float) -> jnp.ndarray:
    """Propagate ``state`` one arc of duration ``step`` under zero-order-held ``control``.

    A single fixed Tsit5 step (``ConstantStepSize``) is exact for this linear,
    ZOH system regardless of solver order, so no substepping or adaptive
    tolerance is needed -- this differs from the reference repo's use of
    diffrax only in that the nonlinear CR3BP/2BP dynamics there require
    genuine adaptive integration, while here diffrax is used for the same
    linear dynamics the CasADi version integrated in closed form.
    """

    solution = diffeqsolve(
        _TERM,
        _SOLVER,
        t0=0.0,
        t1=step,
        dt0=step,
        y0=state,
        args=control,
        stepsize_controller=ConstantStepSize(),
        saveat=SaveAt(t1=True),
        max_steps=4,
    )
    return solution.ys[0]


_diffrax_step_batch = jax.vmap(diffrax_step, in_axes=(1, 1, None), out_axes=1)


def _diffusion_vector_field(t, state, Lc):
    del t, state
    return Lc


_SDE_SOLVER = ShARK()


class PrescribedSpaceTimeLevyPath(AbstractPath):
    """A deterministic one-arc Brownian driver carrying prescribed ``W`` and ``H``.

    ShARK requests the control over the complete integration step with
    ``use_levy=True``. The path therefore exposes both the Brownian increment
    ``W`` and Diffrax's normalized space-time Levy area ``H``. The linear
    interpolation below only defines the ordinary path value outside that
    full-step request; no Brownian subincrements are reconstructed from the
    two supplied random variables.
    """

    brownian_increment: jnp.ndarray
    space_time_levy_area: jnp.ndarray
    duration: float

    @property
    def t0(self):
        return 0.0

    @property
    def t1(self):
        return self.duration

    def evaluate(self, t0, t1=None, left=True, use_levy=False):
        del left
        if t1 is None:
            t1 = t0
            t0 = 0.0

        interval = t1 - t0
        fraction = interval / self.duration
        increment = fraction * self.brownian_increment
        if use_levy:
            return SpaceTimeLevyArea(
                dt=interval,
                W=increment,
                H=fraction * self.space_time_levy_area,
            )
        return increment


def prescribed_integration_sde_step(
    state: jnp.ndarray,
    control: jnp.ndarray,
    brownian_increment: jnp.ndarray,
    space_time_levy_area: jnp.ndarray,
    Lc: jnp.ndarray,
    step: float,
) -> jnp.ndarray:
    """Take one ShARK step using exactly the supplied ``W`` and ``H``.

    The prescribed path contains no random key, so the integration cannot
    introduce another Brownian realization during optimization or automatic
    differentiation. For one interval of length ``step``, callers supply
    independent standard-normal coordinates ``xi`` and ``eta`` through
    ``W = sqrt(step) xi`` and ``H = sqrt(step / 12) eta``.
    """

    prescribed_path = PrescribedSpaceTimeLevyPath(
        brownian_increment=brownian_increment,
        space_time_levy_area=space_time_levy_area,
        duration=step,
    )
    terms = MultiTerm(
        ODETerm(lambda t, y, args: _drift_vector_field(t, y, args[0])),
        ControlTerm(
            lambda t, y, args: _diffusion_vector_field(t, y, args[1]), prescribed_path
        ),
    )
    solution = diffeqsolve(
        terms,
        _SDE_SOLVER,
        t0=0.0,
        t1=step,
        dt0=step,
        y0=state,
        args=(control, Lc),
        stepsize_controller=ConstantStepSize(),
        saveat=SaveAt(t1=True),
        max_steps=1,
    )
    return solution.ys[0]


_prescribed_sde_integration_step_batch = jax.vmap(
    prescribed_integration_sde_step,
    in_axes=(1, 1, 1, 1, None, None),
    out_axes=1,
)


def validate_shark_process_covariance(
    Ac: np.ndarray,
    Qc: np.ndarray,
    Lc: np.ndarray,
    duration: float,
) -> float:
    """Compare one ShARK step with the exact linear-SDE process covariance.

    A symmetric sigma rule integrates the covariance exactly because this
    double-integrator's one-step stochastic map is affine in the independent
    Gaussian variables defining ``W`` and ``H``.
    """

    noise_dimension = Lc.shape[1]
    stochastic_dimension = 2 * noise_dimension
    stochastic_coordinates = np.concatenate(
        [
            np.zeros((stochastic_dimension, 1)),
            np.sqrt(stochastic_dimension) * np.eye(stochastic_dimension),
            -np.sqrt(stochastic_dimension) * np.eye(stochastic_dimension),
        ],
        axis=1,
    )
    weights = unscented_weights(0.0, stochastic_dimension)
    brownian_increments = (
        np.sqrt(duration) * stochastic_coordinates[:noise_dimension]
    )
    space_time_levy_areas = (
        np.sqrt(duration / 12.0) * stochastic_coordinates[noise_dimension:]
    )
    n_sigma = stochastic_coordinates.shape[1]
    propagated = np.asarray(
        _prescribed_sde_integration_step_batch(
            jnp.zeros((NX, n_sigma)),
            jnp.zeros((NU, n_sigma)),
            jnp.asarray(brownian_increments),
            jnp.asarray(space_time_levy_areas),
            jnp.asarray(Lc),
            duration,
        )
    )
    propagated_mean = propagated @ weights
    residuals = propagated - propagated_mean[:, None]
    numerical_covariance = (residuals * weights[None, :]) @ residuals.T
    numerical_covariance = 0.5 * (numerical_covariance + numerical_covariance.T)
    exact_covariance = accumulated_process_covariance(Ac, Qc, duration)
    return float(np.max(np.abs(numerical_covariance - exact_covariance)))


# --------------------------------------------------------------------------- #
#           Smoothed spectral radius of the control covariance
# --------------------------------------------------------------------------- #


def numerical_psqrt_spectral_radius(
    matrix: np.ndarray, eigenvalue_smoothing: float,
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
    matrix: jnp.ndarray, eigenvalue_smoothing: float,
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
        "relative_psqrt_spectral_radius_smoothing_bias": float(np.max(bias) / peak_exact_psqrt_spectral_radius),
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

    weights = jnp.asarray(
        unscented_weights(options.scaling_parameter, total_dimension)
    )
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

        brownian_increments = (
            jnp.sqrt(dt) * stochastic_matrix_pts[:noise_dimension]
        )
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


def evaluate_node_matching_defects(arc_fn, means, feedforward, gains, state_covariances):
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
        state_covariance_error = np.asarray(state_covariance_next) - state_covariances[k + 1]
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
        raise ValueError("initial state covariance diagonal must be positive for Bryson weights")
    if np.any(terminal_variances <= 0.0):
        raise ValueError("final covariance diagonal must be positive for Bryson weights")
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
        "optimality conditions satisfied" in status_text.lower() and "infeasible" not in status_text.lower()
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
                state_indices, state_indices * (n_arcs + 1) + node,
                np.ones(NX), (NX, mean_size),
            )
        }
        for name, node in (("boundary_start", 0), ("boundary_end", n_arcs))
    }
    if stochastic:
        n_tril = len(LTRI)
        positions = np.asarray(LTRI.diagonal_positions)
        jacobian["margin_diagonal"] = {
            "terminal_margin": build_coo_block(state_indices, positions, np.ones(NX), (NX, n_tril))
        }
        arc_count  = NX * (n_arcs + 1)
        columns = (np.arange(n_arcs + 1)[:, None] * n_tril + positions).reshape(-1)
        jacobian["cholesky_diagonal"] = {
            "cholesky_factor": build_coo_block(
                np.arange(arc_count), columns, np.ones(arc_count),
                (arc_count, (n_arcs + 1) * n_tril),
            )
        }
    else:
        path = np.zeros((len(options.path_constraints) * n_arcs, mean_size))
        for path_index, (normal, _) in enumerate(options.path_constraints):
            for node in range(n_arcs):
                path[path_index * n_arcs + node, state_indices * (n_arcs + 1) + node] = normal
        rows, columns = np.nonzero(path)
        jacobian["path"] = {"means": build_coo_block(rows, columns, path[rows, columns], path.shape)}
    return jacobian


def _nominal_propagation_step(state: jnp.ndarray, control: jnp.ndarray, dt: float) -> jnp.ndarray:
    return diffrax_step(state, control, dt)


def _nominal_funcs(xdict: dict, options: Options):
    means = xdict["means"].reshape(NX, options.n_arcs + 1)
    feedforward = xdict["feedforward"].reshape(NU, options.n_arcs)

    propagation_step_batch = jax.vmap(_nominal_propagation_step, in_axes=(1, 1, None), out_axes=1)
    predicted_next = propagation_step_batch(means[:, :-1], feedforward, options.dt)
    defects = (means[:, 1:] - predicted_next).reshape(-1)

    control_energy = options.dt * jnp.sum(feedforward * feedforward)

    funcs = {
        "defects": defects,
        "objective": control_energy,
    }
    return funcs


def _nonlinear_arc_outputs_nominal(local_input: jnp.ndarray, options: Options) -> jnp.ndarray:
    """Nonlinear (Diffrax-propagated) objective and defect quantities owned by one arc."""

    mean = local_input[:NX]
    feedforward = local_input[NX:]
    predicted_mean = _nominal_propagation_step(mean, feedforward, options.dt)
    local_objective = options.dt * jnp.dot(feedforward, feedforward)
    return jnp.concatenate(
        [predicted_mean, jnp.atleast_1d(local_objective)]
    )


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
        jacobian["defects"]["feedforward"][np.ix_(defect_rows, feedforward_columns)] = (
            -local[:NX, slice(NX, NX + NU)]
        )

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


def solve_deterministic_optimization(
    options: Options, A: np.ndarray, B: np.ndarray
) -> NominalTrajectory:
    """Deterministic minimum-energy transfer via multiple shooting on the state vectors.

    Each arc is propagated with a batched fixed-step Diffrax solve.
    """

    n_arcs = options.n_arcs

    means_seed = np.linspace(options.x0_mean, options.xf_mean, n_arcs + 1).T
    feedforward_seed = np.zeros((NU, n_arcs))
    for k in range(n_arcs):
        feedforward_seed[:, k] = 1e-1 

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
        np.broadcast_to(nonlinear_local_structure, (n_arcs,) + nonlinear_local_structure.shape), options
    )
    nonlinear_jac_sparsity = get_jacobian_sparsity(nonlinear_global_jacobian_structure)
    linear_jac = _linear_jacobians(options, stochastic=False)

    # Compile both callbacks before entering the optimization
    jax.block_until_ready(nonlinear_residual_fn(nominal_seed))
    jax.block_until_ready(nonlinear_jac_fn(nominal_seed))

    def objconfun(xdict):
        funcs = nonlinear_residual_fn(xdict)
        return {key: np.asarray(value) for key, value in funcs.items()}, False

    def sens(xdict, funcs):
        nonlinear_jacobian_blocks = nonlinear_jac_fn(xdict)
        jacobian = _assemble_nonlinear_jacobian_nominal(
            np.asarray(nonlinear_jacobian_blocks), options
        )
        return apply_jacobian_sparsity(jacobian, nonlinear_jac_sparsity), False

    optProb = Optimization("nominal", objconfun)
    optProb.addVarGroup("means", NX * (n_arcs + 1), value=means_seed.reshape(-1))
    optProb.addVarGroup("feedforward", NU * n_arcs, value=feedforward_seed.reshape(-1))

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
    solved_feedforward = np.asarray(sol.xStar["feedforward"]).reshape(NU, n_arcs)
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
    scaled_terminal_state_covariance = np.outer(terminal_scaling, terminal_scaling) * seed_covariances[-1]
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


def _terminal_constraint_slack(terminal_state_covariance: jnp.ndarray, options: Options) -> jnp.ndarray:
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
    xdict: dict, options: Options, arc_fn, psi_inv_u: float, phi_inv_path: float
):
    """Evaluate every independent shooting arc in one batched JAX program.

    Each multiple-shooting arc starts from its own
    decision-variable mean and state_covariance. Through `vmap` all arcs, and all sigma points inside each arc, are
    exposed to XLA as nested batch dimensions in one compiled callback.
    """

    n_arcs = options.n_arcs
    n_path = len(options.path_constraints)
    means, feedforward, gains, cholesky_factor_entries, margin_entries = _unpack_stochastic_vars(
        xdict, options
    )

    node_state_covariances = jax.vmap(
        lambda entries: LTRI.to_matrix(entries) @ LTRI.to_matrix(entries).T
    )(cholesky_factor_entries)

    Q = jnp.asarray(options.Q)
    R = jnp.asarray(options.R)
    path_vectors = options.path_matrix  # (n_path, NX)

    batched_propagation_arc = jax.vmap(arc_fn, in_axes=(1, 0, 1, 0), out_axes=(1, 0, 0))
    predicted_means, predicted_covariances, control_covariances = batched_propagation_arc(
        means[:, :-1], node_state_covariances[:-1], feedforward, gains
    )

    mean_defects = (means[:, 1:] - predicted_means).reshape(-1)
    covariance_errors = node_state_covariances[1:] - predicted_covariances
    covariance_defects = jax.vmap(LTRI.pack)(covariance_errors).reshape(-1)

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
        "initial_covariance": LTRI.pack(
            node_state_covariances[0] - jnp.asarray(options.x0_covariance)
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


def _packed_state_covariance(cholesky_factor_entries: jnp.ndarray) -> jnp.ndarray:
    factor = LTRI.to_matrix(cholesky_factor_entries)
    return LTRI.pack(factor @ factor.T)


def _nonlinear_arc_outputs_stochastic(
    local_input: jnp.ndarray,
    options: Options,
    arc_fn,
    psi_inv_u: float,
    phi_inv_path: float,
) -> jnp.ndarray:
    """Nonlinear objective and defect quantities owned by one arc."""

    n_tril = len(LTRI)
    mean = local_input[:NX]
    feedforward = local_input[NX : NX + NU]
    gain = local_input[NX + NU : NX + NU + NU * NX].reshape(NU, NX)
    cholesky_factor_entries = local_input[-n_tril:]
    cholesky_factor = LTRI.to_matrix(cholesky_factor_entries)
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
    path_variances = jnp.einsum("pi,ij,pj->p", path_vectors, state_covariance, path_vectors)
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
            LTRI.pack(covariance_next),
            jnp.atleast_1d(control_chance),
            path_values,
            jnp.atleast_1d(local_objective),
        ]
    )


def get_terminal_residual(
    terminal_input: jnp.ndarray, options: Options
) -> jnp.ndarray:
    n_tril = len(LTRI)
    cholesky_factor = LTRI.to_matrix(terminal_input[:n_tril])
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
):
    """AD-differentiate small nonlinear arc maps, then batch them over the trajectory."""

    means, feedforward, gains, cholesky_factor_entries, margin_entries = _unpack_stochastic_vars(
        xdict, options
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
        lambda values: _nonlinear_arc_outputs_stochastic(
            values, options, arc_fn, psi_inv_u, phi_inv_path
        )
    )
    nonlinear_jacobian_blocks = jax.vmap(local_jacobian)(local_inputs)
    state_covariance_jacobians = jax.vmap(jax.jacrev(_packed_state_covariance))(cholesky_factor_entries)
    terminal_input = jnp.concatenate([cholesky_factor_entries[-1], margin_entries])
    terminal_jacobian = jax.jacrev(
        lambda values: get_terminal_residual(values, options)
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
        "control_chance": allocate_blocks(n_arcs, ("means", "feedforward", "gains", "cholesky_factor")),
        "terminal_residual": allocate_blocks(n_tril, ("cholesky_factor", "terminal_margin")),
        "objective": {
            variable: np.zeros(size) for variable, size in variable_sizes.items()
        },
    }
    for path_index in range(n_path):
        jacobian[f"path_{path_index}"] = allocate_blocks(n_arcs, ("means", "cholesky_factor"))

    state_indices = np.arange(NX)
    control_indices = np.arange(NU)
    gain_indices = np.arange(NU * NX)
    tril_indices = np.arange(n_tril)
    jacobian["initial_covariance"]["cholesky_factor"][:, :n_tril] = state_covariance_jacobians[0]

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
        jacobian["mean_defects"]["means"][
            np.ix_(mean_rows, next_mean_columns)
        ] += np.eye(NX)
        jacobian["mean_defects"]["feedforward"][
            np.ix_(mean_rows, feedforward_columns)
        ] = -local_jac[output_mean, input_feedforward]
        jacobian["mean_defects"]["gains"][np.ix_(mean_rows, gain_columns)] = -local_jac[
            output_mean, input_gain
        ]
        jacobian["mean_defects"]["cholesky_factor"][np.ix_(mean_rows, cholesky_factor_columns)] = (
            -local_jac[output_mean, input_cholesky]
        )

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
            jacobian["control_chance"][variable][np.ix_(control_row, columns)] = local_jac[
                output_control_chance, input_slice
            ][None, :]

        for path_index in range(n_path):
            path_name = f"path_{path_index}"
            output_index = output_path_start + path_index
            jacobian[path_name]["means"][arc_index, mean_columns] = local_jac[
                output_index, input_mean
            ]
            jacobian[path_name]["cholesky_factor"][arc_index, cholesky_factor_columns] = local_jac[
                output_index, input_cholesky
            ]

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
    jacobian["terminal_residual"]["terminal_margin"][:, :] = terminal_jacobian[:, n_tril:]
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

    cholesky_factor_seed = np.stack(
        [
            LTRI.pack_numeric(np.linalg.cholesky(seed.state_covariances[k]))
            for k in range(n_arcs + 1)
        ]
    )

    nonlinear_residual_fn = jax.jit(
        lambda xdict: _stochastic_funcs(xdict, options, arc_fn, psi_inv_u, phi_inv_path)
    )
    nonlinear_jac_fn = jax.jit(
        lambda xdict: _nonlinear_jacobian_stochastic(
            xdict, options, arc_fn, psi_inv_u, phi_inv_path
        )
    )
    n_local_inputs = NX + NU + NU * NX + n_tril
    n_local_outputs = NX + n_tril + 1 + len(options.path_constraints) + 1
    nonlinear_global_jacobian_structure = _assemble_nonlinear_global_jacobian_stochastic(
        np.ones((n_arcs, n_local_outputs, n_local_inputs)),
        np.ones((n_arcs + 1, n_tril, n_tril)),
        np.ones((n_tril, 2 * n_tril)), 
        options,
    )
    nonlinear_jac_sparsity = get_jacobian_sparsity(nonlinear_global_jacobian_structure)
    linear_jac = _linear_jacobians(options, stochastic=True)

    def objconfun(xdict):
        funcs = nonlinear_residual_fn(xdict)
        return {key: np.asarray(value) for key, value in funcs.items()}, False

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
        return apply_jacobian_sparsity(jacobian, nonlinear_jac_sparsity), False

    optProb = Optimization("stochastic", objconfun)
    optProb.addVarGroup("means", NX * (n_arcs + 1), value=seed.means.reshape(-1))
    optProb.addVarGroup("feedforward", NU * n_arcs, value=seed.feedforward.reshape(-1))
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

    solved_means = np.asarray(sol.xStar["means"]).reshape(NX, n_arcs + 1)
    solved_feedforward = np.asarray(sol.xStar["feedforward"]).reshape(NU, n_arcs)
    solved_gains = np.asarray(sol.xStar["gains"]).reshape(n_arcs, NU, NX)
    solved_cholesky = np.asarray(sol.xStar["cholesky_factor"]).reshape(n_arcs + 1, n_tril)
    solved_state_covariances = np.stack(
        [
            LTRI.to_matrix_numeric(solved_cholesky[k])
            @ LTRI.to_matrix_numeric(solved_cholesky[k]).T
            for k in range(n_arcs + 1)
        ]
    )

    control_covariances, mean_defect, state_covariance_defect = evaluate_node_matching_defects(
        arc_fn, solved_means, solved_feedforward, solved_gains, solved_state_covariances
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
            "matching_defect": mean_defect,
            "covariance_matching_defect": state_covariance_defect,
            **smoothing_metrics,
        },
    )


# --------------------------------------------------------------------------- #
#                                Verification
# --------------------------------------------------------------------------- #


def open_loop_state_covariances(
    options: Options,
    Lc: np.ndarray,
    solution: RobustTrajectory,
) -> np.ndarray:
    """Estimate node covariances from feedforward-only SDE rollouts."""

    return monte_carlo_rollout(options, Lc, solution, closed_loop=False)[
        "state_covariances"
    ]


def monte_carlo_rollout(
    options: Options,
    Lc: np.ndarray,
    solution: RobustTrajectory,
    *,
    closed_loop: bool = True,
) -> dict[str, np.ndarray]:
    """Roll out trajectories using independent scrambled Sobol input sets.

    Initial states and stochastic paths use separate random streams. Random
    pairing avoids alignment between the two Sobol sets. Each Brownian point
    supplies independent standard-normal coordinates for both ``W`` and ``H``
    on every arc of one path; trajectories are not IID samples.
    With ``closed_loop=False``, only the feedforward control is applied, subject
    to the same control limit. Both modes use identical samples for a given seed.
    """

    samples = options.monte_carlo_samples
    if samples < 2 or samples & (samples - 1):
        raise ValueError("Monte carlo samples must be a power of two and at least 2.")

    noise_dimension = Lc.shape[1]
    # Starting from one reproducible master seed, spawn three independent streams
    initial_seed, brownian_seed, pairing_seed = np.random.SeedSequence(
        options.monte_carlo_seed
    ).spawn(3)
    initial_state_sampler = qmc.Sobol(
        d=NX,
        scramble=True,
        seed=np.random.default_rng(initial_seed),
    )
    brownian_path_sampler = qmc.Sobol(
        d=options.n_arcs * 2 * noise_dimension,
        scramble=True,
        seed=np.random.default_rng(brownian_seed),
    )
    exponent = int(samples).bit_length() - 1
    initial_state_uniforms = initial_state_sampler.random_base2(m=exponent)
    brownian_path_uniforms = brownian_path_sampler.random_base2(m=exponent)
    # Keep the inverse normal CDF finite, including at a possible zero endpoint.
    eps = np.finfo(float).eps
    initial_state_normals = norm.ppf(np.clip(initial_state_uniforms, eps, 1.0 - eps)).T
    brownian_path_normals = norm.ppf(np.clip(brownian_path_uniforms, eps, 1.0 - eps)).reshape(
        samples, options.n_arcs, 2 * noise_dimension
    )
    # Permute whole Brownian paths, preserving their internal Sobol coordinates.
    pairing = np.random.default_rng(pairing_seed).permutation(samples)
    brownian_path_normals = brownian_path_normals[pairing]

    L0 = np.linalg.cholesky(options.x0_covariance)
    states = solution.means[:, 0:1] + L0 @ initial_state_normals
    trajectories = np.empty((options.n_arcs + 1, NX, samples))
    trajectories[0] = states

    for k in range(options.n_arcs):
        control = np.broadcast_to(
            solution.feedforward[:, k : k + 1], (NU, samples)
        )
        if closed_loop:
            deviation = states - solution.means[:, k : k + 1]
            control = control + solution.gains[k] @ deviation
        magnitude = np.linalg.norm(control, axis=0)
        saturated = magnitude > options.u_max
        scale = np.ones_like(magnitude)
        scale[saturated] = options.u_max / magnitude[saturated]
        applied_control = control * scale[None, :]
        arc_normals = brownian_path_normals[:, k, :].T
        increments = np.sqrt(options.dt) * arc_normals[:noise_dimension]
        space_time_levy_areas = (
            np.sqrt(options.dt / 12.0) * arc_normals[noise_dimension:]
        )
        states = np.asarray(
            _prescribed_sde_integration_step_batch(
                jnp.asarray(states),
                jnp.asarray(applied_control),
                jnp.asarray(increments),
                jnp.asarray(space_time_levy_areas),
                jnp.asarray(Lc),
                options.dt,
            )
        )
        trajectories[k + 1] = states

    state_deviations = trajectories - solution.means.T[:, :, None]
    # np.cov centers the sampled deviations and uses the N - 1 normalization.
    sampled_state_covariances = np.stack(
        [np.cov(deviations) for deviations in state_deviations]
    )
    return {
        "trajectories": trajectories,
        "state_covariances": sampled_state_covariances,
        "terminal_state_covariance": sampled_state_covariances[-1],
    }


# --------------------------------------------------------------------------- #
#                               Plotting 
# --------------------------------------------------------------------------- #


def covariance_ellipse_points(
    covariance: np.ndarray, n_sigma: float, n_points: int = 90
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(
        0.5 * (covariance + covariance.T)
    )
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    angle = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    return n_sigma * (eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ circle)


def plot_trajectory(
    options: Options,
    means: np.ndarray,
    state_covariances: np.ndarray,
    monte_carlo: dict | None,
    output_path: Path,
    *,
    feedforward: np.ndarray | None = None,
) -> None:
    plotter = Plotter(output_path.with_suffix(""))
    figure, axis = plt.subplots(
        figsize=Plotter.SQUARE_DIAGNOSTIC_FIGSIZE, dpi=Plotter.FIGURE_DPI
    )

    x1 = np.linspace(-1.0, 12.0, 200)
    y_line1 = options.b1 - options.a1[0] * x1
    y_line2 = (options.b2 - options.a2[0] * x1) / options.a2[1]
    axis.fill_between(x1, np.minimum(y_line1, y_line2), 8.0, color=Plotter.LIGHT_GREY, zorder=0)
    axis.plot(x1, y_line1, "k--", lw=0.8)
    axis.plot(x1, y_line2, "k--", lw=0.8)

    if monte_carlo is not None:
        trajectories = monte_carlo["trajectories"]
        axis.scatter(
            trajectories[-1, 0, :],
            trajectories[-1, 1, :],
            s=1.0,
            color=Plotter.GREY,
            alpha=0.15,
            linewidths=0.0,
            rasterized=True,
            label="MC",
            zorder=2,
        )

    axis.plot(
        means[0], means[1], "k-+", lw=Plotter.TRAJECTORY_LINE_WIDTH, markersize=4,
        label=r"$\bar{x}(t_k)$", zorder=3,
    )

    for k in range(1, means.shape[1] - 1):
        points = covariance_ellipse_points(state_covariances[k][0:2, 0:2], 3.0)
        axis.plot(
            means[0, k] + points[0],
            means[1, k] + points[1],
            color=Plotter.DARK_GREY,
            lw=0.6,
            label=r"$\Sigma_{x(t_k)}$" if k == 1 else None,
            zorder=2,
        )

    start_points = covariance_ellipse_points(state_covariances[0, :2, :2], 3.0)
    axis.plot(
        means[0, 0] + start_points[0],
        means[1, 0] + start_points[1],
        color=Plotter.DEPARTURE_COLOR,
        lw=0.8,
        label=r"$\Sigma_{x(t_0)}$",
        zorder=4,
    )

    terminal_points = covariance_ellipse_points(state_covariances[-1, :2, :2], 3.0)
    axis.plot(
        means[0, -1] + terminal_points[0],
        means[1, -1] + terminal_points[1],
        color=Plotter.ENDPOINT_COLOR,
        lw=0.8,
        label=r"$\Sigma_{x(t_N)}$",
        zorder=4,
    )

    target_points = covariance_ellipse_points(options.xf_covariance[0:2, 0:2], 3.0)
    axis.plot(
        options.xf_mean[0] + target_points[0],
        options.xf_mean[1] + target_points[1],
        color=Plotter.TARGET_COLOR,
        linestyle="--",
        lw=0.8,
        label=r"$\Sigma_{x,\mathrm{target}}$",
        zorder=5,
    )

    if feedforward is not None:
        # One acceleration arrow at each arc's start; a common scale preserves
        # relative magnitudes.
        axis.quiver(
            means[0, :-1], means[1, :-1], feedforward[0], feedforward[1],
            angles="xy", scale_units="xy", scale=2.0,
            color=Plotter.DARK_BLUE, width=0.004, pivot="tail",
            label="_nolegend_", zorder=6,
        )
        # Use a thin line in the legend instead of Quiver's filled swatch.
        axis.plot([], [], color=Plotter.DARK_BLUE, lw=0.8, label=r"$\bar{u}(t_k)$")

    position_radii = 3.0 * np.sqrt(
        np.maximum(np.diagonal(state_covariances[:, :2, :2], axis1=1, axis2=2), 0.0)
    )
    lower_bounds = np.floor(np.min(means[:2].T - position_radii, axis=0))
    upper_bounds = np.ceil(np.max(means[:2].T + position_radii, axis=0))
    axis.set_xlim(min(0.0, lower_bounds[0]), max(11.0, upper_bounds[0]))
    axis.set_ylim(min(0.0, lower_bounds[1]), max(8.0, upper_bounds[1]))
    axis.set_xlabel(r"$x_1$")
    axis.set_ylabel(r"$x_2$")
    axis.set_box_aspect(1.0)
    axis.set_axisbelow(True)
    plotter._style_2d_axis(axis)
    plotter._legend(
        axis, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=3,
        frameon=True, fancybox=False, edgecolor=Plotter.BLACK,
        facecolor="white", framealpha=1.0, borderpad=0.30,
        columnspacing=0.85, handlelength=1.5,
    )
    figure.subplots_adjust(left=0.16, right=0.95, bottom=0.12, top=0.82)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=Plotter.FIGURE_DPI)
    plt.close(figure)


def plot_control_profiles(
    options: Options,
    solution: RobustTrajectory,
    magnitude_path: Path,
    covariance_path: Path,
) -> None:
    """Plot the feedforward norm and the 3-sigma control ellipses about each feedforward vector.
    """

    plotter = Plotter(magnitude_path.with_suffix(""))
    magnitude_figure, magnitude_axis = plt.subplots(
        figsize=Plotter.SQUARE_DIAGNOSTIC_FIGSIZE, dpi=Plotter.FIGURE_DPI
    )
    covariance_figure, covariance_axis = plt.subplots(
        figsize=Plotter.SQUARE_DIAGNOSTIC_FIGSIZE, dpi=Plotter.FIGURE_DPI
    )

    time_edges = options.dt * np.arange(options.n_arcs + 1)
    control_norms = np.linalg.norm(solution.feedforward, axis=0)
    magnitude_axis.stairs(
        control_norms, time_edges, baseline=None, color=Plotter.BLACK,
        lw=Plotter.TRAJECTORY_LINE_WIDTH,
        label=r"$\|\bar{u}(t_k)\|$",
    )
    magnitude_axis.axhline(
        options.u_max, color=Plotter.RED, linestyle="--", lw=Plotter.GUIDE_LINE_WIDTH,
        label=r"$u_{\max}$",
    )
    magnitude_axis.set_xlim(time_edges[0], time_edges[-1])
    magnitude_axis.set_ylim(
        -0.05 * options.u_max, 1.1 * max(options.u_max, np.max(control_norms))
    )
    magnitude_axis.set_xlabel("time [s]")
    magnitude_axis.set_ylabel(r"$\|\bar{u}(t_k)\|$ [m/s$^2$]")

    covariance_axis.plot(
        solution.feedforward[0], solution.feedforward[1], "k-+",
        lw=Plotter.TRAJECTORY_LINE_WIDTH, markersize=5,
        label=r"$\bar{u}(t_k)$", zorder=3,
    )
    extent = options.u_max
    for k, control_covariance in enumerate(solution.control_covariances):
        ellipse = covariance_ellipse_points(control_covariance, 3.0)
        ellipse = ellipse + solution.feedforward[:, k : k + 1]
        covariance_axis.plot(
            ellipse[0], ellipse[1], color=Plotter.DARK_GREY, lw=0.7,
            label=r"$\Sigma_{u(t_k)}$" if k == 0 else None,
            zorder=2,
        )
        extent = max(extent, float(np.max(np.abs(ellipse))))

    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    covariance_axis.plot(
        options.u_max * np.cos(angle), options.u_max * np.sin(angle),
        color=Plotter.RED, linestyle="--", lw=Plotter.GUIDE_LINE_WIDTH,
        label=r"$u_{\max}$", zorder=1,
    )
    limit = 1.1 * extent
    covariance_axis.set_xlim(-limit, limit)
    covariance_axis.set_ylim(-limit, limit)
    covariance_axis.set_aspect("equal", adjustable="box")
    covariance_axis.set_xlabel(r"$u_1$ [m/s$^2$]")
    covariance_axis.set_ylabel(r"$u_2$ [m/s$^2$]")

    for figure, axis, output_path in (
        (magnitude_figure, magnitude_axis, magnitude_path),
        (covariance_figure, covariance_axis, covariance_path),
    ):
        axis.set_box_aspect(1.0)
        axis.set_axisbelow(True)
        plotter._style_2d_axis(axis)
        plotter._legend(
            axis, loc="lower center", bbox_to_anchor=(0.5, 1.03), ncol=2,
            frameon=True, fancybox=False, edgecolor=Plotter.BLACK,
            facecolor="white", framealpha=1.0, borderpad=0.30,
            columnspacing=0.85, handlelength=1.5,
        )
        figure.subplots_adjust(left=0.16, right=0.95, bottom=0.12, top=0.82)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=Plotter.FIGURE_DPI)
        plt.close(figure)


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
    options = Options()
    A, B, G = discrete_time_matrices(options)
    Ac, _ = continuous_time_matrices()
    Qc, Lc = calibrate_continuous_diffusion(Ac, G @ G.T, options.dt)

    if np.max(np.abs(accumulated_process_covariance(Ac, Qc, options.dt) - G @ G.T)) > 1e-15:
        raise ValueError("Continuous diffusion state_covariance calibration error exceeds 1e-15.")

    shark_covariance_error = validate_shark_process_covariance(
        Ac, Qc, Lc, options.dt
    )
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
