"""
Chance-constrained covariance steering for Earth-Moon CR3BP low-thrust transfers test cases.

The mean trajectory is transcribed by multiple shooting (the node means are decision variables,
tied by matching conditions) while the covariance is single-shot forward from Sigma_0, so that the
propagated covariances are positive semi-definite by construction.

Uncertainty is propagated by the Unscented Transform with
kappa = 0 and a lower-triangular Cholesky factor. The sigma points are integrated
with the fixed-step RK4, wrapped in a CasADi Callback so the
optimiser sees the integration as a blackbox; the callback supplies the exact
Jacobian of the discrete RK4 map, obtained by differentiating the four stages.

Configuration: one Gaussian component (no GMM split, hence a single control
policy), no navigation error (R_bar = 0, H = I), deterministic dynamics (Q_k = 0),
and no mass path constraint. 

The initial guess is the fuel-optimal solution restricted to its knot
points, thus the interval durations are not uniform and the objective sum is weighted by them.
"""

from __future__ import annotations

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
INITIAL_STATE_STD_ND: dict[str, tuple[float, float]] = {
    "halo_l2_to_halo_l1": (1e-6, 1e-6),
    "lyapunov_l1_to_l2": (1e-6, 1e-6),
}
INITIAL_MASS_STD = 0.0      # [kg]
MASS_SCALE = 1e-3        # [kg]

# Callbacks must outlive the expression graph that references them.
# Prevents callback objects from being destroyed by garbage collector.
_CALLBACK_KEEPALIVE: list[casadi.Callback] = []


@dataclass(frozen=True)
class Options:
    """Solver and transcription settings."""

    # Keep every `mesh_stride`-th knot point of the DIRTRAN mesh. 
    mesh_stride: int = 4
    integrator_substeps: int = 4

    violation_parameter: float = 0.05   
    # Covariance reduction factor  
    covariance_reduction: float = 1e4    
    scaling_parameter: float = 0.0

    # Floors the covariance at jitter / n_x, which must stay far below the
    # terminal target while covering the negative eigenvalues of order 1e-15 that
    # round-off leaves behind once the steering contracts a direction.
    cholesky_jitter: float = 1e-11
    spectral_radius_floor: float = 1e-9
    # Box bound on the normalised gains. 
    gain_bound: float = 1e3
    # Lower bound on the diagonal of the terminal margin's Cholesky factor. The
    # map G -> G G^T is a diffeomorphism only where diag(G) > 0; at diag(G) = 0 its
    # Jacobian drops rank, so LICQ fails exactly when the terminal covariance
    # constraint becomes active -- which is precisely where the optimum sits.
    # Bounding the diagonal away from zero keeps the constraint Jacobian full rank
    # at the cost of tightening the requirement.
    terminal_margin_factor_floor: float = 1e-3

    # Target contribution of the cl control effort margin up to a
    # confidence level p after seeding
    cl_control_effort_seed_margin: float = 0.9        

    # IPOPT settings
    max_iter: int = 3000
    tol: float = 1e-6
    constr_viol_tol: float = 1e-9
    dual_inf_tol: float = 1e-4
    compl_inf_tol: float = 1e-6

    # Acceptable fallback
    acceptable_tol: float = 1e-5
    acceptable_constr_viol_tol: float = 1e-8
    acceptable_dual_inf_tol: float = 1e-4
    acceptable_compl_inf_tol: float = 1e-5
    acceptable_iter: int = 25
    print_level: int = 5

    monte_carlo_samples: int = 10000
    monte_carlo_seed: int = 42


def psi_inverse(dimension: int, beta: float) -> float:
    """Psi_d^-1(beta) = sqrt(Phi_d^-1(1 - beta)) with Phi_d the chi-squared CDF."""

    return float(np.sqrt(chi2.ppf(1.0 - beta, dimension)))


def control_covariance_weight(options: Options) -> float:

    """Weight of Eq. 5.86, or its unscented-consistent counterpart."""
    return 1.0 / (2.0 * (NX + options.scaling_parameter))



def unscented_weights(kappa: float) -> np.ndarray:
    """Sigma-point weights c_j of Eq. 5.82-5.83; c_0 vanishes for kappa = 0."""

    weights = np.full(N_SIGMA, 1.0 / (2.0 * (NX + kappa)))
    weights[0] = kappa / (NX + kappa)
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
    requirement reads P'_N <= 1e-4 I with O(1) entries. Gains are carried as
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

    position_std, velocity_std = INITIAL_STATE_STD_ND[case.test_case_id]
    mass_std = INITIAL_MASS_STD / case.m0_wet
    return np.array([position_std] * 3 + [velocity_std] * 3 + [mass_std], dtype=float)


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
        states: np.ndarray,
        controls: np.ndarray,
        step: float,
        substeps: int,
        with_jacobian: bool = False,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Fixed-step RK4 over one arc, zero-order hold on the control.

        Mirrors `integrator.rk4` exactly. When `with_jacobian` is set, the partial
        derivatives of the *discrete* map are accumulated through the four stages,
        so they are consistent with the propagation to machine precision rather
        than being an approximation of the variational equations.
        """

        count = states.shape[1]
        substep = step / substeps
        propagated = np.array(states, dtype=float)
        identity = np.eye(NX)
        state_sensitivity = np.broadcast_to(identity, (count, NX, NX)).copy() if with_jacobian else None
        control_sensitivity = np.zeros((count, NX, NU)) if with_jacobian else None

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
                k1 = self.augm_state_derivatives(propagated, controls)
                k2 = self.augm_state_derivatives(propagated + 0.5 * substep * k1, controls)
                k3 = self.augm_state_derivatives(propagated + 0.5 * substep * k2, controls)
                k4 = self.augm_state_derivatives(propagated + substep * k3, controls)

            propagated = propagated + (substep / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        return propagated, state_sensitivity, control_sensitivity


def _block_diagonal_sparsity(block_rows: int, block_columns: int, blocks: int) -> casadi.Sparsity:
    rows: list[int] = []
    columns: list[int] = []
    for block in range(blocks):
        for column in range(block_columns):
            for row in range(block_rows):
                rows.append(block * block_rows + row)
                columns.append(block * block_columns + column)
    return casadi.Sparsity.triplet(blocks * block_rows, blocks * block_columns, rows, columns)


def _block_index_map(sparsity: casadi.Sparsity, block_rows: int, block_columns: int):
    rows, columns = sparsity.get_triplet()
    rows = np.asarray(rows, dtype=int)
    columns = np.asarray(columns, dtype=int)
    return rows // block_rows, rows % block_rows, columns % block_columns


class ArcPropagatorJacobian(casadi.Callback):
    """Exact Jacobian of the blackbox arc map, block-diagonal over sigma points."""

    def __init__(self, name: str, parent: "ArcPropagator", input_names, output_names) -> None:
        casadi.Callback.__init__(self)
        self.parent = parent
        self._input_names = list(input_names) + list(output_names)
        self._output_names = [f"jac_{output_names[0]}_{item}" for item in input_names]
        self._sparsity = [
            _block_diagonal_sparsity(NX, NX, parent.n_sigma),
            _block_diagonal_sparsity(NX, NU, parent.n_sigma),
        ]
        self._index = [
            _block_index_map(self._sparsity[0], NX, NX),
            _block_index_map(self._sparsity[1], NX, NU),
        ]
        self.construct(name, {})

    def get_n_in(self) -> int:
        return 3

    def get_n_out(self) -> int:
        return 2

    def get_name_in(self, index: int) -> str:
        return self._input_names[index]

    def get_name_out(self, index: int) -> str:
        return self._output_names[index]

    def get_sparsity_in(self, index: int) -> casadi.Sparsity:
        rows = NU if index == 1 else NX
        return casadi.Sparsity.dense(rows, self.parent.n_sigma)

    def get_sparsity_out(self, index: int) -> casadi.Sparsity:
        return self._sparsity[index]

    def eval(self, arg):
        states = np.asarray(arg[0].full(), dtype=float)
        controls = np.asarray(arg[1].full(), dtype=float)
        _, state_sensitivity, control_sensitivity = self.parent.dynamics.propagate(
            states, controls, self.parent.step, self.parent.substeps, with_jacobian=True
        )
        outputs = []
        for blocks, sparsity, index in zip(
            (state_sensitivity, control_sensitivity), self._sparsity, self._index
        ):
            outputs.append(casadi.DM(sparsity, blocks[index].tolist()))
        return outputs


class ArcPropagator(casadi.Callback):
    """Blackbox RK4 propagation of the whole sigma-point ensemble over one arc."""

    def __init__(self, name: str, dynamics: Dynamics, n_sigma: int, step: float, substeps: int) -> None:
        casadi.Callback.__init__(self)
        self.dynamics = dynamics
        self.n_sigma = n_sigma
        self.step = float(step)
        self.substeps = int(substeps)
        self._jacobians: dict[str, ArcPropagatorJacobian] = {}
        self.construct(name, {})

    def get_n_in(self) -> int:
        return 2

    def get_n_out(self) -> int:
        return 1

    def get_name_in(self, index: int) -> str:
        return ("x", "u")[index]

    def get_name_out(self, index: int) -> str:
        return "z"

    def get_sparsity_in(self, index: int) -> casadi.Sparsity:
        return casadi.Sparsity.dense(NX if index == 0 else NU, self.n_sigma)

    def get_sparsity_out(self, index: int) -> casadi.Sparsity:
        return casadi.Sparsity.dense(NX, self.n_sigma)

    def eval(self, arg):
        states = np.asarray(arg[0].full(), dtype=float)
        controls = np.asarray(arg[1].full(), dtype=float)
        propagated, _, _ = self.dynamics.propagate(states, controls, self.step, self.substeps)
        return [casadi.DM(propagated)]

    def has_jacobian(self) -> bool:
        return True

    def get_jacobian(self, name, input_names, output_names, options):
        if name not in self._jacobians:
            jacobian = ArcPropagatorJacobian(name, self, input_names, output_names)
            self._jacobians[name] = jacobian
            _CALLBACK_KEEPALIVE.append(jacobian)
        return self._jacobians[name]


# --------------------------------------------------------------------------- #
# Symbolic helpers
# --------------------------------------------------------------------------- #


def cholesky_lower(matrix, dimension: int, pivot_floor: float = 0.0):
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
                # Floor the pivot, not merely the square root. A trial point where
                # the propagated covariance is numerically indefinite gives a
                # negative pivot; clamping it to something infinitesimal avoids the
                # NaN but the *next* line divides by it, so an infinitesimal clamp
                # produces factor entries of order 1e150 and the covariance
                # recursion detonates. `pivot_floor` is the jitter already added to
                # the matrix, which is exactly the smallest pivot an exactly
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
    turns the factorial expansion into `O(2^n n)`, which is negligible at the
    sizes used here (3 for the epigraph, 6 for the terminal constraint).
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


def largest_eigenvalue_symmetric_3(matrix, relative_floor: float = 1e-14):
    """Closed-form largest eigenvalue of a symmetric 3x3 matrix.

    Used instead of an epigraph slack constrained by Sylvester's criterion: the
    minors of `r^2 I - Sigma^T` scale as `r^2`, `r^4` and `r^6`, and with
    `r ~ 1e-5` here the second and third fall below any usable solver tolerance,
    which leaves the epigraph vacuous. The closed form is scale-free instead: its
    relative accuracy does not depend on the magnitude of the matrix.

    `p` is floored relative to the trace rather than by an absolute constant, so
    the removable singularity at an isotropic matrix is handled at any scale. The
    formula is smooth wherever the largest eigenvalue is simple; a repeated
    largest eigenvalue is non-generic for `Sigma^T = (n_x/6) K P' K^T` with a
    generic gain.
    """

    mean = (matrix[0, 0] + matrix[1, 1] + matrix[2, 2]) / 3.0
    off_diagonal = matrix[0, 1] ** 2 + matrix[0, 2] ** 2 + matrix[1, 2] ** 2
    deviation = (
        (matrix[0, 0] - mean) ** 2 + (matrix[1, 1] - mean) ** 2 + (matrix[2, 2] - mean) ** 2
    )
    spread = casadi.sqrt(deviation / 6.0 + off_diagonal / 3.0 + (relative_floor * mean) ** 2 + 1e-300)
    normalized = (matrix - mean * casadi.DM.eye(NU)) / spread
    argument = determinant(normalized, NU) / 2.0
    argument = casadi.fmax(casadi.fmin(argument, 1.0 - 1e-12), -1.0 + 1e-12)
    return mean + 2.0 * spread * casadi.cos(casadi.acos(argument) / 3.0)


def spectral_radius_expression(matrix, floor: float):
    """rho(A) = sqrt(lambda_max(A)), floored so the gradient stays bounded at A = 0.

    `sqrt` alone is non-smooth at the origin, which is reached whenever a gain
    vanishes. With the floor the augm_state_derivatives behaves like `K / floor` as `K -> 0`
    and therefore tends to zero rather than diverging.
    """

    return casadi.sqrt(largest_eigenvalue_symmetric_3(matrix) + floor ** 2)


def build_arc_function(
    case: TestCase,
    options: Options,
    dynamics: Dynamics,
    normalization: Normalization,
    step: float,
    index: int,
) -> casadi.Function:
    """One arc of Eq. 5.63: (mu, P, S_tilde, K_tilde) -> (mu_next, P_next, SigmaT_tilde).

    Everything around the integration is symbolic, so the chain rule assembles
    itself; only the RK4 propagation is opaque.
    """

    propagator = ArcPropagator(f"arc_rk4_{index}", dynamics, N_SIGMA, step, options.integrator_substeps)
    _CALLBACK_KEEPALIVE.append(propagator)

    mean = casadi.MX.sym("mu", NX)
    covariance = casadi.MX.sym("P", NX, NX)
    open_loop = casadi.MX.sym("S", NU)
    gain = casadi.MX.sym("K", NU, NP)

    weights = unscented_weights(options.scaling_parameter)
    scale = casadi.DM(normalization.matrix)
    inverse_scale = casadi.DM(normalization.inverse_matrix)

    # Sigma points, Eq. 5.80-5.81 with R_bar = 0 so that Y_j = X_j.
    spread = cholesky_lower(
        NX * covariance + options.cholesky_jitter * casadi.DM.eye(NX),
        NX,
        pivot_floor=options.cholesky_jitter,
    )
    physical_spread = scale @ spread
    primed_spread = spread[0:NP, :]

    states = [mean]
    controls = [open_loop]
    for column in range(NX):
        deviation = gain @ primed_spread[:, column]
        states.append(mean + physical_spread[:, column])
        controls.append(open_loop + deviation)
    for column in range(NX):
        deviation = gain @ primed_spread[:, column]
        states.append(mean - physical_spread[:, column])
        controls.append(open_loop - deviation)

    sigma_states = casadi.horzcat(*states)
    sigma_controls = case.max_thrust_nd * casadi.horzcat(*controls)
    propagated = propagator(sigma_states, sigma_controls)

    mean_next = sum(weights[j] * propagated[:, j] for j in range(N_SIGMA))
    covariance_next = casadi.MX.zeros(NX, NX)
    for j in range(N_SIGMA):
        residual = inverse_scale @ (propagated[:, j] - mean_next)
        covariance_next = covariance_next + weights[j] * residual @ residual.T
    covariance_next = 0.5 * (covariance_next + covariance_next.T)

    # Eq. 5.86. Columns 1..6 of the factor carry the position-velocity directions;
    # column 7 is mass-only, so its two sigma points have T_j = S_k and drop out.
    # The plus and minus points of a pair contribute identically, hence the 2.
    control_covariance = casadi.MX.zeros(NU, NU)
    for column in range(NP):
        deviation = gain @ primed_spread[:, column]
        control_covariance = control_covariance + deviation @ deviation.T
    control_covariance = 2.0 * control_covariance_weight(options) * control_covariance
    control_covariance = 0.5 * (control_covariance + control_covariance.T)

    return casadi.Function(
        f"arc_{index}",
        [mean, covariance, open_loop, gain],
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


def propagate_moments(
    arc_functions: list[casadi.Function],
    means: np.ndarray,
    open_loop: np.ndarray,
    gains: np.ndarray,
    initial_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Numerically evaluate the covariance recursion along a given mean trajectory.

    Calls the very same arc functions used by the NLP, so no logic is duplicated.
    """

    n_arcs = open_loop.shape[1]
    covariances = np.empty((n_arcs + 1, NX, NX), dtype=float)
    control_covariances = np.empty((n_arcs, NU, NU), dtype=float)
    propagated_means = np.empty((NX, n_arcs + 1), dtype=float)
    covariances[0] = initial_covariance
    propagated_means[:, 0] = means[:, 0]

    for k in range(n_arcs):
        mean_next, covariance_next, control_covariance = arc_functions[k](
            means[:, k], covariances[k], open_loop[:, k], gains[k]
        )
        propagated_means[:, k + 1] = np.asarray(mean_next.full(), dtype=float).ravel()
        covariances[k + 1] = np.asarray(covariance_next.full(), dtype=float)
        control_covariances[k] = np.asarray(control_covariance.full(), dtype=float)

    return propagated_means, covariances, control_covariances


def spectral_radii(matrices: np.ndarray) -> np.ndarray:
    """rho(A) = sqrt(lambda_max(A)) for a stack of symmetric matrices."""

    eigenvalues = np.linalg.eigvalsh(matrices)
    return np.sqrt(np.clip(eigenvalues[..., -1], 0.0, None))


def closed_loop_transition(
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


def _riccati_gains(
    control_weight: float,
    terminal_weight: float,
    state_matrices: list[np.ndarray],
    control_matrices: list[np.ndarray],
) -> np.ndarray:
    """Backward Riccati recursion on the normalised linearisation.

    The LQR convention is `delta_u = -K delta_x`, whereas Eq. 5.62 uses
    `T_k = S_k + K_k delta_x`, hence the sign flip; the mass column is dropped
    because the policy is driven by position and velocity only.
    """

    state_weight = np.eye(NX)
    weight = control_weight * np.eye(NU)
    riccati = terminal_weight * np.eye(NX)
    gains = np.zeros((len(state_matrices), NU, NP), dtype=float)

    for k in reversed(range(len(state_matrices))):
        state_matrix = state_matrices[k]
        control_matrix = control_matrices[k]
        gain = np.linalg.solve(
            weight + control_matrix.T @ riccati @ control_matrix,
            control_matrix.T @ riccati @ state_matrix,
        )
        gains[k] = -gain[:, 0:NP]
        closed_loop = state_matrix - control_matrix @ gain
        riccati = state_weight + gain.T @ weight @ gain + closed_loop.T @ riccati @ closed_loop
        riccati = 0.5 * (riccati + riccati.T)

    return gains


def _linear_terminal_covariance(
    gains: np.ndarray,
    state_matrices: list[np.ndarray],
    control_matrices: list[np.ndarray],
    initial_covariance: np.ndarray,
) -> float:
    """Terminal lambda_max(P'_N) under the linear closed-loop model."""

    covariance = np.array(initial_covariance, dtype=float)
    for k in range(len(state_matrices)):
        extended = np.zeros((NU, NX))
        extended[:, 0:NP] = gains[k]
        transition = state_matrices[k] + control_matrices[k] @ extended
        covariance = transition @ covariance @ transition.T
        covariance = 0.5 * (covariance + covariance.T)
        if not np.all(np.isfinite(covariance)):
            return float("inf")
    return float(np.linalg.eigvalsh(covariance[0:NP, 0:NP])[-1])


def seed_gains(
    options: Options,
    state_matrices: list[np.ndarray],
    control_matrices: list[np.ndarray],
) -> tuple[np.ndarray, tuple[float, float]]:
    """Warm-start gains from a Riccati recursion with Bryson-rule weights.

    Bryson's rule weights each channel by the inverse square of its largest
    acceptable deviation. In physical units, with `c` the sigma factor,

        Q_ii = 1 / (c sigma_0,i)^2,  R_jj = 1 / T_max^2,  Qf_ii = 1 / (c sigma_f,i)^2

    taking the initial dispersion as the acceptable state deviation, the thrust
    bound as the acceptable control, and the *terminal requirement* for `Qf`.
    Transforming into the normalised coordinates the recursion runs in
    (`dx = D0 dx~`, `dT = T_max du~`) turns the cost `dx' Q dx + du' R du` into
    `dx~' (D0 Q D0) dx~ + T_max^2 du~' R du~`, so

        Q~  = D0 Q D0  = (1/c^2) I,   R~ = T_max^2 R = I,   Qf~ = (1/c^2) reduction I

    because `sigma_f = sigma_0 / sqrt(reduction)` makes `Qf/Q = reduction` exactly.
    Only ratios matter -- scaling all three by a constant leaves the gains
    unchanged -- so this is equivalent to `Q = I`, `R = c^2`, `Qf = reduction`,
    which is how it is passed below. Note `Qf/Q` is not a tuning constant: it is
    the covariance reduction factor itself.

    This replaced a 407-point scan over `(R, Qf)`. The scan ranked candidates by a
    linear closed-loop model, which agrees with the unscented propagation for
    moderate gains and disagrees by orders of magnitude for aggressive ones, so it
    selected exactly the region where its own ranking was invalid. Measured at the
    full mesh, Bryson matches the linear and unscented models to four digits
    (ratio 1.000 on the Lyapunov case) where the scanned optimum showed a 10 %
    gap -- the scanned policy contracts the covariance far enough to reach the
    round-off floor of the sigma-point differencing. Bryson is nominally 11x
    worse on that case and indistinguishable on the Halo case, but both are two
    orders inside the requirement, so the difference is irrelevant for a warm
    start and the better-conditioned one is preferable.
    """

    control_weight = 3.0 ** 2
    terminal_weight = float(options.covariance_reduction)
    gains = _riccati_gains(control_weight, terminal_weight, state_matrices, control_matrices)
    return gains, (control_weight, terminal_weight)


@dataclass
class InitialGuess:
    means: np.ndarray
    open_loop: np.ndarray
    slack: np.ndarray
    gains: np.ndarray
    radius: np.ndarray
    terminal_covariance: np.ndarray
    gain_scale: np.ndarray


def _initial_margin_factor(terminal_covariance: np.ndarray, target: float) -> np.ndarray:
    """Cholesky factor of the terminal margin, or its PSD part when infeasible.

    The seed leaves the terminal covariance well above target, so the margin is
    indefinite and has no Cholesky factor; clipping the negative eigenvalues gives
    the nearest PSD matrix, which is a serviceable starting value for G.
    """

    margin = np.eye(NP) - terminal_covariance[0:NP, 0:NP] / target
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (margin + margin.T))
    clipped = eigenvectors @ np.diag(np.clip(eigenvalues, 1e-12, None)) @ eigenvectors.T
    factor = np.linalg.cholesky(clipped)
    return np.array([factor[row, column] for row in range(NP) for column in range(row + 1)])


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

    The DIRTRAN solution is bang-bang at exactly T_max, so Eq. 5.64 is violated at
    every thrust arc as soon as any feedback is present. The seed gains are scaled
    down until the closed-loop share of the budget leaves room, then the open-loop
    magnitudes are trimmed to fit under what remains.
    """

    means = np.array(ref_traj.states, dtype=float)
    means[:, 0] = case.x0_augmented_state
    open_loop = np.array(ref_traj.controls, dtype=float) / case.max_thrust_nd

    state_matrices, control_matrices = closed_loop_transition(
        case, dynamics, normalization, ref_traj.states, ref_traj.controls,
        ref_traj.steps, options.integrator_substeps,
    )
    # Variable scaling for the gains. In normalised units ||B_tilde|| ~ 5e5,
    # because correcting a 1-sigma deviation costs a negligible fraction of
    # T_max; a unit step in K_tilde would therefore change the closed-loop
    # transition by half a million and every IPOPT step overshoots by orders of
    # magnitude. Scaling by 1 / ||B_tilde|| makes a unit step in the decision
    # variable an O(1) change in the closed-loop dynamics, which is the condition
    # the solver implicitly assumes.
    gain_scale = np.array(
        [1.0 / max(np.linalg.norm(matrix, 2), 1e-300) for matrix in control_matrices]
    )
    gains, seed_weights = seed_gains(options, state_matrices, control_matrices)
    _, seeded_covariances, _ = propagate_moments(
        arc_functions, means, open_loop, gains, normalization.initial_covariance
    )
    unscented_terminal = float(
        np.linalg.eigvalsh(seeded_covariances[-1][0:NP, 0:NP])[-1]
    )
    linear_terminal = _linear_terminal_covariance(
        gains, state_matrices, control_matrices, normalization.initial_covariance
    )
    # The two models agreeing is the check that the seed is not sitting on the
    # round-off floor of the sigma-point differencing; a large gap means the
    # unscented number is noise rather than a tighter covariance.
    print(
        f"    Bryson seed (Q=I, R={seed_weights[0]:.4g}, Qf={seed_weights[1]:.4g}): "
        f"terminal lambda_max = {unscented_terminal * options.covariance_reduction:.4f} "
        f"x target (linear model "
        f"{linear_terminal * options.covariance_reduction:.4f})",
        flush=True,
    )


    radius = np.zeros(ref_traj.n_arcs, dtype=float)
    covariances = np.tile(normalization.initial_covariance, (ref_traj.n_arcs + 1, 1, 1))
    for _ in range(30):
        _, covariances, control_covariances = propagate_moments(
            arc_functions, means, open_loop, gains, normalization.initial_covariance
        )
        radius = spectral_radii(control_covariances)
        worst = float(np.max(psi_inv * radius))
        print(worst)
        if not np.isfinite(worst):
            gains *= 0.5
            continue
        if worst <= options.cl_control_effort_seed_margin:
            break
        gains *= np.sqrt(options.cl_control_effort_seed_margin / worst)

    headroom = np.clip(1.0 - psi_inv * radius, 0.0, None)
    magnitudes = np.linalg.norm(open_loop, axis=0)
    factor = np.ones_like(magnitudes)
    active = magnitudes > 1e-12
    factor[active] = np.minimum(1.0, headroom[active] / magnitudes[active])
    open_loop = open_loop * factor

    return InitialGuess(
        means=means,
        open_loop=open_loop,
        slack=np.linalg.norm(open_loop, axis=0),
        gains=gains,
        radius=radius,
        terminal_covariance=covariances[-1],
        gain_scale=gain_scale,
    )


# --------------------------------------------------------------------------- #
# Nonlinear program
# --------------------------------------------------------------------------- #


@dataclass
class SteeringSolution:
    means: np.ndarray
    open_loop: np.ndarray          # S_tilde, normalised by T_max
    gains: np.ndarray              # K_tilde, normalised
    radius: np.ndarray             # r_tilde
    covariances: np.ndarray        # P, normalised
    control_covariances: np.ndarray
    objective: float
    diagnostics: dict[str, float] = field(default_factory=dict)


def solve_steering_problem(
    case: TestCase,
    options: Options,
    ref_traj: ReferenceTraj,
    arc_functions: list[casadi.Function],
    guess: InitialGuess,
    normalization: Normalization,
    psi_inv: float,
    log_prefix: str,
) -> SteeringSolution:
    """Transcribe and solve Eq. 5.66."""

    n_arcs = ref_traj.n_arcs
    target = 1.0 / options.covariance_reduction

    opti = casadi.Opti()
    means = opti.variable(NX, n_arcs + 1)
    open_loop = opti.variable(NU, n_arcs)
    slack = opti.variable(1, n_arcs)
    # One matrix variable per arc rather than a flattened block: `casadi.reshape`
    # is column-major while NumPy defaults to row-major, and silently transposing
    # the gains is easy to do and hard to notice.
    gains = [opti.variable(NU, NP) for _ in range(n_arcs)]
    # Cholesky factor of the terminal covariance margin; see below.
    margin_factor = opti.variable(NP * (NP + 1) // 2)

    opti.subject_to(means[:, 0] == case.x0_augmented_state)
    opti.subject_to(casadi.vec(slack) >= 0.0)
    for gain in gains:
        opti.subject_to(opti.bounded(-options.gain_bound, casadi.vec(gain), options.gain_bound))

    covariance = casadi.DM(normalization.initial_covariance)
    objective = 0.0
    for k in range(n_arcs):
        mean_next, covariance, control_covariance = arc_functions[k](
            means[:, k], covariance, open_loop[:, k], float(guess.gain_scale[k]) * gains[k]
        )
        # Multiple shooting on the mean, single shooting on the covariance.
        opti.subject_to(means[:, k + 1] == mean_next)

        # ||S_k|| via a slack, as in deterministic_cr3bp, so the cost stays smooth at
        # the coast arcs where S_k vanishes.
        opti.subject_to(casadi.dot(open_loop[:, k], open_loop[:, k]) <= slack[0, k] ** 2)

        # rho(SigmaT_k) in closed form rather than as a constrained slack.
        radius = spectral_radius_expression(control_covariance, options.spectral_radius_floor)

        # Eq. 5.64, normalised by T_max.
        opti.subject_to(slack[0, k] + psi_inv * radius <= 1.0)

        # Eq. 5.65, weighted by the arc duration because the mesh is not uniform.
        objective = objective + float(ref_traj.steps[k]) * (slack[0, k] + psi_inv * radius)

    opti.subject_to(means[0:NP, n_arcs] == case.xf_state)

    # Eq. 5.56 after the congruence scaling of the note. Written as a Cholesky
    # residual rather than through leading principal minors: the sixth minor is a
    # degree-6 polynomial that reaches 1e41 when the terminal covariance starts
    # three orders of magnitude above target, whereas the residual below never
    # exceeds the norm of the matrix itself. G lower triangular with a
    # non-negative diagonal exists if and only if the left-hand side is PSD.
    lower_triangular = casadi.MX.zeros(NP, NP)
    entry = 0
    for row in range(NP):
        for column in range(row + 1):
            lower_triangular[row, column] = margin_factor[entry]
            entry += 1
    opti.subject_to(casadi.diag(lower_triangular) >= options.terminal_margin_factor_floor)
    terminal = casadi.DM.eye(NP) - covariance[0:NP, 0:NP] / target
    residual = terminal - lower_triangular @ lower_triangular.T
    for row in range(NP):
        for column in range(row + 1):
            opti.subject_to(residual[row, column] == 0.0)

    opti.minimize(objective)

    opti.set_initial(means, guess.means)
    opti.set_initial(open_loop, guess.open_loop)
    opti.set_initial(slack, np.maximum(guess.slack, 1e-9).reshape(1, -1))
    for k in range(n_arcs):
        opti.set_initial(gains[k], guess.gains[k] / guess.gain_scale[k])
    opti.set_initial(margin_factor, _initial_margin_factor(guess.terminal_covariance, target))

    ipopt_options = {
        "max_iter": options.max_iter,
        "tol": options.tol,
        "acceptable_tol": options.acceptable_tol,
        "constr_viol_tol": options.constr_viol_tol,
        "acceptable_constr_viol_tol": options.acceptable_constr_viol_tol,
        "acceptable_iter": 25,
        "dual_inf_tol": options.dual_inf_tol,
        "acceptable_dual_inf_tol": options.acceptable_dual_inf_tol,
        "compl_inf_tol": options.compl_inf_tol,
        "acceptable_compl_inf_tol": options.acceptable_compl_inf_tol,
        "mu_strategy": "monotone",
        # The arc map is a first-order blackbox, so no exact Hessian is available.
        "hessian_approximation": "limited-memory",
        "limited_memory_max_history": options.acceptable_iter,
        "print_level": options.print_level,
        "sb": "yes",
    }
    # `expand` must stay off: an MX graph containing Callbacks cannot be expanded.
    opti.solver("ipopt", {"expand": False, "print_time": True}, ipopt_options)

    print(f"{log_prefix}  Solving: {n_arcs} arcs, "
          f"{(n_arcs + 1) * NX + n_arcs * (NU + 1 + NU * NP + 1)} variables", flush=True)
    try:
        solution = opti.solve()
        converged = True
    except RuntimeError as error:
        print(f"{log_prefix}  IPOPT did not converge ({error}); returning the last iterate.", flush=True)
        solution = opti.debug
        converged = False

    solved_means = np.asarray(solution.value(means), dtype=float)
    solved_open_loop = np.asarray(solution.value(open_loop), dtype=float).reshape(NU, n_arcs)
    solved_gains = np.stack(
        [
            guess.gain_scale[k] * np.asarray(solution.value(gain), dtype=float).reshape(NU, NP)
            for k, gain in enumerate(gains)
        ]
    )

    propagated_means, covariances, control_covariances = propagate_moments(
        arc_functions, solved_means, solved_open_loop, solved_gains, normalization.initial_covariance
    )
    matching_defect = float(np.max(np.abs(propagated_means[:, 1:] - solved_means[:, 1:])))
    # rho is an expression, so recover it exactly from the propagated moments.
    solved_radius = np.sqrt(
        spectral_radii(control_covariances) ** 2 + options.spectral_radius_floor ** 2
    )

    return SteeringSolution(
        means=solved_means,
        open_loop=solved_open_loop,
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


def linear_covariance_check(
    case: TestCase,
    options: Options,
    dynamics: Dynamics,
    normalization: Normalization,
    solution: SteeringSolution,
    steps: np.ndarray,
) -> np.ndarray:
    """Re-propagate the covariance linearly through the closed-loop transition.

    Linear propagation forms no differences of nearby O(1) numbers, so it does not
    suffer the cancellation described in the notes. Because the dispersions are
    tiny the dynamics are effectively linear across them, so any disagreement with
    the unscented result measures the round-off floor rather than a modelling gap.
    """

    reference_controls = case.max_thrust_nd * solution.open_loop
    state_matrices, control_matrices = closed_loop_transition(
        case, dynamics, normalization, solution.means, reference_controls, steps, options.integrator_substeps
    )
    covariances = np.empty_like(solution.covariances)
    covariances[0] = normalization.initial_covariance
    for k in range(steps.size):
        extended_gain = np.zeros((NU, NX))
        extended_gain[:, 0:NP] = solution.gains[k]
        transition = state_matrices[k] + control_matrices[k] @ extended_gain
        propagated = transition @ covariances[k] @ transition.T
        covariances[k + 1] = 0.5 * (propagated + propagated.T)
    return covariances


def monte_carlo(
    case: TestCase,
    options: Options,
    dynamics: Dynamics,
    normalization: Normalization,
    solution: SteeringSolution,
    steps: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Apply the converged policy (Eq. 5.69) with thrust saturation.

    Violation statistics use the unsaturated commanded thrust so they still test
    the chance constraint.  Propagation, accumulated effort and peak applied
    thrust use the command clipped to ``T_max``, matching the physical actuator.
    """

    generator = np.random.default_rng(options.monte_carlo_seed)
    samples = options.monte_carlo_samples
    states = solution.means[:, 0:1] + normalization.initial_std[:, None] * generator.standard_normal((NX, samples))

    cumulative = np.zeros(samples, dtype=float)
    violation_fraction = np.zeros(steps.size, dtype=float)
    tolerant_violation_fraction = np.zeros(steps.size, dtype=float)
    worst_exceedance = -np.inf
    peak_thrust = np.zeros(samples, dtype=float)

    for k in range(steps.size):
        deviation = (states[0:NP] - solution.means[0:NP, k : k + 1]) / normalization.scale[0:NP, None]
        commanded_control = case.max_thrust_nd * (
            solution.open_loop[:, k : k + 1] + solution.gains[k] @ deviation
        )
        commanded_magnitudes = np.linalg.norm(commanded_control, axis=0)
        # A bare `>` test is misleading when the open-loop magnitude sits within
        # solver tolerance of the bound: a 1e-5 residual infeasibility then reads
        # as a 100 % chance-constraint violation. Report both the strict fraction
        # and one with a relative tolerance, plus the worst exceedance, so the
        # magnitude of any breach is visible alongside its frequency.
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
        control = commanded_control * saturation_scale[None, :]
        applied_magnitudes = np.minimum(commanded_magnitudes, case.max_thrust_nd)

        peak_thrust = np.maximum(peak_thrust, applied_magnitudes)
        cumulative += float(steps[k]) * applied_magnitudes
        states, _, _ = dynamics.propagate(states, control, float(steps[k]), options.integrator_substeps)

    terminal_deviation = (states - solution.means[:, -1:]) / normalization.scale[:, None]
    empirical_covariance = np.cov(terminal_deviation)

    return {
        "cumulative": cumulative,
        "percentile": float(np.percentile(cumulative, 100.0 * (1.0 - options.violation_parameter))),
        "mean_cost": float(np.mean(cumulative)),
        "violation_fraction": violation_fraction,
        "max_violation_fraction": float(np.max(violation_fraction)),
        "worst_exceedance_n": float(worst_exceedance) * case.thrust_unit,
        "peak_thrust_n": peak_thrust * case.thrust_unit,
        "terminal_covariance": empirical_covariance,
        "terminal_deviation": terminal_deviation,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def compute_diagnostics(
    case: TestCase,
    options: Options,
    ref_traj: ReferenceTraj,
    solution: SteeringSolution,
    normalization: Normalization,
    linear_covariances: np.ndarray,
    monte_carlo_result: dict,
    psi_inv: float,
) -> dict[str, float]:
    target = 1.0 / options.covariance_reduction
    steps = ref_traj.steps

    open_loop_magnitude = np.linalg.norm(solution.open_loop, axis=0)
    deterministic_effort = float(np.sum(steps * open_loop_magnitude))
    feedback_effort = float(np.sum(steps * psi_inv * solution.radius))
    total_effort = deterministic_effort + feedback_effort
    to_kg = case.m0_wet * case.max_thrust_nd / case.exhaust_velocity_nd

    terminal_normalized = solution.covariances[-1][0:NP, 0:NP] / target
    terminal_eigenvalues = np.linalg.eigvalsh(terminal_normalized)

    unscented_terminal = solution.covariances[-1]
    linear_terminal = linear_covariances[-1]
    scale = max(np.max(np.abs(unscented_terminal)), 1e-300)
    covariance_mismatch = float(np.max(np.abs(unscented_terminal - linear_terminal)) / scale)

    mean_error = float(np.max(np.abs(solution.means[0:NP, -1] - case.xf_state)))
    budget = open_loop_magnitude + psi_inv * solution.radius

    empirical = monte_carlo_result["terminal_covariance"][0:NP, 0:NP] / target
    empirical_eigenvalues = np.linalg.eigvalsh(empirical)

    return {
        "n_arcs": float(ref_traj.n_arcs),
        "mesh_stride": float(options.mesh_stride),
        "converged": solution.diagnostics["converged"],
        "matching_defect_nd": solution.diagnostics["matching_defect_nd"],
        "objective_nd": float(solution.objective),
        "deterministic_effort_nd": deterministic_effort,
        "feedback_effort_nd": feedback_effort,
        "total_effort_nd": total_effort,
        "deterministic_effort_kg": deterministic_effort * to_kg,
        "feedback_effort_kg": feedback_effort * to_kg,
        "total_effort_kg": total_effort * to_kg,
        "dirtran_fuel_kg": ref_traj.fuel_consumed,
        "max_thrust_budget": float(np.max(budget)),
        "max_open_loop_n": float(np.max(open_loop_magnitude) * case.max_thrust_nd * case.thrust_unit),
        "max_feedback_n": float(
            np.max(psi_inv * solution.radius) * case.max_thrust_nd * case.thrust_unit
        ),
        "terminal_mean_error_nd": mean_error,
        "terminal_covariance_max_eigenvalue": float(terminal_eigenvalues[-1]),
        "terminal_covariance_satisfied": float(terminal_eigenvalues[-1] <= 1.0 + 1e-6),
        "terminal_position_std_m": float(
            np.sqrt(solution.covariances[-1][0, 0]) * normalization.scale[0] * case.length_unit * 1e3
        ),
        "terminal_velocity_std_ms": float(
            np.sqrt(solution.covariances[-1][3, 3]) * normalization.scale[3] * case.velocity_unit * 1e3
        ),
        "unscented_vs_linear_mismatch": covariance_mismatch,
        # J of Eq. 5.65 re-evaluated from ||S_k|| rather than from the slack
        # sigma_k >= ||S_k||. The two coincide only at convergence, so comparing
        # the Monte Carlo percentile against the raw NLP objective is misleading
        # on a truncated run.
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
    log_prefix: str,
) -> None:
    to_n = case.max_thrust_nd * case.thrust_unit
    lines = [
        "",
        f"{log_prefix}{'=' * 68}",
        f"{log_prefix}{case.display_name}  --  covariance steering summary",
        f"{log_prefix}{'=' * 68}",
        f"{log_prefix}arcs                      : {diagnostics['n_arcs']:.0f} "
        f"(mesh stride {diagnostics['mesh_stride']:.0f})",
        f"{log_prefix}IPOPT converged           : {'yes' if diagnostics['converged'] else 'NO'}",
        f"{log_prefix}max matching defect       : {diagnostics['matching_defect_nd']:.3e} [-]",
        f"{log_prefix}Psi_3^-1(beta)            : {psi_inv:.6f}",
        f"{log_prefix}",
        f"{log_prefix}objective J (Eq. 5.65)    : {diagnostics['total_effort_nd']:.6e} [-]"
        f"   (NLP slack value {diagnostics['objective_nd']:.6e})",
        f"{log_prefix}  open-loop  T_d          : {diagnostics['deterministic_effort_kg']:.6f} kg",
        f"{log_prefix}  closed-loop T_s         : {diagnostics['feedback_effort_kg']:.6f} kg",
        f"{log_prefix}  total                   : {diagnostics['total_effort_kg']:.6f} kg",
        f"{log_prefix}DIRTRAN nominal fuel      : {diagnostics['dirtran_fuel_kg']:.6f} kg",
        f"{log_prefix}",
        f"{log_prefix}max ||S|| + Psi rho       : {diagnostics['max_thrust_budget']:.6f} "
        f"(must be <= 1, i.e. {to_n:.4f} N)",
        f"{log_prefix}  peak open-loop thrust   : {diagnostics['max_open_loop_n']:.6f} N",
        f"{log_prefix}  peak feedback margin    : {diagnostics['max_feedback_n']:.6f} N",
        f"{log_prefix}",
        f"{log_prefix}terminal mean error       : {diagnostics['terminal_mean_error_nd']:.3e} [-]",
        f"{log_prefix}terminal cov. max eigval  : {diagnostics['terminal_covariance_max_eigenvalue']:.6f} "
        f"(must be <= 1)",
        f"{log_prefix}  terminal 1-sigma pos.   : {diagnostics['terminal_position_std_m']:.6e} m",
        f"{log_prefix}  terminal 1-sigma vel.   : {diagnostics['terminal_velocity_std_ms']:.6e} m/s",
        f"{log_prefix}unscented vs linear cov.  : {diagnostics['unscented_vs_linear_mismatch']:.3e} relative",
        f"{log_prefix}",
        f"{log_prefix}Monte Carlo 95th pct.     : {diagnostics['monte_carlo_percentile_nd']:.6e} [-] "
        f"vs predicted {diagnostics['predicted_percentile_nd']:.6e}",
        f"{log_prefix}Monte Carlo mean          : {diagnostics['monte_carlo_mean_nd']:.6e} [-]",
        f"{log_prefix}worst-arc P(||T|| > Tmax) : {diagnostics['monte_carlo_max_violation_fraction']:.4f} "
        f"(must be <= 0.05); worst exceedance "
        f"{diagnostics['monte_carlo_worst_exceedance_n']:+.3e} N",
        f"{log_prefix}  same, 1e-4 relative tol : "
        f"{log_prefix}MC terminal max eigval    : {diagnostics['monte_carlo_terminal_max_eigenvalue']:.6f}",
        f"{log_prefix}{'=' * 68}",
    ]
    print("\n".join(lines), flush=True)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def standard_deviations_physical(
    case: TestCase, normalization: Normalization, covariances: np.ndarray
) -> np.ndarray:
    """1-sigma of each augmented state in metres, m/s and kg."""

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
    solution: SteeringSolution,
    normalization: Normalization,
    monte_carlo_result: dict,
    psi_inv: float,
    output_prefix: Path,
) -> None:
    node_days = ref_traj.node_times * case.time_unit / 86400.0
    arc_days = node_days[:-1]
    target = 1.0 / options.covariance_reduction
    sigma_physical = standard_deviations_physical(case, normalization, solution.covariances)
    target_sigma = standard_deviations_physical(
        case, normalization, (target * np.eye(NX))[None, :, :]
    )[0]

    # 1 - dispersion envelopes.
    figure, axes = plt.subplots(1, 3, figsize=Plotter.THREE_PANEL_FIGSIZE, dpi=Plotter.FIGURE_DPI)
    panels = (
        (slice(0, 3), r"position $1\sigma$ [m]", ("x", "y", "z")),
        (slice(3, 6), r"velocity $1\sigma$ [m/s]", (r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$")),
        (slice(6, 7), r"mass $1\sigma$ [kg]", ("m",)),
    )
    for axis, (columns, label, names) in zip(axes, panels):
        for offset, name in enumerate(names):
            axis.semilogy(node_days, sigma_physical[:, columns][:, offset], lw=1.0, label=name)
        if columns.start < NP:
            axis.axhline(target_sigma[columns.start], color=Plotter.RED, ls="--", lw=0.8, label="target")
        axis.set_xlabel("time [days]")
        axis.set_ylabel(label)
        axis.legend(fontsize=6, frameon=False)
        axis.grid(True, which="both", alpha=0.25, lw=0.4)
    figure.tight_layout()
    figure.savefig(output_prefix.parent / f"{output_prefix.name}_dispersion.png",
                   dpi=Plotter.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)

    # 2 - thrust budget: open loop, feedback allowance and their sum against T_max.
    open_loop_n = np.linalg.norm(solution.open_loop, axis=0) * case.max_thrust_nd * case.thrust_unit
    feedback_n = psi_inv * solution.radius * case.max_thrust_nd * case.thrust_unit
    # Two panels: the closed-loop allowance is ~1e-4 N against a 0.5 N budget, so
    # it is invisible next to the open-loop term on a shared linear axis.
    figure, axes = plt.subplots(1, 2, figsize=Plotter.WIDE_FIGSIZE, dpi=Plotter.FIGURE_DPI)
    axis = axes[0]
    axis.step(arc_days, open_loop_n + feedback_n, where="post", lw=1.1, color=Plotter.RED, label="total")
    axis.step(arc_days, open_loop_n, where="post", lw=0.9, ls="--", color=Plotter.BLACK,
              label=r"$\|S_k\|$")
    axis.axhline(case.max_thrust_n, color=Plotter.GREY, ls="--", lw=0.9, label=r"$T_{\max}$")
    axis.set_xlabel("time [days]")
    axis.set_ylabel("thrust [N]")
    axis.set_ylim(0.0, 1.08 * case.max_thrust_n)
    axis.legend(fontsize=6, frameon=False)
    axis.grid(True, alpha=0.25, lw=0.4)

    axis = axes[1]
    axis.step(arc_days, np.maximum(feedback_n, 1e-16), where="post", lw=0.9, color=Plotter.BLUE)
    axis.set_yscale("log")
    axis.set_xlabel("time [days]")
    axis.set_ylabel(r"$\Psi^{-1}_{n_T}(\beta)\,\rho(\Sigma^T_k)$ [N]")
    axis.set_title("closed-loop allowance", fontsize=7)
    axis.grid(True, which="both", alpha=0.25, lw=0.4)
    figure.tight_layout()
    figure.savefig(output_prefix.parent / f"{output_prefix.name}_thrust_budget.png",
                   dpi=Plotter.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)

    # 3 - covariance ellipses along the transfer, and the terminal one vs target.
    figure, axes = plt.subplots(1, 2, figsize=Plotter.TRIPLE_FIGSIZE[0:2], dpi=Plotter.FIGURE_DPI)
    axis = axes[0]
    axis.plot(solution.means[0], solution.means[1], color=Plotter.BLACK, lw=0.9, label="mean")
    stride = max(solution.covariances.shape[0] // 25, 1)
    length_scale = case.length_unit
    for k in range(0, solution.covariances.shape[0], stride):
        block = solution.covariances[k][0:2, 0:2] * np.outer(normalization.scale[0:2], normalization.scale[0:2])
        points = _ellipse_points(block, 3.0)
        axis.plot(solution.means[0, k] + points[0], solution.means[1, k] + points[1],
                  color=Plotter.BLUE, lw=0.5, alpha=0.7)
    axis.set_xlabel("x [-]")
    axis.set_ylabel("y [-]")
    axis.set_title(r"$3\sigma$ ellipses (xy)", fontsize=8)
    axis.grid(True, alpha=0.25, lw=0.4)

    axis = axes[1]
    terminal = solution.covariances[-1][0:2, 0:2] * np.outer(normalization.scale[0:2], normalization.scale[0:2])
    target_block = target * np.eye(2) * np.outer(normalization.scale[0:2], normalization.scale[0:2])
    for block, colour, label in ((terminal, Plotter.BLUE, "achieved"), (target_block, Plotter.RED, "target")):
        points = _ellipse_points(block, 3.0) * length_scale * 1e3
        axis.plot(points[0], points[1], color=colour, lw=1.0, label=label)
    scatter = monte_carlo_result["terminal_deviation"][0:2] * normalization.scale[0:2, None] * length_scale * 1e3
    axis.scatter(scatter[0], scatter[1], s=1.0, color=Plotter.GREY, alpha=0.35, label="Monte Carlo")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title(r"terminal $3\sigma$", fontsize=8)
    axis.legend(fontsize=6, frameon=False)
    axis.grid(True, alpha=0.25, lw=0.4)
    axis.set_aspect("equal", adjustable="datalim")
    figure.tight_layout()
    figure.savefig(output_prefix.parent / f"{output_prefix.name}_ellipses.png",
                   dpi=Plotter.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)

    # 4 - Monte Carlo cost distribution against the optimised percentile.
    to_kg = case.m0_wet / case.exhaust_velocity_nd
    figure, axis = plt.subplots(figsize=Plotter.SQUARE_DIAGNOSTIC_FIGSIZE, dpi=Plotter.FIGURE_DPI)
    axis.hist(monte_carlo_result["cumulative"] * to_kg, bins=60, color=Plotter.GREY, alpha=0.75)
    axis.axvline(monte_carlo_result["percentile"] * to_kg, color=Plotter.BLUE, lw=1.1,
                 label="Monte Carlo 95th pct.")
    # Eq. 5.65 recomputed from ||S_k||, not the raw NLP objective: the latter is
    # built on the slack sigma_k >= ||S_k||, which only coincides with ||S_k|| at
    # full convergence and otherwise plots a bound that is not one.
    predicted = float(
        np.sum(
            ref_traj.steps
            * (np.linalg.norm(solution.open_loop, axis=0) + psi_inv * solution.radius)
        )
    )
    axis.axvline(predicted * case.max_thrust_nd * to_kg, color=Plotter.RED, ls="--", lw=1.1,
                 label=r"optimised bound (Eq. 5.65)")
    axis.set_xlabel("propellant [kg]")
    axis.set_ylabel("samples")
    axis.legend(fontsize=7, frameon=False)
    axis.grid(True, alpha=0.25, lw=0.4)
    figure.tight_layout()
    figure.savefig(output_prefix.parent / f"{output_prefix.name}_monte_carlo.png",
                   dpi=Plotter.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def _ellipse_points(block: np.ndarray, scale: float, samples: int = 120) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(block)
    radii = scale * np.sqrt(np.clip(eigenvalues, 0.0, None))
    angles = np.linspace(0.0, 2.0 * np.pi, samples)
    circle = np.vstack([np.cos(angles), np.sin(angles)])
    return eigenvectors @ (radii[:, None] * circle)


def save_outputs(
    case: TestCase,
    options: Options,
    ref_traj: ReferenceTraj,
    solution: SteeringSolution,
    normalization: Normalization,
    linear_covariances: np.ndarray,
    monte_carlo_result: dict,
    diagnostics: dict[str, float],
    psi_inv: float,
    output_prefix: Path,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    node_days = ref_traj.node_times * case.time_unit / 86400.0

    # Physical gains, undoing the normalisation: K = T_max * K_tilde * inv(D0').
    physical_gains = (
        case.max_thrust_nd * solution.gains / normalization.scale[None, None, 0:NP]
    )
    sigma_physical = standard_deviations_physical(case, normalization, solution.covariances)

    np.savez(
        output_prefix.with_suffix(".npz"),
        node_times_nd=ref_traj.node_times,
        node_days=node_days,
        steps_nd=ref_traj.steps,
        means=solution.means,
        open_loop_normalized=solution.open_loop,
        open_loop_n=solution.open_loop * case.max_thrust_nd * case.thrust_unit,
        gains_normalized=solution.gains,
        gains=physical_gains,
        radius_normalized=solution.radius,
        covariance_normalized=solution.covariances,
        covariance_linear_check=linear_covariances,
        control_covariance_normalized=solution.control_covariances,
        sigma_physical=sigma_physical,
        normalization_scale_nd=normalization.scale,
        INITIAL_STATE_STD_ND=normalization.initial_std,
        target_ratio=1.0 / options.covariance_reduction,
        psi_inv=psi_inv,
        monte_carlo_cumulative=monte_carlo_result["cumulative"],
        monte_carlo_violation_fraction=monte_carlo_result["violation_fraction"],
        monte_carlo_terminal_covariance=monte_carlo_result["terminal_covariance"],
        diagnostics=np.array(diagnostics, dtype=object),
    )

    open_loop_n = np.linalg.norm(solution.open_loop, axis=0) * case.max_thrust_nd * case.thrust_unit
    feedback_n = psi_inv * solution.radius * case.max_thrust_nd * case.thrust_unit
    table = np.column_stack(
        [
            node_days[:-1],
            ref_traj.steps,
            solution.means[:, :-1].T,
            solution.open_loop.T * case.max_thrust_nd * case.thrust_unit,
            open_loop_n,
            feedback_n,
            open_loop_n + feedback_n,
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

    print(f"{log_prefix}Building {ref_traj.n_arcs} arc functions "
          f"({options.integrator_substeps} RK4 substeps each)", flush=True)
    arc_functions = _build_arc_functions(case, options, dynamics, normalization, ref_traj)

    print(f"{log_prefix}Building the initial guess", flush=True)
    guess = build_initial_guess(
        case,
        options,
        dynamics,
        normalization,
        ref_traj,
        arc_functions,
        psi_inv,
    )

    solution = solve_steering_problem(
        case, options, ref_traj, arc_functions, guess, normalization, psi_inv, log_prefix
    )

    print(f"{log_prefix}Cross-checking the covariance against linear propagation", flush=True)
    linear_covariances = linear_covariance_check(
        case, options, dynamics, normalization, solution, ref_traj.steps
    )

    print(f"{log_prefix}Running {options.monte_carlo_samples} Monte Carlo samples", flush=True)
    monte_carlo_result = monte_carlo(case, options, dynamics, normalization, solution, ref_traj.steps)

    diagnostics = compute_diagnostics(
        case,
        options,
        ref_traj,
        solution,
        normalization,
        linear_covariances,
        monte_carlo_result,
        psi_inv,
    )
    print_summary(case, diagnostics, psi_inv, log_prefix)

    output_prefix = OUTPUT_DIR / case.test_case_id
    save_outputs(
        case, options, ref_traj, solution, normalization, linear_covariances,
        monte_carlo_result, diagnostics, psi_inv, output_prefix,
    )
    print(f"{log_prefix}Wrote {output_prefix.with_suffix('.npz')} and companions", flush=True)
    return diagnostics


def regenerate_outputs(test_case_id: str, options: Options | None = None) -> dict[str, float]:
    """Rebuild the summary, plots and archives from a saved solution.

    Re-runs only the post-processing -- the Monte Carlo, the linear covariance
    cross-check, the diagnostics and the figures -- reading the converged
    trajectory back from the `.npz`. Useful when a reporting or plotting defect is
    found after a long solve, since it avoids repeating the solve itself.
    """

    options = options or Options()
    case = CASE_REGISTRY[test_case_id]()
    log_prefix = f"[{test_case_id}] "
    psi_inv = psi_inverse(NU, options.violation_parameter)

    archive = np.load(OUTPUT_DIR / f"{case.test_case_id}.npz", allow_pickle=True)
    stored = dict(archive["diagnostics"].item())
    ref_traj = load_ref_traj(case, options)
    normalization = build_normalization(case)
    dynamics = Dynamics(case)

    if archive["steps_nd"].size != ref_traj.n_arcs:
        raise ValueError(
            f"Saved solution has {archive['steps_nd'].size} arcs but the current "
            f"mesh_stride={options.mesh_stride} gives {ref_traj.n_arcs}."
        )

    solution = SteeringSolution(
        means=np.asarray(archive["means"], dtype=float),
        open_loop=np.asarray(archive["open_loop_normalized"], dtype=float),
        gains=np.asarray(archive["gains_normalized"], dtype=float),
        radius=np.asarray(archive["radius_normalized"], dtype=float),
        covariances=np.asarray(archive["covariance_normalized"], dtype=float),
        control_covariances=np.asarray(archive["control_covariance_normalized"], dtype=float),
        objective=float(stored["objective_nd"]),
        diagnostics={
            "converged": float(stored["converged"]),
            "matching_defect_nd": float(stored["matching_defect_nd"]),
        },
    )

    print(f"{log_prefix}Regenerating outputs from the saved solution", flush=True)
    linear_covariances = linear_covariance_check(
        case, options, dynamics, normalization, solution, ref_traj.steps
    )
    monte_carlo_result = monte_carlo(case, options, dynamics, normalization, solution, ref_traj.steps)
    diagnostics = compute_diagnostics(
        case,
        options,
        ref_traj,
        solution,
        normalization,
        linear_covariances,
        monte_carlo_result,
        psi_inv,
    )
    print_summary(case, diagnostics, psi_inv, log_prefix)
    save_outputs(
        case, options, ref_traj, solution, normalization, linear_covariances,
        monte_carlo_result, diagnostics, psi_inv, OUTPUT_DIR / case.test_case_id,
    )
    return diagnostics


def _build_arc_functions(
    case: TestCase,
    options: Options,
    dynamics: Dynamics,
    normalization: Normalization,
    ref_traj: ReferenceTraj,
) -> list[casadi.Function]:
    """One arc function per *distinct* duration, reused across arcs.

    The DIRTRAN mesh is refined by bisection, so only a handful of distinct step
    sizes occur; sharing the functions keeps the expression graph small.
    """

    cache: dict[float, casadi.Function] = {}
    functions: list[casadi.Function] = []
    for k, step in enumerate(ref_traj.steps):
        key = float(np.round(step, 14))
        if key not in cache:
            cache[key] = build_arc_function(case, options, dynamics, normalization, float(step), len(cache))
        functions.append(cache[key])
    print(f"    {len(cache)} distinct arc durations across {ref_traj.steps.size} arcs", flush=True)
    return functions


def main() -> None:
    options = Options()
    for test_case_id in ("lyapunov_l1_to_l2", "halo_l2_to_halo_l1"):
        run_test_case(test_case_id, options)


if __name__ == "__main__":
    main()
