"""
Chance-constrained covariance steering for the linear double-integrator test case.

Problem data (dynamics, boundary distributions, path/control chance constraints,
cost weights) are taken verbatim from the reference. The transcription and solution
methodology, however, is the one implemented in ``cr3bp_covariance_steering.py``,
not the thesis's own factorized-covariance (block-matrix / L,V change-of-variables)
SDP. Concretely:

  * The mean trajectory is transcribed by multiple shooting (each node mean is a
    decision variable, tied to its neighbours by a dynamics equality), exactly as
    in the CR3BP script.
  * Two solve modes are provided for the covariance:

      Phase 1 ("single_shot"): the covariance is propagated forward, node to
      node, purely as a function of the mean/gain decision variables (never
      itself a decision variable), via an augmented Unscented Transform with
      kappa = 0 over z_k = [x_k; w_k]. The fresh process noise is propagated as
      part of every sigma point rather than added afterward as G G^T. Because
      the double integrator is exactly linear, the feedback policy is affine,
      and w_k is independent of x_k, this augmented UT reproduces the closed-form
      Gaussian covariance recursion exactly (no linearization error).

      Phase 2 ("multiple_shoot"): the covariance is additionally put on multiple
      shooting. A lower-triangular Cholesky factor L_k of each node covariance
      P_k = L_k L_k^T is introduced as a decision variable (guaranteeing P_k is
      PSD by construction, entirely analogous to how the mean nodes are decision
      variables tied by a matching/defect equality), and the one-step UT map is
      used to define the defect L_k L_k^T - P_predicted(L_{k-1}, ...) = 0 rather
      than substituting the propagated covariance directly into the next arc.

  * The control chance constraint is transcribed exactly as in the CR3BP script:
    a scaled control-covariance is formed from the UT sigma points, its spectral
    radius rho = sqrt(lambda_max) is bounded in closed form (2x2 here, so via the
    standard symmetric-2x2 eigenvalue formula rather than the CR3BP script's 3x3
    formula), and the chance constraint ||u_k|| <= u_max w.p. >= p_u becomes
    ``||v_k|| + Psi^{-1}_{n_u}(1 - p_u) * rho(Sigma^T_k) <= u_max``, with
    Psi^{-1}_{n_u} the same chi-quantile helper (`psi_inverse`) used in the CR3BP
    script. ||v_k|| itself is the epsilon-regularized norm
    ``sqrt(dot(v_k, v_k) + control_norm_epsilon**2)`` rather than an epigraph
    slack variable -- see the objective note below for why.
  * The terminal covariance constraint uses the identical Cholesky-residual
    Loewner-order encoding: with Dt the (diagonal) terminal covariance target,
    I - Dt^{-1/2} P_N Dt^{-1/2} = G G^T, G lower-triangular with a floored
    diagonal, and the strict lower-triangular + diagonal residual entries
    constrained to zero.
  * State path (position) chance constraints are not present in the CR3BP script
    (it has no path constraints), so they are added here using the thesis's own
    second-order-cone reformulation (Eq. 3.52): for row a_j^T x_k <= b_j at
    confidence >= p_x, the UT-propagated covariance gives the standard deviation
    sigma_y = ||sqrt(P_k) a_j||, and the deterministic surrogate is
    ``a_j^T x_bar_k + Phi^{-1}(p_x) * sigma_y <= b_j``.
  * The objective is the thesis's own composite cost, Eq. (4.8), verbatim and
    with NO quantile/chance-margin term added:
    ``J = dt * sum_k ( ||v_k|| + tr(Q_k P_k) + tr(R_k Sigma^T_k) )``.
    The CR3BP script's quantile term ``Psi^{-1}_{n_u}(1-p_u) * rho(Sigma^T_k)``
    is used only inside the control chance *constraint* above -- it is never
    charged into the cost, matching the thesis exactly rather than the CR3BP
    script's own pure quantile-cost formulation (Eq. 3.9/3.43). ||v_k|| is the
    same epsilon-regularized norm as the control chance constraint (and as
    solve_deterministic_nominal's objective), not an epigraph slack: this keeps
    the gradient of the control-effort term finite at v_k = 0 (coast arcs)
    without introducing an extra decision variable per arc.
  * Solved via SNOPT through CasADi's Opti stack, as in the CR3BP script.

Warm start: a deterministic nominal trajectory is solved first
(``solve_deterministic_nominal``) -- a quadratic minimum-energy multiple-shooting
OCP with no covariance and no chance constraints (the position path bounds are
enforced as hard inequalities), playing the same role as the deterministic
reference the CR3BP script loads from disk. ``build_stochastic_seed`` then turns that
nominal trajectory plus a TVLQR/Bryson-rule backward Riccati sweep into a full
warm start (mean, feedforward, gains, and -- by actually propagating that seed
policy through the real UT arc map -- consistent slack/terminal-margin values),
mirroring ``build_initial_guess`` in the CR3BP script.

Two entry points are exposed: ``solve_single_shot`` (Phase 1, reproduces Fig. 4.1)
and ``solve_multiple_shoot`` (Phase 2, covariance multiple shooting with a
Cholesky-factor decision variable at every node). Both are independent stochastic
solves of the *same* deterministic-nominal-derived seed -- Phase 2 is not seeded
from Phase 1's converged solution, only from the shared nominal trajectory (its
extra per-node Cholesky-factor variables are seeded from that same seed policy's
propagated covariances). ``main`` solves the nominal trajectory once, then runs
both phases from it.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import casadi
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, norm

from deterministic_cr3bp import collocation_coefficients

NX = 4        # state: [x1, x2, x3, x4] = [px, py, vx, vy]
NU = 2        # control: [ux, uy]
NW = 4        # fresh standard-normal process-noise dimension per discrete arc
N_AUGMENTED = NX + NW
N_SIGMA = 2 * N_AUGMENTED + 1

OUTPUT_DIR = Path("output/double_integrator_covariance_steering")


# --------------------------------------------------------------------------- #
# Problem data (Babapour thesis, Section 4.1.1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Options:
    dt: float = 0.25
    tf: float = 5.0
    n_arcs: int = 20

    # Boundary distributions.
    x0_mean: np.ndarray = field(
        default_factory=lambda: np.array([2.0, 4.0, 3.0, 2.0])
    )
    x0_covariance: np.ndarray = field(
        default_factory=lambda: np.diag([0.1, 0.1, 0.02, 0.02])
    )
    xf_mean: np.ndarray = field(
        default_factory=lambda: np.array([8.0, 2.0, 0.0, 0.0])
    )
    xf_covariance: np.ndarray = field(
        default_factory=lambda: np.diag([0.06, 0.06, 0.006, 0.006])
    )

    # Path (position) chance constraints, a_j^T x <= b_j.
    a1: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 0.0, 0.0]))
    b1: float = 12.75
    a2: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.1, 0.0, 0.0]))
    b2: float = 8.75
    path_confidence: float = 0.9973   # p_x

    # Control chance constraint.
    u_max: float = 2.0
    control_confidence: float = 0.9973   # p_u

    # Running cost weights (Eq. 4.9).
    Q: np.ndarray = field(default_factory=lambda: 0.01 * np.eye(NX))
    R: np.ndarray = field(default_factory=lambda: np.eye(NU))

    # Unscented transform.
    scaling_parameter: float = 0.0   # kappa
    cholesky_jitter: float = 1e-11
    # Smooth the repeated-eigenvalue corner in lambda_max(SigmaT).  This has
    # covariance units and is deliberately separate from the standard-deviation
    # floor used by the outer square root.  A value of 1e-4 is appropriate for
    # this nondimensional benchmark (representative control-covariance
    # eigenvalues are O(1e-2)); its conservatism is reported after each solve.
    spectral_eigenvalue_smoothing: float = 1e-4
    spectral_radius_floor: float = 1e-9
    terminal_margin_floor: float = 1e-3

    # TVLQR warm start (Bryson's rule), mirroring seed_gains in the CR3BP script.
    warm_start_gains: bool = True
    riccati_terminal_weight: float = 1.0e3
    riccati_control_weight: float = 1.0

    # Deterministic nominal solve (Radau collocation, as in deterministic_cr3bp.py).
    radau_degree: int = 3
    # Explicit RK4 substeps for sigma-point propagation, matching the CR3BP arc
    # construction. One substep is exact for this constant-control double
    # integrator because its continuous-time state matrix is nilpotent.
    integrator_substeps: int = 1
    control_norm_epsilon: float = 1e-6

    solver: str = "snopt"
    major_max_iter: int = 2000
    minor_max_iter: int = 100 * 1500
    # The spectral-radius model is intentionally smoothed at 1e-4. Requiring
    # 1e-8 stationarity from the resulting large, nonconvex NLP caused SNOPT to
    # report EXIT 40 after reaching a constraint-feasible point with a dual
    # residual of only a few 1e-6.  A 1e-5 major optimality tolerance resolves
    # that scale mismatch without relaxing the feasibility requirements.
    major_optimality_tol: float = 1e-7
    major_feasibility_tol: float = 1e-9
    minor_feasibility_tol: float = 1e-9
    elastic_weight: float = 1e8
    print_level: int = 1

    monte_carlo_samples: int = 5000
    monte_carlo_seed: int = 42


def psi_inverse(dimension: int, beta: float) -> float:
    """Psi_d^-1(beta) = sqrt(Phi_d^-1(1 - beta)) with Phi_d the chi-squared CDF.

    Identical helper to the one in cr3bp_covariance_steering.py.
    """

    return float(np.sqrt(chi2.ppf(1.0 - beta, dimension)))


def unscented_weights(kappa: float, dimension: int) -> np.ndarray:
    """Sigma-point weights c_j; c_0 vanishes for kappa = 0."""

    weights = np.full(2 * dimension + 1, 1.0 / (2.0 * (dimension + kappa)))
    weights[0] = kappa / (dimension + kappa)
    return weights


# --------------------------------------------------------------------------- #
# Discrete-time dynamics (exact, linear, zero-order hold -- Eq. 4.2)
# --------------------------------------------------------------------------- #


def system_matrices(options: Options) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            [0.5 * dt ** 2, 0.0],
            [0.0, 0.5 * dt ** 2],
            [dt, 0.0],
            [0.0, dt],
        ]
    )
    G = 0.01 * np.eye(NX)
    return A, B, G


def continuous_time_matrices() -> tuple[np.ndarray, np.ndarray]:
    """Continuous-time double-integrator dynamics xdot = Ac x + Bc u (pdot=v, vdot=u).

    Distinct from ``system_matrices``, whose A, B are the exact discrete-time
    zero-order-hold matrices (already integrated over one full ``dt``, per the
    thesis's Eq. 4.2). Both the explicit RK4 UT arc map and the Radau collocation
    nominal solve need the instantaneous rate of change instead; feeding the
    discrete A, B into either integrator would integrate an already-integrated
    step a second time.
    """

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


def rk4_step(states, controls, Ac, Bc, step: float, substeps: int):
    """Fixed-step RK4 with zero-order-held control, as in the CR3BP propagator."""

    if substeps < 1:
        raise ValueError("integrator_substeps must be at least one")
    propagated = states
    substep = step / substeps
    for _ in range(substeps):
        k1 = Ac @ propagated + Bc @ controls
        k2 = Ac @ (propagated + 0.5 * substep * k1) + Bc @ controls
        k3 = Ac @ (propagated + 0.5 * substep * k2) + Bc @ controls
        k4 = Ac @ (propagated + substep * k3) + Bc @ controls
        propagated = propagated + (substep / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return propagated


def cholesky_lower_triangular_matrix(matrix, dimension: int, pivot_floor: float = 0.0):
    """Lower-triangular Cholesky factor built from scalar operations (MX-safe).

    Identical construction to the one in cr3bp_covariance_steering.py.
    """

    factor: list[list] = [[casadi.MX(0.0)] * dimension for _ in range(dimension)]
    for row in range(dimension):
        for column in range(row + 1):
            total = matrix[row, column]
            for inner in range(column):
                total = total - factor[row][inner] * factor[column][inner]
            if row == column:
                factor[row][column] = casadi.sqrt(casadi.fmax(total, pivot_floor))
            else:
                factor[row][column] = total / factor[column][column]
    return casadi.vertcat(*[casadi.horzcat(*row) for row in factor])


def psqrt_spectral_radius_2x2(
    matrix: np.ndarray,
    eigenvalue_smoothing: float,
    radius_floor: float,
) -> float:
    """Numerical counterpart of :func:`symbolic_psqrt_spectral_radius_2x2`.

    ``eigenvalue_smoothing`` regularizes the norm that separates the two
    eigenvalues and therefore has the same units as the matrix entries.
    ``radius_floor`` regularizes the final square root and has the same units as
    the returned standard deviation.  Keeping these scales separate avoids the
    previous dimensional ambiguity in which one number was used at both levels.
    """

    array = np.asarray(matrix, dtype=float)
    symmetric = 0.5 * (array + array.T)
    a11, a22, a12 = symmetric[0, 0], symmetric[1, 1], symmetric[0, 1]
    mean = 0.5 * (a11 + a22)
    radicand = 0.25 * (a11 - a22) ** 2 + a12 ** 2
    half_spread = np.sqrt(radicand + eigenvalue_smoothing ** 2)
    lambda_max = mean + half_spread
    return float(np.sqrt(max(lambda_max, 0.0) + radius_floor ** 2))


def symbolic_psqrt_spectral_radius_2x2(
    matrix,
    eigenvalue_smoothing: float,
    radius_floor: float,
):
    """Smooth upper approximation of ``sqrt(lambda_max(A))`` for symmetric 2x2 ``A``.

    lambda_max = 0.5*(a11+a22) + sqrt( 0.25*(a11-a22)^2 + a12^2 ), floored on the
    eigenvalue-gap radicand so derivatives remain finite when the eigenvalues
    coincide.  The smoothing is an upper approximation and is consequently
    conservative in the control chance constraint.
    """

    a11, a22, a12 = matrix[0, 0], matrix[1, 1], matrix[0, 1]
    mean = 0.5 * (a11 + a22)
    radicand = 0.25 * (a11 - a22) ** 2 + a12 ** 2
    half_spread = casadi.sqrt(radicand + eigenvalue_smoothing ** 2)
    lambda_max = mean + half_spread
    return casadi.sqrt(casadi.fmax(lambda_max, 0.0) + radius_floor ** 2)


def smoothed_control_radii(
    control_covariances: np.ndarray, options: Options
) -> tuple[np.ndarray, dict[str, float]]:
    """Return constraint-consistent radii and smoothing-bias diagnostics."""

    exact_eigenvalues = np.linalg.eigvalsh(control_covariances)[..., -1]
    exact_radii = np.sqrt(np.clip(exact_eigenvalues, 0.0, None))
    radii = np.array(
        [
            psqrt_spectral_radius_2x2(
                covariance,
                options.spectral_eigenvalue_smoothing,
                options.spectral_radius_floor,
            )
            for covariance in control_covariances
        ]
    )
    bias = radii - exact_radii
    peak_exact_radius = max(float(np.max(exact_radii)), options.spectral_radius_floor)
    psi_inv_u = psi_inverse(NU, 1.0 - options.control_confidence)
    metrics = {
        "max_radius_smoothing_bias": float(np.max(bias)),
        "relative_radius_smoothing_bias": float(np.max(bias) / peak_exact_radius),
        "max_control_margin_smoothing_bias": float(psi_inv_u * np.max(bias)),
    }
    return radii, metrics


def get_arc_function(options: Options, G: np.ndarray) -> casadi.Function:
    """One-step moment map via a state/process-noise augmented UT.

    Sigma points represent z_k = [x_k; w_k] with covariance blkdiag(P_k, I),
    where w_k is fresh and independent at every arc. Feedback uses only the
    state component, u_k = v_k + K_k(x_k-mu_k), and the discrete noise kick is
    applied after deterministic RK4 propagation. Consequently the affine test
    case recovers P_{k+1}=(A+BK)P_k(A+BK)^T+G G^T without adding G G^T
    separately. RK4 is exact here for zero-order-held control because the
    double-integrator state matrix is nilpotent.
    """

    noise_matrix = np.asarray(G, dtype=float)
    if noise_matrix.ndim != 2 or noise_matrix.shape[0] != NX:
        raise ValueError(f"G must have shape ({NX}, n_w)")
    noise_dimension = noise_matrix.shape[1]
    augmented_dimension = NX + noise_dimension
    sigma_scale = augmented_dimension + options.scaling_parameter
    if sigma_scale <= 0.0:
        raise ValueError("augmented dimension + scaling_parameter must be positive")
    n_sigma = 2 * augmented_dimension + 1
    weights = unscented_weights(
        options.scaling_parameter,
        augmented_dimension,
    )

    Ac, Bc = continuous_time_matrices()
    Ac_cs = casadi.DM(Ac)
    Bc_cs = casadi.DM(Bc)
    G_cs = casadi.DM(noise_matrix)

    mean = casadi.MX.sym("mu", NX)
    covariance = casadi.MX.sym("P", NX, NX)
    feedforward = casadi.MX.sym("v", NU)
    gain = casadi.MX.sym("K", NU, NX)

    # The augmented covariance is blkdiag(P, I). Its block diagonal structure
    # lets us factor only the symbolic state block; the noise block has the
    # exact square root sqrt(n_aug+kappa) I.
    state_spread = cholesky_lower_triangular_matrix(
        sigma_scale * covariance
        + options.cholesky_jitter * casadi.DM.eye(NX),
        NX,
        pivot_floor=options.cholesky_jitter,
    )
    zero_noise = casadi.DM.zeros(noise_dimension, 1)
    zero_state = casadi.DM.zeros(NX, 1)
    state_deviations = [state_spread[:, column] for column in range(NX)]
    noise_deviations = [
        np.sqrt(sigma_scale) * casadi.DM.eye(noise_dimension)[:, column]
        for column in range(noise_dimension)
    ]

    state_sigma_points = [mean]
    noise_sigma_points = [zero_noise]
    for sign in (1.0, -1.0):
        for deviation in state_deviations:
            state_sigma_points.append(mean + sign * deviation)
            noise_sigma_points.append(zero_noise)
        for deviation in noise_deviations:
            state_sigma_points.append(mean + zero_state)
            noise_sigma_points.append(sign * deviation)

    controls = [
        feedforward + gain @ (state_sigma_points[j] - mean)
        for j in range(n_sigma)
    ]

    propagated = casadi.horzcat(
        *[
            rk4_step(
                state_sigma_points[j],
                controls[j],
                Ac_cs,
                Bc_cs,
                options.dt,
                options.integrator_substeps,
            )
            + G_cs @ noise_sigma_points[j]
            for j in range(n_sigma)
        ]
    )

    mean_next = sum(weights[j] * propagated[:, j] for j in range(n_sigma))
    covariance_next = casadi.MX.zeros(NX, NX)
    for j in range(n_sigma):
        residual = propagated[:, j] - mean_next
        covariance_next = covariance_next + weights[j] * residual @ residual.T
    covariance_next = 0.5 * (covariance_next + covariance_next.T)

    control_mean = sum(weights[j] * controls[j] for j in range(n_sigma))
    control_covariance = casadi.MX.zeros(NU, NU)
    for j in range(n_sigma):
        deviation = controls[j] - control_mean
        control_covariance = (
            control_covariance
            + weights[j] * deviation @ deviation.T
        )
    control_covariance = 0.5 * (control_covariance + control_covariance.T)

    return casadi.Function(
        "arc",
        [mean, covariance, feedforward, gain],
        [mean_next, covariance_next, control_covariance],
        ["mu", "P", "v", "K"],
        ["mu_next", "P_next", "SigmaT"],
    )


# --------------------------------------------------------------------------- #
# TVLQR gain warm start (Bryson's rule), mirroring seed_gains/tvlqr_gains
# --------------------------------------------------------------------------- #


def tvlqr_gains(
    options: Options, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Backward Riccati recursion with Bryson-rule weights, time-invariant A,B."""

    Q = np.eye(NX) / (np.diag(options.x0_covariance).mean())
    R = options.riccati_control_weight * np.eye(NU)
    P = options.riccati_terminal_weight * np.eye(NX)
    gains = np.zeros((options.n_arcs, NU, NX))
    for k in reversed(range(options.n_arcs)):
        gain = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
        gains[k] = -gain
        closed_loop = A - B @ gain
        P = Q + gain.T @ R @ gain + closed_loop.T @ P @ closed_loop
        P = 0.5 * (P + P.T)
    return gains


# --------------------------------------------------------------------------- #
# Deterministic nominal trajectory (no covariance, no chance constraints)
# --------------------------------------------------------------------------- #


@dataclass
class NominalTrajectory:
    means: np.ndarray          # (NX, n_arcs+1)
    feedforward: np.ndarray    # (NU, n_arcs)
    converged: bool


def solve_deterministic_nominal(
    options: Options, A: np.ndarray, B: np.ndarray
) -> NominalTrajectory:
    """Deterministic minimum-energy transfer x0 -> xN, no covariance/chance terms.

    Plays the same role as the deterministic DIRTRAN solution the CR3BP script
    loads from disk (``load_ref_traj``) to seed its iCS problem, and is
    transcribed with the exact same Radau collocation method as
    ``deterministic_cr3bp.solve_ocp``: each interval k of duration
    ``h = dt`` carries ``radau_degree`` internal collocation-point state
    variables (the Radau roots on (0, 1], via the shared
    ``collocation_coefficients`` helper), the piecewise-constant control u_k is
    held over the whole interval, and the dynamics are enforced at every
    collocation point j by matching the interpolating polynomial's derivative
    to h * f(x_j, u_k) through the differentiation matrix ``c_matrix`` -- the
    same ``polynomial_derivative == h * f_j`` constraint pattern used there,
    with the double integrator's linear f(x, u) = Ac x + Bc u in place of the
    CR3BP equations of motion. Ac, Bc are the *continuous-time* dynamics
    matrices from ``continuous_time_matrices`` (xdot = Ac x + Bc u), not the
    discrete-time A, B this function receives -- those are the exact
    zero-order-hold matrices from ``system_matrices`` (already integrated over
    one full dt), so feeding them into a collocation constraint would
    integrate the already-integrated step a second time. A, B are used here
    only to build the straight-line initial guess via least squares, matching
    what the stochastic phases expect as a rough seed. Boundary conditions and
    the two position half-spaces are hard (non-probabilistic) constraints,
    exactly as in deterministic_cr3bp.py. Its solution is the nominal
    trajectory both stochastic phases are seeded from.

    The warm-start objective is quadratic control energy,
    ``sum_k dt * dot(u_k, u_k)``. Unlike a sum of control norms, it is smooth at
    zero control and distributes effort over the transfer instead of promoting
    coast arcs separated by impulse-like control concentrations. The stochastic
    phases retain their own thesis objective; this change affects only the
    deterministic trajectory used to initialise them.
    """

    n_arcs = options.n_arcs
    degree = options.radau_degree
    tau_root, c_matrix, _, _ = collocation_coefficients(degree)

    opti = casadi.Opti()
    means = opti.variable(NX, n_arcs + 1)
    feedforward = opti.variable(NU, n_arcs)
    stage_vars: list[tuple[int, int, casadi.MX]] = []

    opti.subject_to(means[:, 0] == options.x0_mean)
    opti.subject_to(means[:, n_arcs] == options.xf_mean)

    Ac, Bc = continuous_time_matrices()
    Ac_cs = casadi.DM(Ac)
    Bc_cs = casadi.DM(Bc)
    objective = 0.0
    for k in range(n_arcs):
        interval_states = [means[:, k]]
        for j in range(1, degree):
            x_stage = opti.variable(NX)
            stage_vars.append((k, j, x_stage))
            interval_states.append(x_stage)
        interval_states.append(means[:, k + 1])

        for j in range(1, degree + 1):
            x_j = interval_states[j]
            polynomial_derivative = c_matrix[0, j] * interval_states[0]
            for r in range(1, degree + 1):
                polynomial_derivative += c_matrix[r, j] * interval_states[r]

            f_j = Ac_cs @ x_j + Bc_cs @ feedforward[:, k]
            opti.subject_to(polynomial_derivative == options.dt * f_j)

        for a_vec, b_val in ((options.a1, options.b1), (options.a2, options.b2)):
            opti.subject_to(casadi.DM(a_vec).T @ means[:, k] <= b_val)

        control_energy = casadi.dot(feedforward[:, k], feedforward[:, k])
        objective = objective + options.dt * control_energy

    opti.minimize(objective)

    means_seed = np.linspace(options.x0_mean, options.xf_mean, n_arcs + 1).T
    feedforward_seed = np.zeros((NU, n_arcs))
    for k in range(n_arcs):
        feedforward_seed[:, k] = np.linalg.lstsq(
            B, means_seed[:, k + 1] - A @ means_seed[:, k], rcond=None
        )[0]
    opti.set_initial(means, means_seed)
    opti.set_initial(feedforward, feedforward_seed)
    for k, j, x_stage in stage_vars:
        theta = float(tau_root[j])
        stage_guess = (1.0 - theta) * means_seed[:, k] + theta * means_seed[:, k + 1]
        opti.set_initial(x_stage, stage_guess)

    _configure_snopt(opti, options)

    try:
        solution = run_snopt_solver(opti, "double-integrator-nominal")
        converged = True
    except RuntimeError as error:
        print(f"  SNOPT did not converge ({error}); returning the last iterate.", flush=True)
        solution = opti.debug
        converged = False

    return NominalTrajectory(
        means=np.asarray(solution.value(means)),
        feedforward=np.asarray(solution.value(feedforward)).reshape(NU, n_arcs),
        converged=converged,
    )


# --------------------------------------------------------------------------- #
# Shared warm-start recipe for both stochastic phases
# --------------------------------------------------------------------------- #


@dataclass
class StochasticSeed:
    means: np.ndarray
    feedforward: np.ndarray
    gains: np.ndarray
    covariances: np.ndarray    # (n_arcs+1, NX, NX), from propagating the seed policy
    margin: np.ndarray         # lower-triangular entries of the terminal G, row-major


def build_stochastic_seed(
    options: Options,
    A: np.ndarray,
    B: np.ndarray,
    arc_function: casadi.Function,
    nominal: NominalTrajectory,
    lower_indices: list[tuple[int, int]],
) -> StochasticSeed:
    """Warm start shared by solve_single_shot and solve_multiple_shoot.

    Both stochastic phases start from the very same deterministic nominal
    trajectory and TVLQR gains, then make the terminal-margin variable
    consistent with that seed by actually propagating it through the real UT
    arc map -- mirroring build_initial_guess in the CR3BP script. Neither
    phase is seeded from the other's converged solution. The control-norm
    epigraph slack has no counterpart here since both phases now use the
    epsilon-regularized ||v_k|| directly, with no separate decision variable
    to seed.
    """

    n_arcs = options.n_arcs
    gains_seed = (
        tvlqr_gains(options, A, B)
        if options.warm_start_gains
        else np.zeros((n_arcs, NU, NX))
    )

    _, seed_covariances, _ = propagate_moments(
        arc_function, nominal.means, nominal.feedforward, gains_seed, options.x0_covariance
    )

    inv_std = 1.0 / np.sqrt(np.diag(options.xf_covariance))
    scaled = np.outer(inv_std, inv_std) * seed_covariances[-1]
    slack_seed = np.eye(NX) - scaled
    eigvals, eigvecs = np.linalg.eigh(0.5 * (slack_seed + slack_seed.T))
    slack_psd = eigvecs @ np.diag(np.clip(eigvals, 0.0, None)) @ eigvecs.T
    G_seed = np.linalg.cholesky(slack_psd + options.terminal_margin_floor ** 2 * np.eye(NX))
    margin_seed = np.array([G_seed[row, col] for row, col in lower_indices])

    return StochasticSeed(
        means=nominal.means,
        feedforward=nominal.feedforward,
        gains=gains_seed,
        covariances=seed_covariances,
        margin=margin_seed,
    )


# --------------------------------------------------------------------------- #
# Phase 1: single-shot covariance, multiple-shooting mean
# --------------------------------------------------------------------------- #


@dataclass
class RobustSolution:
    means: np.ndarray                  # (NX, n_arcs+1)
    feedforward: np.ndarray            # (NU, n_arcs)
    gains: np.ndarray                  # (n_arcs, NU, NX)
    covariances: np.ndarray            # (n_arcs+1, NX, NX)
    control_covariances: np.ndarray    # (n_arcs, NU, NU)
    radius: np.ndarray                 # (n_arcs,)
    objective: float
    diagnostics: dict[str, float | str] = field(default_factory=dict)


def propagate_moments(
    arc_function: casadi.Function,
    means: np.ndarray,
    feedforward: np.ndarray,
    gains: np.ndarray,
    initial_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_arcs = feedforward.shape[1]
    covariances = np.empty((n_arcs + 1, NX, NX))
    control_covariances = np.empty((n_arcs, NU, NU))
    propagated_means = np.empty((NX, n_arcs + 1))
    covariances[0] = initial_covariance
    propagated_means[:, 0] = means[:, 0]
    for k in range(n_arcs):
        mean_next, covariance_next, control_covariance = arc_function(
            means[:, k], covariances[k], feedforward[:, k], gains[k]
        )
        propagated_means[:, k + 1] = np.asarray(mean_next.full()).ravel()
        covariances[k + 1] = np.asarray(covariance_next.full())
        control_covariances[k] = np.asarray(control_covariance.full())
    return propagated_means, covariances, control_covariances


def run_snopt_solver(opti: casadi.Opti, tag: str) -> casadi.OptiSol:
    """Solve from a unique directory so SNOPT gets a fresh output file (Windows-safe)."""

    work_dir = Path(tempfile.mkdtemp(prefix=f"casadi-snopt-{tag}-"))
    original = Path.cwd()
    try:
        os.chdir(work_dir)
        return opti.solve()
    finally:
        os.chdir(original)


def solver_diagnostics(
    opti: casadi.Opti,
    solution,
    options: Options,
    solver_accepted: bool,
) -> dict[str, float | str]:
    """Report solver termination separately from returned-point feasibility.

    CasADi raises whenever SNOPT does not return a success code, including EXIT
    40 at an essentially feasible point.  That distinction matters for Phase 1:
    feasibility is useful for diagnosing the transcription, but it is not an
    optimality certificate and must not be relabelled as solver convergence.
    """

    values = np.asarray(solution.value(opti.g), dtype=float).ravel()
    lower = np.asarray(solution.value(opti.lbg), dtype=float).ravel()
    upper = np.asarray(solution.value(opti.ubg), dtype=float).ravel()
    if np.all(np.isfinite(values)):
        lower_violation = np.maximum(lower - values, 0.0)
        upper_violation = np.maximum(values - upper, 0.0)
        max_violation = float(max(np.max(lower_violation), np.max(upper_violation)))
    else:
        max_violation = float("inf")

    # Allow a small reporting guard above SNOPT's requested feasibility
    # tolerance for independent post-solve evaluation and roundoff.
    acceptance_tolerance = max(10.0 * options.major_feasibility_tol, 1e-8)
    stats = opti.stats()
    status = str(stats.get("secondary_return_status") or stats.get("return_status", "unknown"))
    optimality_satisfied = status.strip().lower() == "optimality conditions satisfied"
    feasible = max_violation <= acceptance_tolerance
    return {
        "converged": float(solver_accepted and optimality_satisfied and feasible),
        "solver_accepted": float(solver_accepted),
        "optimality_satisfied": float(optimality_satisfied),
        "feasible": float(feasible),
        "max_constraint_violation": max_violation,
        "solver_status": status,
    }


def _configure_snopt(opti: casadi.Opti, options: Options) -> None:
    snopt_options = {
        "Major iterations limit": options.major_max_iter,
        "Minor iterations limit": max(500, options.minor_max_iter),
        "Iterations limit": options.minor_max_iter,
        "Major optimality tolerance": f"{options.major_optimality_tol:.13g}",
        "Major feasibility tolerance": f"{options.major_feasibility_tol:.13g}",
        "Minor feasibility tolerance": f"{options.minor_feasibility_tol:.13g}",
        # Unlike the CR3BP script (which only notes this as an untested idea), a
        # much larger elastic weight is set here because it was observed
        # empirically to be load-bearing on this problem: with SNOPT's default
        # (1e4) the penalty parameter climbs into the 1e8-1e9 range over ~2000
        # major iterations without reaching the optimality tolerance, because a
        # too-small elastic weight lets the QP subproblem trade feasibility for
        # optimality too cheaply, so the SQP path crawls along the boundary of
        # the terminal-covariance/control-chance-constraint manifold instead of
        # converging. A large fixed elastic weight keeps constraint violations
        # expensive throughout, which reaches the optimality tolerance in a few
        # hundred major iterations instead.
        "Elastic weight": f"{options.elastic_weight:.13g}",
        "Print file": 0,
        "Summary file": 6 if options.print_level else 0,
        "Major print level": options.print_level,
        "Minor print level": 0,
    }
    opti.solver("snopt", {"expand": False, "print_time": True}, snopt_options)


def solve_single_shot(
    options: Options, A: np.ndarray, B: np.ndarray, G: np.ndarray, nominal: NominalTrajectory
) -> RobustSolution:
    """Phase 1: mean on multiple shooting, covariance single-shot forward (UT).

    Seeded by build_stochastic_seed from the deterministic nominal trajectory
    (solve_deterministic_nominal) -- not from any other stochastic solve.
    """

    arc_function = get_arc_function(options, G)
    psi_inv_u = psi_inverse(NU, 1.0 - options.control_confidence)
    z_path = float(norm.ppf(options.path_confidence))
    n_arcs = options.n_arcs
    lower_indices = [(row, col) for row in range(NX) for col in range(row + 1)]

    seed = build_stochastic_seed(options, A, B, arc_function, nominal, lower_indices)

    opti = casadi.Opti()
    means = opti.variable(NX, n_arcs + 1)
    feedforward = opti.variable(NU, n_arcs)
    gains = [opti.variable(NU, NX) for _ in range(n_arcs)]

    opti.subject_to(means[:, 0] == options.x0_mean)

    covariance = casadi.DM(options.x0_covariance)
    objective = 0.0
    quadratic_penalty = 0.0
    Q_cs = casadi.DM(options.Q)
    R_cs = casadi.DM(options.R)

    for k in range(n_arcs):
        mean_next, covariance_next, control_covariance = arc_function(
            means[:, k], covariance, feedforward[:, k], gains[k]
        )
        opti.subject_to(means[:, k + 1] == mean_next)

        # Epsilon-regularized control norm (matching solve_deterministic_nominal)
        # in place of the epigraph slack variable: no separate decision variable,
        # ||v_k|| is replaced everywhere -- objective and chance constraint alike
        # -- by sqrt(dot(v_k,v_k) + eps^2), which has a finite, well-defined
        # gradient at v_k = 0 (coast arcs) instead of the epigraph's kink there.
        control_norm = casadi.sqrt(
            casadi.dot(feedforward[:, k], feedforward[:, k])
            + options.control_norm_epsilon ** 2
        )
        radius = symbolic_psqrt_spectral_radius_2x2(
            control_covariance,
            options.spectral_eigenvalue_smoothing,
            options.spectral_radius_floor,
        )
        opti.subject_to(control_norm + psi_inv_u * radius <= options.u_max)

        # Two position path chance constraints (SOCP surrogate, thesis Eq. 3.52).
        for a_vec, b_val in ((options.a1, options.b1), (options.a2, options.b2)):
            a_cs = casadi.DM(a_vec)
            sigma_y = casadi.sqrt(
                casadi.fmax(a_cs.T @ covariance @ a_cs, 0.0) + options.spectral_radius_floor ** 2
            )
            opti.subject_to(a_cs.T @ means[:, k] + z_path * sigma_y <= b_val)

        # Thesis Eq. 4.8 composite cost: ||v_k|| plus quadratic covariance-trace
        # penalties, with NO quantile/chance-margin term (psi_inv_u * radius) in
        # the objective. That term is used only in the control chance constraint
        # above (control_norm + psi_inv_u*radius <= u_max), never charged into
        # the cost.
        objective = objective + options.dt * control_norm
        quadratic_penalty = quadratic_penalty + options.dt * (
            casadi.trace(Q_cs @ covariance) + casadi.trace(R_cs @ control_covariance)
        )

        covariance = covariance_next

    opti.subject_to(means[:, n_arcs] == options.xf_mean)

    inverse_target_std = casadi.DM(1.0 / np.sqrt(np.diag(options.xf_covariance)))
    scaled_terminal = (
        casadi.diag(inverse_target_std) @ covariance @ casadi.diag(inverse_target_std)
    )
    terminal_slack = casadi.MX.eye(NX) - scaled_terminal

    n_margin = NX * (NX + 1) // 2
    margin = opti.variable(n_margin)
    margin_matrix = [[casadi.MX(0.0)] * NX for _ in range(NX)]
    for (row, col), entry in zip(lower_indices, casadi.vertsplit(margin)):
        margin_matrix[row][col] = entry
    Gmat = casadi.vertcat(*[casadi.horzcat(*row) for row in margin_matrix])
    opti.subject_to(casadi.diag(Gmat) >= options.terminal_margin_floor)
    residual = terminal_slack - Gmat @ Gmat.T
    opti.subject_to(
        casadi.vertcat(*[residual[row, col] for row, col in lower_indices]) == 0.0
    )

    total_objective = objective + quadratic_penalty
    opti.minimize(total_objective)

    opti.set_initial(means, seed.means)
    opti.set_initial(feedforward, seed.feedforward)
    for k in range(n_arcs):
        opti.set_initial(gains[k], seed.gains[k])
    opti.set_initial(margin, seed.margin)

    _configure_snopt(opti, options)

    try:
        solution = run_snopt_solver(opti, "double-integrator-phase1")
        solver_accepted = True
    except RuntimeError as error:
        print(f"  SNOPT did not converge ({error}); returning the last iterate.", flush=True)
        solution = opti.debug
        solver_accepted = False

    solve_metrics = solver_diagnostics(opti, solution, options, solver_accepted)

    solved_means = np.asarray(solution.value(means))
    solved_feedforward = np.asarray(solution.value(feedforward)).reshape(NU, n_arcs)
    solved_gains = np.stack(
        [np.asarray(solution.value(g)).reshape(NU, NX) for g in gains]
    )

    propagated_means, covariances, control_covariances = propagate_moments(
        arc_function, solved_means, solved_feedforward, solved_gains, options.x0_covariance
    )
    matching_defect = float(np.max(np.abs(propagated_means[:, 1:] - solved_means[:, 1:])))
    solved_radius, smoothing_metrics = smoothed_control_radii(control_covariances, options)

    return RobustSolution(
        means=solved_means,
        feedforward=solved_feedforward,
        gains=solved_gains,
        covariances=covariances,
        control_covariances=control_covariances,
        radius=solved_radius,
        objective=float(solution.value(total_objective)),
        diagnostics={
            **solve_metrics,
            "matching_defect": matching_defect,
            **smoothing_metrics,
        },
    )


# --------------------------------------------------------------------------- #
# Phase 2: covariance multiple shooting via a Cholesky-factor decision variable
# --------------------------------------------------------------------------- #


def solve_multiple_shoot(
    options: Options, A: np.ndarray, B: np.ndarray, G: np.ndarray, nominal: NominalTrajectory
) -> RobustSolution:
    """Phase 2: mean AND covariance on multiple shooting.

    Every node covariance P_k is represented by a lower-triangular Cholesky
    factor L_k (P_k = L_k L_k^T), itself a decision variable -- so P_k is PSD by
    construction at every iterate, exactly the same guarantee multiple shooting
    on the mean gives for dynamic feasibility. The one-step UT arc map is used to
    predict P_{k+1} from (mu_k, L_k L_k^T, v_k, K_k); the node value L_{k+1}
    L_{k+1}^T is then tied to that prediction by a defect equality
    (analogous to means[:, k+1] == mean_next), rather than substituting the
    propagated covariance directly into the next arc's input as Phase 1 does.

    Seeded by build_stochastic_seed from the same deterministic nominal
    trajectory as Phase 1 (solve_single_shot) -- not from Phase 1's converged
    solution. The two phases are independent solves of the same seed.
    """

    arc_function = get_arc_function(options, G)
    psi_inv_u = psi_inverse(NU, 1.0 - options.control_confidence)
    z_path = float(norm.ppf(options.path_confidence))
    n_arcs = options.n_arcs
    lower_indices = [(row, col) for row in range(NX) for col in range(row + 1)]

    seed = build_stochastic_seed(options, A, B, arc_function, nominal, lower_indices)

    def cholesky_seed(matrix: np.ndarray) -> np.ndarray:
        eigvals, eigvecs = np.linalg.eigh(0.5 * (matrix + matrix.T))
        psd = eigvecs @ np.diag(np.clip(eigvals, 0.0, None)) @ eigvecs.T
        return np.linalg.cholesky(psd + options.cholesky_jitter * np.eye(NX))

    def unpack_lower(vec) -> object:
        """Assemble a full matrix from its NX*(NX+1)/2 lower-triangular entries."""

        rows = [[casadi.MX(0.0)] * NX for _ in range(NX)]
        for (row, col), entry in zip(lower_indices, casadi.vertsplit(vec)):
            rows[row][col] = entry
        return casadi.vertcat(*[casadi.horzcat(*row) for row in rows])

    opti = casadi.Opti()
    means = opti.variable(NX, n_arcs + 1)
    feedforward = opti.variable(NU, n_arcs)
    gains = [opti.variable(NU, NX) for _ in range(n_arcs)]
    n_chol = NX * (NX + 1) // 2
    chol_entries = [opti.variable(n_chol) for _ in range(n_arcs + 1)]

    opti.subject_to(means[:, 0] == options.x0_mean)
    for entries in chol_entries:
        diag_positions = [i for i, (row, col) in enumerate(lower_indices) if row == col]
        opti.subject_to(entries[diag_positions] >= options.cholesky_jitter)

    L0 = unpack_lower(chol_entries[0])
    opti.subject_to(
        casadi.vertcat(*[(L0 @ L0.T - casadi.DM(options.x0_covariance))[r, c] for r, c in lower_indices])
        == 0.0
    )

    objective = 0.0
    quadratic_penalty = 0.0
    Q_cs = casadi.DM(options.Q)
    R_cs = casadi.DM(options.R)

    node_covariances = []
    for k in range(n_arcs + 1):
        L_k = unpack_lower(chol_entries[k])
        node_covariances.append(L_k @ L_k.T)

    for k in range(n_arcs):
        covariance_k = node_covariances[k]
        mean_next, covariance_next, control_covariance = arc_function(
            means[:, k], covariance_k, feedforward[:, k], gains[k]
        )
        opti.subject_to(means[:, k + 1] == mean_next)

        # Covariance multiple shooting: tie the node's own Cholesky-parameterized
        # covariance to the arc map's prediction via a defect equality.
        covariance_defect = node_covariances[k + 1] - covariance_next
        opti.subject_to(
            casadi.vertcat(*[covariance_defect[r, c] for r, c in lower_indices]) == 0.0
        )

        # Epsilon-regularized control norm (matching solve_deterministic_nominal)
        # in place of the epigraph slack variable: no separate decision variable,
        # ||v_k|| is replaced everywhere -- objective and chance constraint alike
        # -- by sqrt(dot(v_k,v_k) + eps^2), which has a finite, well-defined
        # gradient at v_k = 0 (coast arcs) instead of the epigraph's kink there.
        control_norm = casadi.sqrt(
            casadi.dot(feedforward[:, k], feedforward[:, k])
            + options.control_norm_epsilon ** 2
        )
        radius = symbolic_psqrt_spectral_radius_2x2(
            control_covariance,
            options.spectral_eigenvalue_smoothing,
            options.spectral_radius_floor,
        )
        opti.subject_to(control_norm + psi_inv_u * radius <= options.u_max)

        for a_vec, b_val in ((options.a1, options.b1), (options.a2, options.b2)):
            a_cs = casadi.DM(a_vec)
            sigma_y = casadi.sqrt(
                casadi.fmax(a_cs.T @ covariance_k @ a_cs, 0.0) + options.spectral_radius_floor ** 2
            )
            opti.subject_to(a_cs.T @ means[:, k] + z_path * sigma_y <= b_val)

        # Thesis Eq. 4.8 composite cost: ||v_k|| plus quadratic covariance-trace
        # penalties, with NO quantile/chance-margin term (psi_inv_u * radius) in
        # the objective. That term is used only in the control chance constraint
        # above (control_norm + psi_inv_u*radius <= u_max), never charged into
        # the cost.
        objective = objective + options.dt * control_norm
        quadratic_penalty = quadratic_penalty + options.dt * (
            casadi.trace(Q_cs @ covariance_k) + casadi.trace(R_cs @ control_covariance)
        )

    opti.subject_to(means[:, n_arcs] == options.xf_mean)

    covariance_N = node_covariances[n_arcs]
    inverse_target_std = casadi.DM(1.0 / np.sqrt(np.diag(options.xf_covariance)))
    scaled_terminal = (
        casadi.diag(inverse_target_std) @ covariance_N @ casadi.diag(inverse_target_std)
    )
    terminal_slack = casadi.MX.eye(NX) - scaled_terminal

    n_margin = NX * (NX + 1) // 2
    margin = opti.variable(n_margin)
    margin_matrix = [[casadi.MX(0.0)] * NX for _ in range(NX)]
    for (row, col), entry in zip(lower_indices, casadi.vertsplit(margin)):
        margin_matrix[row][col] = entry
    Gmat = casadi.vertcat(*[casadi.horzcat(*row) for row in margin_matrix])
    opti.subject_to(casadi.diag(Gmat) >= options.terminal_margin_floor)
    residual = terminal_slack - Gmat @ Gmat.T
    opti.subject_to(
        casadi.vertcat(*[residual[row, col] for row, col in lower_indices]) == 0.0
    )

    total_objective = objective + quadratic_penalty
    opti.minimize(total_objective)

    # Seed from the same deterministic-nominal + TVLQR recipe as Phase 1, not
    # from Phase 1's converged solution. The Cholesky-factor node variables have
    # no counterpart in that shared seed, so they are derived here from
    # seed.covariances (the seed policy's own UT-propagated covariances).
    opti.set_initial(means, seed.means)
    opti.set_initial(feedforward, seed.feedforward)
    for k in range(n_arcs):
        opti.set_initial(gains[k], seed.gains[k])
    for k in range(n_arcs + 1):
        L_seed = cholesky_seed(seed.covariances[k])
        opti.set_initial(
            chol_entries[k], np.array([L_seed[row, col] for row, col in lower_indices])
        )
    opti.set_initial(margin, seed.margin)

    _configure_snopt(opti, options)

    try:
        solution = run_snopt_solver(opti, "double-integrator-phase2")
        solver_accepted = True
    except RuntimeError as error:
        print(f"  SNOPT did not converge ({error}); returning the last iterate.", flush=True)
        solution = opti.debug
        solver_accepted = False

    solve_metrics = solver_diagnostics(opti, solution, options, solver_accepted)

    solved_means = np.asarray(solution.value(means))
    solved_feedforward = np.asarray(solution.value(feedforward)).reshape(NU, n_arcs)
    solved_gains = np.stack(
        [np.asarray(solution.value(g)).reshape(NU, NX) for g in gains]
    )
    solved_covariances = np.empty((n_arcs + 1, NX, NX))
    for k in range(n_arcs + 1):
        entries = np.asarray(solution.value(chol_entries[k])).ravel()
        L = np.zeros((NX, NX))
        for (row, col), value in zip(lower_indices, entries):
            L[row, col] = value
        solved_covariances[k] = L @ L.T

    # Recompute control covariances / matching defects from the solved trajectory
    # using the same arc map, for diagnostics consistent with Phase 1.
    _, _, control_covariances = propagate_moments(
        arc_function, solved_means, solved_feedforward, solved_gains, solved_covariances[0]
    )
    propagated_means = np.empty_like(solved_means)
    propagated_means[:, 0] = solved_means[:, 0]
    max_cov_defect = 0.0
    for k in range(n_arcs):
        mean_next, covariance_next, _ = arc_function(
            solved_means[:, k], solved_covariances[k], solved_feedforward[:, k], solved_gains[k]
        )
        propagated_means[:, k + 1] = np.asarray(mean_next.full()).ravel()
        cov_pred = np.asarray(covariance_next.full())
        max_cov_defect = max(
            max_cov_defect, float(np.max(np.abs(cov_pred - solved_covariances[k + 1])))
        )
    matching_defect = float(np.max(np.abs(propagated_means[:, 1:] - solved_means[:, 1:])))

    solved_radius, smoothing_metrics = smoothed_control_radii(control_covariances, options)

    return RobustSolution(
        means=solved_means,
        feedforward=solved_feedforward,
        gains=solved_gains,
        covariances=solved_covariances,
        control_covariances=control_covariances,
        radius=solved_radius,
        objective=float(solution.value(total_objective)),
        diagnostics={
            **solve_metrics,
            "matching_defect": matching_defect,
            "covariance_matching_defect": max_cov_defect,
            **smoothing_metrics,
        },
    )


# --------------------------------------------------------------------------- #
# Verification: open-loop rollout and Monte Carlo
# --------------------------------------------------------------------------- #


def open_loop_covariances(options: Options, A: np.ndarray, B: np.ndarray, G: np.ndarray) -> np.ndarray:
    covariance = options.x0_covariance.copy()
    covariances = [covariance.copy()]
    for _ in range(options.n_arcs):
        covariance = A @ covariance @ A.T + G @ G.T
        covariances.append(covariance.copy())
    return np.stack(covariances)


def monte_carlo_rollout(
    options: Options,
    A: np.ndarray,
    B: np.ndarray,
    G: np.ndarray,
    solution: RobustSolution,
) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(options.monte_carlo_seed)
    samples = options.monte_carlo_samples
    L0 = np.linalg.cholesky(options.x0_covariance + 1e-14 * np.eye(NX))
    states = solution.means[:, 0:1] + L0 @ generator.standard_normal((NX, samples))
    trajectories = np.empty((options.n_arcs + 1, NX, samples))
    trajectories[0] = states

    for k in range(options.n_arcs):
        deviation = states - solution.means[:, k : k + 1]
        control = solution.feedforward[:, k : k + 1] + solution.gains[k] @ deviation
        magnitude = np.linalg.norm(control, axis=0)
        saturated = magnitude > options.u_max
        scale = np.ones_like(magnitude)
        scale[saturated] = options.u_max / magnitude[saturated]
        applied = control * scale[None, :]
        noise = G @ generator.standard_normal((NX, samples))
        states = A @ states + B @ applied + noise
        trajectories[k + 1] = states

    terminal_deviation = states - solution.means[:, -1:]
    sampled_covariance = np.cov(terminal_deviation)
    return {"trajectories": trajectories, "terminal_covariance": sampled_covariance}


# --------------------------------------------------------------------------- #
# Plotting (Fig. 4.1 / 4.2 style)
# --------------------------------------------------------------------------- #


def covariance_ellipse_points(covariance_2x2: np.ndarray, n_sigma: float, n_points: int = 90) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(0.5 * (covariance_2x2 + covariance_2x2.T))
    eigvals = np.clip(eigvals, 0.0, None)
    angle = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    return n_sigma * (eigvecs @ np.diag(np.sqrt(eigvals)) @ circle)


def plot_trajectory(
    options: Options,
    solution: RobustSolution,
    monte_carlo: dict | None,
    title: str,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 5.2), dpi=200)

    # Shaded infeasible region (right of both half-space boundaries).
    x1 = np.linspace(-1.0, 12.0, 200)
    y_line1 = options.b1 - options.a1[0] * x1
    y_line2 = (options.b2 - options.a2[0] * x1) / options.a2[1]
    axis.fill_between(x1, np.minimum(y_line1, y_line2), 8.0, color="#d9d9d9", zorder=0)
    axis.plot(x1, y_line1, "k--", lw=0.8)
    axis.plot(x1, y_line2, "k--", lw=0.8)

    if monte_carlo is not None:
        trajectories = monte_carlo["trajectories"]
        axis.plot(
            trajectories[:, 0, :], trajectories[:, 1, :],
            color="#008080", alpha=0.12, lw=0.4, zorder=1,
        )

    means = solution.means
    axis.plot(means[0], means[1], "k-+", lw=1.2, markersize=4, zorder=3)

    for k in range(0, means.shape[1]):
        points = covariance_ellipse_points(solution.covariances[k][0:2, 0:2], 3.0)
        axis.plot(means[0, k] + points[0], means[1, k] + points[1], color="#333333", lw=0.6, zorder=2)

    start_pts = covariance_ellipse_points(options.x0_covariance[0:2, 0:2], 3.0)
    axis.plot(options.x0_mean[0] + start_pts[0], options.x0_mean[1] + start_pts[1], "k-", lw=1.4)
    axis.text(options.x0_mean[0] - 1.6, options.x0_mean[1] - 0.3, "Start", fontsize=10)

    target_pts = covariance_ellipse_points(options.xf_covariance[0:2, 0:2], 3.0)
    axis.plot(options.xf_mean[0] + target_pts[0], options.xf_mean[1] + target_pts[1], "r-", lw=1.4)
    axis.text(options.xf_mean[0] + 0.3, options.xf_mean[1] - 0.9, "Target", fontsize=10)

    axis.set_xlim(0.0, 11.0)
    axis.set_ylim(0.0, 8.0)
    axis.set_xlabel(r"$x_1$")
    axis.set_ylabel(r"$x_2$")
    axis.set_title(title)
    axis.set_box_aspect(1.0)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def print_summary(label: str, options: Options, solution: RobustSolution) -> None:
    to_show = [
        "",
        "=" * 64,
        f"{label} -- double integrator covariance steering summary",
        "=" * 64,
        f"solver converged          : {'yes' if solution.diagnostics['converged'] else 'no'}",
        f"CasADi accepted solve     : {'yes' if solution.diagnostics['solver_accepted'] else 'no'}",
        f"optimality satisfied      : {'yes' if solution.diagnostics['optimality_satisfied'] else 'no'}",
        f"solver status             : {solution.diagnostics['solver_status']}",
        f"returned point feasible   : {'yes' if solution.diagnostics['feasible'] else 'no'}",
        f"max NLP constraint viol.  : {solution.diagnostics['max_constraint_violation']:.3e}",
        f"max mean matching defect  : {solution.diagnostics['matching_defect']:.3e}",
    ]
    if "covariance_matching_defect" in solution.diagnostics:
        to_show.append(
            f"max cov. matching defect  : {solution.diagnostics['covariance_matching_defect']:.3e}"
        )
    if "max_radius_smoothing_bias" in solution.diagnostics:
        to_show += [
            f"eigenvalue smoothing       : {options.spectral_eigenvalue_smoothing:.3e}",
            f"max radius smoothing bias : {solution.diagnostics['max_radius_smoothing_bias']:.3e}",
            f"max chance-margin bias    : {solution.diagnostics['max_control_margin_smoothing_bias']:.3e}",
            f"bias / peak exact radius  : {solution.diagnostics['relative_radius_smoothing_bias']:.3e}",
        ]
    to_show += [
        f"objective J               : {solution.objective:.6f}",
        f"terminal mean             : {solution.means[:, -1]}",
        f"terminal covariance diag  : {np.diag(solution.covariances[-1])}",
        f"target covariance diag    : {np.diag(options.xf_covariance)}",
        "=" * 64,
    ]
    print("\n".join(to_show), flush=True)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #


def main() -> None:
    options = Options()
    A, B, G = system_matrices(options)

    print("Solving the deterministic nominal trajectory...", flush=True)
    nominal = solve_deterministic_nominal(options, A, B)
    if not nominal.converged:
        print("Deterministic nominal solve did not converge; aborting.", flush=True)
        return

    print("Solving Phase 1 (mean multiple-shooting, covariance single-shot UT)...", flush=True)
    phase1_solution = solve_single_shot(options, A, B, G, nominal)
    print_summary("Phase 1 (single-shot covariance)", options, phase1_solution)

    monte_carlo_1 = monte_carlo_rollout(options, A, B, G, phase1_solution)
    plot_trajectory(
        options,
        phase1_solution,
        monte_carlo_1,
        "Closed-loop (Phase 1: single-shot covariance)",
        OUTPUT_DIR / "phase1_closed_loop.png",
    )

    open_loop_solution = RobustSolution(
        means=phase1_solution.means,
        feedforward=phase1_solution.feedforward,
        gains=np.zeros_like(phase1_solution.gains),
        covariances=open_loop_covariances(options, A, B, G),
        control_covariances=phase1_solution.control_covariances,
        radius=phase1_solution.radius,
        objective=float("nan"),
        diagnostics={"converged": 1.0, "matching_defect": 0.0},
    )
    plot_trajectory(
        options, open_loop_solution, None, "Open-loop (no feedback)",
        OUTPUT_DIR / "phase1_open_loop.png",
    )

    print(
        "\nSolving Phase 2 (covariance multiple-shooting via Cholesky-factor "
        "decision variables, seeded independently from the same deterministic "
        "nominal trajectory as Phase 1)...",
        flush=True,
    )
    phase2_solution = solve_multiple_shoot(options, A, B, G, nominal)
    print_summary("Phase 2 (covariance multiple-shooting)", options, phase2_solution)

    monte_carlo_2 = monte_carlo_rollout(options, A, B, G, phase2_solution)
    plot_trajectory(
        options,
        phase2_solution,
        monte_carlo_2,
        "Closed-loop (Phase 2: covariance multiple-shooting)",
        OUTPUT_DIR / "phase2_closed_loop.png",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUTPUT_DIR / "solutions.npz",
        phase1_means=phase1_solution.means,
        phase1_covariances=phase1_solution.covariances,
        phase1_gains=phase1_solution.gains,
        phase1_feedforward=phase1_solution.feedforward,
        phase2_means=phase2_solution.means,
        phase2_covariances=phase2_solution.covariances,
        phase2_gains=phase2_solution.gains,
        phase2_feedforward=phase2_solution.feedforward,
    )


if __name__ == "__main__":
    main()
