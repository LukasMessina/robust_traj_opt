"""
DIRTRAN and DIRTREL comparison with CasADi SNOPT.

The performance of the methods is analysed through the uncertain-mass pendulum example.
SNOPT itself is not redistributable: this script expects a licensed SNOPT
installation whose runtime directory is already on the operating-system PATH
and whose license is identified by the SNOPT_LICENSE environment variable.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import matplotlib
# matplotlib.use("Agg")
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
from plotter import Plotter


RESULTS_DIR  = Path(__file__).resolve().parent / "output"
# state, control and disturbance dimensions
NX, NU, NW = 2, 1, 1
INTERVALS = 100
INITIAL_DURATION = 2.0            # [s]
INITIAL_STEP = INITIAL_DURATION / INTERVALS
OPTIMALITY_TOLERANCE = 1e-8
MAX_MAJOR_ITER = 1000


@dataclass(frozen=True)
class Config:
    """Optimal control problem configuration.""" 

    length: float = 1.0
    mass_nominal: float = 1.0               # [kg]
    mass_uncertainty: float = 0.2           # [kg]
    torque_limit: float = 3.0               # [Nm]
    theta_initial: float = 0.0              # [rad]
    omega_initial: float = 0.0              # [rad/s]
    theta_final: float = float(np.pi)       # [rad]
    omega_final: float = 0.0                # [rad/s]
  
    g: float = 9.81                         # [m/s^2]
    knots: int = INTERVALS + 1              # [-]
    h_min: float = 1e-4                     # [s]
    h_max: float = 1e-1                     # [s]

    output_dir: Path = RESULTS_DIR
    major_print_level: int = 0
    minor_print_level: int = 0
    MAX_MAJOR_ITER: int = MAX_MAJOR_ITER 
    optimality_tolerance: float = OPTIMALITY_TOLERANCE
    initial_step: float = INITIAL_STEP
    initial_seed: int = 1
    mass_min: float = 0.5
    mass_max: float = 1.5
    mass_step: float = 0.01
    # These parameters define the success thresholds used during verification.
    success_angle: float = 0.01                                         # [rad]
    success_angle_rate: float = 0.01                                    # [rad/s]

    @property
    def Q(self) -> np.ndarray:
        return np.diag([10.0, 1.0])

    @property
    def R(self) -> np.ndarray:
        return np.array([[0.1]])

    @property
    def Q_terminal(self) -> np.ndarray:
        return np.diag([100.0, 100.0])

    @property
    def D(self) -> np.ndarray:
        return np.array([[self.mass_uncertainty**2]])

    @property
    def E_initial(self) -> np.ndarray:
        return np.zeros((NX, NX))

    @property
    def intervals(self) -> int:
        return self.knots - 1
    

@dataclass
class Dynamics:
    f_disc: ca.Function
    A_fun: ca.Function
    B_fun: ca.Function
    G_fun: ca.Function


@dataclass(frozen=True)
class InitialGuess:
    states: np.ndarray
    controls: np.ndarray
    timestep: float


@dataclass
class NLP:
    opti: ca.Opti
    x: ca.MX
    u: ca.MX
    h: ca.MX
    duration: ca.MX
    K: list[ca.MX] | None = None
    E: list[ca.MX] | None = None
    H: list[ca.MX] | None = None
    input_uncertainty: list[ca.MX] | None = None
    robust_cost: ca.MX | None = None


@dataclass
class Solution:
    x: np.ndarray
    u: np.ndarray
    h: float
    K: np.ndarray
    duration: float
    objective: float
    solver_wall_time_seconds: float
    solver_stats: dict[str, Any]
    E: np.ndarray | None = None
    H: np.ndarray | None = None
    input_uncertainty: np.ndarray | None = None
    robust_cost: float | None = None


@dataclass(frozen=True)
class VerificationResults:
    mass_interval: np.ndarray
    angle_errors: np.ndarray
    angular_rate_errors: np.ndarray
    success: np.ndarray
    max_success: float | None


def check_snopt_installation() -> None:
    """Check a preconfigured system SNOPT installation."""

    license_env = os.environ.get("SNOPT_LICENSE")
    if not license_env:
        raise RuntimeError(
            "SNOPT_LICENSE must point to a valid SNOPT license file."
        )

    license_path = Path(license_env).expanduser()
    if not license_path.is_file():
        raise RuntimeError(f"SNOPT license file not found: {license_path}")

    if not ca.has_nlpsol("snopt"):
        raise RuntimeError(
            "CasADi could not load its SNOPT plugin."
        )


def dynamics(cfg: Config) -> Dynamics:
    """Create continuous dynamics, explicit Euler map, and its Jacobians.

    The dynamical system is a point-mass pendulum: 
    - I = m*l^2   [kg*m^2]
    - CoM at l
    """

    x = ca.SX.sym("x", NX)
    u = ca.SX.sym("u", NU)
    w = ca.SX.sym("w", NW)
    h = ca.SX.sym("h")
    mass = cfg.mass_nominal + w[0]
    acceleration = u[0] / (mass * cfg.length**2) - cfg.g / cfg.length * ca.sin(x[0])
    xdot = ca.vertcat(x[1], acceleration)
    # Forward Euler integration scheme
    x_next = x + h * xdot  

    return Dynamics(
        f_disc=ca.Function("f_disc", [x, u, w, h], [x_next]),
        A_fun=ca.Function("A", [x, u, w, h], [ca.jacobian(x_next, x)]),
        B_fun=ca.Function("B", [x, u, w, h], [ca.jacobian(x_next, u)]),
        G_fun=ca.Function("G", [x, u, w, h], [ca.jacobian(x_next, w)]),
    )


def generate_initial_guess(
    cfg: Config,
    timestep: float = INITIAL_STEP,
) -> InitialGuess:
    """Return the direct-transcription initial guess.

    For the default n-knot grid, the timestep is chosen such that the initial
    duration is 2 seconds. The rotational angle is linearly interpolated from
    0 to pi, the rotational rate is zero at all timesteps, and the controls are
    normally distributed with a standard deviation of 0.01 N m.
    """

    theta = np.linspace(cfg.theta_initial, cfg.theta_final, cfg.knots)     # [rad]
    x = np.vstack([theta, np.zeros(cfg.knots, dtype=float)])               # [rad]
    rng = np.random.default_rng(cfg.initial_seed)
    u = 0.01 * rng.standard_normal((NU, cfg.intervals))                    # [Nm]
    return InitialGuess(x, u, timestep)


def solver_options(cfg: Config) -> tuple[dict[str, Any], dict[str, Any]]:
    snopt_options = {
        "Major iterations limit": cfg.MAX_MAJOR_ITER,
        "Iterations limit": 20 * cfg.MAX_MAJOR_ITER,
        "Major optimality tolerance": f"{cfg.optimality_tolerance:.16g}",
        "Major print level": cfg.major_print_level,
        "Minor print level": cfg.minor_print_level,
    }
    plugin_options = {"print_time": True, "snopt": snopt_options}
    return plugin_options, {}


def run_solver(opti: ca.Opti) -> ca.OptiSol:
    """Run SNOPT with solver.out file redirected to TEMP."""

    snopt_work_dir = Path(tempfile.gettempdir()) / "casadi-snopt"
    snopt_work_dir.mkdir(parents=True, exist_ok=True)
    original_work_dir = Path.cwd()
    try:
        os.chdir(snopt_work_dir)
        return opti.solve()
    finally:
        os.chdir(original_work_dir)


def _nominal_constraints(
    opti: ca.Opti,
    cfg: Config,
    dyn: Dynamics,
    x: ca.MX,
    u: ca.MX,
    h: ca.MX,
) -> None:
    opti.subject_to(opti.bounded(cfg.h_min, h, cfg.h_max))
    opti.subject_to(x[:, 0] == ca.DM([cfg.theta_initial, cfg.omega_initial]))
    opti.subject_to(x[:, -1] == ca.DM([cfg.theta_final, cfg.omega_final]))
    for i in range(cfg.intervals):
        opti.subject_to(x[:, i + 1] == dyn.f_disc(x[:, i], u[:, i], 0.0, h))
        opti.subject_to(
            opti.bounded(-cfg.torque_limit, u[0, i], cfg.torque_limit)
        )


def synthesize_tvlqr(
    A_seq: Sequence[Any],
    B_seq: Sequence[Any],
    Q: Any,
    R: Any,
    Q_terminal: Any,
    linear_solve: Callable[[Any, Any], Any],
) -> list[Any]:
    """Run the finite-horizon TVLQR recursion for CasADi or NumPy matrices."""

    P = Q_terminal
    K: list[Any] = [None] * len(A_seq)
    for i in reversed(range(len(A_seq))):
        A_i, B_i = A_seq[i], B_seq[i]
        K_i = linear_solve(R + B_i.T @ P @ B_i, B_i.T @ P @ A_i)
        K[i] = K_i
        A_cl = A_i - B_i @ K_i
        P = Q + K_i.T @ R @ K_i + A_cl.T @ P @ A_cl
        # Remove roundoff-level asymmetry; the Riccati cost matrix is symmetric.
        P = 0.5 * (P + P.T)
    return K


def build_dirtrel_formulation(cfg: Config, dyn: Dynamics) -> NLP:
    """Build the complete DIRTREL NLP formulation as one differentiable CasADi graph."""

    opti = ca.Opti()
    x = opti.variable(NX, cfg.knots)
    u = opti.variable(NU, cfg.intervals)
    h = opti.variable()
    _nominal_constraints(opti, cfg, dyn, x, u, h)
    duration = cfg.intervals * h

    # Build the discrete dynamics sequence of Jacobians on the nominal path.
    A_seq = [dyn.A_fun(x[:, i], u[:, i], 0.0, h) for i in range(cfg.intervals)]
    B_seq = [dyn.B_fun(x[:, i], u[:, i], 0.0, h) for i in range(cfg.intervals)]
    G_seq = [dyn.G_fun(x[:, i], u[:, i], 0.0, h) for i in range(cfg.intervals)]

    # Backward finite-horizon TVLQR. P_N = Q_N is the Riccati boundary value.
    Q, R, Q_terminal = ca.DM(cfg.Q), ca.DM(cfg.R), ca.DM(cfg.Q_terminal)
    K = synthesize_tvlqr(A_seq, B_seq, Q, R, Q_terminal, ca.solve)

    E_i = ca.MX(cfg.E_initial)
    H_i = ca.MX.zeros(NX, NW)
    D = ca.DM(cfg.D)
    E = [E_i]
    H = [H_i]
    input_uncertainty: list[ca.MX] = []
    robust_cost = ca.MX(0.0)

    for i in range(cfg.intervals):
        A_i, B_i, G_i, K_i = A_seq[i], B_seq[i], G_seq[i], K[i]
        A_cl = A_i - B_i @ K_i
        robust_cost += ca.trace((Q + K_i.T @ R @ K_i) @ E_i)

        # NU=1, thus the principal matrix square root is scalar.
        radius = ca.sqrt((K_i @ E_i @ K_i.T)[0, 0])
        input_uncertainty.append(radius)
        opti.subject_to(u[0, i] - radius >= -cfg.torque_limit)
        opti.subject_to(u[0, i] + radius <= cfg.torque_limit)

        E_ip1 = (
            A_cl @ E_i @ A_cl.T
            + A_cl @ H_i @ G_i.T
            + G_i @ H_i.T @ A_cl.T
            + G_i @ D @ G_i.T
        )
        H_ip1 = A_cl @ H_i + G_i @ D
        E_i, H_i = E_ip1, H_ip1
        E.append(E_i)
        H.append(H_i)

    robust_cost += ca.trace(Q_terminal @ E[-1])
    # Time optimal + robust cost
    opti.minimize(duration + robust_cost) 
    return NLP(
        opti=opti,
        x=x,
        u=u,
        h=h,
        K=K,
        E=E,
        H=H,
        input_uncertainty=input_uncertainty,
        robust_cost=robust_cost,
        duration=duration,
    )


def build_dirtran_formulation(cfg: Config, dyn: Dynamics) -> NLP:
    """Build the nominal minimum-time DIRTRAN baseline formulation."""

    opti = ca.Opti()
    x = opti.variable(NX, cfg.knots)
    u = opti.variable(NU, cfg.intervals)
    h = opti.variable()
    _nominal_constraints(opti, cfg, dyn, x, u, h)
    duration = cfg.intervals * h
    opti.minimize(duration)
    return NLP(opti=opti, x=x, u=u, h=h, duration=duration)


def solve_nlp(
    nlp: NLP,
    cfg: Config,
    dyn: Dynamics,
    initial: InitialGuess,
) -> Solution:
    """Solve either DIRTRAN/DIRTREL formulation and perform method-specific post-processing."""

    nlp.opti.set_initial(nlp.x, initial.states)
    nlp.opti.set_initial(nlp.u, initial.controls)
    nlp.opti.set_initial(nlp.h, initial.timestep)
    nlp.opti.solver("snopt", *solver_options(cfg))
    solve_start = perf_counter()
    sol = run_solver(nlp.opti)
    solver_wall_time = perf_counter() - solve_start

    x = np.asarray(sol.value(nlp.x))
    u = np.asarray(sol.value(nlp.u)).reshape(NU, cfg.intervals)
    h = float(sol.value(nlp.h))
    duration = float(sol.value(nlp.duration))

    E = None
    H = None
    input_uncertainty = None
    robust_cost = None
    if nlp.robust_cost is not None:
        K = np.stack(
            [np.asarray(sol.value(K_i)).reshape(NU, NX) for K_i in nlp.K]
        )
        E = np.stack(
            [np.asarray(sol.value(E_i)).reshape(NX, NX) for E_i in nlp.E]
        )
        H = np.stack(
            [np.asarray(sol.value(H_i)).reshape(NX, NW) for H_i in nlp.H]
        )
        input_uncertainty = np.array(
            [float(sol.value(radius)) for radius in nlp.input_uncertainty]
        )
        robust_cost = float(sol.value(nlp.robust_cost))
    else:
        A_seq = [
            np.asarray(dyn.A_fun(x[:, i], u[:, i], 0.0, h), dtype=float)
            for i in range(cfg.intervals)
        ]
        B_seq = [
            np.asarray(dyn.B_fun(x[:, i], u[:, i], 0.0, h), dtype=float)
            for i in range(cfg.intervals)
        ]
        K = np.stack(
            synthesize_tvlqr(
                A_seq,
                B_seq,
                cfg.Q,
                cfg.R,
                cfg.Q_terminal,
                np.linalg.solve,
            )
        )

    return Solution(
        x=x,
        u=u,
        h=h,
        K=K,
        E=E,
        H=H,
        input_uncertainty=input_uncertainty,
        robust_cost=robust_cost,
        duration=duration,
        objective=duration + (robust_cost if robust_cost is not None else 0.0),
        solver_wall_time_seconds=solver_wall_time,
        solver_stats=sol.stats(),
    )


def verify_solution(cfg: Config, dyn: Dynamics, sol: Solution) -> dict[str, Any]:
    """Verify common constraints and, when present, DIRTREL uncertainty fields."""

    max_dynamical_residual = 0.0
    for i in range(cfg.intervals):
        predicted = np.asarray(
            dyn.f_disc(sol.x[:, i], sol.u[:, i], 0.0, sol.h)
        ).ravel()
        max_dynamical_residual = max(
            max_dynamical_residual,
            float(np.max(np.abs(sol.x[:, i + 1] - predicted))),
        )

    max_initial_state_error = np.max(
        np.abs(sol.x[:, 0] - [cfg.theta_initial, cfg.omega_initial])
    )
    max_terminal_state_error = np.max(
        np.abs(sol.x[:, -1] - [cfg.theta_final, cfg.omega_final])
    )
    max_nominal_input_violation = max(
        0.0,
        float(np.max(np.abs(sol.u)) - cfg.torque_limit),
    )
    verification_results = {
        "max_nominal_dynamics_residual": max_dynamical_residual,
        "max_initial_state_error": float(max_initial_state_error),
        "max_terminal_state_error": float(max_terminal_state_error),
        "max_nominal_input_violation": max_nominal_input_violation,
        "minimum_nominal_input_margin": float(
            cfg.torque_limit - np.max(np.abs(sol.u))
        ),
    }

    dirtrel_fields = (sol.E, sol.H, sol.input_uncertainty, sol.robust_cost)
    if not any([field is not None for field in dirtrel_fields]):
        return verification_results

    E = sol.E
    H = sol.H
    input_uncertainty = sol.input_uncertainty
    assert E is not None and H is not None and input_uncertainty is not None

    max_uncertainty_propagation_error = 0.0
    min_ellipsoid_eigenvalue = np.inf
    for i in range(cfg.intervals):
        A_i = np.asarray(dyn.A_fun(sol.x[:, i], sol.u[:, i], 0.0, sol.h))
        B_i = np.asarray(dyn.B_fun(sol.x[:, i], sol.u[:, i], 0.0, sol.h))
        G_i = np.asarray(dyn.G_fun(sol.x[:, i], sol.u[:, i], 0.0, sol.h))
        A_cl = A_i - B_i @ sol.K[i]
        M_i = np.block(
            [
                [E[i], H[i]],
                [H[i].T, cfg.D],
            ]
        )
        F_i = np.block([[A_cl, G_i], [np.zeros((NW, NX)), np.eye(NW)]])
        M_ip1 = F_i @ M_i @ F_i.T
        # D is unchanged by construction, so verify only the E and H blocks.
        max_uncertainty_propagation_error = max(
            max_uncertainty_propagation_error,
            float(np.max(np.abs(M_ip1[:NX, :NX] - E[i + 1]))),
            float(np.max(np.abs(M_ip1[:NX, NX:] - H[i + 1]))),
        )
        min_ellipsoid_eigenvalue = min(
            min_ellipsoid_eigenvalue,
            float(np.linalg.eigvalsh(E[i]).min()),
        )
    min_ellipsoid_eigenvalue = min(
        min_ellipsoid_eigenvalue,
        float(np.linalg.eigvalsh(E[-1]).min()),
    )

    robust_upper = sol.u[0] + input_uncertainty
    robust_lower = sol.u[0] - input_uncertainty
    max_robust_input_violation = max(
        0.0,
        float(np.max(robust_upper - cfg.torque_limit)),
        float(np.max(-cfg.torque_limit - robust_lower)),
    )
    minimum_robust_input_margin = min(
        np.min(cfg.torque_limit - robust_upper),
        np.min(robust_lower + cfg.torque_limit),
    )
    verification_results.update(
        {
            "max_robust_input_violation": max_robust_input_violation,
            "max_uncertainty_propagation_error": max_uncertainty_propagation_error,
            "min_ellipsoid_eigenvalue": min_ellipsoid_eigenvalue,
            "minimum_robust_input_margin": float(minimum_robust_input_margin),
        }
    )
    return verification_results


def wrap_angle_error(theta: float, target: float) -> float:
    return float(abs(np.arctan2(np.sin(theta - target), np.cos(theta - target))))


def simulate_closed_loop(
    cfg: Config,
    dyn: Dynamics,
    sol: Solution,
    actual_mass: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the same discrete model as the paper's explicit-Euler transcription."""

    disturbance = actual_mass - cfg.mass_nominal
    x_sim = np.zeros_like(sol.x)
    u_sim = np.zeros(cfg.intervals)
    x_sim[:, 0] = sol.x[:, 0]
    for i in range(cfg.intervals):
        feedback = float((sol.K[i] @ (x_sim[:, i] - sol.x[:, i])).item())
        u_sim[i] = np.clip(sol.u[0, i] - feedback, -cfg.torque_limit, cfg.torque_limit)
        x_sim[:, i + 1] = np.asarray(
            dyn.f_disc(
                x_sim[:, i],
                np.array([u_sim[i]]),
                np.array([disturbance]),
                sol.h,
            )
        ).ravel()
    return x_sim, u_sim


def robustness_analysis(
    cfg: Config,
    dyn: Dynamics,
    sol: Solution,
    mass_interval: np.ndarray,
    angle_tolerance: float,
    angular_rate_tolerance: float,
) -> VerificationResults:
    angle_errors, angular_rate_errors, success = [], [], []
    for mass in mass_interval:
        x_sim, _ = simulate_closed_loop(cfg, dyn, sol, float(mass))
        angle_error = wrap_angle_error(x_sim[0, -1], cfg.theta_final)
        angular_rate_error = abs(float(x_sim[1, -1] - cfg.omega_final))
        angle_errors.append(angle_error)
        angular_rate_errors.append(angular_rate_error)
        success.append(angle_error <= angle_tolerance and angular_rate_error <= angular_rate_tolerance)

    angle_errors = np.asarray(angle_errors)
    rate_errors = np.asarray(angular_rate_errors)
    successes = np.asarray(success, dtype=bool)
    high_indices = np.flatnonzero(mass_interval >= cfg.mass_nominal - 1e-12)
    max_mass: float | None = None
    if high_indices.size and successes[high_indices[0]]:
        max_mass = float(mass_interval[high_indices[0]])
        for index in high_indices[1:]:
            if not successes[index]:
                break
            max_mass = float(mass_interval[index])
    return VerificationResults(
        mass_interval=mass_interval,
        angle_errors=angle_errors,
        angular_rate_errors=rate_errors,
        success=successes,
        max_success=max_mass,
    )


def plot_trajectory_and_controls(cfg: Config, sol: Solution, output: Path) -> None:
    time = np.arange(cfg.knots) * sol.h
    control_time = np.r_[time[:-1], time[-1]]
    nominal_u = np.r_[sol.u[0], sol.u[0, -1]]
    lower_u = np.r_[sol.u[0] - sol.input_uncertainty, sol.u[0, -1] - sol.input_uncertainty[-1]]
    upper_u = np.r_[sol.u[0] + sol.input_uncertainty, sol.u[0, -1] + sol.input_uncertainty[-1]]

    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    axes[0].plot(
        time,
        sol.x[0],
        "o-",
        ms=3,
        color="tab:blue",
        label="nominal DIRTREL",
    )
    axes[0].axhline(
        cfg.theta_final,
        color="0.4",
        ls=":",
        lw=1,
        label=r"upright $\pi$",
    )
    axes[0].set_ylabel(r"$\theta$ [rad]")
    axes[0].legend(loc="best")

    axes[1].plot(time, sol.x[1], "o-", ms=3, color="tab:orange")
    axes[1].axhline(0.0, color="0.4", ls=":", lw=1)
    axes[1].set_ylabel(r"$\dot\theta$ [rad/s]")

    axes[2].fill_between(
        control_time,
        lower_u,
        upper_u,
        alpha=0.22,
        color="tab:blue",
        label=r"$u_i\pm\sqrt{K_iE_iK_i^T}$",
    )
    # Match the paper's Figure 1 rendering: connect control knot values with
    # straight line segments.  This is a visualization choice only; the NLP
    # still uses the explicit-Euler transcription from equation (1).
    axes[2].plot(control_time, nominal_u, color="tab:blue", label="nominal torque")
    axes[2].axhline(cfg.torque_limit, color="k", ls="--", lw=1)
    axes[2].axhline(-cfg.torque_limit, color="k", ls="--", lw=1, label="torque limits")
    axes[2].set_ylabel(r"$u$ [N m]")
    axes[2].set_xlabel("time [s]")
    axes[2].legend(loc="best")

    fig.suptitle(f"DIRTREL uncertain-mass pendulum (N={cfg.knots})")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_uncertainty_tube(cfg: Config, sol: Solution, output: Path) -> None:
    time = np.arange(cfg.knots) * sol.h
    theta_radius = np.sqrt(np.maximum(sol.E[:, 0, 0], 0.0))
    omega_radius = np.sqrt(np.maximum(sol.E[:, 1, 1], 0.0))
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, nominal, radius, label in (
        (axes[0], sol.x[0], theta_radius, r"$\theta$ [rad]"),
        (axes[1], sol.x[1], omega_radius, r"$\dot\theta$ [rad/s]"),
    ):
        ax.plot(time, nominal, color="tab:blue", label="nominal")
        ax.fill_between(
            time,
            nominal - radius,
            nominal + radius,
            color="tab:blue",
            alpha=0.25,
            label="ellipsoid projection",
        )
        ax.set_ylabel(label)
        ax.legend(loc="best")
    axes[1].set_xlabel("time [s]")
    fig.suptitle("Propagated DIRTREL state-deviation ellipsoid")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_mass_sweep(
    plotter: Plotter,
    cfg: Config,
    sweep: VerificationResults,
    angle_tolerance: float,
    angular_rate_tolerance: float,
) -> Path:
    mass_interval = sweep.mass_interval
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(5.8, 5.0),
        dpi=plotter.FIGURE_DPI,
        sharex=True,
    )
    axes[0].plot(
        mass_interval,
        sweep.angle_errors,
        color=plotter.BLUE,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        label=r"$|\theta_N-\pi|$ (wrapped)",
    )
    axes[0].plot(
        mass_interval,
        sweep.angular_rate_errors,
        color=plotter.ORANGE,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        label=r"$|\dot\theta_N|$",
    )
    axes[0].axhline(
        angle_tolerance,
        color=plotter.BLUE,
        linestyle=":",
        lw=plotter.GUIDE_LINE_WIDTH,
    )
    axes[0].axhline(
        angular_rate_tolerance,
        color=plotter.ORANGE,
        linestyle=":",
        lw=plotter.GUIDE_LINE_WIDTH,
    )
    axes[0].set_ylabel("terminal error")
    plotter._legend(
        axes[0],
        loc="best",
        frameon=True,
        fancybox=False,
        edgecolor=plotter.BLACK,
        facecolor="white",
        framealpha=1.0,
    )

    success = sweep.success.astype(float)
    axes[1].step(
        mass_interval,
        success,
        where="mid",
        color=plotter.GREEN,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        label="DIRTREL",
    )
    axes[1].axvspan(
        cfg.mass_nominal - cfg.mass_uncertainty,
        cfg.mass_nominal + cfg.mass_uncertainty,
        color=plotter.GREY,
        alpha=0.15,
        label="design mass set",
    )
    axes[1].axvline(
        1.3,
        color=plotter.RED,
        linestyle=":",
        lw=plotter.GUIDE_LINE_WIDTH,
        label=r"paper: $m\approx1.3$",
    )
    axes[1].set_yticks([0, 1], ["fail", "success"])
    axes[1].set_ylim(-0.15, 1.15)
    axes[1].set_xlabel("mass [kg]")
    plotter._legend(
        axes[1],
        loc="best",
        frameon=True,
        fancybox=False,
        edgecolor=plotter.BLACK,
        facecolor="white",
        framealpha=1.0,
    )
    for ax in axes:
        plotter._style_2d_axis(
            ax,
            tick_size=plotter.VERIFICATION_TICK_SIZE,
            label_size=plotter.VERIFICATION_LABEL_SIZE,
        )
    fig.tight_layout()
    output = plotter._path("mass_sweep")
    fig.savefig(output, dpi=plotter.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_method_comparison(
    plotter: Plotter,
    cfg: Config,
    dirtrel: Solution,
    dirtran: Solution,
) -> Path:
    """Compare the two optimized nominal trajectories, as in paper Figure 1."""

    if dirtrel.E is None or dirtrel.input_uncertainty is None:
        raise ValueError("The DIRTREL comparison plot requires uncertainty data")

    dirtrel_time = np.arange(cfg.knots) * dirtrel.h
    dirtran_time = np.arange(cfg.knots) * dirtran.h
    dirtrel_control_time = np.r_[dirtrel_time[:-1], dirtrel_time[-1]]
    dirtran_control_time = np.r_[dirtran_time[:-1], dirtran_time[-1]]
    dirtrel_u = np.r_[dirtrel.u[0], dirtrel.u[0, -1]]
    dirtran_u = np.r_[dirtran.u[0], dirtran.u[0, -1]]
    lower_u = np.r_[
        dirtrel.u[0] - dirtrel.input_uncertainty,
        dirtrel.u[0, -1] - dirtrel.input_uncertainty[-1],
    ]
    upper_u = np.r_[
        dirtrel.u[0] + dirtrel.input_uncertainty,
        dirtrel.u[0, -1] + dirtrel.input_uncertainty[-1],
    ]
    theta_radius = np.sqrt(np.maximum(dirtrel.E[:, 0, 0], 0.0))
    angular_rate_radius = np.sqrt(np.maximum(dirtrel.E[:, 1, 1], 0.0))

    fig, axes = plt.subplots(
        1,
        3,
        figsize=plotter.THREE_PANEL_FIGSIZE,
        dpi=plotter.FIGURE_DPI,
    )
    axes[0].fill_between(
        dirtrel_time,
        dirtrel.x[0] - theta_radius,
        dirtrel.x[0] + theta_radius,
        color=plotter.BLUE,
        alpha=0.18,
        linewidth=0.0,
        label="uncertainty tube",
    )
    axes[0].plot(
        dirtrel_time,
        dirtrel.x[0],
        color=plotter.BLUE,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        label="DIRTREL",
    )
    axes[0].plot(
        dirtran_time,
        dirtran.x[0],
        color=plotter.RED,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        linestyle="--",
        zorder=4,
        label="DIRTRAN",
    )
    axes[0].axhline(
        cfg.theta_final,
        color=plotter.BLACK,
        linestyle=":",
        lw=plotter.GUIDE_LINE_WIDTH,
        label="target",
    )
    axes[0].set_ylabel(r"$\theta$ [rad]")

    axes[1].fill_between(
        dirtrel_time,
        dirtrel.x[1] - angular_rate_radius,
        dirtrel.x[1] + angular_rate_radius,
        color=plotter.BLUE,
        alpha=0.18,
        linewidth=0.0,
        label="uncertainty tube",
    )
    axes[1].plot(
        dirtrel_time,
        dirtrel.x[1],
        color=plotter.BLUE,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        label="DIRTREL",
    )
    axes[1].plot(
        dirtran_time,
        dirtran.x[1],
        color=plotter.RED,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        linestyle="--",
        zorder=4,
        label="DIRTRAN",
    )
    axes[1].axhline(
        0.0,
        color=plotter.BLACK,
        linestyle=":",
        lw=plotter.GUIDE_LINE_WIDTH,
        label="target",
    )
    axes[1].set_ylabel(r"$\dot\theta$ [rad/s]")

    axes[2].fill_between(
        dirtrel_control_time,
        lower_u,
        upper_u,
        color=plotter.BLUE,
        alpha=0.18,
        linewidth=0.0,
        label="input tube",
    )
    axes[2].plot(
        dirtrel_control_time,
        dirtrel_u,
        color=plotter.BLUE,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        label="DIRTREL",
    )
    axes[2].plot(
        dirtran_control_time,
        dirtran_u,
        color=plotter.RED,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        linestyle="--",
        zorder=4,
        label="DIRTRAN",
    )
    axes[2].axhline(
        cfg.torque_limit,
        color=plotter.BLACK,
        linestyle=":",
        lw=plotter.GUIDE_LINE_WIDTH,
        label="torque limits",
    )
    axes[2].axhline(
        -cfg.torque_limit,
        color=plotter.BLACK,
        linestyle=":",
        lw=plotter.GUIDE_LINE_WIDTH,
    )
    axes[2].set_ylabel(r"$u$ [N m]")

    for ax in axes:
        ax.set_xlabel("time [s]")
        plotter._style_2d_axis(
            ax,
            tick_size=plotter.VERIFICATION_TICK_SIZE,
            label_size=9.0,
        )
        plotter._legend(
            ax,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            fontsize=5.8,
            frameon=True,
            fancybox=False,
            edgecolor=plotter.BLACK,
            facecolor="white",
            framealpha=1.0,
            borderpad=0.28,
            handlelength=1.2,
        )
    fig.tight_layout()
    output = plotter.output_prefix.parent / "trajectory_comparison.png"
    fig.savefig(output, dpi=plotter.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output


def plot_mass_sweep_comparison(
    plotter: Plotter,
    cfg: Config,
    dirtrel_robustness_analysis: VerificationResults,
    dirtran_robustness_analysis: VerificationResults,
) -> Path:
    """Compare closed-loop terminal errors under the identical mass sweep."""

    mass_interval = dirtrel_robustness_analysis.mass_interval
    fig, axes = plt.subplots(
        1,
        3,
        figsize=plotter.THREE_PANEL_FIGSIZE,
        dpi=plotter.FIGURE_DPI,
    )
    axes[0].plot(
        mass_interval,
        dirtrel_robustness_analysis.angle_errors,
        color=plotter.BLUE,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        label="DIRTREL",
    )
    axes[0].plot(
        mass_interval,
        dirtran_robustness_analysis.angle_errors,
        color=plotter.RED,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        linestyle="--",
        zorder=4,
        label="DIRTRAN",
    )
    axes[0].axhline(
        cfg.success_angle,
        color=plotter.BLACK,
        linestyle=":",
        lw=plotter.GUIDE_LINE_WIDTH,
        label="tolerance",
    )
    axes[0].set_ylabel(r"$|\theta_N-\pi|$ [rad]")

    axes[1].plot(
        mass_interval,
        dirtrel_robustness_analysis.angular_rate_errors,
        color=plotter.BLUE,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        label="DIRTREL",
    )
    axes[1].plot(
        mass_interval,
        dirtran_robustness_analysis.angular_rate_errors,
        color=plotter.RED,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        linestyle="--",
        zorder=4,
        label="DIRTRAN",
    )
    axes[1].axhline(
        cfg.success_angle_rate,
        color=plotter.BLACK,
        linestyle=":",
        lw=plotter.GUIDE_LINE_WIDTH,
        label="tolerance",
    )
    axes[1].set_ylabel(r"$|\dot\theta_N|$ [rad/s]")

    axes[2].step(
        mass_interval,
        dirtrel_robustness_analysis.success.astype(float),
        where="mid",
        color=plotter.BLUE,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        label="DIRTREL",
    )
    axes[2].step(
        mass_interval,
        dirtran_robustness_analysis.success.astype(float),
        where="mid",
        color=plotter.RED,
        lw=plotter.TRAJECTORY_LINE_WIDTH,
        linestyle="--",
        zorder=4,
        label="DIRTRAN",
    )
    axes[2].axvspan(
        cfg.mass_nominal - cfg.mass_uncertainty,
        cfg.mass_nominal + cfg.mass_uncertainty,
        color=plotter.GREY,
        alpha=0.15,
        label="design mass set",
    )
    axes[2].set_yticks([0, 1], ["fail", "success"])
    axes[2].set_ylim(-0.15, 1.15)

    for ax in axes:
        ax.set_xlabel("mass [kg]")
        plotter._style_2d_axis(
            ax,
            tick_size=plotter.VERIFICATION_TICK_SIZE,
            label_size=9.0,
        )
        plotter._legend(
            ax,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            fontsize=5.8,
            frameon=True,
            fancybox=False,
            edgecolor=plotter.BLACK,
            facecolor="white",
            framealpha=1.0,
            borderpad=0.28,
            handlelength=1.2,
        )
    fig.tight_layout()
    output = plotter.output_prefix.parent / "robustness_comparison.png"
    fig.savefig(output, dpi=plotter.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return output


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def get_uncertain_mass_interval(mass_min: float, mass_max: float, step: float) -> np.ndarray:
    count = int(np.floor((mass_max - mass_min) / step + 0.5)) + 1
    mass_interval = mass_min + np.arange(count) * step
    mass_interval[-1] = min(mass_interval[-1], mass_max)
    return mass_interval


def solution_metrics(solution: Solution) -> dict[str, Any]:
    result = {
        "solver": "SNOPT",
        "return_status": solution.solver_stats.get("return_status"),
        "secondary_return_status": solution.solver_stats.get(
            "secondary_return_status"
        ),
        "success": solution.solver_stats.get("success"),
        "solver_wall_time_seconds": solution.solver_wall_time_seconds,
        "timestep": solution.h,
        "duration": solution.duration,
        "objective": solution.objective,
        "maximum_nominal_abs_torque": float(np.max(np.abs(solution.u))),
    }
    if solution.robust_cost is not None:
        result["robust_cost"] = solution.robust_cost
    return result


def sweep_metrics(
    cfg: Config,
    sweep: VerificationResults,
    paper_reported_mass: float,
) -> dict[str, Any]:
    return {
        "angle_tolerance": cfg.success_angle,
        "angular_rate_tolerance": cfg.success_angle_rate,
        "max_success": sweep.max_success,
        "paper_reported_mass": paper_reported_mass,
    }


def build_stats(
    cfg: Config,
    dirtrel: Solution,
    dirtrel_verification: dict[str, Any],
    dirtrel_robustness_analysis: VerificationResults,
    dirtran: Solution,
    dirtran_verification: dict[str, Any],
    dirtran_robustness_analysis: VerificationResults,
) -> dict[str, Any]:
    max_mass_improvement = None
    if dirtrel_robustness_analysis.max_success is not None and dirtran_robustness_analysis.max_success is not None:
        max_mass_improvement = (
            dirtrel_robustness_analysis.max_success - dirtran_robustness_analysis.max_success
        )
    return {
        "configuration": {
            "knots": cfg.knots,
            "g": cfg.g,
            "mass_nominal": cfg.mass_nominal,
            "mass_uncertainty": cfg.mass_uncertainty,
            "torque_limit": cfg.torque_limit,
            "h_bounds": [cfg.h_min, cfg.h_max],
            "initial_timestep": cfg.initial_step,
            "initial_duration": cfg.intervals * cfg.initial_step,
            "initial_random_seed": cfg.initial_seed,
            "snopt_major_optimality_tolerance": cfg.optimality_tolerance,
        },
        "dirtrel": {
            "solution": solution_metrics(dirtrel),
            "verification": dirtrel_verification,
            "robustness_analysis": sweep_metrics(cfg, dirtrel_robustness_analysis, 1.3),
        },
        "dirtran": {
            "solution": solution_metrics(dirtran),
            "verification": dirtran_verification,
            "robustness_analysis": sweep_metrics(cfg, dirtran_robustness_analysis, 1.1),
        },
        "comparison": {
            "duration_difference": (
                dirtrel.duration - dirtran.duration
            ),
            "solver_wall_time_ratio": (
                dirtrel.solver_wall_time_seconds / dirtran.solver_wall_time_seconds
            ),
            "max_success_mass_improvement": max_mass_improvement,
        },
    }


def save_outputs(
    output_dir: Path,
    cfg: Config,
    dirtrel: Solution,
    dirtrel_robustness_analysis: VerificationResults,
    dirtran: Solution,
    dirtran_robustness_analysis: VerificationResults,
    stats: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plotter = Plotter(output_dir / "dirtrel")
    plot_method_comparison(
        plotter,
        cfg,
        dirtrel,
        dirtran,
    )
    plot_mass_sweep_comparison(
        plotter,
        cfg,
        dirtrel_robustness_analysis,
        dirtran_robustness_analysis,
    )
    (output_dir / "stats.json").write_text(
        json.dumps(json_ready(stats), indent=2),
        encoding="utf-8",
    )
    np.savez(
        output_dir / "solutions.npz",
        dirtrel_x=dirtrel.x,
        dirtrel_u=dirtrel.u,
        dirtrel_h=dirtrel.h,
        dirtrel_K=dirtrel.K,
        dirtrel_E=dirtrel.E,
        dirtrel_H=dirtrel.H,
        dirtrel_input_radii=dirtrel.input_uncertainty,
        dirtran_x=dirtran.x,
        dirtran_u=dirtran.u,
        dirtran_h=dirtran.h,
        dirtran_K=dirtran.K,
        mass_interval=dirtrel_robustness_analysis.mass_interval,
        dirtrel_mass_angle_errors=dirtrel_robustness_analysis.angle_errors,
        dirtrel_mass_rate_errors=dirtrel_robustness_analysis.angular_rate_errors,
        dirtrel_mass_successes=dirtrel_robustness_analysis.success,
        dirtran_mass_angle_errors=dirtran_robustness_analysis.angle_errors,
        dirtran_mass_rate_errors=dirtran_robustness_analysis.angular_rate_errors,
        dirtran_mass_successes=dirtran_robustness_analysis.success,
    )


def main() -> None:

    cfg = Config()
    output_dir = cfg.output_dir.expanduser().resolve()
    check_snopt_installation()
    dyn = dynamics(cfg)
    initial = generate_initial_guess(cfg, timestep=cfg.initial_step)
    dirtrel_solution = solve_nlp(build_dirtrel_formulation(cfg, dyn), cfg, dyn, initial)
    dirtran_solution = solve_nlp(build_dirtran_formulation(cfg, dyn), cfg, dyn, initial)
    dirtrel_verification = verify_solution(cfg, dyn, dirtrel_solution)
    dirtran_verification = verify_solution(cfg, dyn, dirtran_solution)
    mass_interval = get_uncertain_mass_interval(cfg.mass_min, cfg.mass_max, cfg.mass_step)
    dirtrel_robustness_analysis = robustness_analysis(
    cfg,
    dyn,
    dirtrel_solution,
    mass_interval,
    cfg.success_angle,
    cfg.success_angle_rate,
    )
    dirtran_robustness_analysis = robustness_analysis(
        cfg,
        dyn,
        dirtran_solution,
        mass_interval,
        cfg.success_angle,
        cfg.success_angle_rate,
    )
    stats = build_stats(
        cfg,
        dirtrel_solution,
        dirtrel_verification,
        dirtrel_robustness_analysis,
        dirtran_solution,
        dirtran_verification,
        dirtran_robustness_analysis,
    )
    save_outputs(
        output_dir,
        cfg,
        dirtrel_solution,
        dirtrel_robustness_analysis,
        dirtran_solution,
        dirtran_robustness_analysis,
        stats,
    )


if __name__ == "__main__":
    main()
