"""
TVLQR + DIRTREL ellipsoid propagation along the fuel-optimal CR3BP reference.

Every computation in this module works in SI units.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import casadi
import matplotlib.pyplot as plt
import numpy as np

from dirtran_cr3bp import (
    CR3BPEarthMoon,
    TestCase,
    OCPSolution,
    eom as eom_nd,
    collocation_coefficients,
    CASE_REGISTRY,
)
import integrator
from plotter import Plotter  # also applies plotter.py's mpl.rcParams (serif/LaTeX/white) globally

NX, NU, NW = 7, 3, 3
REFERENCE_DIR = Path("output/cr3bp_fuel_optimal")
OUTPUT_DIR = Path("output/tvlqr_cr3bp")
SUBSTEPS = 100
SECONDS_PER_DAY = 86400.0
MC_SAMPLES = 2000
MC_SEED = 42
_RESIDUAL_CACHE: dict[int, casadi.Function] = {}
STD_ND: dict[str, np.ndarray] = {
    "halo_l2_to_halo_l1": np.array([1e-8, 1e-8, 1e-8, 5e-9, 5e-9, 5e-9, 1e-5]),
    "lyapunov_l1_to_l2": np.array([5e-7, 5e-7, 5e-7, 1e-7, 1e-7, 1e-7, 1e-5]),
}


def scale_augm_state(case: CR3BPEarthMoon) -> np.ndarray:
    """Convert augmented state to dimensional units."""

    length_scale = case.length_unit * 1e3
    velocity_scale = case.velocity_unit * 1e3
    return np.array([length_scale, length_scale, length_scale, velocity_scale, velocity_scale, velocity_scale, case.m0_wet])


def eom(case: CR3BPEarthMoon, state: Any, control: Any, w: Any) -> Any:
    """Equations of motion."""

    velocity_unit_ms = case.velocity_unit * 1e3            # [m/s]
    accel_unit_ms2 = velocity_unit_ms / case.time_unit          # [m/s^2]
    scale = scale_augm_state(case)

    x_nd = state / scale
    u_nd = control / case.thrust_unit
    w_nd = w / accel_unit_ms2
    xdot_nd = eom_nd(case, x_nd, u_nd, 0.0) + casadi.vertcat(0.0, 0.0, 0.0, w_nd[0], w_nd[1], w_nd[2], 0.0)

    derivative_scale = casadi.vertcat(
        velocity_unit_ms, velocity_unit_ms, velocity_unit_ms,
        accel_unit_ms2, accel_unit_ms2, accel_unit_ms2,
        case.m0_wet / case.time_unit,
    )
    return derivative_scale * xdot_nd


@dataclass(frozen=True)
class BrysonRuleWeights:
    Q: np.ndarray
    Qf: np.ndarray
    R: np.ndarray
    E0: np.ndarray


def bryson_weights(case: TestCase, std0_nd: np.ndarray) -> BrysonRuleWeights:
    """Q = diag(1/(3σ)^2), Qf = 100*Q, R = diag(1/T_max^2)."""

    std0 = np.asarray(std0_nd, dtype=float) * scale_augm_state(case)
    initial_uncertainty = 3.0 * std0
    Q = np.diag(1.0 / initial_uncertainty**2)
    R = np.diag(np.full(NU, 1.0 / case.max_thrust_n**2))
    return BrysonRuleWeights(Q=Q, Qf=100.0 * Q, R=R, E0=np.diag(initial_uncertainty**2))


def load_ref_traj(case: TestCase, reference_dir: Path = REFERENCE_DIR) -> OCPSolution:
    """Reconstruct the fuel-optimal reference and convert its
    state/control/stages from the saved non dimensional units to physical units in place.
    """

    data = np.load(Path(reference_dir) / f"{case.test_case_id}.npz", allow_pickle=True)
    stages = {
        (int(k), int(j)): s
        for k, j, s in zip(data["stage_interval"], data["stage_degree_index"], data["stage_state"])
    }
    ref = OCPSolution(
        mesh=np.asarray(data["mesh_fraction"], dtype=float),
        degrees=np.asarray(data["interval_degrees"], dtype=int),
        x=np.asarray(data["x"], dtype=float),
        u=np.asarray(data["u"], dtype=float),
        sigma=np.asarray(data["sigma"], dtype=float),
        diagnostics={},
        stages=stages,
    )

    scale = scale_augm_state(case)
    ref.x = ref.x * scale[:, None]
    ref.u = ref.u * case.thrust_unit
    ref.sigma = ref.sigma * case.thrust_unit
    ref.stages = {key: value * scale for key, value in ref.stages.items()}
    return ref


def _residual_function(degree: int) -> casadi.Function:
    """Degree-th right-Radau collocation residual
    F(Y; x_k, u_k, w, h) of one interval (cached by degree)."""

    if degree in _RESIDUAL_CACHE:
        return _RESIDUAL_CACHE[degree]
    _, c_matrix, _, _ = collocation_coefficients(degree)
    xk, uk = casadi.MX.sym("xk", NX), casadi.MX.sym("uk", NU)
    w, h = casadi.MX.sym("w", NW), casadi.MX.sym("h")
    Y = casadi.MX.sym("Y", degree * NX)
    states = [xk] + [Y[(j - 1) * NX : j * NX] for j in range(1, degree + 1)]
    environment = CR3BPEarthMoon()  
    residuals = []
    for j in range(1, degree + 1):
        poly_deriv = sum(c_matrix[r, j] * states[r] for r in range(degree + 1))
        residuals.append(poly_deriv - h * eom(environment, states[j], uk, w))
    F = casadi.vertcat(*residuals)
    fn = casadi.Function(
        f"radau_residual_deg{degree}", [Y, xk, uk, w, h],
        [F, casadi.jacobian(F, Y), casadi.jacobian(F, xk), casadi.jacobian(F, uk)],
    )
    _RESIDUAL_CACHE[degree] = fn
    return fn


def get_jacobians(
    degree: int, x_k: np.ndarray, u_k: np.ndarray, interior_states: list[np.ndarray], x_next: np.ndarray, h: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Implicit differentiation (IFT) of the Radau residual at the
    already-converged collocation point (all physical units), returning
    (A, B, G) for the endpoint map x_{k+1} = Phi_k(x_k, u_k, w_k)."""

    fn = _residual_function(degree)
    Y = np.concatenate(interior_states + [x_next])
    F_ref, dFdY_ref, dFdx_ref, dFdu_ref = fn(Y, x_k, u_k, np.zeros(NW), h)
    scale_tiled = np.tile(scale_augm_state(CR3BPEarthMoon()), degree)
    resid_nd_equivalent = np.asarray(F_ref).flatten() / scale_tiled
    max_resid = float(np.max(np.abs(resid_nd_equivalent)))
    if max_resid > 1e-5:
        raise RuntimeError(f"Residual not converged at linearization point (max|F|_nd={max_resid:.3e})")

    dFdY_ref = np.asarray(dFdY_ref)
    dYdx = -np.linalg.solve(dFdY_ref, np.asarray(dFdx_ref))
    dYdu = -np.linalg.solve(dFdY_ref, np.asarray(dFdu_ref))
    selector_matrix = np.zeros((NX, NW))
    selector_matrix[3:6, :] = np.eye(NW)
    dYdw = -np.linalg.solve(dFdY_ref, -h * np.vstack([selector_matrix] * degree))
    rows = slice((degree - 1) * NX, degree * NX)
    return dYdx[rows, :], dYdu[rows, :], dYdw[rows, :]


def tvlqr(
    A_seq: list[np.ndarray], B_seq: list[np.ndarray], Q: np.ndarray, R: np.ndarray, Qf: np.ndarray
) -> list[np.ndarray]:
    """Finite-horizon discrete TVLQR backward Riccati recursion, P_H = Qf.
    Returns gains K_0..K_{H-1}: delta_u_k = -K_k @ delta_x_k."""

    K: list[np.ndarray] = [None] * len(A_seq)  
    P = Qf.copy()
    for i in reversed(range(len(A_seq))):
        A_i, B_i = A_seq[i], B_seq[i]
        K_i = np.linalg.solve(R + B_i.T @ P @ B_i, B_i.T @ P @ A_i)
        K[i] = K_i
        Acl = A_i - B_i @ K_i
        P_next = Q + K_i.T @ R @ K_i + Acl.T @ P @ Acl
        P = 0.5 * (P_next + P_next.T)
    return K


def propagate_uncertainty(
    A_seq: list[np.ndarray], B_seq: list[np.ndarray], G_seq: list[np.ndarray], K_seq: list[np.ndarray],
    E0: np.ndarray, D: np.ndarray,
) -> list[np.ndarray]:
    """DIRTREL uncertainty recursion."""

    E = [0.5 * (E0 + E0.T)]
    for i in range(len(A_seq)):
        Acl = A_seq[i] - B_seq[i] @ K_seq[i]
        E_next = Acl @ E[i] @ Acl.T
        E.append(0.5 * (E_next + E_next.T))
    return E


def propagate_nominal_traj(case: TestCase, x: np.ndarray, u: np.ndarray, h: float) -> np.ndarray:
    """Propagate the nominal trajectory."""

    scale = scale_augm_state(case)
    x_nd = x / scale
    u_nd = u / case.thrust_unit
    h_nd = h / case.time_unit
    dtau_nd = h_nd / SUBSTEPS
    x_next_nd = x_nd.copy()
    for _ in range(SUBSTEPS):
        x_next_nd = integrator.rk4(case, x_next_nd, u_nd, float(np.linalg.norm(u_nd)), dtau_nd)
    return x_next_nd * scale


def get_uncertainty_bound(case: TestCase, E: np.ndarray) -> np.ndarray:
    return np.sqrt(np.clip(np.diag(E), 0.0, None))


def monte_carlo(
    case: TestCase, ref: OCPSolution, K_seq: list[np.ndarray], std0_nd: np.ndarray, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    """Monte carlo verification."""

    scale = scale_augm_state(case)
    std0 = np.asarray(std0_nd, dtype=float) * scale
    n = ref.degrees.size
    x0_nominal = case.x0_augmented_state * scale

    final_deviations = np.zeros((n_samples, NX))
    for s in range(n_samples):
        x = x0_nominal + rng.normal(0.0, std0)
        for k in range(n):
            delta = x - ref.x[:, k]
            u = ref.u[:, k] - K_seq[k] @ delta
            h = case.tof_days * SECONDS_PER_DAY * float(ref.mesh[k + 1] - ref.mesh[k])
            x = propagate_nominal_traj(case, x, u, h)
        final_deviations[s, :] = x - ref.x[:, -1]
    return final_deviations


def print_results_table(results: list[dict[str, Any]]) -> None:

    headers = ["case", "final dr [m]", "final dv [m/s]", "fuel [kg]", "mass margin [kg]", "uncertainty bound dr [m]", "uncertainty bound dv [m/s]"]
    rows = []
    for r in results:
        pos_bound_norm = float(np.linalg.norm(r["uncertainty_bound"]["position_m"]))
        vel_bound_norm = float(np.linalg.norm(r["uncertainty_bound"]["velocity_m_per_s"]))

        rows.append([
            r["case_id"],
            f"{r['final_position_difference_m']:.2e}",
            f"{r['final_velocity_difference_m_per_s']:.3e}",
            f"{r['fuel_consumed_kg']:.3e}",
            f"{r['worst_mass_margin_kg']:.1e}",
            f"{pos_bound_norm:.2e}",
            f"{vel_bound_norm:.2e}",
        ])

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]

    def fmt_row(cells: list[str], *, right_align_numbers: bool = False) -> str:
        return "  ".join(
            cell.rjust(width) if right_align_numbers and index > 0 else cell.ljust(width)
            for index, (cell, width) in enumerate(zip(cells, widths))
        )

    print()
    print(fmt_row(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row, right_align_numbers=True))
    print()


def write_summary(
    f, case: TestCase, k: int, t_day: float, delta_u: np.ndarray, K: np.ndarray, E: np.ndarray
) -> None:
    """Write the TVLQR feedback correction, gain matrix K, and ellipsoid E
    used at one knot to an already-open text file (not the terminal). Both
    delta_u and E are already physical units (N; m,m/s,kg squared)."""

    print(f"--- knot {k:4d}  t={t_day:8.4f} d ---", file=f)
    print(
        f"  delta_u [N] = [{delta_u[0]: .4e} {delta_u[1]: .4e} {delta_u[2]: .4e}]"
        f"  |delta_u|={np.linalg.norm(delta_u):.4e} N",
        file=f,
    )
    print(f"  K ({NU}x{NX}):", file=f)
    for row in K:
        print("    [" + " ".join(f"{v: .4e}" for v in row) + "]", file=f)
    print(f"  E ({NX}x{NX}):", file=f)
    for row in E:
        print("    [" + " ".join(f"{v: .4e}" for v in row) + "]", file=f)


def plot_ellipsoid_evolution(
    case: TestCase, case_id: str, t_days: np.ndarray, E_seq: list[np.ndarray], out_dir: Path
) -> None:
    """One subplot per state channel: standard deviation (sqrt of E's
    diagonal), physical units, vs. time. Styled to match plotter.Plotter."""

    units = ["m", "m", "m", "m/s", "m/s", "m/s", "kg"]
    uncertainty_series = np.array([get_uncertainty_bound(case, E) for E in E_seq])  # (n+1, NX)

    fig, axes = plt.subplots(4, 2, figsize=(2 * Plotter.SQUARE_DIAGNOSTIC_FIGSIZE[0], 10.0), dpi=Plotter.FIGURE_DPI)
    axes_flat = axes.flatten()
    for i in range(NX):
        ax = axes_flat[i]
        ax.semilogy(t_days, uncertainty_series[:, i], color=Plotter.PURPLE, lw=Plotter.TRAJECTORY_LINE_WIDTH, label="_nolegend_")
        ax.set_xlabel("time [days]")
        ax.set_ylabel(rf"$\sqrt{{E_k[{i},{i}]}}$ [{units[i]}]")
        Plotter._style_2d_axis(Plotter, ax, tick_size=Plotter.DIAGNOSTIC_TICK_SIZE, label_size=Plotter.DIAGNOSTIC_LABEL_SIZE)
    axes_flat[-1].axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{case_id}_uncertainty_evolution.png", dpi=Plotter.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_state_control_evolution(
    case: TestCase,
    case_id: str,
    t_days: np.ndarray,
    delta_u_history: np.ndarray,
    delta_x_history: np.ndarray,
    out_dir: Path,
) -> None:
    """TVLQR feedback correction delta_u (N) and the state deviation delta_x
    = x_true - x_ref it's built from (m, m/s, kg), in one figure: delta_u,
    then delta_r, delta_v, delta_m each in their own subplot (they don't
    share units, so can't share an axis), all against the same time axis."""

    component_colors = (Plotter.BLACK, Plotter.BLUE, Plotter.GREEN, Plotter.PURPLE)
    component_linestyles = ("-", "--", "-.", "--")
    legend_kwargs = dict(loc="upper right", frameon=True, fancybox=False, edgecolor=Plotter.BLACK, facecolor="white", framealpha=1.0)

    panel_size = Plotter.SQUARE_DIAGNOSTIC_FIGSIZE[0]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(2 * panel_size, 2 * panel_size),
        dpi=Plotter.FIGURE_DPI,
        sharex=True,
    )
    axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.set_box_aspect(1)
        ax.set_xlabel("time [days]")

    ax = axes_flat[0]
    control_series = (*delta_u_history, np.linalg.norm(delta_u_history, axis=0))
    control_labels = (r"$\delta u_x$", r"$\delta u_y$", r"$\delta u_z$", r"$|\delta u|$")
    for values, label, color, linestyle in zip(
        control_series, control_labels, component_colors, component_linestyles
    ):
        ax.plot(
            t_days,
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            lw=Plotter.REFERENCE_LINE_WIDTH + 1,
        )
    ax.set_ylabel(r"$\delta u$ [N]")
    Plotter._style_2d_axis(Plotter, ax, tick_size=Plotter.DIAGNOSTIC_TICK_SIZE + 1, label_size=Plotter.DIAGNOSTIC_LABEL_SIZE + 1)
    Plotter._legend(Plotter, ax, **legend_kwargs)

    ax = axes_flat[1]
    for i, (label, color, linestyle) in enumerate(
        zip(
            (r"$\delta r_x$", r"$\delta r_y$", r"$\delta r_z$"),
            component_colors,
            component_linestyles,
        )
    ):
        ax.plot(
            t_days,
            delta_x_history[i, :],
            label=label,
            color=color,
            linestyle=linestyle,
            lw=Plotter.REFERENCE_LINE_WIDTH + 1,
        )
    ax.set_ylabel(r"$\delta r$ [m]")
    Plotter._style_2d_axis(Plotter, ax, tick_size=Plotter.DIAGNOSTIC_TICK_SIZE + 1, label_size=Plotter.DIAGNOSTIC_LABEL_SIZE + 1)
    Plotter._legend(Plotter, ax, **legend_kwargs)

    ax = axes_flat[2]
    for i, (label, color, linestyle) in enumerate(
        zip(
            (r"$\delta v_x$", r"$\delta v_y$", r"$\delta v_z$"),
            component_colors,
            component_linestyles,
        )
    ):
        ax.plot(
            t_days,
            delta_x_history[3 + i, :],
            label=label,
            color=color,
            linestyle=linestyle,
            lw=Plotter.REFERENCE_LINE_WIDTH + 1,
    )
    ax.set_ylabel(r"$\delta v$ [m/s]")
    Plotter._style_2d_axis(Plotter, ax, tick_size=Plotter.DIAGNOSTIC_TICK_SIZE + 1, label_size=Plotter.DIAGNOSTIC_LABEL_SIZE + 1)
    Plotter._legend(Plotter, ax, **legend_kwargs)

    ax = axes_flat[3]
    ax.plot(
        t_days,
        delta_x_history[6, :],
        color=component_colors[0],
        linestyle=component_linestyles[0],
        lw=Plotter.REFERENCE_LINE_WIDTH + 1,
        label=r"$\delta m$",
    )
    ax.set_ylabel(r"$\delta m$ [kg]")
    Plotter._style_2d_axis(Plotter, ax, tick_size=Plotter.DIAGNOSTIC_TICK_SIZE + 1, label_size=Plotter.DIAGNOSTIC_LABEL_SIZE + 1)
    Plotter._legend(Plotter, ax, **legend_kwargs)

    fig.tight_layout()
    fig.savefig(out_dir / f"{case_id}_control_state_deviation_evolution.png", dpi=Plotter.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def plot_montecarlo_results(
    case: TestCase, case_id: str, deltas_final: np.ndarray, E_final: np.ndarray, out_dir: Path
) -> None:
    """Scatter the Monte Carlo samples' terminal position deviation against
    the propagated ellipsoid's projection, on all three position planes
    (xy, xz, yz)."""

    axes_to_plot = ((0, 1), (0, 2), (1, 2))
    axis_names = "xyz"
    fig, axes = plt.subplots(1, 3, figsize=Plotter.TRIPLE_FIGSIZE, dpi=Plotter.FIGURE_DPI)

    for ax, (axis_0, axis_1) in zip(axes, axes_to_plot):
        E2 = E_final[np.ix_([axis_0, axis_1], [axis_0, axis_1])]
        eigval, eigvec = np.linalg.eigh(E2)
        theta = np.linspace(0.0, 2.0 * np.pi, 200)
        circle = np.vstack([np.cos(theta), np.sin(theta)])
        ellipse = eigvec @ np.diag(np.sqrt(np.clip(eigval, 0.0, None))) @ circle

        ax.scatter(
            deltas_final[:, axis_0], deltas_final[:, axis_1],
            color=Plotter.PURPLE, s=12, zorder=1.7,
            label=f"Monte Carlo samples (n={deltas_final.shape[0]})",
        )
        ax.plot(
            ellipse[0, :], ellipse[1, :], color=Plotter.BLACK, lw=Plotter.TRAJECTORY_LINE_WIDTH, zorder=4,
            label="Propagated ellipsoid boundary",
        )
        ax.set_xlabel(rf"$\delta r_{{{axis_names[axis_0]}}}$ [m]")
        ax.set_ylabel(rf"$\delta r_{{{axis_names[axis_1]}}}$ [m]")
        Plotter._style_2d_axis(Plotter, ax, equal_axis=True, tick_size=Plotter.PROJECTION_TICK_SIZE)

    Plotter._figure_legend(
        Plotter, fig, list(axes), ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, 0.92),
        frameon=True, fancybox=False, edgecolor=Plotter.BLACK, facecolor="white", framealpha=1.0,
        fontsize=7.4, columnspacing=1.0, handlelength=1.7, borderpad=0.35,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig.savefig(out_dir / f"{case_id}_montecarlo_ensemble.png", dpi=Plotter.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def run_case(case_id: str) -> dict[str, Any]:
    case = CASE_REGISTRY[case_id]()
    ref = load_ref_traj(case)
    weights = bryson_weights(case, STD_ND[case_id])
    D = np.zeros((NW, NW))
    n = ref.degrees.size

    A_seq, B_seq, G_seq = [], [], []
    for k in range(n):
        degree = int(ref.degrees[k])
        h = case.tof_days * SECONDS_PER_DAY * float(ref.mesh[k + 1] - ref.mesh[k])
        interior = [ref.stages[(k, j)] for j in range(1, degree)]
        A, B, G = get_jacobians(degree, ref.x[:, k], ref.u[:, k], interior, ref.x[:, k + 1], h)
        A_seq.append(A)
        B_seq.append(B)
        G_seq.append(G)

    K_seq = tvlqr(A_seq, B_seq, weights.Q, weights.R, weights.Qf)
    E_seq = propagate_uncertainty(A_seq, B_seq, G_seq, K_seq, weights.E0, D)

    out_dir = OUTPUT_DIR / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t_days = ref.mesh * case.tof_days
    x_true = np.zeros((NX, n + 1))
    u_applied = np.zeros((NU, n))
    delta_u_history = np.zeros((NU, n))
    delta_x_history = np.zeros((NX, n))

    scale = scale_augm_state(case)
    perturbation = 3.0 * STD_ND[case.test_case_id] * scale
    x_true[:, 0] = case.x0_augmented_state * scale + perturbation
    with open(out_dir / "summary.txt", "w") as f:
        for k in range(n):
            delta_xk = x_true[:, k] - ref.x[:, k]
            delta_uk = -K_seq[k] @ delta_xk
            print(delta_xk, file=f)
            u = ref.u[:, k] + delta_uk
            h = case.tof_days * SECONDS_PER_DAY * float(ref.mesh[k + 1] - ref.mesh[k])
            x_true[:, k + 1] = propagate_nominal_traj(case, x_true[:, k], u, h)
            u_applied[:, k] = u
            delta_u_history[:, k] = delta_uk
            delta_x_history[:, k] = delta_xk
            write_summary(f, case, k, float(t_days[k]), delta_uk, K_seq[k], E_seq[k])

    pos_diff_m = float(np.linalg.norm(x_true[0:3, -1] - ref.x[0:3, -1]))
    vel_diff_mps = float(np.linalg.norm(x_true[3:6, -1] - ref.x[3:6, -1]))
    mass_diff_kg = float(x_true[6, -1] - ref.x[6, -1])
    fuel_kg = float(x_true[6, 0] - x_true[6, -1])
    mass_margin_kg = float(np.min(x_true[6, :]) - case.m0_dry)
    uncertainty_bound = get_uncertainty_bound(case, E_seq[-1])

    rng = np.random.default_rng(MC_SEED)
    mc_deviations = monte_carlo(case, ref, K_seq, STD_ND[case_id], MC_SAMPLES, rng)
    plot_montecarlo_results(case, case_id, mc_deviations, E_seq[-1], out_dir)

    summary = {
        "case_id": case_id,
        "mission_days": float(case.tof_days),
        "final_position_difference_m": pos_diff_m,
        "final_velocity_difference_m_per_s": vel_diff_mps,
        "final_mass_difference_kg": mass_diff_kg,
        "fuel_consumed_kg": fuel_kg,
        "worst_mass_margin_kg": mass_margin_kg,
        "uncertainty_bound": {
            "position_m": uncertainty_bound[0:3].tolist(),
            "velocity_m_per_s": uncertainty_bound[3:6].tolist(),
            "mass_kg": float(uncertainty_bound[6]),
        },
        "monte_carlo_samples": MC_SAMPLES,
    }

    plot_ellipsoid_evolution(case, case_id, t_days, E_seq, out_dir)
    plot_state_control_evolution(case, case_id, t_days[:-1], delta_u_history, delta_x_history, out_dir)
    np.savez(
        out_dir / "tvlqr_dirtrel.npz",
        t_days=t_days, x_true=x_true, u_applied=u_applied,
        K_used=np.stack(K_seq), E_used=np.stack(E_seq), mc_deltas_final=mc_deviations,
    )
    (out_dir / "tvlqr_dirtrel.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    results = [run_case(case_id) for case_id in CASE_REGISTRY]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {r["case_id"]: r for r in results}
    (OUTPUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print_results_table(results)


if __name__ == "__main__":
    main()
