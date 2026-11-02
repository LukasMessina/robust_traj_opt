"""Deterministic results using the original Plotter typography and geometry.

Plot interpolation never enters the NLP, defect check, or acceptance decision.
"""

import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plotter import Plotter

from .io import write_json
from . import models


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(plotter, fig, suffix):
    path = plotter._path(suffix)
    fig.savefig(path, dpi=plotter.FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def style(plotter, ax, xlabel, ylabel):
    ax.set(xlabel=xlabel, ylabel=ylabel)
    plotter._style_2d_axis(
        ax, tick_size=plotter.DIAGNOSTIC_TICK_SIZE, label_size=plotter.DIAGNOSTIC_LABEL_SIZE
    )


def plot_history(plotter, directory, history, tolerance):
    fig, ax = plt.subplots(figsize=plotter.SQUARE_DIAGNOSTIC_FIGSIZE)
    meshes = []
    defects = []
    for stage in history:
        if stage["method"] != "shooting":
            continue
        meshes.append(stage["arcs"])
        defects.append(stage.get("max_relative_local_defect", np.nan))
    ax.semilogy(meshes, defects, "o-", color=plotter.BLACK, lw=plotter.TRAJECTORY_LINE_WIDTH, ms=3)
    ax.axhline(tolerance, ls=":", color=plotter.RED, lw=plotter.GUIDE_LINE_WIDTH, label="Tolerance")
    style(plotter, ax, "number of shooting arcs", "maximum relative local defect")
    plotter._legend(ax, loc="best")
    fig.tight_layout()
    save(plotter, fig, "refinement")
    fig, axes = plt.subplots(2, 1, figsize=(plotter.WIDE_FIGSIZE[0], 5.7))
    offset = 0
    for index, stage in enumerate(history):
        path = directory / stage["stage_directory"] / "checkpoints" / "progress.jsonl"
        if not path.exists():
            continue
        points = [json.loads(line) for line in path.read_text().splitlines() if line]
        # Callbacks are sampled and their count can differ from IPOPT iter_count.
        # Append the actual saved endpoint instead of presenting the last sample
        # as the final solver result. This changes only the visualisation.
        endpoint_index = max(stage["iterations"], max(p["iteration"] for p in points) + 1)
        endpoint = dict(
            iteration=endpoint_index,
            objective=stage["last_ipopt_metrics"]["obj"],
            max_constraint_violation=stage["max_constraint_violation"],
        )
        points.append(endpoint)
        color = (plotter.BLACK, plotter.BLUE, plotter.RED, plotter.GREEN)[index % 4]
        steps = [offset + p["iteration"] for p in points]
        axes[0].plot(
            steps,
            [p["objective"] for p in points],
            color=color,
            lw=plotter.REFERENCE_LINE_WIDTH,
            label="HS" if index == 0 else f"MS {index-1}",
        )
        axes[1].semilogy(
            steps,
            [max(p["max_constraint_violation"], 1e-18) for p in points],
            color=color,
            lw=plotter.REFERENCE_LINE_WIDTH,
        )
        axes[0].plot(steps[-1], endpoint["objective"], "o", color=color, ms=3)
        axes[1].plot(steps[-1], max(endpoint["max_constraint_violation"], 1e-18),
                     "o", color=color, ms=3)
        offset += endpoint_index + 1
    style(plotter, axes[0], "solver progress (stages concatenated)", "scaled objective")
    style(plotter, axes[1], "solver progress (stages concatenated)", "constraint violation")
    plotter._legend(axes[0], loc="best", ncol=2)
    axes[1].axhline(1e-10, color=plotter.RED, ls=":", lw=plotter.GUIDE_LINE_WIDTH)
    fig.tight_layout()
    save(plotter, fig, "solver_convergence")


def plot_single_phase(problem, arrays, directory, result):
    from .single_phase import is_cr3bp, moon_clearance_km

    directory = Path(directory)
    plotter = Plotter(directory / "trajectory")
    case, mesh = problem.case, arrays["mesh"]
    times = []
    states = []
    controls = []
    for k, h in enumerate(np.diff(mesh)):
        for theta in np.linspace(0.0, 1.0, 9):
            if theta == 0.0:
                x = arrays["states_nd"][:, k]
            elif theta == 1.0:
                x = arrays["states_nd"][:, k + 1]
            else:
                x = np.asarray(
                    problem.flow(
                        arrays["states_nd"][:, k],
                        arrays["throttle"][k] * arrays["direction"][:, k],
                        arrays["throttle"][k],
                        case.tof_nd * h * theta,
                    )[0]
                ).ravel()
            states.append(x)
            controls.append(arrays["thrust_n"][:, k] / case.thrust_unit)
            times.append(case.tof_days * (mesh[k] + h * theta))
    x, u, t = np.asarray(states).T, np.asarray(controls).T, np.asarray(times)
    if is_cr3bp(case):
        departure = (
            models.propagate_periodic_orbit(case, case.x0_augmented_state, case.departure_period_nd)
            if case.departure_period_nd
            else None
        )
        target = (
            models.propagate_periodic_orbit(case, case.xf_augmented_state, case.target_period_nd)
            if case.target_period_nd
            else None
        )
        lagrange = models.get_collinear_lagrange_points(case)
        plotter.plot_traj_projection(case, x, u, departure, target, lagrange)
        plotter.plot_3d_traj(case, x, u, departure, target, lagrange)
        plotter.plot_thrust_time_evolution(
            t, np.linalg.norm(u, axis=0) * case.thrust_unit, case.max_thrust_n
        )
        plotter.plot_moon_distance_time_evolution(t, moon_clearance_km(case, x))
    else:
        plotter.plot_heliocentric_transfer(case, t, x, u)
        plotter.plot_heliocentric_3d(case, t, x, u)
        plotter.plot_inertial_control(t, u * case.thrust_unit, case.max_thrust_n)
    diagnostics = directory / "integration_diagnostics.npz"
    if diagnostics.exists():
        with np.load(diagnostics) as data:
            error = (data["tightest_states_nd"] - arrays["states_nd"]) * problem.mission.scales[
                0
            ].state[:, None]
            plotter.plot_collocation_approx_errors(
                mesh * case.tof_days,
                np.linalg.norm(error[:3], axis=0),
                np.linalg.norm(error[3:6], axis=0),
                abs(error[6]),
            )
            tolerance = read(directory / "configuration.json")["defect_tolerance"]
            plotter.plot_interval_defect(
                mesh, case.tof_days, data["local_relative_defects"], tolerance
            )
    else:
        tolerance = read(directory / "configuration.json")["defect_tolerance"]
    plot_history(plotter, directory, result["history"], tolerance)
    write_json(
        directory / "plot_metadata.json",
        dict(
            style_source="plotter.py",
            text_usetex=plt.rcParams["text.usetex"],
            diagnostic_integration="DOP853 tightest reference; nodal continuous discrepancies",
            trajectory_interpolation="RK7 local arc samples for plotting only",
            format="PNG",
            dpi=Plotter.FIGURE_DPI,
        ),
    )
