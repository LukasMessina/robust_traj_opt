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


def plot_history(plotter, directory, history, tolerance, psyche=False):
    fig, ax = plt.subplots(figsize=plotter.SQUARE_DIAGNOSTIC_FIGSIZE)
    meshes = []
    defects = []
    for stage in history:
        if stage["method"] != "shooting":
            continue
        meshes.append(sum(stage["intervals"]) if psyche else stage["arcs"])
        if psyche:
            file = directory / (stage["name"] + "_defects.json")
            defects.append(
                max(p["max_relative"] for p in read(file)["interval_defects"])
                if file.exists()
                else np.nan
            )
        else:
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
        folder = (
            directory / "checkpoints" / stage["name"]
            if psyche
            else directory / stage["stage_directory"] / "checkpoints"
        )
        path = folder / "progress.jsonl"
        if not path.exists():
            continue
        points = [json.loads(line) for line in path.read_text().splitlines() if line]
        # Callbacks are sampled and their count can differ from IPOPT iter_count.
        # Append the actual saved endpoint instead of presenting the last sample
        # as the final solver result. This changes only the visualisation.
        details = read(directory / (stage["name"] + ".json")) if psyche else stage
        endpoint_index = max(stage["iterations"], max(p["iteration"] for p in points) + 1)
        endpoint = dict(
            iteration=endpoint_index,
            objective=details["last_ipopt_metrics"]["obj"],
            max_constraint_violation=details[
                "max_scaled_constraint_violation" if psyche else "max_constraint_violation"
            ],
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


def plot_psyche(directory):
    from .psyche import Mission, Settings, load_solution, evaluate_phase

    directory = Path(directory)
    result, config = read(directory / "result.json"), read(directory / "configuration.json")
    mission = Mission(Settings(**config["settings"]))
    solution = load_solution(directory / Path(result["last_candidate"]).name)
    plotter = Plotter(directory / "trajectory")
    all_times = []
    all_thrust = []
    all_sun = []
    phase_points = []
    for i, phase in enumerate(solution["phases"]):
        points = []
        for k, h in enumerate(np.diff(phase["mesh"])):
            for theta in np.linspace(0.0, 1.0, 9):
                tau = phase["mesh"][k] + theta * h
                state = evaluate_phase(solution, i, tau, mission)
                epoch = phase["start_et"] + phase["duration_s"] * tau
                points.append(state[:3] / 1000.0)
                sun = state[:3].copy()
                if i in (0, 2):
                    sun += mission.cache.state(399 if i == 0 else 499, epoch)[:3]
                all_sun.append(sun / 149597870700.0)
                all_times.append((epoch - mission.t0) / 86400.0)
                all_thrust.append(phase["thrust_n"][k])
        phase_points.append(np.asarray(points))
    t, u, x = np.asarray(all_times), np.asarray(all_thrust).T, np.asarray(all_sun).T
    plotter.plot_inertial_control(t, u, mission.settings.thrust_max_n)
    for dimensions in (2, 3):
        fig = plt.figure(
            figsize=plotter.SINGLE_FIGSIZE if dimensions == 2 else plotter.THREE_D_FIGSIZE
        )
        ax = fig.add_subplot(111, projection="3d" if dimensions == 3 else None)
        for body, label, color in (
            (399, "Earth", plotter.EARTH_COLOR),
            (499, "Mars", plotter.MARS_COLOR),
        ):
            track = (
                mission.cache.state(body, np.linspace(mission.t0, mission.tf, 1200))[:, :3].T
                / 149597870700.0
            )
            ax.plot(
                *track[:dimensions],
                ls="--",
                color=color,
                lw=plotter.REFERENCE_LINE_WIDTH,
                label=label + " orbit",
            )
        ax.plot(
            *x[:dimensions],
            color=plotter.TRAJECTORY_COLOR,
            lw=plotter.TRAJECTORY_LINE_WIDTH,
            label="Transfer",
        )
        for point, label, color in (
            (np.zeros(3), "Sun", plotter.SUN_COLOR),
            (x[:, 0], "TIP", plotter.EARTH_COLOR),
            (x[:, -1], "Psyche", plotter.GREEN),
        ):
            ax.scatter(
                *point[:dimensions],
                s=plotter.SMALL_MARKER_SIZE,
                color=color,
                edgecolor=plotter.BLACK,
                linewidth=0.6,
                label=label,
                zorder=6,
            )
        ax.set(xlabel="x [AU]", ylabel="y [AU]")
        if dimensions == 3:
            ax.set_zlabel("z [AU]")
            plotter._style_3d_axis(ax)
            ax.view_init(azim=-60, elev=30)
            plotter._set_axes_equal_3d(ax)
            plotter._plot_trajectory_projections_3d(ax, x, xz_far_side=True)
            fig.subplots_adjust(left=0.02, right=0.82, bottom=0.10, top=0.73)
        else:
            plotter._style_2d_axis(ax, equal_axis=True)
            ax.margins(0.12)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.76))
        plotter._heliocentric_legend(fig, ax, gap=0.003 if dimensions == 3 else 0.02)
        save(plotter, fig, "3d" if dimensions == 3 else "projections")
    for i, label in ((0, "Earth"), (2, "Mars")):
        fig, ax = plt.subplots(figsize=plotter.SINGLE_FIGSIZE)
        points = phase_points[i]
        ax.plot(points[:, 0], points[:, 1], color=plotter.BLACK, lw=plotter.TRAJECTORY_LINE_WIDTH)
        radius = mission.settings.earth_radius_m if i == 0 else mission.settings.mars_radius_m
        ax.add_patch(
            plt.Circle((0, 0), radius / 1000.0, color=plotter.GREY, label=label + " surface")
        )
        style(plotter, ax, "x [km]", "y [km]")
        ax.set_aspect("equal", adjustable="box")
        plotter._legend(ax, loc="best")
        fig.tight_layout()
        save(plotter, fig, label.lower() + "_passage")
    defects = read(directory / (Path(result["last_candidate"]).stem + "_defects.json"))[
        "interval_defects"
    ]
    edges = []
    values = []
    for phase, defect in zip(solution["phases"], defects):
        edges.extend(
            (
                (phase["start_et"] - mission.t0 + phase["mesh"][:-1] * phase["duration_s"])
                / (mission.tf - mission.t0)
            ).tolist()
        )
        values.extend(defect["relative"])
    edges.append(1.0)
    plotter.plot_interval_defect(
        np.asarray(edges),
        (mission.tf - mission.t0) / 86400.0,
        np.asarray(values),
        mission.settings.defect_tolerance,
    )
    diagpath = directory / (Path(result["last_candidate"]).stem + "_reintegration.npz")
    if diagpath.exists():
        with np.load(diagpath) as data:
            fig, axes = plt.subplots(1, 2, figsize=plotter.THREE_PANEL_FIGSIZE)
            for i in range(4):
                epoch = (data[f"p{i}_epochs_et"] - mission.t0) / 86400.0
                for ax, key, color in zip(
                    axes, ("position_error_m", "velocity_error_m_s"), (plotter.BLACK, plotter.BLUE)
                ):
                    ax.semilogy(
                        epoch,
                        np.maximum(data[f"p{i}_{key}"], 1e-16),
                        color=color,
                        lw=plotter.TRAJECTORY_LINE_WIDTH,
                    )
            for ax, ylabel in zip(axes, ("position error [m]", "velocity error [m/s]")):
                style(plotter, ax, "time [days]", ylabel)
            fig.tight_layout()
            save(plotter, fig, "verification_errors")
    plot_history(
        plotter, directory, result["history"], mission.settings.defect_tolerance, psyche=True
    )
    write_json(
        directory / "plot_metadata.json",
        dict(
            style_source="plotter.py",
            text_usetex=plt.rcParams["text.usetex"],
            diagnostic_integration="Standard DOP853 history; tighter/tightest endpoint checks in global_reference_accuracy.json",
            trajectory_interpolation="RK7 local arc samples, cached origin translations; plotting only",
            format="PNG",
            dpi=Plotter.FIGURE_DPI,
        ),
    )


def regenerate(directory, case):
    if case == "psyche":
        return plot_psyche(directory)
    from .single_phase import ShootingProblem

    config = read(directory / "configuration.json")
    result = read(directory / "result.json")
    with np.load(directory / "solution.npz") as data:
        arrays = dict(data)
    problem = ShootingProblem(
        models.DETERMINISTIC_CASE_REGISTRY[case](),
        config["objective"],
        config["ipopt_options"],
        mesh=arrays["mesh"],
        incoming=(dict(result, converged=True, method="shooting"), arrays),
    )
    plot_single_phase(problem, arrays, directory, result)
