"""Psyche stochastic figures in the shared Plotter style of the other cases.

Dispersion evolution, thrust budget, heliocentric projection and terminal
distribution follow ``plotter.stochastic.plot_outputs`` and
``plotter.stochastic.plot_earth_mars_outputs``. The Mars B-plane figure is Psyche-specific.
"""

from pathlib import Path
import shutil
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from plotter import Plotter
from verification.stochastic.psyche_monte_carlo import nominal_history

AU = 149597870700.0
# Two-dimensional 99.9% Gaussian ellipse, sqrt(-2 ln 0.001).
BPLANE_SIGMA = float(np.sqrt(-2 * np.log(0.001)))
BPLANE_ALTITUDE_LIMIT_KM = 500.0


def plot_diagnostics(data, arrays, report, output, ensembles=()):
    if shutil.which("latex") is None:
        # Same Computer Modern look without a TeX installation (e.g. WSL).
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.serif": ["cmr10"],
                "mathtext.fontset": "cm",
                "axes.formatter.use_mathtext": True,
            }
        )
    plotter = Plotter(Path(output) / "psyche")
    ensemble = ensembles[0] if ensembles else None
    days = (arrays["epochs_et"] - data.mission.t0) / 86400
    plot_dispersion_evolution(plotter, data, arrays, days)
    plot_thrust_budget(plotter, data, arrays, report, days, ensemble)
    plot_projection(plotter, data, arrays, days, ensemble)
    if ensemble is not None:
        plot_terminal_distribution(plotter, data, arrays, ensemble)
    plot_bplane(plotter, data, arrays, report, ensemble)


def save(plotter, figure, name):
    figure.savefig(plotter._path(name), dpi=Plotter.FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def boxed_legend(plotter, axis, **kwargs):
    plotter._legend(
        axis,
        frameon=True,
        fancybox=False,
        edgecolor=Plotter.BLACK,
        facecolor="white",
        framealpha=1.0,
        **kwargs,
    )


def target_std_si(data):
    return data.covariance_scale[:6] / data.terminal_inverse_std


def heliocentric_nodes(data, arrays):
    """Node means in the Sun-centred frame (Mars-phase nodes are Mars-relative)."""
    states = arrays["states_si"].copy()
    for node, origin in enumerate(data.node_origins):
        if origin == 1:
            states[node, :6] += data.mission.cache.state(499, arrays["epochs_et"][node])
    return states


def plot_dispersion_evolution(plotter, data, arrays, days):
    sigma = np.sqrt(
        np.maximum(np.diagonal(arrays["covariance_si"], axis1=1, axis2=2), 0.0)
    )
    target = target_std_si(data)
    figure, axes = plt.subplots(
        1,
        3,
        figsize=Plotter.THREE_PANEL_FIGSIZE,
        dpi=Plotter.FIGURE_DPI,
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.0)},
    )
    colors = (Plotter.BLUE, Plotter.RED, Plotter.GREY)
    panels = (
        (slice(0, 3), r"$\sigma$ [m]", (r"$x$", r"$y$", r"$z$"), colors),
        (
            slice(3, 6),
            r"$\sigma$ [m/s]",
            (r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"),
            colors,
        ),
        (slice(6, 7), r"$\sigma$ [kg]", (r"$m$",), (Plotter.GREY,)),
    )
    for axis, (columns, label, names, colors) in zip(axes, panels):
        axis.set_box_aspect(1.0)
        for offset, (name, color) in enumerate(zip(names, colors)):
            axis.semilogy(
                days,
                np.maximum(sigma[:, columns][:, offset], 1e-16),
                lw=Plotter.REFERENCE_LINE_WIDTH,
                color=color,
                label=name,
            )
        if columns.start < 6:
            axis.axhline(
                target[columns.start],
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
        boxed_legend(
            plotter, axis, loc="best", fontsize=7.0, borderpad=0.28, handlelength=1.4
        )
    figure.align_ylabels(axes)
    figure.tight_layout()
    save(plotter, figure, "dispersion_evolution")


def plot_thrust_budget(plotter, data, arrays, report, days, ensemble):
    thrust_max = data.mission.settings.thrust_max_n
    nominal = np.linalg.norm(arrays["interval_controls"], axis=1) * thrust_max
    bound = np.asarray(report["control_dispersion_bound_by_interval"]) * thrust_max
    step_days, nominal_step = plotter.plot_zero_order_hold_data(days, nominal)
    _, bound_step = plotter.plot_zero_order_hold_data(days, bound)
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(1.5 * Plotter.WIDE_FIGSIZE[0], 1.5 * Plotter.WIDE_FIGSIZE[1]),
        dpi=Plotter.FIGURE_DPI,
    )
    legend = dict(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        fontsize=8,
        borderaxespad=0.0,
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
    axis.axhline(thrust_max, color=Plotter.RED, ls=":", lw=0.88, label="Maximum thrust")
    axis.set_xlabel("time [days]")
    axis.set_ylabel("thrust [N]")
    axis.set_ylim(0.0, 1.06 * max(thrust_max, float(np.max(nominal))))
    plotter._style_2d_axis(
        axis,
        tick_size=Plotter.DIAGNOSTIC_TICK_SIZE,
        label_size=Plotter.DIAGNOSTIC_LABEL_SIZE,
    )
    boxed_legend(plotter, axis, **legend)
    axis = axes[1]
    peak = float(np.max(bound))
    if ensemble is not None:
        deviation = ensemble["plot_applied_control_deviation_norm_n"]
        _, deviation_step = plotter.plot_zero_order_hold_data(days, deviation)
        lines = axis.plot(
            step_days,
            deviation_step,
            color=Plotter.DARK_GREY,
            alpha=0.28,
            lw=0.48,
            zorder=1,
        )
        lines[0].set_label("Monte Carlo")
        peak = max(peak, float(np.max(deviation)))
    axis.plot(
        step_days,
        bound_step,
        color=Plotter.BLUE,
        ls="--",
        lw=0.82,
        label="Predicted bound",
        zorder=2,
    )
    axis.set_xlabel("time [days]")
    axis.set_ylabel(r"$\|T_{\mathrm{applied}}-\bar{T}\|$ [N]")
    axis.set_ylim(0.0, 1.05 * peak)
    plotter._style_2d_axis(
        axis,
        tick_size=Plotter.DIAGNOSTIC_TICK_SIZE,
        label_size=Plotter.DIAGNOSTIC_LABEL_SIZE,
    )
    boxed_legend(plotter, axis, **legend)
    figure.tight_layout()
    save(plotter, figure, "thrust_budget")


# Projection magnification of the Monte Carlo paths and the predicted 3-sigma covariance ellipses.
PROJECTION_MAGNIFICATION = 800.0


def plot_projection(plotter, data, arrays, days, ensemble):
    nodes = heliocentric_nodes(data, arrays)[:, :3].T / AU
    # Dense mean-path epochs must contain the Monte Carlo substep epochs.
    substeps = (
        1
        if ensemble is None
        else (len(ensemble["plot_epochs_et"]) - 1) // len(data.arcs)
    )
    dense_et, dense = nominal_history(
        data, arrays, points_per_arc=substeps * int(np.ceil(16 / substeps))
    )
    dense = dense[:3] / AU
    magnification = PROJECTION_MAGNIFICATION
    scale_label = plotter.plot_magnification_label(magnification)
    figure, axis = plt.subplots(figsize=Plotter.SINGLE_FIGSIZE, dpi=Plotter.FIGURE_DPI)
    epochs = np.linspace(data.mission.t0, data.mission.tf, 1200)
    for body, name, color in (
        (399, "Earth", Plotter.EARTH_COLOR),
        (499, "Mars", Plotter.MARS_COLOR),
    ):
        track = data.mission.cache.state(body, epochs).T[:3] / AU
        axis.plot(
            track[0],
            track[1],
            "--",
            color=color,
            lw=Plotter.REFERENCE_LINE_WIDTH,
            label=f"{name} orbit",
            zorder=2,
        )
    if ensemble is not None:
        # Deviations at the Monte Carlo substeps, interpolated onto the dense
        # mean path (whose epochs contain the substep epochs).
        history_et = ensemble["plot_epochs_et"]
        stride = (len(dense_et) - 1) // (len(history_et) - 1)
        deviation = (
            ensemble["plot_trajectory_history_si"][:2] / AU - dense[:2, None, ::stride]
        )
        displayed = np.array(
            [
                [
                    dense[i] + magnification * np.interp(dense_et, history_et, sample)
                    for sample in deviation[i]
                ]
                for i in range(2)
            ]
        )
        lines = axis.plot(
            displayed[0].T,
            displayed[1].T,
            color=Plotter.DARK_GREY,
            alpha=0.20,
            lw=0.4,
            zorder=3,
        )
        lines[0].set_label(f"Monte Carlo ({scale_label})")
    axis.plot(
        dense[0],
        dense[1],
        color=Plotter.BLACK,
        lw=Plotter.TRAJECTORY_LINE_WIDTH,
        label="Transfer",
        zorder=4,
    )
    for point, name, color in (
        (np.zeros(3), "Sun", Plotter.SUN_COLOR),
        (nodes[:, 0], "Earth SOI exit", Plotter.EARTH_COLOR),
        (nodes[:, -1], "Psyche", Plotter.GREEN),
    ):
        axis.scatter(
            *point[:2],
            s=58 / (1.5 if name == "Sun" else 2.5) ** 2,
            color=color,
            edgecolor=Plotter.BLACK,
            linewidth=0.6,
            label=name,
            zorder=7,
        )
    # Psyche's deterministic arrow convention: 0.48 AU at full throttle, 60 samples.
    thrust = arrays["interval_controls"].T
    index = np.unique(np.searchsorted(days[:-1], np.linspace(days[0], days[-2], 60)))
    index = index[np.linalg.norm(thrust[:, index], axis=0) > 0.03]
    arrows = 0.48 * thrust[:, index]
    axis.quiver(
        nodes[0, index],
        nodes[1, index],
        arrows[0],
        arrows[1],
        angles="xy",
        scale_units="xy",
        scale=1,
        color=Plotter.THRUST_ARROW_3D_COLOR,
        width=0.0046,
        alpha=0.82,
        zorder=6,
        label="Thrust",
    )
    for node in range(len(days)):
        covariance = arrays["covariance_si"][node, :2, :2] / AU**2
        points = magnification * plotter.plot_covariance_ellipse_points(covariance, 3.0)
        axis.plot(
            nodes[0, node] + points[0],
            nodes[1, node] + points[1],
            color=Plotter.DARK_BLUE,
            lw=0.80,
            alpha=0.90,
            zorder=5,
            label=f"Predicted covariance ({scale_label})"
            if node == 0
            else "_nolegend_",
        )
    axis.set(xlabel="x [AU]", ylabel="y [AU]")
    plotter._style_2d_axis(axis, equal_axis=True)
    axis.set_axisbelow(True)
    axis.margins(0.14)
    figure.tight_layout(rect=(0, 0, 1, 0.73))
    plotter._heliocentric_legend(figure, axis)
    save(plotter, figure, "projections")


def plot_terminal_distribution(plotter, data, arrays, ensemble):
    """Shared corner plot, in units of the terminal target standard deviation."""
    target = target_std_si(data)
    terminal = ensemble["terminal_si"][:, :6]
    terminal = terminal[np.isfinite(terminal).all(axis=1)]
    covariance = arrays["covariance_si"][-1, :6, :6] / np.outer(target, target)
    plotter.plot_terminal_distribution(
        SimpleNamespace(length_unit=1e-3, velocity_unit=1e-3),  # m -> km and m/s
        SimpleNamespace(
            position_covariance_reduction=1.0, velocity_covariance_reduction=1.0
        ),
        SimpleNamespace(gains=np.zeros((1, 3, 6)), covariances=covariance[None]),
        SimpleNamespace(scale=target),
        dict(terminal_deviation=((terminal - arrays["states_si"][-1, :6]) / target).T),
    )


def plot_bplane(plotter, data, arrays, report, ensemble):
    """Plot the focusing-scaled B-plane and the 500 km clearance circle.

    Each sample uses its own asymptotic speed and closest-approach plane.
    The impact circle uses the nominal asymptotic speed.
    """
    mu = data.mission.scales[1].mu
    radius = data.mission.settings.mars_radius_m
    rho = radius + BPLANE_ALTITUDE_LIMIT_KM * 1e3
    encounter = report["encounter"]
    center = np.asarray(encounter["b_si"]) / 1e3
    predicted = np.asarray(encounter["b_covariance_si"]) / 1e6
    c3 = float(np.asarray(encounter["domain"])[0])
    if not (np.isfinite(center).all() and np.isfinite(predicted).all() and c3 > 0):
        print(
            "B-plane figure skipped: the mean closest approach is not a valid hyperbola.",
            flush=True,
        )
        return
    focusing = lambda r: np.sqrt(1 + 2 * mu / (r * c3))
    figure, axis = plt.subplots(figsize=Plotter.WIDE_FIGSIZE, dpi=Plotter.FIGURE_DPI)
    impact = radius * focusing(radius) / focusing(rho) / 1e3
    axis.add_patch(
        Circle(
            (0, 0),
            impact,
            facecolor=Plotter.LIGHT_GREY,
            edgecolor="none",
            alpha=0.45,
            zorder=0,
        )
    )
    angle = np.linspace(0, 2 * np.pi, 4000)
    axis.plot(
        impact * np.cos(angle),
        impact * np.sin(angle),
        color=Plotter.BLACK,
        lw=1.0,
        label=r"Impact Limit ($h=0$)",
        zorder=2,
    )
    limit = rho / 1e3
    axis.plot(
        limit * np.cos(angle),
        limit * np.sin(angle),
        color=Plotter.GREEN,
        ls="--",
        lw=0.9,
        label=rf"$h = {BPLANE_ALTITUDE_LIMIT_KM:.0f}$ km ($B_{{SC}}=\rho$)",
        zorder=2,
    )
    if ensemble is not None:
        samples = ensemble["bplane_si"] / 1e3
        samples = samples[np.isfinite(samples).all(axis=1)]
        # Same sample style as the terminal-distribution corner plot.
        axis.scatter(
            samples[:, 0],
            samples[:, 1],
            s=1.0,
            color=Plotter.GREY,
            alpha=0.12,
            linewidths=0.0,
            rasterized=True,
            zorder=3,
        )
        # Opaque legend proxy: a 12 % alpha point would be invisible in the legend.
        axis.plot(
            [], [], "o", color=Plotter.GREY, ms=2.5, ls="none", label="Monte Carlo"
        )
    percent = r"\%" if plt.rcParams["text.usetex"] else "%"
    curves = [
        (center, predicted, Plotter.RED, "-", f"Predicted Covariance (99.9{percent})")
    ]
    if ensemble is not None:
        curves.append(
            (
                samples.mean(axis=0),
                np.cov(samples, rowvar=False),
                Plotter.DARK_BLUE,
                "-.",
                f"Sample Covariance (99.9{percent})",
            )
        )
    view = [center[:, None]]
    for mean, covariance, color, style, label in curves:
        points = mean[:, None] + plotter.plot_covariance_ellipse_points(
            covariance, BPLANE_SIGMA, samples=400
        )
        axis.plot(*points, color=color, ls=style, lw=1.1, label=label, zorder=4)
        view.append(points)
    if ensemble is not None:
        view.append(samples.T)
    # Zoom on the dispersion, keeping the nearest points of both limit circles.
    direction = center / np.linalg.norm(center)
    view += [(impact * direction)[:, None], (limit * direction)[:, None]]
    view = np.hstack(view)
    low, high = view.min(axis=1), view.max(axis=1)
    middle, half = (low + high) / 2, 0.54 * (high - low)
    # Use upright labels so math mode does not add spacing around the dot.
    axis.set_xlabel(r"B.R$_{SC}(\rho)$ [km]")
    axis.set_ylabel(r"B.T$_{SC}(\rho)$ [km]")
    plotter._style_2d_axis(
        axis,
        tick_size=Plotter.DIAGNOSTIC_TICK_SIZE,
        label_size=Plotter.DIAGNOSTIC_LABEL_SIZE,
    )
    figure.tight_layout()
    # Equal scales: widen the shorter range to the axes' own aspect ratio.
    box = axis.get_position()
    ratio = box.height * figure.get_figheight() / (box.width * figure.get_figwidth())
    half = np.maximum(half, half[::-1] * np.array([1 / ratio, ratio]))
    axis.set_xlim(middle[0] - half[0], middle[0] + half[0])
    axis.set_ylim(middle[1] - half[1], middle[1] + half[1])
    axis.set_aspect("equal", adjustable="box")
    boxed_legend(
        plotter, axis, loc="best", fontsize=7.5, borderpad=0.35, handlelength=1.8
    )
    save(plotter, figure, "bplane")
