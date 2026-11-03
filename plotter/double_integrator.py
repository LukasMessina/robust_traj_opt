"""Trajectory dispersion and control plots for the double integrator."""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
import matplotlib.pyplot as plt
from . import Plotter

if TYPE_CHECKING:
    from stochastic_trajectory_optimization.double_integrator import (
        Options,
        RobustTrajectory,
    )


def covariance_ellipse_points(
    covariance: np.ndarray, n_sigma: float, n_points: int = 90
) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (covariance + covariance.T))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    angle = np.linspace(0.0, 2.0 * np.pi, n_points)
    circle = np.stack([np.cos(angle), np.sin(angle)])
    return n_sigma * (eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ circle)


@plt.rc_context(
    {"text.usetex": False, "font.serif": ["cmr10"], "axes.formatter.use_mathtext": True}
)
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
    axis.fill_between(
        x1, np.minimum(y_line1, y_line2), 8.0, color=Plotter.LIGHT_GREY, zorder=0
    )
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
        means[0],
        means[1],
        "k-+",
        lw=Plotter.TRAJECTORY_LINE_WIDTH,
        markersize=4,
        label=r"$\bar{x}(t_k)$",
        zorder=3,
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
            means[0, :-1],
            means[1, :-1],
            feedforward[0],
            feedforward[1],
            angles="xy",
            scale_units="xy",
            scale=2.0,
            color=Plotter.DARK_BLUE,
            width=0.004,
            pivot="tail",
            label="_nolegend_",
            zorder=6,
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
        axis,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=3,
        frameon=True,
        fancybox=False,
        edgecolor=Plotter.BLACK,
        facecolor="white",
        framealpha=1.0,
        borderpad=0.30,
        columnspacing=0.85,
        handlelength=1.5,
    )
    figure.subplots_adjust(left=0.16, right=0.95, bottom=0.12, top=0.82)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=Plotter.FIGURE_DPI)
    plt.close(figure)


@plt.rc_context(
    {"text.usetex": False, "font.serif": ["cmr10"], "axes.formatter.use_mathtext": True}
)
def plot_control_profiles(
    options: Options,
    solution: RobustTrajectory,
    magnitude_path: Path,
    covariance_path: Path,
) -> None:
    """Plot the feedforward norm and the 3-sigma control ellipses about each feedforward vector."""

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
        control_norms,
        time_edges,
        baseline=None,
        color=Plotter.BLACK,
        lw=Plotter.TRAJECTORY_LINE_WIDTH,
        label=r"$\|\bar{u}(t_k)\|$",
    )
    magnitude_axis.axhline(
        options.u_max,
        color=Plotter.RED,
        linestyle="--",
        lw=Plotter.GUIDE_LINE_WIDTH,
        label=r"$u_{\max}$",
    )
    magnitude_axis.set_xlim(time_edges[0], time_edges[-1])
    magnitude_axis.set_ylim(
        -0.05 * options.u_max, 1.1 * max(options.u_max, np.max(control_norms))
    )
    magnitude_axis.set_xlabel("time [s]")
    magnitude_axis.set_ylabel(r"$\|\bar{u}(t_k)\|$ [m/s$^2$]")

    covariance_axis.plot(
        solution.feedforward[0],
        solution.feedforward[1],
        "k-+",
        lw=Plotter.TRAJECTORY_LINE_WIDTH,
        markersize=5,
        label=r"$\bar{u}(t_k)$",
        zorder=3,
    )
    extent = options.u_max
    for k, control_covariance in enumerate(solution.control_covariances):
        ellipse = covariance_ellipse_points(control_covariance, 3.0)
        ellipse = ellipse + solution.feedforward[:, k : k + 1]
        covariance_axis.plot(
            ellipse[0],
            ellipse[1],
            color=Plotter.DARK_GREY,
            lw=0.7,
            label=r"$\Sigma_{u(t_k)}$" if k == 0 else None,
            zorder=2,
        )
        extent = max(extent, float(np.max(np.abs(ellipse))))

    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    covariance_axis.plot(
        options.u_max * np.cos(angle),
        options.u_max * np.sin(angle),
        color=Plotter.RED,
        linestyle="--",
        lw=Plotter.GUIDE_LINE_WIDTH,
        label=r"$u_{\max}$",
        zorder=1,
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
            axis,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.03),
            ncol=2,
            frameon=True,
            fancybox=False,
            edgecolor=Plotter.BLACK,
            facecolor="white",
            framealpha=1.0,
            borderpad=0.30,
            columnspacing=0.85,
            handlelength=1.5,
        )
        figure.subplots_adjust(left=0.16, right=0.95, bottom=0.12, top=0.82)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_path, dpi=Plotter.FIGURE_DPI)
        plt.close(figure)
