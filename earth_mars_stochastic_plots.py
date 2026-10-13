"""Paper Fig. 2 content, using the project's heliocentric plotting style."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plotter import Plotter


def plot_earth_mars_outputs(case, options, reference, solution, normalization,
                            monte_carlo, output_prefix: Path, dynamics):
    # Import here to keep the plot module independent of solver initialization.
    from stochastic_traj_opt_jax import densified_mean_trajectory

    plotter = Plotter(output_prefix)
    days = reference.node_times * case.time_unit / 86400.0

    substeps = options.monte_carlo_sde_integrator_substeps or options.sde_integrator_substeps
    nominal_dense = densified_mean_trajectory(case, options, dynamics, solution,
                                             reference.steps, samples_per_arc=substeps)
    # The closed-loop spread is far tighter than the open-loop one, so it needs a
    # larger magnification to stay visible next to the transfer line.
    magnifications = {True: 100.0, False: 20.0}
    indices = [0, 1]
    scales = normalization.scale[indices]
    # Same explicitly labelled 3-sigma ellipse convention as the CR3BP plots.
    ellipse_sigma = 3.0

    for feedback in (True, False):
        magnification = magnifications[feedback]
        scale_label = plotter.plot_magnification_label(magnification)
        ensemble = monte_carlo if feedback else monte_carlo["no_feedback"]
        history = ensemble["plot_trajectory_history"]
        displayed = nominal_dense[:3, None, :] + magnification*(history[:3] - nominal_dense[:3, None, :])
        fig, ax = plt.subplots(figsize=Plotter.SINGLE_FIGSIZE, dpi=Plotter.FIGURE_DPI)
        for track, name, color in plotter._heliocentric_tracks(case):
            ax.plot(track[0], track[1], "--", color=color, lw=Plotter.REFERENCE_LINE_WIDTH,
                    label=f"{name} orbit", zorder=2)
        lines = ax.plot(displayed[0].T, displayed[1].T, color=Plotter.DARK_GREY,
                        alpha=.20, lw=.4, zorder=3)
        lines[0].set_label(f"Monte Carlo ({scale_label})")
        ax.plot(nominal_dense[0], nominal_dense[1], color=Plotter.BLACK,
                lw=Plotter.TRAJECTORY_LINE_WIDTH, label="Transfer", zorder=4)
        for point, name, color in ((np.zeros(3), "Sun", Plotter.SUN_COLOR),
                                   (solution.means[:3, 0], "Earth", Plotter.EARTH_COLOR),
                                   (solution.means[:3, -1], "Mars", Plotter.MARS_COLOR)):
            ax.scatter(*point[:2], s=58/(1.5 if name == "Sun" else 2.5)**2,
                       color=color, edgecolor=Plotter.BLACK, linewidth=.6, label=name, zorder=7)
        # Feedforward arrows retain the same purple and layering as deterministic plots.
        thrust = solution.feedforward * case.max_thrust_nd
        points, arrows = plotter._heliocentric_arrows(case, days[:-1], solution.means[:, :-1], thrust)
        ax.quiver(points[0], points[1], arrows[0], arrows[1], angles="xy", scale_units="xy", scale=1,
                  color=Plotter.THRUST_ARROW_3D_COLOR, width=.0046, alpha=.82, zorder=6, label="Thrust")
        if feedback:
            for k in range(len(days)):
                cov = solution.covariances[k][np.ix_(indices, indices)] * np.outer(scales, scales)
                points = magnification * plotter.plot_covariance_ellipse_points(cov, ellipse_sigma)
                # Same colour, alpha and legend entry as the CR3BP projections.
                ax.plot(solution.means[0, k]+points[0], solution.means[1, k]+points[1],
                        color=Plotter.DARK_BLUE, lw=.80, alpha=.90, zorder=5,
                        label=f"Predicted covariance ({scale_label})" if k == 0 else "_nolegend_")
        ax.set(xlabel="x [AU]", ylabel="y [AU]")
        plotter._style_2d_axis(ax, equal_axis=True)
        ax.set_axisbelow(True)
        ax.margins(.14)
        fig.tight_layout(rect=(0, 0, 1, .73))
        plotter._heliocentric_legend(fig, ax)
        suffix = "projections" if feedback else "projections_no_feedback"
        fig.savefig(plotter._path(suffix), dpi=Plotter.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
