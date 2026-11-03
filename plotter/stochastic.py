"""Trajectory validation figures and numerical exports."""

from __future__ import annotations


from pathlib import Path
import json
from dataclasses import asdict
import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq
from plotter import Plotter
import matplotlib.pyplot as plt
from deterministic_trajectory_optimization.models import CR3BPEarthMoon, TestCase

from stochastic_trajectory_optimization.dynamics import (
    Dynamics,
    NP,
    NU,
    NX,
    Normalization,
    Options,
    ReferenceTraj,
    RobustSolution,
    acceleration_diffusion_nd,
    augmented_dimension,
    integration_map,
    terminal_mean_state_target,
)


def std_physical_units(
    case: TestCase, normalization: Normalization, covariances: np.ndarray
) -> np.ndarray:
    """1-sigma of each augmented state in m, m/s and kg."""

    diagonal = np.sqrt(np.clip(np.einsum("kii->ki", covariances), 0.0, None))
    physical = diagonal * normalization.scale[None, :]
    scale = np.array(
        [case.length_unit * 1e3] * 3 + [case.velocity_unit * 1e3] * 3 + [case.m0_wet],
        dtype=float,
    )
    return physical * scale[None, :]


def densified_mean_trajectory(
    case: TestCase,
    options: Options,
    dynamics: Dynamics,
    solution: RobustSolution,
    steps: np.ndarray,
    samples_per_arc: int = 20,
) -> np.ndarray:
    """Re-propagate the solved mean trajectory for plotting only.

    `solution.means` holds one state per node, so drawing it directly joins the
    nodes with straight lines and a curved transfer reads as a chain of
    segments. Here each arc is re-integrated from its own node state under that
    arc's zero-order-hold control, sampling `samples_per_arc` intermediate
    points with the deterministic Dopri8 map. Monte Carlo uses the XMDS2 RK9 SDE
    map and is intentionally handled separately.

    This is a nominal mean-command visualization rather than one stochastic
    realization. It reproduces every optimized node state by construction, so
    the covariance ellipses drawn at the nodes stay attached to it. It is
    display-only: nothing here feeds the NLP, diagnostics, or saved arrays.
    """

    n_arcs = solution.feedforward.shape[1]
    fractions = np.arange(1, samples_per_arc) / float(samples_per_arc)
    columns = [solution.means[:, 0:1]]
    for k in range(n_arcs):
        node_state = solution.means[:, k]
        control = case.max_thrust_nd * solution.feedforward[:, k]
        # Each interior sample is propagated from the node rather than chained
        # from the previous sample, so no sample inherits another's error.
        if fractions.size:
            batch_states = np.repeat(node_state[:, None], fractions.size, axis=1)
            batch_controls = np.repeat(control[:, None], fractions.size, axis=1)
            interior = np.empty((NX, fractions.size), dtype=float)
            for index, fraction in enumerate(fractions):
                propagated, _, _ = dynamics.propagate(
                    batch_states[:, index : index + 1],
                    batch_controls[:, index : index + 1],
                    float(steps[k]) * float(fraction),
                    options.integrator_substeps,
                )
                interior[:, index] = np.asarray(propagated, dtype=float).ravel()
            columns.append(interior)
        # Close the arc on the solved node state itself, so the drawn curve
        # passes through every node exactly.
        columns.append(solution.means[:, k + 1 : k + 2])
    return np.concatenate(columns, axis=1)


def get_collinear_lagrange_points(case: CR3BPEarthMoon) -> dict[str, float]:
    def equilibrium_condition(x):
        r1 = abs(x + case.mu)
        r2 = abs(x - (1.0 - case.mu))
        return (
            x
            - (1.0 - case.mu) * (x + case.mu) / r1**3
            - case.mu * (x - (1.0 - case.mu)) / r2**3
        )

    eps = 1e-9
    return {
        "L1": brentq(equilibrium_condition, -case.mu + eps, 1.0 - case.mu - eps),
        "L2": brentq(equilibrium_condition, 1.0 - case.mu + eps, 1.5),
        "L3": brentq(equilibrium_condition, -1.5, -case.mu - eps),
    }


def propagate_periodic_orbit(case, initial_state, period_nd, steps=1600):
    """Ballistic plotting trajectory using the same Diffrax Dopri8 map."""
    step = integration_map(case, 1)

    @jax.jit
    def rollout(initial):
        def advance(state, _):
            next_state = step(state, jnp.zeros(NU), 0.0, period_nd / steps)
            return next_state, next_state

        return jax.lax.scan(advance, initial, None, length=steps)[1]

    samples = np.asarray(rollout(jnp.asarray(initial_state)))
    return np.column_stack((initial_state, samples.T))


@plt.rc_context({"text.usetex": False, "mathtext.fontset": "cm"})
def plot_outputs(
    case: TestCase,
    options: Options,
    ref_traj: ReferenceTraj,
    solution: RobustSolution,
    normalization: Normalization,
    monte_carlo_result: dict,
    psi_inv: float,
    output_prefix: Path,
    dynamics: Dynamics,
) -> None:
    node_days = ref_traj.node_times * case.time_unit / 86400.0
    position_target = 1.0 / options.position_covariance_reduction
    velocity_target = 1.0 / options.velocity_covariance_reduction
    target_covariance = np.zeros((NX, NX), dtype=float)
    target_covariance[0:3, 0:3] = position_target * np.eye(3)
    target_covariance[3:NP, 3:NP] = velocity_target * np.eye(3)
    sigma_physical = std_physical_units(case, normalization, solution.covariances)
    target_sigma = std_physical_units(
        case, normalization, target_covariance[None, :, :]
    )[0]
    plotter = Plotter(output_prefix)
    rollout_history = np.asarray(
        monte_carlo_result["plot_trajectory_history"], dtype=float
    )
    rollout_applied_control_deviation_norm_n = np.asarray(
        monte_carlo_result["plot_applied_control_deviation_norm_n"],
        dtype=float,
    )

    # 1 - dispersion evolution.
    figure, axes = plt.subplots(
        1,
        3,
        figsize=Plotter.THREE_PANEL_FIGSIZE,
        dpi=Plotter.FIGURE_DPI,
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.0)},
    )
    panels = (
        (
            slice(0, 3),
            r"$\sigma$ [m]",
            (r"$x$", r"$y$", r"$z$"),
            (Plotter.BLUE, Plotter.RED, Plotter.GREY),
        ),
        (
            slice(3, 6),
            r"$\sigma$ [m/s]",
            (r"$\dot{x}$", r"$\dot{y}$", r"$\dot{z}$"),
            (Plotter.BLUE, Plotter.RED, Plotter.GREY),
        ),
        (slice(6, 7), r"$\sigma$ [kg]", (r"$m$",), (Plotter.GREY,)),
    )
    for axis, (columns, label, names, colors) in zip(axes, panels):
        axis.set_box_aspect(1.0)
        for offset, (name, color) in enumerate(zip(names, colors)):
            values = np.maximum(sigma_physical[:, columns][:, offset], 1e-16)
            axis.semilogy(
                node_days,
                values,
                lw=Plotter.REFERENCE_LINE_WIDTH,
                color=color,
                label=name,
            )
        if columns.start < NP:
            axis.axhline(
                target_sigma[columns.start],
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
        plotter._legend(
            axis,
            loc="best",
            fontsize=7.0,
            frameon=True,
            fancybox=False,
            edgecolor=Plotter.BLACK,
            facecolor="white",
            framealpha=1.0,
            borderpad=0.28,
            handlelength=1.4,
        )
    figure.align_ylabels(axes)
    figure.tight_layout()
    figure.savefig(
        output_prefix.parent / f"{output_prefix.name}_dispersion_evolution.png",
        dpi=Plotter.FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)

    # 2 - nominal thrust and the applied-control deviation. The predicted bound
    # includes Gates error before saturation. Radial projection onto the thrust
    # ball cannot increase distance from a feasible nominal control, so this
    # bound also applies conservatively to the saturated applied control.
    nominal_thrust_n = (
        np.linalg.norm(solution.feedforward, axis=0)
        * case.max_thrust_nd
        * case.thrust_unit
    )
    feedback_n = psi_inv * solution.radius * case.max_thrust_nd * case.thrust_unit
    step_days, nominal_step = plotter.plot_zero_order_hold_data(
        node_days, nominal_thrust_n
    )
    _, feedback_bound_step = plotter.plot_zero_order_hold_data(node_days, feedback_n)
    _, rollout_deviation_norm_step = plotter.plot_zero_order_hold_data(
        node_days, rollout_applied_control_deviation_norm_n
    )

    figure, axes = plt.subplots(
        2,
        1,
        figsize=(1.5 * Plotter.WIDE_FIGSIZE[0], 1.5 * Plotter.WIDE_FIGSIZE[1]),
        dpi=Plotter.FIGURE_DPI,
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
    axis.axhline(
        case.max_thrust_n,
        color=Plotter.RED,
        ls=":",
        lw=0.88,
        label="Maximum thrust",
    )
    axis.set_xlabel("time [days]")
    axis.set_ylabel("thrust [N]")
    plotted_peak = max(case.max_thrust_n, float(np.max(nominal_thrust_n)))
    axis.set_ylim(0.0, 1.06 * plotted_peak)
    plotter._style_2d_axis(
        axis,
        tick_size=Plotter.DIAGNOSTIC_TICK_SIZE,
        label_size=Plotter.DIAGNOSTIC_LABEL_SIZE,
    )
    plotter._legend(
        axis,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor=Plotter.BLACK,
        facecolor="white",
        framealpha=1.0,
        fontsize=8,
        borderaxespad=0.0,
    )

    axis = axes[1]
    rollout_lines = axis.plot(
        step_days,
        rollout_deviation_norm_step,
        color=Plotter.DARK_GREY,
        alpha=0.28,
        lw=0.48,
        zorder=1,
    )
    if rollout_lines:
        rollout_lines[0].set_label("Monte Carlo")
    axis.plot(
        step_days,
        feedback_bound_step,
        color=Plotter.BLUE,
        ls="--",
        lw=0.82,
        label="Predicted bound",
        zorder=2,
    )
    axis.set_xlabel("time [days]")
    axis.set_ylabel(r"$\|T_{\mathrm{applied}}-\bar{T}\|$ [N]")
    axis.set_ylim(
        0.0,
        1.05
        * max(
            float(np.max(feedback_n)),
            float(np.max(rollout_applied_control_deviation_norm_n)),
        ),
    )
    plotter._style_2d_axis(
        axis,
        tick_size=Plotter.DIAGNOSTIC_TICK_SIZE,
        label_size=Plotter.DIAGNOSTIC_LABEL_SIZE,
    )
    plotter._legend(
        axis,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        frameon=True,
        fancybox=False,
        edgecolor=Plotter.BLACK,
        facecolor="white",
        framealpha=1.0,
        fontsize=8,  # same size as the upper subplot's legend
        borderaxespad=0.0,
    )

    figure.tight_layout()
    figure.savefig(
        output_prefix.parent / f"{output_prefix.name}_thrust_budget.png",
        dpi=Plotter.FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)

    # 3 - transfer projections with scaled uncertainty layers.
    if case.test_case_id == "earth_mars":
        pass

        plot_earth_mars_outputs(
            case,
            options,
            ref_traj,
            solution,
            normalization,
            monte_carlo_result,
            output_prefix,
            dynamics,
        )
        plotter.plot_terminal_distribution(
            case, options, solution, normalization, monte_carlo_result
        )
        return
    axes_to_plot = plotter._projection_axes_for_case(case)
    if len(axes_to_plot) == 1:
        figure, axis = plt.subplots(
            figsize=Plotter.SINGLE_FIGSIZE, dpi=Plotter.FIGURE_DPI
        )
        projection_axes = [axis]
    else:
        figure, axes = plt.subplots(
            1,
            len(axes_to_plot),
            figsize=Plotter.TRIPLE_SQUARE_FIGSIZE,
            dpi=Plotter.FIGURE_DPI,
        )
        projection_axes = list(np.atleast_1d(axes))

    lagrange = get_collinear_lagrange_points(case)
    departure_orbit = None
    target_orbit = None
    if case.departure_period_nd is not None:
        departure_orbit = propagate_periodic_orbit(
            case, case.x0_augmented_state, case.departure_period_nd
        )
    # A truncated transfer ends at a relative reference node, not at the full
    # target periodic orbit.
    if (
        options.truncated_uniform_mesh_arcs is None
        and case.target_period_nd is not None
    ):
        target_orbit = propagate_periodic_orbit(
            case, case.xf_augmented_state, case.target_period_nd
        )

    # Display-only dense mean trajectory: the arcs re-propagated through the
    # true dynamics, so the drawn nominal is a smooth curve instead of the
    # straight chords that joining the node states produces.
    dense_samples_per_arc = 20
    dense_means = densified_mean_trajectory(
        case,
        options,
        dynamics,
        solution,
        ref_traj.steps,
        samples_per_arc=dense_samples_per_arc,
    )

    # The Monte Carlo trajectory bundle carries one entry per RK9 substep
    # (see monte_carlo_rollouts), landing at uniform arc fractions
    # j/rollout_substeps for j = 1..rollout_substeps -- exactly the fraction
    # grid densified_mean_trajectory samples at, so the two align column for
    # column when built with matching samples_per_arc.
    rollout_substeps_override = options.monte_carlo_sde_integrator_substeps
    rollout_substeps = int(
        rollout_substeps_override
        if rollout_substeps_override is not None
        else options.sde_integrator_substeps
    )
    rollout_dense_means = (
        dense_means
        if rollout_substeps == dense_samples_per_arc
        else densified_mean_trajectory(
            case,
            options,
            dynamics,
            solution,
            ref_traj.steps,
            samples_per_arc=rollout_substeps,
        )
    )

    magnification = 100.0
    projection_scale_label = plotter.plot_magnification_label(magnification)
    covariance_stride = max(solution.covariances.shape[0] // 24, 1)
    covariance_nodes = list(range(0, solution.covariances.shape[0], covariance_stride))
    if covariance_nodes[-1] != solution.covariances.shape[0] - 1:
        covariance_nodes.append(solution.covariances.shape[0] - 1)
    control_nd = case.max_thrust_nd * solution.feedforward
    # The dense curve carries `samples_per_arc` columns per arc plus a final
    # node, so the zero-order-hold control is repeated to match it column for
    # column; `_plot_projection` indexes states and controls together.
    dense_control_nd = np.concatenate(
        [
            np.repeat(
                control_nd[:, k : k + 1],
                dense_samples_per_arc,
                axis=1,
            )
            for k in range(control_nd.shape[1])
        ]
        + [control_nd[:, -1:]],
        axis=1,
    )

    for projection_index, (axis, (axis_0, axis_1)) in enumerate(
        zip(projection_axes, axes_to_plot)
    ):
        # The smooth nominal is drawn from the re-propagated arcs, while the
        # thrust arrows stay on the node arrays: `_plot_thrust_arrows_2d`
        # selects a fixed number of arrows out of the samples whose thrust is
        # active, so handing it the 20x denser grid would pack them solid.
        plotter._plot_projection(
            axis,
            case,
            dense_means,
            dense_control_nd,
            departure_orbit,
            target_orbit,
            lagrange,
            axis_0,
            axis_1,
            arrow_states=solution.means,
            arrow_controls=control_nd,
        )
        # Compare each Monte Carlo substep against the mean state at that same
        # substep (not just the coarse arc-node mean), since rollout_history
        # now resolves the interior of each arc rather than only its endpoints.
        displayed_0 = rollout_dense_means[axis_0, None, :] + magnification * (
            rollout_history[axis_0] - rollout_dense_means[axis_0, None, :]
        )
        displayed_1 = rollout_dense_means[axis_1, None, :] + magnification * (
            rollout_history[axis_1] - rollout_dense_means[axis_1, None, :]
        )
        rollout_lines = axis.plot(
            displayed_0.T,
            displayed_1.T,
            color=Plotter.DARK_GREY,
            alpha=0.2,
            lw=0.38,
            zorder=1,
        )
        if projection_index == 0 and rollout_lines:
            rollout_lines[0].set_label(f"Monte Carlo ({projection_scale_label})")

        indices = [axis_0, axis_1]
        physical_scale = normalization.scale[indices]
        for covariance_index, node in enumerate(covariance_nodes):
            block = solution.covariances[node][np.ix_(indices, indices)]
            block = block * np.outer(physical_scale, physical_scale)
            points = magnification * plotter.plot_covariance_ellipse_points(block, 3.0)
            axis.plot(
                solution.means[axis_0, node] + points[0],
                solution.means[axis_1, node] + points[1],
                color=Plotter.DARK_BLUE,
                lw=0.80,
                alpha=0.90,
                zorder=2,
                label=f"Predicted covariance ({projection_scale_label})"
                if projection_index == 0 and covariance_index == 0
                else "_nolegend_",
            )

    if len(axes_to_plot) > 1:
        # Congruent square panels. This has to run after `_plot_projection`,
        # which calls `_style_2d_axis(equal_axis=True)` and so sets
        # `set_aspect("equal", adjustable="box")`: that keeps the data scale
        # undistorted but shrinks each panel's box to its own data range, so
        # x-y, x-z and y-z otherwise render at three different sizes.
        # Re-asserting the equal data scale with `adjustable="datalim"` moves
        # the freedom into the axis limits, leaving the box free to be squared.
        for projection_axis in projection_axes:
            projection_axis.set_aspect("equal", adjustable="datalim")
            projection_axis.set_box_aspect(1.0)

    is_single_projection = len(axes_to_plot) == 1
    plotter._figure_legend(
        figure,
        projection_axes,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90 if is_single_projection else 0.995),
        frameon=True,
        fancybox=False,
        edgecolor=Plotter.BLACK,
        facecolor="white",
        framealpha=1.0,
        fontsize=6.7,
        columnspacing=0.85,
        handlelength=1.5,
        borderpad=0.30,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.70 if is_single_projection else 0.90))
    figure.savefig(
        output_prefix.parent / f"{output_prefix.name}_projections.png",
        dpi=Plotter.FIGURE_DPI,
        bbox_inches="tight",
    )
    plt.close(figure)

    # 4 - six-dimensional terminal position-velocity distribution matrix.
    plotter.plot_terminal_distribution(
        case,
        options,
        solution,
        normalization,
        monte_carlo_result,
    )


def save_outputs(
    case: TestCase,
    options: Options,
    ref_traj: ReferenceTraj,
    solution: RobustSolution,
    normalization: Normalization,
    monte_carlo_result: dict,
    diagnostics: dict[str, float],
    psi_inv: float,
    output_prefix: Path,
    dynamics: Dynamics,
    *,
    plots=True,
) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    node_days = ref_traj.node_times * case.time_unit / 86400.0

    # Physical gains, undoing the normalisation.
    physical_gains = (
        case.max_thrust_nd * solution.gains / normalization.scale[None, None, 0:NP]
    )
    sigma_physical = std_physical_units(case, normalization, solution.covariances)

    extra_arrays = {}
    if case.test_case_id == "earth_mars":
        extra_arrays = {
            key: np.asarray(case.ephemeris[key])
            for key in (
                "earth_track_si",
                "mars_track_si",
                "earth_track_times_days",
                "mars_track_times_days",
            )
        }
    np.savez(
        output_prefix.with_suffix(".npz"),
        node_times_nd=ref_traj.node_times,
        node_days=node_days,
        steps_nd=ref_traj.steps,
        short_horizon_reference_arcs=(
            -1
            if options.truncated_uniform_mesh_arcs is None
            else options.truncated_uniform_mesh_arcs
        ),
        short_horizon_solve=options.truncated_uniform_mesh_arcs is not None,
        covariance_transcription=np.asarray("multiple_shooting"),
        covariance_scale_variances=solution.covariance_scale_variances,
        terminal_mean_state_target=terminal_mean_state_target(case, options, ref_traj),
        means=solution.means,
        feedforward_normalized=solution.feedforward,
        feedforward_n=solution.feedforward * case.max_thrust_nd * case.thrust_unit,
        gains_normalized=solution.gains,
        gains=physical_gains,
        radius_normalized=solution.radius,
        covariance_normalized=solution.covariances,
        control_covariance_normalized=solution.control_covariances,
        reference_states=ref_traj.states,
        reference_controls=ref_traj.controls,
        reference_fuel_consumed_kg=ref_traj.fuel_consumed,
        sigma_physical=sigma_physical,
        normalization_scale_nd=normalization.scale,
        initial_state_std_nd=normalization.initial_std,
        initial_position_std_km=options.initial_position_std_km,
        initial_velocity_std_ms=options.initial_velocity_std_km_s * 1e3,
        initial_mass_std_kg=options.initial_mass_std_kg,
        final_position_std_km=options.initial_position_std_km
        / np.sqrt(options.position_covariance_reduction),
        final_velocity_std_ms=options.initial_velocity_std_km_s
        * 1e3
        / np.sqrt(options.velocity_covariance_reduction),
        navigation_position_std_km=options.navigation_position_std_km,
        navigation_velocity_std_ms=options.navigation_velocity_std_km_s * 1e3,
        gates_proportional_magnitude_std=options.gates_proportional_magnitude_std,
        gates_proportional_pointing_std=options.gates_proportional_pointing_std,
        acceleration_diffusion_km_s32=options.acceleration_diffusion_km_s32,
        acceleration_diffusion_nd=acceleration_diffusion_nd(case, options),
        sde_integrator=np.asarray("XMDS2_RK9"),
        sde_integrator_substeps=options.sde_integrator_substeps,
        ut_augmented_dimension=augmented_dimension(options),
        ut_sigma_points=2 * augmented_dimension(options) + 1,
        bryson_sigma_factor=options.bryson_sigma_factor,
        # Store the actual NLP weights, rather than the original exporter's
        # uniformly rescaled pair from an earlier objective convention.
        bryson_state_weight=options.state_covariance_weight
        * np.eye(NP)
        / options.bryson_sigma_factor**2,
        bryson_control_weight=options.control_covariance_weight * np.eye(NU),
        terminal_Qf_objective_weight=diagnostics.get("terminal_covariance_weight", 0.0),
        position_covariance_reduction=options.position_covariance_reduction,
        velocity_covariance_reduction=options.velocity_covariance_reduction,
        position_target_ratio=1.0 / options.position_covariance_reduction,
        velocity_target_ratio=1.0 / options.velocity_covariance_reduction,
        target_ratio=np.array(
            [1.0 / options.position_covariance_reduction] * 3
            + [1.0 / options.velocity_covariance_reduction] * 3,
            dtype=float,
        ),
        psi_inv=psi_inv,
        monte_carlo_sampling=np.asarray("independent_pseudorandom_gaussian"),
        monte_carlo_samples=options.monte_carlo_samples,
        monte_carlo_cumulative=monte_carlo_result["cumulative_control_effort"],
        monte_carlo_violation_fraction=monte_carlo_result["violation_fraction"],
        monte_carlo_terminal_covariance=monte_carlo_result["terminal_covariance"],
        monte_carlo_terminal_deviation=monte_carlo_result["terminal_deviation"],
        monte_carlo_plot_trajectory_history=monte_carlo_result[
            "plot_trajectory_history"
        ],
        monte_carlo_plot_applied_control_deviation_norm_n=monte_carlo_result[
            "plot_applied_control_deviation_norm_n"
        ],
        monte_carlo_plot_commanded_control_deviation_norm_n=monte_carlo_result[
            "plot_commanded_control_deviation_norm_n"
        ],
        diagnostics=np.array(diagnostics, dtype=object),
        monte_carlo_plot_executed_thrust_n=monte_carlo_result["plot_executed_thrust_n"],
        monte_carlo_plot_applied_thrust_n=monte_carlo_result["plot_applied_thrust_n"],
        **extra_arrays,
    )

    metadata = {
        "case": case.test_case_id,
        "options": asdict(options),
        "diagnostics": diagnostics,
        "objective": "duration-weighted running cost; terminal covariance cost for standard, hard terminal bound for quantile",
        "navigation": "independent Gaussian position/velocity measurement errors at each control update",
        "thrust_chance_confidence": 1 - options.violation_parameter,
        "length_unit_km": case.length_unit,
        "time_unit_s": case.time_unit,
        "mass_unit_kg": case.m0_wet,
        "thrust_unit_n": case.thrust_unit,
    }
    if case.test_case_id == "earth_mars":
        metadata["ephemeris_source"] = (
            "Fixed tabulated boundary states; two-body plot tracks "
            "(deterministic_trajectory_optimization.models.earth_mars_data)"
        )
        metadata["ephemeris"] = {
            k: v for k, v in case.ephemeris.items() if "track" not in k
        }
        metadata["display_position_deviation_magnification"] = {
            "feedback": 100,
            "no_feedback": 20,
        }
        metadata["display_covariance_sigma_multiplier"] = 3
        metadata["model_assumptions"] = [
            "Bryson covariance objective, not a fuel quantile objective",
            "independent navigation errors, not Kalman-filter covariance propagation",
            "constant thrust and direct mass, not constant acceleration and log mass",
            "CR3BP physical acceleration diffusion and 0.001 kg initial mass dispersion retained",
            "Fixed benchmark boundary states and transfer duration",
        ]
        # Record the independent open-loop ensemble alongside the feedback run.
        open_loop = monte_carlo_result["no_feedback"]
        metadata["no_feedback_max_violation_fraction"] = open_loop[
            "max_violation_fraction"
        ]
    output_prefix.with_suffix(".json").write_text(json.dumps(metadata, indent=2))

    feedforward_magnitude_n = (
        np.linalg.norm(solution.feedforward, axis=0)
        * case.max_thrust_nd
        * case.thrust_unit
    )
    feedback_magnitude_n = (
        psi_inv * solution.radius * case.max_thrust_nd * case.thrust_unit
    )
    table = np.column_stack(
        [
            node_days[:-1],
            ref_traj.steps,
            solution.means[:, :-1].T,
            solution.feedforward.T * case.max_thrust_nd * case.thrust_unit,
            feedforward_magnitude_n,
            feedback_magnitude_n,
            feedforward_magnitude_n + feedback_magnitude_n,
            sigma_physical[:-1, 0:3],
            sigma_physical[:-1, 3:6],
            sigma_physical[:-1, 6:7],
        ]
    )
    header = (
        "t_days,dt_nd,x,y,z,vx,vy,vz,m_nd,Sx_N,Sy_N,Sz_N,S_norm_N,feedback_N,total_N,"
        "sigma_x_m,sigma_y_m,sigma_z_m,sigma_vx_ms,sigma_vy_ms,sigma_vz_ms,sigma_m_kg"
    )
    np.savetxt(
        output_prefix.with_suffix(".csv"),
        table,
        delimiter=",",
        header=header,
        comments="",
    )

    if plots:
        plot_outputs(
            case,
            options,
            ref_traj,
            solution,
            normalization,
            monte_carlo_result,
            psi_inv,
            output_prefix,
            dynamics,
        )


def plot_earth_mars_outputs(
    case,
    options,
    reference,
    solution,
    normalization,
    monte_carlo,
    output_prefix: Path,
    dynamics,
):
    # Import here to keep the plot module independent of solver initialization.
    pass

    plotter = Plotter(output_prefix)
    days = reference.node_times * case.time_unit / 86400.0

    substeps = (
        options.monte_carlo_sde_integrator_substeps or options.sde_integrator_substeps
    )
    nominal_dense = densified_mean_trajectory(
        case, options, dynamics, solution, reference.steps, samples_per_arc=substeps
    )
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
        displayed = nominal_dense[:3, None, :] + magnification * (
            history[:3] - nominal_dense[:3, None, :]
        )
        fig, ax = plt.subplots(figsize=Plotter.SINGLE_FIGSIZE, dpi=Plotter.FIGURE_DPI)
        for track, name, color in plotter._heliocentric_tracks(case):
            ax.plot(
                track[0],
                track[1],
                "--",
                color=color,
                lw=Plotter.REFERENCE_LINE_WIDTH,
                label=f"{name} orbit",
                zorder=2,
            )
        lines = ax.plot(
            displayed[0].T,
            displayed[1].T,
            color=Plotter.DARK_GREY,
            alpha=0.20,
            lw=0.4,
            zorder=3,
        )
        lines[0].set_label(f"Monte Carlo ({scale_label})")
        ax.plot(
            nominal_dense[0],
            nominal_dense[1],
            color=Plotter.BLACK,
            lw=Plotter.TRAJECTORY_LINE_WIDTH,
            label="Transfer",
            zorder=4,
        )
        for point, name, color in (
            (np.zeros(3), "Sun", Plotter.SUN_COLOR),
            (solution.means[:3, 0], "Earth", Plotter.EARTH_COLOR),
            (solution.means[:3, -1], "Mars", Plotter.MARS_COLOR),
        ):
            ax.scatter(
                *point[:2],
                s=58 / (1.5 if name == "Sun" else 2.5) ** 2,
                color=color,
                edgecolor=Plotter.BLACK,
                linewidth=0.6,
                label=name,
                zorder=7,
            )
        # Feedforward arrows retain the same purple and layering as deterministic plots.
        thrust = solution.feedforward * case.max_thrust_nd
        points, arrows = plotter._heliocentric_arrows(
            case, days[:-1], solution.means[:, :-1], thrust
        )
        ax.quiver(
            points[0],
            points[1],
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
        if feedback:
            for k in range(len(days)):
                cov = solution.covariances[k][np.ix_(indices, indices)] * np.outer(
                    scales, scales
                )
                points = magnification * plotter.plot_covariance_ellipse_points(
                    cov, ellipse_sigma
                )
                # Same colour, alpha and legend entry as the CR3BP projections.
                ax.plot(
                    solution.means[0, k] + points[0],
                    solution.means[1, k] + points[1],
                    color=Plotter.DARK_BLUE,
                    lw=0.80,
                    alpha=0.90,
                    zorder=5,
                    label=f"Predicted covariance ({scale_label})"
                    if k == 0
                    else "_nolegend_",
                )
        ax.set(xlabel="x [AU]", ylabel="y [AU]")
        plotter._style_2d_axis(ax, equal_axis=True)
        ax.set_axisbelow(True)
        ax.margins(0.14)
        fig.tight_layout(rect=(0, 0, 1, 0.73))
        plotter._heliocentric_legend(fig, ax)
        suffix = "projections" if feedback else "projections_no_feedback"
        fig.savefig(plotter._path(suffix), dpi=Plotter.FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
