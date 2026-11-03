"""Independent single-phase residual and Monte Carlo validation."""

from __future__ import annotations


from dataclasses import replace
import types
from pathlib import Path
import json
import jax
import jax.numpy as jnp
import numpy as np
from deterministic_trajectory_optimization.models import TestCase

from stochastic_trajectory_optimization.dynamics import (
    Dynamics as Dynamics,
    EARTH_MARS_REFERENCE_DIR as EARTH_MARS_REFERENCE_DIR,
    LTRI as LTRI,
    MTRI as MTRI,
    NL as NL,
    NM as NM,
    NP as NP,
    NU as NU,
    NW as NW,
    NX as NX,
    Normalization as Normalization,
    Options as Options,
    REFERENCE_DIR as REFERENCE_DIR,
    ROOT as ROOT,
    ReferenceTraj as ReferenceTraj,
    RobustSolution as RobustSolution,
    STOCHASTIC_CASE_REGISTRY as STOCHASTIC_CASE_REGISTRY,
    UNIFORM_ARCS_BY_CASE as UNIFORM_ARCS_BY_CASE,
    acceleration_diffusion_nd as acceleration_diffusion_nd,
    augmented_dimension as augmented_dimension,
    build_normalization as build_normalization,
    build_propagation_arc_map as build_propagation_arc_map,
    default_options as default_options,
    gates_execution_error_enabled as gates_execution_error_enabled,
    load_case as load_case,
    navigation_covariance as navigation_covariance,
    navigation_error_enabled as navigation_error_enabled,
    propagate_stochastic_moments as propagate_stochastic_moments,
    psi_inverse as psi_inverse,
    spectral_radius as spectral_radius,
    split_over_devices as split_over_devices,
    step_adjoint as step_adjoint,
    stochastic_integration_map as stochastic_integration_map,
    terminal_mean_state_target as terminal_mean_state_target,
    uniform_arc_count as uniform_arc_count,
    vector_field as vector_field,
)
from plotter.stochastic import save_outputs


def monte_carlo_rollouts(
    case: TestCase,
    options: Options,
    normalization: Normalization,
    solution: RobustSolution,
    steps: np.ndarray,
    *,
    feedback_enabled: bool = True,
) -> dict[str, np.ndarray | float]:
    """Apply the converged policy with maneuver-execution error and thrust saturation.

    The proportional-only Gates execution error is sampled conditionally on the
    commanded (unsaturated) thrust and added to it under the execution-error model. Violation statistics test the executed thrust,
    including execution error and before saturation, against max_thrust,
    matching the NLP chance constraint. The executed sum is then clipped to
    the maximum thrust; propagation, accumulated effort and peak applied thrust
    all use this saturated, executed command, matching the physical actuator.

    Initial-state, navigation-error, process-noise and maneuver-execution-error
    coordinates come from four mutually independent ``jax.random`` streams of
    the key ``monte_carlo_seed``, folded with the arc index. The whole arc
    update (policy, execution error, saturation, SDE step) runs as one jitted
    function on the JAX default backend, so the CPU and GPU versions draw the
    same samples and differ only by floating-point roundoff. With several CPU
    devices (JAX_NUM_CPU_DEVICES) the SDE step is split over them.
    """

    samples = options.monte_carlo_samples
    if samples < 2:
        raise ValueError("monte_carlo_samples must be at least 2")
    substeps_override = options.monte_carlo_sde_integrator_substeps
    substeps = int(
        substeps_override
        if substeps_override is not None
        else options.sde_integrator_substeps
    )
    if (
        substeps
        != (
            substeps_override
            if substeps_override is not None
            else options.sde_integrator_substeps
        )
        or substeps < 1
    ):
        raise ValueError(
            "monte_carlo_sde_integrator_substeps (or sde_integrator_substeps) "
            "must be a positive integer"
        )
    # stochastic_integration_map reads sde_integrator_substeps from `options`
    # directly, so substitute the effective value rather than threading a
    # separate substeps argument through it.
    sde_options = (
        options
        if substeps_override is None
        else replace(options, sde_integrator_substeps=substeps)
    )
    initial_key, navigation_key, process_key, gates_key = jax.random.split(
        jax.random.key(options.monte_carlo_seed), 4
    )
    states = jnp.asarray(solution.means[:, 0]) + jnp.asarray(
        normalization.initial_std
    ) * jax.random.normal(initial_key, (samples, NX))
    # Navigation error: an independent draw at every arc, added to the measured
    # deviation the policy acts on.
    navigation_std = jnp.sqrt(
        jnp.clip(
            jnp.diag(jnp.asarray(navigation_covariance(case, options, normalization))),
            0.0,
            None,
        )
    )
    navigation_enabled = navigation_error_enabled(options)
    gates_enabled = gates_execution_error_enabled(options)
    process_noise_enabled = options.acceleration_diffusion_km_s32 > 0.0
    magnitude_std = options.gates_proportional_magnitude_std
    pointing_std = options.gates_proportional_pointing_std
    max_thrust = case.max_thrust_nd
    scale = jnp.asarray(normalization.scale[0:NP])
    sde_step = stochastic_integration_map(case, sde_options)
    sde_step_batch = split_over_devices(
        jax.vmap(sde_step, in_axes=(0, 0, 0, 0, None)), (True, True, True, True, False)
    )
    # Same computation as sde_step_batch, but also returns every intermediate
    # RK9 substep state; applied only to the plotted subset below so the
    # trajectory bundle traces the actual finer SDE path between arc nodes
    # instead of chording straight across each (possibly coarse) arc.
    sde_step_history_batch = jax.vmap(sde_step.with_history, in_axes=(0, 0, 0, 0, None))
    # Retain a small subset of these exact same SDE realizations for figures.
    # This avoids plotting a separate ensemble with different uncertainty or
    # integration assumptions. One entry per RK9 substep (not just per arc),
    # so the plotted bundle resolves the true path within each arc.
    plot_count = min(100 if case.test_case_id == "earth_mars" else 160, samples)

    @jax.jit
    def arc_step(states, arc, mean, feedforward, gain, duration):
        deviation = (states[:, 0:NP] - mean[0:NP]) / scale
        if navigation_enabled:
            deviation = deviation + navigation_std * jax.random.normal(
                jax.random.fold_in(navigation_key, arc), (samples, NP)
            )
        commanded_control = max_thrust * (
            feedforward
            + (deviation @ gain.T if feedback_enabled else jnp.zeros((samples, NU)))
        )
        commanded_magnitudes = jnp.linalg.norm(commanded_control, axis=1)
        # Maneuver-execution error: proportional-only Gates model, sampled
        # conditionally on the commanded (unsaturated) thrust of each sample,
        # then added to it before the actuator saturates .
        if gates_enabled:
            gates_normals = jax.random.normal(
                jax.random.fold_in(gates_key, arc), (samples, NU)
            )
            unit_command = (
                commanded_control
                / jnp.clip(commanded_magnitudes, 1e-300, None)[:, None]
            )
            parallel = (
                jnp.sum(unit_command * gates_normals, axis=1, keepdims=True)
                * unit_command
            )
            transverse = gates_normals - parallel
            executed_command = commanded_control + commanded_magnitudes[:, None] * (
                magnitude_std * parallel + pointing_std * transverse
            )
        else:
            executed_command = commanded_control
        executed_magnitudes = jnp.linalg.norm(executed_command, axis=1)
        # Radially project each over-limit command onto the thrust ball,
        # preserving its direction. Exceedances are measured before clipping,
        # so actuator saturation cannot hide a chance-constraint violation.
        saturated = executed_magnitudes > max_thrust
        applied_control = (
            executed_command
            * jnp.where(saturated, max_thrust / executed_magnitudes, 1.0)[:, None]
        )
        applied_magnitudes = jnp.minimum(executed_magnitudes, max_thrust)
        if process_noise_enabled:
            brownian_increments = jnp.sqrt(duration / substeps) * jax.random.normal(
                jax.random.fold_in(process_key, arc), (samples, substeps, NW)
            )
        else:
            brownian_increments = jnp.zeros((samples, substeps, NW))
        # The plotted subset also gets its intermediate RK9 substep states,
        # from the exact same states, controls and Brownian draws.
        _, substep_history = sde_step_history_batch(
            states[:plot_count],
            applied_control[:plot_count],
            applied_magnitudes[:plot_count],
            brownian_increments[:plot_count],
            duration,
        )
        nominal_control = max_thrust * feedforward
        new_states = sde_step_batch(
            states, applied_control, applied_magnitudes, brownian_increments, duration
        )
        return new_states, dict(
            violation_fraction=jnp.mean(saturated),
            worst_exceedance=jnp.max(executed_magnitudes) - max_thrust,
            applied_magnitudes=applied_magnitudes,
            history=substep_history,
            executed_thrust=executed_magnitudes[:plot_count],
            applied_thrust=applied_magnitudes[:plot_count],
            commanded_deviation=jnp.linalg.norm(
                commanded_control[:plot_count] - nominal_control, axis=1
            ),
            applied_deviation=jnp.linalg.norm(
                applied_control[:plot_count] - nominal_control, axis=1
            ),
        )

    # Cumulative control effort and peak applied thrust are stored for each sample.
    cumulative_control_effort = jnp.zeros(samples)
    peak_applied_thrust = jnp.zeros(samples)
    history = [states[:plot_count, :, None]]
    arcs = []
    for k in range(steps.size):
        states, arc = arc_step(
            states,
            k,
            jnp.asarray(solution.means[:, k]),
            jnp.asarray(solution.feedforward[:, k]),
            jnp.asarray(solution.gains[k]),
            float(steps[k]),
        )
        cumulative_control_effort = (
            cumulative_control_effort + float(steps[k]) * arc["applied_magnitudes"]
        )
        peak_applied_thrust = jnp.maximum(
            peak_applied_thrust, arc["applied_magnitudes"]
        )
        # substep_history has shape (plot_count, substeps, NX).
        history.append(jnp.swapaxes(arc.pop("history"), 1, 2))
        arc.pop("applied_magnitudes")
        arcs.append(arc)
    arcs = {key: np.asarray(jnp.stack([arc[key] for arc in arcs])) for key in arcs[0]}
    # (NX, plot_count, steps * substeps + 1), the axes of the original NumPy version.
    plot_trajectory_history = np.asarray(jnp.concatenate(history, axis=2)).transpose(
        1, 0, 2
    )
    cumulative_control_effort = np.asarray(cumulative_control_effort)
    states = np.asarray(states).T

    terminal_deviation = (states - solution.means[:, -1:]) / normalization.scale[
        :, None
    ]
    sampled_covariance = np.cov(terminal_deviation)

    return {
        "cumulative_control_effort": cumulative_control_effort,
        "percentile": float(
            np.percentile(
                cumulative_control_effort, 100.0 * (1.0 - options.violation_parameter)
            )
        ),
        "mean_cost": float(np.mean(cumulative_control_effort)),
        "violation_fraction": arcs["violation_fraction"],
        "max_violation_fraction": float(np.max(arcs["violation_fraction"])),
        "worst_exceedance_n": float(np.max(arcs["worst_exceedance"]))
        * case.thrust_unit,
        "peak_applied_thrust_n": np.asarray(peak_applied_thrust) * case.thrust_unit,
        "terminal_covariance": sampled_covariance,
        "terminal_deviation": terminal_deviation,
        "plot_trajectory_history": plot_trajectory_history,
        "feedback_enabled": feedback_enabled,
        "plot_executed_thrust_n": arcs["executed_thrust"] * case.thrust_unit,
        "plot_applied_thrust_n": arcs["applied_thrust"] * case.thrust_unit,
        "plot_commanded_control_deviation_norm_n": arcs["commanded_deviation"]
        * case.thrust_unit,
        "plot_applied_control_deviation_norm_n": arcs["applied_deviation"]
        * case.thrust_unit,
        "backend": jax.default_backend(),
    }


def compute_diagnostics(
    case: TestCase,
    options: Options,
    ref_traj: ReferenceTraj,
    solution: RobustSolution,
    normalization: Normalization,
    monte_carlo_result: dict,
    psi_inv: float,
) -> dict[str, float]:
    position_target = 1.0 / options.position_covariance_reduction
    velocity_target = 1.0 / options.velocity_covariance_reduction
    target_diagonal = np.array(
        [position_target] * 3 + [velocity_target] * 3, dtype=float
    )
    inverse_target_std = 1.0 / np.sqrt(target_diagonal)
    steps = ref_traj.steps

    feedforward_magnitude = np.linalg.norm(solution.feedforward, axis=0)
    deterministic_effort = float(np.sum(steps * feedforward_magnitude))
    feedback_effort = float(np.sum(steps * psi_inv * solution.radius))
    total_effort = deterministic_effort + feedback_effort

    terminal_covariance_target_ratio = solution.covariances[-1][0:NP, 0:NP] * np.outer(
        inverse_target_std, inverse_target_std
    )
    terminal_covariance_target_ratio_eigvals = np.linalg.eigvalsh(
        terminal_covariance_target_ratio
    )

    mean_target = terminal_mean_state_target(case, options, ref_traj)
    componentwise_mean_error = float(
        np.max(np.abs(solution.means[0:NP, -1] - mean_target))
    )
    thrust_budget = feedforward_magnitude + psi_inv * solution.radius

    empirical = monte_carlo_result["terminal_covariance"][0:NP, 0:NP] * np.outer(
        inverse_target_std, inverse_target_std
    )
    empirical_eigenvalues = np.linalg.eigvalsh(empirical)
    position_block_ratio = np.linalg.eigvalsh(
        terminal_covariance_target_ratio[0:3, 0:3]
    )[-1]
    velocity_block_ratio = np.linalg.eigvalsh(
        terminal_covariance_target_ratio[3:NP, 3:NP]
    )[-1]

    return {
        "n_arcs": float(ref_traj.n_arcs),
        "sde_integrator_substeps": float(options.sde_integrator_substeps),
        "acceleration_diffusion_km_s32": float(options.acceleration_diffusion_km_s32),
        "acceleration_diffusion_nd": acceleration_diffusion_nd(case, options),
        "navigation_position_std_km": float(options.navigation_position_std_km),
        "navigation_velocity_std_ms": float(options.navigation_velocity_std_km_s * 1e3),
        "gates_proportional_magnitude_std": float(
            options.gates_proportional_magnitude_std
        ),
        "gates_proportional_pointing_std": float(
            options.gates_proportional_pointing_std
        ),
        "ut_augmented_dimension": float(augmented_dimension(options)),
        "ut_sigma_points": float(2 * augmented_dimension(options) + 1),
        "duration_days": float(ref_traj.node_times[-1] * case.time_unit / 86400.0),
        "short_horizon_solve": float(options.truncated_uniform_mesh_arcs is not None),
        "short_horizon_reference_arcs": float(
            options.truncated_uniform_mesh_arcs
            if options.truncated_uniform_mesh_arcs is not None
            else ref_traj.n_arcs
        ),
        "converged": solution.diagnostics["converged"],
        "state_matching_defect_nd": solution.diagnostics["state_matching_defect_nd"],
        "state_covariance_matching_defect": solution.diagnostics[
            "state_covariance_matching_defect"
        ],
        "scaled_state_covariance_matching_defect": solution.diagnostics[
            "scaled_state_covariance_matching_defect"
        ],
        "state_covariance_scale_min_variance": solution.diagnostics[
            "state_covariance_scale_min_variance"
        ],
        "state_covariance_scale_max_variance": solution.diagnostics[
            "state_covariance_scale_max_variance"
        ],
        "objective_nd": float(solution.objective),
        "objective_mean_control": solution.diagnostics["objective_mean_control"],
        "objective_state_covariance": solution.diagnostics[
            "objective_state_covariance"
        ],
        "objective_control_covariance": solution.diagnostics[
            "objective_control_covariance"
        ],
        "objective_terminal_covariance": solution.diagnostics[
            "objective_terminal_covariance"
        ],
        "bryson_sigma_factor": float(options.bryson_sigma_factor),
        "deterministic_effort_nd": deterministic_effort,
        "stochastic_effort_nd": feedback_effort,
        "total_effort_nd": total_effort,
        "deterministic_effort_kg": deterministic_effort
        * case.m0_wet
        * case.max_thrust_nd
        / case.exhaust_velocity_nd,
        "stochastic_effort_kg": feedback_effort
        * case.m0_wet
        * case.max_thrust_nd
        / case.exhaust_velocity_nd,
        "total_effort_kg": total_effort
        * case.m0_wet
        * case.max_thrust_nd
        / case.exhaust_velocity_nd,
        "nominal_fuel_consumed": ref_traj.fuel_consumed,
        "max_thrust_budget": float(np.max(thrust_budget)),
        "max_feedforward_n": float(
            np.max(feedforward_magnitude) * case.max_thrust_nd * case.thrust_unit
        ),
        "max_feedback_n": float(
            np.max(psi_inv * solution.radius) * case.max_thrust_nd * case.thrust_unit
        ),
        "terminal_componentwise_mean_error_nd": componentwise_mean_error,
        # Position/velocity block ratios are reported for interpretability, but
        # solve_rocp constrains the full 6x6 Loewner order P'_N <= Dt directly
        # (Cholesky-residual encoding), not these two blocks separately.
        "terminal_position_block_ratio": float(position_block_ratio),
        "terminal_velocity_block_ratio": float(velocity_block_ratio),
        "position_covariance_reduction_requested": float(
            options.position_covariance_reduction
        ),
        "velocity_covariance_reduction_requested": float(
            options.velocity_covariance_reduction
        ),
        # The quantity actually constrained: lambda_max(Dt^-1/2 P'_N Dt^-1/2) <= 1.
        "terminal_covariance_max_eigenvalue": float(
            terminal_covariance_target_ratio_eigvals[-1]
        ),
        "terminal_covariance_satisfied": float(
            terminal_covariance_target_ratio_eigvals[-1] <= 1.0 + 1e-12
        ),
        "terminal_position_std_m": float(
            np.sqrt(solution.covariances[-1][0, 0])
            * normalization.scale[0]
            * case.length_unit
            * 1e3
        ),
        "terminal_velocity_std_ms": float(
            np.sqrt(solution.covariances[-1][3, 3])
            * normalization.scale[3]
            * case.velocity_unit
            * 1e3
        ),
        "monte_carlo_percentile_nd": float(monte_carlo_result["percentile"]),
        "predicted_percentile_nd": total_effort * case.max_thrust_nd,
        "monte_carlo_mean_nd": float(monte_carlo_result["mean_cost"]),
        "monte_carlo_percentile_kg": float(monte_carlo_result["percentile"])
        * case.m0_wet
        / case.exhaust_velocity_nd,
        "monte_carlo_max_violation_fraction": float(
            monte_carlo_result["max_violation_fraction"]
        ),
        "monte_carlo_worst_exceedance_n": float(
            monte_carlo_result["worst_exceedance_n"]
        ),
        "monte_carlo_terminal_max_eigenvalue": float(empirical_eigenvalues[-1]),
    }


class TrajectoryResiduals:
    """Independent objective and square-root residuals for a saved native run.

    Initial covariance is fixed; other factors use the saved node scales.
    """

    def __init__(self, case, options, ref, normalization, arc, data):
        self.case, self.options, self.ref, self.n = case, options, ref, ref.n_arcs
        self.stochastic, self.normalization, self.arc = True, normalization, arc
        self.weights = ref.n_arcs * ref.steps / np.sum(ref.steps)
        self.terminal_bound = not bool(data.get("terminal_covariance_cost", 0))
        self.terminal_weight = float(data.get("terminal_covariance_weight", 1.0))
        self.seed = types.SimpleNamespace(gain_scale=np.asarray(data["gain_scale"]))
        self.std = np.asarray(data["node_std"]).T
        self.inv_std = 1.0 / self.std
        self.variances = np.asarray(data["covariance_scale_variances"])
        self.initial_factor = np.asarray(
            LTRI.pack(
                np.linalg.cholesky(normalization.initial_covariance)
                * self.inv_std[0, :, None]
            )
        )
        self.local_values = jax.jit(jax.vmap(self.local_output, in_axes=(0, 0)))
        self.values = jax.jit(self.functions)

    def local_inputs(self, x):
        n = self.n
        parts = [
            x["means"].reshape(NX, n + 1)[:, :-1].T,
            x["feedforward"].reshape(NU, n).T,
        ]
        if self.stochastic:
            entries = x["cholesky_factor"].reshape(n, NL)
            parts += [
                x["gains"].reshape(n, NU * NP),
                jnp.concatenate((self.initial_factor[None], entries[:-1])),
            ]
        return jnp.concatenate(parts, axis=1)

    def local_output(self, values, k):
        mean, feed = values[:NX], values[NX : NX + NU]
        norm = jnp.sqrt(jnp.dot(feed, feed) + self.options.control_norm_eps**2)
        gain = (
            values[NX + NU : NX + NU + NU * NP].reshape(NU, NP)
            * jnp.asarray(self.seed.gain_scale)[k]
        )
        factor = jnp.asarray(self.std)[k, :, None] * LTRI.matrix(values[-NL:])
        covariance = factor @ factor.T
        propagated, covariance_next, control_covariance = self.arc(
            mean, covariance, feed, gain, jnp.asarray(self.ref.steps)[k]
        )
        inv = jnp.asarray(self.inv_std)[k + 1]
        covariance_next = covariance_next * inv[:, None] * inv[None, :]
        chance = norm + psi_inverse(
            NU, self.options.violation_parameter
        ) * spectral_radius(
            control_covariance,
            self.options.spectral_radius_floor,
            self.options.eigenvalue_smoothing,
        )
        objective = (
            chance
            if self.options.objective == "quantile"
            else (
                norm
                + self.options.state_covariance_weight
                * jnp.trace(covariance[:NP, :NP])
                / self.options.bryson_sigma_factor**2
                + self.options.control_covariance_weight * jnp.trace(control_covariance)
            )
        )
        return jnp.concatenate(
            (
                propagated,
                LTRI.pack(jnp.linalg.cholesky(covariance_next)),
                jnp.array([chance, objective]),
            )
        )

    def terminal_output(self, values):
        """Terminal covariance Loewner order P'_N <= Dt via a Cholesky-residual encoding.

        Writing the normalised slack terminal_slack = I_6 - Dt^-1/2 P'_N Dt^-1/2,
        any PSD matrix admits a factorisation terminal_slack = G G^T with G
        lower-triangular, so constraining (terminal_slack - G G^T) to zero is
        equivalent to terminal_slack >= 0, i.e. P'_N <= Dt exactly -- jointly
        over position, velocity, and their cross-covariance.
        """

        factor = LTRI.matrix(values[:NL]) * self.std[-1, :, None]
        covariance = factor @ factor.T
        inv = jnp.sqrt(
            jnp.array(
                [self.options.position_covariance_reduction] * 3
                + [self.options.velocity_covariance_reduction] * 3
            )
        )
        target = covariance[:NP, :NP] * inv[:, None] * inv[None, :]
        margin = MTRI.matrix(values[NL : NL + NM])
        residual = jnp.eye(NP) - target - margin @ margin.T
        return MTRI.pack(residual)

    def terminal_inputs(self, x):
        return jnp.concatenate(
            (x["cholesky_factor"].reshape(self.n, NL)[-1], x["terminal_margin"])
        )

    def functions(self, x):
        local = self.local_values(self.local_inputs(x), jnp.arange(self.n))
        means = x["means"].reshape(NX, self.n + 1)
        result = {
            "mean_defects": (means[:, 1:] - local[:, :NX].T).ravel(),
            "control_chance": local[:, -2],
            "objective": jnp.sum(jnp.asarray(self.weights) * local[:, -1]),
        }
        if self.stochastic:
            entries = x["cholesky_factor"].reshape(self.n, NL)
            result["state_covariance_defects"] = (
                entries - local[:, NX : NX + NL]
            ).ravel()
            result["terminal_residual"] = (
                self.terminal_output(self.terminal_inputs(x))
                if self.terminal_bound
                else jnp.zeros(NM)
            )
            if not self.terminal_bound:
                factor = LTRI.matrix(entries[-1]) * jnp.asarray(self.std[-1, :, None])
                covariance = factor @ factor.T
                inverse_variance = jnp.array(
                    [self.options.position_covariance_reduction] * 3
                    + [self.options.velocity_covariance_reduction] * 3
                )
                result["objective"] += (
                    self.terminal_weight
                    * jnp.sum(jnp.diag(covariance)[:NP] * inverse_variance)
                    / self.options.bryson_sigma_factor**2
                )
        return result

    def objconfun(self, x):
        values = {key: np.asarray(value) for key, value in self.values(x).items()}
        return values, not all(np.all(np.isfinite(v)) for v in values.values())


def recover_solution(nlp, x, *, solver_success):
    """Recover physical arrays and independent residuals from saved solver data."""
    n, options, seed = nlp.n, nlp.options, nlp.seed
    means = np.asarray(x["means"]).reshape(NX, n + 1)
    feed = np.asarray(x["feedforward"]).reshape(NU, n)
    gains = np.asarray(x["gains"]).reshape(n, NU, NP) * seed.gain_scale[:, None, None]
    entries = np.asarray(x["cholesky_factor"]).reshape(n, NL)
    covariances = [nlp.normalization.initial_covariance]
    for k in range(n):
        factor = np.asarray(LTRI.matrix(entries[k])) * nlp.std[k + 1, :, None]
        covariances.append(factor @ factor.T)
    covariances = np.stack(covariances)
    controls, predicted_cov = [], []
    for k in range(n):
        _, p, t = nlp.arc(
            means[:, k], covariances[k], feed[:, k], gains[k], nlp.ref.steps[k]
        )
        controls.append(np.asarray(t))
        predicted_cov.append(np.asarray(p))
    controls = np.stack(controls)
    values, fail = nlp.objconfun(x)
    norm = np.sqrt(np.sum(feed**2, axis=0) + options.control_norm_eps**2)
    radius = np.asarray(
        jax.vmap(
            lambda t: spectral_radius(
                t, options.spectral_radius_floor, options.eigenvalue_smoothing
            )
        )(controls)
    )
    boundary_error = max(
        np.max(np.abs(means[:, 0] - nlp.case.x0_augmented_state)),
        np.max(
            np.abs(
                means[:NP, -1] - terminal_mean_state_target(nlp.case, options, nlp.ref)
            )
        ),
    )
    equality_error = max(
        boundary_error,
        *(
            float(np.max(np.abs(values[key])))
            for key in ("mean_defects", "state_covariance_defects", "terminal_residual")
        ),
    )
    chance_violation = max(0.0, float(np.max(values["control_chance"]) - 1))
    bound_violation = max(
        0.0,
        float(-np.min(entries[:, LTRI.diagonal])),
        float(
            options.terminal_margin_floor
            - np.min(np.asarray(x["terminal_margin"])[MTRI.diagonal])
        )
        if nlp.terminal_bound
        else 0.0,
        0.0
        if options.gain_bound is None
        else float(np.max(np.abs(np.asarray(x["gains"]))) - options.gain_bound),
    )
    terminal_violation = float(np.max(np.abs(values["terminal_residual"])))
    diagnostics = {
        "solver_converged": float(solver_success and not fail),
        "original_residual_acceptance_check": False,
        "converged": float(solver_success and not fail),
        "max_equality_residual": equality_error,
        "control_chance_covariance": "executed control; commanded dispersion, navigation and execution included",
        "max_control_chance_violation": chance_violation,
        "max_bound_violation": bound_violation,
        "max_terminal_constraint_violation": terminal_violation,
        "state_matching_defect_nd": float(np.max(np.abs(values["mean_defects"]))),
        "state_covariance_matching_defect": float(
            np.max(np.abs(np.stack(predicted_cov) - covariances[1:]))
        ),
        "scaled_state_covariance_matching_defect": float(
            np.max(np.abs(values["state_covariance_defects"]))
        ),
        "state_covariance_scale_min_variance": float(np.min(nlp.variances)),
        "state_covariance_scale_max_variance": float(np.max(nlp.variances)),
        "objective_mean_control": float(nlp.weights @ norm),
        "objective_state_covariance": float(
            sum(
                nlp.weights[k]
                * np.trace(covariances[k, :NP, :NP])
                / options.bryson_sigma_factor**2
                for k in range(n)
            )
        ),
        "objective_control_covariance": float(
            nlp.weights @ np.trace(controls, axis1=1, axis2=2)
        ),
        "objective_terminal_covariance": (
            float(
                nlp.terminal_weight
                * np.diag(covariances[-1])[:NP]
                @ np.array(
                    [options.position_covariance_reduction] * 3
                    + [options.velocity_covariance_reduction] * 3
                )
                / options.bryson_sigma_factor**2
            )
            if not nlp.terminal_bound
            else 0.0
        ),
    }
    return RobustSolution(
        means,
        feed,
        gains,
        radius,
        covariances,
        controls,
        float(values["objective"]),
        nlp.variances,
        diagnostics,
    )


RUN_PROBLEM_DATA = (
    "initial_position_std_km",
    "initial_velocity_std_km_s",
    "initial_mass_std_kg",
    "mass_scale_kg",
    "position_covariance_reduction",
    "velocity_covariance_reduction",
    "navigation_position_std_km",
    "navigation_velocity_std_km_s",
)


def max_rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def reference_trajectory(case, data):
    states = np.asarray(data["ref_states"])
    return ReferenceTraj(
        np.asarray(data["node_times_nd"]),
        np.asarray(data["steps_nd"]),
        states,
        np.asarray(data["ref_controls"]),
        float(case.m0_wet * (states[6, 0] - states[6, -1])),
    )


def decision_vector(data):
    gain_scale = np.asarray(data["gain_scale"])
    return {
        "means": np.asarray(data["means"]).ravel(),
        "feedforward": np.asarray(data["feedforward"]).ravel(),
        "gains": (
            np.asarray(data["gains_normalized"]) / gain_scale[:, None, None]
        ).ravel(),
        "cholesky_factor": np.asarray(data["cholesky_factor"]).ravel(),
        "terminal_margin": np.asarray(data["terminal_margin"]),
    }


def seed_check(case, options, normalization, dynamics, arc, ref, data):
    from stochastic_trajectory_optimization.dynamics import (
        normalized_arc_jacobians,
        tvlqr_gains,
    )

    a, b = normalized_arc_jacobians(
        case,
        dynamics,
        normalization,
        ref.states,
        ref.controls,
        ref.steps,
        options.integrator_substeps,
    )
    weights = ref.n_arcs * ref.steps / np.sum(ref.steps)
    state_weight, control_weight = (
        (1.0, 1.0)
        if options.objective == "quantile"
        else (options.state_covariance_weight, options.control_covariance_weight)
    )
    terminal = (
        np.array(
            [options.position_covariance_reduction] * 3
            + [options.velocity_covariance_reduction] * 3
        )
        / options.bryson_sigma_factor**2
    )
    gains = tvlqr_gains(
        state_weight / options.bryson_sigma_factor**2,
        control_weight,
        terminal,
        a,
        b,
        weights,
    )
    gain_scale = np.array(
        [1.0 / max(np.linalg.norm(matrix[:NP], 2), 1e-300) for matrix in b]
    )
    julia_gains = np.asarray(data["seed_gains_normalized"])
    _, covariances, _ = propagate_stochastic_moments(
        arc,
        ref.states,
        ref.controls / case.max_thrust_nd,
        julia_gains,
        normalization.initial_covariance,
        ref.steps,
    )
    return {
        "gain_scale_max_rel": max_rel(data["gain_scale"], gain_scale),
        "tvlqr_gains_max_rel": max_rel(julia_gains, gains),
        "propagated_seed_covariance_max_rel": max_rel(
            data["seed_covariances"], covariances
        ),
    }


def validate(path: Path, samples: int | None, *, plots=True):
    case = load_case(path.stem)
    data = dict(np.load(path))
    options = replace(default_options(path.stem), truncated_uniform_mesh_arcs=None)
    options = replace(
        options,
        terminal_margin_floor=float(data["terminal_margin_floor"]),
        spectral_radius_floor=float(data["spectral_radius_floor"]),
        eigenvalue_smoothing=float(data["eigenvalue_smoothing"]),
        control_norm_eps=float(data["control_norm_eps"]),
    )
    # The run's own problem data, where it records them (it may deviate from the JAX defaults).
    options = replace(
        options,
        **{name: float(data[name]) for name in RUN_PROBLEM_DATA if name in data},
    )
    # The run's gain bound (runs saved before the bound existed had none).
    options = replace(
        options, gain_bound=float(data["gain_bound"]) if "gain_bound" in data else None
    )
    # The run's control-covariance weight (runs saved before the option existed were unweighted).
    options = replace(
        options,
        control_covariance_weight=float(data["control_covariance_weight"])
        if "control_covariance_weight" in data
        else 1.0,
    )
    # The run's state-covariance weight (runs saved before the option existed had 1).
    options = replace(
        options,
        state_covariance_weight=float(data["state_covariance_weight"])
        if "state_covariance_weight" in data
        else 1.0,
    )
    # The run's objective (runs saved before the option existed used the standard cost).
    options = replace(
        options,
        objective="quantile" if int(data.get("quantile_objective", 0)) else "standard",
    )
    # Polar and throttle controls: the model's exact norm (JAX needs a positive eps; 1e-12 is exact to rounding),
    # plain mass flow and Gates factors without jitter.
    if "model_norm_smoothing" in data:
        options = replace(
            options,
            control_norm_eps=max(float(data["model_norm_smoothing"]), 1e-12),
            mass_flow_smoothing=float(data["model_mass_flow_smoothing"]),
            mass_flow_smoothing_feedback_only=bool(
                data.get("model_mass_flow_smoothing_feedback_only", 0)
            ),
            cholesky_jitter=float(data["model_gates_jitter"]),
        )
    if samples is not None:
        options = replace(options, monte_carlo_samples=samples)
    normalization = build_normalization(case, options)
    dynamics = Dynamics(case)
    ref = reference_trajectory(case, data)
    arc = jax.jit(build_propagation_arc_map(case, options, normalization))
    print(f"[{case.test_case_id}] seed check ...", flush=True)
    seed_report = seed_check(case, options, normalization, dynamics, arc, ref, data)

    print(
        f"[{case.test_case_id}] solution check with the reference NLP functions ...",
        flush=True,
    )
    nlp = TrajectoryResiduals(case, options, ref, normalization, arc, data)
    success = bool(data["solver_success"]) and bool(data["restoration_requirement_met"])
    solution = recover_solution(nlp, decision_vector(data), solver_success=success)

    print(
        f"[{case.test_case_id}] Monte Carlo ({options.monte_carlo_samples} samples) and figures ...",
        flush=True,
    )
    monte_carlo = monte_carlo_rollouts(
        case, options, normalization, solution, ref.steps
    )
    if case.test_case_id == "earth_mars":
        monte_carlo["no_feedback"] = monte_carlo_rollouts(
            case, options, normalization, solution, ref.steps, feedback_enabled=False
        )
    psi = psi_inverse(NU, options.violation_parameter)
    diagnostics = {
        **compute_diagnostics(
            case, options, ref, solution, normalization, monte_carlo, psi
        ),
        **solution.diagnostics,
    }
    diagnostics["julia_objective_nd"] = float(data["objective"])
    diagnostics["independent_objective_difference"] = solution.objective - float(
        data["objective"]
    )
    diagnostics["covariance_defect_form"] = "seed-scaled square-root"
    diagnostics["terminal_covariance_constraint"] = nlp.terminal_bound
    diagnostics["terminal_covariance_weight"] = (
        0.0 if nlp.terminal_bound else nlp.terminal_weight
    )
    output_prefix = path.parent / "validation" / case.test_case_id
    save_outputs(
        case,
        options,
        ref,
        solution,
        normalization,
        monte_carlo,
        diagnostics,
        psi,
        output_prefix,
        dynamics,
        plots=plots,
    )
    report = {
        "seed_check": seed_report,
        **{
            k: float(v) if isinstance(v, (int, float, np.number)) else v
            for k, v in diagnostics.items()
        },
    }
    (path.parent / "validation" / f"{case.test_case_id}_validation.json").write_text(
        json.dumps(report, indent=2)
    )
    print_summary(case, report, options)


def print_summary(case, r, options):
    s = r["seed_check"]
    print(f"\n[{case.test_case_id}] validation with the reference code")
    print(
        f"  seed vs reference: gain scale {s['gain_scale_max_rel']:.1e}, TVLQR gains {s['tvlqr_gains_max_rel']:.1e}, "
        f"propagated covariances {s['propagated_seed_covariance_max_rel']:.1e} (max relative difference)"
    )
    print(
        f"  objective {r['objective_nd']:.10f} (Julia {r['julia_objective_nd']:.10f})"
    )
    print(
        f"  max equality residual {r['max_equality_residual']:.3e}, chance violation "
        f"{r['max_control_chance_violation']:.3e}, bound violation {r['max_bound_violation']:.3e}"
    )
    print(
        "  accepted (MadNLP solver success and restoration requirement; residuals diagnostic only): "
        f"{'yes' if r['converged'] else 'no'}"
    )
    print(
        f"  terminal covariance / target, largest eigenvalue: predicted {r['terminal_covariance_max_eigenvalue']:.6f}, "
        f"Monte Carlo {r['monte_carlo_terminal_max_eigenvalue']:.6f}"
    )
    print(
        f"  worst-arc thrust violation frequency {r['monte_carlo_max_violation_fraction']:.4%} "
        f"(limit {options.violation_parameter:.2%})"
    )
