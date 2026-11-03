from integrator.stochastic import ensemble_integrator as ensemble_integrator

"""Continuous ensemble propagation with sample-specific Mars SOI switches.

The optimizer uses scheduled nominal switches. This diagnostic deliberately
tests the different hybrid model requested in the attached PDF; no saltation
correction is inserted into the optimization covariance.

Everything runs in JAX on its default backend (CPU or GPU, chosen by the
caller): the policy, execution error and Brownian draws of each interval in
one jitted function, the RK9 event integrator split over all local devices.
Random draws are keyed by seed, stream, interval and sample index, so CPU,
GPU and any batch size give the same realizations up to roundoff.
"""

from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from verification.stochastic.psyche_model import bplane_vector, scaled_bplane


# Fewer RK9 substeps under-resolve the post-flyby dispersion growth.
MONTE_CARLO_MIN_SUBSTEPS = 16


def nominal_history(data, variables, points_per_arc=16):
    """Heliocentric mean path, restarted from every node mean (as Earth-Mars's
    densified mean): scheduled phases, feedforward thrust, no noise."""
    advance = ensemble_integrator(data, sample_specific=False)
    means = variables["means"] * data.node_scales
    days = variables["times"]
    et = data.mission.t0
    epochs, states = [et], [means[0]]
    for arc in data.arcs:
        y = means[arc.start].copy()
        if arc.phase == 1:
            y[:6] += data.mission.cache.state(499, et)
        y = jnp.asarray(y[None])
        force = jnp.asarray(
            data.mission.settings.thrust_max_n
            * (
                np.zeros(3) if arc.phase == 1 else variables["feedforward"][arc.control]
            )[None]
        )
        mode = jnp.full(1, arc.phase, dtype=jnp.int32)
        duration = float(arc.duration[0] + arc.duration[1:] @ days)
        for _ in range(points_per_arc):
            y = advance(
                y,
                force,
                jnp.zeros((1, 3)),
                et,
                duration / points_per_arc,
                mode,
                jnp.full(1, jnp.inf),
                jnp.full(1, et),
                y,
                jnp.full(1, jnp.nan),
                jnp.full(1, jnp.nan),
            )[0]
            et += duration / points_per_arc
            epochs.append(et)
            states.append(np.asarray(y[0]))
    return np.array(epochs), np.array(states).T


def run_monte_carlo(
    data,
    variables,
    gain_scale,
    samples=50000,
    seed=42,
    substeps=MONTE_CARLO_MIN_SUBSTEPS,
    batch_size=None,
    sample_specific=True,
    plot_count=100,
):
    """batch_size=None propagates every sample at once; a smaller batch only bounds memory."""
    batch_size = samples if batch_size is None else batch_size
    if substeps < MONTE_CARLO_MIN_SUBSTEPS:
        raise ValueError(
            f"Monte Carlo needs at least {MONTE_CARLO_MIN_SUBSTEPS} RK9 substeps"
        )
    if samples < 2 or batch_size < 1:
        raise ValueError(
            "Covariance diagnostics require at least two samples and a positive batch size"
        )
    # Common fine increments couple the 16/32-substep comparison.
    fine_count = max(32, substeps)
    if fine_count % substeps:
        raise ValueError("Use substeps dividing 32, or at least 32")
    advance = ensemble_integrator(data, sample_specific=sample_specific)
    root = jnp.asarray(np.linalg.cholesky(data.initial_covariance_si))
    means = variables["means"] * data.node_scales
    gains = variables["gains"] * gain_scale[:, None, None]
    days = variables["times"]
    cfg = data.mission.settings
    initial_key, navigation_key, execution_key, process_key = jax.random.split(
        jax.random.key(seed), 4
    )
    navigation_std = jnp.asarray(
        np.r_[[2e5] * 3, [0.1] * 3] / data.covariance_scale[:6]
    )
    scale = jnp.asarray(data.covariance_scale[:6])

    def draw(key, index, first, count, shape):
        # Draws of every sample, then this batch's rows: independent of batch size.
        return jax.lax.dynamic_slice_in_dim(
            jax.random.normal(jax.random.fold_in(key, index), (samples,) + shape),
            first,
            count,
        )

    @partial(jax.jit, static_argnames=("count", "ballistic"))
    def interval(y, index, first, mean, feedforward, gain, duration, count, ballistic):
        """Executed, saturated force, Brownian increments and violation counts of one interval."""
        if ballistic:
            command = jnp.zeros((count, 3))
        else:
            nav = draw(navigation_key, index, first, count, (6,)) * navigation_std
            command = feedforward + ((y[:, :6] - mean[:6]) / scale + nav) @ gain.T
        norm = jnp.linalg.norm(command, axis=1)
        unit = command / jnp.maximum(norm[:, None], 1e-300)
        xi = draw(execution_key, index, first, count, (3,))
        parallel = jnp.sum(xi * unit, axis=1)[:, None] * unit
        executed = command + norm[:, None] * (
            0.008 * parallel + 0.0095993 * (xi - parallel)
        )
        executed_norm = jnp.linalg.norm(executed, axis=1)
        applied = executed / jnp.maximum(executed_norm, 1)[:, None]
        fine = draw(process_key, index, first, count, (fine_count, 3)) * jnp.sqrt(
            duration / fine_count
        )
        dw = fine.reshape(count, substeps, fine_count // substeps, 3).sum(axis=2)
        return (
            cfg.thrust_max_n * applied,
            dw,
            jnp.sum(norm > 1),
            jnp.sum(executed_norm > 1),
            cfg.thrust_max_n * jnp.linalg.norm(applied - feedforward, axis=1),
        )

    arrays = {
        k: []
        for k in (
            "terminal_si",
            "closest_state_si",
            "closest_et",
            "minimum_altitude_m",
            "entry_et",
            "exit_et",
        )
    }
    # The first plot_count realizations, at every RK9 substep, for figures.
    plot_count = min(plot_count, samples, batch_size)
    history, history_et, deviation_norm = [], [], []
    violations = np.zeros(data.n_controls)
    executed_violations = np.zeros(data.n_controls)
    invalid_total = 0
    for first in range(0, samples, batch_size):
        count = min(batch_size, samples - first)
        y = (
            jnp.asarray(data.mission.x0)
            + draw(initial_key, 0, first, count, (7,)) @ root.T
        )
        mode = jnp.zeros(count, dtype=jnp.int32)
        best_r = jnp.full(count, jnp.inf)
        best_t = jnp.full(count, data.mission.t0)
        best_y = y
        entry = exit = jnp.full(count, jnp.nan)
        et = data.mission.t0
        if first == 0:
            history.append(y[:plot_count])
            history_et.append(et)
        for index, arc in enumerate(data.arcs):
            if not sample_specific:
                if arc.start == data.mars_entry_node:
                    entry = jnp.full(count, et)
                if arc.start == data.mars_exit_node:
                    exit = jnp.full(count, et)
                mode = jnp.full(count, arc.phase, dtype=jnp.int32)
            mean = means[arc.start].copy()
            if arc.phase == 1:
                mean[:6] += data.mission.cache.state(499, et)
            duration = float(arc.duration[0] + arc.duration[1:] @ days)
            force, dw, command_count, executed_count, deviation = interval(
                y,
                index,
                first,
                jnp.asarray(mean),
                jnp.asarray(variables["feedforward"][arc.control]),
                jnp.asarray(gains[arc.control]),
                duration,
                count=count,
                ballistic=arc.phase == 1,
            )
            violations[arc.control] += int(command_count)
            executed_violations[arc.control] += int(executed_count)
            if first == 0:
                deviation_norm.append(deviation[:plot_count])
            for k in range(substeps):
                y, mode, best_r, best_t, best_y, entry, exit = advance(
                    y,
                    force,
                    dw[:, k],
                    et,
                    duration / substeps,
                    mode,
                    best_r,
                    best_t,
                    best_y,
                    entry,
                    exit,
                )
                et += duration / substeps
                if first == 0:
                    history.append(y[:plot_count])
                    history_et.append(et)
        terminal = np.asarray(y)
        closest = np.array(best_y)
        times = np.asarray(best_t)
        closest[:, :6] -= data.mission.cache.state(499, times)
        arrays["terminal_si"].append(terminal)
        arrays["closest_state_si"].append(closest)
        arrays["closest_et"].append(times)
        arrays["minimum_altitude_m"].append(np.asarray(best_r) - cfg.mars_radius_m)
        arrays["entry_et"].append(np.asarray(entry))
        arrays["exit_et"].append(np.asarray(exit))
        invalid_total += int(
            np.sum(~np.isfinite(terminal).all(axis=1) | (terminal[:, 6] <= 0))
        )
    history = np.asarray(jnp.stack(history, axis=2)).transpose(1, 0, 2)
    history_et = np.array(history_et)
    deviation_norm = np.asarray(jnp.stack(deviation_norm))
    result = {key: np.concatenate(value) for key, value in arrays.items()}
    z = result["closest_state_si"][:, :6]
    c = np.sum(z[:, 3:6] ** 2, axis=1) - 2 * data.mission.scales[1].mu / np.linalg.norm(
        z[:, :3], axis=1
    )
    rho = cfg.mars_radius_m + 500e3
    result["bplane_si"] = np.asarray(
        jax.jit(jax.vmap(lambda x: scaled_bplane(x, data.mission.scales[1].mu, rho)))(
            jnp.asarray(z)
        )
    )
    # Unscaled [B.R, B.T] of each sample, in the B-plane of its own osculating
    # hyperbola at its own closest approach.
    result["bplane_unscaled_si"] = np.asarray(
        jax.jit(jax.vmap(lambda x: bplane_vector(x, data.mission.scales[1].mu)[0]))(
            jnp.asarray(z)
        )
    )
    result["plot_trajectory_history_si"] = history
    result["plot_epochs_et"] = history_et
    result["plot_applied_control_deviation_norm_n"] = deviation_norm
    invalid = (
        (c <= 0)
        | ~np.isfinite(result["bplane_si"]).all(axis=1)
        | ~np.isfinite(result["terminal_si"]).all(axis=1)
        | (result["terminal_si"][:, 6] <= 0)
    )
    unsafe = result["minimum_altitude_m"] < 500e3
    # Exact binomial interval is a sampling uncertainty statement, not a
    # certification of the chance approximation or hybrid model.
    from scipy.stats import beta

    count = int(np.sum(unsafe | invalid))
    confidence = [
        0.0 if count == 0 else float(beta.ppf(0.025, count, samples - count + 1)),
        1.0 if count == samples else float(beta.ppf(0.975, count + 1, samples - count)),
    ]
    finite_terminal = bool(np.isfinite(result["terminal_si"]).all())
    if finite_terminal:
        terminal_covariance = np.cov(result["terminal_si"], rowvar=False, ddof=1)
        target_std = np.r_[[2e6] * 3, [2.0] * 3]
        terminal_ratio = terminal_covariance[:6, :6] / np.outer(target_std, target_std)
        terminal_eigenvalue = float(
            np.linalg.eigvalsh((terminal_ratio + terminal_ratio.T) / 2)[-1]
        )
        terminal_mean_error = (
            np.mean(result["terminal_si"][:, :6], axis=0) - data.mission.xf
        ).tolist()
    else:
        terminal_covariance = terminal_eigenvalue = terminal_mean_error = None
    report = dict(
        samples=samples,
        seed=seed,
        substeps=substeps,
        batch_size=batch_size,
        backend=jax.default_backend(),
        random_draws="jax.random keyed by seed, stream, interval and sample index",
        mode="sample-specific SOI crossings"
        if sample_specific
        else "scheduled nominal switches",
        unsafe_altitude_count=int(np.sum(result["minimum_altitude_m"] < 500e3)),
        unsafe_altitude_fraction=float(np.mean(result["minimum_altitude_m"] < 500e3)),
        invalid_hyperbolic_count=int(
            np.sum((c <= 0) | ~np.isfinite(result["bplane_si"]).all(axis=1))
        ),
        invalid_propagation_count=invalid_total,
        invalid_or_unsafe_count=count,
        invalid_or_unsafe_fraction=count / samples,
        invalid_or_unsafe_binomial_95_interval=confidence,
        missing_entry_count=int(np.sum(~np.isfinite(result["entry_et"]))),
        missing_exit_count=int(np.sum(~np.isfinite(result["exit_et"]))),
        command_violation_fraction=(violations / samples).tolist(),
        executed_violation_fraction_before_saturation=(
            executed_violations / samples
        ).tolist(),
        control_chance_violation_fraction=(executed_violations / samples).tolist(),
        control_chance_covariance="executed; command dispersion and execution uncertainty included",
        terminal_std_si=np.std(result["terminal_si"], axis=0, ddof=1).tolist()
        if finite_terminal
        else None,
        terminal_mean_error_si=terminal_mean_error,
        terminal_covariance_si=terminal_covariance.tolist()
        if finite_terminal
        else None,
        terminal_covariance_target_max_eigenvalue=terminal_eigenvalue,
        terminal_statistics_qualification="Finite-sample estimates; no invalid trajectories removed. Null if nonfinite terminal states occur.",
        event_root_iterations=38,
        control_updates=data.n_powered_controls,
        ballistic_mars=True,
        qualification="Central-body switching differs from scheduled covariance propagation; no sample reset or saltation correction",
    )
    return result, report
