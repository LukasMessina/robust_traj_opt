"""Physical diagnostics and integration studies for a saved Psyche policy."""

from dataclasses import replace
from pathlib import Path
import json
import tomllib
from verification.stochastic.psyche_model import PsycheInput, ROOT
from verification.stochastic.psyche_residuals import PsycheResiduals
from verification.stochastic.psyche_monte_carlo import (
    run_monte_carlo,
    MONTE_CARLO_MIN_SUBSTEPS,
)
from plotter.psyche import plot_diagnostics
import numpy as np
import jax
import jax.numpy as jnp
import stochastic_trajectory_optimization.dynamics as common
from verification.stochastic.psyche_model import Maps
from verification.stochastic.psyche_model import encounter_statistics


def nominal_linearization_accuracy(nlp, z):
    """Compare nominal map Jacobians and recomputed TVLQR gains at a fixed path.

    This is not a reoptimisation of the stochastic feedback gains. State,
    control and free-duration derivatives use the NLP's dimensionless units.
    """
    from verification.stochastic.psyche_residuals import PsycheResiduals

    d, o = nlp.data, nlp.options
    variables = nlp.unpack(z)
    coarse = PsycheResiduals(d, o)
    fine = PsycheResiduals(
        d,
        replace(
            o,
            integrator_substeps=2 * o.integrator_substeps,
            sde_integrator_substeps=2 * o.sde_integrator_substeps,
        ),
    )
    nominal_z = coarse.x0.copy()
    for name in ("means", "feedforward", "times"):
        value = (
            variables[name][d.powered_arcs]
            if name == "feedforward"
            else variables[name]
        )
        nominal_z[coarse.slices[name]] = value.ravel()
    kc, _, jc = weighted_tvlqr(coarse, nominal_z)
    kf, _, jf = weighted_tvlqr(fine, nominal_z)
    phases = []
    for phase in range(3):
        ids = [a.control for a in d.arcs if a.phase == phase]
        errors = {name: [] for name in ("state", "control", "duration")}
        for index in ids:
            for name, cols in (
                ("state", slice(0, 7)),
                ("control", slice(7, 10)),
                ("duration", slice(-2, None)),
            ):
                delta = jf[index][:7, cols] - jc[index][:7, cols]
                errors[name].append(float(np.max(abs(delta))))
        change = kf[ids] - kc[ids]
        phases.append(
            dict(
                phase=phase,
                jacobian_max_absolute={k: max(v) for k, v in errors.items()},
                tvlqr_gain_max_absolute=float(np.max(abs(change))),
                tvlqr_gain_relative_frobenius=float(
                    np.linalg.norm(change) / max(np.linalg.norm(kf[ids]), 1e-30)
                ),
            )
        )
    return dict(
        substeps=[o.integrator_substeps, 2 * o.integrator_substeps],
        phases=phases,
        qualification="Fixed nominal path; recomputed TVLQR gains, not reoptimised stochastic gains",
    )


def recover(nlp, z):
    d = nlp.data
    v = nlp.unpack(z)
    cov = [d.initial_covariance]
    for i, packed in enumerate(v["factors"]):
        factor = nlp.std[i + 1, :, None] * np.asarray(common.LTRI.matrix(packed))
        cov.append(factor @ factor.T)
    et = np.empty(d.n_nodes)
    et[0] = d.mission.t0
    weights = np.empty(d.n_intervals)
    controls = np.empty((d.n_intervals, 3))
    for a in d.arcs:
        duration = float(a.duration[0] + a.duration[1:] @ v["times"])
        et[a.end] = et[a.start] + duration
        weights[a.end - 1] = d.n_intervals * duration / (d.mission.tf - d.mission.t0)
        controls[a.end - 1] = v["feedforward"][a.control]
    cov = np.array(cov)
    cov_si = cov * d.covariance_scale[None, :, None] * d.covariance_scale[None, None, :]
    return dict(
        **v,
        epochs_et=et,
        covariances=cov,
        covariance_si=cov_si,
        states_si=v["means"] * d.node_scales,
        gains_normalized=v["gains"] * nlp.gain_scale[:, None, None],
        interval_weights=weights,
        interval_controls=controls,
    )


def physical_diagnostics(nlp, z):
    d, o = nlp.data, nlp.options
    x = recover(nlp, z)
    values = nlp.values(z)
    residual = values["mean_defects"].reshape(-1, 7) * d.node_scales[1:]
    ca = x["states_si"][d.ca_node]
    acc = -d.mission.scales[1].mu * ca[:3] / np.linalg.norm(ca[:3]) ** 3
    encounter = {
        k: np.asarray(v)
        for k, v in encounter_statistics(
            jnp.asarray(ca[:6]),
            jnp.asarray(x["covariance_si"][d.ca_node, :6, :6]),
            jnp.asarray(acc),
            d.mission.scales[1].mu,
            d.mission.settings.mars_radius_m + 500e3,
        ).items()
    }
    inv = d.terminal_inverse_std
    terminal = x["covariances"][-1, :6, :6] * np.outer(inv, inv)
    state_cost = 0.0
    mean_control_cost = 0.0
    for a in d.arcs:
        w = x["interval_weights"][a.end - 1]
        state_cost += (
            w
            * o.state_covariance_weight
            * np.trace(x["covariances"][a.start, :6, :6])
            / o.bryson_sigma_factor**2
        )
        if a.phase != 1:
            feed = x["feedforward"][a.control]
            mean_control_cost += w * np.sqrt(np.dot(feed, feed) + o.control_norm_eps**2)
    # Standard cost: mean control + state covariance + weighted executed covariance.
    # Quantile cost: mean control + psi^-1 * executed-control spectral radius.
    total = float(values["objective"])
    terminal_cost = (
        float(np.trace(terminal) / o.bryson_sigma_factor**2)
        if nlp.terminal_covariance_cost
        else 0.0
    )
    terms = (
        dict(
            objective_state_covariance=float(state_cost),
            objective_weighted_executed_control_covariance=total
            - mean_control_cost
            - state_cost
            - terminal_cost,
            objective_share_mean_control=mean_control_cost / total,
            objective_share_state_covariance=state_cost / total,
            objective_share_control_covariance=(
                total - mean_control_cost - state_cost - terminal_cost
            )
            / total,
        )
        if o.objective == "standard"
        else dict(
            objective_quantile_dispersion=float(values["objective"] - mean_control_cost)
        )
    )
    report = dict(
        objective=float(values["objective"]),
        objective_kind=o.objective,
        objective_terminal_covariance=terminal_cost,
        objective_mean_control=float(mean_control_cost),
        **terms,
        mean_matching_position_m=float(np.max(np.linalg.norm(residual[:, :3], axis=1))),
        mean_matching_velocity_m_s=float(
            np.max(np.linalg.norm(residual[:, 3:6], axis=1))
        ),
        mean_matching_mass_kg=float(np.max(abs(residual[:, 6]))),
        covariance_matching_scaled=float(np.max(abs(values["covariance_defects"]))),
        terminal_psd_max_eigenvalue=float(np.linalg.eigvalsh(terminal)[-1]),
        terminal_std_si=np.sqrt(np.diag(x["covariance_si"][-1])).tolist(),
        mean_mass_min_kg=float(np.min(x["states_si"][:, 6])),
        mean_final_mass_kg=float(x["states_si"][-1, 6]),
        closest_approach_et=float(x["epochs_et"][d.ca_node]),
        mars_exit_et=float(x["epochs_et"][d.mars_exit_node]),
        closest_approach_mean_altitude_m=float(
            np.linalg.norm(ca[:3]) - d.mission.settings.mars_radius_m
        ),
        control_chance_covariance="executed; commanded dispersion and execution uncertainty included",
        maximum_control_chance_budget=float(np.max(values["control_chance"])),
        maximum_scaled_gain=float(np.max(abs(x["gains"]))),
        encounter=encounter,
    )
    report["nominal_command_energy_n2_s"] = float(
        sum(
            0.5
            * d.mission.settings.thrust_max_n**2
            * (x["epochs_et"][i + 1] - x["epochs_et"][i])
            * np.dot(u, u)
            for i, u in enumerate(x["interval_controls"])
        )
    )
    report["mean_propellant_kg"] = float(x["states_si"][0, 6] - x["states_si"][-1, 6])
    jumps = []
    for node, entering in ((d.mars_entry_node, True), (d.mars_exit_node, False)):
        t = x["epochs_et"][node]
        planet, rate = jax.jvp(d.planet, (jnp.asarray(t),), (jnp.array(1.0),))
        planet, rate = np.asarray(planet), np.asarray(rate)
        native = x["states_si"][node, :6]
        relative = native if entering else native - planet
        solar = relative + planet
        a_sun = -d.mission.scales[0].mu * solar[:3] / np.linalg.norm(solar[:3]) ** 3
        a_mars = (
            -d.mission.scales[1].mu * relative[:3] / np.linalg.norm(relative[:3]) ** 3
        )
        # f+ - DxR f- - R_t with identical held thrust. A separate commanded
        # update, if present, is already part of the scheduled control policy.
        jump = np.r_[rate[:3] - planet[3:6], a_mars - a_sun + rate[3:6]] * (
            1 if entering else -1
        )
        normal = relative[:3] / np.linalg.norm(relative[:3])
        radial_speed = float(
            normal @ (relative[3:6] + (planet[3:6] - rate[:3] if entering else 0.0))
        )
        time_gradient = -normal / radial_speed
        time_sigma = float(
            np.sqrt(
                max(
                    time_gradient @ x["covariance_si"][node, :3, :3] @ time_gradient,
                    0.0,
                )
            )
        )
        jumps.append(
            dict(
                event="entry" if entering else "exit",
                velocity_jump_m_s=jump[:3],
                acceleration_jump_m_s2=jump[3:],
                acceleration_jump_norm_m_s2=float(np.linalg.norm(jump[3:])),
                radial_crossing_speed_m_s=radial_speed,
                linearized_crossing_time_std_s=time_sigma,
                velocity_effect_scale_m_s=float(np.linalg.norm(jump[3:]) * time_sigma),
                qualification="Local first-order scale; no propagation of a saltation correction",
            )
        )
    report["gravity_frame_vector_field_mismatch"] = jumps
    report["control_chance_budget_by_interval"] = [
        (
            0.0
            if d.control_indices[a.control] < 0
            else float(values["control_chance"][d.control_indices[a.control]])
        )
        for a in d.arcs
    ]
    # Chance budget minus the smoothed feedforward norm: psi^-1 times the
    # executed-control spectral radius, the predicted applied-thrust deviation.
    report["control_dispersion_bound_by_interval"] = [
        (
            0.0
            if d.control_indices[a.control] < 0
            else budget
            - float(
                np.sqrt(
                    np.dot(x["feedforward"][a.control], x["feedforward"][a.control])
                    + o.control_norm_eps**2
                )
            )
        )
        for a, budget in zip(d.arcs, report["control_chance_budget_by_interval"])
    ]
    return x, report


def moment_accuracy(nlp, z, *, change_noise_grid=False):
    """Compare the NLP's RK9 moment map with one using twice as many steps.

    With change_noise_grid=False the UT points, weights and Wiener paths are
    identical: each Wiener increment is divided equally between two RK9 steps.
    True selects twice as many independent Wiener increments and hence a
    different augmented UT ensemble; its discrepancy is not pure ODE error.
    The closest-approach clearance change is evaluated on the arc ending at CA.
    """
    d, o = nlp.data, nlp.options
    x = recover(nlp, z)
    fine_options = (
        replace(
            o,
            integrator_substeps=2 * o.integrator_substeps,
            sde_integrator_substeps=2 * o.sde_integrator_substeps,
        )
        if change_noise_grid
        else o
    )
    fine = Maps(d, fine_options, integration_refinement=1 if change_noise_grid else 2)
    rho = d.mission.settings.mars_radius_m + 500e3
    mu = d.mission.scales[1].mu

    def radius(executed):
        return common.psi_inverse(3, o.violation_parameter) * common.spectral_radius(
            executed, o.spectral_radius_floor, o.eigenvalue_smoothing
        )

    def clearance(end, covariance):
        state = end * jnp.asarray(d.node_scales[d.ca_node])
        scale = jnp.asarray(d.covariance_scale[:6])
        acc = -mu * state[:3] / jnp.linalg.norm(state[:3]) ** 3
        return encounter_statistics(
            state[:6],
            covariance[:6, :6] * scale[:, None] * scale[None, :],
            acc,
            mu,
            rho,
        )["chance_clearance"]

    functions = {}
    statistics = [[] for _ in range(3)]
    for a in d.arcs:
        key = (a.phase, a.switch, a.end == d.ca_node)
        if key not in functions:

            def compare(mean, p, u, k, t, duration, template=a):
                arc = replace(template, duration=duration)
                coarse = nlp.maps.stochastic(arc, mean, p, u, k, t)
                refined = fine.stochastic(arc, mean, p, u, k, t)
                encounter_change = (
                    clearance(*refined[:2]) - clearance(*coarse[:2])
                    if template.end == d.ca_node
                    else jnp.array(0.0)
                )
                return (
                    *(b - c for c, b in zip(coarse, refined)),
                    radius(refined[2]) - radius(coarse[2]),
                    encounter_change,
                )

            functions[key] = jax.jit(compare)
        dy, dp, dce, dchance, dencounter = map(
            np.asarray,
            functions[key](
                x["means"][a.start],
                x["covariances"][a.start],
                x["feedforward"][a.control],
                x["gains_normalized"][a.control],
                x["times"],
                a.duration,
            ),
        )
        dy_si = dy * d.node_scales[a.end]
        dp_si = dp * d.covariance_scale[:, None] * d.covariance_scale[None, :]
        std = nlp.std[a.end]
        statistics[a.phase].append(
            dict(
                position_m=float(np.linalg.norm(dy_si[:3])),
                velocity_m_s=float(np.linalg.norm(dy_si[3:6])),
                mass_kg=float(abs(dy_si[6])),
                covariance_normalized=float(np.max(abs(dp))),
                covariance_seed_scaled=float(
                    np.max(abs(dp / std[:, None] / std[None, :]))
                ),
                position_variance_m2=float(np.max(abs(dp_si[:3, :3]))),
                velocity_variance_m2_s2=float(np.max(abs(dp_si[3:6, 3:6]))),
                executed_covariance_normalized=float(np.max(abs(dce))),
                executed_chance_budget_difference=float(abs(dchance)),
                executed_chance_budget_difference_n=float(
                    abs(dchance) * d.mission.settings.thrust_max_n
                ),
                ca_clearance_difference_normalized=float(abs(dencounter)),
                ca_clearance_difference_m=float(abs(dencounter) * rho),
            )
        )
    return dict(
        substeps=[o.sde_integrator_substeps, 2 * o.sde_integrator_substeps],
        comparison="independent finer Wiener/UT grid"
        if change_noise_grid
        else "identical UT and prescribed Wiener paths",
        phases=[
            dict(phase=i, **{k: max(row[k] for row in rows) for k in rows[0]})
            for i, rows in enumerate(statistics)
        ],
    )


def weighted_tvlqr(nlp, z):
    """Duration-weighted Bryson TVLQR on the nominal shooting Jacobians.

    Running weights are w_k*Q and w_k*R with w_k = N*dt_k/T_total. Ballistic Mars
    arcs keep K = 0 but still propagate the Riccati matrix, so the preceding
    cruise feedback accounts for the uncontrolled flyby.
    """
    data, options = nlp.data, nlp.options
    days = nlp.unpack(z)["times"]
    jacobians = nlp.evaluate_local(z, True)
    total = data.mission.tf - data.mission.t0
    qbase = options.state_covariance_weight * np.eye(6) / options.bryson_sigma_factor**2
    rbase = options.control_covariance_weight * np.eye(3)
    s = np.diag(data.terminal_inverse_std**2) / options.bryson_sigma_factor**2
    gains, gain_scale = np.empty((data.n_controls, 3, 6)), np.empty(data.n_controls)
    for index in range(data.n_controls - 1, -1, -1):
        arc, jac = data.arcs[index], jacobians[index]
        input_scale = data.covariance_scale / data.node_scales[arc.start]
        output_scale = data.covariance_scale / data.node_scales[arc.end]
        a = jac[:6, :6] * input_scale[None, :6] / output_scale[:6, None]
        b = jac[:6, 7:10] / output_scale[:6, None]
        gain_scale[index] = (
            1.0 if arc.phase == 1 else 1 / max(np.linalg.norm(b, 2), 1e-12)
        )
        w = data.n_intervals * float(arc.duration[0] + arc.duration[1:] @ days) / total
        q, r = w * qbase, w * rbase
        k = (
            np.zeros((3, 6))
            if arc.phase == 1
            else -np.linalg.solve(r + b.T @ s @ b, b.T @ s @ a)
        )
        gains[index] = k
        closed = a + b @ k
        s = q + k.T @ r @ k + closed.T @ s @ closed
        s = (s + s.T) / 2
    return gains, gain_scale, jacobians


def validate_saved_solution(
    directory,
    plots=True,
    samples=None,
    monte_carlo_seed=None,
    substeps=MONTE_CARLO_MIN_SUBSTEPS,
):
    """samples/monte_carlo_seed default to the other cases' Monte Carlo settings."""
    directory = Path(directory)
    report = tomllib.loads((directory / "psyche_diagnostics.toml").read_text())
    if not (directory / "psyche_solution.npz").exists():
        raise ValueError(
            "No native stochastic candidate was saved; see failed_stage in the diagnostics"
        )
    # The run's mesh (runs exported before the refinement option used the energy mesh).
    data = PsycheInput(
        path=ROOT / report["reference_path"] if "reference_path" in report else None,
        mesh_refinement=report.get("mesh_refinement", 1),
        cruise_subdivision=report.get("cruise_subdivision", 1),
        initial_position_std_m=1000 * report.get("initial_position_std_km", 2000.0),
        terminal_std_si=(
            1000 * report.get("terminal_position_std_km", 2000.0),
            1000 * report.get("terminal_velocity_std_km_s", 0.002),
        ),
    )
    if data.n_intervals != report["n_intervals"]:
        raise ValueError(
            f"The run has {report['n_intervals']} intervals, the reference mesh {data.n_intervals}"
        )
    settings = dict(report["settings"])
    # TOML stores a non-finite gain bound as the string "Inf" (no gain bound)
    settings["gain_bound"] = float(settings["gain_bound"])
    # Polar controls: the model's exact norm (JAX needs a positive eps; 1e-12 is exact to rounding), plain mass
    # flow and Gates factors without jitter (on coast arcs a 1e-12 jitter outweighs Gamma^2 Q_G(d)).
    extra = {}
    if "model_norm_smoothing" in settings:
        settings["control_norm_eps"] = max(settings["model_norm_smoothing"], 1e-12)
        settings["mass_flow_smoothing"] = settings["model_mass_flow_smoothing"]
        extra["cholesky_jitter"] = settings.get("model_gates_jitter", 0.0)
        extra["mass_flow_smoothing_feedback_only"] = settings.get(
            "model_mass_flow_smoothing_feedback_only", True
        )
    options = replace(
        common.default_options("earth_mars"),
        **extra,
        integrator_substeps=report["substeps_per_interval"],
        sde_integrator_substeps=report["substeps_per_interval"],
        **{
            k: settings[k]
            for k in (
                "control_norm_eps",
                "mass_flow_smoothing",
                "gain_bound",
                "control_covariance_weight",
                "eigenvalue_smoothing",
                "spectral_radius_floor",
                "terminal_margin_floor",
            )
        },
        objective={"chance_sum": "quantile"}.get(
            settings.get("objective", "standard"), settings.get("objective", "standard")
        ),
        state_covariance_weight=settings.get("state_covariance_weight", 1.0),
    )
    native = np.load(directory / "psyche_solution.npz")
    saved_seed = np.load(directory / "psyche_seed.npz")
    seed = dict(
        means=saved_seed["means"].T,
        feedforward=saved_seed["feedforward"].T,
        times=saved_seed["days"],
        gains=saved_seed["gains"].transpose(2, 0, 1),
        gain_scale=saved_seed["gain_scale"],
        covariances=saved_seed["covariances"].transpose(2, 0, 1),
    )
    free = report.get("mars_events", "free").startswith("free")
    nlp = PsycheResiduals(
        data,
        options,
        seed=seed,
        covariance_std=native["covariance_std"].T,
        mars_days=None if free else native["days"],
        terminal_covariance_cost=settings.get("objective", "standard") == "standard",
    )
    z = nlp.x0.copy()
    for key, values in dict(
        means=native["means"].T,
        feedforward=native["feedforward"].T,
        times=native["days"],
        gains=native["gains"].transpose(2, 0, 1) / native["gain_scale"][:, None, None],
        factors=native["factors"].T,
        margin=native["margin"],
    ).items():
        if key not in nlp.slices:
            continue
        if key in ("feedforward", "gains"):
            values = values[data.powered_arcs]
        z[nlp.slices[key]] = values.ravel()
    arrays, physical = physical_diagnostics(nlp, z)
    physical.update(
        converged=report["converged"],
        native_solver_status=report["madnlp_status"],
        native_solver_success=report["solver_success"],
        restoration_requirement_met=report["restoration_requirement_met"],
        native_objective=report["objective"],
        independent_objective_difference=physical["objective"] - report["objective"],
        qualification="Independent post-solve verification; no residual acceptance gate",
    )
    np.savez_compressed(directory / "psyche_physical_solution.npz", **arrays)
    (directory / "psyche_physical_diagnostics.json").write_text(
        json.dumps(
            physical,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item(),
        )
        + "\n"
    )
    table = np.column_stack(
        (
            arrays["epochs_et"],
            data.node_origins,
            arrays["states_si"],
            np.sqrt(
                np.maximum(np.diagonal(arrays["covariance_si"], axis1=1, axis2=2), 0)
            ),
        )
    )
    np.savetxt(
        directory / "psyche.csv",
        table,
        delimiter=",",
        comments="",
        header="et_tdb_s,origin_phase,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s,m_kg,sx_m,sy_m,sz_m,svx_m_s,svy_m_s,svz_m_s,sm_kg",
    )
    samples = options.monte_carlo_samples if samples is None else samples
    monte_carlo_seed = (
        options.monte_carlo_seed if monte_carlo_seed is None else monte_carlo_seed
    )
    ensembles = []
    if samples:
        mc, stats = run_monte_carlo(
            data,
            arrays,
            nlp.gain_scale,
            samples=samples,
            seed=monte_carlo_seed,
            substeps=substeps,
        )
        np.savez_compressed(directory / "psyche_monte_carlo.npz", **mc)
        (directory / "psyche_monte_carlo.json").write_text(
            json.dumps(stats, indent=2) + "\n"
        )
        print(
            f"Psyche Monte Carlo seed {monte_carlo_seed}: {stats['unsafe_altitude_count']}/{samples} below 500 km, "
            f"{stats['invalid_hyperbolic_count']} invalid geometries",
            flush=True,
        )
        ensembles.append(mc)
    if plots:
        plot_diagnostics(data, arrays, physical, directory, ensembles)
    print(
        json.dumps(
            {
                k: physical[k]
                for k in (
                    "converged",
                    "objective",
                    "independent_objective_difference",
                    "mean_matching_position_m",
                    "mean_matching_velocity_m_s",
                    "covariance_matching_scaled",
                    "terminal_std_si",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    return physical
