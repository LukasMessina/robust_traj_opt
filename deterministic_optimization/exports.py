"""Compact, original-format public results, independent of solver work files."""

from pathlib import Path
import json
import shutil

import numpy as np
from scipy.integrate import solve_ivp

from . import models
from .io import write_json

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = dict(
    earth_mars="earth_mars_deterministic_traj_opt", psyche="psyche_mission_deterministic_traj_opt"
)
CSV_HEADER = "t_days,x,y,z,vx,vy,vz,m_nd,ux_N,uy_N,uz_N,sigma_N,thrust_norm_N"


def output_prefix(case, objective, output_root=None):
    if case not in (*models.DETERMINISTIC_CASE_REGISTRY, "psyche"):
        raise ValueError("Unknown transfer")
    if objective not in ("fuel", "energy"):
        raise ValueError("Unknown objective")
    root = Path(output_root) if output_root is not None else ROOT / "output"
    return root / FOLDERS.get(case, "cr3bp_deterministic_traj_opt") / f"{objective}_optimal" / case


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def history_compatible(history):
    return [
        dict(stage, fuel_consumed_kg=stage.get("fuel_kg", stage.get("fuel_consumed_kg")))
        for stage in history
    ]


def common_arrays(
    mesh,
    times,
    states,
    controls,
    sigma,
    dense_times,
    dense_states,
    dense_controls,
    dense_sigma,
    integrated,
    scale,
    force,
    defects,
    history,
):
    delta = integrated - dense_states
    return dict(
        transcription=np.array("multiple_shooting"),
        control_parameterization=np.array("zero_order_hold"),
        mesh_fraction=mesh,
        t_days=times,
        x=states,
        u=controls,
        sigma=sigma.reshape(1, -1),
        u_interval_start_days=times[:-1],
        u_interval_end_days=times[1:],
        u_norm_n=np.linalg.norm(controls, axis=0) * force,
        # No Radau stages exist. Empty compatibility fields must not imply a degree.
        interval_degrees=np.zeros(len(mesh) - 1, dtype=int),
        stage_interval=np.array([], dtype=int),
        stage_degree_index=np.array([], dtype=int),
        stage_tau=np.array([]),
        stage_state=np.empty((0, 7)),
        t_dense_days=dense_times,
        x_dense=dense_states,
        u_dense=dense_controls,
        sigma_dense=dense_sigma.reshape(1, -1),
        u_dense_norm_n=np.linalg.norm(dense_controls, axis=0) * force,
        x_integrated=integrated,
        position_error_nd=np.linalg.norm(delta[:3], axis=0),
        position_error_m=np.linalg.norm(delta[:3], axis=0) * scale[0],
        velocity_error_nd=np.linalg.norm(delta[3:6], axis=0),
        velocity_error_m_per_s=np.linalg.norm(delta[3:6], axis=0) * scale[3],
        mass_consumption_difference_kg=-delta[6] * scale[6],
        mass_consumption_abs_difference_kg=abs(delta[6]) * scale[6],
        interval_defect_scaled=defects[:, 0],
        interval_defect_position_m=defects[:, 1],
        interval_defect_velocity_m_per_s=defects[:, 2],
        interval_defect_mass_kg=defects[:, 3],
        history=np.array(history_compatible(history), dtype=object),
    )


def single_phase_data(directory, config, result):
    from .single_phase import ShootingProblem, is_cr3bp, moon_clearance_km

    case = models.DETERMINISTIC_CASE_REGISTRY[config["case"]]()
    with np.load(directory / "solution.npz") as source:
        raw = dict(source)
    problem = ShootingProblem(
        case,
        config["objective"],
        config["ipopt_options"],
        mesh=raw["mesh"],
        incoming=(dict(result, converged=True, method="shooting"), raw),
    )
    mesh, x = raw["mesh"], raw["states_nd"]
    scale = problem.mission.scales[0].state
    controls = raw["thrust_n"] / case.thrust_unit
    sigma = raw["mass_flow_equivalent_n"] / case.thrust_unit
    times, states, thrust, magnitudes, propagated = [], [], [], [], []
    current = case.x0_augmented_state.copy()
    for k, fraction in enumerate(np.diff(mesh)):
        dt = case.tof_nd * fraction
        sample = np.linspace(0, dt, 9)
        answer = solve_ivp(
            lambda t, y: models.eom(case, y, controls[:, k], np.linalg.norm(controls[:, k])),
            (0, dt),
            current,
            method="DOP853",
            rtol=3e-14,
            atol=np.r_[np.full(3, 1e-8), np.full(4, 1e-12)] / scale,
            max_step=dt / 32,
            dense_output=True,
        )
        if not answer.success:
            raise RuntimeError(answer.message)
        propagated.extend(answer.sol(sample).T)
        current = answer.y[:, -1]
        for j, elapsed in enumerate(sample):
            state = (
                x[:, k]
                if j == 0
                else (
                    x[:, k + 1]
                    if j == 8
                    else np.asarray(
                        problem.flow(
                            x[:, k],
                            raw["throttle"][k] * raw["direction"][:, k],
                            raw["throttle"][k],
                            elapsed,
                        )[0]
                    ).ravel()
                )
            )
            states.append(state)
        times.extend(case.tof_days * (mesh[k] + fraction * np.linspace(0, 1, 9)))
        thrust.extend([controls[:, k]] * 9)
        magnitudes.extend([sigma[k]] * 9)
    with np.load(directory / "integration_diagnostics.npz") as diagnostic:
        d = diagnostic["local_defects_nd"] * scale[:, None]
        defects = np.column_stack(
            [
                diagnostic["local_relative_defects"],
                np.linalg.norm(d[:3], axis=0),
                np.linalg.norm(d[3:6], axis=0),
                abs(d[6]),
            ]
        )
    data = common_arrays(
        mesh,
        mesh * case.tof_days,
        x,
        controls,
        sigma,
        np.array(times),
        np.array(states).T,
        np.array(thrust).T,
        np.array(magnitudes),
        np.array(propagated).T,
        scale,
        case.thrust_unit,
        defects,
        result["history"],
    )
    data.update(
        throttle=raw["throttle"],
        direction=raw["direction"],
        length_unit_km=case.length_unit,
        time_unit_s=case.time_unit,
        mass_unit_kg=case.m0_wet,
        thrust_unit_n=case.thrust_unit,
    )
    if is_cr3bp(case):
        data["moon_distance_surface_km"] = moon_clearance_km(case, data["x_dense"])
    else:
        data.update(
            {
                key: np.asarray(case.ephemeris[key])
                for key in (
                    "earth_track_si",
                    "mars_track_si",
                    "earth_track_times_days",
                    "mars_track_times_days",
                )
            }
        )
    meta = dict(
        length_unit_km=case.length_unit,
        time_unit_s=case.time_unit,
        velocity_unit_km_s=case.velocity_unit,
        mass_unit_kg=case.m0_wet,
        force_unit_n=case.thrust_unit,
        exhaust_velocity_nd=case.exhaust_velocity_nd,
        max_thrust_n=case.max_thrust_n,
        isp_s=case.isp,
        tof_days=case.tof_days,
        x0_nd=case.x0_state,
        xf_nd=case.xf_state,
        defect_tolerance=config["defect_tolerance"],
        frame="Sun-centred J2000" if not is_cr3bp(case) else "Earth-Moon barycentric synodic",
        verification="Continuous DOP853, physical thrust norm mass flow, rtol=3e-14; no node resets; report only",
    )
    return data, meta


def psyche_data(directory, config, result):
    from .psyche import Mission, Settings, load_solution, evaluate_phase
    from .psyche_validation import propagate_interval

    mission = Mission(Settings(**config["settings"]))
    solution_path = directory / Path(result["last_candidate"]).name
    solution = load_solution(solution_path)
    scale = mission.scales[1].state
    force = scale[6] * scale[3] ** 2 / scale[0]
    total = mission.tf - mission.t0
    times, states, thrust, magnitudes, integrated, phase_index = [], [], [], [], [], []
    native_states, native_integrated = [], []
    nodes, node_times, controls, sigma = [], [], [], []
    current = mission.x0.copy()

    def heliocentric(state, i, epoch):
        value = state.copy()
        if i in (0, 2):
            value[:6] += mission.cache.state(399 if i == 0 else 499, epoch)
        return value / scale

    for i, phase in enumerate(solution["phases"]):
        epoch_nodes = phase["start_et"] + phase["duration_s"] * phase["mesh"]
        # Keep one global node at each phase boundary, plus exact native arrays below.
        for k, (epoch, state) in enumerate(zip(epoch_nodes, phase["states_si"])):
            if i and k == 0:
                continue
            nodes.append(heliocentric(state, i, epoch))
            node_times.append((epoch - mission.t0) / 86400)
        controls.extend(phase["thrust_n"] / force)
        sigma.extend(phase["sigma_n"] / force)
        for k, fraction in enumerate(np.diff(phase["mesh"])):
            dt = phase["duration_s"] * fraction
            sample = np.linspace(0, dt, 9)
            answer = propagate_interval(
                mission, i, current, phase["thrust_n"][k], np.linalg.norm(phase["thrust_n"][k]), dt
            )
            values = answer.sol(sample).T
            current = values[-1].copy()
            for elapsed, value in zip(sample, values):
                tau = phase["mesh"][k] + elapsed / phase["duration_s"]
                epoch = phase["start_et"] + phase["duration_s"] * tau
                expected = evaluate_phase(solution, i, tau, mission)
                native_states.append(expected)
                native_integrated.append(value)
                states.append(heliocentric(expected, i, epoch))
                integrated.append(heliocentric(value, i, epoch))
                times.append((epoch - mission.t0) / 86400)
                thrust.append(phase["thrust_n"][k] / force)
                magnitudes.append(phase["sigma_n"][k] / force)
                phase_index.append(i)
        if i < 3:
            origin = mission.cache.state(399 if i == 0 else 499, epoch_nodes[-1])
            current[:6] += origin if i in (0, 2) else -origin
    defects = read(solution_path.with_name(solution_path.stem + "_defects.json"))[
        "interval_defects"
    ]
    defects = np.vstack([np.column_stack([d["relative"], d["errors_si"]]) for d in defects])
    node_times = np.array(node_times)
    data = common_arrays(
        node_times * 86400 / total,
        node_times,
        np.array(nodes).T,
        np.array(controls).T,
        np.array(sigma),
        np.array(times),
        np.array(states).T,
        np.array(thrust).T,
        np.array(magnitudes),
        np.array(integrated).T,
        scale,
        force,
        defects,
        result["history"],
    )
    with np.load(solution_path) as native:
        data.update(dict(native))  # Exact four-phase solution, loadable by load_solution.
    # Subtract in each phase's native SI frame before a large heliocentric
    # translation can round away sub-micrometre planet-relative discrepancies.
    native_states, native_integrated = np.array(native_states).T, np.array(native_integrated).T
    delta = native_integrated - native_states
    data.update(
        native_states_dense_si=native_states,
        native_integrated_dense_si=native_integrated,
        position_error_m=np.linalg.norm(delta[:3], axis=0),
        position_error_nd=np.linalg.norm(delta[:3], axis=0) / scale[0],
        velocity_error_m_per_s=np.linalg.norm(delta[3:6], axis=0),
        velocity_error_nd=np.linalg.norm(delta[3:6], axis=0) / scale[3],
        mass_consumption_difference_kg=-delta[6],
        mass_consumption_abs_difference_kg=abs(delta[6]),
    )
    data.update(
        length_unit_km=scale[0] / 1000,
        time_unit_s=mission.scales[1].time,
        mass_unit_kg=scale[6],
        thrust_unit_n=force,
        phase_index_dense=np.array(phase_index),
        epoch_et_dense=mission.t0 + np.array(times) * 86400,
    )
    for body, name in ((399, "earth"), (499, "mars")):
        epochs = np.linspace(mission.t0, mission.tf, 1200)
        data[name + "_track_si"] = mission.cache.state(body, epochs).T
        data[name + "_track_times_days"] = (epochs - mission.t0) / 86400
    meta = dict(
        length_unit_km=scale[0] / 1000,
        time_unit_s=mission.scales[1].time,
        velocity_unit_km_s=scale[3] / 1000,
        mass_unit_kg=scale[6],
        force_unit_n=force,
        max_thrust_n=mission.settings.thrust_max_n,
        isp_s=mission.settings.isp_s,
        tof_days=total / 86400,
        x0_nd=np.array(nodes)[0, :6],
        xf_nd=np.array(nodes)[-1, :6],
        frame="Sun-centred J2000 for common x/u/CSV arrays; p0 and p2 native arrays are planet-relative SI",
        native_dense_frames=[
            "Earth-centred J2000",
            "Sun-centred J2000",
            "Mars-centred J2000",
            "Sun-centred J2000",
        ],
        error_evaluation="Subtract native SI phase states before origin translation; native dense arrays indexed by phase_index_dense",
        defect_tolerance=mission.settings.defect_tolerance,
        verification="Continuous standard DOP853 rtol=2e-12, physical thrust norm mass flow; origin translations, no resets; report only",
        source_solution=solution_path.name,
    )
    return data, meta


def render(prefix, data, meta):
    from .plotting import Plotter

    if meta["case"] == "psyche":
        from .psyche_figures import plot_psyche_export

        return plot_psyche_export(prefix, data, meta)
    plotter = Plotter(prefix)
    case = models.DETERMINISTIC_CASE_REGISTRY[meta["case"]]()
    t, x, u = data["t_dense_days"], data["x_dense"], data["u_dense"]
    if isinstance(case, models.EarthMars):
        plotter.plot_heliocentric_transfer(case, t, x, u)
        plotter.plot_heliocentric_3d(case, t, x, u)
        plotter.plot_inertial_control(t, u * case.thrust_unit, case.max_thrust_n)
    else:
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
        plotter.plot_thrust_time_evolution(t, data["u_dense_norm_n"], case.max_thrust_n)
        plotter.plot_moon_distance_time_evolution(t, data["moon_distance_surface_km"])
    plotter.plot_collocation_approx_errors(
        t,
        data["position_error_m"],
        data["velocity_error_m_per_s"],
        data["mass_consumption_abs_difference_kg"],
    )
    plotter.plot_interval_defect(
        data["mesh_fraction"],
        meta["tof_days"],
        data["interval_defect_scaled"],
        meta["defect_tolerance"],
    )


def export_case(directory, case, objective, output_root=None):
    """Publish a successful run; an incomplete run never replaces accepted outputs."""
    directory = Path(directory)
    config, result = read(directory / "configuration.json"), read(directory / "result.json")
    if not result["converged"]:
        raise ValueError("Cannot publish an unconverged run as an optimal trajectory")
    if config["objective"] != objective or config.get("case", "psyche") != case:
        raise ValueError("Requested export does not match the saved case/objective")
    data, meta = (psyche_data if case == "psyche" else single_phase_data)(directory, config, result)
    data["objective_mode"] = np.array(objective + " optimal")
    diagnostics = dict(
        result, fuel_consumed_kg=result.get("fuel_kg", result.get("candidate_fuel_consumed_kg"))
    )
    meta.update(
        case=case,
        objective_mode=objective + " optimal",
        energy_definition="0.5 * integral(thrust_N**2 dt_s)",
        diagnostics=diagnostics,
        history=history_compatible(result["history"]),
        configuration=config,
        converged=True,
        defect_tolerance_met=True,
        control_formulation="T=Tmax*alpha*d; dot(d,d)=1; mdot=-Tmax*alpha/(Isp*g0)",
        schema="Original deterministic common array names and CSV columns; multiple-shooting native fields added",
        csv_state_units="nondimensional; thrust and mass-flow-equivalent sigma in N",
        source_directory=str(directory.resolve()),
    )
    meta.update(
        max_reintegration_position_error_m=float(data["position_error_m"].max()),
        max_reintegration_velocity_error_m_s=float(data["velocity_error_m_per_s"].max()),
        terminal_reintegration_position_error_m=float(data["position_error_m"][-1]),
        terminal_reintegration_velocity_error_m_s=float(data["velocity_error_m_per_s"][-1]),
        max_reintegration_mass_error_kg=float(data["mass_consumption_abs_difference_kg"].max()),
    )
    diagnostics.update(
        final_mass_kg=float(data["x"][6, -1] * meta["mass_unit_kg"]),
        energy_objective=float(
            0.5 * np.dot(np.diff(data["t_days"]) * 86400, data["u_norm_n"] ** 2)
        ),
    )
    for name in (
        "integration_diagnostics",
        "global_reference_accuracy",
        "sigma_model_reintegration",
    ):
        if (directory / (name + ".json")).exists():
            meta[name] = read(directory / (name + ".json"))
    prefix = output_prefix(case, objective, output_root)
    # Prepare complete outputs before replacing the selected case's public files.
    import tempfile

    prefix.parent.mkdir(parents=True, exist_ok=True)
    (ROOT / ".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="export_", dir=ROOT / ".cache") as temporary:
        staged = Path(temporary) / case
        np.savez_compressed(staged.with_suffix(".npz"), **data)
        np.savetxt(
            staged.with_suffix(".csv"),
            np.column_stack(
                [
                    data["t_dense_days"],
                    data["x_dense"].T,
                    data["u_dense"].T * meta["force_unit_n"],
                    data["sigma_dense"].T * meta["force_unit_n"],
                    data["u_dense_norm_n"],
                ]
            ),
            delimiter=",",
            header=CSV_HEADER,
            comments="",
        )
        render(staged, data, meta)
        write_json(staged.with_suffix(".json"), meta)
        files = list(Path(temporary).iterdir())
        for path in files:
            shutil.copyfile(path, prefix.parent / path.name)
    return prefix


def regenerate_public(case, objective, output_root=None):
    prefix = output_prefix(case, objective, output_root)
    with np.load(prefix.with_suffix(".npz"), allow_pickle=True) as archive:
        data = dict(archive)
    meta = read(prefix.with_suffix(".json"))
    render(prefix, data, meta)
    write_json(prefix.with_suffix(".json"), meta)
