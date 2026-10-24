"""Physical-unit boundary checks, RK7 defects and independent reintegration."""

from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from .rk import rk7
from .psyche import evaluate_phase


def error_record(delta):
    delta = np.asarray(delta)
    result = {
        "position_m": float(np.linalg.norm(delta[:3])),
        "velocity_m_s": float(np.linalg.norm(delta[3:6])),
    }
    if len(delta) == 7:
        result["mass_kg"] = float(abs(delta[6]))
    return result


def algebraic_checks(mission, solution):
    phases, cfg = (solution["phases"], mission.settings)
    epochs = mission.epochs(solution["exit_days"])
    checks = {
        "tip": error_record(phases[0]["states_si"][0] - mission.x0),
        "fixed_mars_entry": error_record(phases[2]["states_si"][0, :6] - mission.xm),
        "fixed_arrival": error_record(phases[3]["states_si"][-1, :6] - mission.xf),
        "arrival_epoch_error_s": float(epochs[-1] - mission.tf),
        "frame_links": [],
        "exits": [],
    }
    for i, body in ((0, 399), (1, 499), (2, 499)):
        a, b = (phases[i]["states_si"][-1].copy(), phases[i + 1]["states_si"][0].copy())
        planet = mission.cache.state(body, epochs[i + 1])
        a[:6] += planet if i != 1 else -planet
        checks["frame_links"].append(error_record(a - b))
    for phase, body in ((0, 399), (2, 499)):
        state = phases[phase]["states_si"][-1]
        r = np.linalg.norm(state[:3])
        radial_speed = float(np.dot(state[:3], state[3:6]) / r)
        checks["exits"].append(
            {
                "body_id": body,
                "soi_residual_m": float(r - mission.soi[body]),
                "outward_speed_m_s": radial_speed,
                "strictly_outward": bool(radial_speed > 0.0),
            }
        )
    all_controls = np.vstack([p["thrust_n"] for p in phases])
    all_sigma = np.concatenate([p["sigma_n"] for p in phases])
    all_mass = np.concatenate([p["states_si"][:, 6] for p in phases])
    norm = np.linalg.norm(all_controls, axis=1)
    fuel_quadrature = sum(
        (p["duration_s"] * np.dot(np.diff(p["mesh"]), p["sigma_n"]) for p in phases)
    ) / (9.80665 * cfg.isp_s)
    slack_fuel = sum(
        (
            p["duration_s"]
            * np.dot(np.diff(p["mesh"]), p["sigma_n"] - np.linalg.norm(p["thrust_n"], axis=1))
            for p in phases
        )
    ) / (9.80665 * cfg.isp_s)
    checks.update(
        max_thrust_n=float(norm.max()),
        min_sigma_n=float(all_sigma.min()),
        max_sigma_n=float(all_sigma.max()),
        min_endpoint_mass_kg=float(all_mass.min()),
        max_sigma_minus_norm_n=float(np.max(all_sigma - norm)),
        max_norm_minus_sigma_n=float(np.max(norm - all_sigma)),
        fuel_quadrature_kg=float(fuel_quadrature),
        mass_flow_slack_excess_fuel_kg=float(slack_fuel),
        fuel_quadrature_minus_mass_loss_kg=float(fuel_quadrature - solution["fuel_consumed_kg"]),
    )
    checks["role"] = "post-solve diagnostics only; no additional acceptance gate"
    checks["physical_fuel_quadrature_kg"] = float(fuel_quadrature - slack_fuel)
    directions = np.vstack([p["direction"] for p in phases])
    throttles = np.concatenate([p["throttle"] for p in phases])
    checks.update(
        mass_flow_model="throttle_direction",
        epigraph_slack_variable=False,
        max_unit_direction_residual=float(np.max(abs(np.sum(directions**2, axis=1) - 1.0))),
        min_throttle=float(throttles.min()),
        max_throttle=float(throttles.max()),
        max_throttle_norm_roundoff_n=float(np.max(abs(all_sigma - norm))),
    )
    return checks


def relative_interval_error(initial, final, error_si, scale):
    """Original optimiser's relative r/v/m defect, with its ND 1e-13 floor."""
    magnitudes = np.array(
        [
            max(np.linalg.norm(initial[:3]), np.linalg.norm(final[:3]), 1e-13 * scale.length),
            max(np.linalg.norm(initial[3:6]), np.linalg.norm(final[3:6]), 1e-13 * scale.velocity),
            max(abs(initial[6]), abs(final[6]), 1e-13 * scale.mass),
        ]
    )
    return float(np.max(np.asarray(error_si) / magnitudes))


def interval_defects(mission, solution, substeps=None):
    """Reintegrate each arc with the configured uniform RK7 subdivision count."""
    records = []
    cfg = mission.settings
    substeps = cfg.defect_substeps_per_interval if substeps is None else substeps
    if substeps < 1:
        raise ValueError("At least one RK7 substep is required")
    for i, phase in enumerate(solution["phases"]):
        errors, relative = ([], [])
        for k, fraction in enumerate(np.diff(phase["mesh"])):
            h = fraction * phase["duration_s"]
            initial = phase["states_si"][k]
            rhs = lambda x: mission.rhs_si(i, x, phase["thrust_n"][k], phase["sigma_n"][k])
            state = initial.copy()
            for _ in range(substeps):
                state = rk7(rhs, state, h / substeps)
            error = error_record(state - phase["states_si"][k + 1])
            errors.append([error["position_m"], error["velocity_m_s"], error["mass_kg"]])
            if not np.all(np.isfinite(errors[-1])):
                raise FloatingPointError(
                    f"Nonfinite local RK7 defect in phase {i + 1}, interval {k}"
                )
            final = phase["states_si"][k + 1]
            relative.append(relative_interval_error(initial, final, errors[-1], mission.scales[i]))
        errors, relative = (np.array(errors), np.array(relative))
        records.append(
            {
                "phase": i + 1,
                "errors_si": errors,
                "criterion": "max relative position/velocity/mass endpoint defect, matching deterministic_traj_opt.py",
                "defect_tolerance": cfg.defect_tolerance,
                "relative": relative,
                "scaled": relative / cfg.defect_tolerance,
                "rk7_substeps_per_interval": substeps,
                "max_error_position_m": float(errors[:, 0].max()),
                "max_error_velocity_m_s": float(errors[:, 1].max()),
                "max_error_mass_kg": float(errors[:, 2].max()),
                "max_relative": float(relative.max()),
                "max_scaled": float(relative.max() / cfg.defect_tolerance),
            }
        )
    return records


def propagate_interval(mission, phase, state, thrust, sigma, duration):
    result = solve_ivp(
        lambda t, y: mission.rhs_si(phase, y, thrust, sigma),
        (0.0, duration),
        state,
        method="DOP853",
        rtol=2e-12,
        atol=np.r_[np.full(3, 1e-05), np.full(3, 1e-09), 1e-09],
        max_step=min(duration / 8.0, 86400.0 if phase in (1, 3) else 900.0),
        dense_output=True,
    )
    if not result.success:
        raise RuntimeError("Independent propagation failed: " + result.message)
    return result


def reintegrate(mission, solution):
    """One continuous trajectory from TIP; never reset the state at Mars."""
    cfg, phases = (mission.settings, solution["phases"])
    state = mission.x0.copy()
    records, arrays = ([], {})
    for i, phase in enumerate(phases):
        start_state = state.copy()
        times, history, errors = ([], [], [])
        planet_roots = []
        origin = mission.origins[i]
        radius_soi = mission.soi.get(origin)
        for k, fraction in enumerate(np.diff(phase["mesh"])):
            dt = fraction * phase["duration_s"]
            elapsed = phase["mesh"][k] * phase["duration_s"]
            physical_sigma = float(np.linalg.norm(phase["thrust_n"][k]))
            sol = propagate_interval(mission, i, state, phase["thrust_n"][k], physical_sigma, dt)
            sample = np.unique(np.r_[sol.t, np.linspace(0.0, dt, 9)])
            y = sol.sol(sample).T
            if radius_soi is not None:
                g = np.linalg.norm(y[:, :3], axis=1) - radius_soi
                for j in np.flatnonzero((g[:-1] < 0) & (g[1:] >= 0)):
                    root = brentq(
                        lambda t: np.linalg.norm(sol.sol(t)[:3]) - radius_soi,
                        sample[j],
                        sample[j + 1],
                        xtol=1e-07,
                    )
                    planet_roots.append(elapsed + root)
            for t, value in zip(sample, y):
                tau = (elapsed + t) / phase["duration_s"]
                expected = evaluate_phase(solution, i, min(max(tau, 0.0), 1.0), mission)
                errors.append(error_record(value - expected))
            times.extend((elapsed + sample).tolist())
            history.extend(y.tolist())
            state = y[-1].copy()
        times, history = (np.array(times), np.array(history))
        epoch = phase["start_et"] + times
        record = {
            "phase": i + 1,
            "start_mismatch": error_record(start_state - phase["states_si"][0]),
            "end_mismatch": error_record(state - phase["states_si"][-1]),
            "max_position_difference_m": max((e["position_m"] for e in errors)),
            "max_velocity_difference_m_s": max((e["velocity_m_s"] for e in errors)),
            "min_mass_kg": float(history[:, 6].min()),
            "sample_count": len(times),
        }
        radius = np.linalg.norm(history[:, :3], axis=1)
        minimum_radius = {399: cfg.earth_radius_m, 499: cfg.mars_radius_m, 10: cfg.sun_radius_m}[
            origin
        ]
        record["minimum_surface_clearance_m"] = float(radius.min() - minimum_radius)
        if radius_soi is not None:
            record["maximum_distance_beyond_soi_m"] = float(max(0.0, np.max(radius - radius_soi)))
            if not planet_roots:
                extension = propagate_interval(
                    mission,
                    i,
                    state,
                    phase["thrust_n"][-1],
                    float(np.linalg.norm(phase["thrust_n"][-1])),
                    600.0,
                )
                g0, g1 = (
                    np.linalg.norm(state[:3]) - radius_soi,
                    np.linalg.norm(extension.y[:3, -1]) - radius_soi,
                )
                if g0 < 0 <= g1:
                    root = brentq(
                        lambda t: np.linalg.norm(extension.sol(t)[:3]) - radius_soi,
                        0.0,
                        600.0,
                        xtol=1e-07,
                    )
                    planet_roots.append(phase["duration_s"] + root)
            record["first_outward_exit_minus_phase_end_s"] = (
                None if not planet_roots else float(planet_roots[0] - phase["duration_s"])
            )
        else:
            record["maximum_depth_inside_soi_m"] = {
                str(b): float(
                    max(
                        0.0,
                        mission.soi[b]
                        - np.min(
                            np.linalg.norm(
                                history[:, :3] - mission.cache.state(b, epoch)[:, :3], axis=1
                            )
                        ),
                    )
                )
                for b in (399, 499)
            }
        if i == 1:
            target_sun = mission.xm + mission.cache.state(499, mission.tm)
            record["fixed_mars_entry_error"] = error_record(state[:6] - target_sun)
        if i == 3:
            record["fixed_arrival_error"] = error_record(state[:6] - mission.xf)
        records.append(record)
        arrays.update(
            {
                f"p{i}_epochs_et": epoch,
                f"p{i}_states_si": history,
                f"p{i}_position_error_m": np.array([e["position_m"] for e in errors]),
                f"p{i}_velocity_error_m_s": np.array([e["velocity_m_s"] for e in errors]),
            }
        )
        if i < 3:
            body = 399 if i == 0 else 499
            planet = mission.cache.state(body, phase["start_et"] + phase["duration_s"])
            state[:6] += planet if i in (0, 2) else -planet
    passed = all(
        (
            r["max_position_difference_m"] < cfg.global_position_tolerance_m
            and r["max_velocity_difference_m_s"] < cfg.global_velocity_tolerance_m_s
            and (r["minimum_surface_clearance_m"] >= -1.0)
            and (r["min_mass_kg"] >= cfg.mass_floor_kg - 1e-05)
            for r in records
        )
    )
    for r in records:
        if "first_outward_exit_minus_phase_end_s" in r:
            dt = r["first_outward_exit_minus_phase_end_s"]
            passed &= dt is not None and abs(dt) < 0.01
    return (
        {
            "reference_tolerances_met": bool(passed),
            "phases": records,
            "role": "post-solve diagnostic only; never changes refinement or convergence status",
            "soi_membership": "geometry report only; being inside/outside an SOI is not a path-constraint violation",
            "method": "DOP853; ZOH controls; physical mass flow uses actual thrust norm, not sigma; continuous TIP-to-arrival propagation with origin translations and no state reset at Mars",
            "path_check_qualification": "Numerical sampling at adaptive integration nodes and eight subintervals per control interval, plus refined planet exit roots; not an analytic all-times certificate.",
        },
        arrays,
    )


def refine_meshes(solution, defects, mission):
    capacity = mission.settings.max_intervals - sum(
        (len(p["mesh"]) - 1 for p in solution["phases"])
    )
    candidates = []
    for i, record in enumerate(defects):
        for k, score in enumerate(record["scaled"]):
            if score > 1.0:
                candidates.append((float(score), i, k))
    selected = {(i, k) for _, i, k in sorted(candidates, reverse=True)[: max(0, capacity)]}
    meshes = []
    for i, p in enumerate(solution["phases"]):
        mesh = p["mesh"]
        meshes.append(
            np.sort(
                np.r_[
                    mesh,
                    [
                        0.5 * (mesh[k] + mesh[k + 1])
                        for k in range(len(mesh) - 1)
                        if (i, k) in selected
                    ],
                ]
            )
        )
    return (meshes, len(selected))
