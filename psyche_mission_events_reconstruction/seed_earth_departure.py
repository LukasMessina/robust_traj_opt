"""Demonstrate and validate the dynamics-only Earth departure initial guess."""

from psyche_mission_events_reconstruction.paths import DATA_ROOT, resolve_input
from pathlib import Path
import hashlib
import json
import numpy as np
import spiceypy as raw
from .ephemeris_cache import EphemerisCache
from .rollout import rollout_to_soi_exit, soi_radius_and_rate, translate_to_sun

ROOT = Path(__file__).resolve().parents[1]
OUT = DATA_ROOT / "soi/earth_rollout"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*, output=OUT, cache_path=None, soi_model=None):
    OUT = Path(output)
    tip_path = DATA_ROOT / "tip/tip_result.json"
    cache_path = (
        Path(cache_path)
        if cache_path is not None
        else DATA_ROOT / "ephemerides/earth_mars_ephemerides.npz"
    )
    tip = json.loads(tip_path.read_text(encoding="utf-8"))
    cache = EphemerisCache.load(cache_path)
    if soi_model is None:
        from .mission_soi import fixed_mission_soi

        soi_model = fixed_mission_soi(399, cache.metadata["mu_m3_s2"])
    if cache.metadata["tip_result_sha256"] != digest(tip_path):
        raise ValueError("TIP boundary source differs from the cached ephemeris provenance")
    if raw.ktotal("ALL"):
        raise RuntimeError("Run this demonstration in a fresh process with no loaded SPICE kernels")
    start = tip["epochs"]["tip"]["et_tdb_s"]
    initial = np.r_[tip["earth_relative_state_si"], 2747.0]
    mu = cache.metadata["mu_m3_s2"]["399"]
    end_limit = start + 10 * 86400
    arc = rollout_to_soi_exit(initial, start, end_limit, 399, cache, soi_model=soi_model)
    refined = rollout_to_soi_exit(
        initial, start, end_limit, 399, cache, rtol=2e-13, max_step_s=1800.0, soi_model=soi_model
    )
    elapsed = np.linspace(0, arc.duration_s, 1001)
    states = arc.at_elapsed(elapsed)
    epochs = start + elapsed
    geometry = np.array([soi_radius_and_rate(cache, 399, t, soi_model) for t in epochs])
    radius = np.linalg.norm(states[:, :3], axis=1)
    c3 = np.sum(states[:, 3:6] ** 2, axis=1) - 2 * mu / radius
    momentum = np.cross(states[:, :3], states[:, 3:6])
    elements = raw.oscelt(initial[:6] / 1000, 0.0, mu / 1000000000.0)
    analytic = np.array([raw.conics(elements, float(t)) * 1000 for t in elapsed])
    position_error = np.linalg.norm(states[:, :3] - analytic[:, :3], axis=1)
    velocity_error = np.linalg.norm(states[:, 3:6] - analytic[:, 3:6], axis=1)
    final = arc.final_state
    exit_radius, radius_rate = soi_radius_and_rate(cache, 399, arc.end_et, soi_model)
    radial_speed = np.dot(final[:3], final[3:6]) / np.linalg.norm(final[:3])
    crossing_rate = radial_speed - radius_rate
    sun_state = translate_to_sun(final, 399, arc.end_et, cache)
    closure = sun_state[:6] - cache.state(399, arc.end_et) - final[:6]
    checks = {
        "max_two_body_position_error_m": float(position_error.max()),
        "max_two_body_velocity_error_m_s": float(velocity_error.max()),
        "exit_epoch_refinement_difference_s": abs(arc.duration_s - refined.duration_s),
        "exit_position_refinement_difference_m": float(
            np.linalg.norm(final[:3] - refined.final_state[:3])
        ),
        "exit_velocity_refinement_difference_m_s": float(
            np.linalg.norm(final[3:6] - refined.final_state[3:6])
        ),
        "max_c3_drift_m2_s2": float(np.max(np.abs(c3 - c3[0]))),
        "max_relative_angular_momentum_drift": float(
            np.max(np.linalg.norm(momentum - momentum[0], axis=1)) / np.linalg.norm(momentum[0])
        ),
        "soi_surface_residual_m": float(radius[-1] - exit_radius),
        "outward_crossing_rate_m_s": float(crossing_rate),
        "max_mass_change_kg": float(np.max(np.abs(states[:, 6] - initial[6]))),
        "origin_translation_position_closure_m": float(np.linalg.norm(closure[:3])),
        "origin_translation_velocity_closure_m_s": float(np.linalg.norm(closure[3:])),
        "spacecraft_ephemeris_queries": 0,
        "spice_kernels_loaded_during_rollout": raw.ktotal("ALL"),
    }
    if (
        position_error.max() > 0.01
        or velocity_error.max() > 1e-06
        or checks["exit_epoch_refinement_difference_s"] > 1e-05
        or (checks["exit_position_refinement_difference_m"] > 0.01)
        or (abs(checks["soi_surface_residual_m"]) > 0.001)
        or (crossing_rate <= 0)
        or (checks["max_mass_change_kg"] != 0)
        or (np.max(radius[:-1] - geometry[:-1, 0]) >= 0)
    ):
        raise AssertionError("Earth coast rollout validation failed")
    raw.furnsh(str(ROOT / "data/jpl_kernels/naif0012.tls"))
    try:
        utc = raw.et2utc(arc.end_et, "ISOC", 6) + "Z"
    finally:
        raw.kclear()
    result = {
        "status": "Earth phase initial guess generated; not a reconstructed or optimised trajectory",
        "start_et_tdb_s": start,
        "start_utc": tip["epochs"]["tip"]["utc"],
        "exit_et_tdb_s": arc.end_et,
        "exit_utc": utc,
        "duration_s": arc.duration_s,
        "duration_days": arc.duration_s / 86400,
        "earth_relative_exit_state_si": final[:6].tolist(),
        "sun_relative_exit_state_si": sun_state[:6].tolist(),
        "exit_soi_radius_m": float(exit_radius),
        "seed_mass_kg": float(final[6]),
        "seed_thrust_n": [0.0, 0.0, 0.0],
        "seed_sigma_n": 0.0,
        "frame": "J2000",
        "units": "m, m/s, kg",
        "time": "ET/TDB seconds since J2000",
        "gravity": "Earth point mass only",
        "initial_mass_assumption": "thesis 2747 kg applied at TIP",
        "soi": (
            "fixed-semimajor-axis Laplace model"
            if soi_model is not None and soi_model.semimajor_axis_m is not None
            else "instantaneous-distance Laplace model"
        ),
        "semimajor_axis_m": None if soi_model is None else soi_model.semimajor_axis_m,
        "integrator": {
            "method": "DOP853 IVP initial guess only",
            "rtol": 2e-12,
            "atol_si": [1e-05] * 3 + [1e-09] * 3 + [1e-10],
            "maximum_step_s": 3600.0,
        },
        "checks": checks,
        "arrival_constraint": "Hard equality to asteroid 2000016 position/velocity at 2029-08-01 12:00:00 TDB; see the active arrival boundary.",
        "remaining": "This module generates only the Earth coast timing history. The optimisation entry point regenerates a small-thrust RK7 state rollout; see README.md.",
        "provenance": {
            "TIP_boundary_sha256": digest(tip_path),
            "planetary_cache_sha256": digest(cache_path),
            "implementation_sha256": {
                p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("rollout.py"))
            },
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "earth_rollout.npz",
        elapsed_s=elapsed,
        epochs_et_tdb_s=epochs,
        earth_relative_states_si=states,
        soi_radii_m=geometry[:, 0],
        c3_drift_m2_s2=c3 - c3[0],
        kepler_position_error_m=position_error,
        kepler_velocity_error_m_s=velocity_error,
    )
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savetxt(
        OUT / "exit_state.csv",
        np.r_[arc.end_et, final, sun_state[:6]][None],
        delimiter=",",
        header="et_tdb_s,x_earth_m,y_earth_m,z_earth_m,vx_earth_m_s,vy_earth_m_s,vz_earth_m_s,mass_kg,x_sun_m,y_sun_m,z_sun_m,vx_sun_m_s,vy_sun_m_s,vz_sun_m_s",
        comments="",
    )
    lines = [
        "PSYCHE: DYNAMICS-ONLY EARTH DEPARTURE INITIAL GUESS",
        "",
        "This is a zero-thrust Earth-point-mass rollout from the fixed TIP state.",
        "It is neither the reconstructed mission trajectory nor an optimisation result.",
        "No spacecraft SPK was loaded or sampled. SOI convention: " + result["soi"],
        f"TIP: {result['start_utc']}",
        f"Computed seed SOI exit: {utc}",
        f"Elapsed time: {arc.duration_s:.9f} s ({arc.duration_s / 86400:.9f} days)",
        f"Exit radius: {exit_radius / 1000:.9f} km",
        f"Seed mass: {final[6]:g} kg; thrust: 0 N.",
        "Exit time and state are computed initial guesses; neither is fixed in the OCP.",
        "The zero-thrust seed does not impose a coast restriction on the optimised trajectory.",
        "Extra digits represent numerical precision, not physical event-timing accuracy.",
        "",
        "Earth-relative exit position [km]: " + ", ".join((f"{v / 1000:.9f}" for v in final[:3])),
        "Earth-relative exit velocity [km/s]: "
        + ", ".join((f"{v / 1000:.12f}" for v in final[3:6])),
        "",
        "Validation:",
    ]
    lines += [f"  {key}: {value}" for key, value in checks.items()]
    lines += [
        "",
        "Remaining initialisation:",
        "Continue the Earth seed under Sun gravity to the fixed Mars-entry epoch.",
        "Start the second chain at the fixed Mars-entry Cartesian state with carried seed mass.",
        "Roll out under Mars gravity to its detected exit, then Sun gravity to fixed arrival.",
        "Retain mismatches at Mars entry and final arrival as explicit initial-guess residuals.",
        "The optimiser must eliminate these residuals; do not snap or reshape seed endpoints.",
        "Arrival position and velocity are HARD equalities to asteroid 2000016 at",
        "2029-08-01 12:00:00 TDB. See the active arrival boundary artifact.",
        "No new numerical Mars entry/arrival target, spacecraft SPK comparison or OCP result",
        "is claimed in this stage. The module supports all four seed phases once targets exist.",
        "",
        "Reproduce (activated det_traj_opt_env): python -m psyche_mission_events_reconstruction."
        + (
            "soi_reconstruction"
            if result["semimajor_axis_m"] is not None
            else "seed_earth_departure"
        ),
        "See README.md for reconstruction and optimisation plotting commands.",
        "IVP method and event-root documentation:",
        "https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html",
    ]
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"exit_utc": utc, "duration_days": result["duration_days"], "checks": checks}, indent=2
        )
    )
    return result


if __name__ == "__main__":
    run()
