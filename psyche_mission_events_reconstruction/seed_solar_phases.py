"""Complete the two forward coast chains and expose infeasible endpoints.

The current asteroid/TDB target is used only for the terminal residual.
SPICE state calls are disabled while generating the initial guess. Kepler
conics validate the numerical integration without reading any ephemerides.
"""

from psyche_mission_events_reconstruction.paths import DATA_ROOT, resolve_input
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import spiceypy as raw
from tudatpy.interface import spice
from .arrival_boundary import ACTIVE_ARRIVAL_BOUNDARY, load_arrival_boundary
from .build_ephemeris_cache import policy
from .ephemeris_cache import EphemerisCache
from .kernels import ROOT, TIP_KERNELS, epoch_record, loaded_kernels, sha256
from .mission_soi import ACTIVE_MARS_BOUNDARY, ACTIVE_STAGE, fixed_mission_soi
from .rollout import make_four_phase_coast_seed, integrate_coast

OUT = DATA_ROOT / "rollout_timing"


def norm_residual(vector):
    vector = np.asarray(vector)
    return {
        "components_si": vector.tolist(),
        "position_norm_m": float(np.linalg.norm(vector[:3])),
        "velocity_norm_m_s": float(np.linalg.norm(vector[3:])),
    }


def run():
    arrival = load_arrival_boundary()
    mars = json.loads(ACTIVE_MARS_BOUNDARY.read_text(encoding="utf-8"))
    tip_path = DATA_ROOT / "tip/tip_result.json"
    tip = json.loads(tip_path.read_text(encoding="utf-8"))
    definition = policy()
    cache_path = resolve_input(mars["planetary_cache"]["path"])
    if sha256(cache_path) != mars["planetary_cache"]["sha256"]:
        raise ValueError("Planetary cache changed")
    cache = EphemerisCache.load(cache_path)
    if cache.metadata["tip_result_sha256"] != sha256(tip_path):
        raise ValueError("TIP/cache provenance mismatch")
    t0, tm, tf = (
        tip["epochs"]["tip"]["et_tdb_s"],
        mars["epoch"]["et_tdb_s"],
        arrival["epoch"]["et_tdb_s"],
    )
    initial = np.r_[tip["earth_relative_state_si"], 2747.0]
    target = np.asarray(arrival["sun_relative_state_si"])
    if raw.ktotal("ALL") or spice.get_total_count_of_kernels_loaded():
        raise RuntimeError("Generate the seed in an empty SPICE session")
    forbidden = AssertionError("SPICE state queries are forbidden during seed construction")
    with patch.object(raw, "spkezr", side_effect=forbidden), patch.object(
        raw, "spkez", side_effect=forbidden
    ), patch.object(spice, "get_body_cartesian_state_at_epoch", side_effect=forbidden):
        seed = make_four_phase_coast_seed(
            initial, t0, mars["mars_relative_state_si"], tm, tf, target, cache
        )
    arcs = seed["arcs"]
    earth_old = json.loads((ACTIVE_STAGE / "earth_rollout/result.json").read_text(encoding="utf-8"))
    mars_old = json.loads((ACTIVE_STAGE / "mars_rollout/result.json").read_text(encoding="utf-8"))
    exit_changes = {
        "earth_exit_epoch_change_s": arcs[0].end_et - earth_old["exit_et_tdb_s"],
        "mars_exit_epoch_change_s": arcs[2].end_et - mars_old["exit_epoch"]["et_tdb_s"],
    }
    if max(map(abs, exit_changes.values())) > 1e-05:
        raise AssertionError("Planet phase no longer reproduces the validated free-exit seed")
    summaries, arrays = ([], {})
    for i, arc in enumerate(arcs, 1):
        elapsed = np.unique(np.r_[np.linspace(0.0, arc.duration_s, 1001), arc.solution.t])
        states = arc.at_elapsed(elapsed)
        epochs = arc.start_et + elapsed
        mu = cache.metadata["mu_m3_s2"][str(arc.origin_id)]
        elements = raw.oscelt(states[0, :6] / 1000.0, 0.0, mu / 1000000000.0)
        conic = np.array([raw.conics(elements, float(t)) * 1000.0 for t in elapsed])
        position_errors = np.linalg.norm(states[:, :3] - conic[:, :3], axis=1)
        velocity_errors = np.linalg.norm(states[:, 3:6] - conic[:, 3:], axis=1)
        radius = np.linalg.norm(states[:, :3], axis=1)
        energy = np.sum(states[:, 3:6] ** 2, axis=1) / 2.0 - mu / radius
        momentum = np.cross(states[:, :3], states[:, 3:6])
        record = {
            "phase": i,
            "origin_id": arc.origin_id,
            "start_et_tdb_s": arc.start_et,
            "end_et_tdb_s": arc.end_et,
            "duration_days": arc.duration_s / 86400.0,
            "initial_state_si": states[0].tolist(),
            "final_state_si": states[-1].tolist(),
            "integration_function_evaluations": arc.solution.nfev,
            "saved_sample_count": len(elapsed),
            "max_conics_position_error_m": float(position_errors.max()),
            "max_conics_velocity_error_m_s": float(velocity_errors.max()),
            "max_specific_energy_drift_m2_s2": float(np.max(abs(energy - energy[0]))),
            "max_relative_angular_momentum_drift": float(
                np.max(np.linalg.norm(momentum - momentum[0], axis=1)) / np.linalg.norm(momentum[0])
            ),
            "max_mass_change_kg": float(np.max(abs(states[:, 6] - initial[6]))),
        }
        if (
            position_errors.max() > 0.5
            or velocity_errors.max() > 1e-06
            or record["max_mass_change_kg"] != 0
        ):
            raise AssertionError(f"Numerical coast validation failed: {record}")
        if arc.origin_id == 10:
            refined = integrate_coast(
                states[0], arc.start_et, arc.end_et, mu, 10, rtol=2e-13, max_step_s=43200.0
            )
            record["endpoint_refinement_difference"] = norm_residual(
                refined.final_state[:6] - states[-1, :6]
            )
            if record["endpoint_refinement_difference"]["position_norm_m"] > 0.5:
                raise AssertionError("Solar integration refinement failed")
            probe = np.unique(np.r_[elapsed, np.arange(0.0, arc.duration_s, 10800.0)])
            sc = arc.at_elapsed(probe)[:, :3]
            record["soi_path_diagnostics"] = {}
            for body in (399, 499):
                model = fixed_mission_soi(body, cache.metadata["mu_m3_s2"])
                sphere = model.radius_rate(None)[0]
                distance = np.linalg.norm(
                    sc - cache.state(body, arc.start_et + probe)[:, :3], axis=1
                )
                index = int(np.argmin(distance))
                record["soi_path_diagnostics"][str(body)] = {
                    "minimum_sampled_centre_distance_m": float(distance[index]),
                    "minimum_sampled_clearance_m": float(distance[index] - sphere),
                    "minimum_sample_epoch_et_tdb_s": float(arc.start_et + probe[index]),
                    "samples_inside_by_more_than_1m": int(
                        np.count_nonzero(distance < sphere - 1.0)
                    ),
                    "sample_count": len(probe),
                    "maximum_probe_spacing_s": 10800.0,
                    "qualification": "Sampled diagnostic; not a continuous path feasibility certificate.",
                }
        else:
            sphere = fixed_mission_soi(arc.origin_id, cache.metadata["mu_m3_s2"]).radius_rate(None)[
                0
            ]
            record["exit_soi_residual_m"] = float(radius[-1] - sphere)
            record["exit_radial_speed_m_s"] = float(
                np.dot(states[-1, :3], states[-1, 3:6]) / radius[-1]
            )
        for name, value in (
            ("elapsed_s", elapsed),
            ("epochs_et_tdb_s", epochs),
            ("states_si", states),
            ("tau", elapsed / arc.duration_s),
            ("thrust_n", np.zeros((len(elapsed), 3))),
            ("conics_position_error_m", position_errors),
            ("conics_velocity_error_m_s", velocity_errors),
        ):
            arrays[f"phase{i}_{name}"] = value
        summaries.append(record)
    if raw.ktotal("ALL") or spice.get_total_count_of_kernels_loaded():
        raise AssertionError("Unexpected kernels during rollout validation")
    with loaded_kernels([ROOT / "data/jpl_kernels" / TIP_KERNELS[0].name]):
        for record in summaries:
            record["start_epoch"] = epoch_record(record["start_et_tdb_s"])
            record["end_epoch"] = epoch_record(record["end_et_tdb_s"])
    arrays["mars_entry_target_sun_state_si"] = np.asarray(
        mars["mars_relative_state_si"]
    ) + cache.state(499, tm)
    arrays["arrival_target_sun_state_si"] = target
    link_errors = {}
    for i, body in ((0, 399), (2, 499)):
        expected = arcs[i].final_state.copy()
        expected[:6] += cache.state(body, arcs[i].end_et)
        link_errors[str(body)] = norm_residual(arcs[i + 1].at_elapsed(0.0)[:6] - expected[:6])
    result = {
        "status": "Four-phase dynamics-only initial guess completed; infeasible coast seed, not an optimised trajectory",
        "frame": "J2000",
        "units": "m, m/s, kg, N",
        "time_scale": "ET/TDB seconds since J2000",
        "initial_guess_policy": definition["initial_guess"],
        "phases": summaries,
        "fixed_arrival_epoch": arrival["epoch"],
        "arrival_target_body_id": 2000016,
        "mars_entry_mismatch": norm_residual(seed["mars_entry_cartesian_residual_si"]),
        "arrival_mismatch": norm_residual(seed["arrival_cartesian_residual_si"]),
        "residual_sign": "propagated state minus fixed target, Sun/J2000",
        "in_chain_link_residuals": link_errors,
        "free_exit_reproduction": exit_changes,
        "seed_thrust_n": [0.0, 0.0, 0.0],
        "seed_mass_kg": 2747.0,
        "seed_fuel_used_kg": 0.0,
        "numerical_validation_passed": True,
        "spice_state_calls_disabled_during_seed": True,
        "kernels_loaded_during_rollout": 0,
        "spacecraft_history_used_as_seed": False,
        "provenance": {
            "tip_path": str(tip_path),
            "tip_sha256": sha256(tip_path),
            "mars_entry_path": str(ACTIVE_MARS_BOUNDARY),
            "mars_entry_sha256": sha256(ACTIVE_MARS_BOUNDARY),
            "arrival_path": str(ACTIVE_ARRIVAL_BOUNDARY),
            "arrival_sha256": sha256(ACTIVE_ARRIVAL_BOUNDARY),
            "planetary_cache": mars["planetary_cache"],
        },
        "limitations": [
            "The two chains are individually dynamically propagated but do not satisfy the fixed downstream state constraints.",
            "Mars entry starts the second chain at its fixed state; the first-chain mismatch is preserved as an infeasible initial phase link, not an impulsive manoeuvre.",
            "Arrival state and epoch are hard asteroid/TDB constraints in the intended NLP; the seed endpoint has not been replaced with the target.",
            "Zero seed thrust and fuel use are initialisation assumptions, not optimal performance results.",
            "Sampled SOI path diagnostics do not prove continuous path feasibility.",
            "This coast history supplies exit-duration guesses only. The optimiser regenerates all states with the small-thrust two-substep RK7 rollout.",
            "No full spacecraft-SPK comparison is attempted for these non-nominal coast seeds; this stage reports hard-target residuals and independent two-body numerical checks.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "four_phase_seed.npz", **arrays)
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    definition["active_initial_guess_artifacts"] = {
        "four_phase": str(OUT / "four_phase_seed.npz"),
        "validation": str(OUT / "result.json"),
    }
    (OUT / "problem_definition.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")
    pointer_path = DATA_ROOT / "active_configuration.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer.update(
        active_stage=str(OUT),
        problem_definition=str(OUT / "problem_definition.json"),
        initial_guess=str(OUT / "four_phase_seed.npz"),
    )
    pointer_path.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    lines = [
        "DYNAMICS-ONLY SOLAR ROLLOUTS TO FIXED MARS AND ASTEROID TARGETS",
        "",
        result["status"],
        "Earth/Mars SOI exits remain free events; their coast-seed values are initial guesses.",
        f"Fixed asteroid endpoint: {arrival['epoch']}",
        "",
        "ENDPOINT MISMATCHES (propagated minus target, Sun/J2000)",
        f"Mars entry: {result['mars_entry_mismatch']}",
        f"Psyche asteroid: {result['arrival_mismatch']}",
        "",
        "Seed thrust: zero everywhere; seed mass: 2747 kg throughout. No optimisation performed.",
        "Spacecraft ephemeris state calls were disabled while generating the seed.",
        "",
        "NUMERICAL CHECKS",
        *[json.dumps(r, indent=2) for r in summaries],
        "",
        "LIMITATIONS",
        *result["limitations"],
        "",
        "Reproduce: python -m psyche_mission_events_reconstruction.seed_solar_phases",
        "See README.md for plotting commands.",
    ]
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("mars_entry_mismatch", "arrival_mismatch", "free_exit_reproduction")
            },
            indent=2,
        )
    )
    return result


if __name__ == "__main__":
    run()
