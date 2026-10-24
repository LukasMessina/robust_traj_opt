"""Mars-only coast seed, followed by separate SPK validation.

No spacecraft history is read until propagation and two-body numerical
checks finish. Only the fixed entry six-state initialises the trajectory.
"""

from __future__ import annotations
from .paths import DATA_ROOT
import json
from pathlib import Path
import numpy as np
import spiceypy as raw
from scipy.optimize import brentq
from tudatpy.interface import spice
from .build_ephemeris_cache import policy
from .ephemeris_cache import EphemerisCache
from .kernels import ROOT, epoch_record, inspect_spk, loaded_kernels, sha256, state_si
from .mars_kernels import KERNELS, ensure_interval, paths_verified
from .rollout import rollout_to_soi_exit, soi_radius_and_rate, translate_to_sun

OUT = DATA_ROOT / "soi/mars_rollout"


def run(*, output=OUT, boundary_path=None, earth_path=None, soi_model=None):
    OUT = Path(output)
    boundary_path = (
        Path(boundary_path)
        if boundary_path is not None
        else DATA_ROOT / "soi/mars_entry_result.json"
    )
    earth_path = (
        Path(earth_path) if earth_path is not None else DATA_ROOT / "soi/earth_rollout/result.json"
    )
    boundary = json.loads(boundary_path.read_text(encoding="utf-8"))
    earth = json.loads(earth_path.read_text(encoding="utf-8"))
    if (
        boundary["status"] != "verified fixed boundary extracted"
        or boundary["spacecraft_id"] != -255
        or boundary["origin_id"] != 499
        or (boundary["frame"] != "J2000")
        or (boundary["units"] != "m, m/s")
        or (boundary["aberration_corrections"] != "NONE")
    ):
        raise ValueError("Unverified or incompatible fixed Mars boundary")
    cache_path = Path(boundary["planetary_cache"]["path"])
    if sha256(cache_path) != boundary["planetary_cache"]["sha256"]:
        raise ValueError("Planetary cache differs from fixed-boundary provenance")
    cache = EphemerisCache.load(cache_path)
    if soi_model is None:
        from .mission_soi import fixed_mission_soi

        soi_model = fixed_mission_soi(499, cache.metadata["mu_m3_s2"])
    mu = cache.metadata["mu_m3_s2"]["499"]
    if mu != boundary["soi_model"]["mu_mars_m3_s2"]:
        raise ValueError("Mars GM differs between boundary and rollout")
    if boundary["soi_model"].get("semimajor_axis_m") != (
        None if soi_model is None else soi_model.semimajor_axis_m
    ):
        raise ValueError("SOI convention differs between fixed boundary and rollout")
    if earth["seed_sigma_n"] != 0.0 or any(earth["seed_thrust_n"]):
        raise ValueError("Cannot infer coast-chain mass from a nonzero-thrust Earth seed")
    start = boundary["epoch"]["et_tdb_s"]
    initial = np.r_[boundary["mars_relative_state_si"], earth["seed_mass_kg"]]
    if raw.ktotal("ALL") or spice.get_total_count_of_kernels_loaded():
        raise RuntimeError("Use a fresh process with both SPICE kernel pools empty")
    horizon = start + 10 * 86400.0
    arc = rollout_to_soi_exit(initial, start, horizon, 499, cache, soi_model=soi_model)
    refined = rollout_to_soi_exit(
        initial, start, horizon, 499, cache, rtol=2e-13, max_step_s=900.0, soi_model=soi_model
    )
    radial = lambda dt: float(np.dot(arc.at_elapsed(dt)[:3], arc.at_elapsed(dt)[3:6]))
    peri_elapsed = brentq(radial, 0.0, arc.duration_s, xtol=1e-08)
    peri = arc.at_elapsed(peri_elapsed)
    uniform_elapsed = np.linspace(0.0, arc.duration_s, 4001)
    elapsed = np.unique(
        np.r_[
            uniform_elapsed,
            peri_elapsed,
            np.linspace(
                max(0.0, peri_elapsed - 10800.0), min(arc.duration_s, peri_elapsed + 10800.0), 1081
            ),
        ]
    )
    states = arc.at_elapsed(elapsed)
    epochs = start + elapsed
    geometry = np.array([soi_radius_and_rate(cache, 499, t, soi_model) for t in epochs])
    radii = np.linalg.norm(states[:, :3], axis=1)
    c3 = np.sum(states[:, 3:6] ** 2, axis=1) - 2 * mu / radii
    momentum = np.cross(states[:, :3], states[:, 3:6])
    elements = raw.oscelt(initial[:6] / 1000.0, 0.0, mu / 1000000000.0)
    conics = np.array([raw.conics(elements, float(dt)) * 1000.0 for dt in elapsed])
    conic_position = np.linalg.norm(states[:, :3] - conics[:, :3], axis=1)
    conic_velocity = np.linalg.norm(states[:, 3:6] - conics[:, 3:6], axis=1)
    final = arc.final_state
    exit_radius, radius_rate = soi_radius_and_rate(cache, 499, arc.end_et, soi_model)
    crossing_rate = np.dot(final[:3], final[3:6]) / np.linalg.norm(final[:3]) - radius_rate
    sun_final = translate_to_sun(final, 499, arc.end_et, cache)
    closure = sun_final[:6] - cache.state(499, arc.end_et) - final[:6]
    checks = {
        "max_kepler_position_difference_m": float(conic_position.max()),
        "max_kepler_velocity_difference_m_s": float(conic_velocity.max()),
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
        "entry_soi_residual_m": float(radii[0] - geometry[0, 0]),
        "exit_soi_residual_m": float(radii[-1] - exit_radius),
        "exit_surface_crossing_rate_m_s": float(crossing_rate),
        "max_mass_change_kg": float(np.max(np.abs(states[:, 6] - initial[6]))),
        "sun_translation_position_closure_m": float(np.linalg.norm(closure[:3])),
        "sun_translation_velocity_closure_m_s": float(np.linalg.norm(closure[3:])),
        "spacecraft_ephemeris_queries_during_rollout": 0,
        "spice_kernels_loaded_during_rollout": raw.ktotal("ALL"),
        "tudat_kernels_loaded_during_rollout": spice.get_total_count_of_kernels_loaded(),
        "sample_count": int(elapsed.size),
    }
    if (
        conic_position.max() > 0.02
        or conic_velocity.max() > 2e-06
        or checks["exit_epoch_refinement_difference_s"] > 1e-05
        or (checks["exit_position_refinement_difference_m"] > 0.02)
        or (abs(checks["exit_soi_residual_m"]) > 0.001)
        or (crossing_rate <= 0)
        or (checks["max_mass_change_kg"] != 0)
        or (np.max(radii[1:-1] - geometry[1:-1, 0]) >= 0)
    ):
        raise AssertionError(f"Mars rollout numerical checks failed: {checks}")
    paths = paths_verified()
    with loaded_kernels(paths):
        mission = inspect_spk(paths[-1])
        mars_metadata = inspect_spk(paths[2])
        planet_metadata = inspect_spk(paths[3])
        for metadata, target in (
            (mission, -255),
            (mars_metadata, 499),
            (planet_metadata, 4),
            (planet_metadata, 10),
        ):
            ensure_interval(metadata, target, start, arc.end_et)
        truth = np.array([state_si(-255, 499, t) for t in epochs])
        raw_exit = raw.spkezr("-255", arc.end_et, "J2000", "NONE", "499")[0] * 1000
        raw_entry = state_si(-255, 499, start)
        entry_difference = raw_entry - initial[:6]
        if (
            np.linalg.norm(entry_difference[:3]) > 0.001
            or np.linalg.norm(entry_difference[3:]) > 1e-06
        ):
            raise ValueError("Stored fixed entry does not reproduce with pinned kernels")
        start_record, end_record, peri_record = map(
            epoch_record, (start, arc.end_et, start + peri_elapsed)
        )
        planet_exit = state_si(499, 10, arc.end_et)
        frame_comparison = sun_final[:6] - (final[:6] + planet_exit)
        bary_segments = [s for s in mission["segments"] if s["centre"] == 4]
        if len(bary_segments) != 1:
            raise ValueError("Unexpected number of Mars-barycentric SPK segments")
        segment_start = bary_segments[0]["start"]
        segment_end = bary_segments[0]["end"]
    difference = states[:, :6] - truth
    position_difference = np.linalg.norm(difference[:, :3], axis=1)
    velocity_difference = np.linalg.norm(difference[:, 3:], axis=1)
    uniform_indices = np.searchsorted(elapsed, uniform_elapsed)
    reference_exit = boundary["reconstructed_exit_reference_only"]
    reference_peri = boundary["closest_approach_reference_only"]
    comparison = {
        "definition": "Rollout minus reconstructed SPK, at identical epochs in Mars-centred J2000. Sample maxima, not rigorous continuous bounds.",
        "sample_count": int(elapsed.size),
        "uniform_rms_sample_count": int(uniform_elapsed.size),
        "max_position_difference_m": float(position_difference.max()),
        "max_velocity_difference_m_s": float(velocity_difference.max()),
        "uniform_time_rms_position_difference_m": float(
            np.sqrt(np.mean(position_difference[uniform_indices] ** 2))
        ),
        "uniform_time_rms_velocity_difference_m_s": float(
            np.sqrt(np.mean(velocity_difference[uniform_indices] ** 2))
        ),
        "at_seed_exit_position_difference_m": float(position_difference[-1]),
        "at_seed_exit_velocity_difference_m_s": float(velocity_difference[-1]),
        "at_seed_exit_difference_si": difference[-1].tolist(),
        "reconstructed_exit_epoch": reference_exit["epoch"],
        "seed_minus_reconstructed_exit_epoch_s": arc.end_et - reference_exit["et_tdb_s"],
        "seed_minus_reconstructed_periapsis_epoch_s": start
        + peri_elapsed
        - reference_peri["epoch"]["et_tdb_s"],
        "seed_minus_reconstructed_periapsis_radius_m": float(
            np.linalg.norm(peri[:3]) - reference_peri["centre_distance_m"]
        ),
        "reference_at_seed_exit_mars_state_si": truth[-1].tolist(),
        "stored_entry_position_reproduction_m": float(np.linalg.norm(entry_difference[:3])),
        "stored_entry_velocity_reproduction_m_s": float(np.linalg.norm(entry_difference[3:])),
        "tudat_vs_raw_at_exit_position_m": float(np.linalg.norm((truth[-1] - raw_exit)[:3])),
        "tudat_vs_raw_at_exit_velocity_m_s": float(np.linalg.norm((truth[-1] - raw_exit)[3:])),
        "cached_vs_direct_translation_position_m": float(np.linalg.norm(frame_comparison[:3])),
        "cached_vs_direct_translation_velocity_m_s": float(np.linalg.norm(frame_comparison[3:])),
        "coverage_verified_for_full_rollout": True,
        "use": "Validation only. No SPK history used for initialisation; no fitting or endpoint snapping.",
    }
    literature = literature_review(boundary, segment_start, segment_end)
    result = {
        "status": "Dynamics-only Mars coast initial guess generated and validated; not an optimisation result",
        "start_epoch": start_record,
        "exit_epoch": end_record,
        "duration_s": arc.duration_s,
        "duration_days": arc.duration_s / 86400.0,
        "mars_relative_initial_state_si": initial[:6].tolist(),
        "mars_relative_exit_state_si": final[:6].tolist(),
        "sun_relative_exit_state_si": sun_final[:6].tolist(),
        "exit_soi_radius_m": float(exit_radius),
        "periapsis": {
            "epoch": peri_record,
            "centre_distance_m": float(np.linalg.norm(peri[:3])),
            "mars_relative_state_si": peri[:6].tolist(),
            "use": "diagnostic only; no new constraint",
        },
        "seed_mass_kg": float(initial[6]),
        "seed_thrust_n": [0.0, 0.0, 0.0],
        "seed_sigma_n": 0.0,
        "mass_provenance": "Carried from Earth coast seed. The ungenerated zero-thrust solar coast preserves mass. This is not a measured Mars mass or a fixed OCP boundary mass.",
        "gravity": "Mars point mass only",
        "mu_mars_m3_s2": mu,
        "frame": "J2000",
        "origin_id": 499,
        "units": "m, m/s, kg",
        "time_scale": "ET/TDB seconds since J2000",
        "soi": (
            "fixed-semimajor-axis Laplace model"
            if soi_model is not None and soi_model.semimajor_axis_m is not None
            else "instantaneous-distance Laplace model"
        ),
        "semimajor_axis_m": None if soi_model is None else soi_model.semimajor_axis_m,
        "integrator": {
            "method": "DOP853 IVP",
            "rtol": 2e-12,
            "atol_si": [1e-05] * 3 + [1e-09] * 3 + [1e-10],
            "max_step_s": 3600.0,
            "search_horizon_s": horizon - start,
            "refined_rtol": 2e-13,
            "refined_max_step_s": 900.0,
        },
        "checks": checks,
        "spk_comparison": comparison,
        "literature_review": literature,
        "limitations": [
            "Mars point mass only; solar tides, Mars harmonics, moons and other perturbations omitted by the requested phase model.",
            "A Mars-centred frame is accelerating, so the omitted solar effect is differential solar acceleration (tides), not the full heliocentric solar acceleration.",
            "The seed has zero thrust, but the OCP Mars phase remains allowed to thrust and move its exit epoch/state.",
            "Published entry covariance unavailable; microseconds describe numerical precision conditional on the SPK/model, not physical accuracy.",
            "This module generates only the Mars coast timing history. Optimisation runs separately through deterministic_traj_opt.py; see README.md.",
        ],
        "provenance": {
            "fixed_mars_boundary_sha256": sha256(boundary_path),
            "earth_seed_sha256": sha256(earth_path),
            "planetary_cache_sha256": sha256(cache_path),
            "validation_kernel_load_order": [
                dict(name=s.name, url=s.url, sha256=s.sha256) for s in KERNELS
            ],
            "implementation_sha256": {
                p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("rollout.py"))
            },
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "mars_rollout.npz",
        elapsed_s=elapsed,
        epochs_et_tdb_s=epochs,
        mars_relative_states_si=states,
        soi_radii_m=geometry[:, 0],
        c3_drift_m2_s2=c3 - c3[0],
        kepler_position_error_m=conic_position,
        kepler_velocity_error_m_s=conic_velocity,
        purpose=np.array("Dynamics-only initial guess; no spacecraft ephemeris history used."),
    )
    np.savez_compressed(
        OUT / "spk_comparison_validation_only.npz",
        epochs_et_tdb_s=epochs,
        mars_relative_spk_states_si=truth,
        difference_si=difference,
        position_difference_m=position_difference,
        velocity_difference_m_s=velocity_difference,
        purpose=np.array("Validation only; prohibited as initial-guess data."),
    )
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT / "literature_review.json").write_text(json.dumps(literature, indent=2), encoding="utf-8")
    (OUT / "problem_definition.json").write_text(json.dumps(policy(), indent=2), encoding="utf-8")
    np.savetxt(
        OUT / "exit_state.csv",
        np.r_[arc.end_et, final, sun_final[:6]][None],
        delimiter=",",
        header="et_tdb_s,x_mars_m,y_mars_m,z_mars_m,vx_mars_m_s,vy_mars_m_s,vz_mars_m_s,mass_kg,x_sun_m,y_sun_m,z_sun_m,vx_sun_m_s,vy_sun_m_s,vz_sun_m_s",
        comments="",
    )
    write_report(result, OUT)
    print(
        json.dumps(
            {
                "exit": end_record,
                "duration_days": result["duration_days"],
                "periapsis": result["periapsis"],
                "checks": checks,
                "spk_comparison": comparison,
            },
            indent=2,
        )
    )
    return result


def literature_review(boundary, segment_start, segment_end):
    """Record the bounded literature search, not an assertion of universal absence."""
    fixed = boundary["fixed_a_sensitivity"]["events"]
    instant_epoch = boundary.get("instantaneous_distance_comparison", {}).get(
        "entry_epoch", boundary["epoch"]
    )
    return {
        "search_date": "2026-10-23",
        "finding": "No explicit exact Mars SOI-entry timestamp found in the primary mission sources and literature inspected. No independent published SOI epoch is available from this search for an event-error comparison.",
        "sources": [
            {
                "title": "NASA's Psyche Mission to Fly by Mars for Gravity Assist",
                "date": "2026-05-08",
                "url": "https://science.nasa.gov/blogs/psyche/2026/05/08/nasas-psyche-mission-to-fly-by-mars-for-gravity-assist/",
                "finding": "Reports May 15 flyby; does not specify an SOI-entry timestamp or radius convention.",
            },
            {
                "title": "NASA's Psyche Mission Aces Mars Flyby, Targets Metal-Rich Asteroid",
                "date": "2026-05-19",
                "url": "https://www.nasa.gov/missions/psyche-mission/nasas-psyche-mission-aces-mars-flyby-targets-metal-rich-asteroid/",
                "finding": "Reports May 15 closest approach and 4609 km surface altitude; no exact SOI entry. Image acquisition times are not SOI-entry events.",
            },
            {
                "title": "Psyche Mission Description and Design Rationale",
                "year": 2025,
                "doi": "10.1007/s11214-025-01218-x",
                "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12521304/",
                "finding": "Mission synopsis and Mars gravity-assist navigation section give May 2026 and operations/maneuver windows relative to closest approach; no exact SOI entry found. Operational phase limits are not SOI crossings.",
            },
            {
                "title": "JPL MDNav reconstructed spacecraft SPK",
                "url": KERNELS[-1].url,
                "finding": "Binary comments identify reconstruction, centre-chain models, creation time and coverage endpoints; do not label SOI-entry/exit timestamps. Exact segment endpoints are data-structure metadata.",
            },
        ],
        "queries": [
            "Psyche Mars flyby May 2026 sphere of influence entry",
            "Psyche Mars SOI 2026 May 14",
            '"Psyche" "2026" "SOI entry"',
            '"Psyche" "2026" "sphere-of-influence" Mars',
            '"Psyche" "Mars" "sphere of influence" site:jpl.nasa.gov',
            '"Psyche" "Mars" "SOI" site:nasa.gov',
            '"Psyche" "May 14" "2026" "07:51"',
        ],
        "scope_limit": "Web search and listed primary sources, not an exhaustive search of internal mission documents. Unverified secondary/forum statements are not timing evidence. The NAIF web index yielded no matching timestamp/label; local pinned binary metadata supplies the segment comparison.",
        "computed_entry_instantaneous": instant_epoch,
        "computed_entry_fixed_a": fixed[0]["epoch"],
        "fixed_a_entry_minus_instantaneous_s": fixed[0]["et_tdb_s"] - instant_epoch["et_tdb_s"],
        "spk_mars_barycentric_segment_start": segment_start,
        "spk_mars_barycentric_segment_end": segment_end,
        "segment_start_minus_fixed_a_entry_s": segment_start["et_tdb_s"] - fixed[0]["et_tdb_s"],
        "segment_start_minus_instantaneous_entry_s": segment_start["et_tdb_s"]
        - instant_epoch["et_tdb_s"],
        "segment_end_minus_fixed_a_exit_s": segment_end["et_tdb_s"] - fixed[1]["et_tdb_s"],
        "interpretation": "The segment-start proximity to the fixed-a crossing is suggestive of a similar boundary choice during SPK construction, but its purpose is undocumented here. It is not independent confirmation of a published SOI event.",
    }


def write_report(r, output=OUT):
    c, v, literature = (r["checks"], r["spk_comparison"], r["literature_review"])
    lines = [
        "PSYCHE: MARS-GRAVITY INITIAL GUESS AND SOI TIMING SOURCE REVIEW",
        "",
        r["status"],
        "",
        "SOI convention: " + r["soi"],
        "Fixed Mars entry: " + r["start_epoch"]["utc"],
        "Dynamically detected seed exit: " + r["exit_epoch"]["utc"],
        f"Exit ET/TDB seconds since J2000: {r['exit_epoch']['et_tdb_s']:.9f}",
        f"Mars-phase duration: {r['duration_s']:.9f} s ({r['duration_days']:.9f} days)",
        f"Exit SOI radius: {r['exit_soi_radius_m'] / 1000:.9f} km",
        "Mars-relative J2000 exit position [km]: "
        + ", ".join((f"{x / 1000:.9f}" for x in r["mars_relative_exit_state_si"][:3])),
        "Mars-relative J2000 exit velocity [km/s]: "
        + ", ".join((f"{x / 1000:.12f}" for x in r["mars_relative_exit_state_si"][3:])),
        f"Closest centre distance: {r['periapsis']['centre_distance_m'] / 1000:.9f} km at {r['periapsis']['epoch']['utc']}",
        f"Constant seed mass: {r['seed_mass_kg']:g} kg; thrust: 0 N.",
        r["mass_provenance"],
        "Exit state/time are seed outputs, not fixed OCP constraints. The Mars phase may thrust in the OCP.",
        "The spacecraft SPK was loaded only after the seed and numerical checks completed.",
        "The fixed entry state is the only spacecraft ephemeris data used to initialise this phase.",
        "",
        "Numerical verification against an independent two-body conic and tighter propagation:",
    ]
    lines += [f"  {key}: {value}" for key, value in c.items()]
    lines += ["", "Comparison with reconstructed SPK (validation only):"]
    lines += [f"  {key}: {value}" for key, value in v.items()]
    lines += ["", "Published-epoch search:", literature["finding"]]
    for source in literature["sources"]:
        lines += [source["title"], source["url"], source["finding"]]
    lines += [
        "",
        "Timing comparison; all timestamps on 2026-05-14, UTC:",
        "Computed instantaneous-distance entry: "
        + literature["computed_entry_instantaneous"]["utc"],
        "Computed fixed-semimajor-axis entry: " + literature["computed_entry_fixed_a"]["utc"],
        "SPK Mars-barycentric segment begins: "
        + literature["spk_mars_barycentric_segment_start"]["utc"],
        f"Segment start minus fixed-a entry: {literature['segment_start_minus_fixed_a_entry_s']:.9f} s",
        f"Segment start minus instantaneous entry: {literature['segment_start_minus_instantaneous_entry_s']:.9f} s",
        literature["interpretation"],
        literature["scope_limit"],
        "",
        "Limitations:",
    ]
    lines += r["limitations"]
    lines += [
        "",
        "Reproduce with det_traj_opt_env active:",
        "  python -m psyche_mission_events_reconstruction."
        + ("soi_reconstruction" if r["semimajor_axis_m"] is not None else "seed_mars_flyby"),
        "Plot with a Matplotlib environment:",
        "See README.md for reconstruction and optimisation plotting commands.",
        "",
        "Arrival is a hard equality to asteroid 2000016 at 2029-08-01 12:00:00 TDB.",
        "See active_configuration.json for the arrival target and solar rollouts; no optimisation was performed.",
    ]
    (Path(output) / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
