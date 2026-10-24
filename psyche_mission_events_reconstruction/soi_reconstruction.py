"""Audit fixed/instantaneous SOI crossings and regenerate selected coast seeds.

Run with the Tudat environment. Archived instantaneous products are retained.
SPK histories are used for boundary extraction and verification, not seeding.
"""

from __future__ import annotations
from psyche_mission_events_reconstruction.paths import DATA_ROOT, resolve_input
import json
from pathlib import Path
import numpy as np
import spiceypy as raw
from scipy.optimize import brentq
from tudatpy.interface import spice
from .build_ephemeris_cache import policy
from .ephemeris_cache import EphemerisCache
from .kernels import (
    ROOT,
    TIP_KERNELS,
    epoch_record,
    inspect_spk,
    kernel_paths,
    loaded_kernels,
    selected_segment,
    sha256,
    state_si,
)
from .mars_kernels import KERNELS, ensure_interval, paths_verified
from .mission_soi import (
    ACTIVE_STAGE as OUT,
    ACTIVE_MARS_BOUNDARY,
    AU_M,
    AXIS_SOURCE,
    EARTH_EMB_SENSITIVITY_AU,
    fixed_mission_soi,
)
from .soi_events import LaplaceSOI, find_soi_crossings
from . import seed_earth_departure, seed_mars_flyby


def inspect_encounter(body, paths, cache):
    with loaded_kernels(paths):
        metadata = inspect_spk(paths[-1])
        planet_meta = inspect_spk(next((p for p in paths if p.name == "de440s.bsp")))
        windows = metadata["coverage"]["-255"]
        if len(windows) != 1:
            raise ValueError("Expected continuous spacecraft coverage")
        start, end = (windows[0]["start"]["et_tdb_s"], windows[0]["end"]["et_tdb_s"])
        for target in (399, 10) if body == 399 else (4, 10):
            ensure_interval(planet_meta, target, start, end)
        if body == 499:
            mars_meta = inspect_spk(next((p for p in paths if p.name == "mar097.bsp")))
            ensure_interval(mars_meta, 499, start, end)
        gm = {
            str(body): float(spice.get_body_gravitational_parameter(str(body))),
            "10": float(spice.get_body_gravitational_parameter("10")),
        }
        if any((gm[k] != cache.metadata["mu_m3_s2"][k] for k in gm)):
            raise ValueError("Cache and event extraction GMs differ")
        models = {
            "fixed_a": fixed_mission_soi(body, gm),
            "instantaneous_distance": LaplaceSOI(gm[str(body)], gm["10"]),
        }
        if body == 399:
            models["fixed_EMB_coefficient_sensitivity"] = LaplaceSOI(
                gm["399"], gm["10"], EARTH_EMB_SENSITIVITY_AU * AU_M
            )
        joins = [s["start"]["et_tdb_s"] for s in metadata["segments"]][1:]
        sc = lambda t: state_si(-255, body, t)
        planet = lambda t: state_si(body, 10, t)
        events = {}
        for name, model in models.items():
            coarse = find_soi_crossings(sc, planet, model, start, end, 1800.0, joins)
            fine = find_soi_crossings(sc, planet, model, start, end, 300.0, joins)
            expected = ["exit"] if body == 399 else ["entry", "exit"]
            if [e["direction"] for e in fine] != expected or [
                e["direction"] for e in coarse
            ] != expected:
                raise AssertionError("Unexpected SOI crossings")
            for event, other in zip(fine, coarse):
                t = event["et_tdb_s"]
                event["epoch"] = epoch_record(t)
                event["coarse_fine_epoch_difference_s"] = t - other["et_tdb_s"]
                event["selected_spacecraft_segment"] = selected_segment(-255, t)
                event["sun_relative_state_si"] = state_si(-255, 10, t).tolist()
                event["cached_sun_relative_link_state_si"] = (sc(t) + cache.state(body, t)).tolist()
                gminus = model.residual_and_rate(sc(t - 1.0), planet(t - 1.0))[0]
                gplus = model.residual_and_rate(sc(t + 1.0), planet(t + 1.0))[0]
                event["residual_one_second_before_m"] = gminus
                event["residual_one_second_after_m"] = gplus
                reference = raw.spkezr("-255", t, "J2000", "NONE", str(body))[0] * 1000.0
                delta = sc(t) - reference
                event["tudat_vs_raw_position_m"] = float(np.linalg.norm(delta[:3]))
                event["tudat_vs_raw_velocity_m_s"] = float(np.linalg.norm(delta[3:]))
                event["cache_planet_position_difference_m"] = float(
                    np.linalg.norm((cache.state(body, t) - planet(t))[:3])
                )
                event["event_minus_nearest_segment_join_s"] = min((t - j for j in joins), key=abs)
                if (
                    abs(event["residual_m"]) > 0.002
                    or abs(t - other["et_tdb_s"]) > 1e-06
                    or gminus * gplus >= 0
                    or (event["tudat_vs_raw_position_m"] > 0.001)
                ):
                    raise AssertionError("SOI event refinement or interface check failed")
            events[name] = fine
        transitions = []
        for before, after in zip(metadata["segments"][:-1], metadata["segments"][1:]):
            if before["centre"] == after["centre"]:
                continue
            t = after["start"]["et_tdb_s"]
            dt = 0.01
            ym2, ym, yp, yp2 = (sc(t + d * dt) for d in (-2, -1, 1, 2))
            position_gap = yp[:3] - dt * yp[3:] - (ym[:3] + dt * ym[3:])
            velocity_gap = yp[3:] - (yp2[3:] - yp[3:]) - (ym[3:] + ym[3:] - ym2[3:])
            transitions.append(
                {
                    "epoch": epoch_record(t),
                    "origin_before": before["centre"],
                    "origin_after": after["centre"],
                    "extrapolated_position_gap_m": float(np.linalg.norm(position_gap)),
                    "extrapolated_velocity_gap_m_s": float(np.linalg.norm(velocity_gap)),
                    "check_step_s": dt,
                }
            )
        plot_data = {}
        for event, transition in zip(events["fixed_a"], transitions):
            t = transition["epoch"]["et_tdb_s"]
            shift = next(
                (
                    e["et_tdb_s"] - t
                    for e in events["instantaneous_distance"]
                    if e["direction"] == event["direction"]
                )
            )
            margin = max(60.0, abs(shift) * 0.1)
            offsets = np.linspace(min(0.0, shift) - margin, max(0.0, shift) + margin, 601)
            label = event["direction"]
            plot_data[label + "_offset_s"] = offsets
            for name in ("fixed_a", "instantaneous_distance"):
                plot_data[label + "_" + name + "_residual_m"] = np.array(
                    [models[name].residual_and_rate(sc(t + d), planet(t + d))[0] for d in offsets]
                )
        closest = None
        if body == 499:
            entry, exit_event = events["fixed_a"]
            begin = entry["et_tdb_s"]
            peri = begin + brentq(
                lambda dt: np.dot(sc(begin + dt)[:3], sc(begin + dt)[3:]),
                0.0,
                exit_event["et_tdb_s"] - begin,
                xtol=1e-08,
            )
            closest = {
                "epoch": epoch_record(peri),
                "centre_distance_m": float(np.linalg.norm(sc(peri)[:3])),
                "planet_relative_state_si": sc(peri).tolist(),
                "use": "validation only; no periapsis constraint added",
            }
    return (
        {
            "body_id": body,
            "spacecraft_spk_metadata": metadata,
            "events": events,
            "origin_transitions": transitions,
            "closest_approach_reference_only": closest,
            "coverage_checked": True,
            "mu_m3_s2": gm,
        },
        plot_data,
    )


def write_mars_boundary(mars, cache_path):
    entry, exit_event = mars["events"]["fixed_a"]
    instant = mars["events"]["instantaneous_distance"]
    gm = mars["mu_m3_s2"]
    model = fixed_mission_soi(499, gm)
    boundary = {
        "schema_version": 2,
        "status": "verified fixed boundary extracted",
        "mission_event": "Mars SOI entry",
        "spacecraft_id": -255,
        "origin_id": 499,
        "frame": "J2000",
        "aberration_corrections": "NONE",
        "units": "m, m/s",
        "time_scale": "ET/TDB seconds since J2000",
        "epoch": entry["epoch"],
        "mars_relative_state_si": entry["planet_relative_state_si"],
        "sun_relative_state_si": entry["sun_relative_state_si"],
        "sun_relative_link_state_si": entry["cached_sun_relative_link_state_si"],
        "entry": entry,
        "reconstructed_exit_reference_only": exit_event,
        "closest_approach_reference_only": mars["closest_approach_reference_only"],
        "soi_model": {
            "convention": "fixed_semimajor_axis",
            "formula": "a_Mars*(mu_Mars/mu_Sun)^(2/5)",
            "semimajor_axis_m": model.semimajor_axis_m,
            "a_au": model.semimajor_axis_m / AU_M,
            "mu_mars_m3_s2": gm["499"],
            "mu_sun_m3_s2": gm["10"],
            "source": AXIS_SOURCE,
            "interpretation": "Frozen J2000 coefficient from JPL approximate-elements Table 1. Planet positions still use DE440/MAR097, not approximate Keplerian elements.",
        },
        "fixed_a_sensitivity": {
            "events": [entry, exit_event],
            "use": "Active convention; key retained for report compatibility",
        },
        "instantaneous_distance_comparison": {
            "entry_epoch": instant[0]["epoch"],
            "events": instant,
            "use": "Archived alternative, not the active boundary",
        },
        "planetary_cache": {
            "path": str(cache_path.resolve()),
            "sha256": sha256(cache_path),
            "validation_report": str(cache_path.parent / "validation.json"),
        },
        "kernel_load_order": [dict(name=s.name, url=s.url, sha256=s.sha256) for s in KERNELS],
        "coverage": mars["spacecraft_spk_metadata"],
        "boundary_semantics": {
            "entry": "Fix Mars-centred six-state and epoch; mass linked, not fixed.",
            "exit": "Free state and epoch on constant-radius outward crossing.",
            "initial_guess": "Forward dynamics only from this boundary; no spacecraft SPK history.",
            "frame_link": "Use cached Mars state for the linked Sun-centred state; do not independently impose raw Sun state and a slightly different cached translation.",
        },
        "implementation_sha256": {
            p.name: sha256(p)
            for p in (
                Path(__file__),
                Path(__file__).with_name("soi_events.py"),
                Path(__file__).with_name("mission_soi.py"),
            )
        },
    }
    ACTIVE_MARS_BOUNDARY.write_text(json.dumps(boundary, indent=2), encoding="utf-8")
    np.savetxt(
        OUT / "mars_entry_state.csv",
        np.r_[
            entry["et_tdb_s"],
            entry["planet_relative_state_si"],
            entry["cached_sun_relative_link_state_si"],
        ][None],
        delimiter=",",
        header="et_tdb_s,x_mars_m,y_mars_m,z_mars_m,vx_mars_m_s,vy_mars_m_s,vz_mars_m_s,x_sun_m,y_sun_m,z_sun_m,vx_sun_m_s,vy_sun_m_s,vz_sun_m_s",
        comments="",
    )


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = DATA_ROOT / "ephemerides/earth_mars_ephemerides.npz"
    validation = json.loads((DATA_ROOT / "ephemerides/validation.json").read_text(encoding="utf-8"))
    if sha256(cache_path) != validation["cache_sha256"]:
        raise ValueError("Unverified planetary cache")
    cache = EphemerisCache.load(cache_path)
    earth, earth_plot = inspect_encounter(399, kernel_paths(), cache)
    mars, mars_plot = inspect_encounter(499, paths_verified(), cache)
    write_mars_boundary(mars, cache_path)
    np.savez_compressed(
        OUT / "soi_comparison_validation_only.npz",
        **{"earth_" + k: v for k, v in earth_plot.items()},
        **{"mars_" + k: v for k, v in mars_plot.items()},
    )
    earth_seed = seed_earth_departure.run(
        output=OUT / "earth_rollout",
        cache_path=cache_path,
        soi_model=fixed_mission_soi(399, cache.metadata["mu_m3_s2"]),
    )
    mars_seed = seed_mars_flyby.run(
        output=OUT / "mars_rollout",
        boundary_path=ACTIVE_MARS_BOUNDARY,
        earth_path=OUT / "earth_rollout/result.json",
        soi_model=fixed_mission_soi(499, cache.metadata["mu_m3_s2"]),
    )
    with np.load(OUT / "earth_rollout/earth_rollout.npz", allow_pickle=False) as a:
        epochs, states = (a["epochs_et_tdb_s"], a["earth_relative_states_si"])
    with loaded_kernels(kernel_paths()):
        ensure_interval(earth["spacecraft_spk_metadata"], -255, epochs[0], epochs[-1])
        truth = np.array([state_si(-255, 399, t) for t in epochs])
    dp = np.linalg.norm(states[:, :3] - truth[:, :3], axis=1)
    dv = np.linalg.norm(states[:, 3:6] - truth[:, 3:], axis=1)
    earth_seed_comparison = {
        "at_seed_exit_position_difference_m": float(dp[-1]),
        "at_seed_exit_velocity_difference_m_s": float(dv[-1]),
        "sampled_max_position_difference_m": float(dp.max()),
        "sampled_max_velocity_difference_m_s": float(dv.max()),
        "seed_minus_reconstructed_exit_s": earth_seed["exit_et_tdb_s"]
        - earth["events"]["fixed_a"][0]["et_tdb_s"],
        "definition": "Same-epoch Earth-centred J2000; SPK evaluated after propagation for validation only",
    }
    np.savez_compressed(
        OUT / "earth_rollout/spk_comparison_validation_only.npz",
        epochs_et_tdb_s=epochs,
        position_difference_m=dp,
        velocity_difference_m_s=dv,
    )
    result = {
        "status": "Fixed-a convention selected; events verified and both planetary coast seeds regenerated",
        "earth": earth,
        "mars": mars,
        "earth_seed_spk_comparison": earth_seed_comparison,
        "selected_axes_au": {"399": 1.0, "499": 1.52371034},
        "earth_axis_note": "Nominal 1 au is an explicit modelling assumption. The JPL Table 1 coefficient 1.00000261 au belongs to the Earth-Moon barycentre and is a labelled sensitivity only. Neither value was fitted to segment timing.",
        "mars_axis_source": AXIS_SOURCE,
        "earth_seed_exit_utc": earth_seed["exit_utc"],
        "mars_seed_exit_utc": mars_seed["exit_epoch"]["utc"],
        "interpretation": "Fixed-radius crossings are substantially closer to SPK origin-transition epochs for both planets. This supports a similar radius convention in kernel construction but does not prove the producer's exact definition. Segment epochs are not imposed as physical events.",
        "limitation": "Neither SOI convention makes single-body dynamics an exact mission reconstruction. Root microseconds are numerical precision, not physical accuracy.",
        "provenance": {
            "Earth_validation_kernels": [
                dict(name=s.name, url=s.url, sha256=s.sha256) for s in TIP_KERNELS
            ],
            "Mars_validation_kernels": [
                dict(name=s.name, url=s.url, sha256=s.sha256) for s in KERNELS
            ],
            "planetary_cache_sha256": sha256(cache_path),
            "active_boundary_sha256": sha256(ACTIVE_MARS_BOUNDARY),
        },
    }
    (OUT / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    definition = policy()
    definition["active_initial_guess_artifacts"] = {
        "earth": str(OUT / "earth_rollout/result.json"),
        "mars": str(OUT / "mars_rollout/result.json"),
    }
    (OUT / "problem_definition.json").write_text(json.dumps(definition, indent=2), encoding="utf-8")
    manifest = {
        "active_stage": str(OUT),
        "problem_definition": str(OUT / "problem_definition.json"),
        "mars_entry_boundary": str(ACTIVE_MARS_BOUNDARY),
        "soi_convention": "fixed_semimajor_axis",
        "archived_stages": ["stage03_earth_rollout", "stage04_mars_entry", "stage05_mars_rollout"],
    }
    (DATA_ROOT / "active_configuration.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_report(result, earth_seed, mars_seed)


def write_report(r, earth_seed, mars_seed):
    lines = [
        "PSYCHE: FIXED-SEMIMAJOR-AXIS SOI UPDATE",
        "",
        r["status"],
        "Earth: a=1 au, planet-centred sphere and Earth-only GM.",
        "Mars: a=1.52371034 au, planet-centred sphere and Mars-only GM.",
        r["earth_axis_note"],
        "Mars coefficient source: " + AXIS_SOURCE,
        "",
    ]
    for name in ("earth", "mars"):
        data = r[name]
        lines += [name.upper() + " RECONSTRUCTED EVENTS AND SPK METADATA"]
        for transition in data["origin_transitions"]:
            lines += [
                f"SPK origin {transition['origin_before']} -> {transition['origin_after']}: {transition['epoch']['utc']}",
                f"Join extrapolation residual: {transition['extrapolated_position_gap_m']:.9g} m, {transition['extrapolated_velocity_gap_m_s']:.9g} m/s",
            ]
        for convention, events in data["events"].items():
            for event in events:
                lines += [
                    f"{convention} {event['direction']}: {event['epoch']['utc']}; radius {event['soi_radius_m'] / 1000:.9f} km; event minus nearest segment transition {event['event_minus_nearest_segment_join_s']:+.9f} s",
                    f"  root residual {event['residual_m']:.9g} m; coarse/fine difference {event['coarse_fine_epoch_difference_s']:.9g} s",
                ]
        lines += [""]
    lines += [
        r["interpretation"],
        r["limitation"],
        "",
        "REGENERATED DYNAMICS-ONLY SEEDS",
        "Earth seed exit: " + earth_seed["exit_utc"],
        "Mars seed exit: " + mars_seed["exit_epoch"]["utc"],
        "The new Mars entry uses the SPK Sun-centred segment just before the origin switch; SPICE converts it to Mars centre 499.",
        "Only boundary states initialise propagation; sampled spacecraft SPKs are validation only.",
        "Both exit states and epochs remain free in the OCP. Radius is fixed, not crossing time.",
        "TIP unchanged; seed thrust zero; mass carried continuously at 2747 kg.",
        "Planetary cache unchanged and already validated; fixed SOI evaluation needs no planetary interpolation, translations still do.",
        "Earth seed SPK comparison: " + json.dumps(r["earth_seed_spk_comparison"]),
        "Mars seed SPK comparison: " + json.dumps(mars_seed["spk_comparison"]),
        "",
        "Numerical seed checks:",
        "Earth: " + json.dumps(earth_seed["checks"]),
        "Mars: " + json.dumps(mars_seed["checks"]),
        "",
        "Artifacts in this stage supersede instantaneous-distance stage03/04/05 for active mission use.",
        "Active configuration pointer: artifacts/active_configuration.json",
        "Arrival is the hard asteroid 2000016 state at 2029-08-01 12:00:00 TDB; see the active arrival artifact.",
        "No solar rollout or OCP solve performed in this update.",
        "",
        "Reproduce: python -m psyche_mission_events_reconstruction.soi_reconstruction",
        "Plots: python deterministic_traj_opt.py --case psyche --plots-only --output output/deterministic/validated",
    ]
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:26]))


if __name__ == "__main__":
    run()
