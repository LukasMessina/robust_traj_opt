"""Extract and validate the Psyche targeting interface point, without an NLP.

Run: <Tudat environment python> -m psyche_mission_events_reconstruction.tip [--download]
Plots are separate and consume saved arrays, never live SPICE queries.
"""

from __future__ import annotations
from psyche_mission_events_reconstruction.paths import DATA_ROOT, resolve_input
import argparse
import csv
import json
import platform
from pathlib import Path
import numpy as np
import spiceypy as raw
import tudatpy
from tudatpy.interface import spice
from .kernels import (
    ROOT,
    TIP_KERNELS,
    CORRECTION,
    FRAME,
    covers,
    epoch_record,
    inspect_spk,
    kernel_paths,
    loaded_kernels,
    selected_segment,
    sha256,
    state_si,
)
from .orbit import hyperbolic_diagnostics

OUT = DATA_ROOT / "tip"
PAPER_URL = "https://electricrocket.org/IEPC_2024/126.pdf"
SCHEDULE_URL = (
    "https://science.nasa.gov/blogs/psyche/2023/10/13/key-milestones-for-todays-psyche-launch/"
)
LAUNCH_UTC = "2023-10-13 14:19:43 UTC"
SECO_UTC = "2023-10-13 15:15:34 UTC"
TIP_UTC = "2023-10-13 15:30:34 UTC"
PLANNED_TIP_UTC = "2023-10-13 15:30:55 UTC"


def differences(a: np.ndarray, b: np.ndarray) -> dict:
    return dict(
        position_m=float(np.linalg.norm(a[:3] - b[:3])),
        velocity_m_s=float(np.linalg.norm(a[3:] - b[3:])),
    )


def run(download: bool = False, out: Path = OUT) -> dict:
    paths = kernel_paths(download)
    out.mkdir(parents=True, exist_ok=True)
    with loaded_kernels(paths):
        epochs = {
            name: float(spice.convert_date_string_to_ephemeris_time(value))
            for name, value in dict(
                launch=LAUNCH_UTC, seco2=SECO_UTC, tip=TIP_UTC, planned_tip=PLANNED_TIP_UTC
            ).items()
        }
        tip_et = epochs["tip"]
        mu = float(spice.get_body_gravitational_parameter("399"))
        if not 398000000000000.0 < mu < 400000000000000.0:
            raise ValueError("Unexpected Earth GM; check SI units and kernel pool")
        spacecraft = inspect_spk(paths[-1])
        planets = inspect_spk(paths[-2])
        intervals = spacecraft["coverage"]["-255"]
        if not covers(intervals, tip_et):
            raise ValueError("TIP is not inside spacecraft binary coverage")
        for target in (3, 399, 10):
            if not covers(planets["coverage"][str(target)], tip_et):
                raise ValueError(f"Incomplete planetary centre chain at TIP: {target}")
        active = selected_segment(-255, tip_et)
        if (
            Path(active["file"]) != paths[-1].resolve()
            or active["centre"] != 399
            or active["frame_id"] != 1
        ):
            raise ValueError("TIP did not select the expected Earth/J2000 reconstructed segment")
        earth_state = state_si(-255, 399, tip_et)
        sun_state = state_si(-255, 10, tip_et)
        earth_sun = state_si(399, 10, tip_et)
        diag = hyperbolic_diagnostics(earth_state, mu)
        sensitivity_dt = np.unique(np.r_[np.linspace(-60, 60, 241), -1, -0.5, 0, 0.5, 1, 21])
        arc_start = intervals[0]["start"]["et_tdb_s"]
        arc_end = tip_et + 2 * 3600
        arc_times = np.unique(
            np.r_[np.linspace(arc_start, arc_end, 481), tip_et, epochs["planned_tip"]]
        )
        query_times = np.unique(np.r_[arc_times, tip_et + sensitivity_dt])
        if not all((covers(intervals, float(t)) for t in query_times)):
            raise ValueError("Diagnostic grid extends outside spacecraft coverage")
        states = np.array([state_si(-255, 399, float(t)) for t in query_times])
        raw_states = np.array(
            [raw.spkezr("-255", float(t), FRAME, CORRECTION, "399")[0] * 1000 for t in query_times]
        )
        delta = states - raw_states
        raw_errors = dict(
            max_position_m=float(np.max(np.linalg.norm(delta[:, :3], axis=1))),
            max_velocity_m_s=float(np.max(np.linalg.norm(delta[:, 3:], axis=1))),
            samples=int(query_times.size),
        )
        closure = differences(sun_state, earth_state + earth_sun)
        timing_rows = []
        for dt in (-60, -21, -1, -0.5, 0, 0.5, 1, 21, 60):
            y = state_si(-255, 399, tip_et + dt)
            d = hyperbolic_diagnostics(y, mu)
            timing_rows.append(
                dict(
                    offset_s=dt,
                    **differences(y, earth_state),
                    c3_km2_s2=d["c3_km2_s2"],
                    c3_change_km2_s2=d["c3_km2_s2"] - diag["c3_km2_s2"],
                )
            )
        baseline = dict(c3_km2_s2=34.0023, rla_deg=113.7071, dla_deg=28.0153)
        comparison = {
            name: dict(computed=diag[name], published=value, delta=diag[name] - value)
            for name, value in baseline.items()
        }
        comparison["nominal_c3"] = dict(
            target=34.0,
            tolerance=0.25,
            delta=diag["c3_km2_s2"] - 34.0,
            within_tolerance=bool(abs(diag["c3_km2_s2"] - 34) <= 0.25),
        )
        comparison["nominal_asymptote"] = dict(
            rla_target_deg=float(np.degrees(1.9845214)),
            dla_target_deg=float(np.degrees(0.4888475)),
            rla_tolerance_deg=float(np.degrees(0.0087266)),
            dla_tolerance_deg=float(np.degrees(0.0043633)),
        )
        checks = {
            "tip_covered": True,
            "selected_direct_earth_segment": True,
            "kernel_checksums": True,
            "tudat_raw_spice_agree": raw_errors["max_position_m"] < 1e-05
            and raw_errors["max_velocity_m_s"] < 1e-08,
            "origin_translation_closes": closure["position_m"] < 0.001
            and closure["velocity_m_s"] < 1e-07,
            "nominal_c3_delivery_requirement": comparison["nominal_c3"]["within_tolerance"],
            "published_c3_agreement_5e_minus5_km2_s2": abs(comparison["c3_km2_s2"]["delta"])
            < 5e-05,
            "published_angles_agreement_1e_minus4_deg": all(
                (abs(comparison[k]["delta"]) < 0.0001 for k in ("rla_deg", "dla_deg"))
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"TIP validation failed: {checks}")
        report = dict(
            schema_version=1,
            stage="01_tip",
            mission_event="TIP",
            status="validated_at_published_whole_second_epoch",
            extraction_backend="tudatpy.interface.spice",
            spacecraft_id=-255,
            origin_id=399,
            frame=FRAME,
            aberration_corrections=CORRECTION,
            state_units=["m", "m", "m", "m/s", "m/s", "m/s"],
            internal_time="ET/TDB seconds since 2000-01-01 12:00:00 TDB",
            epochs={k: epoch_record(t) for k, t in epochs.items()},
            timing_evidence=dict(
                launch_elapsed_seco2_s=3351,
                tip_after_seco2_s=900,
                schedule_minus_postflight_s=21,
                source_resolution_s=1,
                subsecond_uncertainty="not supplied; sensitivity cases are not confidence bounds",
                source=PAPER_URL,
                source_locators="p. 11; p. 12 Figure 11 and Table 1",
                superseded_schedule_source=SCHEDULE_URL,
            ),
            earth_relative_state_si=earth_state.tolist(),
            sun_relative_state_si=sun_state.tolist(),
            earth_relative_to_sun_state_si=earth_sun.tolist(),
            earth_gm_m3_s2=mu,
            earth_gm_source="gm_de440.tpc: BODY399_GM",
            diagnostics=diag,
            comparison=comparison,
            timing_sensitivity=timing_rows,
            validation=dict(
                checks=checks,
                raw_spice_comparison=raw_errors,
                origin_translation_error=closure,
                caveat="Same SPICE source: API consistency is not independent orbit truth.",
            ),
            kernel_load_order=[
                dict(
                    file=p.name,
                    path=str(p.resolve()),
                    url=spec.url,
                    sha256=sha256(p),
                    bytes=p.stat().st_size,
                )
                for p, spec in zip(paths, TIP_KERNELS)
            ],
            spacecraft_spk=spacecraft,
            planetary_spk=planets,
            selected_tip_segment=active,
            environment=dict(
                python=platform.python_version(),
                tudatpy=tudatpy.__version__,
                spiceypy=raw.__version__,
                numpy=np.__version__,
            ),
            limitations=[
                "No subsecond SECO-2 telemetry or SPK state covariance is provided.",
                "The launch SPK starts after SECO-2; cutoff cannot be detected from this SPK.",
                "C3 and asymptotes are Earth two-body osculating quantities; their time variation is not automatically numerical error.",
                "No Earth SOI crossing or optimisation was performed in this stage.",
            ],
        )
        paper = ROOT / "data/psyche_sources/IEPC_2024_126_launch_checkout.pdf"
        if paper.exists():
            report["timing_evidence"]["local_pdf"] = str(paper)
            report["timing_evidence"]["sha256"] = sha256(paper)
        report["implementation"] = [
            dict(file=str(p.relative_to(ROOT)), sha256=sha256(p))
            for p in sorted(Path(__file__).parent.glob("*.py"))
        ]
        c3 = np.einsum("ij,ij->i", states[:, 3:], states[:, 3:]) - 2 * mu / np.linalg.norm(
            states[:, :3], axis=1
        )
        arc_indices = np.searchsorted(query_times, arc_times)
        sensitivity_indices = np.searchsorted(query_times, tip_et + sensitivity_dt)
        np.savez_compressed(
            out / "tip_diagnostics.npz",
            query_et_s=query_times,
            state_si=states,
            c3_km2_s2=c3 / 1000000.0,
            arc_indices=arc_indices,
            sensitivity_indices=sensitivity_indices,
            sensitivity_dt_s=sensitivity_dt,
            tip_et_s=tip_et,
            tip_state_si=earth_state,
        )
        with (out / "tip_state.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "utc",
                    "et_tdb_s",
                    "target",
                    "observer",
                    "frame",
                    "aberration",
                    "x_m",
                    "y_m",
                    "z_m",
                    "vx_m_s",
                    "vy_m_s",
                    "vz_m_s",
                ]
            )
            writer.writerow(
                [report["epochs"]["tip"]["utc"], tip_et, -255, 399, FRAME, CORRECTION, *earth_state]
            )
        with (out / "timing_sensitivity.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(timing_rows[0]))
            writer.writeheader()
            writer.writerows(timing_rows)
        (out / "tip_result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        (out / "report.txt").write_text(format_report(report), encoding="utf-8")
        (out / "launch_spk_comments.txt").write_text(spacecraft["comments"], encoding="utf-8")
    return report


def format_report(r: dict) -> str:
    d, v = (r["diagnostics"], r["validation"])
    y = np.asarray(r["earth_relative_state_si"]) / 1000
    shifted = next((x for x in r["timing_sensitivity"] if x["offset_s"] == 21))
    lines = [
        "PSYCHE TIP RECONSTRUCTION",
        "",
        "Result and timing evidence",
        "--------------------------",
        "The post-flight mission-team report gives SECO-2 at T+55:51 (Figure 11).",
        "Together with its launch time 14:19:43 UTC, this gives:",
        "  SECO-2: 2023-10-13 15:15:34 UTC",
        "  TIP:    2023-10-13 15:30:34 UTC (15 minutes after SECO-2)",
        "The pre-launch T+56:12 schedule is 21 s later and is superseded here.",
        "Timing is published to whole seconds; no subsecond uncertainty is supplied.",
        f"TIP ET/TDB seconds since J2000: {r['epochs']['tip']['et_tdb_s']:.9f}",
        f"TIP TDB calendar: {r['epochs']['tip']['tdb_calendar']}",
        "Extra numerical digits do not imply matching physical timing accuracy.",
        "",
        "Earth-relative geometric state (spacecraft -255, observer 399)",
        "------------------------------------------------------------",
        "Orientation: J2000; aberration corrections: NONE.",
        "Position [km]: " + ", ".join((f"{x:.9f}" for x in y[:3])),
        "Velocity [km/s]: " + ", ".join((f"{x:.12f}" for x in y[3:])),
        "Full SI state and canonical Sun-relative state are stored in tip_result.json.",
        f"Earth GM: {r['earth_gm_m3_s2'] / 1000000000.0:.12f} km^3/s^2 (GM_DE440).",
        f"Geocentric distance: {d['radius_m'] / 1000:.9f} km",
        f"Relative speed: {d['speed_m_s'] / 1000:.12f} km/s",
        "",
        "Launch injection checks",
        "-----------------------",
        "C3 = v.v - 2*mu_E/||r||, with consistent units.",
        f"Computed C3: {d['c3_km2_s2']:.12f} km^2/s^2",
        f"Difference from nominal 34.0: {d['c3_km2_s2'] - 34:.12f} km^2/s^2",
        "Inside thesis delivery tolerance +/-0.25 km^2/s^2: yes.",
        f"Difference from published achieved 34.0023: {d['c3_km2_s2'] - 34.0023:.12g} km^2/s^2",
        f"Outgoing osculating v-infinity magnitude: {d['vinfinity_m_s'] / 1000:.12f} km/s",
        f"Outgoing RLA: {d['rla_deg']:.10f} deg; published achieved 113.7071 deg.",
        f"Outgoing DLA: {d['dla_deg']:.10f} deg; published achieved 28.0153 deg.",
        "These angles use the hyperbola's outgoing asymptote, not the instantaneous velocity direction.",
        "The comparison is with mission-team orbit-determination results (Table 1).",
        "",
        "Binary coverage and provenance",
        "------------------------------",
    ]
    for w in r["spacecraft_spk"]["coverage"]["-255"]:
        lines.append(f"Spacecraft coverage: {w['start']['utc']} through {w['end']['utc']}.")
    for i, seg in enumerate(r["spacecraft_spk"]["segments"], 1):
        lines.append(
            f"Segment {i}: target {seg['target']}, centre {seg['centre']}, {seg['frame']}, type {seg['spk_type']}; {seg['start']['utc']} through {seg['end']['utc']}."
        )
    lines += [
        "TIP uses the first, directly Earth-centred segment. Its Earth-relative state",
        "does not require subtracting heliocentric ephemerides. DE440s supplies the",
        "Earth/Sun centre chain for the saved Sun-relative state and closure check.",
        "All four kernel hashes are pinned and verified; load order is recorded.",
        "Only the specified launch reconstruction supplies spacecraft states.",
        "The kernel starts after SECO-2: it cannot independently time that cutoff.",
        "",
        "Numerical verification",
        "----------------------",
        f"Tudat vs raw-SPICE check: {v['raw_spice_comparison']['samples']} sample epochs.",
        f"Maximum state-query differences: {v['raw_spice_comparison']['max_position_m']:.3g} m, {v['raw_spice_comparison']['max_velocity_m_s']:.3g} m/s.",
        f"Sun/Earth origin-translation closure: {v['origin_translation_error']['position_m']:.6g} m, {v['origin_translation_error']['velocity_m_s']:.6g} m/s.",
        "These validate API/unit/frame consistency against the same underlying SPICE data;",
        "they are not an independent estimate of physical trajectory error.",
        "",
        "Timing sensitivity",
        "------------------",
        f"Using the old +21 s scheduled TIP shifts position by {shifted['position_m'] / 1000:.9f} km",
        f"and velocity by {shifted['velocity_m_s']:.9f} m/s; C3 changes by {shifted['c3_change_km2_s2']:.12g} km^2/s^2.",
        "Sensitivity cases at +/-0.5, +/-1, +/-21 and +/-60 s are supplied in CSV.",
        "They are deterministic perturbations, not inferred timestamp confidence bounds.",
        "Do not infer an exact epoch by forcing C3 to equal 34.0 or the rounded 34.0023.",
        "",
        "Limitations and scope",
        "---------------------",
        *r["limitations"],
        "The two-body diagnostics vary along a perturbed reconstructed arc; that variation",
        "does not establish thrust, and no engine-command history is inferred.",
        "Mass is not contained in this SPK and has not been reconstructed.",
        "The optimisation solver is unchanged. This stage stops at TIP extraction.",
        "",
        "Reproduce from the workspace root",
        "---------------------------------",
        "& 'C:\\Users\\lukas\\anaconda3\\envs\\det_traj_opt_env\\python.exe' -m psyche_mission_events_reconstruction.tip",
        "Add --download only if pinned kernels are missing.",
        "See psyche_mission_events_reconstruction/README.md for current plotting commands.",
        "Use system Python for plotting; this avoids the conda plotting DLL problem observed in stage 00.",
        "",
        "Sources",
        "-------",
        PAPER_URL + " (Oh et al., IEPC-2024-126, pp. 11-12)",
        SCHEDULE_URL + " (superseded pre-launch timing)",
        "Thesis: printed pp. 12, 15-16; parameters retained from stage00_review.",
        *[x["url"] for x in r["kernel_load_order"]],
        "https://py.api.tudat.space/en/latest/interface/spice.html",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    result = run(args.download)
    print(f"TIP: {result['epochs']['tip']['utc']}")
    print(f"C3: {result['diagnostics']['c3_km2_s2']:.12f} km^2/s^2")
    print(f"All {len(result['validation']['checks'])} validation checks passed.")
    print(f"Outputs: {OUT}")
