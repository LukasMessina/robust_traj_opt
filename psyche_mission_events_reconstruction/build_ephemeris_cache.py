"""Precompute Earth/Mars ephemerides, validate, and record the free-exit policy.

Run in det_traj_opt_env. On Windows, activate that environment (or prepend its
Library/bin to PATH) before importing tudatpy.dynamics. No OCP is solved here.
"""

from __future__ import annotations
from psyche_mission_events_reconstruction.paths import DATA_ROOT, resolve_input
import argparse
import json
from pathlib import Path
import time
import numpy as np
import spiceypy as raw
from tudatpy.interface import spice
from .ephemeris_cache import DAY, EphemerisCache
from .kernels import (
    GENERIC,
    MISSION,
    KERNEL_DIR,
    ROOT,
    TIP_KERNELS,
    Kernel,
    epoch_record,
    loaded_kernels,
    selected_segment,
    sha256,
    state_si,
)

OUT = DATA_ROOT / "ephemerides"
PLANET_KERNELS = (
    TIP_KERNELS[0],
    TIP_KERNELS[1],
    Kernel(
        "mar099s.bsp",
        GENERIC + "spk/satellites/mar099s.bsp",
        "997dc93ba640e476da7a494d2237dcdeb145e528db37be8ccee588c615e4e1ff",
    ),
    TIP_KERNELS[2],
)
MAR097_KERNEL = Kernel(
    "mar097.bsp",
    MISSION + "mar097.bsp",
    "201fc87075a7aaefc8b4d7f7545881079f906414fae38c5e36b1786f738dca63",
)
BODIES = {399: "Earth", 499: "Mars"}
POSITION_TOL_M = 0.1
VELOCITY_TOL_M_S = 1e-05
KINEMATIC_TOL_M_S = 0.0001


def policy():
    """Current optimisation policy; coast histories supply timing guesses only."""
    return dict(
        frame="J2000",
        units="m, m/s, kg, N",
        time_scale="ET/TDB seconds since J2000",
        gravity=["Earth", "Sun", "Mars", "Sun"],
        fixed_events=["TIP", "Mars SOI entry", "asteroid rendezvous"],
        free_events=["Earth SOI exit", "Mars SOI exit"],
        soi_convention="fixed_semimajor_axis",
        arrival_epoch_tdb_s=933552000.0,
        arrival_target=2000016,
        initial_guess="Fresh 1e-4 N/component RK7 rollout, two substeps per initial control arc",
        coast_timing_role="Only Earth/Mars coast exit durations initialise the free exit times",
        path_constraints="Central-body surface only; no SOI membership inequalities",
        mass_flow="Tmax * throttle; unit thrust direction",
        optimisation="Independent fuel/energy rollout -> same-objective HS -> multiple shooting with local-defect refinement",
    )


def verify_coverage(lo: float, hi: float, mars_kernel="mar097.bsp") -> list[dict]:
    records = []
    for filename, ids in (("de440s.bsp", (3, 399, 4, 10)), (mars_kernel, (499,))):
        for body in ids:
            window = raw.spkcov(str(KERNEL_DIR / filename), body)
            intervals = [raw.wnfetd(window, i) for i in range(raw.wncard(window))]
            if not any((a <= lo and b >= hi for a, b in intervals)):
                raise ValueError(f"No continuous padded coverage for {body} in {filename}")
            records.append(
                {
                    "file": filename,
                    "target": body,
                    "coverage_et_tdb_s": intervals,
                    "selected_at_middle": selected_segment(body, (lo + hi) / 2),
                }
            )
    return records


def make_cache(t0, tf, step, kernel_records) -> EphemerisCache:
    start = np.floor(t0 / step) * step - 24 * step
    stop = np.ceil(tf / step) * step + 24 * step
    epochs = start + np.arange(int(round((stop - start) / step)) + 1) * step
    mars_kernel = next(
        (
            record["name"]
            for record in kernel_records
            if record["name"] in ("mar097.bsp", "mar099s.bsp")
        )
    )
    coverage = verify_coverage(epochs[0], epochs[-1], mars_kernel)
    histories = np.array([[state_si(body, 10, epoch) for epoch in epochs] for body in BODIES])
    metadata = {
        "schema_version": 1,
        "body_ids": list(BODIES),
        "body_names": list(BODIES.values()),
        "frame": "J2000",
        "origin": "Sun",
        "aberration_corrections": "NONE",
        "time_scale": "ET/TDB seconds since J2000",
        "units": "m, m/s",
        "interpolation": "natural cubic spline, six state components",
        "sample_step_s": float(step),
        "padding_intervals_each_side": 24,
        "valid_interval_et_tdb_s": [t0, tf],
        "valid_interval_calendar": [epoch_record(t0), epoch_record(tf)],
        "kernel_load_order": kernel_records,
        "coverage": coverage,
        "position_tolerance_m": POSITION_TOL_M,
        "velocity_tolerance_m_s": VELOCITY_TOL_M_S,
        "kinematic_consistency_tolerance_m_s": KINEMATIC_TOL_M_S,
        "tolerance_provenance": "engineering acceptance criteria for this cache, not SPK physical accuracy",
    }
    return EphemerisCache(epochs, histories, metadata)


def evaluate_errors(cache: EphemerisCache):
    lo, hi = cache.valid_interval
    t = cache.sample_epochs
    times = np.r_[
        t[:-1] + 0.211324865405187 * np.diff(t),
        t[:-1] + 0.731 * np.diff(t),
        t,
        np.random.default_rng(255).uniform(lo, hi, 2048),
        lo,
        lo + 1,
        hi - 1,
        hi,
    ]
    times = np.unique(times[(times >= lo) & (times <= hi)])
    summaries, diagnostics = ({}, {"epochs": times})
    all_passed = True
    for body, name in BODIES.items():
        truth = np.array([state_si(body, 10, t) for t in times])
        estimated = cache.state(body, times)
        error = estimated - truth
        position = np.linalg.norm(error[:, :3], axis=1)
        velocity = np.linalg.norm(error[:, 3:], axis=1)
        kinematic = np.linalg.norm(cache.derivative(body, times)[:, :3] - estimated[:, 3:], axis=1)
        summary = {
            "count": times.size,
            "max_position_error_m": float(position.max()),
            "rms_position_error_m": float(np.sqrt(np.mean(position**2))),
            "max_velocity_error_m_s": float(velocity.max()),
            "rms_velocity_error_m_s": float(np.sqrt(np.mean(velocity**2))),
            "max_dposition_dt_minus_velocity_m_s": float(kinematic.max()),
        }
        summary["passed"] = bool(
            position.max() <= POSITION_TOL_M
            and velocity.max() <= VELOCITY_TOL_M_S
            and (kinematic.max() <= KINEMATIC_TOL_M_S)
        )
        all_passed &= summary["passed"]
        summaries[name] = summary
        diagnostics[f"{name}_position_error_m"] = position
        diagnostics[f"{name}_velocity_error_m_s"] = velocity
        diagnostics[f"{name}_kinematic_error_m_s"] = kinematic
    return (all_passed, summaries, diagnostics)


def check_backends(cache: EphemerisCache) -> dict:
    """Cross-check native Tudat, CasADi values/derivatives, and round trips."""
    from tudatpy.dynamics import environment_setup
    from tudatpy.math import interpolators

    lo, hi = cache.valid_interval
    epochs = np.unique(np.r_[np.linspace(lo, hi, 1001), lo + 1, hi - 1])
    summary = {}
    for body, name in BODIES.items():
        opts = interpolators.cubic_spline_interpolation(
            boundary_interpolation=interpolators.throw_exception_at_boundary
        )
        settings = environment_setup.ephemeris.interpolated_spice(
            cache.sample_epochs[0],
            cache.sample_epochs[-1],
            cache.metadata["sample_step_s"],
            "Sun",
            "J2000",
            opts,
            name,
        )
        tudat_ephem = environment_setup.create_body_ephemeris(settings, name)
        native = np.array(
            [np.asarray(tudat_ephem.cartesian_state(float(t))).ravel() for t in epochs]
        )
        fun, rate, second = cache.casadi_functions(body)
        symbolic = np.asarray(fun.map(epochs.size)(epochs)).T
        numerical = cache.state(body, epochs)
        d1 = np.asarray(rate.map(epochs.size)(epochs)).T
        d2 = np.asarray(second.map(epochs.size)(epochs)).T
        error_native = np.linalg.norm((native - numerical)[:, :3], axis=1).max()
        error_symbolic = np.linalg.norm((symbolic - numerical)[:, :3], axis=1).max()
        velocity_native = np.linalg.norm((native - numerical)[:, 3:], axis=1).max()
        velocity_symbolic = np.linalg.norm((symbolic - numerical)[:, 3:], axis=1).max()
        error_rate = np.max(np.abs(d1 - cache.derivative(body, epochs)))
        error_second = np.max(np.abs(d2 - cache.derivative(body, epochs, 2)))
        interior = epochs[(epochs > lo + 60) & (epochs < hi - 60)]
        fd = (cache.state(body, interior + 60) - cache.state(body, interior - 60)) / 120
        fd_error = np.max(np.abs(np.asarray(rate.map(interior.size)(interior)).T - fd))
        if (
            error_native > 0.002
            or error_symbolic > 0.002
            or velocity_native > 1e-08
            or (velocity_symbolic > 1e-08)
            or (error_rate > 1e-06)
            or (error_second > 1e-08)
            or (fd_error > 0.0001)
        ):
            raise AssertionError(f"Backend/derivative discrepancy for {name}")
        summary[name] = {
            "comparison_epochs": epochs.size,
            "max_native_tudat_position_difference_m": float(error_native),
            "max_casadi_position_difference_m": float(error_symbolic),
            "max_native_tudat_velocity_difference_m_s": float(velocity_native),
            "max_casadi_velocity_difference_m_s": float(velocity_symbolic),
            "max_casadi_first_derivative_component_difference": float(error_rate),
            "max_casadi_second_derivative_component_difference": float(error_second),
            "max_central_difference_first_derivative_component_difference": float(fd_error),
        }
    return summary


def run(output: Path = OUT, mars_kernel="mar097.bsp"):
    import casadi
    import scipy
    import tudatpy

    output.mkdir(parents=True, exist_ok=True)
    if mars_kernel not in ("mar099s.bsp", "mar097.bsp"):
        raise ValueError("Unsupported Mars kernel")
    specs = (
        PLANET_KERNELS
        if mars_kernel == "mar099s.bsp"
        else (PLANET_KERNELS[0], PLANET_KERNELS[1], MAR097_KERNEL, PLANET_KERNELS[-1])
    )
    paths = [KERNEL_DIR / spec.name for spec in specs]
    for spec, path in zip(specs, paths):
        if not path.is_file() or sha256(path) != spec.sha256:
            raise ValueError(f"Missing or unverified kernel: {path}")
    kernel_records = [dict(name=s.name, url=s.url, sha256=s.sha256) for s in specs]
    start = time.perf_counter()
    attempts = []
    with loaded_kernels(paths):
        tip_path = DATA_ROOT / "tip/tip_result.json"
        tip_document = json.loads(tip_path.read_text(encoding="utf-8"))
        t0 = float(spice.convert_date_string_to_ephemeris_time("2023-10-13 15:30:34 UTC"))
        if abs(tip_document["epochs"]["tip"]["et_tdb_s"] - t0) > 1e-06:
            raise ValueError("Stored TIP epoch disagrees with verified UTC-to-TDB conversion")
        from .arrival_boundary import ARRIVAL_TDB_INPUT

        tf = 933552069.1832606  # Frozen cache validity endpoint; arrival itself is 933552000 TDB.
        for step in (21600.0, 10800.0, 5400.0, 2700.0):
            print(f"Building and validating {step / 3600:g} h table...", flush=True)
            cache = make_cache(t0, tf, step, kernel_records)
            accepted, summaries, diagnostics = evaluate_errors(cache)
            attempts.append({"step_s": step, "bodies": summaries, "passed": bool(accepted)})
            print(json.dumps(attempts[-1]), flush=True)
            if accepted:
                break
        else:
            raise RuntimeError("No interpolation grid met the stated tolerances")
        backend_checks = check_backends(cache)
        cache.metadata["validation"] = summaries
        cache.metadata["software"] = {
            "tudatpy": tudatpy.__version__,
            "casadi": casadi.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        }
        cache.metadata["tip_result_sha256"] = sha256(tip_path)
        cache.metadata["mu_m3_s2"] = {
            str(body): float(spice.get_body_gravitational_parameter(str(body)))
            for body in (10, 399, 499)
        }
        numeric_before_clear = cache.state(399, (t0 + tf) / 2)
    if raw.ktotal("ALL") or spice.get_total_count_of_kernels_loaded():
        raise AssertionError("Kernel pool cleanup failed")
    cache_path = output / "earth_mars_ephemerides.npz"
    cache.save(cache_path)
    restored = EphemerisCache.load(cache_path)
    np.testing.assert_array_equal(restored.state(399, (t0 + tf) / 2), numeric_before_clear)
    for body in BODIES:
        f = restored.casadi_functions(body)[0]
        np.testing.assert_allclose(
            np.asarray(f((t0 + tf) / 2)).ravel(),
            restored.state(body, (t0 + tf) / 2),
            rtol=0,
            atol=0.002,
        )
        f.expand()((t0 + tf) / 2)
        restored.tudat_interpolator(body).interpolate((t0 + tf) / 2)
    elapsed = time.perf_counter() - start
    report = {
        "status": "planetary interpolation validated; no trajectory optimisation performed",
        "attempts": attempts,
        "accepted_grid": cache.metadata,
        "backend_checks": backend_checks,
        "kernel_free_evaluation_verified": True,
        "cache_sha256": sha256(cache_path),
        "cache_bytes": cache_path.stat().st_size,
        "build_and_validation_wall_seconds": elapsed,
        "implementation_sha256": {
            p.name: sha256(p)
            for p in (Path(__file__), Path(__file__).with_name("ephemeris_cache.py"))
        },
        "sources": [
            "https://py.api.tudat.space/en/latest/dynamics/environment_setup/ephemeris.html#tudatpy.dynamics.environment_setup.ephemeris.interpolated_spice",
            "https://py.api.tudat.space/en/latest/math/interpolators.html",
            "https://web.casadi.org/docs/#using-lookup-tables",
        ],
    }
    (output / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "problem_definition.json").write_text(
        json.dumps(policy(), indent=2), encoding="utf-8"
    )
    np.savez_compressed(output / "validation_diagnostics.npz", **diagnostics)
    write_report(output, report)
    print(f"Saved validated cache and report to {output}; {elapsed:.1f} s", flush=True)


def write_report(output, report):
    metadata = report["accepted_grid"]
    lines = [
        "VALIDATED PLANETARY EPHEMERIS CACHE",
        "",
        "Earth 399 and Mars 499 relative to Sun 10; J2000; geometric NONE; m and m/s.",
        "ET/TDB seconds since J2000. Natural cubic state-component interpolation.",
        "No SPICE calls are needed when these stored interpolants are used in optimisation.",
        "Validation measures interpolation error against the selected kernels, not orbit uncertainty.",
        "Current mission policy:",
        json.dumps(policy(), indent=2),
        "",
        "Grid and measured interpolation errors:",
        json.dumps(metadata, indent=2),
        "",
        "Tudat/CasADi cross-checks:",
        json.dumps(report["backend_checks"], indent=2),
        "",
        "The six components are interpolated independently; fitted velocity is not exactly d(position)/dt.",
        "See psyche_mission_events_reconstruction/README.md for fixed boundaries and assumptions.",
        "Rebuild: python -m psyche_mission_events_reconstruction --output output/mission_inputs_new --rebuild-cache",
        "",
        "Sources:",
        *report["sources"],
    ]
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--mars-kernel", choices=("mar099s.bsp", "mar097.bsp"), default="mar097.bsp"
    )
    args = parser.parse_args()
    run(args.output, args.mars_kernel)
