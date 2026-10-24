"""Psyche: independent rollout -> same-objective HS -> adaptive multiple shooting."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .psyche import Mission, Settings, Transcription, save_solution
from .psyche_validation import interval_defects, refine_meshes
from .diagnostics import convergence_policy, local_defects_met, mesh_accepted, postsolve_diagnostics
from .io import write_json, snapshot_sources
from .derivatives import (
    derivative_checks,
    exit_time_derivative_checks,
    integration_checks,
    reintegrate_sigma_model,
    seed_solution,
    control_derivative_checks,
)


def run(output, objective="fuel", settings=None, substeps=20):
    if objective not in ("fuel", "energy"):
        raise ValueError("Choose fuel or energy")
    started = time.perf_counter()
    settings = settings or Settings()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    mission = Mission(settings)
    provenance = mission.provenance()
    provenance["implementation_sha256"] = snapshot_sources(output)
    np.savez_compressed(output / "initial_guess.npz", **mission.seed)
    mass_flow_model = "throttle_direction"
    initialization = "hermite_simpson"
    hs_barrier_strategy = barrier_strategy = "adaptive"
    defect_substeps = settings.defect_substeps_per_interval
    config = dict(
        settings=asdict(settings),
        provenance=provenance,
        event_epochs={"TIP_ET": mission.t0, "Mars_entry_ET": mission.tm, "arrival_ET": mission.tf},
        arrival_target_sun_state_si=mission.xf,
        arrival_target_body_id=2000016,
        objective=objective,
        objective_definition=(
            "initial mass minus final mass" if objective == "fuel" else "0.5 integral dot(T,T) dt"
        ),
        objective_scaling=(
            "divide mass loss by wet mass"
            if objective == "fuel"
            else "divide energy by Tmax^2 times fixed total duration"
        ),
        initial_guess=str(output / "initial_guess.npz"),
        seed_exit_days=mission.seed_exit_days,
        ephemeris_policy="Precomputed differentiable splines; no SPICE calls during optimisation",
        gravity_models=["Earth only", "Sun only", "Mars only", "Sun only"],
        warm_start="Fresh small-thrust rollout -> same-objective HS -> multiple shooting with local-defect refinement",
        initialization=initialization,
        mass_flow_model=mass_flow_model,
        hs_barrier_strategy="adaptive",
        barrier_strategy="adaptive",
        mass_flow_definition="T=Tmax*alpha*d; dot(d,d)=1; mdot=-Tmax*alpha/(Isp*g0)",
        convergence_policy=convergence_policy(settings),
        solver_time_limit=None,
        postsolve_policy="Continuous propagation is report-only; no additional acceptance or refinement",
        internal_surface_fractions="Every internal uniform RK7 step endpoint; no added integration nodes",
        checkpoint_policy="Primal and multipliers every 10 iterations, archival copies every 100",
    )
    # The requested grid is fixed. Report the accuracy comparison without
    # silently increasing the number of integration steps.
    check = integration_checks(mission, seed_solution(mission), substeps)
    initial_checks = [check]
    print(
        f"Pre-solve uniform RK7({substeps}) vs DOP853: {check['maximum_relative_integration_error']:.6g}",
        flush=True,
    )
    derivatives = derivative_checks(mission, substeps)
    write_json(
        output / "pre_solve_checks.json",
        {"integration": initial_checks, "derivatives": derivatives},
    )
    if (
        derivatives["maximum_jacobian_error"] > 2e-6
        or derivatives["maximum_hessian_vector_error"] > 2e-5
    ):
        raise RuntimeError("Shooting flow derivative validation failed")
    if mass_flow_model == "throttle_direction":
        controls = control_derivative_checks(mission, seed_solution(mission), substeps)
        write_json(output / "control_derivative_checks.json", controls)
        if (
            controls["maximum_jacobian_error"] > 2e-6
            or controls["maximum_hessian_vector_error"] > 2e-5
        ):
            raise RuntimeError("Throttle/direction derivative validation failed")
        config.update(
            direction_at_zero_throttle="Free unit vector; no frozen coast schedule",
            thrust_floor=None,
            smoothing_used=False,
            hs_handoff_rule="HS NLP success required; transfer states, throttle, direction and exit times on identical mesh",
        )
    config.update(
        shooting_substeps=substeps,
        integration_grid="uniform",
        integration_validation_reference=1e-11,
        integration_validation_role="Report only; requested subdivision count is never automatically changed",
        additional_integration_splits=0,
    )
    write_json(output / "configuration.json", config)
    history, meshes, guess, defects = [], mission.initial_meshes(), None, None
    termination, latest_path = "not solved", None

    def attempt(meshes, incoming, name, barrier, method="shooting"):
        build_start = time.perf_counter()
        problem = Transcription(mission, meshes, method, incoming, objective, barrier, substeps)
        build_seconds = time.perf_counter() - build_start
        initial = np.asarray(
            problem.opti.debug.value(problem.opti.x, problem.opti.initial())
        ).ravel()
        if not np.all(np.isfinite(initial)):
            raise RuntimeError("Nonfinite shooting initial vector")
        initial_digest = hashlib.sha256(initial.tobytes()).hexdigest()
        write_json(
            output / f"{name}_initialization.json",
            {
                "barrier": barrier,
                "initial_vector_sha256": initial_digest,
                "ipopt_options": problem.ipopt_options.copy(),
                "initial_source": (
                    "dynamics rollout"
                    if incoming is None
                    else (
                        "converged Hermite-Simpson solution"
                        if incoming["method"] == "hs"
                        else "preceding mesh solution"
                    )
                ),
            },
        )
        if method == "shooting" and incoming is not None and incoming["method"] == "hs":
            write_json(output / "hs_to_shooting_handoff.json", verify_hs_handoff(problem, incoming))
        if not any(h["method"] == method for h in history):
            time_checks = exit_time_derivative_checks(problem)
            write_json(output / f"{method}_exit_time_derivatives.json", time_checks)
            if (
                max(
                    time_checks["earth_exit_duration_error"],
                    time_checks["mars_exit_duration_error"],
                )
                > 2e-6
            ):
                raise RuntimeError("Free-exit derivative validation failed")
        print(
            f"{name}: method={method}; objective={objective}; {barrier}; arcs={[len(m)-1 for m in meshes]}; variables={problem.opti.nx}; constraints={problem.opti.ng}",
            flush=True,
        )
        candidate = problem.solve(checkpoint_directory=output / "checkpoints" / name)
        candidate.update(provenance=provenance, settings=asdict(settings))
        save_solution(candidate, output / f"{name}.npz")
        summary = {
            key: candidate[key]
            for key in (
                "solver_status",
                "solver_success",
                "fuel_consumed_kg",
                "objective_value",
                "objective_units",
                "max_scaled_constraint_violation",
                "iterations",
                "wall_seconds",
                "variables",
                "constraints",
            )
        }
        summary.update(
            name=name,
            method=method,
            objective=objective,
            barrier=barrier,
            intervals=[len(m) - 1 for m in meshes],
            assembly_seconds=build_seconds,
            initial_vector_sha256=initial_digest,
        )
        history.append(summary)
        write_json(output / "history.json", history)
        print(json.dumps(summary), flush=True)
        return candidate

    hs_converged, shooting_started = None, False
    if initialization == "hermite_simpson":
        name = f"hs_{objective}"
        guess = attempt(meshes, None, name, "adaptive", "hs")
        latest_path = output / f"{name}.npz"
        hs_converged = bool(guess["solver_success"])
        if hs_converged:
            save_solution(guess, output / "converged_hs_solution.npz")
        else:
            termination = "Hermite-Simpson did not converge; multiple shooting was not started"
            print(termination, flush=True)

    for iteration in range(settings.max_refinements + 1 if hs_converged is not False else 0):
        shooting_started = True
        name = f"shooting_{iteration:02d}"
        candidate = attempt(meshes, guess, name, "adaptive")
        latest_path = output / f"{name}.npz"
        guess, defects = candidate, None
        if not candidate["solver_success"]:
            termination = "Shooting solver did not converge"
            break
        try:
            defects = interval_defects(mission, candidate)
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
            termination = "local defect calculation failed: " + str(exc)
            break
        write_json(
            latest_path.with_name(latest_path.stem + "_defects.json"), {"interval_defects": defects}
        )
        print(
            f"Maximum relative local defect: {max(d['max_relative'] for d in defects):.6g}",
            flush=True,
        )
        if local_defects_met(defects):
            termination = "local defect tolerance reached"
            break
        if iteration == settings.max_refinements:
            termination = "mesh-refinement iteration limit reached"
            break
        meshes, count = refine_meshes(candidate, defects, mission)
        if not count:
            termination = "mesh interval limit reached"
            break
    converged = bool(shooting_started and mesh_accepted(settings, guess, defects))
    # The acceptance decision above is final; these reports cannot trigger refinement.
    if defects is None:
        try:
            defects = interval_defects(mission, guess)
            write_json(
                latest_path.with_name(latest_path.stem + "_defects.json"),
                {"role": "post-failure diagnostic only", "interval_defects": defects},
            )
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
            write_json(output / "local_defect_error.json", {"error": str(exc)})
    diagnostics = postsolve_diagnostics(mission, guess, output, latest_path.stem)
    for name, callback in (
        ("sigma_model_reintegration", lambda: reintegrate_sigma_model(mission, guess)),
        ("final_integration_checks", lambda: integration_checks(mission, guess, substeps)),
    ):
        try:
            record = callback()
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
            record = {"status": "diagnostic unavailable", "error": str(exc)}
        write_json(output / f"{name}.json", record)
    result = {
        "status": (
            f"shooting {objective} converged: {convergence_policy(settings)}"
            if converged
            else f"shooting {objective} incomplete: " + termination
        ),
        "convergence_policy": convergence_policy(settings),
        "converged": converged,
        "objective": objective,
        "termination_reason": termination,
        "initialization": initialization,
        "hs_converged": hs_converged,
        "shooting_started": shooting_started,
        "local_defect_tolerance_met": local_defects_met(defects),
        "last_solver_status": guess["solver_status"],
        "last_candidate": str(latest_path),
        "candidate_fuel_consumed_kg": guess["fuel_consumed_kg"],
        "candidate_final_mass_kg": settings.wet_mass_kg - guess["fuel_consumed_kg"],
        "candidate_objective_value": guess["objective_value"],
        "candidate_objective_units": guess["objective_units"],
        "history": history,
        "total_benchmark_wall_seconds": time.perf_counter() - started,
        "limitations": [
            "Throttle and unit direction give physical mass flow; free coast directions can be poorly conditioned.",
            "Convergence does not prove global optimality or independent trajectory accuracy.",
            f"Local {defect_substeps}-substep RK7 checks and sampled surface constraints do not guarantee global propagation accuracy or all-times surface avoidance.",
            config["warm_start"]
            + ". HS convergence is NLP convergence, not independent propagation accuracy.",
            "Historical timings are single runs, not a controlled statistical performance benchmark.",
        ],
    }
    write_json(output / "result.json", result)
    if converged:
        save_solution(guess, output / "converged_solution.npz")
    from .diagnostics import reference_sensitivity_diagnostic

    reference_sensitivity_diagnostic(output)
    return result


def verify_hs_handoff(problem, incoming):
    """Reject failed/wrong-objective HS seeds and verify the actual NLP seed."""
    if incoming["method"] != "hs" or not incoming["solver_success"]:
        raise ValueError("Multiple shooting requires a converged Hermite-Simpson seed")
    if incoming["objective"] != problem.objective:
        raise ValueError("HS and shooting must use the same objective")
    op, mission = problem.opti, problem.mission
    differences = []
    for i, (mesh, source, (x, u, sigma, _), scale) in enumerate(
        zip(problem.meshes, incoming["phases"], problem.variables, mission.scales)
    ):
        np.testing.assert_array_equal(mesh, source["mesh"])
        expected = (
            source["states_si"].T / scale.state[:, None],
            source["thrust_n"].T / mission.settings.thrust_max_n,
            source["sigma_n"].reshape(1, -1) / mission.settings.thrust_max_n,
        )
        errors = []
        for expr, reference in zip((x, u, sigma), expected):
            actual = np.asarray(op.debug.value(expr, op.initial())).reshape(reference.shape)
            np.testing.assert_allclose(actual, reference, rtol=2e-14, atol=2e-14)
            errors.append(float(np.max(abs(actual - reference))))
        differences.append(
            {
                "max_state_difference_nd": errors[0],
                "max_thrust_difference_nd": errors[1],
                "max_sigma_difference_nd": errors[2],
            }
        )
        if mission.settings.mass_flow_model == "throttle_direction":
            for expr, reference in (
                (problem.direction_variables[i], source["direction"].T),
                (sigma, source["throttle"].reshape(1, -1)),
            ):
                actual = np.asarray(op.debug.value(expr, op.initial())).reshape(reference.shape)
                np.testing.assert_array_equal(actual, reference)
            differences[-1]["throttle_and_direction_identical"] = True
    np.testing.assert_array_equal(
        np.asarray(op.debug.value(problem.exit_days, op.initial())).ravel(), incoming["exit_days"]
    )
    return {
        "verified": True,
        "source_method": "hs",
        "source_solver_status": incoming["solver_status"],
        "objective": problem.objective,
        "exit_times_identical": True,
        "phases": differences,
        "note": "The integrated shooting defects may initially differ from HS collocation defects.",
    }
