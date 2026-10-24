"""Consistent physical-unit summaries for all five transfer cases."""

import json
from pathlib import Path
from .io import write_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def case_record(directory):
    result, config = read(directory / "result.json"), read(directory / "configuration.json")
    if "event_epochs" in config:
        from .psyche import load_solution

        solution = load_solution(directory / Path(result["last_candidate"]).name)
        defect_path = directory / (Path(result["last_candidate"]).stem + "_defects.json")
        defects = read(defect_path)["interval_defects"] if defect_path.exists() else []
        reference_path = directory / "global_reference_accuracy.json"
        reference = (
            read(reference_path)["mass_models"]["physical_thrust_norm"]
            if reference_path.exists()
            else None
        )
        record = dict(
            case="psyche",
            objective=config["objective"],
            arcs=result["history"][-1]["intervals"],
            initial_arcs=config["settings"]["initial_intervals_per_phase"],
            nlp_converged=result["shooting_started"] and result["history"][-1]["solver_success"],
            fuel_kg=solution["fuel_consumed_kg"],
            energy_n2_s=solution["energy_cost_n2_s"],
            solver_status=result["last_solver_status"],
            max_constraint_violation=solution["max_scaled_constraint_violation"],
            max_relative_local_defect=max((d["max_relative"] for d in defects), default=None),
            defect_tolerance=config["settings"]["defect_tolerance"],
            continuous_final_errors=(
                reference["tightest"][-1]["end_mismatch"] if reference else None
            ),
            continuous_reference_difference=(
                reference["tighter"][-1]["difference_from_tightest"] if reference else None
            ),
            solver_seconds=sum(s["wall_seconds"] for s in result["history"]),
            shooting_substeps=config["shooting_substeps"],
        )
    else:
        error = result.get("continuous_physical_thrust_errors", {})
        reference = result.get("continuous_reference_difference", {})
        record = dict(
            case=config["case"],
            objective=config["objective"],
            arcs=result["arcs"],
            initial_arcs=config["initial_intervals"],
            nlp_converged=result["nlp_converged"],
            fuel_kg=result["fuel_kg"],
            energy_n2_s=result["energy_n2_s"],
            solver_status=result["solver_status"],
            max_constraint_violation=result["max_constraint_violation"],
            max_relative_local_defect=result.get("max_relative_local_defect"),
            defect_tolerance=config["defect_tolerance"],
            continuous_final_errors={
                key: error.get("final_" + key) for key in ("position_m", "velocity_m_s", "mass_kg")
            },
            continuous_reference_difference={
                key: reference.get("final_" + key)
                for key in ("position_m", "velocity_m_s", "mass_kg")
            },
            solver_seconds=result["combined_solver_seconds"],
            shooting_substeps=config["shooting_substeps"],
        )
    record.update(
        converged=result["converged"],
        local_defect_tolerance_met=result["local_defect_tolerance_met"],
        termination_reason=result["termination_reason"],
        directory=str(directory.resolve()),
        iterations=[s["iterations"] for s in result["history"]],
    )
    write_json(directory / "run_summary.json", record)
    return record


def report_suite(output):
    output = Path(output)
    records = [case_record(path.parent) for path in sorted(output.glob("*/result.json"))]
    write_json(
        output / "combined_results.json",
        dict(energy_definition="0.5 integral dot(T,T) dt", records=records),
    )
    lines = [
        "DETERMINISTIC OPTIMISATION: HS -> MULTIPLE SHOOTING",
        "Independent fresh rollouts; throttle and unit direction; adaptive barriers.",
        "Convergence requires NLP success and local-defect tolerance. Continuous propagation is report-only.",
        "Energy = 0.5 integral squared physical thrust dt [N^2 s]. Fuel = initial minus final mass [kg].",
        "Reference-integrator differences measure sensitivity, not rigorous accuracy bounds.",
        "CR3BP Moon surfaces are constrained at mesh boundaries only; sampled clearance is not an all-times certificate.",
        "",
        "case | objective | converged | initial -> final arcs | fuel kg | energy N^2 s | max local defect | solver seconds",
    ]
    for r in records:
        lines.append(
            f"{r['case']} | {r['objective']} | {r['converged']} | {r['initial_arcs']} -> {r['arcs']} | "
            f"{r['fuel_kg']:.12g} | {r['energy_n2_s']:.12g} | {r['max_relative_local_defect']} | {r['solver_seconds']:.3f}"
        )
    lines += [
        "",
        "Continuous physical-thrust propagation: final position [m], velocity [m/s], reference position difference [m]",
    ]
    for r in records:
        e, d = r["continuous_final_errors"] or {}, r["continuous_reference_difference"] or {}
        lines.append(
            f"{r['case']} {r['objective']}: {e.get('position_m')}; {e.get('velocity_m_s')}; {d.get('position_m')}"
        )
    lines += [
        "",
        "Completed cases: " + str(sum(r["converged"] for r in records)) + "/" + str(len(records)),
        "Plots: each case directory contains trajectory_*.png, using plotter.py.",
        "Unconverged candidate objective values are not accepted optimal trajectories.",
    ]
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return records
