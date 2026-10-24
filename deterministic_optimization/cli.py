"""Unified command line for the five deterministic transfers."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import traceback

from .io import write_json
from .models import DETERMINISTIC_CASE_REGISTRY, CASE_REGISTRY
from .settings import ipopt_options
from .worker_logging import redirect_worker_output

ROOT = Path(__file__).resolve().parents[1]
CASES = (*DETERMINISTIC_CASE_REGISTRY, "psyche")


def run_one(case, objective, output, print_level, public_root=None):
    """Independent worker: no preceding objective or reference run is a seed."""
    output = Path(output).resolve()
    directory = output / f"{case}_{objective}"
    with (
        (output / f"{case}_{objective}.log").open("w", encoding="utf-8", buffering=1) as log,
        redirect_worker_output(log),
    ):
        try:
            if case == "psyche":
                from .psyche import Settings
                from .psyche_driver import run

                result = run(
                    directory,
                    objective=objective,
                    settings=replace(Settings(), print_level=print_level),
                )
            else:
                from .single_phase import run_case

                result = run_case(case, objective, directory, ipopt_options(print_level))
            if result["converged"]:
                from .exports import export_case

                export_case(directory, case, objective, public_root)
            return dict(
                case=case,
                objective=objective,
                directory=str(directory),
                converged=bool(result["converged"]),
                published=bool(result["converged"]),
            )
        except Exception:
            traceback.print_exc(file=log)
            # A post-solve plotting/export failure cannot change NLP/defect acceptance.
            saved_result = directory / "result.json"
            accepted = (
                bool(json.loads(saved_result.read_text(encoding="utf-8"))["converged"])
                if saved_result.exists()
                else False
            )
            return dict(
                case=case,
                objective=objective,
                directory=str(directory),
                converged=accepted,
                published=False,
                error=traceback.format_exc(),
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=(*CASES, "cr3bp", "all"), default="all")
    parser.add_argument("--objective", choices=("fuel", "energy", "both"), default="both")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "output",
        help="Root for the three case-family result folders",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Optional new directory for solver logs/checkpoints (default: timestamped .cache/deterministic_runs)",
    )
    parser.add_argument(
        "--export-from",
        type=Path,
        help="Export an existing detailed batch without rerunning optimisation",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Independent worker processes; each solver uses one BLAS thread",
    )
    parser.add_argument("--print-level", type=int, default=5)
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regenerate figures from the compact case-family results",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    os.chdir(ROOT)
    cases = (
        CASES
        if args.case == "all"
        else tuple(CASE_REGISTRY) if args.case == "cr3bp" else (args.case,)
    )
    objectives = ("fuel", "energy") if args.objective == "both" else (args.objective,)
    jobs = [(c, o) for c in cases for o in objectives]
    public_root = args.output.resolve()
    if args.plots_only:
        from .exports import regenerate_public

        for c, o in jobs:
            regenerate_public(c, o, public_root)
        return 0
    if args.export_from:
        from .exports import export_case

        for c, o in jobs:
            prefix = export_case(args.export_from / f"{c}_{o}", c, o, public_root)
            status_path = args.export_from / "suite_status.json"
            if status_path.exists():
                statuses = json.loads(status_path.read_text(encoding="utf-8"))
                for status in statuses:
                    if status["case"] == c and status["objective"] == o:
                        if "error" in status:
                            status["previous_export_error"] = status.pop("error")
                        status.update(converged=True, published=True, output_prefix=str(prefix))
                write_json(status_path, statuses)
            print(f"Exported {c} {o}", flush=True)
        return 0
    output = (
        args.work_dir
        or ROOT / ".cache/deterministic_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    ).resolve()
    if output.exists():
        parser.error("Work directory already exists; choose a new --work-dir")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    configuration = dict(
        workflow="fresh rollout -> same-objective HS -> adaptive multiple shooting",
        cases=list(cases),
        objectives=list(objectives),
        ipopt_options=ipopt_options(args.print_level),
        solver_time_limit=None,
        initial_intervals=50,
        psyche_initial_intervals=[60, 50, 90, 50],
        shooting_substeps={"cr3bp": 2, "earth_mars": 20, "psyche": 20},
        defect_substeps=64,
        defect_tolerances={"cr3bp": 5e-13, "earth_mars": 1e-13, "psyche": 5e-14},
        max_refinements=8,
        max_intervals=1800,
        energy_definition="0.5 integral dot(T,T) dt",
        mass_flow_model="throttle_direction",
        jobs=args.jobs,
        public_output_root=str(public_root),
    )
    write_json(output / "configuration.json", configuration)
    results = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_one, c, o, output, args.print_level, public_root): (c, o)
            for c, o in jobs
        }
        for future in as_completed(futures):
            c, o = futures[future]
            result = future.result()
            results.append(result)
            write_json(output / "suite_status.json", results)
            status = "completed" if result["converged"] else "incomplete"
            if result["converged"] and not result["published"]:
                status += "; export failed (saved solution retained)"
            print(f"{c} {o}: {status}", flush=True)
    from .reporting import report_suite

    report_suite(output)
    return 0 if all(r["converged"] and r["published"] for r in results) else 1
