"""Validate saved MadNLP trajectories with independent JAX propagation and Monte Carlo."""

import argparse
import os
from pathlib import Path

from verification.backend import check, select

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "lyapunov_l1_to_l2",
    "halo_l2_to_halo_l1",
    "nrho_l2_to_dro",
    "earth_mars",
    "psyche",
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "gpu"))
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument(
        "--objective", choices=("standard", "chance_sum"), default="standard"
    )
    parser.add_argument(
        "--directory", type=Path, help="saved run directory; requires one case"
    )
    parser.add_argument("--samples", type=int, default=50000)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if args.samples < 2:
        parser.error("--samples must be at least 2 for sample covariance estimates")
    if args.directory and args.case == "all":
        parser.error("--directory requires a single --case")
    os.environ.setdefault("MPLBACKEND", "Agg")
    select(args.mode)
    print(f"Validation on JAX {check(args.mode)}", flush=True)
    output = Path(
        os.environ.get("MADNLP_STOCHASTIC_OUTPUT_ROOT", ROOT / "output/stochastic")
    )
    cases = CASES if args.case == "all" else (args.case,)
    suffix = "_chance_sum" if args.objective == "chance_sum" else ""
    for case in cases:
        directory = args.directory or output / args.mode / (case + suffix)
        if case == "psyche":
            from verification.stochastic.psyche_diagnostics import (
                validate_saved_solution,
            )

            validate_saved_solution(
                directory, plots=not args.no_plots, samples=args.samples
            )
        else:
            from verification.stochastic.single_phase import validate

            validate(directory / f"{case}.npz", args.samples, plots=not args.no_plots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
