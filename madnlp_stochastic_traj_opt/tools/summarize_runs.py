"""Tabulate the diagnostics of every run of an execution mode into output/<mode>/summary.csv.

    python madnlp_stochastic_traj_opt/tools/summarize_runs.py [cpu|gpu]    (default: both modes)
"""

import csv
import sys
import tomllib
from pathlib import Path

OUTPUT = Path(__file__).resolve().parents[1] / "output"
MODES = ("cpu", "gpu")
COLUMNS = (
    ("case", ("case",)),
    ("status", ("madnlp_status",)),
    ("iterations", ("iterations",)),
    ("restoration", ("restoration_iterations",)),
    ("longest_run", ("longest_restoration_run",)),
    ("objective", ("objective",)),
    ("max_violation", ("max_constraint_violation",)),
    ("dual_inf", ("final_kkt", "dual_infeasibility_unscaled")),
    ("complementarity", ("final_kkt", "complementarity_unscaled")),
    ("converged", ("converged",)),
    ("seconds", ("seconds", "solve")),
)


def lookup(data, keys):
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return ""
        data = data[key]
    return data


def summarize(mode):
    directory = OUTPUT / mode
    rows = [{name: lookup(tomllib.loads(path.read_text()), keys) for name, keys in COLUMNS}
            for path in sorted(directory.glob("*/*_diagnostics.toml"))]
    if not rows:
        print(f"{mode}: no runs in {directory}")
        return
    out = directory / "summary.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[name for name, _ in COLUMNS])
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        print(f"{mode} {r['case']:20s} {r['status']:28s} it={r['iterations']!s:>5} resto={r['restoration']!s:>3}/"
              f"{r['longest_run']!s:<2} obj={r['objective']:.8f} viol={r['max_violation']:.2e} "
              f"du={r['dual_inf']:.1e} co={r['complementarity']:.1e} converged={r['converged']}")
    print(f"wrote {out}")


def main():
    modes = sys.argv[1:] or MODES
    unknown = set(modes) - set(MODES)
    if unknown:
        sys.exit(f"unknown mode {', '.join(sorted(unknown))}; expected one of {', '.join(MODES)}")
    for mode in modes:
        summarize(mode)


if __name__ == "__main__":
    main()
