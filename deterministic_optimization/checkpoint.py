"""Atomic primal/dual iterate snapshots and inexpensive progress records."""

import json
from pathlib import Path
import time

import numpy as np


class IterationCheckpoint:
    def __init__(self, problem, directory):
        self.problem = problem
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.started = time.perf_counter()
        self.error_reported = False
        (self.directory / "layout.json").write_text(
            json.dumps(
                {
                    "constraint_groups": problem.group_ranges,
                    "settings": vars(problem.mission.settings),
                    "meshes": [m.tolist() for m in problem.meshes],
                    "note": "Unconverged current IPOPT primal vector and constraint multipliers; not an accepted solution.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def __call__(self, iteration):
        if iteration % 10:
            return
        p, op = self.problem, self.problem.opti
        try:
            value = lambda expr: np.asarray(op.debug.value(expr), dtype=float)
            x, g, lam = value(op.x).ravel(), value(op.g).ravel(), value(op.lam_g).ravel()
            lo, hi = value(op.lbg).ravel(), value(op.ubg).ravel()
            violation = np.maximum(np.maximum(lo - g, g - hi), 0.0)
            record = {
                "iteration": int(iteration),
                "elapsed_s": time.perf_counter() - self.started,
                "objective": float(value(op.f)),
                "max_constraint_violation": float(violation.max()),
                "max_abs_constraint_multiplier": float(np.max(abs(lam))),
            }
            arrays = {
                "primal": x,
                "constraints": g,
                "constraint_multipliers": lam,
                "lower_bounds": lo,
                "upper_bounds": hi,
                "exit_days": value(p.exit_days).ravel(),
            }
            for i, (state, control, sigma, _) in enumerate(p.variables):
                arrays.update(
                    {
                        f"p{i}_states_si": value(state).T * p.mission.scales[i].state,
                        f"p{i}_thrust_n": value(control).reshape(3, -1).T
                        * p.mission.settings.thrust_max_n,
                        f"p{i}_sigma_n": value(sigma).ravel() * p.mission.settings.thrust_max_n,
                    }
                )
                if p.direction_variables[i] is not None:
                    arrays.update(
                        {
                            f"p{i}_direction": value(p.direction_variables[i]).reshape(3, -1).T,
                            f"p{i}_throttle": value(sigma).ravel(),
                        }
                    )
            arrays["metadata"] = np.array(json.dumps(record))
            temporary = self.directory / "latest.tmp.npz"
            np.savez_compressed(temporary, **arrays)
            temporary.replace(self.directory / "latest.npz")
            if iteration % 100 == 0:
                np.savez_compressed(self.directory / f"iteration_{iteration:05d}.npz", **arrays)
            with (self.directory / "progress.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        except (RuntimeError, ValueError, OSError) as exc:
            if not self.error_reported:
                print("Checkpoint warning (solver continues): " + str(exc), flush=True)
                self.error_reported = True
