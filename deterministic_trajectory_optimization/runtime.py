"""JSON serialization and provenance helpers."""

import hashlib
import json
from pathlib import Path
import numpy as np
import shutil
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
import sys


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(jsonable(data), indent=2), encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot_sources(output):
    root = Path(__file__).resolve().parents[1]
    sources = []
    for package in (
        "deterministic_trajectory_optimization",
        "integrator",
        "plotter",
        "data",
        "verification",
    ):
        sources.extend(sorted((root / package).rglob("*.py")))
    hashes = {}
    for source in sources:
        relative = source.relative_to(root)
        target = Path(output) / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative.as_posix()] = digest(source)
    return hashes


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
            x, g, lam = (
                value(op.x).ravel(),
                value(op.g).ravel(),
                value(op.lam_g).ravel(),
            )
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
            for i, (state, control, sigma) in enumerate(p.variables):
                arrays.update(
                    {
                        f"p{i}_states_si": value(state).T * p.mission.scales[i].state,
                        f"p{i}_thrust_n": value(control).reshape(3, -1).T
                        * p.mission.settings.thrust_max_n,
                        f"p{i}_sigma_n": value(sigma).ravel()
                        * p.mission.settings.thrust_max_n,
                        f"p{i}_direction": value(p.direction_variables[i])
                        .reshape(3, -1)
                        .T,
                        f"p{i}_throttle": value(sigma).ravel(),
                    }
                )
            arrays["metadata"] = np.array(json.dumps(record))
            temporary = self.directory / "latest.tmp.npz"
            np.savez_compressed(temporary, **arrays)
            temporary.replace(self.directory / "latest.npz")
            if iteration % 100 == 0:
                np.savez_compressed(
                    self.directory / f"iteration_{iteration:05d}.npz", **arrays
                )
            with (self.directory / "progress.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(record) + "\n")
        except (RuntimeError, ValueError, OSError) as exc:
            if not self.error_reported:
                print("Checkpoint warning (solver continues): " + str(exc), flush=True)
                self.error_reported = True


@contextmanager
def redirect_worker_output(log):
    """Redirect Python streams as well as the native stdout/stderr descriptors.

    On Windows, a console-backed TextIOWrapper uses WriteConsoleW. Redirecting
    its descriptor to a regular file with dup2 does not turn it into a file
    stream. Bind Python output directly to the log during native redirection,
    then restore descriptors before restoring the original Python streams.
    This is process-wide and must only be used in an isolated solver worker.
    """
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()
    saved = {}
    with redirect_stdout(log), redirect_stderr(log):
        try:
            for descriptor in (1, 2):
                saved[descriptor] = os.dup(descriptor)
                os.dup2(log.fileno(), descriptor)
            yield
        finally:
            try:
                log.flush()
            finally:
                for descriptor, backup in saved.items():
                    try:
                        os.dup2(backup, descriptor)
                    finally:
                        os.close(backup)
