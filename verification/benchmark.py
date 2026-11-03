"""Benchmark batched nominal linearization against sequential propagation.

Pass a saved single-phase MadNLP NPZ. Set JAX_PLATFORMS=cpu or cuda before launching.
Timings exclude compilation, synchronize through NumPy, and include device transfers.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import jax
import stochastic_trajectory_optimization.dynamics as d


def benchmark(path, repetitions=15):
    data = np.load(path)
    case = d.load_case(path.stem)
    options = d.default_options(case.test_case_id)
    normalization = d.build_normalization(case, options)
    dynamics = d.Dynamics(case)
    states, controls, steps = data["ref_states"], data["ref_controls"], data["steps_nd"]

    def sequential():
        state_matrices, control_matrices = [], []
        for k, step in enumerate(steps):
            _, a, b = dynamics.propagate(
                states[:, k : k + 1],
                controls[:, k : k + 1],
                float(step),
                1,
                with_jacobian=True,
            )
            state_matrices.append(
                normalization.inverse_matrix @ a[0] @ normalization.matrix
            )
            control_matrices.append(
                normalization.inverse_matrix @ b[0] * case.max_thrust_nd
            )
        return state_matrices, control_matrices

    def batched():
        return d.normalized_arc_jacobians(
            case, dynamics, normalization, states, controls, steps, 1
        )

    reference, candidate = sequential(), batched()
    for a, b in zip(reference, candidate):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-13)
    times = {"sequential": [], "batched": []}
    for _ in range(repetitions):
        for name, function in [("sequential", sequential), ("batched", batched)]:
            start = time.perf_counter()
            function()
            times[name].append(time.perf_counter() - start)
    medians = {name: float(np.median(values)) for name, values in times.items()}
    return dict(
        backend=jax.default_backend(),
        arcs=len(steps),
        repetitions=repetitions,
        max_absolute_differences=[
            float(np.max(np.abs(np.asarray(a) - b)))
            for a, b in zip(reference, candidate)
        ],
        seconds=times,
        medians=medians,
        speedup=medians["sequential"] / medians["batched"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("solution", type=Path)
    parser.add_argument("--repetitions", type=int, default=15)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("at least three repetitions are required")
    result = json.dumps(benchmark(args.solution, args.repetitions), indent=2)
    print(result)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")


if __name__ == "__main__":
    main()
