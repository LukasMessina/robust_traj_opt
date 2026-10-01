"""Validate Julia runs with the unchanged JAX reference code and draw their figures.

    python madnlp_stochastic_traj_opt/tools/validate.py cpu|gpu [--case CASE] [--samples N]

For every case solved in the given execution mode (output/<mode>/<case>/<case>.npz,
written by run.jl):

1. Seed check: the Bryson TVLQR gains and gain scales recomputed by the JAX
   functions from the Julia energy-optimal reference, and the UT-propagated seed
   covariances recomputed by the JAX arc map from the Julia seed gains.
2. Solution check: `recover_solution` of the reference evaluated at the Julia
   optimum (its NLP residuals, objective and acceptance: solver success and
   residuals <= 1e-8). The chance constraint is evaluated with the reference's own
   eigenvalue formula (s^2 = 1e-30), whatever smoothing the Julia run used.
3. The reference's `monte_carlo_rollouts`, `compute_diagnostics` and
   `save_outputs`: the same Monte Carlo validation (NumPy streams, seed 42), NPZ/CSV
   export and the four figures (dispersion evolution, thrust budget, projections,
   terminal distribution).

Everything is written to output/<mode>/<case>/validation/. Runs from any directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import types
from dataclasses import replace
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parents[1] / "output"
sys.path.insert(0, str(ROOT))

# pyoptsparse/SNOPT are only needed to *solve*; nothing evaluated here uses them.
for _name in ("pyoptsparse", "pyoptsparse.pySNOPT", "pyoptsparse.pySNOPT.pySNOPT"):
    sys.modules.setdefault(_name, types.ModuleType(_name))
sys.modules["pyoptsparse"].Optimization = object
sys.modules["pyoptsparse.pySNOPT.pySNOPT"].SNOPT = object

import numpy as np  # noqa: E402
import jax  # noqa: E402
import cr3bp_stochastic_traj_opt_jax as J  # noqa: E402


def max_rel(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def reference_trajectory(case, data):
    states = np.asarray(data["ref_states"])
    return J.ReferenceTraj(np.asarray(data["node_times_nd"]), np.asarray(data["steps_nd"]), states,
                           np.asarray(data["ref_controls"]), float(case.m0_wet * (states[6, 0] - states[6, -1])))


def stochastic_nlp(case, options, ref, normalization, arc, data):
    """The seeded TrajectoryNLP of the reference, without its pyOptSparse problem.

    The node scales are the Julia ones (its Cholesky variables are expressed in
    them); they agree with the reference's own to roundoff.
    """
    nlp = J.TrajectoryNLP.__new__(J.TrajectoryNLP)
    nlp.case, nlp.options, nlp.ref, nlp.n = case, options, ref, ref.n_arcs
    nlp.stochastic, nlp.normalization, nlp.arc = True, normalization, arc
    nlp.seed = types.SimpleNamespace(gain_scale=np.asarray(data["gain_scale"]))
    nlp.std = np.asarray(data["node_std"]).T
    nlp.inv_std = 1.0 / nlp.std
    nlp.variances = np.asarray(data["covariance_scale_variances"])
    nlp.initial_factor = np.asarray(J.LTRI.pack(
        np.linalg.cholesky(normalization.initial_covariance) * nlp.inv_std[0, :, None]))
    nlp.local_values = jax.jit(jax.vmap(nlp.local_output, in_axes=(0, 0)))
    nlp.values = jax.jit(nlp.functions)
    return nlp


def decision_vector(data):
    gain_scale = np.asarray(data["gain_scale"])
    return {
        "means": np.asarray(data["means"]).ravel(),
        "feedforward": np.asarray(data["feedforward"]).ravel(),
        "gains": (np.asarray(data["gains_normalized"]) / gain_scale[:, None, None]).ravel(),
        "cholesky_factor": np.asarray(data["cholesky_factor"]).ravel(),
        "terminal_margin": np.asarray(data["terminal_margin"]),
    }


def seed_check(case, options, normalization, dynamics, arc, ref, data):
    seed = J.build_initial_guess(case, options, dynamics, normalization, ref, arc)
    julia_gains = np.asarray(data["seed_gains_normalized"])
    _, covariances, _ = J.propagate_stochastic_moments(
        arc, seed.means, seed.feedforward, julia_gains, normalization.initial_covariance, ref.steps)
    return {
        "gain_scale_max_rel": max_rel(data["gain_scale"], seed.gain_scale),
        "tvlqr_gains_max_rel": max_rel(julia_gains, seed.gains),
        "propagated_seed_covariance_max_rel": max_rel(data["seed_covariances"], covariances),
    }


def validate(path: Path, samples: int | None):
    case = J.CASE_REGISTRY[path.stem]()
    data = dict(np.load(path))
    options = J.Options(truncated_uniform_mesh_arcs=None)
    options = replace(options, terminal_margin_floor=float(data["terminal_margin_floor"]),
                      spectral_radius_floor=float(data["spectral_radius_floor"]),
                      control_norm_eps=float(data["control_norm_eps"]))
    if samples is not None:
        options = replace(options, monte_carlo_samples=samples)
    normalization = J.build_normalization(case)
    dynamics = J.Dynamics(case)
    ref = reference_trajectory(case, data)
    arc = jax.jit(J.build_propagation_arc_map(case, options, normalization))
    print(f"[{case.test_case_id}] seed check ...", flush=True)
    seed_report = seed_check(case, options, normalization, dynamics, arc, ref, data)

    print(f"[{case.test_case_id}] solution check with the reference NLP functions ...", flush=True)
    nlp = stochastic_nlp(case, options, ref, normalization, arc, data)
    success = bool(data["solver_success"]) and bool(data["restoration_requirement_met"])
    solver_result = types.SimpleNamespace(xStar=decision_vector(data),
                                          optInform={"value": 1 if success else 0, "text": "MadNLP"})
    solution = J.recover_solution(nlp, solver_result)

    print(f"[{case.test_case_id}] Monte Carlo ({options.monte_carlo_samples} samples) and figures ...", flush=True)
    monte_carlo = J.monte_carlo_rollouts(case, options, normalization, solution, ref.steps)
    psi = J.psi_inverse(J.NU, options.violation_parameter)
    diagnostics = {**J.compute_diagnostics(case, options, ref, solution, normalization, monte_carlo, psi),
                   **solution.diagnostics}
    diagnostics["julia_objective_nd"] = float(data["objective"])
    output_prefix = path.parent / "validation" / case.test_case_id
    J.save_outputs(case, options, ref, solution, normalization, monte_carlo, diagnostics, psi,
                   output_prefix, dynamics)
    report = {"seed_check": seed_report, **{k: float(v) for k, v in diagnostics.items()}}
    (path.parent / "validation" / f"{case.test_case_id}_validation.json").write_text(json.dumps(report, indent=2))
    print_summary(case, report, options)


def print_summary(case, r, options):
    s = r["seed_check"]
    print(f"\n[{case.test_case_id}] validation with the reference code")
    print(f"  seed vs reference: gain scale {s['gain_scale_max_rel']:.1e}, TVLQR gains {s['tvlqr_gains_max_rel']:.1e}, "
          f"propagated covariances {s['propagated_seed_covariance_max_rel']:.1e} (max relative difference)")
    print(f"  objective {r['objective_nd']:.10f} (Julia {r['julia_objective_nd']:.10f})")
    print(f"  max equality residual {r['max_equality_residual']:.3e}, chance violation "
          f"{r['max_control_chance_violation']:.3e}, bound violation {r['max_bound_violation']:.3e}")
    print(f"  reference acceptance (solver success and residuals <= {options.major_feasibility_tol:.0e}): "
          f"{'yes' if r['converged'] else 'no'}")
    print(f"  terminal covariance / target, largest eigenvalue: predicted {r['terminal_covariance_max_eigenvalue']:.6f}, "
          f"Monte Carlo {r['monte_carlo_terminal_max_eigenvalue']:.6f}")
    print(f"  worst-arc thrust violation frequency {r['monte_carlo_max_violation_fraction']:.4%} "
          f"(limit {options.violation_parameter:.2%})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("cpu", "gpu"), help="execution mode of the runs")
    parser.add_argument("--case", choices=sorted(J.CASE_REGISTRY), default=None,
                        help="validate only this case; default: every solved case of the mode")
    parser.add_argument("--samples", type=int, default=None, help="Monte Carlo samples (reference default 50000)")
    args = parser.parse_args()
    os.chdir(ROOT)  # the reference resolves its data directories relative to the repository
    cases = [args.case] if args.case else sorted(J.CASE_REGISTRY)
    results = [OUTPUT / args.mode / case / f"{case}.npz" for case in cases]
    results = [path for path in results if path.is_file()]
    if not results:
        sys.exit(f"no result in {OUTPUT / args.mode} for {args.case or 'any case'}; run run.jl --mode={args.mode} first")
    for path in results:
        validate(path, args.samples)


if __name__ == "__main__":
    main()
