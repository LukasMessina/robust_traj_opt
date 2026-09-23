"""Optional cross-language check against the unchanged original Python functions.

Requires numpy, scipy, jax, and diffrax; SNOPT is not required. Run after a Julia
solve. This evaluates the original UT and objective at the Julia solution and
compares its original constraint residuals. No original source is modified.
"""
from __future__ import annotations

import ast
import os
from pathlib import Path
from dataclasses import dataclass, field
import argparse
import tomllib

os.environ["JAX_PLATFORMS"] = "cpu"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.stats import chi2, norm
from diffrax import (AbstractPath, ODETerm, ControlTerm, MultiTerm, SaveAt,
                     ConstantStepSize, SpaceTimeLevyArea, diffeqsolve, ShARK)


def reference_namespace():
    source = Path(__file__).resolve().parents[2] / "double_integrator_stochastic_traj_opt.py"
    keep = {
        "Options", "LowerTriangular", "unscented_weights", "psi_inverse",
        "continuous_time_matrices", "_drift_vector_field", "_diffusion_vector_field",
        "PrescribedSpaceTimeLevyPath", "prescribed_integration_sde_step",
        "build_propagation_arc_map", "regularized_control_norm", "symbolic_psqrt_spectral_radius",
        "_terminal_constraint_slack", "_unpack_stochastic_vars", "_stochastic_funcs",
        "tvlqr_gains",
    }
    definitions = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in keep]
    namespace = dict(globals(), NX=4, NU=2)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    namespace["LTRI"] = namespace["LowerTriangular"](4)
    namespace["_SDE_SOLVER"] = ShARK()
    namespace["_prescribed_sde_integration_step_batch"] = jax.vmap(
        namespace["prescribed_integration_sde_step"], in_axes=(1,1,1,1,None,None), out_axes=1)
    return namespace


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path,help="Julia output directory containing solutions.npz and diagnostics.toml")
    parser.add_argument("--acceptance-tolerance",type=float,
                        help="explicit feasibility threshold for a relaxed sweep case; defaults to the saved strict threshold")
    args=parser.parse_args()
    arrays=np.load(args.output/"solutions.npz")
    report=tomllib.loads((args.output/"diagnostics.toml").read_text())
    ns=reference_namespace()
    problem=report.get("problem", {
        "spectral_eigenvalue_smoothing":report["spectral_eigenvalue_smoothing"]})
    options=ns["Options"](**{key:np.asarray(value) if isinstance(value,list) else value
                            for key,value in problem.items()})
    seed_path=args.output/"initial_seed.npz"
    if seed_path.exists():
        seed=np.load(seed_path)
        uniform=report.get("initial_gain_uniform", "disabled")
        if uniform != "disabled":
            assert np.all(seed["gains"] == uniform)
            print(f"All {seed['gains'].size} initial gain entries equal {uniform:.9g}")
        elif report["tvlqr_control_weight_scale"] == 1 and report["initial_gain_max_abs"] == "unscaled":
            A=np.eye(4)
            A[:2,2:]=options.dt*np.eye(2)
            B=np.vstack((0.5*options.dt**2*np.eye(2),options.dt*np.eye(2)))
            reference=ns["tvlqr_gains"](options,A,B)
            difference=float(np.max(np.abs(seed["gains"]-reference)))
            print(f"Initial gain difference from original Python Bryson TVLQR: {difference:.3e}")
            assert difference < 1e-12
    arc=ns["build_propagation_arc_map"](options,arrays["Lc"])
    xdict={key:jnp.asarray(arrays[key].reshape(-1)) for key in (
        "means","feedforward","gains","cholesky_factor","terminal_margin")}
    funcs=ns["_stochastic_funcs"](xdict,options,arc,
        ns["psi_inverse"](2,1-options.control_confidence),float(norm.ppf(options.path_confidence)))
    objective=float(funcs["objective"])
    violation=max(
        max(float(np.max(np.abs(funcs[key]))) for key in (
            "initial_covariance","mean_defects","state_covariance_defects","terminal_residual")),
        float(np.max(np.asarray(funcs["control_chance"])-options.u_max)),
        float(np.max(np.asarray(funcs["path_0"])-options.b1)),
        float(np.max(np.asarray(funcs["path_1"])-options.b2)),
        float(np.max(np.abs(arrays["means"][:,0]-options.x0_mean))),
        float(np.max(np.abs(arrays["means"][:,-1]-options.xf_mean))),
        -float(np.min(arrays["cholesky_factor"][:,[0,2,5,9]])),
        options.terminal_margin_floor-float(np.min(arrays["terminal_margin"][[0,2,5,9]])))
    objective_difference=abs(objective-report["objective"])
    print(f"Original Python objective: {objective:.12f}")
    print(f"Julia/Python objective difference: {objective_difference:.3e}")
    print(f"Original Python max constraint violation: {violation:.3e}")
    print(f"Julia/Python residual difference: {abs(violation-report['max_constraint_violation']):.3e}")
    assert objective_difference < 1e-9
    assert abs(violation-report["max_constraint_violation"]) < 1e-9
    tolerance = report["acceptance_tolerance"] if args.acceptance_tolerance is None else args.acceptance_tolerance
    assert np.isfinite(tolerance) and tolerance > 0
    print(f"Verification feasibility threshold: {tolerance:.3e}")
    assert violation <= tolerance


if __name__ == "__main__":
    main()
