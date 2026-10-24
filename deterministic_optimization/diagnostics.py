"""Local acceptance and report-only continuous diagnostics."""

import numpy as np
from .io import write_json
from .psyche_validation import algebraic_checks, reintegrate


def local_defects_met(defects):
    return bool(
        defects and all(np.isfinite(d["max_scaled"]) and d["max_scaled"] <= 1.0 for d in defects)
    )


def mesh_accepted(settings, solution, defects):
    """Require NLP success and local consistency; global propagation is separate."""
    return bool(solution["solver_success"] and (local_defects_met(defects)))


def convergence_policy(settings):
    return (
        "IPOPT success and local RK7 defects within tolerance; global diagnostics are report-only"
    )


def postsolve_diagnostics(mission, solution, output, prefix):
    """Diagnostic failures cannot undo an already finished adaptive solve."""
    checks = {"role": "diagnostics only; no acceptance or refinement decisions"}
    for name, callback in (
        ("algebraic", algebraic_checks),
        ("continuous_reintegration", reintegrate),
    ):
        try:
            values = callback(mission, solution)
            if name == "continuous_reintegration":
                values, arrays = values
                np.savez_compressed(output / f"{prefix}_reintegration.npz", **arrays)
            checks[name] = values
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
            checks[name] = {"status": "diagnostic unavailable", "error": str(exc)}
    write_json(output / f"{prefix}_diagnostics.json", checks)
    return checks


def reference_sensitivity_diagnostic(output):
    """A failed tighter reference propagation cannot reject an accepted NLP."""
    from .global_reference import main as reference_diagnostics

    try:
        reference_diagnostics(output, output / "global_reference_accuracy.json")
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        write_json(output / "global_reference_error.json", {
            "status": "diagnostic unavailable", "error": str(exc),
            "used_for_acceptance": False, "used_for_refinement": False,
        })
