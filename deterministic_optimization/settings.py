"""Verified defaults for the consolidated deterministic workflow."""

INITIAL_INTERVALS = 50
PSYCHE_INITIAL_INTERVALS = (60, 50, 90, 50)
DEFECT_SUBSTEPS = 64
MAX_REFINEMENTS = 8
MAX_INTERVALS = 1800


def ipopt_options(print_level=5):
    return dict(
        max_iter=5000,
        tol=1e-9,
        constr_viol_tol=1e-10,
        acceptable_iter=0,
        bound_relax_factor=0.0,
        mu_strategy="adaptive",
        print_level=print_level,
        sb="yes",
        hessian_approximation="exact",
        nlp_scaling_method="gradient-based",
        bound_push=1e-8,
        bound_frac=1e-8,
    )
