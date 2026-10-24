"""Differentiable RK7 arc maps for direct multiple shooting.

The integration grid is uniform and fixed while solving each NLP. Surface and
mass-floor samples reuse internal step endpoints; no extra steps are inserted.
All derivatives (including duration derivatives and Hessians) use CasADi AD.
"""

import casadi as ca
import numpy as np
from integrator import RK7_A, RK7_B


def arc_flow(mission, phase, substeps=20):
    if not isinstance(substeps, int) or substeps < 1:
        raise ValueError("Shooting substeps must be a positive integer")
    x, u = (ca.SX.sym("x", 7), ca.SX.sym("u", 3))
    sigma, dt = (ca.SX.sym("sigma"), ca.SX.sym("dt"))
    effective_sigma = sigma
    stages = []
    for i in range(len(RK7_B)):
        z = x
        for j in range(i):
            if RK7_A[i, j]:
                z = z + dt * RK7_A[i, j] * stages[j]
        stages.append(mission.rhs(phase, z, u, effective_sigma))
    endpoint = x + dt * sum((RK7_B[j] * stages[j] for j in range(len(RK7_B)) if RK7_B[j]))
    step = ca.Function(f"rk7_p{phase}", [x, u, sigma, dt], [endpoint], {"cse": True})
    x, u = (ca.MX.sym("x", 7), ca.MX.sym("u", 3))
    sigma, duration = (ca.MX.sym("sigma"), ca.MX.sym("duration"))
    state, samples = (x, [])
    for k in range(substeps):
        state = step(state, u, sigma, duration / substeps)
        if k < substeps - 1:
            samples.append(state)
    return ca.Function(
        f"shoot_p{phase}_n{substeps}",
        [x, u, sigma, duration],
        [state, ca.horzcat(*samples) if samples else ca.MX.zeros(7, 0)],
    )


def get_arc_flow(mission, phase, substeps):
    if not hasattr(mission, "_shooting_flows"):
        mission._shooting_flows = {}
    key = (phase, substeps)
    if key not in mission._shooting_flows:
        mission._shooting_flows[key] = arc_flow(mission, phase, substeps)
    return mission._shooting_flows[key]


def evaluate_shooting_phase(solution, phase, tau, mission):
    """Sample the integrated arc, not an HS interpolation of shooting nodes."""
    record = solution["phases"][phase]
    mesh = record["mesh"]
    k = min(max(np.searchsorted(mesh, tau, side="right") - 1, 0), len(mesh) - 2)
    theta = np.clip((tau - mesh[k]) / (mesh[k + 1] - mesh[k]), 0.0, 1.0)
    if theta == 0.0:
        return record["states_si"][k].copy()
    if theta == 1.0:
        return record["states_si"][k + 1].copy()
    scale, cfg = (mission.scales[phase], mission.settings)
    flow = get_arc_flow(mission, phase, solution["shooting_substeps"])
    h = record["duration_s"] * (mesh[k + 1] - mesh[k]) * theta / scale.time
    result, _ = flow(
        record["states_si"][k] / scale.state,
        record["thrust_n"][k] / cfg.thrust_max_n,
        record["sigma_n"][k] / cfg.thrust_max_n,
        h,
    )
    return np.asarray(result).ravel() * scale.state
