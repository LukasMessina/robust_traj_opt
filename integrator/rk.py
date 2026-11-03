"""Fixed-step seventh-order Runge–Kutta integration."""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
import casadi as ca

if TYPE_CHECKING:
    from deterministic_trajectory_optimization.models import CR3BPEarthMoon, TestCase

# Eleven-stage explicit Runge-Kutta of order seven, fixed step. The coefficients
# define the seventh-order formula alone,
# not an embedded pair -- there is one weight vector, no second solution, no
# error estimate and no step-size adaptation.
#

RK7_C = np.array(
    [0.0, 2 / 27, 1 / 9, 1 / 6, 5 / 12, 1 / 2, 5 / 6, 1 / 6, 2 / 3, 1 / 3, 1.0]
)


def _rk7_tableau() -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((11, 11))
    a[1, 0] = 2 / 27
    a[2, 0] = 1 / 36
    a[2, 1] = 1 / 12
    a[3, 0] = 1 / 24
    a[3, 2] = 1 / 8
    a[4, 0] = 5 / 12
    a[4, 2] = -25 / 16
    a[4, 3] = 25 / 16
    a[5, 0] = 1 / 20
    a[5, 3] = 1 / 4
    a[5, 4] = 1 / 5
    a[6, 0] = -25 / 108
    a[6, 3] = 125 / 108
    a[6, 4] = -65 / 27
    a[6, 5] = 125 / 54
    a[7, 0] = 31 / 300
    a[7, 4] = 61 / 225
    a[7, 5] = -2 / 9
    a[7, 6] = 13 / 900
    a[8, 0] = 2
    a[8, 3] = -53 / 6
    a[8, 4] = 704 / 45
    a[8, 5] = -107 / 9
    a[8, 6] = 67 / 90
    a[8, 7] = 3
    a[9, 0] = -91 / 108
    a[9, 3] = 23 / 108
    a[9, 4] = -976 / 135
    a[9, 5] = 311 / 54
    a[9, 6] = -19 / 60
    a[9, 7] = 17 / 6
    a[9, 8] = -1 / 12
    a[10, 0] = 2383 / 4100
    a[10, 3] = -341 / 164
    a[10, 4] = 4496 / 1025
    a[10, 5] = -301 / 82
    a[10, 6] = 2133 / 4100
    a[10, 7] = 45 / 82
    a[10, 8] = 45 / 164
    a[10, 9] = 18 / 41

    b = np.zeros(11)
    b[0] = 41 / 840
    b[5] = 34 / 105
    b[6] = 9 / 35
    b[7] = 9 / 35
    b[8] = 9 / 280
    b[9] = 9 / 280
    b[10] = 41 / 840
    return a, b


RK7_A, RK7_B = _rk7_tableau()
# Guard the transcription: the row sums must equal the stage times and the
# weights must sum to one, or the method silently loses its order.
assert np.allclose(RK7_A.sum(axis=1), RK7_C, atol=1e-14)
assert abs(RK7_B.sum() - 1.0) < 1e-14


RK7_STAGES = RK7_A.shape[0]


def integrate_case(
    case: CR3BPEarthMoon,
    state: np.ndarray,
    control: np.ndarray,
    sigma: float,
    step_size: float,
    control_end: np.ndarray | None = None,
    sigma_end: float | None = None,
) -> np.ndarray:
    """Single fixed-step seventh-order Runge-Kutta step.

    By default `control` and `sigma` are held constant across the step; passing
    `control_end`/`sigma_end` makes them vary linearly, each stage evaluating
    them at its own stage time, which keeps the step accurate for linear time-varying
    controls.
    """

    # Lazy import avoids the solver/integrator circular import at module load.
    from deterministic_trajectory_optimization.models import eom

    # `varying` is decided from whether the caller supplied end values at all
    varying = control_end is not None or sigma_end is not None
    if control_end is None:
        control_end = control
    if sigma_end is None:
        sigma_end = sigma

    stages: list[np.ndarray] = []
    for i in range(RK7_STAGES):
        stage_state = state
        for j in range(i):
            coefficient = RK7_A[i, j]
            if coefficient != 0.0:
                stage_state = stage_state + step_size * coefficient * stages[j]
        if varying:
            theta = RK7_C[i]
            stage_control = control + theta * (control_end - control)
            stage_sigma = sigma + theta * (sigma_end - sigma)
        else:
            stage_control, stage_sigma = control, sigma
        stages.append(eom(case, stage_state, stage_control, stage_sigma))

    propagated = state
    for i in range(RK7_STAGES):
        if RK7_B[i] != 0.0:
            propagated = propagated + step_size * RK7_B[i] * stages[i]
    return propagated


"""The original RK7 tableau and arithmetic for numerical rollout/defects."""


def rk7(rhs, state, dt):
    stages = []
    for i in range(len(RK7_B)):
        value = state.copy()
        for j in range(i):
            if RK7_A[i, j]:
                value += dt * RK7_A[i, j] * stages[j]
        stages.append(rhs(value))
    return state + dt * np.einsum("i,ij->j", RK7_B, np.array(stages))


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
    endpoint = x + dt * sum(
        (RK7_B[j] * stages[j] for j in range(len(RK7_B)) if RK7_B[j])
    )
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


def propagate_periodic_orbit(
    case: CR3BPEarthMoon,
    initial_state: np.ndarray,
    period_nd: float,
    steps: int = 1600,
) -> np.ndarray:
    orbit = np.empty((7, steps + 1), dtype=float)
    orbit[:, 0] = initial_state
    h = period_nd / steps
    for k in range(steps):
        orbit[:, k + 1] = integrate_case(case, orbit[:, k], np.zeros(3), 0.0, h)
    return orbit


def integrate_with_dense_control(
    case: TestCase,
    initial_state: np.ndarray,
    t_grid_nd: np.ndarray,
    u_values: np.ndarray,
    sigma_values: np.ndarray,
) -> np.ndarray:
    if u_values.shape[1] != t_grid_nd.size or sigma_values.shape[1] != t_grid_nd.size:
        raise ValueError(
            "Dense ZOH control, sigma, and time arrays must have the same length."
        )

    x_integrated = np.empty((initial_state.size, t_grid_nd.size), dtype=float)
    x_integrated[:, 0] = initial_state

    for k in range(t_grid_nd.size - 1):
        h = float(t_grid_nd[k + 1] - t_grid_nd[k])
        x_integrated[:, k + 1] = integrate_case(
            case,
            x_integrated[:, k],
            u_values[:, k],
            float(sigma_values[0, k]),
            h,
        )
    return x_integrated


def rollout_initial_guess(
    case: TestCase,
    mesh: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from deterministic_trajectory_optimization.models import (
        INITIAL_GUESS_SUBSTEPS_PER_INTERVAL,
    )

    intervals = mesh.size - 1
    u_component_nd = 1e-4 / case.thrust_unit
    u_guess_nd = np.full((3, intervals), u_component_nd, dtype=float)
    sigma_guess_nd = np.full(
        (1, intervals), np.linalg.norm(u_guess_nd[:, 0]), dtype=float
    )

    x_guess = np.empty((7, intervals + 1), dtype=float)
    x_guess[:, 0] = case.x0_augmented_state
    for k in range(intervals):
        h = case.tof_nd * float(mesh[k + 1] - mesh[k])
        substep = h / INITIAL_GUESS_SUBSTEPS_PER_INTERVAL
        state = x_guess[:, k]
        for _ in range(INITIAL_GUESS_SUBSTEPS_PER_INTERVAL):
            state = integrate_case(
                case, state, u_guess_nd[:, k], sigma_guess_nd[0, k], substep
            )
        x_guess[:, k + 1] = state

    return x_guess, u_guess_nd, sigma_guess_nd
