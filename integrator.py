"This module contains the functions which are used to integrate the equations of motion of a system."

from __future__ import annotations

import numpy as np
import deterministic_cr3bp

# Eleven-stage explicit Runge-Kutta of order seven, fixed step. The coefficients
# are Fehlberg's (NASA TR R-287, 1968); this is the seventh-order formula alone,
# not an embedded pair -- there is one weight vector, no second solution, no
# error estimate and no step-size adaptation.
#
# Chosen over the classical RK4 because the arc map is evaluated at a fixed step
# and the optimiser differentiates through it. Measured on a transfer arc, it is
# five times more accurate than RK4 at four substeps while using eleven
# right-hand-side evaluations against sixteen.
RK7_C = np.array(
    [0.0, 2/27, 1/9, 1/6, 5/12, 1/2, 5/6, 1/6, 2/3, 1/3, 1.0]
)


def _rk7_tableau() -> tuple[np.ndarray, np.ndarray]:
    a = np.zeros((11, 11))
    a[1, 0] = 2/27
    a[2, 0] = 1/36;        a[2, 1] = 1/12
    a[3, 0] = 1/24;        a[3, 2] = 1/8
    a[4, 0] = 5/12;        a[4, 2] = -25/16;    a[4, 3] = 25/16
    a[5, 0] = 1/20;        a[5, 3] = 1/4;       a[5, 4] = 1/5
    a[6, 0] = -25/108;     a[6, 3] = 125/108;   a[6, 4] = -65/27;    a[6, 5] = 125/54
    a[7, 0] = 31/300;      a[7, 4] = 61/225;    a[7, 5] = -2/9;      a[7, 6] = 13/900
    a[8, 0] = 2;           a[8, 3] = -53/6;     a[8, 4] = 704/45
    a[8, 5] = -107/9;      a[8, 6] = 67/90;     a[8, 7] = 3
    a[9, 0] = -91/108;     a[9, 3] = 23/108;    a[9, 4] = -976/135
    a[9, 5] = 311/54;      a[9, 6] = -19/60;    a[9, 7] = 17/6;      a[9, 8] = -1/12
    a[10, 0] = 2383/4100;  a[10, 3] = -341/164; a[10, 4] = 4496/1025
    a[10, 5] = -301/82;    a[10, 6] = 2133/4100; a[10, 7] = 45/82
    a[10, 8] = 45/164;     a[10, 9] = 18/41

    b = np.zeros(11)
    b[0] = 41/840; b[5] = 34/105; b[6] = 9/35; b[7] = 9/35
    b[8] = 9/280;  b[9] = 9/280;  b[10] = 41/840
    return a, b


RK7_A, RK7_B = _rk7_tableau()
# Guard the transcription: the row sums must equal the stage times and the
# weights must sum to one, or the method silently loses its order.
assert np.allclose(RK7_A.sum(axis=1), RK7_C, atol=1e-14)
assert abs(RK7_B.sum() - 1.0) < 1e-14

# Stages whose weight is non-zero, and stages any later stage depends on. Every
# stage here is needed, but the lists make the intent explicit for readers
# comparing against the published tableau.
RK7_STAGES = RK7_A.shape[0]


def rk7(
    case: deterministic_cr3bp.CR3BPEarthMoon,
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
    them at its own stage time, which keeps the step accurate for time-varying
    controls.
    """

    # `varying` is decided from whether the caller supplied end values at all,
    # not by comparing them: comparison would break for non-array inputs and
    # would silently switch behaviour when a constant control happens to equal
    # its endpoint.
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
        stages.append(
            deterministic_cr3bp.eom(case, stage_state, stage_control, stage_sigma)
        )

    propagated = state
    for i in range(RK7_STAGES):
        if RK7_B[i] != 0.0:
            propagated = propagated + step_size * RK7_B[i] * stages[i]
    return propagated


# The name the rest of the project calls; kept so call sites read as "the
# integrator" rather than naming an order they do not depend on.
rk4 = rk7
