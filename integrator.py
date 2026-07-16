"This module contains the functions which are used to integrate the equations of motion of a system."

from __future__ import annotations

import numpy as np
import cr3bp


def rk4(
    case: cr3bp.CR3BPEarthMoon,
    state: np.ndarray,
    control: np.ndarray,
    sigma: float,
    step_size: float,
    control_end: np.ndarray | None = None,
    sigma_end: float | None = None,
) -> np.ndarray:
    "Single RK4 step. By default `control` and `sigma` are held constant across "
    "the step; passing `control_end`/`sigma_end` makes them vary linearly, "
    "evaluated at the stage times t, t + h/2, and t + h, which keeps the step "
    "4th-order accurate for time-varying controls."
    if control_end is None:
        control_end = control
    if sigma_end is None:
        sigma_end = sigma

    control_mid = 0.5 * (control + control_end)
    sigma_mid = 0.5 * (sigma + sigma_end)
    k1 = cr3bp.eom(case, state, control, sigma)
    k2 = cr3bp.eom(case, state + 0.5 * step_size * k1, control_mid, sigma_mid)
    k3 = cr3bp.eom(case, state + 0.5 * step_size * k2, control_mid, sigma_mid)
    k4 = cr3bp.eom(case, state + step_size * k3, control_end, sigma_end)
    return state + step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
