"This module contains the functions which are used to integrate the equations of motion of a system."

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cr3bp import CR3BPEarthMoon


def rk4(
    case: CR3BPEarthMoon,
    state: np.ndarray,
    control: np.ndarray,
    sigma: float,
    step_size: float,
) -> np.ndarray:
    from cr3bp import eom

    k1 = eom(case, state, control, sigma)
    k2 = eom(case, state + 0.5 * step_size * k1, control, sigma)
    k3 = eom(case, state + 0.5 * step_size * k2, control, sigma)
    k4 = eom(case, state + step_size * k3, control, sigma)
    return state + step_size * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
