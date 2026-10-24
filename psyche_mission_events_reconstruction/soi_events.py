"""Laplace SOI geometry and bracketed crossings of a geometric ephemeris.

Omitting a semimajor axis selects instantaneous planetary heliocentric
distance. The current mission configuration explicitly supplies fixed axes.
These routines find events in supplied data; they do not build an OCP guess.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import numpy as np
from scipy.optimize import brentq

StateFunction = Callable[[float], np.ndarray]


@dataclass(frozen=True)
class LaplaceSOI:
    mu_planet: float
    mu_sun: float
    semimajor_axis_m: float | None = None

    def __post_init__(self):
        if not 0 < self.mu_planet < self.mu_sun or not np.isfinite(self.mu_sun):
            raise ValueError("Require finite positive GMs with planet GM below Sun GM")
        if self.semimajor_axis_m is not None and (
            not np.isfinite(self.semimajor_axis_m) or self.semimajor_axis_m <= 0
        ):
            raise ValueError("Fixed semimajor axis must be finite and positive")

    @property
    def factor(self):
        return (self.mu_planet / self.mu_sun) ** 0.4

    def radius_rate(self, planet_sun_state):
        if self.semimajor_axis_m is not None:
            return (self.factor * self.semimajor_axis_m, 0.0)
        state = np.asarray(planet_sun_state, dtype=float)
        radius = np.linalg.norm(state[:3])
        if state.shape != (6,) or not np.all(np.isfinite(state)) or radius <= 0:
            raise ValueError("Invalid heliocentric planet state")
        return (self.factor * radius, self.factor * np.dot(state[:3], state[3:]) / radius)

    def residual_and_rate(self, spacecraft_planet_state, planet_sun_state):
        state = np.asarray(spacecraft_planet_state, dtype=float)
        radius = np.linalg.norm(state[:3])
        if state.shape != (6,) or not np.all(np.isfinite(state)) or radius <= 0:
            raise ValueError("Invalid planet-relative spacecraft state")
        soi, soi_rate = self.radius_rate(planet_sun_state)
        return (radius - soi, np.dot(state[:3], state[3:]) / radius - soi_rate)


def refine_bracket(function, left: float, right: float) -> dict:
    """Brent root plus an independently bisected final sign bracket.

    Use an offset variable to avoid relative tolerances on a large J2000 ET.
    The final absolute ET is still limited by its float64 representation.
    """
    if not np.isfinite(left + right) or right <= left:
        raise ValueError("Invalid root bracket")
    gl, gr = (float(function(left)), float(function(right)))
    if not np.isfinite(gl + gr) or gl * gr > 0:
        raise ValueError("Root must be bracketed by finite opposite signs")
    if gl == 0 or gr == 0:
        epoch = left if gl == 0 else right
        return {
            "et_tdb_s": epoch,
            "residual_m": 0.0,
            "initial_bracket_et_tdb_s": [left, right],
            "bisection_bracket_et_tdb_s": [epoch, epoch],
            "bisection_width_s": 0.0,
            "brent_iterations": 0,
            "absolute_epoch_ulp_s": float(abs(np.spacing(epoch))),
        }
    offset, result = brentq(
        lambda s: function(left + s), 0.0, right - left, xtol=1e-08, rtol=1e-14, full_output=True
    )
    epoch = float(left + offset)
    a, b, fa, fb = (left, right, gl, gr)
    while b - a > max(5e-07, 4 * np.spacing(max(abs(a), abs(b)))):
        middle = a + (b - a) / 2
        if middle in (a, b):
            break
        fm = float(function(middle))
        if fm == 0:
            a = b = middle
            break
        if np.signbit(fm) == np.signbit(fa):
            a, fa = (middle, fm)
        else:
            b, fb = (middle, fm)
    if not result.converged or not a - 2 * abs(np.spacing(epoch)) <= epoch <= b + 2 * abs(
        np.spacing(epoch)
    ):
        raise RuntimeError("Independent root refinements disagree")
    return {
        "et_tdb_s": epoch,
        "residual_m": float(function(epoch)),
        "initial_bracket_et_tdb_s": [left, right],
        "bisection_bracket_et_tdb_s": [a, b],
        "bisection_width_s": b - a,
        "brent_iterations": result.iterations,
        "absolute_epoch_ulp_s": float(np.spacing(epoch)),
    }


def find_soi_crossings(
    spacecraft_planet: StateFunction,
    planet_sun: StateFunction,
    model: LaplaceSOI,
    start: float,
    end: float,
    step_s: float = 1800.0,
    extra_epochs=(),
) -> list[dict]:
    """Scan a verified continuous coverage interval, then refine sign changes.

    Finite sampling cannot prove absence of arbitrarily brief crossings. The
    caller must repeat at smaller steps and check encounter geometry. Include
    SPK segment boundaries in extra_epochs to audit possible discontinuities.
    """
    if not np.isfinite(start + end + step_s) or end <= start or step_s <= 0:
        raise ValueError("Invalid event search interval or scan step")
    times = np.unique(np.r_[start + np.arange(0.0, end - start, step_s), end, extra_epochs])
    times = times[(times >= start) & (times <= end)]

    def residual(epoch):
        return model.residual_and_rate(spacecraft_planet(epoch), planet_sun(epoch))[0]

    values = np.array([residual(t) for t in times])
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite event residual in the search interval")
    events = []
    for index in range(times.size - 1):
        a, b = times[index : index + 2]
        fa, fb = values[index : index + 2]
        if fa * fb < 0 or fa == 0 or fb == 0:
            result = refine_bracket(residual, a, b)
            epoch = result["et_tdb_s"]
            if events and abs(epoch - events[-1]["et_tdb_s"]) < 1e-05:
                continue
            state, planet = (spacecraft_planet(epoch), planet_sun(epoch))
            _, rate = model.residual_and_rate(state, planet)
            if abs(rate) < 1e-08:
                raise ValueError("Tangential SOI event requires a separate treatment")
            result.update(
                direction="entry" if rate < 0 else "exit",
                crossing_rate_m_s=float(rate),
                soi_radius_m=float(model.radius_rate(planet)[0]),
                planet_relative_state_si=np.asarray(state).tolist(),
                planet_sun_state_si=np.asarray(planet).tolist(),
            )
            events.append(result)
    return events
