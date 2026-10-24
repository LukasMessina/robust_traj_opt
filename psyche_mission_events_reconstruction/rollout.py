"""Spacecraft initial guesses from forward dynamics, never spacecraft SPKs.

Two anchored coast chains seed the four phases: TIP/Earth -> Sun, and
Mars-entry/Mars -> Sun. The seed need not satisfy the fixed downstream target.
Planetary cached ephemerides are used only for SOI geometry and translations.
The adaptive IVP integrator creates coast timing guesses; it does not replace NLP
transcription or the existing RK7 defect checks in the eventual optimiser.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from .ephemeris_cache import EphemerisCache
from .soi_events import LaplaceSOI


@dataclass
class CoastArc:
    origin_id: int
    start_et: float
    solution: object

    @property
    def duration_s(self) -> float:
        return float(self.solution.t[-1])

    @property
    def end_et(self) -> float:
        return self.start_et + self.duration_s

    def at_elapsed(self, elapsed_s) -> np.ndarray:
        elapsed = np.asarray(elapsed_s, dtype=float)
        if (
            not np.all(np.isfinite(elapsed))
            or np.any(elapsed < 0)
            or np.any(elapsed > self.duration_s)
        ):
            raise ValueError("Requested epoch outside the integrated arc")
        return np.asarray(self.solution.sol(elapsed)).T

    @property
    def final_state(self) -> np.ndarray:
        return self.at_elapsed(self.duration_s)


def coast_rhs(mu: float, augmented_state: np.ndarray) -> np.ndarray:
    r, v = (augmented_state[:3], augmented_state[3:6])
    distance = np.linalg.norm(r)
    if not np.isfinite(distance) or distance <= 0:
        raise ValueError("Singular or nonfinite central-body state")
    return np.r_[v, -mu * r / distance**3, 0.0]


def integrate_coast(
    state_si, epoch, end_epoch, mu, origin_id, *, event=None, rtol=2e-12, max_step_s=3600.0
) -> CoastArc:
    state = np.asarray(state_si, dtype=float)
    if state.shape != (7,) or not np.all(np.isfinite(state)) or state[6] <= 0:
        raise ValueError("A finite Cartesian state and positive mass are required")
    if (
        not np.isfinite(mu)
        or mu <= 0
        or (not np.isfinite(epoch))
        or (not np.isfinite(end_epoch))
        or (end_epoch <= epoch)
    ):
        raise ValueError("Require positive GM and a finite forward time interval")
    if not np.isfinite(max_step_s) or max_step_s <= 0 or (not 0 < rtol < 1):
        raise ValueError("Invalid integration controls")
    result = solve_ivp(
        lambda elapsed, y: coast_rhs(mu, y),
        (0.0, float(end_epoch - epoch)),
        state,
        method="DOP853",
        rtol=rtol,
        atol=np.array([1e-05] * 3 + [1e-09] * 3 + [1e-10]),
        max_step=max_step_s,
        events=event,
        dense_output=True,
    )
    if not result.success:
        raise RuntimeError(f"Coast rollout failed: {result.message}")
    return CoastArc(origin_id, float(epoch), result)


def soi_radius_and_rate(
    cache: EphemerisCache, body: int, epoch: float, soi_model: LaplaceSOI | None = None
):
    """Explicit model; omitted model preserves archived instantaneous runs."""
    mu = cache.metadata["mu_m3_s2"]
    model = soi_model or LaplaceSOI(mu[str(body)], mu["10"])
    if model.mu_planet != mu[str(body)] or model.mu_sun != mu["10"]:
        raise ValueError("SOI model GMs must match the phase and planetary cache")
    if model.semimajor_axis_m is not None:
        return model.radius_rate(None)
    position = cache.state(body, epoch)[:3]
    rate = cache.derivative(body, epoch)[:3]
    return model.radius_rate(np.r_[position, rate])


def rollout_to_soi_exit(
    state_si,
    epoch,
    search_end_epoch,
    body: int,
    cache: EphemerisCache,
    *,
    rtol=2e-12,
    max_step_s=3600.0,
    soi_model: LaplaceSOI | None = None,
) -> CoastArc:
    if body not in (399, 499):
        raise ValueError("This rollout supports Earth or Mars departures")
    lo, hi = cache.valid_interval
    if not lo <= epoch < search_end_epoch <= hi:
        raise ValueError("SOI search must remain inside the validated cache interval")
    mu = cache.metadata["mu_m3_s2"][str(body)]

    def crossing(elapsed, y):
        radius, _ = soi_radius_and_rate(cache, body, epoch + elapsed, soi_model)
        return np.linalg.norm(y[:3]) - radius

    crossing.terminal = True
    crossing.direction = 1
    initial = np.asarray(state_si, dtype=float)
    if initial.shape != (7,) or not np.all(np.isfinite(initial)):
        raise ValueError("Expected a finite seven-component start state")
    if crossing(0, initial) > 1.0:
        raise ValueError("Initial state is outside the SOI by more than 1 m")
    arc = integrate_coast(
        initial, epoch, search_end_epoch, mu, body, event=crossing, rtol=rtol, max_step_s=max_step_s
    )
    if len(arc.solution.t_events[0]) != 1 or arc.duration_s <= 0:
        raise RuntimeError("No positive-time outward SOI crossing in search interval")
    final = arc.final_state
    _, radius_rate = soi_radius_and_rate(cache, body, arc.end_et, soi_model)
    crossing_rate = np.dot(final[:3], final[3:6]) / np.linalg.norm(final[:3]) - radius_rate
    if crossing_rate <= 0:
        raise RuntimeError("Detected event is not an outward crossing")
    return arc


def translate_to_sun(planet_state, body: int, epoch, cache: EphemerisCache):
    state = np.asarray(planet_state, dtype=float).copy()
    if state.shape != (7,):
        raise ValueError("Expected six Cartesian components and mass")
    state[:6] += cache.state(body, epoch)
    return state


def make_four_phase_coast_seed(
    tip_state,
    tip_epoch,
    mars_entry_state,
    mars_entry_epoch,
    arrival_epoch,
    arrival_target_sun_state,
    cache: EphemerisCache,
):
    """Build both anchored chains; fixed target values never reshape the seed.

    TIP input is seven components. Mars entry and arrival target are six.
    The mass propagated from the Earth chain seeds the Mars chain. The two
    fixed Cartesian boundary states are inputs, not trajectory histories.
    Arrival target is used only to report the initial mismatch.
    """
    mars = np.asarray(mars_entry_state, dtype=float)
    target = np.asarray(arrival_target_sun_state, dtype=float)
    if mars.shape != (6,) or target.shape != (6,) or (not np.all(np.isfinite(np.r_[mars, target]))):
        raise ValueError("Verified finite Mars-entry and arrival Cartesian targets required")
    if not tip_epoch < mars_entry_epoch < arrival_epoch <= cache.valid_interval[1]:
        raise ValueError("Invalid mission event ordering")
    mu_sun = cache.metadata["mu_m3_s2"]["10"]
    from .mission_soi import fixed_mission_soi

    earth_model = fixed_mission_soi(399, cache.metadata["mu_m3_s2"])
    mars_model = fixed_mission_soi(499, cache.metadata["mu_m3_s2"])
    earth = rollout_to_soi_exit(
        tip_state, tip_epoch, mars_entry_epoch, 399, cache, soi_model=earth_model
    )
    first_solar = integrate_coast(
        translate_to_sun(earth.final_state, 399, earth.end_et, cache),
        earth.end_et,
        mars_entry_epoch,
        mu_sun,
        10,
        max_step_s=86400.0,
    )
    mars_start = np.r_[mars, first_solar.final_state[6]]
    flyby = rollout_to_soi_exit(
        mars_start, mars_entry_epoch, arrival_epoch, 499, cache, soi_model=mars_model
    )
    second_solar = integrate_coast(
        translate_to_sun(flyby.final_state, 499, flyby.end_et, cache),
        flyby.end_et,
        arrival_epoch,
        mu_sun,
        10,
        max_step_s=86400.0,
    )
    mars_target_sun = translate_to_sun(mars_start, 499, mars_entry_epoch, cache)
    return {
        "arcs": (earth, first_solar, flyby, second_solar),
        "control_seed_n": np.zeros(3),
        "sigma_seed_n": 0.0,
        "mars_entry_cartesian_residual_si": first_solar.final_state[:6] - mars_target_sun[:6],
        "arrival_cartesian_residual_si": second_solar.final_state[:6] - target,
        "note": "Initial guess only. All target and phase-continuity equalities remain mandatory in the NLP.",
    }
