"""Osculating two-body launch diagnostics; input units are m, m/s, m^3/s^2."""

from __future__ import annotations
import numpy as np


def hyperbolic_diagnostics(state_si: np.ndarray, mu_m3_s2: float) -> dict:
    y = np.asarray(state_si, dtype=float)
    if (
        y.shape != (6,)
        or not np.all(np.isfinite(y))
        or (not np.isfinite(mu_m3_s2))
        or (mu_m3_s2 <= 0)
    ):
        raise ValueError("Expected a finite six-state and a positive finite GM")
    r, v = (y[:3], y[3:])
    radius = float(np.linalg.norm(r))
    if radius == 0:
        raise ValueError("Zero position cannot define an orbit")
    c3 = float(v @ v - 2 * mu_m3_s2 / radius)
    h = np.cross(r, v)
    hnorm = np.linalg.norm(h)
    if hnorm == 0 or c3 <= 0:
        raise ValueError("Outgoing asymptote requires a nonradial hyperbolic orbit")
    evec = np.cross(v, h) / mu_m3_s2 - r / radius
    e = float(np.linalg.norm(evec))
    if e <= 1:
        raise ValueError("Hyperbolic eccentricity must exceed one")
    p = evec / e
    q = np.cross(h / hnorm, p)
    outgoing = (-p + np.sqrt(e * e - 1) * q) / e
    outgoing /= np.linalg.norm(outgoing)
    return dict(
        radius_m=radius,
        speed_m_s=float(np.linalg.norm(v)),
        radial_velocity_m_s=float(r @ v / radius),
        c3_m2_s2=c3,
        c3_km2_s2=c3 / 1000000.0,
        vinfinity_m_s=float(np.sqrt(c3)),
        eccentricity=e,
        outgoing_vinfinity_si=(np.sqrt(c3) * outgoing).tolist(),
        rla_deg=float(np.degrees(np.arctan2(outgoing[1], outgoing[0])) % 360),
        dla_deg=float(np.degrees(np.arcsin(np.clip(outgoing[2], -1, 1)))),
    )
