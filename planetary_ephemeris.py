"""Explicit JPL kernels and Tudat SI states for the Earth--Mars benchmark.

See https://py.api.tudat.space/en/latest/interface/spice.html .
SPICE uses internal barycentric links, but every returned planetary state is
the body CENTER relative to the Sun, in J2000, with no aberration correction.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

import numpy as np

KERNEL_DIR = Path(__file__).resolve().parent / "data" / "jpl_kernels"
BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
# Later SPKs take precedence. MAR099s includes DE440 segments, so DE442s MUST
# follow it. MAR099 retains only the Mars-center/satellite offsets in the chain.
KERNELS = (
    ("naif0012.tls", "lsk/naif0012.tls"),
    ("gm_de440.tpc", "pck/gm_de440.tpc"),
    ("mar099s.bsp", "spk/satellites/mar099s.bsp"),
    ("de442s.bsp", "spk/planets/de442s.bsp"),
)
DEPARTURE_TDB = "2024 AUG 11 00:00:00 TDB"
TOF_DAYS = 500.0


def ensure_kernels() -> list[Path]:
    """Download once, using atomic replacement so interrupted files aren't used."""
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, relative_url in KERNELS:
        path = KERNEL_DIR / name
        if not path.is_file():
            print(f"Downloading JPL kernel {name}", flush=True)
            temporary = path.with_suffix(path.suffix + ".part")
            with urlopen(BASE_URL + relative_url, timeout=120) as source, temporary.open("wb") as destination:
                while chunk := source.read(1024 * 1024):
                    destination.write(chunk)
            temporary.replace(path)
        paths.append(path)
    return paths


def load_ephemeris():
    from tudatpy.interface import spice

    paths = ensure_kernels()
    # Explicit kernel set avoids Tudat's default INPOP/NOE ephemerides and GMs.
    spice.clear_kernels()
    for path in paths:
        spice.load_kernel(str(path))
    return spice, paths


def body_state(spice, body: str, epoch: float) -> np.ndarray:
    return np.asarray(spice.get_body_cartesian_state_at_epoch(
        target_body_name=body,
        observer_body_name="Sun",
        reference_frame_name="J2000",
        aberration_corrections="NONE",
        ephemeris_time=float(epoch),
    ), dtype=float).reshape(6)


def earth_mars_data() -> dict:
    """Boundary states and plotting tracks, in meters and meters/second."""
    import tudatpy

    spice, paths = load_ephemeris()
    t0 = float(spice.convert_date_string_to_ephemeris_time(DEPARTURE_TDB))
    tf = t0 + TOF_DAYS * 86400.0
    # This command returns m^3/s^2 (unlike raw SPICE bodvrd's km^3/s^2).
    mu_sun = float(spice.get_body_gravitational_parameter("Sun"))
    if not 1.3e20 < mu_sun < 1.4e20:
        raise ValueError("Unexpected Tudat solar GM or units.")
    from scipy.optimize import brentq

    # Trace the actual ephemeris through one revolution. Measure accumulated
    # longitude in Earth's departure orbital plane only to locate the endpoint;
    # all plotted positions remain direct Sun/J2000 ephemeris queries.
    earth0 = body_state(spice, "Earth", t0)
    radial = earth0[:3] / np.linalg.norm(earth0[:3])
    normal = np.cross(earth0[:3], earth0[3:])
    normal /= np.linalg.norm(normal)
    transverse = np.cross(normal, radial)
    times = np.linspace(t0, tf, 1001)
    search_positions = np.column_stack([body_state(spice, "Earth", t)[:3] for t in times])
    angles = np.unwrap(np.arctan2(transverse @ search_positions, radial @ search_positions))
    crossing = np.flatnonzero(angles >= 2*np.pi)
    if not crossing.size:
        raise ValueError("Earth ephemeris sampling did not bracket one revolution.")
    index = int(crossing[0])
    def longitude(epoch):
        position = body_state(spice, "Earth", epoch)[:3]
        return np.arctan2(transverse @ position, radial @ position)
    revolution_end = brentq(longitude, times[index-1], times[index], xtol=1e-6)
    earth_times = np.linspace(t0, revolution_end, 1001)
    earth = np.column_stack([body_state(spice, "Earth", t) for t in earth_times])
    mars = np.column_stack([body_state(spice, "Mars", t) for t in times])
    return {
        "epoch_start_tdb_s": t0, "epoch_end_tdb_s": tf,
        "departure_tdb": DEPARTURE_TDB,
        "arrival_tdb": "2025 DEC 24 00:00:00 TDB",
        "tof_days": TOF_DAYS, "origin": "Sun", "frame": "J2000",
        "aberration_corrections": "NONE", "body_ids": {"Earth": 399, "Mars": 499, "Sun": 10},
        "mu_sun_m3_s2": mu_sun,
        "gm_source": "spice.get_body_gravitational_parameter('Sun'); gm_de440.tpc",
        "tudatpy_version": getattr(tudatpy, "__version__", "unknown"),
        "kernel_load_order": [
            {"name": p.name, "url": BASE_URL + relative,
             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p, (_, relative) in zip(paths, KERNELS)
        ],
        "earth_departure_state_si": earth[:, 0].tolist(),
        "mars_arrival_state_si": mars[:, -1].tolist(),
        "earth_track_times_days": ((earth_times-t0)/86400.0).tolist(),
        "mars_track_times_days": ((times-t0)/86400.0).tolist(),
        "earth_track_duration_days": (revolution_end-t0)/86400.0,
        "earth_track_extent": "one 2pi longitude revolution in departure orbital plane",
        "earth_track_si": earth.tolist(), "mars_track_si": mars.tolist(),
    }


if __name__ == "__main__":
    ensure_kernels()
