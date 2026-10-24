"""Kernel-free loading of the fixed asteroid rendezvous boundary."""

from __future__ import annotations
from psyche_mission_events_reconstruction.paths import DATA_ROOT, resolve_input
import json
from pathlib import Path
import numpy as np

ARRIVAL_TDB = "2029-08-01 12:00:00.000000"
ARRIVAL_TDB_INPUT = "2029-08-01 12:00:00 TDB"
ARRIVAL_ET = 933552000.0
TARGET_BODY_ID = 2000016
OUT = DATA_ROOT / "arrival"
ACTIVE_ARRIVAL_BOUNDARY = OUT / "arrival_result.json"


def load_arrival_boundary(path=ACTIVE_ARRIVAL_BOUNDARY):
    """Fail closed if the target is missing, unvalidated or uses another frame.

    The authoritative epoch is TDB, independent of future leap seconds.
    UTC is a display conversion only. No SPICE calls occur here.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "status": "verified fixed boundary extracted",
        "constrained_spacecraft_id": -255,
        "target_body_id": TARGET_BODY_ID,
        "origin_id": 10,
        "frame": "J2000",
        "fixed_epoch_scale": "TDB",
        "aberration_corrections": "NONE",
        "units": "m, m/s",
        "source_kind": "asteroid ephemeris prediction",
    }
    if any((data.get(key) != value for key, value in expected.items())):
        raise ValueError("Arrival target identity, frame, units or provenance mismatch")
    if data["epoch"]["tdb_calendar"] != ARRIVAL_TDB:
        raise ValueError("Arrival TDB calendar differs from the requested hard epoch")
    if (
        not np.isfinite(data["epoch"]["et_tdb_s"])
        or abs(data["epoch"]["et_tdb_s"] - ARRIVAL_ET) > 1e-07
    ):
        raise ValueError("Arrival ET disagrees with the fixed TDB epoch")
    state = np.asarray(data["sun_relative_state_si"], dtype=float)
    if state.shape != (6,) or not np.all(np.isfinite(state)):
        raise ValueError("Arrival requires six finite position/velocity components")
    if data.get("validation", {}).get("passed") is not True:
        raise ValueError("Arrival extraction has not passed validation")
    intervals = data["target_binary_coverage"]
    et = data["epoch"]["et_tdb_s"]
    if not any((w["start"]["et_tdb_s"] <= et <= w["end"]["et_tdb_s"] for w in intervals)):
        raise ValueError("Arrival epoch is outside binary asteroid coverage")
    return data
