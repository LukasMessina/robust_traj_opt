"""Extract the fixed asteroid-centre target; never use a spacecraft arrival state."""

from dataclasses import asdict
import json
import numpy as np
import spiceypy as raw
from tudatpy.interface import spice
from .arrival_boundary import (
    ARRIVAL_ET,
    ARRIVAL_TDB_INPUT,
    TARGET_BODY_ID,
    OUT,
    load_arrival_boundary,
)
from .reference_kernel import REFERENCE, checked_path, require_coverage, selected_with_bounds
from .kernels import TIP_KERNELS, state_si, epoch_record, inspect_spk, loaded_kernels, sha256
from .mission_soi import ACTIVE_MARS_BOUNDARY
from .ephemeris_cache import EphemerisCache
from .paths import resolve_input


def run():
    lsk, reference, de440 = map(checked_path, (TIP_KERNELS[0], REFERENCE, TIP_KERNELS[2]))
    with loaded_kernels([lsk, reference]):
        et = float(spice.convert_date_string_to_ephemeris_time(ARRIVAL_TDB_INPUT))
        if et != ARRIVAL_ET:
            raise ValueError("Unexpected TDB epoch conversion")
        metadata = inspect_spk(reference)
        require_coverage(metadata, TARGET_BODY_ID, et)
        state = state_si(TARGET_BODY_ID, 10, et)
        direct = raw.spkezr(str(TARGET_BODY_ID), et, "J2000", "NONE", "10")[0] * 1000.0
        epoch = epoch_record(et)
        segment = selected_with_bounds(TARGET_BODY_ID, et)
        if (segment["target"], segment["centre"], segment["frame_id"], segment["spk_type"]) != (
            TARGET_BODY_ID,
            10,
            1,
            2,
        ):
            raise ValueError("Unexpected asteroid segment")
        difference = (
            state_si(TARGET_BODY_ID, 10, et + 10.0)[:3]
            - state_si(TARGET_BODY_ID, 10, et - 10.0)[:3]
        ) / 20.0
        reverse = state_si(10, TARGET_BODY_ID, et)
    with loaded_kernels([lsk, de440, reference]):
        alternate = state_si(TARGET_BODY_ID, 10, et)
    mars = json.loads(ACTIVE_MARS_BOUNDARY.read_text(encoding="utf-8"))
    cache_record = mars["planetary_cache"]
    cache_path = resolve_input(cache_record["path"])
    if sha256(cache_path) != cache_record["sha256"]:
        raise ValueError("Planetary cache hash mismatch")
    cache = EphemerisCache.load(cache_path)
    if not cache.valid_interval[0] <= et <= cache.valid_interval[1]:
        raise ValueError("Arrival outside cache validity interval")
    passed = (
        np.array_equal(state, direct)
        and np.array_equal(state, -reverse)
        and np.array_equal(state, alternate)
        and np.linalg.norm(difference - state[3:]) < 1e-3
        and not raw.ktotal("ALL")
        and not spice.get_total_count_of_kernels_loaded()
    )
    if not passed:
        raise ValueError("Arrival consistency validation failed")
    result = dict(
        schema_version=3,
        status="verified fixed boundary extracted",
        constrained_spacecraft_id=-255,
        target_body_id=TARGET_BODY_ID,
        target_name="16 Psyche asteroid centre",
        origin_id=10,
        frame="J2000",
        aberration_corrections="NONE",
        units="m, m/s",
        time_scale="ET/TDB seconds since J2000",
        fixed_epoch_scale="TDB",
        source_kind="asteroid ephemeris prediction",
        epoch=epoch,
        sun_relative_state_si=state.tolist(),
        target_binary_coverage=metadata["coverage"][str(TARGET_BODY_ID)],
        selected_target_segment=segment,
        kernel=asdict(REFERENCE),
        load_order=[asdict(TIP_KERNELS[0]), asdict(REFERENCE)],
        source_piece="sb_psyche_ssd_180411.bsp (merged kernel comments)",
        planetary_cache=cache_record,
        asteroid_relative_terminal_state_si=[0.0] * 6,
        validation=dict(
            passed=bool(passed),
            tudat_vs_spiceypy_max_component_difference=float(np.max(abs(state - direct))),
            dposition_dt_minus_velocity_m_s=float(np.linalg.norm(difference - state[3:])),
            finite_difference_step_s=10.0,
            kernel_pools_cleared=True,
        ),
        limitations=[
            "User-selected asteroid-centre rendezvous under Sun-only final dynamics, not a physical captured orbit.",
            "2018 asteroid ephemeris embedded in the 2023 reference; not claimed to be the latest orbit solution.",
            "No mass is inferred from an SPK; final spacecraft mass is free.",
            "UTC display uses naif0012; the fixed boundary is TDB.",
        ],
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "arrival_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    load_arrival_boundary()
    return result
