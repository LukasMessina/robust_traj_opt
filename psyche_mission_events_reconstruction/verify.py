"""Coverage, source hashes and numerical checks for the retained mission inputs."""

import json
from pathlib import Path
import numpy as np
from .paths import DATA_ROOT, resolve_input
from .kernels import (
    TIP_KERNELS,
    kernel_paths,
    loaded_kernels,
    state_si,
    sha256,
    inspect_spk,
    covers,
)
from .mars_kernels import paths_verified
from .reference_kernel import REFERENCE, checked_path
from .arrival_boundary import load_arrival_boundary
from .ephemeris_cache import EphemerisCache
from .mission_soi import fixed_mission_soi


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify():
    tip = read(DATA_ROOT / "tip/tip_result.json")
    mars = read(DATA_ROOT / "soi/mars_entry_result.json")
    arrival = load_arrival_boundary()
    cache_path = resolve_input(mars["planetary_cache"]["path"])
    assert sha256(cache_path) == mars["planetary_cache"]["sha256"]
    cache = EphemerisCache.load(cache_path)
    assert cache.metadata["tip_result_sha256"] == sha256(DATA_ROOT / "tip/tip_result.json")
    checks = []
    for label, paths, body, origin, epoch, expected in (
        (
            "TIP",
            kernel_paths(),
            -255,
            399,
            tip["epochs"]["tip"]["et_tdb_s"],
            tip["earth_relative_state_si"],
        ),
        (
            "Mars entry",
            paths_verified(),
            -255,
            499,
            mars["epoch"]["et_tdb_s"],
            mars["mars_relative_state_si"],
        ),
        (
            "Asteroid arrival",
            [checked_path(TIP_KERNELS[0]), checked_path(REFERENCE)],
            2000016,
            10,
            arrival["epoch"]["et_tdb_s"],
            arrival["sun_relative_state_si"],
        ),
    ):
        with loaded_kernels(paths):
            spk = inspect_spk(paths[-1])
            assert covers(spk["coverage"][str(body)], epoch)
            actual = state_si(body, origin, epoch)
            np.testing.assert_array_equal(actual, expected)
        checks.append(
            dict(event=label, et_tdb_s=epoch, state_exactly_reproduced=True, coverage_verified=True)
        )
    radius = fixed_mission_soi(499, cache.metadata["mu_m3_s2"]).radius_rate(None)[0]
    residual = float(np.linalg.norm(mars["mars_relative_state_si"][:3]) - radius)
    assert abs(residual) < 0.1
    c3 = (
        np.dot(tip["earth_relative_state_si"][3:], tip["earth_relative_state_si"][3:])
        - 2 * tip["earth_gm_m3_s2"] / np.linalg.norm(tip["earth_relative_state_si"][:3])
    ) / 1e6
    assert abs(c3 - 34.0) < 0.25
    seed = read(DATA_ROOT / "rollout_timing/result.json")
    for name, path in (
        ("tip_sha256", DATA_ROOT / "tip/tip_result.json"),
        ("mars_entry_sha256", DATA_ROOT / "soi/mars_entry_result.json"),
        ("arrival_sha256", DATA_ROOT / "arrival/arrival_result.json"),
    ):
        assert seed["provenance"][name] == sha256(path)
    result = dict(
        verified=True,
        data_directory=str(DATA_ROOT),
        events=checks,
        mars_soi_residual_m=residual,
        tip_c3_km2_s2=float(c3),
        nominal_c3_km2_s2=34.0,
        cache_sha256=sha256(cache_path),
        cache_interpolation="natural cubic state-component splines",
        cache_step_s=cache.metadata["sample_step_s"],
        qualification="Numerical and provenance consistency; not an ephemeris uncertainty bound.",
    )
    (DATA_ROOT / "verification.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
