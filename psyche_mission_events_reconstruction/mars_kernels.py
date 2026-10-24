"""Pinned reconstructed Mars encounter kernels and coverage checks."""

from .build_ephemeris_cache import MAR097_KERNEL
from .kernels import KERNEL_DIR, MISSION, TIP_KERNELS, Kernel, sha256

SPACECRAFT_KERNEL = Kernel(
    "psyche_rec_260501-260713_260804_v1.bsp",
    MISSION + "psyche_rec_260501-260713_260804_v1.bsp",
    "65a80b4e8d72ac6bdf8177df54525e39de5f812e05c73d7261a6b3e7673476b6",
)
KERNELS = (TIP_KERNELS[0], TIP_KERNELS[1], MAR097_KERNEL, TIP_KERNELS[2], SPACECRAFT_KERNEL)


def paths_verified():
    paths = [KERNEL_DIR / spec.name for spec in KERNELS]
    for spec, path in zip(KERNELS, paths):
        if not path.is_file() or sha256(path) != spec.sha256:
            raise ValueError(f"Missing or unverified kernel: {path}")
    return paths


def ensure_interval(metadata, target, start, end):
    windows = metadata["coverage"][str(target)]
    if not any((w["start"]["et_tdb_s"] <= start and w["end"]["et_tdb_s"] >= end for w in windows)):
        raise ValueError(f"No continuous coverage of target {target} for requested interval")
