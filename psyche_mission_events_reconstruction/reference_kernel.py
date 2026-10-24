"""Pinned asteroid kernel and authoritative binary segment checks."""

from .kernels import KERNEL_DIR, MISSION, Kernel, covers, epoch_record, selected_segment, sha256
import spiceypy as raw

REFERENCE = Kernel(
    "psyche_ref_231013-311101_231016_v1_merged.bsp",
    MISSION + "psyche_ref_231013-311101_231016_v1_merged.bsp",
    "ac66af58c0e02230e5f652ae323df64c784dc1d4bd242f65fc5aa467779d3958",
)


def checked_path(spec):
    path = KERNEL_DIR / spec.name
    if not path.is_file() or sha256(path) != spec.sha256:
        raise ValueError(f"Missing or changed pinned kernel: {path}")
    return path


def require_coverage(metadata, target, et):
    if not covers(metadata["coverage"].get(str(target), []), et):
        raise ValueError(f"No binary coverage for {target} at ET {et}; extrapolation forbidden")


def selected_with_bounds(target, et):
    selection = selected_segment(target, et)
    _, descriptor, _ = raw.spksfs(target, et, 80)
    dc, _ = raw.dafus(descriptor, 2, 6)
    return dict(selection, start=epoch_record(dc[0]), end=epoch_record(dc[1]))
