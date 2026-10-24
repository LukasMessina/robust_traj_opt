"""Explicit kernel provenance, binary coverage, and isolated SPICE sessions.

TudatPy supplies mission states in SI. SpiceyPy inspects SPK metadata and
cross-checks queries. Both kernel pools are managed explicitly. Use this
context in a dedicated process: SPICE kernel pools are global, not thread-local.
"""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
from urllib.request import urlopen
import numpy as np
import spiceypy as raw_spice
from tudatpy.interface import spice

ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = ROOT / "data/jpl_kernels"
GENERIC = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/"
MISSION = "https://naif.jpl.nasa.gov/pub/naif/PSYCHE/kernels/spk/"
FRAME = "J2000"
CORRECTION = "NONE"


@dataclass(frozen=True)
class Kernel:
    name: str
    url: str
    sha256: str


TIP_KERNELS = (
    Kernel(
        "naif0012.tls",
        GENERIC + "lsk/naif0012.tls",
        "678e32bdb5a744117a467cd9601cd6b373f0e9bc9bbde1371d5eee39600a039b",
    ),
    Kernel(
        "gm_de440.tpc",
        GENERIC + "pck/gm_de440.tpc",
        "924ddf4fb9ead9fe8a1aa55780bcabde40b09d00065d58226e24b68d8092f140",
    ),
    Kernel(
        "de440s.bsp",
        GENERIC + "spk/planets/de440s.bsp",
        "c1c7feeab882263fc493a9d5a5b2ddd71b54826cdf65d8d17a76126b260a49f2",
    ),
    Kernel(
        "psyche_rec_231013-231018_v1.bsp",
        MISSION + "psyche_rec_231013-231018_v1.bsp",
        "74f8e2e478df739ce2e6c57f4287079fbefc987b02f7c9350e4567a3ec4c0e10",
    ),
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def kernel_paths(download: bool = False) -> list[Path]:
    """Check pinned bytes; network access is opt-in and never part of a solve."""
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in TIP_KERNELS:
        path = KERNEL_DIR / spec.name
        if not path.is_file():
            if not download:
                raise FileNotFoundError(f"Missing {path}. Run with --download first.")
            partial = path.with_suffix(path.suffix + ".part")
            with urlopen(spec.url, timeout=60) as source, partial.open("wb") as dest:
                while chunk := source.read(1024 * 1024):
                    dest.write(chunk)
            if sha256(partial) != spec.sha256:
                raise ValueError(f"Checksum mismatch in download {partial}; not installed.")
            partial.replace(path)
        if sha256(path) != spec.sha256:
            raise ValueError(f"Kernel checksum mismatch: {path}")
        paths.append(path)
    return paths


@contextmanager
def loaded_kernels(paths: list[Path]):
    spice.clear_kernels()
    raw_spice.kclear()
    try:
        for path in paths:
            raw_count = raw_spice.ktotal("ALL")
            spice.load_kernel(str(path.resolve()))
            if raw_spice.ktotal("ALL") == raw_count:
                raw_spice.furnsh(str(path.resolve()))
        yield
    finally:
        spice.clear_kernels()
        raw_spice.kclear()


def epoch_record(et: float) -> dict:
    return {
        "et_tdb_s": float(et),
        "utc": raw_spice.et2utc(float(et), "ISOC", 6) + "Z",
        "tdb_calendar": raw_spice.timout(float(et), "YYYY-MM-DD HR:MN:SC.###### ::TDB"),
    }


def state_si(target: int, observer: int, et: float) -> np.ndarray:
    state = np.asarray(
        spice.get_body_cartesian_state_at_epoch(
            target_body_name=str(target),
            observer_body_name=str(observer),
            reference_frame_name=FRAME,
            aberration_corrections=CORRECTION,
            ephemeris_time=float(et),
        ),
        dtype=float,
    ).reshape(6)
    if not np.all(np.isfinite(state)):
        raise ValueError("Nonfinite SPICE state")
    return state


def covers(intervals: list[dict], epoch: float) -> bool:
    return any((w["start"]["et_tdb_s"] <= epoch <= w["end"]["et_tdb_s"] for w in intervals))


def inspect_spk(path: Path) -> dict:
    """Inspect every segment; coverage is per target, not inferred from filename."""
    coverage = {}
    for target in raw_spice.spkobj(str(path)):
        window = raw_spice.spkcov(str(path), int(target))
        coverage[str(int(target))] = [
            {"start": epoch_record(a), "end": epoch_record(b)}
            for a, b in (raw_spice.wnfetd(window, i) for i in range(raw_spice.wncard(window)))
        ]
    handle = raw_spice.dafopr(str(path))
    try:
        comments = []
        done = False
        while not done:
            n, lines, done = raw_spice.dafec(handle, 100, 1000)
            comments.extend(lines[:n])
        raw_spice.dafbfs(handle)
        segments = []
        while raw_spice.daffna():
            dc, ic = raw_spice.dafus(raw_spice.dafgs(), 2, 6)
            segments.append(
                dict(
                    start=epoch_record(dc[0]),
                    end=epoch_record(dc[1]),
                    target=int(ic[0]),
                    centre=int(ic[1]),
                    frame_id=int(ic[2]),
                    frame=raw_spice.frmnam(int(ic[2])),
                    spk_type=int(ic[3]),
                    identifier=raw_spice.dafgn(),
                )
            )
    finally:
        raw_spice.dafcls(handle)
    return dict(file=path.name, coverage=coverage, segments=segments, comments="\n".join(comments))


def selected_segment(target: int, et: float) -> dict:
    handle, description, identifier = raw_spice.spksfs(target, float(et), 80)
    _, ic = raw_spice.dafus(description, 2, 6)
    loaded = [raw_spice.kdata(i, "SPK") for i in range(raw_spice.ktotal("SPK"))]
    filenames = sorted({entry[0] for entry in loaded if entry[3] == handle})
    if len(filenames) != 1:
        raise ValueError("Cannot identify unique SPK filename for selected handle")
    return dict(
        file=str(Path(filenames[0]).resolve()),
        target=int(ic[0]),
        centre=int(ic[1]),
        frame_id=int(ic[2]),
        spk_type=int(ic[3]),
        identifier=identifier,
    )
