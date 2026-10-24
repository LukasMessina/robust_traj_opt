"""JSON serialization and provenance helpers."""

import hashlib
import json
from pathlib import Path
import numpy as np
import shutil


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, data):
    Path(path).write_text(json.dumps(jsonable(data), indent=2), encoding="utf-8")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def snapshot_sources(output):
    root = Path(__file__).resolve().parents[1]
    sources = [root / name for name in ("deterministic_traj_opt.py", "integrator.py", "plotter.py")]
    for package in ("deterministic_optimization", "psyche_mission_events_reconstruction"):
        sources.extend(sorted((root / package).glob("*.py")))
    hashes = {}
    for source in sources:
        relative = source.relative_to(root)
        target = Path(output) / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative.as_posix()] = digest(source)
    return hashes
