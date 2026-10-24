"""Reconstruction artifacts are independent of optimisation result folders."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINNED_DATA = Path(__file__).resolve().parent / "artifacts"
DATA_ROOT = Path(os.environ.get("PSYCHE_RECONSTRUCTION_DATA", PINNED_DATA)).resolve()


def resolve_input(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path
