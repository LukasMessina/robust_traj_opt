"""Verify pinned mission inputs or reproduce them in a new artifact directory."""

import argparse
import json
import os
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="New directory for a reconstruction; omitted means verify pinned inputs",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Re-query the pinned planetary kernels and validate interpolants",
    )
    args = parser.parse_args()
    if args.rebuild_cache and not args.output:
        parser.error("--rebuild-cache requires a new --output directory")
    root = Path(__file__).resolve().parents[1]
    pinned = Path(__file__).resolve().parent / "artifacts"
    os.chdir(root)
    if args.output:
        destination = args.output.resolve()
        if destination.exists():
            parser.error(
                "Choose a new output directory; pinned inputs and previous reconstructions are preserved"
            )
        destination.mkdir(parents=True)
        os.environ["PSYCHE_RECONSTRUCTION_DATA"] = str(destination)
        from .tip import run as tip
        from .build_ephemeris_cache import run as build_cache
        from .ephemeris_cache import EphemerisCache
        from .kernels import sha256
        from .soi_reconstruction import run as soi
        from .reconstruct_arrival import run as arrival
        from .seed_solar_phases import run as timing

        tip()
        if args.rebuild_cache:
            build_cache(destination / "ephemerides", mars_kernel="mar097.bsp")
        else:
            shutil.copytree(pinned / "ephemerides", destination / "ephemerides")
            # Numerical cache arrays remain identical; only provenance follows
            # the newly extracted TIP document and its implementation record.
            path = destination / "ephemerides/earth_mars_ephemerides.npz"
            cache = EphemerisCache.load(path)
            cache.metadata["tip_result_sha256"] = sha256(destination / "tip/tip_result.json")
            cache.save(path)
            validation_path = destination / "ephemerides/validation.json"
            validation = json.loads(validation_path.read_text())
            validation["cache_sha256"] = sha256(path)
            validation["accepted_grid"] = cache.metadata
            validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
        soi()
        arrival()
        timing()
    from .verify import verify

    print(json.dumps(verify(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
