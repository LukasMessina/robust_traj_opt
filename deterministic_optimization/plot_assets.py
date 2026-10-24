"""Download and pin cartographic inputs used only for post-solve figures.

Run explicitly: python -m deterministic_optimization.plot_assets
No network access is performed by plotting or optimisation.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "data/planet_textures"
EARTH_PCK = "earth_000101_270119_261023.bpc"
NAIF = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(item):
    name, url = item
    path = ASSETS / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + ".part")
        with urlopen(url, timeout=120) as source, temporary.open("wb") as target:
            while block := source.read(1024 * 1024):
                target.write(block)
        temporary.replace(path)
    print(name, flush=True)
    return dict(file=name, url=url, sha256=digest(path), bytes=path.stat().st_size)


def main():
    urls = [
        (
            "earth_october_5400.jpg",
            "https://assets.science.nasa.gov/content/dam/science/esd/eo/images/bmng/bmng-base/october/world.200410.3x5400x2700.jpg",
        ),
        ("pck00011.tpc", NAIF + "pck00011.tpc"),
        (EARTH_PCK, NAIF + EARTH_PCK),
        (EARTH_PCK.replace(".bpc", ".cmt"), NAIF + EARTH_PCK.replace(".bpc", ".cmt")),
    ]
    for hemisphere in ("n", "s"):
        for longitude in (225, 315, 45, 135):
            name = f"mars45{hemisphere}{longitude:03d}.png"
            urls.append(("tiles/" + name, "https://mars.asu.edu/data/mdim_color/large/" + name))
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, urls))
    # Preserve geographic boundaries: columns -180..-90, -90..0, 0..90, 90..180 E.
    mosaic = Image.new("RGB", (8192, 4096))
    for row, hemisphere in enumerate(("n", "s")):
        for column, longitude in enumerate((225, 315, 45, 135)):
            with Image.open(ASSETS / f"tiles/mars45{hemisphere}{longitude:03d}.png") as tile:
                mosaic.paste(
                    tile.convert("RGB").resize((2048, 2048), Image.Resampling.LANCZOS),
                    (column * 2048, row * 2048),
                )
    mosaic.save(ASSETS / "mars_viking_8192.jpg", quality=95, subsampling=0)
    metadata = dict(
        source_files=records,
        earth=dict(
            file="earth_october_5400.jpg",
            size=[5400, 2700],
            source="https://science.nasa.gov/earth/earth-observatory/blue-marble-next-generation/base-map/",
            coordinates="WGS84 geodetic latitude; west edge -180 E, north edge +90; eastward columns, southward rows",
            credit="NASA Earth Observatory / Blue Marble Next Generation, October 2004 composite",
            frame="ITRF93",
            orientation_kernel=EARTH_PCK,
        ),
        mars=dict(
            file="mars_viking_8192.jpg",
            size=[8192, 4096],
            source="https://mars.asu.edu/data/mdim_color/",
            coordinates="Planetocentric latitude; west edge -180 E, north edge +90; eastward columns, southward rows",
            credit="NASA Viking Orbiter / USGS MDIM 1.0, hosted by ASU",
            frame="IAU_MARS",
            orientation_kernel="pck00011.tpc",
            sha256=digest(ASSETS / "mars_viking_8192.jpg"),
        ),
        purpose="Static cartographic illustrations, not observations at the mission epochs; no effect on dynamics",
    )
    (ASSETS / "manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
