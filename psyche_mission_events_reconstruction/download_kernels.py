"""Download only the pinned mission inputs, outside optimisation."""

from pathlib import Path
from urllib.request import urlopen
from .kernels import KERNEL_DIR, TIP_KERNELS, sha256
from .build_ephemeris_cache import MAR097_KERNEL
from .mars_kernels import SPACECRAFT_KERNEL
from .reference_kernel import REFERENCE


def download():
    KERNEL_DIR.mkdir(parents=True, exist_ok=True)
    for spec in (*TIP_KERNELS, MAR097_KERNEL, SPACECRAFT_KERNEL, REFERENCE):
        target = KERNEL_DIR / spec.name
        if target.exists():
            if sha256(target) != spec.sha256:
                raise ValueError(f"Existing kernel checksum mismatch: {target}")
            continue
        partial = target.with_suffix(target.suffix + ".part")
        try:
            with urlopen(spec.url, timeout=60) as source, partial.open("wb") as stream:
                while chunk := source.read(1024 * 1024):
                    stream.write(chunk)
            if sha256(partial) != spec.sha256:
                raise ValueError(f"Download checksum mismatch: {spec.url}")
            partial.replace(target)
        finally:
            partial.unlink(missing_ok=True)


if __name__ == "__main__":
    download()
