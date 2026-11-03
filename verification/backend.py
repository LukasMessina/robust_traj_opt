"""JAX device of the post-solve validation, matching the solve's execution mode.

Call `select(mode)` before anything imports JAX:

* ``cpu``: JAX on the CPU, split over every core (JAX_NUM_CPU_DEVICES, used
  by `split_over_devices` for the Monte Carlo batch);
* ``gpu``: JAX on CUDA, without preallocation, since the Julia solve that
  launched the validation still holds a CUDA context.

`check(mode)` after importing JAX fails instead of silently falling back.
"""

import os

MODES = ("cpu", "gpu")


def select(mode):
    if mode not in MODES:
        raise ValueError(f"execution mode must be one of {MODES}, not {mode!r}")
    if mode == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ.setdefault(
            "JAX_NUM_CPU_DEVICES",
            str(
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else (os.cpu_count() or 1)
            ),
        )
    else:
        os.environ["JAX_PLATFORMS"] = "cuda"
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def check(mode):
    import jax

    backend = jax.default_backend()
    if backend != mode:
        raise RuntimeError(f"{mode} validation requested, but JAX runs on {backend}")
    return f"{backend} ({jax.device_count()} device{'s' if jax.device_count() > 1 else ''})"
