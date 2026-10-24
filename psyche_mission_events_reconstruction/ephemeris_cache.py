"""Stored, differentiable planetary ephemerides; no SPICE calls at evaluation.

Natural cubic splines reproduce Tudat's cubic_spline_interpolation convention.
The build step samples Tudat/SPICE. This module needs only NumPy, SciPy and,
for symbolic evaluation, CasADi. Epochs are ET/TDB seconds since J2000; states
are geometric, Sun-centred, J2000, in m and m/s. A padded sampling interval
keeps the natural-spline endpoint conditions outside the usable interval.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline, make_interp_spline

DAY = 86400.0


@dataclass
class EphemerisCache:
    sample_epochs: np.ndarray
    sample_states: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.sample_epochs = np.array(self.sample_epochs, dtype=float, copy=True)
        self.sample_states = np.array(self.sample_states, dtype=float, copy=True)
        self.metadata = json.loads(json.dumps(self.metadata))
        t, y = (self.sample_epochs, self.sample_states)
        ids = self.metadata["body_ids"]
        if (
            t.ndim != 1
            or t.size < 4
            or (not np.all(np.isfinite(t)))
            or (not np.all(np.diff(t) > 0))
        ):
            raise ValueError("Sample epochs must be finite and strictly increasing")
        if (
            len(set(ids)) != len(ids)
            or y.shape != (len(ids), t.size, 6)
            or (not np.all(np.isfinite(y)))
        ):
            raise ValueError("Invalid body IDs or Cartesian sample data")
        lo, hi = self.metadata["valid_interval_et_tdb_s"]
        if not t[0] < lo < hi < t[-1]:
            raise ValueError("Usable interval must lie strictly inside the padded samples")
        required = {
            "frame": "J2000",
            "origin": "Sun",
            "aberration_corrections": "NONE",
            "time_scale": "ET/TDB seconds since J2000",
            "units": "m, m/s",
            "interpolation": "natural cubic spline, six state components",
        }
        if any((self.metadata.get(key) != value for key, value in required.items())):
            raise ValueError("Unsupported cache frame, units, time scale or interpolation")
        self.origin_et = float(t[0])
        self._splines = {
            int(body): make_interp_spline(
                (t - self.origin_et) / DAY, y[i], k=3, bc_type="natural", axis=0
            )
            for i, body in enumerate(ids)
        }
        self.sample_epochs.setflags(write=False)
        self.sample_states.setflags(write=False)

    @property
    def valid_interval(self) -> tuple[float, float]:
        return tuple(self.metadata["valid_interval_et_tdb_s"])

    def _checked_times(self, epochs):
        t = np.asarray(epochs, dtype=float)
        lo, hi = self.valid_interval
        if not np.all(np.isfinite(t)) or np.any(t < lo) or np.any(t > hi):
            raise ValueError("Epoch outside validated ephemeris interval; extrapolation forbidden")
        return (t - self.origin_et) / DAY

    def state(self, body: int, epochs) -> np.ndarray:
        return self._splines[body](self._checked_times(epochs))

    def derivative(self, body: int, epochs, order: int = 1) -> np.ndarray:
        """Derivative of the fitted state with respect to physical seconds.

        Velocity is fitted independently from position, just as in Tudat's
        state interpolation. d(position)/dt - fitted_velocity must therefore
        be checked, and is reported separately from SPICE interpolation error.
        """
        if order not in (1, 2):
            raise ValueError("Only first and second time derivatives are supported")
        return self._splines[body](self._checked_times(epochs), nu=order) / DAY**order

    def casadi_functions(self, body: int):
        """Return state, dstate/dET, d2state/dET2 functions with analytic AD.

        NLP callers MUST bound every variable epoch to valid_interval. Values
        outside the *sample* domain become NaN, never a silent extrapolation.
        Padding remains evaluable for line-search trials. The optimiser's
        feasible epoch bounds must still use the narrower validated interval.
        """
        import casadi as ca

        spline: BSpline = self._splines[body]
        base = ca.Function.bspline(
            f"body_{body}_spline",
            [spline.t.tolist()],
            spline.c.ravel().tolist(),
            [3],
            6,
            {"never_inline": True},
        )
        et = ca.MX.sym("et_tdb_s")
        value = base((et - self.origin_et) / DAY)
        in_samples = ca.logic_and(et >= self.sample_epochs[0], et <= self.sample_epochs[-1])
        value = ca.if_else(in_samples, value, ca.DM(np.full(6, np.nan)))
        derivative = ca.jacobian(value, et)
        second = ca.jacobian(derivative, et)
        return tuple(
            (
                ca.Function(f"body_{body}_{name}", [et], [expr])
                for name, expr in (("state", value), ("rate", derivative), ("second_rate", second))
            )
        )

    def tudat_interpolator(self, body: int):
        """Build the equivalent native Tudat interpolator from the stored table.

        No kernels are required. This low-level object accepts padded epochs;
        use this class's checked state() for ordinary numerical evaluation.
        """
        from tudatpy.math import interpolators

        index = self.metadata["body_ids"].index(body)
        settings = interpolators.cubic_spline_interpolation(
            boundary_interpolation=interpolators.throw_exception_at_boundary
        )
        return interpolators.create_one_dimensional_vector_interpolator(
            dict(zip(self.sample_epochs, self.sample_states[index])), settings
        )

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                sample_epochs=self.sample_epochs,
                sample_states=self.sample_states,
                metadata=np.array(json.dumps(self.metadata)),
            )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> "EphemerisCache":
        with np.load(path, allow_pickle=False) as archive:
            return cls(
                archive["sample_epochs"],
                archive["sample_states"],
                json.loads(str(archive["metadata"])),
            )
