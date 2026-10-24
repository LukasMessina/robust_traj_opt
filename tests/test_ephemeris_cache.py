"""Analytic interpolation/AD and scientific acceptance checks, without SPICE."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from psyche_mission_events_reconstruction.ephemeris_cache import DAY, EphemerisCache

ROOT = Path(__file__).resolve().parents[1]


class AnalyticInterpolationChecks(unittest.TestCase):
    def setUp(self):
        self.epochs = 750000000.0 + DAY * np.arange(9)
        self.rate = np.array([1200.0, -2500.0, 30.0, 0.1, -0.2, 0.03])
        self.offset = np.array([1e9, -2e9, 3e8, 1e4, -2e4, 300.0])
        values = self.offset + (self.epochs - self.epochs[0])[:, None] * self.rate
        self.cache = EphemerisCache(
            self.epochs,
            values[None],
            {
                "body_ids": [399],
                "valid_interval_et_tdb_s": self.epochs[[1, -2]].tolist(),
                "frame": "J2000",
                "origin": "Sun",
                "aberration_corrections": "NONE",
                "time_scale": "ET/TDB seconds since J2000",
                "units": "m, m/s",
                "interpolation": "natural cubic spline, six state components",
            },
        )

    def test_analytic_state_and_physical_time_derivatives(self):
        times = np.linspace(*self.cache.valid_interval, 37)
        expected = self.offset + (times - self.epochs[0])[:, None] * self.rate
        np.testing.assert_allclose(self.cache.state(399, times), expected, rtol=0, atol=2e-6)
        for f, order in zip(self.cache.casadi_functions(399), (0, 1, 2)):
            values = np.asarray(f.map(times.size)(times)).T
            expected_value = (
                expected
                if order == 0
                else np.tile(self.rate if order == 1 else np.zeros(6), (times.size, 1))
            )
            np.testing.assert_allclose(
                values, expected_value, rtol=0, atol=2e-6 if order == 0 else 1e-9
            )
            f.expand()(times[0])

    def test_domain_guards_and_serialisation(self):
        for epoch in (np.nan, np.inf, self.epochs[0], self.epochs[-1]):
            with self.assertRaises(ValueError):
                self.cache.state(399, epoch)
        fun = self.cache.casadi_functions(399)[0]
        self.assertTrue(np.isnan(np.asarray(fun(self.epochs[0] - DAY))).all())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ephemeris.npz"
            self.cache.save(path)
            restored = EphemerisCache.load(path)
            at = sum(self.cache.valid_interval) / 2
            np.testing.assert_array_equal(restored.state(399, at), self.cache.state(399, at))
            self.assertEqual(restored.metadata, self.cache.metadata)


class MissionCacheChecks(unittest.TestCase):
    def test_full_interval_validation_and_kernel_free_derivatives(self):
        output = ROOT / "psyche_mission_events_reconstruction/artifacts/ephemerides"
        report = json.loads((output / "validation.json").read_text(encoding="utf-8"))
        self.assertTrue(report["kernel_free_evaluation_verified"])
        cache = EphemerisCache.load(output / "earth_mars_ephemerides.npz")
        from hashlib import sha256

        self.assertEqual(
            sha256((output / "earth_mars_ephemerides.npz").read_bytes()).hexdigest(),
            report["cache_sha256"],
        )
        times = np.linspace(*cache.valid_interval, 81)
        for body in (399, 499):
            f, df, ddf = cache.casadi_functions(body)
            np.testing.assert_allclose(
                np.asarray(f.map(times.size)(times)).T, cache.state(body, times), rtol=0, atol=0.002
            )
            np.testing.assert_allclose(
                np.asarray(df.map(times.size)(times)).T,
                cache.derivative(body, times),
                rtol=0,
                atol=1e-6,
            )
            self.assertTrue(np.isfinite(np.asarray(ddf.map(times.size)(times))).all())
        for values in report["accepted_grid"]["validation"].values():
            self.assertGreater(values["count"], 10000)
            self.assertLess(values["max_position_error_m"], 0.1)
            self.assertLess(values["max_velocity_error_m_s"], 1e-5)
            self.assertLess(values["max_dposition_dt_minus_velocity_m_s"], 1e-4)


if __name__ == "__main__":
    unittest.main()
