"""Analytic event geometry independent of mission SPK data."""

import unittest

import numpy as np

from psyche_mission_events_reconstruction.soi_events import (
    LaplaceSOI,
    find_soi_crossings,
    refine_bracket,
)


class SOIGeometryChecks(unittest.TestCase):
    def test_two_crossings_with_known_times_and_signs(self):
        model = LaplaceSOI(1.0, 1e5)  # radius = 200*(1e-5)^.4 = 2
        planet = lambda t: np.array([200.0, 0, 0, 0, 0, 0])
        spacecraft = lambda t: np.array([t - 5, 1.0, 0, 1.0, 0, 0])
        roots = find_soi_crossings(spacecraft, planet, model, 0.0, 10.0, step_s=0.7)
        self.assertEqual([r["direction"] for r in roots], ["entry", "exit"])
        np.testing.assert_allclose(
            [r["et_tdb_s"] for r in roots], [5 - np.sqrt(3), 5 + np.sqrt(3)], rtol=0, atol=1e-8
        )
        for root in roots:
            self.assertLess(abs(root["residual_m"]), 1e-8)
            self.assertLessEqual(root["bisection_width_s"], 5e-7)

    def test_exact_grid_roots_are_deduplicated(self):
        model = LaplaceSOI(1.0, 1e5, semimajor_axis_m=np.sqrt(2) / 0.01)
        planet = lambda t: np.array([200.0, 0, 0, 0, 0, 0])
        spacecraft = lambda t: np.array([t - 5, 1.0, 0, 1.0, 0, 0])
        roots = find_soi_crossings(spacecraft, planet, model, 0.0, 10.0, step_s=1)
        np.testing.assert_allclose([r["et_tdb_s"] for r in roots], [4.0, 6.0], atol=1e-8)
        for left, right in ((4.0, 5.0), (3.0, 4.0)):
            result = refine_bracket(lambda t: t - 4, left, right)
            self.assertEqual(result["et_tdb_s"], 4.0)
            self.assertEqual(result["brent_iterations"], 0)

    def test_moving_radius_rate_and_invalid_bracket(self):
        model = LaplaceSOI(1.0, 1e5)
        radius, rate = model.radius_rate([200.0, 0, 0, 3.0, 4.0, 0])
        self.assertAlmostEqual(radius, 2.0)
        self.assertAlmostEqual(rate, 0.03)
        with self.assertRaises(ValueError):
            refine_bracket(lambda t: t * t + 1, 0.0, 1.0)
        with self.assertRaises(ValueError):
            LaplaceSOI(-1.0, 1e5)


if __name__ == "__main__":
    unittest.main()
