"""Independent analytic checks of dynamics-only seed propagation and events."""

import unittest

import numpy as np

from psyche_mission_events_reconstruction.rollout import (
    integrate_coast,
    rollout_to_soi_exit,
    soi_radius_and_rate,
)
from psyche_mission_events_reconstruction.soi_events import LaplaceSOI


class ConstantPlanetEphemeris:
    valid_interval = (0.0, 100.0)
    metadata = {"mu_m3_s2": {"399": 1.0, "10": 1e5}}

    def state(self, body, epoch):
        return np.array([200.0, 0, 0, 0, 0, 0])

    def derivative(self, body, epoch):
        return np.zeros(6)


class RolloutChecks(unittest.TestCase):
    def test_fixed_radius_exit_needs_no_ephemeris_evaluation(self):
        class ForbiddenEphemeris(ConstantPlanetEphemeris):
            def state(self, body, epoch):
                raise AssertionError("Fixed SOI should not evaluate ephemerides")

            def derivative(self, body, epoch):
                raise AssertionError("Fixed SOI has zero radius derivative")

        cache = ForbiddenEphemeris()
        model = LaplaceSOI(1.0, 1e5, 200.0)  # R=2
        self.assertAlmostEqual(soi_radius_and_rate(cache, 399, 0.0, model)[0], 2.0)
        self.assertEqual(soi_radius_and_rate(cache, 399, 99.0, model)[1], 0.0)
        state = np.array([1.0, 0, 0, 0, 2.0, 0, 10.0])
        arc = rollout_to_soi_exit(state, 0.0, 5.0, 399, cache, soi_model=model, max_step_s=0.001)
        self.assertAlmostEqual(arc.duration_s, (4 - np.log(3)) / np.sqrt(8), places=10)

    def test_fixed_soi_rejects_inconsistent_gm(self):
        with self.assertRaisesRegex(ValueError, "GMs must match"):
            soi_radius_and_rate(ConstantPlanetEphemeris(), 399, 0.0, LaplaceSOI(2.0, 1e5, 200.0))

    def test_mars_inbound_boundary_does_not_terminate_at_start(self):
        # Same analytic hyperbola as below, starting exactly on R=2 inbound.
        # An event solver must ignore that root and reach the OUTWARD root.
        class MarsEphemeris(ConstantPlanetEphemeris):
            metadata = {"mu_m3_s2": {"499": 1.0, "10": 1e5}}

        state = np.array([2 / 3, -4 * np.sqrt(2) / 3, 0, np.sqrt(2) / 3, 5 / 3, 0, 10.0])
        arc = rollout_to_soi_exit(state, 0.0, 5.0, 499, MarsEphemeris(), max_step_s=0.001)
        expected = 2 * (4 - np.log(3)) / np.sqrt(8)
        self.assertAlmostEqual(arc.duration_s, expected, places=10)
        np.testing.assert_allclose(
            arc.final_state,
            [2 / 3, 4 * np.sqrt(2) / 3, 0, -np.sqrt(2) / 3, 5 / 3, 0, 10.0],
            rtol=0,
            atol=1e-10,
        )

    def test_outward_event_matches_analytic_hyperbola(self):
        # mu=1, rp=1, vp=2 -> |a|=1/2, e=3. R_SOI=2 gives H=log(3),
        # dt=sqrt(|a|^3/mu)*(e*sinh(H)-H)=(4-log(3))/sqrt(8).
        state = np.array([1.0, 0, 0, 0, 2.0, 0, 10.0])
        arc = rollout_to_soi_exit(state, 0.0, 5.0, 399, ConstantPlanetEphemeris(), max_step_s=0.001)
        expected = (4 - np.log(3)) / np.sqrt(8)
        self.assertAlmostEqual(arc.duration_s, expected, places=10)
        self.assertAlmostEqual(np.linalg.norm(arc.final_state[:3]), 2.0, places=11)
        self.assertEqual(arc.final_state[6], 10.0)
        self.assertGreater(np.dot(arc.final_state[:3], arc.final_state[3:6]), 0)

    def test_no_exit_is_reported_and_dense_output_is_bounded(self):
        state = np.array([1.0, 0, 0, 0, 1.0, 0, 10.0])  # circular orbit, inside R=2
        with self.assertRaisesRegex(RuntimeError, "No positive-time"):
            rollout_to_soi_exit(state, 0.0, 1.0, 399, ConstantPlanetEphemeris(), max_step_s=0.01)
        arc = integrate_coast(state, 0.0, 2 * np.pi, 1.0, 399, max_step_s=0.01)
        np.testing.assert_allclose(arc.final_state, state, rtol=0, atol=1e-10)
        with self.assertRaises(ValueError):
            arc.at_elapsed(arc.duration_s + 1)
        with self.assertRaises(ValueError):
            integrate_coast(state, 1.0, 0.0, 1.0, 399)


if __name__ == "__main__":
    unittest.main()
