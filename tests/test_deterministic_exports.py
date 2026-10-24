"""Public-output routing and cartographic transformations."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from deterministic_optimization.exports import output_prefix, export_case, history_compatible
from deterministic_optimization.psyche_figures import occulted, body_orientation, surface_geometry
from deterministic_optimization.plot_assets import ASSETS


class ExportChecks(unittest.TestCase):
    def test_case_family_and_objective_routing(self):
        root = Path("test-output")
        for case, family in (
            ("earth_mars", "earth_mars_deterministic_traj_opt"),
            ("psyche", "psyche_mission_deterministic_traj_opt"),
            ("nrho_l2_to_dro", "cr3bp_deterministic_traj_opt"),
            ("halo_l2_to_halo_l1", "cr3bp_deterministic_traj_opt"),
            ("lyapunov_l1_to_l2", "cr3bp_deterministic_traj_opt"),
        ):
            for objective in ("fuel", "energy"):
                self.assertEqual(
                    output_prefix(case, objective, root),
                    root / family / (objective + "_optimal") / case,
                )
        with self.assertRaises(ValueError):
            output_prefix("../outside", "fuel", root)

    def test_failed_solve_cannot_overwrite_accepted_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "configuration.json").write_text("{}")
            (directory / "result.json").write_text(json.dumps({"converged": False}))
            with self.assertRaisesRegex(ValueError, "unconverged"):
                export_case(directory, "psyche", "fuel", directory / "public")
            self.assertFalse((directory / "public").exists())

    def test_stochastic_consumer_fuel_history_field(self):
        self.assertEqual(history_compatible([{"fuel_kg": 17.0}])[-1]["fuel_consumed_kg"], 17.0)

    def test_plot_failure_does_not_relabel_a_converged_solve(self):
        from deterministic_optimization.cli import run_one

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def completed_run(directory, **kwargs):
                directory.mkdir()
                result = {"converged": True}
                (directory / "result.json").write_text(json.dumps(result))
                return result

            with patch(
                "deterministic_optimization.psyche_driver.run", side_effect=completed_run
            ), patch(
                "deterministic_optimization.exports.export_case",
                side_effect=RuntimeError("plot failure"),
            ):
                result = run_one("psyche", "fuel", root, 0, root / "public")
            self.assertTrue(result["converged"])
            self.assertFalse(result["published"])
            self.assertIn("plot failure", result["error"])

    def test_front_back_occultation(self):
        # View along +x: front trajectory and off-limb trajectory remain visible.
        p = np.array([[2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [-2.0, 2.0, 0.0]])
        np.testing.assert_array_equal(
            occulted(p, np.array([1.0, 0.0, 0.0]), np.eye(3), (1.0, 1.0, 1.0)), [False, True, False]
        )

    @unittest.skipUnless(
        (ASSETS / "manifest.json").exists(),
        "Download plotting assets to exercise SPICE cartography",
    )
    def test_cartographic_seam_poles_and_rotation(self):
        for body in ("Earth", "Mars"):
            geometry, colors, axes = surface_geometry(body, np.eye(3), resolution=8)
            np.testing.assert_allclose(geometry[:, :, 0], geometry[:, :, -1], atol=2e-12)
            np.testing.assert_allclose(geometry[:, 0, 0], [0, 0, axes[2]], atol=2e-12)
            np.testing.assert_allclose(
                geometry[:, 4, 8], [axes[0], 0, 0], atol=2e-12
            )  # Greenwich at map centre.
            rotation, _, record = body_orientation(body, 750000000.0)
            np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(rotation), 1.0)


if __name__ == "__main__":
    unittest.main()
