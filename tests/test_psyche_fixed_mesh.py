"""Acceptance and defaults for the consolidated workflow."""

import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from deterministic_optimization.psyche import Settings, Transcription
from deterministic_optimization.diagnostics import mesh_accepted
from deterministic_optimization.diagnostics import reference_sensitivity_diagnostic
from deterministic_optimization.settings import ipopt_options
from deterministic_optimization.cli import CASES
from deterministic_optimization.single_phase import (
    default_defect_tolerance,
    default_shooting_substeps,
)
from deterministic_optimization.models import CASE_REGISTRY, EarthMars


class AcceptanceChecks(unittest.TestCase):
    def test_failed_reference_propagation_is_report_only(self):
        accepted = mesh_accepted(Settings(), {"solver_success": True}, [{"max_scaled": 0.5}])
        with tempfile.TemporaryDirectory() as folder:
            with patch("deterministic_optimization.global_reference.main", side_effect=RuntimeError("IVP failed")):
                reference_sensitivity_diagnostic(Path(folder))
            report = json.loads((Path(folder) / "global_reference_error.json").read_text())
        self.assertTrue(accepted)
        self.assertFalse(report["used_for_acceptance"])
        self.assertFalse(report["used_for_refinement"])
        self.assertEqual(report["error"], "IVP failed")

    def test_local_target_required_even_when_refinement_budget_is_zero(self):
        for passes in (0, 8):
            settings = Settings(max_refinements=passes)
            for defects in (None, [{"max_scaled": 2.0}], [{"max_scaled": float("nan")}]):
                self.assertFalse(mesh_accepted(settings, {"solver_success": True}, defects))
            self.assertTrue(
                mesh_accepted(settings, {"solver_success": True}, [{"max_scaled": 0.5}])
            )
            self.assertFalse(
                mesh_accepted(settings, {"solver_success": False}, [{"max_scaled": 0.5}])
            )

    def test_defaults_are_independent_of_saved_experiments(self):
        self.assertEqual(len(CASES), 5)
        s = Settings()
        self.assertEqual(s.initial_intervals_per_phase, (60, 50, 90, 50))
        self.assertEqual(s.mass_flow_model, "throttle_direction")
        self.assertEqual(s.defect_substeps_per_interval, 64)
        self.assertEqual(s.defect_tolerance, 5e-14)
        self.assertEqual((s.max_refinements, s.max_intervals), (8, 1800))
        for factory in CASE_REGISTRY.values():
            self.assertEqual(default_defect_tolerance(factory()), 5e-13)
            self.assertEqual(default_shooting_substeps(factory()), 2)
        self.assertEqual(default_defect_tolerance(EarthMars()), 1e-13)
        self.assertEqual(default_shooting_substeps(EarthMars()), 20)
        options = ipopt_options()
        self.assertEqual(options["mu_strategy"], "adaptive")
        self.assertEqual(options["tol"], 1e-9)
        self.assertEqual(options["constr_viol_tol"], 1e-10)
        self.assertNotIn("max_wall_time", options)
        self.assertNotIn("max_cpu_time", options)

    def test_abandoned_formulations_are_rejected(self):
        for model in ("epigraph", "smooth_norm"):
            with self.assertRaises(ValueError):
                Settings(mass_flow_model=model)
        with self.assertRaises(ValueError):
            Transcription(None, [], method="radau")

    def test_invalid_refinement_counts_rejected(self):
        for count in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                Settings(max_refinements=count)
