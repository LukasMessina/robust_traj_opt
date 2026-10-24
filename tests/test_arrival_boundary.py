"""Protect the revised asteroid/TDB target from stale spacecraft/UTC inputs."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from psyche_mission_events_reconstruction.arrival_boundary import load_arrival_boundary


class ArrivalBoundaryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boundary = load_arrival_boundary()

    def check_rejected(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "boundary.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_arrival_boundary(path)

    def test_wrong_target_frame_or_origin_is_rejected(self):
        for key, wrong in (
            ("target_body_id", -255),
            ("origin_id", 2000016),
            ("frame", "ECLIPJ2000"),
            ("units", "km, km/s"),
        ):
            with self.subTest(key=key):
                data = copy.deepcopy(self.boundary)
                data[key] = wrong
                self.check_rejected(data)

    def test_changed_tdb_or_uncovered_epoch_is_rejected(self):
        data = copy.deepcopy(self.boundary)
        data["epoch"]["tdb_calendar"] = "2029-08-01 12:01:09.183260"
        self.check_rejected(data)
        data = copy.deepcopy(self.boundary)
        data["epoch"]["et_tdb_s"] += 1.0
        self.check_rejected(data)
        data = copy.deepcopy(self.boundary)
        data["target_binary_coverage"][0]["end"]["et_tdb_s"] = data["epoch"]["et_tdb_s"] - 1.0
        self.check_rejected(data)

    def test_archived_spacecraft_boundary_is_rejected(self):
        data = copy.deepcopy(self.boundary)
        data.update(target_body_id=-255, fixed_epoch_scale="UTC")
        self.check_rejected(data)

    def test_incomplete_or_unvalidated_state_is_rejected(self):
        data = copy.deepcopy(self.boundary)
        data["sun_relative_state_si"] = data["sun_relative_state_si"][:3]
        self.check_rejected(data)
        data = copy.deepcopy(self.boundary)
        data["validation"]["passed"] = False
        self.check_rejected(data)


if __name__ == "__main__":
    unittest.main()
