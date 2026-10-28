"""
Unit tests for CloudVisionAI UI theme system, landing page contracts, and SQLite persistence.
"""
import os
import tempfile
import unittest
from pathlib import Path

from ui.theme import DARK, LIGHT, get_tokens
from backend.services.processing.persistence import (
    init_db,
    record_run,
    get_recent_runs,
)


class TestUIAndPersistence(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test_history.db"

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_theme_tokens_structure(self):
        """Ensure dark and light tokens have all required keys."""
        required_keys = [
            "bg_page",
            "bg_surface",
            "bg_card",
            "border",
            "border_accent",
            "text_primary",
            "text_secondary",
            "accent_blue",
            "accent_teal",
        ]
        for k in required_keys:
            self.assertIn(k, DARK)
            self.assertIn(k, LIGHT)

    def test_persistence_lifecycle(self):
        """Ensure SQLite database initializes, records runs, and retrieves them correctly."""
        init_db(self.db_path)
        self.assertTrue(self.db_path.exists())

        # Initially empty
        runs = get_recent_runs(limit=10, db_path=self.db_path)
        self.assertEqual(len(runs), 0)

        # Record a run
        record_run(
            run_id="run_test_001",
            mode="live_acquisition",
            aoi_label="Rome Test AOI",
            s2_date="2024-09-20",
            s1_date="2024-09-20",
            crs="EPSG:32633",
            dimensions="262x200",
            execution_time_s=1.45,
            device="CUDA (RTX 3050)",
            ecrformer_output_path="/path/to/s2_output.tif",
            metrics_summary={"mae": 0.041, "rmse": 0.058, "psnr": 24.68},
            db_path=self.db_path,
        )

        # Retrieve
        runs = get_recent_runs(limit=10, db_path=self.db_path)
        self.assertEqual(len(runs), 1)
        r = runs[0]
        self.assertEqual(r["run_id"], "run_test_001")
        self.assertEqual(r["mode"], "live_acquisition")
        self.assertEqual(r["aoi_label"], "Rome Test AOI")
        self.assertEqual(r["execution_time_s"], 1.45)
        self.assertIsNotNone(r.get("metrics_summary"))
        self.assertEqual(r["metrics_summary"]["psnr"], 24.68)

    def test_cdse_auth_validation(self):
        """Test CDSE credential validation function without live network."""
        from backend.auth.cdse_auth import test_credentials, CDSEAuth, CDSEAuthError

        # Empty credentials check
        ok, msg = test_credentials("", "")
        self.assertFalse(ok)
        self.assertIn("required", msg.lower())

        ok, msg = test_credentials("only_id", "")
        self.assertFalse(ok)
        self.assertIn("required", msg.lower())

        # When environment variables are absent, CDSEAuth should raise CDSEAuthError
        orig_id = os.environ.pop("CDSE_CLIENT_ID", None)
        orig_sec = os.environ.pop("CDSE_CLIENT_SECRET", None)
        try:
            with self.assertRaises(CDSEAuthError):
                CDSEAuth()
        finally:
            if orig_id:
                os.environ["CDSE_CLIENT_ID"] = orig_id
            if orig_sec:
                os.environ["CDSE_CLIENT_SECRET"] = orig_sec


if __name__ == "__main__":
    unittest.main()
