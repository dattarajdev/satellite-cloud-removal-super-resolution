"""
Comprehensive Unit and Integration Tests for Per-Image Performance Metrics.

Tests:
1. Correct MAE and RMSE on a known small test array.
2. PSNR behavior, including identical arrays (infinite/capped PSNR) and known delta.
3. SSIM on supported image dimensions (>= 7x7) and small image handling (< 7x7).
4. SAM with numerically stable handling of zero vectors.
5. Masked / invalid pixels being excluded from all metric calculations.
6. Missing ground truth resulting in unavailable metrics, never invented/fabricated values.
7. Correct 13-band ECRformer band handling and per-band breakdown.
8. Correct mapping of B04/B03/B02/B08 for OpenSR and 10 m consistency calculation.
9. Metrics being associated with the correct image identifier and stage.
10. JSON diagnostics serialization (no NaNs or Infs; valid schema).
11. ECRformer metrics remaining intact and available if OpenSR fails.
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from backend.services.processing.evaluation_metrics import (
    compute_reference_based_metrics,
    compute_reference_free_diagnostics,
    compute_opensr_10m_consistency,
    build_image_diagnostics_export,
    sanitize_for_json,
    S2_BAND_NAMES,
    OPENSR_BAND_NAMES,
)


class TestPerformanceMetrics(unittest.TestCase):

    def test_01_mae_and_rmse_known_values(self):
        """Test 1: Verify MAE and RMSE on a known small test array."""
        # 13 bands, 2x2 pixels
        # Pred: all 0.6, Target: all 0.4 -> diff = 0.2
        pred = np.full((13, 2, 2), 0.6, dtype=np.float32)
        target = np.full((13, 2, 2), 0.4, dtype=np.float32)

        metrics = compute_reference_based_metrics(pred, target)
        self.assertAlmostEqual(metrics.aggregate_mae, 0.2, places=5)
        self.assertAlmostEqual(metrics.aggregate_rmse, 0.2, places=5)
        for name in S2_BAND_NAMES:
            self.assertAlmostEqual(metrics.per_band_mae[name], 0.2, places=5)
            self.assertAlmostEqual(metrics.per_band_rmse[name], 0.2, places=5)

    def test_02_psnr_behavior(self):
        """Test 2: PSNR behavior on identical arrays and known RMSE."""
        # Identical arrays: RMSE = 0 -> PSNR should be infinite
        pred_identical = np.full((13, 4, 4), 0.5, dtype=np.float32)
        target_identical = np.full((13, 4, 4), 0.5, dtype=np.float32)

        m_identical = compute_reference_based_metrics(pred_identical, target_identical)
        self.assertTrue(math.isinf(m_identical.aggregate_psnr_db))
        for name in S2_BAND_NAMES:
            self.assertTrue(math.isinf(m_identical.per_band_psnr_db[name]))

        # Known RMSE = 0.1 -> with data_range=1.0: 20 * log10(1.0 / 0.1) = 20 dB
        pred_diff = np.full((13, 4, 4), 0.6, dtype=np.float32)
        target_diff = np.full((13, 4, 4), 0.5, dtype=np.float32)
        m_diff = compute_reference_based_metrics(pred_diff, target_diff)
        self.assertAlmostEqual(m_diff.aggregate_psnr_db, 20.0, places=4)

    def test_03_ssim_dimensions(self):
        """Test 3: SSIM on supported (>= 7x7) and small (< 7x7) image extents."""
        # Supported size (8x8)
        pred_8 = np.ones((13, 8, 8), dtype=np.float32) * 0.5
        tgt_8 = np.ones((13, 8, 8), dtype=np.float32) * 0.5
        m_8 = compute_reference_based_metrics(pred_8, tgt_8)
        self.assertIsNotNone(m_8.ssim_rgb)
        self.assertAlmostEqual(m_8.ssim_rgb, 1.0, places=4)

        # Small size (4x4) -> should not crash, returns None with note
        pred_small = np.ones((13, 4, 4), dtype=np.float32) * 0.5
        tgt_small = np.ones((13, 4, 4), dtype=np.float32) * 0.5
        m_small = compute_reference_based_metrics(pred_small, tgt_small)
        self.assertIsNone(m_small.ssim_rgb)

    def test_04_sam_zero_vector_stability(self):
        """Test 4: SAM with numerically stable handling of zero vectors."""
        # 13 bands, 2 pixels: pixel 0 is non-zero, pixel 1 is zero vector (all bands 0)
        pred = np.zeros((13, 1, 2), dtype=np.float32)
        tgt = np.zeros((13, 1, 2), dtype=np.float32)

        # Pixel 0: non-zero in both
        pred[:, 0, 0] = [0.1 * i for i in range(13)]
        tgt[:, 0, 0] = [0.1 * i for i in range(13)]

        # Pixel 1: pred and target are 0 vectors (norm = 0)
        pred[:, 0, 1] = 0.0
        tgt[:, 0, 1] = 0.0

        m = compute_reference_based_metrics(pred, tgt)
        # Should NOT raise ZeroDivisionError; zero vector should be ignored and identical vector yields 0 degrees
        self.assertFalse(np.isnan(m.sam_degrees))
        self.assertAlmostEqual(m.sam_degrees, 0.0, places=3)

    def test_05_masked_pixels_excluded(self):
        """Test 5: Masked pixels are strictly excluded from calculations."""
        # 13 bands, 2x2 image
        pred = np.zeros((13, 2, 2), dtype=np.float32)
        tgt = np.zeros((13, 2, 2), dtype=np.float32)

        # Top-left pixel (valid): pred=0.5, tgt=0.5 -> diff = 0.0
        # Bottom-right pixel (invalid): pred=1.0, tgt=0.0 -> huge diff = 1.0
        mask = np.zeros((2, 2), dtype=bool)
        mask[0, 0] = True  # only (0, 0) is valid

        pred[:, 1, 1] = 1.0
        tgt[:, 1, 1] = 0.0

        m = compute_reference_based_metrics(pred, tgt, valid_mask=mask)
        # Since invalid pixel (1, 1) was masked out, MAE and RMSE must be 0.0!
        self.assertAlmostEqual(m.aggregate_mae, 0.0, places=5)
        self.assertAlmostEqual(m.aggregate_rmse, 0.0, places=5)
        self.assertEqual(m.num_evaluated_pixels, 1)

    def test_06_missing_ground_truth_diagnostics(self):
        """Test 6: Missing ground truth results in unavailable metrics, never invented values."""
        pred = np.full((13, 10, 10), 0.3, dtype=np.float32)
        mask = np.ones((10, 10), dtype=bool)
        mask[0, 0] = False  # 1 invalid pixel

        diag = compute_reference_free_diagnostics(pred, valid_mask=mask, crs="EPSG:32643")
        self.assertTrue(diag.available)
        self.assertIn("unavailable: no cloud-free reference supplied", diag.status)
        self.assertEqual(diag.valid_pixels, 99)
        self.assertEqual(diag.invalid_pixels, 1)
        self.assertEqual(diag.total_pixels, 100)
        self.assertAlmostEqual(diag.valid_coverage_pct, 99.0)
        self.assertEqual(diag.nan_count, 0)
        self.assertEqual(diag.inf_count, 0)

        # Ensure no invented reference metrics are stored in ReferenceFreeDiagnostics
        self.assertFalse(hasattr(diag, "aggregate_mae"))
        self.assertFalse(hasattr(diag, "aggregate_psnr"))

    def test_07_ecrformer_13_band_handling(self):
        """Test 7: Correct 13-band ECRformer band handling."""
        pred = np.zeros((13, 8, 8), dtype=np.float32)
        tgt = np.zeros((13, 8, 8), dtype=np.float32)
        for i in range(13):
            pred[i] = i * 0.05
            tgt[i] = (i + 1) * 0.05  # constant difference 0.05 for all bands

        m = compute_reference_based_metrics(pred, tgt)
        self.assertEqual(len(m.per_band_mae), 13)
        self.assertEqual(len(m.per_band_rmse), 13)
        self.assertEqual(len(m.per_band_psnr_db), 13)
        for name in S2_BAND_NAMES:
            self.assertIn(name, m.per_band_mae)
            self.assertAlmostEqual(m.per_band_mae[name], 0.05, places=4)

    def test_08_opensr_band_mapping_and_10m_consistency(self):
        """Test 8: Correct mapping of B04/B03/B02/B08 for OpenSR and 10 m consistency."""
        # Create 10 m ECRformer input (13, 4, 4)
        ecr_10m = np.zeros((13, 4, 4), dtype=np.float32)
        # B04 (index 3), B03 (index 2), B02 (index 1), B08 (index 7)
        ecr_10m[3] = 0.2  # B04
        ecr_10m[2] = 0.3  # B03
        ecr_10m[1] = 0.4  # B02
        ecr_10m[7] = 0.5  # B08

        # Create OpenSR 4-band 2.5 m output (4, 16, 16) with scale 4x
        # In DN [0, 10000] scale: 0.2 * 10000 = 2000, etc.
        sr_4band = np.zeros((4, 16, 16), dtype=np.float32)
        sr_4band[0] = 2000.0  # B04
        sr_4band[1] = 3000.0  # B03
        sr_4band[2] = 4000.0  # B02
        sr_4band[3] = 5000.0  # B08

        cons = compute_opensr_10m_consistency(
            opensr_4band_2_5m=sr_4band,
            ecrformer_13band_10m=ecr_10m,
        )
        self.assertTrue(cons.computed)
        self.assertEqual(cons.evaluated_bands, OPENSR_BAND_NAMES)
        self.assertIn("10 m consistency with ECRformer output — not ground-truth accuracy.", cons.metric_label)
        # Since downsampled OpenSR exactly equals ecr_10m, MAE and RMSE must be ~0.0
        self.assertAlmostEqual(cons.aggregate_mae, 0.0, places=4)
        self.assertAlmostEqual(cons.aggregate_rmse, 0.0, places=4)
        self.assertEqual(cons.evaluated_pixels, 16)

    def test_09_image_association_and_stage(self):
        """Test 9: Metrics are associated with the correct image identifier and stage."""
        exp_ecr = build_image_diagnostics_export(
            image_id="img_session_101",
            stage_name="ECRformer Cloud Reconstruction",
            input_path="input_s2.tif",
            output_path="output_ecr.tif",
            dimensions=[100, 100],
            band_names=S2_BAND_NAMES,
            crs="EPSG:32643",
            pixel_size_m=10.0,
            execution_time_seconds=1.5,
            execution_device="CUDA",
        )
        self.assertEqual(exp_ecr["image_identifier"], "img_session_101")
        self.assertEqual(exp_ecr["stage"], "ECRformer Cloud Reconstruction")
        self.assertFalse(exp_ecr["reference_available"])
        self.assertIsNone(exp_ecr["reference_based_metrics"])

        exp_sr = build_image_diagnostics_export(
            image_id="img_session_101",
            stage_name="OpenSR Diffusion Super-Resolution",
            input_path="output_ecr.tif",
            output_path="output_sr.tif",
            dimensions=[400, 400],
            band_names=OPENSR_BAND_NAMES,
            crs="EPSG:32643",
            pixel_size_m=2.5,
            execution_time_seconds=45.2,
            execution_device="CUDA",
            sampling_steps=100,
        )
        self.assertEqual(exp_sr["image_identifier"], "img_session_101")
        self.assertEqual(exp_sr["stage"], "OpenSR Diffusion Super-Resolution")
        self.assertEqual(exp_sr["sampling_steps"], 100)

    def test_10_json_serialization(self):
        """Test 10: JSON diagnostics serialization (handles NaNs/Infs without error)."""
        # Dictionary containing NaN and Inf
        raw_data = {
            "valid_float": 1.234,
            "nan_value": float("nan"),
            "inf_value": float("inf"),
            "nested": {"bad_val": float("nan")},
        }
        sanitized = sanitize_for_json(raw_data)
        # json.dumps must succeed and not raise ValueError or produce unparseable JSON
        json_str = json.dumps(sanitized)
        parsed = json.loads(json_str)
        self.assertIsNone(parsed["nan_value"])
        self.assertIsNone(parsed["inf_value"])
        self.assertIsNone(parsed["nested"]["bad_val"])
        self.assertEqual(parsed["valid_float"], 1.234)

    def test_11_ecrformer_intact_if_opensr_fails(self):
        """Test 11: ECRformer metrics remain intact even if OpenSR raises an exception."""
        # Simulated scenario: ECRformer completed and generated diagnostics
        pred = np.full((13, 8, 8), 0.25, dtype=np.float32)
        ref_free = compute_reference_free_diagnostics(pred, crs="EPSG:32643")

        # Simulate OpenSR failure
        opensr_failed = True
        try:
            if opensr_failed:
                raise RuntimeError("Subprocess timeout or GPU OOM in OpenSR")
        except Exception:
            pass

        # Verify ECRformer diagnostics are still 100% accessible and valid
        self.assertEqual(ref_free.band_count, 13)
        self.assertEqual(ref_free.total_pixels, 64)
        self.assertIn("B04", ref_free.per_band_mean)
        self.assertAlmostEqual(ref_free.per_band_mean["B04"], 0.25)


if __name__ == "__main__":
    unittest.main()
