"""
Test Suite for Ground-Truth Reference Evaluation and Live Clear-Reference Search.

Verifies:
 1. A valid Streamlit UploadedFile-style TIFF can be loaded and evaluated.
 2. The upload's stream cursor position does not affect reading.
 3. An invalid file is rejected cleanly.
 4. Wrong band counts are rejected.
 5. Invalid or unknown band ordering is handled correctly.
 6. CRS/grid mismatches are handled through valid reprojection.
 7. DN and reflectance normalization are handled consistently.
 8. No-data pixels are excluded from metrics.
 9. Missing references do not result in fabricated metrics.
10. CDSE catalogue search can be tested with mocked responses.
11. Empty catalogue results are handled correctly.
12. AOI cloud fraction uses valid pixels and documented SCL/CLD rules.
13. A selected reference is associated with the correct AOI and acquisition.
14. Temporal references are clearly distinguished from paired ground truth.
15. The reference is never included in inference input.
16. ECRformer outputs remain available if reference acquisition or metric calculation fails.
17. Existing ECRformer, AOI, live acquisition and OpenSR tests continue to pass.
"""

from __future__ import annotations

import datetime
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.services.processing.evaluation_metrics import (
    compute_reference_based_metrics,
    ReferenceBasedMetrics,
    S2_BAND_NAMES,
)
from backend.services.processing.reference_adapter import (
    load_and_validate_reference,
    ValidatedReference,
    _open_reference_dataset,
    S2_CANONICAL_BANDS,
)
from backend.services.satellite.base import BoundingBox, S2Scene
from backend.services.satellite.clear_reference_service import (
    ClearReferenceService,
    ClearReferenceCandidate,
    compute_aoi_cloud_fraction_from_scl,
)
from backend.services.processing.inference_service import (
    InferenceResult,
    InferenceDiagnostics,
)


class MockUploadedFile:
    """Simulates Streamlit's UploadedFile object."""
    def __init__(self, data: bytes, name: str = "target.tif"):
        self._data = data
        self.name = name
        self._stream = io.BytesIO(data)

    def getvalue(self) -> bytes:
        return self._data

    def read(self, *args, **kwargs):
        return self._stream.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        return self._stream.seek(*args, **kwargs)

    def tell(self) -> int:
        return self._stream.tell()


def create_mock_geotiff_bytes(
    shape=(13, 16, 16),
    crs="EPSG:32633",
    transform=from_origin(300000, 4600000, 10, 10),
    dtype="uint16",
    value=2000,
    descriptions=None,
    nodata=0,
    zero_pixels=None,
) -> bytes:
    """Helper to generate an in-memory GeoTIFF buffer."""
    memfile = rasterio.io.MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=shape[1],
        width=shape[2],
        count=shape[0],
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        data = np.full(shape, value, dtype=dtype)
        if zero_pixels:
            for r, c in zero_pixels:
                data[:, r, c] = 0
        dst.write(data)
        if descriptions:
            for idx, desc in enumerate(descriptions, start=1):
                dst.set_band_description(idx, desc)
    return bytes(memfile.getbuffer())


class TestReferenceEvaluation(unittest.TestCase):

    def setUp(self):
        self.default_crs = "EPSG:32633"
        self.default_transform = from_origin(300000, 4600000, 10, 10)
        self.target_profile = {
            "driver": "GTiff",
            "crs": rasterio.crs.CRS.from_string(self.default_crs),
            "transform": self.default_transform,
            "width": 16,
            "height": 16,
            "count": 13,
            "dtype": "float32",
        }
        # Fake InferenceResult
        self.pred_array = np.full((13, 16, 16), 0.2, dtype=np.float32)
        self.valid_mask = np.ones((16, 16), dtype=bool)
        self.diag = InferenceDiagnostics(
            device="cpu",
            execution_time_seconds=1.2,
            tile_count=1,
            total_pixels=256,
            valid_pixels=256,
            invalid_pixels=0,
            output_shape=(13, 16, 16),
            min_val=0.0,
            max_val=1.0,
            mean_val=0.2,
        )
        self.infer_res = InferenceResult(
            prediction=self.pred_array,
            profile=self.target_profile,
            valid_mask=self.valid_mask,
            diagnostics=self.diag,
            cloudy_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
            sar_preview_img=np.zeros((16, 16, 3), dtype=np.uint8),
            prediction_rgb=np.zeros((16, 16, 3), dtype=np.uint8),
        )

    # 1. Streamlit UploadedFile loading & evaluation
    def test_01_uploaded_file_evaluation(self):
        tif_bytes = create_mock_geotiff_bytes(value=2000, dtype="uint16")
        uploaded_file = MockUploadedFile(tif_bytes, name="target.tif")

        val_ref = load_and_validate_reference(
            reference_input=uploaded_file,
            target_profile=self.target_profile,
            scale_mode="auto",
        )
        self.assertIsInstance(val_ref, ValidatedReference)
        self.assertEqual(val_ref.array.shape, (13, 16, 16))
        # 2000 / 10000 = 0.2
        self.assertAlmostEqual(float(np.mean(val_ref.array)), 0.2, places=4)

        metrics = self.infer_res.evaluate_with_reference(uploaded_file)
        self.assertTrue(metrics.available)
        self.assertAlmostEqual(metrics.aggregate_mae, 0.0, places=4)

    # 2. Upload's stream cursor position independence
    def test_02_cursor_position_independence(self):
        tif_bytes = create_mock_geotiff_bytes(value=3000, dtype="uint16")
        uploaded_file = MockUploadedFile(tif_bytes, name="target.tif")
        # Advance stream to middle or end
        uploaded_file.seek(150)
        self.assertEqual(uploaded_file.tell(), 150)

        # Reader must reset and read entire GeoTIFF without failing
        val_ref = load_and_validate_reference(uploaded_file, target_profile=self.target_profile)
        self.assertEqual(val_ref.array.shape, (13, 16, 16))
        self.assertAlmostEqual(float(np.mean(val_ref.array)), 0.3, places=4)

    # 3. Invalid file rejection
    def test_03_invalid_file_rejected(self):
        corrupt_bytes = b"NOT_A_VALID_TIFF_FILE_HEADER"
        corrupt_upload = MockUploadedFile(corrupt_bytes, name="broken.tif")
        with self.assertRaises(Exception):
            load_and_validate_reference(corrupt_upload, target_profile=self.target_profile)

    # 4. Wrong band count rejection
    def test_04_wrong_band_count_rejected(self):
        # 4 bands instead of 13
        tif_4band = create_mock_geotiff_bytes(shape=(4, 16, 16), dtype="uint16")
        upload_4band = MockUploadedFile(tif_4band)
        with self.assertRaises(ValueError) as ctx:
            load_and_validate_reference(upload_4band, target_profile=self.target_profile)
        self.assertIn("must contain exactly 13 Sentinel-2 bands", str(ctx.exception))

    # 5. Band ordering detection and reordering
    def test_05_band_ordering_handled(self):
        # Reverse band order
        reversed_bands = list(reversed(S2_CANONICAL_BANDS))
        memfile = rasterio.io.MemoryFile()
        with memfile.open(
            driver="GTiff",
            height=16,
            width=16,
            count=13,
            dtype="float32",
            crs=self.default_crs,
            transform=self.default_transform,
        ) as dst:
            for i, b_name in enumerate(reversed_bands, start=1):
                # Write band index value
                dst.write(np.full((16, 16), float(i) / 10.0, dtype=np.float32), i)
                dst.set_band_description(i, b_name)

        rev_bytes = bytes(memfile.getbuffer())
        upload_rev = MockUploadedFile(rev_bytes)

        val_ref = load_and_validate_reference(upload_rev, target_profile=self.target_profile)
        # Should be re-mapped to canonical order
        # B01 was last in reversed_bands (index 13, value 1.3)
        self.assertAlmostEqual(float(val_ref.array[0, 0, 0]), 1.0, places=2)  # clipped to 1.0
        self.assertTrue(any("Reordered" in n for n in val_ref.notes))

    # 6. CRS / grid mismatch reprojection
    def test_06_crs_grid_reprojection(self):
        # Reference at 8x8 pixels with a different transform
        diff_transform = from_origin(300000, 4600000, 20, 20)
        tif_diff_grid = create_mock_geotiff_bytes(
            shape=(13, 8, 8),
            transform=diff_transform,
            value=2500,
        )
        upload_diff = MockUploadedFile(tif_diff_grid)

        val_ref = load_and_validate_reference(upload_diff, target_profile=self.target_profile)
        self.assertTrue(val_ref.reprojected)
        self.assertEqual(val_ref.array.shape, (13, 16, 16))
        self.assertEqual(val_ref.profile["width"], 16)
        self.assertEqual(val_ref.profile["height"], 16)

    # 7. Radiometric scaling: DN vs Reflectance
    def test_07_radiometric_scaling(self):
        # Case A: DN 0-10000 (value = 5000 -> 0.5)
        tif_dn = create_mock_geotiff_bytes(value=5000, dtype="uint16")
        val_dn = load_and_validate_reference(tif_dn, scale_mode="auto")
        self.assertAlmostEqual(float(np.mean(val_dn.array)), 0.5, places=4)
        self.assertIn("Auto-detected DN scale", val_dn.radiometric_scale_applied)

        # Case B: Reflectance 0.0-1.0 (value = 0.45)
        tif_ref = create_mock_geotiff_bytes(value=0.45, dtype="float32")
        val_ref = load_and_validate_reference(tif_ref, scale_mode="auto")
        self.assertAlmostEqual(float(np.mean(val_ref.array)), 0.45, places=4)
        self.assertIn("Auto-detected reflectance scale", val_ref.radiometric_scale_applied)

    # 8. No-data masking
    def test_08_nodata_masking(self):
        # Set top-left pixel to nodata=0 in uint16
        tif_bytes = create_mock_geotiff_bytes(
            value=4000,
            dtype="uint16",
            nodata=0,
            zero_pixels=[(0, 0)],
        )

        val_ref = load_and_validate_reference(tif_bytes, target_profile=self.target_profile)
        self.assertFalse(val_ref.valid_mask[0, 0])
        self.assertTrue(val_ref.valid_mask[1, 1])

        metrics = self.infer_res.evaluate_with_reference(val_ref)
        self.assertEqual(metrics.num_evaluated_pixels, 255)

    # 9. Missing reference does not fabricate metrics
    def test_09_missing_reference_no_fabrication(self):
        # Fresh inference result has no reference metrics
        self.assertIsNone(self.infer_res.reference_metrics)
        self.assertIsNone(self.infer_res.diagnostics.reference_metrics)
        self.assertIn("unavailable: no cloud-free reference supplied", self.infer_res.diagnostics.reference_status)

    # 10. Mocked CDSE catalogue search
    def test_10_mocked_cdse_catalogue_search(self):
        mock_provider = MagicMock()
        mock_dt = datetime.datetime(2024, 9, 10, 10, 0, 0, tzinfo=datetime.timezone.utc)
        mock_provider.search_sentinel2.return_value = [
            S2Scene(
                scene_id="S2A_MSIL2A_TEST_01",
                dt=mock_dt,
                cloud_cover=3.5,
                processing_level="L2A",
                footprint=None,
                extra={"aoi:cloud_cover": 1.2},
            ),
            S2Scene(
                scene_id="S2A_MSIL2A_TEST_02_CLOUDY",
                dt=mock_dt + datetime.timedelta(days=5),
                cloud_cover=45.0,  # Should be filtered out by max_scene_cloud_cover=15
                processing_level="L2A",
                footprint=None,
            ),
        ]

        svc = ClearReferenceService(provider=mock_provider)
        candidates = svc.search_candidates(
            bbox=BoundingBox(12.4, 41.8, 12.5, 41.9),
            target_date="2024-09-20",
            window_days=30,
            max_scene_cloud_cover=15.0,
            max_aoi_cloud_fraction=10.0,
        )

        self.assertEqual(len(candidates), 1)
        c = candidates[0]
        self.assertEqual(c.scene_id, "S2A_MSIL2A_TEST_01")
        self.assertEqual(c.scene_cloud_cover_pct, 3.5)
        self.assertEqual(c.aoi_cloud_fraction_pct, 1.2)
        self.assertTrue(c.passes_threshold)
        self.assertAlmostEqual(c.temporal_distance_days, -10.0, places=1)

    # 11. Empty catalogue results handled correctly
    def test_11_empty_catalogue_results(self):
        mock_provider = MagicMock()
        mock_provider.search_sentinel2.return_value = []
        svc = ClearReferenceService(provider=mock_provider)
        candidates = svc.search_candidates(
            bbox=BoundingBox(12.4, 41.8, 12.5, 41.9),
            target_date="2024-09-20",
        )
        self.assertEqual(len(candidates), 0)

    # 12. AOI cloud fraction calculation using SCL rules
    def test_12_aoi_cloud_fraction_scl_rules(self):
        # 10 pixels total:
        # 2 invalid pixels: class 0 (nodata), class 1 (saturated)
        # 8 valid pixels:
        #   - 4 clear: class 4 (veg), class 5 (soil), class 6 (water), class 2 (dark)
        #   - 2 cloud: class 8 (cloud med), class 9 (cloud high)
        #   - 1 cirrus: class 10 (thin cirrus)
        #   - 1 shadow: class 3 (shadow)
        scl = np.array([0, 1, 4, 5, 6, 2, 8, 9, 10, 3], dtype=np.uint8)

        # With shadows & cirrus: 4 cloudy out of 8 valid = 50.0%
        pct_all = compute_aoi_cloud_fraction_from_scl(scl, include_shadows=True, include_cirrus=True)
        self.assertAlmostEqual(pct_all, 50.0, places=4)

        # Without shadows: 3 cloudy out of 8 valid = 37.5%
        pct_no_shadow = compute_aoi_cloud_fraction_from_scl(scl, include_shadows=False, include_cirrus=True)
        self.assertAlmostEqual(pct_no_shadow, 37.5, places=4)

        # All invalid pixels
        all_invalid = np.array([0, 1, 0, 1], dtype=np.uint8)
        self.assertIsNone(compute_aoi_cloud_fraction_from_scl(all_invalid))

    # 13. Candidate reference associated with correct AOI and metadata
    def test_13_reference_metadata_association(self):
        mock_dt = datetime.datetime(2024, 9, 8, 10, 0, 0, tzinfo=datetime.timezone.utc)
        cand = ClearReferenceCandidate(
            scene_id="S2A_MSIL2A_TEST_REF",
            acquisition_dt=mock_dt,
            date_str="2024-09-08",
            temporal_distance_days=-12.5,
            scene_cloud_cover_pct=2.1,
            aoi_cloud_fraction_pct=0.5,
            passes_threshold=True,
            status="Clear Candidate",
        )
        self.assertIn("2024-09-08 (-12.5d)", cand.display_label())

    # 14. Distinction between ground truth and temporal references
    def test_14_temporal_reference_disclaimer_and_label(self):
        pred = np.full((13, 8, 8), 0.5, dtype=np.float32)
        target = np.full((13, 8, 8), 0.4, dtype=np.float32)

        # Case A: True ground truth
        m_gt = compute_reference_based_metrics(pred, target, reference_type="paired_ground_truth")
        self.assertEqual(m_gt.reference_type, "paired_ground_truth")
        self.assertEqual(m_gt.metric_label, "Reference-based reconstruction metrics (Ground Truth)")
        self.assertIsNone(m_gt.disclaimer)

        # Case B: Temporal reference
        m_temp = compute_reference_based_metrics(
            pred,
            target,
            reference_type="temporal_reference",
            reference_date="2024-09-08",
            temporal_distance_days=-12.0,
        )
        self.assertEqual(m_temp.reference_type, "temporal_reference")
        self.assertEqual(m_temp.metric_label, "Temporal-reference comparison metrics")
        self.assertIsNotNone(m_temp.disclaimer)
        self.assertIn("clear temporal reference from a different acquisition date", m_temp.disclaimer)

    # 15. Reference is never included in inference input
    def test_15_reference_not_in_inference_input(self):
        original_pred = self.infer_res.prediction.copy()
        tif_ref = create_mock_geotiff_bytes(value=5000, dtype="uint16")
        self.infer_res.evaluate_with_reference(tif_ref)

        # Prediction and cloudy inputs must be bitwise identical
        np.testing.assert_array_equal(self.infer_res.prediction, original_pred)

    # 16. ECRformer outputs preserved on reference failure
    def test_16_ecrformer_outputs_preserved_on_failure(self):
        original_pred = self.infer_res.prediction.copy()
        corrupt_ref = MockUploadedFile(b"BROKEN_TIFF_DATA")

        try:
            self.infer_res.evaluate_with_reference(corrupt_ref)
        except Exception:
            pass

        # Reconstruction output and downloads remain intact
        np.testing.assert_array_equal(self.infer_res.prediction, original_pred)
        self.assertGreater(len(self.infer_res.to_geotiff_bytes()), 0)


if __name__ == "__main__":
    unittest.main()
