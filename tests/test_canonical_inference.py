"""
Automated Test Suite for Canonical Inference Service.
Verifies all Section S requirements:
1. Model loading & CUDA/CPU devices
2. Mask-safe normalization contract
3. Input validation failure modes
4. Tiled inference on synthetic patch
5. Real CDSE inference end-to-end
6. GeoTIFF spatial metadata preservation
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
import torch

from preprocessing import normalize_s1_safe, normalize_s2_safe
from backend.services.processing import (
    InferenceService,
    InferenceConfig,
    InferenceResult,
    get_device,
    load_ecrformer_model,
)


def test_01_model_loading_and_device():
    print("[TEST 1] Model Loading & Device Detection...")
    ckpt = ROOT / "model.ckpt"
    assert ckpt.exists(), f"Checkpoint must exist at {ckpt}"

    # Test CUDA
    device_gpu = get_device()
    assert "cuda" in str(device_gpu), f"Expected CUDA device, got {device_gpu}"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  - CUDA Device: {device_gpu} ({gpu_name})")

    model_gpu = load_ecrformer_model(ckpt, device_gpu)
    assert next(model_gpu.parameters()).is_cuda, "Model parameters must be on CUDA"
    print("  - GPU load verified")

    # Test CPU fallback
    device_cpu = torch.device("cpu")
    model_cpu = load_ecrformer_model(ckpt, device_cpu)
    assert not next(model_cpu.parameters()).is_cuda, "Model parameters must be on CPU"
    print("  - CPU fallback load verified")
    print("PASSED: test_01_model_loading_and_device\n")


def test_02_mask_safe_normalization_contracts():
    print("[TEST 2] Mask-Safe Normalization Contracts...")
    # Create a synthetic S1 array with -9999 NoData and valid dB values
    s1 = np.full((2, 10, 10), -9999.0, dtype=np.float32)
    # Set valid region (center 6x6) to -15.0 dB
    s1[:, 2:8, 2:8] = -15.0

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1

    s1_norm = normalize_s1_safe(s1, valid_mask=mask)

    # Valid pixels (-15 dB) must be (-15 + 25) / 25 = 0.4
    assert np.allclose(s1_norm[:, 2:8, 2:8], 0.4), "Valid S1 values must be correctly normalized"

    # Invalid pixels (-9999) must be EXPLICITLY 0.0 and NEVER clipped
    invalid_pixels = s1_norm[:, mask == 0]
    assert np.all(invalid_pixels == 0.0), "Invalid S1 pixels must be strictly 0.0"
    print("  - S1 NoData (-9999) protection verified")

    # Test S2 normalization
    s2 = np.zeros((13, 10, 10), dtype=np.float32)
    s2[:, 2:8, 2:8] = 2500.0
    s2_norm = normalize_s2_safe(s2, valid_mask=mask)
    assert np.allclose(s2_norm[:, 2:8, 2:8], 0.25), "Valid S2 must be 0.25"
    assert np.all(s2_norm[:, mask == 0] == 0.0), "Invalid S2 pixels must be strictly 0.0"
    print("  - S2 NoData protection verified")
    print("PASSED: test_02_mask_safe_normalization_contracts\n")


def test_03_input_validation_failure_modes():
    print("[TEST 3] Input Validation Failure Modes...")
    service = InferenceService()

    transform = from_origin(288470.0, 4644380.0, 10.0, 10.0)
    crs_ok = rasterio.crs.CRS.from_epsg(32633)
    crs_bad = rasterio.crs.CRS.from_epsg(4326)

    # 1. Band count mismatch (S1 has 3 bands)
    with MemoryFile() as mf_s1, MemoryFile() as mf_s2:
        with mf_s1.open(driver="GTiff", width=20, height=20, count=3, dtype="float32", crs=crs_ok, transform=transform) as s1_bad, \
            mf_s2.open(driver="GTiff", width=20, height=20, count=13, dtype="uint16", crs=crs_ok, transform=transform) as s2_ok:
            try:
                service.validate_inputs(s1_bad, s2_ok)
                assert False, "Should have raised ValueError for S1 band count"
            except ValueError as e:
                print("  - Caught expected band count error:", str(e))

    # 2. CRS mismatch
    with MemoryFile() as mf_s1, MemoryFile() as mf_s2:
        with mf_s1.open(driver="GTiff", width=20, height=20, count=2, dtype="float32", crs=crs_bad, transform=transform) as s1_bad, \
            mf_s2.open(driver="GTiff", width=20, height=20, count=13, dtype="uint16", crs=crs_ok, transform=transform) as s2_ok:
            try:
                service.validate_inputs(s1_bad, s2_ok)
                assert False, "Should have raised ValueError for CRS mismatch"
            except ValueError as e:
                print("  - Caught expected CRS error:", str(e))

    # 3. Dimension mismatch
    with MemoryFile() as mf_s1, MemoryFile() as mf_s2:
        with mf_s1.open(driver="GTiff", width=25, height=20, count=2, dtype="float32", crs=crs_ok, transform=transform) as s1_bad, \
            mf_s2.open(driver="GTiff", width=20, height=20, count=13, dtype="uint16", crs=crs_ok, transform=transform) as s2_ok:
            try:
                service.validate_inputs(s1_bad, s2_ok)
                assert False, "Should have raised ValueError for dimension mismatch"
            except ValueError as e:
                print("  - Caught expected dimension exception:", str(e))

    print("PASSED: test_03_input_validation_failure_modes\n")

def test_04_tiled_inference_synthetic_patch():
    print("[TEST 4] Tiled Inference on Synthetic Patch (140x140)...")
    service = InferenceService()

    width, height = 140, 140
    transform = from_origin(288470.0, 4644380.0, 10.0, 10.0)
    crs = rasterio.crs.CRS.from_epsg(32633)

    s1_data = np.full((2, height, width), -12.0, dtype=np.float32)
    s2_data = np.full((13, height, width), 2000.0, dtype=np.float32)
    mask_data = np.ones((1, height, width), dtype=np.uint8)

    # Make border invalid
    s1_data[:, :10, :] = -9999.0
    s2_data[:, :10, :] = 0.0
    mask_data[0, :10, :] = 0

    with MemoryFile() as mf_s1, MemoryFile() as mf_s2, MemoryFile() as mf_mask:
        with mf_s1.open(driver="GTiff", width=width, height=height, count=2, dtype="float32", crs=crs, transform=transform) as dst:
            dst.write(s1_data)
        with mf_s2.open(driver="GTiff", width=width, height=height, count=13, dtype="float32", crs=crs, transform=transform) as dst:
            dst.write(s2_data)
        with mf_mask.open(driver="GTiff", width=width, height=height, count=1, dtype="uint8", crs=crs, transform=transform) as dst:
            dst.write(mask_data)

        config = InferenceConfig(tile_size=128, overlap=32)
        result = service.run(
            s1_input=mf_s1,
            s2_cloudy_input=mf_s2,
            validity_mask_input=mf_mask,
            config=config,
        )

        assert result.prediction.shape == (13, height, width), f"Expected shape (13, 140, 140), got {result.prediction.shape}"
        assert result.diagnostics.tile_count == 4, f"Expected 4 tiles (2x2) for 140x140 with tile 128/32, got {result.diagnostics.tile_count}"
        assert np.all(result.prediction[:, :10, :] == 0.0), "Invalid border pixels must be strictly 0.0"
        assert np.any(result.prediction[:, 10:, :] > 0.0), "Valid region must produce valid predictions"
        print("  - Output shape and tile count verified")
        print("  - Invalid pixel zero-masking verified")
        print("PASSED: test_04_tiled_inference_synthetic_patch\n")


def test_05_real_cdse_inference_end_to_end():
    print("[TEST 5] Real CDSE Inference End-to-End...")
    s1_path = ROOT / "harmonization" / "output" / "s1_harmonized_10m.tif"
    s2_path = ROOT / "harmonization" / "output" / "s2_harmonized_10m.tif"
    mask_path = ROOT / "harmonization" / "output" / "validity_mask_10m.tif"

    assert s1_path.exists(), f"Missing {s1_path}"
    assert s2_path.exists(), f"Missing {s2_path}"
    assert mask_path.exists(), f"Missing {mask_path}"

    service = InferenceService()
    config = InferenceConfig(tile_size=128, overlap=32)

    tiles_tracked = []
    def progress(done, total):
        tiles_tracked.append((done, total))

    result = service.run(
        s1_input=s1_path,
        s2_cloudy_input=s2_path,
        validity_mask_input=mask_path,
        config=config,
        progress_callback=progress,
    )

    diag = result.diagnostics
    print(f"  - Device: {diag.device}")
    print(f"  - Execution time: {diag.execution_time_seconds:.3f} s")
    print(f"  - Tiles processed: {diag.tile_count}")
    print(f"  - Total pixels: {diag.total_pixels} (Valid: {diag.valid_pixels}, Invalid: {diag.invalid_pixels})")
    print(f"  - Valid range: min={diag.min_val:.6f}, max={diag.max_val:.6f}, mean={diag.mean_val:.6f}")

    assert result.prediction.shape == (13, 262, 200), f"Expected (13, 262, 200), got {result.prediction.shape}"
    assert diag.total_pixels == 52400
    assert diag.valid_pixels == 48769
    assert diag.invalid_pixels == 3631
    assert diag.tile_count == 6  # 2x3 grid
    assert len(tiles_tracked) == 6

    # Verify all invalid pixels in output are strictly 0.0
    invalid_mask = ~result.valid_mask
    assert np.all(result.prediction[:, invalid_mask] == 0.0), "All 3,631 invalid pixels must be strictly 0.0 in output"
    print("  - All 3,631 invalid pixels strictly confirmed as NoData=0.0")

    # Verify valid pixels have physical range
    assert diag.min_val >= 0.0
    assert diag.max_val <= 1.0
    assert diag.mean_val > 0.0

    print("PASSED: test_05_real_cdse_inference_end_to_end\n")
    return result


def test_06_geotiff_spatial_metadata_preservation(real_result: InferenceResult):
    print("[TEST 6] GeoTIFF Spatial Metadata Preservation...")
    out_tif = ROOT / "harmonization" / "output" / "s2_ecrformer_canonical_10m.tif"
    out_png = ROOT / "harmonization" / "output" / "s2_ecrformer_canonical_preview.png"

    real_result.save_geotiff(out_tif)
    real_result.save_preview_png(out_png)

    assert out_tif.exists(), f"Output TIFF not written to {out_tif}"
    assert out_png.exists(), f"Output PNG not written to {out_png}"

    s2_orig = ROOT / "harmonization" / "output" / "s2_harmonized_10m.tif"
    with rasterio.open(s2_orig) as src_orig, rasterio.open(out_tif) as src_pred:
        assert src_pred.count == 13, f"Expected 13 bands, got {src_pred.count}"
        assert src_pred.dtypes[0] == "float32", f"Expected float32, got {src_pred.dtypes[0]}"
        assert src_pred.nodata == 0.0, f"Expected nodata=0.0, got {src_pred.nodata}"
        assert src_pred.crs == src_orig.crs, f"CRS mismatch: {src_pred.crs} vs {src_orig.crs}"
        assert src_pred.transform == src_orig.transform, f"Transform mismatch"
        assert (src_pred.width, src_pred.height) == (src_orig.width, src_orig.height)

        descriptions = [src_pred.descriptions[i] for i in range(13)]
        expected_bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
        assert descriptions == expected_bands, f"Band descriptions mismatch: {descriptions}"

    tif_bytes = real_result.to_geotiff_bytes()
    png_bytes = real_result.to_png_bytes()
    assert len(tif_bytes) > 100000, f"GeoTIFF bytes size unexpectedly small: {len(tif_bytes)}"
    assert len(png_bytes) > 10000, f"PNG bytes size unexpectedly small: {len(png_bytes)}"

    print("  - CRS EPSG:32633 preserved exactly")
    print("  - Affine transform preserved exactly")
    print("  - Band descriptions B01-B12 verified")
    print("  - NoData = 0.0 verified")
    print("  - In-memory bytes export verified")
    print("PASSED: test_06_geotiff_spatial_metadata_preservation\n")


if __name__ == "__main__":
    print("=" * 78)
    print("RUNNING CANONICAL INFERENCE TEST SUITE")
    print("=" * 78 + "\n")

    test_01_model_loading_and_device()
    test_02_mask_safe_normalization_contracts()
    test_03_input_validation_failure_modes()
    test_04_tiled_inference_synthetic_patch()
    res = test_05_real_cdse_inference_end_to_end()
    test_06_geotiff_spatial_metadata_preservation(res)

    print("=" * 78)
    print("ALL 6 TESTS PASSED SUCCESSFULLY!")
    print("=" * 78)
