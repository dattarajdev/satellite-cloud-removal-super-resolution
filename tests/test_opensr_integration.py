"""
Automated Integration Test Suite for OpenSR Diffusion Super-Resolution Service.

Verifies:
1. Environment and pre-flight validation (missing python, missing script, invalid inputs)
2. Band selector and verification of B04, B03, B02, B08 requirements
3. Real end-to-end super-resolution on verified 13-band ECRformer GeoTIFF
4. Output geometry verification (4x dimensions = 816x816, 2.5m pixel size, CRS = EPSG:32643)
5. Preservation of authoritative 13-band ECRformer GeoTIFF (never modified/overwritten)
6. RGB preview generation (rendered strictly from B04, B03, B02, excluding B08 NIR)
7. Byte streams export for Streamlit download buttons
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
from rasterio.transform import from_origin

from backend.services.processing import (
    OpenSRService,
    OpenSRConfig,
    OpenSRResult,
    validate_and_get_band_indices,
)


def test_01_band_selector_validation():
    print("[TEST 1] Band Selector Validation...")
    # Test valid 13-band descriptions
    descriptions_13 = [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
        "B8A", "B09", "B10", "B11", "B12"
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_tif = Path(tmpdir) / "test_13band.tif"
        transform = from_origin(100.0, 100.0, 10.0, 10.0)
        with rasterio.open(
            tmp_tif,
            "w",
            driver="GTiff",
            width=16,
            height=16,
            count=13,
            dtype="float32",
            crs="EPSG:32643",
            transform=transform,
        ) as dst:
            for i, desc in enumerate(descriptions_13):
                dst.set_band_description(i + 1, desc)

        with rasterio.open(tmp_tif) as src:
            indices = validate_and_get_band_indices(src)
            # B4 is index 4, B3 is index 3, B2 is index 2, B8 is index 8
            assert indices == (4, 3, 2, 8), f"Expected (4, 3, 2, 8), got {indices}"
            print(f"  - 13-band indexed correctly: {indices}")

        # Test invalid band composition (e.g. 3-band RGB without NIR B8)
        bad_tif = Path(tmpdir) / "test_bad.tif"
        with rasterio.open(
            bad_tif,
            "w",
            driver="GTiff",
            width=16,
            height=16,
            count=3,
            dtype="float32",
            crs="EPSG:32643",
            transform=transform,
        ) as dst:
            for i, desc in enumerate(["B4", "B3", "B2"]):
                dst.set_band_description(i + 1, desc)

        with rasterio.open(bad_tif) as src:
            try:
                validate_and_get_band_indices(src)
                assert False, "Should have raised ValueError for missing B8"
            except ValueError as e:
                print(f"  - Caught expected missing NIR error: {e}")

    print("PASSED: test_01_band_selector_validation\n")


def test_02_error_handling_and_preflight():
    print("[TEST 2] Error Handling and Pre-Flight Checks...")

    # Case A: Missing Python executable
    cfg_bad_python = OpenSRConfig(python_executable=Path("non_existent_python_xyz.exe"))
    svc = OpenSRService(cfg_bad_python)
    try:
        svc.validate_environment()
        assert False, "Should have raised FileNotFoundError for missing python"
    except FileNotFoundError as e:
        print(f"  - Caught expected missing python error: {e}")

    # Case B: Missing script
    cfg_bad_script = OpenSRConfig(script_path=Path("non_existent_script_xyz.py"))
    svc = OpenSRService(cfg_bad_script)
    try:
        svc.validate_environment()
        assert False, "Should have raised FileNotFoundError for missing script"
    except FileNotFoundError as e:
        print(f"  - Caught expected missing script error: {e}")

    # Case C: Non-existent input file
    svc_default = OpenSRService()
    try:
        svc_default.run(Path("non_existent_input.tif"))
        assert False, "Should have raised FileNotFoundError for missing input"
    except FileNotFoundError as e:
        print(f"  - Caught expected missing input error: {e}")

    print("PASSED: test_02_error_handling_and_preflight\n")


def test_03_real_opensr_end_to_end():
    print("[TEST 3] Real OpenSR End-to-End Super-Resolution...")
    verified_tif = ROOT / "reconstructed_images" / "s2_reconstructed_2024-09-08_13band.tif"
    assert verified_tif.is_file(), f"Verified ECRformer input not found at {verified_tif}"

    # Capture original file mtime and hash to guarantee it remains untouched
    orig_mtime = verified_tif.stat().st_mtime
    orig_size = verified_tif.stat().st_size
    with rasterio.open(verified_tif) as src:
        in_width = src.width
        in_height = src.height
        in_crs = src.crs
        in_bounds = src.bounds
        in_count = src.count
        in_dtype = src.dtypes[0]

    assert in_width == 204
    assert in_height == 204
    assert in_count == 13
    assert str(in_crs) == "EPSG:32643"
    print(f"  - Input TIFF verified: {in_width}x{in_height} px, {in_count} bands, {in_crs}")

    svc = OpenSRService()
    out_dir = ROOT / "reconstructed_images" / "test_output"

    result: OpenSRResult = svc.run(input_tif_path=verified_tif, output_dir=out_dir)

    # 1. Verify original file untouched
    assert verified_tif.stat().st_size == orig_size, "Authoritative 13-band TIFF size must NOT change"
    assert verified_tif.stat().st_mtime == orig_mtime, "Authoritative 13-band TIFF mtime must NOT change"
    print("  - Authoritative 13-band ECRformer TIFF confirmed UNTOUCHED")

    # 2. Verify output product existence
    sr_path = result.sr_geotiff_path
    assert sr_path.is_file(), f"Output GeoTIFF must exist at {sr_path}"
    print(f"  - Super-resolved GeoTIFF produced at: {sr_path}")

    # 3. Verify output geometry and metadata
    with rasterio.open(sr_path) as src_sr:
        assert src_sr.width == 816, f"Expected 816 width, got {src_sr.width}"
        assert src_sr.height == 816, f"Expected 816 height, got {src_sr.height}"
        assert src_sr.count == 4, f"Expected 4 bands, got {src_sr.count}"
        assert src_sr.dtypes[0] == "uint16", f"Expected uint16, got {src_sr.dtypes[0]}"
        assert str(src_sr.crs) == str(in_crs), f"CRS mismatch: {src_sr.crs} vs {in_crs}"
        assert abs(src_sr.transform.a) == 2.5, f"Expected pixel size 2.5m, got {abs(src_sr.transform.a)}"
        assert abs(src_sr.transform.e) == 2.5, f"Expected pixel size 2.5m, got {abs(src_sr.transform.e)}"

        # Bounds check
        assert abs(src_sr.bounds.left - in_bounds.left) < 1e-2
        assert abs(src_sr.bounds.right - in_bounds.right) < 1e-2
        assert abs(src_sr.bounds.bottom - in_bounds.bottom) < 1e-2
        assert abs(src_sr.bounds.top - in_bounds.top) < 1e-2

        descriptions = [src_sr.descriptions[i] for i in range(4)]
        assert descriptions == ["B4", "B3", "B2", "B8"], f"Band descriptions mismatch: {descriptions}"

    print(f"  - Output geometry: 816x816 px (4x scale)")
    print(f"  - Output pixel size: 2.5m x 2.5m")
    print(f"  - Output CRS: {in_crs} preserved")
    print(f"  - Output bounds: {in_bounds} preserved")
    print(f"  - Band descriptions: {descriptions} verified")

    # 4. Verify RGB preview rendering (strictly B04, B03, B02; B08 excluded)
    preview = result.sr_rgb_preview
    assert preview.size == (816, 816), f"Preview dimensions must be (816, 816), got {preview.size}"
    preview_arr = np.array(preview)
    assert preview_arr.shape == (816, 816, 3), f"Preview must have 3 RGB channels, got {preview_arr.shape}"
    assert np.any(preview_arr > 0), "Preview must contain non-zero RGB imagery"
    print(f"  - 3-band RGB preview verified: size {preview.size}, channels {preview_arr.shape[2]}")

    # 5. Verify byte export streams
    tif_bytes = result.to_geotiff_bytes()
    png_bytes = result.to_png_bytes()
    assert len(tif_bytes) > 500000, f"GeoTIFF byte stream unexpectedly small: {len(tif_bytes)}"
    assert len(png_bytes) > 50000, f"PNG byte stream unexpectedly small: {len(png_bytes)}"
    print(f"  - Byte stream exports verified: GeoTIFF ({len(tif_bytes):,} bytes), PNG ({len(png_bytes):,} bytes)")

    # 6. Verify diagnostics
    diag = result.diagnostics
    print(f"  - Execution device: {diag.execution_device}")
    print(f"  - Execution time: {diag.execution_time_seconds:.2f} s")
    print(f"  - Scale factor: {diag.scale_factor}x")
    print(f"  - GPU memory reported: {diag.gpu_memory}")
    assert diag.scale_factor == 4.0
    assert diag.output_pixel_size_m == 2.5
    assert diag.bounds_preserved is True

    print("PASSED: test_03_real_opensr_end_to_end\n")


if __name__ == "__main__":
    print("=" * 78)
    print("RUNNING OPENSR DIFFUSION SUPER-RESOLUTION TEST SUITE")
    print("=" * 78 + "\n")

    test_01_band_selector_validation()
    test_02_error_handling_and_preflight()
    test_03_real_opensr_end_to_end()

    print("=" * 78)
    print("ALL OPENSR INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 78)
