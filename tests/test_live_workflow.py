"""
End-to-End Verification Test for the Live Acquisition -> Harmonization -> ECRformer Pipeline.
"""
import sys, os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import datetime
from backend.services.satellite.base import BoundingBox, S2Scene, S1Scene
from backend.services.satellite.acquisition_cache import (
    is_s2_cached, is_s1_cached, get_s2_cache_dir, get_s1_cache_dir,
    S2_BAND_NAMES, S1_BAND_NAMES,
)
from backend.services.processing import (
    InferenceService, InferenceConfig, HarmonizationConfig, harmonize_pair
)
from rasterio.enums import Resampling as RioRes

def main():
    print("=" * 70)
    print("TESTING LIVE ACQUISITION WORKFLOW PIPELINE")
    print("=" * 70)

    bbox = BoundingBox(west=12.4500, south=41.9000, east=12.4730, north=41.9230)
    s2_date = "2024-09-21"
    s1_date = "2024-09-20"
    output_size_px = (256, 256)
    cache_root = ROOT / "acquisition" / "data_cache"

    print(f"\n[1] Verifying Acquisition Cache Lookup for AOI {bbox}...")
    s2_hit = is_s2_cached(cache_root, bbox, s2_date, output_size_px)
    s1_hit = is_s1_cached(cache_root, bbox, s1_date, output_size_px)
    print(f"  - S2 ({s2_date}) cached: {s2_hit}")
    print(f"  - S1 ({s1_date}) cached: {s1_hit}")
    assert s2_hit, "S2 should be present in local cache"
    assert s1_hit, "S1 should be present in local cache"

    s2_tif = get_s2_cache_dir(cache_root, bbox, s2_date, output_size_px) / "s2.tif"
    s1_tif = get_s1_cache_dir(cache_root, bbox, s1_date, output_size_px) / "s1.tif"
    print(f"  - S2 TIFF: {s2_tif} ({s2_tif.stat().st_size} bytes)")
    print(f"  - S1 TIFF: {s1_tif} ({s1_tif.stat().st_size} bytes)")

    print(f"\n[2] Testing Harmonization on Acquired Pair...")
    session_dir = ROOT / "harmonization" / "output" / "sessions" / "test_verify"
    session_dir.mkdir(parents=True, exist_ok=True)
    h_config = HarmonizationConfig(
        pixel_size=10.0,
        s2_resampling=RioRes.bilinear,
        s1_resampling=RioRes.bilinear,
        save_previews=True,
        save_validity_mask=True,
    )
    harm_pair = harmonize_pair(
        s2_input_path=s2_tif,
        s1_input_path=s1_tif,
        output_dir=str(session_dir),
        config=h_config,
    )
    print(f"  - Harmonized EPSG: {harm_pair.grid.crs.to_epsg()}")
    print(f"  - S2 Shape: {harm_pair.s2_shape}")
    print(f"  - S1 Shape: {harm_pair.s1_shape}")
    print(f"  - Validation passed: {harm_pair.validation_passed}")
    assert harm_pair.validation_passed, "Harmonization validation must pass"

    print(f"\n[3] Testing Canonical ECRformer Inference on Harmonized Outputs...")
    service = InferenceService(checkpoint_path=ROOT / "model.ckpt")
    infer_cfg = InferenceConfig(tile_size=128, overlap=32, brightness=2.5)
    result = service.run(
        s1_input=harm_pair.s1_path,
        s2_cloudy_input=harm_pair.s2_path,
        validity_mask_input=harm_pair.validity_mask_path,
        config=infer_cfg,
    )
    d = result.diagnostics
    print(f"  - Device: {d.device}")
    print(f"  - Execution time: {d.execution_time_seconds:.3f} s")
    print(f"  - Tiles processed: {d.tile_count}")
    print(f"  - Valid pixels: {d.valid_pixels:,} / {d.total_pixels:,}")
    print(f"  - Output shape: {d.output_shape}")
    print(f"  - Mean reflectance: {d.mean_val:.4f}")

    print(f"\n[4] Testing Byte Exports (GeoTIFF & PNG)...")
    tif_bytes = result.to_geotiff_bytes()
    png_bytes = result.to_png_bytes()
    print(f"  - GeoTIFF bytes: {len(tif_bytes):,} bytes")
    print(f"  - PNG bytes: {len(png_bytes):,} bytes")
    assert len(tif_bytes) > 10000, "GeoTIFF export should produce valid non-empty byte stream"
    assert len(png_bytes) > 1000, "PNG export should produce valid non-empty byte stream"

    print("\n" + "=" * 70)
    print("ALL LIVE WORKFLOW PIPELINE CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()