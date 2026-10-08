"""
Phase 2 Harmonization CLI Runner.

Executes satellite harmonization on real Sentinel-1 and Sentinel-2 acquisitions.
Verifies all 10 harmonization requirements and outputs production-grade GeoTIFFs and diagnostics.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rasterio.crs import CRS
from rasterio.enums import Resampling

from backend.services.processing import (
    HarmonizationConfig,
    harmonize_pair,
    S2_BAND_ORDER,
    S1_BAND_ORDER,
)

_SEP = "=" * 78
_LINE = "-" * 78


def find_cached_inputs(cache_root: Path) -> tuple[Path | None, Path | None]:
    """Auto-detect most recent S2 and S1 GeoTIFFs from acquisition/data_cache."""
    if not cache_root.exists():
        return None, None

    s2_candidates = list(cache_root.glob("**/s2.tif"))
    s1_candidates = list(cache_root.glob("**/s1.tif"))

    s2_path = max(s2_candidates, key=lambda p: p.stat().st_mtime) if s2_candidates else None
    s1_path = max(s1_candidates, key=lambda p: p.stat().st_mtime) if s1_candidates else None
    return s2_path, s1_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2 Satellite Harmonization: Reproject S1 & S2 to 10m UTM common grid."
    )
    parser.add_argument(
        "--s2",
        type=str,
        default=None,
        help="Path to raw 13-band Sentinel-2 L1C GeoTIFF (default: auto-detected from acquisition/data_cache)",
    )
    parser.add_argument(
        "--s1",
        type=str,
        default=None,
        help="Path to raw 2-band Sentinel-1 GRD GeoTIFF (default: auto-detected from acquisition/data_cache)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "harmonization" / "output"),
        help="Directory to write harmonized GeoTIFFs and diagnostics (default: harmonization/output)",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=10.0,
        help="Target grid pixel resolution in meters (default: 10.0)",
    )
    parser.add_argument(
        "--resampling",
        type=str,
        choices=["bilinear", "nearest", "cubic"],
        default="bilinear",
        help="Spatial resampling filter (default: bilinear)",
    )
    parser.add_argument(
        "--target-epsg",
        type=int,
        default=None,
        help="Optional override for target UTM EPSG code (e.g. 32633)",
    )

    args = parser.parse_args()

    print(_SEP)
    print("PHASE 2 - SATELLITE HARMONIZATION PIPELINE")
    print(_SEP)

    # Resolve inputs
    s2_input = Path(args.s2) if args.s2 else None
    s1_input = Path(args.s1) if args.s1 else None

    if s2_input is None or s1_input is None:
        cache_dir = PROJECT_ROOT / "acquisition" / "data_cache"
        detected_s2, detected_s1 = find_cached_inputs(cache_dir)
        if s2_input is None:
            s2_input = detected_s2
        if s1_input is None:
            s1_input = detected_s1

    if s2_input is None or not s2_input.exists():
        print(f"ERROR: Sentinel-2 input file not found or not specified.")
        print("Please provide --s2 /path/to/s2.tif or run Phase 1 acquisition first.")
        return 1

    if s1_input is None or not s1_input.exists():
        print(f"ERROR: Sentinel-1 input file not found or not specified.")
        print("Please provide --s1 /path/to/s1.tif or run Phase 1 acquisition first.")
        return 1

    print(f"Input Sentinel-2: {s2_input}")
    print(f"Input Sentinel-1: {s1_input}")
    print(f"Target Pixel Size: {args.pixel_size} meters")
    print(f"Resampling:        {args.resampling}")
    print(f"Output Directory:  {args.output_dir}")
    print(_LINE)

    # Configure harmonization
    resamp_map = {
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
        "cubic": Resampling.cubic,
    }
    target_crs = CRS.from_epsg(args.target_epsg) if args.target_epsg else None

    config = HarmonizationConfig(
        pixel_size=args.pixel_size,
        s2_resampling=resamp_map[args.resampling],
        s1_resampling=resamp_map[args.resampling],
        target_crs=target_crs,
        save_previews=True,
        save_validity_mask=True,
    )

    # Execute harmonization
    try:
        pair = harmonize_pair(
            s2_input_path=s2_input,
            s1_input_path=s1_input,
            output_dir=args.output_dir,
            config=config,
        )
    except Exception as e:
        print(f"ERROR during harmonization: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print validation table
    diag = pair.diagnostics
    reqs = diag.get("requirements_check", {})
    meta = diag.get("spatial_metadata", {})
    nd = diag.get("nodata_stats", {})

    print("\nHARMONIZATION REQUIREMENTS VALIDATION:")
    print("+-----------------------------------------------------------------+---------+")
    print("| Requirement                                                     | Status  |")
    print("+-----------------------------------------------------------------+---------+")
    titles = [
        ("1_utm_crs_determined", "1. Determine UTM CRS from AOI"),
        ("2_identical_crs", "2. Reproject S1 & S2 to same UTM CRS"),
        ("3_common_output_grid", "3. Common output grid (identical shape & transform)"),
        ("4_explicit_10m_pixel_size", f"4. Explicit {args.pixel_size}m pixel size"),
        ("5_identical_spatial_extent", "5. Identical bounding box extent"),
        ("6_band_orders_preserved", "6. Preserved band orders (S2: 13, S1: 2)"),
        ("7_explicit_resampling", f"7. Explicit spatial resampling ({args.resampling})"),
        ("8_scientific_dtypes_preserved", "8. Scientific dtypes preserved (uint16 / float32)"),
        ("9_nodata_handled_explicitly", "9. Explicit NoData handling"),
        ("10_validation_diagnostics_generated", "10. Full validation diagnostics generated"),
    ]
    for key, title in titles:
        status_str = "PASS" if reqs.get(key, False) else "FAIL"
        print(f"| {title:<63} |  {status_str:<6} |")
    print("+-----------------------------------------------------------------+---------+")

    overall_status = diag.get("status", "FAILED")
    print(f"\nOverall Validation Status: {overall_status}")
    print(_LINE)
    print("SPATIAL & GRID METRICS:")
    print(f"  Target CRS:         {meta.get('crs')} (EPSG:{meta.get('epsg')})")
    print(f"  Output Grid Shape:  {meta.get('width')} x {meta.get('height')} (width x height)")
    print(f"  Pixel Dimensions:   {meta.get('pixel_size_x')}m x {meta.get('pixel_size_y')}m")
    b = meta.get("bounds", {})
    print(f"  UTM Bounding Box:   Left={b.get('left'):.1f}, Bottom={b.get('bottom'):.1f}, Right={b.get('right'):.1f}, Top={b.get('top'):.1f}")
    print(f"  Total Grid Pixels:  {nd.get('total_pixels')}")
    print(f"  Valid S2 Pixels:    {nd.get('s2_valid_pixels')} ({nd.get('s2_valid_percent')}%)")
    print(f"  Valid S1 Pixels:    {nd.get('s1_valid_pixels')} ({nd.get('s1_valid_percent')}%)")
    print(f"  Intersection Valid: {nd.get('intersection_valid_pixels')} ({nd.get('intersection_valid_percent')}%)")
    print(_LINE)
    print("OUTPUT ARTIFACTS GENERATED:")
    print(f"  [1] Harmonized S2 GeoTIFF:  {pair.s2_path}")
    print(f"  [2] Harmonized S1 GeoTIFF:  {pair.s1_path}")
    if pair.validity_mask_path:
        print(f"  [3] Validity Mask GeoTIFF:  {pair.validity_mask_path}")
    if pair.preview_s2_path:
        print(f"  [4] S2 RGB Preview:         {pair.preview_s2_path}")
    if pair.preview_s1_path:
        print(f"  [5] S1 SAR Preview:         {pair.preview_s1_path}")
    if pair.preview_coreg_path:
        print(f"  [6] Co-registration Diff:   {pair.preview_coreg_path}")
    print(f"  [7] JSON Report:            {pair.report_json_path}")
    print(f"  [8] Markdown Report:        {pair.report_json_path.with_suffix('.md')}")
    print(_SEP)

    return 0 if pair.validation_passed else 1


if __name__ == "__main__":
    sys.exit(main())
