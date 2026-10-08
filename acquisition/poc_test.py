"""
Phase 1 Proof-of-Concept: Sentinel-1 + Sentinel-2 Acquisition via CDSE.

PURPOSE
-------
Prove that real Sentinel-1 and Sentinel-2 imagery can be acquired for a
small AOI using the Copernicus Data Space Ecosystem (CDSE) Sentinel Hub APIs,
and that the downloaded data can be validated for ECRformer compatibility.

THIS SCRIPT DOES NOT:
  - Run ECRformer inference
  - Automatically select scenes
  - Apply S1/S2 spatial alignment
  - Build any UI

MODES
-----
Search mode (default — no download flags):
    python acquisition/poc_test.py

Search + download mode:
    python acquisition/poc_test.py --s2-date 2024-09-11 --s1-date 2024-09-07

Custom AOI:
    python acquisition/poc_test.py --west 2.30 --south 48.80 --east 2.52 --north 49.02

Custom date range:
    python acquisition/poc_test.py --start 2024-06-01 --end 2024-06-30

PREREQUISITES
-------------
See acquisition/README.md.

ENVIRONMENT VARIABLES REQUIRED
-------------------------------
    CDSE_CLIENT_ID       (from your CDSE dashboard OAuth client)
    CDSE_CLIENT_SECRET   (from your CDSE dashboard OAuth client)
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
import warnings

import numpy as np

# ---- Locate project root so we can import from the project root ----
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Existing project modules (do NOT modify these)
from preprocessing import normalize_s1, normalize_s2
from visualization  import to_rgb, sar_preview

# New acquisition modules
from backend.auth.cdse_auth import CDSEAuthError
from backend.services.satellite.base import BoundingBox
from backend.services.satellite.cdse_sentinelhub import CDSESentinelHubProvider

# PIL for preview images
from PIL import Image


# -----------------------------------------------------------------------
# Default configuration
#
# Small test AOI near Rome, Italy.
# ~2.3 km EW × 2.6 km NS at 256×256 pixels ≈ 9–10 m/pixel in WGS84.
# Sentinel-1 and Sentinel-2 archive coverage over this area is reliable
# and has been stable since ~2016.
#
# Change these defaults, or override them with CLI arguments.
# -----------------------------------------------------------------------

DEFAULT_AOI = BoundingBox(
    west  = 12.450,
    south = 41.900,
    east  = 12.473,
    north = 41.923,
)

DEFAULT_START_DATE = "2024-09-01"
DEFAULT_END_DATE   = "2024-09-30"

# Output pixel grid for downloaded TIFFs.
# At the default Rome AOI, 256×256 gives ~9 m/pixel EW and ~10 m/pixel NS.
# Increase to 512 for a higher-resolution diagnostic, but download takes longer.
DEFAULT_OUTPUT_SIZE = (256, 256)

# Maximum catalogue results per sensor
CATALOGUE_MAX_RESULTS = 25

# Cache root
CACHE_DIR = os.path.join(_SCRIPT_DIR, "data_cache")


# -----------------------------------------------------------------------
# Printing helpers
# -----------------------------------------------------------------------

_LINE = "-" * 80

def section(title: str) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}")


def subsection(title: str) -> None:
    print(f"\n{_LINE}")
    print(f"  {title}")
    print(_LINE)


def banner() -> None:
    print()
    print("+" + "=" * 62 + "+")
    print("|   Cloud Removal -- Phase 1 Proof-of-Concept                  |")
    print("|   Sentinel-1 + Sentinel-2 Acquisition via CDSE               |")
    print("+" + "-" * 62 + "+")
    print("|  This script DOES NOT run ECRformer inference.               |")
    print("|  It DOES collect evidence about data compatibility.          |")
    print("+" + "=" * 62 + "+")
    print()


# -----------------------------------------------------------------------
# Cache key
# -----------------------------------------------------------------------

def compute_cache_key(
    provider: str,
    collection: str,
    bbox: BoundingBox,
    target_date: str,
    bands: list[str],
    output_size_px: tuple[int, int],
) -> str:
    """
    Compute a short, deterministic hash key for a download request.

    The key encodes all parameters that affect the downloaded data.
    Re-running the same request will hit the same cache directory.
    """
    params = {
        "provider":    provider,
        "collection":  collection,
        "bbox":        bbox.to_list(),
        "target_date": target_date,
        "bands":       sorted(bands),
        "output_size": list(output_size_px),
    }
    raw = json.dumps(params, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# -----------------------------------------------------------------------
# Scene display helpers
# -----------------------------------------------------------------------

def print_s2_scenes(scenes) -> None:
    if not scenes:
        print("  No Sentinel-2 L1C scenes found for this AOI and date range.")
        return

    hdr = (
        f"  {'#':>3}  {'Date':^10}  {'Time UTC':^10}  "
        f"{'Cloud':^8}  {'Level':^5}  Scene ID"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for idx, sc in enumerate(scenes, 1):
        cloud_str = sc.cloud_str()
        print(
            f"  {idx:>3}  {sc.utc_date()}  {sc.utc_time():^10}  "
            f"{cloud_str:^8}  {sc.processing_level:^5}  {sc.scene_id}"
        )
    print(f"\n  Total: {len(scenes)} scene(s)")


def print_s1_scenes(scenes) -> None:
    if not scenes:
        print("  No Sentinel-1 GRD scenes found for this AOI and date range.")
        return

    hdr = (
        f"  {'#':>3}  {'Date':^10}  {'Time UTC':^10}  "
        f"{'Polarization':^13}  {'Orbit':^11}  {'Mode':^4}  Notes"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for idx, sc in enumerate(scenes, 1):
        orbit = sc.orbit_direction or "?"
        mode  = sc.instrument_mode or "?"
        pol   = sc.pol_str()
        note  = "" if sc.has_vv_vh() else "⚠ Not VV+VH"
        print(
            f"  {idx:>3}  {sc.utc_date()}  {sc.utc_time():^10}  "
            f"{pol:^13}  {orbit:^11}  {mode:^4}  {note}"
        )
    print(f"\n  Total: {len(scenes)} scene(s)")


# -----------------------------------------------------------------------
# Rasterio validation
# -----------------------------------------------------------------------

def validate_tiff(path: str, label: str) -> dict:
    """
    Open a GeoTIFF with rasterio and print / return key metadata.

    This is the Phase 1 acceptance validation.
    """
    import rasterio

    subsection(f"{label} GeoTIFF Validation")

    if not os.path.isfile(path):
        print(f"  [ERROR] File not found: {path}")
        return {}

    with rasterio.open(path) as src:
        meta = {
            "path":        path,
            "band_count":  src.count,
            "width":       src.width,
            "height":      src.height,
            "dtype":       str(src.dtypes[0]),
            "crs":         str(src.crs),
            "transform":   list(src.transform)[:6],
            "nodata":      src.nodata,
        }

        # Resolution
        res_x = abs(src.transform.a)
        res_y = abs(src.transform.e)
        meta["res_x_deg"] = res_x
        meta["res_y_deg"] = res_y

        # Approximate GSD at the bbox mid-latitude
        lat_mid = (src.bounds.bottom + src.bounds.top) / 2.0
        gsd_ew = res_x * 111_320.0 * math.cos(math.radians(lat_mid))
        gsd_ns = res_y * 110_540.0
        meta["gsd_ew_m"] = round(gsd_ew, 2)
        meta["gsd_ns_m"] = round(gsd_ns, 2)

        data = src.read()
        meta["global_min"] = float(np.nanmin(data))
        meta["global_max"] = float(np.nanmax(data))
        meta["global_mean"] = float(np.nanmean(data))
        meta["has_nan"] = bool(np.any(np.isnan(data.astype(np.float32))))

        # Per-band stats
        per_band = []
        for b in range(src.count):
            band_data = data[b].astype(np.float64)
            finite = band_data[np.isfinite(band_data)]
            per_band.append({
                "min": float(np.min(finite)) if len(finite) else float("nan"),
                "max": float(np.max(finite)) if len(finite) else float("nan"),
                "mean": float(np.mean(finite)) if len(finite) else float("nan"),
            })
        meta["per_band"] = per_band

    # ---- Print ----
    print(f"  File:        {path}")
    print(f"  Bands:       {meta['band_count']}")
    print(f"  Width:       {meta['width']}  Height: {meta['height']}")
    print(f"  Dtype:       {meta['dtype']}")
    print(f"  CRS:         {meta['crs']}")
    print(f"  Transform:   {meta['transform']}")
    print(f"  Resolution:  {meta['res_x_deg']:.7f}° × {meta['res_y_deg']:.7f}°  "
          f"≈ {meta['gsd_ew_m']:.1f} m EW × {meta['gsd_ns_m']:.1f} m NS")
    print(f"  Nodata:      {meta['nodata']}")
    print(f"  NaN pixels:  {meta['has_nan']}")
    print(f"  Global:      min={meta['global_min']:.4g}  "
          f"max={meta['global_max']:.4g}  "
          f"mean={meta['global_mean']:.4g}")


    print(f"\n  Per-band statistics:")
    for idx, bstat in enumerate(meta["per_band"]):
        print(f"    Band {idx+1:>2}: "
              f"min={bstat['min']:>10.4g}  "
              f"max={bstat['max']:>10.4g}  "
              f"mean={bstat['mean']:>10.4g}")

    return meta


# -----------------------------------------------------------------------
# Normalization diagnostic
# -----------------------------------------------------------------------

def diagnose_s2_normalization(path: str) -> None:
    """
    Apply preprocessing.py::normalize_s2() and report whether the output
    matches ECRformer/SEN12MS-CR expectations.

    Expected:
        Input:  uint16, [0, 10000] DN
        Output: float32, [0, 1]

    This does NOT run the model. It only checks value ranges.
    """
    import rasterio

    subsection("S2 Normalization Diagnostic (preprocessing.py::normalize_s2)")
    print("  This checks whether the downloaded S2 data is compatible")
    print("  with the current ECRformer normalization.")
    print()

    with rasterio.open(path) as src:
        raw = src.read().astype(np.float32)

    print(f"  Raw dtype:           {raw.dtype}")
    print(f"  Raw global min:      {raw.min():.1f}")
    print(f"  Raw global max:      {raw.max():.1f}")
    print(f"  Expected range:      [0, 10000] DN (SEN12MS-CR convention)")

    # Apply existing normalization
    normed = normalize_s2(raw)

    print(f"\n  After normalize_s2():")
    print(f"    min:  {normed.min():.6f}")
    print(f"    max:  {normed.max():.6f}")
    print(f"    mean: {normed.mean():.6f}")
    print(f"    Expected post-norm range: [0.0, 1.0]")

    print("\n  Compatibility assessment:")
    issues = []

    if raw.max() > 10000:
        issues.append(
            f"    [RISK] raw max {raw.max():.0f} > 10000 -- data may NOT be DN units.\n"
            "      The Processing API may have returned reflectance (0-1 float) or\n"
            "      a different radiometric scale."
        )
    elif raw.max() < 50:
        issues.append(
            f"    [RISK] raw max {raw.max():.4f} < 50 -- values look like float reflectance\n"
            "      not DN. Expected range [0, 10000]. Normalization will produce near-zero\n"
            "      values."
        )

    if normed.max() > 1.01:
        issues.append(
            f"    [RISK] normalized max {normed.max():.4f} > 1.0 -- clipping in normalize_s2\n"
            "      may not be sufficient, or input units are wrong."
        )

    if normed.max() < 0.01:
        issues.append(
            "    [RISK] all normalized values near 0 -- input is likely in the wrong\n"
            "      radiometric scale."
        )

    if not issues:
        pct = normed.max() * 100
        print(f"    [OK] Values appear compatible (max normalized = {normed.max():.4f}).")
        if pct < 50:
            print(f"    [INFO] Max reflectance = {pct:.1f}% of scale (normal for non-snow surfaces).")
    else:
        for issue in issues:
            print(issue)

    # Per-band normalization
    print("\n  Per-band normalized ranges:")
    band_names = [
        "B01", "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B8A", "B09", "B10", "B11", "B12"
    ]
    for i, bname in enumerate(band_names[:normed.shape[0]]):
        b = normed[i]
        print(f"    [{i:>2}] {bname}: "
              f"min={b.min():.4f}  max={b.max():.4f}  mean={b.mean():.4f}")


def diagnose_s1_normalization(path: str) -> None:
    """
    Apply preprocessing.py::normalize_s1() and report compatibility.

    Expected:
        Input:  float32, dB scale (typically [-30, 5] dB for land)
        clip:   [-25, 0] dB → normalize to [0, 1]

    Key diagnostic: actual VV and VH min values vs the -25 dB clip.
    If VH extends below -32 dB (common over smooth surfaces), the current
    clip will saturate those pixels — this must be experimentally assessed.
    """
    import rasterio

    subsection("S1 Normalization Diagnostic (preprocessing.py::normalize_s1)")
    print("  This checks whether the downloaded S1 data is compatible")
    print("  with the current ECRformer normalization.")
    print()

    with rasterio.open(path) as src:
        raw = src.read().astype(np.float32)

    print(f"  Raw dtype:           {raw.dtype}")

    band_names = ["VV", "VH"]
    for i, bname in enumerate(band_names[:raw.shape[0]]):
        b = raw[i]
        finite = b[np.isfinite(b)]
        bmin = float(np.min(finite)) if len(finite) else float("nan")
        bmax = float(np.max(finite)) if len(finite) else float("nan")
        print(f"  {bname} raw range:   min={bmin:8.2f} dB  max={bmax:8.2f} dB")

    print(f"\n  Current clip range (normalize_s1): [-25, 0] dB")
    print(f"  NOTE: Some SEN12MS-CR implementations clip VH to [-32.5, 0] dB.")
    print(f"  This diagnostic measures actual ranges to inform that decision.")

    # Apply normalization
    normed = normalize_s1(raw)

    print(f"\n  After normalize_s1():")
    for i, bname in enumerate(band_names[:normed.shape[0]]):
        b = normed[i]
        print(f"    {bname}: min={b.min():.6f}  max={b.max():.6f}  mean={b.mean():.6f}")

    print("\n  Compatibility assessment:")
    issues = []

    vv_raw = raw[0][np.isfinite(raw[0])]
    vv_min = float(np.min(vv_raw)) if len(vv_raw) else float("nan")
    if vv_min < -40:
        issues.append(
            f"    [RISK] VV minimum {vv_min:.1f} dB is well below -40 dB.\n"
            "      This may indicate the data is in linear (not dB) scale."
        )
    elif -25 <= vv_min <= 0:
        print(f"    [OK] VV min {vv_min:.1f} dB is within [-25, 0] clip range.")
    else:
        print(f"    [INFO] VV min {vv_min:.1f} dB is below the -25 dB clip -- will be set to 0.")

    if raw.shape[0] >= 2:
        vh_raw = raw[1][np.isfinite(raw[1])]
        vh_min = float(np.min(vh_raw)) if len(vh_raw) else float("nan")
        if vh_min < -40:
            issues.append(
                f"    [RISK] VH minimum {vh_min:.1f} dB is well below -40 dB.\n"
                "      This may indicate the data is in linear (not dB) scale."
            )
        elif vh_min < -25:
            pct_clipped = float(np.mean(vh_raw < -25)) * 100
            print(
                f"    [INFO] VH min {vh_min:.1f} dB < -25 dB clip -- "
                f"{pct_clipped:.1f}% of VH pixels will be clipped to 0.\n"
                "      This is a known risk. If clipping degrades model output,\n"
                "      consider using -32.5 dB as the VH lower bound."
            )
        else:
            print(f"    [OK] VH min {vh_min:.1f} dB is within [-25, 0] clip range.")

    if not issues:
        if vv_min >= -40:
            print("    [OK] Data appears to be in dB scale.")
    else:
        for issue in issues:
            print(issue)


# -----------------------------------------------------------------------
# Preview generation
# -----------------------------------------------------------------------

def save_s2_preview(tiff_path: str, preview_path: str) -> None:
    """Save an S2 RGB preview PNG (bands B04/B03/B02, indices 3/2/1)."""
    import rasterio

    with rasterio.open(tiff_path) as src:
        raw = src.read().astype(np.float32)

    normed = normalize_s2(raw)

    # to_rgb from visualization.py: bands=(3, 2, 1) = B04, B03, B02 = R, G, B
    rgb = to_rgb(normed, bands=(3, 2, 1), brightness=3.0)

    Image.fromarray(rgb).save(preview_path)
    print(f"  [PREVIEW] S2 RGB -> {preview_path}")


def save_s1_preview(tiff_path: str, preview_path: str) -> None:
    """Save an S1 grayscale preview PNG (mean of VV+VH after normalization)."""
    import rasterio

    with rasterio.open(tiff_path) as src:
        raw = src.read().astype(np.float32)

    normed = normalize_s1(raw)

    # sar_preview from visualization.py: mean of all S1 bands → grayscale
    img = sar_preview(normed)
    img.save(preview_path)
    print(f"  [PREVIEW] S1 grayscale -> {preview_path}")


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1 PoC: Sentinel-1/2 acquisition via CDSE.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search only -- list all available scenes:
  python acquisition/poc_test.py

  # Search + download specific scenes:
  python acquisition/poc_test.py --s2-date 2024-09-11 --s1-date 2024-09-07

  # Custom AOI (Paris):
  python acquisition/poc_test.py --west 2.30 --south 48.80 --east 2.52 --north 49.02

  # Custom date range:
  python acquisition/poc_test.py --start 2024-06-01 --end 2024-06-30 --s2-date 2024-06-15

IMPORTANT: Set environment variables before running:
  Windows CMD:   set CDSE_CLIENT_ID=your_id
  Windows PS:    $env:CDSE_CLIENT_ID = "your_id"
  See acquisition/README.md for full setup instructions.
        """
    )

    # AOI
    parser.add_argument("--west",  type=float, default=DEFAULT_AOI.west,
                        help=f"AOI west longitude (default: {DEFAULT_AOI.west})")
    parser.add_argument("--south", type=float, default=DEFAULT_AOI.south,
                        help=f"AOI south latitude (default: {DEFAULT_AOI.south})")
    parser.add_argument("--east",  type=float, default=DEFAULT_AOI.east,
                        help=f"AOI east longitude (default: {DEFAULT_AOI.east})")
    parser.add_argument("--north", type=float, default=DEFAULT_AOI.north,
                        help=f"AOI north latitude (default: {DEFAULT_AOI.north})")

    # Date range for catalogue search
    parser.add_argument("--start", default=DEFAULT_START_DATE,
                        metavar="YYYY-MM-DD",
                        help=f"Search start date (default: {DEFAULT_START_DATE})")
    parser.add_argument("--end",   default=DEFAULT_END_DATE,
                        metavar="YYYY-MM-DD",
                        help=f"Search end date (default: {DEFAULT_END_DATE})")

    # Download target dates (triggers download mode)
    parser.add_argument("--s2-date", default=None, metavar="YYYY-MM-DD",
                        help="Sentinel-2 acquisition date to download. "
                             "Must be one of the dates shown in search output.")
    parser.add_argument("--s1-date", default=None, metavar="YYYY-MM-DD",
                        help="Sentinel-1 acquisition date to download. "
                             "Must be one of the dates shown in search output.")

    # Output size
    parser.add_argument("--size", type=int, default=256, metavar="N",
                        help="Output pixel size (NxN, default: 256)")

    return parser.parse_args()


def main() -> None:
    banner()
    args = parse_args()

    bbox = BoundingBox(
        west  = args.west,
        south = args.south,
        east  = args.east,
        north = args.north,
    )
    output_size = (args.size, args.size)

    download_mode = args.s2_date is not None or args.s1_date is not None

    # ----------------------------------------------------------------
    # Configuration summary
    # ----------------------------------------------------------------
    section("Configuration")
    print(f"  AOI:         {bbox}")
    print(f"  AOI width:   ~{bbox.approx_width_m() / 1000:.2f} km EW")
    print(f"  AOI height:  ~{bbox.approx_height_m() / 1000:.2f} km NS")
    gsd_ew, gsd_ns = bbox.approx_gsd_m(*output_size)
    print(f"  Search:      {args.start}  →  {args.end}")
    print(f"  Mode:        {'SEARCH + DOWNLOAD' if download_mode else 'SEARCH ONLY'}")
    if download_mode:
        print(f"  S2 date:     {args.s2_date or '(none — search only)'}")
        print(f"  S1 date:     {args.s1_date or '(none — search only)'}")
        print(f"  Output size: {output_size[0]}×{output_size[1]} px")
        print(f"  GSD approx:  {gsd_ew:.1f} m/px EW  ×  {gsd_ns:.1f} m/px NS  (WGS84)")
        print(f"  Cache dir:   {CACHE_DIR}")
    print()

    # ----------------------------------------------------------------
    # Authentication
    # ----------------------------------------------------------------
    section("CDSE Authentication")
    try:
        provider = CDSESentinelHubProvider()
        print("  ✓ Credentials loaded from environment variables.")
    except CDSEAuthError as exc:
        print(f"\n  ✗ Authentication setup failed:\n\n{exc}")
        sys.exit(1)

    # Verify credentials by actually fetching a token
    try:
        provider._auth.get_token()
        print("  ✓ OAuth2 token acquired successfully.")
    except CDSEAuthError as exc:
        print(f"\n  ✗ Could not acquire OAuth2 token:\n\n{exc}")
        sys.exit(1)

    # ----------------------------------------------------------------
    # Sentinel-2 catalogue search
    # ----------------------------------------------------------------
    section("Sentinel-2 L1C Catalogue Search")
    print(f"  Searching {args.start} → {args.end} ...")
    try:
        s2_scenes = provider.search_sentinel2(
            bbox=bbox,
            start_date=args.start,
            end_date=args.end,
            max_results=CATALOGUE_MAX_RESULTS,
        )
    except RuntimeError as exc:
        print(f"  ✗ Sentinel-2 search failed:\n\n{exc}")
        sys.exit(1)

    print()
    print_s2_scenes(s2_scenes)

    if s2_scenes:
        print()
        print("  [INFO] To download a scene, rerun with:")
        print(f"     --s2-date YYYY-MM-DD")
        print("  where YYYY-MM-DD is one of the dates shown above.")

    # ----------------------------------------------------------------
    # Sentinel-1 catalogue search
    # ----------------------------------------------------------------
    section("Sentinel-1 GRD Catalogue Search")
    print(f"  Searching {args.start} → {args.end} ...")
    try:
        s1_scenes = provider.search_sentinel1(
            bbox=bbox,
            start_date=args.start,
            end_date=args.end,
            max_results=CATALOGUE_MAX_RESULTS,
        )
    except RuntimeError as exc:
        print(f"  ✗ Sentinel-1 search failed:\n\n{exc}")
        sys.exit(1)

    print()
    print_s1_scenes(s1_scenes)

    if s1_scenes:
        print()
        print("  [INFO] To download a scene, rerun with:")
        print(f"     --s1-date YYYY-MM-DD")
        print("  where YYYY-MM-DD is one of the dates shown above.")
        vv_vh_count = sum(1 for sc in s1_scenes if sc.has_vv_vh())
        if vv_vh_count < len(s1_scenes):
            print(f"\n  [WARNING] {len(s1_scenes) - vv_vh_count} scene(s) do NOT have VV+VH polarization.")
            print("    ECRformer requires VV+VH. Only IW-GRD scenes with 'VV + VH' are usable.")

    if not download_mode:
        section("Next Step")
        print("  This was a search-only run.")
        print("  Review the scene lists above and choose:")
        print("    - A Sentinel-2 scene with LOW cloud cover (< 30% preferred)")
        print("    - A Sentinel-1 scene with VV+VH polarization and IW mode")
        print("    - Both scenes should cover the same area (the AOI is small,")
        print("      so all scenes will cover it)")
        print()
        print("  Then rerun with explicit dates, for example:")
        s2_example = s2_scenes[0].utc_date() if s2_scenes else "2024-09-11"
        s1_example = s1_scenes[0].utc_date() if s1_scenes else "2024-09-07"
        print(f"    python acquisition/poc_test.py "
              f"--s2-date {s2_example} --s1-date {s1_example}")
        print()
        return

    # ----------------------------------------------------------------
    # Download mode
    # ----------------------------------------------------------------

    # Validate that the requested dates appear in the search results
    if args.s2_date:
        s2_dates = {sc.utc_date() for sc in s2_scenes}
        if args.s2_date not in s2_dates:
            print(f"\n  [WARNING] --s2-date {args.s2_date} was not found in the search results.")
            print(f"    Available S2 dates: {sorted(s2_dates)}")
            print(f"    The Processing API will still attempt to retrieve data for this date.")
            print(f"    If no data exists, it will fail with an error.")

    if args.s1_date:
        s1_dates = {sc.utc_date() for sc in s1_scenes}
        if args.s1_date not in s1_dates:
            print(f"\n  [WARNING] --s1-date {args.s1_date} was not found in the search results.")
            print(f"    Available S1 dates: {sorted(s1_dates)}")

    # ----------------------------------------------------------------
    # Build cache directories
    # ----------------------------------------------------------------

    from backend.services.satellite.cdse_sentinelhub import S2_BAND_NAMES, S1_BAND_NAMES

    cache_key_s2 = compute_cache_key(
        provider="cdse_sentinelhub",
        collection="sentinel-2-l1c",
        bbox=bbox,
        target_date=args.s2_date or "",
        bands=S2_BAND_NAMES,
        output_size_px=output_size,
    )
    cache_key_s1 = compute_cache_key(
        provider="cdse_sentinelhub",
        collection="sentinel-1-grd",
        bbox=bbox,
        target_date=args.s1_date or "",
        bands=S1_BAND_NAMES,
        output_size_px=output_size,
    )

    s2_dir = os.path.join(CACHE_DIR, cache_key_s2)
    s1_dir = os.path.join(CACHE_DIR, cache_key_s1)

    os.makedirs(s2_dir, exist_ok=True)
    os.makedirs(s1_dir, exist_ok=True)

    s2_tif  = os.path.join(s2_dir, "s2.tif")
    s1_tif  = os.path.join(s1_dir, "s1.tif")

    # ----------------------------------------------------------------
    # Download Sentinel-2
    # ----------------------------------------------------------------
    if args.s2_date:
        section(f"Downloading Sentinel-2 L1C  ({args.s2_date})")

        if os.path.isfile(s2_tif):
            size_kb = os.path.getsize(s2_tif) / 1024
            print(f"  [CACHE] {s2_tif} already exists ({size_kb:.1f} KB). Skipping download.")
        else:
            try:
                provider.download_sentinel2(
                    bbox=bbox,
                    target_date=args.s2_date,
                    output_path=s2_tif,
                    output_size_px=output_size,
                )
            except RuntimeError as exc:
                print(f"\n  ✗ S2 download failed:\n\n{exc}")
                sys.exit(1)

        # Save S2 metadata
        s2_meta_scene = next(
            (sc for sc in s2_scenes if sc.utc_date() == args.s2_date), None
        )
        s2_meta = {
            "provider":    "cdse_sentinelhub",
            "collection":  "sentinel-2-l1c",
            "aoi":         {"bbox": bbox.to_list()},
            "target_date": args.s2_date,
            "bands":       S2_BAND_NAMES,
            "output_size": list(output_size),
            "units_expected": "DN [0, 10000]",
            "normalization":  "clip(0,10000) / 10000  [preprocessing.py::normalize_s2]",
            "scene_id":   s2_meta_scene.scene_id if s2_meta_scene else None,
            "cloud_cover": s2_meta_scene.cloud_cover if s2_meta_scene else None,
            "cache_key":   cache_key_s2,
            "poc_created": datetime.datetime.utcnow().isoformat() + "Z",
        }
        with open(os.path.join(s2_dir, "metadata.json"), "w") as fh:
            json.dump(s2_meta, fh, indent=2, default=str)

        # Validate
        validate_tiff(s2_tif, "Sentinel-2 L1C")

        # Normalization diagnostic
        diagnose_s2_normalization(s2_tif)

        # Preview
        subsection("S2 Preview")
        s2_preview = os.path.join(s2_dir, "s2_preview.png")
        try:
            save_s2_preview(s2_tif, s2_preview)
        except Exception as exc:
            print(f"  [WARN] S2 preview generation failed: {exc}")

    # ----------------------------------------------------------------
    # Download Sentinel-1
    # ----------------------------------------------------------------
    if args.s1_date:
        section(f"Downloading Sentinel-1 GRD  ({args.s1_date})")

        if os.path.isfile(s1_tif):
            size_kb = os.path.getsize(s1_tif) / 1024
            print(f"  [CACHE] {s1_tif} already exists ({size_kb:.1f} KB). Skipping download.")
        else:
            try:
                provider.download_sentinel1(
                    bbox=bbox,
                    target_date=args.s1_date,
                    output_path=s1_tif,
                    output_size_px=output_size,
                )
            except RuntimeError as exc:
                print(f"\n  ✗ S1 download failed:\n\n{exc}")
                sys.exit(1)

        # Save S1 metadata
        s1_meta_scene = next(
            (sc for sc in s1_scenes if sc.utc_date() == args.s1_date), None
        )
        s1_meta = {
            "provider":      "cdse_sentinelhub",
            "collection":    "sentinel-1-grd",
            "aoi":           {"bbox": bbox.to_list()},
            "target_date":   args.s1_date,
            "bands":         S1_BAND_NAMES,
            "output_size":   list(output_size),
            "units_expected": "dB (decibel)",
            "normalization":  "clip(-25,0) -> (data+25)/25  [preprocessing.py::normalize_s1]",
            "scene_id":      s1_meta_scene.scene_id if s1_meta_scene else None,
            "polarizations": s1_meta_scene.polarizations if s1_meta_scene else None,
            "orbit_direction": s1_meta_scene.orbit_direction if s1_meta_scene else None,
            "instrument_mode": s1_meta_scene.instrument_mode if s1_meta_scene else None,
            "cache_key":     cache_key_s1,
            "poc_created":   datetime.datetime.utcnow().isoformat() + "Z",
        }
        with open(os.path.join(s1_dir, "metadata.json"), "w") as fh:
            json.dump(s1_meta, fh, indent=2, default=str)

        # Validate
        validate_tiff(s1_tif, "Sentinel-1 GRD")

        # Normalization diagnostic
        diagnose_s1_normalization(s1_tif)

        # Preview
        subsection("S1 Preview")
        s1_preview = os.path.join(s1_dir, "s1_preview.png")
        try:
            save_s1_preview(s1_tif, s1_preview)
        except Exception as exc:
            print(f"  [WARN] S1 preview generation failed: {exc}")

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    section("Phase 1 PoC Summary")

    if args.s2_date:
        print(f"  S2 TIFF:       {s2_tif}")
        print(f"  S2 metadata:   {os.path.join(s2_dir, 'metadata.json')}")
        print(f"  S2 preview:    {os.path.join(s2_dir, 's2_preview.png')}")
    if args.s1_date:
        print(f"  S1 TIFF:       {s1_tif}")
        print(f"  S1 metadata:   {os.path.join(s1_dir, 'metadata.json')}")
        print(f"  S1 preview:    {os.path.join(s1_dir, 's1_preview.png')}")

    print()
    print("  " + "-" * 56)
    print("  IMPORTANT REMINDER:")
    print("  Successful download does NOT mean the data is")
    print("  compatible with the ECRformer checkpoint.")
    print()
    print("  Before connecting to the model, verify:")
    print("    [ ] S2 band count = 13, dtype = uint16 or compatible")
    print("    [ ] S2 value range ~ [0, 10000] after download (not [0,1] float)")
    print("    [ ] S1 band count = 2 (VV, VH), dtype = float32")
    print("    [ ] S1 values are in dB scale (typically [-30, 5])")
    print("    [ ] Both files open correctly with rasterio")
    print("    [ ] CRS and transform are present and geographically correct")
    print("    [ ] Spatial alignment of S1 and S2 is still needed (Phase 2)")
    print("  " + "-" * 56)
    print()
    print("  ECRformer inference: Phase 4 (after Phases 2 and 3)")
    print()


if __name__ == "__main__":
    main()
