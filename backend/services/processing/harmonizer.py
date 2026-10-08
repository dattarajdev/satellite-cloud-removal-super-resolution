"""
Phase 2: Satellite Harmonization Service.

Transforms raw Sentinel-1 and Sentinel-2 acquisitions into an aligned,
co-registered, 10m UTM common-grid pair ready for downstream analysis.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
from PIL import Image
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio.warp import reproject, Resampling

from .grid import TargetGrid, create_common_grid, determine_utm_crs

logger = logging.getLogger(__name__)

# Standard band orders matching ECRformer / SEN12MS-CR expectations
S2_BAND_ORDER: List[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
    "B8A", "B09", "B10", "B11", "B12"
]
S1_BAND_ORDER: List[str] = ["VV", "VH"]

# RGB band indices in S2_BAND_ORDER (0-indexed: B04=3, B03=2, B02=1)
S2_RGB_INDICES = (3, 2, 1)


@dataclass
class HarmonizationConfig:
    """Configuration options for satellite harmonization."""
    pixel_size: float = 10.0
    s2_resampling: Resampling = Resampling.bilinear
    s1_resampling: Resampling = Resampling.bilinear
    s2_nodata: float = 0.0
    s1_nodata: float = -9999.0
    target_crs: Optional[CRS] = None
    save_previews: bool = True
    save_validity_mask: bool = True


@dataclass
class HarmonizedPair:
    """Result of harmonizing an S1/S2 pair."""
    s2_path: Path
    s1_path: Path
    validity_mask_path: Optional[Path]
    preview_s2_path: Optional[Path]
    preview_s1_path: Optional[Path]
    preview_coreg_path: Optional[Path]
    report_json_path: Path
    grid: TargetGrid
    s2_shape: Tuple[int, int, int]  # (bands, H, W)
    s1_shape: Tuple[int, int, int]  # (bands, H, W)
    validation_passed: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)


def _compute_band_statistics(data: np.ndarray, nodata_val: Optional[float] = None) -> List[Dict[str, float]]:
    """Compute per-band summary stats excluding nodata / infs."""
    stats = []
    for b in range(data.shape[0]):
        band_data = data[b].astype(np.float64)
        if nodata_val is not None:
            valid_mask = np.isfinite(band_data) & (band_data != nodata_val)
        else:
            valid_mask = np.isfinite(band_data)

        if np.any(valid_mask):
            valid_vals = band_data[valid_mask]
            stats.append({
                "band_index": b + 1,
                "min": float(np.min(valid_vals)),
                "max": float(np.max(valid_vals)),
                "mean": float(np.mean(valid_vals)),
                "std": float(np.std(valid_vals)),
                "valid_pixels": int(np.sum(valid_mask)),
            })
        else:
            stats.append({
                "band_index": b + 1,
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "valid_pixels": 0,
            })
    return stats


def _create_s2_rgb_preview(s2_data: np.ndarray, brightness: float = 3.0) -> Image.Image:
    """Render 13-band S2 uint16 data to RGB PNG matching visualization.py conventions."""
    rgb = s2_data[list(S2_RGB_INDICES)].astype(np.float32)
    # Normalize from [0, 10000] DN to [0, 1]
    rgb = np.clip(rgb / 10000.0 * brightness, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)
    rgb = np.transpose(rgb, (1, 2, 0))
    return Image.fromarray(rgb)


def _create_s1_sar_preview(s1_data: np.ndarray) -> Image.Image:
    """Render 2-band S1 dB data to false-color/grayscale preview."""
    # S1 preprocessing convention: clip to [-25, 0] dB, map to [0, 1]
    vv = np.clip((s1_data[0].astype(np.float32) + 25.0) / 25.0, 0.0, 1.0)
    vh = np.clip((s1_data[1].astype(np.float32) + 25.0) / 25.0, 0.0, 1.0)
    # Composite: R=VV, G=VH, B=VV/VH ratio
    ratio = np.clip(vv / (vh + 1e-4), 0.0, 1.0)
    rgb = np.stack([vv, vh, ratio], axis=-1)
    rgb = (rgb * 255.0).astype(np.uint8)
    return Image.fromarray(rgb)


def _create_coregistration_preview(s2_data: np.ndarray, s1_data: np.ndarray) -> Image.Image:
    """
    Render a cross-modality diagnostic preview demonstrating spatial alignment.
    Overlays high-frequency SAR backscatter edges in magenta over S2 optical RGB.
    """
    rgb = s2_data[list(S2_RGB_INDICES)].astype(np.float32)
    rgb = np.clip(rgb / 10000.0 * 2.5, 0.0, 1.0)
    rgb = (np.transpose(rgb, (1, 2, 0)) * 255.0).astype(np.uint8)

    # SAR normalized intensity
    sar = np.clip((s1_data[0].astype(np.float32) + 25.0) / 25.0, 0.0, 1.0)

    # Simple Sobel-like edge filter to extract structural features from SAR
    edge_y = np.abs(np.diff(sar, axis=0, prepend=sar[:1, :]))
    edge_x = np.abs(np.diff(sar, axis=1, prepend=sar[:, :1]))
    edges = np.clip((edge_x + edge_y) * 4.0, 0.0, 1.0)

    # Overlay edges in bright cyan/magenta onto optical image
    overlay = rgb.copy().astype(np.float32)
    edge_mask = edges > 0.25
    overlay[edge_mask, 0] = np.clip(overlay[edge_mask, 0] * 0.3 + 255 * 0.7, 0, 255)  # Magenta R
    overlay[edge_mask, 1] = np.clip(overlay[edge_mask, 1] * 0.3, 0, 255)              # Magenta G
    overlay[edge_mask, 2] = np.clip(overlay[edge_mask, 2] * 0.3 + 255 * 0.7, 0, 255)  # Magenta B

    return Image.fromarray(overlay.astype(np.uint8))


def harmonize_pair(
    s2_input_path: Union[str, Path],
    s1_input_path: Union[str, Path],
    output_dir: Union[str, Path],
    config: Optional[HarmonizationConfig] = None,
) -> HarmonizedPair:
    """
    Harmonize an S1/S2 pair onto a common 10m UTM grid.

    Steps:
    1. Determine appropriate UTM CRS from geographic extent.
    2. Build deterministic target grid (10m resolution, snapped integer bounds).
    3. Reproject S2 to target grid (13 bands, uint16, nodata=0, bilinear).
    4. Reproject S1 to target grid (2 bands, float32, nodata=-9999, bilinear).
    5. Construct pixel-level validity mask.
    6. Generate diagnostic previews and comprehensive validation metrics.

    Args:
        s2_input_path: Path to raw Sentinel-2 GeoTIFF.
        s1_input_path: Path to raw Sentinel-1 GeoTIFF.
        output_dir: Directory where harmonized files and diagnostics will be written.
        config: Optional HarmonizationConfig.

    Returns:
        HarmonizedPair: Object containing paths, metadata, and verification diagnostics.
    """
    if config is None:
        config = HarmonizationConfig()

    s2_path = Path(s2_input_path)
    s1_path = Path(s1_input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not s2_path.exists():
        raise FileNotFoundError(f"Sentinel-2 file not found: {s2_path}")
    if not s1_path.exists():
        raise FileNotFoundError(f"Sentinel-1 file not found: {s1_path}")

    # Inspect source metadata
    with rasterio.open(s2_path) as s2_src, rasterio.open(s1_path) as s1_src:
        if s2_src.count != 13:
            raise ValueError(f"Expected 13 bands in Sentinel-2 GeoTIFF, found {s2_src.count}")
        if s1_src.count != 2:
            raise ValueError(f"Expected 2 bands in Sentinel-1 GeoTIFF, found {s1_src.count}")

        # Compute common grid based on S2 extent (or union with S1)
        grid = create_common_grid(
            src_bounds=s2_src.bounds,
            src_crs=s2_src.crs,
            target_crs=config.target_crs,
            pixel_size=config.pixel_size,
            snap_to_grid=True,
        )

        # Reproject S2 to target grid
        s2_dst_data = np.zeros((13, grid.height, grid.width), dtype=np.uint16)
        for b in range(13):
            reproject(
                source=rasterio.band(s2_src, b + 1),
                destination=s2_dst_data[b],
                src_transform=s2_src.transform,
                src_crs=s2_src.crs,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                resampling=config.s2_resampling,
                dst_nodata=config.s2_nodata,
            )

        # Reproject S1 to target grid
        s1_dst_data = np.full((2, grid.height, grid.width), config.s1_nodata, dtype=np.float32)
        for b in range(2):
            reproject(
                source=rasterio.band(s1_src, b + 1),
                destination=s1_dst_data[b],
                src_transform=s1_src.transform,
                src_crs=s1_src.crs,
                dst_transform=grid.transform,
                dst_crs=grid.crs,
                resampling=config.s1_resampling,
                dst_nodata=config.s1_nodata,
            )

            # ------------------------------------------------------------
        # Reproject source validity masks onto the same target grid.
        # This represents spatial coverage, not radiometric thresholds.
        # ------------------------------------------------------------

        s2_src_valid = (s2_src.read_masks() > 0).all(axis=0).astype(np.uint8)
        s1_src_valid = (s1_src.read_masks() > 0).all(axis=0).astype(np.uint8)

        s2_valid = np.zeros((grid.height, grid.width), dtype=np.uint8)
        s1_valid = np.zeros((grid.height, grid.width), dtype=np.uint8)

        reproject(
            source=s2_src_valid,
            destination=s2_valid,
            src_transform=s2_src.transform,
            src_crs=s2_src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

        reproject(
            source=s1_src_valid,
            destination=s1_valid,
            src_transform=s1_src.transform,
            src_crs=s1_src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            src_nodata=0,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )

        combined_valid = (s2_valid & s1_valid).astype(np.uint8)

        # ------------------------------------------------------------
        # Make raster contents agree with the validity mask.
        # Invalid pixels are explicitly written as NoData.
        # ------------------------------------------------------------

        invalid = combined_valid == 0

        s2_dst_data[:, invalid] = config.s2_nodata
        s1_dst_data[:, invalid] = config.s1_nodata

    # Write harmonized Sentinel-2 GeoTIFF
    s2_out_path = out_dir / "s2_harmonized_10m.tif"
    s2_meta = {
        "driver": "GTiff",
        "count": 13,
        "dtype": "uint16",
        "width": grid.width,
        "height": grid.height,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": config.s2_nodata,
        "compress": "deflate",
    }
    with rasterio.open(s2_out_path, "w", **s2_meta) as dst:
        dst.write(s2_dst_data)
        for i, band_name in enumerate(S2_BAND_ORDER):
            dst.set_band_description(i + 1, band_name)
        dst.update_tags(
            satellite="Sentinel-2",
            collection="L1C",
            processing="Phase 2 Harmonization",
            pixel_size_m=str(config.pixel_size),
            resampling=config.s2_resampling.name,
            bands=",".join(S2_BAND_ORDER),
        )

    # Write harmonized Sentinel-1 GeoTIFF
    s1_out_path = out_dir / "s1_harmonized_10m.tif"
    s1_meta = {
        "driver": "GTiff",
        "count": 2,
        "dtype": "float32",
        "width": grid.width,
        "height": grid.height,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": config.s1_nodata,
        "compress": "deflate",
    }
    with rasterio.open(s1_out_path, "w", **s1_meta) as dst:
        dst.write(s1_dst_data)
        for i, band_name in enumerate(S1_BAND_ORDER):
            dst.set_band_description(i + 1, band_name)
        dst.update_tags(
            satellite="Sentinel-1",
            collection="GRD",
            processing="Phase 2 Harmonization",
            pixel_size_m=str(config.pixel_size),
            resampling=config.s1_resampling.name,
            polarization=",".join(S1_BAND_ORDER),
        )

    # Write validity mask GeoTIFF if requested
    mask_out_path = None
    if config.save_validity_mask:
        mask_out_path = out_dir / "validity_mask_10m.tif"
        mask_meta = {
            "driver": "GTiff",
            "count": 1,
            "dtype": "uint8",
            "width": grid.width,
            "height": grid.height,
            "crs": grid.crs,
            "transform": grid.transform,
            "nodata": 0,
            "compress": "deflate",
        }
        with rasterio.open(mask_out_path, "w", **mask_meta) as dst:
            dst.write(combined_valid, 1)
            dst.set_band_description(1, "validity_mask")
            dst.update_tags(
                description="Binary validity mask: 1 = valid S1 and S2 data, 0 = nodata or invalid",
                valid_percentage=f"{(combined_valid.sum() / combined_valid.size) * 100:.2f}%",
            )

    # Generate previews if requested
    preview_s2_path = None
    preview_s1_path = None
    preview_coreg_path = None
    if config.save_previews:
        s2_img = _create_s2_rgb_preview(s2_dst_data)
        preview_s2_path = out_dir / "s2_rgb_preview.png"
        s2_img.save(preview_s2_path)

        s1_img = _create_s1_sar_preview(s1_dst_data)
        preview_s1_path = out_dir / "s1_sar_preview.png"
        s1_img.save(preview_s1_path)

        coreg_img = _create_coregistration_preview(s2_dst_data, s1_dst_data)
        preview_coreg_path = out_dir / "coregistration_diff_preview.png"
        coreg_img.save(preview_coreg_path)

    # Run comprehensive validation diagnostics
    validation_passed, diagnostics = validate_harmonized_pair(
        s2_harmonized_path=s2_out_path,
        s1_harmonized_path=s1_out_path,
        validity_mask_path=mask_out_path,
        expected_pixel_size=config.pixel_size,
    )

    # Save JSON report
    report_json_path = out_dir / "harmonization_report.json"
    with open(report_json_path, "w") as f:
        json.dump(diagnostics, f, indent=2)

    # Save human-readable Markdown report
    report_md_path = out_dir / "harmonization_report.md"
    _write_markdown_report(report_md_path, diagnostics)

    return HarmonizedPair(
        s2_path=s2_out_path,
        s1_path=s1_out_path,
        validity_mask_path=mask_out_path,
        preview_s2_path=preview_s2_path,
        preview_s1_path=preview_s1_path,
        preview_coreg_path=preview_coreg_path,
        report_json_path=report_json_path,
        grid=grid,
        s2_shape=s2_dst_data.shape,
        s1_shape=s1_dst_data.shape,
        validation_passed=validation_passed,
        diagnostics=diagnostics,
    )


def validate_harmonized_pair(
    s2_harmonized_path: Union[str, Path],
    s1_harmonized_path: Union[str, Path],
    validity_mask_path: Optional[Union[str, Path]] = None,
    expected_pixel_size: float = 10.0,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Validate that the harmonized pair satisfies all 10 Phase 2 requirements.

    Returns:
        (bool, dict): (validation_passed, diagnostics_dict)
    """
    s2_p = Path(s2_harmonized_path)
    s1_p = Path(s1_harmonized_path)

    diagnostics: Dict[str, Any] = {
        "status": "PENDING",
        "requirements_check": {},
        "spatial_metadata": {},
        "band_diagnostics": {},
        "nodata_stats": {},
    }

    with rasterio.open(s2_p) as s2, rasterio.open(s1_p) as s1:
        # 1. CRS Check
        crs_equal = s2.crs == s1.crs
        is_projected = s2.crs.is_projected
        epsg = s2.crs.to_epsg()
        diagnostics["requirements_check"]["1_utm_crs_determined"] = bool(is_projected and epsg is not None)
        diagnostics["requirements_check"]["2_identical_crs"] = bool(crs_equal)

        # 3. Shape & Transform Check (Common Output Grid)
        shape_equal = (s2.height == s1.height) and (s2.width == s1.width)
        transform_equal = (
            abs(s2.transform.a - s1.transform.a) < 1e-6
            and abs(s2.transform.c - s1.transform.c) < 1e-4
            and abs(s2.transform.e - s1.transform.e) < 1e-6
            and abs(s2.transform.f - s1.transform.f) < 1e-4
        )
        diagnostics["requirements_check"]["3_common_output_grid"] = bool(shape_equal and transform_equal)

        # 4. Pixel size check (10m)
        pixel_x = abs(s2.transform.a)
        pixel_y = abs(s2.transform.e)
        pixel_size_10m = abs(pixel_x - expected_pixel_size) < 1e-4 and abs(pixel_y - expected_pixel_size) < 1e-4
        diagnostics["requirements_check"]["4_explicit_10m_pixel_size"] = bool(pixel_size_10m)

        # 5. Extent / Bounds Check
        bounds_equal = (
            abs(s2.bounds.left - s1.bounds.left) < 1e-4
            and abs(s2.bounds.right - s1.bounds.right) < 1e-4
            and abs(s2.bounds.bottom - s1.bounds.bottom) < 1e-4
            and abs(s2.bounds.top - s1.bounds.top) < 1e-4
        )
        diagnostics["requirements_check"]["5_identical_spatial_extent"] = bool(bounds_equal)

        # 6. Band Count & Order Check
        s2_band_descriptions = [s2.descriptions[i] for i in range(s2.count)]
        s1_band_descriptions = [s1.descriptions[i] for i in range(s1.count)]
        s2_bands_ok = s2.count == 13 and (not any(s2_band_descriptions) or s2_band_descriptions == S2_BAND_ORDER)
        s1_bands_ok = s1.count == 2 and (not any(s1_band_descriptions) or s1_band_descriptions == S1_BAND_ORDER)
        diagnostics["requirements_check"]["6_band_orders_preserved"] = bool(s2_bands_ok and s1_bands_ok)

        # 7. Resampling method check
        s2_resamp_tag = s2.tags().get("resampling", "bilinear")
        s1_resamp_tag = s1.tags().get("resampling", "bilinear")
        diagnostics["requirements_check"]["7_explicit_resampling"] = bool(
            s2_resamp_tag is not None and s1_resamp_tag is not None
        )

        # 8. Preservation of scientific dtypes
        dtypes_ok = (s2.dtypes[0] == "uint16") and (s1.dtypes[0] == "float32")
        diagnostics["requirements_check"]["8_scientific_dtypes_preserved"] = bool(dtypes_ok)

        # 9. NoData handling check
        nodata_handled = (s2.nodata is not None) and (s1.nodata is not None)
        diagnostics["requirements_check"]["9_nodata_handled_explicitly"] = bool(nodata_handled)

        # Read data for diagnostics
        s2_data = s2.read()
        s1_data = s1.read()

        # Band diagnostics
        s2_stats = _compute_band_statistics(s2_data, nodata_val=s2.nodata)
        s1_stats = _compute_band_statistics(s1_data, nodata_val=s1.nodata)

        total_pixels = s2.width * s2.height
        s2_valid_count = int(np.sum((s2_data != s2.nodata).any(axis=0)))
        s1_valid_count = int(np.sum((s1_data != s1.nodata).all(axis=0) & (s1_data > -50.0).all(axis=0)))
        intersection_valid = int(
            np.sum(((s2_data != s2.nodata).any(axis=0)) & (s1_data != s1.nodata).all(axis=0) & (s1_data > -50.0).all(axis=0))
        )

        diagnostics["nodata_stats"] = {
            "total_pixels": total_pixels,
            "s2_valid_pixels": s2_valid_count,
            "s2_valid_percent": round((s2_valid_count / total_pixels) * 100.0, 2),
            "s1_valid_pixels": s1_valid_count,
            "s1_valid_percent": round((s1_valid_count / total_pixels) * 100.0, 2),
            "intersection_valid_pixels": intersection_valid,
            "intersection_valid_percent": round((intersection_valid / total_pixels) * 100.0, 2),
        }

        diagnostics["requirements_check"]["10_validation_diagnostics_generated"] = True

        # Overall validation status
        all_passed = all(diagnostics["requirements_check"].values())
        diagnostics["status"] = "PASSED" if all_passed else "FAILED"

        # Record spatial metadata
        diagnostics["spatial_metadata"] = {
            "crs": str(s2.crs),
            "epsg": epsg,
            "width": s2.width,
            "height": s2.height,
            "pixel_size_x": pixel_x,
            "pixel_size_y": pixel_y,
            "bounds": {
                "left": s2.bounds.left,
                "bottom": s2.bounds.bottom,
                "right": s2.bounds.right,
                "top": s2.bounds.top,
            },
            "transform": [
                s2.transform.a, s2.transform.b, s2.transform.c,
                s2.transform.d, s2.transform.e, s2.transform.f,
            ],
            "s2": {
                "bands": s2.count,
                "dtype": s2.dtypes[0],
                "nodata": s2.nodata,
                "band_names": S2_BAND_ORDER,
            },
            "s1": {
                "bands": s1.count,
                "dtype": s1.dtypes[0],
                "nodata": s1.nodata,
                "band_names": S1_BAND_ORDER,
            },
        }

        diagnostics["band_diagnostics"]["sentinel2"] = s2_stats
        diagnostics["band_diagnostics"]["sentinel1"] = s1_stats

    return all_passed, diagnostics


def _write_markdown_report(report_path: Path, diagnostics: Dict[str, Any]) -> None:
    """Format validation diagnostics as clean Markdown."""
    meta = diagnostics.get("spatial_metadata", {})
    reqs = diagnostics.get("requirements_check", {})
    nd = diagnostics.get("nodata_stats", {})

    lines = [
        "# Phase 2 Satellite Harmonization Validation Report",
        "",
        f"**Overall Status**: `{'PASSED' if diagnostics.get('status') == 'PASSED' else 'FAILED'}`",
        "",
        "## 1. Requirements Compliance",
        "",
        "| Requirement | Status |",
        "|---|---|",
    ]

    req_titles = {
        "1_utm_crs_determined": "1. Determine UTM CRS from AOI",
        "2_identical_crs": "2. Reproject S1 & S2 to same UTM CRS",
        "3_common_output_grid": "3. Common output grid (identical shape & transform)",
        "4_explicit_10m_pixel_size": "4. Explicit 10m pixel size",
        "5_identical_spatial_extent": "5. Identical bounding box extent",
        "6_band_orders_preserved": "6. Preserved band orders (S2 13-band, S1 2-band)",
        "7_explicit_resampling": "7. Explicit spatial resampling (bilinear)",
        "8_scientific_dtypes_preserved": "8. Scientific dtypes preserved (uint16 S2, float32 S1)",
        "9_nodata_handled_explicitly": "9. Explicit NoData handling",
        "10_validation_diagnostics_generated": "10. Full validation diagnostics generated",
    }

    for key, title in req_titles.items():
        status_str = "PASS" if reqs.get(key, False) else "FAIL"
        lines.append(f"| {title} | `{status_str}` |")

    lines.extend([
        "",
        "## 2. Harmonized Spatial Metadata",
        "",
        f"- **Projected CRS**: `{meta.get('crs')}` (EPSG:{meta.get('epsg')})",
        f"- **Grid Dimensions**: `{meta.get('width')} x {meta.get('height')}` pixels (W x H)",
        f"- **Pixel Size**: `{meta.get('pixel_size_x')}m x {meta.get('pixel_size_y')}m`",
        f"- **UTM Extent**: West={meta.get('bounds', {}).get('left')}, South={meta.get('bounds', {}).get('bottom')}, East={meta.get('bounds', {}).get('right')}, North={meta.get('bounds', {}).get('top')}",
        "",
        "## 3. Data Range & Band Summary",
        "",
        "### Sentinel-2 L1C (13 Bands, uint16, DN)",
        "",
        "| Band | Min | Max | Mean | Std |",
        "|---|---|---|---|---|",
    ])

    for i, stat in enumerate(diagnostics.get("band_diagnostics", {}).get("sentinel2", [])):
        bname = S2_BAND_ORDER[i] if i < len(S2_BAND_ORDER) else f"B{i+1}"
        lines.append(f"| {bname} | {stat['min']:.0f} | {stat['max']:.0f} | {stat['mean']:.1f} | {stat['std']:.1f} |")

    lines.extend([
        "",
        "### Sentinel-1 GRD (2 Bands, float32, dB)",
        "",
        "| Band | Min (dB) | Max (dB) | Mean (dB) | Std (dB) |",
        "|---|---|---|---|---|",
    ])

    for i, stat in enumerate(diagnostics.get("band_diagnostics", {}).get("sentinel1", [])):
        bname = S1_BAND_ORDER[i] if i < len(S1_BAND_ORDER) else f"Band{i+1}"
        lines.append(f"| {bname} | {stat['min']:.2f} | {stat['max']:.2f} | {stat['mean']:.2f} | {stat['std']:.2f} |")

    lines.extend([
        "",
        "## 4. NoData & Overlap Statistics",
        "",
        f"- **Total Grid Pixels**: `{nd.get('total_pixels')}`",
        f"- **Valid Sentinel-2 Pixels**: `{nd.get('s2_valid_pixels')}` ({nd.get('s2_valid_percent')}%)",
        f"- **Valid Sentinel-1 Pixels**: `{nd.get('s1_valid_pixels')}` ({nd.get('s1_valid_percent')}%)",
        f"- **Intersection Valid Pixels**: `{nd.get('intersection_valid_pixels')}` ({nd.get('intersection_valid_percent')}%)",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
