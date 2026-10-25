"""
Reference Adapter and Alignment Module.

Provides robust loading, validation, radiometric scaling, and geospatial alignment
for ground-truth and temporal reference GeoTIFFs used in model evaluation.

Contracts:
- Supports Streamlit UploadedFile, file paths, bytes, memory streams, and rasterio DatasetReader.
- Never depends on stream read cursor position (always resets / reads getvalue()).
- Validates 13 Sentinel-2 bands in canonical order (B01-B12).
- Geospatially reprojects reference to target prediction grid if extents/CRS/transforms differ.
- Accurately normalizes radiometric scale (DN [0, 10000] vs reflectance [0, 1]).
- Distinguishes paired ground truth from temporal references.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import DatasetReader, MemoryFile
from rasterio.warp import reproject

S2_CANONICAL_BANDS: List[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
    "B8A", "B09", "B10", "B11", "B12"
]


@dataclass
class ValidatedReference:
    """Standardized, validated reference container ready for quantitative evaluation."""
    array: np.ndarray  # Shape: (13, H, W) in float32 normalized reflectance [0.0, 1.0]
    valid_mask: np.ndarray  # Shape: (H, W) boolean mask
    profile: Dict[str, Any]
    reference_type: str  # "paired_ground_truth" or "temporal_reference"
    source_name: str
    radiometric_scale_applied: str
    reprojected: bool = False
    acquisition_date: Optional[str] = None
    temporal_distance_days: Optional[float] = None
    notes: List[str] = field(default_factory=list)


def _open_reference_dataset(reference_input: Any) -> Tuple[DatasetReader, Optional[MemoryFile]]:
    """
    Open any supported reference input safely without depending on stream cursor position.

    Returns:
        (reader, optional_memfile)
    """
    # 1. Streamlit UploadedFile or file-like object with getvalue()
    if hasattr(reference_input, "getvalue"):
        data_bytes = reference_input.getvalue()
        memfile = MemoryFile(data_bytes)
        return memfile.open(), memfile

    # 2. Raw bytes or bytearray
    if isinstance(reference_input, (bytes, bytearray)):
        memfile = MemoryFile(reference_input)
        return memfile.open(), memfile

    # 3. io.BytesIO or io.IOBase
    if isinstance(reference_input, io.IOBase):
        if hasattr(reference_input, "seek"):
            reference_input.seek(0)
        data_bytes = reference_input.read()
        memfile = MemoryFile(data_bytes)
        return memfile.open(), memfile

    # 4. MemoryFile
    if isinstance(reference_input, MemoryFile):
        return reference_input.open(), None

    # 5. DatasetReader
    if isinstance(reference_input, DatasetReader):
        return reference_input, None

    # 6. File path (str or Path)
    if isinstance(reference_input, (str, Path)):
        p = Path(reference_input).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Reference file not found: {p}")
        return rasterio.open(p), None

    raise TypeError(
        f"Unsupported reference_target type: {type(reference_input)}. "
        "Expected Streamlit UploadedFile, bytes, file path, or rasterio DatasetReader."
    )


def load_and_validate_reference(
    reference_input: Any,
    target_profile: Optional[Dict[str, Any]] = None,
    scale_mode: str = "auto",  # "auto", "0-1", "0-10000"
    reference_type: str = "paired_ground_truth",  # "paired_ground_truth" or "temporal_reference"
    acquisition_date: Optional[str] = None,
    temporal_distance_days: Optional[float] = None,
) -> ValidatedReference:
    """
    Load, validate, align, and normalize an optical Sentinel-2 reference GeoTIFF.

    Args:
        reference_input: Path, bytes, UploadedFile, or DatasetReader.
        target_profile: Profile of the prediction raster (width, height, crs, transform) for spatial alignment.
        scale_mode: "auto" (detect based on pixel values), "0-1", or "0-10000".
        reference_type: "paired_ground_truth" or "temporal_reference".
        acquisition_date: Optional acquisition date string of the reference.
        temporal_distance_days: Optional distance in days from the target cloudy scene.

    Returns:
        ValidatedReference container.
    """
    src, memfile = _open_reference_dataset(reference_input)
    notes: List[str] = []

    try:
        # 1. Validate band count
        if src.count != 13:
            raise ValueError(
                f"Reference GeoTIFF must contain exactly 13 Sentinel-2 bands. "
                f"Found {src.count} bands in {getattr(src, 'name', 'input')}."
            )

        # 2. Check band descriptions / mapping
        descriptions = [src.descriptions[i] for i in range(src.count)] if src.descriptions else []
        has_valid_desc = any(descriptions) and all(d is not None and d.strip() for d in descriptions)

        if has_valid_desc:
            # Check if all 13 canonical names are present
            upper_desc = [d.strip().upper() for d in descriptions]
            # Standardize names: e.g. "B8A" vs "B08A"
            norm_desc = [d.replace("B08A", "B8A") for d in upper_desc]
            if set(norm_desc) == set(S2_CANONICAL_BANDS):
                # Rearrange if band order is not canonical
                if norm_desc != S2_CANONICAL_BANDS:
                    band_map = [norm_desc.index(b) + 1 for b in S2_CANONICAL_BANDS]
                    raw_data = src.read(band_map).astype(np.float32)
                    notes.append("Reordered reference bands to match canonical order B01-B12.")
                else:
                    raw_data = src.read().astype(np.float32)
            else:
                raw_data = src.read().astype(np.float32)
                notes.append(
                    f"Custom band descriptions found ({descriptions}); "
                    "assumed sequential 13-band Sentinel-2 order (B01-B12)."
                )
        else:
            raw_data = src.read().astype(np.float32)
            notes.append("No band descriptions found; assumed canonical Sentinel-2 band order (B01-B12).")

        # 3. Spatial alignment / Geospatial reprojection if target_profile is supplied
        reprojected = False
        final_profile = src.profile.copy()

        if target_profile is not None:
            tgt_crs = target_profile.get("crs")
            tgt_transform = target_profile.get("transform")
            tgt_width = target_profile.get("width")
            tgt_height = target_profile.get("height")

            needs_reproject = False
            if tgt_width != src.width or tgt_height != src.height:
                needs_reproject = True
            elif str(tgt_crs) != str(src.crs):
                needs_reproject = True
            elif tgt_transform is not None and src.transform is not None:
                for a, b in zip(src.transform, tgt_transform):
                    if abs(a - b) > 1e-4:
                        needs_reproject = True
                        break

            if needs_reproject:
                if tgt_crs is None or tgt_transform is None or tgt_width is None or tgt_height is None:
                    raise ValueError("Target profile missing CRS, transform, width, or height for reprojection.")

                aligned_data = np.zeros((13, tgt_height, tgt_width), dtype=np.float32)
                for b in range(13):
                    reproject(
                        source=raw_data[b],
                        destination=aligned_data[b],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=tgt_transform,
                        dst_crs=tgt_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=src.nodata if src.nodata is not None else 0.0,
                        dst_nodata=0.0,
                    )
                raw_data = aligned_data
                reprojected = True
                final_profile.update({
                    "crs": tgt_crs,
                    "transform": tgt_transform,
                    "width": tgt_width,
                    "height": tgt_height,
                })
                notes.append(
                    f"Geospatially reprojected reference from {src.width}x{src.height} ({src.crs}) "
                    f"to prediction grid {tgt_width}x{tgt_height} ({tgt_crs}) using bilinear resampling."
                )

        # 4. Valid mask detection (finite values, not nodata, positive)
        nodata_val = src.nodata
        if nodata_val is not None:
            valid_mask = (raw_data[0] != nodata_val) & np.all(np.isfinite(raw_data), axis=0) & (raw_data[0] > 0.0)
        else:
            valid_mask = np.all(np.isfinite(raw_data), axis=0) & (raw_data[0] > 0.0)

        # 5. Radiometric scale normalization to [0.0, 1.0]
        finite_vals = raw_data[np.isfinite(raw_data)]
        max_val = float(np.nanmax(finite_vals)) if finite_vals.size > 0 else 0.0

        if scale_mode == "0-10000":
            normalized_array = np.clip(raw_data / 10000.0, 0.0, 1.0)
            applied_scale = "Explicit 0–10000 DN to [0.0, 1.0]"
        elif scale_mode == "0-1":
            normalized_array = np.clip(raw_data, 0.0, 1.0)
            applied_scale = "Explicit [0.0, 1.0] Reflectance"
        else:  # auto
            if max_val > 1.5:
                normalized_array = np.clip(raw_data / 10000.0, 0.0, 1.0)
                applied_scale = f"Auto-detected DN scale (max={max_val:.1f} > 1.5); normalized via /10000.0 to [0, 1]"
            else:
                normalized_array = np.clip(raw_data, 0.0, 1.0)
                applied_scale = f"Auto-detected reflectance scale (max={max_val:.3f} <= 1.5); preserved in [0, 1]"

        source_name = getattr(src, "name", "uploaded_reference.tif")
        if isinstance(source_name, str) and len(source_name) > 60:
            source_name = Path(source_name).name

        return ValidatedReference(
            array=normalized_array.astype(np.float32),
            valid_mask=valid_mask.astype(bool),
            profile=final_profile,
            reference_type=reference_type,
            source_name=source_name,
            radiometric_scale_applied=applied_scale,
            reprojected=reprojected,
            acquisition_date=acquisition_date,
            temporal_distance_days=temporal_distance_days,
            notes=notes,
        )

    finally:
        src.close()
        if memfile is not None:
            memfile.close()
