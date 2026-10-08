"""
Preprocessing and normalization utilities for satellite imagery.
Supports Sentinel-1 SAR and Sentinel-2 optical data.
Implements mask-safe normalization guaranteeing that NoData/invalid pixels
are never clipped or interpreted as physical measurements.
"""

from typing import Optional, Dict, Any
import numpy as np
import rasterio


def normalize_s2_safe(image: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Mask-safe optical normalization for Sentinel-2 (ECRformer / SEN12MS-CR).
    
    Sequence:
        raw raster -> valid-mask selection -> normalize valid pixels -> invalid pixels explicitly set to 0.0.
    
    Valid pixels:
        Raw Sentinel-2 DN in [0, 10000] -> clipped [0, 10000] -> divided by 10000.0 -> [0.0, 1.0].
    Invalid pixels (where valid_mask == 0):
        Explicitly set to 0.0 (neutral model input).
    """
    image = image.astype(np.float32)
    normalized = np.zeros_like(image, dtype=np.float32)

    if valid_mask is not None:
        mask = valid_mask.squeeze().astype(bool)
        if mask.shape != image.shape[1:]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image spatial shape {image.shape[1:]}"
            )
        for c in range(image.shape[0]):
            valid_vals = image[c][mask]
            clipped = np.clip(valid_vals, 0.0, 10000.0)
            normalized[c][mask] = clipped / 10000.0
            normalized[c][~mask] = 0.0
    else:
        valid = np.isfinite(image) & (image > 0.0)
        clipped = np.clip(image, 0.0, 10000.0)
        normalized = np.where(valid, clipped / 10000.0, 0.0).astype(np.float32)

    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_s1_safe(image: np.ndarray, valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Mask-safe SAR normalization for Sentinel-1 (ECRformer / SEN12MS-CR).
    
    Sequence:
        raw raster -> valid-mask selection -> normalize valid pixels -> invalid pixels explicitly set to 0.0.
    
    Valid pixels:
        Raw SAR dB in [-25.0, 0.0] -> clipped [-25.0, 0.0] -> (data + 25.0) / 25.0 -> [0.0, 1.0].
    Invalid pixels (where valid_mask == 0 or NoData <= -500 dB):
        Explicitly set to 0.0 (neutral model input).
        CRITICAL: NoData (-9999.0) is NEVER passed to the linear transform or clipped to -25.0.
    """
    image = image.astype(np.float32)
    normalized = np.zeros_like(image, dtype=np.float32)

    if valid_mask is not None:
        mask = valid_mask.squeeze().astype(bool)
        if mask.shape != image.shape[1:]:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image spatial shape {image.shape[1:]}"
            )
        for c in range(image.shape[0]):
            channel_data = image[c]
            valid_subset = mask & np.isfinite(channel_data) & (channel_data > -500.0)
            clipped = np.clip(channel_data[valid_subset], -25.0, 0.0)
            normalized[c][valid_subset] = (clipped + 25.0) / 25.0
            normalized[c][~mask] = 0.0
    else:
        valid = np.isfinite(image) & (image > -500.0)
        clipped = np.clip(image, -25.0, 0.0)
        norm_vals = (clipped + 25.0) / 25.0
        normalized = np.where(valid, norm_vals, 0.0).astype(np.float32)

    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def normalize_s2(image: np.ndarray) -> np.ndarray:
    """Standard optical normalization (delegates to normalize_s2_safe)."""
    return normalize_s2_safe(image, valid_mask=None)


def normalize_s1(image: np.ndarray) -> np.ndarray:
    """Standard SAR normalization (delegates to normalize_s1_safe)."""
    return normalize_s1_safe(image, valid_mask=None)


def inspect_tiff(path) -> Dict[str, Any]:
    with rasterio.open(path) as src:
        return {
            "path": str(path),
            "bands": src.count,
            "width": src.width,
            "height": src.height,
            "dtype": str(src.dtypes[0]),
            "crs": str(src.crs),
            "nodata": src.nodata,
            "transform": src.transform,
        }


def validate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    with rasterio.open(sample["s1"]) as src:
        s1_bands = src.count
        s1_width = src.width
        s1_height = src.height

    with rasterio.open(sample["s2_cloudy"]) as src:
        cloudy_bands = src.count
        cloudy_width = src.width
        cloudy_height = src.height

    if s1_bands != 2:
        raise ValueError(f"S1 should have 2 bands. Found {s1_bands}.")

    if cloudy_bands != 13:
        raise ValueError(f"Cloudy S2 should have 13 bands. Found {cloudy_bands}.")

    if (s1_width, s1_height) != (cloudy_width, cloudy_height):
        raise ValueError(
            f"S1 dimensions ({s1_width}, {s1_height}) do not match cloudy S2 ({cloudy_width}, {cloudy_height})."
        )

    if "s2_clear" in sample and sample["s2_clear"] is not None:
        with rasterio.open(sample["s2_clear"]) as src:
            clear_bands = src.count
            clear_width = src.width
            clear_height = src.height
        if clear_bands != 13:
            raise ValueError(f"Clear S2 should have 13 bands. Found {clear_bands}.")
        if (clear_width, clear_height) != (cloudy_width, cloudy_height):
            raise ValueError("Clear S2 dimensions do not match cloudy S2.")

    return {
        "width": s1_width,
        "height": s1_height,
        "s1_bands": s1_bands,
        "cloudy_bands": cloudy_bands,
    }
