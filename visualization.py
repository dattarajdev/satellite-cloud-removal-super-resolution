"""
Visualization utilities for Sentinel-1 SAR and Sentinel-2 optical imagery.
"""

from typing import Optional, Tuple
import numpy as np
from PIL import Image


def to_rgb(
    image: np.ndarray,
    bands: Tuple[int, int, int] = (3, 2, 1),
    brightness: float = 2.5,
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert CHW 13-band Sentinel-2 image to RGB uint8 array (H, W, 3).
    Band order: B01=0, B02=1(Blue), B03=2(Green), B04=3(Red), ...
    RGB bands: (3, 2, 1) -> (B04, B03, B02)
    """
    if image.ndim != 3:
        raise ValueError(f"Expected CHW 3D array, received shape {image.shape}")

    rgb = image[list(bands)]
    rgb = np.transpose(rgb, (1, 2, 0))

    if rgb.max() > 1.0:
        rgb = np.clip(rgb / 10000.0, 0.0, 1.0)
    else:
        rgb = np.clip(rgb, 0.0, 1.0)

    rgb = np.clip(rgb * brightness, 0.0, 1.0)
    rgb_uint8 = (rgb * 255.0).astype(np.uint8)

    if valid_mask is not None:
        mask = valid_mask.squeeze().astype(bool)
        rgb_uint8[~mask] = 0

    return rgb_uint8


def to_pil(
    image: np.ndarray,
    bands: Tuple[int, int, int] = (3, 2, 1),
    brightness: float = 2.5,
    valid_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """Convert CHW 13-band image to PIL Image."""
    return Image.fromarray(to_rgb(image, bands=bands, brightness=brightness, valid_mask=valid_mask))


def sar_preview(
    s1: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """
    Create a grayscale SAR visualization (PIL Image) from 2-band SAR (VV, VH).
    Uses VV channel (band 0).
    """
    if s1.ndim != 3 or s1.shape[0] < 1:
        raise ValueError(f"Expected (C, H, W) SAR array with >= 1 channels, got {s1.shape}")

    channel = s1[0].copy()

    if channel.min() < -1.0 or channel.max() > 1.0:
        valid = channel > -500.0
        norm = np.zeros_like(channel, dtype=np.float32)
        norm[valid] = np.clip((channel[valid] + 25.0) / 25.0, 0.0, 1.0)
    else:
        norm = np.clip(channel, 0.0, 1.0)

    if valid_mask is not None:
        mask = valid_mask.squeeze().astype(bool)
        norm[~mask] = 0.0

    sar_uint8 = (norm * 255.0).astype(np.uint8)
    return Image.fromarray(sar_uint8, mode="L")
