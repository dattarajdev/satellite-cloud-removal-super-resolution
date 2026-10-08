"""
Band selector and validation utilities for OpenSR diffusion super-resolution.

OpenSR LDSR-S2 is a 4-band RGB+NIR model expecting Sentinel-2 bands:
- B04 (Red, 665 nm)
- B03 (Green, 560 nm)
- B02 (Blue, 490 nm)
- B08 (Near-Infrared / NIR, 842 nm)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import rasterio


OPENSR_REQUIRED_BANDS: Tuple[str, ...] = ("B4", "B3", "B2", "B8")
OPENSR_FULL_NAMES: Tuple[str, ...] = ("B04", "B03", "B02", "B08")


def normalize_band_name(name: Optional[str]) -> Optional[str]:
    """Normalize band descriptions like 'B04', 'B4', 'band_4' to 'B4'."""
    if not name:
        return None
    normalized = name.strip().upper().replace("_", "").replace(" ", "")
    match = re.fullmatch(r"B0*(\d+)", normalized)
    return f"B{int(match.group(1))}" if match else normalized


def validate_and_get_band_indices(dataset: rasterio.DatasetReader) -> Tuple[int, ...]:
    """
    Validate that the GeoTIFF contains B4, B3, B2, and B8 and return 1-based indices.

    Supports:
    - 13-band Sentinel-2 products with standard descriptions (B01..B12)
    - 13-band Sentinel-2 products with unnamed bands (standard order)
    - 4-band RGBN products with descriptions or standard (B4, B3, B2, B8) order
    """
    raw_descriptions = [dataset.descriptions[i] for i in range(dataset.count)]
    normalized_names = [normalize_band_name(desc) for desc in raw_descriptions]
    named_map: Dict[str, int] = {
        name: idx + 1 for idx, name in enumerate(normalized_names) if name
    }

    # Check if all required bands are present by name
    if all(band in named_map for band in OPENSR_REQUIRED_BANDS):
        return tuple(named_map[band] for band in OPENSR_REQUIRED_BANDS)

    # Standard 13-band Sentinel-2 without descriptions: B1..B12 (1-based: 4, 3, 2, 8)
    if dataset.count == 13 and not any(normalized_names):
        return (4, 3, 2, 8)

    # 4-band unnamed RGBN
    if dataset.count == 4 and not any(normalized_names):
        return (1, 2, 3, 4)

    found_str = ", ".join(n or "unnamed" for n in normalized_names)
    raise ValueError(
        f"Input GeoTIFF does not contain required Sentinel-2 bands {OPENSR_REQUIRED_BANDS}. "
        f"Found {dataset.count} bands ({found_str}). "
        "The OpenSR model requires real B04, B03, B02, and B08 (NIR) data."
    )
