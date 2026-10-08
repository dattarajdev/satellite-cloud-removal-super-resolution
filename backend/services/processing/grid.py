"""
Grid and spatial reference system utilities for Phase 2 satellite harmonization.

Provides robust UTM CRS derivation and deterministic common target grid generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Union, Dict, Any

from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds
from rasterio.warp import transform_bounds


@dataclass(frozen=True)
class TargetGrid:
    """Represents a deterministic geospatial raster grid."""
    crs: CRS
    transform: Affine
    width: int
    height: int
    bounds: Tuple[float, float, float, float]  # min_x, min_y, max_x, max_y in target CRS
    pixel_size: float

    @property
    def resolution(self) -> Tuple[float, float]:
        """Return (res_x, res_y) in meters."""
        return (abs(self.transform.a), abs(self.transform.e))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize grid metadata to dictionary."""
        return {
            "crs": str(self.crs),
            "epsg": self.crs.to_epsg(),
            "width": self.width,
            "height": self.height,
            "pixel_size_m": self.pixel_size,
            "resolution": self.resolution,
            "bounds": list(self.bounds),
            "transform": [
                self.transform.a, self.transform.b, self.transform.c,
                self.transform.d, self.transform.e, self.transform.f,
            ],
        }


def determine_utm_crs(
    lon_or_bbox: Union[float, Tuple[float, float, float, float], list],
    lat: Union[float, None] = None,
) -> CRS:
    """
    Determine the appropriate WGS 84 / UTM zone CRS from coordinates.

    Args:
        lon_or_bbox: Either center longitude (float) or bounding box (min_lon, min_lat, max_lon, max_lat).
        lat: Center latitude (float), required if lon_or_bbox is a float.

    Returns:
        rasterio.crs.CRS: e.g. CRS.from_epsg(32633) for UTM Zone 33N.
    """
    if lat is None:
        if isinstance(lon_or_bbox, (tuple, list)) and len(lon_or_bbox) == 4:
            min_lon, min_lat, max_lon, max_lat = lon_or_bbox
            center_lon = (min_lon + max_lon) / 2.0
            center_lat = (min_lat + max_lat) / 2.0
        else:
            raise ValueError(
                "Must provide either (min_lon, min_lat, max_lon, max_lat) sequence or lon and lat floats."
            )
    else:
        center_lon = float(lon_or_bbox)
        center_lat = float(lat)

    # Normalize longitude to [-180, 180)
    norm_lon = ((center_lon + 180.0) % 360.0) - 180.0
    zone = int(math.floor((norm_lon + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))

    # EPSG: 326xx for North, 327xx for South
    epsg = 32600 + zone if center_lat >= 0.0 else 32700 + zone
    return CRS.from_epsg(epsg)


def create_common_grid(
    src_bounds: Tuple[float, float, float, float],
    src_crs: CRS,
    target_crs: Union[CRS, None] = None,
    pixel_size: float = 10.0,
    snap_to_grid: bool = True,
) -> TargetGrid:
    """
    Construct an explicit, deterministic target grid aligned to pixel_size meters.

    Args:
        src_bounds: (min_x, min_y, max_x, max_y) in src_crs.
        src_crs: CRS of input bounds (e.g. EPSG:4326).
        target_crs: Destination UTM CRS. If None, derived automatically from src_bounds.
        pixel_size: Grid spacing in meters (default: 10.0 m).
        snap_to_grid: If True, snap target bounds to integer multiples of pixel_size.

    Returns:
        TargetGrid: Common target grid with exact 10m pixels and matching transform.
    """
    if target_crs is None:
        # Determine target UTM CRS from WGS84 coordinates
        if src_crs.to_epsg() == 4326:
            target_crs = determine_utm_crs(src_bounds)
        else:
            # Transform bounds to EPSG:4326 to determine UTM zone
            wgs84_bounds = transform_bounds(src_crs, CRS.from_epsg(4326), *src_bounds)
            target_crs = determine_utm_crs(wgs84_bounds)

    # Transform bounding box into target CRS coordinates
    min_x, min_y, max_x, max_y = transform_bounds(
        src_crs, target_crs, *src_bounds
    )

    if snap_to_grid:
        min_x = math.floor(min_x / pixel_size) * pixel_size
        min_y = math.floor(min_y / pixel_size) * pixel_size
        max_x = math.ceil(max_x / pixel_size) * pixel_size
        max_y = math.ceil(max_y / pixel_size) * pixel_size

    width = int(round((max_x - min_x) / pixel_size))
    height = int(round((max_y - min_y) / pixel_size))

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid calculated grid dimensions: width={width}, height={height} from bounds {src_bounds}"
        )

    # rasterio from_bounds: west, south, east, north, width, height
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)

    return TargetGrid(
        crs=target_crs,
        transform=transform,
        width=width,
        height=height,
        bounds=(min_x, min_y, max_x, max_y),
        pixel_size=pixel_size,
    )
