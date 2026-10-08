"""
Abstract satellite data provider interface.

All concrete providers (CDSE/Sentinel Hub, Microsoft Planetary Computer,
Google Earth Engine, etc.) must implement SatelliteProvider.

This keeps the acquisition layer replaceable without touching the
downstream preprocessing or inference pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
import datetime
import math


# -----------------------------------------------------------------------
# Coordinate and scene data structures
# -----------------------------------------------------------------------

@dataclass
class BoundingBox:
    """
    Geographic bounding box in WGS84 (EPSG:4326).

    All values are decimal degrees.
    Convention: [west, south, east, north] for API requests.
    """

    west:  float
    south: float
    east:  float
    north: float

    def to_list(self) -> list[float]:
        """Return [west, south, east, north] — standard GeoJSON bbox order."""
        return [self.west, self.south, self.east, self.north]

    def width_degrees(self) -> float:
        return self.east - self.west

    def height_degrees(self) -> float:
        return self.north - self.south

    def approx_width_m(self) -> float:
        """Approximate east-west extent in metres at the bbox's mid-latitude."""
        lat_mid = (self.south + self.north) / 2.0
        return self.width_degrees() * 111_320.0 * math.cos(math.radians(lat_mid))

    def approx_height_m(self) -> float:
        """Approximate north-south extent in metres."""
        return self.height_degrees() * 110_540.0

    def approx_gsd_m(self, width_px: int, height_px: int) -> tuple[float, float]:
        """
        Approximate ground sampling distance (metres/pixel) for a given
        output pixel grid.

        Returns (gsd_ew, gsd_ns).
        """
        return (
            self.approx_width_m() / width_px,
            self.approx_height_m() / height_px,
        )

    def __repr__(self) -> str:
        return (
            f"BoundingBox(W={self.west:.5f}, S={self.south:.5f}, "
            f"E={self.east:.5f}, N={self.north:.5f})"
        )


@dataclass
class S2Scene:
    """
    Sentinel-2 candidate scene returned by catalogue search.

    cloud_cover: percentage 0–100, or None if unavailable.
    footprint:   GeoJSON geometry dict, or None.
    extra:       Raw provider metadata (provider-specific, do not rely on).
    """

    scene_id:         str
    dt:               datetime.datetime
    cloud_cover:      Optional[float]
    processing_level: str                    # "L1C" or "L2A"
    footprint:        Optional[dict]
    extra:            dict = field(default_factory=dict)

    def cloud_str(self) -> str:
        if self.cloud_cover is None:
            return "  N/A "
        return f"{self.cloud_cover:5.1f}%"

    def utc_date(self) -> str:
        return self.dt.strftime("%Y-%m-%d")

    def utc_time(self) -> str:
        return self.dt.strftime("%H:%M:%S")


@dataclass
class S1Scene:
    """
    Sentinel-1 candidate scene returned by catalogue search.

    polarizations:   list of strings e.g. ["VV", "VH"]
    orbit_direction: "ASCENDING" | "DESCENDING" | None
    instrument_mode: "IW" | "EW" | "SM" | None
    footprint:       GeoJSON geometry dict, or None.
    extra:           Raw provider metadata.
    """

    scene_id:        str
    dt:              datetime.datetime
    product_type:    str               # e.g. "GRD"
    polarizations:   list[str]
    orbit_direction: Optional[str]
    instrument_mode: Optional[str]
    footprint:       Optional[dict]
    extra:           dict = field(default_factory=dict)

    def has_vv_vh(self) -> bool:
        """True if both VV and VH polarizations are available."""
        return "VV" in self.polarizations and "VH" in self.polarizations

    def pol_str(self) -> str:
        return " + ".join(self.polarizations) if self.polarizations else "unknown"

    def utc_date(self) -> str:
        return self.dt.strftime("%Y-%m-%d")

    def utc_time(self) -> str:
        return self.dt.strftime("%H:%M:%S")


# -----------------------------------------------------------------------
# Abstract provider interface
# -----------------------------------------------------------------------

class SatelliteProvider(ABC):
    """
    Abstract interface for satellite data providers.

    Implementations must be independently swappable so the rest of the
    pipeline (preprocessing, ECRformer inference, output) never depends
    on provider-specific details.
    """

    @abstractmethod
    def search_sentinel2(
        self,
        bbox: BoundingBox,
        start_date: str,
        end_date: str,
        max_results: int = 20,
    ) -> list[S2Scene]:
        """
        Search the Sentinel-2 L1C catalogue.

        Args:
            bbox:        Area of interest in WGS84.
            start_date:  Inclusive start date, "YYYY-MM-DD".
            end_date:    Inclusive end date, "YYYY-MM-DD".
            max_results: Maximum number of items to return.

        Returns:
            List of S2Scene objects, sorted by acquisition datetime.
        """

    @abstractmethod
    def search_sentinel1(
        self,
        bbox: BoundingBox,
        start_date: str,
        end_date: str,
        max_results: int = 20,
    ) -> list[S1Scene]:
        """
        Search the Sentinel-1 GRD catalogue.

        Args:
            bbox:        Area of interest in WGS84.
            start_date:  Inclusive start date, "YYYY-MM-DD".
            end_date:    Inclusive end date, "YYYY-MM-DD".
            max_results: Maximum number of items to return.

        Returns:
            List of S1Scene objects, sorted by acquisition datetime.
        """

    @abstractmethod
    def download_sentinel2(
        self,
        bbox: BoundingBox,
        target_date: str,
        output_path: str,
        output_size_px: tuple[int, int] = (256, 256),
    ) -> None:
        """
        Download Sentinel-2 L1C as a 13-band GeoTIFF.

        The band order MUST match the ECRformer/SEN12MS-CR convention:

            Index  Band   Description
              0    B01    Coastal aerosol (60 m native)
              1    B02    Blue           (10 m native)
              2    B03    Green          (10 m native)
              3    B04    Red            (10 m native)
              4    B05    Veg. red edge  (20 m native)
              5    B06    Veg. red edge  (20 m native)
              6    B07    Veg. red edge  (20 m native)
              7    B08    NIR            (10 m native)
              8    B8A    Narrow NIR     (20 m native)
              9    B09    Water vapour   (60 m native)
             10    B10    Cirrus         (60 m native)
             11    B11    SWIR           (20 m native)
             12    B12    SWIR           (20 m native)

        Expected output units: DN, uint16, range [0, 10000].
        (Matches SEN12MS-CR TOA reflectance × 10000.)

        Args:
            bbox:           Area of interest in WGS84.
            target_date:    Scene date, "YYYY-MM-DD".
            output_path:    Destination GeoTIFF path.
            output_size_px: (width, height) in pixels.
        """

    @abstractmethod
    def download_sentinel1(
        self,
        bbox: BoundingBox,
        target_date: str,
        output_path: str,
        output_size_px: tuple[int, int] = (256, 256),
    ) -> None:
        """
        Download Sentinel-1 GRD as a 2-band (VV, VH) GeoTIFF in dB.

        Band order:
            Index  Band   Description
              0    VV     Vertical-Vertical backscatter
              1    VH     Vertical-Horizontal backscatter

        Expected output units: dB (decibel), float32.
        SEN12MS-CR normalization clips to [-25, 0] dB.

        Args:
            bbox:           Area of interest in WGS84.
            target_date:    Scene date, "YYYY-MM-DD".
            output_path:    Destination GeoTIFF path.
            output_size_px: (width, height) in pixels.
        """
