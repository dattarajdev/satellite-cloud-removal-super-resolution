"""
Geocoding and AOI geometry service for satellite data acquisition.

Uses OpenStreetMap Nominatim with proper User-Agent identification,
offline preset fallbacks, in-memory caching, error resilience,
and bounding box geometry calculation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import requests

from backend.services.satellite.base import BoundingBox

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "SatelliteCloudRemovalApp/1.0 (CDSE-Satellite-Research)"

# Pre-cached known presets (guarantees offline availability for benchmark demonstrations)
_KNOWN_PRESETS: Dict[str, Tuple[float, float, str]] = {
    "mumbai, india": (19.0760, 72.8777, "Mumbai, Maharashtra, India"),
    "mumbai": (19.0760, 72.8777, "Mumbai, Maharashtra, India"),
    "airoli, navi mumbai": (19.1585, 72.9994, "Airoli, Navi Mumbai, Maharashtra, India"),
    "neral, maharashtra": (19.0266, 73.3181, "Neral, Karjat, Maharashtra, India"),
    "rome, italy": (41.8933, 12.4829, "Rome, Lazio, Italy"),
    "rome": (41.8933, 12.4829, "Rome, Lazio, Italy"),
    "po valley benchmark": (44.8150, 10.5150, "Po Valley Benchmark, Emilia-Romagna, Italy"),
    "berlin": (52.5125, 13.3950, "Berlin, Germany"),
    "gateway of india": (18.9220, 72.8346, "Gateway of India, Colaba, Mumbai, Maharashtra, India"),
}

# In-memory query cache to avoid redundant external network requests
_GEOCODE_CACHE: Dict[str, Optional["GeocodedLocation"]] = {}


@dataclass
class GeocodedLocation:
    """Represents a geocoded address or landmark."""
    query: str
    display_name: str
    latitude: float
    longitude: float


def geocode_location(query: str, timeout: int = 6) -> Optional[GeocodedLocation]:
    """
    Geocode an address, landmark, or city name into geographic coordinates.

    Returns None if no matching location is found or if the network request fails.
    """
    clean_query = query.strip()
    if not clean_query:
        return None

    cache_key = clean_query.lower()
    if cache_key in _GEOCODE_CACHE:
        return _GEOCODE_CACHE[cache_key]

    # Check offline fallback dictionary
    if cache_key in _KNOWN_PRESETS:
        lat, lon, dname = _KNOWN_PRESETS[cache_key]
        loc = GeocodedLocation(query=clean_query, display_name=dname, latitude=lat, longitude=lon)
        _GEOCODE_CACHE[cache_key] = loc
        return loc

    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json",
    }
    params = {
        "q": clean_query,
        "format": "json",
        "limit": 1,
        "addressdetails": 0,
    }

    try:
        resp = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            results = resp.json()
            if results and isinstance(results, list):
                first = results[0]
                loc = GeocodedLocation(
                    query=clean_query,
                    display_name=first.get("display_name", clean_query),
                    latitude=float(first["lat"]),
                    longitude=float(first["lon"]),
                )
                _GEOCODE_CACHE[cache_key] = loc
                return loc
            else:
                _GEOCODE_CACHE[cache_key] = None
                return None
        else:
            return None
    except Exception:
        return None


def bbox_from_center(lat: float, lon: float, size_km: float) -> BoundingBox:
    """
    Generate a geographic bounding box (WGS84) centered at (lat, lon) with the specified size in km.
    """
    half_km = float(size_km) / 2.0
    # 1 deg latitude ~ 110.54 km
    d_lat = half_km / 110.54
    # 1 deg longitude ~ 111.32 * cos(lat) km
    cos_lat = max(math.cos(math.radians(lat)), 0.01)
    d_lon = half_km / (111.32 * cos_lat)

    w = max(min(lon - d_lon, 180.0), -180.0)
    e = max(min(lon + d_lon, 180.0), -180.0)
    s = max(min(lat - d_lat, 90.0), -90.0)
    n = max(min(lat + d_lat, 90.0), -90.0)

    return BoundingBox(
        west=round(w, 6),
        south=round(s, 6),
        east=round(e, 6),
        north=round(n, 6),
    )


def validate_aoi(bbox: BoundingBox, max_dimension_km: float = 60.0) -> Tuple[bool, Optional[str]]:
    """
    Validate that an AOI satisfies geographic and processing constraints.

    Returns (is_valid, error_message).
    """
    if bbox.west >= bbox.east:
        return False, "Invalid longitude: West bound must be strictly less than East bound."
    if bbox.south >= bbox.north:
        return False, "Invalid latitude: South bound must be strictly less than North bound."

    if not (-180.0 <= bbox.west <= 180.0 and -180.0 <= bbox.east <= 180.0):
        return False, "Longitude coordinates must be within [-180.0, 180.0]."
    if not (-90.0 <= bbox.south <= 90.0 and -90.0 <= bbox.north <= 90.0):
        return False, "Latitude coordinates must be within [-90.0, 90.0]."

    width_km = bbox.approx_width_m() / 1000.0
    height_km = bbox.approx_height_m() / 1000.0

    if width_km > max_dimension_km or height_km > max_dimension_km:
        return False, (
            f"AOI dimensions ({width_km:.1f} km x {height_km:.1f} km) exceed the recommended "
            f"maximum of {max_dimension_km:.0f} km. Please select a smaller AOI to prevent download timeouts."
        )

    return True, None