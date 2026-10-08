"""
Copernicus Data Space Ecosystem (CDSE) + Sentinel Hub implementation
of the SatelliteProvider interface.

APIs used
---------
Catalog:    POST https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search
Processing: POST https://sh.dataspace.copernicus.eu/api/v1/process
Auth:       Managed by CDSEAuth (OAuth2 client_credentials)

Scientific constraints (do not change without re-validating the model)
-----------------------------------------------------------------------
S2  → 13 bands in SEN12MS-CR order (B01…B12, B8A), DN units [0, 10000], uint16
S1  → 2 bands (VV, VH), dB units, float32
       VV / VH dB range used by current normalization: clip(-25, 0)
       VH actual range may extend below -25 dB — the PoC will measure this.

References
----------
CDSE Catalog API:    https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Catalog.html
CDSE Processing API: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Process.html
S2 L1C bands:        https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L1C.html
S1 GRD bands:        https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S1GRD.html
"""

from __future__ import annotations

import datetime
import json
import os
import re
import sys
from typing import Any, Optional

import requests

# ---- locate project root so imports work regardless of CWD ----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from backend.auth.cdse_auth import CDSEAuth
from backend.services.satellite.base import (
    BoundingBox, S1Scene, S2Scene, SatelliteProvider,
)


# -----------------------------------------------------------------------
# CDSE Sentinel Hub API endpoints
# -----------------------------------------------------------------------

_CATALOG_URL = (
    "https://sh.dataspace.copernicus.eu"
    "/api/v1/catalog/1.0.0/search"
)

_PROCESS_URL = (
    "https://sh.dataspace.copernicus.eu"
    "/api/v1/process"
)

# Collection IDs as used by the CDSE Sentinel Hub Catalog API
_S2L1C_COLLECTION = "sentinel-2-l1c"
_S1GRD_COLLECTION  = "sentinel-1-grd"


# -----------------------------------------------------------------------
# Evalscripts
#
# These evalscripts define the exact contract between the Sentinel Hub
# Processing API and our ECRformer model.
#
# DO NOT reorder bands or change units without re-validating the model.
# -----------------------------------------------------------------------

#
# Sentinel-2 L1C — 13 bands, DN units, uint16 output
#
# Band order matches SEN12MS-CR / ECRformer channel convention:
#   [0] B01  Coastal aerosol  60m
#   [1] B02  Blue             10m
#   [2] B03  Green            10m
#   [3] B04  Red              10m
#   [4] B05  Veg. red edge    20m
#   [5] B06  Veg. red edge    20m
#   [6] B07  Veg. red edge    20m
#   [7] B08  NIR              10m
#   [8] B8A  Narrow NIR       20m  ← note: "B8A" in evalscript, not "B08A"
#   [9] B09  Water vapour     60m
#  [10] B10  Cirrus           60m
#  [11] B11  SWIR             20m
#  [12] B12  SWIR             20m
#
# Expected output: UINT16, values in [0, 10000] representing TOA reflectance × 10000.
# Compatible with preprocessing.py::normalize_s2() which clips [0,10000] / 10000.
#
# NOTE: The sentinelhub Processing API applies resampling when output_size_px
# is specified. All 13 bands (mixed native resolution) are resampled to the
# requested output grid. This matches the SEN12MS-CR dataset convention where
# all bands were upsampled to a common 10m grid.
#
_EVALSCRIPT_S2 = """\
//VERSION=3
//
// ECRformer-compatible Sentinel-2 L1C retrieval.
// BAND ORDER: B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
// Units: DN (UINT16, expected range [0, 10000])
// SEN12MS-CR normalization: clip(0, 10000) / 10000
//
function setup() {
  return {
    input: [{
      bands: [
        "B01", "B02", "B03", "B04", "B05", "B06",
        "B07", "B08", "B8A", "B09", "B10", "B11", "B12"
      ],
      units: "DN"
    }],
    output: {
      bands: 13,
      sampleType: "UINT16"
    }
  };
}

function evaluatePixel(sample) {
  return [
    sample.B01, sample.B02, sample.B03, sample.B04,
    sample.B05, sample.B06, sample.B07, sample.B08,
    sample.B8A, sample.B09, sample.B10, sample.B11, sample.B12
  ];
}
"""

#
# Sentinel-1 GRD — VV + VH, dB units, float32 output
#
# Band order:
#   [0] VV  Vertical-Vertical backscatter
#   [1] VH  Vertical-Horizontal backscatter
#
# This channel order is a hypothesis based on SEN12MS-CR conventions.
# It must be experimentally validated against the ECRformer checkpoint.
#
# Expected output: FLOAT32, LINEAR_POWER scale.
# SEN12MS-CR normalization: clip(-25, 0) → (data + 25) / 25
#
# RISK: VH backscatter can be lower than -25 dB for smooth/open water
# surfaces. Some SEN12MS-CR implementations clip VH to [-32.5, 0] dB
# instead. The PoC measures and reports actual per-band min/max values.
#
# Orthorectification: CDSE Sentinel Hub applies Range-Doppler terrain
# correction by default, using the Copernicus DEM. This matches the
# preprocessing applied in SEN12MS-CR, but has not been experimentally
# confirmed at the pixel level.
#
_EVALSCRIPT_S1 = """\
//VERSION=3
//
// ECRformer-compatible Sentinel-1 GRD retrieval.
// BAND ORDER: [VV, VH] — SEN12MS-CR convention (to be verified).
// Units: LINEAR_POWER (FLOAT32)
// SEN12MS-CR normalization: clip(-25, 0) -> (data + 25) / 25
//
function setup() {
  return {
    input: [{
      bands: ["VV", "VH"],
      units: "LINEAR_POWER"
    }],
    output: {
      bands: 2,
      sampleType: "FLOAT32"
    }
  };
}

function evaluatePixel(sample) {
  var vv = 10 * Math.log(Math.max(sample.VV, 1e-10)) / Math.LN10;
  var vh = 10 * Math.log(Math.max(sample.VH, 1e-10)) / Math.LN10;

  return [vv, vh];
}
"""

# The 13 band names in ECRformer/SEN12MS-CR order (for metadata)
S2_BAND_NAMES = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B10", "B11", "B12",
]

S1_BAND_NAMES = ["VV", "VH"]


# -----------------------------------------------------------------------
# Multipart response parser
# -----------------------------------------------------------------------

def _extract_tiff_from_response(response: requests.Response) -> bytes:
    """
    Extract the raw GeoTIFF bytes from a Sentinel Hub Processing API response.

    The Processing API always returns multipart/related, even for a single
    image output. This function handles both the multipart case and the rare
    direct image/tiff case.

    Raises:
        ValueError: If no TIFF part can be found in the response.
    """
    content_type = response.headers.get("Content-Type", "")

    # Rare case: response IS the TIFF directly
    if "image/tiff" in content_type and "multipart" not in content_type:
        return response.content

    if "multipart" not in content_type:
        raise ValueError(
            f"Unexpected Content-Type: {content_type!r}\n"
            f"HTTP status:  {response.status_code}\n"
            f"Body (first 800 bytes):\n{response.text[:800]}"
        )

    # Extract boundary from Content-Type header
    # e.g. multipart/related; boundary="abc123def"
    match = re.search(r'boundary="?([^";,\s]+)"?', content_type, re.IGNORECASE)
    if not match:
        raise ValueError(
            f"Could not extract multipart boundary from: {content_type!r}"
        )

    boundary = match.group(1).strip()
    separator = ("--" + boundary).encode("ascii")
    body = response.content

    parts = body.split(separator)

    for part in parts[1:]:
        # Skip the final boundary marker "--<boundary>--"
        stripped = part.lstrip(b"\r\n")
        if stripped.startswith(b"--"):
            continue

        # Each part has the form:
        #   \r\n<headers>\r\n\r\n<body>
        header_end = part.find(b"\r\n\r\n")
        if header_end == -1:
            continue

        headers_str = part[:header_end].decode("ascii", errors="ignore").lower()
        body_part = part[header_end + 4:]

        if "image/tiff" in headers_str or "application/octet-stream" in headers_str:
            # Strip trailing CRLF added by the multipart serialiser
            return body_part.rstrip(b"\r\n")

    raise ValueError(
        "No TIFF part found in the Sentinel Hub multipart response.\n"
        f"Content-Type: {content_type!r}\n"
        "Possible causes:\n"
        "  - No data is available for the requested AOI and date\n"
        "  - The evalscript has an error\n"
        "  - The Processing API returned an error inside the multipart body"
    )


# -----------------------------------------------------------------------
# CDSE Sentinel Hub provider
# -----------------------------------------------------------------------

class CDSESentinelHubProvider(SatelliteProvider):
    """
    Concrete implementation of SatelliteProvider for the Copernicus Data
    Space Ecosystem (CDSE) + Sentinel Hub APIs.

    All API credentials are managed by CDSEAuth (env vars only).

    See also:
        backend/auth/cdse_auth.py
        acquisition/README.md
    """

    def __init__(self) -> None:
        self._auth = CDSEAuth()

    # ----------------------------------------------------------------
    # Catalogue search (Sentinel Hub Catalog API — STAC-compliant)
    # ----------------------------------------------------------------

    def search_sentinel2(
        self,
        bbox: BoundingBox,
        start_date: str,
        end_date: str,
        max_results: int = 20,
    ) -> list[S2Scene]:
        """Search Sentinel-2 L1C scenes covering bbox in [start_date, end_date]."""
        raw_features = self._stac_search(
            collection=_S2L1C_COLLECTION,
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
        )
        scenes = []
        for feat in raw_features:
            try:
                scenes.append(self._parse_s2_feature(feat))
            except Exception as exc:
                print(f"  [WARN] Skipping malformed S2 feature: {exc}",
                      file=sys.stderr)
        scenes.sort(key=lambda s: s.dt)
        return scenes

    def search_sentinel1(
        self,
        bbox: BoundingBox,
        start_date: str,
        end_date: str,
        max_results: int = 20,
    ) -> list[S1Scene]:
        """Search Sentinel-1 GRD scenes covering bbox in [start_date, end_date]."""
        raw_features = self._stac_search(
            collection=_S1GRD_COLLECTION,
            bbox=bbox,
            start_date=start_date,
            end_date=end_date,
            max_results=max_results,
        )
        scenes = []
        for feat in raw_features:
            try:
                scenes.append(self._parse_s1_feature(feat))
            except Exception as exc:
                print(f"  [WARN] Skipping malformed S1 feature: {exc}",
                      file=sys.stderr)
        scenes.sort(key=lambda s: s.dt)
        return scenes

    def _stac_search(
        self,
        collection: str,
        bbox: BoundingBox,
        start_date: str,
        end_date: str,
        max_results: int,
    ) -> list[dict]:
        """
        POST a STAC search to the Sentinel Hub Catalog API.

        Returns the raw GeoJSON Feature list.
        """
        request_body = {
            "collections": [collection],
            "bbox": bbox.to_list(),
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            "limit": max_results,
        }

        headers = {
            **self._auth.auth_headers(),
            "Content-Type": "application/json",
            "Accept": "application/geo+json",
        }

        try:
            resp = requests.post(
                _CATALOG_URL,
                headers=headers,
                json=request_body,
                timeout=60,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Catalog search network error (collection={collection}): {exc}"
            ) from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"Catalog search failed: HTTP {resp.status_code}\n"
                f"Collection: {collection}\n"
                f"Request:    {json.dumps(request_body, indent=2)}\n"
                f"Response:   {resp.text[:800]}"
            )

        try:
            payload = resp.json()
        except ValueError as exc:
            raise RuntimeError(
                f"Invalid JSON in catalog response: {exc}"
            ) from exc

        return payload.get("features", [])

    @staticmethod
    def _parse_s2_feature(feature: dict) -> S2Scene:
        props = feature.get("properties", {})

        dt_str = (
            props.get("datetime")
            or props.get("end_datetime")
            or props.get("start_datetime", "")
        )
        if not dt_str:
            raise ValueError(f"No datetime in S2 feature: {feature.get('id')}")

        dt = datetime.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))

        cloud_raw = props.get("eo:cloud_cover")
        cloud_cover = float(cloud_raw) if cloud_raw is not None else None

        return S2Scene(
            scene_id=feature.get("id", "unknown"),
            dt=dt,
            cloud_cover=cloud_cover,
            processing_level="L1C",
            footprint=feature.get("geometry"),
            extra=props,
        )

    @staticmethod
    def _parse_s1_feature(feature: dict) -> S1Scene:
        props = feature.get("properties", {})

        dt_str = (
            props.get("datetime")
            or props.get("end_datetime")
            or props.get("start_datetime", "")
        )
        if not dt_str:
            raise ValueError(f"No datetime in S1 feature: {feature.get('id')}")

        dt = datetime.datetime.fromisoformat(dt_str.replace("Z", "+00:00"))

        # Polarizations may appear under different keys depending on SH version
        polarizations: list[str] = (
            props.get("sar:polarizations")
            or props.get("sar:polarisation")
            or props.get("polarizations", [])
        )
        if isinstance(polarizations, str):
            polarizations = [p.strip() for p in polarizations.split(",")]

        orbit = (
            props.get("sat:orbit_state")
            or props.get("orbit_direction")
        )
        if orbit:
            orbit = orbit.upper()

        mode = props.get("sar:instrument_mode") or props.get("s1:mode")

        return S1Scene(
            scene_id=feature.get("id", "unknown"),
            dt=dt,
            product_type=props.get("sar:product_type", "GRD"),
            polarizations=polarizations,
            orbit_direction=orbit,
            instrument_mode=mode,
            footprint=feature.get("geometry"),
            extra=props,
        )

    # ----------------------------------------------------------------
    # Download via Sentinel Hub Processing API
    # ----------------------------------------------------------------

    def download_sentinel2(
        self,
        bbox: BoundingBox,
        target_date: str,
        output_path: str,
        output_size_px: tuple[int, int] = (256, 256),
    ) -> None:
        """
        Download Sentinel-2 L1C for a single acquisition date.

        Uses a 24-hour time window to capture exactly one scene.
        If multiple acquisitions exist on the same day (e.g., at a tile
        overlap), the Processing API will return a mosaic; the metadata.json
        will record which date was requested.
        """
        print(f"  [S2] Requesting Processing API  date={target_date}  "
              f"size={output_size_px[0]}×{output_size_px[1]}px  ...")

        self._process_api_download(
            collection=_S2L1C_COLLECTION,
            evalscript=_EVALSCRIPT_S2,
            bbox=bbox,
            target_date=target_date,
            output_path=output_path,
            output_size_px=output_size_px,
            extra_data_filter={},
        )

    def download_sentinel1(
        self,
        bbox: BoundingBox,
        target_date: str,
        output_path: str,
        output_size_px: tuple[int, int] = (256, 256),
    ) -> None:
        """
        Download Sentinel-1 IW-GRD (VV + VH) for a single acquisition date.

        acquisitionMode="IW"  — Interferometric Wide Swath (land standard)
        polarization="DV"     — Dual Vertical: VV + VH
        """
        print(f"  [S1] Requesting Processing API  date={target_date}  "
              f"size={output_size_px[0]}×{output_size_px[1]}px  ...")

        self._process_api_download(
            collection=_S1GRD_COLLECTION,
            evalscript=_EVALSCRIPT_S1,
            bbox=bbox,
            target_date=target_date,
            output_path=output_path,
            output_size_px=output_size_px,
            extra_data_filter={
                "acquisitionMode": "IW",
                "polarization": "DV",   # Dual Vertical = VV + VH
            },
        )

    def _process_api_download(
        self,
        collection: str,
        evalscript: str,
        bbox: BoundingBox,
        target_date: str,
        output_path: str,
        output_size_px: tuple[int, int],
        extra_data_filter: dict,
    ) -> None:
        """
        POST a Processing API request and write the returned GeoTIFF to disk.

        The output coordinate reference system follows the bbox CRS (CRS84 /
        WGS84 here). Phase 2 harmonization should reproject to UTM to match
        SEN12MS-CR conventions and produce square pixels.
        """
        width_px, height_px = output_size_px

        request_body: dict[str, Any] = {
            "input": {
                "bounds": {
                    "bbox": bbox.to_list(),
                    "properties": {
                        # CRS84 = WGS84 geographic (lon, lat)
                        "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"
                    },
                },
                "data": [{
                    "type": collection,
                    "dataFilter": {
                        "timeRange": {
                            "from": f"{target_date}T00:00:00Z",
                            "to":   f"{target_date}T23:59:59Z",
                        },
                        **extra_data_filter,
                    },
                }],
            },
            "output": {
                "width":  width_px,
                "height": height_px,
                "responses": [{
                    "identifier": "default",
                    "format": {
                        "type": "image/tiff",
                        "parameters": {"compression": "DEFLATE"},
                    },
                }],
            },
            "evalscript": evalscript,
        }

        headers = {
            **self._auth.auth_headers(),
            "Content-Type": "application/json",
            "Accept": "*/*",
        }

        try:
            resp = requests.post(
                _PROCESS_URL,
                headers=headers,
                json=request_body,
                timeout=180,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Processing API network error (collection={collection}): {exc}"
            ) from exc

        if resp.status_code != 200:
            raise RuntimeError(
                f"Processing API failed: HTTP {resp.status_code}\n"
                f"Collection:   {collection}\n"
                f"Target date:  {target_date}\n"
                f"Response body:\n{resp.text[:1000]}\n\n"
                "Possible causes:\n"
                "  - No imagery available for this date and AOI\n"
                "  - Authentication / rate-limit issue\n"
                "  - Invalid evalscript or request parameters"
            )

        tiff_bytes = _extract_tiff_from_response(resp)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "wb") as fh:
            fh.write(tiff_bytes)

        size_kb = len(tiff_bytes) / 1024
        print(f"  [OK] Saved {size_kb:.1f} KB → {output_path}")
