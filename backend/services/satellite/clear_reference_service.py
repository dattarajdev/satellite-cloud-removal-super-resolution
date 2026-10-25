"""
Clear Reference Search and Acquisition Service using Copernicus Data Space Ecosystem (CDSE).

Provides:
1. Search CDSE Sentinel-2 catalogue for clear candidate scenes within a temporal window (e.g. +/- 30 days).
2. Filter by scene-level cloud cover and estimate AOI-specific cloud fraction using SCL/CLD rules.
3. Strict distinction between paired ground truth and temporal references.
4. Download, cache, geospatial alignment, and RGB preview generation of selected candidate scenes.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio

from .base import BoundingBox, S2Scene
from .cdse_sentinelhub import CDSESentinelHubProvider
from ..processing.reference_adapter import ValidatedReference, load_and_validate_reference

logger = logging.getLogger(__name__)

DEFAULT_REFERENCE_CACHE_DIR = (
    Path(__file__).resolve().parents[3] / "acquisition" / "data_cache" / "references"
)


# Sentinel-2 Scene Classification Layer (SCL) definitions:
#  0: NO_DATA (invalid)
#  1: SATURATED_OR_DEFECTIVE (invalid)
#  2: DARK_AREA_PIXELS
#  3: CLOUD_SHADOWS (shadowed)
#  4: VEGETATION
#  5: NOT_VEGETATED (bare soils)
#  6: WATER
#  7: UNCLASSIFIED (uncertain)
#  8: CLOUD_MEDIUM_PROBABILITY (cloudy)
#  9: CLOUD_HIGH_PROBABILITY (cloudy)
# 10: THIN_CIRRUS (cloudy)
# 11: SNOW_OR_ICE


def compute_aoi_cloud_fraction_from_scl(
    scl: np.ndarray,
    include_shadows: bool = True,
    include_cirrus: bool = True,
) -> Optional[float]:
    """
    Compute mask-aware cloud fraction over valid pixels within an AOI using Sentinel-2 SCL.

    Classes counted as cloudy:
      - 8: Cloud medium probability
      - 9: Cloud high probability
      - 10: Thin cirrus (if include_cirrus=True)
      - 3: Cloud shadows (if include_shadows=True)

    Classes excluded as invalid / no-data:
      - 0: No data
      - 1: Saturated or defective

    Returns:
      Float percentage [0.0, 100.0] or None if no valid pixels exist in the AOI.
    """
    scl_flat = np.asarray(scl)
    valid_mask = (scl_flat != 0) & (scl_flat != 1) & np.isfinite(scl_flat)
    valid_count = int(np.sum(valid_mask))
    if valid_count == 0:
        return None

    cloud_classes = [8, 9]
    if include_cirrus:
        cloud_classes.append(10)
    if include_shadows:
        cloud_classes.append(3)

    cloud_mask = np.isin(scl_flat, cloud_classes) & valid_mask
    cloud_count = int(np.sum(cloud_mask))
    return float((cloud_count / valid_count) * 100.0)


@dataclass
class ClearReferenceCandidate:
    """A candidate cloud-free scene found in the CDSE catalogue."""
    scene_id: str
    acquisition_dt: datetime.datetime
    date_str: str
    temporal_distance_days: float
    scene_cloud_cover_pct: Optional[float]
    aoi_cloud_fraction_pct: Optional[float]
    passes_threshold: bool
    status: str
    s2_scene: Optional[S2Scene] = None

    def display_label(self) -> str:
        sign = "+" if self.temporal_distance_days >= 0 else ""
        scene_cloud = f"{self.scene_cloud_cover_pct:.1f}%" if self.scene_cloud_cover_pct is not None else "N/A"
        aoi_cloud = f"{self.aoi_cloud_fraction_pct:.1f}%" if self.aoi_cloud_fraction_pct is not None else "N/A"
        return (
            f"{self.date_str} ({sign}{self.temporal_distance_days:.1f}d) | "
            f"Scene Cloud: {scene_cloud} | AOI Cloud: {aoi_cloud} | {self.status}"
        )


class ClearReferenceService:
    """Service to search, inspect, and acquire clear temporal reference images from CDSE."""

    def __init__(self, provider: Optional[CDSESentinelHubProvider] = None, cache_dir: Optional[Path] = None):
        self.provider = provider or CDSESentinelHubProvider()
        self.cache_dir = Path(cache_dir or DEFAULT_REFERENCE_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def search_candidates(
        self,
        bbox: BoundingBox,
        target_date: Union[str, datetime.date],
        window_days: int = 30,
        max_scene_cloud_cover: float = 15.0,
        max_aoi_cloud_fraction: float = 10.0,
        max_candidates: int = 10,
        collection: str = "sentinel-2-l2a",
    ) -> List[ClearReferenceCandidate]:
        """
        Search for clear Sentinel-2 scenes in a window around target_date.

        Args:
            bbox: AOI bounding box.
            target_date: Target cloudy acquisition date (YYYY-MM-DD or date object).
            window_days: Search window in days before and after target_date.
            max_scene_cloud_cover: Scene-level cloud cover cutoff percentage.
            max_aoi_cloud_fraction: Maximum acceptable estimated cloud fraction over the AOI.
            max_candidates: Maximum candidate scenes to return.
            collection: CDSE collection ("sentinel-2-l2a" or "sentinel-2-l1c").

        Returns:
            List of ClearReferenceCandidate objects sorted by absolute temporal distance.
        """
        if isinstance(target_date, str):
            target_dt = datetime.datetime.strptime(target_date.strip()[:10], "%Y-%m-%d").date()
        else:
            target_dt = target_date

        start_date = (target_dt - datetime.timedelta(days=window_days)).strftime("%Y-%m-%d")
        end_date = (target_dt + datetime.timedelta(days=window_days)).strftime("%Y-%m-%d")

        try:
            raw_scenes = self.provider.search_sentinel2(
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
                max_results=max(max_candidates * 4, 30),
                collection=collection,
            )
        except Exception as e:
            logger.warning(f"CDSE catalogue search failed: {e}")
            return []

        candidates: List[ClearReferenceCandidate] = []

        for scene in raw_scenes:
            scene_date = scene.dt.date()
            temp_delta = (scene_date - target_dt).total_seconds() / 86400.0

            # Exclude acquisitions from the exact same day if looking for an alternative clear scene
            if abs(temp_delta) < 0.2:
                continue

            # Scene-level cloud filtering
            cloud_pct = scene.cloud_cover
            if cloud_pct is not None and cloud_pct > max_scene_cloud_cover:
                continue

            # Estimate AOI-specific cloud fraction
            aoi_cloud_pct = self._estimate_aoi_cloud_fraction(scene, bbox)

            passes = True
            status_parts = []
            if cloud_pct is not None:
                status_parts.append(f"Scene cloud {cloud_pct:.1f}%")
            if aoi_cloud_pct is not None:
                if aoi_cloud_pct <= max_aoi_cloud_fraction:
                    status_parts.append(f"AOI clear ({aoi_cloud_pct:.1f}%)")
                else:
                    passes = False
                    status_parts.append(f"AOI exceeds threshold ({aoi_cloud_pct:.1f}% > {max_aoi_cloud_fraction:.1f}%)")
            else:
                status_parts.append("AOI cloud: N/A (needs download)")

            status_str = "Clear Candidate" if passes else "Cloudy AOI"

            candidates.append(
                ClearReferenceCandidate(
                    scene_id=scene.scene_id,
                    acquisition_dt=scene.dt,
                    date_str=scene_date.strftime("%Y-%m-%d"),
                    temporal_distance_days=round(temp_delta, 1),
                    scene_cloud_cover_pct=cloud_pct,
                    aoi_cloud_fraction_pct=aoi_cloud_pct,
                    passes_threshold=passes,
                    status=status_str,
                    s2_scene=scene,
                )
            )

        # Sort by: passes_threshold (True first), then absolute temporal distance
        candidates.sort(key=lambda c: (not c.passes_threshold, abs(c.temporal_distance_days)))
        return candidates[:max_candidates]

    def _estimate_aoi_cloud_fraction(self, scene: S2Scene, bbox: BoundingBox) -> Optional[float]:
        """
        Estimate AOI-specific cloud fraction from metadata or provider properties.
        Returns percentage [0.0, 100.0] or None if unavailable without downloading.
        """
        props = getattr(scene, "extra", {}) or {}

        # 1. Granular AOI cloud cover if directly provided in STAC
        if "aoi:cloud_cover" in props:
            try:
                return float(props["aoi:cloud_cover"])
            except (ValueError, TypeError):
                pass

        # 2. Check SCL statistics in extra if present
        if "scl_raster" in props:
            try:
                return compute_aoi_cloud_fraction_from_scl(props["scl_raster"])
            except Exception:
                pass

        # 3. If scene-level cloud is very low (< 1%), AOI is almost certainly clear
        if scene.cloud_cover is not None and scene.cloud_cover < 1.0:
            return float(scene.cloud_cover)

        # In standard catalogue queries without raw SCL raster, return None to avoid inventing 0.0%
        return None

    def acquire_and_align_reference(
        self,
        candidate: ClearReferenceCandidate,
        bbox: BoundingBox,
        target_profile: Optional[Dict[str, Any]] = None,
        scale_mode: str = "auto",
        output_size_px: Tuple[int, int] = (256, 256),
    ) -> ValidatedReference:
        """
        Download (or retrieve from cache) and geospatially align the reference image.
        """
        bbox_str = f"{bbox.west:.4f}_{bbox.south:.4f}_{bbox.east:.4f}_{bbox.north:.4f}"
        cache_hash = hashlib.md5(f"{candidate.scene_id}_{bbox_str}".encode()).hexdigest()[:12]
        cached_tif = self.cache_dir / f"ref_{candidate.date_str}_{cache_hash}.tif"

        if not cached_tif.is_file():
            logger.info(f"Downloading clear reference for date {candidate.date_str} to {cached_tif}")
            self.provider.download_sentinel2(
                bbox=bbox,
                target_date=candidate.date_str,
                output_path=str(cached_tif),
                output_size_px=output_size_px,
            )

        # Validate and align to target ECRformer prediction grid
        val_ref = load_and_validate_reference(
            reference_input=cached_tif,
            target_profile=target_profile,
            scale_mode=scale_mode,
            reference_type="temporal_reference",
            acquisition_date=candidate.date_str,
            temporal_distance_days=candidate.temporal_distance_days,
        )
        return val_ref

    @staticmethod
    def to_rgb_preview(ref: ValidatedReference) -> np.ndarray:
        """
        Create a preview RGB image (uint8, HxWx3) from bands B04 (Red), B03 (Green), B02 (Blue).
        """
        # Bands: B04 is index 3, B03 is index 2, B02 is index 1
        arr = ref.array
        r = arr[3]
        g = arr[2]
        b = arr[1]
        rgb = np.stack([r, g, b], axis=-1)

        # Normalize with 2% - 98% percentile stretch for visual clarity
        valid = ref.valid_mask
        if np.any(valid):
            p2 = float(np.percentile(rgb[valid], 2))
            p98 = float(np.percentile(rgb[valid], 98))
            diff = max(p98 - p2, 1e-5)
            rgb_stretched = np.clip((rgb - p2) / diff, 0.0, 1.0)
        else:
            rgb_stretched = np.clip(rgb, 0.0, 1.0)

        return (rgb_stretched * 255.0).astype(np.uint8)
