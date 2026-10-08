"""
Acquisition cache utilities for the satellite data pipeline.

Provides a deterministic cache-key computation (extracted from
acquisition/poc_test.py), cache-hit detection, and metadata persistence
helpers.

The cache layout is:
    acquisition/data_cache/<16-char-hash>/
        s2.tif
        s1.tif
        s2_preview.png
        s1_preview.png
        metadata.json
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

from backend.services.satellite.base import BoundingBox

_PROVIDER = "cdse_sentinelhub"
_S2_COLLECTION = "sentinel-2-l1c"
_S1_COLLECTION = "sentinel-1-grd"

S2_BAND_NAMES = [
    "B01", "B02", "B03", "B04", "B05", "B06",
    "B07", "B08", "B8A", "B09", "B10", "B11", "B12",
]
S1_BAND_NAMES = ["VV", "VH"]


def compute_cache_key(
    provider: str,
    collection: str,
    bbox: BoundingBox,
    target_date: str,
    bands: list[str],
    output_size_px: tuple[int, int],
) -> str:
    params: dict[str, Any] = {
        "provider":    provider,
        "collection":  collection,
        "bbox":        bbox.to_list(),
        "target_date": target_date,
        "bands":       sorted(bands),
        "output_size": list(output_size_px),
    }
    raw = json.dumps(params, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def compute_s2_cache_key(
    bbox: BoundingBox,
    target_date: str,
    output_size_px: tuple[int, int] = (256, 256),
) -> str:
    return compute_cache_key(_PROVIDER, _S2_COLLECTION, bbox, target_date, S2_BAND_NAMES, output_size_px)


def compute_s1_cache_key(
    bbox: BoundingBox,
    target_date: str,
    output_size_px: tuple[int, int] = (256, 256),
) -> str:
    return compute_cache_key(_PROVIDER, _S1_COLLECTION, bbox, target_date, S1_BAND_NAMES, output_size_px)


def get_s2_cache_dir(cache_root, bbox: BoundingBox, target_date: str, output_size_px=(256, 256)) -> Path:
    key = compute_s2_cache_key(bbox, target_date, output_size_px)
    return Path(cache_root) / key


def get_s1_cache_dir(cache_root, bbox: BoundingBox, target_date: str, output_size_px=(256, 256)) -> Path:
    key = compute_s1_cache_key(bbox, target_date, output_size_px)
    return Path(cache_root) / key


def is_s2_cached(cache_root, bbox: BoundingBox, target_date: str, output_size_px=(256, 256)) -> bool:
    tif = get_s2_cache_dir(cache_root, bbox, target_date, output_size_px) / "s2.tif"
    return tif.is_file() and tif.stat().st_size > 0


def is_s1_cached(cache_root, bbox: BoundingBox, target_date: str, output_size_px=(256, 256)) -> bool:
    tif = get_s1_cache_dir(cache_root, bbox, target_date, output_size_px) / "s1.tif"
    return tif.is_file() and tif.stat().st_size > 0


def write_cache_metadata(cache_dir, collection: str, bbox: BoundingBox,
                         target_date: str, bands: list[str],
                         output_size_px: tuple[int, int],
                         extra: Optional[dict] = None) -> None:
    import datetime
    cache_dir = Path(cache_dir)
    key = compute_cache_key(_PROVIDER, collection, bbox, target_date, bands, output_size_px)
    meta: dict[str, Any] = {
        "provider": _PROVIDER,
        "collection": collection,
        "aoi": {"bbox": bbox.to_list()},
        "target_date": target_date,
        "bands": bands,
        "output_size": list(output_size_px),
        "cache_key": key,
        "created": datetime.datetime.utcnow().isoformat() + "Z",
    }
    if extra:
        meta.update(extra)
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)


def read_cache_metadata(cache_dir) -> Optional[dict]:
    meta_path = Path(cache_dir) / "metadata.json"
    if not meta_path.is_file():
        return None
    with open(meta_path, "r", encoding="utf-8") as fh:
        return json.load(fh)