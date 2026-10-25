"""
Performance Metrics and Diagnostics Module for ECRformer and OpenSR.

Implements:
1. Reference-based quality metrics for ECRformer (MAE, RMSE, PSNR, SSIM, SAM, LPIPS).
   - Strict mask-aware calculation excluding NoData and invalid pixels.
   - Comprehensive 13-band per-band breakdown and explicitly defined aggregates.
   - Separate SSIM evaluation on RGB representation (B04, B03, B02) and individual bands.
   - Explicitly documented data ranges and conventions.
2. Reference-free diagnostics for real-world acquisitions lacking ground truth.
   - Per-band min, max, mean, NaN/Inf counts, valid-pixel coverage.
   - Explicit "Reference-based image-quality metrics unavailable: no cloud-free reference supplied."
3. OpenSR diagnostics and 10 m self-consistency evaluation.
   - 4-band selection (B04, B03, B02, B08), 4x scale factor, 2.5 m resolution, CRS, bounds.
   - 10 m self-consistency aggregation against ECRformer baseline.
   - Strict label: "10 m consistency with ECRformer output — not ground-truth accuracy."
4. Serialized JSON diagnostics export per processed image.
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    from skimage.metrics import structural_similarity as ssim_fn
    SKIMAGE_AVAILABLE = True
except ImportError:
    ssim_fn = None
    SKIMAGE_AVAILABLE = False

try:
    import lpips
    import torch
    LPIPS_AVAILABLE = True
except ImportError:
    lpips = None
    LPIPS_AVAILABLE = False

logger = logging.getLogger(__name__)

S2_BAND_NAMES: List[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
    "B8A", "B09", "B10", "B11", "B12"
]
OPENSR_BAND_NAMES: List[str] = ["B04", "B03", "B02", "B08"]


def sanitize_for_json(obj: Any) -> Any:
    """Ensure floats like NaN and Inf are replaced with null for valid JSON serialization."""
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return round(obj, 6)
    elif isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
            return None
        return val
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


@dataclass
class ReferenceBasedMetrics:
    """Quantitative performance metrics computed against a cloud-free ground-truth reference."""
    available: bool = True
    status: str = "Reference-based image-quality metrics computed successfully."
    data_range: float = 1.0  # Documented reflectance range [0.0, 1.0]
    num_evaluated_pixels: int = 0
    evaluated_bands: List[str] = field(default_factory=lambda: list(S2_BAND_NAMES))
    
    # Aggregates across all 13 bands
    aggregate_mae: float = 0.0
    aggregate_rmse: float = 0.0
    aggregate_psnr_db: float = 0.0
    
    # Spectral Angle Mapper (degrees)
    sam_degrees: float = 0.0
    
    # Structural Similarity (RGB: B04, B03, B02)
    ssim_rgb: Optional[float] = None
    ssim_rgb_convention: str = "Computed on RGB representation (B04, B03, B02) with data_range=1.0"
    
    # Per-band SSIM across all 13 bands
    per_band_ssim: Optional[Dict[str, Optional[float]]] = None
    ssim_per_band_convention: str = "Computed per-band with data_range=1.0; invalid pixels replaced by target"
    
    # LPIPS
    lpips_score: Optional[float] = None
    lpips_note: Optional[str] = None
    
    # Per-band breakdowns for all 13 bands
    per_band_mae: Dict[str, float] = field(default_factory=dict)
    per_band_rmse: Dict[str, float] = field(default_factory=dict)
    per_band_psnr_db: Dict[str, float] = field(default_factory=dict)

    # Reference classification
    reference_type: str = "paired_ground_truth"  # "paired_ground_truth" or "temporal_reference"
    reference_source: str = ""
    reference_date: Optional[str] = None
    temporal_distance_days: Optional[float] = None
    metric_label: str = "Reference-based reconstruction metrics (Ground Truth)"
    disclaimer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_json(asdict(self))


@dataclass
class ReferenceFreeDiagnostics:
    """Scientific reference-free diagnostics computed when no ground-truth reference is supplied."""
    available: bool = True
    status: str = "Reference-based image-quality metrics unavailable: no cloud-free reference supplied."
    dimensions: List[int] = field(default_factory=list)  # [H, W]
    band_count: int = 13
    band_names: List[str] = field(default_factory=lambda: list(S2_BAND_NAMES))
    spatial_resolution_m: float = 10.0
    crs: str = ""
    total_pixels: int = 0
    valid_pixels: int = 0
    invalid_pixels: int = 0
    valid_coverage_pct: float = 0.0
    nan_count: int = 0
    inf_count: int = 0
    per_band_min: Dict[str, float] = field(default_factory=dict)
    per_band_max: Dict[str, float] = field(default_factory=dict)
    per_band_mean: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_json(asdict(self))


@dataclass
class OpenSRConsistencyDiagnostics:
    """10 m self-consistency diagnostics between OpenSR output and input ECRformer output."""
    computed: bool = True
    metric_label: str = "10 m consistency with ECRformer output — not ground-truth accuracy."
    disclaimer: str = (
        "This metric measures numerical consistency between the 4x super-resolved output "
        "(downsampled to 10 m via area-weighted block averaging) and the input ECRformer reconstruction. "
        "It is NOT ground-truth super-resolution accuracy."
    )
    downsampling_method: str = "4x4 area-weighted spatial block averaging to 10 m grid"
    radiometric_scaling: str = "OpenSR DN / 10000.0 -> normalized reflectance [0.0, 1.0]"
    evaluated_bands: List[str] = field(default_factory=lambda: list(OPENSR_BAND_NAMES))
    aggregate_mae: float = 0.0
    aggregate_rmse: float = 0.0
    per_band_mae: Dict[str, float] = field(default_factory=dict)
    per_band_rmse: Dict[str, float] = field(default_factory=dict)
    evaluated_pixels: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_json(asdict(self))


def compute_reference_based_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    band_names: Optional[List[str]] = None,
    data_range: float = 1.0,
    reference_type: str = "paired_ground_truth",
    reference_source: str = "",
    reference_date: Optional[str] = None,
    temporal_distance_days: Optional[float] = None,
) -> ReferenceBasedMetrics:
    """
    Compute quantitative reconstruction fidelity metrics against a cloud-free ground truth reference.

    Args:
        prediction: Predicted Sentinel-2 image array of shape (13, H, W) in normalized reflectance [0, 1].
        target: Ground truth Sentinel-2 image array of shape (13, H, W).
                If raw DN in [0, 10000] (max > 1.5), automatically normalized to [0, 1].
        valid_mask: Optional boolean 2D mask (H, W) where True = valid pixel.
        band_names: List of band names (default S2_BAND_NAMES).
        data_range: Radiometric scale maximum for PSNR and SSIM (default 1.0 for reflectance).

    Returns:
        ReferenceBasedMetrics container with per-band and aggregate MAE, RMSE, PSNR, SSIM, SAM, LPIPS.
    """
    b_names = band_names or S2_BAND_NAMES
    pred = np.asarray(prediction, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)

    if pred.ndim != 3 or tgt.ndim != 3:
        raise ValueError(f"Expected 3D arrays (C, H, W). Got pred={pred.shape}, target={tgt.shape}")
    if pred.shape != tgt.shape:
        raise ValueError(f"Shape mismatch: prediction {pred.shape} vs target {tgt.shape}")

    c, h, w = pred.shape
    if c != len(b_names):
        raise ValueError(f"Band count mismatch: array has {c} bands, but {len(b_names)} band names provided.")

    # Radiometric scale harmonization: if target is in [0, 10000] DN scale, normalize to [0, 1]
    tgt_finite = tgt[np.isfinite(tgt)]
    if tgt_finite.size > 0 and float(np.nanmax(tgt_finite)) > 1.5:
        tgt = np.clip(tgt / 10000.0, 0.0, 1.0)
    else:
        tgt = np.clip(tgt, 0.0, 1.0)

    pred = np.clip(pred, 0.0, 1.0)

    # Valid mask handling
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool).squeeze()
        if mask.shape != (h, w):
            raise ValueError(f"Mask shape {mask.shape} does not match spatial dimensions ({h}, {w})")
    else:
        mask = np.ones((h, w), dtype=bool)

    # Ensure mask only includes finite pixels in both pred and tgt
    finite_mask = mask & np.all(np.isfinite(pred), axis=0) & np.all(np.isfinite(tgt), axis=0)
    num_valid = int(finite_mask.sum())

    if num_valid == 0:
        raise ValueError("No valid finite pixels available for reference-based metric calculation.")

    # 1. Per-band and aggregate MAE, RMSE, PSNR
    per_band_mae = {}
    per_band_rmse = {}
    per_band_psnr = {}

    for i, name in enumerate(b_names):
        p_b = pred[i][finite_mask]
        t_b = tgt[i][finite_mask]
        diff = p_b - t_b

        mae_val = float(np.mean(np.abs(diff)))
        rmse_val = float(np.sqrt(np.mean(diff ** 2)))
        if rmse_val > 0.0:
            psnr_val = float(20.0 * np.log10(data_range / max(rmse_val, 1e-12)))
        else:
            psnr_val = float("inf")

        per_band_mae[name] = mae_val
        per_band_rmse[name] = rmse_val
        per_band_psnr[name] = psnr_val

    agg_mae = float(np.mean(list(per_band_mae.values())))
    agg_rmse = float(np.mean(list(per_band_rmse.values())))
    finite_psnrs = [v for v in per_band_psnr.values() if not np.isinf(v)]
    agg_psnr = float(np.mean(finite_psnrs)) if finite_psnrs else float("inf")

    # 2. Spectral Angle Mapper (SAM) across all 13 bands
    pred_vec = pred[:, finite_mask].T  # (N, C)
    tgt_vec = tgt[:, finite_mask].T   # (N, C)

    pred_norm = np.linalg.norm(pred_vec, axis=1)
    tgt_norm = np.linalg.norm(tgt_vec, axis=1)
    denom = pred_norm * tgt_norm
    valid_sam = denom > 1e-8

    if np.any(valid_sam):
        dot_product = np.sum(pred_vec[valid_sam] * tgt_vec[valid_sam], axis=1)
        cos_theta = np.clip(dot_product / denom[valid_sam], -1.0, 1.0)
        sam_degrees = float(np.mean(np.arccos(cos_theta)) * 180.0 / np.pi)
    else:
        sam_degrees = float("nan")

    # 3. Structural Similarity Index (SSIM)
    ssim_rgb = None
    per_band_ssim = {}

    if SKIMAGE_AVAILABLE:
        # Check minimum spatial window size: default win_size=7 requires spatial dimension >= 7
        if min(h, w) >= 7:
            # A. SSIM on RGB representation (B04: index 3, B03: index 2, B02: index 1)
            # Fill invalid pixels with target values so invalid borders do not penalize SSIM
            rgb_indices = (3, 2, 1)  # B04, B03, B02
            pred_rgb = pred[list(rgb_indices)].copy()
            tgt_rgb = tgt[list(rgb_indices)].copy()
            pred_rgb[:, ~finite_mask] = tgt_rgb[:, ~finite_mask]

            rgb_ssim_channels = []
            for ch in range(3):
                score = ssim_fn(
                    tgt_rgb[ch],
                    pred_rgb[ch],
                    data_range=data_range,
                )
                rgb_ssim_channels.append(float(score))
            ssim_rgb = float(np.mean(rgb_ssim_channels))

            # B. Per-band SSIM across all 13 bands
            for i, name in enumerate(b_names):
                p_band = pred[i].copy()
                t_band = tgt[i].copy()
                p_band[~finite_mask] = t_band[~finite_mask]
                b_score = ssim_fn(
                    t_band,
                    p_band,
                    data_range=data_range,
                )
                per_band_ssim[name] = float(b_score)
        else:
            logger.info(f"Spatial extent ({w}x{h}) too small for SSIM calculation (min extent 7 required).")
            ssim_rgb = None
            per_band_ssim = {name: None for name in b_names}
    else:
        per_band_ssim = {name: None for name in b_names}

    # 4. Learned Perceptual Image Patch Similarity (LPIPS)
    lpips_score = None
    lpips_note = None
    if LPIPS_AVAILABLE and min(h, w) >= 32:
        try:
            # LPIPS expects input in [-1, 1], shape (1, 3, H, W)
            loss_fn = lpips.LPIPS(net="alex", verbose=False)
            rgb_indices = [3, 2, 1]
            p_rgb = pred[rgb_indices].copy()
            t_rgb = tgt[rgb_indices].copy()
            p_rgb[:, ~finite_mask] = t_rgb[:, ~finite_mask]

            t_pred = torch.from_numpy(p_rgb * 2.0 - 1.0).unsqueeze(0).float()
            t_tgt = torch.from_numpy(t_rgb * 2.0 - 1.0).unsqueeze(0).float()

            with torch.no_grad():
                val = loss_fn(t_pred, t_tgt)
                lpips_score = float(val.item())
                lpips_note = "Computed on normalized RGB (B04, B03, B02) scaled to [-1, 1] via AlexNet LPIPS."
        except Exception as e:
            lpips_note = f"LPIPS computation error: {e}"
    elif not LPIPS_AVAILABLE:
        lpips_note = "LPIPS metric unavailable: 'lpips' package not installed in environment."
    else:
        lpips_note = "LPIPS metric unavailable: image spatial dimensions too small (requires >= 32x32)."

    if reference_type == "temporal_reference":
        metric_label = "Temporal-reference comparison metrics"
        disclaimer = (
            "This is a clear temporal reference from a different acquisition date. "
            "Reference-based differences may reflect real surface or atmospheric changes and "
            "should not be interpreted as strict reconstruction accuracy."
        )
        if temporal_distance_days is not None:
            dist_str = f"{temporal_distance_days:+.1f} days"
            status = f"Temporal-reference comparison metrics computed against scene from {reference_date or 'different date'} ({dist_str})."
        else:
            status = f"Temporal-reference comparison metrics computed against scene from {reference_date or 'different date'}."
    else:
        metric_label = "Reference-based reconstruction metrics (Ground Truth)"
        disclaimer = None
        status = "Reference-based image-quality metrics computed successfully."

    return ReferenceBasedMetrics(
        available=True,
        status=status,
        data_range=data_range,
        num_evaluated_pixels=num_valid,
        evaluated_bands=b_names,
        aggregate_mae=agg_mae,
        aggregate_rmse=agg_rmse,
        aggregate_psnr_db=agg_psnr,
        sam_degrees=sam_degrees,
        ssim_rgb=ssim_rgb,
        per_band_ssim=per_band_ssim,
        lpips_score=lpips_score,
        lpips_note=lpips_note,
        per_band_mae=per_band_mae,
        per_band_rmse=per_band_rmse,
        per_band_psnr_db=per_band_psnr,
        reference_type=reference_type,
        reference_source=reference_source,
        reference_date=reference_date,
        temporal_distance_days=temporal_distance_days,
        metric_label=metric_label,
        disclaimer=disclaimer,
    )


def compute_reference_free_diagnostics(
    prediction: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    band_names: Optional[List[str]] = None,
    crs: str = "",
    spatial_resolution_m: float = 10.0,
) -> ReferenceFreeDiagnostics:
    """
    Compute reference-free diagnostics for an ECRformer reconstruction when no ground truth is available.
    """
    b_names = band_names or S2_BAND_NAMES
    pred = np.asarray(prediction, dtype=np.float32)
    c, h, w = pred.shape

    total_pixels = int(h * w)

    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool).squeeze()
    else:
        mask = np.ones((h, w), dtype=bool)

    valid_pixels = int(mask.sum())
    invalid_pixels = total_pixels - valid_pixels
    coverage_pct = round(100.0 * valid_pixels / max(total_pixels, 1), 2)

    nan_count = int(np.isnan(pred).sum())
    inf_count = int(np.isinf(pred).sum())

    per_band_min = {}
    per_band_max = {}
    per_band_mean = {}

    for i, name in enumerate(b_names):
        if valid_pixels > 0:
            vals = pred[i][mask]
            finite_vals = vals[np.isfinite(vals)]
            if finite_vals.size > 0:
                per_band_min[name] = float(np.min(finite_vals))
                per_band_max[name] = float(np.max(finite_vals))
                per_band_mean[name] = float(np.mean(finite_vals))
            else:
                per_band_min[name] = per_band_max[name] = per_band_mean[name] = 0.0
        else:
            per_band_min[name] = per_band_max[name] = per_band_mean[name] = 0.0

    return ReferenceFreeDiagnostics(
        available=True,
        status="Reference-based image-quality metrics unavailable: no cloud-free reference supplied.",
        dimensions=[int(h), int(w)],
        band_count=c,
        band_names=b_names,
        spatial_resolution_m=spatial_resolution_m,
        crs=crs,
        total_pixels=total_pixels,
        valid_pixels=valid_pixels,
        invalid_pixels=invalid_pixels,
        valid_coverage_pct=coverage_pct,
        nan_count=nan_count,
        inf_count=inf_count,
        per_band_min=per_band_min,
        per_band_max=per_band_max,
        per_band_mean=per_band_mean,
    )


def compute_opensr_10m_consistency(
    opensr_4band_2_5m: np.ndarray,
    ecrformer_13band_10m: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
) -> OpenSRConsistencyDiagnostics:
    """
    Compute 10 m consistency between 4-band OpenSR output and 13-band ECRformer baseline.

    Procedure:
    1. Downsample OpenSR 2.5 m output to 10 m grid (exact 4x spatial block averaging).
    2. Convert OpenSR DN in [0, 10000] to reflectance in [0.0, 1.0].
    3. Extract matching bands B04, B03, B02, B08 from ECRformer.
    4. Compute per-band and aggregate MAE and RMSE over valid pixels.
    """
    sr = np.asarray(opensr_4band_2_5m, dtype=np.float32)
    ecr = np.asarray(ecrformer_13band_10m, dtype=np.float32)

    if sr.ndim != 3 or sr.shape[0] != 4:
        raise ValueError(f"Expected 4-band OpenSR array (4, H_sr, W_sr). Got {sr.shape}")
    if ecr.ndim != 3 or ecr.shape[0] < 8:
        raise ValueError(f"Expected >=8 bands for ECRformer array. Got {ecr.shape}")

    _, h_sr, w_sr = sr.shape
    _, h_ecr, w_ecr = ecr.shape

    # OpenSR scale is 4x
    if h_sr != h_ecr * 4 or w_sr != w_ecr * 4:
        raise ValueError(
            f"Dimension mismatch for 4x scaling: OpenSR {w_sr}x{h_sr} vs ECRformer {w_ecr}x{h_ecr}"
        )

    # 1. Exact 4x4 area-weighted spatial block averaging
    # Reshape (4, H_ecr, 4, W_ecr, 4) and average over axes 2 and 4
    sr_reshaped = sr.reshape(4, h_ecr, 4, w_ecr, 4)
    sr_downsampled = sr_reshaped.mean(axis=(2, 4))  # (4, H_ecr, W_ecr)

    # 2. Radiometric normalization: if OpenSR is in uint16 DN [0, 10000], divide by 10000.0
    if float(np.nanmax(sr_downsampled)) > 1.5:
        sr_10m_norm = np.clip(sr_downsampled / 10000.0, 0.0, 1.0)
    else:
        sr_10m_norm = np.clip(sr_downsampled, 0.0, 1.0)

    # 3. Extract matching bands from ECRformer:
    # ECRformer 13-band order: B01=0, B02=1, B03=2, B04=3, B05=4, B06=5, B07=6, B08=7...
    # OpenSR band order: B04=0, B03=1, B02=2, B08=3
    ecr_indices = [3, 2, 1, 7]  # B04, B03, B02, B08
    ecr_matched = np.clip(ecr[ecr_indices], 0.0, 1.0)

    # 4. Valid mask
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool).squeeze()
    else:
        mask = np.ones((h_ecr, w_ecr), dtype=bool)

    finite_mask = mask & np.all(np.isfinite(sr_10m_norm), axis=0) & np.all(np.isfinite(ecr_matched), axis=0)
    num_valid = int(finite_mask.sum())

    if num_valid == 0:
        raise ValueError("No valid finite pixels for OpenSR 10 m consistency evaluation.")

    per_band_mae = {}
    per_band_rmse = {}

    for i, name in enumerate(OPENSR_BAND_NAMES):
        p_b = sr_10m_norm[i][finite_mask]
        e_b = ecr_matched[i][finite_mask]
        diff = p_b - e_b
        per_band_mae[name] = float(np.mean(np.abs(diff)))
        per_band_rmse[name] = float(np.sqrt(np.mean(diff ** 2)))

    agg_mae = float(np.mean(list(per_band_mae.values())))
    agg_rmse = float(np.mean(list(per_band_rmse.values())))

    return OpenSRConsistencyDiagnostics(
        computed=True,
        metric_label="10 m consistency with ECRformer output — not ground-truth accuracy.",
        disclaimer=(
            "This metric measures numerical consistency between the 4x super-resolved output "
            "(downsampled to 10 m via area-weighted block averaging) and the input ECRformer reconstruction. "
            "It is NOT ground-truth super-resolution accuracy."
        ),
        downsampling_method="4x4 area-weighted spatial block averaging to 10 m grid",
        radiometric_scaling="OpenSR DN / 10000.0 -> normalized reflectance [0.0, 1.0]",
        evaluated_bands=list(OPENSR_BAND_NAMES),
        aggregate_mae=agg_mae,
        aggregate_rmse=agg_rmse,
        per_band_mae=per_band_mae,
        per_band_rmse=per_band_rmse,
        evaluated_pixels=num_valid,
    )


def build_image_diagnostics_export(
    image_id: str,
    stage_name: str,
    input_path: Optional[str],
    output_path: Optional[str],
    dimensions: List[int],
    band_names: List[str],
    crs: str,
    pixel_size_m: float,
    execution_time_seconds: float,
    execution_device: str,
    reference_metrics: Optional[ReferenceBasedMetrics] = None,
    reference_free_diagnostics: Optional[ReferenceFreeDiagnostics] = None,
    opensr_consistency: Optional[OpenSRConsistencyDiagnostics] = None,
    sampling_steps: Optional[int] = None,
    gpu_memory: Optional[str] = None,
    notes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Construct a clean, schema-compliant JSON export dictionary for a single processed image.
    Contains no secrets or raw credentials, uses strictly null for unavailable metrics.
    """
    has_ref = reference_metrics is not None and reference_metrics.available
    data: Dict[str, Any] = {
        "image_identifier": image_id,
        "stage": stage_name,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "input_file": Path(input_path).name if input_path else None,
        "output_file": Path(output_path).name if output_path else None,
        "dimensions_px": list(dimensions),
        "band_count": len(band_names),
        "band_names": list(band_names),
        "spatial_resolution_m": float(pixel_size_m),
        "coordinate_reference_system": str(crs),
        "execution_device": str(execution_device),
        "execution_time_seconds": round(execution_time_seconds, 3),
        "gpu_memory": gpu_memory,
        "sampling_steps": sampling_steps,
        "reference_available": has_ref,
        "reference_type": reference_metrics.reference_type if has_ref else None,
        "reference_status": (
            reference_metrics.status
            if has_ref
            else "Reference-based image-quality metrics unavailable: no cloud-free reference supplied."
        ),
        "reference_based_metrics": reference_metrics.to_dict() if has_ref else None,
        "reference_free_diagnostics": reference_free_diagnostics.to_dict() if reference_free_diagnostics else None,
        "opensr_10m_consistency": opensr_consistency.to_dict() if opensr_consistency else None,
        "notes_and_warnings": notes or [],
    }
    return sanitize_for_json(data)
