from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import (
    binary_closing,
    disk,
    remove_small_objects,
)
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cloud_generation" / "output"

CLOUDY_S2 = OUT / "s2_cloudy_mock.tif"
VALIDITY = ROOT / "harmonization" / "output" / "validity_mask_10m.tif"
TRUE_MASK = OUT / "cloud_mask.tif"

PRED_MASK = OUT / "predicted_cloud_mask.tif"
SCORE_TIF = OUT / "cloud_score.tif"
PREVIEW = OUT / "predicted_cloud_mask_preview.png"
REPORT = OUT / "cloud_detector_report.json"


BANDS = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B8A", "B09", "B10", "B11", "B12"
]

IDX = {name: i for i, name in enumerate(BANDS)}


def robust_normalize(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalize a band using its scene-specific 99th percentile."""

    values = x[mask]

    if values.size == 0:
        return np.zeros_like(x, dtype=np.float32)

    upper = np.percentile(values, 99.0)

    if upper <= 1e-8:
        return np.zeros_like(x, dtype=np.float32)

    return np.clip(
        x / upper,
        0.0,
        1.0,
    ).astype(np.float32)


def main() -> None:

    print("=" * 72)
    print("PHASE 3D - AUTOMATIC CLOUD DETECTION")
    print("=" * 72)

    # ------------------------------------------------------------
    # Load cloudy S2
    # ------------------------------------------------------------

    with rasterio.open(CLOUDY_S2) as src:

        s2 = src.read().astype(np.float32)
        profile = src.profile.copy()

        print("Bands:", src.descriptions)
        print("Shape:", s2.shape)
        print("CRS:", src.crs)
        print("Resolution:", src.res)

    # ------------------------------------------------------------
    # Load validity mask
    # ------------------------------------------------------------

    with rasterio.open(VALIDITY) as src:
        valid = src.read(1).astype(bool)

    # ------------------------------------------------------------
    # Load TRUE synthetic cloud mask
    #
    # Used ONLY for evaluation.
    # It is NOT used to generate the prediction.
    # ------------------------------------------------------------

    with rasterio.open(TRUE_MASK) as src:
        true_cloud = src.read(1) >= 0.5

    true_cloud &= valid

    # ------------------------------------------------------------
    # Scene-adaptive band normalization
    # ------------------------------------------------------------

    b02 = robust_normalize(s2[IDX["B02"]], valid)
    b03 = robust_normalize(s2[IDX["B03"]], valid)
    b04 = robust_normalize(s2[IDX["B04"]], valid)

    b08 = robust_normalize(s2[IDX["B08"]], valid)
    b10 = robust_normalize(s2[IDX["B10"]], valid)
    b11 = robust_normalize(s2[IDX["B11"]], valid)
    b12 = robust_normalize(s2[IDX["B12"]], valid)

    # ------------------------------------------------------------
    # Feature 1: visible brightness
    # ------------------------------------------------------------

    visible = (
        b02 +
        b03 +
        b04
    ) / 3.0

    # ------------------------------------------------------------
    # Feature 2: spectral whiteness
    #
    # Clouds tend to be relatively neutral across visible bands.
    # ------------------------------------------------------------

    visible_stack = np.stack(
        [b02, b03, b04],
        axis=0,
    )

    visible_mean = np.mean(
        visible_stack,
        axis=0,
    )

    visible_std = np.std(
        visible_stack,
        axis=0,
    )

    whiteness = 1.0 - (
        visible_std /
        (visible_mean + 1e-6)
    )

    whiteness = np.clip(
        whiteness,
        0.0,
        1.0,
    )

    # ------------------------------------------------------------
    # Feature 3: low NDVI tendency
    # ------------------------------------------------------------

    ndvi = (
        s2[IDX["B08"]].astype(np.float32)
        - s2[IDX["B04"]].astype(np.float32)
    ) / (
        s2[IDX["B08"]].astype(np.float32)
        + s2[IDX["B04"]].astype(np.float32)
        + 1e-6
    )

    low_ndvi = np.clip(
        (0.5 - ndvi) / 1.5,
        0.0,
        1.0,
    )

    # ------------------------------------------------------------
    # Feature 4: cirrus-sensitive signal
    # ------------------------------------------------------------

    cirrus = b10

    # ------------------------------------------------------------
    # Feature 5: SWIR brightness
    # ------------------------------------------------------------

    swir = (
        b11 +
        b12
    ) / 2.0

    # ------------------------------------------------------------
    # Combined cloud score
    # ------------------------------------------------------------

    score = (
        0.40 * visible
        + 0.25 * whiteness
        + 0.15 * cirrus
        + 0.10 * swir
        + 0.10 * low_ndvi
    )

    score[~valid] = 0.0

    # Spatial smoothing
    score_smooth = gaussian(
        score,
        sigma=1.2,
        preserve_range=True,
    ).astype(np.float32)

    score_smooth[~valid] = 0.0

    # ------------------------------------------------------------
    # Adaptive threshold
    # ------------------------------------------------------------

    threshold = threshold_otsu(
        score_smooth[valid]
    )

    predicted = (
        score_smooth >= threshold
    )

    predicted &= valid

    # ------------------------------------------------------------
    # Morphological cleanup
    # ------------------------------------------------------------

    predicted = remove_small_objects(
        predicted,
        min_size=20,
    )

    predicted = binary_closing(
        predicted,
        footprint=disk(2),
    )

    predicted &= valid

    # ------------------------------------------------------------
    # Evaluation against TRUE synthetic mask
    # ------------------------------------------------------------

    tp = int(np.sum(predicted & true_cloud))
    fp = int(np.sum(predicted & ~true_cloud & valid))
    fn = int(np.sum(~predicted & true_cloud))
    tn = int(np.sum(~predicted & ~true_cloud & valid))

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0.0
    )

    predicted_coverage = (
        predicted.sum() / valid.sum()
    )

    true_coverage = (
        true_cloud.sum() / valid.sum()
    )

    # ------------------------------------------------------------
    # Save predicted binary mask
    # ------------------------------------------------------------

    mask_profile = profile.copy()

    mask_profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
    )

    with rasterio.open(
        PRED_MASK,
        "w",
        **mask_profile,
    ) as dst:

        dst.write(
            predicted.astype(np.uint8),
            1,
        )

        dst.set_band_description(
            1,
            "predicted_cloud_mask",
        )

        dst.update_tags(
            algorithm="spectral heuristic",
            threshold=float(threshold),
            valid_pixels=int(valid.sum()),
        )

    # ------------------------------------------------------------
    # Save cloud score
    # ------------------------------------------------------------

    score_profile = profile.copy()

    score_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=0.0,
        compress="deflate",
    )

    with rasterio.open(
        SCORE_TIF,
        "w",
        **score_profile,
    ) as dst:

        dst.write(
            score_smooth.astype(np.float32),
            1,
        )

        dst.set_band_description(
            1,
            "cloud_score",
        )

    # ------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------

    preview = np.zeros(
        predicted.shape,
        dtype=np.uint8,
    )

    preview[predicted] = 255

    Image.fromarray(
        preview,
        mode="L",
    ).save(PREVIEW)

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    report = {
        "phase": "3D",
        "algorithm": "scene-adaptive spectral heuristic",
        "weights": {
            "visible_brightness": 0.40,
            "whiteness": 0.25,
            "cirrus_signal": 0.15,
            "swir_brightness": 0.10,
            "low_ndvi": 0.10,
        },
        "threshold": float(threshold),
        "valid_pixels": int(valid.sum()),
        "true_cloud_pixels": int(true_cloud.sum()),
        "predicted_cloud_pixels": int(predicted.sum()),
        "true_coverage": float(true_coverage),
        "predicted_coverage": float(predicted_coverage),
        "confusion_matrix": {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        },
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "IoU": float(iou),
        "outputs": {
            "predicted_mask": str(PRED_MASK),
            "cloud_score": str(SCORE_TIF),
            "preview": str(PREVIEW),
        },
    }

    REPORT.write_text(
        json.dumps(
            report,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Print
    # ------------------------------------------------------------

    print("\nRESULTS")
    print("-" * 72)
    print(f"Threshold:          {threshold:.6f}")
    print(f"True coverage:      {true_coverage * 100:.2f}%")
    print(f"Predicted coverage: {predicted_coverage * 100:.2f}%")

    print("\nCONFUSION MATRIX")
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TN: {tn}")

    print("\nMETRICS")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"IoU:       {iou:.4f}")

    print("\nOUTPUTS")
    print(f"Predicted mask: {PRED_MASK}")
    print(f"Cloud score:    {SCORE_TIF}")
    print(f"Preview:        {PREVIEW}")
    print(f"Report:         {REPORT}")

    print("=" * 72)
    print("PHASE 3D COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()