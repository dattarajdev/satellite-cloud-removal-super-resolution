from __future__ import annotations

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import rasterio
from PIL import Image
from skimage.metrics import structural_similarity

from inference import get_device, load_model, run_sample


PROJECT_ROOT = Path(__file__).resolve().parent.parent

S1_PATH = (
    PROJECT_ROOT
    / "harmonization"
    / "output"
    / "s1_harmonized_10m.tif"
)

CLOUDY_S2_PATH = (
    PROJECT_ROOT
    / "cloud_generation"
    / "output"
    / "s2_cloudy_mock.tif"
)

CLEAR_S2_PATH = (
    PROJECT_ROOT
    / "harmonization"
    / "output"
    / "s2_harmonized_10m.tif"
)

VALIDITY_PATH = (
    PROJECT_ROOT
    / "harmonization"
    / "output"
    / "validity_mask_10m.tif"
)

CHECKPOINT_PATH = PROJECT_ROOT / "model.ckpt"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "cloud_generation"
    / "output"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_TIF = OUTPUT_DIR / "s2_ecrformer_reconstruction.tif"
RECONSTRUCTION_PNG = OUTPUT_DIR / "ecrformer_reconstruction_preview.png"
COMPARISON_PNG = OUTPUT_DIR / "phase3c_comparison.png"
REPORT_PATH = OUTPUT_DIR / "phase3c_evaluation_report.json"


def to_rgb(s2: np.ndarray, brightness: float = 2.5) -> np.ndarray:
    """
    Convert normalized 13-band S2 to RGB.

    Band order:
    B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12

    RGB:
    R = B04 -> index 3
    G = B03 -> index 2
    B = B02 -> index 1
    """

    rgb = np.stack(
        [
            s2[3],
            s2[2],
            s2[1],
        ],
        axis=-1,
    )

    rgb = np.clip(rgb * brightness, 0.0, 1.0)

    return (rgb * 255.0).astype(np.uint8)


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid_mask: np.ndarray,
) -> dict:

    mask = valid_mask.astype(bool)

    pred = prediction[:, mask]
    true = target[:, mask]

    diff = pred - true

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    if rmse > 0:
        psnr = float(20.0 * np.log10(1.0 / rmse))
    else:
        psnr = float("inf")

    # Spectral Angle Mapper
    pred_vec = prediction[:, mask].T
    true_vec = target[:, mask].T

    dot = np.sum(pred_vec * true_vec, axis=1)

    pred_norm = np.linalg.norm(pred_vec, axis=1)
    true_norm = np.linalg.norm(true_vec, axis=1)

    denominator = pred_norm * true_norm

    valid_sam = denominator > 1e-8

    cosine = np.zeros_like(dot)

    cosine[valid_sam] = (
        dot[valid_sam] / denominator[valid_sam]
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    sam = float(
        np.mean(np.arccos(cosine[valid_sam]))
        * 180.0
        / np.pi
    )

    # SSIM on RGB for a meaningful visual-space measurement.
    pred_rgb = np.transpose(
        to_rgb(prediction),
        (2, 0, 1),
    )

    true_rgb = np.transpose(
        to_rgb(target),
        (2, 0, 1),
    )

    # Convert RGB back to [0,1].
    pred_rgb = pred_rgb.astype(np.float32) / 255.0
    true_rgb = true_rgb.astype(np.float32) / 255.0

    # Because invalid pixels should not contribute, replace them
    # with the target value before calculating SSIM.
    valid_2d = mask

    pred_rgb[:, ~valid_2d] = true_rgb[:, ~valid_2d]

    ssim_values = []

    for c in range(3):
        score = structural_similarity(
            true_rgb[c],
            pred_rgb[c],
            data_range=1.0,
        )
        ssim_values.append(score)

    ssim = float(np.mean(ssim_values))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "PSNR_dB": psnr,
        "SAM_degrees": sam,
        "SSIM_RGB": ssim,
    }


def main() -> None:

    print("=" * 76)
    print("PHASE 3C - ECRFORMER SYNTHETIC CLOUD REMOVAL")
    print("=" * 76)

    required_files = [
        S1_PATH,
        CLOUDY_S2_PATH,
        CLEAR_S2_PATH,
        VALIDITY_PATH,
        CHECKPOINT_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    print("\nINPUTS")
    print("-" * 76)
    print("S1:         ", S1_PATH)
    print("Cloudy S2:  ", CLOUDY_S2_PATH)
    print("Clear S2:   ", CLEAR_S2_PATH)
    print("Mask:       ", VALIDITY_PATH)
    print("Checkpoint: ", CHECKPOINT_PATH)

    with rasterio.open(VALIDITY_PATH) as src:
        validity_mask = src.read(1).astype(bool)

    print("\nVALID PIXELS:", int(validity_mask.sum()))
    print("TOTAL PIXELS:", int(validity_mask.size))

    device = get_device()

    print("\nLoading ECRformer...")
    model = load_model(
        CHECKPOINT_PATH,
        device,
    )

    sample = {
        "s1": str(S1_PATH),
        "s2_cloudy": str(CLOUDY_S2_PATH),
        "s2_clear": str(CLEAR_S2_PATH),
    }

    print("\nRunning tiled ECRformer inference...")
    print("-" * 76)

    def progress(done, total):
        print(
            f"\rTiles: {done}/{total}",
            end="",
            flush=True,
        )

    result = run_sample(
        model=model,
        sample=sample,
        device=device,
        tile_size=128,
        overlap=32,
        progress_callback=progress,
    )

    print("\n")

    prediction = result["prediction"]
    target = result["target"]

    print("Prediction shape:", prediction.shape)
    print("Target shape:    ", target.shape)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    metrics = compute_metrics(
        prediction,
        target,
        validity_mask,
    )

    print("=" * 76)
    print("EVALUATION RESULTS")
    print("=" * 76)

    for name, value in metrics.items():
        print(f"{name:<18}: {value:.6f}")

    # ------------------------------------------------------------
    # Save prediction GeoTIFF
    # ------------------------------------------------------------

    profile = result["profile"].copy()

    profile.update(
        count=13,
        dtype="float32",
        compress="deflate",
        nodata=0.0,
    )

    with rasterio.open(
        PREDICTION_TIF,
        "w",
        **profile,
    ) as dst:

        dst.write(
            prediction.astype(np.float32)
        )

        band_names = [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B10",
            "B11",
            "B12",
        ]

        for i, name in enumerate(band_names):
            dst.set_band_description(i + 1, name)

        dst.update_tags(
            processing="Phase 3C ECRformer Synthetic Cloud Removal",
            model="ECRformer",
            input_s1=str(S1_PATH),
            input_cloudy_s2=str(CLOUDY_S2_PATH),
            reference_clear_s2=str(CLEAR_S2_PATH),
        )

    # ------------------------------------------------------------
    # Create visual comparison
    # ------------------------------------------------------------

    cloudy_rgb = to_rgb(
        result["cloudy"],
        brightness=2.5,
    )

    prediction_rgb = to_rgb(
        prediction,
        brightness=2.5,
    )

    target_rgb = to_rgb(
        target,
        brightness=2.5,
    )

    Image.fromarray(
        prediction_rgb
    ).save(RECONSTRUCTION_PNG)

    comparison = np.concatenate(
        [
            cloudy_rgb,
            prediction_rgb,
            target_rgb,
        ],
        axis=1,
    )

    Image.fromarray(
        comparison
    ).save(COMPARISON_PNG)

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    report = {
        "phase": "3C",
        "model": "ECRformer",
        "device": str(device),
        "tile_size": 128,
        "overlap": 32,
        "valid_pixels": int(validity_mask.sum()),
        "total_pixels": int(validity_mask.size),
        "metrics": metrics,
        "inputs": {
            "s1": str(S1_PATH),
            "cloudy_s2": str(CLOUDY_S2_PATH),
            "clear_s2_reference": str(CLEAR_S2_PATH),
        },
        "outputs": {
            "prediction_tif": str(PREDICTION_TIF),
            "reconstruction_preview": str(RECONSTRUCTION_PNG),
            "comparison_preview": str(COMPARISON_PNG),
        },
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nOUTPUTS")
    print("-" * 76)
    print("Prediction TIFF: ", PREDICTION_TIF)
    print("Reconstruction:  ", RECONSTRUCTION_PNG)
    print("Comparison:      ", COMPARISON_PNG)
    print("Report:          ", REPORT_PATH)

    print("\n" + "=" * 76)
    print("PHASE 3C COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()