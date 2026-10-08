from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parent.parent

S2_PATH = PROJECT_ROOT / "harmonization" / "output" / "s2_harmonized_10m.tif"
MASK_PATH = PROJECT_ROOT / "harmonization" / "output" / "validity_mask_10m.tif"

OUTPUT_DIR = PROJECT_ROOT / "cloud_generation" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLOUD_MASK_PATH = OUTPUT_DIR / "cloud_mask.tif"
PREVIEW_PATH = OUTPUT_DIR / "cloud_mask_preview.png"
REPORT_PATH = OUTPUT_DIR / "cloud_generation_report.json"

SEED = 42
TARGET_COVERAGE = 0.35


def resize_noise(height: int, width: int, small_h: int, small_w: int) -> np.ndarray:
    """Generate smooth random noise at multiple spatial scales."""
    rng = np.random.default_rng(SEED)

    small = rng.random((small_h, small_w), dtype=np.float32)

    image = Image.fromarray(
        np.uint8(np.clip(small, 0, 1) * 255),
        mode="L",
    )

    image = image.resize(
        (width, height),
        Image.Resampling.BICUBIC,
    )

    image = image.filter(ImageFilter.GaussianBlur(radius=2))

    return np.asarray(image, dtype=np.float32) / 255.0


def generate_cloud_field(height: int, width: int) -> np.ndarray:
    """Create irregular multi-scale cloud structure."""

    large = resize_noise(
        height,
        width,
        max(4, height // 32),
        max(4, width // 32),
    )

    medium = resize_noise(
        height,
        width,
        max(8, height // 12),
        max(8, width // 12),
    )

    fine = resize_noise(
        height,
        width,
        max(16, height // 5),
        max(16, width // 5),
    )

    field = (
        0.60 * large
        + 0.28 * medium
        + 0.12 * fine
    )

    field -= field.min()
    field /= field.max() + 1e-8

    return field


def make_soft_cloud_mask(
    field: np.ndarray,
    valid_mask: np.ndarray,
    target_coverage: float,
) -> np.ndarray:

    valid_values = field[valid_mask]

    threshold = float(
        np.quantile(valid_values, 1.0 - target_coverage)
    )

    # Width of the soft transition around the cloud boundary.
    transition = 0.10

    alpha = (
        field - (threshold - transition)
    ) / (2.0 * transition)

    alpha = np.clip(alpha, 0.0, 1.0)

    # Smoothstep gives more natural cloud edges.
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)

    # Invalid satellite pixels must remain cloud-free.
    alpha[~valid_mask] = 0.0

    return alpha.astype(np.float32)


def main() -> None:

    if not S2_PATH.exists():
        raise FileNotFoundError(f"S2 file not found: {S2_PATH}")

    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Validity mask not found: {MASK_PATH}")

    print("=" * 70)
    print("PHASE 3A - SYNTHETIC CLOUD MASK GENERATOR")
    print("=" * 70)

    with rasterio.open(S2_PATH) as src:
        profile = src.profile.copy()
        height = src.height
        width = src.width
        crs = src.crs
        transform = src.transform

    with rasterio.open(MASK_PATH) as mask_src:
        valid_mask = mask_src.read(1).astype(bool)

    print(f"Input S2:          {S2_PATH}")
    print(f"Input validity:    {MASK_PATH}")
    print(f"Image size:        {width} x {height}")
    print(f"Target coverage:   {TARGET_COVERAGE * 100:.1f}%")
    print(f"Random seed:       {SEED}")

    field = generate_cloud_field(height, width)

    cloud_alpha = make_soft_cloud_mask(
        field=field,
        valid_mask=valid_mask,
        target_coverage=TARGET_COVERAGE,
    )

    hard_cloud = cloud_alpha >= 0.5

    valid_pixels = int(valid_mask.sum())
    cloud_pixels = int(np.sum(hard_cloud & valid_mask))

    actual_coverage = (
        cloud_pixels / valid_pixels
        if valid_pixels > 0
        else 0.0
    )

    # Save cloud mask as float32 [0, 1].
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        nodata=0.0,
        compress="deflate",
    )

    with rasterio.open(CLOUD_MASK_PATH, "w", **output_profile) as dst:
        dst.write(cloud_alpha, 1)
        dst.set_band_description(1, "synthetic_cloud_alpha")

        dst.update_tags(
            description="Synthetic multi-scale cloud opacity mask",
            target_coverage=f"{TARGET_COVERAGE:.4f}",
            actual_hard_coverage=f"{actual_coverage:.4f}",
            seed=str(SEED),
            phase="Phase 3A Cloud Intelligence",
        )

    # Preview: black = clear, white = cloud.
    preview = np.uint8(np.clip(cloud_alpha, 0.0, 1.0) * 255)

    Image.fromarray(preview, mode="L").save(PREVIEW_PATH)

    report = {
        "phase": "3A",
        "input_s2": str(S2_PATH),
        "input_validity_mask": str(MASK_PATH),
        "width": width,
        "height": height,
        "target_coverage": TARGET_COVERAGE,
        "actual_hard_coverage": actual_coverage,
        "valid_pixels": valid_pixels,
        "cloud_pixels": cloud_pixels,
        "seed": SEED,
        "mask_min": float(cloud_alpha.min()),
        "mask_max": float(cloud_alpha.max()),
        "mask_mean": float(cloud_alpha[valid_mask].mean()),
        "crs": str(crs),
        "output": str(CLOUD_MASK_PATH),
        "preview": str(PREVIEW_PATH),
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("-" * 70)
    print(f"Valid pixels:      {valid_pixels}")
    print(f"Cloud pixels:      {cloud_pixels}")
    print(f"Actual coverage:   {actual_coverage * 100:.2f}%")
    print(f"Mask range:        {cloud_alpha.min():.3f} → {cloud_alpha.max():.3f}")
    print(f"Mask mean:         {cloud_alpha[valid_mask].mean():.3f}")
    print("-" * 70)
    print(f"Cloud mask:        {CLOUD_MASK_PATH}")
    print(f"Preview:           {PREVIEW_PATH}")
    print(f"Report:            {REPORT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()