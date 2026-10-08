from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent

S2_PATH = (
    PROJECT_ROOT
    / "harmonization"
    / "output"
    / "s2_harmonized_10m.tif"
)

MASK_PATH = (
    PROJECT_ROOT
    / "cloud_generation"
    / "output"
    / "cloud_mask.tif"
)

VALIDITY_PATH = (
    PROJECT_ROOT
    / "harmonization"
    / "output"
    / "validity_mask_10m.tif"
)

OUTPUT_DIR = PROJECT_ROOT / "cloud_generation" / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLOUDY_S2_PATH = OUTPUT_DIR / "s2_cloudy_mock.tif"
CLEAR_PREVIEW_PATH = OUTPUT_DIR / "clear_reference_preview.png"
CLOUDY_PREVIEW_PATH = OUTPUT_DIR / "cloudy_preview.png"
REPORT_PATH = OUTPUT_DIR / "cloud_application_report.json"


def make_rgb_preview(
    s2: np.ndarray,
    output_path: Path,
    brightness: float = 2.5,
) -> None:
    """
    Create an RGB preview using:
    R = B04
    G = B03
    B = B02
    """

    # S2_BAND_ORDER:
    # B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B10 B11 B12
    blue = s2[1].astype(np.float32)
    green = s2[2].astype(np.float32)
    red = s2[3].astype(np.float32)

    rgb = np.stack([red, green, blue], axis=-1)

    rgb = np.clip(rgb / 10000.0 * brightness, 0.0, 1.0)
    rgb = (rgb * 255.0).astype(np.uint8)

    Image.fromarray(rgb).save(output_path)


def main() -> None:

    print("=" * 72)
    print("PHASE 3B - SYNTHETIC CLOUD APPLICATION")
    print("=" * 72)

    if not S2_PATH.exists():
        raise FileNotFoundError(f"S2 file not found: {S2_PATH}")

    if not MASK_PATH.exists():
        raise FileNotFoundError(f"Cloud mask not found: {MASK_PATH}")

    if not VALIDITY_PATH.exists():
        raise FileNotFoundError(
            f"Validity mask not found: {VALIDITY_PATH}"
        )

    # ------------------------------------------------------------
    # Load S2
    # ------------------------------------------------------------

    with rasterio.open(S2_PATH) as src:
        s2 = src.read().astype(np.float32)
        profile = src.profile.copy()
        width = src.width
        height = src.height
        crs = src.crs
        transform = src.transform

    # ------------------------------------------------------------
    # Load synthetic cloud opacity mask
    # ------------------------------------------------------------

    with rasterio.open(MASK_PATH) as src:
        cloud_alpha = src.read(1).astype(np.float32)

    # ------------------------------------------------------------
    # Load spatial validity mask
    # ------------------------------------------------------------

    with rasterio.open(VALIDITY_PATH) as src:
        valid_mask = src.read(1).astype(bool)

    if s2.shape[1:] != cloud_alpha.shape:
        raise ValueError(
            f"S2 shape {s2.shape[1:]} does not match "
            f"cloud mask {cloud_alpha.shape}"
        )

    if s2.shape[1:] != valid_mask.shape:
        raise ValueError(
            f"S2 shape {s2.shape[1:]} does not match "
            f"validity mask {valid_mask.shape}"
        )

    # ------------------------------------------------------------
    # Ensure cloud mask only exists on valid satellite pixels
    # ------------------------------------------------------------

    cloud_alpha = np.clip(cloud_alpha, 0.0, 1.0)
    cloud_alpha[~valid_mask] = 0.0

    print(f"Input S2 shape:     {s2.shape}")
    print(f"Image size:         {width} x {height}")
    print(f"Valid pixels:       {valid_mask.sum()}")
    print(
        f"Cloud mean opacity: "
        f"{cloud_alpha[valid_mask].mean():.4f}"
    )

    # ------------------------------------------------------------
    # Estimate band-specific synthetic cloud reflectance
    #
    # We use a high percentile from the actual scene rather than
    # forcing every band to the same DN value.
    #
    # This is a synthetic degradation model, not a physical
    # atmospheric radiative-transfer simulation.
    # ------------------------------------------------------------

    cloud_values = []

    for band_idx in range(s2.shape[0]):

        band = s2[band_idx]

        valid_values = band[valid_mask]

        # Ignore zero values when possible.
        valid_values = valid_values[valid_values > 0]

        if valid_values.size == 0:
            cloud_value = 10000.0
        else:
            cloud_value = float(
                np.percentile(valid_values, 97.0)
            )

        cloud_values.append(cloud_value)

    cloud_values = np.asarray(
        cloud_values,
        dtype=np.float32,
    )

    print("\nSynthetic cloud DN values:")

    for i, value in enumerate(cloud_values):
        print(f"  Band {i + 1:02d}: {value:.2f}")

    # ------------------------------------------------------------
    # Apply synthetic cloud degradation
    #
    # cloudy = clear * (1 - alpha) + cloud_value * alpha
    # ------------------------------------------------------------

    cloudy = np.empty_like(s2)

    for band_idx in range(s2.shape[0]):

        clear_band = s2[band_idx]

        cloudy_band = (
            clear_band * (1.0 - cloud_alpha)
            + cloud_values[band_idx] * cloud_alpha
        )

        cloudy[band_idx] = cloudy_band

    # ------------------------------------------------------------
    # Preserve invalid pixels exactly as NoData
    # ------------------------------------------------------------

    cloudy[:, ~valid_mask] = 0.0

    # Prevent negative values and keep uint16-compatible range.
    cloudy = np.clip(cloudy, 0.0, 65535.0)

    cloudy = cloudy.astype(np.uint16)

    # ------------------------------------------------------------
    # Write cloudy Sentinel-2 GeoTIFF
    # ------------------------------------------------------------

    profile.update(
        driver="GTiff",
        count=13,
        dtype="uint16",
        nodata=0,
        compress="deflate",
    )

    with rasterio.open(
        CLOUDY_S2_PATH,
        "w",
        **profile,
    ) as dst:

        dst.write(cloudy)

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
            processing="Phase 3B Synthetic Cloud Application",
            cloud_model="multi-scale alpha blending",
            cloud_mask=str(MASK_PATH),
            validity_mask=str(VALIDITY_PATH),
            percentile=97,
            physical_model=False,
        )

    # ------------------------------------------------------------
    # Create previews
    # ------------------------------------------------------------

    make_rgb_preview(
        s2=s2,
        output_path=CLEAR_PREVIEW_PATH,
    )

    make_rgb_preview(
        s2=cloudy,
        output_path=CLOUDY_PREVIEW_PATH,
    )

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------

    cloud_pixels = int(
        np.sum((cloud_alpha >= 0.5) & valid_mask)
    )

    valid_pixels = int(valid_mask.sum())

    hard_coverage = (
        cloud_pixels / valid_pixels
        if valid_pixels > 0
        else 0.0
    )

    report = {
        "phase": "3B",
        "input_s2": str(S2_PATH),
        "cloud_mask": str(MASK_PATH),
        "validity_mask": str(VALIDITY_PATH),
        "output_cloudy_s2": str(CLOUDY_S2_PATH),
        "width": width,
        "height": height,
        "bands": 13,
        "valid_pixels": valid_pixels,
        "cloud_pixels": cloud_pixels,
        "hard_cloud_coverage": hard_coverage,
        "cloud_alpha_min": float(cloud_alpha.min()),
        "cloud_alpha_max": float(cloud_alpha.max()),
        "cloud_alpha_mean": float(
            cloud_alpha[valid_mask].mean()
        ),
        "cloud_values_dn": cloud_values.tolist(),
        "method": (
            "Synthetic spectral cloud degradation using "
            "band-specific 97th percentile DN values "
            "and alpha blending."
        ),
        "physical_cloud_model": False,
    }

    REPORT_PATH.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------

    print("\n" + "-" * 72)
    print("OUTPUT")
    print("-" * 72)

    print(f"Cloudy S2:      {CLOUDY_S2_PATH}")
    print(f"Clear preview:  {CLEAR_PREVIEW_PATH}")
    print(f"Cloudy preview: {CLOUDY_PREVIEW_PATH}")
    print(f"Report:         {REPORT_PATH}")

    print("\n" + "=" * 72)
    print("PHASE 3B COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()