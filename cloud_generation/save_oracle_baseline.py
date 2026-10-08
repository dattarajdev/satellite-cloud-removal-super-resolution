from pathlib import Path
import json

import numpy as np
import rasterio
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cloud_generation" / "output"

CLOUDY = OUT / "s2_cloudy_mock.tif"
PRED = OUT / "s2_ecrformer_reconstruction.tif"
CLEAR = ROOT / "harmonization" / "output" / "s2_harmonized_10m.tif"
CLOUD_MASK = OUT / "cloud_mask.tif"
VALIDITY = ROOT / "harmonization" / "output" / "validity_mask_10m.tif"

FUSED_TIF = OUT / "s2_oracle_fused.tif"
FUSED_PNG = OUT / "oracle_fused_preview.png"
REPORT = OUT / "oracle_baseline_report.json"


def rgb(s2):
    # B04, B03, B02
    x = np.stack([s2[3], s2[2], s2[1]], axis=-1)
    x = np.clip(x * 2.5, 0, 1)
    return (x * 255).astype(np.uint8)


def metrics(pred, target, mask):
    d = pred[:, mask] - target[:, mask]

    mae = float(np.mean(np.abs(d)))
    rmse = float(np.sqrt(np.mean(d ** 2)))
    psnr = float(20 * np.log10(1 / rmse))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "PSNR_dB": psnr,
    }


with rasterio.open(CLOUDY) as src:
    cloudy = src.read().astype(np.float32) / 10000.0
    profile = src.profile.copy()

with rasterio.open(PRED) as src:
    prediction = src.read().astype(np.float32)

with rasterio.open(CLEAR) as src:
    clear = src.read().astype(np.float32) / 10000.0

with rasterio.open(CLOUD_MASK) as src:
    cloud_mask = src.read(1) >= 0.5

with rasterio.open(VALIDITY) as src:
    valid = src.read(1).astype(bool)

cloud_mask &= valid

# Oracle fusion:
# use ECRformer only where the synthetic cloud exists.
fused = cloudy.copy()
fused[:, cloud_mask] = prediction[:, cloud_mask]

overall = metrics(fused, clear, valid)
cloud_only = metrics(fused, clear, cloud_mask)
clear_only = metrics(fused, clear, valid & ~cloud_mask)

profile.update(
    count=13,
    dtype="float32",
    nodata=0.0,
    compress="deflate",
)

with rasterio.open(FUSED_TIF, "w", **profile) as dst:
    dst.write(fused)

    names = [
        "B01", "B02", "B03", "B04", "B05", "B06", "B07",
        "B08", "B8A", "B09", "B10", "B11", "B12"
    ]

    for i, name in enumerate(names):
        dst.set_band_description(i + 1, name)

    dst.update_tags(
        processing="Oracle-mask ECRformer fusion",
        cloud_mask="known synthetic ground truth",
    )

Image.fromarray(rgb(fused)).save(FUSED_PNG)

report = {
    "experiment": "oracle_mask_baseline",
    "valid_pixels": int(valid.sum()),
    "cloud_pixels": int(cloud_mask.sum()),
    "clear_pixels": int((valid & ~cloud_mask).sum()),
    "overall": overall,
    "cloud_only": cloud_only,
    "clear_only": clear_only,
}

REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("=" * 60)
print("ORACLE MASK BASELINE")
print("=" * 60)
print(f"Cloud pixels: {cloud_mask.sum()}")
print(f"Clear pixels: {(valid & ~cloud_mask).sum()}")

print("\nOVERALL")
for k, v in overall.items():
    print(f"{k}: {v:.6f}")

print("\nCLOUD REGION")
for k, v in cloud_only.items():
    print(f"{k}: {v:.6f}")

print("\nCLEAR REGION")
for k, v in clear_only.items():
    print(f"{k}: {v:.6f}")

print("\nSaved:")
print(FUSED_TIF)
print(FUSED_PNG)
print(REPORT)