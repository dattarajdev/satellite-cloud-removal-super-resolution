"""
Inference module for ECRformer Cloud Removal.
Thin delegation wrapper around the canonical backend.services.processing.InferenceService.
Maintains backwards compatibility for evaluation scripts while routing all execution
through the single authoritative service.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import rasterio
import torch

from backend.services.processing.inference_service import (
    InferenceConfig,
    InferenceResult,
    InferenceService,
    create_ecrformer_model as create_model,
    extract_state_dict,
    get_device,
    load_ecrformer_model as load_model,
)
from preprocessing import normalize_s1_safe, normalize_s2_safe
from visualization import to_rgb


def save_geotiff(
    prediction: np.ndarray,
    profile: Dict[str, Any],
    path: Union[str, Path],
) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prof = profile.copy()
    prof.update(
        count=13,
        dtype="float32",
        nodata=0.0,
        compress="deflate",
    )
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(prediction.astype(np.float32))
    return out_path


def run_sample(
    model: torch.nn.Module,
    sample: Dict[str, Any],
    device: torch.device,
    tile_size: int = 128,
    overlap: int = 32,
    progress_callback: Optional[Any] = None,
) -> Dict[str, Any]:
    service = InferenceService(device=str(device))
    service._model = model

    config = InferenceConfig(
        tile_size=tile_size,
        overlap=overlap,
        device=str(device),
    )

    s1_path = sample["s1"]
    cloudy_path = sample["s2_cloudy"]
    mask_path = sample.get("validity_mask")

    result = service.run(
        s1_input=s1_path,
        s2_cloudy_input=cloudy_path,
        validity_mask_input=mask_path,
        config=config,
        progress_callback=progress_callback,
    )

    out_dict = {
        "prediction": result.prediction,
        "cloudy": normalize_s2_safe(rasterio.open(cloudy_path).read().astype(np.float32), valid_mask=result.valid_mask),
        "s1": normalize_s1_safe(rasterio.open(s1_path).read().astype(np.float32), valid_mask=result.valid_mask),
        "profile": result.profile,
        "diagnostics": result.diagnostics.to_dict(),
        "result_obj": result,
    }

    if "s2_clear" in sample and sample["s2_clear"] is not None:
        target_raw = rasterio.open(sample["s2_clear"]).read().astype(np.float32)
        out_dict["target"] = normalize_s2_safe(target_raw, valid_mask=result.valid_mask)

    return out_dict


def main():
    parser = argparse.ArgumentParser(description="Canonical ECRformer Inference CLI")
    parser.add_argument("--s1", type=str, default="harmonization/output/s1_harmonized_10m.tif")
    parser.add_argument("--s2", type=str, default="harmonization/output/s2_harmonized_10m.tif")
    parser.add_argument("--mask", type=str, default="harmonization/output/validity_mask_10m.tif")
    parser.add_argument("--checkpoint", type=str, default="model.ckpt")
    parser.add_argument("--out", type=str, default="output/reconstructed_s2.tif")
    parser.add_argument("--preview", type=str, default="output/reconstructed_preview.png")
    parser.add_argument("--tile-size", type=int, default=128)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)

    args = parser.parse_args()

    service = InferenceService(checkpoint_path=args.checkpoint, device=args.device)
    config = InferenceConfig(
        tile_size=args.tile_size,
        overlap=args.overlap,
        device=args.device,
    )

    def progress(done, total):
        print(f"\rInferring tiles: {done}/{total} ({100.0*done/total:.1f}%)", end="", flush=True)

    mask_path = Path(args.mask)
    result = service.run(
        s1_input=args.s1,
        s2_cloudy_input=args.s2,
        validity_mask_input=mask_path if mask_path.exists() else None,
        config=config,
        progress_callback=progress,
    )
    print("\nInference complete!")
    print("Diagnostics:", result.diagnostics.to_dict())

    tif_path = result.save_geotiff(args.out)
    png_path = result.save_preview_png(args.preview)
    print(f"Saved GeoTIFF: {tif_path}")
    print(f"Saved Preview: {png_path}")


if __name__ == "__main__":
    main()
