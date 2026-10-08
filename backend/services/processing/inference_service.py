"""
Canonical Inference Service for Generative AI-Based Satellite Cloud Removal.

Production Data Contract:
- S1 harmonized TIFF (2 bands: VV, VH, float32 dB)
- S2 cloudy TIFF (13 bands: B01-B12, uint16 DN)
- Validity mask TIFF (1 band: uint8, 1=valid, 0=invalid)
- ECRformer checkpoint (model.ckpt)
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.io import DatasetReader, MemoryFile
import torch
from PIL import Image

from models.ecrformer_model import ECRformerModel
from preprocessing import normalize_s1_safe, normalize_s2_safe
from visualization import to_rgb, sar_preview

S2_BAND_NAMES: List[str] = [
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
    "B8A", "B09", "B10", "B11", "B12"
]


@dataclass
class InferenceConfig:
    tile_size: int = 128
    overlap: int = 32
    device: Optional[str] = None
    nodata_value: float = 0.0
    brightness: float = 2.5
    checkpoint_path: Optional[Union[str, Path]] = None


@dataclass
class InferenceDiagnostics:
    device: str
    execution_time_seconds: float
    tile_count: int
    total_pixels: int
    valid_pixels: int
    invalid_pixels: int
    output_shape: Tuple[int, int, int]
    min_val: float
    max_val: float
    mean_val: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "tile_count": self.tile_count,
            "total_pixels": self.total_pixels,
            "valid_pixels": self.valid_pixels,
            "invalid_pixels": self.invalid_pixels,
            "valid_percentage": round(100.0 * self.valid_pixels / max(self.total_pixels, 1), 2),
            "output_shape": list(self.output_shape),
            "valid_min": round(self.min_val, 6),
            "valid_max": round(self.max_val, 6),
            "valid_mean": round(self.mean_val, 6),
        }


@dataclass
class InferenceResult:
    prediction: np.ndarray
    valid_mask: np.ndarray
    profile: Dict[str, Any]
    diagnostics: InferenceDiagnostics
    cloudy_rgb: np.ndarray
    prediction_rgb: np.ndarray
    sar_preview_img: Image.Image
    output_tif_path: Optional[Path] = None
    output_png_path: Optional[Path] = None

    def save_geotiff(self, path: Union[str, Path]) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        prof = self.profile.copy()
        prof.update(
            count=13,
            dtype="float32",
            nodata=0.0,
            compress="deflate",
        )
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(self.prediction.astype(np.float32))
            for i, name in enumerate(S2_BAND_NAMES):
                dst.set_band_description(i + 1, name)
            dst.update_tags(
                processing="ECRformer Canonical Baseline Inference",
                model="ECRformer",
                device=self.diagnostics.device,
                execution_time_s=str(round(self.diagnostics.execution_time_seconds, 2)),
                valid_pixels=str(self.diagnostics.valid_pixels),
                invalid_pixels=str(self.diagnostics.invalid_pixels),
            )
        self.output_tif_path = out_path
        return out_path

    def save_preview_png(self, path: Union[str, Path]) -> Path:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(self.prediction_rgb).save(out_path, format="PNG")
        self.output_png_path = out_path
        return out_path

    def to_geotiff_bytes(self) -> bytes:
        prof = self.profile.copy()
        prof.update(
            count=13,
            dtype="float32",
            nodata=0.0,
            compress="deflate",
        )
        with MemoryFile() as memfile:
            with memfile.open(**prof) as dst:
                dst.write(self.prediction.astype(np.float32))
                for i, name in enumerate(S2_BAND_NAMES):
                    dst.set_band_description(i + 1, name)
            return memfile.read()

    def to_png_bytes(self) -> bytes:
        buf = io.BytesIO()
        Image.fromarray(self.prediction_rgb).save(buf, format="PNG")
        return buf.getvalue()


def get_device(preferred: Optional[str] = None) -> torch.device:
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def create_ecrformer_model() -> ECRformerModel:
    return ECRformerModel(
        in_chans=[2, 13],
        out_chans=13,
        num_layers=4,
        num_blocks=[2, 3, 2, 2],
        features_start=48,
        drop_path_rate=0.0,
        bilinear=False,
        cbam="1ca2+1sa2",
        block_type=["ecrformer", "ecrformer"],
        conv_type="conv",
        norm_type="batch",
        decoupled_input=True,
        bottle_neck="tsa",
        num_refine=4,
        pos_encoding=None,
    )


def extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format; expected dictionary.")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {}
    for key, val in state_dict.items():
        if key.startswith("net."):
            key = key[4:]
        cleaned[key] = val
    return cleaned


def load_ecrformer_model(
    checkpoint_path: Union[str, Path],
    device: torch.device,
) -> ECRformerModel:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")
    model = create_ecrformer_model()
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cleaned_dict = extract_state_dict(checkpoint)
    model.load_state_dict(cleaned_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def compute_tile_positions(length: int, tile_size: int, overlap: int) -> List[int]:
    step = tile_size - overlap
    if step <= 0:
        raise ValueError(f"tile_size ({tile_size}) must be strictly greater than overlap ({overlap})")
    pos = list(range(0, max(1, length - tile_size + 1), step))
    if not pos or pos[-1] + tile_size < length:
        pos.append(max(0, length - tile_size))
    return sorted(list(set(pos)))


class InferenceService:
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        if checkpoint_path is None:
            root = Path(__file__).resolve().parent.parent.parent.parent
            default_ckpt = root / "model.ckpt"
            if default_ckpt.exists():
                checkpoint_path = default_ckpt
            else:
                raise FileNotFoundError(f"Default checkpoint not found at {default_ckpt}")
        self.checkpoint_path = Path(checkpoint_path)
        self.device = get_device(device)
        self._model: Optional[ECRformerModel] = None

    def get_model(self) -> ECRformerModel:
        if self._model is None:
            self._model = load_ecrformer_model(self.checkpoint_path, self.device)
        return self._model

    @staticmethod
    def _open_raster(
        raster_input: Union[str, Path, bytes, MemoryFile, DatasetReader]
    ) -> Tuple[DatasetReader, Optional[MemoryFile]]:
        if isinstance(raster_input, (str, Path)):
            p = Path(raster_input)
            if not p.exists():
                raise FileNotFoundError(f"Raster file not found: {p}")
            return rasterio.open(p), None
        elif isinstance(raster_input, bytes):
            memfile = MemoryFile(raster_input)
            return memfile.open(), memfile
        elif isinstance(raster_input, MemoryFile):
            return raster_input.open(), None
        elif isinstance(raster_input, DatasetReader):
            return raster_input, None
        else:
            raise TypeError(f"Unsupported raster input type: {type(raster_input)}")

    def validate_inputs(
        self,
        s1_src: DatasetReader,
        s2_src: DatasetReader,
        mask_src: Optional[DatasetReader] = None,
    ) -> None:
        if s1_src.count != 2:
            raise ValueError(f"Sentinel-1 must contain exactly 2 bands (VV, VH). Found {s1_src.count}.")
        if s2_src.count != 13:
            raise ValueError(f"Sentinel-2 cloudy must contain exactly 13 bands (B01-B12). Found {s2_src.count}.")
        if (s1_src.width, s1_src.height) != (s2_src.width, s2_src.height):
            raise ValueError(
                f"S1 dimensions ({s1_src.width}x{s1_src.height}) and S2 dimensions "
                f"({s2_src.width}x{s2_src.height}) do not match."
            )
        if s1_src.crs != s2_src.crs:
            raise ValueError(
                f"CRS mismatch: S1 CRS is {s1_src.crs}, but S2 CRS is {s2_src.crs}. Harmonization required."
            )
        for a, b in zip(s1_src.transform, s2_src.transform):
            if abs(a - b) > 1e-4:
                raise ValueError(
                    f"Affine transform mismatch between S1 and S2: {s1_src.transform} vs {s2_src.transform}"
                )

        if mask_src is not None:
            if mask_src.count != 1:
                raise ValueError(f"Validity mask must have 1 band. Found {mask_src.count}.")
            if (mask_src.width, mask_src.height) != (s2_src.width, s2_src.height):
                raise ValueError(
                    f"Validity mask dimensions ({mask_src.width}x{mask_src.height}) do not match S2."
                )
            if mask_src.crs != s2_src.crs:
                raise ValueError(f"Validity mask CRS ({mask_src.crs}) does not match S2 CRS ({s2_src.crs}).")

    def run(
        self,
        s1_input: Union[str, Path, bytes, MemoryFile, DatasetReader],
        s2_cloudy_input: Union[str, Path, bytes, MemoryFile, DatasetReader],
        validity_mask_input: Optional[Union[str, Path, bytes, MemoryFile, DatasetReader]] = None,
        config: Optional[InferenceConfig] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> InferenceResult:
        if config is None:
            config = InferenceConfig(device=str(self.device))

        start_time = time.perf_counter()

        s1_src, s1_mem = self._open_raster(s1_input)
        s2_src, s2_mem = self._open_raster(s2_cloudy_input)
        mask_src, mask_mem = (
            self._open_raster(validity_mask_input) if validity_mask_input is not None else (None, None)
        )

        try:
            self.validate_inputs(s1_src, s2_src, mask_src)

            width = s2_src.width
            height = s2_src.height
            profile = s2_src.profile.copy()

            s1_raw = s1_src.read().astype(np.float32)
            s2_raw = s2_src.read().astype(np.float32)

            if mask_src is not None:
                mask_raw = mask_src.read(1)
                valid_mask = mask_raw.astype(bool)
            else:
                valid_mask = (s1_raw[0] > -500.0) & (s2_raw[0] > 0.0)

            total_pixels = int(width * height)
            valid_pixels = int(valid_mask.sum())
            invalid_pixels = total_pixels - valid_pixels

            s1_norm = normalize_s1_safe(s1_raw, valid_mask=valid_mask)
            s2_norm = normalize_s2_safe(s2_raw, valid_mask=valid_mask)

            model_input = np.concatenate([s1_norm, s2_norm], axis=0).astype(np.float32)

            model = self.get_model()

            xs = compute_tile_positions(width, config.tile_size, config.overlap)
            ys = compute_tile_positions(height, config.tile_size, config.overlap)
            total_tiles = len(xs) * len(ys)

            prediction_sum = np.zeros((13, height, width), dtype=np.float32)
            weights = np.zeros((height, width), dtype=np.float32)

            completed_tiles = 0

            for y in ys:
                for x in xs:
                    h = min(config.tile_size, height - y)
                    w = min(config.tile_size, width - x)

                    tile = model_input[:, y : y + h, x : x + w]

                    padded_tile = np.zeros(
                        (15, config.tile_size, config.tile_size), dtype=np.float32
                    )
                    padded_tile[:, :h, :w] = tile

                    with torch.inference_mode():
                        tensor = torch.from_numpy(padded_tile).unsqueeze(0).to(self.device)
                        out = model(tensor)
                        if isinstance(out, tuple):
                            out = out[0]
                        tile_pred = out.squeeze(0).detach().cpu().numpy().astype(np.float32)

                    tile_pred = tile_pred[:, :h, :w]

                    prediction_sum[:, y : y + h, x : x + w] += tile_pred
                    weights[y : y + h, x : x + w] += 1.0

                    completed_tiles += 1
                    if progress_callback is not None:
                        progress_callback(completed_tiles, total_tiles)

            prediction = prediction_sum / np.maximum(weights, 1e-8)[None, :, :]
            prediction = np.clip(prediction, 0.0, 1.0)

            prediction[:, ~valid_mask] = config.nodata_value

            if valid_pixels > 0:
                valid_pred_vals = prediction[:, valid_mask]
                min_val = float(valid_pred_vals.min())
                max_val = float(valid_pred_vals.max())
                mean_val = float(valid_pred_vals.mean())
            else:
                min_val = max_val = mean_val = 0.0

            exec_time = time.perf_counter() - start_time

            diagnostics = InferenceDiagnostics(
                device=str(self.device),
                execution_time_seconds=exec_time,
                tile_count=total_tiles,
                total_pixels=total_pixels,
                valid_pixels=valid_pixels,
                invalid_pixels=invalid_pixels,
                output_shape=prediction.shape,
                min_val=min_val,
                max_val=max_val,
                mean_val=mean_val,
            )

            cloudy_rgb = to_rgb(s2_norm, brightness=config.brightness, valid_mask=valid_mask)
            prediction_rgb = to_rgb(prediction, brightness=config.brightness, valid_mask=valid_mask)
            sar_img = sar_preview(s1_raw, valid_mask=valid_mask)

            profile.update(
                count=13,
                dtype="float32",
                nodata=config.nodata_value,
                compress="deflate",
            )

            return InferenceResult(
                prediction=prediction,
                valid_mask=valid_mask,
                profile=profile,
                diagnostics=diagnostics,
                cloudy_rgb=cloudy_rgb,
                prediction_rgb=prediction_rgb,
                sar_preview_img=sar_img,
            )

        finally:
            s1_src.close()
            s2_src.close()
            if mask_src is not None:
                mask_src.close()
            if s1_mem is not None:
                s1_mem.close()
            if s2_mem is not None:
                s2_mem.close()
            if mask_mem is not None:
                mask_mem.close()
