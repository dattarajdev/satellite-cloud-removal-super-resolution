"""
OpenSR Diffusion Super-Resolution Integration Service.

Wraps the teammate OpenSR project (diffffusiosm) via isolated subprocess execution
using its dedicated Python virtual environment.

Key contracts:
- Authoritative 13-band ECRformer GeoTIFF is NEVER modified or overwritten.
- Post-processing selects B04, B03, B02, B08 from the 13-band ECRformer output.
- Generates 4-band 2.5m RGBN GeoTIFF (4x super-resolution).
- Generates 3-band RGB preview rendered strictly from B04, B03, B02 (B08 NIR strictly excluded).
- Full geospatial metadata (CRS, bounds, transform) preserved with 2.5m resolution.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import rasterio

from .band_selector import validate_and_get_band_indices

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_TEAMMATE_DIR = PROJECT_ROOT / "diffffusiosm"
DEFAULT_PYTHON_EXEC = DEFAULT_TEAMMATE_DIR / "venv" / "Scripts" / "python.exe"
DEFAULT_SCRIPT_PATH = DEFAULT_TEAMMATE_DIR / "process_tif.py"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reconstructed_images"


@dataclass
class OpenSRConfig:
    """Configuration parameters for the OpenSR diffusion super-resolution service."""
    teammate_dir: Path = field(default_factory=lambda: DEFAULT_TEAMMATE_DIR)
    python_executable: Path = field(default_factory=lambda: DEFAULT_PYTHON_EXEC)
    script_path: Path = field(default_factory=lambda: DEFAULT_SCRIPT_PATH)
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)
    timeout_seconds: int = 600
    brightness: float = 3.0


@dataclass
class OpenSRDiagnostics:
    """Detailed execution diagnostics for the OpenSR stage."""
    input_path: str
    output_path: str
    input_width: int
    input_height: int
    output_width: int
    output_height: int
    scale_factor: float
    input_bands: int
    output_bands: int
    selected_bands: List[str]
    execution_time_seconds: float
    execution_device: str
    sampling_steps: Optional[int]
    gpu_memory: Optional[str]
    output_crs: str
    output_pixel_size_m: float
    bounds_preserved: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": "OpenSR LDSR-S2 (Latent Diffusion)",
            "scale_factor": f"{self.scale_factor}x",
            "input_resolution_m": 10.0,
            "output_resolution_m": self.output_pixel_size_m,
            "input_dimensions": [self.input_width, self.input_height],
            "output_dimensions": [self.output_width, self.output_height],
            "input_bands": self.input_bands,
            "output_bands": self.output_bands,
            "selected_bands": self.selected_bands,
            "execution_device": self.execution_device,
            "execution_time_seconds": round(self.execution_time_seconds, 3),
            "gpu_memory": self.gpu_memory,
            "sampling_steps": self.sampling_steps,
            "output_crs": self.output_crs,
            "bounds_preserved": self.bounds_preserved,
        }


@dataclass
class OpenSRResult:
    """Complete product container for the super-resolved output."""
    sr_geotiff_path: Path
    sr_rgb_preview: Image.Image
    profile: Dict[str, Any]
    diagnostics: OpenSRDiagnostics
    preview_png_path: Optional[Path] = None

    def to_geotiff_bytes(self) -> bytes:
        """Return the 4-band GeoTIFF as raw bytes for Streamlit download."""
        with open(self.sr_geotiff_path, "rb") as f:
            return f.read()

    def to_png_bytes(self) -> bytes:
        """Return the 2.5m RGB preview as PNG bytes for Streamlit download."""
        buf = io.BytesIO()
        self.sr_rgb_preview.save(buf, format="PNG")
        return buf.getvalue()

    def save_geotiff(self, path: Union[str, Path]) -> Path:
        """Copy the 4-band GeoTIFF to an external destination."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.resolve() != self.sr_geotiff_path.resolve():
            shutil.copy2(self.sr_geotiff_path, out_path)
        return out_path

    def save_preview_png(self, path: Union[str, Path]) -> Path:
        """Save the 2.5m RGB preview to disk."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.sr_rgb_preview.save(out_path, format="PNG")
        self.preview_png_path = out_path
        return out_path


class OpenSRService:
    """
    Subprocess orchestrator for OpenSR diffusion super-resolution.

    Executes process_tif.py directly via the teammate project's virtualenv,
    bypassing Run Super Resolution.bat to avoid terminal blocking from 'pause'.
    """

    def __init__(self, config: Optional[OpenSRConfig] = None):
        self.config = config or OpenSRConfig()

    def validate_environment(self) -> None:
        """Validate presence of the teammate Python environment and runner script."""
        if not self.config.python_executable.is_file():
            raise FileNotFoundError(
                f"Teammate Python environment not found: {self.config.python_executable}. "
                "Ensure diffffusiosm/venv is set up."
            )
        if not self.config.script_path.is_file():
            raise FileNotFoundError(
                f"Teammate processing script not found: {self.config.script_path}."
            )

    def validate_input(self, input_path: Path) -> Tuple[int, int, int, Tuple[int, ...]]:
        """Validate that input exists, is a valid GeoTIFF, and contains required bands."""
        if not input_path.is_file() or input_path.suffix.lower() not in {".tif", ".tiff"}:
            raise FileNotFoundError(
                f"Input TIFF not found or invalid format: {input_path}"
            )

        with rasterio.open(input_path) as src:
            band_indices = validate_and_get_band_indices(src)
            return src.width, src.height, src.count, band_indices

    def render_rgb_preview(
        self, sr_tif_path: Path, brightness: float = 3.0
    ) -> Image.Image:
        """
        Render 3-band RGB preview strictly from B04 (Red), B03 (Green), B02 (Blue).
        B08 (NIR) is strictly excluded from RGB visualization.
        """
        with rasterio.open(sr_tif_path) as src:
            # Expected bands in OpenSR output: B4, B3, B2, B8 (1-indexed: 1=B4, 2=B3, 3=B2, 4=B8)
            b4 = src.read(1).astype(np.float32)
            b3 = src.read(2).astype(np.float32)
            b2 = src.read(3).astype(np.float32)

            # Reflectance scale in OpenSR output is uint16 [0, 10000]
            rgb = np.stack([b4, b3, b2], axis=0) / 10000.0 * brightness
            rgb = np.clip(rgb, 0.0, 1.0)
            rgb_uint8 = (np.transpose(rgb, (1, 2, 0)) * 255.0).astype(np.uint8)
            return Image.fromarray(rgb_uint8)

    def run(
        self,
        input_tif_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        config_override: Optional[OpenSRConfig] = None,
    ) -> OpenSRResult:
        """
        Execute OpenSR diffusion super-resolution on an input GeoTIFF.

        Args:
            input_tif_path: Absolute or relative path to the 13-band ECRformer GeoTIFF.
            output_dir: Directory where the super-resolved GeoTIFF should be placed.
            config_override: Optional OpenSRConfig to override defaults.

        Returns:
            OpenSRResult containing output path, diagnostics, and 2.5m RGB preview.
        """
        cfg = config_override or self.config
        input_path = Path(input_tif_path).resolve()
        out_dir = Path(output_dir or cfg.output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Pre-flight verification
        self.validate_environment()
        in_width, in_height, in_bands, _ = self.validate_input(input_path)

        # 2. Subprocess invocation
        cmd = [
            str(cfg.python_executable),
            str(cfg.script_path),
            str(input_path),
        ]
        logger.info(f"Running OpenSR subprocess: {' '.join(cmd)}")

        start_time = time.time()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=cfg.timeout_seconds,
                cwd=str(cfg.teammate_dir),
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(
                f"OpenSR super-resolution timed out after {cfg.timeout_seconds} seconds."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Failed to launch OpenSR subprocess: {exc}") from exc

        elapsed_time = time.time() - start_time

        if proc.returncode != 0:
            err_msg = (
                f"OpenSR subprocess failed with code {proc.returncode}.\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )
            logger.error(err_msg)
            raise RuntimeError(err_msg)

        # 3. Parse execution device from stdout
        exec_device = "CUDA" if "CUDA" in proc.stdout.upper() else "CPU"

        # 4. Locate teammate output and materialize into main project
        teammate_out_tif = cfg.teammate_dir / "outputs" / f"{input_path.stem}_sr.tif"
        if not teammate_out_tif.is_file():
            raise FileNotFoundError(
                f"OpenSR finished but expected output was not found: {teammate_out_tif}"
            )

        destination_tif = out_dir / f"{input_path.stem}_sr_4band_2.5m.tif"
        shutil.copy2(teammate_out_tif, destination_tif)

        # 5. Validate geospatial metadata of output
        with rasterio.open(input_path) as src_in, rasterio.open(destination_tif) as src_out:
            out_width = src_out.width
            out_height = src_out.height
            out_bands = src_out.count
            out_crs = str(src_out.crs)
            out_pixel_size = abs(src_out.transform.a)
            profile = src_out.profile.copy()

            # Verify 4x dimensions
            if out_width != in_width * 4 or out_height != in_height * 4:
                raise ValueError(
                    f"Unexpected output dimensions: {out_width}x{out_height}. "
                    f"Expected 4x input: {in_width * 4}x{in_height * 4}"
                )

            # Verify 4 bands
            if out_bands != 4:
                raise ValueError(f"Expected 4 output bands, found {out_bands}")

            # Verify bounds match within 1e-2 tolerance
            bounds_match = (
                abs(src_out.bounds.left - src_in.bounds.left) < 1e-2
                and abs(src_out.bounds.right - src_in.bounds.right) < 1e-2
                and abs(src_out.bounds.bottom - src_in.bounds.bottom) < 1e-2
                and abs(src_out.bounds.top - src_in.bounds.top) < 1e-2
            )
            if not bounds_match:
                logger.warning(
                    f"Bounds difference detected: in={src_in.bounds} vs out={src_out.bounds}"
                )

        # 6. Render RGB preview strictly from B04, B03, B02 (excluding B08 NIR)
        preview_img = self.render_rgb_preview(destination_tif, brightness=cfg.brightness)
        preview_png_path = out_dir / f"{input_path.stem}_sr_2.5m_preview.png"
        preview_img.save(preview_png_path, format="PNG")

        diagnostics = OpenSRDiagnostics(
            input_path=str(input_path),
            output_path=str(destination_tif),
            input_width=in_width,
            input_height=in_height,
            output_width=out_width,
            output_height=out_height,
            scale_factor=4.0,
            input_bands=in_bands,
            output_bands=out_bands,
            selected_bands=["B04", "B03", "B02", "B08"],
            execution_time_seconds=elapsed_time,
            execution_device=exec_device,
            sampling_steps=None,
            gpu_memory="Unavailable across subprocess boundary",
            output_crs=out_crs,
            output_pixel_size_m=out_pixel_size,
            bounds_preserved=bounds_match,
        )

        return OpenSRResult(
            sr_geotiff_path=destination_tif,
            sr_rgb_preview=preview_img,
            profile=profile,
            diagnostics=diagnostics,
            preview_png_path=preview_png_path,
        )
