"""Run LDSR-S2 on one Sentinel-2 RGBN GeoTIFF.

Drag a TIFF onto ``Run Super Resolution.bat`` or run this file with the TIFF
path as its only argument. The model requires real B4, B3, B2, and B8 data.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import tempfile
from pathlib import Path

import rasterio
import torch

import opensr_utils
from opensr_utils.model_utils.get_models import get_ldsrs2

# opensr-utils writes Unicode status markers; Windows consoles may otherwise use cp1252.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
REQUIRED_BANDS = ("B4", "B3", "B2", "B8")


def choose_input() -> Path | None:
    """Show a file picker when the launcher is opened without a dropped file."""
    try:
        from tkinter import Tk, filedialog

        window = Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        selected = filedialog.askopenfilename(
            title="Choose a Sentinel-2 RGBN GeoTIFF",
            filetypes=[("GeoTIFF", "*.tif *.tiff"), ("All files", "*.*")],
        )
        window.destroy()
        return Path(selected) if selected else None
    except Exception as error:
        raise RuntimeError("Choose a TIFF by dragging it onto Run Super Resolution.bat.") from error


def normalize_band_name(name: str | None) -> str | None:
    if not name:
        return None
    normalized = name.strip().upper().replace("_", "").replace(" ", "")
    match = re.fullmatch(r"B0*(\d+)", normalized)
    return f"B{int(match.group(1))}" if match else normalized


def select_band_indexes(dataset: rasterio.DatasetReader) -> tuple[int, ...]:
    names = [normalize_band_name(name) for name in dataset.descriptions]
    named_indexes = {name: index + 1 for index, name in enumerate(names) if name}

    if all(band in named_indexes for band in REQUIRED_BANDS):
        return tuple(named_indexes[band] for band in REQUIRED_BANDS)

    if dataset.count == 4 and not any(names):
        # Four unnamed bands are accepted only in the standard RGBN order.
        return (1, 2, 3, 4)

    if dataset.count == 13 and not any(names):
        # Standard Sentinel-2 order: B1, B2, B3, B4, B5, B6, B7, B8, ... B12.
        return (4, 3, 2, 8)

    found = ", ".join(name or "unnamed" for name in names)
    raise ValueError(
        "This model needs real Sentinel-2 B4, B3, B2, and B8 bands. "
        f"This file has {dataset.count} band(s): {found or 'unnamed'}. "
        "Band-labelled TIFFs are accepted in any order, as are unnamed standard 13-band "
        "Sentinel-2 TIFFs. Export/download it with B8 (NIR) included; RGB-only TIFFs "
        "cannot be converted reliably."
    )


def prepare_rgbn(source: Path, workspace: Path) -> Path:
    staged = workspace / "input_rgbn.tif"
    with rasterio.open(source) as dataset:
        indexes = select_band_indexes(dataset)
        profile = dataset.profile.copy()
        profile.update(count=4, compress="deflate", predictor=2)

        # OpenSR's Sentinel-2 model expects the common 0-10000 reflectance scale.
        selected_data = [dataset.read(index) for index in indexes]
        minimum = min(float(data.min()) for data in selected_data)
        maximum = max(float(data.max()) for data in selected_data)
        scale = 10000.0 if 0.0 <= minimum and maximum <= 1.5 else 1.0
        if scale != 1.0:
            profile.update(dtype="float32")

        with rasterio.open(staged, "w", **profile) as output:
            for destination_index, data in enumerate(selected_data, start=1):
                output.write(data.astype("float32") * scale, destination_index)
                output.set_band_description(destination_index, REQUIRED_BANDS[destination_index - 1])
        if scale != 1.0:
            print("Input uses 0-1 reflectance; converted it to the 0-10000 Sentinel-2 scale.")
    return staged


def label_output(path: Path) -> None:
    with rasterio.open(path, "r+") as dataset:
        for index, band_name in enumerate(REQUIRED_BANDS, start=1):
            dataset.set_band_description(index, band_name)


def run(source: Path) -> Path:
    source = source.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Choose an existing .tif or .tiff file.")

    OUTPUTS.mkdir(exist_ok=True)
    destination = OUTPUTS / f"{source.stem}_sr.tif"
    workspace = Path(tempfile.mkdtemp(prefix="opensr_", dir=ROOT))

    try:
        prepared = prepare_rgbn(source, workspace)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {device.upper()}. Preparing {source.name}...")
        model = get_ldsrs2(device=device)
        job = opensr_utils.large_file_processing(
            root=str(prepared),
            model=model,
            window_size=(128, 128),
            factor=4,
            overlap=12,
            eliminate_border_px=2,
            device=device,
            gpus=0 if device == "cuda" else None,
            batch_size=1,
            save_preview=False,
            cleanup=True,
            overwrite=True,
        )
        job.run()
        generated = workspace / "sr.tif"
        if not generated.is_file():
            raise RuntimeError("The model completed without creating an output TIFF.")
        if destination.exists():
            destination.unlink()
        shutil.move(str(generated), str(destination))
        label_output(destination)
        return destination
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Super-resolve one Sentinel-2 RGBN GeoTIFF.")
    parser.add_argument("input", nargs="?", help="TIFF to process")
    arguments = parser.parse_args()
    source = Path(arguments.input) if arguments.input else choose_input()
    if source is None:
        print("No file selected.")
        return 0

    try:
        result = run(source)
    except Exception as error:
        print(f"\nCould not process the TIFF:\n{error}", file=sys.stderr)
        return 1

    print(f"\nDone. Your super-resolved TIFF is here:\n{result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
