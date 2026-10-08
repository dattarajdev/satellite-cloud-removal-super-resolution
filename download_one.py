from datasets import load_dataset
import numpy as np
import rasterio
from rasterio.transform import from_origin
import os


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "Hermanni/sen12mscr"

OUTPUT_DIR = "sample"

NUMBER_OF_SAMPLES = 5


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("Loading SEN12MS-CR dataset")
print("=" * 60)

print("\nDataset:")
print(DATASET_NAME)

print("\nStreaming mode enabled...")

ds = load_dataset(
    DATASET_NAME,
    split="train",
    streaming=True
)


# ============================================================
# TIFF SAVER
# ============================================================

def save_multiband_tiff(
    path,
    data
):

    data = np.asarray(
        data
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Handle channel layout
    # --------------------------------------------------------

    if data.ndim != 3:

        raise ValueError(
            f"Expected 3D array, got {data.shape}"
        )

    # If HWC:
    #
    # (256, 256, 13)
    #
    # convert to:
    #
    # (13, 256, 256)

    if (
        data.shape[0] != 1
        and data.shape[0] != 2
        and data.shape[0] != 13
        and data.shape[0] != 15
    ):

        data = np.transpose(
            data,
            (2, 0, 1)
        )

    # --------------------------------------------------------
    # Make sure C,H,W
    # --------------------------------------------------------

    channels, height, width = (
        data.shape
    )

    print(
        f"    Saving {channels} bands "
        f"({height} x {width})"
    )

    # --------------------------------------------------------
    # TIFF profile
    # --------------------------------------------------------

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": channels,
        "height": height,
        "width": width,
        "compress": "deflate",
    }

    # --------------------------------------------------------
    # Write GeoTIFF
    # --------------------------------------------------------

    with rasterio.open(
        path,
        "w",
        **profile
    ) as dst:

        dst.write(
            data
        )

    print(
        f"    ✓ {path}"
    )


# ============================================================
# DOWNLOAD 5 SAMPLES
# ============================================================

iterator = iter(
    ds
)

for index in range(
    NUMBER_OF_SAMPLES
):

    print("\n")
    print("=" * 60)
    print(
        f"SAMPLE {index + 1}/{NUMBER_OF_SAMPLES}"
    )
    print("=" * 60)

    try:

        sample = next(
            iterator
        )

    except StopIteration:

        print(
            "Dataset ended."
        )

        break


    # ========================================================
    # DECODE SAR
    # ========================================================

    sar = np.frombuffer(
        sample["sar"],
        dtype=np.float32
    ).reshape(
        sample["sar_shape"]
    )


    # ========================================================
    # DECODE CLOUDY OPTICAL
    # ========================================================

    cloudy = np.frombuffer(
        sample["cloudy"],
        dtype=np.int16
    ).reshape(
        sample["opt_shape"]
    )


    # ========================================================
    # DECODE TARGET OPTICAL
    # ========================================================

    target = np.frombuffer(
        sample["target"],
        dtype=np.int16
    ).reshape(
        sample["opt_shape"]
    )


    # ========================================================
    # PRINT ORIGINAL SHAPES
    # ========================================================

    print("\nOriginal shapes:")

    print(
        "SAR:    ",
        sar.shape
    )

    print(
        "Cloudy: ",
        cloudy.shape
    )

    print(
        "Target: ",
        target.shape
    )


    # ========================================================
    # SAMPLE FOLDER
    # ========================================================

    sample_dir = os.path.join(
        OUTPUT_DIR,
        f"sample_{index + 1:02d}"
    )

    os.makedirs(
        sample_dir,
        exist_ok=True
    )


    # ========================================================
    # SAVE SAR
    # ========================================================

    sar_path = os.path.join(
        sample_dir,
        "sar.tif"
    )

    save_multiband_tiff(
        sar_path,
        sar
    )


    # ========================================================
    # SAVE CLOUDY
    # ========================================================

    cloudy_path = os.path.join(
        sample_dir,
        "cloudy.tif"
    )

    save_multiband_tiff(
        cloudy_path,
        cloudy
    )


    # ========================================================
    # SAVE TARGET
    # ========================================================

    target_path = os.path.join(
        sample_dir,
        "target.tif"
    )

    save_multiband_tiff(
        target_path,
        target
    )


    # ========================================================
    # METADATA
    # ========================================================

    print("\nMetadata:")

    print(
        "Season :",
        sample.get("season")
    )

    print(
        "Scene  :",
        sample.get("scene")
    )

    print(
        "Patch  :",
        sample.get("patch")
    )


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print(
    f"\nDownloaded {NUMBER_OF_SAMPLES} examples."
)

print(
    f"\nLocation: {OUTPUT_DIR}/"
)

print(
    """
Each sample contains:

sample_01/
    sar.tif       -> 2 bands
    cloudy.tif    -> 13 bands
    target.tif    -> 13 bands

sample_02/
    sar.tif
    cloudy.tif
    target.tif

...

sample_05/
    sar.tif
    cloudy.tif
    target.tif
"""
)

