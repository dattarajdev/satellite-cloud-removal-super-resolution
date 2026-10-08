# Phase 2 — Satellite Harmonization

## Overview

The Satellite Harmonization pipeline bridges raw heterogeneous acquisitions (Sentinel-1 SAR and Sentinel-2 Multi-Spectral) and model-ready tensor inputs. It transforms independently acquired observations into a strictly co-registered, geometrically aligned pair on a common Universal Transverse Mercator (UTM) spatial grid at an explicit **10.0-meter** resolution.

```text
Raw Provider GeoTIFFs (EPSG:4326)
├── S2 L1C (13 bands, 256x256, uint16)
└── S1 GRD ( 2 bands, 256x256, float32)
                     ↓
       [Phase 2: Harmonization]
       - Determine UTM Zone (EPSG:32633)
       - Derive 10m Snapped Common Grid
       - Bilinear Resampling to UTM
       - Preserve Band Sequences & Dtypes
       - Construct Pixel Validity Mask
                     ↓
Harmonized Outputs (EPSG:32633, 10m)
├── s2_harmonized_10m.tif  (13 bands, 200x262, uint16)
├── s1_harmonized_10m.tif  ( 2 bands, 200x262, float32)
├── validity_mask_10m.tif  ( 1 band,  200x262, uint8)
└── Diagnostics & Visual Previews
```

---

## Technical Specifications & Requirements Met

| # | Requirement | Implementation Details | Validation Status |
|---|---|---|---|
| **1** | **UTM CRS Determination** | Auto-derived from center coordinates: `zone = floor((lon + 180) / 6) + 1`; Northern Hemisphere $\rightarrow$ `EPSG:32600 + zone`. For Rome AOI $\rightarrow$ `EPSG:32633` | **PASS** |
| **2** | **Identical UTM Projection** | Both S1 and S2 reprojected to `EPSG:32633` | **PASS** |
| **3** | **Common Output Grid** | Both images share exact pixel dimensions (`200 x 262`) and identical affine transform | **PASS** |
| **4** | **Explicit 10 m Resolution** | Transform pixel scales: $\Delta x = 10.0\text{ m}, \Delta y = 10.0\text{ m}$ | **PASS** |
| **5** | **Identical AOI Extent** | Exact bounding box alignment: Left=288470.0, Bottom=4641760.0, Right=290470.0, Top=4644380.0 | **PASS** |
| **6** | **Band Sequence Preservation** | **S2 (13 bands)**: `B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12`<br>**S1 (2 bands)**: `VV, VH` | **PASS** |
| **7** | **Spatial Resampling** | Explicit bilinear interpolation (`Resampling.bilinear`) for continuous optical reflectance and SAR dB backscatter | **PASS** |
| **8** | **Scientific Dtypes** | S2 stored as `uint16` (native DN); S1 stored as `float32` (calibrated dB) | **PASS** |
| **9** | **NoData Handling** | S2 NoData = `0`; S1 NoData = `-9999.0`. Binary validity mask generated | **PASS** |
| **10** | **Validation Diagnostics** | Automated JSON & Markdown reporting, per-band statistics, and coregistration visual diagnostics | **PASS** |

---

## File Structure

```text
harmonization/
├── run_harmonization.py       # CLI runner for harmonization pipeline
├── README.md                  # This specification and guide
├── __init__.py                # Package exports
└── output/                    # Production outputs for the pair
    ├── s2_harmonized_10m.tif            # 13-band harmonized GeoTIFF
    ├── s1_harmonized_10m.tif            # 2-band harmonized GeoTIFF
    ├── validity_mask_10m.tif            # 1-band uint8 validity mask
    ├── s2_rgb_preview.png               # True-color RGB preview
    ├── s1_sar_preview.png               # Dual-pol false color preview
    ├── coregistration_diff_preview.png  # Spatial alignment edge overlay
    ├── harmonization_report.json        # Machine-readable validation diagnostics
    └── harmonization_report.md          # Human-readable validation report

backend/services/processing/
├── __init__.py                # Module entry point
├── grid.py                    # Deterministic UTM grid and CRS calculation
└── harmonizer.py              # Reprojection, metadata preservation, and validation
```

---

## Usage

### 1. Auto-Harmonize Latest Cached Pair

```bash
venv\Scripts\python harmonization/run_harmonization.py
```

### 2. Custom Inputs and Options

```bash
venv\Scripts\python harmonization/run_harmonization.py \
  --s2 path/to/s2.tif \
  --s1 path/to/s1.tif \
  --output-dir harmonization/output \
  --pixel-size 10.0 \
  --resampling bilinear
```

### 3. Python API Integration

```python
from backend.services.processing import harmonize_pair, HarmonizationConfig

pair = harmonize_pair(
    s2_input_path="path/to/s2.tif",
    s1_input_path="path/to/s1.tif",
    output_dir="harmonization/output",
    config=HarmonizationConfig(pixel_size=10.0),
)

print("Validation passed:", pair.validation_passed)
print("Harmonized grid:", pair.grid.to_dict())
```
