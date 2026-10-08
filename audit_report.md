# Cloud Removal System — Audit Report

> **Status:** Read-only analysis. No code changes made.
> **Date:** 2026-09-27

---

## A. Current Architecture

The existing system is a **monolithic Streamlit application** (`app.py`) that wraps a
fully-implemented ECRformer inference pipeline.

### File inventory

| File | Role |
|---|---|
| [`app.py`](file:///c:/Users/datta/SYC_PROJECTS/temporary/app.py) | Single-file Streamlit UI + all inference logic (874 lines) |
| [`inference.py`](file:///c:/Users/datta/SYC_PROJECTS/temporary/inference.py) | Standalone inference module (tile loop, model loading, GeoTIFF saving) |
| [`preprocessing.py`](file:///c:/Users/datta/SYC_PROJECTS/temporary/preprocessing.py) | `normalize_s1()`, `normalize_s2()`, `inspect_tiff()`, `validate_sample()` |
| [`visualization.py`](file:///c:/Users/datta/SYC_PROJECTS/temporary/visualization.py) | `to_rgb()`, `to_pil()`, `sar_preview()` |
| [`download_one.py`](file:///c:/Users/datta/SYC_PROJECTS/temporary/download_one.py) | One-shot SEN12MS-CR dataset downloader (Hugging Face streaming) |
| [`tiff_viewer.py`](file:///c:/Users/datta/SYC_PROJECTS/temporary/tiff_viewer.py) | Standalone Tkinter desktop TIFF viewer |
| [`models/ecrformer_model.py`](file:///c:/Users/datta/SYC_PROJECTS/temporary/models/ecrformer_model.py) | Full ECRformer PyTorch architecture (428 lines) |
| [`models/module.py`](file:///c:/Users/datta/SYC_PROJECTS/temporary/models/module.py) | ECRformer sub-modules (blocks, attention, bottleneck) |
| [`models/module_util.py`](file:///c:/Users/datta/SYC_PROJECTS/temporary/models/module_util.py) | Layer utils (LayerNorm, split_integer) |
| [`config/base_config.py`](file:///c:/Users/datta/SYC_PROJECTS/temporary/config/base_config.py) | Base config (dataset=sen12mscr, in_chans=[2,13], out_chans=13) |
| [`config/ecrformer_config.py`](file:///c:/Users/datta/SYC_PROJECTS/temporary/config/ecrformer_config.py) | Full model config |
| [`config/ecrformer_light_config.py`](file:///c:/Users/datta/SYC_PROJECTS/temporary/config/ecrformer_light_config.py) | Lightweight variant config |
| [`model.ckpt`](file:///c:/Users/datta/SYC_PROJECTS/temporary/model.ckpt) | ~43 MB checkpoint (strict load verified) |
| [`sample/`](file:///c:/Users/datta/SYC_PROJECTS/temporary/sample/) | 5 SEN12MS-CR samples (sar.tif, cloudy.tif, target.tif) |
| [`requirements.txt`](file:///c:/Users/datta/SYC_PROJECTS/temporary/requirements.txt) | streamlit, numpy, rasterio, pillow, tqdm, einops, timm |

### Structural observation

There are **two parallel implementations** of the same inference logic:
- **`app.py`** — contains its own inline `normalize_sar()`, `normalize_optical()`, `run_inference()`, `predict_tile()`, `make_prediction_tiff()`, `make_png()`
- **`inference.py` + `preprocessing.py` + `visualization.py`** — a refactored module-based implementation with the same logic

The two are **not wired together**. `app.py` does not import `preprocessing.py` or `visualization.py`. `inference.py` imports `preprocessing.py` but is not used by `app.py`.

---

## B. Current Data Flow

```
User uploads .tif files via Streamlit file_uploader
        ↓
read_tiff() — rasterio.MemoryFile → np.float32 array + profile + metadata dict
        ↓
Validation — check band counts (S1 == 2, S2 == 13) and spatial dimensions match
        ↓
normalize_sar()  — clip [-25, 0] → (data + 25) / 25 → [0, 1]
normalize_optical() — clip [0, 10000] → data / 10000 → [0, 1]
        ↓
np.concatenate([sar_normalized, cloudy_normalized], axis=0)
→ model_input shape: (15, H, W)
        ↓
run_inference() → tiled loop → predict_tile() per tile
        ↓
prediction: (13, H, W) float32, clipped [0, 1]
        ↓
    ┌──────────────────────┐
    │                      │
optical_to_rgb(prediction) make_prediction_tiff(prediction, cloudy_profile)
→ (H, W, 3) uint8 PNG     → 13-band float32 GeoTIFF (deflate compressed)
```

### Sentinel-1/Sentinel-2 entry point

Currently the **only entry point for satellite data** is the Streamlit file uploader.
The user must manually locate and upload:
- A 2-band SAR GeoTIFF (from SEN12MS-CR samples via `download_one.py`)
- A 13-band cloudy optical GeoTIFF
- Optionally a 13-band clear optical GeoTIFF (for display comparison only — NOT fed to model)

No live satellite data acquisition exists anywhere in the current code.

### Rasterio read path

`read_tiff()` in `app.py` reads from `rasterio.MemoryFile` (uploaded bytes in RAM).
`inference.py`'s `run_sample()` uses file-path-based `rasterio.open()` with windowed tile reading — more efficient for large files.

---

## C. Current Model Flow

```
ECRformerModel(
    in_chans=[2, 13],     # S1 first, S2 second — decoupled stem
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
    decoupled_input=True,   # separate CNN stems for S1 and S2
    bottle_neck="tsa",      # Restormer-style transformer bottleneck
    num_refine=4,
    pos_encoding=None,
)
```

**Key architectural detail — `decoupled_input=True`:**
The `DecoupledEncoder.forward()` calls `torch.split(x, self.in_ch_list, dim=1)`:
- channels 0–1 → S1 branch (2-ch conv stem)
- channels 2–14 → S2 branch (13-ch conv stem)

This means the **channel ordering is structurally enforced**: S1 must always occupy channels 0–1 and S2 must occupy channels 2–14 of the concatenated input tensor.

**Model output:**
`forward()` returns `(x_temp, (down_projs, up_projs))`.
`app.py` correctly handles this: `if isinstance(output, (tuple, list)): prediction = output[0]`.

---

## D. Current CPU/GPU Flow

```python
# app.py line 19
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load_model() — checkpoint always loaded to CPU first:
checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
# Then model moved to DEVICE:
model = model.to(DEVICE)
model.eval()

# predict_tile() — input tensor moved to DEVICE:
tensor = torch.from_numpy(tile).unsqueeze(0).to(DEVICE)
# Output returned as CPU numpy:
return prediction[0].detach().float().cpu().numpy().astype(np.float32)
```

- Checkpoint always loads to CPU first (safe pattern regardless of GPU availability)
- `@st.cache_resource` ensures model is loaded once and reused across Streamlit reruns
- OOM is caught with `except torch.cuda.OutOfMemoryError`
- `inference.py` has an identical pattern via `get_device()` + `model.to(device)`

---

## E. Existing Reusable Components

The following are **directly reusable** without modification for Phase 1 and beyond:

| Component | Location | Reuse Notes |
|---|---|---|
| `normalize_s1()` | `preprocessing.py` | Correct SEN12MS-CR normalization for live data |
| `normalize_s2()` | `preprocessing.py` | Correct SEN12MS-CR normalization for live data |
| `inspect_tiff()` | `preprocessing.py` | Prints CRS, dims, dtype — needed for Phase 1 acceptance |
| `validate_sample()` | `preprocessing.py` | Band/dimension validation — extend for live data |
| `to_rgb()` | `visualization.py` | `bands=(3,2,1), brightness=3.0` — official ECRformer viz |
| `sar_preview()` | `visualization.py` | Mean of S1 bands → grayscale preview |
| `run_inference()` (app.py) | `app.py` | Tiled accumulation loop — keep as-is |
| `make_prediction_tiff()` | `app.py` | GeoTIFF writer with profile copy — keep as-is |
| `make_png()` | `app.py` | PNG encoder from RGB array |
| `load_model()` (app.py) | `app.py` | Strict checkpoint loading with good error messages |
| `ECRformerModel` | `models/` | The complete trained model — do NOT modify |
| `TIFFViewer` | `tiff_viewer.py` | Tkinter band viewer — useful for local inspection |
| Sample data | `sample/sample_0{1-5}/` | 5 SEN12MS-CR triplets for pipeline testing |

**Also reusable from `inference.py`:**
- `run_sample()` — file-path-based tiled inference with rasterio windowed reads (more efficient than app.py version for large AOIs)
- `save_geotiff()` — path-based GeoTIFF writer

---

## F. Missing Components for Phase 1A (AOI / Map Selection)

Phase 1A requires an interactive map where the user can:
- Click a point, draw a rectangle, or draw a polygon
- See coordinates, bounding box, and AOI area
- Set a date range
- Output a GeoJSON representation of the AOI

**Currently missing — nothing exists for:**

| Missing Component | Notes |
|---|---|
| Interactive map widget | No Leaflet, Folium, or deck.gl integration |
| AOI drawing tools | No rectangle/polygon draw capability |
| GeoJSON AOI representation | No GeoJSON production anywhere in codebase |
| Date range selector | Only file uploaders in current UI |
| AOI area calculation | No spatial math |
| Bounding box display | Nothing |
| Coordinate display | Nothing |

**Implementation candidates:**
- `streamlit-folium` — wraps Leaflet.js, supports draw plugins
- `folium` with `folium.plugins.Draw` — polygon/rectangle/point drawing
- `pydeck` or `streamlit-pydeck` — alternative if more control needed
- Output: GeoJSON `FeatureCollection` or simple `bbox` `[west, south, east, north]`

---

## G. Missing Components for Phase 1B (Scene Search)

Phase 1B requires the backend to:
- Receive AOI + date range
- Query Sentinel-2 L1C and Sentinel-1 GRD catalogues separately
- Return structured scene lists with metadata (cloud cover, polarization, footprint, etc.)
- Display candidate scenes for user selection

**Currently missing — nothing exists for:**

| Missing Component | Notes |
|---|---|
| CDSE / Sentinel Hub authentication | No OAuth2 token acquisition |
| Sentinel Hub Catalog API client | No HTTP requests to `sh.dataspace.copernicus.eu/catalog/v1/search` |
| S2 scene search | Not implemented |
| S1 scene search | Not implemented |
| Scene result display (Streamlit table) | Not implemented |
| Scene selection UI | Not implemented |
| STAC feature parsing | No GeoJSON footprint parsing |
| Cloud cover extraction from STAC | Not implemented |
| Polarization extraction from STAC | Not implemented |
| S1/S2 pairing logic | Not implemented |
| Download/retrieval via Processing API | Not implemented |
| Data cache (`data_cache/<hash>/`) | Not implemented |

---

## H. Proposed Files/Modules for Phase 1

Based on the plan's modular backend structure:

```
backend/
├── services/
│   └── satellite/
│       ├── base.py             # Abstract SatelliteProvider interface
│       └── cdse_sentinelhub.py # CDSE Sentinel Hub implementation
├── api/
│   └── routes_satellite.py    # /api/satellite/search  /api/satellite/acquire
└── auth/
    └── cdse_auth.py           # OAuth2 token management

acquisition/
├── poc_test.py                # First proof-of-concept script (no Streamlit)
└── data_cache/                # Request cache (hash-based)
```

**New Streamlit page (Phase 1A+1B UI):**
```
pages/
└── 01_Acquisition.py          # Map + date range + scene search results
```

This keeps the existing `app.py` (Phase 0 baseline) **completely untouched**.

---

## I. External API / Authentication Requirements

### Copernicus Data Space Ecosystem (CDSE)

| Requirement | Detail |
|---|---|
| Account | Free registration at `dataspace.copernicus.eu` |
| OAuth2 client | Must create OAuth client in account → yields `client_id` + `client_secret` |
| Token endpoint | `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token` |
| Token type | Bearer (short-lived, must be refreshed) |
| Catalog API URL | `https://sh.dataspace.copernicus.eu/catalog/v1/search` |
| Processing API URL | `https://sh.dataspace.copernicus.eu/api/v1/process` |
| Collections | `sentinel-2-l1c`, `sentinel-1-grd` |
| Python SDK | `sentinelhub` (optional but useful) |

**Credential management rule (from plan §14):**
- Store `client_id` and `client_secret` in **environment variables**, never hardcoded
- Never expose credentials to frontend
- Suggested env vars: `CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET`

**Alternative: `sentinelhub-py` library**
The official `sentinelhub` Python package wraps OAuth2 + Catalog + Process APIs and is
available on PyPI. It would significantly simplify implementation and is the recommended
approach for the PoC.

```
pip install sentinelhub
```

New `requirements.txt` additions needed:
```
sentinelhub>=3.10
shapely>=2.0
geojson>=3.0
requests>=2.31
```

---

## J. Scientific Compatibility Risks

This is the most critical section.

### J1. Sentinel-2 L1C — Band Order and Product Level

**Confirmed by SEN12MS-CR research:**
The training dataset uses **Sentinel-2 L1C Top-of-Atmosphere (TOA) reflectance** with all 13 bands in canonical order:

```
Index  Band   Description         Native Resolution
  0    B01    Coastal aerosol     60 m
  1    B02    Blue                10 m
  2    B03    Green               10 m
  3    B04    Red                 10 m
  4    B05    Vegetation red edge 20 m
  5    B06    Vegetation red edge 20 m
  6    B07    Vegetation red edge 20 m
  7    B08    NIR                 10 m
  8    B08A   Narrow NIR          20 m
  9    B09    Water vapour        60 m
 10    B10    SWIR – Cirrus       60 m
 11    B11    SWIR                20 m
 12    B12    SWIR                20 m
```

**In the SEN12MS-CR dataset, all bands were resampled to a common 10 m grid.**

**Risks:**
1. **L2A vs L1C:** Sentinel-2 L2A removes B10 (cirrus), has only 12 bands, and uses Bottom-of-Atmosphere (BOA) reflectance. **L2A is NOT compatible with this checkpoint.**
2. **Band ordering from Sentinel Hub:** If the Processing API evalscript does not request bands in exactly the above order, the channel mapping into the model will be wrong. This will not raise an error — it will silently produce incorrect output.
3. **Radiometric units from Sentinel Hub:** The Processing API returns DN values (UINT16, 0–10000 range for L1C TOA reflectance). The current `normalize_s2()` clips to 0–10000 and divides by 10000. This is the correct normalization **only if** the retrieved data is in these DN units. Some Sentinel Hub evalscripts return reflectance (0.0–1.0 float) directly — which would make normalization wrong.
4. **Resampling:** SEN12MS-CR resampled all bands to 10m. Sentinel Hub can be asked to resample to a specified resolution, but if mixed-resolution bands are returned without resampling, the model will receive spectrally incorrect data at incorrect spatial scales.
5. **Spatial resolution request:** Must explicitly set `resolution` or `resx/resy` in the Processing API request to 10m for all 13 bands.

### J2. Sentinel-1 GRD — Preprocessing Chain

**Confirmed by SEN12MS-CR research:**
The SAR data in SEN12MS-CR is derived from Sentinel-1 GRD, VV and VH polarization, in that order:
```
Index  Channel  Description
  0    VV       Vertical-Vertical backscatter
  1    VH       Vertical-Horizontal backscatter
```

The values are in **dB** and normalized using clip [-25, 0] → divide by 25 + shift.

**Important nuance found during research:**
- Some sources indicate the VH channel may use a different clip range (e.g., [-32.5, 0]) in some SEN12MS-CR implementations.
- The current code applies the **same** `clip(-25, 0)` normalization to **both** VV and VH.
- This requires experimental verification against the actual checkpoint training normalization.

**Risks:**
1. **Sentinel Hub GRD output units:** The Processing API for S1 GRD can return linear backscatter (Sigma0), log-scale backscatter (dB), or gamma0. The current normalization assumes dB values in [-25, 0]. If the API returns linear values (typically 0.0–1.0 or 0–65535), normalization will be completely wrong.
2. **Terrain correction:** SEN12MS-CR used Range-Doppler terrain correction. Sentinel Hub's S1 GRD product has terrain correction options (IW, GRD orthorectification). Must match.
3. **Speckle filtering:** SEN12MS-CR data was not speckle-filtered (raw GRD). If the API applies speckle filtering, the domain changes.
4. **Orbit direction:** Sentinel-1 has ascending and descending passes with different look angles. SEN12MS-CR does not restrict orbit direction. This may affect SAR backscatter values.
5. **Polarization availability:** S1 IW GRD over land should provide VV+VH. But EW mode (used over ocean/polar) has HH+HV. Must filter for IW GRD with VV+VH.

### J3. S1/S2 Spatial Alignment

- SEN12MS-CR patches are already co-registered. Live data will NOT be.
- S1 and S2 have different native CRS, resolution, and grid origins.
- A resampling/reprojection step is required before concatenation.
- The model has **no internal alignment mechanism** — spatial misalignment will corrupt predictions.

### J4. Temporal Pairing

- SEN12MS-CR has S1 and S2 acquisitions paired within the same season (max ~days apart).
- Live acquisitions may be weeks apart (S1 revisit ≈ 6–12 days, S2 ≈ 5 days).
- Large temporal gaps may increase domain shift due to phenological/seasonal changes.

### J5. Domain Shift Summary

> **The most important scientific unknown:** The ECRformer checkpoint was trained on curated SEN12MS-CR patches. Even with perfect preprocessing, generalization to arbitrary live Sentinel observations is not guaranteed. This must be measured experimentally.

---

## K. Recommended First Proof-of-Concept Test

Per the plan (§15), before any UI or backend is built:

**Script:** `acquisition/poc_test.py`

```
Step 1 — Pick a known test AOI
         Use a small 0.1° × 0.1° bounding box over a known area
         (e.g., a flat agricultural region with known cloud cover)

Step 2 — Authenticate with CDSE
         Load CDSE_CLIENT_ID + CDSE_CLIENT_SECRET from environment
         Acquire OAuth2 Bearer token

Step 3 — Search S2 L1C catalogue
         POST to https://sh.dataspace.copernicus.eu/catalog/v1/search
         collection = sentinel-2-l1c
         Print: scene IDs, datetimes, cloud cover, footprints

Step 4 — Search S1 GRD catalogue
         Same endpoint, collection = sentinel-1-grd
         Print: scene IDs, datetimes, polarization, orbit direction

Step 5 — Select one S2 + one S1 scene (manual, not auto)

Step 6 — Download S2 via Processing API
         Request ALL 13 bands (B01–B12 + B08A)
         Request explicit band order matching SEN12MS-CR
         Request resolution = 10m
         Request DN units (UINT16, 0–10000)
         Save as s2.tif

Step 7 — Download S1 via Processing API
         Request VV + VH bands
         Request dB units
         Apply terrain correction
         Save as s1.tif

Step 8 — Open both TIFFs with rasterio
         Print: CRS, transform, width, height, bands, dtype, resolution
         Compare against SEN12MS-CR samples in sample/sample_01/
         Verify band counts, value ranges, and spatial parameters

Step 9 — Display previews
         S2 RGB (bands 3, 2, 1)
         S1 grayscale (mean of VV+VH)

Step 10 — Run existing normalize_s1() / normalize_s2() from preprocessing.py
          Check output value ranges
          Flag any anomalies
```

**Success criteria before proceeding to Phase 1 UI:**
- Real S2 TIFF received with exactly 13 bands in correct order
- Real S1 TIFF received with exactly 2 bands (VV, VH) in dB scale
- CRS and transform are valid and geospatially correct
- Value ranges after normalization match SEN12MS-CR samples
- AOI matches the requested bounding box

---

## Summary Table

| Topic | Status |
|---|---|
| ECRformer inference | ✅ Working, do not touch |
| Tiled inference | ✅ Working |
| GeoTIFF output | ✅ Working |
| PNG preview | ✅ Working |
| SEN12MS-CR normalization | ✅ Implemented (preprocessing.py) |
| Checkpoint loading | ✅ Strict load verified |
| CPU/GPU selection | ✅ Automatic |
| TIFF viewer (Tkinter) | ✅ Working standalone tool |
| Phase 1A — AOI map selection | ❌ Does not exist |
| Phase 1B — S2 catalogue search | ❌ Does not exist |
| Phase 1B — S1 catalogue search | ❌ Does not exist |
| Phase 1B — Scene display/selection | ❌ Does not exist |
| CDSE authentication | ❌ Does not exist |
| Phase 1D — S2 retrieval | ❌ Does not exist |
| Phase 1E — S1 retrieval | ❌ Does not exist |
| Phase 1F — S1/S2 pairing | ❌ Does not exist |
| Phase 1G — data cache | ❌ Does not exist |
| Band compatibility verified | ⚠️ Not yet tested with live data |
| S1 dB range verified | ⚠️ Potential mismatch (VH range) |
| Spatial alignment (live S1+S2) | ⚠️ Will need reprojection |
| Domain generalization | ⚠️ Unknown — must experiment |
