# Cloud Removal System — Phase Build & Research Plan

> **Status:** Working brainstorm / implementation plan.  
> **Important:** This document intentionally does **not** finalize the research architecture. ECRformer remains the current baseline. Model orchestration is postponed until the end-to-end data pipeline is working and real failure cases are measured.

---

## 1. Current Baseline

The existing Streamlit prototype accepts:

- Sentinel-1 SAR GeoTIFF: 2 bands
- Cloudy Sentinel-2 GeoTIFF: 13 bands
- Optional cloud-free Sentinel-2 GeoTIFF for comparison

The current application validates those shapes, normalizes the data, concatenates the 2 SAR + 13 optical bands into a 15-channel model input, runs the full ECRformer, and produces a 13-band prediction. The target image is optional and is not sent to the model; it is only used for comparison/evaluation. fileciteturn13file0L533-L541

The current normalization is:

- Sentinel-2: clip to 0–10000, divide by 10000
- Sentinel-1: clip to -25..0 dB and map to 0..1

These are part of the current prototype and must remain explicit, testable preprocessing stages. fileciteturn13file2L5-L35

The current inference is tiled and uses overlapping tiles with accumulation/averaging. fileciteturn13file0L347-L365

### Output distinction

The system already creates two different output products:

1. **13-band float32 GeoTIFF** — scientific/geospatial output.
2. **RGB PNG** — visualization output.

The GeoTIFF writer keeps the source raster profile and writes 13 float32 bands. fileciteturn13file0L478-L499 The PNG is generated separately from the RGB representation. fileciteturn13file0L502-L508

---

# 2. Product Goal

The user should not have to manually find and upload Sentinel TIFF files.

Target experience:

```text
User selects location / AOI
        ↓
Select date / date range
        ↓
System searches satellite catalogue
        ↓
Find Sentinel-2 optical observation
+
Find Sentinel-1 SAR observation
        ↓
Retrieve only the required AOI
        ↓
Validate / align / preprocess
        ↓
Cloud analysis
        ↓
ECRformer reconstruction
        ↓
13-band cloud-free GeoTIFF
+
RGB preview
        ↓
Generation/progress animation
        ↓
User views / downloads result
```

We should **not build a global satellite archive**. Instead, build a thin acquisition layer around existing EO APIs.

---

# 3. Phase 1 — Satellite Data Acquisition

## Goal

Prove:

```text
AOI + date range
        ↓
Sentinel-2 scene(s)
+
Sentinel-1 scene(s)
        ↓
small AOI raster outputs
+
metadata
```

No ECRformer is required for the first milestone.

## Preferred provider for the first prototype

### Copernicus Data Space Ecosystem + Sentinel Hub

Current official documentation:

- CDSE APIs: https://documentation.dataspace.copernicus.eu/APIs.html
- Sentinel Hub overview: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview.html
- Process API: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Process.html
- Catalog API: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Catalog.html
- CDSE STAC: https://documentation.dataspace.copernicus.eu/APIs/STAC.html
- Sentinel-2 L1C: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L1C.html
- Sentinel-2 L2A: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html
- Sentinel-1 GRD: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S1GRD.html

CDSE documents catalogue APIs for product discovery and Sentinel Hub's Process API for requesting imagery over an AOI/time period. The Processing API hides tile-boundary complexity and can stitch data for the requested area. citeturn790907search1turn790907search2

The Catalog API implements STAC search, which is useful for finding observations by geometry and datetime. citeturn790907search7turn790907search8

### Alternatives to keep possible

Do not implement yet, but keep the provider layer replaceable:

- Microsoft Planetary Computer
- Google Earth Engine
- other STAC-compatible EO providers

---

## Phase 1A — AOI / Map Selection

Frontend:

```text
Map
 ┌─────────────────────────────┐
 │                             │
 │       ┌─────────────┐       │
 │       │     AOI     │       │
 │       └─────────────┘       │
 │                             │
 └─────────────────────────────┘

Date start: [ YYYY-MM-DD ]
Date end:   [ YYYY-MM-DD ]

[ Search observations ]
```

Allow:

- click a point
- draw rectangle
- draw polygon
- show coordinates
- show bounding box
- show AOI area

Internal representation should be GeoJSON.

Do not tie the backend to a specific map library.

---

## Phase 1B — Scene Search

Backend receives:

```json
{
  "aoi": {...},
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD"
}
```

Search separately for:

### Sentinel-2

Return at least:

- scene/product ID
- acquisition datetime
- cloud metadata where available
- footprint
- processing level
- coverage/availability

### Sentinel-1

Return at least:

- scene/product ID
- acquisition datetime
- product type
- polarization
- footprint
- orbit information where available

Do not silently select the first search result.

---

## Phase 1C — Show Scene Choices

Example UI:

```text
Sentinel-2
--------------------------------------------------------
Date          Cloud      Coverage       Select
2026-09-20    18%        AOI covered    ○
2026-09-23    41%        AOI covered    ○
2026-09-25    63%        AOI covered    ○
```

```text
Sentinel-1
--------------------------------------------------------
Date          Polarization     Coverage    Select
2026-09-18    VV/VH             covered     ○
2026-09-22    VV/VH             covered     ○
2026-09-24    VV/VH             covered     ○
```

Later we can add automatic selection.

---

## Phase 1D — Sentinel-2 Retrieval

The first experiment should request a **small AOI**, not a full satellite tile.

Sentinel Hub's Processing API supports tailored requests and can return selected bands for a specified area. citeturn790907search2turn790907search6

### Critical compatibility test

The current ECRformer expects 13 optical channels. We must verify:

- exact band list/order expected by the checkpoint
- product level
- radiometric units/scaling
- spatial resolution
- resampling
- nodata handling
- georeferencing

Sentinel-2 L1C exposes B01–B12 including B10, with mixed native resolutions. citeturn790907search11

Sentinel-2 L2A has a different band situation; therefore **do not assume L2A is automatically compatible with this 13-channel checkpoint**. The provider/product choice is an experiment, not a settled decision. citeturn790907search0

---

## Phase 1E — Sentinel-1 Retrieval

The current model expects 2 SAR channels.

Initial hypothesis:

```text
VV
VH
```

Sentinel-1 GRD documentation supports VV/VH for dual-polarization products and documents available backscatter representations and processing options. citeturn790907search13

We must verify:

- exact polarization
- backscatter coefficient
- units
- terrain correction / orthorectification choice
- speckle handling
- spatial grid
- conversion into the dB-like range expected by the current preprocessing

The current preprocessing clips SAR to -25..0 dB and maps it to 0..1. fileciteturn13file2L40-L72

---

## Phase 1F — S1/S2 Pairing

Do not simply use “latest S1 + latest S2”.

Pair observations using:

1. same/overlapping AOI
2. temporal proximity
3. sufficient spatial coverage
4. required bands/polarization
5. acquisition/data-quality constraints

Record the selected pair:

```json
{
  "sentinel2": {
    "id": "...",
    "datetime": "...",
    "cloud_cover": 0.18
  },
  "sentinel1": {
    "id": "...",
    "datetime": "...",
    "polarization": ["VV", "VH"]
  },
  "time_difference_hours": 36.0
}
```

Do not hide this metadata.

---

## Phase 1G — Download/Processing Cache

Avoid repeatedly downloading the same request.

Suggested structure:

```text
data_cache/
    <request_hash>/
        s2.tif
        s1.tif
        metadata.json
        cloud_data.tif
```

Hash should account for:

- provider
- collection
- AOI
- acquisition
- bands
- processing parameters

---

## Phase 1 Acceptance Criteria

Phase 1 is complete when the application can:

- select AOI
- search S2
- search S1
- show candidate scenes
- select a pair
- retrieve a small AOI
- save S1 and S2 TIFFs
- save metadata
- open TIFFs with Rasterio
- show S2 RGB
- show S1 preview
- print CRS, transform, width, height, bands, dtype and resolution

At this point:

**REAL SATELLITE DATA ACQUISITION = PROVEN**

---

# 4. Phase 2 — Geo / Model Preprocessing

## Goal

Transform live provider imagery into:

```text
S1 tensor = [2,H,W]
S2 tensor = [13,H,W]
```

with correct alignment and model normalization.

### Pipeline

```text
Provider imagery
      ↓
Inspect metadata
      ↓
Spatial harmonization
      ↓
Band mapping
      ↓
Radiometric conversion
      ↓
Normalization
      ↓
NoData / validity mask
      ↓
Model-ready tensors
```

## Metadata to preserve

- CRS
- affine transform
- width
- height
- pixel size
- nodata
- dtype
- acquisition datetime
- footprint

## Spatial alignment

S1 and S2 can have different:

- resolutions
- grids
- extents
- transforms
- dimensions

Choose a common target grid after testing. Do not blindly resize images.

Preserve geospatial meaning:

```text
CRS
+
transform
+
extent
+
pixel alignment
```

## Explicit S2 band mapping

Create configuration, not scattered hard-coded indices:

```yaml
sentinel2:
  model_band_order:
    - B01
    - B02
    - B03
    - ...
```

The exact order must be verified against the ECRformer/SEN12MS-CR loader.

## Explicit S1 mapping

```yaml
sentinel1:
  model_band_order:
    - VV
    - VH
```

Again, this is a hypothesis until experimentally validated.

## Validity / NoData

Maintain:

```text
data
+
validity mask
```

Do not indiscriminately turn invalid pixels into zero without preserving their provenance.

## Phase 2 acceptance

For one real acquired pair:

```text
S1:
  2 bands
  float32
  normalized

S2:
  13 bands
  float32
  normalized

metadata:
  CRS
  transform
  resolution
  datetime
```

Create a preprocessing verification page/script before connecting ECRformer.

---

# 5. Phase 3 — Cloud Intelligence / Cloud Mock Generator

## Goal

Let the user understand where clouds are and what part of the image the reconstruction is expected to address.

### First version: do NOT invent fake clouds

Prefer real cloud-related Sentinel information where available.

Concept:

```text
Sentinel-2
    ↓
cloud probability / cloud classification
    ↓
cloud mask
    ↓
overlay
    ↓
cloud statistics
```

Sentinel-2 L2A documentation provides cloud-related layers such as SCL and cloud probability information through the processing ecosystem. citeturn790907search0

### UI modes

```text
A. Original RGB
B. Cloud overlay
C. Cloud mask
D. Split comparison
E. Cloud statistics
```

Example:

```text
Cloud coverage:          42.8%
High-confidence cloud:   31.5%
Cloud shadow:             6.2%
Clear area:              57.2%
```

### Threshold interaction

Where the source cloud-probability product supports it:

```text
Cloud threshold
0% ───────────●──────── 100%
              65%
```

Updating the threshold updates:

- mask
- overlay
- cloud percentage

and should not rerun ECRformer.

### Important model distinction

The current checkpoint does not take a cloud mask as an input channel. The current inference concatenates only S1 + cloudy S2. fileciteturn13file1L317-L352

Therefore initially:

```text
Cloud mask
   ↓
visualization / analysis
```

not:

```text
Cloud mask → ECRformer input
```

Any future cloud-mask conditioning must be a separate research experiment.

## Phase 3 acceptance

Given an acquired S2 scene, the app can:

- show RGB
- show cloud mask
- show cloud overlay
- calculate cloud percentage
- adjust threshold
- update mask without model inference

---

# 6. Phase 4 — Live ECRformer Reconstruction

Only after Phases 1–3 are proven.

```text
real S1
+
real S2
+
validated preprocessing
        ↓
ECRformer
        ↓
13-band reconstruction
```

Do not modify ECRformer in this phase.

The purpose is to test:

> Can the existing checkpoint generalize from the SEN12MS-CR representation to live Sentinel observations after correct harmonization?

This is a major experiment.

### Tiled inference

Keep the existing tiled inference strategy. The current implementation splits the image into overlapping tiles, processes them, then averages overlapping predictions. fileciteturn13file1L430-L580

This supports larger AOIs without requiring the entire scene to fit into GPU memory.

---

# 7. Phase 5 — Scientific + Visual Outputs

Always produce both:

## Scientific product

```text
cloud_free_prediction.tif
```

- 13 bands
- float32
- georeferenced
- correct CRS
- correct transform
- correct AOI
- compressed

## Visualization product

```text
cloud_free_preview.png
```

The PNG is only a human-facing visualization.

Do not present the PNG as the complete scientific output.

---

# 8. Phase 6 — Generation / Reconstruction Experience

The UI should communicate what the system is actually doing without pretending to expose hidden neural-network “thought”.

Suggested stages:

```text
✓ Observation located
✓ Sentinel-2 retrieved
✓ Sentinel-1 retrieved
✓ Observations aligned
✓ Cloud region analyzed
✓ Model tiles prepared
◉ Reconstructing tiles
○ Merging overlapping predictions
○ Creating GeoTIFF
○ Creating preview
```

## Tile-based visual reveal

The current model actually processes tiles. Use that fact.

Example:

```text
┌────┬────┬────┐
│ ✓  │ ✓  │ ◉  │
├────┼────┼────┤
│ ✓  │ ◉  │    │
├────┼────┼────┤
│    │    │    │
└────┴────┴────┘
```

As real tiles finish, progressively reveal the corresponding reconstructed region.

This is preferable to a fake spinner.

---

# 9. Backend Structure

Recommended modular structure:

```text
backend/
│
├── api/
│   ├── routes_location.py
│   ├── routes_satellite.py
│   ├── routes_cloud.py
│   └── routes_prediction.py
│
├── services/
│   ├── satellite/
│   │   ├── base.py
│   │   └── cdse_sentinelhub.py
│   ├── pairing/
│   │   └── scene_pairer.py
│   ├── preprocessing/
│   │   ├── sentinel1.py
│   │   ├── sentinel2.py
│   │   ├── alignment.py
│   │   └── validation.py
│   ├── cloud/
│   │   ├── mask.py
│   │   └── analysis.py
│   ├── inference/
│   │   ├── model_service.py
│   │   └── tiled_inference.py
│   └── outputs/
│       ├── geotiff.py
│       └── preview.py
│
├── cache/
├── models/
├── data/
└── main.py
```

Frontend should remain independent of provider/model internals.

---

# 10. Initial API Draft

These are working contracts, not final.

### Search

```http
POST /api/satellite/search
```

```json
{
  "aoi": {...},
  "start_date": "2026-09-01",
  "end_date": "2026-09-15"
}
```

### Acquire

```http
POST /api/satellite/acquire
```

```json
{
  "sentinel2_id": "...",
  "sentinel1_id": "...",
  "aoi": {...}
}
```

### Cloud analysis

```http
POST /api/cloud/analyze
```

### Prediction

```http
POST /api/predict
```

### Progress

```http
GET /api/predict/{job_id}
```

Possible progress response:

```json
{
  "status": "processing",
  "stage": "reconstructing",
  "completed_tiles": 15,
  "total_tiles": 32
}
```

### Result

```http
GET /api/result/{job_id}
```

Return:

- preview URL
- GeoTIFF URL
- metadata
- processing information

---

# 11. Data-Compatibility Risk

The major scientific risk is domain compatibility.

Known:

```text
SEN12MS-CR
  ↓
2-band S1 + 13-band S2
  ↓
ECRformer
  ↓
known benchmark behavior
```

Not yet proven:

```text
arbitrary live Sentinel-1 + Sentinel-2
  ↓
our preprocessing
  ↓
same ECRformer checkpoint
  ↓
good reconstruction
```

We must experimentally investigate:

- product level
- band mapping
- radiometric scaling
- S1 preprocessing
- resolution
- resampling
- co-registration
- temporal pairing
- cloud distribution
- domain shift

This should be documented as a research experiment.

Do not claim generalized real-world performance until tested.

---

# 12. Evaluation Rules

When a valid cloud-free reference exists:

- PSNR
- SSIM
- RMSE
- MAE
- SAM
- LPIPS where meaningful

When no cloud-free reference exists:

Do **not** invent PSNR/SSIM/SAM.

Instead report:

- cloud percentage
- runtime
- GPU memory
- output statistics
- spatial/spectral sanity checks
- visual inspection
- acquisition metadata
- preprocessing validity

---

# 13. Development Order

## Phase 1
**Satellite acquisition**

AOI → search → select → retrieve → metadata → TIFF.

## Phase 2
**Harmonization**

S1 + S2 → alignment → band mapping → normalization → validated tensors.

## Phase 3
**Cloud intelligence**

S2 → cloud mask → overlay → statistics.

## Phase 4
**ECRformer**

Validated real data → tiled inference → 13-band prediction.

## Phase 5
**Output**

GeoTIFF + RGB preview + metadata + downloads.

## Phase 6
**Generation experience**

Real processing stages + real tile progress + reconstruction reveal.

## Phase 7
**Full application integration**

```text
Map
 ↓
Satellite search
 ↓
Scene selection
 ↓
Acquire
 ↓
Cloud analysis
 ↓
Preprocess
 ↓
Reconstruct
 ↓
Animate
 ↓
View
 ↓
Download
```

---

# 14. Antigravity Instructions

Antigravity is the implementation agent, but it should work incrementally.

### Current rules

1. **Do not implement model orchestration.**
2. **Do not train a new model.**
3. **Do not modify the ECRformer architecture.**
4. Do not remove the currently working Streamlit baseline.
5. Build the new functionality modularly around the existing baseline.
6. Do not hard-code personal or machine-specific absolute paths.
7. Keep API credentials in environment variables.
8. Never expose provider secrets to the frontend.
9. Preserve GeoTIFF CRS/transform metadata.
10. Never present PNG as the scientific output.
11. Never fabricate clouds or metrics.
12. Never fabricate generation progress.
13. Progress should represent real backend stages.
14. Keep the satellite provider behind an adapter interface.
15. Keep preprocessing and model inference separate.
16. Keep every phase independently testable.

### First Antigravity task

Implement **Phase 1A + Phase 1B only**:

```text
AOI selection
+
Sentinel-1/Sentinel-2 catalogue search
```

Then stop.

Report:

- files created
- files changed
- API endpoints
- provider/API used
- authentication requirements
- sample search request
- sample response
- tests performed
- remaining blockers

Do not automatically proceed to Phase 2.

---

# 15. First Proof-of-Concept Experiment

Before building the complete frontend/backend, build a small Python test:

```text
AOI
 ↓
CDSE catalog search
 ↓
S2 candidates
+
S1 candidates
 ↓
select one pair
 ↓
Sentinel Hub Process API
 ↓
download small AOI
 ↓
save s1.tif
save s2.tif
save metadata.json
 ↓
open using Rasterio
 ↓
print metadata
 ↓
display previews
```

Success criteria:

```text
Real Sentinel-2 TIFF received
+
Real Sentinel-1 TIFF received
+
correct metadata
+
correct AOI
+
visual preview
```

Only after this proof should the code be promoted into the application.

---

# 16. Final Architecture for THIS Stage

This is the current engineering architecture, not the final research contribution:

```text
┌─────────────────────────────────────────────────────┐
│                     USER / WEB UI                   │
│                  Map + Date selection               │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│              SATELLITE DISCOVERY                    │
│               CDSE / Sentinel APIs                  │
└──────────────────────────┬──────────────────────────┘
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
           Sentinel-2          Sentinel-1
            Optical                SAR
                  │                 │
                  └────────┬────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│                 GEO PREPROCESSING                   │
│ band mapping / alignment / normalize / validity   │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                CLOUD INTELLIGENCE                   │
│       mask / overlay / cloud statistics            │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                    ECRFORMER                        │
│              2-band S1 + 13-band S2                │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────┐
│                 13-BAND PREDICTION                  │
└──────────────────────────┬──────────────────────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
            GeoTIFF                RGB Preview
                 │                   │
                 └─────────┬─────────┘
                           ▼
┌─────────────────────────────────────────────────────┐
│          RECONSTRUCTION / PROGRESS UI              │
│       real stages + real tile progress             │
└─────────────────────────────────────────────────────┘
```

---

# 17. Future Research Stage — Intentionally Deferred

Only after the above pipeline works on real acquired Sentinel data:

```text
measure real failure cases
        ↓
identify limitations
        ↓
compare model behavior
        ↓
decide whether multiple specialists are justified
        ↓
possible orchestration architecture
```

Potential future direction:

```text
Real input
   ↓
Scene / cloud-condition analysis
   ↓
Model selection
   ├── ECRformer
   ├── Model B
   └── Model C
   ↓
quality-aware fusion/selection
   ↓
final reconstruction
```

This is deliberately **not** part of the current implementation.

---

# 18. Definition of Success for the Current Stage

The current stage is successful when a user can say:

> “I want imagery from this location and this period.”

and the software can:

```text
find observations
     ↓
retrieve Sentinel-1 + Sentinel-2
     ↓
show what was retrieved
     ↓
show the cloud-covered region
     ↓
prepare model input
     ↓
run ECRformer
     ↓
show reconstruction
     ↓
export a georeferenced 13-band GeoTIFF
```

That will give us a complete end-to-end engineering baseline.

Only then should the project move into model orchestration and deeper research modifications.

---

## Sources

Copernicus Data Space Ecosystem:
- https://documentation.dataspace.copernicus.eu/APIs.html
- https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Overview.html
- https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Process.html
- https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Catalog.html
- https://documentation.dataspace.copernicus.eu/APIs/STAC.html
- https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L1C.html
- https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html
- https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S1GRD.html

ECRformer:
- https://github.com/zzaiyan/ECRformer

SEN12MS-CR:
- https://patricktum.github.io/cloud_removal/sen12mscr/

