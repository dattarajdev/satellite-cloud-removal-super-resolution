# Phase 1 — Satellite Data Acquisition PoC

> **Status:** Proof-of-concept. Read-only diagnostic.  
> **What this does:** Proves that real Sentinel-1 and Sentinel-2 imagery can be
> acquired for a small AOI and validated for ECRformer compatibility.  
> **What this does NOT do:** Run ECRformer, modify any existing code, or
> automatically select scenes.

---

## Contents

```
acquisition/
├── poc_test.py          Main PoC script
├── README.md            This file
└── data_cache/          Downloaded TIFFs and metadata (auto-created)
    └── <hash>/
        ├── s2.tif
        ├── s1.tif
        ├── s2_preview.png
        ├── s1_preview.png
        └── metadata.json
```

---

## Step 1 — Create a CDSE Account and OAuth2 Client

### 1A. Register on CDSE

1. Go to **https://dataspace.copernicus.eu/**
2. Click **Register** (top right)
3. Complete free registration with your email address
4. Verify your email

### 1B. Create an OAuth2 Client

1. Log in to **https://dataspace.copernicus.eu/**
2. Click your username (top right) → **Account**
3. In the left sidebar, click **User Settings** → **OAuth Clients**
4. Click **+ Add new OAuth Client**
5. Enter any **Client Name** (e.g., `cloud-removal-poc`)
6. Click **Create**
7. Copy the **Client ID** and **Client Secret** — you will need them below
8. Keep the Client Secret safe — it cannot be retrieved after this page is closed

> The free CDSE tier provides a monthly quota of Processing Units.  
> A 256×256 pixel download costs approximately 65,536 processing units.  
> The free monthly quota is sufficient for this PoC.

---

## Step 2 — Set Environment Variables

**Do not hard-code credentials. Do not commit them to source control.**

### Windows — Command Prompt (session only)

```cmd
set CDSE_CLIENT_ID=your-client-id-here
set CDSE_CLIENT_SECRET=your-client-secret-here
```

### Windows — PowerShell (session only)

```powershell
$env:CDSE_CLIENT_ID     = "your-client-id-here"
$env:CDSE_CLIENT_SECRET = "your-client-secret-here"
```

### Windows — Permanent (survives reboots)

```powershell
[System.Environment]::SetEnvironmentVariable(
    "CDSE_CLIENT_ID", "your-client-id-here", "User"
)
[System.Environment]::SetEnvironmentVariable(
    "CDSE_CLIENT_SECRET", "your-client-secret-here", "User"
)
```

Restart your terminal after setting permanent variables.

### Verify the variables are set

```powershell
echo $env:CDSE_CLIENT_ID
echo $env:CDSE_CLIENT_SECRET
```

---

## Step 3 — Install Dependencies

The PoC uses only packages already in the project `venv`.  
`requests` is already installed as a transitive dependency of Streamlit.

Activate the existing venv and verify:

```powershell
# From the project root
venv\Scripts\activate

# Verify required packages
python -c "import requests, rasterio, numpy, PIL; print('All OK')"
```

If `requests` is somehow missing:

```powershell
pip install "requests>=2.31"
```

---

## Step 4 — Run the PoC

All commands below assume:
- You are in the **project root** directory: `c:\Users\datta\SYC_PROJECTS\temporary`
- The venv is activated: `venv\Scripts\activate`
- Environment variables are set

### Search mode (no download)

Lists available Sentinel-2 and Sentinel-1 scenes for the default AOI
(Rome, Italy) and the default date range (September 2024).

```powershell
python acquisition/poc_test.py
```

**What you will see:**
- A table of Sentinel-2 L1C scenes with cloud cover percentages
- A table of Sentinel-1 GRD scenes with polarization and orbit direction
- Instructions to choose dates and rerun in download mode

### Download mode

After reviewing the search output, pick a Sentinel-2 date (low cloud cover
preferred) and a Sentinel-1 date (must show VV+VH polarization):

```powershell
python acquisition/poc_test.py --s2-date 2024-09-11 --s1-date 2024-09-07
```

> **Important:** The dates you provide to `--s2-date` and `--s1-date` must
> appear in the search output. This is the manual selection step.
> The PoC does NOT automatically choose the best scene.

### Custom AOI

Replace the default Rome test area with any small bounding box:

```powershell
# Paris, France (~2km × 2km)
python acquisition/poc_test.py --west 2.30 --south 48.85 --east 2.35 --north 48.89

# London, UK (~2km × 2km)  
python acquisition/poc_test.py --west -0.13 --south 51.49 --east -0.08 --north 51.52
```

> Keep the AOI small for the PoC. A 0.05° × 0.05° box (≈ 4–5 km)
> at 256×256 pixels gives ≈ 15–20 m/pixel, close to SEN12MS-CR resolution.
> Larger AOIs work but take longer to download.

### Custom date range

```powershell
python acquisition/poc_test.py --start 2024-06-01 --end 2024-06-30
```

### Larger output (512×512 pixels)

```powershell
python acquisition/poc_test.py --s2-date 2024-09-11 --s1-date 2024-09-07 --size 512
```

---

## Step 5 — Understand the Output

### Files created

```
acquisition/data_cache/
└── <16-char-hash>/
    ├── s2.tif            13-band uint16 GeoTIFF (Sentinel-2 L1C DN)
    ├── s1.tif             2-band float32 GeoTIFF (Sentinel-1 VV+VH dB)
    ├── s2_preview.png    RGB preview (B04/B03/B02)
    ├── s1_preview.png    Grayscale preview (mean VV+VH)
    └── metadata.json     Acquisition parameters and scene metadata
```

The hash encodes: provider + collection + bbox + date + bands + output size.
Running the same request twice uses the cached file (no re-download).

### Terminal output sections

| Section | What it shows |
|---|---|
| Configuration | AOI extents, approximate GSD, mode |
| CDSE Authentication | Token acquisition result |
| Sentinel-2 L1C Catalogue | List of candidate scenes |
| Sentinel-1 GRD Catalogue | List of candidate scenes |
| Downloading Sentinel-2 | API call + file size |
| Sentinel-2 GeoTIFF Validation | CRS, transform, resolution, per-band stats |
| S2 Normalization Diagnostic | normalize_s2() output ranges and compatibility |
| Sentinel-1 GeoTIFF Validation | CRS, transform, resolution, per-band stats |
| S1 Normalization Diagnostic | normalize_s1() output ranges, VH clip check |
| Phase 1 PoC Summary | File paths and compatibility checklist |

### Inspect results with the existing TIFF viewer

```powershell
python tiff_viewer.py
```

Then click **Open TIFF** and navigate to the `data_cache/<hash>/s2.tif` or `s1.tif` file.
Use the **Band** spinner to step through all bands.

### Inspect with rasterio directly

```python
import rasterio, numpy as np
with rasterio.open("acquisition/data_cache/<hash>/s2.tif") as src:
    print(src.meta)
    print(src.crs)
    print(src.transform)
    data = src.read()
    print("shape:", data.shape)
    print("min:", data.min(), "max:", data.max())
```

---

## Known Compatibility Risks

These must be resolved experimentally before connecting the ECRformer checkpoint.

### R1 — Sentinel-2 L1C Band Units ⚠ HIGH RISK

The current ECRformer normalization expects:

```
Input:  uint16, DN, range [0, 10000]
Output: float32, [0, 1]  (after clip + divide)
```

The Sentinel Hub Processing API is configured to return `units: "DN"` in the
evalscript. **Verify** the actual data range in the terminal output.

- If the PoC reports `raw max ≈ 10000` → Compatible ✓
- If the PoC reports `raw max < 2` → The API returned float reflectance [0, 1] — incompatible
- If the PoC reports `raw max ≈ 65535` → UINT16 full scale — incompatible

### R2 — Sentinel-2 Band Order ⚠ HIGH RISK

The evalscript explicitly requests bands in the SEN12MS-CR order:

```
B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B10, B11, B12
```

Any reordering silently corrupts model input — no error is raised.
Verify the band list against the SEN12MS-CR dataset loader.

### R3 — Sentinel-1 VH Clip Range ⚠ MEDIUM RISK

The current `normalize_s1()` clips **both** VV and VH to `[-25, 0]` dB.
Some SEN12MS-CR implementations clip VH to `[-32.5, 0]` dB instead.

The PoC reports actual VH min values. If VH regularly falls below -25 dB
(which is common over smooth surfaces), many VH pixels will be clipped to 0
after normalization, discarding information. This may or may not degrade
reconstruction quality — it requires experimental measurement.

### R4 — S1 Orthorectification Mismatch ⚠ MEDIUM RISK

SEN12MS-CR used Range-Doppler terrain correction (applied with SNAP or
equivalent software before data ingestion into the dataset).

CDSE Sentinel Hub applies its own terrain correction using the Copernicus DEM.
The exact algorithm, DEM version, and output grid may differ from SEN12MS-CR.
This is a known unknown — effects on model performance must be measured.

### R5 — WGS84 vs UTM CRS ⚠ PHASE 2 CONCERN

The PoC downloads data in WGS84 (CRS84) because the input bbox is in
WGS84. This means:

- Pixels are NOT square (EW resolution ≠ NS resolution in degrees)
- The CRS differs from the UTM projections used in SEN12MS-CR

**For the PoC:** This is acceptable — the model operates on pixel values,
not geographic coordinates.

**For Phase 2 (harmonization):** S1 and S2 must be reprojected to a common
UTM CRS before spatial alignment and model input preparation.

### R6 — Domain Generalization ⚠ UNKNOWN RISK

Even with perfect preprocessing, the ECRformer checkpoint was trained on
SEN12MS-CR patches. Generalization to arbitrary live Sentinel observations
is **not guaranteed**. This is the central experiment of Phase 4.

**Do not claim model compatibility based on PoC download success alone.**

---

## API Reference

| API | URL | Auth |
|---|---|---|
| CDSE Token | `https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token` | None (posts credentials) |
| Sentinel Hub Catalog | `https://sh.dataspace.copernicus.eu/api/v1/catalog/1.0.0/search` | Bearer token |
| Sentinel Hub Processing | `https://sh.dataspace.copernicus.eu/api/v1/process` | Bearer token |

Full documentation:
- https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Catalog.html
- https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Process.html
- https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L1C.html
- https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S1GRD.html

---

## Troubleshooting

### "CDSE_CLIENT_ID environment variable is not set"

The environment variable is missing or was set in a different terminal session.
Set it in your **current** terminal session (see Step 2 above).

### "Token request failed: HTTP 401"

Your `CDSE_CLIENT_ID` or `CDSE_CLIENT_SECRET` is wrong or has been revoked.
Log in to the CDSE dashboard and regenerate your OAuth client.

### "Processing API failed: HTTP 400"

Common causes:
- No satellite data exists for the requested date and AOI
- The date was not in the catalogue search results
- The evalscript contains a syntax error (unlikely unless you modified it)

Try a different date from the search results.

### "Processing API failed: HTTP 429"

Rate limit exceeded. Wait a few minutes and try again.

### "No TIFF part found in multipart response"

The Processing API returned data but in an unexpected format.
Check `response.headers["Content-Type"]` for more context.
This may indicate an API change; file an issue.

### "rasterio.errors.NotGeoreferencedWarning"

The downloaded TIFF does not contain CRS/transform metadata.
This is unexpected — Sentinel Hub should embed geospatial metadata.
Report this with the full terminal output.

---

## Next Steps (Require Approval)

After this PoC succeeds:

- **Phase 1C/D/E:** Build the Streamlit AOI map + scene selection UI
- **Phase 2:** Spatial harmonization (UTM reprojection, S1/S2 co-registration)
- **Phase 3:** Cloud mask and cloud percentage from S2 SCL / CLM
- **Phase 4:** Connect validated live data to ECRformer inference
