"""
Satellite Cloud Removal & Reconstruction - Streamlit Application.

Integrates the canonical backend inference pipeline with a full user-facing
satellite acquisition workflow:
  Tab 1: Live Acquisition (AOI Map/Coordinates -> CDSE Catalogue -> Acquisition -> Harmonization -> ECRformer)
  Tab 2: Cached Phase 2 Baseline (Pre-validated CDSE Data / Custom Uploads -> ECRformer)
"""
from __future__ import annotations

import datetime
import io
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling as RioRes
from PIL import Image
import streamlit as st

try:
    from skimage.metrics import structural_similarity
    SKIMAGE_AVAILABLE = True
except ImportError:
    structural_similarity = None
    SKIMAGE_AVAILABLE = False

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.processing import (
    InferenceService,
    InferenceConfig,
    InferenceResult,
    InferenceDiagnostics,
    HarmonizationConfig,
    HarmonizedPair,
    harmonize_pair,
    get_device,
)
from backend.services.satellite.base import BoundingBox, S2Scene, S1Scene
from backend.services.satellite.acquisition_cache import (
    S2_BAND_NAMES,
    S1_BAND_NAMES,
    is_s2_cached,
    is_s1_cached,
    get_s2_cache_dir,
    get_s1_cache_dir,
    write_cache_metadata,
)
from visualization import to_rgb, sar_preview

CACHE_DIR = ROOT / "acquisition" / "data_cache"
HARM_OUTPUT_BASE = ROOT / "harmonization" / "output"
DEFAULT_S1 = HARM_OUTPUT_BASE / "s1_harmonized_10m.tif"
DEFAULT_S2 = HARM_OUTPUT_BASE / "s2_harmonized_10m.tif"
DEFAULT_MASK = HARM_OUTPUT_BASE / "validity_mask_10m.tif"
DEFAULT_CKPT = ROOT / "model.ckpt"

# ---- Page config ----
st.set_page_config(
    page_title="Satellite Cloud Removal | ECRformer",
    page_icon=":satellite:",
    layout="wide",
)

# ---- Sidebar Configuration ----
with st.sidebar:
    st.header("Configuration")

    cid = os.environ.get("CDSE_CLIENT_ID", "").strip()
    csec = os.environ.get("CDSE_CLIENT_SECRET", "").strip()

    if cid and csec:
        st.success("**CDSE Credentials:** Configured")
    else:
        st.warning("**CDSE Credentials:** Missing")
        with st.expander("Enter CDSE Credentials", expanded=False):
            u_cid = st.text_input("CDSE Client ID", value=st.session_state.get("cdse_cid", ""))
            u_csec = st.text_input("CDSE Client Secret", value=st.session_state.get("cdse_csec", ""), type="password")
            if st.button("Apply Credentials", key="btn_apply_creds"):
                if u_cid and u_csec:
                    os.environ["CDSE_CLIENT_ID"] = u_cid.strip()
                    os.environ["CDSE_CLIENT_SECRET"] = u_csec.strip()
                    st.session_state["cdse_cid"] = u_cid.strip()
                    st.session_state["cdse_csec"] = u_csec.strip()
                    st.rerun()

    device_obj = get_device()
    device_str = "CUDA (RTX 3050)" if "cuda" in str(device_obj) else "CPU"
    st.info(f"**Compute Device:** {device_str}")

    if DEFAULT_CKPT.exists():
        st.success(f"**Checkpoint:** model.ckpt ({DEFAULT_CKPT.stat().st_size / 1e6:.1f} MB)")
    else:
        st.error(f"**Checkpoint missing:** {DEFAULT_CKPT}")

    st.divider()
    st.subheader("Inference Settings")
    tile_size = st.select_slider("Tile Size", options=[64, 128, 256], value=128)
    overlap = st.select_slider("Tile Overlap", options=[16, 32, 64], value=32)
    brightness = st.slider("RGB Brightness", 1.0, 5.0, 2.5, 0.1)

# ---- Singleton Inference Service ----
@st.cache_resource
def get_service():
    return InferenceService(checkpoint_path=DEFAULT_CKPT)

service = get_service()

st.title(":satellite: Generative AI Satellite Cloud Removal")
st.caption("CDSE Satellite Acquisition -> Spatial Harmonization -> ECRformer Reconstruction -> 13-Band Analysis-Ready GeoTIFF")

tab_live, tab_cached, tab_eval = st.tabs(["Live Acquisition", "Cached Phase 2 Baseline", "Model Evaluation"])

# ===========================================================================
# TAB 2: Cached Phase 2 Baseline (Validated Workflow)
# ===========================================================================
with tab_cached:
    st.header("Pre-Validated Phase 2 Harmonized Data")
    st.caption("Run canonical ECRformer on verified Phase 2 GeoTIFFs (EPSG:32633, 262x200 10m grid) or custom uploaded inputs.")

    data_mode = st.radio(
        "Input Source",
        ["Cached Phase 2 CDSE Data", "Upload Custom Harmonized GeoTIFFs"],
        horizontal=True,
        key="c_tab_mode",
    )

    s1_input_c = s2_input_c = mask_input_c = None
    input_ready_c = False

    if data_mode == "Cached Phase 2 CDSE Data":
        if DEFAULT_S1.exists() and DEFAULT_S2.exists() and DEFAULT_MASK.exists():
            s1_input_c = DEFAULT_S1
            s2_input_c = DEFAULT_S2
            mask_input_c = DEFAULT_MASK
            input_ready_c = True
            st.success("Pre-loaded verified Phase 2 harmonized pair: Rome Area (EPSG:32633, 262x200, 10m spatial resolution).")
        else:
            st.error("Cached harmonized data not found in harmonization/output/.")
    else:
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            sf1 = st.file_uploader("S1 SAR (2-band TIFF)", type=["tif", "tiff"], key="up_s1")
        with cc2:
            sf2 = st.file_uploader("S2 Optical (13-band TIFF)", type=["tif", "tiff"], key="up_s2")
        with cc3:
            sfm = st.file_uploader("Validity Mask (optional)", type=["tif", "tiff"], key="up_mask")
        if sf1 and sf2:
            s1_input_c = sf1.read()
            s2_input_c = sf2.read()
            mask_input_c = sfm.read() if sfm else None
            input_ready_c = True

    if input_ready_c:
        st.subheader("Input Previews")
        pc1, pc2, pc3 = st.columns(3)
        try:
            if isinstance(s1_input_c, Path):
                with rasterio.open(s1_input_c) as src:
                    s1_arr = src.read()
                with rasterio.open(s2_input_c) as src:
                    s2_arr = src.read()
                with rasterio.open(mask_input_c) as src:
                    mask_arr = src.read(1).astype(bool)
            else:
                with rasterio.io.MemoryFile(s1_input_c) as mf:
                    with mf.open() as src:
                        s1_arr = src.read()
                with rasterio.io.MemoryFile(s2_input_c) as mf:
                    with mf.open() as src:
                        s2_arr = src.read()
                mask_arr = None
                if mask_input_c:
                    with rasterio.io.MemoryFile(mask_input_c) as mf:
                        with mf.open() as src:
                            mask_arr = src.read(1).astype(bool)

            with pc1:
                st.image(sar_preview(s1_arr, valid_mask=mask_arr), caption="S1 SAR (VV/VH False Color)", use_container_width=True)
            with pc2:
                st.image(to_rgb(s2_arr, brightness=brightness, valid_mask=mask_arr), caption="S2 Cloudy Optical (RGB)", use_container_width=True)
            with pc3:
                if mask_arr is not None:
                    st.image(Image.fromarray((mask_arr.astype(np.uint8) * 255), mode="L"),
                             caption=f"Spatial Validity Mask ({mask_arr.sum():,} valid px)", use_container_width=True)
        except Exception as e:
            st.warning(f"Preview rendering notice: {e}")

        st.divider()
        if st.button("Run ECRformer Cloud Removal", type="primary", use_container_width=True, key="cached_btn"):
            st.session_state["cached_run"] = True

        if st.session_state.get("cached_run"):
            _config = InferenceConfig(tile_size=tile_size, overlap=overlap, brightness=brightness)
            pb = st.progress(0, text="Initializing canonical inference...")

            def _cb_c(done, total):
                pb.progress(int(100 * done / max(total, 1)), text=f"Processing tiles: {done}/{total}")

            try:
                result_c = service.run(
                    s1_input=s1_input_c,
                    s2_cloudy_input=s2_input_c,
                    validity_mask_input=mask_input_c,
                    config=_config,
                    progress_callback=_cb_c,
                )
                pb.progress(100, text="Inference Complete!")
                d = result_c.diagnostics
                st.success(f"ECRformer finished in {d.execution_time_seconds:.2f}s on {d.device}")

                r1, r2 = st.columns(2)
                with r1:
                    st.image(result_c.cloudy_rgb, caption="Input Cloudy S2", use_container_width=True)
                with r2:
                    st.image(result_c.prediction_rgb, caption="ECRformer Reconstructed Cloud-Free", use_container_width=True)

                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Device", d.device.split(":")[0].upper())
                m2.metric("Time", f"{d.execution_time_seconds:.2f}s")
                m3.metric("Tiles", d.tile_count)
                m4.metric("Valid px", f"{d.valid_pixels:,}")
                m5.metric("Mean Refl.", f"{d.mean_val:.4f}")

                dc1, dc2 = st.columns(2)
                with dc1:
                    st.download_button(
                        "Download 13-Band GeoTIFF",
                        data=result_c.to_geotiff_bytes(),
                        file_name="s2_cloud_free_13band.tif",
                        mime="image/tiff",
                        use_container_width=True,
                        key="c_dl_tif",
                    )
                with dc2:
                    st.download_button(
                        "Download Preview PNG",
                        data=result_c.to_png_bytes(),
                        file_name="s2_cloud_free_preview.png",
                        mime="image/png",
                        use_container_width=True,
                        key="c_dl_png",
                    )
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
            finally:
                st.session_state["cached_run"] = False
# ===========================================================================
# TAB 1: Live Acquisition Workflow
# ===========================================================================
with tab_live:
    st.header("Live CDSE Acquisition Workflow")
    st.caption("Select an AOI, query Sentinel-2 and Sentinel-1 collections from CDSE, download or load from cache, harmonize, and reconstruct.")

    # ------------------------------------------------------------------
    # STEP 1: Area of Interest (AOI)
    # ------------------------------------------------------------------
    with st.expander("Step 1 - Area of Interest (AOI)", expanded=True):
        st.write("**Choose an AOI Preset or draw a rectangle on the map:**")
        p_col1, p_col2, p_col3 = st.columns(3)
        with p_col1:
            if st.button("Preset: Rome Benchmark (Cached)", key="p_rome"):
                st.session_state["aoi_west"] = 12.4500
                st.session_state["aoi_south"] = 41.9000
                st.session_state["aoi_east"] = 12.4730
                st.session_state["aoi_north"] = 41.9230
                st.rerun()
        with p_col2:
            if st.button("Preset: Po Valley Benchmark", key="p_povalley"):
                st.session_state["aoi_west"] = 10.5000
                st.session_state["aoi_south"] = 44.8000
                st.session_state["aoi_east"] = 10.5300
                st.session_state["aoi_north"] = 44.8300
                st.rerun()
        with p_col3:
            if st.button("Preset: Berlin", key="p_berlin"):
                st.session_state["aoi_west"] = 13.3800
                st.session_state["aoi_south"] = 52.5000
                st.session_state["aoi_east"] = 13.4100
                st.session_state["aoi_north"] = 52.5250
                st.rerun()

        map_col, coord_col = st.columns([3, 2])

        with coord_col:
            st.subheader("Bounding Box (WGS84)")
            west = st.number_input("West (min lon)", value=st.session_state.get("aoi_west", 12.4500), step=0.001, format="%.4f", key="aoi_west")
            south = st.number_input("South (min lat)", value=st.session_state.get("aoi_south", 41.9000), step=0.001, format="%.4f", key="aoi_south")
            east = st.number_input("East (max lon)", value=st.session_state.get("aoi_east", 12.4730), step=0.001, format="%.4f", key="aoi_east")
            north = st.number_input("North (max lat)", value=st.session_state.get("aoi_north", 41.9230), step=0.001, format="%.4f", key="aoi_north")
            output_px = st.select_slider("Download Acquisition Grid", options=[128, 256, 512], value=256, key="output_px")
            st.caption("Download dimensions describe the initial raw acquisition. Final harmonized grid resolution is strictly 10m UTM.")

        with map_col:
            st.subheader("Interactive Map Selection")
            st.caption("Use the rectangle tool on the left toolbar to draw an AOI, then click 'Apply Drawn AOI'.")
            try:
                import folium
                from folium.plugins import Draw
                from streamlit_folium import st_folium

                center_lat = (south + north) / 2.0
                center_lon = (west + east) / 2.0
                m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
                folium.Rectangle(
                    bounds=[[south, west], [north, east]],
                    color="#FF4B4B",
                    fill=True,
                    fill_opacity=0.15,
                    weight=2,
                    tooltip="Current AOI",
                ).add_to(m)
                Draw(
                    draw_options={
                        "polyline": False,
                        "polygon": False,
                        "circle": False,
                        "marker": False,
                        "circlemarker": False,
                        "rectangle": True,
                    },
                    edit_options={"edit": False},
                ).add_to(m)
                map_result = st_folium(m, width=520, height=380, key="aoi_map")
                if map_result and map_result.get("last_active_drawing"):
                    geom = map_result["last_active_drawing"].get("geometry", {})
                    if geom.get("type") == "Polygon":
                        coords = geom["coordinates"][0]
                        lons = [c[0] for c in coords]
                        lats = [c[1] for c in coords]
                        drawn_w = round(min(lons), 6)
                        drawn_s = round(min(lats), 6)
                        drawn_e = round(max(lons), 6)
                        drawn_n = round(max(lats), 6)
                        st.info(f"Drawn rectangle: W={drawn_w} S={drawn_s} E={drawn_e} N={drawn_n}")
                        if st.button("Apply Drawn AOI", key="apply_drawn"):
                            st.session_state["aoi_west"] = drawn_w
                            st.session_state["aoi_south"] = drawn_s
                            st.session_state["aoi_east"] = drawn_e
                            st.session_state["aoi_north"] = drawn_n
                            st.rerun()
            except ImportError:
                st.warning("streamlit-folium is not installed. Using numerical inputs.")

        if west >= east or south >= north:
            st.error("Invalid AOI: west must be strictly less than east, and south must be less than north.")
            bbox_valid = False
            bbox = None
        else:
            bbox_valid = True
            bbox = BoundingBox(west=west, south=south, east=east, north=north)
            output_size_px = (output_px, output_px)
            st.success(
                f"Selected AOI: W={west:.4f}, S={south:.4f}, E={east:.4f}, N={north:.4f} | "
                f"Approx: {bbox.approx_width_m() / 1000:.2f} km x {bbox.approx_height_m() / 1000:.2f} km | "
                f"Raw Download Grid: {output_px}x{output_px} px"
            )

    # ------------------------------------------------------------------
    # STEP 2: Date Range
    # ------------------------------------------------------------------
    with st.expander("Step 2 - Date Range", expanded=True):
        d1, d2 = st.columns(2)
        with d1:
            start_date = st.date_input("Start Date", value=datetime.date(2024, 9, 1), key="start_date")
        with d2:
            end_date = st.date_input("End Date", value=datetime.date(2024, 9, 30), key="end_date")

        if start_date >= end_date:
            st.error("Start date must be strictly before end date.")
            date_valid = False
            start_str = end_str = ""
        else:
            date_valid = True
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

    # ------------------------------------------------------------------
    # STEP 3: CDSE Catalogue Search
    # ------------------------------------------------------------------
    with st.expander("Step 3 - Search CDSE Catalogue", expanded=True):
        search_disabled = not (bbox_valid and date_valid)

        btn_c1, btn_c2 = st.columns(2)
        with btn_c1:
            search_btn = st.button("Search CDSE Catalogue (Live)", disabled=search_disabled, key="search_btn", type="primary")
        with btn_c2:
            offline_btn = st.button("Load Cached Rome Scenes (Offline Demo)", key="offline_btn")

        if search_btn and bbox_valid and date_valid:
            for k in ["s2_scenes", "s1_scenes", "selected_s2_idx", "selected_s1_idx", "catalogue_provider"]:
                st.session_state.pop(k, None)

            try:
                from backend.services.satellite.cdse_sentinelhub import CDSESentinelHubProvider
                provider = CDSESentinelHubProvider()
                st.session_state["catalogue_provider"] = provider
            except Exception as exc:
                st.error(f"Authentication error: {exc}. Please configure CDSE credentials in the sidebar or use the Offline Demo.")
                provider = None

            if provider is not None:
                with st.spinner("Searching Sentinel-2 L1C catalogue..."):
                    try:
                        st.session_state["s2_scenes"] = provider.search_sentinel2(
                            bbox=bbox, start_date=start_str, end_date=end_str, max_results=25,
                        )
                    except Exception as exc:
                        st.error(f"Sentinel-2 search failed: {exc}")
                        st.session_state["s2_scenes"] = []

                with st.spinner("Searching Sentinel-1 GRD catalogue..."):
                    try:
                        st.session_state["s1_scenes"] = provider.search_sentinel1(
                            bbox=bbox, start_date=start_str, end_date=end_str, max_results=25,
                        )
                    except Exception as exc:
                        st.error(f"Sentinel-1 search failed: {exc}")
                        st.session_state["s1_scenes"] = []

        if offline_btn:
            st.session_state["s2_scenes"] = [
                S2Scene(
                    scene_id="S2A_MSIL1C_20240921T100031_N0511_R122_T32TQM_20240921T120138.SAFE",
                    dt=datetime.datetime(2024, 9, 21, 10, 0, 31, tzinfo=datetime.timezone.utc),
                    cloud_cover=4.26,
                    processing_level="L1C",
                    footprint=None,
                )
            ]
            st.session_state["s1_scenes"] = [
                S1Scene(
                    scene_id="S1A_IW_GRDH_1SDV_20240920T051201_20240920T051226_055744_06CF2B_B9A8_COG.SAFE",
                    dt=datetime.datetime(2024, 9, 20, 5, 12, 1, tzinfo=datetime.timezone.utc),
                    product_type="GRD",
                    polarizations=["VV", "VH"],
                    orbit_direction="DESCENDING",
                    instrument_mode="IW",
                    footprint=None,
                )
            ]
            st.session_state["selected_s2_idx"] = 0
            st.session_state["selected_s1_idx"] = 0
            st.success("Loaded verified Rome cached acquisition metadata.")

        # Display S2 Catalogue Results
        if "s2_scenes" in st.session_state:
            s2_scenes = st.session_state["s2_scenes"]
            st.subheader(f"Sentinel-2 L1C Results ({len(s2_scenes)} scenes)")
            if not s2_scenes:
                st.warning("No Sentinel-2 scenes found for this AOI and date range.")
            else:
                s2_df = pd.DataFrame([{
                    "#": i + 1,
                    "Date": sc.utc_date(),
                    "Time UTC": sc.utc_time(),
                    "Cloud %": f"{sc.cloud_cover:.1f}%" if sc.cloud_cover is not None else "N/A",
                    "Scene ID": sc.scene_id,
                } for i, sc in enumerate(s2_scenes)])
                st.dataframe(s2_df, use_container_width=True, hide_index=True)

                s2_opts = {
                    f"[{i + 1}] {sc.utc_date()} | Cloud: {sc.cloud_str().strip()} | {sc.scene_id[:45]}": i
                    for i, sc in enumerate(s2_scenes)
                }
                cur_s2_idx = st.session_state.get("selected_s2_idx")
                chosen_s2 = st.selectbox(
                    "Select Sentinel-2 scene",
                    list(s2_opts.keys()),
                    index=cur_s2_idx if (cur_s2_idx is not None and cur_s2_idx < len(s2_opts)) else None,
                    placeholder="-- Choose an S2 scene --",
                    key="s2_sel",
                )
                if chosen_s2 is not None:
                    st.session_state["selected_s2_idx"] = s2_opts[chosen_s2]

        # Display S1 Catalogue Results
        if "s1_scenes" in st.session_state:
            s1_scenes = st.session_state["s1_scenes"]
            st.subheader(f"Sentinel-1 GRD Results ({len(s1_scenes)} scenes)")
            if not s1_scenes:
                st.warning("No Sentinel-1 scenes found for this AOI and date range.")
            else:
                s1_df = pd.DataFrame([{
                    "#": i + 1,
                    "Date": sc.utc_date(),
                    "Time UTC": sc.utc_time(),
                    "Polarization": sc.pol_str(),
                    "Orbit": sc.orbit_direction or "?",
                    "Mode": sc.instrument_mode or "?",
                    "VV+VH": "Yes" if sc.has_vv_vh() else "No",
                    "Scene ID": sc.scene_id,
                } for i, sc in enumerate(s1_scenes)])
                st.dataframe(s1_df, use_container_width=True, hide_index=True)

                s1_opts = {
                    f"[{i + 1}] {sc.utc_date()} | {sc.pol_str()} {sc.orbit_direction or '?'} | {sc.scene_id[:45]}": i
                    for i, sc in enumerate(s1_scenes)
                }
                cur_s1_idx = st.session_state.get("selected_s1_idx")
                chosen_s1 = st.selectbox(
                    "Select Sentinel-1 scene",
                    list(s1_opts.keys()),
                    index=cur_s1_idx if (cur_s1_idx is not None and cur_s1_idx < len(s1_opts)) else None,
                    placeholder="-- Choose an S1 scene --",
                    key="s1_sel",
                )
                if chosen_s1 is not None:
                    st.session_state["selected_s1_idx"] = s1_opts[chosen_s1]

    # ------------------------------------------------------------------
    # STEP 4: Confirm Selection & Temporal Gap
    # ------------------------------------------------------------------
    with st.expander("Step 4 - Confirm Scene Selection & Temporal Alignment", expanded=True):
        sel_s2_idx = st.session_state.get("selected_s2_idx")
        sel_s1_idx = st.session_state.get("selected_s1_idx")
        s2_scenes_all = st.session_state.get("s2_scenes", [])
        s1_scenes_all = st.session_state.get("s1_scenes", [])

        sel_s2 = s2_scenes_all[sel_s2_idx] if (sel_s2_idx is not None and sel_s2_idx < len(s2_scenes_all)) else None
        sel_s1 = s1_scenes_all[sel_s1_idx] if (sel_s1_idx is not None and sel_s1_idx < len(s1_scenes_all)) else None

        cc1, cc2 = st.columns(2)
        with cc1:
            if sel_s2:
                st.success(
                    f"**Selected S2:** {sel_s2.utc_date()} {sel_s2.utc_time()} UTC | "
                    f"Cloud: {sel_s2.cloud_str().strip()} | {sel_s2.scene_id[:35]}..."
                )
            else:
                st.info("No Sentinel-2 scene selected.")
        with cc2:
            if sel_s1:
                st.success(
                    f"**Selected S1:** {sel_s1.utc_date()} {sel_s1.utc_time()} UTC | "
                    f"{sel_s1.pol_str()} {sel_s1.orbit_direction or '?'} | {sel_s1.scene_id[:35]}..."
                )
            else:
                st.info("No Sentinel-1 scene selected.")

        both_selected = (sel_s2 is not None) and (sel_s1 is not None)

        if both_selected:
            delta_sec = abs((sel_s2.dt - sel_s1.dt).total_seconds())
            delta_days = int(delta_sec // 86400)
            delta_hrs = int((delta_sec % 86400) // 3600)
            s2_dt_str = sel_s2.dt.strftime('%Y-%m-%d %H:%M')
            s1_dt_str = sel_s1.dt.strftime('%Y-%m-%d %H:%M')
            gap_text = f"S2: {s2_dt_str} UTC | S1: {s1_dt_str} UTC | Temporal gap: **{delta_days}d {delta_hrs}h**"

            if delta_days > 7:
                st.warning(f"{gap_text} | Warning: Temporal gap exceeds 7 days. Surface changes may be present.")
            elif delta_days > 3:
                st.info(f"{gap_text} | Temporal gap is between 3 and 7 days. Acceptable for most terrain.")
            else:
                st.success(f"{gap_text} | Excellent temporal alignment (within 3 days)!")
        else:
            st.caption("Select both an S2 scene and an S1 scene above to evaluate temporal gap and enable acquisition.")

    # ------------------------------------------------------------------
    # STEP 5: Acquire, Harmonize & Run ECRformer
    # ------------------------------------------------------------------
    with st.expander("Step 5 - Acquire, Harmonize & Run ECRformer", expanded=True):
        acquire_btn = st.button(
            "Acquire + Harmonize + Run ECRformer",
            type="primary",
            disabled=not both_selected,
            use_container_width=True,
            key="acquire_btn",
        )

        if acquire_btn and both_selected:
            session_id = str(uuid.uuid4())[:8]
            session_harm_dir = HARM_OUTPUT_BASE / "sessions" / session_id
            session_harm_dir.mkdir(parents=True, exist_ok=True)

            log_lines = []
            log_box = st.empty()

            def log(msg, ok=True):
                icon = "[OK]" if ok else "[ERR]"
                log_lines.append(f"{icon} {msg}")
                log_box.code("\n".join(log_lines))

            # --- S2 Acquisition ---
            s2_cache_dir = get_s2_cache_dir(CACHE_DIR, bbox, sel_s2.utc_date(), output_size_px)
            s2_cache_dir.mkdir(parents=True, exist_ok=True)
            s2_tif = s2_cache_dir / "s2.tif"
            s2_ok = False

            if is_s2_cached(CACHE_DIR, bbox, sel_s2.utc_date(), output_size_px):
                log(f"Sentinel-2 optical acquired from cache ({s2_tif.stat().st_size / 1024:.1f} KB)")
                s2_ok = True
            else:
                provider = st.session_state.get("catalogue_provider")
                if provider is None:
                    try:
                        from backend.services.satellite.cdse_sentinelhub import CDSESentinelHubProvider
                        provider = CDSESentinelHubProvider()
                        st.session_state["catalogue_provider"] = provider
                    except Exception as exc:
                        st.error(f"CDSE credentials required for download: {exc}")
                        log("Sentinel-2 download FAILED - credentials not configured", ok=False)

                if provider is not None:
                    log(f"Downloading Sentinel-2 ({sel_s2.utc_date()}) from CDSE...")
                    try:
                        provider.download_sentinel2(
                            bbox=bbox, target_date=sel_s2.utc_date(),
                            output_path=str(s2_tif), output_size_px=output_size_px,
                        )
                        write_cache_metadata(
                            s2_cache_dir, "sentinel-2-l1c", bbox, sel_s2.utc_date(),
                            S2_BAND_NAMES, output_size_px,
                            extra={"scene_id": sel_s2.scene_id, "cloud_cover": sel_s2.cloud_cover},
                        )
                        log(f"Sentinel-2 downloaded successfully ({s2_tif.stat().st_size / 1024:.1f} KB)")
                        s2_ok = True
                    except Exception as exc:
                        st.error(f"Sentinel-2 acquisition failed: {exc}")
                        log(f"Sentinel-2 download FAILED: {exc}", ok=False)

            # --- S1 Acquisition ---
            s1_ok = False
            if s2_ok:
                s1_cache_dir = get_s1_cache_dir(CACHE_DIR, bbox, sel_s1.utc_date(), output_size_px)
                s1_cache_dir.mkdir(parents=True, exist_ok=True)
                s1_tif = s1_cache_dir / "s1.tif"

                if is_s1_cached(CACHE_DIR, bbox, sel_s1.utc_date(), output_size_px):
                    log(f"Sentinel-1 SAR acquired from cache ({s1_tif.stat().st_size / 1024:.1f} KB)")
                    s1_ok = True
                else:
                    provider = st.session_state.get("catalogue_provider")
                    if provider is None:
                        try:
                            from backend.services.satellite.cdse_sentinelhub import CDSESentinelHubProvider
                            provider = CDSESentinelHubProvider()
                            st.session_state["catalogue_provider"] = provider
                        except Exception as exc:
                            st.error(f"CDSE credentials required for download: {exc}")
                            log("Sentinel-1 download FAILED - credentials not configured", ok=False)

                    if provider is not None:
                        log(f"Downloading Sentinel-1 ({sel_s1.utc_date()}) from CDSE...")
                        try:
                            provider.download_sentinel1(
                                bbox=bbox, target_date=sel_s1.utc_date(),
                                output_path=str(s1_tif), output_size_px=output_size_px,
                            )
                            write_cache_metadata(
                                s1_cache_dir, "sentinel-1-grd", bbox, sel_s1.utc_date(),
                                S1_BAND_NAMES, output_size_px,
                                extra={
                                    "scene_id": sel_s1.scene_id,
                                    "polarizations": sel_s1.polarizations,
                                    "orbit_direction": sel_s1.orbit_direction,
                                    "instrument_mode": sel_s1.instrument_mode,
                                },
                            )
                            log(f"Sentinel-1 downloaded successfully ({s1_tif.stat().st_size / 1024:.1f} KB)")
                            s1_ok = True
                        except Exception as exc:
                            st.error(f"Sentinel-1 acquisition failed: {exc}")
                            log(f"Sentinel-1 download FAILED: {exc}", ok=False)

            # --- Phase 2 Spatial Harmonization ---
            harm_ok = False
            harm_pair = None
            if s1_ok and s2_ok:
                log("Executing Phase 2 spatial harmonization (UTM reprojection, 10m grid, validity mask)...")
                try:
                    h_config = HarmonizationConfig(
                        pixel_size=10.0,
                        s2_resampling=RioRes.bilinear,
                        s1_resampling=RioRes.bilinear,
                        save_previews=True,
                        save_validity_mask=True,
                    )
                    harm_pair = harmonize_pair(
                        s2_input_path=s2_tif,
                        s1_input_path=s1_tif,
                        output_dir=str(session_harm_dir),
                        config=h_config,
                    )
                    log(f"Harmonization completed: {harm_pair.s2_shape[2]}x{harm_pair.s2_shape[1]} px, EPSG:{harm_pair.grid.crs.to_epsg()}")
                    harm_ok = True
                except Exception as exc:
                    st.error(f"Harmonization failed: {exc}")
                    log(f"Harmonization FAILED: {exc}", ok=False)

            # --- Canonical ECRformer Inference ---
            if harm_ok and (harm_pair is not None):
                log("Executing ECRformer deep reconstruction on harmonized inputs...")
                try:
                    infer_cfg = InferenceConfig(tile_size=tile_size, overlap=overlap, brightness=brightness)
                    result = service.run(
                        s1_input=harm_pair.s1_path,
                        s2_cloudy_input=harm_pair.s2_path,
                        validity_mask_input=harm_pair.validity_mask_path,
                        config=infer_cfg,
                    )
                    d = result.diagnostics
                    log(f"ECRformer reconstruction succeeded in {d.execution_time_seconds:.2f}s on {d.device}!")
                    log(f"Diagnostics: {d.tile_count} tiles, {d.valid_pixels:,} valid pixels, mean={d.mean_val:.4f}")
                    st.session_state["live_result"] = result
                    st.session_state["live_harm_pair"] = harm_pair
                    st.session_state["live_s2_meta"] = sel_s2
                except Exception as exc:
                    st.error(f"ECRformer inference failed: {exc}")
                    log(f"Inference FAILED: {exc}", ok=False)

    # ------------------------------------------------------------------
    # RESULTS & VISUALIZATION (Tab 1)
    # ------------------------------------------------------------------
    live_res: Optional[InferenceResult] = st.session_state.get("live_result")
    live_harm: Optional[HarmonizedPair] = st.session_state.get("live_harm_pair")
    live_s2 = st.session_state.get("live_s2_meta")

    if live_res is not None:
        st.divider()
        st.subheader("Reconstruction Results")
        diag = live_res.diagnostics
        st.success(f"ECRformer Reconstruction Successful ({diag.execution_time_seconds:.2f}s on {diag.device})")

        # Metric cards
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Compute Device", diag.device.split(":")[0].upper())
        m2.metric("Inference Time", f"{diag.execution_time_seconds:.2f}s")
        m3.metric("Tiles Processed", diag.tile_count)
        m4.metric("Valid Pixels", f"{diag.valid_pixels:,}")
        m5.metric("Mean Reflectance", f"{diag.mean_val:.4f}")

        # Visual Comparison (3 Columns)
        v1, v2, v3 = st.columns(3)
        with v1:
            st.image(live_res.cloudy_rgb, caption="Input Sentinel-2 Cloudy (RGB)", use_container_width=True)
        with v2:
            st.image(live_res.sar_preview_img, caption="Harmonized Sentinel-1 SAR (VV/VH False Color)", use_container_width=True)
        with v3:
            st.image(live_res.prediction_rgb, caption="ECRformer Reconstructed Cloud-Free (RGB)", use_container_width=True)

        if live_harm and live_harm.preview_coreg_path and live_harm.preview_coreg_path.exists():
            with st.expander("SAR / Optical Co-Registration Diagnostics", expanded=False):
                st.image(str(live_harm.preview_coreg_path), caption="SAR High-Frequency Edge Overlay (Magenta) on Optical S2", use_container_width=True)

        # Downloads
        st.subheader("Export & Download")
        dl1, dl2, dl3 = st.columns(3)
        date_str = live_s2.utc_date() if live_s2 else "output"

        with dl1:
            st.download_button(
                "Download 13-Band GeoTIFF",
                data=live_res.to_geotiff_bytes(),
                file_name=f"s2_reconstructed_{date_str}_13band.tif",
                mime="image/tiff",
                use_container_width=True,
                key="live_dl_tif",
            )
        with dl2:
            st.download_button(
                "Download Preview PNG",
                data=live_res.to_png_bytes(),
                file_name=f"s2_reconstructed_{date_str}_preview.png",
                mime="image/png",
                use_container_width=True,
                key="live_dl_png",
            )
        with dl3:
            diag_json = json.dumps(diag.to_dict(), indent=2)
            st.download_button(
                "Download Diagnostics JSON",
                data=diag_json,
                file_name="diagnostics.json",
                mime="application/json",
                use_container_width=True,
                key="live_dl_json",
            )

        with st.expander("Technical Diagnostics & Metadata", expanded=False):
            st.json(diag.to_dict())
            if live_harm:
                st.json({
                    "harmonized_crs": f"EPSG:{live_harm.grid.crs.to_epsg()}",
                    "dimensions_px": [live_harm.s2_shape[2], live_harm.s2_shape[1]],
                    "target_resolution_m": 10.0,
                    "validation_passed": live_harm.validation_passed,
                })
# ===========================================================================
# TAB 3: MODEL EVALUATION (REFERENCE-BASED / RESEARCH)
# ===========================================================================
with tab_eval:
    st.header("Model Evaluation & Research Metrics")
    st.caption(
        "Reference-based evaluation is separate from live inference. "
        "Metrics are shown only when a cloud-free reference is available."
    )

    eval_col1, eval_col2 = st.columns([2, 1])

    with eval_col1:
        st.subheader("Evaluation Dataset")

        default_eval_s1 = ROOT / "harmonization" / "output" / "s1_harmonized_10m.tif"
        default_eval_cloudy = ROOT / "cloud_generation" / "output" / "s2_cloudy_mock.tif"
        default_eval_clear = ROOT / "harmonization" / "output" / "s2_harmonized_10m.tif"
        default_eval_mask = ROOT / "harmonization" / "output" / "validity_mask_10m.tif"
        default_true_cloud = ROOT / "cloud_generation" / "output" / "cloud_mask.tif"
        default_pred_cloud = ROOT / "cloud_generation" / "output" / "predicted_cloud_mask.tif"

        eval_mode = st.radio(
            "Evaluation source",
            ["Use local synthetic-cloud experiment", "Upload reference evaluation data"],
            horizontal=True,
            key="eval_source_mode",
        )

        eval_s1 = eval_cloudy = eval_clear = eval_mask = None
        eval_true_cloud = eval_pred_cloud = None

        if eval_mode == "Use local synthetic-cloud experiment":
            missing = [
                str(p) for p in [
                    default_eval_s1,
                    default_eval_cloudy,
                    default_eval_clear,
                    default_eval_mask,
                ] if not p.exists()
            ]

            if missing:
                st.error("Required local evaluation files are missing:")
                for item in missing:
                    st.write(f"- `{item}`")
            else:
                eval_s1 = default_eval_s1
                eval_cloudy = default_eval_cloudy
                eval_clear = default_eval_clear
                eval_mask = default_eval_mask
                if default_true_cloud.exists():
                    eval_true_cloud = default_true_cloud
                if default_pred_cloud.exists():
                    eval_pred_cloud = default_pred_cloud
                st.success(
                    "Loaded local controlled evaluation set: "
                    "S1 + synthetic cloudy S2 + clear reference S2."
                )
        else:
            up1, up2, up3 = st.columns(3)
            with up1:
                eval_s1 = st.file_uploader(
                    "S1 SAR (2-band TIFF)", type=["tif", "tiff"], key="eval_s1_upload"
                )
            with up2:
                eval_cloudy = st.file_uploader(
                    "Cloudy S2 (13-band TIFF)", type=["tif", "tiff"], key="eval_cloudy_upload"
                )
            with up3:
                eval_clear = st.file_uploader(
                    "Clear/reference S2 (13-band TIFF)", type=["tif", "tiff"], key="eval_clear_upload"
                )

            eval_mask = st.file_uploader(
                "Validity mask (optional)", type=["tif", "tiff"], key="eval_mask_upload"
            )
            eval_true_cloud = st.file_uploader(
                "True cloud mask (optional)", type=["tif", "tiff"], key="eval_true_cloud_upload"
            )
            eval_pred_cloud = st.file_uploader(
                "Predicted cloud mask (optional)", type=["tif", "tiff"], key="eval_pred_cloud_upload"
            )

            if eval_s1 and eval_cloudy and eval_clear:
                st.success("Reference-based evaluation inputs are ready.")

    with eval_col2:
        st.subheader("Metric definitions")
        st.markdown(
            """
**Reconstruction**
- **MAE** — mean absolute reconstruction error
- **RMSE** — root-mean-square reconstruction error
- **PSNR** — reconstruction fidelity in dB
- **SSIM** — RGB structural similarity
- **SAM** — spectral angle between predicted/reference spectra

**Cloud mask**
- **Precision** — predicted cloud pixels that are truly cloud
- **Recall** — true cloud pixels detected
- **F1** — precision/recall balance
- **IoU** — predicted/true cloud overlap
"""
        )

    def _read_raster(source):
        if isinstance(source, (str, Path)):
            with rasterio.open(source) as src:
                return src.read().astype(np.float32), src.profile.copy()

        if hasattr(source, "read"):
            source.seek(0)
            data = source.read()
            with rasterio.io.MemoryFile(data) as mf:
                with mf.open() as src:
                    return src.read().astype(np.float32), src.profile.copy()

        raise TypeError(f"Unsupported input type: {type(source)}")

    def _read_mask(source, shape):
        if source is None:
            return np.ones(shape, dtype=bool)
        arr, _ = _read_raster(source)
        if arr.shape[0] < 1 or arr.shape[1:] != shape:
            raise ValueError(
                f"Validity mask shape {arr.shape} does not match target shape {shape}."
            )
        return arr[0].astype(bool)

    def _normalized_reflectance(arr):
        arr = np.asarray(arr, dtype=np.float32)
        finite = arr[np.isfinite(arr)]
        if finite.size and float(np.nanmax(finite)) > 1.5:
            return np.clip(arr / 10000.0, 0.0, 1.0)
        return np.clip(arr, 0.0, 1.0)

    def _rgb_norm(s2):
        rgb = np.stack([s2[3], s2[2], s2[1]], axis=-1)
        return np.clip(rgb * brightness, 0.0, 1.0)

    def _compute_reconstruction_metrics(pred, target, mask):
        pred = _normalized_reflectance(pred)
        target = _normalized_reflectance(target)
        valid = mask.astype(bool)

        if not valid.any():
            raise ValueError("No valid pixels available for metric computation.")

        diff = pred[:, valid] - target[:, valid]
        mae = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        psnr = float(20.0 * np.log10(1.0 / max(rmse, 1e-12)))

        pred_vec = pred[:, valid].T
        target_vec = target[:, valid].T
        denominator = (
            np.linalg.norm(pred_vec, axis=1)
            * np.linalg.norm(target_vec, axis=1)
        )
        good = denominator > 1e-8

        if np.any(good):
            cosine = np.clip(
                np.sum(pred_vec[good] * target_vec[good], axis=1)
                / denominator[good],
                -1.0,
                1.0,
            )
            sam_deg = float(np.mean(np.arccos(cosine)) * 180.0 / np.pi)
        else:
            sam_deg = float("nan")

        ssim_rgb = None
        if SKIMAGE_AVAILABLE:
            pred_rgb = _rgb_norm(pred)
            target_rgb = _rgb_norm(target)
            pred_rgb[~valid] = target_rgb[~valid]
            ssim_rgb = float(np.mean([
                structural_similarity(
                    target_rgb[:, :, channel],
                    pred_rgb[:, :, channel],
                    data_range=1.0,
                )
                for channel in range(3)
            ]))

        return {
            "MAE": mae,
            "RMSE": rmse,
            "PSNR_dB": psnr,
            "SSIM_RGB": ssim_rgb,
            "SAM_degrees": sam_deg,
        }

    def _cloud_metrics(true_mask_source, pred_mask_source, valid_mask):
        true_arr, _ = _read_raster(true_mask_source)
        pred_arr, _ = _read_raster(pred_mask_source)

        true_mask = (true_arr[0] >= 0.5) & valid_mask
        predicted_mask = (pred_arr[0] >= 0.5) & valid_mask

        tp = int(np.sum(predicted_mask & true_mask))
        fp = int(np.sum(predicted_mask & ~true_mask))
        fn = int(np.sum(~predicted_mask & true_mask))
        tn = int(np.sum(~predicted_mask & ~true_mask))

        precision_value = tp / (tp + fp) if (tp + fp) else 0.0
        recall_value = tp / (tp + fn) if (tp + fn) else 0.0
        f1_value = (
            2.0 * precision_value * recall_value / (precision_value + recall_value)
            if (precision_value + recall_value) else 0.0
        )
        iou_value = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

        valid_count = max(int(valid_mask.sum()), 1)

        return {
            "Precision": precision_value,
            "Recall": recall_value,
            "F1": f1_value,
            "IoU": iou_value,
            "true_coverage_pct": float(true_mask.sum() / valid_count * 100.0),
            "pred_coverage_pct": float(predicted_mask.sum() / valid_count * 100.0),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        }

    ready_eval = eval_s1 is not None and eval_cloudy is not None and eval_clear is not None

    if ready_eval and st.button(
        "Run Reference-Based Evaluation",
        type="primary",
        use_container_width=True,
        key="run_eval_btn",
    ):
        try:
            clear_array, _ = _read_raster(eval_clear)
            validity_mask = _read_mask(eval_mask, clear_array.shape[1:])

            progress = st.progress(0, text="Running ECRformer evaluation...")
            eval_config = InferenceConfig(
                tile_size=tile_size, overlap=overlap, brightness=brightness
            )

            def _eval_progress(done, total):
                progress.progress(
                    int(100 * done / max(total, 1)),
                    text=f"Evaluation inference: {done}/{total} tiles",
                )

            evaluation_result = service.run(
                s1_input=eval_s1,
                s2_cloudy_input=eval_cloudy,
                validity_mask_input=eval_mask,
                config=eval_config,
                progress_callback=_eval_progress,
            )
            progress.progress(100, text="Evaluation inference complete!")

            prediction_bytes = evaluation_result.to_geotiff_bytes()
            with rasterio.io.MemoryFile(prediction_bytes) as mem:
                with mem.open() as src:
                    prediction_array = src.read().astype(np.float32)

            cloudy_array, _ = _read_raster(eval_cloudy)

            cloudy_metrics = _compute_reconstruction_metrics(
                cloudy_array, clear_array, validity_mask
            )
            ecrformer_metrics = _compute_reconstruction_metrics(
                prediction_array, clear_array, validity_mask
            )

            cloud_region_metrics = None
            clear_region_metrics = None
            detection_metrics = None

            if eval_true_cloud is not None:
                true_cloud_array, _ = _read_raster(eval_true_cloud)
                true_cloud_mask = (true_cloud_array[0] >= 0.5) & validity_mask
                clear_region_mask = validity_mask & ~true_cloud_mask

                if true_cloud_mask.any():
                    cloud_region_metrics = {
                        "Cloudy Input": _compute_reconstruction_metrics(
                            cloudy_array, clear_array, true_cloud_mask
                        ),
                        "ECRformer": _compute_reconstruction_metrics(
                            prediction_array, clear_array, true_cloud_mask
                        ),
                    }

                if clear_region_mask.any():
                    clear_region_metrics = {
                        "Cloudy Input": _compute_reconstruction_metrics(
                            cloudy_array, clear_array, clear_region_mask
                        ),
                        "ECRformer": _compute_reconstruction_metrics(
                            prediction_array, clear_array, clear_region_mask
                        ),
                    }

            if eval_true_cloud is not None and eval_pred_cloud is not None:
                detection_metrics = _cloud_metrics(
                    eval_true_cloud, eval_pred_cloud, validity_mask
                )

            st.session_state["evaluation_results"] = {
                "cloudy_metrics": cloudy_metrics,
                "ecrformer_metrics": ecrformer_metrics,
                "cloud_region_metrics": cloud_region_metrics,
                "clear_region_metrics": clear_region_metrics,
                "cloud_detection_metrics": detection_metrics,
                "diagnostics": evaluation_result.diagnostics.to_dict(),
                "cloudy_rgb": evaluation_result.cloudy_rgb,
                "prediction_rgb": evaluation_result.prediction_rgb,
                "clear_rgb": _rgb_norm(_normalized_reflectance(clear_array)),
            }

            st.rerun()

        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    evaluation = st.session_state.get("evaluation_results")

    if evaluation:
        st.divider()
        st.subheader("Reference-Based Reconstruction Evaluation")
        st.info(
            "These metrics require a known cloud-free reference image. "
            "They are for model evaluation and are not live-scene accuracy."
        )

        cloudy_metric = evaluation["cloudy_metrics"]
        ecr_metric = evaluation["ecrformer_metrics"]

        v1, v2, v3 = st.columns(3)
        with v1:
            st.image(
                evaluation["cloudy_rgb"],
                caption="Cloudy S2 Input",
                use_container_width=True,
            )
        with v2:
            st.image(
                evaluation["prediction_rgb"],
                caption="ECRformer Reconstruction",
                use_container_width=True,
            )
        with v3:
            st.image(
                evaluation["clear_rgb"],
                caption="Clear Reference",
                use_container_width=True,
            )

        st.subheader("Reconstruction Metrics")

        st.markdown("**Cloudy Input vs Clear Reference**")
        b1, b2, b3 = st.columns(3)
        b1.metric("MAE", f"{cloudy_metric['MAE']:.6f}")
        b2.metric("RMSE", f"{cloudy_metric['RMSE']:.6f}")
        b3.metric("PSNR", f"{cloudy_metric['PSNR_dB']:.3f} dB")

        st.markdown("**ECRformer vs Clear Reference**")
        e1, e2, e3 = st.columns(3)
        e1.metric("MAE", f"{ecr_metric['MAE']:.6f}")
        e2.metric("RMSE", f"{ecr_metric['RMSE']:.6f}")
        e3.metric("PSNR", f"{ecr_metric['PSNR_dB']:.3f} dB")

        e4, e5 = st.columns(2)
        if ecr_metric["SSIM_RGB"] is not None:
            e4.metric("SSIM (RGB)", f"{ecr_metric['SSIM_RGB']:.4f}")
        else:
            e4.metric("SSIM (RGB)", "Unavailable")
            st.caption("Install scikit-image to enable SSIM.")

        e5.metric("SAM", f"{ecr_metric['SAM_degrees']:.3f}°")

        mae_change = (
            (1.0 - ecr_metric["MAE"] / cloudy_metric["MAE"]) * 100.0
            if cloudy_metric["MAE"] > 0 else 0.0
        )
        rmse_change = (
            (1.0 - ecr_metric["RMSE"] / cloudy_metric["RMSE"]) * 100.0
            if cloudy_metric["RMSE"] > 0 else 0.0
        )
        psnr_gain = ecr_metric["PSNR_dB"] - cloudy_metric["PSNR_dB"]

        st.subheader("Change Compared With Cloudy Input")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE change", f"{mae_change:+.2f}%")
        c2.metric("RMSE change", f"{rmse_change:+.2f}%")
        c3.metric("PSNR change", f"{psnr_gain:+.3f} dB")

        if ecr_metric["PSNR_dB"] > cloudy_metric["PSNR_dB"]:
            st.success(
                "On this reference-based evaluation, ECRformer improves the PSNR over the cloudy input."
            )
        else:
            st.warning(
                "On this reference-based evaluation, ECRformer does not improve the PSNR over the cloudy input. "
                "Treat this as a baseline result, not live-scene accuracy."
            )

        if evaluation["cloud_region_metrics"] is not None or evaluation["clear_region_metrics"] is not None:
            st.divider()
            st.subheader("Region-Wise Reconstruction")

            if evaluation["cloud_region_metrics"] is not None:
                st.markdown("**Known cloud region**")
                rows = []
                for label, metrics in evaluation["cloud_region_metrics"].items():
                    rows.append({
                        "Method": label,
                        "MAE": metrics["MAE"],
                        "RMSE": metrics["RMSE"],
                        "PSNR (dB)": metrics["PSNR_dB"],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            if evaluation["clear_region_metrics"] is not None:
                st.markdown("**Known clear region**")
                rows = []
                for label, metrics in evaluation["clear_region_metrics"].items():
                    rows.append({
                        "Method": label,
                        "MAE": metrics["MAE"],
                        "RMSE": metrics["RMSE"],
                        "PSNR (dB)": metrics["PSNR_dB"],
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if evaluation["cloud_detection_metrics"] is not None:
            st.divider()
            st.subheader("Cloud Detection Metrics")

            cm = evaluation["cloud_detection_metrics"]
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Precision", f"{cm['Precision']:.2%}")
            d2.metric("Recall", f"{cm['Recall']:.2%}")
            d3.metric("F1", f"{cm['F1']:.2%}")
            d4.metric("IoU", f"{cm['IoU']:.2%}")

            d5, d6 = st.columns(2)
            d5.metric("True Cloud Coverage", f"{cm['true_coverage_pct']:.2f}%")
            d6.metric("Predicted Cloud Coverage", f"{cm['pred_coverage_pct']:.2f}%")

            st.caption(
                "Cloud-detection metrics are shown only when both true and predicted cloud masks are available."
            )

        evaluation_report = {
            "reconstruction": {
                "cloudy_input_vs_reference": cloudy_metric,
                "ecrformer_vs_reference": ecr_metric,
                "change_vs_cloudy": {
                    "mae_percent_change": mae_change,
                    "rmse_percent_change": rmse_change,
                    "psnr_gain_db": psnr_gain,
                },
                "cloud_region": evaluation["cloud_region_metrics"],
                "clear_region": evaluation["clear_region_metrics"],
            },
            "cloud_detection": evaluation["cloud_detection_metrics"],
            "diagnostics": evaluation["diagnostics"],
        }

        st.download_button(
            "Download Evaluation Report (JSON)",
            data=json.dumps(evaluation_report, indent=2),
            file_name="model_evaluation_report.json",
            mime="application/json",
            use_container_width=True,
            key="eval_report_download",
        )

        with st.expander("Raw Evaluation Diagnostics", expanded=False):
            st.json(evaluation["diagnostics"])
