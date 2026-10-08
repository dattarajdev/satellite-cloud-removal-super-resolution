"""
Satellite Cloud Removal & Reconstruction - Streamlit Application.

Integrates the canonical backend inference pipeline with a full user-facing
satellite acquisition workflow:
  Tab 1: Live Acquisition (Location Search -> Geocoding -> AOI Generation -> CDSE Catalogue -> Acquisition -> Harmonization -> ECRformer)
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
    OpenSRService,
    OpenSRConfig,
    OpenSRResult,
)
from backend.services.satellite.base import BoundingBox, S2Scene, S1Scene
from backend.services.satellite.geocoding import (
    GeocodedLocation,
    geocode_location,
    bbox_from_center,
    validate_aoi,
)
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
st.caption("Location Search -> CDSE Acquisition -> Spatial Harmonization -> ECRformer Reconstruction -> 13-Band GeoTIFF")

tab_live, tab_cached = st.tabs(["Live Acquisition", "Cached Phase 2 Baseline"])

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
                st.session_state["cached_ecr_result"] = result_c
                st.session_state.pop("cached_sr_result", None)
            except Exception as exc:
                st.error(f"Inference failed: {exc}")

        cached_res: Optional[InferenceResult] = st.session_state.get("cached_ecr_result")
        if cached_res is not None:
            d = cached_res.diagnostics
            st.success(f"ECRformer Reconstruction Successful ({d.execution_time_seconds:.2f}s on {d.device})")

            r1, r2 = st.columns(2)
            with r1:
                st.image(cached_res.cloudy_rgb, caption="Input Cloudy S2", use_container_width=True)
            with r2:
                st.image(cached_res.prediction_rgb, caption="ECRformer Reconstructed Cloud-Free (13 bands | 10 m)", use_container_width=True)

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
                    data=cached_res.to_geotiff_bytes(),
                    file_name="s2_cloud_free_13band.tif",
                    mime="image/tiff",
                    use_container_width=True,
                    key="c_dl_tif",
                )
            with dc2:
                st.download_button(
                    "Download Preview PNG",
                    data=cached_res.to_png_bytes(),
                    file_name="s2_cloud_free_preview.png",
                    mime="image/png",
                    use_container_width=True,
                    key="c_dl_png",
                )

            # Optional OpenSR Super-Resolution Stage
            st.divider()
            st.markdown("### Optional Post-Processing: Diffusion Super-Resolution (OpenSR 4×)")
            st.caption("Apply OpenSR LDSR-S2 4× diffusion super-resolution to bands B04, B03, B02, B08 (10 m -> 2.5 m resolution).")
            st.info("ℹ️ **OpenSR output: B04/B03/B02/B08 only.** The 13-band ECRformer GeoTIFF above remains the authoritative full-spectrum scientific reconstruction product.")

            if st.button("✨ Generate Super-Resolved RGBN (OpenSR 4×)", key="cached_btn_sr", use_container_width=True):
                with st.spinner("Executing OpenSR 4× latent diffusion super-resolution (RTX 3050 CUDA)..."):
                    try:
                        temp_dir = ROOT / "reconstructed_images" / "sessions" / "cached_baseline"
                        temp_dir.mkdir(parents=True, exist_ok=True)
                        temp_input = temp_dir / "ecrformer_13band_10m.tif"
                        cached_res.save_geotiff(temp_input)

                        sr_svc = OpenSRService()
                        sr_res = sr_svc.run(input_tif_path=temp_input, output_dir=temp_dir)
                        st.session_state["cached_sr_result"] = sr_res
                        st.success("OpenSR Super-Resolution completed successfully!")
                    except Exception as exc:
                        st.error(f"OpenSR Super-Resolution failed: {exc}")

            cached_sr: Optional[OpenSRResult] = st.session_state.get("cached_sr_result")
            if cached_sr is not None:
                st.subheader("OpenSR Super-Resolution (4 bands | 2.5 m | 4×)")
                sr_d = cached_sr.diagnostics

                sr_v1, sr_v2 = st.columns(2)
                with sr_v1:
                    st.image(cached_res.prediction_rgb, caption="ECRformer 10 m Baseline (RGB)", use_container_width=True)
                with sr_v2:
                    st.image(cached_sr.sr_rgb_preview, caption="OpenSR 2.5 m Super-Resolved (RGB: B04, B03, B02)", use_container_width=True)

                srm1, srm2, srm3, srm4, srm5 = st.columns(5)
                srm1.metric("Resolution", f"{sr_d.output_pixel_size_m} m")
                srm2.metric("Scale Factor", f"{sr_d.scale_factor}x")
                srm3.metric("Output Dimensions", f"{sr_d.output_width}x{sr_d.output_height}")
                srm4.metric("Device", sr_d.execution_device)
                srm5.metric("SR Time", f"{sr_d.execution_time_seconds:.1f}s")

                srdl1, srdl2, srdl3 = st.columns(3)
                with srdl1:
                    st.download_button(
                        "Download 4-Band SR GeoTIFF (2.5m)",
                        data=cached_sr.to_geotiff_bytes(),
                        file_name="s2_sr_4band_2.5m.tif",
                        mime="image/tiff",
                        use_container_width=True,
                        key="c_sr_dl_tif",
                    )
                with srdl2:
                    st.download_button(
                        "Download 2.5m RGB Preview PNG",
                        data=cached_sr.to_png_bytes(),
                        file_name="s2_sr_2.5m_preview.png",
                        mime="image/png",
                        use_container_width=True,
                        key="c_sr_dl_png",
                    )
                with srdl3:
                    st.download_button(
                        "Download OpenSR Diagnostics JSON",
                        data=json.dumps(sr_d.to_dict(), indent=2),
                        file_name="opensr_diagnostics.json",
                        mime="application/json",
                        use_container_width=True,
                        key="c_sr_dl_json",
                    )

                with st.expander("OpenSR Technical Diagnostics & Metadata", expanded=False):
                    st.json(sr_d.to_dict())
# ===========================================================================
# TAB 1: Live Acquisition Workflow
# ===========================================================================
with tab_live:
    st.header("Live CDSE Acquisition Workflow")
    st.caption("Search a location or landmark, specify AOI extent, query CDSE Sentinel collections, harmonize, and reconstruct.")

    # ---- Canonical State Initialization ----
    if "selected_bbox" not in st.session_state:
        # Default: Rome benchmark (cached data verified)
        st.session_state["selected_bbox"] = BoundingBox(west=12.4500, south=41.9000, east=12.4730, north=41.9230)
    if "resolved_location" not in st.session_state:
        st.session_state["resolved_location"] = GeocodedLocation(
            query="Rome, Italy",
            display_name="Rome, Lazio, Italy (Benchmark Cache)",
            latitude=41.9115,
            longitude=12.4615,
        )
    if "aoi_size_km" not in st.session_state:
        st.session_state["aoi_size_km"] = 2.0
    if "aoi_source" not in st.session_state:
        st.session_state["aoi_source"] = "Default Benchmark (Rome)"
    if "map_center" not in st.session_state:
        st.session_state["map_center"] = (41.9115, 12.4615)

    cur_bbox: BoundingBox = st.session_state["selected_bbox"]
    cur_loc: Optional[GeocodedLocation] = st.session_state.get("resolved_location")

    # ------------------------------------------------------------------
    # STEP 1: Location Search & AOI Selection
    # ------------------------------------------------------------------
    with st.expander("Step 1 - Location Search & Area of Interest (AOI)", expanded=True):
        st.subheader("1. Search Location, City, Address or Landmark")

        search_col1, search_col2 = st.columns([4, 1])
        with search_col1:
            search_query = st.text_input(
                "Search query",
                value="",
                placeholder="e.g. Mumbai, India | Airoli, Navi Mumbai | Neral, Maharashtra | Rome, Italy | Gateway of India",
                label_visibility="collapsed",
                key="loc_search_input",
            )
        with search_col2:
            search_submitted = st.button("Search Location", type="primary", use_container_width=True, key="btn_do_search")

        if search_submitted:
            if search_query.strip():
                with st.spinner(f"Geocoding '{search_query.strip()}'..."):
                    loc = geocode_location(search_query.strip())
                if loc is not None:
                    st.session_state["resolved_location"] = loc
                    st.session_state["map_center"] = (loc.latitude, loc.longitude)
                    st.session_state["selected_bbox"] = bbox_from_center(
                        loc.latitude, loc.longitude, st.session_state["aoi_size_km"]
                    )
                    st.session_state["aoi_source"] = f"Geocoded: {loc.query}"
                    st.rerun()
                else:
                    st.error(f"Could not find location for '{search_query}'. Please check the query or enter coordinates manually.")
            else:
                st.warning("Please enter a location name, address, or landmark.")

        # Quick Preset Buttons
        st.caption("Quick Location Presets:")
        pcol1, pcol2, pcol3, pcol4, pcol5, pcol6 = st.columns(6)
        with pcol1:
            if st.button("Mumbai, India", key="p_mum", use_container_width=True):
                loc = geocode_location("Mumbai, India")
                if loc:
                    st.session_state["resolved_location"] = loc
                    st.session_state["map_center"] = (loc.latitude, loc.longitude)
                    st.session_state["selected_bbox"] = bbox_from_center(loc.latitude, loc.longitude, st.session_state["aoi_size_km"])
                    st.session_state["aoi_source"] = "Preset: Mumbai, India"
                    st.rerun()
        with pcol2:
            if st.button("Airoli, Navi Mumbai", key="p_air", use_container_width=True):
                loc = geocode_location("Airoli, Navi Mumbai")
                if loc:
                    st.session_state["resolved_location"] = loc
                    st.session_state["map_center"] = (loc.latitude, loc.longitude)
                    st.session_state["selected_bbox"] = bbox_from_center(loc.latitude, loc.longitude, st.session_state["aoi_size_km"])
                    st.session_state["aoi_source"] = "Preset: Airoli, Navi Mumbai"
                    st.rerun()
        with pcol3:
            if st.button("Neral, MH", key="p_ner", use_container_width=True):
                loc = geocode_location("Neral, Maharashtra")
                if loc:
                    st.session_state["resolved_location"] = loc
                    st.session_state["map_center"] = (loc.latitude, loc.longitude)
                    st.session_state["selected_bbox"] = bbox_from_center(loc.latitude, loc.longitude, st.session_state["aoi_size_km"])
                    st.session_state["aoi_source"] = "Preset: Neral, Maharashtra"
                    st.rerun()
        with pcol4:
            if st.button("Gateway of India", key="p_gat", use_container_width=True):
                loc = geocode_location("Gateway of India")
                if loc:
                    st.session_state["resolved_location"] = loc
                    st.session_state["map_center"] = (loc.latitude, loc.longitude)
                    st.session_state["selected_bbox"] = bbox_from_center(loc.latitude, loc.longitude, st.session_state["aoi_size_km"])
                    st.session_state["aoi_source"] = "Preset: Gateway of India"
                    st.rerun()
        with pcol5:
            if st.button("Rome (Cached)", key="p_rom", use_container_width=True):
                st.session_state["selected_bbox"] = BoundingBox(west=12.4500, south=41.9000, east=12.4730, north=41.9230)
                st.session_state["resolved_location"] = GeocodedLocation(
                    query="Rome, Italy",
                    display_name="Rome Benchmark, Lazio, Italy (Cache Verified)",
                    latitude=41.9115,
                    longitude=12.4615,
                )
                st.session_state["map_center"] = (41.9115, 12.4615)
                st.session_state["aoi_source"] = "Preset: Rome Benchmark"
                st.rerun()
        with pcol6:
            if st.button("Po Valley", key="p_pov", use_container_width=True):
                st.session_state["selected_bbox"] = BoundingBox(west=10.5000, south=44.8000, east=10.5300, north=44.8300)
                st.session_state["resolved_location"] = GeocodedLocation(
                    query="Po Valley Benchmark",
                    display_name="Po Valley Agricultural Benchmark, Italy",
                    latitude=44.8150,
                    longitude=10.5150,
                )
                st.session_state["map_center"] = (44.8150, 10.5150)
                st.session_state["aoi_source"] = "Preset: Po Valley Benchmark"
                st.rerun()

        st.divider()

        # ---- AOI Size Selection ----
        st.subheader("2. Choose AOI Dimension")
        size_c1, size_c2 = st.columns([3, 2])

        size_presets = {
            "1 km": 1.0,
            "2 km (Standard)": 2.0,
            "5 km": 5.0,
            "10 km": 10.0,
            "Custom": None,
        }

        cur_sz = st.session_state.get("aoi_size_km", 2.0)
        default_idx = 1
        if abs(cur_sz - 1.0) < 0.1:
            default_idx = 0
        elif abs(cur_sz - 2.0) < 0.1:
            default_idx = 1
        elif abs(cur_sz - 5.0) < 0.1:
            default_idx = 2
        elif abs(cur_sz - 10.0) < 0.1:
            default_idx = 3
        else:
            default_idx = 4

        with size_c1:
            size_choice = st.radio(
                "AOI Extent Size Preset",
                list(size_presets.keys()),
                index=default_idx,
                horizontal=True,
                key="radio_aoi_size",
            )

        new_size_km = cur_sz
        with size_c2:
            if size_choice == "Custom":
                new_size_km = st.number_input(
                    "Custom Size (km)",
                    min_value=0.5,
                    max_value=50.0,
                    value=float(cur_sz),
                    step=0.5,
                    key="custom_size_val",
                )
            else:
                new_size_km = size_presets[size_choice]

        if abs(new_size_km - cur_sz) > 1e-4:
            st.session_state["aoi_size_km"] = new_size_km
            center_lat, center_lon = st.session_state["map_center"]
            st.session_state["selected_bbox"] = bbox_from_center(center_lat, center_lon, new_size_km)
            st.rerun()

        cur_bbox = st.session_state["selected_bbox"]
        cur_loc = st.session_state.get("resolved_location")
        center_lat = (cur_bbox.south + cur_bbox.north) / 2.0
        center_lon = (cur_bbox.west + cur_bbox.east) / 2.0
        approx_w_km = cur_bbox.approx_width_m() / 1000.0
        approx_h_km = cur_bbox.approx_height_m() / 1000.0

        # ---- Active AOI Summary Info Box ----
        info_c1, info_c2 = st.columns([3, 2])
        with info_c1:
            loc_label = cur_loc.display_name if cur_loc else "Custom Region"
            st.markdown(f"**Target Location:** {loc_label}")
            st.markdown(f"**Center Coordinates:** `{center_lat:.5f}° N, {center_lon:.5f}° E`")
            st.markdown(f"**Extent Dimension:** `~{approx_w_km:.2f} km × {approx_h_km:.2f} km` (Source: *{st.session_state['aoi_source']}*)")
        with info_c2:
            st.markdown(
                f"**Bounding Box (WGS84):**\n"
                f"- West: `{cur_bbox.west:.5f}` | East: `{cur_bbox.east:.5f}`\n"
                f"- South: `{cur_bbox.south:.5f}` | North: `{cur_bbox.north:.5f}`"
            )

        st.divider()        # ---- Interactive Map & Manual Controls ----
        map_col, draw_col = st.columns([3, 2])

        with map_col:
            st.subheader("3. Interactive Map")
            st.caption("The map shows the active AOI rectangle and center pin. Use the rectangle tool to draw a custom area if desired.")
            try:
                import folium
                from folium.plugins import Draw
                from streamlit_folium import st_folium

                if max(approx_w_km, approx_h_km) <= 1.5:
                    zoom_lvl = 15
                elif max(approx_w_km, approx_h_km) <= 3.0:
                    zoom_lvl = 14
                elif max(approx_w_km, approx_h_km) <= 7.0:
                    zoom_lvl = 13
                elif max(approx_w_km, approx_h_km) <= 15.0:
                    zoom_lvl = 12
                else:
                    zoom_lvl = 11

                f_map = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_lvl)

                folium.Marker(
                    location=[center_lat, center_lon],
                    popup=cur_loc.display_name if cur_loc else "AOI Center",
                    tooltip="AOI Center",
                    icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
                ).add_to(f_map)

                folium.Rectangle(
                    bounds=[[cur_bbox.south, cur_bbox.west], [cur_bbox.north, cur_bbox.east]],
                    color="#FF3333",
                    fill=True,
                    fill_opacity=0.18,
                    weight=2,
                    tooltip=f"Active AOI (~{approx_w_km:.2f} km x {approx_h_km:.2f} km)",
                ).add_to(f_map)

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
                ).add_to(f_map)

                map_result = st_folium(f_map, width=540, height=390, key="aoi_folium_map")
            except Exception as e:
                st.warning(f"Interactive map component notice: {e}")
                map_result = None

        with draw_col:
            st.subheader("Manual Refinement")
            st.caption("Optionally apply a rectangle drawn on the map or enter coordinates directly.")

            drawn_info = None
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
                    import math
                    d_w_km = (drawn_e - drawn_w) * 111.32 * math.cos(math.radians((drawn_s + drawn_n) / 2))
                    d_h_km = (drawn_n - drawn_s) * 110.54
                    drawn_info = (drawn_w, drawn_s, drawn_e, drawn_n, d_w_km, d_h_km)

            if drawn_info:
                dw, ds, de, dn, dw_km, dh_km = drawn_info
                st.info(
                    f"**Drawn on map:**\n"
                    f"- W: `{dw:.4f}` | S: `{ds:.4f}` | E: `{de:.4f}` | N: `{dn:.4f}`\n"
                    f"- Size: `~{dw_km:.2f} km × {dh_km:.2f} km`"
                )
                if st.button("Apply Drawn Rectangle as AOI", type="primary", key="btn_apply_drawn", use_container_width=True):
                    st.session_state["selected_bbox"] = BoundingBox(west=dw, south=ds, east=de, north=dn)
                    st.session_state["resolved_location"] = None
                    st.session_state["aoi_source"] = "Custom Map Drawing"
                    st.session_state["map_center"] = ((ds + dn) / 2.0, (dw + de) / 2.0)
                    st.rerun()
            else:
                st.caption("To draw a custom area, click the rectangle tool in the map toolbar and drag over your target area.")

            with st.expander("Advanced: Enter coordinates manually", expanded=False):
                st.write("Enter exact WGS84 bounding box coordinates:")
                c_mw, c_ms = st.columns(2)
                with c_mw:
                    in_w = st.number_input("West (min lon)", value=float(cur_bbox.west), step=0.001, format="%.4f", key="manual_coord_w")
                with c_ms:
                    in_s = st.number_input("South (min lat)", value=float(cur_bbox.south), step=0.001, format="%.4f", key="manual_coord_s")
                c_me, c_mn = st.columns(2)
                with c_me:
                    in_e = st.number_input("East (max lon)", value=float(cur_bbox.east), step=0.001, format="%.4f", key="manual_coord_e")
                with c_mn:
                    in_n = st.number_input("North (max lat)", value=float(cur_bbox.north), step=0.001, format="%.4f", key="manual_coord_n")

                if st.button("Apply Manual Coordinates", key="btn_apply_manual", use_container_width=True):
                    st.session_state["selected_bbox"] = BoundingBox(west=in_w, south=in_s, east=in_e, north=in_n)
                    st.session_state["resolved_location"] = None
                    st.session_state["aoi_source"] = "Manual Coordinates"
                    st.session_state["map_center"] = ((in_s + in_n) / 2.0, (in_w + in_e) / 2.0)
                    st.rerun()

            output_px = st.select_slider(
                "Acquisition Download Grid",
                options=[128, 256, 512],
                value=256,
                key="slider_raw_grid_px",
            )
            st.caption("Raw acquisition dimension. Downstream spatial harmonization resamples to the authoritative 10m UTM grid.")

        # ---- Final AOI Validation ----
        bbox_valid, bbox_err = validate_aoi(cur_bbox)
        if not bbox_valid:
            st.error(f"AOI Validation Error: {bbox_err}")
            bbox = None
        else:
            bbox = cur_bbox
            output_size_px = (output_px, output_px)
            st.success(
                f"**Active AOI Confirmed:** W={bbox.west:.4f}, S={bbox.south:.4f}, E={bbox.east:.4f}, N={bbox.north:.4f} | "
                f"~{approx_w_km:.2f} km × {approx_h_km:.2f} km | Download Grid: {output_px}×{output_px} px"
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
            st.session_state["selected_bbox"] = BoundingBox(west=12.4500, south=41.9000, east=12.4730, north=41.9230)
            st.session_state["resolved_location"] = GeocodedLocation(
                query="Rome, Italy",
                display_name="Rome Benchmark, Lazio, Italy (Cache Verified)",
                latitude=41.9115,
                longitude=12.4615,
            )
            st.session_state["map_center"] = (41.9115, 12.4615)
            st.session_state["aoi_source"] = "Preset: Rome Benchmark (Cached)"
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
            st.success("Loaded verified Rome cached acquisition metadata and set Rome AOI.")
            st.rerun()

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
                    st.session_state.pop("live_sr_result", None)
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

        # Optional OpenSR Super-Resolution Stage (Live)
        st.divider()
        st.subheader("Optional Post-Processing: Diffusion Super-Resolution (OpenSR 4×)")
        st.caption("Apply OpenSR LDSR-S2 4× diffusion super-resolution to bands B04, B03, B02, B08 (10 m -> 2.5 m resolution).")
        st.info("ℹ️ **OpenSR output: B04/B03/B02/B08 only.** The 13-band ECRformer GeoTIFF above remains the authoritative full-spectrum scientific reconstruction product.")

        if st.button("✨ Generate Super-Resolved RGBN (OpenSR 4×)", key="live_btn_sr", use_container_width=True):
            with st.spinner("Executing OpenSR 4× latent diffusion super-resolution (RTX 3050 CUDA)..."):
                try:
                    live_out_dir = ROOT / "reconstructed_images" / "sessions" / f"session_{date_str}"
                    live_out_dir.mkdir(parents=True, exist_ok=True)
                    live_ecr_tif = live_out_dir / f"s2_ecrformer_{date_str}_13band.tif"
                    live_res.save_geotiff(live_ecr_tif)

                    sr_svc = OpenSRService()
                    sr_res = sr_svc.run(input_tif_path=live_ecr_tif, output_dir=live_out_dir)
                    st.session_state["live_sr_result"] = sr_res
                    st.success("OpenSR Super-Resolution completed successfully!")
                except Exception as exc:
                    st.error(f"OpenSR Super-Resolution failed: {exc}")

        live_sr: Optional[OpenSRResult] = st.session_state.get("live_sr_result")
        if live_sr is not None:
            st.subheader("OpenSR Super-Resolution (4 bands | 2.5 m | 4×)")
            sr_d = live_sr.diagnostics

            sr_lv1, sr_lv2 = st.columns(2)
            with sr_lv1:
                st.image(live_res.prediction_rgb, caption="ECRformer 10 m Baseline (RGB)", use_container_width=True)
            with sr_lv2:
                st.image(live_sr.sr_rgb_preview, caption="OpenSR 2.5 m Super-Resolved (RGB: B04, B03, B02)", use_container_width=True)

            lsrm1, lsrm2, lsrm3, lsrm4, lsrm5 = st.columns(5)
            lsrm1.metric("Resolution", f"{sr_d.output_pixel_size_m} m")
            lsrm2.metric("Scale Factor", f"{sr_d.scale_factor}x")
            lsrm3.metric("Output Dimensions", f"{sr_d.output_width}x{sr_d.output_height}")
            lsrm4.metric("Device", sr_d.execution_device)
            lsrm5.metric("SR Time", f"{sr_d.execution_time_seconds:.1f}s")

            lsrdl1, lsrdl2, lsrdl3 = st.columns(3)
            with lsrdl1:
                st.download_button(
                    "Download 4-Band SR GeoTIFF (2.5m)",
                    data=live_sr.to_geotiff_bytes(),
                    file_name=f"s2_sr_{date_str}_4band_2.5m.tif",
                    mime="image/tiff",
                    use_container_width=True,
                    key="live_sr_dl_tif",
                )
            with lsrdl2:
                st.download_button(
                    "Download 2.5m RGB Preview PNG",
                    data=live_sr.to_png_bytes(),
                    file_name=f"s2_sr_{date_str}_preview.png",
                    mime="image/png",
                    use_container_width=True,
                    key="live_sr_dl_png",
                )
            with lsrdl3:
                st.download_button(
                    "Download OpenSR Diagnostics JSON",
                    data=json.dumps(sr_d.to_dict(), indent=2),
                    file_name=f"opensr_diagnostics_{date_str}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="live_sr_dl_json",
                )

            with st.expander("OpenSR Technical Diagnostics & Metadata", expanded=False):
                st.json(sr_d.to_dict())