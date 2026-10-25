"""
Streamlit UI Components for Per-Image Performance Metrics and Diagnostics.

Provides separate, dedicated, and scientifically rigorous metrics displays for:
1. ECRformer Cloud Reconstruction (13 bands | 10 m):
   - Reference-based evaluation (MAE, RMSE, PSNR, SSIM, SAM, LPIPS) when cloud-free ground truth is supplied.
   - Comprehensive 13-band breakdown table and aggregate metrics.
   - Reference-free diagnostics (per-band min/max/mean, valid-pixel coverage, NaN/Inf counts) when no reference is available.
   - Clear UI status: "Reference-based image-quality metrics unavailable: no cloud-free reference supplied."
2. OpenSR Diffusion Super-Resolution (4 bands: B04, B03, B02, B08 | 2.5 m | 4x):
   - Spatial and pipeline diagnostics (scale, dimensions, pixel size, CRS, geographic bounds, sampling steps, device).
   - Clear UI status: "Reference-based super-resolution accuracy metrics unavailable: no high-resolution ground truth supplied."
   - Optional 10 m consistency evaluation against ECRformer baseline labeled strictly:
     "10 m consistency with ECRformer output — not ground-truth accuracy."
3. Per-image downloadable JSON diagnostics export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import streamlit as st

from .evaluation_metrics import (
    ReferenceBasedMetrics,
    ReferenceFreeDiagnostics,
    OpenSRConsistencyDiagnostics,
    compute_reference_based_metrics,
    compute_reference_free_diagnostics,
    compute_opensr_10m_consistency,
    build_image_diagnostics_export,
    S2_BAND_NAMES,
    OPENSR_BAND_NAMES,
)
from .inference_service import InferenceResult
from .super_resolution.opensr_service import OpenSRResult

BAND_DESCRIPTIONS = {
    "B01": "Coastal aerosol (443 nm)",
    "B02": "Blue (490 nm)",
    "B03": "Green (560 nm)",
    "B04": "Red (665 nm)",
    "B05": "Vegetation Red Edge (705 nm)",
    "B06": "Vegetation Red Edge (740 nm)",
    "B07": "Vegetation Red Edge (783 nm)",
    "B08": "NIR (842 nm)",
    "B8A": "Narrow NIR (865 nm)",
    "B09": "Water vapour (940 nm)",
    "B10": "SWIR - Cirrus (1375 nm)",
    "B11": "SWIR (1610 nm)",
    "B12": "SWIR (2190 nm)",
}


def render_ecrformer_metrics_ui(
    result: InferenceResult,
    image_id: str,
    key_prefix: str,
    allow_reference_upload: bool = True,
) -> None:
    """Render dedicated metrics and diagnostics section for ECRformer."""
    st.markdown("### 📊 ECRformer Performance Metrics & Scientific Diagnostics")
    d = result.diagnostics

    # 1. Pipeline metadata cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stage", "ECRformer")
    c2.metric("Dimensions", f"{d.output_shape[2]}x{d.output_shape[1]} px")
    c3.metric("Bands", f"{d.output_shape[0]} Bands")
    c4.metric("Resolution", f"{d.pixel_size_m} m")
    c5.metric("Execution", f"{d.execution_time_seconds:.2f}s ({d.device.split(':')[0].upper()})")

    # 2. Reference-based evaluation section (Optional upload / compute)
    if allow_reference_upload:
        with st.expander("🎯 Cloud-Free Reference Evaluation (Ground Truth or Temporal)", expanded=False):
            st.caption(
                "Upload a verified cloud-free Sentinel-2 GeoTIFF (13 bands, matching AOI extent) "
                "to evaluate quantitative reconstruction metrics (MAE, RMSE, PSNR, SSIM, SAM, LPIPS). "
                "The target is used strictly for evaluation and is never fed into inference."
            )
            u_c1, u_c2 = st.columns(2)
            with u_c1:
                ref_type_label = st.radio(
                    "Reference Classification",
                    ["Paired Cloud-Free Ground Truth", "Clear Temporal Reference (Different Date)"],
                    key=f"{key_prefix}_ref_type",
                )
            with u_c2:
                scale_label = st.selectbox(
                    "Radiometric Scale Convention",
                    ["Auto-detect", "0–10000 DN (Standard S2)", "0–1 Normalized Reflectance"],
                    key=f"{key_prefix}_scale_mode",
                )

            ref_date_val = None
            if "Temporal" in ref_type_label:
                ref_date_val = st.text_input(
                    "Reference Acquisition Date (YYYY-MM-DD, optional)",
                    value="",
                    key=f"{key_prefix}_ref_date",
                    placeholder="e.g. 2024-09-08",
                )

            ref_file = st.file_uploader(
                "Upload 13-band Sentinel-2 reference GeoTIFF",
                type=["tif", "tiff"],
                key=f"{key_prefix}_ref_upload",
                help="Requires 13 bands. If grids differ, it will be geospatially reprojected.",
            )

            if ref_file is not None:
                if st.button("Calculate Reference-Based Metrics", key=f"{key_prefix}_btn_calc_ref"):
                    try:
                        with st.spinner("Validating, aligning, and computing metrics across all 13 bands..."):
                            scale_mode = "0-10000" if "10000" in scale_label else ("0-1" if "0–1" in scale_label else "auto")
                            ref_type = "temporal_reference" if "Temporal" in ref_type_label else "paired_ground_truth"

                            metrics = result.evaluate_with_reference(
                                reference_target=ref_file,
                                scale_mode=scale_mode,
                                reference_type=ref_type,
                                acquisition_date=ref_date_val if ref_date_val else None,
                            )
                            st.session_state[f"{key_prefix}_ref_metrics"] = metrics
                            st.success("Reference-based metrics computed successfully!")
                    except Exception as e:
                        st.error(f"Error computing reference metrics: {e}")

    # Check if reference metrics are attached
    ref_metrics = result.reference_metrics or st.session_state.get(f"{key_prefix}_ref_metrics")

    if ref_metrics is not None and ref_metrics.available:
        if ref_metrics.reference_type == "temporal_reference":
            st.warning(
                f"⚠️ **{ref_metrics.disclaimer}**\n\n"
                f"**Reference Source:** `{ref_metrics.reference_source}` | "
                f"**Acquisition Date:** `{ref_metrics.reference_date or 'Different Date'}`"
            )
            st.markdown("#### ⏳ Temporal-Reference Comparison Metrics")
        else:
            st.success(f"✅ {ref_metrics.status}")
            st.markdown("#### 🎯 Reference-Based Reconstruction Metrics (Ground Truth)")

        # Aggregate metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Aggregate MAE", f"{ref_metrics.aggregate_mae:.4f}")
        m2.metric("Aggregate RMSE", f"{ref_metrics.aggregate_rmse:.4f}")
        psnr_display = f"{ref_metrics.aggregate_psnr_db:.2f} dB" if not np.isinf(ref_metrics.aggregate_psnr_db) else "∞ (Identical)"
        m3.metric("Aggregate PSNR", psnr_display, help=f"Data range: {ref_metrics.data_range} (normalized reflectance [0.0, 1.0])")
        ssim_display = f"{ref_metrics.ssim_rgb:.4f}" if ref_metrics.ssim_rgb is not None else "N/A"
        m4.metric("SSIM (RGB)", ssim_display, help=ref_metrics.ssim_rgb_convention)
        sam_display = f"{ref_metrics.sam_degrees:.2f}°" if not np.isnan(ref_metrics.sam_degrees) else "N/A"
        m5.metric("SAM", sam_display, help="Spectral Angle Mapper averaged over valid pixels in degrees")

        if ref_metrics.lpips_score is not None:
            st.info(f"👁️ **LPIPS (Perceptual Distance):** {ref_metrics.lpips_score:.4f} ({ref_metrics.lpips_note})")
        elif ref_metrics.lpips_note:
            st.caption(f"ℹ️ {ref_metrics.lpips_note}")

        # Comprehensive 13-Band Breakdown Table
        st.markdown("##### Per-Band Metrics Breakdown (13 Sentinel-2 Bands)")
        table_rows = []
        for name in S2_BAND_NAMES:
            mae_v = ref_metrics.per_band_mae.get(name, 0.0)
            rmse_v = ref_metrics.per_band_rmse.get(name, 0.0)
            psnr_v = ref_metrics.per_band_psnr_db.get(name, 0.0)
            psnr_str = f"{psnr_v:.2f}" if not np.isinf(psnr_v) else "∞"
            ssim_v = ref_metrics.per_band_ssim.get(name) if ref_metrics.per_band_ssim else None
            ssim_str = f"{ssim_v:.4f}" if ssim_v is not None else "N/A"

            table_rows.append({
                "Band": name,
                "Description": BAND_DESCRIPTIONS.get(name, ""),
                "MAE": f"{mae_v:.4f}",
                "RMSE": f"{rmse_v:.4f}",
                "PSNR (dB)": psnr_str,
                "SSIM (Per-Band)": ssim_str,
            })

        # Add Aggregate Mean row
        table_rows.append({
            "Band": "Aggregate (Mean)",
            "Description": "Mean across all 13 Sentinel-2 bands",
            "MAE": f"{ref_metrics.aggregate_mae:.4f}",
            "RMSE": f"{ref_metrics.aggregate_rmse:.4f}",
            "PSNR (dB)": f"{ref_metrics.aggregate_psnr_db:.2f}" if not np.isinf(ref_metrics.aggregate_psnr_db) else "∞",
            "SSIM (Per-Band)": f"{ref_metrics.ssim_rgb:.4f}" if ref_metrics.ssim_rgb is not None else "N/A",
        })
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    else:
        # Reference-free status & diagnostics
        st.info("ℹ️ **Reference-based image-quality metrics unavailable: no cloud-free reference supplied.**")

        with st.expander("🔍 Reference-Free Data Quality & Coverage Diagnostics", expanded=True):
            rf_c1, rf_c2, rf_c3, rf_c4 = st.columns(4)
            rf_c1.metric("Valid Coverage", f"{d.to_dict().get('valid_percentage', 100.0)}%", f"{d.valid_pixels:,} / {d.total_pixels:,} px")
            rf_c2.metric("NoData / Invalid", f"{d.invalid_pixels:,} px")
            rf_c3.metric("NaN Count", str(d.nan_count))
            rf_c4.metric("Inf Count", str(d.inf_count))

            if d.per_band_stats:
                st.markdown("##### Per-Band Radiometric Distributions (Valid Pixels)")
                stats_rows = []
                for name in S2_BAND_NAMES:
                    b_stat = d.per_band_stats.get(name, {"min": 0.0, "max": 0.0, "mean": 0.0})
                    stats_rows.append({
                        "Band": name,
                        "Description": BAND_DESCRIPTIONS.get(name, ""),
                        "Valid Min": f"{b_stat['min']:.4f}",
                        "Valid Max": f"{b_stat['max']:.4f}",
                        "Valid Mean": f"{b_stat['mean']:.4f}",
                    })
                st.dataframe(pd.DataFrame(stats_rows), use_container_width=True, hide_index=True)

    # 3. Per-image downloadable JSON diagnostics file
    diag_export = result.to_diagnostics_export(image_id=image_id)
    diag_json = json.dumps(diag_export, indent=2)

    st.download_button(
        "📥 Download ECRformer Diagnostics (JSON)",
        data=diag_json,
        file_name=f"ecrformer_diagnostics_{image_id}.json",
        mime="application/json",
        use_container_width=True,
        key=f"{key_prefix}_dl_json_metrics",
    )


def render_opensr_metrics_ui(
    sr_res: OpenSRResult,
    image_id: str,
    key_prefix: str,
) -> None:
    """Render dedicated metrics and diagnostics section for OpenSR super-resolution."""
    st.markdown("### 🔬 OpenSR Diffusion Super-Resolution Performance & Diagnostics")
    sr_d = sr_res.diagnostics

    # 1. Pipeline metadata cards
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Stage", "OpenSR LDSR-S2")
    c2.metric("Scale Factor", f"{sr_d.scale_factor}x")
    c3.metric("Selected Bands", "B04, B03, B02, B08")
    c4.metric("Resolution", f"{sr_d.input_pixel_size_m}m -> {sr_d.output_pixel_size_m}m")
    c5.metric("SR Time", f"{sr_d.execution_time_seconds:.1f}s ({sr_d.execution_device})")

    # 2. Pipeline Geometry & Sampling Specs
    with st.expander("📐 Spatial Geometry & Diffusion Configuration", expanded=False):
        g1, g2, g3 = st.columns(3)
        g1.markdown(f"**Input Dimensions:** `{sr_d.input_width} x {sr_d.input_height} px`")
        g1.markdown(f"**Output Dimensions:** `{sr_d.output_width} x {sr_d.output_height} px`")
        g2.markdown(f"**CRS:** `{sr_d.output_crs}`")
        g2.markdown(f"**DDIM Sampling Steps:** `{sr_d.sampling_steps or 100} steps`")
        if sr_d.geographic_bounds:
            b = sr_d.geographic_bounds
            g3.markdown(f"**Bounds (W, S, E, N):**\n`{b['left']}, {b['bottom']}, {b['right']}, {b['top']}`")

    # 3. Ground-truth limitation notice
    st.info(f"ℹ️ **{sr_d.ground_truth_status}**")
    st.caption(
        "There is currently no native 2.5 m ground-truth reference for Sentinel-2 satellite imagery. "
        "Standard reference-based fidelity metrics (PSNR, SSIM, SAM, MAE, RMSE) are therefore NOT calculated "
        "or claimed against ground truth."
    )

    # 4. Optional 10 m Consistency Evaluation
    if sr_d.consistency_10m is not None and sr_d.consistency_10m.computed:
        cons = sr_d.consistency_10m
        st.markdown(f"#### 🔁 {cons.metric_label}")
        st.caption(cons.disclaimer)

        cm1, cm2, cm3 = st.columns(3)
        cm1.metric("10 m Consistency MAE", f"{cons.aggregate_mae:.4f}")
        cm2.metric("10 m Consistency RMSE", f"{cons.aggregate_rmse:.4f}")
        cm3.metric("Evaluated Pixels", f"{cons.evaluated_pixels:,} px")

        # Per-band consistency table
        cons_rows = []
        for name in OPENSR_BAND_NAMES:
            cons_rows.append({
                "Band": name,
                "Description": BAND_DESCRIPTIONS.get(name, ""),
                "Downsampling Method": cons.downsampling_method,
                "Consistency MAE": f"{cons.per_band_mae.get(name, 0.0):.4f}",
                "Consistency RMSE": f"{cons.per_band_rmse.get(name, 0.0):.4f}",
            })
        st.dataframe(pd.DataFrame(cons_rows), use_container_width=True, hide_index=True)

    # 5. Per-image downloadable JSON diagnostics file
    sr_diag_export = sr_res.to_diagnostics_export(image_id=image_id)
    sr_diag_json = json.dumps(sr_diag_export, indent=2)

    st.download_button(
        "📥 Download OpenSR Diagnostics (JSON)",
        data=sr_diag_json,
        file_name=f"opensr_diagnostics_{image_id}.json",
        mime="application/json",
        use_container_width=True,
        key=f"{key_prefix}_dl_sr_json_metrics",
    )
