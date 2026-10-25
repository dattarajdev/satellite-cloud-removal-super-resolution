"""
CloudVisionAI — Landing Page Component.

Implements the cinematic, geospatial-inspired landing page specified in README_CloudVisionAI_Merged.md:
- Header with CloudVisionAI branding, navigation links, theme toggle, and CTA.
- Wide 2-part hero section: headline, supporting text, CTA buttons, quick stats, and satellite imagery preview.
- 3-step workflow overview: Cloud Reconstruction, Reference Evaluation, and OpenSR Super-Resolution.
- 6-card capabilities grid.
- Actionable workspace launch section.
- Clean footer.
"""
from __future__ import annotations

import base64
from pathlib import Path
import streamlit as st

from ui.theme import get_theme, toggle_theme, get_tokens

ROOT = Path(__file__).resolve().parent.parent
HERO_IMAGE_PATH = ROOT / "assets" / "hero_satellite.jpg"


def _get_base64_image(image_path: Path) -> str:
    """Encode image to base64 for reliable HTML embedding."""
    if not image_path.exists():
        return ""
    with open(image_path, "rb") as f:
        data = f.read()
    return f"data:image/jpeg;base64,{base64.b64encode(data).decode('utf-8')}"


def render_landing_page() -> None:
    """Render the full CloudVisionAI landing page."""
    t = get_tokens()
    current_theme = get_theme()

    # -----------------------------------------------------------------------
    # Top Navigation Bar
    # -----------------------------------------------------------------------
    nav_col1, nav_col2, nav_col3 = st.columns([4, 4, 3])

    with nav_col1:
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:12px; padding: 6px 0;">
                <span style="font-size: 1.8rem;">🛰️</span>
                <div>
                    <div class="cvai-brand">CloudVisionAI</div>
                    <div class="cvai-brand-sub">Intelligent Satellite Image Restoration</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav_col2:
        # Subtle anchor navigation indicators
        st.markdown(
            f"""
            <div style="display:flex; gap:20px; align-items:center; justify-content:center; height:100%; font-size:0.85rem; font-weight:500; color:{t['text_secondary']};">
                <span style="cursor:default;">Overview</span>
                <span style="cursor:default;">Workflow</span>
                <span style="cursor:default;">Capabilities</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with nav_col3:
        btn_c1, btn_c2 = st.columns([1, 1.4])
        with btn_c1:
            theme_icon = "☀️ Light" if current_theme == "dark" else "🌙 Dark"
            if st.button(theme_icon, key="landing_theme_toggle", use_container_width=True):
                toggle_theme()
                st.rerun()
        with btn_c2:
            if st.button("Open Workspace →", type="primary", key="landing_top_ws_btn", use_container_width=True):
                st.session_state["active_view"] = "workspace"
                st.rerun()

    st.markdown("<hr style='margin: 12px 0 24px 0;'/>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Hero Section (2-Column)
    # -----------------------------------------------------------------------
    hero_left, hero_right = st.columns([5.5, 4.5], gap="large")

    with hero_left:
        st.markdown(
            f"""
            <div class="cvai-badge">AI-POWERED EARTH OBSERVATION</div>
            <div class="cvai-hero-hl">
                See Beyond<br><span class="grad">the Clouds.</span>
            </div>
            <div class="cvai-hero-sub">
                Reconstruct cloud-obscured satellite imagery with AI-powered multi-sensor cross-attention
                and optional diffusion-based super-resolution. Fusing Sentinel-1 SAR and Sentinel-2 optical data
                for scientific geospatial workflows.
            </div>
            """,
            unsafe_allow_html=True,
        )

        cta_col1, cta_col2 = st.columns([1.2, 1.3])
        with cta_col1:
            if st.button("Launch Workspace →", type="primary", key="hero_primary_cta", use_container_width=True):
                st.session_state["active_view"] = "workspace"
                st.rerun()
        with cta_col2:
            if st.button("Explore Workflow ↓", key="hero_secondary_cta", use_container_width=True):
                st.session_state["scroll_to_workflow"] = True
                st.rerun()

        # Key metrics row
        st.markdown(
            f"""
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 36px; padding: 18px 12px; background: {t['bg_card']}; border: 1px solid {t['border']}; border-radius: 12px;">
                <div style="text-align:center;">
                    <div style="font-size: 1.35rem; font-weight:800; color:{t['accent_blue']};">13</div>
                    <div style="font-size: 0.72rem; color:{t['text_secondary']}; text-transform:uppercase; font-weight:600; margin-top:2px;">Bands Restored</div>
                </div>
                <div style="text-align:center; border-left: 1px solid {t['border']};">
                    <div style="font-size: 1.35rem; font-weight:800; color:{t['accent_blue']};">10 m</div>
                    <div style="font-size: 0.72rem; color:{t['text_secondary']}; text-transform:uppercase; font-weight:600; margin-top:2px;">Native Grid</div>
                </div>
                <div style="text-align:center; border-left: 1px solid {t['border']};">
                    <div style="font-size: 1.35rem; font-weight:800; color:{t['accent_teal']};">4× SR</div>
                    <div style="font-size: 0.72rem; color:{t['text_secondary']}; text-transform:uppercase; font-weight:600; margin-top:2px;">OpenSR Diffusion</div>
                </div>
                <div style="text-align:center; border-left: 1px solid {t['border']};">
                    <div style="font-size: 1.35rem; font-weight:800; color:{t['accent_green']};">S1 + S2</div>
                    <div style="font-size: 0.72rem; color:{t['text_secondary']}; text-transform:uppercase; font-weight:600; margin-top:2px;">SAR-Optical Fusion</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with hero_right:
        if HERO_IMAGE_PATH.exists():
            st.image(
                str(HERO_IMAGE_PATH),
                use_container_width=True,
                caption="Illustrative demonstration pair (Sentinel-2 L1C Cloudy vs. ECRformer Reconstructed)",
            )
            st.markdown(
                """
                <div class="cvai-img-note">
                    * Illustrative demonstration asset. Live and verified inference runs directly inside the workspace.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Demonstration satellite preview asset ready.")

    st.markdown("<hr style='margin: 40px 0 32px 0;'/>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Workflow Overview Section
    # -----------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="text-align: center; max-width: 720px; margin: 0 auto 32px auto;">
            <div class="cvai-badge">SCIENTIFIC PIPELINE</div>
            <h2 style="font-size: 2rem; margin-bottom: 8px;">End-to-End Image Restoration Workflow</h2>
            <p style="color: {t['text_secondary']}; font-size: 0.95rem; line-height: 1.6;">
                From Copernicus Data Space acquisition to cloud reconstruction and optional diffusion super-resolution.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    wf_c1, wf_c2, wf_c3 = st.columns(3, gap="medium")

    with wf_c1:
        st.markdown(
            f"""
            <div class="cvai-feature-card">
                <div style="display:inline-block; padding: 4px 10px; background:{t['accent_blue']}22; color:{t['accent_blue']}; border-radius:6px; font-weight:700; font-size:0.75rem; margin-bottom:12px;">STAGE 01</div>
                <div class="cvai-feat-title">Cloud Reconstruction</div>
                <div class="cvai-feat-desc">
                    <strong>ECRformer Cross-Attention:</strong> Fuses cloudy Sentinel-2 optical bands with cloud-penetrating Sentinel-1 SAR (VV/VH). Reconstructs all 13 spectral bands at 10 m resolution without hallucinating geometric features.
                </div>
                <div style="margin-top:16px; font-size:0.78rem; color:{t['text_muted']}; border-top:1px solid {t['border']}; padding-top:10px;">
                    ✓ 13 Bands (B01–B12)<br>
                    ✓ 10 m Native UTM Grid<br>
                    ✓ Tiled GeoTIFF Generation
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with wf_c2:
        st.markdown(
            f"""
            <div class="cvai-feature-card">
                <div style="display:inline-block; padding: 4px 10px; background:{t['accent_teal']}22; color:{t['accent_teal']}; border-radius:6px; font-weight:700; font-size:0.75rem; margin-bottom:12px;">STAGE 02</div>
                <div class="cvai-feat-title">Quantitative Evaluation</div>
                <div class="cvai-feat-desc">
                    <strong>Ground-Truth & Temporal Search:</strong> Upload a cloud-free reference or search CDSE for clear temporal scenes. Calculates MAE, RMSE, PSNR, SSIM, and Spectral Angle Mapper (SAM) across all 13 bands.
                </div>
                <div style="margin-top:16px; font-size:0.78rem; color:{t['text_muted']}; border-top:1px solid {t['border']}; padding-top:10px;">
                    ✓ Manual 13-Band Upload<br>
                    ✓ Live CDSE Reference Search<br>
                    ✓ Per-Band & Aggregate KPIs
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with wf_c3:
        st.markdown(
            f"""
            <div class="cvai-feature-card">
                <div style="display:inline-block; padding: 4px 10px; background:{t['accent_green']}22; color:{t['accent_green']}; border-radius:6px; font-weight:700; font-size:0.75rem; margin-bottom:12px;">STAGE 03</div>
                <div class="cvai-feat-title">Diffusion Super-Resolution</div>
                <div class="cvai-feat-desc">
                    <strong>OpenSR 4× Upsampling:</strong> Enhances key optical bands (Red, Green, Blue, NIR) from 10 m to 2.5 m resolution using generative latent diffusion, validated with 10 m baseline consistency checks.
                </div>
                <div style="margin-top:16px; font-size:0.78rem; color:{t['text_muted']}; border-top:1px solid {t['border']}; padding-top:10px;">
                    ✓ 4 Bands (B04, B03, B02, B08)<br>
                    ✓ 2.5 m Super-Resolution<br>
                    ✓ 10 m Consistency Diagnostics
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin: 40px 0 32px 0;'/>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Core Capabilities Grid (6 Cards)
    # -----------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="text-align: center; max-width: 720px; margin: 0 auto 32px auto;">
            <div class="cvai-badge">CORE CAPABILITIES</div>
            <h2 style="font-size: 2rem; margin-bottom: 8px;">Engineered for Remote Sensing Science</h2>
            <p style="color: {t['text_secondary']}; font-size: 0.95rem; line-height: 1.6;">
                Built on verified deep learning models and standards-compliant geospatial data formats.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cap_row1_c1, cap_row1_c2, cap_row1_c3 = st.columns(3, gap="medium")
    with cap_row1_c1:
        st.markdown(
            f"""
            <div class="cvai-feature-card">
                <span class="cvai-feat-icon">📡</span>
                <div class="cvai-feat-title">SAR + Optical Fusion</div>
                <div class="cvai-feat-desc">
                    Combines microwave radar signals that penetrate cloud layers with optical multispectral data for realistic surface reflectance reconstruction.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cap_row1_c2:
        st.markdown(
            f"""
            <div class="cvai-feature-card">
                <span class="cvai-feat-icon">🌈</span>
                <div class="cvai-feat-title">Full 13-Band Reconstruction</div>
                <div class="cvai-feat-desc">
                    Reconstructs the full Sentinel-2 spectral range: Coastal Aerosol (B01), RGB, Red Edge (B05-B07), NIR (B08, B8A), Water Vapour (B09), and SWIR (B11, B12).
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cap_row1_c3:
        st.markdown(
            f"""
            <div class="cvai-feature-card">
                <span class="cvai-feat-icon">🌐</span>
                <div class="cvai-feat-title">CDSE Catalogue Integration</div>
                <div class="cvai-feat-desc">
                    Direct connection to Copernicus Data Space Ecosystem via OAuth2 for searching, previewing, and acquiring synchronized Sentinel-1 & 2 scenes.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    cap_row2_c1, cap_row2_c2, cap_row2_c3 = st.columns(3, gap="medium")
    with cap_row2_c1:
        st.markdown(
            f"""
            <div class="cvai-feature-card">
                <span class="cvai-feat-icon">📊</span>
                <div class="cvai-feat-title">Scientific Performance Metrics</div>
                <div class="cvai-feat-desc">
                    Quantitative evaluation with PSNR, SSIM, Spectral Angle Mapper (SAM), MAE, and RMSE against cloud-free references with clear temporal warnings.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cap_row2_c2:
        st.markdown(
            f"""
            <div class="cvai-feature-card">
                <span class="cvai-feat-icon">⚡</span>
                <div class="cvai-feat-title">4× Diffusion Super-Resolution</div>
                <div class="cvai-feat-desc">
                    Optional OpenSR diffusion stage enhances RGB + NIR bands from 10 m to 2.5 m ground resolution while reporting strict consistency diagnostics.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cap_row2_c3:
        st.markdown(
            f"""
            <div class="cvai-feature-card">
                <span class="cvai-feat-icon">🗺️</span>
                <div class="cvai-feat-title">Geospatial GeoTIFF Exports</div>
                <div class="cvai-feat-desc">
                    Every output retains native CRS, affine geotransform, nodata masks, and radiometric scaling conventions ready for QGIS, ArcGIS, or GDAL.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='margin: 40px 0 32px 0;'/>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Bottom CTA Banner
    # -----------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {t['accent_blue']}15, {t['accent_teal']}15); border: 1px solid {t['border_accent']}; border-radius: 16px; padding: 36px 24px; text-align: center; margin: 20px 0;">
            <h2 style="font-size: 1.8rem; margin-bottom: 8px;">Ready to Restore Satellite Imagery?</h2>
            <p style="color: {t['text_secondary']}; max-width: 600px; margin: 0 auto 24px auto; font-size: 0.95rem;">
                Open the interactive workspace to run live CDSE acquisitions, process pre-validated baselines, or evaluate ground-truth metrics.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cta_btn_col1, cta_btn_col2, cta_btn_col3 = st.columns([1.5, 2, 1.5])
    with cta_btn_col2:
        if st.button("Launch Interactive Workspace →", type="primary", key="landing_bottom_cta", use_container_width=True):
            st.session_state["active_view"] = "workspace"
            st.rerun()

    # -----------------------------------------------------------------------
    # Footer
    # -----------------------------------------------------------------------
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 48px; padding-top: 24px; border-top: 1px solid {t['border']}; color: {t['text_muted']}; font-size: 0.78rem;">
            CloudVisionAI · Intelligent Satellite Image Restoration · Copernicus Data Space & ECRformer & OpenSR
        </div>
        """,
        unsafe_allow_html=True,
    )
