"""
CloudVisionAI — Theme System

Injects custom CSS for dark and light themes into the Streamlit app.
Theme choice is persisted via st.session_state["cvai_theme"] = "dark" | "light".
"""
from __future__ import annotations
import streamlit as st


# ---------------------------------------------------------------------------
# Color tokens
# ---------------------------------------------------------------------------
DARK = {
    "bg_page":        "#0D1117",
    "bg_surface":     "#161B22",
    "bg_card":        "#1C2333",
    "bg_card_hover":  "#232D42",
    "border":         "#30363D",
    "border_accent":  "#1F6FEB",
    "text_primary":   "#E6EDF3",
    "text_secondary": "#8B949E",
    "text_muted":     "#6E7681",
    "accent_blue":    "#1F6FEB",
    "accent_teal":    "#0EA5E9",
    "accent_green":   "#10B981",
    "accent_amber":   "#F59E0B",
    "accent_red":     "#EF4444",
    "btn_sec_bg":     "#21262D",
    "btn_sec_fg":     "#C9D1D9",
}

LIGHT = {
    "bg_page":        "#F0F4F8",
    "bg_surface":     "#FFFFFF",
    "bg_card":        "#F8FAFB",
    "bg_card_hover":  "#EFF3F7",
    "border":         "#D0D7DE",
    "border_accent":  "#0969DA",
    "text_primary":   "#1F2328",
    "text_secondary": "#57606A",
    "text_muted":     "#8C959F",
    "accent_blue":    "#0969DA",
    "accent_teal":    "#0284C7",
    "accent_green":   "#059669",
    "accent_amber":   "#D97706",
    "accent_red":     "#DC2626",
    "btn_sec_bg":     "#F3F4F6",
    "btn_sec_fg":     "#1F2328",
}


def get_theme() -> str:
    """Return current theme: 'dark' or 'light'."""
    return st.session_state.get("cvai_theme", "dark")


def toggle_theme() -> None:
    """Toggle between dark and light themes."""
    current = get_theme()
    st.session_state["cvai_theme"] = "light" if current == "dark" else "dark"


def get_tokens() -> dict:
    """Return color token dict for the active theme."""
    return DARK if get_theme() == "dark" else LIGHT


def inject_css() -> None:
    """Inject full CloudVisionAI CSS into the Streamlit app."""
    t = get_tokens()

    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Page background & Header ── */
.stApp, .block-container {{
    background-color: {t['bg_page']} !important;
    color: {t['text_primary']} !important;
}}
header[data-testid="stHeader"] {{
    background-color: {t['bg_page']} !important;
    color: {t['text_primary']} !important;
}}

/* ── Typography ── */
html, body, .stApp, .stMarkdown, p, span, label, div {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}}
.stMarkdown p, .stMarkdown span {{
    color: {t['text_primary']};
}}
h1, h2, h3, h4 {{
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-weight: 700;
    color: {t['text_primary']};
    letter-spacing: -0.02em;
}}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
    color: {t['text_primary']};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background-color: {t['bg_surface']} !important;
    border-right: 1px solid {t['border']} !important;
}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] .stMarkdown {{
    color: {t['text_primary']} !important;
}}
[data-testid="collapsedControl"],
button[data-testid="stSidebarCollapseButton"] {{
    color: {t['text_primary']} !important;
    background-color: {t['bg_card']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 8px !important;
}}
[data-testid="collapsedControl"]:hover,
button[data-testid="stSidebarCollapseButton"]:hover {{
    border-color: {t['border_accent']} !important;
    color: {t['accent_blue']} !important;
}}

/* ── Expander ── */
[data-testid="stExpander"],
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary,
[data-testid="stExpanderDetails"],
div[data-testid="stExpanderDetails"] {{
    background-color: {t['bg_card']} !important;
    color: {t['text_primary']} !important;
    border: 1px solid {t['border']} !important;
    border-radius: 10px !important;
    margin-bottom: 12px;
}}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] summary:focus {{
    background-color: {t['bg_card']} !important;
    color: {t['text_primary']} !important;
}}
[data-testid="stExpander"] summary *,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary div,
[data-testid="stExpander"] summary svg {{
    color: {t['text_primary']} !important;
    fill: {t['text_primary']} !important;
    font-weight: 600 !important;
}}

/* ── Metric cards ── */
[data-testid="stMetric"] {{
    background: {t['bg_card']};
    border: 1px solid {t['border']};
    border-radius: 10px;
    padding: 14px 16px;
    transition: border-color 0.2s;
}}
[data-testid="stMetric"]:hover {{
    border-color: {t['border_accent']};
}}
[data-testid="stMetricLabel"] {{ color: {t['text_secondary']} !important; font-size: 0.78rem; }}
[data-testid="stMetricValue"] {{ color: {t['text_primary']} !important; font-weight: 700; font-size: 1.25rem; }}

/* ── Primary buttons ── */
.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, {t['accent_blue']} 0%, {t['accent_teal']} 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-family: 'Inter', sans-serif;
    transition: opacity 0.2s, transform 0.15s;
    box-shadow: 0 2px 8px {t['accent_blue']}44;
}}
.stButton > button[kind="primary"]:hover {{
    opacity: 0.9;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px {t['accent_blue']}55;
}}

/* ── Secondary buttons ── */
.stButton > button[kind="secondary"],
.stButton > button {{
    background: {t['btn_sec_bg']};
    color: {t['btn_sec_fg']};
    border: 1px solid {t['border']};
    border-radius: 8px;
    font-weight: 500;
    font-family: 'Inter', sans-serif;
    transition: border-color 0.2s;
}}
.stButton > button:hover {{
    border-color: {t['border_accent']};
}}

/* ── Download buttons ── */
.stDownloadButton > button {{
    background: {t['bg_card']} !important;
    color: {t['accent_blue']} !important;
    border: 1px solid {t['border_accent']} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: background 0.2s;
}}
.stDownloadButton > button:hover {{
    background: {t['border_accent']}18 !important;
}}

/* ── Alert boxes ── */
[data-testid="stAlert"] {{
    border-radius: 10px !important;
    border-width: 1px !important;
    font-size: 0.88rem;
}}

/* ── DataFrame ── */
[data-testid="stDataFrame"] iframe {{
    border-radius: 10px;
}}

/* ── Inputs ── */
[data-testid="stTextInput"] > div > div > input,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea,
.stSelectbox select {{
    background: {t['bg_surface']} !important;
    color: {t['text_primary']} !important;
    border-color: {t['border']} !important;
    border-radius: 8px !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: {t['bg_card']} !important;
    border: 2px dashed {t['border']} !important;
    border-radius: 10px !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: {t['bg_surface']};
    border-bottom: 2px solid {t['border']};
    gap: 4px;
    padding: 0 4px;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent;
    color: {t['text_secondary']};
    font-weight: 500;
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
    transition: color 0.2s, background 0.2s;
    font-family: 'Inter', sans-serif;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background: {t['bg_card']};
    color: {t['text_primary']};
}}
.stTabs [aria-selected="true"] {{
    background: {t['bg_card']} !important;
    color: {t['accent_blue']} !important;
    border-bottom: 2px solid {t['accent_blue']} !important;
    font-weight: 600 !important;
}}

/* ── Dividers ── */
hr {{
    border-color: {t['border']} !important;
    margin: 20px 0;
}}

/* ── Progress ── */
[data-testid="stProgress"] > div > div {{
    background: linear-gradient(90deg, {t['accent_blue']}, {t['accent_teal']}) !important;
    border-radius: 4px;
}}

/* ── Custom CloudVisionAI components ── */
.cvai-brand-row {{
    display: flex;
    align-items: center;
    gap: 10px;
}}

.cvai-brand {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, {t['accent_blue']} 0%, {t['accent_teal']} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}}

.cvai-brand-sub {{
    font-size: 0.68rem;
    color: {t['text_muted']};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: -4px;
}}

.cvai-badge {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    background: {t['accent_blue']}18;
    color: {t['accent_blue']};
    border: 1px solid {t['accent_blue']}44;
    margin-bottom: 14px;
}}

.cvai-hero-hl {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(1.8rem, 3.5vw, 3rem);
    font-weight: 800;
    line-height: 1.15;
    letter-spacing: -0.03em;
    color: {t['text_primary']};
    margin-bottom: 14px;
}}

.cvai-hero-hl .grad {{
    background: linear-gradient(135deg, {t['accent_blue']} 0%, {t['accent_teal']} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.cvai-hero-sub {{
    font-size: 1rem;
    line-height: 1.7;
    color: {t['text_secondary']};
    margin-bottom: 28px;
}}

.cvai-feature-card {{
    background: {t['bg_card']};
    border: 1px solid {t['border']};
    border-radius: 14px;
    padding: 22px;
    height: 100%;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.2s, box-shadow 0.25s;
}}
.cvai-feature-card:hover {{
    border-color: {t['border_accent']};
    transform: translateY(-2px);
    box-shadow: 0 8px 24px {t['accent_blue']}20;
}}
.cvai-feat-icon {{
    font-size: 1.9rem;
    display: block;
    margin-bottom: 10px;
}}
.cvai-feat-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.98rem;
    font-weight: 600;
    color: {t['text_primary']};
    margin-bottom: 5px;
}}
.cvai-feat-desc {{
    font-size: 0.84rem;
    color: {t['text_secondary']};
    line-height: 1.6;
}}

.cvai-img-caption {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: {t['text_muted']};
    text-align: center;
    margin-top: 5px;
}}
.cvai-img-note {{
    font-size: 0.66rem;
    color: {t['text_muted']};
    text-align: center;
    font-style: italic;
}}

.cvai-step {{
    display: flex;
    gap: 14px;
    align-items: flex-start;
    padding: 13px 0;
    border-bottom: 1px solid {t['border']};
}}
.cvai-step:last-child {{ border-bottom: none; }}
.cvai-step-num {{
    min-width: 28px; height: 28px;
    border-radius: 50%;
    background: linear-gradient(135deg, {t['accent_blue']}, {t['accent_teal']});
    color: #fff;
    font-weight: 700; font-size: 0.78rem;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0; margin-top: 1px;
}}
.cvai-step-title {{
    font-weight: 600; font-size: 0.92rem;
    color: {t['text_primary']}; margin-bottom: 2px;
}}
.cvai-step-desc {{
    font-size: 0.81rem; color: {t['text_secondary']}; line-height: 1.5;
}}

.cvai-ws-header {{
    padding: 10px 0 14px 0;
    border-bottom: 1px solid {t['border']};
    margin-bottom: 4px;
}}
.cvai-ws-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: {t['text_primary']};
}}
.cvai-ws-sub {{
    font-size: 0.82rem;
    color: {t['text_muted']};
    margin-top: 2px;
}}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)
