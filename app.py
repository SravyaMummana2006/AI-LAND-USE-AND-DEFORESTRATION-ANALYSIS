"""
app.py
------
Streamlit Dashboard — AI-Powered Land Use & Deforestation Analysis System.

Dashboard Sections:
    1. Sidebar         — Upload controls, model settings, run trigger
    2. Image Viewer    — Side-by-side original satellite images
    3. Classification  — Colour-coded land-use overlays for both images
    4. Change Detection— Highlighted deforestation map + heatmap
    5. Charts          — Plotly comparison, gauge, pie, trend charts
    6. Report Panel    — Natural-language findings + risk badge + EIS score
    7. Data Tables     — Area statistics and contour region breakdown

Run:
    streamlit run app.py

Author: AI Land Use Analysis System
"""

import io
import logging
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# ── Project modules ──────────────────────────────────────────────────────────
from preprocess  import preprocess_for_display
from predict     import (
    run_full_prediction,
    compare_predictions,
    get_class_color_legend,
    load_model,
    load_class_labels,
    DEFAULT_CLASS_LABELS
)
from detect_change    import run_change_detection
from report_generator import generate_full_report
from gemini_analyst   import (
    initialise_gemini,
    build_analysis_context,
    DeforestationChatbot,
    render_gemini_report_section,
    render_chatbot_section,
    GEMINI_API_KEY
)

from ai_extensions import (
    render_carbon_section,
    render_trend_section,
    render_vision_section,
    render_location_section,
    render_species_section,
    render_alerts_section,
    AnomalyDetector
)

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Page Config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title  = "Land Use & Deforestation Analyzer",
    page_icon   = "🌿",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────
# Custom CSS — Production-grade styling
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* ── Root Variables ── */
    :root {
        --green-dark   : #1B4332;
        --green-mid    : #2D6A4F;
        --green-light  : #52B788;
        --green-pale   : #D8F3DC;
        --earth        : #8B5E3C;
        --sky          : #1D3557;
        --accent       : #F4A261;
        --danger       : #E63946;
        --warning      : #F4D35E;
        --bg           : #F0F4EF;
        --card-bg      : #FFFFFF;
        --text         : #1A1A2E;
        --muted        : #6C757D;
    }

    /* ── Global ── */
    html, body {
        font-family: 'DM Sans', sans-serif;
        background-color: var(--bg);
        color: var(--text);
    }

    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── App header banner ── */
    .app-banner {
        background: linear-gradient(135deg, #1B4332 0%, #2D6A4F 50%, #40916C 100%);
        border-radius: 14px;
        padding: 32px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(27,67,50,0.28);
    }
    .app-banner::before {
        content: '';
        position: absolute;
        top: -40px; right: -40px;
        width: 220px; height: 220px;
        background: rgba(255,255,255,0.06);
        border-radius: 50%;
    }
    .app-banner::after {
        content: '';
        position: absolute;
        bottom: -60px; left: 30%;
        width: 300px; height: 300px;
        background: rgba(255,255,255,0.04);
        border-radius: 50%;
    }
    .app-banner h1 {
        font-family: 'DM Serif Display', serif;
        color: #FFFFFF;
        font-size: 2.1em;
        margin: 0 0 6px;
        letter-spacing: -0.5px;
        position: relative; z-index: 1;
    }
    .app-banner p {
        color: rgba(255,255,255,0.78);
        font-size: 0.95em;
        margin: 0;
        position: relative; z-index: 1;
    }

    /* ── Section headers ── */
    .section-header {
        font-family: 'DM Serif Display', serif;
        font-size: 1.35em;
        color: var(--green-dark);
        border-left: 4px solid var(--green-light);
        padding-left: 14px;
        margin: 28px 0 16px;
        letter-spacing: -0.3px;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: var(--card-bg);
        border: 1px solid #E2EAE0;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 18px rgba(0,0,0,0.09);
    }
    .metric-card .val {
        font-size: 2.1em;
        font-weight: 600;
        color: var(--green-dark);
        line-height: 1;
    }
    .metric-card .lbl {
        font-size: 0.78em;
        color: var(--muted);
        margin-top: 6px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }

    /* ── Risk badge ── */
    .risk-badge {
        display: inline-block;
        padding: 8px 22px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.05em;
        letter-spacing: 1.2px;
        text-transform: uppercase;
    }

    /* ── Finding blocks ── */
    .finding-block {
        background: #F8FAF8;
        border: 1px solid #DCE8DC;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
        font-size: 0.96em;
        line-height: 1.7;
    }
    .finding-block strong {
        color: var(--green-dark);
        font-weight: 600;
    }

    /* ── Summary box ── */
    .summary-box {
        background: linear-gradient(135deg, #EBF5EE, #F0FAF2);
        border-left: 5px solid var(--green-light);
        border-radius: 0 10px 10px 0;
        padding: 18px 22px;
        font-size: 0.97em;
        line-height: 1.75;
        margin-bottom: 16px;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.04);
    }

    /* ── Image containers ── */
    .img-container {
        border-radius: 10px;
        overflow: hidden;
        border: 2px solid #D8E8D8;
        box-shadow: 0 3px 12px rgba(0,0,0,0.08);
    }

    /* ── EIS score bar ── */
    .eis-bar-wrap {
        background: #E9ECEF;
        border-radius: 50px;
        height: 14px;
        width: 100%;
        margin-top: 8px;
        overflow: hidden;
    }
    .eis-bar-fill {
        height: 100%;
        border-radius: 50px;
        transition: width 0.8s ease;
    }

    /* ── Legend pill ── */
    .legend-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #F4F7F4;
        border: 1px solid #DDE8DD;
        border-radius: 50px;
        padding: 5px 14px 5px 8px;
        font-size: 0.82em;
        font-weight: 500;
        margin: 4px 3px;
    }
    .legend-dot {
        width: 12px; height: 12px;
        border-radius: 50%;
        display: inline-block;
        flex-shrink: 0;
    }

    /* ── Upload zone styling ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed #A8D5B5 !important;
        border-radius: 10px !important;
        background: #F2FAF4 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1B4332 0%, #2D6A4F 100%);
    }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    [data-testid="stSidebar"] .stButton > button {
        background: #52B788 !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        width: 100% !important;
        padding: 12px !important;
        font-size: 1em !important;
        letter-spacing: 0.5px;
        transition: background 0.2s;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: #74C69D !important;
    }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.15) !important; }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: #EDF2ED;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        font-weight: 500;
        color: #2D6A4F !important;
    }
    .stTabs [aria-selected="true"] {
        background: #2D6A4F !important;
        color: white !important;
    }

    /* ── Divider ── */
    hr { border-color: #DCE8DC; margin: 24px 0; }

    /* ── Bullet list ── */
    ul.findings-list {
        padding-left: 20px;
        margin-top: 8px;
    }
    ul.findings-list li {
        margin-bottom: 9px;
        font-size: 0.95em;
        line-height: 1.65;
    }

    /* ── Scrollable table ── */
    .scroll-table { overflow-x: auto; }

    /* ── Status chip ── */
    .status-chip {
        display: inline-block;
        border-radius: 50px;
        padding: 3px 14px;
        font-size: 0.78em;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session State Initialisation
# ─────────────────────────────────────────────
def init_session_state():
    """Initialise all session state keys to prevent KeyError on first load."""
    defaults = {
        "analysis_done"      : False,
        "old_bytes"          : None,
        "new_bytes"          : None,
        "result_old"         : None,
        "result_new"         : None,
        "change_results"     : None,
        "report"             : None,
        "pred_comparison"    : None,
        "error_message"      : None,
        "gemini_report"      : None,
        "chat_messages"      : [],
        "analysis_context"   : None,
        "chatbot"            : None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ─────────────────────────────────────────────
# Helper Utilities
# ─────────────────────────────────────────────

def np_to_pil(arr: np.ndarray) -> Image.Image:
    """Convert a uint8 NumPy RGB array to a PIL Image for Streamlit display."""
    return Image.fromarray(arr.astype(np.uint8))


def bytes_to_display(raw_bytes: bytes) -> np.ndarray:
    """Convert raw uploaded bytes to a display-ready RGB NumPy array."""
    return preprocess_for_display(raw_bytes)


def risk_badge_html(level: str, color: str) -> str:
    """Return an HTML risk badge span for the given risk level and hex color."""
    return (
        f'<span class="risk-badge" '
        f'style="background:{color};color:white;">'
        f'{level.upper()}</span>'
    )


def eis_bar_html(score: float) -> str:
    """Return an HTML EIS progress bar for the given score (0-100)."""
    if score < 25:
        bar_color = "#27AE60"
    elif score < 50:
        bar_color = "#F39C12"
    elif score < 75:
        bar_color = "#E74C3C"
    else:
        bar_color = "#7B241C"

    return f"""
    <div class="eis-bar-wrap">
        <div class="eis-bar-fill"
             style="width:{score}%;background:{bar_color};"></div>
    </div>
    <p style="text-align:right;font-size:0.82em;
              color:#6C757D;margin-top:4px">{score:.1f} / 100</p>
    """


def color_legend_html(class_colors: dict) -> str:
    """Build HTML legend pills for all land-use classes."""
    pills = ""
    for cls, rgb in class_colors.items():
        hex_color = "#{:02X}{:02X}{:02X}".format(*rgb)
        pills += (
            f'<span class="legend-pill">'
            f'<span class="legend-dot" style="background:{hex_color}"></span>'
            f'{cls}</span>'
        )
    return pills


def render_metric_cards(metrics: list):
    """
    Render a row of metric cards.

    Parameters:
        metrics (list): List of (label, value, suffix) tuples.
    """
    cols = st.columns(len(metrics))
    for col, (label, value, suffix) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="val">{value}{suffix}</div>'
                f'<div class="lbl">{label}</div>'
                f'</div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────
# Core Analysis Runner
# ─────────────────────────────────────────────

def run_analysis(old_bytes: bytes, new_bytes: bytes):
    """
    Execute the full analysis pipeline and store results in session state.

    Pipeline:
        1. Land-use prediction on old image
        2. Land-use prediction on new image
        3. Compare predictions
        4. Change detection (deforestation, urban expansion, water change)
        5. Report generation

    Parameters:
        old_bytes (bytes): Raw bytes of the old satellite image.
        new_bytes (bytes): Raw bytes of the new satellite image.
    """
    progress = st.progress(0, text="Initialising pipeline...")

    try:
        # ── Step 1: Predict old image ──
        progress.progress(10, text="Classifying old satellite image...")
        result_old = run_full_prediction(old_bytes)
        logger.info("Old image prediction complete.")

        # ── Step 2: Predict new image ──
        progress.progress(30, text="Classifying new satellite image...")
        result_new = run_full_prediction(new_bytes)
        logger.info("New image prediction complete.")

        # ── Step 3: Compare predictions ──
        progress.progress(50, text="Comparing land use classifications...")
        pred_comparison = compare_predictions(result_old, result_new)

        # ── Step 4: Change detection ──
        progress.progress(65, text="Detecting deforestation and land changes...")
        change_results = run_change_detection(old_bytes, new_bytes)
        logger.info("Change detection complete.")

        # ── Step 5: Generate report ──
        progress.progress(85, text="Generating intelligent report...")
        report = generate_full_report(
            change_results        = change_results,
            old_area_stats        = result_old["area_stats"],
            new_area_stats        = result_new["area_stats"],
            prediction_comparison = pred_comparison,
            save_html             = True,
            save_json             = True
        )

        # ── Store in session state ──
        st.session_state.result_old      = result_old
        st.session_state.result_new      = result_new
        st.session_state.pred_comparison = pred_comparison
        st.session_state.change_results  = change_results
        st.session_state.report          = report
        st.session_state.analysis_done   = True
        st.session_state.error_message   = None

        # ── Build Gemini context & chatbot ──
        if GEMINI_API_KEY:
            initialise_gemini()
            context = build_analysis_context(
                change_results        = change_results,
                old_area_stats        = result_old["area_stats"],
                new_area_stats        = result_new["area_stats"],
                prediction_comparison = pred_comparison
            )
            st.session_state.analysis_context = context
            st.session_state.chatbot = DeforestationChatbot(context)
            st.session_state.gemini_report = None
            st.session_state.chat_messages = []

        progress.progress(100, text="Analysis complete!")
        logger.info("Full pipeline complete.")

    except FileNotFoundError as e:
        st.session_state.error_message = (
            f"Model not found. Please train the model first.\n\n`{e}`"
        )
        st.session_state.analysis_done = False
        progress.empty()

    except Exception as e:
        st.session_state.error_message = f"Analysis failed: {str(e)}"
        st.session_state.analysis_done = False
        progress.empty()
        logger.exception("Pipeline error")


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def render_sidebar() -> tuple:
    """
    Render the sidebar with upload widgets and analysis controls.

    Returns:
        tuple: (old_file, new_file) Streamlit UploadedFile objects or None.
    """
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:16px 0 8px">
            <span style="font-size:2.8em">🌿</span>
            <h2 style="font-family:'DM Serif Display',serif;
                       font-size:1.3em;margin:6px 0 2px">
                Deforestation Analyzer
            </h2>
            <p style="font-size:0.8em;opacity:0.7">
                AI-Powered Satellite Analysis
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### 📡 Upload Satellite Images")

        old_file = st.file_uploader(
            "🕐 Old Image (Earlier Date)",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help="Upload the older satellite image for baseline comparison.",
            key="old_uploader"
        )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        new_file = st.file_uploader(
            "🕒 New Image (Recent Date)",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            help="Upload the recent satellite image to detect changes.",
            key="new_uploader"
        )

        st.markdown("---")
        st.markdown("### ⚙️ Settings")

        st.selectbox(
            "Model Type",
            ["Custom CNN", "MobileNetV2 Transfer"],
            help="Choose the model architecture used for classification.",
            key="model_type"
        )

        st.slider(
            "Change Sensitivity",
            min_value=10, max_value=60, value=30, step=5,
            help="Pixel difference threshold for change detection. "
                 "Lower = more sensitive.",
            key="diff_threshold"
        )

        st.checkbox(
            "Save HTML Report",
            value=True,
            key="save_html",
            help="Export a downloadable HTML report after analysis."
        )

        st.markdown("---")

        # ── Run button ──
        run_ready = old_file is not None and new_file is not None
        if st.button(
            "🔍 Run Analysis" if run_ready else "⬆️ Upload Both Images",
            disabled=not run_ready,
            use_container_width=True
        ):
            st.session_state.old_bytes = old_file.read()
            st.session_state.new_bytes = new_file.read()
            run_analysis(
                st.session_state.old_bytes,
                st.session_state.new_bytes
            )

        if st.button("🗑️ Clear Results",
                     use_container_width=True,
                     type="secondary"):
            for key in ["analysis_done", "result_old", "result_new",
                        "change_results", "report", "pred_comparison",
                        "old_bytes", "new_bytes", "error_message"]:
                st.session_state[key] = None
            st.session_state.analysis_done = False
            st.rerun()

        st.markdown("---")
        st.markdown(
            "<p style='font-size:0.72em;opacity:0.55;text-align:center'>"
            "AI Land Use Analysis v1.0<br>"
            "TensorFlow · OpenCV · Streamlit</p>",
            unsafe_allow_html=True
        )

    return old_file, new_file


# ─────────────────────────────────────────────
# Dashboard Sections
# ─────────────────────────────────────────────

def render_header():
    """Render the top application banner."""
    st.markdown("""
    <div class="app-banner">
        <h1>🌿 AI Land Use & Deforestation Analyzer</h1>
        <p>
            Upload a pair of satellite images to classify land use, detect
            deforestation, quantify forest loss, and generate an intelligent
            environmental impact report — powered by TensorFlow & OpenCV.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_upload_preview(old_bytes: bytes, new_bytes: bytes):
    """
    Render side-by-side preview of the two uploaded satellite images.

    Parameters:
        old_bytes (bytes): Raw bytes of old image.
        new_bytes (bytes): Raw bytes of new image.
    """
    st.markdown('<div class="section-header">📸 Uploaded Satellite Images</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🕐 Old Image (Baseline)**")
        old_display = bytes_to_display(old_bytes)
        st.image(np_to_pil(old_display), use_column_width=True,
                 caption="Original — Before Period")

    with col2:
        st.markdown("**🕒 New Image (Recent)**")
        new_display = bytes_to_display(new_bytes)
        st.image(np_to_pil(new_display), use_column_width=True,
                 caption="Original — After Period")


def render_classification_section(result_old: dict, result_new: dict):
    """
    Render land-use classification overlays for both images
    alongside confidence scores and class legend.

    Parameters:
        result_old (dict): run_full_prediction() result for old image.
        result_new (dict): run_full_prediction() result for new image.
    """
    st.markdown('<div class="section-header">🗺️ Land Use Classification</div>',
                unsafe_allow_html=True)

    # Class color legend
    legend_html = color_legend_html(get_class_color_legend())
    st.markdown(
        f'<div style="margin-bottom:16px">'
        f'<span style="font-size:0.85em;color:#6C757D;'
        f'font-weight:500;margin-right:8px">LEGEND:</span>'
        f'{legend_html}</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🕐 Old Image — Classification Overlay**")
        st.image(np_to_pil(result_old["overlay"]),
                 use_column_width=True,
                 caption=(
                     f"Dominant: {result_old['global_prediction']['predicted_class']} "
                     f"({result_old['global_prediction']['confidence']:.1%})"
                 ))

        # Probability table
        with st.expander("📊 Class Probabilities (Old)", expanded=False):
            probs_old = result_old["global_prediction"]["probabilities"]
            df_probs = pd.DataFrame(
                list(probs_old.items()), columns=["Class", "Probability"]
            ).sort_values("Probability", ascending=False)
            df_probs["Probability"] = df_probs["Probability"].map("{:.2%}".format)
            st.dataframe(df_probs, hide_index=True, use_container_width=True)

    with col2:
        st.markdown("**🕒 New Image — Classification Overlay**")
        st.image(np_to_pil(result_new["overlay"]),
                 use_column_width=True,
                 caption=(
                     f"Dominant: {result_new['global_prediction']['predicted_class']} "
                     f"({result_new['global_prediction']['confidence']:.1%})"
                 ))

        with st.expander("📊 Class Probabilities (New)", expanded=False):
            probs_new = result_new["global_prediction"]["probabilities"]
            df_probs = pd.DataFrame(
                list(probs_new.items()), columns=["Class", "Probability"]
            ).sort_values("Probability", ascending=False)
            df_probs["Probability"] = df_probs["Probability"].map("{:.2%}".format)
            st.dataframe(df_probs, hide_index=True, use_container_width=True)

    # Area statistics side-by-side
    st.markdown("#### 📐 Area Coverage Statistics")
    col_a, col_b = st.columns(2)
    with col_a:
        st.caption("Old Image — Area Breakdown")
        st.dataframe(
            result_old["area_stats"][["Class", "Percentage"]].assign(
                Percentage=lambda df: df["Percentage"].map("{:.2f}%".format)
            ),
            hide_index=True, use_container_width=True
        )
    with col_b:
        st.caption("New Image — Area Breakdown")
        st.dataframe(
            result_new["area_stats"][["Class", "Percentage"]].assign(
                Percentage=lambda df: df["Percentage"].map("{:.2f}%".format)
            ),
            hide_index=True, use_container_width=True
        )


def render_change_detection_section(change_results: dict):
    """
    Render the deforestation detection section:
    - Before/After annotated highlight image
    - Change heatmap
    - Key change metrics
    - Contour region table

    Parameters:
        change_results (dict): run_change_detection() output.
    """
    st.markdown('<div class="section-header">🔥 Deforestation & Change Detection</div>',
                unsafe_allow_html=True)

    # Key change metric cards
    render_metric_cards([
        ("Forest Loss",      f"{change_results['forest_loss_pct']:.1f}",  "%"),
        ("Forest Gain",      f"{change_results['forest_gain_pct']:.1f}",  "%"),
        ("Urban Expansion",  f"{change_results['urban_expansion_pct']:.1f}", "%"),
        ("Water Change",     f"{change_results['water_change_pct']:+.1f}", "%"),
        ("Total Changed",    f"{change_results['total_changed_pct']:.1f}", "%"),
        ("Changed Regions",  str(change_results['num_changed_regions']),    ""),
    ])

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Highlight image + heatmap tabs
    tab1, tab2 = st.tabs([
        "📍 Annotated Before / After",
        "🌡️ Change Intensity Heatmap"
    ])

    with tab1:
        st.image(
            np_to_pil(change_results["highlight_image"]),
            use_column_width=True,
            caption=(
                "Red overlay = forest loss | "
                "Orange overlay = urban expansion | "
                "Red contours = detected change boundaries"
            )
        )

    with tab2:
        col_hm, col_legend = st.columns([3, 1])
        with col_hm:
            st.image(
                np_to_pil(change_results["heatmap"]),
                use_column_width=True,
                caption="Blue = minimal change → Green → Yellow → Red = maximum change"
            )
        with col_legend:
            st.markdown("""
            **Heatmap Scale**

            🔵 **Cool** — Unchanged / stable area

            🟢 **Green** — Low-level change

            🟡 **Yellow** — Moderate change

            🔴 **Red** — High intensity change
            """)

    # Contour region breakdown
    if not change_results["contour_stats"].empty:
        st.markdown("#### 📌 Detected Change Regions")
        display_df = change_results["contour_stats"].copy()
        display_df["Area_pct"] = display_df["Area_pct"].map("{:.3f}%".format)
        st.dataframe(
            display_df[["Region_ID", "Area_px", "Area_pct",
                         "Centroid_X", "Centroid_Y"]],
            hide_index=True,
            use_container_width=True
        )


def render_charts_section(report: dict):
    """
    Render the Plotly analytics charts across two tab rows:
    - Tab 1: Grouped comparison bar chart
    - Tab 2: Forest loss + EIS gauges
    - Tab 3: Change composition pie
    - Tab 4: Net class change horizontal bar

    Parameters:
        report (dict): Full report dict from generate_full_report().
    """
    st.markdown('<div class="section-header">📊 Analytics & Charts</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Coverage Comparison",
        "🎯 Environmental Indicators",
        "🥧 Change Breakdown",
        "📉 Net Class Change"
    ])

    with tab1:
        st.plotly_chart(
            report["charts"]["comparison_chart"],
            use_container_width=True
        )

    with tab2:
        st.plotly_chart(
            report["charts"]["gauge_chart"],
            use_container_width=True
        )

    with tab3:
        st.plotly_chart(
            report["charts"]["pie_chart"],
            use_container_width=True
        )

    with tab4:
        st.plotly_chart(
            report["charts"]["trend_chart"],
            use_container_width=True
        )


def render_report_section(report: dict, change_results: dict):
    """
    Render the intelligent natural-language report panel:
    - Risk badge + EIS score bar
    - Executive summary
    - Detailed findings (forest, urban, water)
    - Per-class change bullets
    - Recommended action
    - Download button for HTML report

    Parameters:
        report (dict): Full report dict from generate_full_report().
        change_results (dict): Change detection results dict.
    """
    st.markdown('<div class="section-header">📋 Intelligent Analysis Report</div>',
                unsafe_allow_html=True)

    findings  = report["findings"]
    risk      = report["risk"]
    eis_score = report["eis_score"]
    metrics   = report["metrics"]

    # ── Risk & EIS header row ──
    col_risk, col_eis = st.columns([1, 2])

    with col_risk:
        st.markdown("**Risk Classification**")
        st.markdown(
            risk_badge_html(risk["level"], risk["color"]),
            unsafe_allow_html=True
        )
        st.markdown(
            f"<p style='font-size:0.82em;color:#6C757D;margin-top:8px'>"
            f"{risk['description']}</p>",
            unsafe_allow_html=True
        )

    with col_eis:
        st.markdown("**Environmental Impact Score (EIS)**")
        st.markdown(eis_bar_html(eis_score), unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:0.82em;color:#6C757D'>"
            "0 = No impact &nbsp;|&nbsp; 100 = Maximum environmental damage</p>",
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Executive Summary ──
    st.markdown("**🌐 Executive Summary**")
    st.markdown(
        f'<div class="summary-box">{findings["summary"]}</div>',
        unsafe_allow_html=True
    )

    # ── Detailed Findings ──
    st.markdown("**🔍 Detailed Findings**")

    st.markdown(
        f'<div class="finding-block">'
        f'<strong>🌳 Forest:</strong> {findings["forest"]}'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="finding-block">'
        f'<strong>🏙️ Urban:</strong> {findings["urban"]}'
        f'</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div class="finding-block">'
        f'<strong>💧 Water:</strong> {findings["water"]}'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Per-Class Bullets ──
    st.markdown("**📌 Per-Class Land Use Changes**")
    bullets_html = "".join(
        f"<li style='margin-bottom:8px'>{b}</li>"
        for b in findings["class_bullets"]
    )
    st.markdown(
        f'<ul class="findings-list">{bullets_html}</ul>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ── Recommended Action ──
    action_color = risk["color"]
    st.markdown(
        f'<div style="background:{action_color}18;border:1.5px solid {action_color}55;'
        f'border-radius:10px;padding:16px 20px;">'
        f'<strong style="color:{action_color}">⚡ Recommended Action</strong><br>'
        f'<span style="font-size:0.96em">{risk["action"]}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # ── Quick Stats Summary Table ──
    with st.expander("📊 Full Metrics Summary", expanded=False):
        summary_df = pd.DataFrame([
            {"Metric": "Forest Cover (Old)",    "Value": f"{metrics['old_forest_pct']:.1f}%"},
            {"Metric": "Forest Cover (New)",    "Value": f"{metrics['new_forest_pct']:.1f}%"},
            {"Metric": "Forest Loss",           "Value": f"{metrics['forest_loss_pct']:.1f}%"},
            {"Metric": "Forest Gain",           "Value": f"{metrics['forest_gain_pct']:.1f}%"},
            {"Metric": "Net Forest Change",     "Value": f"{metrics['net_forest_change_pct']:+.1f}%"},
            {"Metric": "Urban Cover (Old)",     "Value": f"{metrics['old_urban_pct']:.1f}%"},
            {"Metric": "Urban Cover (New)",     "Value": f"{metrics['new_urban_pct']:.1f}%"},
            {"Metric": "Urban Expansion",       "Value": f"{metrics['urban_expansion_pct']:.1f}%"},
            {"Metric": "Water Cover (Old)",     "Value": f"{metrics['old_water_pct']:.1f}%"},
            {"Metric": "Water Cover (New)",     "Value": f"{metrics['new_water_pct']:.1f}%"},
            {"Metric": "Water Change",          "Value": f"{metrics['water_change_pct']:+.1f}%"},
            {"Metric": "Total Changed Area",    "Value": f"{metrics['total_changed_pct']:.1f}%"},
            {"Metric": "Changed Regions",       "Value": str(metrics['num_changed_regions'])},
            {"Metric": "EIS Score",             "Value": f"{eis_score:.1f} / 100"},
            {"Metric": "Risk Level",            "Value": risk["level"]},
        ])
        st.dataframe(summary_df, hide_index=True, use_container_width=True)

    # ── Download HTML Report ──
    html_path = "reports/land_use_report.html"
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        st.download_button(
            label     = "⬇️ Download Full HTML Report",
            data      = html_content,
            file_name = "land_use_deforestation_report.html",
            mime      = "text/html",
            use_container_width=True
        )
    except FileNotFoundError:
        st.info("HTML report file not found. Enable 'Save HTML Report' in settings.")


# ─────────────────────────────────────────────
# Welcome / Empty State
# ─────────────────────────────────────────────

def render_welcome_state():
    """Render the landing state shown before any images are uploaded."""
    st.markdown("""
    <div style="text-align:center;padding:60px 20px">
        <div style="font-size:5em;margin-bottom:16px">🛰️</div>
        <h2 style="font-family:'DM Serif Display',serif;
                   color:#1B4332;font-size:1.8em;margin-bottom:12px">
            Ready to Analyse Satellite Imagery
        </h2>
        <p style="color:#6C757D;max-width:520px;
                  margin:0 auto;font-size:1.02em;line-height:1.7">
            Upload an <strong>old</strong> and a <strong>new</strong> satellite
            image using the sidebar to begin AI-powered land use classification,
            deforestation detection, and environmental impact analysis.
        </p>
        <div style="margin-top:36px;display:flex;
                    justify-content:center;gap:24px;flex-wrap:wrap">
            <div style="background:white;border:1px solid #DCE8DC;
                        border-radius:12px;padding:20px 28px;min-width:150px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.06)">
                <div style="font-size:2em">🌳</div>
                <div style="font-size:0.85em;color:#2D6A4F;
                            font-weight:600;margin-top:6px">
                    Land Use<br>Classification
                </div>
            </div>
            <div style="background:white;border:1px solid #DCE8DC;
                        border-radius:12px;padding:20px 28px;min-width:150px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.06)">
                <div style="font-size:2em">🔍</div>
                <div style="font-size:0.85em;color:#2D6A4F;
                            font-weight:600;margin-top:6px">
                    Deforestation<br>Detection
                </div>
            </div>
            <div style="background:white;border:1px solid #DCE8DC;
                        border-radius:12px;padding:20px 28px;min-width:150px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.06)">
                <div style="font-size:2em">📊</div>
                <div style="font-size:0.85em;color:#2D6A4F;
                            font-weight:600;margin-top:6px">
                    Analytics<br>& Charts
                </div>
            </div>
            <div style="background:white;border:1px solid #DCE8DC;
                        border-radius:12px;padding:20px 28px;min-width:150px;
                        box-shadow:0 2px 8px rgba(0,0,0,0.06)">
                <div style="font-size:2em">📋</div>
                <div style="font-size:0.85em;color:#2D6A4F;
                            font-weight:600;margin-top:6px">
                    Intelligent<br>Report
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main App Entry Point
# ─────────────────────────────────────────────

def main():
    """
    Main Streamlit application entry point.
    Renders the full dashboard in sequential sections.
    """
    # ── Render banner ──
    render_header()

    # ── Sidebar upload controls ──
    old_file, new_file = render_sidebar()

    # ── Error state ──
    if st.session_state.error_message:
        st.error(f"⚠️ {st.session_state.error_message}")
        st.info(
            "If the model is not trained yet, run:\n\n"
            "```bash\npython train_model.py --data_dir ./data\n```"
        )
        return

    # ── Upload preview (show once files are uploaded) ──
    if st.session_state.old_bytes and st.session_state.new_bytes:
        render_upload_preview(
            st.session_state.old_bytes,
            st.session_state.new_bytes
        )

    # ── Analysis results ──
    if st.session_state.analysis_done:
        st.markdown("---")
        render_classification_section(
            st.session_state.result_old,
            st.session_state.result_new
        )

        st.markdown("---")
        render_change_detection_section(st.session_state.change_results)

        st.markdown("---")
        render_charts_section(st.session_state.report)

        st.markdown("---")
        render_report_section(
            st.session_state.report,
            st.session_state.change_results
        )

        # ── Gemini AI sections (only if API key present) ──
        if GEMINI_API_KEY and st.session_state.analysis_context:
            st.markdown("---")
            render_gemini_report_section(st.session_state.analysis_context)

            st.markdown("---")
            if st.session_state.chatbot:
                render_chatbot_section(st.session_state.chatbot)
        else:
            st.markdown("---")
            st.info(
                "💡 **Want AI-powered insights & chatbot?** "
                "Add your free Gemini API key to a `.env` file as "
                "`GEMINI_API_KEY=your_key` and restart the app. "
                "Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey)"
            )

        # ── AI Extensions — always rendered after analysis ──
        st.markdown("---")
        render_alerts_section(st.session_state.change_results)

        st.markdown("---")
        render_carbon_section(st.session_state.change_results)

        st.markdown("---")
        render_trend_section(st.session_state.change_results)

        st.markdown("---")
        render_vision_section(
            st.session_state.old_bytes,
            st.session_state.new_bytes
        )

        st.markdown("---")
        render_location_section(st.session_state.change_results)

        st.markdown("---")
        render_species_section(st.session_state.change_results)

    elif not st.session_state.old_bytes and not st.session_state.new_bytes:
        # ── Welcome / empty state ──
        render_welcome_state()

    elif st.session_state.old_bytes and not st.session_state.new_bytes:
        st.info("✅ Old image uploaded. Now upload the **new (recent) image** in the sidebar.")

    elif not st.session_state.old_bytes and st.session_state.new_bytes:
        st.info("✅ New image uploaded. Now upload the **old (baseline) image** in the sidebar.")

    elif (st.session_state.old_bytes and st.session_state.new_bytes
          and not st.session_state.analysis_done):
        st.success("✅ Both images uploaded. Press **🔍 Run Analysis** in the sidebar to begin.")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    main()