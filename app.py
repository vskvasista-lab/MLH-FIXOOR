"""app.py: Streamlit UI for FIXOOR: Error Screenshot & Endpoint Explainer.

Design Implementation:
- High-fidelity visual styling matching the reference mockup:
    * Custom Plus Jakarta Sans typography with rich geometric weights
    * Background wallpaper with icy mountains, floating 3D glass spheres, and ethereal blue ambient lighting
    * Floating glassmorphic sidebar with rounded right edges and frosted backdrop blur
    * FIXOOR branding with vibrant blue gradient typography
    * Custom pill-tab switcher with glowing royal-blue active tab state
    * Precision-styled drag-and-drop cloud upload zone with dashed border
    * Polished empty-state vector illustration with document sheet and magnifying glass
    * Full-width gradient action buttons with icon accents
- Clean modular architecture:
    * analyzer (Gemma 4 GenAI analysis, prompts, JSON schema recovery)
    * fetcher (secure HTTP request extraction & BeautifulSoup HTML cleaner)
    * imaging (Pillow image upscaling, luminance detection, contrast enhancement)
- Employs importlib.reload on custom modules to avoid Streamlit in-memory module caching bugs.
- Safely guards st.secrets against StreamlitSecretNotFoundError.
"""

import base64
import importlib
import os
from PIL import Image
import streamlit as st
from google.genai import errors

# Reload custom modules dynamically to guarantee fresh code on every Streamlit rerun
import analyzer
import fetcher
import imaging

importlib.reload(analyzer)
importlib.reload(fetcher)
importlib.reload(imaging)

from analyzer import (
    DEFAULT_MODEL_NAME,
    analyze_screenshot,
    analyze_url_error,
)
from fetcher import FetchError, fetch_error_from_url
from imaging import preprocess_image

# ------------------------------------------------------------------------------
# App Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="FIXOOR - Error Screenshot & Endpoint Explainer",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Wallpaper Background Resolution (Base64 for instant zero-lag rendering)
# ------------------------------------------------------------------------------
bg_path = os.path.join(os.path.dirname(__file__), "assets", "background.jpg")
bg_css_rule = ""
if os.path.exists(bg_path):
    try:
        with open(bg_path, "rb") as f_bg:
            bg_b64 = base64.b64encode(f_bg.read()).decode("utf-8")
        bg_css_rule = f"""
        background-image: url('data:image/jpeg;base64,{bg_b64}') !important;
        background-repeat: no-repeat !important;
        background-position: center bottom !important;
        background-size: cover !important;
        background-attachment: fixed !important;
        """
    except Exception:
        bg_css_rule = """
        background: radial-gradient(circle at 85% 15%, rgba(191, 219, 254, 0.45) 0%, transparent 45%),
                    linear-gradient(180deg, #f8fbff 0%, #edf4fc 45%, #e1effc 100%) !important;
        """
else:
    bg_css_rule = """
    background: radial-gradient(circle at 85% 15%, rgba(191, 219, 254, 0.45) 0%, transparent 45%),
                linear-gradient(180deg, #f8fbff 0%, #edf4fc 45%, #e1effc 100%) !important;
    """

# ------------------------------------------------------------------------------
# High-Fidelity Custom CSS
# ------------------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800;900&display=swap');

    /* Global Typography & Canvas */
    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    .stApp {{
        {bg_css_rule}
        color: #1e293b !important;
    }}

    /* Make top Streamlit header bar transparent */
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}

    /* Content Area Max Width & Spacing */
    .block-container {{
        padding-top: 1.75rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 1400px !important;
    }}

    /* Floating Rounded Glassmorphic Sidebar */
    [data-testid="stSidebar"] {{
        background: rgba(255, 255, 255, 0.88) !important;
        backdrop-filter: blur(28px) !important;
        -webkit-backdrop-filter: blur(28px) !important;
        border-right: 1px solid rgba(226, 232, 240, 0.85) !important;
        border-top-right-radius: 26px !important;
        border-bottom-right-radius: 26px !important;
        box-shadow: 6px 0 32px rgba(148, 163, 184, 0.12) !important;
    }}

    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.25rem !important;
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }}

    /* Brand Header in Sidebar */
    .brand-container {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 4px 0 16px 0;
        border-bottom: 1px solid rgba(226, 232, 240, 0.85);
        margin-bottom: 16px;
    }}

    .brand-icon-box {{
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.12);
    }}

    .brand-title {{
        font-size: 20px;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0f172a;
        line-height: 1.15;
    }}

    .brand-tagline {{
        font-size: 11px;
        font-weight: 500;
        color: #64748b;
        letter-spacing: 0.02em;
        margin-top: 2px;
    }}

    /* Status Pill */
    .status-pill {{
        display: flex;
        align-items: center;
        gap: 8px;
        background: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 9px 12px;
        border-radius: 10px;
        font-size: 11.5px;
        font-weight: 600;
        margin-bottom: 14px;
    }}

    .status-pill-icon {{
        background: #10b981;
        color: white;
        border-radius: 4px;
        width: 17px;
        height: 17px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: bold;
    }}

    /* Sidebar Section Titles */
    .sidebar-heading {{
        font-size: 14px;
        font-weight: 700;
        color: #1e293b;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 14px;
        margin-bottom: 10px;
    }}

    /* Hero Section */
    .ai-badge-wrap {{
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 6px;
    }}

    .ai-badge {{
        color: #2563eb;
        font-weight: 800;
        font-size: 11px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 5px;
    }}

    .ai-badge-line {{
        width: 44px;
        height: 1.5px;
        background: #93c5fd;
        border-radius: 2px;
    }}

    .hero-title {{
        font-size: 52px;
        font-weight: 900;
        letter-spacing: -0.03em;
        color: #0f172a;
        margin: 0;
        line-height: 1.05;
    }}

    .hero-title-gradient {{
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 45%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}

    .hero-subtitle {{
        font-size: 23px;
        font-weight: 800;
        color: #0f172a;
        margin-top: 6px;
        margin-bottom: 10px;
        letter-spacing: -0.02em;
    }}

    .hero-description {{
        font-size: 14px;
        color: #475569;
        max-width: 820px;
        line-height: 1.6;
        margin-bottom: 22px;
    }}

    /* Pill Tabs Styling */
    div[data-baseweb="tab-list"] {{
        background: transparent !important;
        border-bottom: none !important;
        gap: 12px !important;
        padding: 0 0 16px 0 !important;
    }}

    div[data-baseweb="tab-border"] {{
        display: none !important;
    }}

    button[data-baseweb="tab"] {{
        background: rgba(255, 255, 255, 0.75) !important;
        border: 1px solid rgba(203, 213, 225, 0.85) !important;
        border-radius: 12px !important;
        padding: 9px 20px !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: #475569 !important;
        box-shadow: 0 2px 6px rgba(148, 163, 184, 0.05) !important;
        transition: all 0.2s ease !important;
    }}

    button[data-baseweb="tab"]:hover {{
        background: #ffffff !important;
        color: #0f172a !important;
        border-color: #94a3b8 !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 50%, #3b82f6 100%) !important;
        border: 1px solid #2563eb !important;
        color: #ffffff !important;
        box-shadow: 0 4px 16px rgba(37, 99, 235, 0.4) !important;
    }}

    /* Container Border Glass Cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255, 255, 255, 0.90) !important;
        backdrop-filter: blur(24px) !important;
        -webkit-backdrop-filter: blur(24px) !important;
        border: 1px solid rgba(255, 255, 255, 0.95) !important;
        border-radius: 22px !important;
        box-shadow: 0 12px 36px rgba(148, 163, 184, 0.12) !important;
        padding: 24px !important;
        min-height: 470px !important;
    }}

    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        border: none !important;
    }}

    /* Card Headers */
    .card-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 4px;
    }}

    .card-title {{
        font-size: 17px;
        font-weight: 700;
        color: #0f172a;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .card-icon-badge {{
        width: 32px;
        height: 32px;
        background: #e0edff;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
    }}

    .help-circle {{
        width: 18px;
        height: 18px;
        border-radius: 50%;
        border: 1.5px solid #94a3b8;
        color: #94a3b8;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 11px;
        font-weight: 700;
        cursor: pointer;
    }}

    .card-subtitle {{
        font-size: 12.5px;
        color: #64748b;
        margin-top: 4px;
        margin-bottom: 12px;
    }}

    /* Custom File Uploader Dropzone matching exact mockup */
    [data-testid="stFileUploader"] {{
        margin-top: 4px;
        margin-bottom: 12px;
    }}

    [data-testid="stFileUploadDropzone"] {{
        background: rgba(248, 251, 255, 0.75) !important;
        border: 2px dashed #93c5fd !important;
        border-radius: 16px !important;
        padding: 24px 16px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
    }}

    [data-testid="stFileUploadDropzone"]:hover {{
        background: rgba(239, 246, 255, 0.95) !important;
        border-color: #3b82f6 !important;
    }}

    /* Hide the default browse button to display a clean unified dropzone */
    [data-testid="stFileUploadDropzone"] button {{
        display: none !important;
    }}

    /* Style the cloud upload icon */
    [data-testid="stFileUploadDropzone"] svg {{
        width: 36px !important;
        height: 36px !important;
        color: #2563eb !important;
        fill: #2563eb !important;
        margin-bottom: 6px !important;
    }}

    /* Text replacements for upload prompt */
    [data-testid="stFileUploadDropzone"] span {{
        font-size: 0px !important;
    }}
    [data-testid="stFileUploadDropzone"] span::before {{
        content: "Click to upload or drag and drop" !important;
        font-size: 13.5px !important;
        font-weight: 600 !important;
        color: #0f172a !important;
        display: block !important;
    }}

    [data-testid="stFileUploadDropzone"] small {{
        font-size: 0px !important;
    }}
    [data-testid="stFileUploadDropzone"] small::before {{
        content: "PNG or JPG • Max 200MB" !important;
        font-size: 11px !important;
        color: #94a3b8 !important;
        display: block !important;
        margin-top: 3px !important;
    }}

    /* Uploaded File Pill */
    [data-testid="stFileUploaderFile"] {{
        background: #ffffff !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 12px !important;
        padding: 10px 14px !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.08) !important;
    }}

    /* Notice Bars */
    .notice-bar {{
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #2563eb;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 12.5px;
        font-weight: 500;
        margin: 12px 0 16px 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }}

    .notice-bar-gray {{
        font-size: 12.5px;
        color: #64748b;
        margin-top: 4px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    /* Primary Action Buttons */
    div.stButton > button[kind="primary"] {{
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 14.5px !important;
        font-weight: 600 !important;
        height: 48px !important;
        box-shadow: 0 4px 18px rgba(37, 99, 235, 0.35) !important;
        transition: all 0.2s ease !important;
    }}

    div.stButton > button[kind="primary"]:hover {{
        background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%) !important;
        box-shadow: 0 6px 22px rgba(37, 99, 235, 0.45) !important;
        transform: translateY(-1px) !important;
    }}

    /* Clean Input Fields */
    input[type="text"], input[type="password"] {{
        border-radius: 10px !important;
        border: 1px solid #cbd5e1 !important;
        font-size: 13px !important;
        background: #ffffff !important;
    }}

    /* Empty State Right Card */
    .empty-state-box {{
        text-align: center;
        padding: 24px 16px 16px 16px;
    }}

    .empty-state-illustration {{
        width: 140px;
        height: 105px;
        margin: 0 auto 16px auto;
        display: flex;
        align-items: center;
        justify-content: center;
    }}

    .empty-state-title {{
        font-size: 16.5px;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 6px;
    }}

    .empty-state-subtitle {{
        font-size: 13px;
        color: #64748b;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# API Key Resolution
# ------------------------------------------------------------------------------
env_file = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_file):
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GEMINI_API_KEY=") and not line.startswith("#"):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val and val not in ["your_api_key_here", "your_gemini_api_key"]:
                        os.environ["GEMINI_API_KEY"] = val
    except Exception:
        pass

resolved_key = ""
try:
    if "GEMINI_API_KEY" in st.secrets:
        resolved_key = str(st.secrets["GEMINI_API_KEY"]).strip()
except Exception:
    pass

if not resolved_key and os.environ.get("GEMINI_API_KEY"):
    resolved_key = os.environ.get("GEMINI_API_KEY", "").strip()

if resolved_key in ["your_api_key_here", "your_gemini_api_key", "your_api_key", "<PASTE API KEY HERE>"]:
    resolved_key = ""


# ------------------------------------------------------------------------------
# Session State Initialization
# ------------------------------------------------------------------------------
if "active_analysis" not in st.session_state:
    st.session_state["active_analysis"] = None

if "active_raw_text" not in st.session_state:
    st.session_state["active_raw_text"] = None

if "active_image" not in st.session_state:
    st.session_state["active_image"] = None

if "active_processed_image" not in st.session_state:
    st.session_state["active_processed_image"] = None

if "active_prep_info" not in st.session_state:
    st.session_state["active_prep_info"] = None

if "last_uploaded_name" not in st.session_state:
    st.session_state["last_uploaded_name"] = None

if "url_preview_data" not in st.session_state:
    st.session_state["url_preview_data"] = None

if "active_diagnosis_type" not in st.session_state:
    st.session_state["active_diagnosis_type"] = "screenshot"


# ==============================================================================
# SIDEBAR: Custom High-Fidelity Panel
# ==============================================================================
with st.sidebar:
    # 1. FIXOOR Brand Header
    st.markdown(
        """
        <div class="brand-container">
            <div class="brand-icon-box">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#2563eb" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z"/>
                </svg>
            </div>
            <div>
                <div class="brand-title">FIXOOR</div>
                <div class="brand-tagline">Explain &bull; Understand &bull; Fix</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Settings Section
    st.markdown(
        """
        <div class="sidebar-heading">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#334155" stroke-width="2.2"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
            Settings
        </div>
        """,
        unsafe_allow_html=True,
    )

    if resolved_key:
        api_key = resolved_key
        st.markdown(
            """
            <div class="status-pill">
                <span class="status-pill-icon">✓</span>
                <span>GEMINI_API_KEY loaded securely.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        api_key = st.text_input(
            "Gemini API Key:",
            type="password",
            placeholder="Enter your API key...",
            help="Obtain an API key from https://aistudio.google.com/",
        ).strip()
        if not api_key:
            st.info("💡 Set `GEMINI_API_KEY` in `.env` or paste it above.")

    model_name = st.text_input(
        "Gemma 4 Model Name:",
        value=DEFAULT_MODEL_NAME,
        help="Default recommended model: gemma-4-26b-a4b-it",
    ).strip()

    st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #e2e8f0;' />", unsafe_allow_html=True)

    # 3. Explanation Depth
    st.markdown('<div class="sidebar-heading">🎯 Explanation Depth</div>', unsafe_allow_html=True)
    explanation_level = st.radio(
        "Choose Level:",
        options=["Beginner", "Intermediate", "Expert"],
        index=0,
        help="Beginner uses intuitive analogies; Expert provides low-level runtime insights.",
    )

    st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #e2e8f0;' />", unsafe_allow_html=True)

    # 4. Multi-Language Selector
    st.markdown('<div class="sidebar-heading">🌐 Language</div>', unsafe_allow_html=True)
    language = st.selectbox(
        "Explanation Language:",
        options=["English", "Hindi", "Telugu", "Tamil", "Spanish"],
        index=0,
        help="Diagnostic explanations will be translated; code and commands remain in English.",
    )

    st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #e2e8f0;' />", unsafe_allow_html=True)

    # 5. Image Preprocessing Controls
    st.markdown('<div class="sidebar-heading">🖼️ Screenshot Preprocessing</div>', unsafe_allow_html=True)
    enhance_toggle = st.checkbox(
        "Enhance screenshot",
        value=True,
        help="Auto-upscale small images and optimize contrast for OCR clarity.",
    )
    invert_toggle = st.checkbox(
        "Invert dark-theme screenshots",
        value=False,
        disabled=not enhance_toggle,
        help="Inverts dark backgrounds to light if the screenshot is detected as dark-themed.",
    )


# ==============================================================================
# MAIN PAGE: Hero Header & Content
# ==============================================================================
st.markdown(
    """
    <div class="ai-badge-wrap">
        <span class="ai-badge">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="#2563eb">
                <path d="M12 0L14.59 9.41L24 12L14.59 14.59L12 24L9.41 14.59L0 12L9.41 9.41L12 0Z"/>
            </svg>
            AI POWERED
        </span>
        <div class="ai-badge-line"></div>
    </div>
    <h1 class="hero-title">FIX<span class="hero-title-gradient">OOR</span></h1>
    <h2 class="hero-subtitle">Error Screenshot &amp; Endpoint Explainer</h2>
    <p class="hero-description">
        Upload a screenshot of your terminal error/traceback OR fetch an HTTP error page/API endpoint directly.
        <strong style="color: #1e293b;">Gemma 4</strong> will analyze the problem, diagnose root causes, and provide exact, copyable fix commands.
    </p>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------------------
# TWO COLUMNS: Left (Inputs & Actions) vs Right (Diagnostics & Empty State)
# ------------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

# ------------------------------------------------------------------------------
# LEFT COLUMN: Tabs for Screenshot Upload & Error Page URL
# ------------------------------------------------------------------------------
with col_left:
    tab_screenshot, tab_url = st.tabs(["🖼️ Screenshot Upload", "🔗 Error Page URL"])

    # --- TAB 1: Screenshot Upload Card ---
    with tab_screenshot:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-header">
                    <div class="card-title">
                        <div class="card-icon-badge">📷</div>
                        Upload Error Screenshot
                    </div>
                    <span class="help-circle" title="Upload terminal screenshot or traceback image">?</span>
                </div>
                <div class="card-subtitle">Upload an error screenshot (PNG or JPG):</div>
                """,
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                "Upload an error screenshot (PNG or JPG):",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
                key="screenshot_uploader",
            )

            # Detect new upload and process
            if uploaded_file is not None and uploaded_file.name != st.session_state["last_uploaded_name"]:
                st.session_state["last_uploaded_name"] = uploaded_file.name
                try:
                    raw_img = Image.open(uploaded_file)
                    st.session_state["active_image"] = raw_img
                    proc_img, p_info = preprocess_image(
                        raw_img,
                        enhance=enhance_toggle,
                        invert_dark=invert_toggle,
                    )
                    st.session_state["active_processed_image"] = proc_img
                    st.session_state["active_prep_info"] = p_info
                    st.session_state["active_analysis"] = None
                    st.session_state["active_raw_text"] = None
                    st.session_state["url_preview_data"] = None
                    st.session_state["active_diagnosis_type"] = "screenshot"
                except Exception as img_err:
                    st.error(f"Failed to read image file: {img_err}")
                    st.session_state["active_image"] = None

            # Display preview or empty notice
            if st.session_state["active_image"] is not None:
                st.image(
                    st.session_state["active_image"],
                    caption="Uploaded Screenshot (Original)",
                    use_container_width=True,
                )

                if st.session_state["active_processed_image"] is not None:
                    p_info = st.session_state.get("active_prep_info") or {}
                    exp_title = "🔍 Enhanced Screenshot Preview"
                    if p_info.get("is_dark"):
                        exp_title += " (Dark Theme Detected 🌙)"
                    if p_info.get("upscaled"):
                        exp_title += " (Auto-Upscaled 🚀)"

                    with st.expander(exp_title, expanded=False):
                        st.image(
                            st.session_state["active_processed_image"],
                            caption=f"Processed (Luminance: {p_info.get('avg_luminance', 'N/A')}, Size: {st.session_state['active_processed_image'].size})",
                            use_container_width=True,
                        )
            else:
                st.markdown(
                    """
                    <div class="notice-bar">
                        <span>👆</span> Upload an error screenshot above to begin.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            explain_screenshot_btn = st.button(
                "🪄 Explain Screenshot Error  ➔",
                type="primary",
                use_container_width=True,
                key="btn_explain_screenshot",
            )

    # --- TAB 2: URL Error Diagnosis Card ---
    with tab_url:
        with st.container(border=True):
            st.markdown(
                """
                <div class="card-header">
                    <div class="card-title">
                        <div class="card-icon-badge">🔗</div>
                        Diagnose Error from URL
                    </div>
                    <span class="help-circle" title="Fetch live HTTP error response from server endpoint">?</span>
                </div>
                <div class="card-subtitle">
                    Inspect live HTTP error responses (e.g. 500, 502, 404, 403, CORS) from web endpoints.
                </div>
                """,
                unsafe_allow_html=True,
            )

            target_url = st.text_input(
                "Target Endpoint URL:",
                placeholder="https://example.com/api/test or http://localhost:3000/error",
                key="target_url_input",
            )

            url_environment = st.radio(
                "Endpoint Type:",
                options=["Local / Development URL", "Public Web URL"],
                index=0,
                horizontal=True,
            )
            is_local = url_environment == "Local / Development URL"

            fetch_and_diagnose_btn = st.button(
                "🌐 Fetch & Diagnose URL  ➔",
                type="primary",
                use_container_width=True,
                key="btn_fetch_url",
            )

            if st.session_state["url_preview_data"] is not None:
                preview = st.session_state["url_preview_data"]
                st.markdown("<hr style='margin: 16px 0; border: none; border-top: 1px solid #e2e8f0;' />", unsafe_allow_html=True)
                status_code = preview["status_code"]
                status_color = "🔴" if status_code >= 500 else ("🟡" if status_code >= 400 else "🟢")
                st.markdown(f"**Status:** {status_color} `HTTP {status_code} {preview['status_text']}`")
                st.caption(f"**URL:** `{preview['url']}`")

                with st.expander("📄 Extracted Visible Body Text", expanded=True):
                    st.code(preview["raw_preview"], language="text")

                if preview.get("headers"):
                    with st.expander("🔍 Key Response Headers", expanded=False):
                        for h_name, h_val in preview["headers"].items():
                            st.markdown(f"- **`{h_name}`:** `{h_val}`")


# ------------------------------------------------------------------------------
# RIGHT COLUMN: Diagnostic & Fix Card (Matching Empty State & Results)
# ------------------------------------------------------------------------------
with col_right:
    with st.container(border=True):
        st.markdown(
            """
            <div class="card-header">
                <div class="card-title">
                    <span style="font-size: 19px;">💡</span> Diagnostic &amp; Fix
                </div>
            </div>
            <div class="notice-bar-gray">
                <span>👉</span> Upload a screenshot or enter an Error Page URL on the left and click Diagnose to view the report.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Trigger Screenshot Analysis
        if explain_screenshot_btn:
            if st.session_state["active_image"] is None:
                st.warning("⚠️ Please upload an error screenshot first.")
            elif not api_key:
                st.error("⚠️ Missing API Key! Please enter your Gemini API Key in the sidebar or `.env` file.")
            else:
                st.session_state["active_diagnosis_type"] = "screenshot"
                st.session_state["url_preview_data"] = None
                with st.spinner(f"Analyzing screenshot with {model_name} ({explanation_level} level, {language})..."):
                    try:
                        img_to_send = st.session_state["active_processed_image"] or st.session_state["active_image"]
                        parsed, raw_text, warning = analyze_screenshot(
                            image=img_to_send,
                            api_key=api_key,
                            model_name=model_name,
                            explanation_level=explanation_level,
                            language=language,
                        )
                        if warning:
                            st.info(f"ℹ️ {warning}")

                        st.session_state["active_analysis"] = parsed
                        st.session_state["active_raw_text"] = raw_text

                    except errors.ClientError as c_err:
                        if c_err.code == 429:
                            st.error("⏳ Rate limit exceeded. Please wait a few moments before trying again.")
                        else:
                            st.error(f"❌ API Client Error ({c_err.code}): {c_err.message}")
                    except errors.ServerError as s_err:
                        st.error(f"❌ Server Error ({s_err.code}): Google endpoint busy. Please retry shortly.")
                    except errors.APIError as a_err:
                        st.error(f"❌ Gemini API Error ({a_err.code}): {a_err.message}")
                    except Exception as ex:
                        st.error(f"❌ Unexpected error: `{str(ex)}`. Please check your API key and connection.")

        # 2. Trigger URL Analysis
        if fetch_and_diagnose_btn:
            if not target_url or not target_url.strip():
                st.warning("⚠️ Please enter a target URL to diagnose.")
            elif not api_key:
                st.error("⚠️ Missing API Key! Please enter your Gemini API Key in the sidebar or `.env` file.")
            else:
                st.session_state["active_diagnosis_type"] = "url"
                fetch_data = None

                with st.spinner(f"Connecting to `{target_url}` and extracting response..."):
                    try:
                        fetch_data = fetch_error_from_url(
                            url=target_url,
                            is_local=is_local,
                            timeout=10,
                        )
                        st.session_state["url_preview_data"] = fetch_data
                    except FetchError as f_err:
                        st.error(str(f_err))
                        st.session_state["url_preview_data"] = None
                    except Exception as gen_err:
                        st.error(f"❌ Unexpected error while fetching URL: {gen_err}")
                        st.session_state["url_preview_data"] = None

                if fetch_data is not None:
                    with st.spinner(f"Diagnosing endpoint with {model_name} ({explanation_level} level, {language})..."):
                        try:
                            parsed, raw_text, warning = analyze_url_error(
                                url=fetch_data["url"],
                                status_code=fetch_data["status_code"],
                                status_text=fetch_data["status_text"],
                                headers=fetch_data["headers"],
                                extracted_text=fetch_data["extracted_text"],
                                api_key=api_key,
                                model_name=model_name,
                                explanation_level=explanation_level,
                                language=language,
                            )
                            if warning:
                                st.info(f"ℹ️ {warning}")

                            st.session_state["active_analysis"] = parsed
                            st.session_state["active_raw_text"] = raw_text

                        except errors.ClientError as c_err:
                            if c_err.code == 429:
                                st.error("⏳ Rate limit exceeded. Please wait a few moments before trying again.")
                            else:
                                st.error(f"❌ API Client Error ({c_err.code}): {c_err.message}")
                        except errors.ServerError as s_err:
                            st.error(f"❌ Server Error ({s_err.code}): Google endpoint busy. Please retry shortly.")
                        except errors.APIError as a_err:
                            st.error(f"❌ Gemini API Error ({a_err.code}): {a_err.message}")
                        except Exception as ex:
                            st.error(f"❌ Unexpected error: `{str(ex)}`. Please check your API key and connection.")

        # ----------------------------------------------------------------------
        # Render Results or Empty State Illustration
        # ----------------------------------------------------------------------
        analysis = st.session_state["active_analysis"]
        raw_output = st.session_state["active_raw_text"]

        if analysis:
            severity = str(analysis.get("severity", "medium")).lower()
            if "high" in severity:
                sev_badge = "🔴 **HIGH SEVERITY**"
            elif "low" in severity:
                sev_badge = "🟢 **LOW SEVERITY**"
            else:
                sev_badge = "🟡 **MEDIUM SEVERITY**"

            st.markdown(f"### {analysis.get('error_type', 'Error Detected')}")
            st.markdown(
                f"**Status:** {sev_badge} &nbsp;|&nbsp; "
                f"**Level:** `{explanation_level}` &nbsp;|&nbsp; "
                f"**Language:** `{language}`"
            )
            st.markdown("---")

            # 1. Summary Card
            st.markdown("#### 📌 Summary")
            st.info(analysis.get("summary", "No summary provided."))

            # 2. Root Cause Card
            st.markdown("#### 🔍 Root Cause")
            st.write(analysis.get("root_cause", "No root cause identified."))

            # 3. Analogy Card
            if analysis.get("analogy"):
                st.markdown("#### 💡 Intuitive Analogy")
                st.success(f"*{analysis.get('analogy')}*")

            # 4. Numbered Fix Steps
            fix_steps = analysis.get("fix_steps", [])
            if fix_steps:
                st.markdown("#### 🛠️ Numbered Steps to Fix & Debug")
                for idx, step in enumerate(fix_steps, 1):
                    st.markdown(f"**Step {idx}: {step.get('title', 'Action')}**")
                    if step.get("explanation"):
                        st.write(step.get("explanation"))
                    cmd = step.get("command_or_code")
                    if cmd and cmd.strip():
                        st.caption("Click to copy:")
                        lang = "python" if any(k in cmd for k in ["def ", "import ", "print(", "class ", " = "]) else "bash"
                        st.code(cmd.strip(), language=lang)

            # 5. Corrected Code / Configuration
            corrected = analysis.get("corrected_code")
            if corrected and corrected.strip():
                st.markdown("#### 📋 Corrected Code / Configuration")
                st.code(corrected.strip(), language="nginx" if "nginx" in corrected.lower() else "python")

            # 6. Prevention Tip
            if analysis.get("prevention_tip"):
                st.markdown("#### 🛡️ Prevention Tip")
                st.warning(f"💡 {analysis.get('prevention_tip')}")

        elif raw_output:
            st.warning("⚠️ Note: Output could not be parsed as structured JSON. Displaying text response:")
            st.markdown(raw_output)

        else:
            # High-Precision Empty State Illustration matching mockup
            st.markdown(
                """
                <div class="empty-state-box">
                    <div class="empty-state-illustration">
                        <svg width="130" height="100" viewBox="0 0 130 100" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <!-- Soft drop shadow -->
                            <rect x="18" y="14" width="70" height="74" rx="10" fill="#94a3b8" opacity="0.08" />
                            <!-- Document Sheet -->
                            <rect x="16" y="12" width="70" height="74" rx="10" fill="#ffffff" stroke="#bfdbfe" stroke-width="2.5" />
                            <!-- Document content lines -->
                            <rect x="26" y="24" width="34" height="6" rx="3" fill="#3b82f6" />
                            <rect x="26" y="38" width="48" height="5" rx="2.5" fill="#e2e8f0" />
                            <rect x="26" y="49" width="38" height="5" rx="2.5" fill="#e2e8f0" />
                            <rect x="26" y="60" width="28" height="5" rx="2.5" fill="#e2e8f0" />
                            <!-- Magnifying Glass -->
                            <g>
                                <path d="M82 68L104 90" stroke="#2563eb" stroke-width="6.5" stroke-linecap="round" />
                                <circle cx="68" cy="52" r="23" fill="#eff6ff" stroke="#2563eb" stroke-width="4.5" />
                                <circle cx="68" cy="52" r="15" fill="#ffffff" opacity="0.75" />
                                <path d="M57 44A13 13 0 0 1 73 39" stroke="#ffffff" stroke-width="2.8" stroke-linecap="round" />
                            </g>
                            <!-- Ambient particle dots -->
                            <circle cx="94" cy="22" r="3" fill="#93c5fd" opacity="0.6" />
                            <circle cx="106" cy="34" r="2.2" fill="#60a5fa" opacity="0.5" />
                        </svg>
                    </div>
                    <div class="empty-state-title">Your detailed explanation will appear here</div>
                    <div class="empty-state-subtitle">Get clear explanations, root causes, and step-by-step fixes.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
