"""app.py: Streamlit UI for Error Screenshot Explainer (Stage 1).

Design Notes:
- Acts strictly as the view and controller layer. No direct image manipulation
  or GenAI API calls reside in this file.
- Preserves the established two-column layout: inputs & image preview on the left,
  diagnostic results & actionable fixes on the right.
- Secret resolution checks st.secrets safely inside a try-except block to prevent
  StreamlitSecretNotFoundError when no secrets.toml is configured, falling back
  to environment variables and .env.
- Commands and code snippets are explicitly rendered using st.code to provide
  native, one-click clipboard copying.
"""

import os
from PIL import Image
import streamlit as st
from google.genai import errors

from analyzer import (
    DEFAULT_MODEL_NAME,
    analyze_screenshot,
)
from imaging import preprocess_image

# ------------------------------------------------------------------------------
# App Configuration
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Error Screenshot Explainer",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# API Key Resolution
# Design Note: Prioritize st.secrets (Streamlit Cloud / local secrets.toml) safely,
# then local .env / environment variables, then fallback to user input.
# Guard against StreamlitSecretNotFoundError when secrets.toml does not exist.
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
    # StreamlitSecretNotFoundError occurs when no secrets.toml exists
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


# ==============================================================================
# SIDEBAR: Configuration & Controls
# ==============================================================================
st.sidebar.title("⚙️ Settings")

# 1. API Key Input
if resolved_key:
    api_key = resolved_key
    st.sidebar.success("✅ `GEMINI_API_KEY` loaded securely.")
else:
    api_key = st.sidebar.text_input(
        "Gemini API Key:",
        type="password",
        placeholder="Enter your API key...",
        help="Obtain an API key from https://aistudio.google.com/",
    ).strip()
    if not api_key:
        st.sidebar.info("💡 Set `GEMINI_API_KEY` in `.env` or paste it above.")

# 2. Model Configuration
model_name = st.sidebar.text_input(
    "Gemma 4 Model Name:",
    value=DEFAULT_MODEL_NAME,
    help="Default model identifier: gemma-4-26b-a4b-it",
).strip()

st.sidebar.markdown("---")

# 3. Explanation Depth
st.sidebar.subheader("🎯 Explanation Depth")
explanation_level = st.sidebar.radio(
    "Choose Level:",
    options=["Beginner", "Intermediate", "Expert"],
    index=0,
    help="Beginner uses intuitive analogies; Expert provides low-level runtime insights.",
)

# 4. Multi-Language Selector
st.sidebar.subheader("🌐 Language")
language = st.sidebar.selectbox(
    "Explanation Language:",
    options=["English", "Hindi", "Telugu", "Tamil", "Spanish"],
    index=0,
    help="Diagnostic explanations will be translated; code and commands remain in English.",
)

st.sidebar.markdown("---")

# 5. Image Preprocessing Controls
st.sidebar.subheader("🖼️ Screenshot Preprocessing")
enhance_toggle = st.sidebar.checkbox(
    "Enhance screenshot",
    value=True,
    help="Auto-upscale small images and optimize contrast for OCR clarity.",
)
invert_toggle = st.sidebar.checkbox(
    "Invert dark-theme screenshots",
    value=False,
    disabled=not enhance_toggle,
    help="Inverts dark backgrounds to light if the screenshot is detected as dark-themed.",
)


# ==============================================================================
# MAIN PAGE: Two-Column Layout
# ==============================================================================
st.title("🛠️ Error Screenshot Explainer")
st.markdown(
    "Upload a screenshot of your terminal error, exception, or stack trace. "
    "Multimodal **Gemma 4** will analyze the problem and provide a structured, beginner-friendly fix."
)

col_left, col_right = st.columns([1, 1], gap="large")

# ------------------------------------------------------------------------------
# LEFT COLUMN: Upload, Previews, & Controls
# ------------------------------------------------------------------------------
with col_left:
    st.subheader("📷 Screenshot Input")

    uploaded_file = st.file_uploader(
        "Upload an error screenshot (PNG or JPG):",
        type=["png", "jpg", "jpeg"],
        help="Upload an image displaying the terminal output, compiler error, or traceback.",
    )

    # Detect file change and trigger preprocessing
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
        except Exception as img_err:
            st.error(f"Failed to read image file: {img_err}")
            st.session_state["active_image"] = None

    # Render image preview if available
    if st.session_state["active_image"] is not None:
        st.image(
            st.session_state["active_image"],
            caption="Uploaded Screenshot (Original)",
            use_container_width=True,
        )

        # Show processed preview expander
        if st.session_state["active_processed_image"] is not None:
            p_info = st.session_state.get("active_prep_info") or {}
            expander_title = "🔍 Enhanced Screenshot Preview"
            if p_info.get("is_dark"):
                expander_title += " (Dark Theme Detected 🌙)"
            if p_info.get("upscaled"):
                expander_title += " (Auto-Upscaled 🚀)"

            with st.expander(expander_title, expanded=False):
                st.image(
                    st.session_state["active_processed_image"],
                    caption=(
                        f"Processed View (Avg Luminance: {p_info.get('avg_luminance', 'N/A')}, "
                        f"Size: {st.session_state['active_processed_image'].size})"
                    ),
                    use_container_width=True,
                )
                if p_info.get("upscaled"):
                    st.caption(
                        f"ℹ️ Upscaled from {p_info.get('original_size')} to {p_info.get('new_size')} "
                        "for crisp character recognition."
                    )
                if p_info.get("is_dark"):
                    st.caption("ℹ️ Dark theme detected: contrast and sharpness enhanced.")
    else:
        st.info("👆 Please upload an error screenshot above to get started.")

    # Action Button
    explain_btn = st.button("🚀 Explain this error", type="primary", use_container_width=True)


# ------------------------------------------------------------------------------
# RIGHT COLUMN: Structured Results
# ------------------------------------------------------------------------------
with col_right:
    st.subheader("💡 Diagnostic & Fix")

    # Handle Analysis Execution
    if explain_btn:
        if st.session_state["active_image"] is None:
            st.warning("⚠️ Please upload an error screenshot before requesting an explanation.")
        elif not api_key:
            st.error("⚠️ Missing API Key! Please enter your Gemini API Key in the sidebar or `.env` file.")
        else:
            with st.spinner(f"Analyzing error with {model_name} ({explanation_level} level, {language})..."):
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

    # --------------------------------------------------------------------------
    # Render Diagnostic Breakdown
    # --------------------------------------------------------------------------
    analysis = st.session_state["active_analysis"]
    raw_output = st.session_state["active_raw_text"]

    if analysis:
        # Severity Badge & Header
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

        # 4. Numbered Fix Steps with One-Click Copyable Code
        fix_steps = analysis.get("fix_steps", [])
        if fix_steps:
            st.markdown("#### 🛠️ Numbered Steps to Fix")
            for idx, step in enumerate(fix_steps, 1):
                st.markdown(f"**Step {idx}: {step.get('title', 'Action')}**")
                if step.get("explanation"):
                    st.write(step.get("explanation"))
                cmd = step.get("command_or_code")
                if cmd and cmd.strip():
                    st.caption("Click to copy:")
                    # Design Note: Detect language heuristically to enable proper syntax highlighting
                    lang = "python" if any(k in cmd for k in ["def ", "import ", "print(", "class ", " = "]) else "bash"
                    st.code(cmd.strip(), language=lang)

        # 5. Corrected Code
        corrected = analysis.get("corrected_code")
        if corrected and corrected.strip():
            st.markdown("#### 📋 Corrected Code")
            st.code(corrected.strip(), language="python")

        # 6. Prevention Tip
        if analysis.get("prevention_tip"):
            st.markdown("#### 🛡️ Prevention Tip")
            st.warning(f"💡 {analysis.get('prevention_tip')}")

    elif raw_output:
        # Fallback to plain text if JSON extraction failed
        st.warning("⚠️ Note: Output could not be parsed as structured JSON. Displaying text response:")
        st.markdown(raw_output)

    else:
        st.caption("👈 Upload a screenshot on the left and click **'Explain this error'** to see the diagnosis.")
