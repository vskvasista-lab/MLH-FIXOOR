import io
import json
import os
import re
import uuid
from datetime import datetime
from PIL import Image, ImageEnhance, ImageOps, ImageStat
import streamlit as st
from google import genai
from google.genai import errors, types

# ==============================================================================
# Model & App Configuration
# Default recommended Gemma 4 multimodal model on Gemini API
# ==============================================================================
DEFAULT_MODEL_NAME = "gemma-4-26b-a4b-it"
MAX_FILE_SIZE_MB = 5

st.set_page_config(
    page_title="Error Screenshot Explainer",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# Helper: Environment and API Key Setup
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

env_api_key = os.environ.get("GEMINI_API_KEY", "").strip()
if env_api_key in ["your_api_key_here", "your_gemini_api_key", "your_api_key", "<PASTE API KEY HERE>"]:
    env_api_key = ""

# ------------------------------------------------------------------------------
# State Initialization
# ------------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

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

if "active_code_context" not in st.session_state:
    st.session_state["active_code_context"] = ""

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "last_uploaded_name" not in st.session_state:
    st.session_state["last_uploaded_name"] = None


# ------------------------------------------------------------------------------
# Image Preprocessing (Pillow)
# ------------------------------------------------------------------------------
def preprocess_image(image: Image.Image, enhance: bool = True, invert_dark: bool = False) -> tuple[Image.Image, dict]:
    """
    Preprocess screenshot:
    - Converts to RGB.
    - Upscales small screenshots so terminal/code text is sharp for OCR.
    - Detects dark themes and enhances contrast or inverts if requested.
    """
    img = image.convert("RGB")
    info = {
        "original_size": image.size,
        "new_size": image.size,
        "upscaled": False,
        "is_dark": False,
        "avg_luminance": 0.0,
        "inverted": False,
        "enhanced": enhance,
    }

    if not enhance:
        return img, info

    w, h = img.size
    # 1. Upscale if small (< 900px width or < 450px height)
    if w < 900 or h < 450:
        scale = max(900 / max(w, 1), 450 / max(h, 1), 1.5)
        scale = min(scale, 3.0)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        info["upscaled"] = True
        info["new_size"] = new_size

    # 2. Luminance detection
    grayscale = img.convert("L")
    stat = ImageStat.Stat(grayscale)
    avg_lum = stat.mean[0]
    info["avg_luminance"] = round(avg_lum, 1)
    is_dark = avg_lum < 95
    info["is_dark"] = is_dark

    # 3. Contrast adjustment
    if is_dark:
        if invert_dark:
            img = ImageOps.invert(img)
            info["inverted"] = True
            img = ImageEnhance.Contrast(img).enhance(1.25)
        else:
            img = ImageOps.autocontrast(img, cutoff=1)
            img = ImageEnhance.Contrast(img).enhance(1.4)
            img = ImageEnhance.Sharpness(img).enhance(1.3)
    else:
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Sharpness(img).enhance(1.2)

    return img, info


# ------------------------------------------------------------------------------
# JSON Parsing and Sanitization
# ------------------------------------------------------------------------------
def extract_and_parse_json(text: str) -> dict | None:
    """Safely extracts JSON from model response text, stripping markdown fences."""
    if not text:
        return None

    cleaned = text.strip()
    # Strip markdown ```json ... ``` or ``` ... ```
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()

    # Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try to locate first '{' and last '}'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


# ------------------------------------------------------------------------------
# Markdown Export Generator
# ------------------------------------------------------------------------------
def build_markdown_report(data: dict, raw_text: str | None, code_context: str) -> str:
    """Builds a formatted Markdown document of the error analysis."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not data:
        return f"# Error Analysis Report\nGenerated on: {now_str}\n\n{raw_text or 'No analysis data available.'}"

    md = []
    md.append(f"# Error Analysis: {data.get('error_type', 'Error Breakdown')}")
    md.append(f"**Severity:** {data.get('severity', 'medium').upper()}  ")
    md.append(f"**Date:** {now_str}  \n")
    md.append("---")

    md.append("\n## Summary")
    md.append(data.get("summary", "N/A"))

    md.append("\n## Root Cause")
    md.append(data.get("root_cause", "N/A"))

    if data.get("analogy"):
        md.append("\n## Analogy")
        md.append(f"> 💡 *{data.get('analogy')}*")

    fix_steps = data.get("fix_steps", [])
    if fix_steps:
        md.append("\n## Numbered Fix Steps")
        for i, step in enumerate(fix_steps, 1):
            md.append(f"\n### Step {i}: {step.get('title', 'Action')}")
            md.append(f"{step.get('explanation', '')}")
            cmd = step.get("command_or_code")
            if cmd:
                md.append(f"\n```bash\n{cmd}\n```")

    if data.get("corrected_code"):
        md.append("\n## Corrected Code")
        md.append(f"```python\n{data.get('corrected_code')}\n```")

    if data.get("prevention_tip"):
        md.append("\n## Prevention Tip")
        md.append(f"🛡️ {data.get('prevention_tip')}")

    if code_context.strip():
        md.append("\n## Related Source Code Provided")
        md.append(f"```\n{code_context}\n```")

    return "\n".join(md)


# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.title("⚙️ Settings & Options")

# 1. API Key
if env_api_key:
    api_key = env_api_key
    st.sidebar.success("✅ `GEMINI_API_KEY` loaded.")
else:
    api_key = st.sidebar.text_input(
        "Gemini API Key:",
        type="password",
        placeholder="Enter your API key...",
        help="Get a free API key from https://aistudio.google.com/",
    ).strip()
    if not api_key:
        st.sidebar.info("💡 Set `GEMINI_API_KEY` in `.env` or paste it above.")

# 2. Model Selection
model_name = st.sidebar.text_input(
    "Gemma 4 Model Name:",
    value=DEFAULT_MODEL_NAME,
    help="Identifier for Gemma 4. Default: gemma-4-26b-a4b-it",
).strip()

# 3. Explanation Level
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Explanation Depth")
explanation_level = st.sidebar.radio(
    "Choose Level:",
    options=["Beginner", "Intermediate", "Expert"],
    index=0,
    help="Adjusts the technical depth and tone of the explanation.",
)

# 4. Language Selector
st.sidebar.subheader("🌐 Language")
language = st.sidebar.selectbox(
    "Response Language:",
    options=["English", "Hindi", "Telugu", "Tamil", "Spanish"],
    index=0,
    help="Explanations will be written in this language. Code and terminal commands stay in English.",
)

# 5. Image Preprocessing Controls
st.sidebar.markdown("---")
st.sidebar.subheader("🖼️ Screenshot Preprocessing")
enhance_toggle = st.sidebar.checkbox(
    "Enhance screenshot",
    value=True,
    help="Auto-upscale small images and optimize contrast for OCR readability.",
)
invert_toggle = st.sidebar.checkbox(
    "Invert dark-theme screenshots",
    value=False,
    disabled=not enhance_toggle,
    help="Convert dark backgrounds to bright paper-style background for improved clarity.",
)

# 6. History
st.sidebar.markdown("---")
st.sidebar.subheader("🕒 Session History")
if not st.session_state["history"]:
    st.sidebar.caption("No analyses yet in this session.")
else:
    for idx, item in enumerate(st.session_state["history"]):
        btn_label = f"[{item['timestamp']}] {item['error_type'][:18]}"
        if st.sidebar.button(btn_label, key=f"hist_btn_{item['id']}", use_container_width=True):
            st.session_state["active_analysis"] = item["analysis_data"]
            st.session_state["active_raw_text"] = item.get("raw_text")
            st.session_state["active_image"] = item["image"]
            st.session_state["active_processed_image"] = item.get("processed_image")
            st.session_state["active_prep_info"] = item.get("prep_info")
            st.session_state["active_code_context"] = item.get("code_context", "")
            st.session_state["chat_history"] = item.get("chat_history", [])
            st.rerun()

    if st.sidebar.button("🗑️ Clear History", use_container_width=True):
        st.session_state["history"] = []
        st.session_state["active_analysis"] = None
        st.session_state["active_raw_text"] = None
        st.session_state["active_image"] = None
        st.session_state["active_processed_image"] = None
        st.session_state["active_prep_info"] = None
        st.session_state["chat_history"] = []
        st.rerun()


# ==============================================================================
# MAIN VIEW: Two-Column Layout
# ==============================================================================
st.title("🛠️ Error Screenshot Explainer")
st.markdown(
    "Upload a screenshot of your terminal error, code exception, or stack trace. "
    "Multimodal **Gemma 4** will diagnose the error, explain it at your preferred skill level, and provide exact, copyable fixes."
)

col_left, col_right = st.columns([1, 1], gap="large")

# ------------------------------------------------------------------------------
# LEFT COLUMN: Upload, Previews & Code Context
# ------------------------------------------------------------------------------
with col_left:
    st.subheader("📷 Screenshot Input")

    uploaded_file = st.file_uploader(
        "Upload your error screenshot (PNG or JPG, max 5 MB):",
        type=["png", "jpg", "jpeg"],
        help="Upload an image showing the error message, terminal output, or stack trace.",
    )

    # Check file size (5 MB limit)
    is_oversized = False
    if uploaded_file is not None:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            is_oversized = True
            st.error(
                f"⚠️ File is too large ({uploaded_file.size / (1024 * 1024):.1f} MB). "
                f"Please upload an image smaller than {MAX_FILE_SIZE_MB} MB."
            )

    # Reset active state when new valid file uploaded
    if uploaded_file and uploaded_file.name != st.session_state["last_uploaded_name"] and not is_oversized:
        st.session_state["last_uploaded_name"] = uploaded_file.name
        try:
            raw_img = Image.open(uploaded_file)
            st.session_state["active_image"] = raw_img
            proc_img, p_info = preprocess_image(raw_img, enhance=enhance_toggle, invert_dark=invert_toggle)
            st.session_state["active_processed_image"] = proc_img
            st.session_state["active_prep_info"] = p_info
            st.session_state["active_analysis"] = None
            st.session_state["active_raw_text"] = None
            st.session_state["chat_history"] = []
        except Exception as img_err:
            st.error(f"Failed to load image: {img_err}")
            st.session_state["active_image"] = None

    # Display image preview
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
                    caption=f"Processed (Luminance: {p_info.get('avg_luminance', 'N/A')}, Size: {st.session_state['active_processed_image'].size})",
                    use_container_width=True,
                )
                if p_info.get("upscaled"):
                    st.caption(f"ℹ️ Upscaled from {p_info.get('original_size')} to {p_info.get('new_size')} for sharp OCR.")
                if p_info.get("is_dark"):
                    st.caption("ℹ️ Dark theme detected: contrast and sharpness boosted for clarity.")
    else:
        st.info("👆 Upload an error screenshot above to begin.")

    # Optional Code Context Expander
    st.markdown("---")
    with st.expander("📝 Optional: Paste Related Source Code", expanded=False):
        code_input = st.text_area(
            "Source Code Context:",
            value=st.session_state["active_code_context"],
            height=160,
            placeholder="Paste snippet of the script or function causing the error here...",
            help="Providing the source code helps Gemma 4 identify the exact line and give a tailored fix.",
        )
        st.session_state["active_code_context"] = code_input

    # Main Action Button
    explain_btn = st.button("🚀 Explain this error", type="primary", use_container_width=True)


# ------------------------------------------------------------------------------
# RIGHT COLUMN: Structured Results & Follow-up Chat
# ------------------------------------------------------------------------------
with col_right:
    st.subheader("💡 Diagnostic & Fix")

    # Handle Analysis Execution
    if explain_btn:
        if is_oversized:
            st.error("⚠️ Please choose an image smaller than 5 MB before proceeding.")
        elif st.session_state["active_image"] is None:
            st.warning("⚠️ Please upload an error screenshot before requesting an explanation.")
        elif not api_key:
            st.error("⚠️ Missing API Key! Please enter your Gemini API Key in the sidebar or check `.env`.")
        else:
            with st.spinner(f"Analyzing error with {model_name} ({explanation_level} level, {language})..."):
                try:
                    client = genai.Client(api_key=api_key)

                    # Prepare image bytes from processed image
                    img_to_send = st.session_state["active_processed_image"] or st.session_state["active_image"]
                    buf = io.BytesIO()
                    img_to_send.save(buf, format="PNG")
                    img_bytes = buf.getvalue()
                    image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")

                    # Structured system instructions
                    code_ctx = st.session_state["active_code_context"].strip()
                    code_section = f"\nRelated source code provided by user:\n```\n{code_ctx}\n```\n" if code_ctx else ""

                    prompt = f"""You are an expert senior software engineer and patient programming mentor.
Examine this error screenshot carefully.{code_section}

Target Audience Level: {explanation_level} developer.
Output Language: {language}.

Instructions:
1. Explain the error accurately for a {explanation_level} developer.
2. All explanations, summaries, analogies, fix steps, and prevention tips MUST be written in {language}.
3. CRITICAL: All code snippets, terminal commands, function names, error types, and variable names MUST remain in English.
4. You MUST return your entire response as a valid JSON code block enclosed in ```json ... ``` with this exact structure:

```json
{{
  "error_type": "Exact name of the error or exception",
  "severity": "low" or "medium" or "high",
  "summary": "1-2 sentence overview of what went wrong",
  "root_cause": "Clear explanation of why this error occurred",
  "analogy": "A simple real-world analogy to illustrate the problem",
  "fix_steps": [
    {{
      "title": "Short title for step",
      "explanation": "Detailed explanation of what to do",
      "command_or_code": "Exact command to run or line of code to change (or empty string if none)"
    }}
  ],
  "corrected_code": "Full corrected code snippet if applicable, otherwise empty string",
  "prevention_tip": "How to prevent this error in the future"
}}
```
"""

                    # Call Model with graceful fallback for server 500 errors
                    response = None
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[prompt, image_part],
                        )
                    except errors.ServerError as serv_err:
                        if model_name != "gemma-4-26b-a4b-it":
                            st.warning(f"Model '{model_name}' hit a temporary server error ({serv_err.code}). Retrying with 'gemma-4-26b-a4b-it'...")
                            response = client.models.generate_content(
                                model="gemma-4-26b-a4b-it",
                                contents=[prompt, image_part],
                            )
                        else:
                            raise

                    if not response or not response.text:
                        st.warning("⚠️ The model returned an empty response. Please try again.")
                    else:
                        raw_output = response.text
                        parsed_json = extract_and_parse_json(raw_output)

                        # Retry once if JSON parsing failed
                        if not parsed_json:
                            try:
                                retry_prompt = f"Convert the following analysis into valid JSON matching the requested schema. Return only the JSON block:\n\n{raw_output}"
                                retry_resp = client.models.generate_content(
                                    model="gemma-4-26b-a4b-it",
                                    contents=[retry_prompt],
                                )
                                if retry_resp and retry_resp.text:
                                    parsed_json = extract_and_parse_json(retry_resp.text)
                            except Exception:
                                parsed_json = None

                        st.session_state["active_analysis"] = parsed_json
                        st.session_state["active_raw_text"] = raw_output
                        st.session_state["chat_history"] = []

                        # Save to session history
                        history_item = {
                            "id": str(uuid.uuid4())[:8],
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "error_type": parsed_json.get("error_type", "Detected Error") if parsed_json else "Error",
                            "summary": parsed_json.get("summary", "") if parsed_json else raw_output[:60],
                            "analysis_data": parsed_json,
                            "raw_text": raw_output,
                            "image": st.session_state["active_image"],
                            "processed_image": st.session_state["active_processed_image"],
                            "prep_info": st.session_state["active_prep_info"],
                            "code_context": st.session_state["active_code_context"],
                            "chat_history": [],
                        }
                        st.session_state["history"].insert(0, history_item)

                except errors.ClientError as c_err:
                    if c_err.code == 429:
                        st.error("⏳ Rate limit exceeded. Please wait a few moments before trying again.")
                    else:
                        st.error(f"❌ API Client Error ({c_err.code}): {c_err.message}")
                except errors.ServerError as s_err:
                    st.error(f"❌ Server Error ({s_err.code}): Google server is busy. Please try again shortly.")
                except errors.APIError as a_err:
                    st.error(f"❌ Gemini API Error ({a_err.code}): {a_err.message}")
                except Exception as ex:
                    st.error(f"❌ An error occurred: `{str(ex)}`. Please check your API key and connection.")

    # --------------------------------------------------------------------------
    # Render Active Analysis
    # --------------------------------------------------------------------------
    analysis = st.session_state["active_analysis"]
    raw_text = st.session_state["active_raw_text"]

    if analysis:
        # Severity Badge & Error Type Header
        severity = str(analysis.get("severity", "medium")).lower()
        if "high" in severity:
            sev_badge = "🔴 **HIGH SEVERITY**"
        elif "low" in severity:
            sev_badge = "🟢 **LOW SEVERITY**"
        else:
            sev_badge = "🟡 **MEDIUM SEVERITY**"

        st.markdown(f"### {analysis.get('error_type', 'Error Detected')}")
        st.markdown(f"**Status:** {sev_badge} &nbsp;|&nbsp; **Level:** `{explanation_level}` &nbsp;|&nbsp; **Language:** `{language}`")
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

        # 4. Numbered Fix Steps with Copyable Code
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
                    # Detect if python or bash
                    lang = "python" if any(k in cmd for k in ["def ", "import ", "print(", "class ", "="]) else "bash"
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

        # 7. Export Markdown Button
        st.markdown("---")
        report_md = build_markdown_report(analysis, raw_text, st.session_state["active_code_context"])
        safe_fname = re.sub(r"[^a-zA-Z0-9_-]", "_", analysis.get("error_type", "error_analysis"))
        st.download_button(
            label="📥 Export Analysis as Markdown",
            data=report_md,
            file_name=f"{safe_fname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    elif raw_text:
        # Fallback to plain text rendering
        st.warning("⚠️ Note: Structured JSON was not returned. Rendering raw model response:")
        st.markdown(raw_text)

        report_md = build_markdown_report(None, raw_text, st.session_state["active_code_context"])
        st.download_button(
            label="📥 Export Raw Analysis as Markdown",
            data=report_md,
            file_name=f"error_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    else:
        st.caption("👈 Upload a screenshot on the left and click **'Explain this error'** to see the structured diagnosis.")

    # --------------------------------------------------------------------------
    # Follow-Up Chat Section
    # --------------------------------------------------------------------------
    if analysis or raw_text:
        st.markdown("---")
        st.markdown("### 💬 Follow-Up Chat")
        st.caption("Have questions about this fix? Ask below and Gemma 4 will guide you!")

        # Display previous chat messages
        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat Input
        user_query = st.chat_input("Ask a follow-up question about this error or fix...")

        if user_query:
            # Display user message immediately
            st.session_state["chat_history"].append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Generate Assistant Reply
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        client = genai.Client(api_key=api_key)

                        # Build conversational context
                        chat_context = [
                            f"You are a patient programming mentor answering follow-up questions about this error at a {explanation_level} developer level.",
                            f"Language: {language}. Code snippets must remain in English.",
                            f"Original error analysis:\n{raw_text or json.dumps(analysis, indent=2)}",
                        ]

                        # Add image if available
                        if st.session_state["active_processed_image"] is not None:
                            buf = io.BytesIO()
                            st.session_state["active_processed_image"].save(buf, format="PNG")
                            chat_context.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))

                        # Add chat history
                        for turn in st.session_state["chat_history"]:
                            chat_context.append(f"{turn['role'].capitalize()}: {turn['content']}")

                        chat_resp = client.models.generate_content(
                            model=model_name,
                            contents=chat_context,
                        )

                        if chat_resp and chat_resp.text:
                            assistant_text = chat_resp.text
                            st.markdown(assistant_text)
                            st.session_state["chat_history"].append({"role": "assistant", "content": assistant_text})
                        else:
                            st.warning("No response received from the assistant. Please try again.")

                    except Exception as chat_err:
                        st.error(f"❌ Failed to send follow-up: {chat_err}")
