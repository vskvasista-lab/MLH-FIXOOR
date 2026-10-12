import os
import streamlit as st
from PIL import Image
from google import genai
from google.genai import errors, types

# ==============================================================================
# Model Configuration
# Model name: <PASTE MODEL NAME HERE>
# Supported Gemma 4 multimodal models on Gemini API: "gemma-4-26b-a4b-it", "gemma-4-31b-it"
# ==============================================================================
DEFAULT_MODEL_NAME = "gemma-4-26b-a4b-it"

# Page configuration
st.set_page_config(
    page_title="Error Screenshot Explainer",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ Error Screenshot Explainer")
st.markdown(
    "Upload a screenshot of your terminal error, code exception, or stack trace. "
    "We will send it to **Gemma 4** to give you beginner-friendly explanations and step-by-step fixes."
)

# ------------------------------------------------------------------------------
# Sidebar: Configuration & API Key
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Configuration")

# API Key handling: Load from .env file or environment, fallback to text box
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
# Ignore placeholder values
if env_api_key in ["your_api_key_here", "your_gemini_api_key", "your_api_key", "<PASTE API KEY HERE>"]:
    env_api_key = ""

if env_api_key:
    api_key = env_api_key
    st.sidebar.success("✅ `GEMINI_API_KEY` loaded securely.")
else:
    api_key = st.sidebar.text_input(
        "Gemini API Key:",
        type="password",
        placeholder="Enter your API key here...",
        help="Get a free Gemini API key from https://aistudio.google.com/",
    ).strip()
    if not api_key:
        st.sidebar.info("💡 Set `GEMINI_API_KEY` in your environment, `.env`, or enter it above.")

# Model selection (defaults to Gemma 4, allows custom paste if desired)
model_name = st.sidebar.text_input(
    "Gemma 4 Model Name:",
    value=DEFAULT_MODEL_NAME,
    help="Model identifier to use with the Gemini API. Default is gemma-4-31b-it.",
).strip()

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### 📖 How to use
    1. Upload a PNG or JPG error screenshot.
    2. Click **Explain this error**.
    3. Read the explanation and follow the numbered steps to fix it.
    """
)

# ------------------------------------------------------------------------------
# State Initialization
# ------------------------------------------------------------------------------
if "explanation" not in st.session_state:
    st.session_state["explanation"] = None
if "last_file_name" not in st.session_state:
    st.session_state["last_file_name"] = None

# ------------------------------------------------------------------------------
# Main Content: File Upload & Action
# ------------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload your error screenshot (PNG or JPG)",
    type=["png", "jpg", "jpeg"],
    help="Drag and drop or browse to upload an image of the error.",
)

# Reset explanation if a new file is uploaded
if uploaded_file and uploaded_file.name != st.session_state["last_file_name"]:
    st.session_state["explanation"] = None
    st.session_state["last_file_name"] = uploaded_file.name

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📷 Uploaded Screenshot")
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.image(image, caption=f"File: {uploaded_file.name}", use_container_width=True)
        except Exception as e:
            st.error(f"Could not open the uploaded image: {e}")
            image = None
    else:
        st.info("👆 Please upload an image above to get started.")
        image = None

with col2:
    st.subheader("💡 Error Explanation & Fix")
    explain_clicked = st.button("🚀 Explain this error", type="primary", use_container_width=True)

    if explain_clicked:
        if uploaded_file is None or image is None:
            st.warning("⚠️ Please upload an error screenshot before requesting an explanation.")
        elif not api_key:
            st.error(
                "⚠️ Missing API Key! Please enter your Gemini API Key in the sidebar "
                "or set the `GEMINI_API_KEY` environment variable."
            )
        else:
            with st.spinner(f"Analyzing error screenshot with {model_name}..."):
                try:
                    # Initialize the google-genai Client
                    client = genai.Client(api_key=api_key)

                    # Prompt tailored for beginner clarity
                    prompt = (
                        "You are an expert senior software engineer and patient programming mentor "
                        "assisting a beginner developer. Examine this error screenshot carefully and "
                        "explain it in clear, simple English:\n\n"
                        "1. What this error means: Explain what happened in plain English without unexplained jargon. "
                        "Use a simple real-world analogy if helpful.\n"
                        "2. Likely cause: Describe the 2 to 3 most common mistakes that trigger this issue "
                        "(e.g., typo in variable name, missing package, incorrect directory, unactivated virtual environment).\n"
                        "3. Numbered steps to fix it: Provide clear, sequential, numbered steps to solve the issue, "
                        "including exact terminal commands and code snippets where relevant.\n"
                    )

                    # Prepare image bytes with MIME type for multimodal Gemma 4
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.getvalue()
                    mime_type = uploaded_file.type or ("image/png" if uploaded_file.name.lower().endswith(".png") else "image/jpeg")
                    image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

                    # Query the Gemma 4 model with multimodal input
                    try:
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[prompt, image_part],
                        )
                    except errors.ServerError as server_err:
                        if model_name != "gemma-4-26b-a4b-it":
                            st.warning(f"Model '{model_name}' hit a temporary server error ({server_err.code}). Retrying with 'gemma-4-26b-a4b-it'...")
                            response = client.models.generate_content(
                                model="gemma-4-26b-a4b-it",
                                contents=[prompt, image_part],
                            )
                        else:
                            raise

                    if response and response.text:
                        st.session_state["explanation"] = response.text
                    else:
                        st.warning("The model did not return any text. Please try again.")

                except errors.APIError as e:
                    st.error(f"❌ Gemini API Error ({e.code}): {e.message}")
                except Exception as e:
                    st.error(
                        f"❌ An unexpected error occurred while communicating with the API:\n\n`{str(e)}`\n\n"
                        "Please verify your API key, model name, and internet connection."
                    )

    # Display saved explanation if available
    if st.session_state["explanation"]:
        st.markdown("### 📝 Analysis")
        st.markdown(st.session_state["explanation"])
    elif uploaded_file is not None and not explain_clicked:
        st.caption("Click **'Explain this error'** to generate the breakdown.")
