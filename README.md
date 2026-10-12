# MLH-FIXOOR: Error Screenshot Explainer 🛠️

A beginner-friendly Streamlit web application that uses Google's multimodal **Gemma 4** model via the **Gemini API** (`google-genai` library) to analyze screenshots of programming errors, exceptions, or terminal stack traces and provide clear explanations and step-by-step fixes.

---

## Features

- 📸 **Screenshot Upload**: Upload error images in PNG or JPG format.
- 🤖 **Gemma 4 Powered**: Uses Google's `gemma-4-26b-a4b-it` (or `gemma-4-31b-it`) via the `google-genai` SDK.
- 👶 **Beginner-Friendly Output**:
  1. **What this error means** (plain English with intuitive analogies).
  2. **Likely causes** (common mistakes like typos, missing modules, wrong directories).
  3. **Numbered steps to fix it** (actionable terminal commands and code changes).
- 🔑 **Flexible API Key Configuration**: Reads directly from `.env`, the `GEMINI_API_KEY` environment variable, or prompts securely via a sidebar password box.
- 🛡️ **Robust Error Handling**: Clean, user-friendly messages for missing files, invalid keys, or API issues.

---

## Quickstart

### 1. Clone or Open the Project
Open a terminal in the project directory:
```bash
git clone https://github.com/vskvasista-lab/MLH-FIXOOR.git
cd MLH-FIXOOR
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# On Windows (PowerShell):
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On macOS/Linux:
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Your Gemini API Key (Optional)
You can set your API key in a `.env` file, as an environment variable, or enter it directly in the app's sidebar:

```bash
# Copy template
cp .env.example .env

# Or set via environment variable:
# Windows (PowerShell):
$env:GEMINI_API_KEY="your_api_key_here"

# Windows (Command Prompt):
set GEMINI_API_KEY=your_api_key_here

# macOS/Linux:
export GEMINI_API_KEY="your_api_key_here"
```
> *Get a free API key from [Google AI Studio](https://aistudio.google.com/).*

### 5. Run the Application
```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Project Structure

```text
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies (streamlit, google-genai, pillow)
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules (protects API keys and venv)
└── README.md           # Project documentation and setup guide
```

---

## Model Configuration

In `app.py`, the default model is set to:
```python
DEFAULT_MODEL_NAME = "gemma-4-26b-a4b-it"  # Supported Gemma 4 multimodal model on Gemini API
```
You can also change the model name directly in the application's sidebar at runtime (e.g., `gemma-4-31b-it`).
