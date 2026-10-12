# MLH-FIXOOR: Error Screenshot Explainer 🛠️

A comprehensive Streamlit web application that uses Google's multimodal **Gemma 4** model via the **Gemini API** (`google-genai` library) to analyze screenshots of programming errors, exceptions, and stack traces. It delivers structured diagnostic reports, copyable fixes, interactive follow-up chat, and multi-language support.

---

## 🌟 Upgraded Features

1. **Structured JSON Output**:
   - Parses model output into structured sections: `error_type`, `severity` (low/medium/high badge), `summary`, `root_cause`, `analogy`, `fix_steps`, `corrected_code`, and `prevention_tip`.
   - Safe JSON extraction with automated retry and plain-text fallback.
2. **One-Click Copy Buttons**:
   - Every terminal command, code fix, and corrected script is rendered in clean `st.code` blocks with built-in copy buttons.
3. **Pillow Image Preprocessing**:
   - Auto-upscales small screenshots for sharper OCR.
   - Detects dark-theme terminal screenshots and optimizes contrast/sharpness.
   - Sidebar toggle for "Enhance screenshot" and "Invert dark-theme screenshots" with an expandable preview.
4. **Explanation Depth Control**:
   - Sidebar selector for **Beginner**, **Intermediate**, or **Expert** developer personas.
5. **Multi-Language Support**:
   - Select between **English**, **Hindi**, **Telugu**, **Tamil**, and **Spanish**. Explanations and analogies are translated while code and commands stay in English.
6. **Interactive Follow-Up Chat**:
   - Ask follow-up questions directly below the diagnosis using `st.chat_input`. Gemma 4 retains the screenshot and conversation context.
7. **Optional Source Code Context**:
   - Expandable text area to paste the related snippet or function for higher-accuracy debugging.
8. **Session History**:
   - Automatically saves each analysis with timestamps and error types in the sidebar. Click any history item to instantly reload that result.
9. **Markdown Export**:
   - One-click download button (`.md`) to save the entire analysis report.
10. **Robust Reliability**:
    - File size validation (limits up to 5 MB with friendly warning).
    - Friendly handling of API rate limits, server errors, and empty responses. Never crashes.

---

## Quickstart

### 1. Clone or Open the Project
```bash
git clone https://github.com/vskvasista-lab/MLH-FIXOOR.git
cd MLH-FIXOOR
```

### 2. Set Up a Virtual Environment
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

### 4. Configure Your Gemini API Key
You can configure your key via `.env`, environment variables, or directly in the app's sidebar:

```bash
# Using .env:
cp .env.example .env
# Edit .env with your key: GEMINI_API_KEY=your_key_here
```
> *Get a free API key from [Google AI Studio](https://aistudio.google.com/).*

### 5. Run the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## Project Structure

```text
├── app.py              # Upgraded Streamlit application with all 10 features
├── requirements.txt    # Python dependencies (streamlit, google-genai, pillow)
├── .env.example        # Environment variable template
├── .gitignore          # Excludes .env, .venv, and caches
└── README.md           # Documentation and guide
```
