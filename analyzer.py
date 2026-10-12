"""analyzer.py: Gemma 4 API calling, prompt construction, and JSON extraction.

Design Notes:
- Isolates all Google GenAI SDK interactions away from the Streamlit UI.
- Strict prompt formatting instructs the model to return valid JSON with schema adherence.
- Two-tier JSON parser strips markdown fences, attempts substring extraction, retries once on
  JSONDecodeError, and falls back to raw text so the app NEVER crashes.
- Enforces language localization for explanations while preserving English for commands and code.
- Pass [image_part, prompt] with the image preceding the prompt for stable multimodal tokenizer caching.
"""

import io
import json
import re
from typing import Any
from PIL import Image
from google import genai
from google.genai import errors, types

DEFAULT_MODEL_NAME = "gemma-4-26b-a4b-it"


def parse_json_response(raw_text: str) -> dict[str, Any] | None:
    """Extracts and parses a JSON object from model output text.

    Design Note: LLMs often wrap JSON in ```json markdown code fences or include
    introductory/trailing text. We clean fences first, attempt standard parsing,
    and then fall back to locating outermost curly brackets.
    """
    if not raw_text:
        return None

    cleaned = raw_text.strip()

    # Step 1: Strip markdown code fences if present
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
        if match:
            cleaned = match.group(1).strip()

    # Step 2: Try direct parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 3: Substring extraction between the first '{' and last '}'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace : last_brace + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    return None


def build_analysis_prompt(
    explanation_level: str = "Beginner",
    language: str = "English",
) -> str:
    """Builds the system prompt instructing Gemma 4 to analyze the error."""
    # Design Note: Enforcing strict English for code while allowing multi-language explanations
    # prevents disastrous syntax bugs (e.g. translating 'pip install' to Hindi or Spanish).
    return f"""You are an expert senior software engineer and patient programming mentor.
Examine this error screenshot carefully.

Target Audience Level: {explanation_level} developer.
Output Language: {language}.

Instructions:
1. Explain the error accurately and thoroughly for a {explanation_level} developer.
2. All explanations, summaries, analogies, fix steps, and prevention tips MUST be written in {language}.
3. CRITICAL: All code snippets, terminal commands, function names, error types, and variable names MUST remain in English.
4. You MUST return your entire response as a valid JSON code block enclosed in ```json ... ``` with this exact structure:

```json
{{
  "error_type": "Exact name of the error or exception (e.g., ModuleNotFoundError, TypeError)",
  "severity": "low" or "medium" or "high",
  "summary": "1-2 sentence overview of what went wrong",
  "root_cause": "Detailed technical or logical explanation of why this error occurred",
  "analogy": "An intuitive real-world analogy to help understand the concept",
  "fix_steps": [
    {{
      "title": "Short title for step",
      "explanation": "Detailed explanation of what to do and why",
      "command_or_code": "Exact command to run or line of code to change (or empty string if none)"
    }}
  ],
  "corrected_code": "Full corrected code snippet if applicable, otherwise empty string",
  "prevention_tip": "Advice on how to avoid this error in the future"
}}
```
"""


def analyze_screenshot(
    image: Image.Image,
    api_key: str,
    model_name: str = DEFAULT_MODEL_NAME,
    explanation_level: str = "Beginner",
    language: str = "English",
) -> tuple[dict[str, Any] | None, str, str | None]:
    """Sends the processed screenshot to Gemma 4 and returns parsed analysis.

    Returns:
        tuple of (parsed_json_dict_or_none, raw_model_text, warning_message_or_none)
    """
    client = genai.Client(api_key=api_key)

    # Design Note: Convert PIL image to PNG bytes and pass via types.Part.from_bytes.
    # Multimodal image part is passed FIRST in contents to align with Gemma's vision encoder prefix.
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    prompt = build_analysis_prompt(
        explanation_level=explanation_level,
        language=language,
    )

    warning_msg = None

    # Design Note: Handle transient 500 errors on specific Gemma variants by falling back
    # to the stable gemma-4-26b-a4b-it variant automatically.
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[image_part, prompt],
        )
    except errors.ServerError as serv_err:
        if model_name != "gemma-4-26b-a4b-it":
            warning_msg = f"Model '{model_name}' hit server error ({serv_err.code}). Retrying with 'gemma-4-26b-a4b-it'..."
            response = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=[image_part, prompt],
            )
        else:
            raise

    raw_text = response.text if response and response.text else ""
    if not raw_text:
        return None, "", "The model returned an empty response."

    parsed = parse_json_response(raw_text)

    # Design Note: If initial parsing fails, perform ONE automated JSON repair retry
    # with the model before falling back to raw text.
    if not parsed:
        try:
            repair_prompt = (
                "Convert the following analysis into valid JSON matching the requested schema. "
                f"Return ONLY the JSON block:\n\n{raw_text}"
            )
            repair_resp = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=[repair_prompt],
            )
            if repair_resp and repair_resp.text:
                parsed = parse_json_response(repair_resp.text)
        except Exception:
            parsed = None

    return parsed, raw_text, warning_msg
