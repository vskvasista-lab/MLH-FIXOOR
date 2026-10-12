"""fetcher.py: Secure HTTP error page fetching and text extraction for Error Screenshot Explainer.

Design Notes:
- Uses `requests` with a realistic User-Agent so WAFs/proxies don't block requests with artificial 403s.
- Enforces a strict 10-second timeout to prevent thread hangs on slow or deadlocked servers.
- Cleans and decomposes noisy HTML tags (scripts, stylesheets, SVGs, iframes) using BeautifulSoup.
- Truncates extracted text cleanly to 4,000 characters to keep prompt tokens high-density and focused.
- Captures status code and headers (e.g., Server, Content-Type, CORS headers) for root-cause analysis.
- Raises user-friendly `FetchError` exceptions for timeouts, connection errors, and invalid URLs.
"""

from typing import Any
import requests
from bs4 import BeautifulSoup


class FetchError(Exception):
    """Custom exception carrying a user-friendly error message for web fetch failures."""

    pass


def fetch_error_from_url(
    url: str,
    is_local: bool = False,
    timeout: int = 10,
) -> dict[str, Any]:
    """Fetches an error page or API endpoint and extracts visible diagnostic text.

    Args:
        url: Target web or API endpoint (e.g., https://example.com/api or http://localhost:3000/err).
        is_local: Whether the target is a local development server.
        timeout: Maximum time in seconds to wait for a response.

    Returns:
        dict containing:
            - url: clean validated target URL
            - status_code: HTTP status code integer
            - status_text: HTTP reason phrase (e.g., "Internal Server Error")
            - headers: dictionary of response headers
            - extracted_text: stripped, human-readable error text (up to 4,000 chars)
            - is_error_status: boolean (True if status >= 400)
            - content_type: response Content-Type header

    Raises:
        FetchError: If the request fails, times out, has an invalid URL, or returns unreachable status.
    """
    clean_url = url.strip()

    # Design Note: Ensure URL has a valid scheme. Users often paste 'localhost:3000/api'
    # without 'http://'. Auto-prefixing 'http://' for local development improves user experience.
    if not clean_url.startswith(("http://", "https://")):
        if is_local or clean_url.startswith(("localhost", "127.0.0.1")):
            clean_url = f"http://{clean_url}"
        else:
            clean_url = f"https://{clean_url}"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 "
            "ErrorScreenshotExplainer/2.0"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml,application/json;q=0.9,*/*;q=0.8",
    }

    try:
        # Design Note: allow_redirects=True allows discovering redirect loops or final landing errors
        response = requests.get(
            clean_url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
        )
    except requests.exceptions.Timeout:
        raise FetchError(
            f"⏱️ Request timed out after {timeout} seconds while connecting to `{clean_url}`. "
            "The server may be overloaded, down, or deadlocked."
        )
    except requests.exceptions.ConnectionError:
        dev_hint = (
            " If this is a local development server, ensure your service is running (e.g., npm run dev) "
            "and listening on the specified port."
            if is_local
            else ""
        )
        raise FetchError(
            f"🔌 Connection failed for `{clean_url}`. Could not establish a connection to the host.{dev_hint}"
        )
    except requests.exceptions.InvalidURL:
        raise FetchError(f"⚠️ Invalid URL format: `{clean_url}`. Please enter a valid URL.")
    except requests.exceptions.RequestException as req_err:
        raise FetchError(f"⚠️ Network error while fetching `{clean_url}`: {req_err}")

    # Design Note: Extract text cleanly using BeautifulSoup.
    # Strip script, style, SVG, noscript, and audio/video tags to remove noise.
    content_type = response.headers.get("Content-Type", "").lower()
    raw_content = response.text or ""

    if "json" in content_type:
        # If the endpoint returned JSON (e.g., API error payload), keep formatted JSON
        clean_text = raw_content.strip()
    else:
        try:
            soup = BeautifulSoup(raw_content, "html.parser")
            for noisy_tag in soup(["script", "style", "svg", "noscript", "canvas", "iframe"]):
                noisy_tag.decompose()
            text_lines = soup.get_text(separator="\n").splitlines()
            clean_text = "\n".join(line.strip() for line in text_lines if line.strip())
        except Exception:
            clean_text = raw_content.strip()

    # If the body is completely empty, provide an informative placeholder
    if not clean_text:
        clean_text = f"[HTTP {response.status_code} {response.reason}] No visible body text returned by the server."

    # Design Note: Truncate to first 4,000 characters to keep prompt tokens dense and focused
    truncated_text = clean_text[:4000]
    if len(clean_text) > 4000:
        truncated_text += "\n\n... [Content truncated to first 4,000 characters] ..."

    # Extract relevant debug headers
    debug_headers = {
        k: v
        for k, v in response.headers.items()
        if k.lower() in [
            "server",
            "content-type",
            "access-control-allow-origin",
            "x-powered-by",
            "date",
            "location",
        ]
    }

    return {
        "url": clean_url,
        "status_code": response.status_code,
        "status_text": response.reason,
        "headers": debug_headers,
        "extracted_text": truncated_text,
        "raw_preview": clean_text[:600],
        "is_error_status": response.status_code >= 400,
        "content_type": content_type,
    }
