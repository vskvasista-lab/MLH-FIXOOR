"""imaging.py: Pillow-based image preprocessing for Error Screenshot Explainer.

Design Notes:
- Decoupled from Streamlit and GenAI so image manipulation can be tested independently.
- Ensures all inputs are standardized to RGB to avoid mode mismatches (e.g., RGBA transparency,
  palette-indexed images, or CMYK).
- Auto-upscaling ensures tiny terminal text remains legible when tokenized by vision models.
- Luminance detection identifies dark themes so contrast can be enhanced to make faint text pop.
"""

from typing import Any
from PIL import Image, ImageEnhance, ImageOps, ImageStat


def preprocess_image(
    image: Image.Image,
    enhance: bool = True,
    invert_dark: bool = False,
) -> tuple[Image.Image, dict[str, Any]]:
    """Preprocesses a screenshot for multimodal vision models.

    Args:
        image: Original PIL Image.
        enhance: Whether to apply upscaling and contrast optimization.
        invert_dark: If True and the image is dark-themed, invert the colors.

    Returns:
        tuple of (processed PIL Image, dict of preprocessing diagnostics).
    """
    # Design Note: Always convert to RGB first. Multimodal APIs expect standard 3-channel RGB;
    # RGBA transparency can cause black background artifacts, and CMYK will fail serialization.
    img = image.convert("RGB")
    original_size = image.size

    info: dict[str, Any] = {
        "original_size": original_size,
        "new_size": original_size,
        "upscaled": False,
        "is_dark": False,
        "avg_luminance": 0.0,
        "inverted": False,
        "enhanced": enhance,
    }

    if not enhance:
        return img, info

    w, h = img.size

    # Design Note: Vision models divide inputs into fixed-size vision patches. Tiny screenshots
    # (e.g., small error popups under 900x450) suffer heavy fidelity loss. Upscaling with
    # Lanczos interpolation reconstructs clear character boundaries for small monospace fonts.
    min_width, min_height = 900, 450
    if w < min_width or h < min_height:
        scale = max(min_width / max(w, 1), min_height / max(h, 1), 1.5)
        scale = min(scale, 3.0)  # Bound maximum scaling factor to prevent excessive memory usage
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
        info["upscaled"] = True
        info["new_size"] = new_size

    # Design Note: Calculate perceived luminance using ITU-R BT.601 standard weights.
    # Luminance < 95 indicates a dark IDE/terminal theme.
    grayscale = img.convert("L")
    stat = ImageStat.Stat(grayscale)
    avg_luminance = float(stat.mean[0])
    info["avg_luminance"] = round(avg_luminance, 1)

    is_dark = avg_luminance < 95.0
    info["is_dark"] = is_dark

    # Design Note: Dark themes with dim colored text often suffer from low contrast.
    # Autocontrast expands the dynamic range, while contrast/sharpness boosters
    # make syntax-highlighted code crisp against the dark background.
    if is_dark:
        if invert_dark:
            img = ImageOps.invert(img)
            info["inverted"] = True
            # Boost contrast of the inverted light version
            img = ImageEnhance.Contrast(img).enhance(1.25)
        else:
            # Cut off 1% of extreme pixels to normalize background noise
            img = ImageOps.autocontrast(img, cutoff=1)
            img = ImageEnhance.Contrast(img).enhance(1.4)
            img = ImageEnhance.Sharpness(img).enhance(1.3)
    else:
        # Subtle boost for light-themed terminals/browsers
        img = ImageEnhance.Contrast(img).enhance(1.15)
        img = ImageEnhance.Sharpness(img).enhance(1.2)

    return img, info
