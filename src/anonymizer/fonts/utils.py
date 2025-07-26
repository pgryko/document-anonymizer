"""
Font Utilities
==============

Utility functions for font detection, analysis, and manipulation.
"""

import logging
import os
import re

logger = logging.getLogger(__name__)


def get_font_info(font_path: str) -> dict[str, any] | None:
    """
    Extract font information from font file.

    Args:
        font_path: Path to font file

    Returns:
        Dictionary with font information or None if failed
    """
    try:
        # Try different font libraries in order of preference
        return (
            _get_font_info_fonttools(font_path)
            or _get_font_info_freetype(font_path)
            or _get_font_info_fallback(font_path)
        )
    except Exception as e:
        logger.debug(f"Failed to get font info for {font_path}: {e}")
        return None


def _get_font_info_fonttools(font_path: str) -> dict[str, any] | None:
    """Get font info using fonttools library."""
    try:
        from fontTools.ttLib import TTFont
        from fontTools.ttLib.tables._n_a_m_e import table__n_a_m_e

        font = TTFont(font_path)
        name_table = font["name"]

        # Extract font names
        font_family = _get_font_name(name_table, 1) or "Unknown"  # Family name
        font_name = _get_font_name(name_table, 4) or font_family  # Full name
        subfamily = _get_font_name(name_table, 2) or "Regular"  # Subfamily

        # Determine style and weight
        style, weight = _parse_style_weight(subfamily)

        return {
            "name": font_name,
            "family": font_family,
            "style": style,
            "weight": weight,
            "subfamily": subfamily,
        }

    except ImportError:
        logger.debug("fonttools not available")
        return None
    except Exception as e:
        logger.debug(f"fonttools failed for {font_path}: {e}")
        return None


def _get_font_name(name_table, name_id: int) -> str | None:
    """Extract font name from name table."""
    try:
        # Try to get English name first
        for record in name_table.names:
            if record.nameID == name_id:
                # Prefer English (language ID 1033 for US English)
                if record.langID == 1033 or record.langID == 0:
                    return str(record)

        # Fallback to any available name
        for record in name_table.names:
            if record.nameID == name_id:
                return str(record)

    except Exception:
        pass

    return None


def _get_font_info_freetype(font_path: str) -> dict[str, any] | None:
    """Get font info using freetype library."""
    try:
        import freetype

        face = freetype.Face(font_path)

        font_family = face.family_name.decode("utf-8") if face.family_name else "Unknown"
        style_name = face.style_name.decode("utf-8") if face.style_name else "Regular"

        # Parse style and weight
        style, weight = _parse_style_weight(style_name)

        return {
            "name": f"{font_family} {style_name}",
            "family": font_family,
            "style": style,
            "weight": weight,
            "subfamily": style_name,
        }

    except ImportError:
        logger.debug("freetype-py not available")
        return None
    except Exception as e:
        logger.debug(f"freetype failed for {font_path}: {e}")
        return None


def _get_font_info_fallback(font_path: str) -> dict[str, any] | None:
    """Fallback font info extraction from filename."""
    try:
        filename = os.path.basename(font_path)
        name_without_ext = os.path.splitext(filename)[0]

        # Try to parse family and style from filename
        # Common patterns: FontFamily-Style.ttf, FontFamily_Style.ttf, FontFamilyStyle.ttf

        # Split on common delimiters
        parts = re.split(r"[-_\s]+", name_without_ext)

        if len(parts) >= 2:
            family = parts[0]
            style_part = "-".join(parts[1:])
        else:
            # Try to extract style from end of name
            family, style_part = _extract_style_from_name(name_without_ext)

        style, weight = _parse_style_weight(style_part)

        return {
            "name": name_without_ext,
            "family": family,
            "style": style,
            "weight": weight,
            "subfamily": style_part,
        }

    except Exception as e:
        logger.debug(f"Fallback font info failed for {font_path}: {e}")
        return None


def _extract_style_from_name(name: str) -> tuple[str, str]:
    """Extract family and style from font name."""
    # Common style suffixes
    style_patterns = [
        r"(.*?)(Bold|Italic|Regular|Light|Medium|Heavy|Black|Thin|ExtraLight|SemiBold|ExtraBold)$",
        r"(.*?)(Bd|It|Rg|Lt|Md|Hv|Bl|Th|XLt|SBd|XBd)$",
        r"(.*?)([BI]+)$",  # B for Bold, I for Italic
    ]

    for pattern in style_patterns:
        match = re.match(pattern, name, re.IGNORECASE)
        if match:
            family = match.group(1).strip()
            style = match.group(2)
            return family, style

    # No style found, entire name is family
    return name, "Regular"


def _parse_style_weight(style_name: str) -> tuple[str, int]:
    """
    Parse style name into style and weight.

    Returns:
        Tuple of (style, weight) where style is one of:
        normal, italic, bold, bold-italic
        And weight is a number from 100-900
    """
    style_name_lower = style_name.lower()

    # Determine if italic
    is_italic = any(
        keyword in style_name_lower for keyword in ["italic", "oblique", "slanted", "it", "obl"]
    )

    # Determine weight
    weight = 400  # Default normal weight

    weight_keywords = {
        "thin": 100,
        "extralight": 200,
        "ultralight": 200,
        "light": 300,
        "normal": 400,
        "regular": 400,
        "medium": 500,
        "semibold": 600,
        "demibold": 600,
        "bold": 700,
        "extrabold": 800,
        "ultrabold": 800,
        "black": 900,
        "heavy": 900,
    }

    for keyword, weight_value in weight_keywords.items():
        if keyword in style_name_lower:
            weight = weight_value
            break

    # Determine style
    is_bold = weight >= 700

    if is_bold and is_italic:
        style = "bold-italic"
    elif is_bold:
        style = "bold"
    elif is_italic:
        style = "italic"
    else:
        style = "normal"

    return style, weight


def detect_font(image_region: any, ocr_result: any = None) -> dict[str, any] | None:
    """
    Detect font characteristics from image region.

    Args:
        image_region: Image region containing text
        ocr_result: Optional OCR result with text information

    Returns:
        Dictionary with detected font characteristics
    """
    try:
        # This is a placeholder for font detection from images
        # Real implementation would analyze:
        # - Character shapes and spacing
        # - Serif vs sans-serif characteristics
        # - Font weight and style
        # - Font size

        font_info = {
            "family": "Arial",  # Detected family
            "style": "normal",  # Detected style
            "weight": 400,  # Detected weight
            "size": 12,  # Detected size in points
            "confidence": 0.7,  # Detection confidence
        }

        return font_info

    except Exception as e:
        logger.debug(f"Font detection failed: {e}")
        return None


def find_similar_font(target_font: str, available_fonts: list[str]) -> str | None:
    """
    Find the most similar font from available fonts.

    Args:
        target_font: Target font name
        available_fonts: List of available font names

    Returns:
        Most similar font name or None
    """
    try:
        target_lower = target_font.lower()

        # Exact match
        for font in available_fonts:
            if font.lower() == target_lower:
                return font

        # Family name match
        target_family = target_font.split()[0].lower()
        for font in available_fonts:
            if font.lower().startswith(target_family):
                return font

        # Similarity scoring
        similarities = []
        for font in available_fonts:
            similarity = _calculate_string_similarity(target_lower, font.lower())
            similarities.append((similarity, font))

        if similarities:
            similarities.sort(key=lambda x: x[0], reverse=True)
            return similarities[0][1]

        return None

    except Exception as e:
        logger.debug(f"Similar font search failed: {e}")
        return None


def _calculate_string_similarity(s1: str, s2: str) -> float:
    """Calculate similarity between two strings."""
    try:
        from difflib import SequenceMatcher

        return SequenceMatcher(None, s1, s2).ratio()
    except Exception:
        # Fallback to simple character overlap
        set1 = set(s1)
        set2 = set(s2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0


def calculate_font_similarity(font1: "FontMetadata", font2: "FontMetadata") -> float:
    """
    Calculate similarity between two fonts.

    Args:
        font1: First font metadata
        font2: Second font metadata

    Returns:
        Similarity score between 0 and 1
    """
    try:
        scores = []

        # Family name similarity
        family_sim = _calculate_string_similarity(font1.family.lower(), font2.family.lower())
        scores.append(family_sim * 0.4)  # 40% weight

        # Style similarity
        style_sim = 1.0 if font1.style == font2.style else 0.0
        scores.append(style_sim * 0.3)  # 30% weight

        # Weight similarity
        weight_diff = abs(font1.weight - font2.weight)
        weight_sim = max(0, 1 - weight_diff / 400)  # Normalize by max weight difference
        scores.append(weight_sim * 0.3)  # 30% weight

        return sum(scores)

    except Exception as e:
        logger.debug(f"Font similarity calculation failed: {e}")
        return 0.0


def get_font_metrics(font_path: str, size: int = 12) -> dict[str, any] | None:
    """
    Get font metrics for text rendering.

    Args:
        font_path: Path to font file
        size: Font size in points

    Returns:
        Dictionary with font metrics
    """
    try:
        # Try different approaches to get font metrics
        return (
            _get_font_metrics_pil(font_path, size)
            or _get_font_metrics_freetype(font_path, size)
            or _get_font_metrics_fallback(font_path, size)
        )

    except Exception as e:
        logger.debug(f"Failed to get font metrics for {font_path}: {e}")
        return None


def _get_font_metrics_pil(font_path: str, size: int) -> dict[str, any] | None:
    """Get font metrics using PIL."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(font_path, size)

        # Create temporary image to measure text
        img = Image.new("RGB", (100, 100), color="white")
        draw = ImageDraw.Draw(img)

        # Measure sample text
        sample_text = "Abcdefg"
        bbox = draw.textbbox((0, 0), sample_text, font=font)

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        return {
            "ascent": height * 0.8,  # Approximate
            "descent": height * 0.2,  # Approximate
            "height": height,
            "max_width": width / len(sample_text),
            "size": size,
        }

    except ImportError:
        logger.debug("PIL not available for font metrics")
        return None
    except Exception as e:
        logger.debug(f"PIL font metrics failed: {e}")
        return None


def _get_font_metrics_freetype(font_path: str, size: int) -> dict[str, any] | None:
    """Get font metrics using freetype."""
    try:
        import freetype

        face = freetype.Face(font_path)
        face.set_char_size(size * 64)  # Size in 1/64th points

        metrics = face.size.metrics

        return {
            "ascent": metrics.ascender / 64,
            "descent": abs(metrics.descender) / 64,
            "height": metrics.height / 64,
            "max_width": metrics.max_advance / 64,
            "size": size,
        }

    except ImportError:
        logger.debug("freetype-py not available for font metrics")
        return None
    except Exception as e:
        logger.debug(f"freetype font metrics failed: {e}")
        return None


def _get_font_metrics_fallback(font_path: str, size: int) -> dict[str, any] | None:
    """Fallback font metrics based on font size."""
    try:
        # Approximate metrics based on common font characteristics
        return {
            "ascent": size * 0.8,
            "descent": size * 0.2,
            "height": size,
            "max_width": size * 0.6,  # Approximate character width
            "size": size,
        }

    except Exception as e:
        logger.debug(f"Fallback font metrics failed: {e}")
        return None


def validate_font_file(font_path: str) -> bool:
    """
    Validate that a file is a valid font file.

    Args:
        font_path: Path to font file

    Returns:
        True if valid font file, False otherwise
    """
    try:
        # Check file extension
        ext = os.path.splitext(font_path)[1].lower()
        valid_extensions = {".ttf", ".otf", ".woff", ".woff2", ".ttc", ".otc"}

        if ext not in valid_extensions:
            return False

        # Check if we can read font info
        font_info = get_font_info(font_path)
        return font_info is not None

    except Exception:
        return False


def create_font_sample(font_path: str, text: str = "Sample Text", size: int = 24) -> any | None:
    """
    Create a sample image showing the font.

    Args:
        font_path: Path to font file
        text: Sample text to render
        size: Font size

    Returns:
        PIL Image with font sample or None if failed
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        font = ImageFont.truetype(font_path, size)

        # Calculate text size
        temp_img = Image.new("RGB", (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)

        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Create image with padding
        padding = 20
        img_width = text_width + 2 * padding
        img_height = text_height + 2 * padding

        img = Image.new("RGB", (img_width, img_height), color="white")
        draw = ImageDraw.Draw(img)

        # Draw text
        draw.text((padding, padding), text, font=font, fill="black")

        return img

    except Exception as e:
        logger.debug(f"Font sample creation failed: {e}")
        return None
