"""Font Management Module
======================

This module provides font management capabilities for consistent text rendering
across different systems during the anonymization process.
"""

from .bundled import BundledFontProvider
from .manager import FontManager
from .models import FontMetadata
from .system import SystemFontProvider
from .utils import detect_font, find_similar_font, get_font_metrics, validate_font_file

__all__ = [
    "BundledFontProvider",
    "FontManager",
    "FontMetadata",
    "SystemFontProvider",
    "detect_font",
    "find_similar_font",
    "get_font_metrics",
    "validate_font_file",
]
