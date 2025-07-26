"""
Font Management Module
======================

This module provides font management capabilities for consistent text rendering
across different systems during the anonymization process.
"""

from .manager import FontManager, FontMetadata
from .bundled import BundledFontProvider
from .system import SystemFontProvider
from .utils import detect_font, find_similar_font, get_font_metrics

__all__ = [
    "FontManager",
    "FontMetadata",
    "BundledFontProvider",
    "SystemFontProvider",
    "detect_font",
    "find_similar_font",
    "get_font_metrics",
]
