"""
OCR Module for Document Anonymization
=====================================

Provides optical character recognition capabilities for detecting text regions
in document images. Supports multiple OCR engines with fallback strategies.
"""

from .engines import EasyOCREngine, PaddleOCREngine, TrOCREngine
from .models import DetectedText, OCRConfig, OCRResult
from .processor import OCRProcessor

__all__ = [
    "DetectedText",
    "EasyOCREngine",
    "OCRConfig",
    "OCRProcessor",
    "OCRResult",
    "PaddleOCREngine",
    "TrOCREngine",
]
