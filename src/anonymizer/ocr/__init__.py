"""
OCR Module for Document Anonymization
=====================================

Provides optical character recognition capabilities for detecting text regions
in document images. Supports multiple OCR engines with fallback strategies.
"""

from .processor import OCRProcessor
from .engines import TrOCREngine, PaddleOCREngine, EasyOCREngine
from .models import DetectedText, OCRResult, OCRConfig

__all__ = [
    "OCRProcessor",
    "TrOCREngine", 
    "PaddleOCREngine",
    "EasyOCREngine",
    "DetectedText",
    "OCRResult",
    "OCRConfig",
]