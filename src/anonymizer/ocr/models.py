"""
OCR Data Models
===============

Data structures for OCR processing and results.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from ..core.models import BoundingBox


class OCREngine(Enum):
    """Supported OCR engines."""

    TROCR = "trocr"
    PADDLEOCR = "paddleocr"
    EASYOCR = "easyocr"
    TESSERACT = "tesseract"


@dataclass
class DetectedText:
    """Single text detection result with bounding box and metadata."""

    text: str
    bbox: BoundingBox
    confidence: float
    language: str | None = None
    orientation: float | None = None  # Text rotation angle in degrees
    font_size: float | None = None  # Estimated font size
    is_handwritten: bool | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Validate detected text after initialization."""
        if not self.text.strip():
            raise ValueError("Text cannot be empty")

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")

        if self.orientation is not None and not (-180 <= self.orientation <= 180):
            raise ValueError(
                f"Orientation must be between -180 and 180 degrees, got {self.orientation}"
            )


@dataclass
class OCRResult:
    """Complete OCR processing result for a document image."""

    detected_texts: list[DetectedText]
    processing_time_ms: float
    engine_used: OCREngine
    image_size: tuple[int, int]  # (width, height)
    success: bool = True
    errors: list[str] = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self):
        """Initialize errors list if None."""
        if self.errors is None:
            self.errors = []

    @property
    def total_text_regions(self) -> int:
        """Number of text regions detected."""
        return len(self.detected_texts)

    @property
    def average_confidence(self) -> float:
        """Average confidence across all detections."""
        if not self.detected_texts:
            return 0.0
        return sum(text.confidence for text in self.detected_texts) / len(self.detected_texts)

    def high_confidence_texts(self, threshold: float = 0.7) -> list[DetectedText]:
        """Filter texts with confidence above threshold."""
        return [text for text in self.detected_texts if text.confidence >= threshold]

    def filter_by_language(self, language: str) -> list[DetectedText]:
        """Filter texts by language."""
        return [
            text
            for text in self.detected_texts
            if text.language and text.language.lower() == language.lower()
        ]

    def get_text_by_confidence_range(self, min_conf: float, max_conf: float) -> list[DetectedText]:
        """Get texts within confidence range."""
        return [text for text in self.detected_texts if min_conf <= text.confidence <= max_conf]


@dataclass
class OCRConfig:
    """Configuration for OCR processing."""

    # Engine selection
    primary_engine: OCREngine = OCREngine.PADDLEOCR
    fallback_engines: list[OCREngine] = None

    # Detection parameters
    min_confidence_threshold: float = 0.5
    max_text_length: int = 1000
    min_text_length: int = 1

    # Image preprocessing
    enable_preprocessing: bool = True
    resize_factor: float = 1.0
    contrast_enhancement: bool = True
    noise_reduction: bool = True

    # Language settings
    languages: list[str] = None  # e.g., ['en', 'es', 'fr']
    detect_language: bool = True

    # Performance settings
    batch_size: int = 1
    use_gpu: bool = True
    max_workers: int = 4
    timeout_seconds: int = 30

    # Text filtering
    filter_short_texts: bool = True
    filter_low_confidence: bool = True
    merge_nearby_texts: bool = False
    merge_distance_threshold: float = 10.0

    # Advanced options
    detect_orientation: bool = True
    estimate_font_size: bool = True
    detect_handwriting: bool = False
    preserve_text_layout: bool = True

    def __post_init__(self):
        """Initialize default values and validate configuration."""
        if self.fallback_engines is None:
            # Default fallback order
            self.fallback_engines = [
                OCREngine.EASYOCR,
                OCREngine.TROCR,
                OCREngine.TESSERACT,
            ]
            # Remove primary engine from fallbacks
            if self.primary_engine in self.fallback_engines:
                self.fallback_engines.remove(self.primary_engine)

        if self.languages is None:
            self.languages = ["en"]  # Default to English

        # Validate parameters
        if not (0.0 <= self.min_confidence_threshold <= 1.0):
            raise ValueError("min_confidence_threshold must be between 0 and 1")

        if self.min_text_length < 0:
            raise ValueError("min_text_length cannot be negative")

        if self.max_text_length < self.min_text_length:
            raise ValueError("max_text_length must be >= min_text_length")

        if self.resize_factor <= 0:
            raise ValueError("resize_factor must be positive")

        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass
class OCRMetrics:
    """Performance metrics for OCR processing."""

    total_processing_time_ms: float
    text_detection_time_ms: float
    text_recognition_time_ms: float
    preprocessing_time_ms: float

    total_texts_detected: int
    high_confidence_detections: int
    low_confidence_detections: int

    engine_used: OCREngine
    fallback_attempted: bool = False
    fallback_engines_tried: list[OCREngine] = None

    memory_usage_mb: float | None = None
    gpu_usage_percent: float | None = None

    def __post_init__(self):
        """Initialize fallback engines list if None."""
        if self.fallback_engines_tried is None:
            self.fallback_engines_tried = []

    @property
    def detection_rate(self) -> float:
        """Rate of successful text detections."""
        if self.total_texts_detected == 0:
            return 0.0
        return self.high_confidence_detections / self.total_texts_detected

    @property
    def average_processing_speed(self) -> float:
        """Average processing speed in texts per second."""
        if self.total_processing_time_ms == 0:
            return 0.0
        return self.total_texts_detected / (self.total_processing_time_ms / 1000)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary for logging/serialization."""
        return {
            "total_processing_time_ms": self.total_processing_time_ms,
            "text_detection_time_ms": self.text_detection_time_ms,
            "text_recognition_time_ms": self.text_recognition_time_ms,
            "preprocessing_time_ms": self.preprocessing_time_ms,
            "total_texts_detected": self.total_texts_detected,
            "high_confidence_detections": self.high_confidence_detections,
            "low_confidence_detections": self.low_confidence_detections,
            "detection_rate": self.detection_rate,
            "average_processing_speed": self.average_processing_speed,
            "engine_used": self.engine_used.value,
            "fallback_attempted": self.fallback_attempted,
            "fallback_engines_tried": [engine.value for engine in self.fallback_engines_tried],
            "memory_usage_mb": self.memory_usage_mb,
            "gpu_usage_percent": self.gpu_usage_percent,
        }
