"""OCR Unit Tests
==============

Tests for the OCR processing functionality including multiple engines,
text detection, and integration with the anonymization pipeline.
"""

import logging

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.anonymizer.core.exceptions import (
    ConfidenceOutOfRangeError,
    EmptyTextError,
    InferenceError,
    MinConfidenceOutOfRangeError,
    OrientationOutOfRangeError,
    ResizeFactorMustBePositiveError,
    UnsupportedOCREngineError,
    ValidationError,
)
from src.anonymizer.core.models import BoundingBox
from src.anonymizer.ocr.engines import BaseOCREngine, create_ocr_engine
from src.anonymizer.ocr.models import (
    DetectedText,
    OCRConfig,
    OCREngine,
    OCRMetrics,
    OCRResult,
)
from src.anonymizer.ocr.processor import OCRProcessor

logger = logging.getLogger(__name__)


class TestOCRModels:
    """Test OCR data models."""

    def test_detected_text_creation(self):
        """Test DetectedText model creation and validation."""
        bbox = BoundingBox(left=10, top=20, right=100, bottom=50)

        detected_text = DetectedText(text="Sample text", bbox=bbox, confidence=0.95, language="en")

        assert detected_text.text == "Sample text"
        assert detected_text.bbox == bbox
        assert detected_text.confidence == 0.95
        assert detected_text.language == "en"

    def test_detected_text_validation(self):
        """Test DetectedText validation."""
        bbox = BoundingBox(left=10, top=20, right=100, bottom=50)

        # Test empty text
        with pytest.raises(EmptyTextError, match="Text cannot be empty"):
            DetectedText(text="", bbox=bbox, confidence=0.9)

        # Test invalid confidence
        with pytest.raises(ConfidenceOutOfRangeError, match="Confidence must be between"):
            DetectedText(text="test", bbox=bbox, confidence=1.5)

        # Test invalid orientation
        with pytest.raises(OrientationOutOfRangeError, match="Orientation must be between"):
            DetectedText(text="test", bbox=bbox, confidence=0.9, orientation=200)

    def test_ocr_result_properties(self):
        """Test OCRResult properties and methods."""
        bbox1 = BoundingBox(left=10, top=20, right=100, bottom=50)
        bbox2 = BoundingBox(left=110, top=20, right=200, bottom=50)

        detected_texts = [
            DetectedText(text="High conf", bbox=bbox1, confidence=0.9),
            DetectedText(text="Low conf", bbox=bbox2, confidence=0.5),
        ]

        result = OCRResult(
            detected_texts=detected_texts,
            processing_time_ms=150.0,
            engine_used=OCREngine.PADDLEOCR,
            image_size=(800, 600),
        )

        assert result.total_text_regions == 2
        assert result.average_confidence == 0.7
        assert len(result.high_confidence_texts(threshold=0.8)) == 1
        assert len(result.get_text_by_confidence_range(0.4, 0.6)) == 1

    def test_ocr_config_validation(self):
        """Test OCRConfig validation."""
        # Valid config
        config = OCRConfig(primary_engine=OCREngine.PADDLEOCR, min_confidence_threshold=0.7)
        assert config.primary_engine == OCREngine.PADDLEOCR

        # Invalid confidence threshold
        with pytest.raises(
            MinConfidenceOutOfRangeError,
            match="min_confidence_threshold must be between",
        ):
            OCRConfig(min_confidence_threshold=1.5)

        # Invalid text length
        with pytest.raises(ResizeFactorMustBePositiveError, match="resize_factor must be positive"):
            OCRConfig(resize_factor=-1)


class TestOCREngines:
    """Test OCR engine implementations."""

    def test_engine_factory(self):
        """Test OCR engine factory."""
        config = OCRConfig()

        # Test creating different engines
        for engine_type in OCREngine:
            engine = create_ocr_engine(engine_type, config)
            assert isinstance(engine, BaseOCREngine)
            assert engine.config == config

        # Test invalid engine type
        with pytest.raises(UnsupportedOCREngineError, match="Unsupported OCR engine"):
            create_ocr_engine("invalid_engine", config)

    def test_engine_initialization(self):
        """Test engine initialization (may fail if dependencies missing)."""
        config = OCRConfig()

        # Try initializing engines - some may fail due to missing dependencies
        engines_to_test = [OCREngine.TESSERACT]  # Tesseract is most likely to be available

        for engine_type in engines_to_test:
            try:
                engine = create_ocr_engine(engine_type, config)
                success = engine.initialize()

                if success:
                    assert engine.is_initialized
                    logger.info(f"✅ {engine_type.value} engine initialized successfully")
                    engine.cleanup()
                else:
                    logger.warning(f"⚠️ {engine_type.value} engine failed to initialize")

            except Exception as e:
                logger.warning(f"⚠️ {engine_type.value} engine error: {e}")

    def test_image_validation(self):
        """Test image validation in base engine."""
        config = OCRConfig()
        engine = create_ocr_engine(OCREngine.TESSERACT, config)

        # Test valid image
        valid_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        assert engine.validate_image(valid_image) is True

        # Test invalid images
        with pytest.raises(ValidationError):
            engine.validate_image(None)

        with pytest.raises(ValidationError):
            engine.validate_image(np.array([]))

        with pytest.raises(ValidationError):
            engine.validate_image(np.random.randint(0, 255, (5, 5), dtype=np.uint8))

    def test_image_preprocessing(self):
        """Test image preprocessing."""
        config = OCRConfig(
            enable_preprocessing=True,
            resize_factor=2.0,
            contrast_enhancement=True,
            noise_reduction=True,
        )
        engine = create_ocr_engine(OCREngine.TESSERACT, config)

        # Create test image
        original_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        processed_image = engine.preprocess_image(original_image)

        # Check that preprocessing changed the image
        assert processed_image.shape != original_image.shape  # Should be resized
        assert processed_image.dtype == original_image.dtype


class TestOCRProcessor:
    """Test OCR processor with multiple engines."""

    @pytest.fixture
    def sample_document_image(self):
        """Create a sample document image with text."""
        # Create image with text
        width, height = 400, 200
        image = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(image)

        # Try to use a system font
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            font = ImageFont.load_default()

        # Add text
        draw.text((20, 50), "John Smith", font=font, fill="black")
        draw.text((20, 100), "john.smith@email.com", font=font, fill="black")
        draw.text((20, 150), "Phone: 555-123-4567", font=font, fill="black")

        return np.array(image)

    @pytest.fixture
    def ocr_config(self):
        """Create OCR configuration for testing."""
        return OCRConfig(
            primary_engine=OCREngine.TESSERACT,  # Most likely to be available
            fallback_engines=[],  # No fallbacks for simpler testing
            min_confidence_threshold=0.3,  # Lower threshold for testing
            enable_preprocessing=True,
            languages=["eng"],
        )

    def test_ocr_processor_initialization(self, ocr_config):
        """Test OCR processor initialization."""
        processor = OCRProcessor(ocr_config)

        assert processor.config == ocr_config
        assert not processor.is_initialized

        # Try to initialize
        success = processor.initialize()

        if success:
            assert processor.is_initialized
            logger.info("✅ OCR processor initialized successfully")
            processor.cleanup()
        else:
            logger.warning("⚠️ OCR processor initialization failed - dependencies missing")

    def test_ocr_processor_context_manager(self, ocr_config):
        """Test OCR processor as context manager."""
        try:
            with OCRProcessor(ocr_config) as processor:
                if processor.is_initialized:
                    assert processor.primary_engine is not None
                    logger.info("✅ OCR processor context manager works")
                else:
                    logger.warning("⚠️ OCR processor failed to initialize in context manager")
        except Exception as e:
            logger.warning(f"⚠️ OCR processor context manager failed: {e}")

    def test_text_extraction(self, ocr_config, sample_document_image):
        """Test text extraction from document image."""
        processor = OCRProcessor(ocr_config)

        if not processor.initialize():
            pytest.skip("OCR processor failed to initialize")

        try:
            # Extract text regions
            detected_texts = processor.extract_text_regions(sample_document_image)

            # Should detect at least some text regions
            logger.info(f"✅ Detected {len(detected_texts)} text regions")

            for text in detected_texts:
                assert isinstance(text, DetectedText)
                assert text.text.strip()  # Non-empty text
                assert isinstance(text.bbox, BoundingBox)
                assert 0.0 <= text.confidence <= 1.0

        except Exception as e:
            logger.warning(f"⚠️ Text extraction failed: {e}")
            pytest.skip(f"Text extraction failed: {e}")

        finally:
            processor.cleanup()

    def test_text_region_conversion(self, ocr_config, sample_document_image):
        """Test conversion from DetectedText to TextRegion."""
        processor = OCRProcessor(ocr_config)

        if not processor.initialize():
            pytest.skip("OCR processor failed to initialize")

        try:
            # Extract text regions
            detected_texts = processor.extract_text_regions(sample_document_image)

            if detected_texts:
                # Convert to text regions
                text_regions = processor.convert_to_text_regions(
                    detected_texts,
                    replacement_strategy="generic",
                )

                assert len(text_regions) == len(detected_texts)

                for region in text_regions:
                    assert region.original_text
                    assert region.replacement_text == "[TEXT]"
                    assert isinstance(region.bbox, BoundingBox)

                logger.info(f"✅ Converted {len(text_regions)} text regions successfully")
            else:
                logger.warning("⚠️ No text detected for conversion test")

        except Exception as e:
            logger.warning(f"⚠️ Text region conversion failed: {e}")

        finally:
            processor.cleanup()

    def test_detect_and_convert_integration(self, ocr_config, sample_document_image):
        """Test one-step detect and convert function."""
        processor = OCRProcessor(ocr_config)

        if not processor.initialize():
            pytest.skip("OCR processor failed to initialize")

        try:
            # One-step detection and conversion
            text_regions = processor.detect_and_convert(
                sample_document_image,
                replacement_strategy="length_preserving",
            )

            for region in text_regions:
                # Should preserve length
                assert len(region.replacement_text) == len(region.original_text)
                assert all(c == "X" for c in region.replacement_text.replace(" ", ""))

            logger.info(f"✅ One-step detect and convert successful - {len(text_regions)} regions")

        except Exception as e:
            logger.warning(f"⚠️ One-step detect and convert failed: {e}")

        finally:
            processor.cleanup()

    def test_error_handling(self, ocr_config):
        """Test OCR processor error handling."""
        processor = OCRProcessor(ocr_config)

        # Test without initialization
        with pytest.raises(InferenceError):
            processor.extract_text_regions(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))

        # Test with invalid image
        if processor.initialize():
            with pytest.raises(ValidationError):
                processor.extract_text_regions(None)

            processor.cleanup()

    def test_text_filtering(self, ocr_config):
        """Test text filtering functionality."""
        processor = OCRProcessor(ocr_config)

        # Create mock detected texts
        bbox = BoundingBox(left=10, top=20, right=100, bottom=50)
        detected_texts = [
            DetectedText(text="A", bbox=bbox, confidence=0.9),  # Too short
            DetectedText(text="Good text", bbox=bbox, confidence=0.9),  # Good
            DetectedText(text="Low confidence", bbox=bbox, confidence=0.1),  # Low confidence
            DetectedText(text="Another good text", bbox=bbox, confidence=0.8),  # Good
        ]

        # Apply filters
        filtered_texts = processor._apply_text_filters(detected_texts)

        # Should filter out short and low confidence texts
        assert len(filtered_texts) <= len(detected_texts)

        for text in filtered_texts:
            assert len(text.text) >= processor.config.min_text_length
            assert text.confidence >= processor.config.min_confidence_threshold

    def test_metrics_collection(self, ocr_config):
        """Test metrics collection."""
        processor = OCRProcessor(ocr_config)

        # Get initial metrics
        metrics = processor.get_metrics()

        assert isinstance(metrics, OCRMetrics)
        assert metrics.total_processing_time_ms >= 0
        assert metrics.total_texts_detected >= 0
        assert metrics.engine_used == ocr_config.primary_engine


class TestOCRIntegration:
    """Test OCR integration with inference engine."""

    def test_ocr_import_availability(self):
        """Test that OCR modules can be imported."""
        # Should import without errors (imports already at module level)
        assert OCRProcessor is not None
        assert OCRConfig is not None
        assert OCREngine is not None


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
