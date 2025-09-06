"""Unit tests for OCR timeout handling - Imperative style.

Tests the OCR timeout functionality with comprehensive coverage of timeout scenarios,
error handling, and fallback strategies.
"""

import time
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.anonymizer.core.models import BoundingBox
from src.anonymizer.ocr.models import DetectedText, OCRConfig, OCREngine, OCRResult
from src.anonymizer.ocr.processor import OCRProcessor, OCRTimeoutError, timeout_context


class TestTimeoutContext:
    """Test timeout context manager functionality."""

    def test_timeout_context_success(self):
        """Test timeout context when operation completes within timeout."""
        with timeout_context(1.0, "test operation"):
            # This should complete without timeout
            time.sleep(0.1)

    def test_timeout_context_timeout(self):
        """Test timeout context when operation exceeds timeout."""
        with pytest.raises(OCRTimeoutError) as exc_info:
            with timeout_context(0.1, "test operation"):
                # This should timeout
                time.sleep(0.2)

        assert exc_info.value.timeout_seconds == 0.1
        assert "test operation" in str(exc_info.value)

    def test_ocr_timeout_error_message(self):
        """Test OCRTimeoutError message formatting."""
        error = OCRTimeoutError(5.0, "text detection")

        assert error.timeout_seconds == 5.0
        assert error.operation == "text detection"
        assert "text detection timed out after 5.0s" in str(error)

    def test_ocr_timeout_error_default_operation(self):
        """Test OCRTimeoutError with default operation name."""
        error = OCRTimeoutError(2.5)

        assert error.timeout_seconds == 2.5
        assert error.operation == "OCR processing"
        assert "OCR processing timed out after 2.5s" in str(error)


class TestOCRProcessorTimeout:
    """Test OCR processor timeout functionality."""

    def create_mock_config(self, **overrides):
        """Create mock OCR config with defaults."""
        defaults = {
            "primary_engine": OCREngine.PADDLEOCR,
            "fallback_engines": [OCREngine.EASYOCR],
            "min_confidence_threshold": 0.5,
            "languages": ["en"],
            "enable_preprocessing": True,
            "timeout_seconds": 2.0,
            "filter_short_texts": True,
            "filter_low_confidence": True,
            "use_gpu": False,
        }
        defaults.update(overrides)
        return OCRConfig(**defaults)

    def create_mock_detected_text(self):
        """Create a mock DetectedText object."""
        return DetectedText(
            text="Sample text",
            bbox=BoundingBox(left=10, top=10, right=100, bottom=30),
            confidence=0.9,
            language="en",
        )

    def create_mock_ocr_result(self, success=True, num_texts=1):
        """Create a mock OCRResult object."""
        detected_texts = []
        if success and num_texts > 0:
            for i in range(num_texts):
                detected_texts.append(self.create_mock_detected_text())

        return OCRResult(
            detected_texts=detected_texts,
            processing_time_ms=100.0,
            engine_used=OCREngine.PADDLEOCR,
            image_size=(640, 480),
            success=success,
        )

    @patch("src.anonymizer.ocr.processor.create_ocr_engine")
    def test_extract_text_regions_with_timeout_override(self, mock_create_engine):
        """Test extract_text_regions with timeout override."""
        config = self.create_mock_config(timeout_seconds=5.0)
        processor = OCRProcessor(config)

        # Mock primary engine
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_engine.detect_text.return_value = self.create_mock_ocr_result()
        mock_create_engine.return_value = mock_engine

        processor.primary_engine = mock_engine
        processor.is_initialized = True

        # Test with timeout override
        image = np.ones((100, 100, 3), dtype=np.uint8)

        result = processor.extract_text_regions(image, timeout_override=1.0)

        # Should complete successfully
        assert len(result) == 1
        assert result[0].text == "Sample text"

    @patch("src.anonymizer.ocr.processor.create_ocr_engine")
    def test_extract_text_regions_primary_engine_timeout(self, mock_create_engine):
        """Test extract_text_regions when primary engine times out."""
        config = self.create_mock_config(timeout_seconds=0.1)
        processor = OCRProcessor(config)

        # Mock primary engine that takes too long
        mock_primary = Mock()
        mock_primary.is_initialized = True

        def slow_detect_text(image):
            time.sleep(0.2)  # Longer than timeout
            return self.create_mock_ocr_result()

        mock_primary.detect_text.side_effect = slow_detect_text

        # Mock fast fallback engine
        mock_fallback = Mock()
        mock_fallback.is_initialized = True
        mock_fallback.detect_text.return_value = self.create_mock_ocr_result()

        processor.primary_engine = mock_primary
        processor.fallback_engines = [mock_fallback]
        processor.is_initialized = True

        image = np.ones((100, 100, 3), dtype=np.uint8)

        # Should fall back to fallback engine
        result = processor.extract_text_regions(image)

        # Should get results from fallback engine
        assert len(result) == 1
        assert result[0].text == "Sample text"

    @patch("src.anonymizer.ocr.processor.create_ocr_engine")
    def test_extract_text_regions_all_engines_timeout(self, mock_create_engine):
        """Test extract_text_regions when all engines timeout."""
        config = self.create_mock_config(timeout_seconds=0.1)
        processor = OCRProcessor(config)

        # Mock engines that all take too long
        def slow_detect_text(image):
            time.sleep(0.2)  # Longer than timeout
            return self.create_mock_ocr_result()

        mock_primary = Mock()
        mock_primary.is_initialized = True
        mock_primary.detect_text.side_effect = slow_detect_text

        mock_fallback = Mock()
        mock_fallback.is_initialized = True
        mock_fallback.detect_text.side_effect = slow_detect_text

        processor.primary_engine = mock_primary
        processor.fallback_engines = [mock_fallback]
        processor.is_initialized = True

        image = np.ones((100, 100, 3), dtype=np.uint8)

        # Should timeout and raise OCRTimeoutError
        with pytest.raises(OCRTimeoutError):
            processor.extract_text_regions(image)

    @patch("src.anonymizer.ocr.processor.create_ocr_engine")
    def test_extract_text_regions_no_time_for_fallbacks(self, mock_create_engine):
        """Test extract_text_regions when primary engine uses all available time."""
        config = self.create_mock_config(timeout_seconds=0.1)  # Very short timeout
        processor = OCRProcessor(config)

        # Mock primary engine that takes longer than the timeout
        mock_primary = Mock()
        mock_primary.is_initialized = True

        def slow_no_results(image):
            time.sleep(0.2)  # Much longer than total timeout
            return self.create_mock_ocr_result(success=True, num_texts=0)  # No results

        mock_primary.detect_text.side_effect = slow_no_results

        # Mock fallback engine that also times out
        mock_fallback = Mock()
        mock_fallback.is_initialized = True

        def slow_fallback(image):
            time.sleep(0.2)  # Also too slow
            return self.create_mock_ocr_result()

        mock_fallback.detect_text.side_effect = slow_fallback

        processor.primary_engine = mock_primary
        processor.fallback_engines = [mock_fallback]
        processor.is_initialized = True

        image = np.ones((100, 100, 3), dtype=np.uint8)

        # Should timeout because all engines are too slow
        with pytest.raises(OCRTimeoutError):
            processor.extract_text_regions(image)

    @patch("src.anonymizer.ocr.processor.create_ocr_engine")
    def test_detect_and_convert_with_timeout(self, mock_create_engine):
        """Test detect_and_convert method with timeout."""
        config = self.create_mock_config(timeout_seconds=1.0)
        processor = OCRProcessor(config)

        # Mock engine
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_engine.detect_text.return_value = self.create_mock_ocr_result()

        processor.primary_engine = mock_engine
        processor.is_initialized = True

        image = np.ones((100, 100, 3), dtype=np.uint8)

        # Test with timeout override
        result = processor.detect_and_convert(image, timeout_override=2.0)

        assert len(result) == 1
        assert result[0].original_text == "Sample text"
        assert result[0].replacement_text == "[TEXT]"

    @patch("src.anonymizer.ocr.processor.create_ocr_engine")
    def test_extract_text_regions_updates_metrics_on_timeout(self, mock_create_engine):
        """Test that metrics are properly updated when timeout occurs."""
        config = self.create_mock_config(timeout_seconds=0.1)
        processor = OCRProcessor(config)

        # Mock engine that times out
        mock_engine = Mock()
        mock_engine.is_initialized = True

        def slow_detect_text(image):
            time.sleep(0.2)
            return self.create_mock_ocr_result()

        mock_engine.detect_text.side_effect = slow_detect_text

        processor.primary_engine = mock_engine
        processor.is_initialized = True

        # Track initial metrics
        initial_processing_time = processor.total_processing_time
        initial_images_processed = processor.total_images_processed

        image = np.ones((100, 100, 3), dtype=np.uint8)

        # Should timeout and update metrics
        with pytest.raises(OCRTimeoutError):
            processor.extract_text_regions(image)

        # Metrics should be updated even on timeout
        assert processor.total_processing_time > initial_processing_time
        assert processor.total_images_processed == initial_images_processed + 1
        assert processor.successful_detections == 0  # No successful detection

    @patch("src.anonymizer.ocr.processor.create_ocr_engine")
    def test_extract_text_regions_handles_unexpected_error(self, mock_create_engine):
        """Test extract_text_regions handles unexpected errors properly."""
        config = self.create_mock_config()
        processor = OCRProcessor(config)

        # Mock engine that raises unexpected error
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_engine.detect_text.side_effect = ValueError("Unexpected error")

        processor.primary_engine = mock_engine
        processor.is_initialized = True

        image = np.ones((100, 100, 3), dtype=np.uint8)

        # Should wrap unexpected error in InferenceError
        from src.anonymizer.core.exceptions import InferenceError

        with pytest.raises(InferenceError) as exc_info:
            processor.extract_text_regions(image)

        assert "OCR processing failed" in str(exc_info.value)

    def test_extract_text_regions_not_initialized(self):
        """Test extract_text_regions when processor is not initialized."""
        config = self.create_mock_config()
        processor = OCRProcessor(config)
        # Don't initialize the processor

        image = np.ones((100, 100, 3), dtype=np.uint8)

        from src.anonymizer.core.exceptions import InferenceError

        with pytest.raises(InferenceError) as exc_info:
            processor.extract_text_regions(image)

        assert "OCR processor not initialized" in str(exc_info.value)

    def test_extract_text_regions_invalid_image(self):
        """Test extract_text_regions with invalid image."""
        config = self.create_mock_config()
        processor = OCRProcessor(config)
        processor.is_initialized = True

        from src.anonymizer.core.exceptions import ValidationError

        # Test with None image
        with pytest.raises(ValidationError):
            processor.extract_text_regions(None)

        # Test with empty image
        empty_image = np.array([])
        with pytest.raises(ValidationError):
            processor.extract_text_regions(empty_image)
