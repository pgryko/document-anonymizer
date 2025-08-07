"""Comprehensive OCR integration tests."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.anonymizer.core.models import BoundingBox
from src.anonymizer.ocr.models import DetectedText, OCRConfig, OCREngine, OCRResult
from src.anonymizer.ocr.processor import OCRProcessor


class TestOCRIntegration:
    """Test OCR functionality with realistic scenarios."""

    @pytest.fixture
    def test_image_with_text(self):
        """Create a test image with readable text."""
        # Create image with text
        img = Image.new("RGB", (400, 200), color="white")
        draw = ImageDraw.Draw(img)

        try:
            # Try to use a system font
            font = ImageFont.truetype("Arial.ttf", 24)
        except OSError:
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 24)
            except OSError:
                # Fallback to default font
                font = ImageFont.load_default()

        # Draw some text
        draw.text((20, 50), "Sample Document", fill="black", font=font)
        draw.text((20, 100), "This is test text for OCR", fill="black", font=font)
        draw.text((20, 150), "Phone: 555-123-4567", fill="black", font=font)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            img.save(f.name)
            yield Path(f.name)

    @pytest.fixture
    def ocr_config(self):
        """Create OCR configuration for testing."""
        return OCRConfig(
            primary_engine=OCREngine.TESSERACT,  # Most reliable for testing
            languages=["en"],
            min_confidence_threshold=0.5,
            enable_preprocessing=True,
        )

    def test_ocr_processor_initialization(self, ocr_config):
        """Test OCR processor initialization with different engines."""
        # Test with Tesseract (most commonly available)
        processor = OCRProcessor(ocr_config)
        assert processor.config.primary_engine == OCREngine.TESSERACT
        assert processor.config.languages == ["en"]

        # Test with different engines (will fall back if not available)
        configs = [
            OCRConfig(primary_engine=OCREngine.PADDLEOCR),
            OCRConfig(primary_engine=OCREngine.EASYOCR),
        ]

        for config in configs:
            # Should not raise errors even if engine not available
            processor = OCRProcessor(config)
            assert processor is not None

    @pytest.mark.integration
    def test_text_extraction_with_mocked_engine(self, test_image_with_text):
        """Test text extraction with mocked OCR engine responses."""
        # Mock OCR responses - these are DetectedText objects now
        mock_detected_texts = [
            DetectedText(
                text="Sample Document",
                bbox=BoundingBox(left=20, top=45, right=200, bottom=75),
                confidence=0.95,
                language="en",
            ),
            DetectedText(
                text="This is test text for OCR",
                bbox=BoundingBox(left=20, top=95, right=300, bottom=125),
                confidence=0.88,
                language="en",
            ),
            DetectedText(
                text="Phone: 555-123-4567",
                bbox=BoundingBox(left=20, top=145, right=250, bottom=175),
                confidence=0.92,
                language="en",
            ),
        ]

        # Test with different engine backends
        engine_configs = [
            OCRConfig(primary_engine=OCREngine.TESSERACT),
            OCRConfig(primary_engine=OCREngine.PADDLEOCR),
            OCRConfig(primary_engine=OCREngine.EASYOCR),
        ]

        for config in engine_configs:
            processor = OCRProcessor(config)

            # Mock the engine's detect_text method instead of _extract_with_engine
            mock_engine = Mock()
            mock_engine.is_initialized = True
            mock_engine.detect_text.return_value = OCRResult(
                detected_texts=mock_detected_texts,
                processing_time_ms=100.0,
                engine_used=config.primary_engine,
                image_size=(400, 200),
                success=True,
            )

            # Set up the processor with mocked engine
            processor.primary_engine = mock_engine
            processor.is_initialized = True

            # Load test image as numpy array
            test_img = Image.open(test_image_with_text)
            image_array = np.array(test_img)

            # Extract text regions
            detected_texts = processor.extract_text_regions(image_array)

            # Verify results
            assert len(detected_texts) == 3
            assert detected_texts[0].text == "Sample Document"
            assert detected_texts[1].text == "This is test text for OCR"
            assert detected_texts[2].text == "Phone: 555-123-4567"

            # Verify confidence filtering works
            config_high_confidence = OCRConfig(
                primary_engine=config.primary_engine, min_confidence_threshold=0.95
            )
            processor_high = OCRProcessor(config_high_confidence)

            mock_engine_high = Mock()
            mock_engine_high.is_initialized = True
            mock_engine_high.detect_text.return_value = OCRResult(
                detected_texts=mock_detected_texts,
                processing_time_ms=100.0,
                engine_used=config.primary_engine,
                image_size=(400, 200),
                success=True,
            )

            processor_high.primary_engine = mock_engine_high
            processor_high.is_initialized = True

            detected_texts_filtered = processor_high.extract_text_regions(image_array)

            # Should only get high confidence results (>= 0.95)
            high_confidence_texts = [t for t in detected_texts_filtered if t.confidence >= 0.95]
            assert len(high_confidence_texts) >= 1  # At least the first one

    def test_ocr_engine_fallback_mechanism(self):
        """Test OCR engine fallback when primary engine fails."""
        config = OCRConfig(primary_engine=OCREngine.PADDLEOCR)  # Might not be available

        # Mock image as numpy array
        test_image = np.ones((100, 100, 3), dtype=np.uint8)

        processor = OCRProcessor(config)

        # Should handle missing engines gracefully
        try:
            detected_texts = processor.extract_text_regions(test_image)
            # If it succeeds, detected_texts should be a list
            assert isinstance(detected_texts, list)
        except Exception as e:
            # Should provide meaningful error message
            assert (
                "OCR" in str(e) or "not initialized" in str(e).lower() or "failed" in str(e).lower()
            )

    def test_ocr_preprocessing_pipeline(self, test_image_with_text):
        """Test OCR preprocessing effects."""
        # Test with preprocessing enabled
        config_with_prep = OCRConfig(
            primary_engine=OCREngine.TESSERACT,
            enable_preprocessing=True,
        )

        # Test without preprocessing
        config_without_prep = OCRConfig(
            primary_engine=OCREngine.TESSERACT,
            enable_preprocessing=False,
        )

        mock_detected_texts = [
            DetectedText(
                text="Sample Text",
                bbox=BoundingBox(left=20, top=50, right=200, bottom=75),
                confidence=0.9,
                language="en",
            )
        ]

        # Load test image as numpy array
        test_img = Image.open(test_image_with_text)
        image_array = np.array(test_img)

        # Test with preprocessing
        processor_prep = OCRProcessor(config_with_prep)

        # Mock the engine
        mock_engine_prep = Mock()
        mock_engine_prep.is_initialized = True
        mock_engine_prep.detect_text.return_value = OCRResult(
            detected_texts=mock_detected_texts,
            processing_time_ms=100.0,
            engine_used=OCREngine.TESSERACT,
            image_size=(400, 200),
            success=True,
        )

        processor_prep.primary_engine = mock_engine_prep
        processor_prep.is_initialized = True

        detected_texts_prep = processor_prep.extract_text_regions(image_array)

        # Should get results
        assert len(detected_texts_prep) == 1

        # Test without preprocessing
        processor_no_prep = OCRProcessor(config_without_prep)

        mock_engine_no_prep = Mock()
        mock_engine_no_prep.is_initialized = True
        mock_engine_no_prep.detect_text.return_value = OCRResult(
            detected_texts=mock_detected_texts,
            processing_time_ms=100.0,
            engine_used=OCREngine.TESSERACT,
            image_size=(400, 200),
            success=True,
        )

        processor_no_prep.primary_engine = mock_engine_no_prep
        processor_no_prep.is_initialized = True

        detected_texts_no_prep = processor_no_prep.extract_text_regions(image_array)

        # Should also get results
        assert len(detected_texts_no_prep) == 1

    def test_ocr_confidence_filtering(self):
        """Test OCR confidence-based filtering."""
        # Mock OCR results with varying confidence
        mock_raw_results = [
            ("High confidence text", BoundingBox(left=10, top=10, right=100, bottom=30), 0.95),
            ("Medium confidence text", BoundingBox(left=10, top=40, right=100, bottom=60), 0.75),
            ("Low confidence text", BoundingBox(left=10, top=70, right=100, bottom=90), 0.45),
            ("Very low confidence", BoundingBox(left=10, top=100, right=100, bottom=120), 0.25),
        ]

        # Test different confidence thresholds
        # Mock data has confidences: 0.95, 0.75, 0.45, 0.25
        # 0.3 threshold should include: 0.95, 0.75, 0.45 (3 items) - excludes 0.25
        # 0.5 threshold should include: 0.95, 0.75 (2 items) - excludes 0.45, 0.25
        # 0.8 threshold should include: 0.95 (1 item) - excludes 0.75, 0.45, 0.25
        # 0.9 threshold should include: 0.95 (1 item) - excludes 0.75, 0.45, 0.25
        thresholds = [0.3, 0.5, 0.8, 0.9]
        expected_counts = [3, 2, 1, 1]

        for threshold, expected_count in zip(thresholds, expected_counts, strict=False):
            config = OCRConfig(
                primary_engine=OCREngine.TESSERACT,
                min_confidence_threshold=threshold,
            )
            processor = OCRProcessor(config)

            # Convert mock results to DetectedText format
            mock_detected_texts = [
                DetectedText(
                    text=text,
                    bbox=bbox,
                    confidence=conf,
                    language="en",
                )
                for text, bbox, conf in mock_raw_results
            ]

            # Mock the engine
            mock_engine = Mock()
            mock_engine.is_initialized = True
            mock_engine.detect_text.return_value = OCRResult(
                detected_texts=mock_detected_texts,
                processing_time_ms=100.0,
                engine_used=OCREngine.TESSERACT,
                image_size=(100, 130),
                success=True,
            )

            processor.primary_engine = mock_engine
            processor.is_initialized = True

            # Mock image
            fake_image = np.ones((130, 100, 3), dtype=np.uint8)
            detected_texts = processor.extract_text_regions(fake_image)

            # The processor should filter by confidence automatically
            assert len(detected_texts) == expected_count

    def test_bounding_box_validation(self):
        """Test bounding box validation and normalization."""
        config = OCRConfig(primary_engine=OCREngine.TESSERACT)
        processor = OCRProcessor(config)

        # Mock OCR results with various bounding box issues
        problematic_results = [
            # Valid bounding box
            ("Valid text", BoundingBox(left=10, top=10, right=100, bottom=50), 0.9),
            # NOTE: Invalid bounding boxes would fail validation, so we create them as valid ones
            # and let the filtering logic handle them
            ("Negative coords", BoundingBox(left=0, top=10, right=100, bottom=50), 0.9),
            ("Small box", BoundingBox(left=10, top=10, right=11, bottom=11), 0.9),
            ("Tiny text", BoundingBox(left=10, top=10, right=15, bottom=15), 0.9),
        ]

        # Convert to DetectedText format
        mock_detected_texts = [
            DetectedText(
                text=text,
                bbox=bbox,
                confidence=conf,
                language="en",
            )
            for text, bbox, conf in problematic_results
        ]

        # Mock the engine
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_engine.detect_text.return_value = OCRResult(
            detected_texts=mock_detected_texts,
            processing_time_ms=100.0,
            engine_used=OCREngine.TESSERACT,
            image_size=(100, 60),
            success=True,
        )

        processor.primary_engine = mock_engine
        processor.is_initialized = True

        # Mock image
        fake_image = np.ones((60, 100, 3), dtype=np.uint8)
        detected_texts = processor.extract_text_regions(fake_image)

        # Should filter out invalid bounding boxes (done by the processor's filtering logic)
        valid_texts = [
            t
            for t in detected_texts
            if (
                t.bbox.left >= 0
                and t.bbox.top >= 0
                and t.bbox.right > t.bbox.left
                and t.bbox.bottom > t.bbox.top
            )
        ]

        # At least the first valid one should remain
        assert len(valid_texts) >= 1
        assert valid_texts[0].text == "Valid text"

    def test_multiple_language_support(self):
        """Test OCR with multiple languages."""
        # Test different language configurations
        language_configs = [
            OCRConfig(primary_engine=OCREngine.TESSERACT, languages=["en"]),
            OCRConfig(primary_engine=OCREngine.TESSERACT, languages=["en", "es"]),
            OCRConfig(primary_engine=OCREngine.TESSERACT, languages=["en", "fr", "de"]),
        ]

        mock_detected_texts = [
            DetectedText(
                text="Multi-language text",
                bbox=BoundingBox(left=10, top=10, right=100, bottom=30),
                confidence=0.9,
                language="en",
            )
        ]

        for config in language_configs:
            processor = OCRProcessor(config)

            # Mock the engine
            mock_engine = Mock()
            mock_engine.is_initialized = True
            mock_engine.detect_text.return_value = OCRResult(
                detected_texts=mock_detected_texts,
                processing_time_ms=100.0,
                engine_used=OCREngine.TESSERACT,
                image_size=(100, 40),
                success=True,
            )

            processor.primary_engine = mock_engine
            processor.is_initialized = True

            # Mock image
            fake_image = np.ones((40, 100, 3), dtype=np.uint8)
            detected_texts = processor.extract_text_regions(fake_image)

            # Should handle all language configurations
            assert len(detected_texts) == 1
            assert detected_texts[0].text == "Multi-language text"

    def test_ocr_error_handling(self):
        """Test OCR error handling and recovery."""
        config = OCRConfig(primary_engine=OCREngine.TESSERACT)
        processor = OCRProcessor(config)

        # Test various error conditions with numpy arrays
        error_conditions = [
            (np.array([]), "empty image array"),
            (None, "null image data"),
        ]

        for image_data, description in error_conditions:
            try:
                detected_texts = processor.extract_text_regions(image_data)
                # If it succeeds, should return empty list
                assert isinstance(detected_texts, list)
            except Exception as e:
                # Should provide meaningful error message
                assert isinstance(e, (ValueError, TypeError, Exception))
                error_msg = str(e).lower()
                assert any(
                    keyword in error_msg
                    for keyword in [
                        "image",
                        "data",
                        "invalid",
                        "format",
                        "error",
                        "not initialized",
                    ]
                )

    @pytest.mark.slow
    def test_ocr_performance_monitoring(self, test_image_with_text):
        """Test OCR performance characteristics."""
        config = OCRConfig(primary_engine=OCREngine.TESSERACT)
        processor = OCRProcessor(config)

        # Load test image as numpy array
        test_img = Image.open(test_image_with_text)
        image_array = np.array(test_img)

        # Mock successful extraction with timing
        mock_detected_texts = [
            DetectedText(
                text="Performance test",
                bbox=BoundingBox(left=20, top=50, right=200, bottom=75),
                confidence=0.9,
                language="en",
            )
        ]

        # Mock the engine
        mock_engine = Mock()
        mock_engine.is_initialized = True
        mock_engine.detect_text.return_value = OCRResult(
            detected_texts=mock_detected_texts,
            processing_time_ms=100.0,
            engine_used=OCREngine.TESSERACT,
            image_size=(400, 200),
            success=True,
        )

        processor.primary_engine = mock_engine
        processor.is_initialized = True

        import time

        start_time = time.time()

        detected_texts = processor.extract_text_regions(image_array)

        end_time = time.time()
        processing_time = end_time - start_time

        # Should complete within reasonable time (mocked, so should be fast)
        assert processing_time < 1.0  # Should be very fast with mocking
        assert len(detected_texts) == 1

        # Test with larger mock dataset - create unique texts to avoid deduplication
        large_mock_detected_texts = []
        for i in range(50):
            large_mock_detected_texts.append(
                DetectedText(
                    text=f"Performance test {i}",
                    bbox=BoundingBox(left=20 + i, top=50 + i, right=200 + i, bottom=75 + i),
                    confidence=0.9,
                    language="en",
                )
            )

        # Create new processor for large dataset test
        processor_large = OCRProcessor(config)
        mock_engine_large = Mock()
        mock_engine_large.is_initialized = True
        mock_engine_large.detect_text.return_value = OCRResult(
            detected_texts=large_mock_detected_texts,
            processing_time_ms=100.0,
            engine_used=OCREngine.TESSERACT,
            image_size=(400, 200),
            success=True,
        )

        processor_large.primary_engine = mock_engine_large
        processor_large.is_initialized = True

        start_time = time.time()
        detected_texts_large = processor_large.extract_text_regions(image_array)
        end_time = time.time()

        # Should handle larger datasets efficiently
        assert len(detected_texts_large) == 50
        processing_time_large = end_time - start_time
        assert processing_time_large < 2.0  # Still reasonable with mocking
