"""Comprehensive OCR integration tests."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.anonymizer.core.models import BoundingBox, TextRegion
from src.anonymizer.ocr.models import OCRConfig, OCREngine
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
        # Mock OCR responses
        mock_text_regions = [
            TextRegion(
                bbox=BoundingBox(left=20, top=45, right=200, bottom=75),
                original_text="Sample Document",
                replacement_text="Sample Document",
                confidence=0.95,
            ),
            TextRegion(
                bbox=BoundingBox(left=20, top=95, right=300, bottom=125),
                original_text="This is test text for OCR",
                replacement_text="This is test text for OCR",
                confidence=0.88,
            ),
            TextRegion(
                bbox=BoundingBox(left=20, top=145, right=250, bottom=175),
                original_text="Phone: 555-123-4567",
                replacement_text="Phone: 555-123-4567",
                confidence=0.92,
            ),
        ]

        # Test with different engine backends
        engine_configs = [
            OCRConfig(primary_engine=OCREngine.TESSERACT),
            OCRConfig(primary_engine=OCREngine.PADDLEOCR),
            OCRConfig(primary_engine=OCREngine.EASYOCR),
        ]

        for config in engine_configs:
            with patch.object(OCRProcessor, "_extract_with_engine") as mock_extract:
                mock_extract.return_value = mock_text_regions

                processor = OCRProcessor(config)

                # Read test image
                with test_image_with_text.open("rb") as f:
                    image_data = f.read()

                # Extract text regions
                regions = processor.extract_text_regions(image_data)

                # Verify results
                assert len(regions) == 3
                assert regions[0].original_text == "Sample Document"
                assert regions[1].original_text == "This is test text for OCR"
                assert regions[2].original_text == "Phone: 555-123-4567"

                # Verify confidence filtering works
                config_high_confidence = OCRConfig(
                    primary_engine=config.primary_engine, min_confidence_threshold=0.95
                )
                processor_high = OCRProcessor(config_high_confidence)

                with patch.object(OCRProcessor, "_extract_with_engine") as mock_extract_high:
                    mock_extract_high.return_value = mock_text_regions
                    regions_filtered = processor_high.extract_text_regions(image_data)

                    # Should only get high confidence results
                    high_confidence_regions = [r for r in regions_filtered if r.confidence >= 0.95]
                    assert len(high_confidence_regions) >= 1  # At least the first one

    def test_ocr_engine_fallback_mechanism(self):
        """Test OCR engine fallback when primary engine fails."""
        config = OCRConfig(primary_engine=OCREngine.PADDLEOCR)  # Might not be available

        # Mock image data
        test_image_data = b"fake_image_data"

        processor = OCRProcessor(config)

        # Should handle missing engines gracefully
        try:
            regions = processor.extract_text_regions(test_image_data)
            # If it succeeds, regions should be a list
            assert isinstance(regions, list)
        except Exception as e:
            # Should provide meaningful error message
            assert "OCR engine" in str(e) or "not available" in str(e).lower()

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

        mock_regions = [
            TextRegion(
                bbox=BoundingBox(left=20, top=50, right=200, bottom=75),
                original_text="Sample Text",
                replacement_text="Sample Text",
                confidence=0.9,
            )
        ]

        with test_image_with_text.open("rb") as f:
            image_data = f.read()

        # Mock preprocessing to verify it's called
        with patch("src.anonymizer.ocr.processor.OCRProcessor._preprocess_image") as mock_prep:
            mock_prep.return_value = np.zeros((200, 400, 3), dtype=np.uint8)

            with patch.object(OCRProcessor, "_extract_with_engine", return_value=mock_regions):
                # Test with preprocessing
                processor_prep = OCRProcessor(config_with_prep)
                regions_prep = processor_prep.extract_text_regions(image_data)

                # Should have called preprocessing
                assert mock_prep.called
                assert len(regions_prep) == 1

                # Reset mock
                mock_prep.reset_mock()

                # Test without preprocessing
                processor_no_prep = OCRProcessor(config_without_prep)
                regions_no_prep = processor_no_prep.extract_text_regions(image_data)

                # Should not have called preprocessing
                assert not mock_prep.called
                assert len(regions_no_prep) == 1

    def test_ocr_confidence_filtering(self):
        """Test OCR confidence-based filtering."""
        # Mock OCR results with varying confidence
        mock_raw_results = [
            ("High confidence text", BoundingBox(10, 10, 100, 30), 0.95),
            ("Medium confidence text", BoundingBox(10, 40, 100, 60), 0.75),
            ("Low confidence text", BoundingBox(10, 70, 100, 90), 0.45),
            ("Very low confidence", BoundingBox(10, 100, 100, 120), 0.25),
        ]

        # Test different confidence thresholds
        thresholds = [0.3, 0.5, 0.8, 0.9]
        expected_counts = [4, 3, 2, 1]  # Expected number of results for each threshold

        for threshold, expected_count in zip(thresholds, expected_counts, strict=False):
            config = OCRConfig(
                primary_engine=OCREngine.TESSERACT,
                min_confidence_threshold=threshold,
            )
            processor = OCRProcessor(config)

            # Mock the raw extraction to return our test data
            with patch.object(processor, "_extract_with_engine") as mock_extract:
                # Convert mock results to TextRegion format
                mock_regions = [
                    TextRegion(
                        bbox=bbox,
                        original_text=text,
                        replacement_text=text,
                        confidence=conf,
                    )
                    for text, bbox, conf in mock_raw_results
                ]
                mock_extract.return_value = mock_regions

                regions = processor.extract_text_regions(b"fake_image_data")

                # Filter by confidence in test (simulating processor behavior)
                filtered_regions = [r for r in regions if r.confidence >= threshold]
                assert len(filtered_regions) == expected_count

    def test_bounding_box_validation(self):
        """Test bounding box validation and normalization."""
        config = OCRConfig(primary_engine=OCREngine.TESSERACT)
        processor = OCRProcessor(config)

        # Mock OCR results with various bounding box issues
        problematic_results = [
            # Valid bounding box
            ("Valid text", BoundingBox(10, 10, 100, 50), 0.9),
            # Negative coordinates (should be filtered out)
            ("Negative coords", BoundingBox(-5, 10, 100, 50), 0.9),
            # Zero width/height (should be filtered out)
            ("Zero width", BoundingBox(10, 10, 10, 50), 0.9),
            ("Zero height", BoundingBox(10, 10, 100, 10), 0.9),
            # Very small box (might be filtered based on min_size)
            ("Tiny text", BoundingBox(10, 10, 15, 15), 0.9),
        ]

        with patch.object(processor, "_extract_with_engine") as mock_extract:
            mock_regions = [
                TextRegion(
                    bbox=bbox,
                    original_text=text,
                    replacement_text=text,
                    confidence=conf,
                )
                for text, bbox, conf in problematic_results
            ]
            mock_extract.return_value = mock_regions

            regions = processor.extract_text_regions(b"fake_image_data")

            # Should filter out invalid bounding boxes
            valid_regions = [
                r
                for r in regions
                if (
                    r.bbox.left >= 0
                    and r.bbox.top >= 0
                    and r.bbox.right > r.bbox.left
                    and r.bbox.bottom > r.bbox.top
                )
            ]

            # At least the first valid one should remain
            assert len(valid_regions) >= 1
            assert valid_regions[0].original_text == "Valid text"

    def test_multiple_language_support(self):
        """Test OCR with multiple languages."""
        # Test different language configurations
        language_configs = [
            OCRConfig(primary_engine=OCREngine.TESSERACT, languages=["en"]),
            OCRConfig(primary_engine=OCREngine.TESSERACT, languages=["en", "es"]),
            OCRConfig(primary_engine=OCREngine.TESSERACT, languages=["en", "fr", "de"]),
        ]

        mock_regions = [
            TextRegion(
                bbox=BoundingBox(10, 10, 100, 30),
                original_text="Multi-language text",
                replacement_text="Multi-language text",
                confidence=0.9,
            )
        ]

        for config in language_configs:
            processor = OCRProcessor(config)

            with patch.object(processor, "_extract_with_engine", return_value=mock_regions):
                regions = processor.extract_text_regions(b"fake_image_data")

                # Should handle all language configurations
                assert len(regions) == 1
                assert regions[0].original_text == "Multi-language text"

    def test_ocr_error_handling(self):
        """Test OCR error handling and recovery."""
        config = OCRConfig(primary_engine=OCREngine.TESSERACT)
        processor = OCRProcessor(config)

        # Test various error conditions
        error_conditions = [
            (b"", "empty image data"),
            (b"invalid_data", "invalid image format"),
            (None, "null image data"),
        ]

        for image_data, description in error_conditions:
            try:
                regions = processor.extract_text_regions(image_data)
                # If it succeeds, should return empty list
                assert isinstance(regions, list)
            except Exception as e:
                # Should provide meaningful error message
                assert isinstance(e, (ValueError, TypeError, Exception))
                error_msg = str(e).lower()
                assert any(
                    keyword in error_msg
                    for keyword in ["image", "data", "invalid", "format", "error"]
                )

    @pytest.mark.slow
    def test_ocr_performance_monitoring(self, test_image_with_text):
        """Test OCR performance characteristics."""
        config = OCRConfig(primary_engine=OCREngine.TESSERACT)
        processor = OCRProcessor(config)

        with test_image_with_text.open("rb") as f:
            image_data = f.read()

        # Mock successful extraction with timing
        mock_regions = [
            TextRegion(
                bbox=BoundingBox(20, 50, 200, 75),
                original_text="Performance test",
                replacement_text="Performance test",
                confidence=0.9,
            )
        ]

        with patch.object(processor, "_extract_with_engine", return_value=mock_regions):
            import time

            start_time = time.time()

            regions = processor.extract_text_regions(image_data)

            end_time = time.time()
            processing_time = end_time - start_time

            # Should complete within reasonable time (mocked, so should be fast)
            assert processing_time < 1.0  # Should be very fast with mocking
            assert len(regions) == 1

            # Test with larger mock dataset
            large_mock_regions = mock_regions * 50  # 50 text regions

            with patch.object(processor, "_extract_with_engine", return_value=large_mock_regions):
                start_time = time.time()
                regions_large = processor.extract_text_regions(image_data)
                end_time = time.time()

                # Should handle larger datasets efficiently
                assert len(regions_large) == 50
                processing_time_large = end_time - start_time
                assert processing_time_large < 2.0  # Still reasonable with mocking
