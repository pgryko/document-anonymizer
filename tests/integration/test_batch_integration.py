"""Comprehensive batch processing integration tests."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from src.anonymizer.batch.processor import (
    BatchProcessor,
    ConsoleProgressCallback,
    create_batch_from_directory,
)
from src.anonymizer.core.models import (
    AnonymizationResult,
    BatchAnonymizationRequest,
    BatchItem,
    BoundingBox,
    TextRegion,
)
from src.anonymizer.inference.engine import InferenceEngine


class TestBatchProcessingIntegration:
    """Test batch processing with real components integration."""

    @pytest.fixture
    def test_images(self):
        """Create test images for integration testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create real test images
            image_files = []
            for i in range(3):
                # Create a simple test image
                test_image = Image.new("RGB", (200, 100), color="white")
                image_path = temp_path / f"test_image_{i}.png"
                test_image.save(image_path)
                image_files.append(image_path)

            output_dir = temp_path / "output"
            output_dir.mkdir()

            yield temp_path, image_files, output_dir

    def test_batch_processing_with_mock_engine(self, test_images):
        """Test complete batch processing workflow with mocked inference engine."""
        temp_path, image_files, output_dir = test_images

        # Create mock inference engine with realistic response
        mock_engine = Mock(spec=InferenceEngine)
        mock_result = AnonymizationResult(
            anonymized_image=np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8),
            generated_patches=[],
            processing_time_ms=150.0,
            success=True,
            errors=[],
        )
        mock_engine.anonymize.return_value = mock_result

        # Create processor
        processor = BatchProcessor(mock_engine)

        # Create batch request
        text_region = TextRegion(
            bbox=BoundingBox(left=10, top=20, right=100, bottom=80),
            original_text="Sample Text",
            replacement_text="[REDACTED]",
            confidence=0.95,
        )

        items = [
            BatchItem(
                item_id=f"item_{i}",
                image_path=image_files[i],
                text_regions=[text_region],
                preserve_formatting=True,
                quality_check=True,
            )
            for i in range(len(image_files))
        ]

        request = BatchAnonymizationRequest(
            items=items,
            output_directory=output_dir,
            preserve_structure=False,
            max_parallel=2,
            batch_size=2,
            continue_on_error=True,
        )

        # Process batch
        result = processor.process_batch(request)

        # Verify results
        assert result.total_items == 3
        assert result.successful_items == 3
        assert result.failed_items == 0
        assert result.success_rate == 100.0
        assert result.total_processing_time_ms > 0

        # Verify output files were created
        for item in items:
            expected_output = output_dir / item.image_path.name
            assert expected_output.exists()

        # Verify inference engine was called for each item
        assert mock_engine.anonymize.call_count == 3

    def test_batch_processing_with_ocr_fallback(self, test_images):
        """Test batch processing with OCR fallback when no text regions provided."""
        temp_path, image_files, output_dir = test_images

        # Mock inference engine (should not be called since no text regions found)
        mock_engine = Mock(spec=InferenceEngine)
        processor = BatchProcessor(mock_engine)

        # Create items without text regions (will trigger OCR)
        items = [
            BatchItem(
                item_id=f"item_{i}",
                image_path=image_files[i],
                text_regions=[],  # Empty - will trigger OCR
                preserve_formatting=True,
                quality_check=True,
            )
            for i in range(len(image_files))
        ]

        request = BatchAnonymizationRequest(
            items=items,
            output_directory=output_dir,
            preserve_structure=False,
            max_parallel=1,
            batch_size=1,
            continue_on_error=True,
        )

        # Mock OCR to return empty results (simulating no text found)
        with patch("src.anonymizer.batch.processor.OCRProcessor") as mock_ocr_class:
            mock_ocr = Mock()
            mock_ocr.extract_text_regions.return_value = []  # No text found
            mock_ocr_class.return_value = mock_ocr

            # Process batch
            result = processor.process_batch(request)

        # Verify results - should succeed by copying original images
        assert result.total_items == 3
        assert result.successful_items == 3
        assert result.failed_items == 0

        # Verify files were copied (not processed through inference engine)
        for item in items:
            expected_output = output_dir / item.image_path.name
            assert expected_output.exists()

        # Inference engine should not be called since no text regions found
        mock_engine.anonymize.assert_not_called()

    def test_batch_processing_with_errors(self, test_images):
        """Test batch processing error handling."""
        temp_path, image_files, output_dir = test_images

        # Create mock engine that fails for second item
        mock_engine = Mock(spec=InferenceEngine)

        def side_effect_anonymize(image_data, text_regions):
            # Fail on second call
            if mock_engine.anonymize.call_count == 2:
                raise RuntimeError("Simulated processing error")

            return AnonymizationResult(
                anonymized_image=np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8),
                generated_patches=[],
                processing_time_ms=100.0,
                success=True,
                errors=[],
            )

        mock_engine.anonymize.side_effect = side_effect_anonymize

        processor = BatchProcessor(mock_engine)

        # Create batch items
        text_region = TextRegion(
            bbox=BoundingBox(left=10, top=20, right=100, bottom=80),
            original_text="Sample Text",
            replacement_text="[REDACTED]",
            confidence=0.95,
        )

        items = [
            BatchItem(
                item_id=f"item_{i}",
                image_path=image_files[i],
                text_regions=[text_region],
                preserve_formatting=True,
                quality_check=True,
            )
            for i in range(len(image_files))
        ]

        request = BatchAnonymizationRequest(
            items=items,
            output_directory=output_dir,
            preserve_structure=False,
            max_parallel=1,  # Sequential to ensure predictable error
            batch_size=1,
            continue_on_error=True,  # Continue processing despite errors
        )

        # Process batch
        result = processor.process_batch(request)

        # Verify partial success
        assert result.total_items == 3
        assert result.successful_items == 2  # One failed
        assert result.failed_items == 1
        assert result.success_rate == pytest.approx(66.7, abs=0.1)

    def test_create_batch_from_directory(self, test_images):
        """Test creating batch request from directory."""
        temp_path, image_files, output_dir = test_images

        # Create some additional files to test filtering
        (temp_path / "text_file.txt").write_text("Not an image")
        (temp_path / "another_image.jpg").write_bytes(b"fake jpg data")

        # Create batch from directory
        request = create_batch_from_directory(
            input_dir=temp_path,
            output_dir=output_dir,
            preserve_structure=True,
            max_parallel=2,
            batch_size=4,
        )

        # Verify request structure
        assert len(request.items) >= 3  # At least our test images
        assert request.output_directory == output_dir
        assert request.preserve_structure is True
        assert request.max_parallel == 2
        assert request.batch_size == 4

        # Verify all items are image files
        for item in request.items:
            assert item.image_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".tiff", ".pdf"]
            assert item.image_path.exists()

    def test_console_progress_callback_integration(self, test_images, capsys):
        """Test console progress callback during actual processing."""
        temp_path, image_files, output_dir = test_images

        # Mock simple engine
        mock_engine = Mock(spec=InferenceEngine)
        mock_result = AnonymizationResult(
            anonymized_image=np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8),
            generated_patches=[],
            processing_time_ms=50.0,
            success=True,
            errors=[],
        )
        mock_engine.anonymize.return_value = mock_result

        processor = BatchProcessor(mock_engine)
        callback = ConsoleProgressCallback(update_interval=0.1)

        # Create simple batch
        text_region = TextRegion(
            bbox=BoundingBox(left=10, top=20, right=100, bottom=80),
            original_text="Test",
            replacement_text="[REDACTED]",
            confidence=0.9,
        )

        items = [
            BatchItem(
                item_id=f"test_{i}",
                image_path=image_files[i],
                text_regions=[text_region],
                preserve_formatting=True,
                quality_check=True,
            )
            for i in range(2)  # Just 2 items for faster test
        ]

        request = BatchAnonymizationRequest(
            items=items,
            output_directory=output_dir,
            preserve_structure=False,
            max_parallel=1,
            batch_size=1,
            continue_on_error=True,
        )

        # Process with console callback
        result = processor.process_batch(request, progress_callback=callback)

        # Capture console output
        captured = capsys.readouterr()

        # Verify progress messages appeared
        assert "Starting batch processing" in captured.out
        assert "Processing item" in captured.out
        assert "Batch processing complete" in captured.out
        assert "Total items: 2" in captured.out
        assert "Success rate:" in captured.out

        # Verify processing succeeded
        assert result.successful_items == 2
        assert result.failed_items == 0
