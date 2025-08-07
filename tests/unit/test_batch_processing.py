"""Tests for Batch Processing System
=================================

Unit tests for batch processing functionality including progress tracking,
memory management, and parallel processing.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.anonymizer.batch.processor import (
    BatchProcessor,
    BatchProgressCallback,
    BatchProgressInfo,
    ConsoleProgressCallback,
    create_batch_from_directory,
)
from src.anonymizer.core.exceptions import DuplicateItemError, EmptyBatchError, NoImageFilesError
from src.anonymizer.core.models import (
    AnonymizationResult,
    BatchAnonymizationRequest,
    BatchAnonymizationResult,
    BatchItem,
    BatchItemResult,
    BoundingBox,
    TextRegion,
)


class TestBatchProgressInfo:
    """Test BatchProgressInfo class."""

    def test_progress_percentage(self):
        """Test progress percentage calculation."""
        progress = BatchProgressInfo(total_items=100, completed_items=25, failed_items=5)
        assert progress.progress_percentage == 25.0

        # Edge case: no items
        progress_empty = BatchProgressInfo(total_items=0, completed_items=0, failed_items=0)
        assert progress_empty.progress_percentage == 100.0

    def test_success_rate(self):
        """Test success rate calculation."""
        progress = BatchProgressInfo(total_items=100, completed_items=20, failed_items=3)
        assert progress.success_rate == 85.0  # (20-3)/20 * 100

        # Edge case: no completed items
        progress_empty = BatchProgressInfo(total_items=100, completed_items=0, failed_items=0)
        assert progress_empty.success_rate == 100.0


class TestBatchProgressCallback:
    """Test BatchProgressCallback classes."""

    def test_base_callback_methods(self):
        """Test base callback methods."""
        callback = BatchProgressCallback()

        # Should not raise errors
        callback.on_start(10)
        callback.on_item_start("item_1", 0)
        callback.on_item_complete("item_1", True, 100.0)
        callback.on_progress(BatchProgressInfo(10, 1, 0))
        callback.on_error("item_1", Exception("test"))

        # Mock result for completion test
        result = Mock(spec=BatchAnonymizationResult)
        callback.on_complete(result)

    def test_console_callback(self, capsys):
        """Test console progress callback."""
        callback = ConsoleProgressCallback(update_interval=0.1)

        callback.on_start(5)
        captured = capsys.readouterr()
        assert "Starting batch processing of 5 items" in captured.out

        callback.on_item_start("test_item", 0)
        captured = capsys.readouterr()
        assert "Processing item 1: test_item" in captured.out

        callback.on_item_complete("test_item", True, 123.5)
        captured = capsys.readouterr()
        assert "✓ test_item - 123.5ms" in captured.out

        # Test progress update
        progress = BatchProgressInfo(
            total_items=5,
            completed_items=2,
            failed_items=0,
            estimated_remaining_ms=3000.0,
            current_memory_mb=256.0,
        )

        time.sleep(0.2)  # Ensure update interval passes
        callback.on_progress(progress)
        captured = capsys.readouterr()
        assert "Progress: 2/5" in captured.out
        assert "40.0%" in captured.out


class TestBatchModels:
    """Test batch processing models."""

    def test_batch_item_creation(self):
        """Test BatchItem creation and validation."""
        text_region = TextRegion(
            bbox=BoundingBox(left=10, top=20, right=100, bottom=80),
            original_text="Test text",
            replacement_text="[REDACTED]",
            confidence=0.95,
        )

        item = BatchItem(
            item_id="test_item_1",
            image_path=Path("/path/to/image.jpg"),
            text_regions=[text_region],
            output_path=Path("/path/to/output.jpg"),
            preserve_formatting=True,
            quality_check=True,
        )

        assert item.item_id == "test_item_1"
        assert len(item.text_regions) == 1
        assert item.preserve_formatting is True

    def test_batch_request_validation(self):
        """Test BatchAnonymizationRequest validation."""
        item = BatchItem(
            item_id="test_item",
            image_path=Path("/test.jpg"),
            text_regions=[],
            preserve_formatting=True,
            quality_check=True,
        )

        request = BatchAnonymizationRequest(
            items=[item],
            output_directory=Path("/output"),
            preserve_structure=True,
            max_parallel=4,
            batch_size=8,
            continue_on_error=True,
        )

        assert len(request.items) == 1
        assert request.max_parallel == 4
        assert request.batch_size == 8

    def test_batch_result_properties(self):
        """Test BatchAnonymizationResult properties."""
        result1 = BatchItemResult(
            item_id="item_1",
            success=True,
            output_path=Path("/output/item_1.jpg"),
            processing_time_ms=150.0,
            generated_patches=[],
            errors=[],
        )

        result2 = BatchItemResult(
            item_id="item_2",
            success=False,
            output_path=None,
            processing_time_ms=50.0,
            generated_patches=[],
            errors=["Processing failed"],
        )

        batch_result = BatchAnonymizationResult(
            results=[result1, result2],
            total_items=2,
            successful_items=1,
            failed_items=1,
            total_processing_time_ms=200.0,
            output_directory=Path("/output"),
        )

        assert batch_result.success_rate == 50.0
        assert len(batch_result.get_successful_items()) == 1
        assert len(batch_result.get_failed_items()) == 1


class TestBatchProcessor:
    """Test BatchProcessor class."""

    @pytest.fixture
    def mock_inference_engine(self):
        """Mock inference engine."""
        engine = Mock()

        # Mock successful anonymization result
        mock_result = AnonymizationResult(
            anonymized_image=np.zeros((100, 100, 3), dtype=np.uint8),
            generated_patches=[],
            processing_time_ms=100.0,
            success=True,
            errors=[],
        )
        engine.anonymize.return_value = mock_result

        return engine

    @pytest.fixture
    def temp_files(self):
        """Create temporary files for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock image files
            image_files = []
            for i in range(3):
                image_file = temp_path / f"test_image_{i}.jpg"
                image_file.write_bytes(b"fake image data")
                image_files.append(image_file)

            output_dir = temp_path / "output"
            output_dir.mkdir()

            yield temp_path, image_files, output_dir

    def test_processor_initialization(self, mock_inference_engine):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor(
            inference_engine=mock_inference_engine,
            max_memory_mb=2048.0,
            cleanup_interval=5,
        )

        assert processor.inference_engine == mock_inference_engine
        assert processor.max_memory_mb == 2048.0
        assert processor.cleanup_interval == 5

    def test_request_validation(self, mock_inference_engine, temp_files):
        """Test batch request validation."""
        temp_path, image_files, output_dir = temp_files
        processor = BatchProcessor(mock_inference_engine)

        # Test empty items
        empty_request = BatchAnonymizationRequest(
            items=[],
            output_directory=output_dir,
            preserve_structure=True,
            max_parallel=1,
            batch_size=1,
            continue_on_error=True,
        )

        with pytest.raises(EmptyBatchError, match="No items to process"):
            processor.process_batch(empty_request)

        # Test duplicate item IDs
        items = [
            BatchItem(
                item_id="duplicate",
                image_path=image_files[0],
                text_regions=[],
            ),
            BatchItem(
                item_id="duplicate",
                image_path=image_files[1],
                text_regions=[],
            ),
        ]

        duplicate_request = BatchAnonymizationRequest(
            items=items,
            output_directory=output_dir,
            preserve_structure=True,
            max_parallel=1,
            batch_size=1,
            continue_on_error=True,
        )

        with pytest.raises(DuplicateItemError, match="Duplicate item IDs found"):
            processor.process_batch(duplicate_request)

    def test_single_item_processing(self, mock_inference_engine, temp_files):
        """Test processing of a single item."""
        temp_path, image_files, output_dir = temp_files
        processor = BatchProcessor(mock_inference_engine)

        # Create text region to ensure inference engine is called
        text_region = TextRegion(
            bbox=BoundingBox(left=10, top=20, right=100, bottom=80),
            original_text="Test text",
            replacement_text="[REDACTED]",
            confidence=0.95,
        )

        item = BatchItem(
            item_id="test_item",
            image_path=image_files[0],
            text_regions=[text_region],
            preserve_formatting=True,
            quality_check=True,
        )

        request = BatchAnonymizationRequest(
            items=[item],
            output_directory=output_dir,
            preserve_structure=False,
            max_parallel=1,
            batch_size=1,
            continue_on_error=True,
        )

        # Mock PIL Image save and shutil - need to mock where Image is imported
        with patch("src.anonymizer.batch.processor.Image") as mock_image, patch("shutil.copy2"):
            mock_pil_image = Mock()
            mock_image.fromarray.return_value = mock_pil_image

            result = processor.process_batch(request)

            assert result.total_items == 1
            assert result.successful_items == 1
            assert result.failed_items == 0
            assert result.success_rate == 100.0

            # Verify inference engine was called
            mock_inference_engine.anonymize.assert_called_once()

            # Verify image was saved
            mock_pil_image.save.assert_called_once()

    def test_parallel_processing(self, mock_inference_engine, temp_files):
        """Test parallel processing of multiple items."""
        temp_path, image_files, output_dir = temp_files
        processor = BatchProcessor(mock_inference_engine)

        items = [
            BatchItem(
                item_id=f"item_{i}",
                image_path=image_files[i],
                text_regions=[],
            )
            for i in range(len(image_files))
        ]

        request = BatchAnonymizationRequest(
            items=items,
            output_directory=output_dir,
            preserve_structure=False,
            max_parallel=2,  # Use parallel processing
            batch_size=2,
            continue_on_error=True,
        )

        with patch("PIL.Image"), patch("shutil.copy2"):
            result = processor.process_batch(request)

            assert result.total_items == 3
            assert result.successful_items == 3
            assert result.failed_items == 0

            # Verify all items were processed
            assert len(result.results) == 3

    def test_error_handling(self, temp_files):
        """Test error handling in batch processing."""
        temp_path, image_files, output_dir = temp_files

        # Mock failing inference engine
        failing_engine = Mock()
        failing_engine.anonymize.side_effect = Exception("Processing failed")

        processor = BatchProcessor(failing_engine)

        # Create text region to ensure processing happens (not skipped for empty regions)
        text_region = TextRegion(
            bbox=BoundingBox(left=10, top=20, right=100, bottom=80),
            original_text="Test text",
            replacement_text="[REDACTED]",
            confidence=0.95,
        )

        item = BatchItem(
            item_id="failing_item",
            image_path=image_files[0],
            text_regions=[text_region],  # Add text region to trigger processing
        )

        request = BatchAnonymizationRequest(
            items=[item],
            output_directory=output_dir,
            preserve_structure=False,
            max_parallel=1,
            batch_size=1,
            continue_on_error=True,
        )

        result = processor.process_batch(request)

        assert result.total_items == 1
        assert result.successful_items == 0
        assert result.failed_items == 1
        assert result.success_rate == 0.0

        failed_item = result.get_failed_items()[0]
        assert failed_item.item_id == "failing_item"
        assert "Processing failed" in failed_item.errors

    def test_memory_monitoring(self, mock_inference_engine):
        """Test memory monitoring functionality."""
        processor = BatchProcessor(mock_inference_engine, max_memory_mb=1024.0)

        # Test memory usage method
        with patch("psutil.Process") as mock_process_class:
            mock_process = Mock()
            mock_process.memory_info.return_value.rss = 512 * 1024 * 1024  # 512MB
            mock_process_class.return_value = mock_process

            memory_mb = processor._get_memory_usage()
            assert memory_mb == 512.0

        # Test memory cleanup
        with patch("gc.collect") as mock_gc_collect:
            processor._cleanup_memory()
            mock_gc_collect.assert_called_once()

    def test_output_path_generation(self, mock_inference_engine, temp_files):
        """Test output path generation."""
        temp_path, image_files, output_dir = temp_files
        processor = BatchProcessor(mock_inference_engine)

        # Test with preserve_structure=True
        item = BatchItem(
            item_id="test_item",
            image_path=image_files[0],
            text_regions=[],
        )

        request = BatchAnonymizationRequest(
            items=[item],
            output_directory=output_dir,
            preserve_structure=True,
            max_parallel=1,
            batch_size=1,
            continue_on_error=True,
        )

        output_path = processor._get_output_path(item, request)
        assert output_path.parent == output_dir
        assert output_path.name == image_files[0].name


class TestBatchDirectoryUtility:
    """Test create_batch_from_directory utility function."""

    def test_create_batch_from_directory(self):
        """Test creating batch request from directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock image files
            image_files = []
            for i, ext in enumerate(["jpg", "png", "tiff"]):
                image_file = temp_path / f"test_{i}.{ext}"
                image_file.write_bytes(b"fake image data")
                image_files.append(image_file)

            # Create subdirectory with more files
            sub_dir = temp_path / "subdir"
            sub_dir.mkdir()
            sub_file = sub_dir / "test_sub.jpg"
            sub_file.write_bytes(b"fake image data")

            output_dir = temp_path / "output"

            # Create batch request
            request = create_batch_from_directory(
                input_dir=temp_path,
                output_dir=output_dir,
                preserve_structure=True,
                max_parallel=2,
                batch_size=4,
            )

            assert len(request.items) == 4  # 3 in root + 1 in subdir
            assert request.output_directory == output_dir
            assert request.preserve_structure is True
            assert request.max_parallel == 2
            assert request.batch_size == 4

            # Check item IDs are unique
            item_ids = [item.item_id for item in request.items]
            assert len(item_ids) == len(set(item_ids))

    def test_empty_directory(self):
        """Test with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "output"

            with pytest.raises(NoImageFilesError, match="No image files found"):
                create_batch_from_directory(
                    input_dir=temp_path,
                    output_dir=output_dir,
                )


class TestBatchIntegration:
    """Integration tests for batch processing."""

    def test_end_to_end_batch_processing(self):
        """Test complete end-to-end batch processing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create input directory with test images
            input_dir = temp_path / "input"
            input_dir.mkdir()

            for i in range(2):
                image_file = input_dir / f"test_{i}.jpg"
                image_file.write_bytes(b"fake image data")

            output_dir = temp_path / "output"

            # Mock inference engine
            mock_engine = Mock()
            mock_result = AnonymizationResult(
                anonymized_image=np.zeros((50, 50, 3), dtype=np.uint8),
                generated_patches=[],
                processing_time_ms=75.0,
                success=True,
                errors=[],
            )
            mock_engine.anonymize.return_value = mock_result

            # Create and run batch processor
            processor = BatchProcessor(mock_engine)
            request = create_batch_from_directory(
                input_dir=input_dir,
                output_dir=output_dir,
                max_parallel=1,
                batch_size=1,
            )

            # Capture progress updates
            progress_updates = []

            class TestCallback(BatchProgressCallback):
                def on_progress(self, progress):
                    progress_updates.append(progress)

            callback = TestCallback()

            with patch("PIL.Image"), patch("shutil.copy2"):
                result = processor.process_batch(request, callback)

                assert result.total_items == 2
                assert result.successful_items == 2
                assert result.failed_items == 0
                assert result.success_rate == 100.0

                # Verify progress updates were received
                assert len(progress_updates) >= 1

                # Verify output directory was created
                assert output_dir.exists()
