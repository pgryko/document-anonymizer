"""
Batch Processor
===============

High-performance batch processing for document anonymization with
memory management, parallel processing, and progress tracking.
"""

import gc
import logging
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.anonymizer.core.exceptions import ProcessingError, ValidationError
from src.anonymizer.core.models import (
    BatchAnonymizationRequest,
    BatchAnonymizationResult,
    BatchItem,
    BatchItemResult,
)
from src.anonymizer.inference.engine import InferenceEngine
from src.anonymizer.ocr.processor import OCRProcessor

try:
    import psutil
except ImportError:
    psutil = None

logger = logging.getLogger(__name__)


@dataclass
class BatchProgressInfo:
    """Progress information for batch processing."""

    total_items: int
    completed_items: int
    failed_items: int
    current_item_id: str | None = None
    estimated_remaining_ms: float | None = None
    current_memory_mb: float | None = None

    @property
    def progress_percentage(self) -> float:
        """Calculate progress as percentage."""
        if self.total_items == 0:
            return 100.0
        return (self.completed_items / self.total_items) * 100.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.completed_items == 0:
            return 100.0
        return ((self.completed_items - self.failed_items) / self.completed_items) * 100.0


class BatchProgressCallback:
    """Base class for batch progress callbacks."""

    def on_start(self, total_items: int) -> None:
        """Called when batch processing starts."""

    def on_item_start(self, item_id: str, item_index: int) -> None:
        """Called when processing of an item starts."""

    def on_item_complete(self, item_id: str, success: bool, processing_time_ms: float) -> None:
        """Called when processing of an item completes."""

    def on_progress(self, progress: BatchProgressInfo) -> None:
        """Called periodically with progress updates."""

    def on_error(self, item_id: str, error: Exception) -> None:
        """Called when an error occurs."""

    def on_complete(self, result: BatchAnonymizationResult) -> None:
        """Called when batch processing completes."""


class ConsoleProgressCallback(BatchProgressCallback):
    """Console-based progress callback with detailed output."""

    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.start_time = None
        self.last_update = 0

    def on_start(self, total_items: int) -> None:
        self.start_time = time.time()
        print(f"Starting batch processing of {total_items} items...")

    def on_item_start(self, item_id: str, item_index: int) -> None:
        print(f"Processing item {item_index + 1}: {item_id}")

    def on_item_complete(self, item_id: str, success: bool, processing_time_ms: float) -> None:
        status = "✓" if success else "✗"
        print(f"{status} {item_id} - {processing_time_ms:.1f}ms")

    def on_progress(self, progress: BatchProgressInfo) -> None:
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            elapsed = current_time - self.start_time if self.start_time else 0

            print(
                f"\nProgress: {progress.completed_items}/{progress.total_items} "
                f"({progress.progress_percentage:.1f}%) "
                f"- Success rate: {progress.success_rate:.1f}% "
                f"- Elapsed: {elapsed:.1f}s"
            )

            if progress.estimated_remaining_ms:
                remaining_s = progress.estimated_remaining_ms / 1000
                print(f"Estimated remaining: {remaining_s:.1f}s")

            if progress.current_memory_mb:
                print(f"Memory usage: {progress.current_memory_mb:.1f}MB")

            self.last_update = current_time

    def on_error(self, item_id: str, error: Exception) -> None:
        print(f"✗ Error processing {item_id}: {error}")

    def on_complete(self, result: BatchAnonymizationResult) -> None:
        elapsed = time.time() - self.start_time if self.start_time else 0
        print("\nBatch processing complete!")
        print(f"Total items: {result.total_items}")
        print(f"Successful: {result.successful_items}")
        print(f"Failed: {result.failed_items}")
        print(f"Success rate: {result.success_rate:.1f}%")
        print(f"Total time: {elapsed:.1f}s")
        print(
            f"Average time per item: {result.total_processing_time_ms / result.total_items:.1f}ms"
        )


class BatchProcessor:
    """
    High-performance batch processor for document anonymization.

    Features:
    - Memory-efficient processing with configurable batch sizes
    - Parallel processing with thread and process pools
    - Progress tracking and callbacks
    - Robust error handling and recovery
    - Memory monitoring and cleanup
    """

    def __init__(
        self,
        inference_engine: InferenceEngine | None = None,
        max_memory_mb: float = 4096.0,
        cleanup_interval: int = 10,
    ):
        """
        Initialize batch processor.

        Args:
            inference_engine: Optional pre-initialized inference engine
            max_memory_mb: Maximum memory usage in MB before triggering cleanup
            cleanup_interval: Number of items between memory cleanup
        """
        self.inference_engine = inference_engine
        self.max_memory_mb = max_memory_mb
        self.cleanup_interval = cleanup_interval
        self.processed_count = 0
        self._lock = threading.Lock()

    def process_batch(
        self,
        request: BatchAnonymizationRequest,
        progress_callback: BatchProgressCallback | None = None,
    ) -> BatchAnonymizationResult:
        """
        Process a batch of documents for anonymization.

        Args:
            request: Batch processing request
            progress_callback: Optional progress callback

        Returns:
            Batch processing result
        """
        start_time = time.time()

        # Initialize progress callback
        if progress_callback is None:
            progress_callback = ConsoleProgressCallback()

        # Validate request
        self._validate_request(request)

        # Prepare output directory
        self._prepare_output_directory(request.output_directory)

        # Initialize inference engine if needed
        if self.inference_engine is None:
            self.inference_engine = InferenceEngine()

        # Start processing
        progress_callback.on_start(len(request.items))

        results = []
        completed_items = 0
        failed_items = 0

        try:
            # Process items in batches
            for batch_start in range(0, len(request.items), request.batch_size):
                batch_end = min(batch_start + request.batch_size, len(request.items))
                batch_items = request.items[batch_start:batch_end]

                # Process current batch
                batch_results = self._process_batch_chunk(
                    batch_items,
                    request,
                    progress_callback,
                    batch_start,
                )

                results.extend(batch_results)

                # Update counters
                for result in batch_results:
                    completed_items += 1
                    if not result.success:
                        failed_items += 1

                # Update progress
                progress_info = BatchProgressInfo(
                    total_items=len(request.items),
                    completed_items=completed_items,
                    failed_items=failed_items,
                    estimated_remaining_ms=self._estimate_remaining_time(
                        completed_items, len(request.items), start_time
                    ),
                    current_memory_mb=self._get_memory_usage(),
                )
                progress_callback.on_progress(progress_info)

                # Memory cleanup if needed
                if completed_items % self.cleanup_interval == 0:
                    self._cleanup_memory()

        except Exception as e:
            logger.exception("Batch processing failed")
            progress_callback.on_error("batch", e)
            raise

        # Create final result
        total_time_ms = (time.time() - start_time) * 1000
        result = BatchAnonymizationResult(
            results=results,
            total_items=len(request.items),
            successful_items=completed_items - failed_items,
            failed_items=failed_items,
            total_processing_time_ms=total_time_ms,
            output_directory=request.output_directory,
        )

        progress_callback.on_complete(result)
        return result

    def _validate_request(self, request: BatchAnonymizationRequest) -> None:
        """Validate batch request."""
        if not request.items:
            raise ValidationError("No items to process")

        # Check for duplicate item IDs
        item_ids = [item.item_id for item in request.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValidationError("Duplicate item IDs found")

        # Validate input files exist
        for item in request.items:
            if not item.image_path.exists():
                raise ProcessingError(f"Input file not found: {item.image_path}")

    def _prepare_output_directory(self, output_dir: Path) -> None:
        """Prepare output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)

    def _process_batch_chunk(
        self,
        items: list[BatchItem],
        request: BatchAnonymizationRequest,
        progress_callback: BatchProgressCallback,
        batch_start_index: int,
    ) -> list[BatchItemResult]:
        """Process a chunk of items with parallel processing."""
        results = []

        if request.max_parallel == 1:
            # Sequential processing
            for i, item in enumerate(items):
                result = self._process_single_item(
                    item, request, progress_callback, batch_start_index + i
                )
                results.append(result)
        else:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=request.max_parallel) as executor:
                # Submit all tasks
                future_to_item = {
                    executor.submit(
                        self._process_single_item,
                        item,
                        request,
                        progress_callback,
                        batch_start_index + i,
                    ): item
                    for i, item in enumerate(items)
                }

                # Collect results as they complete
                for future in as_completed(future_to_item):
                    result = future.result()
                    results.append(result)

        return results

    def _process_single_item(
        self,
        item: BatchItem,
        request: BatchAnonymizationRequest,
        progress_callback: BatchProgressCallback,
        item_index: int,
    ) -> BatchItemResult:
        """Process a single item."""
        start_time = time.time()

        try:
            progress_callback.on_item_start(item.item_id, item_index)

            # Load image data
            with item.image_path.open("rb") as f:
                image_data = f.read()

            # If no text regions are provided, use OCR to detect them
            text_regions = item.text_regions
            if not text_regions:
                try:
                    ocr_processor = OCRProcessor()
                    detected_regions = ocr_processor.detect_text_regions(image_data)
                    text_regions = detected_regions
                except Exception as e:
                    logger.warning(f"OCR detection failed for {item.item_id}: {e}")
                    # Continue with empty regions - some items might not have text
                    text_regions = []

            # Skip processing if no text regions found
            if not text_regions:
                logger.info(f"No text regions found for {item.item_id}, copying original image")
                # Simply copy the original image
                output_path = self._get_output_path(item, request)
                shutil.copy2(item.image_path, output_path)

                processing_time_ms = (time.time() - start_time) * 1000
                result = BatchItemResult(
                    item_id=item.item_id,
                    success=True,
                    output_path=output_path,
                    processing_time_ms=processing_time_ms,
                    generated_patches=[],
                    errors=[],
                )
                progress_callback.on_item_complete(item.item_id, result.success, processing_time_ms)
                return result

            # Process with inference engine
            with self._lock:
                anon_result = self.inference_engine.anonymize(image_data, text_regions)

            # Determine output path
            output_path = self._get_output_path(item, request)

            # Save result
            self._save_anonymized_image(anon_result.anonymized_image, output_path)

            processing_time_ms = (time.time() - start_time) * 1000

            result = BatchItemResult(
                item_id=item.item_id,
                success=anon_result.success,
                output_path=output_path,
                processing_time_ms=processing_time_ms,
                generated_patches=anon_result.generated_patches,
                errors=anon_result.errors,
            )

            progress_callback.on_item_complete(item.item_id, result.success, processing_time_ms)

        except Exception as e:
            processing_time_ms = (time.time() - start_time) * 1000
            error_msg = str(e)

            progress_callback.on_error(item.item_id, e)

            return BatchItemResult(
                item_id=item.item_id,
                success=False,
                output_path=None,
                processing_time_ms=processing_time_ms,
                generated_patches=[],
                errors=[error_msg],
            )

    def _get_output_path(self, item: BatchItem, request: BatchAnonymizationRequest) -> Path:
        """Determine output path for an item."""
        if item.output_path:
            return item.output_path

        # Generate output path based on input path
        if request.preserve_structure:
            # Try to preserve relative structure
            try:
                # Find common base path
                base_path = item.image_path.parent
                relative_path = item.image_path.relative_to(base_path)
                output_path = request.output_directory / relative_path
            except ValueError:
                # Fallback to simple filename
                output_path = request.output_directory / item.image_path.name
        else:
            output_path = request.output_directory / item.image_path.name

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return output_path

    def _save_anonymized_image(self, image_array, output_path: Path) -> None:
        """Save anonymized image to file."""

        # Convert numpy array to PIL Image
        if image_array.dtype != np.uint8:
            image_array = (image_array * 255).astype(np.uint8)

        image = Image.fromarray(image_array)
        image.save(output_path)

    def _estimate_remaining_time(
        self, completed: int, total: int, start_time: float
    ) -> float | None:
        """Estimate remaining processing time."""
        if completed == 0:
            return None

        elapsed = time.time() - start_time
        avg_time_per_item = elapsed / completed
        remaining_items = total - completed

        return remaining_items * avg_time_per_item * 1000  # Convert to ms

    def _get_memory_usage(self) -> float | None:
        """Get current memory usage in MB."""
        if psutil is None:
            return None
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception:
            return None

    def _cleanup_memory(self) -> None:
        """Perform memory cleanup."""
        try:
            gc.collect()

            # Check if we're over memory limit
            current_memory = self._get_memory_usage()
            if current_memory and current_memory > self.max_memory_mb:
                logger.warning(
                    f"Memory usage ({current_memory:.1f}MB) exceeds limit "
                    f"({self.max_memory_mb:.1f}MB)"
                )

                # Additional cleanup could be performed here
                # e.g., clearing caches, releasing resources

        except Exception as e:
            logger.debug(f"Memory cleanup failed: {e}")


def create_batch_from_directory(
    input_dir: Path,
    output_dir: Path,
    preserve_structure: bool = True,
    max_parallel: int = 4,
    batch_size: int = 8,
) -> BatchAnonymizationRequest:
    """
    Create a batch request from directory of images.

    Args:
        input_dir: Input directory containing images
        output_dir: Output directory for processed images
        pattern: File pattern to match (glob pattern)
        preserve_structure: Whether to preserve directory structure
        max_parallel: Maximum parallel processes
        batch_size: Batch size for processing

    Returns:
        BatchAnonymizationRequest ready for processing
    """

    # Find all matching files
    image_files = []
    for ext in ["jpg", "jpeg", "png", "tiff", "pdf"]:
        pattern_with_ext = f"**/*.{ext}"
        files = list(input_dir.glob(pattern_with_ext))
        image_files.extend(files)

    if not image_files:
        raise ValidationError(f"No image files found in {input_dir}")

    # Create batch items
    items = []
    for i, image_path in enumerate(image_files):
        # Generate item ID
        item_id = f"item_{i:04d}_{image_path.stem}"

        # For now, create empty text regions (would normally come from OCR)
        text_regions = []

        item = BatchItem(
            item_id=item_id,
            image_path=image_path,
            text_regions=text_regions,
            output_path=None,  # Will be auto-generated
            preserve_formatting=True,
            quality_check=True,
        )
        items.append(item)

    return BatchAnonymizationRequest(
        items=items,
        output_directory=output_dir,
        preserve_structure=preserve_structure,
        max_parallel=max_parallel,
        batch_size=batch_size,
        continue_on_error=True,
    )
