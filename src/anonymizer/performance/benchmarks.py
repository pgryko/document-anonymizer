"""
Benchmark Suite
===============

Standardized benchmarks for the document anonymization pipeline.
"""

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.anonymizer.core.models import BoundingBox

from .profiler import PerformanceProfiler

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    benchmark_name: str
    timestamp: str
    duration_ms: float
    peak_memory_mb: float
    avg_memory_mb: float
    cpu_percent: float
    success: bool
    error_message: str | None = None
    additional_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ModelBenchmark:
    """
    Benchmarks for individual model components.

    Tests loading time, inference speed, and memory usage
    for VAE, UNet, and text encoder models.
    """

    def __init__(self, profiler: PerformanceProfiler | None = None):
        self.profiler = profiler or PerformanceProfiler()

    def benchmark_model_loading(self, model_path: Path) -> BenchmarkResult:
        """Benchmark model loading time and memory usage."""
        benchmark_name = f"model_loading_{model_path.name}"

        try:
            with self.profiler.profile_operation(benchmark_name):
                # Simulate model loading (actual implementation would load real models)
                time.sleep(0.1)  # Placeholder for actual loading

                # In real implementation, this would:
                # - Load the model from path
                # - Initialize the model on device
                # - Perform warmup inference if needed

                success = True
                error_message = None

        except Exception as e:
            success = False
            error_message = str(e)
            logger.exception(f"Model loading benchmark failed: {e}")

        # Get the latest metrics
        if not self.profiler.metrics:
            return BenchmarkResult(
                benchmark_name=benchmark_name,
                timestamp="",
                duration_ms=0,
                peak_memory_mb=0,
                avg_memory_mb=0,
                cpu_percent=0,
                success=False,
                error_message="No metrics collected",
            )

        metrics = self.profiler.metrics[-1]
        if not metrics:
            return BenchmarkResult(
                benchmark_name=benchmark_name,
                timestamp="",
                duration_ms=0,
                peak_memory_mb=0,
                avg_memory_mb=0,
                cpu_percent=0,
                success=False,
                error_message="No metrics collected",
            )

        return BenchmarkResult(
            benchmark_name=benchmark_name,
            timestamp=metrics.timestamp,
            duration_ms=metrics.duration_ms,
            peak_memory_mb=metrics.peak_memory_mb,
            avg_memory_mb=metrics.avg_memory_mb,
            cpu_percent=metrics.cpu_percent,
            success=success,
            error_message=error_message,
        )

    def benchmark_inference_speed(
        self,
        model_type: str,
        input_size: tuple[int, int] = (512, 512),
        num_iterations: int = 10,
    ) -> BenchmarkResult:
        """Benchmark model inference speed."""
        benchmark_name = f"{model_type}_inference_speed"

        try:
            with self.profiler.profile_operation(benchmark_name):
                # Simulate inference iterations
                for _i in range(num_iterations):
                    # In real implementation, this would:
                    # - Create input tensors of specified size
                    # - Run model inference
                    # - Measure per-iteration timing
                    time.sleep(0.05)  # Placeholder

            success = True
            error_message = None

            # Calculate additional metrics
            latest_metrics = self.profiler.metrics[-1]
            avg_iteration_time = latest_metrics.duration_ms / num_iterations

            additional_metrics = {
                "num_iterations": num_iterations,
                "avg_iteration_ms": avg_iteration_time,
                "input_size": input_size,
                "throughput_fps": (1000 / avg_iteration_time if avg_iteration_time > 0 else 0),
            }

        except Exception as e:
            success = False
            error_message = str(e)
            additional_metrics = None
            logger.exception(f"Inference speed benchmark failed: {e}")

        if not self.profiler.metrics:
            return BenchmarkResult(
                benchmark_name=benchmark_name,
                timestamp="",
                duration_ms=0,
                peak_memory_mb=0,
                avg_memory_mb=0,
                cpu_percent=0,
                success=False,
                error_message="No metrics collected",
            )

        metrics = self.profiler.metrics[-1]

        return BenchmarkResult(
            benchmark_name=benchmark_name,
            timestamp=metrics.timestamp,
            duration_ms=metrics.duration_ms,
            peak_memory_mb=metrics.peak_memory_mb,
            avg_memory_mb=metrics.avg_memory_mb,
            cpu_percent=metrics.cpu_percent,
            success=success,
            error_message=error_message,
            additional_metrics=additional_metrics,
        )


class AnonymizationBenchmark:
    """
    End-to-end benchmarks for the document anonymization pipeline.

    Tests complete workflow performance including:
    - Document loading and preprocessing
    - Text detection (OCR)
    - PII identification (NER)
    - Anonymization (inpainting)
    - Output generation
    """

    def __init__(self, profiler: PerformanceProfiler | None = None):
        self.profiler = profiler or PerformanceProfiler()

    def create_test_document(
        self, size: tuple[int, int] = (1024, 768), format: str = "RGB"
    ) -> Image.Image:
        """Create a synthetic test document image."""
        # Create a white background
        return Image.new(format, size, color="white")

        # In a real implementation, this could:
        # - Add realistic text content
        # - Include various fonts and layouts
        # - Add synthetic PII data for testing
        # - Include different document types (invoices, forms, etc.)


    def benchmark_document_loading(
        self, image_size: tuple[int, int] = (1024, 768), num_documents: int = 5
    ) -> BenchmarkResult:
        """Benchmark document loading and preprocessing."""
        benchmark_name = "document_loading"

        try:
            with self.profiler.profile_operation(benchmark_name):
                for _i in range(num_documents):
                    # Create and process test document
                    self.create_test_document(image_size)

                    # Simulate preprocessing steps
                    # - Image format conversion
                    # - Resizing/normalization
                    # - Tensor conversion
                    time.sleep(0.02)  # Placeholder

            success = True
            error_message = None

            # Additional metrics (after context manager exits and metrics are added)
            latest_metrics = self.profiler.metrics[-1]
            avg_doc_time = latest_metrics.duration_ms / num_documents

            additional_metrics = {
                "num_documents": num_documents,
                "avg_document_ms": avg_doc_time,
                "image_size": image_size,
                "documents_per_second": (1000 / avg_doc_time if avg_doc_time > 0 else 0),
            }

        except Exception as e:
            success = False
            error_message = str(e)
            additional_metrics = None
            logger.exception(f"Document loading benchmark failed: {e}")

        if not self.profiler.metrics:
            return BenchmarkResult(
                benchmark_name=benchmark_name,
                timestamp="",
                duration_ms=0,
                peak_memory_mb=0,
                avg_memory_mb=0,
                cpu_percent=0,
                success=False,
                error_message="No metrics collected",
            )

        metrics = self.profiler.metrics[-1]

        return BenchmarkResult(
            benchmark_name=benchmark_name,
            timestamp=metrics.timestamp,
            duration_ms=metrics.duration_ms,
            peak_memory_mb=metrics.peak_memory_mb,
            avg_memory_mb=metrics.avg_memory_mb,
            cpu_percent=metrics.cpu_percent,
            success=success,
            error_message=error_message,
            additional_metrics=additional_metrics,
        )

    def benchmark_text_detection(
        self, image_size: tuple[int, int] = (1024, 768), num_images: int = 10
    ) -> BenchmarkResult:
        """Benchmark OCR text detection performance."""
        benchmark_name = "text_detection_ocr"

        try:
            with self.profiler.profile_operation(benchmark_name):
                for _i in range(num_images):
                    # Create test image
                    self.create_test_document(image_size)

                    # Simulate OCR processing
                    # - Text region detection
                    # - Character recognition
                    # - Confidence scoring
                    time.sleep(0.1)  # Placeholder for OCR

            success = True
            error_message = None

            latest_metrics = self.profiler.metrics[-1]
            avg_ocr_time = latest_metrics.duration_ms / num_images

            additional_metrics = {
                "num_images": num_images,
                "avg_ocr_ms": avg_ocr_time,
                "image_size": image_size,
                "images_per_second": 1000 / avg_ocr_time if avg_ocr_time > 0 else 0,
            }

        except Exception as e:
            success = False
            error_message = str(e)
            additional_metrics = None
            logger.exception(f"Text detection benchmark failed: {e}")

        if not self.profiler.metrics:
            return BenchmarkResult(
                benchmark_name=benchmark_name,
                timestamp="",
                duration_ms=0,
                peak_memory_mb=0,
                avg_memory_mb=0,
                cpu_percent=0,
                success=False,
                error_message="No metrics collected",
            )

        metrics = self.profiler.metrics[-1]

        return BenchmarkResult(
            benchmark_name=benchmark_name,
            timestamp=metrics.timestamp,
            duration_ms=metrics.duration_ms,
            peak_memory_mb=metrics.peak_memory_mb,
            avg_memory_mb=metrics.avg_memory_mb,
            cpu_percent=metrics.cpu_percent,
            success=success,
            error_message=error_message,
            additional_metrics=additional_metrics,
        )

    def benchmark_pii_detection(self, num_texts: int = 100) -> BenchmarkResult:
        """Benchmark NER PII detection performance."""
        benchmark_name = "pii_detection_ner"

        # Sample texts with PII for testing
        test_texts = [
            "John Smith's email is john.smith@example.com",
            "Call me at 555-123-4567 for more information",
            "SSN: 123-45-6789",
            "Credit card: 4111-1111-1111-1111",
            "Address: 123 Main St, Anytown, CA 12345",
        ] * (
            num_texts // 5 + 1
        )  # Repeat to get desired count

        try:
            with self.profiler.profile_operation(benchmark_name):
                for i in range(num_texts):
                    test_texts[i % len(test_texts)]

                    # Simulate NER processing
                    # - Tokenization
                    # - Entity recognition
                    # - PII classification
                    time.sleep(0.01)  # Placeholder for NER

            success = True
            error_message = None

            latest_metrics = self.profiler.metrics[-1]
            avg_ner_time = latest_metrics.duration_ms / num_texts

            additional_metrics = {
                "num_texts": num_texts,
                "avg_ner_ms": avg_ner_time,
                "texts_per_second": 1000 / avg_ner_time if avg_ner_time > 0 else 0,
            }

        except Exception as e:
            success = False
            error_message = str(e)
            additional_metrics = None
            logger.exception(f"PII detection benchmark failed: {e}")

        if not self.profiler.metrics:
            return BenchmarkResult(
                benchmark_name=benchmark_name,
                timestamp="",
                duration_ms=0,
                peak_memory_mb=0,
                avg_memory_mb=0,
                cpu_percent=0,
                success=False,
                error_message="No metrics collected",
            )

        metrics = self.profiler.metrics[-1]

        return BenchmarkResult(
            benchmark_name=benchmark_name,
            timestamp=metrics.timestamp,
            duration_ms=metrics.duration_ms,
            peak_memory_mb=metrics.peak_memory_mb,
            avg_memory_mb=metrics.avg_memory_mb,
            cpu_percent=metrics.cpu_percent,
            success=success,
            error_message=error_message,
            additional_metrics=additional_metrics,
        )

    def benchmark_inpainting(
        self,
        image_size: tuple[int, int] = (512, 512),
        num_regions: int = 5,
        num_iterations: int = 3,
    ) -> BenchmarkResult:
        """Benchmark diffusion model inpainting performance."""
        benchmark_name = "diffusion_inpainting"

        try:
            with self.profiler.profile_operation(benchmark_name):
                for _i in range(num_iterations):
                    # Create test image and mask
                    self.create_test_document(image_size)

                    # Create random mask regions (simulating PII areas)
                    mask_regions = []
                    for _j in range(num_regions):
                        left = np.random.randint(0, image_size[0] - 100)
                        top = np.random.randint(0, image_size[1] - 50)
                        width = np.random.randint(50, 100)
                        height = np.random.randint(20, 50)
                        right = left + width
                        bottom = top + height
                        mask_regions.append(
                            BoundingBox(left=left, top=top, right=right, bottom=bottom)
                        )

                    # Simulate diffusion inpainting
                    # - Mask preparation
                    # - Noise scheduling
                    # - Denoising steps
                    # - Output generation
                    time.sleep(0.5)  # Placeholder for diffusion

            success = True
            error_message = None

            latest_metrics = self.profiler.metrics[-1]
            avg_inpaint_time = latest_metrics.duration_ms / num_iterations

            additional_metrics = {
                "num_iterations": num_iterations,
                "num_regions_per_image": num_regions,
                "avg_inpainting_ms": avg_inpaint_time,
                "image_size": image_size,
                "images_per_second": (1000 / avg_inpaint_time if avg_inpaint_time > 0 else 0),
            }

        except Exception as e:
            success = False
            error_message = str(e)
            additional_metrics = None
            logger.exception(f"Inpainting benchmark failed: {e}")

        if not self.profiler.metrics:
            return BenchmarkResult(
                benchmark_name=benchmark_name,
                timestamp="",
                duration_ms=0,
                peak_memory_mb=0,
                avg_memory_mb=0,
                cpu_percent=0,
                success=False,
                error_message="No metrics collected",
            )

        metrics = self.profiler.metrics[-1]

        return BenchmarkResult(
            benchmark_name=benchmark_name,
            timestamp=metrics.timestamp,
            duration_ms=metrics.duration_ms,
            peak_memory_mb=metrics.peak_memory_mb,
            avg_memory_mb=metrics.avg_memory_mb,
            cpu_percent=metrics.cpu_percent,
            success=success,
            error_message=error_message,
            additional_metrics=additional_metrics,
        )

    def benchmark_end_to_end(
        self, image_size: tuple[int, int] = (1024, 768), num_documents: int = 3
    ) -> BenchmarkResult:
        """Benchmark the complete end-to-end anonymization pipeline."""
        benchmark_name = "end_to_end_anonymization"

        try:
            with self.profiler.profile_operation(benchmark_name):
                for _i in range(num_documents):
                    # 1. Document loading
                    self.create_test_document(image_size)

                    # 2. Text detection (OCR)
                    time.sleep(0.1)  # Simulate OCR

                    # 3. PII detection (NER)
                    time.sleep(0.05)  # Simulate NER

                    # 4. Mask generation
                    time.sleep(0.02)  # Simulate mask creation

                    # 5. Inpainting
                    time.sleep(0.5)  # Simulate diffusion

                    # 6. Post-processing and output
                    time.sleep(0.03)  # Simulate final steps

            success = True
            error_message = None

            latest_metrics = self.profiler.metrics[-1]
            avg_e2e_time = latest_metrics.duration_ms / num_documents

            additional_metrics = {
                "num_documents": num_documents,
                "avg_e2e_ms": avg_e2e_time,
                "image_size": image_size,
                "documents_per_second": (1000 / avg_e2e_time if avg_e2e_time > 0 else 0),
                "pipeline_stages": [
                    "loading",
                    "ocr",
                    "ner",
                    "masking",
                    "inpainting",
                    "output",
                ],
            }

        except Exception as e:
            success = False
            error_message = str(e)
            additional_metrics = None
            logger.exception(f"End-to-end benchmark failed: {e}")

        if not self.profiler.metrics:
            return BenchmarkResult(
                benchmark_name=benchmark_name,
                timestamp="",
                duration_ms=0,
                peak_memory_mb=0,
                avg_memory_mb=0,
                cpu_percent=0,
                success=False,
                error_message="No metrics collected",
            )

        metrics = self.profiler.metrics[-1]

        return BenchmarkResult(
            benchmark_name=benchmark_name,
            timestamp=metrics.timestamp,
            duration_ms=metrics.duration_ms,
            peak_memory_mb=metrics.peak_memory_mb,
            avg_memory_mb=metrics.avg_memory_mb,
            cpu_percent=metrics.cpu_percent,
            success=success,
            error_message=error_message,
            additional_metrics=additional_metrics,
        )

    def run_full_benchmark_suite(self) -> list[BenchmarkResult]:
        """Run the complete benchmark suite."""
        logger.info("Starting full benchmark suite")

        results = []

        # Document processing benchmarks
        results.append(self.benchmark_document_loading())
        results.append(self.benchmark_text_detection())
        results.append(self.benchmark_pii_detection())
        results.append(self.benchmark_inpainting())
        results.append(self.benchmark_end_to_end())

        logger.info(f"Completed benchmark suite with {len(results)} tests")
        return results

    def save_benchmark_results(self, results: list[BenchmarkResult], filepath: Path) -> bool:
        """Save benchmark results to a JSON file."""
        try:
            data = {
                "benchmark_suite": "document_anonymization",
                "timestamp": time.time(),
                "results": [result.to_dict() for result in results],
            }

            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved benchmark results to {filepath}")
            return True

        except Exception as e:
            logger.exception(f"Failed to save benchmark results: {e}")
            return False
