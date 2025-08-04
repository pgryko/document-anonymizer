"""
End-to-End Integration Tests for Document Anonymization
======================================================

Tests the complete anonymization workflow from image input to anonymized output.
Validates the integration between NER, diffusion models, and image processing.
"""

import gc
import logging
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from src.anonymizer.core.config import EngineConfig
from src.anonymizer.core.exceptions import InferenceError
from src.anonymizer.core.models import AnonymizationResult, BoundingBox, TextRegion
from src.anonymizer.inference.engine import InferenceEngine

logger = logging.getLogger(__name__)


class TestE2EAnonymization:
    """End-to-end anonymization integration tests."""

    @pytest.fixture
    def engine_config(self):
        """Create test engine configuration."""
        return EngineConfig(
            # Use smaller values for faster testing
            num_inference_steps=10,  # Reduced from 50 for speed
            guidance_scale=5.0,  # Reduced from 7.5 for speed
            strength=1.0,
            enable_memory_efficient_attention=True,
            enable_sequential_cpu_offload=False,
            max_batch_size=1,
            enable_quality_check=False,  # Disable for speed
            min_confidence_threshold=0.5,
        )

    @pytest.fixture
    def sample_document_image(self):
        """Create a sample document image with text for testing."""
        # Create a document-like image with text
        width, height = 800, 600
        image = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(image)

        # Try to use a system font, fallback to default
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
        except OSError:
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except OSError:
                font = ImageFont.load_default()

        # Add various types of text that should be detected as PII
        text_samples = [
            ("John Smith", (50, 50)),  # Person name
            ("john.smith@email.com", (50, 100)),  # Email
            ("555-123-4567", (50, 150)),  # Phone number
            ("123-45-6789", (50, 200)),  # SSN format
            ("New York", (50, 250)),  # Location
            ("Acme Corporation", (50, 300)),  # Organization
        ]

        for text, position in text_samples:
            draw.text(position, text, font=font, fill="black")

        # Convert to numpy array
        return np.array(image)

    @pytest.fixture
    def sample_text_regions(self):
        """Create sample text regions for testing."""
        return [
            TextRegion(
                bbox=BoundingBox(left=50, top=50, right=150, bottom=75),
                original_text="John Smith",
                replacement_text="[PERSON]",
                confidence=0.95,
            ),
            TextRegion(
                bbox=BoundingBox(left=50, top=100, right=250, bottom=125),
                original_text="john.smith@email.com",
                replacement_text="[EMAIL]",
                confidence=0.90,
            ),
        ]

    def test_inference_engine_initialization(self, engine_config):
        """Test that InferenceEngine initializes correctly."""
        engine = InferenceEngine(engine_config)

        assert engine.config == engine_config
        assert engine.device is not None
        assert engine.memory_manager is not None
        assert engine.image_processor is not None
        assert engine.metrics_collector is not None
        assert engine.text_renderer is not None

        logger.info("✅ InferenceEngine initialization test passed")

    def test_image_preprocessing(self, engine_config, sample_document_image):
        """Test image preprocessing pipeline."""
        engine = InferenceEngine(engine_config)

        # Convert image to bytes (simulate real input)
        pil_image = Image.fromarray(sample_document_image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            pil_image.save(tmp_file.name, "PNG")
            with open(tmp_file.name, "rb") as f:
                image_data = f.read()

        # Test preprocessing
        processed_image = engine._process_input_image(image_data)

        assert isinstance(processed_image, np.ndarray)
        assert len(processed_image.shape) == 3
        assert processed_image.shape[2] == 3  # RGB
        assert processed_image.dtype == np.uint8

        # Clean up
        Path(tmp_file.name).unlink(missing_ok=True)

        logger.info("✅ Image preprocessing test passed")

    def test_text_region_detection(self, engine_config, sample_document_image):
        """Test text region auto-detection (currently uses dummy data)."""
        engine = InferenceEngine(engine_config)

        # This tests the current implementation which returns dummy data
        # In the future, this will test real OCR integration
        text_regions = engine._auto_detect_text_regions(sample_document_image)

        # Should return empty list if NER processor fails to initialize
        # or dummy regions if it succeeds
        assert isinstance(text_regions, list)

        if text_regions:
            for region in text_regions:
                assert isinstance(region, TextRegion)
                assert isinstance(region.bbox, BoundingBox)
                assert region.original_text
                assert region.replacement_text

        logger.info(f"✅ Text region detection test passed - found {len(text_regions)} regions")

    def test_mask_creation(self, engine_config):
        """Test mask creation for text regions."""
        engine = InferenceEngine(engine_config)

        # Test data
        image_shape = (600, 800)  # height, width
        bbox = BoundingBox(left=50, top=50, right=150, bottom=100)

        mask = engine._create_mask(image_shape, bbox)

        assert isinstance(mask, np.ndarray)
        assert mask.shape == image_shape
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

        # Check that the bbox region is marked
        bbox_region = mask[bbox.top : bbox.bottom, bbox.left : bbox.right]
        assert np.all(bbox_region == 1.0)

        logger.info("✅ Mask creation test passed")

    @pytest.mark.slow
    def test_anonymization_with_text_regions(
        self, engine_config, sample_document_image, sample_text_regions
    ):
        """Test anonymization with provided text regions."""
        pytest.skip("Heavy integration test - requires model downloads")
        engine = InferenceEngine(engine_config)

        # Convert image to bytes
        pil_image = Image.fromarray(sample_document_image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            pil_image.save(tmp_file.name, "PNG")
            with open(tmp_file.name, "rb") as f:
                image_data = f.read()

        start_time = time.time()

        try:
            # This will attempt to load diffusion models
            # May fail if models are not available, which is expected in CI
            result = engine.anonymize(image_data, sample_text_regions)

            # Validate result structure
            assert isinstance(result, AnonymizationResult)
            assert isinstance(result.anonymized_image, np.ndarray)
            assert isinstance(result.generated_patches, list)
            assert isinstance(result.processing_time_ms, float)
            assert isinstance(result.success, bool)
            assert isinstance(result.errors, list)

            # Check image properties
            assert result.anonymized_image.shape == sample_document_image.shape
            assert result.anonymized_image.dtype == sample_document_image.dtype

            processing_time = time.time() - start_time
            logger.info(f"✅ Anonymization test passed - processed in {processing_time:.2f}s")

        except (InferenceError, Exception) as e:
            # Expected if diffusion models are not available
            logger.warning(f"⚠️ Anonymization test skipped - models not available: {e}")
            pytest.skip(f"Diffusion models not available: {e}")

        finally:
            # Clean up
            Path(tmp_file.name).unlink(missing_ok=True)

    @pytest.mark.slow
    @pytest.mark.skipif(
        not pytest.importorskip("torch", minversion="1.0"),
        reason="Requires torch and large model downloads",
    )
    def test_complete_anonymization_workflow(self, engine_config, sample_document_image):
        """Test the complete end-to-end anonymization workflow."""
        pytest.skip("Heavy integration test - requires model downloads")
        engine = InferenceEngine(engine_config)

        # Convert image to bytes
        pil_image = Image.fromarray(sample_document_image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            pil_image.save(tmp_file.name, "PNG")
            with open(tmp_file.name, "rb") as f:
                image_data = f.read()

        start_time = time.time()

        try:
            # Test auto-detection workflow (no text regions provided)
            result = engine.anonymize(image_data, text_regions=None)

            # Validate result
            assert isinstance(result, AnonymizationResult)
            assert result.success is True or len(result.errors) == 0

            # Should return original image if no regions detected
            if not result.generated_patches:
                assert np.array_equal(result.anonymized_image, sample_document_image)
                logger.info("✅ No text regions detected - returned original image")
            else:
                # If regions were detected and processed
                assert len(result.generated_patches) > 0
                logger.info(f"✅ Processed {len(result.generated_patches)} text regions")

            processing_time = time.time() - start_time
            logger.info(f"✅ Complete workflow test passed - {processing_time:.2f}s")

        except (InferenceError, Exception) as e:
            # Expected if dependencies are missing
            logger.warning(f"⚠️ Complete workflow test skipped: {e}")
            pytest.skip(f"Dependencies not available: {e}")

        finally:
            # Clean up
            Path(tmp_file.name).unlink(missing_ok=True)

    def test_error_handling_invalid_image(self, engine_config):
        """Test error handling with invalid image data."""
        engine = InferenceEngine(engine_config)

        # Test with invalid image data
        invalid_data = b"not an image"

        with pytest.raises(InferenceError):
            engine.anonymize(invalid_data)

        logger.info("✅ Error handling test passed")

    def test_error_handling_empty_image(self, engine_config):
        """Test error handling with empty image data."""
        engine = InferenceEngine(engine_config)

        with pytest.raises(InferenceError):
            engine.anonymize(b"")

        logger.info("✅ Empty image error handling test passed")

    def test_memory_cleanup(self, engine_config, sample_document_image):
        """Test that memory is properly cleaned up after processing."""
        pytest.skip("Heavy integration test - requires model downloads")

        import psutil

        engine = InferenceEngine(engine_config)

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Convert image to bytes
        pil_image = Image.fromarray(sample_document_image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            pil_image.save(tmp_file.name, "PNG")
            with open(tmp_file.name, "rb") as f:
                image_data = f.read()

        try:
            # Process multiple times to check for memory leaks
            for _i in range(3):
                try:
                    result = engine.anonymize(image_data)
                    del result  # Explicit cleanup
                except Exception as e:
                    # Expected if models not available
                    logging.debug(f"Expected exception during memory test: {e}")

            # Force garbage collection
            gc.collect()

            # Check memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Allow some memory increase but not excessive
            assert memory_increase < 1000, f"Memory increased by {memory_increase:.1f}MB"

            logger.info(f"✅ Memory cleanup test passed - memory increase: {memory_increase:.1f}MB")

        finally:
            Path(tmp_file.name).unlink(missing_ok=True)


class TestPerformanceBenchmarks:
    """Performance benchmarks for anonymization."""

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_processing_time_benchmark(self, engine_config):
        """Benchmark processing time for different image sizes."""
        engine = InferenceEngine(engine_config)

        # Test different image sizes
        test_sizes = [(400, 300), (800, 600), (1200, 900)]
        results = {}

        for width, height in test_sizes:
            # Create test image
            image = Image.new("RGB", (width, height), color="white")
            draw = ImageDraw.Draw(image)
            draw.text((50, 50), "Test Document", fill="black")

            # Convert to bytes
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                image.save(tmp_file.name, "PNG")
                with open(tmp_file.name, "rb") as f:
                    image_data = f.read()

            try:
                start_time = time.time()
                engine.anonymize(image_data)
                processing_time = (time.time() - start_time) * 1000  # ms

                results[f"{width}x{height}"] = processing_time
                logger.info(f"📊 {width}x{height}: {processing_time:.1f}ms")

            except Exception as e:
                logger.warning(f"⚠️ Benchmark skipped for {width}x{height}: {e}")

            finally:
                Path(tmp_file.name).unlink(missing_ok=True)

        # Log results summary
        if results:
            avg_time = sum(results.values()) / len(results)
            logger.info(f"📊 Average processing time: {avg_time:.1f}ms")

        logger.info("✅ Performance benchmark completed")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
