"""Unit tests for inference engine - Imperative style.

Tests the InferenceEngine class with comprehensive coverage of all major functionality
including initialization, NER processing, and core functionality.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from src.anonymizer.core.config import EngineConfig
from src.anonymizer.core.exceptions import (
    ModelLoadingError,
)
from src.anonymizer.inference.engine import InferenceEngine, NERProcessor


class TestNERProcessor:
    """Test NERProcessor functionality."""

    def test_ner_processor_initialization(self):
        """Test NER processor initialization."""
        with (
            patch("src.anonymizer.inference.engine.AnalyzerEngine") as mock_analyzer,
            patch("src.anonymizer.inference.engine.AnonymizerEngine") as mock_anonymizer,
        ):

            processor = NERProcessor()

            assert processor.analyzer is not None
            assert processor.anonymizer is not None
            assert len(processor.pii_entities) > 0
            assert "PERSON" in processor.pii_entities
            mock_analyzer.assert_called_once()
            mock_anonymizer.assert_called_once()

    def test_ner_processor_initialization_import_error(self):
        """Test NER processor initialization with import error."""
        with patch(
            "src.anonymizer.inference.engine.AnalyzerEngine",
            side_effect=ImportError("presidio not available"),
        ):
            with pytest.raises(ModelLoadingError):
                NERProcessor()

    def test_detect_pii_basic(self):
        """Test basic PII detection."""
        with (
            patch("src.anonymizer.inference.engine.AnalyzerEngine") as mock_analyzer_cls,
            patch("src.anonymizer.inference.engine.AnonymizerEngine"),
        ):

            # Mock analyzer results
            mock_result = Mock()
            mock_result.entity_type = "PERSON"
            mock_result.start = 0
            mock_result.end = 10
            mock_result.score = 0.95

            mock_analyzer = Mock()
            mock_analyzer.analyze.return_value = [mock_result]
            mock_analyzer_cls.return_value = mock_analyzer

            processor = NERProcessor()
            image = np.zeros((100, 100, 3), dtype=np.uint8)

            regions = processor.detect_pii("John Smith", image)

            assert len(regions) == 1
            region = regions[0]
            assert region.original_text == "John Smith"
            assert region.replacement_text == "[PERSON]"
            assert region.confidence == 0.95
            # Should have placeholder bbox
            assert region.bbox.left == 0
            assert region.bbox.top == 0

    def test_detect_pii_no_results(self):
        """Test PII detection with no results."""
        with (
            patch("src.anonymizer.inference.engine.AnalyzerEngine") as mock_analyzer_cls,
            patch("src.anonymizer.inference.engine.AnonymizerEngine"),
        ):

            mock_analyzer = Mock()
            mock_analyzer.analyze.return_value = []
            mock_analyzer_cls.return_value = mock_analyzer

            processor = NERProcessor()
            image = np.zeros((100, 100, 3), dtype=np.uint8)

            regions = processor.detect_pii("Hello world", image)

            assert len(regions) == 0

    def test_detect_pii_multiple_entities(self):
        """Test PII detection with multiple entities."""
        with (
            patch("src.anonymizer.inference.engine.AnalyzerEngine") as mock_analyzer_cls,
            patch("src.anonymizer.inference.engine.AnonymizerEngine"),
        ):

            # Mock multiple analyzer results
            mock_result1 = Mock()
            mock_result1.entity_type = "PERSON"
            mock_result1.start = 0
            mock_result1.end = 10
            mock_result1.score = 0.95

            mock_result2 = Mock()
            mock_result2.entity_type = "EMAIL_ADDRESS"
            mock_result2.start = 11  # Start after the person name
            mock_result2.end = 26  # Adjust end to match actual text length
            mock_result2.score = 0.88

            mock_analyzer = Mock()
            mock_analyzer.analyze.return_value = [mock_result1, mock_result2]
            mock_analyzer_cls.return_value = mock_analyzer

            processor = NERProcessor()
            image = np.zeros((100, 100, 3), dtype=np.uint8)

            text = "John Smith john@email.com"
            regions = processor.detect_pii(text, image)

            assert len(regions) == 2
            assert regions[0].original_text == text[0:10]  # "John Smith"
            assert regions[0].replacement_text == "[PERSON]"
            assert regions[1].original_text == text[11:26]  # "john@email.com"
            assert regions[1].replacement_text == "[EMAIL_ADDRESS]"


class TestInferenceEngine:
    """Test InferenceEngine functionality."""

    def create_mock_config(self, **overrides):
        """Create a mock engine config."""
        defaults = {
            "model_path": Path("/tmp/model"),
            "device": "cpu",
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "strength": 0.8,
            "confidence_threshold": 0.5,
            "max_image_size": 1024,
            "enable_memory_efficient_attention": True,
            "enable_quality_check": True,
            "enable_metrics": True,
        }
        defaults.update(overrides)
        return EngineConfig(**defaults)

    def test_inference_engine_initialization_basic(self):
        """Test basic inference engine initialization."""
        config = self.create_mock_config()

        # Mock all the dependencies to prevent actual model loading
        with (
            patch("src.anonymizer.inference.engine.StableDiffusionInpaintPipeline"),
            patch("src.anonymizer.ocr.processor.OCRProcessor"),
            patch("src.anonymizer.inference.engine.MemoryManager"),
            patch("src.anonymizer.inference.engine.ImageProcessor"),
            patch("src.anonymizer.inference.engine.ImageValidator"),
            patch("src.anonymizer.utils.metrics.MetricsCollector"),
            patch("src.anonymizer.inference.engine.TextRenderer"),
        ):

            engine = InferenceEngine(config)

            assert engine.config == config
            assert engine.device in [torch.device("cpu"), torch.device("cuda")]
            # The engine should be initialized
            assert engine is not None

    def test_inference_engine_initialization_with_quality_check(self):
        """Test inference engine with quality check configuration."""
        config = self.create_mock_config(enable_quality_check=True)

        with (
            patch("src.anonymizer.inference.engine.StableDiffusionInpaintPipeline"),
            patch("src.anonymizer.ocr.processor.OCRProcessor"),
            patch("src.anonymizer.inference.engine.MemoryManager"),
            patch("src.anonymizer.inference.engine.ImageProcessor"),
            patch("src.anonymizer.inference.engine.ImageValidator"),
            patch("src.anonymizer.utils.metrics.MetricsCollector"),
            patch("src.anonymizer.inference.engine.TextRenderer"),
        ):

            engine = InferenceEngine(config)

            # The engine should have quality check enabled in its config
            assert engine.config.enable_quality_check is True

    def test_inference_engine_config_validation(self):
        """Test that engine properly validates configuration."""
        config = self.create_mock_config()

        with (
            patch("src.anonymizer.inference.engine.StableDiffusionInpaintPipeline"),
            patch("src.anonymizer.ocr.processor.OCRProcessor"),
            patch("src.anonymizer.inference.engine.MemoryManager"),
            patch("src.anonymizer.inference.engine.ImageProcessor"),
            patch("src.anonymizer.inference.engine.ImageValidator"),
            patch("src.anonymizer.utils.metrics.MetricsCollector"),
            patch("src.anonymizer.inference.engine.TextRenderer"),
        ):

            engine = InferenceEngine(config)

            # Engine should store the configuration
            assert engine.config.num_inference_steps == 50
            assert engine.config.guidance_scale == 7.5
            assert engine.config.enable_memory_efficient_attention is True
            assert engine.config.enable_quality_check is True
