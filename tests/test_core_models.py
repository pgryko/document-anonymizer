"""
Unit tests for core models - Imperative style.

Tests all critical data structures and validation logic.
"""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from src.anonymizer.core.models import (
    AnonymizationRequest,
    BoundingBox,
    CropData,
    GeneratedPatch,
    GenerationMetadata,
    ModelArtifacts,
    ProcessedImage,
    TextRegion,
    TrainingMetrics,
)


class TestBoundingBox:
    """Test BoundingBox model validation and operations."""

    def test_valid_bounding_box_creation(self):
        """Test creating valid bounding box."""
        bbox = BoundingBox(left=10, top=20, right=100, bottom=80)

        assert bbox.left == 10
        assert bbox.top == 20
        assert bbox.right == 100
        assert bbox.bottom == 80
        assert bbox.width == 90
        assert bbox.height == 60
        assert bbox.area == 5400

    def test_bounding_box_validation_right_greater_than_left(self):
        """Test that right must be greater than left."""
        with pytest.raises(ValidationError) as exc_info:
            BoundingBox(left=100, top=20, right=50, bottom=80)

        assert "right must be greater than left" in str(exc_info.value)

    def test_bounding_box_validation_bottom_greater_than_top(self):
        """Test that bottom must be greater than top."""
        with pytest.raises(ValidationError) as exc_info:
            BoundingBox(left=10, top=80, right=100, bottom=20)

        assert "bottom must be greater than top" in str(exc_info.value)

    def test_bounding_box_validation_negative_coordinates(self):
        """Test that coordinates cannot be negative."""
        with pytest.raises(ValidationError):
            BoundingBox(left=-10, top=20, right=100, bottom=80)

    def test_bounding_box_scale(self):
        """Test bounding box scaling."""
        bbox = BoundingBox(left=10, top=20, right=100, bottom=80)
        scaled = bbox.scale(2.0)

        assert scaled.left == 20
        assert scaled.top == 40
        assert scaled.right == 200
        assert scaled.bottom == 160

    def test_bounding_box_properties(self):
        """Test bounding box computed properties."""
        bbox = BoundingBox(left=0, top=0, right=50, bottom=30)

        assert bbox.width == 50
        assert bbox.height == 30
        assert bbox.area == 1500


class TestTextRegion:
    """Test TextRegion model validation."""

    def test_valid_text_region_creation(self, sample_bbox):
        """Test creating valid text region."""
        region = TextRegion(
            bbox=sample_bbox,
            original_text="John Doe",
            replacement_text="REDACTED",
            confidence=0.95,
        )

        assert region.bbox == sample_bbox
        assert region.original_text == "John Doe"
        assert region.replacement_text == "REDACTED"
        assert region.confidence == 0.95

    def test_text_region_validation_empty_text(self, sample_bbox):
        """Test that text cannot be empty."""
        with pytest.raises(ValidationError):
            TextRegion(
                bbox=sample_bbox,
                original_text="",
                replacement_text="REDACTED",
                confidence=0.95,
            )

    def test_text_region_validation_long_text(self, sample_bbox):
        """Test that text cannot be too long."""
        long_text = "x" * 1001  # Over the 1000 character limit

        with pytest.raises(ValidationError):
            TextRegion(
                bbox=sample_bbox,
                original_text=long_text,
                replacement_text="REDACTED",
                confidence=0.95,
            )

    def test_text_region_validation_confidence_range(self, sample_bbox):
        """Test that confidence must be in valid range."""
        # Test confidence > 1.0
        with pytest.raises(ValidationError):
            TextRegion(
                bbox=sample_bbox,
                original_text="test",
                replacement_text="REDACTED",
                confidence=1.5,
            )

        # Test confidence < 0.0
        with pytest.raises(ValidationError):
            TextRegion(
                bbox=sample_bbox,
                original_text="test",
                replacement_text="REDACTED",
                confidence=-0.1,
            )


class TestAnonymizationRequest:
    """Test AnonymizationRequest model validation."""

    def test_valid_anonymization_request(self, sample_anonymization_request):
        """Test creating valid anonymization request."""
        request = sample_anonymization_request

        assert len(request.image_data) > 0
        assert len(request.text_regions) == 1
        assert request.preserve_formatting is True
        assert request.quality_check is True

    def test_anonymization_request_validation_empty_regions(self, sample_image):
        """Test that text regions cannot be empty."""
        import io

        from PIL import Image

        pil_image = Image.fromarray(sample_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        with pytest.raises(ValidationError):
            AnonymizationRequest(
                image_data=image_data,
                text_regions=[],  # Empty list
                preserve_formatting=True,
                quality_check=True,
            )

    def test_anonymization_request_validation_too_many_regions(
        self, sample_image, sample_text_region
    ):
        """Test that too many text regions are rejected."""
        import io

        from PIL import Image

        pil_image = Image.fromarray(sample_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_data = buffer.getvalue()

        # Create 51 regions (over the 50 limit)
        too_many_regions = [sample_text_region] * 51

        with pytest.raises(ValidationError):
            AnonymizationRequest(
                image_data=image_data,
                text_regions=too_many_regions,
                preserve_formatting=True,
                quality_check=True,
            )


class TestProcessedImage:
    """Test ProcessedImage model."""

    def test_valid_processed_image(self, sample_processed_image):
        """Test creating valid processed image."""
        processed = sample_processed_image

        assert processed.crop.shape[2] == 3  # RGB
        assert processed.mask.ndim == 2  # 2D mask
        assert processed.scale_factor == 1.0
        assert isinstance(processed.original_bbox, BoundingBox)

    def test_processed_image_validation_scale_factor(self, sample_image, sample_bbox):
        """Test that scale factor must be positive."""
        crop = sample_image[
            sample_bbox.top : sample_bbox.bottom, sample_bbox.left : sample_bbox.right
        ]
        mask = np.ones((crop.shape[0], crop.shape[1]), dtype=np.float32)

        with pytest.raises(ValidationError):
            ProcessedImage(
                crop=crop,
                mask=mask,
                original_bbox=sample_bbox,
                scale_factor=0.0,  # Invalid scale factor
            )


class TestGenerationMetadata:
    """Test GenerationMetadata model."""

    def test_valid_generation_metadata(self):
        """Test creating valid generation metadata."""
        metadata = GenerationMetadata(
            processing_time_ms=150.5,
            model_version="v1.0",
            num_inference_steps=50,
            guidance_scale=7.5,
            seed=42,
        )

        assert metadata.processing_time_ms == 150.5
        assert metadata.model_version == "v1.0"
        assert metadata.num_inference_steps == 50
        assert metadata.guidance_scale == 7.5
        assert metadata.seed == 42

    def test_generation_metadata_validation_negative_time(self):
        """Test that processing time cannot be negative."""
        with pytest.raises(ValidationError):
            GenerationMetadata(
                processing_time_ms=-10.0,  # Negative time
                model_version="v1.0",
                num_inference_steps=50,
                guidance_scale=7.5,
            )

    def test_generation_metadata_validation_invalid_steps(self):
        """Test that inference steps must be positive."""
        with pytest.raises(ValidationError):
            GenerationMetadata(
                processing_time_ms=150.5,
                model_version="v1.0",
                num_inference_steps=0,  # Invalid steps
                guidance_scale=7.5,
            )

    def test_generation_metadata_validation_invalid_guidance(self):
        """Test that guidance scale must be positive."""
        with pytest.raises(ValidationError):
            GenerationMetadata(
                processing_time_ms=150.5,
                model_version="v1.0",
                num_inference_steps=50,
                guidance_scale=0.0,  # Invalid guidance scale
            )


class TestGeneratedPatch:
    """Test GeneratedPatch model."""

    def test_valid_generated_patch(self):
        """Test creating valid generated patch."""
        patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        metadata = GenerationMetadata(
            processing_time_ms=150.5,
            model_version="v1.0",
            num_inference_steps=50,
            guidance_scale=7.5,
        )

        generated_patch = GeneratedPatch(patch=patch, confidence=0.85, metadata=metadata)

        assert generated_patch.patch.shape == (64, 64, 3)
        assert generated_patch.confidence == 0.85
        assert generated_patch.metadata == metadata


class TestTrainingMetrics:
    """Test TrainingMetrics model."""

    def test_valid_training_metrics(self):
        """Test creating valid training metrics."""
        metrics = TrainingMetrics(
            epoch=5,
            step=1000,
            total_loss=0.5,
            recon_loss=0.3,
            kl_loss=0.1,
            perceptual_loss=0.1,
            learning_rate=1e-4,
        )

        assert metrics.epoch == 5
        assert metrics.step == 1000
        assert metrics.total_loss == 0.5
        assert metrics.recon_loss == 0.3
        assert metrics.kl_loss == 0.1
        assert metrics.perceptual_loss == 0.1
        assert metrics.learning_rate == 1e-4

    def test_training_metrics_validation_negative_epoch(self):
        """Test that epoch cannot be negative."""
        with pytest.raises(ValidationError):
            TrainingMetrics(
                epoch=-1,  # Negative epoch
                step=1000,
                total_loss=0.5,
                recon_loss=0.3,
                kl_loss=0.1,
                learning_rate=1e-4,
            )

    def test_training_metrics_validation_negative_step(self):
        """Test that step cannot be negative."""
        with pytest.raises(ValidationError):
            TrainingMetrics(
                epoch=5,
                step=-100,  # Negative step
                total_loss=0.5,
                recon_loss=0.3,
                kl_loss=0.1,
                learning_rate=1e-4,
            )

    def test_training_metrics_validation_zero_learning_rate(self):
        """Test that learning rate must be positive."""
        with pytest.raises(ValidationError):
            TrainingMetrics(
                epoch=5,
                step=1000,
                total_loss=0.5,
                recon_loss=0.3,
                kl_loss=0.1,
                learning_rate=0.0,  # Zero learning rate
            )

    def test_training_metrics_to_dict(self):
        """Test converting training metrics to dictionary."""
        metrics = TrainingMetrics(
            epoch=5,
            step=1000,
            total_loss=0.5,
            recon_loss=0.3,
            kl_loss=0.1,
            perceptual_loss=0.1,
            learning_rate=1e-4,
        )

        metrics_dict = metrics.to_dict()

        expected_keys = {
            "epoch",
            "step",
            "total_loss",
            "recon_loss",
            "kl_loss",
            "perceptual_loss",
            "learning_rate",
        }
        assert set(metrics_dict.keys()) == expected_keys
        assert metrics_dict["epoch"] == 5.0
        assert metrics_dict["step"] == 1000.0
        assert metrics_dict["perceptual_loss"] == 0.1

    def test_training_metrics_to_dict_optional_perceptual_loss(self):
        """Test to_dict with optional perceptual loss."""
        metrics = TrainingMetrics(
            epoch=5,
            step=1000,
            total_loss=0.5,
            recon_loss=0.3,
            kl_loss=0.1,
            perceptual_loss=None,  # No perceptual loss
            learning_rate=1e-4,
        )

        metrics_dict = metrics.to_dict()
        assert "perceptual_loss" not in metrics_dict


class TestModelArtifacts:
    """Test ModelArtifacts model."""

    def test_valid_model_artifacts(self, temp_dir):
        """Test creating valid model artifacts."""
        model_path = temp_dir / "model.safetensors"
        config_path = temp_dir / "config.json"

        # Create dummy files
        model_path.touch()
        config_path.touch()

        artifacts = ModelArtifacts(
            model_name="test-model",
            version="v1.0",
            model_path=model_path,
            config_path=config_path,
            metadata={"training_steps": 1000},
        )

        assert artifacts.model_name == "test-model"
        assert artifacts.version == "v1.0"
        assert artifacts.model_path == model_path
        assert artifacts.config_path == config_path
        assert artifacts.metadata["training_steps"] == 1000

    def test_model_artifacts_validation_empty_name(self, temp_dir):
        """Test that model name cannot be empty."""
        model_path = temp_dir / "model.safetensors"
        config_path = temp_dir / "config.json"

        with pytest.raises(ValidationError):
            ModelArtifacts(
                model_name="",  # Empty name
                version="v1.0",
                model_path=model_path,
                config_path=config_path,
            )

    def test_model_artifacts_validation_empty_version(self, temp_dir):
        """Test that version cannot be empty."""
        model_path = temp_dir / "model.safetensors"
        config_path = temp_dir / "config.json"

        with pytest.raises(ValidationError):
            ModelArtifacts(
                model_name="test-model",
                version="",  # Empty version
                model_path=model_path,
                config_path=config_path,
            )

    def test_model_artifacts_to_dict(self, temp_dir):
        """Test converting model artifacts to dictionary."""
        model_path = temp_dir / "model.safetensors"
        config_path = temp_dir / "config.json"

        artifacts = ModelArtifacts(
            model_name="test-model",
            version="v1.0",
            model_path=model_path,
            config_path=config_path,
            metadata={"training_steps": 1000},
        )

        artifacts_dict = artifacts.to_dict()

        expected_keys = {
            "model_name",
            "version",
            "model_path",
            "config_path",
            "metadata",
        }
        assert set(artifacts_dict.keys()) == expected_keys
        assert artifacts_dict["model_name"] == "test-model"
        assert artifacts_dict["version"] == "v1.0"
        assert artifacts_dict["model_path"] == str(model_path)
        assert artifacts_dict["config_path"] == str(config_path)
        assert artifacts_dict["metadata"]["training_steps"] == 1000

    def test_model_artifacts_from_cache(self, temp_dir):
        """Test loading model artifacts from cache."""
        # Create cache structure
        model_path = temp_dir / "model.safetensors"
        config_path = temp_dir / "config.json"
        metadata_path = temp_dir / "metadata.json"

        model_path.touch()
        config_path.touch()

        # Create metadata file
        metadata = {
            "model_name": "cached-model",
            "version": "v2.0",
            "training_steps": 2000,
        }
        with metadata_path.open("w") as f:
            json.dump(metadata, f)

        # Load from cache
        artifacts = ModelArtifacts.from_cache(temp_dir)

        assert artifacts.model_name == "cached-model"
        assert artifacts.version == "v2.0"
        assert artifacts.metadata["training_steps"] == 2000
        assert artifacts.model_path == model_path
        assert artifacts.config_path == config_path

    def test_model_artifacts_from_cache_no_metadata(self, temp_dir):
        """Test loading model artifacts from cache without metadata."""
        # Create cache structure without metadata
        model_path = temp_dir / "model.safetensors"
        config_path = temp_dir / "config.json"

        model_path.touch()
        config_path.touch()

        # Load from cache
        artifacts = ModelArtifacts.from_cache(temp_dir)

        assert artifacts.model_name == "unknown"
        assert artifacts.version == "unknown"
        assert artifacts.metadata == {}


class TestCropData:
    """Test CropData model."""

    def test_valid_crop_data(self, sample_image, sample_bbox):
        """Test creating valid crop data."""
        crop = sample_image[
            sample_bbox.top : sample_bbox.bottom, sample_bbox.left : sample_bbox.right
        ]
        relative_bbox = BoundingBox(left=0, top=0, right=crop.shape[1], bottom=crop.shape[0])

        crop_data = CropData(crop=crop, scale_factor=2.0, relative_bbox=relative_bbox)

        assert crop_data.crop.shape[2] == 3
        assert crop_data.scale_factor == 2.0
        assert crop_data.relative_bbox == relative_bbox

    def test_crop_data_validation_scale_factor(self, sample_image, sample_bbox):
        """Test that scale factor must be positive."""
        crop = sample_image[
            sample_bbox.top : sample_bbox.bottom, sample_bbox.left : sample_bbox.right
        ]
        relative_bbox = BoundingBox(left=0, top=0, right=crop.shape[1], bottom=crop.shape[0])

        with pytest.raises(ValidationError):
            CropData(
                crop=crop,
                scale_factor=0.0,  # Invalid scale factor
                relative_bbox=relative_bbox,
            )
