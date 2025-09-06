"""Unit tests for datasets module - Imperative style.

Tests the dataset loading and validation functionality with comprehensive coverage
of ImageValidator, TextRegionValidator, and core dataset operations.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.anonymizer.core.config import DatasetConfig
from src.anonymizer.core.exceptions import (
    EmptyDatasetError,
    ImageDimensionsTooLargeError,
    ImageDimensionsTooSmallError,
    ImageLoadFailedError,
    ImageTooLargeError,
    ImageValidationFailedError,
    MissingImageNameError,
    NoAnnotationFilesError,
    UnexpectedImageDtypeError,
    UnexpectedImageShapeError,
    UnsupportedImageFormatError,
)
from src.anonymizer.core.models import BoundingBox, TextRegion
from src.anonymizer.training.datasets import (
    AnonymizerDataset,
    ImageValidator,
    SafeAugmentation,
    TextRegionValidator,
    _validate_dataset_not_empty,
    _validate_file_size,
    _validate_image_array_properties,
    _validate_image_dimensions,
    _validate_image_format,
    _validate_image_name,
)


class TestValidationHelpers:
    """Test validation helper functions."""

    def test_validate_file_size_valid(self):
        """Test file size validation with valid size."""
        # Should not raise exception
        _validate_file_size(1000, 2000)

    def test_validate_file_size_too_large(self):
        """Test file size validation with file too large."""
        with pytest.raises(ImageTooLargeError):
            _validate_file_size(3000, 2000)

    def test_validate_image_format_valid(self):
        """Test image format validation with valid format."""
        # Should not raise exception
        _validate_image_format("JPEG", {"JPEG", "PNG"})

    def test_validate_image_format_invalid(self):
        """Test image format validation with invalid format."""
        with pytest.raises(UnsupportedImageFormatError):
            _validate_image_format("WEBP", {"JPEG", "PNG"})

    def test_validate_image_dimensions_valid(self):
        """Test image dimension validation with valid dimensions."""
        # Should not raise exception
        _validate_image_dimensions(1024, 768, 2000, 100)

    def test_validate_image_dimensions_too_large(self):
        """Test image dimension validation with dimensions too large."""
        with pytest.raises(ImageDimensionsTooLargeError):
            _validate_image_dimensions(3000, 2000, 2048, 100)

    def test_validate_image_dimensions_too_small(self):
        """Test image dimension validation with dimensions too small."""
        with pytest.raises(ImageDimensionsTooSmallError):
            _validate_image_dimensions(50, 30, 2048, 100)

    def test_validate_image_array_properties_valid(self):
        """Test image array validation with valid properties."""
        image_array = np.zeros((100, 100, 3), dtype=np.uint8)
        # Should not raise exception
        _validate_image_array_properties(image_array)

    def test_validate_image_array_wrong_dtype(self):
        """Test image array validation with wrong dtype."""
        image_array = np.zeros((100, 100, 3), dtype=np.float32)
        with pytest.raises(UnexpectedImageDtypeError):
            _validate_image_array_properties(image_array)

    def test_validate_image_array_wrong_shape(self):
        """Test image array validation with wrong shape."""
        # Wrong number of channels
        image_array = np.zeros((100, 100, 1), dtype=np.uint8)
        with pytest.raises(UnexpectedImageShapeError):
            _validate_image_array_properties(image_array)

    def test_validate_image_name_valid(self):
        """Test image name validation with valid name."""
        # Should not raise exception
        _validate_image_name("test.jpg")

    def test_validate_image_name_empty(self):
        """Test image name validation with empty name."""
        with pytest.raises(MissingImageNameError):
            _validate_image_name("")

    def test_validate_image_name_none(self):
        """Test image name validation with None name."""
        with pytest.raises(MissingImageNameError):
            _validate_image_name(None)

    def test_validate_dataset_not_empty_valid(self):
        """Test dataset validation with valid dataset."""
        # Should not raise exception
        _validate_dataset_not_empty([1, 2, 3])

    def test_validate_dataset_not_empty_empty(self):
        """Test dataset validation with empty dataset."""
        with pytest.raises(EmptyDatasetError):
            _validate_dataset_not_empty([])


class TestImageValidator:
    """Test ImageValidator functionality."""

    def test_validate_image_file_valid(self):
        """Test image file validation with valid file."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            # Create a small test image
            img = Image.new("RGB", (200, 200), color="white")
            img.save(tmp.name)
            tmp_path = Path(tmp.name)

            try:
                # Mock the file size check to avoid issues
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = 1000  # Small size

                    result = ImageValidator.validate_image_file(tmp_path)
                    assert result is True
            finally:
                tmp_path.unlink(missing_ok=True)

    def test_validate_image_file_too_large(self):
        """Test image file validation with file too large."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)

            try:
                # Mock file size to be too large
                with patch("pathlib.Path.stat") as mock_stat:
                    mock_stat.return_value.st_size = ImageValidator.MAX_IMAGE_SIZE + 1

                    with pytest.raises(ImageValidationFailedError):
                        ImageValidator.validate_image_file(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)

    def test_load_image_safely_valid(self):
        """Test safe image loading with valid image."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            # Create a small test image
            img = Image.new("RGB", (200, 200), color="red")
            img.save(tmp.name)
            tmp_path = Path(tmp.name)

            try:
                # Mock validation to pass
                with patch.object(ImageValidator, "validate_image_file", return_value=True):
                    result = ImageValidator.load_image_safely(tmp_path)

                    assert isinstance(result, np.ndarray)
                    assert result.shape == (200, 200, 3)
                    assert result.dtype == np.uint8
            finally:
                tmp_path.unlink(missing_ok=True)

    def test_load_image_safely_validation_fails(self):
        """Test safe image loading when validation fails."""
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = Path(tmp.name)

            try:
                # Mock validation to fail
                with patch.object(
                    ImageValidator, "validate_image_file", side_effect=ImageValidationFailedError()
                ):
                    with pytest.raises(ImageLoadFailedError):
                        ImageValidator.load_image_safely(tmp_path)
            finally:
                tmp_path.unlink(missing_ok=True)


class TestTextRegionValidator:
    """Test TextRegionValidator functionality."""

    def create_test_region(self, **overrides):
        """Create a test TextRegion with defaults."""
        defaults = {
            "bbox": BoundingBox(left=10, top=10, right=50, bottom=30),
            "original_text": "Test text",
            "replacement_text": "REDACTED",
            "confidence": 0.95,
        }
        defaults.update(overrides)
        return TextRegion(**defaults)

    def test_validate_text_region_valid(self):
        """Test text region validation with valid region."""
        region = self.create_test_region()
        image_shape = (100, 100)  # height, width

        result = TextRegionValidator.validate_text_region(region, image_shape)
        assert result is True

    def test_validate_text_region_constants(self):
        """Test that validator constants are properly defined."""
        assert TextRegionValidator.MAX_TEXT_LENGTH == 1000
        assert TextRegionValidator.MIN_TEXT_LENGTH == 1
        assert TextRegionValidator.MIN_BBOX_SIZE == 10

    def test_validate_text_region_bbox_out_of_bounds(self):
        """Test text region validation with bbox out of bounds."""
        # Bbox extends beyond image
        bbox = BoundingBox(left=10, top=10, right=150, bottom=30)
        region = self.create_test_region(bbox=bbox)
        image_shape = (100, 100)  # height, width (bbox.right=150 > width=100)

        with pytest.raises(Exception):  # BoundingBoxExceedsImageError
            TextRegionValidator.validate_text_region(region, image_shape)


class TestSafeAugmentation:
    """Test SafeAugmentation functionality."""

    def create_test_config(self, **overrides):
        """Create test dataset config."""
        defaults = {
            "train_data_path": Path("/tmp"),
            "crop_size": 512,
            "brightness_range": 0.1,
            "contrast_range": 0.1,
            "rotation_range": 0.0,  # Disabled for now
            "num_workers": 1,
        }
        defaults.update(overrides)
        return DatasetConfig(**defaults)

    def test_augment_image_brightness(self):
        """Test image augmentation with brightness adjustment."""
        config = self.create_test_config(brightness_range=0.2)
        augmentation = SafeAugmentation(config)

        # Create test image and regions
        image = np.ones((100, 100, 3), dtype=np.uint8) * 128  # Gray image
        region = TextRegion(
            bbox=BoundingBox(left=10, top=10, right=50, bottom=30),
            original_text="Test",
            replacement_text="REDACTED",
            confidence=0.9,
        )
        text_regions = [region]

        # Apply augmentation
        aug_image, aug_regions = augmentation.augment_image(image, text_regions)

        # Check results
        assert isinstance(aug_image, np.ndarray)
        assert aug_image.shape == image.shape
        assert len(aug_regions) == len(text_regions)
        # Image should be different due to brightness change (with some tolerance)
        assert not np.array_equal(image, aug_image)

    def test_augment_image_contrast(self):
        """Test image augmentation with contrast adjustment."""
        config = self.create_test_config(brightness_range=0.0, contrast_range=0.2)
        augmentation = SafeAugmentation(config)

        # Create test image with varied pixel values
        image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        text_regions = []

        # Apply augmentation
        aug_image, aug_regions = augmentation.augment_image(image, text_regions)

        # Check results
        assert isinstance(aug_image, np.ndarray)
        assert aug_image.shape == image.shape
        assert len(aug_regions) == 0

    def test_augment_image_error_fallback(self):
        """Test augmentation fallback when error occurs."""
        config = self.create_test_config()
        augmentation = SafeAugmentation(config)

        # Create invalid image data to trigger error
        image = np.ones((100, 100, 3), dtype=np.uint8)
        text_regions = []

        # Mock PIL operations to raise error
        with patch("PIL.Image.fromarray", side_effect=Exception("Test error")):
            aug_image, aug_regions = augmentation.augment_image(image, text_regions)

            # Should return original data on error
            assert np.array_equal(aug_image, image)
            assert aug_regions == text_regions


class TestAnonymizerDataset:
    """Test AnonymizerDataset functionality."""

    def create_test_config(self, **overrides):
        """Create test dataset config."""
        defaults = {
            "train_data_path": Path("/tmp/test_data"),
            "crop_size": 256,
            "brightness_range": 0.0,
            "contrast_range": 0.0,
            "rotation_range": 0.0,
            "num_workers": 0,
        }
        defaults.update(overrides)
        return DatasetConfig(**defaults)

    def test_dataset_no_annotation_files(self):
        """Test dataset initialization with no annotation files."""
        config = self.create_test_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            config.train_data_path = Path(temp_dir)

            with pytest.raises(NoAnnotationFilesError):
                AnonymizerDataset(temp_dir, config, split="train")

    def test_dataset_load_sample_missing_image_name(self):
        """Test dataset sample loading with missing image name."""
        config = self.create_test_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            config.train_data_path = Path(temp_dir)

            # Create annotation file without image_name
            annotation_data = {"text_regions": []}
            annotation_file = Path(temp_dir) / "test.json"
            with annotation_file.open("w") as f:
                json.dump(annotation_data, f)

            dataset = AnonymizerDataset.__new__(AnonymizerDataset)  # Create without __init__
            dataset.config = config
            dataset.data_dir = Path(temp_dir)
            dataset.image_validator = ImageValidator()
            dataset.text_validator = TextRegionValidator()

            # Should return None for invalid sample
            result = dataset._load_sample(annotation_file)
            assert result is None

    @patch("src.anonymizer.training.datasets.ImageValidator.load_image_safely")
    def test_dataset_getitem_valid(self, mock_load_image):
        """Test dataset __getitem__ with valid data."""
        # Mock image loading
        mock_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        mock_load_image.return_value = mock_image

        config = self.create_test_config()

        with tempfile.TemporaryDirectory() as temp_dir:
            config.train_data_path = Path(temp_dir)

            # Create test image file
            test_image_path = Path(temp_dir) / "test.jpg"
            img = Image.new("RGB", (256, 256), color="white")
            img.save(test_image_path)

            # Create annotation file
            annotation_data = {
                "image_name": "test.jpg",
                "text_regions": [
                    {
                        "bbox": {"left": 10, "top": 10, "right": 50, "bottom": 30},
                        "original_text": "Test text",
                        "replacement_text": "REDACTED",
                        "confidence": 0.95,
                    }
                ],
            }
            annotation_file = Path(temp_dir) / "annotation.json"
            with annotation_file.open("w") as f:
                json.dump(annotation_data, f)

            # Create dataset
            dataset = AnonymizerDataset(temp_dir, config, split="train")

            # Test __getitem__
            assert len(dataset) > 0
            item = dataset[0]

            # Verify item structure
            assert "images" in item
            assert "masks" in item
            assert "texts" in item
            assert isinstance(item["images"], torch.Tensor)
            assert isinstance(item["masks"], torch.Tensor)

    def test_dataset_getitem_empty_samples(self):
        """Test dataset __getitem__ with empty samples."""
        config = self.create_test_config()

        # Create dataset with empty samples list
        dataset = AnonymizerDataset.__new__(AnonymizerDataset)  # Create without __init__
        dataset.config = config
        dataset.samples = []
        dataset.split = "train"
        dataset.augmentation = None

        # Should return empty dict after all retries fail
        item = dataset[0]
        assert item == {}
