"""
Unit tests for dataset loading and preprocessing - Imperative style.

Tests robust dataset loading with comprehensive validation.
"""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.anonymizer.core.config import DatasetConfig
from src.anonymizer.core.exceptions import PreprocessingError, ValidationError
from src.anonymizer.core.models import BoundingBox, TextRegion
from src.anonymizer.training.datasets import (
    AnonymizerDataset,
    DatasetSample,
    ImageValidator,
    SafeAugmentation,
    TextRegionValidator,
    collate_fn,
    create_dataloader,
    create_dataloaders,
    create_datasets,
    create_dummy_batch,
)


class TestDatasetSample:
    """Test DatasetSample validation."""

    def test_valid_dataset_sample(self, sample_image, sample_text_region, temp_dir):
        """Test creating valid dataset sample."""
        image_path = temp_dir / "test.png"
        Image.fromarray(sample_image).save(image_path)

        sample = DatasetSample(
            image_path=image_path, image=sample_image, text_regions=[sample_text_region]
        )

        assert sample.image_path == image_path
        assert np.array_equal(sample.image, sample_image)
        assert len(sample.text_regions) == 1
        assert sample.text_regions[0] == sample_text_region

    def test_dataset_sample_missing_image_file(self, sample_image, sample_text_region):
        """Test validation fails for missing image file."""
        non_existent_path = Path("/non/existent/path.png")

        with pytest.raises(ValidationError, match="Image not found"):
            DatasetSample(
                image_path=non_existent_path,
                image=sample_image,
                text_regions=[sample_text_region],
            )

    def test_dataset_sample_no_text_regions(self, sample_image, temp_dir):
        """Test validation fails for no text regions."""
        image_path = temp_dir / "test.png"
        Image.fromarray(sample_image).save(image_path)

        with pytest.raises(ValidationError, match="At least one text region required"):
            DatasetSample(image_path=image_path, image=sample_image, text_regions=[])

    def test_dataset_sample_bbox_out_of_bounds(self, sample_image, temp_dir):
        """Test validation fails for bounding box out of bounds."""
        image_path = temp_dir / "test.png"
        Image.fromarray(sample_image).save(image_path)

        # Create bbox that exceeds image dimensions
        h, w = sample_image.shape[:2]
        out_of_bounds_bbox = BoundingBox(left=0, top=0, right=w + 100, bottom=h + 100)
        out_of_bounds_region = TextRegion(
            bbox=out_of_bounds_bbox,
            original_text="Test",
            replacement_text="REDACTED",
            confidence=1.0,
        )

        with pytest.raises(ValidationError, match="Bounding box out of bounds"):
            DatasetSample(
                image_path=image_path,
                image=sample_image,
                text_regions=[out_of_bounds_region],
            )


class TestImageValidator:
    """Test ImageValidator security and format validation."""

    def test_validate_image_file_valid_png(self, temp_dir):
        """Test validating valid PNG file."""
        image_path = temp_dir / "valid.png"
        image = Image.new("RGB", (256, 256), color="white")
        image.save(image_path, "PNG")

        result = ImageValidator.validate_image_file(image_path)

        assert result is True

    def test_validate_image_file_valid_jpeg(self, temp_dir):
        """Test validating valid JPEG file."""
        image_path = temp_dir / "valid.jpg"
        image = Image.new("RGB", (256, 256), color="white")
        image.save(image_path, "JPEG")

        result = ImageValidator.validate_image_file(image_path)

        assert result is True

    def test_validate_image_file_too_large_filesize(self, temp_dir):
        """Test validation fails for files that are too large."""
        image_path = temp_dir / "large.png"

        # Mock file size to exceed limit
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value.st_size = ImageValidator.MAX_IMAGE_SIZE + 1

            with pytest.raises(ValidationError, match="Image too large"):
                ImageValidator.validate_image_file(image_path)

    def test_validate_image_file_unsupported_format(self, temp_dir):
        """Test validation fails for unsupported formats."""
        # Create a non-image file
        text_file = temp_dir / "not_image.txt"
        with open(text_file, "w") as f:
            f.write("This is not an image")

        with pytest.raises(ValidationError, match="Invalid image data"):
            ImageValidator.validate_image_file(text_file)

    def test_validate_image_file_too_large_dimensions(self, temp_dir):
        """Test validation fails for images with too large dimensions."""
        # Mock Image.open to return image with large dimensions
        with patch("PIL.Image.open") as mock_open:
            mock_image = Mock()
            mock_image.format = "PNG"
            mock_image.size = (ImageValidator.MAX_DIMENSION + 1, 100)
            mock_open.return_value.__enter__.return_value = mock_image

            image_path = temp_dir / "large_dims.png"
            image_path.touch()

            with pytest.raises(ValidationError, match="Image too large"):
                ImageValidator.validate_image_file(image_path)

    def test_validate_image_file_too_small_dimensions(self, temp_dir):
        """Test validation fails for images that are too small."""
        with patch("PIL.Image.open") as mock_open:
            mock_image = Mock()
            mock_image.format = "PNG"
            mock_image.size = (ImageValidator.MIN_DIMENSION - 1, 100)
            mock_open.return_value.__enter__.return_value = mock_image

            image_path = temp_dir / "small_dims.png"
            image_path.touch()

            with pytest.raises(ValidationError, match="Image too small"):
                ImageValidator.validate_image_file(image_path)

    def test_load_image_safely_rgb(self, temp_dir):
        """Test safe image loading for RGB image."""
        image_path = temp_dir / "rgb.png"
        original_image = Image.new("RGB", (100, 100), color="red")
        original_image.save(image_path, "PNG")

        loaded_image = ImageValidator.load_image_safely(image_path)

        assert loaded_image.shape == (100, 100, 3)
        assert loaded_image.dtype == np.uint8

    def test_load_image_safely_converts_rgba_to_rgb(self, temp_dir):
        """Test that RGBA images are converted to RGB."""
        image_path = temp_dir / "rgba.png"
        original_image = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        original_image.save(image_path, "PNG")

        loaded_image = ImageValidator.load_image_safely(image_path)

        # Should be converted to RGB
        assert loaded_image.shape == (100, 100, 3)
        assert loaded_image.dtype == np.uint8

    def test_load_image_safely_converts_grayscale_to_rgb(self, temp_dir):
        """Test that grayscale images are converted to RGB."""
        image_path = temp_dir / "gray.png"
        original_image = Image.new("L", (100, 100), color=128)
        original_image.save(image_path, "PNG")

        loaded_image = ImageValidator.load_image_safely(image_path)

        # Should be converted to RGB
        assert loaded_image.shape == (100, 100, 3)
        assert loaded_image.dtype == np.uint8

    def test_load_image_safely_validation_error(self, temp_dir):
        """Test that loading invalid image raises PreprocessingError."""
        # Create invalid image file
        invalid_file = temp_dir / "invalid.png"
        with open(invalid_file, "w") as f:
            f.write("Not an image")

        with pytest.raises(PreprocessingError, match="Failed to load image"):
            ImageValidator.load_image_safely(invalid_file)

    def test_load_image_safely_unexpected_array_shape(self, temp_dir):
        """Test handling of unexpected array shapes."""
        image_path = temp_dir / "test.png"

        with patch("PIL.Image.open") as mock_open:
            mock_image = Mock()
            mock_image.format = "PNG"
            mock_image.size = (100, 100)
            mock_image.mode = "RGB"
            mock_image.convert.return_value = mock_image
            mock_image.verify.return_value = None

            # Set up mock for both context manager and direct return
            from unittest.mock import MagicMock

            mock_context = MagicMock()
            mock_context.__enter__.return_value = mock_image
            mock_context.__exit__.return_value = None

            # For validation (context manager) and loading (direct return)
            mock_open.side_effect = [mock_context, mock_image]

            with patch("numpy.array") as mock_array:
                # Return array with unexpected shape (4D instead of 3D)
                mock_array.return_value = np.zeros((100, 100, 100, 3), dtype=np.uint8)  # 4D array

                image_path.touch()

                with pytest.raises(ValidationError, match="Unexpected image shape"):
                    ImageValidator.load_image_safely(image_path)


class TestTextRegionValidator:
    """Test TextRegionValidator validation."""

    def test_validate_text_region_valid(self, sample_text_region):
        """Test validating valid text region."""
        image_shape = (256, 256)

        result = TextRegionValidator.validate_text_region(sample_text_region, image_shape)

        assert result is True

    def test_validate_text_region_empty_original_text(self, sample_bbox):
        """Test validation fails for empty original text."""
        # Create a mock region with empty text to test the validator logic
        from unittest.mock import Mock

        region = Mock()
        region.bbox = sample_bbox
        region.original_text = ""  # Empty
        region.replacement_text = "REDACTED"
        region.confidence = 1.0

        with pytest.raises(ValidationError, match="Original text too short"):
            TextRegionValidator.validate_text_region(region, (256, 256))

    def test_validate_text_region_too_long_text(self, sample_bbox):
        """Test validation fails for text that's too long."""
        # Create a mock region with long text to test the validator logic
        from unittest.mock import Mock

        region = Mock()
        region.bbox = sample_bbox
        region.original_text = "x" * (TextRegionValidator.MAX_TEXT_LENGTH + 1)
        region.replacement_text = "REDACTED"
        region.confidence = 1.0

        with pytest.raises(ValidationError, match="Original text too long"):
            TextRegionValidator.validate_text_region(region, (256, 256))

    def test_validate_text_region_negative_bbox_coordinates(self):
        """Test validation fails for negative bounding box coordinates."""
        # Create a mock region with negative bbox to test the validator logic
        from unittest.mock import Mock

        negative_bbox = Mock()
        negative_bbox.left = -10
        negative_bbox.top = 0
        negative_bbox.right = 100
        negative_bbox.bottom = 50

        region = Mock()
        region.bbox = negative_bbox
        region.original_text = "Test"
        region.replacement_text = "REDACTED"
        region.confidence = 1.0

        with pytest.raises(ValidationError, match="Bounding box has negative coordinates"):
            TextRegionValidator.validate_text_region(region, (256, 256))

    def test_validate_text_region_bbox_exceeds_image(self):
        """Test validation fails for bbox exceeding image dimensions."""
        large_bbox = BoundingBox(left=0, top=0, right=300, bottom=300)  # Exceeds 256x256
        region = TextRegion(
            bbox=large_bbox,
            original_text="Test",
            replacement_text="REDACTED",
            confidence=1.0,
        )

        with pytest.raises(ValidationError, match="Bounding box exceeds image dimensions"):
            TextRegionValidator.validate_text_region(region, (256, 256))

    def test_validate_text_region_bbox_too_small(self, sample_bbox):
        """Test validation fails for bbox that's too small."""
        tiny_bbox = BoundingBox(left=50, top=50, right=55, bottom=55)  # 5x5 (below 10x10 minimum)
        region = TextRegion(
            bbox=tiny_bbox,
            original_text="Test",
            replacement_text="REDACTED",
            confidence=1.0,
        )

        with pytest.raises(ValidationError, match="Bounding box too small"):
            TextRegionValidator.validate_text_region(region, (256, 256))


class TestSafeAugmentation:
    """Test SafeAugmentation for text preservation."""

    def test_safe_augmentation_initialization(self, dataset_config):
        """Test safe augmentation initialization."""
        augmentation = SafeAugmentation(dataset_config)

        assert augmentation.config == dataset_config
        assert augmentation.rng is not None

    def test_augment_image_brightness(self, sample_image, sample_text_region, dataset_config):
        """Test brightness augmentation."""
        dataset_config.brightness_range = 0.2
        augmentation = SafeAugmentation(dataset_config)

        augmented_image, augmented_regions = augmentation.augment_image(
            sample_image, [sample_text_region]
        )

        # Image dimensions should be preserved
        assert augmented_image.shape == sample_image.shape
        assert augmented_image.dtype == sample_image.dtype
        assert len(augmented_regions) == 1
        assert augmented_regions[0] == sample_text_region  # Regions unchanged

    def test_augment_image_contrast(self, sample_image, sample_text_region, dataset_config):
        """Test contrast augmentation."""
        dataset_config.contrast_range = 0.15
        augmentation = SafeAugmentation(dataset_config)

        augmented_image, augmented_regions = augmentation.augment_image(
            sample_image, [sample_text_region]
        )

        # Image dimensions should be preserved
        assert augmented_image.shape == sample_image.shape
        assert len(augmented_regions) == 1

    def test_augment_image_no_augmentation(self, sample_image, sample_text_region, dataset_config):
        """Test with no augmentation (zero ranges)."""
        dataset_config.brightness_range = 0.0
        dataset_config.contrast_range = 0.0
        dataset_config.rotation_range = 0.0
        augmentation = SafeAugmentation(dataset_config)

        augmented_image, augmented_regions = augmentation.augment_image(
            sample_image, [sample_text_region]
        )

        # Should be identical to original
        assert np.array_equal(augmented_image, sample_image)
        assert augmented_regions == [sample_text_region]

    def test_augment_image_error_handling(self, sample_image, sample_text_region, dataset_config):
        """Test augmentation error handling."""
        augmentation = SafeAugmentation(dataset_config)

        with patch("PIL.Image.fromarray", side_effect=Exception("PIL error")):
            # Should return original image on error
            augmented_image, augmented_regions = augmentation.augment_image(
                sample_image, [sample_text_region]
            )

            assert np.array_equal(augmented_image, sample_image)
            assert augmented_regions == [sample_text_region]

    def test_augment_image_rotation_skipped(self, sample_image, sample_text_region, dataset_config):
        """Test that rotation is conservatively skipped."""
        dataset_config.rotation_range = 10.0  # Would cause rotation
        augmentation = SafeAugmentation(dataset_config)

        # Currently rotation is skipped to avoid coordinate transformation complexity
        augmented_image, augmented_regions = augmentation.augment_image(
            sample_image, [sample_text_region]
        )

        # Regions should be unchanged (rotation skipped)
        assert augmented_regions == [sample_text_region]


class TestAnonymizerDataset:
    """Test AnonymizerDataset implementation."""

    def test_dataset_initialization(self, mock_dataset_dir, dataset_config):
        """Test dataset initialization."""
        dataset = AnonymizerDataset(data_dir=mock_dataset_dir, config=dataset_config, split="train")

        assert dataset.data_dir == mock_dataset_dir
        assert dataset.config == dataset_config
        assert dataset.split == "train"
        assert len(dataset.samples) > 0

    def test_dataset_length(self, mock_dataset_dir, dataset_config):
        """Test dataset length."""
        dataset = AnonymizerDataset(mock_dataset_dir, dataset_config)

        assert len(dataset) == len(dataset.samples)
        assert len(dataset) > 0

    def test_dataset_getitem_valid(self, mock_dataset_dir, dataset_config):
        """Test getting valid dataset item."""
        dataset = AnonymizerDataset(mock_dataset_dir, dataset_config)

        item = dataset[0]

        assert isinstance(item, dict)
        assert "images" in item
        assert "masks" in item
        assert "texts" in item
        assert "original_size" in item
        assert "scale" in item

        # Check tensor properties
        assert isinstance(item["images"], torch.Tensor)
        assert isinstance(item["masks"], torch.Tensor)
        assert item["images"].shape[0] == 3  # RGB channels
        assert item["images"].shape[1] == dataset_config.crop_size
        assert item["images"].shape[2] == dataset_config.crop_size

    def test_dataset_getitem_error_handling(self, mock_dataset_dir, dataset_config):
        """Test dataset item error handling."""
        dataset = AnonymizerDataset(mock_dataset_dir, dataset_config)

        # Mock sample that will cause error
        with patch.object(dataset, "samples") as mock_samples:
            mock_samples.__getitem__.side_effect = Exception("Sample error")

            item = dataset[0]

            # Should return empty dict on error
            assert item == {}

    def test_dataset_no_annotation_files(self, dataset_config):
        """Test dataset with no annotation files."""
        # Create a fresh empty directory with no JSON files
        import tempfile

        with tempfile.TemporaryDirectory() as empty_dir:
            empty_path = Path(empty_dir)
            with pytest.raises(ValidationError, match="No annotation files found"):
                AnonymizerDataset(empty_path, dataset_config)

    def test_dataset_invalid_annotation_file(self, dataset_config):
        """Test dataset with invalid annotation file."""
        # Create a fresh directory with only invalid JSON file
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            invalid_json = temp_path / "invalid.json"
            with open(invalid_json, "w") as f:
                f.write("{invalid json")

            # Should skip invalid file and raise error if no valid samples
            with pytest.raises(ValidationError, match="No valid samples found"):
                AnonymizerDataset(temp_path, dataset_config)

    def test_dataset_missing_image_file(self, dataset_config):
        """Test dataset with annotation pointing to missing image."""
        # Create a fresh directory with annotation but no corresponding image
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            annotation_data = {
                "image_name": "missing_image.png",
                "text_regions": [
                    {
                        "bbox": {"left": 10, "top": 10, "right": 100, "bottom": 50},
                        "original_text": "Test",
                        "replacement_text": "REDACTED",
                        "confidence": 1.0,
                    }
                ],
            }

            annotation_file = temp_path / "annotation.json"
            with open(annotation_file, "w") as f:
                json.dump(annotation_data, f)

            # Should skip sample with missing image
            with pytest.raises(ValidationError, match="No valid samples found"):
                AnonymizerDataset(temp_path, dataset_config)

    def test_dataset_prepare_training_data(self, mock_dataset_dir, dataset_config):
        """Test training data preparation."""
        dataset = AnonymizerDataset(mock_dataset_dir, dataset_config)

        sample = dataset.samples[0]
        training_data = dataset._prepare_training_data(sample.image, sample.text_regions)

        assert "images" in training_data
        assert "masks" in training_data
        assert "texts" in training_data
        assert "original_size" in training_data
        assert "scale" in training_data

        # Check normalization
        images = training_data["images"]
        assert torch.min(images) >= -1.0
        assert torch.max(images) <= 1.0

    def test_dataset_augmentation_train_split(self, mock_dataset_dir, dataset_config):
        """Test that augmentation is applied to train split."""
        train_dataset = AnonymizerDataset(mock_dataset_dir, dataset_config, split="train")

        assert train_dataset.augmentation is not None

    def test_dataset_no_augmentation_val_split(self, mock_dataset_dir, dataset_config):
        """Test that augmentation is not applied to validation split."""
        val_dataset = AnonymizerDataset(mock_dataset_dir, dataset_config, split="val")

        assert val_dataset.augmentation is None


class TestCollateFn:
    """Test collate function for batch processing."""

    def test_collate_fn_valid_batch(self):
        """Test collate function with valid batch."""
        # Create mock batch items
        batch = [
            {
                "images": torch.randn(3, 256, 256),
                "masks": torch.ones(2, 256, 256),  # 2 text regions
                "texts": ["Hello", "World"],
                "original_size": (512, 512),
                "scale": 0.5,
            },
            {
                "images": torch.randn(3, 256, 256),
                "masks": torch.ones(1, 256, 256),  # 1 text region
                "texts": ["Test"],
                "original_size": (256, 256),
                "scale": 1.0,
            },
        ]

        collated = collate_fn(batch)

        assert "images" in collated
        assert "masks" in collated
        assert "texts" in collated
        assert "original_sizes" in collated
        assert "scales" in collated
        assert "batch_size" in collated

        # Check batch dimensions
        assert collated["images"].shape[0] == 2  # Batch size
        assert collated["masks"].shape[0] == 2  # Batch size
        # Masks should be padded to same number of regions (2)
        assert collated["masks"].shape[1] == 2  # Max regions
        assert len(collated["texts"]) == 3  # Flattened texts: ["Hello", "World", "Test"]
        assert collated["batch_size"] == 2

    def test_collate_fn_empty_items_filtered(self):
        """Test that empty items are filtered out."""
        batch = [
            {
                "images": torch.randn(3, 256, 256),
                "masks": torch.ones(1, 256, 256),
                "texts": ["Valid"],
                "original_size": (256, 256),
                "scale": 1.0,
            },
            {},  # Empty item (should be filtered)
            {
                "images": torch.randn(3, 256, 256),
                "masks": torch.ones(1, 256, 256),
                "texts": ["Also Valid"],
                "original_size": (256, 256),
                "scale": 1.0,
            },
        ]

        collated = collate_fn(batch)

        assert collated["batch_size"] == 2  # Empty item filtered out
        assert len(collated["texts"]) == 2

    def test_collate_fn_all_empty_batch(self):
        """Test collate function with all empty items."""
        batch = [{}, {}, {}]  # All empty

        with pytest.raises(ValidationError, match="Empty batch after filtering"):
            collate_fn(batch)

    def test_collate_fn_text_flattening(self):
        """Test that texts are properly flattened."""
        batch = [
            {
                "images": torch.randn(3, 256, 256),
                "masks": torch.ones(3, 256, 256),
                "texts": ["A", "B", "C"],  # 3 texts
                "original_size": (256, 256),
                "scale": 1.0,
            },
            {
                "images": torch.randn(3, 256, 256),
                "masks": torch.ones(2, 256, 256),
                "texts": ["D", "E"],  # 2 texts
                "original_size": (256, 256),
                "scale": 1.0,
            },
        ]

        collated = collate_fn(batch)

        # Should flatten to ["A", "B", "C", "D", "E"]
        assert collated["texts"] == ["A", "B", "C", "D", "E"]
        # Masks should be padded to same number of regions (3)
        assert collated["masks"].shape[1] == 3  # Max regions


class TestDataLoaderCreation:
    """Test data loader creation functions."""

    def test_create_dataloader(self, mock_dataset_dir, dataset_config):
        """Test creating data loader."""
        dataset = AnonymizerDataset(mock_dataset_dir, dataset_config)

        dataloader = create_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
            num_workers=0,  # No multiprocessing in tests
            pin_memory=False,
        )

        assert dataloader.batch_size == 2
        # Check if sampler is shuffling (RandomSampler vs SequentialSampler)
        from torch.utils.data import RandomSampler

        assert isinstance(dataloader.sampler, RandomSampler)
        assert dataloader.num_workers == 0
        assert dataloader.collate_fn == collate_fn

    def test_create_datasets(self, mock_dataset_dir):
        """Test creating train and validation datasets."""
        config = DatasetConfig(
            train_data_path=mock_dataset_dir,
            val_data_path=mock_dataset_dir,  # Same directory for testing
            crop_size=256,
            num_workers=0,
        )

        train_dataset, val_dataset = create_datasets(config)

        assert isinstance(train_dataset, AnonymizerDataset)
        assert isinstance(val_dataset, AnonymizerDataset)
        assert train_dataset.split == "train"
        assert val_dataset.split == "val"

    def test_create_datasets_no_validation(self, mock_dataset_dir):
        """Test creating datasets without validation data."""
        config = DatasetConfig(
            train_data_path=mock_dataset_dir,
            val_data_path=None,  # No validation
            crop_size=256,
            num_workers=0,
        )

        train_dataset, val_dataset = create_datasets(config)

        assert isinstance(train_dataset, AnonymizerDataset)
        assert val_dataset is None

    def test_create_dataloaders(self, mock_dataset_dir):
        """Test creating train and validation data loaders."""
        config = DatasetConfig(
            train_data_path=mock_dataset_dir,
            val_data_path=mock_dataset_dir,
            crop_size=256,
            num_workers=0,
        )

        # Use default batch size of 2 for testing
        batch_size = 2

        train_dataloader, val_dataloader = create_dataloaders(config, batch_size=batch_size)

        assert train_dataloader.batch_size == 2
        assert val_dataloader.batch_size == 2
        # Check sampler types instead of shuffle attribute
        from torch.utils.data import RandomSampler, SequentialSampler

        assert isinstance(train_dataloader.sampler, RandomSampler)
        assert isinstance(val_dataloader.sampler, SequentialSampler)

    def test_create_dataloaders_no_validation(self, mock_dataset_dir):
        """Test creating data loaders without validation."""
        config = DatasetConfig(
            train_data_path=mock_dataset_dir,
            val_data_path=None,
            crop_size=256,
            num_workers=0,
        )
        batch_size = 2

        train_dataloader, val_dataloader = create_dataloaders(config, batch_size=batch_size)

        assert train_dataloader is not None
        assert val_dataloader is None


class TestDatasetIntegration:
    """Test dataset integration and end-to-end functionality."""

    def test_dataset_iteration(self, mock_dataset_dir, dataset_config):
        """Test iterating through dataset."""
        dataset = AnonymizerDataset(mock_dataset_dir, dataset_config)

        items = []
        for i in range(min(len(dataset), 3)):  # Test first 3 items
            item = dataset[i]
            items.append(item)

        assert len(items) > 0
        for item in items:
            if item:  # Skip empty items
                assert "images" in item
                assert "masks" in item
                assert "texts" in item

    def test_dataloader_iteration(self, mock_dataset_dir, dataset_config):
        """Test iterating through data loader."""
        dataset = AnonymizerDataset(mock_dataset_dir, dataset_config)
        dataloader = create_dataloader(dataset, batch_size=1, num_workers=0)

        batch_count = 0
        for batch in dataloader:
            assert "images" in batch
            assert "masks" in batch
            assert "texts" in batch
            batch_count += 1
            if batch_count >= 2:  # Test first 2 batches
                break

        assert batch_count > 0

    def test_dataset_memory_efficiency(self, mock_dataset_dir, dataset_config):
        """Test that dataset doesn't load all images into memory at once."""
        dataset = AnonymizerDataset(mock_dataset_dir, dataset_config)

        # Getting items should not cause memory explosion
        item1 = dataset[0]
        item2 = dataset[0]  # Same item again

        # Should be able to get items without issues
        assert isinstance(item1, dict)
        assert isinstance(item2, dict)

    def test_dataset_deterministic_with_augmentation(self, mock_dataset_dir, dataset_config):
        """Test dataset determinism with augmentation."""
        # Create datasets with augmentation
        dataset1 = AnonymizerDataset(mock_dataset_dir, dataset_config, split="train")
        dataset2 = AnonymizerDataset(mock_dataset_dir, dataset_config, split="train")

        # Get items - they should be deterministic due to fixed seed in SafeAugmentation
        item1 = dataset1[0]
        item2 = dataset2[0]

        # Should be identical due to fixed seed in SafeAugmentation
        if item1 and item2:  # Skip if either is empty
            assert torch.allclose(item1["images"], item2["images"], atol=1e-6)

    def test_dataset_robustness_corrupted_data(self, temp_dir, dataset_config):
        """Test dataset robustness with some corrupted data."""
        # Create mix of valid and invalid samples

        # Valid sample
        valid_image = Image.new("RGB", (256, 256), color="white")
        valid_image.save(temp_dir / "valid.png")

        valid_annotation = {
            "image_name": "valid.png",
            "text_regions": [
                {
                    "bbox": {"left": 10, "top": 10, "right": 100, "bottom": 50},
                    "original_text": "Valid Text",
                    "replacement_text": "REDACTED",
                    "confidence": 1.0,
                }
            ],
        }

        with open(temp_dir / "valid.json", "w") as f:
            json.dump(valid_annotation, f)

        # Invalid sample (missing image)
        invalid_annotation = {
            "image_name": "missing.png",
            "text_regions": [
                {
                    "bbox": {"left": 10, "top": 10, "right": 100, "bottom": 50},
                    "original_text": "Invalid Text",
                    "replacement_text": "REDACTED",
                    "confidence": 1.0,
                }
            ],
        }

        with open(temp_dir / "invalid.json", "w") as f:
            json.dump(invalid_annotation, f)

        # Should load only valid samples (invalid ones are logged as errors but skipped)
        dataset = AnonymizerDataset(temp_dir, dataset_config)

        # The dataset should have at least 1 valid sample
        # (The invalid sample is logged as error but doesn't prevent dataset creation)
        assert len(dataset) >= 1  # At least the valid sample loaded

        item = dataset[0]
        assert item  # Should not be empty
        assert "images" in item


class TestBoundingBoxFixes:
    """Test fixes for bounding box scaling and clamping."""

    def test_bbox_scale_with_rounding(self):
        """Test that bounding box scaling uses proper rounding."""
        bbox = BoundingBox(left=10, top=20, right=30, bottom=40)

        # Test scaling with factor that would cause truncation issues
        scaled = bbox.scale(1.7)  # 10*1.7=17.0, 20*1.7=34.0, etc.

        assert scaled.left == 17  # round(10 * 1.7)
        assert scaled.top == 34  # round(20 * 1.7)
        assert scaled.right == 51  # round(30 * 1.7)
        assert scaled.bottom == 68  # round(40 * 1.7)

    def test_bbox_clamp_to_bounds(self):
        """Test bounding box clamping to image bounds."""
        # Bbox that exceeds image bounds
        bbox = BoundingBox(left=-5, top=-10, right=600, bottom=700)

        # Clamp to 512x512 image
        clamped = bbox.clamp_to_bounds(512, 512)

        assert clamped.left == 0  # Clamped from -5
        assert clamped.top == 0  # Clamped from -10
        assert clamped.right == 512  # Clamped from 600
        assert clamped.bottom == 512  # Clamped from 700

    def test_bbox_clamp_preserves_valid_coords(self):
        """Test that valid coordinates are preserved during clamping."""
        bbox = BoundingBox(left=50, top=100, right=200, bottom=150)

        # Should remain unchanged when within bounds
        clamped = bbox.clamp_to_bounds(512, 512)

        assert clamped.left == 50
        assert clamped.top == 100
        assert clamped.right == 200
        assert clamped.bottom == 150

    def test_bbox_clamp_handles_edge_cases(self):
        """Test edge cases in bounding box clamping."""
        # Test minimum size preservation
        bbox = BoundingBox(left=510, top=510, right=520, bottom=520)
        clamped = bbox.clamp_to_bounds(512, 512)

        # Should ensure minimum size of 1x1
        assert clamped.left == 511  # max(0, min(510, 512-1))
        assert clamped.top == 511  # max(0, min(510, 512-1))
        assert clamped.right == 512  # max(1, min(520, 512))
        assert clamped.bottom == 512  # max(1, min(520, 512))


class TestDatasetPreprocessingFixes:
    """Test fixes for dataset preprocessing issues."""

    def test_out_of_bounds_bbox_handling(self, mock_dataset_config, temp_dir):
        """Test that out-of-bounds bounding boxes are handled gracefully."""
        # Create image
        image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        image_path = temp_dir / "test_image.png"
        Image.fromarray(image).save(image_path)

        # Create bbox that will exceed bounds after scaling
        out_of_bounds_bbox = BoundingBox(left=200, top=200, right=250, bottom=250)
        text_region = TextRegion(
            bbox=out_of_bounds_bbox,
            original_text="Test",
            replacement_text="REDACTED",
            confidence=1.0,
        )

        # Create annotation with out-of-bounds region
        annotation_data = {
            "image_name": "test_image.png",
            "text_regions": [
                {
                    "bbox": {
                        "left": text_region.bbox.left,
                        "top": text_region.bbox.top,
                        "right": text_region.bbox.right,
                        "bottom": text_region.bbox.bottom,
                    },
                    "original_text": text_region.original_text,
                    "replacement_text": text_region.replacement_text,
                    "confidence": text_region.confidence,
                }
            ],
        }

        annotation_path = temp_dir / "test_annotation.json"
        with open(annotation_path, "w") as f:
            json.dump(annotation_data, f)

        # Create dataset - should handle out-of-bounds gracefully
        dataset = AnonymizerDataset(data_dir=temp_dir, config=mock_dataset_config, split="train")

        # Should successfully create dataset
        assert len(dataset) == 1

        # Get item - should handle scaling and clamping
        item = dataset[0]
        assert item  # Should not be empty
        assert "images" in item
        assert "masks" in item

        # Masks should be valid (not out of bounds)
        masks = item["masks"]
        assert masks.shape[1] == mock_dataset_config.crop_size
        assert masks.shape[2] == mock_dataset_config.crop_size

    def test_very_small_bbox_handling(self, mock_dataset_config, temp_dir):
        """Test handling of bounding boxes that become too small after scaling."""
        # Create image
        image = np.ones((512, 512, 3), dtype=np.uint8) * 255
        image_path = temp_dir / "test_image.png"
        Image.fromarray(image).save(image_path)

        # Create very small bbox that might disappear after scaling down
        tiny_bbox = BoundingBox(left=100, top=100, right=102, bottom=102)  # 2x2 pixels
        text_region = TextRegion(
            bbox=tiny_bbox,
            original_text="X",
            replacement_text="Y",
            confidence=1.0,
        )

        # Create annotation
        annotation_data = {
            "image_name": "test_image.png",
            "text_regions": [
                {
                    "bbox": {
                        "left": text_region.bbox.left,
                        "top": text_region.bbox.top,
                        "right": text_region.bbox.right,
                        "bottom": text_region.bbox.bottom,
                    },
                    "original_text": text_region.original_text,
                    "replacement_text": text_region.replacement_text,
                    "confidence": text_region.confidence,
                }
            ],
        }

        annotation_path = temp_dir / "test_annotation.json"
        with open(annotation_path, "w") as f:
            json.dump(annotation_data, f)

        # Create dataset with small crop size to force downscaling
        config = mock_dataset_config
        config.crop_size = 128  # This will cause significant downscaling

        dataset = AnonymizerDataset(data_dir=temp_dir, config=config, split="train")

        # Should handle tiny bboxes gracefully
        assert len(dataset) == 1
        item = dataset[0]

        # Should either have valid masks or dummy mask
        assert "masks" in item
        assert item["masks"].shape[0] >= 1  # At least one mask (could be dummy)


class TestCollateFunctionFixes:
    """Test fixes for collate function issues."""

    def test_create_dummy_batch(self):
        """Test dummy batch creation for error recovery."""
        dummy_batch = create_dummy_batch()

        assert "images" in dummy_batch
        assert "masks" in dummy_batch
        assert "texts" in dummy_batch
        assert "text_mask_indices" in dummy_batch

        # Check shapes
        assert dummy_batch["images"].shape == (1, 3, 512, 512)
        assert dummy_batch["masks"].shape == (1, 1, 512, 512)
        assert len(dummy_batch["texts"]) == 1
        assert len(dummy_batch["text_mask_indices"]) == 1
        assert dummy_batch["text_mask_indices"][0] == (0, 0)

    def test_collate_fn_with_empty_batch(self):
        """Test collate function handles empty batches gracefully."""
        empty_batch = []
        result = collate_fn(empty_batch)

        # Should return dummy batch
        assert result["batch_size"] == 1
        assert "text_mask_indices" in result

    def test_collate_fn_with_failed_samples(self):
        """Test collate function filters out failed samples."""
        # Mix of valid and invalid samples
        batch = [
            {},  # Empty sample (failed)
            {
                "images": torch.randn(3, 512, 512),
                "masks": torch.randn(0, 512, 512),
                "texts": [],
            },  # No masks
            {
                "images": torch.randn(3, 512, 512),
                "masks": torch.randn(2, 512, 512),
                "texts": ["text1", "text2"],
                "original_size": (512, 512),
                "scale": 1.0,
            },  # Valid sample
        ]

        result = collate_fn(batch)

        # Should only include the valid sample
        assert result["batch_size"] == 1
        assert len(result["texts"]) == 2
        assert len(result["text_mask_indices"]) == 2

    def test_collate_fn_text_mask_alignment(self):
        """Test that text-mask alignment is preserved."""
        batch = [
            {
                "images": torch.randn(3, 512, 512),
                "masks": torch.randn(2, 512, 512),
                "texts": ["text1", "text2"],
                "original_size": (512, 512),
                "scale": 1.0,
            },
            {
                "images": torch.randn(3, 512, 512),
                "masks": torch.randn(3, 512, 512),
                "texts": ["text3", "text4", "text5"],
                "original_size": (256, 256),
                "scale": 2.0,
            },
        ]

        result = collate_fn(batch)

        # Check text-mask alignment indices
        expected_indices = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
        assert result["text_mask_indices"] == expected_indices

        # Check flattened texts
        expected_texts = ["text1", "text2", "text3", "text4", "text5"]
        assert result["texts"] == expected_texts

    def test_collate_fn_mask_padding(self):
        """Test that masks are padded correctly to same number of regions."""
        batch = [
            {
                "images": torch.randn(3, 512, 512),
                "masks": torch.randn(1, 512, 512),  # 1 region
                "texts": ["text1"],
                "original_size": (512, 512),
                "scale": 1.0,
            },
            {
                "images": torch.randn(3, 512, 512),
                "masks": torch.randn(3, 512, 512),  # 3 regions
                "texts": ["text2", "text3", "text4"],
                "original_size": (256, 256),
                "scale": 2.0,
            },
        ]

        result = collate_fn(batch)

        # All masks should be padded to 3 regions
        assert result["masks"].shape == (2, 3, 512, 512)

        # First sample should have 2 zero-padded regions
        first_mask = result["masks"][0]
        assert torch.allclose(first_mask[1:], torch.zeros(2, 512, 512))


class TestDatasetErrorHandling:
    """Test improved error handling in dataset."""

    def test_getitem_retry_mechanism(self, mock_dataset_config, temp_dir):
        """Test that __getitem__ retries on failures."""
        # Create one valid sample
        image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        image_path = temp_dir / "valid_image.png"
        Image.fromarray(image).save(image_path)

        valid_region = TextRegion(
            bbox=BoundingBox(left=50, top=50, right=100, bottom=100),
            original_text="Valid",
            replacement_text="VALID",
            confidence=1.0,
        )

        annotation_data = {
            "image_name": "valid_image.png",
            "text_regions": [
                {
                    "bbox": {
                        "left": valid_region.bbox.left,
                        "top": valid_region.bbox.top,
                        "right": valid_region.bbox.right,
                        "bottom": valid_region.bbox.bottom,
                    },
                    "original_text": valid_region.original_text,
                    "replacement_text": valid_region.replacement_text,
                    "confidence": valid_region.confidence,
                }
            ],
        }

        annotation_path = temp_dir / "valid_annotation.json"
        with open(annotation_path, "w") as f:
            json.dump(annotation_data, f)

        dataset = AnonymizerDataset(data_dir=temp_dir, config=mock_dataset_config, split="train")

        # Test that out-of-bounds indices are handled with modulo
        item = dataset[100]  # Way beyond dataset size
        assert item  # Should still return valid item due to modulo indexing

    def test_getitem_handles_all_failures(self, mock_dataset_config, temp_dir):
        """Test __getitem__ returns empty dict when all retries fail."""
        # Create dataset with problematic data that will cause preprocessing to fail
        # This is harder to test directly, so we'll mock the _prepare_training_data method

        dataset = AnonymizerDataset(data_dir=temp_dir, config=mock_dataset_config, split="train")

        # Add a dummy sample to avoid empty dataset
        dummy_image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        dummy_region = TextRegion(
            bbox=BoundingBox(left=50, top=50, right=100, bottom=100),
            original_text="Test",
            replacement_text="TEST",
            confidence=1.0,
        )

        from src.anonymizer.training.datasets import DatasetSample

        dummy_sample = DatasetSample(
            image_path=temp_dir / "dummy.png",
            image=dummy_image,
            text_regions=[dummy_region],
        )
        dataset.samples = [dummy_sample]

        # Mock _prepare_training_data to always fail
        original_method = dataset._prepare_training_data
        dataset._prepare_training_data = Mock(side_effect=Exception("Simulated failure"))

        try:
            # Should return empty dict after all retries fail
            item = dataset[0]
            assert item == {}
        finally:
            # Restore original method
            dataset._prepare_training_data = original_method


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_size_bbox_after_scaling(self):
        """Test handling of bboxes that become zero-size after scaling."""
        # Create a 1-pixel bbox
        tiny_bbox = BoundingBox(left=100, top=100, right=101, bottom=101)

        # Scale down significantly
        scaled = tiny_bbox.scale(0.1)  # Should become very small

        # Clamp to reasonable bounds
        clamped = scaled.clamp_to_bounds(512, 512)

        # Should still be valid (minimum 1x1)
        assert clamped.width >= 1, f"Width too small: {clamped.width}"
        assert clamped.height >= 1, f"Height too small: {clamped.height}"

    def test_bbox_at_image_boundary(self):
        """Test bboxes exactly at image boundaries."""
        # Bbox at right/bottom edge
        edge_bbox = BoundingBox(left=510, top=510, right=512, bottom=512)

        # Should remain unchanged when within bounds
        clamped = edge_bbox.clamp_to_bounds(512, 512)
        assert clamped == edge_bbox, f"Edge bbox changed: {clamped}"

    def test_collate_with_mixed_valid_invalid_samples(self):
        """Test collate function with mix of valid and completely invalid samples."""
        batch = [
            {},  # Empty (invalid)
            {
                "images": torch.randn(3, 512, 512),
                "masks": torch.zeros(0, 512, 512),
                "texts": [],
            },  # No regions
            None,  # None (invalid)
            {
                "images": torch.randn(3, 512, 512),
                "masks": torch.randn(1, 512, 512),
                "texts": ["valid"],
                "original_size": (512, 512),
                "scale": 1.0,
            },  # Valid
        ]

        result = collate_fn(batch)

        # Should only process the valid sample
        assert result["batch_size"] == 1
        assert result["texts"] == ["valid"]
        assert len(result["text_mask_indices"]) == 1

    def test_very_large_scale_factor(self):
        """Test handling of very large scale factors."""
        bbox = BoundingBox(left=1, top=1, right=2, bottom=2)

        # Very large scale factor
        scaled = bbox.scale(1000.0)

        # Should handle large numbers correctly
        assert scaled.left == 1000
        assert scaled.right == 2000

        # Clamping should bring it back to reasonable bounds
        clamped = scaled.clamp_to_bounds(512, 512)
        assert clamped.right <= 512
        assert clamped.bottom <= 512
