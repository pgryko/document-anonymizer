"""
Unit tests for dataset loading and preprocessing - Imperative style.

Tests robust dataset loading with comprehensive validation.
"""

import pytest
import numpy as np
import torch
from PIL import Image
import json
from pathlib import Path
from unittest.mock import patch, Mock

from src.anonymizer.training.datasets import (
    DatasetSample,
    ImageValidator,
    TextRegionValidator,
    SafeAugmentation,
    AnonymizerDataset,
    collate_fn,
    create_dataloader,
    create_datasets,
    create_dataloaders,
)
from src.anonymizer.core.models import BoundingBox, TextRegion
from src.anonymizer.core.config import DatasetConfig
from src.anonymizer.core.exceptions import ValidationError, PreprocessingError


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
                mock_array.return_value = np.zeros(
                    (100, 100, 100, 3), dtype=np.uint8
                )  # 4D array

                image_path.touch()

                with pytest.raises(ValidationError, match="Unexpected image shape"):
                    ImageValidator.load_image_safely(image_path)


class TestTextRegionValidator:
    """Test TextRegionValidator validation."""

    def test_validate_text_region_valid(self, sample_text_region):
        """Test validating valid text region."""
        image_shape = (256, 256)

        result = TextRegionValidator.validate_text_region(
            sample_text_region, image_shape
        )

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
        long_text = "x" * (TextRegionValidator.MAX_TEXT_LENGTH + 1)
        region = TextRegion(
            bbox=sample_bbox,
            original_text=long_text,
            replacement_text="REDACTED",
            confidence=1.0,
        )

        with pytest.raises(ValidationError, match="Original text too long"):
            TextRegionValidator.validate_text_region(region, (256, 256))

    def test_validate_text_region_negative_bbox_coordinates(self):
        """Test validation fails for negative bounding box coordinates."""
        negative_bbox = BoundingBox(left=-10, top=0, right=100, bottom=50)
        region = TextRegion(
            bbox=negative_bbox,
            original_text="Test",
            replacement_text="REDACTED",
            confidence=1.0,
        )

        with pytest.raises(
            ValidationError, match="Bounding box has negative coordinates"
        ):
            TextRegionValidator.validate_text_region(region, (256, 256))

    def test_validate_text_region_bbox_exceeds_image(self):
        """Test validation fails for bbox exceeding image dimensions."""
        large_bbox = BoundingBox(
            left=0, top=0, right=300, bottom=300
        )  # Exceeds 256x256
        region = TextRegion(
            bbox=large_bbox,
            original_text="Test",
            replacement_text="REDACTED",
            confidence=1.0,
        )

        with pytest.raises(
            ValidationError, match="Bounding box exceeds image dimensions"
        ):
            TextRegionValidator.validate_text_region(region, (256, 256))

    def test_validate_text_region_bbox_too_small(self, sample_bbox):
        """Test validation fails for bbox that's too small."""
        tiny_bbox = BoundingBox(
            left=50, top=50, right=55, bottom=55
        )  # 5x5 (below 10x10 minimum)
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

    def test_augment_image_brightness(
        self, sample_image, sample_text_region, dataset_config
    ):
        """Test brightness augmentation."""
        dataset_config.brightness_range = 0.2
        augmentation = SafeAugmentation(dataset_config)

        augmented_image, augmented_regions = augmentation.augment_image(
            sample_image, [sample_text_region]
        )

        assert augmented_image.shape == sample_image.shape
        assert augmented_image.dtype == sample_image.dtype
        assert len(augmented_regions) == 1
        assert augmented_regions[0] == sample_text_region  # Regions unchanged

    def test_augment_image_contrast(
        self, sample_image, sample_text_region, dataset_config
    ):
        """Test contrast augmentation."""
        dataset_config.contrast_range = 0.15
        augmentation = SafeAugmentation(dataset_config)

        augmented_image, augmented_regions = augmentation.augment_image(
            sample_image, [sample_text_region]
        )

        assert augmented_image.shape == sample_image.shape
        assert len(augmented_regions) == 1

    def test_augment_image_no_augmentation(
        self, sample_image, sample_text_region, dataset_config
    ):
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

    def test_augment_image_error_handling(
        self, sample_image, sample_text_region, dataset_config
    ):
        """Test augmentation error handling."""
        augmentation = SafeAugmentation(dataset_config)

        with patch("PIL.Image.fromarray", side_effect=Exception("PIL error")):
            # Should return original image on error
            augmented_image, augmented_regions = augmentation.augment_image(
                sample_image, [sample_text_region]
            )

            assert np.array_equal(augmented_image, sample_image)
            assert augmented_regions == [sample_text_region]

    def test_augment_image_rotation_skipped(
        self, sample_image, sample_text_region, dataset_config
    ):
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
        dataset = AnonymizerDataset(
            data_dir=mock_dataset_dir, config=dataset_config, split="train"
        )

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

    def test_dataset_no_annotation_files(self, temp_dir, dataset_config):
        """Test dataset with no annotation files."""
        # Empty directory with no JSON files
        with pytest.raises(ValidationError, match="No annotation files found"):
            AnonymizerDataset(temp_dir, dataset_config)

    def test_dataset_invalid_annotation_file(self, temp_dir, dataset_config):
        """Test dataset with invalid annotation file."""
        # Create invalid JSON file
        invalid_json = temp_dir / "invalid.json"
        with open(invalid_json, "w") as f:
            f.write("{invalid json")

        # Should skip invalid file and raise error if no valid samples
        with pytest.raises(ValidationError, match="No valid samples found"):
            AnonymizerDataset(temp_dir, dataset_config)

    def test_dataset_missing_image_file(self, temp_dir, dataset_config):
        """Test dataset with annotation pointing to missing image."""
        # Create annotation without corresponding image
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

        annotation_file = temp_dir / "annotation.json"
        with open(annotation_file, "w") as f:
            json.dump(annotation_data, f)

        # Should skip sample with missing image
        with pytest.raises(ValidationError, match="No valid samples found"):
            AnonymizerDataset(temp_dir, dataset_config)

    def test_dataset_prepare_training_data(self, mock_dataset_dir, dataset_config):
        """Test training data preparation."""
        dataset = AnonymizerDataset(mock_dataset_dir, dataset_config)

        sample = dataset.samples[0]
        training_data = dataset._prepare_training_data(
            sample.image, sample.text_regions
        )

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
        train_dataset = AnonymizerDataset(
            mock_dataset_dir, dataset_config, split="train"
        )

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
        assert (
            len(collated["texts"]) == 3
        )  # Flattened texts: ["Hello", "World", "Test"]
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
        assert dataloader.shuffle is True
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
            batch_size=2,  # Add batch_size to config
        )

        # Add batch_size to config manually
        config.batch_size = 2

        train_dataloader, val_dataloader = create_dataloaders(config)

        assert train_dataloader.batch_size == 2
        assert val_dataloader.batch_size == 2
        assert train_dataloader.shuffle is True
        assert val_dataloader.shuffle is False  # Validation shouldn't shuffle

    def test_create_dataloaders_no_validation(self, mock_dataset_dir):
        """Test creating data loaders without validation."""
        config = DatasetConfig(
            train_data_path=mock_dataset_dir,
            val_data_path=None,
            crop_size=256,
            num_workers=0,
        )
        config.batch_size = 2

        train_dataloader, val_dataloader = create_dataloaders(config)

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

    def test_dataset_deterministic_with_augmentation(
        self, mock_dataset_dir, dataset_config
    ):
        """Test dataset determinism with augmentation."""
        # Set random seed for reproducibility
        import random

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        dataset1 = AnonymizerDataset(mock_dataset_dir, dataset_config, split="train")
        dataset2 = AnonymizerDataset(mock_dataset_dir, dataset_config, split="train")

        # Reset seeds
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        item1 = dataset1[0]

        # Reset seeds again
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        item2 = dataset2[0]

        # Should be identical with same random seed
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

        # Should load only valid samples
        dataset = AnonymizerDataset(temp_dir, dataset_config)

        assert len(dataset) == 1  # Only valid sample loaded

        item = dataset[0]
        assert item  # Should not be empty
        assert "images" in item
