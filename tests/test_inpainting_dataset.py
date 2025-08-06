"""Unit tests for InpaintingDataset.

Tests the new InpaintingDataset class and related functionality.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.anonymizer.core.config import DatasetConfig
from src.anonymizer.training.datasets import (
    InpaintingDataset,
    create_dummy_inpainting_batch,
    create_inpainting_dataloaders,
    inpainting_collate_fn,
)


@pytest.fixture
def mock_training_data():
    """Create mock training data for testing."""

    def _create_data(data_dir: Path, num_samples: int = 2):
        data_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            # Create a simple test image
            image = Image.new("RGB", (256, 256), color=(255, 255, 255))
            pixels = np.array(image)
            pixels[50:100, 50:150] = [255, 0, 0]  # Red rectangle
            pixels[120:170, 100:200] = [0, 255, 0]  # Green rectangle

            image = Image.fromarray(pixels)
            image_path = data_dir / f"sample_{i}.png"
            image.save(image_path)

            # Create corresponding annotation
            annotation = {
                "image_name": f"sample_{i}.png",
                "text_regions": [
                    {
                        "bbox": {"left": 50, "top": 50, "right": 150, "bottom": 100},
                        "original_text": "CONFIDENTIAL",
                        "replacement_text": "[REDACTED]",
                        "confidence": 0.95,
                    },
                    {
                        "bbox": {"left": 100, "top": 120, "right": 200, "bottom": 170},
                        "original_text": "John Doe",
                        "replacement_text": "[NAME]",
                        "confidence": 0.88,
                    },
                ],
            }

            annotation_path = data_dir / f"sample_{i}.json"
            with annotation_path.open("w") as f:
                json.dump(annotation, f)

    return _create_data


class TestInpaintingDataset:
    """Test InpaintingDataset class."""

    def test_inpainting_dataset_creation(self, mock_training_data):
        """Test InpaintingDataset can be created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            mock_training_data(data_path, num_samples=2)

            config = DatasetConfig(train_data_path=data_path, crop_size=512, num_workers=0)

            dataset = InpaintingDataset(data_dir=data_path, config=config, split="train")

            assert len(dataset) == 2
            assert dataset.split == "train"
            assert dataset.config.crop_size == 512

    def test_inpainting_dataset_getitem(self, mock_training_data):
        """Test InpaintingDataset.__getitem__ returns correct format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            mock_training_data(data_path, num_samples=1)

            config = DatasetConfig(train_data_path=data_path, crop_size=512, num_workers=0)

            dataset = InpaintingDataset(data_dir=data_path, config=config, split="train")

            item = dataset[0]

            # Check expected keys
            assert "images" in item
            assert "masks" in item
            assert "texts" in item
            assert "original_size" in item
            assert "scale" in item

            # Check tensor shapes
            assert item["images"].shape == (3, 512, 512)  # RGB
            assert item["masks"].shape == (1, 512, 512)  # Single mask

            # Check data types
            assert isinstance(item["images"], torch.Tensor)
            assert isinstance(item["masks"], torch.Tensor)
            assert isinstance(item["texts"], str)

            # Check image normalization (should be [-1, 1])
            assert item["images"].min() >= -1.0
            assert item["images"].max() <= 1.0

            # Check mask values (should be 0 or 1)
            unique_mask_values = torch.unique(item["masks"])
            assert all(val in [0.0, 1.0] for val in unique_mask_values)

    def test_composite_mask_generation(self, mock_training_data):
        """Test that composite masks are generated correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            mock_training_data(data_path, num_samples=1)

            config = DatasetConfig(
                train_data_path=data_path,
                crop_size=256,  # Smaller for easier testing
                num_workers=0,
            )

            dataset = InpaintingDataset(data_dir=data_path, config=config, split="train")

            item = dataset[0]
            mask = item["masks"][0]  # Remove batch dimension

            # Check that mask has some positive values (text regions)
            assert mask.max() > 0, "Mask should have some regions marked for inpainting"

            # Check that mask is not all ones
            assert mask.min() < 1, "Mask should not cover entire image"

            # Check that positive regions are reasonably sized
            positive_pixels = (mask > 0).sum().item()
            total_pixels = mask.numel()
            coverage = positive_pixels / total_pixels

            assert 0 < coverage < 0.5, f"Mask coverage should be reasonable, got {coverage:.3f}"

    def test_text_conditioning(self, mock_training_data):
        """Test text conditioning for inpainting."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            mock_training_data(data_path, num_samples=1)

            config = DatasetConfig(train_data_path=data_path, crop_size=512, num_workers=0)

            dataset = InpaintingDataset(data_dir=data_path, config=config, split="train")

            item = dataset[0]
            text = item["texts"]

            # Should combine multiple text replacements
            assert "[REDACTED]" in text
            assert "[NAME]" in text

            # Should be a single string (not list)
            assert isinstance(text, str)


class TestInpaintingCollateFunction:
    """Test inpainting collate function."""

    def test_collate_valid_batch(self):
        """Test collate function with valid batch."""
        # Create mock batch items
        batch = [
            {
                "images": torch.randn(3, 512, 512),
                "masks": torch.ones(1, 512, 512),
                "texts": "test text 1",
                "original_size": (256, 256),
                "scale": 2.0,
            },
            {
                "images": torch.randn(3, 512, 512),
                "masks": torch.zeros(1, 512, 512),
                "texts": "test text 2",
                "original_size": (128, 128),
                "scale": 4.0,
            },
        ]

        result = inpainting_collate_fn(batch)

        # Check batch structure
        assert "images" in result
        assert "masks" in result
        assert "texts" in result
        assert "original_sizes" in result
        assert "scales" in result
        assert "batch_size" in result

        # Check tensor shapes
        assert result["images"].shape == (2, 3, 512, 512)
        assert result["masks"].shape == (2, 1, 512, 512)

        # Check other fields
        assert len(result["texts"]) == 2
        assert len(result["original_sizes"]) == 2
        assert len(result["scales"]) == 2
        assert result["batch_size"] == 2

    def test_collate_empty_batch(self):
        """Test collate function with empty batch."""
        result = inpainting_collate_fn([])

        # Should return dummy batch
        assert result["batch_size"] == 1
        assert result["images"].shape == (1, 3, 512, 512)
        assert result["masks"].shape == (1, 1, 512, 512)
        assert len(result["texts"]) == 1

    def test_collate_filtered_batch(self):
        """Test collate function with invalid items that get filtered."""
        batch = [
            {},  # Empty item
            {"images": torch.randn(3, 512, 512)},  # Missing keys
            {"masks": torch.ones(1, 512, 512)},  # Missing keys
        ]

        result = inpainting_collate_fn(batch)

        # Should return dummy batch since all items filtered
        assert result["batch_size"] == 1
        assert "dummy" not in result  # But should be functional dummy


class TestInpaintingDataloaders:
    """Test inpainting dataloader creation."""

    def test_create_inpainting_dataloaders(self, mock_training_data):
        """Test creating inpainting dataloaders."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            mock_training_data(data_path, num_samples=3)

            config = DatasetConfig(train_data_path=data_path, crop_size=512, num_workers=0)

            train_loader, val_loader = create_inpainting_dataloaders(config, batch_size=2)

            # Check train loader
            assert train_loader is not None
            assert len(train_loader) >= 1  # At least one batch

            # Check validation loader (should be None since no val data)
            assert val_loader is None

            # Test iteration
            batch = next(iter(train_loader))
            assert batch["images"].shape[0] <= 2  # Batch size
            assert batch["images"].shape[1:] == (3, 512, 512)
            assert batch["masks"].shape[1:] == (1, 512, 512)


class TestDummyBatch:
    """Test dummy batch creation."""

    def test_create_dummy_inpainting_batch(self):
        """Test dummy batch creation."""
        batch = create_dummy_inpainting_batch()

        # Check structure
        assert "images" in batch
        assert "masks" in batch
        assert "texts" in batch
        assert "original_sizes" in batch
        assert "scales" in batch
        assert "batch_size" in batch

        # Check shapes
        assert batch["images"].shape == (1, 3, 512, 512)
        assert batch["masks"].shape == (1, 1, 512, 512)

        # Check values
        assert batch["batch_size"] == 1
        assert len(batch["texts"]) == 1
        assert batch["texts"][0] == "text"
        assert batch["original_sizes"][0] == (512, 512)
        assert batch["scales"][0] == 1.0
