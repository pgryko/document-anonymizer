"""
Robust Dataset Loading and Preprocessing
======================================

This implementation provides safe, efficient dataset loading for document anonymization:
- Comprehensive input validation and sanitization
- Memory-efficient loading with proper error handling
- Robust image preprocessing with bounds checking
- Augmentation pipeline optimized for text preservation
- Proper tensor handling and normalization
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

from src.anonymizer.core.config import DatasetConfig
from src.anonymizer.core.exceptions import (
    BoundingBoxExceedsImageError,
    BoundingBoxOutOfBoundsError,
    BoundingBoxTooSmallError,
    EmptyBatchAfterFilteringError,
    EmptyDatasetError,
    ImageDimensionsTooLargeError,
    ImageDimensionsTooSmallError,
    ImageNotFoundError,
    ImageTooLargeError,
    InsufficientTextRegionsError,
    InvalidImageDataError,
    MissingImageNameError,
    NegativeCoordinatesError,
    NoAnnotationFilesError,
    NoValidSamplesError,
    NoValidTextRegionsError,
    PreprocessingError,
    ScaledBoundingBoxTooSmallError,
    TextTooLongError,
    TextTooShortError,
    UnexpectedImageDtypeError,
    UnexpectedImageShapeError,
    UnsupportedImageFormatError,
    ValidationError,
)
from src.anonymizer.core.models import BoundingBox, TextRegion

logger = logging.getLogger(__name__)

# Dataset constants
EXPECTED_IMAGE_DIMENSIONS = 3
EXPECTED_RGB_CHANNELS = 3


# Helper functions for TRY301 compliance
def _validate_file_size(file_size: int, max_size: int) -> None:
    """Validate file size is within limits."""
    if file_size > max_size:
        raise ImageTooLargeError(file_size, max_size)


def _validate_image_format(img_format: str, allowed_formats: set[str]) -> None:
    """Validate image format is supported."""
    if img_format not in allowed_formats:
        raise UnsupportedImageFormatError(img_format)


def _validate_image_dimensions(width: int, height: int, max_dim: int, min_dim: int) -> None:
    """Validate image dimensions are within limits."""
    if width > max_dim or height > max_dim:
        raise ImageDimensionsTooLargeError(width, height, max_dim)
    if width < min_dim or height < min_dim:
        raise ImageDimensionsTooSmallError(width, height, min_dim)


def _validate_image_array_properties(image_array: np.ndarray) -> None:
    """Validate image array has expected properties."""
    if image_array.dtype != np.uint8:
        raise UnexpectedImageDtypeError(str(image_array.dtype))
    if (
        len(image_array.shape) != EXPECTED_IMAGE_DIMENSIONS
        or image_array.shape[2] != EXPECTED_RGB_CHANNELS
    ):
        raise UnexpectedImageShapeError(image_array.shape)


def _validate_image_name(image_name: str | None) -> None:
    """Validate image name is present."""
    if not image_name:
        raise MissingImageNameError()


def _validate_dataset_not_empty(samples: list) -> None:
    """Validate dataset has samples."""
    if len(samples) == 0:
        raise EmptyDatasetError()


@dataclass
class DatasetSample:
    """Single dataset sample with validation."""

    image_path: Path
    image: np.ndarray
    text_regions: list[TextRegion]

    def __post_init__(self):
        """Validate sample after initialization."""
        if not self.image_path.exists():
            raise ImageNotFoundError(str(self.image_path))

        if len(self.text_regions) == 0:
            raise InsufficientTextRegionsError()

        h, w = self.image.shape[:2]
        for region in self.text_regions:
            if region.bbox.right > w or region.bbox.bottom > h:
                raise BoundingBoxOutOfBoundsError(str(region.bbox))


class ImageValidator:
    """Validates image inputs for security and correctness."""

    # Security limits
    MAX_IMAGE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_DIMENSION = 8192  # 8K resolution limit
    MIN_DIMENSION = 32  # Minimum usable size
    ALLOWED_FORMATS: ClassVar[set[str]] = {"JPEG", "PNG", "TIFF", "BMP"}

    @classmethod
    def validate_image_file(cls, image_path: Path) -> bool:
        """Validate image file for security and format."""
        try:
            # Check file size
            file_size = image_path.stat().st_size
            _validate_file_size(file_size, cls.MAX_IMAGE_SIZE)

            # Load and validate image
            with Image.open(image_path) as img:
                # Check format
                _validate_image_format(img.format, cls.ALLOWED_FORMATS)

                # Check dimensions
                width, height = img.size
                _validate_image_dimensions(width, height, cls.MAX_DIMENSION, cls.MIN_DIMENSION)

                # Validate image data
                img.verify()

            return True

        except UnidentifiedImageError:
            raise InvalidImageDataError()
        except Exception as e:
            raise ValidationError(f"Image validation failed: {e}")

    @classmethod
    def load_image_safely(cls, image_path: Path) -> np.ndarray:
        """Load image with validation and conversion."""
        try:
            cls.validate_image_file(image_path)
        except ValidationError as e:
            raise PreprocessingError(f"Failed to load image: {e}")

        try:
            # Load image
            image = Image.open(image_path)

            # Convert to RGB if needed
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Convert to numpy array
            image_array = np.array(image)

            # Validate array properties
            _validate_image_array_properties(image_array)

            return image_array

        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            raise PreprocessingError(f"Failed to load image {image_path}: {e}")


class TextRegionValidator:
    """Validates text regions and annotations."""

    MAX_TEXT_LENGTH = 1000
    MIN_TEXT_LENGTH = 1
    MIN_BBOX_SIZE = 10  # Minimum bbox dimension

    @classmethod
    def validate_text_region(cls, region: TextRegion, image_shape: tuple[int, int]) -> bool:
        """Validate text region against image."""
        h, w = image_shape

        # Validate text content
        if len(region.original_text) < cls.MIN_TEXT_LENGTH:
            raise TextTooShortError("Original")

        if len(region.original_text) > cls.MAX_TEXT_LENGTH:
            raise TextTooLongError("Original")

        if len(region.replacement_text) < cls.MIN_TEXT_LENGTH:
            raise TextTooShortError("Replacement")

        if len(region.replacement_text) > cls.MAX_TEXT_LENGTH:
            raise TextTooLongError("Replacement")

        # Validate bounding box
        bbox = region.bbox
        if bbox.left < 0 or bbox.top < 0:
            raise NegativeCoordinatesError()

        if bbox.right > w or bbox.bottom > h:
            raise BoundingBoxExceedsImageError()

        if bbox.width < cls.MIN_BBOX_SIZE or bbox.height < cls.MIN_BBOX_SIZE:
            raise BoundingBoxTooSmallError()

        return True


class SafeAugmentation:
    """Safe augmentation pipeline for text preservation."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.rng = random.Random(42)  # Fixed seed for deterministic behavior

    def augment_image(
        self, image: np.ndarray, text_regions: list[TextRegion]
    ) -> tuple[np.ndarray, list[TextRegion]]:
        """Apply safe augmentations that preserve text readability."""
        try:
            # Convert to PIL for easier manipulation
            pil_image = Image.fromarray(image)
            original_size = pil_image.size

            # Apply brightness adjustment (conservative)
            if self.config.brightness_range > 0:
                brightness_factor = self.rng.uniform(
                    1 - self.config.brightness_range, 1 + self.config.brightness_range
                )
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(brightness_factor)

            # Apply contrast adjustment (conservative)
            if self.config.contrast_range > 0:
                contrast_factor = self.rng.uniform(
                    1 - self.config.contrast_range, 1 + self.config.contrast_range
                )
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(contrast_factor)

            # Skip rotation to preserve image dimensions and avoid coordinate transformation
            # This is a conservative approach that preserves text alignment
            if self.config.rotation_range > 0:
                # For now, we skip rotation to avoid complex coordinate transformation
                # and dimension changes that would break the training pipeline
                pass

            # Ensure image dimensions haven't changed
            if pil_image.size != original_size:
                pil_image = pil_image.resize(original_size, Image.LANCZOS)

            # Convert back to numpy
            augmented_image = np.array(pil_image)

            # Return augmented image with original text regions
            # (text regions unchanged for conservative augmentation)
            return augmented_image, text_regions

        except Exception as e:
            logger.warning(f"Augmentation failed, using original: {e}")
            return image, text_regions


class AnonymizerDataset(Dataset):
    """
    Robust dataset for document anonymization training.

    Features:
    - Comprehensive input validation
    - Memory-efficient loading
    - Safe augmentation pipeline
    - Proper error handling and recovery
    """

    def __init__(
        self,
        data_dir: Path,
        config: DatasetConfig,
        split: str = "train",
        transform: Any | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        self.transform = transform

        # Initialize components
        self.image_validator = ImageValidator()
        self.text_validator = TextRegionValidator()
        self.augmentation = SafeAugmentation(config) if split == "train" else None

        # Load dataset
        self.samples = self._load_dataset()

        logger.info(f"Loaded {len(self.samples)} samples for {split} split")

    def _load_dataset(self) -> list[DatasetSample]:
        """Load and validate dataset samples."""
        samples = []

        # Find annotation files
        annotation_files = list(self.data_dir.glob("*.json"))

        if not annotation_files:
            raise NoAnnotationFilesError(str(self.data_dir))

        for annotation_file in annotation_files:
            try:
                sample = self._load_sample(annotation_file)
                if sample:
                    samples.append(sample)
            except Exception:
                logger.exception(f"Failed to load sample {annotation_file}")
                continue

        if not samples:
            raise NoValidSamplesError()

        return samples

    def _load_sample(self, annotation_file: Path) -> DatasetSample | None:
        """Load single sample with validation."""
        try:
            # Load annotations
            with annotation_file.open() as f:
                data = json.load(f)

            # Get image path
            image_name = data.get("image_name")
            _validate_image_name(image_name)

            image_path = self.data_dir / image_name

            # Load and validate image
            image = self.image_validator.load_image_safely(image_path)

            # Parse text regions
            text_regions = []
            for region_data in data.get("text_regions", []):
                # Parse bounding box
                bbox_data = region_data.get("bbox", {})
                bbox = BoundingBox(
                    left=bbox_data["left"],
                    top=bbox_data["top"],
                    right=bbox_data["right"],
                    bottom=bbox_data["bottom"],
                )

                # Create text region
                region = TextRegion(
                    bbox=bbox,
                    original_text=region_data["original_text"],
                    replacement_text=region_data["replacement_text"],
                    confidence=region_data.get("confidence", 1.0),
                )

                # Validate region
                self.text_validator.validate_text_region(region, image.shape[:2])
                text_regions.append(region)

            # Create sample
            return DatasetSample(image_path=image_path, image=image, text_regions=text_regions)

        except Exception:
            logger.exception(f"Failed to load sample {annotation_file}")
            return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get dataset item with preprocessing and robust error handling."""
        max_retries = 3

        for attempt in range(max_retries):
            try:
                # Check for empty samples list
                _validate_dataset_not_empty(self.samples)

                # Use modulo to handle out-of-bounds indices
                actual_idx = idx % len(self.samples)
                sample = self.samples[actual_idx]

                # Get image and text regions
                image = sample.image.copy()
                text_regions = sample.text_regions.copy()

                # Apply augmentation if training
                if self.augmentation and self.split == "train":
                    image, text_regions = self.augmentation.augment_image(image, text_regions)

                # Prepare training data
                result = self._prepare_training_data(image, text_regions)

                # Validate result has valid data
                if (
                    result
                    and "images" in result
                    and "masks" in result
                    and result["masks"].shape[0] > 0
                ):
                    return result
                logger.warning(f"Invalid result for sample {actual_idx}, retrying...")

            except Exception:
                # Use idx as fallback if actual_idx wasn't set
                error_idx = locals().get("actual_idx", idx)
                logger.exception(f"Failed to get item {error_idx} (attempt {attempt + 1})")

                # Try next sample on error
                if attempt < max_retries - 1 and len(self.samples) > 0:
                    idx = (idx + 1) % len(self.samples)
                    continue

        # If all retries failed, return empty item (will be filtered out)
        logger.error(f"All retries failed for index {idx}, returning empty item")
        return {}

    def _prepare_training_data(
        self, image: np.ndarray, text_regions: list[TextRegion]
    ) -> dict[str, Any]:
        """Prepare data for training."""
        try:
            # Resize image to target size
            target_size = self.config.crop_size
            h, w = image.shape[:2]

            # Calculate scaling
            scale = min(target_size / w, target_size / h)
            new_w, new_h = int(w * scale), int(h * scale)

            # Resize image
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # Pad to target size
            pad_w = target_size - new_w
            pad_h = target_size - new_h
            padded_image = np.pad(
                resized_image,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=255,  # White padding
            )

            # Create individual masks for each text region
            masks = []
            texts = []
            valid_regions = []

            for region in text_regions:
                # Scale bounding box with proper rounding
                scaled_bbox = region.bbox.scale(scale)

                # Clamp coordinates to valid bounds
                clamped_bbox = scaled_bbox.clamp_to_bounds(target_size, target_size)

                # Validate bbox is still meaningful after clamping
                if clamped_bbox.width <= 0 or clamped_bbox.height <= 0:
                    logger.warning(ScaledBoundingBoxTooSmallError(str(clamped_bbox)).args[0])
                    continue

                # Create individual mask for this region
                region_mask = np.zeros((target_size, target_size), dtype=np.float32)
                region_mask[
                    clamped_bbox.top : clamped_bbox.bottom,
                    clamped_bbox.left : clamped_bbox.right,
                ] = 1.0

                masks.append(region_mask)
                texts.append(region.replacement_text)
                valid_regions.append(region)

            # If no valid text regions, create a dummy mask
            if not masks:
                masks.append(np.zeros((target_size, target_size), dtype=np.float32))
                texts.append("")
                logger.warning(NoValidTextRegionsError().args[0])

            # Convert to tensors
            image_tensor = torch.from_numpy(padded_image).permute(2, 0, 1).float() / 255.0
            image_tensor = (image_tensor - 0.5) / 0.5  # Normalize to [-1, 1]

            # Stack masks into tensor (num_regions, height, width)
            mask_tensor = torch.from_numpy(np.stack(masks, axis=0)).float()

            return {
                "images": image_tensor,
                "masks": mask_tensor,
                "texts": texts,
                "original_size": (h, w),
                "scale": scale,
            }

        except Exception as e:
            raise PreprocessingError(f"Failed to prepare training data: {e}")


def create_dummy_batch() -> dict[str, Any]:
    """Create a minimal valid batch for error recovery."""
    dummy_image = torch.zeros(1, 3, 512, 512, dtype=torch.float32)
    dummy_mask = torch.zeros(1, 1, 512, 512, dtype=torch.float32)

    return {
        "images": dummy_image,
        "masks": dummy_mask,
        "texts": [""],
        "original_sizes": [(512, 512)],
        "scales": [1.0],
        "batch_size": 1,
        "text_mask_indices": [(0, 0)],  # Track text-mask alignment
    }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Custom collate function for batch processing with improved error handling."""
    # Handle completely empty batch (no items at all)
    if not batch:
        logger.warning("Empty batch received, returning dummy batch")
        return create_dummy_batch()

    # Filter out empty items and items with no valid text regions
    valid_batch = [
        item
        for item in batch
        if item and len(item.get("texts", [])) > 0 and item["masks"].shape[0] > 0
    ]

    # If all items were filtered out (all items were invalid)
    if not valid_batch:
        raise EmptyBatchAfterFilteringError()

    # Separate batch components
    images = torch.stack([item["images"] for item in valid_batch])

    # Handle masks - pad to same number of regions if needed
    masks_list = [item["masks"] for item in valid_batch]
    max_regions = max(mask.shape[0] for mask in masks_list)

    # Pad masks to same number of regions
    padded_masks = []
    for mask in masks_list:
        if mask.shape[0] < max_regions:
            # Pad with zeros
            padding = torch.zeros(
                max_regions - mask.shape[0],
                mask.shape[1],
                mask.shape[2],
                dtype=mask.dtype,
            )
            padded_mask = torch.cat([mask, padding], dim=0)
        else:
            padded_mask = mask
        padded_masks.append(padded_mask)

    masks = torch.stack(padded_masks)

    # Preserve text-mask alignment
    texts = [item["texts"] for item in valid_batch]  # List of lists
    original_sizes = [item["original_size"] for item in valid_batch]
    scales = [item["scale"] for item in valid_batch]

    # Create text-mask alignment indices and flatten texts
    flat_texts = []
    text_mask_indices = []

    for batch_idx, text_list in enumerate(texts):
        for region_idx, text in enumerate(text_list):
            flat_texts.append(text)
            text_mask_indices.append((batch_idx, region_idx))

    return {
        "images": images,
        "masks": masks,
        "texts": flat_texts,
        "original_sizes": original_sizes,
        "scales": scales,
        "batch_size": len(valid_batch),
        "text_mask_indices": text_mask_indices,  # Track which mask each text belongs to
    }


def create_dataloader(
    dataset: AnonymizerDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create data loader with proper configuration."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def create_datasets(
    config: DatasetConfig,
) -> tuple[AnonymizerDataset, AnonymizerDataset | None]:
    """Create train and validation datasets."""
    # Create training dataset
    train_dataset = AnonymizerDataset(data_dir=config.train_data_path, config=config, split="train")

    # Create validation dataset if path provided
    val_dataset = None
    if config.val_data_path and config.val_data_path.exists():
        val_dataset = AnonymizerDataset(data_dir=config.val_data_path, config=config, split="val")

    return train_dataset, val_dataset


def create_dataloaders(
    config: DatasetConfig,
    batch_size: int = 32,
) -> tuple[DataLoader, DataLoader | None]:
    """Create train and validation data loaders."""
    train_dataset, val_dataset = create_datasets(config)

    # Create train dataloader
    train_dataloader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    # Create validation dataloader
    val_dataloader = None
    if val_dataset:
        val_dataloader = create_dataloader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.num_workers,
        )

    return train_dataloader, val_dataloader
