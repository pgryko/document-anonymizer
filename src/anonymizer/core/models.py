"""Pydantic models for type-safe data structures."""

import json
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from .exceptions import (
    BottomNotGreaterThanTopError,
    EmptyTextRegionsError,
    InvalidStyleError,
    RightNotGreaterThanLeftError,
    TooManyTextRegionsError,
)


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    left: int = Field(..., ge=0, description="Left coordinate")
    top: int = Field(..., ge=0, description="Top coordinate")
    right: int = Field(..., gt=0, description="Right coordinate")
    bottom: int = Field(..., gt=0, description="Bottom coordinate")

    @field_validator("right")
    @classmethod
    def right_greater_than_left(cls, v: int, info: ValidationInfo) -> int:
        if info.data and "left" in info.data and v <= info.data["left"]:
            raise RightNotGreaterThanLeftError()
        return v

    @field_validator("bottom")
    @classmethod
    def bottom_greater_than_top(cls, v: int, info: ValidationInfo) -> int:
        if info.data and "top" in info.data and v <= info.data["top"]:
            raise BottomNotGreaterThanTopError()
        return v

    def scale(self, factor: float) -> "BoundingBox":
        """Scale bounding box by factor with proper rounding."""
        scaled_left = round(self.left * factor)
        scaled_top = round(self.top * factor)
        scaled_right = round(self.right * factor)
        scaled_bottom = round(self.bottom * factor)

        # Ensure minimum size of 1x1
        if scaled_right <= scaled_left:
            scaled_right = scaled_left + 1
        if scaled_bottom <= scaled_top:
            scaled_bottom = scaled_top + 1

        return BoundingBox(
            left=scaled_left,
            top=scaled_top,
            right=scaled_right,
            bottom=scaled_bottom,
        )

    def clamp_to_bounds(self, width: int, height: int) -> "BoundingBox":
        """Clamp bounding box coordinates to image bounds."""
        # Clamp coordinates to valid bounds
        clamped_left = max(0, min(self.left, width - 1))
        clamped_top = max(0, min(self.top, height - 1))
        clamped_right = max(clamped_left + 1, min(self.right, width))
        clamped_bottom = max(clamped_top + 1, min(self.bottom, height))

        return BoundingBox(
            left=clamped_left,
            top=clamped_top,
            right=clamped_right,
            bottom=clamped_bottom,
        )

    @classmethod
    def create_unchecked(cls, left: int, top: int, right: int, bottom: int) -> "BoundingBox":
        """Create BoundingBox without validation (for testing purposes)."""
        return cls.model_construct(left=left, top=top, right=right, bottom=bottom)

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def area(self) -> int:
        return self.width * self.height


class TextRegion(BaseModel):
    """Text region to be anonymized."""

    bbox: BoundingBox
    original_text: str = Field(..., min_length=1, max_length=1000)
    replacement_text: str = Field(..., min_length=1, max_length=1000)
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class AnonymizationRequest(BaseModel):
    """Request for document anonymization."""

    image_data: bytes = Field(..., description="Input image data")
    text_regions: list[TextRegion] = Field(
        default_factory=list,
        description="Text regions to anonymize",
    )
    preserve_formatting: bool = Field(True, description="Preserve text formatting")
    quality_check: bool = Field(True, description="Enable quality verification")

    MAX_TEXT_REGIONS: ClassVar[int] = 50

    @field_validator("text_regions")
    @classmethod
    def validate_text_regions(cls, v: list[TextRegion]) -> list[TextRegion]:
        if len(v) == 0:
            raise EmptyTextRegionsError()
        if len(v) > cls.MAX_TEXT_REGIONS:
            raise TooManyTextRegionsError(cls.MAX_TEXT_REGIONS)
        return v

    class Config:
        arbitrary_types_allowed = True


class ProcessedImage(BaseModel):
    """Processed image data for inference."""

    crop: np.ndarray = Field(..., description="Cropped image region")
    mask: np.ndarray = Field(..., description="Mask for inpainting")
    original_bbox: BoundingBox
    scale_factor: float = Field(1.0, gt=0.0)

    class Config:
        arbitrary_types_allowed = True


class GenerationMetadata(BaseModel):
    """Metadata for generated patch."""

    processing_time_ms: float = Field(..., ge=0.0)
    model_version: str
    num_inference_steps: int = Field(..., ge=1)
    guidance_scale: float = Field(..., gt=0.0)
    seed: int | None = None


class GeneratedPatch(BaseModel):
    """Generated replacement patch."""

    patch: np.ndarray = Field(..., description="Generated image patch")
    confidence: float = Field(..., ge=0.0, le=1.0)
    metadata: GenerationMetadata

    class Config:
        arbitrary_types_allowed = True


class AnonymizationResult(BaseModel):
    """Result of document anonymization."""

    anonymized_image: np.ndarray = Field(..., description="Final anonymized image")
    generated_patches: list[GeneratedPatch]
    processing_time_ms: float = Field(..., ge=0.0)
    success: bool
    errors: list[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class ModelArtifacts(BaseModel):
    """Model artifacts for storage."""

    model_name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    model_path: Path
    config_path: Path
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "model_name": self.model_name,
            "version": self.version,
            "model_path": str(self.model_path),
            "config_path": str(self.config_path),
            "metadata": self.metadata,
        }

    @classmethod
    def from_cache(cls, cache_dir: Path) -> "ModelArtifacts":
        """Load from cache directory."""
        model_path = cache_dir / "model.safetensors"
        config_path = cache_dir / "config.json"
        metadata_path = cache_dir / "metadata.json"

        metadata = {}
        if metadata_path.exists():
            with metadata_path.open() as f:
                metadata = json.load(f)

        return cls(
            model_name=metadata.get("model_name", "unknown"),
            version=metadata.get("version", "unknown"),
            model_path=model_path,
            config_path=config_path,
            metadata=metadata,
        )


class CropData(BaseModel):
    """Data for extracted crop."""

    crop: np.ndarray = Field(..., description="Cropped image")
    scale_factor: float = Field(1.0, gt=0.0)
    relative_bbox: BoundingBox = Field(..., description="Bbox relative to crop")

    class Config:
        arbitrary_types_allowed = True


class TrainingMetrics(BaseModel):
    """Training metrics tracking."""

    epoch: int = Field(..., ge=0)
    step: int = Field(..., ge=0)
    total_loss: float
    recon_loss: float
    kl_loss: float
    perceptual_loss: float | None = None
    learning_rate: float = Field(..., gt=0.0)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging."""
        result = {
            "epoch": float(self.epoch),
            "step": float(self.step),
            "total_loss": self.total_loss,
            "recon_loss": self.recon_loss,
            "kl_loss": self.kl_loss,
            "learning_rate": self.learning_rate,
        }
        if self.perceptual_loss is not None:
            result["perceptual_loss"] = self.perceptual_loss
        return result


class FontInfo(BaseModel):
    """Font information for text rendering."""

    family: str = Field(..., min_length=1, description="Font family name")
    style: str = Field("normal", description="Font style (normal, bold, italic, bold-italic)")
    weight: int = Field(400, ge=100, le=900, description="Font weight")
    size: float = Field(12.0, gt=0.0, description="Font size in points")
    path: str | None = Field(None, description="Path to font file")
    is_bundled: bool = Field(False, description="Whether font is bundled")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="Detection confidence")

    @field_validator("style")
    @classmethod
    def validate_style(cls, v: str) -> str:
        valid_styles = {"normal", "bold", "italic", "bold-italic"}
        if v not in valid_styles:
            raise InvalidStyleError(valid_styles)
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "family": self.family,
            "style": self.style,
            "weight": self.weight,
            "size": self.size,
            "path": self.path,
            "is_bundled": self.is_bundled,
            "confidence": self.confidence,
        }


class BatchItem(BaseModel):
    """Single item in a batch processing request."""

    item_id: str = Field(..., min_length=1, description="Unique identifier for this item")
    image_path: Path = Field(..., description="Path to input image file")
    text_regions: list[TextRegion] = Field(
        default_factory=list,
        max_length=50,
        description="Text regions to anonymize",
    )
    output_path: Path | None = Field(None, description="Optional output path for this item")
    preserve_formatting: bool = Field(True, description="Preserve text formatting")
    quality_check: bool = Field(True, description="Enable quality verification")

    class Config:
        arbitrary_types_allowed = True


class BatchAnonymizationRequest(BaseModel):
    """Request for batch document anonymization."""

    items: list[BatchItem] = Field(..., max_length=100, description="Items to process")
    output_directory: Path = Field(..., description="Base output directory")
    preserve_structure: bool = Field(True, description="Preserve input directory structure")
    max_parallel: int = Field(4, ge=1, le=16, description="Maximum parallel processes")
    batch_size: int = Field(8, ge=1, le=32, description="Batch size for memory management")
    continue_on_error: bool = Field(True, description="Continue processing other items on error")

    class Config:
        arbitrary_types_allowed = True


class BatchItemResult(BaseModel):
    """Result for a single batch item."""

    item_id: str = Field(..., description="Item identifier")
    success: bool = Field(..., description="Whether processing succeeded")
    output_path: Path | None = Field(None, description="Path to output file")
    processing_time_ms: float = Field(..., ge=0.0, description="Processing time in milliseconds")
    generated_patches: list[GeneratedPatch] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list, description="Error messages if any")

    class Config:
        arbitrary_types_allowed = True


class BatchAnonymizationResult(BaseModel):
    """Result of batch document anonymization."""

    results: list[BatchItemResult] = Field(..., description="Results for each item")
    total_items: int = Field(..., ge=0, description="Total number of items processed")
    successful_items: int = Field(..., ge=0, description="Number of successfully processed items")
    failed_items: int = Field(..., ge=0, description="Number of failed items")
    total_processing_time_ms: float = Field(..., ge=0.0, description="Total processing time")
    output_directory: Path = Field(..., description="Output directory used")

    class Config:
        arbitrary_types_allowed = True

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.successful_items / self.total_items) * 100.0

    def get_failed_items(self) -> list[BatchItemResult]:
        """Get list of failed items."""
        return [result for result in self.results if not result.success]

    def get_successful_items(self) -> list[BatchItemResult]:
        """Get list of successful items."""
        return [result for result in self.results if result.success]
