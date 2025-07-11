"""Pydantic models for type-safe data structures."""

from typing import List, Dict, Optional, Any
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, validator


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    left: int = Field(..., ge=0, description="Left coordinate")
    top: int = Field(..., ge=0, description="Top coordinate")
    right: int = Field(..., gt=0, description="Right coordinate")
    bottom: int = Field(..., gt=0, description="Bottom coordinate")

    @validator("right")
    def right_greater_than_left(cls, v, values):
        if "left" in values and v <= values["left"]:
            raise ValueError("right must be greater than left")
        return v

    @validator("bottom")
    def bottom_greater_than_top(cls, v, values):
        if "top" in values and v <= values["top"]:
            raise ValueError("bottom must be greater than top")
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
    text_regions: List[TextRegion] = Field(..., min_items=1, max_items=50)
    preserve_formatting: bool = Field(True, description="Preserve text formatting")
    quality_check: bool = Field(True, description="Enable quality verification")

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
    seed: Optional[int] = None


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
    generated_patches: List[GeneratedPatch]
    processing_time_ms: float = Field(..., ge=0.0)
    success: bool
    errors: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True


class ModelArtifacts(BaseModel):
    """Model artifacts for storage."""

    model_name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    model_path: Path
    config_path: Path
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
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
            import json

            with open(metadata_path) as f:
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
    perceptual_loss: Optional[float] = None
    learning_rate: float = Field(..., gt=0.0)

    def to_dict(self) -> Dict[str, float]:
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
