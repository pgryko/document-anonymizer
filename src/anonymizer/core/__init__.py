"""Core components for document anonymization."""

from .models import (
    AnonymizationRequest,
    AnonymizationResult,
    ProcessedImage,
    GeneratedPatch,
    BoundingBox,
    ModelArtifacts,
)
from .config import (
    VAEConfig,
    UNetConfig,
    EngineConfig,
    PreprocessingConfig,
    R2Config,
)
from .exceptions import (
    AnonymizerError,
    TrainingError,
    InferenceError,
    ValidationError,
    StorageError,
)

__all__ = [
    "AnonymizationRequest",
    "AnonymizationResult",
    "ProcessedImage",
    "GeneratedPatch",
    "BoundingBox",
    "ModelArtifacts",
    "VAEConfig",
    "UNetConfig",
    "EngineConfig",
    "PreprocessingConfig",
    "R2Config",
    "AnonymizerError",
    "TrainingError",
    "InferenceError",
    "ValidationError",
    "StorageError",
]
