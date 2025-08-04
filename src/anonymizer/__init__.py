"""Document Anonymization System
============================

A production-ready document anonymization system using diffusion models
for replacing sensitive text in financial documents.

This implementation fixes critical bugs in the reference implementations:
- Adds missing KL divergence loss to VAE training
- Uses proper hyperparameters for stable training
- Implements robust error handling and validation
- Provides Modal.com GPU training integration
- Uses Cloudflare R2 for model storage
"""

__version__ = "1.0.0"
__author__ = "Document Anonymizer Team"

from .core.config import EngineConfig, UNetConfig, VAEConfig
from .core.exceptions import AnonymizerError, InferenceError, TrainingError
from .core.models import (
    AnonymizationRequest,
    AnonymizationResult,
    FontInfo,
    GeneratedPatch,
    ProcessedImage,
)
from .fonts import FontManager, FontMetadata

__all__ = [
    "AnonymizationRequest",
    "AnonymizationResult",
    "AnonymizerError",
    "EngineConfig",
    "FontInfo",
    "FontManager",
    "FontMetadata",
    "GeneratedPatch",
    "InferenceError",
    "ProcessedImage",
    "TrainingError",
    "UNetConfig",
    "VAEConfig",
]
