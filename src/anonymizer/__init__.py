"""
Document Anonymization System
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

from .core.models import (
    AnonymizationRequest,
    AnonymizationResult,
    ProcessedImage,
    GeneratedPatch,
)
from .core.config import VAEConfig, UNetConfig, EngineConfig
from .core.exceptions import AnonymizerError, TrainingError, InferenceError

__all__ = [
    "AnonymizationRequest",
    "AnonymizationResult",
    "ProcessedImage",
    "GeneratedPatch",
    "VAEConfig",
    "UNetConfig",
    "EngineConfig",
    "AnonymizerError",
    "TrainingError",
    "InferenceError",
]
