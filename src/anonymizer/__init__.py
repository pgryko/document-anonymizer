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

from .core.models import *
from .core.config import *
from .core.exceptions import *

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
