"""Custom exceptions for the document anonymization system."""

from typing import Any


class AnonymizerError(Exception):
    """Base exception for all anonymizer errors."""

    def __init__(self, message: str, details: Any | None = None):
        super().__init__(message)
        self.details = details


class TrainingError(AnonymizerError):
    """Exception raised during model training."""


class InferenceError(AnonymizerError):
    """Exception raised during model inference."""


class ValidationError(AnonymizerError):
    """Exception raised for input validation errors."""


class StorageError(AnonymizerError):
    """Exception raised for storage operation errors."""


class ConfigurationError(AnonymizerError):
    """Exception raised for configuration errors."""


class PatchGenerationError(InferenceError):
    """Exception raised when patch generation fails."""


class TensorCompatibilityError(InferenceError):
    """Exception raised for tensor compatibility issues."""


class ModelLoadError(InferenceError):
    """Exception raised when model loading fails."""


class PreprocessingError(InferenceError):
    """Exception raised during image preprocessing."""


class PostprocessingError(InferenceError):
    """Exception raised during result postprocessing."""
