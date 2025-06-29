"""Custom exceptions for the document anonymization system."""

from typing import Optional, Any


class AnonymizerError(Exception):
    """Base exception for all anonymizer errors."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.details = details


class TrainingError(AnonymizerError):
    """Exception raised during model training."""

    pass


class InferenceError(AnonymizerError):
    """Exception raised during model inference."""

    pass


class ValidationError(AnonymizerError):
    """Exception raised for input validation errors."""

    pass


class StorageError(AnonymizerError):
    """Exception raised for storage operation errors."""

    pass


class ConfigurationError(AnonymizerError):
    """Exception raised for configuration errors."""

    pass


class PatchGenerationError(InferenceError):
    """Exception raised when patch generation fails."""

    pass


class TensorCompatibilityError(InferenceError):
    """Exception raised for tensor compatibility issues."""

    pass


class ModelLoadError(InferenceError):
    """Exception raised when model loading fails."""

    pass


class PreprocessingError(InferenceError):
    """Exception raised during image preprocessing."""

    pass


class PostprocessingError(InferenceError):
    """Exception raised during result postprocessing."""

    pass
