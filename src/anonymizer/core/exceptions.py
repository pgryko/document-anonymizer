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


class ProcessingError(AnonymizerError):
    """Exception raised during general processing operations."""


# Specific exception classes for TRY003 compliance
class EmptyBatchError(ValidationError):
    """Exception raised when batch request contains no items."""

    def __init__(self):
        super().__init__("No items to process")


class DuplicateItemError(ValidationError):
    """Exception raised when batch request contains duplicate item IDs."""

    def __init__(self):
        super().__init__("Duplicate item IDs found")


class InputFileNotFoundError(ProcessingError):
    """Exception raised when input file is not found."""

    def __init__(self, file_path: str):
        super().__init__(f"Input file not found: {file_path}")


class NoImageFilesError(ValidationError):
    """Exception raised when no image files are found in directory."""

    def __init__(self, directory: str):
        super().__init__(f"No image files found in {directory}")


class ModalNotAvailableError(ImportError):
    """Exception raised when Modal is not available."""

    def __init__(self):
        super().__init__("Modal not available. Install with: pip install modal")


class PathSecurityError(ValidationError):
    """Exception raised for path security violations."""

    def __init__(self, field_name: str, pattern: str, path: str):
        super().__init__(f"Dangerous pattern '{pattern}' found in {field_name}: {path}")


class PathResolutionError(ValidationError):
    """Exception raised when path resolution fails."""

    def __init__(self, field_name: str, error: str):
        super().__init__(f"Path resolution failed for {field_name}: {error}")


class PathNotAllowedError(ValidationError):
    """Exception raised when path is not within allowed directories."""

    def __init__(self, field_name: str, path: str, allowed_bases: list):
        super().__init__(
            f"Path not within allowed directories for {field_name}: {path}. "
            f"Allowed bases: {allowed_bases}"
        )


class PathDepthError(ValidationError):
    """Exception raised when path depth exceeds limit."""

    def __init__(self, field_name: str, path: str):
        super().__init__(f"Path depth too deep in {field_name}: {path}")


class InvalidPathError(ValidationError):
    """Exception raised for general path validation errors."""

    def __init__(self, field_name: str, error: str):
        super().__init__(f"Invalid path in {field_name}: {error}")


class UnsupportedFileExtensionError(ValidationError):
    """Exception raised for unsupported file extensions."""

    def __init__(self, field_name: str, extension: str):
        super().__init__(f"Unsupported model file extension in {field_name}: {extension}")


class BetasValidationError(ValueError):
    """Exception raised for invalid beta values."""

    def __init__(self):
        super().__init__("betas must be [beta1, beta2] with 0 <= beta < 1")


class EmptyCredentialError(ValueError):
    """Exception raised for empty credentials."""

    def __init__(self):
        super().__init__("Credential cannot be empty")


class ShortCredentialError(ValueError):
    """Exception raised for credentials that are too short."""

    def __init__(self):
        super().__init__("Credential too short")


class InvalidEndpointUrlError(ValueError):
    """Exception raised for invalid endpoint URLs."""

    def __init__(self):
        super().__init__("Endpoint URL must start with https:// or http://")


class ConfigFileNotFoundError(ConfigurationError):
    """Exception raised when configuration file is not found."""

    def __init__(self, config_path: str):
        super().__init__(f"Configuration file not found: {config_path}")


class EmptyConfigFileError(ConfigurationError):
    """Exception raised when configuration file is empty."""

    def __init__(self, config_path: str):
        super().__init__(f"Empty configuration file: {config_path}")


class InvalidYamlError(ConfigurationError):
    """Exception raised for invalid YAML content."""

    def __init__(self, config_path: str, error: str):
        super().__init__(f"Invalid YAML in {config_path}: {error}")


class ConfigLoadError(ConfigurationError):
    """Exception raised when configuration loading fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to load configuration: {error}")


class RightNotGreaterThanLeftError(ValueError):
    """Exception raised when right coordinate is not greater than left."""

    def __init__(self):
        super().__init__("right must be greater than left")


class BottomNotGreaterThanTopError(ValueError):
    """Exception raised when bottom coordinate is not greater than top."""

    def __init__(self):
        super().__init__("bottom must be greater than top")


class InvalidStyleError(ValueError):
    """Exception raised for invalid font styles."""

    def __init__(self, valid_styles: set):
        super().__init__(f"Style must be one of {valid_styles}")
