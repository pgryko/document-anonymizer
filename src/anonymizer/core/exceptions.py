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


# Dataset-specific exceptions for TRY003 compliance
class ImageNotFoundError(ValidationError):
    """Exception raised when image file is not found."""

    def __init__(self, path: str):
        super().__init__(f"Image not found: {path}")


class InsufficientTextRegionsError(ValidationError):
    """Exception raised when no text regions are provided."""

    def __init__(self):
        super().__init__("At least one text region required")


class BoundingBoxOutOfBoundsError(ValidationError):
    """Exception raised when bounding box exceeds image dimensions."""

    def __init__(self, bbox: str):
        super().__init__(f"Bounding box out of bounds: {bbox}")


class ImageTooLargeError(ValidationError):
    """Exception raised when image file size exceeds limit."""

    def __init__(self, size: int, limit: int):
        super().__init__(f"Image too large: {size} bytes (limit: {limit})")


class UnsupportedImageFormatError(ValidationError):
    """Exception raised for unsupported image formats."""

    def __init__(self, format_name: str):
        super().__init__(f"Unsupported format: {format_name}")


class ImageDimensionsTooLargeError(ValidationError):
    """Exception raised when image dimensions exceed limits."""

    def __init__(self, width: int, height: int, limit: int):
        super().__init__(f"Image too large: {width}x{height} (limit: {limit})")


class ImageDimensionsTooSmallError(ValidationError):
    """Exception raised when image dimensions are too small."""

    def __init__(self, width: int, height: int, min_size: int):
        super().__init__(f"Image too small: {width}x{height} (minimum: {min_size})")


class InvalidImageDataError(ValidationError):
    """Exception raised for corrupted or invalid image data."""

    def __init__(self):
        super().__init__("Invalid image data")


class UnexpectedImageDtypeError(ValidationError):
    """Exception raised for unexpected image data types."""

    def __init__(self, dtype: str):
        super().__init__(f"Unexpected image dtype: {dtype}")


class UnexpectedImageShapeError(ValidationError):
    """Exception raised for unexpected image shapes."""

    def __init__(self, shape: tuple):
        super().__init__(f"Unexpected image shape: {shape}")


class TextTooShortError(ValidationError):
    """Exception raised when text is too short."""

    def __init__(self, text_type: str):
        super().__init__(f"{text_type} text too short")


class TextTooLongError(ValidationError):
    """Exception raised when text is too long."""

    def __init__(self, text_type: str):
        super().__init__(f"{text_type} text too long")


class NegativeCoordinatesError(ValidationError):
    """Exception raised for negative bounding box coordinates."""

    def __init__(self):
        super().__init__("Bounding box has negative coordinates")


class BoundingBoxExceedsImageError(ValidationError):
    """Exception raised when bounding box exceeds image dimensions."""

    def __init__(self):
        super().__init__("Bounding box exceeds image dimensions")


class BoundingBoxTooSmallError(ValidationError):
    """Exception raised when bounding box is too small."""

    def __init__(self):
        super().__init__("Bounding box too small")


class NoAnnotationFilesError(ValidationError):
    """Exception raised when no annotation files are found."""

    def __init__(self, directory: str):
        super().__init__(f"No annotation files found in {directory}")


class NoValidSamplesError(ValidationError):
    """Exception raised when no valid samples are found."""

    def __init__(self):
        super().__init__("No valid samples found")


class MissingImageNameError(ValidationError):
    """Exception raised when image_name is missing from annotation."""

    def __init__(self):
        super().__init__("Missing image_name in annotation")


class EmptyDatasetError(ValueError):
    """Exception raised when dataset has no samples."""

    def __init__(self):
        super().__init__("Dataset has no samples")


class ScaledBoundingBoxTooSmallError(ValidationError):
    """Exception raised when scaled bounding box becomes too small."""

    def __init__(self, bbox: str):
        super().__init__(f"Scaled bbox too small after clamping, skipping: {bbox}")


class NoValidTextRegionsError(ValidationError):
    """Exception raised when no valid text regions are found."""

    def __init__(self):
        super().__init__("No valid text regions found, using dummy mask")


class EmptyBatchAfterFilteringError(ValidationError):
    """Exception raised when batch is empty after filtering."""

    def __init__(self):
        super().__init__("Empty batch after filtering")


# UNet trainer specific exceptions for TRY003 compliance
class DistributedTrainingSetupError(TrainingError):
    """Exception raised when distributed training setup fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to setup distributed training: {error}")


class UNetChannelMismatchError(ValidationError):
    """Exception raised when UNet has unexpected input channels."""

    def __init__(self, actual: int, expected: int = 9):
        super().__init__(f"Expected {expected}-channel UNet, got {actual}")


class UNetInitializationError(ModelLoadError):
    """Exception raised when UNet initialization fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to initialize UNet: {error}")


class VAEInitializationError(ModelLoadError):
    """Exception raised when VAE initialization fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to initialize VAE: {error}")


class VAETrainingFailedError(TrainingError):
    """Exception raised when VAE training fails."""

    def __init__(self, error: str):
        super().__init__(f"VAE training failed: {error}")


class UNetTrainingFailedError(TrainingError):
    """Exception raised when UNet training fails."""

    def __init__(self, error: str):
        super().__init__(f"UNet training failed: {error}")


class VAEEncodingError(TrainingError):
    """Exception raised when VAE encoding fails."""

    def __init__(self, error: str):
        super().__init__(f"VAE encoding failed: {error}")


class VAEDecodingError(TrainingError):
    """Exception raised when VAE decoding fails."""

    def __init__(self, error: str):
        super().__init__(f"VAE decoding failed: {error}")


class TrOCRInitializationError(ModelLoadError):
    """Exception raised when TrOCR initialization fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to initialize TrOCR: {error}")


class NoiseSchedulerInitializationError(ModelLoadError):
    """Exception raised when noise scheduler initialization fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to initialize noise scheduler: {error}")


class TextProjectionSetupError(TrainingError):
    """Exception raised when text projection setup fails."""

    def __init__(self):
        super().__init__("TrOCR and UNet must be initialized before text projection")


class OptimizerInitializationError(TrainingError):
    """Exception raised when optimizer initialization fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to initialize optimizer: {error}")


class LearningRateSchedulerError(TrainingError):
    """Exception raised when learning rate scheduler setup fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to setup learning rate scheduler: {error}")


class TextEmbeddingError(TrainingError):
    """Exception raised when text embedding fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to encode text: {error}")


class TextConditioningError(TrainingError):
    """Exception raised when text conditioning fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to get text conditioning: {error}")


class LatentEncodingError(TrainingError):
    """Exception raised when latent encoding fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to encode to latents: {error}")


class TrainingStepError(TrainingError):
    """Exception raised when a training step fails."""

    def __init__(self, error: str):
        super().__init__(f"Training step failed: {error}")


class ValidationStepError(TrainingError):
    """Exception raised when a validation step fails."""

    def __init__(self, error: str):
        super().__init__(f"Validation step failed: {error}")


class CheckpointSaveError(TrainingError):
    """Exception raised when checkpoint saving fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to save checkpoint: {error}")


class TrainingLoopError(TrainingError):
    """Exception raised when training loop fails."""

    def __init__(self, error: str):
        super().__init__(f"Training loop failed: {error}")


class UNetNotInitializedError(TrainingError):
    """Exception raised when UNet is not initialized."""

    def __init__(self):
        super().__init__("UNet must be initialized before optimizer")


class TrOCRNotInitializedError(TrainingError):
    """Exception raised when TrOCR is not initialized."""

    def __init__(self):
        super().__init__("TrOCR not initialized")


class VAESchedulerNotInitializedError(TrainingError):
    """Exception raised when VAE and noise scheduler are not initialized."""

    def __init__(self):
        super().__init__("VAE and noise scheduler must be initialized")


class ValidationEpochError(TrainingError):
    """Exception raised when validation epoch fails."""

    def __init__(self, error: str):
        super().__init__(f"Validation epoch failed: {error}")


class OptimizerNotInitializedError(TrainingError):
    """Exception raised when optimizer is not initialized."""

    def __init__(self):
        super().__init__("Optimizer must be initialized before scheduler")


class UNetValidationNotInitializedError(TrainingError):
    """Exception raised when UNet is not initialized for validation."""

    def __init__(self):
        super().__init__("UNet not initialized")


class UnsupportedOptimizerError(OptimizerInitializationError):
    """Exception raised for unsupported optimizer types."""

    def __init__(self, optimizer_type: str):
        super().__init__(f"Unsupported optimizer: {optimizer_type}")


class InvalidLossError(TrainingStepError):
    """Exception raised when loss is invalid (NaN or inf)."""

    def __init__(self, loss_value: float):
        super().__init__(f"Invalid loss detected: {loss_value}")


# Additional specific exceptions for common TRY003 patterns


# VAE trainer specific exceptions
class VAENotInitializedError(TrainingError):
    """Exception raised when VAE is not initialized."""

    def __init__(self):
        super().__init__("VAE not initialized")


class VAEOptimizerNotInitializedError(TrainingError):
    """Exception raised when VAE must be initialized before optimizer."""

    def __init__(self):
        super().__init__("VAE must be initialized before optimizer")


class DistributedTrainingSetupFailedError(TrainingError):
    """Exception raised when distributed training setup fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to setup distributed training: {error}")


class InvalidLossDetectedError(TrainingError):
    """Exception raised when invalid loss is detected during training."""

    def __init__(self, total_loss: float, recon_loss: float, kl_loss: float):
        super().__init__(
            f"Invalid loss detected: total={total_loss}, recon={recon_loss}, kl={kl_loss}"
        )


class TrainingStepFailedError(TrainingError):
    """Exception raised when training step fails."""

    def __init__(self, error: str):
        super().__init__(f"Training step failed: {error}")


class CheckpointSaveFailedError(TrainingError):
    """Exception raised when checkpoint saving fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to save checkpoint: {error}")


class ModelArtifactsSaveFailedError(TrainingError):
    """Exception raised when model artifacts saving fails."""

    def __init__(self, error: str):
        super().__init__(f"Failed to save model artifacts: {error}")


# Model downloader specific exceptions
class URLNotAllowedError(ValidationError):
    """Exception raised when URL is not allowed."""

    def __init__(self, url: str):
        super().__init__(f"URL not allowed: {url}")


class DownloadSizeMismatchError(InferenceError):
    """Exception raised when download size doesn't match expected."""

    def __init__(self, expected: int, actual: int):
        super().__init__(f"Download size mismatch: expected {expected}, got {actual}")


class InsufficientDiskSpaceError(InferenceError):
    """Exception raised when insufficient disk space for download."""

    def __init__(self, required_gb: float, available_gb: float):
        super().__init__(
            f"Insufficient disk space. Required: {required_gb:.2f}GB, Available: {available_gb:.2f}GB"
        )


class DownloadFailedAfterRetriesError(InferenceError):
    """Exception raised when download fails after all retry attempts."""

    def __init__(self, max_retries: int, error: str):
        super().__init__(f"Download failed after {max_retries} attempts: {error}")


# Configuration validation exceptions
class ModelNameAndUrlRequiredError(ValidationError):
    """Exception raised when model name and URL are required."""

    def __init__(self):
        super().__init__("Model name and URL are required")


class UnsupportedChecksumTypeError(ValidationError):
    """Exception raised for unsupported checksum types."""

    def __init__(self, checksum_type: str):
        super().__init__(f"Unsupported checksum type: {checksum_type}")


class MaxWorkersPositiveError(ValidationError):
    """Exception raised when max_workers must be positive."""

    def __init__(self):
        super().__init__("max_workers must be positive")


class TimeoutSecondsPositiveError(ValidationError):
    """Exception raised when timeout_seconds must be positive."""

    def __init__(self):
        super().__init__("timeout_seconds must be positive")


class MaxCacheSizePositiveError(ValidationError):
    """Exception raised when max_cache_size_gb must be positive."""

    def __init__(self):
        super().__init__("max_cache_size_gb must be positive")


# HTTP method exceptions
class UnsupportedHTTPMethodError(ValidationError):
    """Exception raised for unsupported HTTP methods."""

    def __init__(self, method: str):
        super().__init__(f"Unsupported HTTP method: {method}")


# Font checksum exceptions
class ChecksumVerificationFailedError(StorageError):
    """Exception raised when font checksum verification fails."""

    def __init__(self, font_file: str):
        super().__init__(f"Checksum verification failed for {font_file}")


# Text rendering exceptions
class EmptyTextError(ValidationError):
    """Exception raised when text cannot be empty."""

    def __init__(self):
        super().__init__("Text cannot be empty")


class TextRenderingFailedError(PreprocessingError):
    """Exception raised when text rendering fails."""

    def __init__(self, error: str):
        super().__init__(f"Text rendering failed: {error}")


class EmptyTextListError(ValidationError):
    """Exception raised when text list cannot be empty."""

    def __init__(self):
        super().__init__("Text list cannot be empty")


# Image operations exceptions
class ImageProcessingError(ProcessingError):
    """Exception raised during image processing operations."""


class ImageDecodeError(ImageProcessingError):
    """Exception raised when image decoding fails."""

    def __init__(self):
        super().__init__("Failed to decode image")


class ImageEncodeError(ImageProcessingError):
    """Exception raised when image encoding fails."""

    def __init__(self):
        super().__init__("Failed to encode image")


class ArrayConversionError(ImageProcessingError):
    """Exception raised when array conversion fails."""

    def __init__(self):
        super().__init__("Failed to convert array")


class PillowDecodeError(ImageProcessingError):
    """Exception raised when Pillow decoding fails."""

    def __init__(self):
        super().__init__("Failed to decode image using Pillow")


class TensorConversionError(ImageProcessingError):
    """Exception raised when tensor conversion fails."""

    def __init__(self):
        super().__init__("Failed to convert to tensor")


class NumpyConversionError(ImageProcessingError):
    """Exception raised when numpy conversion fails."""

    def __init__(self):
        super().__init__("Failed to convert to numpy array")


class ImageResizeError(ImageProcessingError):
    """Exception raised when image resizing fails."""

    def __init__(self):
        super().__init__("Failed to resize image")


class ImagePadError(ImageProcessingError):
    """Exception raised when image padding fails."""

    def __init__(self):
        super().__init__("Failed to pad image")


class ImageCropError(ImageProcessingError):
    """Exception raised when image cropping fails."""

    def __init__(self):
        super().__init__("Failed to crop image")


class ImageNormalizeError(ImageProcessingError):
    """Exception raised when image normalization fails."""

    def __init__(self):
        super().__init__("Failed to normalize image")


class ImageRotationError(ImageProcessingError):
    """Exception raised when image rotation fails."""

    def __init__(self):
        super().__init__("Failed to rotate image")


class ChannelConversionError(ImageProcessingError):
    """Exception raised when channel conversion fails."""

    def __init__(self):
        super().__init__("Failed to convert image channels")


class CompressionError(ImageProcessingError):
    """Exception raised when image compression fails."""

    def __init__(self):
        super().__init__("Failed to compress image")


class DecompressionError(ImageProcessingError):
    """Exception raised when image decompression fails."""

    def __init__(self):
        super().__init__("Failed to decompress image")


# OCR exceptions
class OCREngineError(ProcessingError):
    """Exception raised during OCR engine operations."""


class OCRModelLoadError(OCREngineError):
    """Exception raised when OCR model loading fails."""

    def __init__(self):
        super().__init__("Failed to load OCR model")


class OCRProcessingError(OCREngineError):
    """Exception raised when OCR processing fails."""

    def __init__(self):
        super().__init__("Failed to process OCR")


class OCRBBoxExtractionError(OCREngineError):
    """Exception raised when OCR bounding box extraction fails."""

    def __init__(self):
        super().__init__("Failed to extract bounding boxes from OCR results")


class TesseractError(OCREngineError):
    """Exception raised when Tesseract OCR fails."""

    def __init__(self):
        super().__init__("Tesseract OCR failed")


class EasyOCRError(OCREngineError):
    """Exception raised when EasyOCR fails."""

    def __init__(self):
        super().__init__("EasyOCR failed")


class PaddleOCRError(OCREngineError):
    """Exception raised when PaddleOCR fails."""

    def __init__(self):
        super().__init__("PaddleOCR failed")


class TrOCRProcessingError(OCREngineError):
    """Exception raised when TrOCR processing fails."""

    def __init__(self):
        super().__init__("TrOCR processing failed")


class OCRResultsEmptyError(OCREngineError):
    """Exception raised when OCR returns no results."""

    def __init__(self):
        super().__init__("OCR returned no results")


class OCRBBoxValidationError(OCREngineError):
    """Exception raised when OCR bounding box validation fails."""

    def __init__(self):
        super().__init__("Invalid OCR bounding box format")


# Model and inference exceptions
class ModelDownloadError(ModelLoadError):
    """Exception raised when model download fails."""

    def __init__(self):
        super().__init__("Failed to download model")


class ModelExtractionError(ModelLoadError):
    """Exception raised when model extraction fails."""

    def __init__(self):
        super().__init__("Failed to extract model archive")


class ModelValidationError(ModelLoadError):
    """Exception raised when model validation fails."""

    def __init__(self):
        super().__init__("Model validation failed")


class ModelInstantiationError(ModelLoadError):
    """Exception raised when model instantiation fails."""

    def __init__(self):
        super().__init__("Failed to instantiate model")


class ModelStateLoadError(ModelLoadError):
    """Exception raised when loading model state fails."""

    def __init__(self):
        super().__init__("Failed to load model state")


class TokenizerLoadError(ModelLoadError):
    """Exception raised when tokenizer loading fails."""

    def __init__(self):
        super().__init__("Failed to load tokenizer")


class ProcessorLoadError(ModelLoadError):
    """Exception raised when processor loading fails."""

    def __init__(self):
        super().__init__("Failed to load processor")


class InferenceSetupError(InferenceError):
    """Exception raised when inference setup fails."""

    def __init__(self):
        super().__init__("Failed to setup inference")


class PatchProcessingError(InferenceError):
    """Exception raised when patch processing fails."""

    def __init__(self):
        super().__init__("Failed to process patch")


class ResultPostprocessingError(InferenceError):
    """Exception raised when result postprocessing fails."""

    def __init__(self):
        super().__init__("Failed to postprocess results")


# File I/O exceptions
class FileReadError(StorageError):
    """Exception raised when file reading fails."""

    def __init__(self, file_path: str):
        super().__init__(f"Failed to read file: {file_path}")


class FileWriteError(StorageError):
    """Exception raised when file writing fails."""

    def __init__(self, file_path: str):
        super().__init__(f"Failed to write file: {file_path}")


class FileDeleteError(StorageError):
    """Exception raised when file deletion fails."""

    def __init__(self, file_path: str):
        super().__init__(f"Failed to delete file: {file_path}")


class DirectoryCreateError(StorageError):
    """Exception raised when directory creation fails."""

    def __init__(self, directory: str):
        super().__init__(f"Failed to create directory: {directory}")


class ArchiveExtractionError(StorageError):
    """Exception raised when archive extraction fails."""

    def __init__(self):
        super().__init__("Failed to extract archive")


class ChecksumVerificationError(StorageError):
    """Exception raised when checksum verification fails."""

    def __init__(self):
        super().__init__("Checksum verification failed")


# Training specific exceptions
class DatasetLoadError(TrainingError):
    """Exception raised when dataset loading fails."""

    def __init__(self):
        super().__init__("Failed to load dataset")


class EpochTrainingError(TrainingError):
    """Exception raised when epoch training fails."""

    def __init__(self, epoch: int, error: str):
        super().__init__(f"Epoch {epoch} training failed: {error}")


class ModelSaveError(TrainingError):
    """Exception raised when model saving fails."""

    def __init__(self):
        super().__init__("Failed to save model")


class LossComputationError(TrainingError):
    """Exception raised when loss computation fails."""

    def __init__(self):
        super().__init__("Failed to compute loss")


class BackwardPassError(TrainingError):
    """Exception raised when backward pass fails."""

    def __init__(self):
        super().__init__("Failed to perform backward pass")


class GradientComputationError(TrainingError):
    """Exception raised when gradient computation fails."""

    def __init__(self):
        super().__init__("Failed to compute gradients")


class WeightUpdateError(TrainingError):
    """Exception raised when weight update fails."""

    def __init__(self):
        super().__init__("Failed to update weights")


# Scheduler exceptions
class SchedulerInitializationError(ConfigurationError):
    """Exception raised when scheduler initialization fails."""

    def __init__(self, scheduler_type: str):
        super().__init__(f"Failed to initialize {scheduler_type} scheduler")


class UnsupportedSchedulerError(ConfigurationError):
    """Exception raised for unsupported scheduler types."""

    def __init__(self, scheduler_type: str):
        super().__init__(f"Unsupported scheduler type: {scheduler_type}")


# Network/HTTP exceptions
class NetworkError(AnonymizerError):
    """Exception raised for network-related errors."""


class DownloadError(NetworkError):
    """Exception raised when download fails."""

    def __init__(self):
        super().__init__("Download failed")


class NetworkConnectionError(NetworkError):
    """Exception raised when connection fails."""

    def __init__(self):
        super().__init__("Connection failed")


class NetworkTimeoutError(NetworkError):
    """Exception raised when operation times out."""

    def __init__(self):
        super().__init__("Operation timed out")


class HTTPError(NetworkError):
    """Exception raised for HTTP errors."""

    def __init__(self, status_code: int):
        super().__init__(f"HTTP error: {status_code}")


# Memory exceptions
class AnonymizerMemoryError(AnonymizerError):
    """Exception raised for memory-related errors."""


class OutOfMemoryError(AnonymizerMemoryError):
    """Exception raised when out of memory."""

    def __init__(self):
        super().__init__("Out of memory")


class MemoryLimitExceededError(AnonymizerMemoryError):
    """Exception raised when memory limit is exceeded."""

    def __init__(self, limit: int):
        super().__init__(f"Memory limit exceeded: {limit} MB")


# GPU/CUDA exceptions
class CudaError(AnonymizerError):
    """Exception raised for CUDA-related errors."""


class CudaOutOfMemoryError(CudaError):
    """Exception raised when CUDA runs out of memory."""

    def __init__(self):
        super().__init__("CUDA out of memory")


class CudaDeviceError(CudaError):
    """Exception raised for CUDA device errors."""

    def __init__(self):
        super().__init__("CUDA device error")


# Additional specific exceptions for image dimensions and memory
class ImageMemoryTooLargeError(ValidationError):
    """Exception raised when image memory usage is too large."""

    def __init__(self, memory_bytes: int):
        super().__init__(f"Image too large in memory: {memory_bytes} bytes")


class ScaleFactorTooLargeError(ValidationError):
    """Exception raised when image scale factor is too large."""

    def __init__(self, scale_factor: float):
        super().__init__(f"Scale factor too large: {scale_factor}")


class OutputSizeTooLargeError(ValidationError):
    """Exception raised when output image size would be too large."""

    def __init__(self, width: int, height: int):
        super().__init__(f"Output size too large: {width}x{height}")


class OutputMemoryTooLargeError(ValidationError):
    """Exception raised when output image memory would be too large."""

    def __init__(self, memory_bytes: int):
        super().__init__(f"Output would be too large: {memory_bytes} bytes")


class InvalidCropSizeError(ValidationError):
    """Exception raised for invalid crop dimensions."""

    def __init__(self, width: int, height: int):
        super().__init__(f"Invalid crop size: {width}x{height}")


class CannotPadToSmallerSizeError(ImagePadError):
    """Exception raised when trying to pad to smaller size."""

    def __init__(self):
        super().__init__("Cannot pad to smaller size")


class ColorConversionChannelError(ChannelConversionError):
    """Exception raised when image has wrong channels for color conversion."""

    def __init__(self, conversion_type: str):
        super().__init__(f"{conversion_type} conversion requires correct channel count")


class UnsupportedColorConversionError(ChannelConversionError):
    """Exception raised for unsupported color conversions."""

    def __init__(self, source: str, target: str):
        super().__init__(f"Unsupported conversion: {source} -> {target}")


# Inference engine specific exceptions
class PathOutsideDirectoryError(ValidationError):
    """Exception raised when path is outside allowed directory."""

    def __init__(self, path: str):
        super().__init__(f"Path outside allowed directory: {path}")


class ModelFileNotFoundError(ValidationError):
    """Exception raised when model file is not found."""

    def __init__(self, path: str):
        super().__init__(f"Model file not found: {path}")


class PathNotFileError(ValidationError):
    """Exception raised when path is not a file."""

    def __init__(self, path: str):
        super().__init__(f"Path is not a file: {path}")


class PathValidationFailedError(ValidationError):
    """Exception raised when general path validation fails."""

    def __init__(self, error: str):
        super().__init__(f"Path validation failed: {error}")


class DeviceNotAvailableError(InferenceError):
    """Exception raised when CUDA device is not available."""

    def __init__(self):
        super().__init__("CUDA device not available")


class InferenceMemoryError(InferenceError):
    """Exception raised for inference memory errors."""

    def __init__(self):
        super().__init__("Inference memory error")


class ModelLoadingError(ModelLoadError):
    """Exception raised when model loading fails."""

    def __init__(self):
        super().__init__("Failed to load model")


class MissingOCRResultsError(ValidationError):
    """Exception raised when OCR results are missing."""

    def __init__(self):
        super().__init__("OCR results missing or invalid")


class SecurityValidationError(ValidationError):
    """Exception raised for security validation failures."""

    def __init__(self):
        super().__init__("Security validation failed")


class TensorValidationError(ValidationError):
    """Exception raised for tensor validation failures."""

    def __init__(self):
        super().__init__("Tensor validation failed")


class InferenceSetupFailedError(InferenceError):
    """Exception raised when inference setup fails."""

    def __init__(self):
        super().__init__("Inference setup failed")


class OCRProcessingFailedError(ProcessingError):
    """Exception raised when OCR processing fails."""

    def __init__(self):
        super().__init__("OCR processing failed")


class PatchGenerationFailedError(PatchGenerationError):
    """Exception raised when patch generation fails."""

    def __init__(self):
        super().__init__("Patch generation failed")


class PostprocessingFailedError(PostprocessingError):
    """Exception raised when postprocessing fails."""

    def __init__(self):
        super().__init__("Postprocessing failed")


class ModelNotLoadedError(ModelLoadError):
    """Exception raised when model is not loaded."""

    def __init__(self):
        super().__init__("Model not loaded")


class UnsupportedDeviceError(InferenceError):
    """Exception raised for unsupported devices."""

    def __init__(self):
        super().__init__("Unsupported device")


class InvalidInputImageError(ValidationError):
    """Exception raised for invalid input images."""

    def __init__(self):
        super().__init__("Invalid input image")


class InvalidFontSizeError(ValidationError):
    """Exception raised for invalid font sizes."""

    def __init__(self, font_size: int):
        super().__init__(f"Invalid font size: {font_size}. Must be integer between 1-200")


# OCR validation specific exceptions
class ImageCannotBeNoneError(ValidationError):
    """Exception raised when image is None."""

    def __init__(self):
        super().__init__("Image cannot be None")


class ImageDimensionInvalidError(ValidationError):
    """Exception raised for invalid image dimensions."""

    def __init__(self, dimensions: int):
        super().__init__(f"Image must be 2D or 3D array, got {dimensions}D")


class ImageCannotBeEmptyError(ValidationError):
    """Exception raised when image is empty."""

    def __init__(self):
        super().__init__("Image cannot be empty")


class ImageTooSmallError(ValidationError):
    """Exception raised when image is too small for OCR."""

    def __init__(self, width: int, height: int):
        super().__init__(f"Image too small: {width}x{height}, minimum 10x10")


class OCREngineInitializationError(ModelLoadError):
    """Exception raised when OCR engine initialization fails."""

    def __init__(self):
        super().__init__("OCR engine initialization failed")


class UnsupportedOCREngineError(ValidationError):
    """Exception raised for unsupported OCR engine types."""

    def __init__(self, engine_type: str):
        super().__init__(f"Unsupported OCR engine: {engine_type}")
