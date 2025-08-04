"""Model Manager
=============

High-level interface for managing diffusion models including downloading,
validation, and organization.
"""

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.anonymizer.core.exceptions import (
    ModelNotFoundInRegistryError,
    ModelNotFoundLocallyError,
    ModelValidationError,
)

from .config import ModelConfig, ModelMetadata, ModelSource, ModelType, ValidationResult
from .downloader import ModelDownloader
from .registry import ModelRegistry
from .validator import ModelValidator

logger = logging.getLogger(__name__)


class ModelManager:
    """High-level interface for managing diffusion models.

    Features:
    - Model discovery and downloading
    - Automatic validation and verification
    - Storage management and cleanup
    - Model recommendations
    - Version tracking
    """

    def __init__(self, config: ModelConfig | None = None):
        self.config = config or ModelConfig()

        # Initialize components
        self.downloader = ModelDownloader(self.config)
        self.validator = ModelValidator(strict_mode=self.config.strict_validation)
        self.registry = ModelRegistry(self.config.models_dir / "registry.json")

        logger.info(f"ModelManager initialized with models directory: {self.config.models_dir}")

    def download_model(
        self,
        model_name: str,
        progress_callback: Callable[[int, int], None] | None = None,
        validate: bool = True,
    ) -> ModelMetadata:
        """Download a model by name from the registry.

        Args:
            model_name: Name of model in registry
            progress_callback: Optional progress callback
            validate: Whether to validate after download

        Returns:
            ModelMetadata for downloaded model

        """
        # Get model source from registry
        source = self.registry.get_model(model_name)
        if not source:
            raise ModelNotFoundInRegistryError(model_name)

        logger.info(f"Downloading model: {model_name}")

        # Check if already downloaded
        existing_metadata = self.registry.get_metadata(model_name)
        if existing_metadata and existing_metadata.local_path.exists():
            logger.info(
                f"Model '{model_name}' already downloaded to {existing_metadata.local_path}",
            )

            if validate:
                validation = self.validator.validate_model(
                    existing_metadata.local_path,
                    existing_metadata,
                )
                if not validation.valid:
                    logger.warning("Existing model failed validation, re-downloading...")
                else:
                    self.registry.update_usage(model_name)
                    return existing_metadata

        # Download the model
        try:
            metadata = self.downloader.download_model(source, progress_callback=progress_callback)

            # Validate if requested
            if validate:
                validation = self.validator.validate_model(metadata.local_path, metadata)
                if not validation.valid:
                    # Clean up failed download
                    if metadata.local_path.exists():
                        metadata.local_path.unlink()

                    def _raise_validation_error() -> None:
                        raise ModelValidationError()  # noqa: TRY301

                    _raise_validation_error()

                logger.info("Model validation passed")

            # Register metadata
            self.registry.register_metadata(metadata)

            logger.info(f"Successfully downloaded and registered: {model_name}")

        except Exception:
            logger.exception(f"Failed to download model: {model_name}")
            raise
        else:
            return metadata

    def download_from_huggingface(
        self,
        model_id: str,
        filename: str | None = None,
        validate: bool = True,
    ) -> ModelMetadata:
        """Download model directly from Hugging Face Hub.

        Args:
            model_id: Hugging Face model ID
            filename: Specific file to download (optional)
            validate: Whether to validate after download

        Returns:
            ModelMetadata for downloaded model

        """
        try:
            metadata = self.downloader.download_from_huggingface(model_id, filename)

            # Validate if requested
            if validate:
                validation = self.validator.validate_model(metadata.local_path, metadata)
                if not validation.valid:
                    if metadata.local_path.exists():
                        metadata.local_path.unlink()

                    def _raise_validation_error() -> None:
                        raise ModelValidationError()  # noqa: TRY301

                    _raise_validation_error()

            # Register in local registry
            model_name = f"hf_{model_id.replace('/', '_')}"
            if filename:
                model_name += f"_{filename}"

            source = ModelSource(
                name=model_name,
                url=f"https://huggingface.co/{model_id}",
                format=metadata.format,
                description=f"Downloaded from Hugging Face: {model_id}",
            )

            self.registry.register_model(source)
            self.registry.register_metadata(metadata)

        except Exception:
            logger.exception(f"Failed to download from Hugging Face '{model_id}'")
            raise
        else:
            return metadata

    def validate_model(self, model_path: Path) -> ValidationResult:
        """Validate a model file."""
        return self.validator.validate_model(model_path)

    def list_available_models(self, model_type: ModelType | None = None) -> list[ModelSource]:
        """List models available in registry."""
        return self.registry.list_models(model_type)

    def list_downloaded_models(self) -> list[ModelMetadata]:
        """List models that have been downloaded locally."""
        downloaded = []

        for model_name in self.registry._metadata:
            metadata = self.registry.get_metadata(model_name)
            if metadata and metadata.local_path.exists():
                downloaded.append(metadata)

        return downloaded

    def get_model_info(self, model_name: str) -> dict[str, Any]:
        """Get detailed information about a model."""
        info = self.registry.get_model_info(model_name)

        # Add validation status if downloaded
        metadata = self.registry.get_metadata(model_name)
        if metadata and metadata.local_path.exists():
            try:
                validation = self.validator.validate_compatibility(metadata.local_path)
                info["validation_status"] = "valid" if validation.valid else "invalid"
                if validation.errors:
                    info["validation_errors"] = validation.errors
            except Exception:
                info["validation_status"] = "unknown"

        return info

    def get_recommended_setup(self, use_case: str = "default") -> dict[str, ModelMetadata]:
        """Get recommended model setup for specific use cases.

        Args:
            use_case: One of 'default', 'fast', 'quality', 'custom'

        Returns:
            Dictionary mapping component names to ModelMetadata

        """
        recommended_sources = self.registry.get_recommended_models(use_case)
        setup = {}

        for component, source in recommended_sources.items():
            metadata = self.registry.get_metadata(source.name)
            if metadata and metadata.local_path.exists():
                setup[component] = metadata
            else:
                logger.warning(f"Recommended model '{source.name}' not downloaded")

        return setup

    def ensure_models_available(self, use_case: str = "default") -> bool:
        """Ensure all recommended models for a use case are downloaded.

        Args:
            use_case: Use case identifier

        Returns:
            True if all models are available

        """
        recommended_sources = self.registry.get_recommended_models(use_case)
        all_available = True

        for _component, source in recommended_sources.items():
            metadata = self.registry.get_metadata(source.name)

            if not metadata or not metadata.local_path.exists():
                logger.info(f"Downloading missing model: {source.name}")
                try:
                    self.download_model(source.name)
                except Exception:
                    logger.exception(f"Failed to download {source.name}")
                    all_available = False

        return all_available

    def cleanup_models(self, unused_days: int = 30, dry_run: bool = True) -> dict[str, Any]:
        """Clean up unused models to free disk space.

        Args:
            unused_days: Remove models not used in this many days
            dry_run: If True, only report what would be deleted

        Returns:
            Dictionary with cleanup statistics

        """
        unused_models = self.registry.cleanup_unused_models(unused_days)
        total_size = 0
        deleted_count = 0
        errors = []

        for model_name in unused_models:
            metadata = self.registry.get_metadata(model_name)
            if metadata and metadata.local_path.exists():
                try:
                    size = metadata.local_path.stat().st_size
                    total_size += size

                    if not dry_run:
                        if metadata.local_path.is_dir():
                            shutil.rmtree(metadata.local_path)
                        else:
                            metadata.local_path.unlink()

                        # Remove metadata
                        del self.registry._metadata[model_name]
                        self.registry._save_registry()

                        deleted_count += 1
                        logger.info(f"Deleted unused model: {model_name}")

                except Exception as e:
                    errors.append(f"Failed to delete {model_name}: {e}")

        results = {
            "unused_models": unused_models,
            "total_size_mb": total_size / (1024 * 1024),
            "deleted_count": deleted_count,
            "errors": errors,
            "dry_run": dry_run,
        }

        if dry_run:
            logger.info(
                f"Dry run: Would delete {len(unused_models)} models "
                f"({results['total_size_mb']:.1f} MB)",
            )
        else:
            logger.info(
                f"Cleanup completed: Deleted {deleted_count} models "
                f"({results['total_size_mb']:.1f} MB)",
            )

        return results

    def benchmark_model(self, model_name: str) -> dict[str, Any]:
        """Benchmark a downloaded model's performance."""
        metadata = self.registry.get_metadata(model_name)
        if not metadata or not metadata.local_path.exists():
            raise ModelNotFoundLocallyError(model_name)

        return self.validator.benchmark_model(metadata.local_path)

    def verify_all_models(self) -> dict[str, ValidationResult]:
        """Verify all downloaded models."""
        results = {}

        downloaded_models = self.list_downloaded_models()
        for metadata in downloaded_models:
            try:
                validation = self.validator.validate_model(metadata.local_path, metadata)
                results[metadata.name] = validation
            except Exception as e:
                # Create error result
                error_result = ValidationResult(
                    valid=False,
                    model_path=metadata.local_path,
                    errors=[str(e)],
                    warnings=[],
                )
                results[metadata.name] = error_result

        return results

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics for model directory."""
        total_size = 0
        model_count = 0

        try:
            # Calculate total size
            for metadata in self.registry._metadata.values():
                if metadata.local_path.exists():
                    if metadata.local_path.is_file():
                        total_size += metadata.local_path.stat().st_size
                    else:
                        # Directory - sum all files
                        for file_path in metadata.local_path.rglob("*"):
                            if file_path.is_file():
                                total_size += file_path.stat().st_size
                    model_count += 1

            # Get available space
            disk_usage = shutil.disk_usage(self.config.models_dir)

            return {
                "total_models": model_count,
                "total_size_gb": total_size / (1024**3),
                "available_space_gb": disk_usage.free / (1024**3),
                "used_space_gb": disk_usage.used / (1024**3),
                "models_directory": str(self.config.models_dir),
            }

        except Exception as e:
            logger.exception("Failed to get storage stats")
            return {"error": str(e)}

    def export_model_list(self, export_path: Path) -> bool:
        """Export list of models to a file."""
        return self.registry.export_registry(export_path)

    def search_models(self, query: str) -> list[ModelSource]:
        """Search for models by name or description."""
        return self.registry.search_models(query)

    def cleanup(self):
        """Cleanup manager resources."""
        if hasattr(self.downloader, "cleanup"):
            self.downloader.cleanup()
