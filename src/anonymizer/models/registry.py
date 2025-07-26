"""
Model Registry
==============

Manages a registry of available models for document anonymization.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from .config import ModelFormat, ModelMetadata, ModelSource, ModelType

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry of available and recommended models for document anonymization.

    Features:
    - Predefined model configurations
    - Model recommendations based on use case
    - Version tracking and updates
    - Custom model registration
    """

    def __init__(self, registry_path: Path | None = None):
        self.registry_path = registry_path or Path("models/registry.json")
        self._models: dict[str, ModelSource] = {}
        self._metadata: dict[str, ModelMetadata] = {}

        # Load predefined models
        self._load_predefined_models()

        # Load custom registry if exists
        if self.registry_path.exists():
            self._load_registry()

    def _load_predefined_models(self):
        """Load predefined model configurations."""

        # Stable Diffusion 2.0 Inpainting - Main pipeline
        self._models["sd2-inpainting"] = ModelSource(
            name="sd2-inpainting",
            url="https://huggingface.co/stabilityai/stable-diffusion-2-inpainting",
            format=ModelFormat.DIFFUSERS,
            size_mb=5000,  # Approximate
            description="Stable Diffusion 2.0 inpainting pipeline for document anonymization",
            license="CreativeML Open RAIL++-M License",
        )

        # Individual VAE model
        self._models["sd2-vae"] = ModelSource(
            name="sd2-vae",
            url="https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/vae/diffusion_pytorch_model.safetensors",
            format=ModelFormat.SAFETENSORS,
            size_mb=335,
            checksum="b203b5f2c45806e17d0f92efb8fd0d18c0a46c0bb18a7e78d8b3e19fe07e6a48",
            checksum_type="sha256",
            description="VAE decoder for Stable Diffusion 2.1",
        )

        # Individual UNet model
        self._models["sd2-unet"] = ModelSource(
            name="sd2-unet",
            url="https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/unet/diffusion_pytorch_model.safetensors",
            format=ModelFormat.SAFETENSORS,
            size_mb=3500,
            description="UNet model for Stable Diffusion 2.0 inpainting",
        )

        # Text encoder
        self._models["sd2-text-encoder"] = ModelSource(
            name="sd2-text-encoder",
            url="https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/resolve/main/text_encoder/pytorch_model.bin",
            format=ModelFormat.PYTORCH,
            size_mb=492,
            description="CLIP text encoder for Stable Diffusion 2.0",
        )

        # Alternative lightweight models for faster inference
        self._models["sd1.5-inpainting"] = ModelSource(
            name="sd1.5-inpainting",
            url="https://huggingface.co/runwayml/stable-diffusion-inpainting",
            format=ModelFormat.DIFFUSERS,
            size_mb=4000,
            description="Stable Diffusion 1.5 inpainting - faster but lower quality",
            license="CreativeML Open RAIL-M License",
        )

        # Custom trained models (placeholders)
        self._models["anonymizer-vae-v1"] = ModelSource(
            name="anonymizer-vae-v1",
            url="https://example.com/anonymizer-vae-v1.safetensors",
            format=ModelFormat.SAFETENSORS,
            size_mb=335,
            description="Custom VAE trained for document anonymization",
        )

        self._models["anonymizer-unet-v1"] = ModelSource(
            name="anonymizer-unet-v1",
            url="https://example.com/anonymizer-unet-v1.safetensors",
            format=ModelFormat.SAFETENSORS,
            size_mb=3500,
            description="Custom UNet trained for document anonymization",
        )

        logger.info(f"Loaded {len(self._models)} predefined models")

    def _load_registry(self):
        """Load custom model registry from disk."""
        try:
            with open(self.registry_path) as f:
                data = json.load(f)

            # Load custom models
            if "models" in data:
                for model_data in data["models"]:
                    source = ModelSource(**model_data)
                    self._models[source.name] = source

            # Load metadata
            if "metadata" in data:
                for meta_data in data["metadata"]:
                    metadata = ModelMetadata.from_dict(meta_data)
                    self._metadata[metadata.name] = metadata

            logger.info(f"Loaded registry from {self.registry_path}")

        except Exception as e:
            logger.warning(f"Failed to load registry: {e}")

    def _save_registry(self):
        """Save custom registry to disk."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data for serialization
            data = {
                "models": [],
                "metadata": [],
                "last_updated": datetime.now().isoformat(),
            }

            # Save custom models (not predefined ones)
            predefined_names = {
                "sd2-inpainting",
                "sd2-vae",
                "sd2-unet",
                "sd2-text-encoder",
                "sd1.5-inpainting",
                "anonymizer-vae-v1",
                "anonymizer-unet-v1",
            }

            for name, source in self._models.items():
                if name not in predefined_names:
                    data["models"].append(
                        {
                            "name": source.name,
                            "url": source.url,
                            "format": source.format.value,
                            "size_mb": source.size_mb,
                            "checksum": source.checksum,
                            "checksum_type": source.checksum_type,
                            "requires_auth": source.requires_auth,
                            "license": source.license,
                            "description": source.description,
                        }
                    )

            # Save metadata
            for metadata in self._metadata.values():
                data["metadata"].append(metadata.to_dict())

            with open(self.registry_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Registry saved to {self.registry_path}")

        except Exception as e:
            logger.exception(f"Failed to save registry: {e}")

    def register_model(self, source: ModelSource) -> bool:
        """Register a custom model."""
        try:
            self._models[source.name] = source
            self._save_registry()
            logger.info(f"Registered model: {source.name}")
            return True
        except Exception as e:
            logger.exception(f"Failed to register model {source.name}: {e}")
            return False

    def register_metadata(self, metadata: ModelMetadata) -> bool:
        """Register model metadata."""
        try:
            self._metadata[metadata.name] = metadata
            self._save_registry()
            logger.info(f"Registered metadata for: {metadata.name}")
            return True
        except Exception as e:
            logger.exception(f"Failed to register metadata for {metadata.name}: {e}")
            return False

    def get_model(self, name: str) -> ModelSource | None:
        """Get model source by name."""
        return self._models.get(name)

    def get_metadata(self, name: str) -> ModelMetadata | None:
        """Get model metadata by name."""
        return self._metadata.get(name)

    def list_models(self, model_type: ModelType | None = None) -> list[ModelSource]:
        """List available models, optionally filtered by type."""
        models = list(self._models.values())

        if model_type:
            # Filter by type based on name patterns
            filtered_models = []
            for model in models:
                if self._matches_type(model.name, model_type):
                    filtered_models.append(model)
            return filtered_models

        return models

    def _matches_type(self, name: str, model_type: ModelType) -> bool:
        """Check if model name matches the specified type."""
        name_lower = name.lower()

        if model_type == ModelType.VAE:
            return "vae" in name_lower
        if model_type == ModelType.UNET:
            return "unet" in name_lower
        if model_type == ModelType.TEXT_ENCODER:
            return "text" in name_lower or "encoder" in name_lower
        if model_type == ModelType.FULL_PIPELINE:
            return "inpainting" in name_lower or "pipeline" in name_lower

        return True

    def get_recommended_models(self, use_case: str = "default") -> dict[str, ModelSource]:
        """Get recommended models for specific use cases."""

        if use_case == "fast":
            # Fast inference setup
            return {"pipeline": self._models["sd1.5-inpainting"]}

        if use_case == "quality":
            # High quality setup
            return {"pipeline": self._models["sd2-inpainting"]}

        if use_case == "custom":
            # Custom trained models
            return {
                "vae": self._models["anonymizer-vae-v1"],
                "unet": self._models["anonymizer-unet-v1"],
            }

        # default
        # Balanced setup
        return {
            "pipeline": self._models["sd2-inpainting"],
            "vae": self._models["sd2-vae"],
            "unet": self._models["sd2-unet"],
        }

    def search_models(self, query: str) -> list[ModelSource]:
        """Search models by name or description."""
        query_lower = query.lower()
        results = []

        for model in self._models.values():
            if query_lower in model.name.lower() or (
                model.description and query_lower in model.description.lower()
            ):
                results.append(model)

        return results

    def get_model_info(self, name: str) -> dict[str, str]:
        """Get detailed information about a model."""
        source = self.get_model(name)
        metadata = self.get_metadata(name)

        if not source:
            return {"error": f"Model '{name}' not found"}

        info = {
            "name": source.name,
            "url": source.url,
            "format": source.format.value,
            "description": source.description or "No description available",
            "license": source.license or "Unknown license",
        }

        if source.size_mb:
            info["size"] = f"{source.size_mb} MB"

        if metadata:
            info["local_path"] = str(metadata.local_path)
            info["download_date"] = metadata.download_date or "Unknown"
            info["last_used"] = metadata.last_used or "Never"
            info["usage_count"] = str(metadata.usage_count)

        return info

    def update_usage(self, name: str):
        """Update usage statistics for a model."""
        metadata = self.get_metadata(name)
        if metadata:
            metadata.usage_count += 1
            metadata.last_used = datetime.now().isoformat()
            self._save_registry()

    def cleanup_unused_models(self, days_threshold: int = 30) -> list[str]:
        """Identify models not used in specified days."""
        unused_models = []
        cutoff_date = datetime.now().timestamp() - (days_threshold * 24 * 3600)

        for name, metadata in self._metadata.items():
            if metadata.last_used:
                try:
                    last_used_date = datetime.fromisoformat(metadata.last_used).timestamp()
                    if last_used_date < cutoff_date:
                        unused_models.append(name)
                except ValueError:
                    # Invalid date format, consider unused
                    unused_models.append(name)
            else:
                # Never used
                unused_models.append(name)

        return unused_models

    def export_registry(self, export_path: Path) -> bool:
        """Export current registry to a file."""
        try:
            data = {
                "exported_date": datetime.now().isoformat(),
                "models": [
                    {
                        "name": source.name,
                        "url": source.url,
                        "format": source.format.value,
                        "size_mb": source.size_mb,
                        "description": source.description,
                        "license": source.license,
                    }
                    for source in self._models.values()
                ],
                "metadata": [metadata.to_dict() for metadata in self._metadata.values()],
            }

            export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(export_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Registry exported to {export_path}")
            return True

        except Exception as e:
            logger.exception(f"Failed to export registry: {e}")
            return False
