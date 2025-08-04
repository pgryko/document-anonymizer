"""
Model Validator
===============

Validates downloaded models for integrity, compatibility, and functionality.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import psutil
import safetensors
import torch
from safetensors.torch import load_file

# Optional dependency - handled gracefully
try:
    from diffusers import DiffusionPipeline
except ImportError:
    DiffusionPipeline = None

from .config import ModelFormat, ModelMetadata, ModelType, ValidationResult

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Validates diffusion models for use in document anonymization.

    Features:
    - File integrity validation
    - Format compatibility checking
    - Model loading verification
    - Architecture validation
    - Performance benchmarking
    """

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def validate_model(
        self, model_path: Path, metadata: ModelMetadata | None = None
    ) -> ValidationResult:
        """
        Comprehensive model validation.

        Args:
            model_path: Path to model file
            metadata: Optional model metadata for additional validation

        Returns:
            ValidationResult with detailed validation information
        """
        result = ValidationResult(valid=True, model_path=model_path, errors=[], warnings=[])

        logger.info(f"Validating model: {model_path}")

        try:
            # Basic file validation
            self._validate_file_exists(model_path, result)
            if not result.valid:
                return result

            # File format validation
            self._validate_file_format(model_path, metadata, result)

            # Size validation
            self._validate_file_size(model_path, metadata, result)

            # Checksum validation
            if metadata and metadata.checksum:
                self._validate_checksum(model_path, metadata, result)

            # Content validation
            self._validate_model_content(model_path, metadata, result)

            # Loading validation
            if not result.has_errors:
                self._validate_model_loading(model_path, metadata, result)

            logger.info(f"Validation completed. Valid: {result.valid}")

        except Exception as e:
            result.add_error(f"Validation failed with exception: {e}")
            logger.exception("Model validation failed")

        return result

    def _validate_file_exists(self, model_path: Path, result: ValidationResult):
        """Validate that model file exists and is accessible."""
        if not model_path.exists():
            result.add_error(f"Model file does not exist: {model_path}")
            return

        if not model_path.is_file():
            result.add_error(f"Path is not a file: {model_path}")
            return

        try:
            # Check read permissions
            with model_path.open("rb") as f:
                f.read(1)
        except PermissionError:
            result.add_error(f"No read permission for file: {model_path}")
        except Exception as e:
            result.add_error(f"Cannot access file {model_path}: {e}")

    def _validate_file_format(
        self,
        model_path: Path,
        metadata: ModelMetadata | None,
        result: ValidationResult,
    ):
        """Validate file format compatibility."""
        suffix = model_path.suffix.lower()

        # Determine expected format
        if metadata:
            expected_format = metadata.format
        # Infer from file extension
        elif suffix == ".safetensors":
            expected_format = ModelFormat.SAFETENSORS
        elif suffix in [".pth", ".pt", ".bin"]:
            expected_format = ModelFormat.PYTORCH
        elif model_path.is_dir():
            expected_format = ModelFormat.DIFFUSERS
        else:
            result.add_warning(f"Unknown file format: {suffix}")
            return

        # Validate format-specific requirements
        if expected_format == ModelFormat.SAFETENSORS:
            if suffix != ".safetensors":
                result.add_error(f"Expected .safetensors file, got {suffix}")
            else:
                self._validate_safetensors_format(model_path, result)

        elif expected_format == ModelFormat.PYTORCH:
            if suffix not in [".pth", ".pt", ".bin"]:
                result.add_error(f"Expected PyTorch file (.pth/.pt/.bin), got {suffix}")
            else:
                self._validate_pytorch_format(model_path, result)

        elif expected_format == ModelFormat.DIFFUSERS:
            if not model_path.is_dir():
                result.add_error("Expected directory for diffusers format")
            else:
                self._validate_diffusers_format(model_path, result)

        result.format_valid = not result.has_errors

    def _validate_safetensors_format(self, model_path: Path, result: ValidationResult):
        """Validate SafeTensors format."""
        try:
            # Try to load metadata
            with model_path.open("rb") as f:
                # SafeTensors files start with a JSON header
                header_size = int.from_bytes(f.read(8), "little")
                if header_size <= 0 or header_size > 1024 * 1024:  # Max 1MB header
                    result.add_error("Invalid SafeTensors header size")
                    return

                header_bytes = f.read(header_size)
                try:
                    header = json.loads(header_bytes.decode("utf-8"))
                    if "__metadata__" not in header:
                        result.add_warning("SafeTensors file missing metadata")
                except json.JSONDecodeError:
                    result.add_error("Invalid SafeTensors header JSON")

        except Exception as e:
            result.add_error(f"SafeTensors validation failed: {e}")

    def _validate_pytorch_format(self, model_path: Path, result: ValidationResult):
        """Validate PyTorch format."""
        try:
            # Try to load with torch.load (CPU only for safety)
            checkpoint = torch.load(model_path, map_location="cpu")

            if not isinstance(checkpoint, dict):
                result.add_warning("PyTorch file is not a state dictionary")

        except Exception as e:
            result.add_error(f"PyTorch format validation failed: {e}")

    def _validate_diffusers_format(self, model_path: Path, result: ValidationResult):
        """Validate Diffusers format."""
        required_files = ["model_index.json"]

        for required_file in required_files:
            file_path = model_path / required_file
            if not file_path.exists():
                result.add_error(f"Missing required file: {required_file}")

        # Validate model_index.json
        index_path = model_path / "model_index.json"
        if index_path.exists():
            try:
                with index_path.open() as f:
                    index_data = json.load(f)

                if "_class_name" not in index_data:
                    result.add_warning("model_index.json missing _class_name")

            except json.JSONDecodeError:
                result.add_error("Invalid model_index.json format")

    def _validate_file_size(
        self,
        model_path: Path,
        metadata: ModelMetadata | None,
        result: ValidationResult,
    ):
        """Validate file size."""
        try:
            actual_size = model_path.stat().st_size

            # Check against metadata if available
            if metadata and metadata.size_bytes:
                expected_size = metadata.size_bytes
                size_diff = abs(actual_size - expected_size)

                # Allow 1% difference for filesystem variations
                tolerance = max(1024, expected_size * 0.01)

                if size_diff > tolerance:
                    result.add_error(
                        f"File size mismatch. Expected: {expected_size}, "
                        f"Actual: {actual_size}, Difference: {size_diff}"
                    )
                else:
                    result.size_valid = True

            # Check for reasonable size bounds
            min_size = 1024 * 1024  # 1MB minimum
            max_size = 50 * 1024 * 1024 * 1024  # 50GB maximum

            if actual_size < min_size:
                result.add_warning(f"File seems too small: {actual_size} bytes")
            elif actual_size > max_size:
                result.add_warning(f"File seems very large: {actual_size / 1024**3:.2f}GB")

        except Exception as e:
            result.add_error(f"File size validation failed: {e}")

    def _validate_checksum(
        self, model_path: Path, metadata: ModelMetadata, result: ValidationResult
    ):
        """Validate file checksum."""
        if not metadata.checksum:
            return

        try:

            # Determine hash algorithm (assume SHA256 if not specified)
            hasher = hashlib.sha256()

            with model_path.open("rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)

            actual_checksum = hasher.hexdigest()

            if actual_checksum.lower() != metadata.checksum.lower():
                result.add_error(
                    f"Checksum mismatch. Expected: {metadata.checksum}, "
                    f"Actual: {actual_checksum}"
                )
            else:
                result.checksum_valid = True

        except Exception as e:
            result.add_error(f"Checksum validation failed: {e}")

    def _validate_model_content(
        self,
        model_path: Path,
        metadata: ModelMetadata | None,
        result: ValidationResult,
    ):
        """Validate model content and structure."""
        try:
            if model_path.suffix.lower() == ".safetensors":
                self._validate_safetensors_content(model_path, metadata, result)
            elif model_path.suffix.lower() in [".pth", ".pt", ".bin"]:
                self._validate_pytorch_content(model_path, metadata, result)
            elif model_path.is_dir():
                self._validate_diffusers_content(model_path, metadata, result)

        except Exception as e:
            result.add_error(f"Content validation failed: {e}")

    def _validate_safetensors_content(
        self,
        model_path: Path,
        metadata: ModelMetadata | None,
        result: ValidationResult,
    ):
        """Validate SafeTensors content."""
        try:
            # Load tensor metadata without loading actual tensors
            tensors_info = safetensors.torch.safe_open(model_path, framework="pt")

            tensor_names = list(tensors_info.keys())
            if not tensor_names:
                result.add_error("SafeTensors file contains no tensors")
                return

            # Validate tensor structure based on model type
            if metadata and metadata.model_type:
                self._validate_tensor_structure(tensor_names, metadata.model_type, result)

            logger.debug(f"SafeTensors contains {len(tensor_names)} tensors")

        except Exception as e:
            result.add_error(f"SafeTensors content validation failed: {e}")

    def _validate_pytorch_content(
        self,
        model_path: Path,
        metadata: ModelMetadata | None,
        result: ValidationResult,
    ):
        """Validate PyTorch content."""
        try:
            checkpoint = torch.load(model_path, map_location="cpu")

            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("state_dict", checkpoint)

                tensor_names = list(state_dict.keys())
                if not tensor_names:
                    result.add_error("PyTorch file contains no tensors")
                    return

                # Validate tensor structure
                if metadata and metadata.model_type:
                    self._validate_tensor_structure(tensor_names, metadata.model_type, result)

                logger.debug(f"PyTorch model contains {len(tensor_names)} tensors")

        except Exception as e:
            result.add_error(f"PyTorch content validation failed: {e}")

    def _validate_diffusers_content(
        self,
        model_path: Path,
        _metadata: ModelMetadata | None,
        result: ValidationResult,
    ):
        """Validate Diffusers content."""
        try:
            # Check for expected subdirectories and files
            expected_components = [
                "unet",
                "vae",
                "text_encoder",
                "tokenizer",
                "scheduler",
            ]
            found_components = []

            for component in expected_components:
                component_path = model_path / component
                if component_path.exists():
                    found_components.append(component)

            if not found_components:
                result.add_warning("No expected model components found")
            else:
                logger.debug(f"Found components: {found_components}")

        except Exception as e:
            result.add_error(f"Diffusers content validation failed: {e}")

    def _validate_tensor_structure(
        self, tensor_names: list[str], model_type: ModelType, result: ValidationResult
    ):
        """Validate tensor structure for specific model types."""
        if model_type == ModelType.VAE:
            # VAE should have encoder and decoder components
            has_encoder = any("encoder" in name for name in tensor_names)
            has_decoder = any("decoder" in name for name in tensor_names)

            if not has_encoder:
                result.add_warning("VAE model missing encoder components")
            if not has_decoder:
                result.add_warning("VAE model missing decoder components")

        elif model_type == ModelType.UNET:
            # UNet should have conv and attention layers
            has_conv = any("conv" in name for name in tensor_names)
            has_attention = any("attn" in name for name in tensor_names)

            if not has_conv:
                result.add_warning("UNet model missing conv layers")
            if not has_attention:
                result.add_warning("UNet model missing attention layers")

    def _validate_model_loading(
        self,
        model_path: Path,
        _metadata: ModelMetadata | None,
        result: ValidationResult,
    ):
        """Validate that model can be loaded successfully."""
        try:
            if model_path.suffix.lower() == ".safetensors":
                # Test loading SafeTensors
                tensors = load_file(str(model_path), device="cpu")
                if not tensors:
                    result.add_error("Failed to load tensors from SafeTensors file")
                else:
                    result.loadable = True
                    logger.debug("SafeTensors model loaded successfully")

            elif model_path.suffix.lower() in [".pth", ".pt", ".bin"]:
                # Test loading PyTorch
                checkpoint = torch.load(model_path, map_location="cpu")
                if checkpoint is None:
                    result.add_error("Failed to load PyTorch checkpoint")
                else:
                    result.loadable = True
                    logger.debug("PyTorch model loaded successfully")

            elif model_path.is_dir():
                # Test loading Diffusers pipeline
                try:
                    if DiffusionPipeline is None:
                        return ValidationResult(
                            is_valid=False,
                            errors=["diffusers not available"],
                            metadata=ModelMetadata(
                                name=model_path.name,
                                size_bytes=model_path.stat().st_size,
                                format=ModelFormat.DIFFUSERS,
                                type=ModelType.UNKNOWN,
                            ),
                        )

                    pipeline = DiffusionPipeline.from_pretrained(
                        str(model_path), torch_dtype=torch.float32
                    )
                    if pipeline is None:
                        result.add_error("Failed to load Diffusers pipeline")
                    else:
                        result.loadable = True
                        logger.debug("Diffusers pipeline loaded successfully")

                except ImportError:
                    result.add_warning("Cannot test Diffusers loading - diffusers not available")
                except Exception as e:
                    result.add_error(f"Diffusers loading failed: {e}")

        except Exception as e:
            result.add_error(f"Model loading validation failed: {e}")

    def validate_compatibility(self, model_path: Path) -> ValidationResult:
        """Quick compatibility check for use in inference."""
        result = ValidationResult(valid=True, model_path=model_path, errors=[], warnings=[])

        # Basic checks
        self._validate_file_exists(model_path, result)
        if result.valid:
            self._validate_file_format(model_path, None, result)

        return result

    def benchmark_model(self, model_path: Path) -> dict[str, Any]:
        """Benchmark model loading and inference performance."""
        benchmark_results = {
            "loading_time_ms": 0,
            "memory_usage_mb": 0,
            "inference_time_ms": 0,
            "model_size_mb": 0,
            "device": str(self.device),
        }

        try:

            # Get model size
            if model_path.is_file():
                benchmark_results["model_size_mb"] = model_path.stat().st_size / (1024 * 1024)

            # Get initial memory
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / (1024 * 1024)

            # Benchmark loading
            start_time = time.time()

            if model_path.suffix.lower() == ".safetensors":
                load_file(str(model_path), device="cpu")
            elif model_path.suffix.lower() in [".pth", ".pt", ".bin"]:
                torch.load(model_path, map_location="cpu")

            benchmark_results["loading_time_ms"] = (time.time() - start_time) * 1000

            # Get memory after loading
            final_memory = process.memory_info().rss / (1024 * 1024)
            benchmark_results["memory_usage_mb"] = final_memory - initial_memory

            logger.info(f"Model benchmark completed: {benchmark_results}")

        except Exception as e:
            logger.warning(f"Model benchmarking failed: {e}")

        return benchmark_results
