"""
Model Configuration
===================

Configuration classes for model management, downloading, and validation.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from src.anonymizer.core.exceptions import (
    MaxCacheSizePositiveError,
    MaxWorkersPositiveError,
    ModelNameAndUrlRequiredError,
    TimeoutSecondsPositiveError,
    UnsupportedChecksumTypeError,
)


class ModelType(Enum):
    """Supported model types."""

    VAE = "vae"
    UNET = "unet"
    TEXT_ENCODER = "text_encoder"
    TOKENIZER = "tokenizer"
    SCHEDULER = "scheduler"
    FULL_PIPELINE = "full_pipeline"


class ModelFormat(Enum):
    """Supported model formats."""

    SAFETENSORS = "safetensors"
    PYTORCH = "pytorch"
    DIFFUSERS = "diffusers"
    ONNX = "onnx"
    CHECKPOINT = "checkpoint"


@dataclass
class ModelSource:
    """Model source information."""

    name: str
    url: str
    format: ModelFormat
    size_mb: int | None = None
    checksum: str | None = None
    checksum_type: str = "sha256"
    requires_auth: bool = False
    license: str | None = None
    description: str | None = None

    def __post_init__(self):
        """Validate model source configuration."""
        if not self.name or not self.url:
            raise ModelNameAndUrlRequiredError()

        if self.checksum and self.checksum_type not in ["md5", "sha1", "sha256"]:
            raise UnsupportedChecksumTypeError(self.checksum_type)


@dataclass
class ModelConfig:
    """Configuration for model management."""

    # Storage settings
    models_dir: Path = Path("./models")
    cache_dir: Path = Path("./cache")
    temp_dir: Path = Path("./tmp")

    # Download settings
    max_workers: int = 4
    chunk_size: int = 8192
    timeout_seconds: int = 300
    max_retries: int = 3
    verify_ssl: bool = True

    # Validation settings
    verify_checksums: bool = True
    validate_on_load: bool = True
    strict_validation: bool = False

    # Storage management
    max_cache_size_gb: float = 50.0
    auto_cleanup: bool = True
    keep_compressed: bool = False

    # Security settings
    allow_external_urls: bool = True
    trusted_domains: list[str] = None

    def __post_init__(self):
        """Initialize and validate configuration."""
        if self.trusted_domains is None:
            self.trusted_domains = [
                "huggingface.co",
                "github.com",
                "gitlab.com",
                "gitlab.com",
                "cloudflare.com",
                "s3.amazonaws.com",
                "storage.googleapis.com",
            ]

        # Create directories
        for directory in [self.models_dir, self.cache_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)

        # Validate settings
        if self.max_workers <= 0:
            raise MaxWorkersPositiveError()

        if self.timeout_seconds <= 0:
            raise TimeoutSecondsPositiveError()

        if self.max_cache_size_gb <= 0:
            raise MaxCacheSizePositiveError()


@dataclass
class ModelMetadata:
    """Metadata for downloaded models."""

    name: str
    model_type: ModelType
    format: ModelFormat
    version: str
    source_url: str
    local_path: Path
    size_bytes: int
    checksum: str | None = None
    download_date: str | None = None
    last_used: str | None = None
    usage_count: int = 0
    tags: list[str] | None = None
    description: str | None = None

    def __post_init__(self):
        """Initialize metadata."""
        if self.tags is None:
            self.tags = []

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return {
            "name": self.name,
            "model_type": self.model_type.value,
            "format": self.format.value,
            "version": self.version,
            "source_url": self.source_url,
            "local_path": str(self.local_path),
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "download_date": self.download_date,
            "last_used": self.last_used,
            "usage_count": self.usage_count,
            "tags": self.tags,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        return cls(
            name=data["name"],
            model_type=ModelType(data["model_type"]),
            format=ModelFormat(data["format"]),
            version=data["version"],
            source_url=data["source_url"],
            local_path=Path(data["local_path"]),
            size_bytes=data["size_bytes"],
            checksum=data.get("checksum"),
            download_date=data.get("download_date"),
            last_used=data.get("last_used"),
            usage_count=data.get("usage_count", 0),
            tags=data.get("tags", []),
            description=data.get("description"),
        )


@dataclass
class ValidationResult:
    """Result of model validation."""

    valid: bool
    model_path: Path
    errors: list[str]
    warnings: list[str]
    checksum_valid: bool | None = None
    format_valid: bool | None = None
    size_valid: bool | None = None
    loadable: bool | None = None

    def __post_init__(self):
        """Initialize validation result."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []

    @property
    def has_errors(self) -> bool:
        """Check if validation has errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if validation has warnings."""
        return len(self.warnings) > 0

    def add_error(self, error: str):
        """Add validation error."""
        self.errors.append(error)
        self.valid = False

    def add_warning(self, warning: str):
        """Add validation warning."""
        self.warnings.append(warning)
