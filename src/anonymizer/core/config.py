"""Configuration management for the document anonymization system."""

from typing import Optional, List, Union
from pathlib import Path
import os
import yaml
from pydantic import BaseModel, Field, validator
from .exceptions import ConfigurationError


class OptimizerConfig(BaseModel):
    """Optimizer configuration."""

    type: str = Field("AdamW", description="Optimizer type")
    learning_rate: float = Field(..., gt=0.0, description="Learning rate")
    weight_decay: float = Field(0.01, ge=0.0, description="Weight decay")
    betas: List[float] = Field([0.9, 0.999], description="Adam betas")

    @validator("betas")
    def validate_betas(cls, v):
        if len(v) != 2 or not all(0.0 <= b < 1.0 for b in v):
            raise ValueError("betas must be [beta1, beta2] with 0 <= beta < 1")
        return v


class SchedulerConfig(BaseModel):
    """Learning rate scheduler configuration."""

    type: str = Field("cosine_with_restarts", description="Scheduler type")
    warmup_steps: int = Field(1000, ge=0, description="Warmup steps")
    num_cycles: int = Field(3, ge=1, description="Number of cycles")
    min_lr_ratio: float = Field(0.1, ge=0.0, le=1.0, description="Minimum LR ratio")


class LossConfig(BaseModel):
    """Loss function configuration."""

    kl_weight: float = Field(0.00025, ge=0.0, description="KL divergence weight")
    perceptual_weight: float = Field(0.1, ge=0.0, description="Perceptual loss weight")
    recon_loss_type: str = Field("mse", description="Reconstruction loss type")


class VAEConfig(BaseModel):
    """VAE training configuration."""

    model_name: str = Field("document-anonymizer-vae", description="Model name")
    version: str = Field("v1.0", description="Model version")
    base_model: str = Field(
        "stabilityai/stable-diffusion-2-1-base", description="Base model"
    )

    # Training parameters (FIXED: Increased from reference implementations)
    batch_size: int = Field(16, ge=1, description="Batch size per GPU")
    learning_rate: float = Field(
        5e-4, gt=0.0, description="Learning rate (FIXED: was 5e-6)"
    )
    num_epochs: int = Field(100, ge=1, description="Number of epochs")
    gradient_accumulation_steps: int = Field(
        2, ge=1, description="Gradient accumulation"
    )
    mixed_precision: str = Field("bf16", description="Mixed precision mode")
    gradient_clipping: float = Field(1.0, gt=0.0, description="Gradient clipping")

    # Loss configuration (CRITICAL FIX: Added KL divergence)
    loss: LossConfig = Field(default_factory=LossConfig)

    # Optimizer and scheduler
    optimizer: OptimizerConfig = Field(
        default_factory=lambda: OptimizerConfig(learning_rate=5e-4)
    )
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    # Storage
    checkpoint_dir: Path = Field(
        Path("/tmp/checkpoints"), description="Checkpoint directory"
    )
    save_every_n_steps: int = Field(5000, ge=1, description="Save frequency")
    keep_n_checkpoints: int = Field(
        3, ge=1, description="Number of checkpoints to keep"
    )

    @validator("optimizer")
    def sync_optimizer_lr(cls, v, values):
        """Ensure optimizer LR matches main learning rate."""
        if "learning_rate" in values:
            v.learning_rate = values["learning_rate"]
        return v


class UNetConfig(BaseModel):
    """UNet training configuration."""

    model_name: str = Field("document-anonymizer-unet", description="Model name")
    version: str = Field("v1.0", description="Model version")
    base_model: str = Field(
        "stabilityai/stable-diffusion-2-inpainting", description="Base model"
    )

    # Training parameters (FIXED: Increased from reference implementations)
    batch_size: int = Field(8, ge=1, description="Batch size per GPU")
    learning_rate: float = Field(
        1e-4, gt=0.0, description="Learning rate (FIXED: was 1e-5)"
    )
    num_epochs: int = Field(50, ge=1, description="Number of epochs")
    gradient_accumulation_steps: int = Field(
        4, ge=1, description="Gradient accumulation"
    )
    mixed_precision: str = Field("bf16", description="Mixed precision mode")
    gradient_clipping: float = Field(1.0, gt=0.0, description="Gradient clipping")

    # Diffusion parameters
    num_train_timesteps: int = Field(1000, ge=1, description="Training timesteps")
    noise_schedule: str = Field("scaled_linear", description="Noise schedule")

    # Text conditioning
    text_encoder_lr_scale: float = Field(
        0.1, gt=0.0, le=1.0, description="Text encoder LR scale"
    )
    freeze_text_encoder: bool = Field(False, description="Freeze text encoder")

    # Optimizer and scheduler
    optimizer: OptimizerConfig = Field(
        default_factory=lambda: OptimizerConfig(learning_rate=1e-4)
    )
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    # Storage
    checkpoint_dir: Path = Field(
        Path("/tmp/checkpoints"), description="Checkpoint directory"
    )
    save_every_n_steps: int = Field(2500, ge=1, description="Save frequency")
    keep_n_checkpoints: int = Field(
        3, ge=1, description="Number of checkpoints to keep"
    )


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    train_data_path: Path = Field(..., description="Training data path")
    val_data_path: Optional[Path] = Field(None, description="Validation data path")
    crop_size: int = Field(512, ge=128, description="Crop size")
    num_workers: int = Field(4, ge=0, description="Data loader workers")

    # Augmentation (conservative for text preservation)
    rotation_range: float = Field(
        5.0, ge=0.0, le=45.0, description="Rotation range in degrees"
    )
    brightness_range: float = Field(
        0.1, ge=0.0, le=0.5, description="Brightness variation"
    )
    contrast_range: float = Field(0.1, ge=0.0, le=0.5, description="Contrast variation")


class PreprocessingConfig(BaseModel):
    """Image preprocessing configuration."""

    target_crop_size: int = Field(512, ge=128, description="Target crop size")
    max_scale_factor: float = Field(4.0, gt=1.0, description="Maximum scale factor")
    max_memory_bytes: int = Field(
        1024 * 1024 * 1024, gt=0, description="Max memory per image"
    )
    padding_mode: str = Field("reflect", description="Padding mode")
    interpolation: str = Field("lanczos", description="Interpolation method")


class EngineConfig(BaseModel):
    """Inference engine configuration."""

    # Model paths
    vae_model_path: Optional[str] = Field(None, description="VAE model path")
    unet_model_path: Optional[str] = Field(None, description="UNet model path")

    # Inference parameters
    num_inference_steps: int = Field(50, ge=1, le=1000, description="Inference steps")
    guidance_scale: float = Field(7.5, gt=0.0, description="Guidance scale")
    strength: float = Field(1.0, gt=0.0, le=1.0, description="Inpainting strength")

    # Memory management
    enable_memory_efficient_attention: bool = Field(
        True, description="Enable memory efficient attention"
    )
    enable_sequential_cpu_offload: bool = Field(False, description="Enable CPU offload")
    max_batch_size: int = Field(4, ge=1, description="Maximum batch size")

    # Quality settings
    enable_quality_check: bool = Field(True, description="Enable quality verification")
    min_confidence_threshold: float = Field(
        0.7, ge=0.0, le=1.0, description="Minimum confidence"
    )

    # Preprocessing
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)


class R2Config(BaseModel):
    """Cloudflare R2 storage configuration."""

    endpoint_url: str = Field(..., description="R2 endpoint URL")
    access_key_id: str = Field(..., description="Access key ID")
    secret_access_key: str = Field(..., description="Secret access key")
    bucket_name: str = Field(..., description="Bucket name")
    region: str = Field("auto", description="Region")

    @classmethod
    def from_env(cls) -> "R2Config":
        """Load configuration from environment variables."""
        return cls(
            endpoint_url=os.getenv("R2_ENDPOINT_URL", ""),
            access_key_id=os.getenv("R2_ACCESS_KEY_ID", ""),
            secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", ""),
            bucket_name=os.getenv("R2_BUCKET_NAME", ""),
            region=os.getenv("R2_REGION", "auto"),
        )


class MetricsConfig(BaseModel):
    """Metrics and monitoring configuration."""

    enable_metrics: bool = Field(True, description="Enable metrics collection")
    metrics_backend: str = Field("datadog", description="Metrics backend")
    log_level: str = Field("INFO", description="Log level")
    log_format: str = Field("json", description="Log format")


def load_config_from_yaml(
    config_path: Union[str, Path], config_class: type
) -> BaseModel:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            raise ConfigurationError(f"Empty configuration file: {config_path}")

        return config_class(**config_data)

    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {config_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}")


# Add convenient methods to configuration classes
def _add_yaml_methods():
    """Add YAML loading methods to configuration classes."""

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]):
        """Load configuration from YAML file."""
        return load_config_from_yaml(config_path, cls)

    # Add method to all config classes
    for config_class in [VAEConfig, UNetConfig, EngineConfig, R2Config]:
        config_class.from_yaml = from_yaml


_add_yaml_methods()
