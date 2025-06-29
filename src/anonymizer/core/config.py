"""Configuration management for the document anonymization system."""

from typing import Optional, List, Union
from pathlib import Path
import os
import yaml
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from .exceptions import ConfigurationError


class OptimizerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPTIMIZER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
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


class SchedulerConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SCHEDULER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    """Learning rate scheduler configuration."""

    type: str = Field("cosine_with_restarts", description="Scheduler type")
    warmup_steps: int = Field(1000, ge=0, description="Warmup steps")
    num_cycles: int = Field(3, ge=1, description="Number of cycles")
    min_lr_ratio: float = Field(0.1, ge=0.0, le=1.0, description="Minimum LR ratio")


class LossConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOSS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    """Loss function configuration."""

    kl_weight: float = Field(0.00025, ge=0.0, description="KL divergence weight")
    perceptual_weight: float = Field(0.1, ge=0.0, description="Perceptual loss weight")
    recon_loss_type: str = Field("mse", description="Reconstruction loss type")


class VAEConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VAE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
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


class UNetConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="UNET_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
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


class DatasetConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATASET_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
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


class PreprocessingConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PREPROCESSING_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    """Image preprocessing configuration."""

    target_crop_size: int = Field(512, ge=128, description="Target crop size")
    max_scale_factor: float = Field(4.0, gt=1.0, description="Maximum scale factor")
    max_memory_bytes: int = Field(
        1024 * 1024 * 1024, gt=0, description="Max memory per image"
    )
    padding_mode: str = Field("reflect", description="Padding mode")
    interpolation: str = Field("lanczos", description="Interpolation method")


class EngineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ENGINE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
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


class R2Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="R2_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    """Cloudflare R2 storage configuration."""

    endpoint_url: str = Field(..., description="R2 endpoint URL")
    access_key_id: str = Field(..., description="Access key ID")
    secret_access_key: str = Field(..., description="Secret access key")
    bucket_name: str = Field(..., description="Bucket name")
    region: str = Field("auto", description="Region")

    @classmethod
    def from_env(cls) -> "R2Config":
        """Load configuration from environment variables (backwards compatibility)."""
        # pydantic-settings now handles this automatically
        return cls()


class MetricsConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="METRICS_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    """Metrics and monitoring configuration."""

    enable_metrics: bool = Field(True, description="Enable metrics collection")
    metrics_backend: str = Field("datadog", description="Metrics backend")
    log_level: str = Field("INFO", description="Log level")
    log_format: str = Field("json", description="Log format")


class AppConfig(BaseSettings):
    """Main application configuration that loads from multiple sources."""

    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Environment settings
    environment: str = Field(
        "development", description="Environment (development, staging, production)"
    )
    debug: bool = Field(True, description="Debug mode")
    log_level: str = Field("INFO", description="Application log level")

    # Model and data paths
    models_dir: Path = Field(Path("./models"), description="Models directory")
    data_dir: Path = Field(Path("./data"), description="Data directory")
    output_dir: Path = Field(Path("./output"), description="Output directory")

    # GPU settings
    device: str = Field("auto", description="Device to use (auto, cpu, cuda, mps)")
    enable_mixed_precision: bool = Field(
        True, description="Enable mixed precision training"
    )

    # Configuration file paths (optional)
    vae_config_path: Optional[Path] = Field(None, description="VAE config YAML path")
    unet_config_path: Optional[Path] = Field(None, description="UNet config YAML path")
    engine_config_path: Optional[Path] = Field(
        None, description="Engine config YAML path"
    )

    # Sub-configurations (will be properly loaded with env vars in post_init)
    vae: VAEConfig = Field(default_factory=VAEConfig)
    unet: UNetConfig = Field(default_factory=UNetConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    r2: Optional[R2Config] = Field(default_factory=lambda: None)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)

    def model_post_init(self, __context) -> None:
        """Post-initialization to reload nested configs with proper env var support."""
        # Reload nested configs to pick up environment variables
        # This ensures that VAE_, UNET_, etc. prefixed env vars are loaded
        env_file = getattr(self, "_env_file", ".env")

        # Only reload if we have an env file or if env vars might be set
        if any(
            key.startswith(("VAE_", "UNET_", "ENGINE_", "R2_", "METRICS_"))
            for key in os.environ
        ):
            try:
                self.vae = VAEConfig(
                    _env_file=env_file if env_file and Path(env_file).exists() else None
                )
                self.unet = UNetConfig(
                    _env_file=env_file if env_file and Path(env_file).exists() else None
                )
                self.engine = EngineConfig(
                    _env_file=env_file if env_file and Path(env_file).exists() else None
                )
                self.metrics = MetricsConfig(
                    _env_file=env_file if env_file and Path(env_file).exists() else None
                )
                # R2 is optional since it has required fields
                try:
                    self.r2 = R2Config(
                        _env_file=(
                            env_file if env_file and Path(env_file).exists() else None
                        )
                    )
                except Exception:
                    self.r2 = None
            except Exception:
                # If loading fails, keep the defaults
                pass

    @classmethod
    def load_from_env(cls, env_file: Union[str, Path, None] = ".env") -> "AppConfig":
        """Load configuration from environment variables and .env file."""
        if env_file:
            env_file = Path(env_file)
            if env_file.exists():
                return cls(_env_file=env_file)
        return cls()

    @classmethod
    def load_with_overrides(
        cls,
        env_file: Union[str, Path, None] = ".env",
        vae_yaml: Optional[Union[str, Path]] = None,
        unet_yaml: Optional[Union[str, Path]] = None,
        engine_yaml: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> "AppConfig":
        """Load configuration with YAML overrides for specific components."""
        # Start with env/defaults
        config = cls.load_from_env(env_file)

        # Override with YAML configs if provided
        if vae_yaml:
            config.vae = VAEConfig.from_yaml(vae_yaml)
        elif config.vae_config_path and config.vae_config_path.exists():
            config.vae = VAEConfig.from_yaml(config.vae_config_path)

        if unet_yaml:
            config.unet = UNetConfig.from_yaml(unet_yaml)
        elif config.unet_config_path and config.unet_config_path.exists():
            config.unet = UNetConfig.from_yaml(config.unet_config_path)

        if engine_yaml:
            config.engine = EngineConfig.from_yaml(engine_yaml)
        elif config.engine_config_path and config.engine_config_path.exists():
            config.engine = EngineConfig.from_yaml(config.engine_config_path)

        # Override with any additional kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config


def load_config_from_yaml(
    config_path: Union[str, Path], config_class: type
) -> BaseSettings:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        if config_data is None:
            raise ConfigurationError(f"Empty configuration file: {config_path}")

        # For BaseSettings classes, we can pass the data directly
        # but we need to disable env file loading for this specific instance
        if issubclass(config_class, BaseSettings):
            # Create a temporary config class that doesn't load from .env
            class TempConfig(config_class):
                model_config = SettingsConfigDict(
                    env_file=None,  # Don't load .env for YAML-based configs
                    case_sensitive=False,
                    extra="ignore",
                )

            return TempConfig(**config_data)
        else:
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

    @classmethod
    def from_env_and_yaml(
        cls, yaml_path: Optional[Union[str, Path]] = None, env_file: str = ".env"
    ):
        """Load configuration from environment variables and optionally override with YAML."""
        if yaml_path and Path(yaml_path).exists():
            return cls.from_yaml(yaml_path)
        else:
            # Load from environment variables/.env file
            return cls(_env_file=env_file if Path(env_file).exists() else None)

    # Add methods to all config classes
    for config_class in [
        VAEConfig,
        UNetConfig,
        EngineConfig,
        R2Config,
        DatasetConfig,
        MetricsConfig,
        AppConfig,
    ]:
        config_class.from_yaml = from_yaml
        config_class.from_env_and_yaml = from_env_and_yaml


_add_yaml_methods()
