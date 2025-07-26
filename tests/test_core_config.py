"""
Unit tests for core configuration - Imperative style.

Tests configuration loading, validation, and defaults.
"""

import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.anonymizer.core.config import (
    AppConfig,
    DatasetConfig,
    EngineConfig,
    LossConfig,
    MetricsConfig,
    OptimizerConfig,
    PreprocessingConfig,
    R2Config,
    SchedulerConfig,
    UNetConfig,
    VAEConfig,
    load_config_from_yaml,
)
from src.anonymizer.core.exceptions import ConfigurationError


class TestOptimizerConfig:
    """Test OptimizerConfig validation."""

    def test_valid_optimizer_config(self):
        """Test creating valid optimizer configuration."""
        config = OptimizerConfig(
            type="AdamW", learning_rate=1e-4, weight_decay=0.01, betas=[0.9, 0.999]
        )

        assert config.type == "AdamW"
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.01
        assert config.betas == [0.9, 0.999]

    def test_optimizer_config_defaults(self):
        """Test optimizer configuration defaults."""
        config = OptimizerConfig(learning_rate=1e-4)

        assert config.type == "AdamW"
        assert config.weight_decay == 0.01
        assert config.betas == [0.9, 0.999]

    def test_optimizer_config_validation_negative_lr(self):
        """Test that learning rate must be positive."""
        with pytest.raises(ValidationError):
            OptimizerConfig(learning_rate=-1e-4)

    def test_optimizer_config_validation_negative_weight_decay(self):
        """Test that weight decay cannot be negative."""
        with pytest.raises(ValidationError):
            OptimizerConfig(learning_rate=1e-4, weight_decay=-0.01)

    def test_optimizer_config_validation_invalid_betas(self):
        """Test that betas must be valid."""
        # Test wrong number of betas
        with pytest.raises(ValidationError):
            OptimizerConfig(learning_rate=1e-4, betas=[0.9])

        # Test beta >= 1.0
        with pytest.raises(ValidationError):
            OptimizerConfig(learning_rate=1e-4, betas=[0.9, 1.0])

        # Test negative beta
        with pytest.raises(ValidationError):
            OptimizerConfig(learning_rate=1e-4, betas=[-0.1, 0.9])


class TestSchedulerConfig:
    """Test SchedulerConfig validation."""

    def test_valid_scheduler_config(self):
        """Test creating valid scheduler configuration."""
        config = SchedulerConfig(
            type="cosine_with_restarts",
            warmup_steps=1000,
            num_cycles=3,
            min_lr_ratio=0.1,
        )

        assert config.type == "cosine_with_restarts"
        assert config.warmup_steps == 1000
        assert config.num_cycles == 3
        assert config.min_lr_ratio == 0.1

    def test_scheduler_config_defaults(self):
        """Test scheduler configuration defaults."""
        config = SchedulerConfig()

        assert config.type == "cosine_with_restarts"
        assert config.warmup_steps == 1000
        assert config.num_cycles == 3
        assert config.min_lr_ratio == 0.1

    def test_scheduler_config_validation_negative_warmup(self):
        """Test that warmup steps cannot be negative."""
        with pytest.raises(ValidationError):
            SchedulerConfig(warmup_steps=-100)

    def test_scheduler_config_validation_invalid_cycles(self):
        """Test that num_cycles must be positive."""
        with pytest.raises(ValidationError):
            SchedulerConfig(num_cycles=0)

    def test_scheduler_config_validation_invalid_min_lr_ratio(self):
        """Test that min_lr_ratio must be in valid range."""
        # Test ratio > 1.0
        with pytest.raises(ValidationError):
            SchedulerConfig(min_lr_ratio=1.5)

        # Test negative ratio
        with pytest.raises(ValidationError):
            SchedulerConfig(min_lr_ratio=-0.1)


class TestLossConfig:
    """Test LossConfig validation."""

    def test_valid_loss_config(self):
        """Test creating valid loss configuration."""
        config = LossConfig(kl_weight=0.00025, perceptual_weight=0.1, recon_loss_type="mse")

        assert config.kl_weight == 0.00025
        assert config.perceptual_weight == 0.1
        assert config.recon_loss_type == "mse"

    def test_loss_config_defaults(self):
        """Test loss configuration defaults."""
        config = LossConfig()

        assert config.kl_weight == 0.00025
        assert config.perceptual_weight == 0.1
        assert config.recon_loss_type == "mse"

    def test_loss_config_validation_negative_weights(self):
        """Test that loss weights cannot be negative."""
        with pytest.raises(ValidationError):
            LossConfig(kl_weight=-0.1)

        with pytest.raises(ValidationError):
            LossConfig(perceptual_weight=-0.1)


class TestVAEConfig:
    """Test VAEConfig validation and fixes."""

    def test_valid_vae_config(self):
        """Test creating valid VAE configuration."""
        config = VAEConfig(
            model_name="test-vae",
            version="v1.0",
            base_model="stabilityai/stable-diffusion-2-1-base",
            batch_size=16,
            learning_rate=5e-4,
            num_epochs=100,
        )

        assert config.model_name == "test-vae"
        assert config.version == "v1.0"
        assert config.batch_size == 16
        assert config.learning_rate == 5e-4  # FIXED from 5e-6
        assert config.num_epochs == 100

    def test_vae_config_critical_fixes(self):
        """Test that critical bugs are fixed in VAE config."""
        config = VAEConfig()

        # CRITICAL FIX: Learning rate increased from 5e-6 to 5e-4
        assert config.learning_rate == 5e-4

        # CRITICAL FIX: Batch size increased from 2 to 16
        assert config.batch_size == 16

        # CRITICAL FIX: KL divergence loss included
        assert config.loss.kl_weight > 0
        assert config.loss.kl_weight == 0.00025

        # CRITICAL FIX: Perceptual loss included
        assert config.loss.perceptual_weight > 0
        assert config.loss.perceptual_weight == 0.1

    def test_vae_config_optimizer_lr_sync(self):
        """Test that optimizer learning rate syncs with main LR."""
        config = VAEConfig(learning_rate=1e-3)

        # The optimizer LR should sync with the main config LR via the validator
        assert config.learning_rate == 1e-3  # Main config has correct LR
        assert config.optimizer.learning_rate == 1e-3  # Optimizer syncs with main LR

    def test_vae_config_validation_batch_size(self):
        """Test that batch size must be positive."""
        with pytest.raises(ValidationError):
            VAEConfig(batch_size=0)

    def test_vae_config_validation_learning_rate(self):
        """Test that learning rate must be positive."""
        with pytest.raises(ValidationError):
            VAEConfig(learning_rate=0.0)

    def test_vae_config_validation_epochs(self):
        """Test that num_epochs must be positive."""
        with pytest.raises(ValidationError):
            VAEConfig(num_epochs=0)


class TestUNetConfig:
    """Test UNetConfig validation and fixes."""

    def test_valid_unet_config(self):
        """Test creating valid UNet configuration."""
        config = UNetConfig(
            model_name="test-unet",
            version="v1.0",
            base_model="stabilityai/stable-diffusion-2-inpainting",
            batch_size=8,
            learning_rate=1e-4,
            num_epochs=50,
        )

        assert config.model_name == "test-unet"
        assert config.version == "v1.0"
        assert config.base_model == "stabilityai/stable-diffusion-2-inpainting"
        assert config.batch_size == 8
        assert config.learning_rate == 1e-4  # FIXED from 1e-5
        assert config.num_epochs == 50

    def test_unet_config_critical_fixes(self):
        """Test that critical bugs are fixed in UNet config."""
        config = UNetConfig()

        # CRITICAL FIX: Learning rate increased from 1e-5 to 1e-4
        assert config.learning_rate == 1e-4

        # CRITICAL FIX: Batch size increased from 4 to 8
        assert config.batch_size == 8

        # CORRECT: SD 2.0 inpainting base model (9-channel UNet)
        assert "inpainting" in config.base_model

    def test_unet_config_diffusion_parameters(self):
        """Test UNet diffusion parameters."""
        config = UNetConfig()

        assert config.num_train_timesteps == 1000
        assert config.noise_schedule == "scaled_linear"

    def test_unet_config_text_conditioning(self):
        """Test UNet text conditioning parameters."""
        config = UNetConfig()

        assert 0 < config.text_encoder_lr_scale <= 1.0
        assert isinstance(config.freeze_text_encoder, bool)

    def test_unet_config_validation_timesteps(self):
        """Test that timesteps must be positive."""
        with pytest.raises(ValidationError):
            UNetConfig(num_train_timesteps=0)

    def test_unet_config_validation_text_encoder_lr_scale(self):
        """Test that text encoder LR scale is in valid range."""
        with pytest.raises(ValidationError):
            UNetConfig(text_encoder_lr_scale=0.0)

        with pytest.raises(ValidationError):
            UNetConfig(text_encoder_lr_scale=1.5)


class TestDatasetConfig:
    """Test DatasetConfig validation."""

    def test_valid_dataset_config(self, temp_dir):
        """Test creating valid dataset configuration."""
        config = DatasetConfig(
            train_data_path=temp_dir,
            val_data_path=None,
            crop_size=512,
            num_workers=4,
            rotation_range=5.0,
            brightness_range=0.1,
            contrast_range=0.1,
        )

        assert config.train_data_path == temp_dir
        assert config.val_data_path is None
        assert config.crop_size == 512
        assert config.num_workers == 4
        assert config.rotation_range == 5.0
        assert config.brightness_range == 0.1
        assert config.contrast_range == 0.1

    def test_dataset_config_validation_crop_size(self, temp_dir):
        """Test that crop size has minimum value."""
        with pytest.raises(ValidationError):
            DatasetConfig(train_data_path=temp_dir, crop_size=64)  # Below 128 minimum

    def test_dataset_config_validation_negative_workers(self, temp_dir):
        """Test that num_workers cannot be negative."""
        with pytest.raises(ValidationError):
            DatasetConfig(train_data_path=temp_dir, num_workers=-1)

    def test_dataset_config_validation_augmentation_ranges(self, temp_dir):
        """Test that augmentation ranges are valid."""
        # Test negative rotation
        with pytest.raises(ValidationError):
            DatasetConfig(train_data_path=temp_dir, rotation_range=-5.0)

        # Test excessive rotation
        with pytest.raises(ValidationError):
            DatasetConfig(train_data_path=temp_dir, rotation_range=50.0)

        # Test negative brightness range
        with pytest.raises(ValidationError):
            DatasetConfig(train_data_path=temp_dir, brightness_range=-0.1)

        # Test excessive brightness range
        with pytest.raises(ValidationError):
            DatasetConfig(train_data_path=temp_dir, brightness_range=0.6)


class TestPreprocessingConfig:
    """Test PreprocessingConfig validation."""

    def test_valid_preprocessing_config(self):
        """Test creating valid preprocessing configuration."""
        config = PreprocessingConfig(
            target_crop_size=512,
            max_scale_factor=4.0,
            max_memory_bytes=1024 * 1024 * 1024,
            padding_mode="reflect",
            interpolation="lanczos",
        )

        assert config.target_crop_size == 512
        assert config.max_scale_factor == 4.0
        assert config.max_memory_bytes == 1024 * 1024 * 1024
        assert config.padding_mode == "reflect"
        assert config.interpolation == "lanczos"

    def test_preprocessing_config_defaults(self):
        """Test preprocessing configuration defaults."""
        config = PreprocessingConfig()

        assert config.target_crop_size == 512
        assert config.max_scale_factor == 4.0
        assert config.max_memory_bytes == 1024 * 1024 * 1024
        assert config.padding_mode == "reflect"
        assert config.interpolation == "lanczos"

    def test_preprocessing_config_validation_crop_size(self):
        """Test that crop size has minimum value."""
        with pytest.raises(ValidationError):
            PreprocessingConfig(target_crop_size=64)  # Below 128 minimum

    def test_preprocessing_config_validation_scale_factor(self):
        """Test that scale factor is valid."""
        with pytest.raises(ValidationError):
            PreprocessingConfig(max_scale_factor=1.0)  # Must be > 1.0

    def test_preprocessing_config_validation_memory_limit(self):
        """Test that memory limit is positive."""
        with pytest.raises(ValidationError):
            PreprocessingConfig(max_memory_bytes=0)


class TestEngineConfig:
    """Test EngineConfig validation."""

    def test_valid_engine_config(self):
        """Test creating valid engine configuration."""
        config = EngineConfig(
            vae_model_path="path/to/vae",
            unet_model_path="path/to/unet",
            num_inference_steps=50,
            guidance_scale=7.5,
            strength=1.0,
            enable_memory_efficient_attention=True,
            max_batch_size=4,
            min_confidence_threshold=0.7,
        )

        assert config.vae_model_path == "path/to/vae"
        assert config.unet_model_path == "path/to/unet"
        assert config.num_inference_steps == 50
        assert config.guidance_scale == 7.5
        assert config.strength == 1.0
        assert config.enable_memory_efficient_attention is True
        assert config.max_batch_size == 4
        assert config.min_confidence_threshold == 0.7

    def test_engine_config_defaults(self, monkeypatch):
        """Test engine configuration defaults."""
        # Clear any ENGINE_ environment variables to test true defaults
        engine_env_vars = [key for key in os.environ.keys() if key.startswith("ENGINE_")]
        for var in engine_env_vars:
            monkeypatch.delenv(var, raising=False)

        # Also ensure no .env file is loaded by creating config with explicit env_file=None
        config = EngineConfig(_env_file=None)

        assert config.vae_model_path is None
        assert config.unet_model_path is None
        assert config.num_inference_steps == 50
        assert config.guidance_scale == 7.5
        assert config.strength == 1.0
        assert config.enable_memory_efficient_attention is True
        assert config.enable_sequential_cpu_offload is False
        assert config.max_batch_size == 4
        assert config.enable_quality_check is True
        assert config.min_confidence_threshold == 0.7

    def test_engine_config_validation_inference_steps(self):
        """Test that inference steps are in valid range."""
        with pytest.raises(ValidationError):
            EngineConfig(num_inference_steps=0)

        with pytest.raises(ValidationError):
            EngineConfig(num_inference_steps=1001)

    def test_engine_config_validation_guidance_scale(self):
        """Test that guidance scale is positive."""
        with pytest.raises(ValidationError):
            EngineConfig(guidance_scale=0.0)

    def test_engine_config_validation_strength(self):
        """Test that strength is in valid range."""
        with pytest.raises(ValidationError):
            EngineConfig(strength=0.0)

        with pytest.raises(ValidationError):
            EngineConfig(strength=1.5)

    def test_engine_config_validation_confidence_threshold(self):
        """Test that confidence threshold is in valid range."""
        with pytest.raises(ValidationError):
            EngineConfig(min_confidence_threshold=-0.1)

        with pytest.raises(ValidationError):
            EngineConfig(min_confidence_threshold=1.1)


class TestR2Config:
    """Test R2Config validation."""

    def test_valid_r2_config(self):
        """Test creating valid R2 configuration."""
        config = R2Config(
            endpoint_url="https://endpoint.r2.cloudflarestorage.com",
            access_key_id="test_access_key",
            secret_access_key="test_secret_key",
            bucket_name="test-bucket",
            region="auto",
        )

        assert config.endpoint_url == "https://endpoint.r2.cloudflarestorage.com"
        assert config.access_key_id == "test_access_key"
        assert config.secret_access_key == "test_secret_key"
        assert config.bucket_name == "test-bucket"
        assert config.region == "auto"

    def test_r2_config_from_env(self, monkeypatch):
        """Test loading R2 config from environment variables."""
        # Set environment variables
        monkeypatch.setenv("R2_ENDPOINT_URL", "https://test.endpoint.com")
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "test_key")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test_secret")
        monkeypatch.setenv("R2_BUCKET_NAME", "test-bucket")
        monkeypatch.setenv("R2_REGION", "us-east-1")

        config = R2Config.from_env()

        assert config.endpoint_url == "https://test.endpoint.com"
        assert config.access_key_id == "test_key"
        assert config.secret_access_key == "test_secret"
        assert config.bucket_name == "test-bucket"
        assert config.region == "us-east-1"

    def test_r2_config_auto_env_loading(self, monkeypatch):
        """Test that R2Config automatically loads from environment variables."""
        # Set environment variables with R2 prefix
        monkeypatch.setenv("R2_ENDPOINT_URL", "https://auto.endpoint.com")
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "auto_key")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "auto_secret")
        monkeypatch.setenv("R2_BUCKET_NAME", "auto-bucket")

        # Just instantiate normally - should auto-load from env
        config = R2Config()

        assert config.endpoint_url == "https://auto.endpoint.com"
        assert config.access_key_id == "auto_key"
        assert config.secret_access_key == "auto_secret"
        assert config.bucket_name == "auto-bucket"


class TestMetricsConfig:
    """Test MetricsConfig validation."""

    def test_valid_metrics_config(self):
        """Test creating valid metrics configuration."""
        config = MetricsConfig(
            enable_metrics=True,
            metrics_backend="datadog",
            log_level="INFO",
            log_format="json",
        )

        assert config.enable_metrics is True
        assert config.metrics_backend == "datadog"
        assert config.log_level == "INFO"
        assert config.log_format == "json"

    def test_metrics_config_defaults(self):
        """Test metrics configuration defaults."""
        config = MetricsConfig()

        assert config.enable_metrics is True
        assert config.metrics_backend == "datadog"
        assert config.log_level == "INFO"
        assert config.log_format == "json"


class TestConfigLoading:
    """Test configuration loading from YAML files."""

    def test_load_vae_config_from_yaml(self, temp_dir):
        """Test loading VAE config from YAML."""
        config_data = {
            "model_name": "test-vae",
            "version": "v1.0",
            "base_model": "stabilityai/stable-diffusion-2-1-base",
            "batch_size": 16,
            "learning_rate": 5e-4,
            "num_epochs": 100,
            "loss": {"kl_weight": 0.00025, "perceptual_weight": 0.1},
        }

        config_path = temp_dir / "vae_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = VAEConfig.from_yaml(config_path)

        assert config.model_name == "test-vae"
        assert config.learning_rate == 5e-4
        assert config.loss.kl_weight == 0.00025

    def test_load_config_from_yaml_file_not_found(self):
        """Test loading config from non-existent file."""
        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_config_from_yaml("nonexistent.yaml", VAEConfig)

    def test_load_config_from_yaml_invalid_yaml(self, temp_dir):
        """Test loading config from invalid YAML."""
        config_path = temp_dir / "invalid.yaml"
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [unclosed")

        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_config_from_yaml(config_path, VAEConfig)

    def test_load_config_from_yaml_empty_file(self, temp_dir):
        """Test loading config from empty YAML file."""
        config_path = temp_dir / "empty.yaml"
        config_path.touch()  # Create empty file

        with pytest.raises(ConfigurationError, match="Empty configuration file"):
            load_config_from_yaml(config_path, VAEConfig)

    def test_load_config_from_yaml_validation_error(self, temp_dir):
        """Test loading config with validation errors."""
        config_data = {"batch_size": -1, "learning_rate": 1e-4}  # Invalid batch size

        config_path = temp_dir / "invalid_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ConfigurationError, match="Failed to load configuration"):
            load_config_from_yaml(config_path, VAEConfig)


class TestAppConfig:
    """Test main AppConfig class."""

    def test_valid_app_config(self):
        """Test creating valid app configuration."""
        config = AppConfig(
            environment="production",
            debug=False,
            device="cuda",
            models_dir="./models",
            data_dir="./data",
        )

        assert config.environment == "production"
        assert config.debug is False
        assert config.device == "cuda"
        assert config.models_dir == Path("./models")
        assert config.data_dir == Path("./data")
        assert isinstance(config.vae, VAEConfig)
        assert isinstance(config.unet, UNetConfig)
        assert isinstance(config.engine, EngineConfig)

    def test_app_config_defaults(self):
        """Test app configuration defaults."""
        config = AppConfig()

        assert config.environment == "development"
        assert config.debug is True
        assert config.log_level == "INFO"
        assert config.device == "auto"
        assert config.enable_mixed_precision is True
        assert config.models_dir == Path("./models")
        assert config.data_dir == Path("./data")
        assert config.output_dir == Path("./output")

    def test_app_config_load_from_env(self, monkeypatch):
        """Test loading AppConfig from environment variables."""
        monkeypatch.setenv("APP_ENVIRONMENT", "staging")
        monkeypatch.setenv("APP_DEBUG", "false")
        monkeypatch.setenv("APP_DEVICE", "mps")
        monkeypatch.setenv("VAE_BATCH_SIZE", "32")
        monkeypatch.setenv("UNET_BATCH_SIZE", "16")

        config = AppConfig.load_from_env(env_file=None)

        assert config.environment == "staging"
        assert config.debug is False
        assert config.device == "mps"
        assert config.vae.batch_size == 32
        assert config.unet.batch_size == 16

    def test_app_config_load_with_overrides(self, temp_dir):
        """Test AppConfig with YAML overrides."""
        # Create .env file
        env_file = temp_dir / "test.env"
        env_content = """
APP_ENVIRONMENT=production
APP_DEBUG=false
VAE_BATCH_SIZE=64
"""
        with open(env_file, "w") as f:
            f.write(env_content)

        # Create VAE YAML
        vae_yaml = temp_dir / "vae.yaml"
        vae_yaml_content = """
model_name: override-vae
batch_size: 128
learning_rate: 0.002
"""
        with open(vae_yaml, "w") as f:
            f.write(vae_yaml_content)

        config = AppConfig.load_with_overrides(
            env_file=env_file, vae_yaml=vae_yaml, device="custom"
        )

        assert config.environment == "production"  # From .env
        assert config.debug is False  # From .env
        assert config.vae.model_name == "override-vae"  # From YAML
        assert config.vae.batch_size == 128  # From YAML (overrides .env)
        assert config.device == "custom"  # From kwargs
