"""
Unit tests for configuration environment variable support - Imperative style.

Tests pydantic-settings integration for .env files and environment variables.
"""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.anonymizer.core.config import (
    AppConfig,
    EngineConfig,
    MetricsConfig,
    OptimizerConfig,
    R2Config,
    UNetConfig,
    VAEConfig,
)


class TestEnvironmentVariableSupport:
    """Test environment variable support for all config classes."""

    def test_vae_config_from_env_vars(self, monkeypatch):
        """Test VAE config loading from environment variables."""
        # Set environment variables with VAE prefix
        monkeypatch.setenv("VAE_MODEL_NAME", "test-vae-env")
        monkeypatch.setenv("VAE_BATCH_SIZE", "32")
        monkeypatch.setenv("VAE_LEARNING_RATE", "0.001")
        monkeypatch.setenv("VAE_NUM_EPOCHS", "50")

        config = VAEConfig()

        assert config.model_name == "test-vae-env"
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.num_epochs == 50

    def test_unet_config_from_env_vars(self, monkeypatch):
        """Test UNet config loading from environment variables."""
        monkeypatch.setenv("UNET_MODEL_NAME", "test-unet-env")
        monkeypatch.setenv("UNET_BATCH_SIZE", "16")
        monkeypatch.setenv("UNET_LEARNING_RATE", "0.0005")
        monkeypatch.setenv("UNET_NUM_TRAIN_TIMESTEPS", "500")

        config = UNetConfig()

        assert config.model_name == "test-unet-env"
        assert config.batch_size == 16
        assert config.learning_rate == 0.0005
        assert config.num_train_timesteps == 500

    def test_engine_config_from_env_vars(self, monkeypatch):
        """Test Engine config loading from environment variables."""
        monkeypatch.setenv("ENGINE_VAE_MODEL_PATH", "/path/to/vae")
        monkeypatch.setenv("ENGINE_UNET_MODEL_PATH", "/path/to/unet")
        monkeypatch.setenv("ENGINE_NUM_INFERENCE_STEPS", "25")
        monkeypatch.setenv("ENGINE_GUIDANCE_SCALE", "10.0")

        config = EngineConfig()

        assert config.vae_model_path == "/path/to/vae"
        assert config.unet_model_path == "/path/to/unet"
        assert config.num_inference_steps == 25
        assert config.guidance_scale == 10.0

    def test_r2_config_from_env_vars(self, monkeypatch):
        """Test R2 config loading from environment variables."""
        monkeypatch.setenv("R2_ENDPOINT_URL", "https://test.endpoint.com")
        monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-key-env")
        monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-secret-env")
        monkeypatch.setenv("R2_BUCKET_NAME", "test-bucket-env")
        monkeypatch.setenv("R2_REGION", "us-west-1")

        config = R2Config()

        assert config.endpoint_url == "https://test.endpoint.com"
        assert config.access_key_id == "test-key-env"
        assert config.secret_access_key == "test-secret-env"
        assert config.bucket_name == "test-bucket-env"
        assert config.region == "us-west-1"

    def test_optimizer_config_from_env_vars(self, monkeypatch):
        """Test Optimizer config loading from environment variables."""
        monkeypatch.setenv("OPTIMIZER_TYPE", "Adam")
        monkeypatch.setenv("OPTIMIZER_LEARNING_RATE", "0.002")
        monkeypatch.setenv("OPTIMIZER_WEIGHT_DECAY", "0.005")

        # For BaseSettings, env vars take precedence, but we need to provide required fields
        config = OptimizerConfig()

        # Note: env vars should provide values
        assert config.type == "Adam"
        assert config.learning_rate == 0.002
        assert config.weight_decay == 0.005

    def test_metrics_config_from_env_vars(self, monkeypatch):
        """Test Metrics config loading from environment variables."""
        monkeypatch.setenv("METRICS_ENABLE_METRICS", "false")
        monkeypatch.setenv("METRICS_METRICS_BACKEND", "prometheus")
        monkeypatch.setenv("METRICS_LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("METRICS_LOG_FORMAT", "text")

        config = MetricsConfig()

        assert config.enable_metrics is False
        assert config.metrics_backend == "prometheus"
        assert config.log_level == "DEBUG"
        assert config.log_format == "text"


class TestDotEnvFileSupport:
    """Test .env file support."""

    def test_config_from_env_file(self, temp_dir):
        """Test loading config from .env file."""
        env_file = temp_dir / ".env"
        env_content = """
VAE_MODEL_NAME=vae-from-file
VAE_BATCH_SIZE=64
VAE_LEARNING_RATE=0.0002
UNET_MODEL_NAME=unet-from-file
UNET_BATCH_SIZE=32
ENGINE_NUM_INFERENCE_STEPS=75
R2_ENDPOINT_URL=https://test.endpoint.com
R2_ACCESS_KEY_ID=test-key
R2_SECRET_ACCESS_KEY=test-secret
R2_BUCKET_NAME=bucket-from-file
"""
        with Path(env_file).open("w") as f:
            f.write(env_content)

        # Change to temp directory so .env file is found
        original_cwd = Path.cwd()
        try:
            os.chdir(temp_dir)

            vae_config = VAEConfig()
            unet_config = UNetConfig()
            engine_config = EngineConfig()
            r2_config = R2Config()

            assert vae_config.model_name == "vae-from-file"
            assert vae_config.batch_size == 64
            assert vae_config.learning_rate == 0.0002
            assert unet_config.model_name == "unet-from-file"
            assert unet_config.batch_size == 32
            assert engine_config.num_inference_steps == 75
            assert r2_config.bucket_name == "bucket-from-file"

        finally:
            os.chdir(original_cwd)

    def test_app_config_from_env_file(self, temp_dir):
        """Test AppConfig loading from .env file."""
        env_file = temp_dir / ".env"
        env_content = """
APP_ENVIRONMENT=production
APP_DEBUG=false
APP_DEVICE=cuda
VAE_BATCH_SIZE=128
UNET_BATCH_SIZE=64
ENGINE_NUM_INFERENCE_STEPS=100
"""
        with Path(env_file).open("w") as f:
            f.write(env_content)

        config = AppConfig(_env_file=env_file)

        assert config.environment == "production"
        assert config.debug is False
        assert config.device == "cuda"
        # Note: The nested VAE/UNet configs need to be created separately to load their env vars
        # since AppConfig defaults create them without env loading
        # This test shows the basic AppConfig functionality
        # For nested config env loading, see separate tests


class TestAppConfig:
    """Test the main AppConfig class."""

    def test_app_config_defaults(self):
        """Test AppConfig default values."""
        config = AppConfig()

        assert config.environment == "development"
        assert config.debug is True
        assert config.log_level == "INFO"
        assert config.device == "auto"
        assert config.enable_mixed_precision is True
        assert isinstance(config.vae, VAEConfig)
        assert isinstance(config.unet, UNetConfig)
        assert isinstance(config.engine, EngineConfig)
        assert isinstance(config.metrics, MetricsConfig)

    def test_app_config_load_from_env(self, monkeypatch):
        """Test AppConfig.load_from_env method."""
        monkeypatch.setenv("APP_ENVIRONMENT", "staging")
        monkeypatch.setenv("APP_DEBUG", "false")
        monkeypatch.setenv("APP_DEVICE", "cpu")
        monkeypatch.setenv("VAE_BATCH_SIZE", "8")

        config = AppConfig.load_from_env(env_file=None)

        assert config.environment == "staging"
        assert config.debug is False
        assert config.device == "cpu"
        assert config.vae.batch_size == 8

    def test_app_config_load_with_overrides(self, temp_dir):
        """Test AppConfig.load_with_overrides method."""
        # Create .env file
        env_file = temp_dir / ".env"
        env_content = """
APP_ENVIRONMENT=production
VAE_BATCH_SIZE=64
"""
        with Path(env_file).open("w") as f:
            f.write(env_content)

        # Create YAML config files
        vae_yaml = temp_dir / "vae.yaml"
        vae_yaml_content = """
model_name: yaml-vae
batch_size: 256
learning_rate: 0.001
"""
        with Path(vae_yaml).open("w") as f:
            f.write(vae_yaml_content)

        config = AppConfig.load_with_overrides(env_file=env_file, vae_yaml=vae_yaml, device="mps")

        assert config.environment == "production"  # From .env
        assert config.vae.model_name == "yaml-vae"  # From YAML override
        assert config.vae.batch_size == 256  # From YAML override
        assert config.device == "mps"  # From kwargs override

    def test_app_config_yaml_path_config(self, temp_dir, monkeypatch):
        """Test AppConfig with YAML paths in config."""
        # Create YAML config
        vae_yaml = temp_dir / "vae_config.yaml"
        vae_yaml_content = """
model_name: path-configured-vae
batch_size: 512
"""
        with Path(vae_yaml).open("w") as f:
            f.write(vae_yaml_content)

        # Set env var for config path
        monkeypatch.setenv("APP_VAE_CONFIG_PATH", str(vae_yaml))

        config = AppConfig.load_with_overrides()

        assert config.vae.model_name == "path-configured-vae"
        assert config.vae.batch_size == 512


class TestFromEnvAndYamlMethods:
    """Test the from_env_and_yaml methods."""

    def test_from_env_and_yaml_env_only(self, monkeypatch):
        """Test from_env_and_yaml with environment variables only."""
        monkeypatch.setenv("VAE_MODEL_NAME", "env-only-vae")
        monkeypatch.setenv("VAE_BATCH_SIZE", "128")

        config = VAEConfig.from_env_and_yaml()

        assert config.model_name == "env-only-vae"
        assert config.batch_size == 128

    def test_from_env_and_yaml_yaml_override(self, temp_dir, monkeypatch):
        """Test from_env_and_yaml with YAML override."""
        # Set env vars
        monkeypatch.setenv("VAE_MODEL_NAME", "env-vae")
        monkeypatch.setenv("VAE_BATCH_SIZE", "64")

        # Create YAML that overrides
        yaml_file = temp_dir / "override.yaml"
        yaml_content = """
model_name: yaml-override-vae
batch_size: 256
learning_rate: 0.003
"""
        with Path(yaml_file).open("w") as f:
            f.write(yaml_content)

        config = VAEConfig.from_env_and_yaml(yaml_path=yaml_file)

        assert config.model_name == "yaml-override-vae"  # YAML wins
        assert config.batch_size == 256  # YAML wins
        assert config.learning_rate == 0.003  # From YAML

    def test_from_env_and_yaml_with_env_file(self, temp_dir):
        """Test from_env_and_yaml with custom .env file."""
        custom_env = temp_dir / "custom.env"
        env_content = """
VAE_MODEL_NAME=custom-env-vae
VAE_BATCH_SIZE=1024
"""
        with Path(custom_env).open("w") as f:
            f.write(env_content)

        config = VAEConfig.from_env_and_yaml(env_file=str(custom_env))

        assert config.model_name == "custom-env-vae"
        assert config.batch_size == 1024


class TestConfigPrecedence:
    """Test configuration precedence: env vars > .env file > defaults."""

    def test_env_var_precedence_over_env_file(self, temp_dir, monkeypatch):
        """Test that environment variables take precedence over .env file."""
        # Create .env file
        env_file = temp_dir / ".env"
        env_content = """
VAE_MODEL_NAME=from-env-file
VAE_BATCH_SIZE=64
"""
        with Path(env_file).open("w") as f:
            f.write(env_content)

        # Set environment variable (should override .env file)
        monkeypatch.setenv("VAE_MODEL_NAME", "from-env-var")

        config = VAEConfig(_env_file=env_file)

        assert config.model_name == "from-env-var"  # Env var wins
        assert config.batch_size == 64  # From .env file

    def test_yaml_precedence_over_all(self, temp_dir, monkeypatch):
        """Test that YAML takes precedence over env vars and .env file."""
        # Create .env file
        env_file = temp_dir / ".env"
        env_content = """
VAE_MODEL_NAME=from-env-file
VAE_BATCH_SIZE=64
"""
        with Path(env_file).open("w") as f:
            f.write(env_content)

        # Set environment variable
        monkeypatch.setenv("VAE_MODEL_NAME", "from-env-var")
        monkeypatch.setenv("VAE_BATCH_SIZE", "128")

        # Create YAML config
        yaml_file = temp_dir / "config.yaml"
        yaml_content = """
model_name: from-yaml
batch_size: 256
"""
        with Path(yaml_file).open("w") as f:
            f.write(yaml_content)

        config = VAEConfig.from_yaml(yaml_file)

        assert config.model_name == "from-yaml"  # YAML wins
        assert config.batch_size == 256  # YAML wins


class TestCaseSensitivity:
    """Test case insensitive environment variable handling."""

    def test_case_insensitive_env_vars(self, monkeypatch):
        """Test that environment variables are case insensitive."""
        # Test various case combinations
        monkeypatch.setenv("VAE_MODEL_NAME", "test-model")
        monkeypatch.setenv("vae_batch_size", "32")  # lowercase
        monkeypatch.setenv("Vae_Learning_Rate", "0.001")  # mixed case

        config = VAEConfig()

        assert config.model_name == "test-model"
        assert config.batch_size == 32
        assert config.learning_rate == 0.001


class TestValidationWithEnvVars:
    """Test that validation still works with environment variables."""

    def test_validation_error_from_env_vars(self, monkeypatch):
        """Test that validation errors are raised for invalid env var values."""
        # Set invalid values
        monkeypatch.setenv("VAE_BATCH_SIZE", "-1")  # Invalid: must be >= 1
        monkeypatch.setenv("VAE_LEARNING_RATE", "0.0")  # Invalid: must be > 0

        with pytest.raises(
            ValidationError, match="validation error"
        ):  # Should raise validation error
            VAEConfig()

    def test_type_conversion_from_env_vars(self, monkeypatch):
        """Test that environment variables are properly type converted."""
        monkeypatch.setenv("VAE_BATCH_SIZE", "64")  # String -> int
        monkeypatch.setenv("VAE_LEARNING_RATE", "0.001")  # String -> float
        monkeypatch.setenv("ENGINE_ENABLE_MEMORY_EFFICIENT_ATTENTION", "false")  # String -> bool

        vae_config = VAEConfig()
        engine_config = EngineConfig()

        assert isinstance(vae_config.batch_size, int)
        assert vae_config.batch_size == 64
        assert isinstance(vae_config.learning_rate, float)
        assert vae_config.learning_rate == 0.001
        assert isinstance(engine_config.enable_memory_efficient_attention, bool)
        assert engine_config.enable_memory_efficient_attention is False
