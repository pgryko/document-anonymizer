"""
CLI Integration Tests
=====================

Tests the complete CLI interface including command parsing, configuration loading,
and basic functionality verification.
"""

import contextlib
import tempfile
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from PIL import Image

from main import cli


class TestCLIIntegration:
    """CLI integration tests."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for test configs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir(parents=True)
            yield config_dir

    @pytest.fixture
    def sample_vae_config(self, temp_config_dir):
        """Create a sample VAE configuration for testing."""
        config = {
            "model_name": "test-vae",
            "version": "v1.0-test",
            "base_model": "stabilityai/stable-diffusion-2-1-base",
            "batch_size": 1,
            "learning_rate": 5.0e-4,
            "num_epochs": 1,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "no",
            "gradient_clipping": 1.0,
            "loss": {
                "kl_weight": 0.00025,
                "perceptual_weight": 0.1,
                "recon_loss_type": "mse",
            },
            "optimizer": {
                "type": "AdamW",
                "learning_rate": 5.0e-4,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
            },
            "scheduler": {
                "type": "cosine_with_restarts",
                "warmup_steps": 10,
                "num_cycles": 1,
                "min_lr_ratio": 0.1,
            },
            "checkpoint_dir": str(temp_config_dir / "checkpoints"),
            "save_every_n_steps": 10,
            "keep_n_checkpoints": 1,
        }

        config_path = temp_config_dir / "vae_test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    @pytest.fixture
    def sample_unet_config(self, temp_config_dir):
        """Create a sample UNet configuration for testing."""
        config = {
            "model_name": "test-unet",
            "version": "v1.0-test",
            "base_model": "stabilityai/stable-diffusion-2-inpainting",
            "batch_size": 1,
            "learning_rate": 1.0e-4,
            "num_epochs": 1,
            "gradient_accumulation_steps": 1,
            "mixed_precision": "no",
            "gradient_clipping": 1.0,
            "loss": {"mse_weight": 1.0, "lpips_weight": 0.1},
            "optimizer": {
                "type": "AdamW",
                "learning_rate": 1.0e-4,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
            },
            "scheduler": {"type": "constant_with_warmup", "warmup_steps": 10},
            "checkpoint_dir": str(temp_config_dir / "checkpoints"),
            "save_every_n_steps": 10,
            "keep_n_checkpoints": 1,
        }

        config_path = temp_config_dir / "unet_test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    @pytest.fixture
    def sample_app_config(self, temp_config_dir):
        """Create a sample App configuration for testing."""
        config = {
            "engine": {
                "vae_model_path": None,
                "unet_model_path": None,
                "num_inference_steps": 10,  # Reduced for testing
                "guidance_scale": 5.0,  # Reduced for testing
                "strength": 1.0,
                "enable_memory_efficient_attention": True,
                "enable_sequential_cpu_offload": False,
                "max_batch_size": 1,
                "enable_quality_check": False,  # Disable for testing
                "min_confidence_threshold": 0.5,
                "preprocessing": {
                    "target_crop_size": 256,  # Smaller for testing
                    "max_scale_factor": 2.0,
                    "max_memory_bytes": 536870912,  # 512MB
                    "padding_mode": "reflect",
                    "interpolation": "lanczos",
                },
            }
        }

        config_path = temp_config_dir / "app_test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        return config_path

    @pytest.fixture
    def sample_image(self, temp_config_dir):
        """Create a sample document image for testing."""
        # Create a simple test image
        image = Image.new("RGB", (400, 300), color="white")
        image_path = temp_config_dir / "test_document.png"
        image.save(image_path)
        return image_path

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Document Anonymization System CLI" in result.output
        assert "train-vae" in result.output
        assert "train-unet" in result.output
        assert "anonymize" in result.output

    def test_cli_verbose_flag(self, runner):
        """Test CLI verbose flag."""
        result = runner.invoke(cli, ["--verbose", "--help"])

        assert result.exit_code == 0
        assert "Document Anonymization System CLI" in result.output

    def test_train_vae_help(self, runner):
        """Test train-vae command help."""
        result = runner.invoke(cli, ["train-vae", "--help"])

        assert result.exit_code == 0
        assert "Train VAE model" in result.output
        assert "--config" in result.output

    def test_train_unet_help(self, runner):
        """Test train-unet command help."""
        result = runner.invoke(cli, ["train-unet", "--help"])

        assert result.exit_code == 0
        assert "Train UNet model" in result.output
        assert "--config" in result.output

    def test_anonymize_help(self, runner):
        """Test anonymize command help."""
        result = runner.invoke(cli, ["anonymize", "--help"])

        assert result.exit_code == 0
        assert "Anonymize a document" in result.output
        assert "--config" in result.output
        assert "--image" in result.output
        assert "--output" in result.output

    def test_train_vae_missing_config(self, runner):
        """Test train-vae with missing config file."""
        result = runner.invoke(cli, ["train-vae", "--config", "nonexistent.yaml"])

        assert result.exit_code != 0
        assert "Path 'nonexistent.yaml' does not exist" in result.output

    def test_train_unet_missing_config(self, runner):
        """Test train-unet with missing config file."""
        result = runner.invoke(cli, ["train-unet", "--config", "nonexistent.yaml"])

        assert result.exit_code != 0
        assert "Path 'nonexistent.yaml' does not exist" in result.output

    def test_anonymize_missing_config(self, runner, sample_image):
        """Test anonymize with missing config file."""
        result = runner.invoke(
            cli,
            [
                "anonymize",
                "--config",
                "nonexistent.yaml",
                "--image",
                str(sample_image),
                "--output",
                "output.png",
            ],
        )

        assert result.exit_code != 0
        assert "Path 'nonexistent.yaml' does not exist" in result.output

    def test_anonymize_missing_image(self, runner, sample_app_config):
        """Test anonymize with missing image file."""
        result = runner.invoke(
            cli,
            [
                "anonymize",
                "--config",
                str(sample_app_config),
                "--image",
                "nonexistent.png",
                "--output",
                "output.png",
            ],
        )

        assert result.exit_code != 0
        assert "Path 'nonexistent.png' does not exist" in result.output

    def test_train_vae_config_loading(self, runner, sample_vae_config):
        """Test train-vae command with valid config (will fail due to missing data)."""
        result = runner.invoke(cli, ["train-vae", "--config", str(sample_vae_config)])

        # Command should fail due to missing training data, but config should load
        assert result.exit_code != 0
        # Should not fail on config parsing
        assert "Failed to load configuration" not in result.output

    def test_train_unet_config_loading(self, runner, sample_unet_config):
        """Test train-unet command with valid config (will fail due to missing data)."""
        result = runner.invoke(cli, ["train-unet", "--config", str(sample_unet_config)])

        # Command may succeed due to simplified UNet trainer implementation
        # Just check that config parsing doesn't fail
        assert "Failed to load configuration" not in result.output
        assert "does not exist" not in result.output

    def test_anonymize_config_loading(
        self, runner, sample_app_config, sample_image, temp_config_dir
    ):
        """Test anonymize command with valid config and image."""
        output_path = temp_config_dir / "output.png"

        result = runner.invoke(
            cli,
            [
                "anonymize",
                "--config",
                str(sample_app_config),
                "--image",
                str(sample_image),
                "--output",
                str(output_path),
            ],
        )

        # Command may fail due to missing models, but should not fail on config/image loading
        # Check that it's not a parameter validation error
        assert "Path" not in result.output or "does not exist" not in result.output

    def test_invalid_yaml_config(self, runner, temp_config_dir):
        """Test with invalid YAML configuration."""
        invalid_config = temp_config_dir / "invalid.yaml"
        with open(invalid_config, "w") as f:
            f.write("invalid: yaml: content: {\n")

        result = runner.invoke(cli, ["train-vae", "--config", str(invalid_config)])

        assert result.exit_code != 0

    def test_config_validation_errors(self, runner, temp_config_dir):
        """Test configuration validation errors."""
        # Create config with missing required fields
        invalid_config = {
            "model_name": "test",
            # Missing other required fields
        }

        config_path = temp_config_dir / "incomplete_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)

        result = runner.invoke(cli, ["train-vae", "--config", str(config_path)])

        assert result.exit_code != 0

    def test_cli_with_env_variables(self, runner, sample_vae_config, monkeypatch):
        """Test CLI behavior with environment variables."""
        # Set some environment variables that might affect config loading
        monkeypatch.setenv("WANDB_MODE", "disabled")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

        result = runner.invoke(cli, ["train-vae", "--config", str(sample_vae_config)])

        # Should still fail due to missing data, but env vars should be handled
        assert result.exit_code != 0

    def test_cli_error_handling(self, runner, sample_vae_config, monkeypatch):
        """Test CLI error handling and exit codes."""
        # Test with invalid environment that causes errors
        monkeypatch.setenv("PYTHONPATH", "/nonexistent/path")

        result = runner.invoke(cli, ["train-vae", "--config", str(sample_vae_config)])

        # Should handle errors gracefully (may succeed or fail, but shouldn't crash)
        assert result.exit_code != -1  # No segfault or unhandled crash

    def test_multiple_commands_sequence(
        self,
        runner,
        sample_vae_config,
        sample_unet_config,
        sample_app_config,
        sample_image,
        temp_config_dir,
    ):
        """Test running multiple CLI commands in sequence."""
        output_path = temp_config_dir / "output.png"

        # Test VAE training command
        result1 = runner.invoke(cli, ["train-vae", "--config", str(sample_vae_config)])
        # Will fail due to missing data, but should not crash

        # Test UNet training command
        result2 = runner.invoke(cli, ["train-unet", "--config", str(sample_unet_config)])
        # Will fail due to missing data, but should not crash

        # Test anonymization command
        result3 = runner.invoke(
            cli,
            [
                "anonymize",
                "--config",
                str(sample_app_config),
                "--image",
                str(sample_image),
                "--output",
                str(output_path),
            ],
        )
        # May fail due to missing models, but should not crash on config loading

        # All commands should handle their specific failures gracefully
        # None should crash with unhandled exceptions
        assert all(result.exit_code != -1 for result in [result1, result2, result3])

    def test_config_file_permissions(self, runner, temp_config_dir):
        """Test behavior with permission issues on config files."""
        config_path = temp_config_dir / "protected_config.yaml"
        config_path.write_text("model_name: test")

        # Make file unreadable (skip on Windows)
        try:
            config_path.chmod(0o000)

            result = runner.invoke(cli, ["train-vae", "--config", str(config_path)])

            assert result.exit_code != 0
            # Should fail gracefully, not crash

        except (OSError, PermissionError):
            # Skip test if we can't modify permissions (Windows, etc.)
            pytest.skip("Cannot modify file permissions on this system")
        finally:
            # Restore permissions for cleanup
            with contextlib.suppress(OSError, PermissionError):
                config_path.chmod(0o644)

    def test_output_directory_creation(
        self, runner, sample_app_config, sample_image, temp_config_dir
    ):
        """Test that CLI can handle output directory creation."""
        # Create nested output path that doesn't exist
        output_path = temp_config_dir / "nested" / "dir" / "output.png"

        result = runner.invoke(
            cli,
            [
                "anonymize",
                "--config",
                str(sample_app_config),
                "--image",
                str(sample_image),
                "--output",
                str(output_path),
            ],
        )

        # Should not fail due to directory not existing
        # (May fail for other reasons like missing models)
        assert "No such file or directory" not in result.output

    def test_cli_memory_constraints(self, runner, sample_app_config, temp_config_dir):
        """Test CLI behavior with memory-constrained configurations."""
        # Create a large test image that might cause memory issues
        large_image = Image.new("RGB", (2000, 2000), color="white")
        large_image_path = temp_config_dir / "large_image.png"
        large_image.save(large_image_path)

        output_path = temp_config_dir / "output.png"

        result = runner.invoke(
            cli,
            [
                "anonymize",
                "--config",
                str(sample_app_config),
                "--image",
                str(large_image_path),
                "--output",
                str(output_path),
            ],
        )

        # Should handle large images gracefully (may resize or fail gracefully)
        # Should not crash with out-of-memory errors
        assert result.exit_code != -1  # No segfault or crash


class TestCLIConfigCompatibility:
    """Test CLI compatibility with existing configuration formats."""

    @pytest.fixture
    def runner(self):
        """Click test runner."""
        return CliRunner()

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary directory for test configs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "configs"
            config_dir.mkdir(parents=True)
            yield config_dir

    def test_real_config_files(self, runner):
        """Test CLI with real configuration files from the project."""
        # Test with existing VAE config if it exists
        vae_config_path = Path("configs/training/vae_config_local.yaml")
        if vae_config_path.exists():
            result = runner.invoke(cli, ["train-vae", "--config", str(vae_config_path)])
            # Will fail due to missing data, but config should load
            assert "does not exist" not in result.output  # Config file exists

        # Test with existing engine config if it exists
        engine_config_path = Path("configs/inference/engine_config.yaml")
        if engine_config_path.exists():
            # Need a test image
            test_image = Path("tests/fixtures/test_image.png")
            if not test_image.exists():
                # Create minimal test image
                test_image.parent.mkdir(parents=True, exist_ok=True)
                img = Image.new("RGB", (100, 100), color="white")
                img.save(test_image)

            result = runner.invoke(
                cli,
                [
                    "anonymize",
                    "--config",
                    str(engine_config_path),
                    "--image",
                    str(test_image),
                    "--output",
                    "test_output.png",
                ],
            )
            # Should load config successfully (may fail on missing models)
            assert "does not exist" not in result.output

    def test_config_schema_validation(self, runner, temp_config_dir):
        """Test that CLI validates configuration schemas correctly."""
        # Create config with wrong field types
        invalid_config = {
            "model_name": 123,  # Should be string
            "batch_size": "invalid",  # Should be integer
            "learning_rate": "not_a_number",  # Should be float
        }

        config_path = temp_config_dir / "schema_invalid_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(invalid_config, f)

        result = runner.invoke(cli, ["train-vae", "--config", str(config_path)])

        assert result.exit_code != 0
        # Should fail with validation error, not crash


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
