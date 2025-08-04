"""
Unit tests for VAE trainer - Imperative style.

Tests the corrected VAE trainer with KL divergence loss fix.
"""

import logging
from unittest.mock import Mock, patch

import pytest
import torch

from src.anonymizer.core.config import VAEConfig
from src.anonymizer.core.exceptions import (
    ModelLoadError,
    TrainingError,
    ValidationError,
)
from src.anonymizer.core.models import ModelArtifacts
from src.anonymizer.training.vae_trainer import PerceptualLoss, VAETrainer


class TestPerceptualLoss:
    """Test PerceptualLoss implementation."""

    def test_perceptual_loss_initialization_with_torchvision(self, device):
        """Test perceptual loss initialization when torchvision is available."""
        with patch("torchvision.models.vgg16") as mock_vgg:
            # Mock VGG model
            mock_features = Mock()
            mock_vgg.return_value.features.__getitem__.return_value = mock_features
            mock_features.to.return_value = mock_features
            mock_features.eval.return_value = None
            mock_features.parameters.return_value = []

            perceptual_loss = PerceptualLoss(device)

            assert perceptual_loss.vgg is not None
            mock_vgg.assert_called_once_with(pretrained=True)

    def test_perceptual_loss_initialization_without_torchvision(self, device):
        """Test perceptual loss initialization when torchvision is not available."""
        with patch("torchvision.models.vgg16", side_effect=ImportError):
            perceptual_loss = PerceptualLoss(device)

            assert perceptual_loss.vgg is None

    def test_perceptual_loss_forward_without_vgg(self, device):
        """Test perceptual loss forward pass when VGG is not available."""
        perceptual_loss = PerceptualLoss(device)
        perceptual_loss.vgg = None  # Simulate no VGG

        pred = torch.randn(2, 3, 64, 64, device=device)
        target = torch.randn(2, 3, 64, 64, device=device)

        loss = perceptual_loss(pred, target)

        assert loss.item() == 0.0
        assert loss.device == device

    def test_perceptual_loss_normalize_rgb(self, device):
        """Test image normalization for RGB images."""
        perceptual_loss = PerceptualLoss(device)

        # Test RGB image (3 channels)
        x = torch.randn(2, 3, 64, 64, device=device)
        normalized = perceptual_loss._normalize(x)

        assert normalized.shape == (2, 3, 64, 64)
        assert normalized.device == device

    def test_perceptual_loss_normalize_grayscale(self, device):
        """Test image normalization for grayscale images."""
        perceptual_loss = PerceptualLoss(device)

        # Test grayscale image (1 channel) - should be repeated to 3 channels
        x = torch.randn(2, 1, 64, 64, device=device)
        normalized = perceptual_loss._normalize(x)

        assert normalized.shape == (2, 3, 64, 64)
        assert normalized.device == device

    def test_perceptual_loss_normalize_rgba(self, device):
        """Test image normalization for RGBA images."""
        perceptual_loss = PerceptualLoss(device)

        # Test RGBA image (4 channels) - should drop alpha channel
        x = torch.randn(2, 4, 64, 64, device=device)
        normalized = perceptual_loss._normalize(x)

        assert normalized.shape == (2, 3, 64, 64)
        assert normalized.device == device


class TestVAETrainer:
    """Test VAE trainer with critical bug fixes."""

    def test_vae_trainer_initialization(self, vae_config):
        """Test VAE trainer initialization."""
        trainer = VAETrainer(vae_config)

        assert trainer.config == vae_config
        assert trainer.device.type in ["cuda", "cpu"]
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
        assert trainer.best_loss == float("inf")
        assert trainer.vae is None  # Not initialized yet
        assert trainer.optimizer is None
        assert trainer.scheduler is None

    def test_vae_trainer_setup_distributed(self, vae_config):
        """Test distributed training setup."""
        trainer = VAETrainer(vae_config)

        with patch("src.anonymizer.training.vae_trainer.Accelerator") as mock_accelerator:
            mock_accelerator.return_value.device = torch.device("cpu")

            trainer.setup_distributed()

            assert trainer.accelerator is not None
            mock_accelerator.assert_called_once()

    @patch("src.anonymizer.training.vae_trainer.AutoencoderKL")
    def test_vae_trainer_initialize_vae(self, mock_vae_class, vae_config):
        """Test VAE model initialization."""
        # Mock VAE model
        mock_vae = Mock()
        mock_vae.parameters.return_value = [torch.randn(10)]
        mock_vae_class.from_pretrained.return_value = mock_vae

        trainer = VAETrainer(vae_config)
        vae = trainer._initialize_vae()

        assert vae is not None
        mock_vae_class.from_pretrained.assert_called_once_with(
            vae_config.base_model, subfolder="vae"
        )
        mock_vae.train.assert_called_once()

    @patch("src.anonymizer.training.vae_trainer.AutoencoderKL")
    def test_vae_trainer_initialize_vae_failure(self, mock_vae_class, vae_config):
        """Test VAE initialization failure handling."""
        mock_vae_class.from_pretrained.side_effect = Exception("Download failed")

        trainer = VAETrainer(vae_config)

        with pytest.raises(ModelLoadError, match="Failed to initialize VAE"):
            trainer._initialize_vae()

    def test_vae_trainer_setup_optimizer_adamw(self, vae_config):
        """Test optimizer setup with AdamW."""
        trainer = VAETrainer(vae_config)

        # Mock VAE
        mock_vae = Mock()
        mock_vae.parameters.return_value = [torch.randn(10, requires_grad=True)]
        trainer.vae = mock_vae

        optimizer = trainer._setup_optimizer()

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == vae_config.learning_rate
        assert optimizer.param_groups[0]["weight_decay"] == vae_config.optimizer.weight_decay

    def test_vae_trainer_setup_optimizer_unsupported(self, vae_config):
        """Test optimizer setup with unsupported optimizer type."""
        vae_config.optimizer.type = "SGD"  # Unsupported
        trainer = VAETrainer(vae_config)

        # Mock VAE
        mock_vae = Mock()
        trainer.vae = mock_vae

        with pytest.raises(ValidationError, match="Unsupported optimizer"):
            trainer._setup_optimizer()

    def test_vae_trainer_setup_optimizer_without_vae(self, vae_config):
        """Test optimizer setup without VAE initialization."""
        trainer = VAETrainer(vae_config)

        with pytest.raises(TrainingError, match="VAE must be initialized before optimizer"):
            trainer._setup_optimizer()

    def test_vae_trainer_compute_loss_critical_fix(self, vae_config, device):
        """Test the critical KL divergence loss fix."""
        trainer = VAETrainer(vae_config)

        # Mock VAE with proper encode/decode behavior
        mock_vae = Mock()

        # Mock posterior with KL divergence
        mock_posterior = Mock()
        mock_posterior.sample.return_value = torch.randn(2, 4, 64, 64, device=device)
        mock_posterior.kl.return_value = torch.randn(2, 4, 64, 64, device=device)

        # Mock encode/decode
        mock_encode_result = Mock()
        mock_encode_result.latent_dist = mock_posterior
        mock_vae.encode.return_value = mock_encode_result

        mock_decode_result = Mock()
        mock_decode_result.sample = torch.randn(2, 3, 512, 512, device=device)
        mock_vae.decode.return_value = mock_decode_result

        trainer.vae = mock_vae
        trainer.perceptual_loss = Mock(return_value=torch.tensor(0.1, device=device))

        # Create batch
        batch = {"images": torch.randn(2, 3, 512, 512, device=device)}

        losses = trainer._compute_loss(batch)

        # CRITICAL TEST: Verify KL divergence loss is included
        assert "kl_loss" in losses
        assert "recon_loss" in losses
        assert "perceptual_loss" in losses
        assert "total_loss" in losses

        # Verify KL loss is properly computed
        mock_posterior.kl.assert_called_once()

        # Verify total loss includes all components
        total_loss = losses["total_loss"]
        recon_loss = losses["recon_loss"]
        kl_loss = losses["kl_loss"]
        perceptual_loss = losses["perceptual_loss"]

        # Verify loss computation: total = recon + beta*kl + lambda*perceptual
        expected_total = (
            recon_loss
            + trainer.config.loss.kl_weight * kl_loss
            + trainer.config.loss.perceptual_weight * perceptual_loss
        )

        assert torch.allclose(total_loss, expected_total, atol=1e-6)

    def test_vae_trainer_compute_loss_validation_error(self, vae_config, device):
        """Test loss computation with validation errors."""
        trainer = VAETrainer(vae_config)

        # Mock VAE that returns NaN
        mock_vae = Mock()
        mock_posterior = Mock()
        mock_posterior.sample.return_value = torch.randn(2, 4, 64, 64, device=device)
        mock_posterior.kl.return_value = torch.tensor(float("nan"), device=device)  # NaN KL

        mock_encode_result = Mock()
        mock_encode_result.latent_dist = mock_posterior
        mock_vae.encode.return_value = mock_encode_result

        mock_decode_result = Mock()
        mock_decode_result.sample = torch.randn(2, 3, 512, 512, device=device)
        mock_vae.decode.return_value = mock_decode_result

        trainer.vae = mock_vae
        trainer.perceptual_loss = Mock(return_value=torch.tensor(0.1, device=device))

        batch = {"images": torch.randn(2, 3, 512, 512, device=device)}

        with pytest.raises(TrainingError, match="Invalid loss detected"):
            trainer._compute_loss(batch)

    def test_vae_trainer_train_step(self, vae_config, device):
        """Test single training step."""
        trainer = VAETrainer(vae_config)

        # Mock components
        mock_vae = Mock()
        mock_vae.parameters.return_value = [torch.randn(10, requires_grad=True)]
        trainer.vae = mock_vae

        trainer.optimizer = Mock()
        trainer.optimizer.param_groups = [{"lr": 1e-4}]
        trainer.scheduler = Mock()

        # Mock loss computation
        with patch.object(trainer, "_compute_loss") as mock_compute_loss:
            mock_losses = {
                "total_loss": torch.tensor(0.5, device=device, requires_grad=True),
                "recon_loss": torch.tensor(0.3, device=device, requires_grad=True),
                "kl_loss": torch.tensor(0.1, device=device, requires_grad=True),
                "perceptual_loss": torch.tensor(0.1, device=device, requires_grad=True),
            }
            mock_compute_loss.return_value = mock_losses

            batch = {"images": torch.randn(2, 3, 512, 512, device=device)}

            trainer.train_step(batch)

            # Verify optimizer was called
            trainer.optimizer.step.assert_called_once()
            trainer.optimizer.zero_grad.assert_called_once()

    def test_vae_trainer_train_step_with_accelerator(self, vae_config, device):
        """Test training step with accelerator."""
        trainer = VAETrainer(vae_config)

        # Mock accelerator
        trainer.accelerator = Mock()

        # Mock components
        trainer.vae = Mock()
        trainer.optimizer = Mock()
        trainer.optimizer.param_groups = [{"lr": 1e-4}]

        # Mock loss computation
        with patch.object(trainer, "_compute_loss") as mock_compute_loss:
            mock_losses = {
                "total_loss": torch.tensor(0.5, device=device, requires_grad=True),
                "recon_loss": torch.tensor(0.3, device=device, requires_grad=True),
                "kl_loss": torch.tensor(0.1, device=device, requires_grad=True),
                "perceptual_loss": torch.tensor(0.1, device=device, requires_grad=True),
            }
            mock_compute_loss.return_value = mock_losses

            batch = {"images": torch.randn(2, 3, 512, 512, device=device)}

            trainer.train_step(batch)

            # Verify accelerator was used
            trainer.accelerator.backward.assert_called_once()
            trainer.accelerator.clip_grad_norm_.assert_called_once()

    def test_vae_trainer_validate(self, vae_config, device):
        """Test validation function."""
        trainer = VAETrainer(vae_config)

        # Mock VAE
        mock_vae = Mock()
        trainer.vae = mock_vae

        # Mock dataloader
        mock_batch = {"images": torch.randn(2, 3, 512, 512, device=device)}
        mock_dataloader = [mock_batch, mock_batch]  # Two batches

        # Mock loss computation
        with patch.object(trainer, "_compute_loss") as mock_compute_loss:
            mock_losses = {
                "total_loss": torch.tensor(0.5, device=device),
                "recon_loss": torch.tensor(0.3, device=device),
                "kl_loss": torch.tensor(0.1, device=device),
                "perceptual_loss": torch.tensor(0.1, device=device),
            }
            mock_compute_loss.return_value = mock_losses

            val_losses = trainer.validate(mock_dataloader)

            # Verify validation results
            assert "total_loss" in val_losses
            assert "recon_loss" in val_losses
            assert "kl_loss" in val_losses
            assert "perceptual_loss" in val_losses

            # Should be average of two batches
            assert val_losses["total_loss"] == 0.5

            # Verify VAE was set to eval and back to train
            mock_vae.eval.assert_called_once()
            mock_vae.train.assert_called_once()

    def test_vae_trainer_save_checkpoint(self, vae_config, temp_dir):
        """Test checkpoint saving."""
        trainer = VAETrainer(vae_config)
        trainer.config.checkpoint_dir = temp_dir
        trainer.global_step = 1000
        trainer.current_epoch = 5
        trainer.best_loss = 0.3

        # Mock VAE
        mock_vae = Mock()
        mock_state_dict = {"layer.weight": torch.randn(10, 10)}
        mock_vae.state_dict.return_value = mock_state_dict
        trainer.vae = mock_vae

        with patch("safetensors.torch.save_file") as mock_save:
            save_path = trainer.save_checkpoint()

            # Verify save was called
            mock_save.assert_called_once()

            # Verify save path
            assert save_path.parent == temp_dir
            assert "vae_step_1000" in save_path.name

            # Verify training state was saved
            state_file = save_path.with_suffix(".json")
            assert state_file.exists()

    def test_vae_trainer_save_model(self, vae_config, temp_dir):
        """Test final model saving."""
        trainer = VAETrainer(vae_config)
        trainer.config.checkpoint_dir = temp_dir
        trainer.global_step = 5000
        trainer.current_epoch = 100
        trainer.best_loss = 0.2

        # Mock VAE
        mock_vae = Mock()
        trainer.vae = mock_vae

        with patch.object(trainer, "save_checkpoint") as mock_save_checkpoint:
            mock_save_checkpoint.return_value = temp_dir / "final_model" / "model.safetensors"

            artifacts = trainer.save_model()

            # Verify artifacts
            assert isinstance(artifacts, ModelArtifacts)
            assert artifacts.model_name == vae_config.model_name
            assert artifacts.version == vae_config.version
            assert artifacts.metadata["final_step"] == 5000
            assert artifacts.metadata["final_epoch"] == 100
            assert artifacts.metadata["best_loss"] == 0.2
            assert artifacts.metadata["training_completed"] is True

    def test_vae_trainer_critical_bug_fixes_verification(self):
        """Test that critical bugs are actually fixed in the trainer."""
        # Use default config to verify production settings, not test fixture
        config = VAEConfig()
        trainer = VAETrainer(config)

        # CRITICAL FIX 1: Learning rate should be 5e-4, not 5e-6
        assert trainer.config.learning_rate == 5e-4

        # CRITICAL FIX 2: Batch size should be 16, not 2
        assert trainer.config.batch_size == 16

        # CRITICAL FIX 3: KL divergence weight should be present
        assert trainer.config.loss.kl_weight > 0
        assert trainer.config.loss.kl_weight == 0.00025

        # CRITICAL FIX 4: Perceptual loss weight should be present
        assert trainer.config.loss.perceptual_weight > 0
        assert trainer.config.loss.perceptual_weight == 0.1

    def test_vae_trainer_memory_cleanup(self, vae_config):
        """Test GPU memory cleanup."""
        trainer = VAETrainer(vae_config)

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_empty_cache,
            patch("torch.cuda.synchronize") as mock_synchronize,
        ):

            # This should trigger memory cleanup in the finally block
            try:
                trainer.train([], None)
            except Exception as e:
                logging.debug(f"Expected exception during test cleanup: {e}")

            # Memory cleanup should have been called
            mock_empty_cache.assert_called()
            mock_synchronize.assert_called()

    def test_vae_trainer_error_handling_in_train_step(self, vae_config, device):
        """Test error handling in training step."""
        trainer = VAETrainer(vae_config)

        # Mock components that will fail
        trainer.vae = Mock()
        trainer.optimizer = Mock()
        trainer.optimizer.step.side_effect = RuntimeError("CUDA out of memory")

        with patch.object(trainer, "_compute_loss") as mock_compute_loss:
            mock_losses = {
                "total_loss": torch.tensor(0.5, device=device, requires_grad=True),
                "recon_loss": torch.tensor(0.3, device=device, requires_grad=True),
                "kl_loss": torch.tensor(0.1, device=device, requires_grad=True),
                "perceptual_loss": torch.tensor(0.1, device=device, requires_grad=True),
            }
            mock_compute_loss.return_value = mock_losses

            batch = {"images": torch.randn(2, 3, 512, 512, device=device)}

            with pytest.raises(TrainingError, match="Training step failed"):
                trainer.train_step(batch)
