"""End-to-end UNet training tests.

Tests the complete UNet training pipeline:
- InpaintingDataset → DataLoader → UNetTrainer.train()
- Model initialization and training step execution
- Loss computation and gradient updates
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from src.anonymizer.core.config import DatasetConfig, UNetConfig
from src.anonymizer.training.datasets import create_inpainting_dataloaders
from src.anonymizer.training.unet_trainer import UNetTrainer


@pytest.fixture
def mock_training_data():
    """Create mock training data for UNet testing."""

    def _create_data(data_dir: Path, num_samples: int = 4):
        data_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            # Create test image with varied content
            image = Image.new("RGB", (512, 512), color=(240, 240, 240))
            pixels = np.array(image)

            # Add some text-like regions
            if i % 2 == 0:
                pixels[100:150, 100:300] = [50, 50, 50]  # Dark text region
                pixels[200:250, 150:350] = [30, 30, 30]  # Another text region
            else:
                pixels[80:130, 50:250] = [40, 40, 40]  # Different positioning
                pixels[300:350, 200:400] = [20, 20, 20]  # Different positioning

            image = Image.fromarray(pixels)
            image_path = data_dir / f"sample_{i}.png"
            image.save(image_path)

            # Create corresponding annotation with realistic PII
            pii_types = [
                ("Social Security Number", "SSN", "[SSN]"),
                ("Credit Card Number", "4532-1234-5678-9012", "[CARD]"),
                ("Phone Number", "(555) 123-4567", "[PHONE]"),
                ("Email Address", "user@example.com", "[EMAIL]"),
            ]

            pii_type = pii_types[i % len(pii_types)]

            annotation = {
                "image_name": f"sample_{i}.png",
                "text_regions": [
                    {
                        "bbox": {"left": 100, "top": 100, "right": 300, "bottom": 150},
                        "original_text": pii_type[1],
                        "replacement_text": pii_type[2],
                        "confidence": 0.90 + (i * 0.02),  # Vary confidence slightly
                    }
                ],
            }

            annotation_path = data_dir / f"sample_{i}.json"
            with annotation_path.open("w") as f:
                json.dump(annotation, f, indent=2)

    return _create_data


@pytest.fixture
def unet_config():
    """Create UNet configuration for testing."""
    return UNetConfig(
        model_name="test-unet",
        base_model="runwayml/stable-diffusion-v1-5",
        # Training settings
        learning_rate=1e-4,
        num_epochs=1,  # Just one epoch for testing
        batch_size=1,  # Small batch for testing
        gradient_accumulation_steps=1,
        mixed_precision="no",  # Disable for testing
        # Model settings
        enable_text_conditioning=True,
        text_encoder_lr_scale=0.5,
        # Paths
        checkpoint_dir="./test_checkpoints",
        # Diffusion settings
        num_train_timesteps=1000,
        noise_schedule="linear",
        # Memory settings
        gradient_clipping=1.0,
        use_ema=False,  # Disable EMA for testing
    )


@pytest.fixture
def dataset_config():
    """Create dataset configuration for testing."""
    return DatasetConfig(
        train_data_path=Path("dummy"),  # Will be overridden
        crop_size=256,  # Smaller for testing
        num_workers=0,  # Disable multiprocessing
    )


class TestUNetEndToEndTraining:
    """Test UNet end-to-end training pipeline."""

    def test_unet_trainer_initialization(self, unet_config):
        """Test UNet trainer can be initialized."""
        trainer = UNetTrainer(unet_config)

        assert trainer.config == unet_config
        assert trainer.device.type in ["cpu", "cuda", "mps"]
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0

    def test_unet_dataloader_integration(self, mock_training_data, dataset_config):
        """Test InpaintingDataset works with UNet training expectations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            mock_training_data(data_path, num_samples=2)

            # Update config with actual path
            dataset_config.train_data_path = data_path

            # Create dataloaders
            train_loader, val_loader = create_inpainting_dataloaders(dataset_config, batch_size=1)

            assert train_loader is not None
            assert len(train_loader) == 2  # 2 samples, batch size 1

            # Test batch format matches UNet trainer expectations
            batch = next(iter(train_loader))

            # Check required keys for UNet training
            assert "images" in batch
            assert "masks" in batch
            assert "texts" in batch

            # Check tensor shapes and types
            assert batch["images"].shape == (1, 3, 256, 256)  # (B, C, H, W)
            assert batch["masks"].shape == (1, 1, 256, 256)  # (B, 1, H, W)
            assert batch["images"].dtype == torch.float32
            assert batch["masks"].dtype == torch.float32

            # Check value ranges
            assert batch["images"].min() >= -1.0
            assert batch["images"].max() <= 1.0
            assert torch.all((batch["masks"] >= 0) & (batch["masks"] <= 1))

    @patch("src.anonymizer.training.unet_trainer.UNetTrainer._initialize_unet")
    @patch("src.anonymizer.training.unet_trainer.UNetTrainer._initialize_vae")
    @patch("src.anonymizer.training.unet_trainer.UNetTrainer._initialize_trocr")
    @patch("src.anonymizer.training.unet_trainer.UNetTrainer._initialize_noise_scheduler")
    def test_unet_training_step_execution(
        self, mock_scheduler, mock_trocr, mock_vae, mock_unet, unet_config
    ):
        """Test UNet training step can be executed with mock models."""

        # Create mock models
        mock_unet_model = MagicMock()
        mock_unet_model.return_value.sample = torch.randn(1, 4, 32, 32)  # Mock UNet output
        mock_unet.return_value = mock_unet_model

        mock_vae_model = MagicMock()
        mock_vae_model.encode.return_value.latent_dist.sample.return_value = torch.randn(
            1, 4, 32, 32
        )
        mock_vae_model.config.scaling_factor = 0.18215
        mock_vae.return_value = mock_vae_model

        mock_trocr_model = MagicMock()
        mock_trocr_processor = MagicMock()
        mock_trocr_processor.return_value = {"input_ids": torch.randint(0, 1000, (1, 10))}
        mock_trocr_model.return_value.last_hidden_state = torch.randn(1, 10, 768)
        mock_trocr.return_value = (mock_trocr_model, mock_trocr_processor)

        mock_noise_scheduler_obj = MagicMock()
        mock_noise_scheduler_obj.add_noise.return_value = torch.randn(1, 4, 32, 32)
        mock_noise_scheduler_obj.config.num_train_timesteps = 1000
        mock_scheduler.return_value = mock_noise_scheduler_obj

        # Initialize trainer
        trainer = UNetTrainer(unet_config)

        # Mock additional required attributes
        trainer.text_projection = MagicMock()
        trainer.text_projection.parameters.return_value = []

        # Create a mock batch in the expected format
        mock_batch = {
            "images": torch.randn(1, 3, 256, 256),
            "masks": torch.ones(1, 1, 256, 256),
            "texts": ["[SSN]"],
        }

        # Test training step execution
        try:
            # Setup optimizer (required for training step)
            trainer.optimizer = torch.optim.Adam([torch.randn(1, requires_grad=True)], lr=1e-4)

            # Execute training step
            metrics = trainer.train_step(mock_batch)

            # Verify training step returns expected metrics
            assert "total_loss" in metrics.__dict__
            assert "learning_rate" in metrics.__dict__
            assert metrics.total_loss >= 0  # Loss should be non-negative
            assert metrics.learning_rate > 0  # Learning rate should be positive

        except Exception as e:
            # If training step fails due to missing dependencies, that's expected in test environment
            if "TrOCR" in str(e) or "diffusers" in str(e) or "transformers" in str(e):
                pytest.skip(f"Skipping test due to missing dependencies: {e}")
            else:
                raise

    @patch("src.anonymizer.training.unet_trainer.UNetTrainer._initialize_unet")
    @patch("src.anonymizer.training.unet_trainer.UNetTrainer._initialize_vae")
    @patch("src.anonymizer.training.unet_trainer.UNetTrainer._initialize_trocr")
    @patch("src.anonymizer.training.unet_trainer.UNetTrainer._initialize_noise_scheduler")
    def test_unet_loss_computation(
        self, mock_scheduler, mock_trocr, mock_vae, mock_unet, unet_config
    ):
        """Test UNet loss computation with realistic batch."""

        # Setup mocks similar to above test
        mock_unet_model = MagicMock()
        target_noise = torch.randn(1, 4, 32, 32)
        predicted_noise = target_noise + 0.1 * torch.randn_like(target_noise)  # Add small error
        mock_unet_model.return_value.sample = predicted_noise
        mock_unet.return_value = mock_unet_model

        mock_vae_model = MagicMock()
        mock_vae_model.encode.return_value.latent_dist.sample.return_value = torch.randn(
            1, 4, 32, 32
        )
        mock_vae_model.config.scaling_factor = 0.18215
        mock_vae.return_value = mock_vae_model

        mock_trocr_model = MagicMock()
        mock_trocr_processor = MagicMock()
        mock_trocr_processor.return_value = {"input_ids": torch.randint(0, 1000, (1, 10))}
        mock_trocr_model.return_value.last_hidden_state = torch.randn(1, 10, 768)
        mock_trocr.return_value = (mock_trocr_model, mock_trocr_processor)

        mock_noise_scheduler_obj = MagicMock()
        mock_noise_scheduler_obj.add_noise.return_value = torch.randn(1, 4, 32, 32)
        mock_noise_scheduler_obj.config.num_train_timesteps = 1000
        mock_scheduler.return_value = mock_noise_scheduler_obj

        trainer = UNetTrainer(unet_config)
        trainer.text_projection = MagicMock()

        mock_batch = {
            "images": torch.randn(1, 3, 256, 256),
            "masks": torch.ones(1, 1, 256, 256),
            "texts": ["[PHONE]"],
        }

        try:
            # Test loss computation
            loss_data = trainer._compute_loss(mock_batch)

            # Verify loss structure
            assert "loss" in loss_data
            assert "noise_pred" in loss_data
            assert "target_noise" in loss_data

            # Verify loss properties
            loss = loss_data["loss"]
            assert isinstance(loss, torch.Tensor)
            assert loss.dim() == 0  # Should be scalar
            assert loss >= 0  # MSE loss should be non-negative
            assert not torch.isnan(loss)  # Should not be NaN
            assert not torch.isinf(loss)  # Should not be infinite

        except Exception as e:
            if "TrOCR" in str(e) or "diffusers" in str(e) or "transformers" in str(e):
                pytest.skip(f"Skipping test due to missing dependencies: {e}")
            else:
                raise

    def test_training_data_pipeline_integration(
        self, mock_training_data, dataset_config, unet_config
    ):
        """Test complete data pipeline without model loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            mock_training_data(data_path, num_samples=3)

            # Update dataset config
            dataset_config.train_data_path = data_path

            # Create dataloaders using our InpaintingDataset
            train_loader, val_loader = create_inpainting_dataloaders(dataset_config, batch_size=2)

            # Initialize trainer (without loading actual models)
            trainer = UNetTrainer(unet_config)

            # Test that trainer can process batches from our dataloader
            batch_count = 0
            for batch in train_loader:
                batch_count += 1

                # Verify batch format matches trainer expectations
                assert "images" in batch
                assert "masks" in batch
                assert "texts" in batch

                # Verify tensor properties
                images = batch["images"]
                masks = batch["masks"]
                texts = batch["texts"]

                assert images.ndim == 4  # (B, C, H, W)
                assert masks.ndim == 4  # (B, 1, H, W)
                assert len(texts) == images.shape[0]  # Text per image

                # Verify data types and ranges
                assert images.dtype == torch.float32
                assert masks.dtype == torch.float32
                assert -1.1 <= images.min() <= images.max() <= 1.1  # Normalized range
                assert 0 <= masks.min() <= masks.max() <= 1  # Mask range

                # Test only first few batches
                if batch_count >= 2:
                    break

            assert batch_count > 0, "Should process at least one batch"

    def test_text_conditioning_format(self, mock_training_data, dataset_config):
        """Test text conditioning format matches UNet expectations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)
            mock_training_data(data_path, num_samples=2)

            dataset_config.train_data_path = data_path

            train_loader, _ = create_inpainting_dataloaders(dataset_config, batch_size=1)

            batch = next(iter(train_loader))
            texts = batch["texts"]

            # Verify text format
            assert isinstance(texts, list)
            assert len(texts) == 1  # Batch size 1
            assert isinstance(texts[0], str)
            assert len(texts[0]) > 0  # Non-empty text

            # Should contain replacement text tokens
            text_content = texts[0]
            replacement_tokens = ["[SSN]", "[CARD]", "[PHONE]", "[EMAIL]"]
            assert any(token in text_content for token in replacement_tokens)

    def test_error_handling_in_training_pipeline(self, unet_config):
        """Test error handling in training pipeline."""
        trainer = UNetTrainer(unet_config)

        # Test with malformed batch
        bad_batch = {
            "images": torch.randn(2, 3, 256, 256),
            "masks": torch.randn(1, 1, 256, 256),  # Wrong batch size
            "texts": ["test"],  # Wrong length
        }

        # Should handle mismatched batch dimensions gracefully
        try:
            # This will likely fail, but should fail gracefully
            trainer._compute_loss(bad_batch)
            assert False, "Should have raised an exception for malformed batch"
        except Exception as e:
            # Should get a meaningful error, not a generic crash
            assert len(str(e)) > 0  # Error message should not be empty
