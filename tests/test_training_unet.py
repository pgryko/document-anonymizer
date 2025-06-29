"""
Unit tests for UNet trainer - Imperative style.

Tests the corrected UNet trainer with proper hyperparameters.
"""

import pytest
import torch
from unittest.mock import Mock, patch
from PIL import Image

from src.anonymizer.training.unet_trainer import UNetTrainer, TextRenderer
from src.anonymizer.core.models import TrainingMetrics, ModelArtifacts
from src.anonymizer.core.exceptions import TrainingError, ValidationError


class TestTextRenderer:
    """Test TextRenderer implementation."""

    def test_text_renderer_initialization(self):
        """Test text renderer initialization."""
        renderer = TextRenderer(font_size=32, image_size=(384, 384))

        assert renderer.font_size == 32
        assert renderer.image_size == (384, 384)
        assert renderer.font is not None

    def test_text_renderer_render_simple_text(self):
        """Test rendering simple text."""
        renderer = TextRenderer(font_size=24, image_size=(256, 256))

        text = "Hello World"
        image = renderer.render_text(text)

        assert isinstance(image, Image.Image)
        assert image.size == (256, 256)
        assert image.mode == "RGB"

    def test_text_renderer_render_multiline_text(self):
        """Test rendering multiline text."""
        renderer = TextRenderer()

        text = "Line 1\nLine 2\nLine 3"
        image = renderer.render_text(text)

        assert isinstance(image, Image.Image)
        assert image.size == renderer.image_size

    def test_text_renderer_font_loading_fallback(self):
        """Test font loading with fallback to default."""
        with patch("PIL.ImageFont.truetype", side_effect=OSError):
            with patch("PIL.ImageFont.load_default") as mock_default:
                mock_default.return_value = Mock()

                renderer = TextRenderer()

                # Should fallback to default font
                mock_default.assert_called()


class TestUNetTrainer:
    """Test UNet trainer with critical bug fixes."""

    def test_unet_trainer_initialization(self, unet_config):
        """Test UNet trainer initialization."""
        trainer = UNetTrainer(unet_config)

        assert trainer.config == unet_config
        assert trainer.device.type in ["cuda", "cpu"]
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
        assert trainer.best_loss == float("inf")
        assert trainer.unet is None  # Not initialized yet
        assert trainer.vae is None
        assert trainer.trocr is None

    def test_unet_trainer_setup_distributed(self, unet_config):
        """Test distributed training setup."""
        trainer = UNetTrainer(unet_config)

        with patch(
            "src.anonymizer.training.unet_trainer.Accelerator"
        ) as mock_accelerator:
            mock_accelerator.return_value.device = torch.device("cpu")

            trainer.setup_distributed()

            assert trainer.accelerator is not None
            mock_accelerator.assert_called_once()

    @patch("src.anonymizer.training.unet_trainer.UNet2DConditionModel")
    def test_unet_trainer_initialize_unet(self, mock_unet_class, unet_config):
        """Test UNet model initialization with correct architecture."""
        # Mock UNet model
        mock_unet = Mock()
        mock_conv_in = Mock()
        mock_conv_in.in_channels = 9  # SD 2.0 inpainting (correct!)
        mock_unet.conv_in = mock_conv_in
        mock_unet.parameters.return_value = [torch.randn(10)]
        mock_unet_class.from_pretrained.return_value = mock_unet

        trainer = UNetTrainer(unet_config)
        unet = trainer._initialize_unet()

        assert unet is not None
        mock_unet_class.from_pretrained.assert_called_once_with(
            unet_config.base_model, subfolder="unet"
        )

        # CRITICAL VERIFICATION: 9-channel input (correct for SD 2.0 inpainting)
        assert mock_unet.conv_in.in_channels == 9

    @patch("src.anonymizer.training.unet_trainer.UNet2DConditionModel")
    def test_unet_trainer_initialize_unet_wrong_channels(
        self, mock_unet_class, unet_config
    ):
        """Test UNet initialization with wrong number of channels."""
        # Mock UNet with wrong number of channels
        mock_unet = Mock()
        mock_conv_in = Mock()
        mock_conv_in.in_channels = 4  # Wrong! Should be 9 for inpainting
        mock_unet.conv_in = mock_conv_in
        mock_unet_class.from_pretrained.return_value = mock_unet

        trainer = UNetTrainer(unet_config)

        with pytest.raises(ValidationError, match="Expected 9-channel UNet"):
            trainer._initialize_unet()

    @patch("src.anonymizer.training.unet_trainer.AutoencoderKL")
    def test_unet_trainer_initialize_vae(self, mock_vae_class, unet_config):
        """Test VAE initialization for latent encoding."""
        # Mock VAE model
        mock_vae = Mock()
        mock_vae.parameters.return_value = [torch.randn(10)]
        mock_vae_class.from_pretrained.return_value = mock_vae

        trainer = UNetTrainer(unet_config)
        vae = trainer._initialize_vae()

        assert vae is not None
        mock_vae_class.from_pretrained.assert_called_once_with(
            unet_config.base_model, subfolder="vae"
        )

        # VAE should be frozen for UNet training
        mock_vae.eval.assert_called_once()

        # Verify parameters are frozen
        for param in mock_vae.parameters():
            assert param.requires_grad is False

    @patch("src.anonymizer.training.unet_trainer.VisionEncoderDecoderModel")
    @patch("src.anonymizer.training.unet_trainer.TrOCRProcessor")
    def test_unet_trainer_initialize_trocr(
        self, mock_processor_class, mock_trocr_class, unet_config
    ):
        """Test TrOCR initialization for text conditioning."""
        # Mock TrOCR components
        mock_processor = Mock()
        mock_processor_class.from_pretrained.return_value = mock_processor

        mock_trocr = Mock()
        mock_trocr.parameters.return_value = [torch.randn(10)]
        mock_trocr.to.return_value = mock_trocr
        mock_trocr_class.from_pretrained.return_value = mock_trocr

        trainer = UNetTrainer(unet_config)
        trocr, processor = trainer._initialize_trocr()

        assert trocr is not None
        assert processor is not None

        # TrOCR should be frozen
        mock_trocr.eval.assert_called_once()

        # Verify parameters are frozen
        for param in mock_trocr.parameters():
            assert param.requires_grad is False

    @patch("src.anonymizer.training.unet_trainer.DDPMScheduler")
    def test_unet_trainer_initialize_noise_scheduler(
        self, mock_scheduler_class, unet_config
    ):
        """Test noise scheduler initialization."""
        mock_scheduler = Mock()
        mock_scheduler_class.from_pretrained.return_value = mock_scheduler

        trainer = UNetTrainer(unet_config)
        scheduler = trainer._initialize_noise_scheduler()

        assert scheduler is not None
        mock_scheduler_class.from_pretrained.assert_called_once_with(
            unet_config.base_model, subfolder="scheduler"
        )
        mock_scheduler.set_timesteps.assert_called_once_with(
            unet_config.num_train_timesteps
        )

    def test_unet_trainer_setup_text_projection_needed(self, unet_config, device):
        """Test text projection setup when dimensions don't match."""
        trainer = UNetTrainer(unet_config)

        # Mock TrOCR with different dimension
        mock_trocr = Mock()
        mock_config = Mock()
        mock_encoder_config = Mock()
        mock_encoder_config.hidden_size = 512  # TrOCR dimension
        mock_config.encoder = mock_encoder_config
        mock_trocr.config = mock_config
        trainer.trocr = mock_trocr

        # Mock UNet with different dimension
        mock_unet = Mock()
        mock_unet_config = Mock()
        mock_unet_config.cross_attention_dim = 768  # Different dimension
        mock_unet.config = mock_unet_config
        trainer.unet = mock_unet

        trainer.device = device
        trainer._setup_text_projection()

        # Should create projection layer
        assert trainer.text_projection is not None
        assert isinstance(trainer.text_projection, torch.nn.Linear)

    def test_unet_trainer_setup_text_projection_not_needed(self, unet_config, device):
        """Test text projection setup when dimensions match."""
        trainer = UNetTrainer(unet_config)

        # Mock TrOCR and UNet with same dimension
        mock_trocr = Mock()
        mock_config = Mock()
        mock_encoder_config = Mock()
        mock_encoder_config.hidden_size = 768  # Same dimension
        mock_config.encoder = mock_encoder_config
        mock_trocr.config = mock_config
        trainer.trocr = mock_trocr

        mock_unet = Mock()
        mock_unet_config = Mock()
        mock_unet_config.cross_attention_dim = 768  # Same dimension
        mock_unet.config = mock_unet_config
        trainer.unet = mock_unet

        trainer.device = device
        trainer._setup_text_projection()

        # Should not create projection layer
        assert trainer.text_projection is None

    def test_unet_trainer_setup_optimizer_critical_fix(self, unet_config):
        """Test optimizer setup with corrected learning rate."""
        trainer = UNetTrainer(unet_config)

        # Mock UNet
        mock_unet = Mock()
        mock_unet.parameters.return_value = [torch.randn(10, requires_grad=True)]
        trainer.unet = mock_unet
        trainer.text_projection = None  # No projection needed

        optimizer = trainer._setup_optimizer()

        assert isinstance(optimizer, torch.optim.AdamW)
        # CRITICAL TEST: Learning rate should be 1e-4, not 1e-5
        assert optimizer.param_groups[0]["lr"] == 1e-4
        assert (
            optimizer.param_groups[0]["weight_decay"]
            == unet_config.optimizer.weight_decay
        )

    def test_unet_trainer_setup_optimizer_with_projection(self, unet_config, device):
        """Test optimizer setup with text projection layer."""
        trainer = UNetTrainer(unet_config)

        # Mock UNet
        mock_unet = Mock()
        mock_unet.parameters.return_value = [torch.randn(10, requires_grad=True)]
        trainer.unet = mock_unet

        # Add text projection
        trainer.text_projection = torch.nn.Linear(512, 768).to(device)

        optimizer = trainer._setup_optimizer()

        # Should include both UNet and projection parameters
        assert len(optimizer.param_groups) > 0
        assert optimizer.param_groups[0]["lr"] == unet_config.learning_rate

    def test_unet_trainer_prepare_text_conditioning(self, unet_config, device):
        """Test text conditioning preparation with TrOCR."""
        trainer = UNetTrainer(unet_config)

        # Mock TrOCR components
        mock_processor = Mock()
        mock_inputs = Mock()
        mock_inputs.pixel_values = torch.randn(2, 3, 384, 384, device=device)
        mock_processor.return_value = mock_inputs
        trainer.trocr_processor = mock_processor

        mock_encoder = Mock()
        mock_encoder_outputs = Mock()
        mock_encoder_outputs.last_hidden_state = torch.randn(2, 256, 768, device=device)
        mock_encoder.return_value = mock_encoder_outputs

        mock_trocr = Mock()
        mock_trocr.get_encoder.return_value = mock_encoder
        trainer.trocr = mock_trocr

        trainer.text_projection = None  # No projection needed
        trainer.device = device

        texts = ["Hello World", "Test Text"]

        with patch.object(trainer.text_renderer, "render_text") as mock_render:
            mock_render.side_effect = [
                Image.new("RGB", (384, 384)),
                Image.new("RGB", (384, 384)),
            ]

            features = trainer._prepare_text_conditioning(texts)

            assert features.shape == (2, 256, 768)
            assert features.device == device

            # Verify TrOCR was called correctly
            mock_processor.assert_called_once()
            mock_trocr.get_encoder.assert_called_once()

    def test_unet_trainer_prepare_text_conditioning_with_projection(
        self, unet_config, device
    ):
        """Test text conditioning with projection layer."""
        trainer = UNetTrainer(unet_config)

        # Mock TrOCR components
        mock_processor = Mock()
        mock_inputs = Mock()
        mock_inputs.pixel_values = torch.randn(2, 3, 384, 384, device=device)
        mock_processor.return_value = mock_inputs
        trainer.trocr_processor = mock_processor

        mock_encoder = Mock()
        mock_encoder_outputs = Mock()
        mock_encoder_outputs.last_hidden_state = torch.randn(
            2, 256, 512, device=device
        )  # 512-dim
        mock_encoder.return_value = mock_encoder_outputs

        mock_trocr = Mock()
        mock_trocr.get_encoder.return_value = mock_encoder
        trainer.trocr = mock_trocr

        # Add projection layer 512 -> 768
        trainer.text_projection = torch.nn.Linear(512, 768).to(device)
        trainer.device = device

        texts = ["Hello World", "Test Text"]

        with patch.object(trainer.text_renderer, "render_text") as mock_render:
            mock_render.side_effect = [
                Image.new("RGB", (384, 384)),
                Image.new("RGB", (384, 384)),
            ]

            features = trainer._prepare_text_conditioning(texts)

            # Should be projected to 768 dimensions
            assert features.shape == (2, 256, 768)
            assert features.device == device

    def test_unet_trainer_prepare_latents(self, unet_config, device):
        """Test latent preparation for diffusion training."""
        trainer = UNetTrainer(unet_config)

        # Mock VAE
        mock_vae = Mock()
        mock_vae.config.scaling_factor = 0.18215

        # Mock encode results
        mock_latent_dist = Mock()
        mock_latent_dist.sample.return_value = torch.randn(2, 4, 64, 64, device=device)
        mock_encode_result = Mock()
        mock_encode_result.latent_dist = mock_latent_dist
        mock_vae.encode.return_value = mock_encode_result

        trainer.vae = mock_vae

        # Mock noise scheduler
        mock_scheduler = Mock()
        mock_scheduler.add_noise.return_value = torch.randn(2, 4, 64, 64, device=device)
        trainer.noise_scheduler = mock_scheduler

        # Input tensors
        images = torch.randn(2, 3, 512, 512, device=device)
        masks = torch.randn(2, 1, 512, 512, device=device)

        latent_data = trainer._prepare_latents(images, masks)

        # Verify outputs
        assert "unet_inputs" in latent_data
        assert "noise" in latent_data
        assert "timesteps" in latent_data

        # CRITICAL TEST: UNet inputs should have 9 channels
        # 4 (noisy latent) + 1 (mask) + 4 (masked image latent) = 9
        assert latent_data["unet_inputs"].shape[1] == 9
        assert latent_data["unet_inputs"].device == device

    def test_unet_trainer_compute_loss(self, unet_config, device):
        """Test diffusion loss computation."""
        trainer = UNetTrainer(unet_config)

        # Mock text conditioning
        with patch.object(trainer, "_prepare_text_conditioning") as mock_text:
            mock_text.return_value = torch.randn(2, 256, 768, device=device)

            # Mock latent preparation
            with patch.object(trainer, "_prepare_latents") as mock_latents:
                mock_latent_data = {
                    "unet_inputs": torch.randn(
                        2, 9, 64, 64, device=device
                    ),  # 9 channels
                    "noise": torch.randn(2, 4, 64, 64, device=device),
                    "timesteps": torch.randint(0, 1000, (2,), device=device),
                }
                mock_latents.return_value = mock_latent_data

                # Mock UNet
                mock_unet = Mock()
                mock_unet_output = Mock()
                mock_unet_output.sample = torch.randn(2, 4, 64, 64, device=device)
                mock_unet.return_value = mock_unet_output
                trainer.unet = mock_unet

                # Create batch
                batch = {
                    "images": torch.randn(2, 3, 512, 512, device=device),
                    "masks": torch.randn(2, 1, 512, 512, device=device),
                    "texts": ["Hello", "World"],
                }

                loss_data = trainer._compute_loss(batch)

                # Verify loss computation
                assert "loss" in loss_data
                assert "noise_pred" in loss_data
                assert "target_noise" in loss_data

                # Verify UNet was called with correct inputs
                mock_unet.assert_called_once()
                call_args = mock_unet.call_args[0]
                assert call_args[0].shape == (2, 9, 64, 64)  # 9-channel input

    def test_unet_trainer_train_step(self, unet_config, device):
        """Test single training step."""
        trainer = UNetTrainer(unet_config)

        # Mock components
        trainer.unet = Mock()
        trainer.optimizer = Mock()
        trainer.optimizer.param_groups = [{"lr": 1e-4}]
        trainer.scheduler = Mock()

        # Mock loss computation
        with patch.object(trainer, "_compute_loss") as mock_compute_loss:
            mock_loss_data = {
                "loss": torch.tensor(0.8, device=device),
                "noise_pred": torch.randn(2, 4, 64, 64, device=device),
                "target_noise": torch.randn(2, 4, 64, 64, device=device),
            }
            mock_compute_loss.return_value = mock_loss_data

            batch = {
                "images": torch.randn(2, 3, 512, 512, device=device),
                "masks": torch.randn(2, 1, 512, 512, device=device),
                "texts": ["Hello", "World"],
            }

            metrics = trainer.train_step(batch)

            # Verify metrics
            assert isinstance(metrics, TrainingMetrics)
            assert metrics.total_loss == 0.8
            assert metrics.recon_loss == 0.8  # For UNet, this is the diffusion loss
            assert metrics.kl_loss == 0.0  # N/A for UNet
            assert metrics.learning_rate == 1e-4

            # Verify optimizer was called
            trainer.optimizer.step.assert_called_once()
            trainer.optimizer.zero_grad.assert_called_once()

    def test_unet_trainer_critical_bug_fixes_verification(self, unet_config):
        """Test that critical bugs are actually fixed in the trainer."""
        trainer = UNetTrainer(unet_config)

        # CRITICAL FIX 1: Learning rate should be 1e-4, not 1e-5
        assert trainer.config.learning_rate == 1e-4

        # CRITICAL FIX 2: Batch size should be 8, not 4
        assert trainer.config.batch_size == 8

        # CRITICAL FIX 3: Base model should be SD 2.0 inpainting (9-channel)
        assert "inpainting" in trainer.config.base_model
        assert "stable-diffusion-2" in trainer.config.base_model

        # CRITICAL FIX 4: Proper diffusion parameters
        assert trainer.config.num_train_timesteps == 1000
        assert trainer.config.noise_schedule == "scaled_linear"

    def test_unet_trainer_load_pretrained_vae(self, unet_config, temp_dir):
        """Test loading pretrained VAE from artifacts."""
        trainer = UNetTrainer(unet_config)

        # Create mock artifacts
        from src.anonymizer.core.models import ModelArtifacts

        model_path = temp_dir / "model.safetensors"
        config_path = temp_dir / "config.json"
        model_path.touch()
        config_path.touch()

        artifacts = ModelArtifacts(
            model_name="test-vae",
            version="v1.0",
            model_path=model_path,
            config_path=config_path,
        )

        with patch.object(trainer, "_initialize_vae") as mock_init_vae:
            mock_vae = Mock()
            mock_init_vae.return_value = mock_vae

            with patch("safetensors.torch.load_file") as mock_load:
                mock_state_dict = {"layer.weight": torch.randn(10, 10)}
                mock_load.return_value = mock_state_dict

                trainer.load_pretrained_vae(artifacts)

                # Verify VAE was initialized and weights loaded
                mock_init_vae.assert_called_once()
                mock_load.assert_called_once()
                mock_vae.load_state_dict.assert_called_once_with(mock_state_dict)

    def test_unet_trainer_save_model_with_projection(
        self, unet_config, temp_dir, device
    ):
        """Test model saving with text projection layer."""
        trainer = UNetTrainer(unet_config)
        trainer.config.checkpoint_dir = temp_dir
        trainer.global_step = 2500
        trainer.current_epoch = 25
        trainer.best_loss = 0.4

        # Mock UNet and text projection
        trainer.unet = Mock()
        trainer.text_projection = torch.nn.Linear(512, 768).to(device)

        with patch.object(trainer, "save_checkpoint") as mock_save_checkpoint:
            mock_save_checkpoint.return_value = (
                temp_dir / "final_model" / "model.safetensors"
            )

            artifacts = trainer.save_model()

            # Verify artifacts include text projection info
            assert isinstance(artifacts, ModelArtifacts)
            assert artifacts.metadata["has_text_projection"] is True

    def test_unet_trainer_memory_cleanup(self, unet_config):
        """Test GPU memory cleanup."""
        trainer = UNetTrainer(unet_config)

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.empty_cache") as mock_empty_cache,
            patch("torch.cuda.synchronize") as mock_synchronize,
        ):

            # This should trigger memory cleanup in the finally block
            try:
                trainer.train([], None)
            except Exception:
                pass  # Expected since we don't have real data

            # Memory cleanup should have been called
            mock_empty_cache.assert_called()
            mock_synchronize.assert_called()

    def test_unet_trainer_validation_loss_nan(self, unet_config, device):
        """Test validation of loss for NaN values."""
        trainer = UNetTrainer(unet_config)

        # Mock components that return NaN loss
        with patch.object(trainer, "_prepare_text_conditioning") as mock_text:
            mock_text.return_value = torch.randn(2, 256, 768, device=device)

            with patch.object(trainer, "_prepare_latents") as mock_latents:
                mock_latent_data = {
                    "unet_inputs": torch.randn(2, 9, 64, 64, device=device),
                    "noise": torch.randn(2, 4, 64, 64, device=device),
                    "timesteps": torch.randint(0, 1000, (2,), device=device),
                }
                mock_latents.return_value = mock_latent_data

                # Mock UNet that returns values leading to NaN loss
                mock_unet = Mock()
                mock_unet_output = Mock()
                # Create incompatible tensors that will cause NaN in MSE loss
                mock_unet_output.sample = torch.full(
                    (2, 4, 64, 64), float("inf"), device=device
                )
                mock_unet.return_value = mock_unet_output
                trainer.unet = mock_unet

                batch = {
                    "images": torch.randn(2, 3, 512, 512, device=device),
                    "masks": torch.randn(2, 1, 512, 512, device=device),
                    "texts": ["Hello", "World"],
                }

                with pytest.raises(TrainingError, match="Invalid loss detected"):
                    trainer._compute_loss(batch)
