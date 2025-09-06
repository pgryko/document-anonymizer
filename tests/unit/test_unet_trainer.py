"""Unit tests for UNet trainer - Imperative style.

Tests the UNetTrainer class with comprehensive coverage of training functionality
including model initialization, text conditioning, and training steps.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
from PIL import Image

from src.anonymizer.core.config import OptimizerConfig, SchedulerConfig, UNetConfig
from src.anonymizer.core.exceptions import (
    DistributedTrainingSetupError,
    NoiseSchedulerInitializationError,
    OptimizerNotInitializedError,
    TextConditioningError,
    TextProjectionSetupError,
    TrOCRInitializationError,
    TrOCRNotInitializedError,
    UNetInitializationError,
    UNetNotInitializedError,
    UNetValidationNotInitializedError,
    UnsupportedOptimizerError,
    VAEInitializationError,
    VAESchedulerNotInitializedError,
)
from src.anonymizer.training.unet_trainer import (
    EXPECTED_UNET_INPUT_CHANNELS,
    TextRenderer,
    UNetTrainer,
)


class TestTextRenderer:
    """Test TextRenderer functionality."""

    def test_text_renderer_initialization(self):
        """Test text renderer initialization."""
        renderer = TextRenderer(font_size=24, image_size=(256, 256))

        assert renderer.font_size == 24
        assert renderer.image_size == (256, 256)
        assert renderer.font is not None

    def test_render_text_basic(self):
        """Test basic text rendering."""
        renderer = TextRenderer()

        result = renderer.render_text("Hello World")

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"
        assert result.size == renderer.image_size

    def test_render_text_empty(self):
        """Test rendering empty text."""
        renderer = TextRenderer()

        result = renderer.render_text("")

        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"


class TestUNetTrainer:
    """Test UNetTrainer functionality."""

    def create_mock_config(self, **overrides):
        """Create a mock UNet config."""
        defaults = {
            "model_name": "test-unet",
            "version": "1.0.0",
            "base_model": "stabilityai/stable-diffusion-2-inpainting",
            "checkpoint_dir": Path("/tmp/checkpoints"),
            "mixed_precision": "fp16",
            "gradient_accumulation_steps": 1,
            "num_epochs": 1,
            "num_train_timesteps": 1000,
            "gradient_clipping": 1.0,
            "save_every_n_steps": 100,
            "optimizer": OptimizerConfig(
                type="adamw", learning_rate=1e-4, weight_decay=0.01, betas=(0.9, 0.999)
            ),
            "scheduler": SchedulerConfig(type="cosine", warmup_steps=100),
        }
        defaults.update(overrides)
        return UNetConfig(**defaults)

    def test_unet_trainer_initialization(self):
        """Test UNet trainer initialization."""
        config = self.create_mock_config()

        trainer = UNetTrainer(config)

        assert trainer.config == config
        assert trainer.device.type in ["cpu", "cuda"]
        assert trainer.accelerator is None
        assert trainer.unet is None
        assert trainer.vae is None
        assert trainer.global_step == 0
        assert trainer.current_epoch == 0
        assert trainer.best_loss == float("inf")

    def test_setup_distributed_success(self):
        """Test successful distributed training setup."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        with patch("src.anonymizer.training.unet_trainer.Accelerator") as mock_accelerator:
            mock_acc_instance = Mock()
            mock_acc_instance.device = torch.device("cpu")
            mock_accelerator.return_value = mock_acc_instance

            trainer.setup_distributed()

            assert trainer.accelerator == mock_acc_instance
            mock_accelerator.assert_called_once()

    def test_setup_distributed_failure(self):
        """Test distributed training setup failure."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        with patch(
            "src.anonymizer.training.unet_trainer.Accelerator",
            side_effect=Exception("Setup failed"),
        ):
            with pytest.raises(DistributedTrainingSetupError):
                trainer.setup_distributed()

    def test_initialize_unet_success(self):
        """Test successful UNet initialization."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Create mock UNet with correct number of channels
        mock_unet = Mock()
        mock_unet.conv_in.in_channels = EXPECTED_UNET_INPUT_CHANNELS
        mock_unet.parameters.return_value = [torch.tensor([1.0])]  # Mock parameters

        with patch(
            "src.anonymizer.training.unet_trainer.UNet2DConditionModel.from_pretrained",
            return_value=mock_unet,
        ):
            result = trainer._initialize_unet()

            assert result == mock_unet

    def test_initialize_unet_wrong_channels(self):
        """Test UNet initialization with wrong number of channels."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Create mock UNet with wrong number of channels
        mock_unet = Mock()
        mock_unet.conv_in.in_channels = 4  # Wrong, should be 9

        with patch(
            "src.anonymizer.training.unet_trainer.UNet2DConditionModel.from_pretrained",
            return_value=mock_unet,
        ):
            with pytest.raises(UNetInitializationError):  # This wraps the UNetChannelMismatchError
                trainer._initialize_unet()

    def test_initialize_unet_failure(self):
        """Test UNet initialization failure."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        with patch(
            "src.anonymizer.training.unet_trainer.UNet2DConditionModel.from_pretrained",
            side_effect=Exception("Load failed"),
        ):
            with pytest.raises(UNetInitializationError):
                trainer._initialize_unet()

    def test_initialize_vae_success(self):
        """Test successful VAE initialization."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        mock_vae = Mock()
        mock_vae.parameters.return_value = [Mock(requires_grad=True)]

        with patch(
            "src.anonymizer.training.unet_trainer.AutoencoderKL.from_pretrained",
            return_value=mock_vae,
        ):
            result = trainer._initialize_vae()

            assert result == mock_vae
            mock_vae.eval.assert_called_once()
            # Check that parameters were frozen
            for param in mock_vae.parameters():
                assert param.requires_grad is False

    def test_initialize_vae_failure(self):
        """Test VAE initialization failure."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        with patch(
            "src.anonymizer.training.unet_trainer.AutoencoderKL.from_pretrained",
            side_effect=Exception("Load failed"),
        ):
            with pytest.raises(VAEInitializationError):
                trainer._initialize_vae()

    def test_initialize_trocr_success(self):
        """Test successful TrOCR initialization."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        mock_processor = Mock()
        mock_trocr = Mock()

        # Create mock parameters list instead of iterable issue
        mock_param1 = Mock()
        mock_param1.requires_grad = True
        mock_param2 = Mock()
        mock_param2.requires_grad = True
        mock_trocr.parameters.return_value = [mock_param1, mock_param2]

        # Mock the .to() method to return the mock itself
        mock_trocr.to.return_value = mock_trocr

        with (
            patch(
                "src.anonymizer.training.unet_trainer.TrOCRProcessor.from_pretrained",
                return_value=mock_processor,
            ),
            patch(
                "src.anonymizer.training.unet_trainer.VisionEncoderDecoderModel.from_pretrained",
                return_value=mock_trocr,
            ),
        ):

            trocr, processor = trainer._initialize_trocr()

            assert trocr == mock_trocr
            assert processor == mock_processor
            mock_trocr.eval.assert_called_once()
            mock_trocr.to.assert_called_once()  # Check that .to() was called
            # Check that parameters were frozen
            assert mock_param1.requires_grad is False
            assert mock_param2.requires_grad is False

    def test_initialize_trocr_failure(self):
        """Test TrOCR initialization failure."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        with patch(
            "src.anonymizer.training.unet_trainer.TrOCRProcessor.from_pretrained",
            side_effect=Exception("Load failed"),
        ):
            with pytest.raises(TrOCRInitializationError):
                trainer._initialize_trocr()

    def test_initialize_noise_scheduler_success(self):
        """Test successful noise scheduler initialization."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        mock_scheduler = Mock()

        with patch(
            "src.anonymizer.training.unet_trainer.DDPMScheduler.from_pretrained",
            return_value=mock_scheduler,
        ):
            result = trainer._initialize_noise_scheduler()

            assert result == mock_scheduler
            mock_scheduler.set_timesteps.assert_called_once_with(config.num_train_timesteps)

    def test_initialize_noise_scheduler_failure(self):
        """Test noise scheduler initialization failure."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        with patch(
            "src.anonymizer.training.unet_trainer.DDPMScheduler.from_pretrained",
            side_effect=Exception("Load failed"),
        ):
            with pytest.raises(NoiseSchedulerInitializationError):
                trainer._initialize_noise_scheduler()

    def test_setup_text_projection_no_models(self):
        """Test text projection setup without models."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        with pytest.raises(TextProjectionSetupError):
            trainer._setup_text_projection()

    def test_setup_text_projection_dimension_mismatch(self):
        """Test text projection setup with dimension mismatch."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Mock TrOCR and UNet with different dimensions
        mock_trocr = Mock()
        mock_trocr.config.encoder.hidden_size = 512
        mock_unet = Mock()
        mock_unet.config.cross_attention_dim = 768

        trainer.trocr = mock_trocr
        trainer.unet = mock_unet

        trainer._setup_text_projection()

        # Should create projection layer
        assert trainer.text_projection is not None
        assert isinstance(trainer.text_projection, torch.nn.Linear)

    def test_setup_text_projection_dimensions_match(self):
        """Test text projection setup with matching dimensions."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Mock TrOCR and UNet with same dimensions
        mock_trocr = Mock()
        mock_trocr.config.encoder.hidden_size = 768
        mock_unet = Mock()
        mock_unet.config.cross_attention_dim = 768

        trainer.trocr = mock_trocr
        trainer.unet = mock_unet

        trainer._setup_text_projection()

        # Should not create projection layer
        assert trainer.text_projection is None

    def test_setup_optimizer_no_unet(self):
        """Test optimizer setup without UNet."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        with pytest.raises(UNetNotInitializedError):
            trainer._setup_optimizer()

    def test_setup_optimizer_adamw(self):
        """Test AdamW optimizer setup."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Mock UNet with parameters
        mock_unet = Mock()
        mock_param = torch.tensor([1.0], requires_grad=True)
        mock_unet.parameters.return_value = [mock_param]
        trainer.unet = mock_unet

        optimizer = trainer._setup_optimizer()

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 1e-4
        assert optimizer.param_groups[0]["weight_decay"] == 0.01

    def test_setup_optimizer_unsupported(self):
        """Test setup with unsupported optimizer."""
        config = self.create_mock_config()
        config.optimizer.type = "sgd"  # Unsupported
        trainer = UNetTrainer(config)

        mock_unet = Mock()
        mock_unet.parameters.return_value = [torch.tensor([1.0])]
        trainer.unet = mock_unet

        with pytest.raises(UnsupportedOptimizerError):
            trainer._setup_optimizer()

    def test_setup_scheduler_no_optimizer(self):
        """Test scheduler setup without optimizer."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        mock_dataloader = [1, 2, 3]  # Mock with length

        with pytest.raises(OptimizerNotInitializedError):
            trainer._setup_scheduler(mock_dataloader)

    def test_prepare_text_conditioning_no_trocr(self):
        """Test text conditioning without TrOCR."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        with pytest.raises(TrOCRNotInitializedError):
            trainer._prepare_text_conditioning(["test"])

    def test_prepare_text_conditioning_success(self):
        """Test successful text conditioning."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Mock TrOCR components
        mock_trocr_processor = Mock()
        mock_trocr = Mock()

        # Mock encoder outputs
        mock_encoder_outputs = Mock()
        mock_features = torch.randn(1, 10, 768)  # batch, seq_len, hidden_dim
        mock_encoder_outputs.last_hidden_state = mock_features

        mock_encoder = Mock()
        mock_encoder.return_value = mock_encoder_outputs
        mock_trocr.get_encoder.return_value = mock_encoder

        # Mock processor inputs
        mock_inputs = Mock()
        mock_inputs.pixel_values = torch.randn(1, 3, 384, 384)
        mock_trocr_processor.return_value = mock_inputs

        trainer.trocr = mock_trocr
        trainer.trocr_processor = mock_trocr_processor
        trainer.device = torch.device("cpu")

        result = trainer._prepare_text_conditioning(["test text"])

        assert isinstance(result, torch.Tensor)
        assert result.shape == mock_features.shape

    def test_prepare_text_conditioning_with_projection(self):
        """Test text conditioning with projection layer."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Mock TrOCR components
        mock_trocr_processor = Mock()
        mock_trocr = Mock()

        # Mock encoder outputs
        mock_encoder_outputs = Mock()
        mock_features = torch.randn(1, 10, 512)  # Different dimension
        mock_encoder_outputs.last_hidden_state = mock_features

        mock_encoder = Mock()
        mock_encoder.return_value = mock_encoder_outputs
        mock_trocr.get_encoder.return_value = mock_encoder

        # Mock processor inputs
        mock_inputs = Mock()
        mock_inputs.pixel_values = torch.randn(1, 3, 384, 384)
        mock_trocr_processor.return_value = mock_inputs

        # Mock projection layer
        mock_projection = Mock()
        projected_features = torch.randn(1, 10, 768)
        mock_projection.return_value = projected_features

        trainer.trocr = mock_trocr
        trainer.trocr_processor = mock_trocr_processor
        trainer.text_projection = mock_projection
        trainer.device = torch.device("cpu")

        result = trainer._prepare_text_conditioning(["test text"])

        assert isinstance(result, torch.Tensor)
        mock_projection.assert_called_once_with(mock_features)

    def test_prepare_text_conditioning_error(self):
        """Test text conditioning error handling."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Mock TrOCR components to raise error
        mock_trocr_processor = Mock()
        mock_trocr_processor.side_effect = Exception("Processing failed")

        trainer.trocr = Mock()
        trainer.trocr_processor = mock_trocr_processor

        with pytest.raises(TextConditioningError):
            trainer._prepare_text_conditioning(["test"])

    def test_prepare_latents_no_models(self):
        """Test latent preparation without models."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        images = torch.randn(1, 3, 256, 256)
        masks = torch.randn(1, 1, 256, 256)

        with pytest.raises(VAESchedulerNotInitializedError):
            trainer._prepare_latents(images, masks)

    def test_validate_no_unet(self):
        """Test validation without UNet."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        mock_dataloader = []

        with pytest.raises(UNetValidationNotInitializedError):
            trainer.validate(mock_dataloader)

    def test_prepare_latents_success(self):
        """Test successful latent preparation."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Mock VAE and scheduler
        mock_vae = Mock()
        mock_vae.config.scaling_factor = 0.18215
        mock_latent_dist = Mock()
        mock_latent_dist.sample.return_value = torch.randn(2, 4, 64, 64)
        mock_vae.encode.return_value.latent_dist = mock_latent_dist

        mock_scheduler = Mock()
        mock_scheduler.config.num_train_timesteps = 1000
        mock_scheduler.add_noise.return_value = torch.randn(2, 4, 64, 64)

        trainer.vae = mock_vae
        trainer.noise_scheduler = mock_scheduler

        # Test inputs
        images = torch.randn(2, 3, 512, 512)
        masks = torch.randn(2, 1, 512, 512)

        with (
            patch("torch.randn_like", return_value=torch.randn(2, 4, 64, 64)),
            patch("torch.randint", return_value=torch.tensor([100, 200])),
            patch("torch.no_grad"),
        ):
            result = trainer._prepare_latents(images, masks)

            assert "unet_inputs" in result
            assert "noise" in result
            assert "timesteps" in result
            assert result["unet_inputs"].shape[1] == 9  # 4 + 1 + 4 channels

    def test_compute_loss_success(self):
        """Test successful loss computation."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Mock UNet
        mock_unet = Mock()
        mock_noise_pred = Mock()
        mock_noise_pred.sample = torch.randn(2, 4, 64, 64)
        mock_unet.return_value = mock_noise_pred
        trainer.unet = mock_unet

        # Mock helper methods
        with (
            patch.object(trainer, "_prepare_text_conditioning") as mock_text_cond,
            patch.object(trainer, "_prepare_latents") as mock_latents,
            patch("torch.nn.functional.mse_loss") as mock_mse_loss,
        ):
            mock_text_cond.return_value = torch.randn(2, 77, 768)
            mock_latents.return_value = {
                "unet_inputs": torch.randn(2, 9, 64, 64),
                "noise": torch.randn(2, 4, 64, 64),
                "timesteps": torch.tensor([100, 200]),
            }
            mock_mse_loss.return_value = torch.tensor(0.5)

            batch = {
                "images": torch.randn(2, 3, 512, 512),
                "masks": torch.randn(2, 1, 512, 512),
                "texts": ["text1", "text2"],
            }

            result = trainer._compute_loss(batch)

            assert "loss" in result
            assert "noise_pred" in result
            assert "target_noise" in result
            mock_unet.assert_called_once()
            mock_text_cond.assert_called_once_with(["text1", "text2"])
            mock_latents.assert_called_once()

    def test_train_step_success(self):
        """Test successful training step."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Setup trainer state
        trainer.global_step = 10
        trainer.current_epoch = 1
        trainer.optimizer = Mock()
        trainer.optimizer.param_groups = [{"lr": 1e-4}]
        trainer.scheduler = Mock()
        trainer.unet = Mock()
        trainer.unet.parameters.return_value = [torch.nn.Parameter(torch.randn(10))]

        # Mock loss computation
        mock_loss = torch.tensor(0.5, requires_grad=True)
        loss_data = {
            "loss": mock_loss,
            "noise_pred": torch.randn(2, 4, 64, 64),
            "target_noise": torch.randn(2, 4, 64, 64),
        }

        with patch.object(trainer, "_compute_loss", return_value=loss_data):
            batch = {
                "images": torch.randn(2, 3, 512, 512),
                "masks": torch.randn(2, 1, 512, 512),
                "texts": ["text1", "text2"],
            }

            metrics = trainer.train_step(batch)

            # Verify metrics
            assert metrics.epoch == 1
            assert metrics.step == 11  # Should increment
            assert metrics.total_loss == 0.5
            assert metrics.learning_rate == 1e-4

            # Verify optimizer step was called
            trainer.optimizer.step.assert_called_once()
            trainer.optimizer.zero_grad.assert_called_once()
            trainer.scheduler.step.assert_called_once()

    def test_validate_success(self):
        """Test successful validation."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Setup trainer
        trainer.unet = Mock()

        # Mock validation dataloader
        mock_dataloader = [
            {
                "images": torch.randn(2, 3, 512, 512),
                "masks": torch.randn(2, 1, 512, 512),
                "texts": ["text1", "text2"],
            }
            for _ in range(3)
        ]

        # Mock loss computation
        loss_data = {"loss": torch.tensor(0.3)}

        with (
            patch.object(trainer, "_compute_loss", return_value=loss_data),
            patch("torch.no_grad"),
        ):
            val_losses = trainer.validate(mock_dataloader)

            assert "loss" in val_losses
            assert val_losses["loss"] == 0.3

            # Verify UNet was set to eval and back to train
            trainer.unet.eval.assert_called_once()
            trainer.unet.train.assert_called_once()

    def test_save_checkpoint_success(self):
        """Test successful checkpoint saving."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Setup trainer state
        trainer.global_step = 100
        trainer.current_epoch = 2
        trainer.best_loss = 0.25

        # Mock UNet
        trainer.unet = Mock()
        mock_state_dict = {"conv_in.weight": torch.randn(9, 4, 3, 3)}
        trainer.unet.state_dict.return_value = mock_state_dict

        with (
            patch("safetensors.torch.save_file") as mock_save_file,
            patch("json.dump") as mock_json_dump,
            patch("builtins.open") as mock_open,
        ):
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file

            save_path = trainer.save_checkpoint()

            assert save_path.name == f"unet_step_{trainer.global_step}.safetensors"

            # Verify safetensors save was called
            mock_save_file.assert_called_once()

            # Verify training state was saved
            mock_json_dump.assert_called_once()
            call_args = mock_json_dump.call_args[0]
            training_state = call_args[0]
            assert training_state["global_step"] == 100
            assert training_state["current_epoch"] == 2
            assert training_state["best_loss"] == 0.25

    def test_save_model_success(self):
        """Test successful model saving."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Setup trainer state
        trainer.global_step = 1000
        trainer.current_epoch = 5
        trainer.best_loss = 0.15

        with (
            patch.object(trainer, "save_checkpoint") as mock_save_checkpoint,
            patch("json.dump") as mock_json_dump,
            patch("builtins.open") as mock_open,
        ):
            mock_file = Mock()
            mock_open.return_value.__enter__.return_value = mock_file
            mock_save_checkpoint.return_value = Path("/tmp/model.safetensors")

            artifacts = trainer.save_model()

            assert artifacts.model_name == config.model_name
            assert artifacts.version == config.version
            assert artifacts.metadata["final_step"] == 1000
            assert artifacts.metadata["final_epoch"] == 5
            assert artifacts.metadata["best_loss"] == 0.15
            assert artifacts.metadata["training_completed"] is True

            # Verify checkpoint was saved
            mock_save_checkpoint.assert_called_once()

            # Verify config was saved
            mock_json_dump.assert_called_once()

    def test_create_dataloaders_success(self):
        """Test successful dataloader creation."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        with (
            patch(
                "src.anonymizer.training.datasets.create_inpainting_dataloaders"
            ) as mock_create_loaders,
            patch("src.anonymizer.core.config.DatasetConfig") as mock_dataset_config_cls,
        ):
            mock_train_loader = Mock()
            mock_val_loader = Mock()
            mock_create_loaders.return_value = (mock_train_loader, mock_val_loader)

            mock_dataset_config = Mock()
            mock_dataset_config_cls.return_value = mock_dataset_config

            data_dir = Path("/tmp/data")
            train_loader, val_loader = trainer.create_dataloaders(data_dir, batch_size=4)

            assert train_loader == mock_train_loader
            assert val_loader == mock_val_loader

            # Verify dataset config was created
            mock_dataset_config_cls.assert_called_once()
            mock_create_loaders.assert_called_once_with(
                config=mock_dataset_config,
                batch_size=4,
            )

    def test_train_from_directory_success(self):
        """Test training from directory."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        with (
            patch.object(trainer, "create_dataloaders") as mock_create_dataloaders,
            patch.object(trainer, "train") as mock_train,
        ):
            mock_train_loader = Mock()
            mock_val_loader = Mock()
            mock_create_dataloaders.return_value = (mock_train_loader, mock_val_loader)

            data_dir = Path("/tmp/data")
            trainer.train_from_directory(data_dir, batch_size=8)

            mock_create_dataloaders.assert_called_once_with(data_dir, 8)
            mock_train.assert_called_once_with(mock_train_loader, mock_val_loader)

    def test_load_pretrained_vae_success(self):
        """Test loading pretrained VAE."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Mock VAE artifacts
        mock_artifacts = Mock()
        mock_artifacts.model_name = "test_vae"
        mock_artifacts.model_path = Mock()
        mock_artifacts.model_path.exists.return_value = True

        # Mock VAE initialization
        mock_vae = Mock()

        with (
            patch.object(trainer, "_initialize_vae", return_value=mock_vae) as mock_init_vae,
            patch("safetensors.torch.load_file") as mock_load_file,
        ):
            mock_state_dict = {"decoder.conv_out.weight": torch.randn(3, 512, 3, 3)}
            mock_load_file.return_value = mock_state_dict

            trainer.load_pretrained_vae(mock_artifacts)

            assert trainer.vae == mock_vae
            mock_init_vae.assert_called_once()
            mock_load_file.assert_called_once()
            mock_vae.load_state_dict.assert_called_once_with(mock_state_dict)

    def test_load_pretrained_vae_no_weights(self):
        """Test loading pretrained VAE without custom weights."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Mock VAE artifacts without existing weights
        mock_artifacts = Mock()
        mock_artifacts.model_name = "test_vae"
        mock_artifacts.model_path = Mock()
        mock_artifacts.model_path.exists.return_value = False

        # Mock VAE initialization
        mock_vae = Mock()

        with patch.object(trainer, "_initialize_vae", return_value=mock_vae) as mock_init_vae:
            trainer.load_pretrained_vae(mock_artifacts)

            assert trainer.vae == mock_vae
            mock_init_vae.assert_called_once()
            # Should not call load_state_dict when no custom weights exist
            mock_vae.load_state_dict.assert_not_called()

    def test_train_step_with_accelerator(self):
        """Test training step with accelerator."""
        config = self.create_mock_config()
        trainer = UNetTrainer(config)

        # Setup trainer with accelerator
        trainer.accelerator = Mock()
        trainer.global_step = 5
        trainer.current_epoch = 0
        trainer.optimizer = Mock()
        trainer.optimizer.param_groups = [{"lr": 1e-4}]
        trainer.scheduler = Mock()
        trainer.unet = Mock()
        trainer.unet.parameters.return_value = [torch.nn.Parameter(torch.randn(10))]

        # Mock loss computation
        mock_loss = torch.tensor(0.7, requires_grad=True)
        loss_data = {
            "loss": mock_loss,
            "noise_pred": torch.randn(1, 4, 64, 64),
            "target_noise": torch.randn(1, 4, 64, 64),
        }

        with patch.object(trainer, "_compute_loss", return_value=loss_data):
            batch = {
                "images": torch.randn(1, 3, 512, 512),
                "masks": torch.randn(1, 1, 512, 512),
                "texts": ["text1"],
            }

            metrics = trainer.train_step(batch)

            # Verify accelerator backward was called
            trainer.accelerator.backward.assert_called_once_with(mock_loss)

            # Verify gradient clipping through accelerator
            trainer.accelerator.clip_grad_norm_.assert_called_once()

            assert metrics.step == 6
            assert metrics.total_loss == 0.7

    def test_train_step_gradient_clipping_disabled(self):
        """Test training step with gradient clipping disabled."""
        config = self.create_mock_config(gradient_clipping=0.0)  # Disable clipping
        trainer = UNetTrainer(config)

        # Setup trainer state
        trainer.global_step = 0
        trainer.current_epoch = 0
        trainer.optimizer = Mock()
        trainer.optimizer.param_groups = [{"lr": 1e-4}]
        trainer.scheduler = Mock()
        trainer.unet = Mock()
        trainer.unet.parameters.return_value = [torch.nn.Parameter(torch.randn(10))]

        # Mock loss computation
        mock_loss = torch.tensor(0.3, requires_grad=True)
        loss_data = {
            "loss": mock_loss,
            "noise_pred": torch.randn(1, 4, 64, 64),
            "target_noise": torch.randn(1, 4, 64, 64),
        }

        with (
            patch.object(trainer, "_compute_loss", return_value=loss_data),
            patch("torch.nn.utils.clip_grad_norm_") as mock_clip_grad,
        ):
            batch = {
                "images": torch.randn(1, 3, 512, 512),
                "masks": torch.randn(1, 1, 512, 512),
                "texts": ["text1"],
            }

            metrics = trainer.train_step(batch)

            # Verify gradient clipping was not called
            mock_clip_grad.assert_not_called()

            assert metrics.step == 1
            assert metrics.total_loss == 0.3
