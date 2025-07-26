"""
VAE Trainer with Critical Bug Fixes
==================================

This implementation fixes the most critical bug in the reference implementations:
Missing KL divergence loss in VAE training.

Key fixes:
- Adds proper KL divergence term to VAE loss
- Uses corrected hyperparameters (5e-4 learning rate instead of 5e-6)
- Implements perceptual loss for better text preservation
- Adds comprehensive error handling and validation
- Proper memory management and cleanup
"""

import logging
from pathlib import Path
from typing import Any

import safetensors.torch
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from transformers import get_scheduler

try:
    from torch.utils.tensorboard import SummaryWriter

    HAS_TENSORBOARD = True
except ImportError:
    SummaryWriter = None
    HAS_TENSORBOARD = False
try:
    import torchvision.utils as vutils

    HAS_TORCHVISION = True
except ImportError:
    vutils = None
    HAS_TORCHVISION = False

from src.anonymizer.core.config import VAEConfig
from src.anonymizer.core.exceptions import ModelLoadError, TrainingError, ValidationError
from src.anonymizer.core.models import ModelArtifacts, TrainingMetrics
from src.anonymizer.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class PerceptualLoss(torch.nn.Module):
    """Perceptual loss using VGG features for better text preservation."""

    def __init__(self, device: torch.device):
        super().__init__()
        try:
            from torchvision import models

            # Use VGG16 features for perceptual loss
            vgg = models.vgg16(pretrained=True).features[:16].to(device)
            vgg.eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg
        except ImportError:
            logger.warning("torchvision not available, skipping perceptual loss")
            self.vgg = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss."""
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)

        # Normalize to ImageNet stats
        pred = self._normalize(pred)
        target = self._normalize(target)

        # Extract VGG features
        pred_features = self.vgg(pred)
        target_features = self.vgg(target)

        # Compute MSE between features
        return F.mse_loss(pred_features, target_features)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor to ImageNet stats."""
        # Ensure 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] == 4:
            x = x[:, :3]  # Drop alpha channel

        # Normalize to [-1, 1] to [0, 1]
        x = (x + 1.0) * 0.5

        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)

        return (x - mean) / std


class VAETrainer:
    """VAE trainer with corrected loss function and hyperparameters."""

    def __init__(self, config: VAEConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator: Accelerator | None = None
        self.vae: AutoencoderKL | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: Any | None = None
        self.perceptual_loss: PerceptualLoss | None = None
        self.metrics_collector = MetricsCollector()

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_loss = float("inf")

        # TensorBoard writer (optional)
        self.tb_writer: Any | None = None

    def setup_distributed(self):
        """Setup distributed training with Accelerator."""
        try:
            self.accelerator = Accelerator(
                mixed_precision=self.config.mixed_precision,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                log_with=["tensorboard"],
                project_dir=str(self.config.checkpoint_dir),
            )
            logger.info(f"Initialized accelerator on device: {self.accelerator.device}")
        except Exception as e:
            raise TrainingError(f"Failed to setup distributed training: {e}")

    def _initialize_vae(self) -> AutoencoderKL:
        """Initialize VAE model."""
        try:
            logger.info(f"Loading VAE from {self.config.base_model}")
            vae = AutoencoderKL.from_pretrained(self.config.base_model, subfolder="vae")

            # Ensure model is in training mode
            vae.train()

            logger.info(f"VAE initialized: {sum(p.numel() for p in vae.parameters())} parameters")
            return vae

        except Exception as e:
            raise ModelLoadError(f"Failed to initialize VAE: {e}")

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with corrected learning rate."""
        if self.vae is None:
            raise TrainingError("VAE must be initialized before optimizer")

        # CRITICAL FIX: Use corrected learning rate (5e-4 instead of 5e-6)
        optimizer_config = self.config.optimizer

        if optimizer_config.type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.vae.parameters(),
                lr=optimizer_config.learning_rate,  # 5e-4 - Fixed from 5e-6
                weight_decay=optimizer_config.weight_decay,
                betas=optimizer_config.betas,
            )
        else:
            raise ValidationError(f"Unsupported optimizer: {optimizer_config.type}")

        logger.info(
            f"Optimizer setup: {optimizer_config.type} with LR {optimizer_config.learning_rate}"
        )
        return optimizer

    def _setup_scheduler(self, dataloader: DataLoader) -> Any:
        """Setup learning rate scheduler."""
        if self.optimizer is None:
            raise TrainingError("Optimizer must be initialized before scheduler")

        scheduler_config = self.config.scheduler
        num_training_steps = len(dataloader) * self.config.num_epochs

        scheduler = get_scheduler(
            scheduler_config.type,
            optimizer=self.optimizer,
            num_warmup_steps=scheduler_config.warmup_steps,
            num_training_steps=num_training_steps,
        )

        logger.info(
            f"Scheduler setup: {scheduler_config.type} with {num_training_steps} total steps"
        )
        return scheduler

    def _compute_loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        CRITICAL FIX: Compute VAE loss with proper KL divergence term.

        This was the most critical bug in the reference implementations:
        VAE training only used reconstruction loss, completely ignoring KL divergence.
        This led to poor latent space structure and unstable training.
        """
        images = batch["images"]

        # Encode to latent space
        try:
            posterior = self.vae.encode(images).latent_dist
            latents = posterior.sample()
        except Exception as e:
            raise TrainingError(f"VAE encoding failed: {e}")

        # Decode back to image space
        try:
            reconstructed = self.vae.decode(latents).sample
        except Exception as e:
            raise TrainingError(f"VAE decoding failed: {e}")

        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(reconstructed, images, reduction="mean")

        # CRITICAL FIX: KL divergence loss (was completely missing!)
        kl_loss = posterior.kl().mean()

        # Perceptual loss for better text preservation
        perceptual_loss = torch.tensor(0.0, device=images.device)
        if self.perceptual_loss is not None:
            try:
                perceptual_loss = self.perceptual_loss(reconstructed, images)
            except Exception as e:
                logger.warning(f"Perceptual loss computation failed: {e}")

        # Combined loss with proper weighting
        loss_config = self.config.loss
        total_loss = (
            recon_loss
            + loss_config.kl_weight * kl_loss
            + loss_config.perceptual_weight * perceptual_loss
        )

        # Validate loss values
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            raise TrainingError(
                f"Invalid loss detected: total={total_loss}, recon={recon_loss}, kl={kl_loss}"
            )

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "perceptual_loss": perceptual_loss,
        }

    def train_step(self, batch: dict[str, torch.Tensor]) -> TrainingMetrics:
        """Execute single training step."""
        try:
            # Compute losses
            losses = self._compute_loss(batch)
            total_loss = losses["total_loss"]

            # Backward pass with gradient accumulation
            if self.accelerator:
                self.accelerator.backward(total_loss)
            else:
                total_loss.backward()

            # Gradient clipping
            if self.config.gradient_clipping > 0:
                if self.accelerator:
                    self.accelerator.clip_grad_norm_(
                        self.vae.parameters(), self.config.gradient_clipping
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.vae.parameters(), self.config.gradient_clipping
                    )

            # Optimizer step
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            self.optimizer.zero_grad()

            # Update global step
            self.global_step += 1

            # Create metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            return TrainingMetrics(
                epoch=self.current_epoch,
                step=self.global_step,
                total_loss=losses["total_loss"].item(),
                recon_loss=losses["recon_loss"].item(),
                kl_loss=losses["kl_loss"].item(),
                perceptual_loss=(
                    losses["perceptual_loss"].item() if losses["perceptual_loss"] != 0 else None
                ),
                learning_rate=current_lr,
            )

        except Exception as e:
            raise TrainingError(f"Training step failed: {e}")

    def validate(self, val_dataloader: DataLoader) -> dict[str, float]:
        """Run validation."""
        if self.vae is None:
            raise TrainingError("VAE not initialized")

        self.vae.eval()
        total_losses = {
            "total_loss": 0.0,
            "recon_loss": 0.0,
            "kl_loss": 0.0,
            "perceptual_loss": 0.0,
        }
        num_batches = 0

        with torch.no_grad():
            for batch in val_dataloader:
                try:
                    losses = self._compute_loss(batch)
                    for key, value in losses.items():
                        total_losses[key] += value.item()
                    num_batches += 1
                except Exception as e:
                    logger.warning(f"Validation batch failed: {e}")
                    continue

        self.vae.train()

        # Average losses
        if num_batches > 0:
            avg_losses = {key: value / num_batches for key, value in total_losses.items()}
        else:
            avg_losses = total_losses

        return avg_losses

    def log_reconstructions(self, batch: dict[str, torch.Tensor], step: int, num_images: int = 4):
        """Log reconstruction visualizations to TensorBoard."""
        if not HAS_TORCHVISION or (not self.tb_writer and not self.accelerator):
            return

        try:
            self.vae.eval()
            with torch.no_grad():
                images = batch["images"][:num_images]

                # Encode and decode
                posterior = self.vae.encode(images).latent_dist
                latents = posterior.sample()
                reconstructed = self.vae.decode(latents).sample

                # Normalize for visualization (from [-1, 1] to [0, 1])
                images_vis = (images + 1) / 2
                reconstructed_vis = (reconstructed + 1) / 2

                # Create comparison grid
                comparison = torch.cat([images_vis, reconstructed_vis], dim=0)
                grid = vutils.make_grid(
                    comparison, nrow=num_images, normalize=True, scale_each=True
                )

                # Log to TensorBoard
                if self.tb_writer:
                    self.tb_writer.add_image(
                        "Reconstructions/Original_vs_Reconstructed", grid, step
                    )
                elif self.accelerator:
                    self.accelerator.log({"reconstructions": grid}, step=step)

            self.vae.train()

        except Exception as e:
            logger.warning(f"Failed to log reconstructions: {e}")

    def setup_tensorboard(self, log_dir: Path | None = None):
        """Setup TensorBoard logging."""
        if not HAS_TENSORBOARD:
            logger.warning("TensorBoard not available, skipping setup")
            return

        if log_dir is None:
            log_dir = self.config.checkpoint_dir / "tensorboard"

        log_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer = SummaryWriter(str(log_dir))
        logger.info(f"TensorBoard logging to {log_dir}")

    def save_checkpoint(self, save_path: Path | None = None) -> Path:
        """Save model checkpoint."""
        if save_path is None:
            save_path = self.config.checkpoint_dir / f"vae_step_{self.global_step}.safetensors"

        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Save model weights
            if self.accelerator:
                unwrapped_model = self.accelerator.unwrap_model(self.vae)
            else:
                unwrapped_model = self.vae

            state_dict = unwrapped_model.state_dict()
            safetensors.torch.save_file(state_dict, str(save_path))

            # Save training state
            training_state = {
                "global_step": self.global_step,
                "current_epoch": self.current_epoch,
                "best_loss": self.best_loss,
                "config": self.config.dict(),
            }

            state_path = save_path.with_suffix(".json")
            import json

            with open(state_path, "w") as f:
                json.dump(training_state, f, indent=2, default=str)

            logger.info(f"Checkpoint saved to {save_path}")
            return save_path

        except Exception as e:
            raise TrainingError(f"Failed to save checkpoint: {e}")

    def save_model(self) -> ModelArtifacts:
        """Save final model artifacts."""
        model_dir = self.config.checkpoint_dir / "final_model"
        model_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Save model
            model_path = model_dir / "model.safetensors"
            self.save_checkpoint(model_path)

            # Save config
            config_path = model_dir / "config.json"
            with open(config_path, "w") as f:
                import json

                json.dump(self.config.dict(), f, indent=2, default=str)

            # Create artifacts
            artifacts = ModelArtifacts(
                model_name=self.config.model_name,
                version=self.config.version,
                model_path=model_path,
                config_path=config_path,
                metadata={
                    "final_step": self.global_step,
                    "final_epoch": self.current_epoch,
                    "best_loss": self.best_loss,
                    "training_completed": True,
                },
            )

            logger.info(f"Model artifacts saved: {artifacts.model_name} v{artifacts.version}")
            return artifacts

        except Exception as e:
            raise TrainingError(f"Failed to save model artifacts: {e}")

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None = None,
        wandb_logger: Any | None = None,
    ):
        """
        Main training loop with all critical fixes applied.

        Key improvements:
        1. KL divergence loss properly included
        2. Corrected learning rate (5e-4 instead of 5e-6)
        3. Proper error handling and validation
        4. Memory management and cleanup
        5. Comprehensive metrics tracking
        """
        logger.info("Starting VAE training with corrected implementation")

        try:
            # Initialize components
            self.vae = self._initialize_vae()
            self.perceptual_loss = PerceptualLoss(self.device)
            self.optimizer = self._setup_optimizer()
            self.scheduler = self._setup_scheduler(train_dataloader)

            # Setup distributed training if available
            if self.accelerator:
                self.vae, self.optimizer, train_dataloader, self.scheduler = (
                    self.accelerator.prepare(
                        self.vae, self.optimizer, train_dataloader, self.scheduler
                    )
                )
                if val_dataloader:
                    val_dataloader = self.accelerator.prepare(val_dataloader)

            # Setup W&B model watching
            if wandb_logger and wandb_logger.enabled:
                wandb_logger.watch(self.vae, log="all", log_freq=100)

            # Training loop
            for epoch in range(self.config.num_epochs):
                self.current_epoch = epoch
                logger.info(f"Starting epoch {epoch + 1}/{self.config.num_epochs}")

                # Training phase
                self.vae.train()
                epoch_metrics = []

                for _batch_idx, batch in enumerate(train_dataloader):
                    try:
                        metrics = self.train_step(batch)
                        epoch_metrics.append(metrics)

                        # Log metrics
                        if self.global_step % 100 == 0:
                            self.metrics_collector.record_training_metrics(
                                metrics.to_dict(), self.global_step
                            )
                            logger.info(
                                f"Step {self.global_step}: Loss={metrics.total_loss:.4f}, "
                                f"Recon={metrics.recon_loss:.4f}, KL={metrics.kl_loss:.4f}"
                            )

                        # Save checkpoint
                        if self.global_step % self.config.save_every_n_steps == 0:
                            self.save_checkpoint()

                    except Exception as e:
                        logger.exception(f"Training step failed at step {self.global_step}: {e}")
                        continue

                # Validation phase
                if val_dataloader:
                    try:
                        val_losses = self.validate(val_dataloader)
                        logger.info(f"Validation - Total Loss: {val_losses['total_loss']:.4f}")

                        # Save best model
                        if val_losses["total_loss"] < self.best_loss:
                            self.best_loss = val_losses["total_loss"]
                            best_model_path = self.config.checkpoint_dir / "best_model.safetensors"
                            self.save_checkpoint(best_model_path)
                            logger.info(f"New best model saved with loss: {self.best_loss:.4f}")

                    except Exception as e:
                        logger.exception(f"Validation failed: {e}")

            logger.info("Training completed successfully")

        except Exception as e:
            logger.exception(f"Training failed: {e}")
            raise TrainingError(f"VAE training failed: {e}")

        finally:
            # Cleanup GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
