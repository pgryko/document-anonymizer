"""Weights & Biases integration for experiment tracking."""

import logging
import time
from typing import Any

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    wandb = None
    HAS_WANDB = False

from .modal_config import ModalTrainingConfig

logger = logging.getLogger(__name__)


class WandbLogger:
    """W&B logging utilities for training experiments."""

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        config: dict[str, Any] | None = None,
        enabled: bool = True,
    ):
        self.project = project
        self.entity = entity
        self.name = name or self._generate_run_name()
        self.tags = tags or []
        self.config = config or {}
        self.enabled = enabled and HAS_WANDB
        self.run = None

        if not HAS_WANDB and enabled:
            logger.warning("W&B not available. Install with: pip install wandb")
            self.enabled = False

    def _generate_run_name(self) -> str:
        """Generate a unique run name."""
        timestamp = int(time.time())
        return f"anonymizer-{timestamp}"

    def init(self, **kwargs) -> Any | None:
        """Initialize W&B run."""
        if not self.enabled:
            return None

        try:
            # Merge configs
            final_config = {**self.config, **kwargs.get("config", {})}

            self.run = wandb.init(
                project=self.project,
                entity=self.entity,
                name=self.name,
                tags=self.tags,
                config=final_config,
                **{k: v for k, v in kwargs.items() if k != "config"},
            )

            logger.info(f"W&B run initialized: {self.run.url}")
            return self.run

        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            self.enabled = False
            return None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        """Log metrics to W&B."""
        if not self.enabled or not self.run:
            return

        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            logger.error(f"Failed to log to W&B: {e}")

    def log_model(
        self,
        model_path: str,
        name: str | None = None,
        aliases: list[str] | None = None,
    ) -> None:
        """Log model artifact to W&B."""
        if not self.enabled or not self.run:
            return

        try:
            artifact = wandb.Artifact(
                name=name or "model",
                type="model",
            )
            artifact.add_dir(model_path)
            self.run.log_artifact(artifact, aliases=aliases)
            logger.info(f"Model logged to W&B: {model_path}")
        except Exception as e:
            logger.error(f"Failed to log model to W&B: {e}")

    def log_config(self, config: dict[str, Any]) -> None:
        """Update run configuration."""
        if not self.enabled or not self.run:
            return

        try:
            wandb.config.update(config)
        except Exception as e:
            logger.error(f"Failed to update W&B config: {e}")

    def finish(self) -> None:
        """Finish W&B run."""
        if not self.enabled or not self.run:
            return

        try:
            wandb.finish()
            logger.info("W&B run finished")
        except Exception as e:
            logger.error(f"Failed to finish W&B run: {e}")

    def watch(self, model, log: str = "all", log_freq: int = 100) -> None:
        """Watch model for gradient and parameter logging."""
        if not self.enabled or not self.run:
            return

        try:
            wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as e:
            logger.error(f"Failed to watch model: {e}")


def setup_wandb(
    training_config: ModalTrainingConfig,
    model_config: dict[str, Any],
    platform: str = "modal.com",
) -> WandbLogger:
    """Setup W&B logger for training."""

    # Generate run name
    model_type = training_config.model_type
    timestamp = int(time.time())
    run_name = f"{model_type}-modal-{timestamp}"

    # Prepare configuration
    config = {
        "platform": platform,
        "model_config": model_config,
        "training_config": training_config.get_wandb_config(),
    }

    # Create logger
    logger_instance = WandbLogger(
        project=training_config.wandb_project,
        entity=training_config.wandb_entity,
        name=run_name,
        tags=training_config.wandb_tags + [model_type, platform],
        config=config,
        enabled=training_config.wandb_entity is not None,
    )

    return logger_instance


def log_training_metrics(
    wandb_logger: WandbLogger,
    metrics: dict[str, float],
    step: int,
    epoch: int | None = None,
    phase: str = "train",
) -> None:
    """Log training metrics with proper formatting."""

    # Format metrics with phase prefix
    formatted_metrics = {}
    for key, value in metrics.items():
        if not key.startswith(f"{phase}/"):
            formatted_metrics[f"{phase}/{key}"] = value
        else:
            formatted_metrics[key] = value

    # Add step and epoch info
    formatted_metrics["step"] = step
    if epoch is not None:
        formatted_metrics["epoch"] = epoch

    wandb_logger.log(formatted_metrics, step=step)
