"""Learning rate schedulers for training."""

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_scheduler as get_hf_scheduler

from src.anonymizer.core.config import SchedulerConfig
from src.anonymizer.core.exceptions import ConfigurationError


class CosineAnnealingWithRestartsLR(_LRScheduler):
    """Cosine annealing with warm restarts scheduler."""

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
    ):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate."""
        return [
            self.eta_min
            + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1

        self.T_cur += 1

        if self.T_cur >= self.T_i:
            self.T_cur = 0
            self.T_i *= self.T_mult

        super().step(epoch)


def create_scheduler(
    scheduler_config: SchedulerConfig,
    optimizer: Optimizer,
    num_training_steps: int,
) -> Any:
    """Create learning rate scheduler."""
    scheduler_type = scheduler_config.type.lower()

    try:
        if scheduler_type == "cosine_with_restarts":
            # Calculate T_0 based on total steps and num_cycles
            T_0 = num_training_steps // scheduler_config.num_cycles

            scheduler = CosineAnnealingWithRestartsLR(
                optimizer=optimizer,
                T_0=T_0,
                T_mult=1,
                eta_min=optimizer.param_groups[0]["lr"] * scheduler_config.min_lr_ratio,
            )

            # Wrap with warmup if needed
            if scheduler_config.warmup_steps > 0:
                # Use HuggingFace scheduler for warmup support
                scheduler = get_hf_scheduler(
                    "cosine_with_restarts",
                    optimizer=optimizer,
                    num_warmup_steps=scheduler_config.warmup_steps,
                    num_training_steps=num_training_steps,
                )

        elif scheduler_type in ["linear", "cosine", "polynomial"]:
            # Use HuggingFace schedulers
            scheduler = get_hf_scheduler(
                scheduler_type,
                optimizer=optimizer,
                num_warmup_steps=scheduler_config.warmup_steps,
                num_training_steps=num_training_steps,
            )

        else:

            def _raise_unsupported_error() -> None:
                msg = f"Unsupported scheduler type: {scheduler_type}"
                raise ConfigurationError(msg)  # noqa: TRY301

            _raise_unsupported_error()

    except Exception as e:
        msg = f"Failed to create scheduler: {e}"
        raise ConfigurationError(msg) from e
    else:
        return scheduler
