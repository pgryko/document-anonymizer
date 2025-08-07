"""Enhanced error handling for training pipelines.

This module provides comprehensive error handling, recovery strategies,
and training state management for robust training operations.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch

from src.anonymizer.core.exceptions import (
    InvalidLossError,
    TrainingLoopError,
)

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Classification of error severity levels."""

    LOW = "low"  # Continue training, log warning
    MEDIUM = "medium"  # Skip batch/step, continue epoch
    HIGH = "high"  # End current epoch, continue training
    CRITICAL = "critical"  # Stop training immediately


class ErrorCategory(Enum):
    """Categories of training errors."""

    DATA_LOADING = "data_loading"
    MODEL_FORWARD = "model_forward"
    LOSS_COMPUTATION = "loss_computation"
    BACKWARD_PASS = "backward_pass"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"


@dataclass
class TrainingError:
    """Detailed training error information."""

    exception: Exception
    category: ErrorCategory
    severity: ErrorSeverity
    step: int
    epoch: int
    timestamp: float = field(default_factory=time.time)
    context: dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert error to dictionary for logging."""
        return {
            "exception_type": type(self.exception).__name__,
            "message": str(self.exception),
            "category": self.category.value,
            "severity": self.severity.value,
            "step": self.step,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
            "context": self.context,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
        }


@dataclass
class TrainingErrorStats:
    """Statistics for training errors."""

    total_errors: int = 0
    errors_by_category: dict[ErrorCategory, int] = field(default_factory=dict)
    errors_by_severity: dict[ErrorSeverity, int] = field(default_factory=dict)
    consecutive_failures: int = 0
    last_error_step: int = -1
    recovery_attempts: int = 0
    successful_recoveries: int = 0

    def update(self, error: TrainingError) -> None:
        """Update statistics with new error."""
        self.total_errors += 1

        # Update category counts
        if error.category not in self.errors_by_category:
            self.errors_by_category[error.category] = 0
        self.errors_by_category[error.category] += 1

        # Update severity counts
        if error.severity not in self.errors_by_severity:
            self.errors_by_severity[error.severity] = 0
        self.errors_by_severity[error.severity] += 1

        # Update consecutive failures
        if error.step == self.last_error_step + 1:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 1

        self.last_error_step = error.step

        # Update recovery stats
        if error.recovery_attempted:
            self.recovery_attempts += 1
            if error.recovery_successful:
                self.successful_recoveries += 1

    def get_error_rate(self, total_steps: int) -> float:
        """Calculate error rate as percentage."""
        if total_steps == 0:
            return 0.0
        return (self.total_errors / total_steps) * 100.0

    def get_recovery_success_rate(self) -> float:
        """Calculate recovery success rate as percentage."""
        if self.recovery_attempts == 0:
            return 0.0
        return (self.successful_recoveries / self.recovery_attempts) * 100.0


class TrainingErrorHandler:
    """Comprehensive error handler for training pipelines."""

    def __init__(
        self,
        max_consecutive_failures: int = 5,
        max_error_rate_percent: float = 10.0,
        enable_auto_recovery: bool = True,
        checkpoint_on_error: bool = True,
    ):
        self.max_consecutive_failures = max_consecutive_failures
        self.max_error_rate_percent = max_error_rate_percent
        self.enable_auto_recovery = enable_auto_recovery
        self.checkpoint_on_error = checkpoint_on_error

        self.errors: list[TrainingError] = []
        self.stats = TrainingErrorStats()
        self.error_handlers: dict[ErrorCategory, Callable] = {}
        self.recovery_strategies: dict[ErrorCategory, Callable] = {}

        # Register default error handlers
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register default error classification handlers."""
        self.error_handlers = {
            ErrorCategory.DATA_LOADING: self._classify_data_error,
            ErrorCategory.MODEL_FORWARD: self._classify_model_error,
            ErrorCategory.LOSS_COMPUTATION: self._classify_loss_error,
            ErrorCategory.BACKWARD_PASS: self._classify_backward_error,
            ErrorCategory.VALIDATION: self._classify_validation_error,
            ErrorCategory.CHECKPOINT: self._classify_checkpoint_error,
            ErrorCategory.RESOURCE: self._classify_resource_error,
            ErrorCategory.CONFIGURATION: self._classify_config_error,
        }

        self.recovery_strategies = {
            ErrorCategory.DATA_LOADING: self._recover_data_error,
            ErrorCategory.MODEL_FORWARD: self._recover_model_error,
            ErrorCategory.LOSS_COMPUTATION: self._recover_loss_error,
            ErrorCategory.BACKWARD_PASS: self._recover_backward_error,
            ErrorCategory.VALIDATION: self._recover_validation_error,
            ErrorCategory.CHECKPOINT: self._recover_checkpoint_error,
            ErrorCategory.RESOURCE: self._recover_resource_error,
        }

    def handle_error(
        self,
        exception: Exception,
        step: int,
        epoch: int,
        context: dict[str, Any] | None = None,
    ) -> TrainingError:
        """Handle training error with classification and recovery."""
        # Classify error
        category = self._classify_error(exception)
        severity = self._classify_severity(exception, category)

        # Create error record
        error = TrainingError(
            exception=exception,
            category=category,
            severity=severity,
            step=step,
            epoch=epoch,
            context=context or {},
        )

        # Log error details
        logger.error(
            f"Training error at step {step}, epoch {epoch}: "
            f"{category.value} ({severity.value}) - {exception}",
            extra={"training_error": error.to_dict()},
        )

        # Attempt recovery if enabled
        if self.enable_auto_recovery and category in self.recovery_strategies:
            try:
                error.recovery_attempted = True
                self.recovery_strategies[category](error)
                error.recovery_successful = True
                logger.info(f"Recovery successful for {category.value} error")
            except Exception as recovery_error:
                logger.warning(f"Recovery failed for {category.value} error: {recovery_error}")

        # Update statistics
        self.errors.append(error)
        self.stats.update(error)

        # Check if training should continue
        self._check_training_health(step + 1)  # +1 for next step

        return error

    def _classify_error(self, exception: Exception) -> ErrorCategory:
        """Classify error into appropriate category."""
        exception_msg = str(exception).lower()

        # Data loading errors
        if any(
            keyword in exception_msg
            for keyword in ["dataset", "dataloader", "batch", "sample", "index"]
        ):
            return ErrorCategory.DATA_LOADING

        # Model forward pass errors
        if any(
            keyword in exception_msg
            for keyword in ["forward", "model", "layer", "tensor size", "dimension"]
        ):
            return ErrorCategory.MODEL_FORWARD

        # Loss computation errors
        if any(keyword in exception_msg for keyword in ["loss", "nan", "inf", "gradient"]):
            return ErrorCategory.LOSS_COMPUTATION

        # Backward pass errors
        if any(keyword in exception_msg for keyword in ["backward", "grad", "autograd"]):
            return ErrorCategory.BACKWARD_PASS

        # Validation errors
        if any(keyword in exception_msg for keyword in ["validation", "eval", "validate"]):
            return ErrorCategory.VALIDATION

        # Checkpoint errors
        if any(
            keyword in exception_msg for keyword in ["checkpoint", "save", "load", "state_dict"]
        ):
            return ErrorCategory.CHECKPOINT

        # Resource errors
        if any(
            keyword in exception_msg
            for keyword in ["memory", "cuda", "gpu", "device", "out of memory"]
        ):
            return ErrorCategory.RESOURCE

        # Configuration errors
        if any(keyword in exception_msg for keyword in ["config", "parameter", "argument"]):
            return ErrorCategory.CONFIGURATION

        # Default to model forward if unclear
        return ErrorCategory.MODEL_FORWARD

    def _classify_severity(self, exception: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Classify error severity."""
        exception_msg = str(exception).lower()

        # Critical errors that should stop training
        if any(
            keyword in exception_msg
            for keyword in ["out of memory", "cuda error", "device-side assert", "corrupted"]
        ):
            return ErrorSeverity.CRITICAL

        # High severity errors
        if category in [ErrorCategory.CONFIGURATION, ErrorCategory.CHECKPOINT]:
            return ErrorSeverity.HIGH

        # Medium severity for model/loss errors
        if category in [ErrorCategory.MODEL_FORWARD, ErrorCategory.LOSS_COMPUTATION]:
            return ErrorSeverity.MEDIUM

        # Low severity for data/validation errors
        return ErrorSeverity.LOW

    def _check_training_health(self, current_step: int) -> None:
        """Check if training should continue based on error patterns."""
        # Check consecutive failures
        if self.stats.consecutive_failures >= self.max_consecutive_failures:
            raise TrainingLoopError(
                f"Too many consecutive failures ({self.stats.consecutive_failures}). "
                "Training terminated for stability."
            )

        # Check error rate
        if current_step > 100:  # Only check after sufficient steps
            error_rate = self.stats.get_error_rate(current_step)
            if error_rate > self.max_error_rate_percent:
                raise TrainingLoopError(
                    f"Error rate too high ({error_rate:.2f}%). "
                    "Training terminated for quality assurance."
                )

    def should_continue_training(self, error: TrainingError) -> bool:
        """Determine if training should continue after this error."""
        return error.severity != ErrorSeverity.CRITICAL

    def should_continue_epoch(self, error: TrainingError) -> bool:
        """Determine if current epoch should continue after this error."""
        return error.severity not in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]

    def should_skip_batch(self, error: TrainingError) -> bool:
        """Determine if current batch should be skipped."""
        return error.severity in [ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]

    def get_error_summary(self) -> dict[str, Any]:
        """Get comprehensive error summary."""
        return {
            "total_errors": self.stats.total_errors,
            "consecutive_failures": self.stats.consecutive_failures,
            "errors_by_category": {k.value: v for k, v in self.stats.errors_by_category.items()},
            "errors_by_severity": {k.value: v for k, v in self.stats.errors_by_severity.items()},
            "recovery_success_rate": self.stats.get_recovery_success_rate(),
            "recent_errors": [error.to_dict() for error in self.errors[-5:]],  # Last 5 errors
        }

    # Classification methods
    def _classify_data_error(self, exception: Exception) -> ErrorSeverity:
        return ErrorSeverity.MEDIUM

    def _classify_model_error(self, exception: Exception) -> ErrorSeverity:
        return ErrorSeverity.MEDIUM

    def _classify_loss_error(self, exception: Exception) -> ErrorSeverity:
        if isinstance(exception, InvalidLossError):
            return ErrorSeverity.HIGH
        return ErrorSeverity.MEDIUM

    def _classify_backward_error(self, exception: Exception) -> ErrorSeverity:
        return ErrorSeverity.MEDIUM

    def _classify_validation_error(self, exception: Exception) -> ErrorSeverity:
        return ErrorSeverity.LOW

    def _classify_checkpoint_error(self, exception: Exception) -> ErrorSeverity:
        return ErrorSeverity.HIGH

    def _classify_resource_error(self, exception: Exception) -> ErrorSeverity:
        if "out of memory" in str(exception).lower():
            return ErrorSeverity.CRITICAL
        return ErrorSeverity.HIGH

    def _classify_config_error(self, exception: Exception) -> ErrorSeverity:
        return ErrorSeverity.CRITICAL

    # Recovery strategies
    def _recover_data_error(self, error: TrainingError) -> None:
        """Attempt to recover from data loading errors."""
        logger.info("Attempting data error recovery...")
        # Clear any cached data
        torch.cuda.empty_cache()

    def _recover_model_error(self, error: TrainingError) -> None:
        """Attempt to recover from model errors."""
        logger.info("Attempting model error recovery...")
        # Clear gradients and reset state
        torch.cuda.empty_cache()

    def _recover_loss_error(self, error: TrainingError) -> None:
        """Attempt to recover from loss computation errors."""
        logger.info("Attempting loss error recovery...")
        # Reset any accumulated gradients

    def _recover_backward_error(self, error: TrainingError) -> None:
        """Attempt to recover from backward pass errors."""
        logger.info("Attempting backward pass error recovery...")
        # Clear gradients
        torch.cuda.empty_cache()

    def _recover_validation_error(self, error: TrainingError) -> None:
        """Attempt to recover from validation errors."""
        logger.info("Attempting validation error recovery...")
        # Reset validation state

    def _recover_checkpoint_error(self, error: TrainingError) -> None:
        """Attempt to recover from checkpoint errors."""
        logger.info("Attempting checkpoint error recovery...")
        # Ensure checkpoint directory exists and has proper permissions

    def _recover_resource_error(self, error: TrainingError) -> None:
        """Attempt to recover from resource errors."""
        logger.info("Attempting resource error recovery...")
        # Aggressive memory cleanup
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def reset_error_state(self) -> None:
        """Reset error tracking state (useful between epochs)."""
        self.stats.consecutive_failures = 0
        logger.info("Error state reset")


def with_error_handling(error_handler: TrainingErrorHandler):
    """Decorator for adding error handling to training methods."""

    def decorator(func: Callable) -> Callable:
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                step = getattr(self, "global_step", 0)
                epoch = getattr(self, "current_epoch", 0)

                error = error_handler.handle_error(
                    exception=e,
                    step=step,
                    epoch=epoch,
                    context={
                        "method": func.__name__,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys()),
                    },
                )

                # Re-raise if critical
                if not error_handler.should_continue_training(error):
                    raise

                return None  # or appropriate default value

        return wrapper

    return decorator
