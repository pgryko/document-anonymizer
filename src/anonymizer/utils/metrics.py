"""Metrics collection and monitoring utilities."""

import logging
import time
from contextlib import contextmanager
from typing import Any

# Optional dependencies
try:
    import datadog
    from datadog import statsd
except ImportError:
    datadog = None
    statsd = None

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and report training/inference metrics."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._metrics_backend = None

        if enabled:
            self._initialize_backend()

    def _initialize_backend(self):
        """Initialize metrics backend (e.g., DataDog, Prometheus)."""
        if datadog is not None:
            self._metrics_backend = "datadog"
            logger.info("Initialized DataDog metrics backend")
        else:
            logger.info("DataDog not available, using logging backend")
            self._metrics_backend = "logging"

    def record_training_metrics(self, metrics: dict[str, float], step: int):
        """Record training metrics."""
        if not self.enabled:
            return

        try:
            if self._metrics_backend == "datadog":
                self._record_datadog_metrics(metrics, step, "training")
            else:
                self._record_logging_metrics(metrics, step, "training")
        except Exception as e:
            logger.warning(f"Failed to record training metrics: {e}")

    def record_inference_metrics(self, processing_time_ms: float, success: bool):
        """Record inference performance metrics."""
        if not self.enabled:
            return

        try:
            metrics = {
                "processing_time_ms": processing_time_ms,
                "requests_total": 1.0,
                "success_rate": 1.0 if success else 0.0,
            }

            if self._metrics_backend == "datadog":
                self._record_datadog_metrics(metrics, tags=[f"success:{success}"])
            else:
                self._record_logging_metrics(metrics, prefix="inference")

        except Exception as e:
            logger.warning(f"Failed to record inference metrics: {e}")

    def record_model_performance(self, accuracy: float, confidence: float):
        """Record model quality metrics."""
        if not self.enabled:
            return

        try:
            metrics = {"model_accuracy": accuracy, "model_confidence": confidence}

            if self._metrics_backend == "datadog":
                self._record_datadog_metrics(metrics)
            else:
                self._record_logging_metrics(metrics, prefix="model")

        except Exception as e:
            logger.warning(f"Failed to record model metrics: {e}")

    def _record_datadog_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
        prefix: str = "",
        tags: list | None = None,
    ):
        """Record metrics to DataDog."""
        try:
            if statsd is None:
                logger.warning("DataDog statsd not available")
                return

            for metric_name, value in metrics.items():
                full_name = f"{prefix}.{metric_name}" if prefix else metric_name

                if step is not None:
                    metric_tags = [f"step:{step}"]
                    if tags:
                        metric_tags.extend(tags)
                    statsd.histogram(full_name, value, tags=metric_tags)
                else:
                    statsd.histogram(full_name, value, tags=tags or [])

        except Exception as e:
            logger.warning(f"DataDog metrics failed: {e}")

    def _record_logging_metrics(
        self, metrics: dict[str, float], step: int | None = None, prefix: str = ""
    ):
        """Record metrics via logging."""
        prefix_str = f"{prefix}." if prefix else ""
        step_str = f" [step {step}]" if step is not None else ""

        for metric_name, value in metrics.items():
            logger.info(f"METRIC {prefix_str}{metric_name}: {value:.6f}{step_str}")


@contextmanager
def timer():
    """Context manager for timing operations."""
    start_time = time.time()
    try:
        yield
    finally:
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        logger.debug(f"Operation took {duration_ms:.2f}ms")


def calculate_similarity_metrics(pred: Any, target: Any) -> dict[str, float]:
    """Calculate similarity metrics between prediction and target."""
    try:
        if torch is None or F is None:
            logger.warning("PyTorch not available for similarity metrics")
            return {"error": "PyTorch not available"}

        if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
            # MSE
            mse = F.mse_loss(pred, target).item()

            # PSNR
            psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse))).item()

            # SSIM (simplified version)
            # For full SSIM, would need additional implementation

            return {"mse": mse, "psnr": psnr}

    except Exception as e:
        logger.warning(f"Failed to calculate similarity metrics: {e}")
        return {"mse": 0.0, "psnr": 0.0}
