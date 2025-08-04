"""
Unit tests for metrics utilities - Imperative style.

Tests metrics collection and monitoring functionality.
"""

import time
from unittest.mock import Mock, patch

import torch

from src.anonymizer.utils.metrics import (
    MetricsCollector,
    calculate_similarity_metrics,
    timer,
)


class TestMetricsCollector:
    """Test MetricsCollector implementation."""

    def test_metrics_collector_initialization_enabled(self):
        """Test metrics collector initialization when enabled."""
        with patch.object(MetricsCollector, "_initialize_backend") as mock_init:
            collector = MetricsCollector(enabled=True)

            assert collector.enabled is True
            mock_init.assert_called_once()

    def test_metrics_collector_initialization_disabled(self):
        """Test metrics collector initialization when disabled."""
        with patch.object(MetricsCollector, "_initialize_backend") as mock_init:
            collector = MetricsCollector(enabled=False)

            assert collector.enabled is False
            mock_init.assert_not_called()

    def test_initialize_backend_with_datadog(self):
        """Test backend initialization with DataDog available."""
        collector = MetricsCollector(enabled=False)  # Don't auto-initialize

        with patch("builtins.__import__") as mock_import:
            mock_datadog = Mock()
            mock_import.return_value = mock_datadog

            collector._initialize_backend()

            assert collector._metrics_backend == "datadog"
            # Check that __import__ was called with 'datadog' as the first argument
            mock_import.assert_called()
            call_args = mock_import.call_args[0]
            assert call_args[0] == "datadog"

    def test_initialize_backend_without_datadog(self):
        """Test backend initialization without DataDog."""
        collector = MetricsCollector(enabled=False)  # Don't auto-initialize

        with patch("builtins.__import__", side_effect=ImportError):
            collector._initialize_backend()

            assert collector._metrics_backend == "logging"

    def test_record_training_metrics_enabled(self):
        """Test recording training metrics when enabled."""
        collector = MetricsCollector(enabled=True)
        collector._metrics_backend = "logging"

        metrics = {
            "total_loss": 0.5,
            "recon_loss": 0.3,
            "kl_loss": 0.1,
            "learning_rate": 1e-4,
        }
        step = 1000

        with patch.object(collector, "_record_logging_metrics") as mock_record:
            collector.record_training_metrics(metrics, step)

            mock_record.assert_called_once_with(metrics, step, "training")

    def test_record_training_metrics_disabled(self):
        """Test recording training metrics when disabled."""
        collector = MetricsCollector(enabled=False)

        metrics = {"total_loss": 0.5}
        step = 1000

        with patch.object(collector, "_record_logging_metrics") as mock_record:
            collector.record_training_metrics(metrics, step)

            mock_record.assert_not_called()

    def test_record_training_metrics_datadog(self):
        """Test recording training metrics with DataDog backend."""
        collector = MetricsCollector(enabled=True)
        collector._metrics_backend = "datadog"

        metrics = {"total_loss": 0.5}
        step = 1000

        with patch.object(collector, "_record_datadog_metrics") as mock_record:
            collector.record_training_metrics(metrics, step)

            mock_record.assert_called_once_with(metrics, step, "training")

    def test_record_training_metrics_error_handling(self):
        """Test error handling in training metrics recording."""
        collector = MetricsCollector(enabled=True)

        with patch.object(
            collector, "_record_logging_metrics", side_effect=Exception("Metrics error")
        ):
            # Should not raise exception
            collector.record_training_metrics({"loss": 0.5}, 100)

    def test_record_inference_metrics(self):
        """Test recording inference metrics."""
        collector = MetricsCollector(enabled=True)
        collector._metrics_backend = "logging"

        processing_time_ms = 150.5
        success = True

        with patch.object(collector, "_record_logging_metrics") as mock_record:
            collector.record_inference_metrics(processing_time_ms, success)

            mock_record.assert_called_once()
            call_args = mock_record.call_args[0]
            metrics = call_args[0]

            assert metrics["processing_time_ms"] == 150.5
            assert metrics["requests_total"] == 1.0
            assert metrics["success_rate"] == 1.0

    def test_record_inference_metrics_failure(self):
        """Test recording inference metrics for failed requests."""
        collector = MetricsCollector(enabled=True)
        collector._metrics_backend = "logging"

        with patch.object(collector, "_record_logging_metrics") as mock_record:
            collector.record_inference_metrics(250.0, False)

            call_args = mock_record.call_args[0]
            metrics = call_args[0]

            assert metrics["success_rate"] == 0.0

    def test_record_inference_metrics_datadog(self):
        """Test recording inference metrics with DataDog."""
        collector = MetricsCollector(enabled=True)
        collector._metrics_backend = "datadog"

        with patch.object(collector, "_record_datadog_metrics") as mock_record:
            collector.record_inference_metrics(150.5, True)

            mock_record.assert_called_once()
            call_args = mock_record.call_args[0]
            metrics = call_args[0]
            tags = call_args[1] if len(call_args) > 1 else mock_record.call_args[1]["tags"]

            assert "processing_time_ms" in metrics
            assert "success:True" in tags

    def test_record_model_performance(self):
        """Test recording model performance metrics."""
        collector = MetricsCollector(enabled=True)
        collector._metrics_backend = "logging"

        accuracy = 0.85
        confidence = 0.92

        with patch.object(collector, "_record_logging_metrics") as mock_record:
            collector.record_model_performance(accuracy, confidence)

            call_args = mock_record.call_args[0]
            metrics = call_args[0]

            assert metrics["model_accuracy"] == 0.85
            assert metrics["model_confidence"] == 0.92

    def test_record_datadog_metrics_with_step(self):
        """Test DataDog metrics recording with step."""
        collector = MetricsCollector(enabled=True)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 1000
        prefix = "training"
        tags = ["experiment:test"]

        # Mock the datadog import and statsd
        mock_statsd = Mock()
        mock_datadog_module = Mock()
        mock_datadog_module.statsd = mock_statsd

        with patch("builtins.__import__", return_value=mock_datadog_module):
            collector._record_datadog_metrics(metrics, step, prefix, tags)

            # Verify statsd.histogram was called for each metric
            assert mock_statsd.histogram.call_count == len(metrics)

            # Check first call
            first_call = mock_statsd.histogram.call_args_list[0]
            metric_name, _, call_kwargs = (
                first_call[0][0],
                first_call[0][1],
                first_call[1],
            )

            assert metric_name.startswith(prefix)
            assert "step:1000" in call_kwargs["tags"]
            assert "experiment:test" in call_kwargs["tags"]

    def test_record_datadog_metrics_without_step(self):
        """Test DataDog metrics recording without step."""
        collector = MetricsCollector(enabled=True)

        metrics = {"accuracy": 0.9}
        tags = ["model:test"]

        # Mock the datadog import and statsd
        mock_statsd = Mock()
        mock_datadog_module = Mock()
        mock_datadog_module.statsd = mock_statsd

        with patch("builtins.__import__", return_value=mock_datadog_module):
            collector._record_datadog_metrics(metrics, tags=tags)

            mock_statsd.histogram.assert_called_once()
            call_kwargs = mock_statsd.histogram.call_args[1]
            assert call_kwargs["tags"] == tags

    def test_record_datadog_metrics_error_handling(self):
        """Test DataDog metrics error handling."""
        collector = MetricsCollector(enabled=True)

        with patch("builtins.__import__", side_effect=Exception("DataDog error")):
            # Should not raise exception
            collector._record_datadog_metrics({"loss": 0.5})

    def test_record_logging_metrics_with_step(self):
        """Test logging metrics with step."""
        collector = MetricsCollector(enabled=True)

        metrics = {"loss": 0.5, "accuracy": 0.8}
        step = 500
        prefix = "validation"

        with patch("src.anonymizer.utils.metrics.logger") as mock_logger:
            collector._record_logging_metrics(metrics, step, prefix)

            # Should log each metric
            assert mock_logger.info.call_count == len(metrics)

            # Check log format
            log_calls = mock_logger.info.call_args_list
            assert f"validation.loss: 0.500000 [step {step}]" in log_calls[0][0][0]

    def test_record_logging_metrics_without_step(self):
        """Test logging metrics without step."""
        collector = MetricsCollector(enabled=True)

        metrics = {"accuracy": 0.9}
        prefix = "model"

        with patch("src.anonymizer.utils.metrics.logger") as mock_logger:
            collector._record_logging_metrics(metrics, prefix=prefix)

            mock_logger.info.assert_called_once()
            log_message = mock_logger.info.call_args[0][0]
            assert "model.accuracy: 0.900000" in log_message
            assert "[step" not in log_message  # No step info


class TestTimerContextManager:
    """Test timer context manager."""

    def test_timer_basic_usage(self):
        """Test basic timer usage."""
        with (
            patch("time.time", side_effect=[0.0, 0.1]),  # 100ms duration
            patch("src.anonymizer.utils.metrics.logger") as mock_logger,
        ):
            with timer():
                pass  # Simulate work

            # Should log the duration
            mock_logger.debug.assert_called_once()
            log_message = mock_logger.debug.call_args[0][0]
            assert "100.00ms" in log_message

    def test_timer_with_exception(self):
        """Test timer behavior when exception occurs."""
        with (
            patch("time.time", side_effect=[0.0, 0.05]),  # 50ms duration
            patch("src.anonymizer.utils.metrics.logger") as mock_logger,
        ):
            try:
                with timer():
                    raise ValueError(  # noqa: TRY301, TRY003  # Test exception handling
                        "Test error"
                    )
            except ValueError:
                pass

            # Should still log duration despite exception
            mock_logger.debug.assert_called_once()
            log_message = mock_logger.debug.call_args[0][0]
            assert "50.00ms" in log_message

    def test_timer_zero_duration(self):
        """Test timer with zero duration."""
        with (
            patch("time.time", side_effect=[1.0, 1.0]),  # Same time
            patch("src.anonymizer.utils.metrics.logger") as mock_logger,
        ):
            with timer():
                pass

            log_message = mock_logger.debug.call_args[0][0]
            assert "0.00ms" in log_message

    def test_timer_long_duration(self):
        """Test timer with long duration."""
        with (
            patch("time.time", side_effect=[0.0, 2.5]),  # 2.5 seconds
            patch("src.anonymizer.utils.metrics.logger") as mock_logger,
        ):
            with timer():
                pass

            log_message = mock_logger.debug.call_args[0][0]
            assert "2500.00ms" in log_message


class TestCalculateSimilarityMetrics:
    """Test similarity metrics calculation."""

    def test_calculate_similarity_metrics_tensors(self, device):
        """Test similarity metrics with PyTorch tensors."""
        # Create tensors with known differences
        pred = torch.ones(2, 3, 64, 64, device=device)
        target = torch.zeros(2, 3, 64, 64, device=device)

        metrics = calculate_similarity_metrics(pred, target)

        assert "mse" in metrics
        assert "psnr" in metrics
        assert metrics["mse"] == 1.0  # MSE between 1 and 0 is 1
        assert metrics["psnr"] == 0.0  # PSNR with MSE=1 is 0

    def test_calculate_similarity_metrics_identical_tensors(self, device):
        """Test similarity metrics with identical tensors."""
        tensor = torch.randn(2, 3, 32, 32, device=device)

        metrics = calculate_similarity_metrics(tensor, tensor)

        assert metrics["mse"] == 0.0
        # PSNR should be very high (but will be clamped due to numerical precision)
        assert metrics["psnr"] > 30  # Should be high for identical images

    def test_calculate_similarity_metrics_different_devices(self):
        """Test similarity metrics with tensors on different devices."""
        pred = torch.ones(1, 3, 32, 32)  # CPU
        if torch.cuda.is_available():
            target = torch.zeros(1, 3, 32, 32).cuda()  # GPU

            # Should still work (tensors will be moved)
            metrics = calculate_similarity_metrics(pred, target)
            assert "mse" in metrics
            assert "psnr" in metrics

    def test_calculate_similarity_metrics_different_shapes(self, device):
        """Test similarity metrics with different tensor shapes."""
        pred = torch.ones(1, 3, 32, 32, device=device)
        target = torch.zeros(1, 3, 16, 16, device=device)  # Different size

        # Should handle gracefully or raise appropriate error
        try:
            metrics = calculate_similarity_metrics(pred, target)
            # If it succeeds, should have reasonable values
            assert isinstance(metrics["mse"], float)
            assert isinstance(metrics["psnr"], float)
        except RuntimeError:
            # Expected for incompatible shapes
            pass

    def test_calculate_similarity_metrics_non_tensors(self):
        """Test similarity metrics with non-tensor inputs."""
        pred = [1, 2, 3]  # List
        target = [0, 1, 2]  # List

        metrics = calculate_similarity_metrics(pred, target)

        # Should return default values for non-tensors
        assert metrics["mse"] == 0.0
        assert metrics["psnr"] == 0.0

    def test_calculate_similarity_metrics_mixed_types(self, device):
        """Test similarity metrics with mixed input types."""
        pred = torch.ones(1, 3, 32, 32, device=device)
        target = [[1, 2], [3, 4]]  # List, not tensor

        metrics = calculate_similarity_metrics(pred, target)

        # Should return default values when types don't match
        assert metrics["mse"] == 0.0
        assert metrics["psnr"] == 0.0

    def test_calculate_similarity_metrics_error_handling(self, device):
        """Test error handling in similarity metrics calculation."""
        # Create tensors that might cause computation errors
        pred = torch.full((1, 3, 32, 32), float("inf"), device=device)
        target = torch.zeros(1, 3, 32, 32, device=device)

        # Should handle errors gracefully
        metrics = calculate_similarity_metrics(pred, target)

        # Should return some metrics even if computation fails
        assert "mse" in metrics
        assert "psnr" in metrics

    def test_calculate_similarity_metrics_torch_not_available(self):
        """Test behavior when torch operations fail."""
        with patch("torch.tensor", side_effect=Exception("Torch error")):
            metrics = calculate_similarity_metrics("not_tensor", "also_not_tensor")

            assert metrics["mse"] == 0.0
            assert metrics["psnr"] == 0.0

    def test_calculate_similarity_metrics_numerical_edge_cases(self, device):
        """Test numerical edge cases."""
        # Very small differences - use a larger difference that's still detectable
        pred = torch.ones(1, 1, 10, 10, device=device)
        target = pred + 1e-4  # Small but detectable difference

        metrics = calculate_similarity_metrics(pred, target)

        assert metrics["mse"] > 0  # Should detect small difference
        assert metrics["psnr"] > 30  # Should be high PSNR for small difference

        # Large differences
        pred = torch.full((1, 1, 10, 10), 1000.0, device=device)
        target = torch.full((1, 1, 10, 10), -1000.0, device=device)

        metrics = calculate_similarity_metrics(pred, target)

        assert metrics["mse"] > 1000  # Large MSE
        assert metrics["psnr"] < 0  # Negative PSNR for large differences


class TestMetricsIntegration:
    """Test metrics system integration."""

    def test_full_training_metrics_workflow(self):
        """Test complete training metrics workflow."""
        collector = MetricsCollector(enabled=True)
        collector._metrics_backend = "logging"

        # Simulate training loop
        training_metrics = {
            "total_loss": 0.8,
            "recon_loss": 0.5,
            "kl_loss": 0.2,
            "perceptual_loss": 0.1,
            "learning_rate": 1e-4,
        }

        with patch.object(collector, "_record_logging_metrics") as mock_record:
            # Record metrics for multiple steps
            for step in range(0, 500, 100):
                # Simulate decreasing loss
                adjusted_metrics = {k: v * (1 - step / 1000) for k, v in training_metrics.items()}
                collector.record_training_metrics(adjusted_metrics, step)

            # Should have recorded for each step
            assert mock_record.call_count == 5

    def test_inference_metrics_workflow(self):
        """Test inference metrics workflow."""
        collector = MetricsCollector(enabled=True)
        collector._metrics_backend = "logging"

        with patch.object(collector, "_record_logging_metrics") as mock_record:
            # Simulate multiple inference requests
            collector.record_inference_metrics(150.0, True)
            collector.record_inference_metrics(75.0, True)
            collector.record_inference_metrics(300.0, False)  # Failed request

            assert mock_record.call_count == 3

    def test_model_performance_tracking(self):
        """Test model performance tracking."""
        collector = MetricsCollector(enabled=True)
        collector._metrics_backend = "logging"

        with patch.object(collector, "_record_logging_metrics") as mock_record:
            # Track performance over time
            for epoch in range(5):
                accuracy = 0.5 + epoch * 0.1  # Improving accuracy
                confidence = 0.7 + epoch * 0.05  # Improving confidence
                collector.record_model_performance(accuracy, confidence)

            assert mock_record.call_count == 5

    def test_metrics_with_timer(self, device):
        """Test metrics collection with timing."""
        collector = MetricsCollector(enabled=True)
        collector._metrics_backend = "logging"

        with (
            patch.object(collector, "_record_logging_metrics") as mock_record,
            patch("time.time", side_effect=[0.0, 0.15]),  # 150ms
        ):
            with timer():
                # Simulate inference
                pred = torch.randn(1, 3, 64, 64, device=device)
                target = torch.randn(1, 3, 64, 64, device=device)
                calculate_similarity_metrics(pred, target)

            # Record the timing
            collector.record_inference_metrics(150.0, True)

            # Should have called _record_logging_metrics with inference metrics
            mock_record.assert_called_once()
            call_args = mock_record.call_args[0]
            metrics = call_args[0]
            assert metrics["processing_time_ms"] == 150.0

    def test_disabled_metrics_performance(self):
        """Test that disabled metrics don't impact performance."""
        collector = MetricsCollector(enabled=False)

        # Should be very fast when disabled
        start_time = time.time()

        for i in range(1000):
            collector.record_training_metrics({"loss": 0.5}, i)
            collector.record_inference_metrics(100.0, True)
            collector.record_model_performance(0.8, 0.9)

        elapsed = time.time() - start_time

        # Should complete very quickly (< 0.1 seconds)
        assert elapsed < 0.1
