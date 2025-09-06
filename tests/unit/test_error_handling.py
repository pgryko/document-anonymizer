"""Tests for enhanced training error handling."""

from unittest.mock import Mock

import pytest

from src.anonymizer.core.exceptions import (
    TrainingLoopError,
)
from src.anonymizer.training.error_handler import (
    ErrorCategory,
    ErrorSeverity,
    TrainingError,
    TrainingErrorHandler,
    TrainingErrorStats,
    with_error_handling,
)


class TestTrainingError:
    """Test TrainingError data structure."""

    def test_training_error_creation(self):
        """Test creating a training error."""
        exception = ValueError("Test error")
        error = TrainingError(
            exception=exception,
            category=ErrorCategory.MODEL_FORWARD,
            severity=ErrorSeverity.MEDIUM,
            step=100,
            epoch=5,
            context={"batch_idx": 10},
        )

        assert error.exception == exception
        assert error.category == ErrorCategory.MODEL_FORWARD
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.step == 100
        assert error.epoch == 5
        assert error.context["batch_idx"] == 10
        assert error.recovery_attempted is False
        assert error.recovery_successful is False
        assert isinstance(error.timestamp, float)

    def test_training_error_to_dict(self):
        """Test converting training error to dictionary."""
        exception = RuntimeError("Model forward failed")
        error = TrainingError(
            exception=exception,
            category=ErrorCategory.MODEL_FORWARD,
            severity=ErrorSeverity.HIGH,
            step=50,
            epoch=2,
            context={"operation": "forward_pass"},
        )

        error_dict = error.to_dict()

        assert error_dict["exception_type"] == "RuntimeError"
        assert error_dict["message"] == "Model forward failed"
        assert error_dict["category"] == "model_forward"
        assert error_dict["severity"] == "high"
        assert error_dict["step"] == 50
        assert error_dict["epoch"] == 2
        assert error_dict["context"]["operation"] == "forward_pass"
        assert "timestamp" in error_dict


class TestTrainingErrorStats:
    """Test TrainingErrorStats tracking."""

    def test_empty_stats(self):
        """Test initial empty stats."""
        stats = TrainingErrorStats()

        assert stats.total_errors == 0
        assert len(stats.errors_by_category) == 0
        assert len(stats.errors_by_severity) == 0
        assert stats.consecutive_failures == 0
        assert stats.last_error_step == -1
        assert stats.recovery_attempts == 0
        assert stats.successful_recoveries == 0

    def test_stats_update(self):
        """Test updating stats with errors."""
        stats = TrainingErrorStats()

        # First error
        error1 = TrainingError(
            exception=ValueError("Error 1"),
            category=ErrorCategory.DATA_LOADING,
            severity=ErrorSeverity.LOW,
            step=10,
            epoch=1,
        )
        stats.update(error1)

        assert stats.total_errors == 1
        assert stats.errors_by_category[ErrorCategory.DATA_LOADING] == 1
        assert stats.errors_by_severity[ErrorSeverity.LOW] == 1
        assert stats.consecutive_failures == 1
        assert stats.last_error_step == 10

        # Second consecutive error
        error2 = TrainingError(
            exception=RuntimeError("Error 2"),
            category=ErrorCategory.MODEL_FORWARD,
            severity=ErrorSeverity.MEDIUM,
            step=11,
            epoch=1,
        )
        stats.update(error2)

        assert stats.total_errors == 2
        assert stats.errors_by_category[ErrorCategory.MODEL_FORWARD] == 1
        assert stats.errors_by_severity[ErrorSeverity.MEDIUM] == 1
        assert stats.consecutive_failures == 2
        assert stats.last_error_step == 11

        # Non-consecutive error
        error3 = TrainingError(
            exception=ValueError("Error 3"),
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.LOW,
            step=20,  # Gap from previous
            epoch=1,
            recovery_attempted=True,
            recovery_successful=True,
        )
        stats.update(error3)

        assert stats.total_errors == 3
        assert stats.consecutive_failures == 1  # Reset due to gap
        assert stats.recovery_attempts == 1
        assert stats.successful_recoveries == 1

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        stats = TrainingErrorStats()

        # No errors
        assert stats.get_error_rate(100) == 0.0

        # Add some errors
        for i in range(5):
            error = TrainingError(
                exception=ValueError(f"Error {i}"),
                category=ErrorCategory.DATA_LOADING,
                severity=ErrorSeverity.LOW,
                step=i * 10,
                epoch=1,
            )
            stats.update(error)

        # 5 errors out of 100 steps = 5%
        assert stats.get_error_rate(100) == 5.0

    def test_recovery_success_rate(self):
        """Test recovery success rate calculation."""
        stats = TrainingErrorStats()

        # No recovery attempts
        assert stats.get_recovery_success_rate() == 0.0

        # Add errors with mixed recovery results
        errors = [
            TrainingError(
                ValueError("1"),
                ErrorCategory.DATA_LOADING,
                ErrorSeverity.LOW,
                1,
                1,
                recovery_attempted=True,
                recovery_successful=True,
            ),
            TrainingError(
                ValueError("2"),
                ErrorCategory.MODEL_FORWARD,
                ErrorSeverity.MEDIUM,
                2,
                1,
                recovery_attempted=True,
                recovery_successful=False,
            ),
            TrainingError(
                ValueError("3"),
                ErrorCategory.VALIDATION,
                ErrorSeverity.LOW,
                3,
                1,
                recovery_attempted=True,
                recovery_successful=True,
            ),
        ]

        for error in errors:
            stats.update(error)

        # 2 successful out of 3 attempts = 66.67%
        assert abs(stats.get_recovery_success_rate() - 66.66666666666667) < 0.01


class TestTrainingErrorHandler:
    """Test TrainingErrorHandler functionality."""

    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = TrainingErrorHandler(
            max_consecutive_failures=3,
            max_error_rate_percent=5.0,
            enable_auto_recovery=False,
            checkpoint_on_error=False,
        )

        assert handler.max_consecutive_failures == 3
        assert handler.max_error_rate_percent == 5.0
        assert handler.enable_auto_recovery is False
        assert handler.checkpoint_on_error is False
        assert len(handler.errors) == 0
        assert isinstance(handler.stats, TrainingErrorStats)

    def test_error_classification(self):
        """Test error classification by type and message."""
        handler = TrainingErrorHandler()

        # Test different error types
        test_cases = [
            (ValueError("Dataset index out of range"), ErrorCategory.DATA_LOADING),
            (RuntimeError("Model forward failed"), ErrorCategory.MODEL_FORWARD),
            (ValueError("Loss is nan"), ErrorCategory.LOSS_COMPUTATION),
            (RuntimeError("backward pass failed"), ErrorCategory.BACKWARD_PASS),
            (ValueError("Validation error occurred"), ErrorCategory.VALIDATION),
            (RuntimeError("Failed to save checkpoint"), ErrorCategory.CHECKPOINT),
            (RuntimeError("CUDA out of memory"), ErrorCategory.RESOURCE),
            (ValueError("Invalid configuration parameter"), ErrorCategory.CONFIGURATION),
        ]

        for exception, expected_category in test_cases:
            category = handler._classify_error(exception)
            assert category == expected_category

    def test_severity_classification(self):
        """Test error severity classification."""
        handler = TrainingErrorHandler()

        # Test critical errors
        critical_errors = [
            RuntimeError("CUDA out of memory"),
            RuntimeError("Device-side assert triggered"),
        ]

        for error in critical_errors:
            severity = handler._classify_severity(error, ErrorCategory.RESOURCE)
            assert severity == ErrorSeverity.CRITICAL

        # Test high severity
        high_error = ValueError("Invalid configuration")
        severity = handler._classify_severity(high_error, ErrorCategory.CONFIGURATION)
        assert severity == ErrorSeverity.HIGH

    def test_handle_error_basic(self):
        """Test basic error handling."""
        handler = TrainingErrorHandler(enable_auto_recovery=False)

        exception = ValueError("Test training error")
        error = handler.handle_error(
            exception=exception, step=50, epoch=2, context={"operation": "test"}
        )

        assert error.exception == exception
        assert error.step == 50
        assert error.epoch == 2
        assert error.context["operation"] == "test"
        assert error.recovery_attempted is False

        # Check stats were updated
        assert handler.stats.total_errors == 1
        assert len(handler.errors) == 1

    def test_handle_error_with_recovery(self):
        """Test error handling with recovery."""
        handler = TrainingErrorHandler(enable_auto_recovery=True)

        # Mock successful recovery - patch the recovery strategy directly
        mock_recover = Mock(return_value=None)
        handler.recovery_strategies[ErrorCategory.MODEL_FORWARD] = mock_recover

        exception = RuntimeError("Model forward failed")
        error = handler.handle_error(
            exception=exception,
            step=25,
            epoch=1,
        )

        assert error.recovery_attempted is True
        assert error.recovery_successful is True
        # The recovery method should have been called with the error
        mock_recover.assert_called_once_with(error)

    def test_handle_error_recovery_failure(self):
        """Test error handling when recovery fails."""
        handler = TrainingErrorHandler(enable_auto_recovery=True)

        # Mock recovery that raises exception
        mock_recover = Mock(side_effect=RuntimeError("Recovery failed"))
        handler.recovery_strategies[ErrorCategory.MODEL_FORWARD] = mock_recover

        exception = RuntimeError("Model forward failed")
        error = handler.handle_error(
            exception=exception,
            step=25,
            epoch=1,
        )

        assert error.recovery_attempted is True
        assert error.recovery_successful is False

    def test_training_health_check_consecutive_failures(self):
        """Test training health check for consecutive failures."""
        handler = TrainingErrorHandler(max_consecutive_failures=3)

        # Add consecutive failures
        for i in range(2):
            handler.handle_error(
                exception=ValueError(f"Error {i}"),
                step=i,
                epoch=1,
            )

        # Third consecutive failure should raise exception
        with pytest.raises(TrainingLoopError, match="Too many consecutive failures"):
            handler.handle_error(
                exception=ValueError("Error 2"),
                step=2,
                epoch=1,
            )

    def test_training_health_check_error_rate(self):
        """Test training health check for high error rate."""
        handler = TrainingErrorHandler(max_error_rate_percent=5.0)

        # Add many errors to trigger high error rate
        for i in range(10):
            handler.handle_error(
                exception=ValueError(f"Error {i}"),
                step=i * 10,  # Non-consecutive to avoid consecutive failure limit
                epoch=1,
            )

        # Next error at step 101 should trigger error rate check
        # 11 errors out of 102 steps > 5%
        with pytest.raises(TrainingLoopError, match="Error rate too high"):
            handler.handle_error(
                exception=ValueError("Final error"),
                step=101,
                epoch=1,
            )

    def test_decision_methods(self):
        """Test error handling decision methods."""
        handler = TrainingErrorHandler()

        # Test different severity levels
        low_error = TrainingError(
            ValueError("Low"), ErrorCategory.DATA_LOADING, ErrorSeverity.LOW, 1, 1
        )
        medium_error = TrainingError(
            ValueError("Medium"), ErrorCategory.MODEL_FORWARD, ErrorSeverity.MEDIUM, 2, 1
        )
        high_error = TrainingError(
            ValueError("High"), ErrorCategory.CHECKPOINT, ErrorSeverity.HIGH, 3, 1
        )
        critical_error = TrainingError(
            ValueError("Critical"), ErrorCategory.RESOURCE, ErrorSeverity.CRITICAL, 4, 1
        )

        # should_continue_training
        assert handler.should_continue_training(low_error) is True
        assert handler.should_continue_training(medium_error) is True
        assert handler.should_continue_training(high_error) is True
        assert handler.should_continue_training(critical_error) is False

        # should_continue_epoch
        assert handler.should_continue_epoch(low_error) is True
        assert handler.should_continue_epoch(medium_error) is True
        assert handler.should_continue_epoch(high_error) is False
        assert handler.should_continue_epoch(critical_error) is False

        # should_skip_batch
        assert handler.should_skip_batch(low_error) is False
        assert handler.should_skip_batch(medium_error) is True
        assert handler.should_skip_batch(high_error) is True
        assert handler.should_skip_batch(critical_error) is False  # Critical stops everything

    def test_error_summary(self):
        """Test error summary generation."""
        handler = TrainingErrorHandler()

        # Add various errors
        errors = [
            TrainingError(
                ValueError("1"),
                ErrorCategory.DATA_LOADING,
                ErrorSeverity.LOW,
                1,
                1,
                recovery_attempted=True,
                recovery_successful=True,
            ),
            TrainingError(
                ValueError("2"),
                ErrorCategory.MODEL_FORWARD,
                ErrorSeverity.MEDIUM,
                2,
                1,
                recovery_attempted=True,
                recovery_successful=False,
            ),
            TrainingError(ValueError("3"), ErrorCategory.VALIDATION, ErrorSeverity.LOW, 3, 1),
        ]

        for error in errors:
            handler.errors.append(error)
            handler.stats.update(error)

        summary = handler.get_error_summary()

        assert summary["total_errors"] == 3
        assert summary["errors_by_category"]["data_loading"] == 1
        assert summary["errors_by_category"]["model_forward"] == 1
        assert summary["errors_by_category"]["validation"] == 1
        assert summary["errors_by_severity"]["low"] == 2
        assert summary["errors_by_severity"]["medium"] == 1
        assert len(summary["recent_errors"]) == 3
        assert summary["recovery_success_rate"] == 50.0  # 1 out of 2

    def test_reset_error_state(self):
        """Test resetting error state."""
        handler = TrainingErrorHandler()

        # Add some errors to create state
        handler.handle_error(ValueError("Error 1"), step=1, epoch=1)
        handler.handle_error(ValueError("Error 2"), step=2, epoch=1)

        assert handler.stats.consecutive_failures == 2

        # Reset state
        handler.reset_error_state()

        assert handler.stats.consecutive_failures == 0


class TestErrorHandlingDecorator:
    """Test error handling decorator."""

    def test_with_error_handling_decorator(self):
        """Test the error handling decorator."""
        handler = TrainingErrorHandler(enable_auto_recovery=False)

        @with_error_handling(handler)
        def failing_method(self):
            self.global_step = 10
            self.current_epoch = 2
            raise ValueError("Method failed")

        # Create mock object with required attributes
        mock_trainer = Mock()
        mock_trainer.global_step = 0
        mock_trainer.current_epoch = 0

        # Call decorated method
        result = failing_method(mock_trainer)

        # Should return None instead of raising
        assert result is None

        # Error should be handled
        assert handler.stats.total_errors == 1
        assert handler.errors[0].step == 10
        assert handler.errors[0].epoch == 2

    def test_decorator_critical_error(self):
        """Test decorator with critical error that should re-raise."""
        handler = TrainingErrorHandler(enable_auto_recovery=False)

        @with_error_handling(handler)
        def critical_failing_method(self):
            self.global_step = 5
            self.current_epoch = 1
            raise RuntimeError("CUDA out of memory")

        mock_trainer = Mock()
        mock_trainer.global_step = 0
        mock_trainer.current_epoch = 0

        # Should re-raise critical error
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            critical_failing_method(mock_trainer)
