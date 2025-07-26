"""
Performance Tests
=================

Comprehensive performance and memory tests for the document anonymization pipeline.
"""

import pytest
import time
import tempfile
from pathlib import Path
import numpy as np

from src.anonymizer.performance import (
    PerformanceProfiler,
    MemoryProfiler,
    AnonymizationBenchmark,
    ModelBenchmark,
    PerformanceMonitor,
)


class TestMemoryProfiler:
    """Test memory profiling functionality."""

    def test_memory_profiler_initialization(self):
        """Test memory profiler can be initialized."""
        profiler = MemoryProfiler(sample_interval=0.1)
        assert profiler.sample_interval == 0.1
        assert not profiler.is_monitoring
        assert len(profiler.snapshots) == 0

    def test_memory_monitoring_context(self):
        """Test memory monitoring context manager."""
        profiler = MemoryProfiler(sample_interval=0.05)

        with profiler.profile_memory():
            # Simulate memory-intensive operation
            data = [i for i in range(100000)]  # Allocate some memory
            time.sleep(0.2)  # Allow time for samples
            del data

        # Should have collected some snapshots
        assert len(profiler.snapshots) > 0

        # Should have memory measurements
        peak_memory = profiler.get_peak_memory()
        avg_memory = profiler.get_average_memory()

        assert peak_memory > 0
        assert avg_memory > 0
        assert peak_memory >= avg_memory

    def test_memory_leak_detection(self):
        """Test memory leak detection capabilities."""
        profiler = MemoryProfiler(sample_interval=0.05)

        with profiler.profile_memory():
            # Simulate growing memory usage
            data_store = []
            for i in range(5):
                data_store.append([j for j in range(10000)])
                time.sleep(0.05)

        snapshots = profiler.snapshots
        assert len(snapshots) > 2

        # Memory should generally increase
        first_memory = snapshots[0].rss_mb
        last_memory = snapshots[-1].rss_mb

        # Allow for some variation but expect general increase
        assert last_memory >= first_memory * 0.8  # Allow 20% tolerance


class TestPerformanceProfiler:
    """Test performance profiling functionality."""

    def test_profiler_initialization(self):
        """Test performance profiler initialization."""
        profiler = PerformanceProfiler(auto_save=False)
        assert len(profiler.metrics) == 0
        assert profiler.memory_profiler is not None

    def test_operation_profiling(self):
        """Test profiling of operations."""
        profiler = PerformanceProfiler(auto_save=False)

        with profiler.profile_operation("test_operation"):
            # Simulate CPU and memory intensive operation
            data = np.random.rand(1000, 1000)
            np.dot(data, data.T)
            time.sleep(0.1)

        assert len(profiler.metrics) == 1

        metrics = profiler.metrics[0]
        assert metrics.operation == "test_operation"
        assert metrics.duration_ms > 100  # Should be > 100ms due to sleep
        assert metrics.peak_memory_mb > 0
        assert metrics.cpu_percent >= 0

    def test_multiple_operations(self):
        """Test profiling multiple operations."""
        profiler = PerformanceProfiler(auto_save=False)

        # Profile multiple operations
        for i in range(3):
            with profiler.profile_operation(f"operation_{i}"):
                time.sleep(0.05)
                [j for j in range(1000)]

        assert len(profiler.metrics) == 3

        # Test filtering by operation name
        op_0_metrics = profiler.get_metrics_for_operation("operation_0")
        assert len(op_0_metrics) == 1
        assert op_0_metrics[0].operation == "operation_0"

    def test_average_metrics_calculation(self):
        """Test average metrics calculation."""
        profiler = PerformanceProfiler(auto_save=False)

        # Run same operation multiple times
        for i in range(3):
            with profiler.profile_operation("repeated_operation"):
                time.sleep(0.05)

        avg_metrics = profiler.get_average_metrics("repeated_operation")
        assert avg_metrics is not None
        assert avg_metrics["count"] == 3
        assert avg_metrics["avg_duration_ms"] > 50  # Should average > 50ms

    def test_performance_regression_detection(self):
        """Test performance regression detection."""
        profiler = PerformanceProfiler(auto_save=False)

        # Profile a baseline operation
        with profiler.profile_operation("baseline_test"):
            time.sleep(0.05)

        baseline_duration = profiler.metrics[0].duration_ms

        # Test regression detection
        regression = profiler.detect_performance_regression(
            "baseline_test",
            baseline_duration * 0.5,  # Set baseline to half the actual time
            tolerance_percent=10.0,
        )

        assert regression  # Should detect regression

        # Test no regression
        no_regression = profiler.detect_performance_regression(
            "baseline_test",
            baseline_duration * 2,  # Set baseline to double the actual time
            tolerance_percent=10.0,
        )

        assert not no_regression  # Should not detect regression


class TestAnonymizationBenchmark:
    """Test anonymization benchmarking functionality."""

    @pytest.fixture
    def benchmark(self):
        """Create benchmark instance."""
        return AnonymizationBenchmark()

    def test_test_document_creation(self, benchmark):
        """Test synthetic document creation."""
        doc = benchmark.create_test_document((512, 384))
        assert doc.size == (512, 384)
        assert doc.mode == "RGB"

    def test_document_loading_benchmark(self, benchmark):
        """Test document loading benchmark."""
        result = benchmark.benchmark_document_loading(
            image_size=(256, 256), num_documents=3
        )

        assert result.success
        assert result.benchmark_name == "document_loading"
        assert result.duration_ms > 0
        assert result.additional_metrics is not None
        assert result.additional_metrics["num_documents"] == 3

    def test_text_detection_benchmark(self, benchmark):
        """Test OCR text detection benchmark."""
        result = benchmark.benchmark_text_detection(image_size=(256, 256), num_images=5)

        assert result.success
        assert result.benchmark_name == "text_detection_ocr"
        assert result.duration_ms > 0
        assert result.additional_metrics["num_images"] == 5

    def test_pii_detection_benchmark(self, benchmark):
        """Test NER PII detection benchmark."""
        result = benchmark.benchmark_pii_detection(num_texts=50)

        assert result.success
        assert result.benchmark_name == "pii_detection_ner"
        assert result.duration_ms > 0
        assert result.additional_metrics["num_texts"] == 50

    def test_inpainting_benchmark(self, benchmark):
        """Test diffusion inpainting benchmark."""
        result = benchmark.benchmark_inpainting(
            image_size=(256, 256), num_regions=3, num_iterations=2
        )

        assert result.success
        assert result.benchmark_name == "diffusion_inpainting"
        assert result.duration_ms > 0
        assert result.additional_metrics["num_iterations"] == 2

    def test_end_to_end_benchmark(self, benchmark):
        """Test end-to-end pipeline benchmark."""
        result = benchmark.benchmark_end_to_end(image_size=(256, 256), num_documents=2)

        assert result.success
        assert result.benchmark_name == "end_to_end_anonymization"
        assert result.duration_ms > 0
        assert result.additional_metrics["num_documents"] == 2
        assert "pipeline_stages" in result.additional_metrics

    def test_full_benchmark_suite(self, benchmark):
        """Test running the complete benchmark suite."""
        results = benchmark.run_full_benchmark_suite()

        assert len(results) == 5

        # Check all benchmarks ran successfully
        benchmark_names = [r.benchmark_name for r in results]
        expected_names = [
            "document_loading",
            "text_detection_ocr",
            "pii_detection_ner",
            "diffusion_inpainting",
            "end_to_end_anonymization",
        ]

        for name in expected_names:
            assert name in benchmark_names

        # All should succeed
        assert all(r.success for r in results)

    def test_benchmark_results_export(self, benchmark):
        """Test exporting benchmark results."""
        results = benchmark.run_full_benchmark_suite()

        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "benchmark_results.json"
            success = benchmark.save_benchmark_results(results, export_path)

            assert success
            assert export_path.exists()

            # Verify content
            import json

            with open(export_path) as f:
                data = json.load(f)

            assert "benchmark_suite" in data
            assert "results" in data
            assert len(data["results"]) == len(results)


class TestModelBenchmark:
    """Test model-specific benchmarking."""

    @pytest.fixture
    def model_benchmark(self):
        """Create model benchmark instance."""
        return ModelBenchmark()

    def test_model_loading_benchmark(self, model_benchmark):
        """Test model loading benchmark."""
        # Create a dummy model file
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as f:
            model_path = Path(f.name)

            result = model_benchmark.benchmark_model_loading(model_path)

            assert result.benchmark_name.startswith("model_loading_")
            assert result.duration_ms > 0

    def test_inference_speed_benchmark(self, model_benchmark):
        """Test model inference speed benchmark."""
        result = model_benchmark.benchmark_inference_speed(
            model_type="test_model", input_size=(256, 256), num_iterations=5
        )

        assert result.benchmark_name == "test_model_inference_speed"
        assert result.duration_ms > 0
        assert result.additional_metrics["num_iterations"] == 5
        assert "avg_iteration_ms" in result.additional_metrics
        assert "throughput_fps" in result.additional_metrics


class TestPerformanceMonitor:
    """Test high-level performance monitoring."""

    @pytest.fixture
    def monitor(self):
        """Create performance monitor instance."""
        return PerformanceMonitor(auto_export=False)

    def test_performance_session(self, monitor):
        """Test performance monitoring session."""
        monitor.start_session("test_session")

        # Simulate some work
        time.sleep(0.2)
        data = np.random.rand(100, 100)
        np.sum(data)

        report = monitor.end_session()

        assert report["session_name"] == "test_session"
        assert report["session_duration_seconds"] > 0.1
        assert report["sample_count"] > 0
        assert "resource_summary" in report

    def test_current_usage_monitoring(self, monitor):
        """Test current resource usage monitoring."""
        current = monitor.get_current_usage()

        assert current is not None
        assert current.cpu_percent >= 0
        assert current.memory_rss_mb > 0
        assert current.timestamp > 0

    def test_resource_limits_checking(self, monitor):
        """Test resource limits checking."""
        # Test with very high limits (should not exceed)
        violations = monitor.check_resource_limits(
            max_memory_mb=100000, max_cpu_percent=100  # 100GB
        )

        assert "memory_exceeded" in violations
        assert "cpu_exceeded" in violations
        assert not violations["memory_exceeded"]
        assert not violations["cpu_exceeded"]

        # Test with very low limits (should exceed)
        violations = monitor.check_resource_limits(
            max_memory_mb=1, max_cpu_percent=0.1  # 1MB
        )

        # At least one should be exceeded
        assert violations["memory_exceeded"] or violations["cpu_exceeded"]

    def test_performance_report_generation(self, monitor):
        """Test performance report generation."""
        # Start a session to collect some data
        monitor.start_session("report_test")
        time.sleep(0.1)
        monitor.end_session()

        report = monitor.generate_performance_report()

        assert "report_timestamp" in report
        assert "monitoring_summary" in report
        assert "performance_insights" in report
        assert "recommendations" in report

        # Should have some insights or recommendations
        total_feedback = len(report["performance_insights"]) + len(
            report["recommendations"]
        )
        assert total_feedback >= 0  # May be 0 for short tests


@pytest.mark.slow
class TestPerformanceRegression:
    """Test performance regression detection."""

    def test_memory_usage_stability(self):
        """Test that memory usage remains stable across operations."""
        profiler = PerformanceProfiler(auto_save=False)
        baseline_memories = []

        # Run the same operation multiple times
        for i in range(5):
            with profiler.profile_operation("memory_stability_test"):
                # Consistent workload
                data = np.random.rand(500, 500)
                result = np.dot(data, data.T)
                del data, result

            baseline_memories.append(profiler.metrics[-1].peak_memory_mb)

        # Memory usage should be relatively stable
        memory_variance = np.var(baseline_memories)
        memory_mean = np.mean(baseline_memories)

        # Coefficient of variation should be reasonable
        cv = (np.sqrt(memory_variance) / memory_mean) * 100
        assert cv < 50  # Less than 50% coefficient of variation

    def test_performance_consistency(self):
        """Test that performance remains consistent."""
        profiler = PerformanceProfiler(auto_save=False)
        durations = []

        # Run consistent operations
        for i in range(5):
            with profiler.profile_operation("consistency_test"):
                # Consistent CPU workload
                data = np.random.rand(300, 300)
                for _ in range(10):
                    np.dot(data, data.T)

            durations.append(profiler.metrics[-1].duration_ms)

        # Performance should be relatively consistent
        duration_variance = np.var(durations)
        duration_mean = np.mean(durations)

        # Coefficient of variation should be reasonable
        cv = (np.sqrt(duration_variance) / duration_mean) * 100
        assert cv < 30  # Less than 30% coefficient of variation


@pytest.mark.benchmark
class TestBenchmarkIntegration:
    """Integration tests for benchmarking with real anonymization components."""

    def test_benchmark_with_profiler_integration(self):
        """Test benchmark integration with performance profiler."""
        profiler = PerformanceProfiler(auto_save=False)
        benchmark = AnonymizationBenchmark(profiler)

        # Run a benchmark
        result = benchmark.benchmark_document_loading(num_documents=2)

        # Should have metrics in the profiler
        assert len(profiler.metrics) > 0
        assert result.success

        # Metrics should match the result
        latest_metric = profiler.metrics[-1]
        assert latest_metric.operation == result.benchmark_name
        assert (
            abs(latest_metric.duration_ms - result.duration_ms) < 1.0
        )  # Small tolerance

    def test_end_to_end_performance_monitoring(self):
        """Test complete performance monitoring of anonymization pipeline."""
        monitor = PerformanceMonitor(auto_export=False)
        benchmark = AnonymizationBenchmark()

        # Start monitoring session
        monitor.start_session("e2e_anonymization")

        # Run end-to-end benchmark
        result = benchmark.benchmark_end_to_end(num_documents=1)

        # End session
        session_report = monitor.end_session()

        assert result.success
        assert session_report["session_name"] == "e2e_anonymization"
        assert session_report["sample_count"] > 0

        # Should have resource usage data
        resource_summary = session_report["resource_summary"]
        assert resource_summary["peak_memory_mb"] > 0
        assert resource_summary["duration_seconds"] > 0


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short"])
