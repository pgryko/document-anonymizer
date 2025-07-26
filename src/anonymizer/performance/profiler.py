"""
Performance Profiler
=====================

Tools for profiling memory usage and performance characteristics
of the document anonymization pipeline.
"""

import time
import logging
import psutil
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""

    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    peak_memory_mb: float
    avg_memory_mb: float
    cpu_percent: float
    gpu_memory_mb: Optional[float] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time."""

    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory percentage
    available_mb: float
    gpu_memory_mb: Optional[float] = None


class MemoryProfiler:
    """
    Profiles memory usage during operations.

    Features:
    - Real-time memory monitoring
    - Peak memory tracking
    - Memory leak detection
    - GPU memory monitoring (if available)
    """

    def __init__(self, sample_interval: float = 0.1):
        self.sample_interval = sample_interval
        self.is_monitoring = False
        self.snapshots: List[MemorySnapshot] = []
        self._monitor_thread: Optional[threading.Thread] = None

        # Try to import torch for GPU monitoring
        try:
            import torch

            self.has_gpu = torch.cuda.is_available()
            self._torch = torch
        except ImportError:
            self.has_gpu = False
            self._torch = None

        logger.info(f"MemoryProfiler initialized (GPU available: {self.has_gpu})")

    def _get_gpu_memory_mb(self) -> Optional[float]:
        """Get current GPU memory usage in MB."""
        if not self.has_gpu or not self._torch:
            return None

        try:
            memory_bytes = self._torch.cuda.memory_allocated()
            return memory_bytes / (1024 * 1024)  # Convert to MB
        except Exception:
            return None

    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot."""
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()

        return MemorySnapshot(
            timestamp=time.time(),
            rss_mb=memory_info.rss / (1024 * 1024),
            vms_mb=memory_info.vms / (1024 * 1024),
            percent=process.memory_percent(),
            available_mb=system_memory.available / (1024 * 1024),
            gpu_memory_mb=self._get_gpu_memory_mb(),
        )

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                time.sleep(self.sample_interval)
            except Exception as e:
                logger.warning(f"Error taking memory snapshot: {e}")
                time.sleep(self.sample_interval)

    def start_monitoring(self):
        """Start background memory monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.snapshots.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.debug("Started memory monitoring")

    def stop_monitoring(self) -> List[MemorySnapshot]:
        """Stop monitoring and return collected snapshots."""
        if not self.is_monitoring:
            return self.snapshots

        self.is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

        logger.debug(f"Stopped memory monitoring ({len(self.snapshots)} snapshots)")
        return self.snapshots.copy()

    def get_peak_memory(self) -> float:
        """Get peak RSS memory usage in MB."""
        if not self.snapshots:
            return 0.0
        return max(snapshot.rss_mb for snapshot in self.snapshots)

    def get_average_memory(self) -> float:
        """Get average RSS memory usage in MB."""
        if not self.snapshots:
            return 0.0
        return sum(snapshot.rss_mb for snapshot in self.snapshots) / len(self.snapshots)

    def get_peak_gpu_memory(self) -> Optional[float]:
        """Get peak GPU memory usage in MB."""
        if not self.has_gpu or not self.snapshots:
            return None

        gpu_values = [
            s.gpu_memory_mb for s in self.snapshots if s.gpu_memory_mb is not None
        ]
        return max(gpu_values) if gpu_values else None

    @contextmanager
    def profile_memory(self):
        """Context manager for profiling memory during an operation."""
        self.start_monitoring()
        try:
            yield self
        finally:
            self.stop_monitoring()


class PerformanceProfiler:
    """
    Comprehensive performance profiler for the anonymization pipeline.

    Features:
    - Operation timing
    - Memory usage tracking
    - CPU utilization monitoring
    - Performance regression detection
    - Detailed metrics collection
    """

    def __init__(self, auto_save: bool = True, results_dir: Optional[Path] = None):
        self.auto_save = auto_save
        self.results_dir = results_dir or Path("./performance_results")
        self.metrics: List[PerformanceMetrics] = []
        self.memory_profiler = MemoryProfiler()

        if self.auto_save:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"PerformanceProfiler initialized (results_dir: {self.results_dir})"
        )

    @contextmanager
    def profile_operation(self, operation_name: str):
        """Context manager for profiling a specific operation."""
        logger.debug(f"Starting profiling: {operation_name}")

        # Start monitoring
        start_time = time.time()
        process = psutil.Process()
        start_cpu_times = process.cpu_times()

        self.memory_profiler.start_monitoring()

        try:
            yield

        finally:
            # Stop monitoring and collect metrics
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000

            # Get memory metrics
            self.memory_profiler.stop_monitoring()
            peak_memory = self.memory_profiler.get_peak_memory()
            avg_memory = self.memory_profiler.get_average_memory()

            # Calculate CPU usage
            end_cpu_times = process.cpu_times()
            cpu_time_used = (end_cpu_times.user - start_cpu_times.user) + (
                end_cpu_times.system - start_cpu_times.system
            )
            cpu_percent = (cpu_time_used / (end_time - start_time)) * 100

            # Get GPU memory if available
            gpu_memory = self.memory_profiler.get_peak_gpu_memory()

            # Create metrics
            metrics = PerformanceMetrics(
                operation=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                peak_memory_mb=peak_memory,
                avg_memory_mb=avg_memory,
                cpu_percent=cpu_percent,
                gpu_memory_mb=gpu_memory,
            )

            self.metrics.append(metrics)

            logger.info(
                f"Profiled {operation_name}: {duration_ms:.1f}ms, "
                f"peak memory: {peak_memory:.1f}MB, CPU: {cpu_percent:.1f}%"
            )

            # Auto-save if enabled
            if self.auto_save:
                self._save_metrics()

    def get_metrics_for_operation(
        self, operation_name: str
    ) -> List[PerformanceMetrics]:
        """Get all metrics for a specific operation."""
        return [m for m in self.metrics if m.operation == operation_name]

    def get_average_metrics(self, operation_name: str) -> Optional[Dict[str, float]]:
        """Get average metrics for an operation."""
        operation_metrics = self.get_metrics_for_operation(operation_name)
        if not operation_metrics:
            return None

        return {
            "avg_duration_ms": sum(m.duration_ms for m in operation_metrics)
            / len(operation_metrics),
            "avg_peak_memory_mb": sum(m.peak_memory_mb for m in operation_metrics)
            / len(operation_metrics),
            "avg_cpu_percent": sum(m.cpu_percent for m in operation_metrics)
            / len(operation_metrics),
            "count": len(operation_metrics),
        }

    def detect_performance_regression(
        self,
        operation_name: str,
        baseline_duration_ms: float,
        tolerance_percent: float = 20.0,
    ) -> bool:
        """
        Detect if there's a performance regression compared to baseline.

        Args:
            operation_name: Name of operation to check
            baseline_duration_ms: Expected baseline duration
            tolerance_percent: Acceptable percentage increase

        Returns:
            True if regression detected
        """
        recent_metrics = self.get_metrics_for_operation(operation_name)
        if not recent_metrics:
            return False

        # Check the most recent measurement
        latest_duration = recent_metrics[-1].duration_ms
        threshold = baseline_duration_ms * (1 + tolerance_percent / 100)

        if latest_duration > threshold:
            logger.warning(
                f"Performance regression detected for {operation_name}: "
                f"{latest_duration:.1f}ms > {threshold:.1f}ms threshold"
            )
            return True

        return False

    def _save_metrics(self):
        """Save metrics to disk."""
        if not self.results_dir:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"performance_metrics_{timestamp}.json"
        filepath = self.results_dir / filename

        try:
            data = {
                "timestamp": datetime.now().isoformat(),
                "metrics": [m.to_dict() for m in self.metrics],
            }

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved {len(self.metrics)} metrics to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def export_metrics(self, filepath: Path) -> bool:
        """Export all metrics to a JSON file."""
        try:
            data = {
                "export_timestamp": datetime.now().isoformat(),
                "total_metrics": len(self.metrics),
                "metrics": [m.to_dict() for m in self.metrics],
            }

            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Exported {len(self.metrics)} metrics to {filepath}")
            return True

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False

    def clear_metrics(self):
        """Clear all collected metrics."""
        self.metrics.clear()
        logger.info("Cleared all performance metrics")

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of all collected metrics."""
        if not self.metrics:
            return {"error": "No metrics collected"}

        # Group by operation
        operations = {}
        for metric in self.metrics:
            op_name = metric.operation
            if op_name not in operations:
                operations[op_name] = []
            operations[op_name].append(metric)

        # Calculate statistics for each operation
        summary = {}
        for op_name, op_metrics in operations.items():
            durations = [m.duration_ms for m in op_metrics]
            memory_peaks = [m.peak_memory_mb for m in op_metrics]

            summary[op_name] = {
                "count": len(op_metrics),
                "duration_ms": {
                    "min": min(durations),
                    "max": max(durations),
                    "avg": sum(durations) / len(durations),
                },
                "peak_memory_mb": {
                    "min": min(memory_peaks),
                    "max": max(memory_peaks),
                    "avg": sum(memory_peaks) / len(memory_peaks),
                },
                "cpu_percent_avg": sum(m.cpu_percent for m in op_metrics)
                / len(op_metrics),
            }

        return {
            "total_operations": len(self.metrics),
            "unique_operations": len(operations),
            "operations": summary,
            "generated_at": datetime.now().isoformat(),
        }
