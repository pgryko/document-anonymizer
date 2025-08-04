"""
Resource Monitor
================

Real-time monitoring of system resources during anonymization operations.
"""

import json
import logging
import queue
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)


@dataclass
class ResourceSample:
    """Single resource usage sample."""

    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_rss_mb: float
    memory_vms_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    gpu_memory_mb: float | None = None
    gpu_utilization: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ResourceSummary:
    """Summary of resource usage over a period."""

    duration_seconds: float
    cpu_percent: dict[str, float]  # min, max, avg
    memory_percent: dict[str, float]
    memory_rss_mb: dict[str, float]
    peak_memory_mb: float
    total_disk_io_mb: float
    total_network_mb: float
    gpu_peak_memory_mb: float | None = None
    gpu_avg_utilization: float | None = None
    sample_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ResourceMonitor:
    """
    Monitors system resource usage in real-time.

    Features:
    - CPU, memory, disk, and network monitoring
    - GPU monitoring (if available)
    - Configurable sampling rate
    - Resource usage alerts
    - Historical data collection
    """

    def __init__(self, sample_interval: float = 1.0, max_samples: int = 1000):
        self.sample_interval = sample_interval
        self.max_samples = max_samples
        self.is_monitoring = False

        self.samples: list[ResourceSample] = []
        self._monitor_thread: threading.Thread | None = None
        self._sample_queue = queue.Queue()

        # Initialize process reference
        self.process = psutil.Process()

        # Try to import GPU monitoring
        try:
            import pynvml

            pynvml.nvmlInit()
            self.gpu_available = True
            self._pynvml = pynvml
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            logger.info("GPU monitoring enabled")
        except (ImportError, Exception):
            self.gpu_available = False
            self._pynvml = None
            self.gpu_handle = None
            logger.info("GPU monitoring not available")

        # Baseline measurements for deltas
        self._baseline_disk_io = self.process.io_counters()
        self._baseline_network = psutil.net_io_counters()

        logger.info(f"ResourceMonitor initialized (interval: {sample_interval}s)")

    def _get_gpu_stats(self) -> tuple[float | None, float | None]:
        """Get GPU memory and utilization stats."""
        if not self.gpu_available or not self._pynvml:
            return None, None

        try:
            # Memory info
            mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            memory_mb = mem_info.used / (1024 * 1024)

            # Utilization info
            util_info = self._pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            utilization = util_info.gpu

            return memory_mb, utilization

        except Exception as e:
            logger.warning(f"Error getting GPU stats: {e}")
            return None, None

    def _take_sample(self) -> ResourceSample:
        """Take a resource usage sample."""
        try:
            # CPU and memory
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()

            # Disk I/O
            current_disk_io = self.process.io_counters()
            disk_read_mb = (current_disk_io.read_bytes - self._baseline_disk_io.read_bytes) / (
                1024 * 1024
            )
            disk_write_mb = (current_disk_io.write_bytes - self._baseline_disk_io.write_bytes) / (
                1024 * 1024
            )

            # Network I/O
            current_network = psutil.net_io_counters()
            network_sent_mb = (current_network.bytes_sent - self._baseline_network.bytes_sent) / (
                1024 * 1024
            )
            network_recv_mb = (current_network.bytes_recv - self._baseline_network.bytes_recv) / (
                1024 * 1024
            )

            # GPU stats
            gpu_memory_mb, gpu_utilization = self._get_gpu_stats()

            return ResourceSample(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_rss_mb=memory_info.rss / (1024 * 1024),
                memory_vms_mb=memory_info.vms / (1024 * 1024),
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                gpu_memory_mb=gpu_memory_mb,
                gpu_utilization=gpu_utilization,
            )

        except Exception as e:
            logger.warning(f"Error taking resource sample: {e}")
            # Return a minimal sample
            return ResourceSample(
                timestamp=time.time(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_rss_mb=0.0,
                memory_vms_mb=0.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
            )

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self.is_monitoring:
            try:
                sample = self._take_sample()
                self._sample_queue.put(sample)

                # Maintain sample limit
                if len(self.samples) >= self.max_samples:
                    self.samples.pop(0)  # Remove oldest sample

                time.sleep(self.sample_interval)

            except Exception:
                logger.exception("Error in monitoring loop")
                time.sleep(self.sample_interval)

    def start_monitoring(self):
        """Start resource monitoring."""
        if self.is_monitoring:
            return

        logger.info("Starting resource monitoring")
        self.is_monitoring = True
        self.samples.clear()

        # Reset baselines
        try:
            self._baseline_disk_io = self.process.io_counters()
            self._baseline_network = psutil.net_io_counters()
        except Exception as e:
            logger.warning(f"Failed to reset monitoring baselines: {e}")

        # Start monitoring thread
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> list[ResourceSample]:
        """Stop monitoring and return collected samples."""
        if not self.is_monitoring:
            return self.samples.copy()

        logger.info("Stopping resource monitoring")
        self.is_monitoring = False

        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)

        # Collect any remaining samples from queue
        while not self._sample_queue.empty():
            try:
                sample = self._sample_queue.get_nowait()
                self.samples.append(sample)
            except queue.Empty:
                break

        logger.info(f"Collected {len(self.samples)} resource samples")
        return self.samples.copy()

    def get_current_stats(self) -> ResourceSample | None:
        """Get current resource usage stats."""
        try:
            return self._take_sample()
        except Exception:
            logger.exception("Error getting current stats")
            return None

    def generate_summary(self, samples: list[ResourceSample] | None = None) -> ResourceSummary:
        """Generate summary statistics from samples."""
        if samples is None:
            samples = self.samples

        if not samples:
            return ResourceSummary(
                duration_seconds=0,
                cpu_percent={"min": 0, "max": 0, "avg": 0},
                memory_percent={"min": 0, "max": 0, "avg": 0},
                memory_rss_mb={"min": 0, "max": 0, "avg": 0},
                peak_memory_mb=0,
                total_disk_io_mb=0,
                total_network_mb=0,
                sample_count=0,
            )

        # Calculate duration
        start_time = samples[0].timestamp
        end_time = samples[-1].timestamp
        duration = end_time - start_time

        # Extract metrics
        cpu_values = [s.cpu_percent for s in samples]
        memory_percent_values = [s.memory_percent for s in samples]
        memory_rss_values = [s.memory_rss_mb for s in samples]

        # Calculate statistics
        def calc_stats(values):
            return {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }

        # GPU statistics
        gpu_memory_values = [s.gpu_memory_mb for s in samples if s.gpu_memory_mb is not None]
        gpu_util_values = [s.gpu_utilization for s in samples if s.gpu_utilization is not None]

        gpu_peak_memory = max(gpu_memory_values) if gpu_memory_values else None
        gpu_avg_util = sum(gpu_util_values) / len(gpu_util_values) if gpu_util_values else None

        # Total I/O
        total_disk_io = sum(s.disk_io_read_mb + s.disk_io_write_mb for s in samples)
        total_network = sum(s.network_sent_mb + s.network_recv_mb for s in samples)

        return ResourceSummary(
            duration_seconds=duration,
            cpu_percent=calc_stats(cpu_values),
            memory_percent=calc_stats(memory_percent_values),
            memory_rss_mb=calc_stats(memory_rss_values),
            peak_memory_mb=max(memory_rss_values),
            total_disk_io_mb=total_disk_io,
            total_network_mb=total_network,
            gpu_peak_memory_mb=gpu_peak_memory,
            gpu_avg_utilization=gpu_avg_util,
            sample_count=len(samples),
        )

    def export_samples(self, filepath: Path) -> bool:
        """Export samples to JSON file."""
        try:
            data = {
                "monitoring_session": {
                    "start_time": self.samples[0].timestamp if self.samples else 0,
                    "end_time": self.samples[-1].timestamp if self.samples else 0,
                    "sample_interval": self.sample_interval,
                    "sample_count": len(self.samples),
                },
                "summary": self.generate_summary().to_dict(),
                "samples": [sample.to_dict() for sample in self.samples],
            }

            filepath.parent.mkdir(parents=True, exist_ok=True)
            with filepath.open("w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Exported {len(self.samples)} samples to {filepath}")
            return True

        except Exception:
            logger.exception("Failed to export samples")
            return False


class PerformanceMonitor:
    """
    High-level performance monitoring coordinator.

    Combines resource monitoring with operation profiling
    to provide comprehensive performance insights.
    """

    def __init__(self, results_dir: Path | None = None, auto_export: bool = True):
        self.results_dir = results_dir or Path("./performance_results")
        self.auto_export = auto_export

        # Initialize components
        self.resource_monitor = ResourceMonitor()

        if self.auto_export:
            self.results_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"PerformanceMonitor initialized (results_dir: {self.results_dir})")

    def start_session(self, session_name: str):
        """Start a performance monitoring session."""
        self.session_name = session_name
        self.session_start_time = time.time()

        logger.info(f"Starting performance session: {session_name}")
        self.resource_monitor.start_monitoring()

    def end_session(self) -> dict[str, Any]:
        """End the current session and generate report."""
        if not hasattr(self, "session_name"):
            logger.warning("No active session to end")
            return {}

        # Stop monitoring
        samples = self.resource_monitor.stop_monitoring()
        summary = self.resource_monitor.generate_summary(samples)

        session_duration = time.time() - self.session_start_time

        # Generate session report
        report = {
            "session_name": self.session_name,
            "session_duration_seconds": session_duration,
            "start_time": self.session_start_time,
            "end_time": time.time(),
            "resource_summary": summary.to_dict(),
            "sample_count": len(samples),
        }

        logger.info(
            f"Ended performance session: {self.session_name} "
            f"(duration: {session_duration:.1f}s, samples: {len(samples)})"
        )

        # Auto-export if enabled
        if self.auto_export:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = self.results_dir / f"session_{self.session_name}_{timestamp}.json"
            self.resource_monitor.export_samples(export_path)

        return report

    def get_current_usage(self) -> ResourceSample | None:
        """Get current resource usage."""
        return self.resource_monitor.get_current_stats()

    def check_resource_limits(
        self,
        max_memory_mb: float | None = None,
        max_cpu_percent: float | None = None,
    ) -> dict[str, bool]:
        """Check if current usage exceeds specified limits."""
        current = self.get_current_usage()
        if not current:
            return {"error": True}

        violations = {}

        if max_memory_mb:
            violations["memory_exceeded"] = current.memory_rss_mb > max_memory_mb

        if max_cpu_percent:
            violations["cpu_exceeded"] = current.cpu_percent > max_cpu_percent

        return violations

    def generate_performance_report(self) -> dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.resource_monitor.samples:
            return {"error": "No performance data available"}

        summary = self.resource_monitor.generate_summary()
        current_stats = self.get_current_usage()

        return {
            "report_timestamp": datetime.now().isoformat(),
            "monitoring_summary": summary.to_dict(),
            "current_usage": current_stats.to_dict() if current_stats else None,
            "performance_insights": self._generate_insights(summary),
            "recommendations": self._generate_recommendations(summary),
        }

    def _generate_insights(self, summary: ResourceSummary) -> list[str]:
        """Generate performance insights from summary data."""
        insights = []

        # Memory insights
        if summary.peak_memory_mb > 8000:  # 8GB
            insights.append(
                "High memory usage detected - consider batch processing smaller documents"
            )

        if summary.memory_percent["max"] > 90:
            insights.append("Memory usage exceeded 90% - risk of OOM errors")

        # CPU insights
        if summary.cpu_percent["avg"] < 30:
            insights.append("Low CPU utilization - workload may be I/O bound")
        elif summary.cpu_percent["avg"] > 80:
            insights.append("High CPU utilization - consider parallel processing")

        # GPU insights
        if summary.gpu_peak_memory_mb and summary.gpu_peak_memory_mb > 20000:  # 20GB
            insights.append("High GPU memory usage - consider reducing batch size")

        if summary.gpu_avg_utilization and summary.gpu_avg_utilization < 50:
            insights.append("Low GPU utilization - workload may not be GPU-optimized")

        # I/O insights
        if summary.total_disk_io_mb > 1000:  # 1GB
            insights.append("High disk I/O detected - consider using faster storage")

        return insights

    def _generate_recommendations(self, summary: ResourceSummary) -> list[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # Memory recommendations
        if summary.peak_memory_mb > 16000:  # 16GB
            recommendations.append("Consider implementing memory optimization techniques")
            recommendations.append("Use gradient checkpointing for model training")

        # Performance recommendations
        if summary.cpu_percent["avg"] > 70 and summary.duration_seconds > 60:
            recommendations.append("Consider using multiple worker processes")
            recommendations.append("Implement asynchronous processing for I/O operations")

        # GPU recommendations
        if summary.gpu_avg_utilization and summary.gpu_avg_utilization < 60:
            recommendations.append("Increase batch size to improve GPU utilization")
            recommendations.append("Consider mixed precision training")

        return recommendations
