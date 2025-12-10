"""Model Caching and Batch Optimization for Inference
================================================

High-performance caching and batching system for document anonymization inference:
- Smart model caching with LRU eviction
- Batch processing for improved throughput
- Memory-efficient tensor management
- Thread-safe operations
- Automatic cache warming and preloading
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from diffusers import StableDiffusionInpaintPipeline

from src.anonymizer.core.exceptions import BatchProcessingTimeoutError

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Represents a cached model entry with metadata."""

    model: Any
    model_hash: str
    last_accessed: float
    access_count: int
    memory_size_mb: float
    created_at: float

    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Statistics for model cache performance."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_memory_mb: float = 0.0
    max_memory_mb: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return (self.hits / total * 100.0) if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate_percent": self.hit_rate,
            "total_memory_mb": self.total_memory_mb,
            "max_memory_mb": self.max_memory_mb,
        }


class ModelCache:
    """Thread-safe LRU cache for machine learning models with memory management."""

    def __init__(
        self,
        max_size: int = 3,
        max_memory_mb: float = 8192.0,  # 8GB default
        enable_stats: bool = True,
    ):
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.enable_stats = enable_stats

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats() if enable_stats else None

        # Memory tracking
        self._current_memory_mb = 0.0

        logger.info(f"ModelCache initialized: max_size={max_size}, max_memory={max_memory_mb}MB")

    def _compute_model_hash(self, model_path: str, config_dict: dict[str, Any]) -> str:
        """Compute unique hash for model configuration."""
        # Create hash from model path and relevant config
        hash_input = f"{model_path}:{sorted(config_dict.items())!s}"
        # Use SHA256 for better security (MD5 has collision vulnerabilities)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]

    def _estimate_model_memory(self, model: Any) -> float:
        """Estimate memory usage of a model in MB."""
        try:
            if hasattr(model, "get_memory_footprint"):
                # For Hugging Face models
                return model.get_memory_footprint() / (1024 * 1024)
            if hasattr(model, "state_dict"):
                # For PyTorch models
                total_params = 0
                for param in model.parameters():
                    total_params += param.numel()
                # Estimate: 4 bytes per float32 parameter
                return (total_params * 4) / (1024 * 1024)
            if isinstance(model, StableDiffusionInpaintPipeline):
                # Estimate for diffusion pipeline (rough approximation)
                return 2048.0  # ~2GB for SD pipeline
        except Exception as e:
            logger.warning(f"Failed to estimate model memory: {e}")
            return 512.0
        else:
            # Fallback estimation
            return 512.0  # 512MB default

    def _evict_lru(self, required_memory: float = 0.0):
        """Evict least recently used models to make space."""
        while self._cache and (
            len(self._cache) >= self.max_size
            or self._current_memory_mb + required_memory > self.max_memory_mb
        ):
            # Get LRU item (first item in OrderedDict)
            cache_key, entry = self._cache.popitem(last=False)
            self._current_memory_mb -= entry.memory_size_mb

            if self._stats:
                self._stats.evictions += 1

            logger.info(
                f"Evicted model {cache_key}: freed {entry.memory_size_mb:.1f}MB "
                f"(accessed {entry.access_count} times)"
            )

            # Clean up GPU memory if applicable
            try:
                if hasattr(entry.model, "to") and torch.cuda.is_available():
                    entry.model.to("cpu")
                del entry.model  # Help with garbage collection
            except Exception as e:
                logger.debug(f"Error during model cleanup: {e}")

    def get(
        self,
        model_path: str,
        config_dict: dict[str, Any],
        loader_func: Callable | None = None,
    ) -> tuple[Any | None, bool]:
        """Get model from cache or load it.

        Args:
            model_path: Path to the model
            config_dict: Model configuration dictionary
            loader_func: Function to load model if not cached

        Returns:
            Tuple of (model, is_cache_hit)
        """
        cache_key = self._compute_model_hash(model_path, config_dict)

        with self._lock:
            # Check cache first
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                entry.update_access()

                # Move to end (most recently used)
                self._cache.move_to_end(cache_key)

                if self._stats:
                    self._stats.hits += 1

                logger.debug(f"Cache hit for model {cache_key}")
                return entry.model, True

            # Cache miss - need to load model
            if self._stats:
                self._stats.misses += 1

            if loader_func is None:
                logger.warning(f"Cache miss for model {cache_key} but no loader provided")
                return None, False

            logger.info(f"Cache miss for model {cache_key}, loading...")

            # Load the model
            try:
                start_time = time.time()
                model = loader_func()
                load_time = time.time() - start_time

                if model is None:
                    return None, False

                # Estimate memory usage
                memory_size = self._estimate_model_memory(model)

                # Ensure we have space
                self._evict_lru(required_memory=memory_size)

                # Create cache entry
                entry = CacheEntry(
                    model=model,
                    model_hash=cache_key,
                    last_accessed=time.time(),
                    access_count=1,
                    memory_size_mb=memory_size,
                    created_at=time.time(),
                )

                # Add to cache
                self._cache[cache_key] = entry
                self._current_memory_mb += memory_size

                # Update max memory tracking
                if self._stats:
                    self._stats.max_memory_mb = max(
                        self._stats.max_memory_mb, self._current_memory_mb
                    )
                    self._stats.total_memory_mb = self._current_memory_mb

                logger.info(
                    f"Loaded and cached model {cache_key}: {memory_size:.1f}MB "
                    f"in {load_time:.2f}s (cache: {self._current_memory_mb:.1f}MB total)"
                )

                return model, False

            except Exception:
                logger.exception(f"Failed to load model {model_path}")
                return None, False

    def preload(
        self,
        model_specs: list[tuple[str, dict[str, Any], Callable]],
        max_concurrent: int = 2,
    ):
        """Preload multiple models into cache concurrently.

        Args:
            model_specs: List of (model_path, config_dict, loader_func) tuples
            max_concurrent: Maximum number of concurrent loads
        """
        import concurrent.futures

        logger.info(f"Preloading {len(model_specs)} models with max_concurrent={max_concurrent}")

        def load_single(spec):
            model_path, config_dict, loader_func = spec
            cache_key = self._compute_model_hash(model_path, config_dict)

            with self._lock:
                if cache_key in self._cache:
                    logger.debug(f"Model {cache_key} already cached, skipping preload")
                    return cache_key, True

            # Load outside of lock to avoid blocking
            try:
                start_time = time.time()
                model, is_hit = self.get(model_path, config_dict, loader_func)
                load_time = time.time() - start_time

                if model is not None:
                    logger.info(f"Preloaded model {cache_key} in {load_time:.2f}s")
                    return cache_key, True
                logger.warning(f"Failed to preload model {cache_key}")
                return cache_key, False

            except Exception:
                logger.exception(f"Error preloading model {cache_key}")
                return cache_key, False

        # Execute preloading with thread pool
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(load_single, spec) for spec in model_specs]

            successful = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    cache_key, success = future.result()
                    if success:
                        successful += 1
                except Exception:
                    logger.exception("Preload task failed")

        logger.info(
            f"Preloading complete: {successful}/{len(model_specs)} models loaded successfully"
        )

    def clear(self):
        """Clear all cached models."""
        with self._lock:
            evicted_count = len(self._cache)
            freed_memory = self._current_memory_mb

            # Clean up models
            for entry in self._cache.values():
                try:
                    if hasattr(entry.model, "to") and torch.cuda.is_available():
                        entry.model.to("cpu")
                    del entry.model
                except Exception as e:
                    logger.debug(f"Error during cache clear: {e}")

            self._cache.clear()
            self._current_memory_mb = 0.0

            logger.info(f"Cache cleared: {evicted_count} models, {freed_memory:.1f}MB freed")

    def get_stats(self) -> CacheStats | None:
        """Get cache statistics."""
        if not self._stats:
            return None

        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                total_memory_mb=self._current_memory_mb,
                max_memory_mb=self._stats.max_memory_mb,
            )

    def get_cache_info(self) -> dict[str, Any]:
        """Get detailed cache information."""
        with self._lock:
            entries = []
            for key, entry in self._cache.items():
                entries.append(
                    {
                        "key": key,
                        "memory_mb": entry.memory_size_mb,
                        "access_count": entry.access_count,
                        "last_accessed": entry.last_accessed,
                        "age_seconds": time.time() - entry.created_at,
                    }
                )

            return {
                "total_entries": len(self._cache),
                "total_memory_mb": self._current_memory_mb,
                "max_size": self.max_size,
                "max_memory_mb": self.max_memory_mb,
                "entries": entries,
                "stats": self._stats.to_dict() if self._stats else None,
            }

    def __len__(self) -> int:
        """Get number of cached models."""
        with self._lock:
            return len(self._cache)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Context manager exit with cleanup."""
        self.clear()


@dataclass
class BatchRequest:
    """Represents a single request in a batch."""

    request_id: str
    image_data: bytes
    text_regions: list[Any] | None = None
    callback: Callable | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class BatchResult:
    """Result from batch processing."""

    request_id: str
    result: Any
    processing_time_ms: float
    success: bool
    error: str | None = None


class BatchProcessor:
    """Batch processing system for improved inference throughput."""

    def __init__(
        self,
        max_batch_size: int = 8,
        max_wait_time_ms: float = 100.0,
        enable_dynamic_batching: bool = True,
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_time_ms = max_wait_time_ms
        self.enable_dynamic_batching = enable_dynamic_batching

        self._pending_requests: list[BatchRequest] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._processing_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        logger.info(
            f"BatchProcessor initialized: max_batch_size={max_batch_size}, "
            f"max_wait_time={max_wait_time_ms}ms"
        )

    def start(self, processor_func: Callable[[list[BatchRequest]], list[BatchResult]]):
        """Start the batch processing thread.

        Args:
            processor_func: Function that processes a batch of requests
        """
        if self._processing_thread and self._processing_thread.is_alive():
            logger.warning("Batch processor already running")
            return

        self._processor_func = processor_func
        self._stop_event.clear()
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()

        logger.info("Batch processor started")

    def stop(self):
        """Stop the batch processing thread."""
        self._stop_event.set()

        with self._condition:
            self._condition.notify_all()

        if self._processing_thread:
            self._processing_thread.join(timeout=5.0)
            if self._processing_thread.is_alive():
                logger.warning("Batch processor thread did not stop gracefully")
            else:
                logger.info("Batch processor stopped")

    def submit_request(self, request: BatchRequest) -> "BatchFuture":
        """Submit a request for batch processing.

        Args:
            request: The request to process

        Returns:
            Future object to get the result
        """
        future = BatchFuture()
        request.callback = future._set_result

        with self._condition:
            self._pending_requests.append(request)
            self._condition.notify()

        return future

    def _processing_loop(self):
        """Main processing loop for batching."""
        logger.debug("Batch processing loop started")

        while not self._stop_event.is_set():
            batch_requests = []

            with self._condition:
                # Wait for requests or timeout
                if not self._pending_requests:
                    self._condition.wait(timeout=self.max_wait_time_ms / 1000.0)

                if self._stop_event.is_set():
                    break

                # Collect batch
                if self._pending_requests:
                    batch_size = min(len(self._pending_requests), self.max_batch_size)
                    batch_requests = self._pending_requests[:batch_size]
                    self._pending_requests = self._pending_requests[batch_size:]

            # Process batch if we have requests
            if batch_requests:
                try:
                    start_time = time.time()
                    results = self._processor_func(batch_requests)
                    processing_time = (time.time() - start_time) * 1000

                    logger.debug(
                        f"Processed batch of {len(batch_requests)} requests "
                        f"in {processing_time:.1f}ms"
                    )

                    # Send results to callbacks
                    for result in results:
                        # Find corresponding request and call its callback
                        for request in batch_requests:
                            if request.request_id == result.request_id and request.callback:
                                request.callback(result)
                                break

                except Exception as e:
                    logger.exception("Batch processing failed")
                    # Send error results
                    error_msg = str(e)
                    for request in batch_requests:
                        if request.callback:
                            error_result = BatchResult(
                                request_id=request.request_id,
                                result=None,
                                processing_time_ms=0.0,
                                success=False,
                                error=error_msg,
                            )
                            request.callback(error_result)

        logger.debug("Batch processing loop ended")


class BatchFuture:
    """Future object for batch processing results."""

    def __init__(self):
        self._result: BatchResult | None = None
        self._event = threading.Event()

    def _set_result(self, result: BatchResult):
        """Set the result (called by BatchProcessor)."""
        self._result = result
        self._event.set()

    def get(self, timeout: float | None = None) -> BatchResult:
        """Get the result, blocking until available.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            The batch result

        Raises:
            TimeoutError: If timeout is reached
        """
        if not self._event.wait(timeout=timeout):
            raise BatchProcessingTimeoutError()

        return self._result

    def is_done(self) -> bool:
        """Check if the result is ready."""
        return self._event.is_set()
