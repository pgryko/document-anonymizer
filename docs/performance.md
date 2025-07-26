# Performance Optimization Guide

Comprehensive guide to optimizing the Document Anonymization System for different performance requirements and resource constraints.

## Performance Overview

The system's performance depends on several factors:

1. **Hardware Resources**: CPU, GPU, memory availability
2. **Document Characteristics**: Size, complexity, number of entities
3. **Configuration Settings**: Batch sizes, model selection, optimization flags
4. **Workload Patterns**: Single document vs. batch processing

## Performance Monitoring

### Real-time Monitoring

```python
from src.anonymizer.performance import PerformanceMonitor

# Basic monitoring
monitor = PerformanceMonitor()
monitor.start_session("document_processing")

# Your processing code here
result = anonymizer.anonymize_document("document.pdf")

# Get performance report
report = monitor.end_session()
print(f"Peak memory: {report['resource_summary']['peak_memory_mb']:.1f}MB")
print(f"Average CPU: {report['resource_summary']['cpu_percent']['avg']:.1f}%")
print(f"Duration: {report['session_duration_seconds']:.2f}s")
```

### Continuous Monitoring

```python
from src.anonymizer.performance import PerformanceMonitor
import time

# Long-running monitoring
monitor = PerformanceMonitor(
    sample_interval=1.0,  # Sample every second
    auto_export=True,     # Auto-save results
    results_dir="monitoring/"
)

monitor.start_session("batch_processing")

# Process multiple documents
for doc in documents:
    with monitor.monitor_operation(f"process_{doc.name}"):
        result = anonymizer.anonymize_document(doc.path)

final_report = monitor.end_session()
```

### Performance Metrics

The system tracks various performance metrics:

#### Resource Metrics
- **Memory Usage**: Peak, average, current usage
- **CPU Utilization**: Per-core and overall usage
- **GPU Metrics**: Memory, utilization, temperature
- **I/O Metrics**: Disk reads/writes, network usage

#### Processing Metrics
- **Throughput**: Documents per minute/hour
- **Latency**: End-to-end processing time
- **Component Times**: OCR, NER, anonymization breakdown
- **Queue Metrics**: Batch processing statistics

#### Quality Metrics
- **Detection Accuracy**: PII detection rate
- **False Positive Rate**: Incorrect detections
- **Anonymization Quality**: Visual similarity scores
- **Error Rates**: Processing failures

## Performance Benchmarking

### Benchmark Suite

```bash
# Run complete benchmark suite
python scripts/benchmark.py full-suite

# Quick benchmark (reduced test sizes)
python scripts/benchmark.py full-suite --quick

# Individual component benchmarks
python scripts/benchmark.py text-detection --num-images 100
python scripts/benchmark.py pii-detection --num-texts 1000  
python scripts/benchmark.py inpainting --image-size 512x512
python scripts/benchmark.py end-to-end --num-documents 50
```

### Custom Benchmarks

```python
from src.anonymizer.performance import AnonymizationBenchmark

benchmark = AnonymizationBenchmark()

# OCR performance test
ocr_results = benchmark.benchmark_text_detection(
    image_sizes=[(512, 512), (1024, 768), (2048, 1536)],
    num_images_per_size=20,
    engines=["paddleocr", "easyocr", "tesseract"]
)

# NER performance test
ner_results = benchmark.benchmark_pii_detection(
    text_lengths=[100, 500, 1000, 5000],
    num_texts_per_length=50,
    entity_types=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
)

# End-to-end benchmark
e2e_results = benchmark.benchmark_end_to_end(
    document_types=["simple", "complex", "image_heavy"],
    num_documents_per_type=10,
    anonymization_strategies=["inpainting", "redaction"]
)
```

### Benchmark Analysis

```python
# Analyze benchmark results
from src.anonymizer.performance import BenchmarkAnalyzer

analyzer = BenchmarkAnalyzer()

# Load previous results for comparison
baseline = analyzer.load_benchmark("results/baseline_v1.0.json")
current = analyzer.load_benchmark("results/current_test.json")

# Performance regression analysis
regression_report = analyzer.compare_benchmarks(baseline, current)

if regression_report.has_regressions:
    print("Performance regressions detected:")
    for regression in regression_report.regressions:
        print(f"  {regression.component}: {regression.change_percent:.1f}% slower")
```

## Optimization Strategies

### 1. Hardware Optimization

#### GPU Optimization

```python
# Optimal GPU configuration
config = AnonymizationConfig(
    use_gpu=True,
    mixed_precision=True,  # Use FP16 for 2x speed improvement
    
    # Memory optimization
    enable_memory_efficient_attention=True,
    enable_vae_slicing=True,
    gpu_memory_fraction=0.8,  # Reserve 20% for system
    
    # Batch optimization
    batch_size=8,  # Adjust based on GPU memory
    gradient_accumulation_steps=1,
    
    # Compilation (PyTorch 2.0+)
    compile_models=True,
    compile_mode="max-autotune"
)
```

#### CPU Optimization

```python
# CPU-optimized configuration
config = AnonymizationConfig(
    use_gpu=False,
    
    # Threading optimization
    num_threads=8,  # Match CPU cores
    enable_mkldnn=True,  # Intel optimization
    
    # Memory settings
    batch_size=2,  # Smaller batches for CPU
    memory_optimization=True,
    pin_memory=False,  # Don't pin memory for CPU
    
    # Model selection
    use_lightweight_models=True,
    prefer_cpu_optimized_models=True
)
```

#### Memory Optimization

```python
# Memory-constrained environments
config = AnonymizationConfig(
    # Reduce batch sizes
    batch_size=1,
    
    # Model loading strategy
    load_models_on_demand=True,
    unload_models_after_use=True,
    model_cache_size_gb=2,
    
    # Processing optimization
    enable_gradient_checkpointing=True,
    use_cpu_offload=True,
    
    # Disable caching
    enable_caching=False,
    
    # Memory cleanup
    force_garbage_collection=True,
    cleanup_interval_documents=10
)
```

### 2. Algorithm Optimization

#### OCR Engine Selection

```python
# Performance comparison of OCR engines
ocr_performance = {
    "tesseract": {
        "speed": "fast",
        "accuracy": "good", 
        "memory": "low",
        "best_for": "simple documents, speed-critical applications"
    },
    "paddleocr": {
        "speed": "medium",
        "accuracy": "excellent",
        "memory": "medium",
        "best_for": "complex layouts, multi-language documents"
    },
    "easyocr": {
        "speed": "medium-fast",
        "accuracy": "very good",
        "memory": "medium",
        "best_for": "general purpose, good balance"
    },
    "trotr": {
        "speed": "slow",
        "accuracy": "excellent",
        "memory": "high",
        "best_for": "handwritten text, highest accuracy needs"
    }
}

# Speed-optimized OCR
speed_config = AnonymizationConfig(
    ocr_engines=["tesseract"],  # Fastest option
    ocr_confidence_threshold=0.6,  # Lower threshold for speed
    ocr_preprocessing=False  # Skip preprocessing
)

# Accuracy-optimized OCR  
accuracy_config = AnonymizationConfig(
    ocr_engines=["paddleocr", "easyocr", "trotr"],  # Multiple engines
    ocr_confidence_threshold=0.9,  # High confidence
    ocr_preprocessing=True,  # Enable preprocessing
    ocr_ensemble_voting=True  # Use ensemble results
)
```

#### Anonymization Strategy Selection

```python
# Strategy performance comparison
strategy_performance = {
    "redaction": {
        "speed": "very fast",
        "quality": "low",
        "resource_usage": "minimal",
        "use_case": "speed-critical, privacy compliance"
    },
    "blur": {
        "speed": "fast", 
        "quality": "medium",
        "resource_usage": "low",
        "use_case": "quick anonymization, moderate quality"
    },
    "inpainting": {
        "speed": "slow",
        "quality": "excellent",
        "resource_usage": "high",
        "use_case": "publication-ready documents"
    },
    "replacement": {
        "speed": "fast",
        "quality": "medium-high",
        "resource_usage": "low", 
        "use_case": "text-based anonymization"
    }
}

# Configure based on requirements
if requirements.speed_critical:
    config.anonymization_strategy = "redaction"
elif requirements.quality_critical:
    config.anonymization_strategy = "inpainting"
    config.diffusion_steps = 50
else:
    config.anonymization_strategy = "blur"
```

### 3. Batch Processing Optimization

#### Optimal Batch Sizes

```python
def find_optimal_batch_size(config: AnonymizationConfig) -> int:
    """Find optimal batch size for current hardware."""
    
    # Start with hardware-based estimate
    if config.use_gpu:
        # GPU memory based estimation
        gpu_memory_gb = get_gpu_memory_gb()
        base_batch_size = max(1, int(gpu_memory_gb / 2))  # Conservative estimate
    else:
        # CPU memory based estimation  
        cpu_memory_gb = get_system_memory_gb()
        base_batch_size = max(1, int(cpu_memory_gb / 4))
    
    # Test increasing batch sizes
    optimal_batch_size = base_batch_size
    best_throughput = 0
    
    for batch_size in [base_batch_size, base_batch_size * 2, base_batch_size * 4]:
        try:
            test_config = config.copy()
            test_config.batch_size = batch_size
            
            # Run small benchmark
            throughput = benchmark_batch_processing(test_config, num_documents=10)
            
            if throughput > best_throughput:
                best_throughput = throughput
                optimal_batch_size = batch_size
            else:
                break  # Performance degraded, stop testing
                
        except OutOfMemoryError:
            break  # Hit memory limit
    
    return optimal_batch_size
```

#### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

class OptimizedBatchProcessor:
    def __init__(self, config: AnonymizationConfig):
        self.config = config
        self.num_workers = self._determine_optimal_workers()
    
    def _determine_optimal_workers(self) -> int:
        """Determine optimal number of workers."""
        if self.config.use_gpu:
            # GPU processing: fewer workers to avoid contention
            return min(2, mp.cpu_count() // 2)
        else:
            # CPU processing: more workers
            return min(mp.cpu_count(), 8)  # Cap at 8 to avoid overhead
    
    def process_batch_parallel(self, documents: List[str]) -> List[AnonymizationResult]:
        """Process documents in parallel."""
        
        if self.config.use_gpu:
            # Use ThreadPoolExecutor for GPU (shared memory)
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._process_document, doc) 
                          for doc in documents]
                return [future.result() for future in futures]
        else:
            # Use ProcessPoolExecutor for CPU (parallel processing)
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._process_document, doc)
                          for doc in documents]
                return [future.result() for future in futures]
```

### 4. Model Optimization

#### Model Caching

```python
from src.anonymizer.models import ModelCache

# Intelligent model caching
cache = ModelCache(
    cache_size_gb=10,  # 10GB cache
    eviction_policy="lru",  # Least recently used
    preload_models=["sd2-vae", "sd2-unet"],  # Preload common models
    lazy_loading=True  # Load on demand
)

# Monitor cache performance
cache_stats = cache.get_statistics()
print(f"Cache hit rate: {cache_stats.hit_rate:.2%}")
print(f"Cache size: {cache_stats.size_gb:.1f}GB")
```

#### Model Quantization

```python
# Quantized models for faster inference
config = AnonymizationConfig(
    # Use quantized models
    use_quantized_models=True,
    quantization_method="int8",  # 8-bit quantization
    
    # Model optimization
    optimize_for_inference=True,
    enable_torch_script=True,
    
    # Memory settings
    use_fp16=True,  # Half precision
    enable_attention_slicing=True
)
```

#### Dynamic Model Loading

```python
class DynamicModelManager:
    def __init__(self):
        self.loaded_models = {}
        self.model_usage = {}
        
    def get_model(self, model_name: str, priority: str = "normal"):
        """Get model with dynamic loading based on usage patterns."""
        
        if model_name not in self.loaded_models:
            # Check if we need to free memory
            if self._memory_pressure():
                self._unload_least_used_models()
            
            # Load model
            self.loaded_models[model_name] = self._load_model(model_name)
        
        # Update usage statistics
        self.model_usage[model_name] = time.time()
        
        return self.loaded_models[model_name]
```

## Performance Patterns

### 1. Streaming Processing

```python
class StreamingProcessor:
    def __init__(self, config: AnonymizationConfig):
        self.config = config
        self.buffer_size = 10
        
    def process_stream(self, document_stream: Iterator[str]) -> Iterator[AnonymizationResult]:
        """Process documents in streaming fashion."""
        
        buffer = []
        for document in document_stream:
            buffer.append(document)
            
            if len(buffer) >= self.buffer_size:
                # Process batch
                results = self._process_batch(buffer)
                for result in results:
                    yield result
                buffer.clear()
        
        # Process remaining documents
        if buffer:
            results = self._process_batch(buffer)
            for result in results:
                yield result
```

### 2. Adaptive Processing

```python
class AdaptiveProcessor:
    def __init__(self, base_config: AnonymizationConfig):
        self.base_config = base_config
        self.performance_history = []
        
    def process_document(self, document_path: str) -> AnonymizationResult:
        """Process document with adaptive configuration."""
        
        # Analyze document characteristics
        doc_complexity = self._analyze_document_complexity(document_path)
        
        # Adapt configuration
        config = self._adapt_config(doc_complexity)
        
        # Process with monitoring
        start_time = time.time()
        result = self._process_with_config(document_path, config)
        processing_time = time.time() - start_time
        
        # Update performance history
        self.performance_history.append({
            "complexity": doc_complexity,
            "config": config,
            "processing_time": processing_time,
            "success": result.success
        })
        
        return result
    
    def _adapt_config(self, complexity: float) -> AnonymizationConfig:
        """Adapt configuration based on document complexity."""
        
        config = self.base_config.copy()
        
        if complexity > 0.8:  # High complexity
            config.ocr_engines = ["paddleocr", "easyocr"]
            config.batch_size = 2
            config.ocr_confidence_threshold = 0.8
            
        elif complexity < 0.3:  # Low complexity
            config.ocr_engines = ["tesseract"]
            config.batch_size = 8
            config.ocr_confidence_threshold = 0.6
            
        return config
```

### 3. Caching Strategies

```python
from functools import lru_cache
import hashlib

class IntelligentCache:
    def __init__(self, cache_size_gb: float = 5.0):
        self.cache_size_bytes = int(cache_size_gb * 1024**3)
        self.ocr_cache = {}
        self.ner_cache = {}
        self.model_cache = {}
        
    @lru_cache(maxsize=1000)
    def get_ocr_result(self, image_hash: str, engine: str) -> OCRResult:
        """Cache OCR results based on image content."""
        cache_key = f"{image_hash}_{engine}"
        
        if cache_key in self.ocr_cache:
            return self.ocr_cache[cache_key]
        
        # If not cached, process and cache
        result = self._process_ocr(image_hash, engine)
        self.ocr_cache[cache_key] = result
        return result
    
    def cache_model_result(self, input_hash: str, model_config: str, result: Any):
        """Cache model inference results."""
        cache_key = f"{input_hash}_{model_config}"
        
        # Check cache size limits
        if self._get_cache_size() > self.cache_size_bytes:
            self._evict_oldest_entries()
        
        self.model_cache[cache_key] = {
            "result": result,
            "timestamp": time.time(),
            "access_count": 0
        }
```

## Performance Troubleshooting

### Common Performance Issues

#### 1. High Memory Usage

**Symptoms:**
- OutOfMemoryError exceptions
- System becomes unresponsive
- GPU memory errors

**Solutions:**
```python
# Reduce batch size
config.batch_size = 1

# Enable memory optimization
config.memory_optimization = True
config.enable_cpu_offload = True

# Use gradient checkpointing
config.enable_gradient_checkpointing = True

# Clear cache more frequently
config.cache_cleanup_interval = 5
```

#### 2. Slow Processing Speed

**Symptoms:**
- High processing times per document
- Low throughput
- CPU/GPU underutilization

**Solutions:**
```python
# Optimize batch size
config.batch_size = find_optimal_batch_size(config)

# Use faster engines
config.ocr_engines = ["tesseract"]  # Fastest OCR
config.anonymization_strategy = "redaction"  # Fastest anonymization

# Enable optimizations
config.compile_models = True
config.use_fp16 = True
config.enable_caching = True
```

#### 3. Resource Contention

**Symptoms:**
- Inconsistent performance
- High CPU wait times
- Memory fragmentation

**Solutions:**
```python
# Limit concurrent processing
config.max_concurrent_documents = 2

# Use process isolation
config.use_process_pool = True
config.process_pool_size = 4

# Implement resource limits
config.cpu_limit_percent = 80
config.memory_limit_gb = 8
```

### Performance Profiling

```python
from src.anonymizer.performance import DetailedProfiler

# Enable detailed profiling
profiler = DetailedProfiler(
    profile_memory=True,
    profile_cpu=True,
    profile_gpu=True,
    trace_function_calls=True
)

with profiler.profile_session("detailed_analysis"):
    result = anonymizer.anonymize_document("complex_document.pdf")

# Analyze results
report = profiler.generate_report()

print("Top memory consumers:")
for func, memory_mb in report.memory_hotspots[:5]:
    print(f"  {func}: {memory_mb:.1f}MB")

print("Top CPU consumers:")
for func, cpu_percent in report.cpu_hotspots[:5]:
    print(f"  {func}: {cpu_percent:.1f}%")

# Generate flame graph
profiler.generate_flame_graph("profile_output/")
```

### Performance Testing

```python
import pytest
from src.anonymizer.performance import PerformanceTest

class TestPerformanceRegression:
    def test_processing_speed_regression(self):
        """Test that processing speed hasn't regressed."""
        
        benchmark = PerformanceTest()
        
        # Load baseline performance
        baseline = benchmark.load_baseline("v1.0.0")
        
        # Run current performance test
        current = benchmark.run_standard_benchmark()
        
        # Check for regressions (allow 5% variance)
        speed_regression = (current.avg_processing_time / baseline.avg_processing_time - 1) * 100
        
        assert speed_regression < 5, f"Processing speed regressed by {speed_regression:.1f}%"
    
    def test_memory_usage_regression(self):
        """Test that memory usage hasn't regressed."""
        
        benchmark = PerformanceTest()
        baseline = benchmark.load_baseline("v1.0.0")
        current = benchmark.run_memory_benchmark()
        
        memory_regression = (current.peak_memory_mb / baseline.peak_memory_mb - 1) * 100
        
        assert memory_regression < 10, f"Memory usage regressed by {memory_regression:.1f}%"
    
    @pytest.mark.slow
    def test_throughput_benchmark(self):
        """Test overall system throughput."""
        
        benchmark = PerformanceTest()
        throughput = benchmark.measure_throughput(
            num_documents=100,
            duration_minutes=10
        )
        
        # Minimum acceptable throughput
        min_throughput = 10  # documents per minute
        assert throughput >= min_throughput, f"Throughput too low: {throughput:.1f} docs/min"
```

## Performance Best Practices

### 1. Configuration Tuning

- **Start with defaults** and measure baseline performance
- **Profile before optimizing** to identify actual bottlenecks
- **Test incrementally** - change one parameter at a time
- **Monitor continuously** in production environments

### 2. Resource Management

- **Monitor resource usage** patterns over time
- **Set appropriate limits** to prevent resource exhaustion
- **Use monitoring alerts** for performance degradation
- **Plan capacity** based on actual usage patterns

### 3. Optimization Workflow

1. **Measure baseline performance**
2. **Identify bottlenecks** using profiling
3. **Apply targeted optimizations**
4. **Validate improvements** with benchmarks
5. **Monitor production impact**
6. **Document changes** and their effects

### 4. Production Considerations

- **Load testing** before deployment
- **Gradual rollout** of performance changes
- **Rollback plans** for performance regressions
- **Performance SLAs** and monitoring
- **Regular performance reviews** and optimization cycles