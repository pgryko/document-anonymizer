# System Architecture

This document describes the architecture and design principles of the Document Anonymization System.

## Overview

The system is designed as a modular, extensible pipeline that processes documents through multiple stages to detect and anonymize PII (Personally Identifiable Information).

## Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input PDF     │───▶│  Document       │───▶│   OCR Engine    │
│                 │    │  Preprocessor   │    │   (Multi-engine)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Anonymized     │◀───│  Diffusion      │◀───│  NER Pipeline   │
│     PDF         │    │  Inpainting     │    │ (PII Detection) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │ Performance     │
                       │ Monitoring      │
                       └─────────────────┘
```

## Core Components

### 1. Document Processing Pipeline (`src/anonymizer/core/`)

**Purpose**: Orchestrates the entire anonymization workflow

**Key Components**:
- `InferenceEngine`: Main pipeline coordinator
- `DocumentProcessor`: Handles PDF loading and conversion
- `ImageProcessor`: Image preprocessing and optimization

**Design Patterns**:
- **Pipeline Pattern**: Sequential processing stages
- **Strategy Pattern**: Pluggable anonymization strategies
- **Observer Pattern**: Progress monitoring and callbacks

```python
from src.anonymizer.core.config import AppConfig
from src.anonymizer.inference.engine import InferenceEngine

app_config = AppConfig.from_env_and_yaml()
engine = InferenceEngine(app_config.engine)

# Pipeline inside engine: Input bytes → preprocess → OCR → NER → bbox fusion → inpaint → result
result = engine.anonymize(image_bytes)
```

### 2. OCR Subsystem (`src/anonymizer/ocr/`)

**Purpose**: Multi-engine text detection with fallback mechanisms

**Architecture**:
- **Factory Pattern**: Engine creation and selection
- **Chain of Responsibility**: Fallback between engines
- **Adapter Pattern**: Unified interface for different OCR libraries

```python
# Engine abstraction
class OCREngine(ABC):
    @abstractmethod
    def detect_text(self, image: np.ndarray) -> OCRResult:
        pass

# Multi-engine processor with fallback
class OCRProcessor:
    def __init__(self, engines: List[str]):
        self.engines = [self.engine_factory.create(name) for name in engines]

    def detect_text(self, image: np.ndarray) -> OCRResult:
        for engine in self.engines:
            try:
                result = engine.detect_text(image)
                if result.confidence > self.threshold:
                    return result
            except Exception as e:
                logger.warning(f"Engine {engine} failed: {e}")
        raise OCRError("All engines failed")
```

**Supported Engines**:
- **PaddleOCR**: High accuracy, good for complex layouts
- **EasyOCR**: Fast, good general purpose
- **TrOCR**: Transformer-based, excellent for handwriting
- **Tesseract**: Reliable baseline, wide format support

### 3. NER Pipeline (`src/anonymizer/ner/`)

**Purpose**: Intelligent PII detection using transformer models

**Components**:
- **EntityDetector**: Core NER using Presidio
- **ContextAnalyzer**: Improves accuracy with document context
- **CustomPatterns**: Domain-specific PII patterns

```python
class NERPipeline:
    def __init__(self, config: NERConfig):
        self.analyzer = AnalyzerEngine()
        self.entity_types = config.entity_types
        self.confidence_threshold = config.confidence_threshold

    def detect_entities(self, texts: List[DetectedText]) -> List[PIIEntity]:
        entities = []
        for text in texts:
            results = self.analyzer.analyze(
                text=text.content,
                entities=self.entity_types,
                language='en'
            )
            entities.extend(self._convert_results(results, text.bbox))
        return self._filter_by_confidence(entities)
```

### 4. Diffusion Anonymization (`src/anonymizer/diffusion/`)

**Purpose**: High-quality inpainting using Stable Diffusion models

**Architecture**:
- **Model Manager**: Handles model loading and caching
- **Inpainting Pipeline**: Coordinates diffusion process
- **Memory Optimizer**: Manages GPU memory efficiently

```python
class DiffusionAnonymizer:
    def __init__(self, config: DiffusionConfig):
        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            config.model_path,
            torch_dtype=torch.float16 if config.use_fp16 else torch.float32
        )
        self.device = "cuda" if config.use_gpu else "cpu"

    def anonymize_regions(self, image: Image, masks: List[Mask]) -> Image:
        # Combine masks and run inpainting
        combined_mask = self._combine_masks(masks)
        prompt = self._generate_prompt(image, masks)

        result = self.pipeline(
            prompt=prompt,
            image=image,
            mask_image=combined_mask,
            num_inference_steps=config.num_steps,
            guidance_scale=config.guidance_scale
        )
        return result.images[0]
```

### 5. Performance Monitoring (`src/anonymizer/performance/`)

**Purpose**: Comprehensive performance tracking and optimization

**Components**:
- **ResourceMonitor**: Real-time system monitoring
- **PerformanceProfiler**: Operation timing and memory tracking
- **BenchmarkSuite**: Standardized performance tests

```python
class PerformanceMonitor:
    def __init__(self):
        self.profiler = PerformanceProfiler()
        self.resource_monitor = ResourceMonitor()

    @contextmanager
    def monitor_operation(self, operation_name: str):
        self.resource_monitor.start_monitoring()
        with self.profiler.profile_operation(operation_name):
            yield
        samples = self.resource_monitor.stop_monitoring()
        self._analyze_performance(operation_name, samples)
```

### 6. Model Management (`src/anonymizer/models/`)

**Purpose**: Automated model lifecycle management

**Features**:
- **Automatic Downloads**: From Hugging Face Hub
- **Integrity Verification**: Checksum validation
- **Version Tracking**: Model registry with metadata
- **Storage Optimization**: Cleanup and deduplication

```python
class ModelManager:
    def __init__(self, config: ModelConfig):
        self.downloader = ModelDownloader(config)
        self.validator = ModelValidator()
        self.registry = ModelRegistry()

    def ensure_model_available(self, model_name: str) -> ModelMetadata:
        if not self.registry.is_downloaded(model_name):
            source = self.registry.get_model_source(model_name)
            metadata = self.downloader.download(source)
            self.validator.validate(metadata)
            self.registry.register_metadata(metadata)
        return self.registry.get_metadata(model_name)
```

## Design Principles

### 1. Modularity
- **Separation of Concerns**: Each component has a single responsibility
- **Plugin Architecture**: Easy to add new OCR engines, anonymization strategies
- **Interface-based Design**: Abstract interfaces enable testing and flexibility

### 2. Extensibility
- **Strategy Pattern**: Pluggable algorithms (OCR engines, anonymization methods)
- **Factory Pattern**: Easy creation of new component instances
- **Configuration-driven**: Behavior controlled through configuration files

### 3. Performance
- **Lazy Loading**: Models loaded only when needed
- **Memory Management**: Efficient memory usage with cleanup
- **GPU Optimization**: CUDA acceleration for diffusion models
- **Batch Processing**: Efficient handling of multiple documents

### 4. Reliability
- **Error Handling**: Graceful degradation with fallback mechanisms
- **Logging**: Comprehensive logging for debugging and monitoring
- **Validation**: Input validation and model integrity checks
- **Testing**: Comprehensive test suite with performance regression tests

### 5. Observability
- **Performance Monitoring**: Real-time resource tracking
- **Metrics Collection**: Detailed performance and accuracy metrics
- **Profiling**: Memory and CPU profiling capabilities
- **Benchmarking**: Standardized performance testing

## Data Flow

### 1. Document Ingestion
```
PDF → PyMuPDF → Images → Preprocessing → OCR Ready
```

### 2. Text Detection
```
Image → OCR Engine → Text Regions → Confidence Filtering → Validated Text
```

### 3. PII Detection
```
Text → NER Analysis → Entity Extraction → Confidence Scoring → PII Entities
```

### 4. Anonymization
```
Image + PII Regions → Mask Generation → Diffusion Inpainting → Anonymized Image
```

### 5. Output Generation
```
Anonymized Images → PDF Assembly → Metadata Annotation → Final PDF
```

## Configuration System

### Hierarchical Configuration
```python
from src.anonymizer.core.config import AppConfig, EngineConfig

app = AppConfig(
    engine=EngineConfig(num_inference_steps=50, guidance_scale=7.5, strength=1.0)
)
```

### Environment-based Overrides
```python
# Development
ANONYMIZER_USE_GPU=false
ANONYMIZER_LOG_LEVEL=DEBUG

# Production
ANONYMIZER_USE_GPU=true
ANONYMIZER_BATCH_SIZE=8
ANONYMIZER_MODEL_CACHE_SIZE=10GB
```

## Security Considerations

### 1. Data Privacy
- **In-memory Processing**: Minimize disk writes of sensitive data
- **Secure Cleanup**: Overwrite memory containing PII data
- **Audit Logging**: Track all PII detection and anonymization events

### 2. Model Security
- **Checksum Verification**: Validate model integrity
- **Trusted Sources**: Only download from verified repositories
- **Sandboxing**: Isolate model execution environment

### 3. Access Control
- **API Authentication**: Secure access to anonymization services
- **Input Validation**: Prevent injection attacks
- **Rate Limiting**: Prevent resource exhaustion

## Scalability

### 1. Horizontal Scaling
- **Stateless Design**: No shared state between processing instances
- **Queue-based Processing**: Async processing with message queues
- **Load Balancing**: Distribute requests across multiple instances

### 2. Vertical Scaling
- **GPU Acceleration**: Utilize multiple GPUs for parallel processing
- **Memory Optimization**: Efficient memory usage patterns
- **Batch Processing**: Process multiple documents together

### 3. Cloud Deployment
- **Container Support**: Docker images for easy deployment
- **Kubernetes**: Orchestration and auto-scaling
- **Serverless**: AWS Lambda, Google Cloud Functions support

## Testing Strategy

### 1. Unit Tests
- **Component Isolation**: Test individual components
- **Mock Dependencies**: Use mocks for external services
- **Property-based Testing**: Generate test cases automatically

### 2. Integration Tests
- **End-to-end Workflows**: Test complete pipelines
- **Real Data Testing**: Use anonymized real documents
- **Performance Regression**: Detect performance degradation

### 3. Performance Tests
- **Benchmark Suite**: Standardized performance tests
- **Load Testing**: Test under high load conditions
- **Memory Profiling**: Detect memory leaks and optimization opportunities

## Monitoring and Observability

### 1. Metrics
- **Processing Metrics**: Throughput, latency, error rates
- **Resource Metrics**: CPU, memory, GPU usage
- **Business Metrics**: PII detection accuracy, anonymization quality

### 2. Logging
- **Structured Logging**: JSON format for machine parsing
- **Log Levels**: DEBUG, INFO, WARN, ERROR with appropriate usage
- **Correlation IDs**: Track requests across components

### 3. Tracing
- **Distributed Tracing**: Track requests across microservices
- **Performance Profiling**: Identify bottlenecks
- **Error Tracking**: Centralized error collection and analysis
