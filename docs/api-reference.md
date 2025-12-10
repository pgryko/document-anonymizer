# API Reference

Complete API documentation for the Document Anonymization System.

## Core Classes

### InferenceEngine

Main interface for image anonymization.

```python
from src.anonymizer.core.config import AppConfig
from src.anonymizer.inference.engine import InferenceEngine

app_config = AppConfig.from_env_and_yaml(yaml_path="configs/inference/app_config.yaml")
engine = InferenceEngine(app_config.engine)

result = engine.anonymize(image_data: bytes, text_regions: list[TextRegion] | None = None)
```

**Parameters:**
- `image_data` (bytes): PNG/JPEG/TIFF bytes
- `text_regions` (optional): Pre-specified regions; if omitted, OCR+NER is used

**Returns:**
- `AnonymizationResult`: Image array, patches, timing, success flag, errors

**Example:**
```python
from pathlib import Path
import numpy as np
from PIL import Image

img = Path("input.png").read_bytes()
result = engine.anonymize(img)
Image.fromarray(result.anonymized_image.astype(np.uint8)).save("output.png")
```

Note: Batch processing is exposed via the CLI (`python main.py batch-anonymize ...`).

---

### EngineConfig (via `AppConfig`)

Configuration for `InferenceEngine` loaded from env and optional YAML.

```python
from src.anonymizer.core.config import EngineConfig

config = EngineConfig(
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=1.0,
    enable_memory_efficient_attention=True,
    enable_sequential_cpu_offload=False,
    max_batch_size=4,
)
```

Engine config is normally provided through `AppConfig.from_env_and_yaml(...)`.

---

### AnonymizationResult

Result object containing processing details and statistics.

See `src/anonymizer/core/models.py: AnonymizationResult` for the canonical structure used by the engine.

**Properties:**
- `success_rate`: Percentage of successfully anonymized entities
- `average_confidence`: Average confidence score of detected entities
- `processing_speed`: Documents per second

Example usage is shown above in the `InferenceEngine` section.

---

## OCR Module (`src.anonymizer.ocr`)

### OCRProcessor

Multi-engine OCR processor with fallback capabilities.

```python
class OCRProcessor:
    def __init__(self, config: OCRConfig)
    def initialize(self) -> bool
    def extract_text_regions(self, image: np.ndarray) -> list[DetectedText]
    def register_engine(self, name: str, engine: OCREngine) -> None
```

**Example:**
```python
from src.anonymizer.ocr import OCRProcessor, OCRConfig

config = OCRConfig(
    engines=["paddleocr", "easyocr", "tesseract"],
    confidence_threshold=0.8,
    languages=["en", "es"]
)

processor = OCRProcessor(config)
ocr_result = processor.detect_text(image_array)

print(f"Detected {len(ocr_result.detected_texts)} text regions")
for text in ocr_result.high_confidence_texts(0.9):
    print(f"Text: '{text.content}' at {text.bbox} (confidence: {text.confidence:.2f})")
```

### OCRResult

Container for OCR detection results.

```python
@dataclass
class OCRResult:
    detected_texts: List[DetectedText]
    processing_time_ms: float
    engine_used: str

    def high_confidence_texts(self, threshold: float = 0.8) -> List[DetectedText]
    def filter_by_area(self, min_area: int) -> List[DetectedText]
    def get_text_content(self) -> str
```

### DetectedText

Individual text detection result.

```python
@dataclass
class DetectedText:
    content: str
    bbox: BoundingBox
    confidence: float
    language: Optional[str] = None
    font_info: Optional[FontInfo] = None
```

---

## NER Integration

NER uses Presidio via `NERProcessor` within `InferenceEngine` and does not expose a separate public API here.

### PIIEntity

Detected PII entity information.

```python
@dataclass
class PIIEntity:
    entity_type: str
    text: str
    start: int
    end: int
    confidence: float
    bbox: BoundingBox
    anonymization_strategy: Optional[str] = None
```

---

## Performance Module (`src.anonymizer.performance`)

### PerformanceMonitor

Real-time performance monitoring and profiling.

```python
class PerformanceMonitor:
    def __init__(self, results_dir: Optional[Path] = None, auto_export: bool = True)

    def start_session(self, session_name: str) -> None
    def end_session(self) -> Dict[str, Any]
    def get_current_usage(self) -> Optional[ResourceSample]
    def generate_performance_report(self) -> Dict[str, Any]
```

See `docs/README.md` for a current monitoring example using `InferenceEngine`.

### AnonymizationBenchmark

Standardized benchmarking suite.

```python
class AnonymizationBenchmark:
    def __init__(self, profiler: Optional[PerformanceProfiler] = None)

    def benchmark_document_loading(self, **kwargs) -> BenchmarkResult
    def benchmark_text_detection(self, **kwargs) -> BenchmarkResult
    def benchmark_pii_detection(self, **kwargs) -> BenchmarkResult
    def benchmark_inpainting(self, **kwargs) -> BenchmarkResult
    def benchmark_end_to_end(self, **kwargs) -> BenchmarkResult
    def run_full_benchmark_suite(self) -> List[BenchmarkResult]
```

**Example:**
```python
from src.anonymizer.performance import AnonymizationBenchmark

benchmark = AnonymizationBenchmark()

# Run individual benchmarks
ocr_result = benchmark.benchmark_text_detection(
    image_size=(1024, 768),
    num_images=50
)
print(f"OCR benchmark: {ocr_result.duration_ms:.1f}ms average")

# Run full suite
results = benchmark.run_full_benchmark_suite()
for result in results:
    print(f"{result.benchmark_name}: {result.duration_ms:.1f}ms")
```

---

## Model Management (`src.anonymizer.models`)

### ModelManager

High-level interface for model lifecycle management.

```python
class ModelManager:
    def __init__(self, config: Optional[ModelConfig] = None)

    def download_model(self, model_name: str, **kwargs) -> ModelMetadata
    def list_available_models(self, model_type: Optional[ModelType] = None) -> List[ModelSource]
    def list_downloaded_models(self) -> List[ModelMetadata]
    def ensure_models_available(self, use_case: str = "default") -> bool
    def cleanup_models(self, unused_days: int = 30, dry_run: bool = True) -> Dict[str, Any]
    def get_storage_stats(self) -> Dict[str, Any]
```

**Example:**
```python
from src.anonymizer.models import ModelManager

manager = ModelManager()

# Download required models
manager.download_model("sd2-vae")
manager.download_model("sd2-inpainting")

# Ensure all models for quality use case are available
success = manager.ensure_models_available("quality")
if success:
    print("✅ All quality models available")

# Check storage usage
stats = manager.get_storage_stats()
print(f"Total models: {stats['total_models']}")
print(f"Storage used: {stats['total_size_gb']:.2f} GB")
```

---

## Utilities

### BoundingBox

Geometric bounding box representation.

```python
@dataclass
class BoundingBox:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int
    @property
    def height(self) -> int
    @property
    def area(self) -> int
    @property
    def center(self) -> Tuple[int, int]

    def scale(self, factor: float) -> "BoundingBox"
    def expand(self, pixels: int) -> "BoundingBox"
    def intersects(self, other: "BoundingBox") -> bool
    def intersection(self, other: "BoundingBox") -> Optional["BoundingBox"]
```

**Example:**
```python
bbox = BoundingBox(left=100, top=50, right=200, bottom=100)
print(f"Area: {bbox.area} pixels")
print(f"Center: {bbox.center}")

# Scale up by 1.5x
scaled = bbox.scale(1.5)

# Expand by 10 pixels in all directions
expanded = bbox.expand(10)
```

### ImageProcessor

Image processing utilities.

```python
class ImageProcessor:
    @staticmethod
    def preprocess_for_ocr(image: np.ndarray) -> np.ndarray
    @staticmethod
    def enhance_contrast(image: np.ndarray, factor: float = 1.2) -> np.ndarray
    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray, max_size: int) -> np.ndarray
    @staticmethod
    def create_mask_from_regions(image_shape: Tuple[int, int], regions: List[BoundingBox]) -> np.ndarray
```

---

## Error Handling

### Custom Exceptions

```python
class AnonymizationError(Exception):
    """Base exception for anonymization errors"""
    pass

class OCRError(AnonymizationError):
    """OCR processing failed"""
    pass

class NERError(AnonymizationError):
    """Named entity recognition failed"""
    pass

class InferenceError(AnonymizationError):
    """Model inference failed"""
    pass

class ValidationError(AnonymizationError):
    """Input validation failed"""
    pass

class ConfigurationError(AnonymizationError):
    """Configuration error"""
    pass
```

**Example Error Handling:**
```python
from src.anonymizer import DocumentAnonymizer, OCRError, NERError

try:
    result = anonymizer.anonymize_document("document.pdf")
except OCRError as e:
    print(f"OCR failed: {e}")
    # Fallback to simple redaction
    result = anonymizer.anonymize_document(
        "document.pdf",
        anonymization_strategy="redaction"
    )
except NERError as e:
    print(f"NER failed: {e}")
    # Use manual entity specification
    result = anonymizer.anonymize_document(
        "document.pdf",
        manual_entities=[BoundingBox(100, 100, 200, 120)]
    )
except Exception as e:
    print(f"Unexpected error: {e}")
    # Log error and continue processing
    logger.error(f"Failed to process document: {e}")
```

---

## Configuration Examples

### Basic Configuration

```python
from src.anonymizer import AnonymizationConfig

# Minimal configuration for quick testing
config = AnonymizationConfig(
    ocr_engines=["tesseract"],  # Fast OCR
    entity_types=["PERSON"],    # Only detect names
    use_gpu=False,              # CPU only
    batch_size=1                # One at a time
)
```

### High-Performance Configuration

```python
# Optimized for speed and throughput
config = AnonymizationConfig(
    ocr_engines=["paddleocr"],           # Single high-quality engine
    ocr_confidence_threshold=0.7,        # Lower threshold for speed
    entity_types=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    ner_confidence_threshold=0.8,
    anonymization_strategy="inpainting",
    use_gpu=True,                        # GPU acceleration
    batch_size=8,                        # Large batches
    memory_optimization=True,            # Optimize memory usage
    enable_caching=True                  # Cache models
)
```

### High-Accuracy Configuration

```python
# Optimized for maximum accuracy
config = AnonymizationConfig(
    ocr_engines=["paddleocr", "easyocr", "trotr"],  # Multiple engines
    ocr_confidence_threshold=0.9,                    # High confidence
    entity_types=[  # Comprehensive entity detection
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER",
        "CREDIT_CARD", "IBAN_CODE", "US_SSN",
        "LOCATION", "DATE_TIME"
    ],
    ner_confidence_threshold=0.95,                   # Very high confidence
    anonymization_strategy="inpainting",
    preserve_formatting=True,
    background_generation=True,
    use_gpu=True,
    batch_size=4
)
```

### Production Configuration

```python
# Production-ready configuration
config = AnonymizationConfig(
    # Balanced OCR setup
    ocr_engines=["paddleocr", "easyocr"],
    ocr_confidence_threshold=0.8,

    # Standard PII types
    entity_types=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
    ner_confidence_threshold=0.85,

    # High-quality anonymization
    anonymization_strategy="inpainting",
    preserve_formatting=True,

    # Performance optimization
    use_gpu=True,
    batch_size=6,
    memory_optimization=True,
    enable_caching=True,

    # Monitoring
    enable_performance_monitoring=True,
    enable_audit_logging=True
)
```
