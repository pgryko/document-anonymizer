# API Reference

Complete API documentation for the Document Anonymization System.

## Core Classes

### DocumentAnonymizer

Main interface for document anonymization.

```python
class DocumentAnonymizer:
    def __init__(self, config: Optional[AnonymizationConfig] = None)
```

**Parameters:**
- `config`: Configuration object. If None, uses default configuration.

#### Methods

##### `anonymize_document(input_path, output_path=None, **kwargs) -> AnonymizationResult`

Anonymizes a single document.

**Parameters:**
- `input_path` (str | Path): Path to input PDF file
- `output_path` (str | Path, optional): Path for output PDF. If None, generates automatic name
- `entity_types` (List[str], optional): Override default entity types to detect
- `confidence_threshold` (float, optional): Override confidence threshold
- `preserve_metadata` (bool, optional): Whether to preserve original metadata

**Returns:**
- `AnonymizationResult`: Result object containing processing details

**Example:**
```python
anonymizer = DocumentAnonymizer()
result = anonymizer.anonymize_document(
    "sensitive_doc.pdf",
    "anonymized_doc.pdf",
    entity_types=["PERSON", "EMAIL", "PHONE_NUMBER"],
    confidence_threshold=0.9
)
print(f"Anonymized {result.entities_found} PII entities")
```

##### `anonymize_batch(input_paths, output_dir=None, **kwargs) -> List[AnonymizationResult]`

Batch anonymization of multiple documents.

**Parameters:**
- `input_paths` (List[str | Path]): List of input PDF paths
- `output_dir` (str | Path, optional): Output directory for anonymized files
- `parallel_workers` (int, optional): Number of parallel workers
- `progress_callback` (Callable, optional): Progress update callback

**Returns:**
- `List[AnonymizationResult]`: Results for each processed document

**Example:**
```python
from pathlib import Path

input_files = list(Path("documents/").glob("*.pdf"))
results = anonymizer.anonymize_batch(
    input_files,
    output_dir="anonymized/",
    parallel_workers=4,
    progress_callback=lambda done, total: print(f"{done}/{total} complete")
)
```

---

### AnonymizationConfig

Configuration class for customizing anonymization behavior.

```python
@dataclass
class AnonymizationConfig:
    # OCR Configuration
    ocr_engines: List[str] = field(default_factory=lambda: ["paddleocr", "easyocr"])
    ocr_confidence_threshold: float = 0.7
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    
    # NER Configuration
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE"
    ])
    ner_confidence_threshold: float = 0.8
    custom_patterns: Optional[Dict[str, str]] = None
    
    # Anonymization Configuration
    anonymization_strategy: str = "inpainting"
    preserve_formatting: bool = True
    background_generation: bool = True
    
    # Performance Configuration
    use_gpu: bool = True
    batch_size: int = 4
    memory_optimization: bool = True
    enable_caching: bool = True
```

**Example:**
```python
config = AnonymizationConfig(
    ocr_engines=["paddleocr"],  # Use only PaddleOCR
    entity_types=["PERSON", "EMAIL_ADDRESS"],  # Detect only names and emails
    anonymization_strategy="redaction",  # Simple redaction instead of inpainting
    use_gpu=False,  # CPU-only processing
    batch_size=1  # Process one document at a time
)

anonymizer = DocumentAnonymizer(config)
```

---

### AnonymizationResult

Result object containing processing details and statistics.

```python
@dataclass
class AnonymizationResult:
    input_path: Path
    output_path: Path
    success: bool
    processing_time_ms: float
    entities_found: int
    entities_anonymized: int
    confidence_scores: List[float]
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
```

**Properties:**
- `success_rate`: Percentage of successfully anonymized entities
- `average_confidence`: Average confidence score of detected entities
- `processing_speed`: Documents per second

**Example:**
```python
result = anonymizer.anonymize_document("doc.pdf")

if result.success:
    print(f"✅ Success! Anonymized {result.entities_anonymized}/{result.entities_found} entities")
    print(f"⏱️ Processing time: {result.processing_time_ms:.1f}ms")
    print(f"📊 Average confidence: {result.average_confidence:.2f}")
else:
    print(f"❌ Failed: {result.error_message}")
```

---

## OCR Module (`src.anonymizer.ocr`)

### OCRProcessor

Multi-engine OCR processor with fallback capabilities.

```python
class OCRProcessor:
    def __init__(self, config: OCRConfig)
    
    def detect_text(self, image: np.ndarray) -> OCRResult
    def detect_text_regions(self, image: np.ndarray) -> List[TextRegion]
    def set_engines(self, engines: List[str]) -> None
    def add_custom_engine(self, name: str, engine: OCREngine) -> None
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

## NER Module (`src.anonymizer.ner`)

### NERPipeline

Named Entity Recognition pipeline for PII detection.

```python
class NERPipeline:
    def __init__(self, config: NERConfig)
    
    def detect_entities(self, texts: List[DetectedText]) -> List[PIIEntity]
    def analyze_text(self, text: str) -> List[PIIEntity]
    def add_custom_recognizer(self, recognizer: EntityRecognizer) -> None
```

**Example:**
```python
from src.anonymizer.ner import NERPipeline, NERConfig

config = NERConfig(
    entity_types=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    confidence_threshold=0.9,
    language="en"
)

ner = NERPipeline(config)
entities = ner.detect_entities(detected_texts)

for entity in entities:
    print(f"Found {entity.entity_type}: '{entity.text}' "
          f"(confidence: {entity.confidence:.2f})")
```

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

**Example:**
```python
from src.anonymizer.performance import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_session("batch_processing")

# Your processing code here
for doc in documents:
    result = anonymizer.anonymize_document(doc)

report = monitor.end_session()
print(f"Peak memory: {report['resource_summary']['peak_memory_mb']:.1f}MB")
print(f"Average CPU: {report['resource_summary']['cpu_percent']['avg']:.1f}%")
```

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