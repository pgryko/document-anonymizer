# Document Anonymization System

A comprehensive document anonymization system using NER (Named Entity Recognition) and diffusion models to automatically detect and anonymize PII (Personally Identifiable Information) in PDF documents.

## Features

### 🔍 Advanced Text Detection
- **Multi-engine OCR**: PaddleOCR, EasyOCR, TrOCR, Tesseract with automatic fallback
- **Accurate text extraction** from complex document layouts
- **Confidence-based filtering** for reliable text detection

### 🛡️ Intelligent PII Detection  
- **NER-based PII identification** using Presidio
- **Configurable entity types**: emails, phone numbers, SSNs, credit cards, addresses
- **Custom entity recognition** patterns and rules

### 🎨 High-Quality Anonymization
- **Diffusion model inpainting** using Stable Diffusion 2.0
- **Context-aware replacement** that preserves document aesthetics
- **Multiple anonymization strategies**: redaction, replacement, synthetic generation

### 📊 Performance Monitoring
- **Real-time resource monitoring** (CPU, memory, GPU)
- **Comprehensive benchmarking** suite
- **Performance regression detection**
- **Memory profiling** and optimization insights

### 🔧 Model Management
- **Automated model downloading** from Hugging Face Hub
- **Model validation** and integrity checking
- **Version tracking** and usage statistics
- **Storage optimization** with cleanup tools

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd document-anonymizer

# Install dependencies using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

### Basic Usage

```python
from src.anonymizer import DocumentAnonymizer, AnonymizationConfig

# Configure anonymization
config = AnonymizationConfig(
    ocr_engines=["paddleocr", "easyocr"],  # OCR fallback chain
    entity_types=["PERSON", "EMAIL", "PHONE_NUMBER"],  # PII types to detect
    anonymization_strategy="inpainting",  # Use diffusion models
    confidence_threshold=0.8  # High confidence filtering
)

# Initialize anonymizer
anonymizer = DocumentAnonymizer(config)

# Anonymize a document
result = anonymizer.anonymize_document("input.pdf", "output.pdf")

print(f"Anonymized {result.entities_found} PII entities")
print(f"Processing time: {result.processing_time_ms}ms")
```

### Command Line Interface

```bash
# Anonymize a single document
python -m src.anonymizer.cli anonymize input.pdf --output output.pdf

# Batch process multiple documents
python -m src.anonymizer.cli batch-anonymize documents/ --output anonymized/

# Download required models
python scripts/download_models.py ensure-models --use-case default

# Run performance benchmarks
python scripts/benchmark.py full-suite --quick
```

## Documentation Structure

- [`architecture.md`](architecture.md) - System architecture and design principles
- [`api-reference.md`](api-reference.md) - Complete API documentation
- [`configuration.md`](configuration.md) - Configuration options and settings
- [`examples/`](examples/) - Practical usage examples and tutorials
- [`performance.md`](performance.md) - Performance optimization and monitoring
- [`deployment.md`](deployment.md) - Production deployment guides
- [`troubleshooting.md`](troubleshooting.md) - Common issues and solutions

## Core Components

### 1. OCR Engine
Multi-engine text detection with automatic fallback:
- **PaddleOCR**: High accuracy for complex layouts
- **EasyOCR**: Good balance of speed and accuracy  
- **TrOCR**: Transformer-based recognition
- **Tesseract**: Reliable baseline OCR

### 2. NER Pipeline
PII detection using Presidio:
- Pre-trained models for common entity types
- Custom patterns for domain-specific PII
- Configurable confidence thresholds
- Multi-language support

### 3. Diffusion Inpainting
High-quality anonymization using Stable Diffusion:
- Context-aware background generation
- Preserves document aesthetics
- Batch processing support
- GPU acceleration

### 4. Performance Monitoring
Comprehensive performance tracking:
- Real-time resource monitoring
- Benchmark suite for regression testing
- Memory profiling and leak detection
- Performance optimization insights

## Examples

### Basic Document Processing

```python
from src.anonymizer import DocumentAnonymizer

# Simple anonymization
anonymizer = DocumentAnonymizer()
result = anonymizer.anonymize_document("document.pdf")

# Access results
print(f"Found entities: {[e.type for e in result.entities]}")
print(f"Anonymized regions: {len(result.anonymized_regions)}")
```

### Advanced Configuration

```python
from src.anonymizer import AnonymizationConfig, EntityType

config = AnonymizationConfig(
    # OCR Configuration
    ocr_engines=["paddleocr", "easyocr"],
    ocr_confidence_threshold=0.8,
    
    # NER Configuration  
    entity_types=[EntityType.PERSON, EntityType.EMAIL, EntityType.PHONE],
    ner_confidence_threshold=0.9,
    
    # Anonymization Configuration
    anonymization_strategy="inpainting",
    preserve_formatting=True,
    
    # Performance Configuration
    batch_size=4,
    use_gpu=True,
    memory_optimization=True
)

anonymizer = DocumentAnonymizer(config)
```

### Batch Processing

```python
from pathlib import Path
from src.anonymizer import BatchProcessor

processor = BatchProcessor(
    input_dir=Path("documents/"),
    output_dir=Path("anonymized/"),
    parallel_workers=4
)

# Process all PDFs in directory
results = processor.process_all()

# Generate processing report
report = processor.generate_report(results)
print(f"Processed {report.total_documents} documents")
print(f"Success rate: {report.success_rate:.1%}")
```

### Performance Monitoring

```python
from src.anonymizer.performance import PerformanceMonitor

monitor = PerformanceMonitor()

# Start monitoring session
monitor.start_session("document_processing")

# Your processing code here
result = anonymizer.anonymize_document("large_document.pdf")

# End session and get report
report = monitor.end_session()

print(f"Peak memory: {report.peak_memory_mb:.1f}MB")
print(f"Processing time: {report.duration_seconds:.1f}s")
```

## Model Management

### Download Models

```bash
# List available models
python scripts/download_models.py list-available

# Download specific model
python scripts/download_models.py download sd2-vae

# Download recommended model set
python scripts/download_models.py ensure-models --use-case quality

# Download from Hugging Face directly
python scripts/download_models.py download-hf stabilityai/stable-diffusion-2-inpainting
```

### Model Information

```bash
# Get model details
python scripts/download_models.py info sd2-vae

# Validate downloaded models
python scripts/download_models.py verify-all

# Check storage usage
python scripts/download_models.py storage-stats

# Clean up unused models
python scripts/download_models.py cleanup --unused-days 30
```

## Performance Benchmarking

### Run Benchmarks

```bash
# Quick benchmark suite
python scripts/benchmark.py full-suite --quick

# Individual component benchmarks
python scripts/benchmark.py text-detection --num-images 50
python scripts/benchmark.py pii-detection --num-texts 1000
python scripts/benchmark.py inpainting --image-size 512x512

# End-to-end pipeline benchmark
python scripts/benchmark.py end-to-end --num-documents 10
```

### Monitor Resources

```bash
# Real-time monitoring
python scripts/benchmark.py monitor --duration 60 --interval 1.0

# Analyze benchmark results
python scripts/benchmark.py analyze results/benchmark_20240101_120000.json
```

## Advanced Topics

### Custom OCR Engines

```python
from src.anonymizer.ocr import OCREngine, OCRResult

class CustomOCREngine(OCREngine):
    def detect_text(self, image: np.ndarray) -> OCRResult:
        # Implement custom OCR logic
        detected_texts = self._run_custom_ocr(image)
        return OCRResult(detected_texts=detected_texts)

# Register custom engine
from src.anonymizer.ocr import OCRProcessor
processor = OCRProcessor()
processor.register_engine("custom", CustomOCREngine())
```

### Custom Anonymization Strategies

```python
from src.anonymizer.core import AnonymizationStrategy

class BlurStrategy(AnonymizationStrategy):
    def anonymize_region(self, image: np.ndarray, region: BoundingBox) -> np.ndarray:
        # Apply blur to the region
        blurred = cv2.GaussianBlur(image[region.slice], (15, 15), 0)
        image[region.slice] = blurred
        return image

# Use custom strategy
config = AnonymizationConfig(anonymization_strategy=BlurStrategy())
```

### Performance Optimization

```python
# Memory optimization
config = AnonymizationConfig(
    memory_optimization=True,
    batch_size=2,  # Reduce for lower memory usage
    enable_caching=False  # Disable for memory-constrained environments
)

# GPU optimization
config = AnonymizationConfig(
    use_gpu=True,
    mixed_precision=True,  # Use FP16 for faster inference
    gpu_memory_limit="8GB"  # Limit GPU memory usage
)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite: `python -m pytest`
5. Submit a pull request

## License

This project is licensed under the MIT License. See [LICENSE](../LICENSE) for details.

## Support

- 📚 [Documentation](docs/)
- 🐛 [Issue Tracker](issues/)
- 💬 [Discussions](discussions/)
- 📧 [Email Support](mailto:support@example.com)