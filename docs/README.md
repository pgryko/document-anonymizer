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
from pathlib import Path
import numpy as np
from PIL import Image

from src.anonymizer.core.config import AppConfig
from src.anonymizer.inference.engine import InferenceEngine

# Load config (env + optional YAML overrides)
app_config = AppConfig.from_env_and_yaml(yaml_path="configs/inference/app_config.yaml")

# Initialize inference engine
engine = InferenceEngine(app_config.engine)

# Read input image and anonymize
image_bytes = Path("input.png").read_bytes()
result = engine.anonymize(image_bytes)

if result.success:
    Image.fromarray(result.anonymized_image.astype(np.uint8)).save("output.png")
    print("Saved anonymized image to output.png")
else:
    print(f"Anonymization completed with errors: {', '.join(result.errors)}")
```

### Command Line Interface

```bash
# Anonymize a single image
python main.py anonymize -c configs/inference/app_config.yaml -i input.png -o output.png

# Batch process images in a directory (preserving structure)
python main.py batch-anonymize -i data/raw -o data/anonymized -c configs/inference/app_config.yaml

# Train models
python main.py train-vae -c configs/training/vae_config_local.yaml
python main.py train-unet -c configs/training/unet_config_local.yaml

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
PII detection using Presidio integrated in `NERProcessor` inside `InferenceEngine`:
- Pre-trained recognizers for common entity types
- Custom patterns possible via Presidio configuration
- Confidence thresholding applied during OCR+NER fusion

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

### Basic Image Anonymization

```python
from pathlib import Path
import numpy as np
from PIL import Image

from src.anonymizer.core.config import AppConfig
from src.anonymizer.inference.engine import InferenceEngine

app_config = AppConfig.from_env_and_yaml(yaml_path="configs/inference/app_config.yaml")
engine = InferenceEngine(app_config.engine)

image_bytes = Path("document.png").read_bytes()
result = engine.anonymize(image_bytes)

if result.success:
    Image.fromarray(result.anonymized_image.astype(np.uint8)).save("anonymized.png")
```

### Manual Regions (advanced)

```python
import numpy as np
from src.anonymizer.core.models import BoundingBox, TextRegion

# Provide manual regions to bypass OCR/NER
regions = [
    TextRegion(
        bbox=BoundingBox(left=100, top=50, right=300, bottom=90),
        original_text="john.doe@example.com",
        replacement_text="[EMAIL_ADDRESS]",
        confidence=0.99,
    )
]

result = engine.anonymize(image_bytes, text_regions=regions)
```

### Performance Monitoring

```python
from src.anonymizer.performance import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_session("image_processing")

_ = engine.anonymize(image_bytes)

report = monitor.end_session()
print(f"Peak memory: {report['resource_summary']['peak_memory_mb']:.1f}MB")
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