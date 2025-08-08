# Troubleshooting Guide

Comprehensive troubleshooting guide for common issues with the Document Anonymization System.

## Common Issues and Solutions

### Installation Issues

#### 1. Python Version Compatibility

**Problem**: Installation fails with Python version errors

**Symptoms**:
```
ERROR: Python 3.7 is not supported
SyntaxError: f-strings require Python 3.6+
```

**Solution**:
```bash
# Check Python version
python --version

# Install Python 3.10+ (Ubuntu/Debian)
sudo apt update
sudo apt install python3.10 python3.10-dev python3.10-venv

# Create virtual environment with correct Python
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

#### 2. Dependency Installation Failures

**Problem**: Package installation fails due to missing system dependencies

**Symptoms**:
```
error: Microsoft Visual C++ 14.0 is required
fatal error: 'Python.h' file not found
Building wheel for package failed
```

**Solutions**:

**Linux (Ubuntu/Debian)**:
```bash
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1
```

**Windows**:
```bash
# Install Visual Studio Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or install via chocolatey
choco install visualstudio2019buildtools --package-parameters "--add Microsoft.VisualStudio.Workload.VCTools"
```

**macOS**:
```bash
# Install Xcode command line tools
xcode-select --install

# Install via homebrew
brew install cmake pkg-config
```

#### 3. CUDA/GPU Setup Issues

**Problem**: GPU not detected or CUDA errors

**Symptoms**:
```
RuntimeError: CUDA out of memory
RuntimeError: No CUDA GPUs are available
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver
```

**Solutions**:

**Check GPU and drivers**:
```bash
# Check if GPU is detected
lspci | grep -i nvidia

# Check NVIDIA driver
nvidia-smi

# If driver not installed
sudo apt install nvidia-driver-515
sudo reboot
```

**Install CUDA**:
```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda
```

**Check PyTorch CUDA support**:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA devices: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
```

### Runtime Issues

#### 1. Out of Memory Errors

**Problem**: System runs out of memory during processing

**Symptoms**:
```
OutOfMemoryError: CUDA out of memory
MemoryError: Unable to allocate memory
RuntimeError: [enforce fail at CPUAllocator.cpp:75]
```

**Solutions**:

**Reduce batch size**:
```python
from src.anonymizer.core.config import EngineConfig
config = EngineConfig(
    batch_size=1,  # Reduce from default
    memory_optimization=True,
    enable_cpu_offload=True
)
```

**Enable memory optimizations**:
```python
config = EngineConfig(
    # GPU memory optimization
    enable_memory_efficient_attention=True,
    enable_vae_slicing=True,
    enable_cpu_offload=True,
    
    # General memory optimization
    unload_models_after_use=True,
    force_garbage_collection=True,
    cleanup_interval_documents=5
)
```

**Monitor memory usage**:
```python
from src.anonymizer.performance import MemoryMonitor

monitor = MemoryMonitor()
monitor.start_monitoring()

# Your processing code here
from src.anonymizer.core.config import AppConfig
from src.anonymizer.inference.engine import InferenceEngine
result = InferenceEngine(AppConfig.from_env_and_yaml().engine).anonymize(Path("large_document.png").read_bytes())

memory_stats = monitor.get_current_usage()
print(f"Peak memory: {memory_stats.peak_memory_mb}MB")
```

#### 2. Model Loading Failures

**Problem**: Models fail to load or download

**Symptoms**:
```
FileNotFoundError: Model file not found
OSError: Unable to load weights from checkpoint
ConnectionError: Failed to download model
```

**Solutions**:

**Check model availability**:
```python
from src.anonymizer.models import ModelManager

manager = ModelManager()
available_models = manager.list_available_models()
downloaded_models = manager.list_downloaded_models()

print("Available models:", [m.name for m in available_models])
print("Downloaded models:", [m.name for m in downloaded_models])
```

**Download missing models**:
```bash
# Download specific model
python scripts/download_models.py download sd2-vae

# Download all required models
python scripts/download_models.py ensure-models --use-case default

# Verify model integrity
python scripts/download_models.py verify-all
```

**Check model directory permissions**:
```bash
# Check permissions
ls -la ~/.cache/anonymizer/models/

# Fix permissions if needed
chmod -R 755 ~/.cache/anonymizer/models/
chown -R $USER:$USER ~/.cache/anonymizer/models/
```

#### 3. OCR Engine Failures

**Problem**: OCR engines fail to detect text

**Symptoms**:
```
OCRError: All OCR engines failed
ImportError: No module named 'easyocr'
PaddleOCR initialization failed
```

**Solutions**:

**Install missing OCR dependencies**:
```bash
# PaddleOCR
pip install paddlepaddle paddleocr

# EasyOCR
pip install easyocr

# Tesseract
sudo apt-get install tesseract-ocr libtesseract-dev
pip install pytesseract

# TrOCR
pip install transformers torch
```

**Test individual OCR engines**:
```python
from src.anonymizer.ocr import OCREngineFactory
import numpy as np

# Test engine availability
factory = OCREngineFactory()
available_engines = factory.list_available_engines()
print("Available OCR engines:", available_engines)

# Test specific engine
try:
    engine = factory.create("paddleocr")
    # Test with dummy image
    test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
    result = engine.detect_text(test_image)
    print("PaddleOCR test passed")
except Exception as e:
    print(f"PaddleOCR test failed: {e}")
```

**Use fallback configuration**:
```python
# Robust OCR configuration with fallbacks
config = EngineConfig(
    ocr_engines=["tesseract"],  # Use most reliable engine
    ocr_confidence_threshold=0.5,  # Lower threshold
    ocr_fallback_enabled=True
)
```

#### 4. NER Pipeline Errors

**Problem**: Named Entity Recognition fails

**Symptoms**:
```
NERError: Entity recognition failed
ModuleNotFoundError: No module named 'presidio_analyzer'
spacy.errors.OSError: Can't find model 'en_core_web_sm'
```

**Solutions**:

**Install NER dependencies**:
```bash
# Install Presidio
pip install presidio-analyzer presidio-anonymizer

# Install spaCy model
python -m spacy download en_core_web_sm

# For other languages
python -m spacy download es_core_news_sm  # Spanish
python -m spacy download fr_core_news_sm  # French
```

**Test NER functionality**:
```python
from src.anonymizer.ner import NERPipeline, NERConfig

config = NERConfig(
    entity_types=["PERSON", "EMAIL_ADDRESS"],
    confidence_threshold=0.8
)

ner = NERPipeline(config)

# Test with sample text
test_text = "John Doe's email is john.doe@example.com"
entities = ner.analyze_text(test_text)

print(f"Detected entities: {len(entities)}")
for entity in entities:
    print(f"  {entity.entity_type}: {entity.text} (confidence: {entity.confidence:.2f})")
```

**Use CPU-only configuration**:
```python
# If GPU-related NER issues
config = NERConfig(
    use_gpu=False,
    model_name="en_core_web_sm",  # Use CPU model
    batch_size=1
)
```

### Performance Issues

#### 1. Slow Processing Speed

**Problem**: Document processing is too slow

**Symptoms**:
- High processing times per document
- Low CPU/GPU utilization
- System appears unresponsive

**Diagnosis**:
```python
from src.anonymizer.performance import PerformanceProfiler

profiler = PerformanceProfiler()
profiler.start_profiling()

# Your slow operation
result = InferenceEngine(AppConfig.from_env_and_yaml().engine).anonymize(Path("document.png").read_bytes())

profile_report = profiler.get_profile_report()
print("Top time consumers:")
for func, time_ms in profile_report.time_hotspots[:5]:
    print(f"  {func}: {time_ms:.1f}ms")
```

**Solutions**:

**Optimize configuration**:
```python
# Speed-optimized configuration
config = EngineConfig(
    # Use fastest OCR
    ocr_engines=["tesseract"],
    ocr_confidence_threshold=0.6,
    
    # Use simple anonymization
    anonymization_strategy="redaction",
    
    # Optimize processing
    batch_size=8,
    use_gpu=True,
    memory_optimization=False,  # Trade memory for speed
    enable_caching=True
)
```

**Parallel processing**:
```python
from concurrent.futures import ThreadPoolExecutor

def process_documents_parallel(documents, max_workers=4):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(anonymizer.anonymize_document, doc)
            for doc in documents
        ]
        return [future.result() for future in futures]
```

#### 2. High Memory Usage

**Problem**: System uses excessive memory

**Diagnosis**:
```python
from src.anonymizer.performance import MemoryProfiler
import psutil

# Monitor system memory
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024**2:.1f}MB")

# Profile memory usage
profiler = MemoryProfiler()
with profiler.profile_memory():
    result = InferenceEngine(AppConfig.from_env_and_yaml().engine).anonymize(Path("document.png").read_bytes())

memory_report = profiler.get_memory_report()
print(f"Peak memory: {memory_report.peak_memory_mb:.1f}MB")
```

**Solutions**:

**Memory optimization**:
```python
config = EngineConfig(
    batch_size=1,
    memory_optimization=True,
    unload_models_after_use=True,
    enable_cpu_offload=True,
    cache_size_gb=1,  # Limit cache size
    force_garbage_collection=True
)
```

**Process documents individually**:
```python
def process_large_batch(documents):
    """Process large batches with memory management."""
    results = []
    
    for i, doc in enumerate(documents):
        result = anonymizer.anonymize_document(doc)
        results.append(result)
        
        # Cleanup every 10 documents
        if i % 10 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return results
```

### Configuration Issues

#### 1. Invalid Configuration

**Problem**: Configuration validation fails

**Symptoms**:
```
ConfigurationError: Invalid entity type 'INVALID_TYPE'
ValueError: batch_size must be positive
TypeError: 'NoneType' object is not iterable
```

**Solutions**:

**Validate configuration**:
```python
from src.anonymizer.config import validate_config

config = AnonymizationConfig(
    entity_types=["PERSON", "EMAIL"],  # Valid types
    batch_size=4,  # Positive integer
    ocr_engines=["paddleocr"]  # Valid engines
)

# Validate before use
validation_errors = validate_config(config)
if validation_errors:
    print("Configuration errors:")
    for error in validation_errors:
        print(f"  - {error}")
else:
    print("Configuration is valid")
```

**Use configuration templates**:
```python
from src.anonymizer.config import ConfigurationTemplates

# Use pre-validated templates
config = ConfigurationTemplates.get_production_config()
config = ConfigurationTemplates.get_development_config()
config = ConfigurationTemplates.get_high_accuracy_config()
```

#### 2. Environment Variable Issues

**Problem**: Environment variables not being recognized

**Solutions**:

**Check environment variables**:
```bash
# List all anonymizer environment variables
env | grep ANONYMIZER

# Set required variables
export ANONYMIZER_USE_GPU=true
export ANONYMIZER_LOG_LEVEL=INFO
export ANONYMIZER_MODEL_CACHE_DIR=/opt/models
```

**Debug configuration loading**:
```python
import os
from src.anonymizer.config import load_config

# Enable debug logging
os.environ['ANONYMIZER_LOG_LEVEL'] = 'DEBUG'

# Load configuration with debugging
config = load_config(env_override=True, debug=True)
print("Loaded configuration:")
print(config)
```

### File Processing Issues

#### 1. PDF Processing Errors

**Problem**: PDF files cannot be processed

**Symptoms**:
```
PyMuPDFError: cannot open document
PDFSyntaxError: PDF syntax error
FileNotFoundError: [Errno 2] No such file or directory
```

**Solutions**:

**Check file integrity**:
```python
import fitz  # PyMuPDF

def check_pdf_integrity(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        print(f"PDF is valid: {page_count} pages")
        return True
    except Exception as e:
        print(f"PDF error: {e}")
        return False

# Test your PDF
is_valid = check_pdf_integrity("problematic_file.pdf")
```

**Handle corrupted PDFs**:
```python
def robust_pdf_processing(pdf_path):
    """Process PDF with error handling."""
    try:
        # Try standard processing
        result = InferenceEngine(AppConfig.from_env_and_yaml().engine).anonymize(Path(pdf_path).read_bytes())
        return result
    except Exception as e:
        print(f"Standard processing failed: {e}")
        
        # Try with repair
        try:
            repaired_path = repair_pdf(pdf_path)
            result = InferenceEngine(AppConfig.from_env_and_yaml().engine).anonymize(Path(repaired_path).read_bytes())
            return result
        except Exception as e2:
            print(f"Repair failed: {e2}")
            return None

def repair_pdf(pdf_path):
    """Attempt to repair corrupted PDF."""
    import subprocess
    
    output_path = pdf_path.replace('.pdf', '_repaired.pdf')
    
    # Use gs (Ghostscript) to repair
    subprocess.run([
        'gs', '-o', output_path, '-sDEVICE=pdfwrite',
        '-dPDFSETTINGS=/prepress', pdf_path
    ], check=True)
    
    return output_path
```

#### 2. Large File Processing

**Problem**: Large files cause timeouts or memory issues

**Solutions**:

**Split large PDFs**:
```python
def split_large_pdf(pdf_path, max_pages=10):
    """Split large PDF into smaller chunks."""
    import fitz
    
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    chunks = []
    for start_page in range(0, total_pages, max_pages):
        end_page = min(start_page + max_pages, total_pages)
        
        # Create chunk
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start_page, to_page=end_page-1)
        
        chunk_path = pdf_path.replace('.pdf', f'_chunk_{start_page}-{end_page}.pdf')
        chunk_doc.save(chunk_path)
        chunk_doc.close()
        
        chunks.append(chunk_path)
    
    doc.close()
    return chunks

# Process large file in chunks
large_file = "huge_document.pdf"
chunks = split_large_pdf(large_file, max_pages=5)

results = []
for chunk in chunks:
    result = InferenceEngine(AppConfig.from_env_and_yaml().engine).anonymize(chunk)
    results.append(result)
```

**Stream processing**:
```python
def stream_process_large_file(pdf_path):
    """Process large file page by page."""
    import fitz
    
    doc = fitz.open(pdf_path)
    output_doc = fitz.open()
    
    for page_num in range(len(doc)):
        # Process single page
        page = doc[page_num]
        page_image = page.get_pixmap().tobytes("png")
        
        # Anonymize page image
        anonymized_image = process_single_page_image(page_image)
        
        # Add to output document
        output_page = output_doc.new_page(width=page.rect.width, height=page.rect.height)
        output_page.insert_image(output_page.rect, stream=anonymized_image)
        
        # Clean up memory
        page = None
        if page_num % 10 == 0:
            import gc
            gc.collect()
    
    # Save output
    output_path = pdf_path.replace('.pdf', '_anonymized.pdf')
    output_doc.save(output_path)
    output_doc.close()
    doc.close()
    
    return output_path
```

### Debugging and Diagnostics

#### Enable Debug Logging

```python
import logging

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable specific loggers
logging.getLogger('anonymizer.ocr').setLevel(logging.DEBUG)
logging.getLogger('anonymizer.ner').setLevel(logging.DEBUG)
logging.getLogger('anonymizer.diffusion').setLevel(logging.DEBUG)
```

#### System Diagnostics

```python
def run_system_diagnostics():
    """Run comprehensive system diagnostics."""
    
    diagnostics = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / 1024**3,
    }
    
    # GPU diagnostics
    if torch.cuda.is_available():
        diagnostics["gpu"] = {
            "available": True,
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device(),
            "device_name": torch.cuda.get_device_name(),
            "memory_allocated": torch.cuda.memory_allocated() / 1024**2,
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**2
        }
    else:
        diagnostics["gpu"] = {"available": False}
    
    # Model diagnostics
    from src.anonymizer.models import ModelManager
    manager = ModelManager()
    diagnostics["models"] = {
        "available": len(manager.list_available_models()),
        "downloaded": len(manager.list_downloaded_models()),
        "storage_stats": manager.get_storage_stats()
    }
    
    # OCR diagnostics
    from src.anonymizer.ocr import OCREngineFactory
    factory = OCREngineFactory()
    diagnostics["ocr_engines"] = factory.list_available_engines()
    
    return diagnostics

# Run diagnostics
diag = run_system_diagnostics()
import json
print(json.dumps(diag, indent=2))
```

#### Performance Profiling

```python
def profile_anonymization(document_path):
    """Profile anonymization performance."""
    
    import cProfile
    import pstats
    from io import StringIO
    
    # Run profiler
    profiler = cProfile.Profile()
    profiler.enable()
    
    result = InferenceEngine(AppConfig.from_env_and_yaml().engine).anonymize(Path(document_path).read_bytes())
    
    profiler.disable()
    
    # Generate report
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.sort_stats('cumulative').print_stats(20)
    
    print("Performance Profile:")
    print(s.getvalue())
    
    return result
```

## Getting Help

### Support Channels

1. **Documentation**: Check the [documentation](docs/) for detailed guides
2. **GitHub Issues**: Report bugs and request features at [Issues](https://github.com/your-repo/issues)
3. **Discussions**: Ask questions in [Discussions](https://github.com/your-repo/discussions)
4. **Email Support**: Contact support@example.com for urgent issues

### Reporting Issues

When reporting issues, please include:

1. **System Information**:
   - Operating system and version
   - Python version
   - GPU information (if applicable)
   - Available memory and storage

2. **Environment Details**:
   - Installation method (pip, source, Docker)
   - Configuration used
   - Environment variables set

3. **Error Information**:
   - Complete error message and stack trace
   - Steps to reproduce the issue
   - Input files that cause the problem (if possible)

4. **Logs**:
   - Debug logs showing the issue
   - Performance monitoring data (if applicable)

### Issue Template

```markdown
**System Information:**
- OS: Ubuntu 20.04
- Python: 3.10.5
- GPU: NVIDIA RTX 3080 (8GB)
- Memory: 32GB RAM

**Installation:**
- Method: pip install
- Version: 1.0.0

**Configuration:**
```yaml
anonymization:
  ocr_engines: ["paddleocr", "easyocr"]
  entity_types: ["PERSON", "EMAIL_ADDRESS"]
  use_gpu: true
```

**Issue Description:**
Brief description of the problem...

**Steps to Reproduce:**
1. Step 1
2. Step 2
3. Step 3

**Error Message:**
```
Complete error message and stack trace
```

**Expected Behavior:**
What you expected to happen...

**Additional Context:**
Any other relevant information...
```

## FAQ

### Q: Why is my first document processing slow?

**A**: The first run requires downloading and loading models, which can take several minutes. Subsequent runs will be much faster due to caching.

### Q: Can I run this without a GPU?

**A**: Yes, set `use_gpu=False` in your configuration. Processing will be slower but still functional.

### Q: How do I reduce memory usage?

**A**: Use smaller batch sizes, enable memory optimization, and consider using CPU-only mode for very large documents.

### Q: What file formats are supported?

**A**: Currently, only PDF files are supported. Support for other formats (DOCX, images) may be added in future versions.

### Q: How accurate is the PII detection?

**A**: Accuracy depends on document quality and configuration. Typical accuracy ranges from 85-95% with proper tuning.

### Q: Can I add custom entity types?

**A**: Yes, you can define custom regex patterns for domain-specific entities. See the configuration guide for details.

### Q: Is this production-ready?

**A**: The system is designed for production use with proper configuration, monitoring, and testing. See the deployment guide for production recommendations.

### Q: How do I update to a new version?

**A**: Use `pip install --upgrade document-anonymizer` for pip installations, or pull the latest code for source installations. Always test in a development environment first.