# Configuration Guide

Complete guide to configuring the Document Anonymization System for different use cases and environments.

## Configuration Overview

The system uses a hierarchical configuration approach with multiple levels of customization:

1. **Default Configuration**: Built-in sensible defaults
2. **Configuration Files**: YAML/JSON configuration files
3. **Environment Variables**: Runtime overrides
4. **Runtime Parameters**: Method-level overrides

## Core Configuration Classes

### AnonymizationConfig

Main configuration class that orchestrates all subsystem configurations.

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

## Configuration Categories

### 1. OCR Configuration

Controls text detection behavior and OCR engine selection.

#### Engine Selection

```python
# Single high-accuracy engine
config = AnonymizationConfig(
    ocr_engines=["paddleocr"]
)

# Multi-engine fallback chain
config = AnonymizationConfig(
    ocr_engines=["paddleocr", "easyocr", "tesseract"]
)

# Speed-optimized setup
config = AnonymizationConfig(
    ocr_engines=["tesseract"],  # Fastest option
    ocr_confidence_threshold=0.6  # Lower threshold for speed
)
```

#### Language Support

```python
# Multi-language documents
config = AnonymizationConfig(
    ocr_languages=["en", "es", "fr", "de"],
    ocr_engines=["paddleocr", "easyocr"]  # Both support multiple languages
)

# Language-specific optimization
config = AnonymizationConfig(
    ocr_languages=["zh", "ja"],  # Asian languages
    ocr_engines=["paddleocr"]  # Best for CJK languages
)
```

#### Advanced OCR Settings

```python
from src.anonymizer.ocr import OCRConfig

ocr_config = OCRConfig(
    engines=["paddleocr", "easyocr"],
    confidence_threshold=0.8,
    languages=["en"],
    
    # PaddleOCR specific settings
    paddleocr_use_angle_cls=True,
    paddleocr_use_space_char=True,
    
    # EasyOCR specific settings
    easyocr_paragraph=False,
    easyocr_width_ths=0.7,
    
    # Tesseract specific settings
    tesseract_psm=6,  # Page segmentation mode
    tesseract_oem=3,  # OCR engine mode
    
    # Performance settings
    max_image_size=2048,
    preprocessing_enabled=True,
    contrast_enhancement=1.2
)

config = AnonymizationConfig(ocr_config=ocr_config)
```

### 2. NER Configuration

Controls PII entity detection and recognition patterns.

#### Entity Types

```python
# Financial document processing
config = AnonymizationConfig(
    entity_types=[
        "PERSON",
        "CREDIT_CARD",
        "IBAN_CODE", 
        "US_SSN",
        "US_BANK_NUMBER",
        "DATE_TIME",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS"
    ]
)

# Medical document processing
config = AnonymizationConfig(
    entity_types=[
        "PERSON",
        "DATE_TIME",
        "PHONE_NUMBER",
        "EMAIL_ADDRESS",
        "MEDICAL_LICENSE",
        "US_SSN",
        "LOCATION"
    ]
)

# Basic personal information
config = AnonymizationConfig(
    entity_types=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
)
```

#### Custom Entity Patterns

```python
from src.anonymizer.ner import NERConfig

# Define custom patterns
custom_patterns = {
    "EMPLOYEE_ID": r"EMP-\d{6}",
    "PROJECT_CODE": r"PROJ-[A-Z]{3}-\d{4}",
    "INTERNAL_PHONE": r"\(\d{3}\) \d{3}-\d{4} ext\. \d{3,4}"
}

ner_config = NERConfig(
    entity_types=["PERSON", "EMAIL_ADDRESS", "EMPLOYEE_ID", "PROJECT_CODE"],
    confidence_threshold=0.85,
    custom_patterns=custom_patterns,
    
    # Advanced NER settings
    language="en",
    use_transformer_models=True,
    context_window_size=50,
    enable_context_analysis=True
)

config = AnonymizationConfig(ner_config=ner_config)
```

#### Confidence Tuning

```python
# High precision (fewer false positives)
config = AnonymizationConfig(
    ner_confidence_threshold=0.95,
    entity_types=["PERSON", "EMAIL_ADDRESS"]  # Focus on high-confidence types
)

# High recall (catch more entities, more false positives)
config = AnonymizationConfig(
    ner_confidence_threshold=0.6,
    entity_types=[  # Include more entity types
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
        "LOCATION", "DATE_TIME", "ORGANIZATION"
    ]
)

# Balanced approach
config = AnonymizationConfig(
    ner_confidence_threshold=0.8,
    # Use confidence-based filtering at runtime
    post_process_confidence_filter=True
)
```

### 3. Anonymization Strategy Configuration

Controls how detected PII regions are anonymized.

#### Inpainting Configuration

```python
from src.anonymizer.diffusion import DiffusionConfig

# High-quality inpainting
diffusion_config = DiffusionConfig(
    model_name="stable-diffusion-2-inpainting",
    num_inference_steps=50,  # Higher for better quality
    guidance_scale=7.5,
    strength=1.0,
    
    # Quality settings
    use_fp16=False,  # Full precision for quality
    safety_checker=True,
    requires_safety_checker=True,
    
    # Memory settings
    enable_memory_efficient_attention=True,
    enable_vae_slicing=False,  # Disable for quality
    enable_cpu_offload=False
)

config = AnonymizationConfig(
    anonymization_strategy="inpainting",
    diffusion_config=diffusion_config
)
```

#### Fast Anonymization

```python
# Speed-optimized setup
diffusion_config = DiffusionConfig(
    model_name="stable-diffusion-2-inpainting",
    num_inference_steps=20,  # Fewer steps for speed
    guidance_scale=5.0,
    
    # Speed optimizations
    use_fp16=True,  # Half precision
    enable_vae_slicing=True,
    enable_cpu_offload=True,
    
    # Batch settings
    batch_size=4
)

config = AnonymizationConfig(
    anonymization_strategy="inpainting",
    diffusion_config=diffusion_config,
    batch_size=8  # Larger batches
)
```

#### Alternative Strategies

```python
# Simple redaction (fastest)
config = AnonymizationConfig(
    anonymization_strategy="redaction",
    redaction_color=(0, 0, 0),  # Black bars
    redaction_opacity=1.0
)

# Blur anonymization
config = AnonymizationConfig(
    anonymization_strategy="blur",
    blur_radius=15,
    blur_iterations=3
)

# Text replacement
config = AnonymizationConfig(
    anonymization_strategy="replacement",
    replacement_patterns={
        "PERSON": "[REDACTED NAME]",
        "EMAIL_ADDRESS": "[REDACTED EMAIL]",
        "PHONE_NUMBER": "[REDACTED PHONE]"
    }
)
```

### 4. Performance Configuration

Controls system performance and resource usage.

#### Memory Optimization

```python
# Memory-constrained environment
config = AnonymizationConfig(
    # Reduce batch sizes
    batch_size=1,
    
    # Enable memory optimizations
    memory_optimization=True,
    enable_caching=False,  # Disable caching to save memory
    
    # GPU memory settings
    use_gpu=True,
    gpu_memory_fraction=0.7,  # Limit GPU memory usage
    enable_cpu_offload=True,  # Offload when possible
    
    # Model loading
    load_models_on_demand=True,
    unload_models_after_use=True
)
```

#### High-Performance Configuration

```python
# Performance-optimized setup
config = AnonymizationConfig(
    # Large batch sizes
    batch_size=16,
    
    # GPU optimization
    use_gpu=True,
    mixed_precision=True,
    compile_models=True,  # PyTorch 2.0 compilation
    
    # Caching
    enable_caching=True,
    cache_size_gb=10,
    
    # Parallel processing
    parallel_workers=4,
    prefetch_factor=2,
    
    # Memory settings
    memory_optimization=False,  # Disable for speed
    pin_memory=True
)
```

#### CPU-Only Configuration

```python
# CPU-only processing
config = AnonymizationConfig(
    use_gpu=False,
    batch_size=2,  # Smaller batches for CPU
    
    # CPU optimization
    num_threads=8,  # Match CPU cores
    enable_mkldnn=True,  # Intel optimization
    
    # Memory settings
    memory_optimization=True,
    enable_caching=True,
    
    # Use lightweight models
    anonymization_strategy="redaction"  # Skip diffusion models
)
```

## Environment-Based Configuration

### Environment Variables

```bash
# OCR Configuration
export ANONYMIZER_OCR_ENGINES="paddleocr,easyocr"
export ANONYMIZER_OCR_CONFIDENCE=0.8
export ANONYMIZER_OCR_LANGUAGES="en,es"

# NER Configuration
export ANONYMIZER_ENTITY_TYPES="PERSON,EMAIL_ADDRESS,PHONE_NUMBER"
export ANONYMIZER_NER_CONFIDENCE=0.85

# Performance Configuration
export ANONYMIZER_USE_GPU=true
export ANONYMIZER_BATCH_SIZE=4
export ANONYMIZER_MEMORY_OPTIMIZATION=true

# Model Configuration
export ANONYMIZER_MODEL_CACHE_DIR="/opt/anonymizer/models"
export ANONYMIZER_MODEL_CACHE_SIZE="20GB"

# Logging Configuration
export ANONYMIZER_LOG_LEVEL=INFO
export ANONYMIZER_LOG_FORMAT=json
export ANONYMIZER_LOG_FILE="/var/log/anonymizer.log"
```

### Configuration Files

#### YAML Configuration

```yaml
# config/production.yaml
anonymization:
  ocr:
    engines: ["paddleocr", "easyocr"]
    confidence_threshold: 0.8
    languages: ["en"]
    max_image_size: 2048
    
  ner:
    entity_types:
      - "PERSON"
      - "EMAIL_ADDRESS" 
      - "PHONE_NUMBER"
      - "CREDIT_CARD"
    confidence_threshold: 0.85
    custom_patterns:
      employee_id: "EMP-\\d{6}"
      
  anonymization:
    strategy: "inpainting"
    preserve_formatting: true
    
  performance:
    use_gpu: true
    batch_size: 6
    memory_optimization: true
    parallel_workers: 4
    
  models:
    cache_dir: "/opt/anonymizer/models"
    auto_download: true
    verify_checksums: true
    
  logging:
    level: "INFO"
    format: "json"
    file: "/var/log/anonymizer.log"
```

#### JSON Configuration

```json
{
  "anonymization": {
    "ocr": {
      "engines": ["paddleocr", "tesseract"],
      "confidence_threshold": 0.7,
      "languages": ["en", "es"]
    },
    "ner": {
      "entity_types": ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
      "confidence_threshold": 0.8
    },
    "anonymization": {
      "strategy": "inpainting",
      "preserve_formatting": true
    },
    "performance": {
      "use_gpu": true,
      "batch_size": 4,
      "memory_optimization": true
    }
  }
}
```

### Loading Configurations

```python
from src.anonymizer.config import load_config

# Load from file
config = load_config("config/production.yaml")

# Load with environment overrides
config = load_config("config/base.yaml", env_override=True)

# Load from environment only
config = load_config(env_only=True)

# Programmatic override
config = load_config("config/base.yaml")
config.batch_size = 8  # Override specific setting
config.use_gpu = False
```

## Use Case Configurations

### 1. Development Configuration

```python
# Fast iteration for development
dev_config = AnonymizationConfig(
    # Fast OCR
    ocr_engines=["tesseract"],
    ocr_confidence_threshold=0.6,
    
    # Basic entities only
    entity_types=["PERSON", "EMAIL_ADDRESS"],
    ner_confidence_threshold=0.7,
    
    # Fast anonymization
    anonymization_strategy="redaction",
    
    # Minimal resource usage
    use_gpu=False,
    batch_size=1,
    memory_optimization=True,
    enable_caching=False,
    
    # Verbose logging
    log_level="DEBUG"
)
```

### 2. Production Configuration

```python
# Balanced production setup
prod_config = AnonymizationConfig(
    # Multi-engine OCR with fallback
    ocr_engines=["paddleocr", "easyocr", "tesseract"],
    ocr_confidence_threshold=0.8,
    
    # Comprehensive entity detection
    entity_types=[
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
        "CREDIT_CARD", "US_SSN", "IBAN_CODE"
    ],
    ner_confidence_threshold=0.85,
    
    # High-quality anonymization
    anonymization_strategy="inpainting",
    preserve_formatting=True,
    
    # Optimized performance
    use_gpu=True,
    batch_size=6,
    memory_optimization=True,
    enable_caching=True,
    
    # Production logging
    log_level="INFO",
    enable_audit_logging=True,
    enable_performance_monitoring=True
)
```

### 3. High-Security Configuration

```python
# Maximum security and accuracy
security_config = AnonymizationConfig(
    # All available OCR engines
    ocr_engines=["paddleocr", "easyocr", "trotr", "tesseract"],
    ocr_confidence_threshold=0.9,
    
    # Comprehensive PII detection
    entity_types=[
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
        "US_SSN", "IBAN_CODE", "US_BANK_NUMBER", "DATE_TIME",
        "LOCATION", "ORGANIZATION", "MEDICAL_LICENSE"
    ],
    ner_confidence_threshold=0.95,
    
    # Secure anonymization
    anonymization_strategy="inpainting",
    verify_anonymization=True,
    double_check_pii=True,
    
    # Security settings
    secure_memory_cleanup=True,
    audit_all_operations=True,
    enable_checkpoints=True,
    
    # Performance (secondary to security)
    use_gpu=True,
    batch_size=2,  # Smaller batches for thorough processing
)
```

### 4. High-Throughput Configuration

```python
# Maximum throughput optimization
throughput_config = AnonymizationConfig(
    # Single fastest OCR engine
    ocr_engines=["paddleocr"],
    ocr_confidence_threshold=0.7,
    
    # Essential entities only
    entity_types=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    ner_confidence_threshold=0.8,
    
    # Fast anonymization
    anonymization_strategy="inpainting",
    diffusion_steps=20,  # Reduced steps
    
    # Maximum performance
    use_gpu=True,
    batch_size=16,  # Large batches
    parallel_workers=8,
    memory_optimization=False,  # Speed over memory
    enable_caching=True,
    prefetch_data=True,
    
    # Minimal logging
    log_level="WARNING"
)
```

## Advanced Configuration Patterns

### 1. Dynamic Configuration

```python
class DynamicConfig:
    def __init__(self, base_config: AnonymizationConfig):
        self.base_config = base_config
        
    def get_config_for_document(self, document_path: str) -> AnonymizationConfig:
        # Analyze document characteristics
        doc_info = self.analyze_document(document_path)
        
        # Adjust configuration based on document
        config = copy.deepcopy(self.base_config)
        
        if doc_info.has_complex_layout:
            config.ocr_engines = ["paddleocr", "easyocr"]
            config.ocr_confidence_threshold = 0.8
            
        if doc_info.is_large_file:
            config.batch_size = 2
            config.memory_optimization = True
            
        if doc_info.has_sensitive_data:
            config.ner_confidence_threshold = 0.95
            config.verify_anonymization = True
            
        return config
```

### 2. A/B Testing Configuration

```python
class ABTestConfig:
    def __init__(self):
        self.config_a = AnonymizationConfig(
            anonymization_strategy="inpainting",
            diffusion_steps=50
        )
        
        self.config_b = AnonymizationConfig(
            anonymization_strategy="inpainting", 
            diffusion_steps=30
        )
        
    def get_config(self, user_id: str) -> AnonymizationConfig:
        # Simple hash-based assignment
        if hash(user_id) % 2 == 0:
            return self.config_a
        else:
            return self.config_b
```

### 3. Environment-Aware Configuration

```python
def get_environment_config() -> AnonymizationConfig:
    env = os.getenv("ENVIRONMENT", "development")
    
    if env == "development":
        return AnonymizationConfig(
            use_gpu=False,
            batch_size=1,
            log_level="DEBUG"
        )
    elif env == "staging":
        return AnonymizationConfig(
            use_gpu=True,
            batch_size=4,
            log_level="INFO"
        )
    elif env == "production":
        return AnonymizationConfig(
            use_gpu=True,
            batch_size=8,
            log_level="WARNING",
            enable_monitoring=True
        )
    else:
        raise ValueError(f"Unknown environment: {env}")
```

## Configuration Validation

```python
from src.anonymizer.config import validate_config

def validate_production_config(config: AnonymizationConfig) -> List[str]:
    """Validate configuration for production use."""
    issues = []
    
    # Check OCR configuration
    if len(config.ocr_engines) < 2:
        issues.append("Production should use multiple OCR engines for fallback")
        
    if config.ocr_confidence_threshold < 0.7:
        issues.append("OCR confidence threshold too low for production")
    
    # Check NER configuration
    required_entities = {"PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"}
    if not required_entities.issubset(set(config.entity_types)):
        issues.append("Missing required entity types for production")
    
    # Check performance configuration
    if config.batch_size < 2:
        issues.append("Batch size too small for production throughput")
        
    if not config.use_gpu and config.anonymization_strategy == "inpainting":
        issues.append("GPU recommended for inpainting in production")
    
    return issues

# Usage
config = load_config("config/production.yaml")
issues = validate_production_config(config)
if issues:
    for issue in issues:
        print(f"Configuration issue: {issue}")
```

## Best Practices

### 1. Configuration Management

- **Version Control**: Store configuration files in version control
- **Environment Separation**: Use different configs for dev/staging/prod
- **Validation**: Always validate configurations before deployment
- **Documentation**: Document all configuration options and their effects

### 2. Performance Tuning

- **Profile First**: Use performance monitoring to identify bottlenecks
- **Batch Size**: Start with default, adjust based on available memory
- **GPU Memory**: Monitor GPU utilization and adjust accordingly
- **Caching**: Enable caching in production, disable in memory-constrained environments

### 3. Security Considerations

- **Sensitive Data**: Never log sensitive configuration values
- **Access Control**: Restrict access to production configuration files
- **Audit Trail**: Log all configuration changes
- **Encryption**: Encrypt configuration files containing sensitive settings

### 4. Monitoring and Alerts

```python
# Configuration for monitoring
monitoring_config = {
    "metrics": {
        "processing_time_threshold_ms": 5000,
        "memory_usage_threshold_mb": 8192,
        "error_rate_threshold_percent": 5.0
    },
    "alerts": {
        "email": ["admin@company.com"],
        "slack_webhook": "https://hooks.slack.com/...",
        "pagerduty_key": "your-pagerduty-key"
    }
}
```