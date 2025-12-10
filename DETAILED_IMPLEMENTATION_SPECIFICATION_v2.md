# Document Anonymizer: Implementation Specification v2

## Executive Summary

A production-ready document anonymization system that replaces sensitive information (PII) in documents with realistic synthetic content using diffusion models. The system is designed to be modular, cloud-agnostic, and suitable for both research and production deployment.

### Key Capabilities
- **Document Processing**: PDF, image, and text document anonymization
- **PII Detection**: Named Entity Recognition (NER) for identifying sensitive data
- **Synthetic Generation**: Diffusion model-based realistic text replacement
- **Flexible Deployment**: Support for cloud, on-premise, and hybrid deployments
- **Production Ready**: Error handling, monitoring, and scalability built-in

## System Architecture

### High-Level Architecture
```
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│   Document Input    │────▶│   NER Detection     │────▶│ Diffusion Synthesis │
│  (PDF/Image/Text)   │     │(spaCy/Presidio/vLLM)│     │  (VAE + UNet)       │
└─────────────────────┘     └─────────────────────┘     └─────────────────────┘
                                      │                            │
                                      ▼                            ▼
                            ┌─────────────────────┐     ┌─────────────────────┐
                            │ Bounding Box Coords │     │  Generated Patches  │
                            │   + Entity Types    │     │   + Confidence      │
                            └─────────────────────┘     └─────────────────────┘
                                      │                            │
                                      └────────────┬───────────────┘
                                                   ▼
                                         ┌─────────────────────┐
                                         │  Result Compositor  │
                                         │ (Seamless Blending) │
                                         └─────────────────────┘
                                                   │
                                                   ▼
                                         ┌─────────────────────┐
                                         │ Anonymized Document │
                                         └─────────────────────┘
```

### Component Structure
```
src/anonymizer/
├── core/                    # Core models, configuration, exceptions
│   ├── config.py           # Pydantic settings and configuration models
│   ├── models.py           # Data models (BoundingBox, AnonymizationRequest, etc.)
│   └── exceptions.py       # Custom exceptions
│
├── training/               # Model training components
│   ├── vae_trainer.py     # VAE training implementation
│   ├── unet_trainer.py    # UNet training implementation
│   ├── datasets.py        # Training data loaders
│   └── schedulers.py      # Learning rate schedulers
│
├── inference/             # Inference pipeline
│   ├── engine.py         # Main inference engine
│   ├── preprocessing.py  # Image preprocessing
│   ├── postprocessing.py # Result composition
│   └── verification.py   # Quality verification
│
├── detection/            # PII detection
│   ├── ner_detector.py  # NER-based detection
│   ├── presidio.py      # Presidio integration
│   └── custom_rules.py  # Custom detection rules
│
├── storage/             # Storage abstraction
│   ├── base.py         # Abstract storage interface
│   ├── r2_client.py    # Cloudflare R2 implementation
│   ├── s3_client.py    # AWS S3 implementation
│   └── local_client.py # Local filesystem
│
├── cloud/              # Cloud training providers
│   ├── base.py        # Abstract cloud trainer
│   ├── modal_trainer.py # Modal.com implementation
│   └── local_trainer.py # Local training
│
└── utils/              # Utilities
    ├── image_ops.py    # Image operations
    ├── text_rendering.py # Text rendering
    ├── metrics.py      # Performance metrics
    └── logging.py      # Structured logging
```

## Training Data Requirements

### Dataset Overview
The system requires diverse, high-quality training data to learn effective text anonymization across different document types, languages, and formatting styles.

**Current Dataset Status: 121,914 samples**

| Dataset    | N_train | Languages/Query | Quality | Status |
|------------|---------|-----------------|---------|---------|
| xFund      | 745     | DE, FR, IT, ES, PT | High | ✅ Complete |
| PubLayNet  | 102,079 | EN | Medium* | ⚠️ OCR issues |
| ICDAR2015  | 939     | Multi | High | ✅ Complete |
| MSRA       | 290     | EN/CN | High | ✅ Complete |
| SCUT       | 942     | Multi | High | ✅ Complete |
| ePeriodica | 6,155   | DE, FR | High | 🔄 In progress |
| Bing       | 10,764  | DE, FR | Medium | 🔄 In progress |

*OCR quality issues with PubLayNet require improvement

### Data Quality Requirements
- **Minimum resolution**: 512x512 pixels for training patches
- **OCR confidence**: >0.7 for text detection accuracy
- **Text diversity**: Multiple fonts, sizes, and formatting styles
- **Background variety**: Documents, forms, natural scenes, advertisements
- **Language coverage**: Primary focus on European languages (DE, FR, IT, ES, PT, EN)

### Critical Issues to Address
1. **PubLayNet OCR Quality**: Current OCR results are poor quality
2. **ePeriodica Access**: Collection interrupted due to IP restrictions
3. **Resolution Scaling**: xFund images need upscaling from 224x224
4. **Text Enhancement**: Low-quality document images need preprocessing

## Implementation Phases

### Phase 1: Core Inference Pipeline (Priority: CRITICAL)

#### 1.1 Inference Engine (`inference/engine.py`)
The main orchestrator for document anonymization.

**Key Features:**
- Model lifecycle management (loading, caching, cleanup)
- Request validation and preprocessing
- Batch processing with memory management
- Error recovery and fallback mechanisms

**Implementation Details:**
```python
class DiffusionEngine:
    def __init__(self, config: EngineConfig, storage_client: StorageClient):
        """Initialize with dependency injection for testability."""
        self.config = config
        self.storage = storage_client
        self.models = {}  # Model cache
        self.preprocessor = ImagePreprocessor(config.preprocessing)
        self.postprocessor = ResultProcessor(config.postprocessing)
        self.verifier = QualityVerifier(config.verification)

    async def anonymize_document(self, request: AnonymizationRequest) -> AnonymizationResult:
        """Main anonymization workflow with error handling."""
        try:
            # 1. Load models if not cached
            await self._ensure_models_loaded()

            # 2. Detect PII entities
            entities = await self._detect_entities(request.document)

            # 3. Process each entity
            patches = []
            for entity in entities:
                # Extract and preprocess region
                processed = self.preprocessor.preprocess_image(
                    request.document.image, entity.bbox
                )

                # Generate synthetic replacement
                patch = await self._generate_patch(processed, entity)

                # Verify quality
                confidence = self.verifier.calculate_patch_confidence(patch)
                if confidence > self.config.min_confidence:
                    patches.append(patch)

            # 4. Compose final result
            result_image = self.postprocessor.compose_final_image(
                request.document.image, patches
            )

            # 5. Final quality check
            quality_report = self.verifier.validate_result_quality(result_image)

            return AnonymizationResult(
                image=result_image,
                patches=patches,
                quality_report=quality_report
            )

        except Exception as e:
            logger.error(f"Anonymization failed: {e}", exc_info=True)
            raise AnonymizationError(f"Failed to process document: {str(e)}")
```

#### 1.2 Preprocessing Pipeline (`inference/preprocessing.py`)
Safe and efficient image preprocessing with bounds checking.

**Safety Features:**
- Memory usage limits to prevent OOM
- Coordinate validation and clamping
- Scale factor limits to prevent excessive upscaling
- Proper error messages for debugging

**Key Methods:**
```python
def preprocess_image(self, image: np.ndarray, bbox: BoundingBox) -> ProcessedImage:
    """Extract and prepare image region for diffusion model."""
    # Validate inputs
    bbox = self._validate_bbox(bbox, image.shape[1], image.shape[0])

    # Check memory requirements
    estimated_memory = self._estimate_memory_usage(bbox, self.config.target_size)
    if estimated_memory > self.config.max_memory_bytes:
        raise MemoryError(f"Processing would require {estimated_memory} bytes")

    # Extract crop with padding
    crop_data = self._extract_crop_safely(image, bbox)

    # Resize to model input size
    resized = self._resize_with_aspect_ratio(crop_data.image, self.config.target_size)

    # Generate mask
    mask = self._generate_mask(resized.shape, crop_data.relative_bbox)

    return ProcessedImage(
        image=resized,
        mask=mask,
        original_bbox=bbox,
        scale_factor=resized.shape[0] / crop_data.image.shape[0]
    )
```

#### 1.3 Postprocessing (`inference/postprocessing.py`)
Seamless integration of generated patches into original documents.

**Features:**
- Advanced blending algorithms (Poisson blending, feathering)
- Color matching and adjustment
- Artifact removal
- Format conversion

#### 1.4 Quality Verification (`inference/verification.py`)
Ensure generated content meets quality standards.

**Verification Metrics:**
- Text readability (OCR confidence)
- Visual quality (SSIM, perceptual metrics)
- Consistency across patches
- Anomaly detection

### Phase 2: Detection Pipeline (Priority: HIGH)

#### 2.1 NER Detector (`detection/ner_detector.py`)
SpaCy-based entity detection with custom rules.

**Features:**
- Multi-language support
- Custom entity types (SSN, account numbers, etc.)
- Confidence scoring
- Context-aware detection

#### 2.2 Presidio Integration (`detection/presidio.py`)
Microsoft Presidio for enhanced PII detection.

**Advantages:**
- Pre-built recognizers for common PII
- Regulatory compliance (GDPR, HIPAA)
- Extensible architecture
- Built-in anonymization strategies

### Phase 3: Storage Abstraction (Priority: HIGH)

#### 3.1 Storage Interface (`storage/base.py`)
Unified interface for all storage providers.

```python
class StorageClient(ABC):
    """Abstract base for storage operations."""

    @abstractmethod
    async def upload_model(self, model: ModelArtifacts) -> str:
        """Upload model artifacts and return identifier."""

    @abstractmethod
    async def download_model(self, model_id: str) -> ModelArtifacts:
        """Download model artifacts by identifier."""

    @abstractmethod
    async def list_models(self, prefix: str = "") -> List[ModelInfo]:
        """List available models."""
```

#### 3.2 Implementation Strategy
- **Cloudflare R2**: Cost-effective for large models
- **AWS S3**: Enterprise integration
- **Local Storage**: Development and on-premise
- **Caching Layer**: Redis/Memcached for frequently accessed models

### Phase 4: Cloud Training (Priority: MEDIUM)

#### 4.1 Training Abstraction (`cloud/base.py`)
Provider-agnostic training interface.

**Supported Platforms:**
- Modal.com (serverless GPU)
- AWS SageMaker
- GCP Vertex AI
- Local/On-premise

#### 4.2 Modal Implementation (`cloud/modal_trainer.py`)
Serverless training with automatic scaling.

**Features:**
- Dynamic GPU allocation
- Distributed training support
- Automatic checkpoint uploading
- Cost optimization

### Phase 5: CLI Enhancement (Priority: HIGH)

#### 5.1 Command Structure
```bash
# Document anonymization
document-anonymizer anonymize input.pdf --output output.pdf --confidence 0.8

# Model training
document-anonymizer train vae --dataset ./data --epochs 100
document-anonymizer train unet --vae-checkpoint ./vae.pt --dataset ./data

# Model management
document-anonymizer models list
document-anonymizer models download diffute-v1
document-anonymizer models upload ./checkpoint --name custom-v1

# Configuration
document-anonymizer config show
document-anonymizer config validate
document-anonymizer config generate --environment production
```

#### 5.2 Interactive Features
- Progress bars for long operations
- Real-time logs with filtering
- Interactive confirmation for destructive operations
- Shell completion

### Phase 6: Development UI for QC (Priority: HIGH)

#### 6.1 Simple Web Interface (`dev_ui/`)
A lightweight web interface for quality control during development.

**Purpose:**
- Visual inspection of each pipeline stage
- Manual validation of results
- Parameter tuning with immediate feedback
- Debugging support with step-by-step execution

**UI Components:**
```
dev_ui/
├── app.py              # FastAPI application
├── static/
│   ├── style.css      # Simple styling
│   └── script.js      # Interactive features
├── templates/
│   ├── index.html     # Main dashboard
│   ├── upload.html    # Document upload
│   └── results.html   # Results viewer
└── components/
    ├── pipeline_viewer.py  # Pipeline stage visualization
    ├── image_compare.py    # Before/after comparison
    └── bbox_editor.py      # Manual bounding box editing
```

**Key Features:**
1. **Document Upload**: Drag-and-drop with preview
2. **Pipeline Viewer**: Step-by-step execution with intermediate results
3. **Entity Inspector**: Visual overlay of detected entities
4. **Patch Generator**: Manual trigger of patch generation
5. **Quality Metrics**: Real-time quality scores
6. **Parameter Tuning**: Sliders for confidence thresholds, model parameters
7. **Export Options**: Save intermediate results and configurations

**Implementation:**
```python
# dev_ui/app.py
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tempfile
import json

app = FastAPI(title="Document Anonymizer Dev UI")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document with full pipeline visualization."""
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Process through pipeline with intermediate results
    pipeline_results = await process_with_debugging(tmp_path)

    return {
        "status": "success",
        "results": pipeline_results,
        "stages": [
            {"name": "Original", "image": pipeline_results.original_b64},
            {"name": "Detected Entities", "image": pipeline_results.entities_b64},
            {"name": "Generated Patches", "image": pipeline_results.patches_b64},
            {"name": "Final Result", "image": pipeline_results.final_b64}
        ]
    }

@app.get("/api/pipeline/{stage}")
async def get_pipeline_stage(stage: str, request_id: str):
    """Get detailed information about a specific pipeline stage."""
    # Return stage-specific data for debugging
    pass

@app.post("/api/regenerate_patch")
async def regenerate_patch(patch_id: str, params: dict):
    """Regenerate a specific patch with new parameters."""
    # Useful for parameter tuning
    pass
```

**UI Layout:**
```html
<!-- templates/index.html -->
<div class="container">
    <!-- Upload Section -->
    <div class="upload-zone">
        <input type="file" id="file-input" accept=".pdf,.jpg,.png" />
        <div class="drag-drop-area">Drop document here or click to upload</div>
    </div>

    <!-- Pipeline Progress -->
    <div class="pipeline-progress">
        <div class="stage active">1. Upload</div>
        <div class="stage">2. Entity Detection</div>
        <div class="stage">3. Patch Generation</div>
        <div class="stage">4. Final Composition</div>
    </div>

    <!-- Results Viewer -->
    <div class="results-container">
        <div class="before-after">
            <div class="image-container">
                <h3>Original</h3>
                <img id="original-image" />
                <div class="bbox-overlay" id="bbox-overlay"></div>
            </div>
            <div class="image-container">
                <h3>Anonymized</h3>
                <img id="result-image" />
                <div class="quality-metrics" id="quality-metrics"></div>
            </div>
        </div>

        <!-- Stage Details -->
        <div class="stage-details">
            <h3>Stage Details</h3>
            <div class="tabs">
                <button class="tab active" data-stage="entities">Detected Entities</button>
                <button class="tab" data-stage="patches">Generated Patches</button>
                <button class="tab" data-stage="metrics">Quality Metrics</button>
            </div>
            <div class="tab-content" id="stage-content"></div>
        </div>
    </div>

    <!-- Parameter Tuning -->
    <div class="parameter-panel">
        <h3>Parameters</h3>
        <div class="param-group">
            <label>Confidence Threshold</label>
            <input type="range" id="confidence-slider" min="0" max="1" step="0.1" value="0.7" />
            <span id="confidence-value">0.7</span>
        </div>
        <div class="param-group">
            <label>Guidance Scale</label>
            <input type="range" id="guidance-slider" min="1" max="20" step="0.5" value="7.5" />
            <span id="guidance-value">7.5</span>
        </div>
        <button id="reprocess-btn">Reprocess</button>
    </div>
</div>
```

#### 6.2 Development Workflow
**Typical QC Session:**
1. Upload test document
2. Review detected entities with visual overlay
3. Adjust detection parameters if needed
4. Generate patches and review quality
5. Fine-tune generation parameters
6. Export successful configurations
7. Save test cases for regression testing

**Benefits:**
- **Visual Debugging**: See exactly what the model is detecting and generating
- **Parameter Tuning**: Real-time feedback on parameter changes
- **Quality Assessment**: Immediate visual feedback on result quality
- **Test Case Generation**: Save good/bad examples for testing
- **Documentation**: Screenshots for presentations and papers

### Phase 7: Testing Strategy (Priority: CRITICAL)

#### 6.1 Unit Tests
- **Coverage Target**: >90% for core components
- **Mock Strategy**: Storage, cloud providers, external APIs
- **Property Testing**: Input validation, coordinate calculations

#### 6.2 Integration Tests
- End-to-end anonymization workflow
- Storage provider compatibility
- Model loading and inference
- Error recovery scenarios

#### 6.3 Performance Tests
- Memory usage profiling
- Inference speed benchmarks
- Batch processing efficiency
- GPU utilization

### Phase 7: Production Features (Priority: MEDIUM)

#### 7.1 Monitoring & Observability
- OpenTelemetry integration
- Prometheus metrics
- Structured logging (JSON)
- Distributed tracing

#### 7.2 API Service
- FastAPI-based REST API
- WebSocket support for real-time processing
- OpenAPI documentation
- Rate limiting and authentication

#### 7.3 Deployment Options
- Docker containers with multi-stage builds
- Kubernetes manifests with autoscaling
- Terraform modules for cloud deployment
- Ansible playbooks for on-premise

## Configuration Management

### Environment-Based Configuration
```yaml
# configs/environments/production.yaml
inference:
  device: cuda
  precision: float16
  batch_size: 8
  max_memory_gb: 10

storage:
  provider: cloudflare_r2
  cache_enabled: true
  cache_ttl_hours: 24

monitoring:
  enabled: true
  metrics_port: 9090
  log_level: INFO
```

### Configuration Precedence
1. Command-line arguments
2. Environment variables
3. Configuration files
4. Default values

## Error Handling Strategy

### Error Categories
1. **Recoverable**: Retry with backoff
2. **User Errors**: Clear messages, suggestions
3. **System Errors**: Detailed logging, alerts
4. **Critical Errors**: Graceful shutdown, data preservation

### Error Response Format
```python
class AnonymizationError(Exception):
    def __init__(self, message: str, error_code: str, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
```

## Security Considerations

### Data Security
- Encryption at rest for stored models
- TLS for all network communication
- Secure credential management (HashiCorp Vault, AWS Secrets Manager)
- Audit logging for compliance

### Privacy Features
- Minimal data retention
- On-premise deployment option
- Data residency controls
- GDPR-compliant data handling

## Performance Optimization

### Inference Optimization
- Model quantization (INT8, FP16)
- Batch processing
- GPU memory pooling
- Model caching with LRU eviction

### Scalability
- Horizontal scaling with load balancing
- Queue-based asynchronous processing
- Auto-scaling based on metrics
- Multi-region deployment

## Development Workflow

### Code Quality
- Type hints throughout (mypy strict mode)
- Comprehensive docstrings (Google style)
- Pre-commit hooks (black, ruff, mypy)
- Conventional commits

### CI/CD Pipeline
1. **Linting & Formatting**: black, ruff, mypy
2. **Unit Tests**: pytest with coverage
3. **Integration Tests**: docker-compose based
4. **Security Scanning**: bandit, safety
5. **Performance Tests**: locust for load testing
6. **Deployment**: GitOps with ArgoCD

## Success Metrics

### Technical Metrics
- Inference latency < 2s per page
- Model accuracy > 95% for PII detection
- System uptime > 99.9%
- Memory usage < 4GB per document

### Business Metrics
- Documents processed per hour
- Reduction in manual review time
- Compliance audit pass rate
- User satisfaction score

## Risk Mitigation

### Technical Risks
- **Model Drift**: Regular retraining, A/B testing
- **Performance Degradation**: Continuous monitoring, alerting
- **Data Loss**: Backup strategies, disaster recovery
- **Security Breaches**: Regular audits, penetration testing

### Operational Risks
- **Vendor Lock-in**: Abstraction layers, portable implementations
- **Scaling Issues**: Load testing, capacity planning
- **Knowledge Transfer**: Comprehensive documentation, runbooks

## Conclusion

This specification provides a clear, actionable roadmap for implementing a production-ready document anonymization system. The modular architecture ensures flexibility while maintaining high performance and reliability standards. Priority should be given to the core inference pipeline and CLI functionality to deliver a working system quickly, followed by enhanced features for production deployment.
