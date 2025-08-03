# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Development Commands
```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run specific test types
uv run pytest -m unit          # Unit tests only
uv run pytest -m integration   # Integration tests only
uv run pytest -m gpu          # GPU tests only
uv run pytest -m "not slow"   # Skip slow tests

# Run tests with coverage
uv run pytest --cov=src/anonymizer --cov-report=html

# Code formatting and linting
uv run black src/ tests/
uv run ruff check src/ tests/
uv run mypy src/

# Run single test file
uv run pytest tests/test_core_config.py -v
```

### Training Commands
```bash
# Train VAE model
uv run python main.py train-vae --config configs/training/vae_config.yaml

# Train UNet model  
uv run python main.py train-unet --config configs/training/unet_config.yaml

# Run anonymization
uv run python main.py anonymize --config configs/inference/engine_config.yaml --image input.jpg --output output.jpg

# Batch anonymization
uv run python main.py batch-anonymize --input-dir ./images --output-dir ./output --config configs/inference/engine_config.yaml
```

## Architecture Overview

This is a document anonymization system using diffusion models and Named Entity Recognition (NER) to replace PII in financial documents while preserving document structure.

### Core Components

**Configuration System** (`src/anonymizer/core/config.py`):
- Pydantic-based configuration with environment variable support
- Hierarchical configs: AppConfig contains VAEConfig, UNetConfig, EngineConfig
- YAML file support with env var overrides
- Critical hyperparameter fixes from reference implementations (proper learning rates, KL divergence)

**Training Pipeline** (`src/anonymizer/training/`):
- `vae_trainer.py`: VAE training with KL divergence loss (critical bug fix) - FULLY IMPLEMENTED
- `unet_trainer.py`: UNet training for inpainting - PARTIALLY IMPLEMENTED (needs dataloader and training loop)
- Distributed training support via Accelerate
- Perceptual loss for text preservation
- Memory-efficient training with gradient accumulation

**Inference Engine** (`src/anonymizer/inference/engine.py`):
- Production-ready implementation with comprehensive security
- Secure path validation and memory management
- Integration with OCR (PaddleOCR, EasyOCR, Tesseract) and NER (Presidio)
- Thread-safe operations with proper resource cleanup
- Note: OCR bounding box extraction for NER needs completion (line 151-152)

**Utilities** (`src/anonymizer/utils/`):
- `image_ops.py`: Image processing operations
- `metrics.py`: Training and inference metrics
- `text_rendering.py`: Text rendering utilities

### Key Design Patterns

1. **Configuration-First**: All components use pydantic configs with validation
2. **Error Handling**: Custom exception hierarchy in `core/exceptions.py`
3. **Modular Training**: Separate trainers for VAE and UNet components
4. **Memory Management**: Built-in memory limits and efficient attention mechanisms
5. **Testing**: Comprehensive test suite with GPU/integration/unit markers (NOTE: Current coverage is 19% - needs improvement)

### Critical Fixes from Reference Implementation

The codebase implements fixes for major bugs in the original DiffUTE research:
- VAE learning rate increased from 5e-6 to 5e-4
- Added missing KL divergence loss in VAE training
- Proper batch sizes (16 for VAE, 8 for UNet)
- Perceptual loss for better text preservation

### Configuration Files

- `configs/training/vae_config.yaml`: VAE training parameters with bug fixes
- `configs/training/unet_config.yaml`: UNet training configuration  
- `configs/inference/engine_config.yaml`: Inference engine settings

Environment variables can override any config with prefixes: `VAE_`, `UNET_`, `ENGINE_`, `APP_`, etc.

## Known Issues and TODOs

### High Priority
1. **UNet Training**: Dataloader and training loop need to be implemented in `unet_trainer.py`
2. **OCR Integration**: Bounding box extraction for NER results needs completion (engine.py:151-152)
3. **Test Coverage**: Currently at 19%, needs significant improvement

### Medium Priority
1. **Documentation**: Some API documentation and deployment guides need updating
2. **Performance**: Model caching and batch optimization could improve inference speed
3. **Error Handling**: Timeout handling for OCR operations needs implementation

### Low Priority
1. **Monitoring**: Prometheus metrics and health checks could be added
2. **CLI**: Progress bars and dry-run mode would improve user experience
3. **Code Cleanup**: Some TODO comments need to be addressed

See `CODE_REVIEW_COMPREHENSIVE.md` and `CODE_REVIEW_ACTION_ITEMS.md` for detailed analysis and action plans.