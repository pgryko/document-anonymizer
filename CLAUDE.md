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
- `vae_trainer.py`: VAE training with KL divergence loss (critical bug fix)
- `unet_trainer.py`: UNet training for inpainting
- Distributed training support via Accelerate
- Perceptual loss for text preservation
- Memory-efficient training with gradient accumulation

**Inference Engine** (`src/anonymizer/inference/engine.py`):
- Currently minimal implementation (placeholder)
- Designed for document anonymization pipeline

**Utilities** (`src/anonymizer/utils/`):
- `image_ops.py`: Image processing operations
- `metrics.py`: Training and inference metrics
- `text_rendering.py`: Text rendering utilities

### Key Design Patterns

1. **Configuration-First**: All components use pydantic configs with validation
2. **Error Handling**: Custom exception hierarchy in `core/exceptions.py`
3. **Modular Training**: Separate trainers for VAE and UNet components
4. **Memory Management**: Built-in memory limits and efficient attention mechanisms
5. **Testing**: Comprehensive test suite with GPU/integration/unit markers

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