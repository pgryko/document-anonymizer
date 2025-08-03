# TODO Comments in Codebase

This document tracks all TODO/FIXME comments found in the source code.

## High Priority

### 1. OCR Integration for PII Detection
**File:** `src/anonymizer/inference/engine.py`
- **Line 143:** `# TODO: integrate with other methods for pii detection`
- **Line 151:** `# TODO: add OCR to extract bounding boxes`
- **Description:** The NER processor currently creates dummy bounding boxes. This needs to be integrated with actual OCR results to map detected PII entities to their physical locations in the document.
- **Impact:** Critical for accurate document anonymization

## Medium Priority

### 2. Modal Dependencies Installation
**File:** `src/anonymizer/cloud/modal_app.py`
- **Line 36:** `# TODO: this should install via uv and dependencies in pyproject.toml`
- **Description:** The Modal image creation should use uv package manager and install from pyproject.toml instead of hardcoded pip install commands.
- **Impact:** Better dependency management and consistency

### 3. HuggingFace Hub Integration
**File:** `src/anonymizer/cloud/modal_app.py`
- **Line 205:** `# TODO: Implement HuggingFace Hub upload` (VAE model)
- **Line 340:** `# TODO: Implement HuggingFace Hub upload` (UNet model)
- **Description:** Model upload to HuggingFace Hub is not implemented. This would allow easier model sharing and deployment.
- **Impact:** Nice to have for model distribution

## Implementation Guidelines

### For OCR Integration:
1. Modify `detect_pii()` method to accept OCR results
2. Map NER-detected entities to OCR bounding boxes
3. Handle cases where multiple OCR boxes contain a single entity
4. Add confidence thresholds for matching

### For Modal Dependencies:
1. Create a `requirements.txt` from `pyproject.toml`
2. Or use `uv pip compile` to generate locked dependencies
3. Update Modal image creation to use the generated file

### For HuggingFace Hub:
1. Add `huggingface_hub` dependency
2. Implement upload method using HF API
3. Add authentication handling
4. Include model cards and metadata

## Tracking

These TODOs should be converted to GitHub issues for proper tracking and assignment.