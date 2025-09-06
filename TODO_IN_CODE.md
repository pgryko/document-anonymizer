# TODO Comments in Codebase

**Last Updated:** 2025-08-07  
This document tracks all TODO/FIXME comments found in the source code.

## High Priority

### 1. Confidence Scoring for Generated Patches
**File:** `src/anonymizer/inference/engine.py`
- **Location:** `_anonymize_region`
- **Description:** Confidence is currently hardcoded (0.9). Replace with a score derived from model latents, mask coverage, or SSIM/LPIPS versus inpaint context.
- **Impact:** High; improves reporting and downstream decision-making

## Medium Priority

### 2. HuggingFace Hub Upload
**File:** `src/anonymizer/cloud/modal_app.py`
- **Lines:** Push-to-hub branches in `train_vae`, `train_unet`
- **Description:** Implement model upload using `huggingface_hub`, including auth, model card, and private visibility toggle.
- **Impact:** Medium; simplifies distribution and reproducibility

### 3. CLI Test Coverage
**File:** `main.py`
- **Description:** Add tests with `click.testing.CliRunner` for `anonymize`, `batch-anonymize`, `batch-status`, `train-vae`, `train-unet`.
- **Impact:** Medium; improves reliability of user interface

## Implementation Guidelines

### For Confidence Scoring:
1. Compute per-region SSIM/LPIPS between inpainted region and context
2. Use mask coverage and OCR/NER confidence min as base prior
3. Expose `min_confidence_threshold` in `EngineConfig` (already present)
4. Log score components for observability

### For HF Hub Upload:
1. Add `huggingface_hub` as optional dependency
2. Implement upload with `HfApi().upload_folder()`
3. Use Modal secret for token and support private repos

### For HuggingFace Hub:
1. Add `huggingface_hub` dependency
2. Implement upload method using HF API
3. Add authentication handling
4. Include model cards and metadata

## Tracking

These TODOs should be converted to GitHub issues for proper tracking and assignment.

## Status Notes

- All TODO comments verified as of 2025-08-06
- Line numbers updated to reflect current code positions
- No new TODOs found in recent code changes