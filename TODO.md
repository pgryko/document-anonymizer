# TODO: Document Anonymization System

**Last Updated:** 2025-08-07

## 📊 CURRENT STATUS

**Significant progress made, but key components still need work:**

- ✅ **VAE Training** fully implemented with corrected hyperparameters
- ✅ **UNet Training** fully implemented with training loop
- ⚠️ **InferenceEngine** partially complete - OCR integration needs work
- ✅ **UNet Dataset** implemented - `InpaintingDataset` available and wired via `create_inpainting_dataloaders`
- ❌ **Test Infrastructure** has issues - tests timing out
- ✅ **Security hardening** with path validation and secure file handling
- ✅ **Configuration improvements** with project-relative paths

---

### Critical Issues Found (RESOLVED)

**1. Missing Core Implementation**
- `InferenceEngine.anonymize()` is completely empty - the main functionality doesn't exist
- Training loops don't integrate with data loading (commented out dataloader usage)
- No NER integration despite being core to the anonymization pipeline

**2. Import/Dependency Issues**
- Missing `utils/metrics.py:MetricsCollector` class causes import failures
- Conditional torchvision imports could silently disable critical perceptual loss
- Missing dataset implementations for training pipeline

**3. Security Vulnerabilities**
- No path validation against directory traversal attacks
- Hardcoded temporary paths without secure handling
- Error messages potentially leak system information
- Font loading from system without validation

**4. Technical Debt**
- Pydantic v1 syntax (`@validator`) deprecated in favor of v2 (`@field_validator`)
- Hardcoded paths (`/tmp/checkpoints`) reduce portability
- Broad exception catching masks specific errors

Use structlog for logging

---

## HIGH PRIORITY (Blocking) ⚠️

### 1. CRITICAL: Confidence Scoring and Quality Verification
**File**: `src/anonymizer/inference/engine.py`
**Status**: 🔄 Partially Complete
**Description**:
- Compute non-hardcoded confidence for `GeneratedPatch` based on model outputs and mask fidelity
- Add optional SSIM/LPIPS-based verification gate controlled by config

### 2. HIGH: Batch Inference Optimization
**File**: `src/anonymizer/inference/engine.py`
**Status**: ❌ Pending
**Description**:
- Group multiple regions by image into batched inpainting calls when feasible
- Profile memory/latency; expose `max_regions_per_batch` in config

### 3. CRITICAL: Fix Test Infrastructure
**File**: Various test files
**Status**: ❌ Broken
**Description**: 
- ❌ Tests timing out when running
- ❌ Cannot measure test coverage accurately
- ❌ Need to debug test execution issues
- ❌ May have performance problems in test setup

### 4. SECURITY: Path validation and temp files
**File**: `src/anonymizer/core/config.py`, `src/anonymizer/inference/engine.py`
**Status**: ✅ Completed
**Description**: 
- ✅ Added comprehensive path validation functions
- ✅ Implemented secure path resolution with bounds checking
- ✅ Added validation to all configuration classes
- ✅ Included model path security validation

---

## MEDIUM PRIORITY (Quality/Security) 🔒

### 5. SECURITY: Implement secure temporary file handling
**Status**: ✅ Completed
**Description**: 
- ✅ Used Python's `tempfile` module for secure temp files in InferenceEngine
- ✅ Added proper cleanup mechanisms for temporary data
- ✅ Implemented secure file permissions and validation

### 6. BUG: Fix pydantic validator deprecation warnings
**File**: `src/anonymizer/core/models.py:17`
**Status**: ✅ Completed
**Description**: 
- ✅ Migrated from `@validator` to `@field_validator` (Pydantic v2)
- ✅ Updated validation syntax throughout codebase
- ✅ Tested validation behavior after migration

### 7. BUG: Add error handling for font loading failures
**File**: `src/anonymizer/training/unet_trainer.py:40`
**Status**: ✅ Completed
**Description**: 
- ✅ Improved font fallback mechanisms in TextRenderer
- ✅ Added proper error messages for missing fonts
- ✅ Implemented secure font loading with validation

### 8. IMPROVEMENT: Make hardcoded paths configurable
**Status**: ✅ Completed
**Description**: 
- ✅ Changed hardcoded `/tmp/checkpoints` to `./checkpoints`
- ✅ Updated all configs to use project-relative paths as defaults
- ✅ Added path validation to all configurable paths

### 9. IMPROVEMENT: Fix and expand test suite
**Status**: ❌ Blocked by infrastructure issues
**Description**: 
- ❌ Test infrastructure broken (timeouts)
- ❌ Cannot run tests to measure coverage
- ❌ End-to-end integration tests needed
- ❌ Inference pipeline integration tests needed

---

## LOW PRIORITY (Polish) ✨

### 10. IMPROVEMENT: Add CLI tests using click.testing
**File**: `main.py`
**Status**: ❌ Pending
**Description**: 
- Test all CLI commands and options
- Verify error handling in CLI
- Test configuration file loading

### 11. IMPROVEMENT: Bundle fonts for reproducibility
**File**: `src/anonymizer/utils/text_rendering.py`
**Status**: ❌ Pending
**Description**: 
- Include specific fonts in project assets
- Update font loading to use bundled fonts first
- Document font requirements clearly

### 12. IMPROVEMENT: Add type hints for better IDE support
**Status**: ❌ Pending
**Description**: 
- Complete type annotations in trainer classes
- Add return type hints to all public methods
- Use `typing.Protocol` for interfaces where appropriate

---

## Security Review Notes 🔐

**Critical Security Issues Identified:**

1. **File Path Security**: The codebase uses user-provided paths without proper validation for path traversal attacks
2. **Memory Management**: No resource limits in some areas could lead to DoS via memory exhaustion
3. **Input Validation**: While pydantic provides good validation, there are gaps in handling malicious image data
4. **Error Information Disclosure**: Some error messages might leak system information
5. **Font Loading**: System font loading could be exploited if fonts contain malicious code
6. **Temporary Files**: No secure handling of temporary files during processing

---

## Next Steps Recommendation

**Priority Order for Remaining Work:**

1. **Fix test infrastructure** - Debug timeout issues to enable proper testing
2. **Batch inference** - Implement region batching for inpainting
3. **Improve test coverage** - Once tests run, expand coverage to 80%+
4. **Integration testing** - Add end-to-end tests for full pipeline
5. **HF Hub upload** - Implement optional upload in Modal training functions

## Recent Discoveries (2025-08-06)

- UNet trainer is MORE complete than documented - has full training loop
- Test infrastructure has critical issues preventing execution
- OCR TODOs remain unresolved at specific line numbers
- Main gap is dataset implementation, not trainer logic