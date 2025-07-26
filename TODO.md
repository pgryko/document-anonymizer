# TODO: Document Anonymization System

### Critical Issues Found

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

### 1. CRITICAL: Implement missing InferenceEngine core logic
**File**: `src/anonymizer/inference/engine.py:11`
**Status**: ❌ Pending
**Description**: 
- Add NER pipeline using presidio or spacy
- Implement VAE/UNet inference pipeline
- Add image composition logic for anonymized patches
- look at src/reference_code/original_diffute for reference

### 2. CRITICAL: Complete data loading pipeline for trainers
**File**: `src/anonymizer/training/`
**Status**: ❌ Pending
**Description**: 
- Create Dataset classes for VAE and UNet training
- Implement DataLoader integration in training loops
- Add data preprocessing and augmentation pipelines
- look at src/reference_code/original_diffute for reference

### 3. CRITICAL: Fix import issues - missing modules
**File**: Various
**Status**: ❌ Pending
**Description**: 
- Implement `MetricsCollector` class in `src/anonymizer/utils/metrics.py:31`
- Make torchvision a required dependency if perceptual loss is critical
- Add missing dataset implementations

### 4. SECURITY: Add path validation against directory traversal
**File**: `src/anonymizer/core/config.py:104`
**Status**: ❌ Pending
**Description**: 
- Validate all user-provided file paths
- Implement secure path resolution
- Add path sanitization functions

---

## MEDIUM PRIORITY (Quality/Security) 🔒

### 5. SECURITY: Implement secure temporary file handling
**Status**: ❌ Pending
**Description**: 
- Use Python's `tempfile` module for secure temp files
- Add cleanup mechanisms for temporary data
- Implement proper file permissions

### 6. BUG: Fix pydantic validator deprecation warnings
**File**: `src/anonymizer/core/models.py:17`
**Status**: ❌ Pending
**Description**: 
- Migrate from `@validator` to `@field_validator` (Pydantic v2)
- Update validation syntax throughout codebase
- Test validation behavior after migration

### 7. BUG: Add error handling for font loading failures
**File**: `src/anonymizer/training/unet_trainer.py:40`
**Status**: ❌ Pending
**Description**: 
- Improve font fallback mechanisms
- Add proper error messages for missing fonts
- Consider bundling default fonts with the project

### 8. IMPROVEMENT: Make hardcoded paths configurable
**Status**: ❌ Pending
**Description**: 
- Move `/tmp/checkpoints` and other paths to config
- Use project-relative paths as defaults
- Add CLI options for common path overrides

### 9. IMPROVEMENT: Add comprehensive integration tests
**Status**: ❌ Pending
**Description**: 
- Test complete training pipeline end-to-end
- Add inference pipeline integration tests
- Test configuration loading and validation

---

## LOW PRIORITY (Polish) ✨

### 10. IMPROVEMENT: Add CLI tests using click.testing
**File**: `main.py:34`
**Status**: ❌ Pending
**Description**: 
- Test all CLI commands and options
- Verify error handling in CLI
- Test configuration file loading

### 11. IMPROVEMENT: Bundle fonts for reproducibility
**File**: `src/anonymizer/utils/text_rendering.py:45`
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

The codebase has excellent bones but needs these critical implementations before it can function as intended. Start with HIGH PRIORITY items in order:

1. Implement the missing `MetricsCollector` class to fix import errors
2. Complete the `InferenceEngine` implementation 
3. Add data loading pipeline for training
4. Address security vulnerabilities

The configuration and architectural patterns are solid foundations to build upon.