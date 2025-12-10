# Action Items from Code Review

## High Priority Security Issues

### 1. Fix hardcoded secrets in R2Config
**Location**: `src/anonymizer/core/config.py:348-349`
**Issue**: Access keys stored directly in config objects
**Action**: Move to secure credential manager (AWS Secrets Manager, HashiCorp Vault, or environment-only)
**Priority**: HIGH

### 2. Improve path validation security
**Location**: `src/anonymizer/core/config.py:35-47`, `src/anonymizer/inference/engine.py:348,355`
**Issue**: Uses blacklist approach, hardcoded `/tmp` path
**Action**:
- Implement whitelist of allowed directories
- Make base directory configurable via environment variable
- Add protection against symlinks and path normalization attacks
**Priority**: HIGH

### 3. Fix hardcoded /tmp path in inference engine
**Location**: `src/anonymizer/inference/engine.py:348,355`
**Issue**: Hardcoded paths create security risks
**Action**: Make paths configurable through EngineConfig
**Priority**: HIGH

### 4. Add secure temporary file creation
**Location**: `src/anonymizer/inference/engine.py:508`
**Issue**: Temporary files created with default permissions
**Action**: Set secure permissions (0600) and proper cleanup
**Priority**: HIGH

### 5. Fix test mocking inconsistency
**Location**: `tests/unit/test_batch_processing.py`
**Issue**: Tests mock `anonymize_document` method but actual method is `anonymize`
**Action**: Update test mocks to use correct method name
**Priority**: HIGH

## Medium Priority Issues

### 6. Fix failing e2e tests
**Location**: `tests/integration/test_e2e_anonymization.py`
**Issue**: 3 tests failing - wrong exception types being raised
**Action**: Ensure proper `InferenceError` is raised for invalid/empty images
**Priority**: MEDIUM

### 7. Remove commented-out code
**Location**: `main.py` anonymize command
**Issue**: Several lines of commented-out code present
**Action**: Remove or properly implement commented-out functionality
**Priority**: MEDIUM

### 8. Make hardcoded data paths configurable
**Location**: `main.py` train_vae command
**Issue**: Hardcoded paths for training and validation data
**Action**: Make data paths configurable via command line arguments or config
**Priority**: MEDIUM

### 9. Improve test coverage
**Current**: 61% coverage
**Target**: 80%+ coverage
**Focus Areas**: OCR modules (22%), model management (38%), validators (41%)
**Priority**: MEDIUM

### 10. Add input validation for font loading
**Location**: `src/anonymizer/inference/engine.py:182-186`
**Issue**: Font files loaded without validation
**Action**: Validate font files before loading, implement sandboxed rendering
**Priority**: MEDIUM

### 11. Add comprehensive logging for production
**Issue**: Insufficient logging for production debugging
**Action**: Add structured logging throughout inference pipeline
**Priority**: MEDIUM

## Low Priority Improvements

### 12. Document corrected hyperparameters
**Issue**: README doesn't highlight the critical bug fixes
**Action**: Document VAE (5e-4 LR) and UNet (1e-4 LR) fixes prominently
**Priority**: LOW

### 13. Remove inefficient model_post_init reloading
**Location**: `src/anonymizer/core/config.py:415`
**Issue**: Unnecessary reloading of nested configs in AppConfig
**Action**: Optimize configuration loading to avoid redundant operations
**Priority**: LOW

## Implementation Order

1. **Week 1**: Security fixes (#1-5) - Critical for production readiness
2. **Week 2**: Test improvements (#6-9) - Ensure reliability
3. **Week 3**: Production hardening (#10-11) - Operational readiness
4. **Week 4**: Documentation and optimization (#12-13) - Final polish

## Security Testing Checklist

- [ ] Test directory traversal attacks on path validation
- [ ] Verify secrets are not logged or serialized
- [ ] Test file permission on temporary files
- [ ] Validate font loading security
- [ ] Test input sanitization for all user data
- [ ] Verify OCR engine security (especially with untrusted images)
- [ ] Test resource limits and DoS protection

## Notes

- All security issues should be addressed before production deployment
- Consider adding security scanning to CI/CD pipeline
- Implement monitoring for security events
- Create incident response plan for security issues
