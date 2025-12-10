# Code Review Action Items

**Last Updated:** 2025-08-06
**Status:** Revised based on current codebase analysis

## Critical Issues (P0)

### 1. Confidence Scoring and Verification
**File:** `src/anonymizer/inference/engine.py`
**Issue:** Hardcoded confidence value for `GeneratedPatch`
**Action:**
- Derive confidence from model outputs and verification metrics (e.g., SSIM/LPIPS)
- Add `enable_quality_check` gate (already present) to enforce minimal confidence
**Estimated effort:** 1-2 days

### 2. Batch Inpainting Optimization
**File:** `src/anonymizer/inference/engine.py`
**Issue:** Sequential per-region calls
**Action:**
- Group compatible regions to reduce invocations
- Measure memory; expose batched parameters in `EngineConfig`
**Estimated effort:** 2-3 days

## High Priority (P1)

### 3. Fix Test Infrastructure & Improve Coverage
**Current:** Tests timing out, coverage unmeasurable
**Target:** Fix test execution, then achieve 80% coverage
**Actions:**
- Debug test timeout issues
- Fix test infrastructure problems
- Once tests run, add unit tests for:
  - `VAETrainer` class
  - `InferenceEngine` class
  - Configuration validation
  - Security path validation
- Add integration tests for full pipeline
- Add performance benchmarks
**Estimated effort:** 1-2 days for fixes, then 5-7 days for coverage

### 4. Add Missing Error Handling
**Files:** Multiple
**Actions:**
- Add timeout handling for OCR operations
- Implement retry logic for model loading
- Add graceful degradation for missing components
**Estimated effort:** 2 days

## Medium Priority (P2)

### 5. Documentation Updates
**Actions:**
- Update README/docs with `InferenceEngine` usage (done)
- Correct CLI invocations to `python main.py ...` (done)
- Document config loading via `AppConfig`
- Add examples for programmatic usage (done)
**Estimated effort:** 2-3 days

### 6. Performance Optimizations
**Actions:**
- Implement model caching
- Add batch optimization for inference
- Profile and optimize memory usage
- Consider model quantization
**Estimated effort:** 3-4 days

### 7. Configuration Improvements
**Actions:**
- Add configuration validation command
- Implement configuration migration tool
- Add configuration templates
**Estimated effort:** 2 days

## Low Priority (P3)

### 8. Monitoring Enhancements
**Actions:**
- Add Prometheus metrics export
- Implement health check endpoint
- Create Grafana dashboards
- Add alerting rules
**Estimated effort:** 3 days

### 9. CLI Improvements
**Actions:**
- Add progress bars for long operations
- Implement resume capability for batch processing
- Add dry-run mode
**Estimated effort:** 2 days

### 10. Code Cleanup
**Actions:**
- Remove TODO comments after implementation
- Consolidate duplicate error handling
- Refactor long methods
- Add type hints where missing
**Estimated effort:** 1-2 days

## Security Enhancements

### 11. Additional Security Measures
**Actions:**
- Add rate limiting for inference
- Implement request validation
- Add audit logging
- Create security test suite
**Estimated effort:** 3 days

## Technical Debt

### 12. Dependency Updates
**Actions:**
- Update to latest diffusers version
- Check for security vulnerabilities
- Update development dependencies
- Test compatibility
**Estimated effort:** 1 day

### 13. Code Organization
**Actions:**
- Move constants to dedicated module
- Create factory classes for model initialization
- Separate business logic from infrastructure
**Estimated effort:** 2 days

## Testing Strategy

### Unit Tests Priority
1. Configuration validation
2. Security functions (path validation)
3. Core business logic (anonymization)
4. Training components
5. Utility functions

### Integration Tests Priority
1. End-to-end anonymization
2. Batch processing
3. Model loading and inference
4. OCR + NER pipeline
5. Cloud training workflow

### Performance Tests
1. Inference latency benchmarks
2. Memory usage profiling
3. Batch processing throughput
4. Model loading time
5. Concurrent request handling

## Implementation Order

### Phase 1 (Week 1-2)
- Fix test infrastructure (P0)
- Create UNet dataset implementation (P0)
- Fix OCR integration (P0)
- Start test coverage improvement (P1)

### Phase 2 (Week 3-4)
- Continue test coverage (P1)
- Add error handling (P1)
- Update documentation (P2)

### Phase 3 (Week 5-6)
- Performance optimizations (P2)
- Configuration improvements (P2)
- Security enhancements

### Phase 4 (Week 7-8)
- Monitoring setup (P3)
- CLI improvements (P3)
- Code cleanup and technical debt

## Success Metrics

1. Test coverage > 80%
2. All P0 and P1 issues resolved
3. Documentation complete and accurate
4. Performance benchmarks established
5. Security audit passed
6. Production deployment successful

## Review Schedule

- Weekly progress reviews
- Bi-weekly code reviews
- Monthly security audits
- Quarterly performance reviews

## Recent Changes (2025-08-06)

### Completed Items
- ✅ UNet training loop implementation (previously marked incomplete)
- ✅ Full UNet trainer with validation and checkpointing
- ✅ Text conditioning with TrOCR

### Newly Identified Issues
- Test infrastructure problems causing timeouts
- Missing UNet-specific dataset implementation
- OCR TODOs remain at lines 158, 166 in engine.py

### Status Updates
- UNet trainer is more complete than previously documented
- Main gap is dataset implementation, not trainer logic
- Test coverage cannot be accurately measured until infrastructure fixed
