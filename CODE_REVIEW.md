# Code Review

This document contains a code review of the `document-anonymizer` project.

## Overall Assessment

The `document-anonymizer` project is a well-engineered and robust application. The code is well-structured, follows best practices, and includes a comprehensive set of features. The project's strengths include its security-first design, comprehensive error handling, and support for distributed training.

However, there are a few areas where the project could be improved. These include dependency management, commented-out code, hardcoded paths, and a lack of unit tests for some components.

## Findings and Recommendations

### High-Priority

*   **Public API alignment:** Documentation previously referenced `DocumentAnonymizer` with `anonymize_document`, but the production path uses `InferenceEngine.anonymize(...)` and the Click-based CLI in `main.py`. Docs updated; ensure tests and examples consistently use the current API.
*   **Confidence scoring:** `_anonymize_region` returns a hardcoded confidence. Replace with a computed score (model outputs + verification) and reflect in `AnonymizationResult`.
*   **Batching opportunities:** Region-wise sequential inpainting can be slow on text-dense images. Explore batching or partial tiling strategies with memory guards.

### Medium-Priority

*   **Dependency hygiene:** Audit `pyproject.toml` for duplicates and move optional stacks (Modal, W&B, HF Hub) behind extras.
*   **Type checking strictness:** Reduce blanket `ignore_missing_imports`; target only problematic libs.
*   **Fonts:** Prefer bundled, validated fonts via `fonts/` and `FontManager` before system lookup.

### Low-Priority

*   **CLI structure:** Consider extracting CLI subcommands to `src/anonymizer/cli.py` to slim `main.py`.
*   **Perceptual loss config:** Expose backbone choice for VAE perceptual loss.
*   **Import cleanup:** Remove redundant imports flagged by linter.
*   **Trainer tests:** Add focused unit tests for `VAETrainer`/`UNetTrainer` critical logic.