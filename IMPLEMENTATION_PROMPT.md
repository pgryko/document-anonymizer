# Implementation Prompt for Clean Document Anonymization System

Use this prompt in a new Claude session to begin implementing the clean document anonymization system:

---

## **Implementation Request**

I need you to implement a clean, production-ready document anonymization system based on the detailed analysis and plan provided. You have access to:

1. **Reference Implementations**: Two existing implementations with known critical bugs
2. **Comprehensive Analysis**: Detailed bug analysis and architectural review  
3. **Clean Implementation Plan**: Complete technical blueprint for the new system
4. **Updated Documentation**: Fully documented issues in reference code

## **Context & Background**

**Project**: Document anonymization using DiffUTE (Diffusion-based Universal Text Editing) with Stable Diffusion 2.0 inpainting for replacing sensitive text in financial documents.

**Current Status**: 
- Two reference implementations exist but contain critical training bugs
- Missing KL divergence loss in VAE training (most critical issue)
- Inference bugs in production implementation  
- Architecture is fundamentally sound (uses SD 2.0 inpainting correctly)

**Critical Issues Identified**:
1. **Missing KL Divergence Loss** - VAE training mathematically incorrect
2. **Suboptimal Hyperparameters** - Learning rates 10-100x too low
3. **Implementation Bugs** - Memory leaks, coordinate errors, initialization issues
4. **Security Issues** - Hard-coded credentials, no input validation

## **Available Resources**

**Key Files You Should Examine**:
- `CLEAN_IMPLEMENTATION_PLAN.md` - Complete technical blueprint
- `src/reference_code/README.md` - Overview of existing implementations  
- `src/reference_code/ARCHITECTURE_OVERVIEW.md` - System architecture details
- `src/reference_code/original_diffute/` - Research implementation (critical bugs documented)
- `src/reference_code/annon_code/` - Production implementation (inference bugs documented)

**All Critical Bugs Are Documented** in the reference code with:
- 🐛 Bug markers at specific line numbers
- Detailed explanations of what's wrong
- Correct implementation examples in comments

## **Implementation Requirements**

**Target Architecture**:
- **Training**: Modal.com GPU training with distributed support
- **Storage**: Cloudflare R2 for model artifacts and datasets  
- **Inference**: Production-ready API with proper error handling
- **Base Model**: Stable Diffusion 2.0 Inpainting (already correct)

**Key Technologies**:
- PyTorch with Diffusers library
- Modal.com for cloud GPU training
- Cloudflare R2 for storage  
- Pydantic for type safety
- Structured logging and monitoring

**Critical Fixes Required**:
1. **Add KL divergence loss** to VAE training (`recon_loss + beta * kl_loss`)
2. **Increase learning rates** (VAE: 5e-4, UNet: 1e-4) 
3. **Increase batch sizes** (VAE: 16+, UNet: 8+ per GPU)
4. **Add perceptual loss** for better text preservation
5. **Fix memory management** and cleanup GPU memory properly
6. **Add comprehensive input validation** and bounds checking

## **Implementation Approach**

**Phase 1: Core Training Logic** (Start Here)
1. Implement corrected VAE trainer with KL divergence loss
2. Implement UNet trainer with proper hyperparameters  
3. Create robust dataset loading and preprocessing
4. Add comprehensive error handling and validation

**Phase 2: Cloud Integration**  
1. Modal.com training integration
2. Cloudflare R2 storage client
3. Distributed training setup
4. Model artifact management

**Phase 3: Production Inference**
1. Safe inference engine with proper preprocessing
2. Memory management and cleanup
3. Quality assurance pipeline
4. API endpoint with validation

**Phase 4: Testing & Deployment**
1. Comprehensive test suite
2. Performance benchmarking  
3. Security hardening
4. Production deployment

## **Specific Instructions**

**Project Structure**: Follow the structure in `CLEAN_IMPLEMENTATION_PLAN.md` exactly:
```
document-anonymizer/
├── src/anonymizer/
│   ├── core/          # Pydantic models, config, exceptions
│   ├── training/      # VAE and UNet trainers  
│   ├── inference/     # Production inference engine
│   ├── storage/       # R2 client and caching
│   └── utils/         # Image ops, text rendering, metrics
├── modal_training/    # Modal.com integration
├── configs/           # YAML configuration files
└── tests/            # Comprehensive test suite
```

**Code Quality Requirements**:
- Full type annotations with Pydantic models
- Comprehensive error handling with custom exceptions
- Structured logging throughout
- Input validation and sanitization
- Memory management and cleanup
- Security best practices

**Mathematical Correctness**:
- VAE loss MUST include KL divergence: `total_loss = recon_loss + beta * kl_loss`
- Use appropriate learning rates and batch sizes from the plan
- Implement perceptual loss for text preservation
- Proper tensor operations with shape validation

## **Success Criteria**

**Training**:
- [ ] VAE training converges with proper KL divergence loss
- [ ] UNet training stable with corrected hyperparameters  
- [ ] Models can be trained efficiently on Modal.com
- [ ] Artifacts stored properly in Cloudflare R2

**Inference**:
- [ ] Safe preprocessing with bounds checking
- [ ] Memory efficient batch processing
- [ ] Quality verification pipeline working
- [ ] Production API with proper error handling

**Quality**:
- [ ] All tests pass (unit, integration, e2e)
- [ ] Security validation complete
- [ ] Performance benchmarks meet requirements
- [ ] Documentation complete and accurate

## **Getting Started**

1. **Read the Plans**: Start by thoroughly reviewing `CLEAN_IMPLEMENTATION_PLAN.md`
2. **Examine Reference Code**: Look at documented bugs in existing implementations
3. **Begin with Training**: Implement the corrected VAE trainer first (most critical)
4. **Follow the Phases**: Work through Phase 1 → 2 → 3 → 4 systematically
5. **Test Continuously**: Add tests as you implement each component

## **Questions to Ask**

If you need clarification during implementation:
1. "Should I prioritize X or Y feature first?"
2. "Is this the correct way to implement [specific component]?"
3. "How should I handle [specific edge case] in the training/inference?"
4. "What's the expected behavior when [specific error condition]?"

Begin implementation with Phase 1, starting with the corrected VAE trainer. The complete plan and all bug documentation is available in the repository - use it as your detailed technical specification.

---

**Start your implementation now with**: "I'll begin implementing the clean document anonymization system following the comprehensive plan. Let me start with Phase 1 by creating the corrected VAE trainer with proper KL divergence loss."