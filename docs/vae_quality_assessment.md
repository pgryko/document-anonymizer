# VAE Quality Assessment Guide

This guide provides multiple approaches for visually assessing the quality of your VAE training for document anonymization.

## Quick Start Recommendations

### 1. **Start with Simple Visualization** (5 minutes)
```bash
# Quick reconstruction quality check
poetry run python scripts/visualize_vae_quality.py \
    --config configs/training/vae_config_local.yaml \
    --num-samples 8 \
    --save-path outputs/vae_quality.png
```

### 2. **Interactive Dashboard** (Best for exploration)
```bash
# Launch Streamlit dashboard
poetry run streamlit run scripts/vae_dashboard.py
```

### 3. **TensorBoard Monitoring** (During training)
```bash
# View training logs
poetry run tensorboard --logdir outputs/checkpoints/tensorboard
```

## Detailed Assessment Approaches

### Option 1: Matplotlib Visualization Script ⭐ **Recommended First Step**

**Pros:**
- Quick and simple
- No additional dependencies
- Good for initial assessment
- Saves results for sharing

**Usage:**
```bash
# Basic usage
poetry run python scripts/visualize_vae_quality.py \
    --config configs/training/vae_config_local.yaml

# With trained model
poetry run python scripts/visualize_vae_quality.py \
    --config configs/training/vae_config_local.yaml \
    --checkpoint outputs/checkpoints/vae_step_1000.safetensors \
    --num-samples 12 \
    --analyze-latents

# Save results
poetry run python scripts/visualize_vae_quality.py \
    --config configs/training/vae_config_local.yaml \
    --save-path reports/vae_quality_epoch_10.png
```

**What it shows:**
- Original vs reconstructed images side-by-side
- Pixel-wise difference heatmaps
- MSE loss per image
- Latent space statistics and distributions

### Option 2: Streamlit Interactive Dashboard ⭐ **Best for Exploration**

**Pros:**
- Interactive exploration
- Real-time metric computation
- Batch analysis capabilities
- Professional-looking interface
- Easy to share with team

**Installation:**
```bash
poetry add streamlit plotly
```

**Usage:**
```bash
poetry run streamlit run scripts/vae_dashboard.py
```

**Features:**
- Browse through individual samples
- Interactive metric computation (MSE, MAE, PSNR, SSIM)
- Visual comparison with difference maps
- Batch analysis with distribution plots
- Sample metadata display

### Option 3: TensorBoard Integration ⭐ **Best for Training Monitoring**

**Pros:**
- Integrated with training loop
- Historical tracking
- Scalable for long training runs
- Standard ML monitoring tool

**Usage:**
The VAE trainer now includes TensorBoard logging. To enable:

```python
# In your training script
trainer.setup_tensorboard()  # Optional: specify log_dir

# During training, log reconstructions every N steps
if step % 100 == 0:
    trainer.log_reconstructions(batch, step)
```

**View logs:**
```bash
poetry run tensorboard --logdir outputs/checkpoints/tensorboard
```

### Option 4: Jupyter Notebook Analysis

**Pros:**
- Flexible experimentation
- Easy to document findings
- Good for detailed analysis
- Shareable reports

**Setup:**
```bash
poetry add jupyter ipywidgets
poetry run jupyter notebook
```

Create a notebook with:
- Load model and data
- Interactive widgets for sample selection
- Detailed metric analysis
- Custom visualizations

## Assessment Criteria

### Key Metrics to Monitor

1. **Reconstruction Quality:**
   - MSE (Mean Squared Error): Lower is better
   - PSNR (Peak Signal-to-Noise Ratio): Higher is better (>20 dB is good)
   - SSIM (Structural Similarity): Higher is better (>0.8 is good)

2. **Text Preservation:**
   - Visual inspection of text regions
   - Edge preservation in text areas
   - Font clarity and readability

3. **Latent Space Quality:**
   - Latent distribution should be close to standard normal
   - No mode collapse (check latent diversity)
   - Smooth interpolation between samples

### Red Flags to Watch For

❌ **Poor Quality Indicators:**
- Blurry or distorted text
- High MSE (>0.1) or low PSNR (<15 dB)
- Latent values with extreme ranges
- Inconsistent reconstruction quality across samples

✅ **Good Quality Indicators:**
- Sharp, readable text in reconstructions
- Low MSE (<0.05) and high PSNR (>20 dB)
- Latent distributions close to N(0,1)
- Consistent quality across different document types

## Workflow Recommendations

### During Development:
1. **Quick Check:** Use matplotlib script after each training run
2. **Interactive Exploration:** Use Streamlit dashboard for detailed analysis
3. **Training Monitoring:** Enable TensorBoard logging

### For Production Assessment:
1. **Batch Analysis:** Run dashboard batch analysis on validation set
2. **Quantitative Metrics:** Collect metrics across large sample sets
3. **Qualitative Review:** Manual inspection of edge cases

### For Team Collaboration:
1. **Streamlit Dashboard:** Easy to share and demonstrate
2. **Saved Visualizations:** Include in reports and presentations
3. **TensorBoard Logs:** Share training progress and results

## Alternative Tools Consideration

### Label Studio
**Not recommended for this use case** because:
- Designed for data annotation, not quality assessment
- Overkill for reconstruction evaluation
- Would require custom integration

**Better suited for:**
- Annotating training data quality issues
- Creating ground truth for evaluation metrics
- Manual quality scoring workflows

### Other Options:
- **Weights & Biases (wandb):** Good alternative to TensorBoard
- **MLflow:** For experiment tracking and model registry
- **Gradio:** Alternative to Streamlit for quick demos

## Next Steps

1. **Start Simple:** Begin with the matplotlib visualization script
2. **Scale Up:** Move to Streamlit dashboard for detailed analysis
3. **Integrate:** Add TensorBoard logging to your training loop
4. **Automate:** Create quality assessment as part of your training pipeline

Choose the approach that best fits your workflow and requirements!
