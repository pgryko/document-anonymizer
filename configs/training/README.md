# Training Configurations

This directory contains training configurations optimized for different environments.

## Configuration Types

### Local Configurations (Mac/Development)
- `vae_config_local.yaml` - VAE training for local Mac testing
- `unet_config_local.yaml` - UNet training for local Mac testing

**Optimized for:**
- MPS (Metal Performance Shaders) backend on Mac
- Limited memory constraints
- Fast iteration and debugging
- Minimal resource usage

**Key Settings:**
- Batch size: 1 (VAE), 1 (UNet)
- Mixed precision: Disabled for compatibility
- Epochs: 2 (for quick testing)
- Image crop size: 256px (automatically set)
- Frequent checkpointing for debugging

### Cloud Configurations (GPU Production)
- `vae_config_cloud.yaml` - VAE training for cloud GPU
- `unet_config_cloud.yaml` - UNet training for cloud GPU

**Optimized for:**
- CUDA GPU training
- High memory and compute resources
- Production-quality training
- Efficient GPU utilization

**Key Settings:**
- Batch size: 16 (VAE), 8 (UNet)
- Mixed precision: fp16 enabled
- Epochs: 100 (VAE), 50 (UNet)
- Image crop size: 512px
- Less frequent checkpointing

## Usage

### Local Testing/Debugging
```bash
# Test VAE training locally
uv run python main.py train-vae --config configs/training/vae_config_local.yaml

# Test UNet training locally
uv run python main.py train-unet --config configs/training/unet_config_local.yaml
```

### Cloud Production Training
```bash
# Train VAE in cloud
python main.py train-vae --config configs/training/vae_config_cloud.yaml

# Train UNet in cloud
python main.py train-unet --config configs/training/unet_config_cloud.yaml
```

## Memory Management

### Local (Mac MPS)
- Uses conservative memory settings
- Automatic MPS cache clearing
- Smaller batch sizes and image sizes
- No multiprocessing for data loading

### Cloud (CUDA)
- Optimized for high-memory GPUs
- Mixed precision training
- Larger batch sizes for efficiency
- Multiprocessing data loading

## Key Differences Summary

| Setting | Local | Cloud | Reason |
|---------|-------|-------|--------|
| Batch Size (VAE) | 1 | 16 | Memory constraints vs efficiency |
| Batch Size (UNet) | 1 | 8 | Memory constraints vs efficiency |
| Mixed Precision | No | fp16 | Compatibility vs speed |
| Epochs | 2 | 100/50 | Quick testing vs full training |
| Image Size | 256px | 512px | Memory vs quality |
| Checkpointing | Every 50/25 steps | Every 5000/2500 steps | Debug vs production |
| Data Workers | 0 | 4 | Debugging vs performance |

## Environment Detection

The training script automatically detects if you're using a local configuration (by checking if "local" is in the config filename) and adjusts:
- Image crop size
- Number of data loading workers
- Memory management strategies

## Troubleshooting

### Local Training Issues
- **Out of memory**: Reduce batch size to 1 or use smaller crop size
- **Slow data loading**: Ensure num_workers is 0 for debugging
- **MPS errors**: Try setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0`

### Cloud Training Issues
- **GPU utilization low**: Increase batch size if memory allows
- **Training slow**: Enable mixed precision (fp16)
- **Convergence issues**: Check learning rates and warmup steps
