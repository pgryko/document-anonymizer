# Data Directory Structure

This directory contains all datasets used for training and evaluation.

## Organization

```
data/
├── raw/                    # Original, unprocessed data
│   └── xfund/             # XFUND dataset raw files
├── processed/             # Processed, ready-to-use data
│   └── xfund/            # Processed XFUND dataset
│       ├── vae/          # VAE training data
│       └── unet/         # UNET training data
└── cache/                # Cached preprocessed tensors (gitignored)
```

## Data Sources

### XFUND Dataset
- **Source**: https://github.com/doc-analysis/XFUND
- **Description**: Multilingual Form Understanding dataset
- **Usage**: Document anonymization training
- **Download**: Use `scripts/download_xfund_data.py`

## Processing Pipeline

1. **Download**: Raw data is downloaded to `raw/` directory
2. **Process**: Data is processed and saved to `processed/` directory
3. **Cache**: Preprocessed tensors may be cached in `cache/` for faster loading

## Adding New Datasets

When adding new datasets:
1. Download raw data to `raw/<dataset_name>/`
2. Create processing script in `scripts/`
3. Save processed data to `processed/<dataset_name>/`
4. Update this README with dataset information

## Storage Considerations

- Raw data in `raw/` can be large and may be gitignored
- Processed data in `processed/` should be version controlled if reasonable in size
- Use Git LFS for large binary files
- Consider using DVC (Data Version Control) for very large datasets