#!/usr/bin/env python3
"""
Reorganize XFUND data into a cleaner directory structure
"""

import json
import shutil
from pathlib import Path


def reorganize_data():  # noqa: PLR0912, PLR0915
    """Reorganize XFUND data into train/val splits with cleaner structure."""

    # Define paths
    source_dir = Path("data/xfund_processed")
    target_dir = Path("data/processed/xfund")

    # Process VAE data
    vae_source = source_dir / "vae_data"
    vae_target = target_dir / "vae"

    if vae_source.exists():
        # Create directories
        (vae_target / "train").mkdir(parents=True, exist_ok=True)
        (vae_target / "val").mkdir(parents=True, exist_ok=True)

        # Load split files
        with (vae_source / "train_files.json").open() as f:
            train_files = json.load(f)
        with (vae_source / "val_files.json").open() as f:
            val_files = json.load(f)

        # Move train files
        for metadata_file in train_files:
            base_name = metadata_file.replace("_metadata.json", "")

            # Move metadata
            shutil.copy2(vae_source / metadata_file, vae_target / "train" / metadata_file)

            # Move associated images
            for suffix in ["_original.png", "_masked.png"]:
                img_file = base_name + suffix
                if (vae_source / img_file).exists():
                    shutil.copy2(vae_source / img_file, vae_target / "train" / img_file)

        # Move val files
        for metadata_file in val_files:
            base_name = metadata_file.replace("_metadata.json", "")

            # Move metadata
            shutil.copy2(vae_source / metadata_file, vae_target / "val" / metadata_file)

            # Move associated images
            for suffix in ["_original.png", "_masked.png"]:
                img_file = base_name + suffix
                if (vae_source / img_file).exists():
                    shutil.copy2(vae_source / img_file, vae_target / "val" / img_file)

        # Copy split files
        shutil.copy2(vae_source / "train_files.json", vae_target / "train_files.json")
        shutil.copy2(vae_source / "val_files.json", vae_target / "val_files.json")

        print(f"✓ Reorganized VAE data to {vae_target}")

    # Process UNET data
    unet_source = source_dir / "unet_data"
    unet_target = target_dir / "unet"

    if unet_source.exists():
        # Create directories
        (unet_target / "train").mkdir(parents=True, exist_ok=True)
        (unet_target / "val").mkdir(parents=True, exist_ok=True)

        # Load split files
        with (unet_source / "train_files.json").open() as f:
            train_files = json.load(f)
        with (unet_source / "val_files.json").open() as f:
            val_files = json.load(f)

        # Move train files
        for metadata_file in train_files:
            base_name = metadata_file.replace("_metadata.json", "")

            # Move metadata
            shutil.copy2(unet_source / metadata_file, unet_target / "train" / metadata_file)

            # Move associated images
            for suffix in ["_input.png", "_mask.png"]:
                img_file = base_name + suffix
                if (unet_source / img_file).exists():
                    shutil.copy2(unet_source / img_file, unet_target / "train" / img_file)

        # Move val files
        for metadata_file in val_files:
            base_name = metadata_file.replace("_metadata.json", "")

            # Move metadata
            shutil.copy2(unet_source / metadata_file, unet_target / "val" / metadata_file)

            # Move associated images
            for suffix in ["_input.png", "_mask.png"]:
                img_file = base_name + suffix
                if (unet_source / img_file).exists():
                    shutil.copy2(unet_source / img_file, unet_target / "val" / img_file)

        # Copy split files
        shutil.copy2(unet_source / "train_files.json", unet_target / "train_files.json")
        shutil.copy2(unet_source / "val_files.json", unet_target / "val_files.json")

        print(f"✓ Reorganized UNET data to {unet_target}")

    # Create README
    readme_content = """# XFUND Processed Data

This directory contains processed XFUND dataset for document anonymization training.

## Structure

```
xfund/
├── vae/                    # Data for VAE training
│   ├── train/             # Training samples
│   ├── val/               # Validation samples
│   ├── train_files.json   # List of training files
│   └── val_files.json     # List of validation files
└── unet/                  # Data for UNET training
    ├── train/             # Training samples
    ├── val/               # Validation samples
    ├── train_files.json   # List of training files
    └── val_files.json     # List of validation files
```

## File Formats

### VAE Data
- `*_original.png`: Original document images
- `*_masked.png`: Images with text regions masked in white
- `*_metadata.json`: Annotation data with text regions and labels

### UNET Data
- `*_input.png`: Input document images
- `*_mask.png`: Segmentation masks with different classes for entity types
- `*_metadata.json`: Annotation data with class mappings

## Usage

Use the `AnonymizerDataset` class to load this data:

```python
from src.anonymizer.training.datasets import AnonymizerDataset
from src.anonymizer.core.config import DatasetConfig

config = DatasetConfig(
    train_data_path=Path("data/processed/xfund/vae/train"),
    val_data_path=Path("data/processed/xfund/vae/val"),
    batch_size=32,
    crop_size=512
)

dataset = AnonymizerDataset(
    data_dir=config.train_data_path,
    config=config,
    split="train"
)
```
"""

    with (target_dir / "README.md").open("w") as f:
        f.write(readme_content)

    print(f"✓ Created README at {target_dir / 'README.md'}")

    # Create .gitkeep files for empty directories
    for dir_path in ["models", "outputs", "logs", "cache", "checkpoints"]:
        path = Path(dir_path)
        if path.exists() and not any(path.iterdir()):
            (path / ".gitkeep").touch()
            print(f"✓ Created .gitkeep in {path}")


if __name__ == "__main__":
    reorganize_data()
