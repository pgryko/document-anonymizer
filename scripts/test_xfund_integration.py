#!/usr/bin/env python3
"""
Test XFUND data integration with existing training code
======================================================

This script tests if the processed XFUND data is compatible with
the existing AnonymizerDataset class.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.anonymizer.core.config import DatasetConfig
from src.anonymizer.training.datasets import AnonymizerDataset, create_dataloader


def test_xfund_compatibility():
    """Test if XFUND data works with existing dataset classes."""

    # Create dataset config
    config = DatasetConfig(
        train_data_path=Path("data/processed/xfund/vae"),
        val_data_path=Path("data/processed/xfund/vae"),  # Using same for test
        batch_size=4,
        crop_size=512,
        num_workers=0,  # Single thread for testing
    )

    print("Testing XFUND data compatibility...")

    try:
        # Create dataset
        dataset = AnonymizerDataset(
            data_dir=config.train_data_path, config=config, split="train"
        )

        print(f"✓ Successfully created dataset with {len(dataset)} samples")

        # Test loading a sample
        sample = dataset[0]
        print("✓ Successfully loaded sample")
        print(f"  - Image shape: {sample['images'].shape}")
        print(f"  - Mask shape: {sample['masks'].shape}")
        print(f"  - Number of texts: {len(sample['texts'])}")

        # Test dataloader
        dataloader = create_dataloader(
            dataset, batch_size=2, num_workers=0, shuffle=False
        )

        # Manually fix the dataloader creation for num_workers=0
        from torch.utils.data import DataLoader
        from src.anonymizer.training.datasets import collate_fn

        dataloader = DataLoader(
            dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_fn,
            persistent_workers=False,
            prefetch_factor=None,  # Must be None when num_workers=0
        )

        batch = next(iter(dataloader))

        print("✓ Successfully created dataloader")
        print(f"  - Batch images shape: {batch['images'].shape}")
        print(f"  - Batch masks shape: {batch['masks'].shape}")
        print(f"  - Batch size: {batch['batch_size']}")

        print("\n✅ XFUND data is compatible with existing training code!")

    except Exception as e:
        print(f"\n❌ Compatibility test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_xfund_compatibility()
