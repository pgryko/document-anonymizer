#!/usr/bin/env python3
"""
VAE Quality Visualization Script

Quick visual assessment of VAE reconstruction quality for document anonymization.
Shows original vs reconstructed images with loss metrics.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

from src.anonymizer.core.config import VAEConfig, DatasetConfig
from src.anonymizer.training.datasets import create_dataloaders
from src.anonymizer.training.vae_trainer import VAETrainer


def load_vae_model(checkpoint_path: str, config_path: str) -> VAETrainer:
    """Load trained VAE model."""
    config = VAEConfig.from_env_and_yaml(yaml_path=config_path)
    trainer = VAETrainer(config)

    # Load checkpoint if provided
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        trainer.vae.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from {checkpoint_path}")

    trainer.vae.eval()
    return trainer


def visualize_reconstructions(
    trainer: VAETrainer, dataloader, num_samples: int = 8, save_path: str = None
) -> None:
    """Visualize original vs reconstructed images."""

    # Get a batch of data
    batch = next(iter(dataloader))
    images = batch["images"][:num_samples]

    with torch.no_grad():
        # Encode and decode
        posterior = trainer.vae.encode(images).latent_dist
        latents = posterior.sample()
        reconstructed = trainer.vae.decode(latents).sample

        # Compute metrics
        mse_loss = torch.nn.functional.mse_loss(reconstructed, images, reduction="none")
        mse_per_image = mse_loss.mean(dim=[1, 2, 3])

        # Convert to numpy for visualization
        images_np = ((images + 1) / 2).clamp(0, 1).cpu().numpy()
        reconstructed_np = ((reconstructed + 1) / 2).clamp(0, 1).cpu().numpy()

    # Create visualization
    fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6))

    for i in range(num_samples):
        # Original image
        if images_np.shape[1] == 3:  # RGB
            img_orig = np.transpose(images_np[i], (1, 2, 0))
            img_recon = np.transpose(reconstructed_np[i], (1, 2, 0))
        else:  # Grayscale
            img_orig = images_np[i, 0]
            img_recon = reconstructed_np[i, 0]

        axes[0, i].imshow(img_orig, cmap="gray" if len(img_orig.shape) == 2 else None)
        axes[0, i].set_title(f"Original {i+1}")
        axes[0, i].axis("off")

        axes[1, i].imshow(img_recon, cmap="gray" if len(img_recon.shape) == 2 else None)
        axes[1, i].set_title(f"Reconstructed {i+1}")
        axes[1, i].axis("off")

        # Difference map
        diff = np.abs(img_orig - img_recon)
        if len(diff.shape) == 3:
            diff = np.mean(diff, axis=2)

        axes[2, i].imshow(diff, cmap="hot", vmin=0, vmax=0.5)
        axes[2, i].set_title(f"Diff (MSE: {mse_per_image[i]:.4f})")
        axes[2, i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    plt.show()


def analyze_latent_space(trainer: VAETrainer, dataloader, num_samples: int = 100):
    """Analyze latent space statistics."""
    latents_list = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i * dataloader.batch_size >= num_samples:
                break

            images = batch["images"]
            posterior = trainer.vae.encode(images).latent_dist
            latents = posterior.sample()
            latents_list.append(latents.cpu().numpy())

    # Concatenate all latents
    all_latents = np.concatenate(latents_list, axis=0)[:num_samples]

    # Compute statistics
    mean_latent = np.mean(all_latents, axis=0)
    std_latent = np.std(all_latents, axis=0)

    print(f"Latent space analysis ({num_samples} samples):")
    print(f"Shape: {all_latents.shape}")
    print(f"Mean range: [{mean_latent.min():.3f}, {mean_latent.max():.3f}]")
    print(f"Std range: [{std_latent.min():.3f}, {std_latent.max():.3f}]")

    # Plot latent statistics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(all_latents.flatten(), bins=50, alpha=0.7)
    ax1.set_title("Latent Values Distribution")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Frequency")

    # Plot channel-wise statistics
    channel_means = np.mean(all_latents, axis=(0, 2, 3))
    channel_stds = np.std(all_latents, axis=(0, 2, 3))

    x = np.arange(len(channel_means))
    ax2.bar(x - 0.2, channel_means, 0.4, label="Mean", alpha=0.7)
    ax2.bar(x + 0.2, channel_stds, 0.4, label="Std", alpha=0.7)
    ax2.set_title("Per-Channel Statistics")
    ax2.set_xlabel("Channel")
    ax2.set_ylabel("Value")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize VAE quality")
    parser.add_argument("--config", required=True, help="VAE config file")
    parser.add_argument("--checkpoint", help="Model checkpoint path")
    parser.add_argument(
        "--num-samples", type=int, default=8, help="Number of samples to visualize"
    )
    parser.add_argument("--save-path", help="Path to save visualization")
    parser.add_argument(
        "--analyze-latents", action="store_true", help="Analyze latent space"
    )

    args = parser.parse_args()

    # Load model
    trainer = load_vae_model(args.checkpoint, args.config)

    # Create dataset
    is_local = "local" in args.config.lower()
    crop_size = 256 if is_local else 512

    dataset_config = DatasetConfig(
        train_data_path=Path("data/processed/xfund/vae"),
        val_data_path=Path("data/processed/xfund/vae"),
        crop_size=crop_size,
        num_workers=0,
    )

    train_dataloader, val_dataloader = create_dataloaders(dataset_config, batch_size=8)

    # Visualize reconstructions
    print("Visualizing reconstructions...")
    visualize_reconstructions(
        trainer, val_dataloader or train_dataloader, args.num_samples, args.save_path
    )

    # Analyze latent space if requested
    if args.analyze_latents:
        print("\nAnalyzing latent space...")
        analyze_latent_space(trainer, val_dataloader or train_dataloader)


if __name__ == "__main__":
    main()
