#!/usr/bin/env python3
"""
Interactive VAE Quality Dashboard

Streamlit app for exploring VAE reconstruction quality with interactive controls.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch

from src.anonymizer.core.config import DatasetConfig, VAEConfig
from src.anonymizer.training.datasets import create_dataloaders
from src.anonymizer.training.vae_trainer import VAETrainer

# Constants
RGB_CHANNELS = 3


@st.cache_resource
def load_vae_model(checkpoint_path: str | None, config_path: str) -> VAETrainer:
    """Load VAE model (cached)."""
    config = VAEConfig.from_env_and_yaml(yaml_path=config_path)
    trainer = VAETrainer(config)

    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        trainer.vae.load_state_dict(checkpoint["model_state_dict"])
        st.success(f"Loaded checkpoint from {checkpoint_path}")
    else:
        st.warning("Using untrained model (random weights)")

    trainer.vae.eval()
    return trainer


@st.cache_data
def load_dataset_samples(config_path: str, max_samples: int = 50):
    """Load dataset samples (cached)."""
    is_local = "local" in config_path.lower()
    crop_size = 256 if is_local else 512

    dataset_config = DatasetConfig(
        train_data_path=Path("data/processed/xfund/vae"),
        val_data_path=Path("data/processed/xfund/vae"),
        crop_size=crop_size,
        num_workers=0,
    )

    train_dataloader, val_dataloader = create_dataloaders(dataset_config, batch_size=1)

    # Collect samples
    samples = []
    dataloader = val_dataloader or train_dataloader

    for i, batch in enumerate(dataloader):
        if i >= max_samples:
            break
        samples.append({"images": batch["images"], "texts": batch.get("texts", []), "index": i})

    return samples


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert tensor to displayable image."""
    img = ((tensor + 1) / 2).clamp(0, 1)
    img = img.permute(1, 2, 0) if img.shape[0] == RGB_CHANNELS else img.squeeze(0)
    return img.cpu().numpy()


def compute_reconstruction_metrics(
    original: torch.Tensor, reconstructed: torch.Tensor
) -> dict[str, float]:
    """Compute reconstruction quality metrics."""
    with torch.no_grad():
        mse = torch.nn.functional.mse_loss(reconstructed, original).item()
        mae = torch.nn.functional.l1_loss(reconstructed, original).item()

        # PSNR
        psnr = 20 * torch.log10(2.0 / torch.sqrt(torch.tensor(mse))).item()

        # Calculate SSIM
        def ssim_simple(x, y):
            mu_x = torch.mean(x)
            mu_y = torch.mean(y)
            sigma_x = torch.var(x)
            sigma_y = torch.var(y)
            sigma_xy = torch.mean((x - mu_x) * (y - mu_y))

            c1, c2 = 0.01**2, 0.03**2
            ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
                (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
            )
            return ssim.item()

        ssim = ssim_simple(original, reconstructed)

    return {"MSE": mse, "MAE": mae, "PSNR": psnr, "SSIM": ssim}


def main():  # noqa: PLR0915
    st.set_page_config(page_title="VAE Quality Dashboard", page_icon="🔍", layout="wide")

    st.title("🔍 VAE Quality Assessment Dashboard")
    st.markdown("Interactive exploration of VAE reconstruction quality for document anonymization")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    config_path = st.sidebar.selectbox(
        "Select Config",
        [
            "configs/training/vae_config_local.yaml",
            "configs/training/vae_config_cloud.yaml",
        ],
        index=0,
    )

    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path (optional)", placeholder="path/to/checkpoint.pth"
    )

    # Load model and data
    try:
        trainer = load_vae_model(checkpoint_path if checkpoint_path else None, config_path)
        samples = load_dataset_samples(config_path)

        st.sidebar.success(f"Loaded {len(samples)} samples")

    except Exception as e:
        st.error(f"Failed to load model/data: {e}")
        return

    # Main interface
    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("Sample Selection")

        sample_idx = st.slider("Sample Index", 0, len(samples) - 1, 0)

        sample = samples[sample_idx]

        # Display sample info
        st.subheader("Sample Info")
        st.write(f"**Index:** {sample['index']}")
        if sample["texts"]:
            st.write(f"**Text Regions:** {len(sample['texts'])}")
            with st.expander("View Text Regions"):
                for i, text in enumerate(sample["texts"]):
                    st.write(f"{i+1}. {text}")

    with col2:
        st.header("Reconstruction Analysis")

        # Get reconstruction
        with torch.no_grad():
            images = sample["images"]
            posterior = trainer.vae.encode(images).latent_dist
            latents = posterior.sample()
            reconstructed = trainer.vae.decode(latents).sample

        # Compute metrics
        metrics = compute_reconstruction_metrics(images[0], reconstructed[0])

        # Display metrics
        metric_cols = st.columns(4)
        for i, (name, value) in enumerate(metrics.items()):
            with metric_cols[i]:
                st.metric(name, f"{value:.4f}")

        # Display images
        st.subheader("Visual Comparison")

        img_cols = st.columns(3)

        with img_cols[0]:
            st.write("**Original**")
            original_img = tensor_to_image(images[0])
            st.image(original_img, use_column_width=True)

        with img_cols[1]:
            st.write("**Reconstructed**")
            recon_img = tensor_to_image(reconstructed[0])
            st.image(recon_img, use_column_width=True)

        with img_cols[2]:
            st.write("**Difference**")
            diff = np.abs(original_img - recon_img)
            if len(diff.shape) == RGB_CHANNELS:
                diff = np.mean(diff, axis=2)

            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(diff, cmap="hot", vmin=0, vmax=0.5)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)

    # Batch analysis
    st.header("Batch Analysis")

    if st.button("Analyze Multiple Samples"):
        num_analyze = min(20, len(samples))

        with st.spinner(f"Analyzing {num_analyze} samples..."):
            batch_metrics = []

            for i in range(num_analyze):
                sample = samples[i]
                with torch.no_grad():
                    images = sample["images"]
                    posterior = trainer.vae.encode(images).latent_dist
                    latents = posterior.sample()
                    reconstructed = trainer.vae.decode(latents).sample

                metrics = compute_reconstruction_metrics(images[0], reconstructed[0])
                metrics["sample_idx"] = i
                batch_metrics.append(metrics)

        # Plot metrics distribution
        # pandas imported at module level

        df = pd.DataFrame(batch_metrics)

        fig_cols = st.columns(2)

        with fig_cols[0]:
            fig = px.histogram(df, x="MSE", title="MSE Distribution", nbins=20)
            st.plotly_chart(fig, use_container_width=True)

        with fig_cols[1]:
            fig = px.histogram(df, x="PSNR", title="PSNR Distribution", nbins=20)
            st.plotly_chart(fig, use_container_width=True)

        # Scatter plot
        fig = px.scatter(df, x="MSE", y="PSNR", hover_data=["sample_idx"], title="MSE vs PSNR")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
