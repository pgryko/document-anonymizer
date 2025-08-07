"""Modal.com application for document anonymizer training."""

import contextlib
import logging
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

from src.anonymizer.core.config import DatasetConfig, UNetConfig, VAEConfig
from src.anonymizer.core.exceptions import ModalNotAvailableError
from src.anonymizer.training import UNetTrainer, VAETrainer
from src.anonymizer.training.datasets import create_dataloaders, create_inpainting_dataloaders

try:
    import modal

    HAS_MODAL = True
except ImportError:
    modal = None  # type: ignore[assignment]
    HAS_MODAL = False

from .modal_config import ModalConfig, ModalTrainingConfig
from .wandb_integration import setup_wandb

logger = logging.getLogger(__name__)

# Only create Modal app if modal is available
if HAS_MODAL:
    # Initialize Modal configuration with defaults
    modal_config = ModalConfig(
        app_name="document-anonymizer",
        gpu_type="A100-40GB",
        gpu_count=1,
        memory_gb=32,
        cpu_count=8,
        timeout_hours=6,
        python_version="3.12",
        min_containers=0,
        volume_name="anonymizer-data",
        volume_mount_path="/data",
        wandb_secret_name="wandb-secret",
        hf_secret_name="huggingface-secret",
    )

    # Create Modal app
    app = modal.App(modal_config.app_name)

    # Create Modal image with dependencies using uv and pyproject.toml
    image = (
        modal.Image.debian_slim(python_version=modal_config.python_version)
        .apt_install(
            # System dependencies for OCR, image processing, and font rendering
            "git",
            "wget",
            "curl",
            "libgl1-mesa-glx",
            "libglib2.0-0",  # OpenCV
            "tesseract-ocr",
            "tesseract-ocr-eng",  # Tesseract OCR
            "libglib2.0-0",
            "libsm6",
            "libxext6",
            "libxrender-dev",  # GUI libs
            "libgomp1",  # OpenMP for parallel processing
            "fonts-dejavu-core",
            "fontconfig",  # Font support
        )
        # Install uv for fast dependency management
        .run_commands("pip install --upgrade pip uv")
        # Copy pyproject.toml and install dependencies via uv
        .add_local_file("pyproject.toml", "/root/pyproject.toml")
        .run_commands(
            "cd /root && uv pip install --system .",
            # Ensure modal and wandb are available (should be in dependencies already)
        )
        # Copy the entire codebase into the image
        .add_local_dir("src", "/root/src")
        .add_local_dir("configs", "/root/configs")
        .run_commands(
            # Set up Python path
            "export PYTHONPATH=/root:$PYTHONPATH",
            # Download spaCy language model
            "python -m spacy download en_core_web_sm",
        )
    )

    # Create persistent volume for datasets and checkpoints
    volume = modal.Volume.from_name(modal_config.volume_name, create_if_missing=True)

    # Helper function to get secrets
    def get_secrets() -> list[Any]:
        """Get available Modal secrets."""
        secrets = []
        with contextlib.suppress(Exception):
            secrets.append(modal.Secret.from_name(modal_config.wandb_secret_name))
        with contextlib.suppress(Exception):
            secrets.append(modal.Secret.from_name(modal_config.hf_secret_name))
        return secrets

    secrets = get_secrets()

    @app.function(
        image=image,
        gpu=modal_config.gpu_type,
        volumes={modal_config.volume_mount_path: volume},
        secrets=secrets,
        timeout=modal_config.timeout_hours * 3600,
        memory=modal_config.memory_gb * 1024,
        min_containers=modal_config.min_containers,
    )
    def train_vae(
        config_path: str,
        train_data_path: str,
        val_data_path: str | None = None,
        output_dir: str = "/data/checkpoints",
        wandb_project: str = "document-anonymizer",
        wandb_entity: str | None = None,
        wandb_tags: list | None = None,
        push_to_hub: bool = False,
        hub_model_id: str | None = None,
        hub_private: bool = True,
        resume_from_checkpoint: str | None = None,
        compile_model: bool = False,
    ) -> str:
        """Train VAE model on Modal.com."""
        # Set up Python path
        sys.path.insert(0, "/root")

        try:

            print("🚀 Starting VAE training on Modal.com")
            print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
            print(f"Config: {config_path}")
            print(f"Train data: {train_data_path}")
            print(f"Output: {output_dir}")

            # Load configuration
            with Path(config_path).open() as f:
                config_dict = yaml.safe_load(f)

            # Create VAE config
            vae_config = VAEConfig(**config_dict)

            # Override checkpoint directory
            vae_config.checkpoint_dir = Path(output_dir)

            # Setup W&B logging
            training_config = ModalTrainingConfig(
                model_type="vae",
                config_path=config_path,
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                output_dir=output_dir,
                checkpoint_name=None,
                mixed_precision=vae_config.mixed_precision,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                wandb_tags=wandb_tags or [],
                push_to_hub=push_to_hub,
                hub_model_id=hub_model_id,
                hub_private=hub_private,
                resume_from_checkpoint=resume_from_checkpoint,
                compile_model=compile_model,
            )

            wandb_logger = setup_wandb(
                training_config=training_config,
                model_config=vae_config.model_dump(),
                platform="modal.com",
            )

            # Initialize W&B run
            wandb_logger.init()

            # Create dataset config and data loaders
            dataset_config = DatasetConfig(
                train_data_path=Path(train_data_path),
                val_data_path=Path(val_data_path) if val_data_path else None,
                crop_size=512,
                num_workers=4,
                rotation_range=0.0,
                brightness_range=0.0,
                contrast_range=0.0,
            )

            train_dataloader, val_dataloader = create_dataloaders(
                dataset_config, batch_size=vae_config.batch_size
            )

            # Initialize trainer
            trainer = VAETrainer(vae_config)

            # Setup model watching for W&B
            if wandb_logger.enabled:
                # We'll watch the model after it's initialized in the trainer
                pass

            # Train model
            print("🏋️ Starting training...")
            trainer.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                wandb_logger=wandb_logger,
            )

            # Save final model
            final_artifacts = trainer.save_model()

            # Log model to W&B
            if wandb_logger.enabled:
                wandb_logger.log_model(str(final_artifacts.model_path), name="vae-final")

            # Push to HuggingFace Hub if requested
            if push_to_hub and hub_model_id:
                print(f"📤 Pushing model to HuggingFace Hub: {hub_model_id}")
                # TODO: Implement HuggingFace Hub upload

            # Finish W&B run
            wandb_logger.finish()

            print("✅ VAE training completed successfully!")
            return str(final_artifacts.model_path)

        except Exception:
            logger.exception("VAE training failed")
            if "wandb_logger" in locals():
                wandb_logger.finish()
            raise

    @app.function(
        image=image,
        gpu=modal_config.gpu_type,
        volumes={modal_config.volume_mount_path: volume},
        secrets=secrets,
        timeout=modal_config.timeout_hours * 3600,
        memory=modal_config.memory_gb * 1024,
        min_containers=modal_config.min_containers,
    )
    def train_unet(
        config_path: str,
        train_data_path: str,
        val_data_path: str | None = None,
        output_dir: str = "/data/checkpoints",
        wandb_project: str = "document-anonymizer",
        wandb_entity: str | None = None,
        wandb_tags: list | None = None,
        push_to_hub: bool = False,
        hub_model_id: str | None = None,
        hub_private: bool = True,
        resume_from_checkpoint: str | None = None,
        compile_model: bool = False,
    ) -> str:
        """Train UNet model on Modal.com."""
        # Set up Python path
        sys.path.insert(0, "/root")

        try:

            print("🚀 Starting UNet training on Modal.com")
            print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
            print(f"Config: {config_path}")
            print(f"Train data: {train_data_path}")
            print(f"Output: {output_dir}")

            # Load configuration
            with Path(config_path).open() as f:
                config_dict = yaml.safe_load(f)

            # Create UNet config
            unet_config = UNetConfig(**config_dict)

            # Override checkpoint directory
            unet_config.checkpoint_dir = Path(output_dir)

            # Setup W&B logging
            training_config = ModalTrainingConfig(
                model_type="unet",
                config_path=config_path,
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                output_dir=output_dir,
                checkpoint_name=None,
                mixed_precision=unet_config.mixed_precision,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                wandb_tags=wandb_tags or [],
                push_to_hub=push_to_hub,
                hub_model_id=hub_model_id,
                hub_private=hub_private,
                resume_from_checkpoint=resume_from_checkpoint,
                compile_model=compile_model,
            )

            wandb_logger = setup_wandb(
                training_config=training_config,
                model_config=unet_config.model_dump(),
                platform="modal.com",
            )

            # Initialize W&B run
            wandb_logger.init()

            # Create dataset config and data loaders
            dataset_config = DatasetConfig(
                train_data_path=Path(train_data_path),
                val_data_path=Path(val_data_path) if val_data_path else None,
                crop_size=512,
                num_workers=4,
                rotation_range=0.0,
                brightness_range=0.0,
                contrast_range=0.0,
            )

            train_dataloader, val_dataloader = create_inpainting_dataloaders(
                dataset_config, batch_size=unet_config.batch_size
            )

            # Initialize trainer
            trainer = UNetTrainer(unet_config)

            # Train model
            print("🏋️ Starting training...")
            trainer.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
            )

            # Save final model
            final_artifacts = trainer.save_model()

            # Log model to W&B
            if wandb_logger.enabled:
                wandb_logger.log_model(str(final_artifacts.model_path), name="unet-final")

            # Push to HuggingFace Hub if requested
            if push_to_hub and hub_model_id:
                print(f"📤 Pushing model to HuggingFace Hub: {hub_model_id}")
                # TODO: Implement HuggingFace Hub upload

            # Finish W&B run
            wandb_logger.finish()

            print("✅ UNet training completed successfully!")
            return str(final_artifacts.model_path)

        except Exception:
            logger.exception("UNet training failed")
            if "wandb_logger" in locals():
                wandb_logger.finish()
            raise

else:
    # Create dummy objects when Modal is not available
    app = None
    train_vae = None
    train_unet = None

    def _modal_not_available(*args: Any, **kwargs: Any) -> None:  # noqa: ARG001
        raise ModalNotAvailableError()

    if app is None:
        app = type("DummyApp", (), {"function": lambda *_args, **_kwargs: _modal_not_available()})()
        train_vae = _modal_not_available
        train_unet = _modal_not_available
