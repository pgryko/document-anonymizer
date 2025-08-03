"""Modal.com application for document anonymizer training."""

import logging
import sys
from pathlib import Path

import torch
import yaml

from src.anonymizer.core.config import VAEConfig
from src.anonymizer.training import VAETrainer
from src.anonymizer.training.datasets import create_dataloader

try:
    import modal

    HAS_MODAL = True
except ImportError:
    modal = None
    HAS_MODAL = False

from .modal_config import ModalConfig, ModalTrainingConfig
from .wandb_integration import setup_wandb

logger = logging.getLogger(__name__)

# Only create Modal app if modal is available
if HAS_MODAL:
    # Initialize Modal configuration
    modal_config = ModalConfig()

    # Create Modal app
    app = modal.App(modal_config.app_name)

    # Create Modal image with dependencies
    # TODO: this should install via uv and dependencies in pyproject.toml
    image = (
        modal.Image.debian_slim(python_version=modal_config.python_version)
        .pip_install(
            [
                "torch>=2.6.0",
                "torchvision>=0.19.0",
                "transformers>=4.49.0",
                "diffusers>=0.32.2",
                "accelerate>=1.2.0",
                "safetensors>=0.4.0",
                "pydantic>=2.10.0",
                "pydantic-settings>=2.10.1",
                "pyyaml>=6.0.0",
                "pillow>=10.0.0",
                "opencv-python>=4.8.0",
                "numpy>=1.24.0",
                "wandb>=0.18.0",
                "tqdm>=4.66.0",
                "presidio-analyzer>=2.2.358",
                "presidio-anonymizer>=2.2.358",
                "presidio-image-redactor>=0.0.56",
                "spacy>=3.8.4",
            ]
        )
        .apt_install("git", "wget", "curl", "libgl1-mesa-glx", "libglib2.0-0")
        .run_commands("pip install --upgrade pip")
        # Copy the entire codebase into the image
        .add_local_dir(".", "/root/anonymizer")
    )

    # Create persistent volume for datasets and checkpoints
    volume = modal.Volume.from_name(modal_config.volume_name, create_if_missing=True)

    # Helper function to get secrets
    def get_secrets():
        """Get available Modal secrets."""
        secrets = []
        try:
            secrets.append(modal.Secret.from_name(modal_config.wandb_secret_name))
        except Exception:
            pass  # Secret doesn't exist, that's OK
        try:
            secrets.append(modal.Secret.from_name(modal_config.hf_secret_name))
        except Exception:
            pass  # Secret doesn't exist, that's OK
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
    ):
        """Train VAE model on Modal.com."""

        # Set up Python path
        sys.path.insert(0, "/root/anonymizer")

        try:

            print("🚀 Starting VAE training on Modal.com")
            print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
            print(f"Config: {config_path}")
            print(f"Train data: {train_data_path}")
            print(f"Output: {output_dir}")

            # Load configuration
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)

            # Create VAE config
            vae_config = VAEConfig(**config_dict)

            # Override paths
            vae_config.train_data_path = train_data_path
            if val_data_path:
                vae_config.val_data_path = val_data_path
            vae_config.checkpoint_dir = output_dir

            # Setup W&B logging
            training_config = ModalTrainingConfig(
                model_type="vae",
                config_path=config_path,
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                output_dir=output_dir,
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

            # Create data loaders
            train_dataloader = create_dataloader(
                data_path=train_data_path,
                batch_size=vae_config.batch_size,
                shuffle=True,
                num_workers=vae_config.num_workers,
            )

            val_dataloader = None
            if val_data_path:
                val_dataloader = create_dataloader(
                    data_path=val_data_path,
                    batch_size=vae_config.batch_size,
                    shuffle=False,
                    num_workers=vae_config.num_workers,
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
            final_checkpoint_path = Path(output_dir) / "final_model"
            trainer.save_model(final_checkpoint_path)

            # Log model to W&B
            if wandb_logger.enabled:
                wandb_logger.log_model(str(final_checkpoint_path), name="vae-final")

            # Push to HuggingFace Hub if requested
            if push_to_hub and hub_model_id:
                print(f"📤 Pushing model to HuggingFace Hub: {hub_model_id}")
                # TODO: Implement HuggingFace Hub upload

            # Finish W&B run
            wandb_logger.finish()

            print("✅ VAE training completed successfully!")
            return str(final_checkpoint_path)

        except Exception as e:
            logger.exception(f"VAE training failed: {e}")
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
    ):
        """Train UNet model on Modal.com."""

        # Set up Python path
        sys.path.insert(0, "/root/anonymizer")

        try:
            # Import training modules
            import torch
            import yaml

            from src.anonymizer.core.config import UNetConfig
            from src.anonymizer.training import UNetTrainer
            from src.anonymizer.training.datasets import create_dataloader

            print("🚀 Starting UNet training on Modal.com")
            print(f"GPU: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
            print(f"Config: {config_path}")
            print(f"Train data: {train_data_path}")
            print(f"Output: {output_dir}")

            # Load configuration
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)

            # Create UNet config
            unet_config = UNetConfig(**config_dict)

            # Override paths
            unet_config.train_data_path = train_data_path
            if val_data_path:
                unet_config.val_data_path = val_data_path
            unet_config.checkpoint_dir = output_dir

            # Setup W&B logging
            training_config = ModalTrainingConfig(
                model_type="unet",
                config_path=config_path,
                train_data_path=train_data_path,
                val_data_path=val_data_path,
                output_dir=output_dir,
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

            # Create data loaders
            train_dataloader = create_dataloader(
                data_path=train_data_path,
                batch_size=unet_config.batch_size,
                shuffle=True,
                num_workers=unet_config.num_workers,
            )

            val_dataloader = None
            if val_data_path:
                val_dataloader = create_dataloader(
                    data_path=val_data_path,
                    batch_size=unet_config.batch_size,
                    shuffle=False,
                    num_workers=unet_config.num_workers,
                )

            # Initialize trainer
            trainer = UNetTrainer(unet_config)

            # Train model
            print("🏋️ Starting training...")
            trainer.train(
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                wandb_logger=wandb_logger,
            )

            # Save final model
            final_checkpoint_path = Path(output_dir) / "final_model"
            trainer.save_model(final_checkpoint_path)

            # Log model to W&B
            if wandb_logger.enabled:
                wandb_logger.log_model(str(final_checkpoint_path), name="unet-final")

            # Push to HuggingFace Hub if requested
            if push_to_hub and hub_model_id:
                print(f"📤 Pushing model to HuggingFace Hub: {hub_model_id}")
                # TODO: Implement HuggingFace Hub upload

            # Finish W&B run
            wandb_logger.finish()

            print("✅ UNet training completed successfully!")
            return str(final_checkpoint_path)

        except Exception as e:
            logger.exception(f"UNet training failed: {e}")
            if "wandb_logger" in locals():
                wandb_logger.finish()
            raise

else:
    # Create dummy objects when Modal is not available
    app = None
    train_vae = None
    train_unet = None

    def _modal_not_available(*args, **kwargs):
        raise ImportError("Modal not available. Install with: pip install modal")

    if app is None:
        app = type("DummyApp", (), {"function": lambda *args, **kwargs: _modal_not_available})()
        train_vae = _modal_not_available
        train_unet = _modal_not_available
