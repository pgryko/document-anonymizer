"""Cloud training infrastructure for document anonymization models."""

from .modal_config import ModalConfig, ModalTrainingConfig
from .wandb_integration import WandbLogger, setup_wandb

try:
    from .modal_app import app as modal_app, train_vae, train_unet

    HAS_MODAL = True
except ImportError:
    modal_app = None
    train_vae = None
    train_unet = None
    HAS_MODAL = False

__all__ = [
    "ModalConfig",
    "ModalTrainingConfig",
    "WandbLogger",
    "setup_wandb",
    "modal_app",
    "train_vae",
    "train_unet",
    "HAS_MODAL",
]
