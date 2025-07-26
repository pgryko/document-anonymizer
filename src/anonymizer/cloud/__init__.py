"""Cloud training infrastructure for document anonymization models."""

from .modal_config import ModalConfig, ModalTrainingConfig
from .wandb_integration import WandbLogger, setup_wandb

try:
    from .modal_app import app as modal_app
    from .modal_app import train_unet, train_vae

    HAS_MODAL = True
except ImportError:
    modal_app = None
    train_vae = None
    train_unet = None
    HAS_MODAL = False

__all__ = [
    "HAS_MODAL",
    "ModalConfig",
    "ModalTrainingConfig",
    "WandbLogger",
    "modal_app",
    "setup_wandb",
    "train_unet",
    "train_vae",
]
