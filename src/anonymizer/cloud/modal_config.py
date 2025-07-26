"""Modal.com configuration for cloud training."""

from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModalConfig(BaseSettings):
    """Configuration for Modal.com cloud training."""

    model_config = SettingsConfigDict(
        env_prefix="MODAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Modal app configuration
    app_name: str = Field("document-anonymizer", description="Modal app name")

    # GPU configuration
    gpu_type: str = Field("A100-40GB", description="GPU type for training")
    gpu_count: int = Field(1, ge=1, le=8, description="Number of GPUs")

    # Resource configuration
    memory_gb: int = Field(32, ge=8, le=128, description="Memory in GB")
    cpu_count: int = Field(8, ge=2, le=32, description="Number of CPUs")
    timeout_hours: int = Field(6, ge=1, le=24, description="Training timeout in hours")

    # Container configuration
    python_version: str = Field("3.12", description="Python version")
    min_containers: int = Field(0, ge=0, le=5, description="Minimum warm containers")

    # Volume configuration
    volume_name: str = Field("anonymizer-data", description="Persistent volume name")
    volume_mount_path: str = Field("/data", description="Volume mount path")

    # Secrets configuration
    wandb_secret_name: str = Field("wandb-secret", description="W&B secret name")
    hf_secret_name: str = Field("huggingface-secret", description="HuggingFace secret name")


class ModalTrainingConfig(BaseModel):
    """Training-specific configuration for Modal.com."""

    # Training parameters
    model_type: str = Field(..., description="Model type: 'vae' or 'unet'")
    config_path: str = Field(..., description="Path to training config file")

    # Data configuration
    train_data_path: str = Field(..., description="Training data path")
    val_data_path: str | None = Field(None, description="Validation data path")

    # Output configuration
    output_dir: str = Field("/data/checkpoints", description="Output directory")
    checkpoint_name: str | None = Field(None, description="Checkpoint name")

    # W&B configuration
    wandb_project: str = Field("document-anonymizer", description="W&B project name")
    wandb_entity: str | None = Field(None, description="W&B entity/username")
    wandb_tags: list[str] = Field(default_factory=list, description="W&B tags")

    # HuggingFace Hub configuration
    push_to_hub: bool = Field(False, description="Push model to HuggingFace Hub")
    hub_model_id: str | None = Field(None, description="HuggingFace model ID")
    hub_private: bool = Field(True, description="Make HuggingFace model private")

    # Advanced options
    resume_from_checkpoint: str | None = Field(None, description="Resume from checkpoint")
    compile_model: bool = Field(False, description="Use torch.compile")
    mixed_precision: str = Field("bf16", description="Mixed precision mode")

    def get_wandb_config(self) -> dict[str, Any]:
        """Get W&B configuration dictionary."""
        return {
            "model_type": self.model_type,
            "config_path": self.config_path,
            "train_data_path": self.train_data_path,
            "val_data_path": self.val_data_path,
            "output_dir": self.output_dir,
            "compile_model": self.compile_model,
            "mixed_precision": self.mixed_precision,
            "tags": self.wandb_tags,
        }
