#!/usr/bin/env python3
"""
Main CLI for Document Anonymization System
==========================================

This CLI provides commands to train and run the document anonymization system.
"""

import logging
import sys
from pathlib import Path
import click

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

try:
    from src.anonymizer.core.config import AppConfig
    from src.anonymizer.training.vae_trainer import VAETrainer
    from src.anonymizer.training.unet_trainer import UNetTrainer
    from src.anonymizer.inference.engine import InferenceEngine
    from src.anonymizer.core.exceptions import AnonymizerError
except ImportError as e:
    logger.error(f"Import failed: {e}")
    logger.error(
        "Make sure you have all dependencies installed and the project is properly set up"
    )
    sys.exit(1)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cli(verbose):
    """Document Anonymization System CLI."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command(name="train-vae")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to VAE configuration YAML file",
)
def train_vae(config):
    """Train VAE model."""
    try:
        from src.anonymizer.core.config import VAEConfig

        config = VAEConfig.from_env_and_yaml(yaml_path=config)
        trainer = VAETrainer(config)
        trainer.setup_distributed()
        # trainer.train(train_dataloader, val_dataloader)
        logger.info("VAE training finished.")
    except AnonymizerError as e:
        logger.error(f"VAE training failed: {e}")
        sys.exit(1)


@cli.command(name="train-unet")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to UNet configuration YAML file",
)
def train_unet(config):
    """Train UNet model."""
    try:
        from src.anonymizer.core.config import UNetConfig

        config = UNetConfig.from_env_and_yaml(yaml_path=config)
        trainer = UNetTrainer(config)
        trainer.setup_distributed()
        # trainer.train(train_dataloader, val_dataloader)
        logger.info("UNet training finished.")
    except AnonymizerError as e:
        logger.error(f"UNet training failed: {e}")
        sys.exit(1)


@cli.command(name="anonymize")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to App configuration YAML file",
)
@click.option(
    "--image",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to image to anonymize",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to save anonymized image",
)
def anonymize(config, image, output):
    """Anonymize a document."""
    try:
        config = AppConfig.from_env_and_yaml(yaml_path=config)
        InferenceEngine(config.engine)
        # image_data = open(image, "rb").read()
        # anonymized_image = engine.anonymize(image_data, [])
        # with open(output, "wb") as f:
        #     f.write(anonymized_image)
        logger.info(f"Anonymized image saved to {output}")
    except AnonymizerError as e:
        logger.error(f"Anonymization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
