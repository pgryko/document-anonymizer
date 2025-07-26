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
    from src.anonymizer.batch.processor import (
        BatchProcessor,
        ConsoleProgressCallback,
        create_batch_from_directory,
    )
    from src.anonymizer.core.config import AppConfig
    from src.anonymizer.core.exceptions import AnonymizerError
    from src.anonymizer.core.models import BatchAnonymizationRequest, BatchItem
    from src.anonymizer.inference.engine import InferenceEngine
    from src.anonymizer.training.unet_trainer import UNetTrainer
    from src.anonymizer.training.vae_trainer import VAETrainer
except ImportError as e:
    logger.exception(f"Import failed: {e}")
    logger.exception(
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
        from pathlib import Path

        from src.anonymizer.core.config import DatasetConfig, VAEConfig
        from src.anonymizer.training.datasets import create_dataloaders

        # Load VAE configuration
        vae_config = VAEConfig.from_env_and_yaml(yaml_path=config)

        # Create dataset configuration - optimized for local vs cloud
        is_local = "local" in str(config).lower()
        crop_size = 256 if is_local else 512  # Smaller images for local testing
        num_workers = 0 if is_local else 4  # No multiprocessing for local debugging

        dataset_config = DatasetConfig(
            train_data_path=Path("data/processed/xfund/vae"),
            val_data_path=Path("data/processed/xfund/vae"),
            crop_size=crop_size,
            num_workers=num_workers,
        )

        # Create dataloaders
        logger.info(f"Creating dataloaders (crop_size={crop_size}, workers={num_workers})...")
        train_dataloader, val_dataloader = create_dataloaders(
            dataset_config, batch_size=vae_config.batch_size
        )
        logger.info(
            f"Dataloaders created. Train samples: {len(train_dataloader.dataset)}, "
            f"Val samples: {len(val_dataloader.dataset) if val_dataloader else 0}"
        )

        # Memory management for local testing
        if is_local:
            import torch

            if torch.backends.mps.is_available():
                logger.info("Running on MPS backend - using conservative memory settings")
                # Clear any existing cache
                torch.mps.empty_cache()

        # Initialize and start training
        logger.info(f"Starting VAE training with config: {vae_config.model_name}")
        trainer = VAETrainer(vae_config)
        trainer.setup_distributed()
        trainer.train(train_dataloader, val_dataloader)
        logger.info("VAE training finished.")
    except AnonymizerError as e:
        logger.exception(f"VAE training failed: {e}")
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
        logger.exception(f"UNet training failed: {e}")
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
        logger.exception(f"Anonymization failed: {e}")
        sys.exit(1)


@cli.command(name="batch-anonymize")
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Input directory containing images to anonymize",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for anonymized images",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, path_type=Path),
    help="Path to configuration YAML file (optional)",
)
@click.option(
    "--max-parallel",
    "-p",
    type=int,
    default=4,
    help="Maximum number of parallel processes (default: 4)",
)
@click.option(
    "--batch-size",
    "-b",
    type=int,
    default=8,
    help="Batch size for memory management (default: 8)",
)
@click.option(
    "--preserve-structure",
    is_flag=True,
    default=True,
    help="Preserve input directory structure in output",
)
@click.option(
    "--continue-on-error",
    is_flag=True,
    default=True,
    help="Continue processing other items when one fails",
)
@click.option(
    "--pattern",
    type=str,
    default="*.{jpg,jpeg,png,tiff,pdf}",
    help="File pattern to match (default: *.{jpg,jpeg,png,tiff,pdf})",
)
def batch_anonymize(
    input_dir,
    output_dir,
    config,
    max_parallel,
    batch_size,
    preserve_structure,
    continue_on_error,
    pattern,
):
    """Batch anonymize multiple documents."""
    try:
        # Load configuration if provided
        app_config = None
        if config:
            app_config = AppConfig.from_env_and_yaml(yaml_path=config)

        # Create batch request from directory
        logger.info(f"Scanning {input_dir} for images...")
        batch_request = create_batch_from_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=pattern,
            preserve_structure=preserve_structure,
            max_parallel=max_parallel,
            batch_size=batch_size,
        )
        batch_request.continue_on_error = continue_on_error

        logger.info(f"Found {len(batch_request.items)} items to process")

        # Initialize batch processor
        inference_engine = None
        if app_config:
            inference_engine = InferenceEngine(app_config.engine)

        processor = BatchProcessor(inference_engine=inference_engine)

        # Create progress callback
        progress_callback = ConsoleProgressCallback(update_interval=2.0)

        # Process batch
        result = processor.process_batch(batch_request, progress_callback)

        # Summary
        logger.info("\nBatch processing completed!")
        logger.info(f"Success rate: {result.success_rate:.1f}%")
        logger.info(f"Output directory: {result.output_directory}")

        if result.failed_items > 0:
            logger.warning(f"Failed items: {result.failed_items}")
            for failed_result in result.get_failed_items():
                logger.warning(f"  - {failed_result.item_id}: {', '.join(failed_result.errors)}")

    except AnonymizerError as e:
        logger.exception(f"Batch anonymization failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)


@cli.command(name="batch-status")
@click.option(
    "--result-file",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    help="Path to batch result JSON file",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Output directory to analyze",
)
def batch_status(result_file, output_dir):
    """Show status of batch processing results."""
    try:
        if result_file:
            # Load and display result from JSON file
            import json

            with open(result_file) as f:
                result_data = json.load(f)

            print("Batch Processing Results")
            print("=" * 40)
            print(f"Total items: {result_data['total_items']}")
            print(f"Successful: {result_data['successful_items']}")
            print(f"Failed: {result_data['failed_items']}")
            print(
                f"Success rate: {result_data['successful_items'] / result_data['total_items'] * 100:.1f}%"
            )
            print(f"Total time: {result_data['total_processing_time_ms'] / 1000:.1f}s")
            print(f"Output directory: {result_data['output_directory']}")

        elif output_dir:
            # Analyze output directory

            output_files = list(output_dir.glob("**/*"))
            image_files = [
                f
                for f in output_files
                if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".tiff", ".pdf"}
            ]

            print("Output Directory Analysis")
            print("=" * 40)
            print(f"Directory: {output_dir}")
            print(f"Total files: {len(output_files)}")
            print(f"Image files: {len(image_files)}")

            # File size summary
            total_size = sum(f.stat().st_size for f in output_files if f.is_file())
            print(f"Total size: {total_size / 1024 / 1024:.1f}MB")

        else:
            print("Please provide either --result-file or --output-dir")

    except Exception as e:
        logger.exception(f"Status check failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
