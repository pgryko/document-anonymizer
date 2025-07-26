#!/usr/bin/env python3
"""
Model Download Script
=====================

Command-line utility for downloading and managing diffusion models
for document anonymization.
"""

import click
import logging
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anonymizer.models import ModelManager, ModelConfig
from src.anonymizer.core.exceptions import ValidationError, InferenceError

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--models-dir", type=click.Path(path_type=Path), help="Directory to store models"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx, models_dir: Optional[Path], verbose: bool):
    """Document Anonymization Model Manager."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create model config
    config = ModelConfig()
    if models_dir:
        config.models_dir = models_dir
        config.models_dir.mkdir(parents=True, exist_ok=True)

    # Initialize model manager
    ctx.ensure_object(dict)
    ctx.obj["manager"] = ModelManager(config)
    ctx.obj["config"] = config


@cli.command()
@click.pass_context
def list_available(ctx):
    """List all available models in registry."""
    manager = ctx.obj["manager"]

    models = manager.list_available_models()

    if not models:
        click.echo("No models available in registry.")
        return

    click.echo(f"\n📋 Available Models ({len(models)}):")
    click.echo("=" * 50)

    for model in models:
        size_info = f" ({model.size_mb}MB)" if model.size_mb else ""
        click.echo(f"• {model.name}{size_info}")
        if model.description:
            click.echo(f"  {model.description}")
        click.echo(f"  URL: {model.url}")
        click.echo()


@cli.command()
@click.pass_context
def list_downloaded(ctx):
    """List downloaded models."""
    manager = ctx.obj["manager"]

    models = manager.list_downloaded_models()

    if not models:
        click.echo("No models downloaded yet.")
        return

    click.echo(f"\n💾 Downloaded Models ({len(models)}):")
    click.echo("=" * 50)

    for metadata in models:
        size_mb = metadata.size_bytes / (1024 * 1024)
        click.echo(f"• {metadata.name} ({size_mb:.1f}MB)")
        click.echo(f"  Path: {metadata.local_path}")
        click.echo(f"  Downloaded: {metadata.download_date}")
        click.echo(f"  Used: {metadata.usage_count} times")
        click.echo()


@cli.command()
@click.argument("model_name")
@click.option("--no-validate", is_flag=True, help="Skip validation after download")
@click.pass_context
def download(ctx, model_name: str, no_validate: bool):
    """Download a model by name."""
    manager = ctx.obj["manager"]

    try:
        click.echo(f"🔄 Downloading model: {model_name}")

        # Progress callback
        def progress_callback(downloaded: int, total: int):
            if total > 0:
                percent = (downloaded / total) * 100
                click.echo(
                    f"\r⏳ Progress: {percent:.1f}% ({downloaded}/{total} bytes)",
                    nl=False,
                )

        metadata = manager.download_model(
            model_name, progress_callback=progress_callback, validate=not no_validate
        )

        click.echo(f"\n✅ Successfully downloaded: {model_name}")
        click.echo(f"   📁 Location: {metadata.local_path}")
        click.echo(f"   📊 Size: {metadata.size_bytes / (1024*1024):.1f}MB")

    except (ValidationError, InferenceError) as e:
        click.echo(f"\n❌ Download failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n💥 Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("model_id")
@click.option("--filename", help="Specific file to download")
@click.option("--no-validate", is_flag=True, help="Skip validation after download")
@click.pass_context
def download_hf(ctx, model_id: str, filename: Optional[str], no_validate: bool):
    """Download model from Hugging Face Hub."""
    manager = ctx.obj["manager"]

    try:
        click.echo(f"🔄 Downloading from Hugging Face: {model_id}")
        if filename:
            click.echo(f"   📄 File: {filename}")

        metadata = manager.download_from_huggingface(
            model_id, filename=filename, validate=not no_validate
        )

        click.echo("\n✅ Successfully downloaded from Hugging Face")
        click.echo(f"   📁 Location: {metadata.local_path}")
        click.echo(f"   📊 Size: {metadata.size_bytes / (1024*1024):.1f}MB")

    except (ValidationError, InferenceError) as e:
        click.echo(f"\n❌ Download failed: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n💥 Unexpected error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("model_name")
@click.pass_context
def info(ctx, model_name: str):
    """Get detailed information about a model."""
    manager = ctx.obj["manager"]

    info_dict = manager.get_model_info(model_name)

    if "error" in info_dict:
        click.echo(f"❌ {info_dict['error']}", err=True)
        return

    click.echo(f"\n📋 Model Information: {model_name}")
    click.echo("=" * 50)

    for key, value in info_dict.items():
        if key != "name":
            formatted_key = key.replace("_", " ").title()
            click.echo(f"{formatted_key}: {value}")


@cli.command()
@click.argument("model_name")
@click.pass_context
def validate(ctx, model_name: str):
    """Validate a downloaded model."""
    manager = ctx.obj["manager"]

    # Get model metadata
    metadata = manager.registry.get_metadata(model_name)
    if not metadata:
        click.echo(f"❌ Model '{model_name}' not found", err=True)
        return

    if not metadata.local_path.exists():
        click.echo(f"❌ Model file not found: {metadata.local_path}", err=True)
        return

    click.echo(f"🔍 Validating model: {model_name}")

    try:
        result = manager.validate_model(metadata.local_path)

        if result.valid:
            click.echo("✅ Model validation passed")
        else:
            click.echo("❌ Model validation failed")

            if result.errors:
                click.echo("\nErrors:")
                for error in result.errors:
                    click.echo(f"  • {error}")

            if result.warnings:
                click.echo("\nWarnings:")
                for warning in result.warnings:
                    click.echo(f"  • {warning}")

    except Exception as e:
        click.echo(f"💥 Validation error: {e}", err=True)


@cli.command()
@click.argument("model_name")
@click.pass_context
def benchmark(ctx, model_name: str):
    """Benchmark a model's performance."""
    manager = ctx.obj["manager"]

    try:
        click.echo(f"⏱️ Benchmarking model: {model_name}")

        results = manager.benchmark_model(model_name)

        click.echo("\n📊 Benchmark Results:")
        click.echo("=" * 30)
        click.echo(f"Model Size: {results['model_size_mb']:.1f}MB")
        click.echo(f"Loading Time: {results['loading_time_ms']:.1f}ms")
        click.echo(f"Memory Usage: {results['memory_usage_mb']:.1f}MB")
        click.echo(f"Device: {results['device']}")

    except Exception as e:
        click.echo(f"❌ Benchmark failed: {e}", err=True)


@cli.command()
@click.option(
    "--use-case",
    default="default",
    type=click.Choice(["default", "fast", "quality", "custom"]),
    help="Use case for model recommendation",
)
@click.pass_context
def ensure_models(ctx, use_case: str):
    """Ensure all models for a use case are downloaded."""
    manager = ctx.obj["manager"]

    click.echo(f"🔄 Ensuring models for use case: {use_case}")

    try:
        success = manager.ensure_models_available(use_case)

        if success:
            click.echo("✅ All required models are available")
        else:
            click.echo("❌ Some models failed to download", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"💥 Error ensuring models: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--unused-days",
    default=30,
    type=int,
    help="Remove models unused for this many days",
)
@click.option(
    "--dry-run/--execute",
    default=True,
    help="Dry run (default) or actually delete files",
)
@click.pass_context
def cleanup(ctx, unused_days: int, dry_run: bool):
    """Clean up unused models."""
    manager = ctx.obj["manager"]

    click.echo(f"🧹 Cleaning up models unused for {unused_days} days...")
    if dry_run:
        click.echo("   (This is a dry run - no files will be deleted)")

    try:
        results = manager.cleanup_models(unused_days, dry_run=dry_run)

        if results["unused_models"]:
            click.echo(f"\n📋 Unused Models ({len(results['unused_models'])}):")
            for model_name in results["unused_models"]:
                click.echo(f"  • {model_name}")

            click.echo(f"\n💾 Total size: {results['total_size_mb']:.1f}MB")

            if not dry_run:
                click.echo(f"🗑️ Deleted: {results['deleted_count']} models")

            if results["errors"]:
                click.echo("\n❌ Errors:")
                for error in results["errors"]:
                    click.echo(f"  • {error}")
        else:
            click.echo("✨ No unused models found")

    except Exception as e:
        click.echo(f"💥 Cleanup failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def storage_stats(ctx):
    """Show storage statistics."""
    manager = ctx.obj["manager"]

    try:
        stats = manager.get_storage_stats()

        if "error" in stats:
            click.echo(f"❌ {stats['error']}", err=True)
            return

        click.echo("\n💾 Storage Statistics:")
        click.echo("=" * 30)
        click.echo(f"Models Directory: {stats['models_directory']}")
        click.echo(f"Total Models: {stats['total_models']}")
        click.echo(f"Models Size: {stats['total_size_gb']:.2f}GB")
        click.echo(f"Available Space: {stats['available_space_gb']:.2f}GB")
        click.echo(f"Used Space: {stats['used_space_gb']:.2f}GB")

    except Exception as e:
        click.echo(f"💥 Error getting stats: {e}", err=True)


@cli.command()
@click.pass_context
def verify_all(ctx):
    """Verify all downloaded models."""
    manager = ctx.obj["manager"]

    click.echo("🔍 Verifying all downloaded models...")

    try:
        results = manager.verify_all_models()

        if not results:
            click.echo("No models to verify.")
            return

        valid_count = sum(1 for r in results.values() if r.valid)

        click.echo("\n📊 Verification Results:")
        click.echo("=" * 30)
        click.echo(f"Total Models: {len(results)}")
        click.echo(f"Valid: {valid_count}")
        click.echo(f"Invalid: {len(results) - valid_count}")

        # Show details for invalid models
        for model_name, result in results.items():
            if not result.valid:
                click.echo(f"\n❌ {model_name}:")
                for error in result.errors:
                    click.echo(f"  • {error}")

    except Exception as e:
        click.echo(f"💥 Verification failed: {e}", err=True)


@cli.command()
@click.argument("query")
@click.pass_context
def search(ctx, query: str):
    """Search for models by name or description."""
    manager = ctx.obj["manager"]

    results = manager.search_models(query)

    if not results:
        click.echo(f"No models found matching '{query}'")
        return

    click.echo(f"\n🔍 Search Results for '{query}' ({len(results)}):")
    click.echo("=" * 50)

    for model in results:
        size_info = f" ({model.size_mb}MB)" if model.size_mb else ""
        click.echo(f"• {model.name}{size_info}")
        if model.description:
            click.echo(f"  {model.description}")
        click.echo()


if __name__ == "__main__":
    cli()
