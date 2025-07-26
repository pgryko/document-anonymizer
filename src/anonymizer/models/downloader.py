"""
Model Downloader
================

Handles downloading models from various sources with progress tracking,
validation, and error recovery.
"""

import hashlib
import json
import logging
import shutil
import tempfile
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from ..core.exceptions import InferenceError, ValidationError
from .config import ModelConfig, ModelFormat, ModelMetadata, ModelSource, ModelType

logger = logging.getLogger(__name__)


class DownloadProgress:
    """Progress tracker for downloads."""

    def __init__(self, total_size: int, description: str = "Downloading"):
        self.total_size = total_size
        self.downloaded = 0
        self.start_time = time.time()
        self.pbar = tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=description,
        )

    def update(self, chunk_size: int):
        """Update progress."""
        self.downloaded += chunk_size
        self.pbar.update(chunk_size)

    def close(self):
        """Close progress bar."""
        self.pbar.close()

    @property
    def elapsed_time(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.start_time

    @property
    def download_speed(self) -> float:
        """Get download speed in bytes per second."""
        if self.elapsed_time == 0:
            return 0
        return self.downloaded / self.elapsed_time


class ModelDownloader:
    """
    Downloads and manages diffusion models for document anonymization.

    Features:
    - Progressive download with resume capability
    - Checksum verification
    - Automatic retry on failure
    - URL validation and security checks
    - Progress tracking and reporting
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create HTTP session with appropriate configuration."""
        session = requests.Session()
        session.verify = self.config.verify_ssl

        # Set user agent
        session.headers.update({"User-Agent": "document-anonymizer/1.0.0"})

        return session

    def download_model(
        self,
        source: ModelSource,
        target_path: Path | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ModelMetadata:
        """
        Download a model from the specified source.

        Args:
            source: Model source configuration
            target_path: Optional custom target path
            progress_callback: Optional progress callback function

        Returns:
            ModelMetadata for the downloaded model
        """
        if not self._is_url_allowed(source.url):
            raise ValidationError(f"URL not allowed: {source.url}")

        # Determine target path
        if target_path is None:
            target_path = self._get_default_path(source)

        logger.info(f"Downloading model '{source.name}' from {source.url}")

        # Download with retries
        for attempt in range(self.config.max_retries):
            try:
                return self._download_with_progress(
                    source, target_path, progress_callback, attempt + 1
                )
            except Exception as e:
                logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise InferenceError(
                        f"Download failed after {self.config.max_retries} attempts: {e}"
                    )
                time.sleep(2**attempt)  # Exponential backoff

    def _download_with_progress(
        self,
        source: ModelSource,
        target_path: Path,
        progress_callback: Callable[[int, int], None] | None,
        attempt: int,
    ) -> ModelMetadata:
        """Download model with progress tracking."""
        # Create temp file for download
        with tempfile.NamedTemporaryFile(
            dir=self.config.temp_dir, delete=False, suffix=".tmp"
        ) as temp_file:
            temp_path = Path(temp_file.name)

        try:
            # Get file info
            response = self.session.head(source.url, timeout=self.config.timeout_seconds)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            if total_size == 0:
                logger.warning("Unable to determine file size")

            # Check available space
            self._check_disk_space(target_path.parent, total_size)

            # Download file
            logger.info(f"Downloading to temporary file: {temp_path}")

            response = self.session.get(
                source.url, stream=True, timeout=self.config.timeout_seconds
            )
            response.raise_for_status()

            # Setup progress tracking
            progress = DownloadProgress(total_size, f"Downloading {source.name}")
            downloaded_size = 0

            try:
                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=self.config.chunk_size):
                        if chunk:
                            f.write(chunk)
                            chunk_size = len(chunk)
                            downloaded_size += chunk_size

                            progress.update(chunk_size)

                            if progress_callback:
                                progress_callback(downloaded_size, total_size)

                progress.close()
                logger.info(
                    f"Download completed: {downloaded_size} bytes in {progress.elapsed_time:.2f}s"
                )

            except Exception as e:
                progress.close()
                raise e

            # Verify download
            if total_size > 0 and downloaded_size != total_size:
                raise InferenceError(
                    f"Download size mismatch: expected {total_size}, got {downloaded_size}"
                )

            # Verify checksum if provided
            if source.checksum and self.config.verify_checksums:
                self._verify_checksum(temp_path, source.checksum, source.checksum_type)

            # Move to final location
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_path), str(target_path))

            logger.info(f"Model downloaded successfully to: {target_path}")

            # Create metadata
            metadata = ModelMetadata(
                name=source.name,
                model_type=self._infer_model_type(source.name),
                format=source.format,
                version="1.0",  # Could be extracted from source
                source_url=source.url,
                local_path=target_path,
                size_bytes=downloaded_size,
                checksum=source.checksum,
                download_date=datetime.now().isoformat(),
                description=source.description,
            )

            # Save metadata
            self._save_metadata(metadata)

            return metadata

        except Exception as e:
            # Cleanup temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise e

    def _is_url_allowed(self, url: str) -> bool:
        """Check if URL is allowed based on security settings."""
        if not self.config.allow_external_urls:
            return False

        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Check trusted domains
            for trusted_domain in self.config.trusted_domains:
                if domain.endswith(trusted_domain.lower()):
                    return True

            logger.warning(f"URL domain not in trusted list: {domain}")
            return False

        except Exception as e:
            logger.error(f"Error parsing URL {url}: {e}")
            return False

    def _get_default_path(self, source: ModelSource) -> Path:
        """Get default download path for model."""
        # Create filename from source name and format
        if source.format == ModelFormat.SAFETENSORS:
            filename = f"{source.name}.safetensors"
        elif source.format == ModelFormat.PYTORCH:
            filename = f"{source.name}.pth"
        elif source.format == ModelFormat.DIFFUSERS:
            filename = source.name  # Directory name for diffusers format
        else:
            filename = source.name

        return self.config.models_dir / filename

    def _check_disk_space(self, path: Path, required_bytes: int):
        """Check if there's enough disk space."""
        try:
            stat = shutil.disk_usage(path)
            available = stat.free

            # Add 10% buffer
            required_with_buffer = int(required_bytes * 1.1)

            if available < required_with_buffer:
                raise InferenceError(
                    f"Insufficient disk space. Required: {required_with_buffer / 1024**3:.2f}GB, "
                    f"Available: {available / 1024**3:.2f}GB"
                )

        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

    def _verify_checksum(self, file_path: Path, expected_checksum: str, checksum_type: str):
        """Verify file checksum."""
        logger.info(f"Verifying {checksum_type} checksum...")

        # Select hash algorithm
        if checksum_type.lower() == "md5":
            hasher = hashlib.md5()
        elif checksum_type.lower() == "sha1":
            hasher = hashlib.sha1()
        elif checksum_type.lower() == "sha256":
            hasher = hashlib.sha256()
        else:
            raise ValidationError(f"Unsupported checksum type: {checksum_type}")

        # Calculate checksum
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)

        actual_checksum = hasher.hexdigest()

        if actual_checksum.lower() != expected_checksum.lower():
            raise ValidationError(
                f"Checksum verification failed. Expected: {expected_checksum}, "
                f"Actual: {actual_checksum}"
            )

        logger.info("Checksum verification passed")

    def _infer_model_type(self, name: str) -> ModelType:
        """Infer model type from name."""
        name_lower = name.lower()

        if "vae" in name_lower:
            return ModelType.VAE
        if "unet" in name_lower:
            return ModelType.UNET
        if "text_encoder" in name_lower:
            return ModelType.TEXT_ENCODER
        if "tokenizer" in name_lower:
            return ModelType.TOKENIZER
        if "scheduler" in name_lower:
            return ModelType.SCHEDULER
        return ModelType.FULL_PIPELINE

    def _save_metadata(self, metadata: ModelMetadata):
        """Save model metadata to disk."""
        metadata_path = self.config.models_dir / f"{metadata.name}.json"

        try:
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            logger.debug(f"Metadata saved: {metadata_path}")

        except Exception as e:
            logger.warning(f"Could not save metadata: {e}")

    def download_from_huggingface(
        self, model_id: str, filename: str | None = None, revision: str = "main"
    ) -> ModelMetadata:
        """
        Download model from Hugging Face Hub.

        Args:
            model_id: Hugging Face model ID (e.g., "stabilityai/stable-diffusion-2-inpainting")
            filename: Specific file to download (optional)
            revision: Model revision/branch to use

        Returns:
            ModelMetadata for downloaded model
        """
        try:
            from huggingface_hub import hf_hub_download, hf_hub_url

            if filename:
                # Download specific file
                url = hf_hub_url(model_id, filename=filename, revision=revision)
                local_path = hf_hub_download(model_id, filename=filename, revision=revision)

                source = ModelSource(
                    name=f"{model_id.replace('/', '_')}_{filename}",
                    url=url,
                    format=(
                        ModelFormat.SAFETENSORS
                        if filename.endswith(".safetensors")
                        else ModelFormat.PYTORCH
                    ),
                    description=f"Downloaded from Hugging Face: {model_id}",
                )

                # Create metadata from downloaded file
                file_path = Path(local_path)
                metadata = ModelMetadata(
                    name=source.name,
                    model_type=self._infer_model_type(filename),
                    format=source.format,
                    version=revision,
                    source_url=url,
                    local_path=file_path,
                    size_bytes=file_path.stat().st_size,
                    download_date=datetime.now().isoformat(),
                    description=source.description,
                )

                logger.info(f"Downloaded {filename} from {model_id}")
                return metadata

            raise NotImplementedError("Full model download not yet implemented")

        except ImportError:
            raise InferenceError(
                "huggingface_hub not available. Install with: pip install huggingface_hub"
            )
        except Exception as e:
            raise InferenceError(f"Hugging Face download failed: {e}")

    def cleanup(self):
        """Cleanup downloader resources."""
        if hasattr(self, "session"):
            self.session.close()
