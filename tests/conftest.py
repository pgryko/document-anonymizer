"""
Pytest configuration and fixtures for document anonymization tests.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.anonymizer.core.config import (
    DatasetConfig,
    EngineConfig,
    UNetConfig,
    VAEConfig,
)
from src.anonymizer.core.models import (
    AnonymizationRequest,
    BoundingBox,
    ProcessedImage,
    TextRegion,
)


@pytest.fixture(scope="session")
def device():
    """Get device for testing (GPU if available)."""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Specify exact device
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


@pytest.fixture
def sample_image():
    """Create sample RGB image for testing."""
    # Create 256x256 white image with black text-like rectangle
    image = np.ones((256, 256, 3), dtype=np.uint8) * 255
    # Add black rectangle to simulate text
    image[100:150, 50:200] = 0
    return image


@pytest.fixture
def sample_image_pil():
    """Create sample PIL image."""
    return Image.new("RGB", (256, 256), color="white")


@pytest.fixture
def sample_bbox():
    """Create sample bounding box."""
    return BoundingBox(left=50, top=100, right=200, bottom=150)


@pytest.fixture
def sample_text_region(sample_bbox):
    """Create sample text region."""
    return TextRegion(
        bbox=sample_bbox,
        original_text="John Doe",
        replacement_text="REDACTED",
        confidence=0.95,
    )


@pytest.fixture
def sample_processed_image(sample_image, sample_bbox):
    """Create sample processed image."""
    crop = sample_image[sample_bbox.top : sample_bbox.bottom, sample_bbox.left : sample_bbox.right]
    mask = np.ones((crop.shape[0], crop.shape[1]), dtype=np.float32)

    return ProcessedImage(crop=crop, mask=mask, original_bbox=sample_bbox, scale_factor=1.0)


@pytest.fixture
def sample_anonymization_request(sample_image, sample_text_region):
    """Create sample anonymization request."""
    # Convert image to bytes
    pil_image = Image.fromarray(sample_image)
    import io

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    image_data = buffer.getvalue()

    return AnonymizationRequest(
        image_data=image_data,
        text_regions=[sample_text_region],
        preserve_formatting=True,
        quality_check=True,
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_dataset_dir(temp_dir, sample_image, sample_text_region):
    """Create mock dataset directory with sample data."""
    # Create sample image file
    image_path = temp_dir / "sample_image.png"
    pil_image = Image.fromarray(sample_image)
    pil_image.save(image_path)

    # Create annotation file
    annotation_data = {
        "image_name": "sample_image.png",
        "text_regions": [
            {
                "bbox": {
                    "left": sample_text_region.bbox.left,
                    "top": sample_text_region.bbox.top,
                    "right": sample_text_region.bbox.right,
                    "bottom": sample_text_region.bbox.bottom,
                },
                "original_text": sample_text_region.original_text,
                "replacement_text": sample_text_region.replacement_text,
                "confidence": sample_text_region.confidence,
            }
        ],
    }

    annotation_path = temp_dir / "sample_image.json"
    with open(annotation_path, "w") as f:
        json.dump(annotation_data, f, indent=2)

    return temp_dir


@pytest.fixture
def vae_config():
    """Create VAE configuration for testing."""
    return VAEConfig(
        model_name="test-vae",
        version="v1.0",
        base_model="stabilityai/stable-diffusion-2-1-base",
        batch_size=2,  # Small for testing
        learning_rate=5e-4,
        num_epochs=1,  # Single epoch for testing
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        gradient_clipping=1.0,
    )


@pytest.fixture
def unet_config():
    """Create UNet configuration for testing."""
    return UNetConfig(
        model_name="test-unet",
        version="v1.0",
        base_model="stabilityai/stable-diffusion-2-inpainting",
        batch_size=2,  # Small for testing
        learning_rate=1e-4,
        num_epochs=1,  # Single epoch for testing
        gradient_accumulation_steps=1,
        mixed_precision="fp16",
        gradient_clipping=1.0,
    )


@pytest.fixture
def dataset_config(mock_dataset_dir):
    """Create dataset configuration for testing."""
    return DatasetConfig(
        train_data_path=mock_dataset_dir,
        val_data_path=None,
        crop_size=256,  # Smaller for testing
        num_workers=0,  # No multiprocessing in tests
        rotation_range=2.0,
        brightness_range=0.05,
        contrast_range=0.05,
    )


@pytest.fixture
def engine_config():
    """Create inference engine configuration for testing."""
    return EngineConfig(
        vae_model_path=None,
        unet_model_path=None,
        num_inference_steps=10,  # Reduced for testing
        guidance_scale=7.5,
        strength=1.0,
        enable_memory_efficient_attention=True,
        enable_sequential_cpu_offload=False,
        max_batch_size=2,
        enable_quality_check=True,
        min_confidence_threshold=0.7,
    )


@pytest.fixture
def mock_tensors():
    """Create mock tensors for testing."""
    batch_size = 2
    channels = 3
    height = 256
    width = 256

    return {
        "images": torch.randn(batch_size, channels, height, width),
        "masks": torch.ones(batch_size, 1, height, width),
        "latents": torch.randn(batch_size, 4, height // 8, width // 8),
        "noise": torch.randn(batch_size, 4, height // 8, width // 8),
        "timesteps": torch.randint(0, 1000, (batch_size,)),
    }


@pytest.fixture
def skip_if_no_gpu():
    """Skip test if no GPU available."""
    if not torch.cuda.is_available():
        pytest.skip("GPU not available")


@pytest.fixture
def skip_if_no_models():
    """Skip test if models not available (to avoid downloads in tests)."""
    # This can be used to skip tests that require model downloads
    # We'll mock the models instead


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark GPU tests
        if "gpu" in item.nodeid or "cuda" in item.nodeid:
            item.add_marker(pytest.mark.gpu)

        # Mark slow tests
        if "slow" in item.nodeid or item.get_closest_marker("slow"):
            item.add_marker(pytest.mark.slow)

        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)
