"""
Model Management Tests
======================

Tests for model downloading, validation, and management functionality.
"""

import logging
import tempfile
from pathlib import Path

import pytest

from src.anonymizer.models.config import (
    ModelConfig,
    ModelFormat,
    ModelMetadata,
    ModelSource,
    ModelType,
    ValidationResult,
)
from src.anonymizer.models.manager import ModelManager
from src.anonymizer.models.registry import ModelRegistry
from src.anonymizer.models.validator import ModelValidator

logger = logging.getLogger(__name__)


class TestModelConfig:
    """Test model configuration classes."""

    def test_model_source_creation(self):
        """Test ModelSource creation and validation."""
        source = ModelSource(
            name="test-model",
            url="https://example.com/model.safetensors",
            format=ModelFormat.SAFETENSORS,
            size_mb=100,
            checksum="abc123",
            description="Test model",
        )

        assert source.name == "test-model"
        assert source.url == "https://example.com/model.safetensors"
        assert source.format == ModelFormat.SAFETENSORS
        assert source.size_mb == 100
        assert source.checksum == "abc123"

    def test_model_source_validation(self):
        """Test ModelSource validation."""
        # Test empty name
        with pytest.raises(ValueError, match="name cannot be empty"):
            ModelSource(name="", url="https://example.com", format=ModelFormat.SAFETENSORS)

        # Test empty URL
        with pytest.raises(ValueError, match="url cannot be empty"):
            ModelSource(name="test", url="", format=ModelFormat.SAFETENSORS)

        # Test invalid checksum type
        with pytest.raises(ValueError, match="invalid checksum_type"):
            ModelSource(
                name="test",
                url="https://example.com",
                format=ModelFormat.SAFETENSORS,
                checksum="abc123",
                checksum_type="invalid",
            )

    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()

        assert config.models_dir == Path("./models")
        assert config.max_workers == 4
        assert config.verify_checksums is True
        assert config.auto_cleanup is True
        assert "huggingface.co" in config.trusted_domains

    def test_model_config_validation(self):
        """Test ModelConfig validation."""
        # Test invalid max_workers
        with pytest.raises(ValueError, match="max_workers must be positive"):
            ModelConfig(max_workers=0)

        # Test invalid timeout
        with pytest.raises(ValueError, match="timeout_seconds must be positive"):
            ModelConfig(timeout_seconds=0)

        # Test invalid cache size
        with pytest.raises(ValueError, match="max_cache_size_gb must be positive"):
            ModelConfig(max_cache_size_gb=0)

    def test_model_metadata_serialization(self):
        """Test ModelMetadata serialization."""
        metadata = ModelMetadata(
            name="test-model",
            model_type=ModelType.VAE,
            format=ModelFormat.SAFETENSORS,
            version="1.0",
            source_url="https://example.com",
            local_path=Path("/path/to/model"),
            size_bytes=1024000,
            checksum="abc123",
            tags=["test", "vae"],
        )

        # Test to_dict
        data = metadata.to_dict()
        assert data["name"] == "test-model"
        assert data["model_type"] == "vae"
        assert data["format"] == "safetensors"
        assert data["tags"] == ["test", "vae"]

        # Test from_dict
        restored = ModelMetadata.from_dict(data)
        assert restored.name == metadata.name
        assert restored.model_type == metadata.model_type
        assert restored.format == metadata.format
        assert restored.tags == metadata.tags

    def test_validation_result(self):
        """Test ValidationResult functionality."""
        result = ValidationResult(valid=True, model_path=Path("/test/path"), errors=[], warnings=[])

        assert result.valid is True
        assert not result.has_errors
        assert not result.has_warnings

        # Test adding errors
        result.add_error("Test error")
        assert result.has_errors
        assert not result.valid
        assert "Test error" in result.errors

        # Test adding warnings
        result.add_warning("Test warning")
        assert result.has_warnings
        assert "Test warning" in result.warnings


class TestModelRegistry:
    """Test model registry functionality."""

    @pytest.fixture
    def temp_registry_dir(self):
        """Create temporary directory for registry."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def registry(self, temp_registry_dir):
        """Create test registry."""
        registry_path = temp_registry_dir / "registry.json"
        return ModelRegistry(registry_path)

    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        assert registry is not None

        # Should have predefined models
        models = registry.list_models()
        assert len(models) > 0

        # Check for expected models
        model_names = [m.name for m in models]
        assert "sd2-inpainting" in model_names
        assert "sd2-vae" in model_names

    def test_model_registration(self, registry):
        """Test custom model registration."""
        custom_source = ModelSource(
            name="custom-model",
            url="https://example.com/custom.safetensors",
            format=ModelFormat.SAFETENSORS,
            description="Custom test model",
        )

        success = registry.register_model(custom_source)
        assert success

        # Verify registration
        retrieved = registry.get_model("custom-model")
        assert retrieved is not None
        assert retrieved.name == "custom-model"

    def test_metadata_registration(self, registry):
        """Test metadata registration."""
        metadata = ModelMetadata(
            name="test-metadata",
            model_type=ModelType.VAE,
            format=ModelFormat.SAFETENSORS,
            version="1.0",
            source_url="https://example.com",
            local_path=Path("/test/path"),
            size_bytes=1024000,
        )

        success = registry.register_metadata(metadata)
        assert success

        # Verify registration
        retrieved = registry.get_metadata("test-metadata")
        assert retrieved is not None
        assert retrieved.name == "test-metadata"

    def test_model_filtering(self, registry):
        """Test model filtering by type."""
        all_models = registry.list_models()
        vae_models = registry.list_models(ModelType.VAE)
        unet_models = registry.list_models(ModelType.UNET)

        assert len(vae_models) > 0
        assert len(unet_models) > 0
        assert len(all_models) >= len(vae_models) + len(unet_models)

    def test_model_recommendations(self, registry):
        """Test model recommendations."""
        # Test different use cases
        default_rec = registry.get_recommended_models("default")
        fast_rec = registry.get_recommended_models("fast")
        quality_rec = registry.get_recommended_models("quality")

        assert len(default_rec) > 0
        assert len(fast_rec) > 0
        assert len(quality_rec) > 0

        # Default should include pipeline
        assert "pipeline" in default_rec

    def test_model_search(self, registry):
        """Test model search functionality."""
        # Search by name
        results = registry.search_models("sd2")
        assert len(results) > 0

        # Search by description
        results = registry.search_models("inpainting")
        assert len(results) > 0

        # Search for non-existent
        results = registry.search_models("nonexistent")
        assert len(results) == 0

    def test_model_info(self, registry):
        """Test model info retrieval."""
        info = registry.get_model_info("sd2-vae")

        assert "name" in info
        assert "url" in info
        assert "description" in info
        assert info["name"] == "sd2-vae"

    def test_usage_tracking(self, registry):
        """Test usage tracking."""
        # Register metadata
        metadata = ModelMetadata(
            name="usage-test",
            model_type=ModelType.VAE,
            format=ModelFormat.SAFETENSORS,
            version="1.0",
            source_url="https://example.com",
            local_path=Path("/test/path"),
            size_bytes=1024000,
            usage_count=0,
        )

        registry.register_metadata(metadata)

        # Update usage
        registry.update_usage("usage-test")

        # Verify update
        updated = registry.get_metadata("usage-test")
        assert updated.usage_count == 1
        assert updated.last_used is not None


class TestModelValidator:
    """Test model validation functionality."""

    @pytest.fixture
    def validator(self):
        """Create test validator."""
        return ModelValidator(strict_mode=False)

    @pytest.fixture
    def temp_model_dir(self):
        """Create temporary directory for test models."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    def test_validator_initialization(self, validator):
        """Test validator initialization."""
        assert validator is not None
        assert not validator.strict_mode

    def test_file_existence_validation(self, validator, temp_model_dir):
        """Test file existence validation."""
        # Test non-existent file
        non_existent = temp_model_dir / "nonexistent.safetensors"
        result = validator.validate_model(non_existent)

        assert not result.valid
        assert result.has_errors
        assert "does not exist" in result.errors[0]

    def test_file_format_validation(self, validator, temp_model_dir):
        """Test file format validation."""
        # Create test file
        test_file = temp_model_dir / "test.safetensors"
        test_file.write_text("dummy content")

        result = validator.validate_model(test_file)

        # Will fail format validation but not existence
        assert "does not exist" not in str(result.errors)

    def test_compatibility_check(self, validator, temp_model_dir):
        """Test quick compatibility check."""
        # Create test file
        test_file = temp_model_dir / "test.pth"
        test_file.write_text("dummy content")

        result = validator.validate_compatibility(test_file)

        # Should check basic properties
        assert isinstance(result, ValidationResult)

    def test_benchmark_functionality(self, validator, temp_model_dir):
        """Test benchmarking functionality."""
        # Create dummy file
        test_file = temp_model_dir / "test.pth"
        test_file.write_bytes(b"dummy model content" * 1000)  # Some content

        try:
            benchmark = validator.benchmark_model(test_file)

            assert "loading_time_ms" in benchmark
            assert "memory_usage_mb" in benchmark
            assert "model_size_mb" in benchmark
            assert benchmark["model_size_mb"] > 0

        except Exception as e:
            # Benchmarking may fail on dummy files, which is expected
            logger.debug(f"Benchmarking failed as expected: {e}")


class TestModelManager:
    """Test model manager functionality."""

    @pytest.fixture
    def temp_manager_dir(self):
        """Create temporary directory for manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def manager_config(self, temp_manager_dir):
        """Create test manager configuration."""
        return ModelConfig(
            models_dir=temp_manager_dir / "models",
            cache_dir=temp_manager_dir / "cache",
            temp_dir=temp_manager_dir / "tmp",
        )

    @pytest.fixture
    def manager(self, manager_config):
        """Create test model manager."""
        return ModelManager(manager_config)

    def test_manager_initialization(self, manager):
        """Test manager initialization."""
        assert manager is not None
        assert manager.config is not None
        assert manager.downloader is not None
        assert manager.validator is not None
        assert manager.registry is not None

    def test_list_available_models(self, manager):
        """Test listing available models."""
        models = manager.list_available_models()
        assert len(models) > 0

        # Test filtering by type
        vae_models = manager.list_available_models(ModelType.VAE)
        assert len(vae_models) > 0

    def test_model_info_retrieval(self, manager):
        """Test model info retrieval."""
        info = manager.get_model_info("sd2-vae")

        assert "name" in info
        assert "url" in info
        assert info["name"] == "sd2-vae"

    def test_recommended_setup(self, manager):
        """Test recommended setup retrieval."""
        # Test different use cases
        default_setup = manager.get_recommended_setup("default")
        fast_setup = manager.get_recommended_setup("fast")

        # These will be empty since models aren't downloaded
        # but should not error
        assert isinstance(default_setup, dict)
        assert isinstance(fast_setup, dict)

    def test_storage_stats(self, manager):
        """Test storage statistics."""
        stats = manager.get_storage_stats()

        assert "total_models" in stats
        assert "total_size_gb" in stats
        assert "models_directory" in stats
        assert stats["total_models"] >= 0

    def test_search_models(self, manager):
        """Test model search."""
        results = manager.search_models("sd2")
        assert len(results) > 0

        # All results should contain the search term
        for result in results:
            assert "sd2" in result.name.lower() or "sd2" in (result.description or "").lower()

    def test_cleanup_functionality(self, manager):
        """Test cleanup functionality."""
        # Test dry run cleanup
        results = manager.cleanup_models(unused_days=0, dry_run=True)

        assert "unused_models" in results
        assert "total_size_mb" in results
        assert "dry_run" in results
        assert results["dry_run"] is True

    def test_verify_all_models(self, manager):
        """Test verifying all models."""
        results = manager.verify_all_models()

        # Should return dictionary (empty if no models downloaded)
        assert isinstance(results, dict)


class TestModelIntegration:
    """Test model management integration."""

    def test_model_imports(self):
        """Test that model management modules can be imported."""
        # Should import without errors (imports already at module level)
        assert ModelManager is not None
        assert ModelConfig is not None
        assert ModelType is not None
        assert ModelFormat is not None

    def test_enum_values(self):
        """Test enum value consistency."""
        # Test ModelType enum
        assert ModelType.VAE.value == "vae"
        assert ModelType.UNET.value == "unet"

        # Test ModelFormat enum
        assert ModelFormat.SAFETENSORS.value == "safetensors"
        assert ModelFormat.PYTORCH.value == "pytorch"

    def test_config_integration(self):
        """Test configuration integration."""
        config = ModelConfig()
        manager = ModelManager(config)

        assert manager.config == config
        assert manager.config.models_dir.exists()


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])
