"""Tests for Modal.com cloud integration."""

import pytest


class TestModalIntegration:
    """Test Modal.com cloud training integration."""

    def test_modal_import_availability(self):
        """Test that Modal import is handled gracefully."""
        from src.anonymizer.cloud.modal_app import HAS_MODAL, app, train_unet, train_vae

        # Should either have Modal available or graceful fallback
        assert isinstance(HAS_MODAL, bool)
        assert app is not None
        assert train_vae is not None
        assert train_unet is not None

    @pytest.mark.skipif(condition=True, reason="Modal.com integration requires cloud credentials")
    def test_modal_config_initialization(self):
        """Test Modal configuration initialization (requires Modal)."""
        from src.anonymizer.cloud.modal_app import HAS_MODAL

        if not HAS_MODAL:
            pytest.skip("Modal not available")

        from src.anonymizer.cloud.modal_app import modal_config

        # Verify config has expected attributes
        assert hasattr(modal_config, "app_name")
        assert hasattr(modal_config, "gpu_type")
        assert hasattr(modal_config, "python_version")
        assert modal_config.app_name == "document-anonymizer"
        assert modal_config.gpu_type == "A100-40GB"
        assert modal_config.python_version == "3.12"

    def test_modal_not_available_fallback(self):
        """Test fallback behavior when Modal is not available."""
        from src.anonymizer.cloud.modal_app import ModalNotAvailableError
        from src.anonymizer.core.exceptions import ModalNotAvailableError as CoreModalError

        # Verify exception exists and is properly typed
        assert ModalNotAvailableError is CoreModalError
