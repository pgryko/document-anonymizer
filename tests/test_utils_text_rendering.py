"""
Unit tests for text rendering utilities - Imperative style.

Tests text rendering for training data generation.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from src.anonymizer.core.exceptions import PreprocessingError, ValidationError
from src.anonymizer.utils.text_rendering import (
    FontManager,
    TextRenderer,
    render_text_simple,
)


class TestFontManager:
    """Test FontManager font loading and fallback."""

    def test_font_manager_initialization(self):
        """Test font manager initialization."""
        font_manager = FontManager()

        assert font_manager._font_cache == {}
        assert font_manager._default_font is None

    def test_get_font_successful_load(self):
        """Test successful font loading."""
        font_manager = FontManager()

        with patch.object(font_manager, "_try_load_font") as mock_try_load:
            mock_font = Mock()
            mock_try_load.return_value = mock_font

            font = font_manager.get_font("Arial.ttf", 32)

            assert font == mock_font
            mock_try_load.assert_called_with("Arial.ttf", 32)

    def test_get_font_caching(self):
        """Test font caching mechanism."""
        font_manager = FontManager()

        with patch.object(font_manager, "_load_font") as mock_load:
            mock_font = Mock()
            mock_load.return_value = mock_font

            # First call
            font1 = font_manager.get_font("Arial.ttf", 32)
            # Second call with same parameters
            font2 = font_manager.get_font("Arial.ttf", 32)

            assert font1 == font2
            assert mock_load.call_count == 1  # Should only load once

    def test_get_font_fallback_to_default_fonts(self):
        """Test fallback to default fonts when specific font fails."""
        font_manager = FontManager()

        with patch.object(font_manager, "_try_load_font") as mock_try_load:
            # First call (specific font) fails, second call (default) succeeds
            mock_font = Mock()
            mock_try_load.side_effect = [None, mock_font]

            font = font_manager.get_font("NonExistent.ttf", 32)

            assert font == mock_font
            assert mock_try_load.call_count >= 2

    def test_get_font_fallback_to_pil_default(self):
        """Test fallback to PIL default font when all TrueType fonts fail."""
        font_manager = FontManager()

        with (
            patch.object(font_manager, "_try_load_font", return_value=None),
            patch("PIL.ImageFont.load_default") as mock_default,
        ):
            mock_default_font = Mock()
            mock_default.return_value = mock_default_font

            font = font_manager.get_font("NonExistent.ttf", 32)

            assert font == mock_default_font
            mock_default.assert_called_once()

    def test_try_load_font_successful(self):
        """Test successful font loading."""
        font_manager = FontManager()

        with patch("PIL.ImageFont.truetype") as mock_truetype:
            mock_font = Mock()
            mock_truetype.return_value = mock_font

            font = font_manager._try_load_font("Arial.ttf", 32)

            assert font == mock_font
            mock_truetype.assert_called_once_with("Arial.ttf", 32)

    def test_try_load_font_failure(self):
        """Test font loading failure."""
        font_manager = FontManager()

        with patch("PIL.ImageFont.truetype", side_effect=OSError):
            font = font_manager._try_load_font("NonExistent.ttf", 32)

            assert font is None

    def test_load_font_error_handling(self):
        """Test error handling in font loading."""
        font_manager = FontManager()

        with (
            patch.object(font_manager, "_try_load_font", side_effect=Exception("Unexpected error")),
            patch("PIL.ImageFont.load_default") as mock_default,
        ):
            mock_default_font = Mock()
            mock_default.return_value = mock_default_font

            font = font_manager._load_font("Problem.ttf", 32)

            assert font == mock_default_font
            mock_default.assert_called_once()


class TestTextRenderer:
    """Test TextRenderer implementation."""

    def test_text_renderer_initialization_default(self):
        """Test text renderer initialization with defaults."""
        renderer = TextRenderer()

        assert renderer.default_font_size == 32
        assert renderer.default_image_size == (384, 384)
        assert renderer.font_manager is not None

    def test_text_renderer_initialization_custom(self):
        """Test text renderer initialization with custom parameters."""
        font_manager = FontManager()
        renderer = TextRenderer(
            font_manager=font_manager,
            default_font_size=24,
            default_image_size=(256, 256),
        )

        assert renderer.default_font_size == 24
        assert renderer.default_image_size == (256, 256)
        assert renderer.font_manager == font_manager

    def test_render_text_simple(self):
        """Test rendering simple text."""
        renderer = TextRenderer(default_image_size=(256, 256))

        text = "Hello World"
        image = renderer.render_text(text)

        assert isinstance(image, Image.Image)
        assert image.size == (256, 256)
        assert image.mode == "RGB"

    def test_render_text_multiline(self):
        """Test rendering multiline text."""
        renderer = TextRenderer()

        text = "Line 1\nLine 2\nLine 3"
        image = renderer.render_text(text)

        assert isinstance(image, Image.Image)
        assert image.size == renderer.default_image_size
        assert image.mode == "RGB"

    def test_render_text_custom_colors(self):
        """Test rendering with custom colors."""
        renderer = TextRenderer()

        text = "Custom Colors"
        image = renderer.render_text(text, background_color="black", text_color="white")

        assert isinstance(image, Image.Image)
        assert image.mode == "RGB"

    def test_render_text_custom_alignment(self):
        """Test rendering with different text alignments."""
        renderer = TextRenderer()

        text = "Aligned Text"

        # Test all alignments
        for align in ["left", "center", "right"]:
            image = renderer.render_text(text, align=align)
            assert isinstance(image, Image.Image)

    def test_render_text_custom_font_size(self):
        """Test rendering with custom font size."""
        renderer = TextRenderer()

        text = "Big Text"
        image = renderer.render_text(text, font_size=48)

        assert isinstance(image, Image.Image)

    def test_render_text_custom_image_size(self):
        """Test rendering with custom image size."""
        renderer = TextRenderer()

        text = "Custom Size"
        image = renderer.render_text(text, image_size=(512, 512))

        assert image.size == (512, 512)

    def test_render_text_empty_text_validation(self):
        """Test validation of empty text."""
        renderer = TextRenderer()

        with pytest.raises(ValidationError, match="Text cannot be empty"):
            renderer.render_text("")

        with pytest.raises(ValidationError, match="Text cannot be empty"):
            renderer.render_text("   ")  # Only whitespace

    def test_render_text_too_long_validation(self):
        """Test validation of text that's too long."""
        renderer = TextRenderer()

        long_text = "x" * 1001  # Over the 1000 character limit

        with pytest.raises(ValidationError, match="Text too long"):
            renderer.render_text(long_text)

    def test_render_text_font_manager_integration(self):
        """Test integration with font manager."""
        mock_font_manager = Mock()
        mock_font = Mock()
        mock_font_manager.get_font.return_value = mock_font

        renderer = TextRenderer(font_manager=mock_font_manager)

        with patch("PIL.Image.new") as mock_image_new:
            mock_image = Mock()
            mock_draw = Mock()
            mock_image_new.return_value = mock_image

            with patch("PIL.ImageDraw.Draw", return_value=mock_draw):
                # Mock textbbox to return reasonable bounds
                mock_draw.textbbox.return_value = (0, 0, 100, 30)

                renderer.render_text("Test", font_name="Arial.ttf", font_size=24)

                # Verify font manager was called
                mock_font_manager.get_font.assert_called_once_with("Arial.ttf", 24)

    def test_render_text_error_handling(self):
        """Test error handling in text rendering."""
        renderer = TextRenderer()

        with (
            patch("PIL.Image.new", side_effect=Exception("PIL error")),
            pytest.raises(PreprocessingError, match="Text rendering failed"),
        ):
            renderer.render_text("Test")

    def test_render_text_long_text_warning(self):
        """Test warning for text that may not fit."""
        renderer = TextRenderer(default_image_size=(100, 100))  # Small image

        # Text that will likely exceed image bounds
        long_text = "This is a very long text that probably won't fit in a small image"

        with patch("PIL.ImageDraw.Draw") as mock_draw_class:
            mock_draw = Mock()
            mock_draw_class.return_value = mock_draw
            # Mock textbbox to return bounds larger than image
            mock_draw.textbbox.return_value = (0, 0, 200, 200)  # Larger than 100x100

            # Should not raise error, just warn
            image = renderer.render_text(long_text)
            assert isinstance(image, Image.Image)

    def test_render_text_batch(self):
        """Test batch text rendering."""
        renderer = TextRenderer()

        texts = ["Text 1", "Text 2", "Text 3"]
        images = renderer.render_text_batch(texts)

        assert len(images) == 3
        for image in images:
            assert isinstance(image, Image.Image)
            assert image.size == renderer.default_image_size

    def test_render_text_batch_empty_list(self):
        """Test batch rendering with empty list."""
        renderer = TextRenderer()

        with pytest.raises(ValidationError, match="Text list cannot be empty"):
            renderer.render_text_batch([])

    def test_render_text_batch_error_handling(self):
        """Test batch rendering with individual text failures."""
        renderer = TextRenderer()

        texts = ["Good Text", "", "Another Good Text"]  # Middle text is invalid

        # Should create blank image for failed text
        images = renderer.render_text_batch(texts)

        assert len(images) == 3
        assert all(isinstance(img, Image.Image) for img in images)

    def test_render_text_array(self):
        """Test rendering text as numpy array."""
        renderer = TextRenderer()

        text = "Array Test"
        array = renderer.render_text_array(text)

        assert isinstance(array, np.ndarray)
        assert array.shape == (
            *renderer.default_image_size[::-1],
            3,
        )  # Height, Width, Channels
        assert array.dtype == np.uint8

    def test_estimate_text_size(self):
        """Test text size estimation."""
        renderer = TextRenderer()

        text = "Test Text"

        with patch.object(renderer.font_manager, "get_font") as mock_get_font:
            mock_font = Mock()
            mock_get_font.return_value = mock_font

            with patch("PIL.Image.new") as mock_image_new:
                mock_image = Mock()
                mock_image_new.return_value = mock_image

                with patch("PIL.ImageDraw.Draw") as mock_draw_class:
                    mock_draw = Mock()
                    mock_draw_class.return_value = mock_draw
                    mock_draw.textbbox.return_value = (0, 0, 100, 30)

                    width, height = renderer.estimate_text_size(text)

                    assert width == 100
                    assert height == 30

    def test_estimate_text_size_multiline(self):
        """Test text size estimation for multiline text."""
        renderer = TextRenderer()

        text = "Line 1\nLine 2"

        with patch.object(renderer.font_manager, "get_font") as mock_get_font:
            mock_font = Mock()
            mock_get_font.return_value = mock_font

            with patch("PIL.Image.new") as mock_image_new:
                mock_image = Mock()
                mock_image_new.return_value = mock_image

                with patch("PIL.ImageDraw.Draw") as mock_draw_class:
                    mock_draw = Mock()
                    mock_draw_class.return_value = mock_draw
                    # Each line returns 100x20 bbox
                    mock_draw.textbbox.side_effect = [(0, 0, 100, 20), (0, 0, 80, 20)]

                    width, height = renderer.estimate_text_size(text, font_size=32)

                    # Width should be max of lines (100), height should include spacing
                    assert width == 100
                    assert height > 20  # Should be greater than single line due to spacing

    def test_estimate_text_size_error_handling(self):
        """Test text size estimation error handling."""
        renderer = TextRenderer()

        with patch.object(renderer.font_manager, "get_font", side_effect=Exception("Font error")):
            width, height = renderer.estimate_text_size("Test")

            # Should return zeros on error
            assert width == 0
            assert height == 0


class TestConvenienceFunction:
    """Test convenience function."""

    def test_render_text_simple_function(self):
        """Test simple text rendering convenience function."""
        text = "Simple Test"
        array = render_text_simple(text, size=24, image_size=(200, 200))

        assert isinstance(array, np.ndarray)
        assert array.shape == (200, 200, 3)
        assert array.dtype == np.uint8

    def test_render_text_simple_defaults(self):
        """Test simple rendering with default parameters."""
        text = "Default Test"
        array = render_text_simple(text)

        assert isinstance(array, np.ndarray)
        assert array.shape == (384, 384, 3)  # Default size
        assert array.dtype == np.uint8


class TestTextRenderingEdgeCases:
    """Test edge cases and special scenarios."""

    def test_render_unicode_text(self):
        """Test rendering unicode text."""
        renderer = TextRenderer()

        unicode_text = "Hello 世界 🌍"

        # Should not raise error even if font doesn't support all characters
        image = renderer.render_text(unicode_text)
        assert isinstance(image, Image.Image)

    def test_render_text_with_special_characters(self):
        """Test rendering text with special characters."""
        renderer = TextRenderer()

        special_text = "Special: !@#$%^&*()_+-=[]{}|;:,.<>?"
        image = renderer.render_text(special_text)

        assert isinstance(image, Image.Image)

    def test_render_very_long_single_line(self):
        """Test rendering very long single line."""
        renderer = TextRenderer(default_image_size=(1000, 100))

        long_text = "This is a very long line of text that should extend beyond normal bounds"
        image = renderer.render_text(long_text)

        assert isinstance(image, Image.Image)
        assert image.size == (1000, 100)

    def test_render_many_lines(self):
        """Test rendering many lines of text."""
        renderer = TextRenderer(default_image_size=(400, 600))

        many_lines = "\n".join([f"Line {i}" for i in range(20)])
        image = renderer.render_text(many_lines)

        assert isinstance(image, Image.Image)

    def test_render_single_character(self):
        """Test rendering single character."""
        renderer = TextRenderer()

        image = renderer.render_text("A")

        assert isinstance(image, Image.Image)

    def test_text_color_tuple(self):
        """Test rendering with color as RGB tuple."""
        renderer = TextRenderer()

        image = renderer.render_text(
            "Color Test",
            background_color=(255, 255, 255),  # White
            text_color=(0, 0, 0),  # Black
        )

        assert isinstance(image, Image.Image)

    def test_render_with_very_small_font(self):
        """Test rendering with very small font size."""
        renderer = TextRenderer()

        image = renderer.render_text("Tiny", font_size=8)

        assert isinstance(image, Image.Image)

    def test_render_with_very_large_font(self):
        """Test rendering with very large font size."""
        renderer = TextRenderer(default_image_size=(800, 800))

        image = renderer.render_text("BIG", font_size=200)

        assert isinstance(image, Image.Image)
