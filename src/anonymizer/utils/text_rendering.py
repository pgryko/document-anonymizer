"""Text rendering utilities for generating synthetic text images."""

import logging
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.anonymizer.core.exceptions import PreprocessingError, ValidationError

logger = logging.getLogger(__name__)

# Text rendering constants
MAX_TEXT_LENGTH = 1000


class FontManager:
    """Manages font loading and fallbacks."""

    DEFAULT_FONTS: ClassVar[list[str]] = [
        "Arial.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "NotoSans-Regular.ttf",
    ]

    SYSTEM_FONT_PATHS: ClassVar[list[str]] = [
        "/System/Library/Fonts/",  # macOS
        "/usr/share/fonts/",  # Linux
        "C:/Windows/Fonts/",  # Windows
    ]

    def __init__(self):
        self._font_cache = {}
        self._default_font = None

    def get_font(self, font_name: str | None = None, size: int = 32) -> ImageFont.ImageFont:
        """Get font with fallback handling."""
        cache_key = (font_name, size)

        if cache_key in self._font_cache:
            return self._font_cache[cache_key]

        font = self._load_font(font_name, size)
        self._font_cache[cache_key] = font
        return font

    def _load_font(self, font_name: str | None, size: int) -> ImageFont.ImageFont:
        """Load font with fallbacks."""
        try:
            # Try specific font name first
            if font_name:
                font = self._try_load_font(font_name, size)
                if font:
                    return font

            # Try default fonts
            for default_font in self.DEFAULT_FONTS:
                font = self._try_load_font(default_font, size)
                if font:
                    return font

            # Try system font paths
            for font_path in self.SYSTEM_FONT_PATHS:
                for default_font in self.DEFAULT_FONTS:
                    full_path = Path(font_path) / default_font
                    font = self._try_load_font(str(full_path), size)
                    if font:
                        return font

            # Fallback to default PIL font
            logger.warning("Could not load any TrueType fonts, using default")
            return ImageFont.load_default()

        except Exception:
            logger.exception("Font loading failed")
            return ImageFont.load_default()

    def _try_load_font(self, font_path: str, size: int) -> ImageFont.ImageFont | None:
        """Try to load a specific font."""
        try:
            font = ImageFont.truetype(font_path, size)
            logger.debug(f"Loaded font: {font_path}")
        except OSError:
            return None
        else:
            return font


class TextRenderer:
    """Renders text as images for training data generation."""

    def __init__(
        self,
        font_manager: FontManager | None = None,
        default_font_size: int = 32,
        default_image_size: tuple[int, int] = (384, 384),
    ):
        self.font_manager = font_manager or FontManager()
        self.default_font_size = default_font_size
        self.default_image_size = default_image_size

    def render_text(  # noqa: PLR0912
        self,
        text: str,
        font_name: str | None = None,
        font_size: int | None = None,
        image_size: tuple[int, int] | None = None,
        background_color: str | tuple[int, int, int] = "white",
        text_color: str | tuple[int, int, int] = "black",
        align: str = "center",
    ) -> Image.Image:
        """Render text as PIL Image.

        Args:
            text: Text to render
            font_name: Font name (optional)
            font_size: Font size (optional)
            image_size: Output image size (optional)
            background_color: Background color
            text_color: Text color
            align: Text alignment ("left", "center", "right")

        Returns:
            PIL Image with rendered text

        """
        # Validate inputs
        if not text or not text.strip():
            msg = "Text cannot be empty"
            raise ValidationError(msg)

        if len(text) > MAX_TEXT_LENGTH:
            msg = "Text too long"
            raise ValidationError(msg)

        # Use defaults if not specified
        font_size = font_size or self.default_font_size
        image_size = image_size or self.default_image_size

        try:
            # Load font
            font = self.font_manager.get_font(font_name, font_size)

            # Create image
            image = Image.new("RGB", image_size, color=background_color)
            draw = ImageDraw.Draw(image)

            # Handle multiline text
            lines = text.strip().split("\n")

            # Calculate text layout
            line_heights = []
            line_widths = []

            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]
                line_widths.append(line_width)
                line_heights.append(line_height)

            # Calculate total text dimensions
            total_width = max(line_widths) if line_widths else 0
            total_height = sum(line_heights) + (len(lines) - 1) * font_size // 4  # line spacing

            # Validate text fits in image
            if total_width > image_size[0] or total_height > image_size[1]:
                logger.warning(f"Text may not fit: {total_width}x{total_height} in {image_size}")

            # Calculate starting position
            if align == "center":
                start_x = (image_size[0] - total_width) // 2
            elif align == "right":
                start_x = image_size[0] - total_width - 10
            else:  # left
                start_x = 10

            start_y = (image_size[1] - total_height) // 2

            # Draw each line
            current_y = start_y
            for i, line in enumerate(lines):
                if align == "center":
                    line_x = (image_size[0] - line_widths[i]) // 2
                elif align == "right":
                    line_x = image_size[0] - line_widths[i] - 10
                else:
                    line_x = start_x

                draw.text((line_x, current_y), line, font=font, fill=text_color)
                current_y += line_heights[i] + font_size // 4

        except Exception as e:
            msg = f"Text rendering failed: {e}"
            raise PreprocessingError(msg) from e
        else:
            return image

    def render_text_batch(self, texts: list[str], **kwargs) -> list[Image.Image]:
        """Render multiple texts efficiently."""
        if not texts:
            msg = "Text list cannot be empty"
            raise ValidationError(msg)

        images = []
        for text in texts:
            try:
                image = self.render_text(text, **kwargs)
                images.append(image)
            except Exception as e:
                logger.warning(f"Failed to render text '{text}': {e}")
                # Create blank image as fallback
                image_size = kwargs.get("image_size", self.default_image_size)
                blank = Image.new("RGB", image_size, color="white")
                images.append(blank)

        return images

    def render_text_array(self, text: str, **kwargs) -> np.ndarray:
        """Render text and return as numpy array."""
        image = self.render_text(text, **kwargs)
        return np.array(image)

    def estimate_text_size(
        self,
        text: str,
        font_name: str | None = None,
        font_size: int | None = None,
    ) -> tuple[int, int]:
        """Estimate text size without rendering."""
        try:
            font_size = font_size or self.default_font_size
            font = self.font_manager.get_font(font_name, font_size)

            # Create dummy image for measurement
            dummy_img = Image.new("RGB", (1, 1))
            draw = ImageDraw.Draw(dummy_img)

            # Handle multiline text
            lines = text.strip().split("\n")

            max_width = 0
            total_height = 0

            for line in lines:
                bbox = draw.textbbox((0, 0), line, font=font)
                line_width = bbox[2] - bbox[0]
                line_height = bbox[3] - bbox[1]

                max_width = max(max_width, line_width)
                total_height += line_height

            # Add line spacing
            if len(lines) > 1:
                total_height += (len(lines) - 1) * font_size // 4

        except Exception as e:
            logger.warning(f"Text size estimation failed: {e}")
            return 0, 0
        else:
            return max_width, total_height


# Convenience functions
def render_text_simple(
    text: str,
    size: int = 32,
    image_size: tuple[int, int] = (384, 384),
) -> np.ndarray:
    """Simple text rendering function."""
    renderer = TextRenderer(default_font_size=size, default_image_size=image_size)
    return renderer.render_text_array(text)
