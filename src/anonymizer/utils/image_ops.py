"""Safe image operations with comprehensive error handling."""

import logging

import cv2
import numpy as np

from src.anonymizer.core.exceptions import (
    CannotPadToSmallerSizeError,
    ChannelConversionError,
    ColorConversionChannelError,
    ImageCropError,
    ImageMemoryTooLargeError,
    ImageNormalizeError,
    ImageResizeError,
    ImageTooLargeError,
    InvalidCropSizeError,
    OutputMemoryTooLargeError,
    OutputSizeTooLargeError,
    ScaleFactorTooLargeError,
    UnexpectedImageDtypeError,
    UnexpectedImageShapeError,
    UnsupportedColorConversionError,
)

logger = logging.getLogger(__name__)

# Image processing constants
MIN_IMAGE_DIMENSIONS = 2
MAX_IMAGE_DIMENSIONS = 3
RGB_CHANNELS = 3
RGBA_CHANNELS = 4
GRAYSCALE_CHANNELS = 1
VALID_CHANNEL_COUNTS = [1, 3, 4]


class ImageProcessor:
    """Safe image processing operations."""

    # Safety limits
    MAX_DIMENSION = 8192
    MAX_MEMORY_BYTES = 2 * 1024 * 1024 * 1024  # 2GB

    @classmethod
    def validate_image_array(cls, image: np.ndarray) -> bool:
        """Validate numpy image array."""
        if not isinstance(image, np.ndarray):
            raise UnexpectedImageDtypeError(str(type(image)))

        if len(image.shape) not in [MIN_IMAGE_DIMENSIONS, MAX_IMAGE_DIMENSIONS]:
            raise UnexpectedImageShapeError(image.shape)

        if len(image.shape) == MAX_IMAGE_DIMENSIONS and image.shape[2] not in VALID_CHANNEL_COUNTS:
            raise UnexpectedImageShapeError(image.shape)

        h, w = image.shape[:2]
        if h > cls.MAX_DIMENSION or w > cls.MAX_DIMENSION:
            raise ImageTooLargeError(w, h)

        # Check memory usage
        estimated_memory = image.nbytes
        if estimated_memory > cls.MAX_MEMORY_BYTES:
            raise ImageMemoryTooLargeError(estimated_memory)

        return True

    @classmethod
    def safe_resize(
        cls,
        image: np.ndarray,
        target_size: int | tuple[int, int],
        interpolation: int = cv2.INTER_LANCZOS4,
        max_scale_factor: float = 4.0,
    ) -> np.ndarray:
        """Safely resize image with bounds checking."""
        cls.validate_image_array(image)

        try:
            h, w = image.shape[:2]

            # Calculate target dimensions
            if isinstance(target_size, int):
                # Scale to fit within square
                scale = min(target_size / w, target_size / h)
                new_w, new_h = int(w * scale), int(h * scale)
            else:
                new_w, new_h = target_size

            # Check scale factor safety
            scale_w = new_w / w
            scale_h = new_h / h
            max_scale = max(scale_w, scale_h)

            def _raise_scale_error() -> None:
                raise ScaleFactorTooLargeError(max_scale)  # noqa: TRY301

            if max_scale > max_scale_factor:
                _raise_scale_error()

            # Check output dimensions
            def _raise_size_error() -> None:
                raise OutputSizeTooLargeError(new_w, new_h)  # noqa: TRY301

            if new_w > cls.MAX_DIMENSION or new_h > cls.MAX_DIMENSION:
                _raise_size_error()

            # Estimate output memory
            channels = (
                GRAYSCALE_CHANNELS if len(image.shape) == MIN_IMAGE_DIMENSIONS else image.shape[2]
            )
            estimated_memory = new_w * new_h * channels * image.itemsize

            def _raise_memory_error() -> None:
                raise OutputMemoryTooLargeError(estimated_memory)  # noqa: TRY301

            if estimated_memory > cls.MAX_MEMORY_BYTES:
                _raise_memory_error()

            # Perform resize
            if len(image.shape) == MIN_IMAGE_DIMENSIONS:
                resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            else:
                resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

            logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")

        except cv2.error as e:
            raise ImageResizeError() from e
        except Exception as e:
            raise ImageResizeError() from e
        else:
            return resized

    @classmethod
    def safe_crop(
        cls,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        padding_mode: str = "reflect",
    ) -> np.ndarray:
        """Safely crop image with bounds checking and padding."""
        cls.validate_image_array(image)

        try:
            img_h, img_w = image.shape[:2]

            # Validate crop parameters
            def _raise_crop_size_error() -> None:
                raise InvalidCropSizeError(width, height)  # noqa: TRY301

            if width <= 0 or height <= 0:
                _raise_crop_size_error()

            # Calculate actual crop bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + width), min(img_h, y + height)

            # Extract crop
            crop = (
                image[y1:y2, x1:x2]
                if len(image.shape) == MIN_IMAGE_DIMENSIONS
                else image[y1:y2, x1:x2, :]
            )

            # Add padding if needed
            crop_h, crop_w = crop.shape[:2]
            if crop_h < height or crop_w < width:
                crop = cls._pad_image(crop, width, height, padding_mode)

            logger.debug(f"Cropped image to {crop.shape}")

        except Exception as e:
            raise ImageCropError() from e
        else:
            return crop

    @classmethod
    def _pad_image(
        cls,
        image: np.ndarray,
        target_width: int,
        target_height: int,
        padding_mode: str = "reflect",
    ) -> np.ndarray:
        """Pad image to target size."""
        h, w = image.shape[:2]

        pad_w = target_width - w
        pad_h = target_height - h

        if pad_w < 0 or pad_h < 0:
            raise CannotPadToSmallerSizeError()

        # Calculate padding
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Apply padding
        if len(image.shape) == MIN_IMAGE_DIMENSIONS:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        else:
            pad_width = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

        # Convert padding mode
        if padding_mode == "reflect":
            mode = "reflect"
        elif padding_mode == "constant":
            mode = "constant"
            # Use white for constant padding
            return np.pad(image, pad_width, mode=mode, constant_values=255)
        else:
            mode = "edge"

        return np.pad(image, pad_width, mode=mode)

    @classmethod
    def normalize_image(
        cls,
        image: np.ndarray,
        target_range: tuple[float, float] = (-1.0, 1.0),
    ) -> np.ndarray:
        """Normalize image to target range."""
        cls.validate_image_array(image)

        try:
            # Convert to float
            if image.dtype != np.float32:
                image = image.astype(np.float32)

            # Get current range
            if image.dtype == np.uint8:
                current_min, current_max = 0.0, 255.0
            else:
                current_min, current_max = image.min(), image.max()

            # Normalize to [0, 1]
            if current_max > current_min:
                normalized = (image - current_min) / (current_max - current_min)
            else:
                normalized = np.zeros_like(image)

            # Scale to target range
            target_min, target_max = target_range
            return normalized * (target_max - target_min) + target_min

        except Exception as e:
            raise ImageNormalizeError() from e

    @classmethod
    def convert_color_space(
        cls,
        image: np.ndarray,
        source: str = "BGR",
        target: str = "RGB",
    ) -> np.ndarray:
        """Convert between color spaces."""
        cls.validate_image_array(image)

        # Validate input channels based on source format
        source_upper = source.upper()
        if source_upper in ["BGR", "RGB"] and len(image.shape) != RGB_CHANNELS:
            raise ColorConversionChannelError("Color")
        if source_upper == "GRAY" and len(image.shape) not in [MIN_IMAGE_DIMENSIONS, RGB_CHANNELS]:
            raise ColorConversionChannelError("Grayscale")

        try:
            # Get conversion code
            conversion_map = {
                ("BGR", "RGB"): cv2.COLOR_BGR2RGB,
                ("RGB", "BGR"): cv2.COLOR_RGB2BGR,
                ("BGR", "GRAY"): cv2.COLOR_BGR2GRAY,
                ("RGB", "GRAY"): cv2.COLOR_RGB2GRAY,
                ("GRAY", "RGB"): cv2.COLOR_GRAY2RGB,
                ("GRAY", "BGR"): cv2.COLOR_GRAY2BGR,
            }

            conversion_key = (source.upper(), target.upper())

            def _raise_conversion_error() -> None:
                raise UnsupportedColorConversionError(source, target)  # noqa: TRY301

            if conversion_key not in conversion_map:
                _raise_conversion_error()

            conversion_code = conversion_map[conversion_key]
            converted = cv2.cvtColor(image, conversion_code)

            logger.debug(f"Converted color space: {source} -> {target}")

        except cv2.error as e:
            raise ChannelConversionError() from e
        except Exception as e:
            raise ChannelConversionError() from e
        else:
            return converted


# Convenience functions
def safe_resize(image: np.ndarray, target_size: int | tuple[int, int], **kwargs) -> np.ndarray:
    """Convenient wrapper for safe image resizing."""
    return ImageProcessor.safe_resize(image, target_size, **kwargs)


def safe_crop(image: np.ndarray, x: int, y: int, width: int, height: int, **kwargs) -> np.ndarray:
    """Convenient wrapper for safe image cropping."""
    return ImageProcessor.safe_crop(image, x, y, width, height, **kwargs)
