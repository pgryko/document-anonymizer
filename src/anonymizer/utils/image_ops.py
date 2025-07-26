"""Safe image operations with comprehensive error handling."""

import logging

import cv2
import numpy as np

from src.anonymizer.core.exceptions import PreprocessingError, ValidationError

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Safe image processing operations."""

    # Safety limits
    MAX_DIMENSION = 8192
    MAX_MEMORY_BYTES = 2 * 1024 * 1024 * 1024  # 2GB

    @classmethod
    def validate_image_array(cls, image: np.ndarray) -> bool:
        """Validate numpy image array."""
        if not isinstance(image, np.ndarray):
            raise ValidationError("Image must be numpy array")

        if len(image.shape) not in [2, 3]:
            raise ValidationError(f"Invalid image shape: {image.shape}")

        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            raise ValidationError(f"Invalid number of channels: {image.shape[2]}")

        h, w = image.shape[:2]
        if h > cls.MAX_DIMENSION or w > cls.MAX_DIMENSION:
            raise ValidationError(f"Image too large: {w}x{h}")

        # Check memory usage
        estimated_memory = image.nbytes
        if estimated_memory > cls.MAX_MEMORY_BYTES:
            raise ValidationError(f"Image too large in memory: {estimated_memory} bytes")

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

            if max_scale > max_scale_factor:
                raise ValidationError(f"Scale factor too large: {max_scale}")

            # Check output dimensions
            if new_w > cls.MAX_DIMENSION or new_h > cls.MAX_DIMENSION:
                raise ValidationError(f"Output size too large: {new_w}x{new_h}")

            # Estimate output memory
            channels = 1 if len(image.shape) == 2 else image.shape[2]
            estimated_memory = new_w * new_h * channels * image.itemsize
            if estimated_memory > cls.MAX_MEMORY_BYTES:
                raise ValidationError(f"Output would be too large: {estimated_memory} bytes")

            # Perform resize
            if len(image.shape) == 2:
                resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
            else:
                resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

            logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
            return resized

        except cv2.error as e:
            raise PreprocessingError(f"OpenCV resize failed: {e}")
        except Exception as e:
            raise PreprocessingError(f"Image resize failed: {e}")

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
            if width <= 0 or height <= 0:
                raise ValidationError(f"Invalid crop size: {width}x{height}")

            # Calculate actual crop bounds
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + width), min(img_h, y + height)

            # Extract crop
            crop = image[y1:y2, x1:x2] if len(image.shape) == 2 else image[y1:y2, x1:x2, :]

            # Add padding if needed
            crop_h, crop_w = crop.shape[:2]
            if crop_h < height or crop_w < width:
                crop = cls._pad_image(crop, width, height, padding_mode)

            logger.debug(f"Cropped image to {crop.shape}")
            return crop

        except Exception as e:
            raise PreprocessingError(f"Image crop failed: {e}")

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
            raise ValidationError("Cannot pad to smaller size")

        # Calculate padding
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Apply padding
        if len(image.shape) == 2:
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
        cls, image: np.ndarray, target_range: tuple[float, float] = (-1.0, 1.0)
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
            raise PreprocessingError(f"Image normalization failed: {e}")

    @classmethod
    def convert_color_space(
        cls, image: np.ndarray, source: str = "BGR", target: str = "RGB"
    ) -> np.ndarray:
        """Convert between color spaces."""
        cls.validate_image_array(image)

        # Validate input channels based on source format
        source_upper = source.upper()
        if source_upper in ["BGR", "RGB"] and len(image.shape) != 3:
            raise ValidationError("Color conversion requires 3-channel image")
        if source_upper == "GRAY" and len(image.shape) not in [2, 3]:
            raise ValidationError("Grayscale conversion requires 2D or 3D image")

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
            if conversion_key not in conversion_map:
                raise ValidationError(f"Unsupported conversion: {source} -> {target}")

            conversion_code = conversion_map[conversion_key]
            converted = cv2.cvtColor(image, conversion_code)

            logger.debug(f"Converted color space: {source} -> {target}")
            return converted

        except cv2.error as e:
            raise PreprocessingError(f"Color conversion failed: {e}")
        except Exception as e:
            raise PreprocessingError(f"Color conversion failed: {e}")


# Convenience functions
def safe_resize(image: np.ndarray, target_size: int | tuple[int, int], **kwargs) -> np.ndarray:
    """Convenient wrapper for safe image resizing."""
    return ImageProcessor.safe_resize(image, target_size, **kwargs)


def safe_crop(image: np.ndarray, x: int, y: int, width: int, height: int, **kwargs) -> np.ndarray:
    """Convenient wrapper for safe image cropping."""
    return ImageProcessor.safe_crop(image, x, y, width, height, **kwargs)
