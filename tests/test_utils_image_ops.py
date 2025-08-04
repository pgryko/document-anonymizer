"""
Unit tests for image operations utilities - Imperative style.

Tests safe image processing with bounds checking and error handling.
"""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

from src.anonymizer.core.exceptions import (
    CannotPadToSmallerSizeError,
    ChannelConversionError,
    ColorConversionChannelError,
    ImageCropError,
    ImageMemoryTooLargeError,
    ImageResizeError,
    ImageTooLargeError,
    UnexpectedImageDtypeError,
    UnexpectedImageShapeError,
    ValidationError,
)
from src.anonymizer.utils.image_ops import ImageProcessor, safe_crop, safe_resize


class TestImageProcessor:
    """Test ImageProcessor with safety validation."""

    def test_validate_image_array_valid_rgb(self):
        """Test validating valid RGB image array."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        result = ImageProcessor.validate_image_array(image)

        assert result is True

    def test_validate_image_array_valid_grayscale(self):
        """Test validating valid grayscale image array."""
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        result = ImageProcessor.validate_image_array(image)

        assert result is True

    def test_validate_image_array_valid_rgba(self):
        """Test validating valid RGBA image array."""
        image = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)

        result = ImageProcessor.validate_image_array(image)

        assert result is True

    def test_validate_image_array_invalid_type(self):
        """Test validation fails for non-numpy array."""
        image = [[1, 2, 3], [4, 5, 6]]  # List, not numpy array

        with pytest.raises(UnexpectedImageDtypeError, match="Unexpected image dtype"):
            ImageProcessor.validate_image_array(image)

    def test_validate_image_array_invalid_dimensions(self):
        """Test validation fails for invalid dimensions."""
        # 1D array
        image = np.random.randint(0, 255, (256,), dtype=np.uint8)

        with pytest.raises(UnexpectedImageShapeError, match="Unexpected image shape"):
            ImageProcessor.validate_image_array(image)

        # 4D array
        image = np.random.randint(0, 255, (2, 256, 256, 3), dtype=np.uint8)

        with pytest.raises(UnexpectedImageShapeError, match="Unexpected image shape"):
            ImageProcessor.validate_image_array(image)

    def test_validate_image_array_invalid_channels(self):
        """Test validation fails for invalid number of channels."""
        # 5 channels (invalid)
        image = np.random.randint(0, 255, (256, 256, 5), dtype=np.uint8)

        with pytest.raises(UnexpectedImageShapeError, match="Unexpected image shape"):
            ImageProcessor.validate_image_array(image)

    def test_validate_image_array_too_large_dimensions(self):
        """Test validation fails for too large images."""
        # Create image larger than MAX_DIMENSION (8192)
        with pytest.raises(ImageTooLargeError, match="Image too large"):
            # Don't actually create the large array to save memory
            ImageProcessor.validate_image_array(np.zeros((10000, 10000, 3), dtype=np.uint8))

    def test_validate_image_array_too_much_memory(self):
        """Test validation fails for images requiring too much memory."""
        # Create very large array that would exceed memory limit
        # Use a smaller array but with patch to simulate memory limit
        image = np.zeros((1000, 1000, 3), dtype=np.float64)  # Should be fine normally

        with (
            patch.object(ImageProcessor, "MAX_MEMORY_BYTES", 1000),  # Very small limit
            pytest.raises(ImageMemoryTooLargeError, match="Image too large in memory"),
        ):
            ImageProcessor.validate_image_array(image)

    def test_safe_resize_scale_down(self):
        """Test safe resizing with scale down."""
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        resized = ImageProcessor.safe_resize(image, (256, 256))

        assert resized.shape == (256, 256, 3)
        assert resized.dtype == np.uint8

    def test_safe_resize_scale_up(self):
        """Test safe resizing with scale up."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        resized = ImageProcessor.safe_resize(image, (512, 512))

        assert resized.shape == (512, 512, 3)
        assert resized.dtype == np.uint8

    def test_safe_resize_to_square(self):
        """Test resizing to square maintaining aspect ratio."""
        image = np.random.randint(0, 255, (200, 400, 3), dtype=np.uint8)

        resized = ImageProcessor.safe_resize(image, 300)

        # Should fit within 300x300 while maintaining aspect ratio
        assert max(resized.shape[:2]) <= 300
        assert resized.shape[2] == 3

    def test_safe_resize_grayscale(self):
        """Test resizing grayscale image."""
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        resized = ImageProcessor.safe_resize(image, (128, 128))

        assert resized.shape == (128, 128)
        assert resized.dtype == np.uint8

    def test_safe_resize_excessive_scale_factor(self):
        """Test that excessive scale factors are rejected."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Try to scale by 10x (over the 4x limit)
        with pytest.raises(ImageResizeError, match="Failed to resize image"):
            ImageProcessor.safe_resize(image, (1000, 1000), max_scale_factor=4.0)

    def test_safe_resize_output_too_large(self):
        """Test that output size limits are enforced."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Try to resize to larger than MAX_DIMENSION
        with pytest.raises(ImageResizeError, match="Failed to resize image"):
            ImageProcessor.safe_resize(image, (10000, 10000))

    def test_safe_resize_memory_limit(self):
        """Test that memory limits are enforced."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Mock memory estimation to exceed limit
        with (
            patch.object(ImageProcessor, "MAX_MEMORY_BYTES", 1000),
            pytest.raises(
                ValidationError,
                match="Output would be too large|Image too large in memory",
            ),
        ):  # Very small limit
            ImageProcessor.safe_resize(image, (1000, 1000))

    def test_safe_resize_opencv_error(self):
        """Test handling of OpenCV errors."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        with (
            patch("cv2.resize", side_effect=cv2.error("OpenCV error")),
            pytest.raises(ImageResizeError, match="Failed to resize image"),
        ):
            ImageProcessor.safe_resize(image, (128, 128))

    def test_safe_crop_valid(self):
        """Test safe cropping with valid parameters."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        crop = ImageProcessor.safe_crop(image, 50, 50, 100, 100)

        assert crop.shape == (100, 100, 3)
        assert crop.dtype == np.uint8

    def test_safe_crop_grayscale(self):
        """Test safe cropping of grayscale image."""
        image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

        crop = ImageProcessor.safe_crop(image, 50, 50, 100, 100)

        assert crop.shape == (100, 100)
        assert crop.dtype == np.uint8

    def test_safe_crop_with_padding(self):
        """Test cropping that requires padding."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Crop that extends beyond image bounds
        crop = ImageProcessor.safe_crop(image, 80, 80, 50, 50)

        # Should be padded to requested size
        assert crop.shape == (50, 50, 3)
        assert crop.dtype == np.uint8

    def test_safe_crop_invalid_size(self):
        """Test that invalid crop sizes are rejected."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        # Zero width
        with pytest.raises(ImageCropError, match="Failed to crop image"):
            ImageProcessor.safe_crop(image, 50, 50, 0, 100)

        # Negative height
        with pytest.raises(ImageCropError, match="Failed to crop image"):
            ImageProcessor.safe_crop(image, 50, 50, 100, -50)

    def test_safe_crop_bounds_checking(self):
        """Test that crop bounds are properly checked."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Crop completely outside image bounds - should still work with padding
        # Use constant padding mode to avoid empty axis issues
        crop = ImageProcessor.safe_crop(image, 150, 150, 50, 50, padding_mode="constant")

        assert crop.shape == (50, 50, 3)

    def test_pad_image_reflect_mode(self):
        """Test image padding with reflect mode."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        padded = ImageProcessor._pad_image(image, 100, 100, "reflect")

        assert padded.shape == (100, 100, 3)
        assert padded.dtype == np.uint8

    def test_pad_image_constant_mode(self):
        """Test image padding with constant mode."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        padded = ImageProcessor._pad_image(image, 100, 100, "constant")

        assert padded.shape == (100, 100, 3)
        assert padded.dtype == np.uint8

        # Check that padding is white (255)
        # Top-left corner should be white padding
        assert np.all(padded[0, 0] == 255)

    def test_pad_image_edge_mode(self):
        """Test image padding with edge mode."""
        image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)

        padded = ImageProcessor._pad_image(image, 100, 100, "edge")

        assert padded.shape == (100, 100, 3)
        assert padded.dtype == np.uint8

    def test_pad_image_grayscale(self):
        """Test padding grayscale image."""
        image = np.random.randint(0, 255, (50, 50), dtype=np.uint8)

        padded = ImageProcessor._pad_image(image, 100, 100, "reflect")

        assert padded.shape == (100, 100)
        assert padded.dtype == np.uint8

    def test_pad_image_invalid_target_size(self):
        """Test that padding to smaller size is rejected."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with pytest.raises(CannotPadToSmallerSizeError, match="Cannot pad to smaller size"):
            ImageProcessor._pad_image(image, 50, 50)

    def test_normalize_image_uint8_to_minus_one_one(self):
        """Test normalizing uint8 image to [-1, 1] range."""
        image = np.array([[[0, 127, 255]]], dtype=np.uint8)

        normalized = ImageProcessor.normalize_image(image, (-1.0, 1.0))

        expected = np.array([[[-1.0, -0.003921568, 1.0]]], dtype=np.float32)
        assert np.allclose(normalized, expected, atol=1e-3)

    def test_normalize_image_float_to_zero_one(self):
        """Test normalizing float image to [0, 1] range."""
        image = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)

        normalized = ImageProcessor.normalize_image(image, (0.0, 1.0))

        expected = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
        assert np.allclose(normalized, expected, atol=1e-6)

    def test_normalize_image_constant_value(self):
        """Test normalizing image with constant value."""
        image = np.full((10, 10, 3), 128, dtype=np.uint8)

        normalized = ImageProcessor.normalize_image(image, (-1.0, 1.0))

        # Should be all zeros when min == max
        assert np.allclose(normalized, -1.0, atol=1e-6)

    def test_normalize_image_invalid_image(self):
        """Test normalization with invalid image."""
        image = [[1, 2, 3]]  # Not a numpy array

        with pytest.raises(UnexpectedImageDtypeError):
            ImageProcessor.normalize_image(image)

    def test_convert_color_space_bgr_to_rgb(self):
        """Test BGR to RGB color space conversion."""
        # Create BGR image (Blue, Green, Red)
        bgr_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)

        rgb_image = ImageProcessor.convert_color_space(bgr_image, "BGR", "RGB")

        # BGR [255,0,0] should become RGB [0,0,255] (red in RGB)
        expected = np.array([[[0, 0, 255], [0, 255, 0], [255, 0, 0]]], dtype=np.uint8)
        assert np.array_equal(rgb_image, expected)

    def test_convert_color_space_rgb_to_bgr(self):
        """Test RGB to BGR color space conversion."""
        # Create RGB image
        rgb_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)

        bgr_image = ImageProcessor.convert_color_space(rgb_image, "RGB", "BGR")

        # RGB [255,0,0] should become BGR [0,0,255]
        expected = np.array([[[0, 0, 255], [0, 255, 0], [255, 0, 0]]], dtype=np.uint8)
        assert np.array_equal(bgr_image, expected)

    def test_convert_color_space_rgb_to_gray(self):
        """Test RGB to grayscale conversion."""
        rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        gray_image = ImageProcessor.convert_color_space(rgb_image, "RGB", "GRAY")

        assert gray_image.shape == (100, 100)
        assert gray_image.dtype == np.uint8

    def test_convert_color_space_gray_to_rgb(self):
        """Test grayscale to RGB conversion."""
        # Create actual 2D grayscale image
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        rgb_image = ImageProcessor.convert_color_space(gray_image, "GRAY", "RGB")

        assert rgb_image.shape == (100, 100, 3)
        assert rgb_image.dtype == np.uint8

    def test_convert_color_space_invalid_source_channels(self):
        """Test color conversion with wrong number of channels."""
        # Try to convert grayscale with RGB conversion
        gray_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)

        with pytest.raises(
            ColorConversionChannelError, match="Color conversion requires correct channel count"
        ):
            ImageProcessor.convert_color_space(gray_image, "BGR", "RGB")

    def test_convert_color_space_unsupported_conversion(self):
        """Test unsupported color space conversion."""
        rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with pytest.raises(ChannelConversionError, match="Failed to convert image channels"):
            ImageProcessor.convert_color_space(rgb_image, "RGB", "XYZ")

    def test_convert_color_space_opencv_error(self):
        """Test handling of OpenCV errors in color conversion."""
        rgb_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        with (
            patch("cv2.cvtColor", side_effect=cv2.error("OpenCV error")),
            pytest.raises(ChannelConversionError, match="Failed to convert image channels"),
        ):
            ImageProcessor.convert_color_space(rgb_image, "RGB", "BGR")


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    def test_safe_resize_wrapper(self):
        """Test safe_resize convenience function."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        resized = safe_resize(image, (128, 128))

        assert resized.shape == (128, 128, 3)
        assert resized.dtype == np.uint8

    def test_safe_crop_wrapper(self):
        """Test safe_crop convenience function."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        crop = safe_crop(image, 50, 50, 100, 100)

        assert crop.shape == (100, 100, 3)
        assert crop.dtype == np.uint8

    def test_safe_resize_with_kwargs(self):
        """Test safe_resize with additional keyword arguments."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        resized = safe_resize(
            image, (128, 128), interpolation=cv2.INTER_LINEAR, max_scale_factor=2.0
        )

        assert resized.shape == (128, 128, 3)

    def test_safe_crop_with_kwargs(self):
        """Test safe_crop with additional keyword arguments."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        crop = safe_crop(image, 80, 80, 50, 50, padding_mode="constant")

        assert crop.shape == (50, 50, 3)


class TestImageProcessorEdgeCases:
    """Test edge cases and error conditions."""

    def test_resize_same_size(self):
        """Test resizing to same size."""
        image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        resized = ImageProcessor.safe_resize(image, (256, 256))

        assert resized.shape == (256, 256, 3)
        assert np.array_equal(resized, image)

    def test_crop_entire_image(self):
        """Test cropping entire image."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        crop = ImageProcessor.safe_crop(image, 0, 0, 100, 100)

        assert np.array_equal(crop, image)

    def test_crop_single_pixel(self):
        """Test cropping single pixel."""
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        crop = ImageProcessor.safe_crop(image, 50, 50, 1, 1)

        assert crop.shape == (1, 1, 3)
        assert np.array_equal(crop, image[50:51, 50:51])

    def test_normalize_zero_range(self):
        """Test normalization with zero range target."""
        image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)

        # Target range with zero width
        normalized = ImageProcessor.normalize_image(image, (0.5, 0.5))

        # Should be all 0.5
        assert np.allclose(normalized, 0.5)

    def test_processing_chain(self):
        """Test chaining multiple operations."""
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)

        # Chain: resize -> crop -> normalize
        resized = ImageProcessor.safe_resize(image, (256, 256))
        cropped = ImageProcessor.safe_crop(resized, 50, 50, 156, 156)
        normalized = ImageProcessor.normalize_image(cropped, (-1.0, 1.0))

        assert cropped.shape == (156, 156, 3)
        assert normalized.shape == (156, 156, 3)
        assert normalized.dtype == np.float32
        assert -1.0 <= normalized.min() <= normalized.max() <= 1.0
