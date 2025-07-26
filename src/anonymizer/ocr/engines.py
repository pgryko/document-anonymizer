"""
OCR Engine Implementations
===========================

Multiple OCR engine implementations with consistent interfaces.
Supports TrOCR, PaddleOCR, EasyOCR, and Tesseract with fallback strategies.
"""

import logging
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
import cv2
from PIL import Image

from .models import DetectedText, OCRResult, OCRConfig, OCREngine
from ..core.models import BoundingBox
from ..core.exceptions import ValidationError, InferenceError

logger = logging.getLogger(__name__)


class BaseOCREngine(ABC):
    """Base class for all OCR engines."""

    def __init__(self, config: OCRConfig):
        self.config = config
        self.is_initialized = False
        self._setup_logging()

    def _setup_logging(self):
        """Setup engine-specific logging."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the OCR engine. Returns True if successful."""
        pass

    @abstractmethod
    def detect_text(self, image: np.ndarray) -> OCRResult:
        """Detect text in image. Returns OCRResult."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up engine resources."""
        pass

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Apply preprocessing to improve OCR accuracy."""
        if not self.config.enable_preprocessing:
            return image

        processed = image.copy()

        # Resize if needed
        if self.config.resize_factor != 1.0:
            h, w = processed.shape[:2]
            new_h, new_w = int(h * self.config.resize_factor), int(
                w * self.config.resize_factor
            )
            processed = cv2.resize(
                processed, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4
            )

        # Convert to grayscale for processing
        if len(processed.shape) == 3:
            gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
        else:
            gray = processed.copy()

        # Contrast enhancement
        if self.config.contrast_enhancement:
            gray = cv2.equalizeHist(gray)

        # Noise reduction
        if self.config.noise_reduction:
            gray = cv2.medianBlur(gray, 3)

        # Convert back to RGB if original was color
        if len(image.shape) == 3:
            processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            processed = gray

        return processed

    def validate_image(self, image: np.ndarray) -> bool:
        """Validate input image."""
        if image is None:
            raise ValidationError("Image cannot be None")

        if len(image.shape) not in [2, 3]:
            raise ValidationError(
                f"Image must be 2D or 3D array, got {len(image.shape)}D"
            )

        if image.size == 0:
            raise ValidationError("Image cannot be empty")

        h, w = image.shape[:2]
        if h < 10 or w < 10:
            raise ValidationError(f"Image too small: {w}x{h}, minimum 10x10")

        return True


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR engine implementation - fast and accurate."""

    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.ocr = None

    def initialize(self) -> bool:
        """Initialize PaddleOCR."""
        try:
            import paddleocr

            # Initialize PaddleOCR
            self.ocr = paddleocr.PaddleOCR(
                use_angle_cls=self.config.detect_orientation,
                lang=self.config.languages[0] if self.config.languages else "en",
                use_gpu=self.config.use_gpu,
                show_log=False,
            )

            self.is_initialized = True
            self.logger.info("PaddleOCR initialized successfully")
            return True

        except ImportError:
            self.logger.warning(
                "PaddleOCR not available - install with: pip install paddleocr"
            )
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize PaddleOCR: {e}")
            return False

    def detect_text(self, image: np.ndarray) -> OCRResult:
        """Detect text using PaddleOCR."""
        if not self.is_initialized:
            raise InferenceError("PaddleOCR not initialized")

        self.validate_image(image)
        start_time = time.time()

        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Run OCR
            result = self.ocr.ocr(processed_image, cls=self.config.detect_orientation)

            # Parse results
            detected_texts = []
            if result and result[0]:
                for line in result[0]:
                    if len(line) >= 2:
                        bbox_coords = line[0]
                        text_info = line[1]

                        if len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]

                            # Convert bbox coordinates
                            bbox = self._convert_bbox(bbox_coords)

                            # Filter by confidence and length
                            if (
                                confidence >= self.config.min_confidence_threshold
                                and self.config.min_text_length
                                <= len(text)
                                <= self.config.max_text_length
                            ):

                                detected_text = DetectedText(
                                    text=text,
                                    bbox=bbox,
                                    confidence=confidence,
                                    language=(
                                        self.config.languages[0]
                                        if self.config.languages
                                        else "en"
                                    ),
                                )
                                detected_texts.append(detected_text)

            processing_time = (time.time() - start_time) * 1000

            return OCRResult(
                detected_texts=detected_texts,
                processing_time_ms=processing_time,
                engine_used=OCREngine.PADDLEOCR,
                image_size=(image.shape[1], image.shape[0]),
                success=True,
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"PaddleOCR detection failed: {e}")
            return OCRResult(
                detected_texts=[],
                processing_time_ms=processing_time,
                engine_used=OCREngine.PADDLEOCR,
                image_size=(image.shape[1], image.shape[0]),
                success=False,
                errors=[str(e)],
            )

    def _convert_bbox(self, coords) -> BoundingBox:
        """Convert PaddleOCR bbox format to BoundingBox."""
        # PaddleOCR returns 4 corner points
        x_coords = [point[0] for point in coords]
        y_coords = [point[1] for point in coords]

        left = int(min(x_coords))
        top = int(min(y_coords))
        right = int(max(x_coords))
        bottom = int(max(y_coords))

        return BoundingBox(left=left, top=top, right=right, bottom=bottom)

    def cleanup(self):
        """Clean up PaddleOCR resources."""
        self.ocr = None
        self.is_initialized = False


class EasyOCREngine(BaseOCREngine):
    """EasyOCR engine implementation - supports many languages."""

    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.reader = None

    def initialize(self) -> bool:
        """Initialize EasyOCR."""
        try:
            import easyocr

            # Initialize EasyOCR
            self.reader = easyocr.Reader(
                self.config.languages if self.config.languages else ["en"],
                gpu=self.config.use_gpu,
            )

            self.is_initialized = True
            self.logger.info("EasyOCR initialized successfully")
            return True

        except ImportError:
            self.logger.warning(
                "EasyOCR not available - install with: pip install easyocr"
            )
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize EasyOCR: {e}")
            return False

    def detect_text(self, image: np.ndarray) -> OCRResult:
        """Detect text using EasyOCR."""
        if not self.is_initialized:
            raise InferenceError("EasyOCR not initialized")

        self.validate_image(image)
        start_time = time.time()

        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Run OCR
            results = self.reader.readtext(
                processed_image,
                detail=1,  # Return bounding box info
                paragraph=False,  # Detect individual text lines
            )

            # Parse results
            detected_texts = []
            for result in results:
                bbox_coords, text, confidence = result

                # Convert bbox coordinates
                bbox = self._convert_bbox(bbox_coords)

                # Filter by confidence and length
                if (
                    confidence >= self.config.min_confidence_threshold
                    and self.config.min_text_length
                    <= len(text)
                    <= self.config.max_text_length
                ):

                    detected_text = DetectedText(
                        text=text,
                        bbox=bbox,
                        confidence=confidence,
                        language=(
                            self.config.languages[0] if self.config.languages else "en"
                        ),
                    )
                    detected_texts.append(detected_text)

            processing_time = (time.time() - start_time) * 1000

            return OCRResult(
                detected_texts=detected_texts,
                processing_time_ms=processing_time,
                engine_used=OCREngine.EASYOCR,
                image_size=(image.shape[1], image.shape[0]),
                success=True,
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"EasyOCR detection failed: {e}")
            return OCRResult(
                detected_texts=[],
                processing_time_ms=processing_time,
                engine_used=OCREngine.EASYOCR,
                image_size=(image.shape[1], image.shape[0]),
                success=False,
                errors=[str(e)],
            )

    def _convert_bbox(self, coords) -> BoundingBox:
        """Convert EasyOCR bbox format to BoundingBox."""
        # EasyOCR returns 4 corner points
        x_coords = [point[0] for point in coords]
        y_coords = [point[1] for point in coords]

        left = int(min(x_coords))
        top = int(min(y_coords))
        right = int(max(x_coords))
        bottom = int(max(y_coords))

        return BoundingBox(left=left, top=top, right=right, bottom=bottom)

    def cleanup(self):
        """Clean up EasyOCR resources."""
        self.reader = None
        self.is_initialized = False


class TrOCREngine(BaseOCREngine):
    """TrOCR (Transformer OCR) engine implementation - high accuracy."""

    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.processor = None
        self.model = None
        self.device = None

    def initialize(self) -> bool:
        """Initialize TrOCR."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch

            # Set device
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() and self.config.use_gpu else "cpu"
            )

            # Load TrOCR model and processor
            model_name = (
                "microsoft/trocr-base-printed"  # or "microsoft/trocr-base-handwritten"
            )
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(
                self.device
            )

            self.is_initialized = True
            self.logger.info(f"TrOCR initialized successfully on {self.device}")
            return True

        except ImportError:
            self.logger.warning(
                "TrOCR not available - install with: pip install transformers torch"
            )
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize TrOCR: {e}")
            return False

    def detect_text(self, image: np.ndarray) -> OCRResult:
        """Detect text using TrOCR."""
        if not self.is_initialized:
            raise InferenceError("TrOCR not initialized")

        self.validate_image(image)
        start_time = time.time()

        try:
            # TrOCR requires text detection first (we use CV2 for simple detection)
            text_regions = self._detect_text_regions(image)

            detected_texts = []
            for region_bbox, region_image in text_regions:
                # Convert region to PIL Image
                pil_image = Image.fromarray(region_image)

                # Process with TrOCR
                pixel_values = self.processor(
                    images=pil_image, return_tensors="pt"
                ).pixel_values.to(self.device)

                # Generate text
                with torch.no_grad():
                    generated_ids = self.model.generate(pixel_values)
                    generated_text = self.processor.batch_decode(
                        generated_ids, skip_special_tokens=True
                    )[0]

                # Estimate confidence (TrOCR doesn't provide confidence scores)
                confidence = 0.8  # Default confidence for TrOCR

                # Filter by length
                if (
                    self.config.min_text_length
                    <= len(generated_text)
                    <= self.config.max_text_length
                ):
                    detected_text = DetectedText(
                        text=generated_text,
                        bbox=region_bbox,
                        confidence=confidence,
                        language="en",  # TrOCR is primarily English
                    )
                    detected_texts.append(detected_text)

            processing_time = (time.time() - start_time) * 1000

            return OCRResult(
                detected_texts=detected_texts,
                processing_time_ms=processing_time,
                engine_used=OCREngine.TROCR,
                image_size=(image.shape[1], image.shape[0]),
                success=True,
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"TrOCR detection failed: {e}")
            return OCRResult(
                detected_texts=[],
                processing_time_ms=processing_time,
                engine_used=OCREngine.TROCR,
                image_size=(image.shape[1], image.shape[0]),
                success=False,
                errors=[str(e)],
            )

    def _detect_text_regions(
        self, image: np.ndarray
    ) -> List[Tuple[BoundingBox, np.ndarray]]:
        """Simple text region detection using OpenCV (for TrOCR preprocessing)."""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Apply morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 7))
        opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter small regions
            if w > 30 and h > 15:
                bbox = BoundingBox(left=x, top=y, right=x + w, bottom=y + h)
                region_image = image[y : y + h, x : x + w]
                regions.append((bbox, region_image))

        return regions

    def cleanup(self):
        """Clean up TrOCR resources."""
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "processor"):
            del self.processor

        # Clear GPU cache if using CUDA
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

        self.is_initialized = False


class TesseractEngine(BaseOCREngine):
    """Tesseract OCR engine implementation - traditional OCR."""

    def __init__(self, config: OCRConfig):
        super().__init__(config)

    def initialize(self) -> bool:
        """Initialize Tesseract."""
        try:
            import pytesseract

            # Try to get Tesseract version to verify installation
            version = pytesseract.get_tesseract_version()
            self.is_initialized = True
            self.logger.info(f"Tesseract initialized successfully (version: {version})")
            return True

        except ImportError:
            self.logger.warning(
                "Tesseract not available - install with: pip install pytesseract"
            )
            return False
        except Exception as e:
            self.logger.error(f"Failed to initialize Tesseract: {e}")
            return False

    def detect_text(self, image: np.ndarray) -> OCRResult:
        """Detect text using Tesseract."""
        if not self.is_initialized:
            raise InferenceError("Tesseract not initialized")

        self.validate_image(image)
        start_time = time.time()

        try:
            import pytesseract
            from PIL import Image as PILImage

            # Preprocess image
            processed_image = self.preprocess_image(image)

            # Convert to PIL Image
            pil_image = PILImage.fromarray(processed_image)

            # Get detailed data from Tesseract
            data = pytesseract.image_to_data(
                pil_image,
                output_type=pytesseract.Output.DICT,
                lang=(
                    "+".join(self.config.languages) if self.config.languages else "eng"
                ),
            )

            # Parse results
            detected_texts = []
            n_boxes = len(data["text"])

            for i in range(n_boxes):
                confidence = float(data["conf"][i])
                text = data["text"][i].strip()

                if (
                    text
                    and confidence
                    >= self.config.min_confidence_threshold
                    * 100  # Tesseract uses 0-100 scale
                    and self.config.min_text_length
                    <= len(text)
                    <= self.config.max_text_length
                ):

                    # Create bounding box
                    bbox = BoundingBox(
                        left=data["left"][i],
                        top=data["top"][i],
                        right=data["left"][i] + data["width"][i],
                        bottom=data["top"][i] + data["height"][i],
                    )

                    detected_text = DetectedText(
                        text=text,
                        bbox=bbox,
                        confidence=confidence / 100.0,  # Convert to 0-1 scale
                        language=(
                            self.config.languages[0] if self.config.languages else "en"
                        ),
                    )
                    detected_texts.append(detected_text)

            processing_time = (time.time() - start_time) * 1000

            return OCRResult(
                detected_texts=detected_texts,
                processing_time_ms=processing_time,
                engine_used=OCREngine.TESSERACT,
                image_size=(image.shape[1], image.shape[0]),
                success=True,
            )

        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"Tesseract detection failed: {e}")
            return OCRResult(
                detected_texts=[],
                processing_time_ms=processing_time,
                engine_used=OCREngine.TESSERACT,
                image_size=(image.shape[1], image.shape[0]),
                success=False,
                errors=[str(e)],
            )

    def cleanup(self):
        """Clean up Tesseract resources."""
        # Tesseract doesn't need explicit cleanup
        self.is_initialized = False


# Engine factory
def create_ocr_engine(engine_type: OCREngine, config: OCRConfig) -> BaseOCREngine:
    """Factory function to create OCR engines."""
    engine_map = {
        OCREngine.PADDLEOCR: PaddleOCREngine,
        OCREngine.EASYOCR: EasyOCREngine,
        OCREngine.TROCR: TrOCREngine,
        OCREngine.TESSERACT: TesseractEngine,
    }

    if engine_type not in engine_map:
        raise ValueError(f"Unsupported OCR engine: {engine_type}")

    return engine_map[engine_type](config)
