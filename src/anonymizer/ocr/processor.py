"""
OCR Processor
=============

Main OCR processing class with multi-engine support, fallback strategies,
and integration with the document anonymization pipeline.
"""

import logging
import time

import numpy as np

from src.anonymizer.core.exceptions import InferenceError, ValidationError
from src.anonymizer.core.models import BoundingBox, TextRegion

from .engines import BaseOCREngine, create_ocr_engine
from .models import DetectedText, OCRConfig, OCRMetrics

logger = logging.getLogger(__name__)


class OCRProcessor:
    """
    Main OCR processor with multi-engine support and fallback strategies.

    Features:
    - Multiple OCR engine support (PaddleOCR, EasyOCR, TrOCR, Tesseract)
    - Automatic fallback to alternative engines
    - Text filtering and post-processing
    - Performance monitoring and metrics
    - Integration with NER pipeline
    """

    def __init__(self, config: OCRConfig):
        self.config = config
        self.primary_engine: BaseOCREngine | None = None
        self.fallback_engines: list[BaseOCREngine] = []
        self.is_initialized = False

        # Performance tracking
        self.total_processing_time = 0.0
        self.total_images_processed = 0
        self.successful_detections = 0

        logger.info(f"OCRProcessor initialized with primary engine: {config.primary_engine.value}")

    def initialize(self) -> bool:
        """Initialize OCR engines with fallback support."""
        try:
            # Initialize primary engine
            self.primary_engine = create_ocr_engine(self.config.primary_engine, self.config)
            primary_success = self.primary_engine.initialize()

            if not primary_success:
                logger.warning(
                    f"Primary engine {self.config.primary_engine.value} failed to initialize"
                )

            # Initialize fallback engines
            fallback_success_count = 0
            for engine_type in self.config.fallback_engines:
                try:
                    engine = create_ocr_engine(engine_type, self.config)
                    if engine.initialize():
                        self.fallback_engines.append(engine)
                        fallback_success_count += 1
                        logger.info(f"Fallback engine {engine_type.value} initialized successfully")
                    else:
                        logger.warning(f"Fallback engine {engine_type.value} failed to initialize")
                except Exception as e:
                    logger.warning(f"Failed to create fallback engine {engine_type.value}: {e}")

            # Check if we have at least one working engine
            self.is_initialized = primary_success or fallback_success_count > 0

            if self.is_initialized:
                total_engines = (1 if primary_success else 0) + fallback_success_count
                logger.info(f"OCR processor initialized with {total_engines} working engines")
            else:
                logger.error("No OCR engines available - check dependencies")

        except Exception:
            logger.exception("OCR initialization failed")
            return False
        else:
            return self.is_initialized
            logger.exception("Failed to initialize OCR processor")
            return False

    def extract_text_regions(
        self, image: np.ndarray, min_confidence: float | None = None
    ) -> list[DetectedText]:
        """
        Extract text regions from an image using OCR.

        Args:
            image: Input image as numpy array
            min_confidence: Override default confidence threshold

        Returns:
            List of DetectedText objects with bounding boxes and metadata
        """
        if not self.is_initialized:
            msg = "OCR processor not initialized"
            raise InferenceError(msg)

        if image is None or image.size == 0:
            msg = "Invalid image provided"
            raise ValidationError(msg)

        start_time = time.time()

        # Try primary engine first
        result = None

        if self.primary_engine and self.primary_engine.is_initialized:
            try:
                result = self.primary_engine.detect_text(image)

                if result.success and len(result.detected_texts) > 0:
                    logger.debug(f"Primary engine {self.config.primary_engine.value} succeeded")
                else:
                    logger.warning(
                        f"Primary engine {self.config.primary_engine.value} returned no results"
                    )

            except Exception as e:
                logger.warning(f"Primary engine {self.config.primary_engine.value} failed: {e}")
                result = None

        # Try fallback engines if primary failed or returned no results
        if (
            result is None or not result.success or len(result.detected_texts) == 0
        ) and self.fallback_engines:
            logger.info("Attempting fallback engines")

            for fallback_engine in self.fallback_engines:
                try:
                    result = fallback_engine.detect_text(image)

                    if result.success and len(result.detected_texts) > 0:
                        logger.info(f"Fallback engine {result.engine_used.value} succeeded")
                        break
                    logger.warning(
                        f"Fallback engine {result.engine_used.value} returned no results"
                    )

                except Exception as e:
                    logger.warning(f"Fallback engine failed: {e}")
                    continue

        # If all engines failed
        if result is None or not result.success:
            logger.error("All OCR engines failed")
            return []

        # Apply confidence filtering
        confidence_threshold = min_confidence or self.config.min_confidence_threshold
        filtered_texts = [
            text for text in result.detected_texts if text.confidence >= confidence_threshold
        ]

        # Apply additional filtering
        filtered_texts = self._apply_text_filters(filtered_texts)

        # Update metrics
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.total_images_processed += 1
        if len(filtered_texts) > 0:
            self.successful_detections += 1

        logger.info(
            f"OCR completed: {len(filtered_texts)} text regions detected in {processing_time:.2f}s"
        )

        return filtered_texts

    def convert_to_text_regions(
        self, detected_texts: list[DetectedText], replacement_strategy: str = "generic"
    ) -> list[TextRegion]:
        """
        Convert DetectedText objects to TextRegion objects for anonymization.

        Args:
            detected_texts: List of DetectedText from OCR
            replacement_strategy: Strategy for generating replacement text

        Returns:
            List of TextRegion objects ready for anonymization
        """
        text_regions = []

        for detected_text in detected_texts:
            # Generate replacement text based on strategy
            if replacement_strategy == "generic":
                replacement_text = "[TEXT]"
            elif replacement_strategy == "length_preserving":
                replacement_text = "X" * len(detected_text.text)
            elif replacement_strategy == "word_preserving":
                words = detected_text.text.split()
                replacement_text = " ".join(["[WORD]" for _ in words])
            else:
                replacement_text = "[TEXT]"

            text_region = TextRegion(
                bbox=detected_text.bbox,
                original_text=detected_text.text,
                replacement_text=replacement_text,
                confidence=detected_text.confidence,
            )
            text_regions.append(text_region)

        return text_regions

    def detect_and_convert(
        self, image: np.ndarray, replacement_strategy: str = "generic"
    ) -> list[TextRegion]:
        """
        One-step function to detect text and convert to TextRegion objects.

        Args:
            image: Input image
            replacement_strategy: Strategy for replacement text generation

        Returns:
            List of TextRegion objects ready for anonymization
        """
        detected_texts = self.extract_text_regions(image)
        return self.convert_to_text_regions(detected_texts, replacement_strategy)

    def _apply_text_filters(self, detected_texts: list[DetectedText]) -> list[DetectedText]:
        """Apply various text filters to improve detection quality."""
        filtered_texts = detected_texts.copy()

        # Filter by text length
        if self.config.filter_short_texts:
            filtered_texts = [
                text
                for text in filtered_texts
                if self.config.min_text_length <= len(text.text) <= self.config.max_text_length
            ]

        # Filter by confidence
        if self.config.filter_low_confidence:
            filtered_texts = [
                text
                for text in filtered_texts
                if text.confidence >= self.config.min_confidence_threshold
            ]

        # Merge nearby text regions if enabled
        if self.config.merge_nearby_texts:
            filtered_texts = self._merge_nearby_texts(filtered_texts)

        # Remove duplicates
        return self._remove_duplicate_texts(filtered_texts)

    def _merge_nearby_texts(self, detected_texts: list[DetectedText]) -> list[DetectedText]:
        """Merge text regions that are close together."""
        if len(detected_texts) <= 1:
            return detected_texts

        merged_texts = []
        used_indices = set()

        for i, text1 in enumerate(detected_texts):
            if i in used_indices:
                continue

            # Find nearby texts to merge
            texts_to_merge = [text1]
            used_indices.add(i)

            for j, text2 in enumerate(detected_texts[i + 1 :], i + 1):
                if j in used_indices:
                    continue

                # Calculate distance between bounding boxes
                distance = self._calculate_bbox_distance(text1.bbox, text2.bbox)

                if distance <= self.config.merge_distance_threshold:
                    texts_to_merge.append(text2)
                    used_indices.add(j)

            # Merge the texts
            if len(texts_to_merge) == 1:
                merged_texts.append(text1)
            else:
                merged_text = self._merge_detected_texts(texts_to_merge)
                merged_texts.append(merged_text)

        return merged_texts

    def _calculate_bbox_distance(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate distance between two bounding boxes."""
        # Calculate center points
        center1_x = (bbox1.left + bbox1.right) / 2
        center1_y = (bbox1.top + bbox1.bottom) / 2
        center2_x = (bbox2.left + bbox2.right) / 2
        center2_y = (bbox2.top + bbox2.bottom) / 2

        # Euclidean distance
        return ((center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2) ** 0.5

    def _merge_detected_texts(self, texts: list[DetectedText]) -> DetectedText:
        """Merge multiple DetectedText objects into one."""
        if len(texts) == 1:
            return texts[0]

        # Combine text content
        combined_text = " ".join(text.text for text in texts)

        # Calculate combined bounding box
        all_lefts = [text.bbox.left for text in texts]
        all_tops = [text.bbox.top for text in texts]
        all_rights = [text.bbox.right for text in texts]
        all_bottoms = [text.bbox.bottom for text in texts]

        combined_bbox = BoundingBox(
            left=min(all_lefts),
            top=min(all_tops),
            right=max(all_rights),
            bottom=max(all_bottoms),
        )

        # Average confidence
        avg_confidence = sum(text.confidence for text in texts) / len(texts)

        return DetectedText(
            text=combined_text,
            bbox=combined_bbox,
            confidence=avg_confidence,
            language=texts[0].language,  # Use first text's language
        )

    def _remove_duplicate_texts(self, detected_texts: list[DetectedText]) -> list[DetectedText]:
        """Remove duplicate text detections."""
        unique_texts = []
        seen_texts = set()

        for text in detected_texts:
            # Create a unique key based on text content and position
            key = f"{text.text.strip().lower()}_{text.bbox.left}_{text.bbox.top}"

            if key not in seen_texts:
                seen_texts.add(key)
                unique_texts.append(text)

        return unique_texts

    def get_metrics(self) -> OCRMetrics:
        """Get performance metrics for the OCR processor."""
        high_conf_count = 0
        low_conf_count = 0

        # This is a simplified metric - in a real implementation,
        # you'd track these during processing

        return OCRMetrics(
            total_processing_time_ms=self.total_processing_time * 1000,
            text_detection_time_ms=self.total_processing_time * 800,  # Estimate
            text_recognition_time_ms=self.total_processing_time * 200,  # Estimate
            preprocessing_time_ms=self.total_processing_time * 50,  # Estimate
            total_texts_detected=self.successful_detections,
            high_confidence_detections=high_conf_count,
            low_confidence_detections=low_conf_count,
            engine_used=self.config.primary_engine,
            fallback_attempted=len(self.fallback_engines) > 0,
        )

    def cleanup(self):
        """Clean up all OCR engines and resources."""
        if self.primary_engine:
            try:
                self.primary_engine.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up primary engine: {e}")

        for engine in self.fallback_engines:
            try:
                engine.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up fallback engine: {e}")

        self.primary_engine = None
        self.fallback_engines = []
        self.is_initialized = False

        logger.info("OCR processor cleaned up successfully")

    def __enter__(self):
        """Context manager entry."""
        if not self.is_initialized:
            self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
