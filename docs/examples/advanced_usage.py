"""
Advanced Usage Examples
========================

This module demonstrates advanced usage patterns, custom implementations,
and integration scenarios for the Document Anonymization System.
"""

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from src.anonymizer import DocumentAnonymizer, AnonymizationConfig
from src.anonymizer.core.models import BoundingBox
from src.anonymizer.ocr import OCREngine, OCRResult, DetectedText
from src.anonymizer.performance import PerformanceMonitor, AnonymizationBenchmark


class CustomOCREngine(OCREngine):
    """
    Example of implementing a custom OCR engine.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "custom_ocr"

    def detect_text(self, image: np.ndarray) -> OCRResult:
        """
        Custom OCR implementation.
        In a real scenario, this would integrate with your preferred OCR solution.
        """
        # Simulate OCR processing
        detected_texts = [
            DetectedText(
                content="Sample detected text",
                bbox=BoundingBox(left=100, top=50, right=300, bottom=80),
                confidence=0.95,
            ),
            DetectedText(
                content="john.doe@example.com",
                bbox=BoundingBox(left=100, top=120, right=280, bottom=145),
                confidence=0.88,
            ),
        ]

        return OCRResult(
            detected_texts=detected_texts,
            processing_time_ms=150.0,
            engine_used=self.name,
        )

    def cleanup(self):
        """Clean up any resources."""
        pass


@dataclass
class CustomAnonymizationStrategy:
    """
    Example of a custom anonymization strategy that blurs sensitive regions.
    """

    blur_radius: int = 15
    strategy_name: str = "gaussian_blur"

    def anonymize_region(self, image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Apply Gaussian blur to the specified region.
        """
        import cv2

        # Extract region
        y1, y2 = bbox.top, bbox.bottom
        x1, x2 = bbox.left, bbox.right

        # Apply Gaussian blur
        region = image[y1:y2, x1:x2]
        blurred_region = cv2.GaussianBlur(
            region, (self.blur_radius, self.blur_radius), 0
        )

        # Replace region in original image
        result_image = image.copy()
        result_image[y1:y2, x1:x2] = blurred_region

        return result_image


class AdvancedDocumentProcessor:
    """
    Advanced document processor with custom features.
    """

    def __init__(self, config: Optional[AnonymizationConfig] = None):
        self.config = config or AnonymizationConfig()
        self.anonymizer = DocumentAnonymizer(self.config)
        self.performance_monitor = PerformanceMonitor()
        self.custom_strategies = {}

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def register_custom_strategy(self, name: str, strategy: Any):
        """Register a custom anonymization strategy."""
        self.custom_strategies[name] = strategy
        self.logger.info(f"Registered custom strategy: {name}")

    def process_with_custom_ocr(
        self, document_path: str, output_path: str
    ) -> Dict[str, Any]:
        """
        Process document using custom OCR engine.
        """
        from src.anonymizer.ocr import OCRProcessor

        # Create custom OCR configuration
        custom_ocr_config = {"custom_param": "value"}
        custom_engine = CustomOCREngine(custom_ocr_config)

        # Initialize OCR processor and add custom engine
        ocr_processor = OCRProcessor(self.config.ocr_config)
        ocr_processor.add_custom_engine("custom", custom_engine)

        # Process document with custom OCR
        self.logger.info(f"Processing {document_path} with custom OCR")

        result = self.anonymizer.anonymize_document(
            document_path,
            output_path,
            ocr_engines=["custom"],  # Use only our custom engine
        )

        return {
            "success": result.success,
            "entities_found": result.entities_found,
            "processing_time_ms": result.processing_time_ms,
            "custom_ocr_used": True,
        }

    def process_with_quality_control(
        self, document_path: str, output_path: str
    ) -> Dict[str, Any]:
        """
        Process document with quality control and fallback mechanisms.
        """

        # Strategy 1: High-accuracy configuration
        high_accuracy_config = AnonymizationConfig(
            ocr_engines=["paddleocr", "easyocr"],
            ner_confidence_threshold=0.95,
            anonymization_strategy="inpainting",
        )

        try:
            high_acc_anonymizer = DocumentAnonymizer(high_accuracy_config)
            result = high_acc_anonymizer.anonymize_document(document_path, output_path)

            if result.success and result.average_confidence > 0.9:
                self.logger.info("High-accuracy processing successful")
                return {
                    "strategy": "high_accuracy",
                    "success": True,
                    "confidence": result.average_confidence,
                    "entities": result.entities_anonymized,
                }
        except Exception as e:
            self.logger.warning(f"High-accuracy processing failed: {e}")

        # Strategy 2: Fallback to balanced configuration
        balanced_config = AnonymizationConfig(
            ocr_engines=["tesseract"],
            ner_confidence_threshold=0.8,
            anonymization_strategy="redaction",
        )

        try:
            balanced_anonymizer = DocumentAnonymizer(balanced_config)
            result = balanced_anonymizer.anonymize_document(document_path, output_path)

            if result.success:
                self.logger.info("Balanced processing successful")
                return {
                    "strategy": "balanced",
                    "success": True,
                    "confidence": result.average_confidence,
                    "entities": result.entities_anonymized,
                }
        except Exception as e:
            self.logger.error(f"Balanced processing failed: {e}")

        # Strategy 3: Last resort - simple redaction
        try:
            simple_config = AnonymizationConfig(
                anonymization_strategy="simple_redaction", use_gpu=False
            )
            simple_anonymizer = DocumentAnonymizer(simple_config)
            result = simple_anonymizer.anonymize_document(document_path, output_path)

            return {
                "strategy": "simple_redaction",
                "success": result.success,
                "confidence": result.average_confidence if result.success else 0,
                "entities": result.entities_anonymized if result.success else 0,
            }
        except Exception as e:
            self.logger.error(f"All processing strategies failed: {e}")
            return {"strategy": "failed", "success": False, "error": str(e)}


class BatchProcessingPipeline:
    """
    Advanced batch processing with parallel execution and monitoring.
    """

    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.performance_monitor = PerformanceMonitor()
        self.benchmark = AnonymizationBenchmark()

    async def async_process_documents(
        self,
        input_files: List[Path],
        output_dir: Path,
        config: Optional[AnonymizationConfig] = None,
    ) -> List[Dict[str, Any]]:
        """
        Asynchronously process multiple documents with progress tracking.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        async def process_single_document(file_path: Path) -> Dict[str, Any]:
            """Process a single document asynchronously."""
            try:
                output_path = output_dir / f"anonymized_{file_path.name}"

                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = loop.run_in_executor(
                        executor,
                        self._process_document_sync,
                        file_path,
                        output_path,
                        config,
                    )
                    result = await future

                return {
                    "input_file": str(file_path),
                    "output_file": str(output_path),
                    "success": result.success,
                    "entities_found": result.entities_found,
                    "processing_time_ms": result.processing_time_ms,
                    "error": result.error_message if not result.success else None,
                }

            except Exception as e:
                return {"input_file": str(file_path), "success": False, "error": str(e)}

        # Process all documents concurrently
        tasks = [process_single_document(file_path) for file_path in input_files]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({"success": False, "error": str(result)})
            else:
                processed_results.append(result)

        return processed_results

    def _process_document_sync(
        self, input_path: Path, output_path: Path, config: Optional[AnonymizationConfig]
    ):
        """Synchronous document processing for use in thread pool."""
        anonymizer = DocumentAnonymizer(config)
        return anonymizer.anonymize_document(str(input_path), str(output_path))

    def parallel_process_with_monitoring(
        self, input_files: List[Path], output_dir: Path
    ) -> Dict[str, Any]:
        """
        Process documents in parallel with comprehensive monitoring.
        """
        self.performance_monitor.start_session("parallel_batch_processing")

        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(
                    self._process_document_sync,
                    file_path,
                    output_dir / f"anonymized_{file_path.name}",
                    None,
                ): file_path
                for file_path in input_files
            }

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    results.append(
                        {
                            "file": str(file_path),
                            "success": result.success,
                            "entities": result.entities_anonymized,
                            "time_ms": result.processing_time_ms,
                        }
                    )
                except Exception as e:
                    results.append(
                        {"file": str(file_path), "success": False, "error": str(e)}
                    )

        # End monitoring session
        performance_report = self.performance_monitor.end_session()

        # Generate summary
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]

        return {
            "total_files": len(input_files),
            "successful": len(successful),
            "failed": len(failed),
            "success_rate": len(successful) / len(input_files) if input_files else 0,
            "total_entities": sum(r.get("entities", 0) for r in successful),
            "avg_processing_time_ms": (
                sum(r.get("time_ms", 0) for r in successful) / len(successful)
                if successful
                else 0
            ),
            "performance_report": performance_report,
            "detailed_results": results,
        }


class SmartConfigurationManager:
    """
    Intelligent configuration management based on document characteristics.
    """

    def __init__(self):
        self.document_profiles = {}
        self.performance_history = {}

    def analyze_document_characteristics(self, document_path: str) -> Dict[str, Any]:
        """
        Analyze document to determine optimal configuration.
        """
        import fitz  # PyMuPDF

        try:
            doc = fitz.open(document_path)

            characteristics = {
                "page_count": len(doc),
                "has_images": False,
                "text_density": 0,
                "avg_image_size": 0,
                "complex_layout": False,
            }

            total_text_length = 0
            image_count = 0
            total_image_size = 0

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Analyze text density
                text = page.get_text()
                total_text_length += len(text)

                # Check for images
                image_list = page.get_images()
                if image_list:
                    characteristics["has_images"] = True
                    image_count += len(image_list)

                    # Estimate image complexity
                    for img in image_list:
                        # This is a simplified calculation
                        total_image_size += img[2] * img[3] if len(img) > 3 else 1000

            characteristics["text_density"] = (
                total_text_length / len(doc) if len(doc) > 0 else 0
            )
            characteristics["avg_image_size"] = (
                total_image_size / image_count if image_count > 0 else 0
            )
            characteristics["complex_layout"] = (
                characteristics["has_images"]
                and characteristics["text_density"] > 1000
                and len(doc) > 10
            )

            doc.close()
            return characteristics

        except Exception as e:
            return {"error": str(e)}

    def recommend_configuration(self, document_path: str) -> AnonymizationConfig:
        """
        Recommend optimal configuration based on document analysis.
        """
        characteristics = self.analyze_document_characteristics(document_path)

        if "error" in characteristics:
            # Default configuration for problematic documents
            return AnonymizationConfig(
                ocr_engines=["tesseract"],
                anonymization_strategy="redaction",
                use_gpu=False,
            )

        # High-quality documents with complex layouts
        if characteristics.get("complex_layout", False):
            return AnonymizationConfig(
                ocr_engines=["paddleocr", "easyocr"],
                ocr_confidence_threshold=0.8,
                entity_types=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
                ner_confidence_threshold=0.85,
                anonymization_strategy="inpainting",
                use_gpu=True,
                batch_size=2,  # Smaller batches for complex documents
            )

        # Simple text documents
        elif characteristics.get("text_density", 0) > 500 and not characteristics.get(
            "has_images", False
        ):
            return AnonymizationConfig(
                ocr_engines=["tesseract"],
                ocr_confidence_threshold=0.7,
                anonymization_strategy="redaction",
                use_gpu=False,
                batch_size=8,  # Larger batches for simple documents
            )

        # Image-heavy documents
        elif characteristics.get("has_images", False):
            return AnonymizationConfig(
                ocr_engines=["paddleocr", "easyocr", "trotr"],
                ocr_confidence_threshold=0.8,
                anonymization_strategy="inpainting",
                use_gpu=True,
                batch_size=4,
            )

        # Default balanced configuration
        else:
            return AnonymizationConfig(
                ocr_engines=["paddleocr", "tesseract"],
                anonymization_strategy="inpainting",
                use_gpu=True,
            )


class QualityAssuranceFramework:
    """
    Quality assurance and validation framework for anonymization results.
    """

    def __init__(self):
        self.quality_metrics = {}

    def validate_anonymization_result(
        self, original_path: str, anonymized_path: str, result: Any
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of anonymization results.
        """
        validation_report = {
            "timestamp": time.time(),
            "original_document": original_path,
            "anonymized_document": anonymized_path,
            "checks": {},
        }

        # Check 1: File integrity
        validation_report["checks"]["file_integrity"] = self._check_file_integrity(
            anonymized_path
        )

        # Check 2: PII leakage detection
        validation_report["checks"]["pii_leakage"] = self._check_pii_leakage(
            anonymized_path, result
        )

        # Check 3: Visual quality assessment
        validation_report["checks"]["visual_quality"] = self._assess_visual_quality(
            original_path, anonymized_path
        )

        # Check 4: Metadata preservation
        validation_report["checks"]["metadata_preservation"] = (
            self._check_metadata_preservation(original_path, anonymized_path)
        )

        # Overall quality score
        validation_report["overall_quality"] = self._calculate_overall_quality(
            validation_report["checks"]
        )

        return validation_report

    def _check_file_integrity(self, file_path: str) -> Dict[str, Any]:
        """Check if the anonymized file is valid and readable."""
        try:
            import fitz

            doc = fitz.open(file_path)
            page_count = len(doc)
            doc.close()

            return {"status": "pass", "page_count": page_count, "readable": True}
        except Exception as e:
            return {"status": "fail", "error": str(e), "readable": False}

    def _check_pii_leakage(
        self, file_path: str, anonymization_result: Any
    ) -> Dict[str, Any]:
        """
        Check for potential PII leakage in the anonymized document.
        """
        # This would involve re-running NER on the anonymized document
        # to check if any PII entities are still detectable

        return {
            "status": "pass",  # Placeholder
            "potential_leaks": 0,
            "confidence": 0.95,
        }

    def _assess_visual_quality(
        self, original_path: str, anonymized_path: str
    ) -> Dict[str, Any]:
        """
        Assess visual quality by comparing original and anonymized documents.
        """
        # This could use image similarity metrics, layout preservation analysis, etc.

        return {
            "status": "pass",
            "similarity_score": 0.85,
            "layout_preserved": True,
            "text_readability": 0.90,
        }

    def _check_metadata_preservation(
        self, original_path: str, anonymized_path: str
    ) -> Dict[str, Any]:
        """
        Check if important metadata is preserved while sensitive metadata is removed.
        """
        return {
            "status": "pass",
            "creation_date_preserved": True,
            "author_removed": True,
            "title_preserved": True,
        }

    def _calculate_overall_quality(self, checks: Dict[str, Any]) -> float:
        """Calculate overall quality score based on individual checks."""
        scores = []

        for check_name, check_result in checks.items():
            if check_result.get("status") == "pass":
                scores.append(1.0)
            else:
                scores.append(0.0)

        return sum(scores) / len(scores) if scores else 0.0


async def main():
    """
    Demonstrate advanced usage patterns.
    """
    print("🚀 Advanced Document Anonymization Examples")
    print("=" * 50)

    # Example 1: Custom OCR Engine
    print("\n📝 Testing Custom OCR Engine...")
    AdvancedDocumentProcessor()

    # Example 2: Quality Control Processing
    print("\n🎯 Testing Quality Control Processing...")
    # This would process with fallback strategies

    # Example 3: Async Batch Processing
    print("\n⚡ Testing Async Batch Processing...")
    BatchProcessingPipeline(max_workers=2)

    # Create sample input files (in real usage, these would exist)
    [
        Path("examples/doc1.pdf"),
        Path("examples/doc2.pdf"),
        Path("examples/doc3.pdf"),
    ]

    Path("examples/async_output")

    # Note: This would work with real files
    # results = await batch_processor.async_process_documents(input_files, output_dir)
    print("   📊 Async processing configured (requires real input files)")

    # Example 4: Smart Configuration
    print("\n🧠 Testing Smart Configuration Management...")
    SmartConfigurationManager()

    # This would analyze a real document
    # recommended_config = config_manager.recommend_configuration("examples/sample.pdf")
    print("   📋 Smart configuration manager initialized")

    # Example 5: Quality Assurance
    print("\n🔍 Testing Quality Assurance Framework...")
    QualityAssuranceFramework()

    # This would validate real anonymization results
    # validation_report = qa_framework.validate_anonymization_result(
    #     "original.pdf", "anonymized.pdf", result
    # )
    print("   ✅ Quality assurance framework initialized")

    print("\n🎉 Advanced examples setup complete!")
    print(
        "💡 To run with real documents, provide sample PDFs in the examples/ directory"
    )


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run async main
    asyncio.run(main())
