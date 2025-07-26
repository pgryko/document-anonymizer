"""
Basic Usage Examples
====================

This module demonstrates basic usage patterns for the Document Anonymization System.
"""

from pathlib import Path

from src.anonymizer import AnonymizationConfig, DocumentAnonymizer
from src.anonymizer.core.models import BoundingBox


def example_simple_anonymization():
    """
    Simplest way to anonymize a document with default settings.
    """
    print("=== Simple Anonymization ===")

    # Create anonymizer with default configuration
    anonymizer = DocumentAnonymizer()

    # Process a single document
    input_file = "examples/sample_document.pdf"
    output_file = "examples/anonymized_document.pdf"

    result = anonymizer.anonymize_document(input_file, output_file)

    if result.success:
        print("✅ Successfully anonymized document!")
        print(f"   📄 Input: {result.input_path}")
        print(f"   📄 Output: {result.output_path}")
        print(f"   🔍 Found {result.entities_found} PII entities")
        print(f"   🛡️ Anonymized {result.entities_anonymized} entities")
        print(f"   ⏱️ Processing time: {result.processing_time_ms:.1f}ms")
        print(f"   📊 Average confidence: {result.average_confidence:.2f}")
    else:
        print(f"❌ Anonymization failed: {result.error_message}")


def example_custom_configuration():
    """
    Using custom configuration for specific requirements.
    """
    print("\n=== Custom Configuration ===")

    # Create custom configuration
    config = AnonymizationConfig(
        # Use only PaddleOCR for high accuracy
        ocr_engines=["paddleocr"],
        ocr_confidence_threshold=0.9,
        # Detect only specific PII types
        entity_types=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
        ner_confidence_threshold=0.85,
        # Use redaction instead of inpainting for speed
        anonymization_strategy="redaction",
        # Disable GPU for CPU-only processing
        use_gpu=False,
        # Process one document at a time
        batch_size=1,
    )

    # Create anonymizer with custom config
    anonymizer = DocumentAnonymizer(config)

    # Process document
    result = anonymizer.anonymize_document(
        "examples/business_card.pdf", "examples/anonymized_business_card.pdf"
    )

    if result.success:
        print("✅ Custom anonymization complete!")
        print(f"   🎯 Entity types: {config.entity_types}")
        print(f"   🔧 Strategy: {config.anonymization_strategy}")
        print(f"   📊 Results: {result.entities_anonymized}/{result.entities_found}")


def example_batch_processing():
    """
    Process multiple documents in batch.
    """
    print("\n=== Batch Processing ===")

    # Find all PDF files in input directory
    input_dir = Path("examples/input_documents/")
    pdf_files = list(input_dir.glob("*.pdf"))

    if not pdf_files:
        print("⚠️ No PDF files found in input directory")
        return

    print(f"📁 Found {len(pdf_files)} PDF files to process")

    # Create anonymizer
    anonymizer = DocumentAnonymizer()

    # Progress callback to track processing
    def progress_callback(completed: int, total: int):
        percentage = (completed / total) * 100
        print(f"   Progress: {completed}/{total} ({percentage:.1f}%)")

    # Process all documents in batch
    results = anonymizer.anonymize_batch(
        input_paths=pdf_files,
        output_dir="examples/anonymized_batch/",
        parallel_workers=2,  # Use 2 parallel workers
        progress_callback=progress_callback,
    )

    # Analyze results
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    print("\n📊 Batch Processing Results:")
    print(f"   ✅ Successful: {len(successful)}")
    print(f"   ❌ Failed: {len(failed)}")

    if successful:
        total_entities = sum(r.entities_found for r in successful)
        total_anonymized = sum(r.entities_anonymized for r in successful)
        avg_time = sum(r.processing_time_ms for r in successful) / len(successful)

        print(f"   🔍 Total entities found: {total_entities}")
        print(f"   🛡️ Total entities anonymized: {total_anonymized}")
        print(f"   ⏱️ Average processing time: {avg_time:.1f}ms")

    if failed:
        print("\n❌ Failed documents:")
        for result in failed:
            print(f"   • {result.input_path}: {result.error_message}")


def example_specific_entity_types():
    """
    Anonymize only specific types of PII entities.
    """
    print("\n=== Specific Entity Types ===")

    # Configuration for financial documents
    financial_config = AnonymizationConfig(
        entity_types=[
            "CREDIT_CARD",  # Credit card numbers
            "IBAN_CODE",  # International bank account numbers
            "US_SSN",  # Social Security numbers
            "PERSON",  # Person names
            "PHONE_NUMBER",  # Phone numbers
        ],
        ner_confidence_threshold=0.9,  # High confidence for financial data
        anonymization_strategy="inpainting",
    )

    anonymizer = DocumentAnonymizer(financial_config)
    result = anonymizer.anonymize_document(
        "examples/financial_statement.pdf", "examples/anonymized_financial.pdf"
    )

    print("Financial document anonymization:")
    print(f"   Entity types: {financial_config.entity_types}")
    print(f"   Success: {result.success}")
    print(f"   Entities processed: {result.entities_anonymized}")


def example_with_manual_regions():
    """
    Manually specify regions to anonymize (bypass automatic detection).
    """
    print("\n=== Manual Region Specification ===")

    # Define manual regions to anonymize
    manual_regions = [
        BoundingBox(left=100, top=50, right=300, bottom=80),  # Header region
        BoundingBox(left=50, top=200, right=250, bottom=230),  # Signature area
        BoundingBox(left=400, top=300, right=600, bottom=330),  # Phone number area
    ]

    DocumentAnonymizer()

    # Note: This is a conceptual example - the actual API might be different
    # You would need to implement manual region support in the anonymizer
    print(f"Would anonymize {len(manual_regions)} manually specified regions")
    for i, region in enumerate(manual_regions):
        print(f"   Region {i+1}: {region.width}x{region.height} at ({region.left}, {region.top})")


def example_performance_monitoring():
    """
    Monitor performance during anonymization.
    """
    print("\n=== Performance Monitoring ===")

    # Import performance monitoring
    from src.anonymizer.performance import PerformanceMonitor

    # Create monitor
    monitor = PerformanceMonitor(auto_export=False)

    # Start monitoring session
    monitor.start_session("basic_anonymization")

    # Perform anonymization
    anonymizer = DocumentAnonymizer()
    anonymizer.anonymize_document("examples/large_document.pdf", "examples/anonymized_large.pdf")

    # End monitoring and get report
    performance_report = monitor.end_session()

    print("📊 Performance Metrics:")
    print(f"   ⏱️ Duration: {performance_report['session_duration_seconds']:.2f}s")

    resource_summary = performance_report["resource_summary"]
    print(f"   🧠 Peak Memory: {resource_summary['peak_memory_mb']:.1f}MB")
    print(f"   🔄 Avg CPU: {resource_summary['cpu_percent']['avg']:.1f}%")
    print(f"   📊 Samples: {resource_summary['sample_count']}")

    if resource_summary.get("gpu_peak_memory_mb"):
        print(f"   🎮 Peak GPU Memory: {resource_summary['gpu_peak_memory_mb']:.1f}MB")


def example_error_handling():
    """
    Proper error handling patterns.
    """
    print("\n=== Error Handling ===")

    from src.anonymizer.core.exceptions import (
        InferenceError,
        NERError,
        OCRError,
        ValidationError,
    )

    anonymizer = DocumentAnonymizer()

    try:
        # Attempt to process a potentially problematic document
        result = anonymizer.anonymize_document(
            "examples/corrupted_document.pdf", "examples/output.pdf"
        )

        if result.success:
            print("✅ Document processed successfully")
        else:
            print(f"⚠️ Processing completed with issues: {result.error_message}")

    except ValidationError as e:
        print(f"❌ Input validation failed: {e}")

    except OCRError as e:
        print(f"❌ OCR processing failed: {e}")
        print("   💡 Try using a different OCR engine or preprocessed image")

    except NERError as e:
        print(f"❌ Entity recognition failed: {e}")
        print("   💡 Try using manual region specification")

    except InferenceError as e:
        print(f"❌ Model inference failed: {e}")
        print("   💡 Check GPU memory or try CPU processing")

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("   💡 Check logs for detailed error information")


def example_confidence_analysis():
    """
    Analyze confidence scores of detected entities.
    """
    print("\n=== Confidence Analysis ===")

    config = AnonymizationConfig(
        ner_confidence_threshold=0.5  # Lower threshold to see more detections
    )

    anonymizer = DocumentAnonymizer(config)
    result = anonymizer.anonymize_document("examples/mixed_confidence.pdf")

    if result.success and result.confidence_scores:
        scores = result.confidence_scores

        print("📊 Confidence Score Analysis:")
        print(f"   📈 Highest: {max(scores):.3f}")
        print(f"   📉 Lowest: {min(scores):.3f}")
        print(f"   📊 Average: {sum(scores)/len(scores):.3f}")

        # Analyze distribution
        high_confidence = [s for s in scores if s >= 0.9]
        medium_confidence = [s for s in scores if 0.7 <= s < 0.9]
        low_confidence = [s for s in scores if s < 0.7]

        print(f"   🟢 High confidence (≥0.9): {len(high_confidence)}")
        print(f"   🟡 Medium confidence (0.7-0.9): {len(medium_confidence)}")
        print(f"   🔴 Low confidence (<0.7): {len(low_confidence)}")


def example_model_management():
    """
    Demonstrate model management capabilities.
    """
    print("\n=== Model Management ===")

    from src.anonymizer.models import ModelManager

    # Create model manager
    manager = ModelManager()

    # List available models
    available_models = manager.list_available_models()
    print(f"📦 Available models: {len(available_models)}")

    # Check which models are downloaded
    downloaded_models = manager.list_downloaded_models()
    print(f"💾 Downloaded models: {len(downloaded_models)}")

    # Ensure required models are available
    print("🔄 Ensuring required models are available...")
    success = manager.ensure_models_available("default")

    if success:
        print("✅ All required models are available")
    else:
        print("⚠️ Some models failed to download")

    # Check storage statistics
    stats = manager.get_storage_stats()
    print("📊 Storage stats:")
    print(f"   📁 Models directory: {stats['models_directory']}")
    print(f"   📦 Total models: {stats['total_models']}")
    print(f"   💾 Total size: {stats['total_size_gb']:.2f} GB")


if __name__ == "__main__":
    """
    Run all examples in sequence.
    """
    print("🚀 Document Anonymization System - Basic Usage Examples")
    print("=" * 60)

    # Create example directories if they don't exist
    Path("examples").mkdir(exist_ok=True)
    Path("examples/input_documents").mkdir(exist_ok=True)
    Path("examples/anonymized_batch").mkdir(exist_ok=True)

    # Run examples
    try:
        example_simple_anonymization()
        example_custom_configuration()
        example_batch_processing()
        example_specific_entity_types()
        example_with_manual_regions()
        example_performance_monitoring()
        example_error_handling()
        example_confidence_analysis()
        example_model_management()

        print("\n🎉 All examples completed successfully!")

    except Exception as e:
        print(f"\n💥 Example execution failed: {e}")
        print("   💡 Make sure you have sample documents in the examples/ directory")
