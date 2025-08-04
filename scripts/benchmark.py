#!/usr/bin/env python3
"""
Performance Benchmark Script
=============================

Command-line utility for running performance benchmarks and monitoring
resource usage of the document anonymization pipeline.
"""

import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import click

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.anonymizer.performance import AnonymizationBenchmark, PerformanceMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--results-dir",
    type=click.Path(path_type=Path),
    default="./benchmark_results",
    help="Directory to store results",
)
@click.pass_context
def cli(ctx, verbose: bool, results_dir: Path):
    """Document Anonymization Performance Benchmark Tool."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    # Store in context
    ctx.ensure_object(dict)
    ctx.obj["results_dir"] = results_dir
    ctx.obj["verbose"] = verbose


@cli.command()
@click.option("--image-size", default="1024x768", help="Image size (WxH)")
@click.option("--num-documents", default=5, help="Number of documents to test")
@click.option("--iterations", default=3, help="Number of benchmark iterations")
@click.pass_context
def document_loading(ctx, image_size: str, num_documents: int, iterations: int):
    """Benchmark document loading and preprocessing."""
    results_dir = ctx.obj["results_dir"]

    # Parse image size
    try:
        width, height = map(int, image_size.split("x"))
        size = (width, height)
    except ValueError:
        click.echo("❌ Invalid image size format. Use WIDTHxHEIGHT (e.g., 1024x768)", err=True)
        return

    click.echo("🔄 Benchmarking document loading...")
    click.echo(f"   📐 Image size: {size}")
    click.echo(f"   📄 Documents: {num_documents}")
    click.echo(f"   🔄 Iterations: {iterations}")

    benchmark = AnonymizationBenchmark()
    results = []

    for i in range(iterations):
        click.echo(f"\n⏳ Running iteration {i+1}/{iterations}...")

        result = benchmark.benchmark_document_loading(image_size=size, num_documents=num_documents)

        results.append(result)

        if result.success:
            click.echo(
                f"✅ Iteration {i+1}: {result.duration_ms:.1f}ms, "
                f"peak memory: {result.peak_memory_mb:.1f}MB"
            )
        else:
            click.echo(f"❌ Iteration {i+1} failed: {result.error_message}")

    # Calculate averages
    successful_results = [r for r in results if r.success]
    if successful_results:
        avg_duration = sum(r.duration_ms for r in successful_results) / len(successful_results)
        avg_memory = sum(r.peak_memory_mb for r in successful_results) / len(successful_results)

        click.echo("\n📊 Results Summary:")
        click.echo(f"   Average duration: {avg_duration:.1f}ms")
        click.echo(f"   Average peak memory: {avg_memory:.1f}MB")
        click.echo(f"   Success rate: {len(successful_results)}/{len(results)}")

        # Calculate throughput
        if avg_duration > 0:
            docs_per_second = (num_documents * 1000) / avg_duration
            click.echo(f"   Throughput: {docs_per_second:.2f} documents/second")

    # Save results
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"document_loading_{timestamp}.json"

    benchmark.save_benchmark_results(results, results_file)
    click.echo(f"\n💾 Results saved to: {results_file}")


@cli.command()
@click.option("--image-size", default="512x512", help="Image size for OCR")
@click.option("--num-images", default=10, help="Number of images to process")
@click.option("--iterations", default=3, help="Number of benchmark iterations")
@click.pass_context
def text_detection(ctx, image_size: str, num_images: int, iterations: int):
    """Benchmark OCR text detection performance."""
    results_dir = ctx.obj["results_dir"]

    # Parse image size
    try:
        width, height = map(int, image_size.split("x"))
        size = (width, height)
    except ValueError:
        click.echo("❌ Invalid image size format. Use WIDTHxHEIGHT (e.g., 512x512)", err=True)
        return

    click.echo("🔄 Benchmarking OCR text detection...")
    click.echo(f"   📐 Image size: {size}")
    click.echo(f"   🖼️ Images: {num_images}")
    click.echo(f"   🔄 Iterations: {iterations}")

    benchmark = AnonymizationBenchmark()
    results = []

    for i in range(iterations):
        click.echo(f"\n⏳ Running iteration {i+1}/{iterations}...")

        result = benchmark.benchmark_text_detection(image_size=size, num_images=num_images)

        results.append(result)

        if result.success:
            click.echo(
                f"✅ Iteration {i+1}: {result.duration_ms:.1f}ms, "
                f"peak memory: {result.peak_memory_mb:.1f}MB"
            )
        else:
            click.echo(f"❌ Iteration {i+1} failed: {result.error_message}")

    # Save and display results
    _display_and_save_results(results, "text_detection", results_dir, num_images)


@cli.command()
@click.option("--num-texts", default=100, help="Number of texts to analyze")
@click.option("--iterations", default=3, help="Number of benchmark iterations")
@click.pass_context
def pii_detection(ctx, num_texts: int, iterations: int):
    """Benchmark NER PII detection performance."""
    results_dir = ctx.obj["results_dir"]

    click.echo("🔄 Benchmarking NER PII detection...")
    click.echo(f"   📝 Texts: {num_texts}")
    click.echo(f"   🔄 Iterations: {iterations}")

    benchmark = AnonymizationBenchmark()
    results = []

    for i in range(iterations):
        click.echo(f"\n⏳ Running iteration {i+1}/{iterations}...")

        result = benchmark.benchmark_pii_detection(num_texts=num_texts)

        results.append(result)

        if result.success:
            click.echo(
                f"✅ Iteration {i+1}: {result.duration_ms:.1f}ms, "
                f"peak memory: {result.peak_memory_mb:.1f}MB"
            )
        else:
            click.echo(f"❌ Iteration {i+1} failed: {result.error_message}")

    # Save and display results
    _display_and_save_results(results, "pii_detection", results_dir, num_texts)


@cli.command()
@click.option("--image-size", default="512x512", help="Image size for inpainting")
@click.option("--num-regions", default=5, help="Number of regions to inpaint")
@click.option("--iterations", default=2, help="Number of benchmark iterations")
@click.pass_context
def inpainting(ctx, image_size: str, num_regions: int, iterations: int):
    """Benchmark diffusion model inpainting performance."""
    results_dir = ctx.obj["results_dir"]

    # Parse image size
    try:
        width, height = map(int, image_size.split("x"))
        size = (width, height)
    except ValueError:
        click.echo("❌ Invalid image size format. Use WIDTHxHEIGHT (e.g., 512x512)", err=True)
        return

    click.echo("🔄 Benchmarking diffusion inpainting...")
    click.echo(f"   📐 Image size: {size}")
    click.echo(f"   🎯 Regions per image: {num_regions}")
    click.echo(f"   🔄 Iterations: {iterations}")

    benchmark = AnonymizationBenchmark()
    results = []

    for i in range(iterations):
        click.echo(f"\n⏳ Running iteration {i+1}/{iterations}...")

        result = benchmark.benchmark_inpainting(
            image_size=size,
            num_regions=num_regions,
            num_iterations=1,  # One image per benchmark iteration
        )

        results.append(result)

        if result.success:
            click.echo(
                f"✅ Iteration {i+1}: {result.duration_ms:.1f}ms, "
                f"peak memory: {result.peak_memory_mb:.1f}MB"
            )
        else:
            click.echo(f"❌ Iteration {i+1} failed: {result.error_message}")

    # Save and display results
    _display_and_save_results(results, "inpainting", results_dir, 1)


@cli.command()
@click.option("--image-size", default="1024x768", help="Document image size")
@click.option("--num-documents", default=3, help="Number of documents to process")
@click.option("--iterations", default=2, help="Number of benchmark iterations")
@click.pass_context
def end_to_end(ctx, image_size: str, num_documents: int, iterations: int):
    """Benchmark complete end-to-end anonymization pipeline."""
    results_dir = ctx.obj["results_dir"]

    # Parse image size
    try:
        width, height = map(int, image_size.split("x"))
        size = (width, height)
    except ValueError:
        click.echo("❌ Invalid image size format. Use WIDTHxHEIGHT (e.g., 1024x768)", err=True)
        return

    click.echo("🔄 Benchmarking end-to-end anonymization...")
    click.echo(f"   📐 Image size: {size}")
    click.echo(f"   📄 Documents: {num_documents}")
    click.echo(f"   🔄 Iterations: {iterations}")

    benchmark = AnonymizationBenchmark()
    results = []

    for i in range(iterations):
        click.echo(f"\n⏳ Running iteration {i+1}/{iterations}...")

        result = benchmark.benchmark_end_to_end(image_size=size, num_documents=num_documents)

        results.append(result)

        if result.success:
            click.echo(
                f"✅ Iteration {i+1}: {result.duration_ms:.1f}ms, "
                f"peak memory: {result.peak_memory_mb:.1f}MB"
            )

            # Show pipeline breakdown if available
            if result.additional_metrics and "pipeline_stages" in result.additional_metrics:
                stages = result.additional_metrics["pipeline_stages"]
                click.echo(f"   Pipeline stages: {' → '.join(stages)}")
        else:
            click.echo(f"❌ Iteration {i+1} failed: {result.error_message}")

    # Save and display results
    _display_and_save_results(results, "end_to_end", results_dir, num_documents)


@cli.command()
@click.option("--quick", is_flag=True, help="Run quick benchmark (fewer iterations)")
@click.pass_context
def full_suite(ctx, quick: bool):
    """Run the complete benchmark suite."""
    results_dir = ctx.obj["results_dir"]

    if quick:
        click.echo("🚀 Running quick benchmark suite...")
    else:
        click.echo("🔄 Running full benchmark suite...")

    benchmark = AnonymizationBenchmark()

    click.echo("\n📋 Running benchmark suite:")
    click.echo("=" * 50)

    # Run all benchmarks
    all_results = benchmark.run_full_benchmark_suite()

    # Display summary
    click.echo("\n📊 Benchmark Suite Results:")
    click.echo("=" * 50)

    total_duration = sum(r.duration_ms for r in all_results if r.success)
    peak_memory = max(r.peak_memory_mb for r in all_results if r.success)
    success_count = sum(1 for r in all_results if r.success)

    for result in all_results:
        status = "✅" if result.success else "❌"
        click.echo(
            f"{status} {result.benchmark_name}: {result.duration_ms:.1f}ms, "
            f"{result.peak_memory_mb:.1f}MB"
        )

    click.echo("\n🎯 Summary:")
    click.echo(f"   Total duration: {total_duration:.1f}ms")
    click.echo(f"   Peak memory: {peak_memory:.1f}MB")
    click.echo(f"   Success rate: {success_count}/{len(all_results)}")

    # Save results
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"full_suite_{timestamp}.json"

    benchmark.save_benchmark_results(all_results, results_file)
    click.echo(f"\n💾 Results saved to: {results_file}")


@cli.command()
@click.option("--duration", default=30, help="Monitoring duration in seconds")
@click.option("--interval", default=1.0, help="Sampling interval in seconds")
@click.pass_context
def monitor(_ctx, duration: int, interval: float):
    """Monitor real-time resource usage."""
    click.echo("📊 Starting resource monitoring...")
    click.echo(f"   Duration: {duration} seconds")
    click.echo(f"   Interval: {interval} seconds")

    monitor = PerformanceMonitor(auto_export=False)

    try:
        monitor.start_session("manual_monitoring")

        # Monitor for specified duration
        start_time = time.time()
        while time.time() - start_time < duration:
            current = monitor.get_current_usage()
            if current:
                click.echo(
                    f"\r⏱️  CPU: {current.cpu_percent:5.1f}% | "
                    f"Memory: {current.memory_rss_mb:6.1f}MB | "
                    f"GPU: {current.gpu_memory_mb or 0:6.1f}MB",
                    nl=False,
                )

            time.sleep(interval)

        click.echo()  # New line

        # End session and show report
        session_report = monitor.end_session()

        click.echo("\n📋 Monitoring Summary:")
        click.echo("=" * 30)

        resource_summary = session_report["resource_summary"]
        click.echo(f"Duration: {resource_summary['duration_seconds']:.1f}s")
        click.echo(f"Peak Memory: {resource_summary['peak_memory_mb']:.1f}MB")
        click.echo(f"Avg CPU: {resource_summary['cpu_percent']['avg']:.1f}%")
        click.echo(f"Samples: {resource_summary['sample_count']}")

        if resource_summary.get("gpu_peak_memory_mb"):
            click.echo(f"Peak GPU Memory: {resource_summary['gpu_peak_memory_mb']:.1f}MB")

    except KeyboardInterrupt:
        click.echo("\n\n⏹️  Monitoring stopped by user")
        monitor.end_session()


@cli.command()
@click.argument("results_file", type=click.Path(exists=True, path_type=Path))
def analyze(results_file: Path):
    """Analyze benchmark results from a JSON file."""
    try:
        with Path(results_file).open() as f:
            data = json.load(f)

        if "results" not in data:
            click.echo("❌ Invalid results file format", err=True)
            return

        results = data["results"]

        click.echo(f"📊 Analyzing results from: {results_file}")
        click.echo("=" * 50)

        # Group by benchmark type
        by_benchmark = {}
        for result in results:
            benchmark_name = result["benchmark_name"]
            if benchmark_name not in by_benchmark:
                by_benchmark[benchmark_name] = []
            by_benchmark[benchmark_name].append(result)

        # Analyze each benchmark type
        for benchmark_name, benchmark_results in by_benchmark.items():
            successful = [r for r in benchmark_results if r["success"]]

            if not successful:
                click.echo(f"❌ {benchmark_name}: No successful runs")
                continue

            durations = [r["duration_ms"] for r in successful]
            memories = [r["peak_memory_mb"] for r in successful]

            click.echo(f"\n📈 {benchmark_name} ({len(successful)} runs):")
            click.echo(
                f"   Duration: {min(durations):.1f}ms - {max(durations):.1f}ms "
                f"(avg: {sum(durations)/len(durations):.1f}ms)"
            )
            click.echo(
                f"   Memory: {min(memories):.1f}MB - {max(memories):.1f}MB "
                f"(avg: {sum(memories)/len(memories):.1f}MB)"
            )

            # Show additional metrics if available
            if successful[0].get("additional_metrics"):
                metrics = successful[0]["additional_metrics"]
                if "throughput_fps" in metrics:
                    click.echo(f"   Throughput: {metrics['throughput_fps']:.2f} items/second")

    except Exception as e:
        click.echo(f"❌ Error analyzing results: {e}", err=True)


def _display_and_save_results(results, benchmark_type: str, results_dir: Path, item_count: int):
    """Helper to display and save benchmark results."""
    successful_results = [r for r in results if r.success]

    if successful_results:
        avg_duration = sum(r.duration_ms for r in successful_results) / len(successful_results)
        avg_memory = sum(r.peak_memory_mb for r in successful_results) / len(successful_results)

        click.echo("\n📊 Results Summary:")
        click.echo(f"   Average duration: {avg_duration:.1f}ms")
        click.echo(f"   Average peak memory: {avg_memory:.1f}MB")
        click.echo(f"   Success rate: {len(successful_results)}/{len(results)}")

        # Calculate throughput
        if avg_duration > 0:
            items_per_second = (item_count * 1000) / avg_duration
            click.echo(f"   Throughput: {items_per_second:.2f} items/second")

    # Save results
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"{benchmark_type}_{timestamp}.json"

    benchmark = AnonymizationBenchmark()
    benchmark.save_benchmark_results(results, results_file)
    click.echo(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    cli()
