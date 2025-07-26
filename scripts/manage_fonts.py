#!/usr/bin/env python3
"""
Font Management Script
======================

Script for managing bundled fonts, downloading required fonts,
and managing font installations.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from anonymizer.fonts import FontManager, BundledFontProvider


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def download_bundled_fonts(args) -> None:
    """Download bundled fonts."""
    print("📦 Downloading bundled fonts...")

    fonts_dir = Path(args.fonts_dir) if args.fonts_dir else None
    provider = BundledFontProvider(fonts_dir)

    # Force re-download if requested
    if args.force:
        print("🔄 Force downloading fonts...")
        # Remove existing fonts
        if provider.bundled_fonts_dir.exists():
            import shutil

            shutil.rmtree(provider.bundled_fonts_dir)
        provider.bundled_fonts_dir.mkdir(parents=True, exist_ok=True)
        provider._ensure_bundled_fonts()

    # List downloaded fonts
    fonts = provider.list_fonts()
    print(f"✅ Downloaded {len(fonts)} bundled fonts:")
    for font in fonts:
        print(f"   📄 {font.family} {font.style} ({font.size_bytes // 1024}KB)")


def list_fonts(args) -> None:
    """List available fonts."""
    fonts_dir = Path(args.fonts_dir) if args.fonts_dir else None
    font_manager = FontManager(fonts_dir)

    if args.system_only:
        fonts = font_manager.system_provider.list_fonts()
        print("🖥️ System Fonts:")
    elif args.bundled_only:
        fonts = font_manager.bundled_provider.list_fonts()
        print("📦 Bundled Fonts:")
    else:
        fonts = font_manager.list_available_fonts()
        print("🔤 All Available Fonts:")

    if not fonts:
        print("   No fonts found.")
        return

    # Group by family
    families = {}
    for font in fonts:
        if font.family not in families:
            families[font.family] = []
        families[font.family].append(font)

    for family_name, family_fonts in sorted(families.items()):
        print(f"\n📁 {family_name}:")
        for font in sorted(family_fonts, key=lambda f: f.style):
            status = "📦" if font.is_bundled else "🖥️"
            print(
                f"   {status} {font.style} - {font.filename} ({font.size_bytes // 1024}KB)"
            )


def list_families(args) -> None:
    """List font families."""
    fonts_dir = Path(args.fonts_dir) if args.fonts_dir else None
    font_manager = FontManager(fonts_dir)

    families = font_manager.list_font_families()
    print(f"📚 Found {len(families)} font families:")

    for family in families:
        # Count fonts in family
        family_fonts = [
            f for f in font_manager.list_available_fonts() if f.family == family
        ]
        bundled_count = len([f for f in family_fonts if f.is_bundled])
        system_count = len([f for f in family_fonts if not f.is_bundled])

        status_parts = []
        if bundled_count > 0:
            status_parts.append(f"📦{bundled_count}")
        if system_count > 0:
            status_parts.append(f"🖥️{system_count}")

        status = " ".join(status_parts)
        print(f"   {family} ({status})")


def font_info(args) -> None:
    """Show detailed information about a font."""
    fonts_dir = Path(args.fonts_dir) if args.fonts_dir else None
    font_manager = FontManager(fonts_dir)

    font = font_manager.get_font(args.font_name, args.style)

    if not font:
        print(f"❌ Font not found: {args.font_name} {args.style}")

        # Suggest similar fonts
        similar_fonts = font_manager.find_similar_fonts(args.font_name, max_results=3)
        if similar_fonts:
            print("\n💡 Similar fonts found:")
            for similar_font in similar_fonts:
                print(f"   {similar_font.family} {similar_font.style}")
        return

    print(f"📄 Font Information: {font.name}")
    print(f"   Family: {font.family}")
    print(f"   Style: {font.style}")
    print(f"   Weight: {font.weight}")
    print(f"   Path: {font.path}")
    print(f"   Size: {font.size_bytes // 1024}KB")
    print(f"   Type: {'Bundled' if font.is_bundled else 'System'}")
    print(f"   Checksum: {font.checksum[:16]}...")

    if font.license_info:
        print(f"   License: {font.license_info}")

    # Get font metrics
    from anonymizer.fonts.utils import get_font_metrics

    metrics = get_font_metrics(font.path, 12)
    if metrics:
        print("\n📐 Font Metrics (12pt):")
        print(f"   Height: {metrics['height']:.1f}pt")
        print(f"   Ascent: {metrics['ascent']:.1f}pt")
        print(f"   Descent: {metrics['descent']:.1f}pt")
        print(f"   Max Width: {metrics['max_width']:.1f}pt")


def install_font(args) -> None:
    """Install a font file."""
    fonts_dir = Path(args.fonts_dir) if args.fonts_dir else None
    font_manager = FontManager(fonts_dir)

    font_path = Path(args.font_path)
    if not font_path.exists():
        print(f"❌ Font file not found: {font_path}")
        return

    print(f"📥 Installing font: {font_path.name}")

    if args.system:
        # Install to system
        success = font_manager.system_provider.install_system_font(str(font_path))
    else:
        # Install to bundled fonts
        success = font_manager.install_font(str(font_path))

    if success:
        print("✅ Font installed successfully")
    else:
        print("❌ Font installation failed")


def create_sample(args) -> None:
    """Create a font sample image."""
    fonts_dir = Path(args.fonts_dir) if args.fonts_dir else None
    font_manager = FontManager(fonts_dir)

    font = font_manager.get_font(args.font_name, args.style)
    if not font:
        print(f"❌ Font not found: {args.font_name} {args.style}")
        return

    print(f"🎨 Creating font sample for {font.name}...")

    from anonymizer.fonts.utils import create_font_sample

    sample_text = args.text or "The quick brown fox jumps over the lazy dog."
    size = args.size or 24

    sample_image = create_font_sample(font.path, sample_text, size)

    if sample_image:
        output_path = args.output or f"{font.family}_{font.style}_sample.png"
        sample_image.save(output_path)
        print(f"✅ Font sample saved: {output_path}")
    else:
        print("❌ Failed to create font sample")


def font_stats(args) -> None:
    """Show font statistics."""
    fonts_dir = Path(args.fonts_dir) if args.fonts_dir else None
    font_manager = FontManager(fonts_dir)

    stats = font_manager.get_font_statistics()

    print("📊 Font Statistics:")
    print(f"   Total fonts: {stats['total_fonts']}")
    print(f"   Bundled fonts: {stats['bundled_fonts']}")
    print(f"   System fonts: {stats['system_fonts']}")
    print(f"   Font families: {stats['font_families']}")

    print("\n📝 Font styles:")
    style_stats = {
        k: v
        for k, v in stats.items()
        if k not in ["total_fonts", "bundled_fonts", "system_fonts", "font_families"]
    }
    for style, count in sorted(style_stats.items()):
        print(f"   {style}: {count}")


def cleanup_fonts(args) -> None:
    """Clean up font cache."""
    fonts_dir = Path(args.fonts_dir) if args.fonts_dir else None
    font_manager = FontManager(fonts_dir)

    print("🧹 Cleaning up fonts...")

    cleanup_stats = font_manager.cleanup_fonts(dry_run=args.dry_run)

    print("📊 Cleanup Results:")
    print(f"   Fonts checked: {cleanup_stats['fonts_checked']}")
    print(f"   Duplicates found: {cleanup_stats['duplicates_found']}")
    print(f"   Missing files: {cleanup_stats['missing_files']}")

    if args.dry_run:
        print(f"   Would remove: {cleanup_stats['fonts_removed']} fonts")
        print("   Run without --dry-run to actually remove fonts")
    else:
        print(f"   Fonts removed: {cleanup_stats['fonts_removed']}")


def export_font_list(args) -> None:
    """Export font list to file."""
    fonts_dir = Path(args.fonts_dir) if args.fonts_dir else None
    font_manager = FontManager(fonts_dir)

    output_path = args.output or "font_list.json"

    print(f"📤 Exporting font list to {output_path}...")

    success = font_manager.export_font_list(output_path)

    if success:
        print("✅ Font list exported successfully")
    else:
        print("❌ Failed to export font list")


def create_font_bundle(args) -> None:
    """Create a distributable font bundle."""
    fonts_dir = Path(args.fonts_dir) if args.fonts_dir else None
    provider = BundledFontProvider(fonts_dir)

    output_path = args.output or "bundled_fonts.zip"

    print(f"📦 Creating font bundle: {output_path}")

    success = provider.create_font_bundle(output_path)

    if success:
        print("✅ Font bundle created successfully")
        print("   Bundle contains bundled fonts with license information")
    else:
        print("❌ Failed to create font bundle")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Font Management Script for Document Anonymization System"
    )
    parser.add_argument("--fonts-dir", help="Custom fonts directory")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download bundled fonts
    download_parser = subparsers.add_parser("download", help="Download bundled fonts")
    download_parser.add_argument(
        "--force", action="store_true", help="Force re-download"
    )

    # List fonts
    list_parser = subparsers.add_parser("list", help="List available fonts")
    list_parser.add_argument(
        "--system-only", action="store_true", help="List only system fonts"
    )
    list_parser.add_argument(
        "--bundled-only", action="store_true", help="List only bundled fonts"
    )

    # List font families
    families_parser = subparsers.add_parser("families", help="List font families")

    # Font information
    info_parser = subparsers.add_parser("info", help="Show font information")
    info_parser.add_argument("font_name", help="Font name to query")
    info_parser.add_argument("--style", default="normal", help="Font style")

    # Install font
    install_parser = subparsers.add_parser("install", help="Install font file")
    install_parser.add_argument("font_path", help="Path to font file")
    install_parser.add_argument(
        "--system", action="store_true", help="Install to system fonts"
    )

    # Create sample
    sample_parser = subparsers.add_parser("sample", help="Create font sample")
    sample_parser.add_argument("font_name", help="Font name")
    sample_parser.add_argument("--style", default="normal", help="Font style")
    sample_parser.add_argument("--text", help="Sample text")
    sample_parser.add_argument("--size", type=int, help="Font size")
    sample_parser.add_argument("--output", help="Output file path")

    # Font statistics
    stats_parser = subparsers.add_parser("stats", help="Show font statistics")

    # Cleanup fonts
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up font cache")
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned without removing",
    )

    # Export font list
    export_parser = subparsers.add_parser("export", help="Export font list")
    export_parser.add_argument("--output", help="Output file path")

    # Create font bundle
    bundle_parser = subparsers.add_parser("bundle", help="Create font bundle")
    bundle_parser.add_argument("--output", help="Output bundle path")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    setup_logging(args.log_level)

    try:
        if args.command == "download":
            download_bundled_fonts(args)
        elif args.command == "list":
            list_fonts(args)
        elif args.command == "families":
            list_families(args)
        elif args.command == "info":
            font_info(args)
        elif args.command == "install":
            install_font(args)
        elif args.command == "sample":
            create_sample(args)
        elif args.command == "stats":
            font_stats(args)
        elif args.command == "cleanup":
            cleanup_fonts(args)
        elif args.command == "export":
            export_font_list(args)
        elif args.command == "bundle":
            create_font_bundle(args)
        else:
            print(f"❌ Unknown command: {args.command}")
            parser.print_help()

    except KeyboardInterrupt:
        print("\n❌ Operation cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.log_level == "DEBUG":
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
