"""
Font Management System
======================

Central font management system for handling font detection, loading, and fallbacks
to ensure consistent text rendering during anonymization.
"""

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class FontMetadata:
    """Font metadata information."""

    name: str
    family: str
    style: str  # normal, bold, italic, bold-italic
    weight: int  # 100-900
    path: str
    size_bytes: int
    checksum: str
    is_bundled: bool = False
    license_info: str | None = None

    @property
    def filename(self) -> str:
        """Get the font filename."""
        return os.path.basename(self.path)

    @property
    def extension(self) -> str:
        """Get the font file extension."""
        return os.path.splitext(self.path)[1].lower()

    def __str__(self) -> str:
        return f"{self.family} {self.style} ({self.name})"


class FontManager:
    """
    Central font management system.

    Handles font detection, loading, caching, and fallback mechanisms
    to ensure consistent text rendering across different systems.
    """

    def __init__(self, fonts_dir: Path | None = None):
        """
        Initialize font manager.

        Args:
            fonts_dir: Optional custom fonts directory
        """
        self.fonts_dir = fonts_dir or self._get_default_fonts_dir()
        self.fonts_cache: dict[str, FontMetadata] = {}
        self.loaded_fonts: dict[str, str] = {}  # name -> path mapping
        self.fallback_map: dict[str, list[str]] = {}

        # Font providers
        from .bundled import BundledFontProvider
        from .system import SystemFontProvider

        self.bundled_provider = BundledFontProvider(self.fonts_dir)
        self.system_provider = SystemFontProvider()

        # Initialize font cache
        self._load_font_cache()

        logger.info(f"FontManager initialized with {len(self.fonts_cache)} fonts")

    def _get_default_fonts_dir(self) -> Path:
        """Get default fonts directory."""
        # Try various locations
        possible_dirs = [
            Path.home() / ".cache" / "anonymizer" / "fonts",
            Path("/usr/share/fonts/anonymizer"),
            Path(__file__).parent / "bundled_fonts",
        ]

        for fonts_dir in possible_dirs:
            if fonts_dir.exists() or fonts_dir == possible_dirs[0]:
                fonts_dir.mkdir(parents=True, exist_ok=True)
                return fonts_dir

        return possible_dirs[0]

    def _load_font_cache(self) -> None:
        """Load fonts from all providers into cache."""
        self.fonts_cache.clear()

        # Load bundled fonts
        try:
            bundled_fonts = self.bundled_provider.list_fonts()
            for font in bundled_fonts:
                self.fonts_cache[font.name] = font
                logger.debug(f"Loaded bundled font: {font.name}")
        except Exception as e:
            logger.warning(f"Failed to load bundled fonts: {e}")

        # Load system fonts
        try:
            system_fonts = self.system_provider.list_fonts()
            for font in system_fonts:
                if font.name not in self.fonts_cache:  # Bundled fonts take priority
                    self.fonts_cache[font.name] = font
                    logger.debug(f"Loaded system font: {font.name}")
        except Exception as e:
            logger.warning(f"Failed to load system fonts: {e}")

        # Build fallback mappings
        self._build_fallback_map()

    def _build_fallback_map(self) -> None:
        """Build fallback font mappings."""
        self.fallback_map = {
            # Serif fonts
            "Times": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
            "Times New Roman": ["Times", "Liberation Serif", "DejaVu Serif"],
            "serif": ["Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            # Sans-serif fonts
            "Arial": ["Helvetica", "Liberation Sans", "DejaVu Sans"],
            "Helvetica": ["Arial", "Liberation Sans", "DejaVu Sans"],
            "sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            # Monospace fonts
            "Courier": ["Courier New", "Liberation Mono", "DejaVu Sans Mono"],
            "Courier New": ["Courier", "Liberation Mono", "DejaVu Sans Mono"],
            "monospace": [
                "Courier New",
                "Courier",
                "Liberation Mono",
                "DejaVu Sans Mono",
            ],
            # Common fonts
            "Calibri": ["Arial", "Helvetica", "Liberation Sans"],
            "Verdana": ["Arial", "Helvetica", "Liberation Sans"],
            "Georgia": ["Times New Roman", "Times", "Liberation Serif"],
        }

    def get_font(self, font_name: str, style: str = "normal") -> FontMetadata | None:
        """
        Get font by name and style.

        Args:
            font_name: Font family name
            style: Font style (normal, bold, italic, bold-italic)

        Returns:
            FontMetadata if found, None otherwise
        """
        # Try exact match first
        full_name = f"{font_name}-{style}" if style != "normal" else font_name

        if full_name in self.fonts_cache:
            return self.fonts_cache[full_name]

        # Try family name match
        for name, font in self.fonts_cache.items():
            if font.family.lower() == font_name.lower() and font.style == style:
                return font

        # Try fallback fonts
        if font_name in self.fallback_map:
            for fallback_name in self.fallback_map[font_name]:
                fallback_font = self.get_font(fallback_name, style)
                if fallback_font:
                    logger.info(f"Using fallback font {fallback_font.name} for {font_name}")
                    return fallback_font

        logger.warning(f"Font not found: {font_name} {style}")
        return None

    def get_font_path(self, font_name: str, style: str = "normal") -> str | None:
        """
        Get font file path.

        Args:
            font_name: Font family name
            style: Font style

        Returns:
            Font file path if found, None otherwise
        """
        font = self.get_font(font_name, style)
        return font.path if font else None

    def list_available_fonts(self) -> list[FontMetadata]:
        """List all available fonts."""
        return list(self.fonts_cache.values())

    def list_font_families(self) -> list[str]:
        """List all available font families."""
        families = set()
        for font in self.fonts_cache.values():
            families.add(font.family)
        return sorted(families)

    def install_font(self, font_path: str, metadata: dict | None = None) -> bool:
        """
        Install a font file.

        Args:
            font_path: Path to font file
            metadata: Optional font metadata

        Returns:
            True if installed successfully, False otherwise
        """
        try:
            source_path = Path(font_path)
            if not source_path.exists():
                logger.error(f"Font file not found: {font_path}")
                return False

            # Copy to fonts directory
            dest_path = self.fonts_dir / source_path.name
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            import shutil

            shutil.copy2(source_path, dest_path)

            # Create metadata
            font_metadata = self._create_font_metadata(str(dest_path), metadata)
            if font_metadata:
                self.fonts_cache[font_metadata.name] = font_metadata
                logger.info(f"Installed font: {font_metadata.name}")
                return True
            logger.error(f"Failed to create metadata for font: {font_path}")
            return False

        except Exception as e:
            logger.error(f"Failed to install font {font_path}: {e}")
            return False

    def _create_font_metadata(
        self, font_path: str, metadata: dict | None = None
    ) -> FontMetadata | None:
        """Create font metadata from font file."""
        try:
            from .utils import get_font_info

            font_info = get_font_info(font_path)
            if not font_info:
                return None

            # Calculate checksum
            with open(font_path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()

            # Get file size
            size_bytes = os.path.getsize(font_path)

            # Create metadata
            return FontMetadata(
                name=font_info.get("name", os.path.basename(font_path)),
                family=font_info.get("family", "Unknown"),
                style=font_info.get("style", "normal"),
                weight=font_info.get("weight", 400),
                path=font_path,
                size_bytes=size_bytes,
                checksum=checksum,
                is_bundled=True,
                license_info=metadata.get("license") if metadata else None,
            )

        except Exception as e:
            logger.error(f"Failed to create metadata for {font_path}: {e}")
            return None

    def find_similar_fonts(self, target_font: str, max_results: int = 5) -> list[FontMetadata]:
        """
        Find fonts similar to target font.

        Args:
            target_font: Target font name
            max_results: Maximum number of results

        Returns:
            List of similar fonts sorted by similarity
        """
        from .utils import calculate_font_similarity

        target_meta = self.get_font(target_font)
        if not target_meta:
            logger.warning(f"Target font not found: {target_font}")
            return []

        # Calculate similarity scores
        similarities = []
        for font in self.fonts_cache.values():
            if font.name != target_font:
                similarity = calculate_font_similarity(target_meta, font)
                similarities.append((similarity, font))

        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[0], reverse=True)
        return [font for _, font in similarities[:max_results]]

    def get_font_statistics(self) -> dict[str, int]:
        """Get font statistics."""
        stats = {
            "total_fonts": len(self.fonts_cache),
            "bundled_fonts": len([f for f in self.fonts_cache.values() if f.is_bundled]),
            "system_fonts": len([f for f in self.fonts_cache.values() if not f.is_bundled]),
            "font_families": len(self.list_font_families()),
        }

        # Count by style
        style_counts = {}
        for font in self.fonts_cache.values():
            style_counts[font.style] = style_counts.get(font.style, 0) + 1

        stats.update(style_counts)
        return stats

    def cleanup_fonts(self, dry_run: bool = True) -> dict[str, int]:
        """
        Clean up unused or duplicate fonts.

        Args:
            dry_run: If True, only report what would be cleaned

        Returns:
            Cleanup statistics
        """
        stats = {
            "fonts_checked": 0,
            "duplicates_found": 0,
            "missing_files": 0,
            "fonts_removed": 0,
        }

        # Find duplicates by checksum
        checksums = {}
        duplicates = []

        for font in self.fonts_cache.values():
            stats["fonts_checked"] += 1

            # Check if file exists
            if not os.path.exists(font.path):
                stats["missing_files"] += 1
                if not dry_run:
                    del self.fonts_cache[font.name]
                continue

            # Check for duplicates
            if font.checksum in checksums:
                duplicates.append(font)
                stats["duplicates_found"] += 1
            else:
                checksums[font.checksum] = font

        # Remove duplicates (keep bundled fonts over system fonts)
        if not dry_run:
            for font in duplicates:
                existing = checksums[font.checksum]
                if font.is_bundled and not existing.is_bundled:
                    # Replace system font with bundled font
                    del self.fonts_cache[existing.name]
                    checksums[font.checksum] = font
                else:
                    # Remove duplicate
                    del self.fonts_cache[font.name]
                    stats["fonts_removed"] += 1

        logger.info(f"Font cleanup: {stats}")
        return stats

    def export_font_list(self, output_path: str) -> bool:
        """
        Export font list to file.

        Args:
            output_path: Output file path

        Returns:
            True if successful, False otherwise
        """
        try:
            import json

            font_data = []
            for font in self.fonts_cache.values():
                font_data.append(
                    {
                        "name": font.name,
                        "family": font.family,
                        "style": font.style,
                        "weight": font.weight,
                        "path": font.path,
                        "size_bytes": font.size_bytes,
                        "is_bundled": font.is_bundled,
                        "checksum": font.checksum,
                    }
                )

            with open(output_path, "w") as f:
                json.dump(font_data, f, indent=2)

            logger.info(f"Exported font list to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export font list: {e}")
            return False
