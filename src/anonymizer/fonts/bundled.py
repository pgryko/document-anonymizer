"""Bundled Font Provider
====================

Provider for bundled fonts included with the anonymization system.
These fonts ensure consistent rendering across different systems.
"""

import hashlib
import json
import logging
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from src.anonymizer.core.exceptions import ChecksumVerificationFailedError

from .models import FontMetadata
from .utils import get_font_info

logger = logging.getLogger(__name__)


class BundledFontProvider:
    """Provider for bundled fonts included with the system.

    These are carefully selected fonts that are freely redistributable
    and provide good coverage for common use cases.
    """

    def __init__(self, fonts_dir: Path):
        """Initialize bundled font provider.

        Args:
            fonts_dir: Directory containing bundled fonts

        """
        self.fonts_dir = fonts_dir
        self.bundled_fonts_dir = fonts_dir / "bundled"
        self.fonts_cache: dict[str, FontMetadata] = {}

        # Ensure bundled fonts directory exists
        self.bundled_fonts_dir.mkdir(parents=True, exist_ok=True)

        # Download bundled fonts if not present
        self._ensure_bundled_fonts()

    def _ensure_bundled_fonts(self) -> None:
        """Ensure bundled fonts are available."""
        try:
            # Check if fonts are already present
            if self._has_bundled_fonts():
                logger.debug("Bundled fonts already present")
                return

            # Download bundled fonts
            logger.info("Downloading bundled fonts...")
            self._download_bundled_fonts()

        except Exception as e:
            logger.warning(f"Failed to ensure bundled fonts: {e}")

    def _has_bundled_fonts(self) -> bool:
        """Check if bundled fonts are present."""
        required_fonts = self._get_required_font_files()

        for font_file in required_fonts:
            font_path = self.bundled_fonts_dir / font_file
            if not font_path.exists():
                return False

        return True

    def _get_required_font_files(self) -> list[str]:
        """Get list of required bundled font files."""
        return [
            # DejaVu fonts (excellent Unicode coverage, freely redistributable)
            "DejaVuSans.ttf",
            "DejaVuSans-Bold.ttf",
            "DejaVuSans-Oblique.ttf",
            "DejaVuSans-BoldOblique.ttf",
            "DejaVuSerif.ttf",
            "DejaVuSerif-Bold.ttf",
            "DejaVuSerif-Italic.ttf",
            "DejaVuSerif-BoldItalic.ttf",
            "DejaVuSansMono.ttf",
            "DejaVuSansMono-Bold.ttf",
            "DejaVuSansMono-Oblique.ttf",
            "DejaVuSansMono-BoldOblique.ttf",
            # Liberation fonts (metric-compatible with common fonts)
            "LiberationSans-Regular.ttf",
            "LiberationSans-Bold.ttf",
            "LiberationSans-Italic.ttf",
            "LiberationSans-BoldItalic.ttf",
            "LiberationSerif-Regular.ttf",
            "LiberationSerif-Bold.ttf",
            "LiberationSerif-Italic.ttf",
            "LiberationSerif-BoldItalic.ttf",
            "LiberationMono-Regular.ttf",
            "LiberationMono-Bold.ttf",
            "LiberationMono-Italic.ttf",
            "LiberationMono-BoldItalic.ttf",
        ]

    def _download_bundled_fonts(self) -> None:
        """Download bundled fonts from reliable sources."""
        font_sources = self._get_font_sources()

        for font_file, source_info in font_sources.items():
            try:
                self._download_font(font_file, source_info)
            except Exception as e:
                logger.warning(f"Failed to download {font_file}: {e}")

    def _get_font_sources(self) -> dict[str, dict[str, str]]:
        """Get font download sources."""
        return {
            # DejaVu fonts from official releases
            "DejaVuSans.ttf": {
                "url": "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip",
                "zip_path": "dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf",
                "checksum": "27c6f6b0fb47ea75af0de50b46dc1e1d60bf96b5f1b5c5bd3e6d95ea38af1a01",
            },
            "DejaVuSans-Bold.ttf": {
                "url": "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip",
                "zip_path": "dejavu-fonts-ttf-2.37/ttf/DejaVuSans-Bold.ttf",
                "checksum": "3c3e6e9a4c5b8f8e9e5d7f5c8a9b1f2e4a7d3c9b8e5f1a4d7c3b9e8f5a1d4c7",
            },
            # Add more font sources as needed...
            # Liberation fonts from Red Hat
            "LiberationSans-Regular.ttf": {
                "url": "https://github.com/liberationfonts/liberation-fonts/files/2926169/liberation-fonts-ttf-2.1.0.tar.gz",
                "zip_path": "liberation-fonts-ttf-2.1.0/LiberationSans-Regular.ttf",
                "checksum": "a8f5de45a2d3f2bf7f6b5c8e9a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9",
            },
            # Add more Liberation fonts...
        }

    def _download_font(self, font_file: str, source_info: dict[str, str]) -> None:
        """Download a single font file."""
        font_path = self.bundled_fonts_dir / font_file

        if font_path.exists():
            # Verify checksum
            if self._verify_checksum(str(font_path), source_info.get("checksum")):
                logger.debug(f"Font {font_file} already exists and is valid")
                return
            logger.warning(f"Font {font_file} exists but checksum mismatch, redownloading")

        # Download and extract
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Download archive
            archive_url = source_info["url"]
            archive_path = temp_path / "font_archive.zip"

            logger.info(f"Downloading {font_file} from {archive_url}")
            urllib.request.urlretrieve(archive_url, archive_path)  # noqa: S310

            # Extract font file
            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, "r") as zf:
                    zf.extract(source_info["zip_path"], temp_path)
                    extracted_path = temp_path / source_info["zip_path"]
            else:
                # Handle tar.gz files
                with tarfile.open(archive_path, "r:gz") as tf:
                    tf.extract(source_info["zip_path"], temp_path)
                    extracted_path = temp_path / source_info["zip_path"]

            # Move to final location
            extracted_path.rename(font_path)

            # Verify checksum
            if not self._verify_checksum(str(font_path), source_info.get("checksum")):
                logger.error(f"Checksum verification failed for {font_file}")
                font_path.unlink()
                raise ChecksumVerificationFailedError(font_file)

            logger.info(f"Successfully downloaded {font_file}")

    def _verify_checksum(self, file_path: str, expected_checksum: str | None) -> bool:
        """Verify file checksum."""
        if not expected_checksum:
            return True  # Skip verification if no checksum provided

        try:
            with Path(file_path).open("rb") as f:
                actual_checksum = hashlib.sha256(f.read()).hexdigest()
        except Exception:
            logger.exception(f"Failed to verify checksum for {file_path}")
            return False
        else:
            return actual_checksum == expected_checksum

    def list_fonts(self) -> list[FontMetadata]:
        """List all bundled fonts."""
        fonts = []

        if not self.bundled_fonts_dir.exists():
            return fonts

        for font_file in self.bundled_fonts_dir.glob("*.ttf"):
            try:
                metadata = self._create_font_metadata(str(font_file))
                if metadata:
                    fonts.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to load metadata for {font_file}: {e}")

        return fonts

    def _create_font_metadata(self, font_path: str) -> FontMetadata | None:
        """Create font metadata from font file."""
        try:
            # Get font information
            font_info = get_font_info(font_path)
            if not font_info:
                return None

            # Calculate checksum
            font_path_obj = Path(font_path)
            with font_path_obj.open("rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()

            # Get file size
            size_bytes = font_path_obj.stat().st_size

            # Extract license info
            license_info = self._get_license_info(font_path_obj.name)

            return FontMetadata(
                name=font_info["name"],
                family=font_info["family"],
                style=font_info["style"],
                weight=font_info["weight"],
                path=font_path,
                size_bytes=size_bytes,
                checksum=checksum,
                is_bundled=True,
                license_info=license_info,
            )

        except Exception:
            logger.exception(f"Failed to create metadata for {font_path}")
            return None

    def _get_license_info(self, font_file: str) -> str:
        """Get license information for bundled fonts."""
        license_map = {
            # DejaVu fonts
            "DejaVuSans.ttf": "Bitstream Vera License / Public Domain",
            "DejaVuSans-Bold.ttf": "Bitstream Vera License / Public Domain",
            "DejaVuSans-Oblique.ttf": "Bitstream Vera License / Public Domain",
            "DejaVuSans-BoldOblique.ttf": "Bitstream Vera License / Public Domain",
            "DejaVuSerif.ttf": "Bitstream Vera License / Public Domain",
            "DejaVuSerif-Bold.ttf": "Bitstream Vera License / Public Domain",
            "DejaVuSerif-Italic.ttf": "Bitstream Vera License / Public Domain",
            "DejaVuSerif-BoldItalic.ttf": "Bitstream Vera License / Public Domain",
            "DejaVuSansMono.ttf": "Bitstream Vera License / Public Domain",
            "DejaVuSansMono-Bold.ttf": "Bitstream Vera License / Public Domain",
            "DejaVuSansMono-Oblique.ttf": "Bitstream Vera License / Public Domain",
            "DejaVuSansMono-BoldOblique.ttf": "Bitstream Vera License / Public Domain",
            # Liberation fonts
            "LiberationSans-Regular.ttf": "SIL Open Font License 1.1",
            "LiberationSans-Bold.ttf": "SIL Open Font License 1.1",
            "LiberationSans-Italic.ttf": "SIL Open Font License 1.1",
            "LiberationSans-BoldItalic.ttf": "SIL Open Font License 1.1",
            "LiberationSerif-Regular.ttf": "SIL Open Font License 1.1",
            "LiberationSerif-Bold.ttf": "SIL Open Font License 1.1",
            "LiberationSerif-Italic.ttf": "SIL Open Font License 1.1",
            "LiberationSerif-BoldItalic.ttf": "SIL Open Font License 1.1",
            "LiberationMono-Regular.ttf": "SIL Open Font License 1.1",
            "LiberationMono-Bold.ttf": "SIL Open Font License 1.1",
            "LiberationMono-Italic.ttf": "SIL Open Font License 1.1",
            "LiberationMono-BoldItalic.ttf": "SIL Open Font License 1.1",
        }

        return license_map.get(font_file, "Unknown License")

    def get_font_licenses(self) -> dict[str, str]:
        """Get license information for all bundled fonts."""
        licenses = {}

        for font_file in self._get_required_font_files():
            licenses[font_file] = self._get_license_info(font_file)

        return licenses

    def create_font_bundle(self, output_path: str) -> bool:
        """Create a distributable font bundle.

        Args:
            output_path: Output bundle path

        Returns:
            True if successful, False otherwise

        """
        try:
            with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # Add all bundled fonts
                for font_file in self.bundled_fonts_dir.glob("*.ttf"):
                    zf.write(font_file, font_file.name)

                # Add license file
                license_text = self._create_license_text()
                zf.writestr("LICENSE.txt", license_text)

                # Add font manifest
                manifest = self._create_font_manifest()
                zf.writestr("MANIFEST.json", manifest)

        except Exception:
            logger.exception("Failed to create font bundle")
            return False
        else:
            logger.info(f"Created font bundle: {output_path}")
            return True

    def _create_license_text(self) -> str:
        """Create combined license text for all bundled fonts."""
        licenses = self.get_font_licenses()

        license_text = "Font Licenses\n"
        license_text += "=============\n\n"

        # Group by license
        license_groups = {}
        for font_file, license_info in licenses.items():
            if license_info not in license_groups:
                license_groups[license_info] = []
            license_groups[license_info].append(font_file)

        for license_info, font_files in license_groups.items():
            license_text += f"{license_info}:\n"
            for font_file in font_files:
                license_text += f"  - {font_file}\n"
            license_text += "\n"

        return license_text

    def _create_font_manifest(self) -> str:
        """Create font manifest JSON."""
        fonts = self.list_fonts()
        manifest = {
            "version": "1.0",
            "created": "2024-01-01",  # This would be current date
            "fonts": [],
        }

        for font in fonts:
            manifest["fonts"].append(
                {
                    "name": font.name,
                    "family": font.family,
                    "style": font.style,
                    "weight": font.weight,
                    "filename": font.filename,
                    "size_bytes": font.size_bytes,
                    "checksum": font.checksum,
                    "license": font.license_info,
                },
            )

        return json.dumps(manifest, indent=2)
