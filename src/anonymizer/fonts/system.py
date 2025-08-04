"""
System Font Provider
===================

Provider for system fonts available on the local machine.
Handles detection and loading of fonts from standard system locations.
"""

import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path

from .manager import FontMetadata
from .utils import get_font_info

# Optional Windows dependency
try:
    import winreg
except ImportError:
    winreg = None

logger = logging.getLogger(__name__)


class SystemFontProvider:
    """
    Provider for system fonts available on the local machine.

    Detects and loads fonts from standard system font directories
    across different operating systems.
    """

    def __init__(self):
        """Initialize system font provider."""
        self.system = platform.system().lower()
        self.font_directories = self._get_system_font_directories()
        self.fonts_cache: dict[str, FontMetadata] = {}

        logger.debug(f"SystemFontProvider initialized for {self.system}")
        logger.debug(f"Font directories: {self.font_directories}")

    def _get_system_font_directories(self) -> list[Path]:
        """Get system font directories based on operating system."""
        directories = []

        if self.system == "windows":
            # Windows font directories
            directories.extend(
                [
                    Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts",
                    Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts",
                ]
            )

        elif self.system == "darwin":  # macOS
            # macOS font directories
            directories.extend(
                [
                    Path("/System/Library/Fonts"),
                    Path("/Library/Fonts"),
                    Path.home() / "Library" / "Fonts",
                    Path("/System/Library/Assets/com_apple_MobileAsset_Font6"),
                ]
            )

        else:  # Linux and other Unix-like systems
            # Linux font directories
            directories.extend(
                [
                    Path("/usr/share/fonts"),
                    Path("/usr/local/share/fonts"),
                    Path.home() / ".fonts",
                    Path.home() / ".local" / "share" / "fonts",
                    Path("/usr/share/fonts/truetype"),
                    Path("/usr/share/fonts/opentype"),
                    Path("/usr/share/fonts/TTF"),
                    Path("/usr/share/fonts/OTF"),
                ]
            )

        # Filter to existing directories
        return [d for d in directories if d.exists() and d.is_dir()]

    def list_fonts(self) -> list[FontMetadata]:
        """List all system fonts."""
        fonts = []

        for font_dir in self.font_directories:
            try:
                fonts.extend(self._scan_font_directory(font_dir))
            except Exception as e:
                logger.warning(f"Failed to scan font directory {font_dir}: {e}")

        return fonts

    def _scan_font_directory(self, font_dir: Path) -> list[FontMetadata]:
        """Scan a font directory for font files."""
        fonts = []

        # Supported font extensions
        font_extensions = {".ttf", ".otf", ".woff", ".woff2", ".ttc", ".otc"}

        try:
            for font_file in font_dir.rglob("*"):
                if font_file.is_file() and font_file.suffix.lower() in font_extensions:
                    try:
                        metadata = self._create_font_metadata(str(font_file))
                        if metadata:
                            fonts.append(metadata)
                    except Exception as e:
                        logger.debug(f"Failed to process font {font_file}: {e}")

        except PermissionError:
            logger.debug(f"Permission denied accessing {font_dir}")
        except Exception as e:
            logger.warning(f"Error scanning {font_dir}: {e}")

        return fonts

    def _create_font_metadata(self, font_path: str) -> FontMetadata | None:
        """Create font metadata from system font file."""
        try:

            # Get font information
            font_info = get_font_info(font_path)
            if not font_info:
                return None

            # Calculate checksum (for system fonts, we'll use a lightweight approach)
            checksum = self._calculate_lightweight_checksum(font_path)

            # Get file size
            size_bytes = Path(font_path).stat().st_size

            return FontMetadata(
                name=font_info["name"],
                family=font_info["family"],
                style=font_info["style"],
                weight=font_info["weight"],
                path=font_path,
                size_bytes=size_bytes,
                checksum=checksum,
                is_bundled=False,
                license_info=None,  # System fonts don't include license info
            )

        except Exception as e:
            logger.debug(f"Failed to create metadata for {font_path}: {e}")
            return None

    def _calculate_lightweight_checksum(self, font_path: str) -> str:
        """Calculate a lightweight checksum for system fonts."""
        try:
            # For system fonts, use file size + modification time as a lightweight checksum
            stat = Path(font_path).stat()
            checksum_data = f"{stat.st_size}_{stat.st_mtime}_{Path(font_path).name}"
            return hashlib.sha256(checksum_data.encode()).hexdigest()[
                :16
            ]  # Truncate for lightweight usage
        except Exception:
            # Fallback to filename hash
            return hashlib.sha256(Path(font_path).name.encode()).hexdigest()[
                :16
            ]  # Truncate for lightweight usage

    def find_font(self, font_name: str, style: str = "normal") -> FontMetadata | None:
        """
        Find a specific system font.

        Args:
            font_name: Font family name
            style: Font style

        Returns:
            FontMetadata if found, None otherwise
        """
        # Use system-specific font finding methods
        if self.system == "windows":
            return self._find_windows_font(font_name, style)
        if self.system == "darwin":
            return self._find_macos_font(font_name, style)
        return self._find_linux_font(font_name, style)

    def _find_windows_font(self, font_name: str, _style: str) -> FontMetadata | None:
        """Find font on Windows using registry."""
        try:
            if winreg is None:
                return None

            # Open the font registry key
            font_key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
            )

            # Search for font
            index = 0
            while True:
                try:
                    value_name, value_data, _ = winreg.EnumValue(font_key, index)

                    # Check if this matches our font
                    if font_name.lower() in value_name.lower():
                        font_path = value_data
                        if not Path(font_path).is_absolute():
                            # Relative path, make absolute
                            font_path = str(
                                Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts" / font_path
                            )

                        if Path(font_path).exists():
                            return self._create_font_metadata(font_path)

                    index += 1

                except OSError:
                    break

            winreg.CloseKey(font_key)

        except Exception as e:
            logger.debug(f"Failed to search Windows registry for font {font_name}: {e}")

        return None

    def _find_macos_font(self, font_name: str, _style: str) -> FontMetadata | None:
        """Find font on macOS using system font cache."""
        try:
            # Try using system font database
            # Use system_profiler to get font information
            system_profiler_path = shutil.which("system_profiler")
            if not system_profiler_path:
                logger.warning("system_profiler not found in PATH")
                return None

            result = subprocess.run(
                [system_profiler_path, "SPFontsDataType", "-json"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            if result.returncode == 0:
                json.loads(result.stdout)
                # Parse font data and find matching font
                # This is a simplified implementation

        except Exception as e:
            logger.debug(f"Failed to search macOS fonts for {font_name}: {e}")

        return None

    def _find_linux_font(self, font_name: str, style: str) -> FontMetadata | None:
        """Find font on Linux using fontconfig."""
        try:
            # Use fc-match to find font
            fc_match_path = shutil.which("fc-match")
            if not fc_match_path:
                logger.warning("fc-match not found in PATH")
                return None

            result = subprocess.run(
                [fc_match_path, f"{font_name}:style={style}", "--format=%{file}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            if result.returncode == 0:
                font_path = result.stdout.strip()
                if Path(font_path).exists():
                    return self._create_font_metadata(font_path)

        except Exception as e:
            logger.debug(f"Failed to search Linux fonts for {font_name}: {e}")

        return None

    def get_system_font_info(self) -> dict[str, any]:
        """Get system font information."""
        info = {
            "system": self.system,
            "font_directories": [str(d) for d in self.font_directories],
            "total_directories": len(self.font_directories),
        }

        # Count fonts in each directory
        directory_counts = {}
        for font_dir in self.font_directories:
            try:
                count = len(list(font_dir.rglob("*.ttf"))) + len(list(font_dir.rglob("*.otf")))
                directory_counts[str(font_dir)] = count
            except Exception:
                directory_counts[str(font_dir)] = 0

        info["directory_font_counts"] = directory_counts
        info["total_system_fonts"] = sum(directory_counts.values())

        return info

    def refresh_font_cache(self) -> None:
        """Refresh system font cache."""
        try:
            if self.system == "linux":
                # Refresh fontconfig cache
                fc_cache_path = shutil.which("fc-cache")
                if fc_cache_path:
                    subprocess.run([fc_cache_path, "-f"], timeout=30, check=False)
                else:
                    logger.warning("fc-cache not found in PATH")
                logger.info("Refreshed fontconfig cache")

            elif self.system == "darwin":
                # Clear macOS font cache
                atsutil_path = shutil.which("atsutil")
                if atsutil_path:
                    subprocess.run([atsutil_path, "databases", "-remove"], timeout=30, check=False)
                else:
                    logger.warning("atsutil not found in PATH")
                logger.info("Cleared macOS font cache")

            elif self.system == "windows":
                # Windows font cache is managed automatically
                logger.info("Windows font cache is managed automatically")

        except Exception as e:
            logger.warning(f"Failed to refresh font cache: {e}")

    def install_system_font(self, font_path: str) -> bool:
        """
        Install font to system font directory.

        Args:
            font_path: Path to font file to install

        Returns:
            True if successful, False otherwise
        """
        try:
            source_path = Path(font_path)
            if not source_path.exists():
                logger.error(f"Font file not found: {font_path}")
                return False

            # Determine target directory
            if self.system == "windows":
                target_dir = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
            elif self.system == "darwin":
                target_dir = Path.home() / "Library" / "Fonts"
            else:  # Linux
                target_dir = Path.home() / ".local" / "share" / "fonts"

            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / source_path.name

            # Copy font file
            shutil.copy2(source_path, target_path)

            # Refresh font cache
            self.refresh_font_cache()

            logger.info(f"Installed system font: {target_path}")
            return True

        except Exception:
            logger.exception(f"Failed to install system font {font_path}")
            return False
