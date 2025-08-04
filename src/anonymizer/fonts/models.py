"""
Font data models and types.
"""

from dataclasses import dataclass
from pathlib import Path


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
        return Path(self.path).name

    @property
    def extension(self) -> str:
        """Get the font file extension."""
        return Path(self.path).suffix.lower()

    def __str__(self) -> str:
        return f"{self.family} {self.style} ({self.name})"
