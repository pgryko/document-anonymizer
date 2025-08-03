"""
Font data models and types.
"""

import os
from dataclasses import dataclass


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
