"""PixelGuard: find and remove patient identifiers in DICOM metadata and pixels.

PixelGuard reduces re-identification risk; it does not guarantee
de-identification. Every output must be reviewed against your own
policy before release.
"""

__version__ = "0.1.0"

DISCLAIMER = (
    "PixelGuard reduces, but does not eliminate, re-identification risk. "
    "Automated detection can miss identifiers (unusual fonts, low contrast, "
    "handwriting, identifiers in anatomy such as faces or jewelry, unknown "
    "private tags). Results are not a guarantee of de-identification and must "
    "be reviewed by a qualified person before data is shared."
)
