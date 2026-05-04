"""
PulseM package initialization.

This namespace exposes high-level helpers used across the CLI scripts.
"""

from .data_loader import ECGDataLoader  # noqa: F401
from .model import ECGArtifactDetector, create_lightweight_model  # noqa: F401









