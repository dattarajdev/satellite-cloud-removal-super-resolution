"""
Super-resolution processing package using OpenSR diffusion model.
"""

from .band_selector import (
    OPENSR_REQUIRED_BANDS,
    OPENSR_FULL_NAMES,
    normalize_band_name,
    validate_and_get_band_indices,
)
from .opensr_service import (
    OpenSRConfig,
    OpenSRDiagnostics,
    OpenSRResult,
    OpenSRService,
)

__all__ = [
    "OPENSR_REQUIRED_BANDS",
    "OPENSR_FULL_NAMES",
    "normalize_band_name",
    "validate_and_get_band_indices",
    "OpenSRConfig",
    "OpenSRDiagnostics",
    "OpenSRResult",
    "OpenSRService",
]
