"""
Satellite image processing and harmonization services.
Phase 2: UTM reprojection, common grid generation, band ordering, and co-registration.
Phase 3 Baseline: Canonical Inference Service for ECRformer Cloud Removal.
"""

from .grid import determine_utm_crs, create_common_grid, TargetGrid
from .harmonizer import (
    HarmonizationConfig,
    HarmonizedPair,
    harmonize_pair,
    validate_harmonized_pair,
    S2_BAND_ORDER,
    S1_BAND_ORDER,
)
from .inference_service import (
    InferenceService,
    InferenceConfig,
    InferenceResult,
    InferenceDiagnostics,
    get_device,
    load_ecrformer_model,
)
from .super_resolution import (
    OpenSRService,
    OpenSRConfig,
    OpenSRResult,
    OpenSRDiagnostics,
    validate_and_get_band_indices,
)

__all__ = [
    "determine_utm_crs",
    "create_common_grid",
    "TargetGrid",
    "HarmonizationConfig",
    "HarmonizedPair",
    "harmonize_pair",
    "validate_harmonized_pair",
    "S2_BAND_ORDER",
    "S1_BAND_ORDER",
    "InferenceService",
    "InferenceConfig",
    "InferenceResult",
    "InferenceDiagnostics",
    "get_device",
    "load_ecrformer_model",
    "OpenSRService",
    "OpenSRConfig",
    "OpenSRResult",
    "OpenSRDiagnostics",
    "validate_and_get_band_indices",
]
