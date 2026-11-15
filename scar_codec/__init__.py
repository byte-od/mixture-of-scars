"""`scar_codec`: Domain-Independent Physical & Hardware Signal/Storage Scar Operators (`v5.0.0`)."""

from scar_codec.core import (
    CircularRingBufferOT,
    OrthogonalSubspaceProjector,
    PhysicalVQ4ScarCodec,
    PhysicalVQ4ScarPackage,
    SurpriseGatedAnomalySCache,
    __version__,
    unique_tensor_storage_bytes,
)

__all__ = [
    "__version__",
    "CircularRingBufferOT",
    "OrthogonalSubspaceProjector",
    "PhysicalVQ4ScarCodec",
    "PhysicalVQ4ScarPackage",
    "SurpriseGatedAnomalySCache",
    "unique_tensor_storage_bytes",
]
