"""`scar_codec` (`v6.0.0`) — Consolidated Zero-PyTorch + Native C/SIMD Scientific Scar Codec.

Consolidates all 4 industrial pillars:
  1. Zero-PyTorch `NumPy` + Native C99/SIMD `< 1 us` `O(1)` block engine (`numpy_engine.py`, `c_kernels.c`),
     while remaining 100% backward-compatible with PyTorch tensors when `torch` is installed (`core.py`).
  2. Canonical `.scar` on-disk binary container with `CRC32`, zero-copy `ScarMmapFile` (`binary_format.py`),
     and `numcodecs` / `Zarr` ecosystem plugin `ScarZarrCodec` (`zarr_codec.py`).
  3. Provable deterministic `L_inf` error-bounded mode (`abs_tolerance`) via Dekker `FP16 + FP16` splitting.
  4. Native 3D/ND Spectral-Tucker cube codec `ScarCodecND` & `1.000 bpw` Gosset `E_8` `ScarCodecE8` (`nd_e8.py`).
"""

from __future__ import annotations

__version__ = "6.0.0"

from scar_codec.binary_format import (
    SCARFIELD_MAGIC,
    SCAR_MAGIC,
    ScarFieldMmapContainer,
    ScarMmapFile,
    deserialize_scar_buffer,
    load_scar_file,
    save_scar_file,
    save_scarfield_v7_container,
    serialize_scar_bytes,
)
from scar_codec.nd_e8 import (
    ScarCodecE8,
    ScarCodecND,
    ScarE8Package,
    ScarNDPackage,
    build_canonical_e8_lut256_np,
)
from scar_codec.numpy_engine import (
    BiorthogonalObliqueProjectorNumPy,
    NumPyVQ4ScarCodec,
    NumPyVQ4ScarPackage,
    OrthogonalSubspaceProjectorNumPy,
    SurpriseGatedAnomalySCacheNumPy,
    compute_lossless_pow2_scale,
    e8_spherical_ternary_gemm_np,
    get_compiled_c_lib,
)
from scar_codec.zarr_codec import ScarZarrCodec

try:
    from scar_codec.core import (
        BiorthogonalObliqueProjector,
        CircularRingBufferOT,
        OrthogonalSubspaceProjector,
        PhysicalVQ4ScarCodec,
        PhysicalVQ4ScarPackage,
        SurpriseGatedAnomalySCache,
        unique_tensor_storage_bytes,
    )
    HAS_TORCH = True
except ImportError:  # pragma: no cover - standalone zero-PyTorch runtime!
    HAS_TORCH = False
    PhysicalVQ4ScarCodec = NumPyVQ4ScarCodec  # type: ignore[misc,assignment]
    PhysicalVQ4ScarPackage = NumPyVQ4ScarPackage  # type: ignore[misc,assignment]
    OrthogonalSubspaceProjector = OrthogonalSubspaceProjectorNumPy  # type: ignore[misc,assignment]
    BiorthogonalObliqueProjector = BiorthogonalObliqueProjectorNumPy  # type: ignore[misc,assignment]
    SurpriseGatedAnomalySCache = SurpriseGatedAnomalySCacheNumPy  # type: ignore[misc,assignment]
    CircularRingBufferOT = None  # type: ignore[assignment]
    unique_tensor_storage_bytes = None  # type: ignore[assignment]

__all__ = [
    "__version__",
    "HAS_TORCH",
    "SCARFIELD_MAGIC",
    "SCAR_MAGIC",
    "BiorthogonalObliqueProjector",
    "BiorthogonalObliqueProjectorNumPy",
    "CircularRingBufferOT",
    "NumPyVQ4ScarCodec",
    "NumPyVQ4ScarPackage",
    "OrthogonalSubspaceProjector",
    "OrthogonalSubspaceProjectorNumPy",
    "PhysicalVQ4ScarCodec",
    "PhysicalVQ4ScarPackage",
    "ScarCodecE8",
    "ScarCodecND",
    "ScarE8Package",
    "ScarFieldMmapContainer",
    "ScarMmapFile",
    "ScarNDPackage",
    "ScarZarrCodec",
    "SurpriseGatedAnomalySCache",
    "SurpriseGatedAnomalySCacheNumPy",
    "build_canonical_e8_lut256_np",
    "compute_lossless_pow2_scale",
    "deserialize_scar_buffer",
    "e8_spherical_ternary_gemm_np",
    "get_compiled_c_lib",
    "load_scar_file",
    "save_scar_file",
    "save_scarfield_v7_container",
    "serialize_scar_bytes",
    "unique_tensor_storage_bytes",
]
