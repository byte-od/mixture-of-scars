"""Mixture of Scars (MoS) v5.0, Pure Scar-Native Foundation Engine, SCache O(1) & ScarCodec: Tabula-Rasa D4 VQ4-STE Pretraining, End-to-End O(1) Widrow-Hoff Fast-Weights, 22-Shader WebGPU WGSL 1.0, Sub-2-Bit/VQ4-Scar/ProtectedInt4Scar, Language-Nullspace Cartridges, Sinkhorn Re-Basin, Ghost-Jump Reflex & Physical Signal/Telemetry/CFD Operators."""

import scar_cache as _sc

__version__ = _sc.__version__
__all__ = list(_sc.__all__)

for _sym in __all__:
    globals()[_sym] = getattr(_sc, _sym)
