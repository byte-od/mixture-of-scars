"""SCache (Gradient Scar Memory): Universal O(1) Fast-Weights Associative Memory.

Substrato de memória de tamanho constante O(1) guiada por gradientes,
imunidade de gating por manifold, projeção de conflito ortogonal (PCGrad)
e arquitetura tensorial Multi-Slot com Regra Delta para substituição do KV Cache.
"""

from scar_cache.attention import SlotScarAttention
from scar_cache.core import GradientScarMemory, SCache
from scar_cache.hf_cache import SCacheDynamicCache, SCacheDynamicLayer
from scar_cache.modules import ScarGatedLayer, ScarLinear, SlotScarLinear
from scar_cache.mos import MixtureOfScarsLinear, ScarExpert
from scar_cache.cli_72b import CanonicalMoSEngine, CanonicalMoSGroup
from scar_cache.native_70b import NativeMoS70B
from scar_cache.convert_qwen import stream_convert_qwen72b
from scar_cache.multi_slot import MultiSlotSCache
from scar_cache.projections import (
    delta_rule_update,
    hebbian_outer_product,
    project_conflicting_gradient,
    project_conflicting_gradients_per_slot,
)
from scar_cache.triton_mos_gemm import (
    fused_int4_mos_linear,
    fused_mos_linear,
    triton_fused_int4_mos_linear,
)

__version__ = "1.3.0"

__all__ = [
    "SCache",
    "GradientScarMemory",
    "MultiSlotSCache",
    "SlotScarAttention",
    "SCacheDynamicCache",
    "SCacheDynamicLayer",
    "ScarLinear",
    "ScarGatedLayer",
    "SlotScarLinear",
    "ScarExpert",
    "MixtureOfScarsLinear",
    "CanonicalMoSEngine",
    "CanonicalMoSGroup",
    "NativeMoS70B",
    "stream_convert_qwen72b",
    "fused_mos_linear",
    "fused_int4_mos_linear",
    "triton_fused_int4_mos_linear",
    "project_conflicting_gradient",
    "project_conflicting_gradients_per_slot",
    "delta_rule_update",
    "hebbian_outer_product",
    "__version__",
]
