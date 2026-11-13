"""SCache (Gradient Scar Memory): Universal O(1) Fast-Weights Associative Memory.

Substrato de memória de tamanho constante O(1) guiada por gradientes,
imunidade de gating por manifold, projeção de conflito ortogonal (PCGrad)
e arquitetura tensorial Multi-Slot com Regra Delta para substituição do KV Cache.
"""

from scar_cache.attention import SlotScarAttention
from scar_cache.core import GradientScarMemory, SCache
from scar_cache.hf_cache import (
    SCacheDynamicCache,
    SCacheDynamicLayer,
    SCacheLayerSnapshot,
    SCacheSnapshot,
    SpeculativeCaptureContext,
)
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
from scar_cache.scar_grafting import (
    ScarBank,
    ScarCapsule,
    ScarGraftEngine,
    SleepEpisode,
    extract_span_query_probes,
)
from scar_cache.scar_store_v2 import (
    MissingCapsuleAlert,
    MultiCapsuleGenerationResult,
    MultiCapsuleSegment,
    QuantizedScarCapsule,
    ScarStoreV2,
    dequantize_block64_int4,
    quantize_block64_int4,
)
from scar_cache.speculative_ghost_jump import (
    GhostJumpGenerationResult,
    GhostJumpReflexDecision,
    GhostJumpSpeculativeEngine,
)
from scar_cache.sub2bit_mos import (
    SUB2BIT_BETA,
    Sub2BitDecompositionStats,
    Sub2BitMoSLinear,
    calibrate_linear_activation_scales,
    convert_hf_mlp_to_sub2bit_mos,
    decompose_sub2bit_mos,
    dequantize_sub2bit_coalesced,
    pack_sub2bit_codes,
    quantize_sub2bit_block64,
    sub2bit_mos_linear,
    transcode_mos_v2_shard_to_sub2bit,
    unpack_sub2bit_codes,
)
from scar_cache.triton_mos_gemm import (
    can_use_triton_int2_mos,
    fused_int4_mos_linear,
    fused_mos_linear,
    triton_fused_int2_mos_linear,
    triton_fused_int4_mos_linear,
)

__version__ = "2.0.0a3"

__all__ = [
    "SCache",
    "GradientScarMemory",
    "MultiSlotSCache",
    "SlotScarAttention",
    "SCacheDynamicCache",
    "SCacheDynamicLayer",
    "SCacheLayerSnapshot",
    "SCacheSnapshot",
    "SpeculativeCaptureContext",
    "GhostJumpReflexDecision",
    "GhostJumpGenerationResult",
    "GhostJumpSpeculativeEngine",
    "SleepEpisode",
    "ScarCapsule",
    "ScarBank",
    "ScarGraftEngine",
    "extract_span_query_probes",
    "MissingCapsuleAlert",
    "MultiCapsuleSegment",
    "MultiCapsuleGenerationResult",
    "QuantizedScarCapsule",
    "ScarStoreV2",
    "quantize_block64_int4",
    "dequantize_block64_int4",
    "SUB2BIT_BETA",
    "Sub2BitDecompositionStats",
    "Sub2BitMoSLinear",
    "pack_sub2bit_codes",
    "unpack_sub2bit_codes",
    "quantize_sub2bit_block64",
    "dequantize_sub2bit_coalesced",
    "decompose_sub2bit_mos",
    "sub2bit_mos_linear",
    "calibrate_linear_activation_scales",
    "convert_hf_mlp_to_sub2bit_mos",
    "transcode_mos_v2_shard_to_sub2bit",
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
    "can_use_triton_int2_mos",
    "triton_fused_int2_mos_linear",
    "project_conflicting_gradient",
    "project_conflicting_gradients_per_slot",
    "delta_rule_update",
    "hebbian_outer_product",
    "__version__",
]
