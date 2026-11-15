"""Pure Scar-Native Foundation Neural Architecture (`scar_native` v5.0.0)."""

from scar_native.data import (
    DETACHED_ENTITIES_16,
    LANGUAGES,
    PARALLEL_CONCEPT_QUADS,
    DetachedEntitySpec,
    build_contrastive_nullspace_batch,
    build_logic_sample,
    build_overflow_passkey_sample,
    build_pretraining_tensors,
    format_detached_entity_prompt,
)
from scar_native.model import (
    DenseTransformerBaseline,
    DetachedScarCartridge,
    DetachedScarStore,
    DifferentiableSCacheAttention,
    EpistemicTelemetry,
    LanguageNullspaceProjector,
    RMSNormF32,
    ScarNativeBlock,
    ScarNativeConfig,
    ScarNativeLinear,
    ScarNativeLM,
    ScarNativeMLP,
    ScarNativeO1State,
    build_canonical_d4_lut256,
    compute_excess_kurtosis,
)
from scar_native.tokenizer import ScarNativeTokenizer
from scar_native.webgpu_v5 import ScarNativeWebGPUEngine, export_scar_native_webgpu_bundle

__version__ = "5.0.0"

__all__ = [
    "__version__",
    "DETACHED_ENTITIES_16",
    "LANGUAGES",
    "PARALLEL_CONCEPT_QUADS",
    "DetachedEntitySpec",
    "build_contrastive_nullspace_batch",
    "build_logic_sample",
    "build_overflow_passkey_sample",
    "build_pretraining_tensors",
    "format_detached_entity_prompt",
    "DenseTransformerBaseline",
    "DetachedScarCartridge",
    "DetachedScarStore",
    "DifferentiableSCacheAttention",
    "EpistemicTelemetry",
    "LanguageNullspaceProjector",
    "RMSNormF32",
    "ScarNativeBlock",
    "ScarNativeConfig",
    "ScarNativeLinear",
    "ScarNativeLM",
    "ScarNativeMLP",
    "ScarNativeO1State",
    "build_canonical_d4_lut256",
    "compute_excess_kurtosis",
    "ScarNativeTokenizer",
    "ScarNativeWebGPUEngine",
    "export_scar_native_webgpu_bundle",
]
