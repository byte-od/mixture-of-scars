"""ScarField v7.0 — Pure Post-Transformer Geodesic Scar Intelligence Package."""

from scar_field.byte_shock_simplex import (
    ByteShockFrontEnd,
    ScarFieldReflexDecision,
    SphericalCapSimplexHead,
    tsallis15_fenchel_young_loss,
    tsallis15_spherical_cap_probs,
)
from scar_field.cayley_geodesic import (
    HorizonCulledSymplecticGenerator,
    ReversibleSymplecticCayleyFlow,
    c1_spherical_cap_resonance,
    cayley_woodbury_step,
)
from scar_field.e8_lattice import (
    E8SphericalLinear,
    PackedE8TernaryMatrix,
    build_canonical_e8_lut256,
    fwht64_rademacher,
    pack_ternary_2bit,
    project_rows_to_unit_sphere,
    quantize_ternary_ste,
    rademacher_signs_det,
    unpack_ternary_2bit,
)
from scar_field.engine import (
    SCARFIELD_ASSISTANT_BYTE,
    SCARFIELD_EOT_BYTE,
    SCARFIELD_USER_BYTE,
    ScarFieldChatReply,
    ScarFieldConfig,
    ScarFieldEngine,
    encode_sft_dialogue_bytes_and_mask,
    format_scarfield_chat_turn,
)
from scar_field.proper_time_bus import (
    ChannelSelectiveKalmanScarBus,
    KalmanBusState,
    UScarPoleCartridge,
    UniversalScarCartridgeStore,
    VirtualGhostJumpDelta,
    apply_relativistic_proper_time_phase,
    biorthogonal_shock_bank_insert_step_,
    biorthogonal_shock_bank_read,
    chunkwise_wy_channel_selective_scan,
    compute_proper_time,
)
from scar_field.riemannian_muon import RiemannianMuon, heal_all_e8_scars
from scar_field.s1_scouts import (
    ScoutBatchTelemetry,
    System1CorpusScoutFleet,
    build_scout_g0_step_mask,
    compute_ghost_jump_antiloop_loss,
)
from scar_field.triton_kernels import (
    HAS_TRITON,
    conway_sloane_e8_cuda_fast,
    multiscale_wave_superpose_fast,
    tsallis15_spherical_cap_probs_fast,
)

__version__ = "7.5.0"

__all__ = [
    "ByteShockFrontEnd",
    "ChannelSelectiveKalmanScarBus",
    "E8SphericalLinear",
    "HAS_TRITON",
    "HorizonCulledSymplecticGenerator",
    "KalmanBusState",
    "PackedE8TernaryMatrix",
    "ReversibleSymplecticCayleyFlow",
    "RiemannianMuon",
    "ScarFieldConfig",
    "ScarFieldEngine",
    "ScarFieldReflexDecision",
    "ScoutBatchTelemetry",
    "SphericalCapSimplexHead",
    "System1CorpusScoutFleet",
    "UScarPoleCartridge",
    "UniversalScarCartridgeStore",
    "VirtualGhostJumpDelta",
    "apply_relativistic_proper_time_phase",
    "build_canonical_e8_lut256",
    "build_scout_g0_step_mask",
    "c1_spherical_cap_resonance",
    "cayley_woodbury_step",
    "chunkwise_wy_channel_selective_scan",
    "compute_ghost_jump_antiloop_loss",
    "compute_proper_time",
    "conway_sloane_e8_cuda_fast",
    "fwht64_rademacher",
    "heal_all_e8_scars",
    "multiscale_wave_superpose_fast",
    "pack_ternary_2bit",
    "project_rows_to_unit_sphere",
    "quantize_ternary_ste",
    "rademacher_signs_det",
    "tsallis15_fenchel_young_loss",
    "tsallis15_spherical_cap_probs",
    "tsallis15_spherical_cap_probs_fast",
    "unpack_ternary_2bit",
]
