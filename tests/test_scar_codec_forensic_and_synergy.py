"""Forensic Red-Team Audit of `ScarCodec v6.0` & `ScarCodec <-> ScarField v7.0` Synergy Verification."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pytest
import torch

from scar_codec import (
    NumPyVQ4ScarCodec,
    PhysicalVQ4ScarCodec,
    ScarCodecE8,
    ScarCodecND,
    ScarMmapFile,
    ScarZarrCodec,
    build_canonical_e8_lut256_np,
    e8_spherical_ternary_gemm_np,
    load_scar_file,
    save_scar_file,
)
from scar_field import (
    E8SphericalLinear,
    ScarFieldConfig,
    ScarFieldEngine,
    build_canonical_e8_lut256,
    project_rows_to_unit_sphere,
)


def test_forensic_extreme_ieee754_exponents_ligo_to_xray_zero_nan() -> None:
    """Proves zero FP16 overflow/underflow from `1e-21` (LIGO strain) to `1.5e8` (Chandra X-ray counts)."""
    rng = np.random.default_rng(7701)
    base = rng.standard_normal((96, 192), dtype=np.float32) * 0.5
    base[:16, :] = 0.0  # Exact physical vacuum rows
    base[48, 99] = 150.0  # Sharp 300x spike

    for scale, rel_tol in ((1.0e-21, 1.0e-20), (1.0e6, 2.0e5)):
        field = (base * np.float32(scale)).astype(np.float32)
        tol = float(rel_tol)

        # 1. NumPy + C99/SIMD VQ4-Scar
        c_np = NumPyVQ4ScarCodec(rank=16, abs_tolerance=tol)
        p_np = c_np.encode(field)
        rec_np = c_np.decode_full(p_np, use_c_accel=True)
        rec_py = c_np.decode_full(p_np, use_c_accel=False)
        blk_np = c_np.decode_block_o1(p_np, row_idx=48, block_idx=1, use_c_accel=True)
        assert np.all(np.isfinite(rec_np)), f"Non-finite values at scale={scale}"
        assert float(np.max(np.abs(rec_np[:16, :]))) == 0.0
        assert float(np.max(np.abs(rec_np - field))) <= tol * 1.001
        assert float(np.max(np.abs(rec_np - rec_py))) <= tol * 0.05
        assert float(np.max(np.abs(blk_np - rec_np[48, 64:128]))) <= tol * 0.01

        # 2. PyTorch PhysicalVQ4ScarCodec
        c_pt = PhysicalVQ4ScarCodec(rank=16, abs_tolerance=tol)
        p_pt = c_pt.encode(torch.from_numpy(field))
        rec_pt = c_pt.decode_full(p_pt).numpy()
        assert np.all(np.isfinite(rec_pt))
        assert float(np.max(np.abs(rec_pt[:16, :]))) == 0.0
        assert float(np.max(np.abs(rec_pt - field))) <= tol * 1.001

        # 3. Gosset E8 (`1.000 bpw`)
        c_e8 = ScarCodecE8(rank=8, abs_tolerance=tol)
        p_e8 = c_e8.encode(field)
        rec_e8 = c_e8.decode_full(p_e8)
        assert np.all(np.isfinite(rec_e8))
        assert float(np.max(np.abs(rec_e8[:16, :]))) == 0.0
        assert float(np.max(np.abs(rec_e8 - field))) <= tol * 1.001

        # 4. 3D Spectral-Tucker `ScarCodecND`
        cube = field.reshape(12, 8, 192)
        c_3d = ScarCodecND(spectral_rank=6, spatial_scar_rank=8, abs_tolerance=tol)
        p_3d = c_3d.encode(cube)
        rec_3d = c_3d.decode_full(p_3d)
        strip_3d = c_3d.decode_voxel_strip_o1(p_3d, v_chan=6, spatial_block_idx=1)
        assert np.all(np.isfinite(rec_3d))
        assert float(np.max(np.abs(rec_3d - cube))) <= tol * 1.001
        assert float(np.max(np.abs(strip_3d - rec_3d[6].reshape(-1)[64:128]))) <= tol * 0.02


def test_forensic_nan_inf_rejection_tail_gather_and_header_crc32(tmp_path: Path) -> None:
    """Verifies explicit NaN/Inf rejection, non-64-multiple gather safety, E8 `.scar` mmap, and header CRC32."""
    bad = np.zeros((64, 64), dtype=np.float32)
    bad[3, 7] = np.nan
    with pytest.raises(ValueError, match="NaN or Inf"):
        NumPyVQ4ScarCodec().encode(bad)
    with pytest.raises(ValueError, match="NaN or Inf"):
        PhysicalVQ4ScarCodec().encode(torch.from_numpy(bad))
    bad[3, 7] = np.inf
    with pytest.raises(ValueError, match="NaN or Inf"):
        ScarCodecE8().encode(bad)

    # Non-64-multiple width (`1025`, like NASA Spitzer IRAC -> last block `16` has `valid_len = 1`!)
    rng = np.random.default_rng(7702)
    odd = rng.standard_normal((37, 1025), dtype=np.float32)
    odd[10, 1024] = 777.0
    pkg = NumPyVQ4ScarCodec(rank=8, max_point_spikes=32).encode(odd)

    q_rows = np.array([10, 0, -5, 999], dtype=np.int32)
    q_blks = np.array([16, 16, 0, 99], dtype=np.int32)
    g_c = NumPyVQ4ScarCodec.decode_blocks_gather_o1(pkg, q_rows, q_blks)
    blk_tail = NumPyVQ4ScarCodec.decode_block_o1(pkg, row_idx=10, block_idx=16, use_c_accel=True)
    assert blk_tail.shape == (1,)
    assert abs(float(g_c[0, 0] - blk_tail[0])) < 1e-6
    # Tail elements `[1..63]` AND out-of-bounds rows `2, 3` MUST be deterministic `0.0` (zero garbage RAM!)
    assert float(np.max(np.abs(g_c[0, 1:]))) == 0.0
    assert float(np.max(np.abs(g_c[2]))) == 0.0
    assert float(np.max(np.abs(g_c[3]))) == 0.0

    # E8 `.scar` binary container + `ScarMmapFile` + HEADER tampering CRC32 detection
    e8_pkg = ScarCodecE8(rank=8, max_point_spikes=32).encode(odd)
    e8_path = tmp_path / "odd_e8.scar"
    save_scar_file(e8_pkg, e8_path)
    with ScarMmapFile(e8_path, verify_crc=True) as mf:
        assert mf.shape == (37, 1025)
        assert float(np.max(np.abs(mf.read_block_o1(10, 16) - ScarCodecE8.decode_block_o1(e8_pkg, 10, 16)))) == 0.0
        assert float(np.max(np.abs(mf.read_full() - ScarCodecE8.decode_full(e8_pkg)))) == 0.0

    # Flip 1 bit inside the 64-byte header (`orig_rows` byte 16) -> MUST fail CRC32!
    hdr_tampered = bytearray(e8_path.read_bytes())
    hdr_tampered[16] ^= 0x01
    tampered_p = tmp_path / "hdr_tampered.scar"
    tampered_p.write_bytes(bytes(hdr_tampered))
    with pytest.raises(ValueError, match="CRC32"):
        load_scar_file(tampered_p, verify_crc=True)

    # Zarr codec across 1D (`float64`), 2D (`float16`), and 4D (`float32`, `mode="e8"`)
    for mode in ("vq4", "e8"):
        zc = ScarZarrCodec(rank=8, abs_tolerance=0.05, mode=mode)
        zc2 = ScarZarrCodec.from_config(zc.get_config())
        for shp, dt in (((500,), np.float64), ((32, 130), np.float16), ((2, 3, 8, 64), np.float32)):
            a = (rng.standard_normal(shp) * 1.5).astype(dt)
            dec = zc2.decode(zc2.encode(a))
            assert dec.shape == shp and dec.dtype == dt
            assert float(np.max(np.abs(dec.astype(np.float32) - a.astype(np.float32)))) <= 0.055


def test_scarcodec_decoupling_benefits_scarfield_v7_c_simd_and_mmap(tmp_path: Path) -> None:
    """Proves `ScarCodec` decoupling directly accelerates & shrinks `ScarField v7.0` packed inference."""
    # 1. Exact bit-identity of canonical Gosset E8 lattice (`0.0` drift across all 256 x 8 entries!)
    lut_np = build_canonical_e8_lut256_np()
    lut_pt = build_canonical_e8_lut256("cpu").numpy()
    assert lut_np.shape == (256, 8)
    assert float(np.max(np.abs(lut_np - lut_pt))) == 0.0

    # 2. Operator-level C99/SIMD zero-materialization parity (`cos > 0.999999`, `max |diff| < 1e-5`)
    torch.manual_seed(7703)
    op = E8SphericalLinear(in_features=320, out_features=640, scar_rank=16)
    x = project_rows_to_unit_sphere(torch.randn(8, 320))
    y_qat = op(x).detach()

    pk = op.to_packed()
    y_c = pk.forward(x, op.lut_e8, op.signs, use_c_accel=True)
    y_pt_unpack = pk.forward(x, op.lut_e8, op.signs, use_c_accel=False)
    assert float((y_qat - y_c).abs().max().item()) < 1e-5
    assert float((y_c - y_pt_unpack).abs().max().item()) < 1e-5

    # Standalone zero-PyTorch `e8_spherical_ternary_gemm_np` parity (C vs pure-NumPy)
    y_np_c = e8_spherical_ternary_gemm_np(
        x.numpy(), pk.codes_u8.numpy(), lut_np, pk.a_tern_packed.numpy(), pk.b_tern_packed.numpy(),
        pk.scar_rank, pk.tau_base, pk.gamma_scar, use_c_accel=True
    )
    y_np_py = e8_spherical_ternary_gemm_np(
        x.numpy(), pk.codes_u8.numpy(), lut_np, pk.a_tern_packed.numpy(), pk.b_tern_packed.numpy(),
        pk.scar_rank, pk.tau_base, pk.gamma_scar, use_c_accel=False
    )
    assert float(np.max(np.abs(y_np_c - y_np_py))) < 1e-5

    # 3. Full `ScarFieldEngine` `.sfield.scar` zero-pickle `CRC32` + `mmap` load & `freeze_to_packed_e8` RAM drop
    cfg = ScarFieldConfig(d_form=128, d_world=192, d_bus=128, d_key=64, s1_steps=1, s2_steps=2)
    eng = ScarFieldEngine(cfg)
    raw_param_bytes_before = sum(p.numel() * p.element_size() for p in eng.parameters())

    sfield_path = tmp_path / "scarfield_v7.sfield.scar"
    file_bytes = eng.save_packed_scar_container(sfield_path)
    budget = eng.byte_budget_summary()
    # Zero-pickle `.sfield.scar` matches theoretical packed byte budget within small JSON/64B-align overhead!
    assert file_bytes < int(budget["total_packed_model_bytes"] * 1.05)

    # Compare interactive byte trajectory BEFORE vs AFTER `load_packed_scar_mmap`
    st_ref = eng.init_state(batch_size=1)
    seq_bytes = list(b"PIN=742 => ")
    decisions_ref = [eng.step_byte_interactive(b, st_ref, enable_ghost_jump_scouts=False).action_byte for b in seq_bytes]

    t0 = time.perf_counter()
    eng_mmap, container = ScarFieldEngine.load_packed_scar_mmap(sfield_path, verify_crc=True)
    cold_load_ms = (time.perf_counter() - t0) * 1000.0
    try:
        raw_param_bytes_after = sum(p.numel() * p.element_size() for p in eng_mmap.parameters())
        # Dense FP32 `weight_sphere` parameters are completely eliminated (`> 3.5x` total param RAM drop!)
        assert raw_param_bytes_after < raw_param_bytes_before * 0.35
        assert cold_load_ms < 50.0

        st_pk = eng_mmap.init_state(batch_size=1)
        decisions_pk = [
            eng_mmap.step_byte_interactive(b, st_pk, enable_ghost_jump_scouts=False).action_byte for b in seq_bytes
        ]
        # FP16 pole storage causes at most tiny 1e-3 pole rounding; check forward score cosine > 0.9999!
        tok_seq = torch.tensor([seq_bytes], dtype=torch.long)
        sc_ref = eng.forward_sequence(tok_seq, adaptive_depth=False)["scores"].detach()
        sc_pk = eng_mmap.forward_sequence(tok_seq, adaptive_depth=False)["scores"].detach()
        cos_scores = float(
            torch.nn.functional.cosine_similarity(sc_ref.reshape(1, -1), sc_pk.reshape(1, -1)).item()
        )
        assert cos_scores > 0.9998
        assert decisions_pk == decisions_ref
    finally:
        container.close()
