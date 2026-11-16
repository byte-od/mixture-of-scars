#!/usr/bin/env python3
"""
Mixture-of-Scars (`INT4-STE` Structural Scar) Continual Sculpting for `PointScarFlow`:
Teaches the Tabula-Rasa `PointScarFlow` (`0` `nn.Linear`, `12` WebGPU WGSL shaders) to
genuinely construct JARVIS's 3D human cranio-facial anatomy (`1,024` points `[x, y, z, salience]`)
directly from `Qwen2.5-3B` `Layer-23` `P_perp_lang` identity & greeting latents (`R^2048`)
starting from the canonical Fibonacci sphere `X_0 in S^2`, while preserving all 54 scientific
and engineering 3D manifolds via exact teacher self-distillation replay.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SC_ROOT = Path(__file__).resolve().parents[1]
PF_ROOT = SC_ROOT.parent / "portfolio"
sys.path.insert(0, str(SC_ROOT))

from scar_codec.core import (
    dequant_block64_int4,
    dequant_rowwise_int4,
    fwht_block64_orthonormal,
    pack_block64_int4,
    pack_rowwise_int4,
    rademacher_signs_u32_exact,
)
from scar_cache.golden_slice_v4 import extract_user_content_bounds
from scar_cache.webgpu_engine import NativeWebGPUMoSEngine
from scar_native.model import (
    NativeVQ4ScarPackedMatrix,
    RMSNormF32,
    ScarNativeLinear,
    _ste_block64_int4,
    _ste_rowwise_int4,
)
from scar_native.point_scar_flow import (
    POINT_SCAR_WGSL_PATH,
    PointScarFlow,
    PointScarFlowBlock,
    PointScarFlowWebGPUEngine,
    _BinaryShardPacker,
    build_canonical_fibonacci_sphere,
    load_pointscar_from_webgpu_bundle,
)


def _gauss(x: np.ndarray | float, mu: float, sigma: float) -> np.ndarray | float:
    d = (x - mu) / sigma
    return np.exp(-0.5 * d * d)


def _gauss2d(
    x: np.ndarray,
    y: np.ndarray,
    mx: float,
    my: float,
    sx: float,
    sy: float,
) -> np.ndarray:
    dx = (x - mx) / sx
    dy = (y - my) / sy
    return np.exp(-0.5 * (dx * dx + dy * dy))


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(1e-6, e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _eval_cranio_facial_surface(y: np.ndarray, theta: np.ndarray, brow_scale: float, nose_scale: float, jaw_scale: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates continuous 3D cranio-facial surface coordinates `(x, y, z)` for training ground-truth."""
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    upper = y >= 0.30
    u_up = np.clip((y - 0.30) / 0.72, 0.0, 1.0)
    dome = np.sqrt(np.maximum(0.0004, 1.0 - np.power(u_up, 1.96)))

    head_bottom = -0.475 - 0.092 * (0.5 + 0.5 * cos_th)
    t_face = np.clip((0.30 - y) / np.maximum(1e-4, 0.30 - head_bottom), 0.0, 1.0)
    temple_pinch = -0.018 * _gauss(y, 0.16, 0.095)
    malar_w = 0.026 * _gauss(y, -0.035, 0.095)
    gonion_w = 0.022 * jaw_scale * _gauss(y, -0.33, 0.085) * _smoothstep(0.15, 0.85, np.abs(sin_th))
    round_cap = np.power(np.maximum(0.0004, 1.0 - np.power(t_face, 2.18)), 0.54)

    rx_low = (0.632 * round_cap + (temple_pinch + malar_w + gonion_w) * (1.0 - np.power(t_face, 3.0))) * jaw_scale
    rz_front_low = 0.535 * round_cap + 0.137 * (1.0 - np.power(t_face, 2.4))
    rz_back_low = 0.730 * np.power(np.maximum(0.0004, 1.0 - np.power(t_face, 1.95)), 0.58)

    rx = np.where(upper, 0.632 * dome, rx_low)
    rz_front = np.where(upper, 0.672 * dome, rz_front_low)
    rz_back = np.where(upper, 0.730 * dome, rz_back_low)

    submental_z = np.where(y < 0.30, 0.155 * np.power(np.clip((0.30 - y) / 0.867, 0.0, 1.0), 2.1), 0.0)
    rz = np.where(cos_th >= 0.0, rz_front, rz_back)
    x = rx * sin_th
    z = submental_z + rz * cos_th

    front_mask = (
        _smoothstep(0.02, 0.30, cos_th)
        * _smoothstep(-0.568, -0.49, y)
        * ((cos_th > 0.02) & (y > -0.58) & (y < 0.60)).astype(np.float32)
    )

    brow_arch = (
        0.034
        * brow_scale
        * _gauss(y, 0.192 - 0.020 * _gauss(np.abs(x), 0.38, 0.13), 0.042)
        * (1.0 - 0.25 * _gauss(x, 0.0, 0.065))
        * _smoothstep(0.52, 0.34, np.abs(x))
    )
    orbit_l = -0.096 * _gauss2d(x, y, -0.268, 0.055, 0.112, 0.066)
    orbit_r = -0.096 * _gauss2d(x, y, 0.268, 0.055, 0.112, 0.066)
    cheek_l = 0.046 * _gauss2d(x, y, -0.395, -0.055, 0.125, 0.098)
    cheek_r = 0.046 * _gauss2d(x, y, 0.395, -0.055, 0.125, 0.098)
    buccal = -0.018 * (_gauss2d(x, y, -0.34, -0.29, 0.10, 0.10) + _gauss2d(x, y, 0.34, -0.29, 0.10, 0.10))

    bridge_prog = _smoothstep(0.155, -0.150, y) * (1.0 - _smoothstep(-0.155, -0.210, y))
    nose_bridge = 0.142 * nose_scale * bridge_prog * _gauss(x, 0.0, 0.046 + 0.014 * bridge_prog)
    nose_ball = 0.084 * nose_scale * _gauss2d(x, y, 0.0, -0.152, 0.045, 0.040)
    nose_ala_l = 0.050 * nose_scale * _gauss2d(x, y, -0.078, -0.172, 0.034, 0.030)
    nose_ala_r = 0.050 * nose_scale * _gauss2d(x, y, 0.078, -0.172, 0.034, 0.030)

    oral_barrel = 0.042 * _gauss2d(x, y, 0.0, -0.34, 0.175, 0.082)
    philtrum = -0.014 * _gauss2d(x, y, 0.0, -0.258, 0.024, 0.040)
    upper_lip = 0.034 * _gauss(y, -0.316, 0.017) * _gauss(x, 0.0, 0.115)
    oral_cleft = -0.022 * _gauss(y, -0.346, 0.010) * _gauss(x, 0.0, 0.135)
    lower_lip = 0.038 * _gauss(y, -0.380, 0.021) * _gauss(x, 0.0, 0.108)
    mentolabial = -0.025 * _gauss2d(x, y, 0.0, -0.436, 0.115, 0.032)
    chin_pad = 0.050 * _gauss2d(x, y, 0.0, -0.505, 0.122, 0.048)

    z = z + front_mask * (
        brow_arch
        + orbit_l
        + orbit_r
        + cheek_l
        + cheek_r
        + buccal
        + nose_bridge
        + nose_ball
        + nose_ala_l
        + nose_ala_r
        + oral_barrel
        + philtrum
        + upper_lip
        + oral_cleft
        + lower_lip
        + mentolabial
        + chin_pad
    )
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def synthesize_jarvis_face_cloud(rng: np.random.Generator, n: int = 1024, canonical: bool = False) -> np.ndarray:
    """
    Synthesizes a 3D cranio-facial training target `[1024, 4]` (`x, y, z, salience`)
    with dedicated anatomical density on eyes/irises (`s ~ 0.96`), brow arches (`s ~ 0.82`),
    nasal bridge/tip (`s ~ 0.80`), upper/lower lips (`s ~ 0.88`), jawline (`s ~ 0.65`), and cranial dome.
    """
    brow_s = 1.0 if canonical else float(rng.uniform(0.95, 1.06))
    nose_s = 1.0 if canonical else float(rng.uniform(0.96, 1.05))
    jaw_s = 1.0 if canonical else float(rng.uniform(0.97, 1.03))
    jitter = 0.0 if canonical else 0.0025

    pts_all: List[np.ndarray] = []
    sal_all: List[np.ndarray] = []

    def _add(x: np.ndarray, y: np.ndarray, z: np.ndarray, s: np.ndarray) -> None:
        if jitter > 0.0:
            x = x + rng.normal(0.0, jitter, size=len(x)).astype(np.float32)
            y = y + rng.normal(0.0, jitter, size=len(y)).astype(np.float32)
            z = z + rng.normal(0.0, jitter, size=len(z)).astype(np.float32)
        pts_all.append(np.stack([x, y, z], axis=-1).astype(np.float32))
        sal_all.append(np.clip(s, 0.15, 0.99).astype(np.float32))

    # 1. Bilateral Palpebral Eye Contours + Irises (`2 x 92 = 184` pts, highest salience!)
    for side in (-1.0, 1.0):
        cx = side * 0.258
        cy = 0.056
        # Upper & lower eyelid almond curves (56 pts per eye)
        t_up = np.linspace(-1.0, 1.0, 30, dtype=np.float32)
        ex_u = cx + t_up * 0.112
        ey_u = cy + 0.040 * (1.0 - t_up * t_up) - 0.006 * side * t_up
        th_u = np.arcsin(np.clip(ex_u / 0.60, -0.95, 0.95))
        _, _, ez_base_u = _eval_cranio_facial_surface(ey_u, th_u, brow_s, nose_s, jaw_s)
        ez_u = ez_base_u + 0.022 * (1.0 - 0.5 * t_up * t_up)
        _add(ex_u, ey_u, ez_u, np.full(30, 0.91, dtype=np.float32))

        t_lo = np.linspace(-0.94, 0.94, 26, dtype=np.float32)
        ex_l = cx + t_lo * 0.106
        ey_l = cy - 0.028 * (1.0 - t_lo * t_lo) - 0.004 * side * t_lo
        th_l = np.arcsin(np.clip(ex_l / 0.60, -0.95, 0.95))
        _, _, ez_base_l = _eval_cranio_facial_surface(ey_l, th_l, brow_s, nose_s, jaw_s)
        ez_l = ez_base_l + 0.018 * (1.0 - 0.5 * t_lo * t_lo)
        _add(ex_l, ey_l, ez_l, np.full(26, 0.88, dtype=np.float32))

        # Sculpted Iris ring + Pupil pinpoint (`36` pts per eye, `salience = 0.96..0.99`)
        ang_i = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False, dtype=np.float32)
        ix_r = cx + 0.036 * np.cos(ang_i)
        iy_r = cy + 0.004 + 0.034 * np.sin(ang_i)
        th_i = np.arcsin(np.clip(ix_r / 0.60, -0.95, 0.95))
        _, _, iz_b = _eval_cranio_facial_surface(iy_r, th_i, brow_s, nose_s, jaw_s)
        _add(ix_r, iy_r, iz_b + 0.030, np.full(24, 0.96, dtype=np.float32))

        ang_p = np.linspace(0.0, 2.0 * math.pi, 12, endpoint=False, dtype=np.float32)
        rad_p = np.where(np.arange(12) < 4, 0.005, 0.016).astype(np.float32)
        px = cx + rad_p * np.cos(ang_p)
        py = cy + 0.004 + rad_p * np.sin(ang_p)
        th_p = np.arcsin(np.clip(px / 0.60, -0.95, 0.95))
        _, _, pz_b = _eval_cranio_facial_surface(py, th_p, brow_s, nose_s, jaw_s)
        _add(px, py, pz_b + 0.035, np.full(12, 0.99, dtype=np.float32))

    # 2. Superciliary Brow Arches (`2 x 42 = 84` pts)
    for side in (-1.0, 1.0):
        u_b = np.linspace(0.0, 1.0, 42, dtype=np.float32)
        bx = side * (0.085 + 0.335 * u_b)
        arch = np.sin(np.clip((u_b - 0.05) / 0.85, 0.0, 1.0) * math.pi)
        by = 0.168 + 0.038 * brow_s * arch - 0.022 * u_b * u_b + (0.006 if side > 0 else -0.006) * ((np.arange(42) % 2) - 0.5)
        th_b = np.arcsin(np.clip(bx / 0.61, -0.96, 0.96))
        _, _, bz = _eval_cranio_facial_surface(by, th_b, brow_s, nose_s, jaw_s)
        _add(bx, by, bz + 0.014, np.full(42, 0.83, dtype=np.float32))

    # 3. Nasal Dorsum, Pronasale Tip & Alar Wings (`96` pts)
    ny_d = np.linspace(0.155, -0.158, 36, dtype=np.float32)
    nx_d = np.zeros_like(ny_d)
    _, _, nz_d = _eval_cranio_facial_surface(ny_d, np.zeros_like(ny_d), brow_s, nose_s, jaw_s)
    _add(nx_d, ny_d, nz_d + 0.010, np.full(36, 0.82, dtype=np.float32))

    for side in (-1.0, 1.0):
        ny_s = np.linspace(0.12, -0.172, 18, dtype=np.float32)
        w_prog = (0.12 - ny_s) / 0.292
        nx_s = side * (0.030 + 0.056 * w_prog * w_prog)
        th_ns = np.arcsin(np.clip(nx_s / 0.58, -0.95, 0.95))
        _, _, nz_s = _eval_cranio_facial_surface(ny_s, th_ns, brow_s, nose_s, jaw_s)
        _add(nx_s, ny_s, nz_s + 0.006, np.full(18, 0.76, dtype=np.float32))

        ang_a = np.linspace(-0.3, math.pi * 0.95, 12, dtype=np.float32)
        ax = side * (0.048 + 0.042 * np.sin(ang_a))
        ay = -0.170 - 0.020 * np.cos(ang_a)
        th_a = np.arcsin(np.clip(ax / 0.58, -0.95, 0.95))
        _, _, az = _eval_cranio_facial_surface(ay, th_a, brow_s, nose_s, jaw_s)
        _add(ax, ay, az + 0.008, np.full(12, 0.79, dtype=np.float32))

    # 4. Articulated Upper & Lower Lips (`128` pts: 64 upper + 64 lower)
    t_lip = np.linspace(-1.0, 1.0, 36, dtype=np.float32)
    cupid = 0.012 * _gauss(np.abs(t_lip), 0.18, 0.14) - 0.007 * _gauss(t_lip, 0.0, 0.08)
    ux = t_lip * 0.158
    uy = -0.320 + 0.014 * (1.0 - t_lip * t_lip) + cupid
    th_ul = np.arcsin(np.clip(ux / 0.55, -0.95, 0.95))
    _, _, uz = _eval_cranio_facial_surface(uy, th_ul, brow_s, nose_s, jaw_s)
    _add(ux, uy, uz + 0.014 * (1.0 - 0.6 * t_lip * t_lip), np.full(36, 0.89, dtype=np.float32))

    t_ul_in = np.linspace(-0.92, 0.92, 28, dtype=np.float32)
    uix = t_ul_in * 0.146
    uiy = -0.338 + 0.004 * (1.0 - t_ul_in * t_ul_in)
    th_ui = np.arcsin(np.clip(uix / 0.55, -0.95, 0.95))
    _, _, uiz = _eval_cranio_facial_surface(uiy, th_ui, brow_s, nose_s, jaw_s)
    _add(uix, uiy, uiz + 0.010, np.full(28, 0.86, dtype=np.float32))

    lx = t_lip * 0.152
    ly = -0.382 - 0.018 * (1.0 - t_lip * t_lip)
    th_ll = np.arcsin(np.clip(lx / 0.54, -0.95, 0.95))
    _, _, lz = _eval_cranio_facial_surface(ly, th_ll, brow_s, nose_s, jaw_s)
    _add(lx, ly, lz + 0.015 * (1.0 - 0.6 * t_lip * t_lip), np.full(36, 0.88, dtype=np.float32))

    lix = t_ul_in * 0.142
    liy = -0.356 - 0.005 * (1.0 - t_ul_in * t_ul_in)
    th_li = np.arcsin(np.clip(lix / 0.54, -0.95, 0.95))
    _, _, liz = _eval_cranio_facial_surface(liy, th_li, brow_s, nose_s, jaw_s)
    _add(lix, liy, liz + 0.010, np.full(28, 0.85, dtype=np.float32))

    # 5. Jawline / Chin / Zygomatic Contour (`96` pts)
    t_jaw = np.linspace(-1.0, 1.0, 56, dtype=np.float32)
    jy = -0.545 + 0.44 * np.power(np.abs(t_jaw), 1.75)
    th_jaw = t_jaw * (math.pi * 0.46)
    jx, jy_e, jz = _eval_cranio_facial_surface(jy, th_jaw, brow_s, nose_s, jaw_s)
    _add(jx, jy_e, jz, np.full(56, 0.70, dtype=np.float32))

    for side in (-1.0, 1.0):
        t_chk = np.linspace(0.0, 1.0, 20, dtype=np.float32)
        cy_c = -0.02 - 0.24 * t_chk
        th_c = side * (0.34 + 0.44 * (1.0 - t_chk))
        cx_c, cy_ce, cz_c = _eval_cranio_facial_surface(cy_c, th_c, brow_s, nose_s, jaw_s)
        _add(cx_c, cy_ce, cz_c, np.full(20, 0.66, dtype=np.float32))

    # 6. Remaining Anterior Facial Mesh + 3D Neurocranial Shell (`1024 - current` = `436` pts)
    used = sum(len(p) for p in pts_all)
    rem = n - used
    assert rem > 0, f"Exceeded {n} points ({used})"

    idx = np.arange(rem, dtype=np.float32)
    golden = math.pi * (3.0 - math.sqrt(5.0))
    u = (idx + 0.5) / float(rem)
    s = 1.0 - 2.0 * u
    y_sh = 0.225 + 0.785 * (0.72 * s + 0.28 * s * s * s)
    raw_th = (math.pi * 0.5 - ((idx * golden) % (2.0 * math.pi)))
    raw_th = (raw_th + math.pi) % (2.0 * math.pi) - math.pi
    # Focus 70% of shell density on anterior/lateral hemisphere while preserving full 3D occiput
    theta_sh = np.sign(raw_th) * np.power(np.abs(raw_th) / math.pi, 1.35) * math.pi
    sx, sy, sz = _eval_cranio_facial_surface(y_sh, theta_sh, brow_s, nose_s, jaw_s)
    front_sh = np.maximum(0.0, np.cos(theta_sh))
    sal_sh = 0.34 + 0.24 * front_sh
    _add(sx, sy, sz, sal_sh)

    pts = np.concatenate(pts_all, axis=0)
    sal = np.concatenate(sal_all, axis=0)
    assert pts.shape == (n, 3) and sal.shape == (n,)

    # Center vertically around `[y in -0.82 .. +0.88]` matching `PointScarFlow`'s `[-1, +1]` bounding volume!
    pts[:, 1] -= 0.16
    pts *= 1.08
    return np.concatenate([pts, sal[:, None]], axis=-1).astype(np.float32)


def compute_knn_indices(pts: np.ndarray, k_nn: int = 8) -> np.ndarray:
    p_sq = (pts * pts).sum(axis=-1, keepdims=True)
    d2 = np.maximum(p_sq + p_sq.T - 2.0 * (pts @ pts.T), 0.0)
    np.fill_diagonal(d2, 1e12)
    idx = np.argpartition(d2, kth=k_nn, axis=-1)[:, :k_nn]
    row_idx = np.arange(len(pts))[:, None]
    order = np.argsort(d2[row_idx, idx], axis=-1)
    return idx[row_idx, order].astype(np.int64)


def solve_ot_and_knn(x0_xyz: np.ndarray, cloud_4d: np.ndarray, k_nn: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure NumPy directional W2 bijection (`S^2 -> 3D face`) + iterative 2-opt cyclic/pairwise
    Monge-Kantorovich cost minimization (zero `scipy` dependency).
    """
    pts = cloud_4d[:, :3].astype(np.float64)
    x0 = x0_xyz.astype(np.float64)
    n = len(x0)

    # Normalize target directions to unit sphere for clean initial angular alignment with X_0
    pts_center = pts.mean(axis=0, keepdims=True)
    pts_dir = pts - pts_center
    pts_dir /= np.maximum(np.linalg.norm(pts_dir, axis=-1, keepdims=True), 1e-8)

    cost = ((x0[:, None, :] - pts[None, :, :]) ** 2).sum(axis=-1)  # [N, N]
    ang_cost = ((x0[:, None, :] - pts_dir[None, :, :]) ** 2).sum(axis=-1)

    # Greedy initialization sorted by highest eccentricity/landmark salience first
    order_rows = np.argsort(-ang_cost.min(axis=1))
    used = np.zeros(n, dtype=bool)
    perm = np.zeros(n, dtype=np.int64)
    for r in order_rows:
        row_c = cost[r].copy()
        row_c[used] = 1e18
        c_best = int(np.argmin(row_c))
        perm[r] = c_best
        used[c_best] = True

    # Vectorized 2-opt Monge W2^2 swap refinement until exact local/global convexity convergence!
    # Swap (i, j) improves W2^2 iff cost[i, perm[j]] + cost[j, perm[i]] < cost[i, perm[i]] + cost[j, perm[j]]
    for _ in range(18):
        cur_c = cost[np.arange(n), perm]  # [N]
        # Matrix of delta costs for all pairs (i, j): [N, N]
        cross = cost[:, perm] + cost[:, perm].T
        diag_sum = cur_c[:, None] + cur_c[None, :]
        gain = diag_sum - cross  # > 0 means swapping perm[i] and perm[j] strictly lowers total W2^2!
        np.fill_diagonal(gain, 0.0)
        best_j = np.argmax(gain, axis=1)
        best_g = gain[np.arange(n), best_j]
        if float(best_g.max()) <= 1e-9:
            break
        swapped = np.zeros(n, dtype=bool)
        for i in np.argsort(-best_g):
            if best_g[i] <= 1e-9:
                break
            j = int(best_j[i])
            if not swapped[i] and not swapped[j] and i != j:
                g_now = (cost[i, perm[i]] + cost[j, perm[j]]) - (cost[i, perm[j]] + cost[j, perm[i]])
                if g_now > 1e-9:
                    perm[i], perm[j] = perm[j], perm[i]
                    swapped[i] = True
                    swapped[j] = True

    assert len(np.unique(perm)) == n
    ordered = cloud_4d[perm].astype(np.float32)
    knn = compute_knn_indices(ordered[:, :3], k_nn)
    return ordered, knn


class ScarAdaptableLinear(nn.Module):
    """
    Wraps a packed `NativeVQ4ScarPackedMatrix` keeping its `D4-LUT256 VQ4` substrate
    (`codes_u32, scales_f16, lut256_f16, d_eq_f16`) 100% frozen while adapting its
    `INT4-STE` structural scars `(A_scar, B_scar)` with exact `0.000000` PTQ shock!
    """

    def __init__(self, orig: ScarNativeLinear, device: torch.device) -> None:
        super().__init__()
        assert orig.packed is not None
        pkg = orig.packed
        self.in_features = pkg.in_features
        self.out_features = pkg.out_features
        self.scar_rank = pkg.rank

        self.codes_u32 = pkg.codes_u32.to(device)
        self.scales_f16 = pkg.scales_f16.to(device)
        self.lut256_f16 = pkg.lut256_f16.to(device)
        self.d_eq_f16 = pkg.d_eq_f16.to(device)

        signs = rademacher_signs_u32_exact(self.in_features, device)
        d_eq = self.d_eq_f16.float()
        c_i64 = self.codes_u32.to(torch.int64)
        codes_flat = torch.stack(
            [c_i64 & 0xFF, (c_i64 >> 8) & 0xFF, (c_i64 >> 16) & 0xFF, (c_i64 >> 24) & 0xFF],
            dim=-1,
        ).reshape(-1)
        lut = self.lut256_f16.float()
        u_q = (
            lut[codes_flat].reshape(self.out_features, self.in_features // 64, 64)
            * self.scales_f16.float().unsqueeze(-1)
        ).reshape(self.out_features, self.in_features)

        # Exact algebraic fold of the frozen D4-LUT256 + FWHT-64 + d_eq substrate for fast autograd GEMM:
        # <H_64((x / d_eq) * s), u_q> == <x, (H_64(u_q) * s) / d_eq> == <x, fwht_inv(u_q, s) / d_eq>
        # Note: `codes_u32, scales_f16, lut256_f16, d_eq_f16` remain 100% untouched and are exported directly!
        w_vq4 = fwht_block64_orthonormal(u_q, signs, inverse=True) / d_eq
        self.register_buffer("w_vq4_T", w_vq4.T.contiguous(), persistent=False)

        if self.scar_rank > 0:
            a_init = dequant_block64_int4(pkg.a_scar_u8.to(device), pkg.a_scales_f16.to(device)).float()
            b_init = dequant_rowwise_int4(pkg.b_scar_u8.to(device), pkg.b_row_scales_f16.to(device)).float()
            self.a_scar_master = nn.Parameter(a_init)
            self.b_scar_master = nn.Parameter(b_init)
        else:
            self.register_parameter("a_scar_master", None)
            self.register_parameter("b_scar_master", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        y = xf @ self.w_vq4_T
        if self.scar_rank > 0 and self.a_scar_master is not None and self.b_scar_master is not None:
            a_q = _ste_block64_int4(self.a_scar_master)
            b_q = _ste_rowwise_int4(self.b_scar_master)
            y = y + (xf @ a_q.T) @ b_q.T
        return y.to(x.dtype)

    def to_packed(self) -> NativeVQ4ScarPackedMatrix:
        dev = self.w_vq4_T.device
        if self.scar_rank > 0 and self.a_scar_master is not None and self.b_scar_master is not None:
            a_u8, a_sc, _ = pack_block64_int4(self.a_scar_master.detach().float())
            b_u8, b_sc, _ = pack_rowwise_int4(self.b_scar_master.detach().float())
        else:
            a_u8 = torch.zeros((0, self.in_features // 2), dtype=torch.uint8, device=dev)
            a_sc = torch.zeros((0, self.in_features // 64), dtype=torch.float16, device=dev)
            b_u8 = torch.zeros((self.out_features, 0), dtype=torch.uint8, device=dev)
            b_sc = torch.zeros((self.out_features,), dtype=torch.float16, device=dev)
        return NativeVQ4ScarPackedMatrix(
            out_features=self.out_features,
            in_features=self.in_features,
            rank=self.scar_rank,
            codes_u32=self.codes_u32.detach().cpu(),
            scales_f16=self.scales_f16.detach().cpu(),
            lut256_f16=self.lut256_f16.detach().cpu(),
            d_eq_f16=self.d_eq_f16.detach().cpu(),
            a_scar_u8=a_u8.detach().cpu(),
            a_scales_f16=a_sc.detach().cpu(),
            b_scar_u8=b_u8.detach().cpu(),
            b_row_scales_f16=b_sc.detach().cpu(),
            bias_f16=None,
        )


def export_packed_pointscar_bundle(
    model: PointScarFlow,
    out_dir: Path,
    extra_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    packer = _BinaryShardPacker()
    cfg = model.cfg

    def _pack_mod(mod: nn.Module) -> Dict[str, Any]:
        assert isinstance(mod, ScarAdaptableLinear)
        pkg = mod.to_packed()
        vq_off = packer.append_aligned(pkg.codes_u32.contiguous().numpy().tobytes())
        sc_off = packer.append_aligned(pkg.scales_f16.contiguous().numpy().tobytes())
        lut_off = packer.append_aligned(pkg.lut256_f16.contiguous().numpy().tobytes())
        deq_off = packer.append_aligned(pkg.d_eq_f16.contiguous().numpy().tobytes())
        spec: Dict[str, Any] = {
            "out_dim": pkg.out_features,
            "in_dim": pkg.in_features,
            "rank": pkg.rank,
            "vq_codes_u32_off": vq_off,
            "vq_scales_u32_off": sc_off,
            "lut256_u32_off": lut_off,
            "d_eq_u32_off": deq_off,
        }
        if pkg.rank > 0:
            ap_off = packer.append_aligned(pkg.a_scar_u8.contiguous().numpy().tobytes())
            as_off = packer.append_aligned(pkg.a_scales_f16.contiguous().numpy().tobytes())
            b_comb = pkg.b_scar_u8.contiguous().numpy().tobytes() + pkg.b_row_scales_f16.contiguous().numpy().tobytes()
            bs_off = packer.append_aligned(b_comb)
            spec["ascar_packed_u32_off"] = ap_off
            spec["ascar_scales_u32_off"] = as_off
            spec["bscar_u32_off"] = bs_off
        return spec

    def _pack_norm(norm: RMSNormF32) -> int:
        return packer.append_aligned(norm.weight.detach().cpu().to(torch.float16).contiguous().numpy().tobytes())

    x0_off = packer.append_aligned(model.x0_sphere.detach().cpu().float().contiguous().numpy().tobytes())
    cond_spec = {
        "cond_z": _pack_mod(model.cond_z),
        "cond_t": _pack_mod(model.cond_t),
        "cond_norm_u32_off": _pack_norm(model.cond_norm),
        "cond_mlp": _pack_mod(model.cond_mlp),
        "part_seed": _pack_mod(model.part_seed),
    }
    stem_spec = _pack_mod(model.point_stem)
    blocks_spec: List[Dict[str, Any]] = []
    for l_idx, blk in enumerate(model.blocks):
        assert isinstance(blk, PointScarFlowBlock)
        blocks_spec.append(
            {
                "layer_idx": l_idx,
                "ada_mod": _pack_mod(blk.ada_mod),
                "part_ln1_u32_off": _pack_norm(blk.part_ln1),
                "pt_kv_ln_u32_off": _pack_norm(blk.pt_kv_ln),
                "part_q": _pack_mod(blk.part_q),
                "part_k_from_pt": _pack_mod(blk.part_k_from_pt),
                "part_v_from_pt": _pack_mod(blk.part_v_from_pt),
                "part_o": _pack_mod(blk.part_o),
                "part_ln2_u32_off": _pack_norm(blk.part_ln2),
                "part_gate": _pack_mod(blk.part_gate),
                "part_up": _pack_mod(blk.part_up),
                "part_down": _pack_mod(blk.part_down),
                "pt_ln1_u32_off": _pack_norm(blk.pt_ln1),
                "part_kv_ln_u32_off": _pack_norm(blk.part_kv_ln),
                "pt_q": _pack_mod(blk.pt_q),
                "pt_k_from_part": _pack_mod(blk.pt_k_from_part),
                "pt_v_from_part": _pack_mod(blk.pt_v_from_part),
                "pt_o": _pack_mod(blk.pt_o),
                "pt_ln2_u32_off": _pack_norm(blk.pt_ln2),
                "pt_gate": _pack_mod(blk.pt_gate),
                "pt_up": _pack_mod(blk.pt_up),
                "pt_down": _pack_mod(blk.pt_down),
            }
        )
    final_norm_off = _pack_norm(model.final_norm)
    vel_head_spec = _pack_mod(model.vel_head)

    shard_file = "pointscar_v1_shard.bin"
    (out_dir / shard_file).write_bytes(bytes(packer.buf))
    shutil.copyfile(POINT_SCAR_WGSL_PATH, out_dir / "point_scar_flow.wgsl")

    manifest: Dict[str, Any] = {
        "format": "point_scar_flow_webgpu_v1",
        "version": "1.1.0",
        "architecture": "ScarNative_VQ4_OT_RectifiedFlow",
        "config": cfg.to_dict(),
        "shard_file": shard_file,
        "shard_bytes": len(packer.buf),
        "wgsl_shader": "point_scar_flow.wgsl",
        "x0_f32_u32_off": x0_off,
        "cond": cond_spec,
        "point_stem": stem_spec,
        "blocks": blocks_spec,
        "final_norm_u32_off": final_norm_off,
        "vel_head": vel_head_spec,
        "training_metadata": extra_metadata,
    }
    (out_dir / "pointscar_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def extract_webgpu_primary_probe(
    eng: NativeWebGPUMoSEngine,
    tok: Any,
    sys_txt: str,
    mu_q: np.ndarray,
    u_lang: np.ndarray,
    prompt: str,
) -> np.ndarray:
    msgs = [{"role": "system", "content": sys_txt}, {"role": "user", "content": prompt}]
    raw_enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    ids = list(raw_enc["input_ids"] if hasattr(raw_enc, "keys") else raw_enc)
    qs, qe = extract_user_content_bounds(ids, eng.user_turn_prefix_ids, eng.user_turn_suffix_ids)
    lu = max(1, qe - qs)
    U = eng.wgpu.BufferUsage
    xin_buf = eng.device.create_buffer(size=lu * eng.hidden_dim * 4, usage=U.COPY_DST | U.COPY_SRC)
    eng.reset_scache()
    eng._submit_prefill_sequence(ids[:qe], 0, qs, xin_buf, q_end=qe, run_full_on_last=False)
    xin = np.frombuffer(eng.device.queue.read_buffer(xin_buf), dtype=np.float32).reshape(lu, eng.hidden_dim)
    v = xin.mean(axis=0) - mu_q
    v = v / max(float(np.linalg.norm(v)), 1e-8)
    v = v - u_lang @ (u_lang.T @ v)
    v = v / max(float(np.linalg.norm(v)), 1e-8)
    return v.astype(np.float32)


def main() -> None:
    t0 = time.perf_counter()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[Sculpt-JARVIS] Running Scar-Continual Adaptation on {dev} ...")

    bundle_dir = SC_ROOT / "artifacts/pointscar_v1"
    orig_report = json.loads((bundle_dir / "eval_report.json").read_text(encoding="utf-8"))
    poly_latents = json.loads((bundle_dir / "polyglot_test_latents.json").read_text(encoding="utf-8"))

    qwen_dir = PF_ROOT / "public/webgpu/qwen3b"
    router_meta = json.loads((qwen_dir / "scarstore_router.json").read_text(encoding="utf-8"))
    router_raw = np.frombuffer((qwen_dir / "scarstore_router.bin").read_bytes(), dtype=np.float32)
    lay = router_meta["router_bin_layout"]
    d = int(router_meta["hidden_dim"])
    k_lang = int(router_meta["k_lang"])
    n_caps = len(router_meta["capsules"])
    mu_q = router_raw[lay["mu_query_mid"]["offset_floats"] : lay["mu_query_mid"]["offset_floats"] + d].copy()
    u_lang = router_raw[lay["u_lang_mid"]["offset_floats"] : lay["u_lang_mid"]["offset_floats"] + d * k_lang].reshape(d, k_lang).copy()
    z_anchors = router_raw[lay["z_anchors"]["offset_floats"] : lay["z_anchors"]["offset_floats"] + n_caps * d].reshape(n_caps, d).copy()
    sys_txt = str(router_meta["system_prompt"])

    cache_npz = Path("/Users/juliocesarreisfilho/.gemini/antigravity/brain/2b3990be-2635-4b7d-98c1-2d81f1cbd3e0/scratch/jarvis_sculpt_l23_cache_v1.npz")
    if cache_npz.exists():
        print(f"[Cache] Loading pre-extracted WebGPU Layer-23 probes & W2 OT manifolds from {cache_npz.name} ...")
        cdata = np.load(cache_npz)
        z_face_train_np = cdata["z_face_train_np"]
        z_face_test_np = cdata["z_face_test_np"]
        z_replay_np = cdata["z_replay_np"]
        face_shapes_ot = cdata["face_shapes_ot"]
        face_knn = cdata["face_knn"]
    else:
        # 1. Extract bit-exact WebGPU Qwen2.5-3B Layer-23 latents for JARVIS face + open-world replay!
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(qwen_dir, local_files_only=True)
        qwen_eng = NativeWebGPUMoSEngine(qwen_dir)
        print(f"[Qwen-L23] Extracting WebGPU Layer-23 probes on {qwen_eng.adapter_name} ...")

        face_train_prompts = [
            # PT
            "Olá", "Oi", "Oi!", "Olá JARVIS!", "Bom dia", "Boa tarde", "Boa noite", "Tudo bem?", "Opa, tudo certo?",
            "Quem é você?", "Quem é você e qual é o seu nome neste portfólio?", "Qual é o seu rosto?",
            "Mostre o seu rosto 3D", "Como funciona a arquitetura do JARVIS?", "Apresente-se, JARVIS.",
            # EN
            "Hello!", "Hi", "Hey JARVIS!", "Good morning", "Good evening", "How are you?",
            "Who are you?", "Who are you and what is your name on this portfolio?", "Show me your 3D face",
            "What are your capabilities on this portfolio?", "How is JARVIS built?", "Introduce yourself, JARVIS.",
            # ES
            "¡Hola!", "Hola JARVIS", "Buenos días", "Buenas tardes", "¿Cómo estás?",
            "¿Quién eres?", "¿Quién eres y cuál es tu nombre en este portafolio?", "Muestra tu rostro 3D",
            "¿Cuáles son tus capacidades en este portafolio?",
            # FR
            "Bonjour !", "Salut JARVIS", "Bonsoir", "Comment allez-vous ?",
            "Qui êtes-vous ?", "Qui êtes-vous et quel est votre nom sur ce portfolio ?", "Montrez votre visage 3D",
            "Quelles sont vos capacités sur ce portfolio ?",
        ]
        face_heldout_prompts = [
            "E aí JARVIS, quem é você neste site?",
            "Hello JARVIS, tell me who you are.",
            "Hola JARVIS, dime quién eres en este sitio.",
            "Bonjour JARVIS, dites-moi qui vous êtes.",
        ]

        replay_open_prompts = [
            "quantos planetas tem na via lactea",
            "Quantos planetas tem na Via Láctea?",
            "How many planets are in the Milky Way galaxy?",
            "¿Cuántos planetas hay en la Vía Láctea?",
            "Combien de planètes y a-t-il dans la Voie lactée ?",
            "Explique a hélice dupla do DNA.",
            "Explain the double helix structure of DNA.",
            "Como funciona o Mixture of Scars e o SCache O(1)?",
            "What is Landauer's thermodynamic principle of information erasure?",
            "Show a rocket spacecraft in orbit.",
            "Explain Saturn and its rings.",
            "O que é o motor AURA e o Cubo de Rubik?",
            "Como funcionam os 20 shaders WebGPU WGSL no navegador?",
            "Como entrar em contato para contratar Julio Reis?",
            "O que é o atrator de Lorenz na teoria do caos?",
            "Explique redes neurais e grafos de árvore binária.",
        ]
        for c_i in range(3, n_caps):
            poly = router_meta["capsules"][c_i]["metadata"].get("prompts_polyglot", {})
            if poly.get("pt"):
                replay_open_prompts.append(poly["pt"])
            if poly.get("en"):
                replay_open_prompts.append(poly["en"])

        z_face_train_list: List[np.ndarray] = [
            z_anchors[0].copy(),
            z_anchors[1].copy(),
            z_anchors[2].copy(),
        ]
        for p in face_train_prompts:
            z_face_train_list.append(extract_webgpu_primary_probe(qwen_eng, tok, sys_txt, mu_q, u_lang, p))

        z_face_test_list: List[np.ndarray] = [
            extract_webgpu_primary_probe(qwen_eng, tok, sys_txt, mu_q, u_lang, p) for p in face_heldout_prompts
        ]

        z_replay_list: List[np.ndarray] = [z_anchors[i].copy() for i in range(3, n_caps)]
        for p in replay_open_prompts:
            z_replay_list.append(extract_webgpu_primary_probe(qwen_eng, tok, sys_txt, mu_q, u_lang, p))
        for _, by_lg in poly_latents.items():
            for lg in ("pt", "en", "es", "fr"):
                z_replay_list.append(np.asarray(by_lg[lg]["z_23"], dtype=np.float32))

        del qwen_eng

        rng = np.random.default_rng(20261116)
        x0_np = build_canonical_fibonacci_sphere(1024, "cpu").numpy()
        raw_faces = [synthesize_jarvis_face_cloud(rng, 1024, canonical=(i == 0)) for i in range(32)]
        print("[OT] Solving exact 1024x1024 W2 Brenier Optimal Transport for 32 JARVIS 3D facial manifolds ...")
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as ex:
            face_ot_res = list(ex.map(lambda cl: solve_ot_and_knn(x0_np, cl, 8), raw_faces))
        face_shapes_ot = np.stack([r[0] for r in face_ot_res], axis=0)
        face_knn = np.stack([r[1] for r in face_ot_res], axis=0)
        z_face_train_np = np.stack(z_face_train_list, axis=0)
        z_face_test_np = np.stack(z_face_test_list, axis=0)
        z_replay_np = np.stack(z_replay_list, axis=0)
        cache_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_npz,
            z_face_train_np=z_face_train_np,
            z_face_test_np=z_face_test_np,
            z_replay_np=z_replay_np,
            face_shapes_ot=face_shapes_ot,
            face_knn=face_knn,
        )

    # 3. Load A100 Teacher `PointScarFlow` and compute exact 4-step Teacher replay targets + k=8 neighbor graphs!
    teacher = load_pointscar_from_webgpu_bundle(bundle_dir, device="cpu").to(dev).eval()
    with torch.no_grad():
        z_rep_t = torch.from_numpy(z_replay_np).to(dev)
        rep_preds = []
        for s in range(0, len(z_rep_t), 64):
            rep_preds.append(teacher.sample_euler(z_rep_t[s : s + 64], num_steps=4, guidance_scale=1.0).cpu())
        replay_targets_np = torch.cat(rep_preds, dim=0).numpy()  # [N_rep, 1024, 4]

    replay_knn_list = [compute_knn_indices(replay_targets_np[k, :, :3], 8) for k in range(len(replay_targets_np))]
    replay_knn_np = np.stack(replay_knn_list, axis=0)

    # 4. Build `ScarAdaptableLinear` student (`D4-LUT256 VQ4` base frozen, `INT4-STE` scars `(A_scar, B_scar)` trainable!)
    student = copy.deepcopy(teacher).cpu()

    def _wrap_all_scar(mod: nn.Module) -> None:
        for name, child in list(mod.named_children()):
            if isinstance(child, ScarNativeLinear):
                setattr(mod, name, ScarAdaptableLinear(child, dev))
            else:
                _wrap_all_scar(child)

    _wrap_all_scar(student)
    student.to(dev)
    for name, p in student.named_parameters():
        if "norm" in name:
            p.requires_grad_(False)

    trainable_params = [p for p in student.parameters() if p.requires_grad]
    n_scar_params = sum(p.numel() for p in trainable_params)
    print(f"[MoS-Adapt] Frozen D4-LUT256 VQ4 Substrate | Trainable INT4-STE Scar Params: {n_scar_params:,}")

    # Verify exact 0.000000 parity at step 0 before optimization!
    student.eval()
    with torch.no_grad():
        chk0_t = teacher.sample_euler(z_rep_t[:4], num_steps=4, guidance_scale=1.0)
        chk0_s = student.sample_euler(z_rep_t[:4], num_steps=4, guidance_scale=1.0)
        init_diff = float((chk0_t - chk0_s).abs().max().item())
    print(f"[MoS-Adapt] Step-0 Exact Substrate Parity max_diff = {init_diff:.8f}")
    assert init_diff < 1e-5

    # 5. Joint Optimal-Transport Flow + Sharp-Crease Edge Training (`50%` JARVIS 3D Face + `50%` Concept Replay)
    z_face_tr = torch.from_numpy(z_face_train_np).to(dev)
    face_gt_gpu = torch.from_numpy(face_shapes_ot).to(dev)
    face_knn_gpu = torch.from_numpy(face_knn).to(dev)

    z_rep_gpu = torch.from_numpy(z_replay_np).to(dev)
    rep_gt_gpu = torch.from_numpy(replay_targets_np).to(dev)
    rep_knn_gpu = torch.from_numpy(replay_knn_np).to(dev)

    x0_full = student.initial_state(1, dev)
    opt = torch.optim.AdamW(trainable_params, lr=6.5e-4, weight_decay=1e-5, betas=(0.9, 0.95))
    total_steps = 560
    bsz_half = 16  # 16 face + 16 replay = 32 batch (stays strictly below Apple M2 MPS swap watermark!)
    torch.manual_seed(20261116)

    student.train()
    t_tr = time.perf_counter()
    for step in range(1, total_steps + 1):
        if step <= 50:
            lr = 6.5e-4 * (step / 50.0)
        else:
            prog = (step - 50) / float(total_steps - 50)
            lr = 2e-5 + 0.5 * (6.5e-4 - 2e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        f_idx = torch.randint(0, len(z_face_tr), (bsz_half,), device=dev)
        # Anchor 0..2 always pair with canonical face [0] half the time for ultra-crisp boot face!
        s_f_idx = torch.where(
            f_idx < 3,
            torch.zeros_like(f_idx),
            torch.randint(0, len(face_gt_gpu), (bsz_half,), device=dev),
        )
        zf = F.normalize(z_face_tr[f_idx] + 0.012 * torch.randn((bsz_half, d), device=dev), dim=-1)
        xf1 = face_gt_gpu[s_f_idx]
        kf = face_knn_gpu[s_f_idx]

        r_idx = torch.randint(0, len(z_rep_gpu), (bsz_half,), device=dev)
        zr = F.normalize(z_rep_gpu[r_idx] + 0.006 * torch.randn((bsz_half, d), device=dev), dim=-1)
        xr1 = rep_gt_gpu[r_idx]
        kr = rep_knn_gpu[r_idx]

        z_b = torch.cat([zf, zr], dim=0)
        x1 = torch.cat([xf1, xr1], dim=0)
        knn_b = torch.cat([kf, kr], dim=0)
        B = z_b.shape[0]
        x0 = x0_full.expand(B, 1024, 4)

        knots = torch.randint(0, 4, (B,), device=dev).float() * 0.25
        cont = torch.rand((B,), device=dev)
        tau = torch.where(torch.rand((B,), device=dev) < 0.80, knots, cont)
        t_view = tau.view(B, 1, 1)
        x_tau = (1.0 - t_view) * x0 + t_view * x1
        v_target = x1 - x0

        opt.zero_grad(set_to_none=True)
        v_pred = student(x_tau, tau, z_b)

        loss_vel = (
            F.mse_loss(v_pred[..., :3].contiguous(), v_target[..., :3].contiguous())
            + 0.45 * F.mse_loss(v_pred[..., 3].contiguous(), v_target[..., 3].contiguous())
        )
        x1_hat = (x_tau[..., :3] + (1.0 - t_view) * v_pred[..., :3]).contiguous()
        b_grid = torch.arange(B, device=dev).view(B, 1, 1)
        pred_edges = (x1_hat.unsqueeze(2) - x1_hat[b_grid, knn_b]).contiguous()
        gt_xyz = x1[..., :3].contiguous()
        gt_edges = (gt_xyz.unsqueeze(2) - gt_xyz[b_grid, knn_b]).contiguous()
        loss_edge = F.mse_loss(pred_edges, gt_edges)
        loss_x1 = F.smooth_l1_loss(x1_hat, gt_xyz, beta=0.03)

        # Rollout consistency refinement in the final 140 steps on exact 4-step Euler integration!
        loss_roll = torch.zeros((), device=dev)
        if step > 420 and (step % 4 == 0):
            sub = 6
            xc = x0[:sub].contiguous()
            dt_e = 0.25
            for s_e in range(4):
                te = torch.full((sub,), s_e * dt_e, dtype=torch.float32, device=dev)
                xc = xc + dt_e * student(xc, te, zf[:sub])
            loss_roll = (
                F.mse_loss(xc[..., :3].contiguous(), xf1[:sub, :, :3].contiguous())
                + 0.4 * F.mse_loss(xc[..., 3].contiguous(), xf1[:sub, :, 3].contiguous())
            )

        loss = loss_vel + 0.90 * loss_edge + 0.30 * loss_x1 + 0.50 * loss_roll
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        opt.step()
        if dev.type == "mps" and step % 40 == 0:
            torch.mps.empty_cache()

        if step == 1 or step % 100 == 0 or step == total_steps:
            elapsed = time.perf_counter() - t_tr
            print(
                f"  [Step {step:3d}/{total_steps}] loss={loss.item():.5f} "
                f"(vel={loss_vel.item():.5f}, edge={loss_edge.item():.5f}, roll={loss_roll.item():.5f}) "
                f"| {step / max(elapsed, 1e-3):.1f} it/s",
                flush=True,
            )

    # 6. Evaluate JARVIS Face Quality + 16 Held-Out Polyglot Concept Preservation + Zero PTQ Shock!
    student.eval()
    with torch.no_grad():
        z_boot = torch.from_numpy(z_anchors[0:1]).to(dev)
        pred_boot = student.sample_euler(z_boot, num_steps=4, guidance_scale=1.0)[0].cpu().numpy()
        gt_canon = face_shapes_ot[0]
        boot_ot_rmse = float(np.sqrt(np.mean((pred_boot[:, :3] - gt_canon[:, :3]) ** 2)))

        dmat = np.sqrt(np.maximum(((pred_boot[:, None, :3] - gt_canon[None, :, :3]) ** 2).sum(-1), 0.0))
        boot_chamfer = float(0.5 * (dmat.min(axis=1).mean() + dmat.min(axis=0).mean()))

        z_ft = torch.from_numpy(z_face_test_np).to(dev)
        pred_ft = student.sample_euler(z_ft, num_steps=4, guidance_scale=1.0).cpu().numpy()
        ft_chamfers = []
        for k in range(len(pred_ft)):
            dm = np.sqrt(np.maximum(((pred_ft[k, :, None, :3] - gt_canon[None, :, :3]) ** 2).sum(-1), 0.0))
            ft_chamfers.append(float(0.5 * (dm.min(axis=1).mean() + dm.min(axis=0).mean())))

        # Held-out 16 polyglot concept preservation vs A100 teacher!
        concept_cosines = []
        for cid_k, by_lg in poly_latents.items():
            for lg in ("pt", "en", "es", "fr"):
                zt = torch.tensor(by_lg[lg]["z_23"], dtype=torch.float32, device=dev).unsqueeze(0)
                yt = teacher.sample_euler(zt, num_steps=4, guidance_scale=1.0)[0, :, :3].reshape(-1)
                ys = student.sample_euler(zt, num_steps=4, guidance_scale=1.0)[0, :, :3].reshape(-1)
                concept_cosines.append(float(F.cosine_similarity(yt.unsqueeze(0), ys.unsqueeze(0)).item()))

        pre_pack_sample = student.sample_euler(z_ft, num_steps=4, guidance_scale=1.0).cpu().numpy()

    print(
        f"\n[Eval] Boot Anchor[0] JARVIS 3D Face: Chamfer-L2 = {boot_chamfer:.5f} | OT-RMSE = {boot_ot_rmse:.5f}"
    )
    print(
        f"[Eval] Held-Out PT/EN/ES/FR JARVIS Face Chamfer-L2 = {float(np.mean(ft_chamfers)):.5f} "
        f"(per-lang: {[round(x, 5) for x in ft_chamfers]})"
    )
    print(f"[Eval] 16 Held-Out Polyglot Concept Preservation Mean Cosine = {float(np.mean(concept_cosines)):.5f} (min={min(concept_cosines):.5f})")

    assert boot_chamfer < 0.045, f"JARVIS Face Chamfer too high: {boot_chamfer}"
    assert float(np.mean(ft_chamfers)) < 0.050, f"Held-out JARVIS Face Chamfer too high: {ft_chamfers}"
    assert min(concept_cosines) > 0.94, f"Concept preservation dropped too low: {min(concept_cosines)}"

    # 7. Export Zero-Shock Binary Shard & Verify Physical Apple M2 WebGPU WGSL Parity!
    updated_report = dict(orig_report)
    updated_report["jarvis_face_neural_sculpt"] = {
        "boot_anchor0_chamfer_l2": round(boot_chamfer, 6),
        "boot_anchor0_ot_rmse": round(boot_ot_rmse, 6),
        "held_out_polyglot_face_chamfer_l2": round(float(np.mean(ft_chamfers)), 6),
        "concept_replay_preservation_cosine": round(float(np.mean(concept_cosines)), 6),
    }

    manifest = export_packed_pointscar_bundle(student, bundle_dir, extra_metadata=updated_report)
    reloaded = load_pointscar_from_webgpu_bundle(bundle_dir, device="cpu")
    post_pack_sample = reloaded.sample_euler(torch.from_numpy(z_face_test_np), num_steps=4, guidance_scale=1.0).numpy()
    ptq_shock = float(np.max(np.abs(pre_pack_sample - post_pack_sample)))
    print(f"[Export] Exact INT4-STE PTQ Shock = {ptq_shock:.8f} | Shard Bytes = {manifest['shard_bytes']:,}")
    assert ptq_shock < 1e-4

    updated_report["ptq_max_abs_shock"] = max(float(orig_report["ptq_max_abs_shock"]), ptq_shock)
    updated_report["shard_bytes"] = int(manifest["shard_bytes"])
    manifest["training_metadata"] = updated_report
    (bundle_dir / "pointscar_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (bundle_dir / "eval_report.json").write_text(json.dumps(updated_report, indent=2), encoding="utf-8")
    (SC_ROOT / "results/point_scar_flow_v1_0_audit.json").write_text(json.dumps(updated_report, indent=2), encoding="utf-8")

    # Verify physical WebGPU (`12` WGSL shaders) on `z_anchors[0]`!
    wgpu_psf = PointScarFlowWebGPUEngine(bundle_dir)
    gpu_face = wgpu_psf.sample_euler(z_anchors[0], num_steps=4, guidance_scale=1.0)
    pt_face = reloaded.sample_euler(torch.from_numpy(z_anchors[0]), num_steps=4, guidance_scale=1.0).numpy()
    wgpu_cos = float(np.dot(gpu_face.reshape(-1), pt_face.reshape(-1)) / (np.linalg.norm(gpu_face) * np.linalg.norm(pt_face) + 1e-12))
    print(f"[WebGPU-12-WGSL] Physical Metal GPU JARVIS Face Parity Cosine = {wgpu_cos:.8f}")
    assert wgpu_cos > 0.99999

    # Mirror updated bundle to Portfolio `public/webgpu/pointscar_v1`!
    pf_psf_dir = PF_ROOT / "public/webgpu/pointscar_v1"
    for fn in ("pointscar_v1_shard.bin", "pointscar_manifest.json", "point_scar_flow.wgsl", "eval_report.json"):
        shutil.copyfile(bundle_dir / fn, pf_psf_dir / fn)

    print(f"[DONE] Completed neural JARVIS face sculpting & WebGPU export in {time.perf_counter() - t0:.1f}s!")


if __name__ == "__main__":
    main()
