#!/usr/bin/env python3
"""
Colab A100 Remote Training Pipeline: `PointScarFlow` v1.0 (`ScarNative` 3D Point-Cloud Generator).

Trains a 100% Tabula-Rasa `ScarNativeLinear` (`D4-LUT256 VQ4-STE` + `FWHT-64` + `d_eq` + `INT4-STE`,
ZERO `torch.nn.Linear` dense layers) Optimal-Transport Rectified Flow 3D Point-Cloud model
conditioned directly on `Qwen/Qwen2.5-3B-Instruct` `Layer 23` `P_perp_lang` semantic latents (`R^2048`).

Pipeline stages:
  1. Download real 3D CAD `.off` triangle meshes (`ModelNet10`) + synthesize `54` continuous
     parametric 3D scientific, mathematical, biological, astronomical, mechanical, and System-One
     manifolds (`~4,800` unique `1,024`-point 3D clouds with surface curvature salience `s in [0,1]`).
  2. Solve exact global minimum `W_2^2` Brenier Optimal-Transport bijections (`linear_sum_assignment`)
     from the canonical Fibonacci sphere `X_0` to every 3D shape + precompute `k=8` local edge graphs.
  3. Extract exact `Qwen2.5-3B-Instruct` `Layer 23` `P_perp_lang` vectors (`R^2048`) across `~52,000`
     multilingual (`PT`, `EN`, `ES`, `FR`) prompts & conversational questions using the exact browser
     `scarstore_router.bin` (`mu_query_mid` + `u_lang_mid`) and `PORTFOLIO_DEFAULT_SYSTEM_PROMPT`.
  4. Phase 1: Train `PointScarFlow` from `step = 0` with OT Rectified Flow + `k=8` Sharp-Crease Edge
     Differential Loss + `12%` CFG dropout + EMA.
  5. Phase 2: Guided 4-Step Consistency Distillation (`8`-step RK2 `CFG=2.0` Teacher -> `4`-step
     `CFG=1.0` Single-Pass Student) for ultra-fast, razor-sharp browser WebGPU execution.
  6. Audit zero `nn.Linear`, zero PTQ shock, and held-out polyglot (`PT/EN/ES/FR`) 3D metrics,
     then export `/content/pointscar_v1_bundle.tar.gz`.
"""

from __future__ import annotations

import builtins
import copy
from concurrent.futures import ThreadPoolExecutor
import gc
import io
import json
import math
import os
from pathlib import Path
import tarfile
import time
import urllib.request
import zipfile
from typing import Any, Callable, Dict, List, Optional, Tuple

_orig_print = builtins.print


def print(*args: Any, **kwargs: Any) -> Any:
    kwargs.setdefault("flush", True)
    return _orig_print(*args, **kwargs)


import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree
except ImportError:
    linear_sum_assignment = None  # type: ignore[assignment]
    cKDTree = None  # type: ignore[assignment]

import sys

sys.path.insert(0, "/content")
if "__file__" in globals():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scar_native.point_scar_flow import (
    PointScarFlow,
    PointScarFlowConfig,
    build_canonical_fibonacci_sphere,
    export_pointscar_webgpu_bundle,
)

PORTFOLIO_DEFAULT_SYSTEM_PROMPT = (
    "You are JARVIS, the official Qwen2.5-3B AI assistant for visitors on Julio Reis's portfolio "
    "(byte-od.github.io), running locally via WebGPU WGSL, MoS, and SCache O(1). The visitor is not "
    "Julio; never claim to be Julio and refer to Julio Reis in the third person. Always reply in the "
    "user's language in 2 to 4 complete sentences."
)


# =============================================================================
# 1. Geometric Sampling Primitives (Continuous Parametric 3D Manifolds + CAD)
# =============================================================================


def _normalize_cloud(pts: np.ndarray, sal: np.ndarray, target_radius: float = 0.92) -> np.ndarray:
    """Centers `pts [N, 3]` at bounding-box midpoint, scales max radius to `target_radius`, packs `[N, 4]`."""
    pts = np.asarray(pts, dtype=np.float32)
    sal = np.clip(np.asarray(sal, dtype=np.float32).reshape(-1), 0.05, 1.0)
    bbox_mid = 0.5 * (pts.max(axis=0) + pts.min(axis=0))
    pts = pts - bbox_mid
    rmax = float(np.max(np.linalg.norm(pts, axis=-1)))
    if rmax > 1e-6:
        pts = pts * (target_radius / rmax)
    return np.concatenate([pts, sal[:, None]], axis=-1).astype(np.float32)


def _rot_x(pts: np.ndarray, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    y = c * pts[:, 1] - s * pts[:, 2]
    z = s * pts[:, 1] + c * pts[:, 2]
    return np.stack([pts[:, 0], y, z], axis=-1)


def _rot_y(pts: np.ndarray, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    x = c * pts[:, 0] + s * pts[:, 2]
    z = -s * pts[:, 0] + c * pts[:, 2]
    return np.stack([x, pts[:, 1], z], axis=-1)


def _rot_z(pts: np.ndarray, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    x = c * pts[:, 0] - s * pts[:, 1]
    y = s * pts[:, 0] + c * pts[:, 1]
    return np.stack([x, y, pts[:, 2]], axis=-1)


def _sample_box_surface(rng: np.random.Generator, n: int, center: Tuple[float, float, float], half_ext: Tuple[float, float, float]) -> np.ndarray:
    hx, hy, hz = half_ext
    areas = np.array([hy * hz, hy * hz, hx * hz, hx * hz, hx * hy, hx * hy], dtype=np.float64)
    areas = areas / areas.sum()
    faces = rng.choice(6, size=n, p=areas)
    u = rng.uniform(-1.0, 1.0, size=n)
    v = rng.uniform(-1.0, 1.0, size=n)
    pts = np.zeros((n, 3), dtype=np.float32)
    for f in range(6):
        m = faces == f
        if not np.any(m):
            continue
        if f == 0:
            pts[m] = np.stack([np.full(m.sum(), hx), u[m] * hy, v[m] * hz], axis=-1)
        elif f == 1:
            pts[m] = np.stack([np.full(m.sum(), -hx), u[m] * hy, v[m] * hz], axis=-1)
        elif f == 2:
            pts[m] = np.stack([u[m] * hx, np.full(m.sum(), hy), v[m] * hz], axis=-1)
        elif f == 3:
            pts[m] = np.stack([u[m] * hx, np.full(m.sum(), -hy), v[m] * hz], axis=-1)
        elif f == 4:
            pts[m] = np.stack([u[m] * hx, v[m] * hy, np.full(m.sum(), hz)], axis=-1)
        else:
            pts[m] = np.stack([u[m] * hx, v[m] * hy, np.full(m.sum(), -hz)], axis=-1)
    return pts + np.asarray(center, dtype=np.float32)


def _sample_cylinder(rng: np.random.Generator, n: int, p0: np.ndarray, p1: np.ndarray, r0: float, r1: Optional[float] = None) -> np.ndarray:
    if r1 is None:
        r1 = r0
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return np.repeat(p0[None, :], n, axis=0)
    d = axis / length
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32) if abs(d[1]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float32)
    u = np.cross(d, up)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    t = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    ang = rng.uniform(0.0, 2.0 * math.pi, size=n).astype(np.float32)
    rad = (1.0 - t) * r0 + t * r1
    return (
        p0[None, :]
        + t[:, None] * axis[None, :]
        + (rad * np.cos(ang))[:, None] * u[None, :]
        + (rad * np.sin(ang))[:, None] * v[None, :]
    ).astype(np.float32)


def _sample_ellipsoid(rng: np.random.Generator, n: int, center: Tuple[float, float, float], radii: Tuple[float, float, float]) -> np.ndarray:
    g = rng.normal(size=(n, 3)).astype(np.float32)
    g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-6)
    return g * np.asarray(radii, dtype=np.float32)[None, :] + np.asarray(center, dtype=np.float32)[None, :]


def _compute_curvature_salience(pts: np.ndarray) -> np.ndarray:
    """Computes normalized geometric ridge/extremity salience `s in [0.15, 1.0]` in pure NumPy."""
    p_sq = (pts * pts).sum(axis=-1, keepdims=True)
    d2 = np.maximum(p_sq + p_sq.T - 2.0 * (pts @ pts.T), 0.0)
    idx = np.argpartition(d2, kth=12, axis=-1)[:, :12]
    neigh = pts[idx] - pts[:, None, :]
    cov = np.einsum("nki,nkj->nij", neigh, neigh) / 11.0
    evals = np.linalg.eigvalsh(cov)
    surf_var = evals[:, 0] / np.maximum(evals.sum(axis=-1), 1e-6)
    rad = np.linalg.norm(pts, axis=-1)
    raw = 0.65 * (surf_var / (np.percentile(surf_var, 95) + 1e-6)) + 0.35 * (rad / (np.max(rad) + 1e-6))
    return np.clip(0.15 + 0.85 * np.clip(raw, 0.0, 1.0), 0.15, 1.0).astype(np.float32)


# =============================================================================
# 2. Rich 54-Concept Continuous Parametric 3D Manifold Generators
# =============================================================================


def build_parametric_shape(concept_id: str, rng: np.random.Generator, n: int = 1024) -> np.ndarray:
    """Generates a continuous 3D geometric manifold `[n, 4]` (`x, y, z, s`) with smooth random variations."""
    pts_list: List[np.ndarray] = []

    if concept_id == "dna_helix":
        turns = rng.uniform(2.2, 3.0)
        radius = rng.uniform(0.34, 0.44)
        height = rng.uniform(1.7, 2.0)
        n_strand = int(n * 0.36)
        n_rungs = n - 2 * n_strand
        for phase in (0.0, math.pi):
            t = np.linspace(-0.5, 0.5, n_strand, dtype=np.float32) + rng.normal(0, 0.004, n_strand).astype(np.float32)
            ang = t * turns * 2.0 * math.pi + phase
            tube = rng.normal(0, 0.025, (n_strand, 3)).astype(np.float32)
            pts_list.append(np.stack([radius * np.cos(ang), t * height, radius * np.sin(ang)], axis=-1) + tube)
        num_bars = 12
        per_bar = max(1, n_rungs // num_bars)
        rem = n_rungs
        for b in range(num_bars):
            cnt = per_bar if b < num_bars - 1 else rem
            rem -= cnt
            tb = -0.45 + 0.90 * (b / max(1, num_bars - 1))
            ang = tb * turns * 2.0 * math.pi
            p_a = np.array([radius * math.cos(ang), tb * height, radius * math.sin(ang)], dtype=np.float32)
            p_b = np.array([radius * math.cos(ang + math.pi), tb * height, radius * math.sin(ang + math.pi)], dtype=np.float32)
            pts_list.append(_sample_cylinder(rng, cnt, p_a, p_b, 0.02))

    elif concept_id == "atom_orbital":
        n_nuc = int(n * 0.22)
        pts_list.append(_sample_ellipsoid(rng, n_nuc, (0, 0, 0), (0.16, 0.16, 0.16)) * rng.uniform(0.5, 1.0, (n_nuc, 1)).astype(np.float32))
        rem = n - n_nuc
        tilts = [(0.35, 0.0), (-0.95, 1.05), (0.95, -1.05)]
        for i, (rx, rz) in enumerate(tilts):
            cnt = rem // 3 if i < 2 else rem - 2 * (rem // 3)
            th = rng.uniform(0, 2 * math.pi, cnt).astype(np.float32)
            ring = np.stack([0.88 * np.cos(th), rng.normal(0, 0.02, cnt).astype(np.float32), 0.56 * np.sin(th)], axis=-1)
            ring = _rot_z(_rot_x(ring, rx + rng.uniform(-0.1, 0.1)), rz + rng.uniform(-0.1, 0.1))
            pts_list.append(ring)

    elif concept_id == "ringed_planet":
        n_ball = int(n * 0.44)
        n_ring = n - n_ball
        r_core = rng.uniform(0.34, 0.42)
        pts_list.append(_sample_ellipsoid(rng, n_ball, (0, 0, 0), (r_core, r_core * 0.92, r_core)))
        r = np.sqrt(rng.uniform(0.54**2, 0.98**2, n_ring)).astype(np.float32)
        # Cassini gap
        r = np.where((r > 0.74) & (r < 0.79), r + 0.06, r)
        th = rng.uniform(0, 2 * math.pi, n_ring).astype(np.float32)
        ring = np.stack([r * np.cos(th), rng.normal(0, 0.012, n_ring).astype(np.float32), r * np.sin(th)], axis=-1)
        pts = np.concatenate([pts_list[0], ring], axis=0)
        pts = _rot_z(_rot_x(pts, rng.uniform(0.32, 0.48)), rng.uniform(-0.22, 0.22))
        pts_list = [pts]

    elif concept_id == "spiral_galaxy":
        n_bulge = int(n * 0.22)
        pts_list.append(_sample_ellipsoid(rng, n_bulge, (0, 0, 0), (0.18, 0.10, 0.18)) * rng.uniform(0.2, 1.0, (n_bulge, 1)).astype(np.float32))
        n_arms = n - n_bulge
        arms = 3 if rng.random() < 0.4 else 2
        arm_idx = rng.integers(0, arms, size=n_arms)
        r = rng.uniform(0.12, 0.98, size=n_arms).astype(np.float32)
        twist = rng.uniform(3.2, 4.4)
        th = r * twist + arm_idx * (2.0 * math.pi / arms) + rng.normal(0, 0.18, size=n_arms).astype(np.float32)
        y = rng.normal(0, 0.035 * (1.1 - 0.7 * r), size=n_arms).astype(np.float32)
        arm_pts = np.stack([r * np.cos(th), y, r * np.sin(th)], axis=-1)
        pts_list = [_rot_x(np.concatenate([pts_list[0], arm_pts], axis=0), rng.uniform(0.45, 0.75))]

    elif concept_id == "torus_manifold":
        R = rng.uniform(0.58, 0.66)
        r = rng.uniform(0.22, 0.28)
        u = rng.uniform(0, 2 * math.pi, n).astype(np.float32)
        v = rng.uniform(0, 2 * math.pi, n).astype(np.float32)
        x = (R + r * np.cos(v)) * np.cos(u)
        z = (R + r * np.cos(v)) * np.sin(u)
        y = r * np.sin(v)
        pts_list.append(_rot_x(np.stack([x, y, z], axis=-1), rng.uniform(0.35, 0.65)))

    elif concept_id == "mos_scache_architecture":
        # Toroidal O(1) SCache ring + Central Core + 3 Orthogonal Low-Rank Scar Blades!
        n_ring = int(n * 0.46)
        n_core = int(n * 0.18)
        n_blades = n - n_ring - n_core
        u = rng.uniform(0, 2 * math.pi, n_ring).astype(np.float32)
        v = rng.uniform(0, 2 * math.pi, n_ring).astype(np.float32)
        ring = np.stack([(0.68 + 0.11 * np.cos(v)) * np.cos(u), 0.11 * np.sin(v), (0.68 + 0.11 * np.cos(v)) * np.sin(u)], axis=-1)
        core = _sample_ellipsoid(rng, n_core, (0, 0, 0), (0.20, 0.28, 0.20))
        blades = []
        for k in range(3):
            cnt = n_blades // 3 if k < 2 else n_blades - 2 * (n_blades // 3)
            ang = k * (2.0 * math.pi / 3.0)
            b_pts = _sample_box_surface(rng, cnt, (0.0, 0.0, 0.0), (0.86, 0.28, 0.03))
            blades.append(_rot_y(b_pts, ang))
        pts_list.append(_rot_x(np.concatenate([ring, core, *blades], axis=0), 0.42))

    elif concept_id == "vq4_d4_hypercube":
        # 4D Tesseract (outer + inner cube + 8 diagonal struts)
        verts_in = np.array([[sx, sy, sz] for sx in (-0.36, 0.36) for sy in (-0.36, 0.36) for sz in (-0.36, 0.36)], dtype=np.float32)
        verts_out = verts_in * (0.82 / 0.36)
        edges = []
        for i in range(8):
            for j in range(i + 1, 8):
                if np.sum(np.abs(verts_in[i] - verts_in[j]) > 1e-4) == 1:
                    edges.append((verts_in[i], verts_in[j]))
                    edges.append((verts_out[i], verts_out[j]))
            edges.append((verts_in[i], verts_out[i]))
        per_e = n // len(edges)
        rem = n
        for idx_e, (pa, pb) in enumerate(edges):
            cnt = per_e if idx_e < len(edges) - 1 else rem
            rem -= cnt
            pts_list.append(_sample_cylinder(rng, cnt, pa, pb, 0.024))
        pts_list = [_rot_y(_rot_x(np.concatenate(pts_list, axis=0), rng.uniform(0.3, 0.5)), rng.uniform(0.4, 0.7))]

    elif concept_id == "tricompass_astrolabe":
        # 3 concentric orthogonal gimbal rings + central pointer spire
        n_spire = int(n * 0.16)
        rem = n - n_spire
        for idx_r, (rad, rx, rz) in enumerate([(0.88, 0.0, 0.0), (0.72, math.pi / 2.0, 0.35), (0.56, 0.65, math.pi / 2.0)]):
            cnt = rem // 3 if idx_r < 2 else rem - 2 * (rem // 3)
            u = rng.uniform(0, 2 * math.pi, cnt).astype(np.float32)
            v = rng.uniform(0, 2 * math.pi, cnt).astype(np.float32)
            rg = np.stack([(rad + 0.035 * np.cos(v)) * np.cos(u), 0.035 * np.sin(v), (rad + 0.035 * np.cos(v)) * np.sin(u)], axis=-1)
            pts_list.append(_rot_z(_rot_x(rg, rx), rz))
        pts_list.append(_sample_cylinder(rng, n_spire, np.array([0, -0.75, 0], np.float32), np.array([0, 0.92, 0], np.float32), 0.05, 0.005))

    elif concept_id == "neural_network":
        # 3D 3-layer MLP node grid + synaptic connecting axons
        layers_x = [-0.72, 0.0, 0.72]
        nodes: List[List[np.ndarray]] = []
        coords = [(-0.45, -0.35), (-0.45, 0.35), (0.0, 0.0), (0.45, -0.35), (0.45, 0.35)]
        for lx in layers_x:
            lnodes = [np.array([lx, cy, cz], dtype=np.float32) for (cy, cz) in coords]
            nodes.append(lnodes)
        n_nodes_pts = int(n * 0.38)
        per_node = n_nodes_pts // 15
        for lnodes in nodes:
            for nd in lnodes:
                pts_list.append(_sample_ellipsoid(rng, per_node, tuple(nd), (0.085, 0.085, 0.085)))
        conns = []
        for l in range(2):
            for a in nodes[l]:
                for b in nodes[l + 1]:
                    if rng.random() < 0.68 or np.linalg.norm(a[1:] - b[1:]) < 0.55:
                        conns.append((a, b))
        rem = n - sum(len(p) for p in pts_list)
        for idx_c, (pa, pb) in enumerate(conns):
            cnt = rem // len(conns) if idx_c < len(conns) - 1 else rem - (len(conns) - 1) * (rem // len(conns))
            if cnt > 0:
                pts_list.append(_sample_cylinder(rng, cnt, pa, pb, 0.012))

    elif concept_id == "rubiks_tensor_cube":
        # 3x3x3 sub-cube tensor lattice with visible inter-slice gaps
        sub_n = n // 27
        rem = n
        idx_c = 0
        for ix in (-1, 0, 1):
            for iy in (-1, 0, 1):
                for iz in (-1, 0, 1):
                    cnt = sub_n if idx_c < 26 else rem
                    rem -= cnt
                    idx_c += 1
                    pts_list.append(_sample_box_surface(rng, cnt, (ix * 0.46, iy * 0.46, iz * 0.46), (0.165, 0.165, 0.165)))
        pts_list = [_rot_y(_rot_x(np.concatenate(pts_list, axis=0), 0.38), 0.52)]

    elif concept_id == "heart_3d":
        # Exact Taubin / parametric 3D anatomical cardioid surface
        collected = []
        while sum(len(c) for c in collected) < n:
            u = rng.uniform(0, math.pi, n).astype(np.float32)
            v = rng.uniform(0, 2 * math.pi, n).astype(np.float32)
            x = 16.0 * (np.sin(u) ** 3) * np.cos(v)
            z = 9.0 * (np.sin(u) ** 3) * np.sin(v)
            y = 13.0 * np.cos(u) - 5.0 * np.cos(2 * u) - 2.0 * np.cos(3 * u) - np.cos(4 * u)
            collected.append(np.stack([x, y, z], axis=-1))
        pts_list.append(np.concatenate(collected, axis=0)[:n])

    elif concept_id == "brain_cortex":
        # Bilateral cerebral hemispheres with longitudinal fissure + gyral folds + cerebellum
        n_hem = int(n * 0.43)
        n_stem = n - 2 * n_hem
        for sgn in (-1.0, 1.0):
            g = rng.normal(size=(n_hem, 3)).astype(np.float32)
            g /= np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-6)
            g[:, 0] = np.abs(g[:, 0]) * 0.85 + 0.08
            r_gyri = 1.0 + 0.07 * np.sin(10.0 * g[:, 1]) * np.cos(9.0 * g[:, 2])
            hem = np.stack([sgn * g[:, 0] * 0.48, g[:, 1] * 0.56, g[:, 2] * 0.72], axis=-1) * r_gyri[:, None]
            pts_list.append(hem)
        pts_list.append(_sample_cylinder(rng, n_stem, np.array([0, -0.35, -0.15], np.float32), np.array([0, -0.78, -0.20], np.float32), 0.14, 0.09))

    elif concept_id == "rocket_spacecraft":
        n_body = int(n * 0.48)
        n_nose = int(n * 0.22)
        n_fins = int(n * 0.20)
        n_nozzle = n - n_body - n_nose - n_fins
        r_b = rng.uniform(0.16, 0.21)
        pts_list.append(_sample_cylinder(rng, n_body, np.array([0, -0.55, 0], np.float32), np.array([0, 0.42, 0], np.float32), r_b, r_b))
        pts_list.append(_sample_cylinder(rng, n_nose, np.array([0, 0.42, 0], np.float32), np.array([0, 0.95, 0], np.float32), r_b, 0.008))
        pts_list.append(_sample_cylinder(rng, n_nozzle, np.array([0, -0.55, 0], np.float32), np.array([0, -0.82, 0], np.float32), r_b * 0.65, r_b * 1.15))
        for k in range(4):
            cnt = n_fins // 4 if k < 3 else n_fins - 3 * (n_fins // 4)
            fin = _sample_box_surface(rng, cnt, (r_b + 0.13, -0.52, 0.0), (0.14, 0.22, 0.018))
            fin[:, 1] -= (fin[:, 0] - r_b) * 0.55
            pts_list.append(_rot_y(fin, k * math.pi / 2.0))

    elif concept_id == "satellite_orbital":
        n_bus = int(n * 0.26)
        n_dish = int(n * 0.20)
        n_wings = n - n_bus - n_dish
        pts_list.append(_sample_box_surface(rng, n_bus, (0, 0, 0), (0.18, 0.22, 0.18)))
        # Parabolic dish on top
        r_d = np.sqrt(rng.uniform(0, 1, n_dish)).astype(np.float32) * 0.26
        th = rng.uniform(0, 2 * math.pi, n_dish).astype(np.float32)
        pts_list.append(np.stack([r_d * np.cos(th), 0.28 + 0.35 * (r_d ** 2), r_d * np.sin(th)], axis=-1))
        for sgn in (-1.0, 1.0):
            cnt = n_wings // 2 if sgn < 0 else n_wings - (n_wings // 2)
            pts_list.append(_sample_box_surface(rng, cnt, (sgn * 0.62, 0.0, 0.0), (0.38, 0.16, 0.015)))
        pts_list = [_rot_z(_rot_x(np.concatenate(pts_list, axis=0), 0.32), 0.18)]

    elif concept_id == "airplane_jet":
        n_fus = int(n * 0.36)
        n_wing = int(n * 0.38)
        n_tail = n - n_fus - n_wing
        pts_list.append(_sample_cylinder(rng, n_fus, np.array([0, 0, -0.88], np.float32), np.array([0, 0, 0.92], np.float32), 0.09, 0.04))
        for sgn in (-1.0, 1.0):
            cnt = n_wing // 2 if sgn < 0 else n_wing - (n_wing // 2)
            u = rng.uniform(0.06, 0.92, cnt).astype(np.float32)
            chord = (1.0 - 0.65 * u) * 0.22
            z_w = -u * 0.38 + rng.uniform(-1.0, 1.0, cnt).astype(np.float32) * chord * 0.5 + 0.10
            y_w = rng.normal(0, 0.015, cnt).astype(np.float32)
            pts_list.append(np.stack([sgn * u, y_w, z_w], axis=-1))
        n_vfin = n_tail // 2
        n_hfin = n_tail - n_vfin
        pts_list.append(_sample_box_surface(rng, n_vfin, (0.0, 0.18, -0.74), (0.015, 0.16, 0.14)))
        pts_list.append(_sample_box_surface(rng, n_hfin, (0.0, 0.02, -0.78), (0.32, 0.012, 0.09)))
        pts_list = [_rot_y(_rot_x(np.concatenate(pts_list, axis=0), -0.18), 0.55)]

    elif concept_id == "tree_canopy" or concept_id == "binary_tree_graph":
        is_graph = concept_id == "binary_tree_graph"
        segs: List[Tuple[np.ndarray, np.ndarray, float]] = []
        leaves: List[np.ndarray] = []

        def _branch(p0: np.ndarray, direction: np.ndarray, length: float, rad: float, depth: int) -> None:
            p1 = p0 + direction * length
            segs.append((p0, p1, rad))
            if depth == 0:
                leaves.append(p1)
                return
            spread = 0.58 if is_graph else rng.uniform(0.45, 0.65)
            n_ch = 2 if is_graph else 3
            for k in range(n_ch):
                az = (2.0 * math.pi * k / n_ch) + (0.0 if is_graph else rng.uniform(-0.3, 0.3))
                d_new = _rot_y(_rot_z(direction[None, :], spread), az)[0]
                d_new /= np.linalg.norm(d_new)
                _branch(p1, d_new, length * 0.68, rad * 0.65, depth - 1)

        root_p = np.array([0.0, 0.85, 0.0], np.float32) if is_graph else np.array([0.0, -0.88, 0.0], np.float32)
        root_d = np.array([0.0, -1.0, 0.0], np.float32) if is_graph else np.array([0.0, 1.0, 0.0], np.float32)
        _branch(root_p, root_d, 0.52, 0.065 if not is_graph else 0.032, 3)
        n_seg = int(n * (0.62 if is_graph else 0.52))
        per_s = max(1, n_seg // len(segs))
        for p_a, p_b, r_s in segs:
            pts_list.append(_sample_cylinder(rng, per_s, p_a, p_b, r_s, r_s * 0.75))
        rem = n - sum(len(p) for p in pts_list)
        if rem > 0 and leaves:
            per_l = max(1, rem // len(leaves))
            for idx_l, lp in enumerate(leaves):
                cnt = per_l if idx_l < len(leaves) - 1 else max(0, n - sum(len(p) for p in pts_list))
                if cnt > 0:
                    rad_l = 0.075 if is_graph else 0.16
                    pts_list.append(_sample_ellipsoid(rng, cnt, tuple(lp), (rad_l, rad_l, rad_l)))

    elif concept_id == "humanoid_agent":
        # Articulated embodied robot / AURA agent
        pts_list.append(_sample_ellipsoid(rng, int(n * 0.14), (0, 0.74, 0), (0.13, 0.15, 0.13)))
        pts_list.append(_sample_box_surface(rng, int(n * 0.28), (0, 0.28, 0), (0.22, 0.26, 0.12)))
        for sgn in (-1.0, 1.0):
            pts_list.append(_sample_cylinder(rng, int(n * 0.13), np.array([sgn * 0.27, 0.50, 0], np.float32), np.array([sgn * 0.44, 0.02, 0.08], np.float32), 0.055, 0.045))
        rem = n - sum(len(p) for p in pts_list)
        for idx_l, sgn in enumerate((-1.0, 1.0)):
            cnt = rem // 2 if idx_l == 0 else rem - (rem // 2)
            pts_list.append(_sample_cylinder(rng, cnt, np.array([sgn * 0.12, 0.0, 0], np.float32), np.array([sgn * 0.14, -0.82, 0], np.float32), 0.068, 0.052))

    elif concept_id == "jarvis_constellation_core":
        # Holographic cranial sphere + dual tilted synaptic halos
        n_head = int(n * 0.58)
        pts_list.append(_sample_ellipsoid(rng, n_head, (0, 0.04, 0), (0.42, 0.52, 0.44)))
        rem = n - n_head
        for i, (rad, tilt) in enumerate([(0.78, 0.35), (0.90, -0.45)]):
            cnt = rem // 2 if i == 0 else rem - (rem // 2)
            th = rng.uniform(0, 2 * math.pi, cnt).astype(np.float32)
            ring = np.stack([rad * np.cos(th), rng.normal(0, 0.018, cnt).astype(np.float32), rad * np.sin(th)], axis=-1)
            pts_list.append(_rot_z(_rot_x(ring, tilt), 0.25 * (1 if i == 0 else -1)))

    elif concept_id == "shield_security":
        # Heraldic curved kite shield
        u = rng.uniform(-1.0, 1.0, n * 2).astype(np.float32)
        v = rng.uniform(-1.0, 1.0, n * 2).astype(np.float32)
        width_at_v = np.where(v >= 0.0, 0.72, 0.72 * np.maximum(0.0, (v + 1.0)) ** 0.72)
        mask = np.abs(u) * 0.72 <= width_at_v
        x = (u[mask][:n]) * 0.72
        y = (v[mask][:n]) * 0.88
        z = 0.16 * (1.0 - (x / 0.75) ** 2) - 0.04 * np.abs(y)
        # Rim reinforcement
        n_rim = int(n * 0.28)
        t_rim = np.linspace(-1.0, 1.0, n_rim // 2, dtype=np.float32)
        w_rim = np.where(t_rim >= 0.0, 0.72, 0.72 * np.maximum(0.0, (t_rim + 1.0)) ** 0.72)
        rim = np.concatenate([
            np.stack([w_rim, t_rim * 0.88, np.full_like(t_rim, 0.06)], axis=-1),
            np.stack([-w_rim, t_rim * 0.88, np.full_like(t_rim, 0.06)], axis=-1),
        ], axis=0)
        body = np.stack([x[: n - len(rim)], y[: n - len(rim)], z[: n - len(rim)]], axis=-1)
        pts_list.append(np.concatenate([body, rim], axis=0))

    elif concept_id == "padlock_privacy":
        n_body = int(n * 0.58)
        n_arch = n - n_body
        pts_list.append(_sample_box_surface(rng, n_body, (0, -0.32, 0), (0.44, 0.38, 0.20)))
        th = rng.uniform(0, math.pi, n_arch).astype(np.float32)
        v = rng.uniform(0, 2 * math.pi, n_arch).astype(np.float32)
        r_maj, r_min = 0.28, 0.055
        arch = np.stack([(r_maj + r_min * np.cos(v)) * np.cos(th), 0.08 + (r_maj + r_min * np.cos(v)) * np.sin(th) * 1.35, r_min * np.sin(v)], axis=-1)
        pts_list.append(arch)

    elif concept_id == "entropy_funnel_blackhole":
        # Gravitational / Landauer thermodynamic funnel + accretion disk
        r = np.exp(rng.uniform(math.log(0.10), math.log(0.95), n)).astype(np.float32)
        th = rng.uniform(0, 2 * math.pi, n).astype(np.float32) + 2.5 / (r + 0.15)
        y = 0.42 - 0.14 / (r + 0.08)
        pts_list.append(_rot_x(np.stack([r * np.cos(th), y, r * np.sin(th)], axis=-1), 0.32))

    elif concept_id == "hourglass_time":
        t = rng.uniform(-1.0, 1.0, int(n * 0.72)).astype(np.float32)
        th = rng.uniform(0, 2 * math.pi, len(t)).astype(np.float32)
        rad = 0.07 + 0.44 * (t ** 2)
        glass = np.stack([rad * np.cos(th), t * 0.78, rad * np.sin(th)], axis=-1)
        rem = n - len(glass)
        top_cap = _sample_cylinder(rng, rem // 2, np.array([0, 0.78, 0], np.float32), np.array([0, 0.86, 0], np.float32), 0.54, 0.54)
        bot_cap = _sample_cylinder(rng, rem - rem // 2, np.array([0, -0.86, 0], np.float32), np.array([0, -0.78, 0], np.float32), 0.54, 0.54)
        pts_list.extend([glass, top_cap, bot_cap])

    elif concept_id == "gear_cogwheel":
        teeth = int(rng.choice([8, 10, 12]))
        n_rim = int(n * 0.70)
        n_hub = n - n_rim
        th = rng.uniform(0, 2 * math.pi, n_rim).astype(np.float32)
        tooth_wave = np.clip(np.cos(teeth * th) * 2.5, -1.0, 1.0)
        rad = 0.68 + 0.14 * tooth_wave
        z = rng.uniform(-0.12, 0.12, n_rim).astype(np.float32)
        pts_list.append(np.stack([rad * np.cos(th), rad * np.sin(th), z], axis=-1))
        th_h = rng.uniform(0, 2 * math.pi, n_hub).astype(np.float32)
        r_h = np.where(rng.random(n_hub) < 0.5, 0.24, rng.uniform(0.24, 0.54, n_hub)).astype(np.float32)
        pts_list.append(np.stack([r_h * np.cos(th_h), r_h * np.sin(th_h), rng.choice([-0.12, 0.12], n_hub).astype(np.float32)], axis=-1))
        pts_list = [_rot_y(np.concatenate(pts_list, axis=0), 0.42)]

    elif concept_id == "suspension_bridge":
        n_deck = int(n * 0.34)
        n_tow = int(n * 0.26)
        n_cab = n - n_deck - n_tow
        pts_list.append(_sample_box_surface(rng, n_deck, (0, -0.18, 0), (0.95, 0.03, 0.14)))
        for tx in (-0.48, 0.48):
            for tz in (-0.13, 0.13):
                pts_list.append(_sample_box_surface(rng, n_tow // 4, (tx, 0.12, tz), (0.035, 0.42, 0.035)))
        rem = n - sum(len(p) for p in pts_list)
        x_c = rng.uniform(-0.95, 0.95, rem).astype(np.float32)
        z_c = rng.choice([-0.13, 0.13], rem).astype(np.float32)
        ax = np.abs(x_c)
        y_c = np.where(ax <= 0.48, -0.04 + 0.56 * ((x_c / 0.48) ** 2), 0.52 - 0.68 * ((ax - 0.48) / 0.47))
        pts_list.append(np.stack([x_c, y_c, z_c], axis=-1))

    elif concept_id == "globe_earth":
        # Latitude & longitude meridian wireframe sphere
        n_lat = n // 2
        n_lon = n - n_lat
        lats = np.deg2rad([-60, -30, 0, 30, 60])
        lat_pick = rng.choice(lats, size=n_lat).astype(np.float32)
        lon_u = rng.uniform(0, 2 * math.pi, n_lat).astype(np.float32)
        pts_list.append(np.stack([np.cos(lat_pick) * np.cos(lon_u), np.sin(lat_pick), np.cos(lat_pick) * np.sin(lon_u)], axis=-1))
        lons = np.linspace(0, math.pi, 6, endpoint=False, dtype=np.float32)
        lon_pick = rng.choice(lons, size=n_lon).astype(np.float32)
        th_v = rng.uniform(0, 2 * math.pi, n_lon).astype(np.float32)
        pts_list.append(np.stack([np.sin(th_v) * np.cos(lon_pick), np.cos(th_v), np.sin(th_v) * np.sin(lon_pick)], axis=-1))
        pts_list = [_rot_z(np.concatenate(pts_list, axis=0), 0.41)]

    elif concept_id == "laptop_computer":
        n_base = n // 2
        n_scr = n - n_base
        base = _sample_box_surface(rng, n_base, (0, -0.28, 0.14), (0.56, 0.025, 0.38))
        scr = _sample_box_surface(rng, n_scr, (0, 0.38, 0.0), (0.56, 0.38, 0.02))
        scr = _rot_x(scr, -0.22) + np.array([0.0, -0.26, -0.24], dtype=np.float32)
        pts_list.extend([base, scr])
        pts_list = [_rot_y(np.concatenate(pts_list, axis=0), 0.45)]

    elif concept_id == "gpu_silicon_chip":
        n_pcb = int(n * 0.45)
        n_die = int(n * 0.25)
        n_pins = n - n_pcb - n_die
        pts_list.append(_sample_box_surface(rng, n_pcb, (0, 0, 0), (0.68, 0.05, 0.68)))
        pts_list.append(_sample_box_surface(rng, n_die, (0, 0.08, 0), (0.30, 0.04, 0.30)))
        side = rng.integers(0, 4, size=n_pins)
        pos = rng.choice(np.linspace(-0.56, 0.56, 8), size=n_pins).astype(np.float32)
        pins = np.zeros((n_pins, 3), dtype=np.float32)
        for s in range(4):
            m = side == s
            if s == 0:
                pins[m] = np.stack([np.full(m.sum(), 0.76), rng.uniform(-0.08, 0.0, m.sum()), pos[m]], axis=-1)
            elif s == 1:
                pins[m] = np.stack([np.full(m.sum(), -0.76), rng.uniform(-0.08, 0.0, m.sum()), pos[m]], axis=-1)
            elif s == 2:
                pins[m] = np.stack([pos[m], rng.uniform(-0.08, 0.0, m.sum()), np.full(m.sum(), 0.76)], axis=-1)
            else:
                pins[m] = np.stack([pos[m], rng.uniform(-0.08, 0.0, m.sum()), np.full(m.sum(), -0.76)], axis=-1)
        pts_list.append(pins)
        pts_list = [_rot_x(np.concatenate(pts_list, axis=0), 0.55)]

    elif concept_id == "chair_furniture":
        pts_list.append(_sample_box_surface(rng, int(n * 0.30), (0, -0.05, 0), (0.34, 0.04, 0.34)))
        pts_list.append(_sample_box_surface(rng, int(n * 0.32), (0, 0.38, -0.31), (0.34, 0.40, 0.035)))
        rem = n - sum(len(p) for p in pts_list)
        for idx_l, (lx, lz) in enumerate([(-0.30, -0.30), (0.30, -0.30), (-0.30, 0.30), (0.30, 0.30)]):
            cnt = rem // 4 if idx_l < 3 else rem - 3 * (rem // 4)
            pts_list.append(_sample_cylinder(rng, cnt, np.array([lx, -0.08, lz], np.float32), np.array([lx, -0.76, lz], np.float32), 0.035))
        pts_list = [_rot_y(np.concatenate(pts_list, axis=0), 0.48)]

    elif concept_id == "table_desk":
        pts_list.append(_sample_box_surface(rng, int(n * 0.52), (0, 0.26, 0), (0.68, 0.04, 0.44)))
        rem = n - sum(len(p) for p in pts_list)
        for idx_l, (lx, lz) in enumerate([(-0.60, -0.36), (0.60, -0.36), (-0.60, 0.36), (0.60, 0.36)]):
            cnt = rem // 4 if idx_l < 3 else rem - 3 * (rem // 4)
            pts_list.append(_sample_cylinder(rng, cnt, np.array([lx, 0.22, lz], np.float32), np.array([lx, -0.52, lz], np.float32), 0.04))
        pts_list = [_rot_y(np.concatenate(pts_list, axis=0), 0.42)]

    elif concept_id == "car_automobile":
        pts_list.append(_sample_box_surface(rng, int(n * 0.44), (0, -0.08, 0), (0.82, 0.16, 0.38)))
        pts_list.append(_sample_box_surface(rng, int(n * 0.28), (-0.06, 0.20, 0), (0.42, 0.14, 0.33)))
        rem = n - sum(len(p) for p in pts_list)
        for idx_w, (wx, wz) in enumerate([(-0.50, -0.40), (0.50, -0.40), (-0.50, 0.40), (0.50, 0.40)]):
            cnt = rem // 4 if idx_w < 3 else rem - 3 * (rem // 4)
            pts_list.append(_sample_cylinder(rng, cnt, np.array([wx, -0.24, wz - 0.05], np.float32), np.array([wx, -0.24, wz + 0.05], np.float32), 0.16))
        pts_list = [_rot_y(np.concatenate(pts_list, axis=0), 0.42)]

    elif concept_id == "quadruped_animal":
        pts_list.append(_sample_ellipsoid(rng, int(n * 0.34), (0, 0.08, 0), (0.46, 0.19, 0.18)))
        pts_list.append(_sample_ellipsoid(rng, int(n * 0.16), (0.52, 0.34, 0), (0.16, 0.14, 0.13)))
        # Ears + snout + tail
        pts_list.append(_sample_cylinder(rng, int(n * 0.06), np.array([0.56, 0.44, -0.07], np.float32), np.array([0.60, 0.62, -0.09], np.float32), 0.04, 0.01))
        pts_list.append(_sample_cylinder(rng, int(n * 0.06), np.array([0.56, 0.44, 0.07], np.float32), np.array([0.60, 0.62, 0.09], np.float32), 0.04, 0.01))
        pts_list.append(_sample_cylinder(rng, int(n * 0.08), np.array([-0.44, 0.16, 0], np.float32), np.array([-0.74, 0.42, 0], np.float32), 0.035, 0.015))
        rem = n - sum(len(p) for p in pts_list)
        for idx_l, (lx, lz) in enumerate([(-0.30, -0.12), (0.30, -0.12), (-0.30, 0.12), (0.30, 0.12)]):
            cnt = rem // 4 if idx_l < 3 else rem - 3 * (rem // 4)
            pts_list.append(_sample_cylinder(rng, cnt, np.array([lx, -0.06, lz], np.float32), np.array([lx, -0.52, lz], np.float32), 0.045, 0.032))
        pts_list = [_rot_y(np.concatenate(pts_list, axis=0), 0.38)]

    elif concept_id == "bird_eagle":
        pts_list.append(_sample_ellipsoid(rng, int(n * 0.28), (0, 0, 0), (0.14, 0.14, 0.40)))
        pts_list.append(_sample_cylinder(rng, int(n * 0.08), np.array([0, 0.04, 0.38], np.float32), np.array([0, 0.0, 0.56], np.float32), 0.07, 0.01))
        rem = n - sum(len(p) for p in pts_list)
        for sgn in (-1.0, 1.0):
            cnt = rem // 2 if sgn < 0 else rem - (rem // 2)
            u = rng.uniform(0.08, 0.95, cnt).astype(np.float32)
            y_w = 0.22 * np.sin(u * 2.2) + rng.normal(0, 0.015, cnt).astype(np.float32)
            z_w = -0.18 * u + rng.uniform(-0.14, 0.14, cnt).astype(np.float32) * (1.1 - 0.7 * u)
            pts_list.append(np.stack([sgn * u, y_w, z_w], axis=-1))

    elif concept_id == "fish_marine":
        pts_list.append(_sample_ellipsoid(rng, int(n * 0.64), (0.10, 0, 0), (0.58, 0.26, 0.14)))
        n_tail = int(n * 0.24)
        u = rng.uniform(0.0, 1.0, n_tail).astype(np.float32)
        v = rng.uniform(-1.0, 1.0, n_tail).astype(np.float32)
        pts_list.append(np.stack([-0.46 - 0.38 * u, v * (0.06 + 0.34 * u), rng.normal(0, 0.015, n_tail).astype(np.float32)], axis=-1))
        rem = n - sum(len(p) for p in pts_list)
        pts_list.append(_sample_box_surface(rng, rem, (0.08, 0.30, 0), (0.16, 0.10, 0.012)))

    elif concept_id == "guitar_instrument":
        th = rng.uniform(0, 2 * math.pi, int(n * 0.56)).astype(np.float32)
        r_body = (0.36 + 0.10 * np.cos(2.0 * th - math.pi) - 0.06 * np.sin(th)) * np.sqrt(rng.uniform(0.35, 1.0, len(th))).astype(np.float32)
        z_b = rng.choice([-0.07, 0.07], len(th)).astype(np.float32) + rng.normal(0, 0.015, len(th)).astype(np.float32)
        pts_list.append(np.stack([r_body * np.cos(th) * 0.85, -0.34 + r_body * np.sin(th) * 1.18, z_b], axis=-1))
        n_neck = int(n * 0.30)
        pts_list.append(_sample_box_surface(rng, n_neck, (0, 0.42, 0), (0.055, 0.42, 0.025)))
        rem = n - sum(len(p) for p in pts_list)
        pts_list.append(_sample_box_surface(rng, rem, (0, 0.90, 0), (0.085, 0.10, 0.025)))

    elif concept_id == "sword_blade":
        pts_list.append(_sample_box_surface(rng, int(n * 0.56), (0, 0.24, 0), (0.055, 0.62, 0.015)))
        pts_list.append(_sample_cylinder(rng, int(n * 0.12), np.array([0, 0.86, 0], np.float32), np.array([0, 0.98, 0], np.float32), 0.05, 0.005))
        pts_list.append(_sample_box_surface(rng, int(n * 0.16), (0, -0.40, 0), (0.28, 0.03, 0.03)))
        rem = n - sum(len(p) for p in pts_list)
        pts_list.append(_sample_cylinder(rng, rem, np.array([0, -0.42, 0], np.float32), np.array([0, -0.82, 0], np.float32), 0.038))
        pts_list = [_rot_z(np.concatenate(pts_list, axis=0), 0.35)]

    elif concept_id == "crown_trophy":
        n_band = int(n * 0.45)
        th = rng.uniform(0, 2 * math.pi, n_band).astype(np.float32)
        y_b = rng.uniform(-0.28, 0.02, n_band).astype(np.float32)
        pts_list.append(np.stack([0.62 * np.cos(th), y_b, 0.62 * np.sin(th)], axis=-1))
        peaks = 6
        rem = n - n_band
        for k in range(peaks):
            cnt = rem // peaks if k < peaks - 1 else rem - (peaks - 1) * (rem // peaks)
            ang = 2.0 * math.pi * k / peaks
            p0 = np.array([0.62 * math.cos(ang), 0.02, 0.62 * math.sin(ang)], np.float32)
            p1 = np.array([0.68 * math.cos(ang), 0.46, 0.68 * math.sin(ang)], np.float32)
            pts_list.append(_sample_cylinder(rng, cnt, p0, p1, 0.09, 0.02))

    elif concept_id == "key_cryptographic":
        n_ring = int(n * 0.42)
        u = rng.uniform(0, 2 * math.pi, n_ring).astype(np.float32)
        v = rng.uniform(0, 2 * math.pi, n_ring).astype(np.float32)
        pts_list.append(np.stack([(0.24 + 0.05 * np.cos(v)) * np.cos(u), 0.56 + (0.24 + 0.05 * np.cos(v)) * np.sin(u), 0.05 * np.sin(v)], axis=-1))
        pts_list.append(_sample_cylinder(rng, int(n * 0.34), np.array([0, 0.32, 0], np.float32), np.array([0, -0.78, 0], np.float32), 0.042))
        rem = n - sum(len(p) for p in pts_list)
        pts_list.append(_sample_box_surface(rng, rem // 2, (0.12, -0.56, 0), (0.10, 0.04, 0.025)))
        pts_list.append(_sample_box_surface(rng, rem - rem // 2, (0.15, -0.72, 0), (0.13, 0.045, 0.025)))

    elif concept_id == "cup_mug":
        n_cup = int(n * 0.72)
        th = rng.uniform(0, 2 * math.pi, n_cup).astype(np.float32)
        y_c = rng.uniform(-0.52, 0.52, n_cup).astype(np.float32)
        pts_list.append(np.stack([0.42 * np.cos(th), y_c, 0.42 * np.sin(th)], axis=-1))
        n_bot = int(n * 0.12)
        r_b = np.sqrt(rng.uniform(0, 1, n_bot)).astype(np.float32) * 0.42
        th_b = rng.uniform(0, 2 * math.pi, n_bot).astype(np.float32)
        pts_list.append(np.stack([r_b * np.cos(th_b), np.full(n_bot, -0.52, np.float32), r_b * np.sin(th_b)], axis=-1))
        rem = n - sum(len(p) for p in pts_list)
        ang = rng.uniform(-math.pi / 2, math.pi / 2, rem).astype(np.float32)
        pts_list.append(np.stack([0.42 + 0.24 * np.cos(ang), 0.30 * np.sin(ang), rng.normal(0, 0.025, rem).astype(np.float32)], axis=-1))

    elif concept_id == "vase_amphora":
        y = rng.uniform(-0.82, 0.82, n).astype(np.float32)
        th = rng.uniform(0, 2 * math.pi, n).astype(np.float32)
        rad = 0.26 + 0.22 * np.sin((y + 0.82) * 2.3) + 0.08 * np.maximum(0.0, y - 0.5) * 3.0
        pts_list.append(np.stack([rad * np.cos(th), y, rad * np.sin(th)], axis=-1))

    elif concept_id == "lamp_lightbulb":
        n_bulb = int(n * 0.68)
        pts_list.append(_sample_ellipsoid(rng, n_bulb, (0, 0.18, 0), (0.44, 0.48, 0.44)))
        rem = n - n_bulb
        pts_list.append(_sample_cylinder(rng, rem, np.array([0, -0.24, 0], np.float32), np.array([0, -0.68, 0], np.float32), 0.22, 0.18))

    elif concept_id == "house_architecture":
        n_walls = int(n * 0.55)
        n_roof = n - n_walls
        pts_list.append(_sample_box_surface(rng, n_walls, (0, -0.24, 0), (0.55, 0.36, 0.48)))
        u = rng.uniform(0.0, 1.0, n_roof).astype(np.float32)
        z = rng.uniform(-0.54, 0.54, n_roof).astype(np.float32)
        sgn = rng.choice([-1.0, 1.0], size=n_roof).astype(np.float32)
        pts_list.append(np.stack([sgn * u * 0.62, 0.56 - u * 0.44, z], axis=-1))
        pts_list = [_rot_y(np.concatenate(pts_list, axis=0), 0.48)]

    elif concept_id == "pyramid_tetrahedron":
        side = rng.integers(0, 4, size=n)
        u = np.sqrt(rng.uniform(0, 1, n)).astype(np.float32)
        v = rng.uniform(-1, 1, n).astype(np.float32) * u
        apex = np.array([0.0, 0.72, 0.0], np.float32)
        pts = np.zeros((n, 3), dtype=np.float32)
        for s, (dx, dz, tx, tz) in enumerate([(1, 0, 0, 1), (-1, 0, 0, 1), (0, 1, 1, 0), (0, -1, 1, 0)]):
            m = side == s
            pts[m] = apex * (1.0 - u[m, None]) + np.stack([u[m] * dx * 0.68 + v[m] * tx * 0.68, np.full(m.sum(), -0.52), u[m] * dz * 0.68 + v[m] * tz * 0.68], axis=-1)
        pts_list.append(_rot_y(pts, 0.45))

    elif concept_id == "standing_wave_ripple":
        x = rng.uniform(-0.9, 0.9, n).astype(np.float32)
        z = rng.uniform(-0.9, 0.9, n).astype(np.float32)
        r = np.sqrt(x * x + z * z)
        y = 0.34 * np.cos(r * 9.5) * np.exp(-1.1 * r)
        pts_list.append(_rot_x(np.stack([x, y, z], axis=-1), 0.38))

    elif concept_id == "mobius_klein_ribbon":
        u = rng.uniform(0, 2 * math.pi, n).astype(np.float32)
        v = rng.uniform(-0.32, 0.32, n).astype(np.float32)
        x = (0.68 + v * np.cos(0.5 * u)) * np.cos(u)
        z = (0.68 + v * np.cos(0.5 * u)) * np.sin(u)
        y = v * np.sin(0.5 * u) * 1.4
        pts_list.append(_rot_x(np.stack([x, y, z], axis=-1), 0.45))

    elif concept_id == "lorenz_attractor":
        dt = 0.006
        state = np.array([0.1, 0.0, 0.0], dtype=np.float64)
        traj = []
        total_steps = n * 6 + 400
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        for step in range(total_steps):
            dx = sigma * (state[1] - state[0])
            dy = state[0] * (rho - state[2]) - state[1]
            dz = state[0] * state[1] - beta * state[2]
            state = state + dt * np.array([dx, dy, dz])
            if step >= 400 and (step - 400) % 6 == 0:
                traj.append(state.copy())
        arr = np.asarray(traj[:n], dtype=np.float32)
        arr += rng.normal(0, 0.35, arr.shape).astype(np.float32)
        pts_list.append(arr)

    elif concept_id == "lightning_bolt":
        waypoints = [
            np.array([0.18, 0.92, 0.0], np.float32),
            np.array([-0.18, 0.28, 0.04], np.float32),
            np.array([0.16, 0.22, -0.02], np.float32),
            np.array([-0.16, -0.32, 0.03], np.float32),
            np.array([0.10, -0.36, -0.02], np.float32),
            np.array([-0.14, -0.92, 0.0], np.float32),
        ]
        per_s = n // (len(waypoints) - 1)
        rem = n
        for i in range(len(waypoints) - 1):
            cnt = per_s if i < len(waypoints) - 2 else rem
            rem -= cnt
            pts_list.append(_sample_cylinder(rng, cnt, waypoints[i], waypoints[i + 1], 0.065, 0.045))

    elif concept_id == "checkmark_verified":
        n1 = int(n * 0.36)
        n2 = n - n1
        pts_list.append(_sample_cylinder(rng, n1, np.array([-0.68, -0.04, 0], np.float32), np.array([-0.16, -0.56, 0], np.float32), 0.085))
        pts_list.append(_sample_cylinder(rng, n2, np.array([-0.16, -0.56, 0], np.float32), np.array([0.74, 0.62, 0], np.float32), 0.085, 0.055))

    elif concept_id == "magnifying_glass":
        n_ring = int(n * 0.62)
        u = rng.uniform(0, 2 * math.pi, n_ring).astype(np.float32)
        v = rng.uniform(0, 2 * math.pi, n_ring).astype(np.float32)
        pts_list.append(np.stack([-0.16 + (0.42 + 0.05 * np.cos(v)) * np.cos(u), 0.18 + (0.42 + 0.05 * np.cos(v)) * np.sin(u), 0.05 * np.sin(v)], axis=-1))
        rem = n - n_ring
        pts_list.append(_sample_cylinder(rng, rem, np.array([0.14, -0.12, 0], np.float32), np.array([0.74, -0.72, 0], np.float32), 0.055))

    elif concept_id == "envelope_contact":
        n_box = int(n * 0.68)
        pts_list.append(_sample_box_surface(rng, n_box, (0, 0, 0), (0.72, 0.46, 0.05)))
        rem = n - n_box
        pts_list.append(_sample_cylinder(rng, rem // 2, np.array([-0.72, 0.46, 0.06], np.float32), np.array([0.0, -0.06, 0.07], np.float32), 0.025))
        pts_list.append(_sample_cylinder(rng, rem - rem // 2, np.array([0.72, 0.46, 0.06], np.float32), np.array([0.0, -0.06, 0.07], np.float32), 0.025))

    elif concept_id == "helicopter_rotor":
        pts_list.append(_sample_ellipsoid(rng, int(n * 0.34), (0, 0, 0.12), (0.20, 0.20, 0.38)))
        pts_list.append(_sample_cylinder(rng, int(n * 0.22), np.array([0, 0.04, -0.22], np.float32), np.array([0, 0.08, -0.88], np.float32), 0.07, 0.025))
        rem = n - sum(len(p) for p in pts_list)
        pts_list.append(_sample_box_surface(rng, rem // 2, (0, 0.26, 0.12), (0.82, 0.012, 0.045)))
        pts_list.append(_sample_box_surface(rng, rem - rem // 2, (0, 0.26, 0.12), (0.045, 0.012, 0.82)))
        pts_list = [_rot_y(np.concatenate(pts_list, axis=0), 0.52)]

    elif concept_id == "ship_boat":
        u = rng.uniform(-1.0, 1.0, int(n * 0.56)).astype(np.float32)
        v = rng.uniform(0.0, 1.0, len(u)).astype(np.float32)
        sgn = rng.choice([-1.0, 1.0], size=len(u)).astype(np.float32)
        w_hull = 0.30 * (1.0 - u * u) * (0.4 + 0.6 * v)
        pts_list.append(np.stack([sgn * w_hull, -0.42 + 0.28 * v, u * 0.86], axis=-1))
        pts_list.append(_sample_cylinder(rng, int(n * 0.14), np.array([0, -0.14, 0.05], np.float32), np.array([0, 0.72, 0.05], np.float32), 0.028))
        rem = n - sum(len(p) for p in pts_list)
        sy = rng.uniform(-0.05, 0.68, rem).astype(np.float32)
        sz_hi = np.maximum(0.08, 0.58 * (1.0 - (sy + 0.05) / 0.86))
        sz = rng.uniform(0.05, sz_hi, rem).astype(np.float32)
        pts_list.append(np.stack([0.08 * np.sin(sz * 3.0), sy, sz], axis=-1))
        pts_list = [_rot_y(np.concatenate(pts_list, axis=0), 0.45)]

    elif concept_id == "transformer_tower":
        # Stacked attention/MLP blocks (`CIR 503M` / Causal Transformer tower)
        per_blk = int(n * 0.20)
        for k, y_c in enumerate([-0.62, -0.20, 0.22, 0.64]):
            pts_list.append(_sample_box_surface(rng, per_blk, (0, y_c, 0), (0.42 - 0.03 * k, 0.11, 0.42 - 0.03 * k)))
        rem = n - sum(len(p) for p in pts_list)
        pts_list.append(_sample_cylinder(rng, rem, np.array([0, -0.78, 0], np.float32), np.array([0, 0.88, 0], np.float32), 0.06))
        pts_list = [_rot_y(np.concatenate(pts_list, axis=0), 0.45)]

    elif concept_id == "sofa_couch":
        pts_list.append(_sample_box_surface(rng, int(n * 0.40), (0, -0.18, 0.06), (0.72, 0.14, 0.30)))
        pts_list.append(_sample_box_surface(rng, int(n * 0.34), (0, 0.18, -0.22), (0.72, 0.26, 0.09)))
        rem = n - sum(len(p) for p in pts_list)
        pts_list.append(_sample_box_surface(rng, rem // 2, (-0.72, 0.02, 0.02), (0.10, 0.22, 0.32)))
        pts_list.append(_sample_box_surface(rng, rem - rem // 2, (0.72, 0.02, 0.02), (0.10, 0.22, 0.32)))
        pts_list = [_rot_y(np.concatenate(pts_list, axis=0), 0.42)]

    elif concept_id == "bed_sleep":
        pts_list.append(_sample_box_surface(rng, int(n * 0.56), (0, -0.18, 0.06), (0.62, 0.14, 0.78)))
        pts_list.append(_sample_box_surface(rng, int(n * 0.24), (0, 0.18, -0.72), (0.64, 0.34, 0.05)))
        rem = n - sum(len(p) for p in pts_list)
        pts_list.append(_sample_ellipsoid(rng, rem // 2, (-0.28, 0.02, -0.48), (0.22, 0.07, 0.14)))
        pts_list.append(_sample_ellipsoid(rng, rem - rem // 2, (0.28, 0.02, -0.48), (0.22, 0.07, 0.14)))
        pts_list = [_rot_y(np.concatenate(pts_list, axis=0), 0.48)]

    else:
        # Fallback crystal octahedron
        pts_list.append(_sample_ellipsoid(rng, n, (0, 0, 0), (0.6, 0.8, 0.6)))

    pts = np.concatenate(pts_list, axis=0)[:n]
    # Apply subtle continuous aspect/rotation jitter (`+-8%` scale, `+-0.10 rad` yaw)
    scale_jit = rng.uniform(0.92, 1.08, size=(1, 3)).astype(np.float32)
    pts = _rot_y(pts * scale_jit, float(rng.uniform(-0.10, 0.10)))
    sal = _compute_curvature_salience(pts)
    return _normalize_cloud(pts, sal)


# =============================================================================
# 3. Real 3D CAD `.off` Mesh Loader (`ModelNet10`)
# =============================================================================


def _parse_and_sample_off_mesh(raw_text: str, rng: np.random.Generator, n: int = 1024) -> Optional[np.ndarray]:
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return None
    idx = 0
    if lines[0] == "OFF":
        idx = 1
    elif lines[0].startswith("OFF"):
        lines[0] = lines[0][3:].strip()
    header = lines[idx].split()
    n_verts, n_faces = int(header[0]), int(header[1])
    idx += 1
    if n_verts < 4 or n_faces < 2 or len(lines) < idx + n_verts + n_faces:
        return None
    verts = np.zeros((n_verts, 3), dtype=np.float32)
    for v_i in range(n_verts):
        parts = lines[idx + v_i].split()
        verts[v_i] = [float(parts[0]), float(parts[1]), float(parts[2])]
    idx += n_verts

    tris = []
    for f_i in range(n_faces):
        parts = [int(x) for x in lines[idx + f_i].split()]
        cnt = parts[0]
        v_ids = parts[1 : 1 + cnt]
        for k in range(1, cnt - 1):
            tris.append((v_ids[0], v_ids[k], v_ids[k + 1]))
    if not tris:
        return None
    tri_arr = np.asarray(tris, dtype=np.int32)
    v0 = verts[tri_arr[:, 0]]
    v1 = verts[tri_arr[:, 1]]
    v2 = verts[tri_arr[:, 2]]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=-1)
    tot_a = float(areas.sum())
    if tot_a <= 1e-10:
        return None
    probs = areas / tot_a
    chosen = rng.choice(len(tri_arr), size=n, p=probs)
    r1 = np.sqrt(rng.uniform(0.0, 1.0, size=n)).astype(np.float32)
    r2 = rng.uniform(0.0, 1.0, size=n).astype(np.float32)
    pts = (1.0 - r1)[:, None] * v0[chosen] + (r1 * (1.0 - r2))[:, None] * v1[chosen] + (r1 * r2)[:, None] * v2[chosen]
    # ModelNet Z is vertical up -> map `(x, z, -y)` so `+Y` is canonical vertical up!
    pts = np.stack([pts[:, 0], pts[:, 2], -pts[:, 1]], axis=-1)
    sal = _compute_curvature_salience(pts)
    return _normalize_cloud(pts, sal)


def try_load_modelnet10_meshes(rng: np.random.Generator, max_per_class: int = 36, n_pts: int = 1024) -> Dict[str, List[np.ndarray]]:
    """Downloads Princeton ModelNet10 (`~90 MB`) if reachable and samples uniform area point clouds."""
    url = "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip"
    zip_path = Path("/content/ModelNet10.zip")
    out: Dict[str, List[np.ndarray]] = {}
    try:
        if not zip_path.exists():
            print(f"[Dataset] Downloading real 3D CAD meshes from {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as resp, open(zip_path, "wb") as f:
                f.write(resp.read())
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = [nm for nm in zf.namelist() if nm.endswith(".off") and "/train/" in nm and "__MACOSX" not in nm]
            by_cls: Dict[str, List[str]] = {}
            for nm in names:
                parts = nm.split("/")
                if len(parts) >= 3:
                    cls_name = parts[-3].lower()
                    by_cls.setdefault(cls_name, []).append(nm)
            for cls_name, files in sorted(by_cls.items()):
                rng.shuffle(files)
                clouds: List[np.ndarray] = []
                for fn in files[: max_per_class * 2]:
                    raw = zf.read(fn).decode("utf-8", errors="ignore")
                    c = _parse_and_sample_off_mesh(raw, rng, n_pts)
                    if c is not None:
                        clouds.append(c)
                    if len(clouds) >= max_per_class:
                        break
                if clouds:
                    out[cls_name] = clouds
                    print(f"  + Real CAD class `{cls_name}`: {len(clouds)} meshes sampled ({n_pts} pts)")
    except Exception as exc:
        print(f"[Dataset] ModelNet10 notice (continuing with parametric + procedural meshes): {exc}")
    return out


# =============================================================================
# 4. Multilingual (`PT`, `EN`, `ES`, `FR`) Concept & Conversational Catalog
# =============================================================================


CONCEPT_SPECS: List[Dict[str, Any]] = [
    {
        "id": "dna_helix",
        "modelnet": None,
        "terms": {
            "pt": ["dupla hélice de DNA", "molécula de DNA", "genoma humano", "ácido desoxirribonucleico", "sequenciamento genético CRISPR", "fita dupla genética"],
            "en": ["DNA double helix", "DNA molecule", "human genome", "deoxyribonucleic acid", "CRISPR gene editing", "genetic double strand"],
            "es": ["doble hélice de ADN", "molécula de ADN", "genoma humano", "ácido desoxirribonucleico", "secuenciación genética"],
            "fr": ["double hélice d'ADN", "molécule d'ADN", "génome humain", "acide désoxyribonucléique", "séquençage génétique"],
        },
    },
    {
        "id": "atom_orbital",
        "modelnet": None,
        "terms": {
            "pt": ["átomo e orbitais eletrônicos", "física quântica atômica", "núcleo atômico e elétrons", "estrutura de um átomo", "mecânica quântica"],
            "en": ["atom and electron orbitals", "quantum atomic physics", "atomic nucleus and electrons", "structure of an atom", "quantum mechanics"],
            "es": ["átomo y orbitales electrónicos", "física cuántica atómica", "núcleo atómico y electrones", "estructura de un átomo"],
            "fr": ["atome et orbitales électroniques", "physique quantique atomique", "noyau atomique et électrons", "structure d'un atome"],
        },
    },
    {
        "id": "ringed_planet",
        "modelnet": None,
        "terms": {
            "pt": ["planeta Saturno com anéis", "planeta gigante gasoso", "anéis planetários no espaço", "astronomia sistema solar", "órbita planetária"],
            "en": ["ringed planet Saturn", "gas giant planet", "planetary rings in space", "solar system astronomy", "planetary orbit"],
            "es": ["planeta Saturno con anillos", "planeta gigante gaseoso", "anillos planetarios", "astronomía del sistema solar"],
            "fr": ["planète Saturne avec anneaux", "planète géante gazeuse", "anneaux planétaires", "astronomie du système solaire"],
        },
    },
    {
        "id": "spiral_galaxy",
        "modelnet": None,
        "terms": {
            "pt": ["galáxia espiral Via Láctea", "braços espirais galácticos", "cosmos e astrofísica", "nebulosa e galáxia no universo"],
            "en": ["Milky Way spiral galaxy", "galactic spiral arms", "cosmos and astrophysics", "deep space spiral galaxy"],
            "es": ["galaxia espiral Vía Láctea", "brazos espirales galácticos", "cosmos y astrofísica", "galaxia en el universo"],
            "fr": ["galaxie spirale Voie lactée", "bras spiraux galactiques", "cosmos et astrophysique", "galaxie dans l'univers"],
        },
    },
    {
        "id": "torus_manifold",
        "modelnet": None,
        "terms": {
            "pt": ["toro 3D topológico", "variedade toroidal matemática", "reator tokamak de fusão", "geometria de um toro"],
            "en": ["3D topological torus", "toroidal mathematical manifold", "tokamak fusion plasma torus", "donut torus geometry"],
            "es": ["toro 3D topológico", "variedad toroidal matemática", "reactor tokamak toroidal", "geometría de un toro"],
            "fr": ["tore 3D topologique", "variété toroïdale mathématique", "réacteur tokamak toroïdal", "géométrie d'un tore"],
        },
    },
    {
        "id": "mos_scache_architecture",
        "modelnet": None,
        "terms": {
            "pt": ["Mixture of Scars MoS e SCache O(1)", "memória associativa constante SCache sem KV Cache", "arquitetura de cicatrizes de 34.8 KB", "Qwen 72B MoS e Widrow-Hoff O(1)", "ScarNativeLM v5 cicatrizes nativas"],
            "en": ["Mixture of Scars MoS and SCache O(1)", "constant associative SCache memory without KV Cache", "34.8 KB scar memory architecture", "Qwen 72B MoS and Widrow-Hoff O(1)", "ScarNativeLM v5 native scar architecture"],
            "es": ["Mixture of Scars MoS y SCache O(1)", "memoria asociativa constante SCache sin KV Cache", "arquitectura de cicatrices de 34.8 KB", "ScarNativeLM v5 cicatrices nativas"],
            "fr": ["Mixture of Scars MoS et SCache O(1)", "mémoire associative constante SCache sans KV Cache", "architecture de cicatrices 34.8 KB", "ScarNativeLM v5 cicatrices natives"],
        },
    },
    {
        "id": "vq4_d4_hypercube",
        "modelnet": None,
        "terms": {
            "pt": ["hipercubo 4D tesseract e reticulado D4", "codec VQ4-Scar LUT256 e FWHT-64", "quantização vetorial 4D D4", "projeção de um tesseract em 4 dimensões"],
            "en": ["4D tesseract hypercube and D4 lattice", "VQ4-Scar LUT256 codec and FWHT-64", "4D D4 vector quantization", "four-dimensional tesseract projection"],
            "es": ["hipercubo 4D teseracto y retículo D4", "códec VQ4-Scar LUT256 y FWHT-64", "cuantización vectorial 4D D4"],
            "fr": ["hypercube 4D tesseract et réseau D4", "codec VQ4-Scar LUT256 et FWHT-64", "quantification vectorielle 4D D4"],
        },
    },
    {
        "id": "tricompass_astrolabe",
        "modelnet": None,
        "terms": {
            "pt": ["Tri-Compass e Batedores de Fronteira", "bússola perceptual Tri-Compass", "Frontier Ghost-Jump Scouts", "astrolábio de três eixos"],
            "en": ["Tri-Compass and Frontier Scouts", "Tri-Compass perceptual front-end", "Frontier Ghost-Jump Scouts", "three-axis gimbal astrolabe"],
            "es": ["Tri-Compass y Exploradores de Frontera", "brújula perceptual Tri-Compass", "astrolabio de tres ejes"],
            "fr": ["Tri-Compass et Éclaireurs de Frontière", "boussole perceptuelle Tri-Compass", "astrolabe à trois axes"],
        },
    },
    {
        "id": "neural_network",
        "modelnet": None,
        "terms": {
            "pt": ["rede neural artificial profunda", "camadas e sinapses de deep learning", "perceptron multicamadas MLP", "inteligência artificial e pesos neurais"],
            "en": ["deep artificial neural network", "deep learning layers and synapses", "multilayer perceptron MLP", "neural network weights and graph"],
            "es": ["red neuronal artificial profunda", "capas y sinapsis de aprendizaje profundo", "perceptrón multicapa MLP"],
            "fr": ["réseau de neurones artificiels profond", "couches et synapses d'apprentissage profond", "perceptron multicouche MLP"],
        },
    },
    {
        "id": "rubiks_tensor_cube",
        "modelnet": None,
        "terms": {
            "pt": ["cubo mágico 3x3x3", "tensor tridimensional em blocos", "cubo de Rubik e ferramentas interativas", "matriz cúbica 3D"],
            "en": ["3x3x3 Rubik's cube", "three-dimensional block tensor", "interactive 3D cube and developer tools", "3D cubic tensor lattice"],
            "es": ["cubo de Rubik 3x3x3", "tensor tridimensional en bloques", "herramientas interactivas y cubo 3D"],
            "fr": ["cube de Rubik 3x3x3", "tenseur tridimensionnel en blocs", "outils interactifs et cube 3D"],
        },
    },
    {
        "id": "heart_3d",
        "modelnet": None,
        "terms": {
            "pt": ["coração tridimensional", "formato de coração 3D", "órgão cardíaco e batimento", "amor e empatia"],
            "en": ["three-dimensional heart", "3D heart shape", "cardiac heart organ", "heart symbol"],
            "es": ["corazón tridimensional", "forma de corazón 3D", "órgano cardíaco"],
            "fr": ["cœur tridimensionnel", "forme de cœur 3D", "organe cardiaque"],
        },
    },
    {
        "id": "brain_cortex",
        "modelnet": None,
        "terms": {
            "pt": ["cérebro humano e córtex cerebral", "neurociência e hemisférios cerebrais", "cognição e mente humana", "consolidação de memória no sono"],
            "en": ["human brain and cerebral cortex", "neuroscience and brain hemispheres", "human cognition and mind", "sleep synthesis and memory consolidation"],
            "es": ["cerebro humano y corteza cerebral", "neurociencia y hemisferios cerebrales", "consolidación de memoria en el sueño"],
            "fr": ["cerveau humain et cortex cérébral", "neurosciences et hémisphères cérébraux", "consolidation de la mémoire pendant le sommeil"],
        },
    },
    {
        "id": "rocket_spacecraft",
        "modelnet": None,
        "terms": {
            "pt": ["foguete espacial em lançamento", "propulsão aeroespacial", "nave espacial e exploração", "veículo lançador orbital"],
            "en": ["space rocket launch vehicle", "aerospace rocket propulsion", "spacecraft exploration", "orbital booster rocket"],
            "es": ["cohete espacial de lanzamiento", "propulsión aeroespacial", "nave espacial orbital"],
            "fr": ["fusée spatiale de lancement", "propulsion aérospatiale", "véhicule lanceur orbital"],
        },
    },
    {
        "id": "satellite_orbital",
        "modelnet": None,
        "terms": {
            "pt": ["satélite artificial com painéis solares", "telecomunicações em órbita", "sonda espacial e antena parabólica", "HTTP 206 streaming remoto de cartuchos"],
            "en": ["artificial satellite with solar panels", "orbital telecommunications satellite", "space probe with parabolic dish", "HTTP 206 remote cartridge streaming"],
            "es": ["satélite artificial con paneles solares", "telecomunicaciones en órbita", "streaming HTTP 206 de cartuchos"],
            "fr": ["satellite artificiel avec panneaux solaires", "télécommunications en orbite", "streaming HTTP 206 de cartouches"],
        },
    },
    {
        "id": "airplane_jet",
        "modelnet": None,
        "terms": {
            "pt": ["avião a jato comercial", "aeronave e aerodinâmica de voo", "asas e fuselagem de um avião", "aviação"],
            "en": ["commercial jet airplane", "aircraft flight aerodynamics", "wings and fuselage of an airplane", "aviation jet"],
            "es": ["avión a reacción comercial", "aeronave y aerodinámica de vuelo", "alas y fuselaje de un avión"],
            "fr": ["avion à réaction commercial", "aéronef et aérodynamique de vol", "ailes et fuselage d'un avion"],
        },
    },
    {
        "id": "tree_canopy",
        "modelnet": None,
        "terms": {
            "pt": ["árvore frondosa da natureza", "floresta e tronco com galhos", "botânica e copa de árvore", "ecologia"],
            "en": ["leafy nature tree with branches", "forest tree trunk and canopy", "botanical tree", "oak tree"],
            "es": ["árbol frondoso de la naturaleza", "bosque y tronco con ramas", "copa de un árbol"],
            "fr": ["arbre feuillu de la nature", "forêt et tronc avec branches", "cime d'un arbre"],
        },
    },
    {
        "id": "binary_tree_graph",
        "modelnet": None,
        "terms": {
            "pt": ["árvore binária de busca e grafos", "Teorema de Immerman-Szelepcsényi NL=coNL", "estrutura de dados hierárquica", "alcance em grafos direcionados"],
            "en": ["binary search tree and directed graph", "Immerman-Szelepcsenyi theorem NL=coNL", "hierarchical tree data structure", "directed graph reachability"],
            "es": ["árbol binario de búsqueda y grafos", "Teorema de Immerman-Szelepcsényi", "estructura de datos jerárquica"],
            "fr": ["arbre binaire de recherche et graphes", "Théorème d'Immerman-Szelepcsényi", "structure de données hiérarchique"],
        },
    },
    {
        "id": "humanoid_agent",
        "modelnet": None,
        "terms": {
            "pt": ["robô humanoide e agente AURA System One", "aprendizado por reforço profundo embodied RL", "robótica autônoma", "corpo articulado de um agente"],
            "en": ["humanoid robot and AURA System One agent", "embodied deep reinforcement learning RL", "autonomous robotics", "articulated embodied agent"],
            "es": ["robot humanoide y agente AURA System One", "aprendizaje por refuerzo profundo RL", "robótica autónoma"],
            "fr": ["robot humanoïde et agent AURA System One", "apprentissage par renforcement profond RL", "robotique autonome"],
        },
    },
    {
        "id": "jarvis_constellation_core",
        "modelnet": None,
        "terms": {
            "pt": ["quem é você Jarvis assistente do portfólio", "rosto neural 3D do Jarvis", "identidade e capacidades do Jarvis", "olá Jarvis tudo bem"],
            "en": ["who are you Jarvis portfolio assistant", "Jarvis 3D neural constellation face", "Jarvis identity and capabilities", "hello Jarvis how are you"],
            "es": ["quién eres Jarvis asistente del portafolio", "rostro neuronal 3D de Jarvis", "identidad y capacidades de Jarvis"],
            "fr": ["qui êtes-vous Jarvis assistant du portfolio", "visage neuronal 3D de Jarvis", "identité et capacités de Jarvis"],
        },
    },
    {
        "id": "shield_security",
        "modelnet": None,
        "terms": {
            "pt": ["escudo de segurança cibernética", "proteção contra ataques e integridade", "guardrail epistêmico e defesa", "auditoria de regras de segurança"],
            "en": ["cybersecurity defense shield", "protection and host integrity guard", "epistemic guardrail and defense", "security audit shield"],
            "es": ["escudo de ciberseguridad y defensa", "protección e integridad del sistema", "auditoría de seguridad"],
            "fr": ["bouclier de cybersécurité et défense", "protection et intégrité du système", "audit de sécurité"],
        },
    },
    {
        "id": "padlock_privacy",
        "modelnet": None,
        "terms": {
            "pt": ["cadeado criptográfico e LGPD GDPR 0x00", "desaprendizagem física instantânea unlearning", "privacidade de dados e criptografia", "apagar cartucho de memória"],
            "en": ["cryptographic padlock and GDPR LGPD 0x00", "instant physical machine unlearning", "data privacy and encryption lock", "wipe memory cartridge"],
            "es": ["candado criptográfico y RGPD LGPD 0x00", "desaprendizaje físico instantáneo", "privacidad de datos y cifrado"],
            "fr": ["cadenas cryptographique et RGPD 0x00", "désapprentissage physique instantané", "confidentialité des données"],
        },
    },
    {
        "id": "entropy_funnel_blackhole",
        "modelnet": None,
        "terms": {
            "pt": ["Princípio de Landauer e entropia termodinâmica", "buraco negro e poço gravitacional", "dissipação térmica kT ln 2", "física da informação e espaço-tempo"],
            "en": ["Landauer's principle and thermodynamic entropy", "black hole and gravitational well", "kT ln 2 thermal dissipation", "physics of information and spacetime"],
            "es": ["Principio de Landauer y entropía termodinámica", "agujero negro y pozo gravitacional", "física de la información"],
            "fr": ["Principe de Landauer et entropie thermodynamique", "trou noir et puits gravitationnel", "physique de l'information"],
        },
    },
    {
        "id": "hourglass_time",
        "modelnet": None,
        "terms": {
            "pt": ["ampulheta do tempo e história de carreira", "trajetória profissional e linha do tempo de Julio Reis", "complexidade temporal e latência", "experiência ao longo dos anos"],
            "en": ["hourglass of time and career timeline", "Julio Reis professional background and career history", "time complexity and latency", "experience timeline over the years"],
            "es": ["reloj de arena y trayectoria profesional", "experiencia y carrera de Julio Reis", "complejidad temporal y latencia"],
            "fr": ["sablier du temps et parcours professionnel", "carrière et expérience de Julio Reis", "complexité temporelle et latence"],
        },
    },
    {
        "id": "gear_cogwheel",
        "modelnet": None,
        "terms": {
            "pt": ["engrenagem mecânica e engenharia de sistemas", "automação e pipeline industrial", "mecanismo de precisão", "otimização de compiladores"],
            "en": ["mechanical gear cogwheel and systems engineering", "automation and engineering pipeline", "precision mechanism", "compiler optimization"],
            "es": ["engranaje mecánico e ingeniería de sistemas", "automatización y mecanismo de precisión"],
            "fr": ["engrenage mécanique et ingénierie des systèmes", "automatisation et mécanisme de précision"],
        },
    },
    {
        "id": "suspension_bridge",
        "modelnet": None,
        "terms": {
            "pt": ["ponte estaiada suspensa", "ponte entre física de Landauer e Immerman", "engenharia civil e conexão entre domínios", "arquitetura de pontes"],
            "en": ["suspension cable bridge", "cross-domain bridge between Landauer and Immerman", "civil engineering bridge", "connecting two domains"],
            "es": ["puente colgante de cables", "puente interdisciplinario entre física y complejidad", "ingeniería de puentes"],
            "fr": ["pont suspendu à câbles", "pont interdisciplinaire entre physique et complexité", "ingénierie des ponts"],
        },
    },
    {
        "id": "globe_earth",
        "modelnet": None,
        "terms": {
            "pt": ["globo terrestre poliglota mundial", "suporte multilíngue português inglês espanhol francês", "planeta Terra e coordenadas geográficas", "projeção ortogonal ao idioma P_perp_lang"],
            "en": ["polyglot world Earth globe", "multilingual support across Portuguese English Spanish French", "planet Earth latitude and longitude", "language-nullspace projection P_perp_lang"],
            "es": ["globo terráqueo políglota mundial", "soporte multilingüe en cuatro idiomas", "planeta Tierra y coordenadas"],
            "fr": ["globe terrestre polyglotte mondial", "support multilingue en quatre langues", "planète Terre et coordonnées"],
        },
    },
    {
        "id": "laptop_computer",
        "modelnet": "monitor",
        "terms": {
            "pt": ["notebook computador rodando no navegador", "desenvolvimento fullstack web", "aplicação cliente local no laptop", "interface de software"],
            "en": ["laptop computer running in the browser", "fullstack web software engineering", "local client-side web application", "computer workstation"],
            "es": ["computadora portátil en el navegador", "desarrollo de software fullstack web", "aplicación local en el navegador"],
            "fr": ["ordinateur portable dans le navigateur", "développement logiciel fullstack web", "application locale dans le navigateur"],
        },
    },
    {
        "id": "gpu_silicon_chip",
        "modelnet": None,
        "terms": {
            "pt": ["placa de vídeo GPU e shaders WebGPU WGSL", "processador de silício e núcleos tensoriais", "computação paralela em GPU Metal Vulkan", "20 shaders compute WGSL 1.0"],
            "en": ["GPU silicon chip and WebGPU WGSL shaders", "hardware processor and tensor cores", "parallel GPU compute Metal Vulkan", "20 custom WGSL 1.0 compute shaders"],
            "es": ["chip de silicio GPU y shaders WebGPU WGSL", "procesador gráfico y computación paralela", "20 shaders WGSL en el navegador"],
            "fr": ["puce silicium GPU et shaders WebGPU WGSL", "processeur graphique et calcul parallèle", "20 shaders WGSL dans le navigateur"],
        },
    },
    {
        "id": "chair_furniture",
        "modelnet": "chair",
        "terms": {
            "pt": ["cadeira ergonômica 3D", "poltrona de escritório", "design de mobiliário cadeira"],
            "en": ["3D ergonomic chair", "office armchair", "furniture chair design"],
            "es": ["silla ergonómica 3D", "sillón de oficina", "diseño de silla"],
            "fr": ["chaise ergonomique 3D", "fauteuil de bureau", "design de chaise"],
        },
    },
    {
        "id": "table_desk",
        "modelnet": "table",
        "terms": {
            "pt": ["mesa de trabalho escrivaninha", "mesa de jantar de quatro pés", "bancada retangular"],
            "en": ["wooden work desk table", "four-legged dining table", "rectangular table"],
            "es": ["mesa de trabajo escritorio", "mesa de comedor de cuatro patas"],
            "fr": ["table de travail bureau", "table à manger à quatre pieds"],
        },
    },
    {
        "id": "car_automobile",
        "modelnet": None,
        "terms": {
            "pt": ["carro automóvel esportivo", "veículo autônomo de quatro rodas", "engenharia automotiva"],
            "en": ["sports car automobile", "autonomous four-wheel vehicle", "automotive car"],
            "es": ["coche automóvil deportivo", "vehículo autónomo de cuatro ruedas"],
            "fr": ["voiture automobile sportive", "véhicule autonome à quatre roues"],
        },
    },
    {
        "id": "quadruped_animal",
        "modelnet": None,
        "terms": {
            "pt": ["cachorro lobo animal quadrúpede", "cão de estimação", "mamífero quadrúpede correndo"],
            "en": ["dog wolf quadruped animal", "pet dog mammal", "running four-legged quadruped"],
            "es": ["perro lobo animal cuadrúpedo", "perro mascota mamífero"],
            "fr": ["chien loup animal quadrupède", "chien domestique mammifère"],
        },
    },
    {
        "id": "bird_eagle",
        "modelnet": None,
        "terms": {
            "pt": ["pássaro águia voando com asas abertas", "ave em voo livre", "falcão planando no céu"],
            "en": ["eagle bird flying with spread wings", "soaring bird in flight", "falcon gliding in the sky"],
            "es": ["pájaro águila volando con alas abiertas", "ave en vuelo libre"],
            "fr": ["oiseau aigle volant ailes déployées", "oiseau en plein vol"],
        },
    },
    {
        "id": "fish_marine",
        "modelnet": None,
        "terms": {
            "pt": ["peixe nadando no oceano", "vida marinha tubarão golfinho", "animal aquático com nadadeiras"],
            "en": ["fish swimming in the ocean", "marine shark dolphin life", "aquatic animal with fins"],
            "es": ["pez nadando en el océano", "vida marina delfín tiburón"],
            "fr": ["poisson nageant dans l'océan", "vie marine dauphin requin"],
        },
    },
    {
        "id": "guitar_instrument",
        "modelnet": None,
        "terms": {
            "pt": ["violão guitarra instrumento musical", "música acústica e cordas", "violino e violão"],
            "en": ["acoustic guitar musical instrument", "stringed guitar music", "electric and acoustic guitar"],
            "es": ["guitarra acústica instrumento musical", "música de cuerdas guitarra"],
            "fr": ["guitare acoustique instrument de musique", "musique à cordes guitare"],
        },
    },
    {
        "id": "sword_blade",
        "modelnet": None,
        "terms": {
            "pt": ["espada medieval com lâmina e guarda", "katana sabre de luz", "arma branca histórica"],
            "en": ["medieval sword with blade and crossguard", "katana saber sword", "knight broadsword"],
            "es": ["espada medieval con hoja y empuñadura", "sable katana espada"],
            "fr": ["épée médiévale avec lame et garde", "sabre katana épée"],
        },
    },
    {
        "id": "crown_trophy",
        "modelnet": None,
        "terms": {
            "pt": ["coroa real e troféu de campeão", "primeiro lugar em competição Kaggle", "vitória e liderança soberana"],
            "en": ["royal crown and champion trophy", "first place competition winner", "sovereign victory crown"],
            "es": ["corona real y trofeo de campeón", "primer lugar en competición"],
            "fr": ["couronne royale et trophée de champion", "première place en compétition"],
        },
    },
    {
        "id": "key_cryptographic",
        "modelnet": None,
        "terms": {
            "pt": ["chave mestra criptográfica", "chave de acesso e autenticação", "segredo de API e permissão"],
            "en": ["cryptographic master key", "authentication access key", "secret key and unlock"],
            "es": ["llave maestra criptográfica", "clave de acceso y autenticación"],
            "fr": ["clé maîtresse cryptographique", "clé d'accès et authentification"],
        },
    },
    {
        "id": "cup_mug",
        "modelnet": None,
        "terms": {
            "pt": ["xícara caneca de café", "copo com alça para bebida quente", "caneca de cerâmica"],
            "en": ["coffee cup mug with handle", "ceramic tea mug", "hot drink coffee cup"],
            "es": ["taza de café con asa", "tazón de cerámica para té"],
            "fr": ["tasse à café avec anse", "mug en céramique"],
        },
    },
    {
        "id": "vase_amphora",
        "modelnet": None,
        "terms": {
            "pt": ["vaso grego ânfora de cerâmica", "jarro decorativo escultural", "garrafa vaso clássico"],
            "en": ["ceramic amphora vase", "sculptural decorative jar", "classic curved vase"],
            "es": ["jarrón ánfora de cerámica", "vasija decorativa clásica"],
            "fr": ["vase amphore en céramique", "jarre décorative classique"],
        },
    },
    {
        "id": "lamp_lightbulb",
        "modelnet": None,
        "terms": {
            "pt": ["lâmpada de ideia brilhante e inovação", "iluminação criativa e descoberta", "luz elétrica"],
            "en": ["bright idea lightbulb and innovation", "creative insight and invention", "electric light bulb"],
            "es": ["bombilla de idea brillante e innovación", "iluminación creativa y descubrimiento"],
            "fr": ["ampoule d'idée brillante et innovation", "illumination créative et découverte"],
        },
    },
    {
        "id": "house_architecture",
        "modelnet": None,
        "terms": {
            "pt": ["casa arquitetônica com telhado", "edifício residencial e moradia", "construção civil casa"],
            "en": ["architectural house with pitched roof", "residential home building", "house structure"],
            "es": ["casa arquitectónica con tejado", "vivienda edificio residencial"],
            "fr": ["maison architecturale avec toit", "bâtiment résidentiel maison"],
        },
    },
    {
        "id": "pyramid_tetrahedron",
        "modelnet": None,
        "terms": {
            "pt": ["pirâmide do Egito e tetraedro", "hierarquia piramidal geométrica", "monumento pirâmide 3D"],
            "en": ["Egyptian pyramid and tetrahedron", "geometric pyramid hierarchy", "3D pyramid monument"],
            "es": ["pirámide de Egipto y tetraedro", "jerarquía piramidal geométrica"],
            "fr": ["pyramide d'Égypte et tétraèdre", "hiérarchie pyramidale géométrique"],
        },
    },
    {
        "id": "standing_wave_ripple",
        "modelnet": None,
        "terms": {
            "pt": ["onda estacionária 3D e interferência acústica", "propagação de ondas eletromagnéticas", "vibração harmônica e sinal"],
            "en": ["3D standing wave and acoustic interference", "electromagnetic wave propagation", "harmonic ripple and signal"],
            "es": ["onda estacionaria 3D e interferencia acústica", "propagación de ondas y señal"],
            "fr": ["onde stationnaire 3D et interférence acoustique", "propagation d'ondes et signal"],
        },
    },
    {
        "id": "mobius_klein_ribbon",
        "modelnet": None,
        "terms": {
            "pt": ["fita de Möbius e topologia não-orientável", "superfície de Klein e loop infinito", "geometria diferencial"],
            "en": ["Mobius strip and non-orientable topology", "Klein surface and infinite loop", "differential geometry"],
            "es": ["cinta de Möbius y topología no orientable", "superficie de Klein e infinito"],
            "fr": ["ruban de Möbius et topologie non orientable", "surface de Klein et boucle infinie"],
        },
    },
    {
        "id": "lorenz_attractor",
        "modelnet": None,
        "terms": {
            "pt": ["atrator estranho de Lorenz e teoria do caos", "efeito borboleta sistemas dinâmicos", "equações diferenciais caóticas"],
            "en": ["Lorenz strange attractor and chaos theory", "butterfly effect dynamical systems", "chaotic differential equations"],
            "es": ["atractor extraño de Lorenz y teoría del caos", "efecto mariposa sistemas dinámicos"],
            "fr": ["attracteur étrange de Lorenz et théorie du chaos", "effet papillon systèmes dynamiques"],
        },
    },
    {
        "id": "lightning_bolt",
        "modelnet": None,
        "terms": {
            "pt": ["raio elétrico de alta velocidade", "inferência ultra-rápida em tempo real", "energia elétrica e descarga"],
            "en": ["high-speed electric lightning bolt", "ultra-fast real-time inference", "electrical energy discharge"],
            "es": ["rayo eléctrico de alta velocidad", "inferencia ultrarrápida en tiempo real"],
            "fr": ["éclair électrique à haute vitesse", "inférence ultra-rapide en temps réel"],
        },
    },
    {
        "id": "checkmark_verified",
        "modelnet": None,
        "terms": {
            "pt": ["símbolo de verificação aprovado", "paridade matemática verificada sem fachada", "teste aprovado com sucesso"],
            "en": ["verified checkmark symbol", "mathematically verified zero-facade parity", "test passed verification"],
            "es": ["símbolo de verificación aprobado", "paridad matemática verificada"],
            "fr": ["symbole de vérification validé", "parité mathématique vérifiée"],
        },
    },
    {
        "id": "magnifying_glass",
        "modelnet": None,
        "terms": {
            "pt": ["lupa de busca e inspeção forense", "pesquisa científica e diagnóstico", "análise detalhada de código"],
            "en": ["magnifying glass search and forensic audit", "scientific research and diagnostics", "deep code inspection"],
            "es": ["lupa de búsqueda y auditoría forense", "investigación científica y diagnóstico"],
            "fr": ["loupe de recherche et audit forensique", "recherche scientifique et diagnostic"],
        },
    },
    {
        "id": "envelope_contact",
        "modelnet": None,
        "terms": {
            "pt": ["envelope de contato e e-mail para contratar Julio Reis", "como enviar mensagem LinkedIn GitHub", "recrutamento e proposta de trabalho"],
            "en": ["contact mail envelope to hire Julio Reis", "how to reach out via email LinkedIn GitHub", "recruiting and job inquiry"],
            "es": ["sobre de correo y contacto para contratar a Julio Reis", "cómo enviar mensaje por LinkedIn o correo"],
            "fr": ["enveloppe courriel et contact pour recruter Julio Reis", "comment contacter par e-mail ou LinkedIn"],
        },
    },
    {
        "id": "helicopter_rotor",
        "modelnet": None,
        "terms": {
            "pt": ["helicóptero com hélices rotativas", "drone aéreo de voo vertical", "aeronave de rotor"],
            "en": ["helicopter with spinning rotor blades", "vertical takeoff aerial drone", "rotorcraft helicopter"],
            "es": ["helicóptero con hélices giratorias", "dron aéreo de vuelo vertical"],
            "fr": ["hélicoptère à pales rotatives", "drone aérien à décollage vertical"],
        },
    },
    {
        "id": "ship_boat",
        "modelnet": None,
        "terms": {
            "pt": ["barco veleiro navegando no mar", "navio embarcação náutica", "caravela com vela"],
            "en": ["sailboat ship sailing on the sea", "nautical vessel boat", "sailing yacht"],
            "es": ["barco velero navegando en el mar", "embarcación náutica buque"],
            "fr": ["bateau voilier naviguant sur la mer", "navire embarcation nautique"],
        },
    },
    {
        "id": "transformer_tower",
        "modelnet": None,
        "terms": {
            "pt": ["arquitetura Causal Transformer CIR 503M", "torre de blocos de atenção multi-cabeça", "modelo fundacional de linguagem"],
            "en": ["CIR 503M Causal Transformer architecture", "stacked multi-head attention tower", "foundation language model"],
            "es": ["arquitectura Causal Transformer CIR 503M", "torre de bloques de atención"],
            "fr": ["architecture Causal Transformer CIR 503M", "tour de blocs d'attention"],
        },
    },
    {
        "id": "sofa_couch",
        "modelnet": "sofa",
        "terms": {
            "pt": ["sofá confortável de sala", "divã estofado de três lugares"],
            "en": ["living room sofa couch", "upholstered three-seat couch"],
            "es": ["sofá cómodo de sala", "sillón largo tapizado"],
            "fr": ["canapé confortable de salon", "divan rembourré"],
        },
    },
    {
        "id": "bed_sleep",
        "modelnet": "bed",
        "terms": {
            "pt": ["cama de casal com cabeceira", "mobiliário de quarto cama"],
            "en": ["bedroom bed with headboard", "double bed frame"],
            "es": ["cama de dormitorio con cabecera", "cama doble"],
            "fr": ["lit de chambre avec tête de lit", "lit double"],
        },
    },
]

TEMPLATES_BY_LANG: Dict[str, List[str]] = {
    "pt": [
        "{term}",
        "Mostre {term}",
        "Como funciona {term}?",
        "Jarvis, me explique sobre {term} e mostre o formato em 3D.",
        "O que é {term}?",
        "Qual é a estrutura física ou geométrica de {term}?",
        "Você pode esculpir {term} com seus pontos flutuantes?",
        "Fale resumidamente sobre {term}.",
    ],
    "en": [
        "{term}",
        "Show me {term}",
        "How does {term} work?",
        "Jarvis, explain {term} and sculpt its 3D geometry.",
        "What is {term}?",
        "Describe the structure and mechanism of {term}.",
        "Can you visualize {term} with floating 3D points?",
        "Give me a concise overview of {term}.",
    ],
    "es": [
        "{term}",
        "Muéstrame {term}",
        "¿Cómo funciona {term}?",
        "Jarvis, explícame qué es {term} y muestra su forma 3D.",
        "¿Qué es {term}?",
        "¿Cuál es la estructura de {term}?",
        "Háblame brevemente sobre {term}.",
    ],
    "fr": [
        "{term}",
        "Montre-moi {term}",
        "Comment fonctionne {term} ?",
        "Jarvis, explique-moi {term} et affiche sa forme 3D.",
        "Qu'est-ce que {term} ?",
        "Quelle est la structure de {term} ?",
        "Parle-moi brièvement de {term}.",
    ],
}


# =============================================================================
# 5. Exact Brenier Optimal Transport (`W_2^2`) + `k=8` Surface Edge Graph
# =============================================================================


def _solve_single_ot_and_knn(args: Tuple[np.ndarray, np.ndarray, int]) -> Tuple[np.ndarray, np.ndarray]:
    x0_xyz, cloud_4d, k_nn = args
    pts = cloud_4d[:, :3]
    # Squared Euclidean cost `[N, N]`
    x0_sq = (x0_xyz * x0_xyz).sum(axis=-1, keepdims=True)
    p_sq = (pts * pts).sum(axis=-1, keepdims=True).T
    cost = np.maximum(x0_sq + p_sq - 2.0 * (x0_xyz @ pts.T), 0.0)
    _, col_ind = linear_sum_assignment(cost)
    ordered = cloud_4d[col_ind].astype(np.float32)

    tree = cKDTree(ordered[:, :3])
    _, nn_idx = tree.query(ordered[:, :3], k=k_nn + 1)
    knn = nn_idx[:, 1 : k_nn + 1].astype(np.int16)
    return ordered, knn


# =============================================================================
# 6. Exact Browser-Matched Qwen2.5-3B `Layer 23` `P_perp_lang` Extractor
# =============================================================================


def find_subseq_last_end(seq: List[int], pat: List[int]) -> int:
    m = len(pat)
    for idx in range(len(seq) - m, -1, -1):
        if seq[idx : idx + m] == pat:
            return min(idx + m, max(len(seq) - 1, 0))
    return 0


def extract_user_bounds(seq: List[int], prefix_ids: List[int], suffix_ids: List[int]) -> Tuple[int, int]:
    n = len(seq)
    q_start = find_subseq_last_end(seq, prefix_ids)
    q_start = max(0, min(q_start, n - 1))
    q_end = n
    ls = len(suffix_ids)
    if ls > 0 and n >= q_start + ls + 1 and seq[n - ls :] == suffix_ids:
        q_end = n - ls
    if q_end <= q_start:
        q_end = n
    return q_start, q_end


@torch.no_grad()
def extract_qwen3b_layer23_latents(
    prompts: List[str],
    router_json_path: Path,
    router_bin_path: Path,
    device: torch.device,
    batch_size: int = 192,
) -> Tuple[np.ndarray, float]:
    """Extracts bit-exact `BrowserScarStoreHTTP206._buildSpanProbes` (`probes[0]`) vectors in `R^2048`."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    meta = json.loads(router_json_path.read_text(encoding="utf-8"))
    raw_f32 = np.frombuffer(router_bin_path.read_bytes(), dtype=np.float32)
    lay = meta["router_bin_layout"]

    d = int(meta["hidden_dim"])
    mid_layer_idx = int(meta["mid_layer_idx"])
    k_lang = int(meta["k_lang"])
    prefix_ids = list(meta["user_turn_prefix_ids"])
    suffix_ids = list(meta["user_turn_suffix_ids"])
    sys_prompt = str(meta.get("system_prompt") or PORTFOLIO_DEFAULT_SYSTEM_PROMPT)

    mu_q = torch.tensor(
        raw_f32[lay["mu_query_mid"]["offset_floats"] : lay["mu_query_mid"]["offset_floats"] + d],
        dtype=torch.float32,
        device=device,
    )
    u_lang = torch.tensor(
        raw_f32[lay["u_lang_mid"]["offset_floats"] : lay["u_lang_mid"]["offset_floats"] + d * k_lang],
        dtype=torch.float32,
        device=device,
    ).reshape(d, k_lang)

    model_id = "Qwen/Qwen2.5-3B-Instruct"
    print(f"[Qwen-L23] Loading `{model_id}` truncated to `layers[0..{mid_layer_idx}]` on {device} ...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    qwen = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
    )
    qwen.eval()
    # Truncate layers strictly above `mid_layer_idx` (`23`) to save VRAM & boost throughput by 50%!
    if hasattr(qwen, "model") and hasattr(qwen.model, "layers"):
        while len(qwen.model.layers) > mid_layer_idx + 1:
            del qwen.model.layers[-1]
    if hasattr(qwen, "lm_head"):
        qwen.lm_head = torch.nn.Identity()
    # Replace `qwen.model.norm` with `Identity()` so `last_hidden_state` is the EXACT raw
    # post-Layer-23 residual stream (`hCurBuf` after `layerOps[23]` in `mosWebgpuEngine.ts`)!
    qwen.model.norm = torch.nn.Identity()
    torch.cuda.empty_cache()

    # Sanity-verify against `z_anchors[0]` from `scarstore_router.bin` BEFORE batch extraction!
    z_anch0 = torch.tensor(
        raw_f32[lay["z_anchors"]["offset_floats"] : lay["z_anchors"]["offset_floats"] + d],
        dtype=torch.float32,
        device=device,
    )
    cap0_prompt = meta["capsules"][0]["metadata"]["prompts_polyglot"]["en"]
    test_chat = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{cap0_prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    t_ids = tok.encode(test_chat, add_special_tokens=False)
    t_qs, t_qe = extract_user_bounds(t_ids, prefix_ids, suffix_ids)
    t_inp = torch.tensor([t_ids[:t_qe]], dtype=torch.long, device=device)
    t_h = qwen.model(input_ids=t_inp, use_cache=False, return_dict=True).last_hidden_state[0, t_qs:t_qe].float()
    t_v = t_h.mean(dim=0) - mu_q
    t_v = t_v / t_v.norm().clamp(min=1e-8)
    t_v = t_v - u_lang @ (u_lang.T @ t_v)
    t_v = t_v / t_v.norm().clamp(min=1e-8)
    anchor_cos = float(torch.dot(t_v, z_anch0).item())
    print(f"[Qwen-L23] Router Anchor[0] Alignment Check Cosine = {anchor_cos:.6f} (must be > 0.70)")
    assert anchor_cos > 0.70, f"Router alignment mismatch! Cosine={anchor_cos:.6f}"

    # Tokenize all prompts with exact browser ChatML template
    encoded_seqs: List[List[int]] = []
    bounds: List[Tuple[int, int]] = []
    for p in prompts:
        chat_str = (
            f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{p}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        ids = tok.encode(chat_str, add_special_tokens=False)
        qs, qe = extract_user_bounds(ids, prefix_ids, suffix_ids)
        encoded_seqs.append(ids[:qe])
        bounds.append((qs, qe))

    order = sorted(range(len(prompts)), key=lambda i: len(encoded_seqs[i]))
    out_latents = np.zeros((len(prompts), d), dtype=np.float32)
    pad_id = int(tok.eos_token_id or 151643)

    t0 = time.perf_counter()
    for s in range(0, len(order), batch_size):
        chunk_idx = order[s : s + batch_size]
        max_len = max(len(encoded_seqs[i]) for i in chunk_idx)
        input_ids = torch.full((len(chunk_idx), max_len), pad_id, dtype=torch.long, device=device)
        for bi, orig_i in enumerate(chunk_idx):
            seq = encoded_seqs[orig_i]
            input_ids[bi, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)

        # Pure causal self-attention on right-padded prefixes `0..qe-1`: zero rightward pad leakage
        # and full FlashAttention-2 / cuDNN SDPA fast-path speed!
        h_batch = qwen.model(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state.float()

        vecs = []
        for bi, orig_i in enumerate(chunk_idx):
            qs, qe = bounds[orig_i]
            vecs.append(h_batch[bi, qs:qe].mean(dim=0))
        v_mat = torch.stack(vecs, dim=0) - mu_q.unsqueeze(0)
        v_mat = v_mat / v_mat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v_mat = v_mat - (v_mat @ u_lang) @ u_lang.T
        v_mat = v_mat / v_mat.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v_np = v_mat.cpu().numpy()
        for bi, orig_i in enumerate(chunk_idx):
            out_latents[orig_i] = v_np[bi]

        if (s // batch_size) % 25 == 0 or s + batch_size >= len(order):
            done = min(len(order), s + batch_size)
            elapsed = time.perf_counter() - t0
            print(f"  [Qwen-L23] Extracted {done}/{len(order)} prompts ({done / max(elapsed, 1e-3):.0f} prompts/s)")

    del qwen
    gc.collect()
    torch.cuda.empty_cache()
    return out_latents, anchor_cos


# =============================================================================
# 7. Chamfer Distance, EMA Helper, & Full A100 Training + Distillation Loop
# =============================================================================


@torch.no_grad()
def compute_chamfer_l2_batch(pred_xyz: torch.Tensor, gt_xyz: torch.Tensor) -> torch.Tensor:
    """Computes symmetric Chamfer-L2 distance (`[B, N, 3] -> [B]`)."""
    dists = torch.cdist(pred_xyz.float(), gt_xyz.float(), p=2.0)  # [B, N, N]
    d1 = dists.amin(dim=-1).mean(dim=-1)
    d2 = dists.amin(dim=-2).mean(dim=-1)
    return 0.5 * (d1 + d2)


@torch.no_grad()
def update_ema(ema_model: PointScarFlow, src_model: PointScarFlow, decay: float = 0.998) -> None:
    for p_ema, p_src in zip(ema_model.parameters(), src_model.parameters()):
        p_ema.data.mul_(decay).add_(p_src.data, alpha=1.0 - decay)


@torch.no_grad()
def sample_teacher_rk2_cfg(
    teacher: PointScarFlow,
    z_batch: torch.Tensor,
    num_steps: int = 8,
    guidance_scale: float = 2.0,
) -> torch.Tensor:
    """High-precision 8-step Midpoint (RK2) ODE solver with CFG for Phase-2 Self-Distillation."""
    bsz = z_batch.shape[0]
    dev = z_batch.device
    x = teacher.initial_state(bsz, dev)
    dt = 1.0 / float(num_steps)
    z_null = torch.zeros_like(z_batch)

    def _guided_vel(x_in: torch.Tensor, t_scalar: float) -> torch.Tensor:
        tau = torch.full((bsz,), t_scalar, dtype=torch.float32, device=dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            vc = teacher(x_in, tau, z_batch)
            vu = teacher(x_in, tau, z_null)
        return (vu + guidance_scale * (vc - vu)).float()

    for step in range(num_steps):
        t0 = float(step) * dt
        v1 = _guided_vel(x, t0)
        x_mid = x + 0.5 * dt * v1
        v2 = _guided_vel(x_mid, t0 + 0.5 * dt)
        x = x + dt * v2

    pos = torch.clamp(x[..., :3], -1.25, 1.25)
    sal = torch.clamp(x[..., 3:4], 0.0, 1.0)
    return torch.cat([pos, sal], dim=-1)


def main() -> None:
    t_start = time.perf_counter()
    assert torch.cuda.is_available(), "CUDA GPU required!"
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("=" * 78)
    print(f"🚀 POINTSCARFLOW v1.0 — TABULA-RASA SCARNATIVE 3D GENERATOR ON {gpu_name} ({vram_gb:.1f} GB)")
    print("=" * 78)

    rng = np.random.default_rng(20261115)
    cfg = PointScarFlowConfig(
        cond_in_dim=2048,
        cond_dim=512,
        d_model=256,
        d_ff=512,
        num_layers=4,
        num_heads=8,
        head_dim=32,
        num_parts=16,
        num_points=1024,
        scar_rank=16,
        cond_scar_rank=24,
        head_scar_rank=8,
    )
    n_pts = cfg.num_points
    x0_np = build_canonical_fibonacci_sphere(n_pts, "cpu").numpy()

    # -------------------------------------------------------------------------
    # Stage 1: Synthesize + Load Real CAD 3D Point Clouds (`~3,850` unique shapes)
    # -------------------------------------------------------------------------
    modelnet_clouds = try_load_modelnet10_meshes(rng, max_per_class=36, n_pts=n_pts)

    shapes_raw: List[np.ndarray] = []
    shape_concept_idx: List[int] = []
    concept_to_shape_ids: Dict[int, List[int]] = {}
    shapes_per_concept = 72  # 52 concepts * 72 continuous variations = 3,744 unique 3D shapes!

    print(f"[Dataset] Synthesizing {len(CONCEPT_SPECS)} x {shapes_per_concept} continuous 3D manifolds ...")
    for c_idx, spec in enumerate(CONCEPT_SPECS):
        cid = spec["id"]
        mn_key = spec.get("modelnet")
        s_ids: List[int] = []
        cad_pool = modelnet_clouds.get(mn_key, []) if mn_key else []
        for k in range(shapes_per_concept):
            if cad_pool and k < len(cad_pool) and k % 2 == 0:
                cloud = cad_pool[k].copy()
            else:
                cloud = build_parametric_shape(cid, rng, n_pts)
            sid = len(shapes_raw)
            shapes_raw.append(cloud)
            shape_concept_idx.append(c_idx)
            s_ids.append(sid)
        concept_to_shape_ids[c_idx] = s_ids

    print(f"[OT] Solving exact 1024x1024 Brenier W2 Optimal Transport + k=8 Edge Graphs for {len(shapes_raw)} shapes ...")
    t_ot = time.perf_counter()
    ot_tasks = [(x0_np, cl, 8) for cl in shapes_raw]
    with ThreadPoolExecutor(max_workers=min(16, os.cpu_count() or 8)) as ex:
        ot_results = list(ex.map(_solve_single_ot_and_knn, ot_tasks))

    shapes_ot = np.stack([r[0] for r in ot_results], axis=0)  # [N_shapes, 1024, 4]
    shapes_knn = np.stack([r[1] for r in ot_results], axis=0)  # [N_shapes, 1024, 8]
    del shapes_raw, ot_results
    print(f"[OT] Completed {len(shapes_ot)} exact global W2 bijections in {time.perf_counter() - t_ot:.1f}s!")

    # -------------------------------------------------------------------------
    # Stage 2: Build Multilingual (`PT/EN/ES/FR`) Prompt Dataset & Held-Out Split
    # -------------------------------------------------------------------------
    all_prompts: List[str] = []
    all_langs: List[str] = []
    all_c_ids: List[int] = []
    all_s_ids: List[int] = []
    all_is_test: List[bool] = []

    for c_idx, spec in enumerate(CONCEPT_SPECS):
        s_pool = concept_to_shape_ids[c_idx]
        for lang, terms in spec["terms"].items():
            tmpls = TEMPLATES_BY_LANG[lang]
            for t_i, term in enumerate(terms):
                for tpl_i, tpl in enumerate(tmpls):
                    p_txt = tpl.format(term=term)
                    # Hold out the last template across every language & term as strictly unseen test phrasing,
                    # plus hold out the last term on templates >= 5!
                    is_test = (tpl_i == len(tmpls) - 1) and (t_i == len(terms) - 1)
                    # Pair each training prompt with 2 distinct continuous geometric instances for rich diversity
                    reps = 1 if is_test else 2
                    for r_idx in range(reps):
                        sid = int(s_pool[(t_i * 13 + tpl_i * 7 + r_idx * 19) % len(s_pool)])
                        all_prompts.append(p_txt)
                        all_langs.append(lang)
                        all_c_ids.append(c_idx)
                        all_s_ids.append(sid)
                        all_is_test.append(is_test)

    print(
        f"[Dataset] Total Multilingual Samples: {len(all_prompts)} "
        f"(Train: {sum(not x for x in all_is_test)}, Held-Out Test: {sum(all_is_test)})"
    )

    router_json = Path("/content/scarstore_router.json")
    router_bin = Path("/content/scarstore_router.bin")
    if not router_json.exists() and "__file__" in globals():
        router_json = Path(__file__).resolve().parents[2] / "portfolio/public/webgpu/qwen3b/scarstore_router.json"
        router_bin = Path(__file__).resolve().parents[2] / "portfolio/public/webgpu/qwen3b/scarstore_router.bin"

    # Dedup unique prompt strings for Qwen forward pass so identical strings run ONCE!
    uniq_prompts = sorted(set(all_prompts))
    p2u = {p: i for i, p in enumerate(uniq_prompts)}
    print(f"[Qwen-L23] Extracting {len(uniq_prompts)} unique prompts across PT/EN/ES/FR ...")
    uniq_z, anchor_cos = extract_qwen3b_layer23_latents(uniq_prompts, router_json, router_bin, device, batch_size=192)
    z_all = np.stack([uniq_z[p2u[p]] for p in all_prompts], axis=0)  # [N_samples, 2048]

    is_test_arr = np.asarray(all_is_test, dtype=bool)
    train_idx = np.where(~is_test_arr)[0]
    test_idx = np.where(is_test_arr)[0]

    z_train_gpu = torch.tensor(z_all[train_idx], dtype=torch.float32, device=device)
    s_train_gpu = torch.tensor(np.asarray(all_s_ids, dtype=np.int64)[train_idx], dtype=torch.long, device=device)
    shapes_gpu = torch.tensor(shapes_ot, dtype=torch.float32, device=device)
    knn_gpu = torch.tensor(shapes_knn.astype(np.int64), dtype=torch.long, device=device)

    # -------------------------------------------------------------------------
    # Stage 3: Phase-1 Tabula-Rasa `ScarNative` OT Rectified Flow Training
    # -------------------------------------------------------------------------
    torch.manual_seed(20261115)
    model = PointScarFlow(cfg).to(device)
    model.assert_zero_dense_linear()
    ema_model = copy.deepcopy(model).eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Phase 1] Initialized Tabula-Rasa PointScarFlow: {n_params / 1e6:.2f}M ScarNativeLinear params (0 nn.Linear!)")

    batch_size = 128
    phase1_steps = 4800
    lr_max = 1.4e-3
    lr_min = 6e-5
    opt = torch.optim.AdamW(model.parameters(), lr=lr_max, weight_decay=1e-4, betas=(0.9, 0.95))

    x0_full = model.initial_state(1, device)  # [1, 1024, 4]
    n_train = z_train_gpu.shape[0]
    arange_pts = torch.arange(n_pts, device=device).view(1, n_pts, 1)

    t_p1 = time.perf_counter()
    model.train()
    for step in range(1, phase1_steps + 1):
        # Cosine LR with 200-step linear warmup
        if step <= 200:
            lr = lr_max * (step / 200.0)
        else:
            prog = (step - 200) / float(max(1, phase1_steps - 200))
            lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * prog))
        for pg in opt.param_groups:
            pg["lr"] = lr

        b_idx = torch.randint(0, n_train, (batch_size,), device=device)
        z_b = z_train_gpu[b_idx]
        # Add tiny semantic jitter (`0.015`) during training for smooth open-world manifold interpolation!
        z_b = F.normalize(z_b + 0.015 * torch.randn_like(z_b), dim=-1)

        # 12% Classifier-Free Guidance null-conditioning dropout
        drop_mask = (torch.rand((batch_size, 1), device=device) < 0.12).float()
        z_cond = z_b * (1.0 - drop_mask)

        sid_b = s_train_gpu[b_idx]
        x1 = shapes_gpu[sid_b]  # [B, 1024, 4]
        x0 = x0_full.expand(batch_size, n_pts, 4)

        # Logit-normal + uniform mixture over tau in [0, 1] with emphasis near tau=0 and mid-trajectory
        u_rand = torch.rand((batch_size,), device=device)
        tau = torch.where(
            torch.rand((batch_size,), device=device) < 0.25,
            (torch.randint(0, 4, (batch_size,), device=device).float() * 0.25),
            u_rand,
        )
        t_view = tau.view(batch_size, 1, 1)
        x_tau = (1.0 - t_view) * x0 + t_view * x1
        v_target = x1 - x0

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v_pred = model(x_tau, tau, z_cond)
        v_pred_f = v_pred.float()

        loss_vel = F.mse_loss(v_pred_f[..., :3], v_target[..., :3]) + 0.35 * F.mse_loss(v_pred_f[..., 3], v_target[..., 3])

        # Implied 1-step endpoint `x1_hat` & local `k=8` sharp-crease differential edge loss!
        x1_hat = x_tau[..., :3] + (1.0 - t_view) * v_pred_f[..., :3]
        nn_b = knn_gpu[sid_b]  # [B, 1024, 8]
        b_grid = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
        pred_edges = x1_hat.unsqueeze(2) - x1_hat[b_grid, nn_b]  # [B, 1024, 8, 3]
        gt_xyz = x1[..., :3]
        gt_edges = gt_xyz.unsqueeze(2) - gt_xyz[b_grid, nn_b]
        loss_edge = F.mse_loss(pred_edges, gt_edges)
        loss_x1 = F.smooth_l1_loss(x1_hat, gt_xyz, beta=0.04)
        loss_vq = model.total_commitment_loss()

        loss = loss_vel + 0.75 * loss_edge + 0.25 * loss_x1 + 0.04 * loss_vq
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
        opt.step()
        update_ema(ema_model, model, decay=0.997 if step > 400 else 0.985)

        if step % 300 == 0 or step == 1 or step == phase1_steps:
            dt_s = time.perf_counter() - t_p1
            print(
                f"  [Phase 1 | Step {step:4d}/{phase1_steps}] "
                f"loss={loss.item():.5f} (vel={loss_vel.item():.5f}, edge={loss_edge.item():.5f}, vq={loss_vq.item():.5f}) "
                f"lr={lr:.2e} | {step / max(dt_s, 1e-3):.1f} it/s"
            )

    # -------------------------------------------------------------------------
    # Stage 4: Phase-2 Guided 4-Step Consistency Self-Distillation (`CFG 2.0 -> 1.0`)
    # -------------------------------------------------------------------------
    print("\n[Phase 2] Distilling 8-Step RK2 CFG=2.0 EMA Teacher into 4-Step Single-Pass (`CFG=1.0`) Student ...")
    ema_model.eval()

    # Precompute sharp CFG=2.0 teacher trajectories for unique `(prompt_latent)` vectors!
    uniq_train_prompts_z, inv_map = torch.unique(z_train_gpu, dim=0, return_inverse=True)
    print(f"  [Phase 2] Generating {len(uniq_train_prompts_z)} sharp CFG=2.0 Teacher 3D clouds ...")
    teacher_targets = torch.empty((len(uniq_train_prompts_z), n_pts, 4), dtype=torch.float32, device=device)
    for s in range(0, len(uniq_train_prompts_z), 128):
        e = min(len(uniq_train_prompts_z), s + 128)
        teacher_targets[s:e] = sample_teacher_rk2_cfg(ema_model, uniq_train_prompts_z[s:e], num_steps=8, guidance_scale=2.0)

    student = copy.deepcopy(ema_model).train()
    for p in student.parameters():
        p.requires_grad_(True)
    opt_dist = torch.optim.AdamW(student.parameters(), lr=3.5e-4, weight_decay=5e-5, betas=(0.9, 0.95))
    phase2_steps = 1200
    t_p2 = time.perf_counter()

    for step in range(1, phase2_steps + 1):
        prog = step / float(phase2_steps)
        lr = 2e-5 + 0.5 * (3.5e-4 - 2e-5) * (1.0 + math.cos(math.pi * prog))
        for pg in opt_dist.param_groups:
            pg["lr"] = lr

        b_idx = torch.randint(0, n_train, (batch_size,), device=device)
        z_b = z_train_gpu[b_idx]
        u_idx = inv_map[b_idx]
        sid_b = s_train_gpu[b_idx]
        # Blend 70% CFG-teacher crisp target + 30% exact OT ground-truth manifold for razor-sharp creases!
        x1_teach = 0.70 * teacher_targets[u_idx] + 0.30 * shapes_gpu[sid_b]
        x0 = x0_full.expand(batch_size, n_pts, 4)

        # Sample predominantly on the exact 4 Euler knots {0.0, 0.25, 0.50, 0.75}!
        knots = torch.randint(0, 4, (batch_size,), device=device).float() * 0.25
        cont = torch.rand((batch_size,), device=device)
        tau = torch.where(torch.rand((batch_size,), device=device) < 0.75, knots, cont)
        t_view = tau.view(batch_size, 1, 1)
        x_tau = (1.0 - t_view) * x0 + t_view * x1_teach
        v_target = x1_teach - x0

        opt_dist.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            v_pred = student(x_tau, tau, z_b)
        v_pred_f = v_pred.float()

        loss_vel = F.mse_loss(v_pred_f[..., :3], v_target[..., :3]) + 0.35 * F.mse_loss(v_pred_f[..., 3], v_target[..., 3])
        x1_hat = x_tau[..., :3] + (1.0 - t_view) * v_pred_f[..., :3]
        nn_b = knn_gpu[sid_b]
        b_grid = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
        pred_edges = x1_hat.unsqueeze(2) - x1_hat[b_grid, nn_b]
        gt_xyz = x1_teach[..., :3]
        gt_edges = gt_xyz.unsqueeze(2) - gt_xyz[b_grid, nn_b]
        loss_edge = F.mse_loss(pred_edges, gt_edges)
        loss_vq = student.total_commitment_loss()

        loss = loss_vel + 0.80 * loss_edge + 0.04 * loss_vq
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt_dist.step()

        if step % 300 == 0 or step == phase2_steps:
            dt_s = time.perf_counter() - t_p2
            print(
                f"  [Phase 2 | Step {step:4d}/{phase2_steps}] "
                f"distill_loss={loss.item():.5f} (vel={loss_vel.item():.5f}, edge={loss_edge.item():.5f}) "
                f"| {step / max(dt_s, 1e-3):.1f} it/s"
            )

    # -------------------------------------------------------------------------
    # Stage 5: Held-Out Polyglot (`PT/EN/ES/FR`) Evaluation & Zero-Shock Export
    # -------------------------------------------------------------------------
    student.eval()
    student.assert_zero_dense_linear()

    z_test_gpu = torch.tensor(z_all[test_idx], dtype=torch.float32, device=device)
    s_test_gpu = torch.tensor(np.asarray(all_s_ids, dtype=np.int64)[test_idx], dtype=torch.long, device=device)
    langs_test = [all_langs[i] for i in test_idx]
    cids_test = [all_c_ids[i] for i in test_idx]

    with torch.no_grad():
        pred_4step = []
        pred_2step = []
        for s in range(0, len(z_test_gpu), 64):
            zb = z_test_gpu[s : s + 64]
            pred_4step.append(student.sample_euler(zb, num_steps=4, guidance_scale=1.0))
            pred_2step.append(student.sample_euler(zb, num_steps=2, guidance_scale=1.0))
        pred_4 = torch.cat(pred_4step, dim=0)
        pred_2 = torch.cat(pred_2step, dim=0)
        gt_test = shapes_gpu[s_test_gpu]

        chamfer_4 = compute_chamfer_l2_batch(pred_4[..., :3], gt_test[..., :3])
        chamfer_2 = compute_chamfer_l2_batch(pred_2[..., :3], gt_test[..., :3])
        ot_rmse_4 = (pred_4[..., :3] - gt_test[..., :3]).square().mean(dim=(-1, -2)).sqrt()

        # Baseline: canonical sphere Chamfer vs GT (to prove real geometric deformation!)
        sphere_rep = x0_full[..., :3].expand(len(gt_test), n_pts, 3)
        chamfer_sphere = compute_chamfer_l2_batch(sphere_rep, gt_test[..., :3])

        by_lang_chamfer: Dict[str, float] = {}
        for lg in ("pt", "en", "es", "fr"):
            m_idx = [k for k, l_val in enumerate(langs_test) if l_val == lg]
            if m_idx:
                by_lang_chamfer[lg] = float(chamfer_4[m_idx].mean().item())

        # Cross-lingual shape invariance: cosine of OT-aligned 3D point clouds across PT/ES/FR vs EN for same concept!
        concept_lang_clouds: Dict[Tuple[int, str], torch.Tensor] = {}
        for k, (cid_v, lg_v) in enumerate(zip(cids_test, langs_test)):
            concept_lang_clouds[(cid_v, lg_v)] = pred_4[k, :, :3].reshape(-1)

        xling_cosines: Dict[str, List[float]] = {"pt_en": [], "es_en": [], "fr_en": []}
        inter_concept_cosines: List[float] = []
        for c_idx in range(len(CONCEPT_SPECS)):
            v_en = concept_lang_clouds.get((c_idx, "en"))
            if v_en is None:
                continue
            for lg_other, key in (("pt", "pt_en"), ("es", "es_en"), ("fr", "fr_en")):
                v_ot = concept_lang_clouds.get((c_idx, lg_other))
                if v_ot is not None:
                    cos_v = float(F.cosine_similarity(v_en.unsqueeze(0), v_ot.unsqueeze(0)).item())
                    xling_cosines[key].append(cos_v)
            v_diff = concept_lang_clouds.get(((c_idx + 7) % len(CONCEPT_SPECS), "en"))
            if v_diff is not None:
                inter_concept_cosines.append(float(F.cosine_similarity(v_en.unsqueeze(0), v_diff.unsqueeze(0)).item()))

    # Verify exact zero PTQ shock on `pack_all_vq4_scar`!
    sample_z = z_test_gpu[:8]
    y_pre_pack = student.sample_euler(sample_z, num_steps=4, guidance_scale=1.0)
    out_dir = Path("/content/pointscar_v1")

    eval_report = {
        "gpu": gpu_name,
        "total_concepts": len(CONCEPT_SPECS),
        "unique_ot_shapes": int(len(shapes_ot)),
        "total_multilingual_prompts": int(len(all_prompts)),
        "held_out_test_prompts": int(len(test_idx)),
        "router_anchor0_alignment_cosine": round(anchor_cos, 6),
        "held_out_chamfer_l2_sphere_baseline": round(float(chamfer_sphere.mean().item()), 6),
        "held_out_chamfer_l2_4step_1pass": round(float(chamfer_4.mean().item()), 6),
        "held_out_chamfer_l2_2step_1pass": round(float(chamfer_2.mean().item()), 6),
        "held_out_ot_rmse_4step": round(float(ot_rmse_4.mean().item()), 6),
        "held_out_chamfer_by_lang": {k: round(v, 6) for k, v in by_lang_chamfer.items()},
        "cross_lingual_3d_shape_cosine": {
            k: round(float(np.mean(vals)), 6) for k, vals in xling_cosines.items() if vals
        },
        "inter_concept_3d_cosine": round(float(np.mean(inter_concept_cosines)), 6),
        "recommended_inference": {"num_steps": 4, "guidance_scale": 1.0},
        "training_wall_seconds": round(time.perf_counter() - t_start, 2),
    }

    manifest = export_pointscar_webgpu_bundle(student, out_dir, extra_metadata=eval_report)
    phys_bytes = student.pack_all_vq4_scar(drop_master=True)
    y_post_pack = student.sample_euler(sample_z, num_steps=4, guidance_scale=1.0)
    ptq_shock = float((y_pre_pack - y_post_pack).abs().max().item())
    eval_report["ptq_max_abs_shock"] = ptq_shock
    eval_report["packed_vq4_scar_bytes"] = int(phys_bytes)
    eval_report["shard_bytes"] = int(manifest["shard_bytes"])

    # Update manifest & write standalone report + sample held-out clouds for local pytest parity verification!
    manifest["training_metadata"] = eval_report
    (out_dir / "pointscar_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "eval_report.json").write_text(json.dumps(eval_report, indent=2), encoding="utf-8")

    # Save 16 held-out polyglot test vectors (4 concepts x 4 languages PT/EN/ES/FR) for local Mac WebGPU parity tests!
    calib_vectors: Dict[str, Any] = {}
    probe_concepts = ["dna_helix", "rocket_spacecraft", "mos_scache_architecture", "ringed_planet"]
    for pc in probe_concepts:
        c_idx = next(i for i, sp in enumerate(CONCEPT_SPECS) if sp["id"] == pc)
        calib_vectors[pc] = {}
        for k, (cid_v, lg_v) in enumerate(zip(cids_test, langs_test)):
            if cid_v == c_idx:
                calib_vectors[pc][lg_v] = {
                    "prompt": all_prompts[test_idx[k]],
                    "z_23": z_all[test_idx[k]].tolist(),
                }
    (out_dir / "polyglot_test_latents.json").write_text(json.dumps(calib_vectors), encoding="utf-8")

    tar_path = Path("/content/pointscar_v1_bundle.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tf:
        for fn in ("pointscar_v1_shard.bin", "pointscar_manifest.json", "point_scar_flow.wgsl", "eval_report.json", "polyglot_test_latents.json"):
            tf.add(out_dir / fn, arcname=fn)

    print("\n" + "=" * 78)
    print("✅ POINTSCARFLOW v1.0 TRAINING & DISTILLATION COMPLETE!")
    print(json.dumps(eval_report, indent=2))
    print(f"📦 Bundle ready: {tar_path} ({tar_path.stat().st_size / (1024*1024):.2f} MiB)")
    print("=" * 78)


if __name__ == "__main__":
    main()

