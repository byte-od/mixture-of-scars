"""Elastic Capacity Gearshift & Geodesic Mixture-of-Scars (`MoS`) Ternary Specialist Banks (`v9.1-Elastic`).

Implements Zero-Tax Runtime Capacity Switching for `ScarField-Edge` & `ScarField-Core`:
  1. `ElasticCapacityGear`: Switches active symplectic macro-cycles (`1..N`), `S1/S2` geodesic sub-steps,
     perceptual surprise gating, and `MoS` specialist `top_k` mid-session with `Delta = 0 B` change to
     `d_bus` or `KalmanBusState` (`0.0` episodic state loss).
  2. `MoSSymplecticDomainSpecialist` & `MoSGeodesicDomainBank`: Modular `2-bit` ternary (`{-1, 0, +1}`)
     symplectic Cayley geodesic specialists (`"code"`, `"math"`, `"science"`, `"polyglot"`, ...) mounted
     onto the shared `1.000 bpw` Gosset `E_8` crystal:
       - Zero legacy `nn.Linear`, `RMSNorm`, `ReLU`, or `Softmax`.
       - Exact `||h||_2 == 1.000000` unit-sphere preservation via `cayley_woodbury_step`.
       - Exact `0.000000` attach perturbation at `step = 0` (`gamma = 0` -> `V = 0` -> `Phi_0(h) == h`)
         WITH non-zero `step = 1` gradient (`U != 0` asymmetric Lie-algebra saddle breaker!).
       - Exact `0.000000` cross-domain leakage outside each specialist's `C^1` spherical horizon cap.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from scar_codec.binary_format import ScarFieldMmapContainer, save_scarfield_v7_container
from scar_field.cayley_geodesic import c1_spherical_cap_resonance, cayley_woodbury_step
from scar_field.e8_lattice import (
    pack_ternary_2bit,
    project_rows_to_unit_sphere,
    quantize_ternary_ste,
    unpack_ternary_2bit,
)


@dataclass(frozen=True)
class ElasticCapacityGear:
    """Zero-tax runtime compute gear (`d_bus` and `KalmanBusState` remain 100% invariant across gears)."""

    name: str
    active_macro_cycles: int
    s1_steps: int
    s2_steps: int
    enable_percept_s1_gate: bool
    mos_top_k: int = 1

    @classmethod
    def from_preset(cls, preset: str, max_macro_cycles: int) -> "ElasticCapacityGear":
        key = preset.strip().lower()
        c_max = max(1, int(max_macro_cycles))
        if key == "eco":
            return cls(
                name="eco",
                active_macro_cycles=1,
                s1_steps=1,
                s2_steps=1,
                enable_percept_s1_gate=True,
                mos_top_k=1,
            )
        if key == "balanced":
            return cls(
                name="balanced",
                active_macro_cycles=min(2, c_max),
                s1_steps=1,
                s2_steps=4,
                enable_percept_s1_gate=True,
                mos_top_k=1,
            )
        if key == "deep":
            return cls(
                name="deep",
                active_macro_cycles=c_max,
                s1_steps=1,
                s2_steps=8,
                enable_percept_s1_gate=False,
                mos_top_k=2,
            )
        if key == "frontier":
            return cls(
                name="frontier",
                active_macro_cycles=c_max,
                s1_steps=2,
                s2_steps=12,
                enable_percept_s1_gate=False,
                mos_top_k=2,
            )
        raise ValueError(f"Unknown ElasticCapacityGear preset `{preset}` (expected eco|balanced|deep|frontier)")


def _stiefel_unit_rows(rows: int, cols: int, generator: torch.Generator, device: torch.device | str = "cpu") -> torch.Tensor:
    raw = torch.randn((rows, cols), generator=generator, dtype=torch.float32)
    if rows <= cols:
        q, _ = torch.linalg.qr(raw.T, mode="reduced")
        out = q.T.contiguous()
    else:
        q, _ = torch.linalg.qr(raw, mode="reduced")
        out = project_rows_to_unit_sphere(q)
    return out.to(device=device)


class MoSSymplecticDomainSpecialist(nn.Module):
    """Single `2-bit` Ternary Symplectic Geodesic Domain Specialist on `S^{D_f-1} x S^{D_w-1}`."""

    def __init__(
        self,
        domain_name: str,
        d_form: int,
        d_world: int,
        d_bus: int,
        scar_rank: int = 8,
        horizon_kappa: float = 0.28,
        seed: int = 2026091,
        init_pole: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if scar_rank <= 0 or (scar_rank % 4 != 0):
            raise ValueError(f"scar_rank ({scar_rank}) must be > 0 and divisible by 4 for 2-bit ternary packing")
        if d_form % 4 != 0 or d_world % 4 != 0 or (d_form + d_bus) % 4 != 0:
            raise ValueError("d_form, d_world, and (d_form + d_bus) must be divisible by 4")

        self.domain_name = str(domain_name)
        self.d_form = int(d_form)
        self.d_world = int(d_world)
        self.d_bus = int(d_bus)
        self.joint_ctx_dim = int(d_form + d_bus)
        self.scar_rank = int(scar_rank)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))

        if init_pole is not None:
            pole_init = project_rows_to_unit_sphere(init_pole.detach().float().reshape(1, d_world)).squeeze(0)
        else:
            pole_init = project_rows_to_unit_sphere(torch.randn((1, d_world), generator=gen)).squeeze(0)

        self.domain_pole = nn.Parameter(pole_init)
        self.horizon_kappa = nn.Parameter(torch.tensor(float(horizon_kappa), dtype=torch.float32))

        # Non-zero Stiefel-orthonormal ternary planes (`U != 0`, `A != 0`, `B != 0`) paired with
        # `gamma_form = 0.0` & `gamma_world = 0.0` at `step = 0` -> exact `0.000000` attach shock
        # AND non-zero `dL/d(gamma)` immediately on backward `step = 1`!
        self.u_form_scar = nn.Parameter(_stiefel_unit_rows(scar_rank, d_form, gen))
        self.a_form_scar = nn.Parameter(_stiefel_unit_rows(scar_rank, d_world, gen))
        self.b_form_scar = nn.Parameter(_stiefel_unit_rows(d_form, scar_rank, gen))
        self.gamma_form = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

        self.u_world_scar = nn.Parameter(_stiefel_unit_rows(scar_rank, d_world, gen))
        self.a_world_scar = nn.Parameter(_stiefel_unit_rows(scar_rank, self.joint_ctx_dim, gen))
        self.b_world_scar = nn.Parameter(_stiefel_unit_rows(d_world, scar_rank, gen))
        self.gamma_world = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    @property
    def packed_ternary_bytes(self) -> int:
        n_tern = (
            self.u_form_scar.numel()
            + self.a_form_scar.numel()
            + self.b_form_scar.numel()
            + self.u_world_scar.numel()
            + self.a_world_scar.numel()
            + self.b_world_scar.numel()
        )
        pole_fp16 = self.d_world * 2
        scalars_f32 = 3 * 4  # kappa, gamma_form, gamma_world
        return (n_tern // 4) + pole_fp16 + scalars_f32

    def compute_resonance(self, h_world_2d: torch.Tensor) -> torch.Tensor:
        """Returns `C^1` spherical-cap resonance `g_d in [0, 1]` (`[N]`, exact `0.0` outside horizon)."""
        pole = project_rows_to_unit_sphere(self.domain_pole.unsqueeze(0)).squeeze(0)
        kap = self.horizon_kappa.clamp(-0.20, 0.92)
        cos_sim = torch.matmul(h_world_2d.float(), pole)  # [N]
        return c1_spherical_cap_resonance(cos_sim, kap)

    def forward_active_rows(
        self,
        hf_sel: torch.Tensor,
        hw_sel: torch.Tensor,
        bc_sel: torch.Tensor,
        gate_sel: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies exact `cayley_woodbury_step` on selected active rows `[M, D]` (`||out||_2 == 1.0`)."""
        m_rows = hf_sel.shape[0]
        if m_rows == 0:
            return hf_sel, hw_sel

        r = self.scar_rank
        scale_f = 1.0 / math.sqrt(float(self.d_world * r))
        scale_w = 1.0 / math.sqrt(float(self.joint_ctx_dim * r))

        uf_q, _ = quantize_ternary_ste(project_rows_to_unit_sphere(self.u_form_scar))   # [r, D_f]
        af_q, _ = quantize_ternary_ste(project_rows_to_unit_sphere(self.a_form_scar))   # [r, D_w]
        bf_q, _ = quantize_ternary_ste(project_rows_to_unit_sphere(self.b_form_scar))   # [D_f, r]

        uw_q, _ = quantize_ternary_ste(project_rows_to_unit_sphere(self.u_world_scar))  # [r, D_w]
        aw_q, _ = quantize_ternary_ste(project_rows_to_unit_sphere(self.a_world_scar))  # [r, D_f+D_b]
        bw_q, _ = quantize_ternary_ste(project_rows_to_unit_sphere(self.b_world_scar))  # [D_w, r]

        u_f = project_rows_to_unit_sphere(uf_q).T.unsqueeze(0).expand(m_rows, self.d_form, r)
        coef_f = torch.matmul(hw_sel.float(), af_q.T) * (gate_sel.unsqueeze(-1) * self.gamma_form * scale_f)  # [M, r]
        v_f = bf_q.unsqueeze(0) * coef_f.unsqueeze(1)  # [M, D_f, r]
        hf_next = cayley_woodbury_step(hf_sel, u_f, v_f, dt=1.0)

        joint_ctx = project_rows_to_unit_sphere(torch.cat((hf_next.float(), bc_sel.float()), dim=-1))
        u_w = project_rows_to_unit_sphere(uw_q).T.unsqueeze(0).expand(m_rows, self.d_world, r)
        coef_w = torch.matmul(joint_ctx, aw_q.T) * (gate_sel.unsqueeze(-1) * self.gamma_world * scale_w)  # [M, r]
        v_w = bw_q.unsqueeze(0) * coef_w.unsqueeze(1)  # [M, D_w, r]
        hw_next = cayley_woodbury_step(hw_sel, u_w, v_w, dt=1.0)
        return hf_next.to(hf_sel.dtype), hw_next.to(hw_sel.dtype)

    @torch.no_grad()
    def export_packed_arrays(self) -> Dict[str, np.ndarray]:
        pfx = f"mos::{self.domain_name}"
        out: Dict[str, np.ndarray] = {}
        for tag, param in (
            ("u_form", self.u_form_scar),
            ("a_form", self.a_form_scar),
            ("b_form", self.b_form_scar),
            ("u_world", self.u_world_scar),
            ("a_world", self.a_world_scar),
            ("b_world", self.b_world_scar),
        ):
            _, q_i8 = quantize_ternary_ste(project_rows_to_unit_sphere(param.detach().float()))
            out[f"{pfx}::{tag}::u8"] = pack_ternary_2bit(q_i8).cpu().numpy()
            out[f"{pfx}::{tag}::shape"] = np.asarray(list(param.shape), dtype=np.int32)
        out[f"{pfx}::pole"] = (
            project_rows_to_unit_sphere(self.domain_pole.detach().float().unsqueeze(0))
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float16)
        )
        out[f"{pfx}::scalars"] = np.asarray(
            [
                float(self.horizon_kappa.item()),
                float(self.gamma_form.item()),
                float(self.gamma_world.item()),
            ],
            dtype=np.float32,
        )
        return out

    @torch.no_grad()
    def load_packed_arrays_(self, arrays: Dict[str, Any]) -> None:
        pfx = f"mos::{self.domain_name}"
        for tag, param in (
            ("u_form", self.u_form_scar),
            ("a_form", self.a_form_scar),
            ("b_form", self.b_form_scar),
            ("u_world", self.u_world_scar),
            ("a_world", self.a_world_scar),
            ("b_world", self.b_world_scar),
        ):
            u8_t = torch.from_numpy(np.array(arrays[f"{pfx}::{tag}::u8"], copy=True))
            shp = tuple(int(x) for x in arrays[f"{pfx}::{tag}::shape"])
            unp = unpack_ternary_2bit(u8_t).reshape(shp).to(device=param.device, dtype=param.dtype)
            param.copy_(project_rows_to_unit_sphere(unp))
        pole_t = torch.from_numpy(np.array(arrays[f"{pfx}::pole"], copy=True)).to(
            device=self.domain_pole.device, dtype=self.domain_pole.dtype
        )
        self.domain_pole.copy_(project_rows_to_unit_sphere(pole_t.unsqueeze(0)).squeeze(0))
        sc = arrays[f"{pfx}::scalars"]
        self.horizon_kappa.fill_(float(sc[0]))
        self.gamma_form.fill_(float(sc[1]))
        self.gamma_world.fill_(float(sc[2]))


class MoSGeodesicDomainBank(nn.Module):
    """On-Demand Mixture-of-Scars (`MoS`) Specialist Bank with `C^1` Spherical Auto-Routing."""

    def __init__(self, d_form: int, d_world: int, d_bus: int, top_k: int = 1) -> None:
        super().__init__()
        self.d_form = int(d_form)
        self.d_world = int(d_world)
        self.d_bus = int(d_bus)
        self.top_k = max(1, int(top_k))
        self.specialists = nn.ModuleDict()
        self.pinned_domains: Optional[Tuple[str, ...]] = None
        self.enabled: bool = True
        self.last_domain_activity: Dict[str, float] = {}

    def register_specialist(
        self,
        domain_name: str,
        scar_rank: int = 8,
        horizon_kappa: float = 0.28,
        seed: int = 2026091,
        init_pole: Optional[torch.Tensor] = None,
    ) -> MoSSymplecticDomainSpecialist:
        spec = MoSSymplecticDomainSpecialist(
            domain_name=domain_name,
            d_form=self.d_form,
            d_world=self.d_world,
            d_bus=self.d_bus,
            scar_rank=scar_rank,
            horizon_kappa=horizon_kappa,
            seed=seed,
            init_pole=init_pole,
        )
        dev = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device("cpu")
        self.specialists[domain_name] = spec.to(dev)
        return spec

    def unmount_specialist(self, domain_name: str) -> bool:
        if domain_name in self.specialists:
            del self.specialists[domain_name]
            if self.pinned_domains is not None:
                rem = tuple(d for d in self.pinned_domains if d != domain_name)
                self.pinned_domains = rem if rem else None
            return True
        return False

    def set_pinned_domains(self, domains: Optional[Iterable[str]]) -> None:
        self.pinned_domains = tuple(str(d) for d in domains) if domains is not None else None

    @property
    def total_packed_bytes(self) -> int:
        return sum(spec.packed_ternary_bytes for spec in self.specialists.values())  # type: ignore[attr-defined]

    def forward(
        self,
        h_form: torch.Tensor,
        h_world: torch.Tensor,
        bus_ctx: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if (not self.enabled) or len(self.specialists) == 0:
            return h_form, h_world

        if self.pinned_domains is not None:
            cand_names = [d for d in self.pinned_domains if d in self.specialists]
        else:
            cand_names = list(self.specialists.keys())
        if not cand_names:
            return h_form, h_world

        orig_shape = h_form.shape[:-1]
        n_rows = int(h_form.numel() // self.d_form)
        hf_2d = h_form.reshape(n_rows, self.d_form)
        hw_2d = h_world.reshape(n_rows, self.d_world)
        bc_2d = bus_ctx.reshape(n_rows, self.d_bus)

        gates_list: List[torch.Tensor] = []
        for d_name in cand_names:
            spec: MoSSymplecticDomainSpecialist = self.specialists[d_name]  # type: ignore[assignment]
            gates_list.append(spec.compute_resonance(hw_2d))
        gates_mat = torch.stack(gates_list, dim=-1)  # [N, D_cand]

        k_eff = min(self.top_k, len(cand_names))
        if k_eff < len(cand_names):
            top_vals, top_idx = torch.topk(gates_mat, k=k_eff, dim=-1)
            keep_mask = torch.zeros_like(gates_mat)
            keep_mask.scatter_(1, top_idx, 1.0)
            routed_gates = gates_mat * keep_mask
        else:
            routed_gates = gates_mat

        hf_cur = hf_2d
        hw_cur = hw_2d
        act_stats: Dict[str, float] = {}
        for d_idx, d_name in enumerate(cand_names):
            spec = self.specialists[d_name]  # type: ignore[assignment]
            g_col = routed_gates[:, d_idx]  # [N]
            if not (torch.is_grad_enabled() and (g_col.requires_grad or hf_cur.requires_grad)):
                act_idx = torch.nonzero(g_col > 0.0, as_tuple=False).squeeze(-1)
                act_stats[d_name] = float(act_idx.numel()) / float(max(1, n_rows))
                if act_idx.numel() == 0:
                    continue
                hf_sub, hw_sub = spec.forward_active_rows(
                    hf_cur[act_idx], hw_cur[act_idx], bc_2d[act_idx], g_col[act_idx]
                )
                hf_next = hf_cur.clone()
                hw_next = hw_cur.clone()
                hf_next[act_idx] = hf_sub
                hw_next[act_idx] = hw_sub
                hf_cur, hw_cur = hf_next, hw_next
            else:
                with torch.no_grad():
                    has_any = bool((g_col.detach() > 0.0).any().item()) if g_col.device.type == "cpu" else True
                    if g_col.device.type == "cpu":
                        act_stats[d_name] = float((g_col.detach() > 0.0).float().mean().item())
                if not has_any:
                    continue
                hf_cur, hw_cur = spec.forward_active_rows(hf_cur, hw_cur, bc_2d, g_col)

        if act_stats:
            self.last_domain_activity = act_stats
        return hf_cur.reshape(*orig_shape, self.d_form), hw_cur.reshape(*orig_shape, self.d_world)

    @torch.no_grad()
    def save_packed_mos_container(self, path: Union[str, Path]) -> int:
        meta = {
            "kind": "mos_geodesic_domain_bank_v9_1",
            "d_form": self.d_form,
            "d_world": self.d_world,
            "d_bus": self.d_bus,
            "top_k": self.top_k,
            "domains": [
                {"name": name, "scar_rank": int(spec.scar_rank)}  # type: ignore[attr-defined]
                for name, spec in self.specialists.items()
            ],
        }
        arrays: Dict[str, np.ndarray] = {}
        for spec in self.specialists.values():
            arrays.update(spec.export_packed_arrays())  # type: ignore[attr-defined]
        return save_scarfield_v7_container(meta, arrays, path)

    @classmethod
    @torch.no_grad()
    def load_packed_mos_container(cls, path: Union[str, Path], verify_crc: bool = True) -> "MoSGeodesicDomainBank":
        ct = ScarFieldMmapContainer(path, verify_crc=verify_crc)
        cfg = ct.config
        bank = cls(
            d_form=int(cfg["d_form"]),
            d_world=int(cfg["d_world"]),
            d_bus=int(cfg["d_bus"]),
            top_k=int(cfg.get("top_k", 1)),
        )
        for d_meta in cfg.get("domains", []):
            spec = bank.register_specialist(
                domain_name=str(d_meta["name"]),
                scar_rank=int(d_meta["scar_rank"]),
            )
            spec.load_packed_arrays_(ct.arrays)
        ct.close()
        return bank
