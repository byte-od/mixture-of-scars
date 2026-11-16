"""Riemannian Tangent-Space Muon Optimizer (`Newton-Schulz` on `S^{I-1}`, Zero `AdamW`).

Eliminates `AdamW`'s Euclidean per-coordinate second-moment tensor `v_t` (`50%` lower optimizer
memory footprint!) by projecting 2D matrix gradients onto each unit row's tangent space
`T_w S^{I-1}`, applying 5 quintic Newton-Schulz orthogonalization steps, and retracting cleanly
back onto the unit hypersphere `||w_o||_2 == 1.000000`.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch.optim import Optimizer

from scar_field.e8_lattice import E8SphericalLinear, project_rows_to_unit_sphere


def _newton_schulz_orthogonalize(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Fast 5-step quintic Newton-Schulz semi-orthogonalization of a 2D matrix `g`."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    x = g.float()
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    norm = torch.linalg.matrix_norm(x, ord="fro").clamp(min=eps)
    x = x / norm
    for _ in range(steps):
        xx_t = x @ x.T
        b_mat = b * xx_t + c * (xx_t @ xx_t)
        x = a * x + b_mat @ x
    if transposed:
        x = x.T
    return x.to(g.dtype)


class RiemannianMuon(Optimizer):
    """Single-buffer Riemannian Orthogonalized Momentum optimizer for `ScarField v7.0`."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1.5e-2,
        scalar_lr: float = 5e-3,
        momentum: float = 0.92,
        ns_steps: int = 5,
    ) -> None:
        defaults = dict(lr=lr, scalar_lr=scalar_lr, momentum=momentum, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr: float = group["lr"]
            scalar_lr: float = group["scalar_lr"]
            mom: float = group["momentum"]
            ns_steps: int = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p)
                buf: torch.Tensor = st["momentum_buffer"]

                if p.ndim == 2 and p.shape[1] >= 4:
                    # 1. Project gradient onto tangent space of unit sphere rows `g_perp = g - <g, w> w`
                    w_u = project_rows_to_unit_sphere(p)
                    g_perp = g - (g * w_u).sum(dim=-1, keepdim=True) * w_u

                    # 2. Transport momentum into current tangent plane & apply Nesterov lookahead
                    buf.mul_(mom).add_(g_perp)
                    buf.sub_((buf * w_u).sum(dim=-1, keepdim=True) * w_u)
                    u_nest = g_perp + mom * buf

                    # 3. Semi-orthogonalize via quintic Newton-Schulz + unit-row step normalization
                    u_orth = _newton_schulz_orthogonalize(u_nest, steps=ns_steps)
                    u_orth = u_orth - (u_orth * w_u).sum(dim=-1, keepdim=True) * w_u
                    row_rms = torch.linalg.vector_norm(u_orth, ord=2, dim=-1, keepdim=True).clamp(min=1e-8)
                    step_dir = u_orth / row_rms

                    # 4. Exact spherical retraction `w <- Norm(w - lr * step_dir)`
                    p.copy_(project_rows_to_unit_sphere(w_u - lr * step_dir))
                else:
                    # 1D / scalar parameters: RMS-normalized Nesterov update (`1` state buffer, zero `v_t`!)
                    buf.mul_(mom).add_(g)
                    u_s = g + mom * buf
                    denom = torch.sqrt(torch.mean(u_s * u_s).clamp(min=1e-12))
                    p.add_(u_s / denom, alpha=-scalar_lr)

        return loss


@torch.no_grad()
def heal_all_e8_scars(module: torch.nn.Module) -> int:
    """Triggers closed-form `SVD/RRR` ternary scar healing across all `E8SphericalLinear` layers."""
    count = 0
    for m in module.modules():
        if isinstance(m, E8SphericalLinear):
            m.heal_residual_scars_svd()
            count += 1
    return count
