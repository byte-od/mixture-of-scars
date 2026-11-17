"""Official `numcodecs` / `Zarr` Ecosystem Codec Plugin (`ScarZarrCodec`, `codec_id = "scar"`).

Allows `ScarCodec` (`VQ4-Scar` `2.0 bpw` OR Gosset `E_8` `1.000 bpw` + `INT4` residual scars + optional
deterministic `abs_tolerance` L_inf bound) to drop seamlessly into `Zarr`, `xarray`, `Dask`, and `Astropy`.
"""

from __future__ import annotations

import struct
from typing import Any, Optional

import numpy as np

from scar_codec.binary_format import deserialize_scar_buffer, serialize_scar_bytes
from scar_codec.nd_e8 import ScarCodecE8, ScarE8Package
from scar_codec.numpy_engine import NumPyVQ4ScarCodec

try:
    from numcodecs.abc import Codec as _BaseCodec  # type: ignore[import-not-found]
    from numcodecs.registry import register_codec as _register_codec  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - clean zero-dependency fallback when numcodecs is absent
    class _BaseCodec:  # type: ignore[no-redef]
        codec_id: Optional[str] = None

        def get_config(self) -> dict[str, Any]:
            return {"id": self.codec_id}

    def _register_codec(cls, codec_id=None) -> None:
        return None


ZARR_WRAP_MAGIC = b"SZR6"


class ScarZarrCodec(_BaseCodec):
    """`numcodecs`-compatible array chunk compressor backed by `NumPyVQ4ScarCodec` / `ScarCodecE8` + C/SIMD."""

    codec_id = "scar"

    def __init__(
        self,
        rank: int = 16,
        max_point_spikes: int = 256,
        abs_tolerance: Optional[float] = None,
        mode: str = "vq4",
    ) -> None:
        if mode not in ("vq4", "e8"):
            raise ValueError(f"mode must be 'vq4' or 'e8', got {mode!r}")
        self.rank = int(rank)
        self.max_point_spikes = int(max_point_spikes)
        self.abs_tolerance = float(abs_tolerance) if abs_tolerance is not None else None
        self.mode = mode

    def encode(self, buf: Any) -> bytes:
        arr = np.asarray(buf)
        orig_dtype = arr.dtype.str
        if len(orig_dtype.encode("ascii")) > 8:
            orig_dtype = "<f4"
        orig_shape = arr.shape if arr.ndim >= 1 else (1,)
        arr_f32 = arr.astype(np.float32, copy=False)

        if arr_f32.ndim <= 1:
            n = arr_f32.size
            cols = min(256, max(64, ((int(max(1, n) ** 0.5) + 63) // 64) * 64))
            rows = max(1, (n + cols - 1) // cols)
            mat = np.zeros((rows, cols), dtype=np.float32)
            mat.reshape(-1)[:n] = arr_f32.reshape(-1)
        elif arr_f32.ndim == 2:
            mat = arr_f32
        else:
            last_dim = arr_f32.shape[-1]
            lead = int(np.prod(arr_f32.shape[:-1]))
            mat = arr_f32.reshape(lead, last_dim)

        if self.mode == "e8":
            pkg = ScarCodecE8(
                rank=self.rank,
                max_point_spikes=self.max_point_spikes,
                abs_tolerance=self.abs_tolerance,
            ).encode(mat)
        else:
            pkg = NumPyVQ4ScarCodec(
                rank=self.rank,
                max_point_spikes=self.max_point_spikes,
                abs_tolerance=self.abs_tolerance,
            ).encode(mat)

        scar_blob = serialize_scar_bytes(pkg)
        dt_bytes = orig_dtype.encode("ascii")[:8].ljust(8, b"\x00")
        ndim = len(orig_shape)
        meta = struct.pack(f"<4s8sI{ndim}Q", ZARR_WRAP_MAGIC, dt_bytes, ndim, *orig_shape)
        return meta + scar_blob

    def decode(self, buf: Any, out: Optional[np.ndarray] = None) -> np.ndarray:
        mv = memoryview(buf)
        magic, dt_bytes, ndim = struct.unpack_from("<4s8sI", mv, 0)
        if magic != ZARR_WRAP_MAGIC:
            raise ValueError(f"Invalid ScarZarrCodec chunk header: {magic!r}")
        hdr_len = 16 + 8 * ndim
        orig_shape = struct.unpack_from(f"<{ndim}Q", mv, 16)
        orig_dtype = np.dtype(dt_bytes.rstrip(b"\x00").decode("ascii"))

        pkg = deserialize_scar_buffer(mv[hdr_len:], verify_crc=True)
        if isinstance(pkg, ScarE8Package):
            rec_2d = ScarCodecE8.decode_full(pkg)
        else:
            rec_2d = NumPyVQ4ScarCodec.decode_full(pkg, use_c_accel=True)
        total_n = int(np.prod(orig_shape, dtype=np.int64))
        rec = rec_2d.reshape(-1)[:total_n].reshape(orig_shape).astype(orig_dtype, copy=False)

        if out is not None:
            np.copyto(out.reshape(orig_shape) if out.size == total_n else out, rec)
            return out
        return rec

    def get_config(self) -> dict[str, Any]:
        return {
            "id": self.codec_id,
            "rank": self.rank,
            "max_point_spikes": self.max_point_spikes,
            "abs_tolerance": self.abs_tolerance,
            "mode": self.mode,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ScarZarrCodec":
        return cls(
            rank=config.get("rank", 16),
            max_point_spikes=config.get("max_point_spikes", 256),
            abs_tolerance=config.get("abs_tolerance", None),
            mode=config.get("mode", "vq4"),
        )


_register_codec(ScarZarrCodec, codec_id="scar")
