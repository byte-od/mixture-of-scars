"""Native WebGPU / WGSL Engine & Static Browser Studio Assets (`Horizon 3.3`)."""

from pathlib import Path

WEBGPU_DIR = Path(__file__).resolve().parent
WGSL_KERNELS_PATH = WEBGPU_DIR / "mos_scache_kernels.wgsl"
WEBGPU_APP_DIR = WEBGPU_DIR / "app"

__all__ = ["WEBGPU_DIR", "WGSL_KERNELS_PATH", "WEBGPU_APP_DIR"]
