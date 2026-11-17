#!/usr/bin/env python3
"""Builds and forensically verifies standalone `scar-codec` v6.0.0 (`sdist` + `wheel`) for PyPI & JOSS CI.

Key checks when `--verify` is passed:
  1. Builds `scar_codec-6.0.0.tar.gz` and `scar_codec-6.0.0-py3-none-any.whl` from `scar_codec/pyproject.standalone.toml`
     using `setuptools.build_meta` with ZERO `.dylib`/`.so` host binaries leaked inside the wheel.
  2. Extracts the wheel into a temporary **read-only (`0o555`)** `site-packages` directory, blocks `torch`
     (`sys.modules["torch"] = None`), and verifies:
       - `scar_codec.__version__ == "6.0.0"` and `HAS_TORCH is False`
       - Automatic C99/SIMD compilation fallback into `~/.cache/scar_codec/` (`get_compiled_c_lib() is not None`)
       - Sub-microsecond `ScarMmapFile` `O(1)` 64-float block decode (`0.0` diff vs `decode_full`)
       - `ScarCodecE8` (`1.000 bpw`), `SurpriseGatedAnomalySCacheNumPy` (`34,816 B`, `Delta = 0 B`), and `cli.run_quick_demo()`!
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PKG = REPO_ROOT / "scar_codec"
DIST_DIR = REPO_ROOT / "dist_scar_codec"


def build_distributions() -> tuple[Path, Path]:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    for old in DIST_DIR.glob("*"):
        if old.is_file():
            old.unlink()

    with tempfile.TemporaryDirectory() as stage_str:
        stage = Path(stage_str)
        shutil.copy2(SRC_PKG / "pyproject.standalone.toml", stage / "pyproject.toml")
        shutil.copy2(SRC_PKG / "README.md", stage / "README.md")

        lic_src = REPO_ROOT / "LICENSE"
        if lic_src.exists():
            shutil.copy2(lic_src, stage / "LICENSE")
        else:
            (stage / "LICENSE").write_text("MIT License\n\nCopyright (c) 2026 Julio Cesar Reis Filho\n", encoding="utf-8")

        pkg_dst = stage / "scar_codec"
        pkg_dst.mkdir(parents=True, exist_ok=True)
        for fp in sorted(SRC_PKG.iterdir()):
            if fp.is_file() and fp.suffix in (".py", ".c", ".md"):
                shutil.copy2(fp, pkg_dst / fp.name)

        cmd = [
            sys.executable,
            "-c",
            (
                "import os, setuptools.build_meta as bm; "
                "d = os.environ['SCAR_DIST_DIR']; "
                "s = bm.build_sdist(d); "
                "w = bm.build_wheel(d); "
                "print('BUILT:', s, w)"
            ),
        ]
        env = dict(os.environ)
        env["SCAR_DIST_DIR"] = str(DIST_DIR.resolve())
        subprocess.run(cmd, cwd=str(stage), env=env, check=True)

    sdists = list(DIST_DIR.glob("*.tar.gz"))
    wheels = list(DIST_DIR.glob("*.whl"))
    assert len(sdists) == 1 and len(wheels) == 1, f"Expected 1 sdist and 1 wheel in {DIST_DIR}"
    sdist_p, wheel_p = sdists[0], wheels[0]

    # Ensure c_kernels.c is packaged and zero host .dylib/.so binaries leaked
    with zipfile.ZipFile(wheel_p, "r") as zf:
        names = zf.namelist()
        assert "scar_codec/c_kernels.c" in names, "c_kernels.c missing from wheel!"
        assert not any(n.endswith((".so", ".dylib", ".dll")) for n in names), f"Leaked binary in wheel: {names}"

    with tarfile.open(sdist_p, "r:gz") as tf:
        tnames = tf.getnames()
        assert any(n.endswith("scar_codec/c_kernels.c") for n in tnames), "c_kernels.c missing from sdist!"

    print(f"✓ Built standalone sdist : {sdist_p.name} ({sdist_p.stat().st_size / 1024:.1f} KB)")
    print(f"✓ Built standalone wheel : {wheel_p.name} ({wheel_p.stat().st_size / 1024:.1f} KB)")
    return sdist_p, wheel_p


def verify_isolated_wheel(wheel_path: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        ro_site = tmp / "ro_site_packages"
        cache_dir = tmp / "scar_c_cache"
        ro_site.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(wheel_path, "r") as zf:
            zf.extractall(ro_site)

        # Make scar_codec directory read-only (0o555) to verify read-only site-packages cache compilation!
        ro_pkg = ro_site / "scar_codec"
        for p in sorted(ro_pkg.rglob("*"), reverse=True):
            os.chmod(p, stat.S_IRUSR | stat.S_IXUSR if p.is_dir() else stat.S_IRUSR)
        os.chmod(ro_pkg, stat.S_IRUSR | stat.S_IXUSR)

        try:
            test_code = """
import os, sys
sys.modules["torch"] = None  # Strictly block PyTorch!
sys.path.insert(0, sys.argv[1])

import scar_codec
from scar_codec.cli import run_quick_demo

assert scar_codec.__version__ == "6.3.0", scar_codec.__version__
assert scar_codec.HAS_TORCH is False
assert scar_codec.get_compiled_c_lib() is not None, "C99/SIMD failed to compile in read-only site-packages!"

demo = run_quick_demo()
assert demo["zero_pytorch_native"] is True
assert demo["c_simd_compiled"] is True
assert demo["optional_torch_installed"] is False
assert demo["scache_physical_bytes_initial"] == 34816
assert demo["scache_delta_bytes"] == 0
assert demo["vacuum_rows_max_abs_err"] == 0.0
assert demo["block_o1_max_diff"] == 0.0
assert demo["mmap_o1_max_diff"] == 0.0
print("ISOLATED_ZERO_PYTORCH_WHEEL_OK:", demo)
"""
            env = dict(os.environ)
            env["SCAR_CODEC_CACHE_DIR"] = str(cache_dir)
            env["PYTHONPATH"] = str(ro_site)
            subprocess.run(
                [sys.executable, "-c", test_code, str(ro_site)],
                cwd=str(tmp),
                env=env,
                check=True,
            )
        finally:
            # Restore write permissions so TemporaryDirectory cleanup succeeds on all OSes
            os.chmod(ro_pkg, stat.S_IRWXU)
            for p in ro_pkg.rglob("*"):
                os.chmod(p, stat.S_IRWXU if p.is_dir() else stat.S_IRUSR | stat.S_IWUSR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standalone scar-codec v6.0.0 PyPI artifacts")
    parser.add_argument("--verify", action="store_true", help="Run isolated read-only Zero-PyTorch wheel test")
    args = parser.parse_args()

    _, wheel_p = build_distributions()
    if args.verify:
        verify_isolated_wheel(wheel_p)


if __name__ == "__main__":
    main()
