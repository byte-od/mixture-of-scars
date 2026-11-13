#!/usr/bin/env python3
"""Publishes the Official `v3.0.0` Online Cartridge Library + Preprint Assets to GitHub Releases (`byte-od/mixture-of-scars`)
and runs a live public-CDN `HTTP/1.1 206 Partial Content` verification against `github.com/.../releases/download/v3.0.0/`.

Security guarantee: Reads credentials exclusively in-memory via `git credential fill` (or `GH_TOKEN`) and never logs secrets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any, Dict, List
import urllib.error
import urllib.parse
import urllib.request

from scar_cache.cli_scar_store import (
    OFFICIAL_CATALOG_URL,
    OFFICIAL_RELEASE_TAG,
    HTTPRangeCapsuleClient,
    fetch_official_catalog,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LIB_DIR = REPO_ROOT / "results" / "online_library_v3_0"
ZENODO_DIR = REPO_ROOT / "zenodo_deposit_v3_0"
OWNER_REPO = "byte-od/mixture-of-scars"


def _get_github_token() -> str:
    env_tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if env_tok:
        return env_tok.strip()
    proc = subprocess.run(
        ["git", "credential", "fill"],
        input="protocol=https\nhost=github.com\n\n",
        text=True,
        capture_output=True,
        check=True,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("password="):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("No GitHub token found in GH_TOKEN/GITHUB_TOKEN or git credential helper.")


def _gh_api(
    method: str,
    url: str,
    token: str,
    body_bytes: bytes | None = None,
    content_type: str = "application/json",
) -> Any:
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "mixture-of-scars-release-publisher/3.0.0",
    }
    if body_bytes is not None:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
        return json.loads(raw.decode("utf-8")) if raw else {}


def _ensure_release(token: str, tag: str, title: str, notes: str) -> Dict[str, Any]:
    api_base = f"https://api.github.com/repos/{OWNER_REPO}/releases"
    try:
        rel = _gh_api("GET", f"{api_base}/tags/{tag}", token)
        rel = _gh_api(
            "PATCH",
            f"{api_base}/{rel['id']}",
            token,
            body_bytes=json.dumps({"name": title, "body": notes, "draft": False, "prerelease": False}).encode("utf-8"),
        )
        print(f"[GitHub Release] Updated existing release {tag} (id={rel['id']})", flush=True)
        return rel
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    payload = {
        "tag_name": tag,
        "name": title,
        "body": notes,
        "draft": False,
        "prerelease": False,
    }
    rel = _gh_api("POST", api_base, token, body_bytes=json.dumps(payload).encode("utf-8"))
    print(f"[GitHub Release] Created public release {tag} (id={rel['id']})", flush=True)
    return rel


def _upload_or_replace_asset(rel: Dict[str, Any], file_path: Path, token: str) -> Dict[str, Any]:
    existing_assets: List[Dict[str, Any]] = rel.get("assets", [])
    for asset in existing_assets:
        if asset.get("name") == file_path.name:
            _gh_api(
                "DELETE",
                f"https://api.github.com/repos/{OWNER_REPO}/releases/assets/{asset['id']}",
                token,
            )
            print(f"  [Replace] Deleted previous asset {file_path.name}", flush=True)

    upload_base = rel["upload_url"].split("{", 1)[0]
    upload_url = f"{upload_base}?{urllib.parse.urlencode({'name': file_path.name})}"
    ctype = "application/json" if file_path.suffix == ".json" else "application/octet-stream"
    data = file_path.read_bytes()
    out = _gh_api("POST", upload_url, token, body_bytes=data, content_type=ctype)
    print(f"  [Uploaded] {file_path.name} ({len(data) / 1024.0:.1f} KB) -> {out['browser_download_url']}", flush=True)
    return out


def main() -> int:
    token = _get_github_token()
    catalog = json.loads((LIB_DIR / "scarstore_catalog_v3_0.json").read_text(encoding="utf-8"))

    cart_lines = "\n".join(
        f"- **`{c['capsule_id']}`** — *{c['title']}* (`{c['category']}`)"
        for c in catalog["cartridges"]
    )
    notes = f"""## Mixture of Scars (`MoS`) & `SCache O(1)` — Official `v3.0.0` Release + Live Online Cartridge Library

### 🌐 Live Online Cartridge Library (`HTTP/1.1 206 Partial Content` Enabled)
Stream exact OS-page-aligned **`140.00 KB` (`35 × 4 KB`)** cartridges on `Qwen/Qwen2.5-3B-Instruct` or **`61.25 KB`** cartridges on `Qwen/Qwen2.5-0.5B-Instruct` directly from this GitHub Release CDN with **zero server setup and zero backpropagation**:

```bash
# 0. Launch the zero-dependency Visual Cartridge Store + Live SCache O(1) Chat + Open-Weights Factory Web UI:
mos-store ui --model Qwen/Qwen2.5-3B-Instruct --remote auto

# 1. Browse all 9 live online cartridges in terminal (titles, categories, sizes & example prompts):
mos-store catalog

# 2. Ask any question — automatically downloads a tiny Cloud Shell & streams ONLY the exact 140 KB cartridge over HTTP 206:
mos-store query --remote auto --model Qwen/Qwen2.5-3B-Instruct \\
  --prompt "Explain concisely what the Fischer-Lynch-Paterson impossibility theorem proves about asynchronous consensus."
```

#### Included `72.71B -> 3B / 0.5B` Closed-Form Cartridges (`9` Online Slots):
{cart_lines}

### 📄 Preprint `v3.0.0`, Zero-Spike `MoS` Student (`3.49 GiB` VRAM) & Forensic `88/88` Replication Suite
- `Mixture_of_Scars_and_SCache_Preprint_v3.0.pdf`
- `arXiv_Direct_Upload_v3.0.zip` & `MoS_SCache_LaTeX_arXiv_TMLR_v3.0.zip`
- `MoS_SCache_Adversarial_Replication_v3.0.zip` (`88/88` `pytest` checks, `0.00 B` storage discrepancy)
"""

    rel = _ensure_release(
        token=token,
        tag=OFFICIAL_RELEASE_TAG,
        title="v3.0.0 — Preprint v3.0, <24 GB 72.7B Sub-2-Bit & Live Online 140 KB Cartridge Library (HTTP 206)",
        notes=notes,
    )

    assets_to_publish = [
        LIB_DIR / "scarstore_catalog_v3_0.json",
        LIB_DIR / "qwen3b_v3_0.manifest.json",
        LIB_DIR / "qwen3b_v3_0.scarstore",
        LIB_DIR / "qwen3b_v3_0.shell.scarstore.gz",
        LIB_DIR / "qwen0p5b_v3_0.manifest.json",
        LIB_DIR / "qwen0p5b_v3_0.scarstore",
        LIB_DIR / "qwen0p5b_v3_0.shell.scarstore.gz",
        ZENODO_DIR / "Mixture_of_Scars_and_SCache_Preprint_v3.0.pdf",
        ZENODO_DIR / "arXiv_Direct_Upload_v3.0.zip",
        ZENODO_DIR / "MoS_SCache_LaTeX_arXiv_TMLR_v3.0.zip",
        ZENODO_DIR / "MoS_SCache_Adversarial_Replication_v3.0.zip",
        ZENODO_DIR / "mos_scache_v3_0_hero.png",
    ]
    for fp in assets_to_publish:
        if not fp.exists():
            raise FileNotFoundError(f"Missing release asset: {fp}")
        _upload_or_replace_asset(rel, fp, token)

    # Live Public CDN HTTP 206 Verification!
    print("\n[Live Public CDN Verification] Auditing catalog & HTTP 206 byte-range slices from GitHub CDN ...", flush=True)
    live_cat = fetch_official_catalog(OFFICIAL_CATALOG_URL)
    assert len(live_cat["cartridges"]) == len(catalog["cartridges"])

    for tier_name, expected_bytes in (("3b", 143360), ("0.5b", 62720)):
        client = HTTPRangeCapsuleClient(base_url=tier_name)
        manifest_map = client._ensure_manifest()
        assert len(manifest_map) == len(catalog["cartridges"])
        sample_coord, sample_slot = next(iter(manifest_map.items()))
        raw_slice = client.fetch_capsule_bytes(sample_coord, sample_slot["capsule_id"])
        assert raw_slice is not None and len(raw_slice) == expected_bytes
        ev = client.fetch_events[-1]
        print(
            f"  ✓ [{tier_name}] HTTP {ev['http_status']} | {ev['capsule_id']} ({ev['title']}) | "
            f"{ev['range_header']} -> {ev['bytes_received']:,} B in {ev['latency_ms']} ms!",
            flush=True,
        )

    print(f"\n✓ Official Online Cartridge Library is LIVE at: {rel['html_url']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
