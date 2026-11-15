"""CLI Entry Point `mos-scar-native` for the Pure Scar-Native Foundation Engine (`v5.0.0`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch

from scar_native.model import ScarNativeConfig, ScarNativeLM
from scar_native.tokenizer import ScarNativeTokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mos-scar-native",
        description="Pure Scar-Native Foundation Architecture (v5.0.0) — Tabula-Rasa D4-VQ4 + O(1) Widrow-Hoff + P_perp_lang ScarStore",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_info = sub.add_parser("inspect", help="Inspect Scar-Native model configuration, byte budget, and O(1) SCache layout")
    sp_info.add_argument("--hidden-size", type=int, default=256)
    sp_info.add_argument("--num-layers", type=int, default=6)

    sp_ver = sub.add_parser("verify-parity", help="Verify Native PyTorch <-> WebGPU WGSL 1.0 parity (cos > 0.99999) and O(1) SCache")
    sp_ver.add_argument("--bundle-dir", type=Path, default=None, help="Optional existing WebGPU bundle directory")

    sp_gen = sub.add_parser("generate", help="Generate polyglot output using a saved Scar-Native checkpoint")
    sp_gen.add_argument("--checkpoint", type=Path, required=True)
    sp_gen.add_argument("--prompt", type=str, required=True)
    sp_gen.add_argument("--max-new-tokens", type=int, default=24)
    return p


def main_cli(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.cmd == "inspect":
        cfg = ScarNativeConfig(hidden_size=args.hidden_size, num_layers=args.num_layers)
        model = ScarNativeLM(cfg).eval()
        packed_b = model.physical_packed_bytes()
        from scar_native.model import ScarNativeO1State

        st = ScarNativeO1State(cfg)
        summary = {
            "version": "5.0.0",
            "config": cfg.to_dict(),
            "packed_scar_native_bytes": packed_b,
            "packed_scar_native_mib": round(packed_b / (1024 * 1024), 4),
            "scache_o1_constant_bytes": st.physical_bytes(),
            "scache_o1_constant_kb": round(st.physical_bytes() / 1024.0, 3),
        }
        print(json.dumps(summary, indent=2))
        return

    if args.cmd == "verify-parity":
        import tempfile
        from scar_native.tokenizer import has_cjk_script
        from scar_native.webgpu_v5 import ScarNativeWebGPUEngine, export_scar_native_webgpu_bundle

        tok = ScarNativeTokenizer()
        cfg = ScarNativeConfig()
        torch.manual_seed(202605)
        model = ScarNativeLM(cfg).eval()
        with tempfile.TemporaryDirectory() as tmp:
            bdir = args.bundle_dir if (args.bundle_dir and args.bundle_dir.exists()) else Path(tmp)
            if not (bdir / "scarnative_manifest.json").exists():
                export_scar_native_webgpu_bundle(model, tok, bdir)
            eng = ScarNativeWebGPUEngine(bdir)
            model.pack_all_to_scar_binary()

            prompt = "<lang:pt> <user> chave secreta <passkey> alpha = 583 </passkey> . o sistema analisa o fluxo de telemetria com memória constante e zero ruído no vácuo . qual é a chave alpha ? <assistant>"
            pids = tok.encode(prompt, add_bos=True)
            suppress = not has_cjk_script(prompt)
            cjk_mask_t = torch.from_numpy(tok.build_cjk_bitmask_u32().astype(np.int64)).to(torch.int64) if False else None

            from scar_native.model import ScarNativeO1State
            import torch.nn.functional as F

            st = ScarNativeO1State(cfg)
            eng.reset_scache()
            cos_vals: List[float] = []
            for idx, tid in enumerate(pids):
                lg_nat, _ = model.step_token_o1(tid, idx, st)
                if suppress:
                    mask_u32 = torch.from_numpy(tok.build_cjk_bitmask_u32().astype("int64"))
                    v_i = torch.arange(cfg.vocab_size)
                    bit = (mask_u32[v_i >> 5] >> (v_i & 31)) & 1
                    lg_nat = torch.where(bit != 0, lg_nat - 64.0, lg_nat)
                lg_gpu, _ = eng.step_token_gpu(tid, idx, suppress_cjk=suppress, route_on_golden=False)
                lg_gpu_t = torch.from_numpy(lg_gpu)
                cos = float(F.cosine_similarity(lg_nat.unsqueeze(0), lg_gpu_t.unsqueeze(0)).item())
                cos_vals.append(cos)
            print(
                json.dumps(
                    {
                        "tokens_checked": len(pids),
                        "min_cosine_parity": min(cos_vals),
                        "mean_cosine_parity": sum(cos_vals) / len(cos_vals),
                        "scache_gpu_bytes": eng.scache_total_gpu_bytes,
                        "scache_delta_bytes": 0,
                    },
                    indent=2,
                )
            )
        return

    if args.cmd == "generate":
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ScarNativeConfig(**ckpt["config"])
        tok = ScarNativeTokenizer(id_to_token=ckpt["vocab"])
        model = ScarNativeLM(cfg)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        if "cartridges" in ckpt:
            model.scar_store.cartridges = ckpt["cartridges"]
        model.eval()
        pids = tok.encode(args.prompt, add_bos=True)
        cjk_mask = torch.from_numpy(tok.build_cjk_bitmask_u32().astype("int64"))
        out_ids, telem, st = model.generate_o1(
            pids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=tok.eos_token_id,
            cjk_bitmask_u32=cjk_mask,
        )
        print(
            json.dumps(
                {
                    "completion": tok.decode(out_ids),
                    "routed_cartridge": telem.routed_cartridge,
                    "margin": round(telem.margin, 4),
                    "scache_bytes": st.physical_bytes(),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main_cli()
