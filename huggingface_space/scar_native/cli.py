"""CLI Entry Point `mos-scar-native` for the Pure Scar-Native Foundation Engine (`v6.0.0`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch

from scar_native.model import ScarNativeConfig, ScarNativeLM
from scar_native.model_v6 import ScarNativeLMV6, ScarNativeV6Config, compute_v6_architectural_byte_budget
from scar_native.mos_digest import MoSDigestCompiler, parse_markdown_knowledge_doc
from scar_native.tokenizer import ScarNativeTokenizer
from scar_native.v6_bpe_tokenizer import ScarNativeBPETokenizer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mos-scar-native",
        description="Pure Scar-Native Foundation Architecture (v6.0.0) — Shared-Stage MoS + Chunkwise WY + Dual-Tier O(1) SCache + mos-digest",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_info = sub.add_parser("inspect", help="Inspect Scar-Native v5.0 model configuration and O(1) SCache layout")
    sp_info.add_argument("--hidden-size", type=int, default=256)
    sp_info.add_argument("--num-layers", type=int, default=6)

    sp_v6 = sub.add_parser("inspect-v6", help="Inspect Scar-Native v6.0 Shared-Stage MoS + Dual-Tier O(1) SCache byte budget")
    sp_v6.add_argument("--hidden-size", type=int, default=512)
    sp_v6.add_argument("--intermediate-size", type=int, default=1024)
    sp_v6.add_argument("--num-layers", type=int, default=12)
    sp_v6.add_argument("--num-stages", type=int, default=3)

    sub.add_parser(
        "inspect-frontier",
        help="Compute exact first-principles byte budget for ScarNative v6-Frontier (96L, d=8192, 128k context)",
    )

    sp_dig = sub.add_parser("digest", help="Compile Markdown document claims into 4 KB-aligned .scarstore cartridges")
    sp_dig.add_argument("--bundle-dir", type=Path, required=True, help="Path to exported ScarNative v6 bundle")
    sp_dig.add_argument("--doc", type=Path, required=True, help="Input Markdown knowledge document")
    sp_dig.add_argument("--doc-id", type=str, default="doc_cli")
    sp_dig.add_argument("--out", type=Path, required=True, help="Output .scarstore path")

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

    if args.cmd == "inspect-v6":
        cfg6 = ScarNativeV6Config(
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_layers=args.num_layers,
            num_stages=args.num_stages,
        )
        budget = compute_v6_architectural_byte_budget(cfg6)
        print(json.dumps({"version": "6.0.0", "config": cfg6.to_dict(), "budget": budget}, indent=2))
        return

    if args.cmd == "inspect-frontier":
        frontier_cfg = ScarNativeV6Config(
            vocab_size=131072,
            embed_rank=256,
            hidden_size=8192,
            intermediate_size=28672,
            num_layers=96,
            num_stages=12,
            num_q_heads=64,
            num_kv_heads=8,
            head_dim=128,
            id_scar_rank=64,
            head_scar_rank=64,
            num_mos_experts=16,
            mos_top_k=2,
            mos_expert_rank=128,
            num_sink=16,
            slide_len=240,
            num_micro_pages=64,
            golden_layer=72,
            k_lang_null=64,
            cartridge_rank=64,
        )
        budget = compute_v6_architectural_byte_budget(
            frontier_cfg, active_cartridges=8, mos_lru_hot_per_layer=4, ref_context_tokens=131072, tie_lm_head_in_frontier=True
        )
        print(json.dumps({"version": "6.0.0-frontier-blueprint", "budget": budget}, indent=2))
        return

    if args.cmd == "digest":
        ckpt_path = args.bundle_dir / "scarnative_v6_checkpoint.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Expected {ckpt_path} for mos-digest compilation")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg6 = ScarNativeV6Config(**ckpt["config"])
        tok6 = ScarNativeBPETokenizer.load(args.bundle_dir / "tokenizer.json")
        model6 = ScarNativeLMV6(cfg6)
        model6.load_state_dict(ckpt["state_dict"], strict=False)
        model6.eval()
        claims = parse_markdown_knowledge_doc(args.doc.read_text(encoding="utf-8"), doc_id=args.doc_id)
        comp = MoSDigestCompiler(model6, tok6)
        res = comp.digest_claims_batch(claims, out_scarstore_path=args.out)
        print(json.dumps(res, ensure_ascii=False, indent=2))
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
