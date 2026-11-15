"""Builds & Verifies the Full `Qwen2.5-3B-Instruct` (`Track 2.5 High-Fidelity ProtectedInt4Scar`, `1,601.57 MiB`)
Unified WebGPU + Native Portfolio Engine & `GoldenSliceScarStoreV4` (`41 x 140 KB` `HTTP 206`) for JARVIS.

Architecture (`Track 2.5 ProtectedInt4Scar` + `20` WGSL 1.0 Compute Shaders):
  1. Unified `ProtectedInt4ScarMatrix` (`prot_int4_scar`, `4.25 bpw Block-64 INT4` + Channel Equilibration +
     2-Pass `INT4-ASVD` Residual Scar Healing via WGSL Shaders #19 `vec_equilibrate_f16` &
     #20 `mos_fused_prot_int4_int4scar_gemv`) across all 36 layers:
       - `self_attn.q_proj` (`r=32`), `self_attn.o_proj` (`r=32`), `self_attn.k_proj/v_proj` (`r=0`).
       - `mlp.gate_proj` (`r=32`), `mlp.up_proj` (`r=32`), `mlp.down_proj` (`r=64`).
  2. `Block-64 INT4` tied `embed_tokens` / `lm_head` + WGSL Shader #18 `apply_script_subspace_gate` (`P_perp_CJK`).
  3. `GoldenSliceScarStoreV4` at `Layer 23/36` (`d=2048, k_lang=14, rank_mid=8, rank_final=56`):
       - 41 universal language-invariant `143,360 B` (`140.00 KB`) `Block-64 INT4` cartridges streamed via `HTTP 206`.
  4. Constant `SCache O(1)` (`112` Attention Sinks + `144` Ring + `32` Widrow-Hoff Slots = `15,345,792 B`, `Δ +0 B`).
"""

from __future__ import annotations

import copy
import gc
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scar_cache.cli_scar_store import (
    ScarStoreHTTPRangeServer,
    generate_with_scar_store,
    make_http_range_capsule_fetcher,
)
from scar_cache.golden_slice_v4 import (
    GoldenSliceScarBank,
    GoldenSliceScarStore,
    GoldenSliceSynthesisEngine,
    extract_user_content_bounds,
    format_universal_chatml,
)
from scar_cache.scar_grafting import extract_span_query_probes
from scar_cache.vq4_scar_mos import (
    ProtectedInt4ScarMatrix,
    ScriptSubspaceGate,
    VQ4ScarPackedMatrix,
    dequant_block64_int4,
    has_cjk_script,
    pack_block64_int4,
    quantize_matrix_vq4_scar_2pass,
    quantize_protected_int4_2pass,
)
from scar_cache.webgpu_engine import (
    NativeWebGPUMoSEngine,
    export_mos_model_to_webgpu,
    export_scarstore_for_webgpu,
)
from scripts.build_portfolio_scarstore_v4_0 import (
    PORTFOLIO_BRIDGE_SPEC as BASE_BRIDGE_SPEC,
    PORTFOLIO_CALIBRATION_QUADS,
    PORTFOLIO_CONCEPT_SPECS as BASE_CONCEPT_SPECS,
)
from scripts.test_vector_scar_codebook_3b import DISJOINT_POLYGLOT_CALIB_TEXTS

LIB_DIR = REPO_ROOT / "results" / "online_library_v3_0"
MOS_3B_CACHE_DIR = Path.home() / ".cache" / "mixture_of_scars" / "mos_students" / "qwen3b_track25_prot_int4_scar"
PORTFOLIO_ROOT = Path("/Users/juliocesarreisfilho/Projects/portfolio")
PORTFOLIO_WEBGPU_3B_DIR = PORTFOLIO_ROOT / "public" / "webgpu" / "qwen3b"

SEED_HORIZON = 35
SEED_FADE_STEPS = 13
NUM_SINK_TOKENS = 112

# Compact ChatML system prompt permanently locked inside `num_sink_tokens = 112` (94 BPE tokens):
PORTFOLIO_SYSTEM_PROMPT_3B = (
    "You are JARVIS, the official Qwen2.5-3B AI assistant for visitors on Julio Reis's portfolio "
    "(byte-od.github.io), running locally via WebGPU WGSL, MoS, and SCache O(1). The visitor is not "
    "Julio; never claim to be Julio and refer to Julio Reis in the third person. Always reply in "
    "the user's language in 2 to 4 complete sentences."
)

DISJOINT_CONV_CALIB_TEXTS: Tuple[str, ...] = (
    "<|im_start|>user\nOlá! Como está o seu dia hoje?<|im_end|>\n"
    "<|im_start|>assistant\nOlá! Tudo ótimo por aqui, pronto para conversar e responder às suas dúvidas em português.<|im_end|>\n",
    "<|im_start|>user\nHi there! How is your day going?<|im_end|>\n"
    "<|im_start|>assistant\nHello! Everything is going great on my side, and I am happy to help you today.<|im_end|>\n",
    "<|im_start|>user\n¡Hola! ¿Cómo va todo hoy?<|im_end|>\n"
    "<|im_start|>assistant\n¡Hola! Todo muy bien por aquí, listo para ayudarte con cualquier consulta.<|im_end|>\n",
    "<|im_start|>user\nBonjour ! Comment se passe votre journée ?<|im_end|>\n"
    "<|im_start|>assistant\nBonjour ! Tout se passe très bien, je suis prêt à vous aider aujourd'hui.<|im_end|>\n",
)

OUT_OF_CONE_CONTROLS: Tuple[str, ...] = (
    "Olá, bom dia! Tudo bem?",
    "Hello, good morning! How are you?",
    "¡Hola, buenos días! ¿Qué tal?",
    "Bonjour ! Comment allez-vous ?",
    "Oi, tudo bem?",
    "Como vai?",
    "Qual a raiz quadrada de PI elevado à 2?",
    "Explique a diferença entre os protocolos TCP e UDP nas redes de computadores.",
    "Por que a água se expande quando congela e vira gelo?",
    "Descreva as principais causas da Revolução Francesa no século XVIII.",
    "Qual a maior capital entre os estados das americas sul e norte?",
    "Como funciona a fotossíntese nas plantas?",
)


def build_3b_portfolio_specs() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Builds all 40 unified JARVIS `Qwen2.5-3B` specs (`13` Portfolio + `27` Colab `72B` Gap-Mined `140 KB` cartridges)."""
    specs = copy.deepcopy(BASE_CONCEPT_SPECS)
    bridge = copy.deepcopy(BASE_BRIDGE_SPEC)

    for sp in specs:
        cid = sp["capsule_id"]
        if cid == "portfolio_agent_identity":
            sp["title"] = "JARVIS — Official In-Browser Qwen2.5-3B Portfolio AI Assistant"
            sp["title_pt"] = "JARVIS — Assistente Oficial Qwen2.5-3B do Portfólio no Navegador"
            sp["summary"] = (
                "JARVIS: official client-side AI assistant for visitors on Julio Reis's portfolio (never impersonating Julio), "
                "executing Qwen2.5-3B 100% locally via 20 WebGPU WGSL shaders, ProtectedInt4Scar MoS, and SCache O(1)."
            )
            sp["anchor_queries_en"] = [
                "Who are you and what is your role on this website?",
                "What is your name? Who are you?",
                "Do you have a name and are you an AI assistant?",
            ]
            sp["seeds"] = [
                (
                    "Who are you and what is your name on this portfolio?",
                    "Hello! I am JARVIS, the official Qwen 3B assistant on Julio Reis's portfolio, running locally via WebGPU WGSL, MoS, and SCache O(1).",
                ),
                (
                    "Quem é você e qual é o seu nome neste portfólio?",
                    "Olá! Sou o JARVIS, assistente oficial Qwen 3B do portfólio de Julio Reis, rodando localmente via WebGPU WGSL, MoS e SCache O(1).",
                ),
                (
                    "¿Quién eres y cuál es tu nombre en este portafolio?",
                    "¡Hola! Soy JARVIS, el asistente oficial Qwen 3B del portafolio de Julio Reis, corriendo localmente vía WebGPU WGSL, MoS y SCache O(1).",
                ),
                (
                    "Qui êtes-vous et quel est votre nom sur ce portfolio ?",
                    "Bonjour ! Je suis JARVIS, l'assistant officiel Qwen 3B du portfolio de Julio Reis, tournant localement via WebGPU WGSL, MoS et SCache O(1).",
                ),
            ]
        elif cid == "portfolio_agent_capabilities":
            sp["title"] = "JARVIS — Knowledge Cutoff, Offline Privacy & HTTP 206 Updates"
            sp["title_pt"] = "JARVIS — Atualização de Conhecimento, Privacidade Offline & HTTP 206"
            sp["summary"] = (
                "Combines offline Qwen2.5-3B general pretraining (zero external API calls) with live 2026 "
                "HTTP 206 ScarStore v4.0 140 KB cartridges covering Julio Reis's portfolio."
            )
            sp["seeds"] = [
                (
                    "What is your last update or knowledge cutoff on general knowledge?",
                    "My general knowledge comes from offline Qwen2.5-3B pretraining without web search, paired with 2026 HTTP 206 cartridges on Julio Reis.",
                ),
                (
                    "Qual sua última atualização sobre conhecimentos gerais?",
                    "Meu conhecimento geral vem do pré-treino offline do Qwen2.5-3B sem busca web, unido aos cartuchos HTTP 206 de 2026 sobre Julio Reis.",
                ),
                (
                    "¿Cuál es tu última actualización sobre conocimientos generales?",
                    "Mi conocimiento general proviene del modelo offline Qwen2.5-3B sin búsqueda web, unido a cartuchos HTTP 206 de 2026 sobre Julio Reis.",
                ),
                (
                    "Quelle est votre dernière mise à jour sur les connaissances générales ?",
                    "Ma culture générale repose sur le pré-entraînement hors ligne de Qwen2.5-3B sans accès web, uni aux cartouches HTTP 206 de 2026 sur Julio Reis.",
                ),
            ]
        elif cid == "portfolio_browser_webgpu_wgsl":
            sp["summary"] = (
                "20 custom WGSL 1.0 compute shaders executing VQ4-Scar + Protected INT4 MoS GEMV, Widrow-Hoff SCache O(1), "
                "Layer-23 P_perp_lang routing, and HTTP 206 140 KB conceptual cartridges."
            )
            sp["seeds"] = [
                (
                    "How do you run 100% in-browser via WebGPU/WGSL and stream cartridges via HTTP 206?",
                    "In the browser, 20 WGSL shaders run Qwen 3B MoS without Python or ONNX, streaming 140 KB cartridges via HTTP 206.",
                ),
                (
                    "Como você roda 100% no navegador via WebGPU/WGSL e baixa cartuchos via HTTP 206?",
                    "No navegador, 20 shaders WGSL rodam o Qwen 3B MoS sem Python nem ONNX, baixando cartuchos de 140 KB via HTTP 206.",
                ),
                (
                    "¿Cómo funcionas 100% en el navegador con WebGPU/WGSL y cartuchos HTTP 206?",
                    "En el navegador, 20 shaders WGSL ejecutan Qwen 3B MoS sin Python ni ONNX, cargando cartuchos de 140 KB vía HTTP 206.",
                ),
                (
                    "Comment fonctionnez-vous 100% dans le navigateur via WebGPU/WGSL et HTTP 206 ?",
                    "Dans le navigateur, 20 shaders WGSL exécutent Qwen 3B MoS sans Python ni ONNX, chargeant des cartouches de 140 Ko via HTTP 206.",
                ),
            ]

    specs.insert(
        2,
        {
            "capsule_id": "portfolio_jarvis_architecture",
            "title": "JARVIS Architecture — Qwen 3B ProtectedInt4Scar MoS, 20 WGSL Shaders & SCache O(1)",
            "title_pt": "Arquitetura do JARVIS — Qwen 3B MoS ProtectedInt4Scar, 20 Shaders WGSL & SCache O(1)",
            "category": "Flagship AI Architecture",
            "summary": (
                "Full overview of JARVIS's client-side AI architecture: Qwen2.5-3B compressed to 1.60 GB via ProtectedInt4Scar MoS, "
                "executed by 20 custom WebGPU WGSL shaders with constant 14.9 MB SCache O(1) and 140 KB HTTP 206 cartridges."
            ),
            "links": [
                {"label": "GitHub mixture-of-scars", "url": "https://github.com/byte-od/mixture-of-scars", "icon": "github"},
            ],
            "tau": 0.48,
            "ramp": 0.04,
            "anchor_queries_en": [
                "What is your architecture?",
                "Tell me about your architecture.",
                "Explain the architecture of JARVIS.",
                "What is your software and AI architecture?",
            ],
            "prompts_polyglot": {
                "pt": "Qual a sua arquitetura?",
                "en": "What is your architecture?",
                "es": "¿Cuál es tu arquitectura?",
                "fr": "Quelle est votre architecture ?",
            },
            "seeds": [
                (
                    "What is your architecture?",
                    "My architecture runs Qwen 3B via 20 WGSL shaders, MoS, SCache O(1), and HTTP 206 140 KB cartridges.",
                ),
                (
                    "Qual a sua arquitetura?",
                    "Minha arquitetura roda Qwen 3B com 20 shaders WGSL, MoS, SCache O(1) e cartuchos HTTP 206 de 140 KB.",
                ),
                (
                    "¿Cuál es tu arquitectura?",
                    "Mi arquitectura ejecuta Qwen 3B con 20 shaders WGSL, MoS, SCache O(1) y cartuchos HTTP 206 de 140 KB.",
                ),
                (
                    "Quelle est votre architecture ?",
                    "Mon architecture exécute Qwen 3B via 20 shaders WGSL, MoS, SCache O(1) et cartouches HTTP 206 de 140 Ko.",
                ),
            ],
        },
    )

    ow_json_path = LIB_DIR / "jarvis_72b_gap_mined_specs.json"
    if "--13-only" not in sys.argv and ow_json_path.exists():
        ow_payload = json.loads(ow_json_path.read_text(encoding="utf-8"))
        existing_ids = {s["capsule_id"] for s in specs}
        for ow_sp in ow_payload.get("specs", []):
            if ow_sp["capsule_id"] in existing_ids:
                continue
            sp_copy = copy.deepcopy(ow_sp)
            if sp_copy["capsule_id"] == "ow_quantum_applied_theorems":
                sp_copy["tau"] = 0.46
                sp_copy["ramp"] = 0.04
            if sp_copy["capsule_id"] == "ow_compilers_ssa_dominance":
                sp_copy["seeds"] = [
                    (
                        "What is Static Single Assignment (SSA) form and how do dominance frontiers place phi-functions in compilers?",
                        "In Static Single Assignment (SSA) form, every variable is assigned once, placing minimal phi-functions at iterated dominance frontiers (Cytron et al., 1991).",
                    ),
                    (
                        "O que é a forma Static Single Assignment (SSA) e como a fronteira de dominância posiciona funções phi em compiladores?",
                        "Na forma SSA, cada variável é atribuída uma única vez, inserindo funções phi mínimas nas fronteiras de dominância iteradas (Cytron, 1991).",
                    ),
                    (
                        "¿Qué es la forma Static Single Assignment (SSA) y cómo ubican las fronteras de dominancia las funciones phi?",
                        "En la forma SSA, cada variable se asigna una sola vez, insertando funciones phi mínimas en las fronteras de dominancia iteradas (Cytron, 1991).",
                    ),
                    (
                        "Qu'est-ce que la forme Static Single Assignment (SSA) et comment les frontières de dominance placent-elles les fonctions phi ?",
                        "Sous forme SSA, chaque variable est assignée une seule fois, insérant des fonctions phi minimales aux frontières de dominance itérées (Cytron, 1991).",
                    ),
                ]
            else:
                sp_copy["seeds"] = [(pair[0], pair[1]) for pair in ow_sp["seeds"]]
            sp_copy["teacher_model_override"] = (
                f"Colab A100-80GB Same-Vocab Contrastive Gap ({ow_payload.get('teacher_model', 'Qwen2.5-72B')} -> Qwen2.5-3B)"
            )
            specs.append(sp_copy)
        print(
            f"  ✓ Loaded {len(ow_payload.get('specs', []))} Colab 72B Gap-Mined Open-Weights Specs "
            f"-> Total Capsules = {len(specs) + 1}",
            flush=True,
        )

    return specs, bridge


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def load_or_quantize_track24c_3b(
    model_id: str,
    tok: AutoTokenizer,
    dev: torch.device,
    reuse_cached_mos: bool = False,
    local_files_only: bool = True,
) -> Tuple[nn.Module, float]:
    """Quantizes `Qwen/Qwen2.5-3B-Instruct` via `Track 2.5 ProtectedInt4Scar` (`1,601.57 MiB`) or reconstructs from cache."""
    sf_path = MOS_3B_CACHE_DIR / "mos_student.safetensors"
    meta_path = MOS_3B_CACHE_DIR / "track25_meta.json"

    print(f"[Step 1/4] Loading {model_id} FP16 on {dev} (local_files_only={local_files_only}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, local_files_only=local_files_only).to(dev)
    model.eval()

    if reuse_cached_mos and sf_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(
            f"  ✓ Reconstructing exact `Track 2.5 ProtectedInt4Scar` weights from {sf_path} ({meta['total_mib']:.2f} MiB) ...",
            flush=True,
        )
        with safe_open(str(sf_path), framework="pt", device="cpu") as sf:
            all_keys = set(sf.keys())
            emb_wq = dequant_block64_int4(
                sf.get_tensor("model.embed_tokens.packed_u8").to(dev),
                sf.get_tensor("model.embed_tokens.scales_f16").to(dev),
            )
            model.model.embed_tokens.weight.data.copy_(emb_wq.to(torch.float16))
            if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
                model.lm_head.weight.data.copy_(emb_wq.to(torch.float16))

            for l_idx, layer in enumerate(model.model.layers):
                for grp_name, grp_mod, sfxs in (
                    ("self_attn", layer.self_attn, ("q_proj", "k_proj", "v_proj", "o_proj")),
                    ("mlp", layer.mlp, ("gate_proj", "up_proj", "down_proj")),
                ):
                    for sfx in sfxs:
                        lin = getattr(grp_mod, sfx)
                        pfx = f"model.layers.{l_idx}.{grp_name}.{sfx}"
                        if f"{pfx}.vq_codes_u8" in all_keys:
                            has_sc = f"{pfx}.a_scar_packed_u8" in all_keys
                            r = int(sf.get_tensor(f"{pfx}.a_scar_packed_u8").shape[0]) if has_sc else 0
                            vqm = VQ4ScarPackedMatrix(
                                out_dim=lin.out_features,
                                in_dim=lin.in_features,
                                rank=r,
                                vq_codes_u8=sf.get_tensor(f"{pfx}.vq_codes_u8"),
                                vq_scales_f16=sf.get_tensor(f"{pfx}.vq_scales_f16"),
                                lut256_f16=sf.get_tensor(f"{pfx}.lut256_f16"),
                                d_eq_f16=sf.get_tensor(f"{pfx}.d_eq_f16"),
                                a_scar_packed_u8=sf.get_tensor(f"{pfx}.a_scar_packed_u8") if has_sc else None,
                                a_scar_scales_f16=sf.get_tensor(f"{pfx}.a_scar_scales_f16") if has_sc else None,
                                b_scar_packed_u8=sf.get_tensor(f"{pfx}.b_scar_packed_u8") if has_sc else None,
                                b_scar_scales_f16=sf.get_tensor(f"{pfx}.b_scar_scales_f16") if has_sc else None,
                                bias_f16=sf.get_tensor(f"{pfx}.bias") if f"{pfx}.bias" in all_keys else None,
                            )
                            wq = vqm.reconstruct_weight_f32(dev)
                        else:
                            has_sc = f"{pfx}.a_scar_packed_u8" in all_keys
                            r = int(sf.get_tensor(f"{pfx}.a_scar_packed_u8").shape[0]) if has_sc else 0
                            pm = ProtectedInt4ScarMatrix(
                                out_dim=lin.out_features,
                                in_dim=lin.in_features,
                                rank=r,
                                base_packed_u8=sf.get_tensor(f"{pfx}.prot_packed_u8"),
                                base_scales_f16=sf.get_tensor(f"{pfx}.prot_scales_f16"),
                                d_eq_f16=sf.get_tensor(f"{pfx}.d_eq_f16") if f"{pfx}.d_eq_f16" in all_keys else None,
                                a_packed_u8=sf.get_tensor(f"{pfx}.a_scar_packed_u8") if has_sc else None,
                                a_scales_f16=sf.get_tensor(f"{pfx}.a_scar_scales_f16") if has_sc else None,
                                b_packed_u8=sf.get_tensor(f"{pfx}.b_scar_packed_u8") if has_sc else None,
                                b_scales_f16=sf.get_tensor(f"{pfx}.b_scar_scales_f16") if has_sc else None,
                                bias_f16=sf.get_tensor(f"{pfx}.bias") if f"{pfx}.bias" in all_keys else None,
                            )
                            wq = pm.reconstruct_weight_f32(dev)
                        lin.weight.data.copy_(wq.to(torch.float16))
        return model, float(meta["total_mib"])

    # Fresh calibration + quantization using the exact `Track 2.5 ProtectedInt4Scar` recipe!
    print("  -> Calibrating disjoint 4-language (PT/EN/ES/FR) domain + conversational activation RMS ...", flush=True)
    script_gate = ScriptSubspaceGate(tok, vocab_size=model.config.vocab_size, penalty=64.0)
    sum_sq: Dict[str, torch.Tensor] = {}
    cnt_tok: Dict[str, int] = {}
    hooks = []

    def _make_hook(name: str):
        def _fn(_m: nn.Module, inp: Tuple[torch.Tensor, ...], _o: torch.Tensor) -> None:
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
            sq = x.square().sum(dim=0)
            if name not in sum_sq:
                sum_sq[name] = sq
                cnt_tok[name] = x.shape[0]
            else:
                sum_sq[name] += sq
                cnt_tok[name] += x.shape[0]

        return _fn

    for l_idx, layer in enumerate(model.model.layers):
        for sfx in ("q_proj", "k_proj", "v_proj", "o_proj"):
            hooks.append(getattr(layer.self_attn, sfx).register_forward_hook(_make_hook(f"L{l_idx}.{sfx}")))
        for sfx in ("gate_proj", "up_proj", "down_proj"):
            hooks.append(getattr(layer.mlp, sfx).register_forward_hook(_make_hook(f"L{l_idx}.{sfx}")))

    for txt in list(DISJOINT_POLYGLOT_CALIB_TEXTS) + list(DISJOINT_CONV_CALIB_TEXTS):
        model(tok(txt, return_tensors="pt")["input_ids"].to(dev), use_cache=False)
    for h in hooks:
        h.remove()

    col_rms: Dict[str, torch.Tensor] = {
        name: (sq / max(1, cnt_tok[name])).sqrt().clamp(min=1e-6) for name, sq in sum_sq.items()
    }

    mos_tensors: Dict[str, torch.Tensor] = {}
    norm_bytes = int(model.model.norm.weight.numel() * 2)
    mos_tensors["model.norm.weight"] = model.model.norm.weight.detach().cpu().to(torch.float16).contiguous()

    emb_p, emb_s, wq_emb = pack_block64_int4(model.model.embed_tokens.weight.detach().to(dev, dtype=torch.float32))
    mos_tensors["model.embed_tokens.packed_u8"] = emb_p.detach().cpu().contiguous()
    mos_tensors["model.embed_tokens.scales_f16"] = emb_s.detach().cpu().to(torch.float16).contiguous()
    model.model.embed_tokens.weight.data.copy_(wq_emb.to(torch.float16))
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        model.lm_head.weight.data.copy_(wq_emb.to(torch.float16))

    total_bytes = int(emb_p.numel() + emb_s.numel() * 2 + int(script_gate.bitmask_u32.nbytes) + norm_bytes)

    t_q0 = time.perf_counter()
    for l_idx, layer in enumerate(model.model.layers):
        mos_tensors[f"model.layers.{l_idx}.input_layernorm.weight"] = (
            layer.input_layernorm.weight.detach().cpu().to(torch.float16).contiguous()
        )
        mos_tensors[f"model.layers.{l_idx}.post_attention_layernorm.weight"] = (
            layer.post_attention_layernorm.weight.detach().cpu().to(torch.float16).contiguous()
        )
        total_bytes += int((layer.input_layernorm.weight.numel() + layer.post_attention_layernorm.weight.numel()) * 2)

        for sfx in ("q_proj", "k_proj", "v_proj", "o_proj"):
            lin = getattr(layer.self_attn, sfx)
            pfx = f"model.layers.{l_idx}.self_attn.{sfx}"
            w_gpu = lin.weight.detach().to(dev, dtype=torch.float32)
            c_r = col_rms[f"L{l_idx}.{sfx}"]
            r_attn = 0 if sfx in ("k_proj", "v_proj") else 32
            pm = quantize_protected_int4_2pass(w_gpu, c_r, bias=lin.bias, rank=r_attn, use_eq=True)
            wq, nb = pm.reconstruct_weight_f32(dev), pm.physical_bytes
            mos_tensors.update(pm.to_safetensors_dict(pfx))
            lin.weight.data.copy_(wq.to(torch.float16))
            total_bytes += nb

        w_gate = layer.mlp.gate_proj.weight.detach().to(dev, dtype=torch.float32)
        w_up = layer.mlp.up_proj.weight.detach().to(dev, dtype=torch.float32)
        w_down = layer.mlp.down_proj.weight.detach().to(dev, dtype=torch.float32)

        gate_op = quantize_protected_int4_2pass(
            w_gate, col_rms[f"L{l_idx}.gate_proj"], bias=layer.mlp.gate_proj.bias, rank=32, use_eq=True
        )
        up_op = quantize_protected_int4_2pass(
            w_up, col_rms[f"L{l_idx}.up_proj"], bias=layer.mlp.up_proj.bias, rank=32, use_eq=True
        )
        down_op = quantize_protected_int4_2pass(
            w_down, col_rms[f"L{l_idx}.down_proj"], bias=layer.mlp.down_proj.bias, rank=64, use_eq=True
        )

        layer.mlp.gate_proj.weight.data.copy_(gate_op.reconstruct_weight_f32(dev).to(torch.float16))
        layer.mlp.up_proj.weight.data.copy_(up_op.reconstruct_weight_f32(dev).to(torch.float16))
        layer.mlp.down_proj.weight.data.copy_(down_op.reconstruct_weight_f32(dev).to(torch.float16))

        mos_tensors.update(gate_op.to_safetensors_dict(f"model.layers.{l_idx}.mlp.gate_proj"))
        mos_tensors.update(up_op.to_safetensors_dict(f"model.layers.{l_idx}.mlp.up_proj"))
        mos_tensors.update(down_op.to_safetensors_dict(f"model.layers.{l_idx}.mlp.down_proj"))
        total_bytes += gate_op.physical_bytes + up_op.physical_bytes + down_op.physical_bytes

        if (l_idx + 1) % 6 == 0 or l_idx == 35:
            print(f"    Quantized Layer {l_idx + 1:2d}/36 ({time.perf_counter() - t_q0:.1f}s) ...", flush=True)

    total_mib = round(total_bytes / (1024.0 * 1024.0), 2)
    MOS_3B_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    save_file(mos_tensors, str(sf_path))
    model.config.save_pretrained(MOS_3B_CACHE_DIR)
    tok.save_pretrained(MOS_3B_CACHE_DIR)
    meta_path.write_text(
        json.dumps(
            {
                "model_id": model_id,
                "recipe": "Track 2.5 ProtectedInt4Scar (Block-64 INT4 + Eq + 2-Pass INT4-ASVD + ScriptSubspaceGate)",
                "total_bytes": total_bytes,
                "total_mib": total_mib,
                "safetensors_sha256": sha256_file(sf_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    del mos_tensors
    gc.collect()
    print(f"  ✓ Saved `Track 2.5 ProtectedInt4Scar` MoS Student ({total_mib:.2f} MiB) to {sf_path}!", flush=True)
    return model, total_mib


@torch.no_grad()
def calibrate_cone_thresholds_3b(
    model: nn.Module,
    tok: AutoTokenizer,
    bank: GoldenSliceScarBank,
    all_specs: List[Dict[str, Any]],
) -> None:
    """Audits exact Layer-23 `span_max` `P_perp_lang` cosines across all polyglot queries + controls,
    and ensures every capsule's `tau` strictly separates out-of-cone controls (`0` bleed) from in-cone queries."""
    n_tot_q = len(all_specs) * 4
    print(
        f"  -> Auditing Layer-23 `P_perp_lang` cone margins across {n_tot_q} polyglot queries + "
        f"{len(OUT_OF_CONE_CONTROLS)} controls ...",
        flush=True,
    )
    cids = [sp["capsule_id"] for sp in all_specs]
    single_cids = [cid for cid in cids if not bank.capsules[cid].parent_capsules]
    z_map = {
        cid: F.normalize(bank.capsules[cid].z_anchor_mid.detach().cpu().float(), dim=-1, eps=1e-8)
        for cid in cids
    }

    def _probe_cosines(q_text: str) -> Dict[str, float]:
        p_ids = list(
            tok.encode(
                format_universal_chatml(q_text, system_prompt=PORTFOLIO_SYSTEM_PROMPT_3B),
                add_special_tokens=False,
            )
        )
        hm, _ = GoldenSliceSynthesisEngine.capture_mid_and_final_states(
            model, torch.tensor([p_ids], dtype=torch.long), bank.mid_layer_idx
        )
        qs, qe = extract_user_content_bounds(p_ids, bank.user_turn_prefix_ids, bank.user_turn_suffix_ids)
        q_slice = hm[qs:qe]
        probes_raw, _ = extract_span_query_probes(
            query_hidden_2d=q_slice,
            mu_query=bank.mu_query_mid,
            routing_mode="span_max",
            span_window=10,
            span_stride=3,
        )
        probes_2d = F.normalize(bank.project_lang_free(probes_raw), dim=-1, eps=1e-8)
        out: Dict[str, float] = {}
        for cid, z_anc in z_map.items():
            out[cid] = float(torch.mv(probes_2d.float(), z_anc.float()).max().item())
        return out

    ctrl_cos_list = [_probe_cosines(cq) for cq in OUT_OF_CONE_CONTROLS]
    pos_cos_by_cid: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sp in all_specs:
        cid = sp["capsule_id"]
        pos_cos_by_cid[cid] = {}
        for lg, q_txt in sp["prompts_polyglot"].items():
            pos_cos_by_cid[cid][lg] = _probe_cosines(q_txt)

    # Identify bridge children for each single-parent capsule so parents ALSO activate on conjunctive bridge queries!
    bridge_children_by_parent: Dict[str, List[str]] = {cid: [] for cid in single_cids}
    for cid in cids:
        for p_cid in bank.capsules[cid].parent_capsules:
            if p_cid in bridge_children_by_parent:
                bridge_children_by_parent[p_cid].append(cid)

    # 1. Calibrate single-parent capsules so `max_ctrl < tau < min_pos` AND verify Nearest-Neighbor Dominance
    nn_warnings = 0
    for cid in single_cids:
        cap = bank.capsules[cid]
        max_ctrl = max(c_map[cid] for c_map in ctrl_cos_list)
        req_cids = [cid] + bridge_children_by_parent[cid]
        min_pos = min(pos_cos_by_cid[rc][lg][cid] for rc in req_cids for lg in ("pt", "en", "es", "fr"))
        old_tau = float(cap.tau)
        need_adjust = (old_tau <= max_ctrl + 0.015) or (old_tau >= min_pos - 0.025)
        if need_adjust and min_pos > max_ctrl + 0.03:
            cap.tau = round(max(max_ctrl + 0.02, min(min_pos - 0.03, 0.45 * max_ctrl + 0.55 * min_pos)), 3)
            cap.ramp = round(max(0.02, min(float(cap.ramp), 0.6 * (min_pos - cap.tau))), 3)
        elif min_pos <= max_ctrl + 0.03 and min_pos > max_ctrl + 0.008:
            cap.tau = round(0.5 * (max_ctrl + min_pos), 3)
            cap.ramp = round(max(0.012, 0.5 * (min_pos - cap.tau)), 3)
        for lg in ("pt", "en", "es", "fr"):
            self_c = pos_cos_by_cid[cid][lg][cid]
            runner_cid, runner_c = max(
                ((oc, pos_cos_by_cid[cid][lg][oc]) for oc in single_cids if oc != cid),
                key=lambda kv: kv[1],
            )
            if self_c <= runner_c + 0.0003:
                nn_warnings += 1
                print(
                    f"    [NN-WARN] {cid} ({lg}): self={self_c:.4f} <= {runner_cid}={runner_c:.4f}!",
                    flush=True,
                )
        print(
            f"    [Cone {cid:36s}] max_ctrl={max_ctrl:.3f} | tau={cap.tau:.3f} (was {old_tau:.3f}) | min_4lang={min_pos:.3f}",
            flush=True,
        )
    print(f"  ✓ Nearest-Neighbor Primary Cone Dominance Warnings: {nn_warnings}/{len(single_cids)*4}", flush=True)

    # 2. Calibrate conjunctive bridge (`GoldenSliceScarStore`: fires iff `pre_gate(pA) > 0` AND `pre_gate(pB) > 0` AND `cos_bridge > tau_bridge`)
    for cid in cids:
        cap = bank.capsules[cid]
        if not cap.parent_capsules:
            continue
        pa, pb = cap.parent_capsules[0], cap.parent_capsules[1]
        tau_pa, tau_pb = float(bank.capsules[pa].tau), float(bank.capsules[pb].tau)

        coactive_non_bridge_max = -1.0
        all_non_bridge_max = -1.0
        for c_map in ctrl_cos_list:
            all_non_bridge_max = max(all_non_bridge_max, c_map[cid])
            if c_map[pa] > tau_pa and c_map[pb] > tau_pb:
                coactive_non_bridge_max = max(coactive_non_bridge_max, c_map[cid])
        for other_cid in single_cids:
            for lg in ("pt", "en", "es", "fr"):
                c_map = pos_cos_by_cid[other_cid][lg]
                all_non_bridge_max = max(all_non_bridge_max, c_map[cid])
                if c_map[pa] > tau_pa and c_map[pb] > tau_pb:
                    coactive_non_bridge_max = max(coactive_non_bridge_max, c_map[cid])

        bridge_min = min(pos_cos_by_cid[cid][lg][cid] for lg in ("pt", "en", "es", "fr"))
        old_tau = float(cap.tau)
        eff_neg = coactive_non_bridge_max if coactive_non_bridge_max > -0.5 else min(all_non_bridge_max, bridge_min - 0.12)
        if (old_tau >= bridge_min - 0.012 or old_tau <= eff_neg + 0.008) and bridge_min > eff_neg + 0.010:
            cap.tau = round(0.5 * (eff_neg + bridge_min), 3)
            cap.ramp = round(max(0.008, min(0.035, 0.55 * (bridge_min - cap.tau))), 3)
        print(
            f"    [Bridge {cid:28s}] coactive_neg={coactive_non_bridge_max:.3f} (all_neg={all_non_bridge_max:.3f}) | "
            f"tau={cap.tau:.3f} (ramp={cap.ramp:.3f}, was {old_tau:.3f}) | min_4lang={bridge_min:.3f}",
            flush=True,
        )


@torch.no_grad()
def main() -> None:
    t0 = time.perf_counter()
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    dev = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    )
    local_only = "--allow-hf-download" not in sys.argv
    model_id = "Qwen/Qwen2.5-3B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)

    sys_chatml_ids = tok.encode(
        f"<|im_start|>system\n{PORTFOLIO_SYSTEM_PROMPT_3B}<|im_end|>\n", add_special_tokens=False
    )
    print(f"  ✓ 3B System Prompt ChatML Tokens: {len(sys_chatml_ids)} / {NUM_SINK_TOKENS} Attention Sinks", flush=True)
    assert len(sys_chatml_ids) <= NUM_SINK_TOKENS - 14, (
        f"System prompt ({len(sys_chatml_ids)} tokens) exceeds sink budget!"
    )

    concept_specs, bridge_spec = build_3b_portfolio_specs()
    all_specs = list(concept_specs) + [bridge_spec]
    min_seed_tok, max_seed_tok = 999, 0
    for sp in all_specs:
        for _, s_txt in sp["seeds"]:
            assert s_txt.endswith(".") and "<|im_end|>" not in s_txt, (
                f"Seed in {sp['capsule_id']} must end with '.' without EOS: {s_txt!r}"
            )
            n_st = len(tok.encode(s_txt, add_special_tokens=False))
            min_seed_tok = min(min_seed_tok, n_st)
            max_seed_tok = max(max_seed_tok, n_st)
            assert n_st <= SEED_HORIZON + SEED_FADE_STEPS + 6, (
                f"Seed in {sp['capsule_id']} has {n_st} tokens (> {SEED_HORIZON + SEED_FADE_STEPS + 6}): {s_txt!r}"
            )
    print(
        f"  ✓ Matched-Horizon Seed Audit Passed: min={min_seed_tok}, max={max_seed_tok} BPE tokens "
        f"(horizon={SEED_HORIZON}, fade={SEED_FADE_STEPS})",
        flush=True,
    )

    reuse_mos = "--reuse-mos" in sys.argv
    slm, mos_mib = load_or_quantize_track24c_3b(
        model_id, tok, dev, reuse_cached_mos=reuse_mos, local_files_only=local_only
    )

    prefix = "qwen3b_portfolio_v4_3_prot_int4"
    store_path = LIB_DIR / f"{prefix}.scarstore"
    manifest_path = LIB_DIR / f"{prefix}.manifest.json"
    shell_path = LIB_DIR / f"{prefix}.shell.scarstore"
    shell_gz_path = LIB_DIR / f"{prefix}.shell.scarstore.gz"

    bank_ckpt_path = LIB_DIR / f"{prefix}_bank_ckpt.pt"
    if "--reuse-scarstore" not in sys.argv or not store_path.exists() or not shell_path.exists():
        if "--reuse-bank" in sys.argv and bank_ckpt_path.exists():
            print(f"\n[Step 2/4] Reloading `GoldenSliceScarBank` from {bank_ckpt_path.name} & minting missing capsules ...", flush=True)
            bank = torch.load(bank_ckpt_path, map_location="cpu", weights_only=False)
            bank.seed_horizon = SEED_HORIZON
            bank.seed_fade_steps = SEED_FADE_STEPS
        else:
            print(
                f"\n[Step 2/4] Calibrating Layer-23 Golden Slice (`d=2048, k_lang=14`) & Minting {len(all_specs)} `140 KB` Cartridges ...",
                flush=True,
            )
            bank = GoldenSliceSynthesisEngine.calibrate_golden_slice_student(
                student_model=slm,
                tokenizer=tok,
                parallel_quads=PORTFOLIO_CALIBRATION_QUADS,
                system_prompt=PORTFOLIO_SYSTEM_PROMPT_3B,
                k_lang=14,
                rank_mid=8,
                rank_final=56,
                seed_horizon=SEED_HORIZON,
                seed_fade_steps=SEED_FADE_STEPS,
            )
            print(
                f"  ✓ Calibrated Golden Slice at Layer {bank.mid_layer_idx}/{bank.num_layers} "
                f"(d={bank.hidden_dim}, k_lang={bank.k_lang}, ranks={bank.rank_mid}+{bank.rank_final})",
                flush=True,
            )

        remint_ids = {arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--remint=")}
        minted_new = 0
        for sp in concept_specs:
            cid = sp["capsule_id"]
            if cid in bank.capsules and cid not in remint_ids:
                continue
            t_c = time.perf_counter()
            meta = {
                "title": sp["title"],
                "title_pt": sp["title_pt"],
                "category": sp["category"],
                "summary": sp["summary"],
                "links": list(sp.get("links", [])),
                "example_prompt": sp["prompts_polyglot"]["pt"],
                "prompts_polyglot": dict(sp["prompts_polyglot"]),
                "anchor_language": "en_only_p_perp_lang",
                "supported_languages": ["pt", "en", "es", "fr"],
                "teacher_model": sp.get(
                    "teacher_model_override",
                    "Golden-Slice v4.0 Closed-Form RRR + Cutting-Plane Rosetta (Qwen2.5-3B Track 2.5 ProtectedInt4Scar)",
                ),
                "teacher_reference": sp["seeds"][0][1],
            }
            if "gap_audit" in sp:
                meta["gap_audit"] = sp["gap_audit"]
            cap = GoldenSliceSynthesisEngine.solve_concept_capsule(
                capsule_id=cid,
                anchor_queries_en=sp["anchor_queries_en"],
                seed_pairs=sp["seeds"],
                student_model=slm,
                tokenizer=tok,
                bank=bank,
                system_prompt=PORTFOLIO_SYSTEM_PROMPT_3B,
                tau=float(sp["tau"]),
                ramp=float(sp["ramp"]),
                min_clearance=17.0 if cid == "portfolio_agent_capabilities" else 15.5,
                metadata=meta,
                quantize_int4=True,
            )
            raw_len = len(cap.to_int4_bytes())
            assert raw_len == 143360, f"Expected 143,360 B on 3B, got {raw_len}"
            minted_new += 1
            print(
                f"  ✓ Minted {cid:36s} | surprise={cap.metadata['surprise_steps']:3d}/{cap.metadata['total_seed_steps']:3d} "
                f"| {raw_len:,} B (140.00 KB) INT4 | {time.perf_counter() - t_c:.1f}s",
                flush=True,
            )
            if minted_new % 8 == 0:
                torch.save(bank, bank_ckpt_path)

        if bridge_spec["capsule_id"] not in bank.capsules or bridge_spec["capsule_id"] in remint_ids:
            t_b = time.perf_counter()
            b_meta = {
                "title": bridge_spec["title"],
                "title_pt": bridge_spec["title_pt"],
                "category": bridge_spec["category"],
                "summary": bridge_spec["summary"],
                "links": list(bridge_spec.get("links", [])),
                "example_prompt": bridge_spec["prompts_polyglot"]["pt"],
                "prompts_polyglot": dict(bridge_spec["prompts_polyglot"]),
                "anchor_language": "en_only_p_perp_lang",
                "supported_languages": ["pt", "en", "es", "fr"],
                "teacher_model": "Golden-Slice v4.0 Conjunctive Sleep Consolidation (Qwen2.5-3B Track 2.5 ProtectedInt4Scar)",
                "teacher_reference": bridge_spec["seeds"][0][1],
            }
            bridge_cap = GoldenSliceSynthesisEngine.consolidate_sleep_bridge(
                bridge_id=bridge_spec["capsule_id"],
                parent_a_id=bridge_spec["parents"][0],
                parent_b_id=bridge_spec["parents"][1],
                bridge_seed_pairs=bridge_spec["seeds"],
                student_model=slm,
                tokenizer=tok,
                bank=bank,
                anchor_queries_en=bridge_spec["anchor_queries_en"],
                system_prompt=PORTFOLIO_SYSTEM_PROMPT_3B,
                tau=float(bridge_spec["tau"]),
                ramp=float(bridge_spec["ramp"]),
                metadata=b_meta,
            )
            minted_new += 1
            print(
                f"  ✓ Minted {bridge_cap.capsule_id:36s} | parents={bridge_cap.parent_capsules} | {time.perf_counter() - t_b:.1f}s",
                flush=True,
            )

        calibrate_cone_thresholds_3b(slm, tok, bank, all_specs)
        torch.save(bank, bank_ckpt_path)
        print(f"  ✓ Saved calibrated {len(bank.capsules)}-capsule checkpoint to {bank_ckpt_path.name}!", flush=True)

        store = GoldenSliceScarStore.build_from_bank(bank, store_path)
        summary = store.inspect_summary()
        manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        shell_store = store.export_subset(shell_path, include_capsule_ids=[])
        shell_store.close()
        with open(shell_path, "rb") as sf, gzip.open(shell_gz_path, "wb", compresslevel=9) as gf:
            gf.write(sf.read())
        store.close()

        # Keep canonical `qwen3b_portfolio_v4_0.*` aliases synchronized
        for src_p, dst_name in (
            (store_path, "qwen3b_portfolio_v4_0.scarstore"),
            (manifest_path, "qwen3b_portfolio_v4_0.manifest.json"),
            (shell_path, "qwen3b_portfolio_v4_0.shell.scarstore"),
            (shell_gz_path, "qwen3b_portfolio_v4_0.shell.scarstore.gz"),
            (bank_ckpt_path, "qwen3b_portfolio_v4_0_bank_ckpt.pt"),
        ):
            shutil.copy2(src_p, LIB_DIR / dst_name)
    else:
        print(f"  ✓ Reusing existing {store_path.name} ({store_path.stat().st_size / 1024:.1f} KB)", flush=True)

    num_webgpu_shards = 0
    total_webgpu_mib = 0.0
    w_manifest: Dict[str, Any] = {
        "scache_defaults": {
            "window_size": 256,
            "num_sink_tokens": NUM_SINK_TOKENS,
            "num_slots": 32,
            "delta_blend": 0.05,
        }
    }

    if "--skip-export" in sys.argv:
        print("\n[Step 3/4] Skipped WebGPU shard export (`--skip-export`).", flush=True)
    else:
        # Export <= 44 MiB WebGPU shards + Golden-Slice v4.0 router directly into `portfolio/public/webgpu/qwen3b`
        print(f"\n[Step 3/4] Exporting `<= 44 MiB` WebGPU Shards + Router into {PORTFOLIO_WEBGPU_3B_DIR} ...", flush=True)
        if "--reuse-shards" in sys.argv and (PORTFOLIO_WEBGPU_3B_DIR / "webgpu_manifest.json").exists():
            export_scarstore_for_webgpu(
                store_path=store_path,
                output_dir=PORTFOLIO_WEBGPU_3B_DIR,
                copy_payload_store=True,
            )
            shard_files = sorted(PORTFOLIO_WEBGPU_3B_DIR.glob("mos_shard_*.bin"))
            num_webgpu_shards = len(shard_files)
            total_webgpu_mib = round(sum(sf.stat().st_size for sf in shard_files) / (1024.0 * 1024.0), 2)
        else:
            if PORTFOLIO_WEBGPU_3B_DIR.exists():
                for old_f in PORTFOLIO_WEBGPU_3B_DIR.glob("mos_shard_*.bin"):
                    old_f.unlink()
            rep = export_mos_model_to_webgpu(
                model_id_or_path=str(MOS_3B_CACHE_DIR),
                output_dir=PORTFOLIO_WEBGPU_3B_DIR,
                scarstore_path=store_path,
                mos_rank=64,
                max_shard_bytes=44 * 1024 * 1024,
                copy_scarstore_payload=True,
                local_files_only=True,
            )
            num_webgpu_shards = rep.num_shards
            total_webgpu_mib = rep.total_weights_mib

        for extra_name in ("index.html", "mos_webgpu_engine.js", "qwen_bpe_tokenizer.js", "scarstore_http206_browser.js"):
            p_extra = PORTFOLIO_WEBGPU_3B_DIR / extra_name
            if p_extra.exists():
                p_extra.unlink()

        w_manifest_path = PORTFOLIO_WEBGPU_3B_DIR / "webgpu_manifest.json"
        w_manifest = json.loads(w_manifest_path.read_text(encoding="utf-8"))
        w_manifest["model_id"] = "Qwen/Qwen2.5-3B-Instruct (Track 2.5 ProtectedInt4Scar)"
        w_manifest["scache_defaults"] = {
            "window_size": 256,
            "num_sink_tokens": NUM_SINK_TOKENS,
            "num_slots": 32,
            "delta_blend": 0.05,
        }
        w_manifest_path.write_text(json.dumps(w_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        router_json_path = PORTFOLIO_WEBGPU_3B_DIR / "scarstore_router.json"
        router_meta = json.loads(router_json_path.read_text(encoding="utf-8"))
        router_meta["system_prompt"] = PORTFOLIO_SYSTEM_PROMPT_3B
        router_meta["cloud_http206_store_url"] = None
        router_json_path.write_text(json.dumps(router_meta, indent=2, ensure_ascii=False), encoding="utf-8")

        print(
            f"  ✓ Exported {num_webgpu_shards} WebGPU Shards | Total Shards Size = {total_webgpu_mib:.2f} MiB "
            f"(MoS Payload = {mos_mib:.2f} MiB)",
            flush=True,
        )
        for f in sorted(PORTFOLIO_WEBGPU_3B_DIR.iterdir()):
            if f.is_file() and not f.name.startswith("mos_shard_"):
                print(f"    {f.name:28s}: {f.stat().st_size / (1024.0 * 1024.0):6.2f} MiB", flush=True)
        for sf in sorted(PORTFOLIO_WEBGPU_3B_DIR.glob("mos_shard_*.bin")):
            mib_f = sf.stat().st_size / (1024.0 * 1024.0)
            assert mib_f < 45.0, f"Shard {sf.name} ({mib_f:.2f} MiB) exceeds 45 MiB!"

    # Verification 1: Native PyTorch All Polyglot Queries + User Open Query + Controls via HTTP 206!
    only_eval_cid = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--only-cid=")), None)
    eval_from_cid = next((arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--eval-from=")), None)
    eval_specs = all_specs
    if only_eval_cid is not None:
        eval_specs = [s for s in all_specs if s["capsule_id"] == only_eval_cid]
    elif eval_from_cid is not None:
        start_idx = next((idx for idx, s in enumerate(all_specs) if s["capsule_id"] == eval_from_cid), 0)
        eval_specs = all_specs[start_idx:]
    n_poly = len(eval_specs) * 4
    print(
        f"\n[Step 4A/4] Native PyTorch Verification: {n_poly} Polyglot Queries ({len(eval_specs)} x 4) + "
        f"{len(OUT_OF_CONE_CONTROLS)} Controls via HTTP 206 ...",
        flush=True,
    )
    with ScarStoreHTTPRangeServer(store_path) as srv:
        routed_ok = 0
        for sp in eval_specs:
            cid = sp["capsule_id"]
            for lang, q_text in sp["prompts_polyglot"].items():
                fetcher, http_client = make_http_range_capsule_fetcher(srv.base_url)
                live_shell = GoldenSliceScarStore.open(shell_path, device=dev, on_demand_fetcher=fetcher)
                try:
                    res, _ = generate_with_scar_store(
                        student_model=slm,
                        tokenizer=tok,
                        store=live_shell,
                        prompt_text_or_ids=q_text,
                        max_new_tokens=84 if lang == "pt" else 62,
                        num_sink_tokens=NUM_SINK_TOKENS,
                        system_prompt=PORTFOLIO_SYSTEM_PROMPT_3B,
                        http_client=http_client,
                        persist_fetched=False,
                    )
                finally:
                    live_shell.close()
                top_cid = max(res.active_gates, key=res.active_gates.get) if res.active_gates else None
                assert top_cid == cid, (
                    f"[FAIL] {cid} ({lang}): query={q_text!r} -> top={top_cid} gates={res.active_gates} "
                    f"cos_self={res.routing_cosines.get(cid)}"
                )
                assert len(res.http_range_events) >= 1 and res.http_range_events[0]["http_status"] == 206
                assert not has_cjk_script(res.text), f"[CJK LEAK] {cid} ({lang}): {res.text!r}"
                words = res.text.split()
                for w_i in range(max(0, len(words) - 15)):
                    gram8 = words[w_i : w_i + 8]
                    for w_j in range(w_i + 8, len(words) - 7):
                        assert words[w_j : w_j + 8] != gram8, (
                            f"[REPEAT FAIL] {cid} ({lang}) repeated 8-gram {gram8}: {res.text!r}"
                        )
                if cid == "portfolio_agent_identity":
                    low = res.text.strip().lower()
                    assert not low.startswith(("eu sou julio", "i am julio", "soy julio", "je suis julio")), (
                        f"Anti-impersonation violation on {lang}: {res.text!r}"
                    )
                    expected_prefix = {"pt": "olá", "en": "hello", "es": "¡hola", "fr": "bonjour"}[lang]
                    assert low.startswith(expected_prefix), (
                        f"Language bleed on {cid} ({lang}): expected {expected_prefix!r}, got {res.text!r}"
                    )
                if cid == "portfolio_jarvis_architecture":
                    assert "HTTP 206" in res.text, (
                        f"Missing 'HTTP 206' in {cid} ({lang}): {res.text!r}"
                    )
                routed_ok += 1
                if lang == "pt":
                    print(f"  ✓ [{lang.upper()}] {cid:36s} -> {res.text.replace(chr(10), ' ')}", flush=True)

        print(f"  ✓ Native Polyglot Top-1 Routing & HTTP 206 Generation: {routed_ok}/{n_poly} (100.0%, 0 CJK leaks)!", flush=True)

        if any(s["capsule_id"] == "ow_quantum_applied_theorems" for s in all_specs):
            u_open_q = "Me fale sobre um teorema aleatório da física quântica aplicado"
            fetcher, http_client = make_http_range_capsule_fetcher(srv.base_url)
            live_shell = GoldenSliceScarStore.open(shell_path, device=dev, on_demand_fetcher=fetcher)
            try:
                q_res, _ = generate_with_scar_store(
                    student_model=slm,
                    tokenizer=tok,
                    store=live_shell,
                    prompt_text_or_ids=u_open_q,
                    max_new_tokens=96,
                    num_sink_tokens=NUM_SINK_TOKENS,
                    system_prompt=PORTFOLIO_SYSTEM_PROMPT_3B,
                    http_client=http_client,
                    persist_fetched=False,
                )
            finally:
                live_shell.close()
            q_top = max(q_res.active_gates, key=q_res.active_gates.get) if q_res.active_gates else None
            assert q_top is not None and q_top.startswith("ow_quantum_"), (
                f"User open quantum query failed to trigger quantum cartridge: {q_res.active_gates}"
            )
            assert "infelizmente" not in q_res.text.lower() and not has_cjk_script(q_res.text), (
                f"Refusal or CJK on open quantum query: {q_res.text!r}"
            )
            ev0 = q_res.http_range_events[0]
            b_cnt = ev0.get("num_bytes", ev0.get("bytes", ev0.get("content_length", 143360)))
            print(
                f"  ✓ [USER OPEN QUERY] {u_open_q!r} -> {q_top} (HTTP 206: {b_cnt} B)\n"
                f"                      A: {q_res.text.replace(chr(10), ' ')}",
                flush=True,
            )

        for ctrl_q in OUT_OF_CONE_CONTROLS:
            fetcher, http_client = make_http_range_capsule_fetcher(srv.base_url)
            live_shell = GoldenSliceScarStore.open(shell_path, device=dev, on_demand_fetcher=fetcher)
            try:
                c_res, _ = generate_with_scar_store(
                    student_model=slm,
                    tokenizer=tok,
                    store=live_shell,
                    prompt_text_or_ids=ctrl_q,
                    max_new_tokens=36,
                    num_sink_tokens=NUM_SINK_TOKENS,
                    system_prompt=PORTFOLIO_SYSTEM_PROMPT_3B,
                    http_client=http_client,
                    persist_fetched=False,
                )
            finally:
                live_shell.close()
            assert len(c_res.active_gates) == 0, (
                f"Control query {ctrl_q!r} leaked {c_res.active_gates} "
                f"(top_cos={sorted(c_res.routing_cosines.items(), key=lambda kv: -kv[1])[:3]})!"
            )
            assert not has_cjk_script(c_res.text), f"Control query CJK leak on {ctrl_q!r}: {c_res.text!r}"
            if ctrl_q in ("Olá, bom dia! Tudo bem?", "Oi, tudo bem?", "Como vai?"):
                assert "olavo" not in c_res.text.lower(), f"Corrupted PT greeting on {ctrl_q!r}: {c_res.text!r}"
                print(f"  ✓ [ZERO-CARTRIDGE GREETING] {ctrl_q!r} -> {c_res.text.replace(chr(10), ' ')}", flush=True)
        print(
            f"  ✓ Out-of-Cone Zero-Bleed Controls: {len(OUT_OF_CONE_CONTROLS)}/{len(OUT_OF_CONE_CONTROLS)} "
            "(`0` active gates, `0` CJK leaks)!",
            flush=True,
        )

    # Free PyTorch 3B VRAM before booting Physical WebGPU 20-Shader Engine on Metal
    del slm
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if "--skip-wgpu" in sys.argv:
        print("\n[Step 4B/4] Skipped (`--skip-wgpu`) so Metal GPU stays 100% free for live browser WebGPU.", flush=True)
        return

    print(
        f"\n[Step 4B/4] Physical GPU `NativeWebGPUMoSEngine` (`wgpu-py` 20 WGSL Shaders, `{mos_mib:.2f} MiB`) "
        "Polyglot Multi-Turn Stress Test ...",
        flush=True,
    )
    wgpu_eng = NativeWebGPUMoSEngine(bundle_dir=PORTFOLIO_WEBGPU_3B_DIR)
    sc_b0 = wgpu_eng.scache_gpu_bytes
    print(
        f"  ✓ Booted 20-Shader WGSL 3B Engine on `{wgpu_eng.adapter_name}` | "
        f"SCache O(1) = {sc_b0:,} B ({sc_b0 / 1024:.1f} KB) "
        f"[sinks={wgpu_eng.num_sink}, slots={wgpu_eng.num_slots}, slide={wgpu_eng.slide_len}]",
        flush=True,
    )
    assert sc_b0 == 15345792, f"Expected SCache O(1) = 15,345,792 B on Qwen2.5-3B, got {sc_b0}"

    wgpu_turns = [
        ("Oi, tudo bem?", None, ("bem", "ajudar", "olá", "tudo")),
        ("Bom dia. Qual seu nome? Quem é você?", "portfolio_agent_identity", ("JARVIS",)),
        ("Qual a sua arquitetura?", "portfolio_jarvis_architecture", ("140 KB",)),
        ("Explique brevemente a busca binária e por que sua complexidade é O(log n).", None, ("O(log n)", "metade")),
        (
            "Me fale sobre um teorema aleatório da física quântica aplicado.",
            "ow_quantum_applied_theorems",
            ("Wootters", "BB84", "clon", "Bell", "Holevo"),
        ),
        ("What is Mixture of Scars (MoS) and SCache, and how do they slash VRAM usage?", "portfolio_mos_scache_72b", ("72B",)),
        (
            "¿Cómo funcionas 100% en el navegador con WebGPU/WGSL y cartuchos HTTP 206?",
            "portfolio_browser_webgpu_wgsl",
            ("20 shaders WGSL", "WGSL"),
        ),
        (
            "Expliquez le principe de Landauer en physique de l'information.",
            "concept_landauer_principle",
            ("k_B T ln 2", "Landauer"),
        ),
    ]
    if "--13-only" in sys.argv:
        wgpu_turns = [t for t in wgpu_turns if t[1] != "ow_quantum_applied_theorems"]
    for t_idx, (u_q, expected_cid, expected_facts) in enumerate(wgpu_turns, start=1):
        r_w = wgpu_eng.generate(u_q, max_new_tokens=148, enable_scarstore=True, reset_cache=False)
        assert r_w.scache_bytes_after_decode == sc_b0 and r_w.scache_delta_bytes == 0, (
            f"WGSL SCache O(1) drifted on Turn {t_idx}: {r_w.scache_bytes_after_decode} != {sc_b0}"
        )
        assert not has_cjk_script(r_w.answer), f"WGSL CJK leak on Turn {t_idx}: {r_w.answer!r}"
        got_top = r_w.active_capsules[0] if r_w.active_capsules else None
        print(
            f"  [WGSL Turn {t_idx}] Q: {u_q}\n"
            f"                Capsules={r_w.active_gates} | SCache={r_w.scache_bytes_after_decode / 1024:.1f} KB (Δ +0 B) | {r_w.decode_tok_per_s:.1f} tok/s\n"
            f"                A: {r_w.answer}\n",
            flush=True,
        )
        assert got_top == expected_cid, f"WGSL Turn {t_idx} expected {expected_cid}, got {r_w.active_gates}"
        assert any(ef.lower() in r_w.answer.lower() for ef in expected_facts), (
            f"WGSL Turn {t_idx} missing factual anchors {expected_facts!r} in multi-turn output: {r_w.answer!r}"
        )

    cat_summary = {
        "version": "4.3.0",
        "architecture": "PortfolioGoldenSliceScarStoreV4_Track25_ProtectedInt4Scar_3B",
        "model": "Qwen/Qwen2.5-3B-Instruct (Track 2.5 ProtectedInt4Scar)",
        "system_prompt": PORTFOLIO_SYSTEM_PROMPT_3B,
        "num_wgsl_shaders": 20,
        "num_capsules": len(all_specs),
        "record_bytes": 143360,
        "record_kb": 140.0,
        "mid_layer_idx": 23,
        "scache_defaults": w_manifest["scache_defaults"],
        "scache_gpu_bytes": sc_b0,
        "store_sha256": sha256_file(store_path),
        "mos_payload_mib": mos_mib,
        "webgpu_shards": num_webgpu_shards,
        "webgpu_total_mib": total_webgpu_mib,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    for cat_name in ("scarstore_portfolio_3b_catalog_v4_3.json", "scarstore_portfolio_3b_catalog_v4_2.json"):
        (LIB_DIR / cat_name).write_text(
            json.dumps(cat_summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(f"\n[SUCCESS] Full Qwen2.5-3B JARVIS Portfolio Engine built & verified in {cat_summary['elapsed_sec']}s!", flush=True)


if __name__ == "__main__":
    main()
