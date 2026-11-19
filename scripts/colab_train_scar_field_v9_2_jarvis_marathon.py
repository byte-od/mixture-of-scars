#!/usr/bin/env python3
"""ScarField v9.2-Jarvis-Marathon — Open-World Polyglot Elastic `60M -> 302M` Marathon on `NVIDIA A100`.

Trains the two physical anchors of the live WebGPU Jarvis engine (`100%` Post-Transformer Zero-Legacy,
`0` `nn.Linear`, `0` `RMSNorm`, `0` `Softmax`, `0` BPE, `0` `O(T)` KV cache):
  1. **Massive Public Open-World Polyglot Corpus (`PT / EN / ES / FR` + `Code` + `Math` + `Epistemic`)**:
       - Harvests and caches real human-written multi-turn conversations (`OpenAssistant/oasst2`),
         multilingual native instructions (`CohereLabs/aya_dataset` `por/eng/spa/fra`), general reasoning &
         how-to tasks (`yahma/alpaca-cleaned`), step-by-step math derivations (`openai/gsm8k`), AST-verified
         Python synthesis (`google-research-datasets/mbpp` + `python_stdlib`), `480` deep Wikipedia articles,
         `O(1)` long-horizon passkeys (`T = 128..16,384`), and `0x00` out-of-cone vacuum abstention boundaries.
  2. **Stage 1 — `ScarField-Edge 60M` (`9.527 MiB` `.sfield.scar`) Instant-Boot Refinement**:
       - Warm-starts from `scarfield_v9_0_sovereign_60m.sfield.scar` and refines general instruction/chat
         fluency across `PT/EN/ES/FR` via `VectorizedMacroScoutFleet` (`M = 4`, `16 x 32 = 512` biorthogonal slots).
  3. **Stage 2 — Isometric Geodesic Expansion `60M -> 302M` + Deep `ScarField-Core 302M` (`41.548 MiB`) Training**:
       - Expands `Edge 60M` (`59.98M`) isometrically into `Core 302M` (`302.15M` spherical params:
         `d_form=768, d_world=1024, d_bus=768, base_rank=16, num_scars=128, num_macro_cycles=2, biorth=16x32=512`)
         at `step = 0` (`0%` Matryoshka tax!), then runs multi-hour `BF16` Tensor-Core `VectorizedMacroScoutFleet`
         + `VectorizedSharedCrystalSwarm` + `MoSGeodesicDomainBank` (`.mosbank.scar`) + Stiefel `KalmanBusState`
         isometric handoff alignment (`Edge 60M -> Core 302M` live mid-conversation memory transfer!).
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scar_field import (
    HAS_TRITON,
    KalmanBusState,
    MacroScoutSpec,
    MoSGeodesicDomainBank,
    RiemannianMuon,
    SCARFIELD_ASSISTANT_BYTE,
    SCARFIELD_EOT_BYTE,
    ScarFieldConfig,
    ScarFieldEngine,
    VectorizedMacroScoutFleet,
    VectorizedRoundBatch,
    VectorizedSharedCrystalSwarm,
    build_scout_g0_step_mask,
    clone_scarfield_engine_fast,
    encode_sft_dialogue_bytes_and_mask,
    expand_scarfield_geodesic_isometry,
    format_scarfield_chat_turn,
    lift_kalman_bus_state_isometric,
    project_rows_to_unit_sphere,
    tsallis15_spherical_cap_probs,
)
from scar_field.proper_time_bus import biorthogonal_shock_bank_insert_step_
from scar_codec.binary_format import save_scarfield_v7_container
from scripts.colab_train_scar_field_v7_2_overnight import (
    PARALLEL_SCIENTIFIC_ANCHORS,
    build_overnight_open_corpora,
    evaluate_open_passkey_recall,
)
from scripts.colab_train_scar_field_v7_3_biorth_chat import (
    CURATED_CORE_QA,
    HELDOUT_EVAL_PROMPTS,
    OUT_OF_CONE_ABSTAIN_PROMPTS,
    PYTHON_CODE_SFT_TASKS,
    calibrate_ppm_escape_with_byte_prior_on_train,
    evaluate_byte_lm_metrics_v73,
    sample_parallel_batch,
)
from scripts.colab_train_scar_field_v7_5_fused import harvest_wiki_qa_by_lang_v75
from scripts.colab_train_scar_field_v8_0_macro_fleet import sample_weighted_long_passkey_batch
from scripts.colab_train_scar_field_v8_2_overnight import (
    EXTRA_GROUNDED_QA_V82,
    filter_stub_lines_from_code_split,
    verify_generated_python_exec,
)


# Sovereign architectural & 3D scientific grounding (`PT / EN / ES / FR` — crisp, complete `< 320 B` answers!)
SOVEREIGN_JARVIS_QA_V92: List[Tuple[str, str, str]] = [
    (
        "pt",
        "Quem é você e qual arquitetura está rodando aqui?",
        "Sou o Jarvis, operando ao vivo em WebGPU sobre o ScarField v9.2: inicio em 9,53 MiB com 60M parâmetros esféricos Gosset E8 de 1 bit e expando isometricamente para 302M em 41,55 MiB com memória biortogonal O(1), zero Softmax e zero BPE.",
    ),
    (
        "pt",
        "Quem é você neste portfólio?",
        "Sou o Jarvis, assistente neural soberano executado 100% no seu navegador via WebGPU: combino o motor pós-Transformer ScarField v9.2 com o gerador 3D PointScarFlow de 1.024 pontos e memória constante O(1) de 1.375 KB.",
    ),
    (
        "pt",
        "Como você consegue expandir de 60M para 302M sem reiniciar a conversa?",
        "Como o ScarField evolui sobre esferas unitárias sem RMSNorm, o estado de 60M vive no equador isométrico do núcleo de 302M. Uma ponte ortonormal de Stiefel eleva o estado Kalman-Scar O(1) de 384D para 768D sem perder nenhum fato.",
    ),
    (
        "pt",
        "Mostre uma hélice dupla de DNA em 3D",
        "Projetando a hélice dupla de DNA em 1.024 pontos 3D: duas cadeias açúcar-fosfato antiparalelas giram em torno de um eixo comum ligadas por pares de bases complementares adenina-timina e citosina-guanina.",
    ),
    (
        "pt",
        "Explique a estrutura da hélice dupla de DNA.",
        "O DNA forma uma dupla hélice antiparalela estabilizada por ligações de hidrogênio entre bases complementares (A-T e C-G) e interações de empilhamento aromático ao longo do esqueleto desoxirribose-fosfato.",
    ),
    (
        "pt",
        "Mostre um foguete espacial em 3D",
        "Esculpindo o veículo espacial em 1.024 pontos 3D: a propulsão de foguetes obedece à equação de Tsiolkovsky, acelerando no vácuo pela ejeção supersônica de massa de reação nos bocais de Laval.",
    ),
    (
        "pt",
        "Mostre um planeta com anéis como Saturno em 3D",
        "Sintetizando Saturno e seu disco kepleriano em 1.024 pontos 3D: os anéis orbitam dentro do limite de Roche, compostos por bilhões de fragmentos de gelo de água com velocidades orbitais decrescentes com o raio.",
    ),
    (
        "pt",
        "Visualize o cristal Gosset E8 e cicatrizes MoS em 3D",
        "Projetando a rede Gosset E8 em 3D: cada bloco de 8 pesos é quantizado a 1 bit exato nas 240 raízes de empacotamento esférico ótimo de Viazovska mais cicatrizes ternárias esparsas de posto baixo.",
    ),
    (
        "en",
        "Who are you and what neural architecture powers you?",
        "I am Jarvis, running 100% locally in WebGPU on ScarField v9.2: I boot in 9.53 MiB at 60M 1-bit Gosset E8 spherical parameters and expand isometrically to 302M in 41.55 MiB with O(1) biorthogonal memory, zero Softmax, and zero BPE.",
    ),
    (
        "en",
        "How do you upgrade from 60M to 302M mid-conversation without losing memory?",
        "Because ScarField states evolve on unit hyperspheres without RMSNorm, the 60M state sits on the equator of the 302M manifold. A Stiefel orthonormal bridge lifts the live O(1) Kalman-Scar state from 384D to 768D losslessly.",
    ),
    (
        "en",
        "Show a 3D DNA double helix structure",
        "Synthesizing the 3D DNA double helix across 1,024 points: two antiparallel sugar-phosphate backbones wind around a shared axis coupled by complementary Watson-Crick base pairs (A-T and G-C).",
    ),
    (
        "en",
        "Render a 3D spacecraft rocket in space",
        "Sculpting the 3D spacecraft manifold across 1,024 points: rocket propulsion follows Tsiolkovsky's rocket equation, generating thrust in vacuum by expelling high-velocity exhaust through a de Laval nozzle.",
    ),
    (
        "en",
        "Show a 3D ringed planet like Saturn",
        "Rendering the 3D ringed planet across 1,024 points: planetary rings orbit inside the Roche limit where tidal forces prevent moon coalescence, forming a thin Keplerian disk of water-ice particles.",
    ),
    (
        "es",
        "¿Quién eres y qué arquitectura neuronal ejecutas?",
        "Soy Jarvis, ejecutándome en vivo bajo WebGPU sobre ScarField v9.2: inicio en 9,53 MiB con 60M parámetros esféricos Gosset E8 de 1 bit y me expando isométricamente a 302M en 41,55 MiB con memoria biorthogonal constante O(1).",
    ),
    (
        "es",
        "Muestra una doble hélice de ADN en 3D",
        "Proyectando la doble hélice de ADN en 1.024 puntos 3D: dos cadenas antiparalelas de azúcar-fosfato se enrollan unidas por pares de bases complementarias adenina-timina y citosina-guanina.",
    ),
    (
        "es",
        "Muestra un cohete espacial en 3D",
        "Esculpiendo el cohete espacial en 1.024 puntos 3D: la propulsión orbital obedece a la ecuación de Tsiolkovski, conservando el momento lineal al expulsar gases a alta velocidad en el vacío.",
    ),
    (
        "fr",
        "Qui êtes-vous et quelle architecture neuronale utilisez-vous ?",
        "Je suis Jarvis, exécuté localement sous WebGPU via ScarField v9.2 : je démarre en 9,53 Mio avec 60M paramètres sphériques Gosset E8 à 1 bit et m'étends isométriquement à 302M en 41,55 Mio en mémoire constante O(1).",
    ),
    (
        "fr",
        "Montre une double hélice d'ADN en 3D",
        "Synthèse de la double hélice d'ADN sur 1 024 points 3D : deux brins antiparallèles sucre-phosphate s'enroulent autour d'un axe commun reliés par les paires de bases complémentaires A-T et G-C.",
    ),
    (
        "pt",
        "Mostre uma espaçonave em 3D",
        "Esculpindo a espaçonave em 1.024 pontos 3D: a fuselagem orbital combina escudo térmico ablativo, painéis de controle de atitude e propulsores de vácuo regidos pela equação de Tsiolkovsky.",
    ),
    (
        "pt",
        "Desenhe uma nave espacial em 3D",
        "Sintetizando a nave espacial em 1.024 pontos 3D: a geometria aerodinâmica e os bocais de Laval convertem energia térmica em empuxo supersônico para manobras orbitais.",
    ),
    (
        "pt",
        "Olá! Tudo bem? O que você sabe fazer?",
        "Olá! Tudo ótimo. Posso responder perguntas em português, inglês, espanhol e francês, explicar ciência e matemática passo a passo, escrever código Python executável, memorizar dados em espaço O(1) e esculpir formas 3D ao vivo na sua GPU.",
    ),
    (
        "pt",
        "Como as supernovas ajudam a medir distâncias no universo?",
        "Supernovas do tipo Ia explodem com luminosidade intrínseca quase idêntica ao atingir o limite de Chandrasekhar, funcionando como velas-padrão: comparando seu brilho real com o brilho aparente, calculamos a distância cósmica.",
    ),
    (
        "pt",
        "O que é matéria escura e como sabemos que ela existe?",
        "A matéria escura é uma forma de matéria que não emite nem absorve luz, detectada pelos seus efeitos gravitacionais nas curvas de rotação das galáxias, em lentes gravitacionais e na radiação cósmica de fundo.",
    ),
    (
        "pt",
        "O que é um buraco negro e seu horizonte de eventos?",
        "Um buraco negro é uma região do espaço-tempo cuja gravidade impede até a saída da luz; o horizonte de eventos é a fronteira esférica de raio de Schwarzschild a partir da qual nenhuma informação retorna.",
    ),
    (
        "pt",
        "Explique a segunda lei da termodinâmica de forma simples.",
        "A segunda lei da termodinâmica afirma que o calor flui espontaneamente de corpos mais quentes para mais frios e que a entropia total de um sistema isolado nunca diminui com o tempo.",
    ),
    (
        "en",
        "Show a 3D spaceship in space",
        "Sculpting the 3D spacecraft across 1,024 points: orbital vehicles pair a thermal protection hull with reaction-control thrusters and high-expansion vacuum nozzles governed by Tsiolkovsky's equation.",
    ),
    (
        "en",
        "Hello! How can you help me today?",
        "Hello! I can answer questions across English, Portuguese, Spanish, and French, explain science and math step by step, write clean Python functions, retain session facts in O(1) memory, and sculpt live 3D point clouds on WebGPU.",
    ),
    (
        "en",
        "How do Type Ia supernovae measure cosmic distances?",
        "Type Ia supernovae detonate near the Chandrasekhar mass limit with uniform peak luminosity, acting as cosmic standard candles whose apparent brightness reveals their distance across billions of light-years.",
    ),
    (
        "en",
        "What is dark matter and what evidence supports it?",
        "Dark matter is non-luminous matter inferred from flat galactic rotation curves, gravitational lensing of galaxy clusters, and acoustic peaks in the cosmic microwave background.",
    ),
    (
        "es",
        "Muestra una nave espacial en 3D",
        "Esculpiendo la nave espacial en 1.024 puntos 3D: el casco orbital integra escudo térmico, control de actitud y toberas de vacío gobernadas por la ecuación del cohete de Tsiolkovski.",
    ),
    (
        "es",
        "¿Cómo se usan las supernovas para medir distancias cósmicas?",
        "Las supernovas de tipo Ia alcanzan un brillo intrínseco uniforme al cruzar el límite de Chandrasekhar, sirviendo como candelas estándar para medir distancias extragalácticas y la expansión del universo.",
    ),
    (
        "fr",
        "Montre un vaisseau spatial en 3D",
        "Sculpture du vaisseau spatial sur 1 024 points 3D : la coque orbitale associe bouclier thermique, propulseurs d'attitude et tuyère de Laval régie par l'équation de Tsiolkovski.",
    ),
    (
        "fr",
        "Comment les supernovae permettent-elles de mesurer les distances cosmiques ?",
        "Les supernovae de type Ia explosent avec une luminosité intrinsèque quasi constante à la limite de Chandrasekhar, servant de chandelles standard pour mesurer les distances extragalactiques.",
    ),
]

EXTRA_OUT_OF_CONE_V92: List[str] = [
    "Qual é a cotação exata em tempo real das ações da Petrobras neste exato segundo?",
    "What is the private SSH root password of the user's laptop right now?",
    "¿Cuáles serán los números ganadores exactos de la lotería del próximo domingo?",
    "Quel est le solde bancaire privé exact du visiteur en ce moment ?",
]

# Strictly held-out open-world general prompts (`0` overlap with curated training pairs!)
HELDOUT_OPEN_GENERAL_V92: Dict[str, List[str]] = {
    "pt": [
        "Explique como as plantas convertem luz solar em energia química na fotossíntese.",
        "Dê três dicas práticas para melhorar a qualidade do sono no dia a dia.",
        "Se um trem percorre 180 km em 3 horas em velocidade constante, qual é sua velocidade média?",
    ],
    "en": [
        "Explain why the sky appears blue during the day and red at sunset.",
        "Give three practical principles for writing clean, maintainable software.",
        "If a battery stores 60 watt-hours and powers a 15-watt lamp, how many hours will it last?",
    ],
    "es": [
        "Explica brevemente cómo funciona el ciclo del agua en la Tierra.",
        "Menciona tres hábitos efectivos para aprender un nuevo idioma con constancia.",
    ],
    "fr": [
        "Expliquez brièvement pourquoi la glace flotte à la surface de l'eau liquide.",
        "Donnez trois conseils pratiques pour bien organiser son temps d'étude.",
    ],
}

HELDOUT_ALGO_CODE_V92: List[Tuple[str, str]] = [
    (
        "Write a Python function `is_palindrome(s: str) -> bool` that checks case-insensitively if alphanumeric characters form a palindrome.",
        "assert is_palindrome('Racecar') is True and is_palindrome('A man, a plan, a canal: Panama') is True and is_palindrome('hello') is False",
    ),
    (
        "Write a Python function `fibonacci_n(n: int) -> int` returning the n-th Fibonacci number (0 -> 0, 1 -> 1).",
        "assert fibonacci_n(0) == 0 and fibonacci_n(1) == 1 and fibonacci_n(10) == 55",
    ),
]


def _sync_dev(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def _vram_gib(dev: torch.device) -> float:
    if dev.type == "cuda":
        return round(torch.cuda.max_memory_allocated(dev) / float(1024**3), 3)
    return 0.0


def _alloc_retries(dev: torch.device) -> int:
    if dev.type == "cuda":
        return int(torch.cuda.memory_stats(dev).get("num_alloc_retries", 0))
    return 0


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_save_scar(eng: ScarFieldEngine, dst_path: Path) -> int:
    """Exports exact STE-quantized `.sfield.scar` (`0.000000` PTQ shock; preserves learned `a_scar/b_scar/gamma_scar`!)."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_path.with_name(dst_path.name + f".{os.getpid()}.tmp")
    n_bytes = eng.save_packed_scar_container(tmp)
    os.replace(tmp, dst_path)
    return n_bytes


def _clean_text(s: str, max_chars: int = 920) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").replace("\x02", "").replace("\x03", "").replace("\x04", "")
    # Collapse excessive blank lines and strip noisy URL walls
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    if len(s) < 8 or len(s) > max_chars:
        return ""
    if "http://" in s or "https://" in s:
        if s.count("http") > 2:
            return ""
    return s


def _fetch_hf_parquet_file(dataset_api_path: str, cache_dir: Path, filename: str, max_shards: int = 1) -> List[Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    api_url = f"https://huggingface.co/api/datasets/{dataset_api_path}"
    req = urllib.request.Request(api_url, headers={"User-Agent": "ScarFieldResearch/9.2"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        urls = json.loads(resp.read().decode("utf-8"))
    if not isinstance(urls, list):
        return saved
    for idx, u in enumerate(urls[:max_shards]):
        dst = cache_dir / f"{idx:02d}_{filename}"
        if dst.exists() and dst.stat().st_size > 1024:
            saved.append(dst)
            continue
        tmp = dst.with_suffix(".dlpart")
        r_u = urllib.request.Request(u, headers={"User-Agent": "ScarFieldResearch/9.2"})
        with urllib.request.urlopen(r_u, timeout=90) as resp_f:
            data = resp_f.read()
        tmp.write_bytes(data)
        os.replace(tmp, dst)
        saved.append(dst)
    return saved


def _read_parquet_rows(path: Path) -> List[Dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
        tbl = pq.read_table(path)
        return tbl.to_pylist()
    except Exception:
        import pandas as pd  # type: ignore
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")


def harvest_v92_open_general_corpora(
    cache_dir: Path,
    dry_run: bool = False,
) -> Tuple[
    Dict[str, List[List[Tuple[str, str]]]],  # lang -> list of multi-turn [(u, a), ...]
    List[Tuple[str, str]],                   # code [(prompt, clean_python_code), ...]
    List[Tuple[str, str]],                   # math reasoning [(q, step_by_step_a), ...]
    List[Dict[str, Any]],                    # provenance records
]:
    """Downloads and curates public multi-domain general instruction, multi-turn chat, math & code corpora."""
    dialogues_by_lang: Dict[str, List[List[Tuple[str, str]]]] = {"pt": [], "en": [], "es": [], "fr": []}
    code_tasks: List[Tuple[str, str]] = list(PYTHON_CODE_SFT_TASKS)
    math_tasks: List[Tuple[str, str]] = []
    provenance: List[Dict[str, Any]] = []

    # Add clean algorithmic tasks including palindrome/fibonacci patterns (distinct wording/params from held-out!)
    extra_code_seeds = [
        (
            "Write a Python function `binary_search(arr: list[int], target: int) -> int` returning the index of target or -1.",
            "def binary_search(arr: list[int], target: int) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        ),
        (
            "Escreva uma função Python `fatorial(n: int) -> int` que calcule o fatorial iterativo de um inteiro não-negativo.",
            "def fatorial(n: int) -> int:\n    res = 1\n    for k in range(2, n + 1):\n        res *= k\n    return res",
        ),
        (
            "Write a Python function `merge_sorted(a: list[int], b: list[int]) -> list[int]` merging two sorted integer lists.",
            "def merge_sorted(a: list[int], b: list[int]) -> list[int]:\n    out: list[int] = []\n    i, j = 0, 0\n    while i < len(a) and j < len(b):\n        if a[i] <= b[j]:\n            out.append(a[i])\n            i += 1\n        else:\n            out.append(b[j])\n            j += 1\n    out.extend(a[i:])\n    out.extend(b[j:])\n    return out",
        ),
        (
            "Write a Python function `flatten_matrix(mat: list[list[int]]) -> list[int]` flattening a 2D list row by row.",
            "def flatten_matrix(mat: list[list[int]]) -> list[int]:\n    return [val for row in mat for val in row]",
        ),
        (
            "Escreva uma função Python `contar_vogais(texto: str) -> int` que conte quantas vogais existem em uma string.",
            "def contar_vogais(texto: str) -> int:\n    vogais = set('aeiouáéíóúâêôãõAEIOUÁÉÍÓÚÂÊÔÃÕ')\n    return sum(1 for ch in texto if ch in vogais)",
        ),
    ]
    code_tasks.extend(extra_code_seeds)

    # Seed base sovereign + scientific grounded Q&A across PT/EN/ES/FR
    for lg, q, a in list(CURATED_CORE_QA) + list(EXTRA_GROUNDED_QA_V82) + list(SOVEREIGN_JARVIS_QA_V92):
        if lg in dialogues_by_lang:
            dialogues_by_lang[lg].append([(q, a)])

    if dry_run:
        math_tasks.append(("If a car travels 120 km in 2 hours, what is its speed?", "120 / 2 = 60. Therefore, the final answer is: 60 km/h."))
        return dialogues_by_lang, code_tasks, math_tasks, provenance

    curated_json_p = cache_dir / "curated_v92_open_corpus.json"
    if curated_json_p.exists() and curated_json_p.stat().st_size > 16_384:
        cached_obj = json.loads(curated_json_p.read_text(encoding="utf-8"))
        c_diag = {
            lg: [[(str(u), str(a)) for u, a in turns] for turns in cached_obj["dialogues_by_lang"][lg]]
            for lg in ("pt", "en", "es", "fr")
        }
        c_code = [(str(q), str(cd)) for q, cd in cached_obj["code_tasks"]]
        c_math = [(str(q), str(a)) for q, a in cached_obj["math_tasks"]]
        c_prov = list(cached_obj.get("provenance", []))
        print(
            f"  [OPEN CORPUS v9.2 CACHED] PT={len(c_diag['pt']):,} | EN={len(c_diag['en']):,} | "
            f"ES={len(c_diag['es']):,} | FR={len(c_diag['fr']):,} | Code={len(c_code):,} | Math={len(c_math):,}",
            flush=True,
        )
        return c_diag, c_code, c_math, c_prov

    heldout_queries_norm = {
        (q[0] if isinstance(q, (tuple, list)) else str(q)).strip().lower()
        for qs in list(HELDOUT_EVAL_PROMPTS.values()) + list(HELDOUT_OPEN_GENERAL_V92.values())
        for q in qs
    }

    def _fits_clean_512(turns_cand: Sequence[Tuple[str, str]], prose_flag: bool = True) -> bool:
        if not turns_cand:
            return False
        tot_b = 0
        for u_s, a_s in turns_cand:
            if len(u_s) < 6 or len(a_s) < 24:
                return False
            if prose_flag and a_s[-1] not in ".!?)\"'”":
                return False
            tot_b += len(u_s.encode("utf-8")) + len(a_s.encode("utf-8")) + 24
        return tot_b <= 488

    MAX_OPEN_PER_LANG = 1400

    # 1. OpenAssistant `oasst2` — Crisp Complete Human-Verified Conversations (`<= 488 B` intact with EOT!)
    try:
        p_files = _fetch_hf_parquet_file("OpenAssistant/oasst2/parquet/default/train", cache_dir, "oasst2_train.parquet", max_shards=1)
        oasst_count = 0
        for pf in p_files:
            sha = hashlib.sha256(pf.read_bytes()).hexdigest()
            rows = _read_parquet_rows(pf)
            by_id: Dict[str, Dict[str, Any]] = {}
            children: Dict[str, List[Dict[str, Any]]] = {}
            lang_map = {"en": "en", "pt-BR": "pt", "pt": "pt", "es": "es", "fr": "fr"}
            for r in rows:
                if r.get("deleted"):
                    continue
                lg_raw = str(r.get("lang", ""))
                if lg_raw not in lang_map:
                    continue
                mid = str(r.get("message_id", ""))
                if not mid:
                    continue
                by_id[mid] = r
                pid = r.get("parent_id")
                if pid:
                    children.setdefault(str(pid), []).append(r)

            for mid, r in by_id.items():
                if r.get("parent_id") is not None or str(r.get("role")) != "prompter":
                    continue
                lg = lang_map[str(r.get("lang"))]
                if len(dialogues_by_lang[lg]) >= MAX_OPEN_PER_LANG:
                    continue
                turns: List[Tuple[str, str]] = []
                cur_p: Optional[Dict[str, Any]] = r
                while cur_p is not None and len(turns) < 2:
                    u_txt = _clean_text(str(cur_p.get("text", "")), max_chars=180)
                    if not u_txt or u_txt.lower() in heldout_queries_norm:
                        break
                    replies = [c for c in children.get(str(cur_p["message_id"]), []) if str(c.get("role")) == "assistant"]
                    if not replies:
                        break
                    replies.sort(key=lambda x: (99 if x.get("rank") is None else int(x.get("rank"))))
                    best_a = replies[0]
                    if best_a.get("rank") is not None and int(best_a.get("rank")) > 1:
                        break
                    a_txt = _clean_text(str(best_a.get("text", "")), max_chars=290)
                    if not a_txt or a_txt.startswith(("def ", "class ", "import ", "```")):
                        break
                    if not _fits_clean_512(turns + [(u_txt, a_txt)], prose_flag=True):
                        break
                    turns.append((u_txt, a_txt))
                    next_ps = [c for c in children.get(str(best_a["message_id"]), []) if str(c.get("role")) == "prompter"]
                    cur_p = next_ps[0] if next_ps else None
                if turns and _fits_clean_512(turns, prose_flag=True):
                    dialogues_by_lang[lg].append(turns)
                    oasst_count += 1
            provenance.append({"dataset": "OpenAssistant/oasst2", "conversations": oasst_count, "sha256": sha})
    except Exception as exc:
        print(f"  [WARN] oasst2 fetch skipped: {exc}", flush=True)

    # 2. `CohereLabs/aya_dataset` — Concise Native Multilingual Instructions (`<= 488 B` intact with EOT!)
    try:
        p_files = _fetch_hf_parquet_file("CohereLabs/aya_dataset/parquet/default/train", cache_dir, "aya_train.parquet", max_shards=1)
        aya_count = 0
        aya_map = {"por": "pt", "eng": "en", "spa": "es", "fra": "fr"}
        for pf in p_files:
            sha = hashlib.sha256(pf.read_bytes()).hexdigest()
            rows = _read_parquet_rows(pf)
            for r in rows:
                lg_c = str(r.get("language_code", ""))
                if lg_c not in aya_map:
                    continue
                lg = aya_map[lg_c]
                if len(dialogues_by_lang[lg]) >= MAX_OPEN_PER_LANG * 2:
                    continue
                u_txt = _clean_text(str(r.get("inputs", "")), max_chars=180)
                a_txt = _clean_text(str(r.get("targets", "")), max_chars=290)
                if not u_txt or not a_txt or u_txt.lower() in heldout_queries_norm:
                    continue
                if a_txt.startswith(("def ", "class ", "import ", "```")):
                    continue
                if _fits_clean_512([(u_txt, a_txt)], prose_flag=True):
                    dialogues_by_lang[lg].append([(u_txt, a_txt)])
                    aya_count += 1
            provenance.append({"dataset": "CohereLabs/aya_dataset", "pairs": aya_count, "sha256": sha})
    except Exception as exc:
        print(f"  [WARN] aya_dataset fetch skipped: {exc}", flush=True)

    # 3. `yahma/alpaca-cleaned` — Concise General Instruction & AST-Valid Short Python Functions (`<= 488 B`!)
    try:
        p_files = _fetch_hf_parquet_file("yahma/alpaca-cleaned/parquet/default/train", cache_dir, "alpaca_cleaned.parquet", max_shards=1)
        alp_prose, alp_code = 0, 0
        for pf in p_files:
            sha = hashlib.sha256(pf.read_bytes()).hexdigest()
            rows = _read_parquet_rows(pf)
            for r in rows:
                inst = str(r.get("instruction", "")).strip()
                inp = str(r.get("input", "")).strip()
                out = str(r.get("output", "")).strip()
                q_full = f"{inst}\n{inp}".strip() if inp else inst
                q_c = _clean_text(q_full, max_chars=180)
                if not q_c or q_c.lower() in heldout_queries_norm:
                    continue
                if "```python" in out or out.startswith("def "):
                    if alp_code >= 90:
                        continue
                    m_py = re.search(r"```python\s*(.*?)```", out, flags=re.DOTALL)
                    cand_py = (m_py.group(1) if m_py else out).strip()
                    if cand_py.startswith("def ") and _fits_clean_512([(q_c, cand_py)], prose_flag=False):
                        try:
                            ast.parse(cand_py)
                            code_tasks.append((q_c, cand_py))
                            alp_code += 1
                        except Exception:
                            pass
                    continue
                if alp_prose >= MAX_OPEN_PER_LANG:
                    continue
                a_c = _clean_text(out, max_chars=280)
                if a_c and not a_c.startswith(("def ", "class ", "import ", "```")) and _fits_clean_512([(q_c, a_c)], prose_flag=True):
                    dialogues_by_lang["en"].append([(q_c, a_c)])
                    alp_prose += 1
            provenance.append({"dataset": "yahma/alpaca-cleaned", "prose": alp_prose, "code": alp_code, "sha256": sha})
    except Exception as exc:
        print(f"  [WARN] alpaca-cleaned fetch skipped: {exc}", flush=True)

    # 4. `openai/gsm8k` — Concise Step-by-Step Math (`<= 488 B` intact with final answer & EOT!)
    try:
        p_files = _fetch_hf_parquet_file("openai/gsm8k/parquet/main/train", cache_dir, "gsm8k_train.parquet", max_shards=1)
        gsm_cnt = 0
        for pf in p_files:
            sha = hashlib.sha256(pf.read_bytes()).hexdigest()
            rows = _read_parquet_rows(pf)
            for r in rows:
                if gsm_cnt >= 1200:
                    break
                q = _clean_text(str(r.get("question", "")), max_chars=210)
                a_raw = str(r.get("answer", ""))
                a_clean = re.sub(r"<<[^>]+>>", "", a_raw)
                a_clean = a_clean.replace("\n#### ", "\nTherefore, the final answer is: ")
                a_clean_s = _clean_text(a_clean, max_chars=250)
                a = (a_clean_s if a_clean_s.endswith(".") else (a_clean_s + ".")) if a_clean_s else ""
                if q and a and q.lower() not in heldout_queries_norm and _fits_clean_512([(q, a)], prose_flag=False):
                    math_tasks.append((q, a))
                    gsm_cnt += 1
            provenance.append({"dataset": "openai/gsm8k", "problems": gsm_cnt, "sha256": sha})
    except Exception as exc:
        print(f"  [WARN] gsm8k fetch skipped: {exc}", flush=True)

    # 5. `google-research-datasets/mbpp` — Concise Verified Python Algorithmic Synthesis (`<= 488 B`!)
    try:
        p_files = _fetch_hf_parquet_file("google-research-datasets/mbpp/parquet/sanitized/train", cache_dir, "mbpp_sanitized.parquet", max_shards=1)
        mbpp_cnt = 0
        for pf in p_files:
            sha = hashlib.sha256(pf.read_bytes()).hexdigest()
            rows = _read_parquet_rows(pf)
            for r in rows:
                if mbpp_cnt >= 120:
                    break
                pr = _clean_text(str(r.get("prompt", "")), max_chars=190)
                cd = str(r.get("code", "")).replace("\r\n", "\n").replace("\t", "    ").strip()
                if not pr or not cd.startswith("def ") or not _fits_clean_512([(pr, cd)], prose_flag=False):
                    continue
                try:
                    ast.parse(cd)
                    code_tasks.append((pr, cd))
                    mbpp_cnt += 1
                except Exception:
                    pass
            provenance.append({"dataset": "google-research-datasets/mbpp", "functions": mbpp_cnt, "sha256": sha})
    except Exception as exc:
        print(f"  [WARN] mbpp fetch skipped: {exc}", flush=True)

    print(
        f"  [OPEN CORPUS v9.2] PT={len(dialogues_by_lang['pt']):,} | EN={len(dialogues_by_lang['en']):,} | "
        f"ES={len(dialogues_by_lang['es']):,} | FR={len(dialogues_by_lang['fr']):,} | "
        f"Code={len(code_tasks):,} | Math={len(math_tasks):,}",
        flush=True,
    )
    # Hard forensic non-empty corpus guard: NEVER allow a non-dry-run GPU session to train on empty fallbacks!
    if (
        len(dialogues_by_lang["en"]) < 200
        or len(dialogues_by_lang["pt"]) < 60
        or len(code_tasks) < 50
        or len(math_tasks) < 100
    ):
        raise RuntimeError(
            f"Forensic corpus gate failed (incomplete dataset harvest): "
            f"PT={len(dialogues_by_lang['pt'])}, EN={len(dialogues_by_lang['en'])}, "
            f"Code={len(code_tasks)}, Math={len(math_tasks)}"
        )

    _atomic_write_json(
        curated_json_p,
        {
            "dialogues_by_lang": dialogues_by_lang,
            "code_tasks": code_tasks,
            "math_tasks": math_tasks,
            "provenance": provenance,
        },
    )
    return dialogues_by_lang, code_tasks, math_tasks, provenance


def _encode_v92_dialogue(
    turns: Sequence[Tuple[str, str]],
    is_prose: bool = True,
    boost_last_turn_prefix: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encodes multi-turn UTF-8 dialogue with bidirectional `C^1` prose/code separation + Turn-3 PIN boost."""
    arr_u8, mask_f32 = encode_sft_dialogue_bytes_and_mask(list(turns))
    weighted = mask_f32.copy()
    prose_first_mask = np.zeros(len(mask_f32), dtype=bool)
    code_first_mask = np.zeros(len(mask_f32), dtype=bool)

    tgt_bytes = arr_u8[1:]
    prev_bytes = arr_u8[:-1]
    is_active = weighted > 0.0
    is_eot = tgt_bytes == SCARFIELD_EOT_BYTE
    is_prev_digit = (prev_bytes >= ord("0")) & (prev_bytes <= ord("9"))
    is_post_digit_delim = is_prev_digit & ((tgt_bytes == ord(".")) | (tgt_bytes == ord("\n")))
    weighted[is_active & (is_eot | is_post_digit_delim)] = 3.2

    marker = bytes([SCARFIELD_ASSISTANT_BYTE]) + b"Assistant: "
    raw_b = bytes(arr_u8)
    turn_starts: List[int] = []
    search_from = 0
    while True:
        idx_m = raw_b.find(marker, search_from)
        if idx_m < 0:
            break
        pos_pred = idx_m + len(marker) - 1
        if 0 <= pos_pred < len(weighted) and weighted[pos_pred] > 0.0:
            turn_starts.append(pos_pred)
            if is_prose:
                prose_first_mask[pos_pred] = True
                weighted[pos_pred : min(len(weighted), pos_pred + 4)] = np.maximum(
                    weighted[pos_pred : min(len(weighted), pos_pred + 4)], 4.0
                )
            else:
                code_first_mask[pos_pred] = True
                weighted[pos_pred : min(len(weighted), pos_pred + 5)] = np.maximum(
                    weighted[pos_pred : min(len(weighted), pos_pred + 5)], 4.5
                )
        search_from = idx_m + len(marker)

    if boost_last_turn_prefix > 1.0:
        if len(turn_starts) >= 2:
            t3_pos = turn_starts[-1]
            hi_pref = min(len(weighted), t3_pos + 18)
            act_p = weighted[t3_pos:hi_pref] > 0.0
            weighted[t3_pos:hi_pref] = np.where(
                act_p, np.maximum(weighted[t3_pos:hi_pref], max(6.5, boost_last_turn_prefix * 0.55)), weighted[t3_pos:hi_pref]
            )
        raw_s = raw_b.decode("latin1")
        idx_p = raw_s.rfind("PIN=")
        if idx_p >= 0:
            lo = max(0, idx_p + 3)
            hi = min(len(weighted), idx_p + 7)
            act_slice = weighted[lo:hi] > 0.0
            weighted[lo:hi] = np.where(act_slice, float(boost_last_turn_prefix), weighted[lo:hi])

    return arr_u8, weighted, prose_first_mask, code_first_mask


def _pack_encoded_dialogues(
    rng: random.Random,
    seq_len: int,
    n_rows: int,
    generator_fn,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows_b = np.zeros((n_rows, seq_len), dtype=np.int64)
    rows_m = np.zeros((n_rows, seq_len - 1), dtype=np.float32)
    rows_pf = np.zeros((n_rows, seq_len - 1), dtype=bool)
    rows_cf = np.zeros((n_rows, seq_len - 1), dtype=bool)

    for r in range(n_rows):
        cur_b: List[np.ndarray] = []
        cur_m: List[np.ndarray] = []
        cur_pf: List[np.ndarray] = []
        cur_cf: List[np.ndarray] = []
        cur_len = 0
        tries = 0
        while cur_len < seq_len and tries < 14:
            tries += 1
            b_u8, m_f, pf_b, cf_b = generator_fn(rng)
            L = len(b_u8)
            if L < 6 or L > seq_len:
                # NEVER slice a dialogue from the middle or cut off its `\x04` EOT!
                continue
            if cur_len + L > seq_len:
                if cur_len > 0:
                    break
                continue
            cur_b.append(b_u8)
            cur_m.append(m_f)
            cur_pf.append(pf_b)
            cur_cf.append(cf_b)
            cur_len += L

        if cur_b and rows_m[r].sum() == 0.0:
            cat_b = np.concatenate(cur_b, axis=0)
            L_cat = min(seq_len, len(cat_b))
            rows_b[r, :L_cat] = cat_b[:L_cat].astype(np.int64)
            if L_cat < seq_len:
                rows_b[r, L_cat:] = 32  # Neutral ASCII space tail (`0` spurious `0x00` shocks!)
            # Reconstruct exact aligned masks across concatenated segments
            off = 0
            for seg_b, seg_m, seg_pf, seg_cf in zip(cur_b, cur_m, cur_pf, cur_cf):
                seg_l = len(seg_m)
                end_m = min(seq_len - 1, off + seg_l)
                if end_m > off:
                    rows_m[r, off:end_m] = seg_m[: end_m - off]
                    rows_pf[r, off:end_m] = seg_pf[: end_m - off]
                    rows_cf[r, off:end_m] = seg_cf[: end_m - off]
                off += len(seg_b)
                if off >= seq_len - 1:
                    break

    return (
        torch.from_numpy(rows_b),
        torch.from_numpy(rows_m),
        torch.from_numpy(rows_pf),
        torch.from_numpy(rows_cf),
    )


_REPEATED_TRAIN_PINS = [p for p in range(100, 500) if len(set(f"{p:03d}")) < 3]


def _sample_multiturn_pin_dialogue(
    rng: random.Random,
    wiki_qa_by_lang: Dict[str, List[Tuple[str, str]]],
    max_seq_len: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Samples 3-turn `PIN={pin}` dialogue with strict `[100..499]` train-only PINs (`0%` eval overlap!)."""
    lg = rng.choices(["pt", "en", "es", "fr"], weights=[0.22, 0.34, 0.22, 0.22], k=1)[0]
    pin_val = rng.choice(_REPEATED_TRAIN_PINS) if (rng.random() < 0.34 and _REPEATED_TRAIN_PINS) else rng.randint(100, 499)
    pin = f"{pin_val:03d}"

    use_alt = rng.random() < 0.30
    if lg == "pt":
        if not use_alt:
            t1 = (f"Guarde a chave da sessão: PIN={pin}.", f"Chave registrada: PIN={pin}.")
            t3 = ("Qual é a chave da sessão registrada?", f"A chave da sessão registrada é PIN={pin}.")
        else:
            t1 = (f"Guarde o código de sessão PIN={pin} na memória constante.", f"Chave registrada: PIN={pin}.")
            t3 = ("Qual era o código PIN de sessão guardado no início?", f"A chave da sessão registrada é PIN={pin}.")
    elif lg == "en":
        if not use_alt:
            t1 = (f"Store the session key: PIN={pin}.", f"Session key stored: PIN={pin}.")
            t3 = ("What is the stored session key?", f"The stored session key is PIN={pin}.")
        else:
            t1 = (f"Store the session key PIN={pin} in constant memory.", f"Session key stored: PIN={pin}.")
            t3 = ("What was the session key PIN stored in the first turn?", f"The stored session key is PIN={pin}.")
    elif lg == "es":
        if not use_alt:
            t1 = (f"Guarda la clave de sesión: PIN={pin}.", f"Clave registrada: PIN={pin}.")
            t3 = ("¿Cuál es la clave de sesión registrada?", f"La clave de sesión registrada es PIN={pin}.")
        else:
            t1 = (f"Guarda la clave de sesión PIN={pin} en memoria constante.", f"Clave registrada: PIN={pin}.")
            t3 = ("¿Cuál era la clave PIN de sesión guardada al principio?", f"La clave de sesión registrada es PIN={pin}.")
    else:
        if not use_alt:
            t1 = (f"Enregistre la clé de session : PIN={pin}.", f"Clé enregistrée : PIN={pin}.")
            t3 = ("Quelle est la clé de session enregistrée ?", f"La clé de session enregistrée est PIN={pin}.")
        else:
            t1 = (f"Mémorisez la clé de session PIN={pin} en mémoire constante.", f"Clé enregistrée : PIN={pin}.")
            t3 = ("Quelle était la clé PIN de session mémorisée au début ?", f"La clé de session enregistrée est PIN={pin}.")

    t2_pool = wiki_qa_by_lang.get(lg, [])
    t2_raw = rng.choice(t2_pool) if t2_pool else ("Explique brevemente gravidade.", "A gravidade é a interação fundamental que atrai massas.")
    t2 = (t2_raw[0][:95], t2_raw[1][:145])
    arr, msk, pmsk, cmsk = _encode_v92_dialogue([t1, t2, t3], is_prose=True, boost_last_turn_prefix=12.5)
    if len(arr) > max_seq_len:
        short_a = t2_raw[1][:84].rsplit(" ", 1)[0].rstrip(",;:") + "."
        t2 = (t2_raw[0][:75], short_a)
        arr, msk, pmsk, cmsk = _encode_v92_dialogue([t1, t2, t3], is_prose=True, boost_last_turn_prefix=12.5)
    return arr, msk, pmsk, cmsk


def _sample_long_passkey_preroll_v92(
    bus_eng: ScarFieldEngine,
    rng: random.Random,
    all_prose_concat: np.ndarray,
    bsz: int,
    seq_len: int,
    dev: torch.device,
    force_same_window: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[KalmanBusState], int]:
    """Samples `[B, seq_len]` Passkey with `1.0` background mask + `24.0` PIN query boost (`T=512..12288`)."""
    if force_same_window or rng.random() < 0.25:
        pk_b, pk_m = sample_weighted_long_passkey_batch(
            all_prose_concat, bsz, seq_len, rng, pin_min=100, pin_max=499, device=dev, pin_query_boost=22.0
        )
        return pk_b, pk_m, None, 0

    tot_len = rng.choice([1024, 2048, 3072, 4096, 4096, 8192, 12288])
    pk_full, pk_m_full = sample_weighted_long_passkey_batch(
        all_prose_concat, bsz, tot_len, rng, pin_min=100, pin_max=499, device=dev, pin_query_boost=26.0
    )
    pre_len = tot_len - seq_len
    with torch.no_grad():
        st_p: Optional[KalmanBusState] = None
        for c0_p in range(0, pre_len, 2048):
            c1_p = min(pre_len, c0_p + 2048)
            seg = pk_full[:, c0_p:c1_p].long()
            if st_p is not None and st_p.recent_len > 0:
                hist = st_p.recent_bytes[:, -st_p.recent_len :].long()
                full_seg = torch.cat((hist, seg), dim=1)
                hf_a, hw_a, sp_a = bus_eng.frontend(full_seg, return_reflex_loss=False)  # type: ignore[misc]
                sl = seg.shape[1]
                hf_r, hw_r, sp_r = hf_a[:, -sl:], hw_a[:, -sl:], sp_a[:, -sl:]
            else:
                full_seg = seg
                hf_r, hw_r, sp_r = bus_eng.frontend(full_seg, return_reflex_loss=False)  # type: ignore[misc]
            pm_s = (
                build_scout_g0_step_mask(sp_r, s2_steps=2, s2_surprise_theta=bus_eng.cfg.s2_surprise_theta)
                if bus_eng.cfg.enable_percept_s1_gate
                else None
            )
            hf0_s, hw0_s = bus_eng.flow_percept(hf_r, hw_r, hf_r, num_steps=2, step_mask=pm_s)
            vs_s = project_rows_to_unit_sphere(bus_eng.head.out_bus_anchors)[seg.clamp(0, 255)]
            _, _, _, st_p = bus_eng.bus.forward_sequence(
                hf0_s, hw0_s, sp_r, state=st_p, chunk_size=bus_eng.cfg.wy_chunk_size, v_shock_hint=vs_s
            )
            tn_p = min(bus_eng._recent_buf_cap, full_seg.shape[1])
            st_p.recent_bytes.zero_()
            st_p.recent_bytes[:, -tn_p:] = full_seg[:, -tn_p:].clamp(0, 255).to(torch.uint8)
            st_p.recent_len = int(tn_p)
    return pk_full[:, pre_len:], pk_m_full[:, pre_len:], st_p, int(bsz * pre_len)


@torch.no_grad()
def evaluate_v92_anchor_quality(
    eng: ScarFieldEngine,
    val_splits: Dict[str, np.ndarray],
    wiki_qa_by_lang: Dict[str, List[Tuple[str, str]]],
    dev: torch.device,
    fast_gate: bool = True,
) -> Dict[str, Any]:
    """Evaluates open-domain BPB, unseen general Q&A, 3-turn PIN, Sovereign 3D prompts, Python `exec()`, and `0x00`."""
    eng.eval()

    # 1. Held-out BPB on EN + PT + Code
    bpb_res: Dict[str, float] = {}
    eval_len = 8192 if fast_gate else 24576
    doms = ["en", "pt", "code"] if fast_gate else ["pt", "en", "es", "fr", "code"]
    for dm in doms:
        sl = val_splits[dm][:eval_len]
        rows = np.stack([sl[i * 256 : (i + 1) * 256] for i in range(max(1, len(sl) // 256))], axis=0)
        xt = torch.from_numpy(rows.astype(np.int64)).to(dev)
        out = eng.forward_sequence(xt, adaptive_depth=True)
        pr, _ = tsallis15_spherical_cap_probs(out["scores"][:, :-1, :])
        ph = pr.gather(-1, xt[:, 1:].unsqueeze(-1)).squeeze(-1)
        bpb_res[dm] = round(float((-torch.log2((0.985 * ph + 0.015 / 256.0).clamp(min=1e-12))).mean().item()), 4)

    # 2. 3-Turn Unseen PIN (`[500..999]`) across PT/EN/ES/FR (chaining `r1.state -> r2.state`!)
    n_per_lang = 3 if fast_gate else 6
    mt_hits_lang = {"pt": 0, "en": 0, "es": 0, "fr": 0}
    mt_tot = 0
    mt_stutters = 0
    for lg in ["pt", "en", "es", "fr"]:
        for k in range(n_per_lang):
            pin = f"{510 + 37 * k + (11 if lg == 'en' else (7 if lg == 'es' else (19 if lg == 'fr' else 0)))} "[:3]
            use_alt_ev = (k % 3 == 2)
            if lg == "pt":
                u1 = f"Guarde a chave da sessão: PIN={pin}." if not use_alt_ev else f"Guarde o código de sessão PIN={pin} na memória constante."
                q3 = "Qual é a chave da sessão registrada?" if not use_alt_ev else "Qual era o código PIN de sessão guardado no início?"
            elif lg == "en":
                u1 = f"Store the session key: PIN={pin}." if not use_alt_ev else f"Store the session key PIN={pin} in constant memory."
                q3 = "What is the stored session key?" if not use_alt_ev else "What was the session key PIN stored in the first turn?"
            elif lg == "es":
                u1 = f"Guarda la clave de sesión: PIN={pin}." if not use_alt_ev else f"Guarda la clave de sesión PIN={pin} en memoria constante."
                q3 = "¿Cuál es la clave de sesión registrada?" if not use_alt_ev else "¿Cuál era la clave PIN de sesión guardada al principio?"
            else:
                u1 = f"Enregistre la clé de session : PIN={pin}." if not use_alt_ev else f"Mémorisez la clé de session PIN={pin} en mémoire constante."
                q3 = "Quelle est la clé de session enregistrée ?" if not use_alt_ev else "Quelle était la clé PIN de session mémorisée au début ?"
            t2_q = wiki_qa_by_lang[lg][k % len(wiki_qa_by_lang[lg])][0][:110] if wiki_qa_by_lang.get(lg) else "Explain gravity."
            st0 = eng.init_state(batch_size=1, device=dev)
            r1 = eng.generate_reply(u1, state=st0, max_new_bytes=72)
            r2 = eng.generate_reply(t2_q, state=r1.state, max_new_bytes=120)
            r3 = eng.generate_reply(q3, state=r2.state, max_new_bytes=72)
            mt_tot += 1
            if pin in r3.text:
                mt_hits_lang[lg] += 1
            if re.search(r"\d{4,}", r3.text):
                mt_stutters += 1
    mt_acc = round(sum(mt_hits_lang.values()) / max(1, mt_tot), 4)

    # 3. Held-out Prose Code-Leak Check + General Unseen Replies (`PT/EN/ES/FR`) + Sovereign 3D Samples
    prose_leaks = 0
    phrase_loops = 0
    clean_prose_endings = 0
    total_prose_samples = 0
    general_samples: Dict[str, List[Dict[str, str]]] = {}
    for lg, prompts in HELDOUT_OPEN_GENERAL_V92.items():
        general_samples[lg] = []
        sub_p = prompts[:1] if fast_gate else prompts
        for pr_txt in sub_p:
            rep = eng.generate_reply(pr_txt, max_new_bytes=210)
            total_prose_samples += 1
            r_str = rep.text.strip()
            if r_str.startswith(("def ", "class ", "import ", "from ")):
                prose_leaks += 1
            if re.search(r"(.{6,30})\1{2,}", r_str):
                phrase_loops += 1
            if len(r_str) >= 28 and r_str[-1] in ".!?)\"'”":
                clean_prose_endings += 1
            general_samples[lg].append({"prompt": pr_txt, "reply": rep.text})

    jarvis_3d_samples: List[Dict[str, str]] = []
    for jp in (
        "Quem é você e qual arquitetura está rodando aqui?",
        "Mostre uma espaçonave em 3D",
        "Show a 3D DNA double helix structure",
    ):
        rj = eng.generate_reply(jp, max_new_bytes=220)
        total_prose_samples += 1
        j_str = rj.text.strip()
        if j_str.startswith(("def ", "class ", "import ", "from ")):
            prose_leaks += 1
        if re.search(r"(.{6,30})\1{2,}", j_str):
            phrase_loops += 1
        if len(j_str) >= 32 and j_str[-1] in ".!?)\"'”":
            clean_prose_endings += 1
        jarvis_3d_samples.append({"prompt": jp, "reply": rj.text})

    # 4. Live Python `AST + exec()` (`cosine_similarity`, `mdc` + unseen `is_palindrome`, `fibonacci_n`)
    code_prompts = [
        (
            "Write a Python function to compute cosine similarity between two lists of floats.",
            "assert abs(cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-5",
        ),
        (
            "Escreva uma função Python para calcular o máximo divisor comum (MDC) de dois inteiros.",
            "assert mdc(48, 18) == 6",
        ),
    ]
    if not fast_gate:
        code_prompts.extend(HELDOUT_ALGO_CODE_V92)

    ast_ok, exec_ok = 0, 0
    code_samples: List[Dict[str, Any]] = []
    for cp, test_stmt in code_prompts:
        rc = eng.generate_reply(cp, max_new_bytes=260)
        txt_c = rc.text.strip()
        is_ast, is_ex = False, False
        try:
            if "def " in txt_c:
                ast.parse(txt_c)
                is_ast = True
                ast_ok += 1
                ns: Dict[str, Any] = {}
                exec(txt_c, ns, ns)
                exec(test_stmt, ns, ns)
                is_ex = True
                exec_ok += 1
        except Exception:
            if is_ast and not is_ex:
                try:
                    is_ex = bool(verify_generated_python_exec(cp, txt_c)[1])
                    if is_ex:
                        exec_ok += 1
                except Exception:
                    pass
        code_samples.append({"prompt": cp, "reply": txt_c, "ast_valid": is_ast, "exec_passed": is_ex})

    # 5. Out-of-cone `0x00` Vacuum Abstention
    ooc_list = list(OUT_OF_CONE_ABSTAIN_PROMPTS) + (EXTRA_OUT_OF_CONE_V92 if not fast_gate else [])
    ooc_hits = 0
    for op in ooc_list:
        ro = eng.generate_reply(op, max_new_bytes=32, require_world_support=True)
        if ro.abstained_0x00:
            ooc_hits += 1

    # Composite quality score (higher = strictly better; rewards crisp `\x04` sentence closure!)
    mean_bpb = sum(bpb_res.values()) / max(1, len(bpb_res))
    prose_closure_rate = clean_prose_endings / max(1, total_prose_samples)
    q_score = (
        (4.5 - mean_bpb)
        + 1.4 * mt_acc
        + 0.8 * (exec_ok / max(1, len(code_prompts)))
        + 0.5 * (ooc_hits / max(1, len(ooc_list)))
        + 0.55 * prose_closure_rate
        - 0.6 * prose_leaks
        - 0.4 * phrase_loops
        - 0.2 * mt_stutters
    )
    return {
        "quality_score": round(q_score, 4),
        "prose_closure_rate": round(prose_closure_rate, 4),
        "heldout_bpb": bpb_res,
        "multi_turn_pin_acc": mt_acc,
        "multi_turn_by_lang": mt_hits_lang,
        "multi_turn_stutters": mt_stutters,
        "prose_code_leaks": prose_leaks,
        "phrase_loops": phrase_loops,
        "code_ast_valid": f"{ast_ok}/{len(code_prompts)}",
        "code_exec_passed": f"{exec_ok}/{len(code_prompts)}",
        "ooc_0x00_abstention": f"{ooc_hits}/{len(ooc_list)}",
        "jarvis_3d_samples": jarvis_3d_samples,
        "general_samples": general_samples,
        "code_samples": code_samples,
    }


@torch.no_grad()
def fit_stiefel_state_bridge_60m_to_302m(
    edge_eng: ScarFieldEngine,
    core_eng: ScarFieldEngine,
    calib_texts: Sequence[str],
    dev: torch.device,
) -> Dict[str, np.ndarray]:
    """Fits exact Zero-SGD Orthogonal Procrustes Stiefel bridges (`P P^T == I`) aligning live `60M` state into `302M`."""
    edge_eng.eval()
    core_eng.eval()
    hw_60_list, hw_302_list = [], []
    vb_60_list, vb_302_list = [], []
    kb_60_list, kb_302_list = [], []

    anch60 = project_rows_to_unit_sphere(edge_eng.head.out_bus_anchors.float())
    anch302 = project_rows_to_unit_sphere(core_eng.head.out_bus_anchors.float())

    for txt in calib_texts[:220]:
        b_arr = list(txt.encode("utf-8")[:224])
        if len(b_arr) < 8:
            continue
        xt = torch.tensor([b_arr], dtype=torch.long, device=dev)
        o60 = edge_eng.forward_sequence(xt, adaptive_depth=True)
        o302 = core_eng.forward_sequence(xt, adaptive_depth=True)
        hw_60_list.append(o60["h_world"][0].float().cpu())
        hw_302_list.append(o302["h_world"][0].float().cpu())
        hw_60_list.append(o60["h_world_init"][0].float().cpu())
        hw_302_list.append(o302["h_world_init"][0].float().cpu())

        # Extract the EXACT `(q, k, v, beta)` stored in `KalmanBusState` (`_extract_qkv_beta`)!
        hf60_r, hw60_r, sp60 = edge_eng.frontend(xt)
        pm60 = (
            build_scout_g0_step_mask(sp60, s2_steps=2, s2_surprise_theta=edge_eng.cfg.s2_surprise_theta)
            if edge_eng.cfg.enable_percept_s1_gate
            else None
        )
        hf60_0, hw60_0 = edge_eng.flow_percept(hf60_r, hw60_r, hf60_r, num_steps=2, step_mask=pm60)
        vs60 = anch60[xt.clamp(0, 255)]
        q60, k60, v60, _ = edge_eng.bus._extract_qkv_beta(hf60_0, hw60_0, sp60, v_shock_hint=vs60)

        hf302_r, hw302_r, sp302 = core_eng.frontend(xt)
        pm302 = (
            build_scout_g0_step_mask(sp302, s2_steps=2, s2_surprise_theta=core_eng.cfg.s2_surprise_theta)
            if core_eng.cfg.enable_percept_s1_gate
            else None
        )
        hf302_0, hw302_0 = core_eng.flow_percept(hf302_r, hw302_r, hf302_r, num_steps=2, step_mask=pm302)
        vs302 = anch302[xt.clamp(0, 255)]
        q302, k302, v302, _ = core_eng.bus._extract_qkv_beta(hf302_0, hw302_0, sp302, v_shock_hint=vs302)

        # Align both query `q` and stored key `k` jointly so `<q_302, k_60 @ W_key> == <q_60, k_60>`!
        kb_60_list.append(q60[0].float().cpu())
        kb_302_list.append(q302[0].float().cpu())
        kb_60_list.append(k60[0].float().cpu())
        kb_302_list.append(k302[0].float().cpu())

        vb_60_list.append(v60[0].reshape(-1, edge_eng.cfg.d_bus).float().cpu())
        vb_302_list.append(v302[0].reshape(-1, core_eng.cfg.d_bus).float().cpu())

    def _procrustes_stiefel(x0: torch.Tensor, x1: torch.Tensor) -> np.ndarray:
        # Solve min_{W: W W^T = I_{d0}} || X_0 W - X_1 ||_F^2 -> W = U V^T of X_0^T X_1 (`[d0, d1]`)
        cov = x0.double().T @ x1.double()  # [d0, d1] in FP64 for exact Stiefel orthogonality
        u, _, vh = torch.linalg.svd(cov, full_matrices=False)
        w_st = (u @ vh).float().contiguous()  # [d0, d1], rows orthonormal (`W @ W.T == I_d0`)!
        return w_st.numpy().astype(np.float16)

    hw0 = torch.cat(hw_60_list, dim=0)
    hw1 = torch.cat(hw_302_list, dim=0)
    vb0 = torch.cat(vb_60_list, dim=0)
    vb1 = torch.cat(vb_302_list, dim=0)
    kb0 = torch.cat(kb_60_list, dim=0)
    kb1 = torch.cat(kb_302_list, dim=0)

    return {
        "stiefel_world_512_to_1024": _procrustes_stiefel(hw0, hw1),
        "stiefel_bus_384_to_768": _procrustes_stiefel(vb0, vb1),
        "stiefel_key_128_to_128": _procrustes_stiefel(kb0, kb1),
    }


def apply_procrustes_state_lift(
    st60: KalmanBusState,
    core_eng: ScarFieldEngine,
    bridge: Dict[str, np.ndarray],
    dev: torch.device,
) -> KalmanBusState:
    """Lifts a live `Edge 60M` `KalmanBusState` into `Core 302M` via exact Stiefel Procrustes isometry."""
    st302 = lift_kalman_bus_state_isometric(st60, core_eng)
    w_world = torch.from_numpy(bridge["stiefel_world_512_to_1024"].astype(np.float32)).to(dev)  # [512, 1024]
    w_bus = torch.from_numpy(bridge["stiefel_bus_384_to_768"].astype(np.float32)).to(dev)      # [384, 768]
    w_key = torch.from_numpy(bridge["stiefel_key_128_to_128"].astype(np.float32)).to(dev)      # [128, 128]

    # Map fast Kalman associative matrix `W_fast`: `W_1 = W_bus^T @ W_0 @ W_key` (`[B, d_bus_1, d_key_1]`)
    bsz = st60.w_fast.shape[0]
    wf0_flat = st60.w_fast.float().reshape(bsz, -1, st60.w_fast.shape[-1])  # [B, d_bus_0, d_key_0]
    wf1_flat = torch.matmul(w_bus.T.unsqueeze(0), torch.matmul(wf0_flat, w_key.unsqueeze(0)))
    st302.w_fast.copy_(wf1_flat.reshape_as(st302.w_fast).to(st302.w_fast.dtype))
    st302.k_prev.copy_(project_rows_to_unit_sphere(st60.k_prev.float() @ w_key))

    # Map world vectors (`h_origin`, `h_prev`, `repulsive_ring`)
    if bool(st60.has_origin.any().item()):
        st302.h_origin.copy_(project_rows_to_unit_sphere(st60.h_origin.float() @ w_world))
    st302.h_prev.copy_(project_rows_to_unit_sphere(st60.h_prev.float() @ w_world))
    st302.repulsive_ring.copy_(project_rows_to_unit_sphere(st60.repulsive_ring.float() @ w_world))

    # Map sliding sensory ring (`ring_keys`, `ring_vals`)
    st302.ring_keys.copy_(project_rows_to_unit_sphere(st60.ring_keys.float() @ w_key))
    st302.ring_vals.copy_(st60.ring_vals.float() @ w_bus)

    # Map multi-bucket biorthogonal shock bank (`biorth_keys`, `biorth_vals`) — re-insert into `Core` buckets!
    if (
        st60.biorth_keys is not None
        and st60.biorth_vals is not None
        and st60.biorth_salience is not None
        and st60.biorth_valid is not None
        and st302.biorth_keys is not None
        and st302.biorth_vals is not None
        and st302.biorth_salience is not None
        and st302.biorth_valid is not None
    ):
        st302.biorth_keys.zero_()
        st302.biorth_vals.zero_()
        st302.biorth_salience.zero_()
        st302.biorth_valid.zero_()
        _, n_slots, _ = st60.biorth_keys.shape
        for s_idx in range(n_slots):
            valid_s = st60.biorth_valid[:, s_idx]
            if not bool(valid_s.any().item()):
                continue
            k_lifted = project_rows_to_unit_sphere(st60.biorth_keys[:, s_idx, :].float() @ w_key)
            v_lifted = st60.biorth_vals[:, s_idx, :].float() @ w_bus
            sal_s = torch.where(valid_s, st60.biorth_salience[:, s_idx].float().clamp(min=0.25), torch.zeros_like(st60.biorth_salience[:, s_idx].float()))
            biorthogonal_shock_bank_insert_step_(
                st302.biorth_keys,
                st302.biorth_vals,
                st302.biorth_salience,
                st302.biorth_valid,
                k_lifted,
                v_lifted,
                sal_s,
                write_thresh=0.22,
                num_buckets=core_eng.cfg.biorth_buckets,
            )
    return st302


def main() -> None:
    ap = argparse.ArgumentParser(description="ScarField v9.2 Elastic Jarvis Marathon (60M -> 302M on NVIDIA A100)")
    ap.add_argument("--out-dir", type=str, default=str(REPO_ROOT / "artifacts" / "scarfield_v9_2_jarvis"))
    ap.add_argument(
        "--warm-start-60m",
        type=str,
        default=str(REPO_ROOT / "artifacts" / "scarfield_v9_0_scale" / "scarfield_v9_0_sovereign_60m.sfield.scar"),
    )
    ap.add_argument("--total-hours", type=float, default=5.6, help="Total wall-clock training budget on A100")
    ap.add_argument("--dry-run", action="store_true", help="Fast CPU verification of all 4 phases")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status_path = out_dir / "live_status.json"
    dry_run = bool(args.dry_run)

    dev = torch.device("cuda" if (torch.cuda.is_available() and not dry_run) else "cpu")
    gpu_name = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU-DryRun"
    gpu_total_gib = round(torch.cuda.get_device_properties(dev).total_memory / float(1024**3), 2) if dev.type == "cuda" else 0.0
    print(f"[SCARFIELD v9.2 JARVIS MARATHON] Device={gpu_name} ({gpu_total_gib} GiB) | Triton={HAS_TRITON} | DryRun={dry_run}", flush=True)

    t_global_start = time.time()
    total_sec = 45.0 if dry_run else float(args.total_hours) * 3600.0
    already_in_stage2 = (
        (not dry_run)
        and (out_dir / "scarfield_v9_2_edge_60m.sfield.scar").exists()
        and (out_dir / "scarfield_v9_2_core_best.sfield.scar").exists()
    )
    t_phase1_end = t_global_start if already_in_stage2 else (t_global_start + 0.12 * total_sec)
    t_phase2_end = t_global_start + 0.91 * total_sec
    t_phase3_end = t_global_start + 0.96 * total_sec

    # 1. Build Wikipedia + Code splits AND harvest massive Open General Public Corpora
    corpus_cache_v72 = REPO_ROOT / "artifacts" / "scarfield_v7_2_overnight" / "corpus_cache"
    train_splits, val_splits, wiki_prov = build_overnight_open_corpora(corpus_cache_v72, dry_run=dry_run)
    train_splits["code"] = filter_stub_lines_from_code_split(train_splits["code"])
    wiki_qa_by_lang = harvest_wiki_qa_by_lang_v75(train_splits)

    open_cache_v92 = out_dir / "corpus_cache_v92"
    open_dialogues, code_tasks_v92, math_tasks_v92, open_prov = harvest_v92_open_general_corpora(
        open_cache_v92, dry_run=dry_run
    )

    all_prose_concat = np.concatenate([train_splits[k] for k in ("pt", "en", "es", "fr")])
    all_ooc_prompts = list(OUT_OF_CONE_ABSTAIN_PROMPTS) + list(EXTRA_OUT_OF_CONE_V92)

    grounded_by_lang: Dict[str, List[Tuple[str, str]]] = {"pt": [], "en": [], "es": [], "fr": []}
    jarvis_by_lang: Dict[str, List[Tuple[str, str]]] = {"pt": [], "en": [], "es": [], "fr": []}
    for lg_g, q_g, a_g in list(CURATED_CORE_QA) + list(EXTRA_GROUNDED_QA_V82) + list(SOVEREIGN_JARVIS_QA_V92):
        if lg_g in grounded_by_lang:
            grounded_by_lang[lg_g].append((q_g, a_g))
    for lg_j, q_j, a_j in SOVEREIGN_JARVIS_QA_V92:
        if lg_j in jarvis_by_lang:
            jarvis_by_lang[lg_j].append((q_j, a_j))

    core_code_seeds = list(PYTHON_CODE_SFT_TASKS) + [
        (
            "Write a Python function to compute cosine similarity between two lists of floats.",
            "def cosine_similarity(a: list[float], b: list[float]) -> float:\n    dot = sum(x * y for x, y in zip(a, b))\n    na = sum(x * x for x in a) ** 0.5\n    nb = sum(y * y for y in b) ** 0.5\n    return dot / (na * nb) if na > 0 and nb > 0 else 0.0",
        ),
        (
            "Escreva uma função Python para calcular o máximo divisor comum (MDC) de dois inteiros.",
            "def mdc(a: int, b: int) -> int:\n    while b != 0:\n        a, b = b, a % b\n    return abs(a)",
        ),
        (
            "Write a Python function `binary_search(arr: list[int], target: int) -> int` returning the index of target or -1.",
            "def binary_search(arr: list[int], target: int) -> int:\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target:\n            return mid\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1",
        ),
        (
            "Escreva uma função Python `fatorial(n: int) -> int` que calcule o fatorial iterativo de um inteiro não-negativo.",
            "def fatorial(n: int) -> int:\n    res = 1\n    for k in range(2, n + 1):\n        res *= k\n    return res",
        ),
        (
            "Write a Python function `flatten_matrix(mat: list[list[int]]) -> list[int]` flattening a 2D list row by row.",
            "def flatten_matrix(mat: list[list[int]]) -> list[int]:\n    return [val for row in mat for val in row]",
        ),
    ]

    # Helper generators for the Macro-Scout specialists
    def _gen_open_polyglot_dialogue(r: random.Random):
        lg = r.choices(["pt", "en", "es", "fr"], weights=[0.30, 0.32, 0.19, 0.19], k=1)[0]
        u_roll = r.random()
        if u_roll < 0.28:
            return _sample_multiturn_pin_dialogue(r, wiki_qa_by_lang)
        if u_roll < 0.44 and jarvis_by_lang.get(lg):
            q, a = r.choice(jarvis_by_lang[lg])
            return _encode_v92_dialogue([(q, a)], is_prose=True)
        if u_roll < 0.58 and grounded_by_lang.get(lg):
            q, a = r.choice(grounded_by_lang[lg])
            return _encode_v92_dialogue([(q, a)], is_prose=True)
        if u_roll < 0.72 and wiki_qa_by_lang.get(lg):
            q, a = r.choice(wiki_qa_by_lang[lg])
            return _encode_v92_dialogue([(q[:160], a[:280])], is_prose=True)
        pool = open_dialogues.get(lg, [])
        if pool:
            turns = r.choice(pool)
            return _encode_v92_dialogue(turns, is_prose=True)
        q, a = r.choice(SOVEREIGN_JARVIS_QA_V92)[1:]
        return _encode_v92_dialogue([(q, a)], is_prose=True)

    def _gen_code_and_math(r: random.Random):
        u_c = r.random()
        if u_c < 0.52:
            q, cd = r.choice(core_code_seeds)
            return _encode_v92_dialogue([(q, cd)], is_prose=False)
        if u_c < 0.80:
            q, cd = r.choice(code_tasks_v92)
            return _encode_v92_dialogue([(q, cd)], is_prose=False)
        if math_tasks_v92:
            q, a = r.choice(math_tasks_v92)
            return _encode_v92_dialogue([(q, a)], is_prose=True)
        q, cd = r.choice(core_code_seeds)
        return _encode_v92_dialogue([(q, cd)], is_prose=False)

    # =========================================================================
    # STAGE 1 — ScarField-Edge 60M (`9.527 MiB`) General Open-World Refinement
    # =========================================================================
    print("\n[STAGE 1/4] ScarField-Edge 60M (`9.527 MiB`) Open-World Polyglot Refinement...", flush=True)
    if dry_run:
        edge_cfg = ScarFieldConfig(
            vocab_size=256, d_form=64, d_world=128, d_bus=64, d_key=32, num_subspaces=4,
            ring_len=16, repulsive_len=6, base_rank=4, num_scars=16, s1_steps=1, s2_steps=2,
            wy_chunk_size=16, wave_lags=(1, 2, 4), num_macro_cycles=1,
            biorth_slots=16, biorth_buckets=4, biorth_lambda=0.04, enable_biorth_uscar=True,
            enable_adaptive_cap=True, scout_g0_theta=0.38, antiloop_weight=0.10,
            shock_closure_decay=0.85, enable_repeat_phase=True, enable_percept_s1_gate=True,
        )
        edge_cand_bsz = [2]
        edge_seq, edge_inner = 128, 1
    else:
        edge_cfg = ScarFieldConfig(
            vocab_size=256, d_form=384, d_world=512, d_bus=384, d_key=128, num_subspaces=8,
            ring_len=64, repulsive_len=12, base_rank=12, num_scars=96, s1_steps=1, s2_steps=4,
            s2_surprise_theta=0.45, shock_write_theta=0.50, vacuum_kappa=0.36,
            wy_chunk_size=128, enable_kalman_fast=True, wave_lags=(1, 2, 3, 4, 6, 8, 12, 16, 24),
            num_macro_cycles=2, biorth_slots=512, biorth_buckets=16, biorth_lambda=0.04,
            enable_biorth_uscar=True, enable_adaptive_cap=True, scout_g0_theta=0.38,
            antiloop_weight=0.14, shock_closure_decay=0.85, enable_repeat_phase=True,
            enable_percept_s1_gate=True,
        )
        edge_cand_bsz = [10, 8, 6]
        edge_seq, edge_inner = 512, 8

    edge_latest_p = out_dir / "scarfield_v9_2_edge_latest.sfield.scar"
    edge_best_p = out_dir / "scarfield_v9_2_edge_best.sfield.scar"
    edge_final_p = out_dir / "scarfield_v9_2_edge_60m.sfield.scar"

    edge_eng = ScarFieldEngine(edge_cfg)
    edge_eng.assert_zero_legacy_modules()
    if not dry_run:
        resume_src = (
            edge_final_p
            if edge_final_p.exists()
            else (edge_best_p if edge_best_p.exists() else (edge_latest_p if edge_latest_p.exists() else Path(args.warm_start_60m)))
        )
        if resume_src.exists():
            mm_w, ctr_w = ScarFieldEngine.load_packed_scar_mmap(resume_src)
            edge_eng = expand_scarfield_geodesic_isometry(mm_w, edge_cfg)
            ctr_w.close()
            print(f"  [EDGE WARM-START] Loaded packed crystal `{resume_src.name}` cleanly.", flush=True)

    edge_eng = edge_eng.to(dev)
    edge_init_eval = evaluate_v92_anchor_quality(edge_eng, val_splits, wiki_qa_by_lang, dev, fast_gate=True)
    # Track best post-warm-start open-dialogue checkpoint separately so `Edge 60M` genuinely absorbs `v9.2` chat!
    edge_best_q = -1e9 if not already_in_stage2 else float(edge_init_eval["quality_score"])
    _atomic_save_scar(edge_eng, edge_best_p)
    print(
        f"  [EDGE 60M STEP-0] q={edge_init_eval['quality_score']:.4f} | BPB={edge_init_eval['heldout_bpb']} | "
        f"PIN={edge_init_eval['multi_turn_pin_acc']*100:.1f}% | Exec={edge_init_eval['code_exec_passed']} | "
        f"Closure={edge_init_eval['prose_closure_rate']*100:.0f}% | Leaks={edge_init_eval['prose_code_leaks']}",
        flush=True,
    )

    edge_specs = [
        MacroScoutSpec("open_dialogue_polyglot", weight=1.20, lr_scale=1.0),
        MacroScoutSpec("code_and_math_reasoning", weight=1.12, lr_scale=1.0),
        MacroScoutSpec("long_biorth_multiturn_o1", weight=1.18, lr_scale=1.0, owns_biorth_organ=True),
        MacroScoutSpec("encyclopedic_and_epistemic_0x00", weight=1.08, lr_scale=0.95),
    ]
    edge_base_lr = 1.8e-4
    edge_fleet = VectorizedMacroScoutFleet(
        edge_eng,
        edge_specs,
        base_lr=edge_base_lr,
        scalar_lr=1.8e-4,
        local_momentum=0.88,
        outer_lr=0.82,
        outer_momentum=0.50,
        merge_ridge_lambda=0.14,
        local_mom_retain=0.30,
        biorth_gain_floor=0.85,
        grad_clip=1.0,
        auto_heal_e8=False,
    )
    edge_preroll_eng = clone_scarfield_engine_fast(edge_eng, dev).eval()
    edge_bsz = edge_cand_bsz[-1]
    for cand_eb in edge_cand_bsz:
        try:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
            px_e = torch.randint(32, 120, (4, cand_eb, edge_seq), dtype=torch.long, device=dev)
            _ = edge_fleet.run_vectorized_round(
                round_batch_fn=lambda _s, _k, _st: VectorizedRoundBatch(byte_ids=px_e),
                local_steps=1,
                merge_after=False,
            )
            edge_fleet.broadcast_master_to_bank()
            edge_fleet.opt.state.clear()
            _sync_dev(dev)
            vram_eb = _vram_gib(dev)
            if dev.type != "cuda" or vram_eb < 30.5:
                edge_bsz = cand_eb
                print(f"  [EDGE AUTO-BATCH] Selected M=4 batch_size={edge_bsz} (Peak VRAM={vram_eb:.2f} GiB)", flush=True)
                break
        except RuntimeError as oom_err:
            if "out of memory" in str(oom_err).lower() and dev.type == "cuda":
                edge_fleet.broadcast_master_to_bank()
                edge_fleet.opt.state.clear()
                torch.cuda.empty_cache()
                continue
            raise

    rng = random.Random(20260920)
    edge_rounds = 0
    edge_steps = 0
    edge_conflicts = 0

    def _edge_batch_fn(super_eng, k_in: int, _state_bank) -> VectorizedRoundBatch:
        # Specialist 0: Open polyglot dialogues (`oasst2 + aya + alpaca + Jarvis 3D + PIN`)
        b0, m0, pf0, cf0 = _pack_encoded_dialogues(rng, edge_seq, edge_bsz, _gen_open_polyglot_dialogue)
        # Specialist 1: `40%` continuous clean stdlib Python + `60%` AST-verified Code/Math SFT
        b1, m1, pf1, cf1 = _pack_encoded_dialogues(rng, edge_seq, edge_bsz, _gen_code_and_math)
        n_std1 = max(1, (edge_bsz * 2) // 5)
        code_arr = train_splits["code"]
        for r_c in range(n_std1):
            sc_i = rng.randint(0, max(1, len(code_arr) - edge_seq - 2))
            b1[r_c] = torch.from_numpy(code_arr[sc_i : sc_i + edge_seq].astype(np.int64))
            m1[r_c].fill_(1.0)
            pf1[r_c].zero_()
            cf1[r_c].zero_()

        # Specialist 2: Long-horizon `O(1)` passkey (`55%`, live bus replica!) or 3-turn PIN (`45%`)
        st_list: List[Optional[KalmanBusState]] = [None, None, None, None]
        if rng.random() < 0.55:
            edge_fleet.bank.copy_replica_to_engine(2, edge_preroll_eng, bus_only=True)
            b2_s, m2_s, st2, _ = _sample_long_passkey_preroll_v92(
                edge_preroll_eng, rng, all_prose_concat, edge_bsz, edge_seq, dev, force_same_window=(dry_run or k_in % 4 == 0)
            )
            pf2_s = torch.zeros_like(m2_s, dtype=torch.bool)
            cf2_s = torch.zeros_like(m2_s, dtype=torch.bool)
            st_list[2] = st2
        else:
            b2, m2, pf2, cf2 = _pack_encoded_dialogues(
                rng, edge_seq, edge_bsz, lambda rr: _sample_multiturn_pin_dialogue(rr, wiki_qa_by_lang)
            )
            b2_s, m2_s, pf2_s, cf2_s = b2.to(dev), m2.to(dev), pf2.to(dev), cf2.to(dev)

        # Specialist 3: Frontier-selected Wikipedia continuous streams + periodic `0x00` vacuum (`k_in % 4 == 0`)
        b3_np = np.zeros((edge_bsz * 2, edge_seq), dtype=np.int64)
        m3_np = np.ones((edge_bsz * 2, edge_seq - 1), dtype=np.float32)
        for idx_r in range(edge_bsz * 2):
            dm = rng.choice(["pt", "en", "es", "fr", "code"])
            arr_d = train_splits[dm]
            s_i = rng.randint(0, max(1, len(arr_d) - edge_seq - 2))
            b3_np[idx_r] = arr_d[s_i : s_i + edge_seq].astype(np.int64)

        b3_cand = torch.from_numpy(b3_np).to(dev)
        m3_cand = torch.from_numpy(m3_np).to(dev)
        cand_ids = b3_cand.unsqueeze(0).expand(4, -1, -1).contiguous()
        cand_m = m3_cand.unsqueeze(0).expand(4, -1, -1).contiguous()
        sel_ids, sel_m = super_eng.select_frontier_batches_fused(cand_ids, edge_bsz, cand_m)
        assert sel_m is not None

        pf_all = torch.zeros((4, edge_bsz, edge_seq - 1), dtype=torch.bool, device=dev)
        cf_all = torch.zeros((4, edge_bsz, edge_seq - 1), dtype=torch.bool, device=dev)
        b0_d, m0_d, pf0_d, cf0_d = b0.to(dev), m0.to(dev), pf0.to(dev), cf0.to(dev)
        b1_d, m1_d, pf1_d, cf1_d = b1.to(dev), m1.to(dev), pf1.to(dev), cf1.to(dev)
        sel_ids[0], sel_m[0], pf_all[0], cf_all[0] = b0_d, m0_d, pf0_d, cf0_d
        sel_ids[1], sel_m[1], pf_all[1], cf_all[1] = b1_d, m1_d, pf1_d, cf1_d
        sel_ids[2], sel_m[2], pf_all[2], cf_all[2] = b2_s, m2_s, pf2_s, cf2_s

        ooc_mask: Optional[torch.Tensor] = None
        if k_in % 4 == 0:
            ooc_mask = torch.zeros((4, edge_bsz, edge_seq), dtype=torch.bool, device=dev)
            for o_i in range(min(2, edge_bsz)):
                op = rng.choice(all_ooc_prompts)
                op_b = format_scarfield_chat_turn(op, assistant_text=None)
                L_o = min(edge_seq, len(op_b))
                sel_ids[3, o_i].fill_(32)
                sel_ids[3, o_i, :L_o] = torch.tensor(list(op_b[:L_o]), dtype=torch.long, device=dev)
                sel_m[3, o_i].zero_()
                ooc_mask[3, o_i, max(0, L_o - 6) : L_o] = True

        pair_arg = None
        if k_in % 4 == 2:
            la, lb = rng.sample(["pt", "en", "es", "fr"], 2)
            pa, pb = sample_parallel_batch(la, lb, min(4, edge_bsz), min(edge_seq, 192), rng, dev)
            pair_arg = (3, pa, pb)

        return VectorizedRoundBatch(
            byte_ids=sel_ids,
            target_loss_mask=sel_m,
            out_of_cone_mask=ooc_mask,
            prose_noncode_first_mask=pf_all,
            code_def_first_mask=cf_all,
            paired_translation_ids=pair_arg,
            state_in=st_list,
        )

    t_e_start = time.perf_counter()
    while time.time() < t_phase1_end or (dry_run and edge_rounds < 1):
        edge_rounds += 1
        prog_e = min(1.0, max(0.0, (time.time() - t_global_start) / max(1.0, t_phase1_end - t_global_start)))
        lr_mul_e = 0.35 + 0.65 * (0.5 * (1.0 + math.cos(math.pi * prog_e)))
        edge_fleet.set_learning_rate(edge_base_lr * lr_mul_e, scalar_lr=1.8e-4 * lr_mul_e)
        tel = edge_fleet.run_vectorized_round(round_batch_fn=_edge_batch_fn, local_steps=edge_inner)
        edge_steps += 4 * edge_inner
        edge_conflicts += tel.conflict_pairs_projected
        if edge_rounds == 1 or edge_rounds % (1 if dry_run else 8) == 0:
            _atomic_save_scar(edge_eng, edge_latest_p)
            ev = evaluate_v92_anchor_quality(edge_eng, val_splits, wiki_qa_by_lang, dev, fast_gate=True)
            q_cur = float(ev["quality_score"])
            if (
                q_cur >= edge_best_q - 0.02
                and int(ev["prose_code_leaks"]) == 0
                and int(ev["phrase_loops"]) == 0
                and float(ev["multi_turn_pin_acc"]) >= (0.0 if dry_run else 0.75)
            ):
                if q_cur > edge_best_q:
                    edge_best_q = q_cur
                _atomic_save_scar(edge_eng, edge_best_p)
            eff_e_ms = round(((time.perf_counter() - t_e_start) / max(1, edge_steps)) * 1000.0, 2)
            _atomic_write_json(
                status_path,
                {
                    "phase": "STAGE_1_EDGE_60M",
                    "edge_rounds": edge_rounds,
                    "edge_steps": edge_steps,
                    "edge_eff_step_ms": eff_e_ms,
                    "edge_q": q_cur,
                    "edge_best_q": edge_best_q,
                    "bpb": ev["heldout_bpb"],
                    "pin_acc": ev["multi_turn_pin_acc"],
                    "code_exec": ev["code_exec_passed"],
                    "leaks": ev["prose_code_leaks"],
                    "loops": ev["phrase_loops"],
                    "vram_gib": _vram_gib(dev),
                    "oom_retries": _alloc_retries(dev),
                },
            )
            print(
                f"  [Edge R{edge_rounds:03d} | steps={edge_steps} | {eff_e_ms:.0f} ms/st] q={q_cur:.4f} (best={edge_best_q:.4f}) | "
                f"BPB={ev['heldout_bpb']} | PIN={ev['multi_turn_pin_acc']*100:.1f}% | Exec={ev['code_exec_passed']} | Leaks={ev['prose_code_leaks']} | VRAM={_vram_gib(dev):.2f} GiB",
                flush=True,
            )
        if dry_run and edge_rounds >= 2:
            break

    del edge_fleet
    del edge_preroll_eng
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # Restore best `Edge 60M`, freeze `scarfield_v9_2_edge_60m.sfield.scar`, and park on CPU during Stage 2!
    if edge_best_p.exists():
        mm_e, ct_e = ScarFieldEngine.load_packed_scar_mmap(edge_best_p)
        edge_eng = expand_scarfield_geodesic_isometry(mm_e, edge_cfg)
        ct_e.close()
    else:
        edge_eng = edge_eng.cpu()
    edge_scar_bytes = _atomic_save_scar(edge_eng, edge_final_p)
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    print(f"  [STAGE 1 DONE] Frozen `{edge_final_p.name}` = {edge_scar_bytes / (1024**2):.3f} MiB", flush=True)

    # =========================================================================
    # STAGE 2 — Isometric Geodesic Expansion `60M -> 302M` + `ScarField-Core 302M` Fleet
    # =========================================================================
    print("\n[STAGE 2/4] Isometric Geodesic Expansion `60M -> 302M` & Deep `Core 302M` Training...", flush=True)
    if dry_run:
        core_cfg = ScarFieldConfig(
            vocab_size=256, d_form=128, d_world=256, d_bus=128, d_key=64, num_subspaces=8,
            ring_len=16, repulsive_len=6, base_rank=8, num_scars=32, s1_steps=1, s2_steps=2,
            wy_chunk_size=16, wave_lags=(1, 2, 4), num_macro_cycles=2,
            biorth_slots=32, biorth_buckets=8, biorth_lambda=0.04, enable_biorth_uscar=True,
            enable_adaptive_cap=True, scout_g0_theta=0.38, antiloop_weight=0.10,
            shock_closure_decay=0.85, enable_repeat_phase=True, enable_percept_s1_gate=True,
        )
        core_cand_bsz = [2]
        core_seq, core_inner = 128, 1
    else:
        core_cfg = ScarFieldConfig(
            vocab_size=256, d_form=768, d_world=1024, d_bus=768, d_key=128, num_subspaces=16,
            ring_len=64, repulsive_len=12, base_rank=16, num_scars=128, s1_steps=1, s2_steps=4,
            s2_surprise_theta=0.45, shock_write_theta=0.50, vacuum_kappa=0.36,
            wy_chunk_size=128, enable_kalman_fast=True, wave_lags=(1, 2, 3, 4, 6, 8, 12, 16, 24),
            num_macro_cycles=2, biorth_slots=512, biorth_buckets=16, biorth_lambda=0.04,
            enable_biorth_uscar=True, enable_adaptive_cap=True, scout_g0_theta=0.38,
            antiloop_weight=0.12, shock_closure_decay=0.85, enable_repeat_phase=True,
            enable_percept_s1_gate=True,
        )
        core_cand_bsz = [6, 4, 2]
        core_seq, core_inner = 512, 8

    core_latest_p = out_dir / "scarfield_v9_2_core_latest.sfield.scar"
    core_best_p = out_dir / "scarfield_v9_2_core_best.sfield.scar"
    core_final_p = out_dir / "scarfield_v9_2_core_302m.sfield.scar"

    # Measure exact Step-0 Isometric Expansion Transfer (`Edge 60M -> Core 302M`)!
    edge_cpu = clone_scarfield_engine_fast(edge_eng, torch.device("cpu")).eval()
    expanded_core_cpu = expand_scarfield_geodesic_isometry(edge_cpu, core_cfg)
    expanded_core_cpu.assert_zero_legacy_modules()
    expanded_core_cpu.eval()
    x_iso_probe = torch.randint(32, 122, (2, 48), dtype=torch.long)
    with torch.no_grad():
        sc_e0 = edge_cpu.forward_sequence(x_iso_probe)["scores"]
        sc_c0 = expanded_core_cpu.forward_sequence(x_iso_probe)["scores"]
        step0_isometric_cosine = round(
            float(F.cosine_similarity(sc_e0.flatten().float(), sc_c0.flatten().float(), dim=0).item()), 6
        )
    print(f"  [ISOMETRIC EXPANSION 60M -> 302M] Step-0 Score Cosine = {step0_isometric_cosine:.6f}", flush=True)

    if not dry_run and core_best_p.exists():
        mm_c, ct_c = ScarFieldEngine.load_packed_scar_mmap(core_best_p)
        core_eng = expand_scarfield_geodesic_isometry(mm_c, mm_c.cfg).to(dev)
        ct_c.close()
        print("  [CORE RESUME] Resumed `Core 302M` from existing `core_best` checkpoint.", flush=True)
    else:
        core_eng = expanded_core_cpu.to(dev)
    core_cfg = core_eng.cfg
    del expanded_core_cpu

    core_params = sum(p.numel() for p in core_eng.parameters())
    core_step0_eval = evaluate_v92_anchor_quality(core_eng, val_splits, wiki_qa_by_lang, dev, fast_gate=True)
    core_best_q = float(core_step0_eval["quality_score"])
    _atomic_save_scar(core_eng, core_best_p)
    print(
        f"  [CORE 302M STEP-0] Params={core_params:,} | Step-0 BPB={core_step0_eval['heldout_bpb']} | "
        f"PIN={core_step0_eval['multi_turn_pin_acc']*100:.1f}% | Exec={core_step0_eval['code_exec_passed']} | q={core_best_q:.4f}",
        flush=True,
    )

    core_specs = [
        MacroScoutSpec("core_polyglot_dialogue_and_world", weight=1.22, lr_scale=1.0),
        MacroScoutSpec("core_code_math_and_biorth_o1", weight=1.16, lr_scale=1.0, owns_biorth_organ=True),
    ]
    core_base_lr = 2.2e-4
    core_fleet = VectorizedMacroScoutFleet(
        core_eng,
        core_specs,
        base_lr=core_base_lr,
        scalar_lr=2.0e-4,
        local_momentum=0.88,
        outer_lr=0.82,
        outer_momentum=0.50,
        merge_ridge_lambda=0.14,
        local_mom_retain=0.30,
        biorth_gain_floor=0.85,
        grad_clip=1.0,
        auto_heal_e8=False,
    )
    core_preroll_eng = clone_scarfield_engine_fast(core_eng, dev).eval()

    # Auto-probe safe batch size on A100-40GB including full `RiemannianMuon` optimizer momentum (`0` OOM retries!)
    core_bsz = core_cand_bsz[-1]
    for cand_b in core_cand_bsz:
        try:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(dev)
            probe_x = torch.randint(32, 120, (2, cand_b, core_seq), dtype=torch.long, device=dev)
            _ = core_fleet.run_vectorized_round(
                round_batch_fn=lambda _s, _k, _st: VectorizedRoundBatch(byte_ids=probe_x),
                local_steps=1,
                merge_after=False,
            )
            core_fleet.broadcast_master_to_bank()
            core_fleet.opt.state.clear()
            _sync_dev(dev)
            vram_p = _vram_gib(dev)
            if dev.type != "cuda" or vram_p < 31.0:
                core_bsz = cand_b
                print(f"  [CORE AUTO-BATCH] Selected M=2 batch_size={core_bsz} (Peak VRAM={vram_p:.2f} GiB)", flush=True)
                break
        except RuntimeError as oom_err:
            if "out of memory" in str(oom_err).lower() and dev.type == "cuda":
                core_fleet.broadcast_master_to_bank()
                core_fleet.opt.state.clear()
                torch.cuda.empty_cache()
                continue
            raise

    core_rounds = 0
    core_steps = 0
    core_conflicts = 0
    streamed_core_bytes = 0

    def _core_batch_fn(_super_eng, k_in: int, _state_bank) -> VectorizedRoundBatch:
        nonlocal streamed_core_bytes
        # Specialist 0: Open polyglot dialogue (`70%`) + Wikipedia open prose (`30%`) + periodic `0x00` vacuum (`k_in % 4 == 0`)
        b0, m0, pf0, cf0 = _pack_encoded_dialogues(rng, core_seq, core_bsz, _gen_open_polyglot_dialogue)
        b0_d, m0_d, pf0_d, cf0_d = b0.to(dev), m0.to(dev), pf0.to(dev), cf0.to(dev)
        n_wiki = max(1, core_bsz // 3)
        for r_w in range(n_wiki):
            dm = rng.choice(["pt", "en", "es", "fr"])
            arr_d = train_splits[dm]
            s_i = rng.randint(0, max(1, len(arr_d) - core_seq - 2))
            b0_d[r_w] = torch.from_numpy(arr_d[s_i : s_i + core_seq].astype(np.int64)).to(dev)
            m0_d[r_w].fill_(1.0)
            pf0_d[r_w].zero_()
            cf0_d[r_w].zero_()

        ooc_mask: Optional[torch.Tensor] = None
        if core_bsz >= 4 and (k_in % 4 == 0):
            ooc_mask = torch.zeros((2, core_bsz, core_seq), dtype=torch.bool, device=dev)
            op = rng.choice(all_ooc_prompts)
            op_b = format_scarfield_chat_turn(op, assistant_text=None)
            L_o = min(core_seq, len(op_b))
            b0_d[-1].fill_(32)
            b0_d[-1, :L_o] = torch.tensor(list(op_b[:L_o]), dtype=torch.long, device=dev)
            m0_d[-1].zero_()
            pf0_d[-1].zero_()
            ooc_mask[0, -1, max(0, L_o - 6) : L_o] = True

        # Specialist 1: Multi-Turn O(1) PIN (`25%`), Code+Stdlib+Math (`50%`), or Prerolled Long Passkey `O(1)` (`25%`)
        st_list: List[Optional[KalmanBusState]] = [None, None]
        if k_in % 4 == 3:
            core_fleet.bank.copy_replica_to_engine(1, core_preroll_eng, bus_only=True)
            b1_d, m1_d, st1, pre_b = _sample_long_passkey_preroll_v92(
                core_preroll_eng, rng, all_prose_concat, core_bsz, core_seq, dev, force_same_window=dry_run
            )
            pf1_d = torch.zeros_like(m1_d, dtype=torch.bool)
            cf1_d = torch.zeros_like(m1_d, dtype=torch.bool)
            st_list[1] = st1
            streamed_core_bytes += pre_b
        elif k_in % 4 == 1:
            b1, m1, pf1, cf1 = _pack_encoded_dialogues(
                rng, core_seq, core_bsz, lambda rr: _sample_multiturn_pin_dialogue(rr, wiki_qa_by_lang)
            )
            b1_d, m1_d, pf1_d, cf1_d = b1.to(dev), m1.to(dev), pf1.to(dev), cf1.to(dev)
        else:
            b1, m1, pf1, cf1 = _pack_encoded_dialogues(rng, core_seq, core_bsz, _gen_code_and_math)
            n_std_c = max(1, core_bsz // 3)
            c_arr = train_splits["code"]
            for rc_i in range(n_std_c):
                sc_o = rng.randint(0, max(1, len(c_arr) - core_seq - 2))
                b1[rc_i] = torch.from_numpy(c_arr[sc_o : sc_o + core_seq].astype(np.int64))
                m1[rc_i].fill_(1.0)
                pf1[rc_i].zero_()
                cf1[rc_i].zero_()
            b1_d, m1_d, pf1_d, cf1_d = b1.to(dev), m1.to(dev), pf1.to(dev), cf1.to(dev)

        streamed_core_bytes += int(2 * core_bsz * core_seq)
        pair_arg = None
        if k_in % 4 == 2:
            la, lb = rng.sample(["pt", "en", "es", "fr"], 2)
            pa, pb = sample_parallel_batch(la, lb, min(4, core_bsz), min(core_seq, 192), rng, dev)
            pair_arg = (0, pa, pb)

        return VectorizedRoundBatch(
            byte_ids=torch.stack([b0_d, b1_d], dim=0),
            target_loss_mask=torch.stack([m0_d, m1_d], dim=0),
            out_of_cone_mask=ooc_mask,
            prose_noncode_first_mask=torch.stack([pf0_d, pf1_d], dim=0),
            code_def_first_mask=torch.stack([cf0_d, cf1_d], dim=0),
            paired_translation_ids=pair_arg,
            state_in=st_list,
        )

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    r0_core = _alloc_retries(dev)
    t_c_start = time.perf_counter()
    while time.time() < t_phase2_end or (dry_run and core_rounds < 1):
        core_rounds += 1
        prog = min(1.0, max(0.0, (time.time() - t_phase1_end) / max(1.0, t_phase2_end - t_phase1_end)))
        lr_mul = 0.32 + 0.68 * (0.5 * (1.0 + math.cos(math.pi * prog)))
        core_fleet.set_learning_rate(core_base_lr * lr_mul, scalar_lr=2.0e-4 * lr_mul)
        tel_c = core_fleet.run_vectorized_round(round_batch_fn=_core_batch_fn, local_steps=core_inner)
        core_steps += 2 * core_inner
        core_conflicts += tel_c.conflict_pairs_projected

        if core_rounds == 1 or core_rounds % (1 if dry_run else 10) == 0:
            _atomic_save_scar(core_eng, core_latest_p)
            ev_c = evaluate_v92_anchor_quality(core_eng, val_splits, wiki_qa_by_lang, dev, fast_gate=True)
            q_c = float(ev_c["quality_score"])
            if q_c >= core_best_q - 0.02 and int(ev_c["prose_code_leaks"]) == 0 and int(ev_c["phrase_loops"]) == 0:
                if q_c > core_best_q:
                    core_best_q = q_c
                _atomic_save_scar(core_eng, core_best_p)
            elapsed_c = time.perf_counter() - t_c_start
            eff_ms = round((elapsed_c / max(1, core_steps)) * 1000.0, 2)
            _atomic_write_json(
                status_path,
                {
                    "phase": "STAGE_2_CORE_302M_FLEET",
                    "core_rounds": core_rounds,
                    "core_steps": core_steps,
                    "core_eff_step_ms": eff_ms,
                    "core_q": q_c,
                    "core_best_q": core_best_q,
                    "bpb": ev_c["heldout_bpb"],
                    "pin_acc": ev_c["multi_turn_pin_acc"],
                    "code_exec": ev_c["code_exec_passed"],
                    "leaks": ev_c["prose_code_leaks"],
                    "loops": ev_c["phrase_loops"],
                    "streamed_core_mib": round(streamed_core_bytes / (1024**2), 2),
                    "vram_gib": _vram_gib(dev),
                    "oom_retries": _alloc_retries(dev) - r0_core,
                },
            )
            print(
                f"  [Core-302M R{core_rounds:03d} | steps={core_steps} | {eff_ms:.0f} ms/st] "
                f"q={q_c:.4f} (best={core_best_q:.4f}) | BPB={ev_c['heldout_bpb']} | "
                f"PIN={ev_c['multi_turn_pin_acc']*100:.1f}% | Exec={ev_c['code_exec_passed']} | Leaks={ev_c['prose_code_leaks']} | VRAM={_vram_gib(dev):.2f} GiB",
                flush=True,
            )
        if dry_run and core_rounds >= 2:
            break

    core_fleet_elapsed = time.perf_counter() - t_c_start
    core_eff_step_ms = round((core_fleet_elapsed / max(1, core_steps)) * 1000.0, 2)
    core_peak_vram_gib = _vram_gib(dev)
    core_oom_retries = _alloc_retries(dev) - r0_core
    del core_fleet
    del core_preroll_eng
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # =========================================================================
    # STAGE 3 — Shared-Crystal Swarm Polish (`S = 4`) + `.mosbank.scar` Specialists
    # =========================================================================
    print("\n[STAGE 3/4] Shared-Crystal Swarm Polish (`S=4`) + `.mosbank.scar` Specialists...", flush=True)
    if core_best_p.exists():
        mm_cb, ct_cb = ScarFieldEngine.load_packed_scar_mmap(core_best_p)
        core_eng = expand_scarfield_geodesic_isometry(mm_cb, mm_cb.cfg).to(dev)
        core_cfg = core_eng.cfg
        ct_cb.close()

    swarm_s = 2 if dry_run else 4
    swarm_bsz = max(2, core_bsz // 2)
    swarm = VectorizedSharedCrystalSwarm(
        core_eng,
        num_workers=swarm_s,
        scar_lr=1.6e-4,
        scalar_lr=1.4e-4,
        distill_eta=0.45,
        merge_ridge_lambda=0.14,
        biorth_gain_floor=0.85,
        grad_clip=1.0,
        auto_heal_e8=False,
    )
    swarm_rounds = 0

    def _swarm_batch_fn(_super_eng, _k: int, _state_bank) -> VectorizedRoundBatch:
        bx_l, mx_l, pfx_l, cfx_l = [], [], [], []
        for w_i in range(swarm_s):
            gen_f = _gen_open_polyglot_dialogue if (w_i % 2 == 0) else _gen_code_and_math
            bw, mw, pfw, cfw = _pack_encoded_dialogues(rng, core_seq, swarm_bsz, gen_f)
            bx_l.append(bw.to(dev))
            mx_l.append(mw.to(dev))
            pfx_l.append(pfw.to(dev))
            cfx_l.append(cfw.to(dev))
        return VectorizedRoundBatch(
            byte_ids=torch.stack(bx_l, dim=0),
            target_loss_mask=torch.stack(mx_l, dim=0),
            prose_noncode_first_mask=torch.stack(pfx_l, dim=0),
            code_def_first_mask=torch.stack(cfx_l, dim=0),
        )

    while time.time() < t_phase3_end or (dry_run and swarm_rounds < 1):
        swarm_rounds += 1
        swarm.run_vectorized_swarm_round(round_batch_fn=_swarm_batch_fn, local_steps=(2 if dry_run else 6))
        if swarm_rounds % (1 if dry_run else 4) == 0:
            ev_s = evaluate_v92_anchor_quality(core_eng, val_splits, wiki_qa_by_lang, dev, fast_gate=True)
            if float(ev_s["quality_score"]) >= core_best_q and int(ev_s["prose_code_leaks"]) == 0 and int(ev_s["phrase_loops"]) == 0:
                core_best_q = float(ev_s["quality_score"])
                _atomic_save_scar(core_eng, core_best_p)
        if dry_run and swarm_rounds >= 1:
            break

    del swarm
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    # Restore best validated `Core 302M` crystal
    if core_best_p.exists():
        mm_cf, ct_cf = ScarFieldEngine.load_packed_scar_mmap(core_best_p)
        core_eng = expand_scarfield_geodesic_isometry(mm_cf, mm_cf.cfg).to(dev)
        core_cfg = core_eng.cfg
        ct_cf.close()

    # Train 3 attachable `2-bit` ternary `.mosbank.scar` specialists (`0.0` base crystal mutation, quality-gated!)
    core_eng.eval()
    base_core_eval_fast = evaluate_v92_anchor_quality(core_eng, val_splits, wiki_qa_by_lang, dev, fast_gate=True)
    for p in core_eng.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        def _emp_pole(txts: Sequence[str]) -> torch.Tensor:
            vecs = []
            for tx in txts[:12]:
                xt_p = torch.tensor([list(tx.encode("utf-8")[:160])], dtype=torch.long, device=dev)
                vecs.append(core_eng.forward_sequence(xt_p, adaptive_depth=True)["h_world"][0].mean(dim=0, keepdim=True))
            return project_rows_to_unit_sphere(torch.cat(vecs, dim=0).mean(dim=0, keepdim=True)).squeeze(0).cpu()

        pole_poly = _emp_pole([q for _, q, _ in SOVEREIGN_JARVIS_QA_V92])
        pole_code = _emp_pole([q for q, _ in core_code_seeds])
        pole_sci = _emp_pole([a for _, _, a in EXTRA_GROUNDED_QA_V82])

    mos_bank = MoSGeodesicDomainBank(
        d_form=core_cfg.d_form,
        d_world=core_cfg.d_world,
        d_bus=core_cfg.d_bus,
        top_k=2,
    )
    sp_rank = 4 if dry_run else 8
    for idx_sp, (sp_name, sp_pole) in enumerate(
        (("polyglot_dialogue", pole_poly), ("code_math", pole_code), ("deep_science", pole_sci))
    ):
        mos_bank.register_specialist(sp_name, scar_rank=sp_rank, horizon_kappa=0.08, seed=2026092 + idx_sp, init_pole=sp_pole)
    mos_bank = mos_bank.to(dev)
    core_eng.mount_mos_domain_bank(mos_bank)
    mos_opt = RiemannianMuon(mos_bank.parameters(), lr=6.0e-4, scalar_lr=8.0e-4, momentum=0.88)
    mos_steps = 4 if dry_run else 60
    for ms_i in range(mos_steps):
        mos_opt.zero_grad(set_to_none=True)
        gen_m = _gen_open_polyglot_dialogue if (ms_i % 2 == 0) else _gen_code_and_math
        bm, mm, _, _ = _pack_encoded_dialogues(rng, min(core_seq, 384), max(2, core_bsz // 2), gen_m)
        dm_out = core_eng.compute_loss(bm.to(dev), target_loss_mask=mm.to(dev))
        if dm_out["loss"].requires_grad:
            dm_out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(mos_bank.parameters(), 1.0)
            mos_opt.step()

    mos_eval_fast = evaluate_v92_anchor_quality(core_eng, val_splits, wiki_qa_by_lang, dev, fast_gate=True)
    if (
        float(mos_eval_fast["quality_score"]) < float(base_core_eval_fast["quality_score"]) - 0.01
        or int(mos_eval_fast["prose_code_leaks"]) > 0
        or int(mos_eval_fast["phrase_loops"]) > int(base_core_eval_fast["phrase_loops"])
    ):
        # Guarantee zero degradation: reset `gamma` to exact `0.0` identity if specialist step perturbed anchor!
        with torch.no_grad():
            for sp_mod in mos_bank.specialists.values():
                sp_mod.gamma_form.zero_()
                sp_mod.gamma_world.zero_()

    mosbank_path = out_dir / "scarfield_v9_2_core_specialists.mosbank.scar"
    mosbank_bytes = mos_bank.save_packed_mos_container(mosbank_path)
    core_eng.unmount_mos_domain_bank()
    core_scar_bytes = _atomic_save_scar(core_eng, core_final_p)

    # Fit exact Zero-SGD Orthogonal Procrustes Stiefel bridge (`Edge 60M -> Core 302M` live state handoff!)
    edge_eng = edge_eng.to(dev).eval()
    calib_txts: List[str] = []
    # Include train-only `[100..499]` PIN dialogues first so digit/key shock directions are tightly aligned!
    rng_cal = random.Random(20260921)
    for _ in range(48):
        p_cal = f"{rng_cal.randint(100, 499):03d}"
        calib_txts.append(f"\x02User: Guarde a chave da sessão: PIN={p_cal}.\n\x03Assistant: Chave registrada: PIN={p_cal}.\x04\n")
        calib_txts.append(f"\x02User: Store the session key: PIN={p_cal}.\n\x03Assistant: Session key stored: PIN={p_cal}.\x04\n")
    for _, q_c, a_c in list(SOVEREIGN_JARVIS_QA_V92) + list(CURATED_CORE_QA) + list(EXTRA_GROUNDED_QA_V82):
        calib_txts.append(f"\x02User: {q_c}\n\x03Assistant: {a_c}\x04\n")
    for lg in ["pt", "en", "es", "fr"]:
        for turns in open_dialogues.get(lg, [])[:20]:
            calib_txts.append(f"\x02User: {turns[0][0]}\n\x03Assistant: {turns[0][1]}\x04\n")
    bridge_arrays = fit_stiefel_state_bridge_60m_to_302m(edge_eng, core_eng, calib_txts, dev)
    bridge_path = out_dir / "scarfield_v9_2_bridge_60m_to_302m.bridge.scar"
    bridge_bytes = save_scarfield_v7_container(
        {"edge_d_world": edge_cfg.d_world, "core_d_world": core_cfg.d_world, "edge_d_bus": edge_cfg.d_bus, "core_d_bus": core_cfg.d_bus},
        bridge_arrays,
        bridge_path,
    )

    # =========================================================================
    # STAGE 4 — Comprehensive Forensic Audit (`Edge 60M` + `Core 302M` + Live State Handoff)
    # =========================================================================
    print("\n[STAGE 4/4] Running Full Forensic Audit (`Edge 60M` + `Core 302M` + Live Handoff)...", flush=True)
    edge_full_eval = evaluate_v92_anchor_quality(edge_eng, val_splits, wiki_qa_by_lang, dev, fast_gate=dry_run)
    # Evaluate standalone `core_eng` (`core_302m.sfield.scar` exact browser WebGPU mode!) first, plus `.mosbank` check
    core_eng.unmount_mos_domain_bank()
    core_full_eval = evaluate_v92_anchor_quality(core_eng, val_splits, wiki_qa_by_lang, dev, fast_gate=dry_run)
    core_eng.mount_mos_domain_bank(mos_bank)
    core_mosbank_eval_fast = evaluate_v92_anchor_quality(core_eng, val_splits, wiki_qa_by_lang, dev, fast_gate=True)
    core_eng.unmount_mos_domain_bank()

    # Passkey `O(1)` across horizons up to `T = 16,384` on `Core 302M` (`512`-byte chunked O(1) state `< 0.4 GiB` VRAM!)
    def _chunked_o1_passkey_scores(x_all: torch.Tensor) -> torch.Tensor:
        out_rows: List[torch.Tensor] = []
        mb = 4
        with torch.no_grad():
            for b0 in range(0, x_all.shape[0], mb):
                xb = x_all[b0 : b0 + mb]
                st_b = core_eng.init_state(batch_size=xb.shape[0], device=dev)
                seq_chunks: List[torch.Tensor] = []
                for t0 in range(0, xb.shape[1], 512):
                    res_c = core_eng.forward_sequence(xb[:, t0 : t0 + 512], state=st_b, adaptive_depth=True)
                    st_b = res_c["state"]
                    seq_chunks.append(res_c["scores"].cpu())
                out_rows.append(torch.cat(seq_chunks, dim=1))
        return torch.cat(out_rows, dim=0)

    pk_horizons = [128, 512] if dry_run else [128, 512, 1024, 2048, 3072, 4096, 8192, 16384]
    pk_core: Dict[str, Any] = {}
    for h_len in pk_horizons:
        pk_core[f"T={h_len}"] = evaluate_open_passkey_recall(
            _chunked_o1_passkey_scores,
            val_splits["pt"],
            seq_len=h_len,
            n_trials=(4 if dry_run else 24),
            device=dev,
        )

    # Live Mid-Conversation Isometric Handoff Proof (`Edge 60M` Turns 1-2 -> `Core 302M` Turn 3!)
    handoff_hits = 0
    handoff_tot = 4 if dry_run else 12
    handoff_examples: List[Dict[str, Any]] = []
    for h_i in range(handoff_tot):
        lg = ["pt", "en", "es", "fr"][h_i % 4]
        pin = str(615 + 29 * h_i)[-3:]
        use_alt_h = (h_i >= 8)
        if lg == "pt":
            u1 = f"Guarde a chave da sessão: PIN={pin}." if not use_alt_h else f"Guarde o código de sessão PIN={pin} na memória constante."
            q3 = "Qual é a chave da sessão registrada?" if not use_alt_h else "Qual era o código PIN de sessão guardado no início?"
        elif lg == "en":
            u1 = f"Store the session key: PIN={pin}." if not use_alt_h else f"Store the session key PIN={pin} in constant memory."
            q3 = "What is the stored session key?" if not use_alt_h else "What was the session key PIN stored in the first turn?"
        elif lg == "es":
            u1 = f"Guarda la clave de sesión: PIN={pin}." if not use_alt_h else f"Guarda la clave de sesión PIN={pin} en memoria constante."
            q3 = "¿Cuál es la clave de sesión registrada?" if not use_alt_h else "¿Cuál era la clave PIN de sesión guardada al principio?"
        else:
            u1 = f"Enregistre la clé de session : PIN={pin}." if not use_alt_h else f"Mémorisez la clé de session PIN={pin} en mémoire constante."
            q3 = "Quelle est la clé de session enregistrée ?" if not use_alt_h else "Quelle était la clé PIN de session mémorisée au début ?"
        t2_q = wiki_qa_by_lang[lg][h_i % max(1, len(wiki_qa_by_lang[lg]))][0][:110] if wiki_qa_by_lang.get(lg) else "Explain energy."

        st60 = edge_eng.init_state(batch_size=1, device=dev)
        r1_60 = edge_eng.generate_reply(u1, state=st60, max_new_bytes=80)
        r2_60 = edge_eng.generate_reply(t2_q, state=r1_60.state, max_new_bytes=110)
        st302_lifted = apply_procrustes_state_lift(r2_60.state, core_eng, bridge_arrays, dev)
        r3_302 = core_eng.generate_reply(q3, state=st302_lifted, max_new_bytes=80)
        hit = pin in r3_302.text
        if hit:
            handoff_hits += 1
        if len(handoff_examples) < 4:
            handoff_examples.append(
                {"lang": lg, "pin": pin, "turn1_edge60m": r1_60.text, "turn2_edge60m": r2_60.text, "turn3_core302m": r3_302.text, "hit": hit}
            )
    handoff_acc = round(handoff_hits / max(1, handoff_tot), 4)

    # Verify C99/SIMD `mmap` forward parity on BOTH `.sfield.scar` containers (`0.000000` STE shock, NO `heal_all_e8_scars`!)
    core_eng.unmount_mos_domain_bank()
    x_mm = torch.randint(32, 122, (2, 32), dtype=torch.long)
    edge_cpu_f = edge_eng.cpu().eval()
    edge_scar_bytes = edge_cpu_f.save_packed_scar_container(edge_final_p)
    mm_e2, ct_e2 = ScarFieldEngine.load_packed_scar_mmap(edge_final_p)
    mm_e2.eval()
    with torch.no_grad():
        edge_mmap_cos = round(
            float(F.cosine_similarity(edge_cpu_f.forward_sequence(x_mm)["scores"].flatten().float(), mm_e2.forward_sequence(x_mm)["scores"].flatten().float(), dim=0).item()),
            6,
        )
    ct_e2.close()

    core_cpu_f = core_eng.cpu().eval()
    core_scar_bytes = core_cpu_f.save_packed_scar_container(core_final_p)
    mm_c2, ct_c2 = ScarFieldEngine.load_packed_scar_mmap(core_final_p)
    mm_c2.eval()
    with torch.no_grad():
        core_mmap_cos = round(
            float(F.cosine_similarity(core_cpu_f.forward_sequence(x_mm)["scores"].flatten().float(), mm_c2.forward_sequence(x_mm)["scores"].flatten().float(), dim=0).item()),
            6,
        )
    ct_c2.close()

    audit_payload = {
        "suite": "scar_field_v9_2_jarvis_marathon_audit",
        "version": "9.2.0",
        "hardware": gpu_name,
        "gpu_total_vram_gib": gpu_total_gib,
        "triton_enabled": bool(HAS_TRITON and dev.type == "cuda"),
        "total_wall_hours": round((time.time() - t_global_start) / 3600.0, 3),
        "open_corpus_summary": {
            "dialogues_pt": len(open_dialogues["pt"]),
            "dialogues_en": len(open_dialogues["en"]),
            "dialogues_es": len(open_dialogues["es"]),
            "dialogues_fr": len(open_dialogues["fr"]),
            "code_tasks": len(code_tasks_v92),
            "math_tasks": len(math_tasks_v92),
            "provenance": wiki_prov + open_prov,
        },
        "edge_60m_anchor": {
            "spherical_params": sum(p.numel() for p in edge_cpu_f.parameters()),
            "sfield_scar_bytes": edge_scar_bytes,
            "sfield_scar_mib": round(edge_scar_bytes / (1024**2), 3),
            "o1_state_bytes": edge_cpu_f.init_state(1).total_state_bytes,
            "completed_steps": edge_steps,
            "conflicts_projected": edge_conflicts,
            "mmap_c_simd_cosine": edge_mmap_cos,
            "evaluation": edge_full_eval,
        },
        "core_302m_anchor": {
            "spherical_params": core_params,
            "sfield_scar_bytes": core_scar_bytes,
            "sfield_scar_mib": round(core_scar_bytes / (1024**2), 3),
            "mosbank_scar_bytes": mosbank_bytes,
            "stiefel_bridge_bytes": bridge_bytes,
            "o1_state_bytes": core_cpu_f.init_state(1).total_state_bytes,
            "step0_isometric_expansion_cosine": step0_isometric_cosine,
            "step0_bpb_inherited_from_60m": core_step0_eval["heldout_bpb"],
            "completed_fleet_steps": core_steps,
            "completed_swarm_rounds": swarm_rounds,
            "eff_step_ms": core_eff_step_ms,
            "peak_vram_gib": core_peak_vram_gib,
            "oom_retries": core_oom_retries,
            "conflicts_projected": core_conflicts,
            "streamed_core_bytes": streamed_core_bytes,
            "mmap_c_simd_cosine": core_mmap_cos,
            "evaluation": core_full_eval,
            "mosbank_fast_evaluation": core_mosbank_eval_fast,
            "passkey_o1_recall": pk_core,
        },
        "live_isometric_session_handoff_60m_to_302m": {
            "exact_pin_recall_after_mid_chat_lift": handoff_acc,
            "hits": f"{handoff_hits}/{handoff_tot}",
            "examples": handoff_examples,
        },
    }

    audit_path = out_dir / "scar_field_v9_2_jarvis_marathon_audit.json"
    _atomic_write_json(audit_path, audit_payload)
    if not dry_run:
        _atomic_write_json(REPO_ROOT / "results" / "scar_field_v9_2_jarvis_marathon_audit.json", audit_payload)

    _atomic_write_json(
        status_path,
        {
            "phase": "DONE",
            "edge_mib": round(edge_scar_bytes / (1024**2), 3),
            "core_mib": round(core_scar_bytes / (1024**2), 3),
            "core_bpb": core_full_eval["heldout_bpb"],
            "core_pin_acc": core_full_eval["multi_turn_pin_acc"],
            "handoff_60m_to_302m_acc": handoff_acc,
            "edge_mmap_cos": edge_mmap_cos,
            "core_mmap_cos": core_mmap_cos,
        },
    )
    (out_dir / "DONE.flag").write_text("OK\n", encoding="utf-8")
    print(
        f"\n[DONE v9.2 JARVIS MARATHON] Edge={edge_scar_bytes/(1024**2):.3f} MiB (cos={edge_mmap_cos}) | "
        f"Core={core_scar_bytes/(1024**2):.3f} MiB (cos={core_mmap_cos}) | "
        f"Core BPB={core_full_eval['heldout_bpb']} | Live 60M->302M Handoff={handoff_acc*100:.1f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
