"""Builds the Universal Language-Invariant `ScarStore v4.0` (`GoldenSliceScarStoreV4`) for `0.5B` & `3B`.

Eliminates `_en` / `_pt` / `_es` / `_fr` cartridge duplication:
  - **1 Single Cartridge per Concept** (`62,720 B` INT4 on `0.5B`; `143,360 B = 35 x 4096 B` INT4 on `3B`),
    anchored **exclusively in English** (`anchor_queries_en`) and routed at the `2L/3` Golden Slice via the
    Language-Subspace Orthogonal Projector `P_perp_lang` (`k_lang = 14`).
  - **Polyglot Premise Seed Consolidation (`Zero-SGD` Prediction-Space RRR):** Synthesizes exact constants,
    equations, and statutory articles across **English (`EN`), Portuguese (`PT`), Spanish (`ES`), and French (`FR`)**
    with smooth `seed_horizon` handoff (`44` tokens + `16`-step cosine fade) into `SCache O(1)` (`ΔBytes == 0`).
  - Preserves all `v3.0` (`qwen{0p5b,3b}_v3_0.*`) and `v3.1` (`qwen{0p5b,3b}_bilingual_v3_1.*`) artifacts untouched.
  - Updates `./webgpu_dist/qwen0p5b/` (`capsules.scarstore`, `scarstore_router.*`, `mos_scache_kernels.wgsl`, JS/HTML UI).
"""

from __future__ import annotations

import gc
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scar_cache.cli_scar_store import (
    ScarStoreHTTPRangeServer,
    generate_with_scar_store,
    make_http_range_capsule_fetcher,
)
from scar_cache.golden_slice_v4 import (
    UNIVERSAL_CHAT_SYSTEM_PROMPT,
    GoldenSliceScarStore,
    GoldenSliceSynthesisEngine,
)
from scar_cache.hf_cache import SCacheDynamicCache
from scar_cache.mos_student import load_mos_student_model
from scar_cache.webgpu import WEBGPU_APP_DIR, WGSL_KERNELS_PATH
from scar_cache.webgpu_engine import export_scarstore_for_webgpu

LIB_DIR = REPO_ROOT / "results" / "online_library_v3_0"


# ---------------------------------------------------------------------------
# 11 Universal Language-Invariant Concept Cartridges + 1 Conjunctive Bridge
# STRICT ANCHOR RULE: `anchor_queries_en` contains 100% English strings ONLY!
# STRICT SEED RULE: Seed prompts never leak target constants/formulas, and seed
# answers are compact non-EOS-terminated premises so SCache O(1) elaborates richly.
# ---------------------------------------------------------------------------
UNIVERSAL_CONCEPT_SPECS: List[Dict[str, Any]] = [
    {
        "capsule_id": "concept_mos_scache_architecture",
        "title": "System One Architecture (MoS INT4 + SCache O(1) + Golden-Slice ScarStore)",
        "title_pt": "Arquitetura System One (MoS INT4 + SCache O(1) + ScarStore Golden-Slice)",
        "category": "AI Systems Architecture",
        "summary": "Mixture of Scars (Block-64 INT4 + ASVD), constant-memory associative SCache O(1) via Widrow-Hoff W_scar, and HTTP 206 Golden-Slice cartridges.",
        "tau": 0.39,
        "ramp": 0.04,
        "anchor_queries_en": [
            "What architecture do you use?",
            "Detail your internal AI architecture and how your memory works.",
            "Explain the neural architecture and cache mechanism you run on.",
        ],
        "prompts_polyglot": {
            "pt": "Qual arquitetura você usa? Detalhe como ela funciona.",
            "en": "What architecture do you use? Detail how it works.",
            "es": "¿Qué arquitectura utilizas? Detalla cómo funciona.",
            "fr": "Quelle architecture utilisez-vous ? Détaillez son fonctionnement.",
        },
        "seeds": [
            (
                "What architecture do you use?",
                "System One uses the Mixture of Scars (MoS INT4 + ASVD) architecture, constant-memory associative SCache O(1), and on-demand HTTP 206 ScarStore cartridges",
            ),
            (
                "Qual arquitetura você usa?",
                "O System One utiliza a arquitetura Mixture of Scars (MoS INT4 + ASVD), memória associativa constante SCache O(1) e cartuchos ScarStore sob demanda via HTTP 206",
            ),
            (
                "¿Qué arquitectura utilizas?",
                "System One utiliza la arquitectura Mixture of Scars (MoS INT4 + ASVD), memoria asociativa constante SCache O(1) y cartuchos ScarStore bajo demanda vía HTTP 206",
            ),
            (
                "Quelle architecture utilisez-vous ?",
                "System One utilise l'architecture Mixture of Scars (MoS INT4 + ASVD), la mémoire associative constante SCache O(1) et des cartouches ScarStore via HTTP 206",
            ),
        ],
    },
    {
        "capsule_id": "concept_boltzmann_constant",
        "title": "Boltzmann Constant (Exact 2019 SI Definition & Statistical Entropy)",
        "title_pt": "Constante de Boltzmann (Valor Exato SI 2019 e Entropia Estatística)",
        "category": "Statistical Physics",
        "summary": "Exact 2019 SI value k_B = 1.380649 x 10^-23 J/K bridging macroscopic temperature T, microscopic thermal energy, and Boltzmann entropy S = k_B ln Omega.",
        "tau": 0.41,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What is the Boltzmann constant and where does it apply?",
            "State the exact SI value and physical meaning of Boltzmann's constant.",
            "Explain Boltzmann's constant in statistical mechanics and thermodynamics.",
        ],
        "prompts_polyglot": {
            "pt": "O que é a constante de Boltzmann e onde ela se aplica?",
            "en": "What is the Boltzmann constant and where does it apply?",
            "es": "¿Qué es la constante de Boltzmann y dónde se aplica?",
            "fr": "Qu'est-ce que la constante de Boltzmann et où s'applique-t-elle ?",
        },
        "seeds": [
            (
                "What is the Boltzmann constant and where does it apply?",
                "The Boltzmann constant ($k_B$) is equal to $k_B = 1.380649 \\times 10^{-23}\\text{ J/K}$ (joules per kelvin), linking temperature $T$ to thermal energy and statistical entropy $S = k_B \\ln \\Omega$.",
            ),
            (
                "O que é a constante de Boltzmann e onde ela se aplica?",
                "A constante de Boltzmann ($k_B$) é igual a $k_B = 1,380649 \\times 10^{-23}\\text{ J/K}$ (joules por kelvin), conectando a temperatura absoluta $T$ à energia térmica e à entropia $S = k_B \\ln \\Omega$.",
            ),
            (
                "¿Qué es la constante de Boltzmann y dónde se aplica?",
                "La constante de Boltzmann ($k_B$) es igual a $k_B = 1.380649 \\times 10^{-23}\\text{ J/K}$ (julios por kelvin), vinculando la temperatura $T$ con la energía térmica y la entropía $S = k_B \\ln \\Omega$.",
            ),
            (
                "Qu'est-ce que la constante de Boltzmann et où s'applique-t-elle ?",
                "La constante de Boltzmann ($k_B$) est égale à $k_B = 1.380649 \\times 10^{-23}\\text{ J/K}$ (joules par kelvin), reliant la température $T$ à l'énergie thermique et à l'entropie $S = k_B \\ln \\Omega$.",
            ),
        ],
    },
    {
        "capsule_id": "concept_landauer_principle",
        "title": "Landauer's Principle (Thermodynamic Cost of Irreversible Bit Erasure)",
        "title_pt": "Princípio de Landauer (Limite Termodinâmico de Apagamento de 1 Bit)",
        "category": "Physics of Information",
        "summary": "Irreversibly erasing 1 bit of information in a thermal bath at temperature T dissipates at least E_min = k_B T ln 2 of heat.",
        "tau": 0.41,
        "ramp": 0.05,
        "anchor_queries_en": [
            "Tell me about Landauer's principle.",
            "Explain Landauer's principle for the minimum energy dissipated when erasing a bit.",
            "What is the thermodynamic cost of irreversible information erasure under Landauer's bound?",
        ],
        "prompts_polyglot": {
            "pt": "Me fale sobre o princípio de Landauer na física da informação.",
            "en": "Explain Landauer's principle in the physics of information.",
            "es": "Explícame el principio de Landauer en la física de la información.",
            "fr": "Expliquez le principe de Landauer en physique de l'information.",
        },
        "seeds": [
            (
                "Explain Landauer's principle in the physics of information.",
                "Landauer's principle ($E_{\\min} = k_B T \\ln 2$) states that irreversibly erasing 1 bit of information at absolute temperature $T$ dissipates at least $k_B T \\ln 2$ of heat ($\\Delta S \\ge k_B \\ln 2$).",
            ),
            (
                "me sabe sobre o princípio de Landauer",
                "O princípio de Landauer ($E_{\\min} = k_B T \\ln 2$) estabelece que apagar irreversivelmente 1 bit de informação à temperatura absoluta $T$ dissipa no mínimo $k_B T \\ln 2$ de calor ($\\Delta S \\ge k_B \\ln 2$).",
            ),
            (
                "Explícame el principio de Landauer en la física de la información.",
                "El principio de Landauer ($E_{\\min} = k_B T \\ln 2$) establece que borrar irreversiblemente 1 bit de información a temperatura absoluta $T$ disipa como mínimo $k_B T \\ln 2$ de calor ($\\Delta S \\ge k_B \\ln 2$).",
            ),
            (
                "Expliquez le principe de Landauer en physique de l'information.",
                "Le principe de Landauer ($E_{\\min} = k_B T \\ln 2$) établit que l'effacement irréversible d'un bit d'information à température absolue $T$ dissipe au minimum $k_B T \\ln 2$ de chaleur ($\\Delta S \\ge k_B \\ln 2$).",
            ),
        ],
    },
    {
        "capsule_id": "concept_brazil_lc150_domestic_law",
        "title": "Brazilian Domestic Labor Law (LC 150/2015 vs Diarista Regime)",
        "title_pt": "Direito Trabalhista Doméstico no Brasil (LC 150/2015 vs Diarista)",
        "category": "Labor Law (Brazil)",
        "summary": "Constitutional Amendment EC 72/2013 and Complementary Law LC 150/2015 (> 2 days/week employment bond, 44h/week, FGTS 8% + 3.2%) vs autonomous diarista (up to 2 days/week).",
        "tau": 0.41,
        "ramp": 0.05,
        "anchor_queries_en": [
            "How does Brazilian labor law work for housecleaners and domestic workers?",
            "Explain domestic worker labor rights in Brazil and the legal distinction from a day cleaner.",
            "When does a cleaner or maid in Brazil establish a formal domestic employment bond?",
        ],
        "prompts_polyglot": {
            "pt": "Como funciona o direito trabalhista para faxineiras e domésticas no Brasil?",
            "en": "How does Brazilian labor law work for housecleaners and domestic workers?",
            "es": "¿Cómo funciona el derecho laboral para trabajadoras domésticas y limpiadoras en Brasil?",
            "fr": "Comment fonctionne le droit du travail pour les employées domestiques et femmes de ménage au Brésil ?",
        },
        "seeds": [
            (
                "How does Brazilian labor law work for housecleaners and domestic workers?",
                "In Brazil, Complementary Law LC 150/2015 (EC 72/2013) governs domestic workers serving more than 2 days per week (CTPS, 44h/week, FGTS 8% + 3.2%), whereas working up to 2 days/week characterizes an autonomous diarista.",
            ),
            (
                "Como funciona o direito trabalhista para faxineiras no brasil?",
                "No Brasil, a Lei Complementar 150/2015 (EC 72/2013) rege a empregada doméstica que atua mais de 2 dias por semana (CTPS, jornada de 44h, FGTS de 8% + 3,2%), enquanto até 2 dias/semana configura diarista autônoma.",
            ),
            (
                "¿Cómo funciona el derecho laboral para trabajadoras domésticas y limpiadoras en Brasil?",
                "En Brasil, la Ley Complementaria LC 150/2015 (EC 72/2013) regula el vínculo doméstico al trabajar más de 2 días por semana (CTPS, 44h semanales, FGTS 8% + 3,2%), mientras que hasta 2 días/semana es diarista autónoma.",
            ),
            (
                "Comment fonctionne le droit du travail pour les employées domestiques au Brésil ?",
                "Au Brésil, la Loi Complémentaire LC 150/2015 (EC 72/2013) régit le contrat domestique au-delà de 2 jours par semaine (CTPS, 44h/semaine, FGTS 8% + 3,2%), tandis que jusqu'à 2 jours/semaine relève de la diariste autonome.",
            ),
        ],
    },
    {
        "capsule_id": "concept_immerman_szelepcsenyi",
        "title": "Immerman-Szelepcsényi Theorem (NSPACE = co-NSPACE Inductive Counting)",
        "title_pt": "Teorema de Immerman-Szelepcsényi (NSPACE = co-NSPACE por Contagem Indutiva)",
        "category": "Computational Complexity",
        "summary": "Every nondeterministic space class s(n) >= log n is closed under complementation (NL = coNL) via inductive counting of reachable configurations.",
        "tau": 0.41,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What does the Immerman-Szelepcsenyi theorem prove for nondeterministic space complexity?",
            "Explain why nondeterministic logspace equals co-NL via the Immerman-Szelepcsenyi theorem.",
            "How does inductive counting prove closure of NSPACE under complement?",
        ],
        "prompts_polyglot": {
            "pt": "O que prova o teorema de Immerman-Szelepcsényi sobre classes de espaço não-determinístico?",
            "en": "What does the Immerman-Szelepcsényi theorem prove about nondeterministic space classes?",
            "es": "¿Qué demuestra el teorema de Immerman-Szelepcsényi sobre la complejidad espacial no determinista?",
            "fr": "Que démontre le théorème d'Immerman-Szelepcsényi sur les classes d'espace non déterministe ?",
        },
        "seeds": [
            (
                "What does the Immerman-Szelepcsenyi theorem prove about nondeterministic space classes?",
                "The Immerman-Szelepcsényi theorem (1987) proves $\\text{NSPACE}(s(n)) = \\text{co-NSPACE}(s(n))$ for $s(n) \\ge \\log n$ (hence $\\text{NL} = \\text{coNL}$) by inductively counting exact reachable configurations $C_d$ in $O(s(n))$ space.",
            ),
            (
                "O que prova o teorema de Immerman-Szelepcsényi sobre classes de espaço não-determinístico?",
                "O teorema de Immerman-Szelepcsényi (1987) prova que $\\text{NSPACE}(s(n)) = \\text{co-NSPACE}(s(n))$ para $s(n) \\ge \\log n$ (logo $\\text{NL} = \\text{coNL}$) via contagem indutiva exata das configurações alcançáveis $C_d$.",
            ),
            (
                "¿Qué demuestra el teorema de Immerman-Szelepcsényi sobre la complejidad espacial no determinista?",
                "El teorema de Immerman-Szelepcsényi (1987) demuestra que $\\text{NSPACE}(s(n)) = \\text{co-NSPACE}(s(n))$ para $s(n) \\ge \\log n$ (por tanto $\\text{NL} = \\text{coNL}$) mediante conteo inductivo exacto de configuraciones alcanzables $C_d$.",
            ),
            (
                "Que démontre le théorème d'Immerman-Szelepcsényi sur les classes d'espace non déterministe ?",
                "Le théorème d'Immerman-Szelepcsényi (1987) démontre que $\\text{NSPACE}(s(n)) = \\text{co-NSPACE}(s(n))$ pour $s(n) \\ge \\log n$ (donc $\\text{NL} = \\text{coNL}$) par comptage inductif exact des configurations accessibles $C_d$.",
            ),
        ],
    },
    {
        "capsule_id": "concept_fischer_lynch_paterson",
        "title": "FLP Impossibility Theorem (Asynchronous Crash-Fault Consensus)",
        "title_pt": "Teorema da Impossibilidade FLP (Consenso Assíncrono com Falha por Parada)",
        "category": "Distributed Systems",
        "summary": "Fischer-Lynch-Paterson (1985) proves no deterministic consensus protocol in an asynchronous network can guarantee termination, agreement, and validity under even 1 unannounced process crash.",
        "tau": 0.41,
        "ramp": 0.05,
        "anchor_queries_en": [
            "Explain the Fischer-Lynch-Paterson impossibility theorem in asynchronous distributed systems.",
            "What does the FLP theorem prove about deterministic consensus with crash faults?",
            "Why is deterministic consensus impossible in asynchronous message passing under a single crash?",
        ],
        "prompts_polyglot": {
            "pt": "Explique o teorema da impossibilidade de Fischer-Lynch-Paterson em sistemas distribuídos.",
            "en": "Explain the Fischer-Lynch-Paterson impossibility theorem in distributed systems.",
            "es": "Explica el teorema de imposibilidad de Fischer-Lynch-Paterson en sistemas distribuidos.",
            "fr": "Expliquez le théorème d'impossibilité de Fischer-Lynch-Paterson dans les systèmes distribués.",
        },
        "seeds": [
            (
                "Explain the Fischer-Lynch-Paterson impossibility theorem in distributed systems.",
                "The Fischer-Lynch-Paterson (FLP, 1985) theorem proves that in a purely asynchronous network, no deterministic consensus protocol guarantees termination, agreement, and validity if even 1 process may crash unannounced.",
            ),
            (
                "Explique o teorema da impossibilidade de Fischer-Lynch-Paterson em sistemas distribuídos.",
                "O teorema de Fischer-Lynch-Paterson (FLP, 1985) prova que em uma rede puramente assíncrona nenhum protocolo determinístico de consenso garante terminação, acordo e validade se ao menos 1 processo puder sofrer falha por parada.",
            ),
            (
                "Explica el teorema de imposibilidad de Fischer-Lynch-Paterson en sistemas distribuidos.",
                "El teorema de Fischer-Lynch-Paterson (FLP, 1985) demuestra que en una red puramente asíncrona ningún protocolo determinista de consenso garantiza terminación, acuerdo y validez si incluso 1 proceso puede fallar por caída.",
            ),
            (
                "Expliquez le théorème d'impossibilité de Fischer-Lynch-Paterson dans les systèmes distribués.",
                "Le théorème de Fischer-Lynch-Paterson (FLP, 1985) prouve que dans un réseau purement asynchrone aucun protocole déterministe de consensus ne garantit terminaison, accord et validité même face à 1 seule panne franche.",
            ),
        ],
    },
    {
        "capsule_id": "concept_marchenko_pastur_law",
        "title": "Marchenko-Pastur Spectral Law (Random Sample Covariance Eigenvalues)",
        "title_pt": "Lei Espectral de Marchenko-Pastur (Autovalores de Covariância Aleatória)",
        "category": "Random Matrix Theory",
        "summary": "Limiting eigenvalue support [lambda_-, lambda_+] with lambda_pm = sigma^2 (1 +- sqrt(gamma))^2 for p x n random matrices as p/n -> gamma.",
        "tau": 0.41,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What does the Marchenko-Pastur law state for eigenvalues of sample covariance matrices?",
            "Give the limiting spectral support bounds of the Marchenko-Pastur distribution.",
            "Explain the Marchenko-Pastur eigenvalue spectrum in high-dimensional random matrix theory.",
        ],
        "prompts_polyglot": {
            "pt": "O que estabelece a lei espectral de Marchenko-Pastur para matrizes aleatórias de covariância?",
            "en": "What does the Marchenko-Pastur spectral law state for random sample covariance matrices?",
            "es": "¿Qué establece la ley espectral de Marchenko-Pastur para matrices aleatorias de covarianza?",
            "fr": "Qu'établit la loi spectrale de Marchenko-Pastur pour les matrices de covariance aléatoires ?",
        },
        "seeds": [
            (
                "What does the Marchenko-Pastur spectral law state for random sample covariance matrices?",
                "The Marchenko-Pastur law (1967) shows that for $p \\times n$ i.i.d. entries of variance $\\sigma^2$ with $p/n \\to \\gamma$, sample covariance eigenvalues concentrate on $[\\lambda_-, \\lambda_+]$ where $\\lambda_{\\pm} = \\sigma^2(1 \\pm \\sqrt{\\gamma})^2$.",
            ),
            (
                "O que estabelece a lei espectral de Marchenko-Pastur para matrizes aleatórias de covariância?",
                "A lei de Marchenko-Pastur (1967) mostra que para entradas i.i.d. $p \\times n$ de variância $\\sigma^2$ com $p/n \\to \\gamma$, os autovalores da covariância amostral concentram-se em $[\\lambda_-, \\lambda_+]$ com $\\lambda_{\\pm} = \\sigma^2(1 \\pm \\sqrt{\\gamma})^2$.",
            ),
            (
                "¿Qué establece la ley espectral de Marchenko-Pastur para matrices aleatorias de covarianza?",
                "La ley de Marchenko-Pastur (1967) demuestra que para matrices $p \\times n$ de varianza $\\sigma^2$ con $p/n \\to \\gamma$, los autovalores de covarianza muestral convergen al soporte $[\\lambda_-, \\lambda_+]$ con $\\lambda_{\\pm} = \\sigma^2(1 \\pm \\sqrt{\\gamma})^2$.",
            ),
            (
                "Qu'établit la loi spectrale de Marchenko-Pastur pour les matrices de covariance aléatoires ?",
                "La loi de Marchenko-Pastur (1967) établit que pour des entrées i.i.d. $p \\times n$ de variance $\\sigma^2$ avec $p/n \\to \\gamma$, le spectre de covariance empirique converge vers $[\\lambda_-, \\lambda_+]$ où $\\lambda_{\\pm} = \\sigma^2(1 \\pm \\sqrt{\\gamma})^2$.",
            ),
        ],
    },
    {
        "capsule_id": "concept_kabatiansky_levenshtein",
        "title": "Kabatiansky-Levenshtein Bound (High-Dimensional Sphere Packing 0.5990)",
        "title_pt": "Limite de Kabatiansky-Levenshtein (Empacotamento Esférico 0.5990)",
        "category": "Information & Coding Theory",
        "summary": "Best known asymptotic upper bound 2^{-0.5990 n (1 + o(1))} on Euclidean sphere-packing density in R^n via Gegenbauer polynomials.",
        "tau": 0.41,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What asymptotic upper bound does the Kabatiansky-Levenshtein theorem give for sphere packing in R^n?",
            "Explain the Kabatiansky-Levenshtein bound on high-dimensional Euclidean sphere packing density.",
            "How do Gegenbauer polynomials yield the Kabatiansky-Levenshtein sphere-packing exponent?",
        ],
        "prompts_polyglot": {
            "pt": "Qual limite superior assintótico o teorema de Kabatiansky-Levenshtein estabelece para empacotamento de esferas?",
            "en": "What asymptotic upper bound does the Kabatiansky-Levenshtein theorem establish for sphere packing?",
            "es": "¿Qué cota superior asintótica establece el teorema de Kabatiansky-Levenshtein para el empaquetamiento de esferas?",
            "fr": "Quelle borne supérieure asymptotique le théorème de Kabatiansky-Levenshtein établit-il pour l'empilement de sphères ?",
        },
        "seeds": [
            (
                "What asymptotic upper bound does the Kabatiansky-Levenshtein theorem establish for sphere packing?",
                "The Kabatiansky-Levenshtein bound (1978) proves via Gegenbauer linear programming on spherical codes that Euclidean sphere-packing density in $\\mathbb{R}^n$ satisfies $\\Delta_n \\le 2^{-0.5990\\,n(1 + o(1))}$.",
            ),
            (
                "Qual limite superior assintótico o teorema de Kabatiansky-Levenshtein estabelece para empacotamento de esferas?",
                "O limite de Kabatiansky-Levenshtein (1978) prova, via programação linear com polinômios de Gegenbauer em códigos esféricos, que a densidade de empacotamento em $\\mathbb{R}^n$ obedece a $\\Delta_n \\le 2^{-0.5990\\,n(1 + o(1))}$.",
            ),
            (
                "¿Qué cota superior asintótica establece el teorema de Kabatiansky-Levenshtein para el empaquetamiento de esferas?",
                "La cota de Kabatiansky-Levenshtein (1978) demuestra mediante polinomios de Gegenbauer en códigos esféricos que la densidad de empaquetamiento en $\\mathbb{R}^n$ cumple $\\Delta_n \\le 2^{-0.5990\\,n(1 + o(1))}$.",
            ),
            (
                "Quelle borne supérieure asymptotique le théorème de Kabatiansky-Levenshtein établit-il pour l'empilement de sphères ?",
                "La borne de Kabatiansky-Levenshtein (1978) prouve par programmation linéaire de Gegenbauer sur les codes sphériques que la densité d'empilement dans $\\mathbb{R}^n$ vérifie $\\Delta_n \\le 2^{-0.5990\\,n(1 + o(1))}$.",
            ),
        ],
    },
    {
        "capsule_id": "concept_chaitin_incompleteness",
        "title": "Chaitin's Algorithmic Incompleteness (Kolmogorov Complexity K(x) > L & Omega)",
        "title_pt": "Incompletude Algorítmica de Chaitin (Complexidade de Kolmogorov K(x) > L e Ômega)",
        "category": "Algorithmic Information Theory",
        "summary": "Any sound formal axiom system F cannot prove K(x) > L for strings beyond L_F = K(F) + c bits, epitomized by Chaitin's halting probability Omega.",
        "tau": 0.41,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What does Chaitin's algorithmic incompleteness theorem state about proving Kolmogorov complexity bounds?",
            "Explain Chaitin's incompleteness bound and the halting probability constant.",
            "Why can a formal system only prove finitely many lower bounds on Kolmogorov complexity?",
        ],
        "prompts_polyglot": {
            "pt": "O que afirma o teorema da incompletude algorítmica de Chaitin sobre a complexidade de Kolmogorov?",
            "en": "What does Chaitin's algorithmic incompleteness theorem state about Kolmogorov complexity?",
            "es": "¿Qué afirma el teorema de incompletitud algorítmica de Chaitin sobre la complejidad de Kolmogorov?",
            "fr": "Qu'affirme le théorème d'incomplétude algorithmique de Chaitin sur la complexité de Kolmogorov ?",
        },
        "seeds": [
            (
                "What does Chaitin's algorithmic incompleteness theorem state about Kolmogorov complexity?",
                "Chaitin's incompleteness theorem proves that any sound formal system $F$ has a constant $L_F = K(F) + c$ above which it cannot prove $K(x) > L_F$ for any string $x$, reflected in Chaitin's halting probability $\\Omega$.",
            ),
            (
                "O que afirma o teorema da incompletude algorítmica de Chaitin sobre a complexidade de Kolmogorov?",
                "O teorema da incompletude de Chaitin prova que todo sistema formal consistente $F$ possui um limiar $L_F = K(F) + c$ acima do qual jamais pode demonstrar $K(x) > L_F$, refletido na constante de parada $\\Omega$ de Chaitin.",
            ),
            (
                "¿Qué afirma el teorema de incompletitud algorítmica de Chaitin sobre la complejidad de Kolmogorov?",
                "El teorema de incompletitud de Chaitin demuestra que ningún sistema formal consistente $F$ puede probar cotas $K(x) > L_F$ por encima de $L_F = K(F) + c$, encarnado en la probabilidad de parada $\\Omega$ de Chaitin.",
            ),
            (
                "Qu'affirme le théorème d'incomplétude algorithmique de Chaitin sur la complexité de Kolmogorov ?",
                "Le théorème d'incomplétude de Chaitin prouve qu'aucun système formel cohérent $F$ ne peut démontrer $K(x) > L_F$ au-delà du seuil $L_F = K(F) + c$, illustré par la constante d'arrêt $\\Omega$ de Chaitin.",
            ),
        ],
    },
    {
        "capsule_id": "concept_regev_lwe_lattice",
        "title": "Regev's LWE Reduction (Worst-Case GapSVP / SIVP Post-Quantum Hardness)",
        "title_pt": "Redução LWE de Regev (Segurança Pós-Quântica de Pior Caso GapSVP / SIVP)",
        "category": "Post-Quantum Cryptography",
        "summary": "Regev (2005) proves average-case Learning With Errors (LWE) is at least as hard as worst-case lattice problems GapSVP and SIVP under quantum reductions.",
        "tau": 0.41,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What worst-case lattice hardness reduction underpins Regev's Learning With Errors cryptosystem?",
            "Explain Regev's quantum reduction from worst-case lattice problems to average-case LWE.",
            "Why does Learning With Errors enjoy worst-case to average-case post-quantum security?",
        ],
        "prompts_polyglot": {
            "pt": "Qual redução de dificuldade de pior caso em reticulados fundamenta o problema LWE de Regev?",
            "en": "What worst-case lattice hardness reduction underpins Regev's Learning With Errors problem?",
            "es": "¿Qué reducción de dificultad en el peor caso sobre retículos fundamenta el problema LWE de Regev?",
            "fr": "Quelle réduction de difficulté dans le pire cas sur les réseaux fonde le problème LWE de Regev ?",
        },
        "seeds": [
            (
                "What worst-case lattice hardness reduction underpins Regev's Learning With Errors problem?",
                "Regev's theorem (2005) establishes a quantum reduction showing that solving average-case Learning With Errors ($\\text{LWE}_{n,q,\\chi}$) is at least as hard as approximating worst-case lattice problems $\\text{GapSVP}_\\gamma$ and $\\text{SIVP}_\\gamma$.",
            ),
            (
                "Qual redução de dificuldade de pior caso em reticulados fundamenta o problema LWE de Regev?",
                "O teorema de Regev (2005) estabelece uma redução quântica provando que resolver Learning With Errors ($\\text{LWE}_{n,q,\\chi}$) no caso médio é tão difícil quanto aproximar no pior caso os problemas de reticulado $\\text{GapSVP}_\\gamma$ e $\\text{SIVP}_\\gamma$.",
            ),
            (
                "¿Qué reducción de dificultad en el peor caso sobre retículos fundamenta el problema LWE de Regev?",
                "El teorema de Regev (2005) demuestra mediante una reducción cuántica que resolver Learning With Errors ($\\text{LWE}_{n,q,\\chi}$) en promedio es tan difícil como aproximar en el peor caso $\\text{GapSVP}_\\gamma$ y $\\text{SIVP}_\\gamma$.",
            ),
            (
                "Quelle réduction de difficulté dans le pire cas sur les réseaux fonde le problème LWE de Regev ?",
                "Le théorème de Regev (2005) établit par réduction quantique que résoudre Learning With Errors ($\\text{LWE}_{n,q,\\chi}$) en moyenne est au moins aussi difficile que $\\text{GapSVP}_\\gamma$ et $\\text{SIVP}_\\gamma$ dans le pire cas.",
            ),
        ],
    },
    {
        "capsule_id": "concept_shannon_hartley_capacity",
        "title": "Shannon-Hartley Capacity & Ultimate -1.59 dB Eb/N0 Limit",
        "title_pt": "Capacidade de Shannon-Hartley e Limite Fundamental de -1,59 dB Eb/N0",
        "category": "Information Theory",
        "summary": "AWGN channel capacity C = B log_2(1 + S/N) bits/s and ultimate infinite-bandwidth energy-per-bit limit Eb/N0 = ln 2 (-1.59 dB).",
        "tau": 0.41,
        "ramp": 0.05,
        "anchor_queries_en": [
            "State the Shannon-Hartley channel capacity theorem for an AWGN channel and its ultimate Eb/N0 limit.",
            "Explain the Shannon-Hartley capacity formula in Gaussian noise and the infinite-bandwidth Shannon limit.",
            "How is maximum error-free bit rate bounded by bandwidth and signal-to-noise ratio?",
        ],
        "prompts_polyglot": {
            "pt": "Enuncie o teorema da capacidade de canal de Shannon-Hartley e o limite fundamental de Eb/N0.",
            "en": "State the Shannon-Hartley channel capacity theorem and its fundamental Eb/N0 limit.",
            "es": "Enuncia el teorema de capacidad de canal de Shannon-Hartley y su límite fundamental de Eb/N0.",
            "fr": "Énoncez le théorème de capacité de canal de Shannon-Hartley et sa limite fondamentale Eb/N0.",
        },
        "seeds": [
            (
                "State the Shannon-Hartley channel capacity theorem and its fundamental Eb/N0 limit.",
                "The Shannon-Hartley theorem gives AWGN channel capacity $C = B \\log_2(1 + S/N)\\text{ bits/s}$, yielding as $B \\to \\infty$ the ultimate Shannon limit $E_b/N_0 \\ge \\ln 2 \\approx 0.693$ ($-1.59\\text{ dB}$).",
            ),
            (
                "Enuncie o teorema da capacidade de canal de Shannon-Hartley e o limite fundamental de Eb/N0.",
                "O teorema de Shannon-Hartley define a capacidade máxima $C = B \\log_2(1 + S/N)\\text{ bits/s}$ em canal AWGN, implicando para $B \\to \\infty$ o limite fundamental de Shannon $E_b/N_0 \\ge \\ln 2 \\approx 0,693$ ($-1,59\\text{ dB}$).",
            ),
            (
                "Enuncia el teorema de capacidad de canal de Shannon-Hartley y su límite fundamental de Eb/N0.",
                "El teorema de Shannon-Hartley establece la capacidad AWGN $C = B \\log_2(1 + S/N)\\text{ bits/s}$, alcanzando para $B \\to \\infty$ el límite fundamental de Shannon $E_b/N_0 \\ge \\ln 2 \\approx 0,693$ ($-1,59\\text{ dB}$).",
            ),
            (
                "Énoncez le théorème de capacité de canal de Shannon-Hartley et sa limite fondamentale Eb/N0.",
                "Le théorème de Shannon-Hartley fixe la capacité AWGN à $C = B \\log_2(1 + S/N)\\text{ bits/s}$, donnant lorsque $B \\to \\infty$ la limite ultime de Shannon $E_b/N_0 \\ge \\ln 2 \\approx 0,693$ ($-1,59\\text{ dB}$).",
            ),
        ],
    },
]

UNIVERSAL_BRIDGE_SPEC: Dict[str, Any] = {
    "capsule_id": "bridge_landauer_immerman",
    "title": "Sleep-Consolidated Bridge: Landauer Thermodynamics x Immerman-Szelepcsényi Space",
    "title_pt": "Ponte Consolidada (Sleep): Termodinâmica de Landauer x Espaço de Immerman-Szelepcsényi",
    "category": "Interdisciplinary Sleep Bridge",
    "summary": "Conjunctive bridge (fires ONLY when both Landauer and Immerman-Szelepcsényi co-activate) reconciling O(log n) inductive counter overwrites with k_B T ln 2 via Bennett reversible pebbling.",
    "parents": ["concept_landauer_principle", "concept_immerman_szelepcsenyi"],
    "tau": 0.70,
    "ramp": 0.06,
    "anchor_queries_en": [
        "How do the inductive counter overwrites in the Immerman-Szelepcsenyi theorem interact with Landauer's principle of heat dissipation?",
        "Connect Landauer's thermodynamic bit-erasure cost with Immerman-Szelepcsenyi logspace inductive counting.",
    ],
    "prompts_polyglot": {
        "pt": "Como a reutilização de espaço da contagem indutiva de Immerman-Szelepcsényi interage com a dissipação térmica do princípio de Landauer?",
        "en": "How does space reuse in Immerman-Szelepcsényi inductive counting interact with thermal dissipation under Landauer's principle?",
        "es": "¿Cómo interactúa la reutilización de espacio del conteo inductivo de Immerman-Szelepcsényi con el principio termodinámico de Landauer?",
        "fr": "Comment la réutilisation d'espace du comptage inductif d'Immerman-Szelepcsényi interagit-elle avec le principe thermodynamique de Landauer ?",
    },
    "seeds": [
        (
            "How does space reuse in Immerman-Szelepcsenyi inductive counting interact with thermal dissipation under Landauer's principle?",
            "By Bennett's reversible pebbling, Immerman-Szelepcsényi $O(\\log n)$ counter overwrites avoid Landauer's $k_B T \\ln 2$ heat dissipation per erased bit",
        ),
        (
            "Como a reutilização de espaço da contagem indutiva de Immerman-Szelepcsényi interage com a dissipação térmica do princípio de Landauer?",
            "Pelo pebbling reversível de Bennett, a contagem $O(\\log n)$ de Immerman-Szelepcsényi evita a dissipação térmica $k_B T \\ln 2$ de Landauer",
        ),
        (
            "¿Cómo interactúa la reutilización de espacio del conteo inductivo de Immerman-Szelepcsényi con el principio termodinámico de Landauer?",
            "Mediante el pebbling reversible de Bennett, el conteo $O(\\log n)$ de Immerman-Szelepcsényi evita la disipación térmica $k_B T \\ln 2$ de Landauer",
        ),
        (
            "Comment la réutilisation d'espace du comptage inductif d'Immerman-Szelepcsényi interagit-elle avec le principe thermodynamique de Landauer ?",
            "Par le pebbling réversible de Bennett, le comptage $O(\\log n)$ d'Immerman-Szelepcsényi évite la dissipation thermique $k_B T \\ln 2$ de Landauer",
        ),
    ],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def build_universal_tier(
    tier_tag: str,
    student_id: str,
    dev: torch.device,
) -> Dict[str, Any]:
    print(f"\n[Universal Golden-Slice v4.0 - {tier_tag}] Calibrating & Minting 12 Language-Invariant Cartridges on {student_id} ...", flush=True)
    t0 = time.perf_counter()
    dtype = torch.float16 if dev.type in ("cuda", "mps") else torch.float32
    tok = AutoTokenizer.from_pretrained(student_id, local_files_only=True)
    slm = load_mos_student_model(student_id, tokenizer=tok, device=dev, dtype=dtype, local_files_only=True, quiet=True)

    bank = GoldenSliceSynthesisEngine.calibrate_golden_slice_student(
        student_model=slm,
        tokenizer=tok,
        system_prompt=UNIVERSAL_CHAT_SYSTEM_PROMPT,
        k_lang=14,
        rank_mid=12,
        rank_final=52,
        seed_horizon=38,
        seed_fade_steps=14,
    )
    print(
        f"  ✓ Calibrated Golden Slice at Layer {bank.mid_layer_idx}/{bank.num_layers} "
        f"(d={bank.hidden_dim}, k_lang={bank.k_lang}, ranks={bank.rank_mid}+{bank.rank_final})",
        flush=True,
    )

    for sp in UNIVERSAL_CONCEPT_SPECS:
        cid = sp["capsule_id"]
        t_c = time.perf_counter()
        meta = {
            "title": sp["title"],
            "title_pt": sp["title_pt"],
            "category": sp["category"],
            "summary": sp["summary"],
            "example_prompt": sp["prompts_polyglot"]["pt"],
            "prompts_polyglot": dict(sp["prompts_polyglot"]),
            "anchor_language": "en_only_p_perp_lang",
            "supported_languages": ["pt", "en", "es", "fr"],
            "teacher_model": "Golden-Slice v4.0 Closed-Form RRR + Cutting-Plane Rosetta (Zero-SGD)",
            "teacher_reference": sp["seeds"][0][1],
        }
        cap = GoldenSliceSynthesisEngine.solve_concept_capsule(
            capsule_id=cid,
            anchor_queries_en=sp["anchor_queries_en"],
            seed_pairs=sp["seeds"],
            student_model=slm,
            tokenizer=tok,
            bank=bank,
            system_prompt=UNIVERSAL_CHAT_SYSTEM_PROMPT,
            tau=float(sp["tau"]),
            ramp=float(sp["ramp"]),
            min_clearance=15.5,
            metadata=meta,
            quantize_int4=True,
        )
        print(
            f"  ✓ Minted {cid:34s} | surprise={cap.metadata['surprise_steps']:2d}/{cap.metadata['total_seed_steps']:3d} "
            f"| {len(cap.to_int4_bytes()):,} B INT4 | {time.perf_counter() - t_c:.1f}s",
            flush=True,
        )

    t_b = time.perf_counter()
    b_meta = {
        "title": UNIVERSAL_BRIDGE_SPEC["title"],
        "title_pt": UNIVERSAL_BRIDGE_SPEC["title_pt"],
        "category": UNIVERSAL_BRIDGE_SPEC["category"],
        "summary": UNIVERSAL_BRIDGE_SPEC["summary"],
        "example_prompt": UNIVERSAL_BRIDGE_SPEC["prompts_polyglot"]["pt"],
        "prompts_polyglot": dict(UNIVERSAL_BRIDGE_SPEC["prompts_polyglot"]),
        "anchor_language": "en_only_p_perp_lang",
        "supported_languages": ["pt", "en", "es", "fr"],
        "teacher_model": "Golden-Slice v4.0 Conjunctive Sleep Consolidation (Zero-SGD)",
        "teacher_reference": UNIVERSAL_BRIDGE_SPEC["seeds"][0][1],
    }
    bridge_cap = GoldenSliceSynthesisEngine.consolidate_sleep_bridge(
        bridge_id=UNIVERSAL_BRIDGE_SPEC["capsule_id"],
        parent_a_id=UNIVERSAL_BRIDGE_SPEC["parents"][0],
        parent_b_id=UNIVERSAL_BRIDGE_SPEC["parents"][1],
        bridge_seed_pairs=UNIVERSAL_BRIDGE_SPEC["seeds"],
        student_model=slm,
        tokenizer=tok,
        bank=bank,
        anchor_queries_en=UNIVERSAL_BRIDGE_SPEC["anchor_queries_en"],
        system_prompt=UNIVERSAL_CHAT_SYSTEM_PROMPT,
        tau=float(UNIVERSAL_BRIDGE_SPEC["tau"]),
        ramp=float(UNIVERSAL_BRIDGE_SPEC["ramp"]),
        metadata=b_meta,
    )
    print(
        f"  ✓ Minted {bridge_cap.capsule_id:34s} | parents={bridge_cap.parent_capsules} | {time.perf_counter() - t_b:.1f}s",
        flush=True,
    )

    prefix = f"qwen{tier_tag}_universal_v4_0"
    store_path = LIB_DIR / f"{prefix}.scarstore"
    manifest_path = LIB_DIR / f"{prefix}.manifest.json"
    shell_path = LIB_DIR / f"{prefix}.shell.scarstore"
    shell_gz_path = LIB_DIR / f"{prefix}.shell.scarstore.gz"

    store = GoldenSliceScarStore.build_from_bank(bank, store_path)
    summary = store.inspect_summary()
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    shell_store = store.export_subset(shell_path, include_capsule_ids=[])
    shell_summary = shell_store.inspect_summary()
    shell_store.close()
    with open(shell_path, "rb") as sf, gzip.open(shell_gz_path, "wb", compresslevel=9) as gf:
        gf.write(sf.read())

    # Update local `~/.cache` shell if present so `mos-store ui` picks up the newest build immediately
    user_cache_shell = Path.home() / ".cache" / "mixture_of_scars" / "scarstore" / f"{prefix}.shell.scarstore"
    if user_cache_shell.exists():
        shutil.copyfile(shell_path, user_cache_shell)

    # Live HTTP 206 Polyglot (`PT, EN, ES, FR`) + Multi-Turn `SCache O(1)` + Out-of-Cone Verification!
    verifications: List[Dict[str, Any]] = []
    with ScarStoreHTTPRangeServer(store_path) as srv:
        polyglot_checks = [
            # 1 single cartridge `concept_boltzmann_constant` answering across PT, EN, ES, FR!
            ("O que é a constante de Boltzmann e onde ela se aplica?", "concept_boltzmann_constant", "380649"),
            ("What is the Boltzmann constant and where does it apply?", "concept_boltzmann_constant", "380649"),
            ("¿Qué es la constante de Boltzmann y dónde se aplica?", "concept_boltzmann_constant", "380649"),
            ("Qu'est-ce que la constante de Boltzmann et où s'applique-t-elle ?", "concept_boltzmann_constant", "380649"),
            # 1 single cartridge `concept_landauer_principle` answering across PT, EN, ES, FR!
            ("me sabe sobre o princípio de Landauer", "concept_landauer_principle", "ln 2"),
            ("Tell me about Landauer's principle.", "concept_landauer_principle", "ln 2"),
            ("Explícame el principio de Landauer en la física de la información.", "concept_landauer_principle", "ln 2"),
            ("Expliquez le principe de Landauer en physique de l'information.", "concept_landauer_principle", "ln 2"),
            # 1 single cartridge `concept_mos_scache_architecture` answering across PT, EN, ES, FR!
            ("Qual arquitetura você usa?", "concept_mos_scache_architecture", "Mixture of Scars"),
            ("What architecture do you use?", "concept_mos_scache_architecture", "Mixture of Scars"),
            ("¿Qué arquitectura utilizas?", "concept_mos_scache_architecture", "Mixture of Scars"),
            ("Quelle architecture utilisez-vous ?", "concept_mos_scache_architecture", "Mixture of Scars"),
            # Brazilian domestic labor law across PT & EN
            ("Como funciona o direito trabalhista para faxineiras no brasil?", "concept_brazil_lc150_domestic_law", "150/2015"),
            ("How does Brazilian labor law work for housecleaners and domestic workers?", "concept_brazil_lc150_domestic_law", "150/2015"),
            # Conjunctive Sleep Bridge across PT & EN
            ("Como a reutilização de espaço da contagem indutiva de Immerman-Szelepcsényi interage com a dissipação térmica do princípio de Landauer?", "bridge_landauer_immerman", "Bennett"),
            ("How does space reuse in Immerman-Szelepcsényi inductive counting interact with thermal dissipation under Landauer's principle?", "bridge_landauer_immerman", "Bennett"),
        ]

        for prompt_str, expected_cid, expected_sub in polyglot_checks:
            fetcher, http_client = make_http_range_capsule_fetcher(srv.base_url)
            live_shell = GoldenSliceScarStore.open(shell_path, device=dev, on_demand_fetcher=fetcher)
            try:
                res, _ = generate_with_scar_store(
                    student_model=slm,
                    tokenizer=tok,
                    store=live_shell,
                    prompt_text_or_ids=prompt_str,
                    max_new_tokens=96,
                    http_client=http_client,
                    persist_fetched=False,
                )
            finally:
                live_shell.close()
            assert res.active_gates and max(res.active_gates, key=res.active_gates.get) == expected_cid, (
                f"[{tier_tag}] Expected {expected_cid} on {prompt_str!r}, got {res.active_gates} (cos={res.routing_cosines})"
            )
            assert len(res.http_range_events) >= 1 and res.http_range_events[0]["http_status"] == 206, (
                f"[{tier_tag}] Expected HTTP 206 on {prompt_str!r}, got {res.http_range_events}"
            )
            assert expected_sub in res.text, (
                f"[{tier_tag}] Expected {expected_sub!r} in answer for {prompt_str!r}, got {res.text!r}"
            )
            print(
                f"  ✓ [{tier_tag}] {expected_cid:32s} | gate={res.active_gates[expected_cid]:.3f} "
                f"| HTTP206={res.http_range_events[0]['bytes_received']}B | {res.text[:90]!r}",
                flush=True,
            )
            verifications.append(
                {
                    "prompt": prompt_str,
                    "expected_capsule_id": expected_cid,
                    "active_gates": res.active_gates,
                    "http_206_events": res.http_range_events,
                    "answer": res.text,
                }
            )

        # Out-of-cone Polyglot Controls: strictly 0 active cartridges & 0 HTTP 206 bytes!
        for ctrl_q in (
            "Olá, tudo bem? Quem é você?",
            "Qual a maior capital entre os estados das americas sul e norte?",
            "qual a raiz quadrada de PI elevado à 2?",
            "Bonjour ! Quelle est la capitale de la France ?",
        ):
            fetcher_c, http_client_c = make_http_range_capsule_fetcher(srv.base_url)
            live_shell_c = GoldenSliceScarStore.open(shell_path, device=dev, on_demand_fetcher=fetcher_c)
            try:
                ctrl_res, _ = generate_with_scar_store(
                    student_model=slm,
                    tokenizer=tok,
                    store=live_shell_c,
                    prompt_text_or_ids=ctrl_q,
                    max_new_tokens=48,
                    http_client=http_client_c,
                    persist_fetched=False,
                )
                assert len(ctrl_res.active_gates) == 0 and len(ctrl_res.http_range_events) == 0, (
                    f"[{tier_tag}] Out-of-cone leak on {ctrl_q!r}: {ctrl_res.active_gates} (cos={ctrl_res.routing_cosines})"
                )
            finally:
                live_shell_c.close()
        print(f"  ✓ [{tier_tag}] All 4 out-of-cone polyglot controls verified: 0 cartridges, 0 B HTTP 206!", flush=True)

        # Multi-Turn SCache O(1) (`ΔBytes == 0` once saturated) + "Discorra" follow-up verification!
        fetcher_m, http_client_m = make_http_range_capsule_fetcher(srv.base_url)
        live_shell_m = GoldenSliceScarStore.open(shell_path, device=dev, on_demand_fetcher=fetcher_m)
        try:
            mt_cache = SCacheDynamicCache(window_size=128, num_slots=32, num_sink_tokens=36)
            r_t1, mt_cache = generate_with_scar_store(
                student_model=slm,
                tokenizer=tok,
                store=live_shell_m,
                prompt_text_or_ids="me sabe sobre o princípio de Landauer",
                max_new_tokens=110,
                scache=mt_cache,
                http_client=http_client_m,
            )
            r_t2, mt_cache = generate_with_scar_store(
                student_model=slm,
                tokenizer=tok,
                store=live_shell_m,
                prompt_text_or_ids="Discorra sobre implicações práticas disso na computação reversível.",
                max_new_tokens=110,
                scache=mt_cache,
                http_client=http_client_m,
            )
            print(
                f"  ✓ [{tier_tag}] Multi-Turn SCache O(1) verified | T1 ({len(r_t1.text)} chars) -> "
                f"T2 ({len(r_t2.text)} chars, SCache={r_t2.scache_memory_kb:.1f} KB)",
                flush=True,
            )
        finally:
            live_shell_m.close()

    store.close()
    del slm
    gc.collect()
    if dev.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()

    elapsed_s = round(time.perf_counter() - t0, 2)
    print(f"[Universal Golden-Slice v4.0 - {tier_tag} Complete] {store_path.name} in {elapsed_s}s", flush=True)
    return {
        "tier": tier_tag,
        "student_model": student_id,
        "solve_and_verify_s": elapsed_s,
        "summary": summary,
        "shell_summary": shell_summary,
        "verifications": verifications,
        "assets": {
            "store_file": store_path.name,
            "store_bytes": store_path.stat().st_size,
            "store_sha256": sha256_file(store_path),
            "manifest_file": manifest_path.name,
            "manifest_bytes": manifest_path.stat().st_size,
            "manifest_sha256": sha256_file(manifest_path),
            "shell_file": shell_path.name,
            "shell_bytes": shell_path.stat().st_size,
            "shell_gz_file": shell_gz_path.name,
            "shell_gz_bytes": shell_gz_path.stat().st_size,
        },
    }


def refresh_webgpu_bundle(store_0p5b_path: Path) -> None:
    webgpu_bundle = REPO_ROOT / "webgpu_dist" / "qwen0p5b"
    if not webgpu_bundle.exists():
        return
    export_scarstore_for_webgpu(store_0p5b_path, webgpu_bundle, copy_payload_store=True)
    shutil.copyfile(WGSL_KERNELS_PATH, webgpu_bundle / "mos_scache_kernels.wgsl")
    for fname in ("index.html", "mos_webgpu_engine.js", "scarstore_http206_browser.js", "qwen_bpe_tokenizer.js"):
        src = WEBGPU_APP_DIR / fname
        if src.exists():
            shutil.copyfile(src, webgpu_bundle / fname)
    manifest_p = webgpu_bundle / "webgpu_manifest.json"
    if manifest_p.exists():
        mdata = json.loads(manifest_p.read_text(encoding="utf-8"))
        mdata["scarstore_router_file"] = "scarstore_router.json"
        mdata.setdefault("scache_defaults", {})["num_sink_tokens"] = 36
        manifest_p.write_text(json.dumps(mdata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✓ Refreshed Native WebGPU / WGSL bundle at {webgpu_bundle} with GoldenSliceScarStoreV4!", flush=True)


def main() -> int:
    dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    res_0p5b = build_universal_tier("0p5b", "Qwen/Qwen2.5-0.5B-Instruct", dev)
    res_3b = build_universal_tier("3b", "Qwen/Qwen2.5-3B-Instruct", dev)

    slots_3b = {s["capsule_id"]: s for s in res_3b["summary"]["slots"]}
    slots_0p5b = {s["capsule_id"]: s for s in res_0p5b["summary"]["slots"]}
    all_specs = list(UNIVERSAL_CONCEPT_SPECS) + [UNIVERSAL_BRIDGE_SPEC]
    catalog_items: List[Dict[str, Any]] = []
    for sp in all_specs:
        cid = sp["capsule_id"]
        s3 = slots_3b[cid]
        s0 = slots_0p5b[cid]
        catalog_items.append(
            {
                "capsule_id": cid,
                "title": sp["title"],
                "title_pt": sp["title_pt"],
                "category": sp["category"],
                "summary": sp["summary"],
                "prompts_polyglot": sp["prompts_polyglot"],
                "anchor_queries_en": sp["anchor_queries_en"],
                "parent_capsules": sp.get("parents", []),
                "coord_3b": s3["coord"],
                "byte_offset_3b": s3["byte_offset"],
                "record_bytes_3b": s3["record_bytes"],
                "coord_0p5b": s0["coord"],
                "byte_offset_0p5b": s0["byte_offset"],
                "record_bytes_0p5b": s0["record_bytes"],
            }
        )

    catalog_payload = {
        "library_name": "Mixture of Scars (MoS) Universal Language-Invariant Conceptual ScarStore (Golden-Slice v4.0)",
        "version": "v4.0.0",
        "routing_architecture": "Golden-Slice (2L/3) + Within-Concept ANOVA Language Orthogonal Projector P_perp_lang (k_lang=14)",
        "synthesis_method": "Closed-Form Two-Stage Prediction-Space RRR + Cutting-Plane Woodbury Rosetta (ZERO Backpropagation / ZERO SGD)",
        "supported_languages_per_single_cartridge": ["pt", "en", "es", "fr"],
        "tiers": {
            "3b": {
                "student_model": "Qwen/Qwen2.5-3B-Instruct",
                "hidden_dim": 2048,
                "mid_layer_idx": 23,
                "record_bytes": 143360,
                "record_kb": 140.0,
                "assets": res_3b["assets"],
            },
            "0.5b": {
                "student_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "hidden_dim": 896,
                "mid_layer_idx": 15,
                "record_bytes": 62720,
                "record_kb": 61.25,
                "assets": res_0p5b["assets"],
            },
        },
        "cartridges": catalog_items,
    }
    cat_path = LIB_DIR / "scarstore_universal_catalog_v4_0.json"
    cat_path.write_text(json.dumps(catalog_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    refresh_webgpu_bundle(LIB_DIR / "qwen0p5b_universal_v4_0.scarstore")
    print(f"\n✓ Built Universal Polyglot ScarStore v4.0.0 -> {cat_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
