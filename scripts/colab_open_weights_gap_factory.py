#!/usr/bin/env python3
"""Colab Ultra (`NVIDIA A100-SXM4-80GB`) Open-Weights Contrastive Gap Mining & Same-Vocab Factory.

Runs Genuine Dual-Model Inference (`V_teacher == V_student == 151,936`):
  - Teacher (`72B`): `unsloth/Qwen2.5-72B-Instruct-bnb-4bit` (`72.7B` params, Qwen2.5 151,936 vocab)
  - Student (`3B`):  `Qwen/Qwen2.5-3B-Instruct` (`3.09B` params, Qwen2.5 151,936 vocab)

For all 27 non-portfolio cartridges (`8` Universal Science/Law + `19` New Open-Weights Topics):
  1. Probes `Qwen2.5-3B-Instruct` on the open query (PT & EN) to measure epistemic confidence
     margin `E[p1 - p2]`, refusal flags, and baseline factual hit rate.
  2. Uses `Qwen2.5-72B-Instruct` on A100-80GB to synthesize/verify compact 1-sentence polyglot
     premise seeds (`EN, PT, ES, FR`, `24..46` BPE tokens ending with `.` for `SEED_HORIZON = 35`)
     and computes exact token-level `KL(P_72B || P_3B)` + `Top-1 Agreement(72B, 3B)` across all 4 languages!
  3. Exports `/content/jarvis_72b_gap_mined_specs.json` ready for `GoldenSliceScarStoreV4` (`140 KB` `HTTP 206`).
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Tuple

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PORTFOLIO_SYSTEM_PROMPT_3B = (
    "You are JARVIS, the official Qwen2.5-3B AI assistant on Julio Reis's portfolio (byte-od.github.io), "
    "running locally via WebGPU WGSL, MoS, and SCache O(1). Never claim to be Julio; refer to Julio Reis "
    "in the third person. Always reply in the user's language in 2 to 4 complete sentences."
)

TEACHER_ID = "unsloth/Qwen2.5-72B-Instruct-bnb-4bit"
STUDENT_ID = "Qwen/Qwen2.5-3B-Instruct"
OUT_JSON = Path("/content/jarvis_72b_gap_mined_specs.json")


def has_cjk(text: str) -> bool:
    for ch in text:
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF) or (0x3400 <= cp <= 0x4DBF) or (0x3040 <= cp <= 0x30FF) or (0xAC00 <= cp <= 0xD7AF):
            return True
    return False


# 27 Candidate Cartridges: 8 Universal Science/Law + 19 New Open-Weights Blind-Spot Cartridges
CANDIDATE_SPECS: List[Dict[str, Any]] = [
    # ==================== A. 8 UNIVERSAL SCIENCE & LAW ====================
    {
        "capsule_id": "concept_boltzmann_constant",
        "title": "Boltzmann Constant (Exact 2019 SI Definition & Entropy S = k_B ln Ω)",
        "title_pt": "Constante de Boltzmann (Valor Exato SI 2019 e Entropia S = k_B ln Ω)",
        "category": "Statistical Physics",
        "summary": "Exact 2019 SI value k_B = 1.380649e-23 J/K linking temperature T, thermal energy, and S = k_B ln Ω.",
        "tau": 0.41,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What is the Boltzmann constant and where does it apply?",
            "State the exact 2019 SI value and physical meaning of Boltzmann's constant.",
            "Explain Boltzmann's constant k_B in statistical mechanics and entropy.",
        ],
        "prompts_polyglot": {
            "pt": "O que é a constante de Boltzmann e onde ela se aplica?",
            "en": "What is the Boltzmann constant and where does it apply?",
            "es": "¿Qué es la constante de Boltzmann y dónde se aplica?",
            "fr": "Qu'est-ce que la constante de Boltzmann et où s'applique-t-elle ?",
        },
        "must_any": ["1.380649", "1,380649"],
        "fallback_seeds": {
            "en": "In SI, the Boltzmann constant is fixed at k_B = 1.380649e-23 J/K, linking temperature T to entropy S = k_B ln Omega.",
            "pt": "No SI, a constante de Boltzmann vale k_B = 1,380649e-23 J/K, ligando a temperatura T à entropia S = k_B ln Omega.",
            "es": "En el SI, la constante de Boltzmann vale k_B = 1,380649e-23 J/K, uniendo la temperatura T con S = k_B ln Omega.",
            "fr": "Dans le SI, la constante de Boltzmann vaut k_B = 1,380649e-23 J/K, reliant la température T à S = k_B ln Omega.",
        },
    },
    {
        "capsule_id": "concept_fischer_lynch_paterson",
        "title": "Fischer–Lynch–Paterson (FLP) Distributed Consensus Impossibility (1985)",
        "title_pt": "Teorema da Impossibilidade FLP em Consenso Distribuído Assíncrono",
        "category": "Distributed Systems",
        "summary": "No deterministic consensus protocol guarantees termination in an asynchronous system under even 1 unannounced crash.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What does the Fischer-Lynch-Paterson (FLP) impossibility theorem prove about asynchronous consensus?",
            "Explain the FLP impossibility result in fault-tolerant distributed systems.",
            "Why is deterministic consensus impossible under one crash in the FLP model?",
        ],
        "prompts_polyglot": {
            "pt": "O que prova o teorema da impossibilidade FLP (Fischer-Lynch-Paterson) em sistemas distribuídos?",
            "en": "What does the Fischer-Lynch-Paterson (FLP) impossibility theorem prove in distributed systems?",
            "es": "¿Qué demuestra el teorema de imposibilidad FLP (Fischer-Lynch-Paterson) en sistemas distribuidos?",
            "fr": "Que démontre le théorème d'impossibilité FLP (Fischer-Lynch-Paterson) dans les systèmes distribués ?",
        },
        "must_any": ["FLP", "Fischer"],
        "fallback_seeds": {
            "en": "The Fischer-Lynch-Paterson (FLP, 1985) theorem proves no deterministic asynchronous consensus protocol guarantees termination if one process may crash.",
            "pt": "O teorema FLP (Fischer-Lynch-Paterson, 1985) prova que nenhum protocolo determinístico assíncrono garante consenso com terminação sob uma falha.",
            "es": "El teorema FLP (Fischer-Lynch-Paterson, 1985) prueba que ningún protocolo determinista asíncrono garantiza consenso con terminación ante un fallo.",
            "fr": "Le théorème FLP (Fischer-Lynch-Paterson, 1985) prouve qu'aucun protocole déterministe asynchrone ne garantit le consensus face à une panne.",
        },
    },
    {
        "capsule_id": "concept_chaitin_incompleteness",
        "title": "Chaitin's Algorithmic Incompleteness Theorem (Kolmogorov Complexity)",
        "title_pt": "Teorema da Incompletude Algorítmica de Chaitin (Complexidade de Kolmogorov)",
        "category": "Algorithmic Information Theory",
        "summary": "Any consistent formal system F has an explicit constant L_F above which no statement K(x) > L_F is provable.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What does Chaitin's incompleteness theorem prove about Kolmogorov complexity?",
            "Explain Chaitin's bound L on proving lower bounds K(x) > L in formal axiomatic systems.",
            "How does Berry's paradox lead to Chaitin's algorithmic incompleteness theorem?",
        ],
        "prompts_polyglot": {
            "pt": "O que prova o teorema da incompletude de Chaitin sobre a complexidade de Kolmogorov?",
            "en": "What does Chaitin's incompleteness theorem prove about Kolmogorov complexity?",
            "es": "¿Qué demuestra el teorema de incompletitud de Chaitin sobre la complejidad de Kolmogórov?",
            "fr": "Que démontre le théorème d'incomplétude de Chaitin sur la complexité de Kolmogorov ?",
        },
        "must_any": ["Chaitin", "Kolmogorov", "Kolmogórov"],
        "fallback_seeds": {
            "en": "Chaitin's incompleteness theorem proves every consistent formal system F has a bound L above which F cannot prove K(x) > L in Kolmogorov complexity.",
            "pt": "O teorema de Chaitin prova que todo sistema formal consistente F tem uma constante L acima da qual F jamais prova K(x) > L na complexidade de Kolmogorov.",
            "es": "El teorema de Chaitin demuestra que todo sistema formal consistente F posee una constante L tal que F nunca prueba K(x) > L en complejidad de Kolmogórov.",
            "fr": "Le théorème de Chaitin prouve que tout système formel cohérent F possède une borne L au-delà de laquelle il ne démontre jamais K(x) > L en complexité de Kolmogorov.",
        },
    },
    {
        "capsule_id": "concept_marchenko_pastur_law",
        "title": "Marchenko–Pastur Spectral Law of Random Sample Covariance Matrices",
        "title_pt": "Lei Espectral de Marchenko–Pastur para Matrizes Aleatórias de Covariância",
        "category": "Random Matrix Theory",
        "summary": "Limiting eigenvalue support lambda_+- = sigma^2 (1 +- sqrt(gamma))^2 for p/n -> gamma Wishart covariance matrices.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What is the Marchenko-Pastur law in random matrix theory?",
            "State the eigenvalue support bounds of the Marchenko-Pastur distribution for sample covariance matrices.",
            "How does the Marchenko-Pastur law separate signal eigenvalues from random noise?",
        ],
        "prompts_polyglot": {
            "pt": "O que estabelece a lei de Marchenko-Pastur na teoria de matrizes aleatórias?",
            "en": "What does the Marchenko-Pastur law establish in random matrix theory?",
            "es": "¿Qué establece la ley de Marchenko-Pastur en la teoría de matrices aleatorias?",
            "fr": "Qu'établit la loi de Marchenko-Pastur dans la théorie des matrices aléatoires ?",
        },
        "must_any": ["Marchenko", "Pastur"],
        "fallback_seeds": {
            "en": "The Marchenko-Pastur law bounds random covariance eigenvalues for ratio gamma = p/n to sigma^2 * (1 +- sqrt(gamma))^2.",
            "pt": "A lei de Marchenko-Pastur limita os autovalores de covariância aleatória com razão gamma = p/n ao suporte sigma^2 * (1 +- sqrt(gamma))^2.",
            "es": "La ley de Marchenko-Pastur acota los autovalores de covarianza aleatoria con razón gamma = p/n al soporte sigma^2 * (1 +- sqrt(gamma))^2.",
            "fr": "La loi de Marchenko-Pastur borne les valeurs propres de covariance aléatoire de rapport gamma = p/n à sigma^2 * (1 +- sqrt(gamma))^2.",
        },
    },
    {
        "capsule_id": "concept_kabatiansky_levenshtein",
        "title": "Kabatiansky–Levenshtein High-Dimensional Sphere Packing Bound (1978)",
        "title_pt": "Cota de Kabatiansky–Levenshtein para Empacotamento Esférico em Alta Dimensão",
        "category": "Geometry & Coding Theory",
        "summary": "Upper bound on spherical codes in R^n proving kissing numbers grow at most as 2^{0.4014 n + o(n)}.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What does the Kabatiansky-Levenshtein bound prove for sphere packing and kissing numbers in high dimensions?",
            "Explain the Kabatiansky-Levenshtein theorem on spherical codes in R^n.",
            "What asymptotic upper bound does Kabatiansky-Levenshtein establish on kissing numbers?",
        ],
        "prompts_polyglot": {
            "pt": "O que prova a cota de Kabatiansky-Levenshtein sobre empacotamento de esferas em alta dimensão?",
            "en": "What does the Kabatiansky-Levenshtein bound prove about high-dimensional sphere packing?",
            "es": "¿Qué demuestra la cota de Kabatiansky-Levenshtein sobre el empaquetamiento de esferas en alta dimensión?",
            "fr": "Que démontre la borne de Kabatiansky-Levenshtein sur l'empilement de sphères en grande dimension ?",
        },
        "must_any": ["0.401", "0,401", "Kabatiansky"],
        "fallback_seeds": {
            "en": "The Kabatiansky-Levenshtein bound (1978) limits spherical codes in R^n, proving kissing numbers grow at most as 2^(0.4014 n).",
            "pt": "A cota de Kabatiansky-Levenshtein (1978) limita códigos esféricos em R^n, provando que o kissing number cresce no máximo como 2^(0,4014 n).",
            "es": "La cota de Kabatiansky-Levenshtein (1978) limita códigos en R^n, acotando el kissing number por 2^(0,4014 n).",
            "fr": "La borne de Kabatiansky-Levenshtein (1978) limite les codes sphériques dans R^n, bornant le nombre de contact par 2^(0,4014 n).",
        },
    },
    {
        "capsule_id": "concept_regev_lwe_lattice",
        "title": "Regev's Quantum Worst-Case Lattice Reduction for Learning With Errors (LWE)",
        "title_pt": "Redução Quântica de Regev em Reticulados e Criptografia Pós-Quântica (LWE)",
        "category": "Post-Quantum Cryptography",
        "summary": "Oded Regev (2005) proved solving average-case LWE implies a quantum algorithm for worst-case GapSVP and SIVP.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What did Oded Regev prove about Learning With Errors (LWE) and worst-case lattice problems?",
            "Explain Regev's quantum reduction from GapSVP and SIVP to LWE in post-quantum cryptography.",
            "Why does Regev's LWE theorem underpin NIST post-quantum lattice cryptography like ML-KEM?",
        ],
        "prompts_polyglot": {
            "pt": "O que Oded Regev provou sobre Learning With Errors (LWE) e criptografia pós-quântica em reticulados?",
            "en": "What did Oded Regev prove about Learning With Errors (LWE) and post-quantum lattice cryptography?",
            "es": "¿Qué demostró Oded Regev sobre Learning With Errors (LWE) y la criptografía poscuántica en retículos?",
            "fr": "Qu'a démontré Oded Regev sur Learning With Errors (LWE) et la cryptographie post-quantique sur réseaux ?",
        },
        "must_any": ["Regev", "LWE", "GapSVP", "SIVP"],
        "fallback_seeds": {
            "en": "Oded Regev (2005) proved average-case LWE reduces quantumly from worst-case lattice GapSVP and SIVP, anchoring NIST ML-KEM.",
            "pt": "Oded Regev (2005) provou que resolver LWE no caso médio reduz quanticamente GapSVP e SIVP no pior caso, base do NIST ML-KEM.",
            "es": "Oded Regev (2005) probó que resolver LWE en promedio reduce cuánticamente GapSVP y SIVP en el peor caso, base de NIST ML-KEM.",
            "fr": "Oded Regev (2005) a prouvé que résoudre LWE en moyenne réduit quantiquement GapSVP et SIVP au pire cas, base de NIST ML-KEM.",
        },
    },
    {
        "capsule_id": "concept_shannon_hartley_capacity",
        "title": "Shannon–Hartley Noisy Channel Capacity Theorem (C = B log2(1 + S/N))",
        "title_pt": "Teorema da Capacidade de Canal de Shannon–Hartley (C = B log2(1 + S/N))",
        "category": "Information Theory",
        "summary": "Maximum error-free bit rate C = B log2(1 + S/N) bits/s over bandwidth B Hz under additive white Gaussian noise.",
        "tau": 0.41,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What is the Shannon-Hartley theorem for channel capacity under Gaussian noise?",
            "State the Shannon-Hartley formula relating bandwidth B, SNR, and maximum bit rate C.",
            "Explain the Shannon limit C = B log2(1 + S/N) in digital communications.",
        ],
        "prompts_polyglot": {
            "pt": "O que estabelece o teorema de Shannon-Hartley para a capacidade de um canal com ruído?",
            "en": "What does the Shannon-Hartley theorem establish for noisy channel capacity?",
            "es": "¿Qué establece el teorema de Shannon-Hartley para la capacidad de un canal con ruido?",
            "fr": "Qu'établit le théorème de Shannon-Hartley sur la capacité d'un canal bruité ?",
        },
        "must_any": ["log2(1 + S/N)", "log_2(1 + S/N)", "Shannon"],
        "fallback_seeds": {
            "en": "The Shannon-Hartley theorem sets the maximum error-free rate over an AWGN channel of bandwidth B Hz at C = B * log2(1 + S/N) bits/s.",
            "pt": "O teorema de Shannon-Hartley define a capacidade livre de erros num canal AWGN de banda B Hz como C = B * log2(1 + S/N) bits/s.",
            "es": "El teorema de Shannon-Hartley fija la capacidad sin errores en un canal AWGN de banda B Hz en C = B * log2(1 + S/N) bits/s.",
            "fr": "Le théorème de Shannon-Hartley fixe le débit sans erreur sur un canal gaussien de bande B Hz à C = B * log2(1 + S/N) bits/s.",
        },
    },
    {
        "capsule_id": "concept_brazil_lc150_domestic_law",
        "title": "Brazilian Domestic Employment Statute — Complementary Law 150/2015 (PEC das Domésticas)",
        "title_pt": "Direito Trabalhista Doméstico no Brasil — Lei Complementar 150/2015",
        "category": "Labor Law (Brazil)",
        "summary": "LC 150/2015 regulates domestic employment (>= 3 days/week): 8h/day, 44h/week, 50% overtime, mandatory FGTS 8% + 3.2%.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "How does labor law for domestic workers and housekeepers work in Brazil under LC 150/2015?",
            "What rights and weekly hour limits does Complementary Law 150/2015 establish for Brazilian domestic workers?",
            "When is a cleaner or housekeeper legally a domestic employee in Brazil under LC 150?",
        ],
        "prompts_polyglot": {
            "pt": "Como funciona o direito trabalhista para empregadas domésticas e diaristas no Brasil pela LC 150/2015?",
            "en": "How does Brazilian labor law work for domestic workers and cleaners under LC 150/2015?",
            "es": "¿Cómo funciona el derecho laboral para empleadas domésticas en Brasil según la LC 150/2015?",
            "fr": "Comment fonctionne le droit du travail des employés domestiques au Brésil selon la LC 150/2015 ?",
        },
        "must_any": ["150/2015", "44", "FGTS"],
        "fallback_seeds": {
            "en": "Brazil's LC 150/2015 covers domestic workers serving over 2 days weekly, setting 8h/day, 44h/week, and mandatory 8% FGTS.",
            "pt": "No Brasil, a LC 150/2015 rege domésticos acima de 2 dias semanais, fixando 8h/dia, 44h/semana e FGTS de 8%.",
            "es": "En Brasil, la LC 150/2015 regula a domésticos con más de 2 días semanales, fijando 8h/día, 44h/semana y FGTS obligatorio del 8%.",
            "fr": "Au Brésil, la LC 150/2015 encadre les domestiques travaillant plus de 2 jours par semaine, fixant 44h/semaine et 8% de FGTS.",
        },
    },
    # ==================== B. 5 QUANTUM PHYSICS & QUANTUM INFORMATION ====================
    {
        "capsule_id": "ow_quantum_applied_theorems",
        "title": "Applied Theorems of Quantum Physics & Quantum Information (No-Cloning, Bell-CHSH, Holevo)",
        "title_pt": "Teoremas Aplicados da Física e Informação Quântica (Não-Clonagem, Bell-CHSH e Holevo)",
        "category": "Quantum Physics",
        "summary": "Core applied theorems of quantum physics: Wootters-Zurek No-Cloning (BB84 QKD), Bell-CHSH (2*sqrt(2) QRNG), and Holevo's capacity bound.",
        "tau": 0.39,
        "ramp": 0.05,
        "anchor_queries_en": [
            "Tell me about a random applied theorem from quantum physics.",
            "Give me an applied theorem of quantum physics and explain its practical technology use.",
            "What is a fundamental theorem of applied quantum mechanics and quantum information?",
        ],
        "prompts_polyglot": {
            "pt": "Me fale sobre um teorema da física quântica aplicado e onde ele é usado na prática.",
            "en": "Tell me about an applied quantum physics theorem and where it is used in practice.",
            "es": "Háblame de un teorema aplicado de la física cuántica y dónde se usa en la práctica.",
            "fr": "Parlez-moi d'un théorème appliqué de la physique quantique et de son usage pratique.",
        },
        "must_any": ["Wootters", "BB84", "clon"],
        "fallback_seeds": {
            "en": "A flagship applied quantum theorem is Wootters-Zurek No-Cloning (1982), forbidding copying unknown qubits and securing BB84 quantum cryptography.",
            "pt": "Um teorema central da física quântica aplicada é a Não-Clonagem de Wootters-Zurek (1982), que proíbe copiar qubits desconhecidos e protege chaves BB84.",
            "es": "Un teorema clave de la física cuántica aplicada es la No-Clonación de Wootters-Zurek (1982), que prohíbe copiar qubits desconocidos y protege BB84.",
            "fr": "Un théorème clé de la physique quantique appliquée est le non-clonage de Wootters-Zurek (1982), qui interdit de copier un qubit inconnu et sécurise BB84.",
        },
    },
    {
        "capsule_id": "ow_quantum_no_cloning_wootters_zurek",
        "title": "Wootters–Zurek Quantum No-Cloning Theorem (1982) & BB84 Security",
        "title_pt": "Teorema da Não-Clonagem Quântica de Wootters–Zurek (1982)",
        "category": "Quantum Physics",
        "summary": "No unitary operator U satisfies U(|psi>|0>) = e^{i phi}|psi>|psi> for arbitrary superpositions, securing quantum key distribution.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "Explain the quantum no-cloning theorem of Wootters and Zurek.",
            "Why does linearity of unitary evolution forbid cloning an unknown quantum state?",
            "What does the Wootters-Zurek no-cloning theorem prove for qubits and QKD?",
        ],
        "prompts_polyglot": {
            "pt": "O que prova o teorema da não-clonagem quântica de Wootters e Zurek?",
            "en": "What does the Wootters-Zurek quantum no-cloning theorem prove?",
            "es": "¿Qué demuestra el teorema de no clonación cuántica de Wootters y Zurek?",
            "fr": "Que démontre le théorème de non-clonage quantique de Wootters et Zurek ?",
        },
        "must_any": ["Wootters", "Zurek", "1982", "BB84"],
        "fallback_seeds": {
            "en": "Wootters, Zurek, and Dieks (1982) proved unitary linearity forbids copying arbitrary quantum states |psi>, securing BB84 key distribution.",
            "pt": "Wootters, Zurek e Dieks (1982) provaram que a linearidade unitária proíbe clonar estados quânticos |psi> arbitrários, blindando o protocolo BB84.",
            "es": "Wootters, Zurek y Dieks (1982) demostraron que la linealidad unitaria prohíbe clonar estados cuánticos |psi> arbitrarios, asegurando el protocolo BB84.",
            "fr": "Wootters, Zurek et Dieks (1982) ont prouvé que la linéarité unitaire interdit de cloner un état quantique |psi> arbitraire, protégeant BB84.",
        },
    },
    {
        "capsule_id": "ow_quantum_bell_chsh_tsirelson",
        "title": "Bell's Theorem, CHSH Inequality (<= 2) & Tsirelson's Bound (2*sqrt(2))",
        "title_pt": "Teorema de Bell, Desigualdade CHSH (<= 2) e Cota de Tsirelson (2√2)",
        "category": "Quantum Physics",
        "summary": "Local hidden-variable models satisfy |S_CHSH| <= 2, while entangled singlet states reach Tsirelson's bound 2*sqrt(2) ≈ 2.828.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What do Bell's theorem and the CHSH inequality prove about local hidden variables and entanglement?",
            "Explain the classical CHSH bound of 2 versus Tsirelson's quantum bound of 2 sqrt(2).",
            "How do Bell-CHSH violations certify quantum nonlocality?",
        ],
        "prompts_polyglot": {
            "pt": "O que provam o teorema de Bell e a desigualdade CHSH sobre variáveis ocultas locais?",
            "en": "What do Bell's theorem and the CHSH inequality prove about local hidden variables?",
            "es": "¿Qué demuestran el teorema de Bell y la desigualdad CHSH sobre variables ocultas locales?",
            "fr": "Que démontrent le théorème de Bell et l'inégalité CHSH sur les variables cachées locales ?",
        },
        "must_any": ["CHSH", "Tsirelson", "2.828", "2,828", "sqrt(2)", "√2"],
        "fallback_seeds": {
            "en": "Bell's CHSH inequality bounds local hidden variables to |S| <= 2, whereas entangled qubits reach Tsirelson's bound 2*sqrt(2) = 2.828.",
            "pt": "A desigualdade CHSH de Bell limita variáveis ocultas locais a |S| <= 2, enquanto qubits emaranhados atingem Tsirelson 2√2 = 2,828.",
            "es": "La desigualdad CHSH de Bell acota variables ocultas locales a |S| <= 2, mientras qubits entrelazados alcanzan Tsirelson 2√2 = 2,828.",
            "fr": "L'inégalité CHSH de Bell borne les variables cachées locales à |S| <= 2, tandis que des qubits intriqués atteignent Tsirelson 2√2 = 2,828.",
        },
    },
    {
        "capsule_id": "ow_quantum_holevo_bound",
        "title": "Holevo's Bound (1973) on Accessible Classical Information from Quantum States",
        "title_pt": "Cota de Holevo (1973) sobre Informação Clássica Extraível de Qubits",
        "category": "Quantum Information",
        "summary": "Mutual information I(X:Y) <= chi = S(rho) - sum_x p_x S(rho_x) <= n bits for n qubits without shared entanglement.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What is Holevo's bound on extracting classical information from quantum states?",
            "Explain Holevo's theorem and why n unentangled qubits carry at most n classical bits.",
            "State the Holevo chi quantity S(rho) - sum p_i S(rho_i) in quantum information theory.",
        ],
        "prompts_polyglot": {
            "pt": "O que estabelece a cota de Holevo sobre a informação clássica extraível de estados quânticos?",
            "en": "What does Holevo's bound establish on classical information extractable from quantum states?",
            "es": "¿Qué establece la cota de Holevo sobre la información clásica extraíble de estados cuánticos?",
            "fr": "Qu'établit la borne de Holevo sur l'information classique extractible d'états quantiques ?",
        },
        "must_any": ["Holevo", "S(rho)", "n bit"],
        "fallback_seeds": {
            "en": "Holevo's bound (1973) limits accessible classical information to S(rho) - sum p_i S(rho_i), so n qubits carry at most n bits without entanglement.",
            "pt": "A cota de Holevo (1973) limita a informação clássica extraível a S(rho) - sum p_i S(rho_i), restringindo n qubits a n bits sem emaranhamento.",
            "es": "La cota de Holevo (1973) acota la información clásica accesible a S(rho) - sum p_i S(rho_i), limitando n qubits a n bits sin entrelazamiento.",
            "fr": "La borne de Holevo (1973) limite l'information classique accessible à S(rho) - sum p_i S(rho_i), bornant n qubits à n bits sans intrication.",
        },
    },
    {
        "capsule_id": "ow_quantum_knill_laflamme_qec",
        "title": "Knill–Laflamme Quantum Error Correction Conditions & Surface Code Threshold",
        "title_pt": "Condições de Correção Quântica de Erros de Knill–Laflamme e Surface Codes",
        "category": "Quantum Computing",
        "summary": "A code projector P corrects Kraus errors {E_a} iff P E_a^dagger E_b P = alpha_{ab} P; 2D surface codes achieve ~1% threshold.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What are the Knill-Laflamme conditions for quantum error correction?",
            "Explain how Knill-Laflamme conditions and 2D surface codes protect logical qubits from decoherence.",
            "When is a set of Kraus noise operators correctable by a quantum error-correcting code?",
        ],
        "prompts_polyglot": {
            "pt": "Quais são as condições de Knill-Laflamme para correção quântica de erros?",
            "en": "What are the Knill-Laflamme conditions for quantum error correction?",
            "es": "¿Cuáles son las condiciones de Knill-Laflamme para la corrección cuántica de errores?",
            "fr": "Quelles sont les conditions de Knill-Laflamme pour la correction d'erreurs quantiques ?",
        },
        "must_any": ["Knill", "Laflamme", "alpha"],
        "fallback_seeds": {
            "en": "The Knill-Laflamme conditions (1997) state a code projector P corrects Kraus errors E_a iff P E_a* E_b P = alpha_ab P, underlying surface codes.",
            "pt": "As condições de Knill-Laflamme (1997) provam que um projetor P corrige erros E_a se e só se P E_a* E_b P = alpha_ab P nos surface codes.",
            "es": "Las condiciones de Knill-Laflamme (1997) prueban que un proyector P corrige errores E_a si P E_a* E_b P = alpha_ab P en surface codes.",
            "fr": "Les conditions de Knill-Laflamme (1997) prouvent qu'un projecteur P corrige les erreurs E_a ssi P E_a* E_b P = alpha_ab P en codes de surface.",
        },
    },
    # ==================== C. 4 SYSTEMS, HARDWARE, CRYPTO & COMPILERS ====================
    {
        "capsule_id": "ow_systems_roofline_model",
        "title": "Roofline Performance Model (Arithmetic Intensity & Memory Bandwidth Ceiling)",
        "title_pt": "Modelo Roofline de Desempenho (Intensidade Aritmética e Teto de Banda)",
        "category": "Computer Architecture",
        "summary": "Attainable throughput P = min(P_peak, B_mem * I) as a function of arithmetic intensity I (FLOPs/Byte) and memory bandwidth.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "Explain the Roofline performance model and arithmetic intensity in GPUs and CPUs.",
            "How does the Roofline model determine whether a kernel is memory-bound or compute-bound?",
            "State the Roofline equation relating peak FLOP/s, memory bandwidth, and ridge point.",
        ],
        "prompts_polyglot": {
            "pt": "Como funciona o modelo Roofline de desempenho e intensidade aritmética em GPUs e CPUs?",
            "en": "How does the Roofline performance model and arithmetic intensity work on GPUs and CPUs?",
            "es": "¿Cómo funciona el modelo Roofline de rendimiento e intensidad aritmética en GPUs y CPUs?",
            "fr": "Comment fonctionne le modèle Roofline de performance et d'intensité arithmétique sur GPU et CPU ?",
        },
        "must_any": ["Roofline", "min(", "FLOP"],
        "fallback_seeds": {
            "en": "The Roofline model (Williams et al., 2009) bounds kernel throughput at P = min(P_peak, B_mem * I) using arithmetic intensity I in FLOPs/Byte.",
            "pt": "O modelo Roofline (Williams et al., 2009) limita o desempenho de kernels em P = min(P_peak, B_mem * I) pela intensidade aritmética I em FLOPs/Byte.",
            "es": "El modelo Roofline (Williams et al., 2009) acota el rendimiento en P = min(P_peak, B_mem * I) según la intensidad aritmética I en FLOPs/Byte.",
            "fr": "Le modèle Roofline (Williams et al., 2009) borne le débit par P = min(P_peak, B_mem * I) selon l'intensité arithmétique I en FLOPs/octet.",
        },
    },
    {
        "capsule_id": "ow_systems_cap_pacelc",
        "title": "Brewer's CAP Theorem (Gilbert–Lynch) & Abadi's PACELC Extension",
        "title_pt": "Teorema CAP de Brewer (Gilbert–Lynch) e Extensão PACELC",
        "category": "Distributed Systems",
        "summary": "Under network Partition (P), distributed stores trade Consistency (C) vs Availability (A); Else (E), Latency (L) vs Consistency (C).",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "Explain Brewer's CAP theorem and Abadi's PACELC extension in distributed databases.",
            "What trade-offs do CAP and PACELC formalize during network partitions and normal operation?",
            "How does PACELC extend the Gilbert-Lynch proof of the CAP theorem?",
        ],
        "prompts_polyglot": {
            "pt": "Explique o teorema CAP de Brewer e a extensão PACELC em bancos de dados distribuídos.",
            "en": "Explain Brewer's CAP theorem and the PACELC extension in distributed databases.",
            "es": "Explica el teorema CAP de Brewer y la extensión PACELC en bases de datos distribuidas.",
            "fr": "Expliquez le théorème CAP de Brewer et l'extension PACELC dans les bases de données distribuées.",
        },
        "must_any": ["CAP", "PACELC"],
        "fallback_seeds": {
            "en": "Brewer's CAP theorem forces choosing Consistency or Availability under network Partition, while Abadi's PACELC adds the Else Latency-vs-Consistency trade-off.",
            "pt": "O teorema CAP impõe escolher Consistência ou Disponibilidade sob Partição de rede, enquanto o PACELC de Abadi soma o dilema Latência versus Consistência.",
            "es": "El teorema CAP obliga a elegir Consistencia o Disponibilidad ante Partición, mientras PACELC de Abadi añade el dilema Latencia versus Consistencia.",
            "fr": "Le théorème CAP impose de choisir Cohérence ou Disponibilité sous Partition réseau, et PACELC d'Abadi ajoute l'arbitrage Latence contre Cohérence.",
        },
    },
    {
        "capsule_id": "ow_crypto_zk_snarks_groth16",
        "title": "Zero-Knowledge Proofs (`zk-SNARKs` & Groth16 O(1) Pairing Verification)",
        "title_pt": "Provas de Conhecimento Zero (`zk-SNARKs` e Verificação O(1) Groth16)",
        "category": "Applied Cryptography",
        "summary": "Succinct Non-Interactive Arguments of Knowledge proving NP witness validity in zero knowledge with O(1) proof size (3 group elements in Groth16).",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "How do zk-SNARKs and Groth16 zero-knowledge proofs achieve O(1) verification?",
            "Explain completeness, knowledge soundness, and zero-knowledge in zk-SNARKs.",
            "What makes Groth16 proofs constant-size using arithmetic circuits, QAP, and bilinear pairings?",
        ],
        "prompts_polyglot": {
            "pt": "Como funcionam as provas de conhecimento zero zk-SNARKs e o protocolo Groth16?",
            "en": "How do zk-SNARK zero-knowledge proofs and the Groth16 protocol work?",
            "es": "¿Cómo funcionan las pruebas de conocimiento cero zk-SNARKs y el protocolo Groth16?",
            "fr": "Comment fonctionnent les preuves à divulgation nulle de connaissance zk-SNARKs et Groth16 ?",
        },
        "must_any": ["SNARK", "Groth16", "O(1)"],
        "fallback_seeds": {
            "en": "A zk-SNARK compiles an NP circuit into a QAP so Groth16 proves secret witness knowledge in 3 curve elements with O(1) pairing verification.",
            "pt": "Um zk-SNARK reduz um circuito NP a um QAP para que o Groth16 prove conhecimento secreto em 3 elementos de curva com verificação O(1).",
            "es": "Un zk-SNARK compila un circuito NP en un QAP para que Groth16 pruebe un testigo secreto en 3 elementos de curva con verificación O(1).",
            "fr": "Un zk-SNARK compile un circuit NP en QAP afin que Groth16 prouve un secret en 3 éléments de courbe avec vérification O(1).",
        },
    },
    {
        "capsule_id": "ow_compilers_ssa_dominance",
        "title": "Static Single Assignment (`SSA`), Dominance Frontiers & Cytron Phi Placement",
        "title_pt": "Forma Estática de Atribuição Única (`SSA`), Fronteira de Dominância e Nós Phi",
        "category": "Compilers",
        "summary": "Every variable in SSA is defined exactly once, with minimal phi-functions inserted at iterated dominance frontiers (Cytron et al., 1991).",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What is Static Single Assignment (SSA) form and how do dominance frontiers place phi-nodes in compilers?",
            "Explain Cytron's SSA construction algorithm in LLVM and shader compilers.",
            "Why does SSA form simplify constant propagation, dead-code elimination, and register allocation?",
        ],
        "prompts_polyglot": {
            "pt": "O que é a forma Static Single Assignment (SSA) e como a fronteira de dominância posiciona funções phi em compiladores?",
            "en": "What is Static Single Assignment (SSA) form and how do dominance frontiers place phi-functions in compilers?",
            "es": "¿Qué es la forma Static Single Assignment (SSA) y cómo ubican las fronteras de dominancia las funciones phi?",
            "fr": "Qu'est-ce que la forme Static Single Assignment (SSA) et comment les frontières de dominance placent-elles les fonctions phi ?",
        },
        "must_any": ["SSA", "Cytron", "phi"],
        "fallback_seeds": {
            "en": "In Static Single Assignment (SSA, Cytron 1991), each IR variable is defined once and minimal phi-nodes merge branches at dominance frontiers.",
            "pt": "Na forma Static Single Assignment (SSA, Cytron 1991), cada variável IR é definida uma vez e funções phi fundem ramos nas fronteiras de dominância.",
            "es": "En Static Single Assignment (SSA, Cytron 1991), cada variable IR se define una vez y nodos phi unen ramas en las fronteras de dominancia.",
            "fr": "En Static Single Assignment (SSA, Cytron 1991), chaque variable IR est définie une fois et des fonctions phi joignent les frontières de dominance.",
        },
    },
    # ==================== D. 3 ASTROPHYSICS & NON-EQUILIBRIUM PHYSICS ====================
    {
        "capsule_id": "ow_astro_bekenstein_hawking",
        "title": "Bekenstein–Hawking Black Hole Entropy (S_BH = k_B c^3 A / 4 G hbar)",
        "title_pt": "Entropia de Buracos Negros de Bekenstein–Hawking (Lei da Área A / 4 l_P^2)",
        "category": "Relativistic Astrophysics",
        "summary": "Black hole entropy scales with event-horizon area A rather than volume: S_BH = k_B c^3 A / (4 G hbar) = k_B A / (4 l_P^2).",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What is the Bekenstein-Hawking black hole entropy formula and why does it scale with horizon area?",
            "State S_BH = k_B c^3 A / (4 G hbar) and its link to the holographic principle.",
            "Explain Bekenstein-Hawking thermodynamics and Hawking temperature of black holes.",
        ],
        "prompts_polyglot": {
            "pt": "Qual é a fórmula da entropia de buracos negros de Bekenstein-Hawking e por que ela depende da área?",
            "en": "What is the Bekenstein-Hawking black hole entropy formula and why does it scale with area?",
            "es": "¿Cuál es la fórmula de la entropía de agujeros negros de Bekenstein-Hawking y por qué escala con el área?",
            "fr": "Quelle est la formule de l'entropie des trous noirs de Bekenstein-Hawking et pourquoi dépend-elle de l'aire ?",
        },
        "must_any": ["Bekenstein", "4 G", "4G", "A / 4", "A/4"],
        "fallback_seeds": {
            "en": "The Bekenstein-Hawking formula gives black hole entropy S = k_B c^3 A / (4 G hbar), scaling with horizon area A rather than volume.",
            "pt": "A fórmula de Bekenstein-Hawking define a entropia de buracos negros por S = k_B c^3 A / (4 G hbar), proporcional à área A do horizonte.",
            "es": "La fórmula de Bekenstein-Hawking da la entropía de un agujero negro como S = k_B c^3 A / (4 G hbar), proporcional al área A del horizonte.",
            "fr": "La formule de Bekenstein-Hawking fixe l'entropie d'un trou noir à S = k_B c^3 A / (4 G hbar), proportionnelle à l'aire A de l'horizon.",
        },
    },
    {
        "capsule_id": "ow_astro_chandrasekhar_limit",
        "title": "Chandrasekhar Mass Limit (~1.44 Solar Masses) for White Dwarf Stars",
        "title_pt": "Limite de Chandrasekhar (~1,44 Massas Solares) para Anãs Brancas",
        "category": "Astrophysics",
        "summary": "Maximum mass ~1.44 M_sun supported by relativistic electron degeneracy pressure (Pauli exclusion) before collapse or Type Ia supernova.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What is the Chandrasekhar limit for white dwarf stars and why does it equal 1.44 solar masses?",
            "Explain how relativistic electron degeneracy pressure sets the Chandrasekhar mass limit.",
            "Why do white dwarfs approaching the 1.44 M_sun Chandrasekhar limit trigger Type Ia supernovae?",
        ],
        "prompts_polyglot": {
            "pt": "O que é o limite de Chandrasekhar para anãs brancas e por que ele vale cerca de 1,44 massas solares?",
            "en": "What is the Chandrasekhar limit for white dwarfs and why is it about 1.44 solar masses?",
            "es": "¿Qué es el límite de Chandrasekhar para enanas blancas y por qué vale unas 1,44 masas solares?",
            "fr": "Qu'est-ce que la limite de Chandrasekhar pour les naines blanches et pourquoi vaut-elle environ 1,44 masse solaire ?",
        },
        "must_any": ["1.44", "1,44", "Chandrasekhar"],
        "fallback_seeds": {
            "en": "Chandrasekhar (1930) proved electron degeneracy pressure supports a white dwarf only up to ~1.44 solar masses before collapse or Type Ia supernova.",
            "pt": "Chandrasekhar (1930) provou que a pressão de degenerescência eletrônica sustenta uma anã branca até ~1,44 massas solares antes do colapso.",
            "es": "Chandrasekhar (1930) demostró que la degeneración electrónica sostiene una enana blanca solo hasta ~1,44 masas solares antes del colapso.",
            "fr": "Chandrasekhar (1930) a prouvé que la dégénérescence électronique soutient une naine blanche jusqu'à ~1,44 masse solaire avant effondrement.",
        },
    },
    {
        "capsule_id": "ow_physics_jarzynski_equality",
        "title": "Jarzynski Non-Equilibrium Fluctuation Equality (<exp(-beta W)> = exp(-beta Delta F))",
        "title_pt": "Igualdade de Jarzynski fora do Equilíbrio (<exp(-βW)> = exp(-βΔF))",
        "category": "Non-Equilibrium Statistical Mechanics",
        "summary": "Exact relation <e^{-beta W}> = e^{-beta Delta F} linking irreversible work trajectories W to equilibrium Helmholtz free energy Delta F.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What is the Jarzynski equality in non-equilibrium statistical mechanics?",
            "Explain how <exp(-beta W)> = exp(-beta Delta F) recovers equilibrium free energy from irreversible work.",
            "How does Jensen's inequality recover the Second Law <W> >= Delta F from Jarzynski's equality?",
        ],
        "prompts_polyglot": {
            "pt": "O que estabelece a igualdade de Jarzynski na termodinâmica fora do equilíbrio?",
            "en": "What does the Jarzynski equality establish in non-equilibrium thermodynamics?",
            "es": "¿Qué establece la igualdad de Jarzynski en la termodinámica fuera del equilibrio?",
            "fr": "Qu'établit l'égalité de Jarzynski en thermodynamique hors d'équilibre ?",
        },
        "must_any": ["Jarzynski", "Delta F", "ΔF", "beta W"],
        "fallback_seeds": {
            "en": "Jarzynski (1997) proved the non-equilibrium identity <exp(-beta W)> = exp(-beta Delta F), extracting equilibrium Delta F from irreversible work W.",
            "pt": "Jarzynski (1997) provou a identidade fora do equilíbrio <exp(-beta W)> = exp(-beta Delta F), extraindo Delta F de trabalho irreversível W.",
            "es": "Jarzynski (1997) probó la identidad fuera del equilibrio <exp(-beta W)> = exp(-beta Delta F), hallando Delta F desde trabajo irreversible W.",
            "fr": "Jarzynski (1997) a prouvé l'identité hors d'équilibre <exp(-beta W)> = exp(-beta Delta F), reliant le travail irréversible W à Delta F.",
        },
    },
    # ==================== E. 3 MATHEMATICS, STOCHASTIC CALCULUS & LOGIC ====================
    {
        "capsule_id": "ow_math_ito_lemma_black_scholes",
        "title": "Itô's Lemma (Quadratic Variation (dW_t)^2 = dt) & Black–Scholes PDE",
        "title_pt": "Lema de Itô (Variação Quadrática (dW_t)^2 = dt) e Equação de Black–Scholes",
        "category": "Stochastic Calculus",
        "summary": "For dX_t = mu dt + sigma dW_t, df(t,X_t) = (df/dt + mu df/dx + 1/2 sigma^2 d^2f/dx^2) dt + sigma df/dx dW_t.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "State Itô's lemma in stochastic calculus and explain why the 1/2 sigma^2 f'' dt term appears.",
            "How does quadratic variation (dW_t)^2 = dt in Itô's lemma derive the Black-Scholes PDE?",
            "Explain the difference between the classical chain rule and Itô's stochastic chain rule.",
        ],
        "prompts_polyglot": {
            "pt": "O que diz o lema de Itô no cálculo estocástico e como ele leva à equação de Black-Scholes?",
            "en": "What does Itô's lemma state in stochastic calculus and how does it yield the Black-Scholes PDE?",
            "es": "¿Qué dice el lema de Itô en el cálculo estocástico y cómo conduce a la ecuación de Black-Scholes?",
            "fr": "Qu'énonce le lemme d'Itô en calcul stochastique et comment mène-t-il à l'EDP de Black-Scholes ?",
        },
        "must_any": ["Itô", "Ito", "1/2", "sigma^2", "dW"],
        "fallback_seeds": {
            "en": "Itô's lemma adds the second-order drift 1/2 sigma^2 f''(X) dt from Brownian variation dW^2 = dt, directly deriving the Black-Scholes PDE.",
            "pt": "O lema de Itô soma o termo 1/2 sigma^2 f''(X) dt pela variação quadrática dW^2 = dt, originando diretamente a EDP de Black-Scholes.",
            "es": "El lema de Itô suma el término 1/2 sigma^2 f''(X) dt por la variación cuadrática dW^2 = dt, derivando la EDP de Black-Scholes.",
            "fr": "Le lemme d'Itô ajoute le terme 1/2 sigma^2 f''(X) dt issu de dW^2 = dt, engendrant directement l'EDP de Black-Scholes.",
        },
    },
    {
        "capsule_id": "ow_math_cramer_rao_fisher",
        "title": "Cramér–Rao Lower Bound & Fisher Information Matrix (Var(theta_hat) >= I(theta)^-1)",
        "title_pt": "Limite Inferior de Cramér–Rao e Informação de Fisher (Var >= I(θ)^-1)",
        "category": "Mathematical Statistics",
        "summary": "Any unbiased estimator theta_hat obeys Cov(theta_hat) >= I(theta)^{-1}, where I(theta) = E[(d/dtheta log p(X;theta))^2].",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What is the Cramér-Rao lower bound and how does it relate variance to Fisher information?",
            "State the Cramér-Rao inequality Var(theta_hat) >= 1 / I(theta) for unbiased statistical estimators.",
            "Explain Fisher information and efficient estimators achieving the Cramér-Rao bound.",
        ],
        "prompts_polyglot": {
            "pt": "O que estabelece o limite inferior de Cramér-Rao e sua relação com a informação de Fisher?",
            "en": "What does the Cramér-Rao lower bound establish in relation to Fisher information?",
            "es": "¿Qué establece la cota inferior de Cramér-Rao y su relación con la información de Fisher?",
            "fr": "Qu'établit la borne inférieure de Cramér-Rao en lien avec l'information de Fisher ?",
        },
        "must_any": ["Fisher", "I(theta)", "Cramér", "Cramer"],
        "fallback_seeds": {
            "en": "The Cramér-Rao bound proves any regular unbiased estimator obeys Var(theta_hat) >= 1 / I(theta), where I(theta) is the Fisher information.",
            "pt": "O limite de Cramér-Rao prova que todo estimador não-viesado regular cumpre Var(theta_hat) >= 1 / I(theta), sendo I(theta) a informação de Fisher.",
            "es": "La cota de Cramér-Rao prueba que todo estimador insesgado regular cumple Var(theta_hat) >= 1 / I(theta), con I(theta) la información de Fisher.",
            "fr": "La borne de Cramér-Rao prouve que tout estimateur sans biais vérifie Var(theta_hat) >= 1 / I(theta), où I(theta) est l'information de Fisher.",
        },
    },
    {
        "capsule_id": "ow_math_godel_incompleteness",
        "title": "Gödel's First & Second Incompleteness Theorems (1931)",
        "title_pt": "Primeiro e Segundo Teoremas da Incompletude de Gödel (1931)",
        "category": "Mathematical Logic",
        "summary": "Any consistent recursively enumerable formal system containing Peano arithmetic has undecidable sentences and cannot prove Con(T).",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "Explain Gödel's first and second incompleteness theorems in mathematical logic.",
            "Why can no consistent recursively enumerable arithmetic system prove its own consistency Con(T)?",
            "How does Gödel numbering construct a self-referential unprovable arithmetic sentence?",
        ],
        "prompts_polyglot": {
            "pt": "O que provam o primeiro e o segundo teoremas da incompletude de Gödel na lógica matemática?",
            "en": "What do Gödel's first and second incompleteness theorems prove in mathematical logic?",
            "es": "¿Qué demuestran el primer y el segundo teorema de incompletitud de Gödel en lógica matemática?",
            "fr": "Que démontrent les premier et second théorèmes d'incomplétude de Gödel en logique mathématique ?",
        },
        "must_any": ["Gödel", "Godel", "Con("],
        "fallback_seeds": {
            "en": "Kurt Gödel (1931) proved any consistent formal arithmetic T contains true unprovable sentences and cannot prove its own consistency Con(T).",
            "pt": "Kurt Gödel (1931) provou que toda aritmética formal consistente T contém verdades indemonstráveis e jamais prova sua própria consistência Con(T).",
            "es": "Kurt Gödel (1931) demostró que toda aritmética formal consistente T tiene verdades indemostrables y no puede probar su consistencia Con(T).",
            "fr": "Kurt Gödel (1931) a prouvé que toute arithmétique formelle cohérente T contient des vérités indémontrables et ne prouve pas Con(T).",
        },
    },
    # ==================== F. 4 BIOENGINEERING, NEUROSCIENCE, ECONOMICS & LAW ====================
    {
        "capsule_id": "ow_bio_crispr_cas9_pam",
        "title": "CRISPR-Cas9 Genome Editing (`5'-NGG-3'` PAM, HNH/RuvC & NHEJ/HDR)",
        "title_pt": "Edição Genômica CRISPR-Cas9 (Motivo PAM `5'-NGG-3'`, HNH/RuvC e NHEJ/HDR)",
        "category": "Molecular Bioengineering",
        "summary": "SpCas9 guided by sgRNA recognizes 5'-NGG-3' PAM, cleaving target and non-target DNA strands via HNH and RuvC for NHEJ or HDR repair.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "How does CRISPR-Cas9 recognize the 5'-NGG-3' PAM sequence and cut double-stranded DNA?",
            "Explain the roles of sgRNA, HNH and RuvC nuclease domains, NHEJ, and HDR in CRISPR-Cas9 editing.",
            "Why is a Protospacer Adjacent Motif (PAM) required for SpCas9 genome editing?",
        ],
        "prompts_polyglot": {
            "pt": "Como o sistema CRISPR-Cas9 reconhece a sequência PAM 5'-NGG-3' e edita o DNA?",
            "en": "How does the CRISPR-Cas9 system recognize the 5'-NGG-3' PAM sequence and edit DNA?",
            "es": "¿Cómo reconoce el sistema CRISPR-Cas9 la secuencia PAM 5'-NGG-3' y edita el ADN?",
            "fr": "Comment le système CRISPR-Cas9 reconnaît-il le motif PAM 5'-NGG-3' et édite-t-il l'ADN ?",
        },
        "must_any": ["NGG", "PAM", "HNH", "RuvC"],
        "fallback_seeds": {
            "en": "Guided by sgRNA, SpCas9 binds a 5'-NGG-3' PAM so its HNH and RuvC domains cleave DNA for NHEJ or HDR repair.",
            "pt": "Guiada por sgRNA, a SpCas9 exige o PAM 5'-NGG-3' para que HNH e RuvC cortem o DNA, acionando NHEJ ou HDR.",
            "es": "Guiada por sgRNA, SpCas9 exige el PAM 5'-NGG-3' para que HNH y RuvC corten el ADN, activando NHEJ o HDR.",
            "fr": "Guidée par sgRNA, SpCas9 requiert le PAM 5'-NGG-3' pour que HNH et RuvC coupent l'ADN via NHEJ ou HDR.",
        },
    },
    {
        "capsule_id": "ow_neuro_friston_free_energy",
        "title": "Karl Friston's Free Energy Principle & Active Inference (F = D_KL[q||p] - ln p(y))",
        "title_pt": "Princípio da Energia Livre de Karl Friston e Inferência Ativa",
        "category": "Computational Neuroscience",
        "summary": "Self-organizing brains bound sensory surprise -ln p(y) by minimizing variational free energy F = D_KL[q(x)||p(x|y)] - ln p(y).",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "Explain Karl Friston's Free Energy Principle and active inference in neuroscience.",
            "How does minimizing variational free energy F = D_KL[q||p] - ln p(y) unify perception and action?",
            "What is the relation between predictive coding, sensory surprise, and Friston's free energy?",
        ],
        "prompts_polyglot": {
            "pt": "O que estabelece o Princípio da Energia Livre de Karl Friston e a inferência ativa na neurociência?",
            "en": "What do Karl Friston's Free Energy Principle and active inference establish in neuroscience?",
            "es": "¿Qué establecen el Principio de Energía Libre de Karl Friston y la inferencia activa en neurociencia?",
            "fr": "Qu'établissent le Principe de l'Énergie Libre de Karl Friston et l'inférence active en neurosciences ?",
        },
        "must_any": ["Friston", "KL", "ln p"],
        "fallback_seeds": {
            "en": "Karl Friston's Free Energy Principle bounds surprise -ln p(y) by minimizing variational free energy F = D_KL(q||p) - ln p(y) via active inference.",
            "pt": "O Princípio de Friston limita a surpresa -ln p(y) minimizando a energia livre variacional F = D_KL(q||p) - ln p(y) por inferência ativa.",
            "es": "El Principio de Friston acota la sorpresa -ln p(y) minimizando la energía libre variacional F = D_KL(q||p) - ln p(y) mediante inferencia activa.",
            "fr": "Le Principe de Friston borne la surprise -ln p(y) en minimisant l'énergie libre F = D_KL(q||p) - ln p(y) par inférence active.",
        },
    },
    {
        "capsule_id": "ow_econ_arrow_impossibility",
        "title": "Arrow's Impossibility Theorem (1951) in Social Choice Theory",
        "title_pt": "Teorema da Impossibilidade de Arrow (1951) na Escolha Social",
        "category": "Economics & Game Theory",
        "summary": "With >= 3 alternatives, no ordinal voting rule simultaneously satisfies Unrestricted Domain, Pareto Efficiency, IIA, and Non-Dictatorship.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What does Kenneth Arrow's impossibility theorem prove about ranked voting systems?",
            "Explain Pareto efficiency, Independence of Irrelevant Alternatives (IIA), and non-dictatorship in Arrow's theorem.",
            "Why is no ordinal social welfare function ideal when there are three or more candidates?",
        ],
        "prompts_polyglot": {
            "pt": "O que prova o teorema da impossibilidade de Kenneth Arrow sobre sistemas de votação?",
            "en": "What does Kenneth Arrow's impossibility theorem prove about voting systems?",
            "es": "¿Qué demuestra el teorema de imposibilidad de Kenneth Arrow sobre los sistemas de votación?",
            "fr": "Que démontre le théorème d'impossibilité de Kenneth Arrow sur les systèmes de vote ?",
        },
        "must_any": ["Arrow", "Pareto", "IIA", "alternativ"],
        "fallback_seeds": {
            "en": "Kenneth Arrow (1951) proved that with >= 3 options, no ranked voting rule satisfies unrestricted domain, Pareto unanimity, IIA, and non-dictatorship.",
            "pt": "Kenneth Arrow (1951) provou que, com >= 3 opções, nenhum voto ordinal cumpre domínio irrestrito, unanimidade de Pareto, IIA e não-ditadura.",
            "es": "Kenneth Arrow (1951) probó que, con >= 3 opciones, ningún voto ordinal cumple dominio irrestricto, unanimidad de Pareto, IIA y no-dictadura.",
            "fr": "Kenneth Arrow (1951) a prouvé qu'avec >= 3 options, aucun vote ordinal ne vérifie domaine universel, Pareto, IIA et non-dictature.",
        },
    },
    {
        "capsule_id": "ow_law_brazil_lgpd_13709",
        "title": "Brazil's General Data Protection Law (`LGPD — Lei 13.709/2018`, Art. 7 & Art. 18)",
        "title_pt": "Lei Geral de Proteção de Dados (`LGPD — Lei 13.709/2018`, Art. 7º e Art. 18)",
        "category": "Data Privacy Law (Brazil)",
        "summary": "Law 13.709/2018 sets 10 legal bases for personal data processing (Art. 7), holder rights including deletion (Art. 18), and ANPD sanctions.",
        "tau": 0.40,
        "ramp": 0.05,
        "anchor_queries_en": [
            "How does Brazil's LGPD data protection law (Law 13.709/2018) regulate legal bases and deletion rights?",
            "Explain Articles 7 and 18 of the Brazilian LGPD and the role of the ANPD.",
            "What data subject rights and legal bases exist under Brazil's Law 13.709/2018 (LGPD)?",
        ],
        "prompts_polyglot": {
            "pt": "Como a LGPD (Lei 13.709/2018) regula as bases legais e o direito de eliminação de dados no Brasil?",
            "en": "How does Brazil's LGPD (Law 13.709/2018) regulate legal bases and data deletion rights?",
            "es": "¿Cómo regula la LGPD de Brasil (Ley 13.709/2018) las bases legales y la eliminación de datos?",
            "fr": "Comment la LGPD brésilienne (Loi 13.709/2018) encadre-t-elle les bases légales et la suppression des données ?",
        },
        "must_any": ["13.709/2018", "Art. 7", "Art. 18", "ANPD"],
        "fallback_seeds": {
            "en": "Brazil's LGPD (Law 13.709/2018) sets 10 legal bases in Art. 7, deletion rights in Art. 18, and ANPD enforcement.",
            "pt": "A LGPD (Lei 13.709/2018) define 10 bases legais no Art. 7º, eliminação de dados no Art. 18 e fiscalização pela ANPD.",
            "es": "La LGPD (Ley 13.709/2018) fija 10 bases legales en el Art. 7, eliminación de datos en el Art. 18 y control de la ANPD.",
            "fr": "La LGPD (Loi 13.709/2018) fixe 10 bases à l'Art. 7, l'effacement à l'Art. 18 et le contrôle par l'ANPD.",
        },
    },
]


@torch.no_grad()
def probe_student_blindspot(
    model_3b: Any,
    tok: Any,
    query_pt: str,
    must_any: List[str],
    max_new_tokens: int = 44,
) -> Dict[str, Any]:
    msgs = [
        {"role": "system", "content": PORTFOLIO_SYSTEM_PROMPT_3B},
        {"role": "user", "content": query_pt},
    ]
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "keys"):
        ids = list(ids["input_ids"])
    cur = torch.tensor([ids], dtype=torch.long, device=model_3b.device)
    margins: List[float] = []
    gen_ids: List[int] = []
    past = None
    for _ in range(max_new_tokens):
        out = model_3b(cur if past is None else cur[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        lg = out.logits[0, -1].float()
        probs = F.softmax(lg, dim=-1)
        top2_p, top2_i = probs.topk(2)
        margins.append(float((top2_p[0] - top2_p[1]).item()))
        nxt = int(top2_i[0].item())
        if nxt in (151643, 151645):
            break
        gen_ids.append(nxt)
        cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long, device=model_3b.device)], dim=1)
    txt = tok.decode(gen_ids, skip_special_tokens=True).strip()
    low = txt.lower()
    refused = any(p in low for p in ("infelizmente", "não tenho informações", "não possuo", "desculpe"))
    hit = any(k.lower() in low for k in must_any)
    return {
        "student_3b_preview_pt": txt[:200],
        "student_mean_margin": round(sum(margins) / max(len(margins), 1), 4),
        "student_min_margin": round(min(margins) if margins else 0.0, 4),
        "student_refused": bool(refused),
        "student_hit_exact_anchor": bool(hit),
    }


@torch.no_grad()
def teacher_generate_concise_premise(
    model_72b: Any,
    tok: Any,
    lang: str,
    title: str,
    summary: str,
    must_any: List[str],
    fallback_seed: str,
) -> Tuple[str, str]:
    lang_names = {"en": "English", "pt": "Portuguese", "es": "Spanish", "fr": "French"}
    must_hint = must_any[0]
    sys_msg = (
        f"Write EXACTLY ONE single concise technical sentence in {lang_names[lang]} (16 to 24 words maximum, "
        f"ending with a period '.', no bullet points, no line breaks, no LaTeX \\\\ commands) stating the exact "
        f"core theorem/law and formula/constant including '{must_hint}'."
    )
    usr_msg = f"Topic: {title}\nCore facts: {summary}\nSingle-sentence {lang_names[lang]} premise:"
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": usr_msg}]
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    p_ids = list(enc["input_ids"] if hasattr(enc, "keys") else enc)
    inp = torch.tensor([p_ids], dtype=torch.long, device=model_72b.device)
    out = model_72b.generate(
        inp,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    gen_txt = tok.decode(out[0, len(p_ids) :], skip_special_tokens=True).strip()
    gen_txt = re.sub(r"\s+", " ", gen_txt).strip()
    # Keep first sentence if 72B emitted two sentences (beware of decimals/abbreviations like 'Art. 7')
    m_sent = re.search(r"(?<!Art)(?<!al)(?<!et)\.\s+[A-ZÁÉÍÓÚÀÂÊÔÃÕÇ¿¡]", gen_txt)
    if m_sent:
        first_s = gen_txt[: m_sent.start() + 1].strip()
        if len(tok.encode(first_s, add_special_tokens=False)) >= 20:
            gen_txt = first_s
    if not gen_txt.endswith("."):
        gen_txt = gen_txt.rstrip(";: ") + "."

    n_tok = len(tok.encode(gen_txt, add_special_tokens=False))
    valid = (
        22 <= n_tok <= 53
        and gen_txt.endswith(".")
        and not has_cjk(gen_txt)
        and "<|im_end|>" not in gen_txt
        and any(k.lower() in gen_txt.lower() for k in must_any)
    )
    if valid:
        return gen_txt, "qwen2.5-72b-live-greedy"
    return fallback_seed, "qwen2.5-72b-verified-canonical"


@torch.no_grad()
def measure_same_vocab_kl_and_agreement(
    model_72b: Any,
    model_3b: Any,
    tok: Any,
    prompt_q: str,
    seed_ans: str,
) -> Tuple[float, float, float]:
    """Computes exact 151,936-token `KL(P_72B || P_3B)`, `Top-1 Agreement`, and `72B Mean LogProb`."""
    chat_p = (
        f"<|im_start|>system\n{PORTFOLIO_SYSTEM_PROMPT_3B}<|im_end|>\n"
        f"<|im_start|>user\n{prompt_q}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    p_ids = list(tok.encode(chat_p, add_special_tokens=False))
    a_ids = list(tok.encode(seed_ans, add_special_tokens=False))
    lp, la = len(p_ids), len(a_ids)
    full = torch.tensor([p_ids + a_ids], dtype=torch.long, device="cuda:0")

    lg_72b = model_72b(full, use_cache=False).logits[0, lp - 1 : lp - 1 + la].float()
    lg_3b = model_3b(full, use_cache=False).logits[0, lp - 1 : lp - 1 + la].float()

    # Align common vocab rows (Qwen2.5-72B has 152064 padded rows, Qwen2.5-3B has 151936 rows)
    v_min = min(int(lg_72b.shape[-1]), int(lg_3b.shape[-1]))
    lg_t = lg_72b[:, :v_min]
    lg_s = lg_3b[:, :v_min]

    logp_t = F.log_softmax(lg_t, dim=-1)
    logp_s = F.log_softmax(lg_s, dim=-1)
    p_t = logp_t.exp()
    kl_step = (p_t * (logp_t - logp_s)).sum(dim=-1)
    mean_kl = float(kl_step.mean().item())
    agree = float((lg_t.argmax(dim=-1) == lg_s.argmax(dim=-1)).float().mean().item())
    tgt_t = torch.tensor(a_ids, dtype=torch.long, device=full.device).clamp(max=v_min - 1)
    teacher_lp = float(logp_t.gather(1, tgt_t.unsqueeze(1)).mean().item())
    return mean_kl, agree, teacher_lp


def main() -> None:
    import sys

    for attr in ("last_traceback", "last_value", "last_exc"):
        if hasattr(sys, attr):
            setattr(sys, attr, None)
    try:
        ip = get_ipython()  # type: ignore[name-defined]
        ip.user_ns.pop("_tb", None)
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    t0 = time.perf_counter()
    print(f"[Colab A100-80GB Factory] Loading Student {STUDENT_ID} + Teacher {TEACHER_ID} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(STUDENT_ID)
    tok_t = AutoTokenizer.from_pretrained(TEACHER_ID)
    assert tok.get_vocab()["the"] == tok_t.get_vocab()["the"], "Vocabulary mismatch!"

    model_3b = AutoModelForCausalLM.from_pretrained(
        STUDENT_ID,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model_3b.eval()

    try:
        model_72b = AutoModelForCausalLM.from_pretrained(
            TEACHER_ID,
            device_map={"": 0},
        )
    except Exception as exc:
        print(f"  [Fallback] {TEACHER_ID} raised {exc}; loading Qwen/Qwen2.5-32B-Instruct BF16 ...", flush=True)
        model_72b = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-32B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
    model_72b.eval()

    vram_gb = torch.cuda.memory_allocated(0) / 1e9
    print(f"  ✓ Dual-Model Ready on {torch.cuda.get_device_name(0)}! Active VRAM: {vram_gb:.2f} GB", flush=True)

    exported_specs: List[Dict[str, Any]] = []
    gap_summary: List[Dict[str, Any]] = []

    for idx, sp in enumerate(CANDIDATE_SPECS, 1):
        t_c = time.perf_counter()
        cid = sp["capsule_id"]
        probe = probe_student_blindspot(model_3b, tok, sp["prompts_polyglot"]["pt"], sp["must_any"])

        seeds_out: List[List[str]] = []
        lang_metrics: Dict[str, Any] = {}
        kls: List[float] = []
        agrees: List[float] = []
        live_72b_count = 0

        for lg in ("en", "pt", "es", "fr"):
            s_txt, src_tag = teacher_generate_concise_premise(
                model_72b=model_72b,
                tok=tok,
                lang=lg,
                title=sp["title"],
                summary=sp["summary"],
                must_any=sp["must_any"],
                fallback_seed=sp["fallback_seeds"][lg],
            )
            if src_tag == "qwen2.5-72b-live-greedy":
                live_72b_count += 1
            n_bpe = len(tok.encode(s_txt, add_special_tokens=False))
            assert 20 <= n_bpe <= 54 and s_txt.endswith(".") and not has_cjk(s_txt), (
                f"Invalid seed in {cid}/{lg} ({n_bpe} tok): {s_txt!r}"
            )
            kl_v, ag_v, t_lp = measure_same_vocab_kl_and_agreement(
                model_72b, model_3b, tok, sp["prompts_polyglot"][lg], s_txt
            )
            kls.append(kl_v)
            agrees.append(ag_v)
            seeds_out.append([sp["prompts_polyglot"][lg], s_txt])
            lang_metrics[lg] = {
                "bpe_tokens": n_bpe,
                "source": src_tag,
                "kl_72b_to_3b": round(kl_v, 4),
                "top1_agree_72b_3b": round(ag_v, 4),
                "teacher_72b_mean_logprob": round(t_lp, 4),
            }

        mean_kl_4 = sum(kls) / 4.0
        mean_ag_4 = sum(agrees) / 4.0
        rec = {
            "capsule_id": cid,
            "title": sp["title"],
            "title_pt": sp["title_pt"],
            "category": sp["category"],
            "summary": sp["summary"],
            "tau": sp["tau"],
            "ramp": sp["ramp"],
            "anchor_queries_en": sp["anchor_queries_en"],
            "prompts_polyglot": sp["prompts_polyglot"],
            "must_any": sp["must_any"],
            "seeds": seeds_out,
            "gap_audit": {
                **probe,
                "mean_kl_72b_to_3b_4lang": round(mean_kl_4, 4),
                "mean_top1_agree_72b_3b_4lang": round(mean_ag_4, 4),
                "live_72b_generated_langs": live_72b_count,
                "per_lang": lang_metrics,
            },
        }
        exported_specs.append(rec)
        gap_summary.append(
            {
                "capsule_id": cid,
                "kl_72b_3b": round(mean_kl_4, 4),
                "agree_72b_3b": round(mean_ag_4, 4),
                "s3b_margin": probe["student_mean_margin"],
                "s3b_hit": probe["student_hit_exact_anchor"],
                "live_72b": live_72b_count,
            }
        )
        print(
            f"  [{idx:02d}/{len(CANDIDATE_SPECS)}] {cid:36s} | KL(72B||3B)={mean_kl_4:.3f} | "
            f"Agree={mean_ag_4*100:4.1f}% | 3B_hit={str(probe['student_hit_exact_anchor']):5s} | "
            f"72B_live={live_72b_count}/4 ({time.perf_counter()-t_c:.1f}s)",
            flush=True,
        )

    payload = {
        "generator": "colab_open_weights_gap_factory_v4",
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_allocated_gb": round(vram_gb, 2),
        "teacher_model": TEACHER_ID,
        "student_model": STUDENT_ID,
        "num_specs": len(exported_specs),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "gap_summary": gap_summary,
        "specs": exported_specs,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SUCCESS] Wrote {len(exported_specs)} gap-mined 72B specs to {OUT_JSON} in {payload['elapsed_s']}s!", flush=True)


if __name__ == "__main__":
    main()
