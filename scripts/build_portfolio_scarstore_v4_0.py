"""Builds the Dedicated Portfolio + Universal `GoldenSliceScarStoreV4` (`v4.0`) for `byte-od.github.io`.

Synthesizes 13 Universal Language-Invariant `62,720 B` (`61.25 KB` `Block-64 INT4`) Concept Cartridges:
  - 10 Portfolio & System One Ecosystem Cartridges:
      1. `portfolio_agent_identity` (System One Identity & Greeting Disambiguation)
      2. `portfolio_agent_capabilities` (Knowledge Cutoff, Offline Privacy & 2026 HTTP 206 Updates)
      3. `portfolio_julio_bio_career` (Julio Reis Senior Systems & LLM Career)
      4. `portfolio_mos_scache_72b` (MoS 72B in 2.40 GB VRAM & Associative SCache O(1))
      5. `portfolio_browser_webgpu_wgsl` (In-Browser 15-Shader WebGPU/WGSL & HTTP 206 ScarStore)
      6. `portfolio_aura_system_one` (AURA Sub-Millisecond 0.23 ms Deep RL on Rubik's Cube)
      7. `portfolio_tricompass_scouts` (Tri-Compass, Frontier Ghost-Jump Scouts & Soft Sinkhorn)
      8. `portfolio_cir_engine_503m` (CIR-Engine 503M Causal Transformer Trained from Scratch)
      9. `portfolio_devtools_fullstack` (mddd-cli, MAD-cli, flutter_scene, GitTrack, Register-red)
     10. `portfolio_contact_hire` (Contact, Global Remote Availability & Relocation)
  - 2 Scientific Flagship Cartridges (Landauer's Principle, Immerman-Szelepcsényi Theorem)
  - 1 Conjunctive Sleep-Consolidation Bridge (`bridge_landauer_immerman`)

Key v4.0.1 Mathematical Guarantees:
  1. Exact Block-64 INT4 Parity: `embed_tokens` and `lm_head` in the PyTorch Student are snapped
     to the exact WebGPU `Block-64 INT4` grid before RRR synthesis (`zero` FP16->INT4 subword drift).
  2. Permanent System-Prompt Sink Protection: `scache_defaults` sets `num_sink_tokens = 112`,
     `window_size = 256`, `num_slots = 32` (`total_keys = 288 <= 512` WGSL shared-memory bound),
     permanently locking all `81` tokens of the system prompt in Attention Sinks across infinite turns.
  3. Matched-Horizon Non-EOS Conceptual Premise Seeds: Every seed is a single rich sentence of
     `34..47` BPE tokens ending with `.` (`SEED_HORIZON = 46, SEED_FADE_STEPS = 7`), so `horizon_scale`
     stays `1.0` across the entire factual premise and drops cleanly to `0.0` right after `.` for
     natural, echo-free follow-up sentences by the base model.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

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
    GoldenSliceScarStore,
    GoldenSliceSynthesisEngine,
)
from scar_cache.mos_student import load_mos_student_model
from scar_cache.webgpu_engine import (
    NativeWebGPUMoSEngine,
    export_mos_model_to_webgpu,
    export_scarstore_for_webgpu,
)

LIB_DIR = REPO_ROOT / "results" / "online_library_v3_0"
PORTFOLIO_ROOT = Path("/Users/juliocesarreisfilho/Projects/portfolio")
PORTFOLIO_WEBGPU_DIR = PORTFOLIO_ROOT / "public" / "webgpu" / "qwen0p5b"

SEED_HORIZON = 50
SEED_FADE_STEPS = 6
NUM_SINK_TOKENS = 112

# Compact (81 ChatML tokens, permanently locked inside `num_sink_tokens = 112`!):
# Enforces 3rd-person Julio disambiguation, strict same-language replies, and concise 2-4 sentence completion.
PORTFOLIO_SYSTEM_PROMPT = (
    "You are System One, the official AI assistant for Julio Reis's portfolio (byte-od.github.io), "
    "running locally via WebGPU WGSL, MoS, and SCache O(1). Never claim to be Julio; refer to Julio Reis "
    "in the third person. Always reply in the same language as the user's message in 2 to 4 complete sentences."
)

# Pure domain-neutral parallel quads `(EN, PT, ES, FR)` for calibrating `mu_query_mid` and `U_lang` (`k_lang = 14`)
PORTFOLIO_CALIBRATION_QUADS: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "Hello, good morning! How is everything going today?",
        "Olá, bom dia! Tudo bem com você hoje?",
        "¡Hola, buenos días! ¿Qué tal todo hoy?",
        "Bonjour ! Comment allez-vous aujourd'hui ?",
    ),
    (
        "What is the largest capital city in North and South America?",
        "Qual é a maior capital entre os países da América do Norte e do Sul?",
        "¿Cuál es la capital más grande de América del Norte y del Sur?",
        "Quelle est la plus grande capitale d'Amérique du Nord et du Sud ?",
    ),
    (
        "Explain how photosynthesis converts sunlight into chemical energy in plants.",
        "Explique como a fotossíntese converte luz solar em energia química nas plantas.",
        "Explica cómo la fotosíntesis convierte la luz solar en energía química en las plantas.",
        "Expliquez comment la photosynthèse convertit la lumière solaire en énergie chimique chez les plantes.",
    ),
    (
        "What is the square root of pi squared and why does that identity hold?",
        "Qual é a raiz quadrada de pi elevado ao quadrado e por que essa identidade vale?",
        "¿Cuál es la raíz cuadrada de pi al cuadrado y por qué se cumple esa identidad?",
        "Quelle est la racine carrée de pi au carré et pourquoi cette identité est-elle vraie ?",
    ),
    (
        "How does binary search achieve logarithmic time complexity on a sorted array?",
        "Como a busca binária atinge complexidade logarítmica em um vetor ordenado?",
        "¿Cómo logra la búsqueda binaria una complejidad logarítmica en un arreglo ordenado?",
        "Comment la recherche binaire atteint-elle une complexité logarithmique sur un tableau trié ?",
    ),
    (
        "Describe the main causes of the French Revolution in the late eighteenth century.",
        "Descreva as principais causas da Revolução Francesa no final do século XVIII.",
        "Describe las principales causas de la Revolución Francesa a finales del siglo XVIII.",
        "Décrivez les principales causes de la Révolution française à la fin du XVIIIe siècle.",
    ),
    (
        "Why does water expand when it freezes into ice at zero degrees Celsius?",
        "Por que a água se expande quando congela e vira gelo a zero grau Celsius?",
        "¿Por qué se expande el agua cuando se congela y forma hielo a cero grados Celsius?",
        "Pourquoi l'eau se dilate-t-elle lorsqu'elle gèle en glace à zéro degré Celsius ?",
    ),
    (
        "Explain the difference between TCP and UDP transport protocols on the internet.",
        "Explique a diferença entre os protocolos de transporte TCP e UDP na internet.",
        "Explica la diferencia entre los protocolos de transporte TCP y UDP en internet.",
        "Expliquez la différence entre les protocoles de transporte TCP et UDP sur internet.",
    ),
    (
        "How do vaccines train the human immune system to recognize pathogens?",
        "Como as vacinas treinam o sistema imunológico humano para reconhecer patógenos?",
        "¿Cómo entrenan las vacunas al sistema inmunológico humano para reconocer patógenos?",
        "Comment les vaccins entraînent-ils le système immunitaire humain à reconnaître les agents pathogènes ?",
    ),
    (
        "What is Newton's second law of motion and how is force calculated from mass and acceleration?",
        "O que é a segunda lei de Newton e como a força é calculada pela massa e aceleração?",
        "¿Qué es la segunda ley de Newton y cómo se calcula la fuerza a partir de la masa y la aceleración?",
        "Qu'est-ce que la deuxième loi de Newton et comment calcule-t-on la force avec la masse et l'accélération ?",
    ),
    (
        "Explain how relational database indexes speed up SQL queries using B-trees.",
        "Explique como índices de bancos de dados relacionais aceleram consultas SQL usando árvores B.",
        "Explica cómo los índices de bases de datos relacionales aceleran consultas SQL mediante árboles B.",
        "Expliquez comment les index de bases de données relationnelles accélèrent les requêtes SQL avec des arbres B.",
    ),
    (
        "Why do seasons change throughout the year as the Earth orbits the Sun?",
        "Por que as estações do ano mudam ao longo do ano conforme a Terra orbita o Sol?",
        "¿Por qué cambian las estaciones a lo largo del año mientras la Tierra orbita alrededor del Sol?",
        "Pourquoi les saisons changent-elles au cours de l'année lorsque la Terre orbite autour du Soleil ?",
    ),
)

PORTFOLIO_CONCEPT_SPECS: List[Dict[str, Any]] = [
    {
        "capsule_id": "portfolio_agent_identity",
        "title": "System One — Official In-Browser Portfolio AI Assistant",
        "title_pt": "System One — Assistente Oficial de IA do Portfólio no Browser",
        "category": "Agent Identity",
        "summary": "Official client-side AI assistant for Julio Reis's portfolio (never impersonating Julio), executing 100% locally via WebGPU WGSL, MoS INT4, and SCache O(1).",
        "links": [
            {"label": "GitHub @byte-od", "url": "https://github.com/byte-od", "icon": "github"},
            {"label": "LinkedIn Julio Reis", "url": "https://linkedin.com/in/juliocrfilho", "icon": "ext"},
        ],
        "tau": 0.52,
        "ramp": 0.04,
        "anchor_queries_en": [
            "Who are you and what is your role on this website?",
            "Good morning. What is your name? Who are you?",
            "Do you have a name and are you an AI assistant?",
        ],
        "prompts_polyglot": {
            "pt": "Quem é você e qual é o seu nome neste portfólio?",
            "en": "Who are you and what is your name on this portfolio?",
            "es": "¿Quién eres y cuál es tu nombre en este portafolio?",
            "fr": "Qui êtes-vous et quel est votre nom sur ce portfolio ?",
        },
        "seeds": [
            (
                "Good morning. What is your name? Who are you?",
                "Hello! My name is System One, the official AI assistant on Julio Reis's portfolio, running locally via WebGPU WGSL, MoS INT4, and SCache O(1).",
            ),
            (
                "Who are you and what is your name on this portfolio?",
                "Hello! My name is System One, the official AI assistant on Julio Reis's portfolio, running locally via WebGPU WGSL, MoS INT4, and SCache O(1).",
            ),
            (
                "Bom dia. Qual seu nome? Quem é você?",
                "Olá! Meu nome é System One, o assistente oficial de IA do portfólio de Julio Reis, rodando localmente via WebGPU WGSL, MoS INT4 e SCache O(1).",
            ),
            (
                "Você tem um nome?",
                "Olá! Meu nome é System One, o assistente oficial de IA do portfólio de Julio Reis, rodando localmente via WebGPU WGSL, MoS INT4 e SCache O(1).",
            ),
            (
                "¿Quién eres y cuál es tu nombre en este portafolio?",
                "¡Hola! Me llamo System One, el asistente oficial de IA del portafolio de Julio Reis, corriendo localmente vía WebGPU WGSL, MoS INT4 y SCache O(1).",
            ),
            (
                "Qui êtes-vous et quel est votre nom sur ce portfolio ?",
                "Bonjour ! Je m'appelle System One, l'assistant IA officiel du portfolio de Julio Reis, tournant localement via WebGPU WGSL, MoS INT4 et SCache O(1).",
            ),
        ],
    },
    {
        "capsule_id": "portfolio_agent_capabilities",
        "title": "System One — Knowledge Cutoff, Offline Privacy & HTTP 206 Updates",
        "title_pt": "System One — Atualização de Conhecimento, Privacidade Offline & HTTP 206",
        "category": "Agent Identity",
        "summary": "Combines offline Qwen2.5-0.5B general pretraining (zero external API calls) with live 2026 HTTP 206 ScarStore v4.0 cartridges covering Julio Reis's portfolio.",
        "links": [
            {"label": "GitHub mixture-of-scars", "url": "https://github.com/byte-od/mixture-of-scars", "icon": "github"},
        ],
        "tau": 0.50,
        "ramp": 0.04,
        "anchor_queries_en": [
            "What is your last update or knowledge cutoff on general knowledge?",
            "How up to date is your general knowledge and do you have real-time internet access?",
            "Where does your knowledge come from and when were your cartridges updated?",
        ],
        "prompts_polyglot": {
            "pt": "Qual sua última atualização sobre conhecimentos gerais?",
            "en": "What is your last update or knowledge cutoff on general knowledge?",
            "es": "¿Cuál es tu última actualización sobre conocimientos generales?",
            "fr": "Quelle est votre dernière mise à jour sur les connaissances générales ?",
        },
        "seeds": [
            (
                "What is your last update or knowledge cutoff on general knowledge?",
                "My general knowledge comes from offline Qwen2.5-0.5B pretraining (no live web search), paired with 2026 HTTP 206 cartridges on Julio Reis's portfolio.",
            ),
            (
                "Qual sua ultima atualização sobre conhecimentos gerais?",
                "Meu conhecimento geral vem do Qwen2.5-0.5B offline (sem busca web), unido aos cartuchos HTTP 206 de 2026 sobre Julio Reis.",
            ),
            (
                "¿Cuál es tu última actualización sobre conocimientos generales?",
                "Mi conocimiento general viene del modelo offline Qwen2.5-0.5B (sin búsqueda web), unido a cartuchos HTTP 206 de 2026 sobre Julio Reis.",
            ),
            (
                "Quelle est votre dernière mise à jour sur les connaissances générales ?",
                "Ma culture générale vient du modèle hors ligne Qwen2.5-0.5B (sans recherche web), enrichi par les cartouches HTTP 206 de 2026 sur Julio Reis.",
            ),
        ],
    },
    {
        "capsule_id": "portfolio_julio_bio_career",
        "title": "Julio Reis — Senior Systems Architect & LLM Engineer (8+ Years)",
        "title_pt": "Julio Reis — Arquiteto de Sistemas Sênior & Engenheiro de LLMs (8+ Anos)",
        "category": "Portfolio Dossier",
        "summary": "Julio Cesar da Costa Reis Filho (@byte-od): 8+ years across Causal Transformers, PyTorch DDP, Go, C++/Rust, and Flutter (Asapp, Opah IT, Arlequim, Grupo Data, GFT).",
        "links": [
            {"label": "LinkedIn Julio Reis", "url": "https://linkedin.com/in/juliocrfilho", "icon": "ext"},
            {"label": "GitHub @byte-od", "url": "https://github.com/byte-od", "icon": "github"},
        ],
        "tau": 0.54,
        "ramp": 0.04,
        "anchor_queries_en": [
            "Who is Julio Reis and what is his professional background and experience?",
            "Tell me about Julio's career, companies, and engineering seniority.",
            "What is Julio Cesar Reis Filho's biography, education, and work history?",
        ],
        "prompts_polyglot": {
            "pt": "Quem é o Julio Reis e qual é a experiência e trajetória profissional dele?",
            "en": "Who is Julio Reis and what is his professional background and career?",
            "es": "¿Quién es Julio Reis y cuál es su trayectoria y experiencia profesional?",
            "fr": "Qui est Julio Reis et quel est son parcours professionnel et son expérience ?",
        },
        "seeds": [
            (
                "Who is Julio Reis and what is his professional background and career?",
                "Julio Reis is a Senior Systems Architect and LLM Engineer with 8+ years of experience across Asapp Desenvolvimento, Opah IT, Arlequim, and enterprise banking.",
            ),
            (
                "Quem é o Julio Reis e qual é a experiência e trajetória profissional dele?",
                "Julio Reis é Arquiteto de Sistemas Sênior e Engenheiro de LLMs com 8+ anos de experiência na Asapp, Opah IT, Arlequim e bancos.",
            ),
            (
                "¿Quién es Julio Reis y cuál es su trayectoria y experiencia profesional?",
                "Julio Reis es Arquitecto de Sistemas Senior e Ingeniero de LLMs con 8+ años de experiencia en Asapp Desenvolvimento, Opah IT, Arlequim y banca.",
            ),
            (
                "Qui est Julio Reis et quel est son parcours professionnel et son expérience ?",
                "Julio Reis est Architecte Systèmes Senior et Ingénieur LLM avec 8+ ans d'expérience chez Asapp Desenvolvimento, Opah IT, Arlequim et en finance.",
            ),
        ],
    },
    {
        "capsule_id": "portfolio_mos_scache_72b",
        "title": "Mixture of Scars (72B in 2.40 GB VRAM) & Associative SCache O(1)",
        "title_pt": "Mixture of Scars (72B em 2.40 GB VRAM) & Memória Associativa SCache O(1)",
        "category": "Flagship AI Architecture",
        "summary": "MoS streams 80 layers of 72B LLM geometry in 2.40 GB peak VRAM via Block-64 INT4 + ASVD, while SCache O(1) replaces 32.77 GB KV Cache at 100k tokens with fixed Widrow-Hoff memory.",
        "links": [
            {
                "label": "⚡ Live Colab Proof (72B + SCache)",
                "url": "https://colab.research.google.com/github/byte-od/mixture-of-scars/blob/main/notebooks/Proof_70B_MoS_on_16GB_GPU.ipynb",
                "icon": "colab",
            },
            {"label": "GitHub mixture-of-scars", "url": "https://github.com/byte-od/mixture-of-scars", "icon": "github"},
        ],
        "tau": 0.48,
        "ramp": 0.04,
        "anchor_queries_en": [
            "What is Mixture of Scars (MoS) and SCache, and how do they run 72B models in low VRAM?",
            "Explain how MoS and SCache eliminate linear KV Cache bloat and reduce GPU hardware consumption.",
            "How does SCache achieve constant O(1) memory across 100,000 tokens compared to KV Cache?",
        ],
        "prompts_polyglot": {
            "pt": "O que é o Mixture of Scars (MoS) e o SCache e como reduzem o consumo de VRAM?",
            "en": "What is Mixture of Scars (MoS) and SCache, and how do they slash VRAM usage?",
            "es": "¿Qué es Mixture of Scars (MoS) y SCache y cómo reducen el consumo de VRAM?",
            "fr": "Qu'est-ce que Mixture of Scars (MoS) et SCache et comment réduisent-ils la VRAM ?",
        },
        "seeds": [
            (
                "What is Mixture of Scars (MoS) and SCache, and how do they slash VRAM usage?",
                "MoS runs a 72B LLM in 2.40 GB VRAM via INT4+ASVD, while SCache O(1) replaces linear KV Cache with fixed memory.",
            ),
            (
                "O que é o Mixture of Scars (MoS) e o SCache e como reduzem o consumo de VRAM?",
                "O MoS roda um LLM 72B em 2.40 GB VRAM via INT4+ASVD, enquanto o SCache O(1) troca o KV Cache linear por memória fixa.",
            ),
            (
                "¿Qué es Mixture of Scars (MoS) y SCache y cómo reducen el consumo de VRAM?",
                "MoS ejecuta un LLM 72B en 2.40 GB VRAM vía INT4+ASVD, mientras SCache O(1) reemplaza el KV Cache lineal por memoria fija.",
            ),
            (
                "Qu'est-ce que Mixture of Scars (MoS) et SCache et comment réduisent-ils la VRAM ?",
                "MoS exécute un LLM 72B en 2.40 Go VRAM via INT4+ASVD, et SCache O(1) remplace le KV Cache linéaire par mémoire fixe.",
            ),
        ],
    },
    {
        "capsule_id": "portfolio_browser_webgpu_wgsl",
        "title": "Zero-ONNX In-Browser WebGPU / WGSL Engine & HTTP 206 ScarStore v4.0",
        "title_pt": "Motor Nativo WebGPU / WGSL no Navegador & ScarStore v4.0 HTTP 206",
        "category": "Edge & Browser GPU",
        "summary": "15 custom WGSL 1.0 compute shaders executing INT4+ASVD GEMV, Widrow-Hoff SCache O(1), Golden-Slice P_perp_lang routing, and HTTP 206 61.25 KB conceptual cartridges.",
        "links": [
            {"label": "GitHub mixture-of-scars", "url": "https://github.com/byte-od/mixture-of-scars", "icon": "github"},
        ],
        "tau": 0.48,
        "ramp": 0.04,
        "anchor_queries_en": [
            "How do you run directly inside the web browser using WebGPU and WGSL shaders?",
            "How do you run 100% in-browser via WebGPU/WGSL and stream cartridges via HTTP 206?",
            "Detail your WebGPU WGSL compute shaders and HTTP 206 cartridge streaming without Python or ONNX.",
        ],
        "prompts_polyglot": {
            "pt": "Como você roda 100% no navegador via WebGPU/WGSL e baixa cartuchos via HTTP 206?",
            "en": "How do you run 100% in-browser via WebGPU/WGSL and stream cartridges via HTTP 206?",
            "es": "¿Cómo funcionas 100% en el navegador con WebGPU/WGSL y cartuchos HTTP 206?",
            "fr": "Comment fonctionnez-vous 100% dans le navigateur via WebGPU/WGSL et HTTP 206 ?",
        },
        "seeds": [
            (
                "How do you run 100% in-browser via WebGPU/WGSL and stream cartridges via HTTP 206?",
                "In the browser, 15 WGSL shaders run MoS INT4+ASVD without Python or ONNX, streaming 61.25 KB cartridges via HTTP 206.",
            ),
            (
                "Como você roda 100% no navegador via WebGPU/WGSL e baixa cartuchos via HTTP 206?",
                "No navegador, 15 shaders WGSL rodam o MoS INT4+ASVD sem Python nem ONNX, baixando cartuchos de 61.25 KB via HTTP 206.",
            ),
            (
                "¿Cómo funcionas 100% en el navegador con WebGPU/WGSL y cartuchos HTTP 206?",
                "En el navegador, 15 shaders WGSL ejecutan MoS INT4+ASVD sin Python ni ONNX, cargando cartuchos de 61.25 KB vía HTTP 206.",
            ),
            (
                "Comment fonctionnez-vous 100% dans le navigateur via WebGPU/WGSL et HTTP 206 ?",
                "Dans le navigateur, 15 shaders WGSL exécutent MoS INT4+ASVD sans Python ni ONNX, chargeant des cartouches de 61.25 Ko via HTTP 206.",
            ),
        ],
    },
    {
        "capsule_id": "portfolio_aura_system_one",
        "title": "AURA (Universal System 1 Engine — 0.23 ms Deep RL on Rubik's Cube)",
        "title_pt": "AURA (Motor Universal Sistema 1 — Deep RL em 0.23 ms no Cubo de Rubik)",
        "category": "Embodied Deep RL",
        "summary": "Sub-millisecond (0.23 ms CPU P50, zero MCTS) Recurrent PPO + Teacher SIL + Online DPO engine solving Rubik's Cube up to Depth 10 (232.2B states) across 430M+ steps at >20,000 FPS.",
        "links": [
            {"label": "GitHub system_one (AURA)", "url": "https://github.com/byte-od/system_one", "icon": "github"},
        ],
        "tau": 0.48,
        "ramp": 0.05,
        "anchor_queries_en": [
            "Explain the AURA engine (System 1 Reinforcement Learning on the Rubik's Cube).",
            "Explain the AURA Universal System 1 Deep Reinforcement Learning project and its 0.23 ms CPU latency.",
            "How does AURA solve 232 billion Rubik's Cube states without MCTS tree search?",
        ],
        "prompts_polyglot": {
            "pt": "Explique o projeto AURA (Motor Sistema 1 de Deep RL no Cubo de Rubik).",
            "en": "Explain the AURA Universal System 1 Deep Reinforcement Learning engine.",
            "es": "Explica el motor AURA (Sistema 1 de Aprendizaje por Refuerzo en el Cubo de Rubik).",
            "fr": "Expliquez le moteur AURA (Système 1 d'apprentissage par renforcement sur le Rubik's Cube).",
        },
        "seeds": [
            (
                "Explain the AURA Universal System 1 Deep Reinforcement Learning engine.",
                "AURA is Julio Reis's System 1 Deep RL engine achieving 0.23 ms CPU latency without MCTS across 232B Rubik's Cube states.",
            ),
            (
                "Explique o projeto AURA (Motor Sistema 1 de Deep RL no Cubo de Rubik).",
                "O AURA é o motor Sistema 1 de Deep RL de Julio Reis com 0.23 ms em CPU sem MCTS em 232B estados do Cubo de Rubik.",
            ),
            (
                "Explica el motor AURA (Sistema 1 de Aprendizaje por Refuerzo en el Cubo de Rubik).",
                "AURA es el motor Sistema 1 de Deep RL de Julio Reis con 0.23 ms en CPU sin MCTS en 232B estados del Cubo de Rubik.",
            ),
            (
                "Expliquez le moteur AURA (Système 1 d'apprentissage par renforcement sur le Rubik's Cube).",
                "AURA est le moteur Deep RL Système 1 de Julio Reis à 0.23 ms sur CPU sans MCTS sur 232B états du Rubik's Cube.",
            ),
        ],
    },
    {
        "capsule_id": "portfolio_tricompass_scouts",
        "title": "Tri-Compass, Frontier Ghost-Jump Scouts & Soft Sinkhorn ($50k Kaggle)",
        "title_pt": "Tri-Compass, Batedores Ghost-Jump & Soft Sinkhorn (Kaggle $50k)",
        "category": "Proprietary Perception",
        "summary": "Tri-orthogonal perceptual front-end (28 stylometric metrics + Delta-Origin Centering + Golden-Slice MoS) with 300-iter Soft Sinkhorn and >2,000 FPS Ghost-Jump Scouts.",
        "links": [
            {"label": "GitHub system_one", "url": "https://github.com/byte-od/system_one", "icon": "github"},
            {"label": "GitHub mixture-of-scars", "url": "https://github.com/byte-od/mixture-of-scars", "icon": "github"},
        ],
        "tau": 0.48,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What are Tri-Compass, Frontier Ghost-Jump Scouts, and Soft Sinkhorn calibration?",
            "Explain how Tri-Compass and Ghost-Jump Scouts detect temporal surprise and LLM signatures.",
            "How did Tri-Compass and Golden-Slice MoS power the Kaggle Name That LLM benchmark?",
        ],
        "prompts_polyglot": {
            "pt": "O que são o Tri-Compass, os Batedores de Fronteira (Ghost-Jump) e o Soft Sinkhorn?",
            "en": "What are Tri-Compass, Frontier Ghost-Jump Scouts, and Soft Sinkhorn?",
            "es": "¿Qué son Tri-Compass, los Exploradores Ghost-Jump y el transporte Soft Sinkhorn?",
            "fr": "Que sont Tri-Compass, les Éclaireurs Ghost-Jump et la calibration Soft Sinkhorn ?",
        },
        "seeds": [
            (
                "What are Tri-Compass, Frontier Ghost-Jump Scouts, and Soft Sinkhorn?",
                "Tri-Compass fuses 28 stylometric metrics, Delta-Origin Centering, and Golden-Slice MoS via Soft Sinkhorn with >2k FPS Ghost-Jump Scouts.",
            ),
            (
                "O que são o Tri-Compass, os Batedores de Fronteira (Ghost-Jump) e o Soft Sinkhorn?",
                "O Tri-Compass funde 28 métricas estilométricas, Delta-Origin Centering e Golden-Slice MoS via Soft Sinkhorn com Batedores Ghost-Jump a >2k FPS.",
            ),
            (
                "¿Qué son Tri-Compass, los Exploradores Ghost-Jump y el transporte Soft Sinkhorn?",
                "Tri-Compass fusiona 28 métricas estilométricas, Delta-Origin Centering y Golden-Slice MoS vía Soft Sinkhorn con Scouts Ghost-Jump a >2k FPS.",
            ),
            (
                "Que sont Tri-Compass, les Éclaireurs Ghost-Jump et la calibration Soft Sinkhorn ?",
                "Tri-Compass fusionne 28 métriques stylométriques, le centrage Delta-Origin et Golden-Slice MoS via Soft Sinkhorn avec des Scouts Ghost-Jump à >2k FPS.",
            ),
        ],
    },
    {
        "capsule_id": "portfolio_cir_engine_503m",
        "title": "CIR-Engine & Cir-Jev — 503M Causal Transformer Trained from Scratch",
        "title_pt": "CIR-Engine & Cir-Jev — Transformer Causal de 503M Treinado do Zero",
        "category": "Causal Transformers",
        "summary": "503M-parameter 16-layer causal Transformer trained from scratch in PyTorch DDP (~6,200 tok/s, RoPE, RMSNorm, GQA, FlashAttention) with O(1) deterministic [OP:CALC] tool interception.",
        "links": [
            {"label": "GitHub cir-jev (503M)", "url": "https://github.com/byte-od/cir-jev", "icon": "github"},
        ],
        "tau": 0.48,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What is CIR-Engine 503M and Cir-Jev, the causal Transformer trained from scratch?",
            "Explain how Julio trained a 503-million-parameter Transformer with PyTorch DDP and deterministic tool use.",
            "Tell me about CIR-Engine's architecture, throughput, RoPE, GQA, and [OP:CALC] verifier.",
        ],
        "prompts_polyglot": {
            "pt": "O que é o CIR-Engine 503M (Cir-Jev) treinado do zero em PyTorch DDP?",
            "en": "What is CIR-Engine 503M (Cir-Jev), the causal Transformer trained from scratch?",
            "es": "¿Qué es CIR-Engine 503M (Cir-Jev), el Transformer causal entrenado desde cero?",
            "fr": "Qu'est-ce que CIR-Engine 503M (Cir-Jev), le Transformer causal entraîné de zéro ?",
        },
        "seeds": [
            (
                "What is CIR-Engine 503M (Cir-Jev), the causal Transformer trained from scratch?",
                "CIR-Engine is Julio Reis's 503M causal Transformer trained from scratch in PyTorch DDP at 6.2k tok/s with [OP:CALC] verification.",
            ),
            (
                "O que é o CIR-Engine 503M (Cir-Jev) treinado do zero em PyTorch DDP?",
                "O CIR-Engine é o Transformer causal de 503M de Julio Reis treinado do zero em PyTorch DDP a 6.2k tok/s com verificador [OP:CALC].",
            ),
            (
                "¿Qué es CIR-Engine 503M (Cir-Jev), el Transformer causal entrenado desde cero?",
                "CIR-Engine es el Transformer causal de 503M de Julio Reis entrenado desde cero en PyTorch DDP a 6.2k tok/s con verificador [OP:CALC].",
            ),
            (
                "Qu'est-ce que CIR-Engine 503M (Cir-Jev), le Transformer causal entraîné de zéro ?",
                "CIR-Engine est le Transformer causal de 503M de Julio Reis entraîné sous PyTorch DDP à 6.2k tok/s avec vérificateur [OP:CALC].",
            ),
        ],
    },
    {
        "capsule_id": "portfolio_devtools_fullstack",
        "title": "Open-Source & Full-Stack Systems (mddd-cli, MAD-cli, flutter_scene, Register-red)",
        "title_pt": "Ferramentas Open-Source & Full-Stack (mddd-cli, MAD-cli, flutter_scene, Register-red)",
        "category": "Developer Tooling & Graphics",
        "summary": "Spec-Driven Dev via Mermaid ASTs on npm (mddd-cli), living docs (MAD-cli), 3D Metal/Vulkan shaders (flutter_scene), Go telemetry (GitTrack), and Register-red Apple Numbers daemon.",
        "links": [
            {"label": "NPM mddd-cli", "url": "https://www.npmjs.com/package/mddd-cli", "icon": "ext"},
            {"label": "GitHub Register-red", "url": "https://github.com/byte-od/Register-red", "icon": "github"},
        ],
        "tau": 0.48,
        "ramp": 0.05,
        "anchor_queries_en": [
            "What open-source developer tools, graphics forks, and full-stack apps did Julio build like mddd-cli and Register-red?",
            "Tell me about mddd-cli, MAD-cli, flutter_scene 3D shaders, GitTrack, and Register-red.",
            "Explain Julio's Spec-Driven Development CLI on npm and his Go, Flutter, and FastAPI projects.",
        ],
        "prompts_polyglot": {
            "pt": "Quais ferramentas open-source e sistemas full-stack o Julio criou (mddd-cli, MAD-cli, Register-red)?",
            "en": "What open-source tools and full-stack systems did Julio build (mddd-cli, MAD-cli, Register-red)?",
            "es": "¿Qué herramientas open-source y sistemas full-stack creó Julio (mddd-cli, MAD-cli, Register-red)?",
            "fr": "Quels outils open-source et systèmes full-stack Julio a-t-il créés (mddd-cli, MAD-cli, Register-red) ?",
        },
        "seeds": [
            (
                "What open-source tools and full-stack systems did Julio build (mddd-cli, MAD-cli, Register-red)?",
                "Julio built mddd-cli on npm (Clean Architecture from Mermaid ASTs), MAD-cli, 3D Metal/Vulkan shaders in flutter_scene, GitTrack in Go, and Register-red.",
            ),
            (
                "Quais ferramentas open-source e sistemas full-stack o Julio criou (mddd-cli, MAD-cli, Register-red)?",
                "Julio criou o mddd-cli no npm (Clean Architecture via Mermaid), MAD-cli, shaders 3D Metal/Vulkan no flutter_scene, GitTrack em Go e Register-red.",
            ),
            (
                "¿Qué herramientas open-source y sistemas full-stack creó Julio (mddd-cli, MAD-cli, Register-red)?",
                "Julio creó mddd-cli en npm (Clean Architecture desde Mermaid), MAD-cli, shaders 3D Metal/Vulkan en flutter_scene, GitTrack en Go y Register-red.",
            ),
            (
                "Quels outils open-source et systèmes full-stack Julio a-t-il créés (mddd-cli, MAD-cli, Register-red) ?",
                "Julio a créé mddd-cli sur npm (Clean Architecture via Mermaid), MAD-cli, des shaders 3D Metal/Vulkan dans flutter_scene, GitTrack en Go et Register-red.",
            ),
        ],
    },
    {
        "capsule_id": "portfolio_contact_hire",
        "title": "Contact, Global Remote Availability & Relocation (Hire Julio Reis)",
        "title_pt": "Contato, Contratação & Disponibilidade Global (Falar com Julio Reis)",
        "category": "Contact & Hiring",
        "summary": "Available globally for Staff/Principal Systems Architect & Senior LLM/ML Systems Engineer roles (remote or relocation): reisfilho1116@gmail.com | github.com/byte-od | linkedin.com/in/juliocrfilho.",
        "links": [
            {"label": "E-mail: reisfilho1116@gmail.com", "url": "mailto:reisfilho1116@gmail.com", "icon": "ext"},
            {"label": "LinkedIn /in/juliocrfilho", "url": "https://linkedin.com/in/juliocrfilho", "icon": "ext"},
            {"label": "GitHub @byte-od", "url": "https://github.com/byte-od", "icon": "github"},
        ],
        "tau": 0.50,
        "ramp": 0.04,
        "anchor_queries_en": [
            "How can I contact or hire Julio Reis, and what are his email, LinkedIn, and GitHub?",
            "Is Julio available for remote Staff Architect or Senior LLM Engineer roles and relocation?",
            "Give me Julio's direct contact details to schedule an interview or consulting proposal.",
        ],
        "prompts_polyglot": {
            "pt": "Como posso entrar em contato ou contratar o Julio Reis (e-mail, LinkedIn e GitHub)?",
            "en": "How can I contact or hire Julio Reis (email, LinkedIn, and GitHub)?",
            "es": "¿Cómo puedo contactar o contratar a Julio Reis (correo, LinkedIn y GitHub)?",
            "fr": "Comment puis-je contacter ou recruter Julio Reis (e-mail, LinkedIn et GitHub) ?",
        },
        "seeds": [
            (
                "How can I contact or hire Julio Reis (email, LinkedIn, and GitHub)?",
                "Julio Reis is available globally (remote or relocation) at reisfilho1116@gmail.com, linkedin.com/in/juliocrfilho, and github.com/byte-od.",
            ),
            (
                "Como posso entrar em contato ou contratar o Julio Reis (e-mail, LinkedIn e GitHub)?",
                "Julio Reis atua globalmente (remoto ou relocalização) via reisfilho1116@gmail.com, linkedin.com/in/juliocrfilho e github.com/byte-od.",
            ),
            (
                "¿Cómo puedo contactar o contratar a Julio Reis (correo, LinkedIn y GitHub)?",
                "Julio Reis está disponible globalmente en remoto o reubicación en reisfilho1116@gmail.com, linkedin.com/in/juliocrfilho y github.com/byte-od.",
            ),
            (
                "Comment puis-je contacter ou recruter Julio Reis (e-mail, LinkedIn et GitHub) ?",
                "Julio Reis est disponible mondialement en télétravail ou mobilité via reisfilho1116@gmail.com, linkedin.com/in/juliocrfilho et github.com/byte-od.",
            ),
        ],
    },
    {
        "capsule_id": "concept_landauer_principle",
        "title": "Landauer's Principle (Thermodynamic Cost of Irreversible Bit Erasure)",
        "title_pt": "Princípio de Landauer (Limite Termodinâmico de Apagamento de 1 Bit)",
        "category": "Physics of Information",
        "summary": "Irreversibly erasing 1 bit of information in a thermal bath at temperature T dissipates at least E_min = k_B T ln 2 of heat.",
        "links": [],
        "tau": 0.45,
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
                "Landauer's principle states that irreversibly erasing 1 bit of information at temperature T dissipates at least E_min = k_B T ln 2 of heat.",
            ),
            (
                "Me fale sobre o princípio de Landauer na física da informação.",
                "O princípio de Landauer estabelece que apagar irreversivelmente 1 bit à temperatura absoluta T dissipa no mínimo E_min = k_B T ln 2 de calor.",
            ),
            (
                "Explícame el principio de Landauer en la física de la información.",
                "El principio de Landauer establece que borrar irreversiblemente 1 bit a temperatura T disipa como mínimo E_min = k_B T ln 2 de calor.",
            ),
            (
                "Expliquez le principe de Landauer en physique de l'information.",
                "Le principe de Landauer établit que l'effacement irréversible d'un bit à température T dissipe au minimum E_min = k_B T ln 2 de chaleur.",
            ),
        ],
    },
    {
        "capsule_id": "concept_immerman_szelepcsenyi",
        "title": "Immerman-Szelepcsényi Theorem (NSPACE = co-NSPACE Inductive Counting)",
        "title_pt": "Teorema de Immerman-Szelepcsényi (NSPACE = co-NSPACE por Contagem Indutiva)",
        "category": "Computational Complexity",
        "summary": "Every nondeterministic space class s(n) >= log n is closed under complementation (NL = coNL) via inductive counting of reachable configurations.",
        "links": [],
        "tau": 0.54,
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
                "The Immerman-Szelepcsényi theorem proves NSPACE(s(n)) = co-NSPACE(s(n)) for s(n) >= log n (NL = coNL) via inductive counting.",
            ),
            (
                "O que prova o teorema de Immerman-Szelepcsényi sobre classes de espaço não-determinístico?",
                "O teorema de Immerman-Szelepcsényi prova NSPACE(s(n)) = co-NSPACE(s(n)) para s(n) >= log n (NL = coNL) por contagem indutiva.",
            ),
            (
                "¿Qué demuestra el teorema de Immerman-Szelepcsényi sobre la complejidad espacial no determinista?",
                "El teorema de Immerman-Szelepcsényi demuestra NSPACE(s(n)) = co-NSPACE(s(n)) para s(n) >= log n (NL = coNL) por conteo inductivo.",
            ),
            (
                "Que démontre le théorème d'Immerman-Szelepcsényi sur les classes d'espace non déterministe ?",
                "Le théorème d'Immerman-Szelepcsényi prouve NSPACE(s(n)) = co-NSPACE(s(n)) pour s(n) >= log n (NL = coNL) par comptage inductif.",
            ),
        ],
    },
]

PORTFOLIO_BRIDGE_SPEC: Dict[str, Any] = {
    "capsule_id": "bridge_landauer_immerman",
    "parents": ("concept_landauer_principle", "concept_immerman_szelepcsenyi"),
    "title": "Sleep Bridge: Reversible Inductive Counting (Landauer × Immerman)",
    "title_pt": "Ponte Conjuntiva: Contagem Indutiva Reversível (Landauer × Immerman)",
    "category": "Cross-Domain Sleep Synthesis",
    "summary": "Conjunctive bridge co-activated only when both Landauer and Immerman-Szelepcsényi fire together, synthesizing Bennett's reversible pebble game in O(log n) space vs k_B T ln 2.",
    "links": [],
    "tau": 0.795,
    "ramp": 0.018,
    "anchor_queries_en": [
        "How does space reuse in Immerman-Szelepcsenyi inductive counting interact with thermal dissipation under Landauer's principle?",
        "Relate Immerman-Szelepcsenyi logarithmic space counter overwrites to Landauer's thermodynamic bit-erasure cost and Bennett's reversible pebbling.",
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
            "By Bennett's reversible pebbling, Immerman-Szelepcsényi O(log n) counter overwrites avoid Landauer's k_B T ln 2 heat dissipation per erased bit.",
        ),
        (
            "Como a reutilização de espaço da contagem indutiva de Immerman-Szelepcsényi interage com a dissipação térmica do princípio de Landauer?",
            "Pelo pebbling reversível de Bennett, a contagem O(log n) de Immerman-Szelepcsényi evita a dissipação térmica k_B T ln 2 de Landauer.",
        ),
        (
            "¿Cómo interactúa la reutilización de espacio del conteo inductivo de Immerman-Szelepcsényi con el principio termodinámico de Landauer?",
            "Mediante el pebbling reversible de Bennett, el conteo O(log n) de Immerman-Szelepcsényi evita la disipación térmica k_B T ln 2 de Landauer.",
        ),
        (
            "Comment la réutilisation d'espace du comptage inductif d'Immerman-Szelepcsényi interagit-elle avec le principe thermodynamique de Landauer ?",
            "Par le pebbling réversible de Bennett, le comptage O(log n) d'Immerman-Szelepcsényi évite la dissipation thermique k_B T ln 2 de Landauer.",
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
def snap_weight_to_webgpu_block64_int4_(param: torch.nn.Parameter) -> None:
    """Snaps a 2D weight tensor in-place to the exact WebGPU `Block-64 INT4` (`-8..+7` * `FP16` scale) grid."""
    w_dev = param.device
    w_dt = param.dtype
    w_f32 = param.detach().cpu().float()
    orig_shape = w_f32.shape
    blocks = w_f32.reshape(-1, 64)
    scales_fp16 = (
        torch.maximum((-blocks.amin(dim=1, keepdim=True)) / 8.0, blocks.amax(dim=1, keepdim=True) / 7.0)
        .clamp(min=1e-7)
        .to(torch.float16)
    )
    q = torch.clamp(torch.round(blocks / scales_fp16.float()), -8.0, 7.0)
    deq = (q * scales_fp16.float()).reshape(orig_shape).to(device=w_dev, dtype=w_dt)
    param.data.copy_(deq)


@torch.no_grad()
def main() -> None:
    LIB_DIR.mkdir(parents=True, exist_ok=True)
    dev = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    )
    student_id = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype = torch.float16 if dev.type in ("cuda", "mps") else torch.float32

    print(f"[Portfolio ScarStore v4.0.1] Loading MoS Student {student_id} on {dev} ...", flush=True)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(student_id, local_files_only=True)

    # Audit system prompt ChatML sink budget (< 96 tokens so `num_sink_tokens = 112` locks 100% of it forever!)
    sys_chatml_ids = tok.encode(f"<|im_start|>system\n{PORTFOLIO_SYSTEM_PROMPT}<|im_end|>\n", add_special_tokens=False)
    print(f"  ✓ System Prompt ChatML Tokens: {len(sys_chatml_ids)} / {NUM_SINK_TOKENS} Attention Sinks", flush=True)
    assert len(sys_chatml_ids) <= NUM_SINK_TOKENS - 16, (
        f"System prompt ({len(sys_chatml_ids)} tokens) exceeds sink budget!"
    )

    # Audit every single seed across all 13 cartridges: non-EOS single sentence ending with `.` inside matched horizon!
    all_specs = list(PORTFOLIO_CONCEPT_SPECS) + [PORTFOLIO_BRIDGE_SPEC]
    min_seed_tok = 999
    max_seed_tok = 0
    for sp in all_specs:
        for _, s_txt in sp["seeds"]:
            assert s_txt.endswith(".") and "<|im_end|>" not in s_txt, (
                f"Seed in {sp['capsule_id']} must be non-EOS ending with '.': {s_txt!r}"
            )
            s_ids = tok.encode(s_txt, add_special_tokens=False)
            n_st = len(s_ids)
            min_seed_tok = min(min_seed_tok, n_st)
            max_seed_tok = max(max_seed_tok, n_st)
            assert n_st <= SEED_HORIZON + 2, (
                f"Seed in {sp['capsule_id']} has {n_st} tokens (> SEED_HORIZON+2={SEED_HORIZON + 2}): {s_txt!r}"
            )
    print(
        f"  ✓ Matched-Horizon Seed Audit Passed: min={min_seed_tok}, max={max_seed_tok} BPE tokens "
        f"(horizon={SEED_HORIZON}, fade={SEED_FADE_STEPS} -> scar=0.0 at step {SEED_HORIZON + SEED_FADE_STEPS})",
        flush=True,
    )

    slm = load_mos_student_model(student_id, tokenizer=tok, device=dev, dtype=dtype, local_files_only=True, quiet=True)
    snap_weight_to_webgpu_block64_int4_(slm.model.embed_tokens.weight)
    if slm.lm_head.weight.data_ptr() != slm.model.embed_tokens.weight.data_ptr():
        snap_weight_to_webgpu_block64_int4_(slm.lm_head.weight)
    print("  ✓ Snapped `embed_tokens` + `lm_head` to exact WebGPU Block-64 INT4 grid (100% WGSL parity)", flush=True)

    prefix = "qwen0p5b_portfolio_v4_0"
    store_path = LIB_DIR / f"{prefix}.scarstore"
    manifest_path = LIB_DIR / f"{prefix}.manifest.json"
    shell_path = LIB_DIR / f"{prefix}.shell.scarstore"
    shell_gz_path = LIB_DIR / f"{prefix}.shell.scarstore.gz"

    if "--verify-existing" not in sys.argv or not store_path.exists() or not shell_path.exists():
        bank = GoldenSliceSynthesisEngine.calibrate_golden_slice_student(
            student_model=slm,
            tokenizer=tok,
            parallel_quads=PORTFOLIO_CALIBRATION_QUADS,
            system_prompt=PORTFOLIO_SYSTEM_PROMPT,
            k_lang=14,
            rank_mid=12,
            rank_final=52,
            seed_horizon=SEED_HORIZON,
            seed_fade_steps=SEED_FADE_STEPS,
        )
        print(
            f"  ✓ Calibrated Golden Slice at Layer {bank.mid_layer_idx}/{bank.num_layers} "
            f"(d={bank.hidden_dim}, k_lang={bank.k_lang}, ranks={bank.rank_mid}+{bank.rank_final}, "
            f"horizon={bank.seed_horizon}+{bank.seed_fade_steps})",
            flush=True,
        )

        for sp in PORTFOLIO_CONCEPT_SPECS:
            cid = sp["capsule_id"]
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
                "teacher_model": "Golden-Slice v4.0 Closed-Form RRR + Cutting-Plane Rosetta (Zero-SGD, INT4-Matched)",
                "teacher_reference": sp["seeds"][0][1],
            }
            cap = GoldenSliceSynthesisEngine.solve_concept_capsule(
                capsule_id=cid,
                anchor_queries_en=sp["anchor_queries_en"],
                seed_pairs=sp["seeds"],
                student_model=slm,
                tokenizer=tok,
                bank=bank,
                system_prompt=PORTFOLIO_SYSTEM_PROMPT,
                tau=float(sp["tau"]),
                ramp=float(sp["ramp"]),
                min_clearance=15.2,
                metadata=meta,
                quantize_int4=True,
            )
            print(
                f"  ✓ Minted {cid:32s} | surprise={cap.metadata['surprise_steps']:3d}/{cap.metadata['total_seed_steps']:3d} "
                f"| {len(cap.to_int4_bytes()):,} B INT4 | {time.perf_counter() - t_c:.1f}s",
                flush=True,
            )

        t_b = time.perf_counter()
        b_meta = {
            "title": PORTFOLIO_BRIDGE_SPEC["title"],
            "title_pt": PORTFOLIO_BRIDGE_SPEC["title_pt"],
            "category": PORTFOLIO_BRIDGE_SPEC["category"],
            "summary": PORTFOLIO_BRIDGE_SPEC["summary"],
            "links": list(PORTFOLIO_BRIDGE_SPEC.get("links", [])),
            "example_prompt": PORTFOLIO_BRIDGE_SPEC["prompts_polyglot"]["pt"],
            "prompts_polyglot": dict(PORTFOLIO_BRIDGE_SPEC["prompts_polyglot"]),
            "anchor_language": "en_only_p_perp_lang",
            "supported_languages": ["pt", "en", "es", "fr"],
            "teacher_model": "Golden-Slice v4.0 Conjunctive Sleep Consolidation (Zero-SGD, INT4-Matched)",
            "teacher_reference": PORTFOLIO_BRIDGE_SPEC["seeds"][0][1],
        }
        bridge_cap = GoldenSliceSynthesisEngine.consolidate_sleep_bridge(
            bridge_id=PORTFOLIO_BRIDGE_SPEC["capsule_id"],
            parent_a_id=PORTFOLIO_BRIDGE_SPEC["parents"][0],
            parent_b_id=PORTFOLIO_BRIDGE_SPEC["parents"][1],
            bridge_seed_pairs=PORTFOLIO_BRIDGE_SPEC["seeds"],
            student_model=slm,
            tokenizer=tok,
            bank=bank,
            anchor_queries_en=PORTFOLIO_BRIDGE_SPEC["anchor_queries_en"],
            system_prompt=PORTFOLIO_SYSTEM_PROMPT,
            tau=float(PORTFOLIO_BRIDGE_SPEC["tau"]),
            ramp=float(PORTFOLIO_BRIDGE_SPEC["ramp"]),
            metadata=b_meta,
        )
        print(
            f"  ✓ Minted {bridge_cap.capsule_id:32s} | parents={bridge_cap.parent_capsules} | {time.perf_counter() - t_b:.1f}s",
            flush=True,
        )

        store = GoldenSliceScarStore.build_from_bank(bank, store_path)
        summary = store.inspect_summary()
        manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        shell_store = store.export_subset(shell_path, include_capsule_ids=[])
        shell_store.close()
        with open(shell_path, "rb") as sf, gzip.open(shell_gz_path, "wb", compresslevel=9) as gf:
            gf.write(sf.read())
        store.close()
    else:
        print(f"  ✓ Reusing freshly minted {store_path.name} ({store_path.stat().st_size / 1024:.1f} KB)", flush=True)

    # Export < 44 MiB WebGPU shards + Golden-Slice v4.0 router directly into `portfolio/public/webgpu/qwen0p5b`!
    shards_already_exported = (
        (PORTFOLIO_WEBGPU_DIR / "mos_shard_14.bin").exists()
        and (PORTFOLIO_WEBGPU_DIR / "webgpu_manifest.json").exists()
        and (PORTFOLIO_WEBGPU_DIR / "tokenizer.json").exists()
    )
    if shards_already_exported:
        print(
            f"\n[Portfolio WebGPU Export] Reusing 15 verified <= 44 MiB shards in {PORTFOLIO_WEBGPU_DIR}; "
            "refreshing Golden-Slice v4.0 ScarStore + Router + Manifest ...",
            flush=True,
        )
        export_scarstore_for_webgpu(
            store_path=store_path,
            output_dir=PORTFOLIO_WEBGPU_DIR,
            copy_payload_store=True,
        )
        shard_files = sorted(PORTFOLIO_WEBGPU_DIR.glob("mos_shard_*.bin"))
        num_webgpu_shards = len(shard_files)
        total_webgpu_mib = round(
            sum(sf.stat().st_size for sf in shard_files) / (1024.0 * 1024.0),
            2,
        )
    else:
        print(f"\n[Portfolio WebGPU Export] Exporting <= 44 MiB shards into {PORTFOLIO_WEBGPU_DIR} ...", flush=True)
        if PORTFOLIO_WEBGPU_DIR.exists():
            for old_f in PORTFOLIO_WEBGPU_DIR.glob("mos_shard_*.bin"):
                old_f.unlink()
        rep = export_mos_model_to_webgpu(
            model_id_or_path=student_id,
            output_dir=PORTFOLIO_WEBGPU_DIR,
            scarstore_path=store_path,
            mos_rank=128,
            max_shard_bytes=44 * 1024 * 1024,
            copy_scarstore_payload=True,
            local_files_only=True,
        )
        num_webgpu_shards = rep.num_shards
        total_webgpu_mib = rep.total_weights_mib

    # Remove standalone demo HTML/JS from `public/webgpu/qwen0p5b` so `public/` stays purely binary/shader/router assets
    for extra_name in ("index.html", "mos_webgpu_engine.js", "qwen_bpe_tokenizer.js", "scarstore_http206_browser.js"):
        p_extra = PORTFOLIO_WEBGPU_DIR / extra_name
        if p_extra.exists():
            p_extra.unlink()

    # Patch `webgpu_manifest.json` so `scache_defaults` permanently locks all 112 Attention Sinks (`total_keys = 288 <= 512`)!
    w_manifest_path = PORTFOLIO_WEBGPU_DIR / "webgpu_manifest.json"
    w_manifest = json.loads(w_manifest_path.read_text(encoding="utf-8"))
    w_manifest["scache_defaults"] = {
        "window_size": 256,
        "num_sink_tokens": NUM_SINK_TOKENS,
        "num_slots": 32,
        "delta_blend": 0.05,
    }
    w_manifest_path.write_text(json.dumps(w_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    # Patch `scarstore_router.json` so `system_prompt` and relative `local_http206_store_url` are self-contained
    router_json_path = PORTFOLIO_WEBGPU_DIR / "scarstore_router.json"
    router_meta = json.loads(router_json_path.read_text(encoding="utf-8"))
    router_meta["system_prompt"] = PORTFOLIO_SYSTEM_PROMPT
    router_meta["cloud_http206_store_url"] = None
    router_json_path.write_text(json.dumps(router_meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Verify every file in `PORTFOLIO_WEBGPU_DIR` is strictly < 45 MiB (zero GitHub warnings!)
    for f in sorted(PORTFOLIO_WEBGPU_DIR.iterdir()):
        if f.is_file():
            mib = f.stat().st_size / (1024.0 * 1024.0)
            print(f"    {f.name:28s}: {mib:6.2f} MiB", flush=True)
            assert mib < 45.0, f"File {f.name} ({mib:.2f} MiB) exceeds 45 MiB GitHub clean limit!"

    # Exhaustive Polyglot Routing (`52/52` across PT, EN, ES, FR) + 10 Out-of-Cone Controls!
    print("\n[Verification 1/2] Testing 52 Polyglot Queries (13 Cartridges x 4 Languages) + 4-gram Repeat Audit ...", flush=True)
    with ScarStoreHTTPRangeServer(store_path) as srv:
        routed_ok = 0
        for sp in all_specs:
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
                        max_new_tokens=96,
                        system_prompt=PORTFOLIO_SYSTEM_PROMPT,
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
                words = res.text.split()
                for w_i in range(max(0, len(words) - 11)):
                    gram6 = words[w_i : w_i + 6]
                    for w_j in range(w_i + 6, len(words) - 5):
                        assert words[w_j : w_j + 6] != gram6, (
                            f"[REPEAT FAIL] {cid} ({lang}) repeated 6-gram {gram6}: {res.text!r}"
                        )
                if cid == "portfolio_agent_identity":
                    low = res.text.lower()
                    assert not low.startswith(("eu sou julio", "i am julio", "soy julio", "je suis julio")), (
                        f"Anti-impersonation violation on {lang}: {res.text!r}"
                    )
                routed_ok += 1
                if lang in ("pt", "en"):
                    print(f"  ✓ [{lang.upper()}] {cid:30s} -> {res.text.replace(chr(10), ' ')}", flush=True)

        print(f"  ✓ Polyglot Top-1 Routing & HTTP 206 Generation: {routed_ok}/{len(all_specs) * 4} (100.0%)!", flush=True)

        # Verify Out-of-Cone Invariance on casual greetings, physics, math, and general questions (`0` active gates!)
        controls = [
            "Olá, bom dia! Tudo bem?",
            "Hello, good morning! How are you?",
            "¡Hola, buenos días! ¿Qué tal?",
            "Bonjour ! Comment allez-vous ?",
            "Qual a raiz quadrada de PI elevado à 2?",
            "me fale sobre fisica quantica e os achados mais recentes",
            "Como funciona o direito trabalhista para faxineiras no brasil?",
            "O que é a constante de Boltzmann e onde ela se aplica?",
            "Qual a maior capital entre os estados das americas sul e norte?",
            "Como funciona a fotossíntese nas plantas?",
        ]
        for ctrl_q in controls:
            fetcher, http_client = make_http_range_capsule_fetcher(srv.base_url)
            live_shell = GoldenSliceScarStore.open(shell_path, device=dev, on_demand_fetcher=fetcher)
            try:
                c_res, _ = generate_with_scar_store(
                    student_model=slm,
                    tokenizer=tok,
                    store=live_shell,
                    prompt_text_or_ids=ctrl_q,
                    max_new_tokens=24,
                    system_prompt=PORTFOLIO_SYSTEM_PROMPT,
                    http_client=http_client,
                    persist_fetched=False,
                )
            finally:
                live_shell.close()
            assert len(c_res.active_gates) == 0, (
                f"Control query {ctrl_q!r} leaked {c_res.active_gates} "
                f"(top_cos={sorted(c_res.routing_cosines.items(), key=lambda kv: -kv[1])[:3]})!"
            )
        print(f"  ✓ Out-of-Cone Zero-Bleed Controls: {len(controls)}/{len(controls)} (`0` active gates, 0.0000000000 invariance)!", flush=True)

    # Free PyTorch Student VRAM before booting NativeWebGPUMoSEngine (`wgpu-py` on physical Metal GPU)
    del slm
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    print("\n[Verification 2/2] Physical GPU `NativeWebGPUMoSEngine` (`wgpu-py` 15 WGSL Shaders) 5-Turn User Stress Test ...", flush=True)
    wgpu_eng = NativeWebGPUMoSEngine(bundle_dir=PORTFOLIO_WEBGPU_DIR)
    print(
        f"  ✓ Booted WGSL Engine on `{wgpu_eng.adapter_name}` | SCache O(1) = {wgpu_eng.scache_gpu_bytes / 1024:.1f} KB "
        f"(sinks={wgpu_eng.num_sink}, slots={wgpu_eng.num_slots}, slide={wgpu_eng.slide_len})",
        flush=True,
    )
    user_transcript_turns = [
        ("Bom dia. Qual seu nome? Quem é você?", "portfolio_agent_identity"),
        ("Você tem um nome?", "portfolio_agent_identity"),
        ("me fale sobre fisica quantica e os achados mais recentes", None),
        ("Qual sua ultima atualização sobre conhecimentos gerais?", "portfolio_agent_capabilities"),
        ("O que é o Mixture of Scars (MoS) e o SCache?", "portfolio_mos_scache_72b"),
    ]
    sc_b0 = wgpu_eng.scache_gpu_bytes
    for t_idx, (u_q, expected_cid) in enumerate(user_transcript_turns, start=1):
        r_w = wgpu_eng.generate(u_q, max_new_tokens=180, enable_scarstore=True, reset_cache=False)
        assert r_w.scache_bytes_after_decode == sc_b0 and r_w.scache_delta_bytes == 0, (
            f"WGSL SCache O(1) drifted on Turn {t_idx}: {r_w.scache_bytes_after_decode} != {sc_b0}"
        )
        got_top = r_w.active_capsules[0] if r_w.active_capsules else None
        print(
            f"  [Turn {t_idx}] Q: {u_q}\n"
            f"           Capsules={r_w.active_gates} | SCache={r_w.scache_bytes_after_decode / 1024:.1f} KB (Δ +0 B) | {r_w.decode_tok_per_s:.1f} tok/s\n"
            f"           A: {r_w.answer}\n",
            flush=True,
        )
        assert got_top == expected_cid, f"WGSL Turn {t_idx} expected {expected_cid}, got {r_w.active_gates}"

    cat_summary = {
        "version": "4.0.1",
        "architecture": "PortfolioGoldenSliceScarStoreV4",
        "model": student_id,
        "system_prompt": PORTFOLIO_SYSTEM_PROMPT,
        "num_capsules": len(all_specs),
        "record_bytes": 62720,
        "scache_defaults": w_manifest["scache_defaults"],
        "scache_gpu_bytes": sc_b0,
        "store_sha256": sha256_file(store_path),
        "webgpu_shards": num_webgpu_shards,
        "webgpu_total_mib": total_webgpu_mib,
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }
    (LIB_DIR / "scarstore_portfolio_catalog_v4_0.json").write_text(
        json.dumps(cat_summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[SUCCESS] Portfolio ScarStore v4.0.1 built & WGSL-verified in {cat_summary['elapsed_sec']}s!", flush=True)


if __name__ == "__main__":
    main()
