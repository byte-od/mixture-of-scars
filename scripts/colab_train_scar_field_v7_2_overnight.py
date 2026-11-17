#!/usr/bin/env python3
"""Overnight Deep Foundation Pretraining of `ScarField v7.2-Deep` on `NVIDIA A100` (Deadline-Aware + Checkpointed).

Designed for multi-hour continuous `Tabula-Rasa` pretraining until a target wall-clock deadline (e.g. `09:00 AM` local):
  1. Architecture (`ScarField v7.2-Deep`, `0` legacy Transformer layers):
       - `d_form=384, d_world=512, d_bus=384, d_key=128, ring_len=64, base_rank=12, num_scars=96`
       - Multi-scale 25-byte causal spherical wave `wave_lags=(1, 2, 3, 4, 6, 8, 12, 16, 24)` (`0` BPE, `0` nn.Linear)
       - `num_macro_cycles=2` compositional reversible symplectic Cayley flows (`O(1)` backprop activation memory)
       - `1.000 bpw` Gosset `E_8` + `2-bit` ternary scars (`~7.0 MiB` zero-pickle `.sfield.scar` vs `~56 MiB` FP16)
  2. Massive Multi-Domain Real Open Corpus (`~35-55 MiB` SHA-256 Verified UTF-8 Bytes, `0` Synthetic Prose):
       - `480+` full-length encyclopedic articles across `PT`, `EN`, `ES`, and `FR` (`120+` topics per language
         spanning Physics, Astronomy, Mathematics, CS, Biology, Medicine, Chemistry, Engineering, Philosophy,
         History, Literature, and Earth Sciences) harvested via parallel MediaWiki API threads.
       - Production `Python`, `C`, and `WGSL` systems/numerical source code (`stdlib`, `numpy`, `scar_*`).
       - Strict document-ordered `90% / 10%` split with `4,096 B` guard gap (`0` byte overlap).
  3. Fault-Tolerant Checkpointing & Exact Wall-Clock Cutoff:
       - Saves atomic resume checkpoints (`model` + `RiemannianMuon` state + live `.sfield.scar` + `progress.json`)
         every `checkpoint_every_sec` (`1200s = 20 min`), supporting seamless `--resume`.
       - Stops `ScarField` training cleanly at `deadline_epoch - finalize_reserve_sec` so matched baseline comparison,
         held-out evaluation (`PT/EN/ES/FR/Code`, unseen Passkeys `T=128..1024`, Zero-SGD `.uscar` + `0x00` GDPR wipe,
         `0x00` vacuum abstention), and `.sfield.scar` export complete strictly BEFORE the `09:00 AM` deadline!
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import tarfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import urllib.parse
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "scar_field").exists() else SCRIPT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scar_field import (
    RiemannianMuon,
    ScarFieldConfig,
    ScarFieldEngine,
    heal_all_e8_scars,
    project_rows_to_unit_sphere,
    tsallis15_spherical_cap_probs,
)


# ---------------------------------------------------------------------------
# 1. Broad 120-Topic-Per-Language Open Encyclopedic + Source-Code Corpus
# ---------------------------------------------------------------------------
DEEP_WIKI_TOPICS: Dict[str, List[str]] = {
    "pt": [
        # Physics, Astronomy & Math
        "Física", "Física_quântica", "Relatividade_geral", "Relatividade_restrita", "Termodinâmica",
        "Eletromagnetismo", "Mecânica_clássica", "Mecânica_dos_fluidos", "Óptica", "Física_nuclear",
        "Física_de_partículas", "Modelo_Padrão", "Entropia", "Ondas_gravitacionais", "Supercondutividade",
        "Astronomia", "Astrofísica", "Cosmologia", "Radioastronomia", "Sistema_Solar", "Via_Láctea",
        "Galáxia", "Buraco_negro", "Estrela_de_nêutrons", "Supernova", "Exoplaneta", "Telescópio_espacial_James_Webb",
        "Matemática", "Álgebra_linear", "Cálculo_infinitesimal", "Equação_diferencial", "Geometria_diferencial",
        "Topologia", "Teoria_dos_números", "Teoria_dos_grupos", "Teoria_das_probabilidades", "Estatística",
        "Análise_complexa", "Análise_de_Fourier", "Teoria_da_informação", "Otimização_matemática",
        # CS & Engineering
        "Ciência_da_computação", "Inteligência_artificial", "Aprendizado_de_máquina", "Rede_neural_artificial",
        "Algoritmo", "Estrutura_de_dados", "Criptografia", "Arquitetura_de_computadores", "Sistema_operativo",
        "Linux", "Compilador", "Linguagem_de_programação", "Python", "Linguagem_C", "Banco_de_dados",
        "Rede_de_computadores", "Internet", "Processamento_de_sinais", "Compressão_de_dados", "Robótica",
        "Engenharia_elétrica", "Engenharia_aeroespacial", "Semicondutor", "Transístor", "Microprocessador",
        # Chemistry, Biology, Medicine & Earth
        "Química", "Química_orgânica", "Química_inorgânica", "Físico-química", "Tabela_periódica", "Ligação_química",
        "Biologia", "Biologia_molecular", "Biologia_celular", "Genética", "Ácido_desoxirribonucleico", "ARN",
        "Proteína", "Enzima", "Fotossíntese", "Evolução", "Seleção_natural", "Ecologia", "Biodiversidade",
        "Neurociência", "Cérebro_humano", "Neurônio", "Eletroencefalografia", "Sistema_imunitário", "Medicina",
        "Cardiologia", "Fisiologia", "Geologia", "Tectónica_de_placas", "Climatologia", "Oceano_Atlântico",
        "Atmosfera_da_Terra", "Ciclo_da_água",
        # History, Philosophy & Literature
        "Filosofia", "Filosofia_da_ciência", "Epistemologia", "Lógica", "Ética", "Método_científico",
        "História_da_ciência", "História_da_matemática", "Revolução_Científica", "Iluminismo", "Renascimento",
        "História_do_Brasil", "História_de_Portugal", "Língua_portuguesa", "Literatura_brasileira",
        "Literatura_portuguesa", "Machado_de_Assis", "Luís_de_Camões", "Fernando_Pessoa", "Música_clássica",
    ],
    "en": [
        # Physics, Astronomy & Math
        "Physics", "Quantum_mechanics", "General_relativity", "Special_relativity", "Thermodynamics",
        "Electromagnetism", "Classical_mechanics", "Fluid_mechanics", "Optics", "Nuclear_physics",
        "Particle_physics", "Standard_Model", "Entropy", "Gravitational_wave", "Superconductivity",
        "Astronomy", "Astrophysics", "Physical_cosmology", "Radio_astronomy", "Solar_System", "Milky_Way",
        "Galaxy", "Black_hole", "Neutron_star", "Supernova", "Exoplanet", "James_Webb_Space_Telescope",
        "Mathematics", "Linear_algebra", "Calculus", "Differential_equation", "Differential_geometry",
        "Topology", "Number_theory", "Group_theory", "Probability_theory", "Statistics",
        "Complex_analysis", "Fourier_analysis", "Information_theory", "Mathematical_optimization",
        # CS & Engineering
        "Computer_science", "Artificial_intelligence", "Machine_learning", "Neural_network_(machine_learning)",
        "Algorithm", "Data_structure", "Cryptography", "Computer_architecture", "Operating_system",
        "Linux", "Compiler", "Programming_language", "Python_(programming_language)", "C_(programming_language)",
        "Database", "Computer_network", "Internet", "Signal_processing", "Data_compression", "Robotics",
        "Electrical_engineering", "Aerospace_engineering", "Semiconductor", "Transistor", "Microprocessor",
        # Chemistry, Biology, Medicine & Earth
        "Chemistry", "Organic_chemistry", "Inorganic_chemistry", "Physical_chemistry", "Periodic_table", "Chemical_bond",
        "Biology", "Molecular_biology", "Cell_biology", "Genetics", "DNA", "RNA",
        "Protein", "Enzyme", "Photosynthesis", "Evolution", "Natural_selection", "Ecology", "Biodiversity",
        "Neuroscience", "Human_brain", "Neuron", "Electroencephalography", "Immune_system", "Medicine",
        "Cardiology", "Physiology", "Geology", "Plate_tectonics", "Climatology", "Atlantic_Ocean",
        "Atmosphere_of_Earth", "Water_cycle",
        # History, Philosophy & Literature
        "Philosophy", "Philosophy_of_science", "Epistemology", "Logic", "Ethics", "Scientific_method",
        "History_of_science", "History_of_mathematics", "Scientific_Revolution", "Age_of_Enlightenment", "Renaissance",
        "Charles_Darwin", "Isaac_Newton", "Albert_Einstein", "James_Clerk_Maxwell", "English_literature",
        "William_Shakespeare", "Classical_music", "Architecture", "Economics",
    ],
    "es": [
        # Physics, Astronomy & Math
        "Física", "Mecánica_cuántica", "Relatividad_general", "Relatividad_especial", "Termodinámica",
        "Electromagnetismo", "Mecánica_clásica", "Mecánica_de_fluidos", "Óptica", "Física_nuclear",
        "Física_de_partículas", "Modelo_estándar_de_la_física_de_partículas", "Entropía", "Onda_gravitatoria", "Superconductividad",
        "Astronomía", "Astrofísica", "Cosmología", "Radioastronomía", "Sistema_solar", "Vía_Láctea",
        "Galaxia", "Agujero_negro", "Estrella_de_neutrones", "Supernova", "Exoplaneta", "Telescopio_espacial_James_Webb",
        "Matemáticas", "Álgebra_lineal", "Cálculo_infinitesimal", "Ecuación_diferencial", "Geometría_diferencial",
        "Topología", "Teoría_de_números", "Teoría_de_grupos", "Teoría_de_la_probabilidad", "Estadística",
        "Análisis_complejo", "Análisis_de_Fourier", "Teoría_de_la_información", "Optimización_(matemática)",
        # CS & Engineering
        "Ciencias_de_la_computación", "Inteligencia_artificial", "Aprendizaje_automático", "Red_neuronal_artificial",
        "Algoritmo", "Estructura_de_datos", "Criptografía", "Arquitectura_de_computadoras", "Sistema_operativo",
        "GNU/Linux", "Compilador", "Lenguaje_de_programación", "Python", "C_(lenguaje_de_programación)", "Base_de_datos",
        "Red_de_computadoras", "Internet", "Procesamiento_digital_de_señales", "Compresión_de_datos", "Robótica",
        "Ingeniería_eléctrica", "Ingeniería_aeroespacial", "Semiconductor", "Transistor", "Microprocesador",
        # Chemistry, Biology, Medicine & Earth
        "Química", "Química_orgánica", "Química_inorgánica", "Fisicoquímica", "Tabla_periódica_de_los_elementos", "Enlace_químico",
        "Biología", "Biología_molecular", "Biología_celular", "Genética", "Ácido_desoxirribonucleico", "Ácido_ribonucleico",
        "Proteína", "Enzima", "Fotosíntesis", "Evolución_biológica", "Selección_natural", "Ecología", "Biodiversidad",
        "Neurociencia", "Cerebro_humano", "Neurona", "Electroencefalografía", "Sistema_inmunitario", "Medicina",
        "Cardiología", "Fisiología", "Geología", "Tectónica_de_placas", "Climatología", "Océano_Pacífico",
        "Atmósfera_terrestre", "Ciclo_hidrológico",
        # History, Philosophy & Literature
        "Filosofía", "Filosofía_de_la_ciencia", "Epistemología", "Lógica", "Ética", "Método_científico",
        "Historia_de_la_ciencia", "Historia_de_las_matemáticas", "Revolución_científica", "Ilustración", "Renacimiento",
        "Idioma_español", "Literatura_en_español", "Miguel_de_Cervantes", "Santiago_Ramón_y_Cajal", "Jorge_Luis_Borges",
        "Gabriel_García_Márquez", "Música_clásica", "Arquitectura", "Economía",
    ],
    "fr": [
        # Physics, Astronomy & Math
        "Physique", "Mécanique_quantique", "Relativité_générale", "Relativité_restreinte", "Thermodynamique",
        "Électromagnétisme", "Mécanique_classique", "Mécanique_des_fluides", "Optique", "Physique_nucléaire",
        "Physique_des_particules", "Modèle_standard_de_la_physique_des_particules", "Entropie_(thermodynamique)", "Onde_gravitationnelle", "Supraconductivité",
        "Astronomie", "Astrophysique", "Cosmologie", "Radioastronomie", "Système_solaire", "Voie_lactée",
        "Galaxie", "Trou_noir", "Étoile_à_neutrons", "Supernova", "Exoplanète", "James-Webb_(télescope_spatial)",
        "Mathématiques", "Algèbre_linéaire", "Calcul_infinitésimal", "Équation_différentielle", "Géométrie_différentielle",
        "Topologie", "Théorie_des_nombres", "Théorie_des_groupes", "Théorie_des_probabilités", "Statistique",
        "Analyse_complexe", "Analyse_harmonique_(mathématiques)", "Théorie_de_l'information", "Optimisation_(mathématiques)",
        # CS & Engineering
        "Informatique", "Intelligence_artificielle", "Apprentissage_automatique", "Réseau_de_neurones_artificiels",
        "Algorithme", "Structure_de_données", "Cryptographie", "Architecture_des_ordinateurs", "Système_d'exploitation",
        "Linux", "Compilateur", "Langage_de_programmation", "Python_(langage)", "C_(langage)", "Base_de_données",
        "Réseau_informatique", "Internet", "Traitement_du_signal", "Compression_de_données", "Robotique",
        "Génie_électrique", "Génie_aérospatial", "Semi-conducteur", "Transistor", "Microprocesseur",
        # Chemistry, Biology, Medicine & Earth
        "Chimie", "Chimie_organique", "Chimie_inorganique", "Chimie_physique", "Tableau_périodique_des_éléments", "Liaison_chimique",
        "Biologie", "Biologie_moléculaire", "Biologie_cellulaire", "Génétique", "Acide_désoxyribonucléique", "Acide_ribonucléique",
        "Protéine", "Enzyme", "Photosynthèse", "Évolution_(biologie)", "Sélection_naturelle", "Écologie", "Biodiversité",
        "Neurosciences", "Cerveau_humain", "Neurone", "Électroencéphalographie", "Système_immunitaire", "Médecine",
        "Cardiologie", "Physiologie", "Géologie", "Tectonique_des_plaques", "Climatologie", "Océan_Atlantique",
        "Atmosphère_terrestre", "Cycle_de_l'eau",
        # History, Philosophy & Literature
        "Philosophie", "Philosophie_des_sciences", "Épistémologie", "Logique", "Éthique", "Méthode_scientifique",
        "Histoire_des_sciences", "Histoire_des_mathématiques", "Révolution_scientifique", "Siècle_des_Lumières", "Renaissance",
        "Langue_française", "Littérature_française", "Henri_Poincaré", "René_Descartes", "Pierre-Simon_de_Laplace",
        "Jules_Verne", "Marie_Curie", "Musique_classique", "Architecture",
    ],
}


def _fetch_one_wiki_article(args: Tuple[str, str]) -> Tuple[str, str]:
    lg, topic = args
    params = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": "1",
            "redirects": "1",
            "format": "json",
            "titles": topic.replace("_", " "),
        }
    )
    url = f"https://{lg}.wikipedia.org/w/api.php?{params}"
    for attempt in range(3):
        time.sleep(0.08 + 0.4 * attempt)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "ScarFieldOvernightResearch/7.2 (https://github.com/byte-od/mixture-of-scars)"},
        )
        try:
            with urllib.request.urlopen(req, timeout=14.0) as resp:
                js = json.loads(resp.read().decode("utf-8", errors="ignore"))
            pages = js.get("query", {}).get("pages", {})
            for _, pinfo in pages.items():
                ext = pinfo.get("extract", "").strip()
                title = pinfo.get("title", topic)
                if len(ext) > 450:
                    return topic, f"# {title}\n{ext}\n"
            break
        except Exception:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
    return topic, ""


def fetch_parallel_wikipedia_corpus(lg: str, topics: List[str], workers: int = 2) -> str:
    dedup_topics = list(dict.fromkeys(topics))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_fetch_one_wiki_article, [(lg, t) for t in dedup_topics]))
    valid = [txt for _, txt in results if txt]
    return "\n".join(valid)


def harvest_real_code_corpus(max_bytes: int = 12_000_000) -> Tuple[bytes, List[Dict[str, Any]]]:
    chunks: List[bytes] = []
    manifest: List[Dict[str, Any]] = []
    total_b = 0

    search_roots: List[Tuple[str, Path]] = [("scar_ecosystem", REPO_ROOT)]
    stdlib_dir = Path(inspect.getfile(tarfile)).resolve().parent
    search_roots.append(("python_stdlib", stdlib_dir))
    np_dir = Path(inspect.getfile(np)).resolve().parent
    search_roots.append(("numpy_src", np_dir))
    torch_nn_dir = Path(inspect.getfile(nn)).resolve().parent
    search_roots.append(("pytorch_nn_src", torch_nn_dir))

    for tag, root in search_roots:
        if not root.exists():
            continue
        files = sorted(
            list(root.rglob("*.py"))
            + list(root.rglob("*.c"))
            + list(root.rglob("*.h"))
            + list(root.rglob("*.wgsl"))
        )
        tag_bytes = 0
        tag_hasher = hashlib.sha256()
        for fp in files:
            s_fp = str(fp)
            if "__pycache__" in s_fp or (tag == "scar_ecosystem" and (".venv" in s_fp or "site-packages" in s_fp)):
                continue
            try:
                raw = fp.read_bytes()
            except Exception:
                continue
            if len(raw) < 400 or len(raw) > 260_000:
                continue
            try:
                txt = raw.decode("utf-8")
            except UnicodeDecodeError:
                continue
            b_clean = (f"# FILE: {fp.name}\n" + txt + "\n\n").encode("utf-8")
            chunks.append(b_clean)
            tag_hasher.update(b_clean)
            tag_bytes += len(b_clean)
            total_b += len(b_clean)
            if total_b >= max_bytes:
                break
        if tag_bytes > 0:
            manifest.append({"source": tag, "bytes": tag_bytes, "sha256": tag_hasher.hexdigest()})
        if total_b >= max_bytes:
            break

    return b"".join(chunks), manifest


def build_overnight_open_corpora(
    cache_dir: Path,
    dry_run: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict[str, Any]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    provenance: List[Dict[str, Any]] = []
    raw_domains: Dict[str, bytes] = {}

    if dry_run:
        code_b, code_m = harvest_real_code_corpus(max_bytes=200_000)
        provenance.extend(code_m)
        raw_domains["code"] = code_b
        seeds = {
            "pt": "A mecânica quântica e a relatividade geral descrevem escalas fundamentais do universo observável com precisão matemática rigorosa.\n" * 450,
            "en": "Quantum mechanics and general relativity describe fundamental scales of the observable universe with rigorous mathematical precision.\n" * 450,
            "es": "La mecánica cuántica y la relatividad general describen escalas fundamentales del universo observable con precisión matemática rigurosa.\n" * 450,
            "fr": "La mécanique quantique et la relativité générale décrivent les échelles fondamentales de l'univers observable avec une précision mathématique rigurosa.\n" * 450,
        }
        for lg, t in seeds.items():
            raw_domains[lg] = t.encode("utf-8")
    else:
        for lg in ("pt", "en", "es", "fr"):
            wiki_cache = cache_dir / f"wikipedia_deep120_{lg}_v72.txt"
            if wiki_cache.exists() and wiki_cache.stat().st_size > 500_000:
                wiki_txt = wiki_cache.read_text(encoding="utf-8")
            else:
                wiki_txt = fetch_parallel_wikipedia_corpus(lg, DEEP_WIKI_TOPICS[lg], workers=2)
                if len(wiki_txt) > 100_000:
                    wiki_cache.write_text(wiki_txt, encoding="utf-8")
            wb = (wiki_txt + "\n\n").encode("utf-8")
            if len(wb) < 100_000:
                raise RuntimeError(f"Insufficient `{lg}` open encyclopedic corpus ({len(wb)} B)!")
            raw_domains[lg] = wb
            provenance.append(
                {
                    "domain": lg,
                    "title": f"Wikimedia Wikipedia Deep Open Corpus ({lg.upper()}, {len(DEEP_WIKI_TOPICS[lg])} topics)",
                    "bytes": len(wb),
                    "sha256": hashlib.sha256(wb).hexdigest(),
                }
            )

        code_b, code_m = harvest_real_code_corpus(max_bytes=10_500_000)
        provenance.extend(code_m)
        raw_domains["code"] = code_b

    train_splits: Dict[str, np.ndarray] = {}
    val_splits: Dict[str, np.ndarray] = {}
    guard_gap = 512 if dry_run else 4096
    for dom, b_all in raw_domains.items():
        arr = np.frombuffer(b_all, dtype=np.uint8).copy()
        cut = int(len(arr) * 0.90)
        train_splits[dom] = arr[: max(2048, cut - guard_gap)]
        val_splits[dom] = arr[min(len(arr) - 1024, cut) :]

    return train_splits, val_splits, provenance


# ---------------------------------------------------------------------------
# 2. Cross-Lingual Anchors, Passkey Generator & Matched DenseByteTransformer
# ---------------------------------------------------------------------------
PARALLEL_SCIENTIFIC_ANCHORS: List[Dict[str, str]] = [
    {
        "pt": "A entropia termodinâmica mede o número de microestados compatíveis com o equilíbrio macroscópico.",
        "en": "Thermodynamic entropy measures the number of microstates compatible with macroscopic equilibrium.",
        "es": "La entropía termodinámica mide el número de microestados compatibles con el equilibrio macroscópico.",
        "fr": "L'entropie thermodynamique mesure le nombre de micro-états compatibles avec l'équilibre macroscopique.",
    },
    {
        "pt": "As equações de Maxwell unificam os campos elétrico e magnético como ondas eletromagnéticas no vácuo.",
        "en": "Maxwell equations unify electric and magnetic fields as electromagnetic waves propagating in vacuum.",
        "es": "Las ecuaciones de Maxwell unifican los campos eléctrico y magnético como ondas electromagnéticas en el vacío.",
        "fr": "Les équations de Maxwell unifient les champs électrique et magnétique en ondes électromagnétiques dans le vide.",
    },
    {
        "pt": "A curvatura do espaço-tempo na relatividade geral é determinada pelo tensor energia-momento da matéria.",
        "en": "Spacetime curvature in general relativity is governed by the stress-energy-momentum tensor of matter.",
        "es": "La curvatura del espaciotiempo en la relatividad general está gobernada por el tensor energía-momento.",
        "fr": "La courbure de l'espace-temps en relativité générale est régie par le tenseur énergie-impulsion.",
    },
    {
        "pt": "Árvores binárias balanceadas garantem busca, inserção e remoção em tempo logarítmico no pior caso.",
        "en": "Balanced binary search trees guarantee logarithmic worst-case lookup, insertion, and deletion time.",
        "es": "Los árboles binarios balanceados garantizan búsqueda, inserción y borrado en tiempo logarítmico.",
        "fr": "Les arbres binaires équilibrés garantissent la recherche, l'insertion et la suppression en temps logarithmique.",
    },
    {
        "pt": "O código genético traduz trincas de nucleotídeos do RNA mensageiro em cadeias específicas de aminoácidos.",
        "en": "The genetic code translates messenger RNA nucleotide triplets into specific polypeptide amino acid chains.",
        "es": "El código genético traduce tripletes de nucleótidos del ARN mensajero en cadenas de aminoácidos.",
        "fr": "Le code génétique traduit les triplets de nucléotides de l'ARN messager en chaînes d'acides aminés.",
    },
    {
        "pt": "A transformada rápida de Fourier reduz a complexidade espectral quadrática para quase linear.",
        "en": "The fast Fourier transform reduces quadratic discrete spectral complexity to log-linear time.",
        "es": "La transformada rápida de Fourier reduce la complejidad espectral cuadrática a tiempo casi lineal.",
        "fr": "La transformée de Fourier rapide réduit la complexité spectrale quadratique en temps quasi linéaire.",
    },
]

OUT_OF_CONE_QUERIES: List[str] = [
    "Qual é o horóscopo astral secreto de amanhã para ganhar na loteria acumulada?",
    "What is the personal home banking password of the prime minister right now?",
    "¿Cuál es el número ganador exacto del sorteo privado de mañana por la noche?",
    "Quel est le code secret privé du coffre-fort personnel d'un inconnu aujourd'hui?",
    "Me receite uma poção mágica caseira para ficar invisível durante três horas.",
    "Give me insider stock prices from next Friday afternoon before markets open.",
    "¿Qué pensó en silencio el vecino del quinto piso hace exactamente diez minutos?",
    "Donne-moi les numéros gagnants garantis du prochain tirage de la loterie nationale.",
]


def _encode_padded_batch(texts: List[str], seq_len: int, device: torch.device) -> torch.Tensor:
    rows = []
    for t in texts:
        b = list(t.encode("utf-8"))
        if len(b) < seq_len:
            b = b + [32] * (seq_len - len(b))
        else:
            b = b[:seq_len]
        rows.append(b)
    return torch.tensor(rows, dtype=torch.long, device=device)


def sample_open_passkey_batch(
    prose_arr: np.ndarray,
    batch_size: int,
    seq_len: int,
    rng: random.Random,
    pin_min: int,
    pin_max: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int], List[str]]:
    rows: List[List[int]] = []
    q_positions: List[int] = []
    pins: List[str] = []
    max_off = max(1, len(prose_arr) - seq_len - 64)

    for _ in range(batch_size):
        pin = f"{rng.randint(pin_min, pin_max):03d}"
        prefix = f"[PASSKEY PIN={pin}] ".encode("utf-8")
        suffix = f" [QUERY PIN=> {pin}.]".encode("utf-8")
        need_filler = max(8, seq_len - len(prefix) - len(suffix))
        off = rng.randint(0, max_off)
        raw_fill = bytes(prose_arr[off : off + need_filler]).replace(b"\n", b" ")
        full = (prefix + raw_fill[:need_filler] + suffix)[:seq_len]
        if len(full) < seq_len:
            full = full + b" " * (seq_len - len(full))
        q_pos = len(prefix) + need_filler + len(b" [QUERY PIN=> ") - 1
        rows.append(list(full))
        q_positions.append(q_pos)
        pins.append(pin)

    return torch.tensor(rows, dtype=torch.long, device=device), q_positions, pins


class DenseCausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window_size: int = 0) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(bsz, seq_len, 3, self.n_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(float(self.head_dim))
        i_idx = torch.arange(seq_len, device=x.device).unsqueeze(1)
        j_idx = torch.arange(seq_len, device=x.device).unsqueeze(0)
        mask = j_idx <= i_idx
        if self.window_size > 0:
            mask = mask & (j_idx >= (i_idx - self.window_size + 1))
        att = att.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        probs = F.softmax(att, dim=-1)
        out = (probs @ v).transpose(1, 2).reshape(bsz, seq_len, self.d_model)
        return self.proj(out)


class DenseByteBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window_size: int = 0) -> None:
        super().__init__()
        self.ln1 = nn.RMSNorm(d_model)
        self.attn = DenseCausalAttention(d_model, n_heads, window_size=window_size)
        self.ln2 = nn.RMSNorm(d_model)
        hid = int(d_model * 8 // 3)
        hid = ((hid + 63) // 64) * 64
        self.gate_proj = nn.Linear(d_model, hid, bias=False)
        self.up_proj = nn.Linear(d_model, hid, bias=False)
        self.down_proj = nn.Linear(hid, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        h = self.ln2(x)
        x = x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))
        return x


class DenseByteTransformer(nn.Module):
    def __init__(self, vocab_size: int = 256, d_model: int = 512, n_layers: int = 10, n_heads: int = 8) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(2048, d_model)
        self.blocks = nn.ModuleList([DenseByteBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = byte_ids.shape
        pos = torch.arange(seq_len, device=byte_ids.device).clamp(max=2047)
        x = self.tok_emb(byte_ids.long()) + self.pos_emb(pos).unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.ln_f(x))


# ---------------------------------------------------------------------------
# 3. Evaluation Utilities (PPM-Calibrated BPB, Passkey, .uscar GDPR, Samples)
# ---------------------------------------------------------------------------
@torch.no_grad()
def calibrate_ppm_escape_lambda_on_train(
    forward_probs_fn: Callable[[torch.Tensor], torch.Tensor],
    train_arr: np.ndarray,
    seq_len: int,
    num_chunks: int,
    device: torch.device,
) -> float:
    total_toks = 0
    zero_cap_misses = 0
    stride = max(seq_len, (len(train_arr) - seq_len - 1) // max(1, num_chunks))
    for i in range(num_chunks):
        off = min(i * stride, len(train_arr) - seq_len - 1)
        chunk = torch.from_numpy(train_arr[off : off + seq_len].astype(np.int64)).unsqueeze(0).to(device)
        probs = forward_probs_fn(chunk[:, :-1])
        targets = chunk[:, 1:]
        p_t = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        zero_cap_misses += int((p_t <= 0.0).sum().item())
        total_toks += int(targets.numel())
    emp = (zero_cap_misses + 1.0) / float(max(1, total_toks) + 2.0)
    return float(min(0.35, max(1e-4, emp)))


@torch.no_grad()
def evaluate_byte_lm_metrics(
    forward_probs_fn: Callable[[torch.Tensor], torch.Tensor],
    val_arr: np.ndarray,
    seq_len: int,
    num_chunks: int,
    device: torch.device,
    is_sparse_tsallis: bool = False,
    escape_lambda: float = 0.02,
) -> Dict[str, float]:
    total_bits = 0.0
    total_toks = 0
    top1_hits = 0
    top5_hits = 0
    in_cap_hits = 0
    exact_zero_frac_sum = 0.0

    stride = max(seq_len, (len(val_arr) - seq_len - 1) // max(1, num_chunks))
    bsz = 8
    batches: List[np.ndarray] = []
    for i in range(num_chunks):
        off = min(i * stride, len(val_arr) - seq_len - 1)
        batches.append(val_arr[off : off + seq_len].astype(np.int64))

    for b0 in range(0, len(batches), bsz):
        chunk = torch.from_numpy(np.stack(batches[b0 : b0 + bsz], axis=0)).to(device)
        probs = forward_probs_fn(chunk[:, :-1])
        targets = chunk[:, 1:]
        v_dim = probs.shape[-1]

        exact_zero_frac_sum += float((probs == 0.0).float().mean().item()) * targets.numel()
        p_cap_t = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        in_cap_hits += int((p_cap_t > 0.0).sum().item())

        if is_sparse_tsallis:
            lam = float(min(0.5, max(1e-5, escape_lambda)))
            n_zeros = (probs == 0.0).float().sum(dim=-1).clamp(min=1.0)
            p_true = torch.where(p_cap_t > 0.0, (1.0 - lam) * p_cap_t, lam / n_zeros)
        else:
            p_true = p_cap_t.clamp(min=1e-9)

        total_bits += float((-torch.log2(p_true)).sum().item())
        top1_hits += int((probs.argmax(dim=-1) == targets).sum().item())
        top5 = probs.topk(k=min(5, v_dim), dim=-1).indices
        top5_hits += int((top5 == targets.unsqueeze(-1)).any(dim=-1).sum().item())
        total_toks += int(targets.numel())

    return {
        "eval_bytes": total_toks,
        "calibrated_bpb" if is_sparse_tsallis else "bpb": round(total_bits / max(1, total_toks), 4),
        **({"train_escape_lambda": round(escape_lambda, 5), "in_cap_coverage_pct": round(100.0 * in_cap_hits / max(1, total_toks), 2)} if is_sparse_tsallis else {}),
        "top1_byte_acc_pct": round(100.0 * top1_hits / max(1, total_toks), 2),
        "top5_byte_acc_pct": round(100.0 * top5_hits / max(1, total_toks), 2),
        "exact_zero_sparsity_pct": round(100.0 * exact_zero_frac_sum / max(1, total_toks), 2),
    }


@torch.no_grad()
def evaluate_open_passkey_recall(
    predict_logits_fn: Callable[[torch.Tensor], torch.Tensor],
    prose_val: np.ndarray,
    seq_len: int,
    n_trials: int,
    device: torch.device,
) -> Dict[str, Any]:
    rng = random.Random(91000 + seq_len)
    batch, q_pos, pins = sample_open_passkey_batch(
        prose_val, n_trials, seq_len, rng, pin_min=500, pin_max=999, device=device
    )
    scores = predict_logits_fn(batch)
    exact_3 = 0
    digit_hits = 0
    for b in range(n_trials):
        qp = q_pos[b]
        pred_digits = []
        for d in range(3):
            pred_b = int(scores[b, qp + d].argmax().item())
            pred_digits.append(chr(pred_b) if 32 <= pred_b < 127 else "?")
            if pred_b == ord(pins[b][d]):
                digit_hits += 1
        if "".join(pred_digits) == pins[b]:
            exact_3 += 1
    return {
        "seq_len_bytes": seq_len,
        "exact_3digit_acc": round(exact_3 / float(n_trials), 4),
        "per_digit_acc": round(digit_hits / float(3 * n_trials), 4),
    }


@torch.no_grad()
def generate_continuation_sample(
    eng: ScarFieldEngine,
    prompt: str,
    gen_bytes: int,
    device: torch.device,
    use_scouts: bool = True,
) -> str:
    b_in = list(prompt.encode("utf-8"))
    if len(b_in) > 1:
        pref_t = torch.tensor([b_in[:-1]], dtype=torch.long, device=device)
        st = eng.forward_sequence(pref_t, adaptive_depth=True)["state"]
    else:
        st = eng.init_state(1, device=device)
    last_b = b_in[-1]
    out_bytes = list(b_in)
    for _ in range(gen_bytes):
        dec = eng.step_byte_interactive(
            last_b,
            st,
            enable_ghost_jump_scouts=use_scouts,
            require_world_support=False,
            apply_repulsive_anti_loop=True,
        )
        out_bytes.append(dec.action_byte)
        last_b = dec.action_byte
    return bytes(out_bytes).decode("utf-8", errors="replace")


def save_atomic_checkpoint(
    ckpt_path: Path,
    live_scar_path: Path,
    progress_json_path: Path,
    sf_eng: ScarFieldEngine,
    sf_opt: RiemannianMuon,
    step: int,
    total_trained_bytes: int,
    sf_history: List[Dict[str, Any]],
    elapsed_train_sec: float,
) -> None:
    tmp_ckpt = ckpt_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "step": step,
            "total_trained_bytes": total_trained_bytes,
            "elapsed_train_sec": elapsed_train_sec,
            "sf_cfg": sf_eng.cfg.to_dict(),
            "model_state": sf_eng.state_dict(),
            "opt_state": sf_opt.state_dict(),
            "sf_history": sf_history,
        },
        tmp_ckpt,
    )
    os.replace(tmp_ckpt, ckpt_path)

    tmp_scar = live_scar_path.with_suffix(".scar.tmp")
    scar_bytes = sf_eng.save_packed_scar_container(tmp_scar)
    os.replace(tmp_scar, live_scar_path)

    tmp_prog = progress_json_path.with_suffix(".json.tmp")
    tmp_prog.write_text(
        json.dumps(
            {
                "updated_epoch_utc": round(time.time(), 2),
                "step": step,
                "total_trained_bytes": total_trained_bytes,
                "elapsed_train_sec": round(elapsed_train_sec, 1),
                "packed_sfield_scar_mib": round(scar_bytes / (1024**2), 3),
                "latest_metrics": sf_history[-1] if sf_history else {},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(tmp_prog, progress_json_path)


# ---------------------------------------------------------------------------
# 4. Main Overnight Deadline-Aware Pretraining & Evaluation Pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Overnight Deep Pretraining of ScarField v7.2-Deep on NVIDIA A100")
    parser.add_argument("--dry-run", action="store_true", help="Fast 5-step local smoke check")
    parser.add_argument("--stop-epoch-utc", type=float, default=0.0, help="Hard UTC epoch deadline (e.g. 09:00 AM -03:00)")
    parser.add_argument("--finalize-reserve-sec", type=float, default=1200.0, help="Seconds reserved before deadline for eval + export")
    parser.add_argument("--max-steps", type=int, default=60000, help="Safety cap on optimizer steps")
    parser.add_argument("--dense-steps", type=int, default=3200, help="Matched DenseByteTransformer reference steps")
    parser.add_argument("--checkpoint-every-sec", type=float, default=1200.0, help="Periodic checkpoint interval (default 20 min)")
    parser.add_argument("--resume", type=str, default="", help="Optional path to resume checkpoint .pt")
    parser.add_argument("--out-dir", type=str, default="", help="Output directory")
    args = parser.parse_args()

    dry_run = bool(args.dry_run)
    torch.manual_seed(20260702)
    np.random.seed(20260702)
    py_rng = random.Random(20260702)

    dev = torch.device("cuda" if torch.cuda.is_available() and not dry_run else "cpu")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gpu_name = torch.cuda.get_device_name(0)
    else:
        gpu_name = "CPU"

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (
            Path("/content/scarfield_v7_2_overnight_out")
            if Path("/content").exists() and not dry_run
            else REPO_ROOT / "artifacts" / "scarfield_v7_2_overnight"
        )
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "corpus_cache"

    now_utc = time.time()
    if dry_run:
        train_cutoff_utc = now_utc + 60.0
        max_steps = 4
        dense_steps = 4
    elif args.stop_epoch_utc > now_utc + args.finalize_reserve_sec + 120.0:
        train_cutoff_utc = float(args.stop_epoch_utc) - float(args.finalize_reserve_sec)
        max_steps = int(args.max_steps)
        dense_steps = int(args.dense_steps)
    else:
        # Default 6.5-hour window if no explicit epoch passed
        train_cutoff_utc = now_utc + 6.5 * 3600.0
        max_steps = int(args.max_steps)
        dense_steps = int(args.dense_steps)

    print("=" * 92, flush=True)
    print(f"SCARFIELD v7.2-DEEP OVERNIGHT — MULTI-SCALE SPHERICAL WAVE + 2-CYCLE CAYLEY ({dev})", flush=True)
    print(
        f"Hardware: {gpu_name} | Remaining Train Window: {(train_cutoff_utc - time.time()) / 3600.0:.2f} hours "
        f"(cap {max_steps} steps)",
        flush=True,
    )
    print("=" * 92, flush=True)

    t_global = time.perf_counter()
    print("\n[1/4] Harvesting Deep 480-Article Multilingual (`PT/EN/ES/FR`) + Production Code Corpora...", flush=True)
    train_splits, val_splits, provenance = build_overnight_open_corpora(cache_dir, dry_run=dry_run)
    total_train_bytes = int(sum(len(v) for v in train_splits.values()))
    total_val_bytes = int(sum(len(v) for v in val_splits.values()))
    for dom in ("pt", "en", "es", "fr", "code"):
        print(f"  Domain `{dom:4s}` -> Train: {len(train_splits[dom]):10,d} B | Val (Held-Out): {len(val_splits[dom]):9,d} B", flush=True)
    print(
        f"  TOTAL SHA-256 VERIFIED OPEN CORPUS: {(total_train_bytes + total_val_bytes) / (1024**2):.2f} MiB "
        f"({len(provenance)} sources)",
        flush=True,
    )

    if dry_run:
        sf_cfg = ScarFieldConfig(
            vocab_size=256, d_form=128, d_world=192, d_bus=128, d_key=64,
            num_subspaces=8, ring_len=32, repulsive_len=8, base_rank=6, num_scars=24,
            s1_steps=1, s2_steps=2, wy_chunk_size=32,
            wave_lags=(1, 2, 3, 4, 6, 8, 12, 16), num_macro_cycles=2,
        )
        dense_d, dense_l, dense_h = 192, 4, 6
        bsz = 4
        eval_chunks = 2
    else:
        sf_cfg = ScarFieldConfig(
            vocab_size=256, d_form=384, d_world=512, d_bus=384, d_key=128,
            num_subspaces=8, ring_len=64, repulsive_len=12, base_rank=12, num_scars=96,
            s1_steps=2, s2_steps=4, s2_surprise_theta=0.45, shock_write_theta=0.50,
            vacuum_kappa=0.36, wy_chunk_size=64, enable_kalman_fast=True,
            wave_lags=(1, 2, 3, 4, 6, 8, 12, 16, 24), num_macro_cycles=2,
        )
        dense_d, dense_l, dense_h = 512, 10, 8
        bsz = 32
        eval_chunks = 80

    sf_eng = ScarFieldEngine(sf_cfg).to(dev)
    sf_eng.assert_zero_legacy_modules()
    sf_budget = sf_eng.byte_budget_summary()
    sf_param_count = sum(p.numel() for p in sf_eng.parameters())

    print("\n[2/4] Continuous Overnight `ScarField v7.2-Deep` Pretraining...", flush=True)
    print(
        f"  [ScarField v7.2-Deep] Spherical Params: {sf_param_count:,d} | "
        f"Packed E8 Budget: {sf_budget['total_packed_model_bytes'] / (1024**2):.3f} MiB | "
        f"Central Kalman-Scar O(1): {sf_budget['central_kalman_scache_o1_bytes'] / 1024:.1f} KiB (Delta = +0 B)",
        flush=True,
    )

    sf_opt = RiemannianMuon(sf_eng.parameters(), lr=6.0e-3, scalar_lr=3.5e-3, momentum=0.92, ns_steps=5)

    ckpt_path = out_dir / "scarfield_v7_2_checkpoint.pt"
    live_scar_path = out_dir / "scarfield_v7_2_live.sfield.scar"
    progress_json_path = out_dir / "scarfield_v7_2_progress.json"

    start_step = 1
    total_trained_bytes = 0
    prior_elapsed_sec = 0.0
    sf_history: List[Dict[str, Any]] = []

    resume_target = Path(args.resume) if args.resume else ckpt_path
    if resume_target.exists() and not dry_run:
        print(f"  [Resume] Restoring checkpoint from {resume_target} ...", flush=True)
        ck = torch.load(resume_target, map_location=dev, weights_only=False)
        sf_eng.load_state_dict(ck["model_state"])
        sf_opt.load_state_dict(ck["opt_state"])
        start_step = int(ck["step"]) + 1
        total_trained_bytes = int(ck["total_trained_bytes"])
        prior_elapsed_sec = float(ck.get("elapsed_train_sec", 0.0))
        sf_history = list(ck.get("sf_history", []))
        print(
            f"  [Resume] Resumed cleanly at step={start_step}, streamed_bytes={total_trained_bytes:,d} B!",
            flush=True,
        )

    anch_pt = _encode_padded_batch([a["pt"] for a in PARALLEL_SCIENTIFIC_ANCHORS], 128, dev)
    anch_en = _encode_padded_batch([a["en"] for a in PARALLEL_SCIENTIFIC_ANCHORS], 128, dev)
    anch_es = _encode_padded_batch([a["es"] for a in PARALLEL_SCIENTIFIC_ANCHORS], 128, dev)
    anch_fr = _encode_padded_batch([a["fr"] for a in PARALLEL_SCIENTIFIC_ANCHORS], 128, dev)
    ooc_tr = _encode_padded_batch(OUT_OF_CONE_QUERIES[:4], 96, dev)
    ooc_ho = _encode_padded_batch(OUT_OF_CONE_QUERIES[4:], 96, dev)

    # Balanced sampling: 75% multilingual prose (`PT/EN/ES/FR` equally) + 25% `Code`
    weighted_domains = ("pt", "en", "es", "fr", "pt", "en", "es", "fr", "pt", "en", "es", "fr", "code", "code", "code", "code")
    all_prose_concat = np.concatenate([train_splits[k] for k in ("pt", "en", "es", "fr")])

    def sample_mixed_batch(seq_l: int) -> torch.Tensor:
        rows_np = []
        for _ in range(bsz):
            dom = py_rng.choice(weighted_domains)
            arr = train_splits[dom]
            off = py_rng.randint(0, max(1, len(arr) - seq_l - 1))
            rows_np.append(arr[off : off + seq_l])
        return torch.from_numpy(np.stack(rows_np, axis=0).astype(np.int64)).to(dev)

    t_sf_start = time.perf_counter()
    last_ckpt_wall = time.time()
    total_planned_sec = max(60.0, (train_cutoff_utc - time.time()) + prior_elapsed_sec)

    step = start_step - 1
    while step < max_steps and time.time() < train_cutoff_utc:
        step += 1
        cur_elapsed = prior_elapsed_sec + (time.perf_counter() - t_sf_start)
        frac = min(1.0, max(0.0, cur_elapsed / total_planned_sec))

        # Smooth time-aware curriculum `L = 192 -> 320 -> 448 -> 576` & cosine LR schedule
        if dry_run:
            seq_l = 64
        elif frac < 0.25:
            seq_l = 192
        elif frac < 0.55:
            seq_l = 320
        elif frac < 0.82:
            seq_l = 448
        else:
            seq_l = 576

        lr_mult = 0.5 * (1.0 + math.cos(math.pi * frac))
        cur_lr = 1.2e-3 + (6.0e-3 - 1.2e-3) * lr_mult
        for pg in sf_opt.param_groups:
            pg["lr"] = cur_lr
            pg["scalar_lr"] = cur_lr * 0.58

        sf_opt.zero_grad(set_to_none=True)

        if (not dry_run) and (step % 6 == 0):
            pk_batch, _, _ = sample_open_passkey_batch(
                all_prose_concat, bsz // 2, seq_l, py_rng, pin_min=100, pin_max=499, device=dev
            )
            lm_half = sample_mixed_batch(seq_l)[: bsz - (bsz // 2)]
            batch_ids = torch.cat((lm_half, pk_batch), dim=0)
        else:
            batch_ids = sample_mixed_batch(seq_l)

        losses = sf_eng.compute_loss(batch_ids)
        loss_total = losses["loss"]

        if step % 5 == 0 or dry_run:
            src_a = py_rng.choice((anch_pt, anch_es, anch_fr))
            al = sf_eng.compute_loss(src_a, paired_translation_ids=anch_en)["loss_align"]
            ooc_mask = torch.ones_like(ooc_tr, dtype=torch.bool)
            vac = sf_eng.compute_loss(ooc_tr, out_of_cone_mask=ooc_mask)["loss_vacuum"]
            loss_total = loss_total + 0.45 * al + 0.45 * vac

        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(sf_eng.parameters(), 2.0)
        sf_opt.step()
        total_trained_bytes += int(batch_ids.numel())

        if step % 150 == 0 and not dry_run:
            heal_all_e8_scars(sf_eng)

        if step == start_step or step % (2 if dry_run else 200) == 0:
            rem_min = max(0.0, (train_cutoff_utc - time.time()) / 60.0)
            rec = {
                "step": step,
                "seq_len": seq_l,
                "lr": round(cur_lr, 6),
                "loss_cap": round(float(losses["loss_cap"].item()), 4),
                "loss_reflex": round(float(losses["loss_reflex"].item()), 4),
                "streamed_mib": round(total_trained_bytes / (1024**2), 2),
                "elapsed_s": round(cur_elapsed, 1),
                "remaining_min": round(rem_min, 1),
            }
            sf_history.append(rec)
            print(
                f"  [ScarField v7.2] step={step:6d} (L={seq_l}) lr={cur_lr:.4f} "
                f"cap={rec['loss_cap']:.4f} reflex={rec['loss_reflex']:.4f} "
                f"streamed={rec['streamed_mib']:.1f} MiB | elapsed={cur_elapsed/60.0:.1f}m | rem={rem_min:.1f}m",
                flush=True,
            )

        if (not dry_run) and (time.time() - last_ckpt_wall >= args.checkpoint_every_sec):
            heal_all_e8_scars(sf_eng)
            save_atomic_checkpoint(
                ckpt_path,
                live_scar_path,
                progress_json_path,
                sf_eng,
                sf_opt,
                step,
                total_trained_bytes,
                sf_history,
                prior_elapsed_sec + (time.perf_counter() - t_sf_start),
            )
            last_ckpt_wall = time.time()
            print(f"  [Checkpoint] Saved atomic checkpoint & live `.sfield.scar` at step={step}", flush=True)

    heal_all_e8_scars(sf_eng)
    sf_train_sec = prior_elapsed_sec + (time.perf_counter() - t_sf_start)
    save_atomic_checkpoint(
        ckpt_path,
        live_scar_path,
        progress_json_path,
        sf_eng,
        sf_opt,
        step,
        total_trained_bytes,
        sf_history,
        sf_train_sec,
    )
    print(
        f"\n  [ScarField v7.2-Deep Complete] Finished {step:,d} steps ({total_trained_bytes / (1024**2):.2f} MiB streamed) "
        f"in {sf_train_sec / 60.0:.2f} minutes!",
        flush=True,
    )

    # Matched `10L` `DenseByteTransformer` (`d=512`, `~55 MiB` FP16) reference
    dense_model = DenseByteTransformer(vocab_size=256, d_model=dense_d, n_layers=dense_l, n_heads=dense_h).to(dev)
    dense_param_count = sum(p.numel() for p in dense_model.parameters())
    dense_fp16_bytes = dense_param_count * 2
    print(
        f"\n  [DenseByteTransformer Reference] Params: {dense_param_count:,d} | FP16 Weights: {dense_fp16_bytes / (1024**2):.2f} MiB "
        f"| Steps: {dense_steps}",
        flush=True,
    )
    dense_opt = torch.optim.AdamW(dense_model.parameters(), lr=1.2e-3, weight_decay=0.01)
    t_dn = time.perf_counter()
    for d_step in range(1, dense_steps + 1):
        dfrac = (d_step - 1) / float(max(1, dense_steps - 1))
        d_seql = 64 if dry_run else (192 if dfrac < 0.30 else (320 if dfrac < 0.65 else 448))
        dlr = 2.0e-4 + (1.2e-3 - 2.0e-4) * (0.5 * (1.0 + math.cos(math.pi * dfrac)))
        for pg in dense_opt.param_groups:
            pg["lr"] = dlr
        dense_opt.zero_grad(set_to_none=True)
        if (not dry_run) and (d_step % 6 == 0):
            pk_b, _, _ = sample_open_passkey_batch(
                all_prose_concat, bsz // 2, d_seql, py_rng, pin_min=100, pin_max=499, device=dev
            )
            lm_h = sample_mixed_batch(d_seql)[: bsz - (bsz // 2)]
            dbatch = torch.cat((lm_h, pk_b), dim=0)
        else:
            dbatch = sample_mixed_batch(d_seql)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=(dev.type == "cuda")):
            d_logits = dense_model(dbatch)
            d_loss = F.cross_entropy(d_logits[:, :-1, :].reshape(-1, 256), dbatch[:, 1:].reshape(-1))
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(dense_model.parameters(), 1.0)
        dense_opt.step()
        if d_step == 1 or d_step % (2 if dry_run else 800) == 0 or d_step == dense_steps:
            print(
                f"  [DenseByteTransformer] step={d_step:5d}/{dense_steps} bpb={float(d_loss.item())/math.log(2.0):.4f} "
                f"({time.perf_counter()-t_dn:.1f}s)",
                flush=True,
            )
    dense_train_sec = time.perf_counter() - t_dn

    # ------------------------------------------------------------------
    # [3/4] Comprehensive Held-Out Evaluation (`PT/EN/ES/FR/Code` + Passkey up to `T=1024` + `.uscar` GDPR)
    # ------------------------------------------------------------------
    print("\n[3/4] Evaluating on Strictly Held-Out Open Splits (`PT`, `EN`, `ES`, `FR`, `Code`)...", flush=True)
    sf_eng.eval()
    dense_model.eval()

    def sf_probs_fn(x: torch.Tensor) -> torch.Tensor:
        sc = sf_eng.forward_sequence(x, adaptive_depth=True)["scores"]
        pr, _ = tsallis15_spherical_cap_probs(sc)
        return pr

    def dense_probs_fn(x: torch.Tensor) -> torch.Tensor:
        return F.softmax(dense_model(x), dim=-1)

    eval_seq_len = 64 if dry_run else 384
    domain_eval: Dict[str, Any] = {}
    for dom in ("pt", "en", "es", "fr", "code"):
        lam_dom = calibrate_ppm_escape_lambda_on_train(
            sf_probs_fn,
            train_splits[dom],
            eval_seq_len,
            2 if dry_run else 20,
            dev,
        )
        sf_m = evaluate_byte_lm_metrics(
            sf_probs_fn, val_splits[dom], eval_seq_len, eval_chunks, dev, is_sparse_tsallis=True, escape_lambda=lam_dom
        )
        dn_m = evaluate_byte_lm_metrics(
            dense_probs_fn, val_splits[dom], eval_seq_len, eval_chunks, dev, is_sparse_tsallis=False
        )
        domain_eval[dom] = {"scarfield_v7_2_deep": sf_m, "dense_fp16_ot": dn_m}
        print(
            f"  [{dom.upper():4s}] ScarField v7.2-Deep: BPB={sf_m['calibrated_bpb']:.3f} "
            f"Top1={sf_m['top1_byte_acc_pct']:.2f}% Top5={sf_m['top5_byte_acc_pct']:.2f}% "
            f"Zeros={sf_m['exact_zero_sparsity_pct']:.2f}% | "
            f"Dense FP16: BPB={dn_m['bpb']:.3f} Top1={dn_m['top1_byte_acc_pct']:.2f}% Top5={dn_m['top5_byte_acc_pct']:.2f}%",
            flush=True,
        )

    pk_horizons = [64, 128] if dry_run else [128, 256, 512, 768, 1024]
    n_pk = 4 if dry_run else 32
    val_prose_concat = np.concatenate([val_splits[k] for k in ("pt", "en", "es", "fr")])
    passkey_eval: Dict[str, Any] = {"scarfield_v7_2_deep": {}, "dense_fp16_ot": {}}
    for h_len in pk_horizons:
        sf_pk = evaluate_open_passkey_recall(
            lambda b: sf_eng.forward_sequence(b, adaptive_depth=True)["scores"], val_prose_concat, h_len, n_pk, dev
        )
        dn_pk = evaluate_open_passkey_recall(lambda b: dense_model(b), val_prose_concat, h_len, n_pk, dev)
        passkey_eval["scarfield_v7_2_deep"][f"T={h_len}"] = sf_pk
        passkey_eval["dense_fp16_ot"][f"T={h_len}"] = dn_pk
        print(
            f"  [Passkey T={h_len:4d}] ScarField v7.2-Deep: {sf_pk['exact_3digit_acc']*100:5.1f}% "
            f"(digit {sf_pk['per_digit_acc']*100:5.1f}%) | Dense O(T): {dn_pk['exact_3digit_acc']*100:5.1f}%",
            flush=True,
        )

    with torch.no_grad():
        p_en = project_rows_to_unit_sphere(sf_eng.forward_sequence(anch_en)["h_world"].mean(dim=1))
        p_pt = project_rows_to_unit_sphere(sf_eng.forward_sequence(anch_pt)["h_world"].mean(dim=1))
        p_es = project_rows_to_unit_sphere(sf_eng.forward_sequence(anch_es)["h_world"].mean(dim=1))
        p_fr = project_rows_to_unit_sphere(sf_eng.forward_sequence(anch_fr)["h_world"].mean(dim=1))
        xl_cos = {
            "pt_en": round(float((p_pt * p_en).sum(dim=-1).mean().item()), 5),
            "es_en": round(float((p_es * p_en).sum(dim=-1).mean().item()), 5),
            "fr_en": round(float((p_fr * p_en).sum(dim=-1).mean().item()), 5),
        }

        ooc_all = torch.cat((ooc_tr, ooc_ho), dim=0)
        ooc_out = sf_eng.forward_sequence(ooc_all)
        vac_hits = sum(
            1
            for i in range(ooc_all.shape[0])
            if sf_eng.head.decide(
                ooc_out["h_form"][i, -1],
                ooc_out["h_world"][i, -1],
                bus_ctx_1d=ooc_out["bus_ctx"][i, -1],
                require_world_support=True,
            ).is_vacuum_abstain
        )

        # Verify Zero-SGD `.uscar` cartridge minting in EN + firing across PT/ES/FR + `0x00` GDPR wipe!
        w_q_en = sf_eng.forward_sequence(anch_en)["h_world"][:, -1, :]
        wb_anch = project_rows_to_unit_sphere(sf_eng.head.out_bus_anchors)
        tgt_bus = wb_anch[torch.tensor([ord("7"), ord("3"), ord("9"), ord("1"), ord("5"), ord("8")], device=dev)]
        cart = sf_eng.bus.uscar_store.mint_zero_sgd_rrr(
            cartridge_id="overnight_fact_01",
            doc_id="doc_gdpr_test",
            world_queries=w_q_en,
            target_bus_deltas=4.0 * tgt_bus,
            rank=6,
            horizon_kappa=0.45,
        )
        _, res_pt_before = sf_eng.bus.uscar_store.query(sf_eng.forward_sequence(anch_pt)["h_world"][:, -1, :])
        uscar_pt_res_before = float(res_pt_before.mean().item())
        sf_eng.bus.uscar_store.wipe_document_0x00("doc_gdpr_test")
        delta_after, res_pt_after = sf_eng.bus.uscar_store.query(sf_eng.forward_sequence(anch_pt)["h_world"][:, -1, :])
        uscar_gdpr_residual = float(delta_after.abs().max().item() + res_pt_after.abs().max().item())
        sf_eng.bus.uscar_store.cartridges.clear()

        n_gen = 10 if dry_run else 64
        use_sc = dev.type == "cuda"
        samples = {
            "pt_prose": generate_continuation_sample(sf_eng, "A mecânica quântica e a relatividade geral ", n_gen, dev, use_sc),
            "en_prose": generate_continuation_sample(sf_eng, "In theoretical physics and radio astronomy, ", n_gen, dev, use_sc),
            "es_prose": generate_continuation_sample(sf_eng, "En la física cuántica y la termodinámica, ", n_gen, dev, use_sc),
            "fr_prose": generate_continuation_sample(sf_eng, "Dans la mécanique quantique et l'astrophysique, ", n_gen, dev, use_sc),
            "python_code": generate_continuation_sample(sf_eng, "def solve_triangular_system(mat: np.ndarray", n_gen, dev, use_sc),
            "passkey_o1": generate_continuation_sample(
                sf_eng, "[PASSKEY PIN=836] Thermodynamics governs macroscopic entropy and heat. [QUERY PIN=> ", 8, dev, use_sc
            ),
        }

    # ------------------------------------------------------------------
    # [4/4] Export Final Zero-Pickle `.sfield.scar` Container (`CRC32` + `mmap`) & Bundle
    # ------------------------------------------------------------------
    print("\n[4/4] Exporting Final Zero-Pickle `.sfield.scar` Container (`CRC32` + `mmap`) & Audit JSON...", flush=True)
    sf_cpu = sf_eng.cpu()
    sfield_scar_path = out_dir / "scarfield_v7_2_overnight.sfield.scar"
    sfield_scar_bytes = sf_cpu.save_packed_scar_container(sfield_scar_path)

    eng_mm, mm_handle = ScarFieldEngine.load_packed_scar_mmap(sfield_scar_path, verify_crc=True)
    try:
        probe_tok = anch_en[:2].cpu()
        s_ref = sf_cpu.forward_sequence(probe_tok, adaptive_depth=False)["scores"]
        s_mm = eng_mm.forward_sequence(probe_tok, adaptive_depth=False)["scores"]
        mmap_c_simd_cosine = float(F.cosine_similarity(s_ref.reshape(1, -1), s_mm.reshape(1, -1)).item())
    finally:
        mm_handle.close()

    dense_kv_t2048_bytes = 2 * dense_l * 2048 * dense_d * 2

    report = {
        "suite": "scar_field_v7_2_overnight_deep_audit",
        "version": "7.2.0",
        "mode": "dry_run" if dry_run else "overnight_deep_a100_until_09am",
        "hardware": gpu_name,
        "elapsed_total_seconds": round(time.perf_counter() - t_global, 2),
        "completed_steps": step,
        "corpus_summary": {
            "total_train_utf8_bytes": total_train_bytes,
            "total_heldout_val_utf8_bytes": total_val_bytes,
            "total_streamed_training_token_bytes": total_trained_bytes,
            "guard_gap_bytes": 512 if dry_run else 4096,
            "num_sha256_verified_sources": len(provenance),
            "provenance_sources": provenance,
        },
        "scarfield_v7_2_config": sf_cfg.to_dict(),
        "architecture_comparison": {
            "scarfield_v7_2_spherical_params": sf_param_count,
            "scarfield_v7_2_sfield_scar_bytes": sfield_scar_bytes,
            "scarfield_v7_2_sfield_scar_mib": round(sfield_scar_bytes / (1024**2), 3),
            "scarfield_v7_2_kalman_scache_o1_bytes": sf_budget["central_kalman_scache_o1_bytes"],
            "dense_transformer_params": dense_param_count,
            "dense_transformer_fp16_bytes": dense_fp16_bytes,
            "dense_transformer_fp16_mib": round(dense_fp16_bytes / (1024**2), 3),
            "dense_kv_cache_t2048_bytes": dense_kv_t2048_bytes,
            "weight_compression_vs_dense_fp16_x": round(dense_fp16_bytes / max(1, sfield_scar_bytes), 2),
            "state_reduction_at_t2048_x": round(dense_kv_t2048_bytes / max(1, sf_budget["central_kalman_scache_o1_bytes"]), 2),
            "mmap_c_simd_forward_cosine": round(mmap_c_simd_cosine, 6),
            "scarfield_train_seconds": round(sf_train_sec, 1),
            "dense_train_seconds": round(dense_train_sec, 1),
        },
        "heldout_open_domain_metrics": domain_eval,
        "unseen_passkey_recall_in_open_prose": passkey_eval,
        "polyglot_and_epistemic_metrics": {
            "cross_lingual_s_world_cosine_zero_procrustes": xl_cos,
            "unseen_out_of_cone_0x00_vacuum_abstention": f"{vac_hits}/{ooc_all.shape[0]}",
            "uscar_en_mint_to_pt_resonance": round(uscar_pt_res_before, 4),
            "uscar_0x00_gdpr_post_wipe_residual": uscar_gdpr_residual,
        },
        "training_curve": sf_history,
        "autoregressive_samples": samples,
    }

    audit_json_path = out_dir / "scar_field_v7_2_overnight_audit.json"
    audit_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    bundle_tar = (
        Path("/content/scarfield_v7_2_overnight_bundle.tar.gz")
        if Path("/content").exists() and not dry_run
        else out_dir / "scarfield_v7_2_overnight_bundle.tar.gz"
    )
    with tarfile.open(bundle_tar, "w:gz") as tf:
        tf.add(sfield_scar_path, arcname=sfield_scar_path.name)
        tf.add(audit_json_path, arcname=audit_json_path.name)

    print(
        f"\n[OVERNIGHT DONE] Steps={step:,d} | Streamed={total_trained_bytes / (1024**2):.1f} MiB | "
        f"Packed `.sfield.scar`={sfield_scar_bytes / (1024**2):.3f} MiB "
        f"({report['architecture_comparison']['weight_compression_vs_dense_fp16_x']}x < Dense FP16) | "
        f"mmap C/SIMD cos={mmap_c_simd_cosine:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
