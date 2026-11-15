"""Polyglot Syntactic, Logical, Overflow-Passkey & Detached-Knowledge Benchmark Corpus (`scar_native` v5.0.0).

Strict Zero-Contamination Design:
  - `Pure Trunk Pretraining Corpus`: Teaches syntax, cross-lingual equivalence (`PT`, `EN`, `ES`, `FR`),
    symbolic arithmetic/deduction, and long-range associative passkey retrieval (`T = 140..192`, `> 5x` beyond `W_win = 24`),
    with **ZERO** exposure to the 16 canonical domain entities (`DETACHED_ENTITIES_16`).
  - `Dense Entangled Corpus`: Identical syntactic/logical/passkey stream PLUS the 16 domain facts mixed into SGD,
    modeling how standard Dense Transformers entangle mutable world facts inside dense MLP weights.
  - `Held-Out Evaluation Suites`: Disjoint numerical values, disjoint passkey codes (`5000..9999` vs `1000..4999` in train),
    and `64` cross-lingual (`16 entities x 4 languages`) detached knowledge prompts.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Dict, List, Sequence, Tuple

import torch

from scar_native.tokenizer import ScarNativeTokenizer

LANGUAGES = ("pt", "en", "es", "fr")


@dataclass(frozen=True)
class DetachedEntitySpec:
    entity_id: str
    name: str
    metric_pt: str
    metric_en: str
    metric_es: str
    metric_fr: str
    value: str
    unit: str


# 16 Canonical Domain Facts NEVER seen by the Pure Scar-Native Trunk during pretraining!
DETACHED_ENTITIES_16: Tuple[DetachedEntitySpec, ...] = (
    DetachedEntitySpec("ent_orion", "orion", "órbita", "orbit", "órbita", "orbite", "516", "km"),
    DetachedEntitySpec("ent_helios", "helios", "pressão", "pressure", "presión", "pression", "340", "bar"),
    DetachedEntitySpec("ent_aegis", "aegis", "frequência", "frequency", "frecuencia", "fréquence", "842", "mhz"),
    DetachedEntitySpec("ent_kronos", "kronos", "energia", "energy", "energía", "énergie", "925", "watts"),
    DetachedEntitySpec("ent_vega", "vega", "órbita", "orbit", "órbita", "orbite", "780", "km"),
    DetachedEntitySpec("ent_atlas", "atlas", "pressão", "pressure", "presión", "pression", "415", "kpa"),
    DetachedEntitySpec("ent_titan", "titan", "frequência", "frequency", "frecuencia", "fréquence", "612", "khz"),
    DetachedEntitySpec("ent_zephyr", "zephyr", "energia", "energy", "energía", "énergie", "274", "kelvin"),
    DetachedEntitySpec("ent_nova", "nova", "órbita", "orbit", "órbita", "orbite", "648", "km"),
    DetachedEntitySpec("ent_hyperion", "hyperion", "pressão", "pressure", "presión", "pression", "590", "bar"),
    DetachedEntitySpec("ent_borealis", "borealis", "frequência", "frequency", "frecuencia", "fréquence", "128", "ghz"),
    DetachedEntitySpec("ent_solaris", "solaris", "energia", "energy", "energía", "énergie", "730", "watts"),
    DetachedEntitySpec("ent_artemis", "artemis", "órbita", "orbit", "órbita", "orbite", "394", "km"),
    DetachedEntitySpec("ent_valkyrie", "valkyrie", "pressão", "pressure", "presión", "pression", "865", "kpa"),
    DetachedEntitySpec("ent_meridian", "meridian", "frequência", "frequency", "frecuencia", "fréquence", "456", "mhz"),
    DetachedEntitySpec("ent_polaris", "polaris", "energia", "energy", "energía", "énergie", "318", "kelvin"),
)

# Polyglot filler clauses used to push passkeys far beyond the `W_win = 24` local attention window
_DISTRACTOR_SENTENCES: Dict[str, Tuple[str, ...]] = {
    "pt": (
        "o sistema analisa o fluxo de telemetria com memória constante e zero ruído no vácuo .",
        "a arquitetura separa o estado sintático do módulo de dados por filtro ortogonal seguro .",
        "cada matriz opera em subespaço rápido mantendo o sinal limpo na janela de contexto .",
        "o núcleo verifica o protocolo de atenção contínua sem expandir o registro de memória .",
    ),
    "en": (
        "the system processes the telemetry stream with constant memory and zero vacuum noise .",
        "the architecture separates syntactic state from data modules via secure orthogonal filter .",
        "each matrix operates in fast subspace keeping the clean signal inside the context window .",
        "the core verifies the continuous attention protocol without expanding the memory record .",
    ),
    "es": (
        "el sistema analiza el informe de telemetría con memoria constante y cero ruido en el vacío .",
        "la arquitectura separa el estado del módulo de datos mediante filtro ortogonal seguro .",
        "cada matriz opera en ventana rápida manteniendo la señal limpia en el contexto .",
        "el núcleo verifica la operación de atención continua sin expandir los datos de memoria .",
    ),
    "fr": (
        "le système analyse le rapport de télémétrie avec mémoire constante et zéro bruit dans le vide .",
        "le modèle sépare le vecteur du protocole de données avec un filtre sous-espace sécurisé .",
        "chaque matrice opère dans la fenêtre rapide gardant le résultat précis dans le contexte .",
        "le noyau vérifie le réseau continu sans agrandir la mémoire du système .",
    ),
}

# Cross-lingual parallel concepts for Golden-Slice `P_⊥lang` contrastive nullspace alignment (`L_null`)
PARALLEL_CONCEPT_QUADS: Tuple[Dict[str, str], ...] = (
    {
        "pt": "<lang:pt> <user> como funciona a memória scache constante ? <assistant>",
        "en": "<lang:en> <user> how does the constant scache memory work ? <assistant>",
        "es": "<lang:es> <user> ¿ cómo está la memoria scache constante ? <assistant>",
        "fr": "<lang:fr> <user> comment est la mémoire scache constante ? <assistant>",
        "ans_pt": "a memória scache mantém estado constante o1 com janela rápida e cicatrizes verificadas .",
        "ans_en": "the scache memory keeps constant o1 state with fast window and verified scars .",
        "ans_es": "la memoria scache está en estado constante o1 con ventana rápida y cicatrices .",
        "ans_fr": "la mémoire scache garde un état constant o1 avec fenêtre rapide et cicatrice vérifié .",
    },
    {
        "pt": "<lang:pt> <user> qual é a vantagem do vetor d4 lut256 no vácuo ? <assistant>",
        "en": "<lang:en> <user> what is the advantage of d4 lut256 vector in vacuum ? <assistant>",
        "es": "<lang:es> <user> ¿ cuál es el valor del vector d4 lut256 en vacío ? <assistant>",
        "fr": "<lang:fr> <user> quelle est la valeur du vecteur d4 lut256 dans le vide ? <assistant>",
        "ans_pt": "o vetor d4 lut256 garante zero exato no vácuo e rotação fwht64 rápida .",
        "ans_en": "the d4 lut256 vector guarantees exact zero in vacuum and fast fwht64 rotation .",
        "ans_es": "el vector d4 lut256 da cero exacto en vacío y operación fwht64 rápida .",
        "ans_fr": "le vecteur d4 lut256 donne zéro exacte dans le vide et réseau fwht64 rapide .",
    },
    {
        "pt": "<lang:pt> <user> como o subespaço ortogonal protege cada idioma ? <assistant>",
        "en": "<lang:en> <user> how does the orthogonal subspace protect each language ? <assistant>",
        "es": "<lang:es> <user> ¿ cómo protege el filtro ortogonal cada idioma ? <assistant>",
        "fr": "<lang:fr> <user> comment le sous-espace filtre protège chaque langue ? <assistant>",
        "ans_pt": "o filtro ortogonal separa o idioma do estado de dados com resultado preciso .",
        "ans_en": "the orthogonal filter separates language from data state with accurate result .",
        "ans_es": "el filtro ortogonal separa el idioma de los datos con resultado verificado .",
        "ans_fr": "le filtre sous-espace sépare la langue des données avec résultat précis .",
    },
    {
        "pt": "<lang:pt> <user> olá bom dia como você pode ajudar no sistema ? <assistant>",
        "en": "<lang:en> <user> hello good day how can you help in the system ? <assistant>",
        "es": "<lang:es> <user> hola cómo puedo analizar el protocolo del satélite ? <assistant>",
        "fr": "<lang:fr> <user> bonjour comment peux aider à calculer la télémétrie ? <assistant>",
        "ans_pt": "olá posso explicar a arquitetura calcular o resultado e recuperar o código exato .",
        "ans_en": "hello can explain the architecture compute the result and retrieve the exact code .",
        "ans_es": "hola puedo calcular el resultado y recuperar la clave del satélite confirmado .",
        "ans_fr": "bonjour peux expliquer le modèle calculer la somme et récupérer la clé exacte .",
    },
)


def format_detached_entity_prompt(spec: DetachedEntitySpec, lang: str, abstain_unknown: bool = False) -> Tuple[str, str]:
    """Returns `(prompt_prefix, target_completion)` for a detached factual entity in `PT/EN/ES/FR`.

    When `abstain_unknown=True` (used ONLY during Pure Trunk pretraining so the trunk knows the grammatical frame
    and entity token embeddings without EVER seeing any entity's true value/unit!), the completion outputs the
    canonical unknown placeholder `000 km`.
    """
    val = "000" if abstain_unknown else spec.value
    unit = "km" if abstain_unknown else spec.unit
    if lang == "pt":
        p = f"<lang:pt> <user> <fact> qual é o valor exato de {spec.metric_pt} do módulo {spec.name} ? <assistant>"
        a = f"o valor exato de {spec.name} é {val} {unit} confirmado ."
    elif lang == "en":
        p = f"<lang:en> <user> <fact> what is the exact {spec.metric_en} value of module {spec.name} ? <assistant>"
        a = f"the exact value of {spec.name} is {val} {unit} confirmed ."
    elif lang == "es":
        p = f"<lang:es> <user> <fact> ¿ cuál es el valor exacto de {spec.metric_es} del módulo {spec.name} ? <assistant>"
        a = f"el valor exacto de {spec.name} es {val} {unit} confirmado ."
    elif lang == "fr":
        p = f"<lang:fr> <user> <fact> quelle est la valeur exacte de {spec.metric_fr} du module {spec.name} ? <assistant>"
        a = f"la valeur exacte de {spec.name} est {val} {unit} confirmé ."
    else:
        raise ValueError(f"Unsupported lang: {lang}")
    return p, a


_TRAIN_PASSKEYS: Tuple[int, ...] = tuple(c for c in range(100, 1000) if ((c * 2654435761) & 3) != 0)
_TEST_PASSKEYS: Tuple[int, ...] = tuple(c for c in range(100, 1000) if ((c * 2654435761) & 3) == 0)


_UNITS_POOL: Tuple[str, ...] = ("km", "bar", "mhz", "watts", "kpa", "khz", "kelvin", "ghz")


def build_logic_sample(rng: random.Random, lang: str, train: bool = True) -> Tuple[str, str]:
    """Builds symbolic arithmetic & transitive Modus-Ponens reasoning examples (`train` vs disjoint `held-out` pairs)."""
    while True:
        a = rng.randint(12, 48)
        b = rng.randint(6, 29)
        is_test_pair = (((a * 37 + b * 17) & 3) == 0)
        if train != is_test_pair:
            break
    mode = rng.choice(("sum", "unit_sum", "double", "compare"))
    vars_pool = ("alpha", "beta", "gamma", "delta", "omega", "sigma")
    v1, v2 = rng.sample(vars_pool, 2)

    if mode == "unit_sum":
        res = a + b
        u = rng.choice(_UNITS_POOL)
        if lang == "pt":
            return (
                f"<lang:pt> <user> se {v1} = {a} {u} e {v2} = {b} {u} então qual é a soma ? <assistant>",
                f"logo a soma é {res} {u} confirmado .",
            )
        if lang == "en":
            return (
                f"<lang:en> <user> if {v1} = {a} {u} and {v2} = {b} {u} then what is the sum ? <assistant>",
                f"therefore the sum is {res} {u} confirmed .",
            )
        if lang == "es":
            return (
                f"<lang:es> <user> si {v1} = {a} {u} y {v2} = {b} {u} entonces ¿ cuál es la suma ? <assistant>",
                f"por tanto la suma es {res} {u} confirmado .",
            )
        return (
            f"<lang:fr> <user> si {v1} = {a} {u} et {v2} = {b} {u} alors quelle est la somme ? <assistant>",
            f"donc la somme est {res} {u} confirmé .",
        )

    if mode == "sum":
        res = a + b
        if lang == "pt":
            return (
                f"<lang:pt> <user> se {v1} = {a} e {v2} = {b} então qual é a soma ? <assistant>",
                f"logo a soma resultado é {res} verificado .",
            )
        if lang == "en":
            return (
                f"<lang:en> <user> if {v1} = {a} and {v2} = {b} then what is the sum ? <assistant>",
                f"therefore the sum result is {res} verified .",
            )
        if lang == "es":
            return (
                f"<lang:es> <user> si {v1} = {a} y {v2} = {b} entonces ¿ cuál es la suma ? <assistant>",
                f"por tanto la suma es {res} verificado .",
            )
        return (
            f"<lang:fr> <user> si {v1} = {a} et {v2} = {b} alors quelle est la somme ? <assistant>",
            f"donc la somme résultat est {res} vérifié .",
        )

    if mode == "double":
        res = 2 * a
        if lang == "pt":
            return (
                f"<lang:pt> <user> se {v1} = {a} qual é o dobro de {v1} ? <assistant>",
                f"o dobro resultado é {res} verificado .",
            )
        if lang == "en":
            return (
                f"<lang:en> <user> if {v1} = {a} what is the double of {v1} ? <assistant>",
                f"the double result is {res} verified .",
            )
        if lang == "es":
            return (
                f"<lang:es> <user> si {v1} = {a} ¿ cuál es el doble de {v1} ? <assistant>",
                f"el doble es {res} verificado .",
            )
        return (
            f"<lang:fr> <user> si {v1} = {a} quel est le résultat 2 * {v1} ? <assistant>",
            f"le résultat est {res} vérifié .",
        )

    is_gt = a > b
    if lang == "pt":
        ans = "verdadeiro" if is_gt else "falso"
        return (
            f"<lang:pt> <user> se {v1} = {a} e {v2} = {b} então {v1} maior que {v2} ? <assistant>",
            f"portanto a resposta é {ans} confirmado .",
        )
    if lang == "en":
        ans = "true" if is_gt else "false"
        return (
            f"<lang:en> <user> if {v1} = {a} and {v2} = {b} is {v1} greater than {v2} ? <assistant>",
            f"therefore the answer is {ans} confirmed .",
        )
    if lang == "es":
        ans = "verdadero" if is_gt else "falso"
        return (
            f"<lang:es> <user> si {v1} = {a} y {v2} = {b} ¿ es {v1} mayor que {v2} ? <assistant>",
            f"por tanto la respuesta es {ans} confirmado .",
        )
    ans = "vrai" if is_gt else "faux"
    return (
        f"<lang:fr> <user> si {v1} = {a} et {v2} = {b} est {v1} supérieur à {v2} ? <assistant>",
        f"donc la réponse est {ans} confirmé .",
    )


def build_overflow_passkey_sample(
    rng: random.Random,
    lang: str,
    n_distractor_clauses: int = 4,
    train: bool = True,
) -> Tuple[str, str, str]:
    """Builds a 3-slot associative passkey (`alpha=d1, beta=d2, gamma=d3`) pushed `60..600+` tokens before query!"""
    code_int = rng.choice(_TRAIN_PASSKEYS if train else _TEST_PASSKEYS)
    code_str = str(code_int)
    d1, d2, d3 = code_str[0], code_str[1], code_str[2]
    clauses = [rng.choice(_DISTRACTOR_SENTENCES[lang]) for _ in range(n_distractor_clauses)]
    filler = " ".join(clauses)
    ans = f"alpha = {d1} , beta = {d2} , gamma = {d3} ."

    if lang == "pt":
        prompt = f"<lang:pt> <user> chave secreta <passkey> alpha = {d1} , beta = {d2} , gamma = {d3} </passkey> . {filler} qual é a chave secreta ? <assistant>"
    elif lang == "en":
        prompt = f"<lang:en> <user> secret key <passkey> alpha = {d1} , beta = {d2} , gamma = {d3} </passkey> . {filler} what is the secret key ? <assistant>"
    elif lang == "es":
        prompt = f"<lang:es> <user> clave secreto <passkey> alpha = {d1} , beta = {d2} , gamma = {d3} </passkey> . {filler} ¿ cuál es la clave secreto ? <assistant>"
    else:
        prompt = f"<lang:fr> <user> clé secrète <passkey> alpha = {d1} , beta = {d2} , gamma = {d3} </passkey> . {filler} quelle est la clé secrète ? <assistant>"
    return prompt, ans, code_str


def build_pretraining_tensors(
    tokenizer: ScarNativeTokenizer,
    n_samples: int = 2400,
    seq_len: int = 92,
    include_entangled_facts: bool = False,
    seed: int = 202605,
    prompt_loss_weight: float = 0.12,
    distractor_clause_range: Tuple[int, int] = (2, 4),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generates `[n_samples, seq_len]` `(input_ids, labels, loss_weights)` pretraining corpus (`-100` on pad)."""
    rng = random.Random(seed)
    inputs = torch.full((n_samples, seq_len), tokenizer.pad_token_id, dtype=torch.long)
    labels = torch.full((n_samples, seq_len), -100, dtype=torch.long)
    weights = torch.zeros((n_samples, seq_len), dtype=torch.float32)
    c_min, c_max = distractor_clause_range

    for i in range(n_samples):
        lang = LANGUAGES[i % len(LANGUAGES)]
        roll = rng.random()
        is_passkey = False
        if roll < 0.22:
            spec = rng.choice(DETACHED_ENTITIES_16)
            p_txt, a_txt = format_detached_entity_prompt(
                spec, lang, abstain_unknown=(not include_entangled_facts)
            )
        elif roll < 0.65:
            is_passkey = True
            p_txt, a_txt, _ = build_overflow_passkey_sample(
                rng, lang, n_distractor_clauses=rng.randint(c_min, c_max), train=True
            )
        elif roll < 0.88:
            p_txt, a_txt = build_logic_sample(rng, lang, train=True)
        else:
            quad = rng.choice(PARALLEL_CONCEPT_QUADS)
            p_txt = quad[lang]
            a_txt = quad[f"ans_{lang}"]

        p_ids = tokenizer.encode(p_txt, add_bos=True, add_eos=False)
        a_ids = tokenizer.encode(a_txt, add_bos=False, add_eos=True)
        full_ids = p_ids + a_ids
        if len(full_ids) > seq_len + 1:
            # Never truncate the assistant completion or question tail! Trim excess middle filler tokens if needed.
            keep_p = (seq_len + 1) - len(a_ids)
            tail_n = min(12, max(4, keep_p // 3))
            p_ids = p_ids[: max(4, keep_p - tail_n)] + p_ids[-tail_n:]
            full_ids = (p_ids + a_ids)[: seq_len + 1]

        x = full_ids[:-1]
        y = full_ids[1:]
        L = len(x)
        ans_start = max(0, len(p_ids) - 1)
        inputs[i, :L] = torch.tensor(x, dtype=torch.long)
        labels[i, :L] = torch.tensor(y, dtype=torch.long)
        weights[i, :ans_start] = prompt_loss_weight
        weights[i, ans_start:L] = 1.0
        if is_passkey and prompt_loss_weight > 0.0:
            # Emphasize the 3 associative slot digit targets (`alpha = d1 , beta = d2 , gamma = d3`) identically across all models
            eq_ids = tokenizer.encode("=", add_bos=False)
            if eq_ids:
                eq_tok = eq_ids[0]
                for pos_k in range(ans_start, L):
                    if x[pos_k] == eq_tok:
                        weights[i, pos_k] = 4.0

    return inputs, labels, weights


def build_contrastive_nullspace_batch(
    tokenizer: ScarNativeTokenizer,
    max_len: int = 38,
    step_seed: int = 2026,
    return_trajectories: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Builds cross-lingual `(PT, EN, ES, FR)` alignment batch with ZERO exposure to secret entity facts.

    When `return_trajectories=False`, returns `(batch_ids [N_c, 4, max_len], anchor_idx [N_c, 4])`.
    When `return_trajectories=True`, returns `(batch_ids [20, 4, max_len], anchor_idx [20, 4], traj_idx [16, 12, 4])`
    covering the 4 general concept quads + 16 entity frames with randomized dummy digits/units (`0` secret leakage).
    """
    rng = random.Random(step_seed)
    n_quads = len(PARALLEL_CONCEPT_QUADS)
    n_ent = len(DETACHED_ENTITIES_16)
    total_seq = n_quads + n_ent

    out = torch.full((total_seq, 4, max_len), tokenizer.pad_token_id, dtype=torch.long)
    anchor_idx = torch.zeros((total_seq, 4), dtype=torch.long)
    traj_idx = torch.zeros((n_ent, 12, 4), dtype=torch.long)

    for q_i, quad in enumerate(PARALLEL_CONCEPT_QUADS):
        for l_i, lang in enumerate(LANGUAGES):
            ids = tokenizer.encode(quad[lang], add_bos=True, add_eos=False)[:max_len]
            out[q_i, l_i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            anchor_idx[q_i, l_i] = max(0, len(ids) - 1)

    for e_i, spec in enumerate(DETACHED_ENTITIES_16):
        d1, d2, d3 = rng.sample("0123456789", 3)
        dummy_val = f"{d1}{d2}{d3}"
        dummy_unit = _UNITS_POOL[(e_i + step_seed) % len(_UNITS_POOL)]
        row_i = n_quads + e_i
        for l_i, lang in enumerate(LANGUAGES):
            p_txt, _ = format_detached_entity_prompt(spec, lang, abstain_unknown=True)
            if lang == "pt":
                a_dummy = f"o valor exato de {spec.name} é {dummy_val} {dummy_unit} confirmado ."
            elif lang == "en":
                a_dummy = f"the exact value of {spec.name} is {dummy_val} {dummy_unit} confirmed ."
            elif lang == "es":
                a_dummy = f"el valor exacto de {spec.name} es {dummy_val} {dummy_unit} confirmado ."
            else:
                a_dummy = f"la valeur exacte de {spec.name} est {dummy_val} {dummy_unit} confirmé ."

            p_ids = tokenizer.encode(p_txt, add_bos=True, add_eos=False)
            a_ids = tokenizer.encode(a_dummy, add_bos=False, add_eos=False)
            full = (p_ids + a_ids)[:max_len]
            out[row_i, l_i, : len(full)] = torch.tensor(full, dtype=torch.long)
            anc = max(0, len(p_ids) - 1)
            anchor_idx[row_i, l_i] = min(anc, max_len - 1)
            for r_k in range(12):
                traj_idx[e_i, r_k, l_i] = min(anc + r_k, max(0, len(full) - 1))

    if return_trajectories:
        return out, anchor_idx, traj_idx
    return out, anchor_idx
