"""Open-Weights Cartridge Factory (`v3.0.0`) — Zero-SGD Mass Production of `.scarstore` Cartridges.

Supports three genuine, zero-backpropagation (`0.0` SGD steps) pathways for minting `140.00 KB` (`3B`)
or `61.25 KB` (`0.5B`) `HeadExpandedScarCapsule` cartridges from open-weight models:

  1. `same_vocab_exact`:
     Open-weight Teacher shares the Student's exact tokenizer vocabulary (`V_teacher == V_student`,
     e.g., `Qwen2.5-72B-Instruct`, `Qwen2.5-Coder-32B-Instruct`, `Qwen2.5-Math-72B-Instruct`,
     `Qwen2.5-7B-Instruct`, `DeepSeek-R1-Distill-Qwen-32B`). Captures full dense `[L_a, V]` Teacher
     logits at every step for lossless Vocabulary Rosetta inversion.

  2. `cross_tokenizer_bridge`:
     Open-weight Teacher uses a different tokenizer (`V_teacher != V_student`, e.g., `Llama-3.3`,
     `Mistral`, `Mixtral`, `Gemma-2`, `Command-R+`, `Yi`, `SmolLM2`). The Teacher generates its
     expert answer in its native tokenizer alongside step top-`K` surface token log-probabilities;
     the trajectory and surface candidates are projected onto the Student's vocabulary space over
     the Student's natural grammatical baseline logits, then solved in closed form via Woodbury
     Rosetta + AlphaEdit Null-Space SVD.

  3. `curated_trajectory`:
     Converts pre-generated Q&A trajectories from remote open-weight clusters (`DeepSeek-R1-671B`,
     `Llama-3.1-405B`, vLLM batch exports, or human-verified domain specs) directly into `SleepEpisode`s
     on the Student model in `< 1 s` without loading the giant Teacher locally.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from scar_cache.head_expanded_attention import (
    HeadExpandedAttentionBank,
    HeadExpandedAttentionEngine,
    HeadExpandedScarCapsule,
    HeadExpandedScarStore,
)
from scar_cache.scar_grafting import (
    ScarGraftEngine,
    SleepEpisode,
    _infer_model_device,
    detect_chatml_user_prefix_ids,
    find_subsequence_last_end,
)


def slugify_capsule_id(text: str, prefix: str = "ow") -> str:
    """Converts a human title into a clean ASCII `capsule_id`."""
    cleaned = re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")
    if not cleaned:
        cleaned = "custom_cartridge"
    if not cleaned.startswith(f"{prefix}_") and prefix:
        cleaned = f"{prefix}_{cleaned}"
    return cleaned[:64]


def tokenizers_share_vocabulary(student_tok: Any, teacher_tok: Any) -> bool:
    """Checks whether `student_tok` and `teacher_tok` share identical token-to-ID mappings."""
    if student_tok is teacher_tok:
        return True
    try:
        sv = student_tok.get_vocab()
        tv = teacher_tok.get_vocab()
        if len(sv) != len(tv):
            return False
        probe_tokens = ["the", " of", "1", "A", "\n"]
        for tok_s in probe_tokens:
            if tok_s in sv and sv.get(tok_s) != tv.get(tok_s):
                return False
        return True
    except Exception:
        return False


def discover_local_open_weight_models(hf_cache_dir: Optional[Union[str, Path]] = None) -> List[Dict[str, Any]]:
    """Lists HuggingFace models already cached locally in `~/.cache/huggingface/hub`."""
    root = Path(hf_cache_dir) if hf_cache_dir is not None else (Path.home() / ".cache" / "huggingface" / "hub")
    results: List[Dict[str, Any]] = []
    if not root.exists():
        return results
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        parts = entry.name[len("models--") :].split("--", 1)
        repo_id = f"{parts[0]}/{parts[1]}" if len(parts) == 2 else parts[0]
        family = "qwen_same_vocab" if repo_id.lower().startswith("qwen/qwen2.5") else "cross_tokenizer"
        results.append(
            {
                "repo_id": repo_id,
                "mode_for_qwen_student": "same_vocab_exact" if family == "qwen_same_vocab" else "cross_tokenizer_bridge",
                "cache_dir": str(entry),
            }
        )
    return results


@torch.no_grad()
def build_episodes_from_text_pairs(
    capsule_id: str,
    qa_pairs: Sequence[Dict[str, str]],
    student_model: nn.Module,
    student_tokenizer: Any,
    chat_formatter: Callable[[str], str],
    target_margin: float = 3.6,
    surface_candidates_per_step: Optional[Sequence[Optional[List[Tuple[str, float]]]]] = None,
    source_label: str = "curated_trajectory",
) -> List[SleepEpisode]:
    """Projects `(prompt, expert_answer)` text trajectories onto the Student's logit space (`Zero SGD`).

    At every answer token `t`, preserves the Student's natural background logit distribution on
    non-conflicting vocabulary entries while elevating the Teacher-endorsed target token `y*_t`
    (plus any aligned cross-tokenizer surface candidates) by `target_margin` logit units so
    Woodbury Rosetta Cutting-Plane + AlphaEdit Null-Space solves an exact low-rank cartridge.
    """
    if len(qa_pairs) == 0:
        raise ValueError("qa_pairs must contain at least one {'prompt': ..., 'answer': ...} item.")

    dev = _infer_model_device(student_model)
    prefix_ids = detect_chatml_user_prefix_ids(student_tokenizer)
    eos_id = getattr(student_tokenizer, "eos_token_id", None)

    vocab_surface_to_id: Dict[str, int] = {}
    if surface_candidates_per_step is not None and hasattr(student_tokenizer, "encode"):
        # Lazily encode surface candidate strings to single-token Student IDs when unambiguous
        pass

    episodes: List[SleepEpisode] = []
    for pair_idx, item in enumerate(qa_pairs):
        prompt_raw = str(item["prompt"]).strip()
        answer_raw = str(item["answer"]).strip()
        if not prompt_raw or not answer_raw:
            raise ValueError(f"Empty prompt or answer in qa_pairs[{pair_idx}].")

        formatted_prompt = chat_formatter(prompt_raw)
        p_ids_1d = student_tokenizer(formatted_prompt, return_tensors="pt").input_ids[0].long()
        a_ids_list = list(student_tokenizer.encode(answer_raw, add_special_tokens=False))
        if eos_id is not None and (len(a_ids_list) == 0 or a_ids_list[-1] != int(eos_id)):
            a_ids_list.append(int(eos_id))
        a_ids_1d = torch.tensor(a_ids_list, dtype=torch.long)

        lp = int(p_ids_1d.numel())
        la = int(a_ids_1d.numel())
        full_1d = torch.cat([p_ids_1d, a_ids_1d], dim=0)

        out = student_model(full_1d.unsqueeze(0).to(dev), use_cache=False)
        lg_full = (out.logits if hasattr(out, "logits") else out)[0].detach().float().cpu()
        ans_base = lg_full[lp - 1 : lp - 1 + la].clone()  # [la, V_student]

        step_cands = (
            surface_candidates_per_step[pair_idx]
            if (surface_candidates_per_step is not None and pair_idx < len(surface_candidates_per_step))
            else None
        )

        for t in range(la):
            row = ans_base[t]
            target_id = int(a_ids_1d[t].item())

            if step_cands is not None and t < len(step_cands) and step_cands[t]:
                top_max = float(row.max().item())
                for surf_str, rel_logp in step_cands[t]:
                    if not surf_str:
                        continue
                    if surf_str not in vocab_surface_to_id:
                        enc = student_tokenizer.encode(surf_str, add_special_tokens=False)
                        vocab_surface_to_id[surf_str] = int(enc[0]) if len(enc) == 1 else -1
                    cand_id = vocab_surface_to_id[surf_str]
                    if 0 <= cand_id < int(row.shape[0]) and cand_id != target_id:
                        boosted = top_max + float(target_margin) * 0.65 + float(rel_logp)
                        if boosted > float(row[cand_id].item()):
                            row[cand_id] = boosted

            cur_max = float(row.max().item())
            row[target_id] = cur_max + float(target_margin)

        q_start = find_subsequence_last_end(p_ids_1d.tolist(), prefix_ids)
        episodes.append(
            SleepEpisode(
                capsule_id=capsule_id,
                prompt_ids=p_ids_1d.cpu(),
                full_ids=full_1d.cpu(),
                teacher_logits=ans_base.to(torch.float16).cpu(),
                query_start_idx=q_start,
                surprise_score=1.0,
                metadata={
                    "prompt": prompt_raw,
                    "answer": answer_raw,
                    "mint_mode": source_label,
                },
            )
        )
    return episodes


@torch.no_grad()
def collect_open_weight_episodes(
    capsule_id: str,
    train_prompts: Sequence[str],
    student_model: nn.Module,
    student_tokenizer: Any,
    teacher_model: Any,
    teacher_tokenizer: Optional[Any] = None,
    chat_formatter: Optional[Callable[[str], str]] = None,
    system_prompt: str = "Answer concisely in one precise technical sentence.",
    max_new_tokens: int = 36,
    target_margin: float = 3.6,
    user_turn_prefix_ids: Optional[Sequence[int]] = None,
) -> Tuple[List[SleepEpisode], str]:
    """Collects `SleepEpisode`s from ANY Open-Weight Teacher (same-vocab exact OR cross-tokenizer bridge).

    Returns `(episodes, mint_mode)` where `mint_mode` is `'same_vocab_exact'` or `'cross_tokenizer_bridge'`.
    """
    fmt = chat_formatter or (
        lambda q: f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
    )
    t_tok = teacher_tokenizer if teacher_tokenizer is not None else student_tokenizer

    # Path 1: CanonicalMoSEngine or same-vocabulary HF Teacher -> 100% exact dense logit transposition
    if hasattr(teacher_model, "n_groups") or tokenizers_share_vocabulary(student_tokenizer, t_tok):
        eps = ScarGraftEngine.collect_teacher_episodes(
            capsule_id=capsule_id,
            train_prompts=list(train_prompts),
            teacher_model=teacher_model,
            tokenizer=student_tokenizer,
            chat_formatter=fmt,
            max_new_tokens=max_new_tokens,
            user_turn_prefix_ids=user_turn_prefix_ids,
        )
        for ep in eps:
            ep.metadata["mint_mode"] = "same_vocab_exact"
        return eps, "same_vocab_exact"

    # Path 2: Cross-Tokenizer Open-Weight Teacher -> native generation + surface-aligned Rosetta projection
    t_dev = _infer_model_device(teacher_model)
    t_eos = getattr(t_tok, "eos_token_id", None)
    qa_pairs: List[Dict[str, str]] = []
    all_step_cands: List[Optional[List[Tuple[str, float]]]] = []

    for q_text in train_prompts:
        t_prompt = q_text
        if hasattr(t_tok, "apply_chat_template") and getattr(t_tok, "chat_template", None):
            try:
                t_prompt = t_tok.apply_chat_template(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": q_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                try:
                    t_prompt = t_tok.apply_chat_template(
                        [{"role": "user", "content": f"{system_prompt}\n\n{q_text}"}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    t_prompt = f"Question: {q_text}\nConcise technical answer:"
        else:
            t_prompt = f"Question: {q_text}\nConcise technical answer:"

        t_in = t_tok(t_prompt, return_tensors="pt").input_ids.to(t_dev)
        lp_t = int(t_in.shape[1])
        gen_kwargs: Dict[str, Any] = {"max_new_tokens": int(max_new_tokens), "do_sample": False}
        if t_eos is not None:
            gen_kwargs["eos_token_id"] = int(t_eos)
        if getattr(t_tok, "pad_token_id", None) is not None:
            gen_kwargs["pad_token_id"] = int(t_tok.pad_token_id)
        elif t_eos is not None:
            gen_kwargs["pad_token_id"] = int(t_eos)

        if hasattr(teacher_model, "generate"):
            seq_t = teacher_model.generate(t_in, **gen_kwargs)
        else:
            cur = t_in
            for _ in range(int(max_new_tokens)):
                out_step = teacher_model(cur)
                lg_s = out_step.logits if hasattr(out_step, "logits") else out_step
                nxt = int(lg_s[0, -1].float().argmax(dim=-1).item())
                cur = torch.cat([cur, torch.tensor([[nxt]], dtype=torch.long, device=t_dev)], dim=1)
                if t_eos is not None and nxt == int(t_eos):
                    break
            seq_t = cur

        gen_t_ids = seq_t[0, lp_t:]
        ans_str = t_tok.decode(gen_t_ids, skip_special_tokens=True).strip()
        if not ans_str:
            ans_str = t_tok.decode(gen_t_ids, skip_special_tokens=False).strip() or "."

        # Extract Teacher step top-5 surface candidates for cross-vocab dark-knowledge projection
        step_cands_for_prompt: List[Tuple[str, float]] = []
        try:
            out_t = teacher_model(seq_t, use_cache=False)
        except TypeError:
            out_t = teacher_model(seq_t)
        lg_t = (out_t.logits if hasattr(out_t, "logits") else out_t)[0, lp_t - 1 : lp_t - 1 + int(gen_t_ids.numel())].float().cpu()
        per_step_list: List[List[Tuple[str, float]]] = []
        for s_i in range(int(lg_t.shape[0])):
            logp = torch.log_softmax(lg_t[s_i], dim=-1)
            vals, idxs = logp.topk(min(5, int(logp.shape[0])))
            top0 = float(vals[0].item())
            c_step: List[Tuple[str, float]] = []
            for v_val, idx_val in zip(vals.tolist(), idxs.tolist()):
                surf = t_tok.decode([int(idx_val)], skip_special_tokens=True)
                if surf:
                    c_step.append((surf, float(v_val) - top0))
            per_step_list.append(c_step)

        qa_pairs.append({"prompt": q_text, "answer": ans_str})
        all_step_cands.append(per_step_list if per_step_list else None)

    episodes = build_episodes_from_text_pairs(
        capsule_id=capsule_id,
        qa_pairs=qa_pairs,
        student_model=student_model,
        student_tokenizer=student_tokenizer,
        chat_formatter=fmt,
        target_margin=target_margin,
        surface_candidates_per_step=all_step_cands,
        source_label="cross_tokenizer_bridge",
    )
    return episodes, "cross_tokenizer_bridge"


@torch.no_grad()
def mint_capsules_from_specs(
    specs: Sequence[Dict[str, Any]],
    student_model: nn.Module,
    student_tokenizer: Any,
    head_bank: HeadExpandedAttentionBank,
    chat_formatter: Callable[[str], str],
    teacher_model: Optional[Any] = None,
    teacher_tokenizer: Optional[Any] = None,
    teacher_label: Optional[str] = None,
    rank: int = 60,
    num_key_prototypes: int = 8,
    tau: float = 0.62,
    ramp: float = 0.16,
    max_new_tokens: int = 36,
) -> List[HeadExpandedScarCapsule]:
    """Batch-mints `HeadExpandedScarCapsule` objects from a list of JSON/dict cartridge specifications.

    Each spec dictionary supports:
      - `capsule_id` (optional if `title` provided)
      - `title`, `title_pt`, `category`, `summary`, `example_prompt`
      - Either `episodes`: `[{"prompt": "...", "answer": "..."}, ...]` (curated/offline open-weight pairs)
      - Or `train_prompts`: `["...", "..."]` (live extraction from `teacher_model`)
    """
    minted: List[HeadExpandedScarCapsule] = []
    for raw_spec in specs:
        title = str(raw_spec.get("title") or raw_spec.get("title_pt") or raw_spec.get("capsule_id") or "Custom Cartridge").strip()
        cid = str(raw_spec.get("capsule_id") or slugify_capsule_id(title)).strip()
        title_pt = str(raw_spec.get("title_pt") or title).strip()
        category = str(raw_spec.get("category") or "Open-Weights Custom").strip()
        summary = str(raw_spec.get("summary") or title).strip()

        qa_items = raw_spec.get("episodes") or raw_spec.get("qa_pairs")
        train_prompts = raw_spec.get("train_prompts") or raw_spec.get("prompts")

        if qa_items:
            eps = build_episodes_from_text_pairs(
                capsule_id=cid,
                qa_pairs=qa_items,
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                chat_formatter=chat_formatter,
                source_label=str(raw_spec.get("mint_mode", "curated_trajectory")),
            )
            mint_mode = str(raw_spec.get("mint_mode", "curated_trajectory"))
            example_prompt = str(raw_spec.get("example_prompt") or qa_items[0]["prompt"])
        elif train_prompts:
            if teacher_model is None:
                raise ValueError(
                    f"Cartridge spec {cid!r} provides 'train_prompts' without 'episodes' answers, "
                    f"so a Teacher model (--teacher) is required."
                )
            eps, mint_mode = collect_open_weight_episodes(
                capsule_id=cid,
                train_prompts=list(train_prompts),
                student_model=student_model,
                student_tokenizer=student_tokenizer,
                teacher_model=teacher_model,
                teacher_tokenizer=teacher_tokenizer,
                chat_formatter=chat_formatter,
                max_new_tokens=max_new_tokens,
                user_turn_prefix_ids=head_bank.user_turn_prefix_ids,
            )
            example_prompt = str(raw_spec.get("example_prompt") or train_prompts[0])
        else:
            raise ValueError(f"Cartridge spec {cid!r} must include either 'episodes' ([{{prompt, answer}}]) or 'train_prompts'.")

        meta = {
            "title": title,
            "title_pt": title_pt,
            "category": category,
            "summary": summary,
            "example_prompt": example_prompt,
            "teacher_model": teacher_label or str(raw_spec.get("teacher_model", "Open-Weights Specialist")),
            "mint_mode": mint_mode,
        }
        cap = HeadExpandedAttentionEngine.solve_head_expanded_capsule(
            capsule_id=cid,
            episodes=eps,
            student_model=student_model,
            head_bank=head_bank,
            rank=int(raw_spec.get("rank", rank)),
            num_key_prototypes=int(raw_spec.get("prototypes", num_key_prototypes)),
            tau=float(raw_spec.get("tau", tau)),
            ramp=float(raw_spec.get("ramp", ramp)),
            metadata=meta,
        )
        minted.append(cap)
    return minted


def load_specs_json(spec_json_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads a JSON list (or `{'cartridges': [...]}` object) of open-weights cartridge specs."""
    data = json.loads(Path(spec_json_path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [dict(x) for x in data]
    if isinstance(data, dict) and isinstance(data.get("cartridges"), list):
        return [dict(x) for x in data["cartridges"]]
    if isinstance(data, dict):
        return [dict(data)]
    raise ValueError(f"Unsupported spec JSON root type in {spec_json_path}")
