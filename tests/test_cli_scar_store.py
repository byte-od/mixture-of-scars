"""Unit & Integration Tests for `mos-store` CLI, RFC 7233 HTTP-206 Range Streaming, and Persistent GDPR Disk Wipe (`v2.3.0`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple
import urllib.error
import urllib.request

import torch
import torch.nn as nn

from scar_cache.cli_scar_store import (
    ScarStoreHTTPRangeServer,
    generate_with_scar_store,
    main as mos_store_main,
    make_http_range_capsule_fetcher,
)
from scar_cache.head_expanded_attention import (
    HeadExpandedAttentionEngine,
    HeadExpandedScarStore,
)
from scar_cache.scar_grafting import SleepEpisode


class _TinyHeadStudentBlock(nn.Module):
    def __init__(self, d: int, n_heads: int, d_head: int) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.self_attn = nn.Module()
        self.self_attn.num_heads = n_heads
        self.self_attn.head_dim = d_head
        self.self_attn.q_proj = nn.Linear(d, n_heads * d_head, bias=False)
        self.self_attn.o_proj = nn.Linear(n_heads * d_head, d, bias=False)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d, bias=False), nn.GELU(), nn.Linear(2 * d, d, bias=False))

    def forward(self, x: torch.Tensor, *_args: object, **_kwargs: object) -> torch.Tensor:
        h = self.ln(x)
        q = self.self_attn.q_proj(h)
        x = x + 0.1 * self.self_attn.o_proj(q)
        return x + 0.1 * self.mlp(x)


class _TinyHeadCausalLM(nn.Module):
    def __init__(self, vocab_size: int = 256, d: int = 128, n_layers: int = 4, n_heads: int = 4, d_head: int = 32) -> None:
        super().__init__()
        torch.manual_seed(17)
        self.config = type(
            "Cfg",
            (),
            {
                "vocab_size": vocab_size,
                "hidden_size": d,
                "num_hidden_layers": n_layers,
                "num_attention_heads": n_heads,
                "num_key_value_heads": n_heads,
            },
        )()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, d)
        self.model.layers = nn.ModuleList([_TinyHeadStudentBlock(d, n_heads, d_head) for _ in range(n_layers)])
        self.model.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)

        def _model_fwd(input_ids: torch.Tensor, past_key_values: Optional[object] = None, **_kw: object) -> torch.Tensor:
            x = self.model.embed_tokens(input_ids)
            for blk in self.model.layers:
                x = blk(x)
            if past_key_values is not None and hasattr(past_key_values, "update"):
                bsz, slen, _ = x.shape
                dummy_kv = x.view(bsz, slen, n_heads, d_head).transpose(1, 2).contiguous()
                for l_i in range(n_layers):
                    past_key_values.update(dummy_kv, dummy_kv, l_i)
            return self.model.norm(x)

        self.model.forward = _model_fwd  # type: ignore[method-assign]

    def forward(self, input_ids: torch.Tensor, past_key_values: Optional[object] = None, **_kw: object) -> object:
        h = self.model(input_ids, past_key_values=past_key_values)
        lg = self.lm_head(h)
        return type("Out", (), {"logits": lg})()


class _TinyTok:
    eos_token_id = 2

    def __init__(self, vocab_size: int = 256, offset: int = 10) -> None:
        self.vocab_size = vocab_size
        self.offset = offset

    def get_vocab(self) -> dict[str, int]:
        return {f"tok{i}": i for i in range(self.vocab_size)}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [max(3, ((ord(c) + self.offset) % (self.vocab_size - 12)) + 10) for c in text[:48]]

    def __call__(self, text: str, return_tensors: str = "pt") -> object:
        ids = self.encode(text)
        return type("Enc", (), {"input_ids": torch.tensor([ids], dtype=torch.long)})()

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(65 + (int(i) % 26)) for i in ids)


def _build_two_capsule_store(tmp_path: Path) -> Tuple[_TinyHeadCausalLM, Path]:
    slm = _TinyHeadCausalLM().eval()
    bg = [torch.randint(10, 200, (1, 16)) for _ in range(4)]
    bank = HeadExpandedAttentionEngine.calibrate_head_expanded_student(
        student_model=slm,
        calibration_prompts=bg,
        user_turn_prefix_ids=[11, 12],
    )
    for idx, cid in enumerate(("cap_alpha", "cap_beta")):
        p_ids = torch.tensor([11, 12] + [30 + idx * 40 + j for j in range(10)], dtype=torch.long)
        ans_ids = torch.tensor([120 + idx * 30 + j for j in range(8)], dtype=torch.long)
        full_ids = torch.cat([p_ids, ans_ids], dim=0)
        t_lg = torch.full((8, 256), -4.0, dtype=torch.float16)
        for s_i, tid in enumerate(ans_ids.tolist()):
            t_lg[s_i, tid] = 14.0
        ep = SleepEpisode(
            capsule_id=cid,
            prompt_ids=p_ids,
            full_ids=full_ids,
            teacher_logits=t_lg,
            query_start_idx=2,
        )
        cap = HeadExpandedAttentionEngine.solve_head_expanded_capsule(
            capsule_id=cid,
            episodes=[ep],
            student_model=slm,
            head_bank=bank,
            rank=16,
            num_key_prototypes=4,
            tau=0.55,
            ramp=0.15,
        )
        bank.add_capsule(cap)

    store_path = tmp_path / "master.scarstore"
    st = HeadExpandedScarStore.build_from_bank(bank, store_path)
    st.close()
    return slm, store_path


def test_http_206_range_server_and_client_live_streaming(tmp_path: Path) -> None:
    """Verifies real TCP RFC 7233 `HTTP/1.1 206 Partial Content` streaming for uninstalled capsules."""
    slm, master_path = _build_two_capsule_store(tmp_path)
    consumer_path = tmp_path / "consumer.scarstore"

    rc = mos_store_main(["slice", "--in-store", str(master_path), "--out-store", str(consumer_path), "--include", "cap_alpha"])
    assert rc == 0

    with ScarStoreHTTPRangeServer(master_path, host="127.0.0.1", port=0) as srv:
        # 1. Verify invalid range returns HTTP 416
        bad_req = urllib.request.Request(f"{srv.base_url}/store.scarstore", headers={"Range": "bytes=99999999-999999999"})
        try:
            urllib.request.urlopen(bad_req, timeout=5.0)
            raise AssertionError("Expected HTTP 416 for out-of-bounds Range")
        except urllib.error.HTTPError as e:
            assert e.code == 416

        # 2. Query missing `cap_beta` on consumer store with HTTP-206 Range fetcher attached
        fetcher, client = make_http_range_capsule_fetcher(srv.base_url)
        c_store = HeadExpandedScarStore.open(consumer_path, device="cpu", on_demand_fetcher=fetcher)
        try:
            tok = _TinyTok()
            q_beta = torch.tensor([[11, 12] + [70 + j for j in range(10)]], dtype=torch.long)
            res, _ = generate_with_scar_store(
                student_model=slm,
                tokenizer=tok,
                store=c_store,
                prompt_text_or_ids=q_beta,
                max_new_tokens=6,
                format_chat=False,
                http_client=client,
                persist_fetched=True,
            )
            assert "cap_beta" in res.active_gates
            assert len(res.http_range_events) == 1
            ev = res.http_range_events[0]
            assert ev["http_status"] == 206
            assert ev["bytes_received"] == c_store.record_bytes
            assert srv.http_206_slices_served == 1
            assert srv.total_range_bytes_served == c_store.record_bytes
        finally:
            c_store.close()

    # 3. Verify `persist_fetched=True` permanently installed `cap_beta` into `consumer.scarstore`
    reopened = HeadExpandedScarStore.open(consumer_path, device="cpu")
    try:
        summary = reopened.inspect_summary()
        assert summary["num_active_installed_slots"] == 2
    finally:
        reopened.close()


def test_mos_store_cli_inspect_and_persistent_gdpr_zero_wipe(tmp_path: Path, capsys: object) -> None:
    """Verifies `mos-store unlearn --persist` zeroes exact slot & anchor bytes on disk while preserving siblings."""
    slm, master_path = _build_two_capsule_store(tmp_path)

    # Inspect before unlearn
    assert mos_store_main(["inspect", "--store", str(master_path), "--json"]) == 0
    out_before = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert out_before["num_active_installed_slots"] == 2
    slot0 = out_before["slots"][0]
    slot1 = out_before["slots"][1]
    assert slot0["payload_all_zero_wiped"] is False
    assert slot1["payload_all_zero_wiped"] is False

    # Capture clean baseline on `cap_alpha` query
    q_alpha = torch.tensor([[11, 12] + [30 + j for j in range(10)]], dtype=torch.long)
    base_lg = slm(q_alpha).logits.detach().clone()

    # Persistent GDPR unlearn of `cap_alpha` via CLI
    assert mos_store_main(["unlearn", "--store", str(master_path), "--capsule-id", "cap_alpha", "--persist", "--json"]) == 0
    unlearn_rep = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert unlearn_rep["persisted_to_disk"] is True
    assert unlearn_rep["cryptographically_zeroed_bytes"] == out_before["record_bytes"] + (out_before["hidden_dim"] * 2)

    # Forensic raw-byte check on disk: slot 0 payload is 100% 0x00; slot 1 payload is untouched!
    with open(master_path, "rb") as rf:
        rf.seek(slot0["byte_offset"])
        raw0 = rf.read(slot0["record_bytes"])
        rf.seek(slot1["byte_offset"])
        raw1 = rf.read(slot1["record_bytes"])
    assert all(b == 0 for b in raw0), "Erased slot payload on disk must be 100% cryptographic 0x00 bytes!"
    assert any(b != 0 for b in raw1), "Sibling capsule payload on disk must remain intact!"

    # Reopen from disk in a fresh instance & verify exact `torch.equal` baseline reversion on `cap_alpha`
    reopened = HeadExpandedScarStore.open(master_path, device="cpu")
    reopened.attach(slm)
    try:
        after_lg = slm(q_alpha).logits.detach().clone()
        assert torch.equal(base_lg, after_lg)
        q_beta = torch.tensor([[11, 12] + [70 + j for j in range(10)]], dtype=torch.long)
        _ = slm(q_beta)
        assert "cap_beta" in reopened.last_active_gates
    finally:
        reopened.close()


def test_open_weights_factory_same_vocab_and_cross_tokenizer_bridge(tmp_path: Path) -> None:
    """Verifies Zero-SGD cartridge minting from same-vocab teachers, cross-tokenizer teachers, and curated JSON specs."""
    from scar_cache.open_weights_factory import (
        collect_open_weight_episodes,
        load_specs_json,
        mint_capsules_from_specs,
    )

    slm = _TinyHeadCausalLM(vocab_size=256, d=128).eval()
    student_tok = _TinyTok(vocab_size=256, offset=10)
    fmt = lambda q: f"USER:{q}\nASSISTANT:"
    bank = HeadExpandedAttentionEngine.calibrate_head_expanded_student(
        student_model=slm,
        calibration_prompts=["alpha query", "beta query", "gamma query", "delta query"],
        tokenizer=student_tok,
        chat_formatter=fmt,
        preserved_rank=2,
    )

    # 1. Same-vocabulary Open-Weight Teacher -> exact dense logit transposition
    same_vocab_teacher = _TinyHeadCausalLM(vocab_size=256, d=128).eval()
    eps_same, mode_same = collect_open_weight_episodes(
        capsule_id="ow_same",
        train_prompts=["What is spherical packing?"],
        student_model=slm,
        student_tokenizer=student_tok,
        teacher_model=same_vocab_teacher,
        teacher_tokenizer=student_tok,
        chat_formatter=fmt,
        max_new_tokens=6,
    )
    assert mode_same == "same_vocab_exact"
    assert len(eps_same) == 1 and eps_same[0].teacher_logits.shape[-1] == 256

    # 2. Cross-tokenizer Open-Weight Teacher (different vocab size 312 & token offset!)
    cross_teacher = _TinyHeadCausalLM(vocab_size=312, d=128).eval()
    cross_tok = _TinyTok(vocab_size=312, offset=37)
    eps_cross, mode_cross = collect_open_weight_episodes(
        capsule_id="ow_cross",
        train_prompts=["Explain lattice reduction"],
        student_model=slm,
        student_tokenizer=student_tok,
        teacher_model=cross_teacher,
        teacher_tokenizer=cross_tok,
        chat_formatter=fmt,
        max_new_tokens=6,
    )
    assert mode_cross == "cross_tokenizer_bridge"
    assert len(eps_cross) == 1 and eps_cross[0].teacher_logits.shape[-1] == 256

    # 3. Curated JSON spec batch minting -> real .scarstore
    spec_file = tmp_path / "specs.json"
    spec_file.write_text(
        json.dumps(
            [
                {
                    "title": "Reforma Tributária — IVA Dual",
                    "category": "Tax Law",
                    "episodes": [
                        {
                            "prompt": "Explain dual VAT in tax reform",
                            "answer": "Dual VAT splits federal CBS and subnational IBS.",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    specs = load_specs_json(spec_file)
    caps = mint_capsules_from_specs(
        specs=specs,
        student_model=slm,
        student_tokenizer=student_tok,
        head_bank=bank,
        chat_formatter=fmt,
        rank=16,
        num_key_prototypes=4,
    )
    assert len(caps) == 1
    assert caps[0].metadata["title"] == "Reforma Tributária — IVA Dual"
    assert caps[0].metadata["mint_mode"] == "curated_trajectory"


def test_scarstore_web_ui_status_chat_pull_mint_and_unlearn(tmp_path: Path) -> None:
    """Verifies `ScarStoreWebUI` (`mos-store ui`) endpoints over real HTTP: status, HTTP-206 pull, mint, chat, and LGPD 0x00."""
    from scar_cache.ui_scar_store import ScarStoreWebUI

    slm, master_path = _build_two_capsule_store(tmp_path)
    consumer_path = tmp_path / "ui_consumer.scarstore"
    assert mos_store_main(["slice", "--in-store", str(master_path), "--out-store", str(consumer_path), "--include", "cap_alpha"]) == 0

    tok = _TinyTok()
    with ScarStoreHTTPRangeServer(master_path, host="127.0.0.1", port=0) as range_srv:
        fetcher, http_client = make_http_range_capsule_fetcher(range_srv.base_url)
        c_store = HeadExpandedScarStore.open(consumer_path, device="cpu", on_demand_fetcher=fetcher)
        with ScarStoreWebUI(
            student_model=slm,
            tokenizer=tok,
            store=c_store,
            student_name="TinyStudent",
            http_client=http_client,
            remote_mode=range_srv.base_url,
            host="127.0.0.1",
            port=0,
            max_new_tokens=6,
        ) as ui:
            # 1. GET / serves visual HTML
            html = urllib.request.urlopen(f"{ui.base_url}/", timeout=5.0).read().decode("utf-8")
            assert "ScarStore" in html

            # 2. GET /api/status shows 1 installed + 1 cloud cartridge
            st = json.loads(urllib.request.urlopen(f"{ui.base_url}/api/status", timeout=5.0).read().decode("utf-8"))
            assert st["installed_count"] == 1
            assert st["total_count"] == 2

            # 3. POST /api/pull_capsule installs `cap_beta` via HTTP 206 Range
            req_pull = urllib.request.Request(
                f"{ui.base_url}/api/pull_capsule",
                data=json.dumps({"capsule_id": "cap_beta"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            res_pull = json.loads(urllib.request.urlopen(req_pull, timeout=5.0).read().decode("utf-8"))
            assert res_pull["ok"] is True
            assert res_pull["bytes_installed"] == ui.store.record_bytes

            # 4. POST /api/mint creates a brand-new Open-Weights cartridge in Zero SGD
            req_mint = urllib.request.Request(
                f"{ui.base_url}/api/mint",
                data=json.dumps(
                    {
                        "title": "Especialista SQL — Window Groups",
                        "category": "Databases",
                        "prompt": "Explain SQL GROUPS frame mode",
                        "answer": "GROUPS counts peer groups in ORDER BY.",
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            res_mint = json.loads(urllib.request.urlopen(req_mint, timeout=5.0).read().decode("utf-8"))
            assert res_mint["ok"] is True
            minted_id = res_mint["capsule_id"]

            # 5. POST /api/chat runs live SCache O(1) generation
            req_chat = urllib.request.Request(
                f"{ui.base_url}/api/chat",
                data=json.dumps({"prompt": "Explain SQL GROUPS frame mode"}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            res_chat = json.loads(urllib.request.urlopen(req_chat, timeout=5.0).read().decode("utf-8"))
            assert res_chat["tokens_generated"] >= 1
            assert res_chat["scache_memory_bytes"] > 0

            # 6. POST /api/unlearn wipes minted cartridge with 0x00 bytes on disk
            req_unlearn = urllib.request.Request(
                f"{ui.base_url}/api/unlearn",
                data=json.dumps({"capsule_id": minted_id, "persist": True}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            res_unlearn = json.loads(urllib.request.urlopen(req_unlearn, timeout=5.0).read().decode("utf-8"))
            assert res_unlearn["persisted_to_disk"] is True
            assert res_unlearn["cryptographically_zeroed_bytes"] > 0
        ui.store.close()
