"""Unified Production CLI & HTTP-206 Range SDK (`mos-store`) for `HeadExpandedScarStore` (`v2.3.0`).

Provides:
  1. `ScarStoreHTTPRangeServer` & `HTTPRangeCapsuleClient`:
     - Standards-compliant RFC 7233 `HTTP/1.1 206 Partial Content` (`Range: bytes=start-end`) server
       and zero-dependency client streaming exact `4096 B`-page-aligned capsule slices (`143,360 B =
       140.00 KB = 35 x 4 KB` on `Qwen2.5-3B-Instruct`; `62,720 B = 61.25 KB` on `Qwen2.5-0.5B-Instruct`)
       on demand when a consumer `.scarstore` encounters an uninstalled Kronecker coordinate.
  2. `generate_with_scar_store(...)`:
     - Full-stack `SCacheDynamicCache O(1)` + `HeadExpandedScarStore` chunked-prefill & autoregressive
       generation returning live `ScarStoreQueryResult` telemetry (`+M` expanded Scar-Attention heads,
       inter-head specialization spread, token-by-token weight switching, `mmap` bytes paged, `SCache` KB).
  3. `mos-store` CLI (`build`, `inspect`, `slice`, `serve`, `query`, `sleep-bridge`, `unlearn`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import urllib.request

import torch
import torch.nn as nn

from scar_cache.head_expanded_attention import (
    HeadExpandedAttentionEngine,
    HeadExpandedScarCapsule,
    HeadExpandedScarStore,
    HeadExpandedTelemetry,
)
from scar_cache.hf_cache import SCacheDynamicCache
from scar_cache.scar_grafting import (
     ScarGraftEngine,
    SleepEpisode,
    _infer_model_device,
)

DEFAULT_CALIBRATION_PROMPTS: Tuple[str, ...] = (
    "What is the capital of France?",
    "Write a Python lambda function that squares x.",
    "Solve the linear equation 3x + 7 = 22.",
    "What gas do green plants absorb from the atmosphere during photosynthesis?",
    "Explain briefly how the TCP three-way handshake establishes a connection.",
    "Who wrote the Brazilian novel Dom Casmurro?",
)

DEFAULT_SYSTEM_PROMPT = "Answer concisely in one precise technical sentence."


def format_chatml_prompt(user_query: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{user_query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_episodes_bundle(episodes_pt_path: Union[str, Path]) -> Dict[str, List[SleepEpisode]]:
    """Loads cached `SleepEpisode` dict `{capsule_id: [SleepEpisode, ...]}` from a `.pt` bundle.

    Supports both `{capsule_id: [SleepEpisode, ...]}` AND `{"domains": {capsule_id: {"episodes": [...]}}}`
    (such as `results/titan72b_episodes_v2.pt`).
    """
    raw = torch.load(Path(episodes_pt_path), map_location="cpu", weights_only=False)
    grouped: Dict[str, List[SleepEpisode]] = {}
    if isinstance(raw, dict):
        if "domains" in raw and isinstance(raw["domains"], dict):
            items = {cid: dinfo.get("episodes", []) for cid, dinfo in raw["domains"].items() if isinstance(dinfo, dict)}
        else:
            items = raw.get("episodes_by_capsule", raw)
        for cid, ep_list in items.items():
            if not isinstance(ep_list, (list, tuple)):
                continue
            parsed: List[SleepEpisode] = []
            for item in ep_list:
                if isinstance(item, SleepEpisode):
                    parsed.append(item)
                elif isinstance(item, dict):
                    parsed.append(
                        SleepEpisode(
                            capsule_id=str(item.get("capsule_id", cid)),
                            prompt_ids=item["prompt_ids"].long(),
                            full_ids=item["full_ids"].long(),
                            teacher_logits=item["teacher_logits"].to(torch.float16),
                            query_start_idx=int(item.get("query_start_idx", 0)),
                            surprise_score=float(item.get("surprise_score", 1.0)),
                            metadata=dict(item.get("metadata", {})),
                        )
                    )
            if parsed:
                grouped[str(cid)] = parsed
    return grouped


_RANGE_HEADER_RE = re.compile(r"^bytes=(\d+)-(\d*)$")


class ScarStoreHTTPRangeServer:
    """RFC 7233 `HTTP/1.1 206 Partial Content` server streaming exact `.scarstore` capsule ranges."""

    def __init__(
        self,
        store_path: Union[str, Path],
        host: str = "127.0.0.1",
        port: int = 0,
    ) -> None:
        self.store_path = Path(store_path).resolve()
        if not self.store_path.exists():
            raise FileNotFoundError(f"Master ScarStore file not found: {self.store_path}")
        store_inspect = HeadExpandedScarStore.open(self.store_path, device="cpu")
        try:
            self.summary = store_inspect.inspect_summary()
        finally:
            store_inspect.close()

        self.file_bytes = int(self.store_path.stat().st_size)
        self.requests_served: int = 0
        self.http_206_slices_served: int = 0
        self.total_range_bytes_served: int = 0
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        parent = self

        class _RangeHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return  # Silence noisy stdout logs during benchmarks & CLI queries

            def do_GET(self) -> None:  # noqa: N802
                with parent._lock:
                    parent.requests_served += 1

                if self.path in ("/manifest.json", "/manifest"):
                    payload = json.dumps(parent.summary, ensure_ascii=False).encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                if self.path not in ("/store.scarstore", "/", f"/{parent.store_path.name}"):
                    self.send_error(HTTPStatus.NOT_FOUND, "Unknown ScarStore resource")
                    return

                range_hdr = self.headers.get("Range")
                if range_hdr:
                    m = _RANGE_HEADER_RE.match(range_hdr.strip())
                    if not m:
                        self.send_error(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, "Malformed Range header")
                        return
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else (parent.file_bytes - 1)
                    if start < 0 or end >= parent.file_bytes or start > end:
                        self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                        self.send_header("Content-Range", f"bytes */{parent.file_bytes}")
                        self.send_header("Content-Length", "0")
                        self.end_headers()
                        return
                    length = end - start + 1
                    with open(parent.store_path, "rb") as rf:
                        rf.seek(start)
                        chunk = rf.read(length)
                    with parent._lock:
                        parent.http_206_slices_served += 1
                        parent.total_range_bytes_served += len(chunk)
                    self.send_response(HTTPStatus.PARTIAL_CONTENT)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Range", f"bytes {start}-{start + len(chunk) - 1}/{parent.file_bytes}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                    return

                with open(parent.store_path, "rb") as rf:
                    full_data = rf.read()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(len(full_data)))
                self.end_headers()
                self.wfile.write(full_data)

        self._httpd = ThreadingHTTPServer((host, int(port)), _RangeHandler)
        self.host, self.port = self._httpd.server_address[0], int(self._httpd.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start_background(self) -> "ScarStoreHTTPRangeServer":
        if self._thread is None:
            self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
            self._thread.start()
        return self

    def serve_forever(self) -> None:
        self._httpd.serve_forever()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "ScarStoreHTTPRangeServer":
        return self.start_background()

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


class HTTPRangeCapsuleClient:
    """Zero-dependency HTTP/1.1 `206 Partial Content` capsule fetcher keyed by Kronecker `(row_i, col_j)`."""

    def __init__(self, base_url: str, timeout_s: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self._manifest_cache: Optional[Dict[Tuple[int, int], Dict[str, Any]]] = None
        self.fetch_events: List[Dict[str, Any]] = []

    def _ensure_manifest(self) -> Dict[Tuple[int, int], Dict[str, Any]]:
        if self._manifest_cache is not None:
            return self._manifest_cache
        req = urllib.request.Request(f"{self.base_url}/manifest.json", method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        mapping: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for slot in data.get("slots", []):
            if not slot.get("bitmap_present", True) or slot.get("payload_all_zero_wiped", False):
                continue
            coord = (int(slot["coord"][0]), int(slot["coord"][1]))
            mapping[coord] = dict(slot)
        self._manifest_cache = mapping
        return mapping

    def fetch_capsule_bytes(self, coord: Tuple[int, int], capsule_id: str) -> Optional[bytes]:
        t0 = time.perf_counter()
        manifest = self._ensure_manifest()
        slot = manifest.get((int(coord[0]), int(coord[1])))
        if slot is None:
            return None
        off = int(slot["byte_offset"])
        rec = int(slot["record_bytes"])
        range_spec = f"bytes={off}-{off + rec - 1}"
        req = urllib.request.Request(
            f"{self.base_url}/store.scarstore",
            headers={"Range": range_spec},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            status = int(getattr(resp, "status", resp.getcode()))
            content_range = resp.headers.get("Content-Range", "")
            body = resp.read()
        if status != HTTPStatus.PARTIAL_CONTENT or len(body) != rec:
            raise RuntimeError(
                f"Expected HTTP 206 Partial Content of {rec} B for {capsule_id!r}, "
                f"got status={status}, len={len(body)}"
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.fetch_events.append(
            {
                "capsule_id": capsule_id,
                "coord": [int(coord[0]), int(coord[1])],
                "http_status": status,
                "range_header": range_spec,
                "content_range": content_range,
                "bytes_received": len(body),
                "latency_ms": round(elapsed_ms, 3),
            }
        )
        return body


def make_http_range_capsule_fetcher(
    base_url: str,
    timeout_s: float = 10.0,
) -> Tuple[Callable[[Tuple[int, int], str], Optional[bytes]], HTTPRangeCapsuleClient]:
    client = HTTPRangeCapsuleClient(base_url=base_url, timeout_s=timeout_s)
    return client.fetch_capsule_bytes, client


@dataclass
class ScarStoreQueryResult:
    """Telemetry + generated answer from `generate_with_scar_store`."""

    text: str
    tokens_generated: int
    tok_per_s: float
    elapsed_ms: float
    active_gates: Dict[str, float]
    routing_cosines: Dict[str, float]
    num_expanded_heads: int
    mmap_bytes_read: int
    max_inter_head_spread: float
    per_capsule_mean_weight: Dict[str, float]
    per_capsule_min_weight: Dict[str, float]
    per_capsule_max_weight: Dict[str, float]
    scache_memory_bytes: int
    scache_memory_kb: float
    missing_alerts: List[Dict[str, Any]]
    http_range_events: List[Dict[str, Any]]


@torch.no_grad()
def generate_with_scar_store(
    student_model: nn.Module,
    tokenizer: Any,
    store: HeadExpandedScarStore,
    prompt_text_or_ids: Union[str, torch.Tensor],
    max_new_tokens: int = 48,
    scache: Optional[SCacheDynamicCache] = None,
    window_size: int = 256,
    num_slots: int = 64,
    prefill_chunk_size: int = 64,
    temperature: float = 0.0,
    format_chat: bool = True,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    http_client: Optional[HTTPRangeCapsuleClient] = None,
    persist_fetched: bool = False,
) -> Tuple[ScarStoreQueryResult, SCacheDynamicCache]:
    """Executes chunked prefill + decode with `SCache O(1)` and `HeadExpandedScarStore` attached."""
    dev = _infer_model_device(student_model)
    cache = scache if scache is not None else SCacheDynamicCache(window_size=window_size, num_slots=num_slots)
    was_attached = store._norm_hook_handle is not None
    if not was_attached:
        store.attach(student_model)

    prior_http_count = len(http_client.fetch_events) if http_client is not None else 0

    try:
        if isinstance(prompt_text_or_ids, str):
            raw_str = format_chatml_prompt(prompt_text_or_ids, system_prompt=system_prompt) if format_chat else prompt_text_or_ids
            ids_2d = tokenizer(raw_str, return_tensors="pt").input_ids.to(dev)
        else:
            ids_2d = (prompt_text_or_ids if prompt_text_or_ids.ndim == 2 else prompt_text_or_ids.unsqueeze(0)).to(dev)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        t0 = time.perf_counter()

        seq_len = int(ids_2d.shape[1])
        chunk_sz = max(1, int(prefill_chunk_size))
        last_lg: Optional[torch.Tensor] = None
        for st in range(0, seq_len, chunk_sz):
            chunk = ids_2d[:, st : min(seq_len, st + chunk_sz)]
            out = student_model(chunk, past_key_values=cache, use_cache=True)
            last_lg = (out.logits if hasattr(out, "logits") else out)[:, -1, :]
        assert last_lg is not None

        def _pick_token(lg_1d: torch.Tensor) -> int:
            if temperature <= 1e-6:
                return int(lg_1d.float().argmax(dim=-1).item())
            probs = torch.softmax(lg_1d.float() / float(temperature), dim=-1)
            return int(torch.multinomial(probs, num_samples=1).item())

        nxt = _pick_token(last_lg[0])
        gen_ids: List[int] = [nxt]
        for _ in range(max(0, int(max_new_tokens) - 1)):
            if eos_id is not None and nxt == int(eos_id):
                break
            out = student_model(
                torch.tensor([[nxt]], dtype=torch.long, device=dev),
                past_key_values=cache,
                use_cache=True,
            )
            lg_step = (out.logits if hasattr(out, "logits") else out)[0, -1]
            nxt = _pick_token(lg_step)
            gen_ids.append(nxt)
            if eos_id is not None and nxt == int(eos_id):
                break

        elapsed_s = max(1e-6, time.perf_counter() - t0)
        telem: HeadExpandedTelemetry = store.get_last_telemetry()
        snap_active_gates = {k: round(float(v), 4) for k, v in store.last_active_gates.items()}
        snap_routing_cosines = {k: round(float(v), 4) for k, v in store.last_routing_cosines.items()}
        snap_num_heads = int(store.last_num_expanded_heads)
        snap_mmap_bytes = int(store.last_mmap_bytes_read)
        snap_missing_alerts = [
            {
                "coord": [int(a.coord[0]), int(a.coord[1])],
                "capsule_id": a.capsule_id,
                "coarse_cosine": round(float(a.coarse_cosine), 4),
                "tau": round(float(a.tau), 4),
            }
            for a in store.last_missing_alerts
        ]
        new_http_events = (
            list(http_client.fetch_events[prior_http_count:]) if http_client is not None else []
        )

        if persist_fetched and new_http_events:
            for ev in new_http_events:
                cid_f = str(ev["capsule_id"])
                if cid_f in store._hot_fetched_capsules:
                    cap_hot = store._hot_fetched_capsules[cid_f]
                    store = store.append_capsule(cap_hot)
                    if was_attached:
                        store.attach(student_model)

        res = ScarStoreQueryResult(
            text=tokenizer.decode(gen_ids, skip_special_tokens=True).strip(),
            tokens_generated=len(gen_ids),
            tok_per_s=round(len(gen_ids) / elapsed_s, 2),
            elapsed_ms=round(elapsed_s * 1000.0, 2),
            active_gates=snap_active_gates,
            routing_cosines=snap_routing_cosines,
            num_expanded_heads=snap_num_heads,
            mmap_bytes_read=snap_mmap_bytes,
            max_inter_head_spread=float(telem.max_inter_head_spread),
            per_capsule_mean_weight=dict(telem.per_capsule_mean_weight),
            per_capsule_min_weight=dict(telem.per_capsule_min_weight),
            per_capsule_max_weight=dict(telem.per_capsule_max_weight),
            scache_memory_bytes=int(cache.total_memory_bytes),
            scache_memory_kb=round(cache.total_memory_bytes / 1024.0, 2),
            missing_alerts=snap_missing_alerts,
            http_range_events=new_http_events,
        )
        return res, cache
    finally:
        if not was_attached:
            store.detach()


def _resolve_device_and_dtype(device_arg: Optional[str] = None) -> Tuple[torch.device, torch.dtype]:
    if device_arg:
        dev = torch.device(device_arg)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")
    dtype = torch.float16 if dev.type in ("cuda", "mps") else torch.float32
    return dev, dtype


def _cmd_build(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev, dtype = _resolve_device_and_dtype(args.device)
    tok = AutoTokenizer.from_pretrained(args.student, local_files_only=args.local_files_only)
    slm = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=dtype, local_files_only=args.local_files_only
    ).to(dev).eval()

    fmt = lambda q: format_chatml_prompt(q)
    head_bank = HeadExpandedAttentionEngine.calibrate_head_expanded_student(
        student_model=slm,
        calibration_prompts=DEFAULT_CALIBRATION_PROMPTS,
        tokenizer=tok,
        chat_formatter=fmt,
        preserved_rank=4,
    )

    episodes_by_cid: Dict[str, List[SleepEpisode]] = {}
    if args.episodes_pt:
        episodes_by_cid.update(load_episodes_bundle(args.episodes_pt))

    if args.teacher and args.capsule_id and args.prompt:
        llm = AutoModelForCausalLM.from_pretrained(
            args.teacher, dtype=dtype, local_files_only=args.local_files_only
        ).to(dev).eval()
        eps = ScarGraftEngine.collect_teacher_episodes(
            capsule_id=args.capsule_id,
            train_prompts=args.prompt,
            teacher_model=llm,
            tokenizer=tok,
            chat_formatter=fmt,
            max_new_tokens=args.max_new_tokens,
            user_turn_prefix_ids=head_bank.user_turn_prefix_ids,
        )
        episodes_by_cid[args.capsule_id] = eps

    if not episodes_by_cid:
        raise ValueError("Provide --episodes-pt and/or (--teacher --capsule-id --prompt ...) to build capsules.")

    for cid, eps in episodes_by_cid.items():
        cap = HeadExpandedAttentionEngine.solve_head_expanded_capsule(
            capsule_id=cid,
            episodes=eps,
            student_model=slm,
            head_bank=head_bank,
            rank=args.rank,
            num_key_prototypes=args.prototypes,
            tau=args.tau,
            ramp=args.ramp,
        )
        head_bank.add_capsule(cap)

    store = HeadExpandedScarStore.build_from_bank(head_bank, args.out)
    summary = store.inspect_summary()
    store.close()
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(
            f"[mos-store build] Built {summary['store_path']} | "
            f"slots={summary['num_active_installed_slots']} | "
            f"record={summary['record_bytes']} B ({summary['record_kb']} KB, "
            f"exact_4kb_pages={summary['os_4kb_pages_per_capsule']}) | "
            f"grid={summary['kronecker_grid_cells']:,} cells ({summary['presence_bitmap_kb']} KB bitmap)"
        )
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    store = HeadExpandedScarStore.open(args.store, device="cpu")
    try:
        summary = store.inspect_summary()
    finally:
        store.close()
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(f"ScarStore: {summary['store_path']} ({summary['file_mib']} MiB)")
        print(
            f"  Geometry : d={summary['hidden_dim']} | heads={summary['num_heads']}x{summary['head_dim']} | "
            f"rank={summary['rank']} | P={summary['num_key_prototypes']}"
        )
        print(
            f"  Capsule  : {summary['record_bytes']} B ({summary['record_kb']} KB) | "
            f"4KB-multiple={summary['exact_4kb_page_multiple']} (pages={summary['os_4kb_pages_per_capsule']})"
        )
        print(
            f"  Kronecker: {summary['kronecker_grid_cells']:,} cells | "
            f"bitmap={summary['presence_bitmap_kb']} KB | installed={summary['num_active_installed_slots']}"
        )
        for s in summary["slots"]:
            print(
                f"    - [{s['slot_idx']}] {s['capsule_id']} @ {tuple(s['coord'])} | "
                f"off={s['byte_offset']} (4KB={s['offset_4kb_aligned']}) | "
                f"present={s['bitmap_present']} | wiped={s['payload_all_zero_wiped']}"
            )
    return 0


def _cmd_slice(args: argparse.Namespace) -> int:
    master = HeadExpandedScarStore.open(args.in_store, device="cpu")
    try:
        sliced = master.export_subset(args.out_store, include_capsule_ids=args.include or [])
        summary = sliced.inspect_summary()
        sliced.close()
    finally:
        master.close()
    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(
            f"[mos-store slice] Exported {summary['store_path']} | "
            f"installed={summary['num_active_installed_slots']}/{summary['num_global_directory_entries']} "
            f"({summary['file_mib']} MiB)"
        )
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    srv = ScarStoreHTTPRangeServer(args.store, host=args.host, port=args.port)
    print(
        f"[mos-store serve] Serving {srv.store_path} ({srv.file_bytes:,} B) via HTTP/1.1 206 Range at {srv.base_url}"
    )
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.stop()
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev, dtype = _resolve_device_and_dtype(args.device)
    tok = AutoTokenizer.from_pretrained(args.student, local_files_only=args.local_files_only)
    slm = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=dtype, local_files_only=args.local_files_only
    ).to(dev).eval()

    http_client: Optional[HTTPRangeCapsuleClient] = None
    fetcher: Optional[Callable[[Tuple[int, int], str], Optional[bytes]]] = None
    if args.remote_url:
        fetcher, http_client = make_http_range_capsule_fetcher(args.remote_url)

    store = HeadExpandedScarStore.open(args.store, device=dev, on_demand_fetcher=fetcher)
    store.strict_missing_guard = bool(args.strict_missing_guard)
    store.attach(slm)

    try:
        if args.chat:
            scache = SCacheDynamicCache(window_size=args.window_size, num_slots=args.num_slots)
            print(f"[mos-store chat] Ready on {args.student} ({dev}) + {args.store}. Type 'exit' to quit.")
            while True:
                try:
                    q_line = input("\nYou > ").strip()
                except EOFError:
                    break
                if not q_line or q_line.lower() in ("exit", "quit", ":q"):
                    break
                res, scache = generate_with_scar_store(
                    student_model=slm,
                    tokenizer=tok,
                    store=store,
                    prompt_text_or_ids=q_line,
                    max_new_tokens=args.max_new_tokens,
                    scache=scache,
                    http_client=http_client,
                    persist_fetched=args.persist_fetched,
                )
                print(f"Assistant > {res.text}")
                print(
                    f"  [Telemetry] +Heads={res.num_expanded_heads} | gates={res.active_gates} | "
                    f"spread={res.max_inter_head_spread:.4f} | mmap={res.mmap_bytes_read:,} B | "
                    f"SCache={res.scache_memory_kb:.1f} KB | {res.tok_per_s:.1f} tok/s"
                )
            return 0

        if not args.prompt:
            raise ValueError("Provide --prompt '...' or --chat")
        res, _ = generate_with_scar_store(
            student_model=slm,
            tokenizer=tok,
            store=store,
            prompt_text_or_ids=args.prompt,
            max_new_tokens=args.max_new_tokens,
            window_size=args.window_size,
            num_slots=args.num_slots,
            http_client=http_client,
            persist_fetched=args.persist_fetched,
        )
        if args.json:
            print(json.dumps(asdict(res), indent=2, ensure_ascii=False))
        else:
            print(res.text)
            print(
                f"[mos-store telemetry] +Heads={res.num_expanded_heads} | gates={res.active_gates} | "
                f"spread={res.max_inter_head_spread:.4f} | mmap={res.mmap_bytes_read:,} B | "
                f"HTTP206={len(res.http_range_events)} | SCache={res.scache_memory_kb:.1f} KB | {res.tok_per_s:.1f} tok/s"
            )
        return 0
    finally:
        store.close()


def _cmd_sleep_bridge(args: argparse.Namespace) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev, dtype = _resolve_device_and_dtype(args.device)
    tok = AutoTokenizer.from_pretrained(args.student, local_files_only=args.local_files_only)
    slm = AutoModelForCausalLM.from_pretrained(
        args.student, dtype=dtype, local_files_only=args.local_files_only
    ).to(dev).eval()
    fmt = lambda q: format_chatml_prompt(q)

    store = HeadExpandedScarStore.open(args.store, device="cpu")
    try:
        head_bank = store.to_head_bank()
        if args.episodes_pt:
            ep_map = load_episodes_bundle(args.episodes_pt)
            bridge_eps = ep_map.get(args.bridge_id) or next(iter(ep_map.values()))
        elif args.teacher and args.prompt:
            llm = AutoModelForCausalLM.from_pretrained(
                args.teacher, dtype=dtype, local_files_only=args.local_files_only
            ).to(dev).eval()
            bridge_eps = ScarGraftEngine.collect_teacher_episodes(
                capsule_id=args.bridge_id,
                train_prompts=[args.prompt],
                teacher_model=llm,
                tokenizer=tok,
                chat_formatter=fmt,
                max_new_tokens=args.max_new_tokens,
                user_turn_prefix_ids=head_bank.user_turn_prefix_ids,
            )
        else:
            raise ValueError("Provide --episodes-pt or (--teacher and --prompt) for sleep-bridge.")

        bridge_cap = HeadExpandedAttentionEngine.consolidate_interdisciplinary_bridge(
            bridge_capsule_id=args.bridge_id,
            bridge_episodes=bridge_eps,
            student_model=slm,
            head_bank=head_bank,
            parent_capsules=args.parent if args.parent else None,
            rank=store.rank,
            num_key_prototypes=store.num_key_prototypes,
            tau=args.tau,
            ramp=args.ramp,
        )
        updated = store.append_capsule(bridge_cap, out_path=args.out_store)
        summary = updated.inspect_summary()
        updated.close()
    finally:
        store.close()

    if args.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(
            f"[mos-store sleep-bridge] Consolidated {bridge_cap.capsule_id!r} "
            f"(parents={bridge_cap.parent_capsules}, {store.record_bytes} B) -> {summary['store_path']}"
        )
    return 0


def _cmd_unlearn(args: argparse.Namespace) -> int:
    store = HeadExpandedScarStore.open(args.store, device="cpu")
    try:
        report = store.unlearn(args.capsule_id, persist_to_disk=bool(args.persist))
    finally:
        store.close()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(
            f"[mos-store unlearn] Erased {report['capsule_id']!r} @ {report['coord']} | "
            f"ram_ms={report['unlearn_ms']} ms | persisted={report['persisted_to_disk']} | "
            f"zeroed_bytes={report['cryptographically_zeroed_bytes']:,} B"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mos-store",
        description="HeadExpandedScarStore Production CLI: mmap 4KB-aligned Scar Cartridges, HTTP-206 Range Streaming, Sleep Bridges & <20us GDPR Unlearning.",
    )
    sub = p.add_subparsers(dest="subcommand", required=True)

    pb = sub.add_parser("build", help="Build a .scarstore from SleepEpisode .pt bundles and/or live Teacher prompts")
    pb.add_argument("--student", required=True, help="Student HF model ID (e.g. Qwen/Qwen2.5-3B-Instruct)")
    pb.add_argument("--out", required=True, help="Output .scarstore path")
    pb.add_argument("--episodes-pt", default=None, help="Cached SleepEpisode .pt bundle")
    pb.add_argument("--teacher", default=None, help="Optional live Teacher HF model ID")
    pb.add_argument("--capsule-id", default=None, help="Capsule ID when synthesizing from --teacher")
    pb.add_argument("--prompt", action="append", default=None, help="Training prompt(s) for --teacher")
    pb.add_argument("--rank", type=int, default=60)
    pb.add_argument("--prototypes", type=int, default=8)
    pb.add_argument("--tau", type=float, default=0.62)
    pb.add_argument("--ramp", type=float, default=0.16)
    pb.add_argument("--max-new-tokens", type=int, default=36)
    pb.add_argument("--device", default=None)
    pb.add_argument("--local-files-only", action="store_true")
    pb.add_argument("--json", action="store_true")
    pb.set_defaults(func=_cmd_build)

    pi = sub.add_parser("inspect", help="Inspect physical 4KB alignment, Kronecker grid, and slot wipe status")
    pi.add_argument("--store", required=True, help="Path to .scarstore")
    pi.add_argument("--json", action="store_true")
    pi.set_defaults(func=_cmd_inspect)

    psl = sub.add_parser("slice", help="Export a selective consumer .scarstore (0 B on disk for omitted capsules)")
    psl.add_argument("--in-store", required=True, help="Master .scarstore path")
    psl.add_argument("--out-store", required=True, help="Consumer .scarstore output path")
    psl.add_argument("--include", action="append", default=[], help="Capsule ID to include locally (repeatable)")
    psl.add_argument("--json", action="store_true")
    psl.set_defaults(func=_cmd_slice)

    psv = sub.add_parser("serve", help="Serve a master .scarstore over RFC 7233 HTTP/1.1 206 Partial Content")
    psv.add_argument("--store", required=True, help="Master .scarstore path")
    psv.add_argument("--host", default="127.0.0.1")
    psv.add_argument("--port", type=int, default=8765)
    psv.set_defaults(func=_cmd_serve)

    pq = sub.add_parser("query", help="Query or chat with SCache O(1) + HeadExpandedScarStore + optional HTTP-206 Range")
    pq.add_argument("--student", required=True, help="Student HF model ID (e.g. Qwen/Qwen2.5-3B-Instruct)")
    pq.add_argument("--store", required=True, help="Local .scarstore path")
    pq.add_argument("--prompt", default=None, help="Single query string")
    pq.add_argument("--chat", action="store_true", help="Interactive multi-turn SCache O(1) chat loop")
    pq.add_argument("--remote-url", default=None, help="HTTP-206 Range server URL for missing capsules")
    pq.add_argument("--persist-fetched", action="store_true", help="Persist HTTP-fetched capsules into local store")
    pq.add_argument("--strict-missing-guard", action="store_true", help="Raise error on uninstalled capsule trigger")
    pq.add_argument("--max-new-tokens", type=int, default=48)
    pq.add_argument("--window-size", type=int, default=256)
    pq.add_argument("--num-slots", type=int, default=64)
    pq.add_argument("--device", default=None)
    pq.add_argument("--local-files-only", action="store_true")
    pq.add_argument("--json", action="store_true")
    pq.set_defaults(func=_cmd_query)

    psb = sub.add_parser("sleep-bridge", help="Consolidate an interdisciplinary Sleep Bridge capsule into a .scarstore")
    psb.add_argument("--student", required=True)
    psb.add_argument("--store", required=True)
    psb.add_argument("--bridge-id", required=True)
    psb.add_argument("--parent", action="append", default=[])
    psb.add_argument("--episodes-pt", default=None)
    psb.add_argument("--teacher", default=None)
    psb.add_argument("--prompt", default=None)
    psb.add_argument("--out-store", default=None)
    psb.add_argument("--tau", type=float, default=0.62)
    psb.add_argument("--ramp", type=float, default=0.16)
    psb.add_argument("--max-new-tokens", type=int, default=48)
    psb.add_argument("--device", default=None)
    psb.add_argument("--local-files-only", action="store_true")
    psb.add_argument("--json", action="store_true")
    psb.set_defaults(func=_cmd_sleep_bridge)

    pu = sub.add_parser("unlearn", help="Surgical <20us GDPR unlearn + optional in-place 0x00 disk page erasure")
    pu.add_argument("--store", required=True)
    pu.add_argument("--capsule-id", required=True)
    pu.add_argument("--persist", action="store_true", help="Cryptographically zero slot pages & bitmap on disk")
    pu.add_argument("--json", action="store_true")
    pu.set_defaults(func=_cmd_unlearn)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
