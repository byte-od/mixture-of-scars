"""Zero-Dependency Visual Cartridge Store & Live `SCache O(1)` Chat UI (`mos-store ui`, `v3.0.0`).

Provides an App-Store-style local web interface (`http://127.0.0.1:7860`) backed 100% by the real
`HeadExpandedScarStore`, `SCacheDynamicCache O(1)`, `HTTPRangeCapsuleClient` (`HTTP/1.1 206`),
and `open_weights_factory` (`Zero SGD`), with ZERO external web framework dependencies.
"""

from __future__ import annotations

from dataclasses import asdict
import gc
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import webbrowser

import torch
import torch.nn as nn

from scar_cache.cli_scar_store import (
    DEFAULT_CALIBRATION_PROMPTS,
    DEFAULT_SYSTEM_PROMPT,
    HTTPRangeCapsuleClient,
    OFFICIAL_CATALOG_URL,
    download_cloud_scarstore,
    fetch_official_catalog,
    format_chatml_prompt,
    generate_with_scar_store,
    make_http_range_capsule_fetcher,
    resolve_official_cloud_tier,
    select_chat_system_prompt,
)
from scar_cache.head_expanded_attention import (
    HeadExpandedAttentionEngine,
    HeadExpandedScarCapsule,
    HeadExpandedScarStore,
)
from scar_cache.hf_cache import SCacheDynamicCache
from scar_cache.mos_student import (
    get_accelerator_memory_telemetry,
    load_mos_student_model,
)
from scar_cache.open_weights_factory import (
    discover_local_open_weight_models,
    mint_capsules_from_specs,
)


_UI_HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Mixture of Scars v3.0 — ScarStore &amp; Chat O(1)</title>
<style>
  :root {
    --bg: #0b0f17;
    --panel: #121926;
    --card: #172033;
    --card-hover: #1d2840;
    --border: #26334d;
    --text: #e6edf7;
    --muted: #93a4c3;
    --accent: #38bdf8;
    --accent-2: #818cf8;
    --green: #34d399;
    --amber: #fbbf24;
    --red: #f87171;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.45;
  }
  header {
    background: linear-gradient(90deg, #0f172a 0%, #162038 100%);
    border-bottom: 1px solid var(--border);
    padding: 14px 24px;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }
  .brand { display: flex; align-items: center; gap: 12px; }
  .brand-badge {
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    color: #06101f;
    font-weight: 800;
    padding: 6px 10px;
    border-radius: 8px;
    font-size: 13px;
    letter-spacing: 0.4px;
  }
  .brand h1 { margin: 0; font-size: 17px; font-weight: 700; }
  .brand p { margin: 2px 0 0; font-size: 12px; color: var(--muted); }
  .metrics-bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .pill {
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.25);
    color: var(--text);
    padding: 5px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }
  .pill strong { color: var(--accent); }
  .btn {
    border: 1px solid var(--border);
    background: var(--card);
    color: var(--text);
    padding: 7px 12px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 12.5px;
    font-weight: 600;
    transition: all 0.15s ease;
  }
  .btn:hover { background: var(--card-hover); border-color: var(--accent); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-primary {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    border: none;
    color: #fff;
  }
  .btn-primary:hover { filter: brightness(1.08); }
  .btn-danger {
    background: rgba(248, 113, 113, 0.12);
    border-color: rgba(248, 113, 113, 0.35);
    color: #fca5a5;
  }
  .btn-danger:hover { background: rgba(248, 113, 113, 0.24); border-color: var(--red); }
  .btn-green {
    background: rgba(52, 211, 153, 0.12);
    border-color: rgba(52, 211, 153, 0.35);
    color: #6ee7b7;
  }
  main {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 18px;
    padding: 18px 24px;
    height: calc(100vh - 74px);
  }
  @media (max-width: 1024px) {
    main { grid-template-columns: 1fr; height: auto; }
  }
  .pane {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .pane-header {
    padding: 14px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: 10px;
  }
  .pane-header h2 { margin: 0; font-size: 15px; font-weight: 700; }
  .toolbar {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
  }
  .search-input {
    flex: 1;
    min-width: 180px;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 7px 11px;
    border-radius: 8px;
    font-size: 13px;
  }
  .cards-scroll {
    padding: 14px 16px;
    overflow-y: auto;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 12px;
    align-content: start;
    flex: 1;
  }
  .card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 13px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 10px;
    transition: border-color 0.15s ease;
  }
  .card.active-lit {
    border-color: var(--accent);
    box-shadow: 0 0 0 1px rgba(56, 189, 248, 0.35), 0 8px 24px rgba(14, 165, 233, 0.15);
  }
  .card-top { display: flex; justify-content: space-between; gap: 8px; align-items: flex-start; }
  .cat-tag {
    font-size: 11px;
    color: var(--accent);
    background: rgba(56, 189, 248, 0.1);
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 600;
  }
  .status-tag {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 6px;
    font-weight: 600;
  }
  .st-local { background: rgba(52, 211, 153, 0.14); color: var(--green); }
  .st-cloud { background: rgba(129, 140, 248, 0.16); color: #a5b4fc; }
  .st-ram { background: rgba(251, 191, 36, 0.15); color: var(--amber); }
  .st-wiped { background: rgba(248, 113, 113, 0.15); color: var(--red); }
  .card h3 { margin: 6px 0 3px; font-size: 14px; font-weight: 700; }
  .card .sub { font-size: 12px; color: var(--muted); margin: 0; }
  .meta-line {
    font-size: 11px;
    color: var(--muted);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }
  .card-actions { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 2px; }
  .chat-history {
    flex: 1;
    overflow-y: auto;
    padding: 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .msg {
    padding: 11px 14px;
    border-radius: 12px;
    font-size: 13.5px;
    max-width: 94%;
  }
  .msg-user {
    align-self: flex-end;
    background: linear-gradient(135deg, #0284c7, #4f46e5);
    color: #fff;
  }
  .msg-bot {
    align-self: flex-start;
    background: var(--card);
    border: 1px solid var(--border);
  }
  .telem-box {
    margin-top: 8px;
    padding-top: 8px;
    border-top: 1px dashed var(--border);
    font-size: 11.5px;
    color: var(--muted);
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
  .telem-chip {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid var(--border);
    padding: 2px 7px;
    border-radius: 6px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }
  .telem-chip.hit { border-color: rgba(52, 211, 153, 0.5); color: #6ee7b7; }
  .telem-chip.http { border-color: rgba(251, 191, 36, 0.5); color: var(--amber); }
  .chat-composer {
    padding: 12px 16px;
    border-top: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .chat-row { display: flex; gap: 8px; }
  textarea#chatInput {
    flex: 1;
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 10px;
    padding: 9px 12px;
    font-size: 13.5px;
    resize: none;
    height: 54px;
    font-family: inherit;
  }
  .modal-backdrop {
    display: none;
    position: fixed;
    inset: 0;
    background: rgba(3, 7, 18, 0.78);
    align-items: center;
    justify-content: center;
    z-index: 50;
    padding: 16px;
  }
  .modal-backdrop.open { display: flex; }
  .modal {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 14px;
    max-width: 600px;
    width: 100%;
    padding: 20px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .modal h3 { margin: 0; font-size: 16px; }
  .field { display: flex; flex-direction: column; gap: 4px; }
  .field label { font-size: 12px; color: var(--muted); font-weight: 600; }
  .field input, .field textarea, .field select {
    background: var(--bg);
    border: 1px solid var(--border);
    color: var(--text);
    padding: 8px 10px;
    border-radius: 8px;
    font-size: 13px;
    font-family: inherit;
  }
  .toast {
    position: fixed;
    bottom: 18px;
    right: 22px;
    background: #0f172a;
    border: 1px solid var(--accent);
    color: var(--text);
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 12.5px;
    display: none;
    z-index: 80;
    box-shadow: 0 10px 30px rgba(0,0,0,0.45);
  }
</style>
</head>
<body>
<header>
  <div class="brand">
    <span class="brand-badge">MoS + SCache v3.0</span>
    <div>
      <h1>ScarStore — Biblioteca Visual de Cartuchos Cognitivos</h1>
      <p id="modelSubtitle">Carregando estado do motor...</p>
    </div>
  </div>
  <div class="metrics-bar">
    <span class="pill">VRAM MoS: <strong id="metricVram">-- GiB</strong></span>
    <span class="pill">Cartucho: <strong id="metricRecordKb">-- KB</strong></span>
    <span class="pill">Instalados / Nuvem: <strong id="metricSlots">--</strong></span>
    <span class="pill">SCache O(1): <strong id="metricScache">0.00 MiB</strong></span>
    <button class="btn btn-primary" onclick="openMintModal()">+ Criar Cartucho (Open-Weights)</button>
  </div>
</header>

<main>
  <section class="pane">
    <div class="pane-header">
      <h2>Biblioteca de Cartuchos (Local mmap 4KB + Nuvem HTTP 206)</h2>
      <button class="btn" onclick="refreshStatus()">Atualizar</button>
    </div>
    <div class="toolbar">
      <input id="searchBox" class="search-input" placeholder="Buscar por título, área, tema ou ID..." oninput="renderCards()" />
      <select id="categoryFilter" class="btn" onchange="renderCards()">
        <option value="">Todas as áreas</option>
      </select>
    </div>
    <div id="cardsGrid" class="cards-scroll"></div>
  </section>

  <section class="pane">
    <div class="pane-header">
      <h2>Chat ao Vivo com Roteamento Esférico + SCache O(1)</h2>
      <div style="display:flex; gap:8px; align-items:center;">
        <label style="font-size:12px; color:var(--muted); display:flex; align-items:center; gap:5px; cursor:pointer;">
          <input type="checkbox" id="persistFetchedChk" /> Salvar no SSD ao baixar
        </label>
        <button class="btn" onclick="resetChat()">Limpar Sessão</button>
      </div>
    </div>
    <div id="chatHistory" class="chat-history">
      <div class="msg msg-bot">
        Bem-vindo à <strong>ScarStore v3.0</strong>! Clique em <em>"Testar Pergunta"</em> em qualquer cartucho ao lado (mesmo os que estão apenas na nuvem!) ou faça qualquer pergunta abaixo. Você verá em tempo real qual cartucho acendeu, os bytes exatos via <code>HTTP 206</code> e a memória constante <code>SCache O(1)</code>.
      </div>
    </div>
    <div class="chat-composer">
      <div class="chat-row">
        <textarea id="chatInput" placeholder="Pergunte sobre qualquer cartucho especialista ou um assunto cotidiano (ex: 15 * 14)..." onkeydown="if(event.key==='Enter' && !event.shiftKey){event.preventDefault(); sendChat();}"></textarea>
        <button id="sendBtn" class="btn btn-primary" onclick="sendChat()">Enviar</button>
      </div>
    </div>
  </section>
</main>

<div id="mintModal" class="modal-backdrop">
  <div class="modal">
    <h3>Fábrica de Cartuchos Open-Weights (Zero SGD / Zero Backprop)</h3>
    <p style="margin:0; font-size:12.5px; color:var(--muted);">
      Sintetize um novo cartucho especialista em segundos via inversão fechada de Woodbury + espaço nulo AlphaEdit. Use uma resposta especialista curada ou extraia diretamente de um modelo Open-Weights local.
    </p>
    <div class="field">
      <label>Título Humano do Cartucho (PT-BR / EN)</label>
      <input id="mintTitle" placeholder="Ex: Direito Digital — Marco Civil e LGPD Art. 18" />
    </div>
    <div class="field">
      <label>Categoria</label>
      <input id="mintCategory" value="Open-Weights Specialist" />
    </div>
    <div class="field">
      <label>Fonte Open-Weights (Teacher)</label>
      <select id="mintTeacher" onchange="toggleAnswerBox()">
        <option value="">Resposta Especialista Direta (Instantâneo &lt; 1s — Qualquer Cluster Open-Weights)</option>
      </select>
    </div>
    <div class="field">
      <label>Pergunta-Âncora Principal (define o cone esférico do assunto)</label>
      <input id="mintPrompt1" placeholder="Ex: Qual direito o Art. 18 inciso VI da LGPD garante ao titular de dados pessoais?" />
    </div>
    <div class="field">
      <label>Paráfrase Opcional (melhora a generalização do cone)</label>
      <input id="mintPrompt2" placeholder="Ex: Na LGPD brasileira, o que estabelece o artigo 18 sobre eliminação de dados?" />
    </div>
    <div class="field" id="mintAnswerField">
      <label>Resposta Especialista de Referência (gerada por Open-Weights 70B/405B/R1 ou especialista)</label>
      <textarea id="mintAnswer" rows="3" placeholder="Ex: O Art. 18, VI da LGPD garante ao titular a eliminação dos dados pessoais tratados com seu consentimento, mediante requisição expressa."></textarea>
    </div>
    <div style="display:flex; justify-content:flex-end; gap:8px; margin-top:4px;">
      <button class="btn" onclick="closeMintModal()">Cancelar</button>
      <button id="mintSubmitBtn" class="btn btn-primary" onclick="submitMint()">Cunhar Cartucho (Zero SGD)</button>
    </div>
  </div>
</div>

<div id="toast" class="toast"></div>

<script>
let STATE = { cartridges: [], activeIds: [] };

function showToast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.style.display = 'block';
  clearTimeout(window._toastTimer);
  window._toastTimer = setTimeout(() => { el.style.display = 'none'; }, 4200);
}

function toggleAnswerBox() {
  const t = document.getElementById('mintTeacher').value;
  document.getElementById('mintAnswerField').style.display = t ? 'none' : 'flex';
}

function openMintModal() { document.getElementById('mintModal').classList.add('open'); }
function closeMintModal() { document.getElementById('mintModal').classList.remove('open'); }

async function refreshStatus() {
  const resp = await fetch('/api/status');
  const data = await resp.json();
  STATE = data;
  const mosLabel = data.mos_enabled ? `MoS INT4+ASVD (r=${data.mos_rank})` : 'FP16 Denso';
  document.getElementById('modelSubtitle').textContent =
    `${data.student_model} • ${mosLabel} (${data.device}) • Grade Kronecker ${data.kronecker_grid_cells.toLocaleString()} células • Fonte: ${data.remote_mode}`;
  document.getElementById('metricVram').textContent =
    data.gpu_allocated_gib > 0 ? `${data.gpu_allocated_gib.toFixed(2)} GiB` : `RSS ${data.process_peak_rss_gib.toFixed(2)} GiB`;
  document.getElementById('metricRecordKb').textContent = `${data.record_kb.toFixed(2)} KB`;
  document.getElementById('metricSlots').textContent = `${data.installed_count} locais / ${data.total_count} catálogo`;
  document.getElementById('metricScache').textContent = `${(data.scache_kb / 1024.0).toFixed(2)} MiB fixos`;

  const catSel = document.getElementById('categoryFilter');
  const prevCat = catSel.value;
  const cats = Array.from(new Set(data.cartridges.map(c => c.category || 'Geral'))).sort();
  catSel.innerHTML = '<option value="">Todas as áreas</option>' +
    cats.map(c => `<option value="${escapeHtml(c)}">${escapeHtml(c)}</option>`).join('');
  catSel.value = prevCat;

  const tSel = document.getElementById('mintTeacher');
  const curT = tSel.value;
  let opts = '<option value="">Resposta Especialista Direta (Instantâneo &lt; 1s — Qualquer Cluster Open-Weights)</option>';
  for (const m of (data.local_open_weight_models || [])) {
    const modeLabel = m.mode_for_qwen_student === 'same_vocab_exact' ? 'Mesmo Tokenizador (Logits Exatos)' : 'Ponte Cross-Tokenizer';
    opts += `<option value="${escapeHtml(m.repo_id)}">${escapeHtml(m.repo_id)} — ${modeLabel}</option>`;
  }
  tSel.innerHTML = opts;
  if (curT) tSel.value = curT;
  toggleAnswerBox();
  renderCards();
}

function escapeHtml(s) {
  return String(s || '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

function renderCards() {
  const q = (document.getElementById('searchBox').value || '').toLowerCase().trim();
  const cat = document.getElementById('categoryFilter').value;
  const grid = document.getElementById('cardsGrid');
  const activeSet = new Set(STATE.activeIds || []);

  const filtered = (STATE.cartridges || []).filter(c => {
    if (cat && (c.category || 'Geral') !== cat) return false;
    if (!q) return true;
    const hay = `${c.title_pt} ${c.title} ${c.category} ${c.summary} ${c.capsule_id}`.toLowerCase();
    return hay.includes(q);
  });

  if (!filtered.length) {
    grid.innerHTML = '<p style="color:var(--muted); font-size:13px;">Nenhum cartucho encontrado para esse filtro.</p>';
    return;
  }

  grid.innerHTML = filtered.map(c => {
    let stClass = 'st-cloud', stText = '☁️ Na Nuvem (HTTP 206)';
    if (c.status === 'installed') { stClass = 'st-local'; stText = '💾 Instalado (mmap 4KB)'; }
    else if (c.status === 'hot_ram') { stClass = 'st-ram'; stText = '⚡ Streamed em RAM'; }
    else if (c.status === 'wiped') { stClass = 'st-wiped'; stText = '🧹 Apagado LGPD (0x00)'; }

    const lit = activeSet.has(c.capsule_id) ? 'active-lit' : '';
    const promptEsc = escapeHtml(c.example_prompt || '');
    const canInstall = (c.status === 'cloud' || c.status === 'hot_ram') && STATE.has_remote;
    const canUnlearn = (c.status === 'installed' || c.status === 'hot_ram');

    return `
      <div class="card ${lit}">
        <div>
          <div class="card-top">
            <span class="cat-tag">${escapeHtml(c.category || 'Geral')}</span>
            <span class="status-tag ${stClass}">${stText}</span>
          </div>
          <h3>${escapeHtml(c.title_pt || c.title)}</h3>
          <p class="sub">${escapeHtml(c.summary || c.title)}</p>
        </div>
        <div>
          <div class="meta-line" style="margin-bottom:8px;">
            <span>${c.record_kb.toFixed(2)} KB</span>
            <span>•</span>
            <span>Coord (${c.coord[0]}, ${c.coord[1]})</span>
            <span>•</span>
            <span>${escapeHtml(c.capsule_id)}</span>
          </div>
          <div class="card-actions">
            ${c.example_prompt ? `<button class="btn" onclick="tryPrompt(${JSON.stringify(c.example_prompt).replace(/"/g, '&quot;')})">💬 Testar Pergunta</button>` : ''}
            ${canInstall ? `<button class="btn btn-green" onclick="installCapsule('${escapeHtml(c.capsule_id)}')">⬇️ Instalar (${c.record_kb.toFixed(0)} KB)</button>` : ''}
            ${canUnlearn ? `<button class="btn btn-danger" onclick="unlearnCapsule('${escapeHtml(c.capsule_id)}')">🗑️ Apagar LGPD (0x00)</button>` : ''}
          </div>
        </div>
      </div>
    `;
  }).join('');
}

function tryPrompt(p) {
  document.getElementById('chatInput').value = p;
  sendChat();
}

async function sendChat() {
  const box = document.getElementById('chatInput');
  const prompt = box.value.trim();
  if (!prompt) return;
  box.value = '';
  const btn = document.getElementById('sendBtn');
  btn.disabled = true;

  const hist = document.getElementById('chatHistory');
  hist.innerHTML += `<div class="msg msg-user">${escapeHtml(prompt)}</div>`;
  const pendingId = 'pending_' + Date.now();
  hist.innerHTML += `<div id="${pendingId}" class="msg msg-bot" style="color:var(--muted);">Roteando na esfera Kronecker e gerando com SCache O(1)...</div>`;
  hist.scrollTop = hist.scrollHeight;

  try {
    const persist = document.getElementById('persistFetchedChk').checked;
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ prompt, persist_fetched: persist })
    });
    const res = await resp.json();
    if (!resp.ok) throw new Error(res.error || 'Falha na geração');

    const activeEntries = Object.entries(res.active_gates || {});
    STATE.activeIds = activeEntries.map(([k]) => k);

    let chips = '';
    if (activeEntries.length === 0) {
      chips += `<span class="telem-chip">0 cartuchos ativos (Invariância 0.0000000000)</span>`;
    } else {
      for (const [cid, gate] of activeEntries) {
        const title = (res.active_titles && res.active_titles[cid]) ? res.active_titles[cid] : cid;
        const cos = (res.routing_cosines && res.routing_cosines[cid] !== undefined) ? res.routing_cosines[cid].toFixed(3) : '-';
        chips += `<span class="telem-chip hit">🧩 ${escapeHtml(title)} (gate=${gate.toFixed(2)}, cos=${cos})</span>`;
      }
    }
    for (const ev of (res.http_range_events || [])) {
      chips += `<span class="telem-chip http">⚡ HTTP 206: ${(ev.bytes_received/1024).toFixed(2)} KB em ${ev.latency_ms.toFixed(1)} ms</span>`;
    }
    const scMib = (res.scache_memory_kb / 1024.0).toFixed(2);
    const decTps = (res.decode_tok_per_s || res.tok_per_s || 0).toFixed(1);
    chips += `<span class="telem-chip">+Heads: ${res.num_expanded_heads}</span>`;
    chips += `<span class="telem-chip">SCache O(1): ${scMib} MiB fixos</span>`;
    if (res.gpu_allocated_mib > 0) {
      chips += `<span class="telem-chip">VRAM MoS: ${(res.gpu_allocated_mib/1024.0).toFixed(2)} GiB</span>`;
    }
    chips += `<span class="telem-chip">${decTps} tok/s decode</span>`;

    document.getElementById(pendingId).outerHTML = `
      <div class="msg msg-bot">
        <div>${escapeHtml(res.text)}</div>
        <div class="telem-box">${chips}</div>
      </div>
    `;
    await refreshStatus();
  } catch (err) {
    document.getElementById(pendingId).outerHTML =
      `<div class="msg msg-bot" style="color:var(--red);">Erro: ${escapeHtml(err.message)}</div>`;
  } finally {
    btn.disabled = false;
    hist.scrollTop = hist.scrollHeight;
  }
}

async function resetChat() {
  await fetch('/api/reset_chat', { method: 'POST' });
  STATE.activeIds = [];
  document.getElementById('chatHistory').innerHTML =
    `<div class="msg msg-bot">Sessão <code>SCache O(1)</code> reiniciada. Pronta para nova conversa!</div>`;
  await refreshStatus();
}

async function installCapsule(capsuleId) {
  showToast(`Baixando fatia HTTP 206 de ${capsuleId}...`);
  const resp = await fetch('/api/pull_capsule', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ capsule_id: capsuleId })
  });
  const data = await resp.json();
  if (!resp.ok) { showToast('Erro: ' + (data.error || 'falha')); return; }
  showToast(`Instalado no SSD: ${capsuleId} (${(data.bytes_installed/1024).toFixed(2)} KB alinhado em 4KB)!`);
  await refreshStatus();
}

async function unlearnCapsule(capsuleId) {
  const resp = await fetch('/api/unlearn', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ capsule_id: capsuleId, persist: true })
  });
  const data = await resp.json();
  if (!resp.ok) { showToast('Erro: ' + (data.error || 'falha')); return; }
  STATE.activeIds = (STATE.activeIds || []).filter(x => x !== capsuleId);
  showToast(`LGPD 0x00 concluído em ${data.unlearn_ms.toFixed(3)} ms (${data.cryptographically_zeroed_bytes.toLocaleString()} B zerados no SSD)!`);
  await refreshStatus();
}

async function submitMint() {
  const title = document.getElementById('mintTitle').value.trim();
  const category = document.getElementById('mintCategory').value.trim();
  const teacher = document.getElementById('mintTeacher').value;
  const p1 = document.getElementById('mintPrompt1').value.trim();
  const p2 = document.getElementById('mintPrompt2').value.trim();
  const ans = document.getElementById('mintAnswer').value.trim();
  if (!title || !p1 || (!teacher && !ans)) {
    showToast('Preencha o título, a pergunta-âncora e a resposta (ou escolha um modelo Teacher).');
    return;
  }
  const btn = document.getElementById('mintSubmitBtn');
  btn.disabled = true;
  btn.textContent = 'Cunhando via Woodbury + AlphaEdit...';
  try {
    const resp = await fetch('/api/mint', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        title, category, teacher_model: teacher,
        prompt: p1, prompt_2: p2, answer: ans
      })
    });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || 'Falha ao cunhar cartucho');
    closeMintModal();
    showToast(`Cartucho "${title}" cunhado (${(data.record_bytes/1024).toFixed(2)} KB, Zero SGD)!`);
    await refreshStatus();
  } catch (err) {
    showToast('Erro: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Cunhar Cartucho (Zero SGD)';
  }
}

refreshStatus();
</script>
</body>
</html>
"""


class ScarStoreWebUI:
    """Thread-safe local Web Server exposing the Visual Cartridge Store + Live `SCache O(1)` Chat."""

    def __init__(
        self,
        student_model: nn.Module,
        tokenizer: Any,
        store: HeadExpandedScarStore,
        student_name: str = "Qwen/Qwen2.5-3B-Instruct",
        http_client: Optional[HTTPRangeCapsuleClient] = None,
        catalog_entries: Optional[List[Dict[str, Any]]] = None,
        remote_mode: str = "local",
        host: str = "127.0.0.1",
        port: int = 7860,
        window_size: int = 256,
        num_slots: int = 64,
        max_new_tokens: int = 96,
        local_files_only: bool = False,
    ) -> None:
        self.student_model = student_model
        self.tokenizer = tokenizer
        self.store = store
        self.student_name = student_name
        self.http_client = http_client
        self.remote_mode = remote_mode
        self.window_size = int(window_size)
        self.num_slots = int(num_slots)
        self.max_new_tokens = int(max_new_tokens)
        self.local_files_only = bool(local_files_only)
        self.scache = SCacheDynamicCache(window_size=self.window_size, num_slots=self.num_slots)
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

        if self.store._norm_hook_handle is None:
            self.store.attach(self.student_model)

        self._catalog_by_id: Dict[str, Dict[str, Any]] = {}
        for item in catalog_entries or []:
            if isinstance(item, dict) and item.get("capsule_id"):
                self._catalog_by_id[str(item["capsule_id"])] = dict(item)

        self._local_open_weights = discover_local_open_weight_models()

        parent = self

        class _UIHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
                body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _read_json(self) -> Dict[str, Any]:
                n = int(self.headers.get("Content-Length", "0") or "0")
                if n <= 0:
                    return {}
                return json.loads(self.rfile.read(n).decode("utf-8"))

            def do_GET(self) -> None:  # noqa: N802
                if self.path in ("/", "/index.html"):
                    html = _UI_HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)
                    return
                if self.path.startswith("/api/status"):
                    with parent._lock:
                        self._send_json(HTTPStatus.OK, parent.build_status_payload())
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint {self.path}"})

            def do_POST(self) -> None:  # noqa: N802
                try:
                    body = self._read_json()
                    with parent._lock:
                        if self.path == "/api/chat":
                            self._send_json(HTTPStatus.OK, parent.handle_chat(body))
                        elif self.path == "/api/reset_chat":
                            self._send_json(HTTPStatus.OK, parent.handle_reset_chat())
                        elif self.path == "/api/pull_capsule":
                            self._send_json(HTTPStatus.OK, parent.handle_pull_capsule(body))
                        elif self.path == "/api/unlearn":
                            self._send_json(HTTPStatus.OK, parent.handle_unlearn(body))
                        elif self.path == "/api/mint":
                            self._send_json(HTTPStatus.OK, parent.handle_mint(body))
                        else:
                            self._send_json(HTTPStatus.NOT_FOUND, {"error": f"Unknown endpoint {self.path}"})
                except Exception as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        self._httpd = ThreadingHTTPServer((host, int(port)), _UIHandler)
        self.host, self.port = self._httpd.server_address[0], int(self._httpd.server_address[1])

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _resolve_meta(self, capsule_id: str, inline_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if capsule_id in self._catalog_by_id:
            merged.update(self._catalog_by_id[capsule_id])
        if inline_meta:
            merged.update(inline_meta)
        title = str(merged.get("title") or capsule_id)
        return {
            "title": title,
            "title_pt": str(merged.get("title_pt") or title),
            "category": str(merged.get("category") or "General"),
            "summary": str(merged.get("summary") or title),
            "example_prompt": str(merged.get("example_prompt") or ""),
            "teacher_model": str(merged.get("teacher_model") or "Qwen2.5-72B-Instruct MoS v2"),
        }

    def build_status_payload(self) -> Dict[str, Any]:
        summary = self.store.inspect_summary()
        rec_kb = float(summary["record_kb"])
        by_id: Dict[str, Dict[str, Any]] = {}

        for s in summary.get("slots", []):
            cid = str(s["capsule_id"])
            meta = self._resolve_meta(cid, s.get("metadata"))
            if s.get("payload_all_zero_wiped") or not s.get("bitmap_present"):
                status = "wiped"
            else:
                status = "installed"
            by_id[cid] = {
                "capsule_id": cid,
                "coord": list(s["coord"]),
                "record_kb": rec_kb,
                "status": status,
                **meta,
            }

        for c in summary.get("cloud_directory", []):
            cid = str(c["capsule_id"])
            if cid in by_id:
                continue
            meta = self._resolve_meta(cid, c.get("metadata"))
            if cid in self.store._unlearned_cids:
                status = "wiped"
            elif cid in self.store._hot_fetched_capsules:
                status = "hot_ram"
            elif c.get("installed_locally"):
                status = "installed"
            else:
                status = "cloud"
            by_id[cid] = {
                "capsule_id": cid,
                "coord": list(c["coord"]),
                "record_kb": rec_kb,
                "status": status,
                **meta,
            }

        cartridges = list(by_id.values())
        installed_count = sum(1 for c in cartridges if c["status"] == "installed")
        dev_obj = self.store.mu_query_last.device
        dev_str = str(dev_obj)
        mem_telem = get_accelerator_memory_telemetry(dev_obj)

        return {
            "student_model": self.student_name,
            "device": dev_str,
            "mos_enabled": bool(getattr(self.student_model, "mos_enabled", False)),
            "mos_rank": int(getattr(self.student_model, "mos_rank", 0)),
            "gpu_allocated_mib": float(mem_telem["gpu_allocated_mib"]),
            "gpu_allocated_gib": float(mem_telem["gpu_allocated_gib"]),
            "process_peak_rss_gib": float(mem_telem["process_peak_rss_gib"]),
            "store_path": str(self.store.store_path),
            "remote_mode": self.remote_mode,
            "has_remote": self.http_client is not None,
            "record_bytes": int(summary["record_bytes"]),
            "record_kb": rec_kb,
            "kronecker_grid_cells": int(summary["kronecker_grid_cells"]),
            "installed_count": installed_count,
            "total_count": len(cartridges),
            "scache_bytes": int(self.scache.total_memory_bytes),
            "scache_kb": round(self.scache.total_memory_bytes / 1024.0, 2),
            "scache_mib": round(self.scache.total_memory_bytes / (1024.0**2), 2),
            "cartridges": cartridges,
            "local_open_weight_models": self._local_open_weights,
        }

    def handle_chat(self, body: Dict[str, Any]) -> Dict[str, Any]:
        prompt = str(body.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        persist_fetched = bool(body.get("persist_fetched", False))
        max_tok = int(body.get("max_new_tokens") or self.max_new_tokens)

        prior_http = len(self.http_client.fetch_events) if self.http_client is not None else 0
        res, self.scache = generate_with_scar_store(
            student_model=self.student_model,
            tokenizer=self.tokenizer,
            store=self.store,
            prompt_text_or_ids=prompt,
            max_new_tokens=max_tok,
            scache=self.scache,
            system_prompt=select_chat_system_prompt(prompt),
            http_client=self.http_client,
            persist_fetched=False,
        )

        # If user asked to persist newly streamed HTTP 206 capsules, append & rebind self.store cleanly
        new_events = list(self.http_client.fetch_events[prior_http:]) if self.http_client is not None else []
        if persist_fetched and new_events:
            for ev in new_events:
                cid_f = str(ev["capsule_id"])
                if cid_f in self.store._hot_fetched_capsules:
                    cap_hot = self.store._hot_fetched_capsules[cid_f]
                    self.store.detach()
                    self.store = self.store.append_capsule(cap_hot)
                    self.store.attach(self.student_model)

        payload = asdict(res)
        active_titles: Dict[str, str] = {}
        for cid in res.active_gates:
            g_entry = next(
                (v for v in self.store._global_directory.values() if str(v.get("capsule_id")) == cid),
                None,
            )
            meta = self._resolve_meta(cid, g_entry.get("metadata") if g_entry else None)
            active_titles[cid] = meta["title_pt"]
        payload["active_titles"] = active_titles
        return payload

    def handle_reset_chat(self) -> Dict[str, Any]:
        self.scache = SCacheDynamicCache(window_size=self.window_size, num_slots=self.num_slots)
        return {"ok": True, "scache_kb": 0.0}

    def handle_pull_capsule(self, body: Dict[str, Any]) -> Dict[str, Any]:
        cid = str(body.get("capsule_id") or "").strip()
        if not cid:
            raise ValueError("capsule_id is required.")
        if self.http_client is None:
            raise ValueError("No remote HTTP-206 ScarStore endpoint configured.")

        g_entry = next(
            (v for v in self.store._global_directory.values() if str(v.get("capsule_id")) == cid),
            None,
        )
        if g_entry is None:
            raise KeyError(f"Unknown cloud capsule_id {cid!r}")
        coord = (int(g_entry["row_i"]), int(g_entry["col_j"]))
        raw_bytes = self.http_client.fetch_capsule_bytes(coord, cid)
        if raw_bytes is None:
            raise RuntimeError(f"Remote server did not contain {cid!r} at {coord}")

        cap = HeadExpandedScarCapsule.from_int4_bytes(
            raw_bytes=raw_bytes,
            capsule_id=cid,
            hidden_dim=self.store.hidden_dim,
            num_heads=self.store.num_heads,
            head_dim=self.store.head_dim,
            rank=self.store.rank,
            num_key_prototypes=self.store.num_key_prototypes,
            tau=float(g_entry["tau"]),
            ramp=float(g_entry["ramp"]),
            parent_capsules=g_entry.get("parent_capsules", ()),
            metadata=dict(g_entry.get("metadata", {})),
            device=self.store.mu_query_last.device,
        )
        self.store.detach()
        self.store = self.store.append_capsule(cap)
        self.store.attach(self.student_model)
        return {
            "ok": True,
            "capsule_id": cid,
            "coord": list(coord),
            "bytes_installed": len(raw_bytes),
        }

    def handle_unlearn(self, body: Dict[str, Any]) -> Dict[str, Any]:
        cid = str(body.get("capsule_id") or "").strip()
        if not cid:
            raise ValueError("capsule_id is required.")
        persist = bool(body.get("persist", True))
        return self.store.unlearn(cid, persist_to_disk=persist)

    def handle_mint(self, body: Dict[str, Any]) -> Dict[str, Any]:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        title = str(body.get("title") or "").strip()
        category = str(body.get("category") or "Open-Weights Custom").strip()
        teacher_id = str(body.get("teacher_model") or "").strip()
        p1 = str(body.get("prompt") or "").strip()
        p2 = str(body.get("prompt_2") or "").strip()
        ans = str(body.get("answer") or "").strip()
        if not title or not p1:
            raise ValueError("Both 'title' and 'prompt' are required.")

        prompts = [p1] + ([p2] if p2 else [])
        spec: Dict[str, Any] = {
            "title": title,
            "title_pt": title,
            "category": category,
            "summary": ans[:140] if ans else f"Especialista Open-Weights ({teacher_id})",
            "example_prompt": p1,
        }

        teacher_model = None
        teacher_tok = None
        owns_teacher = False
        if teacher_id:
            dev = self.store.mu_query_last.device
            dtype = torch.float16 if dev.type in ("cuda", "mps") else torch.float32
            if teacher_id == self.student_name:
                teacher_model = self.student_model
                teacher_tok = self.tokenizer
            else:
                teacher_tok = AutoTokenizer.from_pretrained(teacher_id, local_files_only=self.local_files_only)
                teacher_model = AutoModelForCausalLM.from_pretrained(
                    teacher_id,
                    dtype=dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=self.local_files_only,
                ).to(dev).eval()
                owns_teacher = True
            spec["train_prompts"] = prompts
        else:
            if not ans:
                raise ValueError("Provide either 'answer' or 'teacher_model'.")
            spec["episodes"] = [{"prompt": pr, "answer": ans} for pr in prompts]

        sys_prompt = select_chat_system_prompt(p1)
        fmt = lambda q: format_chatml_prompt(q, system_prompt=sys_prompt)
        try:
            with self.store.paused():
                head_bank = HeadExpandedAttentionEngine.calibrate_head_expanded_student(
                    student_model=self.student_model,
                    calibration_prompts=DEFAULT_CALIBRATION_PROMPTS,
                    tokenizer=self.tokenizer,
                    chat_formatter=fmt,
                    preserved_rank=4,
                )
                caps = mint_capsules_from_specs(
                    specs=[spec],
                    student_model=self.student_model,
                    student_tokenizer=self.tokenizer,
                    head_bank=head_bank,
                    chat_formatter=fmt,
                    teacher_model=teacher_model,
                    teacher_tokenizer=teacher_tok,
                    teacher_label=teacher_id or "Open-Weights Curated Trajectory",
                    rank=self.store.rank,
                    num_key_prototypes=self.store.num_key_prototypes,
                )
        finally:
            if owns_teacher and teacher_model is not None:
                del teacher_model
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()

        cap_new = caps[0]
        self._catalog_by_id[cap_new.capsule_id] = dict(cap_new.metadata, capsule_id=cap_new.capsule_id)
        self.store.detach()
        self.store = self.store.append_capsule(cap_new)
        self.store.attach(self.student_model)
        return {
            "ok": True,
            "capsule_id": cap_new.capsule_id,
            "record_bytes": self.store.record_bytes,
            "metadata": dict(cap_new.metadata),
        }

    def start_background(self) -> "ScarStoreWebUI":
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

    def __enter__(self) -> "ScarStoreWebUI":
        return self.start_background()

    def __exit__(self, *_exc: Any) -> None:
        self.stop()


def launch_scar_store_ui_from_args(args: Any) -> int:
    """CLI entrypoint for `mos-store ui`."""
    from scar_cache.cli_scar_store import _resolve_device_and_dtype
    from transformers import AutoTokenizer

    dev, dtype = _resolve_device_and_dtype(args.device)
    tok = AutoTokenizer.from_pretrained(args.student, local_files_only=args.local_files_only)
    slm = load_mos_student_model(
        args.student,
        tokenizer=tok,
        device=dev,
        dtype=dtype,
        local_files_only=args.local_files_only,
        dense_fallback=bool(getattr(args, "dense_student", False)),
    )
    student_d = int(slm.config.hidden_size)

    remote_url = args.remote_url
    store_path: Optional[Path] = Path(args.store) if args.store else None
    if store_path is None or not store_path.exists():
        if not remote_url:
            remote_url = "auto"
        spec = resolve_official_cloud_tier(student_or_tier=args.student, hidden_dim=student_d)
        default_shell = (
            store_path
            if store_path is not None
            else (
                Path.home()
                / ".cache"
                / "mixture_of_scars"
                / "scarstore"
                / f"qwen{spec['tier'].replace('.', 'p')}_v3_0.shell.scarstore"
            )
        )
        if not default_shell.exists():
            print(f"[mos-store ui] Downloading Cloud Shell -> {default_shell} ...", flush=True)
            download_cloud_scarstore(tier_or_student=spec["tier"], out_path=default_shell, mode="shell")
        store_path = default_shell

    http_client: Optional[HTTPRangeCapsuleClient] = None
    fetcher: Optional[Callable[[Tuple[int, int], str], Optional[bytes]]] = None
    catalog_entries: List[Dict[str, Any]] = []
    if remote_url:
        fetcher, http_client = make_http_range_capsule_fetcher(
            remote_url,
            hidden_dim=student_d,
            student_hint=args.student,
        )
        if remote_url.lower() in ("auto", "official", "cloud", "3b", "0.5b", "0p5b"):
            try:
                cat = fetch_official_catalog(OFFICIAL_CATALOG_URL, timeout_s=8.0)
                catalog_entries = list(cat.get("cartridges", []))
            except Exception:
                catalog_entries = []

    store = HeadExpandedScarStore.open(store_path, device=dev, on_demand_fetcher=fetcher)
    ui = ScarStoreWebUI(
        student_model=slm,
        tokenizer=tok,
        store=store,
        student_name=args.student,
        http_client=http_client,
        catalog_entries=catalog_entries,
        remote_mode=remote_url or "local",
        host=args.host,
        port=args.port,
        local_files_only=args.local_files_only,
    )
    print(f"[mos-store ui] Visual Cartridge Store & SCache O(1) Chat live at {ui.base_url}", flush=True)
    if not args.no_browser:
        try:
            webbrowser.open(ui.base_url)
        except Exception:
            pass
    try:
        ui.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ui.stop()
        ui.store.close()
    return 0
