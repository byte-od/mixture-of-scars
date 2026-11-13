# 🏛️ `ScarStore v2` (`v2.0.0`): Arquitetura Pós-Monolítica de Conhecimento em Cones Esparsos

> **Mandato de Integridade Arquitetural (Zero Fachadas):**  
> Todo componente documentado neste guia executa matemática algébrica real em PyTorch puro (`ZERO backpropagation / ZERO SGD`), quantização simétrica `Block-64 INT4` nativa, roteamento esférico com **`0.0000000000` (`torch.equal`)** de interferência fora do cone e paginação `mmap` real sobre o SSD.

---

## ✅ Etapa 1 (`v2.0.0a2`): Cápsulas `61,25 KB INT4` + Grade Esparsa Kronecker `896 × 896` + `mmap` & Trava Anti-Alucinação

**Arquivos Canônicos:** [`scar_cache/scar_store_v2.py`](../scar_cache/scar_store_v2.py) · [`tests/test_scar_store_v2_stage1.py`](../tests/test_scar_store_v2_stage1.py) · [`results/scar_store_v2_stage1_audit.json`](../results/scar_store_v2_stage1_audit.json)

### 1.1. Geometria Física dos Bytes (`d = 896, r = 64`)
1. **Compressão `Block-64 INT4` (`QuantizedScarCapsule`):**
   * `B_scar` (`896 × 64`) $\to$ `28.672 B` (`uint8` empacotado) + `1.792 B` escalas `FP16` = `30.464 B`.
   * `A_scar` (`64 × 896`) $\to$ `28.672 B` (`uint8` empacotado) + `1.792 B` escalas `FP16` = `30.464 B`.
   * `z_anchor` (`896` `FP16`) = `1.792 B`.
   * **Total exato por cápsula:** **`62.720 bytes` (`61,25 KB`)** — redução de `3,69×` sobre os `225,75 KB` `FP16` e **`2.318.508×` menor transferência ativa** que os `135,43 GiB` do `Qwen2.5-72B` `BF16`!
2. **Roteador Kronecker em Duplo Subespaço Rotacionado (`896 × 896 = 802.816` coordenadas):**
   * Rotação ortogonal determinística $\mathbf{R} \in O(896)$ (`1,53 MiB`) + codebooks unitários $\mathbf{U}, \mathbf{V} \in \mathbb{R}^{896 \times 448}$ (`0,765 MiB` cada) + **bitmap de presença de exatos `100.352 bytes` (`98,00 KB`)** cobrindo todas as `802.816` células `(row_i, col_j)`.
   * Triagem em 2 estágios (*Coarse-to-Fine*): filtra as Top-$m$ coordenadas da grade em `~0,05 ms`, verifica a âncora exata $\mathbf{z}_k$ das candidatas instaladas e faz *page fault* `mmap` alinhado em `4096 B` **apenas dos `61,25 KB` que abriram o cone `ReLU`** (`0 bytes` lidos do SSD para perguntas fora do cone!).
3. **Fragmentação Seletiva para Consumidor (`export_subset`) & Guarda Epistêmica (`MissingCapsuleAlert`):**
   * Permite exportar pacotes enxutos contendo apenas domínios selecionados (`0 bytes` de payload no disco para domínios omitidos), mantendo as coordenadas universais `(row_i, col_j)`.
   * Se uma consulta acionar uma coordenada ausente (`bitmap[i, j] == 0`), emite `MissingCapsuleAlert` explícito (ou aciona `on_demand_fetcher` para streaming `HTTP Range` de `61,25 KB`), impedindo alucinação silenciosa.

---

## ✅ Etapa 2 (`v2.0.0a3`): Roteamento Multi-Cartucho *Span-Max* + Orquestração `SCache O(1)` Sem Contaminação

**Arquivos Canônicos:** [`scar_cache/scar_grafting.py`](../scar_cache/scar_grafting.py) (`extract_span_query_probes`) · [`scar_cache/scar_store_v2.py`](../scar_cache/scar_store_v2.py) (`generate_multi_capsule`) · [`tests/test_scar_store_v2_stage2.py`](../tests/test_scar_store_v2_stage2.py) · [`results/scar_store_v2_stage2_audit.json`](../results/scar_store_v2_stage2_audit.json)

### 2.1. Eliminação da Diluição $1/\sqrt{K}$ em Perguntas Multi-Domínio
* **Problema físico diagnosticado:** Em perguntas compostas por $K \ge 3$ tópicos distintos, uma única média global $\bar{\mathbf{q}}$ dilui o cosseno individual ($\text{Immerman-Szelepcsényi}$ caiu para `0,6151 < 0,62`, ficando desligado).
* **Solução *Span-Max Sliding Window*:** Extrai janelas deslizantes normalizadas e centradas (`width = 10`, `stride = 3`) ao longo do turno do usuário e executa projeção Kronecker em lote sobre todas as janelas simultaneamente:
  * Nos 3 teoremas combinados do `72.71B`, elevou os cossenos para **`0,8502`**, **`0,8849`** e **`0,8680`** (`3/3` cápsulas ativadas, lendo exatos `188.160 B = 183,75 KB` via `mmap`), detectando automaticamente a ordem exata no prompt tanto na direção direta (`K-L → Landauer → Immerman`) quanto na reversa (`Immerman → Landauer → K-L`).
  * Manteve todos os 6 prompts de controle irrelevantes em `max_control_cosine = 0,4922 < 0,62` com **`exact_torch_equal_bit_parity = true` (`0.0000000000`)**.

### 2.2. Isolamento de Atenção via Segmentação de Voronoi + `SCache O(1)`
* Quando $M \ge 2$ cápsulas acendem em janelas distintas, `generate_multi_capsule(...)` particiona os tokens do prompt pelas fronteiras de Voronoi entre os centros de pico $(s_m + e_m)/2$ e executa cada sub-cláusula com sua cápsula dedicada (`gate = 1.0`) em um `SCacheDynamicCache` limpo (`2,68–2,73 s` totais para os 3 teoremas completos no Mac M2), eliminando `100%` da contaminação cruzada de atenção entre domínios.

---

## ✅ Etapa 3 (`v2.0.0`): *Dual-Depth Golden-Slice Grafting* (`Layer L/2 + Layer L`) & Ponte Interdisciplinar no Ciclo do Sono

**Arquivos Canônicos:** [`scar_cache/dual_depth_bridge.py`](../scar_cache/dual_depth_bridge.py) · [`tests/test_scar_store_v2_stage3.py`](../tests/test_scar_store_v2_stage3.py) · [`scripts/audit_scar_store_v2_stage3.py`](../scripts/audit_scar_store_v2_stage3.py) · [`results/scar_store_v2_stage3_audit.json`](../results/scar_store_v2_stage3_audit.json)

### 3.1. Por Que Injetar em Duas Profundidades (`Layer 12 + Layer 24`) e Por Que Separar Justaposição vs. Fusão?
* **O gargalo físico não é RAM (`61,25 KB`–`120,75 KB`), é interferência no vetor oculto único $\mathbf{h}_t \in \mathbb{R}^{896}$:**
  1. Em **perguntas multi-parte independentes** (*Etapa 2*), segmentar por Voronoi evita que duas cápsulas disputem simultaneamente o `argmax` do mesmo token.
  2. Em **sínteses interdisciplinares reais** (*Etapa 3*), injetar apenas na saída (`Layer 24`) só soma logits linearmente ($\Delta \ell_A + \Delta \ell_B$), pois não restam camadas de auto-atenção acima. Ao injetar `(B_mid, A_mid)` no ponto **Golden-Slice (`Layer 12/24`)**, as **12 camadas superiores (`Layers 13..24`)** executam *cross-attention* + SwiGLU não-linear sobre os conceitos combinados antes do refino residual `(B_final, A_final)` na `Layer 24`!
  3. **Posto configurável sem necessidade de amputação:** Com `r_mid=32, r_final=32`, a cápsula ocupa os mesmíssimos **`62.720 B` (`61,25 KB` INT4)** da Etapa 1 (prova *maçã-com-maçã*); com posto cheio sem corte (`r_mid=64, r_final=64`), ocupa apenas **`123.648 B` (`120,75 KB` INT4)**.

### 3.2. Resultados Empíricos Auditados (`Qwen2.5-72B-Instruct MoS → Qwen2.5-0.5B-Instruct`)
* **Fidelidade Superior em Paráfrases Inéditas (`61,25 KB INT4`, `ZERO autograd`):**
  * *Immerman-Szelepcsényi:* **`100,0%` Top-1** com o `72.71B` (`KL: 0,4757 → 0,0364`, **`13,1×` menor**).
  * *Princípio de Landauer:* **`95,83%` Top-1** (`KL: 0,2934 → 0,0445`, **`6,6×` menor**).
  * *Kabatiansky-Levenshtein:* **`91,67%` Top-1** (`+8,34 p.p.` acima dos `83,33%` de profundidade única na Etapa 1; `KL: 0,6181 → 0,3412`).
* **Silêncio Bit-a-Bit Fora do Cone em Todas as Camadas:** Nos 6 prompts de controle, `max_control_cosine_layer12 = 0,4355 < tau = 0,70`, com **`exact_torch_equal_layer20 = true`** e **`exact_torch_equal_logits = true`**.
* **Reflexo Epistêmico (`BridgeEscalationDecision`) + Consolidação de Ponte Interdisciplinar (`61,25 KB INT4`):**
  * Antes do sono (`Landauer × Immerman`), `assess_interdisciplinary_epistemic_uncertainty` detecta interseção não consolidada (`is_uncertain = true`, `margin = 0,7053`).
  * Após `consolidate_interdisciplinary_bridge(...)`, a cápsula-ponte `bridge_landauer__immerman` (`parent_capsules` conjuntivos — **zero vazamento** nos domínios puros: `landauer_exact_text_match = true`, `immerman_exact_text_match = true`) eleva `confidence: 0,8398 → 0,9788`, `margin: 0,7053 → 0,9702` e derruba `uncertainty: 0,0345 → 0,0123` (`is_uncertain = false`).


