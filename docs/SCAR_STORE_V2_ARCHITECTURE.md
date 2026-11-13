# 🏛️ `ScarStore v2` (`v2.3.0`): Arquitetura Pós-Monolítica de Conhecimento em Cones Esparsos, Núcleo Cognitivo `3B`, CLI `mos-store` (`HTTP 206 Range`) & Atenção Expandida sob Demanda

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

---

## ✅ Etapa 4 (`v2.1.0`): Expansão Dinâmica de Cabeças de Atenção sob Demanda (*Zero Mid-Transformer Cut*)

**Arquivos Canônicos:** [`scar_cache/head_expanded_attention.py`](../scar_cache/head_expanded_attention.py) · [`tests/test_scar_store_v2_stage4.py`](../tests/test_scar_store_v2_stage4.py) · [`scripts/audit_scar_store_v2_stage4.py`](../scripts/audit_scar_store_v2_stage4.py) · [`results/scar_store_v2_stage4_audit.json`](../results/scar_store_v2_stage4_audit.json)

### 4.1. Formulação Matemática: Aumentar a Atenção no Raciocínio Sem Cortar o Transformer ao Meio
Em vez de cortar o Transformer na `Layer 12/24` (Etapa 3) ou fatiar o prompt (Etapa 2), `HeadExpandedAttentionBank` mantém **todos os `24` blocos internos do Transformer `100%` intactos bit-a-bit (`torch.equal == True`)** e expande dinamicamente o operador de atenção na saída contextual final **apenas quando $M \ge 1$ cones disparam**:
1. **Reuso Zero-FLOP das `14` Cabeças Pré-Treinadas (`q_proj`):** A `Layer 24` já computa $\mathbf{Q}_t = \mathbf{W}_Q \operatorname{LN}_{\text{in}}(\mathbf{x}_t) \in \mathbb{R}^{14 \times 64}$ (`14 × 64 = 896`). Um *hook* somente-leitura reutiliza essas `14` cabeças sem nenhum GEMM extra de $\mathbf{W}_Q$.
2. **Mesmíssimo Orçamento Físico de `62.720 B` (`61,25 KB` `Block-64 INT4`):** Trocando apenas `4` valores singulares residuais (`rank: 64 → 60`, retendo `100%` da energia espectral em episódios de `48` passos), cada `HeadExpandedScarCapsule` armazena `P = 8` protótipos de chave por cabeça (`8 × 14 = 112` chaves de `64` dimensões = `3.808 B` INT4) + `(B_scar, A_scar)` (`57.120 B`) + `z_anchor` (`1.792 B`) = **exatos `62.720 bytes` (`61,25 KB`)**!
3. **Atenção Competitiva por Cabeça e por Token (`M >= 2`):**
   $$s_{t, m, h} = \max_{p \in \{1..P\}} \langle \hat{\mathbf{q}}_{t, h}, \hat{\mathbf{k}}_{m, p, h} \rangle + \frac{1}{\beta}\ln(g_m), \qquad \alpha_{t, m, h} = \frac{\exp(\beta s_{t, m, h})}{\sum_{m'=1}^M \exp(\beta s_{t, m', h})}, \qquad w_{t, m} = \frac{1}{H}\sum_{h=0}^{H-1} \alpha_{t, m, h}$$
   Elimina o colapso estático `0,50 / 0,50`: cabeças semânticas distintas especializam-se em cartuchos distintos no mesmo token (`87,02%` de divergência entre as `14` cabeças!), e o peso $w_{t, m}$ alterna dinamicamente token a token (`0,9979 ↔ 0,0029`) conforme a frase transita entre *Landauer* e *Immerman-Szelepcsényi*!

### 4.2. Experimento Controlado *Head-to-Head*: Etapa 3 (`Dual-Depth`) vs. Etapa 4 (`Head-Expanded`)

| Métrica Auditada (`Qwen2.5-72B MoS → Qwen2.5-0.5B` + `SCache O(1)`) | Etapa 3 (`Dual-Depth L12+L24`) | Etapa 4 (`Head-Expanded`, Sem Corte) | Veredito Técnico |
| :--- | :---: | :---: | :--- |
| **Tamanho Físico por Cápsula (`Block-64 INT4`)** | `62.720 B` (`61,25 KB`) | **`62.720 B` (`61,25 KB`)** | Paridade exata byte a byte (`r=60, P=8`) |
| **Blocos do Transformer Intactos no Raciocínio** | `11 / 24` (corta `Layer 12`) | **`24 / 24` (`torch.equal = true`)** | **Etapa 4** preserva `100%` do backbone |
| **Vazamento Off-Diagonal em Domínio Único** | `28,43%` (`K-L → Immerman`) | **`0,00%` (`0.0000` em todos)** | **Etapa 4** elimina confusão semântica de meio de rede |
| **Tempo Médio de Síntese `ZERO Backprop` / Cápsula** | `1.941,6 ms` (`2` passes) | **`942,6 ms` (`1` passe — `2,06×` mais rápido)** | **Etapa 4** corta pela metade o custo de consolidação |
| **Top-1 / KL Inédito — *Immerman-Szelepcsényi*** | `100,0%` (`KL 0,0364`) | **`100,0%` (`KL 0,0267`)** | **Etapa 4** atinge menor divergência KL |
| **Top-1 / KL Inédito — *Princípio de Landauer*** | `95,83%` (`KL 0,0445`) | **`95,83%`** (`KL 0,1285`) | Empate em Top-1 (`23/24`); Etapa 3 tem menor KL |
| **Top-1 / KL Inédito — *Kabatiansky-Levenshtein*** | **`91,67%`** (`KL 0,3412`) | `87,50%` (`KL 0,3782`, base `83,33%`) | **Etapa 3** vence por `1/24` token em paráfrase geométrica |
| **Síntese Interdisciplinar *Pré-Sono* (`conf` / `margin`)** | `0,8398` / `0,7053` (`uncertain=true`) | **`0,9720` / `0,9607` (`+25,5 p.p.`, `uncertain=false`)** | **Etapa 4** elimina colisão `0,5/0,5` e alterna `0,998 ↔ 0,003` |
| **Síntese Interdisciplinar *Pós-Sono* (`bridge_A__B`)** | `0,9788` / `0,9702` (`4,04 s` solve) | **`0,9884` / `0,9840` (`2,08 s` solve, `m k_B T`)** | **Etapa 4** suporta ponte de sono `1,94×` mais rápida e mais nítida |

---

## ✅ Etapa 5 (`v2.2.0`): Núcleo Cognitivo `Qwen2.5-3B-Instruct` (`140,00 KB = 35 × 4 KB` `mmap`) vs. `0.5B` + Auditoria *Red-Team* Adversarial (`6/6` Vetores)

**Arquivos Canônicos:** [`scar_cache/head_expanded_attention.py`](../scar_cache/head_expanded_attention.py) (`HeadExpandedScarStore`) · [`scripts/audit_scar_store_v2_stage5_3b_vs_0p5b.py`](../scripts/audit_scar_store_v2_stage5_3b_vs_0p5b.py) · [`scripts/adversarial_core_0p5b_vs_3b_suite.py`](../scripts/adversarial_core_0p5b_vs_3b_suite.py) · [`tests/test_scar_store_v2_stage5_adversarial.py`](../tests/test_scar_store_v2_stage5_adversarial.py) · [`results/scar_store_v2_stage5_3b_vs_0p5b_audit.json`](../results/scar_store_v2_stage5_3b_vs_0p5b_audit.json) · [`results/adversarial_0p5b_vs_3b_audit.json`](../results/adversarial_0p5b_vs_3b_audit.json)

### 5.1. Geometria de Hardware Perfeita no `Qwen2.5-3B-Instruct` (`36L`, $d = 2048$, $16 \times 128$ Cabeças Q)
Ao unificar a atenção expandida sem corte (*Etapa 4*) com a paginação `mmap` + grade Kronecker (*Etapas 1–2*) na classe `HeadExpandedScarStore`, a geometria do `Qwen2.5-3B-Instruct` produz um alinhamento de páginas de sistema operacional **100% livre de desperdício (`0 bytes` de *padding* interno)**:
1. **`B_scar` (`2048 × 60`) + `A_scar` (`60 × 2048`) em `Block-64 INT4`:** `2 × (61.440 B + 3.840 B) = 130.560 B`.
2. **Protótipos de Chave por Cabeça (`P = 8 × 16 = 128` chaves de `128` dim):** `8.192 B` (`uint8`) + `512 B` escalas `FP16` = `8.704 B`.
3. **Âncora Esférica `z_anchor` (`2048` `FP16`):** `4.096 B` (`1` página OS exata).
4. **Total Exato por Cápsula (`HeadExpandedScarStore` `3B`):** **`143.360 bytes` = `140,00 KB` = exatas `35 × 4.096 B` páginas `mmap` do SO (`0 B` de *padding* interno; **`1.015.469×` menor** que os `135,43 GiB` do `72B` BF16!)**, sobre uma grade Kronecker de **`2048 × 2048 = 4.194.304` coordenadas** (`512,00 KB` de bitmap em RAM CPU).

### 5.2. Comparação Auditada *Head-to-Head*: Núcleo `0.5B` (`61,25 KB`) vs. Núcleo `3B` (`140,00 KB`)

| Métrica Auditada (`72.71B MoS → Núcleo` + `SCache O(1)`) | Núcleo `0.5B` (`24L`, $d=896$, `61,25 KB`) | Núcleo `3B` (`36L`, $d=2048$, `140,00 KB`) | Ganho Real do Núcleo `3B` |
| :--- | :---: | :---: | :--- |
| **Alinhamento Físico de Páginas `mmap` (`4.096 B`)** | `62.720 B` (`+2.816 B` *padding* p/ `16` págs) | **`143.360 B` (`35 × 4 KB` exatas, `0 B` *pad*)** | Zero bytes desperdiçados em `mmap` |
| **Capacidade da Grade Esparsa Kronecker ($d \times d$)** | `802.816` células (`98 KB` bitmap) | **`4.194.304` células (`512 KB` bitmap)** | **`5,22×` mais domínios endereçáveis** |
| **Blocos Internos `100%` Intactos (`torch.equal`)** | `24 / 24` (`true`) | **`36 / 36` (`true`)** | Zero corte de meio de rede em ambos |
| **Top-1 / KL Inédito — *Kabatiansky-Levenshtein*** | `87,50%` (`KL 0,3782`) | **`95,83%` (`KL 0,0727`, base `0,5296`)** | **`+8,33 p.p.` Top-1 · `5,20×` menor KL** |
| **Top-1 / KL Inédito — *Princípio de Landauer*** | `95,83%` (`KL 0,1285`) | **`100,00%` (`KL 0,0066`, base `0,6446`)** | **`+4,17 p.p.` Top-1 · `19,47×` menor KL** |
| **Top-1 / KL Inédito — *Immerman-Szelepcsényi*** | `100,00%` (`KL 0,0267`) | **`100,00%` (`KL 0,0079`, base `0,2852`)** | **`100%` Top-1 · `3,38×` menor KL** |
| **Média Global Inédita contra o Titã `72.71B`** | `94,44%` (`68/72` tok · `KL médio 0,1778`) | **`98,61%` (`71/72` tok · `KL médio 0,0291`)** | **`6,11×` menor divergência KL média!** |
| **Síntese Interdisciplinar *Pré-Sono* (`2` cápsulas)** | `conf 0,9720` · spread `87,02%` (justapõe 2 frases) | **`conf 0,9787` · spread `93,03%` (funde 1 equação)** | Corrige alucinação $\log m \to T \cdot m$; pós-sono crava **$k_B T \ln 2$** |

### 5.3. Auditoria *Red-Team* Adversarial (`6/6` Ataques Aprovados em Ambos os Núcleos + 2 Descobertas Reais)
Durante a execução da bateria hostil [`scripts/adversarial_core_0p5b_vs_3b_suite.py`](../scripts/adversarial_core_0p5b_vs_3b_suite.py), o *Red-Team* expôs e corrigiu **duas sutilezas reais de produção** antes da aprovação `100%`:
1. **Roteamento Multi-Turno sob *Chunked Prefill* (`_rolling_id_tail` + `_query_hidden_buf`):** Quando um histórico longo (`1.560` tokens) era ingerido em blocos de `64` tokens (`past_len > 0`), o prefixo `<|im_start|>user\n` do Turno 2 podia cruzar fronteiras de *chunk* sem acionar o roteador. O detector deslizante em `HeadExpandedAttentionBank` / `HeadExpandedScarStore` tornou ambos os núcleos `100%` imunes a *prefill* fatiado e conversas multi-turno longas.
2. **Bitmap de Presença em `bytearray` CPU Pura (`0,0106 ms` / `0,0180 ms` `unlearn()`):** Manter o bitmap de presença como tensor em `mps` forçava uma barreira de sincronização GPU (`~20 ms`) após `slm(...)`. Movê-lo para `bytearray` nativo na cache L1/L2 da CPU eliminou *fences* assíncronos da GPU e reduziu o apagamento cirúrgico LGPD/GDPR para **`10,6 µs` (`0.5B`) e `18,0 µs` (`3B`)**!

| Vetor de Ataque *Red-Team* (`results/adversarial_0p5b_vs_3b_audit.json`) | Resultado `0.5B` (`61,25 KB`) | Resultado `3B` (`140,00 KB`) | Onde o `3B` Demonstra Superioridade Cognitiva |
| :--- | :---: | :---: | :--- |
| **1. Auditoria Anti-Fachada (`untyped_storage().data_ptr()`) & Causalidade** | ✅ `Δ = 0,0 B` (`128→1536` tok), `0,0 KB` discrep., `24/24` exatos | ✅ `Δ = 0,0 B` (`128→1536` tok), `0,0 KB` discrep., `36/36` exatos | Zero bytes ocultos e causalidade real comprovada em ambos |
| **2. Armadilha Lexical (`6` Prompts Hostis c/ Palavras-Chave)** | ✅ `0/6` falsos positivos, `max_cos = 0,5019 < 0,62`, `0 B` `mmap` | ✅ `0/6` falsos positivos, `max_cos = 0,5002 < 0,62`, `0 B` `mmap` | Ambos mantêm `torch.equal = true` em SQL, CSS, culinária, Git e firewall |
| **3. Palheiro Envenenado (`1.560` tok `"999 J"`) + Passkey Evicted** | ✅ Rejeita `"999 J"`, `NLL 4,16` (`1,02×` vs `StreamingLLM`), omite $k_B \ln 2$ | ✅ Rejeita `"999 J"`, **`NLL 3,33` (`1,05×`)**, crava **`kT \ln 2`** exato! | **`3B` preserva a fórmula física exata mesmo após `1.560` tok hostis** |
| **4. Colisão Simultânea `3`-Vias (`+3` Cabeças num Único Rollout)** | ✅ `spread = 91,78%`, mas entrelaça sintaxe ao fundir 3 teoremas | ✅ **`spread = 96,99%` (`0,9992 ↔ 0,0001`)**, frase tripla cristalina! | **`3B` sintetiza os 3 teoremas simultâneos em 1 período perfeito** |
| **5. Pilha Completa `MoS` (`INT4+ASVD`) + `SCache O(1)` + `ScarStore`** | ✅ `3,04×` MLP tail (`149,6→49,2 MB`), `100%` Top-1 (`KL 0,0242`) | ✅ **`3,41×` MLP tail (`774,0→227,1 MB`)**, **`100%` Top-1 (`KL 0,0029`)** | **`8,34×` menor divergência KL** na pilha tripla comprimida |
| **6. Guarda `MissingCapsuleAlert` + `HTTP Range` + GDPR `unlearn()`** | ✅ Bloqueia `(2,2)`, pagina `62.720 B`, **`unlearn = 0,0106 ms`** | ✅ Bloqueia `(2,2)`, pagina `143.360 B`, **`unlearn = 0,0180 ms`** | Recuperação sob demanda + reversão `torch.equal` em **`< 20 µs`** |

---

## ✅ Etapa 6 (`v2.3.0`): Produto CLI/SDK Unificado `mos-store`, Streaming Real `HTTP/1.1 206 Partial Content` & Apagamento Forense `0x00` em SSD

**Arquivos Canônicos:** [`scar_cache/cli_scar_store.py`](../scar_cache/cli_scar_store.py) · [`tests/test_cli_scar_store.py`](../tests/test_cli_scar_store.py) · [`scripts/audit_mos_store_cli_v2_3.py`](../scripts/audit_mos_store_cli_v2_3.py) · [`results/mos_store_v2_3_cli_http206_audit.json`](../results/mos_store_v2_3_cli_http206_audit.json)

1. **CLI Oficial `mos-store` (`pyproject.toml`):** Expõe todo o ciclo de vida em 7 subcomandos reprodutíveis em 1 linha (`mos-store build`, `inspect`, `slice`, `serve`, `query` / `--chat`, `sleep-bridge` e `unlearn --persist`).
2. **Servidor & Cliente Reais RFC 7233 `HTTP/1.1 206 Partial Content` (`ScarStoreHTTPRangeServer` + `HTTPRangeCapsuleClient`):**
   * Auditado em soquete TCP real sobre o `Qwen2.5-3B-Instruct`: ao consultar um domínio ausente (`Immerman-Szelepcsényi`) em um `.scarstore` fatiado (`2/3` cápsulas locais), o cliente requisita exatos **`Range: bytes=17629184-17772543` (`143.360 B = 35 × 4 KB`, `17.629.184 % 4096 == 0`)**, recebe `HTTP 206` em `58,3 ms`, recupera `100%` do teorema `72.71B` e persiste o cartucho localmente (`3` slots).
3. **Apagamento Forense `0x00` em Disco (`mos-store unlearn --persist`):**
   * Além do desligamento instantâneo em RAM, sobrescreve cirurgicamente *in-place* (via `seek + write + fsync`, sem deslocar cápsulas vizinhas) o bit no `presence_bitmap` do arquivo, os `4.096 B` do `z_anchor` e todas as **`35 × 4 KB` páginas (`143.360 B`)** do slot com zeros criptográficos **`0x00`** (`147.456 B` zerados em `7,64 ms` com `fsync`), garantindo reversão `torch.equal == True` ao reabrir o arquivo do zero e `0` dano colateral nos demais domínios.





