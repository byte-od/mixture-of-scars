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
