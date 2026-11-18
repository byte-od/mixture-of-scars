# Guia Direto de Publicação Imediata na Zenodo (Pós-`v3.0` + Paper Dedicado de Dados Públicos)

Como o último preprint publicado na Zenodo foi a **`v3.0`** (`DOI: 10.5281/zenodo.24172784`), todos os pacotes abaixo foram revisados, auditados forensicamente contra os arquivos JSON empíricos e empacotados **exclusivamente para a Zenodo** (sem artefatos de `arXiv`/`TMLR`).

---

## Parte A — Evolução da Série Principal (`New Version` sobre `10.5281/zenodo.24172784`)

Na página do registro atual da `v3.0` na Zenodo, clique em **`New version`** em sequência para publicar cada marco evolutivo (ou suba diretamente na ordem `v4.0` $\to$ `v5.0` $\to$ `v6.0` $\to$ **`v8.2.0`** para preservar o histórico versionado completo no Concept DOI):

### 1. Versão `4.0.0` — `zenodo_deposit_v4_0/`
- **Escopo:** Operadores Unificados em Modelos Pré-Treinados (`0.5B -> 3B -> 72.71B <24 GB`, `VQ4-Scar`, `ProtectedInt4Scar` `20` Shaders WebGPU, `GoldenSlice P_⊥lang`) + Extração da biblioteca física independente `ScarCodec` (já atualizada com as métricas forenses auditadas: `188,2x` menor RMSE na EDP de Burgers 2D via Rusanov contra `rFFT` FP16 pareada byte a byte, covariância ruidosa cega e recall parcial `4-ch` mascarado `0,9999`).
- **Arquivos para arrastar no upload da `v4.0.0`:**
  1. `Mixture_of_Scars_and_SCache_Preprint_v4.0.pdf` *(marcar como arquivo principal / preview)*
  2. `MoS_SCache_Adversarial_Replication_v4.0.zip`
  3. `MoS_SCache_LaTeX_Source_v4.0.zip`
  4. `mos_scache_v4_0_hero.png`
- **Metadados prontos para copiar:** `zenodo_deposit_v4_0/ZENODO_METADATA_v4.0.json`

### 2. Versão `5.0.0` — `zenodo_deposit_v5_0/`
- **Escopo:** Primeira fundação *Pure Scar-Native* (`ScarNativeLM v5.0`, pré-treino `Tabula Rasa` `step = 0 -> 520` sem `nn.Linear`, `1,02 MiB`, `0,000000 nats` de choque PTQ, `DifferentiableSCacheAttention` `124,25 KB`, cartuchos `Zero-SGD RRR` `8 KB` e `22` Shaders WebGPU).
- **Arquivos para arrastar no upload da `v5.0.0`:**
  1. `Pure_Scar_Native_and_MoS_Preprint_v5.0.pdf` *(arquivo principal)*
  2. `MoS_ScarNative_Adversarial_Replication_v5.0.zip`
  3. `MoS_ScarNative_LaTeX_Source_v5.0.zip`
  4. `mos_scarnative_v5_0_hero.png`
- **Metadados prontos para copiar:** `zenodo_deposit_v5_0/ZENODO_METADATA_v5.0.json`

### 3. Versão `6.0.0` — `zenodo_deposit_v6_0/`
- **Escopo:** Escala `12L` em `NVIDIA A100` (`ScarNativeLM v6.0` com *Chunkwise WY Widrow-Hoff* `12,63x` mais rápido, *Shared-Stage MoS* `4,13 MiB` / `1,90 MiB` em *Stage-Streaming* com `Δlogits = 0,0`, *Dual-Tier SCache O(1)* com `100%` Passkey até `T = 567`, compilador documental `mos-digest` com Procrustes + chave dupla Ridge `98,96%` em `PT/EN/ES/FR` e `100%` reversão `<abstain>` pós-*wipe* documental `0x00` GDPR, `v6-Frontier 96L` em `0,71 GiB` e gerador 3D contínuo `PointScarFlow v1.0` em `26 + 12` Shaders WebGPU).
- **Arquivos para arrastar no upload da `v6.0.0`:**
  1. `ScarNative_v6_and_PointScarFlow_Preprint_v6.0.pdf` *(arquivo principal)*
  2. `ScarNative_v6_Adversarial_Replication_v6.0.zip`
  3. `ScarNative_v6_LaTeX_Source_v6.0.zip`
  4. `mos_scarnative_v6_0_hero.png`
- **Metadados prontos para copiar:** `zenodo_deposit_v6_0/ZENODO_METADATA_v6.0.json`

### 4. Versão `8.2.0` — `zenodo_deposit_v7_v8/`
- **Escopo:** Fundação Pós-Transformer Zero-Legado (`ScarField v7.0–v7.5`: reticulado `Gosset E8` `1,000 bpw` + cicatrizes ternárias, fluxo simplético reversível de Cayley-Woodbury `11,65x` menor VRAM, tempo próprio relativístico de surpresa, `Chunkwise WY` `306,87x` + `BiorthogonalShockBank` `455.504 B` `O(1)`, kernels `@triton.jit` + C99 SIMD `58,76x` CPU via `.sfield.scar` `mmap cos = 1,000000`, simplex esférico `Tsallis-1.5` e fechamento causal anti-stuttering `0/24`) + Frota Vetorizada de Macro-Batedores Geodésicos & Enxame sobre Cristal $E_8$ Compartilhado (`ScarField v8.0–v8.2`: conjugação simplética exata $M_{\text{fwd}}^{-T} \equiv J M_{\text{rev}}^{-1} J^T$, `3,27x–3,34x` speedup, `0` OOM retries, `59,98M` parâmetros esféricos em `9,527 MiB` vencendo `5/5` domínios *held-out*, `0/8` vazamentos prosa→código, `2/2` Python `exec()` ao vivo, `100%` PIN 3 turnos na `v8.1` e Passkey `O(1)` até `T = 4.096` bytes = `64x` o anel sensorial).
- **Arquivos para arrastar no upload da `v8.2.0`:**
  1. `ScarField_v7_and_v8_Post_Transformer_Preprint_v8.2.pdf` *(arquivo principal)*
  2. `ScarField_v7_v8_Replication_v8.2.zip`
  3. `ScarField_v7_v8_LaTeX_Source_v8.2.zip`
  4. `mos_scarfield_v7_v8_hero.png`
- **Metadados prontos para copiar:** `zenodo_deposit_v7_v8/ZENODO_METADATA_v8.2.json`

---

## Parte B — Novo Registro Independente (`New Upload` na Zenodo — Paper Dedicado de Física & Dados Públicos)

Clique em **`New upload`** na Zenodo (gerando um **novo DOI independente** focado exclusivamente em Física Aplicada, Astronomia Observacional, Telemetria Espacial e Processamento de Sinais, vinculado a `10.5281/zenodo.24172784` via *Related Identifiers*):

### 5. Versão `1.0.0` / `3.0.0` — `zenodo_deposit_scarcodec_public_data_v1_0/` & `zenodo_deposit_scarcodec_v6_3_biorthogonal_v3_0/`
- **Título:** *Empirical Evaluation of Domain-Independent ScarCodec Operators on Public Observational Astronomy, Spacecraft Telemetry, and 64-Channel Biomedical Sensor Archives*
- **Escopo Forense (`0` dados sintéticos, `0` oráculo, `0` alterações no `scar_codec/core.py`):**
  - **`PhysioNet EEGMMIDB v1.0.0` (`64` canais Ag/AgCl, `160 Hz`):** Calibração cega *cross-run* `P_⊥` em `S001R01.edf` aplicada à corrida *held-out* `S001R04.edf` (`10,47x` compressão, `+14,63 dB` no arranjo cortical e `+8,40 dB` no Laplaciano físico de Hjorth em `C3` vs `-4,85 dB` no ADC `2-bit`) + stream cronológico `39.520 × 64` em `34.816 B` (`Δ = 0 B`) com interpolação cega `32 -> 32` eletrodos ocultos a `cos = 0,99667`.
  - **`NASA Space Shuttle Radiator Control` (`Statlog Shuttle`, UCI ID `148`):** Retenção de **`6/6 = 100,0%`** das falhas críticas raras de válvula de *bypass* (`Classes 6 & 7`, prevalência `0,0414%` em `14.500` quadros) em memória `O(1)` (`Δ = 0 B`) vs `0/6` no anel circular de mesmo tamanho em bytes.
  - **Três Arquivos Astronômicos FITS (`NASA Spitzer IRAC`, `CSO Bolocam Sgr A* 1.1 mm`, `FCRAO 14-m L1448 13CO`):** Compressão `11,77x–12,27x` (enquanto `zlib -9` trava em `1,10x–1,14x`), leitura aleatória `O(1)` por bloco `23,2x–34,7x` mais rápida (`Δbloco = 0,0`) e descoberta empírica da lei monotônica de contraste HDR sobre `rFFT` 2D FP16 pareada byte a byte (`4,6x -> 45,9x -> 2.093,6x` menor erro fotométrico de pico).
- **Arquivos para arrastar no upload:**
  1. `ScarCodec_Public_Observational_Datasets_Preprint_v1.0.pdf` *(arquivo principal)*
  2. `ScarCodec_Public_Data_Replication_v1.0.zip`
  3. `ScarCodec_Public_Data_LaTeX_Source_v1.0.zip`
  4. `scarcodec_public_data_v1_0_hero.png`
- **Metadados prontos para copiar:** `zenodo_deposit_scarcodec_public_data_v1_0/ZENODO_METADATA_public_data_v1.0.json`
