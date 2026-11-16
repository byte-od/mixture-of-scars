# Roadmap Arquitetural v7.0 / v8.0 — Arquitetura Hipersférica Limpa ("Zero Contra-Pesos") & Universal ScarStore

> **Status:** Especificação de Pesquisa & Engenharia para `ScarNativeLM v7.0 / v8.0`  
> **Diretriz:** Eliminar todos os cálculos compensatórios herdados da era de adaptação de Transformers euclidianos irrestritos (`v1–v4`), garantindo que pesos, ativações, memória `SCache O(1)` e cartuchos `.uscar` nasçam, treinem e operem em escala estritamente normalizada desde `t = 0`.

---

## 1. Diagnóstico Forense: De Onde Vieram os "Contra-Pesos" na `v1–v6`?

Embora o `ScarNativeLM v5.0 / v6.0` já seja treinado 100% do zero (*Tabula Rasa*, zero `nn.Linear`), a mecânica interna das camadas ainda herdou a geometria euclidiana irrestrita ($\mathbb{R}^D$) criada originalmente na `v1–v4` para comprimir modelos legados de terceiros (`Qwen` / `Yi`):

| Sintoma noEspaço Euclidiano Irrestrito (`v1–v6`) | "Contra-Peso" Compensatório Exigido Hoje | Custo Computacional Desnecessário |
| :--- | :--- | :--- |
| Soma residual aberta $h_{l+1} = h_l + \Delta h$ faz $\|h_l\|_2$ explodir de `1` para `> 7.500` | `RMSNorm` antes de toda atenção/MLP + reescala `coord_rms` no compilador `mos-digest` | Leituras/escritas extras de barreira por camada e sensibilidade de hiperparâmetros à profundidade $L$ |
| Canais *outliers* concentram variância em poucos eixos | Vetor de equilibração `d_eq` + rotação `FWHT-64` (`vq4_equilibrate_fwht64`) antes de cada GEMV | **1 compute shader inteiro extra** disparado antes de cada projeção linear |
| Blocos de pesos têm normas arbitrárias em $\mathbb{R}^{O \times I}$ | Escalas `FP16` por bloco de 64 (`vq_scales`, `a_scales`) e por linha (`b_row_scales`) | Metadados extras de banda de memória e multiplicações de dequantização por bloco |
| `Q, K, V` sem norma unitária nativa | Normalização explícita + `clamp(-20, 20)` + portas sigmóides de contenção no `SCache` | Operações defensivas para impedir divergência da matriz rápida $W_{\text{scar}}$ |
| Mistura entre sintaxe/idioma, RoPE e semântica no mesmo vetor irrestrito $h_{2L/3}$ | `Ridge-LDA` + `Procrustes R_lang` + projetor `U_lang` + chave dupla `z_dual` | Pipeline pós-treino de 4 estágios apenas para isolar a identidade limpa da entidade |

---

## 2. A Solução Matemática (`v7.0 / v8.0`): Variedade Hipersférica Unitária ($\mathbb{S}^{D-1}$) Ponta a Ponta

Quando **todos os estados e linhas de pesos** vivem nativamente na hipersfera unitária $\mathbb{S}^{D-1} = \{x \in \mathbb{R}^D : \|x\|_2 = 1\}$ (onde o RMS de cada coordenada é **identicamente** $1/\sqrt{D}$ em todas as camadas e em todos os passos de treino):

### Pilar 1 — Fluxo Geodésico Esférico (Fim da Explosão Residual e da `RMSNorm` com Pesos)
Em vez de somar vetores sem limite em $\mathbb{R}^D$, cada bloco atualiza o estado por interpolação esférica (LERP normalizado na variedade $\mathbb{S}^{D-1}$):

$$h_{l+1} = \operatorname{Norm}\!\Big(h_l + \alpha_l \big(\operatorname{Norm}(f_l(h_l)) - h_l\big)\Big), \qquad \|h_l\|_2 \equiv 1 \;\;\forall l \in \{0, \dots, L\}$$

* $\alpha_l \in \mathbb{R}^D$ é o único passo de interpolação aprendível (inicializado em $\approx 1/L$).
* **Consequência:** $\|h_0\|_2 = \|h_9\|_2 = \|h_{96}\|_2 = 1,000000$ sempre. Não existe deriva de norma entre um modelo de `12` camadas e um de `96` camadas.

### Pilar 2 — Projeção `Spherical-D4` em 1 Único Dispatch (Fim do `d_eq`, `FWHT-64` e Escalas por Bloco)
* Após cada passo do otimizador no pré-treino, cada linha de peso mestre é re-projetada na esfera unitária ($\|w_i\|_2 \equiv 1$).
* Pela **concentração de medida em $\mathbb{S}^{D-1}$**, vetores unitários isotrópicos não formam *outliers* de canal: cada fatia 4D possui energia esperada exata $4/D$.
* Isso elimina o shader `vq4_equilibrate_fwht64`, o vetor `d_eq` e as tabelas de escalas `FP16` por bloco: o `Spherical-VQ4 + Cicatriz Unitária` passa a rodar em **1 único kernel WGSL fundido por matriz**, com apenas um escalar de temperatura $\tau$ na saída.

### Pilar 3 — `SCache O(1)` Espectralmente Contrativo por Construção
Com $\|k_t\|_2 \equiv 1$, $\|v_t\|_2 \equiv 1$ e $\|q_t\|_2 \equiv 1$ garantidos pela geometria da rede:

$$W_{\text{scar}}^{(t)} = (1 - \beta_t)\, W_{\text{scar}}^{(t-1)} + \beta_t \big(v_t - W_{\text{scar}}^{(t-1)} k_t\big) k_t^\top, \qquad \beta_t \in [0, 1]$$

* Por indução elementar, $\|W_{\text{scar}}^{(t)}\|_2 \le 1$ para $t \to \infty$ **sem nenhum contra-peso, clamp arbitrário ou fator de amortecimento empírico**.

### Pilar 4 — Variedade Produto Desembaraçada & `Universal ScarStore (.uscar)`
Em vez de misturar idioma, posição e fatos nos mesmos $D$ neurônios e tentar separá-los depois com `Procrustes + U_lang + z_dual`, particionamos o estado latente em dois subespaços esféricos ortogonais por design:

$$\mathcal{M} = \mathbb{S}_{\text{syn}}^{D_{\text{syn}} - 1} \times \mathbb{S}_{\text{sem}}^{D_{\text{sem}} - 1}$$

1. **Canal Sintático/Fluência ($\mathbb{S}_{\text{syn}}$):** Carrega idioma (`PT/EN/ES/FR/...`), gramática local e fase posicional RoPE. Cartuchos **nunca** leem nem escrevem aqui.
2. **Canal Semântico Universal ($\mathbb{S}_{\text{sem}}$):** Treinado desde `t = 0` com perda de alinhamento translingual + perda de **uniformidade hipersférica** ($\log \mathbb{E}[\exp(-2\|z_i - z_j\|_2^2)]$), espalhando entidades uniformemente sobre a esfera ($\langle z_i, z_j \rangle \approx 0$ para $i \ne j$ e $\langle z_i^{\text{pt}}, z_i^{\text{en}} \rangle \approx 1$).
3. **`Universal ScarStore (.uscar)` Desacoplada:**
   * Roteamento reduz-se a **um único produto interno puro** $\langle z_{\text{sem}}, k_* \rangle \ge \tau$ (zero `R_lang`, zero `U_lang`, zero `z_dual`, zero falso-roteamento pós-wipe GDPR).
   * Injeção factual é uma rotação geodésica limpa em $\mathbb{S}_{\text{sem}}$, transportável entre modelos de dimensões distintas (`d=512`, `d=2048`, `d=8192`) via uma única matriz Stiefel $(W_{\text{in}}^{(m)}, W_{\text{out}}^{(m)})$ por modelo — sem nunca precisar reescrever os cartuchos `.uscar` quando o tronco evolui.

---

## 3. Fases de Execução Recomendadas (`Zero Legacy`)

1. **`v7.0` — Núcleo Hipersférico Limpo (`Clean Spherical Scar Core`):**
   * Novo pacote isolado (`scar_clean/` — zero código legado da `v1–v6`), com fluxo geodésico $\|h_l\|_2 \equiv 1$, otimizador Riemanniano/projetado na esfera, quantização `Spherical-D4` sem `d_eq`/`FWHT` (redução de ~40% no número de dispatches WebGPU por camada) e `SCache O(1)` contrativo puro.
2. **`v8.0` — Variedade Produto $\mathbb{S}_{\text{syn}} \times \mathbb{S}_{\text{sem}}$ & Protocolo `.uscar` Universal:**
   * Separação estrutural nativa entre sintaxe/idioma e semântica no tronco + especificação aberta da `Universal ScarStore (.uscar)` interoperável entre múltiplos tamanhos de modelo sem recompilação de cartuchos.
