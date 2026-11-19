# Roadmap de Pesquisa & Engenharia — `ScarField v10` (*100% Soberano, Zero Open-Weight*)

> **Diretriz Inviolável de Pureza (`AGENTS.md`):**
> Todas as frentes abaixo operam **100% sem pesos abertos de terceiros e sem modelos externos atuando como professores de pesos** (*Zero Open-Weight Programming*, *Zero Transformer*, *Zero Softmax*, *Zero RMSNorm*, *Zero BPE estático*, *Zero KV-Cache linear $\mathcal{O}(T)$*). Toda inteligência emerge da geometria hipersférica dupla ($S_{\text{form}} \times S_{\text{world}}$), reticulados `Gosset E8`, fluxos simpléticos `Cayley-Woodbury`, operadores biortogonais ($\widetilde{U}^\top U = I$), memória constante `Kalman-Scar O(1)` e `PointScarFlow 3D`.

---

## Matriz Mestre de Priorização (`Importância × Complexidade`)

Ordenada pela **razão Retorno Arquitetural / Complexidade de Implementação** (do ganho mais imediato e cirúrgico até a mudança fundacional mais profunda, para testarmos todas sequencialmente):

| Ordem | Código | Frente Pioneira | Importância (Impacto) | Complexidade | Esforço Estimado | Papel na Arquitetura |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1º** | **`P1`** | **RAG Latente Biortogonal `.uscar` (`Zero-Prompt`) + Wipe `0x00` ao Vivo** | **10 / 10** | **3 / 10** (Baixa) | Curto | Separa `Gramática (Cristal E8)` $\perp$ `Fatos (.uscar Zero-SGD)`; permite até upload de `.md/.txt` pelo visitante com wipe `0x00` provado na tela. |
| **2º** | **`P2`** | **Fibrado Gramatical Neolatino (`PT/ES/FR`) + Cicatrizes `.mosbank` por Idioma** | **9,5 / 10** | **3,5 / 10** (Baixa-Média) | Curto | Tronco românico compartilhado em $S_{\text{form}}$ + `4` cicatrizes ternárias (`~24 KB` cada) ativadas no `passo 0`; elimina *code-switching* e corta dados em `4x–6x`. |
| **3º** | **`P3`** | **Motor de Busca Semântico-Visual Poliglota Instantâneo (`< 5 ms` WebGPU)** | **9,0 / 10** | **2,5 / 10** (Baixa) | Curto | Usa $h_{\text{world}}$ byte-level (`cos > 0,96` entre `PT/EN/ES/FR`, imune a *typos*) para buscar papers, códigos, cartuchos e cenas 3D a cada tecla. |
| **4º** | **`P4`** | **Inicialização Espectral Fechada de Fonotática & UTF-8 (`SVD -> Gosset E8`)** | **8,5 / 10** | **4,0 / 10** (Média) | Curto-Médio | Calcula por álgebra linear pura sobre estatísticas do corpus (sem IA externa!) as transições ortográficas/UTF-8 e grava nos polos $E_8$ no `passo 0`. |
| **5º** | **`P5`** | **Auto-Complete Fantasma Sub-Milissegundo (`System-1 Tsallis Co-Pilot`)** | **8,5 / 10** | **4,5 / 10** (Média) | Médio | Projeta continuação *inline* a `< 2 ms` por tecla exibindo ao vivo a esparsidade exata `Tsallis-1.5` (`~96% zeros`). |
| **6º** | **`P6`** | **Laboratório Interativo de Cirurgia de Memória & Geometria (`Slerp 3D` + `1-Shot`)** | **9,0 / 10** | **5,0 / 10** (Média) | Médio | Visitante grava fatos em `1-Shot` num slot `#k/512`, interpola formas 3D por geodésica `Slerp` a `60 FPS` e executa cirurgia `0x00` seletiva. |
| **7º** | **`P7`** | **Raciocínio Neuro-Simbólico Verificado (`ScarField -> Sandbox WebWorker`)** | **8,5 / 10** | **5,0 / 10** (Média) | Médio | Traduz problemas numéricos/algorítmicos em expressões executáveis verificadas em `< 2 ms` num Worker isolado e injeta o resultado exato no `bus_ctx`. |
| **8º** | **`P8`** | **Batedores `Ghost-Jump` Vetorizados em WGSL (`Lookahead K=4` + `Juiz V`)** | **9,0 / 10** | **6,5 / 10** (Média-Alta) | Médio-Alto | Sondagem paralela de `K = 4` ramos por `2–3` choques via `VirtualGhostJumpDelta` (`+0 B` em `W_fast`), podando loops antes da emissão. |
| **9º** | **`P9`** | **Elevação Morfêmica por Choque (`Shock-Wavelet Lift`, Zero BPE) + Gramática por Slots** | **10 / 10** | **8,0 / 10** (Alta) | Alto | Agrupa corridas ortográficas previsíveis ($s_t < \theta$) em *wavelets* dinâmicas (`~4x` mais capacidade semântica e `3x–4x` mais `B/s`, `256` bytes puros). |

---

## Detalhamento Técnico de Cada Frente Experimental

### `P1` — Separação `Cristal Gramatical E8` $\perp$ `RAG Latente Biortogonal .uscar`
- **Hipótese central:** Em esferas biortogonais ($\widetilde{U}^\top U = I$), fatos semânticos são inscritos em `1-Shot` via $(G + \lambda I)^{-1}$ com interferência cruzada zero (`0.0`). Logo, o cristal `Gosset E8` só precisa aprender **como articular gramaticalmente** qualquer vetor entregue em `bus_ctx`.
- **Experimento:** Compilar documentos Markdown/científicos localmente com `mos-digest` em páginas `.uscar` de `4 KB` (`HTTP 206` + upload local drag-and-drop pelo visitante). Medir fidelidade factual multilíngue (`PT/EN/ES/FR`) com `+0 B` adicionados ao prompt e resíduo pós-wipe `0x00` idêntico a `0.000000`.

### `P2` — Fibrado Gramatical por Família Linguística (`Neolatino PT/ES/FR` + `EN`)
- **Hipótese central:** Português, Espanhol e Francês compartilham `~85–89%` de estrutura sintática e sufixação latina (`-ção/-ción/-tion`, `-dade/-dad/-té`). Separar a fibra superficial em $S_{\text{form}}$ impede *code-switching* involuntário e multiplica a eficiência amostral.
- **Experimento:** Treinar o tronco sintático românico em blocos puros e acoplar `4` especialistas ternários `.mosbank` (`scar_pt`, `scar_es`, `scar_fr`, `scar_en`, `~24 KB` cada) selecionados deterministicamente pelo detector de idioma no `passo 0`. Medir taxa de vazamento inter-idioma (meta: `0,0%`) e `BPB` *held-out*.

### `P3` — Motor de Busca Semântico-Visual Poliglota Instantâneo (`< 5 ms`, Zero Servidor)
- **Hipótese central:** Como o Front-End ondulatório + fluxo perceptual do `ScarField` opera sobre bytes crus em `S_world` com alinhamento `PT/EN/ES/FR` `cos > 0,96`, ele constitui um índice vetorial neural tolerante a erros ortográficos em apenas `9,5 MiB`.
- **Experimento:** Implementar barra de busca instantânea no portfólio ranqueando papers Zenodo, seções técnicas, funções de código e morfismos 3D a cada tecla, exibindo distância geodésica em radianos e esparsidade `Tsallis`.

### `P4` — Inicialização Espectral Fechada de Fonotática & UTF-8 (`SVD -> Gosset E8`)
- **Hipótese central:** Transições de bytes UTF-8 multibyte (`0xC3`), dígrafos (`nh`, `lh`, `ch`, `qu`) e morfemas frequentes seguem matrizes de coocorrência estacionárias que podem ser fatoradas por `SVD` e projetadas via Conway–Sloane diretamente nas `240` raízes do `Gosset E8` (`Zero Open-Weight`, `Zero IA Externa`).
- **Experimento:** Comparar velocidade de convergência gramatical (`passos até BPB < 2,0`) entre inicialização aleatória e inicialização espectral fechada nos polos `byte_form_poles` e `reflex_cap_poles`.

### `P5` — Auto-Complete Fantasma Sub-Milissegundo (`System-1 Tsallis Co-Pilot`)
- **Hipótese central:** O caminho `S1` (`1` passo simplético + simplex `Tsallis-1.5` com `~96%` de zeros exatos) roda em `< 2 ms` no WebGPU, permitindo sugestão preditiva contínua enquanto o humano digita.
- **Experimento:** Renderizar *ghost text* translúcido no campo do Jarvis e medir latência P50/P99 por tecla e taxa de aproveitamento de prefixo.

### `P6` — Laboratório Interativo de Cirurgia de Memória & Geometria (`Weight & Scar Surgery Lab`)
- **Hipótese central:** Nenhum modelo convencional permite ao usuário final inspecionar, gravar em `1-Shot`, interpolar geometricamente em 3D e apagar cirurgicamente memórias individuais sem afetar o restante do estado.
- **Experimento:** Construir bancada visual com (a) gravação `1-Shot` num slot `#k/512` e consulta cruzada em outro idioma, (b) slider de interpolação geodésica `Slerp` em $S_{\text{world}}$ transformando uma nuvem 3D em outra a `60 FPS`, e (c) prova numérica ao vivo de `Δ = 0.000000` nos `511` slots vizinhos após `Wipe 0x00`.

### `P7` — Raciocínio Neuro-Simbólico Verificado (`ScarField -> Sandbox WebWorker`)
- **Hipótese central:** O `ScarField` já atinge `100%` de exatidão estrutural ao emitir funções Python curtas (`AST + exec`). Delegar aritmética/álgebra exata a um avaliador determinístico local elimina erros de cálculo sem inflar parâmetros.
- **Experimento:** Acoplar micro-avaliador determinístico em WebWorker isolado (`sem rede, timeout 5 ms`) alimentando o escalar calculado de volta ao `bus_ctx` antes da frase final.

### `P8` — Batedores `Ghost-Jump` Vetorizados em WGSL (`Lookahead K=4` + `Juiz V(h_world)`)
- **Hipótese central:** A maior causa de deriva autorregressiva em modelos compactos é escolher um byte localmente plausível que leva a uma região de baixa previsibilidade 2 passos adiante.
- **Experimento:** Portar `VirtualGhostJumpDelta` (`scar_field/engine.py`) para kernel WGSL avaliando `K = 4` candidatos em paralelo quando $\Delta p < 0,24$, combinando probabilidade `Tsallis` + `Juiz` $V(h_{\text{world}})$ - penalidade repulsiva.

### `P9` — Elevação Morfêmica Dinâmica por Choque (`Dynamic Shock-Wavelet Lift`, Zero BPE)
- **Hipótese central:** Usar a própria surpresa `Tsallis` ($s_t \ge \theta_{\text{morf}}$) para comprimir sequências ortográficas previsíveis em pulsos semânticos de passo variável multiplica por `~4x` o horizonte e a capacidade conceitual do cristal sem introduzir vocabulário BPE fixo.
- **Experimento:** Implementar agregador causal de onda entre choques consecutivos + decodificador local de rajada ortográfica, validando paridade `PyTorch <-> C99 <-> WGSL` (`cos >= 0.99999`) e ganho real de `B/s` e `BPB` em gramática aberta condicionada por slots.
