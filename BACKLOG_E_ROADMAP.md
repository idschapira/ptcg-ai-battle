# PTCG AI Battle — Backlog Ágil & Roadmap

**Deadline (categoria Simulação):** 16 de Agosto de 2026
**Data base deste plano:** 06/Jul/2026
**Janela útil:** ~6 sprints semanais

---

## 0. Confirmação de Contexto (Análise técnica)

Todos os artefatos da competição estão acessíveis na pasta do projeto e foram inspecionados:

| Artefato | Status | Observação |
| :-- | :-- | :-- |
| `EN_Card_Data.csv` | OK | 2.102 linhas, 17 colunas. **Grão = carta × golpe** (cartas com múltiplos ataques repetem o `Card ID`). |
| `JP_Card_Data.csv` | OK | Mesma estrutura (opcional / paridade de meta). |
| `Card_ID List_EN.pdf` / `_JP.pdf` | OK | Referência visual dos IDs. |
| `ptcg_engine/` | OK | Código-fonte C++ oficial (headers + `Export.cpp`). **Não recriar o motor.** |
| `sample_submission/` | OK | `main.py` + `deck.csv` (60 IDs) + módulo Python `cg`. |

**Insights arquiteturais que mudam a estratégia:**

1. **O engine já entrega os metadados estruturados.** `cg.api.all_card_data()` e `all_attack()` retornam dataclasses (`CardData`, `Attack`) direto do motor. O CSV vira fonte de *enriquecimento/analytics*, não a única fonte de verdade — bom para consistência com o que o agente realmente vê em partida.
2. **Contrato do agente é simples e determinístico.** `agent(obs_dict) -> list[int]`: o motor sempre apresenta as *legal moves* em `obs.select.option`; o agente devolve índices (respeitando `minCount`/`maxCount`, sem duplicados). Na 1ª seleção `obs.select is None` → retornar o deck (60 IDs).
3. **Lookahead nativo disponível.** O engine expõe `search_begin` / `search_step` / `search_end` — permite planejamento tipo MCTS sem simulador próprio. Abre um caminho de *search-based agent* competitivo antes mesmo do RL pesado.
4. **Estado rico e já tipado.** `Observation → State → PlayerState → Pokemon` cobre HP, energias, banco, prizes, condições especiais. A codificação de estado para RL parte daqui, não do zero.

---

## 1. Backlog — Epics & Tasks

### Epic 1 — Data Prep & Pipeline (foco: in-memory Parquet)
Objetivo: transformar `EN_Card_Data.csv` (grão carta×golpe) em um modelo dimensional consultável em milissegundos, carregado em memória.

- **1.1** Scaffold do repo: `/src` (`ingestion`, `environment_wrapper`, `agent_heuristics`, `rl_models`), `/data/raw`, `/data/processed`.
- **1.2** Ingestão com Polars: leitura tipada, normalização de tokens `{G}/{R}/{W}...`, tratamento de `n/a`/vazios.
- **1.3** Modelagem estrela em memória → Parquet: `dim_card` (grão 1 linha/carta), `dim_attack` (golpe + custo/dano/efeito), `bridge_attack_energy` (custo por tipo de energia). Chaves inteiras.
- **1.4** Camada de acesso (`CardIndex`): dicionários `card_id -> struct` e caches para lookup O(1) durante a partida.
- **1.5** Reconciliação CSV ↔ `all_card_data()` do engine (validar cobertura de IDs, logar divergências).
- **1.6** Testes de perfil: memória residente do índice e latência de lookup (< 1 ms/consulta).

### Epic 2 — Wrapper do Motor (integração `ptcg_engine`)
Objetivo: adaptador Python fino entre o motor oficial e o agente/RL.

- **2.1** `EnvironmentWrapper`: `to_observation_class(obs_dict)` → objeto tipado; helpers para enumerar/decodificar `option`.
- **2.2** Harness de self-play local: dois agentes random jogando N partidas, coleta de resultado (`State.result`).
- **2.3** Adaptador de submissão: garantir contrato `main.py` na raiz + leitura de `deck.csv` (`/kaggle_simulations/agent/`).
- **2.4** Interface de busca: wrapper sobre `search_begin/step/end` para rollout/planejamento.
- **2.5** Logger de partidas (parse de `obs.logs`) para depuração e futura imitation learning.

### Epic 3 — Baseline Agents (heurísticas)
Objetivo: agentes por regra fortes e baratos, e primeira submissão válida.

- **3.1** `RandomAgent` validado end-to-end + **primeira submissão** `submission.tar.gz`.
- **3.2** `HeuristicAgent` v1: priorizar maior dano efetivo (considerando fraqueza/resistência), anexar energia, evoluir quando possível, gerir retreat.
- **3.3** Heurística de mulligan/setup e de escolha de prize.
- **3.4** `SearchAgent`: 1–2 ply usando o lookahead do engine com função de avaliação heurística.
- **3.5** Arena interna: ranking round-robin entre baselines (métrica proxy do rating μ do Kaggle).

### Epic 4 — Treinamento e RL (Deep Learning)
Objetivo: agente que aprende, respeitando 12.2 GiB RAM / 2 vCPU.

- **4.1** Encoding de State Space (tensor compacto) e Action Space (índices de `option`) + máscara de ações legais.
- **4.2** Ambiente estilo Gym sobre o `EnvironmentWrapper` (reset/step/reward = resultado + shaping por prizes).
- **4.3** Treino PPO com action masking (baseline: PPO leve; avaliar Stable-Baselines3 vs. implementação enxuta — SB3 pesa em RAM, decidir por perfilamento).
- **4.4** (Opcional) Imitation Learning a partir dos replays diários dos melhores episódios.
- **4.5** Deck optimization: seleção de `deck.csv` acoplada à política treinada.
- **4.6** Empacotamento final: quantização/pruning do modelo para caber em 197.7 MiB e rodar em 2 vCPU dentro do orçamento de tempo por jogada.

---

## 2. Roadmap Semanal (Sprints)

| Sprint | Período | Epic foco | Entregável de saída |
| :-- | :-- | :-- | :-- |
| **S1** | 06–12 Jul | Epic 1 | Repo scaffold + pipeline de ingestão gerando Parquet dimensional em memória; `CardIndex` com lookup O(1). |
| **S2** | 13–19 Jul | Epic 2 + 3.1 | Wrapper do motor + harness self-play; **1ª submissão válida** (RandomAgent) no leaderboard. |
| **S3** | 20–26 Jul | Epic 3 | `HeuristicAgent` v1 + `SearchAgent` 1-ply; arena interna ranqueando baselines. |
| **S4** | 27 Jul–02 Ago | Epic 4 (setup) | State/Action encoding + ambiente Gym com action masking validado (loop de treino roda). |
| **S5** | 03–09 Ago | Epic 4 (treino) | PPO treinando; agente supera baseline heurística na arena interna; deck co-otimizado. |
| **S6** | 10–16 Ago | Epic 4 (final) | Ajuste fino, empacotamento < 197.7 MiB, testes contra meta-decks. **Submissão final (16/Ago).** |

**Marcos (gates):**
- **Gate A (fim S2):** existe submissão pontuando no Kaggle. Sem isso, congela RL e prioriza pipeline de submissão.
- **Gate B (fim S3):** heurística > random com folga → base sólida de reward/eval para RL.
- **Gate C (fim S5):** política de RL ≥ melhor baseline → segue para tuning; senão, submete o `SearchAgent` como fallback competitivo.

**Riscos & mitigação:**
- *RAM estourar com SB3/tensores:* perfilamento em S4; fallback para PPO enxuto e batches menores.
- *Latência por jogada:* `CardIndex` em memória + profundidade de busca adaptativa.
- *Regras complexas de deck legal:* validação do `deck.csv` contra o engine já em S1–S2.

---

*Fonte dos dados: arquivos locais do projeto (`pokemon-tcg-ai-battle/`) e inspeção da API em `cg/api.py`. Página oficial: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview (render client-side; conteúdo confirmado via metadados + material anexado).*
