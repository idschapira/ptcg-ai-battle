# CLAUDE.md — PTCG AI Battle

Guia operacional do projeto. Ler antes de qualquer tarefa. Objetivo: agente de IA para a
competição Kaggle "Pokemon TCG AI Battle" (categoria Simulação, deadline **16/Ago/2026**).

## Regra nº 0 — NÃO recriar o motor
O Kaggle fornece o motor oficial (`ptcg_engine`), vendorizado como o módulo Python `cg/` na raiz.
Nosso código é só o "cérebro": recebe a observação e devolve índices de ações legais. O `cg.api`
é a **fonte de verdade** sobre cartas/ataques/regras — sempre reconciliar dados nossos contra ele.

## Contrato do agente
`agent(obs_dict: dict) -> list[int]`. `obs = cg.api.to_observation_class(obs_dict)`.
- `obs.select is None` → seleção inicial: retornar o deck (60 card IDs de `deck.csv`, com fallback
  `/kaggle_simulations/agent/deck.csv`).
- caso contrário → retornar índices de `obs.select.option`, respeitando `minCount`/`maxCount`, sem
  duplicados, cada índice em `[0, len(option))`.
- O engine re-oferece o MAIN após cada ação que não encerra o turno → **desenvolver o board antes
  de atacar** (bug caro da Sprint 3; bandas de score: evolve 80 > baixar 70 > attach 55 >
  trainer/ability 35–40 > attack 20–65 capado > end).

## Arquitetura de decisão (decidida)
**Policy-net-first.** Runtime = rede policy+value com busca rasa opcional; NÃO é LLM (sandbox é
offline). MCTS/ISMCTS via `cg.search_begin` fica como Camada 2 opcional (PTCG é informação
imperfeita + estocástico, então não é AlphaZero puro). LLM/MCP são ferramentas de DEV
(copiloto + destilação de efeitos offline), nunca runtime.

**Princípio crítico: treinar OFFLINE com PyTorch, mas rodar a INFERÊNCIA em NUMPY PURO na
submissão.** Exportar pesos para `.npz` e implementar o forward em numpy. Torch fica em
`requirements-dev.txt`; o `requirements.txt` de runtime permanece mínimo.

## Restrições do Kaggle (runtime)
- 12.2 GiB RAM, 2 vCPUs, 11.8 GiB HDD. Offline (sem rede/LLM em runtime).
- Submissão = `submission.tar.gz` ≤ 197.7 MiB, com `main.py` na RAIZ + `deck.csv`.
- Latência: milissegundos por jogada. Otimizar memória; preferir Polars/numpy.

## Estrutura do repo
- `cg/` — engine oficial vendorizado (não alterar).
- `src/ingestion/` — pipeline de dados (Polars → Parquet in-memory), `card_index.py`,
  `reconcile.py`, `build_effect_model.py`, `dim_effect_overrides.csv` (curado à mão — versionado),
  `replays_download.py`, `replays_parse.py`.
- `src/environment_wrapper/` — `wrapper.py` (parse, is_initial_selection, legal_option_indices,
  option_summary, enrich), `selfplay.py`, `arena.py`, `recorder.py`.
- `src/agent_heuristics/` — `random_agent.py`, `heuristic_agent.py` (expõe `last_scores`).
- `src/rl_models/` — `encoding.py` (encoders numpy), rede (torch, dev) + inferência (numpy).
- `tests/`, `viewer/battle_viewer.html` (single-file, offline), `main.py`, `deck.csv`.

## Modelo de dados (fatos)
- Star schema em memória: `dim_card` (1 linha/carta), `dim_attack` (alinhado a `all_attack()`),
  `dim_skill` (abilities), `dim_effect` (attack_id × effect_seq), `bridge_attack_energy`. Chaves
  inteiras; lookup O(1) via `CardIndex`/`EffectIndex` (dicts de dataclasses frozen/slots).
- Namespace de energia unificado ao engine: `COLORLESS=0 … TEAM_ROCKET=11`. `竜`→Dragão(N),
  `●`→incolor(C).
- Encoders (`src/rl_models/encoding.py`, numpy — importar, NUNCA reimplementar):
  `ENCODING_DIM=1185`, `OPTION_DIM=154`, `MAX_OPTIONS=64` (overflow → clamp+warning),
  `BENCH_CAP=8` (Area Zero Underdepths, carta 1250).
- `is_cost_payable` vive em `card_index.py`; usado pelo HeuristicAgent e OptionEncoder (mesma
  semântica nos dois).

## Gotchas de domínio (verificados)
- **Tera no banco = imune a dano** (engine oferece como alvo legal mas aplica dano 0). Teras
  próprios são bench-safe; Teras adversários no banco são alvo desperdiçado (penalidade no score).
  Coberto por `tests/test_tera_bench_immunity.py` (teste de contrato — falha se o engine mudar).
- Deck é **fixo** por submissão (não se monta em runtime; oponente é oculto na seleção inicial).
  Deck building = otimização OFFLINE (Epic 4.5): legalidade → semente de arquétipo (LLM) → busca
  por winrate na arena → co-otimização deck↔piloto.

## Convenções de código
Tipado (`typing`), modular, **None-safe em todos os caminhos** (id ausente → None/zeros, nunca
`KeyError`/crash; sempre retornar jogada legal válida). Polars/numpy otimizado para memória.
Avaliar impacto de performance antes de adicionar bibliotecas pesadas de RL.

## Verificação (obrigatória)
- `reconcile.py` deve dar exit 0 (0 missing/mismatch em cards/attacks/skills/tera) após mexer no
  modelo de dados.
- Testes de contrato + `unittest` verdes antes de commitar.
- Gates de qualidade via `arena.py`: Gate B (heurística > random, >65% — atual: 91%),
  Gate C (rede ≥ melhor baseline). `exceptions` sempre = 0.
- `build_submission` valida: main.py na raiz, tamanho < 197.7 MiB, smoke test do pacote extraído.

## Git
- Commits atômicos por task.
- Gitignored: `data/processed/*.parquet`, `data/raw/`, `pokemon-tcg-ai-battle/` (dados brutos do
  Kaggle — reproduzíveis via download, não redistribuir), `submission.tar.gz`, replays e datasets
  (`.npz`), `viewer/recordings/`. Versionado: código, `viewer/battle_viewer.html`,
  `dim_effect_overrides.csv`, `deck.csv`, docs.

## Status atual
Sprints 1–5D concluídas. Gates A, B (91%) e C OK. Produção = par casado DECK-AGNÓSTICO 5D
(`models/policy_value.npz` + `models/feature_stats.npz`): imitação dos líderes sob stats mistas
(3 decks self-play + replays; val top-1 56,4%), Gate C 51–53% sobre o par 5B anterior
(preservado como `*_pre5d.npz`; clone 5A em `*_bc5a.npz`). Latência 545µs/jogada; paridade
torch↔numpy ~1e-14. Deck de submissão segue **Abomasnow**: no gauntlet justo (piloto agnóstico
dos dois lados, `src/deckbuilding/gauntlet.py`) Abomasnow 73% > Lucario 39% > Iono 38% — nenhum
piloto atual executa a linha de setup do Lucario (vs random no Lucario: só 65%). ATENÇÃO: par
policy+stats é CASADO — nunca promover um sem o outro. Value head segue congelado/descalibrado
(critic é da 5C). Próximo: Fase E (especializar piloto no deck de maior teto — Lucario — via
BC/self-play deck-específico) e/ou 5C (self-play RL).
