# PTCG AI Battle — Backlog & Roadmap (consolidado)

**Deadline (categoria Simulação):** 16 de Agosto de 2026 · **Atualizado:** 09/Jul/2026
Documento único de backlog + roadmap. Substitui os adendos anteriores (visualização/replays e deck
building — agora integrados aqui). Guia operacional em `CLAUDE.md`; arquitetura em `ARQUITETURA_DECISAO.md`.

---

## Status atual (09/Jul/2026)

- **Sprints 1–4, 5A, 5B e 5D (parte 2) concluídas** — muito à frente do cronograma semanal original.
- **Gate A** (submissão válida) ✅ · **Gate B** (heurística 91% vs random) ✅ · **Gate C** (rede treinada > baseline) ✅.
- Submissão atual: **par casado DECK-AGNÓSTICO 5D** (`models/policy_value.npz` + `feature_stats.npz`) —
  imitação dos líderes sob stats mistas multi-deck; val top-1 56,4%; Gate C 51% ≥ 49% do par 5B
  (preservado em `*_pre5d.npz`; clone 5A em `*_bc5a.npz`). 5,0 MiB, latência 545µs/jogada (852µs p99),
  paridade torch↔numpy ~4e-14, 0 exceções. **`deck.csv` = Abomasnow** — venceu o gauntlet JUSTO
  (Abomasnow 73% > Lucario 39% > Iono 38%) mesmo com o piloto desconfundido.
- **ATENÇÃO (par casado):** `feature_stats` e a policy são treinadas juntas — nunca promover uma sem a
  outra. Value head segue congelado/descalibrado (será calibrado no 5C).
- **Achado 5D:** o gargalo do deck não é normalização — é o **piloto**, que não executa linhas de setup
  de decks de evolução (Lucario/Dragapult/Clefairy colapsam vs random). O gauntlet mede força-de-deck
  **condicionada ao piloto**; o teto do Lucario (60,4% cabt) segue não capturado.
- **Próxima sessão:** liberar a **Fase 2 do CP4** (gauntlet expandido) — CP4a já instalou
  `seed_dragapult.csv` e `seed_clefairy.csv` (top-cut reais; Gardevoir/Jellicent saiu por ausência da
  carta-núcleo no pool). O prompt da Fase 2 (no histórico do chat) pede **adicionar 1–2 decks de baixo
  setup** (aggro de Básicos) antes de rodar, pois são o único confronto não-confundido pela fraqueza do
  piloto. Pergunta decisória: algum deck de baixo setup supera o Abomasnow? Depois: **Fase E**
  (especializar piloto no deck de maior teto) e/ou **5C** (self-play RL, causa-raiz do setup).

## Arquitetura (resumo)

Policy-net-first; **treina em torch offline, roda inferência em numpy na submissão**; MCTS/busca via
`cg.search_*` = Camada 2 opcional; LLM = ferramenta de dev (destilação de efeitos). Deck é **fixo** por
submissão (deck building = otimização offline). Detalhe em `ARQUITETURA_DECISAO.md` e `CLAUDE.md`.

---

## Backlog — Epics & Tasks

Legenda: ✅ concluído · 🔄 parcial · ⬜ a fazer · ⏸ deferido/opcional

### Epic 1 — Data Prep & Pipeline ✅
Star schema in-memory → Parquet. `dim_card`/`dim_attack`/`dim_skill`/`dim_effect` + `bridge_attack_energy`;
`CardIndex`/`EffectIndex` com lookup O(1); `reconcile.py` exit 0. Tasks 1.1–1.6 concluídas.

### Epic 2 — Wrapper do Motor ✅
- 2.1 `EnvironmentWrapper` (parse, legal_option_indices, option_summary, enrich) ✅
- 2.2 harness de self-play ✅ · 2.3 adaptador de submissão ✅ · 2.5 recorder/log parser ✅
- 2.4 interface de busca `cg.search_*` ⏸ (Camada 2 opcional — só se formos pra MCTS)

### Epic 3 — Baseline Agents ✅ (Gate B 91%)
- 3.1 `RandomAgent` + 1ª submissão ✅ · 3.2 `HeuristicAgent` v1 ✅ · 3.3 mulligan/prize ✅ · 3.5 arena ✅
- 3.4 `SearchAgent` 1-ply ⏸ (deferido — optamos por policy-net-first)

### Epic 4 — Treinamento & RL 🔄
- **4.1** Encoding de estado/ação + máscara ✅ (`ENCODING_DIM=1185`, `OPTION_DIM=154`, `MAX_OPTIONS=64` validado em replays reais: 0 overflow, 100% cobertura)
- **4.2** Harness self-play/arena ✅ (não é Gym formal, mas suficiente para o loop de RL)
- **4.3** Aprendizado da política — refinado em fases:
  - **5A** Behavioral cloning do heurístico ✅ (top-1 96%, paridade torch↔numpy OK, arena 52% vs heurístico / 94% vs random)
  - **5B** Imitação dos **líderes** (replays top do Kaggle) ✅ (val top-1 60,1%; superou o heurístico; re-treinada na 5D sob stats mistas → par 5D promovido)
  - **5C** Self-play RL fine-tune (PPO + action masking) ⬜ → calibra o value head + resolve linhas de setup
- **4.4** Ingestão de replays:
  - 4.4a download ✅ · 4.4b parser ✅ (pareamento ação↔obs corrigido + teste de regressão) · 4.4d ponte com viewer ✅
  - 4.4c warm-start por imitação (= Sprint 5B) ✅
  - 4.4e **coleta diária automatizada** ⬜ (infra Task Scheduler; ver `PROMPT_replays_diarios.md`)
- **4.5** Deck building (offline) 🔄:
  - 4.5a validador de legalidade ✅ · 4.5b sementes de arquétipo ✅ · 4.5c busca por winrate na arena ✅ (`gauntlet.py`) · 4.5d gauntlet de decks reais ✅ · 4.5e co-otimização deck↔piloto ✅ (par 5D agnóstico promovido)
  - **em curso:** expandir campo do gauntlet com meta real (CP4b) → decidir deck de submissão · **Fase E** especialização opcional
- **4.6** Empacotamento final ⬜ (submissão já válida a 5 MiB; quantização/pruning só se necessário)

### Epic 5 — Observabilidade & Battle Viewer 🔄
- 5.1 gravador (recorder) ✅ · 5.2 viewer single-file offline ✅ · 5.3 scores por jogada ✅
- 5.4 redesign estilo batalha Pokémon ⬜ (ver `PROMPT_viewer_redesign.md` + `battle_viewer_concept.html`)

---

## Roadmap (atualizado)

As sprints eram semanais no plano original, mas o Claude Code fecha uma sprint por sessão — estamos
~5 semanas adiantados. O que resta, em ordem sugerida, com folga grande até 16/Ago:

| Fase | Foco | Status |
| :-- | :-- | :-- |
| Fundação (Epics 1–3) | dados, wrapper, heurística, Gate A/B | ✅ |
| Sprint 5A | behavioral cloning do heurístico | ✅ |
| Sprint 5B | imitação dos líderes (replays top) | ✅ |
| Sprint 5D | deck building + co-otimização (Epic 4.5) | ✅ (par 5D promovido; deck Abomasnow) |
| Sprint 5D — CP4b | gauntlet expandido vs meta real → decidir deck | 🔄 próximo |
| Sprint 5C | self-play RL fine-tune (value head + setup) | ⬜ |
| Fase E | especializar piloto no deck de maior teto | ⏸ opcional pós-CP4b |
| Infra paralela | 4.4e coleta diária + 5.4 redesign do viewer | ⬜ quando conveniente |
| Polimento final | empacotamento, testes contra meta | ⬜ até 16/Ago |

**Buffer:** com a fundação + 5A prontas em ~2 dias e ~5,5 semanas até o deadline, o tempo extra vai
para a parte difícil/arriscada (RL e deck building) e para iteração.

## Gates de qualidade
- **Gate A** ✅ submissão válida pontuando.
- **Gate B** ✅ heurística > random (>65% → 91%).
- **Gate C** ✅ política treinada ≥ melhor baseline (par 5D 51% ≥ par 5B 49%, 0 exceções; será reforçado no 5C).
- **Invariantes:** `exceptions=0` em toda arena/self-play; **paridade torch↔numpy obrigatória** antes de cada submissão.

## Riscos & mitigação
- *Imitação não superar o heurístico (corpus pequeno):* ampliar dias de replay (4.4e) antes de concluir.
- *RL instável/caro em 2 vCPU:* rede pequena, avaliar por arena a cada N iters, manter melhor checkpoint; fallback = rede supervisionada.
- *Deck ilegal:* validador 4.5a antes de gastar simulação; verdade final = engine na seleção inicial.
- *Latência/memória:* `CardIndex` in-memory + inferência numpy (~0,42 ms/jogada, com folga).

## Docs relacionados
- `CLAUDE.md` — guia operacional (convenções, gotchas de domínio).
- `ARQUITETURA_DECISAO.md` — arquitetura de decisão detalhada.
- `PROMPT_viewer_redesign.md`, `PROMPT_replays_diarios.md` — prompts prontos para rodar.
- `battle_viewer_concept.html` — referência visual do redesign do viewer.
