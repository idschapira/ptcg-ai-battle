# PTCG AI Battle — Backlog & Roadmap (consolidado)

**Deadline (categoria Simulação):** 16 de Agosto de 2026 · **Atualizado:** 08/Jul/2026
Documento único de backlog + roadmap. Substitui os adendos anteriores (visualização/replays e deck
building — agora integrados aqui). Guia operacional em `CLAUDE.md`; arquitetura em `ARQUITETURA_DECISAO.md`.

---

## Status atual (08/Jul/2026)

- **Sprints 1–4 e 5A concluídas** — muito à frente do cronograma semanal original.
- **Gate A** (submissão válida) ✅ · **Gate B** (heurística 91% vs random) ✅ · **Gate C** (rede > baseline) ⬜ pendente.
- Submissão atual: `NetworkAgent` (clone do heurístico, top-1 96%), 5 MiB, inferência numpy ~0,42 ms/jogada, torch fora do runtime.
- **Próximo:** Sprint 5B — imitação dos líderes.

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
  - **5B** Imitação dos **líderes** (replays top do Kaggle) ⬜ ← **PRÓXIMO**
  - **5C** Self-play RL fine-tune (PPO + action masking) ⬜ → **Gate C**
- **4.4** Ingestão de replays:
  - 4.4a download ✅ · 4.4b parser ✅ (pareamento ação↔obs corrigido + teste de regressão) · 4.4d ponte com viewer ✅
  - 4.4c warm-start por imitação (= Sprint 5B) ⬜
  - 4.4e **coleta diária automatizada** ⬜ (infra Task Scheduler; ver `PROMPT_replays_diarios.md`)
- **4.5** Deck building (offline) ⬜:
  - 4.5a validador de legalidade · 4.5b sementes de arquétipo (LLM) · 4.5c busca por winrate na arena · 4.5d gauntlet de decks reais (usa replays) · 4.5e co-otimização deck↔piloto
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
| Sprint 5B | imitação dos líderes (replays top) | ⬜ próximo |
| Sprint 5C | self-play RL fine-tune → **Gate C** | ⬜ |
| Sprint 5D | deck building (Epic 4.5) | ⬜ |
| Infra paralela | 4.4e coleta diária + 5.4 redesign do viewer | ⬜ quando conveniente |
| Polimento final | co-otimização deck↔piloto, empacotamento, testes contra meta | ⬜ até 16/Ago |

**Buffer:** com a fundação + 5A prontas em ~2 dias e ~5,5 semanas até o deadline, o tempo extra vai
para a parte difícil/arriscada (RL e deck building) e para iteração.

## Gates de qualidade
- **Gate A** ✅ submissão válida pontuando.
- **Gate B** ✅ heurística > random (>65% → 91%).
- **Gate C** ⬜ política treinada ≥ melhor baseline (via arena; fallback = melhor rede supervisionada).
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
