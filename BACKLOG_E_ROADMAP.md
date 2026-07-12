# PTCG AI Battle — Backlog & Roadmap (consolidado)

**Deadlines:** Simulação 16/Ago/2026 · **Strategy** (relatório): entry até 6/Set, submissão 13/Set ·
**Atualizado:** 11/Jul/2026
Documento único de backlog + roadmap. Substitui os adendos anteriores (visualização/replays e deck
building — agora integrados aqui). Guia operacional em `CLAUDE.md`; arquitetura em `ARQUITETURA_DECISAO.md`;
relatório da Strategy em `STRATEGY_JOURNAL.md`.

---

## Status atual (11/Jul/2026)

- **Sprints 1–4, 5A, 5B, 5D, pivô Crustle e tuning v2 concluídos** — muito à frente do cronograma.
- **Duas competições conectadas:** Simulação (agente por ELO) + **Strategy** (relatório escrito;
  entrar na Strategy exige estar na Simulação — ver Epic 6 e `STRATEGY_JOURNAL.md`).
- **Gate A** ✅ · **Gate B** (91% vs random) ✅ · **Gate C** ✅ (re-rodado a cada promoção).
- **SHIP ATUAL: (deck.csv = Crustle LibraryOut, `CrustleAgent` variant="v2")** — heurístico especializado.
  A **v2 = v1 + correção de 2 bugs** (v1 preferia o ataque de DANO ao de MILL num deck de mill; o handler
  de descarte sabotava os próprios picks do Explorer's Guidance) + regras de controle portadas do kernel
  Elo 1208 (gust-to-trap, timing de Xerosic, Colress→Neutralization Zone, pivô proativo, heal cedo).
  A/B interno v2×v1 = **77%**; média vs campo **90,7%** (v1 fazia 86,8%); +18,7pp vs Dragapult.
- **No ladder (múltiplas submissões permitidas): v1 e v2 rodando lado a lado.** ELO real (11/jul, ~19h de
  v2): **v2 ≈ 829→841 > v1 ≈ 586**; **winrate real 59,4%** (38V/26D, 0 empates); topo do leaderboard
  ~1254 (gap ~−420). O A/B interno (77%) confirmou-se no real (+243 ELO da v2 sobre a v1).
- **⚠️ O gauntlet interno SATUROU** (v2 ~90% vs nosso campo) → melhorias já não são mensuráveis contra
  nossos decks; **o sinal confiável agora é o ELO do ladder** (lento). O motor é NÃO-reproduzível
  (`std::random_device` por shuffle) → todo winrate é uma amostra; leitura honesta = IC de Wilson.
- **🎯 ACHADO CENTRAL (caça de misplays nas derrotas reais, fidelidade 474/474):** ~11/12 derrotas são
  **board-wipe** (morremos com deck cheio, 6/6 prêmios, sem banco), NÃO por mill/self-deck-out. Causa
  medida: a regra anti-self-mill de **gatilho relativo** (a que venceu o A/B do mirror) **estrangula o
  setup early** (37 supressões de itens c/ deck >30; board final 0–2 em 11/12). **PRÓXIMO FIX (v3):**
  trocar por piso ABSOLUTO de deck + piso de BOARD (contrapesos do kernel Elo 1208), validado por A/B
  (não regredir o mirror) + série de ELO (confirmar o ganho real).
- **ROLLBACK:** v1 acessível via flag `variant`; ship pré-Crustle = (Abomasnow, NetworkAgent 5D) segue
  empacotado (`placeholder_abomasnow.csv` + `models/policy_value.npz`+`feature_stats.npz`; `*_pre5d`,
  `*_bc5a`). Reverter = 1 linha no `main.py`.
- **Infra de avaliação/observabilidade (nova):** harness A/B com IC de Wilson + verdicto PASS/HOLD/FAIL
  (`ab_test.py`); coleta diária de replays + tracker diário de ELO (ambos no Task Scheduler); fetch dos
  NOSSOS episódios (`fetch_my_episodes.py`) + pipeline de review determinístico (`episode_review.py`).
- **Próximo:** aplicar o fix do board-floor (v3) + validar (A/B + ELO); manter o `STRATEGY_JOURNAL.md`;
  **aceitar as regras da Strategy até 6/Set**; 5C (self-play RL / critic) como aposta de teto alto.

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
  - 4.4e **coleta diária automatizada** ✅ (`scripts/daily_replays.sh` + Task Scheduler; venv dedicado
    `.venv` + auth Kaggle; corpus cresce ~10k pares/dia; bug do índice cacheado corrigido)
- **4.5** Deck building (offline) ✅:
  - 4.5a validador ✅ · 4.5b sementes ✅ · 4.5c busca na arena ✅ (`gauntlet.py`) · 4.5d gauntlet de decks reais ✅ · 4.5e co-otimização deck↔piloto ✅
  - CP4b (campo expandido vs meta real) ✅ → **conclusão: força-de-deck é condicionada ao piloto; pivô
    para o Crustle** (arquétipo stall/LibraryOut, o mais heurística-amigável do relatório de meta)
- **4.6** Empacotamento final ✅ (submissão 5,0 MiB; smoke fim-a-fim nos 2 modos — módulo + exec-sem-
  `__file__` fiel ao loader do Kaggle; guard que aborta build se o piloto empacotado não for o esperado)

### Epic 5 — Observabilidade & Avaliação 🔄
- 5.1 gravador (recorder) ✅ · 5.2 viewer single-file offline ✅ · 5.3 scores por jogada ✅
- 5.4 redesign estilo batalha Pokémon ⏸ (polimento opcional; ver `PROMPT_viewer_redesign.md`)
- 5.5 **harness A/B** ✅ (`ab_test.py`: modos agente/deck/campo, IC de Wilson, verdicto PASS/HOLD/FAIL;
  CRN impossível — motor não-reproduzível, provado na fonte)
- 5.6 **tracker de ELO** ✅ (`scripts/track_elo.sh` + `elo_report.py`; série por submissão via API Kaggle)
- 5.7 **caça de misplays** ✅ (`fetch_my_episodes.py` puxa nossos episódios; `episode_review.py`
  reconstrói decisões via CrustleAgent determinístico; achou o board-wipe como derrota nº1)

### Epic 6 — Strategy Category (relatório) 🔄
Competição-irmã: Kaggle Writeup ≤2.000 palavras. Rubrica **Model 70% / Deck 20% / Report 10%** — premia
rigor/originalidade e **não** exige ELO alto. Prazo: entry 6/Set, submissão 13/Set. Prêmio 8×$30k + final
em Tóquio.
- 6.1 diário vivo `STRATEGY_JOURNAL.md` ✅ (mantido pelo chat; pré-populado com a jornada e a rubrica)
- 6.2 **aceitar as regras da Strategy no Kaggle** ⬜ (até 6/Set — AÇÃO do Ilan)
- 6.3 capturar figuras (matrizes de gauntlet, deltas v1→v2, série de ELO, timeline de decisões) ⬜
- 6.4 destilar o Writeup ≤2.000 palavras ⬜ (até 13/Set)

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
| Sprint 5D — CP4b | gauntlet expandido vs meta real → decidir deck | ✅ (Abomasnow segurou; ranking condicionado ao piloto) |
| Pivô Crustle | seeds reais + de-risk motor + `CrustleAgent` v1 + troca de ship | ✅ |
| Tuning v2 | correção de 2 bugs + regras de controle (kernel) + re-ship | ✅ (A/B 77%; ladder v2 +243 > v1) |
| Infra de avaliação | harness A/B (IC Wilson) + coleta diária + tracker de ELO | ✅ |
| Caça de misplays | fetch dos nossos episódios + review determinístico | ✅ (achado: board-wipe = derrota nº1) |
| **v3 (próximo)** | fix do estrangulamento de setup: piso de deck + piso de board | ⬜ ← **PRÓXIMO** |
| Strategy (Epic 6) | aceitar regras (6/Set) + relatório ≤2000 palavras (13/Set) | 🔄 diário vivo em curso |
| Sprint 5C | self-play RL fine-tune (value head + teto) | ⏸ aposta de teto alto |
| Fase E | pilotos especializados p/ outros decks (Dragapult) | ⏸ se o ELO estagnar |
| Viewer 5.4 | redesign estilo batalha | ⏸ polimento opcional |

**Estado:** ship no ar medindo ELO (59,4% real, convergindo); infra de coleta/avaliação toda automatizada;
próximo lever de dev = v3 (board-floor). ~5 semanas até 16/Ago (Simulação) e ~9 até 13/Set (Strategy).

## Gates de qualidade
- **Gate A** ✅ submissão válida pontuando.
- **Gate B** ✅ heurística > random (>65% → 91%).
- **Gate C** ✅ candidato ≥ melhor baseline a cada promoção (par 5D 51% ≥ par 5B; ship Crustle 53% ≥
  par 5D + campo 86,8% ≥ 72,3% do genérico, 0 exceções; será reforçado no 5C).
- **Invariantes:** `exceptions=0` em toda arena/self-play; **paridade torch↔numpy obrigatória** antes de cada submissão.

## Riscos & mitigação
- *Métrica interna saturada:* o gauntlet não distingue mais melhorias (v2 ~90% vs campo). Sinal confiável
  = ELO do ladder (lento) + A/B no mirror p/ não-regressão. Considerar oponentes internos mais fortes.
- *Fraqueza atual conhecida (board-wipe):* a v2 perde ~11/12 por não montar banco no early (regra
  anti-self-mill sobrecorrige). Fix v3 = piso de deck + piso de board. É a prioridade de dev.
- *Motor não-reproduzível:* todo winrate é amostra (sem seeding); usar IC de Wilson, nunca ler snapshot único.
- *RL instável/caro em 2 vCPU (5C):* rede pequena, avaliar por arena a cada N iters, manter melhor checkpoint.
- *Latência/memória:* `CardIndex` in-memory; CrustleAgent ~160µs/jogada (com folga).

## Docs relacionados
- `CLAUDE.md` — guia operacional (convenções, gotchas de domínio).
- `ARQUITETURA_DECISAO.md` — arquitetura de decisão detalhada.
- `STRATEGY_JOURNAL.md` — diário vivo / fonte do relatório da Strategy (Epic 6).
- `battle_viewer_concept.html`, `PROMPT_viewer_redesign.md` — referência do redesign do viewer (5.4, opcional).
