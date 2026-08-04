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
- 12.2 GiB RAM, 2 vCPUs, 11.8 GiB HDD (números da PÁGINA da competição — **não** aparecem na
  `specification` do ambiente; só orçamento de tempo é verificável nos replays).
- Submissão = `submission.tar.gz` ≤ 197.7 MiB, com `main.py` na RAIZ + `deck.csv`.
- **Orçamento de tempo = BANCO POR EPISÓDIO, não deadline por jogada.** `actTimeout=0` e
  `observation.remainingOverageTime` começa em **600 s por agente por episódio**; estourar =
  status `TIMEOUT` = **desqualificação e derrota**, não lentidão. `runTimeout=2000 s` limita o
  episódio inteiro (os dois agentes). Verificado na `specification` de 1012/1012 replays do corpus.
  O agente LÊ `remainingOverageTime` da própria observação — é assim que se degrada com segurança.
- Ordem de grandeza real do campo (1032 replays, mediana de segundos gastos por episódio):
  Yushin Ito (top-10) **123,8 s**; Majkel1337 26,1 s; nós **2,7 s** (máx. 6,5 s). Yushin estourou
  o banco 4× e perdeu as 4 por forfeit — o teto é real e caro. Temos ~120× de folga não usada.
- Consequência de projeto: um piloto pode PENSAR (busca, rollouts). O que não pode é ficar sem
  banco. Guarda obrigatória: `src/rl_models/budget.py` (reserva intocável + degradação por
  projeção MEDIDA, que se recalibra sozinha em máquina mais lenta — assumir Kaggle ~3× o dev box).

## Estrutura do repo
- `cg/` — engine oficial vendorizado (não alterar).
- `src/ingestion/` — pipeline de dados (Polars → Parquet in-memory), `card_index.py`,
  `reconcile.py`, `build_effect_model.py`, `dim_effect_overrides.csv` (curado à mão — versionado),
  `replays_download.py`, `replays_parse.py`.
- `src/environment_wrapper/` — `wrapper.py` (parse, is_initial_selection, legal_option_indices,
  option_summary, enrich), `selfplay.py`, `arena.py`, `recorder.py`.
- `src/agent_heuristics/` — `random_agent.py`, `heuristic_agent.py` (expõe `last_scores`),
  `crustle_agent.py` (piloto do ship).
- `src/rl_models/` — `encoding.py` (encoders numpy), rede (torch, dev) + inferência (numpy).
  Camada de busca: `determinize.py` (determinização EXATA offline vs. PRESUMIDA em runtime),
  `search_core.py` (elegibilidade + rollout, compartilhado pelos dois agentes de busca),
  `search_agent.py` (dev, recebe o deck do oponente), `budget.py` (guarda do banco de 600 s),
  `opponent_estimator.py` (estima o arquétipo do oponente ao vivo), `runtime_search_agent.py`
  (o submissível: prior + estimador + guarda).
- `src/deckbuilding/archetype_rules.py` — regras de rótulo de arquétipo + mapa arquétipo→decklist
  presumida. **Fonte única**: o radar offline e o estimador de runtime rotulam com as MESMAS regras.
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
- **O engine EXPÕE o texto de regras de Trainers/Energias especiais**: `api.all_card_data()[]
  .skills[].text` cobre **203 cartas** não-Pokémon (77 Item + 27 Tool + 61 Supporter + 26 Stadium
  + 12 Energia especial). Não precisa destilar esses efeitos por LLM — é dado de primeira mão.
  (Pokémon: 218 dos 1056 têm texto de habilidade.)
- **Fatos de carta (reconciliados contra `cg.api`, não contra memória):**
  - **Crustle = id 345, `energyType=1` ({G})**, HP 150, ataque único **`Superb Scissors`
    (attackId 479), custo `[1,0,0]` = {G}{C}{C}, 120 de dano** ("damage isn't affected by any
    effects on your opponent's Active"); habilidade `Mysterious Rock Inn` = previne todo dano de
    Pokémon **{ex}** do oponente. (Existe um segundo Crustle, id 533, {F} — não é o nosso.)
    ⚠️ Este verbete já dizia "`Mini Drain` custo `[1]`" e era a própria armadilha de off-by-one
    documentada abaixo: `Mini Drain` é o attackId **480**, do **Applin (346)** — a carta seguinte.
    Conferir carta→ataque por `CardData.attacks`, nunca por vizinhança de id.
  - **Rock Fighting Energy (id 20) só protege Pokémon {F}**: "…done to the **{F} Pokémon** this
    card is attached to". Logo NÃO protege o Crustle ({G}) — só Great Tusk/Terrakion ({F}, type 6).
    **Mas isso NÃO a torna peso morto**: o Great Tusk é o nosso ativo em 52,4% das decisões reais
    (o mill exige que ele esteja no ativo), então a cláusula está VIVA na maior parte do jogo.
    Nem Rock nem Mist previnem DANO — só *efeitos*; e **contador de dano é efeito**, não dano, o
    que faz as duas anularem o `Powerful Hand` do Alakazam por completo. Matriz de prevenção
    verificada CONTRA O MOTOR em `tests/test_effect_prevention_contract.py`.
  - **Land Collapse = attackId 62, do Great Tusk, custo `[0,0]` = {C}{C}** (é o mill). Custo
    incolor ⇒ **Mist Energy (id 11, provê {C}) paga o mill** — e ainda previne efeitos de ataques.
  - `all_attack()` é **0-based por posição mas os attackId são 1-based**: `atk[i].attackId == i+1`.
    `dim_attack` já usa `offset=1`, então a chave inteira do star schema é o attackId do engine —
    indexar a lista por attackId dá a carta errada por um.

## Convenções de código
Tipado (`typing`), modular, **None-safe em todos os caminhos** (id ausente → None/zeros, nunca
`KeyError`/crash; sempre retornar jogada legal válida). Polars/numpy otimizado para memória.
Avaliar impacto de performance antes de adicionar bibliotecas pesadas de RL.

## Regras de submissão (Kaggle)
- **Cap = 2 submissões ATIVAS.** Ao subir a 3ª, a mais ANTIGA por **DATA** é evictada
  automaticamente — a eviction NÃO olha score. Planejar qual par fica no ar antes de submeter.
- Toda submissão (re)entra com **μ₀ = 600** e precisa reescalar a ladder do zero.
- **O rank do time é o do MELHOR agente ativo, não a média.** Diversidade de portfólio não pontua
  por si: só vale se elevar o teto de alguma das duas vagas.

## Verificação (obrigatória)
- `reconcile.py` deve dar exit 0 (0 missing/mismatch em cards/attacks/skills/tera) após mexer no
  modelo de dados.
- Testes de contrato + `unittest` verdes antes de commitar.
- Gates de qualidade via `arena.py`: Gate B (heurística > random, >65% — atual: 91%),
  Gate C (rede ≥ melhor baseline). `exceptions` sempre = 0.
- `build_submission` valida: main.py na raiz, tamanho < 197.7 MiB, smoke test do pacote extraído.
  Para piloto que PENSA, o smoke mede **tempo de episódio contra o banco de 600 s** (com projeção
  3×), não µs/jogada — µs/jogada não diz nada sobre o recurso que de fato limita.

## Método (erros que já pagamos)
- **A/B a ~150 jogos é RUÍDO.** O motor não é semeável (`battle_start` não expõe seed; os shuffles
  chamam `std::random_device` direto), então não existe pareamento por semente: o spread medido a
  esse N é ~10 pp. Exigir N alto + **IC de Wilson**, nunca winrate cru. `src/environment_wrapper/
  ab_test.py` faz isso; `src/analysis/search_validation.py` shardeia em processos para conseguir
  o N (processos, não threads — `cg.api` tem UM `agent_ptr` global e `search_end()` libera tudo).
- **Calibração é POR CÉLULA**, verificada contra jogo real. Um número offline só vale para o
  oponente (deck **e** piloto) que foi conferido contra a ladder — média de campo interno mente.
- **Teste de contrato tem de usar opção REAL do motor.** Um `option` sintético satisfaz qualquer
  caminho de código e esconde regra que já morreu; dirigir o engine é o que faz o teste falhar
  quando o engine muda.
- **Auditar regra contra o motor, não contra a nossa cópia dela.** Auditar só os decks que já
  passaram por `legality.py` é viés de sobrevivência: use `src/analysis/legality_audit.py --probe`,
  que quebra uma regra de cada vez e pergunta ao engine se ele se importa.

## Git
- Commits atômicos por task.
- Gitignored: `data/processed/*.parquet`, `data/raw/`, `pokemon-tcg-ai-battle/` (dados brutos do
  Kaggle — reproduzíveis via download, não redistribuir), `submission.tar.gz`, replays e datasets
  (`.npz`), `viewer/recordings/`. Versionado: código, `viewer/battle_viewer.html`,
  `dim_effect_overrides.csv`, `deck.csv`, docs.

## Status atual
Sprints 1–5D + pivô Crustle concluídos. Gates A, B (91%) e C OK. **SHIP = (deck.csv = Crustle
LibraryOut, `CrustleAgent`)** — heurístico especializado (`src/agent_heuristics/crustle_agent.py`:
anti-self-mill, Ancient→Land Collapse, muro anti-ex, resposta a não-ex), mecânica de stall
de-riscada no motor (`tests/test_crustle_stall_contract.py`). Evidência: Gate C 53% sobre o ship
anterior; média 86,8% vs campo (genérico: 72,3%); 77,5% vs Dragapult. **ROLLBACK**: ship anterior
= (Abomasnow, NetworkAgent par 5D) — deck em `data/decks/placeholder_abomasnow.csv`, par
`models/policy_value.npz`+`feature_stats.npz` segue EMPACOTADO como piloto reserva (reverter =
restaurar deck.csv + trocar o construtor no `main.py`). Par policy+stats da rede é CASADO — nunca
promover um sem o outro; value head congelado/descalibrado (critic é da 5C). O gauntlet
(`src/deckbuilding/gauntlet.py`) mede força-de-deck condicionada ao piloto. Próximo: monitorar ELO
do ship; 5C (self-play RL) e/ou pilotos especializados para outros decks (Dragapult).

**Candidato REJEITADO — busca com folha por VALUE (04/Ago, não submetido).** `ValueSearchAgent`
= v3 + beam sobre o resto do NOSSO turno com folha = value head treinado (`models/
value_crustle.npz`). Gates A e B passaram (Brier 0,1457; 6 plies por 21% do banco; 0 exceções em
7,5M nós); **Gate C reprovou: −0,28pp, IC95 [−3,15, +2,59]** vs o ship no campo dos clones,
N≈2.000/braço. **Não é falta de poder — o efeito é zero.** O que a rodada PROVOU e vale reusar:
(a) a dependência do modelo de oponente foi de fato removida (clones e campo corrigido dão o mesmo
número — a ladder de fidelidade de 29/Jul não se reproduz); (b) o head NÃO é ruído — a sonda de
inversão custa −34pp, então ele tem sinal grande e com o sinal certo; (c) **o gargalo é que o v3 já
é o juiz mais fino na faixa contestada** — juízo grosso (AUC 0,74) derrubando juízo fino em 55–75%
das decisões custa alguns pp, e a margem converte a perda em empate, não em ganho. Não repetir sem
Expert Iteration (treinar o head nas posições que a BUSCA visita — as folhas de busca estão fora da
distribuição de treino). Ver STRATEGY_JOURNAL [04/Ago].

**Candidato em avaliação — `search_crustle` (NÃO submetido).** Mesmo deck do Final A, piloto =
`RuntimeSearchAgent` = CrustleAgent v3 + busca rasa determinizada que só dispara com (a) leitura
confiante do arquétipo do oponente e (b) banco de tempo disponível. **O piso é o ship atual**:
todo caminho de falha cai no prior, verificado por identidade exata decisão-a-decisão
(`tests/test_runtime_search_agent.py`) e empiricamente a N alto. Build:
`python -m src.build_submission --target search_crustle`. Ver STRATEGY_JOURNAL [29/Jul].
