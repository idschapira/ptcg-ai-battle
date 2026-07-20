# PTCG AI Battle — Strategy Track Journal

**Documento vivo.** Fonte de matéria-prima para o **Kaggle Writeup ≤ 2.000 palavras** da
*PTCG AI Battle Challenge — Strategy Category*. Capturamos aqui a narrativa, as hipóteses testadas,
as evidências e as figuras conforme o projeto avança; em setembro destilamos nas 2.000 palavras.
Mantido pelo "TPM/arquiteto" (chat), em paralelo ao código (Claude Code).

---

## 0. Meta / prazos / rubrica (não perder de vista)

- **Entry Deadline (aceitar as regras da Strategy):** 6/Set/2026 — AÇÃO PENDENTE.
- **Final Submission (Writeup):** 13/Set/2026. Julgamento 14/Set–11/Out.
- **Pré-requisito:** estar na Simulação (✅ estamos, v1+v2 no ladder). Time idêntico nas 2 divisões.
- **Entrega:** Kaggle Writeup (título + subtítulo + análise), **≤ 2.000 palavras**, + Media Gallery
  opcional (imagens/vídeos). Pode anexar repo de código, notebooks, links.
- **Prêmio:** $240k total — 8 finalistas × $30k; top-8 → final presencial em Tóquio.
- **Rubrica:**
  - **Model Score 70%** — clareza e justificativa da abordagem; originalidade e solidez técnica;
    **consistência sob partidas repetidas**; **não depender de estados iniciais/matchups/vantagens
    situacionais**; performance na track.
  - **Deck Score 20%** — conceito de deck claro e alinhado à estratégia; seleção/uso das cartas-chave.
  - **Report Score 10%** — estrutura lógica; uso eficaz de figuras/tabelas/gráficos.
- **Insight-chave da rubrica:** ELO meio-de-tabela NÃO impede nota alta — a Strategy premia
  profundidade/originalidade/rigor. Nosso ativo é o *processo*, não o pico de winrate.
- **Restrição:** não incluir arte/assets licenciados da Pokémon ("Pokémon Elements") na Media Gallery;
  discutir mecânica/cartas no texto é inerente e ok.

---

## 1. A espinha narrativa (arco do relatório)

Uma história de **método vencendo intuição**: cada vez que uma intuição sobre "melhor deck/piloto"
encontrou uma medição rigorosa, a medição revelou um confundidor e nos reorientou — até chegarmos a um
par (deck, piloto) fundamentado e a uma disciplina de avaliação estatística.

1. **Enquadramento & arquitetura.** Agente para track de Simulação (sandbox offline, 2 vCPU, inferência
   em ms). Decisão *policy-net-first*: treinar offline (PyTorch), rodar inferência em **numpy puro** na
   submissão (≤197,7 MiB). Racional: PTCG é informação imperfeita + estocástico + sandbox offline →
   não é AlphaZero puro; LLM/MCP só como ferramentas de DEV, nunca em runtime.
2. **Aprendizado por imitação.** 5A: clone do heurístico (top-1 96%). 5B: imitação dos **líderes** do
   Kaggle (corpus de replays) — superou o heurístico. Value head deixado congelado (honestidade: crítico
   descalibrado, seria 5C).
3. **A investigação de deck-building (contribuição intelectual central).**
   - **Hipótese inicial:** trocar o deck placeholder fraco (Mega Abomasnow, 40,2% no sim interno do
     provedor) por um deck de meta forte (Mega Lucario, 60,4%).
   - **Método:** gauntlet (round-robin de decks com piloto fixo). **Resultado que INVERTEU a hipótese:**
     com nosso piloto, o Abomasnow *ganhava*. Diagnóstico: **o gauntlet estava confundido pela
     familiaridade do piloto** — media competência-do-piloto, não força-de-deck.
   - **Desconfundir:** treinamos um **piloto deck-agnóstico** (normalização sobre corpus multi-deck +
     re-imitação) para ranquear decks sem viés. Verdade revelada: nosso piloto só executa bem decks
     *simples*; o valor do projeto estava preso na **competência geral do piloto**, não no deck. Até
     decks de baixo setup colapsavam vs um agente aleatório — o déficit era gestão de energia multi-tipo
     e montagem de qualquer linha não-trivial.
   - **O insight Crustle.** O relatório de meta argumentava que o arquétipo **Crustle (stall/LibraryOut)**
     é o de *maior eficiência para um agente simples/heurístico*: baixa ramificação ofensiva, jogo
     reativo. Ou seja, o oposto do nosso gargalo.
   - **De-risking no motor (testes de contrato):** confirmamos empiricamente que o engine honra (a) a
     prevenção de dano da habilidade do muro (específica a Pokémon *ex*), (b) o deck-out como condição
     de derrota (nossa win-condition via mill), (c) que os caps de turno/ação não empatam o stall. Achamos
     uma submissão real **rule-based que chegou a Elo 1208** com o arquétipo — prova de viabilidade e
     blueprint de lógica.
   - **Resultado:** o heurístico *genérico* já pilota o Crustle a **72,3% vs o campo**. Especializado
     (regras portadas do blueprint) → **86,8%**, e o processo **descobriu e corrigiu 2 bugs** do próprio
     agente.
4. **Rigor de avaliação (contribuição metodológica).** Ao construir o harness de A/B, provamos na fonte
   do C++ que o motor embaralha com `std::random_device` a cada shuffle → **nenhum jogo é reproduzível;
   todo winrate é uma amostra com ruído irredutível**. Consequência: sementes pareadas são impossíveis;
   a leitura honesta exige **intervalo de confiança (Wilson)** e uma barra de decisão explícita
   (PASS/HOLD/FAIL). Isso responde diretamente ao critério "consistência sob partidas repetidas".
5. **Loop de calibração interno→ladder.** Primeira validação real: o agente com os 2 bugs corrigidos
   subiu **+243,6 pontos de ELO** no ladder (v2 829,6 vs v1 586,0 na coleta de 11/Jul do tracker —
   citar sempre o CSV do tracker como fonte), confirmando que nossa métrica interna (A/B 77%) prevê o
   resultado real.
6. **Limitações honestas** (importante para o critério de robustez): posição meio-de-tabela no ladder;
   fraqueza estrutural do muro contra atacantes **não-ex** (matchup Abomasnow/Kyogre); dependência da
   mecânica de deck-out.

---

## 2. Model Score (70%) — pontos a articular

- **Racional da arquitetura** e do **pivô** policy-net → heurístico especializado: por que trocar não
  foi retirada, mas seguir a evidência (deck simples + piloto simples dominou; o teto de RL para decks
  complexos era caro/arriscado no prazo).
- **Hipóteses testadas × resultados** (montar tabela): tese de troca de deck (refutada→reformulada);
  diagnóstico do confundidor; "Crustle serve a agente simples" (confirmada); correções de bug
  (confirmadas no ladder, +249 ELO).
- **Robustez / anti-situacional:** dominância ampla no campo (6–7 matchups), imunidade estrutural a ex;
  avaliação por IC (consistência), e honestidade sobre a fraqueza não-ex.
- **Reprodutibilidade do PROCESSO** (mesmo com jogos não-reproduzíveis): toda afirmação lastreada em
  winrate medido com IC.

## 3. Deck Score (20%) — Crustle LibraryOut

- **Conceito:** controle/mill. **Crustle** (habilidade "muro": imune a dano de Pokémon *ex*) como parede
  inquebrável num meta saturado de ex; **Great Tusk** como motor de *mill* ativo (win-condition por
  deck-out); disrupção (gust/controle do ativo, disrupção de mão) + cura + stadium defensivo; energias
  especiais defensivas.
- **Alinhamento à estratégia:** minimiza ramificação de decisão ofensiva (casa com piloto rule-based) e
  **contra-ataca estruturalmente os decks dominantes (ex)**. Detalhar cartas-chave e papéis.

## 4. Report Score (10%) — figuras a produzir

- Matriz do gauntlet (deck × deck) — o confundidor e a dominância do Crustle.
- Deltas de campo v1 → v2 (por matchup) — o efeito das correções.
- Gap de ELO real v1 vs v2 no ladder (série temporal, quando o tracker rodar).
- Linha do tempo das decisões (intuição → medição → reorientação).
- **NÃO** usar arte licenciada da Pokémon na Media Gallery.

---

## 5. Ações & threads abertas (Strategy)

- [ ] **Aceitar as regras da Strategy no Kaggle até 6/Set** (Entry Deadline).
- [ ] Confirmar composição de time idêntica nas duas divisões (hoje: solo — Ilan).
- [ ] Capturar figuras/evidências conforme geradas (não deixar pra setembro).
- [ ] Distilar em Writeup ≤2.000 palavras até 13/Set.

## 6. Changelog

- **11/Jul/2026** — Documento criado. Estado do projeto: v1 e v2 no ladder (v2 829,6 > v1 586,0);
  CrustleAgent v2 é o ship; harness A/B com IC de Wilson pronto; coleta diária de replays automatizada.
  Descoberta da track Strategy e mapeamento da rubrica. Próximo: tracker de ELO + seguir afinando o
  CrustleAgent (gap até o top-20 ~230 pts).
- **11/Jul/2026** — Tracker de ELO no ar; 1º dia de dados reais confirma a previsão interna: v2 +243,6
  sobre v1 no ladder (o A/B interno de 77% se materializou na direção certa). Gap ao topo (~1254): −424.
- **11/Jul/2026 — calibração interno×real (figura/ponto forte do relatório):** primeira amostra de
  ladder da v2 (64 jogos, ~19h) = **59,4% winrate real** (38V/26D, 0 empates), com ELO subindo (829→841).
  Contraste com os ~90% do gauntlet interno **quantifica a saturação**: o campo interno inflava a
  leitura; a verdade só veio do ladder. Detalhe diagnóstico: as derrotas mais caras (−14,6/−10,3 ELO)
  foram contra oponentes de rating mais BAIXO → misplays nossos exploráveis (o corpus da caça de
  misplays). 0 empates em 64 confirma que o cap de 3.000 ações não é risco no jogo real.
- **12/Jul/2026 — fix v3 valida a tese (fecha o arco do board-wipe):** implementado o piso absoluto de
  deck (só freia thinning com deck ≤15, ou ≤30 e perdendo) + `desired_field_floor` (board-builders sobem
  a 42 quando <3 Pokémon em jogo). Validação por 3 ângulos que furam a saturação do winrate: (1) mirror
  NÃO regride (v3 48,1%, IC[44,7–51,6] contém 50% → mantém a defesa anti-self-mill); (2) vs agressivos
  95,4%→98,0% (p≈0,002); (3) **diagnóstico mecanístico direto**: taxa de board-wipe 30%→10%, derrotas
  com assinatura de wipe 26→6, board@t5 ~3,78. Figura-chave do relatório: o ciclo completo
  *observar derrota real → medir a causa → corrigir → provar mecanicamente*. Gate de campo completo
  APROVADO: v3 92,1% vs v2 89,6% no agregado (+2,5pp, p≈0,005, 4200 jogos), nenhum matchup regride.
  **Ponto elegante do relatório:** a maior melhora foi o Abomasnow/não-ex (68,0%→78,3%, +10,3pp) — a
  fraqueza estrutural que tínhamos despriorizado foi consertada DE GRAÇA pelo board-floor, porque
  board-wipe (vs agressivo) e ser KO'd por não-ex compartilham a MESMA raiz: banco vazio. Um fix
  principiado, duas fraquezas. v3 empacotada e subida ao Kaggle ao lado da v2 (A/B de ELO real em curso).
- **11/Jul/2026 — ACHADO CENTRAL (a espinha da seção de rigor do relatório):** caça de misplays nas
  derrotas reais da v2 (fidelidade de reconstrução 474/474 = 100%). Padrão: **a v2 não perde como um deck
  de mill — perde como um deck sem banco.** ~11/12 derrotas são *board-wipe* (morremos com deck cheio,
  6/6 prêmios intactos, às vezes a 1 turno de vencer por mill); 0 por self-deck-out ou cap de ações.
  **Causa medida:** a regra anti-self-mill de gatilho RELATIVO (meu deck < deck dele) — a MESMA que
  venceu o A/B do espelho por 77% — estrangula o setup: 37 supressões de itens de consistência com deck
  >30 cartas (25 nos turnos 1–4), levando a board final de 0–2 Pokémon em 11/12. **Lição-tese:** nossa
  MELHOR métrica interna (o mirror A/B) aprovou uma regra NOCIVA no jogo real — no espelho os dois lados
  se auto-estrangulavam igual, escondendo o dano; só o ladder, contra oponentes que montam board e
  atacam, revelou. Correção escopada (próxima sprint): piso ABSOLUTO de deck + piso de board (os 2
  contrapesos do kernel Elo 1208 não-portados: `desired_field_floor` + guard conservador), validada dos
  dois lados (A/B interno garante não-regressão do mirror; ELO do ladder confirma o ganho real).
- **11/Jul/2026 — insight metodológico (candidato forte ao relatório):** o gauntlet interno **saturou** —
  a v2 faz ~90% vs o nosso campo, então o campo perdeu poder de discriminar melhorias futuras. Combinado
  com o motor não-reproduzível (todo winrate é amostra), isso define nossa **hierarquia de evidência**:
  (1) ladder ELO = ground-truth lento; (2) harness A/B com IC = medição controlada mas limitada pela
  força dos oponentes internos; (3) gauntlet vs campo = saturado para o agente atual. Conclusão de
  processo: parar de "tunar às cegas" e investir em (a) observabilidade qualitativa (viewer) e (b)
  oponentes internos mais fortes, antes de mudanças que não conseguiríamos medir. Essa disciplina —
  reconhecer quando a própria métrica saturou — é o tipo de rigor que o Model Score (70%) premia.
- **13/Jul/2026 — caça de misplays rodada 2 (v3 real, 37 derrotas): o teto agora é o DECK.** Submission
  v3 = 54619473, reconciliada por 2 fontes (CLI + tracker). Amostra: 78 jogos, 41V/37D (**52,6%** real);
  ELO do dia: v3 856,8 vs v2 871,3 (v3 ainda paga as derrotas de placement de −93/−94). **Correção de
  premissa:** v3 é piloto-only (deck.csv byte-idêntico ao da v2; hash git confirmado no tarball) — o
  A/B em curso no ladder mede o FIX do piloto, não deck. Fidelidade de reconstrução 1699/1699 = **100%**
  (após consertar o classificador: a visão final *stale* do perdedor rotulava wipe real de "unknown";
  fix commitado em `episode_review.py`, + flag `--variant`). **Perfil v3 vs v2 (mesmo classificador):**
  board-wipe 83%→57%, donks t2–3 eliminados (derrota mais cedo: t7), MAS surge prize-race 19% (+24%
  ambíguo wipe/prize) — em 34/37 derrotas tomamos 0 prêmios e o oponente levou os 6 através do muro.
  Zero misplays das classes conhecidas em 1699 decisões (só empates de score). **Diagnóstico: um
  TORNO** — o muro morre cedo demais (wipes t7–17 com 26–43 cartas nossas no deck) E o mill mal
  completa (derrotas com deck adversário em 0–2 cartas: 85631888 perdeu por exatamente 1 turno).
  O piloto v3 moveu a cauda; o formato persiste com o mesmo deck → limite estrutural. Próxima alavanca:
  A/B offline de densidade de energia (8/60 é o suspeito conhecido) — testar se quebra o torno ou só
  desloca (energia salva o muro mas rouba consistência do mill?).
- **13/Jul/2026 — A/B offline de energia + PRÉ-REGISTRO do A/B de ladder do deck e10.** Offline (4200
  jogos, piloto fixo crustle-v3, mecanismo por jogo com o mesmo estimador do ladder): **e10** (−2
  Pokégear, +2 Basic {F} = 10/60) venceu o gate pareado — família muro-furado (prize-race+ambíguo)
  5,3%→2,9%/jogo (ICs de Wilson **disjuntos**), out-milled 0,7%→0,4%, derrotas-por-um-fio 37→14,
  buffer estável, winrate 90,7%→92,6% (p≈0,075; campo interno saturado). e12 (+4 {F}, −2 Xerosic)
  devolve o ganho (Xerosic importa vs controle). Ship: deck.csv := e10 com **piloto v3 congelado**
  (isola o deck; hash de main/agente no tarball == ship v3). **Critério de leitura PRÉ-REGISTRADO
  (antes de qualquer dado de ladder):** e10 vence a v3 se, sobre ≥30 derrotas colhidas (~2–4 dias de
  ladder): **(a) DECISIVO** — a fração de derrotas da família muro-furado (prize-race + final-KO
  ambíguo, classificador do episode_review) cair vs a referência da v3 = **16/37 ≈ 43%** (predição
  offline: ~metade disso), com ICs de Wilson comparados; e **(b)** ELO do e10 ≥ v3 dentro do IC no
  mesmo período de coleta. Campo interno saturado → o teste mecanístico (a) MANDA; ELO (b) é
  confirmação, não veto — exceto colapso claro (e10 abaixo da v3 além do IC com amostra ≥ à da v3
  atual). Guardrail no ladder: self-deck-out e out-milled continuam ~0 (qualquer aparição vira
  investigação antes de conclusão). Rollback documentado no commit do ship (blob b915628).

## [14/Jul] Estratégia de portfólio: mill + meta, e o moat é o piloto
Decisão de rumo pós-e10: usar os 2 slots de Final como portfólio de eixos diferentes — Crustle mill
(imune a ex) + 2º deck meta com piloto dedicado. Insight central: nosso moat não é o deck, é o PILOTO
especializado; um deck só vale o que seu piloto vale (Lucario 39–50% sob pilotos que não sabem seu
setup). Adicionar arquétipo = adicionar PILOTO, não um csv. Seleção (a validar por survey): o 2º deck
cobre a fraqueza estrutural do Crustle (corridas aggro não-ex) e é escolhido contra o que os líderes
jogam de fato — corpus de replays diário como RADAR de meta (vantagem de informação). Aggro de Básicos
lidera por custo-de-piloto; Dragapult por valor-de-meta; Lucario por último (ROI baixo). Disciplina:
decisão adiada até evidência; A/B e10 + piloto v4 têm prioridade. Recusa explícita do atalho sedutor
(self-play Lucario-vs-mill) por ser sparring contra o próprio counter — mesma classe de erro do
anti-self-mill (métrica interna enganosa). Escolher alvo com dado e nomear a armadilha antes de cair =
material de writeup.

## [19/Jul] Probe Grimmsnarl no ar: gate de persistência aprovado, V3 evictada
A BC-Grimmsnarl estava VETADA como ship imediato (bate o ship 57,5% e o zoo 75,3% ≥ barra 65%, mas
perde do Alakazam-Search 35,6%) e gated em **persistência do meta** — critério pré-declarado: share do
Grimmsnarl ≥ ~25% do topo no radar de 19–20/Jul. **Gate aprovado**: watch de 19/Jul mostra 41% no dia
mais recente (07-18, 82/200 decks), segundo dia consecutivo ≥40% (07-17: 47%) — o topo virou pra
engine de contadores e ficou. Ship como PROBE no slot livre: `submission_grimmsnarl.tar.gz` via
`build_submission --target grimmsnarl` (playbook parametrizado do Spidops; main_grimmsnarl.py
exec-safe; par CASADO bc_grimmsnarl.npz + feature_stats.npz; sentinela deck = 648). Gates: 5,0 MiB;
2 smokes OK com coleta ExIt pausada; latência mean 653µs / p99 1109µs; unittest verde; prova por hash
das 4 peças; main.py/deck.csv rastreados intocados (Final A preservado por arcname). Submissão
**54841794** ("Grimmsnarl BC — probe", 20/Jul 00:48 UTC). Eviction pelo cap de 3 ativas = **V3
(54619473)**, a última não-final; V4 (54667957) e Spidops (54791820) seguem ativas. **Guarda nova
explícita: o PRÓXIMO ship evictaria a Spidops (Final B)** — qualquer ship futuro é decisão de
portfólio, não de conveniência. Leitura da probe: ELO no watch diário (entra na coleta de 20/Jul);
a hipótese testada é a do radar — cobrir o meta que os líderes jogam AGORA vale mais que o zoo
interno sugere.
