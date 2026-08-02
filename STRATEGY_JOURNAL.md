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

## [29/Jul] A premissa mais cara do projeto era um comentário no CLAUDE.md
O `CLAUDE.md` dizia "latência: milissegundos por jogada" desde a Sprint 1. Era falso, e custou meses
de espaço de projeto. A `specification` que vem dentro de todo replay diz outra coisa, e diz igual em
**1012/1012** episódios do corpus: `actTimeout=0` (não existe deadline por jogada) e
`observation.remainingOverageTime` começa em **600 s por agente por episódio** — um BANCO. O agente lê
esse campo na própria observação. Medindo o que o campo gasta de fato: **Yushin Ito (top-10) gasta
123,8 s de mediana por episódio; nós gastávamos 2,7 s** (máx. 6,5 s). Estávamos usando **0,45%** de um
recurso que o líder usa a 21% — e otimizando microssegundos.

A régua do risco também estava no corpus: `statuses` registra 4 `TIMEOUT`, todos do Yushin, todos
derrota por forfeit (`rewards: [1, None]`). Estourar o banco não é lentidão, é desqualificação. Isso
define a assimetria que governa todo o desenho: **gastar de menos custa fração de uma decisão; gastar
de mais custa o jogo inteiro.**

**O bloqueador real não era tempo, era informação.** `cg.search_begin` exige as zonas ocultas do
oponente, o que exige a decklist dele — que offline nós conhecemos e em runtime não. A saída foi um
estimador que rotula o arquétipo pelas cartas que o oponente revela (mesmas regras do radar de meta,
agora com fonte única em `deckbuilding/archetype_rules.py`) e entrega a decklist de consenso daquele
arquétipo como **hipótese**. Duas medidas de confiança, e a segunda é a que importa: `containment` =
que fração das cartas JÁ VISTAS a lista presumida consegue explicar. É ela que percebe um rótulo certo
sobre uma lista errada. Medido nos replays reais: **271.054 decisões, precisão 100% quando confiante**,
cobertura subindo de 9,8% (t1–2) para 99,6% (t15+). Mantém 100% mesmo incluindo os decks cujo rótulo
final é "unknown" (276.255 decisões) — o gate de containment se cala sozinho quando não reconhece.
Limite honesto: o ground truth vem das mesmas regras sobre a observação completa, então isso mede
convergência, não leitura de mente; e um rótulo certo não garante que os 60 presumidos sejam os 60
reais — daí a determinização de runtime reconciliar a diferença em vez de fingir que ela não existe.

**Perfil de risco por construção, não por flag.** Todo caminho que não seja "estimativa confiante E
banco disponível E decisão elegível E determinização fechou" termina no prior, que é o CrustleAgent v3
— o ship atual. O piso do agente novo é a coisa que ele substitui. Isso é testado das duas pontas:
identidade exata decisão-a-decisão contra um CrustleAgent v3 puro num jogo real do motor, e
empiricamente a 396 jogos (51,3%, IC95 [46,4%, 56,1%] — cara-ou-coroa, como tem que ser).

**A guarda de banco não tem constante calibrada nesta máquina.** Ela mede o próprio custo de rollout
(EWMA) e recusa qualquer configuração cuja projeção invada a reserva de 150 s. Numa máquina 3× mais
lenta os rollouts medem 3× mais caro e ela degrada sozinha — que é a única forma honesta de escrever
isso sem poder rodar no Kaggle antes de submeter.

**Higiene de premissas (o resto do que estava errado ou faltando).** O engine EXPÕE o texto de regras
de Trainers e energias especiais — `all_card_data()[].skills[].text`, **203 cartas** não-Pokémon;
nunca precisou de destilação por LLM. `all_attack()` é 0-based por posição mas os `attackId` são
1-based (`atk[i].attackId == i+1`): indexar pela id dá a carta errada por um. E os fatos do nosso
próprio deck: **Crustle (345) é {G}**, então **Rock Fighting Energy não o protege** — o texto diz
"done to the **{F} Pokémon**", que são Great Tusk e Terrakion; **Land Collapse custa {C}{C}**, e é por
isso que **Mist Energy paga o mill**.

**Onde o banco deveria ir depois.** Censo de 40 jogos: das ~55 seleções nossas por jogo, 17,6% são
inelegíveis, 16,7% triviais, 42,1% dominadas (o prior já decidiu) e **23,6% contestadas** — ~13 por
jogo. Concentrar o mesmo banco só nas contestadas compraria **~2,8×** os rollouts em cada uma, sem
nenhuma superfície de falha nova. Paralelismo tem transporte barato (observação = 3,4 KiB, round-trip
de pool de 2 processos = 1,0 ms contra ~86 ms/rollout) mas teto de 2× num box de 2 vCPU e adiciona
ciclo de vida de worker ao caminho crítico de um agente que é desqualificado se travar. Ordem
recomendada: alocação adaptativa primeiro, paralelismo só se ela não bastar.

## [29/Jul] A busca perde para o proprio prior — e o metodo funcionou
Resultado decisivo, e negativo. `search:crustle` contra `CrustleAgent v3` puro, MESMO deck,
N=600, assentos alternados: **44,8%, IC95 [40,9%, 48,8%], p=0,0114 — FAIL.** O IC exclui 50%: nao
e a banda de ~10pp de ruido que este motor produz a N baixo. Rodou limpo — 1726 buscas, 22.076
rollouts, **0 excecoes**, 0 rollouts capados — e com a estimativa de deck essencialmente perfeita
(no espelho o rotulo e o nosso proprio deck). Ela trocou a resposta do prior em **387 de 1726**
decisoes, e essas trocas foram liquidamente ruins.

**Ninguem tinha medido isso.** A busca era "Camada 2 opcional" desde a decisao de arquitetura, e a
intuicao de que buscar > nao buscar nunca passou por um A/B a N alto contra o proprio prior. Passou
agora, e reprovou. Este e o mesmo padrao do resto do projeto — intuicao encontra medicao, medicao
ganha — so que desta vez a vitima foi uma premissa nossa de arquitetura, nao um deck.

**Duas causas candidatas, ambas instrumentadas.**

*A — modelo de oponente.* Os rollouts jogam o lado do oponente com um HeuristicAgent generico, mas
o oponente real daquele A/B e o CrustleAgent v3. Num espelho de stall/mill isso precifica errado
exatamente a corrida de que o matchup depende. Testado (`opponent_pilot="match"`, N=600):
**48,3%, IC95 [44,4%, 52,3%], p=0,41 — HOLD**. Ou seja: com o oponente modelado direito a busca
deixa de ser significativamente pior. Mas **a melhora de +3,5pp NAO e estatisticamente
estabelecida** (z=1,22, p=0,22, ICs se sobrepondo) e o estimador pontual segue abaixo de 50%. O
honesto e: A era provavelmente PARTE do problema, e corrigi-la compra empate, nao vantagem.

*B — maldicao do vencedor (optimizer's curse).* A 4x4 cada candidato vale a media de QUATRO
rollouts Bernoulli. Tomar o argmax sobre quatro estimativas dessas seleciona em boa medida quem
teve sorte, e a estimativa do vencedor e enviesada pra cima pela propria selecao. Sobrepor um prior
bem afinado com base em quatro amostras e uma boa receita para transformar uma politica forte numa
ruidosa. Sintoma consistente: com o modelo de oponente corrigido as trocas SUBIRAM (387 -> 548) e
ainda assim nao venceu. Teste pronto: `override_margin` exige que a busca ganhe por margem antes de
ser obedecida (presets `effect-margin`, `effect-both`).

**Nao se compra HOLD a esse preco.** Mesmo no melhor caso medido, o custo e de 126,4 s de episodio
no pior jogo -> 379 s na projecao 3x = **63% do banco de 600 s**. Gastar 63% do orcamento cuja
exaustao e DESQUALIFICACAO para comprar um empate estatistico e um trade ruim em qualquer leitura.

**O que sobrevive, e vale independente do ship.** A correcao da premissa de 600 s/episodio; o
estimador de arquetipo (100% de precisao quando confiante em 271k decisoes reais); a guarda de
banco com projecao auto-calibrante; e o piso provado nas duas pontas (identidade exata decisao-a-
decisao com o ship, e 51,3% [46,4%, 56,1%] a N=396). O piso e o ponto: um candidato que reprova
pode ser desligado sem custo, porque o que ele degrada para e exatamente o que ja esta no ar.

**Nota de metodo, cara de novo.** Tres dos testes novos eram flaky (falhavam ~1 em 4) por afirmarem
sobre propriedades emergentes de UM jogo — o motor nao e semeavel, entao comprimento de partida e
aleatorio. Pior: o da guarda de banco chaveava no passo do LOOP, entao um jogo curto terminava
antes de drenar o banco e o teste passava **sem testar nada**. Vacuo, nao so ruidoso. Corrigidos
para acumular sobre jogos e para verificar que a condicao que eles dependem de fato ocorreu.

## [29/Jul] O ganho da busca era o nosso próprio modelo de oponente se olhando no espelho
A rodada decisiva. A busca tinha mostrado **+14,7pp vs Grimmsnarl (p=0,0003)** — a célula que vale,
41–47% do topo. Só que o oponente daquele teste era o `HeuristicAgent`, que é **exatamente** a
política que os rollouts usam para modelar o adversário. Modelo perfeito de graça. Repetimos a
mesma célula (N≈396/braço, prior e busca contra o MESMO oponente) trocando só o piloto do
adversário:

| oponente | relação com o modelo dos rollouts | prior → busca | efeito | p |
|---|---|---|---|---|
| HeuristicAgent | **é** o modelo, exato | 46,2% → 60,9% | **+14,7pp** | 0,0003 |
| Módulo paramétrico | `ParametricHeuristicAgent(HeuristicAgent)` = modelo + regras | 51,5% → 57,8% | +6,3pp | 0,074 |
| BC-Grimmsnarl | rede treinada por BC, nada em comum | 51,6% → 49,1% | **−2,5pp** | 0,48 |

**O efeito é monótono na fidelidade do modelo, e some quando ela some.** Não é "a busca funciona";
é "a busca funciona contra quem ela já sabe simular". O caso intermediário não é coincidência: o
módulo paramétrico literalmente herda de `HeuristicAgent`, então é o modelo dos rollouts com uma
camada de regras por cima. Três pontos, uma variável, ordem certa.

Na ladder o adversário é agente de política desconhecida — muito mais perto do BC do que da nossa
heurística. **A expectativa honesta é a faixa do BC: entre nulo e ligeiramente negativo.**

**O horizonte não explica nada** (a outra hipótese). Profundidade mediana de rollout medida:
espelho Crustle **58** seleções, Grimmsnarl **106**, Alakazam **53**. O espelho é o mais RASO e é
onde a busca falha; o Grimmsnarl é quase 2× mais fundo e é onde ela funcionou. Profundidade não
correlaciona com efeito; fidelidade correlaciona. Uma explicação única cobre espelho e campo.

**Alocação adaptativa resolve o custo — e só o custo.** Pular as decisões dominadas (a busca cai de
57% para 19–20% das decisões) leva o pior episódio de 78–80% para **46–49% do banco** na projeção
3×. Bate a meta. Mas o ganho cai junto: −8,8pp vs Heuristic (p=0,019), −5,8pp vs módulo. Contra o
BC não muda nada (−2,5 → −2,2). **Confissão de escopo:** implementei a metade "pular o dominado",
não a metade "reinvestir no contestado". E reinvestir não cabe: a 4×4-no-contestado já custa 46%,
então 4×8 volta para ~92% e 3×8 para ~69% — ambos furam a meta de ≤50%. Dentro do orçamento não
existe profundidade extra para comprar. E não compraria nada mesmo: contra o BC o problema é
fidelidade de modelo, não número de amostras.

**Recomendação: NÃO shipar.** O único ganho grande não sobreviveu ao teste de robustez; o que
sobrevive é indistinguível de zero e custa metade de um banco cujo estouro é desqualificação. O que
mudaria a resposta não é mais compute — é **modelo de oponente por arquétipo** nos rollouts (temos
o estimador que diz qual arquétipo é; falta o piloto especializado para cada um). Enquanto o
rollout modelar todo mundo como heurística genérica, a busca só vai render contra quem joga como
heurística genérica.

**O que fica, independente do veredito.** A guarda de banco recalibrada (o degrau mais rico saiu de
400s para 500s — antes ela nunca engatava, agora engata: `tiers {4x4: 1193, 3x2: 508}`); o
estimador de arquétipo; a alocação adaptativa; e o piso provado nas duas pontas. Um candidato que
reprova sai sem custo porque degrada exatamente para o que já está no ar.

**Fechamento (célula final).** Espelho com alocação adaptativa: **50,3%, IC95 [45,3%, 55,2%]**,
N=396 — exatamente o nível do prior, contra 44,8% da busca cheia. Coerente e sem mistério: a
adaptativa busca em só 16% das decisões no espelho, então joga quase o prior. Confirma o desenho —
quando a busca não tem o que agregar, o filtro a apaga em vez de deixá-la sangrar. Custo do
espelho cai para 47,9s de pior episódio = **24% do banco**. Todas as células: 0 exceptions.

## [30/Jul] Variantes de deck do Crustle: a premissa do "peso morto" estava invertida
Hipótese a testar: as 4 Rock Fighting são peso morto (a Crustle é {G}, a cláusula não a protege),
então os slots comprariam mais como CORPOS — cada corpo ≈ 1 turno, e em ~28% das derrotas o
oponente estava a ≤5 cartas do deck-out. **Resultado: nenhuma variante merece ship, e o motivo é
que a premissa media a coisa errada.**

**A medição que era o portão (e reprovou a premissa).** Baixei os 222 episódios reais da submissão
54917180 — cujo deck é byte-idêntico ao `deck.csv` — e auditei 225 jogos com o piloto do ship
(`src/analysis/deadweight_audit.py`). Duas armadilhas de amostra apanhadas antes de qualquer conta:
`viewer/episodes/` mistura TODAS as nossas submissões (73 jogos de Grimmsnarl, 62 de Abomasnow), e
os dois agentes publicam um `current` por passo com visões divergentes — somar as duas inventa
oscilação de HP e KOs fantasma (contei 1445 KOs de um Pokémon antes de pinar a perspectiva). Com
isso corrigido:

| fator | medido nos 225 jogos reais |
|---|---|
| ativo é {F} (cláusula VIVA) | **57,8% das decisões**; Great Tusk sozinho **52,4%** |
| ativo é {F} E sob ameaça | 56,5% |
| dano absorvido pelo Great Tusk | 47,8% do total, 314 KOs |
| anexações de Rock em host {F} | 331/455 = **72,7%** |

**O mill exige que o Great Tusk esteja no ativo, e ele é {F}.** A premissa raciocinou sobre a
Crustle (que é {G}) e concluiu que a cláusula nunca paga; na prática ela está viva na maior parte
do jogo. Verificado CONTRA O MOTOR, não contra o texto da carta
(`tests/test_effect_prevention_contract.py`, sonda = `Painful Memories` do Uxie, 2 contadores,
dano 0, leitura binária): Mist previne em host {F} e em host {G}; Rock previne em {F} e **não**
previne em {G}; energia comum não previne nada (controle não-vácuo).

**O que a auditoria achou de fato — e é maior que os slots.** `Powerful Hand` do Alakazam é
**89,2% dos ataques do oponente na célula real** (365/409 em 60 jogos). Ele *coloca contadores de
dano*, e contador é EFEITO, não dano — logo Mist/Rock o anulam por inteiro. Só que dos 416 efeitos
preveníveis recebidos, **69% chegaram SEM prevenção**. Isso não é problema de slot, é de piloto
(prioridade de anexação). A célula Alakazam é 26,7% dos jogos reais e a nossa pior (35% real).

**As variantes, medidas de qualquer forma.** Três listas legais (`legality.py` E `battle_start` do
motor), motor de mill intacto nas três (4 Great Tusk + 4 Explorer's Guidance — que é o **único
Supporter Ancient de todo o pool**, então o mill de 4 já estava no teto). N=600 por (arm, célula),
8 células, 0 exceptions, Newcombe para a diferença, quebra por assento:

| célula (share real) | SHIP | V1 corpos | V2 mill | V3 corpos agr. |
|---|---|---|---|---|
| alakazam-heur (26,7%) | 83,5% | +1,2 [−3,0,+5,3] | −0,7 | −0,3 |
| alakazam-BC (mesma célula, piloto honesto) | 89,7% | −1,0 | −1,2 | −3,7 [−7,4,+0,0] |
| grimmsnarl-BC (16,0%) | 45,4% | −0,2 | −3,7 | **−12,6 [−18,0,−7,1]** |
| lucario (16,0%) | 96,8% | +0,8 | −2,3 [−4,7,−0,0] | +0,3 |
| starmie (4,0%) | 25,8% | −2,7 | **−7,8 [−12,5,−3,2]** | **−9,0 [−13,6,−4,4]** |
| espelho (1,3%) | 51,3% | **−6,7 [−12,3,−1,0]** | **−6,8** | **−11,7 [−17,2,−6,0]** |
| agregado ponderado | 75,5% | 75,9% | 73,3% | 71,8% |

Nenhuma célula melhora significativamente em nenhuma variante. V1 é empate; V2 e V3 regridem.

**O mecanismo MOVEU — e é isso que explica o veredito.** O turno da morte subiu (alakazam-BC 15 →
17 → 17 → **19**; kangaskhan 15 → **30**) e a fração de derrotas com o oponente a ≤5 cartas subiu
(32,3% → 36,8% → 40,6% → **44,0%**). Corpos compram turnos, exatamente como a tese dizia. Só que
**os turnos são pagos com a gasolina do mill**: as três variantes trocam energia por não-energia
(10 → 8 → 8 → **6**) e `Land Collapse` custa {C}{C}. Onde o jogo é longo o dano aparece invertido —
no Grimmsnarl o turno da morte CAIU (28 → 25) e o oponente chegou MENOS perto do deck-out (36,7% →
25,8%), com os auto-deck-outs subindo. Os slots não pagavam só proteção; pagavam combustível.

**O controle que separa as duas coisas (V4).** −2 Rock, +2 Basic {F}: energia de volta em 10,
corpos em 13, única variável = a cláusula. Resultado: tudo chato **menos** alakazam-BC, que cai
**−6,0pp [−9,8, −2,2]**. Ou seja: as 2 Rock removidas valiam ~6pp na célula que mais pesa, e o
prejuízo de V2/V3 era energia por cima disso. Fidelidade da célula conferida antes de acreditar:
os dois pilotos internos disparam `Powerful Hand` em 78% / 72% dos seus ataques contra 89,2% do
real — o nível absoluto do winrate segue inflado (81–90% interno vs 35% real), mas **a mecânica
sob teste é exercitada no regime certo**, que é o que a diferença precisa.

**Veredito: NÃO shipar nenhuma variante.** `deck.csv` intocado, verificado idêntico ao HEAD. O que
sobrevive e vale independentemente: a auditoria de peso morto sobre jogos reais, o teste de
contrato da matriz de prevenção, o `observer` de mecanismo em `play_one_game` (mede turno da morte
e proximidade de deck-out sem reimplementar o loop), e a correção de um fato de carta no CLAUDE.md
que tinha caído na própria armadilha de off-by-one que ele documenta (`Mini Drain` é do Applin 346,
não da Crustle 345).

**Onde apontar o próximo esforço, com número.** Não é o deck: é a anexação. 69% dos efeitos
preveníveis chegaram sem prevenção, e a Mist — que protege QUALQUER host, inclusive a Crustle {G} —
é usada 2,02×/jogo, praticamente igual à Rock (2,02) e distribuída quase igual (70,7% no Great
Tusk). Priorizar Mist na Crustle e garantir prevenção no ativo sob ameaça é mudança de PILOTO,
custa 0 slots, e ataca 26,7% do campo pela porta certa.

**Lacuna declarada.** Archaludon é 14,2% dos jogos reais (ganhamos 29/3) e **não tem decklist em
`data/decks/`** — a célula ficou sem medir. Vale minerar a lista consenso dos 32 episódios reais
antes do próximo teste de deck.

## [31/Jul] Prevenção é ECONOMIA, e a economia está fechada por regra; Archaludon fechada
Duas frentes, ambas offline, nenhuma shipa. `deck.csv` intocado.

### (A) Política ou economia? — 73,3% ECONOMIA, e não há alavanca
Dos 277 efeitos preveníveis que chegaram sem prevenção (215 jogos reais, filtro de submissão
54917180 + sentinela de deck + perspectiva fixa no nosso assento):

| classe | n | share |
|---|---|---|
| INEVITÁVEL — nenhuma protetora elegível na mão | 203 | **73,3%** |
| EVITÁVEL — tinha na mão, anexou em outro corpo | 39 | 14,1% |
| EVITÁVEL — anexou no alvo, mas energia SEM cobertura | 35 | 12,6% |

Reposicionamento (cobertura estacionada noutro corpo **e** switch/retreat legal) recupera só
**10/203 = 4,9%** dos inevitáveis. Teto do que política pode endereçar: **~30%**.

**Por que a economia está fechada — e isto é o achado.** 61% das vítimas são {G} (Crustle 41,9% +
Dwebble 14,1%), e varrendo o pool inteiro por "prevent all effects of attacks", a **Mist é a ÚNICA
carta que cobre um ativo {G}** — e já está no teto de 4 cópias. As alternativas não servem:
Battle Cage (Stadium) só cobre o BANCO; Acerola's Mischief só contra {ex} e só com o oponente a
≤2 prêmios; Antique Cover Fossil protege apenas a si mesma (60 HP) e já foi testada e regrediu.
Ou seja: não existe deck legal com mais cobertura para o muro. A alavanca de deck não "falhou por
pouco" — ela **não existe**. Some-se a isso que as variantes V1–V4 já mediram nulo/negativo a
N=600/célula, e a frente de deck encerra.

**A regra candidata não tem volume.** O padrão dos evitáveis é claro: 74/74 anexaram algo, 74,3%
no Great Tusk, enquanto a vítima era a Crustle em 41,9% dos casos. Como a Rock cobre o Great Tusk
tão bem quanto a Mist, a regra óbvia é "não gaste Mist num host {F} se há Rock disponível". Medida
a população exata: **28 trocas em 215 jogos = 0,13/jogo (8,9% das Mist anexadas em host {F})**.
Isso é da ordem do v4 (0,6% das decisões, resultado nulo) — **não paga um A/B caro**. A população
maior (421/3447 = 12,2% dos NOSSOS TURNOS com ativo descoberto e attach protetora legal) não é
almoço grátis: anexar para proteger disputa o mesmo attach que paga o `Land Collapse` ({C}{C}),
então também é economia, não política.

**Veredito (A): ECONOMIA, estruturalmente fechada. Encerra a frente.**

### (B) Archaludon: célula fechada, e ela nunca foi um buraco
Lista reconstruída dos nossos 32 episódios reais (`src/analysis/mine_opponent_deck.py`). O método
NÃO é o do meta_radar: aqui os jogos vêm de **30 times diferentes** jogando o mesmo arquétipo, e
o "máximo já visto" entre times une as techs de todos e estoura 60 (medido: 76). Cada jogo vira
uma observação ruidosa de uma lista-padrão: inclusão por presença (≥50% dos jogos), cópias pela
**moda do máximo por jogo**, e a Basic Energy fecha os 60. Resultado: 49 cartas nomeadas + 11
Basic {M} — e 11 é exatamente a moda observada, o que fecha sozinho. LEGAL e aceita pelo motor.
Descartadas como ruído: Xerosic's (7/32), Hand Trimmer (2/32), Dwebble/Flutter Mane/Great Tusk
(1/32 cada).

| | winrate |
|---|---|
| REAL (32 jogos) | **90,6%** (29-3), IC95 [75,8%, 96,8%] |
| interno, oponente heurístico (N=600) | 97,0% [95,3%, 98,1%] |
| interno, oponente network (N=600) | 97,7% [96,1%, 98,6%] |

Δ(interno − real) = **+6,4pp [+0,0, +21,3]**. Formalmente na fronteira, mas a comparação certa é
de ordem de grandeza: a célula Alakazam erra **+48,5pp [+35,5, +59,7]**. Rótulo honesto:
**levemente otimista, mesmo regime** — não é o tipo de descalibração que invalida a célula. E o
sinal prático é o oposto de um buraco: ganhamos 90,6% no real. As variantes de deck também são
chatas aqui (V1 +1,7pp [−0,1, +3,6], nenhuma significativa).

**Mecanismo das 3 derrotas reais** (perspectiva fixa): em 2 delas o oponente estava a **2 e 3
cartas** do deck-out com 1 prêmio restante — perdemos a corrida por um turno. Nas três o nosso
ativo final era Terrakion/Dwebble, não o Great Tusk: o miller já tinha morrido e o mill parou. Os
ataques deles (Raging Hammer, Hammer In, Metal Defender) são **dano puro, sem efeito prevenível** —
coerente com (A), onde a lista de hits preveníveis é 257 Alakazam + 20 Dragapult e **zero**
Archaludon. Aqui a Mist não teria ajudado; não há resposta ignorada no nosso deck.

**Higiene que faltava.** As `variant_*.csv` da rodada anterior estavam entrando no campo
auto-descoberto do gauntlet — quatro quase-clones da nossa própria lista substituindo células reais
e reponderando toda média de campo. `discover_decks` agora as exclui por padrão
(`include_candidates=True` para pedi-las), e `tests/test_deck_pool_contract.py` passa a exigir que
TODO deck do pool tenha 60 cartas legais que o **motor** aceita, e que o campo não contenha
candidatos e contenha Archaludon.

## [31/Jul] O Alakazam nao mente por disrupcao, mente por TEMPO -- e havia um segundo buraco igual
Duas frentes, offline, nada shipado. `deck.csv` intocado.

### (A) Por que a celula Alakazam mente -- hipotese principal REFUTADA
Coletor unico consumindo `(agent_index, obs, action)`, alimentado pelos replays reais E pelos jogos
internos -- medir os dois lados com codigos diferentes deixaria a comparacao a merce dos proprios
bugs. 59 jogos reais (filtro de submissao + sentinela + assento fixo) vs 300 internos por piloto.

**Enhanced Hammer nao explica nada.** Era a hipotese principal e morre no dado: os reais jogam
**2,00/jogo**, o heuristico **1,97/jogo** -- identicos (o BC-majkel e que subjoga, 1,09). Pior para
a hipotese: a fracao das nossas energias especiais **ARRANCADAS** com o portador vivo e
**38,6% no real e 51,4% no heuristico** -- a celula interna e MAIS dura com a nossa prevencao, nao
menos. Dano medio do Powerful Hand tambem empata (266 real vs 256/273).

**O que explica e tempo.** Ranqueado por caminho mecanico ate o winrate:

| divergencia | real | heuristico | BC-majkel |
|---|---|---|---|
| **Powerful Hands ACERTADOS/jogo** | **6,08** | 3,70 | 4,43 |
| share dos ataques que e Powerful Hand | 89,3% | 75,9% | 69,5% |
| Rare Candy/jogo (monta o Stage 2) | **1,10** | 0,63 | 0,63 |
| turno em que o Alakazam entra | **5,25** | 6,53 | 5,78 |
| -- consequencia: eles DECKAM (chegam a 0) | 28,8% | 67,3% | 66,7% |
| -- deck deles no minimo (mediana) | 5,0 | 0,0 | 0,0 |

O oponente real monta o Alakazam ~1,3 turno antes (Rare Candy 1,75x) e converte 89,3% dos seus
ataques no golpe que mata, contra 76%/70%. Resultado: **1,64x mais Powerful Hands por jogo**, ao
mesmo dano unitario. A conta fecha: 6,08 x 266 = 1617 de dano entregue no real contra 3,70 x 256 =
947 no interno. Com ~150 HP por corpo, 1617 fecha 6 premios com folga e 947 fecha na unha -- que e
exatamente a diferenca entre perder 64% e ganhar 83%.

**E nao e o nosso mill que esta rapido demais.** Cartas milladas por turno: 3,00 real vs 3,50
heuristico; turnos por jogo 19,58 vs 17,90. Mesma vazao, mesma duracao. O que muda e o relogio
DELES. **Veredito: descalibracao explicada por comportamento subjogado, sem residuo material.** A
celula e consertavel com um oponente melhor -- e o matchup real e genuinamente mais duro do que
sabiamos (35,6% e o numero verdadeiro; 83,5% era tempo regalado).

### (B) Cobertura do campo: 58,6% -> 88,4%, e um segundo buraco do mesmo tamanho
Auditoria dos 215 jogos reais decididos. **So 58,6% do campo estava coberto por celula medida** --
e o que faltava nao eram listas, era **mapeamento**:

| arquetipo | share real | WR real | tinha celula? |
|---|---|---|---|
| Alakazam | 27,4% | 35,6% | sim |
| Grimmsnarl | 16,7% | 38,9% | sim |
| **Mega Lucario ex** | **14,9%** | **50,0%** | **NAO (sem mapeamento)** |
| **Archaludon** | **14,9%** | 90,6% | **NAO (lista existia, sem mapeamento)** |
| unknown | 7,0% | 53,3% | nao (cauda longa real) |

Minerei o Lucario dos 32 jogos reais: bate **57/60** com o seed feito a mao que ja estava no disco.
Ou seja, os dois buracos grandes eram entradas faltando em `ARCHETYPE_DECKS`, nao decks faltando --
o tipo de buraco que nao parece nada. Corrigido; cobertura vai a **88,4%**. O resto e cauda longa
(`unknown` = Hop Trevenant 4, Cynthia Garchomp 4, Hydrapple 2 -- nenhum >=5% sozinho).

**E o Lucario escondia um segundo buraco de calibracao do tamanho do Alakazam:**

| celula | piloto interno | real | interno | delta |
|---|---|---|---|---|
| Grimmsnarl | **BC casado** | 38,9% | 45,4% | **+6,5pp** compativel |
| Archaludon | heuristico | 90,6% | 96,3% | +5,7pp compativel |
| Alakazam | heuristico | 35,6% | 83,5% | **+47,9pp** |
| Alakazam | BC-majkel (nao casado) | 35,6% | 89,7% | **+54,1pp** |
| **Mega Lucario** | heuristico | 50,0% | 96,0% | **+46,0pp** |
| Kangaskhan | heuristico | 75,0% | 97,0% | +22,0pp |
| Spidops | heuristico | 12,5% | 82,8% | **+70,3pp** |
| Starmie | heuristico | 62,5% | 25,8% | **-36,7pp** (pessimista!) |

O Lucario tem a **mesma assinatura** do Alakazam: mesma vazao de mill (2,94 vs 3,10/turno), mesma
duracao (20,78 vs 21,45 turnos), mas eles deckam **34,4% no real e 86,7% no interno**.

**A leitura que unifica tudo: o piloto interno subjoga tempo, entao toda celula cuja corrida e
apertada sai inflada; a unica que calibra sem BC casado (Archaludon) e justamente a que ganhamos de
verdade, onde nao ha o que inflar.** E o remedio nao e mais jogos: o unico piloto casado por
arquetipo que temos (`bc_grimmsnarl`) produz a unica celula contestada calibrada, enquanto um BC
clonado de jogador mas aplicado a outro arquetipo (BC-majkel no Alakazam) fica **pior** que o
heuristico. Calibracao = **behaviour cloning casado por arquetipo**. O Starmie ainda e o caso
oposto e nao explicado (interno pessimista), com N real de 8.

**Ressalva de N.** So Alakazam (59) e Grimmsnarl (36) tem N real decente; Lucario e Archaludon tem
32; Kangaskhan 12; Spidops e Starmie 8. Os ICs de Newcombe ja carregam isso -- Spidops e
[+35,6, +81,0], largo mas ainda excluindo zero.

**Bug encontrado e corrigido no caminho.** `mine_opponent_deck` agregava por NOME e resolvia de
volta pelo primeiro id com aquele nome. "Alakazam" e o id 743 (Powerful Hand, a condicao de vitoria
inteira) **e** o 245 (Strange Hacking/Psychic). Re-minerar o Alakazam produzia uma lista que nao
sabe executar o proprio plano. Agora tudo e chaveado por card id; a lista do Archaludon commitada
ontem foi conferida e **nao** foi afetada. Guarda hermetica em `tests/test_mine_opponent_deck.py`,
que primeiro verifica que a colisao ainda existe (senao o teste seria vacuo).

## [31/Jul] O fix de tempo acertou o alvo e NAO calibrou -- o gargalo e o scorer de ataque
Offline, nada shipado. `deck.csv` intocado e verificado identico ao HEAD; 296/298 testes (as 2
falhas de `test_portfolio_watch` seguem pre-existentes no HEAD).

### 1. Auditoria de decks: nenhuma lista aleijada
Varri os 22 decks contra o padrao do bug de homonimo. 19 deles contem alguma carta cujo nome tem
mais de uma impressao no pool (sao 154 nomes colidentes). **Todas as escolhas estao corretas**, em
particular a que importava: `meta_alakazam.csv` usa o **id 743 (Powerful Hand)**, nao o 245. O
Duraludon 169 do Archaludon tem `Raging Hammer`, que e exatamente o ataque que observei nos jogos
reais -- confirmacao independente. **O deficit de Powerful Hands nao e o deck.**

### 2. O deficit de tempo: corrigido, atras de flag, e o motor validou a premissa
Rare Candy chega como PLAY de Trainer, cai em `_TRAINER_BAND` (35) e perde para qualquer EVOLVE
(80+), empatando com todo outro Trainer. Antes de mexer, probe no motor: em 40 jogos da lista
Alakazam, Rare Candy foi oferecida em **0 de 1504** decisoes sem Stage 2 na mao e em 289 das 2093
com Stage 2 -- **o motor ja filtra**, entao toda oferta legal e uma aceleracao real e o scorer nao
precisa re-derivar a condicao. Promovida para `_EVOLVE_BAND + 5` atras de `tempo=False` (novo arm
`heuristic-tempo`).

**Equivalencia do ship.** Nao da para "rodar o mesmo jogo duas vezes" -- o motor nao e semeavel.
A comparacao disponivel e lockstep no MESMO fluxo: um CrustleAgent dirige e um shadow com
`tempo=True` recebe cada observacao e tem de responder igual. Zero divergencias. Mais duas camadas:
`CrustleAgent._tempo` e False por construcao, e `deck.csv` nao contem nenhum acelerador, entao a
regra nao pode disparar nem se ligada. Tudo em `tests/test_tempo_equivalence.py`, sobre opcoes
REAIS do motor.

### 3. O fix acertou o alvo -- e nao moveu a calibracao
| metrica | real | antes | depois |
|---|---|---|---|
| **Rare Candy/jogo** | **1,10** | 0,58 | **1,09** |
| turno em que o Alakazam entra | 5,25 | 6,73 | 6,08 |
| Powerful Hands acertados/jogo | **6,08** | 3,61 | 3,88 |
| eles deckam (chegam a 0) | 28,8% | 71,0% | 67,3% |

Calibracao, mesmos alvos reais, N=600/celula, 0 exceptions:

| celula | real | antes | depois | erro antes | erro depois | melhora |
|---|---|---|---|---|---|---|
| Alakazam | 35,6% | 82,7% | 80,3% | +47,1pp | +44,7pp | **+2,3pp** |
| Mega Lucario | 50,0% | 96,2% | 96,0% | +46,2pp | +46,0pp | +0,2pp |
| Archaludon | 90,6% | 98,5% | 96,0% | +7,9pp | +5,4pp | +2,5pp |
| Kangaskhan | 75,0% | 98,3% | 99,0% | +23,3pp | +24,0pp | -0,7pp |
| Spidops | 12,5% | 80,0% | 78,3% | +67,5pp | +65,8pp | +1,7pp |
| Starmie | 62,5% | 25,2% | 23,8% | -37,3pp | -38,7pp | -1,3pp |

**A metrica-alvo foi de 0,58 para 1,09 contra 1,10 do real -- alvo cravado -- e o winrate andou
2,3pp de 47,1.** O Rare Candy era um deficit real e agora esta corrigido, mas **nao era a restricao
ativa**. Antecipar o Stage 2 em 0,65 turno comprou apenas +0,27 ataque.

### 4. O gargalo real, medido: o scorer nao enxerga o Powerful Hand
`_effect_adjustment` avalia ataques pelo `damage_base` mais um bonus por linha de efeito. O
Powerful Hand tem base **0** (o dano dele vem de COUNTERS escalado pelo tamanho da mao):

| ataque | base | **scorer** | dano real |
|---|---|---|---|
| **Powerful Hand** (Alakazam) | 0 | **13,0** | **~266** |
| Super Psy Bolt (Kadabra) | 30 | 30,0 | 30 |
| Superb Scissors (nossa Crustle) | 120 | 120,0 | 120 |

**O piloto acha que o ataque de 266 vale 13, e portanto prefere o Kadabra de 30.** Subvalorizacao de
~20x. Isso explica o que o Rare Candy nao explicava: adiantar o Alakazam nao adianta se o piloto
nao quer atacar com ele. Bate com o resto da telemetria -- ataques/jogo 4,84 interno vs 6,81 real, e
share de Powerful Hand 74,7% vs 89,3%.

**Nao implementei a correcao nesta rodada, de proposito.** `_effect_adjustment` e compartilhado: o
CrustleAgent chama `super()._main_score`, entao mexer nele MUDA O SHIP e exige a sua propria rodada
com gate de equivalencia -- diferente do fix do Rare Candy, que era inerte para nos por construcao.
E a regra generica certa (`DAMAGE_SCALE` por unidade deveria escalar pela contagem real de
unidades, nao por um chute de 1,5) precisa ser verificada carta a carta antes de virar banda.

### 5. Conclusoes medidas no regime errado, por dependencia da celula inflada
Nenhuma re-rodada agora -- so a lista, ordenada por quanto o resultado dependia da celula:
1. **Busca em runtime (`search_crustle`)** -- a decisao de nao shipar veio de celulas de campo cujo
   oponente e o piloto generico; a escada de fidelidade (Grimmsnarl) usava BC casado e e a menos
   afetada, mas o "vs campo" nao. Maior dependencia.
2. **Variantes de deck (V1-V4)** -- toda a matriz de 8 celulas rodou contra o piloto generico. O
   veredito "nenhuma merece ship" pode sobreviver (as diferencas eram nulas ou negativas), mas o
   ponto de operacao estava deslocado ~46pp em duas celulas grandes.
3. **v4 do piloto (Xerosic + attach protetora)** -- mediu nulo a 0,6% das decisoes; o regime
   inflado torna qualquer efeito defensivo menos visivel, porque o oponente pressiona menos.
4. **Gate B/C e o gauntlet historico** -- mesma ressalva, menor dependencia: comparam pilotos NOSSOS
   sob o mesmo oponente, e o vies e comum aos dois bracos.

## [31/Jul] Avaliacao de dano em escala: o fix esta certo, e a banda de ataque e o teto
Offline, nada shipado. `deck.csv` intocado e identico ao HEAD. 306/308 testes estaveis em 5
rodadas seguidas (as 2 falhas de `test_portfolio_watch` seguem pre-existentes). 0 exceptions em
7.200 jogos.

### 1. O fix: contar a unidade viva em vez de um bonus fixo
`_effect_adjustment` avaliava ataque por `damage_base` + bonus fixo por linha de efeito, cego a
qualquer ataque cujo dano E a escala. Agora a clausula e lida do TEXTO do motor (`(\d+) damage
counters ... for each X` / `does N (more) damage for each X`), convertida em (dano por unidade,
unidade) e cacheada por attackId; a unidade e contada na observacao viva.

Cobertura sobre as **163 clausulas de escala do pool**:
- **73 observaveis no estado** (mao, energias por tipo, contadores de dano, banco, premios tomados,
  tools, corpos em jogo);
- **43 de moeda**, resolvidas por esperanca (`flip 4 coins` -> 2 caras; `flip until tails` -> 1) em
  vez de cair no fallback generico, que valeria o dobro num ataque de 2 moedas;
- **47 nao resolvidas** (contagens de pilha de descarte, "cards you discarded in this way") -> a
  estimativa declarada de 2 unidades, explicita e reportavel, em vez de fingir que sabe.

**116 ataques mudaram de valor.** Maiores altas: Scarring Shout +245, Spicy Rage +238,
**Powerful Hand +227 (13,0 -> 240)**, Resentful Refrain +175. E **5 ataques CAIRAM** (Gadget Show
45 -> 30): o bonus fixo era generoso demais. Corrige nos dois sentidos, nao so infla.

Em jogo real, com o motor oferecendo a opcao: valor medio do Powerful Hand **13,0 -> 313,3**, e
reconhecido como LETAL em **0/767 -> 83%** das decisoes. Antes ele era uma constante 13,0
independente da mao -- e essa e a forma exata do bug, afirmada assim no teste (nao "nunca letal":
um Pokemon com 10 de HP morre para qualquer coisa, e a frase falharia por motivo errado).

### 2. Impacto no ship: INERTE, e a hipotese de "cegueira a ameaca" nao se aplica
**Lockstep sobre os jogos REAIS**: 215 episodios da submissao 54917180, **11.360 decisoes
comparadas, 0 divergencias** com os DOIS flags ligados no shadow. Mais duas camadas estruturais:
nenhum ataque do `deck.csv` tem clausula de escala, e `_scaled_damage` e False por construcao.
Ha ainda um lockstep contra o Alakazam, para exercitar o caminho novo com o oponente escalando.

**(b) A regra manual de Xerosic NAO vira redundante.** Verifiquei quem consome `_attack_value`:
so o scorer das NOSSAS opcoes de ataque (`_main_score` e o contexto ATTACK) -- **nunca** para
avaliar ameaca. A logica de ameaca do CrustleAgent e `_opp_active_is_ex`, uma checagem de tipo de
carta. Entao avaliar ataque corretamente nao torna o piloto menos cego a ameaca: um gatilho manual
"20 x mao >= hp" continua sendo o unico caminho ciente de ameaca. (O piloto v4 nao existe neste
branch.)

### 3. Calibracao: melhora real, pequena -- e o mecanismo diz por que
| celula | real | antes | depois | erro antes | erro depois | melhora |
|---|---|---|---|---|---|---|
| Alakazam | 35,6% | 85,2% | 79,7% | +49,6pp | +44,1pp | **+5,5pp** |
| Mega Lucario | 50,0% | 96,8% | 96,2% | +46,8pp | +46,2pp | +0,7pp |
| Spidops | 12,5% | 80,8% | 77,0% | +68,3pp | +64,5pp | +3,8pp |
| Starmie | 62,5% | 23,7% | 28,8% | -38,8pp | -33,7pp | **+5,2pp** |
| Archaludon | 90,6% | 97,8% | 98,0% | +7,2pp | +7,4pp | -0,2pp |
| Kangaskhan | 75,0% | 96,7% | 97,5% | +21,7pp | +22,5pp | -0,8pp |

Erro absoluto medio **38,7pp -> 36,4pp**. Anda na direcao certa nas duas celulas que mais erravam
em sentidos OPOSTOS (Alakazam otimista -5,5; Starmie pessimista +5,2), o que e o padrao esperado de
uma correcao de avaliacao. Mas e **6% do erro**.

**A telemetria mostra o teto:** share de Powerful Hand 73,7% -> 79,7% (real 89,3%), Powerful Hands
por jogo 3,62 -> 3,87 (real 6,08) e **ataques por jogo 4,91 -> 4,86 (real 6,81) -- inalterado**.

### 4. O gargalo agora e a ARQUITETURA DE BANDAS, nao o valor
`_ATTACK_BAND` = 20, com o comentario explicito no codigo: *"capped so attacking never outranks
development actions"*. O score maximo de um ataque fica em ~31, abaixo de ATTACH (55), PLAY (70) e
EVOLVE (80). O piloto so ataca quando nao ha mais nada a fazer. Isso e **correto para o nosso deck
de mill** e errado para um deck agressivo: o Alakazam real ataca todo turno. **Nenhuma correcao de
VALOR de ataque pode mudar isso** -- e por isso que o valor certo comprou 5,5pp e nao 45.

### 5. Lista do regime errado: criterio NAO atingido, nao reabro ainda
A condicao era "se a calibracao melhorar materialmente". 2,4pp de ~39pp nao e material: o ponto de
operacao das celulas grandes continua deslocado ~44pp. Entao **nada a re-rodar por enquanto**, e a
ordem de dependencia segue a mesma (busca em runtime > variantes de deck > v4 do piloto). O que
mudaria a resposta e a banda de ataque -- se ela for corrigida e o Alakazam cair para a faixa dos
40-50%, ai sim vale re-rodar, comecando pela busca com N>=600/celula.

## [01/Ago] A banda de ataque NAO era o gargalo -- e o alvo telemetrico era um proxy ruim
Offline, nada shipado. `deck.csv` intocado. 319/321 testes (as 2 falhas de `test_portfolio_watch`
seguem pre-existentes; a suite tem ainda 0-2 falhas intermitentes por rodada, verificadas na MESMA
taxa num worktree limpo do HEAD -- motor nao semeavel, nao e regressao). 0 exceptions em 12.000
jogos de calibracao.

### 1. A premissa caiu antes de escrever codigo
O diagnostico da rodada anterior era: `_ATTACK_BAND=20` mantem o ataque abaixo de desenvolver, e
por isso ataques/jogo fica em 4,86 contra 6,81 do real. **Atacar ENCERRA o turno** -- entao a banda
so REORDENA acoes DENTRO do turno; ela so pode custar um ataque se uma acao de desenvolvimento
tomada antes remover a capacidade de atacar. Isso e mensuravel, e foi medido antes de mexer
(`src/analysis/attack_census.py`, 60 jogos da lista Alakazam):

| | antes |
|---|---|
| turnos que OFERECERAM ataque | 319/543 = 58,7% |
| ...e que CONVERTERAM em ataque | 315/319 = **98,7%** |
| turnos SEM ataque na mesa | 224 = **41,3%** |
| ...destes, com ativo de ZERO energia | 210 de 215 |
| ataques nao-PH com Powerful Hand na mesa | **0 de 58** |

**O piloto ataca em 98,7% dos turnos em que atacar e legal, e ja escolhe sempre o ataque certo.**
O deficit inteiro esta nos 41,3% de turnos em que nunca houve ataque na mesa -- corpo errado na
frente, ou sem energia. Nenhuma banda alcanca isso.

Segunda sonda (40 jogos): energia so esta anexavel em 5,38 turnos/jogo e o piloto anexa nos 5,38
(100%). Mas **46 das 155 anexacoes foram para outro corpo com um Alakazam de 0 energia no board**,
e das 45 promocoes em que o Alakazam estava disponivel so 29 o escolheram. Causa: `_attach_score`
(via `_best_affordable_damage`) e `_own_pokemon_score` leem `damage_base`, que e None no Powerful
Hand -- **a mesma cegueira que o fix de escala tirou do `_attack_value` e deixou nos dois scorers
que decidem o BOARD**.

### 2. O que foi implementado (dois flags, ambos default OFF)
**(a) Banda por PERFIL DE DECK** -- o item pre-registrado. `archetype_rules.deck_profile()` deriva
o perfil da lista pelas MESMAS regras do radar e do estimador de runtime: AGGRO para quem fecha na
corrida de premios, DEVELOPMENT para mill/muro/lock e para `UNKNOWN` (o lado conservador). Sob
AGGRO um ataque **LETAL** sobe para 90, acima de EVOLVE; ataque nao-letal fica exatamente onde
estava, porque atacar encerra o turno e trocar desenvolvimento por um golpe que nao mata e pior.
`deck.csv` deriva DEVELOPMENT -- o arm e inerte na nossa lista por construcao, asserido contra a
`arm_factory` de verdade.

**(b) `energy_routing`** -- `_body_damage()` passa a ser compartilhado por `_best_affordable_damage`
(anexacao) e `_own_pokemon_score` (promocao/banco), reusando o mesmo parser de clausula e a mesma
contagem viva de unidade. Os tres scorers finalmente concordam sobre quanto vale um ataque.

### 3. Inercia do ship
Lockstep sobre os JOGOS REAIS -- 222 episodios da submissao 54917180, **11.495 decisoes**:

| shadow | divergencias |
|---|---|
| `energy_routing` + `scaled` + `tempo` forcados | **0 / 11.495** |
| `profile=AGGRO` **forcado** | 144 / 11.495 = 1,25% |

O segundo numero e o teste que importa: a banda **nao** e inofensiva, e o ship esta seguro porque
`deck.csv` deriva DEVELOPMENT, nao porque a regra nao morde. Os dois lados estao em
`tests/test_attack_profile.py`, sobre opcoes REAIS do motor, inclusive o contra-teste que exige que
forcar AGGRO mude alguma coisa (se vier zero, ou a regra morreu ou o teste parou de exercita-la).

### 4. O alvo telemetrico foi ATINGIDO -- pela banda, nao pelo routing (N=300/arm)
| metrica | real | antes | **aggro** | routing | aggro+routing |
|---|---|---|---|---|---|
| **ataques/jogo** | **6,81** | 5,03 | **7,69** | 4,84 | 7,39 |
| share Powerful Hand | 89,3% | 80,4% | 83,6% | 78,4% | 83,2% |
| **Powerful Hands acertados/jogo** | **6,08** | 4,04 | **6,42** | 3,79 | 6,14 |
| premios que ELES tiram/jogo | 5,66 | 5,59 | 5,48 | 5,38 | 5,55 |
| turnos por jogo | 19,58 | 18,12 | 23,09 | 17,75 | 22,72 |
| eles deckam (chegam a 0) | 28,8% | 64,7% | 56,3% | 67,3% | 55,0% |
| NOSSO winrate | **35,6%** | 80,0% | 84,3% | 83,7% | 84,0% |

**E aqui esta a licao.** Os tres alvos telemetricos foram cravados (ataques/jogo 5,03->7,69 contra
6,81; PH acertados 4,04->6,42 contra 6,08) e **os premios que eles tiram nao se moveram**: 5,59 ->
5,48. Cinquenta por cento mais ataques, o mesmo numero de premios. O mecanismo e conhecido e ja
estava documentado no CLAUDE.md: **Powerful Hand poe CONTADORES de dano, que sao efeito, e Mist/Rock
previnem efeito** -- o ataque "letal" que a banda agora prioriza e anulado pela nossa parede. O que
os ataques extras de fato compraram foi o oponente parar de jogar os trainers que o deckavam: os
jogos passam de 18,1 para 23,1 turnos, o deck-out deles cai de 64,7% para 56,3%, e **o nosso
winrate SOBE de 80,0% para 84,3%** -- para longe do alvo real de 35,6%.

`ataques/jogo` era um PROXY, e casar o proxy nao calibrou a celula. O censo mostra por que o proxy
subiu: a conversao continua 98,7% nos dois arms; o que mudou foi o numero de turnos do oponente
(9,05 -> 11,22) e os turnos sem oferta (41,3% -> 30,0%).

### 5. Calibracao, N=600/celula, alvos reais fixos, 0 exceptions
| celula | real | antes | aggro | routing | erro antes | erro aggro | erro routing |
|---|---|---|---|---|---|---|---|
| Alakazam | 35,6% | 81,8% | 85,3% | 79,3% | +46,2 | +49,7 | +43,7 |
| Mega Lucario | 50,0% | 95,0% | 96,0% | 96,7% | +45,0 | +46,0 | +46,7 |
| Archaludon | 90,6% | 96,5% | **89,7%** | 96,8% | +5,9 | **-0,9** | +6,2 |
| Kangaskhan | 75,0% | 98,2% | 98,8% | 97,8% | +23,2 | +23,8 | +22,8 |
| Spidops | 12,5% | 80,8% | 79,3% | **65,2%** | +68,3 | +66,8 | **+52,7** |
| Starmie | 62,5% | 27,7% | 31,7% | **37,8%** | -34,8 | -30,8 | **-24,7** |
| **erro absoluto medio** | | | | | **37,2pp** | **36,4pp** | **32,8pp** |

(aggro+routing: 33,9pp.) Spidops e Kangaskhan derivam DEVELOPMENT, entao o arm aggro e inerte neles
por construcao -- e os dois se moveram so dentro do IC (80,8->79,3 e 98,2->98,8). Controle embutido
que funcionou.

**GATE PRE-REGISTRADO: FALHOU.** Exigia erro do Alakazam <= ~15pp (celula interna em 40-50%). Veio
+49,7pp (aggro) e +43,7pp (routing). **Nao reabro a lista do regime errado nesta rodada.** A ordem
segue a mesma para quando houver gate: busca em runtime (N>=600/celula) > variantes de deck > v4.

### 6. O que sobra, e e um resultado de verdade
1. **O routing e a melhor calibracao ate agora**: erro medio 37,2 -> **32,8pp**, o maior salto de
   qualquer rodada desta serie, e vem das duas celulas que mais erravam depois do Alakazam (Spidops
   +15,6pp, Starmie +10,1pp). E o arm a carregar como oponente padrao daqui pra frente.
2. **A banda calibrou o Archaludon quase perfeitamente** (96,5% -> 89,7% contra 90,6% real) --
   uma celula agressiva de verdade, sem parede que anule o plano dela.
3. **O Alakazam nao mente por agressao.** Tres rodadas ja tentaram: tempo (Rare Candy, alvo cravado,
   +2,3pp), valor (escala, +5,5pp) e agora banda (alvo cravado, **-3,5pp**). A celula erra +44pp
   com o comportamento de ataque casado com o real. A hipotese que sobra e a que a telemetria
   aponta sozinha: **eles tiram 5,66 premios contra os nossos ~5,5 e ainda assim ganham 64% -- o
   que difere e o que acontece com a nossa PREVENCAO**, e a proxima medicao tem de ser essa, nao
   mais uma variavel de ataque.

### 7. Higiene de suite (duas flakes pre-existentes, mesma familia)
Ambas de dimensionamento de amostra, nao de comportamento. (a) Os lockstep de inercia rodavam UM
jogo e guardavam contra vacuidade com `compared > 10`; um jogo pode terminar em 8 decisoes nossas,
e ai a guarda falhava no lugar da assercao que ela protege -- agora sao 5 jogos e guarda em 100.
(b) `test_without_the_fix_the_value_ignores_the_hand` exigia que o valor cego fosse a constante
13,0, mas `_attack_value` soma `_KO_BONUS` quando o valor ja alcanca o HP do defensor: um defensor
em 10 de HP transforma o mesmo 13,0 em 113,0. A assercao agora e lida sobre os ataques que o
defensor SOBREVIVE, que e a afirmacao que se queria fazer. Com as duas, os tres modulos de flag
passaram 10/10 rodadas seguidas.

- `src/deckbuilding/archetype_rules.py`: PROFILE_*, AGGRO_ARCHETYPES, `deck_profile()`
- `src/agent_heuristics/heuristic_agent.py`: `profile=`, `energy_routing=`, `_body_damage`,
  `_scaled_units`, `_is_lethal`
- `src/environment_wrapper/ab_test.py`: arms `-aggro` / `-routing` (perfil derivado do DECK do arm)
- `src/analysis/attack_census.py`: o censo que separa "recusou" de "nunca foi legal" (NOVO)
- `src/analysis/field_calibration.py`: a tabela de calibracao como um comando (NOVO)
- `tests/test_attack_profile.py`: 13 testes -- derivacao, banda e routing sobre opcoes REAIS,
  inercia do ship e o contra-teste da banda

## [01/Ago] O gusting: mecanismo CONFIRMADO, magnitude insuficiente -- e uma metrica quebrada achada
Offline, nada shipado. `deck.csv`/`main.py`/`build_submission.py` identicos ao HEAD. 324/326 testes
(as 2 falhas de `test_portfolio_watch` seguem pre-existentes), 3 rodadas estaveis. 0 exceptions em
7.200 jogos de calibracao.

### 1. A hipotese, e o numero que a decide
Powerful Hand so alcanca o ATIVO e e anulado por Mist/Rock (contador de dano e EFEITO). Entao o
caminho que sobra para o oponente real nao e atacar mais forte, e atacar OUTRO: puxar para o ativo
o corpo sem cobertura. Quando o motor oferece um SWITCH sobre o NOSSO banco ele esta oferecendo uma
ESCOLHA, e a estatistica que decide e condicional: **dado um conjunto com corpo COBERTO e corpo
DESCOBERTO na mesa, quantas vezes levam o descoberto?**

`src/analysis/gust_telemetry.py`, um coletor alimentado pelos dois lados (replays reais com filtro
de submissao + sentinela de arquetipo nos DOIS decks, e jogos internos):

| | REAL (59 jogos) | interno | interno + fix |
|---|---|---|---|
| gusts/jogo | 1,08 | 1,59 | 1,67 |
| turno medio do gust | 12,84 | 10,18 | 10,17 |
| alvo DESPROTEGIDO (todos os gusts) | 100,0% | 81,8% | 97,0% |
| **>> escolhas com coberto E descoberto na mesa** | **6** | **54** | **66** |
| **>> ...e puxaram o DESCOBERTO** | **100,0%** | **11,1%** | **100,0%** |
| HP medio do alvo puxado | 115,8 | 129,1 | 97,5 |

**Achado.** O piloto generico fazia o OPOSTO exato do real. Os alvos dizem tudo: o real puxa
Dwebble x23 (70 de HP, sem energia) e Crustle x23; o nosso puxava **Great Tusk x174** -- o corpo
{F} que carrega a Rock Fighting, ou seja, a unica peca que a prevencao protege. A causa e estrutural:
`_own_pokemon_score` avalia um corpo INIMIGO com a regra das nossas proprias promocoes (maior HP +
dano impresso), entao arrastava a nossa parede.

### 2. Consequencia medida
Dos KOs que sofremos, **16,3% (real) foram num corpo que tinha ACABADO de ser arrastado**, e desses
**85,0% morreram no MESMO turno do gust** -- a sequencia e gust -> KO no mesmo turno, como a hipotese
previa. Nas nossas derrotas reais a fracao e 11,1%. Interno: 25,5% dos KOs pos-gust, mas so 54,8% no
mesmo turno (arrastavam a parede, que nao morre).

### 3. Os OUTROS bypasses: varridos, e nao ha um terceiro
Varri as **30 cartas distintas efetivamente reveladas** do lado deles nos 59 jogos reais (fonte =
corpus, nao a nossa lista reconstruida) contra o texto do motor. Prevencao so para EFEITOS DE ATAQUE
feitos ao portador, entao os contornos possiveis sao: bater no banco, bater com DANO puro, tirar a
energia antes, usar ABILITY, ou puxar outro corpo. Resultado:
- **`Boss's Orders` x146** -- o gust. Medido e corrigido acima.
- **`Enhanced Hammer` x190** -- tirar a cobertura antes de atacar. **Ja estava casado**: 2,00
  descartes/jogo reais contra 1,90 internos, e levaram energia que de fato COBRIA em 91,5% (real)
  contra 90,3% (interno). Nao e deficit.
- **Nenhum ataque que ignore efeito, e nenhuma ability que cause dano.** ⚠️ Meu primeiro parse
  concatenou o texto da skill com o do ataque e fez a ability do Alakazam parecer "Draw 3 cards.
  Place 2 damage counters..."; conferido carta a carta, a ability e **so** "Draw 3 cards" e os
  contadores sao o ATAQUE 1072, que E prevenido. Nao existe bypass por ability.
- `Shaymin` e `Battle Cage` aparecem na varredura por conterem "prevent"/"Benched", mas sao
  protecao DELES, nao contorno da nossa.

### 4. O fix, atras de flag (`gust_targeting`, default OFF)
`_gust_target_score`: cobertura primeiro (ataque prevenido e turno perdido), depois a armadilha --
HP baixo e KO agora, custo de recuo alto e falta de energia sao o que prende o corpo la. A tabela de
cobertura e **derivada do texto do motor**, nao memorizada: energia especial cujo texto diz prevenir
todos os efeitos de ataques feitos ao Pokemon a que esta ligada, com restricao de tipo opcional. Hoje
isso da exatamente Mist (qualquer host) e Rock Fighting (host {F}) -- a matriz que o teste de
contrato ja verificara contra o motor, mas agora vinda da carta.

**Inercia do ship**: lockstep sobre os jogos REAIS, 222 episodios / **11.495 decisoes**, flag forcada
no shadow: **0 divergencias**. (CrustleAgent ja roteia promocao de inimigo pelo proprio
`_v2_trap_target_score`, entao ha duas camadas.)

### 5. Calibracao, N=600/celula, alvos reais fixos, 0 exceptions
| celula | real | antes | gust | erro antes | erro gust |
|---|---|---|---|---|---|
| Alakazam | 35,6% | 82,0% | 78,0% | +46,4 | **+42,4** |
| Mega Lucario | 50,0% | 97,0% | 93,7% | +47,0 | +43,7 |
| Archaludon | 90,6% | 96,8% | 97,0% | +6,2 | +6,4 |
| Kangaskhan | 75,0% | 97,3% | 96,0% | +22,3 | +21,0 |
| Spidops | 12,5% | 81,5% | 80,0% | +69,0 | +67,5 |
| Starmie | 62,5% | 26,0% | 24,5% | -36,5 | -38,0 |
| **erro absoluto medio** | | **37,9pp** | **36,5pp** | | |

(routing+gust: 32,9pp, ainda o melhor conjunto.) Alakazam **-4,0pp, IC95 Newcombe [-8,5, +0,5]** --
o sinal anda na direcao certa mas **atravessa o zero a N=600**: e sugestivo, nao estabelecido.

**GATE: FALHOU.** Exigia erro do Alakazam <= ~15pp; veio +42,4pp. **Veredito da hipotese: o
mecanismo estava LA e estava invertido, e corrigi-lo nao fecha a celula.** E aritmetica: 1,08
gusts/jogo e 16,3% dos nossos KOs. Mesmo um gusting perfeito nao vale 44pp.

### 6. O que esta rodada de fato encontrou: a metrica de premios estava quebrada
Investigando por que "KOs/jogo" e "premios que eles tiram/jogo" nao fechavam entre si, achei que
`cell_telemetry` tirava o MINIMO da contagem de premios do oponente sobre TODAS as observacoes --
inclusive as de SETUP, em que os premios ainda nao foram distribuidos e a lista esta vazia. Vazio e
indistinguivel de "tirou os seis", entao a metrica reportava ~6,0 em praticamente todo jogo, real e
interno. **Eu citei esse numero nas rodadas de 31/Jul e 01/Ago** ("os premios que eles tiram nao se
moveram: 5,59 -> 5,48"). Aquela frase nao se sustenta: a metrica nao media o que dizia medir.

Corrigido (so conta depois da primeira leitura nao-vazia) e com uma segunda leitura independente do
mesmo fato (KOs = corpos nossos que somem do board, que nao depende de campo nenhum de setup):

| | real | interno | interno + gust |
|---|---|---|---|
| **premios que ELES tiram/jogo** | **3,51** | **2,13** | **2,40** |
| KOs que ELES nos dao/jogo | 5,49 | 4,02 | 4,24 |

**O gap que a metrica quebrada escondia e enorme: 3,51 contra 2,13, 65% de diferenca.** E a variavel
de saida que faltava -- todas as outras (ataques/jogo, share de Powerful Hand, dano do PH, Rare
Candy, Enhanced Hammer) ja estao casadas com o real, e mesmo assim o oponente interno **converte
quase metade** dos premios. O fix de gusting move 2,13 -> 2,40, na direcao certa e pequeno demais,
exatamente como o winrate.

A conclusao da rodada anterior ("os ataques extras foram desperdicados") continua de pe, mas agora
apoiada nas duas evidencias que nao dependiam da metrica furada -- o nosso winrate SUBIU (80,0 ->
84,3%) e o deck-out deles CAIU (64,7 -> 56,3%) -- e nao mais na frase sobre premios.

**Proximo passo indicado pelos dados**: parar de casar comportamento de entrada e medir a CONVERSAO
-- por que 4,02 KOs viram 2,13 premios internamente e 5,49 viram 3,51 no real. E a unica variavel de
saida ainda descasada, e agora ela tem um medidor confiavel.

- `src/agent_heuristics/heuristic_agent.py`: `gust_targeting=`, `_gust_target_score`,
  `_is_covered`, `prevention_energies()` (tabela derivada do texto do motor)
- `src/environment_wrapper/ab_test.py`: arms `-gust`
- `src/analysis/gust_telemetry.py`: gusting + Enhanced Hammer, real vs interno (NOVO)
- `src/analysis/cell_telemetry.py`: fix da contagem de premios + KOs/jogo
- `tests/test_gust_targeting.py`: tabela de cobertura vinda do motor, escolha sobre opcoes REAIS,
  e lockstep de inercia do ship

## [01/Ago] Auditoria de conversao: um dos dois relogios NAO existia, e o outro tem uma raiz so
Offline, nada shipado. **Nenhum arquivo de agente tocado nesta rodada** -- so analise e testes;
`git diff` contra `src/agent_heuristics/`, `deck.csv`, `main.py` e `build_submission.py` = vazio.
327/329 testes em 3 rodadas (as 2 de `test_portfolio_watch` seguem pre-existentes). 0 exceptions.

`src/analysis/conversion_audit.py` mede tudo a partir do ESTADO DO TABULEIRO, nao de um modelo de
dano: o dano de um ataque e o delta de HP que o motor de fato aplicou no corpo que estava na frente,
entao "prevenido" nao e inferido de uma leitura de regra -- e o ataque que pousou e nao moveu nada.

### 1. O funil dano -> KO
| metrica | REAL (59) | interno (250) | +gust (250) |
|---|---|---|---|
| turnos em que fomos consultados/jogo | 14,12 | 12,07 | 12,62 |
| ataques no nosso ativo/jogo | 6,81 | 5,08 | 4,99 |
| ...com o nosso ativo COBERTO | 27,9% | 28,7% | 24,1% |
| dano efetivo por ataque (HP medido) | 82,11 | 68,88 | 69,25 |
| dano efetivo/jogo | 559 | 350 | 345 |
| ataques que moveram ZERO | 25,9% | 29,8% | 27,5% |
| ...e o alvo estava coberto | **98,1%** | **81,3%** | 77,8% |
| KOs nossos/jogo | 6,24 | 4,36 | 4,41 |
| **KOs por ataque (conversao)** | **91,5%** | **85,8%** | 88,4% |
| Jumbo Ice Cream (nossas curas)/jogo | 0,03 | 0,01 | 0,01 |

**O gap de KO se decompoe exatamente:** volume 5,08/6,81 = 0,75 vezes eficiencia 85,8/91,5 = 0,938,
e 0,75 x 0,938 = 0,70 = 4,36/6,24. **~83% do deficit e VOLUME de ataque, ~17% e eficiencia.**

Duas leituras que fecham portas:
- **A taxa de cobertura no momento do ataque e a MESMA** (27,9% vs 28,7%). Nao e que o oponente real
  desvie melhor da nossa prevencao -- ele encara a mesma parede.
- **Dos ataques que moveram zero, 98,1% dos reais foram em alvo COBERTO** -- ou seja, quase todo
  ataque real desperdicado foi desperdicado PELA NOSSA PREVENCAO. No interno so 81,3%: cerca de 19%
  dos zeros internos bateram num corpo DESCOBERTO e ainda assim nao fizeram nada. Em termos
  absolutos, 5,6% de todos os ataques internos sao puro desperdicio contra 0,5% dos reais.

### 2. O relogio do mill: o gap NAO EXISTE
| | REAL | interno | +gust |
|---|---|---|---|
| **Land Collapse usados/jogo** | 2,46 | 2,23 | 1,73 |
| **>> ...por turno consultado** | **17,4%** | **18,5%** | 13,7% |

**A vazao do mill e identica** (o interno e ate ligeiramente maior). O "3,00 vs 3,50 cartas/turno"
que motivou esta rodada nao era vazao de mill: e o `mill_per_turn` do `cell_telemetry`, que mede o
consumo do deck deles por CICLO -- e o ciclo entre dois turnos NOSSOS contem o turno deles inteiro,
**incluindo o saque deles**. Mesma classe de erro da metrica de premios da rodada passada: um
agregado que parece medir uma coisa e mede outra. Aqui a medida limpa e contar o EVENTO (o ataque
62), que nao admite ambiguidade. Fato que sustenta isso travado em
`tests/test_conversion_audit.py`, dirigindo o motor: todo ataque oferecido pertence ao ATIVO, e o
Land Collapse e exclusivo do Great Tusk.

### 3. A hipotese do uptime: mecanismo CONFIRMADO, consequencia NULA
| | REAL | interno |
|---|---|---|
| share dos nossos turnos com Great Tusk ativo | 38,2% | 46,3% |
| mortes do Great Tusk/jogo | 1,68 | 1,12 |
| turno da 1a morte | 8,77 | 9,70 |
| jogos em que ele chega a morrer | **89,8%** | **68,8%** |

A hipotese estava certa no mecanismo: o real **mata o nosso miller** (89,8% dos jogos contra 68,8%,
1,68 mortes contra 1,12, e mais cedo), e por isso o uptime dele e menor no real -- 38,2% contra
46,3%. **Mas isso nao vira vazao**: com 21% menos uptime, o real dispara o Land Collapse na MESMA
taxa por turno. Logo o gargalo do mill nao e o uptime do Tusk -- e a disponibilidade de ataque
(energia/corpo), o mesmo achado do `attack_census`.

**Por isso NAO implementei o fix do item 3.** A condicao pre-registrada era "se o uptime explicar o
gap de mill"; nao ha gap de mill para explicar, e priorizar matar o miller seria adicionar um botao
que a medicao nao pede. Nada a calibrar nesta rodada.

### 4. O residuo -- e ele tem UMA raiz
Sobra explicar o volume: por que os jogos reais nos dao 14,12 turnos contra 12,07. A resposta esta
na linha medida nos ciclos em que o Great Tusk **nao** estava no ativo, quando o Land Collapse
**nao pode** disparar e portanto o que sai do deck deles e consumo PROPRIO:

| deck deles consumido/ciclo, SEM Tusk ativo | REAL 3,04 | interno 3,82 |
|---|---|---|

**O oponente interno queima o proprio deck 26% mais rapido que o real.** Isso encadeia tudo:

    saque/busca demais -> ele se decka (64,7% interno vs 28,8% real)
      -> jogos mais curtos (12,07 vs 14,12 turnos)
      -> menos ataques (5,08 vs 6,81)
      -> menos KOs (4,36 vs 6,24)
      -> menos premios (2,13 vs 3,51)
      -> a celula infla ~+44pp

E bate com a evidencia independente da rodada da banda: quando a flag `aggro` fez o oponente atacar
mais e jogar menos trainers, o deck-out dele caiu 64,7% -> 56,3% e os jogos alongaram 18,1 -> 23,1
turnos. Mesmo mecanismo, visto de outro angulo.

**Nao e comportamento de ataque, e GESTAO DE RECURSO.** Todas as tres rodadas anteriores mexeram em
variaveis de ataque porque era la que os numeros pareciam estar; o funil agora diz que o ataque esta
casado e o que nao esta e o piloto generico nao saber parar de sacar. Proximo alvo: uma regra de
conservacao de deck no piloto generico (nao buscar/sacar quando o deck esta baixo ou quando a
corrida de deck-out esta perdida) -- que e, ironicamente, a **regra (i) do nosso proprio
CrustleAgent**, o anti-self-mill, que o piloto generico nunca teve.

- `src/analysis/conversion_audit.py`: funil dano->KO por delta de HP, uptime do Tusk, Land Collapse
  por evento, consumo proprio do deck deles (NOVO)
- `tests/test_conversion_audit.py`: Land Collapse e exclusivo do Great Tusk; todo ataque oferecido
  pertence ao ativo (dirigido contra o motor)
