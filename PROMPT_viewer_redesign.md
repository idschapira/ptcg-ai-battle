# Prompt CLI — Redesign do Battle Viewer (estilo batalha Pokémon)

Rodar no Claude Code quando conveniente (tarefa cosmética, fora do caminho crítico do 5A). Colocar
o arquivo de referência `battle_viewer_concept.html` na raiz do repo (ou em `viewer/`) antes de rodar,
para o Claude Code abrir e copiar o estilo.

---

```
Contexto: repositório do agente Kaggle "Pokemon TCG AI Battle". Já existe viewer/battle_viewer.html
— visualizador single-file (HTML+JS puro, offline) que carrega partidas no schema ptcg-devrecord-v1
e navega passo a passo. Ele tem: normalizeRecord() (único ponto que adapta o arquivo de entrada ao
shape interno) e um branch fromKaggleReplay() para o schema cru do kaggle-environments; controles
play/pause/step/scrub/velocidade + setas do teclado; input de arquivo; auto-load de sample_game.json.

Objetivo: REDESENHAR só a camada de APRESENTAÇÃO do viewer para parecer uma mesa de batalha do
Pokémon TCG, mantendo 100% da lógica de dados e dos controles. Há um mockup de referência visual em
./battle_viewer_concept.html (dados hardcoded — copiar a LINGUAGEM VISUAL, não os dados).

NÃO MEXER (preservar intacto):
- normalizeRecord() e fromKaggleReplay() — toda a lógica de parsing/adaptação de schema.
- Os controles existentes (play/pause/step/scrub/velocidade, setas do teclado), o input de arquivo
  e o auto-load do sample_game.json.
- Single-file, offline, SEM CDN/fontes externas (tem que abrir por http://localhost sem rede).
- Tolerância a campos ausentes (None-safe: mão do oponente = só contagem; sem estádio = ocultar faixa).

REDESENHAR (camada visual, casando com battle_viewer_concept.html):
- Layout de mesa TCG: oponente no topo, você embaixo; barra superior com turno/step/resultado +
  controles de playback estilizados.
- Pokémon ATIVO em destaque (moldura dourada no lado que está jogando, neutra no outro): nome (ex
  em vermelho/negrito), barra de HP colorida por razão (verde >50%, amarelo 20–50%, vermelho <20%)
  com "atual / max HP", e energias como PIPS CIRCULARES coloridos por tipo (usar o namespace de
  energia COLORLESS..TEAM_ROCKET; mapa de cor por tipo via CSS vars, como no mockup: W azul, G verde,
  R vermelho, L amarelo, P roxo, F marrom, D grafite, M cinza, C claro).
- Banco como mini-cartas (nome + mini barra de HP). Badges de condição especial (poison/burn/asleep/
  paralyzed/confused) e de Tera. Faixa de estádio quando houver.
- Pilhas de deck/descarte/prizes como cartas empilhadas com contagem (prizes e mão do oponente como
  verso de carta). Sua mão como leque de cartas na base (com "+N" quando exceder o espaço).
- Painel lateral "Decisão do agente" (MANTER — é a feature de debug mais importante): tipo do select
  + nº de opções, lista de opções com o score à direita, a escolhida destacada em verde com ✓, e o
  log de eventos decodificado abaixo. Quando não houver scores (ex.: RandomAgent), mostrar as opções
  sem score.
- Paleta 100% via CSS custom properties no topo (fácil de retema); nada de libs externas.

Data-binding: mapear os campos do ptcg-devrecord-v1 (ativo/banco: id→nome via snapshot, hp/maxHp,
energias, condições; mão própria detalhada, oponente só contagem; prizes/deck/descarte; stadium;
option_summary→lista de opções; scores; índices escolhidos; logs) para os elementos acima. Reusar o
que normalizeRecord() já expõe — não recalcular nada.

Validação: abrir via http.server e confirmar SEM erros de console em três entradas —
(1) sample_game.json, (2) uma gravação de viewer/recordings/, (3) um replay real parseado
(--emit-viewer). Tirar screenshot de cada um. Commit atômico: "feat(viewer): redesign estilo
batalha Pokémon (camada visual, lógica de dados intacta)".
```

---

## Notas
- Tarefa cosmética e independente — pode rodar antes ou depois do 5A, mas como consome tokens do
  Claude Code, faz sentido agrupar com outras tarefas de viewer/UI numa sessão dedicada.
- O `battle_viewer_concept.html` é só referência de estilo; os dados reais vêm do schema do repo.
- Se quiser, dá para adicionar depois: destacar no board a carta afetada por cada log, animação de
  dano, e um "mini-mapa" de prizes restantes por lado.
