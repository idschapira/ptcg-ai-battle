# Task 4.4e — Coleta diária automatizada de replays dos líderes

## Onde entra no roadmap / backlog
- **Epic 4.4 (Ingestão de Replays) → nova Task 4.4e.** Infra de background: configurar uma vez,
  roda sozinha até 16/Ago acumulando um corpus dos melhores agentes.
- **Consumidores:** 4.4c (warm-start por imitação, Sprint 5) e 4.5d (gauntlet de decks reais para
  o deck building, Sprints 5–6). Quanto antes começar, maior e mais diverso o corpus.
- **Quando configurar:** agora (fora do caminho crítico). O agendamento roda na SUA máquina — a
  tarefa programada do Cowork não serve (não tem suas credenciais Kaggle, nem o repo, nem o script).

## Defaults embutidos
- Puxar a data de **ontem** (a de hoje pode não estar publicada).
- **Top ~100 episódios/dia** (não os ~21 GB do dia inteiro).
- **Apagar os replays brutos após parsear** — guarda só o `.npz` (senão acumula GBs).
- **Um dataset por data** (`replay_dataset_<data>.npz`) — acumula em vez de sobrescrever.
- Idempotente (pula datas já processadas) e None-safe (se o dia não estiver publicado, loga e sai 0).

---

## Parte 1 — Prompt CLI (rodar no Claude Code)

```
Contexto: repositório do agente Kaggle "Pokemon TCG AI Battle". Já existem
src/ingestion/replays_download.py e replays_parse.py (parse com pareamento ação↔observação já
corrigido e coberto por teste de regressão). Hoje o parse grava um único data/processed/
replay_dataset.npz (sobrescreve). O download/parse roda no venv do projeto.

Objetivo: habilitar ACUMULAÇÃO diária de replays dos líderes, para montar um corpus ao longo das
semanas (alimenta o warm-start por imitação e o gauntlet de deck building). Task 4.4e.

Tarefas:
1. replays_parse.py: suportar saída POR DATA. Novo argumento --out (ou --date) que grave em
   data/processed/replays/replay_dataset_<date>.npz em vez de sobrescrever o arquivo único.
   Manter o comportamento antigo como default para não quebrar nada.
2. (opcional, útil p/ Sprint 5) função/utilitário merge que faça glob de
   data/processed/replays/replay_dataset_*.npz e concatene num dataset de treino único
   (states, options_flat, option_counts, labels, episode_ids) — ou documentar que a Sprint 5
   fará o glob no carregamento.
3. Criar scripts/daily_replays.sh (bash), tipado nos passos e defensivo:
   a. Ativar o venv do projeto (usar o python/venv correto deste repo).
   b. Calcular a data de ONTEM (YYYY-MM-DD).
   c. Se data/processed/replays/replay_dataset_<ontem>.npz já existe -> logar "já processado" e
      sair 0 (idempotente).
   d. Rodar: python -m src.ingestion.replays_download --date <ontem> --max-episodes 100
      Se a data não estiver publicada / auth falhar -> logar e sair 0 (sem quebrar).
   e. Rodar: python -m src.ingestion.replays_parse --date <ontem> --sides winner
      (gravando o .npz por data).
   f. APAGAR os replays brutos daquela data em data/raw/replays/ após o parse OK (guardar só o npz).
   g. Anexar uma linha de resumo a data/processed/replays/daily.log (data, nº de pares, cobertura,
      overflows de MAX_OPTIONS, tamanho do npz).
4. .gitignore: garantir que data/processed/replays/ (npz + log) e data/raw/replays/ estejam
   ignorados.
5. Validação: rodar scripts/daily_replays.sh uma vez manualmente (dry para uma data publicada,
   ex. 2026-07-05) e me mostrar a linha do daily.log e o tamanho do npz gerado. Rodar de novo e
   confirmar que ele detecta "já processado" e sai sem rebaixar. Commit atômico:
   "feat(replays): coleta diária acumulativa por data (Task 4.4e)".
```

## Parte 2 — Agendamento (Windows Task Scheduler — recomendado)

O cron do WSL2 não persiste quando o terminal fecha; o Task Scheduler do Windows dispara o WSL
mesmo fechado. Passos:

1. Descobrir o nome da distro: no PowerShell, `wsl -l -v` (ex.: `Ubuntu`).
2. Abrir **Task Scheduler** → **Create Task**.
   - *General:* nome "PTCG replays diários"; marcar "Run whether user is logged on or not".
   - *Triggers:* New → Daily → 08:00.
   - *Actions:* New → Program/script: `wsl.exe`
     Argumentos:
     `-d Ubuntu bash -lc "cd /mnt/c/Users/ilans/Claude/Projects/PTCG-AI-Battle-Challenge-Simulation && bash scripts/daily_replays.sh"`
   - *Conditions:* desmarcar "Start only if on AC power" se for notebook.
3. Testar: botão direito na tarefa → **Run**, e conferir a linha nova em
   `data/processed/replays/daily.log`.

### Alternativa — cron no WSL (menos confiável)
```bash
sudo service cron start          # precisa rodar a cada boot do WSL
crontab -e
# adicionar:
0 8 * * * cd /mnt/c/Users/ilans/Claude/Projects/PTCG-AI-Battle-Challenge-Simulation && bash scripts/daily_replays.sh >> data/processed/replays/daily.log 2>&1
```
Só funciona enquanto houver um processo WSL vivo — por isso o Task Scheduler é preferível.

---

## Notas
- Volume: 100 episódios/dia × ~2 MB ≈ 200 MB brutos/dia, mas são apagados após o parse; sobra só o
  npz (dezenas/centenas de KB por dia). Sem risco de encher o disco.
- Revisar o corpus antes da Sprint 5: `wc -l data/processed/replays/daily.log` e somar os pares.
- Se algum dia registrar overflow de MAX_OPTIONS>0 no log, me avisa — é sinal de deck real
  estourando o cap de 64 (o teste que a gente monitorava).
