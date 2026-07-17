#!/usr/bin/env bash
# Portfolio watch diário (roda DEPOIS de daily_replays.sh + track_elo.sh
# no mesmo job): avalia alertas sobre radar_history.csv + elo_log.csv e
# escreve data/portfolio_watch.md. Não precisa de raw nem de rede.
#
# Idempotente e defensivo (padrão daily_replays): falta de venv sai 0
# (não quebra o agendamento); falha real do watch sai 1.
#
# Agendamento (Windows Task Scheduler -> wsl.exe, ação 3 do job diário):
#   -d Ubuntu bash -lc "cd /mnt/c/Users/ilans/Claude/Projects/PTCG-AI-Battle-Challenge-Simulation && bash scripts/portfolio_watch.sh"

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || { echo "[portfolio_watch] ERRO: repo root inacessível"; exit 1; }

log() { echo "[portfolio_watch $(date +%FT%T)] $*"; }

if [ ! -f ".venv/bin/activate" ]; then
    log "sem .venv — pulando (agendamento segue vivo)"
    exit 0
fi
# shellcheck disable=SC1091
source ".venv/bin/activate"

# harvest de segurança: se algum dia de raw ainda existe (ex.: baixado à
# mão), colhe antes do watch — idempotente, sem raw presente é no-op.
if ! python -m src.analysis.portfolio_watch --harvest-all; then
    log "AVISO: harvest-all falhou (segue para o watch mesmo assim)"
fi

if ! python -m src.analysis.portfolio_watch; then
    log "ERRO: watch falhou"
    exit 1
fi
log "watch OK -> data/portfolio_watch.md"
exit 0
