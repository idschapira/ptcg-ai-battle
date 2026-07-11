#!/usr/bin/env bash
# Tracker diário de ELO das submissões (publicScore) + topo do leaderboard.
#
# Anexa a data/elo/elo_log.csv uma linha por submissão COMPLETE (com
# timestamp da coleta) e 1 linha-resumo a data/elo/elo.log. Idempotente
# por dia: se já coletou hoje, sai 0 sem tocar em nada. Falta de venv ou
# credencial também sai 0 (não quebra o agendamento).
#
# Agendamento (Windows Task Scheduler -> wsl.exe), mesmo padrão do
# coletor de replays:
#   -d Ubuntu bash -lc "cd /mnt/c/Users/ilans/Claude/Projects/PTCG-AI-Battle-Challenge-Simulation && bash scripts/track_elo.sh"

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || { echo "[track_elo] ERRO: repo root inacessível"; exit 1; }

log() { echo "[track_elo $(date +%FT%T)] $*"; }

if [ ! -f ".venv/bin/activate" ]; then
    log "sem .venv — pulando (agendamento segue vivo)"
    exit 0
fi
# shellcheck disable=SC1091
source ".venv/bin/activate"

if [ ! -f "$HOME/.kaggle/kaggle.json" ] \
        && { [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; }; then
    log "sem credencial Kaggle — pulando"
    exit 0
fi

TODAY="$(date +%F)"
ELO_DIR="data/elo"
CSV_PATH="$ELO_DIR/elo_log.csv"
LOG_PATH="$ELO_DIR/elo.log"
mkdir -p "$ELO_DIR"

if [ -f "$CSV_PATH" ] && grep -q "^${TODAY}," "$CSV_PATH"; then
    log "já coletado hoje ($TODAY) — nada a fazer"
    exit 0
fi

python - "$TODAY" "$CSV_PATH" "$LOG_PATH" <<'PY'
import csv
import os
import subprocess
import sys
from datetime import datetime

today, csv_path, log_path = sys.argv[1:4]
COMPETITION = "pokemon-tcg-ai-battle"


def run_kaggle(*args: str) -> str:
    """CLI stdout; the kaggle CLI is noisy on stderr and may exit non-zero
    even on success — judge by parseable output, not the return code."""
    try:
        result = subprocess.run([sys.executable, "-m", "kaggle", *args],
                                capture_output=True, text=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"kaggle {' '.join(args)} falhou: {exc}", file=sys.stderr)
        return ""
    return result.stdout or ""


def parse_cli_csv(text: str, header_prefix: str) -> list[dict]:
    """Parse the CLI's csv, skipping banner/pagination junk lines."""
    lines = [line for line in text.splitlines() if line.strip()]
    for i, line in enumerate(lines):
        if line.startswith(header_prefix):
            return list(csv.DictReader(lines[i:]))
    return []


subs = parse_cli_csv(
    run_kaggle("competitions", "submissions", "-c", COMPETITION, "--csv"),
    "ref,")
if not subs:
    print("ERRO: nenhuma submissão parseada (auth/rede?)", file=sys.stderr)
    sys.exit(1)

top_score = ""
board = parse_cli_csv(
    run_kaggle("competitions", "leaderboard", "-c", COMPETITION,
               "--show", "--csv"),
    "teamId,")
if board:  # first data row is the current #1
    top_score = (board[0].get("score") or "").strip()

complete = [s for s in subs if "COMPLETE" in (s.get("status") or "")]
now = datetime.now().isoformat(timespec="seconds")

is_new = not os.path.exists(csv_path)
with open(csv_path, "a", newline="", encoding="utf-8") as fh:
    writer = csv.writer(fh)
    if is_new:
        writer.writerow(["collect_date", "collect_ts", "ref", "description",
                         "submit_date", "status", "public_score"])
    for sub in complete:
        writer.writerow([today, now, sub.get("ref", ""),
                         sub.get("description", ""), sub.get("date", ""),
                         "COMPLETE", sub.get("publicScore", "")])
    if top_score:
        writer.writerow([today, now, "leaderboard", "TOP", "", "TOP",
                         top_score])

parts = [f"{s.get('description') or s.get('ref')}="
         f"{s.get('publicScore') or '?'}" for s in complete]
summary = f"{today} " + " | ".join(parts)
if top_score:
    summary += f" | top={top_score}"
with open(log_path, "a", encoding="utf-8") as fh:
    fh.write(summary + "\n")
print(summary)
PY
status=$?
if [ $status -ne 0 ]; then
    log "coleta falhou (exit $status)"
    exit 1
fi
log "coleta OK -> $CSV_PATH"
exit 0
