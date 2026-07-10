#!/usr/bin/env bash
# Task 4.4e — coleta diária acumulativa de replays dos líderes (WSL/cron).
#
# Baixa os top-100 episódios de ONTEM, parseia (--sides winner) para
# data/processed/replays/replay_dataset_<data>.npz, apaga os brutos,
# refaz o replay_corpus.npz e registra 1 linha em daily.log.
#
# Idempotente: se o npz do dia já existe, sai 0 sem tocar em nada.
# Falhas ESPERADAS (dia ainda não publicado, sem credencial) saem 0 para
# não quebrar o agendamento; falhas reais (parse/merge) saem 1.
#
# Agendamento (Windows Task Scheduler -> wsl.exe):
#   -d Ubuntu bash -lc "cd /mnt/c/Users/ilans/Claude/Projects/PTCG-AI-Battle-Challenge-Simulation && bash scripts/daily_replays.sh"

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT" || { echo "[daily_replays] ERRO: repo root inacessível: $REPO_ROOT"; exit 1; }

log() { echo "[daily_replays $(date +%FT%T)] $*"; }

# a. venv obrigatório (o python do sistema não tem polars/numpy/kaggle)
if [ ! -f ".venv/bin/activate" ]; then
    log "ERRO: .venv não encontrado em $REPO_ROOT/.venv — crie o venv antes"
    exit 1
fi
# shellcheck disable=SC1091
source ".venv/bin/activate"

# b. credencial Kaggle: ausência não é erro do agendamento
if [ ! -f "$HOME/.kaggle/kaggle.json" ] \
        && { [ -z "${KAGGLE_USERNAME:-}" ] || [ -z "${KAGGLE_KEY:-}" ]; }; then
    log "sem credencial Kaggle (~/.kaggle/kaggle.json ou KAGGLE_USERNAME/KAGGLE_KEY) — pulando"
    exit 0
fi

# c. alvo = ontem
TARGET_DATE="$(date -d "yesterday" +%F)"
OUT_DIR="data/processed/replays"
OUT_NPZ="$OUT_DIR/replay_dataset_${TARGET_DATE}.npz"
META_JSON="$OUT_DIR/replay_dataset_${TARGET_DATE}.meta.json"
DAILY_LOG="$OUT_DIR/daily.log"
RAW_DIR="data/raw/replays/${TARGET_DATE}"
mkdir -p "$OUT_DIR"

# d. idempotência
if [ -f "$OUT_NPZ" ]; then
    log "já processado: $OUT_NPZ — nada a fazer"
    exit 0
fi

# e. download (dia não publicado / rede / auth -> tenta amanhã, exit 0)
log "download ${TARGET_DATE} (top 100 episódios)"
if ! python -m src.ingestion.replays_download --date "$TARGET_DATE" --max-episodes 100; then
    log "download de ${TARGET_DATE} falhou (dia não publicado ainda?) — pulando sem erro"
    exit 0
fi

# f. parse (falha aqui é erro real; raw fica para diagnóstico)
log "parse ${TARGET_DATE} (--sides winner)"
if ! python -m src.ingestion.replays_parse --date "$TARGET_DATE" --sides winner; then
    log "ERRO: parse de ${TARGET_DATE} falhou — raw mantido em $RAW_DIR"
    exit 1
fi

# g. só o npz fica; brutos são reproduzíveis via download
rm -rf "$RAW_DIR"
log "raw apagado: $RAW_DIR"

# h. corpus acumulado
log "merge -> replay_corpus.npz"
if ! python -m src.ingestion.replays_merge; then
    log "ERRO: merge falhou"
    exit 1
fi

# i. 1 linha no daily.log (pares, cobertura, overflows, tamanho).
#    Overflow ocasional de MAX_OPTIONS=64 é ESPERADO/benigno (clamp
#    seguro, visto no CP4b) — só investigar se aparecer em volume.
LINE="$(python - "$TARGET_DATE" "$META_JSON" "$OUT_NPZ" <<'PY'
import json
import os
import sys

date, meta_path, npz_path = sys.argv[1:4]
try:
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
except OSError:
    meta = {}
coverage = meta.get("coverage")
if coverage is not None:
    head = (f"{date} games={meta.get('games', '?')} "
            f"pairs={meta.get('decision_pairs', '?')} coverage={coverage:.4%}")
else:
    head = f"{date} meta ausente"
print(f"{head} overflows={meta.get('overflow_warnings', '?')} "
      f"npz_bytes={os.path.getsize(npz_path)}")
PY
)"
echo "$LINE" >> "$DAILY_LOG"
log "daily.log += $LINE"
exit 0
