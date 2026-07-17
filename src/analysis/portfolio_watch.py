"""Portfolio watch diário: radar persistente + alertas + resumo legível.

Com os 2 finais no ladder (Final A = Crustle-e10 V4, Final B = Spidops
BC v2), a rotina diária cobre três perguntas: como vão os ELOs, como vai
o meta, e algum watch-item disparou?

Modos (CLI):
    --harvest YYYY-MM-DD   colhe o radar do dia a partir de
                           data/raw/replays/<dia> e APPENDA em
                           data/processed/portfolio/radar_history.csv
                           (idempotente por dia). Chamado pelo
                           daily_replays.sh ANTES de apagar o raw — o
                           raw é efêmero, a série é persistente.
    --harvest-all          idem para todo dia presente no raw (backfill).
    (default)              watch: lê radar_history.csv + data/elo/
                           elo_log.csv, avalia os alertas e escreve
                           data/portfolio_watch.md. Sem raw necessário.

Alertas (limiares em WatchConfig, começar conservador; sem alerta =
linha silenciosa "OK"):
  (a) classe ignore-effects (decks com atacante que fura prevenção de
      dano, o early-warning do Starmie/muro) com share sustentado;
  (b) arquétipo fora da cobertura da dupla acima do limiar e crescendo;
  (c) shift num pilar: Alakazam colapsando (tira a força do Final B) ou
      counter de Spidops (C+Kangaskhan/stall) inflando.
  (d) guarda de eviction: um final com publicScore estático por N coletas
      enquanto outro ativo se move (pode ter parado de jogar).

A detecção da classe ignore-effects é DINÂMICA (varre dim_attack por
texto de efeito), então cartas novas do pool entram sozinhas. O próprio
Crustle (Superb Scissors) pertence à classe mas é EXCLUÍDO da contagem:
ele é o muro dos stalls já medidos e contaminaria o sinal com 12-20% de
share permanente (fadiga de alarme); um deck novo abusando dele como
anti-muro dispara o alerta de arquétipo não-coberto.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Final, Iterable

from ..ingestion.build_card_model import PROCESSED_DIR, REPO_ROOT
from ..ingestion.card_index import CardIndex
from ..ingestion.replays_download import REPLAYS_DIR
from .meta_radar import UNKNOWN, extract_decks, label_archetype

logger = logging.getLogger(__name__)

PORTFOLIO_DIR: Final[Path] = PROCESSED_DIR / "portfolio"
RADAR_HISTORY: Final[Path] = PORTFOLIO_DIR / "radar_history.csv"
ELO_LOG: Final[Path] = REPO_ROOT / "data" / "elo" / "elo_log.csv"
WATCH_MD: Final[Path] = REPO_ROOT / "data" / "portfolio_watch.md"

_HISTORY_COLUMNS: Final[tuple[str, ...]] = (
    "day", "archetype", "n_decks", "n_ignore_fx")
_DAY_RE: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# refs das submissões que compõem o portfólio (elo_log usa ref do CLI)
FINALS: Final[dict[str, str]] = {
    "54667957": "Final A (Crustle e10, V4)",
    "54791820": "Final B (Spidops BC v2)",
}
ACTIVE_REFS: Final[tuple[str, ...]] = ("54619473", "54667957", "54791820")

# arquétipos que a dupla A+B cobre (medidos no gauntlet honesto) — o
# buraco conhecido (Starmie) tem alerta próprio via ignore-effects.
COVERED_ARCHETYPES: Final[frozenset[str]] = frozenset({
    "Alakazam box (non-ex)",
    "Team Rocket Spidops (non-ex)",
    "Crustle mill (ours)",
    "Crustle + Mega Kangaskhan stall",
    "Crustle stall (other)",
    "Mega Starmie / Mega Froslass",  # coberto pelo alerta (a), não aqui
})


@dataclass(frozen=True)
class WatchConfig:
    """Limiares dos alertas (conservadores de propósito)."""

    ignore_fx_share: float = 0.05      # (a) share da classe por dia
    ignore_fx_days: int = 2            # (a) dias consecutivos acima
    uncovered_share: float = 0.10      # (b) share do arquétipo não-coberto
    alakazam_floor: float = 0.45       # (c) média 3d abaixo disso = colapso
    counter_ceiling: float = 0.25      # (c) média 3d de C+K acima disso
    static_collections: int = 3        # (d) coletas com score estático


# ------------------------------------------------------------------ #
# classe ignore-effects (dinâmica, a partir do modelo de dados)
# ------------------------------------------------------------------ #


def ignore_effects_card_names() -> frozenset[str]:
    """Nomes de cartas com ataque que ignora efeitos no Pokémon ativo
    adversário (a via que fura o muro do Crustle). Varredura por texto —
    o pool é fixo por temporada, mas a lista se atualiza sozinha se o
    modelo de dados for reconstruído com cartas novas."""
    import polars as pl

    att = pl.read_parquet(PROCESSED_DIR / "dim_attack.parquet")
    card = pl.read_parquet(PROCESSED_DIR / "dim_card.parquet")
    pierce = att.filter(
        pl.col("effect").str.contains(r"(?i)effects on your opponent")
        & pl.col("effect").str.contains(r"(?i)isn.t affected|not affected"))
    joined = pierce.join(card, on="card_id", how="left")
    return frozenset(joined.get_column("card_name").drop_nulls().to_list())


# ------------------------------------------------------------------ #
# harvest (chamado com o raw do dia ainda em disco)
# ------------------------------------------------------------------ #


def _harvested_days() -> set[str]:
    if not RADAR_HISTORY.exists():
        return set()
    with open(RADAR_HISTORY, newline="", encoding="utf-8") as fh:
        return {row["day"] for row in csv.DictReader(fh)}


def harvest_day(day: str, index: CardIndex | None = None) -> int:
    """Colhe o radar de data/raw/replays/<day> para o histórico.

    Idempotente: dia já colhido -> 0 linhas novas. Retorna o nº de decks
    observados (0 também quando o raw do dia não existe)."""
    day_dir = REPLAYS_DIR / day
    if day in _harvested_days():
        logger.info("dia %s já colhido — nada a fazer", day)
        return 0
    if not day_dir.is_dir():
        logger.warning("raw ausente para %s (%s)", day, day_dir)
        return 0
    index = index if index is not None else CardIndex()
    # ver docstring do módulo: o muro em si não conta para a classe
    pierce_names = ignore_effects_card_names() - {"Crustle"}

    counts: Counter[str] = Counter()
    ignore_fx: Counter[str] = Counter()
    n_decks = 0
    for path in sorted(day_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                replay = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("replay ilegível %s: %s", path.name, exc)
            continue
        for deck in extract_decks(replay, day, index):
            if deck.is_ours:
                continue
            n_decks += 1
            counts[deck.archetype] += 1
            if any(name in pierce_names for name in deck.copies_by_name):
                ignore_fx[deck.archetype] += 1
    if n_decks == 0:
        logger.warning("nenhum deck observado em %s", day_dir)
        return 0

    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    is_new = not RADAR_HISTORY.exists()
    with open(RADAR_HISTORY, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(_HISTORY_COLUMNS)
        for archetype, n in sorted(counts.items()):
            writer.writerow([day, archetype, n, ignore_fx.get(archetype, 0)])
    logger.info("%s: %d decks, %d arquétipos -> %s",
                day, n_decks, len(counts), RADAR_HISTORY.name)
    return n_decks


def harvest_all() -> int:
    """Backfill: colhe todo dia presente no raw (idempotente)."""
    index = CardIndex()
    total = 0
    for day_dir in sorted(REPLAYS_DIR.iterdir()) if REPLAYS_DIR.exists() else []:
        if day_dir.is_dir() and _DAY_RE.match(day_dir.name):
            total += harvest_day(day_dir.name, index)
    return total


# ------------------------------------------------------------------ #
# alertas (funções puras sobre a série — unit-testáveis)
# ------------------------------------------------------------------ #


@dataclass
class DaySnapshot:
    """Radar de um dia: share por arquétipo + share da classe pierce."""

    day: str
    total: int
    shares: dict[str, float]
    ignore_fx_share: float


def load_history(path: Path = RADAR_HISTORY) -> list[DaySnapshot]:
    if not path.exists():
        return []
    per_day: dict[str, list[dict]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            per_day[row["day"]].append(row)
    snapshots = []
    for day in sorted(per_day):
        rows = per_day[day]
        total = sum(int(r["n_decks"]) for r in rows)
        if total == 0:
            continue
        snapshots.append(DaySnapshot(
            day=day,
            total=total,
            shares={r["archetype"]: int(r["n_decks"]) / total for r in rows},
            ignore_fx_share=sum(int(r["n_ignore_fx"]) for r in rows) / total,
        ))
    return snapshots


def _tail_mean(snapshots: list[DaySnapshot], archetype: str,
               n_days: int) -> float:
    tail = snapshots[-n_days:]
    if not tail:
        return 0.0
    return sum(s.shares.get(archetype, 0.0) for s in tail) / len(tail)


def evaluate_alerts(snapshots: list[DaySnapshot],
                    config: WatchConfig = WatchConfig()) -> list[str]:
    """Lista de alertas disparados (vazia = tudo OK)."""
    alerts: list[str] = []
    if not snapshots:
        return ["radar_history vazio — harvest nunca rodou?"]

    # (a) classe ignore-effects sustentada
    tail = snapshots[-config.ignore_fx_days:]
    if (len(tail) >= config.ignore_fx_days
            and all(s.ignore_fx_share >= config.ignore_fx_share for s in tail)):
        series = ", ".join(f"{s.day[5:]}={s.ignore_fx_share:.0%}" for s in tail)
        alerts.append(
            f"IGNORE-EFFECTS: classe fura-muro com share >= "
            f"{config.ignore_fx_share:.0%} por {len(tail)} dias ({series}) — "
            f"early-warning do buraco Starmie (nenhum final cobre)")

    # (b) arquétipo não-coberto acima do limiar e crescendo
    latest = snapshots[-1]
    previous = snapshots[-2] if len(snapshots) >= 2 else None
    for archetype, share in sorted(latest.shares.items(), key=lambda kv: -kv[1]):
        if archetype in COVERED_ARCHETYPES or archetype == UNKNOWN:
            continue
        growing = previous is None or share > previous.shares.get(archetype, 0.0)
        if share >= config.uncovered_share and growing:
            alerts.append(
                f"NAO-COBERTO: '{archetype}' em {share:.0%} no dia "
                f"{latest.day} e crescendo — fora da dupla A+B")

    # (c) pilares: Alakazam colapsando / counter de Spidops inflando
    alakazam = _tail_mean(snapshots, "Alakazam box (non-ex)", 3)
    if alakazam < config.alakazam_floor:
        alerts.append(
            f"PILAR: Alakazam em {alakazam:.0%} (média 3d) < "
            f"{config.alakazam_floor:.0%} — a vantagem do Final B "
            f"(76,7% nesse matchup) perde peso")
    counter = _tail_mean(snapshots, "Crustle + Mega Kangaskhan stall", 3)
    if counter > config.counter_ceiling:
        alerts.append(
            f"PILAR: Crustle+Kangaskhan em {counter:.0%} (média 3d) > "
            f"{config.counter_ceiling:.0%} — counter do Spidops-BC "
            f"(41,4% nesse matchup) inflando")
    return alerts


# ------------------------------------------------------------------ #
# ELO (série do tracker) + guarda de eviction
# ------------------------------------------------------------------ #


@dataclass
class EloSeries:
    """publicScore por data de coleta, por ref (+ topo do leaderboard)."""

    by_ref: dict[str, list[tuple[str, float]]] = field(default_factory=dict)
    top: list[tuple[str, float]] = field(default_factory=list)


def load_elo(path: Path = ELO_LOG) -> EloSeries:
    series = EloSeries()
    if not path.exists():
        return series
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            score_text = (row.get("public_score") or "").strip()
            try:
                score = float(score_text)
            except ValueError:
                continue
            day = row.get("collect_date") or ""
            ref = (row.get("ref") or "").strip()
            if ref == "leaderboard":
                series.top.append((day, score))
            else:
                series.by_ref.setdefault(ref, []).append((day, score))
    return series


def eviction_guard(series: EloSeries,
                   config: WatchConfig = WatchConfig()) -> list[str]:
    """(d) score de um final estático por N coletas enquanto outro ativo
    se move — proxy barato de 'parou de receber episódios/evictado'."""
    alerts: list[str] = []
    n = config.static_collections
    moved_any = False
    static_finals: list[str] = []
    for ref in ACTIVE_REFS:
        points = series.by_ref.get(ref, [])[-n:]
        if len(points) < n:
            continue  # série curta demais para acusar
        scores = [score for _, score in points]
        if max(scores) != min(scores):
            moved_any = True
        elif ref in FINALS:
            static_finals.append(ref)
    if moved_any:
        for ref in static_finals:
            alerts.append(
                f"GUARDA: {FINALS[ref]} (ref {ref}) com score estático há "
                f"{n} coletas enquanto outra ativa se move — conferir "
                f"eviction/episódios no Kaggle")
    return alerts


# ------------------------------------------------------------------ #
# resumo legível
# ------------------------------------------------------------------ #


def _fmt_trend(points: list[tuple[str, float]]) -> str:
    if not points:
        return "sem coleta"
    day, score = points[-1]
    if len(points) >= 2:
        delta = score - points[-2][1]
        return f"{score:.1f} ({delta:+.1f} vs coleta anterior, {day})"
    return f"{score:.1f} (primeira coleta, {day})"


def render_watch(snapshots: list[DaySnapshot], series: EloSeries,
                 alerts: list[str]) -> str:
    lines = [f"# Portfolio watch — {date.today().isoformat()}", ""]

    lines.append("## Finais no ladder")
    top_txt = _fmt_trend(series.top)
    for ref, label in FINALS.items():
        points = series.by_ref.get(ref, [])
        gap = ""
        if points and series.top:
            gap = f" | gap pro topo: {series.top[-1][1] - points[-1][1]:.1f}"
        lines.append(f"- **{label}**: {_fmt_trend(points)}{gap}")
    lines.append(f"- topo do leaderboard: {top_txt}")
    lines.append("")

    lines.append("## Radar (share por dia, últimos 5)")
    tail = snapshots[-5:]
    if tail:
        archetypes = Counter()
        for snap in tail:
            for archetype, share in snap.shares.items():
                archetypes[archetype] += share
        header = "| arquétipo | " + " | ".join(s.day[5:] for s in tail) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(tail) + 1))
        for archetype, _ in archetypes.most_common(6):
            cells = " | ".join(f"{s.shares.get(archetype, 0.0):.0%}"
                               for s in tail)
            lines.append(f"| {archetype} | {cells} |")
        cells = " | ".join(f"{s.ignore_fx_share:.0%}" for s in tail)
        lines.append(f"| _classe ignore-effects_ | {cells} |")
    else:
        lines.append("(sem histórico de radar)")
    lines.append("")

    lines.append("## Alertas")
    if alerts:
        lines.extend(f"- ⚠️ {alert}" for alert in alerts)
    else:
        lines.append("- OK: nenhum watch-item disparado")
    lines.append("")
    return "\n".join(lines)


def run_watch(config: WatchConfig = WatchConfig()) -> int:
    snapshots = load_history()
    series = load_elo()
    alerts = evaluate_alerts(snapshots, config) + eviction_guard(series, config)
    WATCH_MD.parent.mkdir(parents=True, exist_ok=True)
    WATCH_MD.write_text(render_watch(snapshots, series, alerts),
                        encoding="utf-8")
    status = "ALERTAS: " + " || ".join(alerts) if alerts else "OK (sem alertas)"
    print(f"portfolio watch -> {WATCH_MD.relative_to(REPO_ROOT)} | {status}")
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harvest", metavar="YYYY-MM-DD", default=None,
                        help="colhe o radar do dia a partir do raw")
    parser.add_argument("--harvest-all", action="store_true",
                        help="backfill de todos os dias presentes no raw")
    args = parser.parse_args()

    if args.harvest is not None:
        if not _DAY_RE.match(args.harvest):
            raise SystemExit(f"dia inválido: {args.harvest}")
        harvest_day(args.harvest)
    elif args.harvest_all:
        harvest_all()
    else:
        run_watch()


if __name__ == "__main__":
    main()
