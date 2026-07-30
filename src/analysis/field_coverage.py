"""What share of the REAL field do our measured cells actually cover?

Archaludon was 14.2% of our ladder games and had no decklist, so the
gauntlet could not test it and nobody noticed — a cell you cannot measure
looks exactly like a cell that is fine. This module makes that failure
mode visible on purpose: it tabulates the opponent archetypes we actually
faced, our real winrate against each, and whether a cell exists.

"Covered" means BOTH of:
  * an entry in archetype_rules.ARCHETYPE_DECKS (the label maps to a
    decklist), and
  * that file existing on disk and surviving the field sweep.
A label with no decklist is reported as a hole, sized by its real share,
so the next reconstruction is chosen by evidence and not by whim.

Priority for closing a hole = share x (1 - real winrate): a big slice we
already lose is worth more than a big slice we already win.

Sample discipline: submission-id filter + deck sentinel + fixed seat.

Run from the repo root:
    python -m src.analysis.field_coverage
    python -m src.analysis.field_coverage --json out.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..deckbuilding.archetype_rules import ARCHETYPE_DECKS, label_archetype
from ..deckbuilding.gauntlet import discover_decks
from ..environment_wrapper.ab_test import wilson_interval
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.card_index import CardIndex
from .fetch_my_episodes import EPISODES_DIR
from .meta_radar import observed_serials as _observed_serials

OUR_TEAM: Final[str] = "Ilan Schapira"
OUR_DECK_LABEL: Final[str] = "Crustle mill (ours)"
SUB_INDEX: Final[Path] = (REPO_ROOT / "data" / "processed" /
                          "episodes_index" / "sub_54917180.json")
HOLE_THRESHOLD: Final[float] = 0.05


@dataclass
class Cell:
    archetype: str
    games: int = 0
    wins: int = 0

    @property
    def winrate(self) -> float | None:
        return self.wins / self.games if self.games else None


def survey(episodes_dir: Path, team: str, allowed: set[int] | None,
           index: CardIndex) -> tuple[dict[str, Cell], int]:
    cells: dict[str, Cell] = {}
    total = 0
    for path in sorted(episodes_dir.glob("*.json")):
        stem = os.path.splitext(path.name)[0]
        if allowed is not None and stem.isdigit() and int(stem) not in allowed:
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                replay = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        names = ((replay.get("info") or {}).get("TeamNames")) or []
        if team not in names:
            continue
        seat = names.index(team)
        revealed: dict[int, Counter] = {0: Counter(), 1: Counter()}
        for (player, _s), card_id in _observed_serials(replay).items():
            card = index.get_card(card_id)
            if card is not None and player in revealed:
                revealed[player][card.card_name] += 1
        if label_archetype(revealed[seat]) != OUR_DECK_LABEL:
            continue
        rewards = replay.get("rewards") or []
        if len(rewards) != 2 or rewards[0] is None or rewards[0] == rewards[1]:
            continue
        label = label_archetype(revealed[1 - seat])
        cell = cells.setdefault(label, Cell(label))
        cell.games += 1
        total += 1
        if rewards[seat] > rewards[1 - seat]:
            cell.wins += 1
    return cells, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes-dir", type=Path, default=EPISODES_DIR)
    parser.add_argument("--team", type=str, default=OUR_TEAM)
    parser.add_argument("--submission-index", type=Path, default=SUB_INDEX)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    index = CardIndex()
    allowed = None
    if args.submission_index.exists():
        with open(args.submission_index, encoding="utf-8") as fh:
            allowed = {int(i) for i in
                       (json.load(fh).get("episode_ids") or [])}

    cells, total = survey(args.episodes_dir, args.team, allowed, index)
    if not total:
        raise SystemExit("nenhum episódio utilizável")
    field = discover_decks()
    on_disk = {path.name for path in field.values()}

    def covered(label: str) -> tuple[bool, str]:
        rel = ARCHETYPE_DECKS.get(label)
        if rel is None:
            return False, "sem decklist mapeada"
        name = Path(rel).name
        if name not in on_disk:
            return False, f"mapeada para {name}, AUSENTE do campo"
        return True, name

    print("=" * 100)
    print(f"COBERTURA DO CAMPO REAL — {total} jogos decididos "
          f"(submissão 54917180 + sentinela de deck)")
    print("=" * 100)
    print(f"  {'arquétipo':34s} {'jogos':>6s} {'share':>7s} {'WR real':>9s} "
          f"{'IC95':>16s}  {'célula':>8s}  decklist")
    covered_share = 0.0
    rows = []
    for label, cell in sorted(cells.items(), key=lambda kv: -kv[1].games):
        share = cell.games / total
        ok, note = covered(label)
        if ok:
            covered_share += share
        lo, hi = wilson_interval(cell.wins, cell.games)
        print(f"  {label[:34]:34s} {cell.games:6d} {share:7.1%} "
              f"{cell.winrate:9.1%} [{lo:5.1%},{hi:5.1%}]  "
              f"{'SIM' if ok else 'NÃO':>8s}  {note}")
        rows.append({"archetype": label, "games": cell.games,
                     "share": share, "winrate": cell.winrate,
                     "ci": [lo, hi], "covered": ok, "note": note})

    print(f"\n  CAMPO COBERTO POR CÉLULA MEDIDA: {covered_share:.1%}")
    print(f"  campo NÃO coberto:                {1 - covered_share:.1%}")

    holes = [r for r in rows if not r["covered"]]
    holes.sort(key=lambda r: -(r["share"] * (1 - (r["winrate"] or 0))))
    print(f"\n-- buracos, por prioridade = share x (1 - WR real) --")
    if not holes:
        print("  (nenhum)")
    for row in holes:
        priority = row["share"] * (1 - (row["winrate"] or 0))
        flag = "  <== ACIONÁVEL (share >= 5%)" \
            if row["share"] >= HOLE_THRESHOLD else ""
        print(f"  {row['archetype'][:34]:34s} share {row['share']:5.1%}  "
              f"WR {row['winrate']:5.1%}  prioridade {priority:.3f}{flag}")

    payload = {"total_games": total, "covered_share": covered_share,
               "cells": rows}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"\njson: {args.json}")


if __name__ == "__main__":
    main()
