"""Reconstruct an opponent archetype's 60 from OUR real episodes.

The gauntlet can only measure a cell we hold a decklist for, so an
archetype with no csv in data/decks/ is a cell that silently never gets
tested. Archaludon was 14.2% of our real ladder games and had no list —
this module closes that kind of gap from the replays we already own.

Method, and why it is not meta_radar's:
  meta_radar builds a per-TEAM consensus as "max copies ever seen in one
  game", which floors that team's list. Here the games come from ~30
  DIFFERENT teams all playing the same archetype, so taking a max across
  teams unions their tech choices and overshoots 60 (measured: 76). The
  reconstruction instead treats each game as one noisy observation of a
  shared stock list and asks two questions per card:

    presence  in what share of games did the card appear at all?
              Cards below --min-presence are tech/noise and are dropped;
              a card seen in 1 of 32 games is not part of the archetype.
    copies    the MODE of the per-game maximum. The mode is deliberate:
              the max-of-max is set by the single luckiest game (every
              copy drawn) and the mean is dragged by games we ended
              early, while the mode is the count most lists actually run.

Basic Energy is then used as the free variable to land exactly on 60 —
it is the only card type with no 4-copy cap, and it is the slot real
lists flex. If the named cards alone overshoot 60 the reconstruction is
reported as FAILED rather than silently trimmed.

Everything is observation-floored: a card never drawn in 32 games is
invisible, so the output is a HYPOTHESIS about the archetype, never the
opponent's true list. Callers must treat it as such.

Sample discipline: submission-id filter + deck sentinel on our side
(viewer/episodes/ mixes every submission we ever ran), and card
ownership comes from the engine's own playerIndex on each card dict.

Run from the repo root:
    python -m src.analysis.mine_opponent_deck --archetype "Archaludon ex box" \
        --out data/decks/meta_archaludon.csv
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final

from cg import game

from ..deckbuilding.archetype_rules import label_archetype
from ..deckbuilding.legality import validate_deck
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.card_index import CardIndex
from .fetch_my_episodes import EPISODES_DIR
from .meta_radar import observed_serials as _observed_serials

OUR_TEAM: Final[str] = "Ilan Schapira"
OUR_DECK_LABEL: Final[str] = "Crustle mill (ours)"
SUB_INDEX: Final[Path] = (REPO_ROOT / "data" / "processed" /
                          "episodes_index" / "sub_54917180.json")
DECK_SIZE: Final[int] = 60
STAGE_BASIC_ENERGY: Final[int] = 1


def _mode(values: list[int]) -> int:
    """Most common value; ties break HIGH (a list runs the fuller count)."""
    counts = Counter(values)
    best = max(counts.values())
    return max(v for v, c in counts.items() if c == best)


def collect(episodes_dir: Path, archetype: str, team: str,
            allowed: set[int] | None,
            index: CardIndex) -> tuple[list[Counter], Counter, Counter]:
    """(per-game opponent card counts, results, opposing team names)."""
    per_game: list[Counter] = []
    results: Counter = Counter()
    teams: Counter = Counter()
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
        for (player, _serial), card_id in _observed_serials(replay).items():
            card = index.get_card(card_id)
            if card is not None and player in revealed:
                revealed[player][card.card_name] += 1
        if label_archetype(revealed[seat]) != OUR_DECK_LABEL:
            continue
        if label_archetype(revealed[1 - seat]) != archetype:
            continue
        per_game.append(revealed[1 - seat])
        teams[names[1 - seat]] += 1
        rewards = replay.get("rewards") or []
        if len(rewards) == 2 and rewards[0] is not None:
            if rewards[0] == rewards[1]:
                results["draw"] += 1
            else:
                results["win" if rewards[seat] > rewards[1 - seat]
                        else "loss"] += 1
    return per_game, results, teams


def reconstruct(per_game: list[Counter], index: CardIndex,
                min_presence: float) -> tuple[Counter, list[str], list[str]]:
    """(name -> copies, kept names, dropped names) before energy fill."""
    n = len(per_game)
    presence: Counter = Counter()
    maxima: dict[str, list[int]] = defaultdict(list)
    for observation in per_game:
        for name, count in observation.items():
            presence[name] += 1
            maxima[name].append(count)

    by_name = {card.card_name: card for card in index.cards.values()} \
        if hasattr(index, "cards") else {}
    kept: Counter = Counter()
    keep_names, dropped = [], []
    for name, seen in presence.items():
        if seen / n < min_presence:
            dropped.append(f"{name} ({seen}/{n} jogos)")
            continue
        card = by_name.get(name)
        if card is not None and card.stage_code == STAGE_BASIC_ENERGY:
            continue                       # energy is the fill variable
        kept[name] = _mode(maxima[name])
        keep_names.append(name)
    return kept, keep_names, dropped


def to_ids(counts: Counter, index: CardIndex) -> list[int]:
    by_name: dict[str, int] = {}
    for card in index.cards.values():
        by_name.setdefault(card.card_name, card.card_id)
    ids: list[int] = []
    for name, copies in counts.items():
        card_id = by_name.get(name)
        if card_id is None:
            raise SystemExit(f"carta desconhecida no consenso: {name!r}")
        ids.extend([card_id] * copies)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archetype", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes-dir", type=Path, default=EPISODES_DIR)
    parser.add_argument("--team", type=str, default=OUR_TEAM)
    parser.add_argument("--submission-index", type=Path, default=SUB_INDEX)
    parser.add_argument("--min-presence", type=float, default=0.5,
                        help="drop cards seen in fewer than this share of "
                             "games (tech/noise, not the archetype)")
    parser.add_argument("--energy", type=str, default=None,
                        help="Basic Energy card name used to fill to 60 "
                             "(default: the most-seen Basic Energy)")
    args = parser.parse_args()

    index = CardIndex()
    allowed = None
    if args.submission_index.exists():
        with open(args.submission_index, encoding="utf-8") as fh:
            allowed = {int(i) for i in
                       (json.load(fh).get("episode_ids") or [])}

    per_game, results, teams = collect(args.episodes_dir, args.archetype,
                                       args.team, allowed, index)
    if not per_game:
        raise SystemExit(f"nenhum episódio contra {args.archetype!r}")
    print(f"== {args.archetype} ==")
    print(f"{len(per_game)} jogos reais, {len(teams)} times distintos; "
          f"NOSSO resultado: {dict(results)}")

    kept, keep_names, dropped = reconstruct(per_game, index,
                                            args.min_presence)
    named_total = sum(kept.values())
    print(f"\ncartas nomeadas (presença >= {args.min_presence:.0%}, "
          f"cópias = MODA do máximo por jogo): {named_total}")
    for name, copies in sorted(kept.items(), key=lambda kv: (-kv[1], kv[0])):
        seen = sum(1 for o in per_game if name in o)
        print(f"  {copies}x {name[:32]:32s} (visto em {seen}/{len(per_game)})")
    if dropped:
        print(f"\ndescartadas como tech/ruído (< {args.min_presence:.0%}):")
        for text in dropped:
            print(f"  - {text}")

    if named_total > DECK_SIZE:
        raise SystemExit(f"FALHOU: cartas nomeadas somam {named_total} > 60 "
                         f"— reconstrução ambígua, revise --min-presence")

    # energy fills the remainder
    energy_counts: Counter = Counter()
    for observation in per_game:
        for name, count in observation.items():
            card = next((c for c in index.cards.values()
                         if c.card_name == name), None)
            if card is not None and card.stage_code == STAGE_BASIC_ENERGY:
                energy_counts[name] = max(energy_counts[name], count)
    energy_name = args.energy or (energy_counts.most_common(1)[0][0]
                                  if energy_counts else None)
    if energy_name is None:
        raise SystemExit("nenhuma Basic Energy observada — informe --energy")
    fill = DECK_SIZE - named_total
    kept[energy_name] = kept.get(energy_name, 0) + fill
    print(f"\npreenchimento: {fill}x {energy_name} "
          f"(máximo observado num jogo: {energy_counts[energy_name]}) "
          f"-> INFERIDO, não observado como contagem exata")

    ids = sorted(to_ids(kept, index))
    if len(ids) != DECK_SIZE:
        raise SystemExit(f"FALHOU: {len(ids)} cartas, esperado 60")

    report = validate_deck(ids, index)
    print(f"\nlegalidade: {'LEGAL' if report.ok else 'ILEGAL'}")
    for error in report.errors:
        print(f"  ERRO {error}")
    for warning in getattr(report, "warnings", []) or []:
        print(f"  aviso {warning}")
    if not report.ok:
        raise SystemExit("reconstrução ilegal — não gravada")

    obs, start = game.battle_start(list(ids), list(ids))
    engine_ok = obs is not None
    game.battle_finish()
    print(f"motor (battle_start): {'OK' if engine_ok else start.errorType}")
    if not engine_ok:
        raise SystemExit("motor rejeitou a lista — não gravada")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(str(i) for i in ids) + "\n",
                        encoding="utf-8")
    print(f"\ngravado: {args.out}")


if __name__ == "__main__":
    main()
