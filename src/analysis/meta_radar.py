"""Meta radar: mine OBSERVED decklists from the daily top-episode corpus.

Characterizes what the TOP of the ladder actually plays (observability
only — no production impact, no deck/pilot building). Per episode, both
players' decks are reconstructed from every card the replay ever showed:
cards carry a stable ``serial`` (0-59 for player 0, 60-119 for player 1)
plus ``playerIndex``, so deduplicating (playerIndex, serial) across all
observations yields the observed portion of each 60-card list (cards
never drawn/played stay unobserved — coverage is reported, not assumed).

Archetype labels come from CORE-CARD rules (all names present -> label),
resolved with the same apostrophe-insensitive normalization used by
src/deckbuilding/reconcile_archetypes.py. Rules are ordered specific ->
generic; unmatched decks are "unknown" and their most-seen Pokémon are
surfaced so new archetypes can be promoted into rules.

Our own ladder agents (team "Ilan Schapira", v1-v4) are split out of the
leader distribution. Optionally (--own) the locally fetched episodes in
viewer/episodes/ are cross-referenced: opponent decks are labeled and
our real result per archetype is tabulated (BIASED sample — the fetch
default downloads losses).

Outputs (gitignored):
    data/processed/meta_radar/decks.csv          one row per observed deck
    data/processed/meta_radar/radar_by_day.csv   archetype x day counts

Run from the repo root:
    python -m src.analysis.meta_radar --own
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterable

from ..deckbuilding.reconcile_archetypes import _norm
from ..ingestion.build_card_model import PROCESSED_DIR, REPO_ROOT
from ..ingestion.card_index import Card, CardIndex
from ..ingestion.replays_download import REPLAYS_DIR

logger = logging.getLogger(__name__)

OUT_DIR: Final[Path] = PROCESSED_DIR / "meta_radar"
OWN_EPISODES_DIR: Final[Path] = REPO_ROOT / "viewer" / "episodes"
OUR_TEAM_NAMES: Final[frozenset[str]] = frozenset({"Ilan Schapira"})
_DAY_DIR: Final[re.Pattern[str]] = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Ordered core-card rules (first full match wins). Empirical ladder cores
# first (mined 2026-07-14 from the top-100 corpus), then the archetypes
# the initial meta report predicted, so their (non-)appearance is counted.
# Names are engine card names, apostrophe-insensitive via _norm.
ARCHETYPE_RULES: Final[tuple[tuple[str, frozenset[str]], ...]] = (
    ("Alakazam box (non-ex)", frozenset({"Alakazam", "Kadabra"})),
    ("Team Rocket Spidops (non-ex)", frozenset({"Team Rocket's Spidops"})),
    ("Crustle mill (ours)", frozenset({"Crustle", "Great Tusk"})),
    ("Crustle + Mega Kangaskhan stall",
     frozenset({"Crustle", "Mega Kangaskhan ex"})),
    ("Dragapult ex", frozenset({"Dragapult ex"})),
    ("Mega Lucario ex", frozenset({"Mega Lucario ex"})),
    ("Lillie's Clefairy", frozenset({"Lillie's Clefairy"})),
    ("Gardevoir ex / Jellicent ex", frozenset({"Gardevoir ex"})),
    ("Slowking / Kyurem", frozenset({"Slowking", "Kyurem"})),
    ("Iono's Bellibolt ex", frozenset({"Iono's Bellibolt ex"})),
    # ladder archetype promoted from the 2026-07-12 "unknown" cluster
    # (team taksai): Mega Starmie ex / Mega Froslass ex + Cinderace.
    ("Mega Starmie / Mega Froslass", frozenset({"Mega Starmie ex"})),
    ("Mega Starmie / Mega Froslass", frozenset({"Mega Froslass ex"})),
    ("Crustle stall (other)", frozenset({"Crustle"})),
    # weak fallbacks for partially observed decks: pieces unique to the
    # archetype's evolution line still identify it when the top of the
    # line was never drawn/seen.
    ("Alakazam box (non-ex)", frozenset({"Kadabra"})),
    ("Alakazam box (non-ex)", frozenset({"Alakazam"})),
    ("Alakazam box (non-ex)", frozenset({"Abra"})),
    ("Team Rocket Spidops (non-ex)",
     frozenset({"Team Rocket's Tarountula"})),
)
UNKNOWN: Final[str] = "unknown"

_POKEMON_STAGES: Final[frozenset[int]] = frozenset({7, 8, 9})


@dataclass(frozen=True)
class ObservedDeck:
    """One player's observed cards in one episode (subset of their 60)."""

    day: str
    episode_id: int
    player_index: int
    team: str
    copies_by_name: dict[str, int]   # observed copies per card name
    unknown_ids: tuple[int, ...]     # ids the CardIndex does not know
    archetype: str
    is_ours: bool

    @property
    def n_observed(self) -> int:
        return sum(self.copies_by_name.values()) + len(self.unknown_ids)


def _walk_cards(node: Any, out: dict[tuple[int, int], int]) -> None:
    """Collect every {id, serial, playerIndex} card dict, recursively."""
    if isinstance(node, dict):
        if "id" in node and "serial" in node and "playerIndex" in node:
            player, serial = node["playerIndex"], node["serial"]
            if isinstance(player, int) and isinstance(serial, int):
                out[(player, serial)] = node["id"]
        for value in node.values():
            _walk_cards(value, out)
    elif isinstance(node, list):
        for value in node:
            _walk_cards(value, out)


def observed_serials(replay: dict) -> dict[tuple[int, int], int]:
    """(playerIndex, serial) -> card_id over every observation in a replay."""
    seen: dict[tuple[int, int], int] = {}
    for step in replay.get("steps") or []:
        if not isinstance(step, list):
            continue
        for entry in step:
            if not isinstance(entry, dict):
                continue
            current = (entry.get("observation") or {}).get("current")
            if current:
                _walk_cards(current, seen)
    return seen


def label_archetype(names: Iterable[str]) -> str:
    present = {_norm(name) for name in names}
    for label, core in ARCHETYPE_RULES:
        if all(_norm(name) in present for name in core):
            return label
    return UNKNOWN


def extract_decks(replay: dict, day: str, index: CardIndex) -> list[ObservedDeck]:
    """Both players' observed decks for one replay (None-safe)."""
    info = replay.get("info") or {}
    teams = info.get("TeamNames") or ["?", "?"]
    episode_id = int(info.get("EpisodeId") or 0)
    seen = observed_serials(replay)

    decks: list[ObservedDeck] = []
    for player in (0, 1):
        copies: Counter[str] = Counter()
        unknown: list[int] = []
        for (p, _serial), card_id in seen.items():
            if p != player:
                continue
            card = index.get_card(card_id)
            if card is None:
                unknown.append(card_id)
            else:
                copies[card.card_name] += 1
        team = str(teams[player]) if player < len(teams) else "?"
        decks.append(ObservedDeck(
            day=day,
            episode_id=episode_id,
            player_index=player,
            team=team,
            copies_by_name=dict(copies),
            unknown_ids=tuple(sorted(unknown)),
            archetype=label_archetype(copies),
            is_ours=team in OUR_TEAM_NAMES,
        ))
    return decks


def _iter_replay_files(corpus_dir: Path) -> Iterable[tuple[str, Path]]:
    """(day, path) for every daily replay JSON (skips sample/_index)."""
    for day_dir in sorted(corpus_dir.iterdir()) if corpus_dir.exists() else []:
        if not day_dir.is_dir() or not _DAY_DIR.match(day_dir.name):
            continue
        for path in sorted(day_dir.glob("*.json")):
            yield day_dir.name, path


def mine_corpus(corpus_dir: Path = REPLAYS_DIR,
                index: CardIndex | None = None) -> list[ObservedDeck]:
    index = index if index is not None else CardIndex()
    decks: list[ObservedDeck] = []
    episodes = 0
    for day, path in _iter_replay_files(corpus_dir):
        try:
            with open(path, encoding="utf-8") as fh:
                replay = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("unreadable replay %s: %s", path.name, exc)
            continue
        episodes += 1
        decks.extend(extract_decks(replay, day, index))
    logger.info("mined %d episodes -> %d observed decks", episodes, len(decks))
    return decks


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def _attacker_profile(deck_names: Iterable[str],
                      index: CardIndex) -> list[tuple[str, Card]]:
    """(name, card) for observed Pokémon that actually have attacks."""
    by_name: dict[str, Card] = {}
    for card in index.cards.values():
        by_name.setdefault(card.card_name, card)
    profile = []
    for name in deck_names:
        card = by_name.get(name)
        if card and card.stage_code in _POKEMON_STAGES and card.attack_ids:
            profile.append((name, card))
    return profile


def write_csvs(decks: list[ObservedDeck], out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "decks.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["day", "episode_id", "player_index", "team",
                         "archetype", "is_ours", "n_observed", "cards"])
        for deck in decks:
            cards = "; ".join(f"{n}x {name}" for name, n
                              in sorted(deck.copies_by_name.items()))
            writer.writerow([deck.day, deck.episode_id, deck.player_index,
                             deck.team, deck.archetype, int(deck.is_ours),
                             deck.n_observed, cards])
    days = sorted({d.day for d in decks})
    archetypes = sorted({d.archetype for d in decks})
    counts = Counter((d.day, d.archetype) for d in decks if not d.is_ours)
    with open(out_dir / "radar_by_day.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["archetype"] + days + ["total"])
        for archetype in archetypes:
            row = [counts.get((day, archetype), 0) for day in days]
            writer.writerow([archetype] + row + [sum(row)])

    # consensus decklist per leader team: max copies of each card ever
    # observed in one game floors the team's true list (60-card seeds
    # for the 2nd-archetype work).
    consensus: dict[str, Counter[str]] = defaultdict(Counter)
    games_per_team: Counter[str] = Counter()
    for deck in decks:
        if deck.is_ours:
            continue
        games_per_team[deck.team] += 1
        for name, n in deck.copies_by_name.items():
            consensus[deck.team][name] = max(consensus[deck.team][name], n)
    with open(out_dir / "leader_decklists.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["team", "games", "cards_floor", "card", "copies"])
        for team, cards in sorted(consensus.items()):
            floor = sum(cards.values())
            for name, n in sorted(cards.items(), key=lambda kv: (-kv[1], kv[0])):
                writer.writerow([team, games_per_team[team], floor, name, n])


def print_report(decks: list[ObservedDeck], index: CardIndex) -> None:
    leader = [d for d in decks if not d.is_ours]
    ours = [d for d in decks if d.is_ours]
    episodes = len({(d.day, d.episode_id) for d in decks})
    labeled = [d for d in leader if d.archetype != UNKNOWN]
    print(f"episodes: {episodes} | observed decks: {len(decks)} "
          f"(ours: {len(ours)}) | labeled: {len(labeled)}/{len(leader)} "
          f"({len(labeled) / max(len(leader), 1):.0%})")

    well_observed = sum(1 for d in leader if d.n_observed >= 45)
    print(f"deck observation: {well_observed}/{len(leader)} decks with "
          f">=45/60 cards seen")
    unknown_ids = Counter()
    for deck in decks:
        unknown_ids.update(deck.unknown_ids)
    if unknown_ids:
        print(f"unknown card ids: {dict(unknown_ids.most_common(10))}")

    print("\n== radar (leader decks, share per day) ==")
    days = sorted({d.day for d in leader})
    by_day: dict[str, Counter[str]] = defaultdict(Counter)
    for deck in leader:
        by_day[deck.day][deck.archetype] += 1
    archetype_totals = Counter(d.archetype for d in leader)
    for archetype, total in archetype_totals.most_common():
        trend = "  ".join(
            f"{day[5:]}: {by_day[day][archetype]:3d}"
            f" ({by_day[day][archetype] / max(sum(by_day[day].values()), 1):4.0%})"
            for day in days)
        print(f"  {archetype:34s} {total:4d}  | {trend}")

    print("\n== teams -> archetype ==")
    team_arch = Counter((d.team, d.archetype) for d in leader)
    for (team, archetype), n in team_arch.most_common():
        print(f"  {team:20s} {archetype:34s} {n}")

    print("\n== attacker profile per archetype (wall-threat read) ==")
    for archetype, _total in archetype_totals.most_common():
        names: set[str] = set()
        for deck in leader:
            if deck.archetype == archetype:
                names.update(deck.copies_by_name)
        attackers = _attacker_profile(names, index)
        non_ex = [n for n, c in attackers if not c.is_ex and not c.is_mega_ex]
        print(f"  {archetype}: attackers={len(attackers)} "
              f"non-ex={len(non_ex)} {sorted(non_ex)[:6]}")

    unknown_decks = [d for d in leader if d.archetype == UNKNOWN]
    if unknown_decks:
        print("\n== unknown decks: most-seen Pokémon (promote to rules?) ==")
        seen: Counter[str] = Counter()
        for deck in unknown_decks:
            for name, card in _attacker_profile(deck.copies_by_name, index):
                seen[name] += 1
        for name, n in seen.most_common(12):
            print(f"  {n:3d}  {name}")


def report_own_episodes(own_dir: Path, index: CardIndex) -> None:
    """Label opponents in our fetched episodes (result-biased sample)."""
    paths = sorted(own_dir.glob("*.json")) if own_dir.exists() else []
    results: dict[str, Counter[str]] = defaultdict(Counter)
    for path in paths:
        try:
            with open(path, encoding="utf-8") as fh:
                replay = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("unreadable episode %s: %s", path.name, exc)
            continue
        decks = extract_decks(replay, "own", index)
        us = next((d for d in decks if d.is_ours), None)
        them = next((d for d in decks if not d.is_ours), None)
        if us is None or them is None:
            continue
        rewards = replay.get("rewards") or []
        if len(rewards) == 2 and rewards[0] != rewards[1]:
            outcome = ("win" if rewards[us.player_index] ==
                       max(rewards) else "loss")
        else:
            outcome = "draw"
        results[them.archetype][outcome] += 1
    if not results:
        print("\n(no own episodes found to cross-reference)")
        return
    print(f"\n== our episodes vs archetype ({sum(sum(c.values()) for c in results.values())} "
          f"games, BIASED: fetch defaults to losses) ==")
    for archetype, counter in sorted(results.items(),
                                     key=lambda kv: -sum(kv[1].values())):
        print(f"  {archetype:34s} W{counter['win']}/L{counter['loss']}"
              f"/D{counter['draw']}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=REPLAYS_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--own", action="store_true",
                        help="also cross-reference viewer/episodes/")
    args = parser.parse_args()

    index = CardIndex()
    decks = mine_corpus(args.corpus, index)
    if not decks:
        print(f"no replays under {args.corpus}")
        return
    write_csvs(decks, args.out)
    print_report(decks, index)
    if args.own:
        report_own_episodes(OWN_EPISODES_DIR, index)
    print(f"\ncsv: {args.out.relative_to(REPO_ROOT)}"
          if args.out.is_relative_to(REPO_ROOT) else f"\ncsv: {args.out}")


if __name__ == "__main__":
    main()
