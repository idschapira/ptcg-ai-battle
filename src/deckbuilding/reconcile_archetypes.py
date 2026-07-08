"""Reconcile the meta-report archetypes against the engine card pool.

Maps every card named in "Análise Inicial Meta Pokémon TCG.md" (the
Sprint-5D strategic input: LAIC/NAIC 2026 archetypes + cabt internal
winrates) to real card_ids via CardIndex, and prints a coverage report
per archetype: which cards exist in the ~2000-card cabt pool, under
which exact name/id, and which are absent (the pool is a SUBSET of the
physical Standard format — absences are expected and strategic).

Matching: engine names use the typographic apostrophe (U+2019) and
English card names. Queries resolve exact-first, then substring (which
surfaces pool variants, e.g. "Gardevoir ex" -> "Mega Gardevoir ex").

Run from the repo root:
    python -m src.deckbuilding.reconcile_archetypes
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Final

from ..ingestion.card_index import Card, CardIndex

# Report names per archetype (cabt winrates from the report where known).
ARCHETYPES: Final[dict[str, tuple[str, ...]]] = {
    "Mega Lucario ex (60.4% cabt)": (
        "Mega Lucario ex", "Riolu", "Hariyama", "Makuhita", "Solrock",
        "Maximum Belt", "Premium Power Pro", "Gravity Mountain",
    ),
    "Dragapult ex / Dusknoir (55.6% cabt)": (
        "Dragapult ex", "Drakloak", "Dreepy", "Dusknoir", "Dusclops",
        "Duskull", "Budew", "Unfair Stamp", "Neo Upper Energy",
    ),
    "Gardevoir ex / Jellicent ex (LAIC champion)": (
        "Gardevoir ex", "Kirlia", "Ralts", "Jellicent ex", "Frillish",
        "Munkidori",
    ),
    "Lillie's Clefairy (NAIC champion)": (
        "Lillie’s Clefairy", "Mega Kangaskhan ex", "Meowth ex",
        "Area Zero Underdepths",
    ),
    "Crustle stall (NAIC top 8)": (
        "Crustle", "Dwebble", "Hero’s Cape", "Pokémon Center Lady",
        "Jumbo Ice Cream", "Xerosic’s Machinations", "Eri",
    ),
    "Slowking / Kyurem / Metagross": (
        "Slowking", "Slowpoke", "Kyurem", "Metagross", "Metang", "Beldum",
        "Ciphermaniac’s Codebreaking", "Brave Bangle",
    ),
    "Iono (43.8% cabt)": (
        "Iono’s Bellibolt ex", "Iono’s Tadbulb", "Iono’s Voltorb",
        "Iono’s Electrode", "Iono’s Wattrel", "Iono’s Kilowattrel",
        "Boss’s Orders",
    ),
}

# Cross-archetype tech cards the report singles out (threat recognition).
TECH_CARDS: Final[tuple[str, ...]] = (
    "Shaymin", "Budew", "Boss’s Orders", "Unfair Stamp",
)


def _norm(name: str) -> str:
    """Case/apostrophe-insensitive form ('’' == \"'\")."""
    return name.replace("’", "'").casefold()


@dataclass(frozen=True)
class Resolution:
    """One report name resolved against the pool."""

    query: str
    cards: tuple[Card, ...]   # empty -> missing from the pool
    exact: bool               # False -> substring/variant match

    @property
    def found(self) -> bool:
        return bool(self.cards)


def resolve(query: str, index: CardIndex) -> Resolution:
    """Exact-name match first; fall back to substring (pool variants)."""
    wanted = _norm(query)
    exact = tuple(c for c in index.cards.values()
                  if _norm(c.card_name) == wanted)
    if exact:
        return Resolution(query, exact, True)
    partial = tuple(c for c in index.cards.values()
                    if wanted in _norm(c.card_name))
    return Resolution(query, partial, False)


def reconcile(index: CardIndex) -> dict[str, list[Resolution]]:
    return {archetype: [resolve(q, index) for q in queries]
            for archetype, queries in ARCHETYPES.items()}


def _print_resolution(res: Resolution) -> None:
    if not res.found:
        print(f"  MISSING  {res.query}")
        return
    kind = "exact " if res.exact else "variant"
    ids = ", ".join(f"{c.card_id}={c.card_name}" for c in res.cards)
    print(f"  {kind}  {res.query} -> {ids}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    index = CardIndex()

    total_found = total = 0
    for archetype, resolutions in reconcile(index).items():
        found = sum(res.found for res in resolutions)
        total_found += found
        total += len(resolutions)
        print(f"\n{archetype}: {found}/{len(resolutions)} cards in pool")
        for res in resolutions:
            _print_resolution(res)

    print(f"\ntech cards (threat recognition):")
    for query in TECH_CARDS:
        _print_resolution(resolve(query, index))

    print(f"\noverall archetype coverage: {total_found}/{total} "
          f"({total_found / total:.0%})")


if __name__ == "__main__":
    main()
