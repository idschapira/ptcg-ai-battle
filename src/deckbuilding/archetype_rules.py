"""Core-card archetype rules — the single source of truth for labelling.

Extracted from src/analysis/meta_radar.py so that BOTH the offline radar
and the RUNTIME opponent estimator label decks with exactly the same
rules. meta_radar re-exports these names, so the offline reports are
unchanged; the runtime side can import this module without dragging in
Polars/replay/analysis dependencies (stdlib only — deliberately).

Rules are ordered specific -> generic; the first rule whose core cards
are ALL present wins. Unmatched decks are ``UNKNOWN`` and, at runtime,
mean "no presumed decklist" -> the agent falls back to its prior.

``ARCHETYPE_DECKS`` maps a label to the repo-relative decklist we use as
the PRESUMED 60 when we meet that archetype. Those lists are consensus
reconstructions of ladder decks (mined by meta_radar), not the
opponent's true list — everything downstream must treat them as a
hypothesis, never as fact.
"""

from __future__ import annotations

from typing import Final, Iterable

UNKNOWN: Final[str] = "unknown"


def norm(name: str) -> str:
    """Case/apostrophe-insensitive form ('’' == "'")."""
    return name.replace("’", "'").casefold()


# Ordered core-card rules (first full match wins). Empirical ladder cores
# first (mined 2026-07-14 from the top-100 corpus), then the archetypes
# the initial meta report predicted, so their (non-)appearance is counted.
# Names are engine card names, apostrophe-insensitive via norm.
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
    # mid-ladder archetypes promoted from OUR episodes' "unknown"
    # opponents (A/B e10 reading, 2026-07-16):
    ("Archaludon ex box", frozenset({"Archaludon ex"})),
    ("Archaludon ex box", frozenset({"Duraludon"})),
    ("Marnie's Grimmsnarl ex", frozenset({"Marnie's Grimmsnarl ex"})),
    ("Marnie's Grimmsnarl ex", frozenset({"Marnie's Impidimp"})),
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

# label -> repo-relative presumed decklist (60 ids, one per line).
# Only archetypes we actually hold a reconstructed list for appear here;
# every other label (and UNKNOWN) means "no deck hypothesis".
ARCHETYPE_DECKS: Final[dict[str, str]] = {
    "Alakazam box (non-ex)": "data/decks/meta_alakazam.csv",
    "Team Rocket Spidops (non-ex)": "data/decks/meta_spidops.csv",
    "Marnie's Grimmsnarl ex": "data/decks/meta_grimmsnarl.csv",
    "Mega Starmie / Mega Froslass": "data/decks/meta_starmie.csv",
    "Crustle + Mega Kangaskhan stall": "data/decks/meta_crustle_kangaskhan.csv",
    # the mirror: an opponent playing our own list
    "Crustle mill (ours)": "data/decks/candidate_crustle_e10.csv",
}


def label_archetype(names: Iterable[str]) -> str:
    """First rule whose core cards are all present, else ``UNKNOWN``."""
    present = {norm(name) for name in names}
    for label, core in ARCHETYPE_RULES:
        if all(norm(name) in present for name in core):
            return label
    return UNKNOWN


__all__ = ["ARCHETYPE_DECKS", "ARCHETYPE_RULES", "UNKNOWN",
           "label_archetype", "norm"]
