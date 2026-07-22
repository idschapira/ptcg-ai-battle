"""Deck rule modules + the deck -> module/theta default registry.

`MODULES` is the versioned mapping from a deck name (the data/decks stem
minus its seed_/meta_/placeholder_/candidate_ prefix) to the module that
pilots it. A deck with no entry gets the base DeckModule — i.e. plain
generic heuristic play, which is the correct answer for a deck nobody
has written rules for yet.
"""

from __future__ import annotations

from typing import Final

from ..module import DeckModule
from ..theta import Theta
from .crustle import CrustleModule
from .grimmsnarl import GrimmsnarlModule

#: One shared instance per module: they are stateless past `bind`, and
#: the fitness harness builds many agents per second.
_REGISTRY: Final[tuple[DeckModule, ...]] = (
    CrustleModule(),
    GrimmsnarlModule(),
)

MODULES: Final[dict[str, DeckModule]] = {m.name: m for m in _REGISTRY}

#: deck name -> module name. Everything else falls back to generic.
DECK_MODULE: Final[dict[str, str]] = {
    deck: module.name for module in _REGISTRY for deck in module.decks
}

_GENERIC: Final[DeckModule] = DeckModule()


def module_for_deck(deck: str) -> DeckModule:
    """The pilot module for a deck name (generic when unmapped)."""
    return MODULES.get(DECK_MODULE.get(deck, ""), _GENERIC)


def default_theta_for_deck(deck: str) -> Theta:
    """The versioned deck -> theta_default entry (the evolution seed)."""
    return module_for_deck(deck).default_theta()


__all__ = ["MODULES", "DECK_MODULE", "CrustleModule", "GrimmsnarlModule",
           "module_for_deck", "default_theta_for_deck"]
