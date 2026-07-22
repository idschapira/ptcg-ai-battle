"""DeckModule: the per-deck rules unit that a Theta parameterizes.

A module is a pure OVERLAY. Each hook receives the generic
HeuristicAgent score as `base` and returns either a new score or `None`
meaning "no opinion — keep the generic score". Returning `None` is the
normal case; that is what makes a module safe on an unfamiliar board and
what preserves the shipped generic fallback everywhere a rule's signals
are missing.

The agent wraps every hook in try/except and falls back to `base`, so a
module bug degrades to generic play instead of an illegal answer. Rules
therefore do not need defensive plumbing of their own — they should read
signals through `BoardView` (which already answers None for unknown) and
express the strategy plainly.

Hooks:
    main_score        MAIN-menu option (play/attach/evolve/attack/...)
    own_pokemon_score promotion / benching / gust-target choice
    attach_score      which pokémon receives an energy
    select_score      a scored answer for an arbitrary SelectContext
                      (search picks, discards, damage-counter targets)

`select_score` returns (score_fn, min_count, max_count) so a module can
take over a context the generic agent answers by index order.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from cg.api import Option, SelectContext

from .board import BoardView
from .theta import EMPTY_SCHEMA, Theta, ThetaSchema

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .parametric_agent import ParametricHeuristicAgent

ScoreFn = Callable[[Option], float]
SelectPlan = tuple[ScoreFn, int, int]


class DeckModule:
    """Base class: every hook declines, so it plays exactly generic."""

    name: str = "generic"
    #: decks this module is the natural pilot for (data/decks stem names)
    decks: tuple[str, ...] = ()
    schema: ThetaSchema = EMPTY_SCHEMA

    def bind(self, agent: "ParametricHeuristicAgent") -> None:
        """Cache deck-specific index lookups once, at construction."""

    # ---- hooks (None => keep the generic score) ---- #

    def main_score(self, view: BoardView, option: Option,
                   base: float) -> float | None:
        return None

    def own_pokemon_score(self, view: BoardView, option: Option,
                          for_active: bool, base: float) -> float | None:
        return None

    def attach_score(self, view: BoardView, option: Option,
                     base: float) -> float | None:
        return None

    def select_score(self, view: BoardView,
                     context: SelectContext) -> Optional[SelectPlan]:
        return None

    # ---- genome helpers ---- #

    def default_theta(self) -> Theta:
        return self.schema.defaults()

    def theta_from(self, mapping) -> Theta:
        return self.schema.from_dict(mapping)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} " \
               f"knobs={len(self.schema)}>"


__all__ = ["DeckModule", "ScoreFn", "SelectPlan"]
