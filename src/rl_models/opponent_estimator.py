"""Live opponent-deck estimator — the blocker that makes runtime search possible.

``cg.api.search_begin`` needs the opponent's hidden zones, which means it
needs their 60-card list. Offline we know it; in a real game we do not.
This estimator watches the cards the opponent reveals as the game goes
on and, once it is confident enough, hands the search a PRESUMED list:
the consensus decklist of the archetype it labelled (see
deckbuilding.archetype_rules, the same rules the offline meta radar
uses).

Card identity comes from the engine's ``serial`` (stable per physical
card, 0-59 for player 0 and 60-119 for player 1) paired with
``playerIndex`` — deduplicating (playerIndex, serial) across every
observation of the episode yields exactly the opponent cards we have
been shown, with correct copy counts. This is the same mechanism
meta_radar mines replays with, so what we measure offline is what we get
online.

Confidence is deliberately two-sided:

``n_observed``
    How much of the opponent's deck we have actually been shown. Early
    turns reveal a handful of cards and any label off those is a guess.

``containment``
    Of the cards we HAVE seen, what fraction the presumed list can
    account for (copies included). This is the honest test of the
    hypothesis: a label can be right while the specific list we hold is
    wrong, and containment is what notices. It also predicts whether
    ``runtime_determinize`` will have to invent cards.

Below either threshold there is no deck hypothesis and the caller plays
its prior. That is the intended default — the estimator starts silent
and only speaks once the evidence arrives.

None-safe throughout: malformed observations, unknown card ids and
missing decklists degrade to "no hypothesis", never to an exception.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from ..deckbuilding.archetype_rules import (ARCHETYPE_DECKS, UNKNOWN,
                                            label_archetype)
from ..ingestion.build_card_model import REPO_ROOT
from ..ingestion.card_index import CardIndex

logger = logging.getLogger(__name__)

# Defaults chosen from the per-turn accuracy sweep over the real replay
# corpus (see src/analysis/estimator_accuracy.py): the knee where the
# label stops being a coin flip without giving up most of the game.
DEFAULT_MIN_OBSERVED: Final[int] = 8
DEFAULT_MIN_CONTAINMENT: Final[float] = 0.75


def read_deck_ids(path: Path) -> list[int]:
    """One card id per line. Returns [] instead of raising (runtime)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    ids: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if line and line.lstrip("-").isdigit():
            ids.append(int(line))
    return ids


@dataclass
class Estimate:
    """What the estimator currently believes about the opponent."""

    archetype: str = UNKNOWN
    deck_ids: tuple[int, ...] = ()
    n_observed: int = 0
    containment: float = 0.0
    confident: bool = False
    reason: str = "no-observations"

    @property
    def usable(self) -> bool:
        return self.confident and len(self.deck_ids) > 0


@dataclass
class EstimatorStats:
    """Aggregable counters (share one instance across an evaluation)."""

    resets: int = 0
    updates: int = 0
    confident_calls: int = 0
    labels: Counter = field(default_factory=Counter)
    reasons: Counter = field(default_factory=Counter)

    def summary(self) -> str:
        share = (self.confident_calls / self.updates) if self.updates else 0.0
        return (f"updates {self.updates}, confident {self.confident_calls} "
                f"({share:.0%}), labels {dict(self.labels.most_common(6))}, "
                f"reasons {dict(self.reasons.most_common(6))}")


class OpponentDeckEstimator:
    """Accumulates revealed opponent cards and proposes a presumed deck.

    One instance per episode is the intended use, but ``observe`` detects
    a new game (deck-selection observation, or the turn counter going
    backwards) and resets itself, so a module-level instance that
    survives across episodes cannot leak one game's reads into the next.
    """

    def __init__(self, index: CardIndex | None = None,
                 min_observed: int = DEFAULT_MIN_OBSERVED,
                 min_containment: float = DEFAULT_MIN_CONTAINMENT,
                 decks_root: Path = REPO_ROOT,
                 stats: EstimatorStats | None = None) -> None:
        self._index = index if index is not None else CardIndex()
        self._min_observed = max(1, min_observed)
        self._min_containment = min_containment
        self._decks_root = decks_root
        self.stats = stats if stats is not None else EstimatorStats()
        self._deck_cache: dict[str, tuple[int, ...]] = {}
        self._seen: dict[int, int] = {}     # opponent serial -> card id
        self._last_turn: int = -1
        self.last_estimate: Estimate = Estimate()

    # ------------------------------------------------------------------ #
    # Deck hypotheses
    # ------------------------------------------------------------------ #

    def presumed_deck(self, archetype: str) -> tuple[int, ...]:
        """The consensus 60 we hold for ``archetype`` (empty if none)."""
        if archetype in self._deck_cache:
            return self._deck_cache[archetype]
        relative = ARCHETYPE_DECKS.get(archetype)
        deck: tuple[int, ...] = ()
        if relative:
            ids = read_deck_ids(self._decks_root / relative)
            if len(ids) == 60:
                deck = tuple(ids)
            else:
                logger.warning("presumed deck %s has %d ids (want 60) — "
                               "ignoring", relative, len(ids))
        self._deck_cache[archetype] = deck
        return deck

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        self._seen.clear()
        self._last_turn = -1
        self.last_estimate = Estimate()
        self.stats.resets += 1

    def observe(self, obs_dict: Any) -> Estimate:
        """Fold one observation in and return the current estimate."""
        try:
            return self._observe(obs_dict)
        except Exception:  # noqa: BLE001 — an estimator must never crash a game
            logger.debug("estimator failed on an observation", exc_info=True)
            self.last_estimate = Estimate(reason="estimator-error")
            self.stats.reasons["estimator-error"] += 1
            return self.last_estimate

    def _observe(self, obs_dict: Any) -> Estimate:
        if not isinstance(obs_dict, dict):
            return self.last_estimate
        if obs_dict.get("select") is None and obs_dict.get("current") is None:
            self.reset()      # deck-selection observation == a new episode
            return self.last_estimate
        state = obs_dict.get("current")
        if not isinstance(state, dict):
            return self.last_estimate
        our_seat = state.get("yourIndex")
        players = state.get("players") or []
        if our_seat not in (0, 1) or len(players) != 2:
            return self.last_estimate

        turn = state.get("turn")
        if isinstance(turn, int):
            if turn < self._last_turn:
                self.reset()  # turn counter went backwards == a new episode
            self._last_turn = turn

        _collect_serials(state, 1 - our_seat, self._seen)
        self.stats.updates += 1
        self.last_estimate = self._estimate()
        self.stats.labels[self.last_estimate.archetype] += 1
        self.stats.reasons[self.last_estimate.reason] += 1
        if self.last_estimate.usable:
            self.stats.confident_calls += 1
        return self.last_estimate

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def observed_counts(self) -> Counter:
        """card id -> copies of it the opponent has revealed."""
        return Counter(self._seen.values())

    def _estimate(self) -> Estimate:
        counts = self.observed_counts()
        n_observed = sum(counts.values())
        names = []
        for card_id in counts:
            card = self._index.get_card(card_id)
            if card is not None:
                names.append(card.card_name)
        archetype = label_archetype(names)
        deck = self.presumed_deck(archetype)

        if archetype == UNKNOWN:
            return Estimate(archetype, (), n_observed, 0.0, False,
                            "unknown-archetype")
        if not deck:
            return Estimate(archetype, (), n_observed, 0.0, False,
                            "no-presumed-deck")

        presumed = Counter(deck)
        covered = sum(min(n, presumed.get(card_id, 0))
                      for card_id, n in counts.items())
        containment = covered / n_observed if n_observed else 0.0

        if n_observed < self._min_observed:
            return Estimate(archetype, deck, n_observed, containment, False,
                            "too-few-observed")
        if containment < self._min_containment:
            return Estimate(archetype, deck, n_observed, containment, False,
                            "low-containment")
        return Estimate(archetype, deck, n_observed, containment, True, "ok")


def _collect_serials(node: Any, player: int, out: dict[int, int]) -> None:
    """Record every {id, serial, playerIndex} card belonging to ``player``.

    Recursive because cards hide in nested zones (attached energy, tools,
    pre-evolutions); ``serial`` is the engine's stable per-card identity,
    so a card seen twice is recorded once and copy counts stay honest.
    """
    if isinstance(node, dict):
        card_id = node.get("id")
        serial = node.get("serial")
        owner = node.get("playerIndex")
        if (isinstance(card_id, int) and isinstance(serial, int)
                and owner == player):
            out[serial] = card_id
        for value in node.values():
            _collect_serials(value, player, out)
    elif isinstance(node, list):
        for value in node:
            _collect_serials(value, player, out)


__all__ = ["DEFAULT_MIN_CONTAINMENT", "DEFAULT_MIN_OBSERVED", "Estimate",
           "EstimatorStats", "OpponentDeckEstimator", "read_deck_ids"]
