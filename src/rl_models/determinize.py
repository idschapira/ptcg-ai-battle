"""Determinization of the hidden zones — the input to cg.search_begin.

Two flavours, deliberately separated because they carry DIFFERENT
epistemic weight:

``determinize`` (EXACT, offline)
    Both decklists are known, so the hidden multiset is a fact: the
    opponent's unseen cards are exactly ``their 60 - what we have seen``.
    If the arithmetic does not close, the state has a zone we do not
    model, and we return None rather than invent cards. This is what the
    counterfactual analysis and the offline SearchAgent use.

``runtime_determinize`` (PRESUMED, submission)
    In a real game the opponent's list is hidden, so we feed a PRESUMED
    60 (an archetype's consensus list, see deckbuilding.archetype_rules).
    A presumed list is a hypothesis and will not close exactly — the
    opponent runs a tech we did not predict, or fewer copies than we
    assumed. Refusing to search on any mismatch would mean never
    searching, so the opponent's pool is RECONCILED to the size the
    engine requires: trimmed when we over-predicted, padded from the
    presumed list when we under-predicted. OUR OWN side stays exact (we
    know our 60) and still returns None when it fails, because a failure
    there means we misread the state, not the opponent.

    The reconciliation is bounded: ``max_padding`` caps how many cards we
    are willing to invent. Past that the hypothesis is too wrong to be
    worth searching under, and the caller falls back to its prior.

Both return ``(your_deck, your_prize, opp_deck, opp_prize, opp_hand)``
in cg.api.search_begin order, or None.
"""

from __future__ import annotations

import dataclasses
import json
import random
from collections import Counter
from typing import Any, Final

# How far a presumed decklist may be wrong before searching under it
# stops being defensible (cards invented, out of the ~60 hidden).
DEFAULT_MAX_PADDING: Final[int] = 12


def _to_dict(observation: Any) -> dict:
    """Engine observation dataclass -> plain dict (agent contract shape)."""
    return json.loads(json.dumps(dataclasses.asdict(observation),
                                 default=int))


def visible_ids(state: dict, player: int) -> Counter:
    """Multiset of ``player``'s cards visible anywhere in ``state``."""
    seen: Counter = Counter()

    def add_card(card: dict | None) -> None:
        if card and card.get("playerIndex") == player:
            seen[card["id"]] += 1

    def add_pokemon(pokemon: dict | None, zone_owner: int) -> None:
        if not pokemon:
            return
        if zone_owner == player:
            seen[pokemon["id"]] += 1
        for key in ("energyCards", "tools", "preEvolution"):
            for card in pokemon.get(key) or []:
                add_card(card)

    for zone_owner, ps in enumerate(state.get("players") or []):
        for pokemon in ps.get("active") or []:
            add_pokemon(pokemon, zone_owner)
        for pokemon in ps.get("bench") or []:
            add_pokemon(pokemon, zone_owner)
        for card in ps.get("discard") or []:
            add_card(card)
        for card in ps.get("prize") or []:
            if card is not None:
                add_card(card)
        for card in ps.get("hand") or []:
            add_card(card)
    for card in state.get("stadium") or []:
        add_card(card)
    for card in state.get("looking") or []:
        if card is not None:
            add_card(card)
    return seen


def _pool(deck_ids: list[int], state: dict, player: int) -> list[int]:
    """Cards of ``player``'s list not yet visible (Counter clamps at 0)."""
    remaining = Counter(deck_ids) - visible_ids(state, player)
    return [cid for cid, n in remaining.items() for _ in range(n)]


def _own_side(state: dict, our_seat: int, our_deck: list[int],
              rng: random.Random) -> tuple[list[int], list[int]] | None:
    """(deck, prize) for OUR seat — exact, None when it does not close."""
    ps = state["players"][our_seat]
    pool = _pool(our_deck, state, our_seat)
    hidden_prize = sum(1 for c in ps["prize"] if c is None)
    if len(pool) != ps["deckCount"] + hidden_prize:
        return None
    rng.shuffle(pool)
    deck = pool[:ps["deckCount"]]
    prize = ([c["id"] for c in ps["prize"] if c is not None]
             + pool[ps["deckCount"]:])
    return deck, prize


def _split_opponent(pool: list[int], ps: dict, hidden_prize: int,
                    ) -> tuple[list[int], list[int], list[int]]:
    """Cut a shuffled opponent pool into (deck, prize, hand)."""
    deck = pool[:ps["deckCount"]]
    cut = ps["deckCount"] + hidden_prize
    prize = ([c["id"] for c in ps["prize"] if c is not None]
             + pool[ps["deckCount"]:cut])
    hand = pool[cut:]
    return deck, prize, hand


def determinize(obs_dict: dict, our_seat: int, our_deck: list[int],
                opp_deck_ids: list[int],
                rng: random.Random) -> tuple | None:
    """EXACT sample of (your_deck, your_prize, opp_deck, opp_prize, opp_hand).

    None if either multiset fails to close (zone we do not model) — the
    point is skipped and counted, we never invent cards.
    """
    state = obs_dict["current"]
    them = 1 - our_seat
    own = _own_side(state, our_seat, our_deck, rng)
    if own is None:
        return None
    your_deck, your_prize = own

    ps_them = state["players"][them]
    pool = _pool(opp_deck_ids, state, them)
    hidden_prize = sum(1 for c in ps_them["prize"] if c is None)
    expected = ps_them["deckCount"] + hidden_prize + ps_them["handCount"]
    if len(pool) != expected:
        return None
    rng.shuffle(pool)
    opp_deck, opp_prize, opp_hand = _split_opponent(pool, ps_them,
                                                    hidden_prize)
    return your_deck, your_prize, opp_deck, opp_prize, opp_hand


def runtime_determinize(obs_dict: dict, our_seat: int, our_deck: list[int],
                        presumed_opp_deck: list[int], rng: random.Random,
                        max_padding: int = DEFAULT_MAX_PADDING,
                        ) -> tuple | None:
    """PRESUMED sample: our side exact, the opponent reconciled to fit.

    Returns None when our own side does not close (we misread the state)
    or when reconciling the opponent would require inventing more than
    ``max_padding`` cards (the deck hypothesis is too wrong to search
    under). Never raises on a merely inaccurate hypothesis.
    """
    state = obs_dict.get("current")
    if not isinstance(state, dict):
        return None
    them = 1 - our_seat
    players = state.get("players") or []
    if len(players) != 2 or not presumed_opp_deck:
        return None
    own = _own_side(state, our_seat, our_deck, rng)
    if own is None:
        return None
    your_deck, your_prize = own

    ps_them = players[them]
    pool = _pool(presumed_opp_deck, state, them)
    hidden_prize = sum(1 for c in ps_them["prize"] if c is None)
    needed = ps_them["deckCount"] + hidden_prize + ps_them["handCount"]
    if needed < 0:
        return None
    if len(pool) < needed - max_padding:
        return None  # hypothesis too thin: we would be inventing the game
    rng.shuffle(pool)
    if len(pool) > needed:
        # we over-predicted: drop the surplus (pool is already shuffled,
        # so this is a uniform sample of the presumed hidden cards).
        pool = pool[:needed]
    elif len(pool) < needed:
        # we under-predicted: pad from the presumed list itself, so the
        # invented cards are at least archetype-plausible, then reshuffle
        # once so the padding does not all land in the same zone.
        missing = needed - len(pool)
        pool.extend(presumed_opp_deck[rng.randrange(len(presumed_opp_deck))]
                    for _ in range(missing))
        rng.shuffle(pool)
    opp_deck, opp_prize, opp_hand = _split_opponent(pool, ps_them,
                                                    hidden_prize)
    return your_deck, your_prize, opp_deck, opp_prize, opp_hand


__all__ = ["DEFAULT_MAX_PADDING", "determinize", "runtime_determinize",
           "visible_ids"]
