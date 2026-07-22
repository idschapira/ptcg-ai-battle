"""BoardView: the None-safe board reads every deck module shares.

The deck-specific rules in `modules/` should never touch `obs.current`
directly — that is where the KeyError/IndexError crashes live, and a
module that reaches into the raw state cannot survive a deck it has
never seen. Everything here answers `None` / `False` / `[]` for unknown
rather than raising, so a rule written against a missing signal simply
declines to fire and the generic HeuristicAgent score stands.

The view is built once per scored option batch and carries the agent
(for the CardIndex and the wrapper) plus the active `Theta`, so rule
code reads as `view.theta["..."]` and `view.opp_active_is_ex()`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from cg.api import AreaType, Observation, Option, Pokemon

from ..ingestion.card_index import Card
from .theta import Theta

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .parametric_agent import ParametricHeuristicAgent


class BoardView:
    """Read-only, crash-proof lens over one Observation."""

    __slots__ = ("agent", "obs", "theta")

    def __init__(self, agent: "ParametricHeuristicAgent", obs: Observation,
                 theta: Theta) -> None:
        self.agent = agent
        self.obs = obs
        self.theta = theta

    # ------------------------------------------------------------------ #
    # Card / option resolution
    # ------------------------------------------------------------------ #

    def card(self, card_id: int | None) -> Card | None:
        return self.agent._index.get_card(card_id) if card_id is not None else None

    def card_of(self, pokemon: Pokemon | None) -> Card | None:
        return self.card(pokemon.id) if pokemon is not None else None

    def option_card_id(self, option: Option) -> int | None:
        """Card behind an option, including DECK-area picks (searches
        expose the deck through obs.select.deck, which the generic
        wrapper resolver treats as face-down)."""
        card_id = self.agent._wrapper.resolve_card_id(self.obs, option)
        if card_id is None and option.area == AreaType.DECK:
            try:
                select = self.obs.select
                deck = select.deck if select is not None else None
                entry = deck[option.index] if deck else None
                card_id = entry.id if entry is not None else None
            except (IndexError, TypeError, AttributeError):
                card_id = None
        return card_id

    def option_card(self, option: Option) -> Card | None:
        card_id = self.option_card_id(option)
        return self.card(card_id) if card_id is not None else None

    # ------------------------------------------------------------------ #
    # Seats
    # ------------------------------------------------------------------ #

    @property
    def me(self) -> int | None:
        state = self.obs.current
        return state.yourIndex if state is not None else None

    @property
    def them(self) -> int | None:
        me = self.me
        return None if me is None else 1 - me

    def _player(self, player_index: int | None):
        state = self.obs.current
        if state is None or player_index is None:
            return None
        try:
            return state.players[player_index]
        except (IndexError, TypeError, AttributeError):
            return None

    # ------------------------------------------------------------------ #
    # Pokémon in play
    # ------------------------------------------------------------------ #

    def pokemon_at(self, player_index: int | None, area: AreaType | None,
                   index: int | None) -> Pokemon | None:
        return self.agent._pokemon_at(self.obs.current, player_index, area,
                                      index)

    def my_active(self) -> Pokemon | None:
        return self.agent._my_active(self.obs)

    def opp_active(self) -> Pokemon | None:
        return self.agent._opp_active(self.obs)

    def _bench(self, player_index: int | None) -> list[Pokemon]:
        player = self._player(player_index)
        if player is None:
            return []
        try:
            return [p for p in (player.bench or []) if p is not None]
        except TypeError:
            return []

    def my_bench(self) -> list[Pokemon]:
        return self._bench(self.me)

    def opp_bench(self) -> list[Pokemon]:
        return self._bench(self.them)

    def my_field(self) -> list[Pokemon]:
        """Active + bench, skipping empty slots."""
        active = self.my_active()
        return ([active] if active is not None else []) + self.my_bench()

    def opp_field(self) -> list[Pokemon]:
        active = self.opp_active()
        return ([active] if active is not None else []) + self.opp_bench()

    def my_field_ids(self) -> list[int]:
        return [p.id for p in self.my_field() if p.id is not None]

    def field_count(self) -> int | None:
        """How many of MY pokémon are in play; None when unreadable."""
        player = self._player(self.me)
        if player is None:
            return None
        try:
            return (len([p for p in (player.active or []) if p is not None])
                    + len([p for p in (player.bench or []) if p is not None]))
        except TypeError:
            return None

    def my_active_is(self, card_id: int) -> bool:
        active = self.my_active()
        return active is not None and active.id == card_id

    def count_in_field(self, card_id: int) -> int:
        return sum(1 for p in self.my_field() if p.id == card_id)

    # ------------------------------------------------------------------ #
    # Resource counts
    # ------------------------------------------------------------------ #

    def deck_counts(self) -> tuple[int | None, int | None]:
        """(my deckCount, opponent deckCount)."""
        mine = self._player(self.me)
        theirs = self._player(self.them)
        return (getattr(mine, "deckCount", None),
                getattr(theirs, "deckCount", None))

    def opp_hand_count(self) -> int | None:
        return getattr(self._player(self.them), "handCount", None)

    def my_hand(self) -> Sequence:
        player = self._player(self.me)
        try:
            return [c for c in (player.hand or []) if c is not None] \
                if player is not None else []
        except TypeError:
            return []

    def my_hand_ids(self) -> list[int]:
        return [c.id for c in self.my_hand() if getattr(c, "id", None) is not None]

    def my_discard_ids(self) -> list[int]:
        """Card ids in OUR discard pile (empty when unreadable).

        Visible information: decks that recur or scale off the discard
        (Riptide, Hammer-lanche's self-mill) read their whole plan here."""
        player = self._player(self.me)
        if player is None:
            return []
        try:
            return [c.id for c in (player.discard or [])
                    if c is not None and getattr(c, "id", None) is not None]
        except TypeError:
            return []

    def count_attached(self, energy_code: int) -> int:
        """How many energies of one type are attached across MY field."""
        total = 0
        for pokemon in self.my_field():
            try:
                total += sum(1 for e in (pokemon.energies or [])
                             if int(e) == energy_code)
            except (TypeError, ValueError):
                continue
        return total

    def energies_on(self, pokemon: Pokemon | None) -> list[int]:
        if pokemon is None:
            return []
        try:
            return [int(e) for e in (pokemon.energies or [])]
        except (TypeError, ValueError):
            return []

    def prizes_left(self, mine: bool) -> int | None:
        player = self._player(self.me if mine else self.them)
        try:
            prize = player.prize if player is not None else None
            return len(prize) if prize is not None else None
        except TypeError:
            return None

    def stadium_id(self) -> int | None:
        state = self.obs.current
        try:
            stadium = state.stadium[0] if state is not None and state.stadium \
                else None
            return stadium.id if stadium is not None else None
        except (IndexError, TypeError, AttributeError):
            return None

    # ------------------------------------------------------------------ #
    # Threat reads
    # ------------------------------------------------------------------ #

    def is_ex(self, pokemon: Pokemon | None) -> bool:
        card = self.card_of(pokemon)
        return card is not None and bool(card.is_ex or card.is_mega_ex)

    def opp_active_is_ex(self) -> bool | None:
        """True/False when the opposing active is known; None otherwise."""
        card = self.card_of(self.opp_active())
        if card is None:
            return None
        return bool(card.is_ex or card.is_mega_ex)

    def opp_any_ex(self) -> bool:
        return any(self.is_ex(p) for p in self.opp_field())

    def opp_field_all_ex(self) -> bool:
        """Known active is ex AND every known bench entry is ex."""
        if self.opp_active_is_ex() is not True:
            return False
        for pokemon in self.opp_bench():
            card = self.card_of(pokemon)
            if card is None or not (card.is_ex or card.is_mega_ex):
                return False
        return True

    def damage_on(self, pokemon: Pokemon | None) -> int:
        """Damage already taken (0 when unknown) — the counter engines
        and the heal rules both key off this."""
        if pokemon is None:
            return 0
        try:
            if pokemon.maxHp is None or pokemon.hp is None:
                return 0
            return max(0, int(pokemon.maxHp) - int(pokemon.hp))
        except (TypeError, ValueError):
            return 0

    def is_opponent_option(self, option: Option) -> bool:
        """True when the option points at an OPPONENT's pokémon (gust
        targets); False for our own promotions and for unknown seats."""
        me = self.me
        return (option.playerIndex is not None and me is not None
                and option.playerIndex != me)


__all__ = ["BoardView"]
