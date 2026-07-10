"""CrustleAgent: Crustle-deck strategy overlay on the generic heuristic.

Kernel-inspired rules (soutasakurai Elo-1208 rule groups), layered as
score ADJUSTMENTS over HeuristicAgent's generic bands — every rule is
None-safe and falls back to the generic score when its signals are
missing, so the agent stays legal on any deck/observation:

(i)   ANTI-SELF-MILL: deck searches/thinning (Poffin, Ultra Ball, Poké
      Pad, Fighting Gong, Pokégear, Colress, Explorer's Guidance) are
      suppressed below the END score when our own deck is low OR we are
      LOSING the deck-out race (my deck < opponent's). Deck-out is a
      LOSS for the emptied player (tests/test_crustle_stall_contract).
(ii)  MILL SEQUENCING: Explorer's Guidance is an *Ancient* Supporter —
      Great Tusk's Land Collapse mills 1 + 3 more if an Ancient
      Supporter was played this turn. When Great Tusk is active, ready
      to attack, and our deck is healthy, Guidance outranks every other
      trainer (still below attach/evolve, so development goes first).
(iii) WALL LOGIC: with Crustle active ("Mysterious Rock Inn" prevents
      ALL damage from opposing {ex}) and the opponent's active being an
      ex, never retreat the wall; promotion prompts prefer Crustle when
      the opponent's whole field is ex.
(iv)  NON-EX THREAT: the ability does NOT block non-ex attackers. When
      the opponent's active is non-ex: urgent Jumbo Ice Cream heal
      (80 HP, needs 3+ energies on the active), no wall bias, and a
      tie-break bonus toward attacking (capped below the trainer band —
      development must still come first: Sprint-3 lesson).
"""

from __future__ import annotations

from typing import Final

from cg.api import Observation, Option, OptionType

from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex, is_cost_payable
from .heuristic_agent import HeuristicAgent

CRUSTLE: Final[int] = 345
GREAT_TUSK: Final[int] = 58
EXPLORERS_GUIDANCE: Final[int] = 1185
JUMBO_ICE_CREAM: Final[int] = 1147

# Items/Supporters that consume our OWN deck (search/thin/discard).
SELF_THINNERS: Final[frozenset[int]] = frozenset({
    1086,  # Buddy-Buddy Poffin (search 2 basics)
    1121,  # Ultra Ball (search 1, discard 2 from hand)
    1122,  # Pokégear 3.0 (top-7 look, take supporter)
    1142,  # Fighting Gong (search {F} energy/pokémon)
    1152,  # Poké Pad (search non-Rule-Box pokémon)
    1194,  # Colress's Tenacity (search stadium + energy)
    1185,  # Explorer's Guidance (top-6: keep 2, DISCARD 4 — heaviest)
})

LOW_DECK: Final[int] = 15
SUPPRESSED_SCORE: Final[float] = 0.2   # below END (0.5): pass instead
GUIDANCE_COMBO_SCORE: Final[float] = 48.0  # above trainers, below attach
HEAL_URGENT_SCORE: Final[float] = 60.0     # above trainers/attach
JUMBO_MIN_ENERGIES: Final[int] = 3
JUMBO_MIN_DAMAGE: Final[int] = 60
ATTACK_PRESSURE_BONUS: Final[float] = 4.0
ATTACK_PRESSURE_CAP: Final[float] = 34.5   # never outrank the trainer band
WALL_PROMOTE_BONUS: Final[float] = 25.0
WALL_USELESS_MALUS: Final[float] = 10.0
_ATTACK_BAND_FLOOR: Final[float] = 20.0


class CrustleAgent(HeuristicAgent):
    """HeuristicAgent + Crustle-stall strategy (see module docstring)."""

    __slots__ = ("_land_collapse",)

    def __init__(
        self,
        seed: int | None = None,
        deck_path: str | None = None,
        index: CardIndex | None = None,
        effects: EffectIndex | None = None,
    ) -> None:
        super().__init__(seed=seed, deck_path=deck_path, index=index,
                         effects=effects)
        self._land_collapse = tuple(self._index.attacks_of(GREAT_TUSK))

    # ------------------------------------------------------------------ #
    # Signals (all None-safe: unknown -> None / False)
    # ------------------------------------------------------------------ #

    def _deck_counts(self, obs: Observation) -> tuple[int | None, int | None]:
        """(my deckCount, opponent deckCount)."""
        state = obs.current
        if state is None or state.yourIndex is None:
            return None, None
        try:
            me = state.players[state.yourIndex]
            opp = state.players[1 - state.yourIndex]
            return me.deckCount, opp.deckCount
        except (IndexError, TypeError, AttributeError):
            return None, None

    def _losing_mill_race(self, obs: Observation) -> bool:
        """True when thinning our own deck is a liability."""
        mine, theirs = self._deck_counts(obs)
        if mine is None:
            return False
        if mine <= LOW_DECK:
            return True
        return theirs is not None and mine < theirs

    def _opp_active_is_ex(self, obs: Observation) -> bool | None:
        """True/False when the opposing active is known; None otherwise."""
        card = self._card_of(self._opp_active(obs))
        if card is None:
            return None
        return bool(card.is_ex or card.is_mega_ex)

    def _opp_field_all_ex(self, obs: Observation) -> bool:
        """Known active is ex AND every known bench entry is ex."""
        if self._opp_active_is_ex(obs) is not True:
            return False
        state = obs.current
        try:
            bench = state.players[1 - state.yourIndex].bench or []
        except (IndexError, TypeError, AttributeError):
            return False
        for pokemon in bench:
            card = self._card_of(pokemon)
            if card is None or not (card.is_ex or card.is_mega_ex):
                return False
        return True

    def _my_active_is(self, obs: Observation, card_id: int) -> bool:
        active = self._my_active(obs)
        return active is not None and active.id == card_id

    def _great_tusk_ready(self, obs: Observation) -> bool:
        """Great Tusk active with Land Collapse payable right now."""
        active = self._my_active(obs)
        if active is None or active.id != GREAT_TUSK:
            return False
        energies = [int(e) for e in (active.energies or [])]
        return any(is_cost_payable(attack.cost, energies)
                   for attack in self._land_collapse)

    # ------------------------------------------------------------------ #
    # Score overlays
    # ------------------------------------------------------------------ #

    def _main_score(self, obs: Observation, option: Option) -> float:
        base = super()._main_score(obs, option)
        try:
            return self._crustle_adjust(obs, option, base)
        except Exception:
            return base

    def _crustle_adjust(self, obs: Observation, option: Option,
                        base: float) -> float:
        kind = option.type
        opp_is_ex = self._opp_active_is_ex(obs)

        if kind == OptionType.PLAY:
            card_id = self._wrapper.resolve_card_id(obs, option)
            if card_id in SELF_THINNERS and self._losing_mill_race(obs):
                return SUPPRESSED_SCORE                          # rule (i)
            if (card_id == EXPLORERS_GUIDANCE
                    and self._great_tusk_ready(obs)):
                return GUIDANCE_COMBO_SCORE                      # rule (ii)
            if card_id == JUMBO_ICE_CREAM and opp_is_ex is False:
                active = self._my_active(obs)
                if (active is not None and active.maxHp
                        and active.maxHp - active.hp >= JUMBO_MIN_DAMAGE
                        and len(active.energies or []) >= JUMBO_MIN_ENERGIES):
                    return HEAL_URGENT_SCORE                     # rule (iv)
            return base
        if (kind == OptionType.RETREAT and opp_is_ex is True
                and self._my_active_is(obs, CRUSTLE)):
            return SUPPRESSED_SCORE  # rule (iii): never break the wall
        if (kind == OptionType.ATTACK and opp_is_ex is False
                and base >= _ATTACK_BAND_FLOOR):
            # rule (iv): under a non-ex threat, lean into acting — but
            # never above the trainer band (development still first)
            return max(base, min(base + ATTACK_PRESSURE_BONUS,
                                 ATTACK_PRESSURE_CAP))
        return base

    def _own_pokemon_score(self, obs: Observation, option: Option,
                           for_active: bool) -> float:
        base = super()._own_pokemon_score(obs, option, for_active)
        if not for_active:
            return base
        try:
            card_id = self._wrapper.resolve_card_id(obs, option)
            if card_id != CRUSTLE:
                return base
            if self._opp_field_all_ex(obs):
                return base + WALL_PROMOTE_BONUS                 # rule (iii)
            if self._opp_active_is_ex(obs) is False:
                return base - WALL_USELESS_MALUS                 # rule (iv)
            return base
        except Exception:
            return base


__all__ = ["CrustleAgent"]
