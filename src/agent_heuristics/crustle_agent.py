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

variant="v2" (kernel Elo-1208 blueprint port; v1 stays the SHIPPED
default) adds, all None-safe and capped under the same invariants:
(A) gust-to-trap: Boss's Orders / Lisia's Appeal timing (trappable
    bench = 0 energies + retreat>=1; endgame lock when the opponent's
    deck is nearly milled out or on their last prize) and trap-aware
    TARGET choice (high retreat, no energy — not near-KO);
(B) Xerosic's Machinations when the opponent's hand is huge (>=8),
    never burning the supporter slot of a live Guidance+Tusk turn;
(C) Colress's Tenacity fetches Neutralization Zone under ex pressure;
(D) proactive pivot: Switch/Retreat toward a ready Great Tusk (mill
    online) or toward Crustle under ex pressure;
(E) mill over damage: Land Collapse outranks Giant Tusk; energy
    attachment prioritizes Great Tusk until Land Collapse is payable;
(F) search/discard handlers (TO_HAND / DISCARD) — the generic default
    picked the FIRST indices, wasting Explorer's Guidance picks;
(H) earlier heal: Jumbo at >=40 damage regardless of the threat type.
"""

from __future__ import annotations

from typing import Final

from cg.api import (AreaType, Observation, Option, OptionType, Pokemon,
                    SelectContext)

from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import Card, CardIndex, is_cost_payable
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

# ---- v2 (kernel blueprint) card ids and score bands ---- #
BOSS_ORDERS: Final[int] = 1182
LISIA_APPEAL: Final[int] = 1204
XEROSIC: Final[int] = 1197
COLRESS: Final[int] = 1194
NEUTRAL_ZONE: Final[int] = 1247   # Neutralization Zone (defensive stadium)
SWITCH_ITEM: Final[int] = 1123
DWEBBLE: Final[int] = 344
TERRAKION: Final[int] = 607

# Supporter ordering mirrors the kernel: Guidance combo (48) stays king;
# everything sits above plain trainers (35) and below attach (55).
V2_ENDGAME_GUST: Final[float] = 46.0    # Boss/Lisia to lock the mill win
V2_SWITCH_PIVOT: Final[float] = 45.0
V2_XEROSIC_BIGHAND: Final[float] = 44.0
V2_COLRESS_ZONE: Final[float] = 43.0
V2_PRESSURE_GUST: Final[float] = 40.0
V2_GUST_IDLE: Final[float] = 5.0        # no trappable target: don't waste
V2_RETREAT_PIVOT: Final[float] = 16.0
V2_LAND_COLLAPSE: Final[float] = 34.8   # mill > damage, still < trainers
V2_JUMBO_MIN_DAMAGE: Final[int] = 40
V2_OPP_BIG_HAND: Final[int] = 8
V2_ENDGAME_OPP_DECK: Final[int] = 10


class CrustleAgent(HeuristicAgent):
    """HeuristicAgent + Crustle-stall strategy (see module docstring)."""

    __slots__ = ("_land_collapse", "_mill_attack_ids", "_v2")

    def __init__(
        self,
        seed: int | None = None,
        deck_path: str | None = None,
        index: CardIndex | None = None,
        effects: EffectIndex | None = None,
        variant: str = "v1",
    ) -> None:
        super().__init__(seed=seed, deck_path=deck_path, index=index,
                         effects=effects)
        self._land_collapse = tuple(self._index.attacks_of(GREAT_TUSK))
        # the mill attack references the opponent's deck in its effect
        self._mill_attack_ids = frozenset(
            a.attack_id for a in self._land_collapse
            if a.effect and "deck" in a.effect.lower())
        self._v2 = variant == "v2"

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
        return self._can_mill(active)

    # ------------------------------------------------------------------ #
    # v2 signals (kernel blueprint; all None-safe)
    # ------------------------------------------------------------------ #

    def _can_mill(self, pokemon: Pokemon | None) -> bool:
        if pokemon is None or pokemon.id != GREAT_TUSK:
            return False
        energies = [int(e) for e in (pokemon.energies or [])]
        return any(is_cost_payable(attack.cost, energies)
                   for attack in self._land_collapse
                   if attack.attack_id in self._mill_attack_ids)

    def _my_bench(self, obs: Observation) -> list[Pokemon]:
        state = obs.current
        try:
            return [p for p in (state.players[state.yourIndex].bench or [])
                    if p is not None]
        except (IndexError, TypeError, AttributeError):
            return []

    def _opp_bench(self, obs: Observation) -> list[Pokemon]:
        state = obs.current
        try:
            return [p for p in (state.players[1 - state.yourIndex].bench or [])
                    if p is not None]
        except (IndexError, TypeError, AttributeError):
            return []

    def _trappable(self, pokemon: Pokemon | None,
                   basic_only: bool = False) -> bool:
        """Kernel trap target: no energy attached and a real retreat cost."""
        card = self._card_of(pokemon)
        if pokemon is None or card is None:
            return False
        if basic_only and card.stage_code != 7:
            return False
        return not (pokemon.energies or []) and (card.retreat_cost or 0) >= 1

    def _opp_trappable_bench(self, obs: Observation,
                             basic_only: bool = False) -> bool:
        return any(self._trappable(p, basic_only) for p in self._opp_bench(obs))

    def _endgame_lock(self, obs: Observation) -> bool:
        """Opponent nearly milled out or on their last prize: trap NOW."""
        state = obs.current
        try:
            opp = state.players[1 - state.yourIndex]
            if opp.deckCount is not None and opp.deckCount <= V2_ENDGAME_OPP_DECK:
                return True
            return opp.prize is not None and len(opp.prize) <= 1
        except (IndexError, TypeError, AttributeError):
            return False

    def _opp_any_ex(self, obs: Observation) -> bool:
        if self._opp_active_is_ex(obs) is True:
            return True
        for pokemon in self._opp_bench(obs):
            card = self._card_of(pokemon)
            if card is not None and (card.is_ex or card.is_mega_ex):
                return True
        return False

    def _opp_hand_count(self, obs: Observation) -> int | None:
        state = obs.current
        try:
            return state.players[1 - state.yourIndex].handCount
        except (IndexError, TypeError, AttributeError):
            return None

    def _zone_needed(self, obs: Observation) -> bool:
        """Neutralization Zone neither in play nor already in hand."""
        state = obs.current
        try:
            stadium = state.stadium[0] if state.stadium else None
            if stadium is not None and stadium.id == NEUTRAL_ZONE:
                return False
            hand = state.players[state.yourIndex].hand or []
            return all(c is None or c.id != NEUTRAL_ZONE for c in hand)
        except (IndexError, TypeError, AttributeError):
            return False

    def _pivot_on_bench(self, obs: Observation) -> float | None:
        """Score of the pivot the bench offers (None when there is none):
        a mill-ready Great Tusk, or Crustle under opposing ex pressure."""
        bench = self._my_bench(obs)
        if any(self._can_mill(p) for p in bench) and not self._great_tusk_ready(obs):
            return V2_SWITCH_PIVOT
        if (self._opp_active_is_ex(obs) is True
                and any(p.id == CRUSTLE for p in bench)
                and not self._my_active_is(obs, CRUSTLE)):
            return V2_SWITCH_PIVOT - 1.0
        return None

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
                # v2 rule (C) precedence: fetching the defensive Zone under
                # ex pressure beats the RELATIVE race trigger — but never
                # the ABSOLUTE low-deck invariant (rule i stays king there).
                mine, _ = self._deck_counts(obs)
                zone_rescue = (self._v2 and card_id == COLRESS
                               and self._zone_needed(obs)
                               and self._opp_any_ex(obs)
                               and (mine is None or mine > LOW_DECK))
                if not zone_rescue:
                    return SUPPRESSED_SCORE                      # rule (i)
            if (card_id == EXPLORERS_GUIDANCE
                    and self._great_tusk_ready(obs)):
                return GUIDANCE_COMBO_SCORE                      # rule (ii)
            if card_id == JUMBO_ICE_CREAM:
                active = self._my_active(obs)
                healable = (active is not None and active.maxHp
                            and len(active.energies or []) >= JUMBO_MIN_ENERGIES)
                if healable and self._v2:                        # rule (H)
                    if active.maxHp - active.hp >= V2_JUMBO_MIN_DAMAGE:
                        return HEAL_URGENT_SCORE
                elif (healable and opp_is_ex is False
                        and active.maxHp - active.hp >= JUMBO_MIN_DAMAGE):
                    return HEAL_URGENT_SCORE                     # rule (iv)
            if self._v2:
                v2 = self._v2_play_adjust(obs, card_id)
                if v2 is not None:
                    return v2
            return base
        if kind == OptionType.RETREAT:
            if opp_is_ex is True and self._my_active_is(obs, CRUSTLE):
                return SUPPRESSED_SCORE  # rule (iii): never break the wall
            if self._v2:                                         # rule (D)
                pivot = self._pivot_on_bench(obs)
                if pivot is not None:
                    return V2_RETREAT_PIVOT
            return base
        if kind == OptionType.ATTACK:
            if self._v2 and option.attackId in self._mill_attack_ids:
                base = max(base, V2_LAND_COLLAPSE)               # rule (E)
            if opp_is_ex is False and base >= _ATTACK_BAND_FLOOR:
                # rule (iv): under a non-ex threat, lean into acting — but
                # never above the trainer band (development still first)
                return max(base, min(base + ATTACK_PRESSURE_BONUS,
                                     ATTACK_PRESSURE_CAP))
            return base
        return base

    def _v2_play_adjust(self, obs: Observation,
                        card_id: int | None) -> float | None:
        """v2 PLAY rules (A/B/C/D); None -> fall through to the base score."""
        if card_id in (BOSS_ORDERS, LISIA_APPEAL):               # rule (A)
            basic_only = card_id == LISIA_APPEAL
            if not self._opp_trappable_bench(obs, basic_only):
                return V2_GUST_IDLE
            if self._endgame_lock(obs):
                return V2_ENDGAME_GUST
            if self._opp_any_ex(obs):
                return V2_PRESSURE_GUST
            return None
        if card_id == XEROSIC:                                   # rule (B)
            hand = self._opp_hand_count(obs)
            if hand is not None and hand >= V2_OPP_BIG_HAND:
                return V2_XEROSIC_BIGHAND + min(hand - V2_OPP_BIG_HAND, 5) * 0.2
            return None
        if card_id == COLRESS:                                   # rule (C)
            if self._zone_needed(obs) and self._opp_any_ex(obs):
                return V2_COLRESS_ZONE
            return None
        if card_id == SWITCH_ITEM:                               # rule (D)
            return self._pivot_on_bench(obs)
        return None

    def _own_pokemon_score(self, obs: Observation, option: Option,
                           for_active: bool) -> float:
        base = super()._own_pokemon_score(obs, option, for_active)
        try:
            if self._v2 and for_active:
                trap = self._v2_trap_target_score(obs, option)
                if trap is not None:
                    return trap                                  # rule (A)
            if not for_active:
                return base
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

    def _v2_trap_target_score(self, obs: Observation,
                              option: Option) -> float | None:
        """Gust target choice (Boss/Lisia promote an OPPONENT pokémon):
        trap the heaviest, least-powered body — not the near-KO one."""
        state = obs.current
        if (state is None or option.playerIndex is None
                or option.playerIndex == state.yourIndex):
            return None  # our own promotion: not a gust target
        pokemon = self._pokemon_at(state, option.playerIndex,
                                   option.area, option.index)
        card = self._card_of(pokemon)
        if pokemon is None or card is None:
            return None
        return (10.0 + 7.0 * (card.retreat_cost or 0)
                - 8.0 * len(pokemon.energies or []))


    # ------------------------------------------------------------------ #
    # v2 selection handlers (search / discard) + attach priority
    # ------------------------------------------------------------------ #

    def _decide(self, obs: Observation) -> list[int]:
        select = obs.select
        if self._v2 and select is not None:
            try:
                if select.context == SelectContext.TO_HAND:      # rule (F)
                    return self._pick_top(
                        select.option, select.minCount, select.maxCount,
                        lambda i, o: self._search_value(obs, o))
                if select.context in (SelectContext.DISCARD,
                                      SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
                    return self._pick_top(
                        select.option, select.minCount, select.minCount,
                        lambda i, o: 100.0 - self._keep_value(obs, o))
            except Exception:
                pass  # fall through to the generic (always legal) path
        return super()._decide(obs)

    def _select_card(self, obs: Observation, option: Option) -> Card | None:
        """Resolve a CARD option, including DECK-area picks (searches
        expose the deck through obs.select.deck, which the generic
        wrapper resolver treats as face-down)."""
        card_id = self._wrapper.resolve_card_id(obs, option)
        if card_id is None and option.area == AreaType.DECK:
            try:
                deck = obs.select.deck if obs.select is not None else None
                entry = deck[option.index] if deck else None
                card_id = entry.id if entry is not None else None
            except (IndexError, TypeError, AttributeError):
                card_id = None
        return self._index.get_card(card_id) if card_id is not None else None

    def _search_value(self, obs: Observation, option: Option) -> float:
        """What to take into hand (Ultra Ball / Pokégear / Guidance picks)."""
        card = self._select_card(obs, option)
        if card is None:
            return 0.0
        cid = card.card_id
        field_ids = [p.id for p in
                     ([self._my_active(obs)] if self._my_active(obs) else [])
                     + self._my_bench(obs)]
        if cid == NEUTRAL_ZONE:
            return 100.0
        if cid == EXPLORERS_GUIDANCE:
            return 95.0 if self._great_tusk_ready(obs) else 40.0
        if cid == GREAT_TUSK:
            return 85.0 if GREAT_TUSK not in field_ids else 45.0
        if cid == COLRESS and self._zone_needed(obs) and self._opp_any_ex(obs):
            return 80.0
        if cid == CRUSTLE:
            return 78.0 if DWEBBLE in field_ids else 35.0
        if cid == DWEBBLE:
            return 64.0 if DWEBBLE not in field_ids else 22.0
        if cid in (BOSS_ORDERS, LISIA_APPEAL):
            return 60.0 if self._opp_trappable_bench(obs) else 15.0
        if card.stage_code in (1, 2):  # energy: fuel the mill first
            needs_fuel = any(p.id == GREAT_TUSK and not self._can_mill(p)
                             for p in self._my_bench(obs)
                             + ([self._my_active(obs)]
                                if self._my_active(obs) else []))
            return 56.0 if needs_fuel else 20.0
        return 10.0

    def _keep_value(self, obs: Observation, option: Option) -> float:
        """How much a card is worth KEEPING (discards throw the lowest)."""
        card = self._select_card(obs, option)
        if card is None:
            return 20.0  # unknown: middle value, never the forced keep
        cid = card.card_id
        if cid == NEUTRAL_ZONE:
            return 99.0
        if cid == EXPLORERS_GUIDANCE:
            return 95.0
        if cid == GREAT_TUSK:
            return 90.0
        if cid == CRUSTLE:
            return 75.0 if any(p.id == DWEBBLE for p in self._my_bench(obs))\
                or self._my_active_is(obs, DWEBBLE) else 40.0
        if cid == DWEBBLE:
            return 60.0
        if cid == JUMBO_ICE_CREAM:
            return 50.0
        if card.stage_code in (1, 2):
            return 55.0
        if cid in (BOSS_ORDERS, LISIA_APPEAL):
            return 45.0
        return 20.0

    def _attach_score(self, obs: Observation, option: Option) -> float:
        base = super()._attach_score(obs, option)
        if not self._v2:
            return base
        try:                                                     # rule (E)
            state = obs.current
            target = self._pokemon_at(state, state.yourIndex,
                                      option.inPlayArea, option.inPlayIndex)
            if (target is not None and target.id == GREAT_TUSK
                    and not self._can_mill(target)):
                return base + 10.0  # mill fuel outranks any other target
            return base
        except Exception:
            return base


__all__ = ["CrustleAgent"]
