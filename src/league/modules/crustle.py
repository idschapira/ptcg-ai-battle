"""CrustleModule: the SHIPPED CrustleAgent v3, re-expressed as a module.

This is a RETROFIT, not a rewrite. Every rule of
`src/agent_heuristics/crustle_agent.py` at variant="v3" is transcribed
here with its constants lifted into a ThetaSchema whose defaults are
those same constants, so

    ParametricHeuristicAgent(CrustleModule(), theta=defaults)

answers identically to `CrustleAgent(variant="v3")` on every
observation. `tests/test_league_crustle_equivalence.py` asserts that
bit-for-bit over thousands of real decision points; the shipped agent
itself is untouched and remains what production runs.

The rule numbering below is the v3 docstring's — read that module for
the strategy rationale. What this file adds is the GENOME: which
constants are knobs, what their legal bands are, and therefore what
Phase 2 is allowed to move. Bands are chosen to preserve the structural
invariant the Sprint-3 bug taught us — development actions must outrank
attacking — so no reachable theta can make the agent attack into an
undeveloped board.
"""

from __future__ import annotations

from typing import Final, Optional

from cg.api import Option, OptionType, Pokemon, SelectContext

from ...ingestion.card_index import is_cost_payable
from ..board import BoardView
from ..module import DeckModule, SelectPlan
from ..theta import ParamSpec, ThetaSchema

# ---- card ids (facts about the deck, not knobs) ---- #
CRUSTLE: Final[int] = 345
DWEBBLE: Final[int] = 344
GREAT_TUSK: Final[int] = 58
EXPLORERS_GUIDANCE: Final[int] = 1185
JUMBO_ICE_CREAM: Final[int] = 1147
BOSS_ORDERS: Final[int] = 1182
LISIA_APPEAL: Final[int] = 1204
XEROSIC: Final[int] = 1197
COLRESS: Final[int] = 1194
NEUTRAL_ZONE: Final[int] = 1247
SWITCH_ITEM: Final[int] = 1123

SELF_THINNERS: Final[frozenset[int]] = frozenset({
    1086,  # Buddy-Buddy Poffin
    1121,  # Ultra Ball
    1122,  # Pokégear 3.0
    1142,  # Fighting Gong
    1152,  # Poké Pad
    1194,  # Colress's Tenacity
    1185,  # Explorer's Guidance
})

BOARD_BUILDERS: Final[frozenset[int]] = frozenset({
    1086, 1121, 1142, 1152,
})

#: Structural ceiling: no attack score may reach the trainer band.
_TRAINER_BAND: Final[float] = 35.0
_ATTACH_BAND: Final[float] = 55.0

SCHEMA: Final[ThetaSchema] = ThetaSchema((
    # ---- rule (i) / (I): deck-out guards ---- #
    ParamSpec("low_deck", 15, 4, 30, integral=True,
              doc="absolute deck floor: below it, never thin (rule i)"),
    ParamSpec("race_floor", 30, 10, 60, integral=True,
              doc="v3 rule (I): the relative race only matters this low"),
    ParamSpec("suppressed_score", 0.2, 0.0, 0.49,
              doc="score of a suppressed thinner; must stay under END=0.5"),
    # ---- rule (II): board floor ---- #
    ParamSpec("desired_field_floor", 3, 1, 6, integral=True,
              doc="pokémon in play below which rebuilding is urgent"),
    ParamSpec("rebuild_score", 42.0, _TRAINER_BAND, _ATTACH_BAND - 1.0,
              doc="urgent rebuild band: > trainers, < attach"),
    # ---- rule (ii): mill sequencing ---- #
    ParamSpec("guidance_combo_score", 48.0, _TRAINER_BAND, _ATTACH_BAND - 1.0,
              doc="Explorer's Guidance when Land Collapse is live"),
    ParamSpec("land_collapse", 34.8, 20.0, _TRAINER_BAND - 0.05,
              doc="rule (E) mill-over-damage floor; stays under trainers"),
    # ---- rule (iv)/(H): heal + attack pressure ---- #
    ParamSpec("heal_urgent_score", 60.0, _ATTACH_BAND, 90.0,
              doc="Jumbo Ice Cream when the active is worth saving"),
    ParamSpec("jumbo_min_energies", 3, 1, 5, integral=True),
    ParamSpec("jumbo_min_damage", 40, 10, 120, integral=True),
    ParamSpec("attack_pressure_bonus", 4.0, 0.0, 15.0),
    ParamSpec("attack_pressure_cap", 34.5, 20.0, _TRAINER_BAND - 0.05,
              doc="hard ceiling: attacking never outranks development"),
    ParamSpec("attack_band_floor", 20.0, 0.0, 40.0,
              doc="only real attacks get the pressure bonus"),
    # ---- rule (iii): the wall ---- #
    ParamSpec("wall_promote_bonus", 25.0, 0.0, 60.0),
    ParamSpec("wall_useless_malus", 10.0, 0.0, 60.0),
    # ---- rules (A)-(D): supporter ordering ---- #
    ParamSpec("endgame_gust", 46.0, _TRAINER_BAND, _ATTACH_BAND - 1.0),
    ParamSpec("switch_pivot", 45.0, _TRAINER_BAND, _ATTACH_BAND - 1.0),
    ParamSpec("pivot_crustle_offset", 1.0, 0.0, 10.0,
              doc="how far the Crustle pivot sits under the mill pivot"),
    ParamSpec("xerosic_bighand", 44.0, _TRAINER_BAND, _ATTACH_BAND - 1.0),
    ParamSpec("colress_zone", 43.0, _TRAINER_BAND, _ATTACH_BAND - 1.0),
    ParamSpec("pressure_gust", 40.0, _TRAINER_BAND, _ATTACH_BAND - 1.0),
    ParamSpec("gust_idle", 5.0, 0.0, _TRAINER_BAND,
              doc="no trappable target: do not burn the supporter"),
    ParamSpec("retreat_pivot", 16.0, 0.0, 30.0),
    ParamSpec("opp_big_hand", 8, 4, 12, integral=True),
    ParamSpec("xerosic_hand_step", 0.2, 0.0, 2.0),
    ParamSpec("xerosic_hand_cap", 5, 0, 10, integral=True),
    ParamSpec("endgame_opp_deck", 10, 2, 25, integral=True),
    # ---- rule (A): trap-target weights ---- #
    ParamSpec("trap_base", 10.0, 0.0, 40.0),
    ParamSpec("trap_retreat_weight", 7.0, 0.0, 25.0,
              doc="prefer heavy retreat costs — the trap sticks"),
    ParamSpec("trap_energy_weight", 8.0, 0.0, 25.0,
              doc="penalty per energy: a powered body is not a trap"),
    # ---- rule (E): attach priority ---- #
    ParamSpec("tusk_fuel_bonus", 10.0, 0.0, 30.0),
    # ---- rule (F): search / discard valuation ---- #
    ParamSpec("search_zone", 100.0, 0.0, 100.0),
    ParamSpec("search_guidance_live", 95.0, 0.0, 100.0),
    ParamSpec("search_guidance_idle", 40.0, 0.0, 100.0),
    ParamSpec("search_tusk_missing", 85.0, 0.0, 100.0),
    ParamSpec("search_tusk_have", 45.0, 0.0, 100.0),
    ParamSpec("search_colress_zone", 80.0, 0.0, 100.0),
    ParamSpec("search_crustle_ready", 78.0, 0.0, 100.0),
    ParamSpec("search_crustle_idle", 35.0, 0.0, 100.0),
    ParamSpec("search_dwebble_missing", 64.0, 0.0, 100.0),
    ParamSpec("search_dwebble_have", 22.0, 0.0, 100.0),
    ParamSpec("search_gust_live", 60.0, 0.0, 100.0),
    ParamSpec("search_gust_idle", 15.0, 0.0, 100.0),
    ParamSpec("search_energy_fuel", 56.0, 0.0, 100.0),
    ParamSpec("search_energy_idle", 20.0, 0.0, 100.0),
    ParamSpec("search_other", 10.0, 0.0, 100.0),
    ParamSpec("keep_zone", 99.0, 0.0, 100.0),
    ParamSpec("keep_guidance", 95.0, 0.0, 100.0),
    ParamSpec("keep_tusk", 90.0, 0.0, 100.0),
    ParamSpec("keep_crustle_ready", 75.0, 0.0, 100.0),
    ParamSpec("keep_crustle_idle", 40.0, 0.0, 100.0),
    ParamSpec("keep_dwebble", 60.0, 0.0, 100.0),
    ParamSpec("keep_jumbo", 50.0, 0.0, 100.0),
    ParamSpec("keep_energy", 55.0, 0.0, 100.0),
    ParamSpec("keep_gust", 45.0, 0.0, 100.0),
    ParamSpec("keep_other", 20.0, 0.0, 100.0),
    ParamSpec("keep_unknown", 20.0, 0.0, 100.0),
))


class CrustleModule(DeckModule):
    """Crustle LibraryOut stall — the shipped v3 rules, parameterized."""

    name = "crustle"
    decks = ("crustle", "crustle_e10", "crustle_e12", "crustle_kangaskhan")
    schema = SCHEMA

    __slots__ = ("_land_collapse", "_mill_attack_ids")

    def __init__(self) -> None:
        self._land_collapse: tuple = ()
        self._mill_attack_ids: frozenset[int] = frozenset()

    def bind(self, agent) -> None:
        self._land_collapse = tuple(agent._index.attacks_of(GREAT_TUSK))
        # the mill attack references the opponent's deck in its effect
        self._mill_attack_ids = frozenset(
            a.attack_id for a in self._land_collapse
            if a.effect and "deck" in a.effect.lower())

    # ------------------------------------------------------------------ #
    # Deck-specific signals (the generic ones live on BoardView)
    # ------------------------------------------------------------------ #

    def _can_mill(self, pokemon: Pokemon | None) -> bool:
        if pokemon is None or pokemon.id != GREAT_TUSK:
            return False
        energies = [int(e) for e in (pokemon.energies or [])]
        return any(is_cost_payable(attack.cost, energies)
                   for attack in self._land_collapse
                   if attack.attack_id in self._mill_attack_ids)

    def _tusk_ready(self, view: BoardView) -> bool:
        """Great Tusk active with Land Collapse payable right now."""
        active = view.my_active()
        if active is None or active.id != GREAT_TUSK:
            return False
        return self._can_mill(active)

    def _losing_mill_race(self, view: BoardView) -> bool:
        """v3 rule (I): a healthy deck never suppresses setup."""
        mine, theirs = view.deck_counts()
        if mine is None:
            return False
        if mine <= view.theta.i("low_deck"):
            return True
        losing = theirs is not None and mine < theirs
        return mine <= view.theta.i("race_floor") and losing

    def _board_thin(self, view: BoardView) -> bool:
        count = view.field_count()
        return count is not None and count < view.theta.i("desired_field_floor")

    def _trappable(self, view: BoardView, pokemon: Pokemon | None,
                   basic_only: bool = False) -> bool:
        """Kernel trap target: no energy attached and a real retreat cost."""
        card = view.card_of(pokemon)
        if pokemon is None or card is None:
            return False
        if basic_only and card.stage_code != 7:
            return False
        return not (pokemon.energies or []) and (card.retreat_cost or 0) >= 1

    def _opp_trappable_bench(self, view: BoardView,
                             basic_only: bool = False) -> bool:
        return any(self._trappable(view, p, basic_only)
                   for p in view.opp_bench())

    def _endgame_lock(self, view: BoardView) -> bool:
        """Opponent nearly milled out or on their last prize: trap NOW."""
        deck = view.deck_counts()[1]
        if deck is not None and deck <= view.theta.i("endgame_opp_deck"):
            return True
        prizes = view.prizes_left(mine=False)
        return prizes is not None and prizes <= 1

    def _zone_needed(self, view: BoardView) -> bool:
        """Neutralization Zone neither in play nor already in hand."""
        if view.obs.current is None:
            return False  # unreadable board: decline (v3 parity)
        if view.stadium_id() == NEUTRAL_ZONE:
            return False
        return NEUTRAL_ZONE not in view.my_hand_ids()

    def _pivot_on_bench(self, view: BoardView) -> float | None:
        """Score of the pivot the bench offers (None when there is none):
        a mill-ready Great Tusk, or Crustle under opposing ex pressure."""
        theta = view.theta
        bench = view.my_bench()
        if any(self._can_mill(p) for p in bench) and not self._tusk_ready(view):
            return theta["switch_pivot"]
        if (view.opp_active_is_ex() is True
                and any(p.id == CRUSTLE for p in bench)
                and not view.my_active_is(CRUSTLE)):
            return theta["switch_pivot"] - theta["pivot_crustle_offset"]
        return None

    # ------------------------------------------------------------------ #
    # MAIN overlay
    # ------------------------------------------------------------------ #

    def main_score(self, view: BoardView, option: Option,
                   base: float) -> float | None:
        kind = option.type
        if kind == OptionType.PLAY:
            return self._play_score(view, option, base)
        if kind == OptionType.RETREAT:
            return self._retreat_score(view, base)
        if kind == OptionType.ATTACK:
            return self._attack_score(view, option, base)
        return None

    def _play_score(self, view: BoardView, option: Option,
                    base: float) -> float | None:
        theta = view.theta
        card_id = view.agent._wrapper.resolve_card_id(view.obs, option)
        mine, _ = view.deck_counts()
        low_deck = theta.i("low_deck")

        if (card_id in BOARD_BUILDERS and self._board_thin(view)
                and (mine is None or mine > low_deck)):
            # rule (II): urgent board rebuild outranks suppression and
            # breaks the trainer-band tie — only the absolute low-deck
            # invariant (rule i) stays above survival.
            return max(base, theta["rebuild_score"])

        if card_id in SELF_THINNERS and self._losing_mill_race(view):
            # rule (C) precedence: fetching the defensive Zone under ex
            # pressure beats the RELATIVE race trigger — but never the
            # ABSOLUTE low-deck invariant.
            zone_rescue = (card_id == COLRESS and self._zone_needed(view)
                           and view.opp_any_ex()
                           and (mine is None or mine > low_deck))
            if not zone_rescue:
                return theta["suppressed_score"]                  # rule (i)

        if card_id == EXPLORERS_GUIDANCE and self._tusk_ready(view):
            return theta["guidance_combo_score"]                  # rule (ii)

        if card_id == JUMBO_ICE_CREAM:                            # rule (H)
            active = view.my_active()
            healable = (active is not None and active.maxHp
                        and len(active.energies or [])
                        >= theta.i("jumbo_min_energies"))
            if healable and view.damage_on(active) >= theta.i("jumbo_min_damage"):
                return theta["heal_urgent_score"]

        return self._supporter_score(view, card_id)

    def _supporter_score(self, view: BoardView,
                         card_id: int | None) -> float | None:
        """Rules (A)/(B)/(C)/(D); None -> keep the generic score."""
        theta = view.theta
        if card_id in (BOSS_ORDERS, LISIA_APPEAL):                # rule (A)
            basic_only = card_id == LISIA_APPEAL
            if not self._opp_trappable_bench(view, basic_only):
                return theta["gust_idle"]
            if self._endgame_lock(view):
                return theta["endgame_gust"]
            if view.opp_any_ex():
                return theta["pressure_gust"]
            return None
        if card_id == XEROSIC:                                    # rule (B)
            hand = view.opp_hand_count()
            big = theta.i("opp_big_hand")
            if hand is not None and hand >= big:
                return (theta["xerosic_bighand"]
                        + min(hand - big, theta.i("xerosic_hand_cap"))
                        * theta["xerosic_hand_step"])
            return None
        if card_id == COLRESS:                                    # rule (C)
            if self._zone_needed(view) and view.opp_any_ex():
                return theta["colress_zone"]
            return None
        if card_id == SWITCH_ITEM:                                # rule (D)
            return self._pivot_on_bench(view)
        return None

    def _retreat_score(self, view: BoardView, base: float) -> float | None:
        theta = view.theta
        if view.opp_active_is_ex() is True and view.my_active_is(CRUSTLE):
            return theta["suppressed_score"]   # rule (iii): keep the wall
        if self._pivot_on_bench(view) is not None:                # rule (D)
            return theta["retreat_pivot"]
        return None

    def _attack_score(self, view: BoardView, option: Option,
                      base: float) -> float | None:
        theta = view.theta
        if option.attackId in self._mill_attack_ids:              # rule (E)
            base = max(base, theta["land_collapse"])
        if (view.opp_active_is_ex() is False
                and base >= theta["attack_band_floor"]):
            # rule (iv): lean into acting under a non-ex threat — but
            # never above the trainer band (development still first).
            return max(base, min(base + theta["attack_pressure_bonus"],
                                 theta["attack_pressure_cap"]))
        return base

    # ------------------------------------------------------------------ #
    # Promotion / gust targets
    # ------------------------------------------------------------------ #

    def own_pokemon_score(self, view: BoardView, option: Option,
                          for_active: bool, base: float) -> float | None:
        theta = view.theta
        if for_active:
            trap = self._trap_target_score(view, option)           # rule (A)
            if trap is not None:
                return trap
        else:
            return None
        card_id = view.agent._wrapper.resolve_card_id(view.obs, option)
        if card_id != CRUSTLE:
            return None
        if view.opp_field_all_ex():
            return base + theta["wall_promote_bonus"]             # rule (iii)
        if view.opp_active_is_ex() is False:
            return base - theta["wall_useless_malus"]             # rule (iv)
        return None

    def _trap_target_score(self, view: BoardView,
                           option: Option) -> float | None:
        """Gust target choice (Boss/Lisia promote an OPPONENT pokémon):
        trap the heaviest, least-powered body — not the near-KO one."""
        if not view.is_opponent_option(option):
            return None  # our own promotion: not a gust target
        pokemon = view.pokemon_at(option.playerIndex, option.area,
                                  option.index)
        card = view.card_of(pokemon)
        if pokemon is None or card is None:
            return None
        theta = view.theta
        return (theta["trap_base"]
                + theta["trap_retreat_weight"] * (card.retreat_cost or 0)
                - theta["trap_energy_weight"] * len(pokemon.energies or []))

    # ------------------------------------------------------------------ #
    # Attach priority + search/discard handlers
    # ------------------------------------------------------------------ #

    def attach_score(self, view: BoardView, option: Option,
                     base: float) -> float | None:
        target = view.pokemon_at(view.me, option.inPlayArea, option.inPlayIndex)
        if (target is not None and target.id == GREAT_TUSK
                and not self._can_mill(target)):
            return base + view.theta["tusk_fuel_bonus"]           # rule (E)
        return None

    def select_score(self, view: BoardView,
                     context: SelectContext) -> Optional[SelectPlan]:
        select = view.obs.select
        if select is None:
            return None
        if context == SelectContext.TO_HAND:                      # rule (F)
            return (lambda o: self._search_value(view, o),
                    select.minCount, select.maxCount)
        if context in (SelectContext.DISCARD,
                       SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
            return (lambda o: 100.0 - self._keep_value(view, o),
                    select.minCount, select.minCount)
        return None

    def _search_value(self, view: BoardView, option: Option) -> float:
        """What to take into hand (Ultra Ball / Pokégear / Guidance picks)."""
        theta = view.theta
        card = view.option_card(option)
        if card is None:
            return 0.0
        cid = card.card_id
        field_ids = view.my_field_ids()
        if cid == NEUTRAL_ZONE:
            return theta["search_zone"]
        if cid == EXPLORERS_GUIDANCE:
            return (theta["search_guidance_live"] if self._tusk_ready(view)
                    else theta["search_guidance_idle"])
        if cid == GREAT_TUSK:
            return (theta["search_tusk_missing"] if GREAT_TUSK not in field_ids
                    else theta["search_tusk_have"])
        if cid == COLRESS and self._zone_needed(view) and view.opp_any_ex():
            return theta["search_colress_zone"]
        if cid == CRUSTLE:
            return (theta["search_crustle_ready"] if DWEBBLE in field_ids
                    else theta["search_crustle_idle"])
        if cid == DWEBBLE:
            return (theta["search_dwebble_missing"] if DWEBBLE not in field_ids
                    else theta["search_dwebble_have"])
        if cid in (BOSS_ORDERS, LISIA_APPEAL):
            return (theta["search_gust_live"] if self._opp_trappable_bench(view)
                    else theta["search_gust_idle"])
        if card.stage_code in (1, 2):  # energy: fuel the mill first
            needs_fuel = any(p.id == GREAT_TUSK and not self._can_mill(p)
                             for p in view.my_field())
            return (theta["search_energy_fuel"] if needs_fuel
                    else theta["search_energy_idle"])
        return theta["search_other"]

    def _keep_value(self, view: BoardView, option: Option) -> float:
        """How much a card is worth KEEPING (discards throw the lowest)."""
        theta = view.theta
        card = view.option_card(option)
        if card is None:
            return theta["keep_unknown"]  # never the forced keep
        cid = card.card_id
        if cid == NEUTRAL_ZONE:
            return theta["keep_zone"]
        if cid == EXPLORERS_GUIDANCE:
            return theta["keep_guidance"]
        if cid == GREAT_TUSK:
            return theta["keep_tusk"]
        if cid == CRUSTLE:
            has_dwebble = (any(p.id == DWEBBLE for p in view.my_bench())
                           or view.my_active_is(DWEBBLE))
            return (theta["keep_crustle_ready"] if has_dwebble
                    else theta["keep_crustle_idle"])
        if cid == DWEBBLE:
            return theta["keep_dwebble"]
        if cid == JUMBO_ICE_CREAM:
            return theta["keep_jumbo"]
        if card.stage_code in (1, 2):
            return theta["keep_energy"]
        if cid in (BOSS_ORDERS, LISIA_APPEAL):
            return theta["keep_gust"]
        return theta["keep_other"]


__all__ = ["CrustleModule", "SCHEMA"]
