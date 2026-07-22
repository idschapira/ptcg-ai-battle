"""GrimmsnarlModule: Marnie's Grimmsnarl ex — setup, counters, targeting.

The second module, and the one that proves the framework generalizes:
written from scratch against the deck's actual engine behaviour rather
than retrofitted. The deck (data/decks/meta_grimmsnarl.csv) is a stage-2
aggro-tempo shell:

  Marnie's Impidimp -> Morgrem -> Grimmsnarl ex (320 HP, Shadow Bullet
  180 + 30 to a bench). Its ability PUNK UP fires when Grimmsnarl is
  played FROM HAND to evolve: search up to 5 Basic {D} Energy and attach
  them to your Marnie's Pokémon. That single trigger is the deck's whole
  energy engine, which is why Rare Candy and evolving outrank everything
  (the BC-Grimmsnarl policy that seeded this theta plays EVOLVE 61.9% and
  ABILITY 80.8% of the time it is legal — far above the generic agent).

  Munkidori's ADRENA-BRAIN moves up to 3 damage counters from one of our
  Pokémon onto one of theirs, once per turn, if it has {D} attached. It
  is a repeatable 30-damage reach AND a heal for the 320 HP attacker —
  hence the counter-engine rules: pick the destination that converts to
  a KO, take the counters off the body worth the most prizes.

  Froslass's FREEZING SHROUD chips every Pokémon WITH AN ABILITY on both
  sides during checkup. Note the cost: Munkidori and Grimmsnarl ex both
  have abilities, so Froslass damages our own engine. It is a knob
  (`froslass_bench_value`), deliberately not a conviction.

Engine facts verified by probe, not assumed from card text (the engine
exposes no rules text for Trainers):
  Boss's Orders -> SWITCH prompt on an OPPONENT pokémon (the gust).
  Team Rocket's Petrel, Poké Pad, Night Stretcher, Pokégear, Dawn ->
    TO_HAND (they are SEARCH cards; Petrel is NOT a targeting card).
  Buddy-Buddy Poffin -> TO_BENCH.  Rare Candy -> EVOLVE.

ROBUSTNESS is the property that makes this worth putting in the league:
every rule is keyed on card ids that may simply be absent, and each hook
returns None when its signals are missing, so the module degrades to
plain generic play on an unfamiliar board instead of misfiring. That is
asserted directly in tests/test_league_grimmsnarl.py by piloting a
completely different deck with it.
"""

from __future__ import annotations

from typing import Final, Optional

from cg.api import Option, OptionType, Pokemon, SelectContext

from ..board import BoardView
from ..module import DeckModule, SelectPlan
from ..theta import ParamSpec, ThetaSchema

# ---- card ids (facts about the deck, not knobs) ---- #
IMPIDIMP: Final[int] = 646
MORGREM: Final[int] = 647
GRIMMSNARL_EX: Final[int] = 648
SNORUNT: Final[int] = 860
FROSLASS: Final[int] = 104
MUNKIDORI: Final[int] = 112
DARK_ENERGY: Final[int] = 7

RARE_CANDY: Final[int] = 1079
UNFAIR_STAMP: Final[int] = 1080
POFFIN: Final[int] = 1086
NIGHT_STRETCHER: Final[int] = 1097
POKEGEAR: Final[int] = 1122
TOOL_SCRAPPER: Final[int] = 1137
POKE_PAD: Final[int] = 1152
BOSS_ORDERS: Final[int] = 1182
PETREL: Final[int] = 1219
LILLIE: Final[int] = 1227
DAWN: Final[int] = 1231
SPIKEMUTH_GYM: Final[int] = 1259

#: The Marnie's line — Punk Up only attaches to these.
MARNIE_LINE: Final[frozenset[int]] = frozenset({IMPIDIMP, MORGREM,
                                                GRIMMSNARL_EX})
#: Our own Pokémon that Freezing Shroud chips (they have Abilities).
ABILITY_HOLDERS: Final[frozenset[int]] = frozenset({GRIMMSNARL_EX, MUNKIDORI,
                                                    FROSLASS})
#: Searches that put a body on the board — the anti-brick rescue set.
BOARD_BUILDERS: Final[frozenset[int]] = frozenset({POFFIN, POKE_PAD,
                                                   NIGHT_STRETCHER, PETREL})

_TRAINER_BAND: Final[float] = 35.0
_ATTACH_BAND: Final[float] = 55.0
_PLAY_POKEMON_BAND: Final[float] = 70.0
_EVOLVE_BAND: Final[float] = 80.0

#: One Adrena-Brain activation moves up to 3 counters = 30 damage.
ADRENA_BRAIN_DAMAGE: Final[int] = 30

SCHEMA: Final[ThetaSchema] = ThetaSchema((
    # ---- setup: the stage-2 climb is the deck ---- #
    ParamSpec("rare_candy_score", 78.0, _ATTACH_BAND, _EVOLVE_BAND - 1.0,
              doc="Rare Candy: the shortcut to Punk Up; just under evolving"),
    ParamSpec("evolve_grimmsnarl_bonus", 15.0, 0.0, 40.0,
              doc="evolving INTO Grimmsnarl ex also fires Punk Up (5 energy)"),
    ParamSpec("evolve_line_bonus", 5.0, 0.0, 25.0,
              doc="any other Marnie's / Froslass evolution step"),
    ParamSpec("desired_field_floor", 3, 1, 6, integral=True),
    ParamSpec("rebuild_score", 44.0, _TRAINER_BAND, _ATTACH_BAND - 1.0,
              doc="thin board: a search that puts bodies down is urgent"),
    # ---- the ability engine (BC prior: ABILITY 80.8%) ---- #
    # The two bands below are bounded so their SUM can never reach the
    # attach band (55): Adrena-Brain only works with {D} attached, so
    # attaching must always be sequenced first, in the same turn.
    ParamSpec("ability_band", 50.0, _TRAINER_BAND, 50.0,
              doc="generic band is 40; this deck's abilities are its engine"),
    ParamSpec("adrena_brain_live_bonus", 4.0, 0.0, 4.5,
              doc="extra when counters are actually available to move"),
    # ---- counter engine: where the 30 damage goes ---- #
    ParamSpec("counter_base", 10.0, 0.0, 40.0),
    ParamSpec("counter_lethal_bonus", 60.0, 0.0, 120.0,
              doc="destination the moved counters would actually KO"),
    ParamSpec("counter_ex_bonus", 15.0, 0.0, 50.0,
              doc="two prizes are worth more than one"),
    ParamSpec("counter_damage_weight", 0.3, 0.0, 2.0,
              doc="per point of damage already on the destination"),
    ParamSpec("counter_tera_bench_malus", 100.0, 0.0, 200.0,
              doc="Tera on the bench nullifies damage — never spend it there"),
    ParamSpec("source_base", 10.0, 0.0, 40.0),
    ParamSpec("source_damage_weight", 0.4, 0.0, 2.0),
    ParamSpec("source_ex_bonus", 25.0, 0.0, 60.0,
              doc="pull counters off the 320 HP attacker first"),
    # ---- Boss's Orders: drag the prize, do not trap ---- #
    ParamSpec("boss_live_score", 46.0, _TRAINER_BAND, _ATTACH_BAND - 1.0),
    ParamSpec("boss_idle_score", 5.0, 0.0, _TRAINER_BAND,
              doc="no worthwhile drag: keep the supporter"),
    ParamSpec("boss_target_hp_threshold", 180, 60, 330, integral=True,
              doc="HP at or under which a benched body is worth dragging"),
    ParamSpec("drag_base", 10.0, 0.0, 40.0),
    ParamSpec("drag_lethal_bonus", 40.0, 0.0, 100.0),
    ParamSpec("drag_ex_bonus", 12.0, 0.0, 50.0),
    ParamSpec("drag_hp_weight", 0.08, 0.0, 1.0,
              doc="per HP remaining: prefer the body we can finish"),
    # ---- supporters / stadium ---- #
    ParamSpec("draw_supporter_score", 38.0, 0.0, _ATTACH_BAND - 1.0,
              doc="Lillie's Determination / Dawn refills"),
    ParamSpec("unfair_stamp_score", 40.0, 0.0, _ATTACH_BAND - 1.0),
    ParamSpec("stadium_score", 36.0, 0.0, _ATTACH_BAND - 1.0),
    ParamSpec("froslass_bench_value", 8.0, -20.0, 40.0,
              doc="Freezing Shroud chips OUR ability holders too: a cost"),
    # ---- attach: fuel the line, seed Munkidori ---- #
    ParamSpec("attach_marnie_bonus", 8.0, 0.0, 25.0),
    ParamSpec("attach_munkidori_seed", 12.0, 0.0, 30.0,
              doc="the FIRST {D} on Munkidori switches Adrena-Brain on"),
    # ---- benching (Poffin) ---- #
    ParamSpec("bench_impidimp", 90.0, 0.0, 120.0),
    ParamSpec("bench_munkidori", 70.0, 0.0, 120.0),
    ParamSpec("bench_snorunt", 45.0, 0.0, 120.0),
    # ---- search valuation (TO_HAND) ---- #
    ParamSpec("search_grimmsnarl_live", 95.0, 0.0, 100.0,
              doc="a Morgrem is waiting: this card is the turn"),
    ParamSpec("search_grimmsnarl_idle", 45.0, 0.0, 100.0),
    ParamSpec("search_rare_candy", 85.0, 0.0, 100.0),
    ParamSpec("search_impidimp_missing", 80.0, 0.0, 100.0),
    ParamSpec("search_impidimp_have", 25.0, 0.0, 100.0),
    ParamSpec("search_munkidori_missing", 75.0, 0.0, 100.0),
    ParamSpec("search_munkidori_have", 20.0, 0.0, 100.0),
    ParamSpec("search_boss_live", 65.0, 0.0, 100.0),
    ParamSpec("search_boss_idle", 15.0, 0.0, 100.0),
    ParamSpec("search_energy_short", 60.0, 0.0, 100.0),
    ParamSpec("search_energy_idle", 18.0, 0.0, 100.0),
    ParamSpec("search_froslass_line", 40.0, 0.0, 100.0),
    ParamSpec("search_builder", 50.0, 0.0, 100.0),
    ParamSpec("search_other", 10.0, 0.0, 100.0),
    # ---- discard valuation ---- #
    ParamSpec("keep_grimmsnarl", 95.0, 0.0, 100.0),
    ParamSpec("keep_rare_candy", 88.0, 0.0, 100.0),
    ParamSpec("keep_marnie_line", 70.0, 0.0, 100.0),
    ParamSpec("keep_munkidori", 65.0, 0.0, 100.0),
    ParamSpec("keep_energy", 55.0, 0.0, 100.0),
    ParamSpec("keep_boss", 45.0, 0.0, 100.0),
    ParamSpec("keep_other", 20.0, 0.0, 100.0),
    ParamSpec("keep_unknown", 20.0, 0.0, 100.0),
))


class GrimmsnarlModule(DeckModule):
    """Marnie's Grimmsnarl ex: setup -> Punk Up -> counters -> prizes."""

    name = "grimmsnarl"
    decks = ("grimmsnarl",)
    schema = SCHEMA

    __slots__ = ()

    # ------------------------------------------------------------------ #
    # Deck-specific signals (None-safe: absent card -> the rule declines)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _dark_count(pokemon: Pokemon | None) -> int:
        if pokemon is None:
            return 0
        try:
            return sum(1 for e in (pokemon.energies or [])
                       if int(e) == DARK_ENERGY)
        except (TypeError, ValueError):
            return 0

    def _has_on_field(self, view: BoardView, card_id: int) -> bool:
        return view.count_in_field(card_id) > 0

    def _grimmsnarl_target_waiting(self, view: BoardView) -> bool:
        """A Morgrem in play means Grimmsnarl ex in hand is the whole turn."""
        return self._has_on_field(view, MORGREM)

    def _adrena_brain_live(self, view: BoardView) -> bool:
        """Munkidori powered, with counters on our side worth moving."""
        munkidori = [p for p in view.my_field() if p.id == MUNKIDORI]
        if not any(self._dark_count(p) > 0 for p in munkidori):
            return False
        return any(view.damage_on(p) > 0 for p in view.my_field())

    def _board_thin(self, view: BoardView) -> bool:
        count = view.field_count()
        return count is not None and count < view.theta.i("desired_field_floor")

    def _energy_short(self, view: BoardView) -> bool:
        """The Marnie's body in front cannot pay for Shadow Bullet yet."""
        active = view.my_active()
        if active is None or active.id not in MARNIE_LINE:
            return False
        return self._dark_count(active) < 2

    def _drag_worth_it(self, view: BoardView) -> bool:
        """Somebody on their bench is a better target than their active."""
        threshold = view.theta.i("boss_target_hp_threshold")
        for pokemon in view.opp_bench():
            try:
                if pokemon.hp is not None and int(pokemon.hp) <= threshold:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    # ------------------------------------------------------------------ #
    # MAIN overlay
    # ------------------------------------------------------------------ #

    def main_score(self, view: BoardView, option: Option,
                   base: float) -> float | None:
        kind = option.type
        if kind == OptionType.ABILITY:
            return self._ability_score(view, base)
        if kind == OptionType.EVOLVE:
            return self._evolve_score(view, option, base)
        if kind == OptionType.PLAY:
            return self._play_score(view, option, base)
        return None

    def _ability_score(self, view: BoardView, base: float) -> float | None:
        """This deck's abilities ARE its engine: raise them off the floor.

        Grimmsnarl's Punk Up fires on evolution (not from this menu), so
        what shows up here is Adrena-Brain — free reach and a free heal."""
        theta = view.theta
        if not self._has_on_field(view, MUNKIDORI):
            return None  # unfamiliar ability: leave the generic band alone
        score = theta["ability_band"]
        if self._adrena_brain_live(view):
            score += theta["adrena_brain_live_bonus"]
        return max(base, score)

    def _evolve_score(self, view: BoardView, option: Option,
                      base: float) -> float | None:
        theta = view.theta
        card_id = view.option_card_id(option)
        if card_id == GRIMMSNARL_EX:
            # Punk Up: 5 Basic {D} onto the Marnie's line in one action.
            return base + theta["evolve_grimmsnarl_bonus"]
        if card_id in (MORGREM, FROSLASS):
            return base + theta["evolve_line_bonus"]
        return None

    def _play_score(self, view: BoardView, option: Option,
                    base: float) -> float | None:
        theta = view.theta
        card_id = view.option_card_id(option)
        if card_id is None:
            return None

        if card_id == RARE_CANDY:
            return theta["rare_candy_score"]
        if card_id == BOSS_ORDERS:
            return (theta["boss_live_score"] if self._drag_worth_it(view)
                    else theta["boss_idle_score"])
        if card_id in BOARD_BUILDERS and self._board_thin(view):
            return max(base, theta["rebuild_score"])
        if card_id in (LILLIE, DAWN):
            return theta["draw_supporter_score"]
        if card_id == UNFAIR_STAMP:
            return theta["unfair_stamp_score"]
        if card_id == SPIKEMUTH_GYM:
            return theta["stadium_score"]
        if card_id in (SNORUNT, FROSLASS):
            # Freezing Shroud is symmetric and we run three ability
            # holders: benching the line is a trade, not a freebie.
            return base + theta["froslass_bench_value"]
        return None

    # ------------------------------------------------------------------ #
    # Promotion / benching / gust targets
    # ------------------------------------------------------------------ #

    def own_pokemon_score(self, view: BoardView, option: Option,
                          for_active: bool, base: float) -> float | None:
        if for_active and view.is_opponent_option(option):
            return self._drag_target_score(view, option)   # Boss's Orders
        card_id = view.option_card_id(option)
        if card_id is None:
            return None
        if not for_active:
            return self._bench_score(view, card_id, base)
        return None

    def _bench_score(self, view: BoardView, card_id: int,
                     base: float) -> float | None:
        """Poffin / setup benching: bodies that climb beat bodies that chip."""
        theta = view.theta
        if card_id == IMPIDIMP:
            return theta["bench_impidimp"]
        if card_id == MUNKIDORI:
            return theta["bench_munkidori"]
        if card_id == SNORUNT:
            return theta["bench_snorunt"]
        return None

    def _drag_target_score(self, view: BoardView,
                           option: Option) -> float | None:
        """Boss's Orders is a PRIZE tool here, not a trap (contrast the
        Crustle module): pull the body we can actually knock out."""
        pokemon = view.pokemon_at(option.playerIndex, option.area,
                                  option.index)
        card = view.card_of(pokemon)
        if pokemon is None or card is None:
            return None
        theta = view.theta
        score = theta["drag_base"]
        try:
            hp = int(pokemon.hp) if pokemon.hp is not None else None
        except (TypeError, ValueError):
            hp = None
        if hp is not None:
            score -= theta["drag_hp_weight"] * hp
            best = self._best_damage_out(view)
            if best is not None and best >= hp:
                score += theta["drag_lethal_bonus"]
        if card.is_ex or card.is_mega_ex:
            score += theta["drag_ex_bonus"]
        return score

    def _best_damage_out(self, view: BoardView) -> float | None:
        """Printed damage our active could put out (None when unknown)."""
        active = view.my_active()
        if active is None or active.id is None:
            return None
        attacks = view.agent._index.attacks_of(active.id)
        if not attacks:
            return None
        return max(float(a.damage_base or 0) for a in attacks)

    # ------------------------------------------------------------------ #
    # Attach: fuel the line, switch Munkidori on
    # ------------------------------------------------------------------ #

    def attach_score(self, view: BoardView, option: Option,
                     base: float) -> float | None:
        theta = view.theta
        target = view.pokemon_at(view.me, option.inPlayArea, option.inPlayIndex)
        if target is None or target.id is None:
            return None
        if target.id == MUNKIDORI and self._dark_count(target) == 0:
            # one {D} is the whole activation cost of Adrena-Brain
            return base + theta["attach_munkidori_seed"]
        if target.id in MARNIE_LINE:
            return base + theta["attach_marnie_bonus"]
        return None

    # ------------------------------------------------------------------ #
    # Select handlers: counters, searches, discards
    # ------------------------------------------------------------------ #

    def select_score(self, view: BoardView,
                     context: SelectContext) -> Optional[SelectPlan]:
        select = view.obs.select
        if select is None:
            return None
        if context in (SelectContext.DAMAGE_COUNTER,
                       SelectContext.DAMAGE_COUNTER_ANY):
            return (lambda o: self._counter_destination(view, o),
                    select.minCount, max(select.minCount, 1))
        if context == SelectContext.REMOVE_DAMAGE_COUNTER:
            return (lambda o: self._counter_source(view, o),
                    select.minCount, max(select.minCount, 1))
        if context == SelectContext.TO_HAND:
            return (lambda o: self._search_value(view, o),
                    select.minCount, select.maxCount)
        if context in (SelectContext.DISCARD,
                       SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
            return (lambda o: 100.0 - self._keep_value(view, o),
                    select.minCount, select.minCount)
        return None

    def _counter_destination(self, view: BoardView, option: Option) -> float:
        """Where Adrena-Brain's counters go: convert them into a prize."""
        theta = view.theta
        pokemon = view.pokemon_at(option.playerIndex, option.area,
                                  option.index)
        card = view.card_of(pokemon)
        score = theta["counter_base"]
        if pokemon is None:
            return score
        # Tera on the BENCH nullifies damage entirely (verified in
        # tests/test_tera_bench_immunity.py): never spend counters there.
        if (card is not None and card.is_tera
                and option.area is not None and option.area.name == "BENCH"):
            return score - theta["counter_tera_bench_malus"]
        score += theta["counter_damage_weight"] * view.damage_on(pokemon)
        try:
            if pokemon.hp is not None and int(pokemon.hp) <= ADRENA_BRAIN_DAMAGE:
                score += theta["counter_lethal_bonus"]
        except (TypeError, ValueError):
            pass
        if card is not None and (card.is_ex or card.is_mega_ex):
            score += theta["counter_ex_bonus"]
        return score

    def _counter_source(self, view: BoardView, option: Option) -> float:
        """Which of ours the counters come off: the biggest prize we own."""
        theta = view.theta
        pokemon = view.pokemon_at(option.playerIndex, option.area,
                                  option.index)
        card = view.card_of(pokemon)
        score = theta["source_base"]
        if pokemon is None:
            return score
        score += theta["source_damage_weight"] * view.damage_on(pokemon)
        if card is not None and (card.is_ex or card.is_mega_ex):
            score += theta["source_ex_bonus"]
        return score

    def _search_value(self, view: BoardView, option: Option) -> float:
        """Petrel / Poké Pad / Night Stretcher / Pokégear picks."""
        theta = view.theta
        card = view.option_card(option)
        if card is None:
            return 0.0
        cid = card.card_id
        if cid == GRIMMSNARL_EX:
            return (theta["search_grimmsnarl_live"]
                    if self._grimmsnarl_target_waiting(view)
                    else theta["search_grimmsnarl_idle"])
        if cid == RARE_CANDY:
            return theta["search_rare_candy"]
        if cid in (IMPIDIMP, MORGREM):
            return (theta["search_impidimp_have"]
                    if self._has_on_field(view, IMPIDIMP)
                    else theta["search_impidimp_missing"])
        if cid == MUNKIDORI:
            return (theta["search_munkidori_have"]
                    if self._has_on_field(view, MUNKIDORI)
                    else theta["search_munkidori_missing"])
        if cid == BOSS_ORDERS:
            return (theta["search_boss_live"] if self._drag_worth_it(view)
                    else theta["search_boss_idle"])
        if cid in (SNORUNT, FROSLASS):
            return theta["search_froslass_line"]
        if cid in BOARD_BUILDERS:
            return theta["search_builder"]
        if card.stage_code in (1, 2):  # energy
            return (theta["search_energy_short"] if self._energy_short(view)
                    else theta["search_energy_idle"])
        return theta["search_other"]

    def _keep_value(self, view: BoardView, option: Option) -> float:
        """How much a card is worth KEEPING (discards throw the lowest)."""
        theta = view.theta
        card = view.option_card(option)
        if card is None:
            return theta["keep_unknown"]
        cid = card.card_id
        if cid == GRIMMSNARL_EX:
            return theta["keep_grimmsnarl"]
        if cid == RARE_CANDY:
            return theta["keep_rare_candy"]
        if cid in (IMPIDIMP, MORGREM):
            return theta["keep_marnie_line"]
        if cid == MUNKIDORI:
            return theta["keep_munkidori"]
        if card.stage_code in (1, 2):
            return theta["keep_energy"]
        if cid == BOSS_ORDERS:
            return theta["keep_boss"]
        return theta["keep_other"]


__all__ = ["GrimmsnarlModule", "SCHEMA"]
