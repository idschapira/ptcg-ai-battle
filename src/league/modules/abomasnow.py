"""AbomasnowModule: the raw racer — and the portfolio's BYPASS candidate.

The deck (data/decks/placeholder_abomasnow.csv) is deliberately crude:
35 Basic {W} Energy, Snover -> Mega Abomasnow ex (350 HP), 2 Kyogre,
and draw/search/accel supporters. What makes it strategically
interesting is what it does NOT have: **zero Pokémon with an Ability,
and no rule text that depends on a persistent effect.**

That is the bypass hypothesis. Both of our other candidates win through
EFFECTS — Crustle's Mysterious Rock Inn is a damage-prevention effect,
Grimmsnarl leans on Froslass's Freezing Shroud and Munkidori's
Adrena-Brain. Mega Starmie ex's Nebula Beam (210, explicitly "isn't
affected by Weakness or Resistance, or by any effects on your
opponent's Active Pokémon") walks straight through that class of plan,
which is why Starmie is the shared worst cell of both. Abomasnow has no
effect to strip, so an effect-ignoring attacker gains nothing special
against it. Whether that actually converts is measured, not assumed:
tests/test_league_abomasnow.py asserts the module's rules, and the
bypass VERDICT comes from the fitness harness.

The two rules that carry the deck:

  HAMMER-LANCHE (cost {W}{W}): discard the top 6 of your deck, 100
  damage per Basic {W} discarded. With 35 {W} in 60 cards that is ~350
  expected — but it is also a 6-card self-mill, so it races our own
  deck-out. The module estimates the payoff from OBSERVABLE state
  (total {W} minus the ones already in discard, in hand, and attached,
  over deckCount) rather than a hardcoded average, and refuses to fire
  it below `hammer_min_deck`.

  RIPTIDE (Kyogre, cost {W}): 20 per Basic {W} in the discard, THEN
  shuffles them back. It is both a damage payoff for the Hammer-lanche
  mill and the deck's only anti-deck-out recovery, so its value rises
  as our deck empties — the one rule where low deck makes an attack
  MORE attractive, not less.

Engine facts verified by probe (the engine exposes no Trainer text):
  Mega Signal, Cyrano -> TO_HAND (search).  Waitress -> ATTACH_TO
  (energy acceleration).  Lillie's Determination -> MAIN (draw).
  Mega Abomasnow ex is a normal Stage-1 evolution from Snover.

Note the generic agent answers ATTACH_TO by INDEX ORDER — with four
Waitress in the list that is a real amount of wasted acceleration, so
the module takes that context over.
"""

from __future__ import annotations

from typing import Final, Optional

from cg.api import Option, OptionType, Pokemon, SelectContext

from ..board import BoardView
from ..module import DeckModule, SelectPlan
from ..theta import ParamSpec, ThetaSchema

# ---- card ids (facts about the deck, not knobs) ---- #
KYOGRE: Final[int] = 721
SNOVER: Final[int] = 722
MEGA_ABOMASNOW: Final[int] = 723
WATER_ENERGY: Final[int] = 3        # Basic {W} Energy card id AND type code
MAXIMUM_BELT: Final[int] = 1158
MEGA_SIGNAL: Final[int] = 1145
CYRANO: Final[int] = 1205
LILLIE: Final[int] = 1227
WAITRESS: Final[int] = 1235

ATTACKERS: Final[frozenset[int]] = frozenset({MEGA_ABOMASNOW, KYOGRE})

#: Card facts used by the damage estimators (printed numbers).
HAMMER_MILL_DEPTH: Final[int] = 6
HAMMER_DAMAGE_PER_W: Final[float] = 100.0
RIPTIDE_DAMAGE_PER_W: Final[float] = 20.0
FROST_BARRIER_DAMAGE: Final[float] = 200.0

_TRAINER_BAND: Final[float] = 35.0
_ATTACH_BAND: Final[float] = 55.0

SCHEMA: Final[ThetaSchema] = ThetaSchema((
    # ---- the climb ---- #
    ParamSpec("evolve_mega_bonus", 18.0, 0.0, 40.0,
              doc="Snover -> Mega Abomasnow ex is the whole deck"),
    ParamSpec("desired_field_floor", 3, 1, 6, integral=True),
    ParamSpec("rebuild_score", 44.0, _TRAINER_BAND, _ATTACH_BAND - 1.0),
    # ---- Hammer-lanche: payoff vs self-mill ---- #
    ParamSpec("deck_w_total", 35, 0, 60, integral=True,
              doc="Basic {W} in the list; the density estimator's anchor"),
    ParamSpec("hammer_expected_w_fallback", 3.5, 0.0, 6.0,
              doc="used only when deckCount is unreadable"),
    ParamSpec("hammer_min_deck", 12, 0, 40, integral=True,
              doc="never mill 6 more when the deck is this thin"),
    ParamSpec("hammer_confidence", 1.0, 0.4, 1.6,
              doc="how much to trust the density estimate (variance is high)"),
    # ---- Riptide: the recycler ---- #
    ParamSpec("riptide_deck_floor", 15, 0, 40, integral=True,
              doc="deck at or under this: shuffling {W} back is the point"),
    ParamSpec("riptide_recycle_bonus", 60.0, 0.0, 200.0,
              doc="damage-equivalent value of undoing our own mill"),
    # ---- attack banding (mirrors the generic shape; stays <= 35) ---- #
    # Bounds chosen so the WORST case reachable by mutation —
    # floor.high + 450/scale.low + ko.high — is exactly 35.0, the
    # generic agent's own attack ceiling. Development therefore keeps
    # outranking attacking for every genome (the Sprint-3 lesson).
    ParamSpec("attack_band_floor", 20.0, 15.0, 20.0),
    ParamSpec("attack_scale", 45.0, 45.0, 90.0),
    ParamSpec("attack_ko_bonus", 5.0, 0.0, 5.0),
    # ---- energy routing ---- #
    ParamSpec("attach_mega_bonus", 12.0, 0.0, 30.0),
    ParamSpec("attach_kyogre_bonus", 4.0, 0.0, 30.0),
    ParamSpec("attach_ready_malus", 6.0, 0.0, 30.0,
              doc="stop over-loading a body that can already attack"),
    ParamSpec("belt_score", 42.0, 0.0, _ATTACH_BAND - 1.0,
              doc="Maximum Belt on the attacker"),
    # ---- supporters ---- #
    ParamSpec("draw_supporter_score", 38.0, 0.0, _ATTACH_BAND - 1.0),
    ParamSpec("waitress_score", 46.0, 0.0, _ATTACH_BAND - 1.0,
              doc="acceleration beats plain draw in a racer"),
    # ---- benching ---- #
    ParamSpec("bench_snover", 90.0, 0.0, 120.0),
    ParamSpec("bench_kyogre", 60.0, 0.0, 120.0),
    # ---- search (TO_HAND) ---- #
    ParamSpec("search_mega_live", 95.0, 0.0, 100.0,
              doc="a Snover is waiting: this card is the turn"),
    ParamSpec("search_mega_idle", 50.0, 0.0, 100.0),
    ParamSpec("search_snover_missing", 85.0, 0.0, 100.0),
    ParamSpec("search_snover_have", 25.0, 0.0, 100.0),
    ParamSpec("search_kyogre_late", 70.0, 0.0, 100.0,
              doc="deck thinning out: Kyogre is the recycler"),
    ParamSpec("search_kyogre_early", 22.0, 0.0, 100.0),
    ParamSpec("search_energy", 45.0, 0.0, 100.0),
    ParamSpec("search_belt", 30.0, 0.0, 100.0),
    ParamSpec("search_other", 12.0, 0.0, 100.0),
    # ---- discard ---- #
    ParamSpec("keep_mega", 95.0, 0.0, 100.0),
    ParamSpec("keep_snover", 75.0, 0.0, 100.0),
    ParamSpec("keep_kyogre", 65.0, 0.0, 100.0),
    ParamSpec("keep_energy", 50.0, 0.0, 100.0),
    ParamSpec("keep_other", 20.0, 0.0, 100.0),
    ParamSpec("keep_unknown", 20.0, 0.0, 100.0),
))


class AbomasnowModule(DeckModule):
    """Mega Abomasnow ex racer: no abilities, no effects, just tempo."""

    name = "abomasnow"
    decks = ("abomasnow",)
    schema = SCHEMA

    __slots__ = ("_hammer_ids", "_frost_ids", "_riptide_ids", "_swirl_ids")

    def __init__(self) -> None:
        self._hammer_ids: frozenset[int] = frozenset()
        self._frost_ids: frozenset[int] = frozenset()
        self._riptide_ids: frozenset[int] = frozenset()
        self._swirl_ids: frozenset[int] = frozenset()

    def bind(self, agent) -> None:
        """Resolve attacks by EFFECT TEXT, not by hardcoded ids — the
        same reprint under a new id keeps working."""
        def pick(card_id: int, needle: str) -> frozenset[int]:
            return frozenset(
                a.attack_id for a in agent._index.attacks_of(card_id)
                if a.effect and needle in a.effect.lower())

        mega = agent._index.attacks_of(MEGA_ABOMASNOW)
        self._hammer_ids = pick(MEGA_ABOMASNOW, "top 6")
        self._frost_ids = frozenset(
            a.attack_id for a in mega if a.attack_id not in self._hammer_ids)
        self._riptide_ids = pick(KYOGRE, "shuffle those cards")
        kyogre = agent._index.attacks_of(KYOGRE)
        self._swirl_ids = frozenset(
            a.attack_id for a in kyogre if a.attack_id not in self._riptide_ids)

    # ------------------------------------------------------------------ #
    # Estimators (observable state, not hardcoded averages)
    # ------------------------------------------------------------------ #

    def _water_seen(self, view: BoardView) -> int:
        """Basic {W} we can already account for: discard + hand + attached."""
        seen = sum(1 for cid in view.my_discard_ids() if cid == WATER_ENERGY)
        seen += sum(1 for cid in view.my_hand_ids() if cid == WATER_ENERGY)
        seen += view.count_attached(WATER_ENERGY)
        return seen

    def _expected_hammer_water(self, view: BoardView) -> float:
        """Expected Basic {W} among the top `HAMMER_MILL_DEPTH` cards."""
        theta = view.theta
        deck = view.deck_counts()[0]
        if deck is None or deck <= 0:
            return theta["hammer_expected_w_fallback"]
        remaining = max(0.0, theta["deck_w_total"] - self._water_seen(view))
        density = min(1.0, remaining / deck)
        return min(float(HAMMER_MILL_DEPTH), float(deck)) * density

    def _water_in_discard(self, view: BoardView) -> int:
        return sum(1 for cid in view.my_discard_ids() if cid == WATER_ENERGY)

    def _attack_estimate(self, view: BoardView,
                         attack_id: int | None) -> float | None:
        """Damage-equivalent value of one of OUR attacks; None = no rule."""
        if attack_id is None:
            return None
        theta = view.theta
        deck = view.deck_counts()[0]

        if attack_id in self._hammer_ids:
            if deck is not None and deck <= theta.i("hammer_min_deck"):
                # milling 6 more races our own deck-out: take the flat 200
                return 0.0
            expected = self._expected_hammer_water(view)
            return HAMMER_DAMAGE_PER_W * expected * theta["hammer_confidence"]

        if attack_id in self._frost_ids:
            return FROST_BARRIER_DAMAGE

        if attack_id in self._riptide_ids:
            value = RIPTIDE_DAMAGE_PER_W * self._water_in_discard(view)
            if deck is not None and deck <= theta.i("riptide_deck_floor"):
                # the ONE case where a thin deck makes an attack better:
                # Riptide shuffles the milled {W} back in
                value += theta["riptide_recycle_bonus"]
            return value

        return None

    def _needs_energy(self, view: BoardView, pokemon: Pokemon | None) -> bool:
        """True when this body cannot yet use its best attack."""
        if pokemon is None or pokemon.id is None:
            return True
        from ...ingestion.card_index import is_cost_payable
        energies = view.energies_on(pokemon)
        attacks = view.agent._index.attacks_of(pokemon.id)
        if not attacks:
            return False
        return not any(is_cost_payable(a.cost, energies) for a in attacks)

    def _board_thin(self, view: BoardView) -> bool:
        count = view.field_count()
        return count is not None and count < view.theta.i("desired_field_floor")

    # ------------------------------------------------------------------ #
    # MAIN overlay
    # ------------------------------------------------------------------ #

    def main_score(self, view: BoardView, option: Option,
                   base: float) -> float | None:
        kind = option.type
        if kind == OptionType.ATTACK:
            return self._attack_main_score(view, option, base)
        if kind == OptionType.EVOLVE:
            card_id = view.option_card_id(option)
            if card_id == MEGA_ABOMASNOW:
                return base + view.theta["evolve_mega_bonus"]
            return None
        if kind == OptionType.PLAY:
            return self._play_score(view, option, base)
        return None

    def _attack_main_score(self, view: BoardView, option: Option,
                           base: float) -> float | None:
        """Rank our attacks by estimated damage, inside the generic band.

        Shape mirrors HeuristicAgent's ATTACK branch on purpose, so the
        Sprint-3 invariant holds: the ceiling is still the trainer band
        and development actions keep outranking attacking."""
        estimate = self._attack_estimate(view, option.attackId)
        if estimate is None:
            return None
        theta = view.theta
        score = theta["attack_band_floor"] + min(estimate, 450.0) / theta["attack_scale"]
        opponent = view.opp_active()
        if opponent is not None and opponent.hp is not None:
            try:
                if estimate >= float(opponent.hp):
                    score += theta["attack_ko_bonus"]
            except (TypeError, ValueError):
                pass
        return score

    def _play_score(self, view: BoardView, option: Option,
                    base: float) -> float | None:
        theta = view.theta
        card_id = view.option_card_id(option)
        if card_id is None:
            return None
        if card_id == WAITRESS:
            return theta["waitress_score"]
        if card_id == LILLIE:
            return theta["draw_supporter_score"]
        if card_id == MAXIMUM_BELT:
            return theta["belt_score"]
        if card_id in (MEGA_SIGNAL, CYRANO) and self._board_thin(view):
            return max(base, theta["rebuild_score"])
        return None

    # ------------------------------------------------------------------ #
    # Benching
    # ------------------------------------------------------------------ #

    def own_pokemon_score(self, view: BoardView, option: Option,
                          for_active: bool, base: float) -> float | None:
        if for_active:
            return None
        card_id = view.option_card_id(option)
        if card_id == SNOVER:
            return view.theta["bench_snover"]
        if card_id == KYOGRE:
            return view.theta["bench_kyogre"]
        return None

    # ------------------------------------------------------------------ #
    # Energy routing (manual attach AND Waitress acceleration)
    # ------------------------------------------------------------------ #

    def attach_score(self, view: BoardView, option: Option,
                     base: float) -> float | None:
        target = view.pokemon_at(view.me, option.inPlayArea, option.inPlayIndex)
        delta = self._attach_preference(view, target)
        return None if delta is None else base + delta

    def _attach_preference(self, view: BoardView,
                           target: Pokemon | None) -> float | None:
        if target is None or target.id is None:
            return None
        theta = view.theta
        if target.id == MEGA_ABOMASNOW:
            delta = theta["attach_mega_bonus"]
        elif target.id == KYOGRE:
            delta = theta["attach_kyogre_bonus"]
        else:
            return None
        if not self._needs_energy(view, target):
            delta -= theta["attach_ready_malus"]
        return delta

    # ------------------------------------------------------------------ #
    # Select handlers
    # ------------------------------------------------------------------ #

    def select_score(self, view: BoardView,
                     context: SelectContext) -> Optional[SelectPlan]:
        select = view.obs.select
        if select is None:
            return None
        if context in (SelectContext.ATTACK, SelectContext.DISABLE_ATTACK):
            return (lambda o: self._attack_choice(view, o),
                    select.minCount, max(select.minCount, 1))
        if context == SelectContext.ATTACH_TO:
            # the generic agent answers this by INDEX ORDER; with four
            # Waitress that is a lot of misrouted acceleration
            return (lambda o: self._attach_target(view, o),
                    select.minCount, max(select.minCount, 1))
        if context == SelectContext.TO_HAND:
            return (lambda o: self._search_value(view, o),
                    select.minCount, select.maxCount)
        if context in (SelectContext.DISCARD,
                       SelectContext.DISCARD_CARD_OR_ATTACHED_CARD):
            return (lambda o: 100.0 - self._keep_value(view, o),
                    select.minCount, select.minCount)
        return None

    def _attack_choice(self, view: BoardView, option: Option) -> float:
        """Hammer-lanche vs Frost Barrier vs Riptide, on the estimates."""
        estimate = self._attack_estimate(view, option.attackId)
        if estimate is None:
            # no rule for this attack: fall back on the generic valuation
            return view.agent._attack_value(option.attackId, view.obs)
        return estimate

    def _attach_target(self, view: BoardView, option: Option) -> float:
        target = view.pokemon_at(option.playerIndex, option.area, option.index)
        if target is None:
            target = view.pokemon_at(view.me, option.inPlayArea,
                                     option.inPlayIndex)
        delta = self._attach_preference(view, target)
        return 10.0 + (delta if delta is not None else 0.0)

    def _search_value(self, view: BoardView, option: Option) -> float:
        theta = view.theta
        card = view.option_card(option)
        if card is None:
            return 0.0
        cid = card.card_id
        field = view.my_field_ids()
        deck = view.deck_counts()[0]
        if cid == MEGA_ABOMASNOW:
            return (theta["search_mega_live"] if SNOVER in field
                    else theta["search_mega_idle"])
        if cid == SNOVER:
            return (theta["search_snover_have"] if SNOVER in field
                    else theta["search_snover_missing"])
        if cid == KYOGRE:
            late = deck is not None and deck <= theta.i("riptide_deck_floor")
            return (theta["search_kyogre_late"] if late
                    else theta["search_kyogre_early"])
        if cid == WATER_ENERGY or card.stage_code in (1, 2):
            return theta["search_energy"]
        if cid == MAXIMUM_BELT:
            return theta["search_belt"]
        return theta["search_other"]

    def _keep_value(self, view: BoardView, option: Option) -> float:
        theta = view.theta
        card = view.option_card(option)
        if card is None:
            return theta["keep_unknown"]
        cid = card.card_id
        if cid == MEGA_ABOMASNOW:
            return theta["keep_mega"]
        if cid == SNOVER:
            return theta["keep_snover"]
        if cid == KYOGRE:
            return theta["keep_kyogre"]
        if cid == WATER_ENERGY or card.stage_code in (1, 2):
            return theta["keep_energy"]
        return theta["keep_other"]


__all__ = ["AbomasnowModule", "SCHEMA"]
