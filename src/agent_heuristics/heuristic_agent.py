"""HeuristicAgent v1: greedy typed policy over CardIndex + dim_effect.

Decision quality is intentionally simple (policy-net comes later); the
hard requirements are the engine contract and total crash-safety: every
path falls back to a legal answer even when data is missing.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Final

from cg.api import (
    AreaType,
    Observation,
    Option,
    OptionType,
    Pokemon,
    SelectContext,
    State,
)

from ..deckbuilding.archetype_rules import PROFILE_AGGRO, PROFILE_DEVELOPMENT
from ..environment_wrapper.wrapper import EnvironmentWrapper
from ..ingestion.build_card_model import EnergyType
from ..ingestion.build_effect_model import EffectIndex, EffectRow, EffectType
from ..ingestion.card_index import Attack, Card, CardIndex, is_cost_payable
from .random_agent import read_deck_csv

# MAIN score bands. The engine re-prompts MAIN after every non-turn-ending
# action, so development actions MUST outrank attacking: the greedy argmax
# then walks evolve -> play -> attach -> trainer/ability -> attack -> end
# within a single turn (attacking ends it).
_KO_BONUS: Final[float] = 100.0
_EVOLVE_BAND: Final[float] = 80.0
_PLAY_POKEMON_BAND: Final[float] = 70.0
_ATTACH_BAND: Final[float] = 55.0
_ABILITY_BAND: Final[float] = 40.0
_TRAINER_BAND: Final[float] = 35.0
_ATTACK_BAND: Final[float] = 20.0
_RETREAT_LOW_HP: Final[float] = 15.0
_END_SCORE: Final[float] = 0.5

# PROFILE-DEPENDENT attack band (see archetype_rules.deck_profile).
# The bands above are the Crustle lesson: the board is the win condition,
# so development outranks attacking. For a deck whose plan IS the prize
# race that reasoning does not hold, and an AGGRO profile lets a LETHAL
# attack outrank development — nothing on the board is worth more than
# the knockout it would spend.
#
# Measured ceiling, stated up front because it bounds what this can buy:
# probed over 60 games of the Alakazam list, 96.6% of the turns that ever
# offered an attack already ended in one (312/323). Attacking ends the
# turn, so the band only REORDERS actions inside a turn; it cannot add
# attacks to turns where no attack was ever legal, and 40% of their turns
# are exactly that (210 of 215 with a zero-energy active). The band is a
# tie-break, not the constraint — `energy_routing` below is.
_AGGRO_LETHAL_BAND: Final[float] = 90.0   # above EVOLVE (80 + hp/40)

_STATUS_VALUE: Final[dict[int, float]] = {0: 20.0, 1: 20.0, 2: 30.0, 3: 30.0, 4: 25.0}

_ENERGY_SYMBOL: Final[dict[str, int]] = {
    "C": 0, "G": 1, "R": 2, "W": 3, "L": 4, "P": 5,
    "F": 6, "D": 7, "M": 8, "N": 9,
}


@dataclass(frozen=True)
class ScaledClause:
    """An attack's damage-per-unit and which board quantity is the unit."""

    per_unit: float          # damage (already converted from counters)
    unit: str                # key understood by HeuristicAgent._unit_count
    energy_code: int | None  # for the typed-energy units
    # Units that are not on the board but ARE known in expectation —
    # coin flips. "Flip 4 coins ... for each heads" is 2 heads in
    # expectation, and "flip until tails" is 1. Carrying the number here
    # keeps the estimate explicit instead of falling into the generic
    # unresolved fallback, which would value a 2-coin attack at double.
    fixed_units: float | None = None


_ENGINE_ATTACK_TEXT: dict[int, str] | None = None


def _engine_attack_text(attack_id: int) -> str | None:
    """Attack rules text, straight from the engine (our star schema drops
    it: dim_attack keeps the distilled effect rows, not the prose)."""
    global _ENGINE_ATTACK_TEXT
    if _ENGINE_ATTACK_TEXT is None:
        try:
            import cg.api as _api
            _ENGINE_ATTACK_TEXT = {
                a.attackId: (a.text or "") for a in _api.all_attack()
            }
        except Exception:
            _ENGINE_ATTACK_TEXT = {}
    return _ENGINE_ATTACK_TEXT.get(attack_id)


def parse_scaled_clause(text: str | None) -> ScaledClause | None:
    """Damage-scaling clause of an attack, or None if it has none.

    Pure text -> structure; no engine or observation involved, so it is
    cached per attack id by the caller and unit-testable on its own.
    """
    if not text:
        return None
    flat = text.replace("\n", " ")
    for kind, pattern in _SCALE_SHAPES:
        match = pattern.search(flat)
        if match is None:
            continue
        try:
            magnitude = float(match.group(1))
        except (TypeError, ValueError):
            return None
        per_unit = (magnitude * _DAMAGE_PER_COUNTER if kind == "counters"
                    else magnitude)
        phrase = match.group(2)
        if _HEADS_RE.search(phrase):
            return ScaledClause(per_unit, "coin", None,
                                fixed_units=_expected_heads(flat))
        for unit_pattern, key in _UNIT_PATTERNS:
            unit_match = unit_pattern.search(phrase)
            if unit_match is None:
                continue
            code = None
            if key == "my_active_energy_typed":
                code = _ENERGY_SYMBOL.get(unit_match.group(1).upper())
                if code is None:
                    return None
            return ScaledClause(per_unit, key, code)
        return ScaledClause(per_unit, "unresolved", None)
    return None


_HEADS_RE: Final["re.Pattern[str]"] = re.compile(r"\bheads\b", re.I)
_FLIP_N_RE: Final["re.Pattern[str]"] = re.compile(
    r"flip\s+(\d+)\s+coins", re.I)
_FLIP_UNTIL_RE: Final["re.Pattern[str]"] = re.compile(
    r"flip a coin until you get tails", re.I)


def _expected_heads(text: str) -> float:
    """Heads in expectation: N/2 for N coins, 1 for flip-until-tails."""
    match = _FLIP_N_RE.search(text)
    if match:
        try:
            return float(match.group(1)) / 2.0
        except (TypeError, ValueError):
            return 1.0
    if _FLIP_UNTIL_RE.search(text):
        return 1.0     # sum of a geometric series with p=1/2
    return 1.0

# Cards that ARE an evolution, but arrive as a Trainer PLAY option and so
# land in _TRAINER_BAND (35) where they lose to every real EVOLVE (80+)
# and tie with every other Trainer. Measured consequence on the ladder
# corpus (31/Jul): the generic pilot plays Rare Candy 0.63x/game against
# 1.10x for real opponents and brings its Stage 2 online ~1.3 turns late,
# which is most of the +46/+48pp inflation in every close-race cell.
#
# Promoting them is SAFE because the engine itself gates the option:
# probed over 40 games of the Alakazam list, Rare Candy was offered as a
# PLAY in 0 of 1504 decisions with no Stage 2 in hand, and in 289 of the
# 2093 decisions where one was held (it also needs a matching Basic in
# play). So every legal Rare Candy PLAY is a genuine two-stage jump, and
# the scorer does not need to re-derive the condition.
EVOLUTION_ACCELERATORS: Final[frozenset[int]] = frozenset({
    1079,   # Rare Candy — Basic -> Stage 2, skipping Stage 1
})
# above a normal EVOLVE (80 + hp/40, so ~83.5 for a 140 HP Stage 2):
# skipping a whole stage is strictly more tempo than taking one step.
_ACCELERATOR_BONUS: Final[float] = 5.0

# --------------------------------------------------------------------------
# Attacks whose damage SCALES with a board quantity
# --------------------------------------------------------------------------
# _effect_adjustment values an attack as damage_base + a fixed bonus per
# effect row, which is blind to any attack whose damage is the scale.
# Measured consequence (31/Jul): Alakazam's Powerful Hand — "Place 2
# damage counters on your opponent's Active Pokémon FOR EACH CARD IN YOUR
# HAND", base 0 — scored 13.0 while delivering ~266, so the generic pilot
# preferred Kadabra's 30-damage Super Psy Bolt. 163 attacks in the pool
# carry a scaling clause, so this is not one card's problem.
#
# The clause is parsed from the attack TEXT (the engine exposes it) into
# (damage per unit, unit key) once per attack id, then the unit is counted
# on the LIVE observation. Where the unit is not observable — "for each
# heads", "for each card you discarded in this way" — no resolver exists
# and the old fixed-bonus path stays, so nothing silently invents a
# number. `unresolved_scaled_units()` reports which those are.
_DAMAGE_PER_COUNTER: Final[float] = 10.0

_SCALE_SHAPES: Final[tuple[tuple[str, "re.Pattern[str]"], ...]] = (
    # "Place N damage counters on your opponent's ... for each X"
    ("counters",
     re.compile(r"place\s+(\d+)\s+damage\s+counters?\s+on\s+your\s+"
                r"opponent[^.]{0,40}?for each ([^.,]{3,70})", re.I)),
    # "This attack does N more damage for each X"
    ("damage",
     re.compile(r"does\s+(\d+)\s+more\s+damage\s+for each ([^.,]{3,70})",
                re.I)),
    # "This attack does N damage for each X" (base is normally 0)
    ("damage",
     re.compile(r"does\s+(\d+)\s+damage\s+for each ([^.,]{3,70})", re.I)),
)

# unit phrase -> resolver key, ordered SPECIFIC -> GENERIC (first match
# wins), exactly like archetype_rules: "{W} energy attached to this
# Pokémon" must not be eaten by the generic "energy attached to this".
_UNIT_PATTERNS: Final[tuple[tuple["re.Pattern[str]", str], ...]] = (
    (re.compile(r"card in your hand", re.I), "my_hand"),
    (re.compile(r"card in your opponent.s hand", re.I), "opp_hand"),
    (re.compile(r"damage counter on this pok", re.I), "my_active_damage"),
    (re.compile(r"damage counter on your opponent.s active", re.I),
     "opp_active_damage"),
    (re.compile(r"\{(\w)\} energy attached to this pok", re.I),
     "my_active_energy_typed"),
    (re.compile(r"energy attached to this pok", re.I), "my_active_energy"),
    (re.compile(r"energy attached to your opponent.s active", re.I),
     "opp_active_energy"),
    (re.compile(r"energy attached to all of your opponent.s pok", re.I),
     "opp_board_energy"),
    (re.compile(r"energy attached to all of your pok", re.I),
     "my_board_energy"),
    (re.compile(r"benched pok.mon \(both", re.I), "both_bench"),
    (re.compile(r"of your benched pok", re.I), "my_bench"),
    (re.compile(r"of your opponent.s benched pok", re.I), "opp_bench"),
    (re.compile(r"prize card your opponent has taken", re.I),
     "opp_prizes_taken"),
    (re.compile(r"prize card you have taken", re.I), "my_prizes_taken"),
    (re.compile(r"pok.mon tool attached to all of your pok", re.I),
     "my_board_tools"),
    (re.compile(r"pok.mon tool attached to all pok", re.I), "both_tools"),
    (re.compile(r"of your basic pok.mon in play", re.I), "my_basics"),
    (re.compile(r"of your pok.mon in play", re.I), "my_board"),
)

# units we cannot see on the board; the declared fallback below is used
# instead of pretending to know. Kept explicit so the estimate is a
# stated assumption rather than a magic constant buried in a branch.
_UNRESOLVED_UNITS_ESTIMATE: Final[float] = 2.0

# --------------------------------------------------------------------------
# ENERGY ROUTING — the same blindness, in the paths that DECIDE THE BOARD
# --------------------------------------------------------------------------
# `scaled_damage` taught _attack_value to count the scaling unit live.
# Two more scorers still read `damage_base` alone and so still value a
# scaling attacker at zero:
#
#   _best_affordable_damage  -> _attach_score: "how much damage does this
#                               body unlock if I put the energy HERE"
#   _own_pokemon_score       -> promotion/switch/bench: "which body do I
#                               want in front"
#
# Powerful Hand's printed base is None, so under both scorers Alakazam is
# a 0-damage body: energy goes to whatever has printed damage and the
# promotion prompt prefers the fattest, not the attacker. Measured over
# 40 games of the list (31/Jul): 46 of 155 attachments went somewhere
# else while an Alakazam sat on the board with zero energy, and only
# 29 of 45 promotions that COULD have promoted Alakazam did. Downstream,
# 40% of their turns never offered an attack at all, 210 of 215 of them
# with an empty active — which is the real reason attacks/game reads 4.86
# against 6.81 for real opponents.
#
# Routing reuses the SAME clause parser and the SAME live unit count, so
# the three scorers finally agree about what an attack is worth.


class HeuristicAgent:
    """Greedy one-ply evaluator satisfying the competition contract.

    After each __call__, `last_scores` holds the per-option scores aligned
    with obs.select.option (None when the handler is not score-based) —
    consumed by the dev game recorder.
    """

    __slots__ = ("_index", "_effects", "_wrapper", "_deck_path", "_rng",
                 "_tempo", "_scaled_damage", "_profile", "_energy_routing",
                 "_scale_cache", "last_scores")

    def __init__(
        self,
        seed: int | None = None,
        deck_path: str | None = None,
        index: CardIndex | None = None,
        effects: EffectIndex | None = None,
        tempo: bool = False,
        scaled_damage: bool = False,
        profile: str = PROFILE_DEVELOPMENT,
        energy_routing: bool = False,
    ) -> None:
        """``tempo=True`` promotes EVOLUTION_ACCELERATORS out of the
        trainer band (see the constant). ``scaled_damage=True`` values
        attacks whose damage scales with a board quantity by counting the
        unit live instead of using a fixed per-row bonus.
        ``profile=PROFILE_AGGRO`` lets a LETHAL attack outrank development
        (see _AGGRO_LETHAL_BAND); ``energy_routing=True`` extends the
        scaled valuation to the attach and promotion scorers, so energy
        and the active slot go to the body that actually threatens.

        All four default to the SHIPPED behaviour so every existing
        caller — the ship's CrustleAgent above all — stays byte-identical;
        tests/test_tempo_equivalence.py, tests/test_scaled_damage.py and
        tests/test_attack_profile.py hold that line decision-by-decision.
        """
        self._index = index if index is not None else CardIndex()
        self._effects = effects if effects is not None else EffectIndex()
        self._wrapper = EnvironmentWrapper(self._index)
        self._deck_path = deck_path
        self._rng = random.Random(seed)
        self._tempo = tempo
        self._scaled_damage = scaled_damage
        self._profile = profile
        self._energy_routing = energy_routing
        self._scale_cache: dict[int, ScaledClause | None] = {}
        self.last_scores: list[float] | None = None

    # ------------------------------------------------------------------ #
    # Contract entry point
    # ------------------------------------------------------------------ #

    def __call__(self, obs_dict: dict) -> list[int]:
        self.last_scores = None
        try:
            obs = self._wrapper.parse(obs_dict)
            if obs.select is None:
                return read_deck_csv(self._deck_path)
            return self._decide(obs)
        except Exception:
            self.last_scores = None
            return self._safe_answer(obs_dict)

    @staticmethod
    def _safe_answer(obs_dict: dict) -> list[int]:
        """Legal fallback derived from the raw dict, immune to parse bugs."""
        try:
            select = obs_dict.get("select") or {}
            min_count = int(select.get("minCount", 1))
            max_count = int(select.get("maxCount", 1))
            n_options = len(select.get("option") or [])
            count = max(min_count, min(1, max_count))
            return list(range(min(count, n_options)))
        except Exception:
            return [0]

    # ------------------------------------------------------------------ #
    # Board reads (all None-safe)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _pokemon_at(
        state: State | None, player_index: int | None, area: AreaType | None, index: int | None
    ) -> Pokemon | None:
        if state is None or player_index is None or area is None or index is None:
            return None
        try:
            player = state.players[player_index]
            if area == AreaType.ACTIVE:
                return player.active[index]
            if area == AreaType.BENCH:
                return player.bench[index]
        except (IndexError, TypeError):
            return None
        return None

    def _my_active(self, obs: Observation) -> Pokemon | None:
        return self._pokemon_at(obs.current, obs.current.yourIndex, AreaType.ACTIVE, 0)

    def _opp_active(self, obs: Observation) -> Pokemon | None:
        return self._pokemon_at(obs.current, 1 - obs.current.yourIndex, AreaType.ACTIVE, 0)

    def _card_of(self, pokemon: Pokemon | None) -> Card | None:
        return self._index.get_card(pokemon.id) if pokemon is not None else None

    # ------------------------------------------------------------------ #
    # Energy / attack math
    # ------------------------------------------------------------------ #

    @staticmethod
    def _affordable(attack: Attack, energies: list[int]) -> bool:
        return is_cost_payable(attack.cost, energies)

    def _effect_adjustment(self, attack: Attack, base: float,
                           skip_scaling: bool = False) -> float:
        """Expected-value tweak from dim_effect rows; scale gates multiply.

        ``skip_scaling`` drops the rows that stand in for damage-per-unit
        (DAMAGE_BONUS/DAMAGE_SCALE/COUNTERS with a per-unit condition)
        because the caller already counted the unit on the live board.
        Without it the two paths would both charge for the same clause.
        """
        bonus = 0.0
        multiplier = 1.0
        scaling_types = (int(EffectType.DAMAGE_BONUS),
                         int(EffectType.DAMAGE_SCALE),
                         int(EffectType.COUNTERS))
        for row in self._effects.effects_of(attack.attack_id):
            if (skip_scaling and row.effect_type in scaling_types
                    and row.condition in ("per_unit", "per_unit_minus",
                                          "each", None)):
                continue
            discount = 0.5 if row.coin_flip else 0.8
            effect = row.effect_type
            if effect == int(EffectType.DAMAGE_BONUS) and row.magnitude:
                per_unit = row.condition in ("per_unit", "per_unit_minus")
                sign = -1.0 if row.condition == "per_unit_minus" else 1.0
                units = 2.0 if per_unit else 1.0
                bonus += sign * row.magnitude * units * discount
            elif effect == int(EffectType.DAMAGE_SCALE) and row.magnitude:
                # printed base is one unit; expect ~2 units in play
                bonus += row.magnitude * (1.0 if row.coin_flip else 1.5)
            elif effect == int(EffectType.STATUS):
                bonus += _STATUS_VALUE.get(row.magnitude or -1, 20.0) * discount
            elif effect == int(EffectType.BENCH_DAMAGE) and row.magnitude:
                bonus += row.magnitude * (2.0 if row.condition == "each" else 1.0) * 0.5
            elif effect == int(EffectType.SNIPE) and row.magnitude:
                bonus += row.magnitude * 0.5
            elif effect == int(EffectType.COUNTERS) and row.magnitude:
                bonus += row.magnitude * 10 * 0.5
            elif effect == int(EffectType.SELF_DAMAGE) and row.magnitude:
                bonus -= row.magnitude * 0.7
            elif effect == int(EffectType.ENERGY_DISCARD_SELF):
                bonus -= 15.0
            elif effect in (int(EffectType.HEAL), int(EffectType.DRAW),
                            int(EffectType.SEARCH), int(EffectType.ENERGY_ACCEL)):
                bonus += 10.0
            elif effect in (int(EffectType.OPP_LOCK), int(EffectType.GUST)):
                bonus += 15.0
            elif effect == int(EffectType.SELF_LOCK):
                bonus -= 10.0
            elif effect == int(EffectType.FAIL_UNLESS_HEADS):
                multiplier *= 0.5
            elif effect == int(EffectType.FAIL_CONDITION):
                multiplier *= 0.7
        return (base + bonus) * multiplier

    def _scaled_clause(self, attack_id: int) -> ScaledClause | None:
        """Cached text parse of the attack's damage-scaling clause."""
        if attack_id not in self._scale_cache:
            self._scale_cache[attack_id] = parse_scaled_clause(
                _engine_attack_text(attack_id))
        return self._scale_cache[attack_id]

    def _unit_count(self, unit: str, energy_code: int | None,
                    obs: Observation) -> float | None:
        """Live count of a scaling unit, or None when unobservable."""
        state = obs.current
        if state is None:
            return None
        try:
            me = state.players[state.yourIndex]
            them = state.players[1 - state.yourIndex]
        except (IndexError, TypeError):
            return None
        mine = self._my_active(obs)
        theirs = self._opp_active(obs)

        def board(player) -> list:
            active = [p for p in (player.active or []) if p]
            return active + [p for p in (player.bench or []) if p]

        def energies(pokemon, code: int | None = None) -> int:
            if pokemon is None:
                return 0
            values = [int(e) for e in (pokemon.energies or [])]
            return len(values) if code is None else values.count(code)

        def damage_counters(pokemon) -> int:
            if pokemon is None or not pokemon.maxHp:
                return 0
            return max(0, int(pokemon.maxHp) - int(pokemon.hp)) // 10

        if unit == "my_hand":
            return float(me.handCount or len(me.hand or []))
        if unit == "opp_hand":
            return float(them.handCount or len(them.hand or []))
        if unit == "my_active_damage":
            return float(damage_counters(mine))
        if unit == "opp_active_damage":
            return float(damage_counters(theirs))
        if unit == "my_active_energy_typed":
            return float(energies(mine, energy_code))
        if unit == "my_active_energy":
            return float(energies(mine))
        if unit == "opp_active_energy":
            return float(energies(theirs))
        if unit == "my_board_energy":
            return float(sum(energies(p) for p in board(me)))
        if unit == "opp_board_energy":
            return float(sum(energies(p) for p in board(them)))
        if unit == "my_bench":
            return float(len([p for p in (me.bench or []) if p]))
        if unit == "opp_bench":
            return float(len([p for p in (them.bench or []) if p]))
        if unit == "both_bench":
            return float(len([p for p in (me.bench or []) if p])
                         + len([p for p in (them.bench or []) if p]))
        if unit == "opp_prizes_taken":
            return float(max(0, 6 - len(them.prize or [])))
        if unit == "my_prizes_taken":
            return float(max(0, 6 - len(me.prize or [])))
        if unit == "my_board_tools":
            return float(sum(len(p.tools or []) for p in board(me)))
        if unit == "both_tools":
            return float(sum(len(p.tools or []) for p in board(me))
                         + sum(len(p.tools or []) for p in board(them)))
        if unit == "my_board":
            return float(len(board(me)))
        if unit == "my_basics":
            count = 0
            for pokemon in board(me):
                card = self._index.get_card(pokemon.id)
                if card is not None and card.stage_code == 7:
                    count += 1
            return float(count)
        return None

    def _scaled_units(self, clause: ScaledClause, obs: Observation) -> float:
        """How many units the clause scales over, on the live board.

        Coin clauses carry their own expectation; an unobservable unit
        falls to the DECLARED estimate rather than inventing a number.
        """
        if clause.fixed_units is not None:
            return clause.fixed_units
        units = self._unit_count(clause.unit, clause.energy_code, obs)
        return _UNRESOLVED_UNITS_ESTIMATE if units is None else units

    def _body_damage(self, attack: Attack, obs: Observation | None) -> float:
        """Damage an attack represents when judging a BODY (not a move).

        Printed base only, exactly as before — unless energy routing is
        on, in which case the scaling clause is counted too, so a body
        whose whole damage IS the scale stops reading as harmless.
        """
        damage = float(attack.damage_base or 0)
        if not self._energy_routing or obs is None:
            return damage
        clause = self._scaled_clause(attack.attack_id)
        if clause is None:
            return damage
        return damage + clause.per_unit * self._scaled_units(clause, obs)

    def _attack_value(self, attack_id: int | None, obs: Observation) -> float:
        """Expected value of using an attack (damage-equivalent units)."""
        attack = self._index.get_attack(attack_id) if attack_id is not None else None
        if attack is None:
            return 30.0  # unknown id: attacking still beats passing
        my_card = self._card_of(self._my_active(obs))
        opp = self._opp_active(obs)
        opp_card = self._card_of(opp)

        damage = float(attack.damage_base or 0)
        # Damage that SCALES with the board is the attack's whole point
        # when the printed base is 0; count the unit live instead of
        # letting the fixed per-row bonus stand in for it.
        scaled_applied = False
        if self._scaled_damage and attack_id is not None:
            clause = self._scaled_clause(attack_id)
            if clause is not None:
                damage += clause.per_unit * self._scaled_units(clause, obs)
                scaled_applied = True
        if opp_card is not None and my_card is not None and my_card.type_code is not None:
            if opp_card.weakness_code == my_card.type_code:
                damage *= 2
            if opp_card.resistance_code == my_card.type_code:
                damage = max(0.0, damage - 30)
        value = self._effect_adjustment(attack, damage,
                                        skip_scaling=scaled_applied)
        if opp is not None and value >= opp.hp:
            value += _KO_BONUS
        return value

    def _best_affordable_damage(self, card_id: int | None, energies: list[int],
                                obs: Observation | None = None) -> float:
        if card_id is None:
            return 0.0
        best = 0.0
        for attack in self._index.attacks_of(card_id):
            if self._affordable(attack, energies):
                best = max(best, self._body_damage(attack, obs))
        return best

    # ------------------------------------------------------------------ #
    # Selection handlers
    # ------------------------------------------------------------------ #

    def _decide(self, obs: Observation) -> list[int]:
        select = obs.select
        assert select is not None
        ctx = select.context
        options = select.option

        if ctx == SelectContext.MAIN:
            return [self._best_index(options, lambda i, o: self._main_score(obs, o))]
        if ctx in (SelectContext.ATTACK, SelectContext.DISABLE_ATTACK):
            return [self._best_index(options, lambda i, o: self._attack_value(o.attackId, obs))]
        if ctx in (SelectContext.SETUP_ACTIVE_POKEMON, SelectContext.TO_ACTIVE,
                   SelectContext.SWITCH):
            return [self._best_index(options, lambda i, o: self._own_pokemon_score(obs, o, for_active=True))]
        if ctx in (SelectContext.SETUP_BENCH_POKEMON, SelectContext.TO_BENCH,
                   SelectContext.TO_FIELD):
            # Bench aggressively: an empty bench loses to any active KO.
            return self._pick_top(options, select.maxCount, select.maxCount,
                                  lambda i, o: self._own_pokemon_score(obs, o, for_active=False))
        if ctx in (SelectContext.DAMAGE, SelectContext.DAMAGE_COUNTER,
                   SelectContext.DAMAGE_COUNTER_ANY):
            return self._pick_top(options, select.minCount, max(select.minCount, 1),
                                  lambda i, o: self._enemy_target_score(obs, o))
        if ctx in (SelectContext.HEAL, SelectContext.REMOVE_DAMAGE_COUNTER):
            return self._pick_top(options, select.minCount, max(select.minCount, 1),
                                  lambda i, o: self._heal_target_score(obs, o))
        if ctx == SelectContext.IS_FIRST:
            return [self._yes_no(options, want_yes=False)]  # second player attacks first
        if ctx == SelectContext.MULLIGAN:
            return [self._yes_no(options, want_yes=False)]
        if ctx == SelectContext.MORE_DEVOLVE:
            return [self._yes_no(options, want_yes=False)]
        if ctx in (SelectContext.ACTIVATE, SelectContext.FIRST_EFFECT,
                   SelectContext.COIN_HEAD):
            return [self._yes_no(options, want_yes=True)]
        if ctx in (SelectContext.DRAW_COUNT, SelectContext.DAMAGE_COUNTER_COUNT,
                   SelectContext.REMOVE_DAMAGE_COUNTER_COUNT):
            return [self._best_index(options, lambda i, o: float(o.number or 0))]
        return self._default_answer(select.minCount, select.maxCount, len(options))

    # ---- scoring helpers ---- #

    def _is_lethal(self, value: float, obs: Observation) -> bool:
        """Would this attack knock the opposing active out?

        Read back out of the value rather than recomputed: _attack_value
        adds _KO_BONUS exactly when the damage already reached the
        defender's HP, so subtracting it recovers the raw comparison
        without valuing the attack twice.
        """
        opp = self._opp_active(obs)
        return opp is not None and value - _KO_BONUS >= opp.hp

    def _main_score(self, obs: Observation, option: Option) -> float:
        kind = option.type
        state = obs.current
        if kind == OptionType.ATTACK:
            # capped so attacking never outranks development actions —
            # those keep the MAIN prompt open, attacking ends the turn
            value = self._attack_value(option.attackId, obs)
            if (self._profile == PROFILE_AGGRO
                    and self._is_lethal(value, obs)):
                # AGGRO profile: a knockout is worth more than anything
                # the rest of the turn could develop, so it stops losing
                # to evolve/play/attach (see _AGGRO_LETHAL_BAND).
                return _AGGRO_LETHAL_BAND + min(value, 450.0) / 45.0
            return _ATTACK_BAND + min(value, 450.0) / 45.0 + (5.0 if value >= _KO_BONUS else 0.0)
        if kind == OptionType.ATTACH:
            return self._attach_score(obs, option)
        if kind == OptionType.EVOLVE:
            evolved = self._wrapper.resolve_card_id(obs, option)
            card = self._index.get_card(evolved) if evolved is not None else None
            return _EVOLVE_BAND + ((card.hp or 0) / 40.0 if card else 0.0)
        if kind == OptionType.PLAY:
            card_id = self._wrapper.resolve_card_id(obs, option)
            if self._tempo and card_id in EVOLUTION_ACCELERATORS:
                # the engine only offers this when it really evolves
                return _EVOLVE_BAND + _ACCELERATOR_BONUS
            card = self._index.get_card(card_id) if card_id is not None else None
            if card is None:
                return _TRAINER_BAND
            if card.hp is not None:  # a Pokémon: develop the board
                return _PLAY_POKEMON_BAND + card.hp / 100.0
            return _TRAINER_BAND
        if kind == OptionType.ABILITY:
            return _ABILITY_BAND
        if kind == OptionType.RETREAT:
            active = self._my_active(obs)
            if active is not None and active.maxHp and active.hp <= active.maxHp * 0.4:
                bench = state.players[state.yourIndex].bench if state else []
                if any(p.hp > active.hp for p in bench):
                    return _RETREAT_LOW_HP
            return 1.0
        if kind == OptionType.END:
            return _END_SCORE
        return 1.0

    def _attach_score(self, obs: Observation, option: Option) -> float:
        state = obs.current
        if state is None:
            return 5.0
        energy_id = None
        if option.area == AreaType.HAND:
            card = self._wrapper._card_at(state, state.yourIndex, option.area, option.index)
            energy_id = card.id if card else None
        energy_card = self._index.get_card(energy_id) if energy_id is not None else None
        energy_code = energy_card.type_code if energy_card is not None else int(EnergyType.COLORLESS)

        target = self._pokemon_at(state, state.yourIndex, option.inPlayArea, option.inPlayIndex)
        if target is None:
            return 5.0
        energies = [int(e) for e in (target.energies or [])]
        now = self._best_affordable_damage(target.id, energies, obs)
        then = self._best_affordable_damage(target.id, energies + [energy_code],
                                            obs)
        gain = then - now
        active_bonus = 3.0 if option.inPlayArea == AreaType.ACTIVE else 0.0
        return _ATTACH_BAND + active_bonus + min(gain, 200.0) / 20.0

    def _own_pokemon_score(self, obs: Observation, option: Option, for_active: bool) -> float:
        card_id = self._wrapper.resolve_card_id(obs, option)
        card = self._index.get_card(card_id) if card_id is not None else None
        if card is None:
            return 0.0
        best_damage = max((self._body_damage(a, obs)
                           for a in self._index.attacks_of(card.card_id)),
                          default=0.0)
        score = (card.hp or 0) / 10.0 + best_damage / 10.0
        # Tera Pokémon are immune to attack damage on the Bench (verified in
        # tests/test_tera_bench_immunity.py): keep them there, not in front.
        if card.is_tera:
            score += -8.0 if for_active else 8.0
        return score

    def _enemy_target_score(self, obs: Observation, option: Option) -> float:
        pokemon = self._pokemon_at(obs.current, option.playerIndex, option.area, option.index)
        card = self._card_of(pokemon)
        score = 10.0
        if pokemon is not None:
            score += max(0.0, 30.0 - pokemon.hp / 10.0)  # prefer near-KO targets
        if card is not None and card.is_tera and option.area == AreaType.BENCH:
            score -= 100.0  # damage would be nullified — never waste it
        return score

    def _heal_target_score(self, obs: Observation, option: Option) -> float:
        pokemon = self._pokemon_at(obs.current, option.playerIndex, option.area, option.index)
        if pokemon is None or not pokemon.maxHp:
            return 0.0
        return float(pokemon.maxHp - pokemon.hp)

    # ---- generic pickers ---- #

    @staticmethod
    def _yes_no(options: list[Option], want_yes: bool) -> int:
        wanted = OptionType.YES if want_yes else OptionType.NO
        for i, option in enumerate(options):
            if option.type == wanted:
                return i
        return 0

    def _score_options(self, options: list[Option], score) -> list[float]:
        scores = [self._score_safe(score, i, option) for i, option in enumerate(options)]
        self.last_scores = scores
        return scores

    def _best_index(self, options: list[Option], score) -> int:
        scores = self._score_options(options, score)
        return max(range(len(scores)), key=lambda i: scores[i]) if scores else 0

    def _pick_top(self, options: list[Option], min_count: int, max_count: int, score) -> list[int]:
        scores = self._score_options(options, score)
        count = max(min_count, min(max_count, len(options)))
        ranked = sorted(range(len(options)), key=lambda i: -scores[i])
        return ranked[:count]

    @staticmethod
    def _score_safe(score, i: int, option: Option) -> float:
        try:
            value = score(i, option)
            return value if value is not None else 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _default_answer(min_count: int, max_count: int, n_options: int) -> list[int]:
        count = max(min_count, min(1, max_count))
        return list(range(min(count, n_options)))
