"""HeuristicAgent v1: greedy typed policy over CardIndex + dim_effect.

Decision quality is intentionally simple (policy-net comes later); the
hard requirements are the engine contract and total crash-safety: every
path falls back to a legal answer even when data is missing.
"""

from __future__ import annotations

import random
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

from ..environment_wrapper.wrapper import EnvironmentWrapper
from ..ingestion.build_card_model import EnergyType
from ..ingestion.build_effect_model import EffectIndex, EffectRow, EffectType
from ..ingestion.card_index import Attack, Card, CardIndex
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

_STATUS_VALUE: Final[dict[int, float]] = {0: 20.0, 1: 20.0, 2: 30.0, 3: 30.0, 4: 25.0}


class HeuristicAgent:
    """Greedy one-ply evaluator satisfying the competition contract."""

    __slots__ = ("_index", "_effects", "_wrapper", "_deck_path", "_rng")

    def __init__(
        self,
        seed: int | None = None,
        deck_path: str | None = None,
        index: CardIndex | None = None,
        effects: EffectIndex | None = None,
    ) -> None:
        self._index = index if index is not None else CardIndex()
        self._effects = effects if effects is not None else EffectIndex()
        self._wrapper = EnvironmentWrapper(self._index)
        self._deck_path = deck_path
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ #
    # Contract entry point
    # ------------------------------------------------------------------ #

    def __call__(self, obs_dict: dict) -> list[int]:
        try:
            obs = self._wrapper.parse(obs_dict)
            if obs.select is None:
                return read_deck_csv(self._deck_path)
            return self._decide(obs)
        except Exception:
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
        pool = list(energies)
        colorless_needed = 0
        for energy_type, qty in attack.cost:
            if energy_type == int(EnergyType.COLORLESS):
                colorless_needed += qty
                continue
            for _ in range(qty):
                if energy_type in pool:
                    pool.remove(energy_type)
                elif int(EnergyType.RAINBOW) in pool:
                    pool.remove(int(EnergyType.RAINBOW))
                else:
                    return False
        return len(pool) >= colorless_needed

    def _effect_adjustment(self, attack: Attack, base: float) -> float:
        """Expected-value tweak from dim_effect rows; scale gates multiply."""
        bonus = 0.0
        multiplier = 1.0
        for row in self._effects.effects_of(attack.attack_id):
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

    def _attack_value(self, attack_id: int | None, obs: Observation) -> float:
        """Expected value of using an attack (damage-equivalent units)."""
        attack = self._index.get_attack(attack_id) if attack_id is not None else None
        if attack is None:
            return 30.0  # unknown id: attacking still beats passing
        my_card = self._card_of(self._my_active(obs))
        opp = self._opp_active(obs)
        opp_card = self._card_of(opp)

        damage = float(attack.damage_base or 0)
        if opp_card is not None and my_card is not None and my_card.type_code is not None:
            if opp_card.weakness_code == my_card.type_code:
                damage *= 2
            if opp_card.resistance_code == my_card.type_code:
                damage = max(0.0, damage - 30)
        value = self._effect_adjustment(attack, damage)
        if opp is not None and value >= opp.hp:
            value += _KO_BONUS
        return value

    def _best_affordable_damage(self, card_id: int | None, energies: list[int]) -> float:
        if card_id is None:
            return 0.0
        best = 0.0
        for attack in self._index.attacks_of(card_id):
            if self._affordable(attack, energies):
                best = max(best, float(attack.damage_base or 0))
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

    def _main_score(self, obs: Observation, option: Option) -> float:
        kind = option.type
        state = obs.current
        if kind == OptionType.ATTACK:
            # capped so attacking never outranks development actions —
            # those keep the MAIN prompt open, attacking ends the turn
            value = self._attack_value(option.attackId, obs)
            return _ATTACK_BAND + min(value, 450.0) / 45.0 + (5.0 if value >= _KO_BONUS else 0.0)
        if kind == OptionType.ATTACH:
            return self._attach_score(obs, option)
        if kind == OptionType.EVOLVE:
            evolved = self._wrapper.resolve_card_id(obs, option)
            card = self._index.get_card(evolved) if evolved is not None else None
            return _EVOLVE_BAND + ((card.hp or 0) / 40.0 if card else 0.0)
        if kind == OptionType.PLAY:
            card_id = self._wrapper.resolve_card_id(obs, option)
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
        now = self._best_affordable_damage(target.id, energies)
        then = self._best_affordable_damage(target.id, energies + [energy_code])
        gain = then - now
        active_bonus = 3.0 if option.inPlayArea == AreaType.ACTIVE else 0.0
        return _ATTACH_BAND + active_bonus + min(gain, 200.0) / 20.0

    def _own_pokemon_score(self, obs: Observation, option: Option, for_active: bool) -> float:
        card_id = self._wrapper.resolve_card_id(obs, option)
        card = self._index.get_card(card_id) if card_id is not None else None
        if card is None:
            return 0.0
        best_damage = max((float(a.damage_base or 0) for a in self._index.attacks_of(card.card_id)),
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

    def _best_index(self, options: list[Option], score) -> int:
        best_i, best_s = 0, float("-inf")
        for i, option in enumerate(options):
            s = self._score_safe(score, i, option)
            if s > best_s:
                best_i, best_s = i, s
        return best_i

    def _pick_top(self, options: list[Option], min_count: int, max_count: int, score) -> list[int]:
        count = max(min_count, min(max_count, len(options)))
        ranked = sorted(range(len(options)),
                        key=lambda i: -self._score_safe(score, i, options[i]))
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
