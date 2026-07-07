"""Deterministic Observation -> numpy encoders for the policy net.

Consumed by BOTH the policy network (Sprint 5) and the replay parser
(Prompt 4C): the encoding here is the single source of truth. numpy only,
no torch. Every path is None-safe: unknown ids or missing fields encode
as zeros, never raise.

## State vector layout (ENCODING_DIM = GLOBAL + 18*POKEMON_SLOT + HAND)

GLOBAL block (31):
    0 turn/50                 1 turnActionCount/20      2 yourIndex
    3 first_player_is_you     4 first_player_unknown
    5..8 supporterPlayed, stadiumPlayed, energyAttached, retreated
    9 stadium_present        10 stadium_is_yours
   11..14 my deckCount/60, handCount/20, prizeLeft/6, discardCount/60
   15..18 opp same
   19..23 my poisoned/burned/asleep/paralyzed/confused
   24..28 opp same
   29..30 my benchMax/8, opp benchMax/8

POKEMON_SLOT block (62), repeated 18x in order:
    [my active, opp active, my bench 0..7, opp bench 0..7]
    0 present            1 hp/300              2 hp/maxHp
    3 appearThisTurn     4..15 type one-hot (EnergyType 0..11)
   16..18 stage one-hot (basic/stage1/stage2)
   19..21 is_ex, is_mega_ex, is_tera
   22 tera_bench_immunity (is_tera AND slot is a bench slot)
   23 retreat_cost/4
   24..35 weakness one-hot   36..47 resistance one-hot
   48..59 attached energy count per EnergyType /5
   60 total energy/5    61 tool count/2

HAND block (38) — own hand only (opponent hand is a count in GLOBAL):
    0..8 count per Stage category (Stage enum 1..9) /10
    9 hand size /20
   10..37 multi-hot of EffectType present among hand Pokémon attacks

## Option vector layout (OPTION_DIM = 154)

    0..16   OptionType one-hot (0..16)
   17..27   SelectType one-hot (0..10)
   28..76   SelectContext one-hot (0..48, clamped for future values)
   77..80   minCount/5, maxCount/10, remainDamageCounter/10, remainEnergyCost/5
   81..100  referenced card: present, hp/300, type one-hot 12,
            stage one-hot 3, is_ex, is_mega_ex, is_tera
  101..133  referenced attack: present, damage_base/300, cost_total/5,
            payable-by-my-active, coin_flip_any, effect multi-hot 28
  134..150  target: area one-hot 12 (AreaType 1..12), is_self, is_opp,
            index/8, inPlay_is_active, inPlayIndex/8
  151       number/10 (COUNT options)
  152..153  toolIndex present, energyIndex present
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np

from cg.api import AreaType, Observation, Option, Pokemon, State

from ..environment_wrapper.wrapper import EnvironmentWrapper
from ..ingestion.build_card_model import EnergyType, Stage
from ..ingestion.build_effect_model import EffectIndex, EffectType
from ..ingestion.card_index import Card, CardIndex, is_cost_payable

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Dimensions
# --------------------------------------------------------------------------- #

N_ENERGY: Final[int] = 12          # EnergyType 0..11
N_EFFECT: Final[int] = len(EffectType)  # 28
N_OPTION_TYPE: Final[int] = 17     # OptionType 0..16
N_SELECT_TYPE: Final[int] = 11     # SelectType 0..10
N_SELECT_CONTEXT: Final[int] = 49  # SelectContext 0..48 (clamped beyond)
N_AREA: Final[int] = 12            # AreaType 1..12

BENCH_CAP: Final[int] = 8   # Area Zero Underdepths raises benchMax to 8
GLOBAL_DIM: Final[int] = 31
POKEMON_SLOT_DIM: Final[int] = 62
N_SLOTS: Final[int] = 2 + 2 * BENCH_CAP  # my/opp active + my/opp bench
HAND_DIM: Final[int] = 10 + N_EFFECT

ENCODING_DIM: Final[int] = GLOBAL_DIM + N_SLOTS * POKEMON_SLOT_DIM + HAND_DIM
OPTION_DIM: Final[int] = (N_OPTION_TYPE + N_SELECT_TYPE + N_SELECT_CONTEXT + 4
                          + 20 + (5 + N_EFFECT) + 17 + 1 + 2)

# Calibrated over 12,185 selections across 300 self-play games:
# p99.9 = 42, observed max = 56 (MAIN). 64 adds headroom for larger decks.
MAX_OPTIONS: Final[int] = 64

_STAGE_TO_SLOT: Final[dict[int, int]] = {
    int(Stage.BASIC_POKEMON): 0,
    int(Stage.STAGE_1_POKEMON): 1,
    int(Stage.STAGE_2_POKEMON): 2,
}


class StateEncoder:
    """Observation -> float32[ENCODING_DIM], deterministic and None-safe."""

    __slots__ = ("_index", "_effects")

    def __init__(self, index: CardIndex | None = None,
                 effects: EffectIndex | None = None) -> None:
        self._index = index if index is not None else CardIndex()
        self._effects = effects if effects is not None else EffectIndex()

    # ---- pokemon slot ---- #

    def _write_pokemon(self, out: np.ndarray, base: int,
                       pokemon: Pokemon | None, is_bench: bool) -> None:
        if pokemon is None:
            return
        card = self._index.get_card(pokemon.id)
        out[base + 0] = 1.0
        out[base + 1] = (pokemon.hp or 0) / 300.0
        out[base + 2] = (pokemon.hp / pokemon.maxHp) if pokemon.maxHp else 0.0
        out[base + 3] = 1.0 if pokemon.appearThisTurn else 0.0
        if card is not None:
            if card.type_code is not None and 0 <= card.type_code < N_ENERGY:
                out[base + 4 + card.type_code] = 1.0
            slot = _STAGE_TO_SLOT.get(card.stage_code or -1)
            if slot is not None:
                out[base + 16 + slot] = 1.0
            out[base + 19] = 1.0 if card.is_ex else 0.0
            out[base + 20] = 1.0 if card.is_mega_ex else 0.0
            out[base + 21] = 1.0 if card.is_tera else 0.0
            out[base + 22] = 1.0 if (card.is_tera and is_bench) else 0.0
            out[base + 23] = (card.retreat_cost or 0) / 4.0
            if card.weakness_code is not None and 0 <= card.weakness_code < N_ENERGY:
                out[base + 24 + card.weakness_code] = 1.0
            if card.resistance_code is not None and 0 <= card.resistance_code < N_ENERGY:
                out[base + 36 + card.resistance_code] = 1.0
        energies = pokemon.energies or []
        for energy in energies:
            code = int(energy)
            if 0 <= code < N_ENERGY:
                out[base + 48 + code] += 1.0 / 5.0
        out[base + 60] = len(energies) / 5.0
        out[base + 61] = len(pokemon.tools or []) / 2.0

    # ---- blocks ---- #

    def _write_global(self, out: np.ndarray, state: State) -> None:
        you = state.yourIndex
        me, opp = state.players[you], state.players[1 - you]
        out[0] = state.turn / 50.0
        out[1] = state.turnActionCount / 20.0
        out[2] = float(you)
        out[3] = 1.0 if state.firstPlayer == you else 0.0
        out[4] = 1.0 if state.firstPlayer == -1 else 0.0
        out[5] = 1.0 if state.supporterPlayed else 0.0
        out[6] = 1.0 if state.stadiumPlayed else 0.0
        out[7] = 1.0 if state.energyAttached else 0.0
        out[8] = 1.0 if state.retreated else 0.0
        stadium = state.stadium[0] if state.stadium else None
        out[9] = 1.0 if stadium is not None else 0.0
        out[10] = 1.0 if (stadium is not None and stadium.playerIndex == you) else 0.0
        for offset, player in ((11, me), (15, opp)):
            out[offset + 0] = player.deckCount / 60.0
            out[offset + 1] = player.handCount / 20.0
            out[offset + 2] = len(player.prize) / 6.0
            out[offset + 3] = len(player.discard) / 60.0
        for offset, player in ((19, me), (24, opp)):
            out[offset + 0] = 1.0 if player.poisoned else 0.0
            out[offset + 1] = 1.0 if player.burned else 0.0
            out[offset + 2] = 1.0 if player.asleep else 0.0
            out[offset + 3] = 1.0 if player.paralyzed else 0.0
            out[offset + 4] = 1.0 if player.confused else 0.0
        out[29] = me.benchMax / float(BENCH_CAP)
        out[30] = opp.benchMax / float(BENCH_CAP)

    def _write_hand(self, out: np.ndarray, base: int, state: State) -> None:
        me = state.players[state.yourIndex]
        hand = me.hand or []
        for hand_card in hand:
            card = self._index.get_card(hand_card.id)
            if card is None:
                continue
            stage = card.stage_code
            if stage is not None and 1 <= stage <= 9:
                out[base + stage - 1] += 1.0 / 10.0
            for attack in self._index.attacks_of(card.card_id):
                for row in self._effects.effects_of(attack.attack_id):
                    if 0 <= row.effect_type < N_EFFECT:
                        out[base + 10 + row.effect_type] = 1.0
        out[base + 9] = len(hand) / 20.0

    def encode(self, obs: Observation) -> np.ndarray:
        out = np.zeros(ENCODING_DIM, dtype=np.float32)
        state = obs.current
        if state is None:  # initial deck selection: nothing on board yet
            return out
        self._write_global(out, state)

        you = state.yourIndex
        me, opp = state.players[you], state.players[1 - you]
        base = GLOBAL_DIM
        my_active = me.active[0] if me.active else None
        opp_active = opp.active[0] if opp.active else None
        self._write_pokemon(out, base, my_active, is_bench=False)
        self._write_pokemon(out, base + POKEMON_SLOT_DIM, opp_active, is_bench=False)
        for slot in range(BENCH_CAP):
            offset = base + (2 + slot) * POKEMON_SLOT_DIM
            self._write_pokemon(out, offset,
                                me.bench[slot] if slot < len(me.bench) else None,
                                is_bench=True)
            offset = base + (2 + BENCH_CAP + slot) * POKEMON_SLOT_DIM
            self._write_pokemon(out, offset,
                                opp.bench[slot] if slot < len(opp.bench) else None,
                                is_bench=True)
        self._write_hand(out, GLOBAL_DIM + N_SLOTS * POKEMON_SLOT_DIM, state)
        return out


class OptionEncoder:
    """(Observation, Option) -> float32[OPTION_DIM] via EnvironmentWrapper.enrich."""

    __slots__ = ("_index", "_effects", "_wrapper")

    def __init__(self, index: CardIndex | None = None,
                 effects: EffectIndex | None = None) -> None:
        self._index = index if index is not None else CardIndex()
        self._effects = effects if effects is not None else EffectIndex()
        self._wrapper = EnvironmentWrapper(self._index)

    def _my_active_energies(self, obs: Observation) -> list[int]:
        state = obs.current
        if state is None:
            return []
        try:
            active = state.players[state.yourIndex].active
            if active and active[0] is not None:
                return [int(e) for e in (active[0].energies or [])]
        except (IndexError, TypeError):
            pass
        return []

    def _write_card(self, out: np.ndarray, base: int, card: Card | None) -> None:
        if card is None:
            return
        out[base + 0] = 1.0
        out[base + 1] = (card.hp or 0) / 300.0
        if card.type_code is not None and 0 <= card.type_code < N_ENERGY:
            out[base + 2 + card.type_code] = 1.0
        slot = _STAGE_TO_SLOT.get(card.stage_code or -1)
        if slot is not None:
            out[base + 14 + slot] = 1.0
        out[base + 17] = 1.0 if card.is_ex else 0.0
        out[base + 18] = 1.0 if card.is_mega_ex else 0.0
        out[base + 19] = 1.0 if card.is_tera else 0.0

    def encode(self, obs: Observation, option: Option) -> np.ndarray:
        out = np.zeros(OPTION_DIM, dtype=np.float32)
        select = obs.select
        option_type = int(option.type) if option.type is not None else 0
        if 0 <= option_type < N_OPTION_TYPE:
            out[option_type] = 1.0
        base = N_OPTION_TYPE
        if select is not None:
            select_type = int(select.type)
            if 0 <= select_type < N_SELECT_TYPE:
                out[base + select_type] = 1.0
            context = min(int(select.context), N_SELECT_CONTEXT - 1)
            if context >= 0:
                out[base + N_SELECT_TYPE + context] = 1.0
            counts = base + N_SELECT_TYPE + N_SELECT_CONTEXT
            out[counts + 0] = select.minCount / 5.0
            out[counts + 1] = select.maxCount / 10.0
            out[counts + 2] = select.remainDamageCounter / 10.0
            out[counts + 3] = select.remainEnergyCost / 5.0

        enriched = self._wrapper.enrich(obs, option)
        card_base = base + N_SELECT_TYPE + N_SELECT_CONTEXT + 4
        self._write_card(out, card_base, enriched.card)

        attack_base = card_base + 20
        attack = enriched.attack
        if attack is not None:
            out[attack_base + 0] = 1.0
            out[attack_base + 1] = (attack.damage_base or 0) / 300.0
            out[attack_base + 2] = attack.cost_total / 5.0
            out[attack_base + 3] = 1.0 if is_cost_payable(
                attack.cost, self._my_active_energies(obs)) else 0.0
            rows = self._effects.effects_of(attack.attack_id)
            out[attack_base + 4] = 1.0 if any(r.coin_flip for r in rows) else 0.0
            for row in rows:
                if 0 <= row.effect_type < N_EFFECT:
                    out[attack_base + 5 + row.effect_type] = 1.0

        target_base = attack_base + 5 + N_EFFECT
        if option.area is not None and 1 <= int(option.area) <= N_AREA:
            out[target_base + int(option.area) - 1] = 1.0
        you = obs.current.yourIndex if obs.current is not None else 0
        if option.playerIndex is not None:
            out[target_base + 12] = 1.0 if option.playerIndex == you else 0.0
            out[target_base + 13] = 1.0 if option.playerIndex != you else 0.0
        if option.index is not None:
            out[target_base + 14] = option.index / 8.0
        if option.inPlayArea is not None:
            out[target_base + 15] = 1.0 if option.inPlayArea == AreaType.ACTIVE else 0.0
        if option.inPlayIndex is not None:
            out[target_base + 16] = option.inPlayIndex / 8.0

        tail = target_base + 17
        out[tail + 0] = (option.number or 0) / 10.0
        out[tail + 1] = 1.0 if option.toolIndex is not None else 0.0
        out[tail + 2] = 1.0 if option.energyIndex is not None else 0.0
        return out


def build_action_mask(
    obs: Observation,
    state_encoder: StateEncoder | None = None,
    option_encoder: OptionEncoder | None = None,
    max_options: int = MAX_OPTIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode all legal options, padded to max_options.

    Returns (options [max_options, OPTION_DIM] float32, mask [max_options] bool).
    Overflowing options are clamped (and logged) — the mask never marks a
    padded row legal, and a real option is only lost beyond max_options.
    """
    encoder = option_encoder if option_encoder is not None else OptionEncoder()
    options_matrix = np.zeros((max_options, OPTION_DIM), dtype=np.float32)
    mask = np.zeros(max_options, dtype=bool)
    indices = EnvironmentWrapper.legal_option_indices(obs)
    if len(indices) > max_options:
        logger.warning("option overflow: %d options > cap %d (clamped)",
                       len(indices), max_options)
        indices = indices[:max_options]
    assert obs.select is not None or not indices
    for i in indices:
        options_matrix[i] = encoder.encode(obs, obs.select.option[i])
        mask[i] = True
    return options_matrix, mask


# --------------------------------------------------------------------------- #
# Profiling entry point
# --------------------------------------------------------------------------- #


def _profile(n_games: int = 8) -> None:
    import time

    from cg import game as cg_game

    from ..agent_heuristics.heuristic_agent import HeuristicAgent
    from ..agent_heuristics.random_agent import RandomAgent, read_deck_csv

    index = CardIndex()
    effects = EffectIndex()
    state_encoder = StateEncoder(index, effects)
    option_encoder = OptionEncoder(index, effects)
    wrapper = EnvironmentWrapper(index)

    observations: list[Observation] = []
    deck = read_deck_csv()
    agents = (HeuristicAgent(seed=1, index=index, effects=effects), RandomAgent(seed=2))
    for _ in range(n_games):
        obs_dict, _ = cg_game.battle_start(list(deck), list(deck))
        try:
            for _ in range(2000):
                if obs_dict["current"]["result"] != -1:
                    break
                observations.append(wrapper.parse(obs_dict))
                acting = obs_dict["current"]["yourIndex"]
                obs_dict = cg_game.battle_select(agents[acting](obs_dict))
        finally:
            cg_game.battle_finish()

    print(f"ENCODING_DIM = {ENCODING_DIM}  OPTION_DIM = {OPTION_DIM}  "
          f"MAX_OPTIONS = {MAX_OPTIONS}")
    print(f"observations collected: {len(observations)}")

    t0 = time.perf_counter()
    for obs in observations:
        state_encoder.encode(obs)
    state_us = (time.perf_counter() - t0) / len(observations) * 1e6

    t0 = time.perf_counter()
    n_options = 0
    for obs in observations:
        if obs.select is not None:
            for option in obs.select.option:
                option_encoder.encode(obs, option)
                n_options += 1
    option_us = (time.perf_counter() - t0) / max(n_options, 1) * 1e6

    t0 = time.perf_counter()
    for obs in observations:
        build_action_mask(obs, state_encoder, option_encoder)
    mask_us = (time.perf_counter() - t0) / len(observations) * 1e6

    print(f"StateEncoder.encode:   {state_us:8.1f} us/obs   (goal < 100 us)")
    print(f"OptionEncoder.encode:  {option_us:8.1f} us/option over {n_options} options")
    print(f"build_action_mask:     {mask_us:8.1f} us/obs (all options + padding)")


if __name__ == "__main__":
    _profile()
