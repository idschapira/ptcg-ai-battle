"""Empirical contract tests for the Crustle-stall mechanics (Sprint 5D pivot).

Locks the three engine behaviors the stall archetype depends on — these
fail loudly if a future engine version changes any of them:

(a) ex-specificity of the ability. Crustle 345's "Mysterious Rock Inn"
    ("Prevent all damage done to this Pokémon by attacks from your
    opponent's Pokémon {ex}") really ZEROES damage from an ex attacker
    (Kyurem ex, Slash 50) and really LETS THROUGH damage from a non-ex
    attacker (Shaymin, Rear Kick 50). Both attackers are chosen with
    colorless costs, plain damage (no effect text) and no Fire typing
    (Crustle is weak to Fire), so the deltas are unconfounded.

(b) deck-out (the archetype's ACTIVE win condition — Great Tusk mills):
    a player whose deck is empty at the start of their turn LOSES
    immediately (GameProc.h: FinishReason::Deck0). Verified by a
    pass-only mirror: the first player draws first, empties first, and
    loses at turn ~95 — decisively, not as a draw.

(c) caps/draws: the engine draws a game only at turn >= 10000
    (GameProc.h) or 3000 total engine actions (BattleData.h,
    FinishReason::Other) — the pass-only mirror ends by deck-out around
    turn 95 / ~100 selections, far below both caps, so pure stalling
    cannot run the clock into a draw; deck-out resolves first.

The engine has no seed API, so scenario games replay until they
materialize (setup mulligans make each attempt cheap; capped attempts).

Run from the repo root:  python -m unittest tests.test_crustle_stall_contract
"""

from __future__ import annotations

import unittest
from typing import Final

from cg import game
from cg.api import OptionType, SelectContext

from src.environment_wrapper.wrapper import EnvironmentWrapper
from src.ingestion.card_index import CardIndex

DWEBBLE: Final[int] = 344
CRUSTLE: Final[int] = 345
KYUREM_EX: Final[int] = 509      # Slash (id 721): {C}{C}, 50, no effect
SLASH: Final[int] = 721
SHAYMIN: Final[int] = 45         # Rear Kick (id 43): {C}{C}, 50, no effect
REAR_KICK: Final[int] = 43
TING_LU: Final[int] = 41         # inert basic for the pass-only mirror
F_ENERGY: Final[int] = 6
W_ENERGY: Final[int] = 3

DEFENDER_DECK: Final[list[int]] = [DWEBBLE] * 4 + [CRUSTLE] * 4 + [W_ENERGY] * 52
PASS_DECK: Final[list[int]] = [TING_LU] * 4 + [W_ENERGY] * 56

MAX_ATTEMPTS: Final[int] = 80
MAX_SELECTIONS: Final[int] = 3000


class _StallProbe:
    """Attacker (P0) rushes its plain attack; defender (P1) evolves the
    active Dwebble into Crustle and otherwise passes. Defender goes
    first whenever it wins the coin, so Crustle is up before the second
    hit lands."""

    def __init__(self, wrapper: EnvironmentWrapper, attack_id: int) -> None:
        self._wrapper = wrapper
        self._attack_id = attack_id

    @staticmethod
    def _defender_active(obs_dict: dict) -> tuple[int, int] | None:
        active = obs_dict["current"]["players"][1]["active"]
        if not active or active[0] is None:
            return None
        return active[0]["id"], active[0]["hp"]

    def _choose(self, obs_dict: dict) -> tuple[list[int], bool]:
        """Returns (answer, is_attack_by_p0)."""
        obs = self._wrapper.parse(obs_dict)
        select = obs.select
        assert select is not None
        acting = obs.current.yourIndex
        options = select.option
        ctx = select.context

        if ctx == SelectContext.IS_FIRST:
            yes = next(i for i, o in enumerate(options) if o.type == OptionType.YES)
            no = next(i for i, o in enumerate(options) if o.type == OptionType.NO)
            return ([yes] if acting == 1 else [no]), False
        if ctx == SelectContext.MULLIGAN:
            return [next((i for i, o in enumerate(options)
                          if o.type == OptionType.NO), 0)], False
        if ctx == SelectContext.MAIN and acting == 1:
            for i, option in enumerate(options):
                if option.type == OptionType.EVOLVE:
                    return [i], False
            return [next(i for i, o in enumerate(options)
                         if o.type == OptionType.END)], False
        if ctx == SelectContext.MAIN and acting == 0:
            for i, option in enumerate(options):
                if option.type == OptionType.ATTACK and option.attackId == self._attack_id:
                    return [i], True
            for i, option in enumerate(options):
                if option.type == OptionType.ATTACH:
                    return [i], False
            return [next(i for i, o in enumerate(options)
                         if o.type == OptionType.END)], False
        return list(range(select.minCount)), False

    def measure_attack_on_crustle(self, attacker_deck: list[int]) -> int | None:
        """Play one game; return the defender-active HP delta for the
        first attack that resolves while Crustle is active, else None."""
        obs_dict, start = game.battle_start(list(attacker_deck),
                                            list(DEFENDER_DECK))
        if obs_dict is None:
            raise RuntimeError(f"battle_start failed: {start.errorType}")
        try:
            for _ in range(MAX_SELECTIONS):
                if obs_dict["current"]["result"] != -1:
                    return None
                before = self._defender_active(obs_dict)
                answer, is_attack = self._choose(obs_dict)
                obs_dict = game.battle_select(answer)
                if is_attack and before is not None and before[0] == CRUSTLE:
                    after = self._defender_active(obs_dict)
                    if after is not None and after[0] == CRUSTLE:
                        return after[1] - before[1]
            return None
        finally:
            game.battle_finish()


class TestCrustleAbilityExSpecificity(unittest.TestCase):
    wrapper: EnvironmentWrapper

    @classmethod
    def setUpClass(cls) -> None:
        cls.wrapper = EnvironmentWrapper(CardIndex())

    def _measure(self, attacker: int, attack_id: int) -> int:
        probe = _StallProbe(self.wrapper, attack_id)
        deck = [attacker] * 4 + [F_ENERGY] * 56
        for _ in range(MAX_ATTEMPTS):
            delta = probe.measure_attack_on_crustle(deck)
            if delta is not None:
                return delta
        self.fail(f"attack on active Crustle never materialized "
                  f"in {MAX_ATTEMPTS} games")

    def test_ability_zeroes_damage_from_ex(self) -> None:
        delta = self._measure(KYUREM_EX, SLASH)
        self.assertEqual(delta, 0,
                         "Mysterious Rock Inn no longer prevents ex damage")

    def test_ability_lets_non_ex_damage_through(self) -> None:
        delta = self._measure(SHAYMIN, REAR_KICK)
        self.assertEqual(delta, -50,
                         "non-ex damage must land on Crustle normally")


class TestDeckOutContract(unittest.TestCase):
    """Pass-only mirror: first player empties first and must LOSE, far
    below the 10000-turn / 3000-action draw caps."""

    def test_deckout_is_a_loss_for_the_emptied_player(self) -> None:
        wrapper = EnvironmentWrapper(CardIndex())
        obs_dict, start = game.battle_start(list(PASS_DECK), list(PASS_DECK))
        self.assertIsNotNone(obs_dict, f"battle_start failed: {start}")
        first_player: int | None = None
        try:
            for _ in range(MAX_SELECTIONS):
                state = obs_dict["current"]
                if state["result"] != -1:
                    break
                obs = wrapper.parse(obs_dict)
                select = obs.select
                options = select.option
                acting = state["yourIndex"]
                if select.context == SelectContext.IS_FIRST:
                    first_player = acting
                    answer = [next(i for i, o in enumerate(options)
                                   if o.type == OptionType.YES)]
                elif select.context == SelectContext.MAIN:
                    answer = [next(i for i, o in enumerate(options)
                                   if o.type == OptionType.END)]
                else:
                    answer = list(range(select.minCount))
                obs_dict = game.battle_select(answer)
        finally:
            game.battle_finish()

        state = obs_dict["current"]
        self.assertIn(state["result"], (0, 1),
                      "pass-only mirror must end decisively (deck-out), "
                      "not as a draw — the caps must not trigger first")
        self.assertIsNotNone(first_player)
        loser = 1 - state["result"]
        self.assertEqual(loser, first_player,
                         "the player who draws first must deck-out first")
        self.assertEqual(state["players"][loser]["deckCount"], 0,
                         "the loser must have an empty deck")
        self.assertLess(state["turn"], 200,
                        "deck-out must resolve near turn ~95, far below caps")


if __name__ == "__main__":
    unittest.main()
