"""Unit/contract tests for the CrustleAgent strategy overlay.

Method: capture ONE real MAIN observation from an engine game with the
Crustle deck (so the dict shape is authentic), then MUTATE the raw dict
to create each rule's signal state and score SYNTHETIC options through
the agent — comparing against the generic HeuristicAgent on the very
same observation. Every rule is asserted as a *difference in behavior*
between specialized and generic (or between signal on/off), so a broken
rule cannot pass by riding the generic score.

Run from the repo root:  python -m unittest tests.test_crustle_agent
"""

from __future__ import annotations

import copy
import unittest
from typing import Final

from cg import game
from cg.api import AreaType, OptionType, SelectContext
from cg.api import Option as ApiOption

from src.agent_heuristics.crustle_agent import (CRUSTLE, EXPLORERS_GUIDANCE,
                                                GREAT_TUSK, JUMBO_ICE_CREAM,
                                                CrustleAgent)
from src.agent_heuristics.heuristic_agent import HeuristicAgent
from src.deckbuilding.legality import read_deck_ids
from src.ingestion.build_card_model import REPO_ROOT
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex

POFFIN: Final[int] = 1086
KYUREM_EX: Final[int] = 509
KYOGRE: Final[int] = 721   # non-ex attacker (the Gate-C wall leak)
F_ENERGY: Final[int] = 6
END_SCORE: Final[float] = 0.5
TRAINER_BAND: Final[float] = 35.0
MAX_GAMES: Final[int] = 20


def _capture_main_obs(deck: list[int]) -> dict:
    """Play a Crustle mirror until a MAIN prompt where the acting player
    has a visible hand and both actives are known; return that obs."""
    driver = HeuristicAgent(seed=7)
    for _ in range(MAX_GAMES):
        obs_dict, start = game.battle_start(list(deck), list(deck))
        if obs_dict is None:
            raise RuntimeError(f"battle_start failed: {start.errorType}")
        try:
            for _ in range(600):
                state = obs_dict["current"]
                if state["result"] != -1:
                    break
                select = obs_dict.get("select") or {}
                me = state["yourIndex"]
                players = state["players"]
                if (select.get("context") == int(SelectContext.MAIN)
                        and players[me].get("hand")
                        and players[me]["active"] and players[me]["active"][0]
                        and players[1 - me]["active"]
                        and players[1 - me]["active"][0]):
                    return copy.deepcopy(obs_dict)
                obs_dict = game.battle_select(driver(obs_dict))
        finally:
            game.battle_finish()
    raise RuntimeError("no usable MAIN observation materialized")


class TestCrustleRules(unittest.TestCase):
    base_obs: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        deck = read_deck_ids(REPO_ROOT / "data" / "decks" / "seed_crustle.csv")
        cls.base_obs = _capture_main_obs(deck)
        cls.crustle = CrustleAgent(index=cls.index, effects=cls.effects)
        cls.generic = HeuristicAgent(index=cls.index, effects=cls.effects)

    # ---- mutation helpers (raw dict -> parsed Observation) ---- #

    def _mutated(self, my_deck: int | None = None, opp_deck: int | None = None,
                 hand0_id: int | None = None, my_active: dict | None = None,
                 opp_active_id: int | None = None,
                 opp_bench_ids: list[int] | None = None):
        obs_dict = copy.deepcopy(self.base_obs)
        state = obs_dict["current"]
        me = state["yourIndex"]
        mine, theirs = state["players"][me], state["players"][1 - me]
        if my_deck is not None:
            mine["deckCount"] = my_deck
        if opp_deck is not None:
            theirs["deckCount"] = opp_deck
        if hand0_id is not None:
            mine["hand"][0]["id"] = hand0_id
        if my_active is not None:
            mine["active"][0].update(my_active)
        if opp_active_id is not None:
            theirs["active"][0]["id"] = opp_active_id
        if opp_bench_ids is not None:
            theirs["bench"] = [dict(theirs["active"][0], id=cid, hp=100)
                               for cid in opp_bench_ids]
        return self.crustle._wrapper.parse(obs_dict)

    @staticmethod
    def _play_hand0() -> ApiOption:
        return ApiOption(type=OptionType.PLAY, area=AreaType.HAND, index=0)

    # ---- rule (i): anti-self-mill ---- #

    def test_thinner_suppressed_when_own_deck_low(self) -> None:
        healthy = self._mutated(my_deck=40, opp_deck=40, hand0_id=POFFIN)
        low = self._mutated(my_deck=8, opp_deck=40, hand0_id=POFFIN)
        option = self._play_hand0()
        self.assertGreater(self.crustle._main_score(healthy, option),
                           END_SCORE, "healthy deck: thinning stays useful")
        self.assertLess(self.crustle._main_score(low, option), END_SCORE,
                        "low deck: thinning must rank below passing")
        self.assertGreater(self.generic._main_score(low, option), END_SCORE,
                           "the generic heuristic has no such rule")

    def test_thinner_suppressed_when_losing_the_race(self) -> None:
        losing = self._mutated(my_deck=20, opp_deck=30, hand0_id=POFFIN)
        winning = self._mutated(my_deck=30, opp_deck=20, hand0_id=POFFIN)
        option = self._play_hand0()
        self.assertLess(self.crustle._main_score(losing, option), END_SCORE)
        self.assertGreater(self.crustle._main_score(winning, option), END_SCORE)

    # ---- rule (ii): mill sequencing ---- #

    def test_guidance_boosted_when_great_tusk_ready(self) -> None:
        ready = self._mutated(my_deck=40, opp_deck=40,
                              hand0_id=EXPLORERS_GUIDANCE,
                              my_active={"id": GREAT_TUSK,
                                         "energies": [F_ENERGY, F_ENERGY]})
        not_ready = self._mutated(my_deck=40, opp_deck=40,
                                  hand0_id=EXPLORERS_GUIDANCE)
        option = self._play_hand0()
        boosted = self.crustle._main_score(ready, option)
        self.assertGreater(boosted, TRAINER_BAND,
                           "Ancient supporter must outrank plain trainers "
                           "when it turns Land Collapse into a 4-card mill")
        self.assertGreater(boosted, self.crustle._main_score(not_ready, option))

    # ---- rule (iii): wall logic ---- #

    def test_never_retreat_the_wall_against_ex(self) -> None:
        vs_ex = self._mutated(my_active={"id": CRUSTLE},
                              opp_active_id=KYUREM_EX)
        vs_non_ex = self._mutated(my_active={"id": CRUSTLE},
                                  opp_active_id=KYOGRE)
        option = ApiOption(type=OptionType.RETREAT)
        self.assertLess(self.crustle._main_score(vs_ex, option), END_SCORE,
                        "retreating the wall against an ex must never win")
        self.assertGreaterEqual(self.crustle._main_score(vs_non_ex, option),
                                self.generic._main_score(vs_non_ex, option),
                                "no wall bias when the threat is non-ex")

    def test_promotion_prefers_crustle_against_all_ex_field(self) -> None:
        # TO_ACTIVE-style option resolving to the mutated hand card.
        all_ex = self._mutated(hand0_id=CRUSTLE, opp_active_id=KYUREM_EX,
                               opp_bench_ids=[])
        non_ex = self._mutated(hand0_id=CRUSTLE, opp_active_id=KYOGRE)
        option = ApiOption(type=OptionType.CARD, area=AreaType.HAND, index=0,
                           playerIndex=all_ex.current.yourIndex)
        self.assertGreater(
            self.crustle._own_pokemon_score(all_ex, option, for_active=True),
            self.generic._own_pokemon_score(all_ex, option, for_active=True))
        self.assertLess(
            self.crustle._own_pokemon_score(non_ex, option, for_active=True),
            self.generic._own_pokemon_score(non_ex, option, for_active=True))

    # ---- rule (iv): non-ex threat ---- #

    def test_urgent_heal_under_non_ex_threat(self) -> None:
        threatened = self._mutated(
            hand0_id=JUMBO_ICE_CREAM, opp_active_id=KYOGRE,
            my_active={"id": CRUSTLE, "hp": 70, "maxHp": 150,
                       "energies": [F_ENERGY] * 3})
        vs_ex = self._mutated(
            hand0_id=JUMBO_ICE_CREAM, opp_active_id=KYUREM_EX,
            my_active={"id": CRUSTLE, "hp": 70, "maxHp": 150,
                       "energies": [F_ENERGY] * 3})
        option = self._play_hand0()
        self.assertGreater(self.crustle._main_score(threatened, option),
                           TRAINER_BAND,
                           "heal must become urgent when the wall leaks")
        self.assertLessEqual(self.crustle._main_score(vs_ex, option),
                             TRAINER_BAND,
                             "vs ex the wall absorbs — no urgency")

    def test_attack_pressure_bonus_stays_below_trainer_band(self) -> None:
        attack = self.index.attacks_of(CRUSTLE)[0]
        vs_non_ex = self._mutated(my_active={"id": CRUSTLE},
                                  opp_active_id=KYOGRE)
        option = ApiOption(type=OptionType.ATTACK, attackId=attack.attack_id)
        specialized = self.crustle._main_score(vs_non_ex, option)
        generic = self.generic._main_score(vs_non_ex, option)
        self.assertGreaterEqual(specialized, generic)
        self.assertLessEqual(specialized, TRAINER_BAND,
                             "pressure must never outrank development")

    # ---- overlay safety ---- #

    def test_contract_intact_end_to_end(self) -> None:
        # The specialized agent still answers the captured real prompt
        # with a legal index, and garbage input falls back safely.
        answer = self.crustle(copy.deepcopy(self.base_obs))
        n_options = len(self.base_obs["select"]["option"])
        self.assertTrue(all(0 <= i < n_options for i in answer))
        # garbage input: must not crash; [] is the legal empty answer
        # for an observation with no selectable options
        self.assertIsInstance(self.crustle({"current": None}), list)


if __name__ == "__main__":
    unittest.main()
