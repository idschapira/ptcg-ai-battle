"""Tests for the CrustleAgent v3 rules (board-wipe fix over v2).

Same method as tests/test_crustle_agent_v2.py: one real captured MAIN
observation, raw-dict mutations, synthetic options — every v3 rule is
asserted as a DIFFERENCE against v2 on the identical observation (v2 is
the shipped agent; the flag must cleanly separate the behaviors), plus
the non-regression asserts: with a genuinely low deck v3 still brakes
(the mirror defense), and the representative v2 rules stay active.

Run from the repo root:  python -m unittest tests.test_crustle_agent_v3
"""

from __future__ import annotations

import copy
import unittest
from typing import Final

from cg.api import AreaType, OptionType
from cg.api import Option as ApiOption

from src.agent_heuristics.crustle_agent import (GREAT_TUSK,
                                                V3_REBUILD_SCORE, XEROSIC,
                                                CrustleAgent)
from src.deckbuilding.legality import read_deck_ids
from src.ingestion.build_card_model import REPO_ROOT
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex

POFFIN: Final[int] = 1086       # board builder (2 basics to the bench)
POKEGEAR: Final[int] = 1122     # thinner that does NOT develop the board
KYUREM_EX: Final[int] = 509
F_ENERGY: Final[int] = 6
TRAINER_BAND: Final[float] = 35.0
END_SCORE: Final[float] = 0.5


def _capture_main_obs(deck: list[int]) -> dict:
    from tests.test_crustle_agent import _capture_main_obs as capture
    return capture(deck)


class TestCrustleV3Rules(unittest.TestCase):
    base_obs: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        deck = read_deck_ids(REPO_ROOT / "data" / "decks" / "seed_crustle.csv")
        cls.base_obs = _capture_main_obs(deck)
        cls.v2 = CrustleAgent(index=cls.index, effects=cls.effects,
                              variant="v2")
        cls.v3 = CrustleAgent(index=cls.index, effects=cls.effects,
                              variant="v3")

    # ---- mutation helpers (raw dict -> parsed Observation) ---- #

    def _mutated(self, hand0_id: int | None = None,
                 my_deck: int | None = None, opp_deck: int | None = None,
                 my_bench: list[dict] | None = None,
                 my_active: dict | None = None,
                 opp_active_id: int | None = None,
                 opp_hand_count: int | None = None):
        obs_dict = copy.deepcopy(self.base_obs)
        state = obs_dict["current"]
        me = state["yourIndex"]
        mine, theirs = state["players"][me], state["players"][1 - me]
        template = dict(mine["active"][0])
        if my_deck is not None:
            mine["deckCount"] = my_deck
        if opp_deck is not None:
            theirs["deckCount"] = opp_deck
        if hand0_id is not None:
            mine["hand"][0]["id"] = hand0_id
        if my_bench is not None:
            mine["bench"] = [dict(template, **b) for b in my_bench]
        if my_active is not None:
            mine["active"][0].update(my_active)
        if opp_active_id is not None:
            theirs["active"][0]["id"] = opp_active_id
        if opp_hand_count is not None:
            theirs["handCount"] = opp_hand_count
        return self.v3._wrapper.parse(obs_dict)

    @staticmethod
    def _play0() -> ApiOption:
        return ApiOption(type=OptionType.PLAY, area=AreaType.HAND, index=0)

    # ---- rule (I): conservative deck guard ---- #

    def test_full_deck_never_suppresses_setup(self) -> None:
        # THE board-wipe cause: turn-1-like state, nearly full deck, one
        # card behind in the race. Board healthy so rule (II) stays out.
        obs = self._mutated(hand0_id=POFFIN, my_deck=44, opp_deck=45,
                            my_bench=[{}, {}])
        option = self._play0()
        self.assertGreater(self.v3._main_score(obs, option), END_SCORE,
                           "v3 must NOT suppress consistency on a full deck")
        self.assertLess(self.v2._main_score(obs, option), END_SCORE,
                        "v2's relative trigger suppressed here (the bug)")

    def test_race_guard_still_brakes_when_low_and_losing(self) -> None:
        losing = self._mutated(hand0_id=POFFIN, my_deck=25, opp_deck=35,
                               my_bench=[{}, {}])
        winning = self._mutated(hand0_id=POFFIN, my_deck=25, opp_deck=20,
                                my_bench=[{}, {}])
        option = self._play0()
        self.assertLess(self.v3._main_score(losing, option), END_SCORE,
                        "low deck AND losing the race: the mirror defense")
        self.assertGreater(self.v3._main_score(winning, option), END_SCORE,
                           "low-ish deck but AHEAD: no reason to freeze")

    def test_absolute_low_deck_invariant_is_king(self) -> None:
        # even WINNING the race, deck <= LOW_DECK suppresses (v2 parity)
        obs = self._mutated(hand0_id=POFFIN, my_deck=8, opp_deck=4,
                            my_bench=[{}, {}])
        option = self._play0()
        self.assertLess(self.v3._main_score(obs, option), END_SCORE)
        self.assertLess(self.v2._main_score(obs, option), END_SCORE)

    # ---- rule (II): desired_field_floor ---- #

    def test_board_floor_rescues_builders_from_suppression(self) -> None:
        # suppression conditions active for v3 too (deck 25 < 35, low):
        # a thin board rescues the BUILDER, not the plain thinner.
        poffin = self._mutated(hand0_id=POFFIN, my_deck=25, opp_deck=35,
                               my_bench=[])
        pokegear = self._mutated(hand0_id=POKEGEAR, my_deck=25, opp_deck=35,
                                 my_bench=[])
        option = self._play0()
        self.assertGreaterEqual(self.v3._main_score(poffin, option),
                                V3_REBUILD_SCORE,
                                "thin board: rebuild outranks suppression")
        self.assertLess(self.v3._main_score(pokegear, option), END_SCORE,
                        "Pokégear builds no board: stays suppressed")
        self.assertLess(self.v2._main_score(poffin, option), END_SCORE,
                        "v2 wiped here: no field floor")

    def test_board_floor_never_overrides_low_deck_invariant(self) -> None:
        obs = self._mutated(hand0_id=POFFIN, my_deck=8, opp_deck=35,
                            my_bench=[])
        self.assertLess(self.v3._main_score(obs, self._play0()), END_SCORE,
                        "never rebuild into deck-out")

    # ---- rule (II) tie-break: builders beat tied 35.0 trainers ---- #

    def test_thin_board_tiebreak_prefers_builder(self) -> None:
        # deck healthy and race even: no suppression anywhere. Xerosic
        # with a small opponent hand scores the plain trainer band, so
        # v2 TIES it with Poffin at 35.0; v3 must break toward the board.
        poffin = self._mutated(hand0_id=POFFIN, my_deck=44, opp_deck=44,
                               my_bench=[], opp_hand_count=4)
        xerosic = self._mutated(hand0_id=XEROSIC, my_deck=44, opp_deck=44,
                                my_bench=[], opp_hand_count=4)
        option = self._play0()
        self.assertGreater(self.v3._main_score(poffin, option),
                           self.v3._main_score(xerosic, option))
        self.assertEqual(self.v2._main_score(poffin, option),
                         self.v2._main_score(xerosic, option),
                         "v2 ties them: first index wins arbitrarily")

    def test_healthy_board_gets_no_boost(self) -> None:
        obs = self._mutated(hand0_id=POFFIN, my_deck=44, opp_deck=44,
                            my_bench=[{}, {}])  # field = 3: floor met
        option = self._play0()
        self.assertEqual(self.v3._main_score(obs, option),
                         self.v2._main_score(obs, option),
                         "v3 == v2 whenever the board is developed")

    # ---- v2 rules stay active in v3 (representative asserts) ---- #

    def test_v3_keeps_v2_mill_over_damage(self) -> None:
        tusk = self.index.attacks_of(GREAT_TUSK)
        mill = next(a for a in tusk
                    if a.effect and "deck" in a.effect.lower())
        damage = next(a for a in tusk if a.attack_id != mill.attack_id)
        obs = self._mutated(my_active={"id": GREAT_TUSK,
                                       "energies": [F_ENERGY] * 4},
                            opp_active_id=KYUREM_EX)
        self.assertGreater(
            self.v3._main_score(obs, ApiOption(type=OptionType.ATTACK,
                                               attackId=mill.attack_id)),
            self.v3._main_score(obs, ApiOption(type=OptionType.ATTACK,
                                               attackId=damage.attack_id)),
            "v2 rule (E) must survive in v3")

    def test_v3_xerosic_on_big_hand(self) -> None:
        big = self._mutated(hand0_id=XEROSIC, opp_hand_count=9,
                            my_bench=[{}, {}], my_deck=44, opp_deck=44)
        self.assertGreater(self.v3._main_score(big, self._play0()),
                           TRAINER_BAND, "v2 rule (B) must survive in v3")

    # ---- overlay safety ---- #

    def test_v3_contract_and_garbage_safety(self) -> None:
        answer = self.v3(copy.deepcopy(self.base_obs))
        n_options = len(self.base_obs["select"]["option"])
        self.assertTrue(all(0 <= i < n_options for i in answer))
        self.assertIsInstance(self.v3({"current": None}), list)


if __name__ == "__main__":
    unittest.main()
