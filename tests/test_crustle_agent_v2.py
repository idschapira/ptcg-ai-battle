"""Tests for the CrustleAgent v2 rules (kernel Elo-1208 blueprint port).

Same method as tests/test_crustle_agent.py: one real captured MAIN
observation, raw-dict mutations, synthetic options — and every v2 rule
is asserted as a DIFFERENCE against v1 on the identical observation, so
the flag cleanly separates the two behaviors (v1 is the shipped agent).

Run from the repo root:  python -m unittest tests.test_crustle_agent_v2
"""

from __future__ import annotations

import copy
import unittest
from typing import Final

from cg.api import AreaType, OptionType, SelectContext
from cg.api import Option as ApiOption

from src.agent_heuristics.crustle_agent import (BOSS_ORDERS, COLRESS, CRUSTLE,
                                                DWEBBLE, EXPLORERS_GUIDANCE,
                                                GREAT_TUSK, JUMBO_ICE_CREAM,
                                                SWITCH_ITEM, V2_LAND_COLLAPSE,
                                                XEROSIC, CrustleAgent)
from src.deckbuilding.legality import read_deck_ids
from src.ingestion.build_card_model import REPO_ROOT
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex

KYUREM_EX: Final[int] = 509     # ex, retreat cost >= 1 (trappable when dry)
KYOGRE: Final[int] = 721        # non-ex
F_ENERGY: Final[int] = 6
TRAINER_BAND: Final[float] = 35.0
END_SCORE: Final[float] = 0.5


def _capture_main_obs(deck: list[int]) -> dict:
    from tests.test_crustle_agent import _capture_main_obs as capture
    return capture(deck)


class TestCrustleV2Rules(unittest.TestCase):
    base_obs: dict

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        deck = read_deck_ids(REPO_ROOT / "data" / "decks" / "seed_crustle.csv")
        cls.base_obs = _capture_main_obs(deck)
        cls.v1 = CrustleAgent(index=cls.index, effects=cls.effects)
        cls.v2 = CrustleAgent(index=cls.index, effects=cls.effects,
                              variant="v2")
        tusk = cls.index.attacks_of(GREAT_TUSK)
        cls.mill_attack = next(a for a in tusk
                               if a.effect and "deck" in a.effect.lower())
        cls.damage_attack = next(a for a in tusk
                                 if a.attack_id != cls.mill_attack.attack_id)

    # ---- mutation helpers ---- #

    def _mutated(self, hand0_id: int | None = None,
                 my_active: dict | None = None,
                 my_bench: list[dict] | None = None,
                 opp_active_id: int | None = None,
                 opp_bench: list[dict] | None = None,
                 my_deck: int | None = None,
                 opp_deck: int | None = None,
                 opp_hand_count: int | None = None):
        obs_dict = copy.deepcopy(self.base_obs)
        state = obs_dict["current"]
        me = state["yourIndex"]
        mine, theirs = state["players"][me], state["players"][1 - me]
        template = dict(mine["active"][0])
        if my_deck is not None:
            mine["deckCount"] = my_deck
        if hand0_id is not None:
            mine["hand"][0]["id"] = hand0_id
        if my_active is not None:
            mine["active"][0].update(my_active)
        if my_bench is not None:
            mine["bench"] = [dict(template, **b) for b in my_bench]
        if opp_active_id is not None:
            theirs["active"][0]["id"] = opp_active_id
        if opp_bench is not None:
            theirs["bench"] = [dict(template, **b) for b in opp_bench]
        if opp_deck is not None:
            theirs["deckCount"] = opp_deck
        if opp_hand_count is not None:
            theirs["handCount"] = opp_hand_count
        return self.v2._wrapper.parse(obs_dict)

    def _play0(self) -> ApiOption:
        return ApiOption(type=OptionType.PLAY, area=AreaType.HAND, index=0)

    # ---- rule (A): gust-to-trap timing ---- #

    def test_boss_endgame_lock_outranks_trainers(self) -> None:
        obs = self._mutated(hand0_id=BOSS_ORDERS, opp_active_id=KYUREM_EX,
                            opp_bench=[{"id": KYUREM_EX, "energies": []}],
                            opp_deck=8)
        option = self._play0()
        self.assertGreater(self.v2._main_score(obs, option), TRAINER_BAND)
        self.assertLessEqual(self.v1._main_score(obs, option), TRAINER_BAND)

    def test_boss_wasted_without_trappable_target(self) -> None:
        # bench pokémon already powered: nothing to trap -> don't burn it
        obs = self._mutated(hand0_id=BOSS_ORDERS, opp_active_id=KYUREM_EX,
                            opp_bench=[{"id": KYUREM_EX,
                                        "energies": [F_ENERGY, F_ENERGY]}],
                            opp_deck=8)
        option = self._play0()
        self.assertLess(self.v2._main_score(obs, option), TRAINER_BAND)
        self.assertGreater(self.v2._main_score(obs, option), END_SCORE)

    # ---- rule (A): gust target choice ---- #

    def test_gust_target_prefers_trapped_body(self) -> None:
        # heavy dry body (retreat, no energy) vs powered one
        obs = self._mutated(opp_bench=[
            {"id": KYUREM_EX, "energies": [], "hp": 230, "maxHp": 230},
            {"id": KYUREM_EX, "energies": [F_ENERGY, F_ENERGY],
             "hp": 30, "maxHp": 230},
        ])
        opp = 1 - obs.current.yourIndex
        dry = ApiOption(type=OptionType.CARD, area=AreaType.BENCH, index=0,
                        playerIndex=opp)
        powered = ApiOption(type=OptionType.CARD, area=AreaType.BENCH, index=1,
                            playerIndex=opp)
        self.assertGreater(
            self.v2._own_pokemon_score(obs, dry, for_active=True),
            self.v2._own_pokemon_score(obs, powered, for_active=True),
            "v2 must trap the dry heavy body, not chase the near-KO")
        self.assertEqual(
            self.v1._own_pokemon_score(obs, dry, for_active=True),
            self.v1._own_pokemon_score(obs, powered, for_active=True),
            "v1 has no trap-awareness (same card -> same score)")

    # ---- rule (B): Xerosic timing ---- #

    def test_xerosic_boosted_on_big_hand_only(self) -> None:
        big = self._mutated(hand0_id=XEROSIC, opp_hand_count=9)
        small = self._mutated(hand0_id=XEROSIC, opp_hand_count=4)
        option = self._play0()
        self.assertGreater(self.v2._main_score(big, option), TRAINER_BAND)
        self.assertLessEqual(self.v2._main_score(small, option), TRAINER_BAND)
        self.assertLessEqual(self.v1._main_score(big, option), TRAINER_BAND)

    # ---- rule (C): Colress fetches the Zone under ex pressure ---- #

    def test_colress_boosted_when_zone_needed_vs_ex(self) -> None:
        # losing the race RELATIVELY (30 < 35): Zone fetch must still win
        vs_ex = self._mutated(hand0_id=COLRESS, opp_active_id=KYUREM_EX,
                              my_deck=30, opp_deck=35)
        vs_non_ex = self._mutated(hand0_id=COLRESS, opp_active_id=KYOGRE,
                                  opp_bench=[], my_deck=30, opp_deck=35)
        option = self._play0()
        self.assertGreater(self.v2._main_score(vs_ex, option), TRAINER_BAND)
        self.assertLess(self.v2._main_score(vs_non_ex, option), END_SCORE,
                        "no ex threat: the race suppression rules")
        self.assertLess(self.v1._main_score(vs_ex, option), END_SCORE)

    def test_colress_still_suppressed_at_absolute_low_deck(self) -> None:
        obs = self._mutated(hand0_id=COLRESS, opp_active_id=KYUREM_EX,
                            my_deck=8, opp_deck=35)
        self.assertLess(self.v2._main_score(obs, self._play0()), END_SCORE,
                        "the absolute anti-self-mill invariant is king")

    # ---- rule (D): proactive pivot ---- #

    def test_switch_pivots_to_ready_tusk_on_bench(self) -> None:
        obs = self._mutated(hand0_id=SWITCH_ITEM,
                            my_active={"id": DWEBBLE, "energies": []},
                            my_bench=[{"id": GREAT_TUSK,
                                       "energies": [F_ENERGY, F_ENERGY]}])
        option = self._play0()
        self.assertGreater(self.v2._main_score(obs, option), TRAINER_BAND)
        self.assertLessEqual(self.v1._main_score(obs, option), TRAINER_BAND)

    def test_retreat_pivot_beats_generic_but_never_breaks_wall(self) -> None:
        pivot = self._mutated(my_active={"id": DWEBBLE, "energies": []},
                              my_bench=[{"id": GREAT_TUSK,
                                         "energies": [F_ENERGY, F_ENERGY]}],
                              opp_active_id=KYOGRE)
        wall = self._mutated(my_active={"id": CRUSTLE, "energies": []},
                             my_bench=[{"id": GREAT_TUSK,
                                        "energies": [F_ENERGY, F_ENERGY]}],
                             opp_active_id=KYUREM_EX)
        option = ApiOption(type=OptionType.RETREAT)
        self.assertGreater(self.v2._main_score(pivot, option),
                           self.v1._main_score(pivot, option))
        self.assertLess(self.v2._main_score(wall, option), END_SCORE,
                        "rule (iii) still wins: never retreat the wall vs ex")

    # ---- rule (E): mill over damage ---- #

    def test_land_collapse_outranks_giant_tusk(self) -> None:
        obs = self._mutated(my_active={"id": GREAT_TUSK,
                                       "energies": [F_ENERGY] * 4},
                            opp_active_id=KYUREM_EX)
        mill = ApiOption(type=OptionType.ATTACK,
                         attackId=self.mill_attack.attack_id)
        damage = ApiOption(type=OptionType.ATTACK,
                           attackId=self.damage_attack.attack_id)
        self.assertGreater(self.v2._main_score(obs, mill),
                           self.v2._main_score(obs, damage),
                           "the mill IS the win condition")
        self.assertLess(self.v1._main_score(obs, mill),
                        self.v1._main_score(obs, damage),
                        "v1 (generic damage greed) prefers Giant Tusk")
        self.assertLessEqual(self.v2._main_score(obs, mill), TRAINER_BAND)

    # ---- rule (F): search / discard handlers ---- #

    def _selection_obs(self, context: SelectContext, hand_ids: list[int],
                       min_count: int, max_count: int):
        obs_dict = copy.deepcopy(self.base_obs)
        me = obs_dict["current"]["yourIndex"]
        hand = obs_dict["current"]["players"][me]["hand"]
        while len(hand) < len(hand_ids):
            hand.append(dict(hand[0]))
        for i, cid in enumerate(hand_ids):
            hand[i]["id"] = cid
        obs_dict["select"]["context"] = int(context)
        obs_dict["select"]["minCount"] = min_count
        obs_dict["select"]["maxCount"] = max_count
        obs_dict["select"]["option"] = [
            {"type": int(OptionType.CARD), "area": int(AreaType.HAND),
             "index": i, "playerIndex": me} for i in range(len(hand_ids))]
        return obs_dict

    def test_discard_protects_guidance_and_tusk(self) -> None:
        # Ultra Ball cost: discard 2 of [Guidance, Great Tusk, Switch, Boss]
        obs_dict = self._selection_obs(
            SelectContext.DISCARD,
            [EXPLORERS_GUIDANCE, GREAT_TUSK, SWITCH_ITEM, BOSS_ORDERS],
            min_count=2, max_count=2)
        answer = self.v2(copy.deepcopy(obs_dict))
        self.assertEqual(len(answer), 2)
        self.assertNotIn(0, answer, "must keep Explorer's Guidance")
        self.assertNotIn(1, answer, "must keep Great Tusk")
        v1_answer = self.v1(copy.deepcopy(obs_dict))
        self.assertEqual(v1_answer, [0, 1],
                         "v1 default discards the FIRST indices (the bug)")

    def test_to_hand_search_prefers_great_tusk(self) -> None:
        obs_dict = self._selection_obs(
            SelectContext.TO_HAND,
            [JUMBO_ICE_CREAM, GREAT_TUSK, SWITCH_ITEM],
            min_count=1, max_count=1)
        answer = self.v2(copy.deepcopy(obs_dict))
        self.assertEqual(answer, [1], "search must fetch the mill attacker")

    # ---- overlay safety ---- #

    def test_v2_contract_and_garbage_safety(self) -> None:
        answer = self.v2(copy.deepcopy(self.base_obs))
        n_options = len(self.base_obs["select"]["option"])
        self.assertTrue(all(0 <= i < n_options for i in answer))
        self.assertIsInstance(self.v2({"current": None}), list)


if __name__ == "__main__":
    unittest.main()
