"""EQUIVALENCE GATE: the parameterized Crustle module == shipped v3.

The retrofit is only worth anything if it is behaviourally free. This
asserts that on a REAL distribution of decisions, not a handful of
synthetic ones:

1. `test_replay_equivalence` records every observation the shipped
   CrustleAgent(variant="v3") faces across full games against several
   field decks, then replays each one through
   ParametricHeuristicAgent(CrustleModule(), defaults) and asserts the
   answers are identical index-for-index. Both agents are deterministic
   (scoring never touches the rng), so any divergence is a real bug.

2. `test_score_equivalence_on_v3_rule_cases` re-runs the targeted
   observations from tests/test_crustle_agent_v3.py and compares raw
   FLOAT scores — catching drift that happens to leave the argmax alone.

Production is untouched: this test imports the shipped agent as the
reference and never modifies it.

Run from the repo root:
    python -m unittest tests.test_league_crustle_equivalence
"""

from __future__ import annotations

import copy
import unittest
from typing import Final

from cg import game
from cg.api import AreaType, OptionType
from cg.api import Option as ApiOption

from src.agent_heuristics.crustle_agent import CrustleAgent
from src.deckbuilding.legality import read_deck_ids
from src.ingestion.build_card_model import REPO_ROOT
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex
from src.league.modules.crustle import CrustleModule
from src.league.parametric_agent import ParametricHeuristicAgent

DECKS_DIR: Final = REPO_ROOT / "data" / "decks"
#: opponents chosen to exercise different branches: ex pressure, a
#: mirror (mill race), and an aggro deck (non-ex threat rules).
OPPONENTS: Final[tuple[str, ...]] = ("meta_grimmsnarl", "seed_crustle",
                                     "meta_spidops")
GAMES_PER_OPPONENT: Final[int] = 10
MAX_STEPS: Final[int] = 20_000
POFFIN: Final[int] = 1086
POKEGEAR: Final[int] = 1122
XEROSIC: Final[int] = 1197
GREAT_TUSK: Final[int] = 58
KYUREM_EX: Final[int] = 509
F_ENERGY: Final[int] = 6


def _record_observations(deck: list[int], opponent: list[int], index: CardIndex,
                         effects: EffectIndex, games: int) -> list[dict]:
    """Every obs_dict the shipped v3 agent decides on, driving seat 0."""
    seen: list[dict] = []
    for game_index in range(games):
        pilot = CrustleAgent(seed=game_index, index=index, effects=effects,
                             variant="v3")
        foil = CrustleAgent(seed=500 + game_index, index=index,
                            effects=effects, variant="v3")
        obs_dict, _ = game.battle_start(list(deck), list(opponent))
        if obs_dict is None:
            continue
        try:
            for _ in range(MAX_STEPS):
                state = obs_dict["current"]
                if state["result"] != -1:
                    break
                who = state["yourIndex"]
                if who == 0 and obs_dict.get("select") is not None:
                    seen.append(copy.deepcopy(obs_dict))
                obs_dict = game.battle_select((pilot if who == 0 else foil)(obs_dict))
        finally:
            game.battle_finish()
    return seen


class TestCrustleRetrofitEquivalence(unittest.TestCase):
    """The shipped v3 agent and the module must be indistinguishable."""

    observations: list[dict]

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(DECKS_DIR / "candidate_crustle_e10.csv")
        cls.shipped = CrustleAgent(index=cls.index, effects=cls.effects,
                                   variant="v3")
        cls.module = ParametricHeuristicAgent(
            module=CrustleModule(), index=cls.index, effects=cls.effects)
        cls.observations = []
        for name in OPPONENTS:
            cls.observations += _record_observations(
                cls.deck, read_deck_ids(DECKS_DIR / f"{name}.csv"),
                cls.index, cls.effects, GAMES_PER_OPPONENT)

    def test_recorded_a_real_distribution(self) -> None:
        self.assertGreater(len(self.observations), 1500,
                           "too few decision points to call this a gate")

    def test_replay_equivalence(self) -> None:
        """Same answer, index for index, on every recorded decision."""
        mismatches: list[str] = []
        for i, obs_dict in enumerate(self.observations):
            want = self.shipped(copy.deepcopy(obs_dict))
            got = self.module(copy.deepcopy(obs_dict))
            if want != got:
                ctx = (obs_dict.get("select") or {}).get("context")
                mismatches.append(f"step {i} (context {ctx}): "
                                  f"v3={want} module={got}")
        self.assertEqual(mismatches[:10], [],
                         f"{len(mismatches)}/{len(self.observations)} "
                         f"decisions diverged")

    def test_replay_score_equivalence(self) -> None:
        """Raw per-option scores match, not just the argmax."""
        drifted: list[str] = []
        for i, obs_dict in enumerate(self.observations):
            self.shipped(copy.deepcopy(obs_dict))
            want = self.shipped.last_scores
            self.module(copy.deepcopy(obs_dict))
            got = self.module.last_scores
            if want != got:
                drifted.append(f"step {i}: {want} != {got}")
        self.assertEqual(drifted[:5], [],
                         f"{len(drifted)} decisions had drifting scores")


class TestCrustleRuleCaseEquivalence(unittest.TestCase):
    """Float-level parity on the v3 rule cases (the targeted asserts)."""

    base_obs: dict

    @classmethod
    def setUpClass(cls) -> None:
        from tests.test_crustle_agent import _capture_main_obs
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        deck = read_deck_ids(DECKS_DIR / "seed_crustle.csv")
        cls.base_obs = _capture_main_obs(deck)
        cls.shipped = CrustleAgent(index=cls.index, effects=cls.effects,
                                   variant="v3")
        cls.module = ParametricHeuristicAgent(
            module=CrustleModule(), index=cls.index, effects=cls.effects)

    def _mutated(self, **kwargs):
        obs_dict = copy.deepcopy(self.base_obs)
        state = obs_dict["current"]
        me = state["yourIndex"]
        mine, theirs = state["players"][me], state["players"][1 - me]
        template = dict(mine["active"][0])
        if kwargs.get("my_deck") is not None:
            mine["deckCount"] = kwargs["my_deck"]
        if kwargs.get("opp_deck") is not None:
            theirs["deckCount"] = kwargs["opp_deck"]
        if kwargs.get("hand0_id") is not None:
            mine["hand"][0]["id"] = kwargs["hand0_id"]
        if kwargs.get("my_bench") is not None:
            mine["bench"] = [dict(template, **b) for b in kwargs["my_bench"]]
        if kwargs.get("my_active") is not None:
            mine["active"][0].update(kwargs["my_active"])
        if kwargs.get("opp_active_id") is not None:
            theirs["active"][0]["id"] = kwargs["opp_active_id"]
        if kwargs.get("opp_hand_count") is not None:
            theirs["handCount"] = kwargs["opp_hand_count"]
        return self.module._wrapper.parse(obs_dict)

    def _assert_same(self, obs, option, label: str) -> None:
        self.assertEqual(self.module._main_score(obs, option),
                         self.shipped._main_score(obs, option), label)

    @staticmethod
    def _play0() -> ApiOption:
        return ApiOption(type=OptionType.PLAY, area=AreaType.HAND, index=0)

    def test_rule_I_deck_guard_cases(self) -> None:
        for label, kwargs in (
            ("full deck, behind in race",
             dict(hand0_id=POFFIN, my_deck=44, opp_deck=45, my_bench=[{}, {}])),
            ("low and losing",
             dict(hand0_id=POFFIN, my_deck=25, opp_deck=35, my_bench=[{}, {}])),
            ("low but ahead",
             dict(hand0_id=POFFIN, my_deck=25, opp_deck=20, my_bench=[{}, {}])),
            ("absolute low deck",
             dict(hand0_id=POFFIN, my_deck=8, opp_deck=4, my_bench=[{}, {}])),
        ):
            self._assert_same(self._mutated(**kwargs), self._play0(), label)

    def test_rule_II_board_floor_cases(self) -> None:
        for label, kwargs in (
            ("thin board, builder",
             dict(hand0_id=POFFIN, my_deck=25, opp_deck=35, my_bench=[])),
            ("thin board, plain thinner",
             dict(hand0_id=POKEGEAR, my_deck=25, opp_deck=35, my_bench=[])),
            ("thin board, deck-out risk",
             dict(hand0_id=POFFIN, my_deck=8, opp_deck=35, my_bench=[])),
            ("tie-break vs xerosic",
             dict(hand0_id=XEROSIC, my_deck=44, opp_deck=44, my_bench=[],
                  opp_hand_count=4)),
            ("healthy board",
             dict(hand0_id=POFFIN, my_deck=44, opp_deck=44, my_bench=[{}, {}])),
        ):
            self._assert_same(self._mutated(**kwargs), self._play0(), label)

    def test_rule_E_mill_over_damage(self) -> None:
        obs = self._mutated(my_active={"id": GREAT_TUSK,
                                       "energies": [F_ENERGY] * 4},
                            opp_active_id=KYUREM_EX)
        for attack in self.index.attacks_of(GREAT_TUSK):
            self._assert_same(obs, ApiOption(type=OptionType.ATTACK,
                                             attackId=attack.attack_id),
                              f"attack {attack.move_name}")

    def test_rule_B_xerosic_big_hand(self) -> None:
        for hand in (4, 8, 9, 15):
            obs = self._mutated(hand0_id=XEROSIC, opp_hand_count=hand,
                                my_bench=[{}, {}], my_deck=44, opp_deck=44)
            self._assert_same(obs, self._play0(), f"opp hand {hand}")

    def test_garbage_safety(self) -> None:
        self.assertIsInstance(self.module({"current": None}), list)
        answer = self.module(copy.deepcopy(self.base_obs))
        n_options = len(self.base_obs["select"]["option"])
        self.assertTrue(all(0 <= i < n_options for i in answer))


if __name__ == "__main__":
    unittest.main()
