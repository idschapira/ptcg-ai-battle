"""Contract: the submissible search agent plays legal, and its floor is the ship.

The whole risk argument for shipping search rests on one claim — that
every failure path lands on CrustleAgent v3, the agent already on the
ladder. This file tries to make that claim false.

The floor test is an EXACT-IDENTITY check over a real engine game, not a
winrate comparison: with search disabled the agent must return the same
indices as a bare CrustleAgent v3, decision for decision. A winrate test
would pass at 50% even if the two agents disagreed constantly, because
the mirror is symmetric; identity is the property we actually depend on.

Contract tests here use REAL options from a live engine game. A
synthetic option dict can be built to satisfy any code path and would
hide a rule that no longer exists — the same reason the Tera-bench and
Crustle-stall contracts drive the engine rather than a fixture.

Run from the repo root:
    python -m unittest tests.test_runtime_search_agent
"""

from __future__ import annotations

import copy
import unittest

from src.agent_heuristics.crustle_agent import CrustleAgent
from src.agent_heuristics.random_agent import RandomAgent
from src.deckbuilding.legality import read_deck_ids
from src.environment_wrapper.selfplay import play_one_game
from src.ingestion.build_card_model import REPO_ROOT
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex
from src.rl_models.budget import BudgetGuard, SearchTier
from src.rl_models.opponent_estimator import OpponentDeckEstimator
from src.rl_models.runtime_search_agent import RuntimeSearchAgent

OUR_DECK = REPO_ROOT / "deck.csv"
OPP_DECK = REPO_ROOT / "data" / "decks" / "meta_alakazam.csv"

# A ladder that always grants the cheapest useful search, so tests that
# want the search EXERCISED do not have to fight the budget.
CHEAP_LADDER = (SearchTier("2x1", 2, 1, 0.0),)


class TestRuntimeSearchAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.our_deck = read_deck_ids(OUR_DECK)
        cls.opp_deck = read_deck_ids(OPP_DECK)

    def _agent(self, seed: int = 0, **kwargs) -> RuntimeSearchAgent:
        kwargs.setdefault("guard", BudgetGuard(ladder=CHEAP_LADDER,
                                               reserve_s=0.0))
        return RuntimeSearchAgent(deck_path=str(OUR_DECK), index=self.index,
                                  effects=self.effects, seed=seed,
                                  own_deck_ids=list(self.our_deck), **kwargs)

    # ------------------------------------------------------------------ #
    # The floor: search off == the current ship, exactly
    # ------------------------------------------------------------------ #

    def test_blind_agent_is_identical_to_its_prior_decision_by_decision(
            self) -> None:
        """The fallback floor, as an identity over a real game."""
        blind = self._agent(seed=4, enable_search=False)
        prior = CrustleAgent(seed=4, deck_path=str(OUR_DECK),
                             index=self.index, effects=self.effects,
                             variant="v3")
        from cg import game as cg_game

        obs_dict, start = cg_game.battle_start(list(self.our_deck),
                                               list(self.opp_deck))
        compared = 0
        try:
            self.assertIsNotNone(obs_dict, getattr(start, "errorType", None))
            opponent = RandomAgent(seed=5)
            for _ in range(400):
                current = obs_dict["current"]
                if current["result"] != -1:
                    break
                if current["yourIndex"] == 0:
                    mine = blind(copy.deepcopy(obs_dict))
                    theirs = prior(copy.deepcopy(obs_dict))
                    self.assertEqual(
                        mine, theirs,
                        f"blind agent diverged from its prior: "
                        f"{mine} vs {theirs}")
                    compared += 1
                    answer = mine
                else:
                    answer = opponent(obs_dict)
                obs_dict = cg_game.battle_select(answer)
        finally:
            cg_game.battle_finish()
        self.assertGreater(compared, 10,
                           "game ended before the floor was exercised")

    def test_unknown_archetype_never_searches(self) -> None:
        """No deck hypothesis -> no search, whatever else is true."""
        agent = self._agent(seed=6)
        # an estimator that can never become confident
        agent.estimator = OpponentDeckEstimator(index=self.index,
                                                min_containment=2.0)
        opponent = RandomAgent(seed=7)
        result, _turns = play_one_game((agent, opponent),
                                       list(self.our_deck),
                                       list(self.opp_deck))
        self.assertIn(result, (0, 1, 2))
        self.assertEqual(agent.stats.searched, 0,
                         f"searched without a deck estimate: "
                         f"{agent.stats.summary()}")
        self.assertEqual(agent.stats.exceptions, 0)

    # ------------------------------------------------------------------ #
    # Search actually runs, and stays legal while it does
    # ------------------------------------------------------------------ #

    def test_full_game_search_exercised_zero_exceptions(self) -> None:
        agent = self._agent(seed=8)
        opponent = RandomAgent(seed=9)
        result, turns = play_one_game((agent, opponent),
                                      list(self.our_deck),
                                      list(self.opp_deck))
        self.assertIn(result, (0, 1, 2))
        self.assertGreater(turns, 0)
        self.assertEqual(agent.stats.exceptions, 0,
                         f"exceptions during search: {agent.stats.summary()}")
        self.assertGreater(agent.stats.searched, 0,
                           f"search never exercised: {agent.stats.summary()}")
        self.assertGreater(agent.stats.rollouts, 0)

    def test_every_answer_is_a_legal_option_index(self) -> None:
        """Walk a real game and validate the contract on every answer."""
        agent = self._agent(seed=10)
        opponent = RandomAgent(seed=11)
        from cg import game as cg_game

        obs_dict, start = cg_game.battle_start(list(self.our_deck),
                                               list(self.opp_deck))
        checked = 0
        try:
            self.assertIsNotNone(obs_dict, getattr(start, "errorType", None))
            for _ in range(400):
                current = obs_dict["current"]
                if current["result"] != -1:
                    break
                if current["yourIndex"] == 0:
                    answer = agent(copy.deepcopy(obs_dict))
                    select = obs_dict.get("select") or {}
                    options = select.get("option") or []
                    self.assertTrue(answer, "empty answer")
                    self.assertEqual(len(answer), len(set(answer)),
                                     f"duplicate indices: {answer}")
                    for i in answer:
                        self.assertTrue(0 <= i < len(options),
                                        f"illegal index {i} of {len(options)}")
                    min_count = select.get("minCount")
                    max_count = select.get("maxCount")
                    if isinstance(min_count, int):
                        self.assertGreaterEqual(len(answer), min_count)
                    if isinstance(max_count, int):
                        self.assertLessEqual(len(answer), max_count)
                    checked += 1
                else:
                    answer = opponent(obs_dict)
                obs_dict = cg_game.battle_select(answer)
        finally:
            cg_game.battle_finish()
        self.assertGreater(checked, 10)
        self.assertEqual(agent.stats.exceptions, 0)

    # ------------------------------------------------------------------ #
    # Budget degradation, end to end through the agent
    # ------------------------------------------------------------------ #

    def test_an_exhausted_bank_stops_the_search_mid_game(self) -> None:
        """Report a drained bank and the agent must go quiet immediately."""
        agent = self._agent(seed=12)
        agent.guard = BudgetGuard(reserve_s=150.0)
        opponent = RandomAgent(seed=13)
        from cg import game as cg_game

        obs_dict, start = cg_game.battle_start(list(self.our_deck),
                                               list(self.opp_deck))
        try:
            self.assertIsNotNone(obs_dict, getattr(start, "errorType", None))
            for step in range(400):
                current = obs_dict["current"]
                if current["result"] != -1:
                    break
                if current["yourIndex"] == 0:
                    poisoned = copy.deepcopy(obs_dict)
                    # halfway through, claim the bank is nearly gone
                    poisoned["remainingOverageTime"] = (
                        600.0 if step < 40 else 5.0)
                    answer = agent(poisoned)
                    if step >= 40:
                        self.assertEqual(
                            agent.stats.fallback_reasons.get("budget", 0) > 0
                            or agent.stats.searched == 0, True,
                            "kept searching on an empty bank")
                else:
                    answer = opponent(obs_dict)
                obs_dict = cg_game.battle_select(answer)
        finally:
            cg_game.battle_finish()
        self.assertEqual(agent.stats.exceptions, 0)
        self.assertGreater(agent.guard.stats.skipped_reserve, 0,
                           "the reserve guard never engaged")

    # ------------------------------------------------------------------ #
    # Contract edges
    # ------------------------------------------------------------------ #

    def test_initial_selection_returns_the_deck_and_resets_state(self) -> None:
        agent = self._agent(seed=14)
        agent.estimator.observe({"select": {}, "current": {
            "turn": 5, "yourIndex": 0,
            "players": [{"active": []}, {"active": []}]}})
        deck = agent({"select": None, "logs": [], "current": None})
        self.assertEqual(deck, list(self.our_deck))
        self.assertEqual(len(deck), 60)
        self.assertEqual(agent.estimator.observed_counts(), {})

    def test_malformed_observations_do_not_raise(self) -> None:
        agent = self._agent(seed=15)
        for obs in (None, {}, {"select": None}, {"select": {}, "current": None},
                    {"select": {"option": []}, "current": {}}):
            with self.subTest(obs=obs):
                answer = agent(obs)  # type: ignore[arg-type]
                self.assertIsInstance(answer, list)


if __name__ == "__main__":
    unittest.main()
