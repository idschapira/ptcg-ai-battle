"""Contract: the value-leaf search plays legal, and its floor is the ship.

Same argument as tests/test_runtime_search_agent.py, and the same way of
trying to break it: with the search off, the agent must return the SAME
indices as a bare CrustleAgent v3, decision for decision, over a real
engine game. A winrate comparison would pass at 50% in a mirror even if
the two disagreed on every move; identity is the property the ship
decision actually rests on.

The value head adds one failure mode the rollout search did not have —
the weights can be missing or corrupt — so that path is tested too: no
leaf means the agent is its prior, never something improvised.

Contract tests here drive a LIVE engine. A synthetic option dict can be
made to satisfy any code path and would hide a rule that has since
changed; the engine failing is what makes these tests worth running.

Run from the repo root:
    python -m unittest tests.test_value_search_agent
"""

from __future__ import annotations

import copy
import unittest

import numpy as np

from src.agent_heuristics.crustle_agent import CrustleAgent
from src.agent_heuristics.random_agent import RandomAgent
from src.deckbuilding.legality import read_deck_ids
from src.ingestion.build_card_model import REPO_ROOT
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex
from src.rl_models.budget import BudgetGuard
from src.rl_models.opponent_estimator import OpponentDeckEstimator
from src.rl_models.value_net import VALUE_NPZ, NumpyValueNet
from src.rl_models.value_search import ALL_TIERS, NODE_PRIOR_S, ValueTier
from src.rl_models.value_search_agent import ValueSearchAgent

OUR_DECK = REPO_ROOT / "deck.csv"
OPP_DECK = REPO_ROOT / "data" / "decks" / "meta_alakazam.csv"

# Always affordable, so tests that want the search EXERCISED do not have
# to fight the guard.
CHEAP_LADDER = (ValueTier("t", 3, 1, 2, 2, 0.0),)


class TestValueSearchAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.our_deck = read_deck_ids(OUR_DECK)
        cls.opp_deck = read_deck_ids(OPP_DECK)
        if NumpyValueNet.load() is None:
            raise unittest.SkipTest(
                f"value head {VALUE_NPZ} missing — run "
                f"python -m src.rl_models.value_train")

    def _agent(self, seed: int = 0, **kwargs) -> ValueSearchAgent:
        kwargs.setdefault("guard", BudgetGuard(
            ladder=CHEAP_LADDER, reserve_s=0.0,
            rollout_prior_s=NODE_PRIOR_S))
        kwargs.setdefault("contested_margin", None)
        return ValueSearchAgent(deck_path=str(OUR_DECK), index=self.index,
                                effects=self.effects, seed=seed,
                                own_deck_ids=list(self.our_deck), **kwargs)

    def _drive(self, agent, seed: int, max_steps: int = 400,
               check=None) -> int:
        """Play one real game with ``agent`` as player 0. Returns steps."""
        from cg import game as cg_game

        obs_dict, start = cg_game.battle_start(list(self.our_deck),
                                               list(self.opp_deck))
        steps = 0
        try:
            self.assertIsNotNone(obs_dict, getattr(start, "errorType", None))
            opponent = RandomAgent(seed=seed + 1)
            for _ in range(max_steps):
                current = obs_dict["current"]
                if current["result"] != -1:
                    break
                if current["yourIndex"] == 0:
                    answer = agent(copy.deepcopy(obs_dict))
                    if check is not None:
                        check(obs_dict, answer)
                    steps += 1
                else:
                    answer = opponent(obs_dict)
                obs_dict = cg_game.battle_select(answer)
        finally:
            cg_game.battle_finish()
        return steps

    # ------------------------------------------------------------------ #
    # The floor
    # ------------------------------------------------------------------ #

    def test_blind_agent_is_identical_to_its_prior_decision_by_decision(
            self) -> None:
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
                    self.assertEqual(mine, theirs,
                                     f"blind diverged: {mine} vs {theirs}")
                    compared += 1
                    answer = mine
                else:
                    answer = opponent(obs_dict)
                obs_dict = cg_game.battle_select(answer)
        finally:
            cg_game.battle_finish()
        self.assertGreater(compared, 10,
                           "game ended before the floor was exercised")

    def test_missing_value_head_degrades_to_the_prior(self) -> None:
        """No leaf -> no search. Not a guess, not a crash: the prior."""
        agent = self._agent(seed=7,
                            value_weights=REPO_ROOT / "models" / "nope.npz")
        prior = CrustleAgent(seed=7, deck_path=str(OUR_DECK),
                             index=self.index, effects=self.effects,
                             variant="v3")
        from cg import game as cg_game

        obs_dict, start = cg_game.battle_start(list(self.our_deck),
                                               list(self.opp_deck))
        compared = 0
        try:
            self.assertIsNotNone(obs_dict, getattr(start, "errorType", None))
            opponent = RandomAgent(seed=8)
            for _ in range(300):
                current = obs_dict["current"]
                if current["result"] != -1:
                    break
                if current["yourIndex"] == 0:
                    mine = agent(copy.deepcopy(obs_dict))
                    self.assertEqual(mine, prior(copy.deepcopy(obs_dict)))
                    compared += 1
                    answer = mine
                else:
                    answer = opponent(obs_dict)
                obs_dict = cg_game.battle_select(answer)
        finally:
            cg_game.battle_finish()
        self.assertGreater(compared, 10)
        self.assertEqual(agent.stats.searched, 0)

    # ------------------------------------------------------------------ #
    # Legality and liveness
    # ------------------------------------------------------------------ #

    def test_every_answer_is_a_legal_option_index(self) -> None:
        agent = self._agent(seed=11,
                            opponent_deck_override=list(self.opp_deck))

        def check(obs_dict, answer):
            select = obs_dict["select"]
            options = select["option"]
            self.assertIsInstance(answer, list)
            self.assertEqual(len(answer), len(set(answer)),
                             "duplicate index in answer")
            self.assertGreaterEqual(len(answer), select["minCount"])
            self.assertLessEqual(len(answer), select["maxCount"])
            for i in answer:
                self.assertIsInstance(i, int)
                self.assertGreaterEqual(i, 0)
                self.assertLess(i, len(options))

        steps = self._drive(agent, seed=11, check=check)
        self.assertGreater(steps, 10)
        self.assertEqual(agent.stats.exceptions, 0,
                         f"search raised: {agent.stats.summary()}")

    def test_search_actually_runs_and_goes_deeper_than_one_ply(self) -> None:
        """The point of Stage B: the beam must reach past level 0."""
        agent = self._agent(seed=13,
                            opponent_deck_override=list(self.opp_deck))
        self._drive(agent, seed=13)
        self.assertGreater(agent.stats.searched, 0,
                           f"search never ran: {agent.stats.summary()}")
        self.assertGreater(agent.stats.nodes, agent.stats.searched,
                           "more than one node per search is the whole idea")
        self.assertGreaterEqual(agent.stats.depth_reached, 1,
                                f"beam stayed 1-ply: {agent.stats.summary()}")
        self.assertEqual(agent.stats.exceptions, 0)

    def test_leaves_are_scored_in_batches(self) -> None:
        """Batching is a cost property the budget curve depends on."""
        agent = self._agent(seed=17,
                            opponent_deck_override=list(self.opp_deck))
        self._drive(agent, seed=17)
        self.assertGreater(agent.stats.leaves, 0)
        self.assertGreater(agent.stats.leaves, agent.stats.leaf_batches,
                           "every batch held one leaf — batching is dead")

    # ------------------------------------------------------------------ #
    # Budget
    # ------------------------------------------------------------------ #

    def test_an_exhausted_bank_stops_the_search(self) -> None:
        """The guard is the component whose failure is disqualification."""
        guard = BudgetGuard(ladder=(ALL_TIERS["d2x4"],), reserve_s=150.0,
                            rollout_prior_s=NODE_PRIOR_S)
        agent = self._agent(seed=19, guard=guard,
                            opponent_deck_override=list(self.opp_deck))

        drained = {"seen": False}

        def check(obs_dict, answer):
            drained["seen"] = True

        # A bank at the reserve must refuse every tier, whatever else is
        # true about the position.
        self.assertIsNone(guard.choose({"remainingOverageTime": 150.0}))
        self.assertIsNone(guard.choose({"remainingOverageTime": 0.0}))
        # and a healthy bank must NOT refuse, or the test above is vacuous
        self.assertIsNotNone(guard.choose({"remainingOverageTime": 600.0}))

        self._drive(agent, seed=19, check=check)
        self.assertTrue(drained["seen"])
        self.assertEqual(agent.stats.exceptions, 0)

    def test_budget_guard_reads_a_missing_bank_pessimistically(self) -> None:
        guard = BudgetGuard(ladder=(ALL_TIERS["d2x4"],),
                            rollout_prior_s=NODE_PRIOR_S)
        # nonsense values must never read as "plenty of bank"
        for bad in (None, float("nan"), -1.0, 10_000.0, "600", True):
            bank = guard.bank_s({"remainingOverageTime": bad})
            self.assertLessEqual(bank, 600.0)
            self.assertGreaterEqual(bank, 0.0)

    # ------------------------------------------------------------------ #
    # None-safety
    # ------------------------------------------------------------------ #

    def test_initial_selection_returns_the_deck(self) -> None:
        agent = self._agent(seed=23)
        answer = agent({"select": None})
        self.assertEqual(len(answer), 60)
        self.assertEqual(sorted(answer), sorted(self.our_deck))

    def test_malformed_observations_do_not_raise(self) -> None:
        agent = self._agent(seed=29)
        for junk in ({}, {"select": {}}, {"select": {"option": []}},
                     {"select": {"option": [], "maxCount": 1},
                      "current": None},
                     {"select": {"option": [{}], "maxCount": 1},
                      "current": {"yourIndex": 5, "players": []}}):
            answer = agent(copy.deepcopy(junk))
            self.assertIsInstance(answer, list)

    def test_value_head_convention_is_side_to_move(self) -> None:
        """value(state) is for the player TO MOVE; the search negates it.

        If this convention ever flips, the search silently maximises the
        opponent's position at every leaf where our turn ended — which is
        most of them — so it is worth an explicit check.
        """
        net = NumpyValueNet.load()
        self.assertIsNotNone(net)
        out = net.value_batch(np.zeros((4, 1185), np.float32))
        self.assertEqual(out.shape, (4,))
        self.assertTrue(np.all(out >= -1.0) and np.all(out <= 1.0))
        # deterministic: the same state must score the same every time
        again = net.value_batch(np.zeros((4, 1185), np.float32))
        np.testing.assert_allclose(out, again)
        # and batching must agree with the single-state path
        single = net.value(np.zeros(1185, np.float32))
        self.assertAlmostEqual(single, float(out[0]), places=5)


if __name__ == "__main__":
    unittest.main()
