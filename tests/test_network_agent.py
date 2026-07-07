"""Contract tests for NetworkAgent (numpy policy-net runtime brain).

No torch required: uses the exported models/policy_value.npz when present
and asserts graceful degradation when it is missing.

Run from the repo root:  python -m unittest tests.test_network_agent
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex
from src.rl_models.network_agent import NetworkAgent
from src.rl_models.network_numpy import WEIGHTS_PATH


def _fake_obs(min_count: int = 1, max_count: int = 1, n_options: int = 3,
              context: int = 0) -> dict:
    return {
        "select": {
            "type": 0, "context": context, "minCount": min_count,
            "maxCount": max_count, "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": [{"type": 14} for _ in range(n_options)],
            "deck": None, "contextCard": None, "effect": None,
        },
        "logs": [],
        "current": None,
    }


class TestNetworkAgentContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.agent = NetworkAgent(index=cls.index, effects=cls.effects)

    def test_initial_selection_returns_deck(self) -> None:
        deck = self.agent({"select": None, "logs": [], "current": None})
        self.assertEqual(len(deck), 60)
        self.assertTrue(all(isinstance(card_id, int) for card_id in deck))

    def test_single_select_returns_one_valid_index(self) -> None:
        answer = self.agent(_fake_obs(n_options=5))
        self.assertEqual(len(answer), 1)
        self.assertTrue(0 <= answer[0] < 5)

    def test_multi_select_respects_counts_and_uniqueness(self) -> None:
        answer = self.agent(_fake_obs(min_count=2, max_count=3, n_options=6))
        self.assertGreaterEqual(len(answer), 2)
        self.assertLessEqual(len(answer), 3)
        self.assertEqual(len(answer), len(set(answer)))
        self.assertTrue(all(0 <= i < 6 for i in answer))

    def test_zero_max_count_returns_empty(self) -> None:
        self.assertEqual(self.agent(_fake_obs(min_count=0, max_count=0)), [])

    def test_garbage_observation_still_answers_legally(self) -> None:
        answer = self.agent({"select": {"minCount": 1, "maxCount": 1,
                                        "option": [{}, {}]},
                             "logs": None, "current": {"broken": True}})
        self.assertIsInstance(answer, list)
        self.assertTrue(all(isinstance(i, int) and 0 <= i < 2 for i in answer))

    def test_weights_present_and_net_active(self) -> None:
        # 5A ships the trained npz; the agent must actually use the net.
        self.assertTrue(WEIGHTS_PATH.exists(),
                        "models/policy_value.npz missing — run "
                        "python -m src.rl_models.network_numpy --export")
        self.assertIsNone(self.agent._fallback)
        self.agent(_fake_obs(n_options=4))
        self.assertIsNotNone(self.agent.last_scores)
        self.assertEqual(len(self.agent.last_scores), 4)
        self.assertIsNotNone(self.agent.last_value)

    def test_missing_weights_falls_back_to_heuristic(self) -> None:
        agent = NetworkAgent(index=self.index, effects=self.effects,
                             weights_path=Path("does/not/exist.npz"))
        self.assertIsNotNone(agent._fallback)
        answer = agent(_fake_obs(n_options=3))
        self.assertEqual(len(answer), 1)
        self.assertTrue(0 <= answer[0] < 3)
        deck = agent({"select": None, "logs": [], "current": None})
        self.assertEqual(len(deck), 60)


if __name__ == "__main__":
    unittest.main()
