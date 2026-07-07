"""Tests for the RL state/option encoders.

Run from the repo root:  python -m unittest tests.test_encoding
"""

from __future__ import annotations

import unittest

import numpy as np

from cg import game
from cg.api import to_observation_class

from src.agent_heuristics.heuristic_agent import HeuristicAgent
from src.agent_heuristics.random_agent import RandomAgent, read_deck_csv
from src.environment_wrapper.wrapper import EnvironmentWrapper
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex
from src.rl_models.encoding import (
    ENCODING_DIM,
    MAX_OPTIONS,
    OPTION_DIM,
    OptionEncoder,
    StateEncoder,
    build_action_mask,
)


def _fake_pokemon(card_id: int) -> dict:
    return {"id": card_id, "serial": 1, "hp": 100, "maxHp": 120,
            "appearThisTurn": False, "energies": [1, 0], "energyCards": [],
            "tools": [], "preEvolution": []}


def _fake_player(card_id: int) -> dict:
    return {"active": [_fake_pokemon(card_id)], "bench": [_fake_pokemon(card_id)],
            "benchMax": 5, "deckCount": 40, "discard": [], "prize": [None] * 6,
            "handCount": 5, "hand": [], "poisoned": False, "burned": False,
            "asleep": False, "paralyzed": False, "confused": False}


def _fake_obs(card_id: int, attack_id: int) -> dict:
    return {
        "select": {
            "type": 0, "context": 0, "minCount": 1, "maxCount": 1,
            "remainDamageCounter": 0, "remainEnergyCost": 0,
            "option": [{"type": 13, "attackId": attack_id},
                       {"type": 3, "area": 2, "index": 0, "playerIndex": 0},
                       {"type": 14}],
            "deck": None, "contextCard": None, "effect": None,
        },
        "logs": [],
        "current": {
            "turn": 3, "turnActionCount": 1, "yourIndex": 0, "firstPlayer": 0,
            "supporterPlayed": False, "stadiumPlayed": False,
            "energyAttached": False, "retreated": False, "result": -1,
            "stadium": [], "looking": None,
            "players": [_fake_player(card_id), _fake_player(card_id)],
        },
    }


class TestEncoders(unittest.TestCase):
    index: CardIndex
    effects: EffectIndex
    state_encoder: StateEncoder
    option_encoder: OptionEncoder
    wrapper: EnvironmentWrapper

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.state_encoder = StateEncoder(cls.index, cls.effects)
        cls.option_encoder = OptionEncoder(cls.index, cls.effects)
        cls.wrapper = EnvironmentWrapper(cls.index)

    # ---- shapes & determinism ---- #

    def test_shapes(self) -> None:
        obs = to_observation_class(_fake_obs(card_id=41, attack_id=35))
        state = self.state_encoder.encode(obs)
        self.assertEqual(state.shape, (ENCODING_DIM,))
        self.assertEqual(state.dtype, np.float32)
        option = self.option_encoder.encode(obs, obs.select.option[0])
        self.assertEqual(option.shape, (OPTION_DIM,))
        matrix, mask = build_action_mask(obs, self.state_encoder, self.option_encoder)
        self.assertEqual(matrix.shape, (MAX_OPTIONS, OPTION_DIM))
        self.assertEqual(mask.shape, (MAX_OPTIONS,))
        self.assertEqual(mask.sum(), 3)
        self.assertTrue(mask[:3].all())

    def test_determinism(self) -> None:
        obs_dict = _fake_obs(card_id=41, attack_id=35)
        a = self.state_encoder.encode(to_observation_class(obs_dict))
        b = self.state_encoder.encode(to_observation_class(obs_dict))
        np.testing.assert_array_equal(a, b)
        obs = to_observation_class(obs_dict)
        oa = self.option_encoder.encode(obs, obs.select.option[0])
        ob = self.option_encoder.encode(obs, obs.select.option[0])
        np.testing.assert_array_equal(oa, ob)

    def test_known_ids_populate_features(self) -> None:
        obs = to_observation_class(_fake_obs(card_id=41, attack_id=35))
        state = self.state_encoder.encode(obs)
        self.assertGreater(float(np.abs(state).sum()), 0.0)
        option = self.option_encoder.encode(obs, obs.select.option[0])
        self.assertGreater(float(np.abs(option).sum()), 0.0)

    # ---- fail-safety ---- #

    def test_unknown_ids_encode_as_zeros_without_raising(self) -> None:
        obs = to_observation_class(_fake_obs(card_id=60_000, attack_id=60_000))
        state = self.state_encoder.encode(obs)
        self.assertFalse(np.isnan(state).any())
        # pokemon slots still mark presence/hp, but no card-derived features
        option = self.option_encoder.encode(obs, obs.select.option[0])
        self.assertFalse(np.isnan(option).any())
        matrix, mask = build_action_mask(obs, self.state_encoder, self.option_encoder)
        self.assertEqual(int(mask.sum()), 3)

    def test_initial_selection_encodes_to_zeros(self) -> None:
        obs = to_observation_class({"select": None, "logs": [], "current": None})
        state = self.state_encoder.encode(obs)
        self.assertEqual(float(np.abs(state).sum()), 0.0)
        matrix, mask = build_action_mask(obs, self.state_encoder, self.option_encoder)
        self.assertEqual(int(mask.sum()), 0)

    # ---- live self-play sweep ---- #

    def test_selfplay_sweep_no_nan_no_exception(self) -> None:
        deck = read_deck_csv()
        agents = (HeuristicAgent(seed=11, index=self.index, effects=self.effects),
                  RandomAgent(seed=12))
        n_encoded = 0
        global_min, global_max = float("inf"), float("-inf")
        for _ in range(4):
            obs_dict, _ = game.battle_start(list(deck), list(deck))
            try:
                for _ in range(2000):
                    if obs_dict["current"]["result"] != -1:
                        break
                    obs = self.wrapper.parse(obs_dict)
                    state = self.state_encoder.encode(obs)
                    matrix, mask = build_action_mask(obs, self.state_encoder,
                                                     self.option_encoder)
                    for arr in (state, matrix):
                        self.assertFalse(np.isnan(arr).any(), "NaN in encoding")
                        self.assertFalse(np.isinf(arr).any(), "inf in encoding")
                    self.assertEqual(int(mask.sum()),
                                     min(len(obs.select.option), MAX_OPTIONS))
                    global_min = min(global_min, float(state.min()), float(matrix.min()))
                    global_max = max(global_max, float(state.max()), float(matrix.max()))
                    n_encoded += 1
                    acting = obs_dict["current"]["yourIndex"]
                    obs_dict = game.battle_select(agents[acting](obs_dict))
            finally:
                game.battle_finish()
        print(f"\n[sweep] encoded {n_encoded} observations, "
              f"feature range [{global_min:.3f}, {global_max:.3f}]")
        self.assertGreater(n_encoded, 50)


if __name__ == "__main__":
    unittest.main()
