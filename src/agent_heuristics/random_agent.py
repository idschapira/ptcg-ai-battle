"""Uniform-random agent satisfying the competition contract.

agent(obs_dict) -> list[int] where each element is a distinct index into
obs.select.option and the list length is within [minCount, maxCount].
On the initial selection (obs.select is None) it returns the 60-card deck
read from deck.csv.
"""

from __future__ import annotations

import os
import random
from typing import Final

from cg.api import Observation, to_observation_class

DECK_SIZE: Final[int] = 60
KAGGLE_DECK_PATH: Final[str] = "/kaggle_simulations/agent/deck.csv"


def read_deck_csv(path: str | None = None) -> list[int]:
    """Read the 60 card ids, trying local deck.csv then the Kaggle agent dir."""
    file_path = path if path is not None else "deck.csv"
    if not os.path.exists(file_path):
        file_path = KAGGLE_DECK_PATH
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.read().split("\n")
    return [int(lines[i]) for i in range(DECK_SIZE)]


class RandomAgent:
    """Picks a uniform-random legal answer for every selection."""

    __slots__ = ("_rng", "_deck_path")

    def __init__(self, seed: int | None = None, deck_path: str | None = None) -> None:
        self._rng = random.Random(seed)
        self._deck_path = deck_path

    def __call__(self, obs_dict: dict) -> list[int]:
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            return read_deck_csv(self._deck_path)
        return self.act(obs)

    def act(self, obs: Observation) -> list[int]:
        """Choose k in [minCount, maxCount] distinct option indices at random."""
        select = obs.select
        assert select is not None
        n_options = len(select.option)
        count = self._rng.randint(select.minCount, min(select.maxCount, n_options))
        return self._rng.sample(range(n_options), count)
