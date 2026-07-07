"""NetworkAgent: the policy-net runtime brain (pure numpy, no torch).

Pipeline per selection: encode state + legal options (encoding.py),
normalize with the frozen stats (normalization.py), forward through the
exported weights (network_numpy.py), answer with the top-k legal indices
by policy logit.

k follows the cloned teacher's shape: bench-development prompts take
maxCount (an empty bench loses to any active KO — Sprint-3 lesson), every
other prompt takes max(minCount, 1) capped by maxCount.

None-safety ladder: missing weights file -> full HeuristicAgent fallback
(it ships in the same bundle); any per-call exception -> the same raw-dict
legal fallback the heuristic uses. The initial selection (select is None)
returns the 60-card deck.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from cg.api import Observation, SelectContext

from ..agent_heuristics.heuristic_agent import HeuristicAgent
from ..agent_heuristics.random_agent import read_deck_csv
from ..environment_wrapper.wrapper import EnvironmentWrapper
from ..ingestion.build_effect_model import EffectIndex
from ..ingestion.card_index import CardIndex
from . import encoding
from .encoding import MAX_OPTIONS, OptionEncoder, StateEncoder
from .network_numpy import NumpyPolicyValueNet
from .normalization import FeatureStats

_BENCH_CONTEXTS: Final[frozenset[int]] = frozenset({
    int(SelectContext.SETUP_BENCH_POKEMON),
    int(SelectContext.TO_BENCH),
    int(SelectContext.TO_FIELD),
})


class NetworkAgent:
    """Kaggle-contract agent over the numpy policy net."""

    __slots__ = ("_index", "_effects", "_wrapper", "_state_encoder",
                 "_option_encoder", "_stats", "_net", "_fallback",
                 "_deck_path", "last_scores", "last_value")

    def __init__(
        self,
        deck_path: str | None = None,
        index: CardIndex | None = None,
        effects: EffectIndex | None = None,
        weights_path=None,
    ) -> None:
        self._index = index if index is not None else CardIndex()
        self._effects = effects if effects is not None else EffectIndex()
        self._wrapper = EnvironmentWrapper(self._index)
        self._state_encoder = StateEncoder(self._index, self._effects)
        self._option_encoder = OptionEncoder(self._index, self._effects)
        self._stats = FeatureStats.load()
        self._net = (NumpyPolicyValueNet.load(weights_path)
                     if weights_path is not None else NumpyPolicyValueNet.load())
        self._fallback: HeuristicAgent | None = None
        if self._net is None:  # no weights: degrade to the heuristic, not to random
            self._fallback = HeuristicAgent(deck_path=deck_path,
                                            index=self._index,
                                            effects=self._effects)
        self._deck_path = deck_path
        self.last_scores: list[float] | None = None
        self.last_value: float | None = None

    # ------------------------------------------------------------------ #
    # Contract entry point
    # ------------------------------------------------------------------ #

    def __call__(self, obs_dict: dict) -> list[int]:
        self.last_scores = None
        self.last_value = None
        if self._fallback is not None:
            return self._fallback(obs_dict)
        try:
            obs = self._wrapper.parse(obs_dict)
            if obs.select is None:
                return read_deck_csv(self._deck_path)
            return self._decide(obs)
        except Exception:
            self.last_scores = None
            self.last_value = None
            return HeuristicAgent._safe_answer(obs_dict)

    # ------------------------------------------------------------------ #
    # Policy
    # ------------------------------------------------------------------ #

    def _decide(self, obs: Observation) -> list[int]:
        select = obs.select
        assert select is not None and self._net is not None
        n_options = len(select.option)
        if n_options == 0:
            return []
        if n_options == 1:
            return [] if select.maxCount == 0 and select.minCount == 0 else [0]

        state_vec = self._stats.normalize_state(self._state_encoder.encode(obs))
        option_matrix, mask = encoding.build_action_mask(
            obs, self._state_encoder, self._option_encoder)
        legal = self._stats.normalize_options(option_matrix[mask])
        logits, value = self._net.forward(state_vec, legal)
        self.last_scores = [float(x) for x in logits]
        self.last_value = value

        count = self._answer_count(int(select.context),
                                   select.minCount, select.maxCount,
                                   scored=len(logits))
        ranked = np.argsort(-logits)  # descending; ties break by lower index
        return [int(i) for i in ranked[:count]]

    @staticmethod
    def _answer_count(context: int, min_count: int, max_count: int,
                      scored: int) -> int:
        """How many indices to answer (bounded by the scored options)."""
        if context in _BENCH_CONTEXTS:
            count = max(min_count, max_count)  # develop the bench greedily
        else:
            count = max(min_count, min(1, max_count))
        return min(count, scored, MAX_OPTIONS)  # 0 -> answer [] (like teacher)


__all__ = ["NetworkAgent"]


# --------------------------------------------------------------------------- #
# Latency benchmark (dev): µs per __call__ over real decision points
# --------------------------------------------------------------------------- #


def _bench(n_games: int = 6, seed: int = 0) -> None:
    import copy
    import time

    from cg import game as cg_game

    from ..agent_heuristics.heuristic_agent import HeuristicAgent
    from ..agent_heuristics.random_agent import read_deck_csv

    index = CardIndex()
    effects = EffectIndex()
    agent = NetworkAgent(index=index, effects=effects)
    assert agent._fallback is None, "weights missing — export before benching"

    deck = read_deck_csv()
    observations: list[dict] = []
    for game_index in range(n_games):
        opponent = HeuristicAgent(seed=seed + game_index, index=index,
                                  effects=effects)
        agents = (agent, opponent) if game_index % 2 == 0 else (opponent, agent)
        obs_dict, _ = cg_game.battle_start(list(deck), list(deck))
        try:
            for _ in range(20_000):
                state = obs_dict["current"]
                if state["result"] != -1:
                    break
                acting = state["yourIndex"]
                if agents[acting] is agent:
                    observations.append(copy.deepcopy(obs_dict))
                obs_dict = cg_game.battle_select(agents[acting](obs_dict))
        finally:
            cg_game.battle_finish()

    times_us: list[float] = []
    for obs_dict in observations:
        t0 = time.perf_counter()
        agent(obs_dict)
        times_us.append((time.perf_counter() - t0) * 1e6)
    times_us.sort()
    n = len(times_us)
    print(f"decision points:  {n} (from {n_games} games vs heuristic)")
    print(f"latency mean:     {sum(times_us) / n:8.1f} us/move")
    print(f"latency p50:      {times_us[n // 2]:8.1f} us/move")
    print(f"latency p99:      {times_us[min(n - 1, int(n * 0.99))]:8.1f} us/move")
    print(f"latency max:      {times_us[-1]:8.1f} us/move")


if __name__ == "__main__":
    _bench()
