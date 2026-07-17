"""Contrato do SearchAgent (Camada 2): joga legal, degrada legal.

Smoke real (engine vivo): um jogo completo com busca mínima (2x2)
precisa terminar sem exceção — nem do harness nem dos rollouts — e a
busca precisa ter sido de fato exercida (searched > 0). O fallback é
pinado: sem search_begin_input a resposta vem do prior e é legal.

Rodar da raiz do repo:  python -m unittest tests.test_search_agent
"""

from __future__ import annotations

import copy
import unittest

from src.agent_heuristics.heuristic_agent import HeuristicAgent
from src.agent_heuristics.random_agent import RandomAgent
from src.deckbuilding.legality import read_deck_ids
from src.environment_wrapper.selfplay import play_one_game
from src.ingestion.build_card_model import REPO_ROOT
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex
from src.rl_models.search_agent import SearchAgent, SearchStats

DECK_PATH = REPO_ROOT / "data" / "decks" / "meta_alakazam.csv"


def _make_search_agent(index: CardIndex, effects: EffectIndex,
                       deck: list[int], stats: SearchStats,
                       seed: int = 0) -> SearchAgent:
    """Busca mínima (2 candidatos x 2 determinizações), prior heurístico."""
    prior = HeuristicAgent(seed=seed, index=index, effects=effects)
    rollout = lambda s: HeuristicAgent(seed=s, index=index,  # noqa: E731
                                       effects=effects)
    return SearchAgent(prior, deck, deck, rollout, rollout,
                       n_candidates=2, n_determinizations=2,
                       seed=seed, stats=stats)


class TestSearchAgentContract(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(DECK_PATH)

    def test_full_game_zero_exceptions_and_search_exercised(self) -> None:
        stats = SearchStats()
        agent = _make_search_agent(self.index, self.effects, self.deck,
                                   stats, seed=7)
        opponent = RandomAgent(seed=8)
        result, turns = play_one_game((agent, opponent),
                                      list(self.deck), list(self.deck))
        self.assertIn(result, (0, 1, 2))
        self.assertGreater(turns, 0)
        self.assertEqual(stats.exceptions, 0,
                         f"exceções na busca: {stats.summary()}")
        self.assertGreater(stats.searched, 0,
                           f"busca nunca exercida: {stats.summary()}")
        self.assertGreater(stats.rollouts, 0)

    def test_missing_search_input_falls_back_to_legal_prior(self) -> None:
        from cg import game as cg_game

        stats = SearchStats()
        agent = _make_search_agent(self.index, self.effects, self.deck,
                                   stats, seed=3)
        obs_dict, start = cg_game.battle_start(list(self.deck),
                                               list(self.deck))
        try:
            self.assertIsNotNone(obs_dict, getattr(start, "errorType", None))
            stripped = copy.deepcopy(obs_dict)
            stripped.pop("search_begin_input", None)
            answer = agent(stripped)
            select = stripped.get("select") or {}
            options = select.get("option") or []
            self.assertTrue(answer, "resposta vazia")
            self.assertEqual(len(answer), len(set(answer)),
                             "índices duplicados")
            for i in answer:
                self.assertTrue(0 <= i < len(options),
                                f"índice ilegal {i} de {len(options)}")
            self.assertEqual(stats.fallback_reasons["no-search-input"], 1)
            self.assertEqual(stats.searched, 0)
            self.assertEqual(stats.exceptions, 0)
        finally:
            cg_game.battle_finish()

    def test_initial_deck_selection_returns_own_deck(self) -> None:
        stats = SearchStats()
        agent = _make_search_agent(self.index, self.effects, self.deck,
                                   stats)
        self.assertEqual(agent({"select": None}), list(self.deck))


if __name__ == "__main__":
    unittest.main()
