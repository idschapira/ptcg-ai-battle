"""The tempo fix must change the generic pilot and NOTHING else.

Two obligations, both checked against the real engine rather than against
synthetic options (a hand-built Option satisfies any code path and hides
rules that have already died):

  1. with tempo=True the pilot actually promotes Rare Candy out of the
     trainer band on a REAL Rare Candy option the engine offered, and
     with tempo=False it does not — otherwise the flag is decorative;
  2. CrustleAgent, the shipped pilot, is byte-identical decision-by-
     decision to its pre-change self. It subclasses HeuristicAgent, so a
     scorer change is exactly the kind of edit that leaks into the ship.

Obligation 2 is enforced the way the league retrofit did it: replay a
full game and compare every answer, not just the outcome.

Run from the repo root:  python -m unittest tests.test_tempo_equivalence
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Final

from cg.api import OptionType

from src.agent_heuristics.crustle_agent import CrustleAgent
from src.agent_heuristics.heuristic_agent import (EVOLUTION_ACCELERATORS,
                                                  _EVOLVE_BAND,
                                                  _TRAINER_BAND,
                                                  HeuristicAgent)
from src.deckbuilding.legality import read_deck_ids
from src.environment_wrapper.selfplay import play_one_game
from src.environment_wrapper.wrapper import EnvironmentWrapper
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex

RARE_CANDY: Final[int] = 1079
ALAKAZAM_DECK: Final[Path] = Path("data/decks/meta_alakazam.csv")
OUR_DECK: Final[Path] = Path("deck.csv")
MAX_GAMES: Final[int] = 40


class _ScoreCapture:
    """Plays a deck and captures the score of a real Rare Candy option."""

    def __init__(self, index: CardIndex, effects: EffectIndex,
                 tempo: bool, seed: int) -> None:
        self._agent = HeuristicAgent(index=index, effects=effects,
                                     seed=seed, tempo=tempo)
        self._wrapper = EnvironmentWrapper(index)
        self.candy_scores: list[float] = []
        self.other_trainer_scores: list[float] = []

    def __call__(self, obs_dict: dict) -> list[int]:
        answer = self._agent(obs_dict)
        scores = self._agent.last_scores
        if not scores:
            return answer
        try:
            obs = self._wrapper.parse(obs_dict)
            select = obs.select
            if select is None or len(scores) != len(select.option):
                return answer
            for i, option in enumerate(select.option):
                if option.type != OptionType.PLAY:
                    continue
                card_id = self._wrapper.resolve_card_id(obs, option)
                card = self._wrapper._index.get_card(card_id) \
                    if card_id is not None else None
                if card_id == RARE_CANDY:
                    self.candy_scores.append(scores[i])
                elif card is not None and card.hp is None:
                    self.other_trainer_scores.append(scores[i])
        except Exception:
            pass
        return answer


class TestTempoFlagIsRealAndScoped(unittest.TestCase):
    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(ALAKAZAM_DECK)

    def _collect(self, tempo: bool) -> _ScoreCapture:
        """Drive real games until the engine offers a Rare Candy."""
        for game_index in range(MAX_GAMES):
            probe = _ScoreCapture(self.index, self.effects, tempo,
                                  seed=game_index)
            other = HeuristicAgent(index=self.index, effects=self.effects,
                                   seed=1000 + game_index, tempo=tempo)
            play_one_game((probe, other), list(self.deck), list(self.deck))
            if probe.candy_scores:
                return probe
        self.fail("o motor nunca ofereceu Rare Candy — o teste seria vácuo")

    def test_the_accelerator_set_is_not_empty(self) -> None:
        self.assertIn(RARE_CANDY, EVOLUTION_ACCELERATORS)

    def test_tempo_off_leaves_rare_candy_in_the_trainer_band(self) -> None:
        probe = self._collect(tempo=False)
        for score in probe.candy_scores:
            self.assertAlmostEqual(score, _TRAINER_BAND, places=6)

    def test_tempo_on_promotes_rare_candy_above_evolve(self) -> None:
        probe = self._collect(tempo=True)
        for score in probe.candy_scores:
            self.assertGreaterEqual(score, _EVOLVE_BAND)
        # and it must now beat the trainers it used to tie with
        if probe.other_trainer_scores:
            self.assertGreater(min(probe.candy_scores),
                               max(probe.other_trainer_scores))


class TestShippedCrustleIsUnchanged(unittest.TestCase):
    """CrustleAgent must be bit-identical: it subclasses the pilot."""

    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(OUR_DECK)

    def test_crustle_defaults_to_tempo_off(self) -> None:
        agent = CrustleAgent(index=self.index, effects=self.effects,
                             variant="v3")
        self.assertFalse(agent._tempo,
                         "o ship herdaria a mudança de tempo — o flag tem "
                         "de continuar desligado para o CrustleAgent")

    def test_tempo_flag_is_inert_over_real_crustle_decisions(self) -> None:
        """Same observation stream, flag on vs off: identical every time.

        Replaying "the same game" twice is not available here — the engine
        exposes no seed and shuffles from std::random_device, so two runs
        diverge for reasons that have nothing to do with this change. The
        comparison that IS available, and is the one that matters, is
        lockstep on a single stream: one CrustleAgent drives, and at every
        decision a shadow pilot with tempo=True is handed the very same
        observation and must answer identically.
        """
        divergences: list[str] = []
        compared = 0

        driver = CrustleAgent(index=self.index, effects=self.effects,
                              variant="v3", seed=7)
        shadow = CrustleAgent(index=self.index, effects=self.effects,
                              variant="v3", seed=7)
        shadow._tempo = True          # the ONLY difference

        def lockstep(obs_dict: dict) -> list[int]:
            nonlocal compared
            answer = driver(obs_dict)
            mirrored = shadow(dict(obs_dict))
            compared += 1
            if answer != mirrored:
                turn = (obs_dict.get("current") or {}).get("turn")
                divergences.append(f"t{turn}: {answer} != {mirrored}")
            return answer

        opponent = CrustleAgent(index=self.index, effects=self.effects,
                                variant="v3", seed=8)
        play_one_game((lockstep, opponent), list(self.deck), list(self.deck))

        self.assertGreater(compared, 20,
                           "quase nenhuma decisão comparada — teste vácuo")
        self.assertEqual(divergences, [],
                         f"o flag de tempo mudou o piloto do ship em "
                         f"{len(divergences)}/{compared} decisões")

    def test_our_deck_holds_no_accelerator(self) -> None:
        """Second layer: the rule cannot fire on our list even if enabled."""
        self.assertFalse(
            set(self.deck) & EVOLUTION_ACCELERATORS,
            "deck.csv passou a conter um acelerador — a garantia de que o "
            "flag não pode afetar o ship deixa de ser estrutural")


if __name__ == "__main__":
    unittest.main()
