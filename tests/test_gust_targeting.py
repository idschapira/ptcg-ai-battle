"""Gust targeting: drag the UNCOVERED body, and leave the ship alone.

Powerful Hand only reaches the Active Spot and is nullified outright by
Mist/Rock (damage counters are an EFFECT — verified against the engine in
tests/test_effect_prevention_contract.py). So a real opponent's way past
our prevention is not a bigger attack, it is a different victim: gust the
body with no cover into the Active Spot, then hit it.

Measured on the ladder corpus against internal games (see
src/analysis/gust_telemetry.py): when the engine offered BOTH a covered
and a bare body, real opponents took the bare one 6 of 6 times and the
generic pilot took it 0 of 28 — it drags Great Tusk, the {F} body holding
the Rock Fighting Energy, because _own_pokemon_score scores an enemy
body with the rule meant for our own promotions (highest HP + damage).

Lines held here:
  * the cover table is read from the ENGINE'S text, not memorized, and
    it reproduces the verified matrix (Mist any host, Rock {F} only);
  * on REAL engine options the flag flips the choice to the bare body;
  * CrustleAgent — the ship — does not move, lockstep, flag forced on.

Run from the repo root:  python -m unittest tests.test_gust_targeting
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Final

from cg.api import SelectContext

from src.agent_heuristics.crustle_agent import CrustleAgent
from src.agent_heuristics.heuristic_agent import (HeuristicAgent,
                                                  prevention_energies)
from src.deckbuilding.legality import read_deck_ids
from src.environment_wrapper.selfplay import play_one_game
from src.environment_wrapper.wrapper import EnvironmentWrapper
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex

MIST: Final[int] = 11
ROCK: Final[int] = 20
FIGHTING: Final[int] = 6
ALAKAZAM_DECK: Final[Path] = Path("data/decks/meta_alakazam.csv")
OUR_DECK: Final[Path] = Path("deck.csv")


class TestCoverTableFromEngineText(unittest.TestCase):
    """The prevention matrix, re-derived from the cards themselves."""

    def test_mist_covers_any_host_rock_only_fighting(self) -> None:
        table = prevention_energies()
        self.assertIn(MIST, table)
        self.assertIn(ROCK, table)
        self.assertIsNone(table[MIST], "Mist cobre qualquer host")
        self.assertEqual(table[ROCK], FIGHTING,
                         "Rock Fighting so cobre host {F}")

    def test_nothing_else_in_the_pool_is_attached_cover(self) -> None:
        """If a third printing appears the rule picks it up — but today
        it is exactly these two, and a silent third entry would mean the
        clause is matching something it should not (Battle Cage and
        Acerola's Mischief both say 'prevent' and are NOT attached cover).
        """
        self.assertEqual(set(prevention_energies()), {MIST, ROCK})


class _Tap:
    def __init__(self, inner, wrapper: EnvironmentWrapper, sink) -> None:
        self._inner, self._wrapper, self._sink = inner, wrapper, sink

    def __call__(self, obs_dict: dict) -> list[int]:
        answer = self._inner(obs_dict)
        try:
            obs = self._wrapper.parse(obs_dict)
            if obs.select is not None:
                self._sink(obs)
        except Exception:
            pass
        return answer


class TestGustChoiceOnRealOptions(unittest.TestCase):
    """Drive the engine and read REAL gust options from both pilots."""

    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.alakazam = read_deck_ids(ALAKAZAM_DECK)
        cls.our_deck = read_deck_ids(OUR_DECK)

    def test_the_flag_flips_the_choice_to_the_uncovered_body(self) -> None:
        gusting = HeuristicAgent(index=self.index, effects=self.effects,
                                 seed=5, tempo=True, scaled_damage=True,
                                 gust_targeting=True)
        plain = HeuristicAgent(index=self.index, effects=self.effects,
                               seed=5, tempo=True, scaled_damage=True)
        wrapper = EnvironmentWrapper(self.index)
        choices = 0
        bare = {"gust": 0, "plain": 0}

        def covered(obs, option) -> bool:
            state = obs.current
            pokemon = plain._pokemon_at(state, option.playerIndex,
                                        option.area, option.index)
            return plain._is_covered(pokemon, plain._card_of(pokemon))

        def sink(obs) -> None:
            nonlocal choices
            if obs.select.context != SelectContext.SWITCH:
                return
            state = obs.current
            theirs = [o for o in obs.select.option
                      if o.playerIndex is not None
                      and o.playerIndex != state.yourIndex]
            if len(theirs) < 2:
                return
            # only the decisions where cover was actually AT STAKE
            if len({covered(obs, o) for o in theirs}) < 2:
                return
            choices += 1
            for tag, agent in (("gust", gusting), ("plain", plain)):
                pick = max(theirs,
                           key=lambda o: agent._own_pokemon_score(obs, o, True))
                if not covered(obs, pick):
                    bare[tag] += 1

        driver = _Tap(gusting, wrapper, sink)
        for game in range(40):
            foe = CrustleAgent(index=self.index, effects=self.effects,
                               variant="v3", seed=200 + game)
            play_one_game((driver, foe), list(self.alakazam),
                          list(self.our_deck))
            if choices > 15:
                break
        self.assertGreater(choices, 5,
                           "o motor nunca ofereceu um gust com coberto E "
                           "descoberto na mesa — a assercao seria vazia")
        self.assertEqual(bare["gust"], choices,
                         f"com a flag, {bare['gust']}/{choices} escolhas "
                         f"pegaram o corpo SEM cobertura")
        self.assertLess(bare["plain"], choices,
                        "o piloto generico ja escolhia certo — entao nao "
                        "havia bug a corrigir")


class TestShippedCrustleIsUnchanged(unittest.TestCase):
    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(OUR_DECK)
        cls.alakazam = read_deck_ids(ALAKAZAM_DECK)

    def test_crustle_defaults_to_gust_targeting_off(self) -> None:
        agent = CrustleAgent(index=self.index, effects=self.effects,
                             variant="v3")
        self.assertFalse(agent._gust_targeting)

    def test_flag_is_inert_over_real_crustle_decisions(self) -> None:
        """Lockstep, flag forced ON in the shadow. CrustleAgent overrides
        _own_pokemon_score and already routes enemy promotions through
        its own trap scorer, so this is the layer that has to hold."""
        driver = CrustleAgent(index=self.index, effects=self.effects,
                              variant="v3", seed=41)
        shadow = CrustleAgent(index=self.index, effects=self.effects,
                              variant="v3", seed=41)
        shadow._gust_targeting = True          # the ONLY difference
        divergences = compared = 0

        def lockstep(obs_dict: dict) -> list[int]:
            nonlocal divergences, compared
            answer = driver(obs_dict)
            compared += 1
            if answer != shadow(dict(obs_dict)):
                divergences += 1
            return answer

        foe = HeuristicAgent(index=self.index, effects=self.effects, seed=42)
        for _game in range(5):
            play_one_game((lockstep, foe), list(self.deck),
                          list(self.alakazam))
        self.assertGreater(compared, 100, "quase nada comparado — teste vácuo")
        self.assertEqual(divergences, 0,
                         f"o fix mudou o piloto do ship em "
                         f"{divergences}/{compared} decisões")


if __name__ == "__main__":
    unittest.main()
