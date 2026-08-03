"""CrustleAgent v4 (defensive): the rules fire where they should, and v3
is untouched.

v4 = v3 plus the two threat-aware rules — nothing in v1-v3 ever asks
"what is about to happen to me", it reads our own board or the
opponent's CARD TYPE. v4 adds:
  (A) Xerosic when a HAND-SCALED attack is lethal on our bare active
      (capping their hand at 3 turns ~240 damage into ~60);
  (B) covering that active instead of fuelling the mill.

Both are gated on the "for each card in your hand" clause, which is what
makes the measured scope so narrow: the rules cannot fire against an
archetype that has no such attack, and that is asserted here rather than
left as a surprise in the A/B (measured: v4 differs from v3 on 0.38% of
decisions vs Alakazam and 0.00% vs Grimmsnarl and Mega Lucario).

Run from the repo root:  python -m unittest tests.test_crustle_v4
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Final

from src.agent_heuristics.crustle_agent import CrustleAgent
from src.deckbuilding.legality import read_deck_ids
from src.environment_wrapper.selfplay import play_one_game
from src.environment_wrapper.wrapper import EnvironmentWrapper
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex

OUR_DECK: Final[Path] = Path("deck.csv")
ALAKAZAM_DECK: Final[Path] = Path("data/decks/meta_alakazam.csv")
GRIMMSNARL_DECK: Final[Path] = Path("data/decks/meta_grimmsnarl.csv")


class TestVariantWiring(unittest.TestCase):
    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()

    def _agent(self, variant: str) -> CrustleAgent:
        return CrustleAgent(index=self.index, effects=self.effects,
                            variant=variant)

    def test_v4_keeps_every_earlier_rule_and_v3_does_not_gain_v4(self) -> None:
        v3, v4 = self._agent("v3"), self._agent("v4")
        self.assertTrue(v3._v2 and v3._v3)
        self.assertFalse(v3._v4, "a v3 nao pode herdar regra da v4")
        self.assertTrue(v4._v2 and v4._v3 and v4._v4)


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


class TestThreatSignalOnRealBoards(unittest.TestCase):
    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(OUR_DECK)
        cls.alakazam = read_deck_ids(ALAKAZAM_DECK)
        cls.grimmsnarl = read_deck_ids(GRIMMSNARL_DECK)

    def _threats(self, opponent: list[int], games: int) -> list[float]:
        agent = CrustleAgent(index=self.index, effects=self.effects,
                             variant="v4", seed=71)
        wrapper = EnvironmentWrapper(self.index)
        seen: list[float] = []

        def sink(obs) -> None:
            seen.append(agent._hand_scaled_threat(obs))

        driver = _Tap(agent, wrapper, sink)
        foe = CrustleAgent(index=self.index, effects=self.effects,
                           variant="v3", seed=72)
        for game in range(games):
            play_one_game((driver, foe), list(self.deck), list(opponent))
        return seen

    def test_the_threat_is_seen_against_a_hand_scaled_attacker(self) -> None:
        seen = self._threats(self.alakazam, 6)
        self.assertGreater(len(seen), 100, "quase nada observado")
        live = [t for t in seen if t > 0.0]
        self.assertTrue(live,
                        "o sinal nunca disparou contra o Alakazam — a regra "
                        "da v4 nao alcanca o arquetipo que ela existe para "
                        "responder")
        # 2 damage counters per card in hand: a real hand puts the threat
        # well past a chip hit
        self.assertGreater(max(live), 100.0)

    def test_it_is_silent_against_an_archetype_without_that_clause(
            self) -> None:
        """Why the A/B measured 0.00% on two of the three biggest cells:
        the trigger is a property of the OPPONENT'S CARD, not of the
        regime. Stated here so the scope is a contract, not a surprise."""
        seen = self._threats(self.grimmsnarl, 6)
        self.assertGreater(len(seen), 100, "quase nada observado")
        self.assertEqual([t for t in seen if t > 0.0], [],
                         "o Grimmsnarl nao tem ataque escalado por mao; "
                         "se isto disparou, o sinal esta lendo outra coisa")


class TestV4IsAStrictSupersetOfV3(unittest.TestCase):
    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(OUR_DECK)
        cls.grimmsnarl = read_deck_ids(GRIMMSNARL_DECK)

    def test_without_the_trigger_v4_answers_exactly_like_v3(self) -> None:
        driver = CrustleAgent(index=self.index, effects=self.effects,
                              variant="v3", seed=81)
        shadow = CrustleAgent(index=self.index, effects=self.effects,
                              variant="v4", seed=81)
        divergences = compared = 0

        def lockstep(obs_dict: dict) -> list[int]:
            nonlocal divergences, compared
            answer = driver(obs_dict)
            compared += 1
            if answer != shadow(dict(obs_dict)):
                divergences += 1
            return answer

        foe = CrustleAgent(index=self.index, effects=self.effects,
                           variant="v3", seed=82)
        for _game in range(5):
            play_one_game((lockstep, foe), list(self.deck),
                          list(self.grimmsnarl))
        self.assertGreater(compared, 100, "quase nada comparado — teste vácuo")
        self.assertEqual(divergences, 0,
                         f"sem gatilho de ameaca a v4 tem de ser a v3, e "
                         f"divergiu em {divergences}/{compared}")


if __name__ == "__main__":
    unittest.main()
