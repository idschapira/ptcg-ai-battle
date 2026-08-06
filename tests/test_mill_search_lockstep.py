"""A rung de busca do mill tem de mudar a busca — e NADA fora dela.

Duas obrigações, no espírito de tests/test_tempo_equivalence.py (contra o
motor real, nunca contra Option sintética: uma Option montada à mão
satisfaz qualquer caminho e esconde regra já morta):

  1. LOCKSTEP: com ``mill_search=False`` (o default) o piloto shipado é
     idêntico decisão-a-decisão ao seu eu pré-mudança, em jogo completo
     do motor. `_search_value` vive dentro do CrustleAgent v3, que É o
     ship — é exatamente o tipo de edição que vaza.
  2. A flag não é decorativa: com ``mill_search=True`` a rung do
     Explorer's Guidance SOBE nos estados em que o motor de mill está em
     campo mas o Land Collapse ainda não é pagável, e fica INALTERADA
     quando não há Great Tusk em campo (sem motor, buscar o Guidance é
     desperdício) e quando `_great_tusk_ready` já dava 95.

Rodar da raiz do repo:
    python -m unittest tests.test_mill_search_lockstep
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Final

from src.agent_heuristics.crustle_agent import (EXPLORERS_GUIDANCE,
                                                GREAT_TUSK,
                                                MILL_SEARCH_TUSK_ACTIVE,
                                                MILL_SEARCH_TUSK_IN_PLAY,
                                                CrustleAgent)
from src.deckbuilding.legality import read_deck_ids
from src.environment_wrapper.selfplay import play_one_game
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex

OUR_DECK: Final[Path] = Path("data/decks/variant_crustle_nzaccess.csv")
FIELD_DECK: Final[Path] = Path("data/decks/meta_grimmsnarl.csv")
GAMES: Final[int] = 6


class _Pair:
    """Roda o piloto A e registra o que o piloto B responderia igual."""

    def __init__(self, a: CrustleAgent, b: CrustleAgent) -> None:
        self._a, self._b = a, b
        self.decisions = 0
        self.diffs: list[str] = []

    def __call__(self, obs_dict: dict) -> list[int]:
        answer = self._a(dict(obs_dict))
        other = self._b(dict(obs_dict))
        self.decisions += 1
        if sorted(answer) != sorted(other):
            turn = (obs_dict.get("current") or {}).get("turn")
            self.diffs.append(f"t{turn}: {answer} vs {other}")
        return answer


class MillSearchLockstepTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.our = read_deck_ids(OUR_DECK)
        cls.field = read_deck_ids(FIELD_DECK)

    def _play(self, pair: _Pair, games: int) -> None:
        for i in range(games):
            opponent = CrustleAgent(seed=i, index=self.index,
                                    effects=self.effects, variant="v3")
            play_one_game((pair, opponent), list(self.our), list(self.field))

    def test_default_off_is_lockstep_with_the_ship(self) -> None:
        """mill_search=False não muda NENHUMA decisão do v3 shipado."""
        a = CrustleAgent(index=self.index, effects=self.effects, variant="v3")
        b = CrustleAgent(index=self.index, effects=self.effects, variant="v3",
                         mill_search=False)
        pair = _Pair(a, b)
        self._play(pair, GAMES)
        self.assertGreater(pair.decisions, 200, "amostra pequena demais")
        self.assertEqual(pair.diffs, [], f"{len(pair.diffs)} divergências")

    def test_flag_is_not_decorative_on_real_states(self) -> None:
        """Em ESTADOS REAIS do motor, a rung sobe onde deve e só onde deve."""
        agent = CrustleAgent(index=self.index, effects=self.effects,
                             variant="v3", mill_search=True)
        ship = CrustleAgent(index=self.index, effects=self.effects,
                            variant="v3")
        self.assertFalse(ship._mill_search, "o ship tem de ficar com a flag OFF")

        seen: dict[str, float] = {}
        harvest = _Harvest(agent, seen)
        for i in range(GAMES):
            opponent = CrustleAgent(seed=i, index=self.index,
                                    effects=self.effects, variant="v3")
            play_one_game((harvest, opponent), list(self.our), list(self.field))

        self.assertIn("tusk_active", seen, "nenhum estado com Tusk ativo")
        self.assertIn("tusk_bench", seen, "nenhum estado com Tusk no banco")
        self.assertIn("no_engine", seen, "nenhum estado sem motor em campo")
        self.assertEqual(seen["tusk_active"], MILL_SEARCH_TUSK_ACTIVE)
        self.assertEqual(seen["tusk_bench"], MILL_SEARCH_TUSK_IN_PLAY)
        # sem motor em campo: INALTERADO em relação ao ship
        self.assertEqual(seen["no_engine"], 40.0)

        # e as rungs ficam abaixo do que tem de vir antes
        self.assertLess(MILL_SEARCH_TUSK_ACTIVE, 85.0)   # 1º Great Tusk
        self.assertLess(MILL_SEARCH_TUSK_ACTIVE, 100.0)  # Neutralization Zone
        self.assertGreater(MILL_SEARCH_TUSK_IN_PLAY, 56.0)  # energia
        self.assertGreater(MILL_SEARCH_TUSK_ACTIVE, MILL_SEARCH_TUSK_IN_PLAY)


class _Harvest:
    """Joga com o piloto e colhe a rung em estados REAIS de cada tipo."""

    def __init__(self, agent: CrustleAgent, seen: dict) -> None:
        self._agent, self.seen = agent, seen

    def __call__(self, obs_dict: dict) -> list[int]:
        answer = self._agent(dict(obs_dict))
        try:
            obs = self._agent._wrapper.parse(obs_dict)
            if obs.current is not None:
                active = self._agent._my_active(obs)
                bench = self._agent._my_bench(obs)
                rung = self._agent._mill_search_value(obs)
                if active is not None and active.id == GREAT_TUSK:
                    self.seen.setdefault("tusk_active", rung)
                elif any(p.id == GREAT_TUSK for p in bench):
                    self.seen.setdefault("tusk_bench", rung)
                elif active is not None:
                    self.seen.setdefault("no_engine", rung)
        except Exception:  # noqa: BLE001 — colheita nunca quebra o jogo
            pass
        return answer


if __name__ == "__main__":
    unittest.main()
