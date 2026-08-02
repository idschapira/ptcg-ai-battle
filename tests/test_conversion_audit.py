"""Contracts under the conversion audit's headline numbers.

The audit's residual — "the internal opponent burns its own deck 26%
faster" — is read off the cycles in which our Great Tusk was NOT the
active, on the reasoning that Land Collapse cannot fire then, so whatever
left the opponent's deck was their own doing. That reasoning rests on two
facts about the engine, and facts get asserted, not assumed:

  * Land Collapse (62) belongs to Great Tusk and to nothing else, so no
    other body of ours can mill with it;
  * an attack option is only ever offered for the ACTIVE Pokémon, which
    is what makes "Tusk not active" equal to "no mill this turn".

The second one is checked by DRIVING the engine, not by quoting a rule:
over real games, every ATTACK option the engine offers us must belong to
the card sitting in our Active Spot.

Run from the repo root:  python -m unittest tests.test_conversion_audit
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Final

import cg.api as api
from cg.api import OptionType, SelectContext

from src.agent_heuristics.crustle_agent import CrustleAgent
from src.analysis.conversion_audit import GREAT_TUSK, LAND_COLLAPSE
from src.deckbuilding.legality import read_deck_ids
from src.environment_wrapper.selfplay import play_one_game
from src.environment_wrapper.wrapper import EnvironmentWrapper
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex

OUR_DECK: Final[Path] = Path("deck.csv")
ALAKAZAM_DECK: Final[Path] = Path("data/decks/meta_alakazam.csv")


class TestMillIsGreatTuskOnly(unittest.TestCase):
    index: CardIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()

    def test_land_collapse_belongs_to_great_tusk(self) -> None:
        owners = [card.cardId for card in api.all_card_data()
                  if LAND_COLLAPSE in (card.attacks or ())]
        self.assertEqual(owners, [GREAT_TUSK],
                         f"Land Collapse deixou de ser exclusivo do Great "
                         f"Tusk: donos={owners}")

    def test_it_is_the_mill_and_not_the_damage_attack(self) -> None:
        text = api.all_attack()[LAND_COLLAPSE - 1].text or ""
        self.assertIn("deck", text.lower())
        self.assertIn("discard", text.lower())


class TestAttackOptionsAreActiveOnly(unittest.TestCase):
    """Driven against the engine: only the ACTIVE is offered an attack."""

    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(OUR_DECK)
        cls.alakazam = read_deck_ids(ALAKAZAM_DECK)

    def test_every_attack_offered_belongs_to_our_active(self) -> None:
        wrapper = EnvironmentWrapper(self.index)
        checked = 0
        tusk_offers = 0
        agent = CrustleAgent(index=self.index, effects=self.effects,
                             variant="v3", seed=61)

        def tap(obs_dict: dict) -> list[int]:
            nonlocal checked, tusk_offers
            answer = agent(obs_dict)
            try:
                obs = wrapper.parse(obs_dict)
                select = obs.select
                if select is None or select.context != SelectContext.MAIN:
                    return answer
                state = obs.current
                active = state.players[state.yourIndex].active
                active = active[0] if active else None
                if active is None:
                    return answer
                allowed = set(self.index.get_card(active.id).attack_ids
                              if self.index.get_card(active.id) else ())
                for option in select.option:
                    if option.type != OptionType.ATTACK:
                        continue
                    checked += 1
                    self.assertIn(
                        option.attackId, allowed,
                        f"o motor ofereceu o ataque {option.attackId} com "
                        f"{active.id} no ativo — 'Tusk fora do ativo => sem "
                        f"mill' deixa de valer")
                    if option.attackId == LAND_COLLAPSE:
                        tusk_offers += 1
                        self.assertEqual(active.id, GREAT_TUSK)
            except AssertionError:
                raise
            except Exception:
                pass
            return answer

        # Sample until the check is real, with a bound: how many attack
        # options a run produces is stochastic (the engine has no seed),
        # and a fixed game count makes the vacuity guard decide the test
        # instead of the rule it guards.
        foe = CrustleAgent(index=self.index, effects=self.effects,
                           variant="v3", seed=62)
        for _game in range(40):
            play_one_game((tap, foe), list(self.deck), list(self.alakazam))
            if checked > 60 and tusk_offers > 0:
                break
        self.assertGreater(checked, 50, "quase nada conferido — teste vácuo")
        self.assertGreater(tusk_offers, 0,
                           "o Land Collapse nunca foi oferecido — o teste "
                           "não chegou a exercitar o caso que importa")


if __name__ == "__main__":
    unittest.main()
