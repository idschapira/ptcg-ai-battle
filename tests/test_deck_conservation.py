"""Deck conservation in the GENERIC pilot: correct, and inert for the ship.

Measured root cause of the Alakazam inflation (conversion audit, 01/Ago):
on the cycles where our Great Tusk was not active — so Land Collapse
provably could not fire and everything leaving the opponent's deck was
their own doing — the internal opponent consumed 3.82 cards per cycle
against 3.04 for real ladder opponents. It decks ITSELF out in 64.7% of
games against 28.8% on the ladder.

CrustleAgent has had the fix since v3 (rule i). The generic HeuristicAgent
never had it. This ports the SHAPE of that rule — absolute floor, plus the
deck race only once the deck is genuinely low — and derives WHAT counts as
a thinner from the engine's own card text, because the generic pilot flies
every deck and cannot carry a hand-listed set per archetype.

Lines held here:
  * the cost parser reproduces, from the engine text alone, the seven
    thinners CrustleAgent lists by hand — with the right magnitudes;
  * on OUR deck the derived set is EXACTLY that hand-curated set and the
    thresholds coincide, which is why the rule is inert for the ship
    structurally and not by luck;
  * on REAL engine options a starved pilot drops a thinner below END
    (it passes) while leaving non-thinners alone;
  * CrustleAgent does not move, lockstep, flag forced on.

Run from the repo root:  python -m unittest tests.test_deck_conservation
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Final

import cg.api as api
from cg.api import OptionType, SelectContext

from src.agent_heuristics.crustle_agent import (LOW_DECK, SELF_THINNERS,
                                                V3_RACE_FLOOR, CrustleAgent)
from src.agent_heuristics.heuristic_agent import (DEFAULT_DECK_FLOOR,
                                                  DEFAULT_RACE_FLOOR,
                                                  _END_SCORE, _TRAINER_BAND,
                                                  HeuristicAgent, deck_cost,
                                                  deck_costs)
from src.deckbuilding.legality import read_deck_ids
from src.environment_wrapper.selfplay import play_one_game
from src.environment_wrapper.wrapper import EnvironmentWrapper
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex

OUR_DECK: Final[Path] = Path("deck.csv")
ALAKAZAM_DECK: Final[Path] = Path("data/decks/meta_alakazam.csv")
EXPLORERS_GUIDANCE: Final[int] = 1185
POFFIN: Final[int] = 1086
ULTRA_BALL: Final[int] = 1121


class TestDeckCostParser(unittest.TestCase):
    """Read off the engine's text, not off memory."""

    def test_it_reproduces_the_hand_curated_thinners(self) -> None:
        table = deck_costs()
        missing = sorted(c for c in SELF_THINNERS if c not in table)
        self.assertEqual(missing, [],
                         f"o parser perdeu thinners que o CrustleAgent "
                         f"lista a mao: {missing}")

    def test_the_magnitudes_match_what_the_cards_say(self) -> None:
        table = deck_costs()
        # "Look at the top 6 cards ... Discard the other cards" -> all 6
        self.assertEqual(table[EXPLORERS_GUIDANCE], 6.0)
        # "Search your deck for up to 2 Basic Pokémon" -> 2
        self.assertEqual(table[POFFIN], 2.0)
        # "Search your deck for a Pokémon" -> 1
        self.assertEqual(table[ULTRA_BALL], 1.0)

    def test_a_peek_that_shuffles_back_costs_less_than_one_that_discards(
            self) -> None:
        """Pokégear looks at 7 and puts 6 BACK; Guidance keeps 2 of 6 and
        discards 4. Only what leaves the deck counts."""
        peek = deck_cost("Look at the top 7 cards of your deck. You may "
                         "reveal a Supporter card you find there and put it "
                         "into your hand. Shuffle the other cards back into "
                         "your deck.")
        burn = deck_cost("Look at the top 6 cards of your deck and put 2 of "
                         "them into your hand. Discard the other cards.")
        self.assertEqual(peek, 1.0)
        self.assertEqual(burn, 6.0)
        self.assertLess(peek, burn)

    def test_a_card_that_touches_no_deck_costs_nothing(self) -> None:
        self.assertEqual(deck_cost("Heal 80 damage from 1 of your Pokémon."),
                         0.0)
        self.assertEqual(deck_cost(None), 0.0)

    def test_no_pokemon_is_ever_classified_as_a_thinner(self) -> None:
        """Suppressing a Pokémon would suppress board development, which
        is the failure mode v3 was written to avoid."""
        table = deck_costs()
        cards = {c.cardId: c for c in api.all_card_data()}
        with_hp = [cid for cid in table
                   if getattr(cards.get(cid), "hp", None)]
        self.assertEqual(with_hp, [])


class TestInertiaIsStructural(unittest.TestCase):
    """Why the ship cannot move: on OUR list the derived rule IS its rule."""

    index: CardIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.deck = set(read_deck_ids(OUR_DECK))

    def test_on_our_deck_the_derived_set_equals_the_curated_one(self) -> None:
        derived = {c for c in self.deck if c in deck_costs()}
        curated = {c for c in self.deck if c in SELF_THINNERS}
        self.assertEqual(derived, curated,
                         "o parser generico e a lista do CrustleAgent "
                         "discordam sobre o NOSSO deck — a inercia do ship "
                         "deixa de ser estrutural")
        self.assertEqual(len(derived), 7)

    def test_the_triggers_coincide(self) -> None:
        self.assertEqual(DEFAULT_DECK_FLOOR, LOW_DECK)
        self.assertEqual(DEFAULT_RACE_FLOOR, V3_RACE_FLOOR)


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


class TestSuppressionOnRealOptions(unittest.TestCase):
    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.alakazam = read_deck_ids(ALAKAZAM_DECK)
        cls.our_deck = read_deck_ids(OUR_DECK)

    def test_a_starved_pilot_passes_instead_of_thinning(self) -> None:
        """Scored on REAL PLAY options. The floor is a parameter, so
        'starved' is produced by raising it rather than by waiting for a
        rare board — the rule under test is the same either way."""
        starved = HeuristicAgent(index=self.index, effects=self.effects,
                                 seed=8, tempo=True, scaled_damage=True,
                                 deck_conservation=True, deck_floor=60)
        plain = HeuristicAgent(index=self.index, effects=self.effects,
                               seed=8, tempo=True, scaled_damage=True)
        wrapper = EnvironmentWrapper(self.index)
        thinners = others = 0

        def sink(obs) -> None:
            nonlocal thinners, others
            if obs.select.context != SelectContext.MAIN:
                return
            for option in obs.select.option:
                if option.type != OptionType.PLAY:
                    continue
                card_id = plain._wrapper.resolve_card_id(obs, option)
                hot = starved._main_score(obs, option)
                cold = plain._main_score(obs, option)
                if card_id in deck_costs():
                    thinners += 1
                    self.assertLess(hot, _END_SCORE,
                                    "um thinner tem de cair ABAIXO do END "
                                    "para o piloto passar em vez de sacar")
                    self.assertGreaterEqual(cold, _END_SCORE)
                else:
                    others += 1
                    self.assertEqual(hot, cold,
                                     "a regra tocou uma carta que nao "
                                     "consome deck")

        driver = _Tap(starved, wrapper, sink)
        for game in range(25):
            foe = CrustleAgent(index=self.index, effects=self.effects,
                               variant="v3", seed=300 + game)
            play_one_game((driver, foe), list(self.alakazam),
                          list(self.our_deck))
            if thinners > 25 and others > 25:
                break
        self.assertGreater(thinners, 10, "amostra insuficiente de thinner")
        self.assertGreater(others, 10, "amostra insuficiente de nao-thinner")

    def test_a_healthy_deck_never_suppresses(self) -> None:
        """The conservatism that v1/v2 got wrong: a full deck must play
        its setup, or the pilot strangles itself out of the board."""
        agent = HeuristicAgent(index=self.index, effects=self.effects,
                               seed=9, deck_conservation=True)
        wrapper = EnvironmentWrapper(self.index)
        checked = [0]

        def sink(obs) -> None:
            state = obs.current
            mine = state.players[state.yourIndex].deckCount
            if mine is None or mine <= DEFAULT_RACE_FLOOR:
                return
            checked[0] += 1
            self.assertFalse(agent._deck_starved(obs),
                             f"deck com {mine} cartas nao pode estar "
                             f"'faminto'")

        driver = _Tap(agent, wrapper, sink)
        for game in range(20):
            foe = CrustleAgent(index=self.index, effects=self.effects,
                               variant="v3", seed=99 + game)
            play_one_game((driver, foe), list(self.alakazam),
                          list(self.our_deck))
            if checked[0] > 60:
                break
        self.assertGreater(checked[0], 20, "quase nada conferido")


class TestShippedCrustleIsUnchanged(unittest.TestCase):
    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(OUR_DECK)
        cls.alakazam = read_deck_ids(ALAKAZAM_DECK)

    def test_crustle_defaults_to_conservation_off(self) -> None:
        agent = CrustleAgent(index=self.index, effects=self.effects,
                             variant="v3")
        self.assertFalse(agent._deck_conservation)

    def test_flag_is_inert_over_real_crustle_decisions(self) -> None:
        driver = CrustleAgent(index=self.index, effects=self.effects,
                              variant="v3", seed=51)
        shadow = CrustleAgent(index=self.index, effects=self.effects,
                              variant="v3", seed=51)
        shadow._deck_conservation = True        # the ONLY difference
        divergences = compared = 0

        def lockstep(obs_dict: dict) -> list[int]:
            nonlocal divergences, compared
            answer = driver(obs_dict)
            compared += 1
            if answer != shadow(dict(obs_dict)):
                divergences += 1
            return answer

        foe = HeuristicAgent(index=self.index, effects=self.effects, seed=52)
        for _game in range(5):
            play_one_game((lockstep, foe), list(self.deck),
                          list(self.alakazam))
        self.assertGreater(compared, 100, "quase nada comparado — teste vácuo")
        self.assertEqual(divergences, 0,
                         f"o fix mudou o piloto do ship em "
                         f"{divergences}/{compared} decisões")


if __name__ == "__main__":
    unittest.main()
