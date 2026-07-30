"""Scaled-damage valuation: correct, and inert for the ship.

`_effect_adjustment` values an attack as damage_base plus a fixed bonus
per effect row, which is blind to attacks whose damage IS the scale.
Measured before this change: Alakazam's Powerful Hand — "Place 2 damage
counters on your opponent's Active Pokémon for each card in your hand",
base 0 — scored 13.0 while delivering ~266, and was recognised as lethal
in 0 of 767 real decisions.

This module holds three lines:
  * the parser turns real engine text into the right clause, including
    the units it must REFUSE to resolve (an unobservable unit has to fall
    into the declared estimate, not silently invent a number);
  * on REAL engine options the valuation lands and the attack starts
    being seen as lethal;
  * CrustleAgent — the ship — does not move. _effect_adjustment is
    shared and CrustleAgent calls super()._main_score, so this is exactly
    the edit that leaks into production. Checked lockstep on one
    observation stream, because the engine has no seed and "run the same
    game twice" is not available.

Run from the repo root:  python -m unittest tests.test_scaled_damage
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Final

import cg.api as api
from cg.api import OptionType

from src.agent_heuristics.crustle_agent import CrustleAgent
from src.agent_heuristics.heuristic_agent import (HeuristicAgent,
                                                  parse_scaled_clause)
from src.deckbuilding.legality import read_deck_ids
from src.environment_wrapper.selfplay import play_one_game
from src.environment_wrapper.wrapper import EnvironmentWrapper
from src.ingestion.build_effect_model import EffectIndex
from src.ingestion.card_index import CardIndex

POWERFUL_HAND: Final[int] = 1072
ALAKAZAM_DECK: Final[Path] = Path("data/decks/meta_alakazam.csv")
OUR_DECK: Final[Path] = Path("deck.csv")


class TestParser(unittest.TestCase):
    def test_powerful_hand_is_20_damage_per_card_in_hand(self) -> None:
        text = api.all_attack()[POWERFUL_HAND - 1].text
        clause = parse_scaled_clause(text)
        self.assertIsNotNone(clause)
        assert clause is not None
        # 2 damage counters = 20 damage, once per card in hand
        self.assertEqual(clause.per_unit, 20.0)
        self.assertEqual(clause.unit, "my_hand")

    def test_coin_units_use_expectation_not_the_generic_fallback(self) -> None:
        clause = parse_scaled_clause(
            "Flip 4 coins. This attack does 30 damage for each heads.")
        assert clause is not None
        self.assertEqual(clause.unit, "coin")
        self.assertEqual(clause.fixed_units, 2.0)

    def test_unobservable_unit_is_refused_not_invented(self) -> None:
        clause = parse_scaled_clause(
            "This attack does 20 damage for each Energy card in your "
            "discard pile.")
        assert clause is not None
        self.assertEqual(clause.unit, "unresolved")

    def test_plain_attacks_have_no_clause(self) -> None:
        for attack_id in (479, 62):     # Superb Scissors, Land Collapse
            text = api.all_attack()[attack_id - 1].text
            self.assertIsNone(parse_scaled_clause(text),
                              f"aid={attack_id} não deveria ter escala")


class TestValuationOnRealOptions(unittest.TestCase):
    """Drive the engine and read the score of a REAL Powerful Hand."""

    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(ALAKAZAM_DECK)

    def _values(self, scaled: bool) -> list[tuple[float, int, int]]:
        """(attack value, defender hp, our hand size) whenever PH is offered."""
        wrapper = EnvironmentWrapper(self.index)
        seen: list[tuple[float, int, int]] = []

        def make(seed: int):
            agent = HeuristicAgent(index=self.index, effects=self.effects,
                                   seed=seed, scaled_damage=scaled)

            def call(obs_dict: dict) -> list[int]:
                answer = agent(obs_dict)
                try:
                    obs = wrapper.parse(obs_dict)
                    select = obs.select
                    if select and any(o.type == OptionType.ATTACK
                                      and o.attackId == POWERFUL_HAND
                                      for o in select.option):
                        target = agent._opp_active(obs)
                        state = obs.current
                        if target is not None and state is not None:
                            hand = state.players[state.yourIndex].handCount
                            seen.append((
                                agent._attack_value(POWERFUL_HAND, obs),
                                target.hp, int(hand or 0)))
                except Exception:
                    pass
                return answer
            return call

        # The engine is not seedable, so how many Powerful Hands a run
        # offers is stochastic. Keep playing until the sample is big
        # enough to assert on; a thin sample would make these tests pass
        # for the wrong reason.
        for game_index in range(60):
            play_one_game((make(game_index), make(900 + game_index)),
                          list(self.deck), list(self.deck))
            if len(seen) > 60:
                break
        self.assertGreater(len(seen), 20,
                           "amostra insuficiente de Powerful Hand — a "
                           "asserção seguinte seria fraca demais")
        return seen

    def test_without_the_fix_the_value_ignores_the_hand(self) -> None:
        """The bug, stated precisely: a flat 13.0 whatever the hand holds.

        (Not "never lethal" — a Pokémon already down to 10 HP dies to
        anything, so that phrasing would fail for a reason that has
        nothing to do with the valuation.)
        """
        seen = self._values(scaled=False)
        self.assertTrue(seen, "o motor nunca ofereceu Powerful Hand")
        values = {round(v, 6) for v, _hp, _hand in seen}
        hands = {hand for _v, _hp, hand in seen}
        self.assertEqual(values, {13.0},
                         f"esperado 13.0 constante, veio {sorted(values)}")
        self.assertGreater(len(hands), 1,
                           "a mão nunca variou — o teste não prova cegueira")

    def test_with_the_fix_the_value_tracks_the_hand_and_kills(self) -> None:
        seen = self._values(scaled=True)
        self.assertTrue(seen, "o motor nunca ofereceu Powerful Hand")
        mean = sum(v for v, _hp, _hand in seen) / len(seen)
        self.assertGreater(mean, 100.0,
                           f"valor médio {mean:.1f} continua baixo demais "
                           f"para um ataque que entrega ~266")
        # the value must MOVE with the hand: that is the whole clause
        by_hand = {}
        for value, _hp, hand in seen:
            by_hand.setdefault(hand, value)
        self.assertGreater(len(set(by_hand.values())), 1,
                           "o valor não acompanha o tamanho da mão")
        lethal = sum(1 for v, hp, _h in seen if v >= hp)
        self.assertGreater(lethal / len(seen), 0.5,
                           "o ataque que de fato mata tem de ser "
                           "reconhecido como letal na maioria das vezes")


class TestShippedCrustleIsUnchanged(unittest.TestCase):
    index: CardIndex
    effects: EffectIndex

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.effects = EffectIndex()
        cls.deck = read_deck_ids(OUR_DECK)

    def test_crustle_defaults_to_scaled_damage_off(self) -> None:
        agent = CrustleAgent(index=self.index, effects=self.effects,
                             variant="v3")
        self.assertFalse(agent._scaled_damage)

    def test_no_attack_in_our_deck_scales(self) -> None:
        """Structural guarantee: the rule has nothing to bite on."""
        attacks = api.all_attack()
        scaling = []
        for card_id in set(self.deck):
            card = self.index.get_card(card_id)
            for attack_id in (card.attack_ids if card else ()):
                clause = parse_scaled_clause(attacks[attack_id - 1].text)
                if clause is not None:
                    scaling.append((card.card_name, attack_id))
        self.assertEqual(scaling, [],
                         "um ataque nosso passou a ter cláusula de escala — "
                         "a inércia do fix para o ship deixa de ser "
                         "estrutural")

    def test_flag_is_inert_over_real_crustle_decisions(self) -> None:
        """Lockstep: same stream, flag on vs off, identical answers."""
        divergences: list[str] = []
        compared = 0
        driver = CrustleAgent(index=self.index, effects=self.effects,
                              variant="v3", seed=7)
        shadow = CrustleAgent(index=self.index, effects=self.effects,
                              variant="v3", seed=7)
        shadow._scaled_damage = True      # the ONLY difference

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
        self.assertGreater(compared, 20, "quase nada comparado — teste vácuo")
        self.assertEqual(divergences, [],
                         f"o fix mudou o piloto do ship em "
                         f"{len(divergences)}/{compared} decisões")

    def test_flag_is_inert_against_a_scaling_opponent(self) -> None:
        """Our side must not move even when the OPPONENT scales.

        The Crustle lockstep above only sees our own attacks; this one
        puts the ship in front of Alakazam, whose whole deck is the
        scaling clause, so the shared scorer is exercised on a board
        where the new code path is live for the other player.
        """
        divergences = 0
        compared = 0
        alakazam = read_deck_ids(ALAKAZAM_DECK)
        driver = CrustleAgent(index=self.index, effects=self.effects,
                              variant="v3", seed=11)
        shadow = CrustleAgent(index=self.index, effects=self.effects,
                              variant="v3", seed=11)
        shadow._scaled_damage = True

        def lockstep(obs_dict: dict) -> list[int]:
            nonlocal divergences, compared
            answer = driver(obs_dict)
            compared += 1
            if answer != shadow(dict(obs_dict)):
                divergences += 1
            return answer

        foe = HeuristicAgent(index=self.index, effects=self.effects, seed=12)
        play_one_game((lockstep, foe), list(self.deck), list(alakazam))
        self.assertGreater(compared, 10, "quase nada comparado — teste vácuo")
        self.assertEqual(divergences, 0)


if __name__ == "__main__":
    unittest.main()
