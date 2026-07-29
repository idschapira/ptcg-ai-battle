"""Contract: the estimator abstains by default and never invents a deck.

The estimator is the only thing standing between the search and a
fabricated opponent, so what matters is not how often it is right but
how it FAILS. Every test here is about the abstain path: an unknown
archetype, a list we do not hold, an observation shaped wrong, a game
boundary that must not leak reads into the next episode.

The one accuracy claim is deliberately weak and end-to-end: fed a real
engine game against a known deck, the label must converge. The
quantified version of that claim lives in
src/analysis/estimator_accuracy.py, measured over the real replay
corpus, because a synthetic fixture cannot tell you what the ladder
actually plays.

Run from the repo root:
    python -m unittest tests.test_opponent_estimator
"""

from __future__ import annotations

import unittest

from src.deckbuilding.archetype_rules import (ARCHETYPE_DECKS, UNKNOWN,
                                              label_archetype)
from src.deckbuilding.legality import read_deck_ids
from src.ingestion.build_card_model import REPO_ROOT
from src.ingestion.card_index import CardIndex
from src.rl_models.opponent_estimator import OpponentDeckEstimator

ALAKAZAM = REPO_ROOT / "data" / "decks" / "meta_alakazam.csv"
OUR_DECK = REPO_ROOT / "deck.csv"


def _obs(cards: list[dict], turn: int = 3, our_seat: int = 0) -> dict:
    """An observation whose opponent bench holds ``cards``."""
    players = [{"active": [], "bench": [], "hand": [], "discard": [],
                "prize": [], "deckCount": 40, "handCount": 5}
               for _ in range(2)]
    players[1 - our_seat]["bench"] = cards
    return {"select": {"option": [{}, {}], "maxCount": 1},
            "current": {"turn": turn, "yourIndex": our_seat,
                        "players": players}}


def _card(card_id: int, serial: int, player: int) -> dict:
    return {"id": card_id, "serial": serial, "playerIndex": player}


class TestOpponentEstimator(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.index = CardIndex()
        cls.alakazam = read_deck_ids(ALAKAZAM)

    # ------------------------------------------------------------------ #
    # Every presumed decklist must exist and be legal-sized
    # ------------------------------------------------------------------ #

    def test_every_mapped_archetype_resolves_to_sixty_cards(self) -> None:
        estimator = OpponentDeckEstimator(index=self.index)
        for label in ARCHETYPE_DECKS:
            with self.subTest(label=label):
                deck = estimator.presumed_deck(label)
                self.assertEqual(len(deck), 60,
                                 f"{label} -> {len(deck)} ids")
                for card_id in deck:
                    self.assertIsNotNone(self.index.get_card(card_id),
                                         f"{label} has unknown id {card_id}")

    def test_unmapped_and_unknown_labels_yield_no_deck(self) -> None:
        estimator = OpponentDeckEstimator(index=self.index)
        self.assertEqual(estimator.presumed_deck(UNKNOWN), ())
        self.assertEqual(estimator.presumed_deck("Dragapult ex"), ())
        self.assertEqual(estimator.presumed_deck("not an archetype"), ())

    # ------------------------------------------------------------------ #
    # Abstention
    # ------------------------------------------------------------------ #

    def test_starts_with_no_hypothesis(self) -> None:
        estimator = OpponentDeckEstimator(index=self.index)
        self.assertFalse(estimator.last_estimate.usable)
        self.assertEqual(estimator.last_estimate.archetype, UNKNOWN)

    def test_a_handful_of_cards_is_not_enough(self) -> None:
        """Right label, too little evidence -> still abstains."""
        estimator = OpponentDeckEstimator(index=self.index, min_observed=8)
        core = [c for c in self.alakazam
                if (self.index.get_card(c) or None) is not None][:3]
        estimate = estimator.observe(_obs(
            [_card(cid, 60 + i, 1) for i, cid in enumerate(core)]))
        self.assertFalse(estimate.usable, estimate)
        self.assertLess(estimate.n_observed, 8)

    def test_cards_outside_the_presumed_list_destroy_confidence(self) -> None:
        """Containment is what notices a right label on a WRONG list.

        The label rule only needs the Alakazam line, so an opponent can
        trip it while playing 60 cards we do not hold. Feeding the line
        plus a pile of cards absent from every presumed list must drive
        containment down and the estimator back to silence.
        """
        estimator = OpponentDeckEstimator(index=self.index, min_observed=4,
                                          min_containment=0.75)
        line = [cid for cid in self.alakazam
                if (self.index.get_card(cid) is not None
                    and self.index.get_card(cid).card_name
                    in {"Abra", "Kadabra", "Alakazam"})][:3]
        self.assertTrue(line, "fixture deck has no Alakazam line")

        # cards in NO presumed decklist, so they cannot be accounted for
        mapped: set[int] = set()
        for label in ARCHETYPE_DECKS:
            mapped.update(estimator.presumed_deck(label))
        foreign = [cid for cid in sorted(self.index.cards)
                   if cid not in mapped][:25]
        self.assertGreaterEqual(len(foreign), 20, "no foreign cards found")

        cards = [_card(cid, 60 + i, 1)
                 for i, cid in enumerate(line + foreign)]
        estimate = estimator.observe(_obs(cards))
        self.assertLess(estimate.containment, 0.75, estimate)
        self.assertFalse(estimate.usable,
                         f"searched under a list it cannot account for: "
                         f"{estimate}")

    def test_malformed_observations_never_raise(self) -> None:
        estimator = OpponentDeckEstimator(index=self.index)
        for obs in (None, {}, [], "x", 7, {"current": "nope"},
                    {"select": {}, "current": {"players": []}},
                    {"select": {}, "current": {"yourIndex": 9,
                                               "players": [{}, {}]}}):
            with self.subTest(obs=obs):
                estimate = estimator.observe(obs)
                self.assertFalse(estimate.usable)

    # ------------------------------------------------------------------ #
    # Episode boundaries
    # ------------------------------------------------------------------ #

    def test_reset_clears_the_read(self) -> None:
        estimator = OpponentDeckEstimator(index=self.index, min_observed=1)
        estimator.observe(_obs([_card(self.alakazam[0], 60, 1)]))
        self.assertGreater(sum(estimator.observed_counts().values()), 0)
        estimator.reset()
        self.assertEqual(estimator.observed_counts(), {})

    def test_a_turn_going_backwards_starts_a_new_episode(self) -> None:
        """A reused instance must not carry last game's reads forward."""
        estimator = OpponentDeckEstimator(index=self.index, min_observed=1)
        estimator.observe(_obs([_card(self.alakazam[0], 60, 1)], turn=12))
        estimator.observe(_obs([], turn=1))
        self.assertEqual(estimator.observed_counts(), {},
                         "reads leaked across an episode boundary")

    def test_only_the_opponents_cards_are_counted(self) -> None:
        """Our own revealed cards must never feed the opponent model."""
        estimator = OpponentDeckEstimator(index=self.index, min_observed=1)
        ours = [_card(cid, i, 0) for i, cid in enumerate(self.alakazam[:10])]
        estimator.observe(_obs(ours, our_seat=0))
        self.assertEqual(estimator.observed_counts(), {},
                         "counted our own cards as the opponent's")

    def test_the_same_card_seen_twice_is_counted_once(self) -> None:
        """Serial identity is what keeps copy counts honest."""
        estimator = OpponentDeckEstimator(index=self.index, min_observed=1)
        card = _card(self.alakazam[0], 60, 1)
        estimator.observe(_obs([card]))
        estimator.observe(_obs([card]))
        estimator.observe(_obs([card]))
        self.assertEqual(sum(estimator.observed_counts().values()), 1)

    # ------------------------------------------------------------------ #
    # Convergence, end to end on a real engine game
    # ------------------------------------------------------------------ #

    def test_converges_to_the_right_label_over_a_real_game(self) -> None:
        from cg import game as cg_game

        from src.agent_heuristics.random_agent import RandomAgent

        our_deck = read_deck_ids(OUR_DECK)
        estimator = OpponentDeckEstimator(index=self.index)
        expected = label_archetype(
            [c.card_name for c in
             (self.index.get_card(i) for i in self.alakazam) if c])
        self.assertNotEqual(expected, UNKNOWN, "fixture deck is unlabelable")

        obs_dict, start = cg_game.battle_start(our_deck, list(self.alakazam))
        confident = False
        try:
            self.assertIsNotNone(obs_dict, getattr(start, "errorType", None))
            agents = (RandomAgent(seed=1), RandomAgent(seed=2))
            for _ in range(400):
                current = obs_dict["current"]
                if current["result"] != -1:
                    break
                if current["yourIndex"] == 0:
                    estimate = estimator.observe(obs_dict)
                    if estimate.usable:
                        confident = True
                        self.assertEqual(estimate.archetype, expected,
                                         f"wrong confident label: {estimate}")
                obs_dict = cg_game.battle_select(
                    agents[current["yourIndex"]](obs_dict))
        finally:
            cg_game.battle_finish()
        self.assertTrue(confident,
                        "never became confident against a known deck")


if __name__ == "__main__":
    unittest.main()
